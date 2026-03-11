use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use regex::Regex;
use serde_json::Value;

use crate::agent::{Agent, build_runtime};
use crate::config::Config;
use crate::environment::{build_environment, maybe_run_startup};
use crate::model::{ApiModel, ApiStyle, Model};
use crate::types::{BenchmarkInstance, Trajectory};

const DATASET_ALIASES: &[(&str, &str)] = &[
    ("lite", "princeton-nlp/SWE-Bench_Lite"),
    ("verified", "princeton-nlp/SWE-Bench_Verified"),
    ("full", "princeton-nlp/SWE-Bench"),
    ("multimodal", "princeton-nlp/SWE-Bench_Multimodal"),
    ("multilingual", "swe-bench/SWE-Bench_Multilingual"),
    ("smith", "SWE-bench/SWE-smith"),
    ("_test", "klieret/swe-bench-dummy-test-dataset"),
    ("rebench", "nebius/SWE-rebench"),
];

pub fn load_instances(dataset_spec: &str, split: &str) -> Result<Vec<BenchmarkInstance>> {
    if is_hf_dataset_spec(dataset_spec) {
        return load_hf_dataset(dataset_spec, split);
    }
    let raw: String = if dataset_spec.starts_with("http://") || dataset_spec.starts_with("https://")
    {
        reqwest::blocking::get(dataset_spec)
            .with_context(|| format!("failed to fetch dataset {dataset_spec}"))?
            .error_for_status()
            .with_context(|| format!("dataset request failed {dataset_spec}"))?
            .text()
            .context("failed to decode dataset body")?
    } else {
        fs::read_to_string(dataset_spec)
            .with_context(|| format!("failed to read dataset {dataset_spec}"))?
    };
    if dataset_spec.ends_with(".jsonl") {
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str::<BenchmarkInstance>(line).context("invalid jsonl instance")
            })
            .collect()
    } else {
        serde_json::from_str(&raw).context("invalid json dataset")
    }
}

pub fn resolve_dataset_path(dataset: Option<&Path>, subset: &str, split: &str) -> Result<String> {
    if let Some(dataset) = dataset {
        return Ok(dataset.to_string_lossy().into_owned());
    }
    if let Some(dataset_id) = DATASET_ALIASES
        .iter()
        .find(|(name, _)| *name == subset)
        .map(|(_, dataset_id)| (*dataset_id).to_string())
    {
        return Ok(dataset_id);
    }
    if let Ok(base_url) = std::env::var("MSWEA_SWEBENCH_DATASET_BASE_URL") {
        let filename = format!("{subset}.{split}.json");
        return Ok(format!("{}/{}", base_url.trim_end_matches('/'), filename));
    }
    let root = std::env::var("MSWEA_SWEBENCH_DATASET_DIR")
        .context("dataset path not provided; set --dataset or MSWEA_SWEBENCH_DATASET_DIR")?;
    let filename = format!("{subset}.{split}.json");
    Ok(PathBuf::from(root).join(filename).display().to_string())
}

fn is_hf_dataset_spec(dataset_spec: &str) -> bool {
    dataset_spec.starts_with("hf://")
        || (dataset_spec.contains('/')
            && !dataset_spec.starts_with("http://")
            && !dataset_spec.starts_with("https://")
            && !Path::new(dataset_spec).exists())
}

fn normalize_hf_dataset_spec(dataset_spec: &str) -> String {
    dataset_spec.trim_start_matches("hf://").to_string()
}

fn load_hf_dataset(dataset_spec: &str, split: &str) -> Result<Vec<BenchmarkInstance>> {
    let dataset = normalize_hf_dataset_spec(dataset_spec);
    let config = std::env::var("MSWEA_HF_DATASET_CONFIG").unwrap_or_else(|_| "default".to_string());
    let mut offset = 0usize;
    let mut instances = Vec::new();
    loop {
        let url = reqwest::Url::parse_with_params(
            "https://datasets-server.huggingface.co/rows",
            &[
                ("dataset", dataset.as_str()),
                ("config", config.as_str()),
                ("split", split),
                ("offset", &offset.to_string()),
                ("length", "100"),
            ],
        )?;
        let response: Value = reqwest::blocking::get(url)
            .with_context(|| format!("failed to fetch huggingface dataset {dataset}:{split}"))?
            .error_for_status()
            .with_context(|| format!("huggingface dataset request failed {dataset}:{split}"))?
            .json()
            .context("failed to decode huggingface dataset rows response")?;
        let rows = response["rows"]
            .as_array()
            .context("huggingface dataset rows missing")?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let row_value = row.get("row").cloned().unwrap_or_else(|| row.clone());
            instances.push(
                serde_json::from_value::<BenchmarkInstance>(row_value)
                    .context("invalid benchmark instance in huggingface dataset")?,
            );
        }
        offset += rows.len();
        if offset >= response["num_rows_total"].as_u64().unwrap_or(offset as u64) as usize {
            break;
        }
    }
    Ok(instances)
}

pub fn filter_instances(
    mut instances: Vec<BenchmarkInstance>,
    filter_spec: &str,
    slice_spec: &str,
    shuffle: bool,
) -> Result<Vec<BenchmarkInstance>> {
    if !filter_spec.is_empty() {
        let regex = Regex::new(filter_spec)?;
        instances.retain(|inst| regex.is_match(&inst.instance_id));
    }
    if shuffle {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        instances.shuffle(&mut rng);
    }
    if !slice_spec.is_empty() {
        let parts: Vec<Option<usize>> = slice_spec
            .split(':')
            .map(|part| {
                if part.is_empty() {
                    Ok(None)
                } else {
                    part.parse::<usize>()
                        .map(Some)
                        .context("invalid slice component")
                }
            })
            .collect::<Result<_>>()?;
        if parts.len() > 2 {
            bail!("slice must look like start:end");
        }
        let start = parts.first().copied().flatten().unwrap_or(0);
        let end = parts.get(1).copied().flatten().unwrap_or(instances.len());
        instances = instances
            .into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect();
    }
    Ok(instances)
}

fn build_model(config: &Config) -> Result<Box<dyn Model>> {
    let model_class = config.model.model_class.as_str();
    let boxed: Box<dyn Model> = match model_class {
        "openai_compatible" | "litellm" | "litellm_textbased" | "default" | "text" => {
            let api_key = std::env::var(&config.model.api_key_env).with_context(|| {
                format!(
                    "missing API key in environment variable {}",
                    config.model.api_key_env
                )
            })?;
            Box::new(ApiModel::openai_compatible(
                config.model.model_name.clone(),
                config.model.base_url.clone(),
                api_key,
                config.model.temperature,
                ApiStyle::ChatCompletions,
                config.model.multimodal_regex.clone(),
                config.model.cost_tracking.clone(),
                config.model.set_cache_control.clone(),
            )?)
        }
        "litellm_response" | "response" => {
            let api_key = std::env::var(&config.model.api_key_env).with_context(|| {
                format!(
                    "missing API key in environment variable {}",
                    config.model.api_key_env
                )
            })?;
            Box::new(ApiModel::openai_compatible(
                config.model.model_name.clone(),
                config.model.base_url.clone(),
                api_key,
                config.model.temperature,
                ApiStyle::Responses,
                config.model.multimodal_regex.clone(),
                config.model.cost_tracking.clone(),
                config.model.set_cache_control.clone(),
            )?)
        }
        "openrouter" | "openrouter_textbased" => Box::new(ApiModel::openrouter(
            config.model.model_name.clone(),
            std::env::var("OPENROUTER_API_KEY")
                .context("missing OPENROUTER_API_KEY for openrouter model")?,
            config.model.temperature,
            false,
            config.model.multimodal_regex.clone(),
            config.model.cost_tracking.clone(),
            config.model.set_cache_control.clone(),
        )?),
        "openrouter_response" => Box::new(ApiModel::openrouter(
            config.model.model_name.clone(),
            std::env::var("OPENROUTER_API_KEY")
                .context("missing OPENROUTER_API_KEY for openrouter model")?,
            config.model.temperature,
            true,
            config.model.multimodal_regex.clone(),
            config.model.cost_tracking.clone(),
            config.model.set_cache_control.clone(),
        )?),
        "requesty" => Box::new(ApiModel::requesty(
            config.model.model_name.clone(),
            std::env::var("REQUESTY_API_KEY")
                .context("missing REQUESTY_API_KEY for requesty model")?,
            config.model.temperature,
            config.model.multimodal_regex.clone(),
            config.model.cost_tracking.clone(),
            config.model.set_cache_control.clone(),
        )?),
        "portkey" => Box::new(ApiModel::portkey(
            config.model.model_name.clone(),
            std::env::var("PORTKEY_API_KEY")
                .context("missing PORTKEY_API_KEY for portkey model")?,
            config.model.temperature,
            false,
            config.model.provider.clone(),
            config.model.litellm_model_name_override.clone(),
            config.model.multimodal_regex.clone(),
            config.model.cost_tracking.clone(),
            config.model.set_cache_control.clone(),
        )?),
        "portkey_response" => Box::new(ApiModel::portkey(
            config.model.model_name.clone(),
            std::env::var("PORTKEY_API_KEY")
                .context("missing PORTKEY_API_KEY for portkey model")?,
            config.model.temperature,
            true,
            config.model.provider.clone(),
            config.model.litellm_model_name_override.clone(),
            config.model.multimodal_regex.clone(),
            config.model.cost_tracking.clone(),
            config.model.set_cache_control.clone(),
        )?),
        _ => bail!("unknown model_class {}", model_class),
    };
    Ok(boxed)
}

pub async fn process_single_instance(
    mut config: Config,
    instance: BenchmarkInstance,
    output_dir: PathBuf,
) -> Result<Trajectory> {
    if config.environment.image.is_none() {
        match config.environment.environment_class {
            crate::config::EnvironmentClass::Docker => {
                config.environment.image = Some(get_swebench_image_name(&instance));
            }
            crate::config::EnvironmentClass::Singularity
            | crate::config::EnvironmentClass::Contree => {
                config.environment.image =
                    Some(format!("docker://{}", get_swebench_image_name(&instance)));
            }
            _ => {}
        }
    }
    let output_path = output_dir
        .join(&instance.instance_id)
        .join(format!("{}.traj.json", instance.instance_id));
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let environment = build_environment(&config.environment)?;
    maybe_run_startup(environment.as_ref(), &config.run.env_startup_command)?;
    let runtime = build_runtime(config.clone(), output_path);
    let model = build_model(&config)?;
    let mut agent = Agent::new(runtime, model, environment);
    agent
        .run(&instance.problem_statement, Some(instance.instance_id))
        .await
}

pub async fn run_batch(
    config: Config,
    instances: Vec<BenchmarkInstance>,
    output_dir: PathBuf,
    workers: usize,
    redo_existing: bool,
) -> Result<()> {
    fs::create_dir_all(&output_dir)?;
    let preds = Arc::new(Mutex::new(serde_json::Map::<String, Value>::new()));
    let statuses = Arc::new(Mutex::new(
        std::collections::BTreeMap::<String, Vec<String>>::new(),
    ));
    let progress = ProgressBar::new(instances.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )?
        .progress_chars("##-"),
    );
    let runtime = tokio::runtime::Handle::current();
    let existing = if !redo_existing {
        load_existing_preds(&output_dir.join("preds.json"))?
    } else {
        std::collections::BTreeSet::new()
    };
    let instances: Vec<_> = instances
        .into_iter()
        .filter(|instance| redo_existing || !existing.contains(&instance.instance_id))
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .context("failed to build rayon pool")?;

    let results: Vec<Result<()>> = pool.install(|| {
        instances
            .into_par_iter()
            .map(|instance| {
                let config = config.clone();
                let output_dir = output_dir.clone();
                let preds = Arc::clone(&preds);
                let statuses = Arc::clone(&statuses);
                let progress = progress.clone();
                runtime.block_on(async move {
                    let instance_id = instance.instance_id.clone();
                    progress.set_message(instance_id.clone());
                    let traj = process_single_instance(config, instance, output_dir).await?;
                    let mut guard = preds.lock().expect("preds mutex poisoned");
                    guard.insert(
                        instance_id.clone(),
                        serde_json::json!({
                            "instance_id": instance_id,
                            "model_name_or_path": traj.info.model_name,
                            "model_patch": traj.info.submission,
                        }),
                    );
                    let mut status_guard = statuses.lock().expect("statuses mutex poisoned");
                    status_guard
                        .entry(traj.info.exit_status.clone())
                        .or_default()
                        .push(instance_id);
                    progress.inc(1);
                    Ok(())
                })
            })
            .collect()
    });

    for result in results {
        result?;
    }
    progress.finish_with_message("completed");

    let preds_path = output_dir.join("preds.json");
    let guard = preds.lock().expect("preds mutex poisoned");
    fs::write(preds_path, serde_json::to_string_pretty(&*guard)?)?;
    let statuses_path = output_dir.join("exit_statuses.yaml");
    let statuses_guard = statuses.lock().expect("statuses mutex poisoned");
    fs::write(statuses_path, serde_yaml::to_string(&*statuses_guard)?)?;
    for (status, instances) in statuses_guard.iter() {
        println!("{status}: {}", instances.len());
    }
    Ok(())
}

fn get_swebench_image_name(instance: &BenchmarkInstance) -> String {
    if let Some(image) = &instance.image_name {
        return image.clone();
    }
    if let Some(image) = &instance.docker_image {
        return image.clone();
    }
    let docker_safe = instance.instance_id.replace("__", "_1776_").to_lowercase();
    format!("docker.io/swebench/sweb.eval.x86_64.{docker_safe}:latest")
}

fn load_existing_preds(path: &Path) -> Result<std::collections::BTreeSet<String>> {
    if !path.exists() {
        return Ok(std::collections::BTreeSet::new());
    }
    let raw = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&raw).context("invalid preds.json")?;
    let mut set = std::collections::BTreeSet::new();
    if let Some(map) = value.as_object() {
        set.extend(map.keys().cloned());
    }
    Ok(set)
}
