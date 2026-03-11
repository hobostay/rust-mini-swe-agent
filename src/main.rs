mod agent;
mod bench;
mod config;
mod environment;
mod inspector;
mod model;
mod template;
mod types;

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use agent::{Agent, build_runtime, print_trajectory_summary};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use config::{AgentMode, Config, load_global_env};
use environment::{build_environment, maybe_run_startup};
use model::{ApiModel, ApiStyle};

#[derive(Debug, Parser)]
#[command(name = "rust-mini-swe-agent")]
#[command(about = "A Rust recreation of mini-swe-agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Mini(MiniArgs),
    Bench(BenchArgs),
    BenchSingle(BenchSingleArgs),
    Inspector(InspectorArgs),
    Config(ConfigArgs),
}

#[derive(Debug, Args, Clone)]
struct SharedRunArgs {
    #[arg(short = 'm', long)]
    model: Option<String>,
    #[arg(short = 'c', long = "config")]
    config_specs: Vec<String>,
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long)]
    model_class: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    api_key_env: Option<String>,
    #[arg(long)]
    cwd: Option<PathBuf>,
    #[arg(long)]
    environment_class: Option<String>,
    #[arg(long)]
    image: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    yolo: bool,
}

#[derive(Debug, Args)]
struct MiniArgs {
    #[command(flatten)]
    shared: SharedRunArgs,
    #[arg(short, long)]
    task: Option<String>,
}

#[derive(Debug, Args)]
struct BenchArgs {
    #[command(flatten)]
    shared: SharedRunArgs,
    #[arg(long)]
    dataset: Option<PathBuf>,
    #[arg(long, default_value = "lite")]
    subset: String,
    #[arg(long, default_value = "dev")]
    split: String,
    #[arg(long, default_value = "")]
    filter: String,
    #[arg(long, default_value = "")]
    slice: String,
    #[arg(long, default_value_t = false)]
    shuffle: bool,
    #[arg(short = 'w', long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = false)]
    redo_existing: bool,
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Debug, Args)]
struct BenchSingleArgs {
    #[command(flatten)]
    shared: SharedRunArgs,
    #[arg(long)]
    dataset: Option<PathBuf>,
    #[arg(long, default_value = "lite")]
    subset: String,
    #[arg(long, default_value = "dev")]
    split: String,
    #[arg(long)]
    instance: String,
    #[arg(long = "output-file")]
    output_file: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct InspectorArgs {
    #[arg(default_value = ".")]
    path: PathBuf,
    #[arg(long)]
    step: Option<usize>,
}

#[derive(Debug, Args)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommands,
}

#[derive(Debug, Subcommand)]
enum ConfigCommands {
    Show,
    Set { key: String, value: String },
    Unset { key: String },
    Setup,
}

#[tokio::main]
async fn main() -> Result<()> {
    load_global_env()?;
    let cli = Cli::parse();
    match cli.command {
        Commands::Mini(args) => run_mini(args).await,
        Commands::Bench(args) => run_bench(args).await,
        Commands::BenchSingle(args) => run_bench_single(args).await,
        Commands::Inspector(args) => run_inspector(args),
        Commands::Config(args) => run_config(args),
    }
}

async fn run_mini(args: MiniArgs) -> Result<()> {
    let config = load_and_apply_shared(args.shared.clone(), "mini")?;
    let task = match args.task.or_else(|| config.run.task.clone()) {
        Some(task) => task,
        None => read_task_from_stdin()?,
    };
    let output_path = config
        .agent
        .output_path
        .clone()
        .unwrap_or_else(Config::default_output_path);
    let environment = build_environment(&config.environment)?;
    maybe_run_startup(environment.as_ref(), &config.run.env_startup_command)?;
    let model = build_model(&config)?;
    let runtime = build_runtime(config, output_path);
    let mut agent = Agent::new(runtime, model, environment);
    let trajectory = agent.run(&task, None).await?;
    print_trajectory_summary(&trajectory)
}

async fn run_bench(args: BenchArgs) -> Result<()> {
    let config = load_and_apply_shared(args.shared, "benchmarks/swebench")?;
    let dataset_path =
        bench::resolve_dataset_path(args.dataset.as_deref(), &args.subset, &args.split)?;
    let instances = bench::load_instances(&dataset_path, &args.split)?;
    let instances = bench::filter_instances(instances, &args.filter, &args.slice, args.shuffle)?;
    bench::run_batch(
        config,
        instances,
        args.output_dir,
        args.workers,
        args.redo_existing,
    )
    .await
}

async fn run_bench_single(args: BenchSingleArgs) -> Result<()> {
    let config = load_and_apply_shared(args.shared, "benchmarks/swebench")?;
    let dataset_path =
        bench::resolve_dataset_path(args.dataset.as_deref(), &args.subset, &args.split)?;
    let instances = bench::load_instances(&dataset_path, &args.split)?;
    let instance = if args.instance.chars().all(|c| c.is_ascii_digit()) {
        let index = args.instance.parse::<usize>()?;
        instances
            .into_iter()
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("instance index out of range: {}", index))?
    } else {
        instances
            .into_iter()
            .find(|inst| inst.instance_id == args.instance)
            .ok_or_else(|| anyhow::anyhow!("instance not found: {}", args.instance))?
    };
    let output_dir = args
        .output_file
        .unwrap_or_else(|| Config::default_output_path());
    let output_dir = if output_dir.extension().and_then(|x| x.to_str()) == Some("json") {
        output_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        output_dir
    };
    let traj = bench::process_single_instance(config, instance, output_dir).await?;
    print_trajectory_summary(&traj)
}

fn run_inspector(args: InspectorArgs) -> Result<()> {
    let files = inspector::collect_trajectory_files(&args.path)?;
    if args.step.is_none() {
        return inspector::run_tui(files);
    }
    for file in files {
        inspector::print_trajectory(&file, args.step)?;
    }
    Ok(())
}

fn run_config(args: ConfigArgs) -> Result<()> {
    fs::create_dir_all(Config::global_config_dir())?;
    let env_path = Config::global_env_file();
    match args.command {
        ConfigCommands::Show => {
            if env_path.exists() {
                println!("{}", fs::read_to_string(env_path)?);
            }
        }
        ConfigCommands::Set { key, value } => {
            upsert_env_var(&env_path, &key, Some(&value))?;
        }
        ConfigCommands::Unset { key } => {
            upsert_env_var(&env_path, &key, None)?;
        }
        ConfigCommands::Setup => {
            let model: String = dialoguer::Input::new()
                .with_prompt("Default model name")
                .allow_empty(true)
                .interact_text()?;
            let key_name: String = dialoguer::Input::new()
                .with_prompt("API key env name")
                .default("OPENAI_API_KEY".to_string())
                .interact_text()?;
            let key_value: String = dialoguer::Input::new()
                .with_prompt("API key value")
                .allow_empty(true)
                .interact_text()?;
            if !model.trim().is_empty() {
                upsert_env_var(&env_path, "MSWEA_RUST_MODEL", Some(model.trim()))?;
            }
            if !key_value.trim().is_empty() {
                upsert_env_var(&env_path, &key_name, Some(key_value.trim()))?;
            }
            println!("saved {}", env_path.display());
        }
    }
    Ok(())
}

fn upsert_env_var(path: &Path, key: &str, value: Option<&str>) -> Result<()> {
    let existing = if path.exists() {
        fs::read_to_string(path)?
    } else {
        String::new()
    };
    let mut lines: Vec<String> = existing
        .lines()
        .filter(|line| !line.starts_with(&format!("{key}=")))
        .map(ToOwned::to_owned)
        .collect();
    if let Some(value) = value {
        lines.push(format!("{key}={value}"));
    }
    fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}

fn load_and_apply_shared(shared: SharedRunArgs, default_config: &str) -> Result<Config> {
    let config_specs = if shared.config_specs.is_empty() {
        vec![default_config.to_string()]
    } else {
        shared.config_specs.clone()
    };
    let mut config = Config::load(&config_specs)?;
    if let Some(model) = shared.model {
        config.model.model_name = model;
    }
    if let Some(model_class) = shared.model_class {
        config.model.model_class = model_class;
    }
    if let Some(base_url) = shared.base_url {
        config.model.base_url = base_url;
    }
    if let Some(api_key_env) = shared.api_key_env {
        config.model.api_key_env = api_key_env;
    }
    if let Some(cwd) = shared.cwd {
        config.environment.cwd = Some(cwd);
    }
    if let Some(image) = shared.image {
        config.environment.image = Some(image);
    }
    if let Some(output) = shared.output {
        config.agent.output_path = Some(output);
    }
    if let Some(mode) = shared.mode {
        config.agent.mode = parse_mode(&mode)?;
    }
    if shared.yolo {
        config.agent.mode = AgentMode::Yolo;
    }
    if let Some(environment_class) = shared.environment_class {
        config.environment.environment_class = match environment_class.as_str() {
            "local" => config::EnvironmentClass::Local,
            "docker" => config::EnvironmentClass::Docker,
            "singularity" => config::EnvironmentClass::Singularity,
            "bubblewrap" => config::EnvironmentClass::Bubblewrap,
            "contree" => config::EnvironmentClass::Contree,
            _ => bail!("unknown environment_class {}", environment_class),
        };
    }
    Ok(config)
}

fn parse_mode(mode: &str) -> Result<AgentMode> {
    match mode {
        "human" => Ok(AgentMode::Human),
        "confirm" => Ok(AgentMode::Confirm),
        "yolo" => Ok(AgentMode::Yolo),
        _ => bail!("unknown mode {}", mode),
    }
}

fn build_model(config: &Config) -> Result<Box<dyn model::Model>> {
    let model_class = config.model.model_class.as_str();
    let boxed: Box<dyn model::Model> = match model_class {
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

fn read_task_from_stdin() -> Result<String> {
    eprintln!("Enter the task on stdin, then press Ctrl-D:");
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let task = buf.trim().to_string();
    if task.is_empty() {
        bail!("task was empty");
    }
    Ok(task)
}
