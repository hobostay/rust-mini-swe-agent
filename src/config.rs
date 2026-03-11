use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    Human,
    Confirm,
    Yolo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub system_template: String,
    pub instance_template: String,
    #[serde(default)]
    pub step_limit: usize,
    #[serde(default = "default_cost_limit")]
    pub cost_limit: f64,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
    #[serde(default = "default_agent_mode")]
    pub mode: AgentMode,
    #[serde(default)]
    pub whitelist_actions: Vec<String>,
    #[serde(default = "default_confirm_exit")]
    pub confirm_exit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentClass {
    Local,
    Docker,
    Singularity,
    Bubblewrap,
    Contree,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    #[serde(default = "default_environment_class")]
    pub environment_class: EnvironmentClass,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub image_tag: Option<String>,
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default = "default_interpreter")]
    pub interpreter: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub forward_env: Vec<String>,
    #[serde(default = "default_cwd_auto_create")]
    pub cwd_auto_create: bool,
    #[serde(default)]
    pub import_username: Option<String>,
    #[serde(default)]
    pub import_password: Option<String>,
    #[serde(default)]
    pub contree_config: BTreeMap<String, Value>,
    #[serde(default = "default_sandbox_build_retries")]
    pub sandbox_build_retries: usize,
    #[serde(default = "default_global_args")]
    pub global_args: Vec<String>,
    #[serde(default = "default_exec_args")]
    pub exec_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    #[serde(default = "default_model_class")]
    pub model_class: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub set_cache_control: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default = "default_cost_tracking")]
    pub cost_tracking: String,
    #[serde(default)]
    pub litellm_model_name_override: Option<String>,
    #[serde(default)]
    pub multimodal_regex: String,
    #[serde(default = "default_observation_template")]
    pub observation_template: String,
    #[serde(default = "default_format_error_template")]
    pub format_error_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunConfig {
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub env_startup_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub agent: AgentConfig,
    pub environment: EnvironmentConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub run: RunConfig,
}

fn default_cost_limit() -> f64 {
    3.0
}

fn default_timeout() -> u64 {
    30
}

fn default_temperature() -> f64 {
    0.2
}

fn default_interpreter() -> Vec<String> {
    vec!["bash".to_string(), "-lc".to_string()]
}

fn default_cwd_auto_create() -> bool {
    true
}

fn default_sandbox_build_retries() -> usize {
    3
}

fn default_global_args() -> Vec<String> {
    vec!["--quiet".to_string()]
}

fn default_exec_args() -> Vec<String> {
    vec![
        "--contain".to_string(),
        "--cleanenv".to_string(),
        "--fakeroot".to_string(),
    ]
}

fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_model_class() -> String {
    "openai_compatible".to_string()
}

fn default_api_key_env() -> String {
    "OPENAI_API_KEY".to_string()
}

fn default_cost_tracking() -> String {
    "default".to_string()
}

fn default_observation_template() -> String {
    "Observation:\n```text\n{{ output.output }}\n```\nreturncode: {{ output.returncode }}\n{% if output.exception_info %}exception: {{ output.exception_info }}{% endif %}".to_string()
}

fn default_format_error_template() -> String {
    "Your previous response could not be parsed into a valid bash action. Return a bash tool call or a fenced bash code block.".to_string()
}

fn default_agent_mode() -> AgentMode {
    AgentMode::Confirm
}

fn default_confirm_exit() -> bool {
    true
}

fn default_environment_class() -> EnvironmentClass {
    EnvironmentClass::Local
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agent: AgentConfig {
                system_template: "You are a helpful assistant that can interact with a computer.".to_string(),
                instance_template: "Please solve this issue: {{task}}\n\nYou can execute bash commands and edit files to implement the necessary changes.\n\n## Recommended Workflow\n1. Analyze the codebase.\n2. Reproduce the issue.\n3. Edit the source.\n4. Verify the fix.\n5. Test edge cases.\n6. Finish with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n\nEvery response must include at least one bash tool call.".to_string(),
                step_limit: 0,
                cost_limit: default_cost_limit(),
                output_path: None,
                mode: default_agent_mode(),
                whitelist_actions: Vec::new(),
                confirm_exit: true,
            },
            environment: EnvironmentConfig {
                environment_class: default_environment_class(),
                cwd: None,
                timeout_secs: default_timeout(),
                image: None,
                image_tag: None,
                executable: None,
                interpreter: default_interpreter(),
                env: BTreeMap::new(),
                forward_env: Vec::new(),
                cwd_auto_create: default_cwd_auto_create(),
                import_username: None,
                import_password: None,
                contree_config: BTreeMap::new(),
                sandbox_build_retries: default_sandbox_build_retries(),
                global_args: default_global_args(),
                exec_args: default_exec_args(),
            },
            model: ModelConfig {
                model_name: env::var("MSWEA_RUST_MODEL")
                    .or_else(|_| env::var("OPENAI_MODEL"))
                    .unwrap_or_else(|_| "gpt-4.1-mini".to_string()),
                model_class: default_model_class(),
                base_url: default_base_url(),
                api_key_env: default_api_key_env(),
                temperature: default_temperature(),
                set_cache_control: None,
                provider: None,
                cost_tracking: default_cost_tracking(),
                litellm_model_name_override: None,
                multimodal_regex: String::new(),
                observation_template: default_observation_template(),
                format_error_template: default_format_error_template(),
            },
            run: RunConfig {
                task: None,
                env_startup_command: None,
            },
        }
    }
}

impl Config {
    pub fn load(config_specs: &[String]) -> Result<Self> {
        let mut merged = serde_json::to_value(Self::default())?;
        if config_specs.is_empty() {
            return serde_json::from_value(merged).context("failed to build default config");
        }
        for spec in config_specs {
            let value = get_config_from_spec(spec)?;
            merge_values(&mut merged, value);
        }
        serde_json::from_value(merged).context("failed to deserialize merged config")
    }

    pub fn global_config_dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rust-mini-swe-agent")
    }

    pub fn global_env_file() -> PathBuf {
        Self::global_config_dir().join(".env")
    }

    pub fn default_output_path() -> PathBuf {
        Self::global_config_dir().join("last_run.traj.json")
    }

    pub fn builtin_config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config")
    }
}

pub fn load_global_env() -> Result<()> {
    let env_path = Config::global_env_file();
    if env_path.exists() {
        dotenvy::from_path_iter(&env_path)
            .with_context(|| format!("failed to read {}", env_path.display()))?
            .flatten()
            .for_each(|item| {
                if env::var_os(&item.0).is_none() {
                    // SAFETY: process-wide env mutation is intentional during startup.
                    unsafe { env::set_var(item.0, item.1) };
                }
            });
    }
    Ok(())
}

fn get_config_from_spec(spec: &str) -> Result<Value> {
    if spec.contains('=') {
        return key_value_spec_to_json(spec);
    }
    let path = get_config_path(spec)?;
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse config {}", path.display()))?;
    serde_json::to_value(yaml_value).context("failed to convert yaml config to json")
}

fn get_config_path(spec: &str) -> Result<PathBuf> {
    let requested = PathBuf::from(spec);
    let with_suffix = if requested.extension().is_none() {
        requested.with_extension("yaml")
    } else {
        requested.clone()
    };
    let candidates = [
        with_suffix.clone(),
        env::current_dir()?.join(&with_suffix),
        Config::builtin_config_dir().join(&with_suffix),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!("config file not found for spec {}", spec)
}

fn key_value_spec_to_json(spec: &str) -> Result<Value> {
    let (key, raw_value) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("invalid key=value config spec: {}", spec))?;
    let value: Value =
        serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_string()));
    let keys: Vec<&str> = key.split('.').collect();
    let mut leaf = value;
    for segment in keys.iter().rev() {
        let mut map = BTreeMap::new();
        map.insert((*segment).to_string(), leaf);
        leaf = serde_json::to_value(map)?;
    }
    Ok(leaf)
}

fn merge_values(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(target_map), Value::Object(source_map)) => {
            for (key, value) in source_map {
                merge_values(target_map.entry(key).or_insert(Value::Null), value);
            }
        }
        (slot, value) => *slot = value,
    }
}
