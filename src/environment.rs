use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client as BlockingClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use wait_timeout::ChildExt;

use crate::config::{EnvironmentClass, EnvironmentConfig};
use crate::types::{Action, CommandOutput};

pub trait Environment: Send + Sync {
    fn execute(&self, action: &Action) -> Result<CommandOutput>;
}

pub struct LocalEnvironment {
    cwd: Option<PathBuf>,
    timeout_secs: u64,
    executable: String,
    env: Vec<(String, String)>,
    forward_env: Vec<String>,
}

pub struct DockerEnvironment {
    cwd: Option<PathBuf>,
    timeout_secs: u64,
    executable: String,
    interpreter: Vec<String>,
    env: Vec<(String, String)>,
    forward_env: Vec<String>,
    container_id: String,
}

pub struct SingularityEnvironment {
    cwd: Option<PathBuf>,
    timeout_secs: u64,
    executable: String,
    interpreter: Vec<String>,
    env: Vec<(String, String)>,
    forward_env: Vec<String>,
    sandbox_dir: PathBuf,
    global_args: Vec<String>,
    exec_args: Vec<String>,
}

pub struct BubblewrapEnvironment {
    cwd: Option<PathBuf>,
    timeout_secs: u64,
    executable: String,
    interpreter: Vec<String>,
    env: Vec<(String, String)>,
    forward_env: Vec<String>,
}

pub struct ContreeEnvironment {
    image: String,
    image_tag: Option<String>,
    cwd: Option<PathBuf>,
    timeout_secs: u64,
    executable: String,
    interpreter: Vec<String>,
    env: Vec<(String, String)>,
    forward_env: Vec<String>,
    cwd_auto_create: bool,
    import_username: Option<String>,
    import_password: Option<String>,
    contree_env: Vec<(String, String)>,
    rest: Option<ContreeRestClient>,
}

struct ContreeRestClient {
    client: BlockingClient,
    base_url: String,
    auth_header: HeaderValue,
    current_image: Mutex<String>,
}

impl LocalEnvironment {
    pub fn new(
        cwd: Option<PathBuf>,
        timeout_secs: u64,
        executable: String,
        env: Vec<(String, String)>,
        forward_env: Vec<String>,
    ) -> Self {
        Self {
            cwd,
            timeout_secs,
            executable,
            env,
            forward_env,
        }
    }
}

impl DockerEnvironment {
    pub fn new(
        image: String,
        cwd: Option<PathBuf>,
        timeout_secs: u64,
        executable: String,
        interpreter: Vec<String>,
        env: Vec<(String, String)>,
        forward_env: Vec<String>,
    ) -> Result<Self> {
        let container_id = start_docker_container(&executable, &image, cwd.as_deref())?;
        Ok(Self {
            cwd,
            timeout_secs,
            executable,
            interpreter,
            env,
            forward_env,
            container_id,
        })
    }
}

impl SingularityEnvironment {
    pub fn new(
        image: String,
        cwd: Option<PathBuf>,
        timeout_secs: u64,
        executable: String,
        interpreter: Vec<String>,
        env: Vec<(String, String)>,
        forward_env: Vec<String>,
        sandbox_build_retries: usize,
        global_args: Vec<String>,
        exec_args: Vec<String>,
    ) -> Result<Self> {
        let sandbox_dir = unique_temp_path("rust-mini-swe-agent-singularity");
        build_singularity_sandbox(&executable, &image, &sandbox_dir, sandbox_build_retries)?;
        Ok(Self {
            cwd,
            timeout_secs,
            executable,
            interpreter,
            env,
            forward_env,
            sandbox_dir,
            global_args,
            exec_args,
        })
    }
}

impl BubblewrapEnvironment {
    pub fn new(
        cwd: Option<PathBuf>,
        timeout_secs: u64,
        executable: String,
        interpreter: Vec<String>,
        env: Vec<(String, String)>,
        forward_env: Vec<String>,
    ) -> Self {
        Self {
            cwd,
            timeout_secs,
            executable,
            interpreter,
            env,
            forward_env,
        }
    }
}

impl ContreeEnvironment {
    pub fn new(
        image: String,
        image_tag: Option<String>,
        cwd: Option<PathBuf>,
        timeout_secs: u64,
        executable: String,
        interpreter: Vec<String>,
        env: Vec<(String, String)>,
        forward_env: Vec<String>,
        cwd_auto_create: bool,
        import_username: Option<String>,
        import_password: Option<String>,
        contree_config: &std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<Self> {
        let contree_env = contree_env_from_config(contree_config);
        let rest = build_contree_rest_client(
            &image,
            image_tag.clone(),
            import_username.clone(),
            import_password.clone(),
            contree_config,
        )?;
        let environment = Self {
            image,
            image_tag,
            cwd,
            timeout_secs,
            executable,
            interpreter,
            env,
            forward_env,
            cwd_auto_create,
            import_username,
            import_password,
            contree_env,
            rest,
        };
        if environment.cwd_auto_create
            && let Some(cwd) = &environment.cwd
            && cwd != Path::new("/")
        {
            let action = Action {
                command: format!("mkdir -p {}", shell_escape(&cwd.display().to_string())),
                tool_call_id: None,
            };
            let _ = environment.execute(&action)?;
        }
        Ok(environment)
    }
}

fn run_command(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    timeout_secs: u64,
    env: &[(String, String)],
    forward_env: &[String],
) -> Result<CommandOutput> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in forward_env {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }
    for (key, value) in env {
        command.env(key, value);
    }
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {} {:?}", program, args))?;

    let status = match child.wait_timeout(Duration::from_secs(timeout_secs))? {
        Some(status) => status,
        None => {
            child.kill().ok();
            child.wait().ok();
            return Ok(CommandOutput {
                output: String::new(),
                returncode: -1,
                exception_info: format!("command timed out after {}s", timeout_secs),
            });
        }
    };

    let output = child
        .wait_with_output()
        .context("failed to collect command output")?;
    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }

    Ok(CommandOutput {
        output: combined,
        returncode: status.code().unwrap_or(-1),
        exception_info: String::new(),
    })
}

impl Environment for LocalEnvironment {
    fn execute(&self, action: &Action) -> Result<CommandOutput> {
        let args = vec!["-lc".to_string(), action.command.clone()];
        run_command(
            &self.executable,
            &args,
            self.cwd.as_deref().or(Some(Path::new("."))),
            self.timeout_secs,
            &self.env,
            &self.forward_env,
        )
    }
}

impl Environment for DockerEnvironment {
    fn execute(&self, action: &Action) -> Result<CommandOutput> {
        let workdir = self
            .cwd
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/workspace".to_string());
        let mut args = vec!["exec".to_string(), "-w".to_string(), workdir];
        for key in &self.forward_env {
            if let Ok(value) = std::env::var(key) {
                args.push("-e".to_string());
                args.push(format!("{key}={value}"));
            }
        }
        for (key, value) in &self.env {
            args.push("-e".to_string());
            args.push(format!("{key}={value}"));
        }
        args.extend([self.container_id.clone()]);
        args.extend(self.interpreter.clone());
        args.push(action.command.clone());
        run_command(&self.executable, &args, None, self.timeout_secs, &[], &[])
    }
}

impl Environment for SingularityEnvironment {
    fn execute(&self, action: &Action) -> Result<CommandOutput> {
        let mut args = self.global_args.clone();
        args.push("exec".to_string());
        args.extend(self.exec_args.clone());
        if let Some(cwd) = &self.cwd {
            args.push("--pwd".to_string());
            args.push(cwd.display().to_string());
        }
        for key in &self.forward_env {
            if let Ok(value) = std::env::var(key) {
                args.push("--env".to_string());
                args.push(format!("{key}={value}"));
            }
        }
        for (key, value) in &self.env {
            args.push("--env".to_string());
            args.push(format!("{key}={value}"));
        }
        args.push("--writable".to_string());
        args.push(self.sandbox_dir.display().to_string());
        args.extend(self.interpreter.clone());
        args.push(action.command.clone());
        run_command(&self.executable, &args, None, self.timeout_secs, &[], &[])
    }
}

impl Environment for BubblewrapEnvironment {
    fn execute(&self, action: &Action) -> Result<CommandOutput> {
        let workdir = self
            .cwd
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| ".".to_string());
        let mut args = vec![
            "--unshare-user-try".to_string(),
            "--ro-bind".to_string(),
            "/usr".to_string(),
            "/usr".to_string(),
            "--ro-bind".to_string(),
            "/bin".to_string(),
            "/bin".to_string(),
            "--ro-bind".to_string(),
            "/lib".to_string(),
            "/lib".to_string(),
            "--ro-bind".to_string(),
            "/lib64".to_string(),
            "/lib64".to_string(),
            "--ro-bind".to_string(),
            "/etc".to_string(),
            "/etc".to_string(),
            "--tmpfs".to_string(),
            "/tmp".to_string(),
            "--proc".to_string(),
            "/proc".to_string(),
            "--dev".to_string(),
            "/dev".to_string(),
            "--bind".to_string(),
            workdir.clone(),
            workdir.clone(),
            "--chdir".to_string(),
            workdir,
        ];
        for (key, value) in &self.env {
            args.push("--setenv".to_string());
            args.push(key.clone());
            args.push(value.clone());
        }
        for key in &self.forward_env {
            if let Ok(value) = std::env::var(key) {
                args.push("--setenv".to_string());
                args.push(key.clone());
                args.push(value);
            }
        }
        args.extend(self.interpreter.clone());
        args.push(action.command.clone());
        run_command(&self.executable, &args, None, self.timeout_secs, &[], &[])
    }
}

impl Environment for ContreeEnvironment {
    fn execute(&self, action: &Action) -> Result<CommandOutput> {
        if let Some(rest) = &self.rest {
            return rest.execute(
                &self.interpreter,
                self.cwd.as_deref(),
                self.timeout_secs,
                &self.env,
                &self.forward_env,
                action,
            );
        }
        let workdir = self
            .cwd
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/".to_string());
        let wrapped =
            build_contree_shell_command(&workdir, &self.env, &self.forward_env, &action.command);
        let mut args = vec![
            "run".to_string(),
            "--rm".to_string(),
            "-w".to_string(),
            workdir,
        ];
        args.push(normalize_contree_image(&self.image));
        if let Some(image_tag) = &self.image_tag {
            args.push("--tag".to_string());
            args.push(image_tag.clone());
        }
        if let Some(username) = &self.import_username {
            args.push("--username".to_string());
            args.push(username.clone());
        }
        if let Some(password) = &self.import_password {
            args.push("--password".to_string());
            args.push(password.clone());
        }
        args.extend(self.interpreter.clone());
        args.push(wrapped);
        run_command(
            &self.executable,
            &args,
            None,
            self.timeout_secs,
            &self.contree_env,
            &[],
        )
    }
}

pub fn build_environment(config: &EnvironmentConfig) -> Result<Box<dyn Environment>> {
    let executable = config
        .executable
        .clone()
        .unwrap_or_else(|| match config.environment_class {
            EnvironmentClass::Local => "/bin/sh".to_string(),
            EnvironmentClass::Docker => "docker".to_string(),
            EnvironmentClass::Singularity => "singularity".to_string(),
            EnvironmentClass::Bubblewrap => "bwrap".to_string(),
            EnvironmentClass::Contree => "contree".to_string(),
        });
    let env_pairs: Vec<(String, String)> = config
        .env
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    match config.environment_class {
        EnvironmentClass::Local => Ok(Box::new(LocalEnvironment::new(
            config.cwd.clone(),
            config.timeout_secs,
            executable,
            env_pairs,
            config.forward_env.clone(),
        ))),
        EnvironmentClass::Docker => {
            let image = config
                .image
                .clone()
                .ok_or_else(|| anyhow::anyhow!("docker environment requires image"))?;
            Ok(Box::new(DockerEnvironment::new(
                image,
                config.cwd.clone(),
                config.timeout_secs,
                executable,
                config.interpreter.clone(),
                env_pairs,
                config.forward_env.clone(),
            )?))
        }
        EnvironmentClass::Singularity => {
            let image = config
                .image
                .clone()
                .ok_or_else(|| anyhow::anyhow!("singularity environment requires image"))?;
            Ok(Box::new(SingularityEnvironment::new(
                image,
                config.cwd.clone(),
                config.timeout_secs,
                executable,
                config.interpreter.clone(),
                env_pairs,
                config.forward_env.clone(),
                config.sandbox_build_retries,
                config.global_args.clone(),
                config.exec_args.clone(),
            )?))
        }
        EnvironmentClass::Bubblewrap => Ok(Box::new(BubblewrapEnvironment::new(
            config.cwd.clone(),
            config.timeout_secs,
            executable,
            config.interpreter.clone(),
            env_pairs,
            config.forward_env.clone(),
        ))),
        EnvironmentClass::Contree => {
            let image = config
                .image
                .clone()
                .ok_or_else(|| anyhow::anyhow!("contree environment requires image"))?;
            Ok(Box::new(ContreeEnvironment::new(
                image,
                config.image_tag.clone(),
                config.cwd.clone(),
                config.timeout_secs,
                executable,
                config.interpreter.clone(),
                env_pairs,
                config.forward_env.clone(),
                config.cwd_auto_create,
                config.import_username.clone(),
                config.import_password.clone(),
                &config.contree_config,
            )?))
        }
    }
}

pub fn maybe_run_startup(
    environment: &dyn Environment,
    startup_command: &Option<String>,
) -> Result<()> {
    if let Some(command) = startup_command {
        let output = environment.execute(&Action {
            command: command.clone(),
            tool_call_id: None,
        })?;
        if output.returncode != 0 {
            bail!("startup command failed: {}", output.output);
        }
    }
    Ok(())
}

fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn unique_temp_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

fn start_docker_container(executable: &str, image: &str, cwd: Option<&Path>) -> Result<String> {
    let workdir = cwd
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "/workspace".to_string());
    let container_name = format!(
        "rust-mini-swe-agent-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let args = vec![
        "run".to_string(),
        "-d".to_string(),
        "--rm".to_string(),
        "--name".to_string(),
        container_name,
        "-w".to_string(),
        workdir,
        image.to_string(),
        "sleep".to_string(),
        "2h".to_string(),
    ];
    let output = run_command(executable, &args, None, 120, &[], &[])?;
    if output.returncode != 0 {
        bail!("failed to start docker container: {}", output.output);
    }
    Ok(output.output.trim().to_string())
}

fn build_singularity_sandbox(
    executable: &str,
    image: &str,
    sandbox_dir: &Path,
    retries: usize,
) -> Result<()> {
    let max_retries = retries.max(1);
    for attempt in 0..max_retries {
        fs::create_dir_all(sandbox_dir).ok();
        let args = vec![
            "build".to_string(),
            "--sandbox".to_string(),
            sandbox_dir.display().to_string(),
            image.to_string(),
        ];
        let output = run_command(executable, &args, None, 300, &[], &[])?;
        if output.returncode == 0 {
            return Ok(());
        }
        let _ = fs::remove_dir_all(sandbox_dir);
        if attempt + 1 == max_retries {
            bail!("failed to build singularity sandbox: {}", output.output);
        }
    }
    Ok(())
}

fn normalize_contree_image(image: &str) -> String {
    image.strip_prefix("docker://").unwrap_or(image).to_string()
}

fn build_contree_shell_command(
    cwd: &str,
    env: &[(String, String)],
    forward_env: &[String],
    command: &str,
) -> String {
    let mut wrapped = String::new();
    for key in forward_env {
        if let Ok(value) = std::env::var(key) {
            wrapped.push_str(&format!("export {key}={}; ", shell_escape(&value)));
        }
    }
    for (key, value) in env {
        wrapped.push_str(&format!("export {key}={}; ", shell_escape(value)));
    }
    wrapped.push_str(&format!("cd {} && {}", shell_escape(cwd), command));
    wrapped
}

fn contree_env_from_config(
    config: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Vec<(String, String)> {
    let mut env = Vec::new();
    for (key, value) in config {
        let env_key = format!("CONTREE_{}", key.replace('-', "_").to_ascii_uppercase());
        let env_value = match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        env.push((env_key, env_value));
    }
    env
}

fn build_contree_rest_client(
    image: &str,
    image_tag: Option<String>,
    import_username: Option<String>,
    import_password: Option<String>,
    config: &std::collections::BTreeMap<String, Value>,
) -> Result<Option<ContreeRestClient>> {
    let base_url = config
        .get("base_url")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("CONTREE_BASE_URL").ok());
    let token = config
        .get("token")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("CONTREE_TOKEN").ok());
    let (Some(base_url), Some(token)) = (base_url, token) else {
        return Ok(None);
    };

    let client = BlockingClient::new();
    let auth_header = HeaderValue::from_str(&format!("Bearer {token}"))
        .context("invalid contree bearer token")?;
    let tag = image_tag.unwrap_or_else(|| derive_contree_tag(image));
    import_contree_image(
        &client,
        &base_url,
        &auth_header,
        image,
        &tag,
        import_username,
        import_password,
    )?;
    Ok(Some(ContreeRestClient {
        client,
        base_url: normalize_contree_base_url(&base_url),
        auth_header,
        current_image: Mutex::new(format!("tag:{tag}")),
    }))
}

fn import_contree_image(
    client: &BlockingClient,
    base_url: &str,
    auth_header: &HeaderValue,
    image: &str,
    tag: &str,
    import_username: Option<String>,
    import_password: Option<String>,
) -> Result<()> {
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, auth_header.clone());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let mut body = json!({
        "registry": normalize_contree_image(image),
        "tag": tag,
        "timeout": 300,
    });
    if import_username.is_some() || import_password.is_some() {
        body["credentials"] = json!({
            "username": import_username.unwrap_or_default(),
            "password": import_password.unwrap_or_default(),
        });
    }
    let response = client
        .post(format!(
            "{}/images/import",
            normalize_contree_base_url(base_url)
        ))
        .headers(headers)
        .json(&body)
        .send()
        .context("failed to import contree image")?;
    if !response.status().is_success() && response.status().as_u16() != 202 {
        bail!("contree image import failed with {}", response.status());
    }
    if let Some(location) = response
        .headers()
        .get("Location")
        .and_then(|value| value.to_str().ok())
    {
        let client = ContreeRestClient {
            client: client.clone(),
            base_url: normalize_contree_base_url(base_url),
            auth_header: auth_header.clone(),
            current_image: Mutex::new(format!("tag:{tag}")),
        };
        let _ = client.poll_operation(location)?;
    }
    Ok(())
}

impl ContreeRestClient {
    fn execute(
        &self,
        interpreter: &[String],
        cwd: Option<&Path>,
        timeout_secs: u64,
        env: &[(String, String)],
        forward_env: &[String],
        action: &Action,
    ) -> Result<CommandOutput> {
        let cwd = cwd
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "/".to_string());
        let command = build_contree_shell_command(&cwd, env, forward_env, &action.command);
        let image = self
            .current_image
            .lock()
            .expect("contree image mutex poisoned")
            .clone();
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, self.auth_header.clone());
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let body = json!({
            "image": image,
            "command": command,
            "shell": true,
            "cwd": cwd,
            "timeout": timeout_secs,
            "env": merged_env(env, forward_env),
            "disposable": false,
            "truncate_output_at": 1048576,
            "stdin": Value::Null,
            "hostname": Value::Null,
            "interpreter": interpreter,
        });
        let response = self
            .client
            .post(format!("{}/instances", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .context("failed to create contree instance")?;
        if !response.status().is_success() {
            bail!(
                "contree instance creation failed with {}",
                response.status()
            );
        }
        let location = response
            .headers()
            .get("Location")
            .and_then(|value| value.to_str().ok())
            .context("contree instance response missing Location header")?;
        let operation = self.poll_operation(location)?;
        if let Some(image_ref) = extract_contree_image_ref(&operation) {
            *self
                .current_image
                .lock()
                .expect("contree image mutex poisoned") = image_ref;
        }
        let result = &operation["result"];
        let stdout = result["stdout"]["content"].as_str().unwrap_or_default();
        let stderr = result["stderr"]["content"].as_str().unwrap_or_default();
        Ok(CommandOutput {
            output: format!("{stdout}{stderr}"),
            returncode: result["exit_code"].as_i64().unwrap_or(-1) as i32,
            exception_info: if result["timed_out"].as_bool().unwrap_or(false) {
                "command timed out".to_string()
            } else {
                String::new()
            },
        })
    }

    fn poll_operation(&self, location: &str) -> Result<Value> {
        let url = if location.starts_with("http://") || location.starts_with("https://") {
            location.to_string()
        } else if location.starts_with("/v1/") {
            format!("{}{}", self.base_url.trim_end_matches("/v1"), location)
        } else if location.starts_with('/') {
            format!("{}{}", self.base_url.trim_end_matches("/v1"), location)
        } else {
            format!("{}/{}", self.base_url, location.trim_start_matches('/'))
        };
        loop {
            let response = self
                .client
                .get(&url)
                .header(AUTHORIZATION, self.auth_header.clone())
                .send()
                .with_context(|| format!("failed polling contree operation {}", url))?;
            if !response.status().is_success() {
                bail!(
                    "contree operation polling failed with {}",
                    response.status()
                );
            }
            let retry_after = response
                .headers()
                .get("Retry-After")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(1);
            let operation: Value = response.json().context("invalid contree operation json")?;
            match operation["status"].as_str().unwrap_or_default() {
                "SUCCESS" => return Ok(operation),
                "FAILED" | "CANCELLED" => bail!("contree operation failed: {}", operation),
                _ => std::thread::sleep(Duration::from_secs(retry_after)),
            }
        }
    }
}

fn merged_env(env: &[(String, String)], forward_env: &[String]) -> Value {
    let mut map = serde_json::Map::new();
    for key in forward_env {
        if let Ok(value) = std::env::var(key) {
            map.insert(key.clone(), Value::String(value));
        }
    }
    for (key, value) in env {
        map.insert(key.clone(), Value::String(value.clone()));
    }
    Value::Object(map)
}

fn derive_contree_tag(image: &str) -> String {
    let mut tag = normalize_contree_image(image)
        .replace('/', "-")
        .replace(':', "-")
        .replace('.', "-");
    if tag.len() > 48 {
        tag.truncate(48);
    }
    format!("rust-mini-{}", tag)
}

fn normalize_contree_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

fn extract_contree_image_ref(operation: &Value) -> Option<String> {
    let result = &operation["result"];
    for key in ["image", "image_uuid", "imageUUID", "uuid"] {
        if let Some(value) = result[key].as_str() {
            if value.starts_with("tag:") {
                return Some(value.to_string());
            }
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn normalizes_contree_docker_prefix() {
        assert_eq!(
            normalize_contree_image("docker://docker.io/swebench/example:latest"),
            "docker.io/swebench/example:latest"
        );
        assert_eq!(
            normalize_contree_image("docker.io/swebench/example:latest"),
            "docker.io/swebench/example:latest"
        );
    }

    #[test]
    fn contree_config_maps_to_env_vars() {
        let mut config = BTreeMap::new();
        config.insert("base_url".to_string(), json!("https://example.invalid"));
        config.insert("token".to_string(), json!("secret"));
        config.insert("timeout".to_string(), json!(30));

        let env = contree_env_from_config(&config);

        assert!(env.contains(&(
            "CONTREE_BASE_URL".to_string(),
            "https://example.invalid".to_string()
        )));
        assert!(env.contains(&("CONTREE_TOKEN".to_string(), "secret".to_string())));
        assert!(env.contains(&("CONTREE_TIMEOUT".to_string(), "30".to_string())));
    }

    #[test]
    fn contree_shell_command_wraps_env_and_cwd() {
        let command = build_contree_shell_command(
            "/workspace",
            &[("A".to_string(), "1".to_string())],
            &[],
            "pytest -q",
        );

        assert!(command.contains("export A='1';"));
        assert!(command.contains("cd '/workspace' && pytest -q"));
    }

    #[test]
    fn normalizes_contree_base_url_with_v1_suffix() {
        assert_eq!(
            normalize_contree_base_url("https://contree.example"),
            "https://contree.example/v1"
        );
        assert_eq!(
            normalize_contree_base_url("https://contree.example/v1"),
            "https://contree.example/v1"
        );
    }

    #[test]
    fn derives_contree_tag_from_image_name() {
        let tag = derive_contree_tag("docker://docker.io/swebench/sweb.eval.x86_64.demo:latest");
        assert!(tag.starts_with("rust-mini-"));
        assert!(tag.contains("docker-io-swebench"));
        assert!(tag.len() <= "rust-mini-".len() + 48);
    }

    #[test]
    fn merged_env_prefers_explicit_env_over_forwarded() {
        let value = {
            let _guard = patch_env("FORWARDED_ONLY", Some("host"));
            let _guard2 = patch_env("CONFLICT", Some("host"));
            merged_env(
                &[("CONFLICT".to_string(), "config".to_string())],
                &["FORWARDED_ONLY".to_string(), "CONFLICT".to_string()],
            )
        };
        assert_eq!(value["FORWARDED_ONLY"], "host");
        assert_eq!(value["CONFLICT"], "config");
    }

    struct EnvGuard {
        key: String,
        old: Option<String>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(old) = &self.old {
                unsafe { std::env::set_var(&self.key, old) };
            } else {
                unsafe { std::env::remove_var(&self.key) };
            }
        }
    }

    fn patch_env(key: &str, value: Option<&str>) -> EnvGuard {
        let old = std::env::var(key).ok();
        match value {
            Some(value) => unsafe { std::env::set_var(key, value) },
            None => unsafe { std::env::remove_var(key) },
        }
        EnvGuard {
            key: key.to_string(),
            old,
        }
    }
}

impl Drop for DockerEnvironment {
    fn drop(&mut self) {
        if self.container_id.is_empty() {
            return;
        }
        let _ = Command::new(&self.executable)
            .args(["rm", "-f", &self.container_id])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

impl Drop for SingularityEnvironment {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.sandbox_dir);
    }
}
