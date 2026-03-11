use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Result, bail};
use regex::Regex;
use serde_json::{Value, json};

use crate::config::{AgentMode, Config};
use crate::environment::Environment;
use crate::model::Model;
use crate::template::{
    render_agent_template, render_format_error_template, render_observation_template,
};
use crate::types::{Action, CommandOutput, Message, RuntimePaths, Trajectory, TrajectoryInfo};

const SUBMIT_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

pub struct AgentRuntime {
    pub config: Config,
    pub paths: RuntimePaths,
}

pub struct Agent {
    runtime: AgentRuntime,
    model: Box<dyn Model>,
    environment: Box<dyn Environment>,
    messages: Vec<Message>,
    raw_responses: Vec<Value>,
    cost: f64,
    n_calls: usize,
}

impl Agent {
    pub fn new(
        runtime: AgentRuntime,
        model: Box<dyn Model>,
        environment: Box<dyn Environment>,
    ) -> Self {
        Self {
            runtime,
            model,
            environment,
            messages: Vec::new(),
            raw_responses: Vec::new(),
            cost: 0.0,
            n_calls: 0,
        }
    }

    pub async fn run(&mut self, task: &str, instance_id: Option<String>) -> Result<Trajectory> {
        self.reset(task);
        let mut exit_status = "Incomplete".to_string();
        let mut submission = String::new();

        loop {
            if self.runtime.config.agent.step_limit > 0
                && self.n_calls >= self.runtime.config.agent.step_limit
            {
                exit_status = "LimitsExceeded".to_string();
                break;
            }
            if self.runtime.config.agent.cost_limit > 0.0
                && self.cost > self.runtime.config.agent.cost_limit
            {
                exit_status = "LimitsExceeded".to_string();
                break;
            }

            let message = self.query().await?;
            let outputs = self.execute_actions(&message.actions)?;
            for (action, output) in message.actions.iter().cloned().zip(outputs.into_iter()) {
                if is_submission(&output) {
                    if self.runtime.config.agent.confirm_exit && !self.confirm_submission()? {
                        self.messages.push(Message {
                            role: "user".to_string(),
                            content: "Submission rejected by user. Continue working.".to_string(),
                            tool_call_id: None,
                            actions: vec![],
                            cost: None,
                        });
                        continue;
                    }
                    exit_status = "Submitted".to_string();
                    submission = extract_submission(&output.output);
                }
                self.messages.push(observation_message(
                    action,
                    output,
                    &self.runtime.config.model.observation_template,
                )?);
            }
            self.save_partial()?;
            if exit_status == "Submitted" {
                break;
            }
        }

        let trajectory = Trajectory {
            info: TrajectoryInfo {
                exit_status,
                submission,
                model_name: self.model.model_name().to_string(),
                instance_cost: self.cost,
                api_calls: self.n_calls,
            },
            messages: self.messages.clone(),
            raw_responses: self.raw_responses.clone(),
            instance_id,
        };
        self.save_trajectory(&trajectory)?;
        Ok(trajectory)
    }

    fn reset(&mut self, task: &str) {
        self.messages.clear();
        self.raw_responses.clear();
        self.cost = 0.0;
        self.n_calls = 0;
        self.messages.push(Message {
            role: "system".to_string(),
            content: render_agent_template(
                &self.runtime.config.agent.system_template,
                task,
                self.model.model_name(),
                self.n_calls,
                self.cost,
            )
            .unwrap_or_else(|_| self.runtime.config.agent.system_template.clone()),
            tool_call_id: None,
            actions: vec![],
            cost: None,
        });
        self.messages.push(Message {
            role: "user".to_string(),
            content: render_agent_template(
                &self.runtime.config.agent.instance_template,
                task,
                self.model.model_name(),
                self.n_calls,
                self.cost,
            )
            .unwrap_or_else(|_| {
                self.runtime
                    .config
                    .agent
                    .instance_template
                    .replace("{{task}}", task)
            }),
            tool_call_id: None,
            actions: vec![],
            cost: None,
        });
    }

    async fn query(&mut self) -> Result<Message> {
        if matches!(self.runtime.config.agent.mode, AgentMode::Human) {
            let command = self.prompt_and_handle_commands("human mode command")?;
            let msg = Message {
                role: "user".to_string(),
                content: format!("User command:\n```bash\n{}\n```", command),
                tool_call_id: None,
                actions: vec![Action {
                    command,
                    tool_call_id: None,
                }],
                cost: None,
            };
            self.messages.push(msg.clone());
            return Ok(msg);
        }

        let mut format_error_prompted = false;
        loop {
            match self.model.query(&self.messages).await {
                Ok(response) => {
                    self.n_calls += 1;
                    self.cost += response.message.cost.unwrap_or(0.0);
                    self.raw_responses.push(response.raw_response);
                    self.messages.push(response.message.clone());
                    return Ok(response.message);
                }
                Err(error) if !format_error_prompted && is_format_error(&error.to_string()) => {
                    format_error_prompted = true;
                    self.messages.push(Message {
                        role: "user".to_string(),
                        content: render_format_error_template(
                            &self.runtime.config.model.format_error_template,
                            &error.to_string(),
                            self.model.model_name(),
                            self.n_calls,
                            self.cost,
                        )
                        .unwrap_or_else(|_| {
                            self.runtime.config.model.format_error_template.clone()
                        }),
                        tool_call_id: None,
                        actions: vec![],
                        cost: None,
                    });
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn execute_actions(&mut self, actions: &[Action]) -> Result<Vec<CommandOutput>> {
        if self.should_confirm(actions)? && !self.confirm_actions(actions.len())? {
            self.messages.push(Message {
                role: "user".to_string(),
                content: "Commands not executed. User rejected them.".to_string(),
                tool_call_id: None,
                actions: vec![],
                cost: None,
            });
            return Ok(actions
                .iter()
                .map(|_| CommandOutput {
                    output: String::new(),
                    returncode: -1,
                    exception_info: "action was not executed".to_string(),
                })
                .collect());
        }

        let mut outputs = Vec::with_capacity(actions.len());
        for action in actions {
            outputs.push(self.environment.execute(action)?);
        }
        Ok(outputs)
    }

    fn should_confirm(&self, actions: &[Action]) -> Result<bool> {
        if !matches!(self.runtime.config.agent.mode, AgentMode::Confirm) {
            return Ok(false);
        }
        let regexes: Result<Vec<Regex>, _> = self
            .runtime
            .config
            .agent
            .whitelist_actions
            .iter()
            .map(|p| Regex::new(p))
            .collect();
        let regexes = regexes?;
        Ok(actions
            .iter()
            .any(|a| !regexes.iter().any(|re| re.is_match(&a.command))))
    }

    fn confirm_actions(&mut self, n_actions: usize) -> Result<bool> {
        loop {
            let input = self.prompt_raw(&format!(
                "Execute {n_actions} action(s)? [Enter=yes, /y, /c, /u, /h, text=no]"
            ))?;
            match input.trim() {
                "" => return Ok(true),
                "/y" => {
                    self.runtime.config.agent.mode = AgentMode::Yolo;
                    return Ok(true);
                }
                "/c" => {
                    self.runtime.config.agent.mode = AgentMode::Confirm;
                    return Ok(true);
                }
                "/u" => {
                    self.runtime.config.agent.mode = AgentMode::Human;
                    return Ok(false);
                }
                "/h" => {
                    self.print_help();
                }
                _ => return Ok(false),
            }
        }
    }

    fn confirm_submission(&mut self) -> Result<bool> {
        loop {
            let input = self.prompt_raw(
                "Agent wants to finish. Accept submission? [Enter=yes, /u continue, text=no]",
            )?;
            match input.trim() {
                "" => return Ok(true),
                "/u" => {
                    self.runtime.config.agent.mode = AgentMode::Human;
                    return Ok(false);
                }
                "/h" => self.print_help(),
                _ => return Ok(false),
            }
        }
    }

    fn prompt_and_handle_commands(&mut self, prompt: &str) -> Result<String> {
        loop {
            let input = self.prompt_raw(prompt)?;
            match input.trim() {
                "/y" => {
                    self.runtime.config.agent.mode = AgentMode::Yolo;
                    continue;
                }
                "/c" => {
                    self.runtime.config.agent.mode = AgentMode::Confirm;
                    continue;
                }
                "/u" => {
                    self.runtime.config.agent.mode = AgentMode::Human;
                    continue;
                }
                "/h" => {
                    self.print_help();
                    continue;
                }
                other => return Ok(other.to_string()),
            }
        }
    }

    fn prompt_raw(&self, prompt: &str) -> Result<String> {
        print!("{prompt}: ");
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        Ok(buf.trim_end().to_string())
    }

    fn print_help(&self) {
        println!(
            "Current mode: {:?}\n/y -> yolo\n/c -> confirm\n/u -> human\n/h -> help",
            self.runtime.config.agent.mode
        );
    }

    fn save_partial(&self) -> Result<()> {
        let value = json!({
            "info": {
                "exit_status": "",
                "submission": "",
                "model_name": self.model.model_name(),
                "instance_cost": self.cost,
                "api_calls": self.n_calls,
            },
            "messages": self.messages,
            "raw_responses": self.raw_responses,
        });
        if let Some(parent) = self.runtime.paths.output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &self.runtime.paths.output_path,
            serde_json::to_string_pretty(&value)?,
        )?;
        Ok(())
    }

    fn save_trajectory(&self, trajectory: &Trajectory) -> Result<()> {
        if let Some(parent) = self.runtime.paths.output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &self.runtime.paths.output_path,
            serde_json::to_string_pretty(trajectory)?,
        )?;
        Ok(())
    }
}

fn is_format_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("no tool calls")
        || lower.contains("no parseable")
        || lower.contains("unknown tool call")
        || lower.contains("invalid tool arguments")
        || lower.contains("missing command in tool arguments")
        || lower.contains("function_call arguments")
}

fn observation_message(
    action: Action,
    output: CommandOutput,
    observation_template: &str,
) -> Result<Message> {
    let content = render_observation_template(observation_template, &output).unwrap_or_else(|_| {
        json!({
            "returncode": output.returncode,
            "output": output.output,
            "exception_info": output.exception_info,
        })
        .to_string()
    });
    Ok(Message {
        role: if action.tool_call_id.is_some() {
            "tool".to_string()
        } else {
            "user".to_string()
        },
        content,
        tool_call_id: action.tool_call_id,
        actions: vec![],
        cost: None,
    })
}

fn is_submission(output: &CommandOutput) -> bool {
    output.returncode == 0
        && output
            .output
            .trim_start()
            .lines()
            .next()
            .map(|line| line.trim() == SUBMIT_SENTINEL)
            .unwrap_or(false)
}

fn extract_submission(output: &str) -> String {
    let mut lines = output.trim_start().lines();
    if lines.next().map(|line| line.trim()) != Some(SUBMIT_SENTINEL) {
        return String::new();
    }
    lines.collect::<Vec<_>>().join("\n")
}

pub fn print_trajectory_summary(trajectory: &Trajectory) -> Result<()> {
    if trajectory.info.exit_status.is_empty() {
        bail!("trajectory exit status was empty");
    }
    println!(
        "exit_status={} api_calls={} cost={:.4}",
        trajectory.info.exit_status, trajectory.info.api_calls, trajectory.info.instance_cost
    );
    if !trajectory.info.submission.is_empty() {
        println!("submission:\n{}", trajectory.info.submission);
    }
    Ok(())
}

pub fn build_runtime(config: Config, output_path: PathBuf) -> AgentRuntime {
    AgentRuntime {
        config,
        paths: RuntimePaths { output_path },
    }
}
