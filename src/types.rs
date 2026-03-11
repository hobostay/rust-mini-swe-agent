use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub actions: Vec<Action>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    pub output: String,
    pub returncode: i32,
    pub exception_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryInfo {
    pub exit_status: String,
    pub submission: String,
    pub model_name: String,
    pub instance_cost: f64,
    pub api_calls: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub info: TrajectoryInfo,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub raw_responses: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkInstance {
    pub instance_id: String,
    pub problem_statement: String,
    #[serde(default)]
    pub image_name: Option<String>,
    #[serde(default)]
    pub docker_image: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub output_path: PathBuf,
}
