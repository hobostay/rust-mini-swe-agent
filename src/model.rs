use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::types::{Action, Message};

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub message: Message,
    pub raw_response: Value,
}

#[async_trait]
pub trait Model: Send + Sync {
    fn model_name(&self) -> &str;
    async fn query(&self, messages: &[Message]) -> Result<ModelResponse>;
}

#[derive(Clone, Copy)]
pub enum ApiStyle {
    ChatCompletions,
    Responses,
}

pub struct ApiModel {
    client: Client,
    model_name: String,
    endpoint: String,
    headers: HeaderMap,
    temperature: f64,
    api_style: ApiStyle,
    multimodal_regex: Option<Regex>,
    cost_tracking: String,
    set_cache_control: Option<String>,
    provider: Option<String>,
    cost_model_override: Option<String>,
    is_portkey: bool,
}

impl ApiModel {
    pub fn openai_compatible(
        model_name: String,
        base_url: String,
        api_key: String,
        temperature: f64,
        api_style: ApiStyle,
        multimodal_regex: String,
        cost_tracking: String,
        set_cache_control: Option<String>,
    ) -> Result<Self> {
        let normalized_cache_control = normalize_cache_control(&model_name, set_cache_control);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("invalid authorization header")?,
        );
        let endpoint = match api_style {
            ApiStyle::ChatCompletions => {
                format!("{}/chat/completions", base_url.trim_end_matches('/'))
            }
            ApiStyle::Responses => format!("{}/responses", base_url.trim_end_matches('/')),
        };
        Ok(Self {
            client: Client::new(),
            model_name,
            endpoint,
            headers,
            temperature,
            api_style,
            multimodal_regex: compile_optional_regex(&multimodal_regex)?,
            cost_tracking,
            set_cache_control: normalized_cache_control,
            provider: None,
            cost_model_override: None,
            is_portkey: false,
        })
    }

    pub fn openrouter(
        model_name: String,
        api_key: String,
        temperature: f64,
        responses: bool,
        multimodal_regex: String,
        cost_tracking: String,
        set_cache_control: Option<String>,
    ) -> Result<Self> {
        let normalized_cache_control = normalize_cache_control(&model_name, set_cache_control);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("invalid authorization header")?,
        );
        let endpoint = if responses {
            "https://openrouter.ai/api/v1/responses"
        } else {
            "https://openrouter.ai/api/v1/chat/completions"
        };
        Ok(Self {
            client: Client::new(),
            model_name,
            endpoint: endpoint.to_string(),
            headers,
            temperature,
            api_style: if responses {
                ApiStyle::Responses
            } else {
                ApiStyle::ChatCompletions
            },
            multimodal_regex: compile_optional_regex(&multimodal_regex)?,
            cost_tracking,
            set_cache_control: normalized_cache_control,
            provider: None,
            cost_model_override: None,
            is_portkey: false,
        })
    }

    pub fn requesty(
        model_name: String,
        api_key: String,
        temperature: f64,
        multimodal_regex: String,
        cost_tracking: String,
        set_cache_control: Option<String>,
    ) -> Result<Self> {
        let normalized_cache_control = normalize_cache_control(&model_name, set_cache_control);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("invalid authorization header")?,
        );
        headers.insert(
            HeaderName::from_static("http-referer"),
            HeaderValue::from_static("https://github.com/SWE-agent/mini-swe-agent"),
        );
        headers.insert(
            HeaderName::from_static("x-title"),
            HeaderValue::from_static("rust-mini-swe-agent"),
        );
        Ok(Self {
            client: Client::new(),
            model_name,
            endpoint: "https://router.requesty.ai/v1/chat/completions".to_string(),
            headers,
            temperature,
            api_style: ApiStyle::ChatCompletions,
            multimodal_regex: compile_optional_regex(&multimodal_regex)?,
            cost_tracking,
            set_cache_control: normalized_cache_control,
            provider: None,
            cost_model_override: None,
            is_portkey: false,
        })
    }

    pub fn portkey(
        model_name: String,
        api_key: String,
        temperature: f64,
        responses: bool,
        provider: Option<String>,
        cost_model_override: Option<String>,
        multimodal_regex: String,
        cost_tracking: String,
        set_cache_control: Option<String>,
    ) -> Result<Self> {
        let normalized_cache_control = normalize_cache_control(&model_name, set_cache_control);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            HeaderName::from_static("x-portkey-api-key"),
            HeaderValue::from_str(&api_key).context("invalid portkey api key header")?,
        );
        if let Ok(trace_id) = std::env::var("PORTKEY_TRACE_ID") {
            headers.insert(
                HeaderName::from_static("x-portkey-trace-id"),
                HeaderValue::from_str(&trace_id).context("invalid portkey trace id header")?,
            );
        }
        let virtual_key = std::env::var("PORTKEY_VIRTUAL_KEY").ok();
        if let Some(ref virtual_key) = virtual_key {
            headers.insert(
                HeaderName::from_static("x-portkey-virtual-key"),
                HeaderValue::from_str(&virtual_key)
                    .context("invalid portkey virtual key header")?,
            );
        }
        let provider = if virtual_key.is_none() {
            provider
                .or_else(|| std::env::var("PORTKEY_PROVIDER").ok())
                .or_else(|| infer_provider_from_model_name(&model_name))
        } else {
            None
        };
        let endpoint = if responses {
            "https://api.portkey.ai/v1/responses"
        } else {
            "https://api.portkey.ai/v1/chat/completions"
        };
        Ok(Self {
            client: Client::new(),
            model_name,
            endpoint: endpoint.to_string(),
            headers,
            temperature,
            api_style: if responses {
                ApiStyle::Responses
            } else {
                ApiStyle::ChatCompletions
            },
            multimodal_regex: compile_optional_regex(&multimodal_regex)?,
            cost_tracking,
            set_cache_control: normalized_cache_control,
            provider,
            cost_model_override,
            is_portkey: true,
        })
    }
}

#[async_trait]
impl Model for ApiModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn query(&self, messages: &[Message]) -> Result<ModelResponse> {
        let payload = build_payload(
            &self.model_name,
            messages,
            self.temperature,
            self.api_style,
            self.multimodal_regex.as_ref(),
            self.set_cache_control.as_deref(),
            self.provider.as_deref(),
            self.is_portkey,
        );
        let response = self
            .client
            .post(&self.endpoint)
            .headers(self.headers.clone())
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("model request failed for {}", self.endpoint))?;

        let status = response.status();
        let raw: Value = response
            .json()
            .await
            .context("invalid model json response")?;
        if !status.is_success() {
            bail!("model request failed with {}: {}", status, raw);
        }

        match self.api_style {
            ApiStyle::ChatCompletions => parse_chat_response(
                raw,
                &self.cost_tracking,
                self.cost_model_override.as_deref(),
                self.is_portkey,
            ),
            ApiStyle::Responses => parse_responses_response(
                raw,
                &self.cost_tracking,
                self.cost_model_override.as_deref(),
                self.is_portkey,
            ),
        }
    }
}

fn build_payload(
    model_name: &str,
    messages: &[Message],
    temperature: f64,
    api_style: ApiStyle,
    multimodal_regex: Option<&Regex>,
    set_cache_control: Option<&str>,
    provider: Option<&str>,
    is_portkey: bool,
) -> Value {
    let mut prepared_messages: Vec<Value> = messages
        .iter()
        .map(|m| match api_style {
            ApiStyle::ChatCompletions => {
                let mut value = json!({
                    "role": m.role,
                    "content": content_value(&m.content, api_style, multimodal_regex),
                });
                if let Some(tool_call_id) = &m.tool_call_id {
                    value["tool_call_id"] = Value::String(tool_call_id.clone());
                }
                value
            }
            ApiStyle::Responses => json!({
                "type": "message",
                "role": m.role,
                "content": content_value(&m.content, api_style, multimodal_regex),
            }),
        })
        .collect();
    reorder_anthropic_assistant_messages(&mut prepared_messages, api_style);
    apply_cache_control(&mut prepared_messages, api_style, set_cache_control);

    let mut payload = match api_style {
        ApiStyle::ChatCompletions => json!({
            "model": model_name,
            "messages": prepared_messages,
            "temperature": temperature,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }]
        }),
        ApiStyle::Responses => json!({
            "model": model_name,
            "input": prepared_messages,
            "temperature": temperature,
            "tools": [{
                "type": "function",
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }]
        }),
    };
    if let Some(provider) = provider {
        payload["provider"] = Value::String(provider.to_string());
    }
    if is_portkey {
        if let Ok(metadata) = std::env::var("PORTKEY_METADATA")
            && let Ok(value) = serde_json::from_str::<Value>(&metadata)
        {
            payload["metadata"] = value;
        }
        if let Ok(cache_namespace) = std::env::var("PORTKEY_CACHE_NAMESPACE") {
            payload["cache_namespace"] = Value::String(cache_namespace);
        }
    }
    payload
}

fn parse_chat_response(
    raw: Value,
    cost_tracking: &str,
    cost_model_override: Option<&str>,
    is_portkey: bool,
) -> Result<ModelResponse> {
    let choice = raw["choices"]
        .get(0)
        .context("missing first choice in model response")?;
    let assistant = &choice["message"];
    let content = extract_text_content(&assistant["content"]);
    let actions = parse_chat_actions(assistant)?;
    let cost = extract_cost(&raw, cost_model_override, is_portkey)?;
    if cost.unwrap_or(0.0) <= 0.0 && cost_tracking != "ignore_errors" {
        bail!("missing cost information in chat completion response");
    }
    Ok(ModelResponse {
        message: Message {
            role: "assistant".to_string(),
            content,
            tool_call_id: None,
            actions,
            cost,
        },
        raw_response: raw,
    })
}

fn parse_responses_response(
    raw: Value,
    cost_tracking: &str,
    cost_model_override: Option<&str>,
    is_portkey: bool,
) -> Result<ModelResponse> {
    let output = raw["output"]
        .as_array()
        .context("responses output missing")?;
    let mut text_parts = Vec::new();
    let mut actions = Vec::new();
    for item in output {
        match item["type"].as_str().unwrap_or_default() {
            "message" => {
                if let Some(contents) = item["content"].as_array() {
                    for c in contents {
                        if let Some(text) = c["text"].as_str() {
                            text_parts.push(text.to_string());
                        }
                    }
                }
            }
            "function_call" => {
                let name = item["name"].as_str().unwrap_or_default();
                if name != "bash" {
                    bail!("unknown function call in responses API: {}", name);
                }
                let arguments = item["arguments"]
                    .as_str()
                    .context("responses function_call arguments missing")?;
                let parsed: Value = serde_json::from_str(arguments)
                    .with_context(|| format!("invalid tool arguments: {arguments}"))?;
                let command = parsed["command"]
                    .as_str()
                    .context("missing command in responses arguments")?
                    .to_string();
                actions.push(Action {
                    command,
                    tool_call_id: item["call_id"].as_str().map(ToOwned::to_owned),
                });
            }
            _ => {}
        }
    }
    if actions.is_empty()
        && let Some(command) = parse_text_command(&text_parts.join("\n\n"))
    {
        actions.push(Action {
            command,
            tool_call_id: None,
        });
    }
    if actions.is_empty() {
        bail!("responses API response contained no parseable actions");
    }
    let cost = extract_cost(&raw, cost_model_override, is_portkey)?;
    if cost.unwrap_or(0.0) <= 0.0 && cost_tracking != "ignore_errors" {
        bail!("missing cost information in responses api response");
    }
    Ok(ModelResponse {
        message: Message {
            role: "assistant".to_string(),
            content: text_parts.join("\n\n"),
            tool_call_id: None,
            actions,
            cost,
        },
        raw_response: raw,
    })
}

fn parse_chat_actions(assistant: &Value) -> Result<Vec<Action>> {
    if let Some(tool_calls) = assistant["tool_calls"].as_array()
        && !tool_calls.is_empty()
    {
        let mut actions = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            let function = &call["function"];
            let name = function["name"].as_str().unwrap_or_default();
            if name != "bash" {
                bail!("unknown tool call: {}", name);
            }
            let arguments = function["arguments"]
                .as_str()
                .context("tool call arguments were not a string")?;
            let parsed: Value = serde_json::from_str(arguments)
                .with_context(|| format!("invalid tool arguments: {arguments}"))?;
            let command = parsed["command"]
                .as_str()
                .context("missing command in tool arguments")?
                .to_string();
            actions.push(Action {
                command,
                tool_call_id: call["id"].as_str().map(ToOwned::to_owned),
            });
        }
        return Ok(actions);
    }
    let content = extract_text_content(&assistant["content"]);
    if let Some(command) = parse_text_command(&content) {
        return Ok(vec![Action {
            command,
            tool_call_id: None,
        }]);
    }
    bail!("model response contained no tool calls and no parseable bash code block")
}

fn parse_text_command(content: &str) -> Option<String> {
    let patterns = [
        r"```mswea_bash_command\s*(?P<cmd>[\s\S]*?)```",
        r"```bash\s*(?P<cmd>[\s\S]*?)```",
        r"```\s*(?P<cmd>[\s\S]*?)```",
    ];
    for pattern in patterns {
        let regex = Regex::new(pattern).ok()?;
        if let Some(captures) = regex.captures(content) {
            let command = captures.name("cmd")?.as_str().trim().to_string();
            if !command.is_empty() {
                return Some(command);
            }
        }
    }
    None
}

fn compile_optional_regex(pattern: &str) -> Result<Option<Regex>> {
    if pattern.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(Regex::new(pattern).with_context(|| {
        format!("invalid multimodal_regex: {pattern}")
    })?))
}

fn content_value(content: &str, api_style: ApiStyle, multimodal_regex: Option<&Regex>) -> Value {
    if let Some(regex) = multimodal_regex {
        let mut items = Vec::new();
        let mut last = 0usize;
        for captures in regex.captures_iter(content) {
            if let Some(m) = captures.get(0) {
                if m.start() > last {
                    items.push(text_part(&content[last..m.start()], api_style));
                }
                items.push(image_part(m.as_str(), api_style));
                last = m.end();
            }
        }
        if !items.is_empty() {
            if last < content.len() {
                items.push(text_part(&content[last..], api_style));
            }
            return Value::Array(items);
        }
    }
    match api_style {
        ApiStyle::ChatCompletions => Value::String(content.to_string()),
        ApiStyle::Responses => Value::Array(vec![text_part(content, api_style)]),
    }
}

fn text_part(text: &str, api_style: ApiStyle) -> Value {
    match api_style {
        ApiStyle::ChatCompletions => json!({"type": "text", "text": text}),
        ApiStyle::Responses => json!({"type": "input_text", "text": text}),
    }
}

fn image_part(path: &str, api_style: ApiStyle) -> Value {
    let url = if path.starts_with("http://") || path.starts_with("https://") {
        path.to_string()
    } else {
        format!("file://{path}")
    };
    match api_style {
        ApiStyle::ChatCompletions => json!({"type": "image_url", "image_url": {"url": url}}),
        ApiStyle::Responses => json!({"type": "input_image", "image_url": url}),
    }
}

fn reorder_anthropic_assistant_messages(messages: &mut [Value], api_style: ApiStyle) {
    if !matches!(api_style, ApiStyle::ChatCompletions) {
        return;
    }
    for message in messages.iter_mut() {
        if message["role"].as_str().unwrap_or_default() != "assistant" {
            continue;
        }
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        let Some(items) = content.as_array_mut() else {
            continue;
        };
        let mut thinking = Vec::new();
        let mut other = Vec::new();
        for item in items.drain(..) {
            if is_anthropic_thinking_block(&item) {
                thinking.push(item);
            } else {
                other.push(item);
            }
        }
        if thinking.is_empty() {
            *items = other;
            continue;
        }
        if other.is_empty() {
            other.push(json!({"type": "text", "text": ""}));
        }
        thinking.extend(other);
        *items = thinking;
    }
}

fn apply_cache_control(
    messages: &mut [Value],
    api_style: ApiStyle,
    set_cache_control: Option<&str>,
) {
    if set_cache_control != Some("default_end") {
        return;
    }
    for message in messages.iter_mut() {
        clear_cache_control_marker(message);
    }
    let Some(last) = messages.last_mut() else {
        return;
    };
    match api_style {
        ApiStyle::ChatCompletions | ApiStyle::Responses => {
            set_cache_control_marker(last);
        }
    }
}

fn extract_text_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("content").and_then(Value::as_str))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn normalize_cache_control(model_name: &str, set_cache_control: Option<String>) -> Option<String> {
    if set_cache_control.is_some() {
        return set_cache_control;
    }
    let lower = model_name.to_ascii_lowercase();
    if ["anthropic", "claude", "sonnet", "opus"]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        Some("default_end".to_string())
    } else {
        None
    }
}

fn extract_cost(
    raw: &Value,
    cost_model_override: Option<&str>,
    is_portkey: bool,
) -> Result<Option<f64>> {
    if let Some(cost) = raw["usage"]["cost"]
        .as_f64()
        .or_else(|| raw["usage"]["total_cost"].as_f64())
    {
        return Ok(Some(cost));
    }
    if !is_portkey {
        return Ok(None);
    }
    if let Some(cost) = raw["cost"].as_f64() {
        return Ok(Some(cost));
    }
    if let Some(costs) = raw["usage"]["costs"].as_array() {
        let total = costs
            .iter()
            .filter_map(|entry| entry["amount"].as_f64())
            .sum::<f64>();
        if total > 0.0 {
            return Ok(Some(total));
        }
    }
    if let Some(model_override) = cost_model_override
        && let Some(tokens) = extract_token_counts(raw)
        && let Some(cost) =
            approximate_known_model_cost(model_override, tokens.0, tokens.1, tokens.2)
    {
        return Ok(Some(cost));
    }
    Ok(None)
}

fn extract_token_counts(raw: &Value) -> Option<(u64, u64, u64)> {
    let usage = &raw["usage"];
    let total = usage["total_tokens"].as_u64()?;
    let mut prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let completion = usage["completion_tokens"].as_u64().unwrap_or(0);
    if total >= completion && total.saturating_sub(prompt).saturating_sub(completion) != 0 {
        prompt = total.saturating_sub(completion);
    }
    Some((prompt, completion, total))
}

fn approximate_known_model_cost(
    model_name: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    _total_tokens: u64,
) -> Option<f64> {
    let lower = model_name.to_ascii_lowercase();
    let (input_per_million, output_per_million) = if lower.contains("gpt-4.1-mini") {
        (0.40, 1.60)
    } else if lower.contains("gpt-4.1") {
        (2.00, 8.00)
    } else if lower.contains("claude-3-5-sonnet") || lower.contains("claude-sonnet-4") {
        (3.00, 15.00)
    } else if lower.contains("claude-3-7-sonnet") {
        (3.00, 15.00)
    } else {
        return None;
    };
    Some(
        (prompt_tokens as f64 / 1_000_000.0) * input_per_million
            + (completion_tokens as f64 / 1_000_000.0) * output_per_million,
    )
}

fn infer_provider_from_model_name(model_name: &str) -> Option<String> {
    let provider = model_name.split('/').next()?.trim();
    if provider.is_empty() || provider == model_name {
        return None;
    }
    Some(provider.to_string())
}

fn is_anthropic_thinking_block(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    )
}

fn clear_cache_control_marker(message: &mut Value) {
    if let Some(obj) = message.as_object_mut() {
        obj.remove("cache_control");
    }
    if let Some(content) = message.get_mut("content")
        && let Some(items) = content.as_array_mut()
    {
        for item in items {
            if let Some(obj) = item.as_object_mut() {
                obj.remove("cache_control");
            }
        }
    }
}

fn set_cache_control_marker(message: &mut Value) {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let Some(content) = message.get_mut("content") else {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
        }
        return;
    };
    match content {
        Value::Null => {
            if let Some(obj) = message.as_object_mut() {
                obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            }
        }
        Value::String(text) => {
            *content = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
        }
        Value::Array(items) => {
            if items.is_empty() {
                items.push(json!({
                    "type": "text",
                    "text": "",
                    "cache_control": {"type": "ephemeral"}
                }));
            } else if let Some(first) = items.first_mut()
                && let Some(obj) = first.as_object_mut()
            {
                obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            }
        }
        _ => {}
    }
    if role == "tool"
        && let Some(obj) = message.as_object_mut()
    {
        if let Some(content) = obj.get_mut("content")
            && let Some(items) = content.as_array_mut()
            && let Some(first) = items.first_mut()
            && let Some(item_obj) = first.as_object_mut()
        {
            item_obj.remove("cache_control");
        }
        obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
    }
}
