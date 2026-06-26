//! OpenAI wire mapping: parse `/v1/chat/completions` requests into a [`GenReq`]
//! and format `chat.completion` / `chat.completion.chunk` JSON. Pure functions
//! over `serde_json::Value` (no framework).

use ds4_engine::tokenizer::ThinkMode;
use crate::backend::sampling::SampleParams;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gen::GenReq;
use crate::tools_dsml::ToolCall;

/// OpenAI streaming `delta.tool_calls` array (one delta carrying the full calls).
pub fn tool_calls_delta(calls: &[ToolCall]) -> Value {
    let arr: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "index": i, "id": c.id, "type": "function",
                "function": { "name": c.name, "arguments": c.arguments }
            })
        })
        .collect();
    json!({ "tool_calls": arr })
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn gen_id() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("chatcmpl-{n:x}")
}

/// Plain text of an OpenAI `content` (string, or array of `{type:"text",text}`).
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn stop_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    }
}

fn think_mode(req: &Value) -> ThinkMode {
    match req.get("reasoning_effort").and_then(|v| v.as_str()) {
        Some("none") | Some("minimal") => ThinkMode::None,
        Some("max") | Some("xmax") => ThinkMode::Max,
        _ => ThinkMode::High,
    }
}

/// Parse a chat-completions request body. Returns `(GenReq, stream?)`.
pub fn parse_chat_request(req: &Value, ctx_size: u32) -> Result<(GenReq, bool), String> {
    let messages: Vec<(String, String)> = req
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("missing `messages` array")?
        .iter()
        .map(|m| {
            (
                m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string(),
                m.get("content").map(content_text).unwrap_or_default(),
            )
        })
        .collect();
    if messages.is_empty() {
        return Err("empty `messages`".into());
    }

    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let think = think_mode(req);

    // When thinking, mirror ds4_server.c:10102 sampling overrides.
    let params = if think.enabled() {
        SampleParams { temperature: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0 }
    } else {
        SampleParams {
            temperature: req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            top_k: req.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            top_p: req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            min_p: req.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        }
    };

    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });

    let requested = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as u32;
    let max_tokens = requested.min(ctx_size.saturating_sub(64)).max(1);

    let stop = stop_list(req.get("stop"));
    let tools = req.get("tools").and_then(|t| t.as_array()).filter(|a| !a.is_empty()).map(|arr| {
        arr.iter()
            .filter_map(|t| {
                let f = t.get("function").unwrap_or(t);
                let name = f.get("name")?.as_str()?;
                let desc = f.get("description").and_then(|d| d.as_str()).unwrap_or("");
                let params = f.get("parameters").map(|p| p.to_string()).unwrap_or_else(|| "{}".into());
                Some(format!("### {name}\n{desc}\nParameters (JSON Schema): {params}\n"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    });

    Ok((GenReq { messages, params, seed, max_tokens, think, stop, tools }, stream))
}

/// One `chat.completion.chunk` SSE payload.
pub fn chunk_json(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
    .to_string()
}

/// The non-streaming `chat.completion` response.
#[allow(clippy::too_many_arguments)]
pub fn final_completion_json(
    id: &str,
    created: u64,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCall],
    finish: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    // OpenAI: content is null when only tool_calls are present.
    let mut message = if tool_calls.is_empty() {
        json!({"role": "assistant", "content": content})
    } else {
        json!({"role": "assistant", "content": Value::Null})
    };
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id, "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments }
                })
            })
            .collect();
    }
    json!({
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
    .to_string()
}

/// Parse a legacy `/v1/completions` request (raw text prompt, no thinking).
pub fn parse_completion_request(req: &Value, ctx: u32) -> Result<(GenReq, bool), String> {
    let prompt = match req.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(""),
        _ => return Err("missing `prompt`".into()),
    };
    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let params = SampleParams {
        temperature: req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        top_k: req.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        top_p: req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        min_p: req.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    };
    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    let max_tokens = (req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(256))
        .min(ctx.saturating_sub(64) as u64)
        .max(1) as u32;
    let stop = stop_list(req.get("stop"));
    Ok((
        GenReq {
            messages: vec![("user".to_string(), prompt)],
            params,
            seed,
            max_tokens,
            think: ThinkMode::None,
            stop,
            tools: None,
        },
        stream,
    ))
}

pub fn text_completion_chunk(id: &str, created: u64, model: &str, text: &str, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": "text_completion", "created": created, "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": finish}]
    })
    .to_string()
}

pub fn final_text_completion(
    id: &str, created: u64, model: &str, text: &str, finish: &str, prompt_tokens: usize, completion_tokens: usize,
) -> String {
    json!({
        "id": id, "object": "text_completion", "created": created, "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}
    })
    .to_string()
}

pub fn models_json(model_id: &str) -> String {
    json!({
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "ds4"}]
    })
    .to_string()
}

pub fn model_json(model_id: &str) -> String {
    json!({"id": model_id, "object": "model", "created": 0, "owned_by": "ds4"}).to_string()
}
