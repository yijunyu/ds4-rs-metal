//! OpenAI Responses API `/v1/responses` (for Codex). Functional core: parse the
//! request into a [`GenReq`], stream `response.created` →
//! `response.output_text.delta` → `response.completed` (event names from
//! ds4_server.c ~5996/6139/6154), and a non-streaming `response` object.

use serde_json::{json, Value};
use std::io::Result as IoResult;
use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

use ds4_engine::tokenizer::ThinkMode;
use crate::backend::sampling::SampleParams;

use crate::gen::{GenEvent, GenReq};
use crate::http;
use crate::tools_dsml::ToolCall;

pub fn gen_id() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("resp_{n:x}")
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Extract text from a Responses `input` (string, or array of items with
/// `content` arrays of `{type, text}` parts).
fn item_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|it| {
                if let Some(t) = it.get("text").and_then(|t| t.as_str()) {
                    t.to_string()
                } else if let Some(c) = it.get("content") {
                    item_text(c)
                } else {
                    String::new()
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn think_mode(req: &Value) -> ThinkMode {
    match req.get("reasoning").and_then(|r| r.get("effort")).and_then(|s| s.as_str()) {
        Some("none") | Some("minimal") => ThinkMode::None,
        Some("high") => ThinkMode::Max,
        _ => ThinkMode::High,
    }
}

/// Parse a Responses request into a [`GenReq`] + `stream?`.
pub fn parse_responses_request(req: &Value, ctx: u32) -> Result<(GenReq, bool), String> {
    let mut messages: Vec<(String, String)> = Vec::new();
    if let Some(instr) = req.get("instructions").and_then(|i| i.as_str()) {
        if !instr.is_empty() {
            messages.push(("system".to_string(), instr.to_string()));
        }
    }
    match req.get("input") {
        Some(Value::String(s)) => messages.push(("user".to_string(), s.clone())),
        Some(arr @ Value::Array(_)) => {
            // Items may carry roles; fall back to a single user turn.
            if let Value::Array(items) = arr {
                let mut any_role = false;
                for it in items {
                    if let Some(role) = it.get("role").and_then(|r| r.as_str()) {
                        any_role = true;
                        messages.push((role.to_string(), item_text(it.get("content").unwrap_or(it))));
                    }
                }
                if !any_role {
                    messages.push(("user".to_string(), item_text(arr)));
                }
            }
        }
        _ => return Err("missing `input`".into()),
    }
    if messages.is_empty() {
        return Err("empty `input`".into());
    }

    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let think = think_mode(req);
    let params = if think.enabled() {
        SampleParams { temperature: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0 }
    } else {
        SampleParams {
            temperature: req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            top_k: 0,
            top_p: req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            min_p: 0.0,
        }
    };
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    let requested = req.get("max_output_tokens").and_then(|v| v.as_u64()).unwrap_or(4096) as u32;
    let max_tokens = requested.min(ctx.saturating_sub(64)).max(1);
    let tools = req.get("tools").and_then(|t| t.as_array()).filter(|a| !a.is_empty()).map(|arr| {
        arr.iter()
            .filter_map(|t| {
                let name = t.get("name").or_else(|| t.get("function").and_then(|f| f.get("name")))?.as_str()?;
                let desc = t.get("description").and_then(|d| d.as_str()).unwrap_or("");
                Some(format!("### {name}\n{desc}\n"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    });

    Ok((GenReq { messages, params, seed, max_tokens, think, stop: Vec::new(), tools }, stream))
}

/// Streaming state for the Responses API.
pub struct RespSink {
    id: String,
    model: String,
    created: u64,
    started: bool,
    text_started: bool,
}

impl RespSink {
    pub fn new(model: &str) -> Self {
        Self { id: gen_id(), model: model.to_string(), created: now_secs(), started: false, text_started: false }
    }

    fn ensure_created(&mut self, s: &mut TcpStream) -> IoResult<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let data = json!({
            "type": "response.created",
            "response": {"id": self.id, "object": "response", "created_at": self.created,
                         "model": self.model, "status": "in_progress", "output": []}
        });
        http::write_sse_event(s, "response.created", &data.to_string())
    }

    pub fn on(&mut self, s: &mut TcpStream, ev: GenEvent) -> IoResult<()> {
        self.ensure_created(s)?;
        match ev {
            // Reasoning is surfaced as reasoning-summary text deltas.
            GenEvent::Reasoning(t) => http::write_sse_event(
                s,
                "response.reasoning_summary_text.delta",
                &json!({"type": "response.reasoning_summary_text.delta", "delta": t}).to_string(),
            ),
            GenEvent::Content(t) => {
                self.text_started = true;
                http::write_sse_event(
                    s,
                    "response.output_text.delta",
                    &json!({"type": "response.output_text.delta", "delta": t}).to_string(),
                )
            }
            GenEvent::ToolCalls(calls) => {
                for c in &calls {
                    http::write_sse_event(
                        s,
                        "response.output_item.added",
                        &json!({"type": "response.output_item.added",
                                "item": {"type": "function_call", "call_id": c.id, "name": c.name, "arguments": c.arguments}})
                        .to_string(),
                    )?;
                }
                Ok(())
            }
            GenEvent::Done { finish_reason, prompt_tokens, completion_tokens } => {
                if self.text_started {
                    http::write_sse_event(
                        s,
                        "response.output_text.done",
                        &json!({"type": "response.output_text.done"}).to_string(),
                    )?;
                }
                let data = json!({
                    "type": "response.completed",
                    "response": {"id": self.id, "object": "response", "created_at": self.created,
                                 "model": self.model, "status": "completed",
                                 "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens,
                                           "total_tokens": prompt_tokens + completion_tokens},
                                 "_finish": finish_reason}
                });
                http::write_sse_event(s, "response.completed", &data.to_string())
            }
            GenEvent::Error(e) => http::write_sse_event(
                s,
                "error",
                &json!({"type": "error", "message": e}).to_string(),
            ),
        }
    }
}

/// Non-streaming Responses `response` object.
#[allow(clippy::too_many_arguments)]
pub fn final_response_json(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCall],
    input_tokens: usize,
    output_tokens: usize,
) -> String {
    let mut output: Vec<Value> = Vec::new();
    if !reasoning.is_empty() {
        output.push(json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]}));
    }
    if !content.is_empty() {
        output.push(json!({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": content}]}));
    }
    for c in tool_calls {
        output.push(json!({"type": "function_call", "call_id": c.id, "name": c.name, "arguments": c.arguments}));
    }
    json!({
        "id": id, "object": "response", "created_at": now_secs(), "model": model, "status": "completed",
        "output": output,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    })
    .to_string()
}
