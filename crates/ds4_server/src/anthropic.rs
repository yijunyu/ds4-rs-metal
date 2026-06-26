//! Anthropic `/v1/messages` wire mapping (for Claude Code). Port of the antirez
//! ds4-server Anthropic emitter (ds4_server.c ~6733-7490): message_start →
//! content_block_start/delta/stop (thinking, text, tool_use) → message_delta →
//! message_stop. Reuses the shared generation core via [`GenReq`]/[`GenEvent`].

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
    format!("msg_{n:x}")
}

fn stop_reason(finish: &str) -> &'static str {
    match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// Text of an Anthropic `content` field (string, or array of blocks: `text`,
/// `tool_result`, …; we extract any `text`).
fn block_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                b.get("text").and_then(|t| t.as_str()).map(String::from).or_else(|| {
                    // tool_result content can itself be a string or block array.
                    b.get("content").map(block_text)
                })
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn think_mode(req: &Value) -> ThinkMode {
    // Anthropic `thinking: {type: "enabled"|"disabled", budget_tokens}`.
    match req.get("thinking").and_then(|t| t.get("type")).and_then(|s| s.as_str()) {
        Some("disabled") => ThinkMode::None,
        Some("enabled") => ThinkMode::High,
        _ => ThinkMode::High, // DS4 reasons by default
    }
}

/// Parse an Anthropic messages request into a [`GenReq`] + `stream?`.
pub fn parse_messages_request(req: &Value, ctx: u32) -> Result<(GenReq, bool), String> {
    let mut messages: Vec<(String, String)> = Vec::new();
    // `system` (string or array of text blocks) → a leading system message.
    if let Some(sys) = req.get("system") {
        let s = block_text(sys);
        if !s.is_empty() {
            messages.push(("system".to_string(), s));
        }
    }
    let arr = req.get("messages").and_then(|m| m.as_array()).ok_or("missing `messages`")?;
    for m in arr {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string();
        let content = m.get("content").map(block_text).unwrap_or_default();
        messages.push((role, content));
    }
    if messages.is_empty() {
        return Err("empty `messages`".into());
    }

    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let think = think_mode(req);
    let params = if think.enabled() {
        SampleParams { temperature: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0 }
    } else {
        SampleParams {
            temperature: req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            top_k: req.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            top_p: req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            min_p: 0.0,
        }
    };
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    let requested = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(4096) as u32;
    let max_tokens = requested.min(ctx.saturating_sub(64)).max(1);
    let stop = match req.get("stop_sequences") {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let tools = req.get("tools").and_then(|t| t.as_array()).filter(|a| !a.is_empty()).map(|arr| {
        arr.iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?;
                let desc = t.get("description").and_then(|d| d.as_str()).unwrap_or("");
                let schema = t.get("input_schema").map(|p| p.to_string()).unwrap_or_else(|| "{}".into());
                Some(format!("### {name}\n{desc}\nParameters (JSON Schema): {schema}\n"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    });

    Ok((GenReq { messages, params, seed, max_tokens, think, stop, tools }, stream))
}

#[derive(PartialEq, Clone, Copy)]
enum Block {
    None,
    Thinking,
    Text,
}

/// Streaming state machine: maps [`GenEvent`]s to Anthropic SSE events.
pub struct AnthSink {
    open: Block,
    index: u32,
    id: String,
    model: String,
    started: bool,
}

impl AnthSink {
    pub fn new(model: &str) -> Self {
        Self { open: Block::None, index: 0, id: gen_id(), model: model.to_string(), started: false }
    }

    fn ensure_started(&mut self, s: &mut TcpStream) -> IoResult<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let data = json!({
            "type": "message_start",
            "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        http::write_sse_event(s, "message_start", &data.to_string())
    }

    fn close_block(&mut self, s: &mut TcpStream) -> IoResult<()> {
        if self.open != Block::None {
            http::write_sse_event(
                s,
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": self.index}).to_string(),
            )?;
            self.index += 1;
            self.open = Block::None;
        }
        Ok(())
    }

    fn open_block(&mut self, s: &mut TcpStream, kind: Block) -> IoResult<()> {
        if self.open == kind {
            return Ok(());
        }
        self.close_block(s)?;
        let cb = if kind == Block::Thinking {
            json!({"type": "thinking", "thinking": "", "signature": ""})
        } else {
            json!({"type": "text", "text": ""})
        };
        http::write_sse_event(
            s,
            "content_block_start",
            &json!({"type": "content_block_start", "index": self.index, "content_block": cb}).to_string(),
        )?;
        self.open = kind;
        Ok(())
    }

    /// Handle one generation event.
    pub fn on(&mut self, s: &mut TcpStream, ev: GenEvent) -> IoResult<()> {
        self.ensure_started(s)?;
        match ev {
            GenEvent::Reasoning(t) => {
                self.open_block(s, Block::Thinking)?;
                http::write_sse_event(
                    s,
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": self.index,
                            "delta": {"type": "thinking_delta", "thinking": t}})
                    .to_string(),
                )
            }
            GenEvent::Content(t) => {
                self.open_block(s, Block::Text)?;
                http::write_sse_event(
                    s,
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": self.index,
                            "delta": {"type": "text_delta", "text": t}})
                    .to_string(),
                )
            }
            GenEvent::ToolCalls(calls) => {
                self.close_block(s)?;
                for c in &calls {
                    http::write_sse_event(
                        s,
                        "content_block_start",
                        &json!({"type": "content_block_start", "index": self.index,
                                "content_block": {"type": "tool_use", "id": c.id, "name": c.name, "input": {}}})
                        .to_string(),
                    )?;
                    http::write_sse_event(
                        s,
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": self.index,
                                "delta": {"type": "input_json_delta", "partial_json": c.arguments}})
                        .to_string(),
                    )?;
                    http::write_sse_event(
                        s,
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": self.index}).to_string(),
                    )?;
                    self.index += 1;
                }
                Ok(())
            }
            GenEvent::Done { finish_reason, completion_tokens, .. } => {
                self.close_block(s)?;
                http::write_sse_event(
                    s,
                    "message_delta",
                    &json!({"type": "message_delta",
                            "delta": {"stop_reason": stop_reason(&finish_reason), "stop_sequence": Value::Null},
                            "usage": {"output_tokens": completion_tokens}})
                    .to_string(),
                )?;
                http::write_sse_event(s, "message_stop", &json!({"type": "message_stop"}).to_string())
            }
            GenEvent::Error(e) => http::write_sse_event(
                s,
                "error",
                &json!({"type": "error", "error": {"type": "api_error", "message": e}}).to_string(),
            ),
        }
    }
}

/// Non-streaming `/v1/messages` response.
#[allow(clippy::too_many_arguments)]
pub fn final_message_json(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCall],
    finish: &str,
    input_tokens: usize,
    output_tokens: usize,
) -> String {
    let mut blocks: Vec<Value> = Vec::new();
    if !reasoning.is_empty() {
        blocks.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    if !content.is_empty() {
        blocks.push(json!({"type": "text", "text": content}));
    }
    for c in tool_calls {
        let input: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
        blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": input}));
    }
    json!({
        "id": id, "type": "message", "role": "assistant", "model": model,
        "content": blocks, "stop_reason": stop_reason(finish), "stop_sequence": Value::Null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
    })
    .to_string()
}
