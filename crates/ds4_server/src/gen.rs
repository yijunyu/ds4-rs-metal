//! Generation core: drives `DecodeSession` (prefill + sample loop), detokenizes
//! incrementally, splits reasoning vs content, and pushes [`GenEvent`]s to an
//! `emit` callback. Synchronous (runs on the single Metal thread).

use ds4_engine::tokenizer::{ChatMessage, SpecialTokens, StreamingDecoder, ThinkMode, Vocab};
use crate::backend::decode_runner::{DecodeRunner, DecodeSession};
use crate::backend::sampling::{sample, SampleParams, SampleRng};

use crate::tools_dsml::{ToolCall, ToolScanner};
use std::time::Instant;

/// A parsed generation request (wire-agnostic).
pub struct GenReq {
    pub messages: Vec<(String, String)>, // (role, content)
    pub params: SampleParams,
    pub seed: u64,
    pub max_tokens: u32,
    pub think: ThinkMode,
    pub stop: Vec<String>,
    pub tools: Option<String>,
}

/// Streamed generation output.
pub enum GenEvent {
    Reasoning(String),
    Content(String),
    ToolCalls(Vec<ToolCall>),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// Run one generation, invoking `emit` for each event. If `emit` returns
/// `Err` (e.g. the HTTP client disconnected), generation aborts early.
pub fn generate<F>(
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    req: &GenReq,
    emit: &mut F,
) -> std::io::Result<()>
where
    F: FnMut(GenEvent) -> std::io::Result<()>,
{
    let msgs: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|(r, c)| ChatMessage { role: r, content: c })
        .collect();
    // DS4_SERVER_PROFILE=1 → per-request engine-vs-service timing breakdown on
    // stderr. tokenize/prefill once; then per-token sample (service) / emit
    // (detok+split+scan+SSE, service) / step (engine GPU forward).
    let profile = std::env::var("DS4_SERVER_PROFILE").ok().as_deref() == Some("1");
    let (mut t_sample, mut t_emit, mut t_step) = (0u128, 0u128, 0u128);

    let t0 = Instant::now();
    let prompt = vocab.build_chat_prompt(st, &msgs, req.think, req.tools.as_deref());
    let prompt_tokens = prompt.len();
    let tokenize_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let mut session = DecodeSession::new(runner);
    if let Err(e) = session.prefill(&prompt) {
        return emit(GenEvent::Error(format!("prefill: {e}")));
    }
    let prefill_us = t1.elapsed().as_micros();

    let mut rng = SampleRng::new(req.seed);
    let mut detok = StreamingDecoder::new(vocab);
    let mut splitter = ThinkSplitter::new(req.think.enabled());
    let mut scanner = ToolScanner::new(req.tools.is_some());
    let mut content_acc = String::new();
    let mut completion_tokens = 0usize;
    let mut finish = "length".to_string();

    // Route a final-content fragment through the tool scanner; stream whatever
    // is not part of a tool-call block. Returns true if a stop sequence hit.
    let mut emit_content =
        |scanner: &mut ToolScanner, content_acc: &mut String, emit: &mut F, text: &str| -> std::io::Result<bool> {
            let streamable = scanner.feed(text);
            if streamable.is_empty() {
                return Ok(false);
            }
            content_acc.push_str(&streamable);
            emit(GenEvent::Content(streamable))?;
            Ok(req.stop.iter().any(|s| !s.is_empty() && content_acc.contains(s)))
        };

    for _ in 0..req.max_tokens {
        let ts = Instant::now();
        let tok = sample(session.logits(), &req.params, &mut rng);
        t_sample += ts.elapsed().as_micros();
        if tok == st.eos {
            finish = "stop".to_string();
            break;
        }
        completion_tokens += 1;
        let te = Instant::now();
        let text = detok.push(tok as u32);
        if !text.is_empty() {
            let (reasoning, content) = splitter.feed(&text);
            if !reasoning.is_empty() {
                emit(GenEvent::Reasoning(reasoning))?;
            }
            if !content.is_empty() && emit_content(&mut scanner, &mut content_acc, emit, &content)? {
                finish = "stop".to_string();
                break;
            }
        }
        t_emit += te.elapsed().as_micros();
        let tp = Instant::now();
        if let Err(e) = session.step(tok) {
            return emit(GenEvent::Error(format!("decode step: {e}")));
        }
        t_step += tp.elapsed().as_micros();
    }

    if profile {
        let n = completion_tokens.max(1) as f64;
        eprintln!(
            "ds4-profile: prompt={prompt_tokens}tok gen={completion_tokens}tok | \
             tokenize={:.1}ms prefill={:.1}ms | per-tok: step(engine)={:.2}ms \
             sample(service)={:.2}ms emit(service)={:.2}ms | \
             service-total={:.2}ms/tok ({:.1}%)",
            tokenize_us as f64 / 1e3,
            prefill_us as f64 / 1e3,
            t_step as f64 / 1e3 / n,
            t_sample as f64 / 1e3 / n,
            t_emit as f64 / 1e3 / n,
            (t_sample + t_emit) as f64 / 1e3 / n,
            (t_sample + t_emit) as f64 / (t_sample + t_emit + t_step).max(1) as f64 * 100.0,
        );
    }

    let tail = detok.flush();
    if !tail.is_empty() {
        let (r, c) = splitter.flush(&tail);
        if !r.is_empty() {
            emit(GenEvent::Reasoning(r))?;
        }
        if !c.is_empty() {
            let _ = emit_content(&mut scanner, &mut content_acc, emit, &c)?;
        }
    }

    // Resolve any buffered tool-call block.
    let (tool_calls, trailing) = scanner.finish();
    if let Some(calls) = tool_calls {
        if !calls.is_empty() {
            finish = "tool_calls".to_string();
            emit(GenEvent::ToolCalls(calls))?;
        }
    } else if !trailing.is_empty() {
        emit(GenEvent::Content(trailing))?;
    }

    emit(GenEvent::Done { finish_reason: finish, prompt_tokens, completion_tokens })
}

/// Incremental `<think>…</think>` splitter (port of ds4_server.c thinking_state,
/// 9287): classifies streamed text into reasoning (inside) vs content (outside),
/// consuming the tags. Buffers a short tail so tags spanning token boundaries
/// are detected.
struct ThinkSplitter {
    inside: bool,
    pending: String,
}

const TAG_OPEN: &str = "<think>";
const TAG_CLOSE: &str = "</think>";
const TAG_MAX_TAIL: usize = 7; // len("</think>") - 1

impl ThinkSplitter {
    fn new(inside: bool) -> Self {
        Self { inside, pending: String::new() }
    }

    fn feed(&mut self, text: &str) -> (String, String) {
        self.pending.push_str(text);
        let mut reasoning = String::new();
        let mut content = String::new();
        loop {
            let (tag, bucket): (&str, &mut String) = if self.inside {
                (TAG_CLOSE, &mut reasoning)
            } else {
                (TAG_OPEN, &mut content)
            };
            if let Some(idx) = self.pending.find(tag) {
                bucket.push_str(&self.pending[..idx]);
                self.pending.drain(..idx + tag.len());
                self.inside = !self.inside;
                continue;
            }
            let keep = safe_tail(&self.pending);
            let emit_to = self.pending.len() - keep;
            bucket.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }
        (reasoning, content)
    }

    fn flush(&mut self, extra: &str) -> (String, String) {
        self.pending.push_str(extra);
        let rest = std::mem::take(&mut self.pending);
        if self.inside {
            (rest, String::new())
        } else {
            (String::new(), rest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_block_split() {
        let mut s = ThinkSplitter::new(true); // thinking → start inside
        let (r, c) = s.feed("reasoning text</think>final answer");
        assert_eq!(r, "reasoning text");
        assert_eq!(c, "final answer");
    }

    #[test]
    fn tag_spanning_chunks() {
        let mut s = ThinkSplitter::new(true);
        let (r1, c1) = s.feed("think bits</thi");
        assert_eq!(r1, "think bits");
        assert_eq!(c1, "");
        let (r2, c2) = s.feed("nk>done");
        assert_eq!(r2, "");
        assert_eq!(c2, "done");
    }

    #[test]
    fn non_thinking_is_all_content() {
        let mut s = ThinkSplitter::new(false);
        let (r, c) = s.feed("just content, no tags");
        assert_eq!(r, "");
        assert_eq!(c, "just content, no tags");
    }

    #[test]
    fn feed_emits_eagerly_when_no_partial_tag() {
        let mut s = ThinkSplitter::new(true);
        let (r, c) = s.feed("partial reasoning"); // no '<' → all emitted now
        assert_eq!(r, "partial reasoning");
        assert_eq!(c, "");
    }

    #[test]
    fn flush_emits_retained_tag_tail() {
        let mut s = ThinkSplitter::new(true);
        let (r1, _) = s.feed("reasoning<"); // trailing '<' retained as possible tag
        assert_eq!(r1, "reasoning");
        let (r2, c2) = s.flush(""); // no tag completed → emit the retained '<'
        assert_eq!(r2, "<");
        assert_eq!(c2, "");
    }
}

/// Bytes of trailing `s` to retain as a possible partial tag (≤ `TAG_MAX_TAIL`,
/// on a char boundary, only if it's a prefix of `<think>` or `</think>`).
fn safe_tail(s: &str) -> usize {
    let max = TAG_MAX_TAIL.min(s.len());
    let bytes = s.as_bytes();
    for back in (1..=max).rev() {
        let start = s.len() - back;
        if !s.is_char_boundary(start) {
            continue;
        }
        if bytes[start] == b'<' {
            let tail = &s[start..];
            if TAG_OPEN.starts_with(tail) || TAG_CLOSE.starts_with(tail) {
                return back;
            }
        }
    }
    0
}
