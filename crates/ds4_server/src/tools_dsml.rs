//! Parse the model's DSML tool-call format into OpenAI `tool_calls`. Port of
//! ds4_server.c `parse_generated_message_ex` (4416-4592):
//!
//! ```text
//! <｜DSML｜tool_calls>
//!   <｜DSML｜invoke name="TOOL">
//!     <｜DSML｜parameter name="P" string="true|false">VALUE</｜DSML｜parameter>
//!   </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//! Both full (`<｜DSML｜…`) and short (`<DSML｜…`) variants are accepted.
//! `string="true"` → raw text (XML-unescaped); `string="false"` → JSON value.

use serde_json::{Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object string (OpenAI `function.arguments`).
    pub arguments: String,
}

/// The three marker sets (full / short / legacy). Each: (tool_calls_start,
/// tool_calls_end, invoke_start, invoke_end, param_start, param_end).
const VARIANTS: &[[&str; 6]] = &[
    [
        "<｜DSML｜tool_calls>",
        "</｜DSML｜tool_calls>",
        "<｜DSML｜invoke",
        "</｜DSML｜invoke>",
        "<｜DSML｜parameter",
        "</｜DSML｜parameter>",
    ],
    [
        "<DSML｜tool_calls>",
        "</DSML｜tool_calls>",
        "<DSML｜invoke",
        "</DSML｜invoke>",
        "<DSML｜parameter",
        "</DSML｜parameter>",
    ],
    [
        "<tool_calls>",
        "</tool_calls>",
        "<invoke",
        "</invoke>",
        "<parameter",
        "</parameter>",
    ],
];

/// Substrings that indicate a tool-call block is starting (for streaming
/// detection). Any occurrence switches the content stream into buffering mode.
pub const START_HINTS: &[&str] = &["<｜DSML｜tool_calls", "<DSML｜tool_calls", "<tool_calls>"];

fn gen_call_id() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("call_{n:x}")
}

/// Extract `key="value"` from a tag fragment.
fn attr(tag: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

/// Parse `text` for a DSML tool-calls block. Returns the calls if a complete,
/// well-formed block is present.
pub fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    for v in VARIANTS {
        let [tc_start, tc_end, inv_start, inv_end, p_start, p_end] = *v;
        if let Some(pos) = text.find(tc_start) {
            if let Some(calls) = parse_block(&text[pos + tc_start.len()..], tc_end, inv_start, inv_end, p_start, p_end) {
                if !calls.is_empty() {
                    return Some(calls);
                }
            }
        }
    }
    None
}

fn parse_block(
    mut s: &str,
    tc_end: &str,
    inv_start: &str,
    inv_end: &str,
    p_start: &str,
    p_end: &str,
) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    loop {
        s = s.trim_start();
        if s.starts_with(tc_end) {
            return Some(calls);
        }
        if !s.starts_with(inv_start) {
            return None;
        }
        let tag_end = s.find('>')?;
        let name = attr(&s[..tag_end + 1], "name")?;
        s = &s[tag_end + 1..];

        let mut args: Map<String, Value> = Map::new();
        loop {
            s = s.trim_start();
            if s.starts_with(inv_end) {
                s = &s[inv_end.len()..];
                break;
            }
            if !s.starts_with(p_start) {
                return None;
            }
            let pe = s.find('>')?;
            let ptag = &s[..pe + 1];
            let pname = attr(ptag, "name")?;
            let pstring = attr(ptag, "string");
            let value_start = &s[pe + 1..];
            let vend = value_start.find(p_end)?;
            let raw_value = &value_start[..vend];
            let is_string = pstring.as_deref() != Some("false"); // default true
            let val: Value = if is_string {
                Value::String(unescape(raw_value))
            } else {
                serde_json::from_str(raw_value.trim()).unwrap_or_else(|_| Value::String(raw_value.to_string()))
            };
            args.insert(pname, val);
            s = &value_start[vend + p_end.len()..];
        }

        calls.push(ToolCall {
            id: gen_call_id(),
            name,
            arguments: Value::Object(args).to_string(),
        });
    }
}

/// Streaming separator: passes through final content until a tool-call block
/// begins, then buffers the rest for [`Self::finish`] to parse.
pub struct ToolScanner {
    enabled: bool,
    in_block: bool,
    buf: String,
}

impl ToolScanner {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, in_block: false, buf: String::new() }
    }

    /// Feed final-content text; returns content to stream now (empty once a tool
    /// block has begun).
    pub fn feed(&mut self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();
        }
        self.buf.push_str(text);
        if self.in_block {
            return String::new();
        }
        // Earliest start-hint position, if any.
        let hit = START_HINTS.iter().filter_map(|h| self.buf.find(h)).min();
        if let Some(idx) = hit {
            self.in_block = true;
            // Stream content before the block; keep the block buffered.
            let before = self.buf[..idx].to_string();
            self.buf = self.buf[idx..].to_string();
            return before;
        }
        // No block yet — stream all but a retained tail that might begin a hint.
        let keep = hint_tail(&self.buf);
        let emit_to = self.buf.len() - keep;
        let out = self.buf[..emit_to].to_string();
        self.buf = self.buf[emit_to..].to_string();
        out
    }

    /// End of stream: if a tool block was buffered, parse it. Returns
    /// `(tool_calls, trailing_content)` — trailing content is whatever was held
    /// back but is not actually a tool block.
    pub fn finish(&mut self) -> (Option<Vec<ToolCall>>, String) {
        let buf = std::mem::take(&mut self.buf);
        if !self.enabled {
            return (None, buf);
        }
        if let Some(calls) = parse_tool_calls(&buf) {
            (Some(calls), String::new())
        } else {
            (None, buf) // not a valid block → return as content
        }
    }
}

/// Retain trailing bytes of `s` that could be the start of a hint marker.
/// `max` must cover the LONGEST hint: `<｜DSML｜tool_calls` is 21 bytes (each
/// fullwidth `｜` is 3 UTF-8 bytes), so a fixed 16 split the hint mid-stream when
/// the detokenizer landed the buffer past byte 16 — the partial tag streamed out
/// as content and the full hint never matched (tool calls leaked as raw DSML).
fn hint_tail(s: &str) -> usize {
    let longest = START_HINTS.iter().map(|h| h.len()).max().unwrap_or(16);
    let max = longest.min(s.len());
    for back in (1..=max).rev() {
        let start = s.len() - back;
        if !s.is_char_boundary(start) {
            continue;
        }
        let tail = &s[start..];
        if START_HINTS.iter().any(|h| h.starts_with(tail)) {
            return back;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_variant() {
        let t = "some text <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n\
            <｜DSML｜parameter name=\"query\" string=\"true\">deep learning</｜DSML｜parameter>\n\
            <｜DSML｜parameter name=\"limit\" string=\"false\">5</｜DSML｜parameter>\n\
            </｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = parse_tool_calls(t).expect("calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["query"], "deep learning");
        assert_eq!(v["limit"], 5);
    }

    #[test]
    fn no_block_returns_none() {
        assert!(parse_tool_calls("just a normal answer, 391.").is_none());
    }

    #[test]
    fn scanner_streams_then_buffers() {
        let mut sc = ToolScanner::new(true);
        let a = sc.feed("Here you go: ");
        assert_eq!(a, "Here you go: ");
        let b = sc.feed("<｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls>");
        assert_eq!(b, ""); // block buffered, not streamed
        let (calls, trailing) = sc.finish();
        assert!(calls.is_some());
        assert_eq!(trailing, "");
        assert_eq!(calls.unwrap()[0].name, "f");
    }

    /// Regression: the detokenizer streams the start hint in tiny fragments, so
    /// the scanner must retain a partial hint LONGER than the old fixed 16 bytes
    /// (`<｜DSML｜tool_calls` is 21 bytes). Before the fix, a char-by-char feed
    /// split the hint past byte 16 → it streamed out as content → tool calls
    /// leaked as raw DSML (the live Odysseus-spike failure).
    #[test]
    fn scanner_split_start_hint_char_by_char() {
        let full = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\">\
            <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\
            </｜DSML｜invoke></｜DSML｜tool_calls>";
        let mut sc = ToolScanner::new(true);
        let mut streamed = String::new();
        for ch in full.chars() {
            streamed.push_str(&sc.feed(&ch.to_string()));
        }
        let (calls, trailing) = sc.finish();
        // The hint must NOT have leaked into the streamed/trailing content.
        assert!(!streamed.contains("DSML"), "start hint leaked: {streamed:?}");
        assert!(!trailing.contains("DSML"), "block leaked as trailing: {trailing:?}");
        let calls = calls.expect("tool call parsed from char-by-char stream");
        assert_eq!(calls[0].name, "get_weather");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["city"], "Paris");
    }
}
