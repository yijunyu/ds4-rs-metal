//! Validates the pure-Rust tokenizer against a real DS4 GGUF's vocab metadata.
//! `GgufFile::open` reads only the header + metadata (tokenizer arrays, ~MB),
//! NOT the 86GB weight section — so this is fast and needs no Metal.
//!
//! Opt-in: DS4_GGUF=/path/to/ds4flash.gguf.

use std::path::PathBuf;

use ds4_engine::gguf::GgufFile;
use ds4_engine::tokenizer::{ChatMessage, StreamingDecoder, ThinkMode, Vocab};

fn open() -> Option<Vocab> {
    let p = std::env::var("DS4_GGUF").ok()?;
    let path = PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF not a file — skipping");
        return None;
    }
    let g = GgufFile::open(&path).expect("open gguf metadata");
    Some(Vocab::from_gguf(&g).expect("Vocab::from_gguf"))
}

#[test]
fn from_gguf_special_tokens_present() {
    let Some(v) = open() else { return };
    let st = v.special_tokens().expect("special tokens");
    eprintln!(
        "special: bos={} eos={} user={} assistant={} think_start={} think_end={} dsml={}",
        st.bos, st.eos, st.user, st.assistant, st.think_start, st.think_end, st.dsml
    );
    assert!(st.bos >= 0 && st.eos >= 0 && st.user >= 0 && st.assistant >= 0 && st.think_end >= 0);
    assert!(v.n_vocab() > 100_000, "expected ~129k vocab, got {}", v.n_vocab());
}

#[test]
fn encode_decode_roundtrip() {
    let Some(v) = open() else { return };
    for text in ["Hello, world!", "fn main() { println!(\"hi\"); }", "你好，世界 🌍", "café — naïve"] {
        let ids = v.encode(text);
        let dec = v.decode(&ids).expect("decode");
        assert_eq!(dec, text, "round-trip mismatch for {text:?}");
    }
}

#[test]
fn streaming_decode_matches_whole() {
    let Some(v) = open() else { return };
    // Include multi-byte chars likely split across byte-level BPE tokens.
    let text = "你好世界🌍 — streaming UTF-8 résumé café";
    let ids = v.encode(text);
    let whole = v.decode(&ids).expect("decode");
    let mut sd = StreamingDecoder::new(&v);
    let mut streamed = String::new();
    for &id in &ids {
        streamed.push_str(&sd.push(id));
    }
    streamed.push_str(&sd.flush());
    assert_eq!(streamed, whole, "streaming decode must equal whole-sequence decode");
}

#[test]
fn chat_prompt_structure() {
    let Some(v) = open() else { return };
    let st = v.special_tokens().unwrap();
    let msgs = [
        ChatMessage { role: "system", content: "You are helpful." },
        ChatMessage { role: "user", content: "Hi there" },
    ];
    let think = v.build_chat_prompt(&st, &msgs, ThinkMode::High, None);
    assert_eq!(think[0], st.bos, "prompt starts with bos");
    assert!(think.contains(&st.user), "has user marker");
    assert!(think.contains(&st.assistant), "has assistant marker");
    assert_eq!(*think.last().unwrap(), st.think_start, "thinking opens <think>");

    let nothink = v.build_chat_prompt(&st, &msgs, ThinkMode::None, None);
    assert_eq!(*nothink.last().unwrap(), st.think_end, "non-thinking closes </think>");
}
