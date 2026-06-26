//! Minimal env-honoring chunk prefill (for RESID_SUM / HEADS / SEL probes):
//! prefills the last DS4_DEC_LEN tokens with WHATEVER chunk env the caller set
//! (no defaults forced), then greedy-decodes DS4_DEC_NDEC tokens.
//! Opt-in: DS4_GGUF + [DS4_PROBE_PROMPT] [DS4_DEC_LEN=256] [DS4_DEC_NDEC=0]. macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

#[test]
fn chunk_resid_probe() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_resid_probe."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path).expect("prompt");
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let l = l.min(full.len());
    let prompt = &full[full.len() - l..];
    let n_dec: u32 = std::env::var("DS4_DEC_NDEC").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let runner = DecodeRunner::open(&path, 128).expect("open");
    let mut s = DecodeSession::new(&runner);
    s.prefill(prompt).expect("prefill");
    let mut out = Vec::new();
    for _ in 0..n_dec {
        let logits = s.logits();
        let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; best = i as i32; } }
        out.push(best as u32);
        if s.step(best).is_err() { break; }
    }
    if n_dec > 0 {
        eprintln!("[probe] decode: {:?}", out);
        eprintln!("[probe] text: {}", vocab.decode(&out).unwrap_or_default());
    }
}
