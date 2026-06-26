//! Minimal DecodeRunner argmax decode driver — exercises the token_embd
//! compact-storage path (DecodeRunner::open + embed() per token) WITHOUT the
//! antirez Engine. Used for the branch-vs-main A/B: token-identity, tok/s, and
//! peak RSS (under /usr/bin/time -l).
//!
//! Opt-in: DS4_GGUF=/path/to/ds4flash.gguf. DS4_BENCH_TOKENS=N (default 24).
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_metal::decode_runner::DecodeRunner;

#[test]
fn decode_runner_embed_bench() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping decode_runner_embed_bench.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF={p} is not a regular file — skipping.");
        return;
    }
    let n_decode: u32 = std::env::var("DS4_BENCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    // Deterministic synthetic prompt — the A/B only needs identical input on
    // both branches; greedy argmax decode is deterministic given tokens.
    // DS4_BENCH_PROMPT_LEN>16 exercises the LONG-PROMPT staged/chain encoder
    // branches (the ones that read f32 weights and panic under lean).
    let prompt_len: i32 = std::env::var("DS4_BENCH_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let prompt: Vec<i32> = (1..=prompt_len).collect();
    let raw_cap = (prompt.len() as u32 + n_decode + 8).max(256);

    let runner = DecodeRunner::open(&path, raw_cap).expect("DecodeRunner::open");
    let (tokens, prefill_s, decode_s) = runner
        .run_argmax_timed(&prompt, n_decode, -1)
        .expect("run_argmax_timed");
    eprintln!("EMITTED {} tokens: {:?}", tokens.len(), tokens);
    eprintln!(
        "decode_tok_s = {:.3} ({} tok in {:.3}s; prefill {:.3}s)",
        tokens.len() as f64 / decode_s.max(1e-9),
        tokens.len(),
        decode_s,
        prefill_s
    );
}
