//! Long-context lean regression via `DecodeSession` (the EXACT daemon path).
//!
//! The daemon runs `DecodeSession::feed` → `decode_token_via_first_half`
//! (single-buffer encoder). On prompts past ~2048 tokens the INDEXER engages,
//! which drops the encoder off its fast/chain path onto
//! `run_ffn_half_metal_scoped`'s f32 fallback (`run_ffn_half` →
//! `moe_and_shared_chain_batched`) for the hash-routed layers (0/1/2). Under
//! lean that read the emptied f32 shared weights → panic at macos.rs:6403
//! (`w_gate.len 0`) — the 2026-06-02 28k-daemon crash. The fix keeps the f32
//! shared weights for hash layers (mirroring the f32-router carve-out).
//!
//! Why the prior `lean_long_prompt_regression` (run_argmax_timed) MISSED it:
//! that test drives the trait `decode_step_with_attn` path, not `DecodeSession`;
//! and it sizes raw_cap ≥ prompt, so neither the indexer nor the slow path runs.
//!
//! Default raw_cap=4096 / prompt=2500: indexer engages (>2048) with NO slot
//! overflow (2500 < 4096) — the minimal faithful reproduction. ~10 min (loads
//! the model twice, prefills 2500 tokens each). Override with DS4_TEST_RAW_CAP /
//! DS4_TEST_PROMPT_LEN.
//!
//! Opt-in (loads the real model): `DS4_GGUF=/path/to/ds4flash.gguf`.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

#[test]
fn lean_compressor_session_matches_nonlean() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping lean_compressor_session_matches_nonlean.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF={p} is not a regular file — skipping.");
        return;
    }

    // Default: indexer engages (prompt > ~2048) with NO slot overflow
    // (prompt < raw_cap), so the daemon's long-context slow path runs without
    // the harness's `slot >= raw_cap` assert. raw_cap < prompt would instead
    // overflow the raw KV ring before reaching the slow-path FFN.
    let raw_cap: u32 = std::env::var("DS4_TEST_RAW_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let plen: i32 = std::env::var("DS4_TEST_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2500);
    let prompt: Vec<i32> = (1..=plen).collect();
    let n_decode: usize = std::env::var("DS4_TEST_N_DECODE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let run = |lean: &str| -> Vec<i32> {
        std::env::set_var("DS4_LEAN_WEIGHTS", lean);
        let runner = DecodeRunner::open(&path, raw_cap).expect("DecodeRunner::open");
        let mut sess = DecodeSession::new(&runner);
        let _pf = std::time::Instant::now();
        sess.prefill(&prompt).expect("prefill");
        if std::env::var("DS4_TEST_SINGLE").is_ok() {
            eprintln!("[prefill-timing] {} tok in {:.2}s ({:.1} ms/tok)",
                prompt.len(), _pf.elapsed().as_secs_f64(),
                _pf.elapsed().as_secs_f64() * 1000.0 / prompt.len() as f64);
        }
        let mut out = Vec::with_capacity(n_decode);
        // Profile mode times the decode loop (excludes prefill) → decode tok/s
        // at this context length. Skip the first token (warm-up) when averaging.
        let timing = std::env::var("DS4_TEST_SINGLE").is_ok();
        let mut t_start: Option<std::time::Instant> = None;
        for i in 0..n_decode {
            let tok = argmax(sess.logits());
            out.push(tok);
            if timing && i == 1 { t_start = Some(std::time::Instant::now()); }
            sess.step(tok).expect("step");
        }
        if let Some(t0) = t_start {
            let secs = t0.elapsed().as_secs_f64();
            let n = (n_decode - 1) as f64; // tokens 1..n_decode timed
            eprintln!(
                "[decode-timing] ctx={} {:.2} tok/s ({} tokens in {:.3}s)",
                prompt.len(), n / secs, n_decode - 1, secs
            );
        }
        out
    };

    let lean = run("1");
    // DS4_TEST_SINGLE=1: skip the non-lean pass (profiling mode — one load only).
    if std::env::var("DS4_TEST_SINGLE").is_ok() {
        std::env::remove_var("DS4_LEAN_WEIGHTS");
        eprintln!("lean_compressor_session (SINGLE/profile) TOKENS={:?}", lean);
        return;
    }
    let nonlean = run("0");
    std::env::remove_var("DS4_LEAN_WEIGHTS");

    assert_eq!(
        lean, nonlean,
        "lean compressor-path decode diverged from non-lean.\n  lean   ={lean:?}\n  nonlean={nonlean:?}"
    );
    eprintln!(
        "lean_compressor_session_matches_nonlean PASS: {} tokens byte-identical \
         (lean == non-lean), prompt={} raw_cap={raw_cap} (long-context slow path)",
        lean.len(),
        prompt.len()
    );
}

fn argmax(logits: &[f32]) -> i32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as i32
}
