//! Times `DecodeSession::prefill` — the event-pipeline / chunk batched-prefill
//! path (the one the "304 tok/s @3000" headline came from). The other benches
//! (ds4-infer = ds4_engine::prefill_ffi; decode_runner_embed_bench =
//! run_argmax_timed) use PER-TOKEN prefill, so they can't reproduce it.
//!
//! Opt-in: DS4_GGUF=/path/to/ds4flash.gguf
//!   DS4_BENCH_PROMPT_LEN=3000 (default)  DS4_RAW_CAP=128 (production SWA)
//!   DS4_PREFILL_CHUNK is read inside DecodeSession::prefill to select the path.
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};
use std::time::Instant;

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as i32
}

#[test]
fn session_prefill_timed() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping session_prefill_timed.");
        return;
    };
    let path = std::path::PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF={p} is not a regular file — skipping.");
        return;
    }
    let n_prompt: i32 = std::env::var("DS4_BENCH_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let n_decode: u32 = std::env::var("DS4_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let raw_cap: u32 = std::env::var("DS4_RAW_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let prompt: Vec<i32> = (1..=n_prompt).collect();

    let runner = DecodeRunner::open(&path, raw_cap).expect("DecodeRunner::open");
    let mut session = DecodeSession::new(&runner);

    // ── prefill (the event-pipeline batched path — the 304 number) ──
    let t = Instant::now();
    session.prefill(&prompt).expect("DecodeSession::prefill");
    let prefill_s = t.elapsed().as_secs_f64();

    // ── decode (same DecodeSession, greedy) ──
    let mut tok = argmax(session.logits());
    let td = Instant::now();
    for _ in 0..n_decode {
        session.step(tok).expect("DecodeSession::step");
        tok = argmax(session.logits());
    }
    let decode_s = td.elapsed().as_secs_f64();

    eprintln!(
        "SESSION_BENCH: prefill_tok_s = {:.2}  decode_tok_s = {:.2}  \
         ({} prompt tok in {:.3}s; {} decode tok in {:.3}s; raw_cap={raw_cap})",
        n_prompt as f64 / prefill_s.max(1e-9),
        n_decode as f64 / decode_s.max(1e-9),
        n_prompt,
        prefill_s,
        n_decode,
        decode_s,
    );
}
