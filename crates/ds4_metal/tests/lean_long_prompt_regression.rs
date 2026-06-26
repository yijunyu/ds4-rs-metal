//! Regression test for the lean long-prompt crash (2026-06-02).
//!
//! Under lean weights the server skips the f32 duplicates of the q8/f16-backed
//! weights. The SHORT-prompt decode path uses the q8-correct merged encoder
//! branches, but a LONG prompt (>~300 tokens) drives the staged/chain branches,
//! which used to read the now-empty f32 attn/output/hc/shared weights directly
//! and panic ("attn_q_a / w_o_a shape mismatch (0 …)", empty-slice index, …).
//!
//! This asserts that a 400-token prompt decodes BYTE-FOR-BYTE the same under
//! lean (`DS4_LEAN_WEIGHTS=1`, the q8/f16 staged path) and non-lean (`=0`, the
//! f32 staged path) — i.e. the staged branches are now q8/f16-correct, not just
//! non-panicking. Short-prompt A/Bs never covered this; that was the validation
//! gap behind the incident.
//!
//! Opt-in (loads the real model twice): `DS4_GGUF=/path/to/ds4flash.gguf`.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_metal::decode_runner::DecodeRunner;

#[test]
fn lean_long_prompt_matches_nonlean() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping lean_long_prompt_matches_nonlean.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF={p} is not a regular file — skipping.");
        return;
    }

    // 400-token deterministic prompt — long enough to drive the staged/chain
    // branches (the ones that crashed). Greedy argmax decode is deterministic.
    let prompt: Vec<i32> = (1..=400).collect();
    let n_decode: u32 = 24;
    // raw_cap defaults above the prompt (raw ring holds everything). Set
    // DS4_TEST_RAW_CAP < prompt to drive the COMPRESSOR/long-context branch
    // (the daemon runs raw_cap=8192; a 28k prompt overflows it). That branch
    // routes non-hash layers through run_ffn_half's f32 shared-weights path,
    // which lean empties — the 2026-06-02 28k daemon crash. A small raw_cap
    // reproduces it cheaply (no 28k tokens needed).
    let raw_cap = std::env::var("DS4_TEST_RAW_CAP")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or((prompt.len() as u32 + n_decode + 8).max(256));

    let run = |lean: &str| -> Vec<i32> {
        // DecodeRunner::open reads DS4_LEAN_WEIGHTS at build time. The two opens
        // are sequential, so the process-global env is unambiguous per run.
        std::env::set_var("DS4_LEAN_WEIGHTS", lean);
        let runner = DecodeRunner::open(&path, raw_cap).expect("DecodeRunner::open");
        let (tokens, _prefill_s, _decode_s) = runner
            .run_argmax_timed(&prompt, n_decode, -1)
            .expect("run_argmax_timed");
        tokens
    };

    let lean = run("1"); // q8/f16 staged path
    let nonlean = run("0"); // f32 staged path
    std::env::remove_var("DS4_LEAN_WEIGHTS");

    assert_eq!(
        lean.len(),
        n_decode as usize,
        "lean decode produced {} tokens, expected {n_decode}",
        lean.len()
    );
    assert_eq!(
        lean, nonlean,
        "lean long-prompt decode diverged from non-lean — the staged/chain q8/f16 \
         path is not bit-identical to the f32 path.\n  lean   ={lean:?}\n  nonlean={nonlean:?}"
    );
    eprintln!(
        "lean_long_prompt_matches_nonlean PASS: {} tokens byte-identical (lean == non-lean) \
         over a {}-token prompt",
        lean.len(),
        prompt.len()
    );
}
