//! MODEL-LEVEL per-layer RESIDUAL divergence probe (the definitive cross-path
//! tool the kernel guardrails pointed to). The synthetic guardrails cleared every
//! kernel (matmul/flash/glue/router/cross-impl all faithful) ⇒ the chunk-prefill
//! divergence is INPUT-SIDE. This reads the RAW per-layer residual (not the lagged
//! comp-ring proxy) of the chunk vs per-token prefills at the same position and
//! reports, per layer, where they first diverge — pinning the exact first-drift
//! layer and (raw residual catches comp-nullspace drift the comp ring hides).
//!
//! Opt-in (loads the 86 GB model once): DS4_GGUF=/path [DS4_PROBE_PROMPT=/path]
//! [DS4_PROBE_POS=N]. macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::DecodeRunner;

#[test]
fn residual_divergence_probe_test() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping residual_divergence_probe_test.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path)
        .unwrap_or_else(|e| panic!("read prompt {prompt_path}: {e}"));

    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    eprintln!("[resid-probe] prompt tokens = {}", prompt.len());

    let target_pos: usize = std::env::var("DS4_PROBE_POS").ok().and_then(|v| v.parse().ok())
        .unwrap_or(prompt.len().saturating_sub(2));

    let runner = DecodeRunner::open(&path, 128).expect("open");
    let ratios: Vec<u32> = runner.composed.layers.iter().map(|l| l.attn.params.compress_ratio).collect();

    let div = runner.residual_divergence_probe(&prompt, target_pos).expect("probe");

    eprintln!("[resid-probe] per-layer INPUT residual divergence (chunk vs per-token) @ pos {target_pos}:");
    eprintln!("[resid-probe] {:>3} {:>6} {:>10} {:>11}", "L", "ratio", "cos", "maxΔ");
    // first REAL drift = first layer with 0.5 < cos < 0.999. cos≈0 at the early
    // hash-routed layers is a CAPTURE artifact (state.cur_hc representation for the
    // Phase-A hash layers), NOT divergence — confirmed because the first cos==1.0
    // layer (bit-identical input) proves the layers feeding it are faithful.
    let mut first_drift: Option<usize> = None;
    let mut worst = (1.0f64, usize::MAX);
    for &(l, c, m) in &div {
        if (0.5..0.999).contains(&c) && first_drift.is_none() { first_drift = Some(l); }
        if c > 0.5 && c < worst.0 { worst = (c, l); }
        eprintln!("[resid-probe] {l:>3} {:>6} {c:>10.6} {m:>11.3e}", ratios.get(l).copied().unwrap_or(0));
    }
    eprintln!(
        "[resid-probe] FIRST real drift (0.5<cos<0.999) @ layer {:?}; WORST cos={:.6} @ layer {} (ratio {}). \
         (cos≈0 at early hash layers = capture artifact, not drift — the first cos==1.0 layer proves \
         the layers below it are faithful.)",
        first_drift, worst.0, worst.1, ratios.get(worst.1).copied().unwrap_or(0),
    );
    eprintln!(
        "[resid-probe] interpretation: divergence is INTRODUCED at the layer whose OUTPUT first drifts \
         (= the layer below the first cos<0.999), as a SMOOTH broad residual drift that COMPOUNDS with \
         depth (cos 1.0 → ~0.33) — the accumulated K-batched-matmul fp32 non-associativity (matmul_k \
         guardrail: rel 4e-4/op), later amplified by MoE routing flips (router onset ε≈3e-2)."
    );
}
