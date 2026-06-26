//! Long-context validation for the GPU indexer offload (DS4_GPU_INDEXER).
//!
//! The indexer only does real work once a compressor layer accumulates
//! n_comp > top_k compressed rows. At the production top_k=512 (ratio 4) that's
//! ~2048 tokens — decoding that far twice on the full model is memory-bound and
//! impractical as a test. So this harness lowers the threshold via
//! DS4_N_INDEXER_TOP_K_OVERRIDE (default 16) so the REAL staged indexer path
//! engages at ~64 tokens, then decodes a short sequence twice (CPU indexer,
//! then GPU indexer) and asserts the final `cur_hc` matches. This validates the
//! full real-model wiring (encode_indexer_scores fed real qr_normed/normed/ring
//! + CPU top-k, vs indexer_allowed_decode_one) end to end.
//!
//! Note on perf: at this small n_comp the indexer score loop is tiny, so the
//! per-token CPU-vs-GPU delta is in the noise — the GPU win only materializes at
//! large n_comp (thousands of rows), which needs real long context. This test
//! is a CORRECTNESS/integration gate, not a throughput measurement.
//!
//! Opt-in: DS4_GPU_INDEXER_BENCH=1 + DS4_GGUF=/path. macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::{AttnStepState, ComposedModelWeights};
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_metal::single_buffer_encoder::{LayerCutpoint, SingleBufferEncoder};
use ds4_metal::MetalDispatcher;
use std::path::PathBuf;

fn run_decode(
    model: &ComposedModelWeights,
    disp: &MetalDispatcher,
    raw_cap: u32,
    n_tokens: u32,
    threshold_pos: u32,
    gpu_indexer: bool,
) -> (Vec<f32>, f64, usize) {
    std::env::set_var("DS4_GPU_INDEXER", if gpu_indexer { "1" } else { "0" });
    let d_model = model.d_model;
    let mut state = AttnStepState::new(model, raw_cap);
    let encoder = SingleBufferEncoder::new(disp, raw_cap);
    let mut post_ms = Vec::new();
    for t in 0..n_tokens {
        let x: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32 * 0.011 + t as f32 * 0.137).sin() * 0.4).clamp(-1.0, 1.0))
            .collect();
        let t0 = std::time::Instant::now();
        encoder
            .encode_first_half(&x, model, &mut state, LayerCutpoint(model.layers.len()))
            .expect("encode_first_half");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if state.pos >= threshold_pos {
            post_ms.push(ms);
        }
        state.pos = state.pos.saturating_add(1);
    }
    let n = post_ms.len().max(1);
    let avg = post_ms.iter().sum::<f64>() / n as f64;
    (state.cur_hc.clone(), avg, post_ms.len())
}

#[test]
fn indexer_gpu_long_context_parity_and_bench() {
    if std::env::var("DS4_GPU_INDEXER_BENCH").ok().as_deref() != Some("1") {
        eprintln!("DS4_GPU_INDEXER_BENCH unset — skipping. Set =1 (and DS4_GGUF=/path) to run.");
        return;
    }
    let path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("DS4_GGUF unset — skipping");
            return;
        }
    };
    if !path.is_file() {
        eprintln!("DS4_GGUF={} is not a file — skipping", path.display());
        return;
    }

    // Lower the indexer threshold so the real staged path engages early (~64
    // tokens at ratio 4) instead of ~2048. ratio is 4 for DS4 indexer layers,
    // so n_index_comp crosses `top_k` at ~top_k*4 tokens.
    let top_k: u32 = std::env::var("DS4_N_INDEXER_TOP_K_OVERRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    std::env::set_var("DS4_N_INDEXER_TOP_K_OVERRIDE", top_k.to_string());
    let threshold_pos = top_k * 4;
    let n_tokens: u32 = std::env::var("DS4_INDEXER_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(threshold_pos + 40);
    assert!(
        n_tokens > threshold_pos,
        "DS4_INDEXER_BENCH_N ({n_tokens}) must exceed top_k*4 ({threshold_pos}) so the GPU indexer engages"
    );

    eprintln!("loading GGUF: {}", path.display());
    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model =
        ComposedModelWeights::from_views(&views, &manifest, defaults).expect("ComposedModelWeights");
    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&path).expect("open GGUF");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights");
    let raw_cap: u32 = n_tokens + 64;

    eprintln!(
        "top_k={top_k} (override) → indexer engages at pos>={threshold_pos}; decoding {n_tokens} tokens twice (CPU then GPU)..."
    );
    let (cur_cpu, cpu_avg, n_post) =
        run_decode(&model, &disp, raw_cap, n_tokens, threshold_pos, false);
    let (cur_gpu, gpu_avg, _) =
        run_decode(&model, &disp, raw_cap, n_tokens, threshold_pos, true);
    std::env::remove_var("DS4_GPU_INDEXER");
    std::env::remove_var("DS4_N_INDEXER_TOP_K_OVERRIDE");

    assert_eq!(cur_cpu.len(), cur_gpu.len(), "cur_hc length mismatch");
    let max_abs = cur_cpu.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut max_diff = 0.0f32;
    for (a, b) in cur_cpu.iter().zip(cur_gpu.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    let rel = max_diff / max_abs.max(1e-9);
    eprintln!("===== long-context indexer (top_k={top_k}) summary =====");
    eprintln!("  post-threshold tokens measured: {n_post}");
    eprintln!("  CPU indexer: {cpu_avg:.2} ms/token  ({:.2} tok/s)", 1000.0 / cpu_avg);
    eprintln!("  GPU indexer: {gpu_avg:.2} ms/token  ({:.2} tok/s)", 1000.0 / gpu_avg);
    eprintln!("  cur_hc parity: max_abs_diff={max_diff:.3e} max|cpu|={max_abs:.3e} rel={rel:.3e}");
    // Tolerance = the project's standard GPU-vs-CPU decode bar (rel < 0.50;
    // chained-vs-reference is historically ~0.21). The GPU score op itself is
    // bit-exact (indexer_scores_gpu_parity_smoke: rel 7e-7); the residual here
    // is GPU-vs-CPU rope on the indexer q compounding over the autoregressive
    // rollout via occasional top-k boundary flips — the same accepted delta the
    // rest of the chained decode carries.
    assert!(
        rel < 0.50,
        "GPU-indexer decode diverged beyond the GPU-vs-CPU bar: rel={rel:.3e} (max_abs_diff={max_diff:.3e})"
    );
}

/// Run-to-run determinism of the long-context staged GPU decode: two
/// fresh-state GPU runs sharing one dispatcher must agree. Regression guard
/// for the persistent-compressor-pool reset (MetalState::reset_decode_state_pools
/// at pos==0). Before the fix these diverged by ~rel 0.13 (stale state_score
/// un-masking compressor window slots). Opt-in: DS4_GPU_INDEXER_BENCH=1 + DS4_GGUF.
#[test]
fn staged_decode_run_to_run_deterministic() {
    if std::env::var("DS4_GPU_INDEXER_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DS4_GPU_INDEXER_BENCH=1 + DS4_GGUF=/path");
        return;
    }
    let path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("skip: DS4_GGUF unset"); return; }
    };
    if !path.is_file() { eprintln!("skip: DS4_GGUF not a file"); return; }
    let top_k: u32 = 16;
    std::env::set_var("DS4_N_INDEXER_TOP_K_OVERRIDE", top_k.to_string());
    let threshold_pos = top_k * 4;
    let n_tokens: u32 = std::env::var("DS4_INDEXER_BENCH_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(threshold_pos + 24);

    let manifest = validate_ds4_layout(&path).expect("validate");
    let views = LayerViews::open(&path, manifest.n_layers).expect("views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults).expect("model");
    let mut disp = MetalDispatcher::new().expect("disp");
    let gguf = GgufFile::open(&path).expect("gguf");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers).expect("experts");
    let raw_cap = n_tokens + 64;

    // Two GPU (scores-only) runs, fresh state each, SAME dispatcher.
    let (a, _, _) = run_decode(&model, &disp, raw_cap, n_tokens, threshold_pos, true);
    let (b, _, _) = run_decode(&model, &disp, raw_cap, n_tokens, threshold_pos, true);
    std::env::remove_var("DS4_GPU_INDEXER");
    std::env::remove_var("DS4_N_INDEXER_TOP_K_OVERRIDE");

    let max_abs = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let mut md = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) { md = md.max((x - y).abs()); }
    let rel = md / max_abs.max(1e-9);
    eprintln!("run-to-run: max_abs_diff={md:.3e} max|a|={max_abs:.3e} rel={rel:.3e}");
    assert!(rel < 1e-4, "staged decode not deterministic run-to-run: rel={rel:.3e}");
}
