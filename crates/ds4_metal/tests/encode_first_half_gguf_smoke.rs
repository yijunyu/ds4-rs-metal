//! Phase E M5.4.5.5 — GGUF-backed end-to-end smoke for
//! `SingleBufferEncoder::encode_first_half`.
//!
//! Opt-in: set `DS4_GGUF=/path/to/ds4flash.gguf` to run. Without it
//! the test prints a one-line skip and passes — keeps CI green on
//! machines without a DS4 GGUF on disk.
//!
//! Drives both:
//!   - `decode_step_with_attn_to_residual` (the trait-dispatch reference)
//!   - `SingleBufferEncoder::encode_first_half(l_split=n_layers)` (new
//!     unified-cb path)
//!
//! Asserts `state.cur_hc` is bit-close between the two paths (within
//! fp8 e4m3 propagation tolerance — strict bit-equivalence is gated
//! on M5.2.4 kv_fp8_store CPU/GPU unification; see
//! `project_m5_kv_fp8_store_divergence`).
//!
//! M5.4.5.5 added hash-routing support (DS4 V4 Flash layers 0/1/2);
//! compressor/indexer support follows in M5.4.5.6+ as needed by the
//! first layer that triggers a bail on this fixture.
//!
//! Memory cost: full DS4 V4 Flash model load (~10 GB), all 43 layers
//! of QuantizedExpertWeights. Designed to run sparingly — opt-in only.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::{
    decode_step_with_attn_to_residual, AttnStepState, ComposedModelWeights, DecodeConfig,
};
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_metal::single_buffer_encoder::{LayerCutpoint, SingleBufferEncoder};
use ds4_metal::MetalDispatcher;

fn open_gguf_or_skip(test_name: &str) -> Option<PathBuf> {
    let path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "DS4_GGUF unset — skipping {test_name}. Set DS4_GGUF=/path/to/ds4flash.gguf to run."
            );
            return None;
        }
    };
    if !path.is_file() {
        eprintln!(
            "DS4_GGUF={} is not a regular file — skipping {test_name}.",
            path.display()
        );
        return None;
    }
    Some(path)
}

#[test]
fn encode_first_half_matches_decode_step_residual_on_real_gguf() {
    let Some(path) = open_gguf_or_skip("encode_first_half end-to-end vs decode_step") else {
        return;
    };

    eprintln!("loading GGUF: {}", path.display());
    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights::from_views");
    eprintln!(
        "model loaded: d_model={} vocab={} n_layers={}",
        model.d_model,
        model.vocab_size,
        model.layers.len()
    );
    // M5.4.5.6: encode_first_half now supports hash-routing + compressor
    // + indexer layers. The full 43-layer model can run end-to-end.

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&path).expect("open GGUF for expert weight load");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (all layers)");
    eprintln!("loaded {} layers of expert weights", manifest.n_layers);

    let raw_cap: u32 = 256;
    let cfg = DecodeConfig::default();
    let d_model = model.d_model;
    let x: Vec<f32> = (0..d_model)
        .map(|i| ((i as f32 * 0.011).sin() * 0.4).clamp(-1.0, 1.0))
        .collect();

    // ── Reference: trait-dispatch path.
    let mut state_ref = AttnStepState::new(&model, raw_cap);
    eprintln!("running decode_step_with_attn_to_residual...");
    let t_ref = std::time::Instant::now();
    let _final_hidden = decode_step_with_attn_to_residual(
        &disp,
        &disp,
        x.clone(),
        &model,
        &mut state_ref,
        &cfg,
        raw_cap,
    )
    .expect("decode_step_with_attn_to_residual");
    eprintln!("decode_step took {:?}", t_ref.elapsed());

    // ── New path: a separate MetalDispatcher so persistent KV buffers
    //   start zero-initialised (the reference path wrote to
    //   state.kv_storage via the trait kv_fp8_store, not to persistent).
    let mut disp_new = MetalDispatcher::new().expect("MetalDispatcher::new (new path)");
    disp_new
        .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (new path)");
    let mut state_new = AttnStepState::new(&model, raw_cap);
    let encoder = SingleBufferEncoder::new(&disp_new, raw_cap);
    eprintln!("running encode_first_half (l_split=n_layers)...");
    let t_new = std::time::Instant::now();
    let _ = encoder
        .encode_first_half(
            &x,
            &model,
            &mut state_new,
            LayerCutpoint(model.layers.len()),
        )
        .expect("encode_first_half");
    eprintln!("encode_first_half took {:?}", t_new.elapsed());

    // ── state.cur_hc bit-close comparison (fp8 e4m3 propagation envelope).
    assert_eq!(
        state_ref.cur_hc.len(),
        state_new.cur_hc.len(),
        "cur_hc length mismatch"
    );
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut nonzero = 0usize;
    for (a, b) in state_ref.cur_hc.iter().zip(state_new.cur_hc.iter()) {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if a.abs() > max_ref {
            max_ref = a.abs();
        }
        if a.abs() > 1e-8 {
            nonzero += 1;
        }
    }
    let rel = max_abs / max_ref.max(1e-6);
    eprintln!(
        "cur_hc compare: len={} nonzero={} max_abs={:.6} max_ref={:.6} rel={:.4}",
        state_ref.cur_hc.len(),
        nonzero,
        max_abs,
        max_ref,
        rel
    );
    assert!(nonzero > 0, "reference cur_hc is all-zero — model didn't run");
    // fp8 e4m3 propagation across all 43 layers + persistent KV vs CPU
    // mirror kv_fp8_store divergence (see project_m5_kv_fp8_store_divergence
    // memory). Per-layer ~13% e4m3 mantissa drift compounds through the
    // residual stream; 43 layers × hash-routing layers 0/1 + compressor
    // layers + indexer layers measure ~23% rel on the q2 GGUF. Strict
    // bit-close lands with M5.2.4 (kv_fp8_store CPU/GPU unification).
    // Bench-grade evaluation (token-level argmax match across N decode
    // steps) ships with M5.4.5.bench.
    assert!(
        rel < 0.50,
        "encode_first_half diverged from decode_step beyond expected fp8 envelope: \
         max_abs={max_abs:.6}, max_ref={max_ref:.6}, rel={rel:.4}"
    );
}

/// M5 #100 multi-token regression — exercises the slow_path bridge.
///
/// The single-token smoke above only runs pos=0, where
/// `comp_sel_slice` is `None` (the indexer hasn't emitted yet, that
/// happens at pos≥3 for ratio==4 layers). So the bridge condition
/// fails and the prior non-bridge path fires.
///
/// This test decodes N tokens, advancing state.pos between calls,
/// so at pos≥3 the bridge actually runs (covering both the
/// scope-aware compressor flash_attn AND the GPU hc_collapse_norm
/// port for FFN-half). Compares state.cur_hc after the final token.
///
/// Opt-in: same `DS4_GGUF=` switch. Cost: N × ~1s per path.
#[test]
fn encode_first_half_matches_decode_step_residual_on_real_gguf_multi_token() {
    let Some(path) =
        open_gguf_or_skip("encode_first_half multi-token vs decode_step")
    else {
        return;
    };

    eprintln!("loading GGUF: {}", path.display());
    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights::from_views");
    eprintln!(
        "model loaded: d_model={} vocab={} n_layers={}",
        model.d_model,
        model.vocab_size,
        model.layers.len()
    );

    let raw_cap: u32 = 256;
    let cfg = DecodeConfig::default();
    let d_model = model.d_model;
    // 6 tokens — ensures the bridge fires at pos=3, 4, 5 (first
    // indexer emission lands at pos=3 for ratio==4 layers, then
    // every position after that has comp_sel_slice = Some).
    let n_tokens: u32 = 6;

    let xs: Vec<Vec<f32>> = (0..n_tokens)
        .map(|t| {
            (0..d_model)
                .map(|i| {
                    ((i as f32 * 0.011 + t as f32 * 0.137).sin() * 0.4)
                        .clamp(-1.0, 1.0)
                })
                .collect()
        })
        .collect();

    // ── Reference: trait-dispatch path, N decode_step calls.
    let mut disp_ref = MetalDispatcher::new().expect("MetalDispatcher::new (ref)");
    let gguf = GgufFile::open(&path).expect("open GGUF for expert weight load");
    disp_ref
        .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (ref)");
    let mut state_ref = AttnStepState::new(&model, raw_cap);
    eprintln!("running {} decode_step calls (reference)...", n_tokens);
    let t_ref = std::time::Instant::now();
    for (t, x) in xs.iter().enumerate() {
        let _ = decode_step_with_attn_to_residual(
            &disp_ref,
            &disp_ref,
            x.clone(),
            &model,
            &mut state_ref,
            &cfg,
            raw_cap,
        )
        .expect("decode_step");
        eprintln!(
            "  ref token {} done, state.pos now {}",
            t, state_ref.pos
        );
    }
    eprintln!("decode_step (×{}) took {:?}", n_tokens, t_ref.elapsed());

    // ── New path: encode_first_half ×N, manually advancing state.pos
    //   between calls (encode_first_half does NOT auto-advance pos).
    let mut disp_new = MetalDispatcher::new().expect("MetalDispatcher::new (new)");
    disp_new
        .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (new)");
    let mut state_new = AttnStepState::new(&model, raw_cap);
    let encoder = SingleBufferEncoder::new(&disp_new, raw_cap);
    eprintln!("running {} encode_first_half calls (new path)...", n_tokens);
    let t_new = std::time::Instant::now();
    for (t, x) in xs.iter().enumerate() {
        let _ = encoder
            .encode_first_half(
                x,
                &model,
                &mut state_new,
                LayerCutpoint(model.layers.len()),
            )
            .expect("encode_first_half");
        state_new.pos = state_new.pos.saturating_add(1);
        eprintln!(
            "  new token {} done, state.pos now {}",
            t, state_new.pos
        );
    }
    eprintln!(
        "encode_first_half (×{}) took {:?}",
        n_tokens,
        t_new.elapsed()
    );

    // ── state.cur_hc bit-close comparison at the final token.
    //   Bridge fires for tokens 3, 4, 5 on indexer layers (compress
    //   _ratio==4, V4 Flash layers 2-42). Any numerical drift from
    //   the bridge code path (incl. the GPU hc_collapse_norm port)
    //   accumulates through those tokens.
    assert_eq!(
        state_ref.cur_hc.len(),
        state_new.cur_hc.len(),
        "cur_hc length mismatch"
    );
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut nonzero = 0usize;
    for (a, b) in state_ref.cur_hc.iter().zip(state_new.cur_hc.iter()) {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if a.abs() > max_ref {
            max_ref = a.abs();
        }
        if a.abs() > 1e-8 {
            nonzero += 1;
        }
    }
    let rel = max_abs / max_ref.max(1e-6);
    eprintln!(
        "[multi-token] cur_hc compare after pos={}: len={} nonzero={} \
         max_abs={:.6} max_ref={:.6} rel={:.4}",
        state_ref.pos.saturating_sub(1),
        state_ref.cur_hc.len(),
        nonzero,
        max_abs,
        max_ref,
        rel,
    );
    assert!(nonzero > 0, "reference cur_hc is all-zero — model didn't run");
    // Same envelope as the single-token smoke. fp8 e4m3 propagation
    // grows with token count + layer depth; 6 tokens × 43 layers is
    // still within the 0.50 budget. If this test starts failing
    // it indicates a real divergence in the bridge code path.
    assert!(
        rel < 0.50,
        "encode_first_half multi-token diverged from decode_step beyond \
         expected fp8 envelope: max_abs={max_abs:.6}, max_ref={max_ref:.6}, \
         rel={rel:.4}"
    );
}

/// M5 Phase D review — sustained-throughput profile.
///
/// Drives `encode_first_half` for N tokens (default 100; override via
/// `DS4_PROFILE_N=`) and reports per-token wall + tok/s. Warmup phase
/// (the first `WARMUP_TOKENS`) is excluded from the steady-state timing
/// — that's where Metal pipeline specialization, residency setup, and
/// KV cache warm-in cost would otherwise distort the average.
///
/// Per-token timing per-iteration is printed for the steady-state phase
/// so the run-to-run variance is visible. The sustained tok/s reported
/// at the end is over steady-state tokens only.
///
/// Bit-equivalence is NOT checked here; the multi-token regression
/// above already covers that. This test purely measures wall time.
///
/// Opt-in: same DS4_GGUF= switch. Cost: N × ~0.4-0.5 s per token. At
/// the default N=100 that's roughly 45-60 s.
#[test]
fn encode_first_half_sustained_throughput_on_real_gguf() {
    let Some(path) = open_gguf_or_skip("encode_first_half throughput profile")
    else {
        return;
    };

    let n_tokens: u32 = std::env::var("DS4_PROFILE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    const WARMUP_TOKENS: u32 = 5;
    assert!(
        n_tokens > WARMUP_TOKENS,
        "DS4_PROFILE_N must be > {} (warmup tokens skipped from steady-state)",
        WARMUP_TOKENS
    );

    eprintln!("loading GGUF: {}", path.display());
    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights::from_views");
    eprintln!(
        "model loaded: d_model={} vocab={} n_layers={}",
        model.d_model,
        model.vocab_size,
        model.layers.len()
    );

    let raw_cap: u32 = 1024.max(n_tokens + 16);
    let d_model = model.d_model;

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&path).expect("open GGUF for expert weight load");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights");
    let mut state = AttnStepState::new(&model, raw_cap);
    let encoder = SingleBufferEncoder::new(&disp, raw_cap);

    let mut per_token_ms = Vec::with_capacity(n_tokens as usize);
    eprintln!("===== {} token decode (warmup={}) =====", n_tokens, WARMUP_TOKENS);

    let total_start = std::time::Instant::now();
    for t in 0..n_tokens {
        let x: Vec<f32> = (0..d_model)
            .map(|i| {
                ((i as f32 * 0.011 + t as f32 * 0.137).sin() * 0.4)
                    .clamp(-1.0, 1.0)
            })
            .collect();
        let t_start = std::time::Instant::now();
        let _ = encoder
            .encode_first_half(
                &x,
                &model,
                &mut state,
                LayerCutpoint(model.layers.len()),
            )
            .expect("encode_first_half");
        state.pos = state.pos.saturating_add(1);
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        per_token_ms.push(elapsed_ms);
        if t < WARMUP_TOKENS || t < 10 || t % 10 == 0 || t == n_tokens - 1 {
            eprintln!(
                "  t={:>3} elapsed={:>7.1} ms {}",
                t,
                elapsed_ms,
                if t < WARMUP_TOKENS { "(warmup)" } else { "" }
            );
        }
    }
    let total_elapsed = total_start.elapsed();

    let steady: &[f64] = &per_token_ms[(WARMUP_TOKENS as usize)..];
    let steady_sum: f64 = steady.iter().sum();
    let steady_avg = steady_sum / steady.len() as f64;
    let steady_min = steady.iter().cloned().fold(f64::INFINITY, f64::min);
    let steady_max = steady.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let steady_tok_per_s = 1000.0 / steady_avg;

    let warmup_sum: f64 = per_token_ms[..WARMUP_TOKENS as usize].iter().sum();
    let warmup_avg = warmup_sum / WARMUP_TOKENS as f64;

    eprintln!("===== profile summary =====");
    eprintln!("  total wall (incl. warmup): {:.2} s", total_elapsed.as_secs_f64());
    eprintln!(
        "  warmup tokens ({}): avg {:.1} ms/token",
        WARMUP_TOKENS, warmup_avg
    );
    eprintln!(
        "  steady-state tokens ({}): avg {:.1} ms/token (min {:.1}, max {:.1})",
        steady.len(),
        steady_avg,
        steady_min,
        steady_max,
    );
    eprintln!("  sustained tok/s: {:.2}", steady_tok_per_s);
    eprintln!("  antirez baseline on same rig+model: ~21 tok/s ({:.1}× behind)",
              21.21 / steady_tok_per_s);
}

/// FULL decode throughput: `decode_token_via_first_half_argmax` (43 layers +
/// final_norm + lm_head matvec d_model→vocab + GPU argmax). The companion to
/// `encode_first_half_sustained_throughput` (layers only) — the delta between
/// the two is the lm-head tail. Set `DS4_TAIL_PROFILE=1` for the per-token
/// layers-vs-tail split. Opt-in: same DS4_GGUF= switch.
#[test]
fn full_decode_argmax_throughput_on_real_gguf() {
    let Some(path) = open_gguf_or_skip("full-decode argmax throughput") else {
        return;
    };
    let n_tokens: u32 = std::env::var("DS4_PROFILE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    const WARMUP: u32 = 5;
    assert!(n_tokens > WARMUP, "DS4_PROFILE_N must be > {}", WARMUP);

    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights::from_views");
    let raw_cap: u32 = 1024.max(n_tokens + 16);
    let d_model = model.d_model;

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&path).expect("open GGUF for expert weight load");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights");
    let mut state = AttnStepState::new(&model, raw_cap);
    let encoder = SingleBufferEncoder::new(&disp, raw_cap);

    eprintln!(
        "===== FULL DECODE ({} tok, warmup={}, vocab={}) =====",
        n_tokens, WARMUP, model.vocab_size
    );
    let mut per_token_ms = Vec::with_capacity(n_tokens as usize);
    let mut tokens = Vec::with_capacity(n_tokens as usize);
    for t in 0..n_tokens {
        let x: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32 * 0.011 + t as f32 * 0.137).sin() * 0.4).clamp(-1.0, 1.0))
            .collect();
        let t0 = std::time::Instant::now();
        let tok = encoder
            .decode_token_via_first_half_argmax(&x, &model, &mut state)
            .expect("decode_token_via_first_half_argmax");
        state.pos = state.pos.saturating_add(1);
        per_token_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        tokens.push(tok);
    }
    // token sequence (deterministic fn of synthetic hidden → lm_head) — used to
    // A/B the q8 vs f32 lm-head tail for token-identity.
    let checksum: i64 = tokens.iter().map(|&t| t as i64).sum();
    eprintln!("  tokens[..8]={:?}  checksum={}", &tokens[..8.min(tokens.len())], checksum);
    let steady = &per_token_ms[WARMUP as usize..];
    let avg = steady.iter().sum::<f64>() / steady.len() as f64;
    let mn = steady.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "  steady-state ({}): avg {:.1} ms/token (min {:.1})",
        steady.len(),
        avg,
        mn
    );
    eprintln!("  FULL-decode tok/s: {:.2}", 1000.0 / avg);
}
