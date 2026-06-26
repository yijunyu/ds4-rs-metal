//! Phase E M5.4.5.bench — token-level bench of
//! `MetalDispatcher::decode_token_unified` vs the trait-dispatch
//! `decode_step_with_attn` reference.
//!
//! Opt-in: set `DS4_GGUF=/path/to/ds4flash.gguf` to run. Without it
//! the test prints a skip and passes.
//!
//! For each of N tokens (set `DS4_BENCH_TOKENS=K`, default 2):
//!   - Reference: `decode_step_with_attn(&disp, &disp, x, ...)` →
//!     logits → `sample_argmax` → token id
//!   - Unified:   `disp.decode_token_unified(x, ...)` (routes through
//!     `encode_first_half` + output_hc_head_one + tail_lm_head_batched)
//!     → token id
//!
//! Prints per-step timings, total wall, tok/s for both paths, the
//! speedup ratio, and a top-K logit overlap metric (informational —
//! argmax-equality requires kv_fp8_store CPU/GPU unification per
//! `project_m5_kv_fp8_store_divergence`).
//!
//! **Asserts** (correctness bars that DO hold today):
//!   - Both paths complete without panicking on a real DS4 V4 Flash
//!     model across N steps. This validates that `encode_first_half`
//!     handles real production weights end-to-end for multi-step
//!     decode with state persistence between tokens.
//!   - Logit vectors are finite (no NaN/Inf).
//!
//! **Does NOT assert**:
//!   - argmax equality — pending M5.2.4 (kv_fp8_store CPU/GPU
//!     unification); printed for inspection.
//!   - tok/s speedup — current `encode_first_half` doesn't yet collapse
//!     per-layer cb count below the trait path. The architectural
//!     win is M5.4.5.unify (single-cb-per-token rearchitecture). This
//!     bench is the measurement surface for tracking progress.
//!
//! Memory cost: full DS4 V4 Flash model load (~10 GB) + N decode steps.
//! Designed to run sparingly — opt-in only.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::{
    decode_step_with_attn, sample_argmax, AttnStepState, ComposedModelWeights, DecodeConfig,
};
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
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
fn decode_token_unified_multi_step_bench() {
    let Some(path) = open_gguf_or_skip("decode_token_unified token-argmax bench") else {
        return;
    };

    let n_tokens: u32 = std::env::var("DS4_BENCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

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

    let mut disp_ref = MetalDispatcher::new().expect("MetalDispatcher::new (reference)");
    let gguf = GgufFile::open(&path).expect("open GGUF for expert weight load");
    disp_ref
        .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (reference)");
    let mut disp_uni = MetalDispatcher::new().expect("MetalDispatcher::new (unified)");
    disp_uni
        .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (unified)");

    // Task #82: page-warm experiment — measured to make the bench
    // SLOWER, not faster, despite forcing 77.91GB of mmap pages
    // resident in 7.7s. Conclusion: memory residency is NOT the
    // dominant source of moe_shared_chain's per-call wait variance.
    // Disabled by default; opt in with DS4_WARM_PAGES=1.
    //
    // Notable side effect when enabled: step 0 argmax agreed between
    // ref and uni for the first time (an M5.2.4+M5.2.5+M5.2.6 stack
    // win surfaced because warm-up gave both dispatchers an identical
    // page-residency starting state). Wall time still regressed.
    //
    // The `warm_up_expert_pages` API stays available on
    // `MetalDispatcher` for diagnostic use; the bench just doesn't
    // call it by default.
    if std::env::var("DS4_WARM_PAGES").ok().as_deref() == Some("1") {
        let t_warm = std::time::Instant::now();
        let warmed_ref = disp_ref.warm_up_expert_pages();
        let warmed_uni = disp_uni.warm_up_expert_pages();
        eprintln!(
            "warmed expert pages: ref={:.2}GB uni={:.2}GB in {:?}",
            warmed_ref as f64 / 1e9,
            warmed_uni as f64 / 1e9,
            t_warm.elapsed()
        );
    }

    // Phase F task #90 — GPU warm-up via kernel_touch_u8_stride (antirez
    // ds4_metal.m:657). Replaces the broken CPU warm-up from #82: instead
    // of CPU reads (which evict other pages), we dispatch one byte-touch
    // per stride from the GPU itself, mirroring antirez's warm_model_views.
    // Stride is configurable via DS4_METAL_MODEL_WARMUP_STRIDE_MB (matches
    // antirez env var); default 1 MiB.
    if std::env::var("DS4_WARM_PAGES_GPU").ok().as_deref() == Some("1") {
        let stride_mb = std::env::var("DS4_METAL_MODEL_WARMUP_STRIDE_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1);
        let stride_bytes = stride_mb * 1024 * 1024;
        let t_warm = std::time::Instant::now();
        let warmed_ref = disp_ref
            .warm_up_expert_pages_gpu(stride_bytes)
            .expect("warm_up_expert_pages_gpu (ref)");
        let warmed_uni = disp_uni
            .warm_up_expert_pages_gpu(stride_bytes)
            .expect("warm_up_expert_pages_gpu (uni)");
        eprintln!(
            "gpu-warmed expert pages: ref={:.2}GB uni={:.2}GB stride={}MB in {:?}",
            warmed_ref as f64 / 1e9,
            warmed_uni as f64 / 1e9,
            stride_mb,
            t_warm.elapsed()
        );
    }

    let raw_cap: u32 = 256;
    let cfg = DecodeConfig::default();
    let d_model = model.d_model;

    let mut state_ref = AttnStepState::new(&model, raw_cap);
    let mut state_uni = AttnStepState::new(&model, raw_cap);

    // Phase F task #87 — opt-in `DS4_BENCH_POS_SEED=N` seeds both
    // states' `pos` and `kv_pos` to N and prefills `kv_storage` /
    // persistent KV buffers with N rows of deterministic data per
    // layer. This makes the FIRST timed decode step see n_raw=N+1;
    // for N=31 that's 32 → the encode_first_half fast-path
    // (n_raw % 32 == 0, > 0) fires immediately on dense/hash-routing
    // layers (DS4 V4 Flash layers 0/1). Without seeding, n_raw starts
    // at 1 and only crosses 32 after 31 real decode steps (~hour at
    // current speed), so the fast path is never measured.
    //
    // Skips compressor/indexer layers (compress_ratio > 1 → fast
    // path gated on `comp_*.is_none()`). Only layers 0/1 of DS4 V4
    // Flash benefit; remaining layers stay on the slow path.
    let pos_seed: u32 = std::env::var("DS4_BENCH_POS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if pos_seed > 0 {
        eprintln!(
            "seeding state.pos={pos_seed} and prefilling {pos_seed} rows of \
             kv_storage + persistent KV per layer"
        );
        let n_layers = model.layers.len();
        for layer_idx in 0..n_layers {
            let p = &model.layers[layer_idx].attn.params;
            let row = p.n_lora_kv as usize;
            let prefill_rows = pos_seed as usize;
            // Deterministic per-(layer, slot, lane) bytes; the EXACT
            // values don't matter for perf — only that ref and uni
            // see byte-identical data.
            let prefill: Vec<f32> = (0..prefill_rows * row)
                .map(|i| ((layer_idx as f32) * 0.13 + (i as f32) * 0.0011).sin() * 0.05)
                .collect();
            // CPU KV mirror used by the trait `flash_attn_decode`.
            state_ref.kv_storage[layer_idx][..prefill.len()].copy_from_slice(&prefill);
            state_uni.kv_storage[layer_idx][..prefill.len()].copy_from_slice(&prefill);
            state_ref.kv_pos[layer_idx] = pos_seed;
            state_uni.kv_pos[layer_idx] = pos_seed;
            // Persistent buffers used by `flash_attn_decode_persistent`.
            let cache_byte_len =
                (raw_cap as usize) * row * std::mem::size_of::<f32>();
            for d in [&disp_ref, &disp_uni] {
                let buf = d.kv_buffer_or_alloc(layer_idx as u32, cache_byte_len);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.as_ptr(),
                        buf.contents() as *mut f32,
                        prefill.len(),
                    );
                }
            }
        }
        state_ref.pos = pos_seed;
        state_uni.pos = pos_seed;
    }

    let mut total_ref_us: u128 = 0;
    let mut total_uni_us: u128 = 0;
    let mut agreed = 0u32;
    let mut argmax_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_tokens as usize);

    // Phase F task #88 — `DS4_BENCH_UNI_FIRST=1` swaps the per-step
    // order so uni runs BEFORE ref. Useful for diagnosing the
    // [[m5-bench-order-artifact]] — moe wait grows monotonically with
    // cumulative cb count, so whichever path runs first gets the fast
    // moe regime and whichever runs second pays the saturated wait.
    let uni_first = std::env::var("DS4_BENCH_UNI_FIRST").ok().as_deref() == Some("1");

    for step in 0..n_tokens {
        // Deterministic per-step input (in production this would be
        // the previous step's argmax token's embedding; here we just
        // need both paths to see the same input).
        let x: Vec<f32> = (0..d_model)
            .map(|i| (((i as f32) * 0.011 + (step as f32) * 0.07).sin() * 0.4).clamp(-1.0, 1.0))
            .collect();

        let (logits_ref, ref_us, logits_uni, uni_us) = if uni_first {
            let t_uni = std::time::Instant::now();
            let logits_uni = disp_uni
                .decode_token_unified(x.clone(), &model, &mut state_uni, raw_cap)
                .expect("decode_token_unified");
            let uni_us = t_uni.elapsed().as_micros();
            let t_ref = std::time::Instant::now();
            let logits_ref = decode_step_with_attn(
                &disp_ref,
                &disp_ref,
                x,
                &model,
                &mut state_ref,
                &cfg,
                raw_cap,
            )
            .expect("decode_step_with_attn");
            let ref_us = t_ref.elapsed().as_micros();
            (logits_ref, ref_us, logits_uni, uni_us)
        } else {
            let t_ref = std::time::Instant::now();
            let logits_ref = decode_step_with_attn(
                &disp_ref,
                &disp_ref,
                x.clone(),
                &model,
                &mut state_ref,
                &cfg,
                raw_cap,
            )
            .expect("decode_step_with_attn");
            let ref_us = t_ref.elapsed().as_micros();
            let t_uni = std::time::Instant::now();
            let logits_uni = disp_uni
                .decode_token_unified(x, &model, &mut state_uni, raw_cap)
                .expect("decode_token_unified");
            let uni_us = t_uni.elapsed().as_micros();
            (logits_ref, ref_us, logits_uni, uni_us)
        };

        total_ref_us += ref_us;
        total_uni_us += uni_us;
        let argmax_ref = sample_argmax(&logits_ref);
        let argmax_uni = sample_argmax(&logits_uni);

        // Sanity: both paths produce finite logits.
        assert!(
            logits_ref.iter().all(|v| v.is_finite()),
            "reference logits contain non-finite at step {step}"
        );
        assert!(
            logits_uni.iter().all(|v| v.is_finite()),
            "unified logits contain non-finite at step {step}"
        );

        argmax_pairs.push((argmax_ref, argmax_uni));
        if argmax_ref == argmax_uni {
            agreed += 1;
        }
        eprintln!(
            "step {step}: ref_argmax={argmax_ref} uni_argmax={argmax_uni} \
             ref_ms={:.1} uni_ms={:.1}",
            ref_us as f64 / 1000.0,
            uni_us as f64 / 1000.0,
        );

        // Phase F task #94 — dump logit distribution + top-5 for both
        // paths so we can tell "uni is degenerate" from "uni is noisy".
        // Argmax pairs like (11621, 33) could mean uni's vector is
        // dominated by a few small indices (broken) OR is fine but
        // happens to peak elsewhere (just noisy across 43 layers).
        let summarize = |label: &str, lg: &[f32]| {
            let mut top: Vec<(f32, usize)> =
                lg.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = top.iter().take(5)
                .map(|(v, i)| format!("{i}:{v:+.3}"))
                .collect();
            let n = lg.len() as f64;
            let mean: f64 = lg.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var: f64 = lg.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let mn = lg.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "  {label}: top5=[{}] | mean={mean:+.3} std={:.3} min={mn:+.3} max={mx:+.3}",
                top5.join(" "), var.sqrt(),
            );
        };
        summarize("ref", &logits_ref);
        summarize("uni", &logits_uni);
        // Cosine similarity between the two logit vectors.
        let dot: f64 = logits_ref.iter().zip(&logits_uni)
            .map(|(&a, &b)| a as f64 * b as f64).sum();
        let na: f64 = logits_ref.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = logits_uni.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let cos = if na * nb > 0.0 { dot / (na * nb) } else { 0.0 };
        eprintln!("  cos(ref,uni)={cos:+.4}");
    }

    eprintln!(
        "totals: ref={:.2}s uni={:.2}s ref_tok/s={:.3} uni_tok/s={:.3} \
         speedup={:.2}× agreed={}/{}",
        total_ref_us as f64 / 1e6,
        total_uni_us as f64 / 1e6,
        n_tokens as f64 * 1e6 / total_ref_us as f64,
        n_tokens as f64 * 1e6 / total_uni_us as f64,
        total_ref_us as f64 / total_uni_us as f64,
        agreed,
        n_tokens
    );
    eprintln!("argmax_pairs: {:?}", argmax_pairs);

    // Phase F task #89 — buffer audit. Reports the MTLBuffer count
    // each dispatcher has materialized after a full decode step. Used
    // to compare against antirez's 1-2 chunked-view design (per
    // [[m5-antirez-insights]] §(2)).
    for (label, disp) in [("ref", &disp_ref), ("uni", &disp_uni)] {
        let r = disp.buffer_audit();
        eprintln!(
            "buffer_audit[{label}]: total={} bytes={:.2}GB | expert={}/{:.2}GB \
             weight_cache={}/{:.2}GB kv_cache={}/{:.2}GB moe_scratch={} (pools={})",
            r.total_buffers(),
            r.total_bytes() as f64 / 1e9,
            r.n_expert_buffers,
            r.expert_bytes_total as f64 / 1e9,
            r.n_weight_cache_buffers,
            r.weight_cache_bytes_total as f64 / 1e9,
            r.n_kv_cache_buffers,
            r.kv_cache_bytes_total as f64 / 1e9,
            r.n_moe_scratch_buffers,
            r.n_moe_scratch_pools,
        );
        // Phase F task #93 — zero-copy vs memcpy breakdown.
        eprintln!(
            "weight_cache[{label}]: no_copy={}/{:.2}GB memcpy={}/{:.2}GB \
             (no_copy saves duplicate RAM)",
            r.n_weight_no_copy,
            r.weight_no_copy_bytes as f64 / 1e9,
            r.n_weight_memcpy,
            r.weight_memcpy_bytes as f64 / 1e9,
        );
    }
    // Informational only: argmax-equality is pending M5.2.4 (kv_fp8_store
    // CPU/GPU unification — see project_m5_kv_fp8_store_divergence). The
    // 23% rel error per token measured by the encode_first_half_gguf_smoke
    // is large enough to flip logit top dimensions in a 129k vocab. The
    // bench prints the pairs above for human inspection; future hardening
    // (after M5.2.4) will flip this into an assertion.
}
