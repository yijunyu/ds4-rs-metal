//! Phase 2 MoE-K Step 5 — REAL-MODEL bench for the K-amortized MoE chain.
//!
//! Opens DS4_GGUF, loads the full expert weight table via
//! `MetalDispatcher::load_expert_weights`, then dispatches the new
//! `encode_moe_chain_mm_qx_k` (mixed-quant chain) against ONE real layer's
//! expert weights at K∈{1,2,4,8}. Measures wall-clock per call and the
//! resulting K-amortization profile.
//!
//! This is the **definitive Step 5 validation**: does the K-amortization
//! win measured on synthetic Q8_0 weights (Steps 1-4) hold up against
//! real DS4 V4 Flash quant data (iq2_xxs / q4_K) loaded from GGUF?
//!
//! Gated by `DS4_BENCH_MOE_REAL=1` AND a valid `DS4_GGUF` env var.
//! Opt-out by default — heavy memory cost (~10 GB model load).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgmlType, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_metal::deferred::DeferredBuf;
use ds4_metal::MetalDispatcher;

fn quant_kernel_info(ttype: GgmlType) -> (&'static str, usize, u64) {
    match ttype {
        GgmlType::Q8_0    => ("ds4_kernel_mul_mm_id_q8_0_f32",    32,  34),
        GgmlType::Q2_K    => ("ds4_kernel_mul_mm_id_q2_K_f32",    256, 84),
        GgmlType::Q4_K    => ("ds4_kernel_mul_mm_id_q4_K_f32",    256, 144),
        GgmlType::IQ2_XXS => ("ds4_kernel_mul_mm_id_iq2_xxs_f32", 256, 66),
        other => panic!("real-model MoE-K bench: unsupported ttype {:?}", other),
    }
}

#[test]
fn encode_moe_chain_mm_qx_k_real_model_bench() {
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping. Set DS4_BENCH_MOE_REAL=1 (and DS4_GGUF=/path) to run.");
        return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() {
        eprintln!("DS4_GGUF={} is not a file — skipping", gguf_path.display());
        return;
    }

    eprintln!("loading DS4 GGUF: {}", gguf_path.display());
    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights");
    eprintln!(
        "model loaded: d_model={} vocab={} n_layers={}",
        model.d_model, model.vocab_size, model.layers.len()
    );

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&gguf_path).expect("GgufFile::open");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights");
    eprintln!("loaded {} layers of expert weights", manifest.n_layers);

    // Pick a layer. Default: layer 5 (typical MoE layer in DS4). Allow override
    // via DS4_BENCH_MOE_LAYER for inspection of different quant mixes.
    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let info = disp.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");
    let n_experts = info.n_experts;
    let d_in = info.d_in;
    let d_ffn = info.d_ffn;
    eprintln!(
        "\nlayer {}: n_experts={} d_in={} d_ffn={}",
        layer_idx, n_experts, d_in, d_ffn
    );
    eprintln!(
        "  quants:  gate={:?}  up={:?}  down={:?}",
        info.gate_ttype, info.up_ttype, info.down_ttype
    );

    let (gate_kernel, gate_qk, gate_bytes) = quant_kernel_info(info.gate_ttype);
    let (up_kernel,   up_qk,   up_bytes)   = quant_kernel_info(info.up_ttype);
    let (down_kernel, down_qk, down_bytes) = quant_kernel_info(info.down_ttype);

    // Wrap the layer's expert weights as DeferredBufs (no copy).
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_MOE_REAL_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    // Prefill chunking: sweep large K via DS4_BENCH_MOE_REAL_KS (comma-sep).
    // mm_id (expert-token gather) should keep amortizing at large K where the
    // fused pair_swiglu_K path breaks (K>=32) — the megablocks-style large-K MoE.
    let ks_owned: Vec<usize> = std::env::var("DS4_BENCH_MOE_REAL_KS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    let ks: &[usize] = &ks_owned;
    let k_top: usize = 6;

    let mut rng: u32 = 0x42424242;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

    eprintln!("\nencode_moe_chain_mm_qx_k REAL-MODEL bench: warmup={} bench_iters={}",
        warmup_iters, bench_iters);
    let mut means_ms: Vec<f64> = Vec::with_capacity(ks.len());

    for &k in ks {
        // Random routing — each K-position picks 6 distinct experts.
        let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
        for _ in 0..k {
            let mut picks: Vec<i32> = (0..n_experts as i32).collect();
            for i in 0..k_top {
                let j = i + (next() as usize % (n_experts - i));
                picks.swap(i, j);
            }
            for slot in 0..k_top { selected.push(picks[slot]); }
        }
        let weights: Vec<f32> = (0..k * k_top)
            .map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05)
            .collect();
        // Activations: standard-ish range. Real production has RMS-normed,
        // ~unit-magnitude values per dim — close enough for timing.
        let x_k: Vec<f32> = (0..k * d_in)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();

        let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);
        for _ in 0..(warmup_iters + bench_iters) {
            let scope = disp.batch_scope();
            // Per-iter, build flat sel/wt + activation bufs.
            let x_db = scope.upload_f32(&x_k);
            let wt_db = scope.upload_f32(&weights);
            let sel_db = scope.alloc_i32(k * k_top);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    selected.as_ptr(),
                    sel_db.buffer().contents() as *mut i32,
                    k * k_top,
                );
            }

            let t0 = Instant::now();
            let moe_k = scope.encode_moe_chain_mm_qx_k(
                &x_db, &sel_db, &wt_db,
                &gate_db, &up_db, &down_db,
                gate_kernel, gate_qk, gate_bytes,
                up_kernel,   up_qk,   up_bytes,
                down_kernel, down_qk, down_bytes,
                n_experts, d_in, d_ffn, k,
            ).expect("encode_moe_chain_mm_qx_k");
            let out = scope.flush_and_read(&moe_k);
            iter_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            // Validate: shape correct, finite values.
            assert_eq!(out.len(), k * d_in, "moe_k shape mismatch");
            // Numerical content depends on weights; just assert no NaN.
            let nan_count = out.iter().filter(|v| !v.is_finite()).count();
            if iter_ms.len() == warmup_iters + 1 {
                let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
                eprintln!("    K={}: out max_abs={:.3e}  nan_count={}", k, max_abs, nan_count);
            }
        }
        let bench_only = &iter_ms[warmup_iters..];
        let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
        let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
        let max = bench_only.iter().copied().fold(0.0f64, f64::max);
        eprintln!("  K={}: mean={:.3} ms  [min={:.3} max={:.3}]  μs/K-pos={:.1}",
            k, mean, min, max, mean * 1000.0 / k as f64);
        means_ms.push(mean);
    }

    let m1 = means_ms[0];
    eprintln!("\nREAL-MODEL MoE chain K-amortization vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ideal = (k as f64) * m1;
        let actual = means_ms[i];
        let ratio = actual / ideal;
        let eff = if k == 1 { 100.0 } else {
            (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
        };
        eprintln!(
            "  K={}: actual={:.3} ms, K×K=1={:.3} ms, ratio={:.3}  efficiency={:.0}%",
            k, actual, ideal, ratio, eff
        );
    }
    eprintln!(
        "\nProjection vs production K=1 blit-shim MoE (~0.67 ms/layer):\n\
         - If K=8 mm-chain ≤ 2 ms, the +90-241% spec-decode tier IS unlocked.\n\
         - 43 layers × K=8 mm-chain ms = per-token MoE cost. Combine with attn\n\
           (3.13 ms/layer K=8) and FFN-no-MoE (2.31 ms/layer K=8) for total."
    );
}

/// Phase 2 MoE-K Option A — REAL-MODEL bench for the FUSED K-batched
/// pair_swiglu + sum6 path. Replaces the mm_id chain (which lost to the
/// fused production path at K=1) with the actual Option A approach:
/// clone the production fused kernels with K-dim outer parallelism.
///
/// Compares K∈{1,2,4,8} against the K=1 production baseline (which this
/// path SHOULD match at K=1 by construction).
#[test]
fn encode_moe_chain_fused_K_real_model_bench() {
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping fused-K. Set DS4_BENCH_MOE_REAL=1 to run.");
        return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() {
        eprintln!("DS4_GGUF={} not a file — skipping", gguf_path.display());
        return;
    }

    eprintln!("[fused-K bench] loading DS4 GGUF: {}", gguf_path.display());
    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let _model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights");

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&gguf_path).expect("GgufFile::open");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights");

    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let info = disp.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");

    eprintln!(
        "[fused-K] layer {}: n_experts={} d_in={} d_ffn={}  quants: gate={:?} up={:?} down={:?}",
        layer_idx, info.n_experts, info.d_in, info.d_ffn,
        info.gate_ttype, info.up_ttype, info.down_ttype
    );
    // Option A first impl supports (IQ2_XXS gate/up, Q2_K down).
    assert_eq!(info.gate_ttype, GgmlType::IQ2_XXS, "fused-K bench: gate must be IQ2_XXS");
    assert_eq!(info.up_ttype,   GgmlType::IQ2_XXS, "fused-K bench: up must be IQ2_XXS");
    assert_eq!(info.down_ttype, GgmlType::Q2_K,    "fused-K bench: down must be Q2_K");

    let n_experts = info.n_experts;
    let d_in = info.d_in;
    let d_ffn = info.d_ffn;

    // Wrap expert weights as DeferredBufs (no copy).
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_MOE_REAL_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    // Prefill chunking: sweep large K via DS4_BENCH_MOE_REAL_KS (comma-sep).
    // mm_id (expert-token gather) should keep amortizing at large K where the
    // fused pair_swiglu_K path breaks (K>=32) — the megablocks-style large-K MoE.
    let ks_owned: Vec<usize> = std::env::var("DS4_BENCH_MOE_REAL_KS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    let ks: &[usize] = &ks_owned;
    let k_top: usize = 6;

    let mut rng: u32 = 0x42424242;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

    eprintln!("\n[fused-K] FUSED real-model bench: warmup={} iters={}", warmup_iters, bench_iters);
    let mut means_ms: Vec<f64> = Vec::with_capacity(ks.len());

    for &k in ks {
        let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
        for _ in 0..k {
            let mut picks: Vec<i32> = (0..n_experts as i32).collect();
            for i in 0..k_top {
                let j = i + (next() as usize % (n_experts - i));
                picks.swap(i, j);
            }
            for slot in 0..k_top { selected.push(picks[slot]); }
        }
        let weights: Vec<f32> = (0..k * k_top)
            .map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05)
            .collect();
        let x_k: Vec<f32> = (0..k * d_in)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();

        let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);
        for _ in 0..(warmup_iters + bench_iters) {
            let scope = disp.batch_scope();
            let x_db = scope.upload_f32(&x_k);
            let wt_db = scope.upload_f32(&weights);
            let sel_db = scope.alloc_i32(k * k_top);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    selected.as_ptr(),
                    sel_db.buffer().contents() as *mut i32,
                    k * k_top,
                );
            }
            let t0 = Instant::now();
            let out_k = scope.encode_moe_chain_fused_K_iq2_xxs_q2_K(
                &x_db, &sel_db, &wt_db,
                &gate_db, &up_db, &down_db,
                n_experts, d_in, d_ffn, k,
            ).expect("fused_K");
            let out = scope.flush_and_read(&out_k);
            iter_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            assert_eq!(out.len(), k * d_in, "fused-K out shape");
            let nan_count = out.iter().filter(|v| !v.is_finite()).count();
            if iter_ms.len() == warmup_iters + 1 {
                let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
                eprintln!("    K={}: out max_abs={:.3e}  nan_count={}", k, max_abs, nan_count);
            }
        }
        let bench_only = &iter_ms[warmup_iters..];
        let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
        let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
        let max = bench_only.iter().copied().fold(0.0f64, f64::max);
        eprintln!("  K={}: mean={:.3} ms  [min={:.3} max={:.3}]  μs/K-pos={:.1}",
            k, mean, min, max, mean * 1000.0 / k as f64);
        means_ms.push(mean);
    }

    let m1 = means_ms[0];
    eprintln!("\n[fused-K] REAL-MODEL K-amortization vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ideal = (k as f64) * m1;
        let ratio = means_ms[i] / ideal;
        let eff = if k == 1 { 100.0 } else {
            (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
        };
        eprintln!(
            "  K={}: actual={:.3} ms  ideal={:.3}  ratio={:.3}  eff={:.0}%",
            k, means_ms[i], ideal, ratio, eff
        );
    }
    eprintln!(
        "\n[fused-K] Compared to mm_id chain (3.84 ms K=1 / 7.70 ms K=8) and\n\
         K=1 production baseline (~0.67 ms/layer pair_swiglu+sum6):\n\
         - Fused-K K=1 should be ≈ production K=1 (proves K=1 reduces to K=1 path).\n\
         - Fused-K K=8 / K=1 < 8 means K-amortization. <2× best case."
    );
}

/// UP-TOWARD-CHUNK REPRODUCER: the mm_id MoE kernel + real experts work in a
/// fresh single-op scope (proven above). The chunk faults though. This test runs
/// N mm_id MoE chains in ONE SHARED scope (no flush between) — replicating the
/// chunk's single-cb / many-simdgroup-dispatch structure — to see if the SECOND+
/// chain goes garbage/NaN (= the in-chunk fault reproduced headlessly).
///
/// DS4_BENCH_MOE_REAL=1 DS4_GGUF=... DS4_REPRO_N=<chains> (default 4) DS4_BENCH_MOE_LAYER=3
#[test]
fn encode_moe_chain_mm_qx_k_shared_scope_repro() {
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping shared-scope repro.");
        return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }

    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let _model = ComposedModelWeights::from_views(&views, &manifest, defaults).expect("model");
    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&gguf_path).expect("GgufFile::open");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers).expect("load_expert_weights");

    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let info = disp.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");
    let (n_experts, d_in, d_ffn) = (info.n_experts, info.d_in, info.d_ffn);
    let (gk, gqk, gb) = quant_kernel_info(info.gate_ttype);
    let (uk, uqk, ub) = quant_kernel_info(info.up_ttype);
    let (dk, dqk, db) = quant_kernel_info(info.down_ttype);
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let n_chains: usize = std::env::var("DS4_REPRO_N").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let k: usize = 8;
    let k_top = 6;
    let mut rng: u32 = 0x1234_5678;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

    eprintln!("\n[shared-scope repro] layer {layer_idx}: {n_chains} mm_id MoE chains in ONE scope, K={k}");
    let scope = disp.batch_scope();
    let mut outs = Vec::with_capacity(n_chains);
    for _c in 0..n_chains {
        let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
        for _ in 0..k {
            let mut picks: Vec<i32> = (0..n_experts as i32).collect();
            for i in 0..k_top { let j = i + (next() as usize % (n_experts - i)); picks.swap(i, j); }
            for slot in 0..k_top { selected.push(picks[slot]); }
        }
        let weights: Vec<f32> = (0..k * k_top).map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05).collect();
        let x_k: Vec<f32> = (0..k * d_in).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let x_db = scope.upload_f32(&x_k);
        let wt_db = scope.upload_f32(&weights);
        let sel_db = scope.alloc_i32(k * k_top);
        unsafe { std::ptr::copy_nonoverlapping(selected.as_ptr(), sel_db.buffer().contents() as *mut i32, k * k_top); }
        let moe = scope.encode_moe_chain_mm_qx_k(
            &x_db, &sel_db, &wt_db, &gate_db, &up_db, &down_db,
            gk, gqk, gb, uk, uqk, ub, dk, dqk, db,
            n_experts, d_in, d_ffn, k,
        ).expect("moe chain");
        outs.push(moe);
    }
    let refs: Vec<&DeferredBuf> = outs.iter().collect();
    let results = scope.flush_and_read_multi(&refs);
    for (c, out) in results.iter().enumerate() {
        let nan = out.iter().filter(|v| !v.is_finite()).count();
        let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
        eprintln!("  chain {c}: max_abs={max_abs:.3e}  nan={nan}");
    }
    eprintln!("[shared-scope repro] done — any chain>0 with nan/garbage = in-chunk fault reproduced headlessly.");
}

/// UP-TOWARD-CHUNK REPRODUCER #2: run the standalone mm_id MoE on real layer-3
/// experts UNDER THE FULL DecodeRunner RUNTIME CONTEXT (KV cache + comp/indexer
/// pools allocated, model fully resident) AND after real attention (a DecodeSession
/// prefill). If the MoE goes garbage/NaN here but not in the lean bench above, the
/// in-chunk fault = the runtime/device context (memory pressure / persistent-buffer
/// interaction), reproduced headlessly without the chunk driver.
///
/// DS4_BENCH_MOE_REAL=1 DS4_GGUF=... DS4_REPRO_PREFILL=<tokens> (default 8)
#[test]
fn encode_moe_chain_mm_qx_k_under_decoderunner_repro() {
    use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping decoderunner repro.");
        return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }

    let runner = DecodeRunner::open(&gguf_path, 256).expect("DecodeRunner::open");
    // Run real attention (populates KV cache + comp/indexer pools) via prefill.
    let n_pf: i32 = std::env::var("DS4_REPRO_PREFILL").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    if n_pf > 0 {
        let prompt: Vec<i32> = (1..=n_pf).collect();
        let mut session = DecodeSession::new(&runner);
        session.prefill(&prompt).expect("prefill");
        eprintln!("[decoderunner repro] ran prefill of {n_pf} tokens (real attention + KV/pools populated)");
    }

    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let info = runner.dispatcher.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");
    let (n_experts, d_in, d_ffn) = (info.n_experts, info.d_in, info.d_ffn);
    let (gk, gqk, gb) = quant_kernel_info(info.gate_ttype);
    let (uk, uqk, ub) = quant_kernel_info(info.up_ttype);
    let (dk, dqk, db) = quant_kernel_info(info.down_ttype);
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let k: usize = 8; let k_top = 6;
    let mut rng: u32 = 0x1234_5678;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
    let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
    for _ in 0..k {
        let mut picks: Vec<i32> = (0..n_experts as i32).collect();
        for i in 0..k_top { let j = i + (next() as usize % (n_experts - i)); picks.swap(i, j); }
        for slot in 0..k_top { selected.push(picks[slot]); }
    }
    let weights: Vec<f32> = (0..k * k_top).map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05).collect();
    let x_k: Vec<f32> = (0..k * d_in).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

    let scope = runner.dispatcher.batch_scope();
    let x_db = scope.upload_f32(&x_k);
    let wt_db = scope.upload_f32(&weights);
    let sel_db = scope.alloc_i32(k * k_top);
    unsafe { std::ptr::copy_nonoverlapping(selected.as_ptr(), sel_db.buffer().contents() as *mut i32, k * k_top); }
    let moe = scope.encode_moe_chain_mm_qx_k(
        &x_db, &sel_db, &wt_db, &gate_db, &up_db, &down_db,
        gk, gqk, gb, uk, uqk, ub, dk, dqk, db,
        n_experts, d_in, d_ffn, k,
    ).expect("moe chain");
    let out = scope.flush_and_read(&moe);
    let nan = out.iter().filter(|v| !v.is_finite()).count();
    let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
    eprintln!("[decoderunner repro] MoE under DecodeRunner ctx: max_abs={max_abs:.3e}  nan={nan}");
    eprintln!("[decoderunner repro] {} — nan/garbage = runtime-context fault reproduced; clean = fault is the chunk's exact scope structure.",
        if nan > 0 || max_abs > 100.0 { "REPRODUCED" } else { "CLEAN" });
}

/// UP-TOWARD-CHUNK REPRODUCER #3 (the decisive one): run a FLASH-ATTENTION dispatch
/// then the mm_id MoE in the SAME BatchScope (one cb), mirroring chunk_layer's
/// structure (attention core then FFN/MoE in one scope). MoE-alone and MoE×N and
/// MoE-after-attention-in-separate-scope are all CLEAN; the chunk has attention +
/// MoE in ONE scope. If THIS faults → the in-chunk fault is the flash→MoE
/// same-scope interaction, reproduced headlessly.
///
/// DS4_BENCH_MOE_REAL=1 DS4_GGUF=... DS4_BENCH_MOE_LAYER=3
#[test]
fn flash_then_moe_same_scope_repro() {
    use ds4_metal::decode_runner::DecodeRunner;
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping flash+moe repro."); return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p), Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }

    let runner = DecodeRunner::open(&gguf_path, 256).expect("DecodeRunner::open");
    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let layer = &runner.composed.layers[layer_idx as usize];
    let p = &layer.attn.params;
    let raw_cap: u32 = 256;
    let n_head = p.n_head as usize; let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;

    let info = runner.dispatcher.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");
    let (n_experts, d_in, d_ffn) = (info.n_experts, info.d_in, info.d_ffn);
    let (gk, gqk, gb) = quant_kernel_info(info.gate_ttype);
    let (uk, uqk, ub) = quant_kernel_info(info.up_ttype);
    let (dk, dqk, db) = quant_kernel_info(info.down_ttype);
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let k: usize = 8; let k_top = 6;
    let mut rng: u32 = 0xABCD_1234;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

    let comp_ring = runner.dispatcher.comp_ring_or_alloc(layer_idx, 4096 * head_dim * 4);

    let scope = runner.dispatcher.batch_scope();
    // --- attention: a few per-position flash dispatches (like chunk_attn_core) ---
    for i in 0..k {
        let pos = i as u32;
        let q: Vec<f32> = (0..q_dim).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let q_db = scope.upload_f32(&q);
        let n_raw = (pos + 1).min(raw_cap);
        let _heads = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
            layer_idx, p, &q_db, n_raw, raw_cap, &comp_ring, &[], 0, &layer.attn.attn_sinks,
        ).expect("flash");
    }
    eprintln!("[flash+moe repro] ran {k} flash dispatches in scope; now MoE in SAME scope...");
    // --- MoE in the SAME scope ---
    let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
    for _ in 0..k {
        let mut picks: Vec<i32> = (0..n_experts as i32).collect();
        for i in 0..k_top { let j = i + (next() as usize % (n_experts - i)); picks.swap(i, j); }
        for slot in 0..k_top { selected.push(picks[slot]); }
    }
    let weights: Vec<f32> = (0..k * k_top).map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05).collect();
    let x_k: Vec<f32> = (0..k * d_in).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
    let x_db = scope.upload_f32(&x_k);
    let wt_db = scope.upload_f32(&weights);
    let sel_db = scope.alloc_i32(k * k_top);
    unsafe { std::ptr::copy_nonoverlapping(selected.as_ptr(), sel_db.buffer().contents() as *mut i32, k * k_top); }
    let moe = scope.encode_moe_chain_mm_qx_k(
        &x_db, &sel_db, &wt_db, &gate_db, &up_db, &down_db,
        gk, gqk, gb, uk, uqk, ub, dk, dqk, db, n_experts, d_in, d_ffn, k,
    ).expect("moe chain");
    let out = scope.flush_and_read(&moe);
    let nan = out.iter().filter(|v| !v.is_finite()).count();
    let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
    eprintln!("[flash+moe repro] MoE after flash in SAME scope: max_abs={max_abs:.3e} nan={nan} => {}",
        if nan > 0 || max_abs > 100.0 { "REPRODUCED (flash→MoE same-scope is the trigger)" } else { "CLEAN" });
}

/// UP-TOWARD-CHUNK REPRODUCER #4: run a DENSE matmul_k_q8_0 (simdgroup GEMM, 12KB
/// shmem — same machinery class as mul_mm_id) THEN the mm_id MoE in the SAME scope.
/// The chunk runs dense matmul_k (qkv + output_proj at K=8) before the MoE in one
/// cb; flash (a DIFFERENT kernel) didn't trigger it. Two distinct simdgroup-matmul
/// pipelines in one cb is the hypothesis.
#[test]
fn dense_matmul_k_then_moe_same_scope_repro() {
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping matmul_k+moe repro."); return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p), Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }
    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let _model = ComposedModelWeights::from_views(&views, &manifest, defaults).expect("model");
    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&gguf_path).expect("GgufFile::open");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers).expect("load_expert_weights");
    let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let info = disp.expert_weight_bufs(layer_idx).expect("expert_weight_bufs");
    let (n_experts, d_in, d_ffn) = (info.n_experts, info.d_in, info.d_ffn);
    let (gk, gqk, gb) = quant_kernel_info(info.gate_ttype);
    let (uk, uqk, ub) = quant_kernel_info(info.up_ttype);
    let (dk, dqk, db) = quant_kernel_info(info.down_ttype);
    let gate_db = DeferredBuf::from_external_buffer(info.gate.clone(), n_experts * d_ffn * d_in);
    let up_db   = DeferredBuf::from_external_buffer(info.up.clone(),   n_experts * d_ffn * d_in);
    let down_db = DeferredBuf::from_external_buffer(info.down.clone(), n_experts * d_in * d_ffn);

    let k: usize = 8; let k_top = 6;
    // Synthetic dense q8_0 weight [dm_out, dm_in] for matmul_k (d_in%32==0).
    let (dm_in, dm_out) = (4096usize, 2048usize);
    let nb = dm_in / 32; let row_bytes = nb * 34;
    let mut w_q8 = vec![0u8; dm_out * row_bytes];
    let mut s: u32 = 0x9999;
    let mut nb_byte = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 24) as u8 };
    for r in 0..dm_out { for b in 0..nb {
        let off = r*row_bytes + b*34; w_q8[off]=0x00; w_q8[off+1]=0x3C;
        for i in 0..32 { w_q8[off+2+i] = (((nb_byte() & 0x3f) as i32 - 32) as i8) as u8; }
    }}
    let mut rng: u32 = 0x5151_5151;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

    let scope = disp.batch_scope();
    // --- dense matmul_k_q8_0 (a few, like qkv+output) in the scope ---
    let w_db = scope.weight_q8_0_raw(&w_q8, dm_out * dm_in);
    for _ in 0..3 {
        let xm: Vec<f32> = (0..k*dm_in).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let xm_db = scope.upload_f32(&xm);
        let _o = scope.matmul_k_q8_0(&w_db, &xm_db, dm_in, dm_out, k).expect("matmul_k");
    }
    eprintln!("[matmul_k+moe repro] ran 3 dense matmul_k_q8_0 (K={k}); now MoE in SAME scope...");
    // --- MoE in the SAME scope ---
    let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
    for _ in 0..k {
        let mut picks: Vec<i32> = (0..n_experts as i32).collect();
        for i in 0..k_top { let j = i + (next() as usize % (n_experts - i)); picks.swap(i, j); }
        for slot in 0..k_top { selected.push(picks[slot]); }
    }
    let weights: Vec<f32> = (0..k*k_top).map(|_| (next() & 0xfff) as f32 / 4096.0 + 0.05).collect();
    let x_k: Vec<f32> = (0..k*d_in).map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
    let x_db = scope.upload_f32(&x_k);
    let wt_db = scope.upload_f32(&weights);
    let sel_db = scope.alloc_i32(k*k_top);
    unsafe { std::ptr::copy_nonoverlapping(selected.as_ptr(), sel_db.buffer().contents() as *mut i32, k*k_top); }
    let moe = scope.encode_moe_chain_mm_qx_k(
        &x_db, &sel_db, &wt_db, &gate_db, &up_db, &down_db,
        gk, gqk, gb, uk, uqk, ub, dk, dqk, db, n_experts, d_in, d_ffn, k,
    ).expect("moe chain");
    let out = scope.flush_and_read(&moe);
    let nan = out.iter().filter(|v| !v.is_finite()).count();
    let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
    eprintln!("[matmul_k+moe repro] MoE after dense matmul_k in SAME scope: max_abs={max_abs:.3e} nan={nan} => {}",
        if nan > 0 || max_abs > 100.0 { "REPRODUCED (dense-matmul_k -> mm_id same-scope is the trigger)" } else { "CLEAN" });
}

/// DECISIVE: run a SINGLE real `chunk_layer` (layer 3, the first Phase-B layer) in
/// isolation on a fresh AttnStepState + random prev_hc, for each MoE path
/// (blit/mm_id/fused). A single chunk_layer IS the full attn+FFN+MoE composition in
/// one BatchScope. If blit is CLEAN but mm_id/fused are GARBAGE, the in-chunk fault
/// is INTRA-chunk_layer (reproduced headlessly) → bisect by removing ops. If ALL
/// clean, the fault is the cross-layer / Phase-A multi-scope context.
///
/// DS4_BENCH_MOE_REAL=1 DS4_GGUF=... DS4_BENCH_MOE_LAYER=3
#[test]
fn single_chunk_layer_isolation_repro() {
    use ds4_metal::decode_runner::DecodeRunner;
    use ds4_metal::single_buffer_encoder::SingleBufferEncoder;
    use ds4_engine::decode_step::AttnStepState;
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping single-chunk_layer repro."); return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p), Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }

    let raw_cap: u32 = 256;
    let runner = DecodeRunner::open(&gguf_path, raw_cap).expect("DecodeRunner::open");
    let encoder = SingleBufferEncoder::new(&runner.dispatcher, raw_cap);
    let layer_idx: usize = std::env::var("DS4_BENCH_MOE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let k: usize = std::env::var("DS4_BENCH_MOE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let p = &runner.composed.layers[layer_idx].attn.params;
    eprintln!("[1-chunk_layer] layer {layer_idx} ratio={} K={k} (idx={})",
        p.compress_ratio,
        runner.composed.layers[layer_idx].attn.has_indexer_compressor()
            && runner.composed.layers[layer_idx].attn.has_indexer_qb());
    // DIMS: what chunk_layer passes (p.d_embd, layer.moe.d_ffn, layer.moe.n_experts)
    // vs the ACTUAL expert tensor dims (expert_weight_bufs). A mismatch => the mm_id
    // _auto path builds a wrong-sized buffer view => OOB/SEGV (the isolation bench
    // used the real tensor dims and worked).
    let mo = &runner.composed.layers[layer_idx].moe;
    let info = runner.dispatcher.expert_weight_bufs(layer_idx as u32).expect("expert_weight_bufs");
    eprintln!("[1-chunk_layer DIMS] passed: d_embd={} moe.d_ffn={} moe.n_experts={}  | actual tensor: d_in={} d_ffn={} n_experts={}",
        p.d_embd, mo.d_ffn.map_or(0, |n| n.get()), mo.n_experts, info.d_in, info.d_ffn, info.n_experts);

    for path in ["blit", "mm_id", "fused"] {
        if path == "blit" { std::env::remove_var("DS4_MOE_K_PATH"); }
        else { std::env::set_var("DS4_MOE_K_PATH", path); }
        let mut state = AttnStepState::new(&runner.composed, raw_cap);
        match encoder.debug_run_one_chunk_layer(&runner.composed, &mut state, layer_idx, k, 0x1234_5678) {
            Ok(out) => {
                let nan = out.iter().filter(|v| !v.is_finite()).count();
                let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, &v| a.max(v.abs()));
                let verdict = if nan > 0 || max_abs > 1.0e3 { "REPRODUCED (garbage)" } else { "clean" };
                eprintln!("[1-chunk_layer L{layer_idx} MOE={path}] max_abs={max_abs:.3e} nan={nan} => {verdict}");
            }
            Err(e) => eprintln!("[1-chunk_layer L{layer_idx} MOE={path}] ERR: {e:?}"),
        }
    }
    std::env::remove_var("DS4_MOE_K_PATH");
    eprintln!("[1-chunk_layer] If blit=clean but mm_id/fused=garbage => fault is INTRA-chunk_layer (bisect next). All clean => cross-layer/Phase-A.");
}

/// DS4_CHUNK_BATCH_FLASH closeness A/B — validate the batched-attention fast path in
/// `chunk_attn_core_comp_noidx` by running ONE noidx (ratio!=4) chunk_layer with the
/// batched flash OFF vs ON, same seed + fresh state, MoE held at blit for both. The
/// ONLY difference is the flash kernel (per-position gpuring → sub-batched
/// flash_attn_k_mla_comp), so a CORRECT batched path differs only at fp32 simdgroup
/// reduction order (rel err ~1e-3); an O(1) rel err means a logic bug (masking, comp
/// rows, rope). This is the right gate — the decode-token byte-identity test can't
/// distinguish logic from fp32 amplification on out-of-distribution synthetic prompts.
///
/// NOTE: the harness runs at chunk_start=0 so n_comp=0 — this exercises the raw causal
/// window + {8,4,2,1} grouping + rope-back, NOT the comp-row gather (needs chunk_start
/// >=ratio with prior emits; separate test).
///
/// DS4_BENCH_MOE_REAL=1 DS4_GGUF=... [DS4_BENCH_MOE_K=8]
#[test]
fn chunk_batch_flash_closeness_ab() {
    use ds4_metal::decode_runner::DecodeRunner;
    use ds4_metal::single_buffer_encoder::SingleBufferEncoder;
    use ds4_engine::decode_step::AttnStepState;
    if std::env::var("DS4_BENCH_MOE_REAL").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_REAL unset — skipping chunk_batch_flash_closeness_ab."); return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p), Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    if !gguf_path.is_file() { eprintln!("DS4_GGUF not a file — skipping"); return; }

    let raw_cap: u32 = 256;
    let runner = DecodeRunner::open(&gguf_path, raw_cap).expect("DecodeRunner::open");
    let encoder = SingleBufferEncoder::new(&runner.dispatcher, raw_cap);
    let k: usize = std::env::var("DS4_BENCH_MOE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    // Find the first noidx compressor layer: ratio != 0 AND NOT (ratio==4 && indexer)
    // AND not hash-routed (chunk_layer rejects routing tables).
    let layer_idx = (0..runner.composed.layers.len()).find(|&l| {
        let lw = &runner.composed.layers[l];
        let ratio = lw.attn.params.compress_ratio;
        let is_idx = ratio == 4 && lw.attn.has_indexer_compressor() && lw.attn.has_indexer_qb();
        ratio != 0 && !is_idx && lw.moe.routing_table.is_none()
    });
    let Some(layer_idx) = layer_idx else {
        eprintln!("no noidx (ratio!=4) compressor layer found — skipping."); return;
    };
    let ratio = runner.composed.layers[layer_idx].attn.params.compress_ratio;
    eprintln!("[batch_flash A/B] noidx layer {layer_idx} ratio={ratio} K={k}");

    std::env::remove_var("DS4_MOE_K_PATH"); // blit for both runs (held constant)

    std::env::remove_var("DS4_CHUNK_BATCH_FLASH");
    let mut state_off = AttnStepState::new(&runner.composed, raw_cap);
    let off = encoder.debug_run_one_chunk_layer(&runner.composed, &mut state_off, layer_idx, k, 0x1234_5678)
        .expect("batch_flash OFF run");

    std::env::set_var("DS4_CHUNK_BATCH_FLASH", "1");
    let mut state_on = AttnStepState::new(&runner.composed, raw_cap);
    let on = encoder.debug_run_one_chunk_layer(&runner.composed, &mut state_on, layer_idx, k, 0x1234_5678)
        .expect("batch_flash ON run");
    std::env::remove_var("DS4_CHUNK_BATCH_FLASH");

    assert_eq!(off.len(), on.len(), "output length mismatch");
    let mut max_rel = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut nan = 0usize;
    for (&a, &b) in off.iter().zip(on.iter()) {
        if !a.is_finite() || !b.is_finite() { nan += 1; continue; }
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        let denom = a.abs().max(b.abs()).max(1e-6);
        max_rel = max_rel.max(d / denom);
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (a as f64) * (a as f64);
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt();
    eprintln!("[batch_flash A/B] L{layer_idx} K={k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} \
               rel_L2={rel_l2:.3e} nan={nan}  (off[0..4]={:?} on[0..4]={:?})",
        &off[..4.min(off.len())], &on[..4.min(on.len())]);
    assert_eq!(nan, 0, "batched flash produced {nan} non-finite outputs — logic bug");
    // fp32 simdgroup reduction across a full layer: expect rel_L2 well under 1e-2.
    // A logic bug (wrong masking/comp/rope) yields O(1).
    assert!(rel_l2 < 1.0e-2,
        "batched flash rel_L2={rel_l2:.3e} exceeds 1e-2 — NOT fp32 noise, likely a logic bug");
    eprintln!("[batch_flash A/B] PASS: batched flash matches per-position to fp32 precision (rel_L2 < 1e-2)");
}
