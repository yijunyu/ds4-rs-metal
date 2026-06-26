//! Phase 2 MoE-K Option A — END-TO-END encode_ffn_chain_k real-model bench.
//!
//! Runs `BatchScope::encode_ffn_chain_k` on one real DS4 V4 Flash layer
//! (loaded from DS4_GGUF), comparing the default blit-shim path against
//! the new DS4_MOE_K_PATH=fused path. Measures wall-clock per K and
//! computes per-K-position speedup of fused vs blit.
//!
//! This is the closure of Option A: the prior MoE-only bench measured
//! mul_mm_id vs blit; this measures the FULL FFN chain (HC-expand-attn-
//! split → hc_collapse_norm → router_logits → router_finalize → MoE +
//! shared → hc_expand_add_split) at K∈{1,2,4,8} both ways.
//!
//! Gated by `DS4_BENCH_FFN_E2E=1` AND `DS4_GGUF=/path`. macOS-only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_metal::MetalDispatcher;

fn next_lcg(rng: &mut u32) -> u32 {
    *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
    *rng
}

#[test]
fn encode_ffn_chain_k_e2e_real_bench() {
    if std::env::var("DS4_BENCH_FFN_E2E").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_FFN_E2E unset — skipping. Set DS4_BENCH_FFN_E2E=1 + DS4_GGUF=/path to run.");
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

    let layer_idx: usize = std::env::var("DS4_BENCH_MOE_LAYER")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let layer = &model.layers[layer_idx];
    let attn = &layer.attn;
    let moe = &layer.moe;

    // Shapes.
    let d_embd: usize = model.d_model;       // 4096
    let n_hc: usize = 4;                       // DS4 V4 Flash
    let mix_hc = 2 * n_hc + n_hc * n_hc;     // 24
    let hc_dim = n_hc * d_embd;                // 16384
    let n_experts = moe.n_experts;             // 256
    // moe.d_ffn is 0 by design (decode_step.rs:382): the MetalDispatcher
    // reads d_ffn from QuantizedExpertWeights instead. Pull it from there.
    let exp_info = disp.expert_weight_bufs(layer_idx as u32).expect("expert_weight_bufs");
    let d_ffn = exp_info.d_ffn;                // expert intermediate
    let shared_dim = attn.shared_dim;          // shared-expert intermediate
    let sinkhorn_iters: i32 = attn.params.hc_sinkhorn_iter as i32;
    let hc_eps = attn.params.hc_eps;
    let rms_eps = attn.params.rms_eps;

    eprintln!(
        "layer {}: d_embd={} d_ffn={} shared_dim={} n_experts={}",
        layer_idx, d_embd, d_ffn, shared_dim, n_experts
    );

    // Random inputs (K-dependent allocated per K-iteration).
    let mut rng: u32 = 0x1234_5678;

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_FFN_E2E_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    // Prefill chunking spike: sweep large K via DS4_BENCH_FFN_E2E_KS (comma-sep);
    // default is the spec-decode set. For prefill, K is the chunk size (64-256),
    // where weight-read amortization is far stronger than spec-decode K=4-8.
    let ks_owned: Vec<usize> = std::env::var("DS4_BENCH_FFN_E2E_KS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    let ks: &[usize] = &ks_owned;

    // Two passes: blit (default), then fused (DS4_MOE_K_PATH=fused).
    let mut blit_ms: Vec<f64> = vec![0.0; ks.len()];
    let mut fused_ms: Vec<f64> = vec![0.0; ks.len()];
    let mut mm_id_ms: Vec<f64> = vec![0.0; ks.len()];

    // fused (pair_swiglu_K) breaks at K>=32 — dropped from the large-K sweep;
    // blit is K-linear in the MoE (gets worse at large K). mm_id is the path.
    for (pass_name, path_val) in [("blit", None), ("mm_id", Some("mm_id"))] {
        eprintln!("\n=== pass: {} (DS4_MOE_K_PATH={:?}) ===", pass_name, path_val);
        match path_val {
            Some(v) => std::env::set_var("DS4_MOE_K_PATH", v),
            None => std::env::remove_var("DS4_MOE_K_PATH"),
        }
        for (ki, &k) in ks.iter().enumerate() {
            // Fresh random inputs per K.
            let attn_out_k_vec: Vec<f32> = (0..k * d_embd)
                .map(|_| (next_lcg(&mut rng) & 0xffff) as f32 / 65536.0 - 0.5)
                .collect();
            let cur_hc_k_vec: Vec<f32> = (0..k * hc_dim)
                .map(|_| (next_lcg(&mut rng) & 0xffff) as f32 / 65536.0 - 0.5)
                .collect();
            let attn_split_k_vec: Vec<f32> = (0..k * mix_hc)
                .map(|_| (next_lcg(&mut rng) & 0xffff) as f32 / 65536.0 - 0.5)
                .collect();
            let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];

            let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);
            for _ in 0..(warmup_iters + bench_iters) {
                let mut scope = disp.batch_scope();
                let attn_out_db = scope.upload_f32(&attn_out_k_vec);
                let cur_hc_db = scope.upload_f32(&cur_hc_k_vec);
                let attn_split_db = scope.upload_f32(&attn_split_k_vec);
                let hc_ffn_fn_db        = scope.weight_f32(&attn.hc_ffn_fn);
                let hc_ffn_scale_db     = scope.weight_f32(&attn.hc_ffn_scale);
                let hc_ffn_base_db      = scope.weight_f32(&attn.hc_ffn_base);
                let hc_ffn_norm_gamma_db = scope.weight_f32(&attn.hc_ffn_norm_gamma);
                let unit_gamma_db       = scope.weight_f32(&unit_gamma_hc);
                let w_router_db         = scope.weight_f32(&moe.w_router);
                let router_bias_db      = scope.upload_f32(&moe.router_bias);

                let t0 = Instant::now();
                let out = scope.encode_ffn_chain_k(
                    &attn_out_db, &cur_hc_db, &attn_split_db,
                    &hc_ffn_fn_db, false, &hc_ffn_scale_db, &hc_ffn_base_db, &hc_ffn_norm_gamma_db,
                    &unit_gamma_db,
                    &w_router_db, false, &router_bias_db,
                    &attn.w_shared_gate, &attn.w_shared_up, &attn.w_shared_down,
                    &attn.w_shared_gate_q8, &attn.w_shared_up_q8, &attn.w_shared_down_q8,
                    n_hc, d_embd, n_experts, d_ffn, shared_dim,
                    sinkhorn_iters, hc_eps, rms_eps,
                    layer_idx as u32, k,
                    None,
                ).expect("encode_ffn_chain_k");
                let result = scope.flush_and_read(&out);
                iter_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

                if iter_ms.len() == warmup_iters + 1 {
                    assert_eq!(result.len(), k * hc_dim, "after_ffn_k shape mismatch");
                    let nan_count = result.iter().filter(|v| !v.is_finite()).count();
                    let max_abs = result.iter().filter(|v| v.is_finite())
                        .fold(0.0f32, |a, &v| a.max(v.abs()));
                    eprintln!("    K={}: max_abs={:.3e}  nan_count={}", k, max_abs, nan_count);
                }
            }
            let bench_only = &iter_ms[warmup_iters..];
            let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
            let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
            let max = bench_only.iter().copied().fold(0.0f64, f64::max);
            eprintln!("  K={}: mean={:.3} ms  [min={:.3} max={:.3}]  μs/K-pos={:.1}",
                k, mean, min, max, mean * 1000.0 / k as f64);
            match pass_name {
                "blit"  => blit_ms[ki]  = mean,
                "fused" => fused_ms[ki] = mean,
                "mm_id" => mm_id_ms[ki] = mean,
                _ => unreachable!(),
            }
        }
    }
    std::env::remove_var("DS4_MOE_K_PATH");

    // Comparison table.
    eprintln!("\n=== FFN chain real-model speedup (fused vs blit) ===");
    eprintln!("{:>4}  {:>10}  {:>10}  {:>8}  {:>14}",
              "K", "blit ms", "fused ms", "speedup", "fused μs/K-pos");
    for (ki, &k) in ks.iter().enumerate() {
        let speedup = blit_ms[ki] / fused_ms[ki];
        let us_per_kpos = fused_ms[ki] * 1000.0 / k as f64;
        eprintln!("{:>4}  {:>10.3}  {:>10.3}  {:>7.2}x  {:>14.1}",
                  k, blit_ms[ki], fused_ms[ki], speedup, us_per_kpos);
    }

    // Per-K-position speedup (the spec-decode-relevant metric).
    let blit_k1 = blit_ms[0];
    eprintln!("\nFused per-K-pos vs blit K=1:");
    for (ki, &k) in ks.iter().enumerate() {
        let fused_us_per_kpos = fused_ms[ki] * 1000.0 / k as f64;
        let blit_us_per_kpos = blit_k1 * 1000.0;
        let ratio = fused_us_per_kpos / blit_us_per_kpos;
        eprintln!("  K={}: fused/blit_K1 per-K-pos ratio = {:.3} ({:.2}x speedup)",
                  k, ratio, 1.0 / ratio);
    }
}
