//! Phase 2 — encode_ffn_chain_k component-by-component K-amortization bench.
//!
//! Measures every FFN-half primitive EXCEPT the MoE expert chain (which
//! requires real GGUF expert weights, not synthesizable from a test-only
//! integration entry point). For the MoE itself, the K-amortization shape
//! is structurally K-linear by design (blit-shim around the K=1 kernel) —
//! see `encode_moe_and_shared_chain_k` doc.
//!
//! Components measured:
//!   1. hc_expand_attn_split_k   — attn-half HC fold (K-linear, cheap)
//!   2. hc_collapse_norm_k       — FFN-side collapse (K-linear via blit shim)
//!   3. encode_router_logits_k   — K iters of router matvec + flat softplus
//!   4. encode_router_finalize_k — K iters of top-6 + weight norm
//!   5. encode_shared_chain_q8_k — fully K-AMORTIZED (matvec_k_q8_0 × 3 + swiglu)
//!   6. hc_expand_add_split_k    — FFN HC fold (K-linear, cheap)
//!
//! Reports per-component μs/K-position; summed total gives the "FFN minus MoE"
//! K-amortization upper bound. Combine with `encode_attn_chain_k` bench results
//! to project full-layer K=8 throughput (MoE estimated as 8× K=1 baseline).
//!
//! Gated by `DS4_BENCH_FFN_K=1`. macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use std::time::Instant;

fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
    assert_eq!(d_in % 32, 0, "Q8_0 needs d_in %32 (got {})", d_in);
    let nb = d_in / 32;
    let row_bytes = nb * 34;
    let mut v = vec![0u8; d_out * row_bytes];
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };
    for r in 0..d_out {
        for b in 0..nb {
            let off = r * row_bytes + b * 34;
            v[off] = 0x00;
            v[off + 1] = 0x3C;
            for i in 0..32 {
                v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
            }
        }
    }
    v
}

#[test]
fn encode_ffn_components_k_bench() {
    if std::env::var("DS4_BENCH_FFN_K").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_FFN_K unset — skipping FFN component K bench. Set DS4_BENCH_FFN_K=1 to run.");
        return;
    }

    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {}", e);
            return;
        }
    };

    // Production-scale dims (DS4 V4 Flash).
    let n_hc: usize = 4;
    let n_embd: usize = 4096;
    let d_embd: usize = n_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let hc_dim = n_hc * d_embd;
    let sinkhorn_iters: i32 = 5;
    let hc_eps: f32 = 1e-6;
    let rms_eps: f32 = 1e-5;
    let n_experts: usize = 256;
    let shared_dim: usize = 2048; // %32 ✓ (Q8_0 in matvec_k_q8_0)

    let mut rng: u32 = 0xCAFE_1234;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };

    // FFN-side HC weights.
    let hc_ffn_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
        .collect();
    let hc_ffn_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
    let hc_ffn_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + i as f32 * 0.01).collect();
    let hc_ffn_norm_gamma: Vec<f32> = (0..n_embd)
        .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05)
        .collect();
    let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];

    // Router weights (f32 since router goes through mul_mv_f32).
    let w_router: Vec<f32> = (0..n_experts * d_embd)
        .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
        .collect();
    let router_bias: Vec<f32> = (0..256).map(|i| (i as f32 * 0.007).sin() * 0.01).collect();

    // Shared-expert weights (Q8_0 bytes).
    let sh_w_gate = rand_q8(shared_dim, d_embd, 0xAA);
    let sh_w_up = rand_q8(shared_dim, d_embd, 0xBB);
    let sh_w_down = rand_q8(d_embd, shared_dim, 0xCC);

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_FFN_K_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let ks: &[usize] = &[1, 2, 4, 8];
    eprintln!(
        "encode_ffn (no-MoE) component bench at production dims: n_embd={} n_experts={} shared_dim={}",
        n_embd, n_experts, shared_dim
    );
    eprintln!("  warmup={}  bench_iters={}", warmup_iters, bench_iters);

    // (component, K) → mean ms
    let component_names = [
        "hc_expand_attn_split_k",
        "hc_collapse_norm_k",
        "router_logits_k",
        "router_finalize_k",
        "shared_chain_q8_k",
        "hc_expand_add_split_k",
        "TOTAL (no MoE)",
    ];
    let mut per_component_ms: Vec<Vec<f64>> = vec![vec![0.0; ks.len()]; component_names.len()];

    for (ki, &k) in ks.iter().enumerate() {
        let attn_out_k: Vec<f32> = (0..k * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();
        let cur_hc_k: Vec<f32> = (0..k * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();
        let attn_split_k: Vec<f32> = (0..k * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();

        // Pre-generate the synthetic data used across iters/components.
        let normed_for_router: Vec<f32> = (0..k * d_embd)
            .map(|i| ((i as f32 * 0.013).sin() * 0.4).clamp(-1.0, 1.0))
            .collect();
        let probs_for_finalize: Vec<f32> = (0..k * 256)
            .map(|i| ((i as f32 * 0.011).sin() * 0.5).abs())
            .collect();
        let shared_for_add: Vec<f32> = (0..k * d_embd)
            .map(|i| (i as f32 * 0.009).sin() * 0.3)
            .collect();
        let routed_for_add: Vec<f32> = (0..k * d_embd)
            .map(|i| (i as f32 * 0.011).cos() * 0.3)
            .collect();
        let after_attn_for_add: Vec<f32> = (0..k * hc_dim)
            .map(|i| (i as f32 * 0.005).sin() * 0.2)
            .collect();

        // Single-scope (single cb) timing: all 6 components dispatched in one
        // cb, ONE flush. Mirrors how encode_ffn_chain_k would commit per layer.
        let mut comp_iter_ms: Vec<Vec<f64>> = vec![Vec::new(); component_names.len()];

        for _ in 0..(warmup_iters + bench_iters) {
            let scope = disp.batch_scope();
            // Upload all inputs once.
            let attn_db = scope.upload_f32(&attn_out_k);
            let cur_hc_db = scope.upload_f32(&cur_hc_k);
            let sp_db = scope.upload_f32(&attn_split_k);
            let hcfn_db = scope.weight_f32(&hc_ffn_fn);
            let hcsc_db = scope.weight_f32(&hc_ffn_scale);
            let hcb_db = scope.weight_f32(&hc_ffn_base);
            let hcg_db = scope.weight_f32(&hc_ffn_norm_gamma);
            let unit_db = scope.weight_f32(&unit_gamma_hc);
            let normed_router_db = scope.upload_f32(&normed_for_router);
            let w_r_db = scope.weight_f32(&w_router);
            let probs_fin_db = scope.upload_f32(&probs_for_finalize);
            let bias_db = scope.upload_f32(&router_bias);
            let wg_db = scope.weight_q8_0_raw(&sh_w_gate, shared_dim * d_embd);
            let wu_db = scope.weight_q8_0_raw(&sh_w_up, shared_dim * d_embd);
            let wd_db = scope.weight_q8_0_raw(&sh_w_down, d_embd * shared_dim);
            let sh_add_db = scope.upload_f32(&shared_for_add);
            let rt_add_db = scope.upload_f32(&routed_for_add);
            let aa_add_db = scope.upload_f32(&after_attn_for_add);

            // Encode all 6 components.
            let t0 = Instant::now();
            let _after_attn = scope
                .hc_expand_attn_split_k(&attn_db, &cur_hc_db, &sp_db, n_hc, d_embd, k)
                .expect("hc_expand_attn_split_k");
            let (_split_k, _cur_k, _normed_k) = scope
                .hc_collapse_norm_k(
                    &cur_hc_db, &hcfn_db, &hcsc_db, &hcb_db, &hcg_db,
                    n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps, &unit_db, k, false,
                )
                .expect("hc_collapse_norm_k");
            let _probs_k = scope
                .encode_router_logits_k(&w_r_db, &normed_router_db, n_experts, d_embd, k, false)
                .expect("router_logits_k");
            let _sel_wt = scope
                .encode_router_finalize_k(&probs_fin_db, &bias_db, k)
                .expect("router_finalize_k");
            let _shared_out = scope
                .encode_shared_chain_q8_k(&normed_router_db, &wg_db, &wu_db, &wd_db, d_embd, shared_dim, k)
                .expect("shared_chain_q8_k");
            let after_ffn = scope
                .hc_expand_add_split_k(&sh_add_db, &rt_add_db, &aa_add_db, &sp_db, n_hc, d_embd, k)
                .expect("hc_expand_add_split_k");
            let _ = scope.flush_and_read(&after_ffn);
            comp_iter_ms[component_names.len() - 1].push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // Only "TOTAL (no MoE)" is meaningful in the single-scope path.
        let bench_only = &comp_iter_ms[component_names.len() - 1][warmup_iters..];
        let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
        per_component_ms[component_names.len() - 1][ki] = mean;
        for c in 0..component_names.len() - 1 {
            per_component_ms[c][ki] = -1.0; // sentinel: not measured per-component
        }
    }

    // Pretty-print results table.
    eprintln!(
        "\n{:<26}  {:>10}  {:>10}  {:>10}  {:>10}",
        "component", "K=1 ms", "K=2 ms", "K=4 ms", "K=8 ms"
    );
    eprintln!("{}", "-".repeat(78));
    for (c, name) in component_names.iter().enumerate() {
        let v = &per_component_ms[c];
        if v[0] < 0.0 {
            eprintln!("{:<26}  (per-component not measured in single-scope mode)", name);
        } else {
            eprintln!(
                "{:<26}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}",
                name, v[0], v[1], v[2], v[3]
            );
        }
    }

    // K-amortization ratios for total.
    let total_row = &per_component_ms[component_names.len() - 1];
    eprintln!("\nFFN-no-MoE K-amortization vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ideal = (k as f64) * total_row[0];
        let actual = total_row[i];
        let ratio = actual / ideal;
        let efficiency = if k == 1 {
            100.0
        } else {
            (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
        };
        eprintln!(
            "  K={}: actual={:.3} ms, K×K=1={:.3} ms, ratio={:.3}  efficiency={:.0}%",
            k, actual, ideal, ratio, efficiency
        );
    }

    eprintln!(
        "\n*** NOTE: MoE excluded. Per-layer MoE blit-shim cost is K-linear at K× the K=1 MoE.\n\
              Production K=1 MoE-only time (from DS4_OP_TRACE on real model) is the closure.\n\
              Combine with encode_attn_chain_k_bench for full-layer K-amortization picture."
    );
}
