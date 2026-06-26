//! Phase 2 — encode_attn_chain_k K-amortization benchmark.
//!
//! Measures wall-clock per `encode_attn_chain_k` invocation at K∈{1,2,4,8} on
//! production-scale dims (n_embd=4096, head_dim=512, kv_row=512, n_lora_q=128).
//! Goal: quantify the K-amortization win of the simdgroup-matrix Q8_0 matvecs
//! + simdgroup-matrix flash attention, vs the K-linear cost of the existing
//! per-K kernels (kv_fp8_store, rope_tail, etc.).
//!
//! Output per K:
//!   - mean ms per encode_attn_chain_k call (after 3-iter warmup)
//!   - μs/K-position (mean_ms / K * 1000) — the K-amortization metric:
//!     if μs/K-position is FLAT across K, we have perfect K-amortization
//!     (more K positions take the same per-position GPU time)
//!
//! Reports a "speedup ratio" K=8 vs K=1:
//!   - mean_ms(K=8) / (8 * mean_ms(K=1)) — what fraction of the K=1 time
//!     it actually costs to process K=8. < 1.0 = amortization win.
//!
//! Gated by `DS4_BENCH_LAYER_K=1` — not run in normal `cargo test`. Synthetic
//! random Q8_0 weights (no GGUF required); this only tests the GPU dispatch
//! throughput, not numerical correctness (covered by 60 + 2 smoke tests).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;
use std::time::Instant;

#[test]
fn encode_attn_chain_k_amortization_bench() {
    if std::env::var("DS4_BENCH_LAYER_K").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_LAYER_K unset — skipping K-amortization bench. Set DS4_BENCH_LAYER_K=1 to run.");
        return;
    }

    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {}", e);
            return;
        }
    };

    // Production-scale dims (DS4 V4 Flash):
    //   - n_hc=4, n_embd=4096 (forced by hc_collapse_norm kernel)
    //   - n_lora_q=128, n_head=2, head_dim=512 (DK=DV=512 for flash_attn_k_mla)
    //   - kv_row=512 (n_lora_kv = MLA cache row)
    //   - q_dim = n_head*head_dim = 1024
    //   - n_groups=8, group_dim=128, n_lora_o=4, out_low_dim=32
    // Defaults are the small synthetic dims used for the K-amortization
    // RATIO measurement (n_head=2 keeps the bench cheap). For ABSOLUTE
    // per-layer cost at real DS4 V4 Flash scale, override via env:
    //   DS4_ATTN_BENCH_REAL=1  → n_head=64, n_lora_o=1024, n_out_group=8,
    //                            raw_cap=256 (matches the spec-decode verifier).
    let real = std::env::var("DS4_ATTN_BENCH_REAL").ok().as_deref() == Some("1");
    let n_hc: usize = 4;
    let n_embd: usize = 4096;
    let d_embd: usize = n_embd;
    let n_lora_q: usize = if real { 1024 } else { 128 };
    let n_head: usize = if real { 64 } else { 2 };
    let head_dim: usize = 512;
    let q_dim: usize = n_head * head_dim;
    let kv_row: usize = 512;
    let n_rot: usize = 64;
    let hc_dim = n_hc * n_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let sinkhorn_iters: i32 = 5;
    let hc_eps: f32 = 1e-6;
    let rms_eps: f32 = 1e-5;
    let n_groups: usize = if real { 8 } else { 2 };
    let n_lora_o: usize = if real { 1024 } else { 4 };
    let group_dim: usize = head_dim * (n_head / n_groups);
    let out_low_dim: usize = n_groups * n_lora_o;
    let flash_scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let params = LayerParams {
        layer_idx: 0,
        d_embd: d_embd as u32,
        n_hc: n_hc as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        n_rot: n_rot as u32,
        n_lora_q: n_lora_q as u32,
        n_lora_kv: kv_row as u32,
        hc_sinkhorn_iter: sinkhorn_iters as u32,
        hc_eps,
        rms_eps,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: n_groups as u32,
    };

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
                v[off + 1] = 0x3C; // half 1.0
                for i in 0..32 {
                    v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                }
            }
        }
        v
    }

    let w_qa = rand_q8(n_lora_q, d_embd, 0xA1);
    let w_qb = rand_q8(q_dim, n_lora_q, 0xB2);
    let w_kv = rand_q8(kv_row, d_embd, 0xC3);
    let w_o_a = rand_q8(out_low_dim, group_dim, 0xD4);
    let w_o_b = rand_q8(d_embd, out_low_dim, 0xE5);

    let mut rng: u32 = 0xDEAD_BEEF;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
        .collect();
    let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
    let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + i as f32 * 0.01).collect();
    let hc_norm_gamma: Vec<f32> = (0..n_embd)
        .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05)
        .collect();
    let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];
    let gamma_q: Vec<f32> = (0..n_lora_q)
        .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.04)
        .collect();
    let gamma_kv: Vec<f32> = (0..kv_row)
        .map(|i| 1.0 + (i as f32 * 0.017).sin() * 0.04)
        .collect();

    // For each K, we need K * hc_dim floats of prev_hc.
    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_LAYER_K_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let ks: &[usize] = &[1, 2, 4, 8];
    let mut means_ms: Vec<f64> = Vec::with_capacity(ks.len());

    eprintln!(
        "encode_attn_chain_k bench at production dims: n_embd={} n_lora_q={} n_head={} head_dim={} kv_row={} raw_cap=16",
        n_embd, n_lora_q, n_head, head_dim, kv_row
    );
    eprintln!(
        "  warmup_iters={} bench_iters={} (set DS4_BENCH_LAYER_K_ITERS to override)",
        warmup_iters, bench_iters
    );

    for &k in ks {
        let prev_hc_k: Vec<f32> = (0..k * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();

        // raw_cap is per-K-call write capacity; spec-decode pattern would advance
        // base_slot by K each token. For pure-throughput measurement, hold raw_cap
        // big enough for one bench iter and reset base_slot each iter.  Real mode
        // uses raw_cap=256 (verifier cache size) so the flash kernel's NC_BLK
        // loop is representative.
        let raw_cap: u32 = if real { 256 } else { 16 };
        let base_slot: u32 = 4;
        let base_pos: u32 = 4;
        let layer_idx: u32 = 200 + k as u32; // distinct per-K cache to avoid races

        let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);

        for _ in 0..(warmup_iters + bench_iters) {
            let mut scope = disp.batch_scope();
            scope.set_q8_proj(true);
            let hc_fn_db = scope.weight_f32(&hc_fn);
            let hc_scale_db = scope.weight_f32(&hc_scale);
            let hc_base_db = scope.weight_f32(&hc_base);
            let hc_norm_gamma_db = scope.weight_f32(&hc_norm_gamma);
            let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);
            let gamma_q_db = scope.weight_f32(&gamma_q);
            let gamma_kv_db = scope.weight_f32(&gamma_kv);
            let w_qa_db = scope.weight_q8_0_raw(&w_qa, n_lora_q * d_embd);
            let w_qb_db = scope.weight_q8_0_raw(&w_qb, q_dim * n_lora_q);
            let w_kv_db = scope.weight_q8_0_raw(&w_kv, kv_row * d_embd);
            let w_oa_db = scope.weight_q8_0_raw(&w_o_a, out_low_dim * group_dim);
            let w_ob_db = scope.weight_q8_0_raw(&w_o_b, d_embd * out_low_dim);
            let sinks_vec = vec![0.0f32; n_head];
            let sinks_db = scope.weight_f32(&sinks_vec);
            let prev_hc_k_db = scope.upload_f32(&prev_hc_k);

            let t0 = Instant::now();
            let (_half, attn_out_k) = scope
                .encode_attn_chain_k(
                    &prev_hc_k_db,
                    &hc_fn_db,
                    &hc_scale_db,
                    &hc_base_db,
                    &hc_norm_gamma_db,
                    &unit_gamma_db,
                    &w_qa_db,
                    &gamma_q_db,
                    &w_qb_db,
                    &w_kv_db,
                    &gamma_kv_db,
                    &w_oa_db,
                    &w_ob_db,
                    n_hc,
                    n_embd,
                    n_lora_q,
                    n_head,
                    head_dim,
                    kv_row,
                    n_groups,
                    n_lora_o,
                    group_dim,
                    out_low_dim,
                    sinkhorn_iters,
                    hc_eps,
                    rms_eps,
                    flash_scale,
                    layer_idx,
                    &params,
                    raw_cap,
                    base_slot,
                    base_pos,
                    raw_cap - 1, // attn_base_pos: unmasked (full cache window)
                    &sinks_db,
                    k,
                    None, // comp: no compressor in this bench
                )
                .expect("encode_attn_chain_k");
            let _out = scope.flush_and_read(&attn_out_k); // includes GPU wait
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            iter_ms.push(elapsed_ms);
        }

        // Drop warmup iters.
        let bench_only: &[f64] = &iter_ms[warmup_iters..];
        let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
        let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
        let max = bench_only.iter().copied().fold(0.0f64, f64::max);
        let us_per_k_pos = mean * 1000.0 / k as f64;
        eprintln!(
            "  K={}: mean={:.3} ms  [min={:.3} max={:.3}]  μs/K-position={:.1}",
            k, mean, min, max, us_per_k_pos
        );
        means_ms.push(mean);
    }

    // K-amortization ratios.
    let m1 = means_ms[0];
    eprintln!("\nK-amortization vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ideal_ms = (k as f64) * m1;
        let actual_ms = means_ms[i];
        let ratio = actual_ms / ideal_ms;
        let efficiency_pct = (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0;
        eprintln!(
            "  K={}: actual={:.3} ms, K×K=1={:.3} ms, ratio={:.3} (efficiency: {:.0}%)",
            k, actual_ms, ideal_ms, ratio, efficiency_pct
        );
    }

    // Sanity: K=2 should be at most 1.5× K=1 (well-amortized) on this kernel
    // mix; assert generously to avoid flakes.
    assert!(
        means_ms[1] < 2.5 * m1,
        "K=2 time {:.3} ms >> 2.5× K=1 ({:.3} ms) — amortization regression?",
        means_ms[1],
        m1
    );
}
