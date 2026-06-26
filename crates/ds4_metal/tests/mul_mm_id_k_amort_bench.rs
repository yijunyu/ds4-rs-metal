//! Phase 2 MoE-K Step 1 — Recon bench for the K-amortized MoE matmul
//! using the upstream `kernel_mul_mm_id_q8_0_f32` (already loaded, never
//! dispatched by decode). Dispatches map0 → mul_mm_id at K∈{1,2,4,8} on
//! synthetic Q8_0 expert weights.
//!
//! Purpose: validate the K-amortization architectural bet BEFORE committing
//! to the full encode_moe_routed_step_mm_k integration (see roadmap
//! "MoE-K discovery: kernels already exist!").
//!
//! Synthetic Q8_0 keeps test setup simple (no Q4_K/IQ2_XXS block format
//! work needed). The K-amortization RATIO across K should be representative
//! of what mul_mm_id_q4_K_f32 and mul_mm_id_iq2_xxs_f32 deliver, since
//! they all share the same NR0=64/NR1=32 tile structure and weight-read
//! amortization pattern.
//!
//! Gated by `DS4_BENCH_MOE_K=1`. macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use std::time::Instant;

#[test]
fn mul_mm_id_q8_0_k_amortization_bench() {
    if std::env::var("DS4_BENCH_MOE_K").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_K unset — skipping MoE-K recon bench. Set DS4_BENCH_MOE_K=1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
    };

    // Shapes: production-like at half scale to keep weight memory tractable.
    // DS4 V4 Flash: n_experts=256, d_ffn~2048, d_in~4096; here use halved
    // dims so total weight bytes ≈ 70 MB instead of 280 MB.
    let n_experts: usize = 256;
    let d_in: usize = 2048;       // %32 ✓ (Q8_0 block)
    let d_ffn: usize = 1024;       // %64 ✓ (mul_mm_id NR0 tile)
    let k_top: usize = 6;          // hard-coded for ne20=6 (DS4 top-6 routing)

    // Synthetic Q8_0 expert weights: [n_experts][d_ffn][d_in] block-packed.
    let nb_per_row = (d_in / 32) * 34;
    let nb_per_expert = nb_per_row * d_ffn;
    let total_weight_bytes = nb_per_expert * n_experts;
    let mut w_q8: Vec<u8> = vec![0u8; total_weight_bytes];
    let mut rng: u32 = 0xC0FFEE_42;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
    for r in 0..(n_experts * d_ffn) {
        for b in 0..(d_in / 32) {
            let off = r * (d_in / 32) * 34 + b * 34;
            w_q8[off] = 0x00; w_q8[off + 1] = 0x3C;  // half 1.0
            for i in 0..32 {
                w_q8[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
            }
        }
    }
    eprintln!("synthetic Q8_0 weights: {} experts × {} d_ffn × {} d_in = {:.1} MB",
        n_experts, d_ffn, d_in, total_weight_bytes as f64 / 1e6);

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_MOE_K_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    let ks: &[usize] = &[1, 2, 4, 8];
    let mut means_ms: Vec<f64> = Vec::with_capacity(ks.len());

    eprintln!(
        "\nmul_mm_id_q8_0_f32 K-amortization bench at n_experts={} d_ffn={} d_in={} (k_top=6 selected/token)",
        n_experts, d_ffn, d_in
    );
    eprintln!("  warmup={} bench_iters={}", warmup_iters, bench_iters);

    for &k in ks {
        // Random K-position routing: each of K tokens picks 6 distinct experts (no
        // dedup across K — realistic for random spec-decode candidates).
        let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
        for _ in 0..k {
            let mut picks: Vec<i32> = (0..n_experts as i32).collect();
            for i in 0..k_top {
                // Fisher-Yates partial shuffle: swap i ↔ random in [i, n_experts).
                let j = i + (next() as usize % (n_experts - i));
                picks.swap(i, j);
            }
            for slot in 0..k_top {
                selected.push(picks[slot]);
            }
        }
        assert_eq!(selected.len(), k * k_top);

        // Random K-position activations [K, d_in].
        let x_k: Vec<f32> = (0..k * d_in)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();

        let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);

        for _ in 0..(warmup_iters + bench_iters) {
            let scope = disp.batch_scope();
            // Uploads OUTSIDE the timing window (mimics production where
            // weights are pre-uploaded and persist across calls).
            let w_db = scope.weight_q8_0_raw(&w_q8, n_experts * d_ffn * d_in);
            let x_db = scope.upload_f32(&x_k);
            // selected as i32 → upload via alloc_i32 + memcpy.
            let sel_db = scope.alloc_i32(k * k_top);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    selected.as_ptr(),
                    sel_db.buffer().contents() as *mut i32,
                    k * k_top,
                );
            }

            let t0 = Instant::now();
            let (tpe, ids) = scope
                .encode_mul_mm_id_map0_k(&sel_db, n_experts, k)
                .expect("map0");
            let dst = scope
                .encode_mul_mm_id_q8_0_k(&w_db, &x_db, &tpe, &ids, n_experts, d_in, d_ffn, k, 1)
                .expect("mul_mm_id_q8_0");
            let _out = scope.flush_and_read(&dst);  // GPU wait
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            iter_ms.push(elapsed_ms);
        }

        let bench_only = &iter_ms[warmup_iters..];
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

    let m1 = means_ms[0];
    eprintln!("\nK-amortization vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ideal_ms = (k as f64) * m1;
        let actual_ms = means_ms[i];
        let ratio = actual_ms / ideal_ms;
        let efficiency = if k == 1 { 100.0 } else {
            (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
        };
        eprintln!(
            "  K={}: actual={:.3} ms, K×K=1={:.3} ms, ratio={:.3}  efficiency={:.0}%",
            k, actual_ms, ideal_ms, ratio, efficiency
        );
    }

    eprintln!(
        "\n*** Compare to projected K=1 MoE production cost ~0.67 ms/layer.\n\
              If mul_mm_id K=8 ratio < 0.5 (efficiency > 50%), the K-amortization\n\
              architectural bet PAYS OFF — proceed with full encode_moe_routed_step_mm_k.\n\
              If ratio > 0.8 (efficiency < 20%), the bench tile sizing or expert\n\
              sparsity defeats K-amortization at this scale — investigate before\n\
              committing to integration."
    );
}

/// Phase 2 MoE-K Step 4 — iq2_xxs + q4_K K-amortization recon. DS4 production
/// uses these quants for MoE experts (iq2_xxs predominantly, q4_K for some
/// down projections). Validates that the same K-amortization architectural
/// pattern that lifted Q8_0 to 86% efficiency at K=8 holds for the lower-bit
/// quants as well.
///
/// Random bytes for the expert weights — the kernel runs even with garbage
/// quant data (no UB beyond producing meaningless arithmetic outputs); only
/// the dispatch TIMING is measured here. Numerical correctness against
/// pair_swiglu+sum6 defers to Step 5's real-model bench.
#[test]
#[allow(non_snake_case)]
fn mul_mm_id_iq2_xxs_q4_K_k_amortization_bench() {
    if std::env::var("DS4_BENCH_MOE_K").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_K unset — skipping. Set DS4_BENCH_MOE_K=1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
    };

    // Dims: d_in / d_ffn divisible by 256 (Q4_K / IQ2_XXS QK_K) AND by 64
    // (mul_mm_id NR0). 256 satisfies both.
    let n_experts: usize = 256;
    let d_in: usize = 2048;       // %256 ✓
    let d_ffn: usize = 1024;       // %256 ✓ and %64 ✓
    let k_top: usize = 6;

    // Quant bytes per row: ceil(d_in/256) * block_bytes.
    let q4k_bytes_per_row = (d_in / 256) * 144;
    let iq2_bytes_per_row = (d_in / 256) * 66;
    let q4k_total = n_experts * d_ffn * q4k_bytes_per_row;
    let iq2_total = n_experts * d_ffn * iq2_bytes_per_row;

    // Random bytes — kernel produces garbage but runs at correct speed.
    let mut rng: u32 = 0x1357_9BDF;
    let mut next_byte = || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 24) as u8
    };
    let mut w_q4k: Vec<u8> = (0..q4k_total).map(|_| next_byte()).collect();
    let mut w_iq2: Vec<u8> = (0..iq2_total).map(|_| next_byte()).collect();
    // Avoid totally-out-of-range scales: for Q4_K stamp the scale bytes with
    // small magnitudes so the kernel's accumulation doesn't OVERFLOW into NaN.
    // (The K-amortization profile doesn't depend on this — only the absolute
    // timing might shift slightly if NaN propagation triggers slow paths.)
    for chunk in w_q4k.chunks_mut(144) {
        // First 2 bytes = half d, next 2 = half dmin. Set both to half(0.5) = 0x3800.
        chunk[0] = 0x00; chunk[1] = 0x38;
        chunk[2] = 0x00; chunk[3] = 0x38;
    }
    for chunk in w_iq2.chunks_mut(66) {
        // First 2 bytes = half d. Set to half(1.0) = 0x3C00.
        chunk[0] = 0x00; chunk[1] = 0x3C;
    }

    eprintln!(
        "\niq2_xxs / q4_K mul_mm_id K-amortization at n_experts={} d_ffn={} d_in={}",
        n_experts, d_ffn, d_in
    );
    eprintln!(
        "  weights: q4_K={:.1} MB, iq2_xxs={:.1} MB",
        q4k_total as f64 / 1e6, iq2_total as f64 / 1e6
    );

    let warmup_iters: usize = 3;
    let bench_iters: usize = std::env::var("DS4_BENCH_MOE_K_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let ks: &[usize] = &[1, 2, 4, 8];

    for variant in ["q4_K", "iq2_xxs"] {
        eprintln!("\n--- mul_mm_id_{} ---", variant);
        let mut means: Vec<f64> = Vec::with_capacity(ks.len());

        for &k in ks {
            // Random routing.
            let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
            for _ in 0..k {
                let mut picks: Vec<i32> = (0..n_experts as i32).collect();
                for i in 0..k_top {
                    let j = i + (next_byte() as usize % (n_experts - i));
                    picks.swap(i, j);
                }
                for slot in 0..k_top { selected.push(picks[slot]); }
            }
            let x_k: Vec<f32> = (0..k * d_in)
                .map(|i| ((i as f32 * 0.0123).sin() * 0.3).clamp(-0.5, 0.5))
                .collect();

            let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);
            for _ in 0..(warmup_iters + bench_iters) {
                let scope = disp.batch_scope();
                let w_db = match variant {
                    "q4_K" => scope.weight_q8_0_raw(&w_q4k, n_experts * d_ffn * d_in),
                    "iq2_xxs" => scope.weight_q8_0_raw(&w_iq2, n_experts * d_ffn * d_in),
                    _ => unreachable!(),
                };
                let x_db = scope.upload_f32(&x_k);
                let sel_db = scope.alloc_i32(k * k_top);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        selected.as_ptr(),
                        sel_db.buffer().contents() as *mut i32,
                        k * k_top,
                    );
                }
                let t0 = Instant::now();
                let (tpe, ids) = scope.encode_mul_mm_id_map0_k(&sel_db, n_experts, k).expect("map0");
                let dst = match variant {
                    "q4_K" => scope.encode_mul_mm_id_q4_K_k(
                        &w_db, &x_db, &tpe, &ids, n_experts, d_in, d_ffn, k, 1
                    ).expect("mul_mm_id_q4_K"),
                    "iq2_xxs" => scope.encode_mul_mm_id_iq2_xxs_k(
                        &w_db, &x_db, &tpe, &ids, n_experts, d_in, d_ffn, k, 1
                    ).expect("mul_mm_id_iq2_xxs"),
                    _ => unreachable!(),
                };
                let _out = scope.flush_and_read(&dst);
                iter_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let bench_only = &iter_ms[warmup_iters..];
            let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
            let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
            let max = bench_only.iter().copied().fold(0.0f64, f64::max);
            eprintln!(
                "  K={}: mean={:.3} ms  [min={:.3} max={:.3}]  μs/K-pos={:.1}",
                k, mean, min, max, mean * 1000.0 / k as f64
            );
            means.push(mean);
        }
        let m1 = means[0];
        eprintln!("  K-amortization vs K=1:");
        for (i, &k) in ks.iter().enumerate() {
            let ratio = means[i] / ((k as f64) * m1);
            let eff = if k == 1 { 100.0 } else {
                (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
            };
            eprintln!(
                "    K={}: ratio={:.3}  efficiency={:.0}%",
                k, ratio, eff
            );
        }
    }
}
