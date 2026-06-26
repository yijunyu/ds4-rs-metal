//! Phase 2 MoE-K Option A — synthetic smoke for the Q4_K fused K chain.
//!
//! The DS4 V4 Flash q2 quantization uses IQ2_XXS+Q2_K across all 43 layers
//! (per `tests/dump_layer_quants.rs`). The Q4_K variant of pair_swiglu_K +
//! sum6_K is for OTHER GGUF quantizations of DS4 (Q4_K_M, Q4_K_S) which
//! aren't present in `~/models/ds4flash-q2.gguf`.
//!
//! This smoke verifies the Q4_K K-batched kernel COMPILES + DISPATCHES
//! correctly at K∈{1,2,4,8} via synthetic Q4_K bytes. Kernel internals
//! produce mathematically meaningless output on random bytes but the
//! dispatch path itself is exercised end-to-end.
//!
//! Gated by `DS4_BENCH_MOE_Q4K=1`. macOS-only.

#![cfg(target_os = "macos")]

use std::time::Instant;
use ds4_metal::MetalDispatcher;

#[test]
#[allow(non_snake_case)]
fn encode_moe_chain_fused_K_q4_K_synthetic_smoke() {
    if std::env::var("DS4_BENCH_MOE_Q4K").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_Q4K unset — skipping Q4_K smoke. Set DS4_BENCH_MOE_Q4K=1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
    };

    // Production-like dims. d_in / d_ffn must be %256 for q4_K (QK_K=256).
    let n_experts: usize = 256;
    let d_in: usize = 2048;
    let d_ffn: usize = 1024;
    let k_top: usize = 6;

    // Synthetic Q4_K bytes — 144 bytes per 256-quant block.
    let q4k_bytes_per_row = (d_in / 256) * 144;
    let q4k_down_bytes_per_row = (d_ffn / 256) * 144;
    let gate_up_total = n_experts * d_ffn * q4k_bytes_per_row;
    let down_total    = n_experts * d_in  * q4k_down_bytes_per_row;

    let mut rng: u32 = 0xC0DE_C0DE;
    let mut next_byte = || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 24) as u8
    };
    let mut w_gate: Vec<u8> = (0..gate_up_total).map(|_| next_byte()).collect();
    let mut w_up:   Vec<u8> = (0..gate_up_total).map(|_| next_byte()).collect();
    let mut w_down: Vec<u8> = (0..down_total).map(|_| next_byte()).collect();
    // Q4_K block header: 2 bytes d (half) + 2 bytes dmin (half) + 12 bytes
    // scales. Set d/dmin to half(0.5) to keep values bounded.
    for chunk in w_gate.chunks_mut(144) { chunk[0] = 0; chunk[1] = 0x38; chunk[2] = 0; chunk[3] = 0x38; }
    for chunk in w_up.chunks_mut(144)   { chunk[0] = 0; chunk[1] = 0x38; chunk[2] = 0; chunk[3] = 0x38; }
    for chunk in w_down.chunks_mut(144) { chunk[0] = 0; chunk[1] = 0x38; chunk[2] = 0; chunk[3] = 0x38; }

    eprintln!(
        "\nencode_moe_chain_fused_K_q4_K_q4_K smoke at n_experts={} d_in={} d_ffn={}",
        n_experts, d_in, d_ffn
    );
    eprintln!(
        "  weights: gate={:.1} MB, up={:.1} MB, down={:.1} MB",
        gate_up_total as f64 / 1e6,
        gate_up_total as f64 / 1e6,
        down_total as f64 / 1e6
    );

    let warmup_iters: usize = 3;
    let bench_iters: usize = 10;
    let ks: &[usize] = &[1, 2, 4, 8];
    let mut means_ms: Vec<f64> = Vec::with_capacity(ks.len());

    for &k in ks {
        let mut selected: Vec<i32> = Vec::with_capacity(k * k_top);
        for _ in 0..k {
            let mut picks: Vec<i32> = (0..n_experts as i32).collect();
            for i in 0..k_top {
                let j = i + (next_byte() as usize % (n_experts - i));
                picks.swap(i, j);
            }
            for slot in 0..k_top { selected.push(picks[slot]); }
        }
        let weights: Vec<f32> = (0..k * k_top)
            .map(|_| (next_byte() as f32) / 255.0 + 0.05).collect();
        let x_k: Vec<f32> = (0..k * d_in)
            .map(|i| ((i as f32 * 0.013).sin() * 0.3).clamp(-0.5, 0.5))
            .collect();

        let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);
        for _ in 0..(warmup_iters + bench_iters) {
            let scope = disp.batch_scope();
            let wg = scope.weight_q8_0_raw(&w_gate, n_experts * d_ffn * d_in);
            let wu = scope.weight_q8_0_raw(&w_up,   n_experts * d_ffn * d_in);
            let wd = scope.weight_q8_0_raw(&w_down, n_experts * d_in  * d_ffn);
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
            let out = scope.encode_moe_chain_fused_K_q4_K_q4_K(
                &x_db, &sel_db, &wt_db,
                &wg, &wu, &wd,
                n_experts, d_in, d_ffn, k,
            ).expect("encode_moe_chain_fused_K_q4_K_q4_K");
            let result = scope.flush_and_read(&out);
            iter_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            if iter_ms.len() == warmup_iters + 1 {
                assert_eq!(result.len(), k * d_in, "q4_K_q4_K out shape");
                let nan_count = result.iter().filter(|v| !v.is_finite()).count();
                let max_abs = result.iter().filter(|v| v.is_finite())
                    .fold(0.0f32, |a, &v| a.max(v.abs()));
                eprintln!("    K={}: max_abs={:.3e}  nan_count={}", k, max_abs, nan_count);
            }
        }
        let bench_only = &iter_ms[warmup_iters..];
        let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
        eprintln!("  K={}: mean={:.3} ms   μs/K-pos={:.1}",
            k, mean, mean * 1000.0 / k as f64);
        means_ms.push(mean);
    }

    let m1 = means_ms[0];
    eprintln!("\nQ4_K fused-K chain K-amortization (synthetic) vs K=1:");
    for (i, &k) in ks.iter().enumerate() {
        let ratio = means_ms[i] / ((k as f64) * m1);
        let eff = if k == 1 { 100.0 } else {
            (1.0 - (ratio - 1.0 / k as f64) / (1.0 - 1.0 / k as f64)) * 100.0
        };
        eprintln!(
            "  K={}: actual={:.3} ms   ratio={:.3}  eff={:.0}%",
            k, means_ms[i], ratio, eff
        );
    }
}
