//! Phase E M4 (M4 #330p) — `router_finalize` lock-in test.
//!
//! Discovery during M4: `MetalDispatcher::router_finalize` is already a
//! GPU path. The trait override (`lib.rs:467`) delegates to
//! `router_finalize_impl` (`macos.rs:1096`) which uses two GPU kernels:
//!
//!   - `ds4_dsv4_router_finalize_one` (bitonic top-6 select with bias)
//!   - `ds4_dsv4_router_weights_one`  (softmax-normalized weights from probs)
//!
//! `DS4_OP_TRACE` showed this op at 283 us avg wait — already running
//! on GPU end-to-end. M4 just locks in CPU↔GPU equivalence with a
//! smoke test at the production shape (n_experts=256, k=6, has_bias=1).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use ds4_metal::MetalDispatcher;

fn build_inputs() -> (Vec<f32>, Vec<f32>) {
    // DS4 V4 Flash: 256 experts, top-k=6, has_bias=1.
    let n = 256;
    // probs are post-softplus_sqrt (positive). Use a non-trivial spread
    // so the top-6 selection is meaningful.
    let probs: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.0271).sin() * 0.4 + 0.5).max(0.01))
        .collect();
    // bias is centered around zero. Some negative biases should
    // demote certain probs from the top-6 — exercises the bias gating.
    let bias: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.011).cos() * 0.3))
        .collect();
    (probs, bias)
}

#[test]
fn router_finalize_matches_cpu_at_ds4_shape() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuDispatcher;
    let (probs, bias) = build_inputs();
    let k = 6;

    let (gpu_sel, gpu_w) = disp.router_finalize(&probs, &bias, k);
    let (cpu_sel, cpu_w) = cpu.router_finalize(&probs, &bias, k);

    // Top-k selection result must be the same SET (order may vary
    // across the bitonic sort vs CPU sort — antirez gives no guarantee
    // on tie-breaking order). Compare as sorted sets.
    let mut gpu_sel_sorted = gpu_sel.clone();
    let mut cpu_sel_sorted = cpu_sel.clone();
    gpu_sel_sorted.sort();
    cpu_sel_sorted.sort();
    assert_eq!(
        gpu_sel_sorted, cpu_sel_sorted,
        "router_finalize selected set differs: gpu={:?} cpu={:?}",
        gpu_sel, cpu_sel
    );

    // Weights, when re-aligned to the GPU's selected order, should be
    // bit-close to the CPU output. Build CPU's (idx → weight) map.
    use std::collections::HashMap;
    let cpu_map: HashMap<usize, f32> = cpu_sel.iter().zip(cpu_w.iter()).map(|(&i, &w)| (i, w)).collect();
    for (i, &sel) in gpu_sel.iter().enumerate() {
        let cw = *cpu_map.get(&sel).unwrap_or(&f32::NAN);
        let gw = gpu_w[i];
        let abs = (gw - cw).abs();
        assert!(
            abs < 1e-5,
            "weight for expert {sel}: gpu={gw} cpu={cw} |diff|={abs}"
        );
    }
}

#[test]
fn router_finalize_weights_all_positive() {
    // Lock against silent-zero / silent-garbage failure modes.
    // DS4's router uses bias-adjusted top-k + post-softplus_sqrt
    // weights — they're positive but NOT necessarily sum-to-1.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let (probs, bias) = build_inputs();
    let (_sel, weights) = disp.router_finalize(&probs, &bias, 6);
    assert_eq!(weights.len(), 6);
    for &w in &weights {
        assert!(w > 0.0 && w.is_finite(), "non-positive/non-finite weight {w}");
    }
}
