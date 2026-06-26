//! macOS-only Metal matvec_f32 smoke test.
//!
//! Validates the refactored `matvec_f32_impl` (now using `specialized_pipeline`
//! with FC indices 600/601 to dispatch `ds4_kernel_mul_mv_f32_f32_4`) against
//! a CPU reference at small sizes.
//!
//! Tolerance: 1e-4 absolute. The Metal kernel uses simdgroup reductions over
//! f32; small accumulation order differences vs the CPU dot-product can yield
//! ~1e-6 per term, scaling linearly with d_in.

#![cfg(target_os = "macos")]

use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use ds4_metal::MetalDispatcher;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_case(d_in: usize, d_out: usize, tol: f32) {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuDispatcher;

    let w: Vec<f32> = (0..d_out * d_in)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5))
        .collect();
    let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.07 - 0.3).cos()).collect();

    let metal_out = dispatcher.matvec_f32(&w, &x, d_out);
    let cpu_out = cpu.matvec_f32(&w, &x, d_out);

    let err = max_abs_diff(&metal_out, &cpu_out);
    assert!(
        err < tol,
        "matvec_f32 d_in={d_in} d_out={d_out}: err = {err}, expected < {tol}"
    );
}

#[test]
fn matvec_f32_128_64() {
    // d_in=128 hits nxpsg=8 (128%128==0, 128%256!=0). nsg=clamp(ceil(128/128),1,8)=1.
    run_case(128, 64, 1e-4);
}

#[test]
fn matvec_f32_256_64() {
    // d_in=256 hits nxpsg=16 (256%256==0). nsg=clamp(ceil(256/128),1,8)=2.
    run_case(256, 64, 1e-4);
}

#[test]
fn matvec_f32_64_32() {
    // d_in=64 hits nxpsg=4 (64%128!=0, 64%4==0). nsg=clamp(ceil(64/128),1,8)=1.
    run_case(64, 32, 1e-4);
}

/// M4 #330 — Weight-cache hit path: call matvec_f32 twice with the same
/// stable weight slice (same `(ptr, len)`) and a different `x` each time.
/// The cache should serve the second call's weight buffer from `MetalState`
/// without re-uploading, and the numerical result must match a fresh
/// CPU dot-product on each input.
#[test]
fn matvec_f32_weight_cache_reused_across_calls() {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuDispatcher;
    let d_in = 256usize;
    let d_out = 64usize;
    let w: Vec<f32> = (0..d_out * d_in)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5))
        .collect();
    let x1: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.07 - 0.3).cos()).collect();
    let x2: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.11 + 0.2).sin()).collect();

    let m1 = dispatcher.matvec_f32(&w, &x1, d_out);
    let m2 = dispatcher.matvec_f32(&w, &x2, d_out);

    let c1 = cpu.matvec_f32(&w, &x1, d_out);
    let c2 = cpu.matvec_f32(&w, &x2, d_out);

    assert!(max_abs_diff(&m1, &c1) < 1e-4, "first call diverges");
    assert!(
        max_abs_diff(&m2, &c2) < 1e-4,
        "second call diverges — weight cache likely served stale data"
    );
}

/// M4 #330 — DS4_WEIGHT_CACHE=0 disables the cache; per-call upload
/// remains numerically correct.
#[test]
fn matvec_f32_weight_cache_disabled() {
    // Safety: tests in this crate are not parallel-sensitive to this
    // env (only this test reads it), but be tidy and restore.
    let prev = std::env::var("DS4_WEIGHT_CACHE").ok();
    // SAFETY: set/remove env in single-threaded test path.
    unsafe {
        std::env::set_var("DS4_WEIGHT_CACHE", "0");
    }
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuDispatcher;
    let w: Vec<f32> = (0..128 * 64).map(|i| (i as f32 * 0.011).sin()).collect();
    let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).cos()).collect();
    let m = dispatcher.matvec_f32(&w, &x, 64);
    let c = cpu.matvec_f32(&w, &x, 64);
    assert!(max_abs_diff(&m, &c) < 1e-4);
    unsafe {
        match prev {
            Some(v) => std::env::set_var("DS4_WEIGHT_CACHE", v),
            None => std::env::remove_var("DS4_WEIGHT_CACHE"),
        }
    }
}
