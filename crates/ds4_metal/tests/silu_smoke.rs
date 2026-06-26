//! M4 #330i — macOS-only Metal silu smoke test.
//!
//! Two modes:
//!   - default  → `x / (1 + exp(-x))`           positive-branch always
//!   - fidelity → antirez `x * sigmoid_stable(x)`  (DS4_SILU_FIDELITY=1)
//!
//! Validation: compare against `ds4_engine::forward::silu` at the matching
//! `DS4_SILU_FIDELITY` env setting. Metal `exp` is not spec-required to be
//! byte-identical to libm's `expf`, so we assert L∞ < 1e-6, which is tight
//! enough to catch a wrong-branch (off-by-eintr) regression while
//! allowing the f32 math-lib rounding latitude that Metal documents.
//!
//! Test env handling: SILU_ENV_LOCK serializes the env mutations so
//! parallel tests don't race on DS4_SILU_FIDELITY.

#![cfg(target_os = "macos")]

use ds4_engine::forward::silu as silu_cpu;
use ds4_metal::MetalDispatcher;
use std::sync::Mutex;

static SILU_ENV_LOCK: Mutex<()> = Mutex::new(());

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Sample inputs spanning both branches with extremes and small magnitudes.
fn sample_inputs() -> Vec<f32> {
    vec![
        -100.0, -20.0, -10.0, -5.0, -2.5, -1.0, -0.5, -0.1, -1e-4, -1e-6,
        0.0, 1e-6, 1e-4, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 100.0,
    ]
}

#[test]
fn silu_default_matches_cpu_oracle() {
    let _g = SILU_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DS4_SILU_FIDELITY");
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.silu(&x, false);
    let c: Vec<f32> = x.iter().map(|&xi| silu_cpu(xi)).collect();
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-6, "default silu: L∞ = {err}, expected < 1e-6");
}

#[test]
fn silu_fidelity_matches_antirez_cpu_oracle() {
    let _g = SILU_ENV_LOCK.lock().unwrap();
    std::env::set_var("DS4_SILU_FIDELITY", "1");
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.silu(&x, true);
    let c: Vec<f32> = x.iter().map(|&xi| silu_cpu(xi)).collect();
    std::env::remove_var("DS4_SILU_FIDELITY");
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-6, "fidelity silu: L∞ = {err}, expected < 1e-6");
}

#[test]
fn silu_modes_diverge_on_some_input() {
    // The two kernels MUST produce at least one differing bit-pattern
    // across the sampled inputs — otherwise the fidelity gate is dead-
    // wired and we'd silently revert M4 #311. We don't pin a specific
    // delta because Metal's f32 `exp` has implementation latitude and
    // can collapse the two formulae at any single point.
    let _g = SILU_ENV_LOCK.lock().unwrap();
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let def = d.silu(&x, false);
    let fid = d.silu(&x, true);
    let any_diff = def
        .iter()
        .zip(fid.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        any_diff,
        "default vs fidelity silu produced identical bits on every sampled input — gate is dead-wired"
    );
}

#[test]
fn silu_empty_input() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    assert!(d.silu(&[], false).is_empty());
    assert!(d.silu(&[], true).is_empty());
}
