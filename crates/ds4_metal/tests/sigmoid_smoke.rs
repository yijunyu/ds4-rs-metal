//! M4 #330j — macOS-only Metal sigmoid smoke test.
//!
//! Two modes:
//!   - default  → `1 / (1 + exp(-x))`               positive-branch always
//!   - fidelity → antirez `sigmoid_stable(x)` branched on sign
//!
//! Validation: compare against a CPU reference implementation that
//! mirrors `decode_step.rs::output_hc_head_one` (the gated sigmoid
//! that M4 #315 closed). L∞ < 1e-6 — see silu_smoke.rs for the
//! same tolerance rationale.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn sigmoid_cpu(x: f32, fidelity: bool) -> f32 {
    if fidelity {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn sample_inputs() -> Vec<f32> {
    vec![
        -100.0, -20.0, -10.0, -5.0, -2.5, -1.0, -0.5, -0.1, -1e-4, -1e-6,
        0.0, 1e-6, 1e-4, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 100.0,
    ]
}

#[test]
fn sigmoid_default_matches_cpu() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.sigmoid(&x, false);
    let c: Vec<f32> = x.iter().map(|&xi| sigmoid_cpu(xi, false)).collect();
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-6, "default sigmoid: L∞ = {err}, expected < 1e-6");
}

#[test]
fn sigmoid_fidelity_matches_cpu() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.sigmoid(&x, true);
    let c: Vec<f32> = x.iter().map(|&xi| sigmoid_cpu(xi, true)).collect();
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-6, "fidelity sigmoid: L∞ = {err}, expected < 1e-6");
}

#[test]
fn sigmoid_modes_diverge_on_some_input() {
    // Catches dead-wired gate (both symbols loading the same pipeline).
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let def = d.sigmoid(&x, false);
    let fid = d.sigmoid(&x, true);
    let any_diff = def
        .iter()
        .zip(fid.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        any_diff,
        "default vs fidelity sigmoid produced identical bits everywhere — gate is dead-wired"
    );
}

#[test]
fn sigmoid_empty_input() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    assert!(d.sigmoid(&[], false).is_empty());
    assert!(d.sigmoid(&[], true).is_empty());
}
