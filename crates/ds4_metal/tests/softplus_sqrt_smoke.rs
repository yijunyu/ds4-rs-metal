//! M4 #330k — macOS-only Metal softplus_sqrt smoke test.
//!
//! Two modes:
//!   - default  → sqrt(stable_softplus(x))         numerically-stable identity
//!   - fidelity → antirez piecewise (ds4.c:4867)   x>20: sqrt(x);
//!                                                 x<-20: sqrt(exp(x));
//!                                                 else:  sqrt(log1p(exp(x)))
//!
//! Validation: compare against a CPU reference that mirrors the kernel
//! formulas. Metal `exp` / `log` are f32 with implementation latitude;
//! L∞ < 1e-5 because softplus_sqrt has a log() inside that adds ~1 ULP
//! over the silu/sigmoid floor (1e-6).

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn softplus_sqrt_cpu(x: f32, fidelity: bool) -> f32 {
    if fidelity {
        let sp = if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0_f32 + x.exp()).ln()
        };
        sp.sqrt()
    } else {
        let ax = x.abs();
        let sp = x.max(0.0) + (1.0_f32 + (-ax).exp()).ln();
        sp.sqrt()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Sample inputs span both fast-path branches (x>20, x<-20) and the
/// log1p(exp) interior region. Excludes degenerate values.
fn sample_inputs() -> Vec<f32> {
    vec![
        -100.0, -50.0, -25.0, -20.5, -20.0, -19.5, -10.0, -5.0, -1.0, -0.1, -1e-4,
        0.0, 1e-4, 0.1, 1.0, 5.0, 10.0, 19.5, 20.0, 20.5, 25.0, 50.0, 100.0,
    ]
}

#[test]
fn softplus_sqrt_default_matches_cpu() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.softplus_sqrt_fidelity(&x, false);
    let c: Vec<f32> = x.iter().map(|&xi| softplus_sqrt_cpu(xi, false)).collect();
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-5, "default softplus_sqrt: L∞ = {err}, expected < 1e-5");
}

#[test]
fn softplus_sqrt_fidelity_matches_cpu() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let m = d.softplus_sqrt_fidelity(&x, true);
    let c: Vec<f32> = x.iter().map(|&xi| softplus_sqrt_cpu(xi, true)).collect();
    let err = max_abs_diff(&m, &c);
    assert!(err < 1e-5, "fidelity softplus_sqrt: L∞ = {err}, expected < 1e-5");
}

#[test]
fn softplus_sqrt_modes_diverge_on_some_input() {
    // Catches a dead-wired gate (both symbols loading the same pipeline).
    // The two formulas algebraically agree but differ in f32 rounding —
    // at least one sampled input must bit-differ across modes.
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = sample_inputs();
    let def = d.softplus_sqrt_fidelity(&x, false);
    let fid = d.softplus_sqrt_fidelity(&x, true);
    let any_diff = def
        .iter()
        .zip(fid.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        any_diff,
        "default vs fidelity softplus_sqrt produced identical bits everywhere — gate is dead-wired"
    );
}

#[test]
fn softplus_sqrt_empty_input() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    assert!(d.softplus_sqrt_fidelity(&[], false).is_empty());
    assert!(d.softplus_sqrt_fidelity(&[], true).is_empty());
}
