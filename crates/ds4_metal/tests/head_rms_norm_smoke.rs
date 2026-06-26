//! M4 #330e — macOS-only Metal head_rms_norm smoke test.
//!
//! Validates `MetalDispatcher::head_rms_norm` against the CPU oracle from
//! decode_step.rs:656-663 (f32 accumulator path; matches the default-OFF
//! state of DS4_HEAD_RMS_F64_FIDELITY).
//!
//! Tolerance: 1e-4 absolute. The Metal kernel uses simdgroup reductions
//! over f32; small accumulation-order differences vs the CPU loop can yield
//! ~1e-6 per term, scaling linearly with head_dim.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

/// Matches decode_step.rs:656-663 verbatim (f32 accumulator).
fn head_rms_cpu_oracle(x: &[f32], n_head: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    for h in 0..n_head {
        let chunk = &mut out[h * head_dim..(h + 1) * head_dim];
        let ss: f32 = chunk.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for v in chunk.iter_mut() {
            *v *= scale;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_case(n_head: usize, head_dim: usize, eps: f32, tol: f32) {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n = n_head * head_dim;
    // Adversarial-ish input: mix small and large magnitudes per head so the
    // per-head RMS is genuinely distinct across heads.
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let h = (i / head_dim) as f32;
            let phase = (h * 0.31 + (i % head_dim) as f32 * 0.07).sin();
            phase * (1.0 + h * 0.5)
        })
        .collect();

    let metal_out = dispatcher.head_rms_norm(&x, n_head, head_dim, eps);
    let cpu_out = head_rms_cpu_oracle(&x, n_head, head_dim, eps);

    assert_eq!(metal_out.len(), cpu_out.len());
    let err = max_abs_diff(&metal_out, &cpu_out);
    assert!(
        err < tol,
        "head_rms_norm n_head={n_head} head_dim={head_dim}: err = {err}, expected < {tol}"
    );
}

#[test]
fn head_rms_norm_8_heads_64_dim() {
    // n_head=8, head_dim=64: tcount picks 64; one simdgroup per head.
    run_case(8, 64, 1e-6, 1e-4);
}

#[test]
fn head_rms_norm_16_heads_128_dim() {
    // n_head=16, head_dim=128: tcount=128, four simdgroups per head.
    run_case(16, 128, 1e-6, 1e-4);
}

#[test]
fn head_rms_norm_64_heads_192_dim() {
    // DS4 V4 Flash decode shape: n_head=64, head_dim=192 (q_dim=12288).
    run_case(64, 192, 1e-6, 5e-4);
}
