//! M4 #330o — Phase C.3a layer-head qa+rms batched encoder smoke tests.
//!
//! `layer_qa_rms_batched` packs `matvec_f32(w_qa) → rms_norm(gamma_q)` for the
//! per-layer Q-projection head into ONE `MTLCommandBuffer`. These tests verify
//! it produces bit-identical output to running the two ops sequentially through
//! the same dispatcher. Saves one commit+wait+readback per layer (~43 per token
//! once C.3a.2 wires it into the encoder layer loop).
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(d_embd: usize, n_lora_q: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let w_qa: Vec<f32> = (0..n_lora_q * d_embd)
        .map(|i| ((i as f32 * 0.009).cos() * 0.25))
        .collect();
    let gamma_q: Vec<f32> = (0..n_lora_q)
        .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
        .collect();
    (x, w_qa, gamma_q)
}

fn reference_qr_normed(
    disp: &MetalDispatcher,
    x: &[f32],
    w_qa: &[f32],
    gamma_q: &[f32],
    n_lora_q: usize,
    eps_rms: f32,
) -> Vec<f32> {
    use ds4_engine::dispatch::KernelDispatcher;
    let qr = disp.matvec_f32(w_qa, x, n_lora_q);
    disp.rms_norm(&qr, gamma_q, eps_rms)
}

#[test]
fn qa_rms_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let n_lora_q = 64;
    let eps = 1e-5_f32;
    let (x, w_qa, gamma_q) = build_inputs(d_embd, n_lora_q);

    let batched = disp.layer_qa_rms_batched(&x, &w_qa, &gamma_q, n_lora_q, eps);
    let reference = reference_qr_normed(&disp, &x, &w_qa, &gamma_q, n_lora_q, eps);

    assert_eq!(batched.len(), reference.len(), "qr_normed length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "qa_rms batched != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

#[test]
fn qa_rms_matches_sequential_ds4_shape() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // DS4 V4 Flash production shape: d_embd=7168, n_lora_q=1536.
    // Use a scaled-down but layout-faithful variant: d_embd=512, n_lora_q=128.
    let d_embd = 512;
    let n_lora_q = 128;
    let eps = 1e-5_f32;
    let (x, w_qa, gamma_q) = build_inputs(d_embd, n_lora_q);

    let batched = disp.layer_qa_rms_batched(&x, &w_qa, &gamma_q, n_lora_q, eps);
    let reference = reference_qr_normed(&disp, &x, &w_qa, &gamma_q, n_lora_q, eps);

    assert_eq!(batched.len(), reference.len(), "qr_normed length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "qa_rms batched (ds4 shape) != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

/// Different gamma values produce different outputs — locks the test against
/// silent fallback (a future regression where gamma_q is dropped on the floor
/// would still pass against an identity-gamma reference).
#[test]
fn qa_rms_gamma_affects_output() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let n_lora_q = 64;
    let eps = 1e-5_f32;
    let (x, w_qa, gamma_q_a) = build_inputs(d_embd, n_lora_q);
    let gamma_q_b: Vec<f32> = (0..n_lora_q).map(|i| 1.5 + (i as f32) * 0.001).collect();

    let out_a = disp.layer_qa_rms_batched(&x, &w_qa, &gamma_q_a, n_lora_q, eps);
    let out_b = disp.layer_qa_rms_batched(&x, &w_qa, &gamma_q_b, n_lora_q, eps);

    let diff: f32 = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "different gamma produced bit-identical output — rms_norm pass likely skipped (max |diff| = {diff})"
    );
}
