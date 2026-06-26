//! M4 #330m — Phase C tail-batched lm_head encoder smoke tests.
//!
//! `tail_lm_head_batched` packs `rms_norm(γ)` → (optional `q8_0_round_trip`) →
//! `matvec_f32(lm_head)` into ONE `MTLCommandBuffer`. These tests verify it
//! produces bit-identical logits to running the three ops sequentially through
//! the same dispatcher. The point of batching is to save commit + wait +
//! readback round-trips, NOT to change numerics — so the gate is exact
//! equality, not a tolerance.
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(d_embd: usize, vocab_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5 + 0.1).clamp(-2.0, 2.0))
        .collect();
    let gamma: Vec<f32> = (0..d_embd)
        .map(|i| 1.0 + (i as f32 * 0.007).cos() * 0.05)
        .collect();
    let lm_head: Vec<f32> = (0..vocab_size * d_embd)
        .map(|i| ((i as f32 * 0.011).cos() * 0.3))
        .collect();
    (x, gamma, lm_head)
}

fn reference_logits(
    disp: &MetalDispatcher,
    x: &[f32],
    gamma: &[f32],
    eps: f32,
    want_q80: bool,
    lm_head: &[f32],
    vocab_size: usize,
) -> Vec<f32> {
    use ds4_engine::dispatch::KernelDispatcher;
    let normed = disp.rms_norm(x, gamma, eps);
    let normed_for_lm = if want_q80 {
        disp.q8_0_round_trip(&normed)
    } else {
        normed
    };
    disp.matvec_f32(lm_head, &normed_for_lm, vocab_size)
}

#[test]
fn tail_matches_sequential_no_q80() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let vocab_size = 64;
    let eps = 1e-5_f32;
    let (x, gamma, lm_head) = build_inputs(d_embd, vocab_size);

    let batched = disp.tail_lm_head_batched(&x, &gamma, eps, false, &lm_head, vocab_size);
    let reference = reference_logits(&disp, &x, &gamma, eps, false, &lm_head, vocab_size);

    assert_eq!(batched.len(), reference.len(), "logits length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "tail batched != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

#[test]
fn tail_matches_sequential_with_q80() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let vocab_size = 64;
    let eps = 1e-5_f32;
    let (x, gamma, lm_head) = build_inputs(d_embd, vocab_size);

    let batched = disp.tail_lm_head_batched(&x, &gamma, eps, true, &lm_head, vocab_size);
    let reference = reference_logits(&disp, &x, &gamma, eps, true, &lm_head, vocab_size);

    assert_eq!(batched.len(), reference.len(), "logits length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "tail batched (q80) != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

/// q80 path measurably perturbs the logits vs no-q80 — locks the test against
/// silent fallback (e.g., a future regression where `want_q80=true` quietly
/// skips the round-trip).
#[test]
fn q80_path_differs_from_default() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let vocab_size = 64;
    let eps = 1e-5_f32;
    let (x, gamma, lm_head) = build_inputs(d_embd, vocab_size);

    let no_q80 = disp.tail_lm_head_batched(&x, &gamma, eps, false, &lm_head, vocab_size);
    let with_q80 = disp.tail_lm_head_batched(&x, &gamma, eps, true, &lm_head, vocab_size);

    let diff: f32 = no_q80
        .iter()
        .zip(&with_q80)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "q80 path produced bit-identical logits to default — round-trip likely skipped (max |diff| = {diff})"
    );
}
