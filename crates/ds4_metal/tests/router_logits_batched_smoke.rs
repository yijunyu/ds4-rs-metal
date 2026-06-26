//! M4 #330o — Phase C.3e router_logits_batched smoke tests.
//!
//! `router_logits_batched` packs:
//!   logits = matvec_f32(w_router, h_norm, n_experts)
//!   probs  = softplus_sqrt(logits)
//! into ONE `MTLCommandBuffer`. These tests verify bit-identical `probs`
//! against running the two ops sequentially through the same dispatcher.
//! Saves 1 commit+wait+readback per layer once C.3e is wired into
//! `decode_step_with_attn_to_residual`.
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(n_experts: usize, d_in: usize) -> (Vec<f32>, Vec<f32>) {
    let h_norm: Vec<f32> = (0..d_in)
        .map(|i| ((i as f32 * 0.041).sin() * 0.31 + 0.07).clamp(-2.0, 2.0))
        .collect();
    let w_router: Vec<f32> = (0..n_experts * d_in)
        .map(|i| ((i as f32 * 0.0067).cos() * 0.19))
        .collect();
    (h_norm, w_router)
}

fn reference(
    disp: &MetalDispatcher,
    w_router: &[f32],
    h_norm: &[f32],
    n_experts: usize,
) -> Vec<f32> {
    use ds4_engine::dispatch::KernelDispatcher;
    let logits = disp.matvec_f32(w_router, h_norm, n_experts);
    disp.softplus_sqrt(&logits)
}

#[test]
fn router_logits_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_experts = 32usize;
    let d_in = 64usize;
    let (h_norm, w_router) = build_inputs(n_experts, d_in);

    let probs_b = disp.router_logits_batched(&w_router, &h_norm, n_experts);
    let probs_r = reference(&disp, &w_router, &h_norm, n_experts);

    assert_eq!(probs_b.len(), probs_r.len(), "probs len mismatch");
    for (i, (b, r)) in probs_b.iter().zip(&probs_r).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "probs diverged at i={i}: batched={b} reference={r}"
        );
    }
}

#[test]
fn router_logits_matches_sequential_ds4_shape() {
    // Scaled-down DS4 V4 Flash router shape: ds4 production uses
    //   n_experts = 256, d_embd = 7168.
    // Keep the test fast with a layout-faithful compact variant.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_experts = 64usize;
    let d_in = 256usize;
    let (h_norm, w_router) = build_inputs(n_experts, d_in);

    let probs_b = disp.router_logits_batched(&w_router, &h_norm, n_experts);
    let probs_r = reference(&disp, &w_router, &h_norm, n_experts);

    for (i, (b, r)) in probs_b.iter().zip(&probs_r).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "probs ds4 shape diverged at i={i}: batched={b} reference={r}"
        );
    }
}

/// Independence canary: swapping `w_router` MUST change `probs`. Locks
/// against a silent regression where the batched primitive bypasses the
/// matvec entirely (e.g. feeds the input directly into softplus_sqrt).
#[test]
fn router_logits_depends_on_w_router() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_experts = 32usize;
    let d_in = 64usize;
    let (h_norm, w_router_a) = build_inputs(n_experts, d_in);
    let w_router_b: Vec<f32> = (0..n_experts * d_in)
        .map(|i| ((i as f32 * 0.0119).sin() * 0.27))
        .collect();

    let probs_a = disp.router_logits_batched(&w_router_a, &h_norm, n_experts);
    let probs_b = disp.router_logits_batched(&w_router_b, &h_norm, n_experts);

    let max_diff: f32 = probs_a
        .iter()
        .zip(&probs_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "different w_router produced bit-identical probs — matvec likely skipped (max |diff| = {max_diff})"
    );
}
