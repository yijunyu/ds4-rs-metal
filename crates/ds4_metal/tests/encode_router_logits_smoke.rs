//! Phase E M5.4.4 — composable `encode_router_logits` smoke test.
//!
//! Bit-identical to the inherent `router_logits_batched` when fed
//! the same inputs — same matvec + softplus_sqrt kernels.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

#[test]
fn encode_router_logits_matches_inherent() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // DS4 V4 Flash production shape: n_experts=256, d_embd=4096.
    // Smaller for test speed but preserving divisibility constraints
    // (n_experts % 4 == 0, d_embd % 4 == 0).
    let n_experts = 256;
    let d_embd = 1024;

    let h_norm: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let w_router: Vec<f32> = (0..n_experts * d_embd)
        .map(|i| ((i as f32) * 0.009).cos() * 0.1)
        .collect();

    // Inherent path.
    let probs_inherent = disp.router_logits_batched(&w_router, &h_norm, n_experts);

    // Composable path via BatchScope.
    let scope = disp.batch_scope();
    let h_b = scope.upload_f32(&h_norm);
    let w_b = scope.upload_f32(&w_router);
    let probs_buf = scope
        .encode_router_logits(&w_b, &h_b, n_experts, false)
        .expect("encode_router_logits");
    let probs_scope = scope.flush_and_read(&probs_buf);

    // Same kernels, same args, same data → bit-identical.
    assert_eq!(probs_inherent.len(), probs_scope.len());
    for (i, (a, b)) in probs_inherent.iter().zip(&probs_scope).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "probs[{i}] inherent={a} scope={b}"
        );
    }
}
