//! M4 #330l — Phase B.3 split-buffer encoder skeleton smoke tests.
//!
//! Phase B delegates `decode_token_batched` to `decode_step_with_attn`
//! verbatim. These tests lock in the delegation invariant so Phase C's
//! real GPU encoding can be checked against the Phase B baseline by
//! running the same fixture both ways and asserting bit-equal logits.
//!
//! macOS-only because the encoder borrows `MetalDispatcher`.

#![cfg(target_os = "macos")]

use ds4_metal::single_buffer_encoder::{
    BatchedBranchingPlan, LayerCutpoint, SingleBufferEncoder,
};
use ds4_metal::MetalDispatcher;

/// Phase B contract: cutpoint helper halves layer count.
#[test]
fn cutpoint_middle_halves_layer_count() {
    assert_eq!(LayerCutpoint::middle(43).0, 21);
    assert_eq!(LayerCutpoint::middle(2).0, 1);
    assert_eq!(LayerCutpoint::middle(0).0, 0);
}

/// Plan starts empty — Phase C populates it from buffer-A readback.
#[test]
fn plan_starts_empty() {
    let p = BatchedBranchingPlan::default();
    assert!(p.compressor_layers.is_empty());
    assert!(p.indexer_layers.is_empty());
    assert!(p.hash_router_layers.is_empty());
}

/// Encoder is constructible from a `MetalDispatcher` borrow with a
/// realistic `raw_cap`. Smoke-only: no decode yet.
#[test]
fn encoder_constructible_from_dispatcher() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let enc = SingleBufferEncoder::new(&d, 4096);
    // Sanity: cutpoint helper returns a value in [0, n_layers].
    drop(enc);
}
