//! Phase E M5.4.5-prep — `BatchScope::encode_moe_and_shared_chain` smoke test.
//!
//! Numerical equivalence against the inherent
//! `moe_and_shared_chain_batched_inherent` requires real GGUF-backed
//! `QuantizedExpertWeights` — those are loaded by
//! `MetalDispatcher::load_expert_weights` and only exercised in
//! `trace_cpu_vs_dequant_real_gguf.rs`. The unified-cb orchestrator
//! (M5.4.5) and the e2e bench exercise this method at production
//! shape.
//!
//! For this smoke test we verify:
//!   1. The method is callable from the public BatchScope API.
//!   2. Without `load_expert_weights`, it returns a clean error
//!      (not a panic). The caller sees the "loaded=0" message
//!      pointing at the init-time setup.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

#[test]
fn encode_moe_and_shared_chain_errors_when_weights_missing() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let scope = disp.batch_scope();

    // No expert weights loaded — method should return an Err pointing
    // at the missing load_expert_weights call, not panic.
    let n_lora_q = 4usize;
    let d_ffn = 64usize;
    let shared_dim = 32u32;
    let d_in = 32usize;
    let d_embd = 32usize;

    let moe_x = vec![0.0f32; d_in];
    let moe_selected = vec![0usize; 6];
    let moe_weights = vec![0.0f32; 6];
    let sh_ffn_norm = vec![0.0f32; d_embd];
    let sh_w_gate = vec![0.0f32; (shared_dim as usize) * d_embd];
    let sh_w_up = vec![0.0f32; (shared_dim as usize) * d_embd];
    let sh_w_down = vec![0.0f32; d_embd * (shared_dim as usize)];

    let _ = n_lora_q; // silence unused

    let result = scope.encode_moe_and_shared_chain(
        0,
        &moe_x,
        &moe_selected,
        &moe_weights,
        d_ffn,
        &sh_ffn_norm,
        &sh_w_gate,
        &sh_w_up,
        &sh_w_down,
        shared_dim,
        false,
    );

    // `(DeferredBuf, DeferredBuf)` isn't Debug, so we can't use expect_err.
    let err = match result {
        Ok(_) => panic!("expected error when no expert weights loaded"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("no QuantizedExpertWeights")
            && msg.contains("loaded=0")
            && msg.contains("load_expert_weights"),
        "error message should pinpoint the missing init step: got {msg}"
    );
}

#[test]
fn encode_moe_and_shared_chain_with_router_bufs_errors_when_weights_missing() {
    // M5 task #98 — parity smoke for the GPU-router-bufs variant.
    // Bit-equivalence with the CPU-slice path is structural (same
    // kernels, same args; only the data binding for selected/weights
    // changes); the real-GGUF check in `trace_cpu_vs_dequant_real_gguf`
    // (opt-in via DS4_GGUF=) and the e2e bench exercise it at
    // production shape. This test just guards the surface so a
    // signature regression fails loudly.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let scope = disp.batch_scope();

    let d_ffn = 64usize;
    let shared_dim = 32u32;
    let d_in = 32usize;
    let d_embd = 32usize;

    let moe_x = vec![0.0f32; d_in];
    let sh_ffn_norm = vec![0.0f32; d_embd];
    let sh_w_gate = vec![0.0f32; (shared_dim as usize) * d_embd];
    let sh_w_up = vec![0.0f32; (shared_dim as usize) * d_embd];
    let sh_w_down = vec![0.0f32; d_embd * (shared_dim as usize)];

    // Allocate the 6-element router output buffers via the scope so
    // their DeferredBuf::n_elements matches the precondition.
    let selected_buf = scope.alloc_i32(6);
    let weights_buf = scope.alloc_f32(6);

    let result = scope.encode_moe_and_shared_chain_with_router_bufs(
        0,
        &moe_x,
        &selected_buf,
        &weights_buf,
        d_ffn,
        &sh_ffn_norm,
        &sh_w_gate,
        &sh_w_up,
        &sh_w_down,
        shared_dim,
        false,
        &[], &[], &[],
    );

    let err = match result {
        Ok(_) => panic!("expected error when no expert weights loaded"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("no QuantizedExpertWeights")
            && msg.contains("loaded=0")
            && msg.contains("load_expert_weights"),
        "error message should pinpoint the missing init step: got {msg}"
    );
}
