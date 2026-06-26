//! M4 #330o — Phase C.3g shared_expert_impl single-cmdbuf smoke tests.
//!
//! `shared_expert_impl` (the path used when `DS4_SILU_FIDELITY=1`) now
//! packs `(w_gate · ffn_norm)` + `(w_up · ffn_norm)` into ONE
//! `MTLCommandBuffer` via `shared_expert_gate_up_batched`. The CPU
//! `shared_expert_swiglu` post-amble is unchanged.
//!
//! These tests assert that packed output equals the sequential
//! `matvec_f32` × 2 baseline BIT-FOR-BIT (`to_bits()` equality). Both
//! paths share the same `ds4_kernel_mul_mv_f32_f32_4` pipeline + the
//! same FC constants, so byte-identity is the right gate.
//!
//! macOS-only — requires a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{AttentionDispatcher, LayerParams};
use ds4_engine::dispatch::KernelDispatcher;
use ds4_metal::MetalDispatcher;

fn params(d_embd: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd,
        n_hc: 1,
        n_head: 4,
        head_dim: 8,
        n_rot: 4,
        n_lora_q: 16,
        n_lora_kv: 16,
        hc_sinkhorn_iter: 0,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 2,
    }
}

/// Re-derive the expected output by running gate and up matvecs as two
/// separate `matvec_f32` calls (matches pre-C.3g semantics exactly,
/// because the post-amble swiglu is unchanged and runs CPU-side).
fn shared_expert_swiglu_seq(
    dispatcher: &MetalDispatcher,
    ffn_norm: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    sd: usize,
) -> Vec<f32> {
    let g = dispatcher.matvec_f32(w_gate, ffn_norm, sd);
    let u = dispatcher.matvec_f32(w_up, ffn_norm, sd);
    // SwiGLU: silu_default(g[i]) * u[i] (the M4 #313 default-OFF branch).
    // We compare via `shared_expert` which runs the full impl, so we
    // only need the matvec parity check here — the swiglu post-amble
    // is shared with the packed path.
    let mut out = vec![0.0f32; sd];
    for i in 0..sd {
        let gi = g[i];
        // default-branch silu: gi * sigmoid(gi) = gi / (1 + exp(-gi)).
        let s = gi / (1.0 + (-gi).exp());
        out[i] = s * u[i];
    }
    out
}

#[test]
fn shared_expert_packed_matches_sequential_small() {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(64);
    let sd: u32 = 32; // shared_dim, %2 ok; d_embd=64 %4 ok.
    let d_embd = p.d_embd as usize;
    let ffn_norm: Vec<f32> = (0..d_embd).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
    let w_gate: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.013).cos() * 0.25)
        .collect();
    let w_up: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.019 + 0.1).sin() * 0.3)
        .collect();

    let packed = dispatcher.shared_expert(&p, &ffn_norm, &w_gate, &w_up, sd, /*_clamp*/ 0.0);
    let seq = shared_expert_swiglu_seq(&dispatcher, &ffn_norm, &w_gate, &w_up, sd as usize);

    assert_eq!(packed.len(), seq.len(), "output len mismatch");
    for (i, (pv, sv)) in packed.iter().zip(&seq).enumerate() {
        assert_eq!(
            pv.to_bits(),
            sv.to_bits(),
            "shared_expert diverged at i={i}: packed={pv} sequential={sv}"
        );
    }
}

#[test]
fn shared_expert_packed_matches_sequential_ds4_shape() {
    // Scaled-down DS4 V4 Flash shared-expert shape: production
    //   shared_dim = 2048, d_embd = 7168. Use a layout-faithful
    //   smaller variant.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(256);
    let sd: u32 = 128;
    let d_embd = p.d_embd as usize;
    let ffn_norm: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.029).sin() * 0.41).clamp(-1.5, 1.5))
        .collect();
    let w_gate: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.0083).cos() * 0.19)
        .collect();
    let w_up: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.0061 - 0.2).sin() * 0.21)
        .collect();

    let packed = dispatcher.shared_expert(&p, &ffn_norm, &w_gate, &w_up, sd, 0.0);
    let seq = shared_expert_swiglu_seq(&dispatcher, &ffn_norm, &w_gate, &w_up, sd as usize);

    for (i, (pv, sv)) in packed.iter().zip(&seq).enumerate() {
        assert_eq!(
            pv.to_bits(),
            sv.to_bits(),
            "shared_expert ds4-shape diverged at i={i}: packed={pv} sequential={sv}"
        );
    }
}

/// Independence canary: swapping `w_up` MUST change `out`. Locks against
/// a silent regression where the second pass (up matvec) is dropped and
/// the output ends up depending only on `w_gate`.
#[test]
fn shared_expert_packed_depends_on_w_up() {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(64);
    let sd: u32 = 32;
    let d_embd = p.d_embd as usize;
    let ffn_norm: Vec<f32> = (0..d_embd).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
    let w_gate: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.013).cos() * 0.25)
        .collect();
    let w_up_a: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.019 + 0.1).sin() * 0.3)
        .collect();
    let w_up_b: Vec<f32> = (0..sd as usize * d_embd)
        .map(|i| ((i as f32) * 0.027 - 0.4).cos() * 0.33)
        .collect();

    let out_a = dispatcher.shared_expert(&p, &ffn_norm, &w_gate, &w_up_a, sd, 0.0);
    let out_b = dispatcher.shared_expert(&p, &ffn_norm, &w_gate, &w_up_b, sd, 0.0);
    let max_diff: f32 = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "different w_up produced bit-identical output — up matvec likely skipped (max |diff| = {max_diff})"
    );
}
