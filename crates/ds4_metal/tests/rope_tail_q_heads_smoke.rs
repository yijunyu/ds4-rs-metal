//! Phase F task #86 — `BatchScope::rope_tail_q_heads_in_place` smoke.
//!
//! Validates that applying rope_tail to all n_head heads of a
//! `[n_head, head_dim]` row-major buffer in ONE GPU dispatch matches
//! the CPU loop currently in `encode_first_half_inner` (per-head
//! `apply_rope_tail_with_table`).
//!
//! Once this is solid, `encode_first_half_inner`'s rope_q forward +
//! backward CPU passes become GPU ops in the same shared scope —
//! one step closer to the antirez one-cb-per-token pattern.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    apply_rope_tail_with_table, precompute_rope_tail_table, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 8,
        head_dim: 128,
        n_rot: 16,
        n_lora_q: 64,
        n_lora_kv: 128,
        hc_sinkhorn_iter: 5,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 1,
    }
}

fn run_case(backward: bool, pos: u32) {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let n_rot = p.n_rot as usize;

    // Deterministic q_heads buffer.
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i as f32 * 0.013).sin() * 0.4).clamp(-1.5, 1.5))
        .collect();

    // CPU reference: per-head loop matching the encode_first_half_inner code.
    let mut q_cpu = q.clone();
    let table = precompute_rope_tail_table(
        n_rot,
        pos,
        backward,
        p.rope_freq_base,
        p.rope_freq_scale,
        p.rope_ext_factor,
        p.rope_attn_factor,
        p.rope_orig_ctx,
    );
    for head in q_cpu.chunks_mut(head_dim) {
        let tail = &mut head[head_dim - n_rot..];
        apply_rope_tail_with_table(tail, &table, n_rot);
    }

    // GPU path: ONE BatchScope op for all n_head heads.
    let scope = disp.batch_scope();
    let q_buf = scope.upload_f32(&q);
    scope
        .rope_tail_q_heads_in_place(&q_buf, n_head, head_dim, &p, pos, backward)
        .expect("rope_tail_q_heads_in_place");
    let q_gpu = scope.flush_and_read(&q_buf);

    assert_eq!(q_gpu.len(), q_cpu.len());
    // Pure fp32 sin/cos through Metal vs CPU; expect very tight match.
    let mut max_abs = 0.0f32;
    for (i, (g, c)) in q_gpu.iter().zip(&q_cpu).enumerate() {
        let d = (g - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 5e-6,
            "rope_q[{i}] (backward={backward} pos={pos}): gpu={g} cpu={c} |diff|={d}"
        );
    }
    eprintln!(
        "rope_tail_q_heads backward={backward} pos={pos}: max_abs={max_abs}"
    );
}

#[test]
fn rope_tail_q_heads_forward_matches_cpu() {
    run_case(false, 5);
}

#[test]
fn rope_tail_q_heads_backward_matches_cpu() {
    run_case(true, 5);
}

#[test]
fn rope_tail_q_heads_forward_pos_0_matches_cpu() {
    run_case(false, 0);
}

#[test]
fn rope_tail_q_heads_forward_pos_31_matches_cpu() {
    run_case(false, 31);
}
