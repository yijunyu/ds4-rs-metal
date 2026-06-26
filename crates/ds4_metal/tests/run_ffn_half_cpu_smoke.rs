//! Phase E M5.4.5.4 — `run_ffn_half` correctness smoke (CPU backend).
//!
//! `run_ffn_half` is generic over `KernelDispatcher + AttentionDispatcher`
//! so it can be unit-tested with `CpuDispatcher` + `CpuAttentionDispatcher`
//! on synthetic small-shape weights (the Metal MoE path has hardcoded
//! production-shape constraints that block synthetic Metal smokes).
//!
//! Asserts that `run_ffn_half` produces the SAME output as inlining the
//! same trait calls in the same order — locking the wiring against
//! drift. Bit-identical (no quantization in the CPU oracle).
//!
//! macOS gating dropped: this smoke runs on any host.

use ds4_engine::attn_dispatch::{
    decode_attn_ffn_post_with, decode_attn_ffn_pre_with, AttnLayerWeights, AttnPrefixOut,
    CpuAttentionDispatcher, LayerParams,
};
use ds4_engine::decode_step::{ComposedLayerWeights, LayerWeights};
use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use ds4_metal::single_buffer_encoder::run_ffn_half;

fn tiny_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 8,
        n_hc: 2,
        n_head: 2,
        head_dim: 4,
        n_rot: 2,
        n_lora_q: 4,
        n_lora_kv: 4,
        hc_sinkhorn_iter: 1,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 0,
        n_out_group: 2,
    }
}

fn build_tiny_layer() -> ComposedLayerWeights {
    let p = tiny_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let shared_dim: u32 = 4;
    let sd = shared_dim as usize;

    let attn = AttnLayerWeights {
        params: p,
        hc_attn_fn: vec![0.05; hc_dim * mix_hc],
        hc_attn_scale: vec![1.0, 0.5, 0.5],
        hc_attn_base: vec![0.0; mix_hc],
        hc_ffn_fn: (0..hc_dim * mix_hc)
            .map(|i| (i as f32 * 0.01).sin() * 0.05)
            .collect(),
        hc_ffn_scale: vec![0.9, 0.4, 0.4],
        hc_ffn_base: (0..mix_hc).map(|i| (i as f32) * 0.005).collect(),
        hc_attn_fn_f16: Vec::new().into(),
        hc_ffn_fn_f16: Vec::new().into(),
        hc_norm_gamma: vec![1.0; d_embd],
        hc_ffn_norm_gamma: (0..d_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).cos() * 0.05)
            .collect(),
        qkv_gamma_q: vec![1.0; 4],
        qkv_gamma_kv: vec![1.0; 4],
        attn_q_a: vec![0.0; 32],
        attn_q_b: vec![0.0; 32],
        attn_kv: vec![0.0; 32],
        attn_q_a_q8: Vec::new().into(),
        attn_q_b_q8: Vec::new().into(),
        attn_kv_q8: Vec::new().into(),
        w_o_a_q8: Vec::new().into(),
        w_o_b_q8: Vec::new().into(),
        w_shared_gate_q8: Vec::new().into(),
        w_shared_up_q8: Vec::new().into(),
        w_shared_down_q8: Vec::new().into(),
        w_o_a: vec![0.0; 32],
        w_o_b: vec![0.0; 32],
        attn_sinks: vec![-1e9; 2],
        w_shared_gate: (0..sd * d_embd)
            .map(|i| (i as f32 * 0.011).sin() * 0.1)
            .collect(),
        w_shared_up: (0..sd * d_embd)
            .map(|i| (i as f32 * 0.013).cos() * 0.1)
            .collect(),
        w_shared_down: (0..d_embd * sd)
            .map(|i| (i as f32 * 0.017).sin() * 0.08)
            .collect(),
        shared_dim,
        shared_clamp: 7.0,
        attn_compressor_kv: Vec::new(),
        attn_compressor_gate: Vec::new(),
        attn_compressor_ape: Vec::new(),
        attn_compressor_norm: Vec::new(),
        indexer_compressor_kv: Vec::new(),
        indexer_compressor_gate: Vec::new(),
        indexer_compressor_ape: Vec::new(),
        indexer_compressor_norm: Vec::new(),
        attn_compressor_kv_f16: Vec::new().into(),
        attn_compressor_gate_f16: Vec::new().into(),
        indexer_compressor_kv_f16: Vec::new().into(),
        indexer_compressor_gate_f16: Vec::new().into(),
        indexer_attn_q_b: Vec::new(),
        indexer_proj: Vec::new(),
        indexer_attn_q_b_f16: Vec::new().into(),
        indexer_proj_f16: Vec::new().into(),
    };

    let n_experts = 4usize;
    let n_experts_used = 2usize;
    let d_ffn = 8usize;
    let moe = LayerWeights {
        n_experts,
        n_experts_used,
        d_model: d_embd,
        d_ffn: std::num::NonZeroUsize::new(d_ffn),
        attn_norm_gamma: vec![1.0; d_embd],
        w_attn: Vec::new(),
        ffn_norm_gamma: vec![1.0; d_embd],
        w_router: (0..n_experts * d_embd)
            .map(|i| (i as f32 * 0.019).sin() * 0.05)
            .collect(),
        w_router_f16: Vec::new().into(),
        router_bias: (0..n_experts).map(|i| (i as f32) * 0.01).collect(),
        routing_table: None,
        w_gate_exps: (0..n_experts * d_ffn * d_embd)
            .map(|i| (i as f32 * 0.007).sin() * 0.05)
            .collect(),
        w_up_exps: (0..n_experts * d_ffn * d_embd)
            .map(|i| (i as f32 * 0.009).cos() * 0.04)
            .collect(),
        w_down_exps: (0..n_experts * d_embd * d_ffn)
            .map(|i| (i as f32 * 0.005).sin() * 0.06)
            .collect(),
    };

    ComposedLayerWeights { attn, moe }
}

#[test]
fn run_ffn_half_matches_inline_cpu_pipeline() {
    let layer = build_tiny_layer();
    let p = &layer.attn.params;
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let pos: u32 = 3;

    // Construct a deterministic AttnPrefixOut (the input to the FFN-half).
    let after_attn_hc: Vec<f32> = (0..hc_dim)
        .map(|i| ((i as f32 * 0.029).sin() * 0.3).clamp(-1.0, 1.0))
        .collect();
    let hc_split_attn: Vec<f32> = (0..2 * n_hc + n_hc * n_hc)
        .map(|i| 0.1 + (i as f32) * 0.05)
        .collect();
    let normed: Vec<f32> = (0..d_embd).map(|i| (i as f32 * 0.031).cos() * 0.5).collect();
    let prefix = AttnPrefixOut {
        after_attn_hc,
        hc_split_attn,
        normed,
    };

    let k = CpuDispatcher;
    let a = CpuAttentionDispatcher;
    let layer_idx = 0usize;

    // ── Path A: through run_ffn_half.
    let out_a = run_ffn_half(&k, &a, layer_idx, &layer, &prefix, pos);

    // ── Path B: inline reference.
    let ffn_pre = decode_attn_ffn_pre_with(&a, &layer.attn, &prefix);
    let h_norm = &ffn_pre.ffn_normed;
    let probs = k.router_logits_batched(&layer.moe.w_router, h_norm, layer.moe.n_experts);
    let (selected, weights) = k.router_finalize(
        &probs,
        &layer.moe.router_bias,
        layer.moe.n_experts_used,
    );
    let silu_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let (routed_out, precomputed_shared): (Vec<f32>, Option<Vec<f32>>) = if silu_fidelity {
        let r = k.moe_routed_step(
            layer_idx as u32,
            h_norm,
            &selected,
            &weights,
            &layer.moe.w_gate_exps,
            &layer.moe.w_up_exps,
            &layer.moe.w_down_exps,
            layer.moe.d_ffn.map_or(0, |n| n.get()),
        );
        (r, None)
    } else {
        let ffn_in_owned;
        let ffn_in: &[f32] = if want_q80 {
            ffn_in_owned = ds4_engine::forward::q8_0_round_trip(&ffn_pre.ffn_normed);
            &ffn_in_owned
        } else {
            &ffn_pre.ffn_normed
        };
        let (r, s) = k.moe_and_shared_chain_batched(
            layer_idx as u32,
            h_norm,
            &selected,
            &weights,
            &layer.moe.w_gate_exps,
            &layer.moe.w_up_exps,
            &layer.moe.w_down_exps,
            layer.moe.d_ffn.map_or(0, |n| n.get()),
            ffn_in,
            &layer.attn.w_shared_gate,
            &layer.attn.w_shared_up,
            &layer.attn.w_shared_down,
            layer.attn.shared_dim,
            want_q80,
        );
        (r, Some(s))
    };
    let out_b = decode_attn_ffn_post_with(
        &k,
        &a,
        &layer.attn,
        &prefix,
        &ffn_pre,
        &routed_out,
        pos,
        precomputed_shared.as_deref(),
    );

    // ── Bit-identical: same operands, same kernels, same order.
    assert_eq!(out_a.len(), out_b.len(), "after_ffn_hc length mismatch");
    for (i, (a, b)) in out_a.iter().zip(&out_b).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "after_ffn_hc[{i}] diverged: helper={a} inline={b}"
        );
    }
}
