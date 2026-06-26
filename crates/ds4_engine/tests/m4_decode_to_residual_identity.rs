//! M4 #330n — Phase C.2 helper-extraction byte-identity gate.
//!
//! `decode_step_with_attn` was refactored into:
//!   1. `decode_step_with_attn_to_residual` — HC expand + per-layer body +
//!      `output_hc_head_one` + state.pos bump → returns `final_hidden`.
//!   2. wrapper tail — `rms_norm(γ_final) → (q8_0_round_trip if env) →
//!      matvec_f32(lm_head)` → returns logits.
//!
//! The `SingleBufferEncoder::decode_token_batched` path (macOS) calls (1)
//! then replaces (2) with a single-`MTLCommandBuffer` `tail_lm_head_batched`
//! call. Encoder-vs-trait equivalence is validated on Mac by the
//! `tail_lm_head_batched_smoke` suite (exact-bits gate on the tail).
//!
//! This Linux-runnable test locks in the OTHER half of that equivalence:
//! `decode_step_with_attn` == `decode_step_with_attn_to_residual` +
//! manual tail composition, bit-for-bit on CPU with two independent
//! `AttnStepState`s. If a future refactor lets the helper and the wrapper
//! drift, this test fails before the encoder smoke can.

use ds4_engine::attn_dispatch::{AttnLayerWeights, CpuAttentionDispatcher, LayerParams};
use ds4_engine::decode_step::{
    decode_step_with_attn, decode_step_with_attn_to_residual, AttnStepState,
    ComposedLayerWeights, ComposedModelWeights, DecodeConfig, LayerWeights,
};
use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};

fn synthetic_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4,
        n_hc: 2,
        n_head: 2,
        head_dim: 4,
        n_rot: 2,
        n_lora_q: 4,
        n_lora_kv: 4,
        hc_sinkhorn_iter: 1,
        hc_eps: 1e-6,
        rms_eps: 1e-6,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 2,
    }
}

fn synthetic_attn_layer(params: LayerParams) -> AttnLayerWeights {
    let n_hc = params.n_hc as usize;
    let d_embd = params.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let q_dim = params.q_dim();
    let n_groups = params.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = 3;
    let out_low_dim = n_groups * n_lora_o;
    let shared_dim = 3u32;
    let sd = shared_dim as usize;

    AttnLayerWeights {
        hc_attn_fn: vec![0.1; hc_dim * mix_hc],
        hc_attn_scale: vec![1.0; 3],
        hc_attn_base: vec![0.0; mix_hc],
        hc_ffn_fn: vec![0.1; hc_dim * mix_hc],
        hc_ffn_scale: vec![1.0; 3],
        hc_ffn_base: vec![0.0; mix_hc],
        hc_attn_fn_f16: Vec::new().into(),
        hc_ffn_fn_f16: Vec::new().into(),
        hc_norm_gamma: vec![1.0; d_embd],
        hc_ffn_norm_gamma: vec![1.0; d_embd],
        qkv_gamma_q: vec![1.0; params.n_lora_q as usize],
        qkv_gamma_kv: vec![1.0; params.n_lora_kv as usize],
        attn_q_a: vec![0.0; d_embd * params.n_lora_q as usize],
        attn_q_b: vec![0.0; (params.n_lora_q as usize) * q_dim],
        attn_kv: vec![0.0; d_embd * params.n_lora_kv as usize],
        attn_q_a_q8: Vec::new().into(),
        attn_q_b_q8: Vec::new().into(),
        attn_kv_q8: Vec::new().into(),
        w_o_a_q8: Vec::new().into(),
        w_o_b_q8: Vec::new().into(),
        w_shared_gate_q8: Vec::new().into(),
        w_shared_up_q8: Vec::new().into(),
        w_shared_down_q8: Vec::new().into(),
        w_o_a: vec![0.1; out_low_dim * group_dim],
        w_o_b: vec![0.2; d_embd * out_low_dim],
        attn_sinks: vec![-1e9; params.n_head as usize],
        w_shared_gate: vec![0.1; sd * d_embd],
        w_shared_up: vec![0.15; sd * d_embd],
        w_shared_down: vec![0.05; d_embd * sd],
        shared_dim,
        shared_clamp: 7.0,
        params,
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
    }
}

fn synthetic_moe_layer(d: usize) -> LayerWeights {
    let n_experts = 4;
    let n_used = 2;
    let d_ffn = d;
    let mut exp_w = vec![0.0f32; d * d];
    for i in 0..d {
        exp_w[i * d + i] = 1.0;
    }
    let mut stacked = Vec::with_capacity(n_experts * d * d);
    for _ in 0..n_experts {
        stacked.extend_from_slice(&exp_w);
    }
    LayerWeights {
        d_model: d,
        d_ffn: std::num::NonZeroUsize::new(d_ffn),
        n_experts,
        n_experts_used: n_used,
        attn_norm_gamma: vec![1.0; d],
        w_attn: vec![0.0; d * d],
        ffn_norm_gamma: vec![1.0; d],
        w_router: vec![0.5; n_experts * d],
        w_router_f16: Vec::new().into(),
        router_bias: vec![0.0; n_experts],
        w_gate_exps: stacked.clone(),
        w_up_exps: stacked.clone(),
        w_down_exps: stacked,
        routing_table: None,
    }
}

fn synthetic_composed(n_layers: usize, vocab: usize) -> ComposedModelWeights {
    let params = synthetic_params();
    let d = params.d_embd as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(ComposedLayerWeights {
            attn: synthetic_attn_layer(params),
            moe: synthetic_moe_layer(d),
        });
    }
    let mut lm_head = vec![0.0f32; vocab * d];
    for v in 0..vocab {
        lm_head[v * d + (v % d)] = 1.0;
    }
    ComposedModelWeights {
        layers,
        final_norm_gamma: vec![1.0; d],
        lm_head,
        lm_head_q8: Vec::new().into(),
        lm_head_q4: Vec::new(),
        vocab_size: vocab,
        d_model: d,
        output_hc_base: None,
        output_hc_fn: None,
        output_hc_scale: None,
    }
}

/// `decode_step_with_attn` and `to_residual + manual tail` must produce
/// bit-equal logits on the same input + same state. This is the contract
/// `SingleBufferEncoder::decode_token_batched` relies on (it composes
/// `to_residual + tail_lm_head_batched` and claims byte-identity to the
/// trait path).
#[test]
fn helper_plus_manual_tail_matches_trait_path_bit_for_bit() {
    let model = synthetic_composed(2, 16);
    let cfg = DecodeConfig::default();
    let raw_cap = 16u32;

    let mut state_trait = AttnStepState::new(&model, raw_cap);
    let mut state_helper = AttnStepState::new(&model, raw_cap);

    let x: Vec<f32> = (0..model.d_model)
        .map(|i| (i as f32) * 0.125 - 0.5)
        .collect();

    let logits_trait = decode_step_with_attn(
        &CpuDispatcher,
        &CpuAttentionDispatcher,
        x.clone(),
        &model,
        &mut state_trait,
        &cfg,
        raw_cap,
    )
    .expect("trait path");

    let final_hidden = decode_step_with_attn_to_residual(
        &CpuDispatcher,
        &CpuAttentionDispatcher,
        x,
        &model,
        &mut state_helper,
        &cfg,
        raw_cap,
    )
    .expect("residual helper");

    let normed = CpuDispatcher.rms_norm(&final_hidden, &model.final_norm_gamma, cfg.eps_rms);
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let normed_for_lm = if want_q80 {
        ds4_engine::forward::q8_0_round_trip(&normed)
    } else {
        normed
    };
    let logits_helper = CpuDispatcher.matvec_f32(&model.lm_head, &normed_for_lm, model.vocab_size);

    assert_eq!(logits_trait.len(), logits_helper.len());
    for (i, (a, b)) in logits_trait.iter().zip(&logits_helper).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "helper-tail diverged from trait path at logit {i}: trait={a} helper={b}"
        );
    }

    assert_eq!(state_trait.pos, state_helper.pos);
    assert_eq!(state_trait.kv_pos, state_helper.kv_pos);
}

/// Over multiple consecutive decode steps the equivalence still holds.
/// Catches state-mutation drift between the helper and the wrapper that
/// might only surface after more than one step.
#[test]
fn helper_tail_matches_trait_path_across_steps() {
    let model = synthetic_composed(3, 8);
    let cfg = DecodeConfig::default();
    let raw_cap = 8u32;

    let mut state_trait = AttnStepState::new(&model, raw_cap);
    let mut state_helper = AttnStepState::new(&model, raw_cap);

    for step in 0..4u32 {
        let x: Vec<f32> = (0..model.d_model)
            .map(|i| (i as f32) * 0.1 + (step as f32) * 0.05)
            .collect();

        let logits_trait = decode_step_with_attn(
            &CpuDispatcher,
            &CpuAttentionDispatcher,
            x.clone(),
            &model,
            &mut state_trait,
            &cfg,
            raw_cap,
        )
        .expect("trait path");

        let final_hidden = decode_step_with_attn_to_residual(
            &CpuDispatcher,
            &CpuAttentionDispatcher,
            x,
            &model,
            &mut state_helper,
            &cfg,
            raw_cap,
        )
        .expect("residual helper");

        let normed =
            CpuDispatcher.rms_norm(&final_hidden, &model.final_norm_gamma, cfg.eps_rms);
        let logits_helper =
            CpuDispatcher.matvec_f32(&model.lm_head, &normed, model.vocab_size);

        for (i, (a, b)) in logits_trait.iter().zip(&logits_helper).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step}: helper-tail diverged from trait path at logit {i}: trait={a} helper={b}"
            );
        }
        assert_eq!(state_trait.pos, state_helper.pos);
        assert_eq!(state_trait.kv_pos, state_helper.kv_pos);
    }
}
