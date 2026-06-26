//! M4 #225 Phase B1 — composed (KernelDispatcher × AttentionDispatcher)
//! decode step harness.
//!
//! Drives `decode_step_with_attn` over a synthetic 2-layer model for N
//! consecutive decode positions, then asserts:
//! - two independent CPU dispatcher pairs produce byte-identical logits
//!   (proves the composed step is deterministic);
//! - KV-cache state advances by 1 per step on every layer;
//! - logit sequence over N steps is stable shape.
//!
//! In Phase B3 the second dispatcher pair becomes `(MetalDispatcher,
//! MetalDispatcher)` (it implements both traits); the test body is the
//! same shape with a 1e-5 tolerance check swapped in around the logits.

use ds4_engine::attn_dispatch::{AttnLayerWeights, CpuAttentionDispatcher, LayerParams};
use ds4_engine::decode_step::{
    decode_step_with_attn, AttnStepState, ComposedLayerWeights, ComposedModelWeights, DecodeConfig,
    LayerWeights,
};
use ds4_engine::dispatch::CpuDispatcher;

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
    // Stacked identity expert weights (one expert reproduces input).
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

#[test]
fn cpu_x_cpu_composed_step_is_bit_equal() {
    let model = synthetic_composed(2, 16);
    let cfg = DecodeConfig::default();
    let raw_cap = 16u32;

    let k_a = CpuDispatcher;
    let k_b = CpuDispatcher;
    let a_a = CpuAttentionDispatcher;
    let a_b = CpuAttentionDispatcher;

    let mut state_a = AttnStepState::new(&model, raw_cap);
    let mut state_b = AttnStepState::new(&model, raw_cap);

    let x: Vec<f32> = (0..model.d_model)
        .map(|i| (i as f32) * 0.125 - 0.5)
        .collect();

    let logits_a =
        decode_step_with_attn(&k_a, &a_a, x.clone(), &model, &mut state_a, &cfg, raw_cap)
            .expect("step A");
    let logits_b =
        decode_step_with_attn(&k_b, &a_b, x, &model, &mut state_b, &cfg, raw_cap).expect("step B");

    assert_eq!(logits_a.len(), model.vocab_size);
    assert_eq!(
        logits_a, logits_b,
        "CPU×CPU composed logits must be bit-equal"
    );
    assert_eq!(state_a.kv_pos, state_b.kv_pos);
    assert_eq!(state_a.pos, state_b.pos);
    assert_eq!(state_a.pos, 1);
}

#[test]
fn composed_step_advances_kv_per_layer() {
    let model = synthetic_composed(3, 8);
    let cfg = DecodeConfig::default();
    let raw_cap = 8u32;

    let mut state = AttnStepState::new(&model, raw_cap);
    let x = vec![0.25f32; model.d_model];

    for step in 0..4u32 {
        let logits = decode_step_with_attn(
            &CpuDispatcher,
            &CpuAttentionDispatcher,
            x.clone(),
            &model,
            &mut state,
            &cfg,
            raw_cap,
        )
        .expect("composed step");
        assert_eq!(logits.len(), model.vocab_size);
        for &kv_pos in &state.kv_pos {
            assert_eq!(kv_pos, step + 1, "each layer advances KV write pos by 1");
        }
        assert_eq!(state.pos, step + 1);
    }
}

/// Build a synthetic 1-layer ratio==4 model with non-empty main-compressor +
/// indexer weights, so `decode_step_with_attn` actually enters the M4 #267
/// indexer block (commit 783f7b4d). Without this fixture every existing
/// composed test uses ratio==1 → the indexer block is dormant.
fn synthetic_ratio4_composed_with_indexer(vocab: usize) -> ComposedModelWeights {
    use ds4_engine::attn_dispatch::{
        DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
    };
    let mut params = synthetic_params();
    params.compress_ratio = 4;
    let d = params.d_embd as usize;

    let mut attn = synthetic_attn_layer(params);

    // Main compressor weights: shape `[in_dim, coff*head_dim]` for kv/gate,
    // `[coff*head_dim, ratio]` for ape, `[head_dim]` for norm. coff=2 on
    // ratio==4. `head_dim=4` from synthetic_params.
    let ratio = params.compress_ratio as usize;
    let head_dim = params.head_dim as usize;
    let coff = 2;
    let width = coff * head_dim;
    let in_dim = d;
    attn.attn_compressor_kv = vec![0.1f32; in_dim * width];
    attn.attn_compressor_gate = vec![0.05f32; in_dim * width];
    attn.attn_compressor_ape = vec![0.0f32; width * ratio];
    attn.attn_compressor_norm = vec![1.0f32; head_dim];

    // Indexer compressor weights: same shape rules but with
    // head_dim = DS4_N_INDEXER_HEAD_DIM = 128.
    let idx_head_dim = DS4_N_INDEXER_HEAD_DIM as usize;
    let idx_width = coff * idx_head_dim;
    attn.indexer_compressor_kv = vec![0.01f32; in_dim * idx_width];
    attn.indexer_compressor_gate = vec![0.005f32; in_dim * idx_width];
    attn.indexer_compressor_ape = vec![0.0f32; idx_width * ratio];
    attn.indexer_compressor_norm = vec![1.0f32; idx_head_dim];

    // Indexer projection weights (M4 #267):
    //   w_q_b: `[n_lora_q, DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM]`
    //   w_proj: `[d_embd, DS4_N_INDEXER_HEAD]`
    let idx_n_head = DS4_N_INDEXER_HEAD as usize;
    let n_lora_q = params.n_lora_q as usize;
    attn.indexer_attn_q_b = vec![0.001f32; n_lora_q * idx_n_head * idx_head_dim];
    attn.indexer_proj = vec![0.002f32; d * idx_n_head];

    let layers = vec![ComposedLayerWeights {
        attn,
        moe: synthetic_moe_layer(d),
    }];
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

/// M4 #267: prove the indexer wire-up in `decode_step_with_attn` (commit
/// 783f7b4d) is reachable. Drives 4 decode steps over a ratio==4 1-layer
/// model with non-empty indexer weights; asserts:
/// - `state.index_state_kv[0]` and `state.index_state_score[0]` are
///   non-empty (allocated only on ratio==4 layers per AttnStepState::new);
/// - `state.n_index_comp[0]` reaches 1 by `state.pos == 4` (ratio==4 emits
///   one compressed row per 4 input positions);
/// - `state.index_comp_kv_ring[0].len() == DS4_N_INDEXER_HEAD_DIM` after
///   the emit (one row of 128 floats);
/// - the decode step produces finite logits with no panic.
#[test]
fn ratio4_indexer_wire_up_is_reachable_and_emits_row() {
    use ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM;

    let model = synthetic_ratio4_composed_with_indexer(16);
    let cfg = DecodeConfig::default();
    let raw_cap = 8u32;
    let mut state = AttnStepState::new(&model, raw_cap);

    // AttnStepState::new must allocate the indexer state on ratio==4.
    // If decode_step.rs:486-510 regresses (e.g. drops the ratio==4 branch)
    // these will be empty and the indexer block will silently skip.
    assert!(
        !state.index_state_kv[0].is_empty(),
        "ratio==4 layer must allocate index_state_kv"
    );
    assert!(
        !state.index_state_score[0].is_empty(),
        "ratio==4 layer must allocate index_state_score"
    );

    let x: Vec<f32> = (0..model.d_model)
        .map(|i| (i as f32) * 0.0625 + 0.1)
        .collect();

    // Drive 4 decode steps. Antirez `compressor_decode_one` (ds4.c:6208)
    // emits a pooled row at positions where `(pos+1) % ratio == 0`, i.e.
    // pos=3 for ratio=4. After step 4 (pos counter advanced from 0→4),
    // exactly one row should have landed in the indexer ring.
    for _ in 0..4u32 {
        let logits = decode_step_with_attn(
            &CpuDispatcher,
            &CpuAttentionDispatcher,
            x.clone(),
            &model,
            &mut state,
            &cfg,
            raw_cap,
        )
        .expect("ratio==4 decode step");
        assert_eq!(logits.len(), model.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit[{i}]={l} non-finite under ratio==4");
        }
    }

    assert_eq!(state.pos, 4, "4 decode steps must advance pos to 4");
    assert_eq!(
        state.n_index_comp[0], 1,
        "ratio==4 must emit exactly 1 indexer-compressed row by pos=4"
    );
    assert_eq!(
        state.index_comp_kv_ring[0].len(),
        DS4_N_INDEXER_HEAD_DIM as usize,
        "emitted indexer row must be DS4_N_INDEXER_HEAD_DIM floats wide"
    );
}

/// M4 #267 RED-companion: confirm the indexer block in decode_step_with_attn
/// (the `p.compress_ratio == 4 && !indexer_compressor_kv.is_empty() &&
/// !indexer_attn_q_b.is_empty()` gate) actually gates. Same ratio==4 fixture
/// as the previous test, but with the indexer projection weights cleared.
/// If a future refactor accidentally drops the `!indexer_attn_q_b.is_empty()`
/// arm, the indexer block would run on a 0-length w_q_b slice and either
/// panic or silently produce a spurious row — both regressions this test
/// would catch.
///
/// Pair with `ratio4_indexer_wire_up_is_reachable_and_emits_row`: together
/// they bracket the gate (weights present → n_index_comp advances to 1;
/// weights absent → stays at 0). This is the RED-verification mandated by
/// `feedback_aggressive_effective_tests.md` — without it the positive test
/// alone might pass against an always-on impl.
#[test]
fn ratio4_indexer_block_skipped_when_indexer_weights_empty() {
    let mut model = synthetic_ratio4_composed_with_indexer(16);
    // Clear ONLY the indexer projection weights. Main compressor + indexer
    // compressor stay populated, so the indexer-compressor side allocation
    // (state.index_state_kv) still exists; only the indexer call itself is
    // gated off.
    model.layers[0].attn.indexer_attn_q_b.clear();
    model.layers[0].attn.indexer_proj.clear();

    let cfg = DecodeConfig::default();
    let raw_cap = 8u32;
    let mut state = AttnStepState::new(&model, raw_cap);
    let x: Vec<f32> = (0..model.d_model)
        .map(|i| (i as f32) * 0.0625 + 0.1)
        .collect();

    for _ in 0..4u32 {
        decode_step_with_attn(
            &CpuDispatcher,
            &CpuAttentionDispatcher,
            x.clone(),
            &model,
            &mut state,
            &cfg,
            raw_cap,
        )
        .expect("ratio==4 step with cleared indexer weights");
    }

    // The indexer-compressor block runs only inside the same gated `if` as
    // indexer_allowed_decode_one, so clearing indexer_attn_q_b skips BOTH:
    // n_index_comp must stay at 0 and the indexer ring stays empty.
    assert_eq!(
        state.n_index_comp[0], 0,
        "indexer must skip emission when indexer weights are empty"
    );
    assert!(
        state.index_comp_kv_ring[0].is_empty(),
        "indexer ring must stay empty when gated off"
    );
}

#[test]
fn composed_step_produces_finite_logits() {
    let model = synthetic_composed(2, 16);
    let cfg = DecodeConfig::default();
    let raw_cap = 8u32;
    let mut state = AttnStepState::new(&model, raw_cap);
    let x = vec![0.1f32; model.d_model];

    let logits = decode_step_with_attn(
        &CpuDispatcher,
        &CpuAttentionDispatcher,
        x,
        &model,
        &mut state,
        &cfg,
        raw_cap,
    )
    .expect("step");
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{i}] = {l} is not finite");
    }
}
