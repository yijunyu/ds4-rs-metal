//! M4 #365 Phase D Slice 1 — bit-identity test for `attn_prefix_batched`.
//!
//! Asserts that routing Span A through the new optional trait method
//! (DS4_ATTN_PREFIX_BATCHED=1) produces bit-equal outputs to the sequential
//! 6-call body. This is the seam Metal will override to fuse the 6 GPU
//! dispatches into one MTLCommandBuffer.

use ds4_engine::attn_dispatch::{
    decode_attn_prefix_with, AttnLayerWeights, AttnStepInputs, CpuAttentionDispatcher,
    KvCacheView, LayerParams,
};

fn tiny_params() -> LayerParams {
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

fn tiny_layer(params: LayerParams) -> AttnLayerWeights {
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
        hc_attn_fn: (0..hc_dim * mix_hc)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1)
            .collect(),
        hc_attn_scale: vec![1.0, 0.7, 0.3],
        hc_attn_base: (0..mix_hc).map(|i| (i as f32) * 0.01).collect(),
        hc_ffn_fn: vec![0.1; hc_dim * mix_hc],
        hc_ffn_scale: vec![1.0; 3],
        hc_ffn_base: vec![0.0; mix_hc],
        hc_attn_fn_f16: Vec::new().into(),
        hc_ffn_fn_f16: Vec::new().into(),
        hc_norm_gamma: (0..d_embd).map(|i| 1.0 + (i as f32) * 0.05).collect(),
        hc_ffn_norm_gamma: vec![1.0; d_embd],
        qkv_gamma_q: vec![1.0; params.n_lora_q as usize],
        qkv_gamma_kv: (0..params.n_lora_kv as usize)
            .map(|i| 1.0 + (i as f32) * 0.03)
            .collect(),
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
        w_o_a: (0..out_low_dim * group_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.1)
            .collect(),
        w_o_b: (0..d_embd * out_low_dim)
            .map(|i| ((i as f32) * 0.019).sin() * 0.2)
            .collect(),
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

fn run_prefix(batched: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let params = tiny_params();
    let layer = tiny_layer(params);
    let kv_row = params.n_lora_kv as usize;
    let cap: u32 = 8;

    let cur_hc: Vec<f32> = (0..params.hc_dim())
        .map(|i| (i as f32) * 0.125 - 0.5)
        .collect();
    let kv_raw_row: Vec<f32> = (0..params.n_lora_kv)
        .map(|i| ((i as f32) * 0.07).sin())
        .collect();
    let q_heads: Vec<f32> = (0..params.q_dim())
        .map(|i| ((i as f32) * 0.05).cos())
        .collect();
    let routed_out: Vec<f32> = vec![0.0; params.d_embd as usize];
    let mut storage = vec![0.0f32; kv_row * cap as usize];

    // Set env var inside the test (synchronised — both branches reset it).
    if batched {
        unsafe { std::env::set_var("DS4_ATTN_PREFIX_BATCHED", "1") };
    } else {
        unsafe { std::env::remove_var("DS4_ATTN_PREFIX_BATCHED") };
    }

    let disp = CpuAttentionDispatcher;
    let out = decode_attn_prefix_with(
        &disp,
        &layer,
        &mut AttnStepInputs {
            cur_hc: &cur_hc,
            kv_raw_row: &kv_raw_row,
            q_heads: &q_heads,
            pos: 3,
            kv_view: KvCacheView {
                raw: &mut storage,
                raw_cap: cap,
                pos: 3,
            },
            n_raw: 4,
            raw_start: 0,
            routed_out: &routed_out,
            kv_comp_rows: None,
            n_comp: 0,
            comp_selected: None,
            n_selected: 0,
        },
    );
    (
        out.after_attn_hc,
        out.hc_split_attn,
        out.normed,
        storage,
    )
}

#[test]
fn attn_prefix_batched_default_matches_sequential() {
    // Run both paths through the same CpuAttentionDispatcher. The default
    // `attn_prefix_batched` impl calls the same 6 trait methods in the same
    // order, so outputs must be bit-equal.
    let seq = run_prefix(false);
    let bat = run_prefix(true);

    assert_eq!(seq.0, bat.0, "after_attn_hc divergence");
    assert_eq!(seq.1, bat.1, "hc_split_attn divergence");
    assert_eq!(seq.2, bat.2, "normed divergence");
    assert_eq!(seq.3, bat.3, "kv storage divergence");

    // Cleanup so other tests aren't affected.
    unsafe { std::env::remove_var("DS4_ATTN_PREFIX_BATCHED") };
}
