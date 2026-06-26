//! M4 #225 Phase A — attention-half driver harness.
//!
//! Stitches the AttentionDispatcher methods through one decode position via
//! `decode_attn_layer_with`. Two independent `CpuAttentionDispatcher`
//! instances must produce bit-equal outputs on the same inputs — proves the
//! driver shape is deterministic before the Metal dispatcher is swapped in.

use ds4_engine::attn_dispatch::{
    decode_attn_layer_with, AttnLayerWeights, AttnStepInputs, CpuAttentionDispatcher, KvCacheView,
    LayerParams,
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

#[test]
fn cpu_x_cpu_attn_layer_is_bit_equal() {
    let params = tiny_params();
    let layer = tiny_layer(params);
    let kv_row = params.n_lora_kv as usize;
    let cap: u32 = 8;

    let cur_hc: Vec<f32> = (0..params.hc_dim())
        .map(|i| (i as f32) * 0.125 - 0.5)
        .collect();
    let kv_raw_row: Vec<f32> = (0..params.n_lora_kv).map(|i| (i as f32) * 0.0625).collect();
    let q_heads: Vec<f32> = (0..params.q_dim()).map(|i| (i as f32) * 0.03125).collect();
    let routed_out: Vec<f32> = vec![0.0; params.d_embd as usize];

    let mut storage_a = vec![0.0f32; kv_row * cap as usize];
    let mut storage_b = vec![0.0f32; kv_row * cap as usize];

    let disp_a = CpuAttentionDispatcher;
    let disp_b = CpuAttentionDispatcher;

    let out_a = decode_attn_layer_with(
        &disp_a,
        &layer,
        AttnStepInputs {
            cur_hc: &cur_hc,
            kv_raw_row: &kv_raw_row,
            q_heads: &q_heads,
            pos: 0,
            kv_view: KvCacheView {
                raw: &mut storage_a,
                raw_cap: cap,
                pos: 0,
            },
            n_raw: 1,
            raw_start: 0,
            routed_out: &routed_out,
            kv_comp_rows: None,
            n_comp: 0,
            comp_selected: None,
            n_selected: 0,
        },
    );

    let out_b = decode_attn_layer_with(
        &disp_b,
        &layer,
        AttnStepInputs {
            cur_hc: &cur_hc,
            kv_raw_row: &kv_raw_row,
            q_heads: &q_heads,
            pos: 0,
            kv_view: KvCacheView {
                raw: &mut storage_b,
                raw_cap: cap,
                pos: 0,
            },
            n_raw: 1,
            raw_start: 0,
            routed_out: &routed_out,
            kv_comp_rows: None,
            n_comp: 0,
            comp_selected: None,
            n_selected: 0,
        },
    );

    assert_eq!(out_a.after_ffn_hc, out_b.after_ffn_hc);
    assert_eq!(out_a.normed, out_b.normed);
    assert_eq!(storage_a, storage_b);
}

#[test]
fn attn_layer_writes_kv_row_at_pos() {
    let params = tiny_params();
    let layer = tiny_layer(params);
    let kv_row = params.n_lora_kv as usize;
    let cap: u32 = 4;

    let cur_hc = vec![0.5f32; params.hc_dim()];
    let kv_raw_row = vec![2.0f32; kv_row];
    let q_heads = vec![0.25f32; params.q_dim()];
    let routed_out = vec![0.0f32; params.d_embd as usize];

    let mut storage = vec![0.0f32; kv_row * cap as usize];
    let disp = CpuAttentionDispatcher;

    let _ = decode_attn_layer_with(
        &disp,
        &layer,
        AttnStepInputs {
            cur_hc: &cur_hc,
            kv_raw_row: &kv_raw_row,
            q_heads: &q_heads,
            pos: 0,
            kv_view: KvCacheView {
                raw: &mut storage,
                raw_cap: cap,
                pos: 0,
            },
            n_raw: 1,
            raw_start: 0,
            routed_out: &routed_out,
            kv_comp_rows: None,
            n_comp: 0,
            comp_selected: None,
            n_selected: 0,
        },
    );

    // First KV slot should have been written (non-zero entries somewhere).
    let row0_nonzero = storage[..kv_row].iter().any(|&v| v != 0.0);
    assert!(
        row0_nonzero,
        "first KV slot should be populated after one decode step"
    );
    // Second slot should still be zero.
    let row1_zero = storage[kv_row..2 * kv_row].iter().all(|&v| v == 0.0);
    assert!(row1_zero, "second KV slot should be untouched");
}

#[test]
fn attn_layer_output_shapes() {
    let params = tiny_params();
    let layer = tiny_layer(params);
    let kv_row = params.n_lora_kv as usize;
    let cap: u32 = 4;

    let cur_hc = vec![0.25f32; params.hc_dim()];
    let kv_raw_row = vec![0.5f32; kv_row];
    let q_heads = vec![0.25f32; params.q_dim()];
    let routed_out = vec![0.0f32; params.d_embd as usize];
    let mut storage = vec![0.0f32; kv_row * cap as usize];

    let out = decode_attn_layer_with(
        &CpuAttentionDispatcher,
        &layer,
        AttnStepInputs {
            cur_hc: &cur_hc,
            kv_raw_row: &kv_raw_row,
            q_heads: &q_heads,
            pos: 0,
            kv_view: KvCacheView {
                raw: &mut storage,
                raw_cap: cap,
                pos: 0,
            },
            n_raw: 1,
            raw_start: 0,
            routed_out: &routed_out,
            kv_comp_rows: None,
            n_comp: 0,
            comp_selected: None,
            n_selected: 0,
        },
    );
    assert_eq!(out.after_ffn_hc.len(), params.hc_dim());
    assert_eq!(out.normed.len(), params.d_embd as usize);
}
