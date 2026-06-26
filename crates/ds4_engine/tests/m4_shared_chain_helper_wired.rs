//! M4 #330o Phase C.3b.3 helper-wired equivalence gate.
//!
//! After C.3b.3 `decode_attn_ffn_post_with` routes the shared-expert FFN body
//! (gate matvec + up matvec + SwiGLU + optional q8_0 + down matvec) through
//! `KernelDispatcher::shared_chain_batched` when `DS4_SILU_FIDELITY=0`
//! (the default). Under `DS4_SILU_FIDELITY=1` the helper falls back to the
//! legacy sequential `A::shared_expert` + `A::shared_down_hc_expand_add`
//! trait calls.
//!
//! These tests run `decode_attn_ffn_post_with` directly and compare it
//! against a manual hand-stamped composition of the same primitives — once
//! per branch (silu-fid 0 and 1), each crossed with `DS4_Q8_0_ACT={0,1}`.
//! That locks each branch against future drift INDEPENDENTLY of the other,
//! since the two modes are NOT bit-equal to each other (they intentionally
//! use different silu identities — that's the whole point of the gate).
//!
//! The Metal-side override of `shared_chain_batched` is exercised by the
//! macOS-only `shared_chain_batched_smoke.rs` suite in `ds4_metal/tests/`.

use ds4_engine::attn_dispatch::{
    decode_attn_ffn_post_with, hc_expand_add_only, AttentionDispatcher, AttnLayerWeights,
    AttnPrefixOut, CpuAttentionDispatcher, FfnPreOut, LayerParams,
};
use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use std::sync::Mutex;

// Env vars are process-global; serialise across tests in this file so
// toggles don't leak between concurrent runs.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    prev_silu: Option<String>,
    prev_q80: Option<String>,
}
impl EnvGuard {
    fn set(silu_fidelity: bool, q8_0_act: bool) -> Self {
        let prev_silu = std::env::var("DS4_SILU_FIDELITY").ok();
        let prev_q80 = std::env::var("DS4_Q8_0_ACT").ok();
        if silu_fidelity {
            std::env::set_var("DS4_SILU_FIDELITY", "1");
        } else {
            std::env::remove_var("DS4_SILU_FIDELITY");
        }
        if q8_0_act {
            std::env::set_var("DS4_Q8_0_ACT", "1");
        } else {
            std::env::remove_var("DS4_Q8_0_ACT");
        }
        Self { prev_silu, prev_q80 }
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.prev_silu {
            Some(v) => std::env::set_var("DS4_SILU_FIDELITY", v),
            None => std::env::remove_var("DS4_SILU_FIDELITY"),
        }
        match &self.prev_q80 {
            Some(v) => std::env::set_var("DS4_Q8_0_ACT", v),
            None => std::env::remove_var("DS4_Q8_0_ACT"),
        }
    }
}

fn synthetic_params(d_embd: u32, n_hc: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd,
        n_hc,
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

fn make_layer_with_shared(params: LayerParams, shared_dim: u32) -> AttnLayerWeights {
    let d_embd = params.d_embd as usize;
    let n_hc = params.n_hc as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let q_dim = params.q_dim();
    let n_groups = params.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = 3;
    let out_low_dim = n_groups * n_lora_o;
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
        w_shared_gate: (0..sd * d_embd).map(|i| 0.1 + (i as f32) * 0.013).collect(),
        w_shared_up: (0..sd * d_embd).map(|i| 0.05 + (i as f32) * 0.017).collect(),
        w_shared_down: (0..d_embd * sd).map(|i| 0.07 + (i as f32) * 0.011).collect(),
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

fn make_ffn_pre(d_embd: usize, n_hc: usize, seed: f32) -> FfnPreOut {
    // hc_split_ffn layout used by decode_attn_ffn_post_with:
    //   [pre n_hc] [post n_hc] [comb n_hc*n_hc]
    let total = 2 * n_hc + n_hc * n_hc;
    let hc_split_ffn: Vec<f32> = (0..total)
        .map(|i| 0.1 + (i as f32 + seed) * 0.07)
        .collect();
    let ffn_normed: Vec<f32> =
        (0..d_embd).map(|i| (i as f32 * 0.07 + seed * 0.13).sin()).collect();
    FfnPreOut {
        ffn_normed,
        hc_split_ffn,
    }
}

fn make_prefix(d_embd: usize, n_hc: usize, seed: f32) -> AttnPrefixOut {
    let total_attn = 2 * n_hc + n_hc * n_hc;
    AttnPrefixOut {
        normed: vec![0.0; d_embd],
        after_attn_hc: (0..n_hc * d_embd)
            .map(|i| (i as f32 * 0.029 + seed).cos())
            .collect(),
        hc_split_attn: (0..total_attn).map(|i| 0.1 + (i as f32) * 0.05).collect(),
    }
}

fn check_branch_bit_identity(silu_fidelity: bool, q8_0_act: bool) {
    let _guard = ENV_LOCK.lock().unwrap();
    // QK8_0=32 means a useful want_q80 sd must be div by 32 if exercising the
    // Metal primitive — but on CPU the `q8_0_round_trip` handles arbitrary
    // lengths by zero-padding the last block. We pick sd=32 to be a clean
    // multiple, d_embd=32 likewise.
    let p = synthetic_params(32, 2);
    let layer = make_layer_with_shared(p, 32);
    let prefix = make_prefix(32, 2, 0.3);
    let ffn_pre = make_ffn_pre(32, 2, 0.7);
    let routed_out: Vec<f32> = (0..32).map(|i| (i as f32 * 0.041).sin() * 0.2).collect();
    let pos = 5u32;
    let n_hc = p.n_hc as usize;
    let hc_split_post = &ffn_pre.hc_split_ffn[n_hc..2 * n_hc];
    let hc_split_comb = &ffn_pre.hc_split_ffn[2 * n_hc..2 * n_hc + n_hc * n_hc];

    let _eg = EnvGuard::set(silu_fidelity, q8_0_act);

    let got = decode_attn_ffn_post_with(
        &CpuDispatcher,
        &CpuAttentionDispatcher,
        &layer,
        &prefix,
        &ffn_pre,
        &routed_out,
        pos,
        None,
    );

    // Hand-stamped reference: replay the exact branch logic with `silu_fidelity`.
    let want = if silu_fidelity {
        let shared_mid = CpuAttentionDispatcher.shared_expert(
            &layer.params,
            &ffn_pre.ffn_normed,
            &layer.w_shared_gate,
            &layer.w_shared_up,
            layer.shared_dim,
            layer.shared_clamp,
        );
        CpuAttentionDispatcher.shared_down_hc_expand_add(
            &layer.params,
            &shared_mid,
            &layer.w_shared_down,
            &routed_out,
            &prefix.after_attn_hc,
            hc_split_post,
            hc_split_comb,
        )
    } else {
        let ffn_in: Vec<f32> = if q8_0_act {
            ds4_engine::forward::q8_0_round_trip(&ffn_pre.ffn_normed)
        } else {
            ffn_pre.ffn_normed.clone()
        };
        let shared_out = CpuDispatcher.shared_chain_batched(
            &ffn_in,
            &layer.w_shared_gate,
            &layer.w_shared_up,
            &layer.w_shared_down,
            layer.shared_dim,
            q8_0_act,
        );
        hc_expand_add_only(
            &layer.params,
            &shared_out,
            &routed_out,
            &prefix.after_attn_hc,
            hc_split_post,
            hc_split_comb,
        )
    };

    assert_eq!(got.len(), want.len(), "output length mismatch");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "silu_fid={silu_fidelity} q80={q8_0_act} idx={i}: got={g} want={w}"
        );
    }
}

#[test]
fn helper_branch_default_no_q80_matches_hand_stamped_reference() {
    check_branch_bit_identity(false, false);
}

#[test]
fn helper_branch_default_with_q80_matches_hand_stamped_reference() {
    check_branch_bit_identity(false, true);
}

#[test]
fn helper_branch_silu_fidelity_no_q80_matches_legacy_trait_path() {
    check_branch_bit_identity(true, false);
}

#[test]
fn helper_branch_silu_fidelity_with_q80_matches_legacy_trait_path() {
    check_branch_bit_identity(true, true);
}
