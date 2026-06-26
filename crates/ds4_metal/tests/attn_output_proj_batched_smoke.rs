//! M4 #330o — Phase C.3f attn_output_proj_impl single-cmdbuf smoke tests.
//!
//! `attn_output_proj_impl` now packs (`n_groups` grouped matvecs) +
//! (one dense matvec) into ONE `MTLCommandBuffer`. These tests assert
//! that the packed path produces BIT-IDENTICAL output to running the
//! same matvecs sequentially through `matvec_f32_impl`, since both
//! paths use the same `ds4_kernel_mul_mv_f32_f32_4` pipeline with the
//! same FC constants. Bit-identity is the right gate here: a tolerance
//! test would mask buffer-offset bugs that happen to land within fp
//! noise.
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{AttentionDispatcher, LayerParams};
use ds4_metal::MetalDispatcher;

fn params(n_head: u32, head_dim: u32, d_embd: u32, n_out_group: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd,
        n_hc: 1,
        n_head,
        head_dim,
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
        n_out_group,
    }
}

/// Run `attn_output_proj` two ways:
///   A) the packed (C.3f) path via `dispatcher.attn_output_proj(...)`,
///   B) a hand-coded sequential path that does the n_groups + dense
///      matvecs through `matvec_f32` and applies the same HC fold.
/// Assert byte-equality.
fn run_both_and_assert_bit_identical(
    dispatcher: &MetalDispatcher,
    p: &LayerParams,
    heads: &[f32],
    w_o_a: &[f32],
    w_o_b: &[f32],
    cur_hc: &[f32],
    hc_split_post: &[f32],
    hc_split_comb: &[f32],
) {
    use ds4_engine::dispatch::KernelDispatcher;

    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;
    let q_dim = heads.len();
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = w_o_a.len() / (n_groups * group_dim);
    let out_low_dim = n_groups * n_lora_o;

    // Path A: packed.
    let packed = dispatcher.attn_output_proj(
        p,
        heads,
        w_o_a,
        w_o_b,
        cur_hc,
        hc_split_post,
        hc_split_comb,
    );

    // Path B: sequential matvec_f32 reproduction.
    let mut attn_low = vec![0.0f32; out_low_dim];
    for g in 0..n_groups {
        let mat = &w_o_a[(g * n_lora_o) * group_dim..(g + 1) * n_lora_o * group_dim];
        let vec_g = &heads[g * group_dim..(g + 1) * group_dim];
        let out_g = dispatcher.matvec_f32(mat, vec_g, n_lora_o);
        attn_low[g * n_lora_o..(g + 1) * n_lora_o].copy_from_slice(&out_g);
    }
    let attn_out = dispatcher.matvec_f32(w_o_b, &attn_low, d_embd);
    let mut hand = vec![0.0f32; n_hc * d_embd];
    for dst in 0..n_hc {
        let base = dst * d_embd;
        let w_post = hc_split_post[dst];
        for e in 0..d_embd {
            let mut acc = w_post * attn_out[e];
            for src in 0..n_hc {
                acc += hc_split_comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
            }
            hand[base + e] = acc;
        }
    }

    assert_eq!(packed.len(), hand.len(), "output len mismatch");
    for (i, (p_v, h_v)) in packed.iter().zip(&hand).enumerate() {
        assert_eq!(
            p_v.to_bits(),
            h_v.to_bits(),
            "attn_output_proj diverged at i={i}: packed={p_v} hand-sequential={h_v}"
        );
    }
}

#[test]
fn attn_output_proj_packed_matches_sequential_small() {
    // n_head=4 head_dim=8 → q_dim=32; n_groups=2 → group_dim=16.
    // n_lora_o=8 → out_low_dim=16 (mult of 4 for stage-2 float4 path).
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(4, 8, 64, 2);
    let q_dim = (p.n_head * p.head_dim) as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = 8usize;
    let out_low_dim = n_groups * n_lora_o;
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;

    let heads: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.11 - 0.5).sin()).collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.017).cos() * 0.25)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.013 + 0.3).sin() * 0.3)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| (i as f32 * 0.05).cos())
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|h| 0.6 + 0.1 * h as f32).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| ((i as f32) * 0.11 - 0.3).cos() * 0.3)
        .collect();

    run_both_and_assert_bit_identical(
        &dispatcher,
        &p,
        &heads,
        &w_o_a,
        &w_o_b,
        &cur_hc,
        &hc_split_post,
        &hc_split_comb,
    );
}

#[test]
fn attn_output_proj_packed_matches_sequential_ds4_shape() {
    // Scaled-down DS4 V4 Flash output-proj shape:
    //   q_dim = n_head * head_dim = 16 * 16 = 256 (prod: 128*128 = 16384)
    //   n_groups = 8
    //   group_dim = 32 (prod: 2048)
    //   n_lora_o = 16
    //   out_low_dim = 128 (mult of 4 ✓)
    //   d_embd = 128 (mult of 4 ✓)
    // Validates path picks `nxpsg=4`/`nsg=1` for stage 1 (group_dim=32 → 32<128)
    // and `nxpsg=4`/`nsg=1` for stage 2 (out_low_dim=128 → 128%128==0 → nxpsg=8).
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(16, 16, 128, 8);
    let q_dim = (p.n_head * p.head_dim) as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = 16usize;
    let out_low_dim = n_groups * n_lora_o;
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;

    let heads: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32 * 0.029).sin() * 0.6).clamp(-1.5, 1.5))
        .collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.0083).cos() * 0.21)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.0061 - 0.2).sin() * 0.18)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| (i as f32 * 0.043).cos())
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|h| 0.55 + 0.07 * h as f32).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| ((i as f32) * 0.09).cos() * 0.27)
        .collect();

    run_both_and_assert_bit_identical(
        &dispatcher,
        &p,
        &heads,
        &w_o_a,
        &w_o_b,
        &cur_hc,
        &hc_split_post,
        &hc_split_comb,
    );
}

/// Independence canary: swapping `w_o_a` MUST change `attn_out`. Locks
/// against a silent regression where stage-1 dispatches are dropped and
/// stage-2 ends up reading zeros (or stale data from a prior call) from
/// `attn_low_buf`.
#[test]
fn attn_output_proj_packed_depends_on_w_o_a() {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = params(4, 8, 64, 2);
    let q_dim = (p.n_head * p.head_dim) as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o = 8usize;
    let out_low_dim = n_groups * n_lora_o;
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;

    let heads: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.11 - 0.5).sin()).collect();
    let w_o_a_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.017).cos() * 0.25)
        .collect();
    let w_o_a_b: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.029).sin() * 0.31)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.013 + 0.3).sin() * 0.3)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| (i as f32 * 0.05).cos())
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|h| 0.6 + 0.1 * h as f32).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| ((i as f32) * 0.11 - 0.3).cos() * 0.3)
        .collect();

    let out_a = dispatcher.attn_output_proj(
        &p,
        &heads,
        &w_o_a_a,
        &w_o_b,
        &cur_hc,
        &hc_split_post,
        &hc_split_comb,
    );
    let out_b = dispatcher.attn_output_proj(
        &p,
        &heads,
        &w_o_a_b,
        &w_o_b,
        &cur_hc,
        &hc_split_post,
        &hc_split_comb,
    );
    let max_diff: f32 = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "different w_o_a produced bit-identical output — stage-1 likely skipped (max |diff| = {max_diff})"
    );
}
