//! M4 #365 Phase D Slice 1 — Metal-side bit-identity test for
//! `attn_prefix_batched`.
//!
//! Confirms that on macOS the default `attn_prefix_batched` impl (which
//! calls the 6 underlying trait methods in sequence) returns the same
//! outputs as a hand-rolled sequence of the same 6 trait methods. When a
//! Metal override lands that fuses the 6 GPU dispatches, this test
//! becomes the gate that prevents numerical drift.
//!
//! macOS-only — we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    precompute_rope_tail_table, apply_rope_tail_with_table, AttentionDispatcher, HcKind,
    KvCacheView, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn tiny_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 16,
        n_hc: 4,
        n_head: 4,
        head_dim: 16,
        n_rot: 4,
        n_lora_q: 16,
        n_lora_kv: 16,
        hc_sinkhorn_iter: 2,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 2,
    }
}

#[test]
fn metal_attn_prefix_batched_matches_sequential() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = tiny_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let q_dim = p.q_dim();
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    // n_lora_o must satisfy attn_output_matmuls_batched's NR0=2 precondition.
    let n_lora_o = 4;
    let out_low_dim = n_groups * n_lora_o;
    let n_lora_kv = p.n_lora_kv as usize;
    let n_rot = p.n_rot as usize;

    let cur_hc: Vec<f32> = (0..hc_dim).map(|i| ((i as f32) * 0.023).sin() * 0.5).collect();
    let kv_raw_row: Vec<f32> = (0..n_lora_kv).map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();
    let q_heads: Vec<f32> = (0..q_dim).map(|i| ((i as f32) * 0.041).sin() * 0.2).collect();
    let hc_attn_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| ((i as f32) * 0.013).sin() * 0.1)
        .collect();
    let hc_attn_scale = vec![1.0_f32, 0.7, 0.3];
    let hc_attn_base: Vec<f32> = (0..mix_hc).map(|i| (i as f32) * 0.01).collect();
    let hc_norm_gamma: Vec<f32> = (0..d_embd).map(|i| 1.0 + (i as f32) * 0.05).collect();
    let qkv_gamma_kv: Vec<f32> = (0..n_lora_kv).map(|i| 1.0 + (i as f32) * 0.03).collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.017).cos() * 0.1)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.019).sin() * 0.2)
        .collect();
    let attn_sinks = vec![-1e9_f32; p.n_head as usize];

    let cap: u32 = 8;
    let pos: u32 = 3;

    // Sequential reference path (mirrors `decode_attn_prefix_with`).
    let mut storage_seq = vec![0.0f32; n_lora_kv * cap as usize];
    let (after_attn_hc_seq, hc_split_attn_seq, normed_seq) = {
        let (_cur, normed, hc_split_attn) = disp.hc_collapse_norm(
            &p,
            HcKind::Attn,
            &hc_attn_fn,
            &hc_attn_scale,
            &hc_attn_base,
            &cur_hc,
            Some(&hc_norm_gamma),
        );
        let hc_split_post = hc_split_attn[n_hc..2 * n_hc].to_vec();
        let hc_split_comb = hc_split_attn[2 * n_hc..2 * n_hc + n_hc * n_hc].to_vec();

        let mut kv_normed = disp.kv_rms_norm_row(&p, &kv_raw_row, &qkv_gamma_kv);
        disp.rope_tail(&p, &mut kv_normed[n_lora_kv - n_rot..n_lora_kv], pos, false);
        let mut view = KvCacheView {
            raw: &mut storage_seq,
            raw_cap: cap,
            pos,
        };
        disp.kv_fp8_store(&p, &kv_normed, &mut view);

        let mut q_heads_rot = q_heads.clone();
        let head_dim = p.head_dim as usize;
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            false,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in q_heads_rot.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }

        let heads = disp.flash_attn_decode(
            &p,
            &q_heads_rot,
            view.raw,
            4,
            view.raw_cap,
            0,
            None,
            0,
            None,
            0,
            &attn_sinks,
        );

        let mut heads_back = heads;
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            true,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in heads_back.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }

        let after = disp.attn_output_proj(
            &p,
            &heads_back,
            &w_o_a,
            &w_o_b,
            &cur_hc,
            &hc_split_post,
            &hc_split_comb,
        );
        (after, hc_split_attn, normed)
    };

    // Batched path through trait method.
    let mut storage_bat = vec![0.0f32; n_lora_kv * cap as usize];
    let mut view_bat = KvCacheView {
        raw: &mut storage_bat,
        raw_cap: cap,
        pos,
    };
    let out = disp.attn_prefix_batched(
        &p,
        &hc_attn_fn,
        &hc_attn_scale,
        &hc_attn_base,
        &cur_hc,
        &hc_norm_gamma,
        &kv_raw_row,
        &qkv_gamma_kv,
        pos,
        &mut view_bat,
        &q_heads,
        4,
        0,
        None,
        0,
        None,
        0,
        &attn_sinks,
        &w_o_a,
        &w_o_b,
    );

    assert_eq!(
        out.after_attn_hc.len(),
        after_attn_hc_seq.len(),
        "after_attn_hc length"
    );
    for (i, (b, r)) in out
        .after_attn_hc
        .iter()
        .zip(&after_attn_hc_seq)
        .enumerate()
    {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "after_attn_hc bit-diff at i={i}: batched={b} seq={r}"
        );
    }
    assert_eq!(out.hc_split_attn, hc_split_attn_seq, "hc_split_attn diff");
    assert_eq!(out.normed, normed_seq, "normed diff");
    assert_eq!(storage_bat, storage_seq, "kv storage diff");
}

/// M4 #365 Phase D Slice 2 — bit-identity coverage of the indexer-selected
/// path: `kv_comp_rows` + `comp_selected` both non-None. This is what fires
/// on DS4 layers ≥3 (`compress_ratio == 4`); the basic test above covers
/// only the layer-0 (`None`/`None`) case. Adding this widens the gate so a
/// future Metal override of `attn_prefix_batched` can't silently regress
/// the indexer-mixed flash-attention path.
#[test]
fn metal_attn_prefix_batched_matches_sequential_with_indexer() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = tiny_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let q_dim = p.q_dim();
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    // n_lora_o must satisfy attn_output_matmuls_batched's NR0=2 precondition.
    let n_lora_o = 4;
    let out_low_dim = n_groups * n_lora_o;
    let n_lora_kv = p.n_lora_kv as usize;
    let n_rot = p.n_rot as usize;

    let cur_hc: Vec<f32> = (0..hc_dim).map(|i| ((i as f32) * 0.029).sin() * 0.4).collect();
    let kv_raw_row: Vec<f32> = (0..n_lora_kv).map(|i| ((i as f32) * 0.061).cos() * 0.25).collect();
    let q_heads: Vec<f32> = (0..q_dim).map(|i| ((i as f32) * 0.037).sin() * 0.18).collect();
    let hc_attn_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| ((i as f32) * 0.011).sin() * 0.08)
        .collect();
    let hc_attn_scale = vec![1.0_f32, 0.65, 0.35];
    let hc_attn_base: Vec<f32> = (0..mix_hc).map(|i| (i as f32) * 0.012).collect();
    let hc_norm_gamma: Vec<f32> = (0..d_embd).map(|i| 1.0 + (i as f32) * 0.04).collect();
    let qkv_gamma_kv: Vec<f32> = (0..n_lora_kv).map(|i| 1.0 + (i as f32) * 0.025).collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.015).cos() * 0.09)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.021).sin() * 0.18)
        .collect();
    let attn_sinks = vec![-1e9_f32; p.n_head as usize];

    let cap: u32 = 8;
    let pos: u32 = 5;

    // Indexer scratch: 6 compressed rows, top-3 selected.
    let n_comp: u32 = 6;
    let n_selected: u32 = 3;
    let kv_comp_rows: Vec<f32> = (0..(n_comp as usize) * n_lora_kv)
        .map(|i| ((i as f32) * 0.019).sin() * 0.15)
        .collect();
    let comp_selected: Vec<u32> = vec![1, 3, 5];

    // Sequential reference path mirrors `decode_attn_prefix_with`'s 6-call
    // body, but with the indexer-selected flash-attention arm.
    let mut storage_seq = vec![0.0f32; n_lora_kv * cap as usize];
    let (after_attn_hc_seq, hc_split_attn_seq, normed_seq) = {
        let (_cur, normed, hc_split_attn) = disp.hc_collapse_norm(
            &p,
            HcKind::Attn,
            &hc_attn_fn,
            &hc_attn_scale,
            &hc_attn_base,
            &cur_hc,
            Some(&hc_norm_gamma),
        );
        let hc_split_post = hc_split_attn[n_hc..2 * n_hc].to_vec();
        let hc_split_comb = hc_split_attn[2 * n_hc..2 * n_hc + n_hc * n_hc].to_vec();

        let mut kv_normed = disp.kv_rms_norm_row(&p, &kv_raw_row, &qkv_gamma_kv);
        disp.rope_tail(&p, &mut kv_normed[n_lora_kv - n_rot..n_lora_kv], pos, false);
        let mut view = KvCacheView {
            raw: &mut storage_seq,
            raw_cap: cap,
            pos,
        };
        disp.kv_fp8_store(&p, &kv_normed, &mut view);

        let mut q_heads_rot = q_heads.clone();
        let head_dim = p.head_dim as usize;
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            false,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in q_heads_rot.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }

        let heads = disp.flash_attn_decode(
            &p,
            &q_heads_rot,
            view.raw,
            4,
            view.raw_cap,
            0,
            Some(&kv_comp_rows),
            n_comp,
            Some(&comp_selected),
            n_selected,
            &attn_sinks,
        );

        let mut heads_back = heads;
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            true,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in heads_back.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }

        let after = disp.attn_output_proj(
            &p,
            &heads_back,
            &w_o_a,
            &w_o_b,
            &cur_hc,
            &hc_split_post,
            &hc_split_comb,
        );
        (after, hc_split_attn, normed)
    };

    // Batched path with the indexer-selected arm.
    let mut storage_bat = vec![0.0f32; n_lora_kv * cap as usize];
    let mut view_bat = KvCacheView {
        raw: &mut storage_bat,
        raw_cap: cap,
        pos,
    };
    let out = disp.attn_prefix_batched(
        &p,
        &hc_attn_fn,
        &hc_attn_scale,
        &hc_attn_base,
        &cur_hc,
        &hc_norm_gamma,
        &kv_raw_row,
        &qkv_gamma_kv,
        pos,
        &mut view_bat,
        &q_heads,
        4,
        0,
        Some(&kv_comp_rows),
        n_comp,
        Some(&comp_selected),
        n_selected,
        &attn_sinks,
        &w_o_a,
        &w_o_b,
    );

    assert_eq!(
        out.after_attn_hc.len(),
        after_attn_hc_seq.len(),
        "after_attn_hc length (indexer)"
    );
    for (i, (b, r)) in out
        .after_attn_hc
        .iter()
        .zip(&after_attn_hc_seq)
        .enumerate()
    {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "after_attn_hc indexer bit-diff at i={i}: batched={b} seq={r}"
        );
    }
    assert_eq!(out.hc_split_attn, hc_split_attn_seq, "hc_split_attn indexer diff");
    assert_eq!(out.normed, normed_seq, "normed indexer diff");
    assert_eq!(storage_bat, storage_seq, "kv storage indexer diff");
}
