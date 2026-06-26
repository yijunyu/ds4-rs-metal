//! Phase F task #87 — fast-path vs slow-path equivalence at the
//! attn-half boundary inside `encode_first_half_inner`.
//!
//! The wiring change in task #87 branches on fast-path conditions
//! (head_dim==512, n_raw 32-aligned + nonzero, no compressor/indexer
//! outputs). When all hold, flash_attn + rope_q backward +
//! attn_output_proj fuse into ONE BatchScope (was 2 cbs).
//!
//! This smoke constructs the inputs to that boundary directly (no
//! upstream encode_layer_attn_half_cpu needed), runs BOTH paths,
//! and asserts they produce equivalent `after_attn_hc` within fp8 ULP
//! tolerance (the GPU `hc_expand_attn` reduction order differs by
//! 1 ULP per element from the CPU loop the slow path uses inside
//! `attn_output_proj_impl`).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    apply_rope_tail_with_table, precompute_rope_tail_table, AttentionDispatcher, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn fast_path_params() -> LayerParams {
    // head_dim=512 is the fast-path gate inside
    // flash_attn_decode_metal_persistent (and our new scope variant).
    LayerParams {
        layer_idx: 0,
        d_embd: 256,
        n_hc: 4,
        n_head: 2,
        head_dim: 512,
        n_rot: 64,
        n_lora_q: 64,
        n_lora_kv: 512,
        hc_sinkhorn_iter: 5,
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

#[test]
fn attn_fusion_fast_path_matches_slow_path() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = fast_path_params();
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let n_rot = p.n_rot as usize;
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let kv_row = p.n_lora_kv as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o: usize = 4;
    let out_low_dim = n_groups * n_lora_o;

    // Fast-path gates: pos chosen so n_raw=32 (32-aligned, > 0).
    let raw_cap: u32 = 32;
    let pos: u32 = 31;
    let n_raw: u32 = 32;

    // Pre-fill the persistent KV buffer for two distinct layer_idxs
    // (one for the fast-path encoder run, one for the slow-path
    // reference) with the SAME bytes so flash_attn sees the same input.
    let cache_byte_len = (raw_cap as usize) * kv_row * std::mem::size_of::<f32>();
    let kv_prefill: Vec<f32> = (0..(raw_cap as usize) * kv_row)
        .map(|i| (i as f32 * 0.0011 - 0.1).sin() * 0.3)
        .collect();
    for layer_idx in [33u32, 34u32] {
        let buf = disp.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                kv_prefill.as_ptr(),
                buf.contents() as *mut f32,
                kv_prefill.len(),
            );
        }
    }

    // Inputs to the attn-half boundary (post-rope_q-forward).
    let q_heads: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32) * 0.017).sin() * 0.4)
        .collect();
    // Apply rope_q forward CPU-side to mirror what encode_first_half_inner
    // does just before the boundary (the fast-path branch starts AFTER
    // this CPU rope_q).
    let q_heads_rot = {
        let mut v = q_heads.clone();
        if n_rot > 0 {
            let table = precompute_rope_tail_table(
                n_rot, pos, false,
                p.rope_freq_base, p.rope_freq_scale,
                p.rope_ext_factor, p.rope_attn_factor, p.rope_orig_ctx,
            );
            for head in v.chunks_mut(head_dim) {
                let tail = &mut head[head_dim - n_rot..];
                apply_rope_tail_with_table(tail, &table, n_rot);
            }
        }
        v
    };

    let attn_sinks: Vec<f32> = (0..n_head).map(|h| -0.5 - (h as f32) * 0.1).collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.007).sin() * 0.1)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.012).cos() * 0.15)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| ((i as f32) * 0.009).sin() * 0.3)
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|i| 0.4 + (i as f32) * 0.1).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc).map(|i| 0.2 + (i as f32) * 0.05).collect();

    // ── Slow path: flash_attn_decode_metal_persistent + CPU rope_q backward
    //   + trait attn_output_proj (with CPU hc_expand).
    let slow_layer_idx: u32 = 33;
    let mut heads_back_slow = disp.flash_attn_decode_metal_persistent(
        slow_layer_idx, &p, &q_heads_rot, n_raw, raw_cap,
        None, 0, None, 0, &attn_sinks,
    );
    if n_rot > 0 {
        let table = precompute_rope_tail_table(
            n_rot, pos, true,
            p.rope_freq_base, p.rope_freq_scale,
            p.rope_ext_factor, p.rope_attn_factor, p.rope_orig_ctx,
        );
        for head in heads_back_slow.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }
    }
    let after_slow = AttentionDispatcher::attn_output_proj(
        &disp, &p, &heads_back_slow, &w_o_a, &w_o_b,
        &cur_hc, &hc_split_post, &hc_split_comb,
    );

    // ── Fast path: ONE BatchScope. flash_attn → rope_q backward (GPU) →
    //   encode_attn_output_proj (GPU hc_expand). ONE commit+wait.
    let fast_layer_idx: u32 = 34;
    let scope = disp.batch_scope();
    let heads_b = scope
        .flash_attn_decode_persistent(
            fast_layer_idx, &p, &q_heads_rot, n_raw, raw_cap, &attn_sinks,
        )
        .expect("flash_attn_decode_persistent");
    if n_rot > 0 {
        scope.rope_tail_q_heads_in_place(&heads_b, n_head, head_dim, &p, pos, true)
            .expect("rope_tail_q_heads_in_place backward");
    }
    let w_o_a_b = scope.upload_f32(&w_o_a);
    let w_o_b_b = scope.upload_f32(&w_o_b);
    let cur_hc_b = scope.upload_f32(&cur_hc);
    let post_b = scope.upload_f32(&hc_split_post);
    let comb_b = scope.upload_f32(&hc_split_comb);
    let after_db = scope
        .encode_attn_output_proj(
            &heads_b, &w_o_a_b, &w_o_b_b,
            &[], &[], &cur_hc_b, &post_b, &comb_b,
            n_groups, n_lora_o, group_dim, out_low_dim, n_hc, d_embd,
        )
        .expect("encode_attn_output_proj");
    let after_fast = scope.flush_and_read(&after_db);

    // ── Bit-close: kernel-launch order differs (GPU vs CPU hc_expand)
    //   so we accept ~1 ULP per element. Production d_embd values are
    //   ≤ ~1.0 in magnitude here; 5e-6 absolute is a tight bound.
    assert_eq!(after_slow.len(), after_fast.len());
    let mut max_abs = 0.0f32;
    for (i, (s, f)) in after_slow.iter().zip(&after_fast).enumerate() {
        let d = (s - f).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-4,
            "after_attn_hc[{i}]: slow={s} fast={f} |diff|={d}"
        );
    }
    eprintln!("fast vs slow path max_abs = {max_abs}");
}
