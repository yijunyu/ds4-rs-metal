//! Phase F task #86 — `BatchScope::encode_attn_output_proj` smoke.
//!
//! Validates the composed scope op (`encode_attn_output_matmuls` +
//! `hc_expand_attn`) against the existing `MetalDispatcher::attn_output_proj`
//! trait method which packs the same two stages into its own
//! command buffer. Same kernels in the same order — bit-identity
//! is the right bar.
//!
//! Once this lands + flash_attn_decode_persistent (also scope), the
//! per-layer attn-half boundary in encode_first_half_inner can fuse
//! into a single BatchScope:
//!   encode_layer_attn_half → rope_tail_q_heads_in_place →
//!   flash_attn_decode_persistent → rope_tail_q_heads_in_place (bwd) →
//!   encode_attn_output_proj
//! → one cb per layer's attn-half (vs 3-4 today).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{AttentionDispatcher, LayerParams};
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 128,
        n_hc: 4,
        n_head: 2,
        head_dim: 64,
        n_rot: 16,
        n_lora_q: 64,
        n_lora_kv: 64,
        hc_sinkhorn_iter: 0,
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
fn encode_attn_output_proj_matches_trait() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o: usize = 4;
    let out_low_dim = n_groups * n_lora_o;

    // Deterministic inputs (heads, w_o_a, w_o_b, cur_hc, hc_split sections).
    let heads: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32) * 0.017).sin() * 0.4)
        .collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.011).cos() * 0.15)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.013).sin() * 0.12)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| ((i as f32) * 0.009).cos() * 0.25)
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|i| 0.5 + (i as f32) * 0.125).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| 0.25 + (i as f32 * 0.05) - ((i % 3) as f32) * 0.1)
        .collect();

    // ── Reference: trait `attn_output_proj` (own cb).
    let ref_out = AttentionDispatcher::attn_output_proj(
        &disp, &p, &heads, &w_o_a, &w_o_b, &cur_hc, &hc_split_post, &hc_split_comb,
    );

    // ── Scope: encode_attn_output_proj in a shared BatchScope.
    let scope = disp.batch_scope();
    let heads_b = scope.upload_f32(&heads);
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
    let scope_out = scope.flush_and_read(&after_db);

    // Bit-CLOSE (not bit-identical): the trait `attn_output_proj_impl`
    // performs the hc_expand step on CPU (Rust nested loop) while the
    // scope op uses the GPU `ds4_dsv4_hc_expand` kernel. The math is
    // the same but fp32 accumulation order differs by 1 ULP at d_embd
    // = 128 / n_hc = 4 scale. The matvec stages share the same kernel
    // so they're bit-identical; only the final hc_expand step drifts.
    assert_eq!(ref_out.len(), scope_out.len());
    let mut max_abs = 0.0f32;
    for (i, (a, b)) in ref_out.iter().zip(&scope_out).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 5e-6,
            "after_attn_hc[{i}]: ref={a} scope={b} |diff|={d}"
        );
    }
    eprintln!("attn_output_proj scope vs trait max_abs={max_abs}");
}

#[test]
fn encode_attn_output_proj_chains_in_a_shared_scope() {
    // Validates the composed scope op can be followed by additional
    // encoded work in the SAME scope without intermediate commit.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o: usize = 4;
    let out_low_dim = n_groups * n_lora_o;

    let heads: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32) * 0.019).sin() * 0.3)
        .collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.007).cos() * 0.1)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.012).sin() * 0.13)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| ((i as f32) * 0.005).cos() * 0.2)
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|i| 0.4 + (i as f32) * 0.1).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| 0.2 + (i as f32 * 0.03))
        .collect();

    let scope = disp.batch_scope();
    let heads_b = scope.upload_f32(&heads);
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

    // Follow-up op in the same scope: rms_norm of after_attn_hc.
    let gamma = vec![1.0f32; n_hc * d_embd];
    let gamma_b = scope.upload_f32(&gamma);
    let normed_db = scope
        .rms_norm_mul(&after_db, &gamma_b, p.rms_eps)
        .expect("rms_norm_mul follows attn_output_proj");

    let outs = scope.flush_and_read_multi(&[&after_db, &normed_db]);
    assert_eq!(outs[0].len(), n_hc * d_embd);
    assert_eq!(outs[1].len(), n_hc * d_embd);
    for (i, &v) in outs[0].iter().enumerate() {
        assert!(v.is_finite(), "after_attn_hc[{i}] not finite: {v}");
    }
    for (i, &v) in outs[1].iter().enumerate() {
        assert!(v.is_finite(), "normed[{i}] not finite: {v}");
    }
}
