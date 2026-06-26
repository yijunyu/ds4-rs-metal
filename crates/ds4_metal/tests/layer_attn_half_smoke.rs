//! Phase E M5.4.1 — `BatchScope::encode_layer_attn_half` smoke test.
//!
//! Composes 4 BatchScope sub-chains into one named building block:
//! hc_collapse_norm + encode_attn_qkv_chain + rms_norm_mul(kv) +
//! rope_tail_in_place(kv tail). Verifies bit-identical output to
//! running the equivalent ops via separate scopes / inherent methods.
//!
//! This is the foundation block of the per-token unified encoder
//! (M5.4): every layer's attention-half goes through this method.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 8,
        head_dim: 64,
        n_rot: 16,
        n_lora_q: 64,
        n_lora_kv: 128,
        hc_sinkhorn_iter: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 1,
    }
}

#[test]
fn encode_layer_attn_half_matches_sequential() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let n_lora_q = p.n_lora_q as usize;
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let kv_row = p.n_lora_kv as usize;
    let pos: u32 = 5;

    let prev_hc: Vec<f32> = (0..hc_dim)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| (i as f32 * 0.0011).sin() * 0.05)
        .collect();
    let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
    let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let hc_norm_gamma: Vec<f32> = (0..d_embd)
        .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05)
        .collect();
    let unit_gamma: Vec<f32> = vec![1.0f32; hc_dim];
    let attn_q_a: Vec<f32> = (0..n_lora_q * d_embd)
        .map(|i| (i as f32 * 0.009).cos() * 0.25)
        .collect();
    let gamma_q: Vec<f32> = (0..n_lora_q)
        .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
        .collect();
    let attn_q_b: Vec<f32> = (0..q_dim * n_lora_q)
        .map(|i| (i as f32 * 0.013).sin() * 0.2)
        .collect();
    let attn_kv: Vec<f32> = (0..kv_row * d_embd)
        .map(|i| (i as f32 * 0.019).cos() * 0.15)
        .collect();
    let qkv_gamma_kv: Vec<f32> = (0..kv_row)
        .map(|i| 1.0 + (i as f32 * 0.021).sin() * 0.04)
        .collect();

    // ── Path A: encode_layer_attn_half — all 4 chains in ONE scope.
    let mut scope = disp.batch_scope();
    let prev_hc_b = scope.upload_f32(&prev_hc);
    let hc_fn_b = scope.upload_f32(&hc_fn);
    let scale_b = scope.upload_f32(&hc_scale);
    let base_b = scope.upload_f32(&hc_base);
    let hc_gamma_b = scope.upload_f32(&hc_norm_gamma);
    let unit_gamma_b = scope.upload_f32(&unit_gamma);
    let attn_q_a_b = scope.upload_f32(&attn_q_a);
    let gamma_q_b = scope.upload_f32(&gamma_q);
    let attn_q_b_b = scope.upload_f32(&attn_q_b);
    let attn_kv_b = scope.upload_f32(&attn_kv);
    let gamma_kv_b = scope.upload_f32(&qkv_gamma_kv);

    let outs_buf = scope
        .encode_layer_attn_half(
            &prev_hc_b, &hc_fn_b, &scale_b, &base_b, &hc_gamma_b, &unit_gamma_b,
            false,
            &attn_q_a_b, &gamma_q_b, &attn_q_b_b, &attn_kv_b, &gamma_kv_b,
            n_hc, d_embd, n_lora_q, n_head, head_dim, kv_row,
            p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps,
            &p, pos,
        )
        .expect("encode_layer_attn_half");
    let outs = scope.flush_and_read_multi(&[
        &outs_buf.qr_normed,
        &outs_buf.q_heads,
        &outs_buf.kv_normed_rotated,
    ]);
    let qr_normed_a = &outs[0];
    let q_heads_a = &outs[1];
    let kv_rotated_a = &outs[2];

    // ── Path B: sequential — hc_collapse_norm in one scope, read
    //   normed, then attn_qkv_chain_batched (inherent) for QKV,
    //   then kv_norm_rope_chain (inherent) for rms+rope.
    let mut scope_b = disp.batch_scope();
    let prev_hc_b2 = scope_b.upload_f32(&prev_hc);
    let hc_fn_b2 = scope_b.upload_f32(&hc_fn);
    let scale_b2 = scope_b.upload_f32(&hc_scale);
    let base_b2 = scope_b.upload_f32(&hc_base);
    let hc_gamma_b2 = scope_b.upload_f32(&hc_norm_gamma);
    let unit_gamma_b2 = scope_b.upload_f32(&unit_gamma);
    let (_split_b, _cur_b, normed_buf_b) = scope_b
        .hc_collapse_norm(
            &prev_hc_b2, &hc_fn_b2, &scale_b2, &base_b2, &hc_gamma_b2,
            n_hc, d_embd, p.hc_sinkhorn_iter as i32,
            p.hc_eps, p.rms_eps, &unit_gamma_b2,
            false,
        )
        .expect("hc_collapse_norm (B)");
    let normed_b = scope_b.flush_and_read(&normed_buf_b);

    let (qr_normed_b, q_heads_b, kv_raw_row_b) = disp
        .attn_qkv_chain_batched(
            &normed_b, &attn_q_a, &gamma_q, n_lora_q,
            &attn_q_b, n_head, head_dim, p.rms_eps,
            &attn_kv, kv_row,
        )
        .expect("attn_qkv_chain_batched (B)");
    let kv_rotated_b = disp.kv_norm_rope_chain(
        &kv_raw_row_b, &qkv_gamma_kv, &p, pos, p.rms_eps,
    ).expect("kv_norm_rope_chain (B)");

    // Bit-identical: A and B feed the SAME GPU-produced `normed`
    // into the SAME QKV kernels, then the SAME kv-row through the
    // SAME rms_norm + rope_tail.
    assert_eq!(qr_normed_a.len(), qr_normed_b.len());
    for (i, (a, b)) in qr_normed_a.iter().zip(qr_normed_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "qr_normed[{i}] A={a} B={b}");
    }
    assert_eq!(q_heads_a.len(), q_heads_b.len());
    for (i, (a, b)) in q_heads_a.iter().zip(q_heads_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "q_heads[{i}] A={a} B={b}");
    }
    assert_eq!(kv_rotated_a.len(), kv_rotated_b.len());
    for (i, (a, b)) in kv_rotated_a.iter().zip(kv_rotated_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "kv_rotated[{i}] A={a} B={b}");
    }
}
