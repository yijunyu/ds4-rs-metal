//! Phase E M5.3 — per-layer encoder skeleton smoke test.
//!
//! Proves that two existing chains (`hc_collapse_norm` and
//! `attn_qkv_chain` via the new `encode_attn_qkv_chain`) compose
//! into a SINGLE `BatchScope` and produce bit-identical output to
//! running them as two separate scopes/cbs.
//!
//! This is the first concrete demonstration of the M5 unified-cb
//! pattern: GPU intermediates flow between ops without round-trip
//! through CPU.
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
        n_head: 8,        // smaller than DS4 prod (128) to keep test fast
        head_dim: 64,
        n_rot: 16,
        n_lora_q: 64,     // (n_lora_q % 4 == 0, % 2 == 0)
        n_lora_kv: 128,   // (% 2 == 0)
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
fn hc_collapse_then_attn_qkv_one_cb_matches_two_cb() {
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

    // Synthetic but deterministic inputs.
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
    let unit_gamma_hc: Vec<f32> = vec![1.0f32; hc_dim];
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

    // ── Path A (FUSED): one BatchScope encodes hc_collapse_norm +
    //   encode_attn_qkv_chain. flat/mix/normed never touch CPU.
    let scope = disp.batch_scope();
    let prev_hc_b = scope.upload_f32(&prev_hc);
    let hc_fn_b = scope.upload_f32(&hc_fn);
    let scale_b = scope.upload_f32(&hc_scale);
    let base_b = scope.upload_f32(&hc_base);
    let hc_gamma_b = scope.upload_f32(&hc_norm_gamma);
    let unit_gamma_b = scope.upload_f32(&unit_gamma_hc);
    let attn_q_a_b = scope.upload_f32(&attn_q_a);
    let gamma_q_b = scope.upload_f32(&gamma_q);
    let attn_q_b_b = scope.upload_f32(&attn_q_b);
    let attn_kv_b = scope.upload_f32(&attn_kv);

    let (_split, _cur, normed_a) = scope
        .hc_collapse_norm(
            &prev_hc_b, &hc_fn_b, &scale_b, &base_b, &hc_gamma_b,
            n_hc, d_embd, p.hc_sinkhorn_iter as i32,
            p.hc_eps, p.rms_eps, &unit_gamma_b,
            false,
        )
        .expect("hc_collapse_norm chain");
    let (qr_normed_a_buf, q_heads_a_buf, kv_raw_row_a_buf) = scope
        .encode_attn_qkv_chain(
            &normed_a, &attn_q_a_b, &gamma_q_b, &attn_q_b_b, &attn_kv_b,
            n_lora_q, n_head, head_dim, p.rms_eps, kv_row,
        )
        .expect("encode_attn_qkv_chain");
    let outs = scope.flush_and_read_multi(&[
        &qr_normed_a_buf, &q_heads_a_buf, &kv_raw_row_a_buf,
    ]);
    let qr_normed_a = &outs[0];
    let q_heads_a = &outs[1];
    let kv_raw_row_a = &outs[2];

    // ── Path B (SEQUENTIAL): two scopes/cbs. Stage 1 produces
    //   normed; stage 2 (the existing inherent attn_qkv_chain_batched)
    //   consumes the normed Vec<f32>.
    let scope_b = disp.batch_scope();
    let prev_hc_b2 = scope_b.upload_f32(&prev_hc);
    let hc_fn_b2 = scope_b.upload_f32(&hc_fn);
    let scale_b2 = scope_b.upload_f32(&hc_scale);
    let base_b2 = scope_b.upload_f32(&hc_base);
    let hc_gamma_b2 = scope_b.upload_f32(&hc_norm_gamma);
    let unit_gamma_b2 = scope_b.upload_f32(&unit_gamma_hc);
    let (_split2, _cur2, normed_buf_b) = scope_b
        .hc_collapse_norm(
            &prev_hc_b2, &hc_fn_b2, &scale_b2, &base_b2, &hc_gamma_b2,
            n_hc, d_embd, p.hc_sinkhorn_iter as i32,
            p.hc_eps, p.rms_eps, &unit_gamma_b2,
            false,
        )
        .expect("hc_collapse_norm chain (B)");
    let normed_b_vec = scope_b.flush_and_read(&normed_buf_b);

    let (qr_normed_b, q_heads_b, kv_raw_row_b) = disp
        .attn_qkv_chain_batched(
            &normed_b_vec, &attn_q_a, &gamma_q, n_lora_q,
            &attn_q_b, n_head, head_dim, p.rms_eps,
            &attn_kv, kv_row,
        )
        .expect("attn_qkv_chain_batched (B)");

    // Bit-identical: both paths feed the SAME GPU-produced `normed`
    // values into the SAME GPU kernels with the SAME args. The only
    // difference is whether the QKV ops were encoded into the same
    // cb as the upstream hc_collapse_norm.
    assert_eq!(qr_normed_a.len(), qr_normed_b.len());
    for (i, (a, b)) in qr_normed_a.iter().zip(qr_normed_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "qr_normed[{i}] fused={a} sequential={b}");
    }
    assert_eq!(q_heads_a.len(), q_heads_b.len());
    for (i, (a, b)) in q_heads_a.iter().zip(q_heads_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "q_heads[{i}] fused={a} sequential={b}");
    }
    assert_eq!(kv_raw_row_a.len(), kv_raw_row_b.len());
    for (i, (a, b)) in kv_raw_row_a.iter().zip(kv_raw_row_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "kv_raw_row[{i}] fused={a} sequential={b}");
    }
}
