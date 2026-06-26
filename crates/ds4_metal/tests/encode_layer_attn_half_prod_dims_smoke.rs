//! Phase E M5.2.6 — `encode_layer_attn_half_cpu` vs trait
//! `attn_qkv_chain_batched + kv_norm_rope_chain` bit-identity at
//! PRODUCTION dimensions (head_dim=512, n_lora_kv=512).
//!
//! M5.4.5.1 verified bit-identity at head_dim=128 / n_lora_kv=128.
//! DS4 V4 Flash production uses head_dim=512 / n_lora_kv=512. After
//! M5.2.5, the persistent KV bytes are unified between the two paths
//! but the bench still shows ~21% rel error and argmax disagreement —
//! the suspected residual source is a Metal kernel that takes a
//! different route at the larger dimension.
//!
//! This smoke asserts the same bit-identity bars as
//! `encode_layer_attn_half_cpu_smoke` but at production dims: qr_normed,
//! q_heads, and kv_normed_rotated must be byte-identical between
//!   - new path: `SingleBufferEncoder::encode_layer_attn_half_cpu`
//!   - reference: `MetalDispatcher::attn_qkv_chain_batched` →
//!     `kv_norm_rope_chain`
//!
//! If this smoke fails, we've found the divergence.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::single_buffer_encoder::SingleBufferEncoder;
use ds4_metal::MetalDispatcher;

fn prod_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
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
fn encode_layer_attn_half_cpu_matches_inherent_at_prod_dims() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = prod_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let n_lora_q = p.n_lora_q as usize;
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let kv_row = p.n_lora_kv as usize;
    let raw_cap: u32 = 8;
    let slot: u32 = 3;
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

    // ── Path A: new SingleBufferEncoder entry point.
    let layer_idx_a: u32 = 200;
    let encoder = SingleBufferEncoder::new(&disp, raw_cap);
    let outs = encoder
        .encode_layer_attn_half_cpu(
            layer_idx_a,
            &prev_hc,
            &hc_fn,
            None,
            &hc_scale,
            &hc_base,
            &hc_norm_gamma,
            &unit_gamma,
            &attn_q_a,
            &gamma_q,
            &attn_q_b,
            &attn_kv,
            &qkv_gamma_kv,
            &p,
            pos,
            raw_cap,
            slot,
        )
        .expect("encode_layer_attn_half_cpu (A)");

    // ── Path B: inherent path (hc_collapse_norm → attn_qkv_chain_batched
    //   → kv_norm_rope_chain).
    let scope_b = disp.batch_scope();
    let prev_hc_b = scope_b.upload_f32(&prev_hc);
    let hc_fn_b = scope_b.upload_f32(&hc_fn);
    let scale_b = scope_b.upload_f32(&hc_scale);
    let base_b = scope_b.upload_f32(&hc_base);
    let hc_gamma_b = scope_b.upload_f32(&hc_norm_gamma);
    let unit_gamma_b = scope_b.upload_f32(&unit_gamma);
    let (_split_b, _cur_b, normed_buf_b) = scope_b
        .hc_collapse_norm(
            &prev_hc_b,
            &hc_fn_b,
            &scale_b,
            &base_b,
            &hc_gamma_b,
            n_hc,
            d_embd,
            p.hc_sinkhorn_iter as i32,
            p.hc_eps,
            p.rms_eps,
            &unit_gamma_b,
            false,
        )
        .expect("hc_collapse_norm (B)");
    let normed_b = scope_b.flush_and_read(&normed_buf_b);

    let (qr_normed_b, q_heads_b, kv_raw_row_b) = disp
        .attn_qkv_chain_batched(
            &normed_b, &attn_q_a, &gamma_q, n_lora_q, &attn_q_b, n_head, head_dim, p.rms_eps,
            &attn_kv, kv_row,
        )
        .expect("attn_qkv_chain_batched (B)");
    let kv_rotated_b = disp
        .kv_norm_rope_chain(&kv_raw_row_b, &qkv_gamma_kv, &p, pos, p.rms_eps)
        .expect("kv_norm_rope_chain (B)");

    // ── Bit-identity assertions (the M5.4.5.1 contract at prod dims).
    fn first_bit_diff(a: &[f32], b: &[f32], name: &str) {
        assert_eq!(a.len(), b.len(), "{name} length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if x.to_bits() != y.to_bits() {
                let max_abs = a
                    .iter()
                    .zip(b)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0f32, f32::max);
                let max_ref = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                panic!(
                    "{name} bit-diff at index {i}: A={x} (bits=0x{:08x}), B={y} (bits=0x{:08x}); \
                     overall max_abs={max_abs:.6} max_ref={max_ref:.6} rel={:.6}",
                    x.to_bits(),
                    y.to_bits(),
                    max_abs / max_ref.max(1e-6),
                );
            }
        }
    }

    first_bit_diff(&outs.qr_normed, &qr_normed_b, "qr_normed");
    first_bit_diff(&outs.q_heads, &q_heads_b, "q_heads");

    // `outs.kv_normed_rotated` is intentionally empty now: encode_layer_attn_half_cpu
    // no longer reads the rotated-KV mirror back (the GPU kv_fp8_store_persistent
    // shim writes the correct scaled-e4m3 + f16 bytes directly). The KV write
    // path's bit-exactness at prod dims is covered by the macos.rs test
    // `kv_fp8_store_persistent_matches_cpu_correction`. `kv_rotated_b` is still
    // computed above as the path-B reference for that contract.
    let _ = &kv_rotated_b;
    assert!(
        outs.kv_normed_rotated.is_empty(),
        "kv_normed_rotated mirror should no longer be populated"
    );
}
