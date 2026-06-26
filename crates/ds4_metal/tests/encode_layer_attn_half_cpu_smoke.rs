//! Phase E M5.4.5.1 — `SingleBufferEncoder::encode_layer_attn_half_cpu`
//! smoke test.
//!
//! Drives the per-layer attn-half through the new `SingleBufferEncoder`
//! entry point and asserts bit-identical equivalence with the legacy
//! inherent path:
//!
//!   Path A: `SingleBufferEncoder::encode_layer_attn_half_cpu`
//!     → `BatchScope::encode_layer_attn_half` + `kv_fp8_store_persistent`
//!     in ONE command buffer, flushed once.
//!
//!   Path B: inherent two-cb sequence (the path M4 #330p left in place):
//!     1. `MetalDispatcher::attn_qkv_chain_batched`   (cb1)
//!     2. `MetalDispatcher::kv_norm_rope_chain`       (cb2)
//!     3. `MetalDispatcher::kv_fp8_store_persistent`  (cb3)
//!
//!   The hc_collapse_norm output (Path A's first stage) is also
//!   shared via inherent for Path B: read normed via a one-shot
//!   `BatchScope::hc_collapse_norm`.
//!
//! All three of qr_normed, q_heads, kv_normed_rotated must be
//! bit-identical (each kernel is the SAME shader running on the SAME
//! operands; only host/GPU sync points differ). The persistent KV
//! cache buffer's contents at `slot` must also match Path B's
//! independent kv_fp8_store_persistent.
//!
//! This is the bit-identical proof for the M5.4.5.1 wiring on which
//! M5.4.5.2's multi-layer driver will stand.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::single_buffer_encoder::SingleBufferEncoder;
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 8,
        head_dim: 128, // == n_lora_kv for kv_fp8_store contract
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

fn read_buf(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
    }
    out
}

#[test]
fn encode_layer_attn_half_cpu_matches_inherent_path() {
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
    let raw_cap: u32 = 8;
    let slot: u32 = 3;
    let pos: u32 = 5;

    // Synthetic deterministic inputs.
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
    //   → kv_norm_rope_chain → kv_fp8_store_persistent).
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
    let layer_idx_b: u32 = 201;
    let cache_buf_b =
        disp.kv_fp8_store_persistent(layer_idx_b, &p, &kv_rotated_b, raw_cap, slot);
    let cache_b_full = read_buf(&cache_buf_b, (raw_cap as usize) * kv_row);

    // ── Bit-identical equivalence ─────────────────────────────────
    assert_eq!(outs.qr_normed.len(), qr_normed_b.len());
    for (i, (a, b)) in outs.qr_normed.iter().zip(qr_normed_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "qr_normed[{i}] A={a} B={b}");
    }
    assert_eq!(outs.q_heads.len(), q_heads_b.len());
    for (i, (a, b)) in outs.q_heads.iter().zip(q_heads_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "q_heads[{i}] A={a} B={b}");
    }
    // `outs.kv_normed_rotated` is intentionally empty: encode_layer_attn_half_cpu
    // no longer reads the rotated-KV mirror back, because the GPU
    // kv_fp8_store_persistent shim now writes the correct (scaled-e4m3 + f16)
    // bytes directly. The equivalence that matters — both paths landing
    // identical bytes in the persistent KV slot — is asserted below.
    assert!(
        outs.kv_normed_rotated.is_empty(),
        "kv_normed_rotated mirror should no longer be populated"
    );

    // ── Persistent KV cache buffer contents at `slot` match.
    // Path A wrote into the persistent buffer for `layer_idx_a`; read it
    // back via kv_buffer_or_alloc (same lazy slot).
    let cache_byte_len = (raw_cap as usize) * kv_row * std::mem::size_of::<f32>();
    let cache_a = disp.kv_buffer_or_alloc(layer_idx_a, cache_byte_len);
    let cache_a_full = read_buf(&cache_a, (raw_cap as usize) * kv_row);
    assert_eq!(cache_a_full.len(), cache_b_full.len());
    for (i, (a, b)) in cache_a_full.iter().zip(cache_b_full.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "persistent cache[{i}] A={a} B={b}"
        );
    }
}
