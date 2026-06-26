//! Phase E M5.4.2 — `BatchScope::kv_fp8_store_persistent` composable
//! smoke test.
//!
//! Verifies that chaining `encode_layer_attn_half` +
//! `kv_fp8_store_persistent` in ONE BatchScope produces the same
//! persistent-buffer contents as running them across two scopes
//! (encode_layer_attn_half + inherent `kv_fp8_store_persistent`).
//!
//! Each path takes the same inputs through the same GPU kernels —
//! bit-identity is the right bar.
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
        head_dim: 128,    // == n_lora_kv for kv_fp8_store contract
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
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            out.as_mut_ptr(),
            n,
        );
    }
    out
}

#[test]
fn attn_half_then_kv_fp8_one_cb_matches_two_cb() {
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

    // ── Path A: encode_layer_attn_half + kv_fp8_store_persistent
    //   ALL in ONE BatchScope. kv_normed_rotated never touches CPU
    //   between rope_tail and kv_fp8_store.
    let layer_idx_a: u32 = 100;
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

    let half = scope
        .encode_layer_attn_half(
            &prev_hc_b, &hc_fn_b, &scale_b, &base_b, &hc_gamma_b, &unit_gamma_b,
            false,
            &attn_q_a_b, &gamma_q_b, &attn_q_b_b, &attn_kv_b, &gamma_kv_b,
            n_hc, d_embd, n_lora_q, n_head, head_dim, kv_row,
            p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps,
            &p, pos,
        )
        .expect("encode_layer_attn_half (A)");
    let cache_a = scope
        .kv_fp8_store_persistent(
            layer_idx_a, &half.kv_normed_rotated, &p, raw_cap, slot,
        )
        .expect("kv_fp8_store_persistent (A)");
    // Flush; cache_a is the persistent buffer's deferred handle.
    let _ = scope.flush_and_read_multi(&[&cache_a]);
    let cache_a_full = read_buf(&cache_a.buffer().clone(), (raw_cap as usize) * kv_row);

    // ── Path B: encode_layer_attn_half in one scope → read
    //   kv_normed_rotated → inherent `kv_fp8_store_persistent`.
    let layer_idx_b: u32 = 101;
    let mut scope_b = disp.batch_scope();
    let prev_hc_b2 = scope_b.upload_f32(&prev_hc);
    let hc_fn_b2 = scope_b.upload_f32(&hc_fn);
    let scale_b2 = scope_b.upload_f32(&hc_scale);
    let base_b2 = scope_b.upload_f32(&hc_base);
    let hc_gamma_b2 = scope_b.upload_f32(&hc_norm_gamma);
    let unit_gamma_b2 = scope_b.upload_f32(&unit_gamma);
    let attn_q_a_b2 = scope_b.upload_f32(&attn_q_a);
    let gamma_q_b2 = scope_b.upload_f32(&gamma_q);
    let attn_q_b_b2 = scope_b.upload_f32(&attn_q_b);
    let attn_kv_b2 = scope_b.upload_f32(&attn_kv);
    let gamma_kv_b2 = scope_b.upload_f32(&qkv_gamma_kv);
    let half_b = scope_b
        .encode_layer_attn_half(
            &prev_hc_b2, &hc_fn_b2, &scale_b2, &base_b2, &hc_gamma_b2,
            &unit_gamma_b2,
            false,
            &attn_q_a_b2, &gamma_q_b2, &attn_q_b_b2, &attn_kv_b2, &gamma_kv_b2,
            n_hc, d_embd, n_lora_q, n_head, head_dim, kv_row,
            p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps,
            &p, pos,
        )
        .expect("encode_layer_attn_half (B)");
    let kv_rotated_b = scope_b.flush_and_read(&half_b.kv_normed_rotated);

    let cache_b = disp.kv_fp8_store_persistent(
        layer_idx_b, &p, &kv_rotated_b, raw_cap, slot,
    );
    let cache_b_full = read_buf(&cache_b, (raw_cap as usize) * kv_row);

    assert_eq!(cache_a_full.len(), cache_b_full.len());
    for (i, (a, b)) in cache_a_full.iter().zip(cache_b_full.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "persistent cache[{i}] A={a} B={b} (one-cb vs two-cb)"
        );
    }
}
