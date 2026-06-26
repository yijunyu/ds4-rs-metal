//! Phase E M3 (M4 #330p) — verify `BatchScope::rms_norm_mul` covers
//! `AttentionDispatcher::kv_rms_norm_row` semantics at production shape.
//!
//! `kv_rms_norm_row`'s default impl (attn_dispatch.rs:369-384) is a plain
//! `rms_norm(kv_raw, gamma_kv, rms_eps)` over `n_lora_kv` floats. The
//! existing `BatchScope::rms_norm_mul` op does the same computation, so
//! M3 adds no new BatchScope op — it just locks in the equivalence with
//! a bit-close smoke test. The fused decoder at M5 will call
//! `rms_norm_mul` directly inside the one-cb-per-token encode.
//!
//! Standalone wiring of kv_rms_norm_row → GPU on the production path
//! was explicitly tried and regressed tok/s (Phase C-B Slice 4, commit
//! 78f43488). The point of M3 is only to confirm the GPU op produces
//! the right values; it stays unwired until M5 amortizes the cost.
//!
//! macOS-only — needs a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    AttentionDispatcher, CpuAttentionDispatcher, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn ds4_params(n_lora_kv: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 1,
        head_dim: n_lora_kv,
        n_rot: 64,
        n_lora_q: 1536,
        n_lora_kv,
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

fn build_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let raw: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.011).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let gamma: Vec<f32> = (0..n)
        .map(|i| 1.0 + (i as f32 * 0.017).sin() * 0.05)
        .collect();
    (raw, gamma)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(1e-6);
        assert!(
            abs < tol || rel < tol,
            "{label} at i={i}: gpu={g} cpu={w} |diff|={abs} rel={rel}"
        );
    }
}

#[test]
fn rms_norm_mul_matches_kv_rms_norm_row_ds4_shape() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // DS4 V4 Flash production: n_lora_kv == head_dim == 512 (see
    // `decode_step.rs:1867` comment).
    let n = 512;
    let p = ds4_params(n as u32);
    let (raw, gamma) = build_inputs(n);

    let want = CpuAttentionDispatcher.kv_rms_norm_row(&p, &raw, &gamma);

    let scope = disp.batch_scope();
    let x_b = scope.upload_f32(&raw);
    let g_b = scope.upload_f32(&gamma);
    let out_b = scope.rms_norm_mul(&x_b, &g_b, p.rms_eps).unwrap();
    let got = scope.flush_and_read(&out_b);

    // CPU uses f64 sum reduction; GPU uses f32 simdgroup reduction.
    // Bit-identity isn't guaranteed but absolute diff stays small for
    // n=512 normalized inputs.
    assert_close(&got, &want, 1e-4, "kv_rms_norm_row n=512");
}

#[test]
fn rms_norm_mul_matches_kv_rms_norm_row_small() {
    // Stress at a smaller shape (still %4-aligned for the float4 kernel).
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n = 128;
    let p = ds4_params(n as u32);
    let (raw, gamma) = build_inputs(n);

    let want = CpuAttentionDispatcher.kv_rms_norm_row(&p, &raw, &gamma);

    let scope = disp.batch_scope();
    let x_b = scope.upload_f32(&raw);
    let g_b = scope.upload_f32(&gamma);
    let out_b = scope.rms_norm_mul(&x_b, &g_b, p.rms_eps).unwrap();
    let got = scope.flush_and_read(&out_b);

    assert_close(&got, &want, 1e-4, "kv_rms_norm_row n=128");
}
