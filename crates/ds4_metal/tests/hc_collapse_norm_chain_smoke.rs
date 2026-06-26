//! Phase E M5-prep — `BatchScope::hc_collapse_norm` chain test.
//!
//! Composes 3 BatchScope ops into a single-cb GPU equivalent of
//! `AttentionDispatcher::hc_collapse_norm`:
//!   1. rms_norm_mul(prev_hc, unit_γ)         — Phase C-A op
//!   2. matvec_f32(hc_fn, flat, mix_hc)       — Phase C-A op
//!   3. hc_split_weighted_sum_norm4(...)      — Phase E M2 op
//!
//! All three encoders go into ONE cb; the intermediate `flat` and
//! `mix` buffers never round-trip through CPU. Verifies bit-close
//! match with CPU reference at DS4 V4 Flash production shape
//! (n_hc=4, n_embd=4096).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    AttentionDispatcher, CpuAttentionDispatcher, HcKind, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 1,
        head_dim: 1,
        n_rot: 1,
        n_lora_q: 1,
        n_lora_kv: 1,
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

fn build_inputs(p: &LayerParams) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let prev_hc: Vec<f32> = (0..hc_dim)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| (i as f32 * 0.0011).sin() * 0.05)
        .collect();
    let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
    let hc_base: Vec<f32> = (0..mix_hc)
        .map(|i| 0.1 + (i as f32) * 0.01)
        .collect();
    let hc_norm_gamma: Vec<f32> = (0..d_embd)
        .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05)
        .collect();
    (prev_hc, hc_fn, hc_scale, hc_base, hc_norm_gamma)
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
fn hc_collapse_norm_chain_matches_cpu() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_hc = p.n_hc as usize;
    let n_embd = p.d_embd as usize;
    let hc_dim = n_hc * n_embd;
    let (prev_hc, hc_fn, hc_scale, hc_base, hc_norm_gamma) = build_inputs(&p);

    // CPU reference.
    let (cur_cpu, normed_cpu, split_cpu) = CpuAttentionDispatcher.hc_collapse_norm(
        &p,
        HcKind::Attn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &prev_hc,
        Some(&hc_norm_gamma),
    );

    // GPU chain.
    let unit_gamma_hc_dim = vec![1.0f32; hc_dim];
    let scope = disp.batch_scope();
    let prev_hc_b = scope.upload_f32(&prev_hc);
    let hc_fn_b = scope.upload_f32(&hc_fn);
    let scale_b = scope.upload_f32(&hc_scale);
    let base_b = scope.upload_f32(&hc_base);
    let gamma_b = scope.upload_f32(&hc_norm_gamma);
    let unit_gamma_b = scope.upload_f32(&unit_gamma_hc_dim);

    let (split_buf, cur_buf, normed_buf) = scope
        .hc_collapse_norm(
            &prev_hc_b,
            &hc_fn_b,
            &scale_b,
            &base_b,
            &gamma_b,
            n_hc,
            n_embd,
            p.hc_sinkhorn_iter as i32,
            p.hc_eps,
            p.rms_eps,
            &unit_gamma_b,
            false,
        )
        .expect("hc_collapse_norm chain");
    let outs = scope.flush_and_read_multi(&[&split_buf, &cur_buf, &normed_buf]);
    let split_gpu = &outs[0];
    let cur_gpu = &outs[1];
    let normed_gpu = &outs[2];

    // Tolerance budget — same class as the M2 standalone smoke test
    // (split 5e-5, cur 1e-3, normed 1e-4). The chain adds 2 stages
    // upstream (matvec + rms_norm_plain) which also drift in fp32.
    // Empirically the chain output is within those bounds.
    assert_close(split_gpu, &split_cpu, 5e-5, "split (chain)");
    assert_close(cur_gpu, &cur_cpu, 2e-3, "cur (chain)");
    assert_close(normed_gpu, &normed_cpu, 2e-4, "normed (chain)");
}
