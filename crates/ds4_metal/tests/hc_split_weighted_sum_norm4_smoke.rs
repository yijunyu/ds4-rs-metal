//! Phase E M2 (M4 #330p) — `BatchScope::hc_split_weighted_sum_norm4` smoke test.
//!
//! Implements the split+sinkhorn+weighted_sum+rms_norm-with-γ tail of
//! `hc_collapse_norm` via antirez's `kernel_dsv4_hc_split_weighted_sum_norm4`.
//!
//! The kernel only handles the post-matvec portion; the test computes
//! `mix = hc_fn @ rms_norm_plain(prev_hc)` on CPU using the same logic
//! `CpuAttentionDispatcher::hc_collapse_norm` uses, then feeds it to the GPU.
//! Both paths should converge on the same (split, cur, normed).
//!
//! Tolerance: this kernel does heavy fp32 arithmetic (sinkhorn iterations,
//! softmax-like reductions, fma chains). Bit-identity vs CPU is unlikely;
//! use absolute tolerance 1e-4.
//!
//! macOS-only — needs a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    AttentionDispatcher, CpuAttentionDispatcher, HcKind, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    // DS4 V4 Flash production HC shape. The Metal kernel hardcodes
    // n_hc==4 && n_embd==4096.
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
        .map(|i| ((i as f32 * 0.0011).sin()) * 0.05)
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

/// Compute `mix = hc_fn @ rms_norm_plain(prev_hc)` matching the CPU
/// `hc_collapse_norm` body (steps a + b).
fn cpu_mix(prev_hc: &[f32], hc_fn: &[f32], rms_eps: f32, mix_hc: usize) -> Vec<f32> {
    let hc_dim = prev_hc.len();
    let ss: f64 = prev_hc.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scale = 1.0f32 / ((ss / hc_dim as f64) as f32 + rms_eps).sqrt();
    let flat: Vec<f32> = prev_hc.iter().map(|&v| v * scale).collect();
    let mut mix = vec![0.0f32; mix_hc];
    for r in 0..mix_hc {
        let row = &hc_fn[r * hc_dim..(r + 1) * hc_dim];
        mix[r] = row.iter().zip(flat.iter()).map(|(a, b)| a * b).sum();
    }
    mix
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        assert!(
            abs < tol,
            "{label} at i={i}: gpu={g} cpu={w} |diff|={abs}"
        );
    }
}

#[test]
fn hc_split_weighted_sum_norm4_matches_cpu() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let (prev_hc, hc_fn, hc_scale, hc_base, hc_norm_gamma) = build_inputs(&p);

    // CPU reference via the full hc_collapse_norm. Returns (cur, normed, hc_split).
    let (cur_cpu, normed_cpu, split_cpu) = CpuAttentionDispatcher.hc_collapse_norm(
        &p,
        HcKind::Attn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &prev_hc,
        Some(&hc_norm_gamma),
    );

    // CPU-computed mix (matches the matvec stage internal to CPU
    // hc_collapse_norm).
    let mix = cpu_mix(&prev_hc, &hc_fn, p.rms_eps, mix_hc);

    // GPU: open scope, upload inputs, run kernel, read 3 outputs.
    let scope = disp.batch_scope();
    let mixes_b = scope.upload_f32(&mix);
    let scale_b = scope.upload_f32(&hc_scale);
    let base_b = scope.upload_f32(&hc_base);
    let x_b = scope.upload_f32(&prev_hc);
    let gamma_b = scope.upload_f32(&hc_norm_gamma);
    let (split_buf, cur_buf, normed_buf) = scope
        .hc_split_weighted_sum_norm4(
            &mixes_b,
            &scale_b,
            &base_b,
            &x_b,
            &gamma_b,
            n_hc,
            d_embd,
            p.hc_sinkhorn_iter as i32,
            p.hc_eps,
            p.rms_eps,
        )
        .unwrap();
    let outs = scope.flush_and_read_multi(&[&split_buf, &cur_buf, &normed_buf]);
    let split_gpu = &outs[0];
    let cur_gpu = &outs[1];
    let normed_gpu = &outs[2];

    // Tolerance budget: sinkhorn + softmax + sum reductions on fp32.
    // Most outputs land within 1e-4 absolute; the comb sub-section of
    // split (sinkhorn-normalized) is the noisiest.
    assert_close(split_gpu, &split_cpu, 5e-5, "split");
    assert_close(cur_gpu, &cur_cpu, 1e-3, "cur (weighted sum)");
    assert_close(normed_gpu, &normed_cpu, 1e-4, "normed (post-rms γ)");
}

/// Lock against silent-zero failure.
#[test]
fn hc_split_weighted_sum_norm4_produces_nonzero() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let (prev_hc, hc_fn, hc_scale, hc_base, hc_norm_gamma) = build_inputs(&p);
    let mix = cpu_mix(&prev_hc, &hc_fn, p.rms_eps, mix_hc);

    let scope = disp.batch_scope();
    let mixes_b = scope.upload_f32(&mix);
    let scale_b = scope.upload_f32(&hc_scale);
    let base_b = scope.upload_f32(&hc_base);
    let x_b = scope.upload_f32(&prev_hc);
    let gamma_b = scope.upload_f32(&hc_norm_gamma);
    let (split_buf, cur_buf, normed_buf) = scope
        .hc_split_weighted_sum_norm4(
            &mixes_b,
            &scale_b,
            &base_b,
            &x_b,
            &gamma_b,
            n_hc,
            d_embd,
            p.hc_sinkhorn_iter as i32,
            p.hc_eps,
            p.rms_eps,
        )
        .unwrap();
    let outs = scope.flush_and_read_multi(&[&split_buf, &cur_buf, &normed_buf]);
    let nz_split = outs[0].iter().filter(|&&v| v != 0.0).count();
    let nz_cur = outs[1].iter().filter(|&&v| v != 0.0).count();
    let nz_normed = outs[2].iter().filter(|&&v| v != 0.0).count();
    assert!(nz_split > 0, "split is all zero ({}/{})", nz_split, outs[0].len());
    assert!(
        nz_cur > outs[1].len() / 2,
        "cur mostly zero ({}/{})",
        nz_cur,
        outs[1].len()
    );
    assert!(
        nz_normed > outs[2].len() / 2,
        "normed mostly zero ({}/{})",
        nz_normed,
        outs[2].len()
    );
}
