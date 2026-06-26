//! Phase E M1 (M4 #330p) — `BatchScope::hc_expand_add` smoke test.
//!
//! GPU equivalent of `ds4_engine::attn_dispatch::hc_expand_add_only`.
//! Uses antirez's `kernel_dsv4_hc_expand` via the bridge. Compares to
//! the CPU reference at production-realistic and small shapes.
//!
//! The accumulation order on GPU may differ slightly from CPU's
//! `for src in 0..n_hc { acc += ... }` (each thread loops the same
//! way, so order is consistent). Bit-identity holds: no transcendentals,
//! all multiply-adds, fp32 throughout.
//!
//! macOS-only — needs a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{hc_expand_add_only, LayerParams};
use ds4_metal::MetalDispatcher;

fn tiny_params(n_hc: u32, d_embd: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd,
        n_hc,
        n_head: 1,
        head_dim: 1,
        n_rot: 1,
        n_lora_q: 1,
        n_lora_kv: 1,
        hc_sinkhorn_iter: 1,
        hc_eps: 1e-6,
        rms_eps: 1e-6,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 1,
    }
}

fn build_inputs(
    n_hc: usize,
    d_embd: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let shared: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 - 0.05).clamp(-2.0, 2.0))
        .collect();
    let routed: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.013).cos() * 0.3 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let after_attn: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| ((i as f32 * 0.011).sin() * 0.25))
        .collect();
    let hc_split_post: Vec<f32> =
        (0..n_hc).map(|i| 0.5 + (i as f32) * 0.125).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| 0.25 + (i as f32 * 0.05) - ((i % 3) as f32) * 0.1)
        .collect();
    (shared, routed, after_attn, hc_split_post, hc_split_comb)
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
fn hc_expand_add_matches_cpu_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 4;
    let d_embd = 256;
    let p = tiny_params(n_hc as u32, d_embd as u32);
    let (shared, routed, after_attn, hc_post, hc_comb) = build_inputs(n_hc, d_embd);

    let want = hc_expand_add_only(&p, &shared, &routed, &after_attn, &hc_post, &hc_comb);

    let scope = disp.batch_scope();
    let sb = scope.upload_f32(&shared);
    let rb = scope.upload_f32(&routed);
    let ab = scope.upload_f32(&after_attn);
    let pb = scope.upload_f32(&hc_post);
    let cb = scope.upload_f32(&hc_comb);
    let out = scope.hc_expand_add(&sb, &rb, &ab, &pb, &cb, n_hc, d_embd).unwrap();
    let got = scope.flush_and_read(&out);

    // Bit-identity is the goal — fp32 fma vs scalar mac may differ at the
    // last bit on some chips. Tolerance picked at 1e-5 absolute; if both
    // paths use the same fma policy this is comfortably tight.
    assert_close(&got, &want, 1e-5, "hc_expand_add small");
}

#[test]
fn hc_expand_add_matches_cpu_ds4_shape() {
    // DS4 V4 Flash production: n_hc=4, d_embd=4096.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 4;
    let d_embd = 4096;
    let p = tiny_params(n_hc as u32, d_embd as u32);
    let (shared, routed, after_attn, hc_post, hc_comb) = build_inputs(n_hc, d_embd);

    let want = hc_expand_add_only(&p, &shared, &routed, &after_attn, &hc_post, &hc_comb);

    let scope = disp.batch_scope();
    let sb = scope.upload_f32(&shared);
    let rb = scope.upload_f32(&routed);
    let ab = scope.upload_f32(&after_attn);
    let pb = scope.upload_f32(&hc_post);
    let cb = scope.upload_f32(&hc_comb);
    let out = scope.hc_expand_add(&sb, &rb, &ab, &pb, &cb, n_hc, d_embd).unwrap();
    let got = scope.flush_and_read(&out);

    assert_close(&got, &want, 1e-5, "hc_expand_add ds4 shape");
}

/// Lock against silent-zero failures.
#[test]
fn hc_expand_add_produces_nonzero() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 4;
    let d_embd = 256;
    let (shared, routed, after_attn, hc_post, hc_comb) = build_inputs(n_hc, d_embd);

    let scope = disp.batch_scope();
    let sb = scope.upload_f32(&shared);
    let rb = scope.upload_f32(&routed);
    let ab = scope.upload_f32(&after_attn);
    let pb = scope.upload_f32(&hc_post);
    let cb = scope.upload_f32(&hc_comb);
    let out = scope.hc_expand_add(&sb, &rb, &ab, &pb, &cb, n_hc, d_embd).unwrap();
    let got = scope.flush_and_read(&out);

    let nonzero = got.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > got.len() / 2,
        "hc_expand_add output mostly zero ({}/{})",
        nonzero,
        got.len()
    );
}
