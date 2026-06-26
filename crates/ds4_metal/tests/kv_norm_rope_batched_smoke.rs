//! M4 #330p Phase C-B Slice 4 — `kv_norm_rope_batched` smoke test.
//!
//! Fuses `kv_rms_norm_row + rope_tail(KV tail)` into one MTLCommandBuffer.
//! This test asserts bit-identity vs running the two trait methods
//! sequentially. The chain has no transcendentals (rms_norm and rope_tail
//! use sqrt + sin/cos via Metal's correctly-rounded paths internally for
//! these inputs), but rope_tail uses `pow(freq_base, ...)` and trig
//! functions — Metal's `pow`/`sin`/`cos` are not guaranteed bit-identical
//! across implementations. So we compare against the same Metal trait
//! path, where the same kernel runs in both — exact bit identity holds.
//!
//! macOS-only — needs a real Metal device.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{AttentionDispatcher, LayerParams};
use ds4_metal::MetalDispatcher;

fn tiny_params(n_lora_kv: u32, n_rot: u32, head_dim: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 256,
        n_hc: 1,
        n_head: 1,
        head_dim,
        n_rot,
        n_lora_q: 16,
        n_lora_kv,
        hc_sinkhorn_iter: 1,
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

fn build_inputs(n_lora_kv: usize) -> (Vec<f32>, Vec<f32>) {
    let kv_raw: Vec<f32> = (0..n_lora_kv)
        .map(|i| ((i as f32 * 0.013).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let gamma: Vec<f32> = (0..n_lora_kv)
        .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
        .collect();
    (kv_raw, gamma)
}

fn assert_bit_identical(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label} at i={i}: batched={g} sequential={w}"
        );
    }
}

#[test]
fn kv_norm_rope_matches_sequential() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // n_lora_kv must be %4 (rms_norm float4 lanes). n_rot must be even.
    let p = tiny_params(64, 16, 64);
    let n_lora_kv = p.n_lora_kv as usize;
    let n_rot = p.n_rot as usize;
    let (kv_raw, gamma) = build_inputs(n_lora_kv);
    let pos: u32 = 3;

    // Fused: one cb.
    let batched = disp.kv_norm_rope_batched(&p, &kv_raw, &gamma, pos);

    // Sequential reference via the same Metal trait methods.
    let mut reference = disp.kv_rms_norm_row(&p, &kv_raw, &gamma);
    disp.rope_tail(&p, &mut reference[n_lora_kv - n_rot..n_lora_kv], pos, false);

    assert_bit_identical("kv_norm_rope batched != sequential", &batched, &reference);
}

#[test]
fn kv_norm_rope_matches_sequential_ds4_shape() {
    // Scaled DS4 V4 Flash KV shape (head_dim=512, n_rot=64 in production;
    // use a smaller variant here that still exercises the float4-lane and
    // rope-tail-half pipelines).
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = tiny_params(128, 32, 128);
    let n_lora_kv = p.n_lora_kv as usize;
    let n_rot = p.n_rot as usize;
    let (kv_raw, gamma) = build_inputs(n_lora_kv);
    let pos: u32 = 17;

    let batched = disp.kv_norm_rope_batched(&p, &kv_raw, &gamma, pos);

    let mut reference = disp.kv_rms_norm_row(&p, &kv_raw, &gamma);
    disp.rope_tail(&p, &mut reference[n_lora_kv - n_rot..n_lora_kv], pos, false);

    assert_bit_identical(
        "kv_norm_rope batched (ds4 shape) != sequential",
        &batched,
        &reference,
    );
}

/// Locks against silent-zero failures.
#[test]
fn kv_norm_rope_produces_nonzero_output() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = tiny_params(64, 16, 64);
    let n_lora_kv = p.n_lora_kv as usize;
    let (kv_raw, gamma) = build_inputs(n_lora_kv);
    let out = disp.kv_norm_rope_batched(&p, &kv_raw, &gamma, 5);
    let nonzero = out.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > out.len() / 2,
        "kv_norm_rope_batched output mostly zero ({}/{})",
        nonzero,
        out.len()
    );
}
