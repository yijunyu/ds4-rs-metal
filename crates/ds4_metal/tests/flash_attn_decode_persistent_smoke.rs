//! Phase E M5.2.3 — `flash_attn_decode_metal_persistent` smoke test.
//!
//! Verifies that reading KV from the persistent buffer pool produces
//! the same output as passing the same data as a `&[f32]` slice
//! through the existing `flash_attn_decode_metal` path. The two
//! paths share the kernel + dispatch + args; only the SOURCE of the
//! raw KV bytes differs.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{AttentionDispatcher, LayerParams};
use ds4_metal::MetalDispatcher;

fn fa_dk512_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 128,
        n_hc: 1,
        n_head: 2,
        head_dim: 512,
        n_rot: 64,
        n_lora_q: 128,
        n_lora_kv: 512,
        hc_sinkhorn_iter: 0,
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
fn flash_attn_decode_persistent_matches_inline_kv() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = fa_dk512_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    let raw_cap: u32 = 64;
    let n_raw: u32 = 32;

    // Deterministic Q + KV cache + attn_sinks.
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.0017 - 0.4).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..raw_cap as usize * row)
        .map(|i| (i as f32 * 0.0009 + 0.1).cos() * 0.5)
        .collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.1 * h as f32).collect();

    // Reference: pass kv_raw as a slice via the existing surface.
    let ref_out = disp.flash_attn_decode(
        &params, &q, &kv_raw, n_raw, raw_cap, 0, None, 0, None, 0, &attn_sinks,
    );

    // Persistent path: stage the same kv_raw into the persistent
    // buffer for layer 13 (any unused layer index), then call
    // `flash_attn_decode_metal_persistent`.
    let byte_size = (raw_cap as usize) * row * std::mem::size_of::<f32>();
    let buf = disp.kv_buffer_or_alloc(13, byte_size);
    unsafe {
        std::ptr::copy_nonoverlapping(
            kv_raw.as_ptr(),
            buf.contents() as *mut f32,
            kv_raw.len(),
        );
    }
    let pers_out = disp.flash_attn_decode_metal_persistent(
        13, &params, &q, n_raw, raw_cap, None, 0, None, 0, &attn_sinks,
    );

    // Bit-identical: same kernel, same dispatch, same data, same Q.
    // AttnHeadsOut = Vec<f32> (heads × head_dim, row-major).
    assert_eq!(pers_out.len(), ref_out.len(), "output length mismatch");
    for (i, (a, b)) in pers_out.iter().zip(ref_out.iter()).enumerate() {
        // Both go through `flash_attn_decode_metal` with identical
        // inputs — bit-identical is the right bar.
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "head[{i}] differs: persistent={a} reference={b}"
        );
    }
}
