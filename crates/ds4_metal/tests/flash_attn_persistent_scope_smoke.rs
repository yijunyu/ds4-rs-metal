//! Phase F task #86 — `BatchScope::flash_attn_decode_persistent` smoke.
//!
//! Validates that flash_attn encoded INTO a shared BatchScope (no
//! intermediate commit) produces the same heads output as the
//! existing `MetalDispatcher::flash_attn_decode_metal_persistent`
//! which commits its own cb. Same kernel + same dispatch args; only
//! the cb-lifecycle differs.
//!
//! When this is solid + `rope_tail_q_heads_in_place` lands (already
//! shipped), the next step is wiring them together in encode_first_half:
//!   encode_layer_attn_half (one scope) → rope_tail_q_heads → flash_attn_persistent
//! all in the SAME scope, collapsing 3+ cbs to 1 per layer at the
//! attn-half boundary.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

fn fa_params() -> LayerParams {
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
fn flash_attn_persistent_scope_matches_reference() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = fa_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    let raw_cap: u32 = 64;
    let n_raw: u32 = 32; // 32-aligned, > 0 — fast path

    // Deterministic Q + KV + attn_sinks.
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.0017 - 0.4).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..(raw_cap as usize) * row)
        .map(|i| (i as f32 * 0.0009 + 0.1).cos() * 0.5)
        .collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.1 * h as f32).collect();

    // Populate the persistent buffer for two distinct layer_idxs (one for
    // reference, one for scope path) with the SAME bytes so flash_attn
    // sees identical inputs.
    let cache_byte_len = (raw_cap as usize) * row * std::mem::size_of::<f32>();
    for layer_idx in [11u32, 12u32] {
        let buf = disp.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                kv_raw.as_ptr(),
                buf.contents() as *mut f32,
                kv_raw.len(),
            );
        }
    }

    // ── Reference: existing `flash_attn_decode_metal_persistent` (own cb).
    let ref_out = disp.flash_attn_decode_metal_persistent(
        11, &params, &q, n_raw, raw_cap, None, 0, None, 0, &attn_sinks,
    );

    // ── Scope path: `BatchScope::flash_attn_decode_persistent` encodes
    //   into a shared scope. Flush at end reads the heads output.
    let scope = disp.batch_scope();
    let heads_db = scope
        .flash_attn_decode_persistent(12, &params, &q, n_raw, raw_cap, &attn_sinks)
        .expect("flash_attn_decode_persistent");
    let scope_out = scope.flush_and_read(&heads_db);

    // Bit-identical: same kernel, same dispatch args, same KV input.
    assert_eq!(ref_out.len(), scope_out.len());
    for (i, (a, b)) in ref_out.iter().zip(&scope_out).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "heads[{i}] ref={a} scope={b}"
        );
    }
}

#[test]
fn flash_attn_persistent_scope_chains_with_more_ops() {
    // Validates that flash_attn can be the FIRST op in a scope,
    // followed by additional encoded work, all in ONE cb. We don't
    // need a meaningful downstream op here — just prove the scope
    // accepts further work and reads back the flash_attn output
    // correctly after that work has also been committed.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = fa_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    let raw_cap: u32 = 64;
    let n_raw: u32 = 32;

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i as f32) * 0.013).sin() * 0.3)
        .collect();
    let kv_raw: Vec<f32> = (0..(raw_cap as usize) * row)
        .map(|i| (i as f32 * 0.0019 - 0.2).cos() * 0.4)
        .collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| -0.05 - (h as f32) * 0.02).collect();

    let cache_byte_len = (raw_cap as usize) * row * std::mem::size_of::<f32>();
    let layer_idx: u32 = 17;
    {
        let buf = disp.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                kv_raw.as_ptr(),
                buf.contents() as *mut f32,
                kv_raw.len(),
            );
        }
    }

    let scope = disp.batch_scope();
    let heads_db = scope
        .flash_attn_decode_persistent(layer_idx, &params, &q, n_raw, raw_cap, &attn_sinks)
        .expect("flash_attn_decode_persistent");

    // Encode a trivial follow-up op into the same scope: an in-place
    // rms_norm with identity gamma. This validates the cb stays open
    // and the encoder can be re-created after flash_attn's 2-pass
    // dispatch. The op operates on heads_db (which has n_head*head_dim
    // elements, divisible by 4 since head_dim=512).
    let gamma: Vec<f32> = vec![1.0f32; n_head * head_dim];
    let gamma_b = scope.upload_f32(&gamma);
    let normed_db = scope
        .rms_norm_mul(&heads_db, &gamma_b, params.rms_eps)
        .expect("rms_norm_mul after flash_attn");

    // Flush — commits the cb (containing flash_attn vec+reduce + rms_norm)
    // and reads back. Both outputs should be valid since they were both
    // encoded into the same cb in order.
    let outs = scope.flush_and_read_multi(&[&heads_db, &normed_db]);
    assert_eq!(outs[0].len(), n_head * head_dim);
    assert_eq!(outs[1].len(), n_head * head_dim);
    // rms_norm of the heads should produce finite values
    for (i, &v) in outs[1].iter().enumerate() {
        assert!(v.is_finite(), "normed[{i}] not finite: {v}");
    }
}
