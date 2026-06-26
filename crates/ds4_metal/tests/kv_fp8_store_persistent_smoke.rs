//! Phase E M5.2.2 — `kv_fp8_store_persistent` smoke test.
//!
//! Verifies that the persistent-buffer kv_fp8_store path:
//!   1. produces values within e4m3 tolerance of the input (NOPE
//!      region) and f16 tolerance (n_rot tail) — matching the
//!      existing `kv_fp8_store_round_trip_within_e4m3_tolerance`
//!      test for the trait/CPU-oracle path
//!   2. writes to the correct SLOT of the persistent buffer
//!   3. persists data across calls — writing to slot 0 then slot 2
//!      doesn't clobber slot 0
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

fn small_params() -> LayerParams {
    // Same shape as the existing `kv_fp8_store_round_trip_within_e4m3
    // _tolerance` test in attn_smoke.rs — `n_lora_kv == head_dim` so
    // the kernel can detect the n_rot tail.
    LayerParams {
        layer_idx: 0,
        d_embd: 32,
        n_hc: 1,
        n_head: 1,
        head_dim: 16,
        n_rot: 4,
        n_lora_q: 1,
        n_lora_kv: 16,
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
fn kv_fp8_store_persistent_round_trip_within_tolerance() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let row = params.n_lora_kv as usize;
    let raw_cap = 4u32;

    let kv: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.31).sin()).collect();
    let buf = disp.kv_fp8_store_persistent(7, &params, &kv, raw_cap, 0);

    let full = read_buf(&buf, (raw_cap as usize) * row);
    let stored_row = &full[..row];

    let n_nope = params.n_lora_kv as usize;
    let n_rot_t = params.n_rot as usize;
    // n_nope = head_dim - n_rot = 16 - 4 = 12 in the kernel's terms,
    // but `params.n_lora_kv` is the full row span here. The kernel
    // computes n_nope = head_dim - n_rot internally.
    let n_nope_eff = (params.head_dim as usize).saturating_sub(n_rot_t);

    // e4m3 region — accept up to 13% relative drift (3-bit mantissa).
    for (i, (&got, &want)) in stored_row[..n_nope_eff]
        .iter()
        .zip(&kv[..n_nope_eff])
        .enumerate()
    {
        let rel = if want.abs() > 1e-3 {
            (got - want).abs() / want.abs()
        } else {
            (got - want).abs()
        };
        assert!(
            rel < 0.13,
            "e4m3 row[{i}]: got={got}, want={want}, rel={rel}"
        );
    }
    // f16 region — tighter (~1e-3 ULP at this magnitude).
    for (i, (&got, &want)) in stored_row[n_nope_eff..n_nope_eff + n_rot_t]
        .iter()
        .zip(&kv[n_nope_eff..n_nope_eff + n_rot_t])
        .enumerate()
    {
        let abs = (got - want).abs();
        assert!(abs < 1e-2, "f16 row[{i}]: got={got}, want={want}, |diff|={abs}");
    }
}

#[test]
fn kv_fp8_store_persistent_slot_isolation() {
    // Writing to slot 0 then slot 2 must NOT clobber slot 0.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let row = params.n_lora_kv as usize;
    let raw_cap = 4u32;
    let total = (raw_cap as usize) * row;

    let kv_a: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.31).sin()).collect();
    let kv_b: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.13).cos() * 0.7).collect();

    let _buf_a = disp.kv_fp8_store_persistent(11, &params, &kv_a, raw_cap, 0);
    let buf_after_b = disp.kv_fp8_store_persistent(11, &params, &kv_b, raw_cap, 2);

    let full = read_buf(&buf_after_b, total);
    // Slot 0 (offset 0..row) should still contain ~kv_a (within e4m3 tol).
    // Slot 1 (row..2*row) should still be all zeros (never written).
    // Slot 2 (2*row..3*row) should contain ~kv_b.
    // Slot 3 (3*row..4*row) should still be zeros.

    // Slot 0 still has kv_a-quantized values (non-zero).
    let slot0 = &full[..row];
    let slot1 = &full[row..2 * row];
    let slot2 = &full[2 * row..3 * row];
    let slot3 = &full[3 * row..4 * row];

    let nz_0 = slot0.iter().filter(|&&v| v != 0.0).count();
    let nz_1 = slot1.iter().filter(|&&v| v != 0.0).count();
    let nz_2 = slot2.iter().filter(|&&v| v != 0.0).count();
    let nz_3 = slot3.iter().filter(|&&v| v != 0.0).count();

    assert!(nz_0 > row / 2, "slot 0 mostly zero after second write ({nz_0}/{row})");
    assert_eq!(nz_1, 0, "slot 1 should still be unwritten ({nz_1} non-zero)");
    assert!(nz_2 > row / 2, "slot 2 should contain kv_b ({nz_2}/{row})");
    assert_eq!(nz_3, 0, "slot 3 should still be unwritten ({nz_3} non-zero)");

    // Slot 0 still resembles kv_a (within e4m3 tolerance over NOPE).
    let n_rot_t = params.n_rot as usize;
    let n_nope_eff = (params.head_dim as usize).saturating_sub(n_rot_t);
    for (i, (&got, &want)) in slot0[..n_nope_eff].iter().zip(&kv_a[..n_nope_eff]).enumerate() {
        let rel = if want.abs() > 1e-3 {
            (got - want).abs() / want.abs()
        } else {
            (got - want).abs()
        };
        assert!(
            rel < 0.13,
            "after second write, slot 0[{i}] drifted: got={got} want={want} rel={rel}"
        );
    }
}

#[test]
fn kv_fp8_store_persistent_distinct_layers() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let row = params.n_lora_kv as usize;
    let raw_cap = 4u32;
    let kv: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.31).sin()).collect();

    let buf_a = disp.kv_fp8_store_persistent(0, &params, &kv, raw_cap, 0);
    let buf_b = disp.kv_fp8_store_persistent(1, &params, &kv, raw_cap, 0);

    use metal::foreign_types::ForeignType;
    assert_ne!(
        buf_a.as_ptr() as usize,
        buf_b.as_ptr() as usize,
        "kv_fp8_store_persistent for distinct layers must use distinct buffers"
    );
}
