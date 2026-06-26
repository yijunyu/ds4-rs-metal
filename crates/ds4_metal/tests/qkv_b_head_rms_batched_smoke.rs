//! M4 #330o — Phase C.3d.2 batched Q/K/V split chain smoke tests.
//!
//! `qkv_b_head_rms_batched` packs:
//!   q_heads_raw = matvec_f32(w_q_b,  qr_normed_q,  q_dim)
//!   q_heads     = head_rms_norm(q_heads_raw, n_head, head_dim, eps)
//!   kv_raw_row  = matvec_f32(w_kv,   normed_kv,    kv_row)
//! into ONE `MTLCommandBuffer`. These tests verify bit-identical
//! `(q_heads, kv_raw_row)` against running the three ops sequentially
//! through the same dispatcher. Saves 2 commit+wait+readback per layer
//! once C.3d.3 wires it into `decode_step_with_attn_to_residual`.
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_q_inputs(q_dim: usize, d_qb: usize) -> (Vec<f32>, Vec<f32>) {
    let qr: Vec<f32> = (0..d_qb)
        .map(|i| ((i as f32 * 0.017).sin() * 0.42 + 0.03).clamp(-2.0, 2.0))
        .collect();
    let w_q_b: Vec<f32> = (0..q_dim * d_qb)
        .map(|i| ((i as f32 * 0.0053).cos() * 0.21))
        .collect();
    (qr, w_q_b)
}

fn build_kv_inputs(kv_row: usize, d_kv: usize) -> (Vec<f32>, Vec<f32>) {
    let normed_kv: Vec<f32> = (0..d_kv)
        .map(|i| ((i as f32 * 0.029).cos() * 0.37 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let w_kv: Vec<f32> = (0..kv_row * d_kv)
        .map(|i| ((i as f32 * 0.0083).sin() * 0.17))
        .collect();
    (normed_kv, w_kv)
}

fn reference(
    disp: &MetalDispatcher,
    qr_normed_q: &[f32],
    w_q_b: &[f32],
    q_dim: usize,
    n_head: usize,
    head_dim: usize,
    eps_rms: f32,
    normed_kv: &[f32],
    w_kv: &[f32],
    kv_row: usize,
) -> (Vec<f32>, Vec<f32>) {
    use ds4_engine::dispatch::KernelDispatcher;
    let q_raw = disp.matvec_f32(w_q_b, qr_normed_q, q_dim);
    let q_heads = disp.head_rms_norm(&q_raw, n_head, head_dim, eps_rms);
    let kv_raw_row = disp.matvec_f32(w_kv, normed_kv, kv_row);
    (q_heads, kv_raw_row)
}

#[test]
fn qkv_b_head_rms_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_head = 4usize;
    let head_dim = 16usize;
    let q_dim = n_head * head_dim;
    let d_qb = 32usize;
    let kv_row = 32usize;
    let d_kv = 32usize;
    let eps = 1e-5_f32;
    let (qr, w_q_b) = build_q_inputs(q_dim, d_qb);
    let (normed_kv, w_kv) = build_kv_inputs(kv_row, d_kv);

    let (q_h, kv_r) = disp.qkv_b_head_rms_batched(
        &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv, kv_row,
    );
    let (q_h_ref, kv_r_ref) = reference(
        &disp, &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv, kv_row,
    );

    assert_eq!(q_h.len(), q_h_ref.len(), "q_heads len mismatch");
    for (i, (b, r)) in q_h.iter().zip(&q_h_ref).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "q_heads diverged at i={i}: batched={b} reference={r}"
        );
    }
    assert_eq!(kv_r.len(), kv_r_ref.len(), "kv_raw_row len mismatch");
    for (i, (b, r)) in kv_r.iter().zip(&kv_r_ref).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "kv_raw_row diverged at i={i}: batched={b} reference={r}"
        );
    }
}

#[test]
fn qkv_b_head_rms_matches_sequential_ds4_shape() {
    // Scaled-down DS4 V4 Flash layer-0 shape: ds4 production has
    //   n_head=128, head_dim=128 → q_dim=16384; n_lora_q=1536 (d_qb);
    //   n_lora_kv=512 (kv_row); d_embd=7168 (d_kv).
    // Use a layout-faithful but compact variant so the test stays fast.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_head = 8usize;
    let head_dim = 64usize;
    let q_dim = n_head * head_dim;
    let d_qb = 128usize;
    let kv_row = 64usize;
    let d_kv = 256usize;
    let eps = 1e-5_f32;
    let (qr, w_q_b) = build_q_inputs(q_dim, d_qb);
    let (normed_kv, w_kv) = build_kv_inputs(kv_row, d_kv);

    let (q_h, kv_r) = disp.qkv_b_head_rms_batched(
        &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv, kv_row,
    );
    let (q_h_ref, kv_r_ref) = reference(
        &disp, &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv, kv_row,
    );

    for (i, (b, r)) in q_h.iter().zip(&q_h_ref).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "q_heads ds4 shape diverged at i={i}: batched={b} reference={r}"
        );
    }
    for (i, (b, r)) in kv_r.iter().zip(&kv_r_ref).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "kv_raw_row ds4 shape diverged at i={i}: batched={b} reference={r}"
        );
    }
}

/// Independence canary: swapping `w_kv` after a first call MUST change
/// `kv_raw_row` but leave `q_heads` byte-identical. Locks against silent
/// argument crossover (a future regression that fed `qr_normed_q` to the
/// KV-side matvec would still produce valid floats).
#[test]
fn qkv_b_head_rms_q_and_kv_are_independent() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_head = 4usize;
    let head_dim = 16usize;
    let q_dim = n_head * head_dim;
    let d_qb = 32usize;
    let kv_row = 32usize;
    let d_kv = 32usize;
    let eps = 1e-5_f32;
    let (qr, w_q_b) = build_q_inputs(q_dim, d_qb);
    let (normed_kv, w_kv_a) = build_kv_inputs(kv_row, d_kv);
    let w_kv_b: Vec<f32> = (0..kv_row * d_kv)
        .map(|i| ((i as f32 * 0.0119).cos() * 0.34))
        .collect();

    let (q_h_a, kv_a) = disp.qkv_b_head_rms_batched(
        &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv_a, kv_row,
    );
    let (q_h_b, kv_b) = disp.qkv_b_head_rms_batched(
        &qr, &w_q_b, q_dim, n_head, head_dim, eps,
        &normed_kv, &w_kv_b, kv_row,
    );

    for (i, (a, b)) in q_h_a.iter().zip(&q_h_b).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "q_heads must not depend on w_kv: i={i} a={a} b={b}"
        );
    }
    let max_kv_diff: f32 = kv_a
        .iter()
        .zip(&kv_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_kv_diff > 1e-6,
        "different w_kv produced bit-identical kv_raw_row — KV-side matvec likely skipped (max |diff| = {max_kv_diff})"
    );
}
