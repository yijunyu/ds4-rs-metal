//! M4 #330p Phase C-B Slice 2 — `attn_qkv_chain_batched` smoke test.
//!
//! `attn_qkv_chain_batched` (defined in `crates/ds4_metal/src/deferred.rs`)
//! fuses the attention-half QKV chain into ONE `MTLCommandBuffer` via
//! `BatchScope`:
//!
//! ```text
//! matvec(attn_q_a) → rms_norm(gamma_q) → matvec(attn_q_b) →
//!     head_rms_norm → matvec(attn_kv)
//! ```
//!
//! It replaces the prior pair of `layer_qa_rms_batched`
//! (1 cb: matvec + rms_norm) + `qkv_b_head_rms_batched`
//! (1 cb: matvec + head_rms + matvec). This test asserts the fused
//! output is BIT-IDENTICAL to running those two ops sequentially —
//! same MSL kernels, same FC keys, same args. The chain has no
//! transcendentals (no `exp`), so bit-identity is the right bar.
//!
//! macOS-only — needs a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(
    d_embd: usize,
    n_lora_q: usize,
    q_dim: usize,
    kv_row: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let normed: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
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
    (normed, attn_q_a, gamma_q, attn_q_b, attn_kv)
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
fn attn_qkv_chain_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");

    let d_embd = 256;
    let n_lora_q = 64;
    let n_head = 4;
    let head_dim = 32;
    let q_dim = n_head * head_dim;
    let kv_row = 64;
    let eps = 1e-5_f32;

    let (normed, attn_q_a, gamma_q, attn_q_b, attn_kv) =
        build_inputs(d_embd, n_lora_q, q_dim, kv_row);

    // Fused path: one cb.
    let (qr_normed_b, q_heads_b, kv_raw_row_b) = disp
        .attn_qkv_chain_batched(
            &normed,
            &attn_q_a,
            &gamma_q,
            n_lora_q,
            &attn_q_b,
            n_head,
            head_dim,
            eps,
            &attn_kv,
            kv_row,
        )
        .expect("attn_qkv_chain_batched");

    // Sequential reference: two cbs via the existing _batched ops.
    let qr_normed_ref =
        disp.layer_qa_rms_batched(&normed, &attn_q_a, &gamma_q, n_lora_q, eps);
    let (q_heads_ref, kv_raw_row_ref) = disp.qkv_b_head_rms_batched(
        &qr_normed_ref,
        &attn_q_b,
        q_dim,
        n_head,
        head_dim,
        eps,
        &normed,
        &attn_kv,
        kv_row,
    );

    assert_bit_identical("qr_normed", &qr_normed_b, &qr_normed_ref);
    assert_bit_identical("q_heads", &q_heads_b, &q_heads_ref);
    assert_bit_identical("kv_raw_row", &kv_raw_row_b, &kv_raw_row_ref);
}

#[test]
fn attn_qkv_chain_matches_sequential_ds4_shape() {
    // Scaled-down DS4 V4 Flash production shape: d_embd=7168, n_lora_q=1536,
    // q_dim=4096 (n_head=128, head_dim=128 → here scaled to /4), n_lora_kv=576.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");

    let d_embd = 512;
    let n_lora_q = 128;
    let n_head = 8;
    let head_dim = 64;
    let q_dim = n_head * head_dim; // 512
    let kv_row = 128;
    let eps = 1e-5_f32;

    let (normed, attn_q_a, gamma_q, attn_q_b, attn_kv) =
        build_inputs(d_embd, n_lora_q, q_dim, kv_row);

    let (qr_normed_b, q_heads_b, kv_raw_row_b) = disp
        .attn_qkv_chain_batched(
            &normed,
            &attn_q_a,
            &gamma_q,
            n_lora_q,
            &attn_q_b,
            n_head,
            head_dim,
            eps,
            &attn_kv,
            kv_row,
        )
        .expect("attn_qkv_chain_batched");

    let qr_normed_ref =
        disp.layer_qa_rms_batched(&normed, &attn_q_a, &gamma_q, n_lora_q, eps);
    let (q_heads_ref, kv_raw_row_ref) = disp.qkv_b_head_rms_batched(
        &qr_normed_ref,
        &attn_q_b,
        q_dim,
        n_head,
        head_dim,
        eps,
        &normed,
        &attn_kv,
        kv_row,
    );

    assert_bit_identical("qr_normed (ds4 shape)", &qr_normed_b, &qr_normed_ref);
    assert_bit_identical("q_heads (ds4 shape)", &q_heads_b, &q_heads_ref);
    assert_bit_identical("kv_raw_row (ds4 shape)", &kv_raw_row_b, &kv_raw_row_ref);
}

/// Locks against silent-zero failure modes: at least one output must
/// have non-trivial signal.
#[test]
fn attn_qkv_chain_produces_nonzero_output() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");

    let d_embd = 256;
    let n_lora_q = 64;
    let n_head = 4;
    let head_dim = 32;
    let q_dim = n_head * head_dim;
    let kv_row = 64;
    let eps = 1e-5_f32;

    let (normed, attn_q_a, gamma_q, attn_q_b, attn_kv) =
        build_inputs(d_embd, n_lora_q, q_dim, kv_row);

    let (qr_normed, q_heads, kv_raw_row) = disp
        .attn_qkv_chain_batched(
            &normed,
            &attn_q_a,
            &gamma_q,
            n_lora_q,
            &attn_q_b,
            n_head,
            head_dim,
            eps,
            &attn_kv,
            kv_row,
        )
        .expect("attn_qkv_chain_batched");

    let count_nonzero = |v: &[f32]| v.iter().filter(|&&x| x != 0.0).count();
    assert!(
        count_nonzero(&qr_normed) > qr_normed.len() / 2,
        "qr_normed mostly zero ({}/{})",
        count_nonzero(&qr_normed),
        qr_normed.len()
    );
    assert!(
        count_nonzero(&q_heads) > q_heads.len() / 2,
        "q_heads mostly zero ({}/{})",
        count_nonzero(&q_heads),
        q_heads.len()
    );
    assert!(
        count_nonzero(&kv_raw_row) > kv_raw_row.len() / 2,
        "kv_raw_row mostly zero ({}/{})",
        count_nonzero(&kv_raw_row),
        kv_raw_row.len()
    );
}
