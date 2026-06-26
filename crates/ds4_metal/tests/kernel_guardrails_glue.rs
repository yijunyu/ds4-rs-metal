//! GUARDRAIL TEST — `hc_collapse_norm_k` (the K-batched FFN-half residual glue
//! that produces the ROUTER INPUT). The chunk-prefill bisection cleared the
//! attention kernels (matmul_k, flash) and localized the broad per-token drift
//! UPSTREAM, in the K-batched glue feeding the router. This kernel's default
//! fused path uses a BATCHED `matvec_f32_k` (ne11=K) whose bit-identity to the
//! per-token K=1 `hc_collapse_norm` is ASSERTED IN A COMMENT but never tested —
//! so this guardrail verifies the claim (a batched matvec can reduce in a
//! different order ⇒ fp32 non-associative drift the router then amplifies).
//!
//! Guardrails:
//!   • K-batch (fused) ≡ K=1 `hc_collapse_norm`, per row, on the `normed` output
//!     (the router input). Close{tol} reports the actual worst_rel — 0 ⇒ the
//!     "bit-identical" comment holds; ~1e-6 ⇒ a non-assoc drift (small; not the
//!     cos-0.34 broad-drift source, but a regression beyond tol would be caught).
//!   • determinism.
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_metal::MetalDispatcher;

const N_HC: usize = 4;
const N_EMBD: usize = 4096;
const MIX_HC: usize = 2 * N_HC + N_HC * N_HC; // 24
const SINKHORN: i32 = 20;
const HC_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

// shared (K-invariant) weights
fn hc_fn() -> Vec<f32> { (0..N_HC * N_EMBD * MIX_HC).map(|i| (i as f32 * 0.0011).sin() * 0.05).collect() }
fn hc_scale() -> Vec<f32> { vec![1.0, 0.5, 2.0] }
fn hc_base() -> Vec<f32> { (0..MIX_HC).map(|i| 0.1 + (i as f32) * 0.01).collect() }
fn hc_gamma() -> Vec<f32> { (0..N_EMBD).map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect() }

#[test]
fn hc_collapse_norm_k_guardrails() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let k = 8usize;
    let hc_dim = N_HC * N_EMBD;
    // K DISTINCT prev_hc rows (different phase per row → distinct router inputs).
    let prev_k: Vec<f32> = (0..k * hc_dim)
        .map(|i| {
            let row = i / hc_dim;
            let j = i % hc_dim;
            ((j as f32 * 0.017 + row as f32 * 0.3).sin() * 0.4 + 0.05).clamp(-2.0, 2.0)
        })
        .collect();
    let (fnw, scl, base, gam) = (hc_fn(), hc_scale(), hc_base(), hc_gamma());
    let unit = vec![1.0f32; hc_dim];

    // batched (default fused path) — return the `normed` output [K, n_embd].
    let batched_normed = || {
        let s = disp.batch_scope();
        let p = s.upload_f32(&prev_k);
        let (fb, sb, bb, gb, ub) = (
            s.upload_f32(&fnw), s.upload_f32(&scl), s.upload_f32(&base),
            s.upload_f32(&gam), s.upload_f32(&unit),
        );
        let (_split, _cur, normed) = s
            .hc_collapse_norm_k(&p, &fb, &sb, &bb, &gb, N_HC, N_EMBD, SINKHORN, HC_EPS, RMS_EPS, &ub, k, false)
            .expect("hc_collapse_norm_k");
        s.flush_and_read(&normed)
    };

    // K=1 reference: per-token `hc_collapse_norm` on row r → `normed` [n_embd].
    let per_row = |r: usize| {
        let s = disp.batch_scope();
        let p = s.upload_f32(&prev_k[r * hc_dim..(r + 1) * hc_dim]);
        let (fb, sb, bb, gb, ub) = (
            s.upload_f32(&fnw), s.upload_f32(&scl), s.upload_f32(&base),
            s.upload_f32(&gam), s.upload_f32(&unit),
        );
        let (_split, _cur, normed) = s
            .hc_collapse_norm(&p, &fb, &sb, &bb, &gb, N_HC, N_EMBD, SINKHORN, HC_EPS, RMS_EPS, &ub, false)
            .expect("hc_collapse_norm");
        s.flush_and_read(&normed)
    };

    // (2) K-batch (fused) ≡ K=1 per row. Tol catches a real divergence; the
    // reported worst_rel reveals whether the "bit-identical" claim truly holds.
    assert_k_batch_equiv(
        "hc_collapse_norm_k.normed",
        // measured worst_rel ≈ 4e-7 (fused batched matvec_f32_k vs K=1 matvec):
        // NOT bit-identical (the kernel's comment overclaims) but fp32-epsilon —
        // ~6 orders too small to be the cos-0.34 broad drift. 1e-5 = 25× headroom.
        k, N_EMBD, Equiv::Close { rel_tol: 1e-5 },
        batched_normed, per_row,
    );

    // (1) Determinism.
    assert_deterministic("hc_collapse_norm_k", 3, batched_normed);
}
