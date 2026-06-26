//! GUARDRAIL TEST — `flash_attn_k_mla_comp_masked` (the noidx chunk SWA flash:
//! online-softmax over [raw window | comp rows] + per-head sink, per-query
//! additive f16 mask). The chunk-prefill bisection localized the first broad
//! residual drift to the noidx core that uses THIS kernel — so it's a prime
//! suspect. Guardrails (synthetic, no model):
//!   • K-batch≡K=1   — query r batched (K=8) ≡ query r run alone (K=1); each
//!     query is an independent masked flash, so the batch must not perturb it.
//!   • determinism   — same inputs ⇒ byte-identical.
//!   • oracle-close at REFERENCE precision — vs a CPU online-softmax oracle with
//!     KV/q f16-rounded (the kernel casts KV to half internally). LESSON: the
//!     oracle is built at the kernel's ACTUAL precision (f16), NOT idealised f32
//!     — testing against an f32 oracle here would flag a non-bug and cost runs.
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_metal::MetalDispatcher;

const NEG: u16 = 0xFC00; // f16 -inf (mask: skip); 0x0000 = attend.

/// CPU online-softmax oracle for one (query, head), KV+q at f16 reference
/// precision. `kv` rows = [raw window (n_cache) | comp (n_comp)], dk each; MLA
/// shares K=V so the attended row is also the value. Additive mask: 0 attend,
/// else skip. Per-head sink (score only, no value).
fn oracle_row(
    q: &[f32], kv: &[f32], mask_row: &[u16], n_rows: usize, dk: usize, scale: f32, sink: f32,
) -> Vec<f32> {
    let qh: Vec<f32> = q.iter().map(|&x| f16_round(x)).collect();
    let (mut m, mut s) = (f32::MIN / 2.0, 0.0f32);
    let mut o = vec![0.0f32; dk];
    let attend = |k: &[f32], m: &mut f32, s: &mut f32, o: &mut [f32]| {
        let score: f32 = qh.iter().zip(k).map(|(&a, &b)| a * f16_round(b)).sum::<f32>() * scale;
        let nm = m.max(score);
        let (os, rs) = ((*m - nm).exp(), (score - nm).exp());
        *s = *s * os + rs;
        for (oi, &ki) in o.iter_mut().zip(k) { *oi = *oi * os + f16_round(ki) * rs; }
        *m = nm;
    };
    for r in 0..n_rows {
        if mask_row[r] == 0 {
            attend(&kv[r * dk..(r + 1) * dk], &mut m, &mut s, &mut o);
        }
    }
    // sink: score only, no value.
    let nm = m.max(sink);
    let (os, rs) = ((m - nm).exp(), (sink - nm).exp());
    s = s * os + rs;
    for oi in o.iter_mut() { *oi *= os; }
    let inv = if s == 0.0 { 0.0 } else { 1.0 / s };
    o.iter().map(|&v| v * inv).collect()
}

#[test]
fn flash_attn_k_mla_comp_masked_guardrails() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let (n_head, dk, n_cache, n_comp, k) = (8usize, 512usize, 8usize, 4usize, 8usize);
    let n_rows = n_cache + n_comp;
    let scale = 1.0f32 / (dk as f32).sqrt();

    let q = rand_f32(k * n_head * dk, 0x11);
    let raw = rand_f32(n_cache * dk, 0x22);
    let comp = rand_f32(n_comp * dk, 0x33);
    let sinks = rand_f32(n_head, 0x44);
    // Per-query mask: a sliding causal-ish window — query t attends raw cols
    // [0..=min(t,n_cache-1)] and comp cols [0..=t/2], rest -inf. Guarantees ≥1
    // attended col per query and exercises the additive-mask path.
    let mut mask = vec![NEG; k * n_rows];
    for t in 0..k {
        for c in 0..=(t.min(n_cache - 1)) { mask[t * n_rows + c] = 0; }
        let cvis = (t / 2 + 1).min(n_comp);
        for c in 0..cvis { mask[t * n_rows + n_cache + c] = 0; }
    }

    let run = |k_pos: usize, q_in: &[f32], mask_in: &[u16]| -> Vec<f32> {
        let s = disp.batch_scope();
        let q_db = s.upload_f32(q_in);
        let kv_db = s.upload_f32(&raw);
        let comp_db = s.upload_f32(&comp);
        let sink_db = s.upload_f32(&sinks);
        let out = s.flash_attn_k_mla_comp_masked(
            &q_db, &kv_db, comp_db.buffer(), n_comp as u32, mask_in,
            n_head, dk, dk, n_cache, k_pos, scale, &sink_db,
        ).expect("flash_attn_k_mla_comp_masked");
        s.flush_and_read(&out)
    };

    // (3) Oracle-close at REFERENCE precision (KV/q f16-rounded). Structural:
    // correct rows attended, masking, online softmax, sink. f16 dot ⇒ loose tol.
    let gpu = run(k, &q, &mask);
    let mut oracle = vec![0.0f32; k * n_head * dk];
    for t in 0..k {
        for h in 0..n_head {
            // gather this (t,h)'s KV rows: raw window then comp, dk each.
            let mut kv = Vec::with_capacity(n_rows * dk);
            kv.extend_from_slice(&raw); // n_cache rows
            kv.extend_from_slice(&comp); // n_comp rows
            let qth = &q[(t * n_head + h) * dk..(t * n_head + h) * dk + dk];
            let mrow = &mask[t * n_rows..(t + 1) * n_rows];
            let o = oracle_row(qth, &kv, mrow, n_rows, dk, scale, sinks[h]);
            oracle[(t * n_head + h) * dk..(t * n_head + h) * dk + dk].copy_from_slice(&o);
        }
    }
    // Tight gate: measured rel_L2 ≈ 1.8e-4 vs the f16-reference oracle (an f32
    // oracle would show a spurious ~1e-2 f16 gap — match the kernel's precision).
    assert_close_to_oracle("flash_attn_k_mla_comp_masked", &gpu, &oracle, 3e-3);

    // (2) K-batch ≡ K=1: query r of the K=8 batch ≡ that query run alone (K=1).
    let row_dim = n_head * dk;
    assert_k_batch_equiv(
        "flash_attn_k_mla_comp_masked",
        k, row_dim, Equiv::Close { rel_tol: 1e-3 },
        || gpu.clone(),
        |r| {
            let qr = q[r * row_dim..(r + 1) * row_dim].to_vec();
            let mr = mask[r * n_rows..(r + 1) * n_rows].to_vec();
            run(1, &qr, &mr)
        },
    );

    // (1) Determinism.
    assert_deterministic("flash_attn_k_mla_comp_masked", 3, || run(k, &q, &mask));
}
