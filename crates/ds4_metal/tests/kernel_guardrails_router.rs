//! GUARDRAIL TEST — the MoE ROUTER (`encode_router_logits_k` + `encode_router_finalize`),
//! the DISCRETE AMPLIFIER the chunk-prefill bisection hinges on: a tiny upstream
//! residual drift shifts router logits and FLIPS an expert pick → the FFN output
//! changes by O(1) → incoherence. The guardrails:
//!   • logits K-batch≡K=1 — the K-batched router-logit path is faithful to the
//!     per-token K=1 path (so a selection difference can only come from the INPUT
//!     drift, not the router kernel — consistent with the cross-impl finding).
//!   • finalize matches CPU exactly — softplus_sqrt + top-k + bias selection.
//!   • determinism.
//!   • SENSITIVITY (the headline diagnostic) — perturb the router input h_norm by
//!     ε and count how many of the top-6 experts flip: quantifies the discrete
//!     amplification that turns a broad numeric drift into incoherence.
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_engine::dispatch::KernelDispatcher;
use ds4_metal::MetalDispatcher;

const N_EXP: usize = 256;
const D_EMBD: usize = 4096;
const TOP_K: usize = 6;

fn read_i32(buf: &metal::Buffer, n: usize) -> Vec<i32> {
    let mut out = vec![0i32; n];
    unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const i32, out.as_mut_ptr(), n); }
    out
}

#[test]
fn router_guardrails() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let k = 8usize;
    let w_router = rand_f32(N_EXP * D_EMBD, 0x71);
    let bias = rand_f32(N_EXP, 0x72);
    // K distinct router inputs (one per token).
    let h_k: Vec<f32> = (0..k * D_EMBD)
        .map(|i| { let r = i / D_EMBD; ((i as f32 * 0.013 + r as f32 * 0.5).sin()) * 0.3 }).collect();

    // (2) logits K-batch ≡ K=1, per row (softplus_sqrt probs). matvec reduction ⇒ Close.
    assert_k_batch_equiv(
        "encode_router_logits_k", k, N_EXP, Equiv::Close { rel_tol: 2e-3 },
        || {
            let s = disp.batch_scope();
            let w = s.upload_f32(&w_router);
            let h = s.upload_f32(&h_k);
            let p = s.encode_router_logits_k(&w, &h, N_EXP, D_EMBD, k, false).expect("logits_k");
            s.flush_and_read(&p)
        },
        |r| {
            let s = disp.batch_scope();
            let w = s.upload_f32(&w_router);
            let h = s.upload_f32(&h_k[r * D_EMBD..(r + 1) * D_EMBD]);
            let p = s.encode_router_logits(&w, &h, N_EXP, false).expect("logits");
            s.flush_and_read(&p)
        },
    );

    // Compute row-0 probs once (GPU), reused for finalize tests.
    let probs0 = {
        let s = disp.batch_scope();
        let w = s.upload_f32(&w_router);
        let h = s.upload_f32(&h_k[..D_EMBD]);
        let p = s.encode_router_logits(&w, &h, N_EXP, false).expect("logits");
        s.flush_and_read(&p)
    };

    // (4a) finalize selection matches CPU router_finalize exactly (same probs in).
    let gpu_sel = {
        let s = disp.batch_scope();
        let pb = s.upload_f32(&probs0);
        let bb = s.upload_f32(&bias);
        let (sel, w) = s.encode_router_finalize(&pb, &bb).expect("finalize");
        let sel_c = sel.buffer().clone();
        let _ = s.flush_and_read(&w);
        let mut v = read_i32(&sel_c, TOP_K); v.sort_unstable(); v
    };
    let (cpu_sel, _w) = disp.router_finalize(&probs0, &bias, TOP_K);
    let mut cpu_sel: Vec<i32> = cpu_sel.iter().map(|&i| i as i32).collect();
    cpu_sel.sort_unstable();
    assert_selection_matches("encode_router_finalize", &gpu_sel, &cpu_sel);

    // (1) determinism of the finalize selection (as f32 ids).
    assert_deterministic("encode_router_finalize", 3, || {
        let s = disp.batch_scope();
        let pb = s.upload_f32(&probs0);
        let bb = s.upload_f32(&bias);
        let (sel, w) = s.encode_router_finalize(&pb, &bb).expect("finalize");
        let sel_c = sel.buffer().clone();
        let _ = s.flush_and_read(&w);
        read_i32(&sel_c, TOP_K).into_iter().map(|i| i as f32).collect()
    });

    // (4b) SENSITIVITY SWEEP — perturb the router INPUT by growing ε, recompute
    // logits → CPU finalize, count flipped experts. THIS is the chunk-incoherence
    // amplifier: upstream drift → expert flip. The ONSET ε (where flip% leaves 0)
    // is the drift magnitude that starts breaking routing — small drift (early
    // layers) survives; large drift (deep layers, where the bisection saw flip%
    // climb to 88%) flips. Reported, not asserted (intrinsic to top-k near ties).
    let (base_sel, _w) = disp.router_finalize(&probs0, &bias, TOP_K);
    let base_sel: Vec<i32> = base_sel.iter().map(|&i| i as i32).collect();
    let noise = rand_f32(D_EMBD, 0x73);
    for eps in [1e-3f32, 1e-2, 3e-2, 1e-1, 3e-1] {
        let mut h_pert = h_k[..D_EMBD].to_vec();
        for (x, &n) in h_pert.iter_mut().zip(noise.iter()) { *x += eps * n; }
        let probs_p = {
            let s = disp.batch_scope();
            let w = s.upload_f32(&w_router);
            let h = s.upload_f32(&h_pert);
            let p = s.encode_router_logits(&w, &h, N_EXP, false).expect("logits");
            s.flush_and_read(&p)
        };
        let (sel_p, _w) = disp.router_finalize(&probs_p, &bias, TOP_K);
        let sel_p: Vec<i32> = sel_p.iter().map(|&i| i as i32).collect();
        let flipped = selection_sensitivity(&base_sel, &sel_p);
        eprintln!(
            "[guardrail] MoE router sensitivity: ε={eps:.0e} → {:.1}% of top-{TOP_K} experts flip",
            flipped * 100.0,
        );
    }
    eprintln!(
        "[guardrail] MoE router: ↑ the ε where flip% leaves 0 is the upstream-drift magnitude \
         that starts breaking routing — the chunk-incoherence amplification mechanism."
    );
}
