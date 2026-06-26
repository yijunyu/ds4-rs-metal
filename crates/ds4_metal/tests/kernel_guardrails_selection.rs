//! GUARDRAIL TEST — `encode_indexer_topk_batched` (the discrete top-k SELECTION
//! that picks which compressed rows attention attends). Discrete selection is
//! the AMPLIFIER in the chunk "@3000" bisection: a tiny upstream score drift
//! flips a pick and the attended set changes discontinuously. These guardrails
//! ensure the selection kernel is itself exact + deterministic, and QUANTIFY its
//! sensitivity (how readily it flips) so it's flagged as a prime suspect when a
//! downstream-coherence bug appears.
//!
//! Guardrails:
//!   • selection matches CPU top-k exactly (per token)
//!   • determinism (same scores ⇒ same picks every run)
//!   • sensitivity diagnostic (ε score perturbation → flipped-fraction; reported)
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_metal::MetalDispatcher;

/// CPU top-k indices by descending score (ties → lowest index), ascending order.
fn cpu_topk(row: &[f32], top_k: usize) -> Vec<i32> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let mut picks: Vec<i32> = order.into_iter().take(top_k).map(|i| i as i32).collect();
    picks.sort_unstable();
    picks
}

fn gpu_topk(disp: &MetalDispatcher, scores: &[f32], n_comp: usize, n_tokens: usize, top_k: usize) -> Vec<i32> {
    let s = disp.batch_scope();
    let s_db = s.upload_f32(scores);
    let sel = s.encode_indexer_topk_batched(&s_db, n_comp, n_tokens, top_k).expect("topk");
    s.flush_and_read_i32(&sel)
}

#[test]
fn indexer_topk_selection_guardrails() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let (n_tokens, n_comp, top_k) = (16usize, 256usize, 8usize);
    // Well-separated scores (clear top_k per token) → selection must be EXACT.
    let scores = rand_f32(n_tokens * n_comp, 0xBEEF);

    // (4a) GPU selection matches CPU top-k exactly, per token.
    let gpu = gpu_topk(&disp, &scores, n_comp, n_tokens, top_k);
    for t in 0..n_tokens {
        let mut got: Vec<i32> = gpu[t * top_k..(t + 1) * top_k].to_vec();
        got.sort_unstable();
        let want = cpu_topk(&scores[t * n_comp..(t + 1) * n_comp], top_k);
        assert_selection_matches(&format!("indexer_topk t{t}"), &got, &want);
    }

    // (1) Determinism — same scores ⇒ identical picks (as f32-cast ids) every run.
    assert_deterministic("indexer_topk_batched", 3, || {
        gpu_topk(&disp, &scores, n_comp, n_tokens, top_k)
            .into_iter().map(|i| i as f32).collect()
    });

    // (4b) Sensitivity DIAGNOSTIC: perturb scores by a tiny ε and report how many
    // picks flip. High flip-fraction ⇒ this op amplifies upstream drift — exactly
    // the mechanism that turned chunk K-batch non-associativity into incoherence.
    let eps = 1e-3f32;
    let mut pert = scores.clone();
    let noise = rand_f32(pert.len(), 0x5151);
    for (p, &n) in pert.iter_mut().zip(noise.iter()) { *p += eps * n; }
    let gpu_pert = gpu_topk(&disp, &pert, n_comp, n_tokens, top_k);
    let flipped = selection_sensitivity(&gpu, &gpu_pert);
    eprintln!(
        "[guardrail] indexer_topk_batched: ε={eps:.0e} perturbation flips {:.1}% of picks \
         (diagnostic — high ⇒ amplifies upstream drift)", flipped * 100.0,
    );
    // Not an assertion (sensitivity is intrinsic to top-k near ties); surfaced so
    // a reviewer weighs it when this kernel sits upstream of a coherence bug.
}
