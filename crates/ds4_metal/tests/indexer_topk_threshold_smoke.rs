//! Parity smoke for the parallel threshold-select top-k
//! (`ds4_dsv4_indexer_topk_threshold`) against the greedy
//! (`ds4_dsv4_indexer_topk_greedy`) and a CPU oracle.
//!
//! The threshold kernel must select the SAME SET as the greedy — the top_k
//! rows by score, ties broken by LOWEST index — but via ~32 parallel-count
//! passes instead of top_k sequential rounds. Order within the output is
//! irrelevant (the flash gather is a permutation-invariant softmax), so we
//! compare SETS. This is the @3000 decode-wall fix (greedy was ~27 ms/token).
//!
//! macOS-only.
#![cfg(target_os = "macos")]

use std::collections::BTreeSet;

use ds4_metal::MetalDispatcher;

/// CPU oracle: indices of the top_k scores, ties → lowest index (stable by
/// descending score then ascending index), returned as a set.
fn cpu_topk_set(scores: &[f32], top_k: usize) -> BTreeSet<i32> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b)) // tie → lower index first
    });
    order.truncate(top_k);
    order.into_iter().map(|i| i as i32).collect()
}

fn run_case(disp: &MetalDispatcher, n_comp: usize, top_k: usize, seed: u64) {
    // Pseudo-random scores in a mix of signs/magnitudes (the real indexer
    // scores are signed dot products).
    let scores: Vec<f32> = (0..n_comp)
        .map(|i| {
            let x = (i as u64).wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503));
            let f = ((x >> 11) as f32) / (1u64 << 21) as f32; // [0,1)
            (f - 0.5) * 6.0 // [-3, 3)
        })
        .collect();

    let oracle = cpu_topk_set(&scores, top_k);

    // GPU greedy (mutates its scores scratch → upload a private copy).
    let greedy: BTreeSet<i32> = {
        let scope = disp.batch_scope();
        let s_db = scope.upload_f32(&scores);
        let sel = scope
            .encode_indexer_topk(&s_db, n_comp, top_k)
            .expect("greedy topk");
        scope.flush_and_read_i32(&sel).into_iter().collect()
    };

    // GPU threshold-select (read-only scores).
    let threshold: BTreeSet<i32> = {
        let scope = disp.batch_scope();
        let s_db = scope.upload_f32(&scores);
        let sel = scope
            .encode_indexer_topk_threshold(&s_db, n_comp, top_k)
            .expect("threshold topk");
        scope.flush_and_read_i32(&sel).into_iter().collect()
    };

    assert_eq!(
        threshold.len(),
        top_k,
        "threshold selected {} != top_k {} (n_comp={n_comp} seed={seed})",
        threshold.len(),
        top_k
    );
    assert_eq!(
        threshold, oracle,
        "threshold set != CPU oracle (n_comp={n_comp} top_k={top_k} seed={seed})"
    );
    assert_eq!(
        threshold, greedy,
        "threshold set != GPU greedy set (n_comp={n_comp} top_k={top_k} seed={seed})"
    );
}

#[test]
fn threshold_topk_matches_greedy_production_dims() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Production decode regime: top_k=512, n_comp just past it (long context).
    for seed in 0..6 {
        run_case(&disp, 600, 512, seed);
        run_case(&disp, 770, 512, seed);
        run_case(&disp, 2048, 512, seed);
    }
}

#[test]
fn threshold_topk_edge_cases() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    run_case(&disp, 513, 512, 1); // top_k == n_comp - 1
    run_case(&disp, 100, 1, 2); // top_k == 1
    run_case(&disp, 256, 256, 3); // top_k == n_comp (all)
    run_case(&disp, 1000, 999, 4);
    run_case(&disp, 33, 16, 5); // sub-threadgroup n_comp
}

#[test]
fn threshold_topk_handles_ties() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Many equal scores → the threshold == Kt tie-fill (lowest index) must
    // match the greedy's lowest-index tie-break exactly.
    let n_comp = 800usize;
    let top_k = 512usize;
    let scores: Vec<f32> = (0..n_comp)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }) // heavy ties at 1.0 and 0.0
        .collect();
    let oracle = cpu_topk_set(&scores, top_k);
    let threshold: BTreeSet<i32> = {
        let scope = disp.batch_scope();
        let s_db = scope.upload_f32(&scores);
        let sel = scope
            .encode_indexer_topk_threshold(&s_db, n_comp, top_k)
            .expect("threshold topk ties");
        scope.flush_and_read_i32(&sel).into_iter().collect()
    };
    assert_eq!(threshold.len(), top_k);
    assert_eq!(threshold, oracle, "tie-break mismatch vs CPU oracle");
}
