//! Validates the BATCHED top-k `encode_indexer_topk_batched` (step 2 of the
//! antirez batched-prefill-attention port; wires emitted ds4_argsort_f32_i32_
//! desc) against a per-token CPU oracle. Each token's row must yield the top_k
//! comp indices by DESCENDING score (ties -> lowest index), and -INFINITY
//! (causally-masked) rows must sort to the end.
//!
//! macOS-only.
#![cfg(target_os = "macos")]

use std::collections::BTreeSet;

use ds4_metal::MetalDispatcher;

fn cpu_topk_set(row: &[f32], top_k: usize) -> BTreeSet<i32> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.into_iter().take(top_k).map(|i| i as i32).collect()
}

#[test]
fn indexer_topk_batched_matches_cpu() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_tokens = 24usize;
    let n_comp = 300usize;
    let top_k = 8usize;

    let mut seed = 0xC0FFEEu64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // Per-token scores; mask the tail of each row to -inf (mimics causal mask:
    // token t sees visible=(t+1) comp rows here, so comp >= visible -> -inf).
    let mut scores = vec![0.0f32; n_tokens * n_comp];
    for t in 0..n_tokens {
        let visible = (t + 1).min(n_comp);
        for c in 0..n_comp {
            scores[t * n_comp + c] = if c < visible { rnd() } else { f32::NEG_INFINITY };
        }
    }

    let gpu_sel: Vec<i32> = {
        let scope = disp.batch_scope();
        let s_db = scope.upload_f32(&scores);
        let sel = scope
            .encode_indexer_topk_batched(&s_db, n_comp, n_tokens, top_k)
            .expect("encode_indexer_topk_batched");
        scope.flush_and_read_i32(&sel)
    };

    let mut mism = 0usize;
    for t in 0..n_tokens {
        let row = &scores[t * n_comp..(t + 1) * n_comp];
        let oracle = cpu_topk_set(row, top_k);
        let gpu: BTreeSet<i32> = gpu_sel[t * top_k..(t + 1) * top_k].iter().copied().collect();
        // Among the top_k by score, the visible (finite) ones must match the
        // oracle's visible picks; the fill (-inf, when visible<top_k) is any
        // valid index, so compare the FINITE-score picks as the gate.
        let visible = (t + 1).min(n_comp);
        let oracle_fin: BTreeSet<i32> =
            oracle.iter().copied().filter(|&i| (i as usize) < visible).collect();
        let gpu_fin: BTreeSet<i32> =
            gpu.iter().copied().filter(|&i| (i as usize) < visible).collect();
        if oracle_fin != gpu_fin {
            if mism < 3 {
                eprintln!("token {t} (visible={visible}) FIN mismatch: oracle={oracle_fin:?} gpu={gpu_fin:?}");
            }
            mism += 1;
        }
    }
    eprintln!("[indexer_topk_batched] {} / {n_tokens} tokens mismatched", mism);
    assert_eq!(mism, 0, "batched top_k diverges from CPU oracle on {mism} tokens");
}

/// n_comp > 1024 exercises the threshold-select batched path (the argsort can't
/// fit). Same CPU-oracle gate. This is the long-context single-chunk prefill
/// regime (ctx>4096 → n_comp = ctx/4 > 1024).
#[test]
fn indexer_topk_batched_above_1024_matches_cpu() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_tokens = 16usize;
    let n_comp = 2048usize; // > 1024 → threshold-select path
    let top_k = 64usize;

    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // Spread visibility across the rows so some tokens have visible < top_k
    // (forcing the fill path) and some visible >> top_k.
    let mut scores = vec![0.0f32; n_tokens * n_comp];
    for t in 0..n_tokens {
        let visible = ((t + 1) * n_comp / n_tokens).clamp(1, n_comp);
        for c in 0..n_comp {
            scores[t * n_comp + c] = if c < visible { rnd() } else { f32::NEG_INFINITY };
        }
    }

    let gpu_sel: Vec<i32> = {
        let scope = disp.batch_scope();
        let s_db = scope.upload_f32(&scores);
        let sel = scope
            .encode_indexer_topk_batched(&s_db, n_comp, n_tokens, top_k)
            .expect("encode_indexer_topk_batched (>1024)");
        scope.flush_and_read_i32(&sel)
    };

    let mut mism = 0usize;
    for t in 0..n_tokens {
        let row = &scores[t * n_comp..(t + 1) * n_comp];
        let oracle = cpu_topk_set(row, top_k);
        let gpu: BTreeSet<i32> = gpu_sel[t * top_k..(t + 1) * top_k]
            .iter()
            .copied()
            .filter(|&i| i >= 0)
            .collect();
        let visible = ((t + 1) * n_comp / n_tokens).clamp(1, n_comp);
        let oracle_fin: BTreeSet<i32> =
            oracle.iter().copied().filter(|&i| (i as usize) < visible).collect();
        let gpu_fin: BTreeSet<i32> =
            gpu.iter().copied().filter(|&i| (i as usize) < visible).collect();
        if oracle_fin != gpu_fin {
            if mism < 3 {
                eprintln!("token {t} (visible={visible}) FIN mismatch: oracle={oracle_fin:?} gpu={gpu_fin:?}");
            }
            mism += 1;
        }
    }
    eprintln!("[indexer_topk_batched>1024] {mism} / {n_tokens} tokens mismatched");
    assert_eq!(mism, 0, "threshold-batched top_k diverges from CPU oracle on {mism} tokens");
}
