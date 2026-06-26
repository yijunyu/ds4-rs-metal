//! Validates the BATCHED indexer-scores kernel `encode_indexer_scores_tiled`
//! (port of antirez `kernel_dsv4_indexer_scores_tiled_f32`) against a CPU
//! oracle. Step 1 of the antirez batched-prefill-attention port: replaces the
//! per-position `encode_indexer_scores` loop with one dispatch producing
//! scores[n_tokens][n_comp], with the causal mask (comp >= (pos0+token+1)/ratio
//! -> -INFINITY) baked in.
//!
//! macOS-only.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

#[test]
fn indexer_scores_tiled_matches_cpu_oracle() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_tokens = 20usize;
    let n_comp = 37usize;
    let n_head = 64usize;
    let head_dim = 128usize;
    let pos0 = 0u32;
    let ratio = 4u32;
    let scale = 1.0f32 / ((head_dim * n_head) as f32).sqrt();

    // Deterministic pseudo-random fill in [-1, 1).
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let q_k: Vec<f32> = (0..n_tokens * n_head * head_dim).map(|_| rnd()).collect();
    let weights_k: Vec<f32> = (0..n_tokens * n_head).map(|_| rnd()).collect();
    let index_ring: Vec<f32> = (0..n_comp * head_dim).map(|_| rnd()).collect();

    // GPU batched scores.
    let gpu: Vec<f32> = {
        let scope = disp.batch_scope();
        let q_db = scope.upload_f32(&q_k);
        let w_db = scope.upload_f32(&weights_k);
        let r_db = scope.upload_f32(&index_ring);
        let s = scope
            .encode_indexer_scores_tiled(
                &q_db, &w_db, &r_db, n_comp, n_tokens, n_head, head_dim, pos0, ratio,
            )
            .expect("encode_indexer_scores_tiled");
        scope.flush_and_read(&s)
    };

    // CPU oracle: scores[token][c] = (c < visible)
    //   ? sum_h max(0, dot(q[token,h], index_ring[c])) * weights[token,h] * scale
    //   : -INFINITY ; visible = min((pos0+token+1)/ratio, n_comp).
    let mut worst_rel = 0.0f64;
    let mut n_masked_ok = 0usize;
    let mut n_finite = 0usize;
    for t in 0..n_tokens {
        let visible = (((pos0 as usize + t + 1) / ratio as usize).min(n_comp)) as usize;
        for c in 0..n_comp {
            let g = gpu[t * n_comp + c];
            if c >= visible {
                assert!(
                    g.is_infinite() && g < 0.0,
                    "token {t} comp {c}: expected -inf (masked), got {g}"
                );
                n_masked_ok += 1;
                continue;
            }
            let mut acc = 0.0f32;
            for h in 0..n_head {
                let q = &q_k[(t * n_head + h) * head_dim..(t * n_head + h) * head_dim + head_dim];
                let k = &index_ring[c * head_dim..c * head_dim + head_dim];
                let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                acc += dot.max(0.0) * (weights_k[t * n_head + h] * scale);
            }
            let rel = ((g - acc).abs() / acc.abs().max(1e-3)) as f64;
            worst_rel = worst_rel.max(rel);
            n_finite += 1;
        }
    }
    eprintln!(
        "[indexer_scores_tiled] worst_rel={worst_rel:.3e} over {n_finite} finite, \
         {n_masked_ok} masked(-inf) OK (n_tokens={n_tokens} n_comp={n_comp})"
    );
    assert!(
        worst_rel < 2e-3,
        "batched indexer scores diverge from CPU oracle: worst_rel={worst_rel:.3e}"
    );
}
