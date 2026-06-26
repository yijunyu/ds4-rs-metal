//! GPU indexer-score parity smoke test (DS4 long-context indexer offload).
//!
//! `BatchScope::encode_indexer_scores` is the GPU port of the dominant compute
//! in `indexer_allowed_decode_one` (the `n_comp > DS4_N_INDEXER_TOP_K` path that
//! only fires past ~2048 tokens of context). This test validates that the GPU
//! scores + the resulting top-512 selection match an inline CPU reference on
//! synthetic `n_comp = 600` inputs — no model load, fast.
//!
//! Rope is exercised separately (rope_tail_q_heads_in_place has its own tests);
//! here `n_rot = 0` so both paths skip it, isolating the q-matvec + weights-matvec
//! + score-kernel math. GPU uses float4+simd_sum reductions, so scores match the
//! CPU's sequential dot only to a small relative tolerance; the top-512 *set*
//! must still agree (allowing a couple of near-tie boundary flips).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

fn params_no_rope() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 512,
        n_hc: 4,
        n_head: 64,
        head_dim: 128,
        n_rot: 0, // skip rope — isolates the score math
        n_lora_q: 256,
        n_lora_kv: 128,
        hc_sinkhorn_iter: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 4,
        n_out_group: 1,
    }
}

#[test]
fn encode_indexer_scores_matches_cpu_reference() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let p = params_no_rope();
    let n_head = p.n_head as usize; // 64
    let head_dim = p.head_dim as usize; // 128
    let q_dim = n_head * head_dim; // 8192
    let n_lora_q = p.n_lora_q as usize; // 256
    let d_embd = p.d_embd as usize; // 512
    let n_comp = 600usize; // > DS4_N_INDEXER_TOP_K (512) → real path
    let top_k = 512usize;

    // Deterministic small-magnitude inputs (LCG), centered around 0.
    let mut rng: u32 = 0x1234_5678;
    let mut next = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5 // ~[-0.5, 0.5)
    };
    let qr: Vec<f32> = (0..n_lora_q).map(|_| next()).collect();
    let cur: Vec<f32> = (0..d_embd).map(|_| next()).collect();
    let w_q_b: Vec<f32> = (0..q_dim * n_lora_q).map(|_| next() * 0.1).collect();
    let w_proj: Vec<f32> = (0..d_embd * n_head).map(|_| next() * 0.1).collect();
    let ring: Vec<f32> = (0..n_comp * head_dim).map(|_| next()).collect();

    // ---- GPU path ----
    let gpu_scores = {
        let scope = disp.batch_scope();
        let qr_db = scope.upload_f32(&qr);
        let cur_db = scope.upload_f32(&cur);
        let ring_db = scope.upload_f32(&ring);
        let scores_db = scope
            .encode_indexer_scores(
                &qr_db, &cur_db, &w_q_b, &w_proj, None, None, &ring_db, n_comp, n_head, head_dim, &p, 0,
            )
            .expect("encode_indexer_scores");
        scope.flush_and_read(&scores_db)
    };
    assert_eq!(gpu_scores.len(), n_comp);

    // ---- CPU reference (same math, sequential reduction, no rope) ----
    // q[j] = Σ_k w_q_b[j*n_lora_q + k] * qr[k]  (col-major == row-major [q_dim,n_lora_q])
    let mut q = vec![0.0f32; q_dim];
    for j in 0..q_dim {
        let row = &w_q_b[j * n_lora_q..(j + 1) * n_lora_q];
        let mut acc = 0.0f32;
        for k in 0..n_lora_q {
            acc += row[k] * qr[k];
        }
        q[j] = acc;
    }
    // weights[h] = w_proj · cur (UNSCALED here; scale applied in the score sum)
    let mut weights = vec![0.0f32; n_head];
    for h in 0..n_head {
        let row = &w_proj[h * d_embd..(h + 1) * d_embd];
        let mut acc = 0.0f32;
        for k in 0..d_embd {
            acc += row[k] * cur[k];
        }
        weights[h] = acc;
    }
    let scale = 1.0f32 / ((head_dim * n_head) as f32).sqrt();
    let mut cpu_scores = vec![0.0f32; n_comp];
    for c in 0..n_comp {
        let kv = &ring[c * head_dim..(c + 1) * head_dim];
        let mut s = 0.0f32;
        for h in 0..n_head {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += kv[d] * qh[d];
            }
            if dot < 0.0 {
                dot = 0.0;
            }
            s += dot * weights[h] * scale;
        }
        cpu_scores[c] = s;
    }

    // ---- Score agreement (small rel tol for float4+simd_sum vs sequential) ----
    let max_abs_cpu = cpu_scores.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut max_abs_diff = 0.0f32;
    for c in 0..n_comp {
        max_abs_diff = max_abs_diff.max((gpu_scores[c] - cpu_scores[c]).abs());
    }
    let rel = max_abs_diff / max_abs_cpu.max(1e-9);
    eprintln!(
        "indexer scores: n_comp={n_comp} max|cpu|={max_abs_cpu:.4e} max_abs_diff={max_abs_diff:.4e} rel={rel:.3e}"
    );
    assert!(
        rel < 1e-3,
        "GPU indexer scores diverge from CPU: rel={rel:.3e} (max_abs_diff={max_abs_diff:.3e}, max|cpu|={max_abs_cpu:.3e})"
    );

    // ---- Top-512 selection set agreement (greedy argmax, same as CPU path) ----
    let topk_set = |scores: &[f32]| -> std::collections::HashSet<usize> {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.into_iter().take(top_k).collect()
    };
    let gpu_sel = topk_set(&gpu_scores);
    let cpu_sel = topk_set(&cpu_scores);
    let inter = gpu_sel.intersection(&cpu_sel).count();
    eprintln!("top-{top_k} selection intersection: {inter}/{top_k}");
    // Allow a few near-tie boundary flips (rows within score-tol of the cutoff).
    assert!(
        inter >= top_k - 4,
        "GPU top-{top_k} selection disagrees with CPU: only {inter}/{top_k} shared"
    );
}

/// GPU cooperative greedy top-k (`encode_indexer_topk`) vs the CPU greedy in
/// `indexer_allowed_decode_one`. Synthetic scores, no model load.
#[test]
fn encode_indexer_topk_matches_cpu_greedy() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_comp = 2000usize;
    let top_k = 512usize;
    // Deterministic scores, mostly distinct with a few deliberate ties to
    // exercise the lowest-index tie-break.
    let mut rng: u32 = 0x9e37_79b9;
    let mut next = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut scores: Vec<f32> = (0..n_comp).map(|_| next()).collect();
    for i in 0..8 {
        scores[500 + i] = 0.5;
        scores[1000 + i] = 0.5;
    }

    let gpu_sel: std::collections::HashSet<i32> = {
        let scope = disp.batch_scope();
        let scores_db = scope.upload_f32(&scores);
        let sel_db = scope
            .encode_indexer_topk(&scores_db, n_comp, top_k)
            .expect("encode_indexer_topk");
        scope.flush_and_read_i32(&sel_db).into_iter().collect()
    };
    assert_eq!(gpu_sel.len(), top_k, "GPU selected {} != top_k", gpu_sel.len());

    // CPU greedy (mirror of the top-k pass in indexer_allowed_decode_one).
    let mut allowed = vec![false; n_comp];
    for _ in 0..top_k {
        let mut best = 0usize;
        let mut bs = f32::NEG_INFINITY;
        for c in 0..n_comp {
            if !allowed[c] && scores[c] > bs {
                best = c;
                bs = scores[c];
            }
        }
        allowed[best] = true;
    }
    let cpu_sel: std::collections::HashSet<i32> =
        (0..n_comp).filter(|&c| allowed[c]).map(|c| c as i32).collect();

    let inter = gpu_sel.intersection(&cpu_sel).count();
    eprintln!("topk: n_comp={n_comp} top_k={top_k} intersection={inter}/{top_k}");
    assert_eq!(inter, top_k, "GPU greedy top-k differs from CPU: {inter}/{top_k} shared");
}

/// Reproduce the e2e regime: small n_comp (just above top_k), top_k=16, with
/// distinct AND tied scores. Compares the GPU top-k set vs the CPU *sort*
/// top-k (the scores-only decode path's algorithm) — the exact pair the 3-way
/// benchmark found diverging.
#[test]
fn encode_indexer_topk_small_ncomp_cases() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    let top_k = 16usize;
    let cases: &[(usize, &str)] = &[(17, "distinct"), (17, "ties"), (20, "ties"), (33, "ties")];
    for &(n_comp, kind) in cases {
        let mut rng: u32 = 0xdead_beef ^ (n_comp as u32);
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut scores: Vec<f32> = (0..n_comp).map(|_| next()).collect();
        if kind == "ties" {
            // Cluster several rows at identical values straddling the cutoff.
            for c in 0..n_comp {
                if c % 3 == 0 {
                    scores[c] = 0.25;
                }
            }
        }
        // CPU sort top-k (scores-only decode path algorithm): (score desc, idx asc).
        let mut order: Vec<u32> = (0..n_comp as u32).collect();
        order.sort_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(top_k);
        let cpu_set: std::collections::HashSet<i32> =
            order.into_iter().map(|i| i as i32).collect();
        // GPU greedy top-k.
        let gpu_set: std::collections::HashSet<i32> = {
            let scope = disp.batch_scope();
            let sdb = scope.upload_f32(&scores);
            let sel = scope
                .encode_indexer_topk(&sdb, n_comp, top_k)
                .expect("topk");
            scope.flush_and_read_i32(&sel).into_iter().collect()
        };
        let inter = gpu_set.intersection(&cpu_set).count();
        eprintln!("  small-case n_comp={n_comp} {kind}: gpu_len={} inter={inter}/{top_k}", gpu_set.len());
        assert_eq!(
            inter, top_k,
            "n_comp={n_comp} {kind}: GPU top-k != CPU sort ({inter}/{top_k})"
        );
    }
}
