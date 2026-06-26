//! BISECTION (ds4-comp-prefill): localize the chunk-prefill "@3000 generic
//! content" gap to a layer + ring. The CLI fork proved per-token prefill is
//! coherent (recalls the needle) while the full chunk stack is generic, and that
//! removing any chunk flag collapses to degenerate BOS (fallbacks broken) — so
//! flag-bisection is impossible. This compares the post-prefill STATE of the two
//! paths over the SAME real long prompt and reports, per layer, the
//! compressed-KV ring divergence (cos + max|Δ|) so the first/worst diverging
//! compressor/indexer layer is identified in ONE model load.
//!
//! Opt-in: DS4_GGUF=/path/to/ds4flash.gguf [DS4_PROBE_PROMPT=/path/to/prompt.txt]
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Per-ROW divergence signature: treat a/b as [n_rows, hd] and compute, per row,
/// the cosine. Returns (frac rows cos<0.5 "flipped", frac rows cos<0.99 "perturbed",
/// min row cos). A SPARSE signature (small flipped-frac, low min) ⇒ discrete
/// per-token events (MoE expert-routing flips). A BROAD signature (perturbed-frac≈1,
/// no flipped) ⇒ uniform numeric drift.
fn row_divergence(a: &[f32], b: &[f32], hd: usize) -> (f64, f64, f64) {
    if hd == 0 || a.len() < hd {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = a.len() / hd;
    let mut flipped = 0usize;
    let mut perturbed = 0usize;
    let mut min_c = 1.0f64;
    for r in 0..n {
        let ra = &a[r * hd..(r + 1) * hd];
        let rb = &b[r * hd..(r + 1) * hd];
        let c = cos(ra, rb);
        if c < 0.5 {
            flipped += 1;
        }
        if c < 0.99 {
            perturbed += 1;
        }
        if c < min_c {
            min_c = c;
        }
    }
    (flipped as f64 / n as f64, perturbed as f64 / n as f64, min_c)
}

#[test]
fn chunk_vs_pertoken_state_divergence() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_vs_pertoken_state_divergence.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() {
        eprintln!("DS4_GGUF={p} not a file — skipping.");
        return;
    }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt"
            .to_string()
    });
    let text = std::fs::read_to_string(&prompt_path)
        .unwrap_or_else(|e| panic!("read prompt {prompt_path}: {e}"));

    // Tokenize with our own Vocab — exact-vs-antirez ids don't matter here; both
    // prefills get the IDENTICAL sequence, so any state delta is path-only.
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf open");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let prompt: Vec<i32> = vocab
        .encode(text.trim_end_matches('\n'))
        .into_iter()
        .map(|t| t as i32)
        .collect();
    eprintln!("[state-bisect] prompt tokens = {}", prompt.len());

    let raw_cap: u32 = 128;
    let runner = DecodeRunner::open(&path, raw_cap).expect("DecodeRunner::open");

    let snapshot = |chunk: bool| -> (Vec<u32>, Vec<Vec<f32>>, Vec<u32>, Vec<Vec<f32>>) {
        if chunk {
            std::env::set_var("DS4_PREFILL_CHUNK", "8192");
            std::env::set_var("DS4_CHUNK_SWA_KFLASH", "1");
            std::env::set_var("DS4_CHUNK_ATTN_NOSYNC", "1");
            std::env::set_var("DS4_CHUNK_BATCHED_IDX", "1");
            std::env::set_var("DS4_CHUNK_FUSED_COMP", "1");
        } else {
            for v in [
                "DS4_PREFILL_CHUNK",
                "DS4_CHUNK_SWA_KFLASH",
                "DS4_CHUNK_ATTN_NOSYNC",
                "DS4_CHUNK_BATCHED_IDX",
                "DS4_CHUNK_FUSED_COMP",
            ] {
                std::env::remove_var(v);
            }
        }
        let mut s = DecodeSession::new(&runner);
        s.prefill(&prompt).expect("prefill");
        let st = s.state();
        (
            st.n_comp.clone(),
            st.comp_kv_ring.clone(),
            st.n_index_comp.clone(),
            st.index_comp_kv_ring.clone(),
        )
    };

    // Per-token first (the coherent reference), then chunk.
    let (n_pt, comp_pt, ni_pt, idx_pt) = snapshot(false);
    let (n_ck, comp_ck, ni_ck, idx_ck) = snapshot(true);
    // Second chunk run, IDENTICAL config — measures the chunk path's own run-to-run
    // VARIANCE (non-determinism / race), separate from chunk-vs-per-token divergence.
    let (_n_ck2, comp_ck2, _ni_ck2, _idx_ck2) = snapshot(true);
    for v in [
        "DS4_PREFILL_CHUNK",
        "DS4_CHUNK_SWA_KFLASH",
        "DS4_CHUNK_ATTN_NOSYNC",
        "DS4_CHUNK_BATCHED_IDX",
        "DS4_CHUNK_FUSED_COMP",
    ] {
        std::env::remove_var(v);
    }

    let ratios: Vec<u32> = runner
        .composed
        .layers
        .iter()
        .map(|l| l.attn.params.compress_ratio)
        .collect();
    let n_layers = ratios.len();

    eprintln!(
        "[state-bisect] {:>3} {:>6} {:>8} {:>10} {:>11} {:>10}  ck-vs-ck (run-to-run non-determinism)",
        "L", "ratio", "ncomp", "comp_cos", "comp_maxΔ", "ckck_cos"
    );
    let mut worst_comp = (1.0f64, usize::MAX);
    let mut worst_idx = (1.0f64, usize::MAX);
    let mut worst_ckck = (1.0f64, usize::MAX);
    for l in 0..n_layers {
        let nc_eq = if n_pt[l] == n_ck[l] {
            format!("{}", n_pt[l])
        } else {
            format!("{}≠{}", n_pt[l], n_ck[l])
        };
        let (ccos, cmax) = if !comp_pt[l].is_empty()
            && comp_pt[l].len() == comp_ck[l].len()
        {
            (cos(&comp_pt[l], &comp_ck[l]), max_abs_diff(&comp_pt[l], &comp_ck[l]))
        } else {
            (f64::NAN, f32::NAN)
        };
        let ni_eq = if ni_pt[l] == ni_ck[l] {
            format!("{}", ni_pt[l])
        } else {
            format!("{}≠{}", ni_pt[l], ni_ck[l])
        };
        let (icos, imax) = if !idx_pt[l].is_empty() && idx_pt[l].len() == idx_ck[l].len() {
            (cos(&idx_pt[l], &idx_ck[l]), max_abs_diff(&idx_pt[l], &idx_ck[l]))
        } else {
            (f64::NAN, f32::NAN)
        };
        // chunk-run-1 vs chunk-run-2 (same config): run-to-run non-determinism.
        let ckck = if !comp_ck[l].is_empty() && comp_ck[l].len() == comp_ck2[l].len() {
            cos(&comp_ck[l], &comp_ck2[l])
        } else {
            f64::NAN
        };
        if ccos.is_finite() && ccos < worst_comp.0 {
            worst_comp = (ccos, l);
        }
        if icos.is_finite() && icos < worst_idx.0 {
            worst_idx = (icos, l);
        }
        if ckck.is_finite() && ckck < worst_ckck.0 {
            worst_ckck = (ckck, l);
        }
        // Per-ROW divergence signature (sparse routing-flip vs broad drift).
        let (flip_f, pert_f, minrow) = if !comp_pt[l].is_empty()
            && comp_pt[l].len() == comp_ck[l].len()
            && n_pt[l] > 0
        {
            let hd = comp_pt[l].len() / n_pt[l] as usize;
            row_divergence(&comp_pt[l], &comp_ck[l], hd)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        let _ = (ni_eq, icos, imax);
        eprintln!(
            "[state-bisect] {l:>3} {:>6} {nc_eq:>8} {ccos:>10.6} {cmax:>11.3e} {ckck:>10.6}  flip%={:>6.2} pert%={:>6.2} minRowCos={:>8.4}",
            ratios[l], flip_f * 100.0, pert_f * 100.0, minrow,
        );
    }
    let rat = |l: usize| ratios.get(l).copied().unwrap_or(0);
    eprintln!(
        "[state-bisect] WORST comp-ring cos={:.6} @ layer {} (ratio {});  WORST idx-ring cos={:.6} @ layer {} (ratio {})",
        worst_comp.0, worst_comp.1, rat(worst_comp.1),
        worst_idx.0, worst_idx.1, rat(worst_idx.1),
    );
    eprintln!(
        "[state-bisect] WORST chunk-vs-chunk (run-to-run) cos={:.6} @ layer {} (ratio {}) — \
         if <<1, the chunk prefill is NON-DETERMINISTIC (race), so chunk-vs-pertoken divergence \
         is partly noise and single-run coherence is unreliable.",
        worst_ckck.0, worst_ckck.1, rat(worst_ckck.1),
    );
    eprintln!(
        "[state-bisect] ROW-SIGNATURE: small flip% (rows cos<0.5) with most rows clean ⇒ SPARSE \
         discrete per-token events = MoE expert-routing FLIPS (confirms routing-amplification root). \
         pert%≈100 with no flips ⇒ BROAD uniform numeric drift (would point to matmul, not routing)."
    );
    eprintln!(
        "[state-bisect] interpretation: comp_cos≈1 everywhere ⇒ compressor CONTENT faithful \
         (gap is attention/selection, transient); comp_cos<0.99 at some layer ⇒ fused-comp \
         CONTENT diverges there (start the fix at the first such layer)."
    );
}
