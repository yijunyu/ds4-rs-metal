//! HEADS/SEL probe for the chunk>raw_cap deterministic corruption: compare the
//! per-position rope-backed flash heads of LAYER `DS4_HEADS_LAYER` (default 6,
//! first ratio-4 idx layer) between the COHERENT per-token prefill and the
//! corrupt single-chunk crossing prefill (chunk_start=0, K>raw_cap=128). The
//! first position where the heads diverge structurally (cos << 1 beyond fp32
//! drift) localizes the bug; the chunk sel coverage check ([SEL] lines) tells
//! whether the indexer selection misses visible comp rows.
//!
//! Opt-in: DS4_GGUF=/path [DS4_PROBE_PROMPT=/path] [DS4_DEC_LEN=256]
//!         [DS4_HEADS_LAYER=6]. macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

#[test]
fn chunk_pertoken_heads_diff() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_pertoken_heads_diff."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path).expect("prompt");
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let l = l.min(full.len());
    let prompt = &full[full.len() - l..];
    let layer: usize = std::env::var("DS4_HEADS_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    eprintln!("[hdiff] L={l} layer={layer}");

    let runner = DecodeRunner::open(&path, 128).expect("open");
    let p6 = &runner.composed.layers[layer].attn.params;
    let q_dim = p6.n_head as usize * p6.head_dim as usize;
    let init = l - 1; // prefill captures init = l-1 positions on both paths

    let pt_file = std::env::temp_dir().join("ds4_pt_heads.bin");
    let ck_file = std::env::temp_dir().join("ds4_ck_heads.bin");

    // Reference: per-token (default) or a chunk run with DS4_HDIFF_CHUNK_REF
    // (chunk-vs-chunk A/B, e.g. coherent 128 vs corrupt whole-chunk).
    for v in ["DS4_PREFILL_CHUNK","DS4_CHUNK_SWA_KFLASH","DS4_CHUNK_ATTN_NOSYNC",
              "DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP","DS4_CHUNK_MAX_CTX",
              "DS4_CHUNK_HEADS_DUMP","DS4_CHUNK_HEADS_FILE","DS4_CHUNK_SEL_DUMP"] {
        std::env::remove_var(v);
    }
    if let Ok(refk) = std::env::var("DS4_HDIFF_CHUNK_REF") {
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
        std::env::set_var("DS4_PREFILL_CHUNK", refk);
        // SWA_KFLASH excluded — known-incoherent at chunk>raw_cap (NaN at tile
        // boundary), not used by the production whole-chunk stack.
        for k in ["DS4_CHUNK_ATTN_NOSYNC","DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP"] {
            std::env::set_var(k, "1");
        }
        std::env::set_var("DS4_CHUNK_HEADS_DUMP", layer.to_string());
        std::env::set_var("DS4_CHUNK_HEADS_FILE", pt_file.to_string_lossy().to_string());
        let mut s = DecodeSession::new(&runner);
        s.prefill(prompt).expect("ref chunk prefill");
        for v in ["DS4_CHUNK_HEADS_DUMP","DS4_CHUNK_HEADS_FILE"] { std::env::remove_var(v); }
    } else {
        std::env::set_var("DS4_PT_HEADS_DUMP", layer.to_string());
        std::env::set_var("DS4_PT_HEADS_FILE", pt_file.to_string_lossy().to_string());
        let mut s = DecodeSession::new(&runner);
        s.prefill(prompt).expect("pt prefill");
        std::env::remove_var("DS4_PT_HEADS_DUMP");
        std::env::remove_var("DS4_PT_HEADS_FILE");
    }

    // Single whole-chunk crossing run (production chunk stack), heads + sel dumps.
    std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
    let chunk_sz = std::env::var("DS4_HDIFF_CHUNK").unwrap_or_else(|_| "8192".into());
    std::env::set_var("DS4_PREFILL_CHUNK", chunk_sz);
    // SWA_KFLASH excluded — known-incoherent at chunk>raw_cap (NaN at tile
    // boundary), not used by the production whole-chunk stack.
    for k in ["DS4_CHUNK_ATTN_NOSYNC","DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP"] {
        std::env::set_var(k, "1");
    }
    std::env::set_var("DS4_CHUNK_HEADS_DUMP", layer.to_string());
    std::env::set_var("DS4_CHUNK_HEADS_FILE", ck_file.to_string_lossy().to_string());
    std::env::set_var("DS4_CHUNK_SEL_DUMP", layer.to_string());
    {
        let mut s = DecodeSession::new(&runner);
        s.prefill(prompt).expect("ck prefill");
    }
    for v in ["DS4_CHUNK_MAX_CTX","DS4_PREFILL_CHUNK","DS4_CHUNK_SWA_KFLASH","DS4_CHUNK_ATTN_NOSYNC",
              "DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP","DS4_CHUNK_HEADS_DUMP",
              "DS4_CHUNK_HEADS_FILE","DS4_CHUNK_SEL_DUMP"] {
        std::env::remove_var(v);
    }

    let pt = std::fs::read(&pt_file).expect("pt heads file");
    let ck = std::fs::read(&ck_file).expect("ck heads file");
    let ptf: &[f32] = unsafe { std::slice::from_raw_parts(pt.as_ptr() as *const f32, pt.len() / 4) };
    let ckf: &[f32] = unsafe { std::slice::from_raw_parts(ck.as_ptr() as *const f32, ck.len() / 4) };
    let n = (ptf.len() / q_dim).min(ckf.len() / q_dim).min(init);
    eprintln!("[hdiff] comparing {n} positions (q_dim={q_dim})");
    let mut first_bad: Option<usize> = None;
    let mut min_cos = 1.0f64;
    for i in 0..n {
        let a = &ptf[i * q_dim..(i + 1) * q_dim];
        let b = &ckf[i * q_dim..(i + 1) * q_dim];
        let c = cos(a, b);
        min_cos = min_cos.min(c);
        let bad = c < 0.999;
        if bad && first_bad.is_none() { first_bad = Some(i); }
        if bad || (110..140).contains(&i) {
            eprintln!("[hdiff] pos={i:>4} cos={c:.6}{}", if bad { "  <-- DIVERGES" } else { "" });
        }
    }
    eprintln!("[hdiff] first structurally divergent pos = {first_bad:?}; min cos = {min_cos:.6}");
}
