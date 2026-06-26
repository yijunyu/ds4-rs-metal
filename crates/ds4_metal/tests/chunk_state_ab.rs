//! Decode-input state A/B for the chunk>raw_cap corruption: run the corrupt
//! whole-chunk prefill (split=15) and the coherent chunk-128 prefill over the
//! SAME prompt, snapshot EVERYTHING decode reads (cur_hc, comp/idx rings,
//! pools, n_comp, fp8 raw KV ring contents), and report per layer which input
//! diverges beyond fp32 drift. Localizes the corruption without perturbing the
//! GPU schedule (read-only after prefill).
//!
//! Opt-in: DS4_GGUF + [DS4_PROBE_PROMPT] [DS4_DEC_LEN=1536]. macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64; na += (x as f64).powi(2); nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

struct Snap {
    cur_hc: Vec<f32>,
    n_comp: Vec<u32>,
    comp: Vec<Vec<f32>>,
    idx: Vec<Vec<f32>>,
    pool_kv: Vec<Vec<f32>>,
    kv_ring: Vec<Vec<f32>>, // fp8 ring dequant bytes as raw f32 reinterpret (byte compare)
    logits_argmax: i32,
}

#[test]
fn chunk_state_ab() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_state_ab."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path).expect("prompt");
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
    let l = l.min(full.len());
    let prompt = &full[full.len() - l..];

    let raw_cap = 128u32;
    let runner = DecodeRunner::open(&path, raw_cap).expect("open");
    let n_layers = runner.composed.layers.len();

    let mut snap = |chunk: &str, split: &str| -> Snap {
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
        std::env::set_var("DS4_PREFILL_CHUNK", chunk);
        // SWA_KFLASH excluded — known-incoherent at chunk>raw_cap (NaN at tile
        // boundary), not used by the production whole-chunk stack.
        for k in ["DS4_CHUNK_ATTN_NOSYNC","DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP"] {
            std::env::set_var(k, "1");
        }
        std::env::set_var("DS4_PHASEB_SPLIT", split);
        let mut s = DecodeSession::new(&runner);
        s.prefill(prompt).expect("prefill");
        let st = s.state();
        let mut kv_ring = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let lp = &runner.composed.layers[li].attn.params;
            let row = lp.n_lora_kv as usize;
            let buf = runner.dispatcher.kv_buffer_or_alloc(li as u32, raw_cap as usize * row * 4);
            let v = unsafe {
                std::slice::from_raw_parts(buf.contents() as *const f32, raw_cap as usize * row)
            }.to_vec();
            kv_ring.push(v);
        }
        let am = {
            let lg = s.logits();
            let mut b = 0i32; let mut bv = f32::NEG_INFINITY;
            for (i, &v) in lg.iter().enumerate() { if v > bv { bv = v; b = i as i32; } }
            b
        };
        Snap {
            cur_hc: st.cur_hc.clone(),
            n_comp: st.n_comp.clone(),
            comp: st.comp_kv_ring.clone(),
            idx: st.index_comp_kv_ring.clone(),
            pool_kv: st.comp_state_kv.clone(),
            kv_ring,
            logits_argmax: am,
        }
    };

    let a = snap("8192", "15"); // corrupt whole-chunk (deterministic w/ split=15)
    let b = snap("128", "0");   // coherent
    eprintln!("[ab] argmax whole={} chunk128={}", a.logits_argmax, b.logits_argmax);
    eprintln!("[ab] cur_hc cos={:.6}", cos(&a.cur_hc, &b.cur_hc));
    for li in 0..n_layers {
        if a.n_comp[li] != b.n_comp[li] {
            eprintln!("[ab] L{li} n_comp {} != {}", a.n_comp[li], b.n_comp[li]);
        }
        let c_comp = if a.comp[li].len() == b.comp[li].len() && !a.comp[li].is_empty() {
            cos(&a.comp[li], &b.comp[li]) } else { f64::NAN };
        let c_idx = if a.idx[li].len() == b.idx[li].len() && !a.idx[li].is_empty() {
            cos(&a.idx[li], &b.idx[li]) } else { f64::NAN };
        let c_pool = if a.pool_kv[li].len() == b.pool_kv[li].len() && !a.pool_kv[li].is_empty() {
            cos(&a.pool_kv[li], &b.pool_kv[li]) } else { f64::NAN };
        let c_kv = cos(&a.kv_ring[li], &b.kv_ring[li]);
        if c_comp.min(c_idx).min(c_pool).min(c_kv) < 0.999 || c_comp.is_nan() {
            eprintln!("[ab] L{li:>2} comp={c_comp:.6} idx={c_idx:.6} pool={c_pool:.6} kvring={c_kv:.6}");
        }
    }
    eprintln!("[ab] done");
}
