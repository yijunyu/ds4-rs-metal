//! Measure the CHUNK-PREFILL coherence onset — the context length at which the
//! chunk path stops recalling the long-range needle ("gamma") — to set
//! DS4_CHUNK_MAX_CTX precisely (currently a conservative 2048 default).
//!
//! One model load; sweeps context lengths by taking the LAST L tokens of the
//! needle prompt (every record contains the needle, the question is at the end,
//! so any window is a valid needle test). For each L: FORCE the chunk path
//! (DS4_CHUNK_MAX_CTX=0 disables the routing gate) + full chunk stack, prefill,
//! greedy-decode, and check whether the output recalls "gamma". Onset = the
//! smallest L that fails. Chunk prefill is fast, so the whole sweep is cheap.
//!
//! Opt-in: DS4_GGUF=/path [DS4_PROBE_PROMPT=/path] [DS4_ONSET_LENS=256,512,...].
//! macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; best = i as i32; } }
    best
}

#[test]
fn chunk_coherence_onset_sweep() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_coherence_onset_sweep."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path).unwrap_or_else(|e| panic!("read {prompt_path}: {e}"));
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    eprintln!("[onset] full prompt tokens = {}", full.len());

    let lens: Vec<usize> = std::env::var("DS4_ONSET_LENS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3000]);

    let runner = DecodeRunner::open(&path, 128).expect("open");
    let n_dec = 40u32;
    let eos = -1i32; // detok ignores; just decode n_dec tokens

    // DS4_ONSET_AUTO=1: validate the PRODUCTION routing gate (DecodeSession::prefill)
    // — set NO chunk env at all and let the gate auto-route prompts in
    // (raw_cap, 4000] through the fast chunk stack. This is the real regression
    // guard for the wired-in default. Otherwise FORCE chunk explicitly (the
    // benchmarking path: DS4_CHUNK_MAX_CTX=0 + DS4_PREFILL_CHUNK=8192).
    let auto = std::env::var("DS4_ONSET_AUTO").ok().as_deref() == Some("1");
    if !auto {
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
        std::env::set_var("DS4_PREFILL_CHUNK", "8192");
        // Perf knobs default to the fast batched stack but honor a caller-set value,
        // so the SYNC chunk path (e.g. DS4_CHUNK_BATCHED_IDX=0) can be measured for
        // determinism vs the nosync fast path.
        // DS4_CHUNK_SWA_KFLASH is DELIBERATELY EXCLUDED from this default-on set: it
        // has a tile-boundary hazard (tile t+1's ring stores clobber tile t's window
        // reads at chunk>raw_cap) that injects NaN at the chunk tail → deterministic
        // BOS. The nosync batched path supersedes it coherently for single-chunk
        // prefill, and production never enables it. Forcing it on here is what made
        // every prior whole-chunk coherence run read BOS (the phantom "Phase-B logic
        // bug"). Set DS4_CHUNK_SWA_KFLASH=1 explicitly only to reproduce that fault.
        for k in ["DS4_CHUNK_ATTN_NOSYNC",
                  "DS4_CHUNK_BATCHED_IDX", "DS4_CHUNK_FUSED_COMP"] {
            if std::env::var(k).is_err() { std::env::set_var(k, "1"); }
        }
    }

    eprintln!("[onset] CHUNK-path needle ('gamma') recall vs context length:");
    let mut last_good = 0usize;
    let mut first_bad = 0usize;
    // DS4_ONSET_REPEAT=N: re-run the whole length sweep N times in ONE model-load
    // to expose run-to-run nondeterminism (the chunk path is known to flip
    // gamma-recall stochastically). Per (L, repeat) one greedy decode + recall.
    let repeats: u32 = std::env::var("DS4_ONSET_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for rep in 0..repeats {
    for &l in &lens {
        if l > full.len() { continue; }
        let prompt = &full[full.len() - l..]; // last L tokens (records + question)
        let mut s = DecodeSession::new(&runner);
        if s.prefill(prompt).is_err() { eprintln!("[onset] rep={rep} L={l}: prefill ERR"); continue; }
        let mut out = Vec::with_capacity(n_dec as usize);
        for _ in 0..n_dec {
            let t = argmax(s.logits());
            if t == eos { break; }
            out.push(t as u32);
            if s.step(t).is_err() { break; }
        }
        let txt = vocab.decode(&out).unwrap_or_default();
        let recalls = txt.to_lowercase().contains("gamma");
        if recalls { last_good = last_good.max(l); } else if first_bad == 0 { first_bad = l; }
        let head: String = txt.chars().take(72).collect();
        eprintln!("[onset] rep={rep} L={l:>5}  gamma={}  | {head}", if recalls { "YES" } else { "no " });
    }
    }
    for v in ["DS4_CHUNK_MAX_CTX","DS4_PREFILL_CHUNK","DS4_CHUNK_SWA_KFLASH",
              "DS4_CHUNK_ATTN_NOSYNC","DS4_CHUNK_BATCHED_IDX","DS4_CHUNK_FUSED_COMP"] {
        std::env::remove_var(v);
    }
    eprintln!(
        "[onset] last length recalling gamma = {last_good}; first failing = {first_bad}. \
         ⇒ set DS4_CHUNK_MAX_CTX at/below the onset (between {last_good} and {first_bad}).",
    );
}
