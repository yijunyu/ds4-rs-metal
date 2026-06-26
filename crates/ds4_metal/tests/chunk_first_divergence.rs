//! FIRST-DIVERGENCE race localizer for the chunk-prefill nondeterminism.
//!
//! Runs the SAME chunk prefill TWICE (same config) and prints, per layer, the
//! run-to-run divergence of the captured layer-input residual. For deterministic
//! code every layer is bit-identical (cos 1.0, maxΔ 0); the FIRST layer with
//! cos<1 / maxΔ>0 is where a data RACE first manifests — direct localization, no
//! knob sweep. Honors caller env so the cross-layer chaining race (chaining on)
//! vs the finer long-context race (DS4_CHAIN=0) is isolable.
//!
//! Opt-in: DS4_GGUF=/path [DS4_PROBE_PROMPT=/path] [DS4_PROBE_POS=N]
//!         [DS4_CHAIN=0] [DS4_CHUNK_BATCHED_IDX=0 ...]. macOS-only.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::DecodeRunner;

#[test]
fn chunk_first_divergence_test() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_first_divergence_test.");
        return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path)
        .unwrap_or_else(|e| panic!("read prompt {prompt_path}: {e}"));

    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    eprintln!("[firstdiv] prompt tokens = {}", prompt.len());

    let target_pos: usize = std::env::var("DS4_PROBE_POS").ok().and_then(|v| v.parse().ok())
        .unwrap_or(prompt.len().saturating_sub(2));

    let runner = DecodeRunner::open(&path, 128).expect("open");
    let ratios: Vec<u32> = runner.composed.layers.iter().map(|l| l.attn.params.compress_ratio).collect();

    let div = runner.chunk_first_divergence(&prompt, target_pos).expect("probe");

    let bufname = |w: f32| match w as usize { 0 => "resid", 1 => "comp_ring", 2 => "index_ring", _ => "kv_ring" };
    eprintln!("[firstdiv] run-to-run (chunk vs chunk, SAME config) per-layer @ pos {target_pos}:");
    eprintln!("[firstdiv] {:>3} {:>6} {:>10} {:>10}", "L", "ratio", "min_cos", "buffer");
    let mut first_race: Option<(usize, &str)> = None;
    for &(l, c, w) in &div {
        // cos≈0 at the early hash-routed layers is a capture-representation artifact
        // (resid only); a REAL race shows as 0<cos<1 at a layer whose siblings are 1.0.
        if (0.5..0.99999).contains(&c) && first_race.is_none() { first_race = Some((l, bufname(w))); }
        eprintln!("[firstdiv] {l:>3} {:>6} {c:>10.6} {:>10}", ratios.get(l).copied().unwrap_or(0), bufname(w));
    }
    eprintln!(
        "[firstdiv] FIRST run-to-run divergence (0.5<cos<1) @ {:?}. All cos==1.0 ⇒ deterministic \
         (no race this run). A divergent comp_ring/index_ring with bit-identical resid ⇒ the race \
         is the COMPRESSOR RING WRITE (decode-consumed, residual-independent), not the compute path.",
        first_race,
    );
}

/// Fire-rate anchor: run N fresh prefills and count how many collapse to the
/// all-zeros (token 0) signature. Opt-in: DS4_GGUF + [DS4_DEC_LEN] [DS4_RATE_TRIALS].
/// Run twice — DS4_CHAIN unset (chain on) vs DS4_CHAIN=0 — to split chaining-race
/// from cb-resource-limit.
#[test]
fn chunk_decode_allzeros_rate_test() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_decode_allzeros_rate_test."); return;
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
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1536).min(full.len());
    let trials: u32 = std::env::var("DS4_RATE_TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let n_dec: u32 = std::env::var("DS4_DEC_NDEC").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let chain = std::env::var("DS4_CHAIN").unwrap_or_else(|_| "(on)".into());
    let prompt = &full[full.len() - l..];
    let runner = DecodeRunner::open(&path, 128).expect("open");
    let results = runner.decode_allzeros_rate(prompt, trials, n_dec).expect("rate");
    let streams: Vec<&Vec<i32>> = results.iter().map(|(s, _)| s).collect();
    // First decode step where ANY two trials disagree, and # of distinct streams.
    let mut first_div: Option<usize> = None;
    'outer: for step in 0..n_dec as usize {
        let t0 = streams[0].get(step);
        for s in &streams { if s.get(step) != t0 { first_div = Some(step); break 'outer; } }
    }
    // SPLIT the token-0 streams: cb-fault (true all-zero logits) vs BOS-prediction.
    let cb_fault = results.iter().filter(|(_, z)| *z).count();
    let bos_pred = results.iter().filter(|(s, z)| s.first() == Some(&0) && !z).count();
    use std::collections::HashSet;
    let distinct: HashSet<&Vec<i32>> = streams.iter().copied().collect();
    eprintln!("[rate] L={l} trials={trials} n_decode={n_dec} DS4_CHAIN={chain}");
    eprintln!("[rate] CB-FAULT(all-zero logits)={cb_fault}/{trials}  BOS-incoherence={bos_pred}/{trials}  distinct_streams={}/{trials}  FIRST_DIVERGENT_STEP={first_div:?}", distinct.len());
    for (i, (s, z)) in results.iter().enumerate() {
        let head: Vec<i32> = s.iter().take(12).copied().collect();
        eprintln!("[rate] trial {i} {}: {head:?}...", if *z { "CB-FAULT" } else { "ok-logits" });
    }
}

/// TTFT A/B: time DecodeSession::prefill per-token vs chunked (gate clamps chunk to
/// raw_cap above DS4_CHUNK_MAX_CTX) in ONE model load. DS4_GGUF + [DS4_DEC_LEN].
#[test]
fn prefill_ttft_ab() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping prefill_ttft_ab."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path).unwrap();
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1536).min(full.len());
    let prompt = &full[full.len() - l..];
    let runner = DecodeRunner::open(&path, 128).expect("open");
    let time = |label: &str, chunk: Option<&str>| {
        match chunk { Some(c) => std::env::set_var("DS4_PREFILL_CHUNK", c), None => std::env::remove_var("DS4_PREFILL_CHUNK") }
        let t = std::time::Instant::now();
        let mut s = ds4_metal::decode_runner::DecodeSession::new(&runner);
        s.prefill(prompt).expect("prefill");
        let dt = t.elapsed().as_secs_f64();
        eprintln!("[ttft] {label}: {:.2}s = {:.2} tok/s", dt, l as f64 / dt);
    };
    time("warmup(per-token)", None);
    time("per-token", None);
    time("chunk(gate→128)", Some("8192"));
    time("chunk(gate→128) rep", Some("8192"));
    std::env::remove_var("DS4_PREFILL_CHUNK");
}

/// DECODE-phase first-divergence: run prefill + N argmax decode steps TWICE and find
/// the first decode STEP whose token differs between the runs (prefill is provably
/// deterministic, so a divergence is a decode-phase race). Uses the LAST L tokens of
/// the needle prompt. Opt-in: DS4_GGUF + [DS4_DEC_LEN=N] [DS4_DEC_NDEC=N].
#[test]
fn chunk_decode_divergence_test() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping chunk_decode_divergence_test."); return;
    };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF={p} not a file — skipping."); return; }
    let prompt_path = std::env::var("DS4_PROBE_PROMPT").unwrap_or_else(|_| {
        "benchmarks/ds4_msl/upstream/ds4/tests/test-vectors/prompts/long_memory_archive.txt".into()
    });
    let text = std::fs::read_to_string(&prompt_path)
        .unwrap_or_else(|e| panic!("read prompt {prompt_path}: {e}"));
    let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("gguf");
    let vocab = ds4_engine::tokenizer::Vocab::from_gguf(&gguf).expect("vocab");
    let full: Vec<i32> = vocab.encode(text.trim_end_matches('\n')).into_iter().map(|t| t as i32).collect();
    let l: usize = std::env::var("DS4_DEC_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
    let n_dec: u32 = std::env::var("DS4_DEC_NDEC").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let l = l.min(full.len());
    let prompt = &full[full.len() - l..];
    eprintln!("[decdiv] L={l} n_decode={n_dec}");

    let runner = DecodeRunner::open(&path, 128).expect("open");
    let (a, b, first) = runner.chunk_decode_divergence(prompt, n_dec).expect("probe");
    eprintln!("[decdiv] stream A = {a:?}");
    eprintln!("[decdiv] stream B = {b:?}");
    match first {
        Some(step) => eprintln!(
            "[decdiv] FIRST DECODE DIVERGENCE @ step {step}: A={} B={} ⇒ decode-phase race confirmed; \
             the race first flips the token at decode step {step}.", a[step], b[step]),
        None => eprintln!("[decdiv] streams IDENTICAL ⇒ no decode race fired this run (both warm); \
             retry or use a higher-flip length."),
    }
}

