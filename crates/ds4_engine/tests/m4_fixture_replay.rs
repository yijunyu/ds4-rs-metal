//! M4 #292 — offline antirez-fixture replay harness (Linux-side bisect).
//!
//! Captures antirez `cur_hc` per-(layer, slot) fingerprints into committed
//! fixtures (one file per prompt × pos) and replays our CPU oracle on the
//! same GGUF, asserting RMS within tolerance + top-2 dim agreement.
//!
//! Lets the M4 cluster-drift bisect work move off AWS Mac (which costs
//! money per hour) and onto Linux + a local DS4 GGUF, by reading the
//! fixture as ground truth instead of re-running antirez.
//!
//! ## Running
//!
//! ```text
//! DS4_GGUF=/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
//!   cargo test -p ds4_engine --test m4_fixture_replay -- --nocapture
//! ```
//!
//! Without `DS4_GGUF` the test prints a one-line skip and passes — keeps
//! CI green on machines without a DS4 GGUF on disk. The fixture file is
//! always parsed (cheap) so format regressions are caught even without
//! the model.
//!
//! ## Capturing new fixtures
//!
//! See `scripts/m4_capture_antirez_fixture.sh` (Mac-side). Schema is in
//! `crates/ds4_engine/tests/fixtures/antirez/README` — header lines
//! `# key value`, then one line per (layer, slot) with the slot's RMS
//! and top-3 (dim, value) pairs.

use std::path::{Path, PathBuf};

use ds4_engine::attn_dispatch::{CpuAttentionDispatcher, DefaultsDs4, CURRENT_TOKEN_HINT};
use ds4_engine::decode_step::{
    arm_hc_slot_recorder, decode_step_with_attn, drain_hc_slot_recorder, AttnStepState,
    ComposedModelWeights, DecodeConfig, HcSlotSample,
};
use ds4_engine::dispatch::CpuDispatcher;
use ds4_engine::gguf::validate_ds4_layout;
use ds4_engine::layer_view::LayerViews;

#[derive(Debug, Clone)]
struct SlotRecord {
    layer: usize,
    slot: usize,
    rms: f64,
    top3: [(usize, f64); 3],
}

#[derive(Debug, Clone)]
struct Fixture {
    prompt: String,
    prompt_len: usize,
    pos: u32,
    prompt_token_ids: Vec<i32>,
    slots: Vec<SlotRecord>,
}

fn parse_fixture(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    let mut prompt = String::new();
    let mut prompt_len: usize = 0;
    let mut pos: u32 = 0;
    let mut prompt_token_ids: Vec<i32> = Vec::new();
    let mut slots: Vec<SlotRecord> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ") {
            let mut parts = rest.splitn(2, ' ');
            let key = parts.next().unwrap_or("");
            let val = parts.next().unwrap_or("");
            match key {
                "prompt" => prompt = val.to_string(),
                "prompt_len" => prompt_len = val.parse().expect("prompt_len u32"),
                "pos" => pos = val.parse().expect("pos u32"),
                "prompt_token_ids" => {
                    prompt_token_ids = val
                        .split(',')
                        .map(|s| s.parse::<i32>().expect("token id i32"))
                        .collect();
                }
                _ => {}
            }
            continue;
        }
        // Data line: layer slot rms d0 v0 d1 v1 d2 v2
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cols.len(), 9, "bad fixture line (want 9 cols): {line:?}");
        let layer: usize = cols[0].parse().expect("layer usize");
        let slot: usize = cols[1].parse().expect("slot usize");
        let rms: f64 = cols[2].parse().expect("rms f64");
        let d0: usize = cols[3].parse().expect("d0 usize");
        let v0: f64 = cols[4].parse().expect("v0 f64");
        let d1: usize = cols[5].parse().expect("d1 usize");
        let v1: f64 = cols[6].parse().expect("v1 f64");
        let d2: usize = cols[7].parse().expect("d2 usize");
        let v2: f64 = cols[8].parse().expect("v2 f64");
        slots.push(SlotRecord {
            layer,
            slot,
            rms,
            top3: [(d0, v0), (d1, v1), (d2, v2)],
        });
    }
    assert!(prompt_len > 0, "fixture missing prompt_len");
    assert!(
        !prompt_token_ids.is_empty(),
        "fixture missing prompt_token_ids"
    );
    assert_eq!(
        prompt_token_ids.len(),
        prompt_len,
        "prompt_token_ids count {} != prompt_len {}",
        prompt_token_ids.len(),
        prompt_len
    );
    assert!(!slots.is_empty(), "fixture has no slot records");
    Fixture {
        prompt,
        prompt_len,
        pos,
        prompt_token_ids,
        slots,
    }
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as i32
}

/// Test entry point: parses the fixture, then if `DS4_GGUF` is set runs
/// the CPU oracle and asserts against it.
#[test]
fn m4_fixture_replay_pos42() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("antirez")
        .join("m4_short_pos42_hc_slots.txt");
    let fixture = parse_fixture(&fixture_path);
    eprintln!(
        "fixture: prompt={} prompt_len={} pos={} slot_records={}",
        fixture.prompt,
        fixture.prompt_len,
        fixture.pos,
        fixture.slots.len()
    );
    assert_eq!(fixture.pos, 42);
    assert_eq!(fixture.prompt_len, 27);
    // 43 layers × 4 slots per layer (DS4 V4 Flash topology).
    assert_eq!(fixture.slots.len(), 43 * 4);

    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "DS4_GGUF unset — skipping replay (fixture parse OK). \
                 Set DS4_GGUF=/path/to/ds4.gguf to run the full assertion."
            );
            return;
        }
    };
    if !gguf_path.is_file() {
        eprintln!(
            "DS4_GGUF={} is not a regular file — skipping replay.",
            gguf_path.display()
        );
        return;
    }
    if cfg!(target_os = "macos")
        && std::env::var("DS4_FULL_FIXTURE_REPLAY_ON_MAC")
            .ok()
            .as_deref()
            != Some("1")
    {
        eprintln!(
            "macOS fixture parse OK — skipping full CPU replay because it materializes \
             multi-GB f32 tables. Set DS4_FULL_FIXTURE_REPLAY_ON_MAC=1 only on a machine \
             with enough memory."
        );
        return;
    }

    // Load the model. validate_ds4_layout + LayerViews::open both honour
    // mmap on Unix, so an 80 GiB GGUF doesn't slurp.
    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let composed = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights::from_views");

    // Embedding table (dequant once; reuse for every step).
    let embed_h = views
        .global
        .get("embed")
        .expect("global tensor: embed (token_embd.weight)");
    let embed_table = views.dequant_f32_simple(embed_h).expect("dequant embed");
    let d_model = manifest.d_model as usize;
    let vocab = manifest.vocab_size as usize;
    assert_eq!(embed_table.len(), vocab * d_model);

    let embed_of = |tok: i32| -> Vec<f32> {
        let t = tok as usize;
        assert!(t < vocab, "token {tok} out of vocab {vocab}");
        embed_table[t * d_model..(t + 1) * d_model].to_vec()
    };

    let cfg = DecodeConfig::default();
    let raw_cap: u32 = 2304; // matches antirez DS4_RAW_KV_ROWS default.
    let mut state = AttnStepState::new(&composed, raw_cap);

    let k = CpuDispatcher;
    let a = CpuAttentionDispatcher;

    // Arm the in-process HC recorder. Combined with `DS4_DUMP_HC_PER_SLOT=POS`
    // it fires only when state.pos == POS; without the env var the recorder
    // captures whatever pos we step through (we control that via the loop).
    // We do NOT set the env var here: setting it would also dump to stderr,
    // which the harness wants quiet. The recorder gate (state.pos == target)
    // is hit exactly once during the run — at the very last call.
    std::env::set_var("DS4_DUMP_HC_PER_SLOT", fixture.pos.to_string());
    let _ = arm_hc_slot_recorder();

    // Step through prompt + decode until state.pos == fixture.pos at the
    // *start* of the final call. decode_step_with_attn increments
    // state.pos at the end of each call, so 43 calls total drives pos
    // 0..=42 (the 43rd call enters with state.pos == 42).
    let total_calls: u32 = fixture.pos + 1;
    assert!(
        total_calls as usize >= fixture.prompt_token_ids.len(),
        "fixture.pos+1 ({}) < prompt_len ({})",
        total_calls,
        fixture.prompt_token_ids.len()
    );

    let mut last_logits: Vec<f32> = Vec::new();
    let mut last_tok: i32 = fixture.prompt_token_ids[0];
    for call_idx in 0..total_calls {
        let tok: i32 = if (call_idx as usize) < fixture.prompt_token_ids.len() {
            fixture.prompt_token_ids[call_idx as usize]
        } else {
            // Decode-time: feed back greedy argmax of last_logits.
            argmax(&last_logits)
        };
        CURRENT_TOKEN_HINT.with(|c| c.set(tok));
        let x = embed_of(tok);
        last_logits = decode_step_with_attn(&k, &a, x, &composed, &mut state, &cfg, raw_cap)
            .unwrap_or_else(|e| panic!("decode_step_with_attn @ call {call_idx} tok={tok}: {e}"));
        last_tok = tok;
    }
    let _ = last_tok; // last argmax intentionally unused — assertion is over cur_hc fingerprints.

    // Disarm and harvest samples.
    let samples = drain_hc_slot_recorder();
    std::env::remove_var("DS4_DUMP_HC_PER_SLOT");

    assert!(
        !samples.is_empty(),
        "HC recorder captured 0 samples — did state.pos ever hit {}? \
         Final state.pos={}",
        fixture.pos,
        state.pos
    );

    // Build (layer, slot) -> sample index for quick lookup.
    use std::collections::HashMap;
    let mut by_key: HashMap<(usize, usize), HcSlotSample> = HashMap::new();
    for s in samples.iter() {
        // We only want samples from the very last call (state.pos was bumped
        // to fixture.pos at the start of the final call). The thread-local
        // recorder accumulates across calls if multiple calls match the gate,
        // but with DS4_DUMP_HC_PER_SLOT=fixture.pos and our pos schedule the
        // gate fires exactly once. Defensive: keep the latest sample.
        by_key.insert((s.layer, s.slot), s.clone());
    }

    // Assertions. Print a digestible per-(layer, slot) report on mismatch
    // so a bisect immediately sees WHERE we drifted.
    const RMS_REL_TOL: f64 = 0.05; // 5 % per spec.
    let mut failures: Vec<String> = Vec::new();
    for fx in fixture.slots.iter() {
        let key = (fx.layer, fx.slot);
        let ours = match by_key.get(&key) {
            Some(s) => s,
            None => {
                failures.push(format!("MISSING L{} S{}", fx.layer, fx.slot));
                continue;
            }
        };
        // RMS tolerance: relative; protect against 0.
        let denom = fx.rms.abs().max(1e-9);
        let rel = (ours.rms - fx.rms).abs() / denom;
        if rel > RMS_REL_TOL {
            failures.push(format!(
                "RMS L{} S{}: ours={:.4} fixture={:.4} rel={:.3}",
                fx.layer, fx.slot, ours.rms, fx.rms, rel
            ));
        }
        // Top-2 dim agreement (top-1 + top-2 dims must be a subset of fixture's
        // top-3 — the rank of values inside a near-tie cluster is allowed to
        // drift but the *set* of dominant dims is the stronger fingerprint).
        let fixt_top_dims: [usize; 3] = [fx.top3[0].0, fx.top3[1].0, fx.top3[2].0];
        let our_top1 = ours.top3.first().map(|p| p.0).unwrap_or(usize::MAX);
        let our_top2 = ours.top3.get(1).map(|p| p.0).unwrap_or(usize::MAX);
        let agree1 = fixt_top_dims.contains(&our_top1);
        let agree2 = fixt_top_dims.contains(&our_top2);
        if !(agree1 && agree2) {
            failures.push(format!(
                "DIMS L{} S{}: ours_top2=[{},{}] not in fixture_top3=[{},{},{}]",
                fx.layer,
                fx.slot,
                our_top1,
                our_top2,
                fixt_top_dims[0],
                fixt_top_dims[1],
                fixt_top_dims[2]
            ));
        }
    }

    if !failures.is_empty() {
        eprintln!(
            "M4 fixture replay FAILED: {} mismatch(es) of {} (layer, slot) records",
            failures.len(),
            fixture.slots.len()
        );
        for f in failures.iter().take(32) {
            eprintln!("  {f}");
        }
        if failures.len() > 32 {
            eprintln!("  … and {} more", failures.len() - 32);
        }
        panic!(
            "M4 fixture replay diverged from antirez at pos={}",
            fixture.pos
        );
    }

    eprintln!(
        "M4 fixture replay PASS: {} (layer, slot) records within ±{:.0}% RMS at pos={}",
        fixture.slots.len(),
        RMS_REL_TOL * 100.0,
        fixture.pos
    );
}
