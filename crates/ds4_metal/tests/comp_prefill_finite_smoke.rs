//! FAST in-model finiteness smoke for COMP_PREFILL — one short prompt, single
//! prefill, asserts ON logits are finite. Used to bisect the NaN source
//! (DS4_COMP_PREFILL_NO_FILL toggles the pool-fill companion). Much faster than
//! the full integration test (one prefill, short prompt).
//!
//! DS4_GGUF=... [DS4_TEST_PROMPT_LEN=160 DS4_TEST_RAW_CAP=512 DS4_TEST_CHUNK=128
//!  DS4_COMP_PREFILL_NO_FILL=1]
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_metal::decode_runner::{DecodeRunner, DecodeSession};

#[test]
fn comp_prefill_in_model_finite() {
    if std::env::var("DS4_COMP_PREFILL_FINITE").is_err() {
        eprintln!("DS4_COMP_PREFILL_FINITE unset — skipping.");
        return;
    }
    let Ok(p) = std::env::var("DS4_GGUF") else { eprintln!("DS4_GGUF unset — skipping."); return; };
    let path = PathBuf::from(&p);
    if !path.is_file() { eprintln!("DS4_GGUF not a file — skipping."); return; }
    let n_prompt: i32 = std::env::var("DS4_TEST_PROMPT_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(160);
    let raw_cap: u32 = std::env::var("DS4_TEST_RAW_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let chunk: usize = std::env::var("DS4_TEST_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let prompt: Vec<i32> = (1..=n_prompt).collect();

    std::env::set_var("DS4_LEAN_WEIGHTS", "0");
    let runner = DecodeRunner::open(&path, raw_cap).expect("open");

    std::env::set_var("DS4_PREFILL_CHUNK", chunk.to_string());
    std::env::set_var("DS4_CHUNK_SWA_KFLASH", "1");
    std::env::set_var("DS4_CHUNK_COMP_PREFILL", "1");
    let mut s = DecodeSession::new(&runner);
    s.prefill(&prompt).expect("prefill");
    let logits = s.logits().to_vec();
    std::env::remove_var("DS4_PREFILL_CHUNK");
    std::env::remove_var("DS4_CHUNK_SWA_KFLASH");
    std::env::remove_var("DS4_CHUNK_COMP_PREFILL");
    std::env::remove_var("DS4_LEAN_WEIGHTS");

    let nan = logits.iter().filter(|v| !v.is_finite()).count();
    let no_fill = std::env::var("DS4_COMP_PREFILL_NO_FILL").ok().as_deref() == Some("1");
    eprintln!(
        "[comp_prefill_finite] prompt={n_prompt} raw_cap={raw_cap} chunk={chunk} no_fill={no_fill} \
         non_finite={nan}/{}",
        logits.len()
    );
    assert_eq!(nan, 0, "COMP_PREFILL produced {nan} non-finite logits");
    eprintln!("[comp_prefill_finite] PASS: ON logits finite");
}
