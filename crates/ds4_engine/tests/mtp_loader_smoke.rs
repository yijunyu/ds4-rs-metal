//! Phase 3 — MTP drafter weights loader smoke test.
//!
//! Opens the MTP GGUF (download via
//! `benchmarks/ds4_msl/upstream/ds4/download_model.sh mtp`, default location
//! `benchmarks/ds4_msl/upstream/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`)
//! and validates all 32 tensors load with correct shapes + types per
//! `mtp_weights_validate_layout` (ds4.c:2207).
//!
//! Gated by `DS4_MTP_GGUF` env var pointing at the MTP file. Skips
//! gracefully if unset or missing (no MTP file on disk).

use std::path::PathBuf;

use ds4_engine::mtp::MtpWeights;

#[test]
fn mtp_loader_validates_all_tensors() {
    let path = match std::env::var("DS4_MTP_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "DS4_MTP_GGUF unset — skipping MTP loader smoke. Download via:\n  \
                 cd benchmarks/ds4_msl/upstream/ds4 && ./download_model.sh mtp\n  \
                 then re-run with DS4_MTP_GGUF=/path/to/.../DeepSeek-V4-Flash-MTP-*.gguf"
            );
            return;
        }
    };
    if !path.is_file() {
        eprintln!("DS4_MTP_GGUF={} is not a regular file — skipping", path.display());
        return;
    }

    eprintln!("loading MTP GGUF: {}", path.display());
    let w = MtpWeights::from_path(&path)
        .unwrap_or_else(|e| panic!("MtpWeights::from_path failed: {:#}", e));

    eprintln!("MTP block: n_experts={} d_ffn={}", w.n_experts, w.d_ffn);
    eprintln!("MTP tensors (32):");
    for (name, t) in [
        ("e_proj",          &w.e_proj),
        ("h_proj",          &w.h_proj),
        ("enorm",           &w.enorm),
        ("hnorm",           &w.hnorm),
        ("norm",            &w.norm),
        ("hc_head_base",    &w.hc_head_base),
        ("hc_head_fn",      &w.hc_head_fn),
        ("hc_head_scale",   &w.hc_head_scale),
        ("hc_attn_fn",      &w.hc_attn_fn),
        ("hc_attn_scale",   &w.hc_attn_scale),
        ("hc_attn_base",    &w.hc_attn_base),
        ("attn_norm",       &w.attn_norm),
        ("attn_q_a",        &w.attn_q_a),
        ("attn_q_a_norm",   &w.attn_q_a_norm),
        ("attn_q_b",        &w.attn_q_b),
        ("attn_kv",         &w.attn_kv),
        ("attn_kv_a_norm",  &w.attn_kv_a_norm),
        ("attn_sinks",      &w.attn_sinks),
        ("attn_output_a",   &w.attn_output_a),
        ("attn_output_b",   &w.attn_output_b),
        ("hc_ffn_fn",       &w.hc_ffn_fn),
        ("hc_ffn_scale",    &w.hc_ffn_scale),
        ("hc_ffn_base",     &w.hc_ffn_base),
        ("ffn_norm",        &w.ffn_norm),
        ("ffn_gate_inp",    &w.ffn_gate_inp),
        ("ffn_exp_probs_b", &w.ffn_exp_probs_b),
        ("ffn_gate_exps",   &w.ffn_gate_exps),
        ("ffn_up_exps",     &w.ffn_up_exps),
        ("ffn_down_exps",   &w.ffn_down_exps),
        ("ffn_gate_shexp",  &w.ffn_gate_shexp),
        ("ffn_up_shexp",    &w.ffn_up_shexp),
        ("ffn_down_shexp",  &w.ffn_down_shexp),
    ] {
        eprintln!(
            "  {:<18} {:>10?} {:?}   @ +{}",
            name, t.info.ttype, t.info.dims, t.abs_offset,
        );
    }

    // Phase 3 Step 4a — exercise the new data-access helpers.
    // dequant_f32 covers F32/F16/BF16/Q8_0; raw_bytes covers everything.
    let mtp_bytes = std::fs::read(&path).expect("read MTP GGUF bytes");
    eprintln!("\nPhase 3 Step 4a — MtpTensor data-access helpers:");

    // F32 norm (small, dequantizes to mostly 1.0-ish floats).
    let enorm_vec = w.enorm.dequant_f32(&mtp_bytes).expect("enorm dequant");
    let enorm_mean: f32 = enorm_vec.iter().sum::<f32>() / enorm_vec.len() as f32;
    eprintln!("  enorm.dequant_f32: {} floats  mean={:.4}", enorm_vec.len(), enorm_mean);
    assert_eq!(enorm_vec.len(), 4096, "enorm should have n_embd=4096 floats");
    assert!(enorm_vec.iter().all(|v| v.is_finite()), "enorm has non-finite");

    // Q8_0 dequant: e_proj is 4096*4096 = 16.7M elements → 17.8 MB Q8_0 bytes.
    let e_proj_vec = w.e_proj.dequant_f32(&mtp_bytes).expect("e_proj dequant");
    eprintln!("  e_proj.dequant_f32: {} floats", e_proj_vec.len());
    assert_eq!(e_proj_vec.len(), 4096 * 4096);
    let e_proj_nonzero = e_proj_vec.iter().filter(|v| **v != 0.0).count();
    eprintln!("  e_proj nonzero count: {}/{}", e_proj_nonzero, e_proj_vec.len());
    assert!(e_proj_nonzero > e_proj_vec.len() / 2, "e_proj mostly nonzero expected");

    // Raw bytes for Q8_0 path (used by BatchScope::weight_q8_0_raw).
    let e_proj_raw = w.e_proj.raw_bytes(&mtp_bytes).expect("e_proj raw");
    eprintln!("  e_proj.raw_bytes: {} bytes", e_proj_raw.len());
    assert_eq!(e_proj_raw.len(), (4096 * 4096 / 32) * 34, "Q8_0 byte size mismatch");

    // F16 plain tensor (hc_attn_fn or hc_head_fn).
    let hc_head_fn_vec = w.hc_head_fn.dequant_f32(&mtp_bytes).expect("hc_head_fn dequant");
    eprintln!("  hc_head_fn.dequant_f32: {} floats", hc_head_fn_vec.len());
    assert_eq!(hc_head_fn_vec.len() as u64, w.hc_head_fn.n_elements());
    assert!(hc_head_fn_vec.iter().all(|v| v.is_finite()), "hc_head_fn has non-finite");

    // BF16 tensor — try ffn_gate_inp (router weights, plain F16/BF16/F32).
    let router_vec = w.ffn_gate_inp.dequant_f32(&mtp_bytes).expect("ffn_gate_inp dequant");
    eprintln!("  ffn_gate_inp.dequant_f32: {} floats  ttype={:?}",
        router_vec.len(), w.ffn_gate_inp.info.ttype);
    assert!(router_vec.iter().all(|v| v.is_finite()));

    // Q4_K expert tensor — dequant should bail (routed via load_mtp_expert_weights).
    let q4k_result = w.ffn_gate_exps.dequant_f32(&mtp_bytes);
    assert!(q4k_result.is_err(), "Q4_K dequant should bail");
    eprintln!("  ffn_gate_exps.dequant_f32: correctly bailed for Q4_K");

    // raw_bytes works for any quant.
    let q4k_raw = w.ffn_gate_exps.raw_bytes(&mtp_bytes).expect("ffn_gate_exps raw");
    eprintln!("  ffn_gate_exps.raw_bytes: {} bytes (Q4_K)", q4k_raw.len());

    eprintln!("\nAll MtpTensor helpers OK.");
}
