//! Phase 3 Step 4c — REAL-MODEL MTP drafter latency bench.
//!
//! Opens both base GGUF (DS4_GGUF) + MTP GGUF (DS4_MTP_GGUF), builds the
//! `MtpDraftBundle`, and measures `BatchScope::encode_mtp_draft` per-call
//! latency on real DS4-MTP weights with synthetic prev_hc + token_embed.
//!
//! This is the first end-to-end exercise of the full Phase 3 pipeline
//! (MTP loader → expert_weights → bundle → encode_mtp_draft) on real
//! data. Per-draft-token latency = the cost the verifier compares
//! against; combined with accept-rate (Step 5) it determines the spec-
//! decode tok/s win.
//!
//! Gated by `DS4_BENCH_MTP_DRAFT=1` + `DS4_GGUF` + `DS4_MTP_GGUF`.
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use ds4_engine::attn_dispatch::{DefaultsDs4, LayerParams};
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_engine::mtp::MtpWeights;
use ds4_metal::mtp_bundle::MtpDraftBundle;
use ds4_metal::MetalDispatcher;

/// Find lm_head tensor in a GGUF and extract its raw bytes via mmap slice.
/// MTP shares the base model's LM head, so this comes from the base file.
fn extract_lm_head_q8_bytes(
    base_gguf: &GgufFile,
    base_bytes: &[u8],
) -> anyhow::Result<Vec<u8>> {
    // Try common names: lm_head appears as "output.weight" in modern DS4 GGUFs.
    let lm_head_ti = base_gguf
        .tensors
        .iter()
        .find(|t| t.name == "output.weight" || t.name == "lm_head.weight")
        .ok_or_else(|| {
            anyhow::anyhow!("base GGUF missing lm_head tensor (output.weight / lm_head.weight)")
        })?;
    let n_elems: u64 = lm_head_ti.dims.iter().product();
    let nb_per_block = 34u64; // Q8_0
    anyhow::ensure!(n_elems % 32 == 0, "lm_head Q8_0 n_elems {} not %32", n_elems);
    let byte_len = (n_elems / 32) * nb_per_block;
    let start = (base_gguf.tensor_data_offset + lm_head_ti.offset) as usize;
    let end = start + byte_len as usize;
    anyhow::ensure!(
        end <= base_bytes.len(),
        "lm_head byte range [{}, {}) exceeds base GGUF size {}",
        start, end, base_bytes.len()
    );
    Ok(base_bytes[start..end].to_vec())
}

#[test]
fn encode_mtp_draft_real_model_latency_bench() {
    if std::env::var("DS4_BENCH_MTP_DRAFT").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MTP_DRAFT unset — skipping. \
                   Set =1 + DS4_GGUF + DS4_MTP_GGUF to run.");
        return;
    }
    let base_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    let mtp_path = match std::env::var("DS4_MTP_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_MTP_GGUF unset — skipping"); return; }
    };
    if !base_path.is_file() || !mtp_path.is_file() {
        eprintln!("base or MTP GGUF not found — skipping");
        return;
    }

    eprintln!("loading base GGUF: {}", base_path.display());
    let manifest = validate_ds4_layout(&base_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&base_path, manifest.n_layers).expect("LayerViews");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let base_model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights");
    eprintln!(
        "base model: d_model={} vocab={} n_layers={}",
        base_model.d_model, base_model.vocab_size, base_model.layers.len()
    );

    let base_gguf = GgufFile::open(&base_path).expect("GgufFile base");
    let lm_head_q8_bytes = extract_lm_head_q8_bytes(&base_gguf, views.bytes.as_ref())
        .expect("extract lm_head bytes");
    eprintln!("base lm_head Q8_0 bytes: {} MB",
        lm_head_q8_bytes.len() as f64 / 1e6);

    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    disp.load_expert_weights(&base_gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights (base)");

    eprintln!("loading MTP GGUF: {}", mtp_path.display());
    let mtp_gguf = GgufFile::open(&mtp_path).expect("GgufFile MTP");
    let mtp_bytes = std::fs::read(&mtp_path).expect("read MTP bytes");
    let mtp_layer_idx = disp
        .load_mtp_expert_weights(&mtp_gguf, &mtp_bytes)
        .expect("load_mtp_expert_weights");
    eprintln!("loaded MTP at slot {}", mtp_layer_idx);

    let mtp = MtpWeights::from_gguf(&mtp_gguf).expect("MtpWeights::from_gguf");
    eprintln!("MtpWeights: n_experts={} d_ffn={}", mtp.n_experts, mtp.d_ffn);

    // Build the bundle.
    let t_bundle = Instant::now();
    let bundle = MtpDraftBundle::from_mtp(&mtp, &mtp_bytes, lm_head_q8_bytes, base_model.vocab_size)
        .expect("MtpDraftBundle::from_mtp");
    eprintln!(
        "MtpDraftBundle built in {:.2}s (n_embd={} n_hc={} d_ffn={} shared_dim={})",
        t_bundle.elapsed().as_secs_f64(),
        bundle.shape.n_embd, bundle.shape.n_hc, bundle.shape.d_ffn, bundle.shape.shared_dim,
    );

    // Synthetic prev_hc + token_embed (random; we're measuring kernel
    // latency, not correctness).
    let n_embd = bundle.shape.n_embd;
    let n_hc = bundle.shape.n_hc;
    let hc_dim = n_embd * n_hc;
    let mut rng: u32 = 0xCAFE_BABE;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
    let prev_hc_data: Vec<f32> = (0..hc_dim)
        .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
    let token_embed_data: Vec<f32> = (0..n_embd)
        .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

    // Layer params for the drafter — borrow from the FIRST base layer.
    // The MTP block shares the same shape constraints as a base decode
    // layer (rope freq, etc.). antirez uses the same DS4 V4 Flash
    // architectural constants.
    let layer_params: LayerParams = base_model.layers[0].attn.params.clone();
    let raw_cap: u32 = 256;     // MTP raw cache window
    let base_slot: u32 = 0;
    let base_pos: u32 = 0;

    // Warm up + bench.
    let warmup_iters: usize = 2;
    let bench_iters: usize = std::env::var("DS4_BENCH_MTP_DRAFT_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let mut iter_ms: Vec<f64> = Vec::with_capacity(warmup_iters + bench_iters);

    eprintln!("\nbenching encode_mtp_draft: warmup={} iters={}", warmup_iters, bench_iters);
    for i in 0..(warmup_iters + bench_iters) {
        let mut scope = disp.batch_scope();
        scope.set_q8_proj(true);

        let prev_hc_db = scope.upload_f32(&prev_hc_data);
        let token_embed_db = scope.upload_f32(&token_embed_data);
        let dbufs = bundle.upload_to_scope(&scope);

        let t0 = Instant::now();
        let logits = scope.encode_mtp_draft(
            &prev_hc_db, &token_embed_db,
            dbufs.input_mix(), dbufs.layer(), dbufs.output_head(),
            bundle.shape, &layer_params,
            mtp_layer_idx, raw_cap, base_slot, base_pos,
            /*attn_base_pos=*/0, // single-draft bench: MTP cache empty at iter 0
        ).expect("encode_mtp_draft");
        let result = scope.flush_and_read(&logits);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        iter_ms.push(elapsed_ms);

        if i == warmup_iters {
            assert_eq!(result.len(), bundle.shape.vocab, "logits shape");
            let nan_count = result.iter().filter(|v| !v.is_finite()).count();
            let max_abs = result.iter().filter(|v| v.is_finite())
                .fold(0.0f32, |a, &v| a.max(v.abs()));
            let max_v = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_v = result.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "    first-bench logits: shape={} nan_count={} max_abs={:.3e} range=[{:.3e}, {:.3e}]",
                result.len(), nan_count, max_abs, min_v, max_v
            );
        }
    }

    let bench_only = &iter_ms[warmup_iters..];
    let mean = bench_only.iter().copied().sum::<f64>() / bench_only.len() as f64;
    let min = bench_only.iter().copied().fold(f64::INFINITY, f64::min);
    let max = bench_only.iter().copied().fold(0.0f64, f64::max);
    eprintln!("\nREAL-MODEL encode_mtp_draft latency:");
    eprintln!("  mean = {:.3} ms   [min={:.3}  max={:.3}]   drafts/sec ≈ {:.1}",
        mean, min, max, 1000.0 / mean);
    eprintln!(
        "\nProjection: at K=8 spec-decode, K-1 sequential drafts + 1 verifier pass:\n\
         - K=8 drafter cost: 7 × {:.3} = {:.3} ms\n\
         - At measured fused-K verifier 4.13 ms × 43 layers = 178 ms per K=8 batch\n\
         - Drafter is {:.1}% of K=8 forward (drafter / (drafter + verifier))",
        mean, 7.0 * mean,
        (7.0 * mean) / (7.0 * mean + 178.0) * 100.0
    );
}
