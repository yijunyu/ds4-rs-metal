//! Phase 3 Step 5.2 — end-to-end MTP spec-decode runner test.
//!
//! Loads base + MTP GGUFs, prefills a small synthetic prompt, runs N
//! spec-decode iters (drafter → verifier → accept), reports tok/s +
//! accept rate. This is the single-step measurement that produces the
//! canonical end-to-end number the K-amortization work has been
//! building toward.
//!
//! Gated by `DS4_BENCH_SPEC=1` + `DS4_GGUF` + `DS4_MTP_GGUF`. Skips
//! gracefully if any unset.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgmlType, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_engine::mtp::MtpWeights;
use ds4_metal::base_run_bundle::upload_base_layers_to_scope;
use ds4_metal::mtp_bundle::{run_mtp_chain_drafts, MtpDraftBundle};
use ds4_metal::spec_decode::{
    accept_longest_prefix_greedy, prefill_to_residual,
};
use ds4_metal::MetalDispatcher;

/// Build the BPE vocab from the GGUF tokenizer metadata and encode `text`.
/// Prepends BOS (tokenizer.ggml.bos_token_id) if present, mirroring antirez's
/// prompt encoding. Returns token ids as i32.
fn build_vocab_and_encode(g: &GgufFile, text: &str) -> Result<Vec<i32>> {
    use ds4_engine::gguf::MetaValue;
    use ds4_engine::tokenizer::Vocab;

    let str_array = |key: &str| -> Result<Vec<String>> {
        match g.get_meta(key) {
            Some(MetaValue::Array { values, .. }) => Ok(values.iter().filter_map(|v| {
                if let MetaValue::String(s) = v { Some(s.clone()) } else { None }
            }).collect()),
            _ => bail!("GGUF missing string-array metadata: {}", key),
        }
    };
    let tokens = str_array("tokenizer.ggml.tokens")?;
    // Merges are optional for some vocabs; tolerate absence.
    let merges = str_array("tokenizer.ggml.merges").unwrap_or_default();
    anyhow::ensure!(!tokens.is_empty(), "empty tokenizer.ggml.tokens");

    let vocab = Vocab::new(tokens, merges);
    let mut ids: Vec<i32> = vocab.encode(text).into_iter().map(|t| t as i32).collect();

    // Prepend BOS if the GGUF declares one.
    if let Some(bos) = g.get_meta("tokenizer.ggml.bos_token_id") {
        let bos_id = match bos {
            MetaValue::U32(v) => Some(*v as i32),
            MetaValue::I32(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as i32),
            _ => None,
        };
        if let Some(b) = bos_id { ids.insert(0, b); }
    }
    anyhow::ensure!(!ids.is_empty(), "tokenizer produced no tokens for prompt");
    Ok(ids)
}

fn extract_lm_head_q8_bytes(g: &GgufFile, bytes: &[u8]) -> Result<Vec<u8>> {
    let ti = g.tensors.iter()
        .find(|t| t.name == "output.weight" || t.name == "lm_head.weight")
        .ok_or_else(|| anyhow!("base GGUF missing output.weight/lm_head.weight"))?;
    let n_elems: u64 = ti.dims.iter().product();
    if n_elems % 32 != 0 { bail!("lm_head n_elems {} not %32", n_elems); }
    let byte_len = (n_elems / 32) * 34;
    let start = (g.tensor_data_offset + ti.offset) as usize;
    Ok(bytes[start..start + byte_len as usize].to_vec())
}

fn extract_token_embd_f32(g: &GgufFile, bytes: &[u8], vocab: usize, d_model: usize) -> Result<Vec<f32>> {
    let ti = g.tensors.iter()
        .find(|t| t.name == "token_embd.weight")
        .ok_or_else(|| anyhow!("base GGUF missing token_embd.weight"))?;
    let n_elems = (vocab * d_model) as u64;
    let actual: u64 = ti.dims.iter().product();
    if actual != n_elems {
        bail!("token_embd dims {:?} != vocab*d_model = {}*{}", ti.dims, vocab, d_model);
    }
    let start = (g.tensor_data_offset + ti.offset) as usize;
    match ti.ttype {
        GgmlType::F32 => {
            let end = start + (n_elems * 4) as usize;
            let mut out = vec![0.0f32; n_elems as usize];
            for (i, c) in bytes[start..end].chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let end = start + (n_elems * 2) as usize;
            let mut out = vec![0.0f32; n_elems as usize];
            for (i, c) in bytes[start..end].chunks_exact(2).enumerate() {
                out[i] = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            Ok(out)
        }
        other => bail!("token_embd: unhandled ttype {:?}", other),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 { if mant == 0 { 0.0 } else { (mant as f32) * 2f32.powi(-24) } }
            else if exp == 31 { if mant == 0 { f32::INFINITY } else { f32::NAN } }
            else { (mant as f32 + 1024.0) * 2f32.powi(exp as i32 - 25) };
    if sign != 0 { -f } else { f }
}

#[test]
fn spec_decode_real_step_bench() {
    if std::env::var("DS4_BENCH_SPEC").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_SPEC unset — skipping. \
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
        eprintln!("base or MTP GGUF not found — skipping"); return;
    }

    let result = (|| -> Result<()> {
        eprintln!("loading base GGUF: {}", base_path.display());
        let manifest = validate_ds4_layout(&base_path)?;
        let views = LayerViews::open(&base_path, manifest.n_layers)?;
        let defaults = DefaultsDs4::ds4_v4_flash();
        let model = ComposedModelWeights::from_views(&views, &manifest, defaults)?;
        eprintln!("base loaded: d_model={} vocab={} n_layers={}",
            model.d_model, model.vocab_size, model.layers.len());

        let base_gguf = GgufFile::open(&base_path)?;
        let lm_head_q8 = extract_lm_head_q8_bytes(&base_gguf, views.bytes.as_ref())?;
        let token_embd = extract_token_embd_f32(
            &base_gguf, views.bytes.as_ref(), model.vocab_size, model.d_model,
        )?;
        eprintln!("extracted lm_head Q8 ({:.1} MB), token_embd f32 ({:.1} MB)",
            lm_head_q8.len() as f64 / 1e6,
            (token_embd.len() * 4) as f64 / 1e6);

        let mut disp = MetalDispatcher::new()?;
        disp.load_expert_weights(&base_gguf, views.bytes.as_ref(), manifest.n_layers)?;

        let mtp_gguf = GgufFile::open(&mtp_path)?;
        let mtp_bytes = std::fs::read(&mtp_path)?;
        let mtp_slot = disp.load_mtp_expert_weights(&mtp_gguf, &mtp_bytes)?;
        let mtp = MtpWeights::from_gguf(&mtp_gguf)?;
        let bundle = MtpDraftBundle::from_mtp(&mtp, &mtp_bytes, lm_head_q8, model.vocab_size)?;
        eprintln!("MTP loaded at slot {} (n_experts={} d_ffn={})",
            mtp_slot, mtp.n_experts, mtp.d_ffn);

        // Prompt: real text (DS4_BENCH_SPEC_PROMPT="some text") tokenized via the
        // GGUF BPE vocab, else a synthetic [0..N) prompt (DS4_BENCH_SPEC_PROMPT_LEN,
        // default 16) for fast diagnostic runs. Real text is REQUIRED for a
        // meaningful accept rate — the synthetic prompt is incoherent so the MTP
        // drafter degenerates and only drafts[0] ever matches (~20% ceiling).
        let prompt_len: i32 = std::env::var("DS4_BENCH_SPEC_PROMPT_LEN")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let prompt_tokens: Vec<i32> = match std::env::var("DS4_BENCH_SPEC_PROMPT") {
            Ok(text) if !text.is_empty() => {
                let toks = build_vocab_and_encode(&base_gguf, &text)?;
                eprintln!("tokenized prompt ({} chars) → {} tokens: {:?}",
                    text.len(), toks.len(),
                    &toks[..toks.len().min(24)]);
                anyhow::ensure!(
                    toks.iter().all(|&t| (t as usize) < model.vocab_size),
                    "tokenizer produced out-of-vocab id",
                );
                toks
            }
            _ => (0..prompt_len).collect(),
        };
        anyhow::ensure!(!prompt_tokens.is_empty(), "empty prompt");
        let raw_cap: u32 = 256;
        eprintln!("\nprefilling {}-token prompt...", prompt_tokens.len());
        let t_prefill = Instant::now();
        let (prev_hc_single, mut prefill_state) = prefill_to_residual(
            &disp, &model, &token_embd, &prompt_tokens, raw_cap,
        )?;
        eprintln!("prefill: {} tokens in {:.2}s ({:.1} tok/s)",
            prompt_tokens.len(),
            t_prefill.elapsed().as_secs_f64(),
            prompt_tokens.len() as f64 / t_prefill.elapsed().as_secs_f64());

        // CRITICAL: seed the GPU persistent KV cache with the prefill prefix.
        // The CPU decode path keeps KV in prefill_state.kv_storage; the
        // K-position verifier reads the GPU persistent buffer (kv_buffer_or_alloc)
        // — without this sync it attends over a ZERO prefix → garbage logits.
        // Populate each slot via kv_fp8_store_persistent (the SAME FP8+f16 store
        // the verifier's current-slot path uses) so the prefix is in the
        // verifier's exact format (the raw f16-only copy left it inconsistent
        // with the FP8-snapped current slot → residual after_attn divergence).
        // Seed GPU persistent KV from the CPU mirror for slots [from, upto).
        // Seeding only the NEW range each iter (not [0, base_pos) every time)
        // avoids an O(n²) re-seed of the whole growing prefix — that redundant
        // full re-seed was the cause of the per-iter time climbing 245ms→1700ms
        // within a run. Prior slots are already correct on GPU (verifier-written
        // + back-synced), so re-storing them is pure waste.
        let populate_kv_fp8 = |disp: &MetalDispatcher, state: &ds4_engine::decode_step::AttnStepState, from: u32, upto: u32| {
            for (li, layer) in model.layers.iter().enumerate() {
                let p = &layer.attn.params;
                let row = p.n_lora_kv as usize;
                let kv = &state.kv_storage[li];
                for slot in from..upto {
                    let s = slot as usize;
                    let _ = disp.kv_fp8_store_persistent(
                        li as u32, p, &kv[s * row..(s + 1) * row], raw_cap, slot,
                    );
                }
            }
        };
        // (prefill_to_residual now back-syncs GPU→CPU KV internally for the
        // fused path, so prefill_state.kv_storage is populated for both paths.)
        populate_kv_fp8(&disp, &prefill_state, 0, prefill_state.pos);

        let mut base_pos = prefill_state.pos;
        let mut base_slot = base_pos % raw_cap;
        let mut last_token = *prompt_tokens.last().unwrap();
        let mut prev_hc = prev_hc_single;
        let mut mtp_pos = base_pos;
        let mut mtp_slot_cursor = mtp_pos % raw_cap;
        // MTP has its own raw KV cache (antirez ds4.c:7895 `mtp_n_raw`).
        // 0 at session start (never prefilled — drafter relies on prev_hc
        // for base context).  Each draft commits +1 row; on accept-N we
        // keep mtp_base_raw + N (line 16148).  Per-iter base value:
        let mut mtp_n_raw: u32 = 0;

        // The drafter uses the SAME LayerParams as base layer 0 (DS4 V4
        // Flash MTP block mirrors base layer-0 attention shape).
        let mtp_layer_params = model.layers[0].attn.params.clone();
        let n_iters: usize = std::env::var("DS4_BENCH_SPEC_ITERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        // K=4 is the sweet spot on diverse real text. E2E A/B (Roman-history
        // prompt, fused glue): K=4 = 8.07 tok/s @ 40.6% accept vs K=8 = 5.27
        // tok/s @ 20.3%. Both accept the SAME 26 tokens — K=8 burns 2× the
        // draft slots for nothing because the 1-layer MTP drafter collapses
        // past draft[2-3], so the extra verify cost is pure loss. Override
        // with DS4_BENCH_SPEC_K; supported K ∈ {1,2,4,8} (K=3 panics matvec_k).
        let k: usize = std::env::var("DS4_BENCH_SPEC_K")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        std::env::set_var("DS4_MOE_K_PATH", "fused");

        eprintln!("\nspec-decode loop: K={} iters={} (DS4_MOE_K_PATH=fused)", k, n_iters);
        let mut n_drafted: usize = 0;
        let mut n_accepted: usize = 0;
        let mut emitted: Vec<i32> = Vec::new();

        // ===== FAITHFULNESS CHECK (DS4_FAITHFUL_CHECK) =====
        // A faithful verifier emits EXACTLY the base greedy sequence. Run a pure
        // K=1 greedy decode (decode_step WITH compressor — the default — + the
        // base output head, the same encode_mtp_output_head projection logits_0
        // uses) from the post-prefill state, then compare to the spec-decode
        // `emitted` after the loop. Run BEFORE the loop (GPU compressor pools are
        // at the post-prefill state here); restore the pools from the unmutated
        // prefill_state CPU mirror afterward so the spec-decode loop is unaffected.
        // Carries (base-greedy token seq, per-step base residual). ref_resid[k]
        // = decode_step(ref_seq[k]).cur_hc = the residual predicting position
        // base_pos+k+1 — the SAME thing verifier row k should produce (when
        // ref_seq[k] == drafts[k]). Used for the row-2 bisection: comparing
        // cur_hc_k row k vs ref_resid[k] splits "verifier layer-chain residual
        // diverges" from "verifier output head diverges".
        // 3rd element: decode_step's FIRST in-batch comp emit row per ratio==4
        // layer (= comp_kv_ring[prefill_n_comp]), for the emit-row comparison
        // (DS4_COMPARE_EMIT) against the verifier's GPU ring.
        let faithful_ref: Option<(Vec<i32>, Vec<Vec<f32>>, std::collections::HashMap<usize, Vec<f32>>)> = if std::env::var("DS4_FAITHFUL_CHECK").is_ok() {
            use ds4_engine::decode_step::{
                decode_step_with_attn_to_residual, DecodeConfig, sample_argmax,
            };
            let cfg = DecodeConfig::default();
            let shp = bundle.shape;
            let unit_gamma = vec![1.0f32; shp.n_hc * model.d_model];
            let n_ref = n_iters * k; // upper bound on emitted length
            let mut rs = prefill_state.clone();
            let mut cur_hc = prev_hc.clone();
            let mut ref_seq: Vec<i32> = Vec::with_capacity(n_ref);
            let mut ref_resid: Vec<Vec<f32>> = Vec::with_capacity(n_ref);
            for i in 0..n_ref {
                // project cur_hc → base logits → argmax (predicts pos base_pos+i)
                let next = {
                    let sc = disp.batch_scope();
                    sc.set_q8_proj(true);
                    let phc = sc.upload_f32(&cur_hc);
                    let ug = sc.weight_f32(&unit_gamma);
                    let fn_db = sc.weight_f32(model.output_hc_fn.as_ref().unwrap());
                    let sc_db = sc.weight_f32(model.output_hc_scale.as_ref().unwrap());
                    let ba_db = sc.weight_f32(model.output_hc_base.as_ref().unwrap());
                    let fnorm = sc.weight_f32(&model.final_norm_gamma);
                    let lm = sc.weight_q8_0_raw(&bundle.base_lm_head_q8,
                        model.d_model * model.vocab_size);
                    let mut sc = sc;
                    let logits_db = sc.encode_mtp_output_head(
                        &phc, &ug, &fn_db, &sc_db, &ba_db, &fnorm, &lm,
                        model.d_model, shp.n_hc, model.vocab_size,
                        shp.rms_eps, shp.hc_eps,
                    )?;
                    let logits = sc.flush_and_read(&logits_db);
                    sample_argmax(&logits) as i32
                };
                ref_seq.push(next);
                // advance: decode_step(next) at pos base_pos+i → cur_hc for next
                let pos = base_pos + i as u32;
                rs.pos = pos;
                for kp in rs.kv_pos.iter_mut() { *kp = pos % raw_cap; }
                let es = (next as usize) * model.d_model;
                let embed = token_embd[es..es + model.d_model].to_vec();
                decode_step_with_attn_to_residual(
                    &disp, &disp, embed, &model, &mut rs, &cfg, raw_cap,
                )?;
                cur_hc = rs.cur_hc.clone();
                // residual predicting base_pos+i+1 (= verifier row i's cur_hc).
                ref_resid.push(cur_hc.clone());
            }
            eprintln!("[FAITHFUL] base greedy ref ({} toks) = {:?}", ref_seq.len(), ref_seq);
            // Capture decode_step's first in-batch comp emit per ratio==4 layer
            // (= comp_kv_ring row at index prefill n_comp).
            let mut ref_emit: std::collections::HashMap<usize, Vec<f32>> =
                std::collections::HashMap::new();
            for (li, layer) in model.layers.iter().enumerate() {
                let p = &layer.attn.params;
                if p.compress_ratio == 4 {
                    let hd = p.head_dim as usize;
                    let nprev = prefill_state.n_comp[li] as usize;
                    let ring = &rs.comp_kv_ring[li];
                    if std::env::var("DS4_COMPARE_EMIT").is_ok() {
                        eprintln!("[emit-cap] layer {:2}: prefill_n_comp={} rs.n_comp={} ring_rows={}",
                            li, nprev, rs.n_comp[li], ring.len() / hd);
                    }
                    if ring.len() >= (nprev + 1) * hd {
                        ref_emit.insert(li, ring[nprev * hd..(nprev + 1) * hd].to_vec());
                    }
                }
            }
            // Restore the GPU compressor pools + comp ring to the post-prefill
            // state (the reference greedy mutated them) so the spec-decode loop's
            // decode_step advance is unaffected.
            for (li, layer) in model.layers.iter().enumerate() {
                let p = &layer.attn.params;
                if p.compress_ratio == 4 {
                    disp.populate_compressor_state(
                        li as u32, &prefill_state.comp_state_kv[li],
                        &prefill_state.comp_state_score[li], false,
                    );
                    disp.populate_compressor_state(
                        li as u32, &prefill_state.index_state_kv[li],
                        &prefill_state.index_state_score[li], true,
                    );
                    if prefill_state.n_comp[li] > 0 {
                        disp.populate_comp_ring(
                            li as u32, &prefill_state.comp_kv_ring[li],
                            raw_cap as usize, p.head_dim as usize,
                        );
                    }
                }
            }
            // EMIT-ROW STEP (ii): capture decode_step's layer-DS4_DUMP_NORMED_LAYER
            // normed for the pos-(base_pos+1) token (= the in-batch emit row's
            // draft, ref_seq[1]), to compare vs the verifier's normed_k[row1]. A
            // focused 2-token decode (ref_seq[0]@base_pos then ref_seq[1]@base_pos+1)
            // on a clone sets the same KV the verifier row 1 attends. normed is
            // pre-compressor, so the GPU pool state is irrelevant here.
            if let Some(dl) = std::env::var("DS4_DUMP_RESID_LAYER").ok()
                .or_else(|| std::env::var("DS4_DUMP_NORMED_LAYER").ok())
                .or_else(|| std::env::var("DS4_DUMP_FLASH_LAYER").ok())
                .and_then(|s| s.parse::<usize>().ok())
            {
                use ds4_engine::decode_step::{NORMED_CAPTURE, LAYER_RESID_CAPTURE};
                use ds4_engine::attn_dispatch::{FLASH_CAPTURE, ATTN_OUT_CAPTURE};
                let prn = |label: &str, tok: i32, pos: u32, v: &[f32]| {
                    if v.len() >= 6 {
                        let rms = (v.iter().map(|&x| (x as f64) * (x as f64))
                            .sum::<f64>() / v.len() as f64).sqrt();
                        eprintln!("[{}] layer {} token{}@pos{}: rms={:.4e} head={:?}",
                            label, dl, tok, pos, rms, &v[..6]);
                    }
                };
                let mut rs2 = prefill_state.clone();
                for (j, &tok) in ref_seq.iter().take(2).enumerate() {
                    let pos = base_pos + j as u32;
                    rs2.pos = pos;
                    for kp in rs2.kv_pos.iter_mut() { *kp = pos % raw_cap; }
                    // All row-1 (ref_seq[1]=6397, j==1) captures, to match the
                    // verifier's row-1 flash/attn_out/after_attn dumps.
                    if j == 1 {
                        FLASH_CAPTURE.with(|c| *c.borrow_mut() = (dl, Vec::new()));
                        ATTN_OUT_CAPTURE.with(|c| *c.borrow_mut() = (dl, Vec::new()));
                        NORMED_CAPTURE.with(|c| *c.borrow_mut() = (dl, Vec::new()));
                        LAYER_RESID_CAPTURE.with(|c| *c.borrow_mut() = (dl, Vec::new(), Vec::new()));
                    }
                    let es = (tok as usize) * model.d_model;
                    let embed = token_embd[es..es + model.d_model].to_vec();
                    decode_step_with_attn_to_residual(
                        &disp, &disp, embed, &model, &mut rs2, &cfg, raw_cap,
                    )?;
                }
                let p1 = base_pos + 1;
                let t1 = ref_seq[1];
                let ds_flash = FLASH_CAPTURE.with(|c| c.borrow().1.clone());
                FLASH_CAPTURE.with(|c| *c.borrow_mut() = (usize::MAX, Vec::new()));
                prn("ds-flash", t1, p1, &ds_flash);
                let ds_attnout = ATTN_OUT_CAPTURE.with(|c| c.borrow().1.clone());
                ATTN_OUT_CAPTURE.with(|c| *c.borrow_mut() = (usize::MAX, Vec::new()));
                prn("ds-attnout", t1, p1, &ds_attnout);
                let ds_normed = NORMED_CAPTURE.with(|c| c.borrow().1.clone());
                NORMED_CAPTURE.with(|c| *c.borrow_mut() = (usize::MAX, Vec::new()));
                prn("ds-normed", t1, p1, &ds_normed);
                let (ds_aa, ds_af) = LAYER_RESID_CAPTURE.with(|c| {
                    let b = c.borrow(); (b.1.clone(), b.2.clone())
                });
                LAYER_RESID_CAPTURE.with(|c| *c.borrow_mut() = (usize::MAX, Vec::new(), Vec::new()));
                prn("ds-afterattn", t1, p1, &ds_aa);
                prn("ds-afterffn", t1, p1, &ds_af);
            }
            Some((ref_seq, ref_resid, ref_emit))
        } else {
            None
        };

        // Upload the 43 base layers' weights ONCE — they're read-only constants
        // (no cb writes them), and the underlying GPU buffers are globally cached
        // on the dispatcher, so the returned DeferredBufs (plain buffer handles)
        // stay valid across every iter's scope. Re-uploading per iter cost ~42ms
        // of redundant wrapper/hashmap rebuild (profiled "upload=42ms"). One
        // persistent scope holds them for the loop's lifetime.
        let weights_scope = disp.batch_scope();
        weights_scope.set_q8_proj(true);
        let owned_layers = upload_base_layers_to_scope(&weights_scope, &model);

        // DS4_ACCEPT_PROFILE per-position tallies (drafter accuracy by draft index).
        let mut pos_total = vec![0u64; k];
        let mut pos_hit = vec![0u64; k];
        let mut cond_total = vec![0u64; k];
        let mut cond_hit = vec![0u64; k];

        let t_decode = Instant::now();
        for iter in 0..n_iters {
            let t_iter = Instant::now();
            let mut iter_first_div = usize::MAX;

            let prof = std::env::var("DS4_SPEC_PROFILE").is_ok();

            // (a) Drafter chain.
            let t_draft = Instant::now();
            // DS4_SPEC_DRAFTER=pld: prompt-lookup drafting — n-gram match over
            // (prompt ++ emitted), training-free, no MTP forward (sidesteps the
            // 1-layer-MTP accuracy ceiling). Pads rejected slots so the K-verifier
            // shape is unchanged. DS4_PLD_NGRAM = max n-gram (default 3).
            let use_pld = std::env::var("DS4_SPEC_DRAFTER").ok().as_deref() == Some("pld");
            let drafts: Vec<i32> = if use_pld {
                let mut hist = prompt_tokens.clone();
                hist.extend_from_slice(&emitted);
                let max_ng: usize = std::env::var("DS4_PLD_NGRAM")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
                let mut d = ds4_metal::spec_decode::prompt_lookup_draft(&hist, k, max_ng, 1);
                let n_hit = d.len();
                while d.len() < k {
                    d.push(last_token);
                }
                if prof {
                    eprintln!("  [pld] ngram-hit {}/{}", n_hit, k);
                }
                d
            } else {
                let mut extract_embed = |t: i32| -> Vec<f32> {
                    let s = (t as usize) * model.d_model;
                    token_embd[s..s + model.d_model].to_vec()
                };
                run_mtp_chain_drafts(
                    &disp, &prev_hc, last_token, mtp_pos, mtp_slot_cursor,
                    mtp_n_raw, k,
                    &bundle, &mut extract_embed, &mtp_layer_params,
                    mtp_slot, raw_cap,
                )?
            };
            n_drafted += k;
            let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

            // (b) Verifier — fresh scope.
            // DS4_VERIFY_COMPRESSOR: seed the GPU comp ring from the prefill-built
            // CPU comp_kv_ring (the ring starts at zero on GPU — same class as the
            // KV zero-prefix), so the per-draft ring-aware flash attends the real
            // long-range compressed context. Must run BEFORE the verify scope
            // borrows `disp`. comp_n_vec is the per-layer prefill ring count.
            let verify_comp = std::env::var("DS4_VERIFY_COMPRESSOR").is_ok();
            // Seed GPU comp ring + sliding-window state from the prefill CPU
            // mirror, per ratio==4 layer. The window (compressor_state_kv/score)
            // is needed for the per-draft EMIT (store_one slides over it); the
            // ring is what the flash attends. Both start at zero/garbage on GPU.
            let seed_comp = |disp: &MetalDispatcher, st: &ds4_engine::decode_step::AttnStepState| {
                for (li, layer) in model.layers.iter().enumerate() {
                    let p = &layer.attn.params;
                    if p.compress_ratio == 4 {
                        disp.populate_compressor_state(
                            li as u32, &st.comp_state_kv[li], &st.comp_state_score[li], false,
                        );
                        if st.n_comp[li] > 0 {
                            disp.populate_comp_ring(
                                li as u32, &st.comp_kv_ring[li],
                                raw_cap as usize, p.head_dim as usize,
                            );
                        }
                    }
                }
            };
            let comp_n_vec: Vec<u32> = if verify_comp {
                seed_comp(&disp, &prefill_state);
                // SEED COMPARISON (DS4_COMPARE_EMIT, iter 0): after seeding, the
                // GPU compressor pool must be bit-identical to the prefill CPU
                // mirror (populate_compressor_state is a memcpy). If not, the
                // seed (size/layout) is broken — the root of the emit divergence.
                if iter == 0 && std::env::var("DS4_COMPARE_EMIT").is_ok() {
                    for li in [2usize, 4, 6] {
                        let p = &model.layers[li].attn.params;
                        if p.compress_ratio != 4 { continue; }
                        let n = prefill_state.comp_state_kv[li].len();
                        let kv_buf = disp.compressor_state_kv_or_alloc(li as u32, n * 4);
                        let sc_buf = disp.compressor_state_score_or_alloc(li as u32, n * 4);
                        let gpu_kv = unsafe { std::slice::from_raw_parts(kv_buf.contents() as *const f32, n) };
                        let gpu_sc = unsafe { std::slice::from_raw_parts(sc_buf.contents() as *const f32, n) };
                        let cmp = |gpu: &[f32], cpu: &[f32]| -> (f32, f64) {
                            let (mut ma, mut s2) = (0.0f32, 0.0f64);
                            for (a, b) in cpu.iter().zip(gpu.iter()) {
                                let d = (a - b).abs(); if d > ma { ma = d; }
                                s2 += (*a as f64) * (*a as f64);
                            }
                            (ma, (s2 / cpu.len().max(1) as f64).sqrt())
                        };
                        let (ma_kv, rms_kv) = cmp(gpu_kv, &prefill_state.comp_state_kv[li]);
                        let (ma_sc, rms_sc) = cmp(gpu_sc, &prefill_state.comp_state_score[li]);
                        eprintln!("[seed-cmp] layer {} (len={}): state_kv max_abs={:.3e} rms={:.3e} {} | state_score max_abs={:.3e} rms={:.3e} {}",
                            li, n, ma_kv, rms_kv, if ma_kv < 1e-5 { "IDENTICAL" } else { "DIFFERS!" },
                            ma_sc, rms_sc, if ma_sc < 1e-3 { "IDENTICAL" } else { "DIFFERS!" });
                    }
                }
                prefill_state.n_comp.clone()
            } else {
                Vec::new()
            };
            let comp_n_opt: Option<&[u32]> =
                if verify_comp { Some(&comp_n_vec) } else { None };

            // Build base verifier layer bundles + output-head DeferredBufs.
            let t_verify = Instant::now();
            let scope = disp.batch_scope();
            scope.set_q8_proj(true);
            let t_upload = Instant::now();
            // owned_layers hoisted out of the loop (read-only weights, cached
            // GPU buffers) — no per-iter re-upload.
            let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;
            // DS4_VERIFY_N_LAYERS truncates the chain for per-layer cost
            // profiling (slope of chain_ms vs N = true per-layer cost in the
            // chain). Numerics break when truncated — timing diagnostic only.
            let n_verify_layers: usize = std::env::var("DS4_VERIFY_N_LAYERS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(model.layers.len())
                .min(model.layers.len());
            // DS4_VERIFY_SAME_LAYER=1: repeat layer 0 N times instead of 43
            // distinct layers. Touches ONE layer's weight working set →
            // resident/cached. If this is fast but distinct-layers is slow,
            // the per-layer GPU inflation is memory-residency thrashing.
            let same_layer = std::env::var("DS4_VERIFY_SAME_LAYER").ok().as_deref() == Some("1");
            let layer_bundles: Vec<_> = (0..n_verify_layers)
                .map(|i| {
                    let li = if same_layer { 0 } else { i };
                    owned_layers[li].as_verify_bundle(&model.layers[li])
                })
                .collect();

            let n_hc = mtp_layer_params.n_hc as usize;
            let hc_dim = n_hc * model.d_model;

            // Verifier layer-0 input: row k = embed(drafts[k]) replicated
            // across n_hc.  Each row writes its own KV at slot base_slot+k
            // and produces logits for position base_pos+k+1 (LM head over
            // slot's hidden state).  See spec-decode alignment below — these
            // K verifier outputs verify drafts[1..K]; drafts[0] is verified
            // separately by projecting prev_hc through the base output head.
            let mut prev_hc_k = vec![0.0f32; k * hc_dim];
            for (k_idx, &dtok) in drafts.iter().enumerate() {
                let embed_start = (dtok as usize) * model.d_model;
                let embed = &token_embd[embed_start..embed_start + model.d_model];
                for h in 0..n_hc {
                    let dst = k_idx * hc_dim + h * model.d_model;
                    prev_hc_k[dst..dst + model.d_model].copy_from_slice(embed);
                }
            }
            let prev_hc_k_db = scope.upload_f32(&prev_hc_k);

            // logits_0: project prev_hc (HC residual after prefill — model's
            // state right before predicting position base_pos) through the
            // base output head (RMS norm + hc_head_fn + hc_weighted_sum +
            // final_norm + lm_head).  Antirez: `metal_graph_encode_output_head`
            // (ds4.c:9590) does this; we reuse `encode_mtp_output_head` with
            // BASE weights because the pipeline shape is identical (only the
            // 5th-stage norm differs: MTP uses `mtp.norm`, base uses
            // `final_norm` — encode_mtp_output_head takes that γ as an arg).
            let prev_hc_db = scope.upload_f32(&prev_hc);
            let unit_gamma_hc_vec: Vec<f32> = vec![1.0; hc_dim];
            let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc_vec);

            // Shape params from MtpDraftBundle (verifier uses same DS4 V4
            // Flash architectural constants for base layers).
            let shape = bundle.shape;
            let layer_params_iter = layer_bundles.iter().map(|_| {});
            let _ = layer_params_iter;

            let mut scope = scope;  // shadow as mut for &mut self calls
            let t_chain = Instant::now();
            let cur_hc_k = scope.encode_verify_layers_K(
                &prev_hc_k_db, &layer_bundles, &unit_gamma_hc_db,
                shape.n_hc, shape.n_embd, shape.n_lora_q,
                shape.n_head, shape.head_dim, shape.kv_row,
                shape.n_groups, shape.n_lora_o, shape.group_dim, shape.out_low_dim,
                shape.n_experts, shape.d_ffn, shape.shared_dim,
                shape.sinkhorn_iters, shape.hc_eps, shape.rms_eps, shape.flash_scale,
                raw_cap, base_slot, base_pos, k,
                comp_n_opt,
            )?;
            // Profiling split: encode_verify_layers_K returns after CPU-side
            // command encoding (no GPU wait); commit_wait_stage then forces the
            // GPU to run. So encode_ms = CPU encode cost, chain_gpu_ms = GPU
            // busy. The full chain_ms (= encode + GPU) is what the loop pays.
            let mut chain_ms = 0.0f64;
            let mut encode_ms = 0.0f64;
            if prof {
                encode_ms = t_chain.elapsed().as_secs_f64() * 1000.0;
                scope.commit_wait_stage("verify_layer_chain");
                chain_ms = t_chain.elapsed().as_secs_f64() * 1000.0;
            }

            // (c) Verifier output head — needs base model's output_hc_* + final_norm + lm_head.
            // ComposedModelWeights has these as f32. lm_head_q8 was already extracted as
            // raw Q8_0; for the verifier we need DeferredBufs from the bundle.
            let hc_head_fn_vec = model.output_hc_fn.as_ref()
                .ok_or_else(|| anyhow!("base model missing output_hc_fn (required for HC head)"))?;
            let hc_head_scale_vec = model.output_hc_scale.as_ref()
                .ok_or_else(|| anyhow!("base model missing output_hc_scale"))?;
            let hc_head_base_vec = model.output_hc_base.as_ref()
                .ok_or_else(|| anyhow!("base model missing output_hc_base"))?;

            let hc_head_fn_db    = scope.weight_f32(hc_head_fn_vec);
            let hc_head_scale_db = scope.weight_f32(hc_head_scale_vec);
            let hc_head_base_db  = scope.weight_f32(hc_head_base_vec);
            let final_norm_db    = scope.weight_f32(&model.final_norm_gamma);
            // Reuse the MTP draft bundle's base_lm_head_q8 — same bytes,
            // identity-cached upload via cached_weight_buffer.
            let lm_head_db = scope.weight_q8_0_raw(&bundle.base_lm_head_q8,
                model.d_model * model.vocab_size);

            let verify_logits_k = scope.encode_verify_output_head_K(
                &cur_hc_k, &unit_gamma_hc_db,
                &hc_head_fn_db, &hc_head_scale_db, &hc_head_base_db,
                &final_norm_db, &lm_head_db,
                model.d_model, n_hc, model.vocab_size, k,
                shape.rms_eps, shape.hc_eps,
            )?;

            // logits_0 from prev_hc → predict position base_pos = verify
            // drafts[0].  Uses encode_mtp_output_head with BASE weights:
            // base's output_hc_* for HC head, final_norm_γ in the 5th stage
            // (where MTP would use mtp.norm), and base.lm_head Q8.
            let logits_0_db = scope.encode_mtp_output_head(
                &prev_hc_db, &unit_gamma_hc_db,
                &hc_head_fn_db, &hc_head_scale_db, &hc_head_base_db,
                /*mtp_norm_gamma=*/&final_norm_db,
                &lm_head_db,
                model.d_model, n_hc, model.vocab_size,
                shape.rms_eps, shape.hc_eps,
            )?;

            let t_flush = Instant::now();
            let outs = scope.flush_and_read_multi(
                &[&verify_logits_k, &cur_hc_k, &logits_0_db],
            );
            let flush_ms = t_flush.elapsed().as_secs_f64() * 1000.0;
            let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
            let verifier_logits = &outs[0];      // [K, vocab]
            let cur_hc_k_vec = &outs[1];          // [K, hc_dim]
            let logits_0 = &outs[2];              // [vocab]

            // Stitch full_logits = [logits_0, verifier_logits[0..K-1]] so
            // full_logits[k] verifies drafts[k] (= candidate for position
            // base_pos+k).  verifier_logits[K-1] is the BONUS for position
            // base_pos+K — used only if all K drafts accepted (as the
            // correction-or-extension token).  accept_longest_prefix_greedy
            // takes [K, vocab] of logits matching drafts[0..K].
            let mut full_logits = Vec::with_capacity(k * model.vocab_size);
            full_logits.extend_from_slice(logits_0);
            for k_idx in 0..(k - 1) {
                let row_start = k_idx * model.vocab_size;
                full_logits.extend_from_slice(
                    &verifier_logits[row_start..row_start + model.vocab_size],
                );
            }
            assert_eq!(full_logits.len(), k * model.vocab_size);

            // (d) Accept.
            let result = accept_longest_prefix_greedy(
                &drafts, &full_logits, model.vocab_size,
            );
            n_accepted += result.accept_len;
            emitted.extend(&result.emitted);
            // DS4_ACCEPT_PROFILE: per-position accept accounting. The verifier
            // is faithful (emits base-greedy), so verifier_argmax[p] is the TRUE
            // next token at position p. Tally, per draft position p, how often
            // the drafter proposed the correct token — reveals WHERE the drafter
            // breaks (depth-degradation vs flat per-token error).
            if std::env::var("DS4_ACCEPT_PROFILE").is_ok() {
                let mut prefix_ok = true;
                for p in 0..k {
                    let row = &full_logits[p * model.vocab_size..(p + 1) * model.vocab_size];
                    let want = ds4_metal::spec_decode::argmax_f32(row) as i32;
                    let got = drafts[p];
                    pos_total[p] += 1;
                    if got == want { pos_hit[p] += 1; }
                    // CONDITIONAL: draft[p] hit rate GIVEN all prior drafts were
                    // correct. Isolates the drafter's inherent p-ahead accuracy
                    // (correct context) from error-compounding (wrong context).
                    // If cond_hit[p] ≈ pos_hit[p], degradation is INHERENT (1-layer
                    // drafter can't predict p ahead). If cond_hit[p] >> pos_hit[p],
                    // it's COMPOUNDING (later drafts fed wrong prior tokens).
                    if prefix_ok {
                        cond_total[p] += 1;
                        if got == want { cond_hit[p] += 1; } else { prefix_ok = false; }
                    }
                    if got != want && iter_first_div == usize::MAX { iter_first_div = p; }
                }
                let _ = iter_first_div;
            }
            eprintln!("  iter {}: drafts={:?} accept_len={} emit={:?} ({:.1} ms)",
                iter, drafts, result.accept_len, result.emitted,
                t_iter.elapsed().as_secs_f64() * 1000.0);
            if prof {
                eprintln!("    profile: draft={:.0}ms verify={:.0}ms (upload={:.0}ms chain={:.0}ms [cpu_encode={:.0}ms gpu={:.0}ms] head+flush={:.0}ms)",
                    draft_ms, verify_ms, upload_ms, chain_ms, encode_ms, chain_ms - encode_ms, flush_ms);
            }

            // ROW-BY-ROW BISECTION (DS4_FAITHFUL_CHECK, iter 0): for each row k
            // where drafts[k] == base greedy ref_seq[k] (so the verifier row and
            // decode_step decode the SAME token over the SAME prefix), compare:
            //   (1) verifier_logits[k] argmax vs base greedy ref_seq[k+1]
            //       (= what verifier_logits[k] SHOULD predict), and
            //   (2) cur_hc_k row k  vs  ref_resid[k] (= decode_step's residual).
            // (1) localizes which row first diverges; (2) splits the cause:
            //   cur_hc matches + logits differ  => OUTPUT-HEAD bug at that row;
            //   cur_hc differs                  => LAYER-CHAIN residual bug.
            if iter == 0 {
                if let Some((ref_seq, ref_resid, ref_emit)) = &faithful_ref {
                    // EMIT-ROW COMPARISON (DS4_COMPARE_EMIT): verifier's GPU comp
                    // ring in-batch emit (row prefill_n_comp) vs decode_step's
                    // comp_kv_ring entry, per ratio==4 layer. Compare at the FIRST
                    // ratio==4 layer (clean residual) to isolate emit-computation
                    // error from residual compounding. The verify scope was just
                    // flushed (outs read), so the GPU ring is current.
                    if verify_comp && std::env::var("DS4_COMPARE_EMIT").is_ok() {
                        eprintln!("  [emit-cmp] verifier GPU comp ring in-batch emit vs decode_step comp_kv_ring:");
                        for (li, layer) in model.layers.iter().enumerate() {
                            let p = &layer.attn.params;
                            if p.compress_ratio != 4 { continue; }
                            let hd = p.head_dim as usize;
                            let nprev = prefill_state.n_comp[li] as usize;
                            let ref_row = match ref_emit.get(&li) { Some(r) => r, None => continue };
                            // read GPU ring rows [0..nprev+1); take row nprev.
                            let mut ring = vec![0.0f32; (nprev + 1) * hd];
                            disp.read_comp_ring(li as u32, raw_cap as usize, hd, nprev + 1, &mut ring);
                            let ver_row = &ring[nprev * hd..(nprev + 1) * hd];
                            let (mut ma, mut s2) = (0.0f32, 0.0f64);
                            for (a, b) in ref_row.iter().zip(ver_row.iter()) {
                                let d = (a - b).abs(); if d > ma { ma = d; }
                                s2 += (*a as f64) * (*a as f64);
                            }
                            let rms = (s2 / hd as f64).sqrt();
                            eprintln!("    layer {:2}: max_abs={:.3e} rms_ref={:.3e} rel={:.3e} ({})",
                                li, ma, rms, ma as f64 / rms.max(1e-9),
                                if (ma as f64 / rms.max(1e-9)) < 0.05 { "MATCH" } else { "DIVERGE" });
                        }
                    }
                    eprintln!("  [row-bisect] (drafts vs base greedy; verifier_logits[k] predicts pos base_pos+k+1)");
                    for kk in 0..k {
                        // verifier_logits[kk] argmax (row kk, predicts base_pos+kk+1).
                        let vrow = &verifier_logits[kk * model.vocab_size..(kk + 1) * model.vocab_size];
                        let v_arg = ds4_metal::spec_decode::argmax_f32(vrow) as i32;
                        let base_next = ref_seq.get(kk + 1).copied().unwrap_or(-1);
                        // cur_hc row kk vs decode_step residual ref_resid[kk].
                        let (mut ma, mut s2) = (0.0f32, 0.0f64);
                        if kk < ref_resid.len() {
                            let vhc = &cur_hc_k_vec[kk * hc_dim..(kk + 1) * hc_dim];
                            for (a, b) in ref_resid[kk].iter().zip(vhc.iter()) {
                                let d = (a - b).abs(); if d > ma { ma = d; }
                                s2 += (*a as f64) * (*a as f64);
                            }
                        }
                        let rms = (s2 / hc_dim as f64).sqrt();
                        let rel = ma as f64 / rms.max(1e-9);
                        let draft_eq_base = drafts.get(kk).copied() == ref_seq.get(kk).copied();
                        eprintln!("    row {}: draft={} base[{}]={} (draft==base:{}) | verifier_logits[{}] argmax={} vs base_next={} ({}) | cur_hc rel={:.3e} ({})",
                            kk, drafts.get(kk).copied().unwrap_or(-1), kk, ref_seq.get(kk).copied().unwrap_or(-1), draft_eq_base,
                            kk, v_arg, base_next,
                            if v_arg == base_next { "LOGITS-MATCH" } else { "LOGITS-DIVERGE" },
                            rel,
                            if rel < 0.10 { "HC-MATCH→head-bug-if-logits-diverge" } else { "HC-DIVERGE→layer-chain-bug" });
                    }
                }
            }

            // CONSISTENCY CHECK (DS4_VERIFY_CONSISTENCY=1, iter 0 only):
            // does the K-position verifier's row-0 logits (= full_logits[1],
            // which verifies drafts[1]) match a ground-truth K=1 base decode of
            // drafts[0]?  Row 0 = base layer-chain on embed(drafts[0]) at slot
            // base_pos → predicts position base_pos+1.  A real autoregressive
            // base decode of drafts[0] at base_pos predicts the same.  If these
            // DIVERGE, the verifier (not the drafter) is why drafts[1+] never
            // accept.  Uses a state clone so decode_step doesn't corrupt the
            // live KV (it re-writes slot base_pos with the same token).
            if iter == 0 && std::env::var("DS4_VERIFY_CONSISTENCY").ok().as_deref() == Some("1") {
                use ds4_engine::decode_step::{decode_step_with_attn_to_residual, DecodeConfig};
                let mut st = prefill_state.clone();
                st.pos = base_pos;
                for kp in st.kv_pos.iter_mut() { *kp = base_slot; }
                let d0 = drafts[0];
                let es = (d0 as usize) * model.d_model;
                let embed = token_embd[es..es + model.d_model].to_vec();
                let cfg = DecodeConfig::default();
                let _ = decode_step_with_attn_to_residual(
                    &disp, &disp, embed.clone(), &model, &mut st, &cfg, raw_cap,
                )?;
                // Compare the RESIDUAL cur_hc directly (isolates the layer chain
                // from the output head). decode_step truncates to DS4_DECODE_N_LAYERS;
                // the verifier truncates to DS4_VERIFY_N_LAYERS — set them equal to
                // bisect which layer first diverges.
                let ref_hc = &st.cur_hc;                       // [hc_dim] after L decode layers
                let ver_hc = &cur_hc_k_vec[0..ref_hc.len()];   // verifier row 0 after L layers
                let mut max_abs = 0.0f32;
                let mut sum_ref2 = 0.0f64;
                let mut first_div: Option<usize> = None;
                for (i, (a, b)) in ref_hc.iter().zip(ver_hc.iter()).enumerate() {
                    let d = (a - b).abs();
                    if d > max_abs { max_abs = d; }
                    sum_ref2 += (*a as f64) * (*a as f64);
                    if first_div.is_none() && d > 1e-2 { first_div = Some(i); }
                }
                let rms_ref = (sum_ref2 / ref_hc.len() as f64).sqrt();
                let rel = max_abs as f64 / rms_ref.max(1e-9);
                eprintln!("  [consistency] cur_hc: verifier row0 vs decode_step(drafts[0]={}) \
                    @ DECODE_N_LAYERS={} VERIFY_N_LAYERS={}:\n    \
                    max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div_idx={:?} ({})",
                    d0,
                    std::env::var("DS4_DECODE_N_LAYERS").unwrap_or_else(|_| "all".into()),
                    std::env::var("DS4_VERIFY_N_LAYERS").unwrap_or_else(|_| "all".into()),
                    max_abs, rms_ref, rel, first_div,
                    if (rel) < 0.05 { "MATCH (within 5% rel)" } else { "DIVERGE" });

                // OP-LEVEL bisect (layer 0): compare the FIRST op, hc_collapse_norm.
                // ref = MetalDispatcher::hc_collapse_norm (what decode_step uses);
                // ver = the verifier's hc_collapse_norm_k(K=1). Same layer-0 weights
                // + input (embed(drafts[0]) replicated across n_hc). Localizes
                // whether the divergence starts at the very first op or downstream.
                {
                    use ds4_engine::attn_dispatch::{AttentionDispatcher, HcKind};
                    let a0 = &model.layers[0].attn;
                    let p0 = &a0.params;
                    let hc_dim = n_hc * model.d_model;
                    let mut prev_hc = vec![0.0f32; hc_dim];
                    for h in 0..n_hc {
                        prev_hc[h * model.d_model..(h + 1) * model.d_model]
                            .copy_from_slice(&embed);
                    }
                    // ref (decode_step's dispatcher path)
                    let (_c, normed_ref, _s) = disp.hc_collapse_norm(
                        p0, HcKind::Attn,
                        &a0.hc_attn_fn, &a0.hc_attn_scale, &a0.hc_attn_base,
                        &prev_hc, Some(&a0.hc_norm_gamma),
                    );  // CPU returns (cur, normed, split)
                    let split_ref = &_s;
                    // ver (K-position path, K=1) — returns (split, cur, normed)
                    let sc = disp.batch_scope();
                    let phc = sc.upload_f32(&prev_hc);
                    let wfn = sc.weight_f32(&a0.hc_attn_fn);
                    let wsc = sc.weight_f32(&a0.hc_attn_scale);
                    let wba = sc.weight_f32(&a0.hc_attn_base);
                    let wgm = sc.weight_f32(&a0.hc_norm_gamma);
                    let ug = sc.weight_f32(&unit_gamma_hc_vec);
                    let mut sc = sc;
                    let (split_k, _ck, normed_k) = sc.hc_collapse_norm_k(
                        &phc, &wfn, &wsc, &wba, &wgm,
                        n_hc, model.d_model, shape.sinkhorn_iters, shape.hc_eps, shape.rms_eps,
                        &ug, 1, false,
                    )?;
                    let hc_outs = sc.flush_and_read_multi(&[&normed_k, &split_k]);
                    let normed_ver = &hc_outs[0];
                    let split_ver = &hc_outs[1];
                    let mut ma = 0.0f32; let mut s2 = 0.0f64; let mut fd: Option<usize> = None;
                    for (i, (r, v)) in normed_ref.iter().zip(normed_ver.iter()).enumerate() {
                        let d = (r - v).abs();
                        if d > ma { ma = d; }
                        s2 += (*r as f64) * (*r as f64);
                        if fd.is_none() && d > 1e-3 { fd = Some(i); }
                    }
                    let rms = (s2 / normed_ref.len() as f64).sqrt();
                    // SPLIT comparison (post + comb sections feed hc_expand).
                    {
                        let n = split_ref.len().min(split_ver.len());
                        let (mut sma, mut ss2) = (0.0f32, 0.0f64);
                        for i in 0..n {
                            let d = (split_ref[i] - split_ver[i]).abs();
                            if d > sma { sma = d; }
                            ss2 += (split_ref[i] as f64).powi(2);
                        }
                        let srms = (ss2 / n as f64).sqrt();
                        eprintln!("  [op-bisect] hc_collapse SPLIT (mix_hc={}): ref={:?} ver={:?}\n    \
                            max_abs={:.3e} rms={:.3e} rel={:.3e} ({})",
                            n, &split_ref[..n.min(12)], &split_ver[..n.min(12)],
                            sma, srms, sma as f64 / srms.max(1e-9),
                            if (sma as f64 / srms.max(1e-9)) < 0.05 { "MATCH" }
                            else { "DIVERGE → bug is IN hc_collapse split (post/comb sections)" });
                    }
                    eprintln!("  [op-bisect] hc_collapse_norm (layer0, first op): \
                        len={} max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div={:?} ({})",
                        normed_ref.len(), ma, rms, ma as f64 / rms.max(1e-9), fd,
                        if (ma as f64 / rms.max(1e-9)) < 0.05 { "MATCH → bug is DOWNSTREAM (qkv/flash)" }
                        else { "DIVERGE → bug is IN hc_collapse_norm" });

                    // OP 2: qkv chain. ref = decode_step's attn_qkv_chain_batched
                    // (f32 weights); ver = encode_attn_qkv_chain_k (Q8 weights).
                    // Same `normed` input. Compare q_heads. (Q8 vs f32 gives ~1%
                    // rel; a gross >>5% means a real qkv geometry bug at n_lora_q=1024.)
                    use ds4_engine::dispatch::KernelDispatcher;
                    let n_lora_q = p0.n_lora_q as usize;
                    let kv_row = p0.n_lora_kv as usize;
                    let n_head = p0.n_head as usize;
                    let head_dim = p0.head_dim as usize;
                    let d_embd = model.d_model;
                    let q_dim = n_head * head_dim;
                    let (_qr, q_heads_ref, kv_raw_ref) = KernelDispatcher::attn_qkv_chain_batched(
                        &disp,
                        &normed_ref, &a0.attn_q_a, &a0.qkv_gamma_q, n_lora_q,
                        &a0.attn_q_b, n_head, head_dim, shape.rms_eps, &a0.attn_kv, kv_row,
                    );
                    let sc2 = disp.batch_scope();
                    sc2.set_q8_proj(true);
                    let nd = sc2.upload_f32(&normed_ref);
                    let qa = sc2.weight_q8_0_raw(&a0.attn_q_a_q8, n_lora_q * d_embd);
                    let gq = sc2.weight_f32(&a0.qkv_gamma_q);
                    let qb = sc2.weight_q8_0_raw(&a0.attn_q_b_q8, q_dim * n_lora_q);
                    let kvw = sc2.weight_q8_0_raw(&a0.attn_kv_q8, kv_row * d_embd);
                    let (_qrk, q_heads_k, _kvk) = sc2.encode_attn_qkv_chain_k(
                        &nd, &qa, &gq, &qb, &kvw,
                        n_lora_q, n_head, head_dim, shape.rms_eps, kv_row, d_embd, 1,
                    )?;
                    let q_heads_ver = sc2.flush_and_read(&q_heads_k);
                    let (mut ma2, mut s22, mut fd2) = (0.0f32, 0.0f64, None::<usize>);
                    for (i, (r, v)) in q_heads_ref.iter().zip(q_heads_ver.iter()).enumerate() {
                        let d = (r - v).abs();
                        if d > ma2 { ma2 = d; }
                        s22 += (*r as f64) * (*r as f64);
                        if fd2.is_none() && d > 1e-2 { fd2 = Some(i); }
                    }
                    let rms2 = (s22 / q_heads_ref.len() as f64).sqrt();
                    eprintln!("  [op-bisect] qkv chain q_heads (layer0): \
                        len={} max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div={:?} ({})",
                        q_heads_ref.len(), ma2, rms2, ma2 as f64 / rms2.max(1e-9), fd2,
                        if (ma2 as f64 / rms2.max(1e-9)) < 0.10 { "MATCH (≤10%, Q8 noise) → bug in FLASH" }
                        else { "DIVERGE → bug is IN qkv chain" });

                    // OP 3: ATTN-HALF output (splits attn-half vs ffn/MoE half).
                    // ref = decode_step with DS4_DISABLE_FFN → cur_hc = attn-only
                    // (prefix.after_attn_hc). ver = encode_attn_chain_k → attn_out_k,
                    // then hc_expand_attn_split → after_attn (same hc_dim semantics).
                    std::env::set_var("DS4_DISABLE_FFN", "1");
                    let mut st2 = prefill_state.clone();
                    st2.pos = base_pos;
                    for kp in st2.kv_pos.iter_mut() { *kp = base_slot; }
                    let _ = decode_step_with_attn_to_residual(
                        &disp, &disp, embed.clone(), &model, &mut st2, &cfg, raw_cap,
                    )?;
                    std::env::remove_var("DS4_DISABLE_FFN");
                    let attn_only_ref = st2.cur_hc.clone();

                    let sc3 = disp.batch_scope();
                    sc3.set_q8_proj(true);
                    let phc3 = sc3.upload_f32(&prev_hc);
                    let w_hcfn = sc3.weight_f32(&a0.hc_attn_fn);
                    let w_hcsc = sc3.weight_f32(&a0.hc_attn_scale);
                    let w_hcba = sc3.weight_f32(&a0.hc_attn_base);
                    let w_anorm = sc3.weight_f32(&a0.hc_norm_gamma);
                    let w_ug = sc3.weight_f32(&unit_gamma_hc_vec);
                    let w_qa = sc3.weight_q8_0_raw(&a0.attn_q_a_q8, n_lora_q * d_embd);
                    let w_gq = sc3.weight_f32(&a0.qkv_gamma_q);
                    let w_qb = sc3.weight_q8_0_raw(&a0.attn_q_b_q8, q_dim * n_lora_q);
                    let w_kv = sc3.weight_q8_0_raw(&a0.attn_kv_q8, kv_row * d_embd);
                    let w_gkv = sc3.weight_f32(&a0.qkv_gamma_kv);
                    let w_oa = sc3.weight_q8_0_raw(&a0.w_o_a_q8, shape.out_low_dim * shape.group_dim);
                    let w_ob = sc3.weight_q8_0_raw(&a0.w_o_b_q8, d_embd * shape.out_low_dim);
                    let w_sinks = sc3.weight_f32(&a0.attn_sinks);
                    let mut sc3 = sc3;
                    let (half3, attn_out_k3) = sc3.encode_attn_chain_k(
                        &phc3, &w_hcfn, &w_hcsc, &w_hcba, &w_anorm, &w_ug,
                        &w_qa, &w_gq, &w_qb, &w_kv, &w_gkv, &w_oa, &w_ob,
                        n_hc, d_embd, n_lora_q, n_head, head_dim, kv_row,
                        shape.n_groups, shape.n_lora_o, shape.group_dim, shape.out_low_dim,
                        shape.sinkhorn_iters, shape.hc_eps, shape.rms_eps, shape.flash_scale,
                        0, p0, raw_cap, base_slot, base_pos, base_pos, &w_sinks, 1,
                        None,
                    )?;
                    let after_attn_k = sc3.hc_expand_attn_split(
                        &attn_out_k3, &phc3, &half3.split_k, n_hc, d_embd,
                    )?;
                    // Read after_attn (OP3), kv_normed (OP4), AND the internal
                    // split (to confirm encode_attn_chain_k's hc_collapse used
                    // the corrected sinkhorn). attn_out too (to compare vs CPU).
                    let sc3_outs = sc3.flush_and_read_multi(
                        &[&after_attn_k, &half3.kv_normed_rotated_k, &half3.split_k, &attn_out_k3],
                    );
                    let after_attn_ver = &sc3_outs[0];
                    let kv_norm_ver = &sc3_outs[1];
                    let split_internal = &sc3_outs[2];
                    let attn_out_internal = &sc3_outs[3];
                    eprintln!("  [op3-debug] encode_attn_chain_k internal split comb[0..4]={:?} (should be ~[0.885,0.015,0.034,0.017] if sinkhorn=20)",
                        &split_internal[2*n_hc..2*n_hc+4]);
                    {
                        // Verifier attn_out rms + top-3 |val| (compare to decode_step's
                        // DS4_DUMP_ATTN_OUT=<base_pos> tap to isolate flash/output-proj
                        // in the real chain vs decode_step).
                        let ss: f64 = attn_out_internal.iter().map(|&v| (v as f64).powi(2)).sum();
                        let rms = (ss / attn_out_internal.len() as f64).sqrt();
                        let mut ranked: Vec<(usize, f32)> = attn_out_internal.iter().copied().enumerate().collect();
                        ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                        eprintln!("  [op3-debug] VERIFIER attn_out rms={:.4} top3={:?}", rms, &ranked[..3]);
                    }
                    // CPU after_attn from the SAME internal attn_out + split + prev_hc.
                    {
                        let post = &split_internal[n_hc..2*n_hc];
                        let comb = &split_internal[2*n_hc..];
                        let mut cpu_aa = vec![0.0f32; n_hc * d_embd];
                        for dst in 0..n_hc { for e in 0..d_embd {
                            let mut acc = post[dst] * attn_out_internal[e];
                            for src in 0..n_hc { acc += comb[dst + src*n_hc] * prev_hc[src*d_embd + e]; }
                            cpu_aa[dst*d_embd+e] = acc;
                        }}
                        let mut m = 0.0f32; let mut s = 0.0f64;
                        for (a, b) in cpu_aa.iter().zip(after_attn_ver.iter()) {
                            let dd = (a-b).abs(); if dd > m { m = dd; } s += (*a as f64).powi(2);
                        }
                        let rr = (s / cpu_aa.len() as f64).sqrt();
                        eprintln!("  [op3-debug] verifier after_attn vs CPU-of-its-own-(attn_out,split,prev_hc): max_abs={:.3e} rel={:.3e} (if MATCH, the verifier internals are self-consistent → OP3 ref mismatch)", m, m as f64/rr.max(1e-9));
                    }
                    let (mut ma3, mut s23, mut fd3) = (0.0f32, 0.0f64, None::<usize>);
                    for (i, (r, v)) in attn_only_ref.iter().zip(after_attn_ver.iter()).enumerate() {
                        let d = (r - v).abs();
                        if d > ma3 { ma3 = d; }
                        s23 += (*r as f64) * (*r as f64);
                        if fd3.is_none() && d > 1e-2 { fd3 = Some(i); }
                    }
                    let rms3 = (s23 / attn_only_ref.len() as f64).sqrt();
                    eprintln!("  [op-bisect] attn-half after_attn (layer0): \
                        len={} max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div={:?} ({})",
                        attn_only_ref.len(), ma3, rms3, ma3 as f64 / rms3.max(1e-9), fd3,
                        if (ma3 as f64 / rms3.max(1e-9)) < 0.10 { "MATCH → attn-half OK, bug is in FFN/MoE half" }
                        else { "DIVERGE → bug is IN attn-half (flash/kv/rope/output-proj)" });

                    // OP 4: kv_normed_rotated (splits kv-rms/rope from {q-rope, flash}).
                    // ref = decode_step's kv path: qkv_rms_norm_rows (rms w/ gamma_kv)
                    // + rope_tail @ base_pos, on the matched kv_raw. ver = the
                    // verifier's half.kv_normed_rotated_k (exposed).
                    let (_qrn, mut kv_norm_ref) = AttentionDispatcher::qkv_rms_norm_rows(
                        &disp, p0, &vec![0.0f32; n_lora_q], &kv_raw_ref,
                        &a0.qkv_gamma_q, &a0.qkv_gamma_kv,
                    );
                    // decode_step ropes ONLY the [n_lora_kv-n_rot .. n_lora_kv] tail
                    // slice (attn_dispatch.rs:410/580), not the full row.
                    let n_rot = p0.n_rot as usize;
                    AttentionDispatcher::rope_tail(
                        &disp, p0, &mut kv_norm_ref[kv_row - n_rot..kv_row], base_pos, false,
                    );
                    let (mut ma4, mut s24, mut fd4) = (0.0f32, 0.0f64, None::<usize>);
                    for (i, (r, v)) in kv_norm_ref.iter().zip(kv_norm_ver.iter()).enumerate() {
                        let d = (r - v).abs();
                        if d > ma4 { ma4 = d; }
                        s24 += (*r as f64) * (*r as f64);
                        if fd4.is_none() && d > 1e-2 { fd4 = Some(i); }
                    }
                    let rms4 = (s24 / kv_norm_ref.len().max(1) as f64).sqrt();
                    eprintln!("  [op-bisect] kv_normed_rotated (layer0): \
                        len={}/{} max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div={:?} ({})",
                        kv_norm_ref.len(), kv_norm_ver.len(), ma4, rms4, ma4 as f64 / rms4.max(1e-9), fd4,
                        if (ma4 as f64 / rms4.max(1e-9)) < 0.10 { "MATCH → kv-path OK, bug is q-rope or FLASH" }
                        else { "DIVERGE → bug is IN kv-rms/rope" });

                    // OP 5: q-rope (splits q-rope from FLASH). Apply both ropes to
                    // the matched pre-rope q_heads_ref. ref = decode_step per-head
                    // tail rope (rope_tail on [head_dim-n_rot..head_dim] per head).
                    // ver = verifier rope_tail_q_heads_in_place_k.
                    let mut q_rope_ref = q_heads_ref.clone();
                    for h in 0..n_head {
                        let base = h * head_dim + (head_dim - n_rot);
                        AttentionDispatcher::rope_tail(
                            &disp, p0, &mut q_rope_ref[base..base + n_rot], base_pos, false,
                        );
                    }
                    let sc4 = disp.batch_scope();
                    let qh = sc4.upload_f32(&q_heads_ref);
                    let mut sc4 = sc4;
                    sc4.rope_tail_q_heads_in_place_k(&qh, n_head, head_dim, p0, base_pos, 1, false)?;
                    let q_rope_ver = sc4.flush_and_read(&qh);
                    let (mut ma5, mut s25, mut fd5) = (0.0f32, 0.0f64, None::<usize>);
                    for (i, (r, v)) in q_rope_ref.iter().zip(q_rope_ver.iter()).enumerate() {
                        let d = (r - v).abs();
                        if d > ma5 { ma5 = d; }
                        s25 += (*r as f64) * (*r as f64);
                        if fd5.is_none() && d > 1e-2 { fd5 = Some(i); }
                    }
                    let rms5 = (s25 / q_rope_ref.len() as f64).sqrt();
                    eprintln!("  [op-bisect] q-rope (layer0): \
                        len={} max_abs={:.3e} rms_ref={:.3e} rel={:.3e} first_div={:?} ({})",
                        q_rope_ref.len(), ma5, rms5, ma5 as f64 / rms5.max(1e-9), fd5,
                        if (ma5 as f64 / rms5.max(1e-9)) < 0.10 { "MATCH → q-rope OK → bug is the MLA FLASH kernel" }
                        else { "DIVERGE → bug is IN q-rope (rope_tail_q_heads_in_place_k)" });
                }
            }

            // (e) Advance. Two paths; reuse is the DEFAULT (set
            // DS4_SPEC_REUSE_GPU_KV=0 for the re-decode reference):
            //
            //  REUSE (default): read the accepted drafts' KV back from the
            //  verifier's GPU buffer (no re-decode) and decode_step ONLY the last
            //  emitted token for a faithful prev_hc. Cuts base work from
            //  advance_count re-decodes to 1. Measured +67% throughput (0.12→0.20
            //  tok/s) at the SAME 62.5% accept vs re-decode, now that the verifier
            //  is faithful (the rope-back fix). [An earlier 25% regression here was
            //  the rope bug on the UNFAITHFUL verifier, not the reuse logic.]
            //
            //  RE-DECODE (DS4_SPEC_REUSE_GPU_KV=0): re-run decode_step on EVERY
            //  emitted token — rebuilds full base state decode_step-faithfully.
            //  Correctness reference; costs advance_count base passes/iter.
            let accept_len = result.accept_len;
            let base_pos_pre = base_pos;
            let reuse_gpu_kv = std::env::var("DS4_SPEC_REUSE_GPU_KV").as_deref() != Ok("0");

            use ds4_engine::decode_step::{decode_step_with_attn_to_residual, DecodeConfig};
            let cfg = DecodeConfig::default();

            // RESTORE the GPU compressor pools to the pre-verify (prefill) state:
            // the per-draft emit mutated them speculatively, but the advance
            // reads+advances the same persistent pools and must start from the
            // correct window.
            if verify_comp {
                seed_comp(&disp, &prefill_state);
            }

            // Decode one token for the advance. DEFAULT = fused decode_token_unified
            // (+ GPU→CPU KV back-sync of the written slot; back-sync measured 0ms).
            // Fully faithful (28/28) AND fast once warm: ~95ms/decode, ~245ms/iter
            // total → 2.24 tok/s (vs ~0.2 with the unfused advance). DS4_ADVANCE_UNIFIED=0
            // for the legacy unfused decode_step reference. [An earlier run showed
            // 2s→33s "stalls" — that was a COLD/contended outlier; DS4_ADVANCE_PROFILE
            // showed decode_token_unified warms 1694→640→92ms and the back-sync is 0ms.]
            let advance_unified = std::env::var("DS4_ADVANCE_UNIFIED").as_deref() != Ok("0");
            let decode_one = |st: &mut ds4_engine::decode_step::AttnStepState, tok: i32, pos: u32|
                -> anyhow::Result<()> {
                st.pos = pos;
                for kp in st.kv_pos.iter_mut() { *kp = pos % raw_cap; }
                let es = (tok as usize) * model.d_model;
                let embed = token_embd[es..es + model.d_model].to_vec();
                if advance_unified {
                    let adv_prof = std::env::var("DS4_ADVANCE_PROFILE").is_ok();
                    let td = std::time::Instant::now();
                    let _ = disp.decode_token_unified(embed, &model, st, raw_cap)?;
                    let decode_ms = td.elapsed().as_secs_f64() * 1000.0;
                    let tb = std::time::Instant::now();
                    let slot = pos % raw_cap;
                    for (li, layer) in model.layers.iter().enumerate() {
                        let row = layer.attn.params.n_lora_kv as usize;
                        disp.read_persistent_kv_slots(
                            li as u32, raw_cap, row, slot, 1, &mut st.kv_storage[li],
                        );
                    }
                    if adv_prof {
                        eprintln!("    [adv] decode_token_unified={:.0}ms kv_backsync={:.0}ms",
                            decode_ms, tb.elapsed().as_secs_f64() * 1000.0);
                    }
                } else {
                    let _ = decode_step_with_attn_to_residual(
                        &disp, &disp, embed, &model, st, &cfg, raw_cap,
                    )?;
                }
                Ok(())
            };

            if reuse_gpu_kv {
                // Perf-path: reuse verifier GPU KV for accepted drafts; decode
                // only the last emitted token for a faithful prev_hc + its KV.
                let kv_row_adv = model.layers[0].attn.params.n_lora_kv as usize;
                if accept_len > 0 {
                    for (li, kvs) in prefill_state.kv_storage.iter_mut().enumerate() {
                        disp.read_persistent_kv_slots(
                            li as u32, raw_cap, kv_row_adv,
                            base_slot, accept_len as u32, kvs,
                        );
                    }
                }
                let last_tok = result.next_seed_token; // = emitted.last()
                let last_pos = base_pos_pre + (result.advance_count as u32 - 1);
                decode_one(&mut prefill_state, last_tok, last_pos)?;
                prev_hc = prefill_state.cur_hc.clone();
            } else {
                // Correct-path: re-decode every emitted token (full state sync).
                for (j, &tok) in result.emitted.iter().enumerate() {
                    decode_one(&mut prefill_state, tok, base_pos_pre + j as u32)?;
                }
                prev_hc = prefill_state.cur_hc.clone();
            }

            base_pos = base_pos_pre + result.advance_count as u32;
            base_slot = base_pos % raw_cap;
            prefill_state.pos = base_pos;
            for kp in prefill_state.kv_pos.iter_mut() { *kp = base_pos % raw_cap; }
            // Re-sync the GPU buffer for the NEXT iter's verifier with ONLY the
            // newly emitted tokens' slots [base_pos_pre, base_pos) — the prefix
            // is already correct on GPU. (Seeding [0, base_pos) every iter was
            // O(n²) and made per-iter climb 245ms→1700ms over a run.)
            populate_kv_fp8(&disp, &prefill_state, base_pos_pre, base_pos);
            let _ = cur_hc_k_vec; // (verifier residual NOT reused — see above)

            mtp_pos = base_pos;
            mtp_slot_cursor = mtp_pos % raw_cap;
            // Mirror antirez DS4_MTP_KEEP_ACCEPTED (ds4.c:16148): keep the
            // accepted prefix in the MTP raw cache.  The correction token's
            // MTP row was never written by the drafter (drafter only emits
            // drafts), so on correction we keep accept_len; on all-accept we
            // keep K.
            mtp_n_raw = (mtp_n_raw + accept_len as u32)
                .min(raw_cap.saturating_sub(1));
            last_token = result.next_seed_token;
        }
        std::env::remove_var("DS4_MOE_K_PATH");

        let elapsed = t_decode.elapsed().as_secs_f64();
        let tok_per_sec = emitted.len() as f64 / elapsed;
        let accept_rate = n_accepted as f64 / n_drafted as f64;
        eprintln!("\n=== spec-decode result ===");
        eprintln!("  emitted: {} tokens in {:.3}s → {:.2} tok/s",
            emitted.len(), elapsed, tok_per_sec);
        eprintln!("  accept rate: {}/{} = {:.1}%", n_accepted, n_drafted, accept_rate * 100.0);
        if std::env::var("DS4_ACCEPT_PROFILE").is_ok() {
            eprintln!("  [accept-profile] per draft-position hit rate (drafter vs faithful verifier):");
            for p in 0..k {
                let t = pos_total[p].max(1);
                let ct = cond_total[p].max(1);
                eprintln!("    draft[{}]: marginal {}/{}={:.0}%  | conditional(prior-all-correct) {}/{}={:.0}%",
                    p, pos_hit[p], pos_total[p], 100.0 * pos_hit[p] as f64 / t as f64,
                    cond_hit[p], cond_total[p], 100.0 * cond_hit[p] as f64 / ct as f64);
            }
            eprintln!("  [accept-profile] conditional ≈ marginal → INHERENT depth limit; conditional >> marginal → COMPOUNDING (fixable)");
        }

        // Faithfulness verdict: a faithful verifier's emitted sequence must equal
        // the pure base greedy decode. Compare the longest matching prefix.
        if let Some((ref_seq, _, _)) = &faithful_ref {
            let n = emitted.len().min(ref_seq.len());
            let lmp = (0..n).take_while(|&i| emitted[i] == ref_seq[i]).count();
            eprintln!("\n[FAITHFUL] spec emitted = {:?}", emitted);
            eprintln!("[FAITHFUL] base greedy  = {:?}", &ref_seq[..n]);
            eprintln!("[FAITHFUL] longest matching prefix = {}/{}  ({})", lmp, n,
                if lmp == n { "FULLY FAITHFUL — verifier emits base greedy" }
                else { "DIVERGES at the position above the mismatch" });
        }

        eprintln!("  vs baseline 18.82 tok/s (K=1, blit-shim MoE):");
        let speedup = tok_per_sec / 18.82;
        eprintln!("    → {:.2}x speedup ({:+.0}%)", speedup, (speedup - 1.0) * 100.0);

        Ok(())
    })();

    if let Err(e) = result {
        panic!("spec_decode_real_step_bench failed: {:#}", e);
    }
}
