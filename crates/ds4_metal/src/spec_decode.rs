//! Phase 3 Step 4d.4-5 — speculative-decoding accept logic + KV state
//! management contract.
//!
//! Spec-decode steps once the GPU forward pass produces K verifier logits:
//!
//! ```text
//!   drafts[K]     ← run_mtp_chain_drafts(...)
//!   verify[K,V]   ← scope.encode_verify_layers_K + encode_verify_output_head_K
//!   accept_result ← accept_longest_prefix_greedy(&drafts, &verify, vocab)
//!   /* advance KV pos by accept_result.advance_count for both base + MTP */
//! ```
//!
//! ## Accept policy (greedy)
//!
//! Standard speculative decoding with argmax sampling. At step k:
//! - `verify_argmax[k]` = what the verifier would emit at position pos+k.
//! - If `drafts[k] == verify_argmax[k]`, accept draft.
//! - Otherwise, the verifier "corrects" — accept k drafts, emit
//!   `verify_argmax[k]` as the (k+1)-th token, stop.
//!
//! Emit length: `accept_len + bonus`, where `bonus = 1` if the verifier
//! corrected (accept_len < K) and `0` if all K drafts matched (no extra
//! verifier output to emit beyond the K accepted).
//!
//! ## KV state management (caller-tracked)
//!
//! The K-position `encode_layer_k` path is STATELESS w.r.t. dispatcher
//! `state.kv_pos` — caller passes explicit `base_slot` + `base_pos`
//! per call. So spec-decode KV rollback is just bookkeeping at the
//! caller level:
//!
//! - After each spec-decode iter:
//!     `base_pos  += accept_result.advance_count`
//!     `base_slot += accept_result.advance_count`  (mod raw_cap for SWA)
//!     `mtp_pos   += accept_result.advance_count`
//!     `mtp_slot  += accept_result.advance_count`  (mod raw_cap for MTP)
//!
//! - Speculative writes beyond the accepted prefix (slots
//!   base_slot+accept_count..base_slot+K) are NOT physically cleared
//!   — they're overwritten by the next iter's encode_layer_k writes.

/// Greedy accept result.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptResult {
    /// Number of consecutive drafts that match the verifier's argmax.
    pub accept_len: usize,
    /// Tokens to emit from this spec-decode step: first
    /// `accept_len` are accepted drafts; an optional final entry is the
    /// verifier's "correction" token when `accept_len < K`.
    pub emitted: Vec<i32>,
    /// How far to advance `pos` + `kv_slot` for the next iter:
    /// `accept_len + 1` if the verifier corrected, else `K`.
    /// Equals `emitted.len()`.
    pub advance_count: usize,
    /// Token to seed the next drafter chain (the last emitted token).
    pub next_seed_token: i32,
}

/// Phase 3 Step 4d.4 — greedy accept (argmax-sampling) for K-position
/// spec-decode. Compares K draft tokens against K verifier logits;
/// accepts the longest matching prefix; if the verifier disagrees,
/// emits its corrected token as the bonus.
///
/// `verify_logits` layout: `[K, vocab]` flat (K rows of `vocab` logits).
/// Caller pulls this from `BatchScope::flush_and_read(&logits_K)` where
/// `logits_K` came from `encode_verify_output_head_K`.
pub fn accept_longest_prefix_greedy(
    drafts: &[i32],
    verify_logits: &[f32],
    vocab: usize,
) -> AcceptResult {
    let k = drafts.len();
    assert_eq!(
        verify_logits.len(),
        k * vocab,
        "verify_logits len {} != K*vocab = {}*{}",
        verify_logits.len(), k, vocab
    );

    let mut accept_len: usize = 0;
    let mut emitted: Vec<i32> = Vec::with_capacity(k + 1);
    let mut bonus: Option<i32> = None;
    let mut next_seed: i32 = drafts.last().copied().unwrap_or(0);

    for i in 0..k {
        let row_start = i * vocab;
        let verify_argmax = argmax_f32(&verify_logits[row_start..row_start + vocab]) as i32;
        if drafts[i] == verify_argmax {
            emitted.push(drafts[i]);
            accept_len += 1;
            next_seed = drafts[i];
        } else {
            // Verifier corrects — emit its choice as the bonus and stop.
            bonus = Some(verify_argmax);
            emitted.push(verify_argmax);
            next_seed = verify_argmax;
            break;
        }
    }

    let advance_count = emitted.len();
    let _ = bonus;
    AcceptResult { accept_len, emitted, advance_count, next_seed_token: next_seed }
}

/// argmax over an f32 slice (NaN-safe: returns 0 if all NaN).
pub fn argmax_f32(xs: &[f32]) -> usize {
    let mut best_i: usize = 0;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i; }
    }
    best_i
}

/// Prompt-lookup drafting (training-free spec-decode draft source). Instead of
/// a neural drafter, search the token `history` for the most recent earlier
/// occurrence of its own trailing n-gram and propose the tokens that FOLLOWED
/// that occurrence as the next-`k` draft.
///
/// This sidesteps the 1-layer-MTP drafter's accuracy ceiling entirely (it uses
/// no model) and is effectively free (a CPU n-gram scan). It excels when the
/// output reuses input/earlier tokens — code edits, summarization, RAG,
/// tool-call echoes, repetitive or structured text — and returns `< k` (often
/// 0) draft tokens on genuinely novel prose, where the verifier then just
/// corrects. Mirrors HF `prompt_lookup_num_tokens`: tries n-gram sizes
/// `max_ngram..=min_ngram` (longest match wins), scanning most-recent-first.
///
/// Returns up to `k` proposed tokens (caller pads to `K` for the K-batched
/// verifier; padded slots are rejected at no correctness cost).
pub fn prompt_lookup_draft(history: &[i32], k: usize, max_ngram: usize, min_ngram: usize) -> Vec<i32> {
    let n = history.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let max_ng = max_ngram.min(n.saturating_sub(1)).max(1);
    let min_ng = min_ngram.clamp(1, max_ng);
    for ng in (min_ng..=max_ng).rev() {
        let suffix = &history[n - ng..];
        // Most-recent earlier occurrence first (recent context is most
        // predictive). `s` is the match start; follow tokens begin at `s + ng`.
        for s in (0..n - ng).rev() {
            if &history[s..s + ng] == suffix {
                let fstart = s + ng;
                if fstart >= n {
                    continue;
                }
                let take = k.min(n - fstart);
                if take > 0 {
                    return history[fstart..fstart + take].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Phase 3 Step 4d.6 — prompt prefill helper. Runs the prompt through
/// the base model via the existing K=1 `decode_step_with_attn_to_residual`
/// path. Populates `state.kv_pos[layer]` and `state.cur_hc` for each
/// prompt token; the final `state.cur_hc` becomes the initial `prev_hc`
/// for the K-position spec-decode loop.
///
/// `token_embd_table` is `[vocab, d_model]` f32 — caller dequants from
/// the base GGUF's `token_embd.weight` tensor (typically F16). Each
/// prompt token's embedding row gets fed to the per-layer chain.
///
/// **Returns** `(final_prev_hc, state)`:
/// - `final_prev_hc`: `Vec<f32>` of size `n_hc * d_embd` — the HC residual
///   after running all prompt tokens. Use this as the K-position
///   spec-decode loop's initial `prev_hc_K` (after K-broadcast).
/// - `state`: `AttnStepState` with `state.pos == prompt_tokens.len()`
///   and `state.kv_pos[layer]` advanced per layer. The base KV cache is
///   populated for slots `0..prompt_tokens.len()`.
///
/// For K-position spec-decode after prefill:
/// - `base_pos`  = `state.pos`
/// - `base_slot` = `state.pos % raw_cap`  (each layer's kv_pos)
/// - K-broadcast `final_prev_hc` to `[K, n_hc, d_embd]` via copies.
#[cfg(target_os = "macos")]
pub fn prefill_to_residual(
    disp: &crate::MetalDispatcher,
    model: &ds4_engine::decode_step::ComposedModelWeights,
    token_embd_table: &[f32],
    prompt_tokens: &[i32],
    raw_cap: u32,
) -> anyhow::Result<(Vec<f32>, ds4_engine::decode_step::AttnStepState)> {
    use anyhow::bail;
    use ds4_engine::decode_step::{AttnStepState, DecodeConfig, decode_step_with_attn_to_residual};

    let d_model = model.d_model;
    let vocab = model.vocab_size;
    if token_embd_table.len() != vocab * d_model {
        bail!(
            "prefill_to_residual: token_embd_table {} != vocab*d_model = {}*{}",
            token_embd_table.len(), vocab, d_model
        );
    }
    if prompt_tokens.is_empty() {
        bail!("prefill_to_residual: prompt_tokens empty");
    }

    let mut state = AttnStepState::new(model, raw_cap);
    let cfg = DecodeConfig::default();

    let prof = std::env::var("DS4_PREFILL_PROFILE").is_ok();
    // Prefill decoder: the FUSED single-cb path (decode_token_unified,
    // ~85ms/token) by DEFAULT vs the unfused per-op decode_step (~5.3s/token,
    // DS4_PREFILL_UNIFIED=0 for the legacy reference). The fused path writes raw
    // KV only to the GPU persistent buffer (not state.kv_storage), so we read it
    // back into the CPU mirror after the loop — keeping prefill_to_residual's
    // contract (populated state.kv_storage) identical for both paths. Validated
    // 11/11 faithful + 66.7% accept with the unified path.
    let unified = std::env::var("DS4_PREFILL_UNIFIED").as_deref() != Ok("0");
    for (i, &token) in prompt_tokens.iter().enumerate() {
        if token < 0 || (token as usize) >= vocab {
            bail!(
                "prefill_to_residual: prompt_tokens[{}]={} out of vocab range [0,{})",
                i, token, vocab
            );
        }
        let start = (token as usize) * d_model;
        let embed: Vec<f32> = token_embd_table[start..start + d_model].to_vec();
        let t0 = std::time::Instant::now();
        if unified {
            let _logits = disp.decode_token_unified(embed, model, &mut state, raw_cap)?;
        } else {
            let _final_hidden = decode_step_with_attn_to_residual(
                disp, disp, embed, model, &mut state, &cfg, raw_cap,
            )?;
        }
        if prof {
            eprintln!("  [prefill] token {:2} (pos {}) = {:.0} ms",
                i, state.pos, t0.elapsed().as_secs_f64() * 1000.0);
        }
        // state.pos advanced internally per token.
    }

    // Back-sync GPU→CPU KV mirror for the fused path (it never wrote
    // state.kv_storage). Without this, callers reading state.kv_storage (KV
    // re-seeding, K=1 advance decode_step) see a zero prefix. No-op for the
    // unfused path (it already filled state.kv_storage), but reading the GPU
    // buffer back is harmless+idempotent there too — gate on `unified` to skip
    // the work.
    if unified {
        let n_layers = model.layers.len();
        let upto = state.pos;
        for li in 0..n_layers {
            let row = model.layers[li].attn.params.n_lora_kv as usize;
            disp.read_persistent_kv_slots(
                li as u32, raw_cap, row, 0, upto, &mut state.kv_storage[li],
            );
        }
    }

    let final_prev_hc = state.cur_hc.clone();
    Ok((final_prev_hc, state))
}

/// Phase 3 Step 4d.6 — K-broadcast a single prev_hc (1 token's HC
/// residual) into the `[K, n_hc, n_embd]` layout the K-position
/// spec-decode verifier needs.
///
/// All K candidates branch from the same context, so initially they
/// all share the same prev_hc. After the first spec-decode iter, each
/// K-row diverges as the drafter chains tokens.
///
/// `single` is `[n_hc * n_embd]` (flat); returns `[K * n_hc * n_embd]`.
#[allow(non_snake_case)]
pub fn broadcast_prev_hc_K(single: &[f32], k_positions: usize) -> Vec<f32> {
    let hc_dim = single.len();
    let mut out = Vec::with_capacity(k_positions * hc_dim);
    for _ in 0..k_positions {
        out.extend_from_slice(single);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot_logits(k: usize, vocab: usize, picks: &[i32]) -> Vec<f32> {
        let mut out = vec![0.0f32; k * vocab];
        for (row, &pick) in picks.iter().enumerate() {
            out[row * vocab + pick as usize] = 1.0;
        }
        out
    }

    #[test]
    fn accept_all_k_when_all_match() {
        // drafts = [42, 17, 99], verify argmax = [42, 17, 99] → accept all 3.
        let vocab = 256;
        let drafts = vec![42, 17, 99];
        let verify = one_hot_logits(3, vocab, &[42, 17, 99]);
        let r = accept_longest_prefix_greedy(&drafts, &verify, vocab);
        assert_eq!(r.accept_len, 3);
        assert_eq!(r.emitted, vec![42, 17, 99]);
        assert_eq!(r.advance_count, 3);
        assert_eq!(r.next_seed_token, 99);
    }

    #[test]
    fn accept_zero_when_first_mismatch() {
        // drafts = [42, 17], verify = [50, 17] → accept 0, emit verifier 50.
        let vocab = 100;
        let drafts = vec![42, 17];
        let verify = one_hot_logits(2, vocab, &[50, 17]);
        let r = accept_longest_prefix_greedy(&drafts, &verify, vocab);
        assert_eq!(r.accept_len, 0);
        assert_eq!(r.emitted, vec![50]);
        assert_eq!(r.advance_count, 1);
        assert_eq!(r.next_seed_token, 50);
    }

    #[test]
    fn accept_partial_with_correction() {
        // drafts = [10, 20, 30, 40], verify = [10, 20, 35, 40].
        // Accept drafts[0..2] = [10, 20], correction = 35.
        let vocab = 64;
        let drafts = vec![10, 20, 30, 40];
        let verify = one_hot_logits(4, vocab, &[10, 20, 35, 40]);
        let r = accept_longest_prefix_greedy(&drafts, &verify, vocab);
        assert_eq!(r.accept_len, 2);
        assert_eq!(r.emitted, vec![10, 20, 35]);
        assert_eq!(r.advance_count, 3);
        assert_eq!(r.next_seed_token, 35);
    }

    #[test]
    fn accept_k1_match() {
        let vocab = 8;
        let drafts = vec![5];
        let verify = one_hot_logits(1, vocab, &[5]);
        let r = accept_longest_prefix_greedy(&drafts, &verify, vocab);
        assert_eq!(r.accept_len, 1);
        assert_eq!(r.emitted, vec![5]);
        assert_eq!(r.advance_count, 1);
        assert_eq!(r.next_seed_token, 5);
    }

    #[test]
    fn accept_k1_mismatch() {
        let vocab = 8;
        let drafts = vec![5];
        let verify = one_hot_logits(1, vocab, &[3]);
        let r = accept_longest_prefix_greedy(&drafts, &verify, vocab);
        assert_eq!(r.accept_len, 0);
        assert_eq!(r.emitted, vec![3]);
        assert_eq!(r.advance_count, 1);
        assert_eq!(r.next_seed_token, 3);
    }

    #[test]
    fn pld_repeat_match() {
        // history "a b c d  ... a b c" → suffix "a b c" matched earlier → "d ...".
        let h = vec![1, 2, 3, 4, 9, 9, 1, 2, 3];
        // suffix [1,2,3] (ng=3) earlier occurrence at s=0 → follow history[3..7].
        assert_eq!(prompt_lookup_draft(&h, 4, 3, 1), vec![4, 9, 9, 1]);
        // k caps the count.
        assert_eq!(prompt_lookup_draft(&h, 1, 3, 1), vec![4]);
    }

    #[test]
    fn pld_no_match_returns_empty() {
        // strictly increasing → no repeated n-gram → no draft.
        let h = vec![1, 2, 3, 4, 5];
        assert!(prompt_lookup_draft(&h, 4, 3, 1).is_empty());
        assert!(prompt_lookup_draft(&[], 4, 3, 1).is_empty());
    }

    #[test]
    fn pld_longest_ngram_wins() {
        // "7 1 2" appears once; "1 2" appears twice. suffix ends "...7 1 2".
        // ng=3 ("7 1 2") matches the earlier occurrence at s=0 → follow [5].
        let h = vec![7, 1, 2, 5, 8, 1, 2, 6, 7, 1, 2];
        assert_eq!(prompt_lookup_draft(&h, 2, 3, 1), vec![5, 8]);
    }
}
