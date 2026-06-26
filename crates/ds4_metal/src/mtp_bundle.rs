//! Phase 3 Step 4b — MTP drafter weight bundle.
//!
//! Bridges `ds4_engine::mtp::MtpWeights` (tensor handles + offsets) to the
//! `BatchScope::encode_mtp_draft` API (which takes pre-uploaded DeferredBufs
//! grouped into `MtpDraftInputMix` / `MtpDraftLayerWeights` /
//! `MtpDraftOutputHead`). One `MtpDraftBundle` is constructed once at MTP
//! model load time — it owns the CPU-side Vec<f32>/Vec<u8> for every
//! drafter tensor — and per decode token the caller uploads to the
//! current scope via `upload_to_scope` (cached via `cached_weight_buffer`
//! so the second+ call is no-copy).
//!
//! ```ignore
//! // Once at startup:
//! let mtp = MtpWeights::from_path("…/DS4-V4-Flash-MTP.gguf")?;
//! let mtp_bytes = std::fs::read("…/DS4-V4-Flash-MTP.gguf")?;
//! let base_output = base_model.weights.output;  // [n_embd, vocab] Q8_0
//! let bundle = MtpDraftBundle::from_mtp(&mtp, &mtp_bytes, &base_output_bytes)?;
//!
//! // Per decode token:
//! let scope = disp.batch_scope();
//! let bufs = bundle.upload_to_scope(&scope);
//! let logits = scope.encode_mtp_draft(
//!     &prev_hc_db, &token_embed_db,
//!     bufs.input_mix(), bufs.layer(), bufs.output_head(),
//!     bundle.shape, …,
//! )?;
//! ```

#[cfg(target_os = "macos")]
use anyhow::Result;

#[cfg(target_os = "macos")]
use ds4_engine::mtp::MtpWeights;

#[cfg(target_os = "macos")]
use crate::deferred::{
    BatchScope, DeferredBuf,
    MtpDraftInputMix, MtpDraftLayerWeights, MtpDraftOutputHead, MtpDraftShape,
};

/// CPU-side bundle of all MTP drafter weights, ready to upload to a scope.
/// Owns each tensor's Vec<f32> / Vec<u8> — created once per model load.
#[cfg(target_os = "macos")]
pub struct MtpDraftBundle {
    // Input mix.
    pub enorm_gamma: Vec<f32>,
    pub e_proj_q8:   Vec<u8>,
    pub hnorm_gamma: Vec<f32>,
    pub h_proj_q8:   Vec<u8>,

    // Layer block — attn half.
    pub hc_attn_fn:   Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub attn_norm:    Vec<f32>,
    pub attn_q_a_q8:  Vec<u8>,
    pub gamma_q:      Vec<f32>,   // = attn_q_a_norm
    pub attn_q_b_q8:  Vec<u8>,
    pub attn_kv_q8:   Vec<u8>,
    pub gamma_kv:     Vec<f32>,   // = attn_kv_a_norm
    pub attn_sinks:   Vec<f32>,   // [n_head] per-head softmax sink logits
    pub w_o_a_q8:     Vec<u8>,
    pub w_o_b_q8:     Vec<u8>,

    // Layer block — FFN half.
    pub hc_ffn_fn:    Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub hc_ffn_base:  Vec<f32>,
    pub ffn_norm:     Vec<f32>,
    pub w_router:     Vec<f32>,   // = ffn_gate_inp
    pub router_bias:  Vec<f32>,   // = ffn_exp_probs_b
    pub sh_w_gate_q8: Vec<u8>,
    pub sh_w_up_q8:   Vec<u8>,
    pub sh_w_down_q8: Vec<u8>,

    // Output head.
    pub mtp_norm:        Vec<f32>,
    pub hc_head_fn:      Vec<f32>,
    pub hc_head_scale:   Vec<f32>,
    pub hc_head_base:    Vec<f32>,
    pub unit_gamma_hc:   Vec<f32>,  // [hc_dim] all-ones, for plain-RMS step
    pub base_lm_head_q8: Vec<u8>,   // base.output (LM head); shared with verifier

    // Shape + scalar params bundled for encode_mtp_draft.
    pub shape: MtpDraftShape,
}

#[cfg(target_os = "macos")]
impl MtpDraftBundle {
    /// Build the bundle from MTP file + base LM-head Q8_0 bytes. The base
    /// LM-head bytes (`base.output.weight` tensor) come from the base
    /// model's GGUF — the drafter shares vocab projection with the
    /// verifier.
    ///
    /// `vocab` and `shape` derive from the supplied dims.
    pub fn from_mtp(
        mtp: &MtpWeights,
        mtp_bytes: &[u8],
        base_lm_head_q8_bytes: Vec<u8>,
        vocab: usize,
    ) -> Result<Self> {
        // Dimensions from MtpWeights are the source of truth — they match
        // the MTP architecture (DS4 V4 Flash constants).
        let n_embd = mtp.enorm.n_elements() as usize;     // 4096
        let n_hc = mtp.hc_head_base.n_elements() as usize; // 4
        let n_lora_q = mtp.attn_q_a_norm.n_elements() as usize;  // 128
        let head_dim = mtp.attn_kv_a_norm.n_elements() as usize; // 512
        let n_head = mtp.attn_sinks.n_elements() as usize;       // 64
        let kv_row = head_dim;                            // MLA: n_lora_kv = head_dim
        let n_experts = mtp.n_experts as usize;
        let d_ffn = mtp.d_ffn as usize;
        // Shared-expert intermediate width from ffn_gate_shexp shape [n_embd, n_ff_exp].
        let shared_dim = mtp.ffn_gate_shexp.info.dims[1] as u32;

        // Output-projection geometry — DS4 V4 Flash uses n_out_group=8 +
        // n_lora_o=1024 (per antirez ds4.c:91-93). group_dim is the per-group
        // input dim for the per-group matmul = head_dim × (n_head/n_groups).
        let n_groups = 8;                                          // DS4_N_OUT_GROUP
        let n_lora_o = 1024;                                       // DS4_N_LORA_O
        let out_low_dim = n_groups * n_lora_o;                     // 8192
        let group_dim = head_dim * (n_head / n_groups);            // 512 * 8 = 4096

        let shape = MtpDraftShape {
            n_embd, n_hc, n_lora_q, n_head, head_dim, kv_row,
            n_groups, n_lora_o, group_dim, out_low_dim,
            n_experts, d_ffn, shared_dim, vocab,
            // DS4 V4 Flash hc_collapse sinkhorn iteration count (antirez
            // DS4_N_SINKHORN / ds4_v4_flash defaults = 20). MUST match the base
            // model's hc_sinkhorn_iter: with too few iterations the COMB section
            // of hc_split doesn't converge to doubly-stochastic (rows sum < 1),
            // producing a wrong HC-mixing matrix → wrong attn residual → wrong
            // verifier logits → ~0% spec-decode accept on drafts[1+]. (Was 5;
            // the verifier + MTP drafter both read shape.sinkhorn_iters.)
            sinkhorn_iters: 20,
            hc_eps: 1e-6,
            rms_eps: 1e-5,
            flash_scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let hc_dim = n_hc * n_embd;

        Ok(Self {
            // Input mix.
            enorm_gamma: mtp.enorm.dequant_f32(mtp_bytes)?,
            e_proj_q8:   mtp.e_proj.raw_bytes(mtp_bytes)?.to_vec(),
            hnorm_gamma: mtp.hnorm.dequant_f32(mtp_bytes)?,
            h_proj_q8:   mtp.h_proj.raw_bytes(mtp_bytes)?.to_vec(),

            // Layer attn half.
            hc_attn_fn:    mtp.hc_attn_fn.dequant_f32(mtp_bytes)?,
            hc_attn_scale: mtp.hc_attn_scale.dequant_f32(mtp_bytes)?,
            hc_attn_base:  mtp.hc_attn_base.dequant_f32(mtp_bytes)?,
            attn_norm:     mtp.attn_norm.dequant_f32(mtp_bytes)?,
            attn_q_a_q8:   mtp.attn_q_a.raw_bytes(mtp_bytes)?.to_vec(),
            gamma_q:       mtp.attn_q_a_norm.dequant_f32(mtp_bytes)?,
            attn_q_b_q8:   mtp.attn_q_b.raw_bytes(mtp_bytes)?.to_vec(),
            attn_kv_q8:    mtp.attn_kv.raw_bytes(mtp_bytes)?.to_vec(),
            gamma_kv:      mtp.attn_kv_a_norm.dequant_f32(mtp_bytes)?,
            attn_sinks:    mtp.attn_sinks.dequant_f32(mtp_bytes)?,
            w_o_a_q8:      mtp.attn_output_a.raw_bytes(mtp_bytes)?.to_vec(),
            w_o_b_q8:      mtp.attn_output_b.raw_bytes(mtp_bytes)?.to_vec(),

            // Layer FFN half.
            hc_ffn_fn:    mtp.hc_ffn_fn.dequant_f32(mtp_bytes)?,
            hc_ffn_scale: mtp.hc_ffn_scale.dequant_f32(mtp_bytes)?,
            hc_ffn_base:  mtp.hc_ffn_base.dequant_f32(mtp_bytes)?,
            ffn_norm:     mtp.ffn_norm.dequant_f32(mtp_bytes)?,
            w_router:     mtp.ffn_gate_inp.dequant_f32(mtp_bytes)?,
            router_bias:  mtp.ffn_exp_probs_b.dequant_f32(mtp_bytes)?,
            sh_w_gate_q8: mtp.ffn_gate_shexp.raw_bytes(mtp_bytes)?.to_vec(),
            sh_w_up_q8:   mtp.ffn_up_shexp.raw_bytes(mtp_bytes)?.to_vec(),
            sh_w_down_q8: mtp.ffn_down_shexp.raw_bytes(mtp_bytes)?.to_vec(),

            // Output head.
            mtp_norm:        mtp.norm.dequant_f32(mtp_bytes)?,
            hc_head_fn:      mtp.hc_head_fn.dequant_f32(mtp_bytes)?,
            hc_head_scale:   mtp.hc_head_scale.dequant_f32(mtp_bytes)?,
            hc_head_base:    mtp.hc_head_base.dequant_f32(mtp_bytes)?,
            unit_gamma_hc:   vec![1.0; hc_dim],
            base_lm_head_q8: base_lm_head_q8_bytes,

            shape,
        })
    }

    /// Upload all weights to the current scope. Returns a `MtpDraftUploaded`
    /// owning the DeferredBufs; the caller borrows from it to construct the
    /// per-call MtpDraftInputMix / MtpDraftLayerWeights / MtpDraftOutputHead
    /// reference structs.
    ///
    /// First call per `(scope, weight)` pair uploads via `cached_weight_buffer`;
    /// subsequent calls reuse the same metal::Buffer (identity-keyed cache).
    ///
    /// Lifetime: the returned `MtpDraftUploaded` borrows from `self` (the
    /// bundle) only — DeferredBuf is a value type holding a cloned
    /// metal::Buffer, so scope's lifetime doesn't propagate. This means
    /// the caller can then mutably borrow `scope` to call
    /// `encode_mtp_draft` without conflict.
    pub fn upload_to_scope(&self, scope: &BatchScope<'_>) -> MtpDraftUploaded<'_> {
        let q8_proj = |bytes: &[u8], n_weights: usize| -> DeferredBuf {
            scope.weight_q8_0_raw(bytes, n_weights)
        };
        let f32_buf = |slice: &[f32]| -> DeferredBuf { scope.weight_f32(slice) };

        let shape = &self.shape;
        let hc_dim = shape.n_hc * shape.n_embd;
        let q_dim = shape.n_head * shape.head_dim;
        let mix_hc = 2 * shape.n_hc + shape.n_hc * shape.n_hc;
        let _ = mix_hc;

        MtpDraftUploaded {
            bundle: self,
            // Input mix DeferredBufs.
            d_enorm_gamma: f32_buf(&self.enorm_gamma),
            d_e_proj_q8:   q8_proj(&self.e_proj_q8, shape.n_embd * shape.n_embd),
            d_hnorm_gamma: f32_buf(&self.hnorm_gamma),
            d_h_proj_q8:   q8_proj(&self.h_proj_q8, shape.n_embd * shape.n_embd),
            // Layer attn-half DeferredBufs.
            d_hc_attn_fn:    f32_buf(&self.hc_attn_fn),
            d_hc_attn_scale: f32_buf(&self.hc_attn_scale),
            d_hc_attn_base:  f32_buf(&self.hc_attn_base),
            d_attn_norm:     f32_buf(&self.attn_norm),
            d_attn_q_a_q8:   q8_proj(&self.attn_q_a_q8, shape.n_lora_q * shape.n_embd),
            d_gamma_q:       f32_buf(&self.gamma_q),
            d_attn_q_b_q8:   q8_proj(&self.attn_q_b_q8, q_dim * shape.n_lora_q),
            d_attn_kv_q8:    q8_proj(&self.attn_kv_q8, shape.kv_row * shape.n_embd),
            d_gamma_kv:      f32_buf(&self.gamma_kv),
            d_attn_sinks:    f32_buf(&self.attn_sinks),
            d_w_o_a_q8:      q8_proj(&self.w_o_a_q8, shape.out_low_dim * shape.group_dim),
            d_w_o_b_q8:      q8_proj(&self.w_o_b_q8, shape.n_embd * shape.out_low_dim),
            // Layer FFN-half DeferredBufs.
            d_hc_ffn_fn:    f32_buf(&self.hc_ffn_fn),
            d_hc_ffn_scale: f32_buf(&self.hc_ffn_scale),
            d_hc_ffn_base:  f32_buf(&self.hc_ffn_base),
            d_ffn_norm:     f32_buf(&self.ffn_norm),
            d_w_router:     f32_buf(&self.w_router),
            d_router_bias:  f32_buf(&self.router_bias),
            // Output head DeferredBufs.
            d_mtp_norm:        f32_buf(&self.mtp_norm),
            d_hc_head_fn:      f32_buf(&self.hc_head_fn),
            d_hc_head_scale:   f32_buf(&self.hc_head_scale),
            d_hc_head_base:    f32_buf(&self.hc_head_base),
            d_unit_gamma_hc:   f32_buf(&self.unit_gamma_hc),
            d_base_lm_head_q8: q8_proj(&self.base_lm_head_q8, hc_dim / shape.n_hc * shape.vocab),
        }
    }
}

/// Phase 3 Step 4b — DeferredBufs uploaded into the current scope, ready
/// to feed the encode_mtp_draft reference structs. Returned by
/// `MtpDraftBundle::upload_to_scope`.
#[cfg(target_os = "macos")]
pub struct MtpDraftUploaded<'a> {
    pub bundle: &'a MtpDraftBundle,

    // Input mix.
    pub d_enorm_gamma: DeferredBuf,
    pub d_e_proj_q8:   DeferredBuf,
    pub d_hnorm_gamma: DeferredBuf,
    pub d_h_proj_q8:   DeferredBuf,

    // Layer attn-half.
    pub d_hc_attn_fn:    DeferredBuf,
    pub d_hc_attn_scale: DeferredBuf,
    pub d_hc_attn_base:  DeferredBuf,
    pub d_attn_norm:     DeferredBuf,
    pub d_attn_q_a_q8:   DeferredBuf,
    pub d_gamma_q:       DeferredBuf,
    pub d_attn_q_b_q8:   DeferredBuf,
    pub d_attn_kv_q8:    DeferredBuf,
    pub d_gamma_kv:      DeferredBuf,
    pub d_attn_sinks:    DeferredBuf,
    pub d_w_o_a_q8:      DeferredBuf,
    pub d_w_o_b_q8:      DeferredBuf,

    // Layer FFN-half.
    pub d_hc_ffn_fn:    DeferredBuf,
    pub d_hc_ffn_scale: DeferredBuf,
    pub d_hc_ffn_base:  DeferredBuf,
    pub d_ffn_norm:     DeferredBuf,
    pub d_w_router:     DeferredBuf,
    pub d_router_bias:  DeferredBuf,

    // Output head.
    pub d_mtp_norm:        DeferredBuf,
    pub d_hc_head_fn:      DeferredBuf,
    pub d_hc_head_scale:   DeferredBuf,
    pub d_hc_head_base:    DeferredBuf,
    pub d_unit_gamma_hc:   DeferredBuf,
    pub d_base_lm_head_q8: DeferredBuf,
}

#[cfg(target_os = "macos")]
impl<'a> MtpDraftUploaded<'a> {
    /// Construct the input-mix reference struct from the uploaded DeferredBufs.
    pub fn input_mix(&'a self) -> MtpDraftInputMix<'a> {
        MtpDraftInputMix {
            enorm_gamma: &self.d_enorm_gamma,
            e_proj_q8:   &self.d_e_proj_q8,
            hnorm_gamma: &self.d_hnorm_gamma,
            h_proj_q8:   &self.d_h_proj_q8,
        }
    }

    /// Construct the layer-weights reference struct from the uploaded DeferredBufs.
    pub fn layer(&'a self) -> MtpDraftLayerWeights<'a> {
        MtpDraftLayerWeights {
            hc_attn_fn:    &self.d_hc_attn_fn,
            hc_attn_scale: &self.d_hc_attn_scale,
            hc_attn_base:  &self.d_hc_attn_base,
            attn_norm:     &self.d_attn_norm,
            attn_q_a_q8:   &self.d_attn_q_a_q8,
            gamma_q:       &self.d_gamma_q,
            attn_q_b_q8:   &self.d_attn_q_b_q8,
            attn_kv_q8:    &self.d_attn_kv_q8,
            gamma_kv:      &self.d_gamma_kv,
            attn_sinks:    &self.d_attn_sinks,
            w_o_a_q8:      &self.d_w_o_a_q8,
            w_o_b_q8:      &self.d_w_o_b_q8,
            hc_ffn_fn:     &self.d_hc_ffn_fn,
            hc_ffn_scale:  &self.d_hc_ffn_scale,
            hc_ffn_base:   &self.d_hc_ffn_base,
            ffn_norm:      &self.d_ffn_norm,
            w_router:      &self.d_w_router,
            router_bias:   &self.d_router_bias,
            // F32 shared-expert vecs are empty — the q8 path
            // (DS4_Q8_PROJ default true) handles MTP shared experts.
            sh_w_gate:     &[],
            sh_w_up:       &[],
            sh_w_down:     &[],
            sh_w_gate_q8:  &self.bundle.sh_w_gate_q8,
            sh_w_up_q8:    &self.bundle.sh_w_up_q8,
            sh_w_down_q8:  &self.bundle.sh_w_down_q8,
        }
    }

    /// Construct the output-head reference struct from the uploaded DeferredBufs.
    pub fn output_head(&'a self) -> MtpDraftOutputHead<'a> {
        MtpDraftOutputHead {
            unit_gamma_hc:   &self.d_unit_gamma_hc,
            hc_head_fn:      &self.d_hc_head_fn,
            hc_head_scale:   &self.d_hc_head_scale,
            hc_head_base:    &self.d_hc_head_base,
            mtp_norm:        &self.d_mtp_norm,
            base_lm_head_q8: &self.d_base_lm_head_q8,
        }
    }
}

/// Phase 3 Step 4d — chained MTP drafting: produce K speculative tokens
/// by calling `encode_mtp_draft_with_out_hc` K times, threading each
/// draft's `out_hc` into the next draft's `prev_hc` (antirez ping-pong
/// pattern at ds4.c:16154-16177).
///
/// Each iteration:
///   1. Open fresh scope (each draft requires its own flush + argmax +
///      next-token embed extraction).
///   2. Upload prev_hc + token_embed + bundle.
///   3. encode_mtp_draft_with_out_hc → (out_hc, logits).
///   4. flush_and_read logits → host vec.
///   5. argmax(logits) → draft_token.
///   6. Keep out_hc as the next iter's prev_hc (metal::Buffer survives
///      across scopes; reference-counted, lives until our DeferredBuf
///      handle drops).
///   7. Increment slot + pos for the MTP KV cache.
///
/// `extract_embed(token_id) -> Vec<f32>` extracts one row of the BASE
/// model's `token_embd` (caller closure — bundle doesn't hold base
/// weights).
///
/// Returns `Vec<i32>` of length `n_drafts`. `drafts[0] = seed_token`'s
/// MTP prediction; `drafts[i]` is `drafts[i-1]`'s MTP prediction.
#[cfg(target_os = "macos")]
pub fn run_mtp_chain_drafts(
    disp: &crate::MetalDispatcher,
    initial_prev_hc: &[f32],
    seed_token: i32,
    initial_pos: u32,
    initial_slot: u32,
    // `initial_mtp_n_raw`: MTP cache fill count BEFORE this chain starts
    // (antirez `mtp_n_raw`).  0 at the very first draft of a session; grows
    // by `n_drafts` per call the caller chooses to commit.  Defines the
    // causal attn window for each draft: iter i sees
    // `initial_mtp_n_raw + i + 1` cache rows.
    initial_mtp_n_raw: u32,
    n_drafts: usize,
    bundle: &MtpDraftBundle,
    extract_embed: &mut dyn FnMut(i32) -> Vec<f32>,
    layer_params: &ds4_engine::attn_dispatch::LayerParams,
    mtp_layer_idx: u32,
    raw_cap: u32,
) -> Result<Vec<i32>> {
    let hc_dim = bundle.shape.n_hc * bundle.shape.n_embd;
    anyhow::ensure!(
        initial_prev_hc.len() == hc_dim,
        "run_mtp_chain_drafts: initial_prev_hc {} != hc_dim {}",
        initial_prev_hc.len(), hc_dim,
    );

    // Upload the initial prev_hc into a stand-alone scope, then keep the
    // buffer handle alive across draft iters.
    let (mut prev_hc_buf, mut prev_hc_elems) = {
        let scope = disp.batch_scope();
        let db = scope.upload_f32(initial_prev_hc);
        (db.buffer().clone(), db.len())
        // scope drops here — buffer survives via the clone above.
    };

    let mut drafts: Vec<i32> = Vec::with_capacity(n_drafts);
    let mut last_token = seed_token;
    let mut pos = initial_pos;
    let mut slot = initial_slot;

    for draft_iter in 0..n_drafts {
        let token_embed = extract_embed(last_token);
        anyhow::ensure!(
            token_embed.len() == bundle.shape.n_embd,
            "extract_embed returned {} elems, expected n_embd={}",
            token_embed.len(), bundle.shape.n_embd,
        );

        // Wrap the persistent prev_hc buffer as a DeferredBuf for this scope.
        let prev_hc_db = DeferredBuf::from_external_buffer(prev_hc_buf.clone(), prev_hc_elems);

        let mut scope = disp.batch_scope();
        scope.set_q8_proj(true);
        let token_embed_db = scope.upload_f32(&token_embed);
        let dbufs = bundle.upload_to_scope(&scope);

        // MTP attn window for THIS draft: covers initial_mtp_n_raw + draft_iter
        // previously-written rows + the row being written now → total
        // `initial_mtp_n_raw + draft_iter + 1` rows.  Pass
        // `attn_base_pos = initial_mtp_n_raw + draft_iter`; the kernel's
        // per-q causal mask gives q=0 a window of attn_base_pos+1.
        let attn_base_pos = initial_mtp_n_raw + draft_iter as u32;
        let (out_hc, logits) = scope.encode_mtp_draft_with_out_hc(
            &prev_hc_db, &token_embed_db,
            dbufs.input_mix(), dbufs.layer(), dbufs.output_head(),
            bundle.shape, layer_params,
            mtp_layer_idx, raw_cap, slot, pos, attn_base_pos,
        )?;

        // Capture the out_hc buffer BEFORE flush_and_read consumes scope.
        let out_hc_buf = out_hc.buffer().clone();
        let out_hc_elems = out_hc.len();

        let logits_vec = scope.flush_and_read(&logits);

        // argmax → next draft token.
        let mut top_idx = 0usize;
        let mut top_v = f32::NEG_INFINITY;
        for (i, &v) in logits_vec.iter().enumerate() {
            if v > top_v { top_v = v; top_idx = i; }
        }
        let next_token = top_idx as i32;
        drafts.push(next_token);

        // Thread for next iter.
        prev_hc_buf = out_hc_buf;
        prev_hc_elems = out_hc_elems;
        last_token = next_token;
        pos += 1;
        slot += 1;
    }

    Ok(drafts)
}

#[cfg(not(target_os = "macos"))]
pub struct MtpDraftBundle;  // placeholder for non-macOS builds
