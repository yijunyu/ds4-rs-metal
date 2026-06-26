//! M4 #225 Phase B3c — Mac-only runtime entrypoint.
//!
//! `run_decode_argmax_metal(path, prompt_tokens, n_decode)` loads the DS4
//! GGUF, builds `ComposedModelWeights::from_views`, constructs a
//! `MetalDispatcher` (both `KernelDispatcher` + `AttentionDispatcher`),
//! runs a Rust-side prefill over `prompt_tokens` followed by `n_decode`
//! greedy-argmax decode steps, and returns the generated token stream
//! for comparison against the antirez baseline.
//!
//! On Linux this function returns an error — the Metal back end is
//! macOS-only. The main.rs caller is `#[cfg(target_os = "macos")]` so
//! the Linux side of the binary never reaches here at runtime.

use anyhow::{anyhow, bail, Result};
use std::path::Path;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::{
    decode_step_with_attn, AttnStepState, ComposedModelWeights, DecodeConfig,
};
use ds4_engine::gguf::{validate_ds4_layout, GgmlType, GgufFile, ModelManifest};
use ds4_engine::layer_view::{f16_bits_to_f32, LayerViews, TensorHandle};

use crate::MetalDispatcher;

/// All the heavy state needed to drive our decode pipeline on a real
/// DS4 GGUF. Built once per session.
pub struct DecodeRunner {
    pub manifest: ModelManifest,
    pub composed: ComposedModelWeights,
    // Embedding stored COMPACT: for F16/F32 `token_embd` (the DS4 layout uses
    // F16) we keep only the tensor handle and dequant ONE row per `embed()`
    // call straight from the mmap — saving the ~2 GB f32 [vocab × d_model] copy
    // (129k × 4096 × 4 B ≈ 2.1 GB). Bit-identical to the old full-dequant path.
    // `embed_fallback` holds a dense f32 table only for the rare case of a
    // quantized embedding tensor (not F16/F32), where per-row dequant isn't wired.
    embed_handle: TensorHandle,
    embed_fallback: Option<Vec<f32>>,
    pub dispatcher: MetalDispatcher,
    pub raw_cap: u32,
    // Retain the mmap so the dispatcher's mmap_ptr fields stay valid.
    // Drop order: `dispatcher` before `_views_keepalive` (struct fields drop
    // in declaration order).
    _views_keepalive: LayerViews,
}

/// DS4_WEIGHT_STATS diagnostic — sum the owned weight buffers in
/// `ComposedModelWeights` by category, in GB. Reveals what makes up the
/// anonymous malloc slab (the idle-compressed ~31 GB) so the no-copy refactor
/// can target the biggest contributors. f32 Vecs cost 4 B/elem; q8 Vec<u8> 1 B.
fn log_weight_stats(c: &ComposedModelWeights) {
    let f = |v: &Vec<f32>| v.len() * 4;
    let u = |v: &[u8]| v.len();
    let (mut attn_lora, mut attn_out, mut shared, mut q8, mut misc, mut moe) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut misc_hc, mut misc_comp, mut misc_idx, mut misc_gamma) = (0usize, 0usize, 0usize, 0usize);
    for l in &c.layers {
        let a = &l.attn;
        attn_lora += f(&a.attn_q_a) + f(&a.attn_q_b) + f(&a.attn_kv);
        attn_out += f(&a.w_o_a) + f(&a.w_o_b);
        shared += f(&a.w_shared_gate) + f(&a.w_shared_up) + f(&a.w_shared_down);
        q8 += u(&a.attn_q_a_q8) + u(&a.attn_q_b_q8) + u(&a.attn_kv_q8)
            + u(&a.w_o_a_q8) + u(&a.w_o_b_q8)
            + u(&a.w_shared_gate_q8) + u(&a.w_shared_up_q8) + u(&a.w_shared_down_q8);
        misc += f(&a.hc_attn_fn) + f(&a.hc_attn_scale) + f(&a.hc_attn_base)
            + f(&a.hc_ffn_fn) + f(&a.hc_ffn_scale) + f(&a.hc_ffn_base)
            + f(&a.hc_norm_gamma) + f(&a.hc_ffn_norm_gamma)
            + f(&a.qkv_gamma_q) + f(&a.qkv_gamma_kv) + f(&a.attn_sinks)
            + f(&a.attn_compressor_kv) + f(&a.attn_compressor_gate)
            + f(&a.attn_compressor_ape) + f(&a.attn_compressor_norm)
            + f(&a.indexer_compressor_kv) + f(&a.indexer_compressor_gate)
            + f(&a.indexer_compressor_ape) + f(&a.indexer_compressor_norm)
            + f(&a.indexer_attn_q_b) + f(&a.indexer_proj);
        misc_hc += f(&a.hc_attn_fn) + f(&a.hc_attn_scale) + f(&a.hc_attn_base)
            + f(&a.hc_ffn_fn) + f(&a.hc_ffn_scale) + f(&a.hc_ffn_base)
            + f(&a.hc_norm_gamma) + f(&a.hc_ffn_norm_gamma);
        misc_comp += f(&a.attn_compressor_kv) + f(&a.attn_compressor_gate)
            + f(&a.attn_compressor_ape) + f(&a.attn_compressor_norm);
        misc_idx += f(&a.indexer_compressor_kv) + f(&a.indexer_compressor_gate)
            + f(&a.indexer_compressor_ape) + f(&a.indexer_compressor_norm)
            + f(&a.indexer_attn_q_b) + f(&a.indexer_proj);
        misc_gamma += f(&a.qkv_gamma_q) + f(&a.qkv_gamma_kv) + f(&a.attn_sinks);
        let m = &l.moe;
        moe += f(&m.w_attn) + f(&m.w_router) + f(&m.router_bias)
            + f(&m.attn_norm_gamma) + f(&m.ffn_norm_gamma)
            + f(&m.w_gate_exps) + f(&m.w_up_exps) + f(&m.w_down_exps);
    }
    let gb = |b: usize| b as f64 / 1e9;
    eprintln!(
        "ds4-weight-stats (GB): attn_lora_f32={:.2} attn_out_f32={:.2} \
         shared_f32={:.2} misc_f32={:.2} moe_f32={:.2} lm_head_f32={:.2} | \
         q8_copies={:.2} lm_head_q8={:.2} lm_head_q4={:.2} | \
         TOTAL_f32={:.2} TOTAL_q8={:.2}",
        gb(attn_lora), gb(attn_out), gb(shared), gb(misc), gb(moe),
        gb(f(&c.lm_head)),
        gb(q8), gb(u(&c.lm_head_q8)), gb(u(&c.lm_head_q4)),
        gb(attn_lora + attn_out + shared + misc + moe + f(&c.lm_head)),
        gb(q8 + u(&c.lm_head_q8) + u(&c.lm_head_q4)),
    );
    eprintln!(
        "ds4-weight-stats misc_f32 breakdown (GB): hc={:.2} compressor={:.2} \
         indexer={:.2} gamma/sinks={:.3}",
        gb(misc_hc), gb(misc_comp), gb(misc_idx), gb(misc_gamma),
    );
}

impl DecodeRunner {
    /// Open the GGUF, validate the DS4 layout, build composed weights +
    /// embedding table, init the Metal dispatcher, and load per-layer
    /// quantized expert weight tables.
    pub fn open(path: &Path, raw_cap: u32) -> Result<Self> {
        let manifest = validate_ds4_layout(path)?;
        let views = LayerViews::open(path, manifest.n_layers)?;

        let defaults = DefaultsDs4::ds4_v4_flash();
        // Lean build: never allocate the f32 duplicates of the q8-backed weights
        // (the encoder/DecodeSession path reads them via q8). Avoids the
        // allocate-then-free that leaves ~22 GB in malloc's free-cache (inflated
        // phys_footprint). `free_dead_f32_weights` below is then a safety no-op.
        // The trait/run_argmax path must build via `from_views` (keeps f32).
        //
        // DS4_LEAN_WEIGHTS=0 ESCAPE HATCH: build non-lean (keep f32) for the
        // long-prompt staged/chain encoder branches that still read f32 weights
        // directly (attn output proj, etc.) and panic under lean. Costs ~22 GB of
        // anon f32 but restores correctness for all prompt lengths until those
        // branches are made q8-aware. `free_dead_f32_weights` is skipped too
        // (see ds4_server) so the f32 stays resident.
        let lean = std::env::var("DS4_LEAN_WEIGHTS").map(|v| v != "0").unwrap_or(true);
        let composed = if lean {
            ComposedModelWeights::from_views_lean(&views, &manifest, defaults)?
        } else {
            eprintln!("ds4-server: DS4_LEAN_WEIGHTS=0 — non-lean build (f32 weights kept; higher RAM)");
            ComposedModelWeights::from_views(&views, &manifest, defaults)?
        };

        if std::env::var("DS4_WEIGHT_STATS").is_ok() {
            log_weight_stats(&composed);
        }

        let embed_handle = views
            .global
            .get("embed")
            .ok_or_else(|| anyhow!("missing global tensor: embed (token_embd.weight)"))?
            .clone();
        let n = (manifest.vocab_size as usize) * (manifest.d_model as usize);
        anyhow::ensure!(
            embed_handle.n_elems() as usize == n,
            "embed tensor {} elems != vocab×d_model = {}×{}",
            embed_handle.n_elems(), manifest.vocab_size, manifest.d_model,
        );
        // Per-row dequant for F16/F32 (no 2 GB f32 copy); dense fallback only for
        // a quantized embedding (not expected for DS4, whose embed is F16).
        let embed_fallback = match embed_handle.ttype {
            GgmlType::F16 => {
                anyhow::ensure!(
                    views.bytes_for(&embed_handle)?.len() == n * 2,
                    "embed F16 byte size mismatch"
                );
                None
            }
            GgmlType::F32 => {
                anyhow::ensure!(
                    views.bytes_for(&embed_handle)?.len() == n * 4,
                    "embed F32 byte size mismatch"
                );
                None
            }
            _ => Some(views.dequant_f32_simple(&embed_handle)?),
        };

        let mut dispatcher = MetalDispatcher::new()?;
        // Load per-layer quantized experts into the dispatcher's
        // internal table — `moe_routed_step` resolves by layer_idx.
        let gguf = GgufFile::open(path)?;
        // Router expert-weight rescale from metadata (Flash 1.5, PRO 2.5).
        if let Some(s) = ds4_engine::gguf::meta_f32(
            &gguf,
            &["deepseek4.expert_weights_scale", "deepseek2.expert_weights_scale"],
        ) {
            ds4_engine::moe::set_router_scale(s);
        }
        dispatcher.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)?;

        Ok(Self {
            manifest,
            composed,
            embed_handle,
            embed_fallback,
            dispatcher,
            raw_cap,
            _views_keepalive: views,
        })
    }

    /// Phase 1 of the no-copy weight refactor: free the f32 weight buffers
    /// whose q8 equivalent the encoder hot path (`decode_token_via_first_half`)
    /// already uses, reclaiming ~6.5 GB of anonymous (idle-compressed) RAM.
    ///
    /// Freed only when the q8 source is present (exactly when the hot path
    /// selects q8 via `use_w_q8`): per layer `w_shared_gate/up/down` (when all
    /// three `*_q8` present). These f32 slices are never handed to
    /// `cached_weight_buffer` on the encoder path, so no `MTLBuffer` points
    /// into them — freeing is dangle-free; the only consumer was an f32
    /// length-validation in `shared_chain_batched` (now skipped on the q8 path).
    ///
    /// NOTE: `lm_head` f32 is deliberately NOT freed — the server samples from
    /// FULL vocab logits, whose tail (`tail_lm_head_batched`) reads the f32
    /// lm_head (the q8 lm_head buffer feeds only the GPU-argmax fast path).
    /// `single_buffer_encoder.rs:3409` panics if it's missing.
    ///
    /// SAFE ONLY for the `DecodeSession`/encoder path (what the server uses).
    /// The trait decode path (`run_argmax` → `decode_step_with_attn`) still
    /// reads the shared f32 buffers, so callers of THAT path must not call this.
    /// Returns bytes freed.
    pub fn free_dead_f32_weights(&mut self) -> usize {
        // The attn q/kv LoRA + output projections take the encoder's raw-q8
        // path only when DS4_Q8_PROJ is on (default) AND the raw q8 bytes are
        // present; the `weight_q8_0(f32)` re-quantize fallback (q8_proj but raw
        // empty) still reads the f32, so only free when the RAW bytes exist.
        let q8_proj = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true);
        let mut freed = 0usize;
        for l in &mut self.composed.layers {
            // Hash-routed layers (0/1/2) keep their f32 shared weights for the
            // long-context slow path's f32 `run_ffn_half` fallback — must match
            // `from_layer_view`'s `skip_shared` carve-out, else we'd free f32
            // the loader deliberately kept and re-introduce the 2026-06-02 crash.
            let is_hash_layer = l.moe.routing_table.is_some();
            let a = &mut l.attn;
            // Shared expert — hot path uses q8 whenever all three q8 present.
            if !is_hash_layer
                && !a.w_shared_gate_q8.is_empty()
                && !a.w_shared_up_q8.is_empty()
                && !a.w_shared_down_q8.is_empty()
            {
                freed += (a.w_shared_gate.len() + a.w_shared_up.len() + a.w_shared_down.len()) * 4;
                a.w_shared_gate = Vec::new();
                a.w_shared_up = Vec::new();
                a.w_shared_down = Vec::new();
            }
            // Attn q_a/q_b/kv LoRA — encoder q8_raw branch needs all three.
            if q8_proj
                && !a.attn_q_a_q8.is_empty()
                && !a.attn_q_b_q8.is_empty()
                && !a.attn_kv_q8.is_empty()
            {
                freed += (a.attn_q_a.len() + a.attn_q_b.len() + a.attn_kv.len()) * 4;
                a.attn_q_a = Vec::new();
                a.attn_q_b = Vec::new();
                a.attn_kv = Vec::new();
            }
            // Output projection (o_a grouped + o_b dense) — encoder q8 branch.
            if q8_proj && !a.w_o_a_q8.is_empty() && !a.w_o_b_q8.is_empty() {
                freed += (a.w_o_a.len() + a.w_o_b.len()) * 4;
                a.w_o_a = Vec::new();
                a.w_o_b = Vec::new();
            }
        }
        // lm_head f32 (~2.1GB): full-logits sampling now uses the q8 lm-head
        // matvec (tail_lm_head_full_q8), bit-identical when there's no q8
        // activation round-trip. Free the f32 when that q8 path is active.
        let lmhead_q8_active = !self.composed.lm_head_q8.is_empty()
            && std::env::var("DS4_Q8_LMHEAD").ok().as_deref() != Some("0")
            && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1");
        if lmhead_q8_active {
            freed += self.composed.lm_head.len() * 4;
            self.composed.lm_head = Vec::new();
        }
        // Return the freed pages to the OS. macOS malloc retains large freed
        // chunks on its free list (they show as MALLOC_LARGE (empty), still
        // resident in phys_footprint); pressure-relief madvise's them away so
        // the footprint actually reflects the ~25 GB freed here.
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" {
                fn malloc_zone_pressure_relief(
                    zone: *mut std::ffi::c_void,
                    goal: usize,
                ) -> usize;
            }
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
        freed
    }

    /// Look up the f32 embedding row for `token`. Bounds-checked. Dequants ONE
    /// row from the mmap'd F16/F32 `token_embd` (no 2 GB f32 table); the F16 path
    /// is byte-identical to `LayerViews::dequant_f32_simple`.
    pub fn embed(&self, token: i32) -> Result<Vec<f32>> {
        let d = self.manifest.d_model as usize;
        let v = self.manifest.vocab_size as usize;
        let t = token as usize;
        if t >= v {
            bail!("token id {token} out of vocab range {v}");
        }
        if let Some(tbl) = &self.embed_fallback {
            return Ok(tbl[t * d..(t + 1) * d].to_vec());
        }
        let raw = self._views_keepalive.bytes_for(&self.embed_handle)?;
        match self.embed_handle.ttype {
            GgmlType::F16 => {
                let row = &raw[t * d * 2..(t + 1) * d * 2];
                Ok(row
                    .chunks_exact(2)
                    .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            GgmlType::F32 => {
                let row = &raw[t * d * 4..(t + 1) * d * 4];
                Ok(row
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            // Quantized embed goes through embed_fallback (handled above).
            _ => unreachable!("embed_fallback set for quantized embedding"),
        }
    }

    /// CROSS-PATH per-layer RESIDUAL divergence probe (model-level). Pins WHERE
    /// the chunk-prefill and per-token-prefill residual streams first diverge —
    /// at the raw residual level (NOT the lagged comp-ring proxy, which hides
    /// drift in the compressor projection's nullspace). Both paths compute the
    /// layer-INPUT residual for the same absolute position `target_pos`; this
    /// returns, per layer, (layer, cos, max|Δ|) between them.
    ///
    /// Per-token side: prefill `prompt[..=target_pos]` (final per-layer capture =
    /// position `target_pos`, via the RESID_CAP thread-local). Chunk side: prefill
    /// the full prompt with DS4_CHUNK_HALF_CHECK (captures `prev_hc_k` per layer);
    /// read row `target_pos`. `target_pos` must be ≤ prompt.len()-2 (a chunk-init
    /// position). Diagnostic-only (mutates env + thread-local); not for production.
    pub fn residual_divergence_probe(
        &self,
        prompt: &[i32],
        target_pos: usize,
    ) -> Result<Vec<(usize, f64, f32)>> {
        fn cos(a: &[f32], b: &[f32]) -> f64 {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) { d += x as f64 * y as f64; na += (x as f64).powi(2); nb += (y as f64).powi(2); }
            d / (na.sqrt() * nb.sqrt()).max(1e-30)
        }
        fn maxd(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max) }

        anyhow::ensure!(
            target_pos + 1 < prompt.len(),
            "residual_divergence_probe: target_pos {target_pos} must be ≤ prompt.len()-2 ({})",
            prompt.len().saturating_sub(2)
        );
        let n_layers = self.composed.layers.len();
        let first = &self.composed.layers[0].attn.params;
        let hc_dim = first.n_hc as usize * first.d_embd as usize;

        let chunk_vars = ["DS4_PREFILL_CHUNK", "DS4_CHUNK_SWA_KFLASH", "DS4_CHUNK_ATTN_NOSYNC",
                          "DS4_CHUNK_BATCHED_IDX", "DS4_CHUNK_FUSED_COMP", "DS4_CHUNK_HALF_CHECK"];
        // The chunk-attention PERF knobs default to the fast batched path, but
        // honor a caller-set value so the attention-side drift can be localized
        // (e.g. DS4_CHUNK_BATCHED_IDX=0 collapses to fully per-position attention).
        // Snapshot the caller's intent before the per-token side wipes them.
        // SWA_KFLASH is NOT in this default-on set — it is known-incoherent at
        // chunk>raw_cap (tile-boundary NaN) and not part of the production stack;
        // defaulting it on poisoned this probe's chunk reference. It IS still wiped
        // and restored via `chunk_vars`, so a caller who sets it explicitly is
        // honored (to diff the faulty path), but the probe never enables it itself.
        let perf_knobs = ["DS4_CHUNK_ATTN_NOSYNC",
                          "DS4_CHUNK_BATCHED_IDX", "DS4_CHUNK_FUSED_COMP"];
        let caller_perf: Vec<(&str, Option<String>)> =
            perf_knobs.iter().map(|&k| (k, std::env::var(k).ok())).collect();

        // ── per-token: capture per-layer input residual at target_pos ──
        // DS4_CHAIN=0 is REQUIRED: the GPU-resident layer chain (default-on) keeps
        // cur_hc resident and does NOT write state.cur_hc back per layer, so the
        // capture would read stale CPU state. Chaining is bit-identical, so the
        // non-chained reference is the same residuals — just CPU-live per layer.
        for v in chunk_vars { std::env::remove_var(v); }
        std::env::set_var("DS4_CHAIN", "0");
        crate::single_buffer_encoder::resid_cap_set(true);
        {
            let mut s = DecodeSession::new(self);
            s.prefill(&prompt[..=target_pos])?; // final capture = position target_pos
        }
        let pt = crate::single_buffer_encoder::resid_cap_take().unwrap_or_default();
        std::env::remove_var("DS4_CHAIN");

        // ── chunk: prefill full prompt, HALF_CHECK captures prev_hc_k per layer ──
        if std::env::var("DS4_PREFILL_CHUNK").is_err() {
            std::env::set_var("DS4_PREFILL_CHUNK", "8192");
        }
        std::env::set_var("DS4_CHUNK_HALF_CHECK", "1");
        // Perf knobs: caller value wins, else default to the fast batched path ("1").
        for (k, caller) in &caller_perf {
            std::env::set_var(k, caller.as_deref().unwrap_or("1"));
        }
        // Prefill only through target_pos+1 (single chunk; chunk-init = [0..=target_pos]),
        // so the run length is tunable by target_pos — keeps the lever-B K=1-loop
        // validation tractable. HALF_CHECK row target_pos = the chunk's residual there.
        let chunk_prompt = &prompt[..=target_pos + 1];
        let k_positions = chunk_prompt.len() - 1; // chunk-init token count
        {
            let mut s = DecodeSession::new(self);
            s.prefill(chunk_prompt)?;
        }
        for v in chunk_vars { std::env::remove_var(v); }

        // ── compare per layer at target_pos ──
        let mut out = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let buf = self.dispatcher.kv_buffer_or_alloc(
                l as u32 + 2_000_000, k_positions * hc_dim * std::mem::size_of::<f32>());
            let ck = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, k_positions * hc_dim) };
            let ck_row = &ck[target_pos * hc_dim..(target_pos + 1) * hc_dim];
            let Some(pt_row) = pt.get(l).filter(|v| v.len() == hc_dim) else { continue };
            out.push((l, cos(pt_row, ck_row), maxd(pt_row, ck_row)));
        }
        Ok(out)
    }

    /// Fire-rate counter for the decode-phase all-zeros race: run `trials` fresh
    /// prefills (each ending in feed(last), the per-token decode whose logits seed
    /// step 0) and record the step-0 argmax token. Returns (n_allzeros, first_tokens)
    /// where n_allzeros counts trials whose logits collapsed to token 0 (the all-zeros
    /// signature). Honors caller env (DS4_CHAIN, perf knobs); forces only structural
    /// chunk vars. The systematic anchor: CHAIN on vs CHAIN=0 fire-rate.
    /// Returns per trial `(stream, step0_logits_all_zero)`. The bool DISTINGUISHES
    /// the two token-0 failure modes the streams alone can't: a true all-zero-logits
    /// CB FAULT (bool=true) vs genuine BOS-PREDICTION incoherence (argmax=0 on real,
    /// nonzero logits, bool=false). Critical because both yield a [0,0,..] stream.
    pub fn decode_allzeros_rate(
        &self,
        prompt: &[i32],
        trials: u32,
        n_decode: u32,
    ) -> Result<Vec<(Vec<i32>, bool)>> {
        fn argmax(l: &[f32]) -> i32 {
            let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() { if v > bv { bv = v; best = i as i32; } }
            best
        }
        // Honor a caller-set chunk size (e.g. DS4_PREFILL_CHUNK=128 to keep tiles
        // within raw_cap); default to one whole-prompt chunk.
        if std::env::var("DS4_PREFILL_CHUNK").is_err() {
            if std::env::var("DS4_PREFILL_CHUNK").is_err() {
            std::env::set_var("DS4_PREFILL_CHUNK", "8192");
        }
        }
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
        // Diagnostic: OBSERVE all-zeros rather than have the production guard bail.
        std::env::set_var("DS4_NO_ZERO_HIDDEN_GUARD", "1");
        let mut out_all = Vec::with_capacity(trials as usize);
        for _ in 0..trials {
            let mut s = DecodeSession::new(self);
            s.prefill(prompt)?;
            let step0_all_zero = s.logits().iter().all(|&v| v == 0.0); // CB FAULT vs BOS
            let mut out = Vec::with_capacity(n_decode as usize);
            for _ in 0..n_decode {
                let t = argmax(s.logits());
                out.push(t);
                s.step(t)?;
            }
            out_all.push((out, step0_all_zero));
        }
        std::env::remove_var("DS4_PREFILL_CHUNK");
        std::env::remove_var("DS4_CHUNK_MAX_CTX");
        std::env::remove_var("DS4_NO_ZERO_HIDDEN_GUARD");
        Ok(out_all)
    }

    /// DECODE-phase first-divergence: run `prefill + n_decode argmax steps` TWICE
    /// (fresh sessions). The prefill is provably deterministic (see
    /// `chunk_first_divergence`), so both runs reach identical post-prefill state;
    /// any divergence in the two decoded token streams is a DECODE-phase race. Returns
    /// (stream_a, stream_b, first_divergent_step). Honors caller env (chunk perf
    /// knobs, DS4_CHAIN); forces only the structural chunk-prefill vars.
    pub fn chunk_decode_divergence(
        &self,
        prompt: &[i32],
        n_decode: u32,
    ) -> Result<(Vec<i32>, Vec<i32>, Option<usize>)> {
        fn argmax(l: &[f32]) -> i32 {
            let mut best = 0i32; let mut bv = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() { if v > bv { bv = v; best = i as i32; } }
            best
        }
        if std::env::var("DS4_PREFILL_CHUNK").is_err() {
            std::env::set_var("DS4_PREFILL_CHUNK", "8192");
        }
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");
        // Diagnostic: OBSERVE all-zeros rather than have the production guard bail.
        std::env::set_var("DS4_NO_ZERO_HIDDEN_GUARD", "1");
        let run = |runner: &Self| -> Result<Vec<i32>> {
            let mut s = DecodeSession::new(runner);
            s.prefill(prompt)?;
            let mut out = Vec::with_capacity(n_decode as usize);
            for _ in 0..n_decode {
                let t = argmax(s.logits());
                out.push(t);
                s.step(t)?;
            }
            Ok(out)
        };
        let a = run(self)?;
        let b = run(self)?;
        std::env::remove_var("DS4_PREFILL_CHUNK");
        std::env::remove_var("DS4_CHUNK_MAX_CTX");
        std::env::remove_var("DS4_NO_ZERO_HIDDEN_GUARD");
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        Ok((a, b, first))
    }

    /// FIRST-DIVERGENCE race localizer: run the SAME chunk prefill TWICE with the
    /// SAME config and report, per layer, the run-to-run divergence of the captured
    /// layer-input residual (HALF_CHECK `prev_hc_k`). For deterministic code every
    /// layer is bit-identical (cos 1.0, maxΔ 0); the FIRST layer with cos<1 / maxΔ>0
    /// is where a data RACE first manifests this run — direct localization, no
    /// trial-and-error knob sweep.
    ///
    /// Honors caller env for `DS4_CHAIN` and the chunk perf knobs, so the cross-layer
    /// chaining race (chaining on) and the finer long-context race (`DS4_CHAIN=0`) can
    /// be isolated. Forces only the structural capture vars. `target_pos` ≤ len-2.
    pub fn chunk_first_divergence(
        &self,
        prompt: &[i32],
        target_pos: usize,
    ) -> Result<Vec<(usize, f64, f32)>> {
        fn cos(a: &[f32], b: &[f32]) -> f64 {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) { d += x as f64 * y as f64; na += (x as f64).powi(2); nb += (y as f64).powi(2); }
            d / (na.sqrt() * nb.sqrt()).max(1e-30)
        }
        fn maxd(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max) }

        anyhow::ensure!(
            target_pos + 1 < prompt.len(),
            "chunk_first_divergence: target_pos {target_pos} must be ≤ prompt.len()-2 ({})",
            prompt.len().saturating_sub(2)
        );
        let n_layers = self.composed.layers.len();
        let first = &self.composed.layers[0].attn.params;
        let hc_dim = first.n_hc as usize * first.d_embd as usize;

        // Structural capture vars only (force chunk + per-layer HALF_CHECK capture).
        // CHAIN and the perf knobs are left to the caller so each race is isolable.
        if std::env::var("DS4_PREFILL_CHUNK").is_err() {
            std::env::set_var("DS4_PREFILL_CHUNK", "8192");
        }
        std::env::set_var("DS4_CHUNK_HALF_CHECK", "1");
        std::env::set_var("DS4_CHUNK_MAX_CTX", "0");

        let chunk_prompt = &prompt[..=target_pos + 1];
        let k_positions = chunk_prompt.len() - 1;
        // Persistent compressor ring sizes (mirror single_buffer_encoder): the
        // main/index comp rings the GPU-resident compressor WRITES during prefill
        // and the DECODE reads — but the residual does NOT depend on. A race in the
        // ring write leaves prev_hc_k bit-identical (the compute path) yet makes
        // decode nondeterministic. comp_ring_rows = max(raw_cap, 32768/4 + 8).
        let ring_rows = (self.raw_cap as usize).max(32768 / 4 + 8);
        const IDX_HD: usize = ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize;
        // capture[0]=prev_hc_k residual, [1]=comp_ring, [2]=index_comp_ring, per layer.
        let read_capture = |runner: &Self| -> Vec<[Vec<f32>; 4]> {
            (0..n_layers).map(|l| {
                let hd = runner.composed.layers[l].attn.params.head_dim as usize;
                let resid = {
                    let b = runner.dispatcher.kv_buffer_or_alloc(
                        l as u32 + 2_000_000, k_positions * hc_dim * std::mem::size_of::<f32>());
                    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, k_positions * hc_dim) }.to_vec()
                };
                let comp = {
                    let b = runner.dispatcher.comp_ring_or_alloc(l as u32, ring_rows * hd * 4);
                    let n = b.length() as usize / 4;
                    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
                };
                let idx = {
                    let b = runner.dispatcher.index_comp_ring_or_alloc(l as u32, ring_rows * IDX_HD * 4);
                    let n = b.length() as usize / 4;
                    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
                };
                // fp8 KV ring (decode-consumed; NOT covered by the residual capture).
                let kv = {
                    let row = runner.composed.layers[l].attn.params.n_lora_kv as usize;
                    let b = runner.dispatcher.kv_buffer_or_alloc(
                        l as u32, runner.raw_cap as usize * row * std::mem::size_of::<f32>());
                    let n = b.length() as usize / 4;
                    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
                };
                [resid, comp, idx, kv]
            }).collect()
        };

        // RUN A → snapshot the per-layer capture to CPU before later runs overwrite it.
        { let mut s = DecodeSession::new(self); s.prefill(chunk_prompt)?; }
        let run_a = read_capture(self);
        // DS4_FD_CHUNK_B=K: run B (and later trials) uses chunk size K while run A
        // keeps the caller's chunk size — a cross-CONFIG diff (e.g. 128 vs 256)
        // that cancels the common chunk-vs-pertoken fp32 drift and shows only the
        // bug introduced by the larger chunk.
        if let Ok(cb) = std::env::var("DS4_FD_CHUNK_B") {
            std::env::set_var("DS4_PREFILL_CHUNK", cb);
        }
        // DS4_FD_TRIALS=N (default 1): run N identical prefills in this same
        // model-load, diff EACH against run A, keep the worst per layer/buffer.
        // Amortizes the ~3-min model load — the race is intermittent (~10-25%),
        // so volume is what buys statistical separation.
        let extra: u32 = std::env::var("DS4_FD_TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut run_b = { let mut s = DecodeSession::new(self); s.prefill(chunk_prompt)?; read_capture(self) };
        for t in 1..extra {
            let cur = { let mut s = DecodeSession::new(self); s.prefill(chunk_prompt)?; read_capture(self) };
            for l in 0..n_layers {
                for k in 0..4 {
                    if cos(&run_a[l][k], &cur[l][k]) < cos(&run_a[l][k], &run_b[l][k]) {
                        run_b[l][k] = cur[l][k].clone();
                    }
                }
            }
            eprintln!("[firstdiv] trial {}/{} done", t + 1, extra);
        }

        for v in ["DS4_PREFILL_CHUNK", "DS4_CHUNK_HALF_CHECK", "DS4_CHUNK_MAX_CTX"] {
            std::env::remove_var(v);
        }
        // Report the WORST (lowest cos) of {residual, comp_ring, index_ring} per
        // layer, plus a tag of which buffer diverged. Encodes the buffer in maxΔ's
        // companion: returns (layer, min_cos, which) where which: 0=resid 1=comp 2=idx 3=kv.
        Ok((0..n_layers).map(|l| {
            let c = [
                cos(&run_a[l][0], &run_b[l][0]),
                cos(&run_a[l][1], &run_b[l][1]),
                cos(&run_a[l][2], &run_b[l][2]),
                cos(&run_a[l][3], &run_b[l][3]),
            ];
            // pick the most-divergent buffer (lowest cos that is a real <1 divergence)
            let mut which = 0usize; let mut lo = c[0];
            for (i, &ci) in c.iter().enumerate() { if ci < lo { lo = ci; which = i; } }
            (l, lo, which as f32)
        }).collect())
    }

    /// Run a Rust-side prefill over `prompt_tokens`, returning the
    /// argmax-sampled `n_decode` token stream. Each step feeds the
    /// previously-sampled token back through the embedding table.
    ///
    /// `pos = 0` starts a fresh KV cache.
    pub fn run_argmax(
        &self,
        prompt_tokens: &[i32],
        n_decode: u32,
        eos_token: i32,
    ) -> Result<Vec<i32>> {
        let cfg = DecodeConfig::default();
        let mut state = AttnStepState::new(&self.composed, self.raw_cap);
        let progress = std::env::var("DS4_PROGRESS").ok().is_some_and(|v| v != "0");
        let t_start = std::time::Instant::now();

        // Prefill: drive every prompt token through decode_step_with_attn
        // to populate the KV cache layer-by-layer. The final logits from
        // the last prompt token also seed the first-decode argmax.
        let mut last_logits = Vec::new();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let t_tok = std::time::Instant::now();
            let x = self.embed(tok)?;
            // M4 #289: surface input token id to hash-routed MoE selection
            // (`ffn_gate_tid2eid` lookup in layers 0/1/2). Stays as 0 in
            // pipelines that don't carry a routing table.
            ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.set(tok));
            last_logits = decode_step_with_attn(
                &self.dispatcher,
                &self.dispatcher,
                x,
                &self.composed,
                &mut state,
                &cfg,
                self.raw_cap,
            )?;
            if progress {
                eprintln!(
                    "[progress] prefill pos={}/{} tok={} step={:.2}s elapsed={:.2}s",
                    i + 1,
                    prompt_tokens.len(),
                    tok,
                    t_tok.elapsed().as_secs_f32(),
                    t_start.elapsed().as_secs_f32(),
                );
            }
        }
        if last_logits.is_empty() {
            bail!("run_argmax: empty prompt — at least one token required");
        }

        // M4 #266 diagnostic: dump n_comp after prefill so we can tell whether
        // the compressor fired during prompt processing. Always-on; cheap.
        {
            let mut counts: Vec<u32> = state.n_comp.clone();
            let max = counts.iter().copied().max().unwrap_or(0);
            let min = counts.iter().copied().min().unwrap_or(0);
            let nz = counts.iter().filter(|&&x| x > 0).count();
            counts.truncate(8);
            eprintln!(
                "[compressor-diag] post-prefill state.pos={} n_comp[0..8]={:?} min={} max={} nz_layers={}/{}",
                state.pos, counts, min, max, nz, state.n_comp.len()
            );
        }

        // Decode: argmax(last_logits) → emit + step.
        let mut out = Vec::with_capacity(n_decode as usize);
        let dump_n: Option<u32> = std::env::var("DS4_DUMP_LOGITS_FIRST_N")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        for step in 0..n_decode {
            let t_step = std::time::Instant::now();
            if let Some(i) = last_logits.iter().position(|v| v.is_nan()) {
                bail!("decode step {step}: logit[{i}] is NaN — divergence in MetalDispatcher pipeline");
            }
            if let Some(n) = dump_n {
                if step < n {
                    dump_topk_logits(step, &last_logits, 10);
                }
            }
            let tok = argmax_i32(&last_logits);
            if tok == eos_token {
                break;
            }
            out.push(tok);
            let x = self.embed(tok)?;
            // M4 #289: set token-id hint for hash-routed MoE (layers 0/1/2).
            ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.set(tok));
            // M4 #277 — dump embed-row RMS at first 4 decode steps; gated.
            if std::env::var("DS4_DUMP_EMBED_RMS").ok().as_deref() == Some("1") && step < 4 {
                let n = x.len() as f64;
                let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let rms = (ss / n).sqrt();
                let amin = x.iter().cloned().fold(f32::INFINITY, f32::min);
                let amax = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "EMBED_RMS step={} tok={} d={} rms={:.6} min={:.4} max={:.4}",
                    step,
                    tok,
                    x.len(),
                    rms,
                    amin,
                    amax,
                );
            }
            last_logits = decode_step_with_attn(
                &self.dispatcher,
                &self.dispatcher,
                x,
                &self.composed,
                &mut state,
                &cfg,
                self.raw_cap,
            )?;
            if progress {
                eprintln!(
                    "[progress] decode step={}/{} tok={} step={:.2}s elapsed={:.2}s",
                    step + 1,
                    n_decode,
                    tok,
                    t_step.elapsed().as_secs_f32(),
                    t_start.elapsed().as_secs_f32(),
                );
            }
        }
        Ok(out)
    }

    /// Same as [`Self::run_argmax`] but returns `(tokens, prefill_s, decode_s)`
    /// so callers (e.g. M4 perf-gate bench) can compute prefill/decode tok/s
    /// without instrumenting decode_step_with_attn.
    ///
    /// M4 #330p: routes through `SingleBufferEncoder::decode_token_batched`
    /// so the perf gate exercises the C.1 + C.3c tail-batched encoder path
    /// (rms_norm + q8_0? + lm_head matvec packed into one MTLCommandBuffer).
    /// Bit-equivalent to `decode_step_with_attn` per M4 #330m smoke; saves
    /// 2 cwr/token on the perf path.
    pub fn run_argmax_timed(
        &self,
        prompt_tokens: &[i32],
        n_decode: u32,
        eos_token: i32,
    ) -> Result<(Vec<i32>, f64, f64)> {
        let mut state = AttnStepState::new(&self.composed, self.raw_cap);
        let encoder =
            crate::single_buffer_encoder::SingleBufferEncoder::new(&self.dispatcher, self.raw_cap);
        let l_split = encoder.cutpoint_for(&self.composed);

        if prompt_tokens.is_empty() {
            bail!("run_argmax_timed: empty prompt — at least one token required");
        }
        // Greedy decode keeps the argmax on GPU (no full-logit readback).
        // DS4_FULL_LOGITS=1 reverts to the full-logit path + CPU argmax (for
        // sampling / logit diagnostics).
        let full_logits = std::env::var("DS4_FULL_LOGITS").ok().as_deref() == Some("1");
        // DS4_FAST_DECODE (default-on, =0 reverts): drive each token through the
        // chained single-cb `encode_first_half` path (the ~22 tok/s sustained
        // chain + GPU-argmax lm-head tail) instead of the slow trait-dispatch
        // reference (`decode_step_with_attn_to_residual`, the old ~2 tok/s
        // cutpoint path). This is the full end-to-end decode counterpart of the
        // encode_first_half_sustained_throughput bench.
        let fast = std::env::var("DS4_FAST_DECODE").ok().as_deref() != Some("0");
        let next_token = |runner: &Self,
                              encoder: &crate::single_buffer_encoder::SingleBufferEncoder,
                              state: &mut AttnStepState,
                              tok: i32|
         -> Result<i32> {
            let x = runner.embed(tok)?;
            ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.set(tok));
            match (fast, full_logits) {
                (true, false) => {
                    encoder.decode_token_via_first_half_argmax(&x, &runner.composed, state)
                }
                (true, true) => Ok(argmax_i32(
                    &encoder.decode_token_via_first_half(&x, &runner.composed, state)?,
                )),
                (false, true) => Ok(argmax_i32(&encoder.decode_token_batched(
                    x,
                    &runner.composed,
                    state,
                    l_split,
                )?)),
                (false, false) => {
                    encoder.decode_token_argmax(x, &runner.composed, state, l_split)
                }
            }
        };

        let prefill_start = std::time::Instant::now();
        let mut next_tok = 0i32;
        for &tok in prompt_tokens.iter() {
            next_tok = next_token(self, &encoder, &mut state, tok)?;
        }
        let prefill_s = prefill_start.elapsed().as_secs_f64();

        let decode_start = std::time::Instant::now();
        let mut out = Vec::with_capacity(n_decode as usize);
        for _ in 0..n_decode {
            let tok = next_tok;
            if tok == eos_token {
                break;
            }
            out.push(tok);
            next_tok = next_token(self, &encoder, &mut state, tok)?;
        }
        let decode_s = decode_start.elapsed().as_secs_f64();
        ds4_engine::op_timer::report_and_reset();
        Ok((out, prefill_s, decode_s))
    }
}

/// Streaming inference session over a [`DecodeRunner`], using the fast
/// single-cb decode chain (the `DS4_FAST_DECODE` path, full logits). Prefill
/// the prompt once, then drive one token at a time — the caller samples from
/// [`Self::logits`] and feeds the chosen token back via [`Self::step`]. The
/// persistent KV / compressor state lives in `AttnStepState`.
///
/// This mirrors `run_argmax_timed`'s `(fast,full_logits)=(true,true)` path
/// (`SingleBufferEncoder::decode_token_via_first_half`) split into prefill/step
/// so an HTTP server can stream + sample + detokenize between steps.
///
/// NOT `Send`/`Sync` (borrows the Metal dispatcher) — must live on the single
/// Metal worker thread.
pub struct DecodeSession<'r> {
    runner: &'r DecodeRunner,
    encoder: crate::single_buffer_encoder::SingleBufferEncoder<'r>,
    state: AttnStepState,
    last_logits: Vec<f32>,
}

impl<'r> DecodeSession<'r> {
    /// Start a fresh session (empty KV cache, pos=0).
    pub fn new(runner: &'r DecodeRunner) -> Self {
        let encoder = crate::single_buffer_encoder::SingleBufferEncoder::new(
            &runner.dispatcher,
            runner.raw_cap,
        );
        let state = AttnStepState::new(&runner.composed, runner.raw_cap);
        Self { runner, encoder, state, last_logits: Vec::new() }
    }

    /// Feed one token through the fast chain, caching full vocab logits.
    fn feed(&mut self, tok: i32) -> Result<()> {
        let x = self.runner.embed(tok)?;
        // Hash-routed MoE token hint (layers 0/1/2), as in run_argmax.
        ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.set(tok));
        self.last_logits = self.encoder.decode_token_via_first_half(
            &x,
            &self.runner.composed,
            &mut self.state,
        )?;
        // Flush any weight buffers pinned to the residency set during this
        // token's first-touch (batched commit+requestResidency). No-op once the
        // weight caches are warm. Keeps the ~31 GB of projection/q8 weight
        // buffers GPU-resident so they aren't idle-compressed between requests.
        self.runner.dispatcher.commit_residency();
        Ok(())
    }

    /// Prefill the prompt (≥1 token). After this, [`Self::logits`] holds the
    /// distribution for the first generated token.
    pub fn prefill(&mut self, prompt_tokens: &[i32]) -> Result<()> {
        let Some((&last, init)) = prompt_tokens.split_last() else {
            bail!("DecodeSession::prefill: empty prompt — at least one token required");
        };
        // DS4_PREFILL_CHUNK=K (opt-in): batch the `init` prompt tokens through
        // the chunked-prefill driver (K consecutive positions per chunk; the
        // matmuls K-batched, the attention sequential) instead of the per-token
        // first half. Leaves the last token for the normal `feed` (full logits).
        //
        // ★ PREFILL ROUTING (2026-06-10 clean-boot bisect — supersedes the old
        // "chunk is never long-context-faithful, clamp to raw_cap" gate). The
        // single-chunk fast stack reaches ~115 tok/s @3000 (4.5× per-token) and is
        // COHERENT (gamma needle recall validated 1024-3300) once two things hold,
        // both now DEFAULTS: DS4_CHUNK_SWA_KFLASH OFF (its tile-boundary NaN was the
        // entire old incoherence — Phase-B was innocent; commit a2d8c7a6) and the
        // fused HC-collapse UNCAPPED (the K≤128 cap was the 2× perf regression;
        // commit 49dd3105). The old "stochastic word-salad at every length" was the
        // test harnesses force-enabling SWA_KFLASH; production never set it.
        //
        // The fast path is SINGLE-chunk (chunk_start==0; the nosync batched cores
        // require it). Within one chunk the only ctx limit is the compressor ring:
        // comp_ring_rows = DS4_MAX_CTX_ROWS/4 + 8 = 8200 rows ⇒ n_comp ≤ 8200 ⇒
        // ctx ≤ ~32768 (n_comp = ctx / min_ratio = ctx/4). The old n_comp ≤ 1024
        // argsort cap is GONE (commit: encode_indexer_topk_batched delegates to the
        // threshold-select batched kernel above 1024). So AUTO-route prompts with
        // raw_cap < init.len() ≤ SINGLE_CHUNK_CAP through the fast chunk path;
        // per-token otherwise (tiny prompts ≤ raw_cap: trivial + all-raw ==
        // per-token anyway; > cap: would need MULTI-chunk, chunk_start>0, which the
        // nosync cores don't support yet → safe per-token). Overrides:
        // DS4_PREFILL_PERTOKEN=1 forces per-token; DS4_PREFILL_CHUNK=K sets an
        // explicit chunk size and routes through chunk at ANY length (benchmarking).
        // FIDELITY NOTE: PHASE_A_BATCH (the 74→115 lever, folds the hash layers
        // L0-2 into the K-batch) is defaulted ON — needle recall passes with it —
        // but a synthetic-summarization fidelity edge was historically flagged; set
        // DS4_CHUNK_PHASE_A_BATCH=0 to run those layers per-token (~74 tok/s, most
        // faithful) if a content regression appears.
        // Cap = the validated-FAITHFUL boundary, NOT the structural max. The topk
        // path no longer crashes at any n_comp (threshold-select), and the chunk
        // stack is coherent well past 8192, BUT needle recall degrades vs per-token
        // at the longest contexts: measured gamma recall chunk-vs-per-token = both
        // YES at 6144 (n_comp 1536), but at 8192 (n_comp 2048) per-token recalls and
        // chunk does NOT (accumulated K-batch drift). So cap at 6144 (faithful);
        // 6144 < init ≤ ring-max(~32768) stays per-token (slower but faithful) until
        // a long-ctx fidelity fix lands, after which raise toward the ring ceiling.
        // 8192-drift bisect (2026-06-11) CONFIRMED 6144 is the right boundary: at
        // 8192 the chunk path is DETERMINISTICALLY non-recalling (4/4 identical
        // gamma=no), and the per-layer chunk-vs-per-token residual cos declines
        // SMOOTHLY from L1 (0.967) to a worst 0.28 @L21 — distributed K-batch fp32
        // non-associativity, NOT a single localizable op. So this is a real fidelity
        // ceiling, not a bug to fix; keep the cap at the measured-faithful 6144.
        const SINGLE_CHUNK_CAP: usize = 6144;
        // STRUCTURAL ceiling: one chunk emits ~K/MIN_RATIO(=4) compressor rows into a
        // comp_ring fixed at DS4_MAX_CTX_ROWS/4 + 8 = 8200 rows. A single chunk above
        // ~32768 tokens overflows the ring (unguarded OOB store). The cap is a
        // FIDELITY boundary that must also stay ≤ this STRUCTURAL max; assert it so a
        // future bump can't silently exceed the ring.
        const SINGLE_CHUNK_STRUCTURAL_MAX: usize = 32768;
        const _: () = assert!(
            SINGLE_CHUNK_CAP <= SINGLE_CHUNK_STRUCTURAL_MAX,
            "SINGLE_CHUNK_CAP exceeds the comp_ring-supported ctx (~32768) — a single \
             auto-chunk would overflow the 8200-row compressor ring"
        );
        let raw_cap = self.runner.raw_cap as usize;
        let force_pertoken = std::env::var("DS4_PREFILL_PERTOKEN").ok().as_deref() == Some("1");
        let explicit_chunk = std::env::var("DS4_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&k| k >= 1);
        // Auto-enable the fast chunk path for single-chunk prompts above the raw
        // window. An explicit DS4_PREFILL_CHUNK always routes through chunk (honors
        // the caller's size, any length).
        let auto_chunk = !force_pertoken
            && explicit_chunk.is_none()
            && init.len() > raw_cap
            && init.len() <= SINGLE_CHUNK_CAP;
        let use_chunk = !init.is_empty()
            && !force_pertoken
            && (explicit_chunk.is_some() || auto_chunk);
        if use_chunk {
            // Default the coherent fast-stack flags (each individually overridable;
            // SWA_KFLASH is intentionally left unset = OFF). These select the batched
            // GPU cores; without them prefill_chunk runs the slow per-position path.
            // DS4_CHUNK_DEFER_RESYNC (Phase 1, docs/PREFILL_SINGLE_CB_PLAN.md): defers
            // the per-layer compressor CPU-mirror resyncs past the terminal wait,
            // removing the in-layer drains → +5.4% prefill @3000 (110.6→116.6 tok/s,
            // same-GPU A/B) and BYTE-IDENTICAL decode (chunk_defer_resync_identity).
            // DS4_CHUNK_BATCH_ROUTER: collapse the MoE router-logits + shared-swiglu
            // per-token loops into batched dispatches (broadcast mul_mv / grid.x=K) →
            // −~1200 dispatches/layer, +4.5% prefill @3000 (116.3→121.5, same-GPU A/B),
            // BYTE-IDENTICAL decode (chunk_flag_identity @600 & @3000).
            // DS4_CHUNK_HASH_GPU + DS4_CHUNK_POOL_SCRATCH (docs/PREFILL_BOUNDED_WORKING_SET.md,
            // T1-T4): GPU-resident hash-router weights (removes the per-layer probs_k CPU
            // drain on layers 0/1/2 — antirez ds4_metal.m:20xx computes these GPU-resident)
            // + the bounded scratch pool that PINS the merged-cb working set resident
            // (MTLResidencySet, queue-bound) so it can't be evicted under daemon memory
            // pressure. The two are a UNIT: HASH_GPU enlarges the cb (router+MoE merge) and
            // is pressure-safe ONLY with the pinned pool. +3.1% prefill @3000 (250.0→257.7,
            // clean A/B best-of-6), BYTE-IDENTICAL UNDER 73GB PRESSURE (chunk_flag_identity,
            // ce-default, pool correctness-clean → genuine pass, not the pool+PIPELINE false
            // pass). PIPELINE is DELIBERATELY NOT defaulted here: pool reuse relies on
            // sequential layers; commit_pipelined races the reused buffer (WAR corruption).
            for flag in [
                "DS4_CHUNK_BATCHED_IDX", "DS4_CHUNK_ATTN_NOSYNC",
                "DS4_CHUNK_FUSED_COMP", "DS4_ATTN_OUT_GEMM", "DS4_CHUNK_PHASE_A_BATCH",
                "DS4_CHUNK_DEFER_RESYNC", "DS4_CHUNK_BATCH_ROUTER",
                "DS4_CHUNK_HASH_GPU", "DS4_CHUNK_POOL_SCRATCH",
            ] {
                if std::env::var(flag).is_err() {
                    std::env::set_var(flag, "1");
                }
            }
            // K-adaptive bubble recovery: pack 2 layers per cb (commit_every=2) with
            // the redundant crossing event-splits OFF (EVENT_ORDER=0, which alone is
            // byte-identical — the per-layer drain already orders the hazard). This
            // halves the per-layer drain bubbles → ~+2.5% prefill @3000, BYTE-IDENTICAL
            // (chunk_commit_every_identity @3000/256). The per-cb transient footprint
            // scales with layers×K, and commit_every=2 is a DETERMINISTIC per-cb cap:
            // byte-identical up to K≈3000 but ALL-ZEROS at K≥4096 (fails with 128GB
            // free → not memory-pressure, a fixed per-cb resource limit). So enable it
            // only at K ≤ the validated-safe bound; longer prompts keep commit_every=1.
            // Requires DEFER_RESYNC=1: with commit_every>1 the per-layer CPU-mirror
            // resync runs mid-cb and reads an incomplete pool unless deferred to the
            // terminal (ce=2+DEFER=0 diverges; ce=2+DEFER=1 == ce=1, byte-identical).
            const SAFE_CE2_CAP: usize = 3000;
            let defer_on = std::env::var("DS4_CHUNK_DEFER_RESYNC").as_deref() == Ok("1");
            if init.len() <= SAFE_CE2_CAP && defer_on {
                if std::env::var("DS4_CHUNK_COMMIT_EVERY").is_err() {
                    std::env::set_var("DS4_CHUNK_COMMIT_EVERY", "2");
                }
                if std::env::var("DS4_CHUNK_EVENT_ORDER").is_err() {
                    std::env::set_var("DS4_CHUNK_EVENT_ORDER", "0");
                }
                // Bounded event-pipeline (DS4_CHUNK_COMMIT_EVENT=1 + DS4_CHUNK_EVENT_WINDOW=1):
                // GPU-ordered MTLEvent cb splits with ≤1 cb pending (≤2 in flight). The next
                // cb is pre-queued event-ordered behind the current → the GPU runs them
                // back-to-back with NO inter-cb drain bubble (busy/span 88%→100%), recovering
                // the ~1.3s/11% inter-cb GPU-flush bubble → +13.6% prefill @3000 (246.9→280.4
                // tok/s, clean A/B best-of-3) and BYTE-IDENTICAL (DS4_PREFILL_EVENT_AB:
                // maxdiff=0.0, cos=1.0 vs the sync-drain baseline @600 & @3000).
                //
                // Earlier this was DEFAULT-OFF because the unbounded/undrained cbs each held
                // FRESH ~hundreds-of-MB scratch (moe_mid/shared_mid/hc_after_attn) that wasn't
                // resident → ZEROS-on-read under daemon pressure → BOS collapse. That premise
                // is now FIXED: DS4_CHUNK_POOL_SCRATCH (default-ON above) makes the in-flight
                // scratch O(1) AND pinned-resident (the "OR O(1) pooled scratch" re-enable
                // condition), and window=1 bounds in-flight to ≤2 cbs. Validated coherent
                // UNDER 77GB ds4-server pressure (DS4_PREFILL_EVENT_AB: maxdiff=0.0 vs baseline,
                // tok/s held at 280 — no masked collapse). window≥3 has a separate length-
                // dependent hazard → keep window=1 (full bubble win at the tightest, safest depth).
                if std::env::var("DS4_CHUNK_COMMIT_EVENT").is_err() {
                    std::env::set_var("DS4_CHUNK_COMMIT_EVENT", "1");
                }
                if std::env::var("DS4_CHUNK_EVENT_WINDOW").is_err() {
                    std::env::set_var("DS4_CHUNK_EVENT_WINDOW", "1");
                }
            }
            let chunk_size = explicit_chunk.unwrap_or_else(|| init.len().max(1));
            if explicit_chunk.is_some() && init.len() > SINGLE_CHUNK_CAP {
                eprintln!(
                    "ds4: chunk-prefill init len {} > single-chunk cap {SINGLE_CHUNK_CAP} \
                     with explicit DS4_PREFILL_CHUNK={chunk_size} — multi-chunk \
                     (chunk_start>0) is unvalidated and may be incoherent.",
                    init.len(),
                );
            }
            let n_embd = self.runner.composed.layers[0].attn.params.d_embd as usize;
            let mut start = 0usize;
            while start < init.len() {
                let end = (start + chunk_size).min(init.len());
                let kk = end - start;
                let mut embeds = Vec::with_capacity(kk * n_embd);
                for &tok in &init[start..end] {
                    embeds.extend_from_slice(&self.runner.embed(tok)?);
                }
                let token_ids: Vec<i32> = init[start..end].to_vec();
                self.encoder.prefill_chunk(
                    &embeds, &token_ids, &self.runner.composed,
                    &mut self.state, start as u32, kk,
                )?;
                self.runner.dispatcher.commit_residency();
                start = end;
            }
            dump_prefill_state("CHUNK", &self.state);
            self.feed(last)?;
            return Ok(());
        }
        // All but the last prompt token: run the first half (KV/compressor/
        // cur_hc update) WITHOUT the lm-head — only the last token's logits are
        // needed. Cuts the per-token 129k-vocab matmul + full-logit readback
        // across the prompt (the dominant @3000 TTFT overhead). DS4_PREFILL_NO_SKIP=1
        // reverts to the full per-token feed (for the A/B; identical decode state).
        let skip = std::env::var("DS4_PREFILL_NO_SKIP").ok().as_deref() != Some("1");
        for &tok in init {
            if skip {
                let x = self.runner.embed(tok)?;
                ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.set(tok));
                self.encoder
                    .prefill_step(&x, &self.runner.composed, &mut self.state)?;
                self.runner.dispatcher.commit_residency();
            } else {
                self.feed(tok)?;
            }
        }
        dump_prefill_state("PERTOK", &self.state);
        // DS4_PT_HEADS_DUMP read-out: write the per-token reference heads (rows
        // 0..init.len(), captured by the resident layer probe) to DS4_PT_HEADS_FILE.
        if let (Ok(l), Ok(f)) = (
            std::env::var("DS4_PT_HEADS_DUMP"),
            std::env::var("DS4_PT_HEADS_FILE"),
        ) {
            if let Ok(l) = l.parse::<usize>() {
                let p = &self.runner.composed.layers[l].attn.params;
                let q_dim = p.n_head as usize * p.head_dim as usize;
                let rows = init.len().min(4096);
                let dbg = self.runner.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 6_000_000, 4096 * q_dim * std::mem::size_of::<f32>());
                let bytes = unsafe {
                    std::slice::from_raw_parts(dbg.contents() as *const u8, rows * q_dim * 4)
                };
                std::fs::write(&f, bytes).expect("write DS4_PT_HEADS_FILE");
                eprintln!("[PT-HEADS] wrote {f} ({rows} pos x {q_dim} f32)");
            }
        }
        // Last token: full logits for the first generated token.
        self.feed(last)?;
        Ok(())
    }

    /// Advance by one already-chosen token (the sampled/emitted token); updates
    /// [`Self::logits`] for the next choice.
    pub fn step(&mut self, token: i32) -> Result<()> {
        self.feed(token)
    }

    /// Full vocab logits from the most recent prefill/step.
    pub fn logits(&self) -> &[f32] {
        &self.last_logits
    }

    /// Current decode position (tokens consumed).
    pub fn pos(&self) -> u32 {
        self.state.pos
    }

    /// Read-only access to the post-prefill attention state — exposed so
    /// integration tests can compare the compressor ring CONTENTS + `n_comp`
    /// (not just the logits) across code paths. See the STAGE 1 fused
    /// chunk-prefill integration test (`comp_prefill_integration.rs`).
    pub fn state(&self) -> &ds4_engine::decode_step::AttnStepState {
        &self.state
    }
}

/// DS4_DUMP_PREFILL_STATE: per-layer post-prefill compressor/KV state — to diff
/// the chunked vs per-token prefill (find the first diverging layer/field).
fn dump_prefill_state(label: &str, state: &ds4_engine::decode_step::AttnStepState) {
    if std::env::var("DS4_DUMP_PREFILL_STATE").is_err() {
        return;
    }
    let n = state.kv_pos.len();
    let hd = 512usize;
    for l in 0..n {
        let nc = state.n_comp[l] as usize;
        if nc == 0 { continue; }
        let r = &state.comp_kv_ring[l];
        // Head of emit ROW 0 (first/single emit — exact at chunk=4) vs the LAST
        // emit row (the multi-emit rotation product). If row0 matches but the
        // last row drifts → the inter-emit rotation is the bug.
        let row0: Vec<f32> = r.get(0..3).map(|s| s.to_vec()).unwrap_or_default();
        let lastoff = (nc - 1) * hd;
        let rowL: Vec<f32> = r.get(lastoff..lastoff + 3).map(|s| s.to_vec()).unwrap_or_default();
        eprintln!(
            "[STATE {label}] L{l} n_comp={nc} row0={row0:.6?} row{}={rowL:.6?}",
            nc - 1,
        );
    }
}

fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as i32
}

/// Diagnostic helper for bisecting --correctness divergence. Prints to stdout:
///   - the top-k logits as (idx, value)
///   - statistics: min, max, mean, stddev across the full vocab
///   - the value at the antirez-known argmax index (201 for the m4_short prompt)
/// Enabled by `DS4_DUMP_LOGITS_FIRST_N=N` env var.
fn dump_topk_logits(step: u32, logits: &[f32], k: usize) {
    let antirez_known = 201usize;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let topk: Vec<(usize, f32)> = idx.iter().take(k).map(|&i| (i, logits[i])).collect();
    let mut lmin = f32::INFINITY;
    let mut lmax = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for &v in logits {
        if v < lmin {
            lmin = v;
        }
        if v > lmax {
            lmax = v;
        }
        sum += v as f64;
        sum2 += (v as f64) * (v as f64);
    }
    let n = logits.len() as f64;
    let mean = sum / n;
    let var = (sum2 / n) - mean * mean;
    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
    println!("DUMP_LOGITS step={step} top{k}:");
    for (i, v) in &topk {
        println!("  ({i:>6}, {v:>10.4})");
    }
    let rank_201 = idx
        .iter()
        .position(|&i| i == antirez_known)
        .unwrap_or(usize::MAX);
    let val_201 = if antirez_known < logits.len() {
        logits[antirez_known]
    } else {
        f32::NAN
    };
    println!(
        "  antirez_known_idx={antirez_known} value={val_201:.4} rank={rank_201}  vocab_min={lmin:.4} max={lmax:.4} mean={mean:.4} std={std:.4}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        let v = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax_i32(&v), 3);
    }

    #[test]
    fn argmax_handles_neg_inf_clean() {
        let v = vec![f32::NEG_INFINITY, -1.0, -0.5];
        assert_eq!(argmax_i32(&v), 2);
    }
}
