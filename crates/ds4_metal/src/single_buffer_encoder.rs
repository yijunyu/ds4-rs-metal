//! M4 #330l — Phase B.3 split-buffer hybrid encoder skeleton.
//!
//! Goal (Phase C): per token, encode all 43 layers + lm_head into TWO
//! MTLCommandBuffers separated by one batched CPU branching step. Final
//! readback is just the argmax token id. See
//! `project_m4_330h_option_3_full_design.md` for the full plan.
//!
//! Phase B (this module): lock in the API surface. The encoder owns no
//! GPU encoding yet — `decode_token_batched` currently delegates to
//! `decode_step_with_attn`, the same path `--correctness` uses. The
//! value of this scaffold is:
//!
//!  - Defines `SingleBufferEncoder` as a parallel API to `MetalDispatcher`,
//!    intentionally NOT a `KernelDispatcher` impl so the trait path keeps
//!    its 18 fidelity gates intact.
//!  - Defines `BatchedBranchingPlan` — the encoder-internal record of
//!    which layer is `L_split` and what state slots cross the buffer
//!    boundary. Phase C consumes this.
//!  - Defines the `--bench` entry point `run_argmax_batched`, called
//!    only from perf-gate mode. Correctness mode keeps using
//!    `run_argmax`/`run_argmax_timed` on the trait dispatcher.
//!
//! Once Phase C populates real GPU encoding into buffers A and B,
//! the API here doesn't change — only the body of `encode_layers_into`
//! switches from CPU delegation to actual `metal::ComputeCommandEncoder`
//! calls.

#![cfg(target_os = "macos")]

use anyhow::{bail, Result};

// `BatchedBranchingPlan` is re-exported below at the `pub use` declaration;
// removing the redundant private `use` to avoid an E0252 duplicate-import.
use ds4_engine::attn_dispatch::{
    apply_rope_tail_with_table, decode_attn_ffn_post_with, decode_attn_ffn_pre_with,
    precompute_rope_tail_table, AttentionDispatcher, AttnPrefixOut, FfnPreOut, LayerParams,
};
use ds4_engine::decode_step::{
    decode_step_with_attn_to_residual, output_hc_head_one, AttnStepState, ComposedLayerWeights,
    ComposedModelWeights, DecodeConfig,
};
use ds4_engine::dispatch::KernelDispatcher;

// ── DS4_CHUNK_KPROF: per-stage wall-time profiler for the chunk-prefill path. ──
// Gated; when on, chunk_layer commit_wait's after each stage and accumulates the
// stage's wall time (encode + GPU) into a thread-local. prefill_chunk also records
// Phase A. Dumped (cumulative) at the end of each prefill_chunk call — read the LAST
// line for grand totals. Serializes the path (profiling only).
fn chunk_kprof_on() -> bool {
    std::env::var("DS4_CHUNK_KPROF").is_ok()
}
/// MTLEvent-ordered cb splits at the chunk>raw_cap hazard boundaries (default
/// ON; DS4_CHUNK_EVENT_ORDER=0 disables). Same boundaries the f6e2a768 CPU
/// drains target in the SYNC path; here the split orders GPU work WITHOUT a
/// CPU wait (a mid-scope `commit_wait_stage` corrupts the NOSYNC branches'
/// CPU-mirror resync logic — never use it there). Every call site is also
/// gated on chunk_start+K > raw_cap, so chunks fitting the ring are unchanged.
fn chunk_event_order_on() -> bool {
    std::env::var("DS4_CHUNK_EVENT_ORDER").ok().as_deref() != Some("0")
}
/// PHASE 1 (docs/PREFILL_SINGLE_CB_PLAN.md): defer the per-layer nosync compressor
/// CPU-mirror resync (pool/ring reads) past the terminal wait, removing the per-layer
/// GPU drain that forces one-cb-per-layer. Default OFF until validated bit-identical.
fn defer_chunk_resync() -> bool {
    std::env::var("DS4_CHUNK_DEFER_RESYNC").ok().as_deref() == Some("1")
}
/// DS4_PHASEB_SPLIT bisect mask — MTLEvent splits BETWEEN the Phase-B batched
/// dispatches (chunk crossing raw_cap only). bit0=before scores, bit1=before
/// top-k, bit2=before mixed-attention, bit3=after mixed-attention. Diagnostic
/// for localizing the in-cb Phase-B race; default 0 (off).
fn phaseb_split_mask() -> u32 {
    std::env::var("DS4_PHASEB_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
/// Warn ONCE that DS4_CHUNK_SWA_KFLASH is enabled at chunk>raw_cap, where its
/// tile-boundary ring-store/window-gather hazard injects a tail NaN → BOS. The
/// nosync batched path is the coherent default; this lever is for fault repro only.
fn warn_swa_kflash_incoherent() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[ds4] WARNING: DS4_CHUNK_SWA_KFLASH=1 with chunk>raw_cap is KNOWN-INCOHERENT \
             (tile-boundary NaN → BOS). The nosync batched chunk path is the coherent \
             default; unset DS4_CHUNK_SWA_KFLASH for production prefill."
        );
    });
}
thread_local! {
    static CHUNK_KPROF: std::cell::RefCell<std::collections::BTreeMap<&'static str, u128>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    // Per-stage compute-dispatch count (Task 0 follow-up: localize the ~5x-antirez
    // per-layer dispatch density). Tracks the delta of macos::dispatch_count() between
    // chunk_kprof_add calls. Dumped alongside the time breakdown.
    static CHUNK_KPROF_DISP: std::cell::RefCell<std::collections::BTreeMap<&'static str, u64>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    static CHUNK_KPROF_LAST_DISP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Per-layer-input RESIDUAL capture for the cross-path divergence probe
    /// (`DecodeRunner::residual_divergence_probe`). When `Some`, the per-token
    /// `encode_first_half_inner` snapshots `state.cur_hc` (the layer input) into
    /// `[layer]` at the start of each layer, every token (overwriting) — so after
    /// prefilling N tokens it holds the (N-1)-th token's per-layer input residual.
    /// The chunk side is captured separately via DS4_CHUNK_HALF_CHECK buffers.
    static RESID_CAP: std::cell::RefCell<Option<Vec<Vec<f32>>>> = const { std::cell::RefCell::new(None) };
}
/// Enable/disable per-token per-layer residual capture (probe-only).
pub(crate) fn resid_cap_set(on: bool) {
    RESID_CAP.with(|c| *c.borrow_mut() = if on { Some(Vec::new()) } else { None });
}
/// Take the captured per-layer residuals (clears the capture).
pub(crate) fn resid_cap_take() -> Option<Vec<Vec<f32>>> {
    RESID_CAP.with(|c| c.borrow_mut().take())
}
fn chunk_kprof_add(stage: &'static str, us: u128) {
    CHUNK_KPROF.with(|m| *m.borrow_mut().entry(stage).or_insert(0) += us);
    // Snapshot the dispatch-count delta for this stage.
    let now = crate::macos::dispatch_count();
    let last = CHUNK_KPROF_LAST_DISP.with(|c| c.replace(now));
    let delta = now.saturating_sub(last);
    CHUNK_KPROF_DISP.with(|m| *m.borrow_mut().entry(stage).or_insert(0) += delta);
}
fn chunk_kprof_dump() {
    CHUNK_KPROF.with(|m| {
        let m = m.borrow();
        let disp = CHUNK_KPROF_DISP.with(|d| d.borrow().clone());
        let total: u128 = m.values().sum();
        let disp_total: u64 = disp.values().sum();
        eprintln!("[CHUNK_KPROF] cumulative total={total}us  dispatches={disp_total}");
        for (k, v) in m.iter() {
            eprintln!(
                "[CHUNK_KPROF]   {k:>16}: {v:>10}us  {:5.1}%   disp={:>6}",
                *v as f64 / total.max(1) as f64 * 100.0,
                disp.get(k).copied().unwrap_or(0),
            );
        }
    });
}

/// Phase E M5.4.5.4 — generic per-layer FFN-half helper.
///
/// Drives the FFN-half of one decode layer (steps 6-10 of
/// `decode_attn_layer_with`):
///   1. `decode_attn_ffn_pre_with` → ffn_normed + hc_split_ffn
///   2. `router_logits_batched` + `router_finalize`
///   3. `moe_routed_step` (under `DS4_SILU_FIDELITY=1`) OR
///      `moe_and_shared_chain_batched` (default)
///   4. `decode_attn_ffn_post_with` → after_ffn_hc
///
/// Generic over `K + A` so it can be unit-tested against
/// `CpuDispatcher`/`CpuAttentionDispatcher` with synthetic small-shape
/// weights (Metal MoE has hardcoded production-shape constraints —
/// `n_experts=256`, `d_embd % 256 == 0`, loaded
/// `QuantizedExpertWeights` — that block synthetic Metal unit tests).
///
/// Used by `SingleBufferEncoder::encode_first_half` on Metal with the
/// `MetalDispatcher` impl of both traits.
pub fn run_ffn_half<K, A>(
    k: &K,
    a: &A,
    layer_idx: usize,
    layer: &ComposedLayerWeights,
    prefix: &AttnPrefixOut,
    pos: u32,
) -> Vec<f32>
where
    K: KernelDispatcher,
    A: AttentionDispatcher,
{
    let ffn_pre = decode_attn_ffn_pre_with(a, &layer.attn, prefix);
    let h_norm = &ffn_pre.ffn_normed;
    let probs = k.router_logits_batched(&layer.moe.w_router_as_f32(), h_norm, layer.moe.n_experts);

    // Hash-routing branch (DS4 V4 Flash layers 0/1/2) — mirrors
    // decode_step.rs:998-1029. `routing_table[token_id*k..]` supplies
    // `selected`; `weights` derived from probs via the antirez
    // hash-router weight formula.
    let (selected, weights) = if let Some(table) = layer.moe.routing_table.as_ref() {
        let k_used = layer.moe.n_experts_used;
        let token_id = ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.get()) as usize;
        let base = token_id.saturating_mul(k_used);
        let end = base.saturating_add(k_used);
        if end > table.len() {
            panic!(
                "run_ffn_half: routing_table layer={layer_idx} token_id={token_id} \
                 k={k_used} oob (table.len={})",
                table.len()
            );
        }
        let sel: Vec<usize> = table[base..end].iter().map(|&v| v as usize).collect();
        let w = ds4_engine::moe::hash_router_weights_from_probs(&probs, &sel, k_used);
        (sel, w)
    } else {
        k.router_finalize(
            &probs,
            &layer.moe.router_bias,
            layer.moe.n_experts_used,
        )
    };

    let silu_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let (routed_out, precomputed_shared_out): (Vec<f32>, Option<Vec<f32>>) = if silu_fidelity {
        let r = k.moe_routed_step(
            layer_idx as u32,
            h_norm,
            &selected,
            &weights,
            &layer.moe.w_gate_exps,
            &layer.moe.w_up_exps,
            &layer.moe.w_down_exps,
            layer.moe.d_ffn.map_or(0, |n| n.get()),
        );
        (r, None)
    } else {
        let ffn_in_owned;
        let ffn_in: &[f32] = if want_q80 {
            ffn_in_owned = ds4_engine::forward::q8_0_round_trip(&ffn_pre.ffn_normed);
            &ffn_in_owned
        } else {
            &ffn_pre.ffn_normed
        };
        let (r, s) = k.moe_and_shared_chain_batched(
            layer_idx as u32,
            h_norm,
            &selected,
            &weights,
            &layer.moe.w_gate_exps,
            &layer.moe.w_up_exps,
            &layer.moe.w_down_exps,
            layer.moe.d_ffn.map_or(0, |n| n.get()),
            ffn_in,
            &layer.attn.w_shared_gate,
            &layer.attn.w_shared_up,
            &layer.attn.w_shared_down,
            layer.attn.shared_dim,
            want_q80,
        );
        (r, Some(s))
    };

    decode_attn_ffn_post_with(
        k,
        a,
        &layer.attn,
        prefix,
        &ffn_pre,
        &routed_out,
        pos,
        precomputed_shared_out.as_deref(),
    )
}

/// M5 task #99 — Metal-specific scope-aware FFN-half. Fuses the GPU
/// router pipeline (router_logits → router_finalize → moe →
/// shared_chain) into a single `BatchScope` so router_logits,
/// router_finalize, and moe + shared chain together commit ONE cb
/// per non-hash layer (vs three in `run_ffn_half`'s trait-dispatch
/// path).
///
/// Falls back to the generic `run_ffn_half` for:
///   - `DS4_SILU_FIDELITY=1` — uses `moe_routed_step` (no shared chain), no
///     scope variant yet.
///   - Hash-routing layers (`moe.routing_table.is_some()`, V4 Flash 0/1/2) —
///     selected/weights are precomputed on CPU before any router GPU call,
///     so there's no router_finalize to chain with.
///
/// Bit-equivalent to `run_ffn_half` on the default path: same kernels
/// (matvec + softplus_sqrt for logits, router_finalize_one +
/// router_weights_one for selection, the moe pair + sum6 + shared
/// chain for the FFN body), same args; only the cb boundaries
/// between the stages are removed.
pub fn run_ffn_half_metal_scoped(
    disp: &crate::MetalDispatcher,
    layer_idx: usize,
    layer: &ComposedLayerWeights,
    prefix: &AttnPrefixOut,
    pos: u32,
) -> Vec<f32> {
    let silu_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
    let has_hash_routing = layer.moe.routing_table.is_some();
    if silu_fidelity || has_hash_routing {
        return run_ffn_half(disp, disp, layer_idx, layer, prefix, pos);
    }

    let sp = std::env::var("DS4_STAGE_PROFILE").ok().as_deref() == Some("1");
    let t0 = std::time::Instant::now();
    let ffn_pre = decode_attn_ffn_pre_with(disp, &layer.attn, prefix);
    let t_pre = t0.elapsed().as_secs_f64() * 1000.0;
    let h_norm = &ffn_pre.ffn_normed;
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");

    let ffn_in_owned;
    let ffn_in: &[f32] = if want_q80 {
        ffn_in_owned = ds4_engine::forward::q8_0_round_trip(h_norm);
        &ffn_in_owned
    } else {
        h_norm
    };

    let (routed_out, shared_out) = {
        let scope = disp.batch_scope();
        // Router pipeline: router_logits → router_finalize → moe, all in
        // one cb. encode_router_logits uses the same matvec +
        // softplus_sqrt kernels as the inherent `router_logits_batched`;
        // bit-equivalent.
        let h_norm_db = scope.upload_f32(h_norm);
        let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
        let probs_db = scope
            .encode_router_logits(&w_router_db, &h_norm_db, layer.moe.n_experts, w_router_r_f16)
            .expect("encode_router_logits");
        let bias_db = scope.weight_f32(&layer.moe.router_bias);
        let (sel_db, w_db) = scope
            .encode_router_finalize(&probs_db, &bias_db)
            .expect("encode_router_finalize");
        let (moe_db, shared_db) = scope
            .encode_moe_and_shared_chain_with_router_bufs(
                layer_idx as u32,
                h_norm,
                &sel_db,
                &w_db,
                layer.moe.d_ffn.map_or(0, |n| n.get()),
                ffn_in,
                &layer.attn.w_shared_gate,
                &layer.attn.w_shared_up,
                &layer.attn.w_shared_down,
                layer.attn.shared_dim,
                want_q80,
                &layer.attn.w_shared_gate_q8,
                &layer.attn.w_shared_up_q8,
                &layer.attn.w_shared_down_q8,
            )
            .expect("encode_moe_and_shared_chain_with_router_bufs");
        let _ = &t_pre;
        let outs = scope.flush_and_read_multi(&[&moe_db, &shared_db]);
        let mut it = outs.into_iter();
        let r = it.next().expect("routed_out");
        let s = it.next().expect("shared_out");
        (r, s)
    };

    let t_scope = t0.elapsed().as_secs_f64() * 1000.0 - t_pre;
    let r = decode_attn_ffn_post_with(
        disp, disp, &layer.attn, prefix, &ffn_pre, &routed_out, pos, Some(&shared_out),
    );
    if sp && layer_idx == 5 {
        eprintln!("[ffn L5] pre={t_pre:.1} moe+shared={t_scope:.1} post={:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0 - t_pre - t_scope);
    }
    r
}

/// Phase E M5.4.5.1 — CPU-side mirror of `deferred::LayerAttnHalfOuts`.
///
/// Returned by `SingleBufferEncoder::encode_layer_attn_half_cpu`, which
/// runs `BatchScope::encode_layer_attn_half` + `kv_fp8_store_persistent`
/// in ONE command buffer and flushes once. The CPU vectors land here
/// for use by the (still CPU-side) flash_attn / attn_output / hc_expand
/// downstream until those move into the same scope.
///
/// `kv_normed_rotated` is the value that was written into the persistent
/// per-layer KV cache via `kv_fp8_store_persistent`; the caller doesn't
/// need it for the subsequent flash_attn read (the persistent buffer is
/// the source-of-truth for FA) but it's surfaced for debugging /
/// bit-identical assertions against the trait-dispatch path.
#[derive(Debug, Clone)]
pub struct CpuLayerAttnHalfOuts {
    pub normed: Vec<f32>,
    pub split: Vec<f32>,
    pub qr_normed: Vec<f32>,
    pub q_heads: Vec<f32>,
    pub kv_normed_rotated: Vec<f32>,
}

/// Phase E M5.4.5.2 — outputs from `SingleBufferEncoder::encode_first_half`.
///
/// Currently surfaces only per-layer attn-half outputs (`Vec` indexed by
/// `0..l_split`). M5.4.5.3 will add the post-attn-half threading state
/// (final cur_hc, per-layer post-attn HC residual, etc.) needed to
/// hand off to `encode_second_half`.
#[derive(Debug, Clone)]
pub struct FirstHalfOutputs {
    pub per_layer: Vec<CpuLayerAttnHalfOuts>,
}

/// Layer-chaining (step 10): a compressor finish deferred to token end. The
/// handle owns the per-layer pool buffers; `emit_db` is the GPU-resident emit
/// row to read back (bridge layers on emit positions); `layer_idx`/`is_indexer`
/// pick which per-layer state the finish mutates. The `CompressorInputs` are
/// rebuilt from the model at finish time (no borrow held across the loop).
struct ChainFinish {
    handle: crate::compressor::CompressorScopeHandle,
    emit_db: Option<crate::deferred::DeferredBuf>,
    layer_idx: usize,
    is_indexer: bool,
}

/// Like [`ChainFinish`] but for the chunked-prefill cores: carries the
/// per-token `pos` so the deferred finish can run the position-dependent
/// rope-tail / ratio==4 rotation at chunk end (the decode chain uses
/// `state.pos`, but a chunk spans many positions in one scope).
struct ChunkCompFinish {
    handle: crate::compressor::CompressorScopeHandle,
    emit_db: Option<crate::deferred::DeferredBuf>,
    layer_idx: usize,
    is_indexer: bool,
    pos: u32,
}

/// STAGE 1 fused-prefill finish record (DS4_CHUNK_COMP_PREFILL). The fused
/// `compressor_prefill_noidx` kernel already wrote the emit rows into the GPU
/// comp ring; the post-chunk finish only needs to append those rows to the CPU
/// `comp_kv_ring` mirror and bump `n_comp` (NO pool resync / rotation — the
/// noidx ratio!=4 pool is transient scratch the fused kernel never touches).
struct ChunkCompPrefillFinish {
    /// The persistent GPU comp ring buffer the kernel wrote into.
    comp_ring: metal::Buffer,
    layer_idx: usize,
    /// First comp row written this chunk (== n_comp at chunk start).
    comp_row0: u32,
    /// Number of emit rows written this chunk.
    n_emit: u32,
    head_dim: usize,
}

/// PHASE 1 (DS4_CHUNK_DEFER_RESYNC, docs/PREFILL_SINGLE_CB_PLAN.md): deferred
/// CPU-mirror resync for the production NOSYNC compressor path. The per-layer
/// nosync resync (N1a noidx, sbe ~2348 / IDX-N1 idx, sbe ~3199) drains the GPU
/// mid-chunk ONLY to read the per-layer pools + rings back to the CPU mirrors for
/// decode-after-prefill. Those buffers are per-layer-persistent and no later GPU
/// dispatch in the chunk depends on the CPU mirror, so the read defers past the
/// terminal `wait_all_and_drop` — removing the per-layer drain (the one-cb-per-layer
/// split). The count bumps (n_comp/n_index_comp) stay IMMEDIATE because Phase-B reads
/// them to size scores/top-k; `comp_row0` captures n_comp BEFORE the bump so the
/// post-terminal append lands at the right ring offset.
struct ChunkResync {
    layer_idx: usize,
    /// main compressor pool kv/score (per-layer persistent).
    main_pk: metal::Buffer,
    main_ps: metal::Buffer,
    /// indexer pool kv/score (idx layers only).
    idx_pk: Option<metal::Buffer>,
    idx_ps: Option<metal::Buffer>,
    /// main comp ring (per-layer persistent) + idx ring (idx only).
    comp_ring: metal::Buffer,
    idx_ring: Option<metal::Buffer>,
    /// first comp row written this chunk (== n_comp BEFORE the immediate bump).
    comp_row0: u32,
    n_emit: u32,
    head_dim: usize,
    idx_hd: usize,
}

/// Resident no-drain handoff from `encode_first_half_resident` (the
/// `DS4_RESIDENT_TAIL` path). When the whole token stayed chained, the final
/// HC residual stays GPU-resident in `hc` instead of being read back to CPU;
/// `cbs` are the token's async-committed (not-yet-waited) layer command
/// buffers, and `finishes` are the deferred compressor/indexer state updates.
/// The caller folds the output-head + lm-head + argmax in-scope on `hc`,
/// flushes ONCE (which waits `cbs` by commit order), then runs `finishes`.
pub struct ResidentChain {
    hc: crate::deferred::DeferredBuf,
    cbs: Vec<metal::CommandBuffer>,
    finishes: Vec<ChainFinish>,
}

/// Where the per-token decode splits into two MTLCommandBuffers.
///
/// Phase C uses this to decide which layers go into buffer A vs B; the
/// boundary is also where the single batched CPU readback fires.
/// Defaulting to `n_layers / 2` keeps the two GPU halves roughly balanced.
#[derive(Debug, Clone, Copy)]
pub struct LayerCutpoint(pub usize);

impl LayerCutpoint {
    pub fn middle(n_layers: usize) -> Self {
        Self(n_layers / 2)
    }
}

// `BatchedBranchingPlan` lives in `ds4_engine::batched_branching` (M4 #365
// Phase D); this re-export lets callers reach it through the encoder module
// without a second import path.
pub use ds4_engine::batched_branching::BatchedBranchingPlan;

/// macOS host time in seconds — same base as MTLCommandBuffer `GPUStartTime`
/// (CACurrentMediaTime / mach_absolute_time). For the DS4_STARTUP_PROBE
/// commit→GPU-start latency measurement.
#[allow(deprecated)] // libc mach_* still correct; mach2 migration not worth it for a gated probe
fn mach_now_secs() -> f64 {
    use std::sync::OnceLock;
    static TB: OnceLock<(f64, f64)> = OnceLock::new();
    let (numer, denom) = *TB.get_or_init(|| {
        let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
        unsafe {
            libc::mach_timebase_info(&mut info);
        }
        (info.numer as f64, info.denom as f64)
    });
    let t = unsafe { libc::mach_absolute_time() } as f64;
    t * numer / denom / 1.0e9
}

/// (GPUStartTime, GPUEndTime) of a completed command buffer, in the same
/// mach host-time base as `mach_now_secs` (seconds).
fn cb_gpu_start_end_secs(cb: &metal::CommandBufferRef) -> (f64, f64) {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    unsafe {
        let p: *mut objc::runtime::Object = std::mem::transmute(cb.as_ptr());
        let s: f64 = msg_send![p, GPUStartTime];
        let e: f64 = msg_send![p, GPUEndTime];
        (s, e)
    }
}

/// Phase B.3 split-buffer encoder.
///
/// Owns a borrow of the underlying `MetalDispatcher` because it
/// shares the same `device` / `command_queue` / kernel pipelines.
/// Row count for the compressed/indexer KV rings. These hold up to
/// `ctx / compress_ratio` rows (n_comp grows as pos/ratio) — INDEPENDENT of
/// `raw_cap`, which only bounds the RAW (SWA) window (antirez sizes
/// `comp_cap = ctx/ratio + 2` separately, ds4.c:6349). The full-raw default
/// (raw_cap ≥ ctx/4) happened to cover this; a small SWA `raw_cap` (e.g. 128)
/// does NOT, so size the rings for the daemon's max ctx at the smallest ratio
/// (4) plus margin. (ctx > 32768 would need this raised / threaded through.)
const DS4_MAX_CTX_ROWS: usize = 32768;
const DS4_MIN_COMPRESS_RATIO: usize = 4;
#[inline]
fn comp_ring_rows(raw_cap: u32) -> usize {
    (raw_cap as usize).max(DS4_MAX_CTX_ROWS / DS4_MIN_COMPRESS_RATIO + 8)
}

/// Phase C will give it its own command-buffer pool but for now it
/// just delegates dispatch through the trait surface.
pub struct SingleBufferEncoder<'a> {
    dispatcher: &'a crate::MetalDispatcher,
    cfg: DecodeConfig,
    raw_cap: u32,
    /// STAGE 1 fused chunk-prefill finishes (DS4_CHUNK_COMP_PREFILL). Pushed by
    /// `chunk_attn_core_comp_noidx` during the scope build, drained by
    /// `prefill_chunk` AFTER the terminal `wait_all_and_drop` (so the GPU comp
    /// ring rows the kernel wrote are valid to read back for the CPU mirror).
    comp_prefill_finishes: std::cell::RefCell<Vec<ChunkCompPrefillFinish>>,
    /// PHASE 1 (DS4_CHUNK_DEFER_RESYNC): per-layer nosync compressor CPU-mirror
    /// resyncs deferred past the terminal wait (removes the per-layer drain).
    chunk_resyncs: std::cell::RefCell<Vec<ChunkResync>>,
    /// DS4_CHUNK_HEADS_DUMP diagnostic: (layer, k_positions, q_dim) captured.
    heads_dump: std::cell::Cell<Option<(usize, usize, usize)>>,
    /// DS4_CHUNK_SEL_DUMP diagnostic: (layer, k_positions, k_sel) captured.
    sel_dump: std::cell::Cell<Option<(usize, usize, usize)>>,
}

impl<'a> SingleBufferEncoder<'a> {
    /// Construct the encoder. `raw_cap` is the per-layer KV ring
    /// capacity (same value `decode_runner` passes to `AttnStepState::new`).
    pub fn new(dispatcher: &'a crate::MetalDispatcher, raw_cap: u32) -> Self {
        Self {
            dispatcher,
            cfg: DecodeConfig::default(),
            raw_cap,
            comp_prefill_finishes: std::cell::RefCell::new(Vec::new()),
            chunk_resyncs: std::cell::RefCell::new(Vec::new()),
            heads_dump: std::cell::Cell::new(None),
            sel_dump: std::cell::Cell::new(None),
        }
    }

    /// Compute the default cutpoint for a model. `L_split = n_layers / 2`.
    pub fn cutpoint_for(&self, model: &ComposedModelWeights) -> LayerCutpoint {
        LayerCutpoint::middle(model.layers.len())
    }

    /// Phase D (M4 #365). Classify which layers need host sync between the
    /// buffer-A and buffer-B halves of the per-token encode. Public so
    /// callers can inspect the partitioning before/after a decode step
    /// without re-walking the model.
    pub fn plan_for(&self, model: &ComposedModelWeights) -> BatchedBranchingPlan {
        BatchedBranchingPlan::for_model(model)
    }

    /// Decode a single token, returning logits.
    ///
    /// Phase C.2 (M4 #330n): runs the residual half through the trait
    /// dispatcher (`decode_step_with_attn_to_residual`) and replaces the
    /// final `rms_norm → (q8_0?) → matvec_f32(lm_head)` tail with one
    /// `MTLCommandBuffer` via `tail_lm_head_batched`. Result is
    /// bit-identical to `decode_step_with_attn` because both the tail
    /// kernels and the residual helper run the same ops in the same order;
    /// only host/GPU sync points are batched.
    ///
    /// Phase E M5.4.5 (next): replace the trait-dispatch residual call
    /// with a per-layer encoder loop that uses the composable BatchScope
    /// ops landed in M5.4.1-M5.4.5-prep:
    ///
    ///   - `BatchScope::encode_layer_attn_half` (M5.4.1) — hc_collapse +
    ///     qkv + kv_norm_rope + tail rope
    ///   - `BatchScope::kv_fp8_store_persistent` (M5.4.2) — KV write
    ///   - `MetalDispatcher::flash_attn_decode_metal_persistent` (M5.2.3)
    ///     — flash_attn read from persistent buffer (cb boundary remains
    ///     pending GPU f32→f16 conversion)
    ///   - `BatchScope::encode_attn_output_matmuls` (M5.4.3)
    ///   - `BatchScope::hc_expand_attn` (M5.4.5-prep) — attn-half residual
    ///   - `BatchScope::hc_collapse_norm` (M5.1) — ffn-half normed
    ///   - `BatchScope::encode_router_logits` (M5.4.4)
    ///   - `BatchScope::encode_router_finalize` (M5.4.4-followup)
    ///   - `BatchScope::encode_moe_and_shared_chain` (M5.4.5-prep)
    ///   - `BatchScope::hc_expand_add` (M1) — final ffn-half residual
    ///
    /// Two cb boundaries remain inside a layer:
    ///   1. flash_attn_decode (needs GPU f32→f16 of KV; deferred)
    ///   2. router_finalize → moe (selected/weights still CPU vecs in
    ///      the moe kernel ABI; could be lifted to GPU buffers by a
    ///      future moe-kernel ABI tweak)
    ///
    /// Estimated per-layer cb count after M5.4.5 wiring: 3 (down from
    /// ~6-8 today). Per-token cb count: ~130 + LM head + argmax.
    /// Bench impact: see `DS4_OP_TRACE` per-op wait totals; expect
    /// ~3× tok/s gain from collapsing 2× cb commits per layer at the
    /// remaining boundaries.
    ///
    /// `l_split` is currently honoured only by recording it on the
    /// plan struct for later inspection.
    /// Greedy variant of `decode_token_batched`: returns the GPU-argmax token
    /// id directly (no full ~vocab_size logit readback). Use for greedy decode;
    /// `decode_token_batched` (full logits) is for sampling / diagnostics.
    pub fn decode_token_argmax(
        &self,
        x: Vec<f32>,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
    ) -> Result<i32> {
        if l_split.0 > model.layers.len() {
            bail!(
                "decode_token_argmax: L_split={} exceeds n_layers={}",
                l_split.0,
                model.layers.len()
            );
        }
        let final_hidden = decode_step_with_attn_to_residual(
            self.dispatcher,
            self.dispatcher,
            x,
            model,
            state,
            &self.cfg,
            self.raw_cap,
        )?;
        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        Ok(self.tail_lm_head_argmax(&final_hidden, model, want_q80))
    }

    pub fn decode_token_batched(
        &self,
        x: Vec<f32>,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
    ) -> Result<Vec<f32>> {
        if l_split.0 > model.layers.len() {
            bail!(
                "decode_token_batched: L_split={} exceeds n_layers={}",
                l_split.0,
                model.layers.len()
            );
        }

        let final_hidden = decode_step_with_attn_to_residual(
            self.dispatcher,
            self.dispatcher,
            x,
            model,
            state,
            &self.cfg,
            self.raw_cap,
        )?;

        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        Ok(self.tail_lm_head_batched(&final_hidden, model, want_q80))
    }

    /// Phase E M5.4.5.bench — decode one token via the unified-cb path.
    ///
    /// Runs `encode_first_half(l_split=n_layers)` to update
    /// `state.cur_hc` through all layers, then applies the same tail
    /// the trait-dispatch path uses:
    ///   - `output_hc_head_one(state.cur_hc)` → `final_hidden` (or
    ///     slot-0 fallback when output_hc_* tensors are absent),
    ///   - `state.pos += 1`,
    ///   - `tail_lm_head_batched(final_hidden)` → logits.
    ///
    /// Equivalent semantically to `decode_step_with_attn`'s output for
    /// dense+hash-routing+compressor+indexer DS4 V4 Flash models.
    /// Numerical drift bound matches the GGUF smoke envelope (see
    /// `project_m5_kv_fp8_store_divergence` memory): per-token rel error
    /// dominated by kv_fp8_store CPU-vs-GPU drift, ~23% rel on the q2
    /// DS4 V4 Flash GGUF for a single token at pos=0. Strict bit-close
    /// gated on M5.2.4 kv_fp8_store unification.

    /// SSD-streaming: drain the scope so GPU-selected expert ids are readable,
    /// ensure them in the layer's cache, and return a fresh ids buffer holding
    /// SLOT ids (cache pool indices). No-op (returns input) when streaming off.
    fn stream_remap_sel(
        &self,
        scope: &mut crate::deferred::BatchScope<'_>,
        layer_idx: usize,
        sel_db: crate::deferred::DeferredBuf,
        w_db: &crate::deferred::DeferredBuf,
    ) -> crate::deferred::DeferredBuf {
        if std::env::var("DS4_SSD_STREAM").is_err()
            || std::env::var("DS4_SSD_STUB").map(|v| v == "0").unwrap_or(false)
        {
            return sel_db;
        }
        let _ = scope.commit_wait_read_multi(&[w_db]);
        let n = sel_db.len();
        let ids: Vec<i32> = unsafe {
            std::slice::from_raw_parts(sel_db.buffer().contents() as *const i32, n)
        }
        .to_vec();
        let (slots, _g, _u, _d) = self
            .dispatcher
            .streaming_expert_bind(layer_idx as u32, &ids)
            .expect("ssd-stream: no expert cache for layer");
        let slot_db = scope.alloc_i32(n);
        unsafe {
            std::ptr::copy_nonoverlapping(slots.as_ptr(), slot_db.buffer().contents() as *mut i32, n);
        }
        slot_db
    }

    pub fn decode_token_via_first_half(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
    ) -> Result<Vec<f32>> {
        let final_hidden = self.decode_first_half_to_final_hidden(x, model, state)?;
        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        let logits = self.tail_lm_head_batched(&final_hidden, model, want_q80);
        // All-zero logits = a silently-faulted lm-head command buffer (the fault
        // is DOWNSTREAM of final_hidden, which is fine here — empirically the
        // tail_lm_head cb is what zeroes under GPU resource/perturbation stress,
        // status!=Error so assert_cb_ok can't catch it). A real lm-head matmul of a
        // nonzero hidden is never exactly all-zero. Surface it instead of emitting
        // an argmax=0 garbage token. DS4_NO_ZERO_HIDDEN_GUARD=1 disables (so the
        // diagnostic fire-rate tools can OBSERVE the zeros). See the
        // ds4-chunk-prefill-decode-chain-race memory.
        if std::env::var("DS4_NO_ZERO_HIDDEN_GUARD").is_err()
            && logits.iter().all(|&v| v == 0.0)
        {
            anyhow::bail!(
                "decode pos={}: lm-head logits all-zero — a command buffer silently \
                 faulted (GPU resource/perturbation limit; status!=Error). The output \
                 would be a garbage all-zeros token stream. On a heavily-run box this \
                 is the documented GPU-perturbation cb-fault — reboot to clear.",
                state.pos.saturating_sub(1),
            );
        }
        Ok(logits)
    }

    /// Prefill a prompt token: run the 43-layer first half (updates the KV
    /// cache / compressor / indexer / `cur_hc` residual + advances `pos`) but
    /// SKIP the `output_hc` fold and the 129k-vocab lm-head tail — only the
    /// LAST prompt token needs logits. `final_hidden` is the lm-head input and
    /// is never fed back, so dropping it is exact for the prefill state. Saves
    /// the per-token lm-head matmul + full-logit readback across the prompt
    /// (the @3000 TTFT cost). The caller runs the last token through
    /// `decode_token_via_first_half` for the first generated token's logits.
    pub fn prefill_step(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
    ) -> Result<()> {
        self.encode_first_half(x, model, state, LayerCutpoint(model.layers.len()))?;
        crate::deferred::drain_trace_dump_and_reset(&format!("pos={}", state.pos));
        state.pos = state.pos.saturating_add(1);
        Ok(())
    }

    /// Shared core of the first-half decode entries: run the full fast chain
    /// (`encode_first_half` over all layers), fold `cur_hc` → `final_hidden`
    /// (`d_embd`, mirrors decode_step.rs:1326-1342), and advance `pos`. Callers
    /// apply the lm-head tail — full logits via `tail_lm_head_batched`, or GPU
    /// argmax via `tail_lm_head_argmax`.
    fn decode_first_half_to_final_hidden(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
    ) -> Result<Vec<f32>> {
        let prof = std::env::var("DS4_TAIL_PROFILE").is_ok();
        let tc = std::time::Instant::now();
        self.encode_first_half(x, model, state, LayerCutpoint(model.layers.len()))?;
        let chain_us = tc.elapsed().as_micros();
        let first = &model.layers[0].attn.params;
        let n_hc = first.n_hc as usize;
        let d_embd = first.d_embd as usize;
        let to = std::time::Instant::now();
        let final_hidden = self.fold_output_hc(model, state, n_hc, d_embd);
        if prof {
            eprintln!(
                "[phase] chain={}us output_hc={}us (separate cb + cur_hc readback)",
                chain_us,
                to.elapsed().as_micros(),
            );
        }
        // Step 0 drain instrumentation: per-token flush_and_read* count by call site.
        crate::deferred::drain_trace_dump_and_reset(&format!("pos={}", state.pos));
        // Step 1 divergence diff: dump a ratio=128 layer's comp ring at the first
        // emits so DS4_FUSED_COMP=1 (resident) and =0 (staged) can be compared.
        // Both paths populate comp_kv_ring (staged directly; resident via the
        // finish). If the rings match, the bug is in the flash comp-attention;
        // if they differ, it's the resident emit row.
        if std::env::var("DS4_COMP_DUMP").is_ok()
            && (state.pos == 127 || state.pos == 255)
            && state.comp_kv_ring.len() > 3
        {
            for l in [3usize, 5] {
                let r = &state.comp_kv_ring[l];
                let sum: f64 = r.iter().map(|&x| x as f64).sum();
                let head: Vec<f32> = r.iter().take(4).copied().collect();
                eprintln!(
                    "[comp-dump] pos={} L{l} n_comp={} ring_len={} sum={:.6} head={:?}",
                    state.pos, state.n_comp[l], r.len(), sum, head,
                );
            }
        }
        // All-zero `final_hidden` = a silently-faulted command buffer in the layer
        // chain. A cb that hits a GPU resource/perturbation limit yields ZEROED
        // outputs WITHOUT status=Error, so `assert_cb_ok` cannot catch it (the
        // documented decode all-zeros); a real 43-layer residual fold is never
        // exactly all-zero. Surface it here instead of letting the lm-head emit an
        // argmax=0 garbage token stream (this guards BOTH the full-logits and the
        // GPU-argmax decode paths, which share `final_hidden`). See the
        // ds4-chunk-prefill-decode-chain-race memory. DS4_NO_ZERO_HIDDEN_GUARD=1
        // disables (diagnostic, to inspect the raw all-zeros behavior).
        if std::env::var("DS4_NO_ZERO_HIDDEN_GUARD").is_err()
            && final_hidden.iter().all(|&v| v == 0.0)
        {
            anyhow::bail!(
                "decode pos={}: final_hidden all-zero — a layer-chain command buffer \
                 silently faulted (GPU resource/perturbation limit; status!=Error, so \
                 the cb-error guard can't catch it). The output would be a garbage \
                 all-zeros token stream. On a heavily-run box this is the documented \
                 GPU-perturbation cb-fault — reboot to clear; see the \
                 ds4-chunk-prefill-decode-chain-race memory.",
                state.pos
            );
        }
        state.pos = state.pos.saturating_add(1);
        Ok(final_hidden)
    }

    /// Fold the (CPU-resident) `state.cur_hc` HC residual → `final_hidden`
    /// (`d_embd`) via the output-HC head. Shared by the drain path
    /// (`decode_first_half_to_final_hidden`) and the resident-tail fallback.
    ///
    /// The output-HC fold is tiny (~80K MACs: rms_norm(hc_dim) + matvec
    /// hc_dim→n_hc + a d_embd fold). `cur_hc` is already on CPU after the
    /// chain drain, so running it on the CPU dispatcher avoids a ~0.45ms
    /// SEPARATE GPU cb + readback per token — pure seam. DS4_CPU_OUTPUT_HC=0
    /// reverts to the Metal dispatch.
    fn fold_output_hc(
        &self,
        model: &ComposedModelWeights,
        state: &AttnStepState,
        n_hc: usize,
        d_embd: usize,
    ) -> Vec<f32> {
        match (
            model.output_hc_base.as_deref(),
            model.output_hc_fn.as_deref(),
            model.output_hc_scale.as_deref(),
        ) {
            (Some(base), Some(fn_w), Some(scale)) => {
                if std::env::var("DS4_CPU_OUTPUT_HC").ok().as_deref() != Some("0") {
                    output_hc_head_one(
                        &ds4_engine::dispatch::CpuDispatcher,
                        &state.cur_hc, base, fn_w, scale,
                        n_hc, d_embd, self.cfg.eps_rms,
                    )
                } else {
                    output_hc_head_one(
                        self.dispatcher,
                        &state.cur_hc, base, fn_w, scale,
                        n_hc, d_embd, self.cfg.eps_rms,
                    )
                }
            }
            _ => state.cur_hc[..d_embd].to_vec(),
        }
    }

    /// Greedy fast decode: the `encode_first_half` chain + GPU argmax tail
    /// (no full ~vocab logit readback). The throughput-optimised counterpart
    /// to `decode_token_via_first_half` (which returns full logits for
    /// sampling/diagnostics). Used by the ds4-infer decode runner's fast path.
    pub fn decode_token_via_first_half_argmax(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
    ) -> Result<i32> {
        let prof = std::env::var("DS4_TAIL_PROFILE").is_ok();
        // DS4_RESIDENT_TAIL (default-on; =0 reverts to the drain path): keep the
        // final HC residual GPU-resident and fold the output-head + q8 lm-head +
        // argmax in-scope on it, with ONE drain at token end (read back just the
        // 4-byte token id, not cur_hc). Needs the output-HC head tensors + a q8
        // lm-head; falls through to the drain path without them, or when a layer
        // breaks the chain mid-token. Token-identical to the drain path,
        // +~1% decode (seam-overhead removal); see encode_first_half_gguf_smoke.
        // The resident path folds the q8 lm-head in-scope (encode_mtp_output_head
        // → matvec_q8_0); when the lm-head was re-quantized to q4 (q8 dropped),
        // fall to the drain path + the standalone q4 tail instead.
        let resident_on = std::env::var("DS4_RESIDENT_TAIL").ok().as_deref() != Some("0");
        let have_hc_head = model.output_hc_base.is_some()
            && model.output_hc_fn.is_some()
            && model.output_hc_scale.is_some()
            && !model.lm_head_q8.is_empty()
            && model.lm_head_q4.is_empty();
        if resident_on && have_hc_head {
            return self.decode_token_resident_tail(x, model, state, prof);
        }
        let t0 = std::time::Instant::now();
        let final_hidden = self.decode_first_half_to_final_hidden(x, model, state)?;
        let layers_us = t0.elapsed().as_micros();
        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        let t1 = std::time::Instant::now();
        let tok = self.tail_lm_head_argmax(&final_hidden, model, want_q80);

        if prof {
            eprintln!(
                "[tail] layers={}us tail={}us (lm_head {}->{} + rms_norm + argmax)",
                layers_us,
                t1.elapsed().as_micros(),
                model.d_model,
                model.vocab_size,
            );
        }
        Ok(tok)
    }

    /// `DS4_RESIDENT_TAIL` greedy decode. Runs the chain in no-drain mode; when
    /// the whole token stayed chained, the final HC residual is GPU-resident,
    /// so the output-head (rms → hc_head_fn matvec → sigmoid gate →
    /// hc_weighted_sum fold → final rms → q8 lm-head) + argmax all encode
    /// in-scope on it and flush ONCE — eliminating the cur_hc→CPU readback and
    /// the separate lm-head command buffer's launch latency. The single flush
    /// waits the chain cbs by command-queue commit order; the deferred
    /// compressor finishes run after (they need the chain GPU work complete).
    /// If a layer broke the chain (drained mid-loop), `cur_hc` is on CPU and we
    /// fold + tail on the CPU/standalone path exactly as the drain path does.
    ///
    /// In-scope math mirrors `output_hc_head_one` + `tail_lm_head_argmax_q8`
    /// (same kernels) — token-identical to the drain path modulo GPU-vs-CPU
    /// float reassociation in the fold/gate (the accepted chained-path rel
    /// delta), validated by the GGUF fidelity gates.
    fn decode_token_resident_tail(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        prof: bool,
    ) -> Result<i32> {
        let first = &model.layers[0].attn.params;
        let n_hc = first.n_hc as usize;
        let d_embd = first.d_embd as usize;
        let vocab = model.vocab_size;
        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");

        let t0 = std::time::Instant::now();
        let mut resident: Option<ResidentChain> = None;
        self.encode_first_half_resident(
            x, model, state, LayerCutpoint(model.layers.len()), &mut resident,
        )?;
        state.pos = state.pos.saturating_add(1);
        let layers_us = t0.elapsed().as_micros();

        let Some(ResidentChain { hc, cbs, finishes }) = resident else {
            // Chain broke mid-loop → cur_hc already on CPU. Fold + tail the
            // standard (drain-path) way.
            let final_hidden = self.fold_output_hc(model, state, n_hc, d_embd);
            let tok = self.tail_lm_head_argmax(&final_hidden, model, want_q80);
            if prof {
                eprintln!("[tail] layers={}us tail=cpu-fallback (chain not resident)", layers_us);
            }
            return Ok(tok);
        };

        let t1 = std::time::Instant::now();
        // Fold the output-head + q8 lm-head + argmax in-scope on the resident
        // HC. Inputs reuse the host-ptr-keyed weight caches (warm after token 0).
        let unit_gamma_hc = vec![1.0f32; n_hc * d_embd];
        let scope = self.dispatcher.batch_scope();
        let cur_hc_db = hc; // resident final-HC buffer (n_hc * d_embd)
        let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);
        let fn_db = scope.weight_f32(model.output_hc_fn.as_deref().unwrap());
        let scale_db = scope.weight_f32(model.output_hc_scale.as_deref().unwrap());
        let base_db = scope.weight_f32(model.output_hc_base.as_deref().unwrap());
        let final_norm_db = scope.weight_f32(&model.final_norm_gamma);
        let lm_head_db = scope.weight_q8_0_raw(&model.lm_head_q8, vocab * d_embd);

        let logits = scope.encode_mtp_output_head(
            &cur_hc_db,
            &unit_gamma_db,
            &fn_db, &scale_db, &base_db,
            &final_norm_db,
            &lm_head_db,
            d_embd, n_hc, vocab,
            self.cfg.eps_rms,
            // DS4_HC_EPS — matches the additive trust eps in output_hc_head_one.
            1.0e-6,
        )?;
        let token_db = scope.argmax(&logits, vocab)?;
        // Single drain: commits + waits the tail cb, which by commit order
        // implies the chain cbs (committed earlier on the same queue) are done.
        let tok = scope.flush_and_read_i32(&token_db)[0];
        // Chain GPU work is now complete; the cbs were held alive until here.
        drop(cbs);
        // Deferred compressor/indexer state updates (read emit pools, append rings).
        self.run_chain_finishes(model, state, finishes);
        if prof {
            eprintln!(
                "[tail] layers={}us resident_tail={}us (in-scope output_hc + q8 lm_head {}->{} + argmax, 1 drain)",
                layers_us,
                t1.elapsed().as_micros(),
                model.d_model,
                vocab,
            );
        }
        Ok(tok)
    }

    /// Phase E M5.4.5.1 — single-layer attn-half driven from
    /// `SingleBufferEncoder`. Uploads the per-layer weight slices,
    /// encodes `BatchScope::encode_layer_attn_half` +
    /// `kv_fp8_store_persistent` into ONE command buffer, commits/waits
    /// exactly once, and returns the CPU-side outputs.
    ///
    /// Side effect: the persistent per-layer KV cache buffer for
    /// `layer_idx` is updated at `slot` (lazy-allocated by
    /// `MetalState::kv_buffer_or_alloc` on first call).
    ///
    /// `unit_gamma_hc_dim` must be a ones vector of length `n_hc *
    /// n_embd` (used by the rms_norm_plain step inside
    /// `hc_collapse_norm`). The caller owns it because we want to share
    /// one allocation across the per-token layer loop in M5.4.5.2.
    ///
    /// Future M5.4.5.2 will drive this for every layer in [0..l_split]
    /// inside `encode_first_half`, using `ComposedLayerWeights` as the
    /// source of weight slices and `AttnStepState::pos` /
    /// `AttnStepState::kv_pos[layer_idx]` for the rotary / slot args.
    ///
    /// Bit-identical to the legacy two-cb path
    /// (`hc_collapse_norm` → `attn_qkv_chain_batched` →
    /// `kv_norm_rope_chain` → `kv_fp8_store_persistent`) — every kernel
    /// runs in the same order against the same operands; only host/GPU
    /// sync points are batched. See
    /// `tests/encode_layer_attn_half_cpu_smoke.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_attn_half_cpu(
        &self,
        layer_idx: u32,
        prev_hc: &[f32],
        hc_attn_fn: &[f32],
        hc_attn_fn_f16: Option<&[u8]>,
        hc_attn_scale: &[f32],
        hc_attn_base: &[f32],
        hc_norm_gamma: &[f32],
        unit_gamma_hc_dim: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        attn_q_b: &[f32],
        attn_kv: &[f32],
        qkv_gamma_kv: &[f32],
        params: &LayerParams,
        pos: u32,
        raw_cap: u32,
        slot: u32,
    ) -> Result<CpuLayerAttnHalfOuts> {
        let n_hc = params.n_hc as usize;
        let n_embd = params.d_embd as usize;
        let n_lora_q = params.n_lora_q as usize;
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let kv_row = params.n_lora_kv as usize;

        // Phase E quick-win (#81): switch lifetime-stable weight slices
        // from `upload_f32` (fresh metal::Buffer per call) to `weight_f32`
        // (lazy cached via `cached_weight_buffer`, keyed by host_ptr +
        // byte_len). The 9 weight slices below come from
        // `ComposedLayerWeights` which is held for the model's lifetime,
        // so the host_ptr stays stable across decode steps — first call
        // per layer uploads, subsequent calls reuse.
        //
        // `prev_hc` is an activation (mutates per token + per layer) and
        // stays on `upload_f32`. `unit_gamma_hc_dim` is built fresh per
        // decode token by `encode_first_half_inner` (so within one
        // token the pointer is stable across 43 layers — one upload,
        // 42 cache hits — but a fresh Vec each token would leak buffers
        // via the cache; safer to `upload_f32` it too).
        // Phase F task #92: ref's `hc_collapse_norm_impl` (macos.rs:5550)
        // does NOT cache the hc_* slices — it uses fresh per-call buffers.
        // The uni path's prior `weight_f32` for these added 4 × 43 = 172
        // unique cache entries that ref doesn't have, inflating the
        // buffer-count delta in [[m5-buffer-audit]] without correctness
        // benefit. Switch to `upload_f32` to match ref's behavior. The
        // hc_* slices are small (≤ ~64 KB each) so the per-decode memcpy
        // cost is negligible.
        let mut scope = self.dispatcher.batch_scope();
        let prev_hc_b = scope.upload_f32(prev_hc);
        // f16 no-copy (lean) else upload f32 (matches ref's fresh-buffer behavior).
        let (hc_fn_b, hc_fn_is_f16) = match hc_attn_fn_f16 {
            Some(b) => (scope.weight_f16(b), true),
            None => (scope.upload_f32(hc_attn_fn), false),
        };
        let hc_scale_b = scope.upload_f32(hc_attn_scale);
        let hc_base_b = scope.upload_f32(hc_attn_base);
        let hc_gamma_b = scope.upload_f32(hc_norm_gamma);
        let unit_gamma_b = scope.upload_f32(unit_gamma_hc_dim);
        let attn_q_a_b = scope.weight_f32(attn_q_a);
        // Phase F task #92: ref's `attn_qkv_chain_batched` (deferred.rs:2018)
        // uses `upload_f32(gamma_q)` not `weight_f32`. Match its behavior so
        // ref+uni share cache keys → drops 2 × 43 = 86 unique entries from
        // the uni dispatcher's weight cache. gamma_q and qkv_gamma_kv are
        // small (n_lora_q / kv_row × f32 ≤ 2 KB) so per-decode memcpy cost
        // is negligible.
        let gamma_q_b = scope.upload_f32(gamma_q);
        let attn_q_b_b = scope.weight_f32(attn_q_b);
        let attn_kv_b = scope.weight_f32(attn_kv);
        let gamma_kv_b = scope.upload_f32(qkv_gamma_kv);

        let half = scope.encode_layer_attn_half(
            &prev_hc_b,
            &hc_fn_b,
            &hc_scale_b,
            &hc_base_b,
            &hc_gamma_b,
            &unit_gamma_b,
            hc_fn_is_f16,
            &attn_q_a_b,
            &gamma_q_b,
            &attn_q_b_b,
            &attn_kv_b,
            &gamma_kv_b,
            n_hc,
            n_embd,
            n_lora_q,
            n_head,
            head_dim,
            kv_row,
            params.hc_sinkhorn_iter as i32,
            params.hc_eps,
            params.rms_eps,
            params,
            pos,
        )?;
        // Encode kv_fp8_store_persistent into the SAME scope — keeps the
        // persistent buffer write inside one cb commit.
        let _cache = scope.kv_fp8_store_persistent(
            layer_idx,
            &half.kv_normed_rotated,
            params,
            raw_cap,
            slot,
        )?;

        // `kv_normed_rotated` is NOT read back: the GPU `kv_fp8_store_persistent`
        // above already wrote the correct bytes into the persistent slot (see
        // the "no CPU correction needed" note in the per-layer driver), so the
        // CPU no longer needs the mirror.
        let cpu = scope.flush_and_read_multi(&[
            &half.normed,
            &half.split,
            &half.qr_normed,
            &half.q_heads,
        ]);
        Ok(CpuLayerAttnHalfOuts {
            normed: cpu[0].clone(),
            split: cpu[1].clone(),
            qr_normed: cpu[2].clone(),
            q_heads: cpu[3].clone(),
            kv_normed_rotated: Vec::new(),
        })
    }

    /// Scope-merge foundation: same dispatches as
    /// [`encode_layer_attn_half_cpu`] but returns the open scope and the
    /// GPU-resident outputs as DeferredBufs without flushing, so the
    /// compressor can encode into the same cb (`normed` stays resident).
    /// `encode_layer_attn_half_cpu` is the flush-and-read wrapper around
    /// this. Returned bufs hold owned buffers, so they outlive nothing
    /// they don't already own.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_attn_half_open<'s>(
        &'s self,
        layer_idx: u32,
        prev_hc: &[f32],
        hc_attn_fn: &[f32],
        hc_attn_fn_f16: Option<&[u8]>,
        hc_attn_scale: &[f32],
        hc_attn_base: &[f32],
        hc_norm_gamma: &[f32],
        unit_gamma_hc_dim: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        attn_q_b: &[f32],
        attn_kv: &[f32],
        qkv_gamma_kv: &[f32],
        // Raw Q8_0 bytes for the q/kv projections. MUST be forwarded (not left
        // empty): under lean weights the f32 `attn_q_a`/`q_b`/`kv` above are empty
        // (the q8 path is authoritative), so an empty-q8 + empty-f32 buffer would
        // be 0-length and `encode_attn_qkv_chain` would panic on the shape check.
        attn_q_a_q8: &[u8],
        attn_q_b_q8: &[u8],
        attn_kv_q8: &[u8],
        params: &LayerParams,
        pos: u32,
        raw_cap: u32,
        slot: u32,
    ) -> Result<(crate::deferred::BatchScope<'s>, crate::deferred::LayerAttnHalfOuts)> {
        self.encode_layer_attn_half_open_resident(
            layer_idx, prev_hc, None, hc_attn_fn, hc_attn_fn_f16, hc_attn_scale, hc_attn_base,
            hc_norm_gamma, unit_gamma_hc_dim, attn_q_a, gamma_q, attn_q_b, attn_kv,
            qkv_gamma_kv, attn_q_a_q8, attn_q_b_q8, attn_kv_q8, params, pos, raw_cap, slot,
        )
    }

    /// Layer-chaining (step 10) variant of [`Self::encode_layer_attn_half_open`]:
    /// when `prev_hc_resident` is `Some`, the previous layer's GPU-resident
    /// `after_ffn` (cur_hc) is bound directly as `prev_hc` — no readback +
    /// re-upload of the 16K-element residual between layers. The buffer comes
    /// from a different (already-committed-async) scope; Metal serializes cbs
    /// within the queue so this layer's reads see it. `prev_hc` (the CPU slice)
    /// is then ignored.
    #[allow(clippy::too_many_arguments)]
    /// Encode the attention output projection (`w_o_a` grouped → `w_o_b` dense)
    /// into `scope`, preferring the resident raw-Q8_0 path (1 byte/weight, the
    /// harness's #2 attn-half consumer) when `DS4_Q8_PROJ` is set and the layer
    /// carries raw bytes; else the f32 path. Returns `attn_out` (length d_embd).
    #[allow(clippy::too_many_arguments)]
    fn encode_output_proj(
        &self,
        scope: &crate::deferred::BatchScope<'_>,
        attn: &ds4_engine::attn_dispatch::AttnLayerWeights,
        heads_b: &crate::deferred::DeferredBuf,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        d_embd: usize,
    ) -> Result<crate::deferred::DeferredBuf> {
        let q8 = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true)
            && !attn.w_o_a_q8.is_empty()
            && !attn.w_o_b_q8.is_empty();
        let (_low, attn_out) = if q8 {
            // Element count from the q8 bytes (34 B / 32 weights), NOT the f32
            // Vec — the f32 dup is freed by `free_dead_f32_weights` in the server.
            let a = scope.weight_q8_0_raw(&attn.w_o_a_q8, (attn.w_o_a_q8.len() / 34) * 32);
            let b = scope.weight_q8_0_raw(&attn.w_o_b_q8, (attn.w_o_b_q8.len() / 34) * 32);
            scope.encode_attn_output_matmuls_q8(
                heads_b, &a, &b, n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
            )?
        } else {
            let a = scope.weight_f32(&attn.w_o_a);
            let b = scope.weight_f32(&attn.w_o_b);
            scope.encode_attn_output_matmuls(
                heads_b, &a, &b, n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
            )?
        };
        Ok(attn_out)
    }

    /// Encode the attn-half + kv_fp8_store into an EXISTING scope (no new scope
    /// created, no commit). Sibling of `encode_layer_attn_half_open_resident`;
    /// used by the multi-layer-per-cb path (DS4_LAYERS_PER_CB>1) to share one
    /// scope across consecutive layers and reduce inter-cb idle.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_attn_half_in_scope<'s, 'sc>(
        &'s self,
        scope: &'sc mut crate::deferred::BatchScope<'s>,
        layer_idx: u32,
        prev_hc: &[f32],
        prev_hc_resident: Option<&crate::deferred::DeferredBuf>,
        hc_attn_fn: &[f32],
        hc_attn_fn_f16: Option<&[u8]>,
        hc_attn_scale: &[f32],
        hc_attn_base: &[f32],
        hc_norm_gamma: &[f32],
        unit_gamma_hc_dim: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        attn_q_b: &[f32],
        attn_kv: &[f32],
        qkv_gamma_kv: &[f32],
        attn_q_a_q8: &[u8],
        attn_q_b_q8: &[u8],
        attn_kv_q8: &[u8],
        params: &LayerParams,
        pos: u32,
        raw_cap: u32,
        slot: u32,
    ) -> Result<crate::deferred::LayerAttnHalfOuts> {
        let n_hc = params.n_hc as usize;
        let n_embd = params.d_embd as usize;
        let n_lora_q = params.n_lora_q as usize;
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let kv_row = params.n_lora_kv as usize;
        let prev_hc_upload;
        let prev_hc_b: &crate::deferred::DeferredBuf = match prev_hc_resident {
            Some(r) => r,
            None => {
                prev_hc_upload = scope.upload_f32(prev_hc);
                &prev_hc_upload
            }
        };
        let (hc_fn_b, hc_fn_is_f16) = scope.weight_hc(hc_attn_fn, hc_attn_fn_f16);
        let hc_scale_b = scope.weight_f32(hc_attn_scale);
        let hc_base_b = scope.weight_f32(hc_attn_base);
        let hc_gamma_b = scope.weight_f32(hc_norm_gamma);
        let unit_gamma_b = scope.weight_f32(unit_gamma_hc_dim);
        let q8_proj = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true);
        scope.set_q8_proj(q8_proj);
        let q8_raw = q8_proj
            && !attn_q_a_q8.is_empty()
            && !attn_q_b_q8.is_empty()
            && !attn_kv_q8.is_empty();
        let (attn_q_a_b, attn_q_b_b, attn_kv_b) = if q8_raw {
            (
                // Element counts from the q8 bytes (34 B / 32 weights), NOT the
                // f32 Vecs (freed by `free_dead_f32_weights` in the server).
                scope.weight_q8_0_raw(attn_q_a_q8, (attn_q_a_q8.len() / 34) * 32),
                scope.weight_q8_0_raw(attn_q_b_q8, (attn_q_b_q8.len() / 34) * 32),
                scope.weight_q8_0_raw(attn_kv_q8, (attn_kv_q8.len() / 34) * 32),
            )
        } else if q8_proj {
            (
                scope.weight_q8_0(attn_q_a),
                scope.weight_q8_0(attn_q_b),
                scope.weight_q8_0(attn_kv),
            )
        } else {
            (
                scope.weight_f32(attn_q_a),
                scope.weight_f32(attn_q_b),
                scope.weight_f32(attn_kv),
            )
        };
        let gamma_q_b = scope.weight_f32(gamma_q);
        let gamma_kv_b = scope.weight_f32(qkv_gamma_kv);
        let half = scope.encode_layer_attn_half(
            prev_hc_b, &hc_fn_b, &hc_scale_b, &hc_base_b, &hc_gamma_b, &unit_gamma_b,
            hc_fn_is_f16,
            &attn_q_a_b, &gamma_q_b, &attn_q_b_b, &attn_kv_b, &gamma_kv_b,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            params.hc_sinkhorn_iter as i32, params.hc_eps, params.rms_eps, params, pos,
        )?;
        let _cache = scope.kv_fp8_store_persistent(
            layer_idx, &half.kv_normed_rotated, params, raw_cap, slot,
        )?;
        Ok(half)
    }

    pub fn encode_layer_attn_half_open_resident<'s>(
        &'s self,
        layer_idx: u32,
        prev_hc: &[f32],
        prev_hc_resident: Option<&crate::deferred::DeferredBuf>,
        hc_attn_fn: &[f32],
        hc_attn_fn_f16: Option<&[u8]>,
        hc_attn_scale: &[f32],
        hc_attn_base: &[f32],
        hc_norm_gamma: &[f32],
        unit_gamma_hc_dim: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        attn_q_b: &[f32],
        attn_kv: &[f32],
        qkv_gamma_kv: &[f32],
        // Raw GGUF block_q8_0 bytes for the q/kv projections (owned,
        // page-aligned). Empty → fall back to re-quantize / f32. Used only when
        // DS4_Q8_PROJ is set.
        attn_q_a_q8: &[u8],
        attn_q_b_q8: &[u8],
        attn_kv_q8: &[u8],
        params: &LayerParams,
        pos: u32,
        raw_cap: u32,
        slot: u32,
    ) -> Result<(crate::deferred::BatchScope<'s>, crate::deferred::LayerAttnHalfOuts)> {
        let mut scope = self.dispatcher.batch_scope();
        let half = self.encode_layer_attn_half_in_scope(
            &mut scope, layer_idx, prev_hc, prev_hc_resident, hc_attn_fn, hc_attn_fn_f16,
            hc_attn_scale,
            hc_attn_base, hc_norm_gamma, unit_gamma_hc_dim, attn_q_a, gamma_q, attn_q_b,
            attn_kv, qkv_gamma_kv, attn_q_a_q8, attn_q_b_q8, attn_kv_q8, params, pos,
            raw_cap, slot,
        )?;
        Ok((scope, half))
    }

    /// Layer-chaining (step 10) token-end: wait on all the async-committed
    /// per-layer cbs, read the final resident `cur_hc` back to CPU, then run
    /// every deferred compressor finish (pool resync + ratio-4 rotation +
    /// comp_kv_ring append). Also used as the fallback when a layer can't
    /// chain. Consumes the chain state. `state.cur_hc` is updated iff `hc`
    /// is `Some` (it carries the last chained layer's after_ffn).
    fn finish_chain(
        &self,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        hc: Option<crate::deferred::DeferredBuf>,
        cbs: Vec<metal::CommandBuffer>,
        finishes: Vec<ChainFinish>,
    ) {
        let prof = std::env::var("DS4_CHAIN_PROFILE").is_ok();
        let n_cbs = cbs.len();
        let t_wait = std::time::Instant::now();
        // Command buffers on one queue complete in commit order, so waiting on
        // the LAST committed cb implies all prior ones are done. Waiting each
        // of the 40 individually added ~0.3ms/call of completion-handler /
        // dispatch overhead (~13ms/token) for no benefit.
        if let Some(last) = cbs.last() {
            last.wait_until_completed();
        }
        // Check EVERY chain cb for status=Error, not just completion. finish_chain
        // previously only waited on the last cb (commit-order ⇒ all complete) but
        // never verified none ERRORED — a silently-errored chain cb leaves its
        // output zero, propagating to an all-zeros final_hidden / logits with no
        // surfaced failure (the decode-phase all-zeros race localized via
        // chunk_decode_divergence). Turn that silent GPU UB into a clean panic.
        for cb in &cbs {
            crate::deferred::assert_cb_ok(cb, "finish_chain:chain_cb");
        }
        let wait_us = t_wait.elapsed().as_micros();
        // GPU-busy (sum of per-cb GPUStart→End) vs span (first start → last end).
        // span - sum = inter-cb idle (scheduling gaps); intra-cb idle (encoder
        // barriers) is hidden inside each cb's start→end.
        if prof {
            use metal::foreign_types::ForeignType;
            use objc::{msg_send, sel, sel_impl};
            let (mut sum_us, mut min_s, mut max_e) = (0.0f64, f64::INFINITY, 0.0f64);
            for cb in &cbs {
                let (s, e): (f64, f64) = unsafe {
                    let p: *mut objc::runtime::Object = std::mem::transmute(cb.as_ptr());
                    (msg_send![p, GPUStartTime], msg_send![p, GPUEndTime])
                };
                sum_us += (e - s) * 1e6;
                min_s = min_s.min(s);
                max_e = max_e.max(e);
            }
            let span_us = (max_e - min_s) * 1e6;
            eprintln!(
                "[chain] gpu_busy={:.0}us span={:.0}us inter_cb_idle={:.0}us",
                sum_us, span_us, span_us - sum_us
            );
        }
        let t_fin = std::time::Instant::now();
        if let Some(hc) = hc {
            state.cur_hc = unsafe {
                std::slice::from_raw_parts(hc.buffer().contents() as *const f32, hc.len())
            }
            .to_vec();
        }
        self.run_chain_finishes(model, state, finishes);
        if prof {
            eprintln!(
                "[chain] n_cbs={} gpu_wait={}us finish_cpu={}us",
                n_cbs, wait_us, t_fin.elapsed().as_micros()
            );
        }
    }

    /// Run the deferred compressor/indexer finishes accumulated across a
    /// chained token (the `ChainFinish` queue): resync the CPU mirror from
    /// the GPU emit pools, append emitted rows to the comp/indexer KV rings,
    /// bump `n_comp`/`n_index_comp`. Each `compressor_finish_in_scope` issues
    /// its own GPU dispatch(es), so the chain GPU work MUST already be
    /// complete (the caller waits the chain — directly via `finish_chain`, or
    /// by commit-order through the resident fused tail's flush — before
    /// invoking this). Extracted from `finish_chain` so the resident-tail
    /// decode path can defer it until AFTER the lm-head tail.
    fn run_chain_finishes(
        &self,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        finishes: Vec<ChainFinish>,
    ) {
        for f in finishes {
            let layer = &model.layers[f.layer_idx];
            let p = &layer.attn.params;
            let pooled = match &f.emit_db {
                Some(e) => unsafe {
                    std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                }
                .to_vec(),
                None => Vec::new(),
            };
            if f.is_indexer {
                let (kv_f16, gate_f16) = layer.attn.indexer_compressor_f16();
                let comp = ds4_engine::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.indexer_compressor_kv,
                    w_gate: &layer.attn.indexer_compressor_gate,
                    w_kv_f16: kv_f16,
                    w_gate_f16: gate_f16,
                    w_ape: &layer.attn.indexer_compressor_ape,
                    w_norm: &layer.attn.indexer_compressor_norm,
                    head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                    compress_ratio: p.compress_ratio,
                };
                if let Some(row) = crate::compressor::compressor_finish_in_scope(
                    f.handle, self.dispatcher, p, &comp, pooled,
                    &mut state.index_state_kv[f.layer_idx],
                    &mut state.index_state_score[f.layer_idx], state.pos,
                ) {
                    state.index_comp_kv_ring[f.layer_idx].extend_from_slice(&row);
                    state.n_index_comp[f.layer_idx] += 1;
                }
            } else {
                let (kv_f16, gate_f16) = layer.attn.attn_compressor_f16();
                let comp = ds4_engine::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.attn_compressor_kv,
                    w_gate: &layer.attn.attn_compressor_gate,
                    w_kv_f16: kv_f16,
                    w_gate_f16: gate_f16,
                    w_ape: &layer.attn.attn_compressor_ape,
                    w_norm: &layer.attn.attn_compressor_norm,
                    head_dim: p.head_dim,
                    compress_ratio: p.compress_ratio,
                };
                if let Some(row) = crate::compressor::compressor_finish_in_scope(
                    f.handle, self.dispatcher, p, &comp, pooled,
                    &mut state.comp_state_kv[f.layer_idx],
                    &mut state.comp_state_score[f.layer_idx], state.pos,
                ) {
                    state.comp_kv_ring[f.layer_idx].extend_from_slice(&row);
                    state.n_comp[f.layer_idx] += 1;
                }
            }
        }
    }

    /// Phase E M5.4.5.4 — per-token first-half driver: full per-layer
    /// pipeline (attn-half + post-attn-half + FFN-half).
    ///
    /// For each layer in [0..l_split]:
    ///   1. attn-half via `encode_layer_attn_half_cpu` (BatchScope:
    ///      `hc_collapse_norm → attn_qkv_chain → kv_norm_rope →
    ///      kv_fp8_store_persistent` — one cb per layer)
    ///   2. rope_q forward on Q heads (CPU)
    ///   3. `flash_attn_decode_metal_persistent` reading from the
    ///      persistent KV buffer populated in step 1
    ///   4. rope_q backward on heads (CPU)
    ///   5. `attn_output_proj` + hc_expand_attn → after_attn_hc
    ///   6-10. `run_ffn_half` (decode_attn_ffn_pre + router + MoE +
    ///      shared + decode_attn_ffn_post) → after_ffn_hc;
    ///      `state.cur_hc = after_ffn_hc`
    ///
    /// **What this milestone DOES NOT do:**
    ///   - support compressor (`compress_ratio > 1`) or hash-routing
    ///     (`routing_table.is_some()`) layers — bail until M5.4.5.5.
    ///
    /// Constraints (must hold on Metal):
    ///   - `head_dim == 512` and `n_lora_kv == head_dim` (production
    ///     shape; required by `flash_attn_decode_metal_persistent`).
    ///   - `layer.moe.n_experts == 256` and `d_embd % 256 == 0`
    ///     (production shape; required by `router_finalize` and
    ///     `moe_routed_step` on Metal).
    ///   - Expert weights pre-loaded via `MetalState::load_expert_weights`
    ///     (Metal MoE reads quantized expert tensors from
    ///     `expert_weights: Vec<QuantizedExpertWeights>`, NOT from the
    ///     `LayerWeights` f32 slices).
    ///
    /// These constraints are enforced by Metal kernels at dispatch time;
    /// the encoder doesn't pre-check them. End-to-end correctness is
    /// validated by the GGUF-gated integration smoke
    /// (`encode_first_half_gguf_smoke.rs`).
    ///
    /// Side effects on `state`:
    ///   - `state.cur_hc` initialized to the HC expansion of `x`, then
    ///     advanced to `after_ffn_hc` per layer.
    ///   - `state.kv_pos[i] += 1` for each layer encoded.
    ///   - Persistent KV cache buffer for each `layer_idx` is updated
    ///     at slot = `kv_pos[i]`.
    pub fn encode_first_half(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
    ) -> Result<FirstHalfOutputs> {
        self.encode_first_half_inner(x, model, state, l_split, true, None)
    }

    /// `encode_first_half`, but in **resident no-drain mode**: when the whole
    /// token stayed chained, the final HC residual stays GPU-resident — the
    /// resident buffer + its async cbs + deferred compressor finishes are
    /// returned via `resident_out` (and `cur_hc` is NOT read back to CPU). If
    /// any layer broke the chain (drained mid-loop), `resident_out` is left
    /// `None` and `state.cur_hc` holds the CPU result as usual. Used by the
    /// resident fused decode tail (`DS4_RESIDENT_TAIL`).
    pub fn encode_first_half_resident(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
        resident_out: &mut Option<ResidentChain>,
    ) -> Result<FirstHalfOutputs> {
        self.encode_first_half_inner(x, model, state, l_split, true, Some(resident_out))
    }

    /// Phase E M5.4.5.4 — attn-half-only variant of `encode_first_half`.
    ///
    /// Same as `encode_first_half` through step 5 (post-attn-half
    /// `attn_output_proj`); skips steps 6-10 (FFN-half: hc_collapse_norm
    /// + router + MoE + shared + hc_expand_add). `state.cur_hc` is
    /// advanced to `after_attn_hc` (NOT `after_ffn_hc`).
    ///
    /// Exists for unit tests that can't satisfy the production-shape
    /// MoE constraints (`n_experts == 256`, `d_embd % 256 == 0`,
    /// preloaded `QuantizedExpertWeights`) the full `encode_first_half`
    /// requires. Same return type for ergonomics — the `per_layer`
    /// outputs are populated from the attn-half step.
    pub fn encode_first_half_attn_only(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
    ) -> Result<FirstHalfOutputs> {
        self.encode_first_half_inner(x, model, state, l_split, false, None)
    }

    /// CHUNKED PREFILL (WIP, DS4_PREFILL_CHUNK) — sequential attention core for
    /// ONE raw (no-compressor) layer. Given the BATCHED attn-half outputs for K
    /// consecutive prompt tokens at positions [chunk_start..chunk_start+K), store
    /// each token's KV into the SWA circular ring (slot = pos % raw_cap) and run a
    /// SEQUENTIAL per-token raw flash. Sequential (not the K<=8 K-flash) so it
    /// scales to large chunks AND wraps the SWA window; each token attends its own
    /// [0..pos] window. The expensive matmuls (qkv before, output_proj+FFN after)
    /// are batched by the caller; this per-token loop is cheap at raw_cap=128
    /// (~7s total over the whole prefill). Returns [K, q_dim] rope-back'd heads.
    #[allow(clippy::too_many_arguments)]
    fn chunk_attn_core_raw(
        &self,
        scope: &mut crate::deferred::BatchScope<'_>,
        layer_idx: u32,
        p: &LayerParams,
        half: &crate::deferred::LayerAttnHalfOutsK,
        attn_sinks: &[f32],
        chunk_start: u32,
        k_positions: usize,
        raw_cap: u32,
    ) -> Result<crate::deferred::DeferredBuf> {
        let n_head = p.n_head as usize;
        let head_dim = p.head_dim as usize;
        let kv_row = p.n_lora_kv as usize;
        let q_dim = n_head * head_dim;
        // RoPE-forward all K q_heads at consecutive positions (one fused dispatch).
        scope.rope_tail_q_heads_in_place_k(
            &half.q_heads_k, n_head, head_dim, p, chunk_start, k_positions, false,
        )?;
        // Dummy comp ring (bound but unread when n_selected=0).
        let comp_ring = self.dispatcher.comp_ring_or_alloc(
            layer_idx as u32,
            comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
        );
        let heads_k = scope.alloc_f32(k_positions * q_dim);
        // DS4_CHUNK_SWA_KFLASH=1: lift the chunk_end<=raw_cap guard via an
        // SWA-windowed sliding workspace. A query at pos attends only the last
        // min(pos+1, raw_cap) raw rows + comp rows, so we tile the chunk into
        // TILES of `raw_cap` positions: per tile, gather the pre-tile SWA window
        // out of the persistent ring (BEFORE the tile's store overwrites those
        // slots), stack the in-tile raw rows from `half` directly, build one f16
        // workspace, run one ne01=tk flash with a per-query SWA causal mask.
        // n_comp=0 (raw layer). Mirrors the noidx SWA path.
        //
        // ★ KNOWN-INCOHERENT at chunk>raw_cap (2026-06-10 clean-boot bisect): the
        // tile-boundary hazard below (tile t+1's ring stores clobber slots tile t's
        // window gathers still read; Metal in-cb hazard tracking does not serialize
        // it at K>raw_cap) injects a NaN at the chunk tail → deterministic BOS. The
        // nosync batched raw path (below) supersedes it COHERENTLY for single-chunk
        // prefill, so SWA_KFLASH is OFF in production and excluded from every test
        // harness's default-on perf set. Only set DS4_CHUNK_SWA_KFLASH=1 to
        // reproduce the fault. Multi-chunk (chunk_start>0) is the only remaining
        // motivation, and it needs this hazard fixed before it can ship.
        let swa_kflash = std::env::var("DS4_CHUNK_SWA_KFLASH").ok().as_deref() == Some("1");
        if swa_kflash && chunk_start + k_positions as u32 > raw_cap {
            warn_swa_kflash_incoherent();
        }
        if swa_kflash {
            let w = raw_cap; // SWA window size == raw ring capacity
            const NEG: u16 = 0xFC00;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let sinks_db = scope.upload_f32(attn_sinks);
            let mut t = 0u32;
            let kp = k_positions as u32;
            while t < kp {
                let t0 = chunk_start + t;
                let tk = (kp - t).min(w);
                // pre-tile SWA window: positions [t0-pre .. t0-1] from the ring.
                let pre = t0.min(w.saturating_sub(1));
                let ring = self.dispatcher.kv_buffer_or_alloc(
                    layer_idx, raw_cap as usize * kv_row * std::mem::size_of::<f32>());
                // Pad the raw window to a multiple of 8 (the comp kernel's C=8 row tile);
                // padding rows are zeroed and masked off per query.
                let n_win = pre + tk;
                let n_raw8 = n_win.div_ceil(8) * 8;
                let raw_win = scope.alloc_swa_raw_window(n_raw8 as usize, head_dim);
                // 1. gather the pre-window aside (fp8 ring values) BEFORE the tile store
                //    overwrites those slots, store the tile, then gather the in-tile rows
                //    back from the ring (fp8) so every window row matches the per-token
                //    reference (which flashes against the fp8 ring).
                if pre > 0 {
                    scope.gather_ring_window(&ring, &raw_win, 0, raw_cap, t0 - pre, pre, head_dim as u32);
                }
                for j in 0..tk {
                    let pos = t0 + j;
                    let slot = pos % raw_cap;
                    let kv_j = scope.slice_out_f32(
                        &half.kv_normed_rotated_k, (t + j) as usize * kv_row, kv_row);
                    scope.kv_fp8_store_persistent(layer_idx, &kv_j, p, raw_cap, slot)?;
                }
                scope.gather_ring_window(&ring, &raw_win, pre, raw_cap, t0, tk, head_dim as u32);
                // 2. per-query SWA causal mask [tk, n_raw8] over the contiguous window
                //    (n_comp=0). Window row s maps to absolute position: s<pre →
                //    t0-pre+s; else t0+(s-pre). Padding rows s>=n_win stay NEG.
                let nr8 = n_raw8 as usize;
                let mut mask = vec![NEG; tk as usize * nr8];
                for r in 0..tk {
                    let pos = t0 + r;
                    let base = r as usize * nr8;
                    for s in 0..(n_win as usize) {
                        let pj = if (s as u32) < pre { t0 - pre + s as u32 }
                                 else { t0 + (s as u32 - pre) };
                        if pj <= pos && pj + w > pos { mask[base + s] = 0; }
                    }
                }
                // 3. sub-batched (K∈{8,4,2,1}) grow-only-scratch flash over the window.
                let mut g = 0u32;
                while g < tk {
                    let rem = tk - g;
                    let gk = if rem >= 8 { 8 } else if rem >= 4 { 4 } else if rem >= 2 { 2 } else { 1 };
                    let q_group = scope.slice_out_f32(
                        &half.q_heads_k, (t + g) as usize * q_dim, gk as usize * q_dim);
                    let mask_group =
                        mask[(g as usize * nr8)..((g + gk) as usize * nr8)].to_vec();
                    let heads_g = scope.flash_attn_k_mla_comp_masked(
                        &q_group, &raw_win, &comp_ring, 0, &mask_group,
                        n_head, head_dim, head_dim, n_raw8 as usize, gk as usize, scale, &sinks_db,
                    )?;
                    scope.copy_buf_into(&heads_g, heads_k.buffer(), (t + g) as usize * q_dim);
                    g += gk;
                }
                // SWA-tile boundary (same hazard class as the noidx core's
                // chunk_tile_event): tile t+1's ring stores overwrite slots tile
                // t's window gathers still read; in-cb hazard tracking is
                // unreliable at K>raw_cap. GPU-ordered split, no CPU drain.
                if chunk_start + kp > raw_cap && t + tk < kp && chunk_event_order_on() {
                    scope.event_split("raw_tile_event");
                }
                t += tk;
            }
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            return Ok(heads_k);
        }
        // NOTE: this raw SWA path shares the noidx core's chunk>raw_cap hazard
        // (tile t+1 ring stores overwriting rows tile t still reads). It only runs
        // under DS4_CHUNK_PHASE_A_BATCH (hash prefix); if that is ever defaulted
        // on, port the noidx core's tile-boundary drain here too.
        // GPU-resident scaffold removal (DS4_CHUNK_RAW_KFLASH=1): replace the K
        // per-position {store, flash, rope-back, copy} with 4 batched dispatches —
        // KV store_k, one shared raw workspace, ne01=K flash with per-query causal
        // mask, rope-back_k. Mirrors the noidx KFLASH lever with n_comp=0.
        // Same no-ring-wrap guard as KFLASH.
        // DS4_CHUNK_BLKFLASH: antirez-style all-raw prefill — SWA-128 mask + the
        // block-skipping non-vec flash (O(N·128)) instead of full-causal vec flash
        // (O(N²)). The lever to match antirez's 255 vs our 192 tok/s. UNVALIDATED.
        let blkflash = std::env::var("DS4_CHUNK_BLKFLASH").ok().as_deref() == Some("1")
            && chunk_start + k_positions as u32 <= raw_cap;
        let raw_kflash = std::env::var("DS4_CHUNK_RAW_KFLASH").ok().as_deref() == Some("1")
            && chunk_start + k_positions as u32 <= raw_cap;
        if raw_kflash || blkflash {
            let chunk_end = chunk_start + k_positions as u32;
            // blk flash needs 64-aligned n_total (ncpsg=64, has_kvpad=false).
            let align = if blkflash { 64 } else { 32 };
            let n_total_padded = chunk_end.div_ceil(align) * align;
            // BLKFLASH: direct f32→f16 cast (antirez-style, skips fp8-store+gather).
            // vec path: keep the fp8-ring + gather workspace.
            let workspace = if blkflash {
                scope.kv_f16_workspace_direct(
                    &half.kv_normed_rotated_k, k_positions, head_dim, n_total_padded,
                )?
            } else {
                let kv_cache = scope.kv_fp8_store_persistent_k(
                    layer_idx, &half.kv_normed_rotated_k, p, raw_cap, chunk_start, k_positions,
                )?;
                scope.build_chunk_kv_workspace(
                    &kv_cache, &comp_ring, chunk_end, 0, head_dim as u32, n_total_padded,
                )?
            };
            const NEG: u16 = 0xFC00;
            const N_SWA: u32 = 128; // model sliding-window (deepseek4 n_swa)
            let ntp = n_total_padded as usize;
            let mut mask = vec![NEG; k_positions * ntp];
            for r in 0..k_positions {
                let pos = chunk_start + r as u32;
                let base = r * ntp;
                for s in 0..(chunk_end as usize) {
                    // full-causal (vec path) vs SWA-128 window (blk path, faithful).
                    let attend = (s as u32) <= pos
                        && (!blkflash || pos - (s as u32) < N_SWA);
                    if attend { mask[base + s] = 0; }
                }
            }
            let heads_kf = if blkflash {
                scope.flash_attn_ext_blk(
                    p, &half.q_heads_k, &workspace, n_total_padded, &mask, attn_sinks, k_positions,
                )?
            } else {
                scope.flash_attn_decode_k(
                    p, &half.q_heads_k, &workspace, n_total_padded, &mask, attn_sinks, k_positions,
                )?
            };
            scope.copy_buf_into(&heads_kf, heads_k.buffer(), 0);
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            return Ok(heads_k);
        }
        // NOSYNC batched raw core: replace the K per-position (store + SWA flash +
        // rope-back) iterations with a wrap-aware batched KV store + ONE batched
        // mixed-attention flash (n_comp=0 → raw window only) + batched rope-back.
        // The prefill flash reads kv_normed_rotated_k (f32) directly, so the fp8 ring
        // is decode-only → store just the last raw_cap positions (≤2 dispatches).
        // Single chunk only (chunk_start==0). ratio passed as 1 (kernel needs >0; the
        // comp loop runs 0× so it's unused).
        let nosync = std::env::var("DS4_CHUNK_ATTN_NOSYNC").ok().as_deref() == Some("1")
            && chunk_start == 0;
        if nosync {
            // Same >raw_cap boundary as the idx core's nosync_kvwrap_event: the
            // wrap stores read half.* written earlier in this cb and overwrite
            // ALL ring slots. GPU-ordered split, no CPU drain.
            if k_positions as u32 > raw_cap && chunk_event_order_on() {
                scope.event_split("raw_kvwrap_event");
            }
            let kp_u = k_positions as u32;
            let store_n = kp_u.min(raw_cap);
            let first = kp_u - store_n;
            let s0 = first % raw_cap;
            let count1 = (raw_cap - s0).min(store_n);
            let slice1 = scope.slice_out_f32(&half.kv_normed_rotated_k, first as usize * kv_row, count1 as usize * kv_row);
            scope.kv_fp8_store_persistent_k(layer_idx, &slice1, p, raw_cap, s0, count1 as usize)?;
            let count2 = store_n - count1;
            if count2 > 0 {
                let slice2 = scope.slice_out_f32(&half.kv_normed_rotated_k, (first + count1) as usize * kv_row, count2 as usize * kv_row);
                scope.kv_fp8_store_persistent_k(layer_idx, &slice2, p, raw_cap, 0, count2 as usize)?;
            }
            let sinks_db = scope.upload_f32(attn_sinks);
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let dummy_sel = scope.alloc_i32(k_positions.max(1));
            let comp_ring_db = crate::deferred::DeferredBuf::from_external_buffer(comp_ring.clone(), 1);
            let heads = scope.encode_indexed_mixed_attention(
                &half.q_heads_k, &half.kv_normed_rotated_k, &comp_ring_db, &dummy_sel, &sinks_db,
                k_positions, n_head, head_dim, k_positions, 0, 0, 1,
                raw_cap, chunk_start, 0, k_positions as u32, scale,
            )?;
            scope.copy_buf_into(&heads, heads_k.buffer(), 0);
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            return Ok(heads_k);
        }
        for i in 0..k_positions {
            let pos = chunk_start + i as u32;
            let slot = pos % raw_cap;
            let n_raw = (pos + 1).min(raw_cap);
            let kv_i = scope.slice_out_f32(&half.kv_normed_rotated_k, i * kv_row, kv_row);
            scope.kv_fp8_store_persistent(layer_idx, &kv_i, p, raw_cap, slot)?;
            let q_i = scope.slice_out_f32(&half.q_heads_k, i * q_dim, q_dim);
            let heads_i = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                layer_idx, p, &q_i, n_raw, raw_cap, &comp_ring, &[], 0, attn_sinks,
            )?;
            scope.rope_tail_q_heads_in_place(&heads_i, n_head, head_dim, p, pos, true)?;
            scope.copy_buf_into(&heads_i, heads_k.buffer(), i * q_dim);
        }
        Ok(heads_k)
    }

    /// CHUNKED PREFILL (WIP) — sequential attention core for a COMPRESSOR layer
    /// WITHOUT an indexer (the ratio=128 odd layers; coff=1, no 2-window
    /// rotation). Per token: store KV (SWA), build the compressor emit GPU-side
    /// into the per-layer comp ring (compressor_encode_in_scope writes row
    /// n_comp), flash over [raw window | comp_ring[0..post_n_comp]], rope-back.
    /// The GPU comp ring + manually-tracked `n_comp` are self-sufficient across
    /// the chunk (the emit lands in the ring before the next token's flash — the
    /// intra-chunk dependency — with NO per-token drain). The CompressorScope
    /// handles are returned so the caller can run the (deferred) CPU-mirror
    /// finishes at chunk end. n_comp is bumped here.
    #[allow(clippy::too_many_arguments)]
    fn chunk_attn_core_comp_noidx(
        &self,
        scope: &mut crate::deferred::BatchScope<'_>,
        layer: &ComposedLayerWeights,
        layer_idx: usize,
        p: &LayerParams,
        half: &crate::deferred::LayerAttnHalfOutsK,
        state: &mut AttnStepState,
        chunk_start: u32,
        k_positions: usize,
        raw_cap: u32,
    ) -> Result<(crate::deferred::DeferredBuf, Vec<ChunkCompFinish>)> {
        let n_head = p.n_head as usize;
        let head_dim = p.head_dim as usize;
        let kv_row = p.n_lora_kv as usize;
        let n_embd = p.d_embd as usize;
        let q_dim = n_head * head_dim;
        let ratio = p.compress_ratio;
        scope.rope_tail_q_heads_in_place_k(
            &half.q_heads_k, n_head, head_dim, p, chunk_start, k_positions, false,
        )?;
        let comp_ring = self.dispatcher.comp_ring_or_alloc(
            layer_idx as u32,
            comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
        );
        let (kv_f16, gate_f16) = layer.attn.attn_compressor_f16();
        let main_comp = ds4_engine::attn_dispatch::CompressorInputs {
            w_kv: &layer.attn.attn_compressor_kv,
            w_gate: &layer.attn.attn_compressor_gate,
            w_kv_f16: kv_f16,
            w_gate_f16: gate_f16,
            w_ape: &layer.attn.attn_compressor_ape,
            w_norm: &layer.attn.attn_compressor_norm,
            head_dim: p.head_dim,
            compress_ratio: ratio,
        };
        let heads_k = scope.alloc_f32(k_positions * q_dim);
        let mut finishes: Vec<ChunkCompFinish> = Vec::with_capacity(k_positions);
        // Local emit counter for the in-chunk flash. state.n_comp is bumped by
        // the deferred finishes (post-flush) to stay coherent with the CPU
        // comp_kv_ring mirror; within the chunk the GPU comp ring (rows written
        // at `n_local`) is the source of truth the flash gathers.
        let mut n_local = state.n_comp[layer_idx];

        // DS4_CHUNK_SWA_KFLASH=1: lift the chunk_end<=raw_cap guard for the noidx
        // (ratio==128) compressor layer. Inside a 4096 chunk there ARE emits (one
        // every `ratio` positions) — split into TILES of `ratio` rows so within a
        // tile no emit occurs (the emit lands at the tile's last position). Per tile:
        // run the per-position compressor encodes (pool update + boundary emit into
        // the comp ring), build an SWA-windowed raw workspace + the pre-existing comp
        // rows, run ONE ne01=tk flash with a per-query SWA causal mask, then advance
        // n_local by the tile's emit count. Same flash-count/ratio reduction as the
        // (guarded) KFLASH path, but valid for chunk >> raw_cap.
        // ★ KNOWN-INCOHERENT at chunk>raw_cap — see chunk_attn_core_raw's SWA_KFLASH
        // note: the tile-boundary ring-store/window-gather hazard injects a tail NaN.
        // Off in production; the nosync batched path is the coherent default.
        let swa_kflash = std::env::var("DS4_CHUNK_SWA_KFLASH").ok().as_deref() == Some("1");
        if swa_kflash && ratio != 0 && chunk_start + k_positions as u32 > raw_cap {
            warn_swa_kflash_incoherent();
        }
        if swa_kflash && ratio != 0 {
            const NEG: u16 = 0xFC00;
            let w = raw_cap; // SWA window size == raw ring capacity
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let sinks_db = scope.upload_f32(&layer.attn.attn_sinks);
            let kp = k_positions as u32;

            // STAGE 1 (DS4_CHUNK_COMP_PREFILL): fuse the whole-chunk compressor
            // build into ONE dispatch. The fused kernel writes every emit row of
            // this chunk into the GPU comp ring up front (before any tile flash);
            // the tile loop then only stores raw KV + flashes (the per-position
            // store/pool/rms/rope/ring-copy ~ratio dispatches × n_emit vanish).
            // Only valid for the noidx (ratio != 4) layer this core serves and
            // when chunk_start is a ratio boundary (production: chunk%ratio==0,
            // raw_cap==ratio==DS4_N_SWA) so the per-emit pos_mod tiling matches
            // antirez ds4_gpu_compressor_prefill_tensor. f4-aligned only.
            let comp_prefill = std::env::var("DS4_CHUNK_COMP_PREFILL").ok().as_deref() == Some("1")
                && ratio != 4
                && head_dim % 4 == 0
                && n_embd % 4 == 0;
            if comp_prefill {
                // n_emit emits land at positions where (chunk_start+i+1)%ratio==0.
                let n_emit = (0..k_positions)
                    .filter(|&i| (chunk_start + i as u32 + 1) % ratio == 0)
                    .count() as u32;
                // Always project the whole chunk (the projections feed BOTH the
                // emit kernel and the trailing pool fill below).
                let main_w = head_dim; // coff==1 ⇒ width == head_dim
                let kv_k = crate::compressor::comp_proj_k(
                    scope, main_comp.w_kv, main_comp.w_kv_f16, &half.normed_k,
                    n_embd, main_w, k_positions)?;
                let sc_k = crate::compressor::comp_proj_k(
                    scope, main_comp.w_gate, main_comp.w_gate_f16, &half.normed_k,
                    n_embd, main_w, k_positions)?;
                if n_emit > 0 {
                    scope.compressor_prefill_noidx(
                        &kv_k, &sc_k, &layer.attn.attn_compressor_ape,
                        &layer.attn.attn_compressor_norm, &comp_ring,
                        head_dim as u32, ratio, p.n_rot, chunk_start, n_local, n_emit, p, p.rms_eps,
                    )?;
                    // Defer the CPU-mirror append (n_comp + comp_kv_ring) to after
                    // the chunk flush — the GPU ring rows aren't readable yet.
                    self.comp_prefill_finishes.borrow_mut().push(ChunkCompPrefillFinish {
                        comp_ring: comp_ring.clone(),
                        layer_idx,
                        comp_row0: n_local,
                        n_emit,
                        head_dim,
                    });
                }
                // Store the TRAILING partial group (positions after the last emit
                // boundary within the chunk) into the persistent compressor STATE
                // pool, so the next per-position decode (the feed token + the next
                // chunk's first group) pools across the chunk boundary correctly.
                // ONE dispatch. `cutoff` = the in-chunk index just past the last
                // emit (robust to a non-ratio-aligned chunk_start: find the last
                // local index whose absolute pos is an emit boundary).
                let last_emit_local = (0..k_positions)
                    .rev()
                    .find(|&i| (chunk_start + i as u32 + 1) % ratio == 0);
                let cutoff = match last_emit_local {
                    Some(i) => i as u32 + 1,
                    None => 0,
                };
                let rem = k_positions as u32 - cutoff;
                let do_fill = std::env::var("DS4_COMP_PREFILL_NO_FILL").ok().as_deref() != Some("1");
                if rem > 0 && do_fill {
                    let bytes = state.comp_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
                    let pk = self.dispatcher.compressor_state_kv_or_alloc(layer_idx as u32, bytes);
                    let ps = self.dispatcher.compressor_state_score_or_alloc(layer_idx as u32, bytes);
                    scope.compressor_pool_fill_noidx(
                        &kv_k, &sc_k, &layer.attn.attn_compressor_ape, &pk, &ps,
                        head_dim as u32, ratio, chunk_start, cutoff, rem,
                    )?;
                }
            }

            let mut t = 0u32;
            while t < kp {
                let t0 = chunk_start + t;
                // Tile ends at the next ratio boundary (so the only in-tile emit is the
                // last position), at chunk end, OR after `raw_cap` positions — whichever
                // comes first. The raw_cap cap keeps tk <= raw_cap so the in-tile SWA
                // window fits the ring's `gather_ring_window` (one wrap max). In
                // production ratio==raw_cap (==DS4_N_SWA), so the boundary IS the cap;
                // the min() only bites when raw_cap < ratio (e.g. isolation tests).
                let next_boundary = (t0 / ratio + 1) * ratio; // first pos > t0 with (pos)%ratio==0
                let tile_end = next_boundary.min(chunk_start + kp).min(t0 + w);
                let tk = tile_end - t0;
                // 1. compressor encodes for every tile position (pool update; emit lands
                //    in the ring at row n_local for the boundary position). n_emit_in
                //    counts emits BEFORE the tile flash (0); the boundary emit (if any)
                //    is the LAST position and is included for that query only.
                let n_comp_start = n_local;
                let mut emit_row_written = false;
                let pre = t0.min(w.saturating_sub(1));
                let ring = self.dispatcher.kv_buffer_or_alloc(
                    layer_idx as u32, raw_cap as usize * kv_row * std::mem::size_of::<f32>());
                // Pad the raw window to a multiple of 8 (comp kernel C=8 row tile);
                // padding rows are zeroed + masked off.
                let n_win = pre + tk;
                let n_raw8 = n_win.div_ceil(8) * 8;
                let raw_win = scope.alloc_swa_raw_window(n_raw8 as usize, head_dim);
                // Gather the pre-window (fp8 ring) BEFORE the tile's stores overwrite it.
                if pre > 0 {
                    scope.gather_ring_window(&ring, &raw_win, 0, raw_cap, t0 - pre, pre, head_dim as u32);
                }
                for j in 0..tk {
                    let pos = t0 + j;
                    let slot = pos % raw_cap;
                    let should_compress = (pos + 1) % ratio == 0;
                    let kv_j = scope.slice_out_f32(
                        &half.kv_normed_rotated_k, (t + j) as usize * kv_row, kv_row);
                    scope.kv_fp8_store_persistent(layer_idx as u32, &kv_j, p, raw_cap, slot)?;
                    // COMP_PREFILL: the fused kernel already built+wrote every emit
                    // row of this chunk into comp_ring above; skip the per-position
                    // encode + its deferred finish (the fused-prefill finish does the
                    // CPU-mirror append). Otherwise run the per-position encode.
                    if !comp_prefill {
                        let normed_j = scope.slice_out_f32(&half.normed_k, (t + j) as usize * n_embd, n_embd);
                        let (handle, emit_db) = crate::compressor::compressor_encode_in_scope(
                            scope, self.dispatcher, p, &main_comp, &normed_j,
                            state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                            Some((&comp_ring, n_local)),
                        )?;
                        finishes.push(ChunkCompFinish { handle, emit_db, layer_idx, is_indexer: false, pos });
                    }
                    if should_compress {
                        emit_row_written = true; // emit at ring row n_comp_start (n_local now)
                        n_local += 1;
                    }
                }
                // Gather the in-tile rows (fp8 ring) AFTER the stores.
                scope.gather_ring_window(&ring, &raw_win, pre, raw_cap, t0, tk, head_dim as u32);
                // 2. comp rows attended this tile = the pre-existing [0..n_comp_start) plus
                //    (if the tile's last position emitted) its OWN row n_comp_start. They
                //    are read straight from comp_ring by the kernel; gated per-query below.
                let n_comp_flash = n_comp_start + emit_row_written as u32;
                // 3. per-query SWA causal mask [tk, n_raw8 + n_comp_flash].
                //    raw cols [0..n_raw8): row s → pos pj; open iff pj<=pos<pj+w (pad = NEG).
                //    comp cols [n_raw8..n_raw8+n_comp_flash): rows [0..n_comp_start) open for
                //    all; the emit row (index n_comp_start) open ONLY for the emitter.
                let mask_row = n_raw8 as usize + n_comp_flash as usize;
                let mut mask = vec![NEG; tk as usize * mask_row];
                for r in 0..tk {
                    let pos = t0 + r;
                    let base = r as usize * mask_row;
                    for s in 0..(n_win as usize) {
                        let pj = if (s as u32) < pre { t0 - pre + s as u32 }
                                 else { t0 + (s as u32 - pre) };
                        if pj <= pos && pj + w > pos { mask[base + s] = 0; }
                    }
                    for c in 0..(n_comp_start as usize) {
                        mask[base + n_raw8 as usize + c] = 0;
                    }
                    if emit_row_written && r == tk - 1 {
                        mask[base + n_raw8 as usize + n_comp_start as usize] = 0;
                    }
                }
                // 4. ONE flash over the tile window. PRODUCTION (raw_cap == DS4_N_SWA ==
                //    ratio): use the grow-only-scratch comp kernel (no per-tile ~536 MB
                //    tmp), sub-batched into K∈{8,4,2,1}. The test-only "inconsistent"
                //    regime raw_cap < ratio (the SWA window is SMALLER than the compress
                //    period, which never happens in deployment — antirez ties both to
                //    DS4_N_SWA) drives a degenerate tiling (tk capped by the tiny window
                //    while emits stay ratio-apart) that perturbs the comp kernel; there
                //    we fall back to the proven f16-workspace flash_attn_decode_k (the
                //    536 MB tmp is irrelevant at the small raw_cap of those tests).
                // BISECT (DS4_CHUNK_NOIDX_F16_FALLBACK=1): force the f16-workspace
                // fallback (build_chunk_kv_workspace + flash_attn_decode_k) — the SAME
                // structure as the per-token build_extended_kv→flash_attn_decode
                // reference — instead of the production f32 comp kernel
                // (flash_attn_k_mla_comp_masked). Tests whether the noidx core's f32
                // comp-kernel is the @3000 chunk-divergence source.
                let use_comp_kernel = raw_cap >= ratio
                    && std::env::var("DS4_CHUNK_NOIDX_F16_FALLBACK").ok().as_deref() != Some("1");
                if use_comp_kernel {
                    let mut g = 0u32;
                    while g < tk {
                        let rem = tk - g;
                        let gk = if rem >= 8 { 8 } else if rem >= 4 { 4 } else if rem >= 2 { 2 } else { 1 };
                        let q_group = scope.slice_out_f32(
                            &half.q_heads_k, (t + g) as usize * q_dim, gk as usize * q_dim);
                        let mask_group =
                            mask[(g as usize * mask_row)..((g + gk) as usize * mask_row)].to_vec();
                        let heads_g = scope.flash_attn_k_mla_comp_masked(
                            &q_group, &raw_win, &comp_ring, n_comp_flash, &mask_group,
                            n_head, head_dim, head_dim, n_raw8 as usize, gk as usize, scale, &sinks_db,
                        )?;
                        scope.copy_buf_into(&heads_g, heads_k.buffer(), (t + g) as usize * q_dim);
                        g += gk;
                    }
                } else {
                    // Fallback: build the f16 workspace [raw_win(n_win) | comp(n_comp_flash)]
                    // 32-padded and run the masked f16-vec flash (arbitrary K in one
                    // dispatch). Mask must be rebuilt for the n_win (un-padded) raw layout.
                    let n_total = n_win + n_comp_flash;
                    let n_total_padded = n_total.div_ceil(32) * 32;
                    let workspace = scope.build_chunk_kv_workspace(
                        &raw_win, &comp_ring, n_win, n_comp_flash, head_dim as u32, n_total_padded,
                    )?;
                    let ntp = n_total_padded as usize;
                    let mut wmask = vec![NEG; tk as usize * ntp];
                    for r in 0..tk {
                        let pos = t0 + r;
                        let base = r as usize * ntp;
                        for s in 0..(n_win as usize) {
                            let pj = if (s as u32) < pre { t0 - pre + s as u32 }
                                     else { t0 + (s as u32 - pre) };
                            if pj <= pos && pj + w > pos { wmask[base + s] = 0; }
                        }
                        for c in 0..(n_comp_start as usize) {
                            wmask[base + n_win as usize + c] = 0;
                        }
                        if emit_row_written && r == tk - 1 {
                            wmask[base + n_win as usize + n_comp_start as usize] = 0;
                        }
                    }
                    let q_tile = scope.slice_out_f32(
                        &half.q_heads_k, t as usize * q_dim, tk as usize * q_dim);
                    let heads_tf = scope.flash_attn_decode_k(
                        p, &q_tile, &workspace, n_total_padded, &wmask, &layer.attn.attn_sinks, tk as usize,
                    )?;
                    scope.copy_buf_into(&heads_tf, heads_k.buffer(), t as usize * q_dim);
                }
                // SWA-tile boundary drain (DEFAULT when the chunk crosses raw_cap):
                // when chunk_start+K > raw_cap, tile t+1's pool stores + ring stores
                // OVERWRITE the rows tile t's emit pool kernel / window gathers still
                // read. In-cb hazards do NOT serialize this reliably at K>128 —
                // chunk=256 was a DETERMINISTIC BOS word-salad (L7 comp-ring row0
                // -0.1159→-0.1049). A commit+wait at the tile boundary restores the
                // chunk=128 baseline bit-exactly (per-tile commit_keep_open was NOT
                // enough at K=192). Cost: (K/raw_cap - 1) drains/layer — zero when the
                // chunk fits the ring. DS4_CHUNK_TILE_COMMIT=0 disables (debug),
                // "1" = commit no-wait (debug), "wait"/default = drain.
                let crosses_ring = chunk_start + kp > raw_cap;
                match std::env::var("DS4_CHUNK_TILE_COMMIT").ok().as_deref() {
                    Some("0") => {}
                    Some("1") => scope.commit_keep_open(),
                    _ => {
                        if crosses_ring && t + tk < kp {
                            if chunk_event_order_on() {
                                // MTLEvent split: same boundary, GPU-ordered
                                // without idling the CPU/GPU on a drain.
                                scope.event_split("chunk_tile_event");
                            } else {
                                scope.commit_wait_stage("chunk_tile_wait");
                            }
                        }
                    }
                }
                t += tk;
            }
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            // DS4_CHUNK_HEADS_DUMP=<layer> diagnostic (same as the idx core).
            if let Ok(v) = std::env::var("DS4_CHUNK_HEADS_DUMP") {
                if v.parse::<usize>().ok() == Some(layer_idx)
                    && (chunk_start as usize + k_positions) <= 4096
                {
                    let dbg = self.dispatcher.kv_buffer_or_alloc(
                        layer_idx as u32 + 4_000_000,
                        4096 * q_dim * std::mem::size_of::<f32>());
                    scope.copy_buf_into(&heads_k, &dbg, chunk_start as usize * q_dim);
                    self.heads_dump.set(Some((
                        layer_idx, chunk_start as usize + k_positions, q_dim,
                    )));
                }
            }
            return Ok((heads_k, finishes));
        }

        // DS4_CHUNK_BATCH_FLASH: batched-attention path — replace the K sequential
        // per-position flash dispatches with sub-batched K-query flash_attn_k_mla_comp
        // (groups of {8,4,2,1}, the kernel's K-limit). Only the NO-WITHIN-CHUNK-EMIT
        // case (true for ratio>=K, e.g. ratio=128 K<=64): all K queries attend to the
        // same fixed comp set [0..n_comp_start] (comp_avail all 0), matching the
        // per-position core's dense comp_sel. NOTE: flash_attn_k_mla_comp is a
        // DIFFERENT kernel than the per-position gpuring flash, so output is
        // numerically-equivalent but NOT byte-identical (fp32 simdgroup reduction
        // order) — validate by closeness, not byte-equality. Emit-bearing chunks fall
        // through to the per-position loop (correctness).
        let batch_flash = std::env::var("DS4_CHUNK_BATCH_FLASH").ok().as_deref() == Some("1");
        let any_emit = (0..k_positions)
            .any(|i| ratio != 0 && (chunk_start + i as u32 + 1) % ratio == 0);
        if batch_flash && !any_emit {
            let n_comp_start = n_local;
            // 1. store all K KV + run all K compressor encodes (pool/ring update; no
            //    emit since !any_emit). Capture the persistent KV cache buffer.
            let mut kv_cache_db: Option<crate::deferred::DeferredBuf> = None;
            for i in 0..k_positions {
                let pos = chunk_start + i as u32;
                let slot = pos % raw_cap;
                let kv_i = scope.slice_out_f32(&half.kv_normed_rotated_k, i * kv_row, kv_row);
                kv_cache_db = Some(scope.kv_fp8_store_persistent(layer_idx as u32, &kv_i, p, raw_cap, slot)?);
                let normed_i = scope.slice_out_f32(&half.normed_k, i * n_embd, n_embd);
                let (handle, emit_db) = crate::compressor::compressor_encode_in_scope(
                    scope, self.dispatcher, p, &main_comp, &normed_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, n_local)),
                )?;
                finishes.push(ChunkCompFinish { handle, emit_db, layer_idx, is_indexer: false, pos });
            }
            let kv_cache = kv_cache_db.expect("kv_cache (k>=1)");
            // 2. sub-batched K-query flash over [raw causal window | comp[0..n_comp_start]].
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let comp_avail = vec![0u32; n_comp_start as usize]; // pre-existing rows: all queries
            let sinks_db = scope.upload_f32(&layer.attn.attn_sinks);
            let mut g = 0usize;
            while g < k_positions {
                let rem = k_positions - g;
                let gk = if rem >= 8 { 8 } else if rem >= 4 { 4 } else if rem >= 2 { 2 } else { 1 };
                let q_group = scope.slice_out_f32(&half.q_heads_k, g * q_dim, gk * q_dim);
                let heads_g = scope.flash_attn_k_mla_comp(
                    &q_group, &kv_cache, &comp_ring, n_comp_start, &comp_avail,
                    n_head, head_dim, head_dim, raw_cap as usize, gk, scale,
                    chunk_start + g as u32, &sinks_db,
                )?;
                scope.copy_buf_into(&heads_g, heads_k.buffer(), g * q_dim);
                g += gk;
            }
            // 3. rope-back all K outputs (mirrors per-position rope_tail back).
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            return Ok((heads_k, finishes));
        }

        // LEVER A (DS4_CHUNK_KFLASH): replace the K sequential per-position flash
        // dispatches with a SINGLE ne01=K flash over a SHARED f16 workspace gathered
        // ONCE (vs the per-position O(n²) re-gather). Same no-within-chunk-emit guard
        // as BATCH_FLASH (ratio>=K). Uses the lighter f16 vec kernel
        // (flash_attn_decode_k_metal, byte-identical to the per-position flash per the
        // isolation test) over [raw[0..chunk_end] | comp[0..n_comp_start]] + a
        // per-query causal mask. fp32-CLOSE (not byte-identical): the comp rows sit at
        // a fixed workspace offset → a different simdgroup reduction order than the
        // per-position gpuring flash. Gate behind closeness, not byte-equality.
        let kflash = std::env::var("DS4_CHUNK_KFLASH").ok().as_deref() == Some("1");
        if kflash && !any_emit && chunk_start + k_positions as u32 <= raw_cap {
            let n_comp_start = n_local;
            // 1. store all K KV (capture the persistent cache) + run all K compressor
            //    encodes (pool/ring update; no emit since !any_emit).
            let mut kv_cache_db: Option<crate::deferred::DeferredBuf> = None;
            for i in 0..k_positions {
                let pos = chunk_start + i as u32;
                let slot = pos % raw_cap;
                let kv_i = scope.slice_out_f32(&half.kv_normed_rotated_k, i * kv_row, kv_row);
                kv_cache_db = Some(scope.kv_fp8_store_persistent(layer_idx as u32, &kv_i, p, raw_cap, slot)?);
                let normed_i = scope.slice_out_f32(&half.normed_k, i * n_embd, n_embd);
                let (handle, emit_db) = crate::compressor::compressor_encode_in_scope(
                    scope, self.dispatcher, p, &main_comp, &normed_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, n_local)),
                )?;
                finishes.push(ChunkCompFinish { handle, emit_db, layer_idx, is_indexer: false, pos });
            }
            let kv_cache = kv_cache_db.expect("kv_cache (k>=1)");
            // 2. gather the SHARED workspace ONCE: raw[0..chunk_end] then comp[0..n_comp_start].
            let chunk_end = chunk_start + k_positions as u32;
            let n_total = chunk_end + n_comp_start;
            let n_total_padded = n_total.div_ceil(32) * 32;
            let workspace = scope.build_chunk_kv_workspace(
                &kv_cache, &comp_ring, chunk_end, n_comp_start, head_dim as u32, n_total_padded,
            )?;
            // 3. per-query causal mask [K, n_total_padded]: query r (pos=chunk_start+r)
            //    sees raw row s iff s<=pos; all comp rows (pre-existing, comp_avail=0);
            //    padding rows masked.
            const NEG: u16 = 0xFC00;
            let ntp = n_total_padded as usize;
            let mut mask = vec![NEG; k_positions * ntp];
            for r in 0..k_positions {
                let pos = chunk_start + r as u32;
                let base = r * ntp;
                for s in 0..(chunk_end as usize) {
                    if (s as u32) <= pos { mask[base + s] = 0; }
                }
                for s in (chunk_end as usize)..(n_total as usize) {
                    mask[base + s] = 0;
                }
            }
            // 4. single ne01=K flash over the shared workspace.
            let heads_kf = scope.flash_attn_decode_k(
                p, &half.q_heads_k, &workspace, n_total_padded, &mask,
                &layer.attn.attn_sinks, k_positions,
            )?;
            scope.copy_buf_into(&heads_kf, heads_k.buffer(), 0);
            // 5. rope-back all K outputs.
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            return Ok((heads_k, finishes));
        }

        // NOSYNC batched noidx core (ratio==128): wrap-aware batched KV store + fused
        // whole-chunk compressor (compressor_prefill_noidx → comp ring, + pool_fill for
        // decode continuation) + ONE batched mixed-attention flash attending ALL comp
        // rows (sel=identity per token, the kernel's causal `visible=(qpos+1)/ratio`
        // break gives each token comp[0..visible]). Replaces the K per-position
        // (store + compressor encode + flash) iterations. Single chunk only. The
        // prefill flash reads kv_normed_rotated_k (f32), so the fp8 ring is decode-only
        // → store just the last raw_cap positions.
        let nosync = std::env::var("DS4_CHUNK_ATTN_NOSYNC").ok().as_deref() == Some("1")
            && chunk_start == 0
            && ratio != 0 && ratio != 4
            && head_dim % 4 == 0 && n_embd % 4 == 0;
        if nosync {
            let kp_u = k_positions as u32;
            // 1. wrap-aware batched KV store (last raw_cap positions → ≤2 dispatches).
            let store_n = kp_u.min(raw_cap);
            let first_s = kp_u - store_n;
            let s0 = first_s % raw_cap;
            let count1 = (raw_cap - s0).min(store_n);
            let slice1 = scope.slice_out_f32(&half.kv_normed_rotated_k, first_s as usize * kv_row, count1 as usize * kv_row);
            scope.kv_fp8_store_persistent_k(layer_idx as u32, &slice1, p, raw_cap, s0, count1 as usize)?;
            let count2 = store_n - count1;
            if count2 > 0 {
                let slice2 = scope.slice_out_f32(&half.kv_normed_rotated_k, (first_s + count1) as usize * kv_row, count2 as usize * kv_row);
                scope.kv_fp8_store_persistent_k(layer_idx as u32, &slice2, p, raw_cap, 0, count2 as usize)?;
            }
            // 2. project + fused compressor → comp ring, + trailing-partial pool fill.
            let main_w = head_dim; // coff==1
            let kv_k = crate::compressor::comp_proj_k(scope, main_comp.w_kv, main_comp.w_kv_f16, &half.normed_k, n_embd, main_w, k_positions)?;
            let sc_k = crate::compressor::comp_proj_k(scope, main_comp.w_gate, main_comp.w_gate_f16, &half.normed_k, n_embd, main_w, k_positions)?;
            let n_emit = (0..k_positions).filter(|&i| (chunk_start + i as u32 + 1) % ratio == 0).count() as u32;
            if n_emit > 0 {
                scope.compressor_prefill_noidx(
                    &kv_k, &sc_k, main_comp.w_ape, main_comp.w_norm, &comp_ring,
                    head_dim as u32, ratio, p.n_rot, chunk_start, n_local, n_emit, p, p.rms_eps,
                )?;
            }
            let cutoff = (0..k_positions).rev().find(|&i| (chunk_start + i as u32 + 1) % ratio == 0).map(|i| i as u32 + 1).unwrap_or(0);
            let rem = kp_u - cutoff;
            let m_bytes = state.comp_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
            let pk = self.dispatcher.compressor_state_kv_or_alloc(layer_idx as u32, m_bytes);
            let ps = self.dispatcher.compressor_state_score_or_alloc(layer_idx as u32, m_bytes);
            if rem > 0 {
                scope.compressor_pool_fill_noidx(&kv_k, &sc_k, main_comp.w_ape, &pk, &ps, head_dim as u32, ratio, chunk_start, cutoff, rem)?;
            }
            // 3. batched mixed-attention over [raw SWA window | all comp rows].
            let n_comp_total = n_local + n_emit;
            let sinks_db = scope.upload_f32(&layer.attn.attn_sinks);
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads = if n_comp_total > 0 {
                let nc = n_comp_total as usize;
                let row: Vec<i32> = (0..nc as i32).collect();
                let mut sel = Vec::with_capacity(k_positions * nc);
                for _ in 0..k_positions { sel.extend_from_slice(&row); }
                let sel_db = scope.upload_i32(&sel);
                let comp_ring_db = crate::deferred::DeferredBuf::from_external_buffer(comp_ring.clone(), nc * head_dim);
                scope.encode_indexed_mixed_attention(
                    &half.q_heads_k, &half.kv_normed_rotated_k, &comp_ring_db, &sel_db, &sinks_db,
                    k_positions, n_head, head_dim, k_positions, nc, nc, ratio,
                    raw_cap, chunk_start, 0, kp_u, scale,
                )?
            } else {
                let dummy_sel = scope.alloc_i32(k_positions.max(1));
                let comp_ring_db = crate::deferred::DeferredBuf::from_external_buffer(comp_ring.clone(), 1);
                scope.encode_indexed_mixed_attention(
                    &half.q_heads_k, &half.kv_normed_rotated_k, &comp_ring_db, &dummy_sel, &sinks_db,
                    k_positions, n_head, head_dim, k_positions, 0, 0, 1,
                    raw_cap, chunk_start, 0, kp_u, scale,
                )?
            };
            scope.copy_buf_into(&heads, heads_k.buffer(), 0);
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            // 4. resync CPU mirrors ONCE (comp ring rows + pool + count) for decode.
            // PHASE 1 (DS4_CHUNK_DEFER_RESYNC): the pk/ps pools + comp_ring are
            // per-layer-persistent and nothing later in the chunk reads the CPU
            // mirror, so DEFER the read past the terminal wait (removes the per-layer
            // drain). The count bump stays immediate (harmless here — noidx returns;
            // the deferred append uses comp_row0=n_local captured pre-bump).
            if defer_chunk_resync() {
                self.chunk_resyncs.borrow_mut().push(ChunkResync {
                    layer_idx, main_pk: pk.clone(), main_ps: ps.clone(),
                    idx_pk: None, idx_ps: None, comp_ring: comp_ring.clone(),
                    idx_ring: None, comp_row0: n_local, n_emit, head_dim, idx_hd: 0,
                });
                state.n_comp[layer_idx] += n_emit;
                return Ok((heads_k, Vec::new()));
            }
            scope.commit_wait_stage("nosync_noidx_resync");
            unsafe {
                std::ptr::copy_nonoverlapping(pk.contents() as *const f32,
                    state.comp_state_kv[layer_idx].as_mut_ptr(), state.comp_state_kv[layer_idx].len());
                std::ptr::copy_nonoverlapping(ps.contents() as *const f32,
                    state.comp_state_score[layer_idx].as_mut_ptr(), state.comp_state_score[layer_idx].len());
                if n_emit > 0 {
                    let src = (comp_ring.contents() as *const f32).add(n_local as usize * head_dim);
                    state.comp_kv_ring[layer_idx].extend_from_slice(
                        std::slice::from_raw_parts(src, n_emit as usize * head_dim));
                }
            }
            state.n_comp[layer_idx] += n_emit;
            return Ok((heads_k, Vec::new()));
        }

        // INCREMENT 1 (see the idx core): batch the independent per-position ops —
        // KV store (512→1) + compressor projection (byte-identical) — when the chunk
        // doesn't wrap the raw ring. coff=1 for ratio!=4 → width = head_dim.
        let batched = std::env::var("DS4_CHUNK_SEQ").ok().as_deref() != Some("1")
            && chunk_start + k_positions as u32 <= raw_cap;
        let main_w = head_dim;
        let (kv_main_k, sc_main_k) = if batched {
            scope.kv_fp8_store_persistent_k(layer_idx as u32, &half.kv_normed_rotated_k, p, raw_cap, chunk_start, k_positions)?;
            (
                Some(crate::compressor::comp_proj_k(scope, main_comp.w_kv, main_comp.w_kv_f16, &half.normed_k, n_embd, main_w, k_positions)?),
                Some(crate::compressor::comp_proj_k(scope, main_comp.w_gate, main_comp.w_gate_f16, &half.normed_k, n_embd, main_w, k_positions)?),
            )
        } else {
            (None, None)
        };

        for i in 0..k_positions {
            let pos = chunk_start + i as u32;
            let slot = pos % raw_cap;
            let n_raw = (pos + 1).min(raw_cap);
            let should_compress = ratio != 0 && (pos + 1) % ratio == 0;
            if !batched {
                let kv_i = scope.slice_out_f32(&half.kv_normed_rotated_k, i * kv_row, kv_row);
                scope.kv_fp8_store_persistent(layer_idx as u32, &kv_i, p, raw_cap, slot)?;
            }
            let normed_i = scope.slice_out_f32(&half.normed_k, i * n_embd, n_embd);
            let (handle, emit_db) = if let (Some(kvm), Some(scm)) = (&kv_main_k, &sc_main_k) {
                let kv_main_i = scope.slice_out_f32(kvm, i * main_w, main_w);
                let sc_main_i = scope.slice_out_f32(scm, i * main_w, main_w);
                crate::compressor::compressor_encode_with_proj(
                    scope, self.dispatcher, p, &main_comp, &kv_main_i, &sc_main_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, n_local)),
                )?
            } else {
                crate::compressor::compressor_encode_in_scope(
                    scope, self.dispatcher, p, &main_comp, &normed_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, n_local)),
                )?
            };
            finishes.push(ChunkCompFinish { handle, emit_db, layer_idx, is_indexer: false, pos });
            let post_n_comp = n_local + should_compress as u32;
            let q_i = scope.slice_out_f32(&half.q_heads_k, i * q_dim, q_dim);
            let comp_sel: Vec<u32> = (0..post_n_comp).collect();
            let heads_i = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                layer_idx as u32, p, &q_i, n_raw, raw_cap, &comp_ring, &comp_sel,
                post_n_comp, &layer.attn.attn_sinks,
            )?;
            scope.copy_buf_into(&heads_i, heads_k.buffer(), i * q_dim);
            n_local = post_n_comp;
        }
        // INCREMENT 2: batched rope-back over all K (per-position-independent).
        scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
        // CPU-mirror finishes (state mirror resync + comp_kv_ring append) are
        // returned for the caller to run AFTER the chunk's GPU work flushes —
        // deferred like the bridge's `finish_chain`/`run_chain_finishes`. The
        // GPU comp ring already carries the emits the flashes consumed, so the
        // deferral is safe for ratio≠4 (coff=1, no 2-window rotation: the
        // single-window pool is overwritten each store, and each emit row was
        // captured per-token in `emit_db`). For ratio==4 the per-quad rotation
        // forces an inline flush+finish — handled by the (separate) indexed core.
        Ok((heads_k, finishes))
    }

    /// STAGE 1 (DS4_CHUNK_COMP_PREFILL) post-flush finish: the fused
    /// `compressor_prefill_noidx` kernel already wrote the emit rows into the
    /// GPU comp ring. Here — AFTER the scope's terminal `wait_all_and_drop` —
    /// read each emit row back and append it to the CPU `comp_kv_ring` mirror,
    /// bumping `n_comp`. This is the noidx ratio!=4 analogue of the per-position
    /// deferred finishes: NO pool resync / rotation (the transient pool the
    /// fused kernel never touched stays untouched), only the CPU-mirror coherence
    /// the post-chunk decode + comp-dump rely on.
    fn apply_comp_prefill_finishes(&self, state: &mut AttnStepState) {
        let finishes = self.comp_prefill_finishes.borrow_mut().drain(..).collect::<Vec<_>>();
        for f in finishes {
            // The CPU mirror must already be at comp_row0 (the per-chunk path
            // appends in chunk order; assert to catch any mis-sequencing — the
            // exact bug class STAGE 1 is guarding against).
            debug_assert_eq!(
                state.n_comp[f.layer_idx], f.comp_row0,
                "comp-prefill finish out of order: n_comp={} != comp_row0={} (layer {})",
                state.n_comp[f.layer_idx], f.comp_row0, f.layer_idx,
            );
            let hd = f.head_dim;
            for e in 0..f.n_emit as usize {
                let row = (f.comp_row0 as usize + e) * hd;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        (f.comp_ring.contents() as *const f32).add(row),
                        hd,
                    )
                };
                state.comp_kv_ring[f.layer_idx].extend_from_slice(slice);
            }
            state.n_comp[f.layer_idx] += f.n_emit;
        }
    }

    /// PHASE 1 (DS4_CHUNK_DEFER_RESYNC): replay the per-layer nosync compressor
    /// CPU-mirror resyncs deferred from N1a/IDX-N1, AFTER the terminal
    /// `wait_all_and_drop` (so the per-layer-persistent pools + rings are GPU-complete
    /// and host-readable). The count bumps (n_comp/n_index_comp) and the gpu_rows flag
    /// already happened immediately during the chunk; here we only fill the CPU mirrors
    /// that decode-after-prefill reads. `comp_row0` was captured pre-bump, so each
    /// append lands at the right ring offset (nothing else touched this layer's mirror
    /// between the chunk and here). No-op when not deferring (the vec is empty).
    fn apply_chunk_resyncs(&self, state: &mut AttnStepState) {
        let resyncs = self.chunk_resyncs.borrow_mut().drain(..).collect::<Vec<_>>();
        for r in resyncs {
            let li = r.layer_idx;
            unsafe {
                std::ptr::copy_nonoverlapping(r.main_pk.contents() as *const f32,
                    state.comp_state_kv[li].as_mut_ptr(), state.comp_state_kv[li].len());
                std::ptr::copy_nonoverlapping(r.main_ps.contents() as *const f32,
                    state.comp_state_score[li].as_mut_ptr(), state.comp_state_score[li].len());
                if let (Some(ipk), Some(ips)) = (r.idx_pk.as_ref(), r.idx_ps.as_ref()) {
                    std::ptr::copy_nonoverlapping(ipk.contents() as *const f32,
                        state.index_state_kv[li].as_mut_ptr(), state.index_state_kv[li].len());
                    std::ptr::copy_nonoverlapping(ips.contents() as *const f32,
                        state.index_state_score[li].as_mut_ptr(), state.index_state_score[li].len());
                }
                if r.n_emit > 0 {
                    let m_src = (r.comp_ring.contents() as *const f32)
                        .add(r.comp_row0 as usize * r.head_dim);
                    state.comp_kv_ring[li].extend_from_slice(
                        std::slice::from_raw_parts(m_src, r.n_emit as usize * r.head_dim));
                    if let Some(ir) = r.idx_ring.as_ref() {
                        let i_src = (ir.contents() as *const f32)
                            .add(r.comp_row0 as usize * r.idx_hd);
                        state.index_comp_kv_ring[li].extend_from_slice(
                            std::slice::from_raw_parts(i_src, r.n_emit as usize * r.idx_hd));
                    }
                }
            }
        }
    }

    /// Deferred CPU-mirror finish for a chunk compressor emit (mirrors the
    /// bridge's `run_chain_finishes`): after the chunk GPU work has flushed,
    /// read back the GPU-finished `emit_db` as the pooled row, run
    /// `compressor_finish_in_scope` (state-mirror resync + ratio==4 rotation),
    /// and append the emit to the per-layer comp ring.
    fn run_chunk_compressor_finishes(
        &self,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        finishes: Vec<ChunkCompFinish>,
    ) {
        for f in finishes {
            let layer = &model.layers[f.layer_idx];
            let p = &layer.attn.params;
            let pooled = match &f.emit_db {
                Some(e) => unsafe {
                    std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                }
                .to_vec(),
                None => Vec::new(),
            };
            if f.is_indexer {
                let (kv_f16, gate_f16) = layer.attn.indexer_compressor_f16();
                let comp = ds4_engine::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.indexer_compressor_kv,
                    w_gate: &layer.attn.indexer_compressor_gate,
                    w_kv_f16: kv_f16,
                    w_gate_f16: gate_f16,
                    w_ape: &layer.attn.indexer_compressor_ape,
                    w_norm: &layer.attn.indexer_compressor_norm,
                    head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                    compress_ratio: p.compress_ratio,
                };
                if let Some(row) = crate::compressor::compressor_finish_in_scope(
                    f.handle, self.dispatcher, p, &comp, pooled,
                    &mut state.index_state_kv[f.layer_idx],
                    &mut state.index_state_score[f.layer_idx], f.pos,
                ) {
                    state.index_comp_kv_ring[f.layer_idx].extend_from_slice(&row);
                    state.n_index_comp[f.layer_idx] += 1;
                }
            } else {
                let (kv_f16, gate_f16) = layer.attn.attn_compressor_f16();
                let comp = ds4_engine::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.attn_compressor_kv,
                    w_gate: &layer.attn.attn_compressor_gate,
                    w_kv_f16: kv_f16,
                    w_gate_f16: gate_f16,
                    w_ape: &layer.attn.attn_compressor_ape,
                    w_norm: &layer.attn.attn_compressor_norm,
                    head_dim: p.head_dim,
                    compress_ratio: p.compress_ratio,
                };
                if let Some(row) = crate::compressor::compressor_finish_in_scope(
                    f.handle, self.dispatcher, p, &comp, pooled,
                    &mut state.comp_state_kv[f.layer_idx],
                    &mut state.comp_state_score[f.layer_idx], f.pos,
                ) {
                    // Canonical append+bump (matches `run_chain_finishes`): the
                    // GPU comp ring already fed the in-chunk flash; this keeps
                    // the CPU mirror + n_comp coherent for post-chunk decode.
                    state.comp_kv_ring[f.layer_idx].extend_from_slice(&row);
                    state.n_comp[f.layer_idx] += 1;
                }
            }
        }
    }

    /// CHUNKED PREFILL (WIP) — sequential attention core for a ratio==4
    /// COMPRESSOR + INDEXER layer (L0,1,2,4,6,…: the even/hash-routed layers).
    /// Mirrors the production resident decode path (single_buffer_encoder.rs
    /// ~2066-2170) per token, fed the K-batched attn-half (`half.*_k`):
    ///
    ///   per token i (pos = chunk_start + i):
    ///     1. store KV (SWA slot = pos % raw_cap)
    ///     2. main compressor encode → comp ring at row n_comp (in-scope)
    ///     3. indexer compressor encode (is_indexer, no ring)
    ///     4. flash: if the indexer has engaged (post_n_index > top_k) run the
    ///        GPU-resident selection (sync idx ring ← CPU source of truth, append
    ///        this token's emit, encode_indexer_scores → topk → GPU sel) and
    ///        flash with `sel`; else attend ALL comp rows (0..post_n_comp).
    ///     5. rope-back the flash output, copy into heads_k.
    ///     6. ON EMIT ((pos+1)%ratio==0): commit_wait the scope, then run
    ///        compressor_finish_in_scope for BOTH main + indexer — the ratio==4
    ///        2-window POOL ROTATION (sync_pool_to_mirror → shift → write_pool_all)
    ///        MUST land before the next quad's stores. This serializes a
    ///        flush+finish every `ratio` tokens (cheap vs the batched matmuls).
    ///        The non-emit tokens' handles are dropped (the emit's
    ///        sync_pool_to_mirror captures the whole quad's pool).
    ///
    /// Bit-equivalence target: chunk=1 == per-token resident decode. Takes
    /// `&mut scope` because the per-quad rotation needs `commit_wait_stage`.
    #[allow(clippy::too_many_arguments)]
    fn chunk_attn_core_comp_idx(
        &self,
        scope: &mut crate::deferred::BatchScope<'_>,
        layer: &ComposedLayerWeights,
        layer_idx: usize,
        p: &LayerParams,
        half: &crate::deferred::LayerAttnHalfOutsK,
        state: &mut AttnStepState,
        chunk_start: u32,
        k_positions: usize,
        raw_cap: u32,
    ) -> Result<crate::deferred::DeferredBuf> {
        let n_head = p.n_head as usize;
        let head_dim = p.head_dim as usize;
        let kv_row = p.n_lora_kv as usize;
        let n_embd = p.d_embd as usize;
        let n_lora_q = p.n_lora_q as usize;
        let q_dim = n_head * head_dim;
        let ratio = p.compress_ratio;
        let n_rot = p.n_rot as usize;
        let idx_hd = ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize;
        let idx_nhead = ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD as usize;
        let top_k = ds4_engine::attn_dispatch::ds4_n_indexer_top_k();
        let fast_topk = std::env::var("DS4_FAST_TOPK").ok().as_deref() != Some("0");
        let indexer_active = ratio == 4
            && layer.attn.has_indexer_compressor()
            && layer.attn.has_indexer_qb();

        let (main_kv_f16, main_gate_f16) = layer.attn.attn_compressor_f16();
        let main_comp = ds4_engine::attn_dispatch::CompressorInputs {
            w_kv: &layer.attn.attn_compressor_kv,
            w_gate: &layer.attn.attn_compressor_gate,
            w_kv_f16: main_kv_f16,
            w_gate_f16: main_gate_f16,
            w_ape: &layer.attn.attn_compressor_ape,
            w_norm: &layer.attn.attn_compressor_norm,
            head_dim: p.head_dim,
            compress_ratio: ratio,
        };
        let (idx_kv_f16, idx_gate_f16) = layer.attn.indexer_compressor_f16();
        let idx_comp = ds4_engine::attn_dispatch::CompressorInputs {
            w_kv: &layer.attn.indexer_compressor_kv,
            w_gate: &layer.attn.indexer_compressor_gate,
            w_kv_f16: idx_kv_f16,
            w_gate_f16: idx_gate_f16,
            w_ape: &layer.attn.indexer_compressor_ape,
            w_norm: &layer.attn.indexer_compressor_norm,
            head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
            compress_ratio: ratio,
        };

        // RoPE-forward all K q_heads at consecutive positions (one fused dispatch).
        scope.rope_tail_q_heads_in_place_k(
            &half.q_heads_k, n_head, head_dim, p, chunk_start, k_positions, false,
        )?;
        let comp_ring = self.dispatcher.comp_ring_or_alloc(
            layer_idx as u32,
            comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
        );
        let heads_k = scope.alloc_f32(k_positions * q_dim);

        // DS4_CHUNK_BATCHED_IDX: whole-chunk batched indexer selection + flash in
        // Phase-B (single-chunk only). DS4_CHUNK_ATTN_NOSYNC (requires batched_idx):
        // GPU-RESIDENT compressor — chain store→pool→rms→rope→ring-write→GPU-rotate
        // (compressor_rotate_ratio4, bit-equivalent to finish_emit's CPU rotation) in
        // the scope's cb with NO per-quad commit_wait + CPU readback; emit rows are
        // GPU-written into the comp/index rings; CPU mirrors resynced ONCE at chunk
        // end. Eliminates the ~681 emits × per-quad drains per idx layer.
        let batched_idx = std::env::var("DS4_CHUNK_BATCHED_IDX").ok().as_deref() == Some("1")
            && indexer_active
            && chunk_start == 0;
        let nosync = batched_idx
            && std::env::var("DS4_CHUNK_ATTN_NOSYNC").ok().as_deref() == Some("1");
        // DS4_CHUNK_FUSED_COMP (requires nosync): replace the per-position compressor
        // (store_one×K + pool/rms/rope/ring×emits + rotate×emits) with ONE fused
        // compressor_prefill_idx dispatch over the whole chunk (main + index), reading
        // the batched projections. The per-position loop then does ONLY the KV fp8
        // store. Decode-continuation pool state is restored by a tiny store_one+rotate
        // warm-up over the last quad+partial. This is the antirez fused-compressor
        // approach for ratio==4 — the path past the per-position dispatch overhead.
        let fused_comp = nosync
            && std::env::var("DS4_CHUNK_FUSED_COMP").ok().as_deref() == Some("1");

        // INCREMENT 1 (dispatch-count reduction, default-on): batch the INDEPENDENT
        // per-position ops across the chunk when the chunk does NOT wrap the raw ring
        // (chunk_start+K <= raw_cap — always true for a prefill-sized raw_cap; the SWA
        // wrap case can't store-ahead without breaking per-position causality, so it
        // falls back per-position). Batches: (1) KV store → kv_fp8_store_persistent_k
        // (512 dispatches → 1), (2) compressor + indexer kv/score projections →
        // comp_proj_k (byte-identical to the per-position matvecs). The store/pool/
        // flash stay per-position (sequential pool + causal flash). DS4_CHUNK_SEQ=1
        // forces the old fully-sequential path. Byte-identical to per-position.
        let batched = std::env::var("DS4_CHUNK_SEQ").ok().as_deref() != Some("1")
            && chunk_start + k_positions as u32 <= raw_cap;
        // Compressor PROJECTIONS (comp_proj_k matmuls) don't touch the raw ring, so
        // they can be batched even when the chunk wraps it — decouple from the
        // store-ahead gate under nosync (the prefill flash reads kv_normed_rotated_k,
        // not the fp8 ring, so wrap doesn't break causality). Replaces ~K per-position
        // matvecs/compressor with ONE matmul_k. The KV fp8 store stays per-position
        // when wrapping (`if !batched` below) — only the projections batch here.
        let batch_proj = batched || nosync;
        let main_w = 2 * head_dim;
        let idx_w = 2 * idx_hd;
        let (kv_main_k, sc_main_k, kv_idx_k, sc_idx_k) = if batch_proj {
            if batched {
                scope.kv_fp8_store_persistent_k(layer_idx as u32, &half.kv_normed_rotated_k, p, raw_cap, chunk_start, k_positions)?;
            }
            let kvm = crate::compressor::comp_proj_k(scope, main_comp.w_kv, main_comp.w_kv_f16, &half.normed_k, n_embd, main_w, k_positions)?;
            let scm = crate::compressor::comp_proj_k(scope, main_comp.w_gate, main_comp.w_gate_f16, &half.normed_k, n_embd, main_w, k_positions)?;
            let (kvi, sci) = if indexer_active {
                (
                    Some(crate::compressor::comp_proj_k(scope, idx_comp.w_kv, idx_comp.w_kv_f16, &half.normed_k, n_embd, idx_w, k_positions)?),
                    Some(crate::compressor::comp_proj_k(scope, idx_comp.w_gate, idx_comp.w_gate_f16, &half.normed_k, n_embd, idx_w, k_positions)?),
                )
            } else { (None, None) };
            (Some(kvm), Some(scm), kvi, sci)
        } else {
            (None, None, None, None)
        };

        // NOSYNC + wrap (K > raw_cap): batch the KV fp8 store into ≤2 dispatches
        // instead of K per-position stores (the @3000 prefill's dominant cost — ~57k
        // single-position store encoders across the idx layers). Under nosync/batched_idx
        // the prefill flash reads `kv_normed_rotated_k` (not the fp8 ring), so the fp8
        // ring is DECODE-only: only the last `raw_cap` positions (the SWA window)
        // survive, mapping to slots [pos%raw_cap]. Those raw_cap consecutive positions
        // form exactly ONE ring wrap → two contiguous batched stores.
        if nosync && !batched {
            // Boundary (kv ring wrap vs upstream writers): the ≤2 wrap stores
            // below read half.kv_normed_rotated_k (written by the attn-half
            // stage earlier in this cb) and overwrite ALL raw ring slots.
            // !batched ⇒ chunk_start+K > raw_cap. GPU-ordered split, NO CPU
            // drain (a drain here corrupts the nosync mirror-resync logic).
            if chunk_event_order_on() {
                scope.event_split("nosync_kvwrap_event");
            }
            let kp_u = k_positions as u32;
            let store_n = kp_u.min(raw_cap);
            let first = kp_u - store_n;
            let s0 = first % raw_cap;
            let count1 = (raw_cap - s0).min(store_n);
            let slice1 = scope.slice_out_f32(
                &half.kv_normed_rotated_k, first as usize * kv_row, count1 as usize * kv_row);
            scope.kv_fp8_store_persistent_k(layer_idx as u32, &slice1, p, raw_cap, s0, count1 as usize)?;
            let count2 = store_n - count1;
            if count2 > 0 {
                let slice2 = scope.slice_out_f32(
                    &half.kv_normed_rotated_k, (first + count1) as usize * kv_row, count2 as usize * kv_row);
                scope.kv_fp8_store_persistent_k(layer_idx as u32, &slice2, p, raw_cap, 0, count2 as usize)?;
            }
        }

        // DS4_CHUNK_IDX_KFLASH=1: tile the ratio==4 idx core by quad and replace the
        // `ratio` per-position flashes per quad with ONE masked SWA flash — the same
        // sliding-window construction the noidx core uses (fresh private raw_win, comp
        // rows read from the persistent ring). ONLY valid while the indexer selection
        // never engages in this chunk (post_n_index <= top_k throughout — true for
        // positions <= top_k*ratio, i.e. the first ~2048 tokens; @600 fully, @3000's
        // bulk). The per-quad finish (drain + pool rotation) is KEPT unchanged — that
        // sync is load-bearing for the ratio==4 two-window rotation; only the flash is
        // batched. If selection would engage anywhere in the chunk, fall through to the
        // per-position path (which has the GPU-resident selection branch).
        let idx_kflash = std::env::var("DS4_CHUNK_IDX_KFLASH").ok().as_deref() == Some("1")
            && indexer_active
            && ratio != 0
            && raw_cap >= ratio
            && chunk_start + k_positions as u32 <= raw_cap.max(top_k * ratio)
            // max post_n_index in the chunk <= top_k → selection never engages.
            && state.n_index_comp[layer_idx] + (k_positions as u32).div_ceil(ratio) <= top_k;
        if idx_kflash {
            const NEG: u16 = 0xFC00;
            let w = raw_cap;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let sinks_db = scope.upload_f32(&layer.attn.attn_sinks);
            let kp_u = k_positions as u32;
            let ring = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32, raw_cap as usize * kv_row * std::mem::size_of::<f32>());
            let mut t = 0u32;
            while t < kp_u {
                let t0 = chunk_start + t;
                let next_boundary = (t0 / ratio + 1) * ratio;
                let tile_end = next_boundary.min(chunk_start + kp_u).min(t0 + w);
                let tk = tile_end - t0;
                let n_comp_start = state.n_comp[layer_idx];
                let mut emit_row_written = false;
                let pre = t0.min(w.saturating_sub(1));
                let n_win = pre + tk;
                let n_raw8 = n_win.div_ceil(8) * 8;
                let raw_win = scope.alloc_swa_raw_window(n_raw8 as usize, head_dim);
                if pre > 0 {
                    scope.gather_ring_window(&ring, &raw_win, 0, raw_cap, t0 - pre, pre, head_dim as u32);
                }
                // Per-position: KV store + main compressor (→comp ring) + idx compressor
                // (→pool state). Capture the emitter's handles for the per-quad finish.
                let mut emit_main: Option<(crate::compressor::CompressorScopeHandle, Option<crate::deferred::DeferredBuf>, u32)> = None;
                let mut emit_idx: Option<(crate::compressor::CompressorScopeHandle, Option<crate::deferred::DeferredBuf>)> = None;
                for j in 0..tk {
                    let pos = t0 + j;
                    let slot = pos % raw_cap;
                    let should_compress = (pos + 1) % ratio == 0;
                    // KV store: batched above (kv_fp8_store_persistent_k) when the chunk
                    // fits the ring; else per-position here. Never both (double-store).
                    if !batched {
                        let kv_j = scope.slice_out_f32(
                            &half.kv_normed_rotated_k, (t + j) as usize * kv_row, kv_row);
                        scope.kv_fp8_store_persistent(layer_idx as u32, &kv_j, p, raw_cap, slot)?;
                    }
                    let _ = slot;
                    let normed_j = scope.slice_out_f32(&half.normed_k, (t + j) as usize * n_embd, n_embd);
                    let (main_handle, main_emit_db) =
                        if let (Some(kvm), Some(scm)) = (&kv_main_k, &sc_main_k) {
                            let kv_main_j = scope.slice_out_f32(kvm, (t + j) as usize * main_w, main_w);
                            let sc_main_j = scope.slice_out_f32(scm, (t + j) as usize * main_w, main_w);
                            crate::compressor::compressor_encode_with_proj(
                                &*scope, self.dispatcher, p, &main_comp, &kv_main_j, &sc_main_j,
                                state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                                Some((&comp_ring, n_comp_start)),
                            )?
                        } else {
                            crate::compressor::compressor_encode_in_scope(
                                &*scope, self.dispatcher, p, &main_comp, &normed_j,
                                state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                                Some((&comp_ring, n_comp_start)),
                            )?
                        };
                    let (idx_handle, idx_emit_db) =
                        if let (Some(kvi), Some(sci)) = (&kv_idx_k, &sc_idx_k) {
                            let kv_idx_j = scope.slice_out_f32(kvi, (t + j) as usize * idx_w, idx_w);
                            let sc_idx_j = scope.slice_out_f32(sci, (t + j) as usize * idx_w, idx_w);
                            crate::compressor::compressor_encode_with_proj(
                                &*scope, self.dispatcher, p, &idx_comp, &kv_idx_j, &sc_idx_j,
                                state.index_state_kv[layer_idx].len(), pos, layer_idx as u32, true,
                                None,
                            )?
                        } else {
                            crate::compressor::compressor_encode_in_scope(
                                &*scope, self.dispatcher, p, &idx_comp, &normed_j,
                                state.index_state_kv[layer_idx].len(), pos, layer_idx as u32, true,
                                None,
                            )?
                        };
                    if should_compress {
                        emit_row_written = true;
                        emit_main = Some((main_handle, main_emit_db, pos));
                        emit_idx = Some((idx_handle, idx_emit_db));
                    }
                }
                scope.gather_ring_window(&ring, &raw_win, pre, raw_cap, t0, tk, head_dim as u32);
                let n_comp_flash = n_comp_start + emit_row_written as u32;
                let mask_row = n_raw8 as usize + n_comp_flash as usize;
                let mut mask = vec![NEG; tk as usize * mask_row];
                for r in 0..tk {
                    let pos = t0 + r;
                    let base = r as usize * mask_row;
                    for s in 0..(n_win as usize) {
                        let pj = if (s as u32) < pre { t0 - pre + s as u32 }
                                 else { t0 + (s as u32 - pre) };
                        if pj <= pos && pj + w > pos { mask[base + s] = 0; }
                    }
                    for c in 0..(n_comp_start as usize) {
                        mask[base + n_raw8 as usize + c] = 0;
                    }
                    if emit_row_written && r == tk - 1 {
                        mask[base + n_raw8 as usize + n_comp_start as usize] = 0;
                    }
                }
                let mut g = 0u32;
                while g < tk {
                    let rem = tk - g;
                    let gk = if rem >= 8 { 8 } else if rem >= 4 { 4 } else if rem >= 2 { 2 } else { 1 };
                    let q_group = scope.slice_out_f32(
                        &half.q_heads_k, (t + g) as usize * q_dim, gk as usize * q_dim);
                    let mask_group =
                        mask[(g as usize * mask_row)..((g + gk) as usize * mask_row)].to_vec();
                    let heads_g = scope.flash_attn_k_mla_comp_masked(
                        &q_group, &raw_win, &comp_ring, n_comp_flash, &mask_group,
                        n_head, head_dim, head_dim, n_raw8 as usize, gk as usize, scale, &sinks_db,
                    )?;
                    scope.copy_buf_into(&heads_g, heads_k.buffer(), (t + g) as usize * q_dim);
                    g += gk;
                }
                // Per-quad finish (drain + pool rotation) — KEPT, the load-bearing sync.
                if let Some((main_handle, main_emit_db, pos)) = emit_main {
                    scope.commit_wait_stage("chunk_comp_finish");
                    let pooled_main = match &main_emit_db {
                        Some(e) => unsafe {
                            std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                        }.to_vec(),
                        None => Vec::new(),
                    };
                    if let Some(row) = crate::compressor::compressor_finish_in_scope(
                        main_handle, self.dispatcher, p, &main_comp, pooled_main,
                        &mut state.comp_state_kv[layer_idx],
                        &mut state.comp_state_score[layer_idx], pos,
                    ) {
                        state.comp_kv_ring[layer_idx].extend_from_slice(&row);
                        state.n_comp[layer_idx] += 1;
                    }
                    if let Some((idx_handle, idx_emit_db)) = emit_idx {
                        let pooled_idx = match &idx_emit_db {
                            Some(e) => unsafe {
                                std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                            }.to_vec(),
                            None => Vec::new(),
                        };
                        if let Some(row) = crate::compressor::compressor_finish_in_scope(
                            idx_handle, self.dispatcher, p, &idx_comp, pooled_idx,
                            &mut state.index_state_kv[layer_idx],
                            &mut state.index_state_score[layer_idx], pos,
                        ) {
                            state.index_comp_kv_ring[layer_idx].extend_from_slice(&row);
                            state.n_index_comp[layer_idx] += 1;
                        }
                    }
                }
                t += tk;
            }
            if n_rot > 0 {
                scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
            }
            return Ok(heads_k);
        }

        // DS4_CHUNK_BATCHED_IDX (default off): step 4 of the antirez batched-
        // prefill-attention port. Phase-A (the per-position loop below) builds the
        // KV / main / index compressor rings ONLY (no flash); Phase-B (after the
        // loop) does the whole chunk's indexer selection + flash in 3 batched
        // dispatches (encode_indexer_scores_tiled → encode_indexer_topk_batched →
        // encode_indexed_mixed_attention) — antirez's proven-coherent path. Only
        // single-chunk (chunk_start==0): the batched mixed-attention flash reads the
        // raw KV as a contiguous f32 ring (half.kv_normed_rotated_k holds all K
        // positions linearly), so multi-chunk (chunk_start>0) — which needs a
        // persistent f32 SWA ring spanning prior chunks — falls back to per-position.
        let nosync_idx_ring = if nosync {
            Some(self.dispatcher.index_comp_ring_or_alloc(
                layer_idx as u32,
                comp_ring_rows(raw_cap) * idx_hd * std::mem::size_of::<f32>(),
            ))
        } else {
            None
        };
        let mut emit_count: u32 = 0;

        let kp = chunk_kprof_on();
        for i in 0..k_positions {
            let pos = chunk_start + i as u32;
            let slot = pos % raw_cap;
            let n_raw = (pos + 1).min(raw_cap);
            let should_compress = ratio != 0 && (pos + 1) % ratio == 0;
            let post_n_comp = state.n_comp[layer_idx] + should_compress as u32;
            let post_n_index = state.n_index_comp[layer_idx] + should_compress as u32;

            // 1. store KV (SWA ring slot) — batched above unless wrapping. Under
            //    nosync the wrap case is also batched (the ≤2-dispatch store above),
            //    so the per-position store only runs in the legacy (!nosync) path.
            if !batched && !nosync {
                let kv_i = scope.slice_out_f32(&half.kv_normed_rotated_k, i * kv_row, kv_row);
                scope.kv_fp8_store_persistent(layer_idx as u32, &kv_i, p, raw_cap, slot)?;
            }
            // FUSED_COMP: skip the per-position compressor + flash; the whole-chunk
            // fused compressor + Phase-B flash run after the loop.
            if fused_comp {
                continue;
            }

            // 2-3. main + indexer compressor encode (main appends to comp ring) —
            //       projections batched above when `batched`.
            let normed_i = scope.slice_out_f32(&half.normed_k, i * n_embd, n_embd);
            if std::env::var("DS4_NORMED_TRACE").is_ok() && layer_idx == 4 && pos < chunk_start + 4 {
                let nv = scope.debug_read(&normed_i);
                let sum: f64 = nv.iter().map(|&x| x as f64).sum();
                eprintln!("[NORMED chunkcore] L{layer_idx} pos={pos} sum={sum:.6} head={:?}", &nv[..4.min(nv.len())]);
            }
            let (main_handle, main_emit_db) = if let (Some(kvm), Some(scm)) = (&kv_main_k, &sc_main_k) {
                let kv_main_i = scope.slice_out_f32(kvm, i * main_w, main_w);
                let sc_main_i = scope.slice_out_f32(scm, i * main_w, main_w);
                crate::compressor::compressor_encode_with_proj(
                    &*scope, self.dispatcher, p, &main_comp, &kv_main_i, &sc_main_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, state.n_comp[layer_idx] + emit_count)),
                )?
            } else {
                crate::compressor::compressor_encode_in_scope(
                    &*scope, self.dispatcher, p, &main_comp, &normed_i,
                    state.comp_state_kv[layer_idx].len(), pos, layer_idx as u32, false,
                    Some((&comp_ring, state.n_comp[layer_idx] + emit_count)),
                )?
            };
            let idx_build = if indexer_active {
                if let (Some(kvi), Some(sci)) = (&kv_idx_k, &sc_idx_k) {
                    let kv_idx_i = scope.slice_out_f32(kvi, i * idx_w, idx_w);
                    let sc_idx_i = scope.slice_out_f32(sci, i * idx_w, idx_w);
                    Some(crate::compressor::compressor_encode_with_proj(
                        &*scope, self.dispatcher, p, &idx_comp, &kv_idx_i, &sc_idx_i,
                        state.index_state_kv[layer_idx].len(), pos, layer_idx as u32, true,
                        nosync_idx_ring.as_ref().map(|r| (r, state.n_index_comp[layer_idx] + emit_count)),
                    )?)
                } else {
                    Some(crate::compressor::compressor_encode_in_scope(
                        &*scope, self.dispatcher, p, &idx_comp, &normed_i,
                        state.index_state_kv[layer_idx].len(), pos, layer_idx as u32, true,
                        nosync_idx_ring.as_ref().map(|r| (r, state.n_index_comp[layer_idx] + emit_count)),
                    )?)
                }
            } else {
                None
            };

            if kp { chunk_kprof_add("2a_compress", scope.commit_wait_gpu_us()); }

            // DS4_IDX_POOL_DRAIN (diagnostic): serialize the per-position compressor
            // pool encodes by draining after each one. Tests whether the intra-quad
            // (unbarriered, one-scope) pool read-modify-write races vs the per-token
            // path (which commits each position).
            if std::env::var("DS4_IDX_POOL_DRAIN").is_ok() {
                scope.commit_wait_stage("idx_pool_serialize");
            }

            // 4. flash over [raw window | comp ring], with indexer selection past
            //    top_k. q_heads slice is already rope-fwd'd by the batched call.
            //    Under DS4_CHUNK_BATCHED_IDX the per-position flash is SKIPPED here
            //    (Phase-A builds rings only); Phase-B below flashes the whole chunk.
            if !batched_idx {
            let q_i = scope.slice_out_f32(&half.q_heads_k, i * q_dim, q_dim);
            let heads_i = if indexer_active && post_n_index > top_k {
                let idx_ring = self.dispatcher.index_comp_ring_or_alloc(
                    layer_idx as u32,
                    comp_ring_rows(raw_cap) * idx_hd * std::mem::size_of::<f32>(),
                );
                // Sync the GPU idx ring's committed rows from the CPU ring (the
                // source of truth, maintained by the deferred finish; a no-op in
                // steady state), then append THIS token's not-yet-committed emit.
                let n_committed = state.n_index_comp[layer_idx];
                let synced = state.index_comp_ring_gpu_rows[layer_idx];
                if synced < n_committed {
                    let cpu = &state.index_comp_kv_ring[layer_idx];
                    let start = synced as usize * idx_hd;
                    let end = n_committed as usize * idx_hd;
                    debug_assert!(cpu.len() >= end, "idx ring CPU shorter than n_comp");
                    unsafe {
                        let dst = (idx_ring.contents() as *mut f32).add(start);
                        std::ptr::copy_nonoverlapping(cpu.as_ptr().add(start), dst, end - start);
                    }
                    state.index_comp_ring_gpu_rows[layer_idx] = n_committed;
                }
                if let Some((_, Some(ref emit))) = idx_build {
                    scope.copy_buf_into(emit, &idx_ring, n_committed as usize * idx_hd);
                }
                let index_ring_db = crate::deferred::DeferredBuf::from_external_buffer(
                    idx_ring, post_n_index as usize * idx_hd,
                );
                let qr_i = scope.slice_out_f32(&half.qr_normed_k, i * n_lora_q, n_lora_q);
                let (qb_f16, proj_f16) = layer.attn.indexer_scoring_f16();
                let scores_db = scope.encode_indexer_scores(
                    &qr_i, &normed_i,
                    &layer.attn.indexer_attn_q_b, &layer.attn.indexer_proj,
                    qb_f16, proj_f16, &index_ring_db, post_n_index as usize,
                    idx_nhead, idx_hd, p, pos,
                )?;
                let k_sel = (top_k as usize).min(post_n_index as usize);
                if std::env::var("DS4_SEL_TRACE").is_ok() && layer_idx <= 6 && post_n_index <= top_k + 2 {
                    let ring = scope.debug_read(&index_ring_db);
                    let nc = post_n_index as usize;
                    let r0: Vec<f32> = ring.get(0..3).map(|s| s.to_vec()).unwrap_or_default();
                    let rl: Vec<f32> = ring.get((nc-1)*idx_hd..(nc-1)*idx_hd+3).map(|s| s.to_vec()).unwrap_or_default();
                    let s = scope.debug_read(&scores_db);
                    let sum: f64 = s.iter().map(|&x| x as f64).sum();
                    eprintln!("[SEL CHUNK] L{layer_idx} pos={pos} n_idx={post_n_index} ring0={r0:.6?} ringL={rl:.6?} scsum={sum:.6}");
                }
                let sel_db = if fast_topk {
                    scope.encode_indexer_topk_threshold(&scores_db, post_n_index as usize, k_sel)?
                } else {
                    scope.encode_indexer_topk(&scores_db, post_n_index as usize, k_sel)?
                };
                scope.flash_attn_decode_persistent_compressor_qbuf_gpuring_sel(
                    layer_idx as u32, p, &q_i, n_raw, raw_cap,
                    &comp_ring, &sel_db, k_sel as u32, &layer.attn.attn_sinks,
                )?
            } else {
                let comp_sel: Vec<u32> = (0..post_n_comp).collect();
                scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                    layer_idx as u32, p, &q_i, n_raw, raw_cap,
                    &comp_ring, &comp_sel, post_n_comp, &layer.attn.attn_sinks,
                )?
            };

            // 5. collect raw flash output (rope-back batched after the loop —
            //    rope is per-position-independent → byte-identical).
            scope.copy_buf_into(&heads_i, heads_k.buffer(), i * q_dim);
            if kp { chunk_kprof_add("2b_flash", scope.commit_wait_gpu_us()); }
            } // end `if !batched_idx` (per-position flash)

            // 6. On emit: flush + finish both compressors (the ratio==4 pool
            //    rotation must land before the next quad's stores).
            if should_compress && nosync {
                // GPU-RESIDENT rotation: the emit row is already GPU-written into
                // the comp/index rings (compressor finish_gpu); rotate the pools'
                // two windows in-GPU (front:=back, bit-equivalent to finish_emit's
                // CPU rotation) in this same cb — NO commit_wait, NO CPU readback.
                // The next quad's store_one (next iter, same cb) sees the rotated
                // pool by GPU ordering. CPU mirrors resynced once at chunk end.
                let m_bytes = state.comp_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
                let mpk = self.dispatcher.compressor_state_kv_or_alloc(layer_idx as u32, m_bytes);
                let mps = self.dispatcher.compressor_state_score_or_alloc(layer_idx as u32, m_bytes);
                scope.compressor_rotate_ratio4(&mpk, &mps, head_dim as u32)?;
                let i_bytes = state.index_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
                let ipk = self.dispatcher.indexer_state_kv_or_alloc(layer_idx as u32, i_bytes);
                let ips = self.dispatcher.indexer_state_score_or_alloc(layer_idx as u32, i_bytes);
                scope.compressor_rotate_ratio4(&ipk, &ips, idx_hd as u32)?;
                emit_count += 1;
            } else if should_compress {
                scope.commit_wait_stage("chunk_comp_finish");
                let pooled_main = match &main_emit_db {
                    Some(e) => unsafe {
                        std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                    }
                    .to_vec(),
                    None => Vec::new(),
                };
                if let Some(row) = crate::compressor::compressor_finish_in_scope(
                    main_handle, self.dispatcher, p, &main_comp, pooled_main,
                    &mut state.comp_state_kv[layer_idx],
                    &mut state.comp_state_score[layer_idx], pos,
                ) {
                    state.comp_kv_ring[layer_idx].extend_from_slice(&row);
                    state.n_comp[layer_idx] += 1;
                }
                if let Some((idx_handle, idx_emit_db)) = idx_build {
                    let pooled_idx = match &idx_emit_db {
                        Some(e) => unsafe {
                            std::slice::from_raw_parts(e.buffer().contents() as *const f32, e.len())
                        }
                        .to_vec(),
                        None => Vec::new(),
                    };
                    if let Some(row) = crate::compressor::compressor_finish_in_scope(
                        idx_handle, self.dispatcher, p, &idx_comp, pooled_idx,
                        &mut state.index_state_kv[layer_idx],
                        &mut state.index_state_score[layer_idx], pos,
                    ) {
                        state.index_comp_kv_ring[layer_idx].extend_from_slice(&row);
                        state.n_index_comp[layer_idx] += 1;
                    }
                }
            }
            // Non-emit handles drop here (no rotation; the emit's
            // sync_pool_to_mirror captures the whole quad's resident pool).

            // DS4_CHUNK_POS_COMMIT diagnostic: split the cb every quad in the
            // per-position idx core (commit WITHOUT wait; queue order preserved).
            if should_compress
                && std::env::var("DS4_CHUNK_POS_COMMIT").ok().as_deref() == Some("1")
            {
                scope.commit_keep_open();
            }
        }

        // FUSED_COMP: build ALL emit rows for the chunk in ONE dispatch (main +
        // index), then a tiny store_one+rotate warm-up over the last quad+partial to
        // leave the GPU compressor pools in the exact state decode continues from.
        if fused_comp {
            let n_emits = (k_positions as u32) / ratio; // complete quads (chunk_start==0)
            if n_emits > 0 {
                // Boundary (>raw_cap chunks): the fused compressor reads the whole
                // chunk's batched projections (written earlier in this cb) and
                // writes the comp/index rings the Phase-B flash gathers.
                if chunk_start + k_positions as u32 > raw_cap && chunk_event_order_on() {
                    scope.event_split("fused_comp_event");
                }
                let (kvm, scm) = (kv_main_k.as_ref().unwrap(), sc_main_k.as_ref().unwrap());
                scope.compressor_prefill_idx(
                    kvm, scm, main_comp.w_ape, main_comp.w_norm, &comp_ring,
                    head_dim as u32, n_rot as u32, chunk_start,
                    state.n_comp[layer_idx], n_emits, p, 1.0e-6,
                )?;
                if let (Some(kvi), Some(sci), Some(iring)) =
                    (&kv_idx_k, &sc_idx_k, nosync_idx_ring.as_ref())
                {
                    scope.compressor_prefill_idx(
                        kvi, sci, idx_comp.w_ape, idx_comp.w_norm, iring,
                        idx_hd as u32, n_rot as u32, chunk_start,
                        state.n_index_comp[layer_idx], n_emits, p, 1.0e-6,
                    )?;
                }
                // Pool warm-up for decode continuation: zero the pools, then replay
                // store_one (+ per-quad GPU rotate) for the LAST complete quad +
                // trailing partial — leaving prev=last-quad, cur=partial, exactly as
                // the per-position path would. Reuses the proven kernels (≤8 store_one
                // + ≤2 rotate / layer). kv/sc are sliced from the batched projections.
                let m_bytes = state.comp_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
                let mpk = self.dispatcher.compressor_state_kv_or_alloc(layer_idx as u32, m_bytes);
                let mps = self.dispatcher.compressor_state_score_or_alloc(layer_idx as u32, m_bytes);
                let i_bytes = state.index_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
                let ipk = self.dispatcher.indexer_state_kv_or_alloc(layer_idx as u32, i_bytes);
                let ips = self.dispatcher.indexer_state_score_or_alloc(layer_idx as u32, i_bytes);
                unsafe {
                    std::ptr::write_bytes(mpk.contents() as *mut u8, 0, m_bytes);
                    std::ptr::write_bytes(mps.contents() as *mut u8, 0, m_bytes);
                    std::ptr::write_bytes(ipk.contents() as *mut u8, 0, i_bytes);
                    std::ptr::write_bytes(ips.contents() as *mut u8, 0, i_bytes);
                }
                // APE in kernel layout [ratio, width] (transpose of model [width, ratio]).
                let ape_k = |w_ape: &[f32], w: usize| -> Vec<f32> {
                    let mut a = vec![0.0f32; ratio as usize * w];
                    for j in 0..w { for r in 0..ratio as usize { a[r * w + j] = w_ape[j * ratio as usize + r]; } }
                    a
                };
                let ape_m = ape_k(main_comp.w_ape, main_w);
                let ape_i = ape_k(idx_comp.w_ape, idx_w);
                let ape_m_b = unsafe { std::slice::from_raw_parts(ape_m.as_ptr() as *const u8, ape_m.len() * 4) };
                let ape_i_b = unsafe { std::slice::from_raw_parts(ape_i.as_ptr() as *const u8, ape_i.len() * 4) };
                let warm_start = if n_emits >= 1 { (n_emits - 1) * ratio } else { 0 };
                for pos in warm_start..k_positions as u32 {
                    let i = pos as usize;
                    let kv_main_i = scope.slice_out_f32(kvm, i * main_w, main_w);
                    let sc_main_i = scope.slice_out_f32(scm, i * main_w, main_w);
                    scope.compressor_store_one_db(&kv_main_i, &sc_main_i, ape_m_b, &mpk, &mps, main_w as u32, ratio, pos, false)?;
                    if let (Some(kvi), Some(sci)) = (&kv_idx_k, &sc_idx_k) {
                        let kv_idx_i = scope.slice_out_f32(kvi, i * idx_w, idx_w);
                        let sc_idx_i = scope.slice_out_f32(sci, i * idx_w, idx_w);
                        scope.compressor_store_one_db(&kv_idx_i, &sc_idx_i, ape_i_b, &ipk, &ips, idx_w as u32, ratio, pos, false)?;
                    }
                    if (pos + 1) % ratio == 0 {
                        scope.compressor_rotate_ratio4(&mpk, &mps, head_dim as u32)?;
                        scope.compressor_rotate_ratio4(&ipk, &ips, idx_hd as u32)?;
                    }
                }
                emit_count = n_emits; // drives the shared nosync resync below
            }
        }

        // NOSYNC chunk-end resync: ONE commit_wait flushes the whole Phase-A cb
        // (all stores/pools/rms/rope/ring-writes/rotations), then mirror the GPU
        // pools + rings back to the CPU state ONCE so decode-after-prefill (which
        // resyncs the pool on its first compress) and any CPU-ring reader see the
        // exact post-prefill state — byte-identical to the per-position finish, but
        // with 1 sync/layer instead of ~681. Phase-B reads the GPU rings directly.
        if nosync && emit_count > 0 {
            // Crossing chunks add event_splits (wrap-store/fused-comp), which
            // push cbs into pending_cbs; commit_wait_stage CPU-waits ONLY the
            // current cb, so the CPU mirror reads below would race the still-
            // running split cbs (stale/garbage pools + rings → the BOS
            // corruption). Drain EVERYTHING before reading.
            let hd_m = head_dim;            // main compressor row width (=512)
            let n0 = state.n_comp[layer_idx] as usize;
            let m_bytes = state.comp_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
            let mpk = self.dispatcher.compressor_state_kv_or_alloc(layer_idx as u32, m_bytes);
            let mps = self.dispatcher.compressor_state_score_or_alloc(layer_idx as u32, m_bytes);
            let i_bytes = state.index_state_kv[layer_idx].len() * std::mem::size_of::<f32>();
            let ipk = self.dispatcher.indexer_state_kv_or_alloc(layer_idx as u32, i_bytes);
            let ips = self.dispatcher.indexer_state_score_or_alloc(layer_idx as u32, i_bytes);
            // PHASE 1 (DS4_CHUNK_DEFER_RESYNC): all of mpk/mps/ipk/ips + comp_ring +
            // idx_ring are per-layer-persistent; the only in-chunk consumers are the
            // COUNT bumps (Phase-B reads n_comp/n_index) and the gpu_rows flag (makes
            // IDX-N2 a no-op) — both kept immediate. The pool/ring CPU reads are only
            // for decode-after-prefill, so DEFER them past the terminal wait (drops the
            // per-layer drain_all → the one-cb-per-layer split).
            if defer_chunk_resync() {
                self.chunk_resyncs.borrow_mut().push(ChunkResync {
                    layer_idx, main_pk: mpk.clone(), main_ps: mps.clone(),
                    idx_pk: Some(ipk.clone()), idx_ps: Some(ips.clone()),
                    comp_ring: comp_ring.clone(),
                    idx_ring: nosync_idx_ring.as_ref().map(|r| (*r).clone()),
                    comp_row0: n0 as u32, n_emit: emit_count, head_dim: hd_m, idx_hd,
                });
                state.n_comp[layer_idx] += emit_count;
                state.n_index_comp[layer_idx] += emit_count;
                state.index_comp_ring_gpu_rows[layer_idx] = state.n_index_comp[layer_idx];
                // fall through to Phase-B (reads the resident GPU rings directly).
            } else {
            scope.drain_all_stage("nosync_chunk_resync");
            unsafe {
                // pools → CPU mirrors
                std::ptr::copy_nonoverlapping(mpk.contents() as *const f32,
                    state.comp_state_kv[layer_idx].as_mut_ptr(), state.comp_state_kv[layer_idx].len());
                std::ptr::copy_nonoverlapping(mps.contents() as *const f32,
                    state.comp_state_score[layer_idx].as_mut_ptr(), state.comp_state_score[layer_idx].len());
                std::ptr::copy_nonoverlapping(ipk.contents() as *const f32,
                    state.index_state_kv[layer_idx].as_mut_ptr(), state.index_state_kv[layer_idx].len());
                std::ptr::copy_nonoverlapping(ips.contents() as *const f32,
                    state.index_state_score[layer_idx].as_mut_ptr(), state.index_state_score[layer_idx].len());
                // GPU rings (rows [n0 .. n0+emit_count]) → CPU ring mirrors
                let m_src = (comp_ring.contents() as *const f32).add(n0 * hd_m);
                state.comp_kv_ring[layer_idx].extend_from_slice(
                    std::slice::from_raw_parts(m_src, emit_count as usize * hd_m));
                if let Some(ir) = nosync_idx_ring.as_ref() {
                    let i_src = (ir.contents() as *const f32).add(n0 * idx_hd);
                    state.index_comp_kv_ring[layer_idx].extend_from_slice(
                        std::slice::from_raw_parts(i_src, emit_count as usize * idx_hd));
                }
            }
            state.n_comp[layer_idx] += emit_count;
            state.n_index_comp[layer_idx] += emit_count;
            // The GPU index ring is already populated by the loop's ring-writes;
            // mark it synced so Phase-B's CPU→GPU copy is a no-op (synced==n_index).
            state.index_comp_ring_gpu_rows[layer_idx] = state.n_index_comp[layer_idx];
            } // end non-defer (immediate-resync) branch
        }

        // PHASE-B (DS4_CHUNK_BATCHED_IDX): the whole chunk's indexer selection +
        // mixed-attention flash in 3 batched dispatches, replacing the per-position
        // flash skipped above. Phase-A (the loop) has built the persistent main comp
        // ring (GPU `comp_ring`) and the CPU index ring (state.index_comp_kv_ring).
        // antirez's proven-coherent path: batched q_k/weights_k → tiled indexer
        // scores (causal mask baked) → batched top_k → indexed mixed-attention over
        // [SWA raw window | top_k comp rows], per-token causal/window masking inside
        // the kernel.
        if batched_idx {
            // Boundary (>raw_cap chunks): the batched scores/top-k/mixed-attention
            // read the comp + index rings and the raw KV written by Phase-A/the
            // fused compressor in this cb. GPU-ordered split, no CPU drain (the
            // nosync chunk-end resync above keeps the CPU mirrors; a mid-scope
            // drain here corrupts that resync's invariants).
            if chunk_start + k_positions as u32 > raw_cap && chunk_event_order_on() {
                scope.event_split("nosync_flashb_event");
            }
            let n_comp = state.n_comp[layer_idx];
            let n_index = state.n_index_comp[layer_idx];
            debug_assert_eq!(
                n_comp, n_index,
                "PHASE-B: main/index comp ring counts diverged (n_comp={n_comp} n_index={n_index})"
            );
            if n_comp == 0 {
                // No compressor rows emitted yet (chunk shorter than `ratio`): the
                // whole chunk attends only the raw SWA window. Per-position raw flash
                // (cheap — runs only for the first <ratio positions of layer-3+).
                for i in 0..k_positions {
                    let pos = chunk_start + i as u32;
                    let n_raw = (pos + 1).min(raw_cap);
                    let q_i = scope.slice_out_f32(&half.q_heads_k, i * q_dim, q_dim);
                    let empty: Vec<u32> = Vec::new();
                    let heads_i = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                        layer_idx as u32, p, &q_i, n_raw, raw_cap,
                        &comp_ring, &empty, 0, &layer.attn.attn_sinks,
                    )?;
                    scope.copy_buf_into(&heads_i, heads_k.buffer(), i * q_dim);
                }
            } else {
                // (a) batched indexer queries q_k = w_q_b · qr_normed_k, rope-fwd'd.
                let idx_q_dim = idx_nhead * idx_hd;
                let (qb_f16, proj_f16) = layer.attn.indexer_scoring_f16();
                // DS4_IDX_MM (default on): use the weight-stationary tiled matmul
                // (matmul_k_f16) instead of the per-token mul_mv (matmul_f16_k). The
                // mul_mv re-streams the whole [n_lora_q × idx_q_dim] f16 weight ONCE
                // PER TOKEN (K× weight traffic → ~150ms/layer, the dominant indexer
                // cost); the tiled matmul loads each weight tile once per 32-token tile.
                let idx_mm = std::env::var("DS4_IDX_MM").ok().as_deref() != Some("0");
                let q_k = if let Some(bytes) = qb_f16.filter(|_| n_lora_q % 8 == 0 && idx_mm) {
                    let w = scope.weight_f16(bytes);
                    scope.matmul_k_f16(&w, &half.qr_normed_k, n_lora_q, idx_q_dim, k_positions)?
                } else if let Some(bytes) = qb_f16.filter(|_| n_lora_q % 4 == 0 && idx_q_dim % 2 == 0) {
                    let w = scope.weight_f16(bytes);
                    scope.matmul_f16_k(&w, &half.qr_normed_k, n_lora_q, idx_q_dim, k_positions)?
                } else {
                    let w = scope.weight_f32(&layer.attn.indexer_attn_q_b);
                    scope.matmul_f32_k(&w, &half.qr_normed_k, n_lora_q, idx_q_dim, k_positions)?
                };
                scope.rope_tail_q_heads_in_place_k(&q_k, idx_nhead, idx_hd, p, chunk_start, k_positions, false)?;
                // (b) batched indexer weights = w_proj · normed_k → [K, idx_nhead].
                let weights_k = if let Some(bytes) = proj_f16.filter(|_| n_embd % 8 == 0 && idx_mm) {
                    let w = scope.weight_f16(bytes);
                    scope.matmul_k_f16(&w, &half.normed_k, n_embd, idx_nhead, k_positions)?
                } else if let Some(bytes) = proj_f16.filter(|_| n_embd % 4 == 0 && idx_nhead % 2 == 0) {
                    let w = scope.weight_f16(bytes);
                    scope.matmul_f16_k(&w, &half.normed_k, n_embd, idx_nhead, k_positions)?
                } else {
                    let w = scope.weight_f32(&layer.attn.indexer_proj);
                    scope.matmul_f32_k(&w, &half.normed_k, n_embd, idx_nhead, k_positions)?
                };
                // (c) sync the GPU index ring from the CPU ring (full n_index rows).
                let idx_ring = self.dispatcher.index_comp_ring_or_alloc(
                    layer_idx as u32,
                    comp_ring_rows(raw_cap) * idx_hd * std::mem::size_of::<f32>(),
                );
                // PHASE 1 (DS4_CHUNK_DEFER_RESYNC): this CPU→GPU copy (IDX-N2) is
                // REDUNDANT — `nosync_idx_ring` and this `idx_ring` are the same
                // per-layer buffer (both index_comp_ring_or_alloc(layer_idx)), already
                // populated by compressor_prefill_idx. When deferring, the CPU mirror
                // isn't filled yet anyway, so SKIP the copy; gpu_rows is set in IDX-N1.
                if !(defer_chunk_resync()
                    && state.index_comp_ring_gpu_rows[layer_idx] >= n_index)
                {
                    let cpu = &state.index_comp_kv_ring[layer_idx];
                    let end = n_index as usize * idx_hd;
                    debug_assert!(cpu.len() >= end, "PHASE-B: idx ring CPU shorter than n_index");
                    unsafe {
                        std::ptr::copy_nonoverlapping(cpu.as_ptr(), idx_ring.contents() as *mut f32, end);
                    }
                    state.index_comp_ring_gpu_rows[layer_idx] = n_index;
                }
                let idx_ring_db = crate::deferred::DeferredBuf::from_external_buffer(
                    idx_ring, n_index as usize * idx_hd,
                );
                // (d) scores → top_k → mixed-attention flash over the whole chunk.
                let crossing = chunk_start + k_positions as u32 > raw_cap;
                let split_mask = if crossing { phaseb_split_mask() } else { 0 };
                if kp { chunk_kprof_add("2c_idxsetup", scope.commit_wait_gpu_us()); }
                if split_mask & 1 != 0 { scope.event_split("phaseb_pre_scores"); }
                let scores_db = scope.encode_indexer_scores_tiled(
                    &q_k, &weights_k, &idx_ring_db, n_comp as usize, k_positions,
                    idx_nhead, idx_hd, chunk_start, ratio,
                )?;
                if kp { chunk_kprof_add("2c_score", scope.commit_wait_gpu_us()); }
                let k_sel = (top_k as usize).min(n_comp as usize);
                if split_mask & 2 != 0 { scope.event_split("phaseb_pre_topk"); }
                let sel_db = scope.encode_indexer_topk_batched(
                    &scores_db, n_comp as usize, k_positions, k_sel,
                )?;
                if kp { chunk_kprof_add("2c_topk", scope.commit_wait_gpu_us()); }
                // DS4_CHUNK_SEL_DUMP=<layer>: GPU-copy this layer's per-token sel
                // rows (i32 [k_positions, k_sel]) into a persistent buffer
                // (key layer+5M), read after the terminal wait.
                if let Ok(v) = std::env::var("DS4_CHUNK_SEL_DUMP") {
                    if v.parse::<usize>().ok() == Some(layer_idx) && chunk_start == 0 {
                        let dbg = self.dispatcher.kv_buffer_or_alloc(
                            layer_idx as u32 + 5_000_000,
                            k_positions * k_sel * std::mem::size_of::<i32>());
                        scope.copy_buf_into(&sel_db, &dbg, 0);
                        self.sel_dump.set(Some((layer_idx, k_positions, k_sel)));
                    }
                }
                let sinks_db = scope.upload_f32(&layer.attn.attn_sinks);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                // Single chunk (chunk_start==0): kv_normed_rotated_k holds all K
                // positions linearly → raw ring of K rows, no wrap (raw_start=0,
                // raw_cap=K, n_raw=K); the SWA window is `raw_cap` (the function's
                // ring size = DS4_N_SWA in production), enforced by the kernel mask.
                let comp_ring_db = crate::deferred::DeferredBuf::from_external_buffer(
                    comp_ring.clone(), n_comp as usize * head_dim,
                );
                // BISECT confirming experiment (DS4_CHUNK_IDX_FP8_RAW=1): the idx
                // mixed-attention reads RAW KV from the full-f32 kv_normed_rotated_k,
                // whereas the per-token reference + the noidx core read FP8-SNAPPED raw
                // from the ring. Snap the chunk's raw KV to fp8 (into a scratch buffer
                // keyed by a fake layer) and feed THAT, to test whether the raw-precision
                // mismatch is the chunk-coherence root cause.
                let idx_fp8_raw = std::env::var("DS4_CHUNK_IDX_FP8_RAW").ok().as_deref() == Some("1");
                let snapped_raw = if idx_fp8_raw {
                    Some(scope.kv_fp8_store_persistent_k(
                        layer_idx as u32 + 7_000_000, &half.kv_normed_rotated_k, p,
                        k_positions as u32, 0, k_positions,
                    )?)
                } else {
                    None
                };
                let raw_ring_db = snapped_raw.as_ref().unwrap_or(&half.kv_normed_rotated_k);
                if split_mask & 4 != 0 { scope.event_split("phaseb_pre_mixattn"); }
                let heads = scope.encode_indexed_mixed_attention(
                    &half.q_heads_k, raw_ring_db, &comp_ring_db, &sel_db, &sinks_db,
                    k_positions, n_head, head_dim,
                    k_positions, n_comp as usize, k_sel, ratio,
                    raw_cap, chunk_start, 0, k_positions as u32, scale,
                )?;
                if split_mask & 8 != 0 { scope.event_split("phaseb_post_mixattn"); }
                scope.copy_buf_into(&heads, heads_k.buffer(), 0);
            }
            if kp { chunk_kprof_add("2b_flash_batched", scope.commit_wait_gpu_us()); }
        }

        // INCREMENT 2: batched rope-back over all K (512 dispatches → 1).
        if n_rot > 0 {
            scope.rope_tail_q_heads_in_place_k(&heads_k, n_head, head_dim, p, chunk_start, k_positions, true)?;
        }
        // DS4_CHUNK_HEADS_DUMP=<layer>: GPU-copy this layer's heads into a
        // persistent buffer (key layer+4M), read after the chunk drains (light,
        // one layer only — localizes the first corrupt position).
        if let Ok(v) = std::env::var("DS4_CHUNK_HEADS_DUMP") {
            if v.parse::<usize>().ok() == Some(layer_idx)
                && (chunk_start as usize + k_positions) <= 4096
            {
                let dbg = self.dispatcher.kv_buffer_or_alloc(
                    layer_idx as u32 + 4_000_000,
                    4096 * q_dim * std::mem::size_of::<f32>());
                scope.copy_buf_into(&heads_k, &dbg, chunk_start as usize * q_dim);
                self.heads_dump.set(Some((
                    layer_idx, chunk_start as usize + k_positions, q_dim,
                )));
            }
        }
        Ok(heads_k)
    }

    /// CHUNKED PREFILL (WIP) — one full transformer layer over a chunk of K
    /// consecutive prefill positions, fed the layer-input residual `prev_hc_k`
    /// ([K, n_hc*n_embd]). Returns the layer-output residual `after_ffn_k` (the
    /// next layer's `prev_hc_k`) + the deferred compressor finishes the caller
    /// must run AFTER the chunk's GPU work flushes (empty for raw / ratio==4
    /// idx layers — those finish inline; non-empty for ratio==128 noidx layers).
    ///
    /// Composition (mirrors the per-token resident layer, sbe.rs ~2023-2473,
    /// but with the matmuls K-batched and the attention sequential per-token):
    ///   1. encode_layer_attn_half_k  — hc_collapse + qkv chain + kv rms/rope
    ///      (batched over K)
    ///   2. sequential attention core, picked by compress_ratio:
    ///        ratio==0            → chunk_attn_core_raw
    ///        ratio==4 + indexer  → chunk_attn_core_comp_idx (inline per-quad
    ///                              finish/rotation + GPU-resident indexer sel)
    ///        else (ratio==128)   → chunk_attn_core_comp_noidx (deferred finishes)
    ///   3. encode_attn_output_matmuls_q8_k  — w_o_a/w_o_b output proj (batched)
    ///   4. encode_ffn_chain_k (DS4_MOE_K_PATH=mm_id) — hc_expand_attn +
    ///      hc_collapse + router + MoE + shared + hc_expand_add (batched)
    ///
    /// GUARD: hash-routed layers (`routing_table.is_some()`, the early L0/1/2)
    /// are NOT yet supported — encode_ffn_chain_k uses the learned router, and
    /// no K-batched per-token routing-table path exists. The driver must run
    /// those layers via the per-token decode path (or a future K hash-router).
    ///
    /// LEAN: uploads the f32 hc/router weights via weight_f32 — validate under
    /// DS4_LEAN_WEIGHTS=0 (f32 present). Lean-f16 hc is a follow-up (mirrors the
    /// staged-path lean fixes). Bit-equivalence target: chunk=1 == per-token.
    #[allow(clippy::too_many_arguments)]
    fn chunk_layer(
        &self,
        scope: &mut crate::deferred::BatchScope<'_>,
        layer: &ComposedLayerWeights,
        layer_idx: usize,
        p: &LayerParams,
        prev_hc_k: &crate::deferred::DeferredBuf,
        state: &mut AttnStepState,
        chunk_start: u32,
        k_positions: usize,
        raw_cap: u32,
        // Phase-A hash layers: the K chunk token ids (routing-table lookup). When the
        // layer has a routing_table + token_ids is Some, the FFN uses K-batched hash
        // routing; else top-k. token_ids=None on a hash layer → error.
        token_ids: Option<&[i32]>,
    ) -> Result<(crate::deferred::DeferredBuf, Vec<ChunkCompFinish>)> {
        let hash_route: Option<(&[i32], &[i32], usize)> = match (&layer.moe.routing_table, token_ids) {
            (Some(table), Some(tids)) => Some((table.as_slice(), tids, layer.moe.n_experts_used)),
            (Some(_), None) => anyhow::bail!(
                "chunk_layer: hash-routed layer {layer_idx} needs token_ids for K-batched hash routing"),
            (None, _) => None,
        };
        let n_hc = p.n_hc as usize;
        let n_embd = p.d_embd as usize;
        let n_lora_q = p.n_lora_q as usize;
        let n_head = p.n_head as usize;
        let head_dim = p.head_dim as usize;
        let kv_row = p.n_lora_kv as usize;
        let q_dim = n_head * head_dim;
        let n_groups = p.n_out_group as usize;
        let group_dim = q_dim / n_groups;
        // n_lora_o from w_o_a (f32 len, or q8 byte len → f32 count when lean).
        let w_o_a_len = if layer.attn.w_o_a.is_empty() {
            (layer.attn.w_o_a_q8.len() / 34) * 32
        } else {
            layer.attn.w_o_a.len()
        };
        let n_lora_o = w_o_a_len / (n_groups * group_dim);
        let out_low_dim = n_groups * n_lora_o;
        let ratio = p.compress_ratio;
        let sinkhorn_iters = p.hc_sinkhorn_iter as i32;
        let unit_gamma_hc = vec![1.0f32; n_hc * n_embd];

        if std::env::var("DS4_NORMED_TRACE").is_ok() && layer_idx == 4 {
            let scope2 = &mut *scope;
            let iv = scope2.debug_read(prev_hc_k);
            let sum: f64 = iv.iter().map(|&x| x as f64).sum();
            eprintln!("[INPUT chunkcore] L{layer_idx} cs={chunk_start} k={k_positions} prev_hc.sum={sum:.6} head={:?}", &iv[..4.min(iv.len())]);
        }
        let kprof = chunk_kprof_on();
        let mut kprof_t = std::time::Instant::now();
        // DS4_CHUNK_HALF_CHECK: also capture this layer's INPUT residual (prev_hc_k) into a
        // debug buffer (key = layer+2_000_000). If [PREVHC_NAN] fires, the cross-layer
        // residual handoff (prior layer's after_ffn_k) is already NaN → the race is in the
        // residual stream, upstream of this layer's half compute.
        if std::env::var("DS4_CHUNK_HALF_CHECK").is_ok() {
            let hc_dim = n_hc * n_embd;
            let dbg = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32 + 2_000_000, k_positions * hc_dim * std::mem::size_of::<f32>());
            scope.copy_buf_into(prev_hc_k, &dbg, 0);
        }
        // 1. attn-half (batched matmuls): hc_collapse_norm + qkv chain + kv rope.
        // Lean weights drop the f32 hc_fn copy (f16 retained) — bind whichever
        // exists; the K-batched collapse dispatches the matching kernel.
        let (hc_attn_fn_db, hc_attn_fn_is_f16) =
            scope.weight_hc(&layer.attn.hc_attn_fn, layer.attn.hc_attn_fn_f16());
        let hc_attn_scale_db = scope.weight_f32(&layer.attn.hc_attn_scale);
        let hc_attn_base_db = scope.weight_f32(&layer.attn.hc_attn_base);
        let hc_norm_gamma_db = scope.weight_f32(&layer.attn.hc_norm_gamma);
        let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);
        let gamma_q_db = scope.weight_f32(&layer.attn.qkv_gamma_q);
        let gamma_kv_db = scope.weight_f32(&layer.attn.qkv_gamma_kv);
        let q_a_q8_db = scope.weight_q8_0_raw(&layer.attn.attn_q_a_q8, n_lora_q * n_embd);
        let q_b_q8_db = scope.weight_q8_0_raw(&layer.attn.attn_q_b_q8, q_dim * n_lora_q);
        let kv_q8_db = scope.weight_q8_0_raw(&layer.attn.attn_kv_q8, kv_row * n_embd);
        let half = scope.encode_layer_attn_half_k(
            prev_hc_k,
            &hc_attn_fn_db, &hc_attn_scale_db, &hc_attn_base_db, &hc_norm_gamma_db,
            &unit_gamma_db,
            &q_a_q8_db, &gamma_q_db, &q_b_q8_db, &kv_q8_db, &gamma_kv_db,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            sinkhorn_iters, p.hc_eps, p.rms_eps, p, chunk_start, k_positions,
            hc_attn_fn_is_f16,
        )?;
        // DS4_CHUNK_HALF_CHECK: GPU-copy the half's kv_normed_rotated_k (the exact input
        // the KV store reads) into a persistent per-layer debug buffer (key = layer+1_000_000),
        // scanned after wait_all_and_drop in prefill_chunk. A GPU copy (no CPU read) → does
        // not force a sync, so async timing is preserved. Discriminates: NaN in the half
        // output (producer race upstream of the store) vs introduced at/after the store.
        if std::env::var("DS4_CHUNK_HALF_CHECK").is_ok() {
            let dbg = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32 + 1_000_000, k_positions * kv_row * std::mem::size_of::<f32>());
            scope.copy_buf_into(&half.kv_normed_rotated_k, &dbg, 0);
            let dbg_q = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32 + 10_000_000, k_positions * q_dim * std::mem::size_of::<f32>());
            scope.copy_buf_into(&half.q_heads_k, &dbg_q, 0);
        }
        if kprof {
            scope.commit_wait_stage("kprof_half");
            chunk_kprof_add("1_attn_half", kprof_t.elapsed().as_micros());
            kprof_t = std::time::Instant::now();
        }

        // 2. sequential attention core (per-token causal window + compressor).
        let (heads_k, finishes) = if ratio == 0 {
            (
                self.chunk_attn_core_raw(
                    scope, layer_idx as u32, p, &half, &layer.attn.attn_sinks,
                    chunk_start, k_positions, raw_cap,
                )?,
                Vec::new(),
            )
        } else if ratio == 4
            && layer.attn.has_indexer_compressor()
            && layer.attn.has_indexer_qb()
        {
            (
                self.chunk_attn_core_comp_idx(
                    scope, layer, layer_idx, p, &half, state,
                    chunk_start, k_positions, raw_cap,
                )?,
                Vec::new(),
            )
        } else if layer.attn.has_attn_compressor() {
            self.chunk_attn_core_comp_noidx(
                scope, layer, layer_idx, p, &half, state,
                chunk_start, k_positions, raw_cap,
            )?
        } else {
            // ratio!=0, not a full-indexer layer, and NO main compressor: the
            // hash-routed L0/1/2. decode_step gates the main compressor on
            // has_attn_compressor() and (no full indexer →) attends raw-only, so the
            // K-batched equivalent is the raw core. (Fixes the comp_noidx empty-weight
            // panic when folding Phase A into Phase B.)
            (
                self.chunk_attn_core_raw(
                    scope, layer_idx as u32, p, &half, &layer.attn.attn_sinks,
                    chunk_start, k_positions, raw_cap,
                )?,
                Vec::new(),
            )
        };

        if kprof {
            scope.commit_wait_stage("kprof_core");
            chunk_kprof_add("2_attn_core", kprof_t.elapsed().as_micros());
            kprof_t = std::time::Instant::now();
        }

        // 3. output proj (batched): w_o_a (group) → w_o_b (d_embd).
        let w_o_a_q8_db = scope.weight_q8_0_raw(&layer.attn.w_o_a_q8, out_low_dim * group_dim);
        let w_o_b_q8_db = scope.weight_q8_0_raw(&layer.attn.w_o_b_q8, n_embd * out_low_dim);
        let (_low_k, attn_out_k) = scope.encode_attn_output_matmuls_q8_k(
            &heads_k, &w_o_a_q8_db, &w_o_b_q8_db,
            n_groups, n_lora_o, group_dim, out_low_dim, n_embd, k_positions,
        )?;
        // DS4_CHUNK_HALF_CHECK: capture attn_out_k (key layer+8M) + heads_k (key
        // layer+9M) — GPU copies, scanned after the terminal wait — to bisect
        // output-proj vs FFN-chain as the NaN producer.
        if std::env::var("DS4_CHUNK_HALF_CHECK").is_ok() {
            let dbg = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32 + 8_000_000, k_positions * n_embd * std::mem::size_of::<f32>());
            scope.copy_buf_into(&attn_out_k, &dbg, 0);
            let dbg_h = self.dispatcher.kv_buffer_or_alloc(
                layer_idx as u32 + 9_000_000, k_positions * q_dim * std::mem::size_of::<f32>());
            scope.copy_buf_into(&heads_k, &dbg_h, 0);
        }
        if kprof {
            scope.commit_wait_stage("kprof_output");
            chunk_kprof_add("3_output_proj", kprof_t.elapsed().as_micros());
            kprof_t = std::time::Instant::now();
        }

        // 4. ffn chain (batched, mm_id MoE): hc_expand_attn → hc_collapse →
        //    router → MoE + shared → hc_expand_add → after_ffn_k.
        // DS4_CHUNK_FFN_COMMIT diagnostic: put the FFN in its own cb ("1" =
        // commit no-wait, "wait" = full drain) — bisects cb-size vs cross-cb.
        match std::env::var("DS4_CHUNK_FFN_COMMIT").ok().as_deref() {
            Some("1") => scope.commit_keep_open(),
            Some("wait") => scope.commit_wait_stage("chunk_ffn_split"),
            _ => {}
        }
        let (hc_ffn_fn_db, hc_ffn_fn_is_f16) =
            scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
        let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
        let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
        let hc_ffn_norm_gamma_db = scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
        let (w_router_db, w_router_is_f16) =
            scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
        let router_bias_db = scope.upload_f32(&layer.moe.router_bias);
        let after_ffn_k = scope.encode_ffn_chain_k(
            &attn_out_k, prev_hc_k, &half.split_k,
            &hc_ffn_fn_db, hc_ffn_fn_is_f16, &hc_ffn_scale_db, &hc_ffn_base_db, &hc_ffn_norm_gamma_db,
            &unit_gamma_db,
            &w_router_db, w_router_is_f16, &router_bias_db,
            &layer.attn.w_shared_gate, &layer.attn.w_shared_up, &layer.attn.w_shared_down,
            &layer.attn.w_shared_gate_q8, &layer.attn.w_shared_up_q8, &layer.attn.w_shared_down_q8,
            n_hc, n_embd, layer.moe.n_experts, layer.moe.d_ffn.map_or(0, |n| n.get()), layer.attn.shared_dim,
            sinkhorn_iters, p.hc_eps, p.rms_eps, layer_idx as u32, k_positions,
            hash_route,
        )?;
        if kprof {
            scope.commit_wait_stage("kprof_ffn");
            chunk_kprof_add("4_ffn_moe", kprof_t.elapsed().as_micros());
        }

        Ok((after_ffn_k, finishes))
    }

    /// TEST-ONLY isolation entry: run ONE `chunk_layer` for `layer_idx` on a
    /// caller-provided (freshly-`AttnStepState::new`'d) state + a RANDOM prev_hc_k,
    /// in a single fresh scope, and return the layer output `after_ffn_k` for a
    /// garbage/NaN check. A single chunk_layer IS the full attn+FFN+MoE composition
    /// in one BatchScope — so this reproduces the in-chunk MoE fault iff the trigger
    /// is intra-chunk_layer (vs the cross-layer / Phase-A context). Honors
    /// DS4_MOE_K_PATH. NOT for production — bypasses Phase A and the multi-layer
    /// chaining; for the chunk-prefill MoE-fault bisect only.
    pub fn debug_run_one_chunk_layer(
        &self,
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        layer_idx: usize,
        k_positions: usize,
        seed: u32,
    ) -> Result<Vec<f32>> {
        let layer = &model.layers[layer_idx];
        let p = &layer.attn.params;
        let n_hc = p.n_hc as usize;
        let n_embd = p.d_embd as usize;
        let hc_dim = n_hc * n_embd;
        let mut rng = seed | 1;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let prev_hc: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5)
            .collect();
        let mut scope = self.dispatcher.batch_scope();
        let prev_hc_k = scope.upload_f32(&prev_hc);
        let (after_ffn_k, _finishes) = self.chunk_layer(
            &mut scope, layer, layer_idx, p, &prev_hc_k, state, 0, k_positions, self.raw_cap, None,
        )?;
        Ok(scope.flush_and_read(&after_ffn_k))
    }

    /// CHUNKED PREFILL (WIP, DS4_PREFILL_CHUNK) — process ONE chunk of
    /// `k_positions` consecutive prompt positions [chunk_start, chunk_start+K),
    /// advancing the KV cache + compressor/indexer state for ALL layers. NO
    /// lm-head (the caller does the last prompt token via the per-token feed).
    ///
    /// Two phases, because the early hash-routed layers (`routing_table`, a
    /// contiguous L0..n_hash prefix) have no K-batched router:
    ///   Phase A — layers [0, n_hash): the existing per-token first half
    ///     (`encode_first_half(l_split=n_hash)`) run K times (sets
    ///     CURRENT_TOKEN_HINT + state.pos per token, advances kv_pos/n_comp for
    ///     those layers). Each token's residual after layer n_hash-1 is captured
    ///     and stacked into `prev_hc` [K, n_hc*n_embd].
    ///   Phase B — layers [n_hash, N): one BatchScope, `chunk_layer` per layer,
    ///     threading `prev_hc_k` → `after_ffn_k`. Deferred (ratio==128) finishes
    ///     are collected and run after a single commit_wait (idx layers finish
    ///     inline). Sets kv_pos[L] = chunk_start+K.
    ///
    /// On no hash prefix (n_hash==0), Phase A is just the HC-expand of the
    /// embeddings (n_hc copies each) + the pos==0 pool reset.
    ///
    /// Bit-equivalence target: a sequence prefilled via prefill_chunk (any K)
    /// reaches the SAME decode state as the per-token prefill_step path.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk(
        &self,
        embeds_k: &[f32],     // [k_positions * n_embd] token embeddings
        token_ids_k: &[i32],  // [k_positions] token ids (hash-routing hint)
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        chunk_start: u32,
        k_positions: usize,
    ) -> Result<()> {
        let n_layers = model.layers.len();
        if n_layers == 0 || k_positions == 0 {
            return Ok(());
        }
        // Scope the antirez single-encoder to this chunk prefill only (no-op unless
        // DS4_CHUNK_SHARED_ENC=1). Ends the shared encoder + deactivates on drop so
        // the following decode chain keeps its own encoder handling.
        let _shared_guard = crate::macos::shared_active_guard();
        let first = &model.layers[0].attn.params;
        let n_hc = first.n_hc as usize;
        let n_embd = first.d_embd as usize;
        let hc_dim = n_hc * n_embd;
        let raw_cap = self.raw_cap;
        anyhow::ensure!(
            embeds_k.len() == k_positions * n_embd,
            "prefill_chunk: embeds_k {} != k*n_embd = {}*{}",
            embeds_k.len(), k_positions, n_embd
        );
        anyhow::ensure!(
            token_ids_k.len() == k_positions,
            "prefill_chunk: token_ids_k {} != k {}", token_ids_k.len(), k_positions
        );

        let async_commit = std::env::var("DS4_CHUNK_COMMIT_ASYNC").ok().as_deref() == Some("1");
        let layer_flush = std::env::var("DS4_CHUNK_LAYER_FLUSH").ok().as_deref() == Some("1");
        // Async cb chaining (no per-layer GPU wait) requires fresh flash scratch for
        // the WHOLE chunk prefill — Phase A (per-token hash layers) AND Phase B. The
        // per-layer flash-scratch REUSE (global flash_scratch_layer atomic) aliases a
        // layer's grow-only scratch across in-flight, not-yet-waited flashes. Proven:
        // global reuse-off (DS4_FLASH_SCRATCH_REUSE=0) → async deterministic + sync-
        // faithful; Phase-B-only suppression was insufficient (NaN/nondeterministic),
        // so the suppression must wrap Phase A too. A Drop guard restores reuse on
        // every exit (incl. `?`). The hot K=1 per-token decode chain never aliases
        // (one flash per layer) → it keeps reuse.
        struct ScratchSuppressGuard<'a>(&'a crate::MetalDispatcher, bool);
        impl Drop for ScratchSuppressGuard<'_> {
            fn drop(&mut self) {
                if self.1 {
                    self.0.set_flash_scratch_suppress(false);
                    self.0.set_moe_scratch_suppress(false);
                }
            }
        }
        let suppress_reuse = async_commit && !layer_flush;
        if suppress_reuse {
            self.dispatcher.set_flash_scratch_suppress(true);
            // The pooled MoE scratch is shared across all layers and relies on a
            // per-call GPU wait; async removes it → fresh per-call scratch instead.
            self.dispatcher.set_moe_scratch_suppress(true);
        }
        let _scratch_guard = ScratchSuppressGuard(self.dispatcher, suppress_reuse);

        // The fused HC-collapse default is now UNCAPPED (coherent at full K under
        // SYNC — recovers prefill @3000 to ~115 tok/s, see deferred.rs hc_collapse).
        // But the uncapped fused path RACES under async commit (~1/6 BOS). When async
        // is requested and the caller did NOT pin the cap explicitly, re-cap to 128
        // (the safe value: per-K fallback above it) for the duration of this prefill.
        // A Drop guard restores the prior env on every exit (incl. `?`).
        struct FuseCapGuard(Option<String>, bool);
        impl Drop for FuseCapGuard {
            fn drop(&mut self) {
                if self.1 {
                    match &self.0 {
                        Some(v) => std::env::set_var("DS4_HC_COLLAPSE_K_FUSE_MAX", v),
                        None => std::env::remove_var("DS4_HC_COLLAPSE_K_FUSE_MAX"),
                    }
                }
            }
        }
        let recap_async = async_commit
            && std::env::var("DS4_HC_COLLAPSE_K_FUSE_MAX").is_err();
        let _fuse_cap_guard = FuseCapGuard(
            std::env::var("DS4_HC_COLLAPSE_K_FUSE_MAX").ok(),
            recap_async,
        );
        if recap_async {
            std::env::set_var("DS4_HC_COLLAPSE_K_FUSE_MAX", "128");
        }

        // Hash-routed layers must form a contiguous L0 prefix (the simple
        // Phase A/B split). DS4 V4-Flash: L0/1/2.
        let mut n_hash = 0usize;
        while n_hash < n_layers && model.layers[n_hash].moe.routing_table.is_some() {
            n_hash += 1;
        }
        for l in n_hash..n_layers {
            anyhow::ensure!(
                model.layers[l].moe.routing_table.is_none(),
                "prefill_chunk: hash-routed layer {l} after non-hash prefix — the chunk \
                 driver needs hash layers as a contiguous L0 prefix"
            );
        }

        let dbg = std::env::var("DS4_CHUNK_DEBUG").is_ok();
        if dbg {
            eprintln!(
                "[chunk] start={chunk_start} k={k_positions} n_hash={n_hash} n_layers={n_layers}"
            );
        }

        // ── Phase A: hash prefix per-token → stacked residual. ──
        let kprof = chunk_kprof_on();
        let kprof_ta = std::time::Instant::now();
        // DS4_CHUNK_PHASE_A_BATCH: fold the per-token hash layers L0..n_hash into the
        // K-batched Phase B (chunk_layer with K-batched hash routing) instead of
        // encode_first_half per token (~16% of prefill). NOT byte-identical (K-batched
        // attn/MoE differs from per-token at fp32) → quality-gated.
        let phase_a_batch = n_hash > 0
            && std::env::var("DS4_CHUNK_PHASE_A_BATCH").ok().as_deref() == Some("1");
        let b_start = if phase_a_batch { 0 } else { n_hash };
        let mut prev_hc = vec![0.0f32; k_positions * hc_dim];
        if n_hash > 0 && !phase_a_batch {
            for i in 0..k_positions {
                let embed_i = &embeds_k[i * n_embd..(i + 1) * n_embd];
                state.pos = chunk_start + i as u32;
                ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT
                    .with(|c| c.set(token_ids_k[i]));
                if dbg { eprintln!("[chunk] phaseA token {i} (pos {})", chunk_start + i as u32); }
                self.encode_first_half(embed_i, model, state, LayerCutpoint(n_hash))?;
                prev_hc[i * hc_dim..(i + 1) * hc_dim].copy_from_slice(&state.cur_hc);
            }
            if dbg { eprintln!("[chunk] phaseA done"); }
        } else {
            // b_start==0 (no hash prefix, or phase_a_batch): Phase B starts from L0 with
            // the embeddings HC-expanded (each slot a copy of the embedding).
            if chunk_start == 0 {
                self.dispatcher.reset_decode_state_pools();
            }
            for i in 0..k_positions {
                let embed_i = &embeds_k[i * n_embd..(i + 1) * n_embd];
                for h in 0..n_hc {
                    let o = i * hc_dim + h * n_embd;
                    prev_hc[o..o + n_embd].copy_from_slice(embed_i);
                }
            }
        }

        if kprof {
            chunk_kprof_add("0_phaseA", kprof_ta.elapsed().as_micros());
        }

        // ── Phase B: layers [n_hash, N) K-batched via chunk_layer. ──
        // Commit policy. TWO distinct failure modes were found and characterized:
        //   - async cb chaining (commit_keep_open, no per-layer wait) used to be a
        //     NONDETERMINISTIC race (output varied run-to-run). ROOT CAUSE: the
        //     per-layer flash-scratch REUSE (the global flash_scratch_layer atomic)
        //     aliased a layer's grow-only scratch across the K per-position flashes
        //     while the prior flash was still in-flight. FIXED by suppressing scratch
        //     reuse for the async Phase-B loop (set_flash_scratch_suppress below) —
        //     proven via chunk_async_logit_closeness (async run1==run2, rel_L2 0.0 vs
        //     sync, cos=1.0). Localized with the DS4_FLASH_SCRATCH_REUSE=0 A/B.
        //   - commit_every>=8 (SYNC or async) → DETERMINISTIC all-zeros: a cb
        //     command-count limit at ~8 heavy mm_id layers per cb (assert_cb_ok does
        //     NOT catch it — the cb yields zeroed outputs without status=Error). Its
        //     DS4_PREFILL_TTFT "speedup" is a MIRAGE (that test never checks output).
        //     So commit_every MUST stay 1. async (commit_every=1) is now correct AND
        //     faster — the GPU queue stays fed instead of draining ~40×/chunk.
        let commit_every: usize = std::env::var("DS4_CHUNK_COMMIT_EVERY")
            .ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(1);
        // commit_every>=8 SYNC is still a DETERMINISTIC all-zeros (a cb command-count
        // limit at ~8 heavy mm_id layers/cb; assert_cb_ok doesn't catch it). async_commit
        // is NOW CORRECT (the per-layer flash-scratch REUSE race that made it
        // nondeterministic is fixed by suppressing reuse below — proven via
        // chunk_async_logit_closeness: async run1==run2, byte-identical to sync).
        if commit_every != 1 && !layer_flush {
            eprintln!(
                "ds4 chunk-prefill: WARNING — DS4_CHUNK_COMMIT_EVERY={commit_every} is an \
                 INVESTIGATION-ONLY config and produces INCORRECT output (all-zeros at a \
                 cb command-count limit). Use commit_every=1 (the default)."
            );
        }
        let mut scope = self.dispatcher.batch_scope();
        let mut prev_hc_k = scope.upload_f32(&prev_hc);
        let mut all_finishes: Vec<ChunkCompFinish> = Vec::new();
        for layer_idx in b_start..n_layers {
            let layer = &model.layers[layer_idx];
            let p = &layer.attn.params;
            if dbg {
                eprintln!(
                    "[chunk] phaseB layer {layer_idx} ratio={} idx={}",
                    p.compress_ratio,
                    layer.attn.has_indexer_compressor() && layer.attn.has_indexer_qb()
                );
            }
            // token_ids only needed by hash layers (chunk_layer derives hash routing
            // from layer.moe.routing_table + these); harmless for non-hash layers.
            let tids = if phase_a_batch { Some(&token_ids_k[..]) } else { None };
            let (after_ffn_k, finishes) = self.chunk_layer(
                &mut scope, layer, layer_idx, p, &prev_hc_k, state,
                chunk_start, k_positions, raw_cap, tids,
            )?;
            prev_hc_k = after_ffn_k;
            // DS4_CHUNK_RESID_SUM: per-layer per-position residual checksums for the
            // first chunk — diff chunk=128 vs chunk=256 to localize the first
            // diverging (layer, position). Diagnostic only (forces a per-layer wait).
            if std::env::var("DS4_CHUNK_RESID_SUM").is_ok() {
                scope.commit_wait_stage("resid_sum");
                let v = scope.debug_read(&prev_hc_k);
                let row = hc_dim;
                let mut sums = String::new();
                for i in 0..k_positions.min(512) {
                    let s: f64 = v[i * row..(i + 1) * row].iter().map(|&x| x as f64).sum();
                    sums.push_str(&format!("{s:.6} "));
                }
                eprintln!("[RESID c={chunk_start}] L{layer_idx} {sums}");
            }
            // DS4_CHUNK_RESID_GPUCAP: per-layer residual capture for the first
            // chunk — GPU-copies after_ffn_k into a persistent per-layer debug
            // buffer (key layer+3_000_000), dumped after the terminal wait. No CPU
            // wait at encode time, BUT the extra ~5 MB blit/layer measurably
            // PERTURBS the schedule (it flipped even chunk=128 to NaN pre-fix) —
            // treat as load amplifier, not a faithful full-speed probe.
            if std::env::var("DS4_CHUNK_RESID_GPUCAP").is_ok() && chunk_start == 0 {
                let dbg = self.dispatcher.kv_buffer_or_alloc(
                    layer_idx as u32 + 3_000_000,
                    k_positions * hc_dim * std::mem::size_of::<f32>());
                scope.copy_buf_into(&prev_hc_k, &dbg, 0);
            }
            all_finishes.extend(finishes);
            state.kv_pos[layer_idx] = chunk_start + k_positions as u32;
            // split the cb every `commit_every` layers (not on the last — the
            // terminal wait_all_and_drop commits the tail).
            let at_boundary = (layer_idx - b_start + 1) % commit_every == 0;
            // Task 0 diagnostic (docs/PREFILL_GPU_RESIDENT_PLAN.md): event-ordered
            // split (commit, signal MTLEvent, next cb waits GPU-side → no CPU wait,
            // no bubble, GPU-ordered). DS4_CHUNK_EVENT_ONLY=K → event_split ONLY at
            // the boundary after relative layer K (single split, rest sync drain) to
            // isolate a per-boundary hazard from a many-split resource fault.
            // DS4_CHUNK_COMMIT_EVENT=1 → all boundaries.
            let rel = layer_idx - b_start;
            let event_here = !layer_flush && (
                std::env::var("DS4_CHUNK_COMMIT_EVENT").ok().as_deref() == Some("1")
                || std::env::var("DS4_CHUNK_EVENT_ONLY").ok()
                    .and_then(|v| v.parse::<usize>().ok()) == Some(rel)
                // EVENT_FIRST=N → event_split the first N boundaries (sync after) to
                // bisect the resource-fault threshold.
                || std::env::var("DS4_CHUNK_EVENT_FIRST").ok()
                    .and_then(|v| v.parse::<usize>().ok()).is_some_and(|n| rel < n)
            );
            if layer_idx + 1 < n_layers && (layer_flush || at_boundary) {
                if async_commit && !layer_flush {
                    scope.commit_keep_open(); // async chain (no wait); scratch-reuse suppressed above
                } else if event_here {
                    // Task 0: in-flight event cbs > ~8-11 fault to all-zeros (resource
                    // limit, NOT a CPU-sync hazard); a full mid-chunk drain_all corrupts
                    // the chain. DS4_CHUNK_EVENT_WINDOW=W (default 6) uses a SLIDING wait
                    // — bound in-flight ≤ W by waiting the OLDEST cb (already done by
                    // then → near-zero bubble). The antirez-style bounded pipeline.
                    let window: usize = std::env::var("DS4_CHUNK_EVENT_WINDOW")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(6);
                    scope.event_split_windowed("chunk_commit_event", window);
                } else if chunk_start + k_positions as u32 > raw_cap
                    && std::env::var("DS4_CHUNK_DRAIN_ALL").ok().as_deref() == Some("1")
                {
                    // Crossing chunks (chunk_start+K > raw_cap): the crossing
                    // event splits push cbs into pending_cbs that
                    // commit_wait_stage never CPU-waits — pending heavy cbs
                    // accumulate across layers and silently fault (zeroed
                    // outputs, no status error) → the deterministic BOS
                    // corruption. Drain EVERY committed cb at the layer
                    // boundary. Non-crossing chunks are unchanged (no splits).
                    scope.drain_all_stage("chunk_drain_all");
                } else if std::env::var("DS4_CHUNK_PIPELINE").ok().as_deref() == Some("1") {
                    // 1-deep pipeline: overlap the next-cb encode with this cb's GPU run,
                    // 1-cb-in-flight (no residency collapse). Recovers the ~15% inter-cb bubble.
                    scope.commit_pipelined("chunk_pipeline");
                } else {
                    scope.commit_wait_stage("chunk_commit_sync"); // correct (drains cb)
                }
            }
        }
        if dbg { eprintln!("[chunk] phaseB done, commit_wait + finishes"); }
        // Terminal commit+wait that CONSUMES the scope (commits the current cb +
        // waits all). MUST be a consuming terminal, NOT commit_wait_stage —
        // commit_wait_stage installs a fresh uncommitted cb after committing the
        // old one, and with no Drop-commit on BatchScope that cb leaks a Metal
        // command-queue slot per chunk. The queue caps at 64 in-flight cbs, so
        // ~64 chunks in, new_command_buffer() blocks forever (the pos≈63 hang).
        // wait_all_and_drop leaves nothing dangling. The deferred (ratio==128)
        // finishes then read the GPU-resident emit rows (GPU already waited;
        // emit_db buffers outlive the scope) — finish_gpu makes them CPU-only.
        scope.wait_all_and_drop();
        // DS4_CHUNK_HEADS_DUMP read-out (one layer, after the terminal wait).
        if let Some((l, kp_cap, q_dim)) = self.heads_dump.take() {
            let dbg = self.dispatcher.kv_buffer_or_alloc(
                l as u32 + 4_000_000, 4096 * q_dim * std::mem::size_of::<f32>());
            let v = unsafe {
                std::slice::from_raw_parts(dbg.contents() as *const f32, kp_cap * q_dim)
            };
            let mut sums = String::new();
            for i in 0..kp_cap.min(256) {
                let s: f64 = v[i * q_dim..(i + 1) * q_dim].iter().map(|&x| x as f64).sum();
                sums.push_str(&format!("{s:.6} "));
            }
            eprintln!("[HEADS] L{l} {sums}");
            // Full per-position vectors → file (raw f32 LE) for offline diff
            // against the per-token DS4_PT_HEADS_DUMP capture.
            if let Ok(f) = std::env::var("DS4_CHUNK_HEADS_FILE") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                };
                std::fs::write(&f, bytes).expect("write DS4_CHUNK_HEADS_FILE");
                eprintln!("[HEADS] wrote {} ({} pos x {} f32)", f, kp_cap, q_dim);
            }
        }
        // DS4_CHUNK_SEL_DUMP read-out: per-position selected comp rows.
        if let Some((l, kp_cap, k_sel)) = self.sel_dump.take() {
            let dbg = self.dispatcher.kv_buffer_or_alloc(
                l as u32 + 5_000_000, kp_cap * k_sel * std::mem::size_of::<i32>());
            let v = unsafe {
                std::slice::from_raw_parts(dbg.contents() as *const i32, kp_cap * k_sel)
            };
            let ratio = 4usize;
            for i in 0..kp_cap {
                let visible = (i + 1) / ratio; // chunk_start==0
                let row = &v[i * k_sel..(i + 1) * k_sel];
                // The mixed-attn kernel iterates the row and BREAKS on the first
                // idx >= visible (skipping idx<0). Reproduce: collect attended set.
                let mut attended: Vec<i32> = Vec::new();
                for &idx in row {
                    if idx < 0 { continue; }
                    if idx as usize >= visible { break; }
                    attended.push(idx);
                }
                attended.sort_unstable();
                let full = attended.len() == visible
                    && attended.iter().enumerate().all(|(j, &x)| x as usize == j);
                if !full {
                    eprintln!(
                        "[SEL] L{l} pos={i} visible={visible} attended={}/{} MISSING (row head={:?})",
                        attended.len(), visible, &row[..k_sel.min(8)]);
                }
            }
            eprintln!("[SEL] L{l} coverage check done ({kp_cap} pos, k_sel={k_sel})");
        }
        // DS4_CHUNK_RESID_GPUCAP dump (after the terminal wait — full-speed capture).
        if std::env::var("DS4_CHUNK_RESID_GPUCAP").is_ok() && chunk_start == 0 {
            for l in b_start..n_layers {
                let dbg = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 3_000_000, k_positions * hc_dim * std::mem::size_of::<f32>());
                let v = unsafe {
                    std::slice::from_raw_parts(dbg.contents() as *const f32, k_positions * hc_dim)
                };
                let mut sums = String::new();
                for i in 0..k_positions.min(256) {
                    let s: f64 = v[i * hc_dim..(i + 1) * hc_dim].iter().map(|&x| x as f64).sum();
                    sums.push_str(&format!("{s:.6} "));
                }
                eprintln!("[RESID] L{l} {sums}");
            }
        }
        // STAGE 1 fused prefill finishes FIRST (they advance n_comp + comp_kv_ring
        // for the noidx layers); the per-position finishes (other layers) follow.
        // The two sets are over DISJOINT layers (a layer uses one path or the
        // other), so the order between them is immaterial — but applying the fused
        // ones here keeps the n_comp==comp_row0 invariant check meaningful.
        self.apply_comp_prefill_finishes(state);
        // PHASE 1: replay the deferred nosync compressor CPU-mirror resyncs now that
        // the terminal wait_all_and_drop has flushed the per-layer pools + rings.
        self.apply_chunk_resyncs(state);
        self.run_chunk_compressor_finishes(model, state, all_finishes);

        // DS4_CHUNK_HALF_CHECK: scan the per-layer half-output debug buffers (copied in
        // chunk_layer) for non-finite values. If [HALF_NAN] fires at the same chunk/layer
        // as the KV-cache [NAN], the NaN is already in half.kv_normed_rotated_k (a producer
        // race in the attention half: qkv matmul / rms / rope). If [NAN] fires but
        // [HALF_NAN] does NOT, the half output is clean post-completion and the store read
        // it early / the cache was corrupted otherwise (cross-cb / store-context).
        if std::env::var("DS4_CHUNK_HALF_CHECK").is_ok() {
            let hc_dim = n_hc * n_embd;
            // q_heads (10M, pre-core) → heads (9M) → attn_out (8M): bisect.
            for l in b_start..n_layers {
                let lp = &model.layers[l].attn.params;
                let qd = lp.n_head as usize * lp.head_dim as usize;
                let dbg_q = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 10_000_000, k_positions * qd * std::mem::size_of::<f32>());
                let sq = unsafe {
                    std::slice::from_raw_parts(dbg_q.contents() as *const f32, k_positions * qd)
                };
                if let Some(pp) = sq.iter().position(|v| !v.is_finite()) {
                    eprintln!("[QHEADS_NAN] chunk_start={chunk_start} layer={l} idx={pp} (row {}, col {})",
                        pp / qd, pp % qd);
                    break;
                }
                let dbg_h = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 9_000_000, k_positions * qd * std::mem::size_of::<f32>());
                let sh = unsafe {
                    std::slice::from_raw_parts(dbg_h.contents() as *const f32, k_positions * qd)
                };
                if let Some(pp) = sh.iter().position(|v| !v.is_finite()) {
                    eprintln!("[HEADS_NAN] chunk_start={chunk_start} layer={l} idx={pp} (row {}, col {})",
                        pp / qd, pp % qd);
                    break;
                }
                let ne = lp.d_embd as usize;
                let dbg_o = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 8_000_000, k_positions * ne * std::mem::size_of::<f32>());
                let so = unsafe {
                    std::slice::from_raw_parts(dbg_o.contents() as *const f32, k_positions * ne)
                };
                if let Some(pp) = so.iter().position(|v| !v.is_finite()) {
                    eprintln!("[ATTNOUT_NAN] chunk_start={chunk_start} layer={l} idx={pp} (row {}, col {})",
                        pp / ne, pp % ne);
                    break;
                }
            }
            // prev_hc (residual input) first — the upstream-most signal.
            for l in b_start..n_layers {
                let dbg = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 2_000_000, k_positions * hc_dim * std::mem::size_of::<f32>());
                let s = unsafe {
                    std::slice::from_raw_parts(dbg.contents() as *const f32, k_positions * hc_dim)
                };
                if let Some(pp) = s.iter().position(|v| !v.is_finite()) {
                    eprintln!("[PREVHC_NAN] chunk_start={chunk_start} layer={l} idx={pp} (row {}, col {})",
                        pp / hc_dim, pp % hc_dim);
                    break;
                }
            }
            for l in b_start..n_layers {
                let lp = &model.layers[l].attn.params;
                let kvr = lp.n_lora_kv as usize;
                let dbg = self.dispatcher.kv_buffer_or_alloc(
                    l as u32 + 1_000_000, k_positions * kvr * std::mem::size_of::<f32>());
                let s = unsafe {
                    std::slice::from_raw_parts(dbg.contents() as *const f32, k_positions * kvr)
                };
                if let Some(pp) = s.iter().position(|v| !v.is_finite()) {
                    eprintln!("[HALF_NAN] chunk_start={chunk_start} layer={l} idx={pp} (row {}, col {})",
                        pp / kvr, pp % kvr);
                    break;
                }
            }
        }

        // DS4_CHUNK_NAN_CHECK: bisect the async 512-tok NaN. Runs AFTER wait_all_and_drop
        // (GPU synced) so it does NOT alter the async-within-chunk timing. Scans each
        // layer's persistent KV cache (written region) + comp ring (n_comp rows) for
        // non-finite values and reports the FIRST bad (chunk_start, layer, kind).
        if std::env::var("DS4_CHUNK_NAN_CHECK").is_ok() {
            let n_raw_valid = (chunk_start + k_positions as u32).min(raw_cap) as usize;
            'scan: for l in 0..n_layers {
                let lp = &model.layers[l].attn.params;
                let row = lp.n_lora_kv as usize;
                let hd = lp.head_dim as usize;
                // KV cache (raw_cap*row f32), valid rows [0..n_raw_valid).
                let kv_buf = self.dispatcher.kv_buffer_or_alloc(
                    l as u32, raw_cap as usize * row * std::mem::size_of::<f32>());
                let kv = unsafe {
                    std::slice::from_raw_parts(kv_buf.contents() as *const f32, n_raw_valid * row)
                };
                if let Some(p) = kv.iter().position(|v| !v.is_finite()) {
                    eprintln!("[NAN] chunk_start={chunk_start} layer={l} kind=KV idx={p} (row {}, of {} valid rows)",
                        p / row, n_raw_valid);
                    break 'scan;
                }
                // comp ring (comp_ring_rows*hd f32), valid rows [0..n_comp).
                let ncomp = state.n_comp[l] as usize;
                if ncomp > 0 {
                    let cr_buf = self.dispatcher.comp_ring_or_alloc(
                        l as u32, comp_ring_rows(raw_cap) * hd * std::mem::size_of::<f32>());
                    let cr = unsafe {
                        std::slice::from_raw_parts(cr_buf.contents() as *const f32, ncomp * hd)
                    };
                    if let Some(p) = cr.iter().position(|v| !v.is_finite()) {
                        eprintln!("[NAN] chunk_start={chunk_start} layer={l} kind=COMP idx={p} (row {}, of {ncomp} comp rows)",
                            p / hd);
                        break 'scan;
                    }
                }
            }
        }

        state.pos = chunk_start + k_positions as u32;
        if kprof {
            chunk_kprof_dump();
        }
        Ok(())
    }

    fn encode_first_half_inner(
        &self,
        x: &[f32],
        model: &ComposedModelWeights,
        state: &mut AttnStepState,
        l_split: LayerCutpoint,
        with_ffn_half: bool,
        resident_out: Option<&mut Option<ResidentChain>>,
    ) -> Result<FirstHalfOutputs> {
        let t_token = std::time::Instant::now();
        if l_split.0 > model.layers.len() {
            bail!(
                "encode_first_half: l_split={} exceeds n_layers={}",
                l_split.0,
                model.layers.len()
            );
        }
        if l_split.0 == 0 {
            return Ok(FirstHalfOutputs {
                per_layer: Vec::new(),
            });
        }

        let first = &model.layers[0].attn.params;
        let n_hc = first.n_hc as usize;
        let d_embd = first.d_embd as usize;
        if x.len() != d_embd {
            bail!(
                "encode_first_half: x has {} elements, expected d_embd={}",
                x.len(),
                d_embd
            );
        }

        // Initial HC expansion: every slot gets a copy of x. Matches
        // `decode_step_with_attn_to_residual`'s setup (ds4_engine
        // decode_step.rs:671-675).
        let mut cur_hc = vec![0.0f32; n_hc * d_embd];
        for h in 0..n_hc {
            cur_hc[h * d_embd..(h + 1) * d_embd].copy_from_slice(x);
        }
        state.cur_hc = cur_hc;

        let unit_gamma_hc = vec![1.0f32; n_hc * d_embd];
        let raw_cap = self.raw_cap;
        let pos = state.pos;

        // DS4_STARTUP_PROBE: measure the per-token timeline in one clock to
        // locate the commit→GPU-start latency.
        let startup_probe = std::env::var("DS4_STARTUP_PROBE").is_ok();
        let token_start_secs = if startup_probe { mach_now_secs() } else { 0.0 };
        let mut first_commit_secs: Option<f64> = None;

        // Sequence start: re-init the persistent GPU compressor/indexer state
        // pools (kv→0, score→-1e9). They live on the dispatcher and are filled
        // only on first alloc, so a fresh sequence reusing the same dispatcher
        // would otherwise inherit the prior sequence's window — stale scores
        // un-mask not-yet-written slots in the pooled softmax, causing
        // run-to-run divergence in the long-context (compressor-active) decode.
        if pos == 0 {
            self.dispatcher.reset_decode_state_pools();
        }

        let mut per_layer = Vec::with_capacity(l_split.0);
        let stage_prof = std::env::var("DS4_STAGE_PROFILE").ok().as_deref() == Some("1");
        let (mut t_attn, mut t_comp, mut t_ffn) = (0.0f64, 0.0f64, 0.0f64);
        // ffn-stage sub-timers (DS4_STAGE_PROFILE): split the t_ffn bucket
        // into flash-attn output-proj vs the actual MoE/ffn-half, plus a
        // count of fused bridge layers (which can't be split).
        let (mut t_attnout, mut t_moe) = (0.0f64, 0.0f64);
        let (mut n_bridge, mut n_split) = (0u32, 0u32);

        // ── Layer chaining (step 10, DS4_CHAIN=1). When on, the merged
        //   bridge/no-comp-row branches commit their cb WITHOUT waiting and
        //   keep `cur_hc` GPU-resident (`chain_hc`): the next layer binds it as
        //   prev_hc + output_proj residual, so the GPU queue stays full across
        //   layers (no per-layer commit+wait idle). All compressor finishes
        //   (pool resync / rotation / comp_kv_ring append) defer to token end,
        //   after a single wait on `chain_cbs`. If a layer can't take a merged
        //   branch, the chain is materialized first (wait + read cur_hc).
        // SSD-streaming: expert ids must be CPU-known before each MoE encode
        // (cache ensure + slot remap), so force the readback (non-chain) paths.
        let ssd_stream = std::env::var("DS4_SSD_STREAM").is_ok();
        let chain_on =
            std::env::var("DS4_CHAIN").ok().as_deref() != Some("0") && !ssd_stream;
        // DS4_CHAIN_MAX_LAYERS=N (diagnostic bisect): layers < N take their
        // normal chained/merged branches; layers >= N are forced staged.
        let chain_max: usize = std::env::var("DS4_CHAIN_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        // DS4_CHAIN_HASH (default ON; =0 reverts): let hash-routed layers
        // (V4-Flash 0/1/2, routing_table.is_some()) take the chained
        // bridge_merge/merge_raw branches too, instead of the slow synchronous
        // staged path. Their expert ids are a per-token routing-table lookup
        // (CPU, upfront) and weights come from the GPU router probs via
        // router_weights_one — no mid-layer readback, so they chain. Shrinks
        // the staged head (~11→7.5ms), +6% tok/s, rel 0.118 vs reference.
        let hash_chain_on =
            std::env::var("DS4_CHAIN_HASH").ok().as_deref() != Some("0") && !ssd_stream;
        // DS4_LAYERS_PER_CB=K: pack K consecutive chain-loop layers into one
        // cb (one MTLCommandBuffer per K layers instead of 1 per layer).
        // Reduces GPU-side inter-cb scheduling gaps (~35us each) at the cost of
        // larger cbs. Default 1 = legacy per-layer cb. K=2 halves cb count;
        // K>n_layers folds the whole token into one cb. The shared scope is
        // hoisted via `chain_scope: Option<BatchScope>`; encode_layer_attn_half_
        // _in_scope encodes attn-half+kv_fp8 into the existing scope without
        // creating a new one.
        let layers_per_cb: usize = std::env::var("DS4_LAYERS_PER_CB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(1);
        let mut chain_hc: Option<crate::deferred::DeferredBuf> = None;
        let mut chain_cbs: Vec<metal::CommandBuffer> = Vec::new();
        let mut chain_finishes: Vec<ChainFinish> = Vec::new();
        let mut chain_scope: Option<crate::deferred::BatchScope<'_>> = None;
        let mut layers_in_batch: usize = 0;
        let t_chain_loop = std::time::Instant::now();
        let mut chain_commit_us: u128 = 0;

        for layer_idx in 0..l_split.0 {
            if startup_probe && first_commit_secs.is_none() && !chain_cbs.is_empty() {
                // First chain cb was committed during the previous iteration.
                first_commit_secs = Some(mach_now_secs());
            }
            let _sp = std::time::Instant::now();
            let layer = &model.layers[layer_idx];
            let p = &layer.attn.params;
            // Per-layer-input residual capture (probe): snapshot the layer input
            // (state.cur_hc) per layer, overwriting each token → final = last token.
            RESID_CAP.with(|c| {
                if let Some(layers) = c.borrow_mut().as_mut() {
                    if layers.len() <= layer_idx { layers.resize(layer_idx + 1, Vec::new()); }
                    layers[layer_idx] = state.cur_hc.clone();
                }
            });
            // Hash-routing layers handled inside `run_ffn_half`.
            // Compressor (compress_ratio>1) + indexer (compress_ratio==4)
            // handled below — pre-attn steps run on `normed_x`.

            // SWA circular ring (Step 3): the physical KV slot wraps at raw_cap
            // so the raw window slides (antirez DS4_N_SWA=128: most-recent
            // raw_cap rows). `lin_slot` stays the linear per-layer counter; the
            // store writes `slot = lin_slot % raw_cap`. Each raw row carries its
            // own rope (rope uses the absolute `pos`, NOT slot), so attention is
            // order-invariant and the flash's `gather [0..n_raw]` is correct in
            // both regimes. Bit-identical when raw_cap >= ctx (slot == lin_slot);
            // for ctx > raw_cap it evicts the oldest raw row (lives on as
            // compressed), matching antirez's `kv_cache_push_raw` slide.
            let lin_slot = state.kv_pos[layer_idx];
            let slot = lin_slot % raw_cap;

            // ── Fused single-cb raw-layer path (DS4_FUSED_RAW=1, WIP).
            //   For non-compressor layers, encode attn-half → GPU rope_q fwd
            //   → resident-q flash_attn → GPU rope_q back → output_proj →
            //   hc_collapse_norm → router → moe+shared → hc_expand_add ALL in
            //   ONE BatchScope, keeping q_heads/split/normed GPU-resident
            //   (no attn-half readback, no q round-trip). cur_hc is uploaded
            //   once per layer and read back once at the end (full residency
            //   is task #1). Matches the antirez single-cb structure for the
            //   raw layers; compressor layers still use the staged path below.
            // NOTE: a compressor layer at n_comp==0 is NOT safely a raw layer —
            // our DSA compressor builds + attends compressed rows from early
            // tokens (not only past raw_cap, unlike antirez's empty window), so
            // routing it through the raw fused path drops attended context
            // (measured: garbage output, though ~2x faster — the resident path's
            // payoff). The real win needs a resident path that ALSO attends the
            // comp rows (full-residency refactor), not this shortcut.
            let is_raw_layer =
                p.compress_ratio == 0 || !layer.attn.has_attn_compressor();
            let fused_raw = std::env::var("DS4_FUSED_RAW").ok().as_deref() == Some("1")
                && !ssd_stream
                && with_ffn_half
                && is_raw_layer
                && p.head_dim == 512
                && layer.moe.routing_table.is_none()
                && std::env::var("DS4_SILU_FIDELITY").ok().as_deref() != Some("1")
                && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1");
            if fused_raw {
                if std::env::var("DS4_LAYER_TRACE").is_ok() { eprintln!("[branch BR-fusedraw] L{layer_idx}"); }
                // bridge_merge uses its own fresh scope + synchronous flush. If
                // chain_scope has pending batched layers (DS4_LAYERS_PER_CB>1),
                // commit them FIRST so cb order on the queue stays correct
                // (queue is FIFO; bridge_merge's later cb must come after the
                // chain_scope cb that holds prior layers' work).
                if let Some(s) = chain_scope.take() {
                    let _tc = std::time::Instant::now();
                    chain_cbs.extend(s.commit_detach());
                    chain_commit_us += _tc.elapsed().as_micros();
                    layers_in_batch = 0;
                }
                let head_dim = p.head_dim as usize;
                let n_rot = p.n_rot as usize;
                let n_hc_p = p.n_hc as usize;
                let n_raw = (state.pos + 1).min(raw_cap);
                let q_dim = (p.n_head as usize) * head_dim;
                let n_groups = p.n_out_group as usize;
                let group_dim = q_dim / n_groups;
                let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                let out_low_dim = n_groups * n_lora_o;

                let (mut scope, half) = self.encode_layer_attn_half_open(
                    layer_idx as u32,
                    &state.cur_hc,
                    &layer.attn.hc_attn_fn,
                    layer.attn.hc_attn_fn_f16(),
                    &layer.attn.hc_attn_scale,
                    &layer.attn.hc_attn_base,
                    &layer.attn.hc_norm_gamma,
                    &unit_gamma_hc,
                    &layer.attn.attn_q_a,
                    &layer.attn.qkv_gamma_q,
                    &layer.attn.attn_q_b,
                    &layer.attn.attn_kv,
                    &layer.attn.qkv_gamma_kv,
                    &layer.attn.attn_q_a_q8,
                    &layer.attn.attn_q_b_q8,
                    &layer.attn.attn_kv_q8,
                    p,
                    pos,
                    raw_cap,
                    slot,
                )?;
                state.kv_pos[layer_idx] = lin_slot + 1;

                if n_rot > 0 {
                    scope.rope_tail_q_heads_in_place(
                        &half.q_heads, p.n_head as usize, head_dim, p, pos, false,
                    )?;
                }
                let heads_b = scope.flash_attn_decode_persistent_compressor_qbuf(
                    layer_idx as u32, p, &half.q_heads, n_raw, raw_cap,
                    &[], &[], 0, &layer.attn.attn_sinks,
                )?;
                if n_rot > 0 {
                    scope.rope_tail_q_heads_in_place(
                        &heads_b, p.n_head as usize, head_dim, p, pos, true,
                    )?;
                }
                let cur_hc_b = scope.upload_f32(&state.cur_hc);
                let attn_out = self.encode_output_proj(
                    &scope, &layer.attn, &heads_b,
                    n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
                )?;
                let after_attn_db = scope.hc_expand_attn_split(
                    &attn_out, &cur_hc_b, &half.split, n_hc_p, d_embd,
                )?;

                // ffn-half in the same scope.
                let (hc_ffn_fn_db, hc_ffn_is_f16) = scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
                let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
                let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
                let hc_ffn_norm_gamma_db = scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
                let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc);
                let (ffn_split_db, _cur_db, normed_db) = scope.hc_collapse_norm(
                    &after_attn_db, &hc_ffn_fn_db, &hc_ffn_scale_db, &hc_ffn_base_db,
                    &hc_ffn_norm_gamma_db, n_hc_p, d_embd,
                    p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps, &unit_gamma_hc_db, hc_ffn_is_f16,
                )?;
                let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
                let probs_db = scope.encode_router_logits(
                    &w_router_db, &normed_db, layer.moe.n_experts, w_router_r_f16,
                )?;
                let bias_db = scope.weight_f32(&layer.moe.router_bias);
                let (sel_db, w_db) = scope.encode_router_finalize(&probs_db, &bias_db)?;
                let sel_db = self.stream_remap_sel(&mut scope, layer_idx, sel_db, &w_db);
                let (moe_db, shared_db) = scope
                    .encode_moe_and_shared_chain_with_router_bufs_db(
                        layer_idx as u32, &normed_db, &sel_db, &w_db, layer.moe.d_ffn.map_or(0, |n| n.get()),
                        &normed_db, &layer.attn.w_shared_gate, &layer.attn.w_shared_up,
                        &layer.attn.w_shared_down, layer.attn.shared_dim,
                        &layer.attn.w_shared_gate_q8, &layer.attn.w_shared_up_q8,
                        &layer.attn.w_shared_down_q8,
                    )?;
                let after_ffn_db = scope.hc_expand_add_split(
                    &shared_db, &moe_db, &after_attn_db, &ffn_split_db, n_hc_p, d_embd,
                )?;
                let after_ffn_hc = scope.flush_and_read(&after_ffn_db);
                state.cur_hc = after_ffn_hc;

                per_layer.push(CpuLayerAttnHalfOuts {
                    normed: Vec::new(),
                    split: Vec::new(),
                    qr_normed: Vec::new(),
                    q_heads: Vec::new(),
                    kv_normed_rotated: Vec::new(),
                });
                if stage_prof {
                    t_ffn += _sp.elapsed().as_secs_f64() * 1000.0;
                    n_split += 1;
                }
                continue;
            }

            // ── Fused single-cb BRIDGE path (step 9b, default-on;
            //   DS4_BRIDGE_MERGE=0 reverts to the staged path below). The
            //   compressor-layer analogue of `fused_raw`: attn-half →
            //   compressor (matvec+store+pool+emit, with the emit row written
            //   GPU-side into the comp ring) → GPU rope_q → resident-q flash
            //   over [raw KV | GPU comp ring] → rope_q back → output_proj →
            //   ffn-half, ALL in ONE command buffer. The flash gathers
            //   comp_rows from the GPU ring (no CPU comp_kv_ring upload) and
            //   comp_sel is the all-allowed [0..n_comp) the indexer selection
            //   early-returns — valid while n_comp ≤ DS4_N_INDEXER_TOP_K (512),
            //   so no `normed`/`qr_normed` readback is needed. The compressor
            //   CPU finish (pool resync + ratio-4 rotation + comp_kv_ring
            //   append, for the next token and the non-merged fallback) runs
            //   after the single flush. Otherwise falls through.
            {
                let ratio = p.compress_ratio;
                let should_compress = ratio != 0 && (state.pos + 1) % ratio == 0;
                let post_n_comp = state.n_comp[layer_idx] + should_compress as u32;
                let post_n_index = state.n_index_comp[layer_idx] + should_compress as u32;
                let indexer_active = ratio == 4
                    && layer.attn.has_indexer_compressor()
                    && layer.attn.has_indexer_qb();
                let gpu_rope_q_on = p.n_rot > 0
                    && std::env::var("DS4_GPU_ROPE_Q").ok().as_deref() != Some("0");
                let env_ok = !ssd_stream
                    && std::env::var("DS4_BRIDGE_MERGE").ok().as_deref() != Some("0")
                    && std::env::var("DS4_COMPRESSOR_METAL").ok().as_deref() != Some("0")
                    && std::env::var("DS4_COMPRESSOR_POOL").ok().as_deref() != Some("0")
                    && std::env::var("DS4_COMPRESSOR_FUSE").ok().as_deref() != Some("0")
                    && std::env::var("DS4_COMPRESSOR_FINISH_GPU").ok().as_deref() != Some("0")
                    && std::env::var("DS4_SILU_FIDELITY").ok().as_deref() != Some("1")
                    && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1");
                // DS4_FUSED_COMP (default ON; =0 reverts to the staged path):
                // extend the resident bridge path to attn-compressor layers that
                // have NO indexer (the ratio=128 odd layers L3,5..41), which
                // otherwise take the staged 4-drain path. They attend all comp rows
                // (comp_sel=0..n_comp) — valid since with no indexer there is no
                // top-k selection. The idx compressor build/finish is skipped for
                // them. Pairs with the f32-weight force for these layers in
                // attn_dispatch.rs/decode_step.rs (the resident path is f16-precision
                // -sensitive — f16 corrupts the output, f32 is coherent + fast).
                // Long-ctx decode 5.10→9.90 tok/s @3000, 6.03→23.32 @600 (= antirez);
                // short prompts (<128 tok) never engage it (=0 reverts everything).
                let fused_comp =
                    std::env::var("DS4_FUSED_COMP").ok().as_deref() != Some("0");
                let bridge_merge = env_ok
                    && layer_idx < chain_max
                    && with_ffn_half
                    && layer.attn.has_attn_compressor()
                    && post_n_comp > 0
                    && p.head_dim == 512
                    && gpu_rope_q_on
                    && (layer.moe.routing_table.is_none() || hash_chain_on)
                    && ((ratio == 4
                        && indexer_active
                        && (post_n_index
                            <= ds4_engine::attn_dispatch::ds4_n_indexer_top_k()
                            // DS4_FUSED_COMP (default ON): keep indexer layers
                            // resident PAST top_k too — the in-scope GPU indexer
                            // (scores → top-k → sel buffer) feeds the flash with
                            // no readback, so the chain never drains. =0 reverts
                            // to the staged 3-drain path beyond ~2048 tokens.
                            || fused_comp))
                        || (fused_comp && !indexer_active));

                if std::env::var("DS4_BRIDGE_DEBUG").is_ok()
                    && !bridge_merge
                    && layer.attn.has_attn_compressor()
                    && state.pos > 590
                    && state.pos < 596
                {
                    eprintln!(
                        "[bridge-debug] L{layer_idx} ratio={} has_attn_comp={} has_idx_comp={} has_idx_qb={} indexer_active={} post_n_index={} top_k={} post_n_comp={} head_dim={} gpu_rope={} env_ok={}",
                        p.compress_ratio,
                        layer.attn.has_attn_compressor(),
                        layer.attn.has_indexer_compressor(),
                        layer.attn.has_indexer_qb(),
                        indexer_active,
                        post_n_index,
                        ds4_engine::attn_dispatch::ds4_n_indexer_top_k(),
                        post_n_comp,
                        p.head_dim,
                        gpu_rope_q_on,
                        env_ok,
                    );
                }
                if bridge_merge {
                if std::env::var("DS4_LAYER_TRACE").is_ok() { eprintln!("[branch BR-bridge] L{layer_idx}"); }
                    let head_dim = p.head_dim as usize;
                    let n_rot = p.n_rot as usize;
                    let n_hc_p = p.n_hc as usize;
                    let n_raw = (state.pos + 1).min(raw_cap);
                    let q_dim = (p.n_head as usize) * head_dim;
                    let n_groups = p.n_out_group as usize;
                    let group_dim = q_dim / n_groups;
                    let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                    let out_low_dim = n_groups * n_lora_o;

                    let (main_kv_f16, main_gate_f16) = layer.attn.attn_compressor_f16();
                    let (idx_kv_f16, idx_gate_f16) = layer.attn.indexer_compressor_f16();
                    let main_comp = ds4_engine::attn_dispatch::CompressorInputs {
                        w_kv: &layer.attn.attn_compressor_kv,
                        w_gate: &layer.attn.attn_compressor_gate,
                        w_kv_f16: main_kv_f16,
                        w_gate_f16: main_gate_f16,
                        w_ape: &layer.attn.attn_compressor_ape,
                        w_norm: &layer.attn.attn_compressor_norm,
                        head_dim: p.head_dim,
                        compress_ratio: ratio,
                    };
                    let idx_comp = ds4_engine::attn_dispatch::CompressorInputs {
                        w_kv: &layer.attn.indexer_compressor_kv,
                        w_gate: &layer.attn.indexer_compressor_gate,
                        w_kv_f16: idx_kv_f16,
                        w_gate_f16: idx_gate_f16,
                        w_ape: &layer.attn.indexer_compressor_ape,
                        w_norm: &layer.attn.indexer_compressor_norm,
                        head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                        compress_ratio: ratio,
                    };

                    let comp_ring = self.dispatcher.comp_ring_or_alloc(
                        layer_idx as u32,
                        comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
                    );

                    // Per-stage GPU profiling harness (DS4_KERNEL_PROFILE): commit
                    // +wait each stage separately (DS4_OP_TRACE prints its GPU
                    // time) and run the non-chained finish. Synchronous, so it
                    // kills overlap — profiling-only. Off → normal chained path.
                    // DS4_KPROF_MIN_POS gates kprof to pos >= N so a long-context
                    // profile prefills normally and only profiles the high-ctx
                    // tokens (prefilling 3000 tok under per-stage commit+wait is
                    // otherwise catastrophically slow).
                    let kprof = std::env::var("DS4_KERNEL_PROFILE").is_ok()
                        && state.pos as usize
                            >= std::env::var("DS4_KPROF_MIN_POS")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(0usize);
                    if chain_scope.is_none() {
                        chain_scope = Some(self.dispatcher.batch_scope());
                    }
                    let scope: &mut crate::deferred::BatchScope<'_> =
                        chain_scope.as_mut().unwrap();
                    let half = self.encode_layer_attn_half_in_scope(
                        scope,
                        layer_idx as u32,
                        &state.cur_hc,
                        if chain_on && !kprof { chain_hc.as_ref() } else { None },
                        &layer.attn.hc_attn_fn,
                        layer.attn.hc_attn_fn_f16(),
                        &layer.attn.hc_attn_scale,
                        &layer.attn.hc_attn_base,
                        &layer.attn.hc_norm_gamma,
                        &unit_gamma_hc,
                        &layer.attn.attn_q_a,
                        &layer.attn.qkv_gamma_q,
                        &layer.attn.attn_q_b,
                        &layer.attn.attn_kv,
                        &layer.attn.qkv_gamma_kv,
                        &layer.attn.attn_q_a_q8,
                        &layer.attn.attn_q_b_q8,
                        &layer.attn.attn_kv_q8,
                        p,
                        pos,
                        raw_cap,
                        slot,
                    )?;
                    state.kv_pos[layer_idx] = lin_slot + 1;
                    debug_assert_eq!(half.normed.len() % 4, 0, "bridge_merge: normed_len not f4-aligned");
                    if kprof { scope.commit_wait_stage("attn_half"); }

                    // Compressor + indexer matvec→store→pool→emit into the
                    // scope. Main writes its emit row GPU-side into comp_ring
                    // at row n_comp; indexer needs no ring (flash reads only
                    // the main ring).
                    if std::env::var("DS4_COMP_DUMP").is_ok()
                        && state.pos == 127
                        && (layer_idx == 3 || layer_idx == 5)
                    {
                        let n = scope.debug_read(&half.normed);
                        let sum: f64 = n.iter().map(|&x| x as f64).sum();
                        eprintln!(
                            "[normed-dump RESIDENT] L{layer_idx} pos={} len={} sum={:.6} head={:?}",
                            state.pos, n.len(), sum, &n[..4.min(n.len())]
                        );
                    }
                    if std::env::var("DS4_NORMED_TRACE").is_ok() && layer_idx == 4 && state.pos < 4 {
                        let sum: f64 = state.cur_hc.iter().map(|&x| x as f64).sum();
                        eprintln!("[INPUT resident] L{layer_idx} pos={} cur_hc.sum={:.6} head={:?}", state.pos, sum, &state.cur_hc[..4.min(state.cur_hc.len())]);
                        let nv = scope.debug_read(&half.normed);
                        let sum: f64 = nv.iter().map(|&x| x as f64).sum();
                        eprintln!("[NORMED resident] L{layer_idx} pos={} sum={:.6} head={:?}", state.pos, sum, &nv[..4.min(nv.len())]);
                    }
                    let (main_handle, main_emit_db) = crate::compressor::compressor_encode_in_scope(
                        &scope, self.dispatcher, p, &main_comp, &half.normed,
                        state.comp_state_kv[layer_idx].len(), state.pos, layer_idx as u32, false,
                        Some((&comp_ring, state.n_comp[layer_idx])),
                    )?;
                    // Indexer compressor: only layers that HAVE one (ratio-4).
                    // The DS4_FUSED_COMP path runs no-indexer layers; skip it.
                    let idx_build: Option<(
                        crate::compressor::CompressorScopeHandle,
                        Option<crate::deferred::DeferredBuf>,
                    )> = if indexer_active {
                        Some(crate::compressor::compressor_encode_in_scope(
                            &scope, self.dispatcher, p, &idx_comp, &half.normed,
                            state.index_state_kv[layer_idx].len(), state.pos, layer_idx as u32, true,
                            None,
                        )?)
                    } else {
                        None
                    };
                    if kprof { scope.commit_wait_stage("compressor"); }

                    // GPU rope_q fwd → resident-q flash over [raw | comp ring].
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &half.q_heads, p.n_head as usize, head_dim, p, pos, false,
                        )?;
                    }
                    // Past top_k (~2048 tok) the indexer must SELECT top_k of the
                    // post_n_index compressed rows (below it, all rows are
                    // attended). Done FULLY in-scope so the chain never drains:
                    // sync the GPU index ring's committed rows from the CPU ring
                    // (source of truth, maintained by the deferred finish; a no-op
                    // in steady state), append THIS token's emit row (not yet in
                    // the CPU ring), score (encode_indexer_scores) and cooperative
                    // top-k (encode_indexer_topk) → a GPU sel buffer fed straight
                    // into the flash with NO readback. This is what kept @3000 on
                    // the staged 3-drain path; now it stays resident like @600.
                    let top_k = ds4_engine::attn_dispatch::ds4_n_indexer_top_k();
                    let heads_b = if indexer_active && post_n_index > top_k {
                        let idx_hd =
                            ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize;
                        let idx_ring = self.dispatcher.index_comp_ring_or_alloc(
                            layer_idx as u32,
                            comp_ring_rows(raw_cap) * idx_hd * std::mem::size_of::<f32>(),
                        );
                        let n_committed = state.n_index_comp[layer_idx];
                        let synced = state.index_comp_ring_gpu_rows[layer_idx];
                        if synced < n_committed {
                            let cpu = &state.index_comp_kv_ring[layer_idx];
                            let start = synced as usize * idx_hd;
                            let end = n_committed as usize * idx_hd;
                            debug_assert!(cpu.len() >= end, "idx ring CPU shorter than n_comp");
                            unsafe {
                                let dst = (idx_ring.contents() as *mut f32).add(start);
                                std::ptr::copy_nonoverlapping(
                                    cpu.as_ptr().add(start), dst, end - start,
                                );
                            }
                            state.index_comp_ring_gpu_rows[layer_idx] = n_committed;
                        }
                        // gpu_rows stays at n_committed so next token re-syncs this
                        // token's row from the (then-populated) CPU source.
                        if let Some((_, Some(ref emit))) = idx_build {
                            scope.copy_buf_into(emit, &idx_ring, n_committed as usize * idx_hd);
                        }
                        let index_ring_db = crate::deferred::DeferredBuf::from_external_buffer(
                            idx_ring, post_n_index as usize * idx_hd,
                        );
                        let (qb_f16, proj_f16) = layer.attn.indexer_scoring_f16();
                        let scores_db = scope.encode_indexer_scores(
                            &half.qr_normed, &half.normed,
                            &layer.attn.indexer_attn_q_b, &layer.attn.indexer_proj,
                            qb_f16, proj_f16, &index_ring_db, post_n_index as usize,
                            ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD as usize,
                            idx_hd, p, pos,
                        )?;
                        if kprof { scope.commit_wait_stage("indexer_scores"); }
                        if std::env::var("DS4_SEL_TRACE").is_ok() && layer_idx <= 6 && post_n_index <= top_k + 2 {
                            let ring = scope.debug_read(&index_ring_db);
                            let nc = post_n_index as usize;
                            let r0: Vec<f32> = ring.get(0..3).map(|s| s.to_vec()).unwrap_or_default();
                            let rl: Vec<f32> = ring.get((nc-1)*idx_hd..(nc-1)*idx_hd+3).map(|s| s.to_vec()).unwrap_or_default();
                            let s = scope.debug_read(&scores_db);
                            let sum: f64 = s.iter().map(|&x| x as f64).sum();
                            eprintln!("[SEL PERTOK] L{layer_idx} pos={} n_idx={post_n_index} ring0={r0:.6?} ringL={rl:.6?} scsum={sum:.6}", state.pos);
                        }
                        let k_sel = (top_k as usize).min(post_n_index as usize);
                        // DS4_FAST_TOPK (default ON): parallel threshold-select
                        // (~32 count passes) instead of the single-threadgroup
                        // top_k(512)-round greedy that profiled as the @3000 wall
                        // (~27 ms/token). Same SET (top_k by score, ties→lowest
                        // index). =0 reverts to the greedy.
                        let sel_db = if std::env::var("DS4_FAST_TOPK").ok().as_deref()
                            != Some("0")
                        {
                            scope.encode_indexer_topk_threshold(
                                &scores_db, post_n_index as usize, k_sel,
                            )?
                        } else {
                            scope.encode_indexer_topk(
                                &scores_db, post_n_index as usize, k_sel,
                            )?
                        };
                        if kprof { scope.commit_wait_stage("indexer_sel"); }
                        scope.flash_attn_decode_persistent_compressor_qbuf_gpuring_sel(
                            layer_idx as u32, p, &half.q_heads, n_raw, raw_cap,
                            &comp_ring, &sel_db, k_sel as u32, &layer.attn.attn_sinks,
                        )?
                    } else {
                        let comp_sel: Vec<u32> = (0..post_n_comp).collect();
                        scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                            layer_idx as u32, p, &half.q_heads, n_raw, raw_cap,
                            &comp_ring, &comp_sel, post_n_comp, &layer.attn.attn_sinks,
                        )?
                    };
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &heads_b, p.n_head as usize, head_dim, p, pos, true,
                        )?;
                    }
                    // DS4_PT_HEADS_DUMP=<layer>: per-token reference flash-heads
                    // capture (rope-backed, mirrors DS4_CHUNK_HEADS_DUMP). GPU
                    // copy into a persistent buffer (key layer+6M) at row=pos;
                    // read back at end of prefill (decode_runner writes the file).
                    if let Ok(v) = std::env::var("DS4_PT_HEADS_DUMP") {
                        if v.parse::<usize>().ok() == Some(layer_idx) && (pos as usize) < 4096 {
                            let q_dim = p.n_head as usize * head_dim;
                            let dbg = self.dispatcher.kv_buffer_or_alloc(
                                layer_idx as u32 + 6_000_000,
                                4096 * q_dim * std::mem::size_of::<f32>());
                            scope.copy_buf_into(&heads_b, &dbg, pos as usize * q_dim);
                        }
                    }
                    if kprof { scope.commit_wait_stage("flash"); }

                    // output_proj + ffn-half (identical to fused_raw). cur_hc is
                    // the resident chained residual when chaining, else uploaded.
                    let cur_hc_upload;
                    let cur_hc_b: &crate::deferred::DeferredBuf =
                        match (chain_on, chain_hc.as_ref()) {
                            (true, Some(r)) => r,
                            _ => {
                                cur_hc_upload = scope.upload_f32(&state.cur_hc);
                                &cur_hc_upload
                            }
                        };
                    let attn_out = self.encode_output_proj(
                        &scope, &layer.attn, &heads_b,
                        n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
                    )?;
                    let after_attn_db = scope.hc_expand_attn_split(
                        &attn_out, cur_hc_b, &half.split, n_hc_p, d_embd,
                    )?;
                    if kprof { scope.commit_wait_stage("output_proj"); }
                    let (hc_ffn_fn_db, hc_ffn_is_f16) = scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
                    let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
                    let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
                    let hc_ffn_norm_gamma_db = scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
                    let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc);
                    let (ffn_split_db, _cur_db, normed_db) = scope.hc_collapse_norm(
                        &after_attn_db, &hc_ffn_fn_db, &hc_ffn_scale_db, &hc_ffn_base_db,
                        &hc_ffn_norm_gamma_db, n_hc_p, d_embd,
                        p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps, &unit_gamma_hc_db, hc_ffn_is_f16,
                    )?;
                    if kprof { scope.commit_wait_stage("hc_collapse_ffn"); }
                    let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
                    let probs_db = scope.encode_router_logits(
                        &w_router_db, &normed_db, layer.moe.n_experts, w_router_r_f16,
                    )?;
                    let (sel_db, w_db) = if let Some(table) =
                        layer.moe.routing_table.as_ref()
                    {
                        // Hash-routed (V4-Flash 0/1/2): expert ids are the
                        // per-token routing-table lookup (CPU, known upfront);
                        // weights come from the GPU router probs via
                        // router_weights_one (no readback) so the layer chains.
                        let k_used = layer.moe.n_experts_used;
                        let token_id = ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT
                            .with(|c| c.get()) as usize;
                        let base = token_id.saturating_mul(k_used);
                        anyhow::ensure!(
                            base + k_used <= table.len(),
                            "hash chain: routing_table oob (token_id={token_id}, k={k_used})"
                        );
                        let sel_db = scope.alloc_i32(k_used);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                table[base..base + k_used].as_ptr(),
                                sel_db.buffer().contents() as *mut i32,
                                k_used,
                            );
                        }
                        let w_db = scope.encode_router_weights_one(&probs_db, &sel_db)?;
                        (sel_db, w_db)
                    } else {
                        let bias_db = scope.weight_f32(&layer.moe.router_bias);
                        scope.encode_router_finalize(&probs_db, &bias_db)?
                    };
                    let sel_db = self.stream_remap_sel(scope, layer_idx, sel_db, &w_db);
                    let (moe_db, shared_db) = scope
                        .encode_moe_and_shared_chain_with_router_bufs_db(
                            layer_idx as u32, &normed_db, &sel_db, &w_db, layer.moe.d_ffn.map_or(0, |n| n.get()),
                            &normed_db, &layer.attn.w_shared_gate, &layer.attn.w_shared_up,
                            &layer.attn.w_shared_down, layer.attn.shared_dim,
                            &layer.attn.w_shared_gate_q8, &layer.attn.w_shared_up_q8,
                            &layer.attn.w_shared_down_q8,
                        )?;
                    if kprof { scope.commit_wait_stage("router_moe_shared"); }
                    let after_ffn_db = scope.hc_expand_add_split(
                        &shared_db, &moe_db, &after_attn_db, &ffn_split_db, n_hc_p, d_embd,
                    )?;

                    if chain_on && !kprof {
                        // Chained: commit async (no wait), keep after_ffn
                        // resident as the next layer's cur_hc, defer both
                        // compressor finishes (incl. the GPU emit rows) to
                        // token end. With DS4_LAYERS_PER_CB>1, accumulate K
                        // layers in the same scope before commit_detach so the
                        // GPU sees one cb per K layers and pays the ~35us
                        // inter-cb gap K× less often.
                        chain_hc = Some(after_ffn_db);
                        chain_finishes.push(ChainFinish {
                            handle: main_handle, emit_db: main_emit_db,
                            layer_idx, is_indexer: false,
                        });
                        if let Some((ih, ie)) = idx_build {
                            chain_finishes.push(ChainFinish {
                                handle: ih, emit_db: ie,
                                layer_idx, is_indexer: true,
                            });
                        }
                        layers_in_batch += 1;
                        if layers_in_batch >= layers_per_cb {
                            let _tc = std::time::Instant::now();
                            let s = chain_scope.take().unwrap();
                            chain_cbs.extend(s.commit_detach());
                            chain_commit_us += _tc.elapsed().as_micros();
                            layers_in_batch = 0;
                        }
                    } else {
                        // ── ONE flush: after_ffn_hc + the emit rows (for the CPU
                        //   finish / comp_kv_ring mirror). ──
                        let (idx_handle, idx_emit_db) = match idx_build {
                            Some((h, e)) => (Some(h), e),
                            None => (None, None),
                        };
                        let mut flush_bufs: Vec<&crate::deferred::DeferredBuf> = vec![&after_ffn_db];
                        if let Some(ref e) = main_emit_db { flush_bufs.push(e); }
                        if let Some(ref e) = idx_emit_db { flush_bufs.push(e); }
                        let s = chain_scope.take().unwrap();
                        layers_in_batch = 0;
                        let cpu = s.flush_and_read_multi(&flush_bufs);
                        state.cur_hc = cpu[0].clone();
                        let mut pull = 1usize;
                        let main_pooled = if main_emit_db.is_some() {
                            let v = cpu[pull].clone(); pull += 1; v
                        } else { Vec::new() };
                        let idx_pooled = if idx_emit_db.is_some() {
                            cpu[pull].clone()
                        } else { Vec::new() };

                        if let Some(row) = crate::compressor::compressor_finish_in_scope(
                            main_handle, self.dispatcher, p, &main_comp, main_pooled,
                            &mut state.comp_state_kv[layer_idx],
                            &mut state.comp_state_score[layer_idx], state.pos,
                        ) {
                            state.comp_kv_ring[layer_idx].extend_from_slice(&row);
                            state.n_comp[layer_idx] += 1;
                        }
                        if let Some(ih) = idx_handle {
                            if let Some(row) = crate::compressor::compressor_finish_in_scope(
                                ih, self.dispatcher, p, &idx_comp, idx_pooled,
                                &mut state.index_state_kv[layer_idx],
                                &mut state.index_state_score[layer_idx], state.pos,
                            ) {
                                state.index_comp_kv_ring[layer_idx].extend_from_slice(&row);
                                state.n_index_comp[layer_idx] += 1;
                            }
                        }
                    }

                    per_layer.push(CpuLayerAttnHalfOuts {
                        normed: Vec::new(),
                        split: Vec::new(),
                        qr_normed: Vec::new(),
                        q_heads: Vec::new(),
                        kv_normed_rotated: Vec::new(),
                    });
                    if stage_prof {
                        t_ffn += _sp.elapsed().as_secs_f64() * 1000.0;
                        n_bridge += 1;
                    }
                    continue;
                }

                // ── Fused single-cb UNCOMPRESSED path (DS4_CHAIN_UNCOMP,
                //   default-on; =0 reverts). The two uncompressed layers
                //   (DS4-Flash 0/1 — no `attn_compressor_kv`, hash-routed)
                //   attend RAW KV only and have NO compressor state, so this
                //   is `merge_raw` minus the compressor encode/fold/finish:
                //   attn-half → GPU rope_q → resident-q RAW flash (n_selected=0)
                //   → output_proj → ffn-half (hash MoE) in ONE cb. Before this,
                //   layers 0/1 took the staged path and drained the chain at
                //   token start — they ARE the residual short-context head
                //   (PR #14 chained layer 2; this chains 0/1). No ChainFinish
                //   is pushed (no compressor mirror to resync).
                let uncomp_merge = std::env::var("DS4_CHAIN_UNCOMP").ok().as_deref()
                        != Some("0")
                    && std::env::var("DS4_SILU_FIDELITY").ok().as_deref() != Some("1")
                    && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1")
                    && std::env::var("DS4_COMPRESSOR_METAL").ok().as_deref() != Some("0")
                    && with_ffn_half
                    && layer_idx < chain_max
                    && !layer.attn.has_attn_compressor()
                    && p.head_dim == 512
                    && gpu_rope_q_on
                    && (layer.moe.routing_table.is_none() || hash_chain_on);
                if uncomp_merge {
                                        let head_dim = p.head_dim as usize;
                    let n_rot = p.n_rot as usize;
                    let n_hc_p = p.n_hc as usize;
                    let n_raw = (state.pos + 1).min(raw_cap);
                                        let q_dim = (p.n_head as usize) * head_dim;
                    let n_groups = p.n_out_group as usize;
                    let group_dim = q_dim / n_groups;
                                        let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                                        let out_low_dim = n_groups * n_lora_o;

                    if chain_scope.is_none() {
                        chain_scope = Some(self.dispatcher.batch_scope());
                    }
                    let scope: &mut crate::deferred::BatchScope<'_> =
                        chain_scope.as_mut().unwrap();
                    let half = self.encode_layer_attn_half_in_scope(
                        scope,
                        layer_idx as u32,
                        &state.cur_hc,
                        if chain_on { chain_hc.as_ref() } else { None },
                        &layer.attn.hc_attn_fn,
                        layer.attn.hc_attn_fn_f16(),
                        &layer.attn.hc_attn_scale,
                        &layer.attn.hc_attn_base,
                        &layer.attn.hc_norm_gamma,
                        &unit_gamma_hc,
                        &layer.attn.attn_q_a,
                        &layer.attn.qkv_gamma_q,
                        &layer.attn.attn_q_b,
                        &layer.attn.attn_kv,
                        &layer.attn.qkv_gamma_kv,
                        &layer.attn.attn_q_a_q8,
                        &layer.attn.attn_q_b_q8,
                        &layer.attn.attn_kv_q8,
                        p,
                        pos,
                        raw_cap,
                        slot,
                    )?;
                    state.kv_pos[layer_idx] = lin_slot + 1;

                    // Raw flash (n_selected=0) via the gpuring variant so kv_raw
                    // is bound as a GPU buffer — the store this cb just wrote is
                    // earlier in THIS cb, so a CPU snapshot would be stale.
                    let comp_ring = self.dispatcher.comp_ring_or_alloc(
                        layer_idx as u32,
                        comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
                    );
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &half.q_heads, p.n_head as usize, head_dim, p, pos, false,
                        )?;
                    }
                    let heads_b = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                        layer_idx as u32, p, &half.q_heads, n_raw, raw_cap,
                        &comp_ring, &[], 0, &layer.attn.attn_sinks,
                    )?;
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &heads_b, p.n_head as usize, head_dim, p, pos, true,
                        )?;
                    }
                    let cur_hc_upload;
                    let cur_hc_b: &crate::deferred::DeferredBuf =
                        match (chain_on, chain_hc.as_ref()) {
                            (true, Some(r)) => r,
                            _ => {
                                cur_hc_upload = scope.upload_f32(&state.cur_hc);
                                &cur_hc_upload
                            }
                        };
                    let attn_out = self.encode_output_proj(
                        &scope, &layer.attn, &heads_b,
                        n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
                    )?;
                    let after_attn_db = scope.hc_expand_attn_split(
                        &attn_out, cur_hc_b, &half.split, n_hc_p, d_embd,
                    )?;
                    let (hc_ffn_fn_db, hc_ffn_is_f16) = scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
                    let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
                    let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
                    let hc_ffn_norm_gamma_db = scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
                    let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc);
                    let (ffn_split_db, _cur_db, normed_db) = scope.hc_collapse_norm(
                        &after_attn_db, &hc_ffn_fn_db, &hc_ffn_scale_db, &hc_ffn_base_db,
                        &hc_ffn_norm_gamma_db, n_hc_p, d_embd,
                        p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps, &unit_gamma_hc_db, hc_ffn_is_f16,
                    )?;
                    let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
                    let probs_db = scope.encode_router_logits(
                        &w_router_db, &normed_db, layer.moe.n_experts, w_router_r_f16,
                    )?;
                    let (sel_db, w_db) = if let Some(table) =
                        layer.moe.routing_table.as_ref()
                    {
                        // Hash-routed (V4-Flash 0/1): expert ids = per-token
                        // routing-table lookup (CPU); weights via router_weights_one
                        // (no readback) so the layer chains.
                        let k_used = layer.moe.n_experts_used;
                        let token_id = ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT
                            .with(|c| c.get()) as usize;
                        let base = token_id.saturating_mul(k_used);
                        anyhow::ensure!(
                            base + k_used <= table.len(),
                            "uncomp chain: routing_table oob (token_id={token_id}, k={k_used})"
                        );
                        let sel_db = scope.alloc_i32(k_used);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                table[base..base + k_used].as_ptr(),
                                sel_db.buffer().contents() as *mut i32,
                                k_used,
                            );
                        }
                        let w_db = scope.encode_router_weights_one(&probs_db, &sel_db)?;
                        (sel_db, w_db)
                    } else {
                        let bias_db = scope.weight_f32(&layer.moe.router_bias);
                        scope.encode_router_finalize(&probs_db, &bias_db)?
                    };
                    let sel_db = self.stream_remap_sel(scope, layer_idx, sel_db, &w_db);
                    let (moe_db, shared_db) = scope
                        .encode_moe_and_shared_chain_with_router_bufs_db(
                            layer_idx as u32, &normed_db, &sel_db, &w_db, layer.moe.d_ffn.map_or(0, |n| n.get()),
                            &normed_db, &layer.attn.w_shared_gate, &layer.attn.w_shared_up,
                            &layer.attn.w_shared_down, layer.attn.shared_dim,
                            &layer.attn.w_shared_gate_q8, &layer.attn.w_shared_up_q8,
                            &layer.attn.w_shared_down_q8,
                        )?;
                    let after_ffn_db = scope.hc_expand_add_split(
                        &shared_db, &moe_db, &after_attn_db, &ffn_split_db, n_hc_p, d_embd,
                    )?;

                    let _ = &comp_ring; // bound but unused (n_selected=0)
                    if chain_on {
                        chain_hc = Some(after_ffn_db);
                        // No ChainFinish: uncompressed layers have no compressor
                        // state to resync.
                        layers_in_batch += 1;
                        if layers_in_batch >= layers_per_cb {
                            let _tc = std::time::Instant::now();
                            let s = chain_scope.take().unwrap();
                            chain_cbs.extend(s.commit_detach());
                            chain_commit_us += _tc.elapsed().as_micros();
                            layers_in_batch = 0;
                        }
                    } else {
                        let s = chain_scope.take().unwrap();
                        layers_in_batch = 0;
                        let after_ffn_hc = s.flush_and_read(&after_ffn_db);
                        state.cur_hc = after_ffn_hc;

                    }

                    per_layer.push(CpuLayerAttnHalfOuts {
                        normed: Vec::new(),
                        split: Vec::new(),
                        qr_normed: Vec::new(),
                        q_heads: Vec::new(),
                        kv_normed_rotated: Vec::new(),
                    });
                    if stage_prof {
                        t_ffn += _sp.elapsed().as_secs_f64() * 1000.0;
                        n_split += 1;
                    }
                    continue;
                }

                // ── Fused single-cb NO-COMP-ROW path (step 9c, default-on;
                //   DS4_MERGE_RAW=0 reverts). A compressor layer with NO
                //   compressed rows yet (post_n_comp==0 — ratio-128 layers at
                //   pos<128, ratio-4 before the first emit at pos<3) attends
                //   RAW KV only, so its compressor does NOT feed this token's
                //   flash and is DEFERRED to after the merged cb. attn-half →
                //   GPU rope_q → resident-q RAW flash (compressor kernel,
                //   n_selected=0, pads non-32-aligned n_raw) → output_proj →
                //   ffn-half in ONE cb; then the compressor state update runs
                //   on the read-back normed (own cb, no emit at post_n_comp==0).
                let merge_raw = std::env::var("DS4_MERGE_RAW").ok().as_deref() != Some("0")
                    && std::env::var("DS4_SILU_FIDELITY").ok().as_deref() != Some("1")
                    && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1")
                    && std::env::var("DS4_COMPRESSOR_METAL").ok().as_deref() != Some("0")
                    && with_ffn_half
                    && ratio != 0
                    && layer.attn.has_attn_compressor()
                    && post_n_comp == 0
                    && post_n_index == 0
                    && p.head_dim == 512
                    && gpu_rope_q_on
                    && layer_idx < chain_max
                    && (layer.moe.routing_table.is_none() || hash_chain_on);
                if merge_raw {
                    let head_dim = p.head_dim as usize;
                    let n_rot = p.n_rot as usize;
                    let n_hc_p = p.n_hc as usize;
                    let n_raw = (state.pos + 1).min(raw_cap);
                    let q_dim = (p.n_head as usize) * head_dim;
                    let n_groups = p.n_out_group as usize;
                    let group_dim = q_dim / n_groups;
                    let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                    let out_low_dim = n_groups * n_lora_o;

                    // Per-stage GPU profiling (DS4_KERNEL_PROFILE): commit+wait
                    // each stage separately (DS4_OP_TRACE prints its GPU time) —
                    // same harness as bridge_merge, but for the ratio-128 layers
                    // that actually fire on this model (bridge_merge needs
                    // ratio==4 / indexer, never present here). Synchronous (kills
                    // overlap) → profiling-only; off → normal chained path
                    // unchanged. The mid-layer commits split this layer's cb into
                    // per-stage cbs; resident bufs survive (queue order + wait).
                    let kprof = std::env::var("DS4_KERNEL_PROFILE").is_ok();

                    if chain_scope.is_none() {
                        chain_scope = Some(self.dispatcher.batch_scope());
                    }
                    let scope: &mut crate::deferred::BatchScope<'_> =
                        chain_scope.as_mut().unwrap();
                    let half = self.encode_layer_attn_half_in_scope(
                        scope,
                        layer_idx as u32,
                        &state.cur_hc,
                        if chain_on { chain_hc.as_ref() } else { None },
                        &layer.attn.hc_attn_fn,
                        layer.attn.hc_attn_fn_f16(),
                        &layer.attn.hc_attn_scale,
                        &layer.attn.hc_attn_base,
                        &layer.attn.hc_norm_gamma,
                        &unit_gamma_hc,
                        &layer.attn.attn_q_a,
                        &layer.attn.qkv_gamma_q,
                        &layer.attn.attn_q_b,
                        &layer.attn.attn_kv,
                        &layer.attn.qkv_gamma_kv,
                        &layer.attn.attn_q_a_q8,
                        &layer.attn.attn_q_b_q8,
                        &layer.attn.attn_kv_q8,
                        p,
                        pos,
                        raw_cap,
                        slot,
                    )?;
                    state.kv_pos[layer_idx] = lin_slot + 1;
                    if kprof { scope.commit_wait_stage("attn_half"); }

                    // Fold the compressor state-update (matvec+store, NO emit
                    // since post_n_comp==0) into THIS cb, reading the resident
                    // `normed` — no deferred own-cb compressor, no normed
                    // readback. The store writes the per-layer pool; the finish
                    // below resyncs the CPU mirror. It doesn't touch the raw flash.
                    let (main_kv_f16, main_gate_f16) = layer.attn.attn_compressor_f16();
                    let main_comp = ds4_engine::attn_dispatch::CompressorInputs {
                        w_kv: &layer.attn.attn_compressor_kv,
                        w_gate: &layer.attn.attn_compressor_gate,
                        w_kv_f16: main_kv_f16,
                        w_gate_f16: main_gate_f16,
                        w_ape: &layer.attn.attn_compressor_ape,
                        w_norm: &layer.attn.attn_compressor_norm,
                        head_dim: p.head_dim,
                        compress_ratio: ratio,
                    };
                    let (main_handle, _main_none) = crate::compressor::compressor_encode_in_scope(
                        &scope, self.dispatcher, p, &main_comp, &half.normed,
                        state.comp_state_kv[layer_idx].len(), state.pos, layer_idx as u32, false, None,
                    )?;
                    let idx_folded = if indexer_active {
                        let (idx_kv_f16, idx_gate_f16) = layer.attn.indexer_compressor_f16();
                        let idx_comp = ds4_engine::attn_dispatch::CompressorInputs {
                            w_kv: &layer.attn.indexer_compressor_kv,
                            w_gate: &layer.attn.indexer_compressor_gate,
                            w_kv_f16: idx_kv_f16,
                            w_gate_f16: idx_gate_f16,
                            w_ape: &layer.attn.indexer_compressor_ape,
                            w_norm: &layer.attn.indexer_compressor_norm,
                            head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                            compress_ratio: ratio,
                        };
                        let (h, _n) = crate::compressor::compressor_encode_in_scope(
                            &scope, self.dispatcher, p, &idx_comp, &half.normed,
                            state.index_state_kv[layer_idx].len(), state.pos, layer_idx as u32, true, None,
                        )?;
                        Some((h, idx_comp))
                    } else {
                        None
                    };
                    if kprof { scope.commit_wait_stage("compressor"); }

                    // Raw flash (n_selected=0) via the gpuring variant so kv_raw
                    // is bound as a GPU buffer — the kv_fp8_store_persistent that
                    // wrote this token's slot is earlier in THIS cb, so a CPU
                    // .contents() snapshot would be stale. comp_ring is bound but
                    // not read (n_selected=0).
                    let comp_ring = self.dispatcher.comp_ring_or_alloc(
                        layer_idx as u32,
                        comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
                    );
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &half.q_heads, p.n_head as usize, head_dim, p, pos, false,
                        )?;
                    }
                    let heads_b = scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                        layer_idx as u32, p, &half.q_heads, n_raw, raw_cap,
                        &comp_ring, &[], 0, &layer.attn.attn_sinks,
                    )?;
                    if n_rot > 0 {
                        scope.rope_tail_q_heads_in_place(
                            &heads_b, p.n_head as usize, head_dim, p, pos, true,
                        )?;
                    }
                    // DS4_PT_HEADS_DUMP=<layer>: per-token reference flash-heads
                    // capture (rope-backed, mirrors DS4_CHUNK_HEADS_DUMP). GPU
                    // copy into a persistent buffer (key layer+6M) at row=pos;
                    // read back at end of prefill (decode_runner writes the file).
                    if let Ok(v) = std::env::var("DS4_PT_HEADS_DUMP") {
                        if v.parse::<usize>().ok() == Some(layer_idx) && (pos as usize) < 4096 {
                            let q_dim = p.n_head as usize * head_dim;
                            let dbg = self.dispatcher.kv_buffer_or_alloc(
                                layer_idx as u32 + 6_000_000,
                                4096 * q_dim * std::mem::size_of::<f32>());
                            scope.copy_buf_into(&heads_b, &dbg, pos as usize * q_dim);
                        }
                    }
                    if kprof { scope.commit_wait_stage("flash"); }
                    let cur_hc_upload;
                    let cur_hc_b: &crate::deferred::DeferredBuf =
                        match (chain_on, chain_hc.as_ref()) {
                            (true, Some(r)) => r,
                            _ => {
                                cur_hc_upload = scope.upload_f32(&state.cur_hc);
                                &cur_hc_upload
                            }
                        };
                    let attn_out = self.encode_output_proj(
                        &scope, &layer.attn, &heads_b,
                        n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
                    )?;
                    let after_attn_db = scope.hc_expand_attn_split(
                        &attn_out, cur_hc_b, &half.split, n_hc_p, d_embd,
                    )?;
                    if kprof { scope.commit_wait_stage("output_proj"); }
                    let (hc_ffn_fn_db, hc_ffn_is_f16) = scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
                    let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
                    let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
                    let hc_ffn_norm_gamma_db = scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
                    let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc);
                    let (ffn_split_db, _cur_db, normed_db) = scope.hc_collapse_norm(
                        &after_attn_db, &hc_ffn_fn_db, &hc_ffn_scale_db, &hc_ffn_base_db,
                        &hc_ffn_norm_gamma_db, n_hc_p, d_embd,
                        p.hc_sinkhorn_iter as i32, p.hc_eps, p.rms_eps, &unit_gamma_hc_db, hc_ffn_is_f16,
                    )?;
                    if kprof { scope.commit_wait_stage("hc_collapse_ffn"); }
                    let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
                    let probs_db = scope.encode_router_logits(
                        &w_router_db, &normed_db, layer.moe.n_experts, w_router_r_f16,
                    )?;
                    let (sel_db, w_db) = if let Some(table) =
                        layer.moe.routing_table.as_ref()
                    {
                        // Hash-routed (V4-Flash 0/1/2): expert ids are the
                        // per-token routing-table lookup (CPU, known upfront);
                        // weights come from the GPU router probs via
                        // router_weights_one (no readback) so the layer chains.
                        let k_used = layer.moe.n_experts_used;
                        let token_id = ds4_engine::attn_dispatch::CURRENT_TOKEN_HINT
                            .with(|c| c.get()) as usize;
                        let base = token_id.saturating_mul(k_used);
                        anyhow::ensure!(
                            base + k_used <= table.len(),
                            "hash chain: routing_table oob (token_id={token_id}, k={k_used})"
                        );
                        let sel_db = scope.alloc_i32(k_used);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                table[base..base + k_used].as_ptr(),
                                sel_db.buffer().contents() as *mut i32,
                                k_used,
                            );
                        }
                        let w_db = scope.encode_router_weights_one(&probs_db, &sel_db)?;
                        (sel_db, w_db)
                    } else {
                        let bias_db = scope.weight_f32(&layer.moe.router_bias);
                        scope.encode_router_finalize(&probs_db, &bias_db)?
                    };
                    let sel_db = self.stream_remap_sel(scope, layer_idx, sel_db, &w_db);
                    let (moe_db, shared_db) = scope
                        .encode_moe_and_shared_chain_with_router_bufs_db(
                            layer_idx as u32, &normed_db, &sel_db, &w_db, layer.moe.d_ffn.map_or(0, |n| n.get()),
                            &normed_db, &layer.attn.w_shared_gate, &layer.attn.w_shared_up,
                            &layer.attn.w_shared_down, layer.attn.shared_dim,
                            &layer.attn.w_shared_gate_q8, &layer.attn.w_shared_up_q8,
                            &layer.attn.w_shared_down_q8,
                        )?;
                    let after_ffn_db = scope.hc_expand_add_split(
                        &shared_db, &moe_db, &after_attn_db, &ffn_split_db, n_hc_p, d_embd,
                    )?;
                    if kprof { scope.commit_wait_stage("moe"); }

                    let _ = &comp_ring; // bound but unused (n_selected=0)
                    if chain_on {
                        chain_hc = Some(after_ffn_db);
                        chain_finishes.push(ChainFinish {
                            handle: main_handle, emit_db: None, layer_idx, is_indexer: false,
                        });
                        if let Some((h, _idx_comp)) = idx_folded {
                            chain_finishes.push(ChainFinish {
                                handle: h, emit_db: None, layer_idx, is_indexer: true,
                            });
                        }
                        layers_in_batch += 1;
                        if layers_in_batch >= layers_per_cb {
                            let _tc = std::time::Instant::now();
                            let s = chain_scope.take().unwrap();
                            chain_cbs.extend(s.commit_detach());
                            chain_commit_us += _tc.elapsed().as_micros();
                            layers_in_batch = 0;
                        }
                    } else {
                        let s = chain_scope.take().unwrap();
                        layers_in_batch = 0;
                        let after_ffn_hc = s.flush_and_read(&after_ffn_db);
                        state.cur_hc = after_ffn_hc;

                        // Compressor finish: resync the CPU mirror from the pool the
                        // folded store wrote. post_n_comp==0 ⇒ no emit ⇒ returns None.
                        let _ = crate::compressor::compressor_finish_in_scope(
                            main_handle, self.dispatcher, p, &main_comp, Vec::new(),
                            &mut state.comp_state_kv[layer_idx],
                            &mut state.comp_state_score[layer_idx], state.pos,
                        );
                        if let Some((h, idx_comp)) = idx_folded {
                            let _ = crate::compressor::compressor_finish_in_scope(
                                h, self.dispatcher, p, &idx_comp, Vec::new(),
                                &mut state.index_state_kv[layer_idx],
                                &mut state.index_state_score[layer_idx], state.pos,
                            );
                        }
                    }

                    per_layer.push(CpuLayerAttnHalfOuts {
                        normed: Vec::new(),
                        split: Vec::new(),
                        qr_normed: Vec::new(),
                        q_heads: Vec::new(),
                        kv_normed_rotated: Vec::new(),
                    });
                    if stage_prof {
                        t_ffn += _sp.elapsed().as_secs_f64() * 1000.0;
                        n_split += 1;
                    }
                    continue;
                }

                // DS4_CHAIN_DIAG: reaching here means NEITHER chained branch
                // fired (both `continue` above) → this layer takes the staged
                // path and drains the chain. Log why so we know which case
                // (e.g. ratio-128 with comp rows: ratio!=4 fails bridge_merge,
                // post_n_comp>0 fails merge_raw) still breaks the chain.
                if std::env::var("DS4_CHAIN_DIAG").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[chain-diag] pos={} layer={} STAGED ratio={} post_n_comp={} \
                         post_n_index={} indexer_active={} head_dim={} gpu_rope_q={} \
                         hash={} compkv_empty={}",
                        state.pos, layer_idx, ratio, post_n_comp, post_n_index,
                        indexer_active, p.head_dim, gpu_rope_q_on,
                        layer.moe.routing_table.is_some(),
                        layer.attn.attn_compressor_kv.is_empty(),
                    );
                }
            }

            // Chain fallback: a layer that can't take a merged branch needs
            // the prior chained layers' cur_hc on CPU before it runs the staged
            // path. Drain the chain (wait + read cur_hc + run deferred finishes).
            // First commit any pending batched chain_scope so the drain covers
            // all prior work.
            if let Some(s) = chain_scope.take() {
                let _tc = std::time::Instant::now();
                chain_cbs.extend(s.commit_detach());
                chain_commit_us += _tc.elapsed().as_micros();
                layers_in_batch = 0;
            }
            if chain_on && chain_hc.is_some() {
                let hc = chain_hc.take();
                let cbs = std::mem::take(&mut chain_cbs);
                let finishes = std::mem::take(&mut chain_finishes);
                self.finish_chain(model, state, hc, cbs, finishes);
            }

            // ── 1. Attn-half via BatchScope, with the fuse-eligible
            //   compressor(s) folded into the SAME cb. encode_layer_attn_half_open
            //   keeps the scope open so the main (and indexer) compressor's
            //   matvec+store(+pool) read the resident `normed` instead of being
            //   re-uploaded into their own command buffers — saving one cb per
            //   compressor per layer. We flush ONCE for the attn-half outputs
            //   plus any pooled emit rows, then finish the compressors on CPU.
            let main_comp = (p.compress_ratio != 0
                && layer.attn.has_attn_compressor())
            .then(|| ds4_engine::attn_dispatch::CompressorInputs {
                w_kv: &layer.attn.attn_compressor_kv,
                w_gate: &layer.attn.attn_compressor_gate,
                w_kv_f16: layer.attn.attn_compressor_f16().0,
                w_gate_f16: layer.attn.attn_compressor_f16().1,
                w_ape: &layer.attn.attn_compressor_ape,
                w_norm: &layer.attn.attn_compressor_norm,
                head_dim: p.head_dim,
                compress_ratio: p.compress_ratio,
            });
            let idx_comp = (p.compress_ratio == 4
                && layer.attn.has_indexer_compressor()
                && layer.attn.has_indexer_qb())
            .then(|| ds4_engine::attn_dispatch::CompressorInputs {
                w_kv: &layer.attn.indexer_compressor_kv,
                w_gate: &layer.attn.indexer_compressor_gate,
                w_kv_f16: layer.attn.indexer_compressor_f16().0,
                w_gate_f16: layer.attn.indexer_compressor_f16().1,
                w_ape: &layer.attn.indexer_compressor_ape,
                w_norm: &layer.attn.indexer_compressor_norm,
                head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                compress_ratio: p.compress_ratio,
            });

            let (scope, half) = self.encode_layer_attn_half_open(
                layer_idx as u32,
                &state.cur_hc,
                &layer.attn.hc_attn_fn,
                layer.attn.hc_attn_fn_f16(),
                &layer.attn.hc_attn_scale,
                &layer.attn.hc_attn_base,
                &layer.attn.hc_norm_gamma,
                &unit_gamma_hc,
                &layer.attn.attn_q_a,
                &layer.attn.qkv_gamma_q,
                &layer.attn.attn_q_b,
                &layer.attn.attn_kv,
                &layer.attn.qkv_gamma_kv,
                &layer.attn.attn_q_a_q8,
                &layer.attn.attn_q_b_q8,
                &layer.attn.attn_kv_q8,
                p,
                pos,
                raw_cap,
                slot,
            )?;
            state.kv_pos[layer_idx] = lin_slot + 1;

            let normed_len = half.normed.len();
            let use_metal_compressor =
                std::env::var("DS4_COMPRESSOR_METAL").ok().as_deref() != Some("0");
            let fuse_main = use_metal_compressor
                && main_comp.as_ref().is_some_and(|c| {
                    crate::compressor::compressor_can_fuse_in_scope(c, normed_len)
                });
            let fuse_idx = use_metal_compressor
                && idx_comp.as_ref().is_some_and(|c| {
                    crate::compressor::compressor_can_fuse_in_scope(c, normed_len)
                });

            // Encode fuse-eligible compressors into the open attn-half scope.
            let (main_handle, main_pooled_db) = if fuse_main {
                let (h, pd) = crate::compressor::compressor_encode_in_scope(
                    &scope,
                    self.dispatcher,
                    p,
                    main_comp.as_ref().unwrap(),
                    &half.normed,
                    state.comp_state_kv[layer_idx].len(),
                    state.pos,
                    layer_idx as u32,
                    false,
                    None,
                )?;
                (Some(h), pd)
            } else {
                (None, None)
            };
            let (idx_handle, idx_pooled_db) = if fuse_idx {
                let (h, pd) = crate::compressor::compressor_encode_in_scope(
                    &scope,
                    self.dispatcher,
                    p,
                    idx_comp.as_ref().unwrap(),
                    &half.normed,
                    state.index_state_kv[layer_idx].len(),
                    state.pos,
                    layer_idx as u32,
                    true,
                    None,
                )?;
                (Some(h), pd)
            } else {
                (None, None)
            };

            // Single flush: attn-half outputs + any pooled emit rows.
            let mut flush_bufs: Vec<&crate::deferred::DeferredBuf> =
                vec![&half.normed, &half.split, &half.qr_normed, &half.q_heads];
            if let Some(ref pd) = main_pooled_db {
                flush_bufs.push(pd);
            }
            if let Some(ref pd) = idx_pooled_db {
                flush_bufs.push(pd);
            }
            let cpu = scope.flush_and_read_multi(&flush_bufs);
            if std::env::var("DS4_COMP_DUMP").is_ok()
                && state.pos == 127
                && (layer_idx == 3 || layer_idx == 5)
            {
                let n = &cpu[0];
                let sum: f64 = n.iter().map(|&x| x as f64).sum();
                eprintln!(
                    "[normed-dump STAGED] L{layer_idx} pos={} len={} sum={:.6} head={:?}",
                    state.pos, n.len(), sum, &n[..4.min(n.len())]
                );
            }
            let outs = CpuLayerAttnHalfOuts {
                normed: cpu[0].clone(),
                split: cpu[1].clone(),
                qr_normed: cpu[2].clone(),
                q_heads: cpu[3].clone(),
                kv_normed_rotated: Vec::new(),
            };
            let mut pull = 4usize;
            let main_pooled = if main_pooled_db.is_some() {
                let v = cpu[pull].clone();
                pull += 1;
                v
            } else {
                Vec::new()
            };
            let idx_pooled = if idx_pooled_db.is_some() {
                cpu[pull].clone()
            } else {
                Vec::new()
            };

            if stage_prof { t_attn += _sp.elapsed().as_secs_f64() * 1000.0; }
            let _sp = std::time::Instant::now();

            // ── 1a. Persistent KV slot — no CPU correction needed.
            //
            // `BatchScope::kv_fp8_store_persistent` (encoded inside the
            // attn-half scope) writes the just-computed `kv_normed_rotated`
            // to the layer's persistent buffer via the GPU shim
            // `ds4_dsv4_kv_fp8_store`. As of the shim rewrite that GPU write
            // is bit-exact to the antirez-correct algorithm (per-64-block
            // scaled e4m3 + f16 round-trip; proven by the macos.rs test
            // `kv_fp8_store_persistent_matches_cpu_correction`), so the CPU
            // re-quantize that used to overwrite this slot is gone — along
            // with the `kv_normed_rotated` readback it required.

            // ── 1b. Compressor (M4 #266, mirrors decode_step.rs:830-856).
            //   On compressor layers (`compress_ratio != 0` and
            //   `attn_compressor_kv` populated), project `normed_x`
            //   through compressor weights, update per-layer state,
            //   append emitted compressed row on `(pos+1) % ratio == 0`.
            //
            //   The GPU orchestrator port (`compressor_decode_one_metal`,
            //   M5 Phase D #5) is now default-on: matvecs run on GPU,
            //   dropping comp ~167->22ms/token (~2x decode). Set
            //   `DS4_COMPRESSOR_METAL=0` to fall back to the CPU path.
            if let Some(comp) = main_comp.as_ref() {
                let emit = if fuse_main {
                    // Folded into the attn-half cb above — just resync the CPU
                    // mirror and run the emit tail on the read-back pooled row.
                    crate::compressor::compressor_finish_in_scope(
                        main_handle.unwrap(),
                        self.dispatcher,
                        p,
                        comp,
                        main_pooled,
                        &mut state.comp_state_kv[layer_idx],
                        &mut state.comp_state_score[layer_idx],
                        state.pos,
                    )
                } else if use_metal_compressor {
                    crate::compressor::compressor_decode_one_metal(
                        self.dispatcher,
                        p,
                        comp,
                        &outs.normed,
                        &mut state.comp_state_kv[layer_idx],
                        &mut state.comp_state_score[layer_idx],
                        state.pos,
                        layer_idx as u32,
                        false,
                    )
                } else {
                    ds4_engine::attn_dispatch::compressor_decode_one(
                        self.dispatcher,
                        p,
                        comp,
                        &outs.normed,
                        &mut state.comp_state_kv[layer_idx],
                        &mut state.comp_state_score[layer_idx],
                        state.pos,
                    )
                };
                if let Some(row) = emit {
                    // single-cb step 9: mirror the emit row into the GPU comp
                    // ring at row n_comp (lockstep with comp_kv_ring) so the
                    // bridge flash can gather comp_rows straight off the GPU.
                    let n = state.n_comp[layer_idx] as usize;
                    let ring = self.dispatcher.comp_ring_or_alloc(
                        layer_idx as u32,
                        comp_ring_rows(raw_cap) * row.len() * std::mem::size_of::<f32>(),
                    );
                    unsafe {
                        let dst = (ring.contents() as *mut f32).add(n * row.len());
                        std::ptr::copy_nonoverlapping(row.as_ptr(), dst, row.len());
                    }
                    state.comp_kv_ring[layer_idx].extend_from_slice(&row);
                    state.n_comp[layer_idx] += 1;
                }
            }

            // ── 1b. Indexer (M4 #267, ratio==4 only; decode_step.rs:874-931).
            //   Run a second compressor over the indexer weights, then
            //   `indexer_allowed_decode_one(normed, qr_norm)` to produce
            //   the top-k bitmap over compressed rows. Pass that as
            //   `comp_selected` to flash_attn.
            let mut indexer_allowed: Option<Vec<u32>> = None;
            if p.compress_ratio == 4
                && layer.attn.has_indexer_compressor()
                && layer.attn.has_indexer_qb()
            {
                let icomp = idx_comp.as_ref().expect("idx_comp Some under this guard");
                let emit = if fuse_idx {
                    // Folded into the attn-half cb above.
                    crate::compressor::compressor_finish_in_scope(
                        idx_handle.unwrap(),
                        self.dispatcher,
                        p,
                        icomp,
                        idx_pooled,
                        &mut state.index_state_kv[layer_idx],
                        &mut state.index_state_score[layer_idx],
                        state.pos,
                    )
                } else if use_metal_compressor {
                    crate::compressor::compressor_decode_one_metal(
                        self.dispatcher,
                        p,
                        icomp,
                        &outs.normed,
                        &mut state.index_state_kv[layer_idx],
                        &mut state.index_state_score[layer_idx],
                        state.pos,
                        layer_idx as u32,
                        true,
                    )
                } else {
                    ds4_engine::attn_dispatch::compressor_decode_one(
                        self.dispatcher,
                        p,
                        icomp,
                        &outs.normed,
                        &mut state.index_state_kv[layer_idx],
                        &mut state.index_state_score[layer_idx],
                        state.pos,
                    )
                };
                if let Some(row) = emit {
                    state.index_comp_kv_ring[layer_idx].extend_from_slice(&row);
                    state.n_index_comp[layer_idx] += 1;
                }
                let n_index_comp = state.n_index_comp[layer_idx];
                if n_index_comp > 0 {
                    // DS4_GPU_INDEXER (default ON; =0 reverts to CPU): for the
                    // long-context path (n_comp > top_k, ~>2048 tokens) move the
                    // dominant indexer SCORE compute to the GPU
                    // (encode_indexer_scores) and do the cheap greedy top-k on
                    // CPU over the read-back scores. The `n_comp > top_k` guard
                    // means this is a NO-OP for short-context decode — the CPU
                    // `indexer_allowed_decode_one` early-returns all-allowed at
                    // n_comp <= top_k, and this gate likewise doesn't fire — so
                    // default-on only changes the long-context regime (~28x).
                    // DEFAULT OFF (was default-on): the GPU long-context indexer is the
                    // prime suspect for antirez's "long generations output random text" report
                    // — it engages ONLY past n_comp>top_k (~>2048 tokens), exactly the
                    // long-generation regime, and diverges from the CPU `indexer_allowed_decode_one`
                    // oracle. Routing long-context selection through the proven CPU path is
                    // correctness-first (worst case a perf trade, never a correctness regression).
                    // Re-enable the GPU speedup with DS4_GPU_INDEXER=1 once the path is verified.
                    let gpu_indexer = std::env::var("DS4_GPU_INDEXER").ok().as_deref()
                        == Some("1")
                        && n_index_comp > ds4_engine::attn_dispatch::ds4_n_indexer_top_k();
                    let selected: Vec<u32> = if gpu_indexer {
                        let scope = self.dispatcher.batch_scope();
                        let qr_db = scope.upload_f32(&outs.qr_normed);
                        let cur_db = scope.upload_f32(&outs.normed);
                        // GPU-resident ring: lazily sync only the rows appended
                        // since last time (the CPU ring is the source of truth —
                        // it's fed by all three emit paths: staged, bridge-merge
                        // finish, and chained finish_chain). Usually 0-1 new rows
                        // per token; the first GPU-indexer token copies the small
                        // backlog. Avoids re-uploading the whole [n_comp,128] ring
                        // each token (the win at large n_comp / production top_k).
                        let hd = ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize;
                        let ring_buf = self.dispatcher.index_comp_ring_or_alloc(
                            layer_idx as u32,
                            comp_ring_rows(self.raw_cap) * hd * std::mem::size_of::<f32>(),
                        );
                        let synced = state.index_comp_ring_gpu_rows[layer_idx];
                        if synced < n_index_comp {
                            let cpu = &state.index_comp_kv_ring[layer_idx];
                            let start = synced as usize * hd;
                            let end = n_index_comp as usize * hd;
                            debug_assert!(cpu.len() >= end, "index ring CPU shorter than n_comp");
                            unsafe {
                                let dst = (ring_buf.contents() as *mut f32).add(start);
                                std::ptr::copy_nonoverlapping(
                                    cpu.as_ptr().add(start),
                                    dst,
                                    end - start,
                                );
                            }
                            state.index_comp_ring_gpu_rows[layer_idx] = n_index_comp;
                        }
                        if std::env::var("DS4_INDEXER_RING_CHECK").ok().as_deref()
                            == Some("1")
                        {
                            let n = n_index_comp as usize * hd;
                            let resident = unsafe {
                                std::slice::from_raw_parts(ring_buf.contents() as *const f32, n)
                            };
                            let cpu = &state.index_comp_kv_ring[layer_idx];
                            let mut md = 0f32;
                            for i in 0..n.min(cpu.len()) {
                                md = md.max((resident[i] - cpu[i]).abs());
                            }
                            eprintln!("[ring_check] layer={layer_idx} n_comp={n_index_comp} max_diff={md:.3e}");
                        }
                        let ring_db = crate::deferred::DeferredBuf::from_external_buffer(
                            ring_buf,
                            n_index_comp as usize * hd,
                        );
                        let (qb_f16, proj_f16) = layer.attn.indexer_scoring_f16();
                        let scores_db = scope.encode_indexer_scores(
                            &qr_db,
                            &cur_db,
                            &layer.attn.indexer_attn_q_b,
                            &layer.attn.indexer_proj,
                            qb_f16,
                            proj_f16,
                            &ring_db,
                            n_index_comp as usize,
                            ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD as usize,
                            ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize,
                            p,
                            state.pos,
                        )?;
                        let top_k = (ds4_engine::attn_dispatch::ds4_n_indexer_top_k()
                            as usize)
                            .min(n_index_comp as usize);
                        if std::env::var("DS4_GPU_INDEXER_TOPK").ok().as_deref()
                            == Some("1")
                        {
                            // GPU cooperative greedy top-k: keeps the selection on
                            // GPU (no scores[n_comp] readback), reads back only the
                            // top_k indices. Sort ascending to match the CPU filter.
                            let sel_db = scope
                                .encode_indexer_topk(&scores_db, n_index_comp as usize, top_k)?;
                            let mut sel: Vec<u32> = scope
                                .flush_and_read_i32(&sel_db)
                                .into_iter()
                                .map(|i| i as u32)
                                .collect();
                            sel.sort_unstable();
                            sel
                        } else {
                            // CPU greedy over the read-back scores (descending
                            // sort, take top_k). Matches indexer_allowed_decode_one;
                            // emit ascending indices like the CPU filter does.
                            let scores = scope.flush_and_read(&scores_db);
                            let mut order: Vec<u32> = (0..scores.len() as u32).collect();
                            order.sort_by(|&a, &b| {
                                scores[b as usize]
                                    .partial_cmp(&scores[a as usize])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            order.truncate(top_k);
                            order.sort_unstable();
                            order
                        }
                    } else {
                        let inputs = ds4_engine::attn_dispatch::IndexerInputs {
                            w_q_b: &layer.attn.indexer_attn_q_b,
                            w_proj: &layer.attn.indexer_proj,
                            index_comp_kv: &state.index_comp_kv_ring[layer_idx],
                            n_comp: n_index_comp,
                            n_head: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD,
                            head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                            top_k: ds4_engine::attn_dispatch::ds4_n_indexer_top_k(),
                        };
                        let (allowed, _n_sel) =
                            ds4_engine::attn_dispatch::indexer_allowed_decode_one(
                                self.dispatcher,
                                p,
                                &inputs,
                                &outs.normed,
                                &outs.qr_normed,
                                state.pos,
                            );
                        let mut s: Vec<u32> = allowed
                            .iter()
                            .enumerate()
                            .filter_map(|(c, &a)| if a { Some(c as u32) } else { None })
                            .collect();
                        s.shrink_to_fit();
                        s
                    };
                    indexer_allowed = Some(selected);
                }
            }

            if stage_prof { t_comp += _sp.elapsed().as_secs_f64() * 1000.0; }
            let _sp = std::time::Instant::now();
            // ── 2. rope_q forward on Q heads — applied below, once the
            // flash_attn path is chosen: on GPU for the bridge path
            // (DS4_GPU_ROPE_Q, default-on) so q stays resident across
            // rope→flash, on CPU for the slow_path that still uploads q
            // as a plain slice.
            let n_rot = p.n_rot as usize;
            let head_dim = p.head_dim as usize;

            // ── 3-5. flash_attn + rope_q backward + attn_output_proj.
            //
            // Phase F task #87: branch on the BatchScope fast-path
            // conditions (head_dim==512, n_raw 32-aligned + nonzero,
            // no compressor/indexer outputs). When all hold, fuse
            // these 3 ops into ONE shared scope (saves 1 cb/layer
            // for dense + hash-routed layers at pos>=31). Otherwise
            // fall back to the existing 2-cb path so the CPU
            // fallback inside flash_attn_decode_impl still fires.
            let n_raw = (state.pos + 1).min(raw_cap);
            let comp_rows_slice: Option<&[f32]> = if state.n_comp[layer_idx] > 0 {
                Some(&state.comp_kv_ring[layer_idx])
            } else {
                None
            };
            let n_comp_val = state.n_comp[layer_idx];
            let (comp_sel_slice, n_sel) = if let Some(ref sel) = indexer_allowed {
                (Some(sel.as_slice()), sel.len() as u32)
            } else {
                (None, 0u32)
            };
            let fast_path = head_dim == 512
                && n_raw % 32 == 0
                && n_raw > 0
                && comp_rows_slice.is_none()
                && comp_sel_slice.is_none();

            let n_hc_p = p.n_hc as usize;
            let hc_split_post_attn = &outs.split[n_hc_p..2 * n_hc_p];
            let hc_split_comb_attn =
                &outs.split[2 * n_hc_p..2 * n_hc_p + n_hc_p * n_hc_p];

            // M5 task #100 — bridged slow_path: when the layer has both
            // compressor rows AND indexer selections, the GPU compressor
            // flash_attn covers it. We can chain flash_attn → rope_back
            // → attn_output_proj → ffn-half (router → moe + shared) in
            // ONE BatchScope with one checkpoint between post-attn and
            // ffn-half (after_attn_hc must land on CPU for the still-CPU
            // hc_collapse_ffn step). Saves the cb commits between
            // flash_attn ↔ output_proj ↔ ffn_half — ~1 cb/layer on
            // V4 Flash compressor layers.
            //
            // Falls back to the existing fast_path / slow_path + run_ffn_half
            // when:
            //   - !with_ffn_half (attn-only mode)
            //   - fast_path (no compressor — fast_path branch already scopes)
            //   - silu_fidelity (uses moe_routed_step, not the shared chain)
            //   - hash-routing layers (need CPU probs for hash weights derivation)
            //   - comp_rows_slice / comp_sel_slice is None (GPU compressor
            //     path inapplicable; CPU flash_attn fallback fires)
            let silu_fidelity =
                std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
            let has_hash_routing = layer.moe.routing_table.is_some();
            let bridge_slow_path = with_ffn_half
                && !fast_path
                && !silu_fidelity
                && !has_hash_routing
                && comp_rows_slice.is_some()
                && comp_sel_slice.is_some();

            // rope_q forward. The bridge path ropes q on the GPU
            // (DS4_GPU_ROPE_Q, default-on) — q is uploaded raw and rope'd
            // in-place inside the flash scope via rope_tail_q_heads_in_place
            // (backward=false), the exact inverse of the backward rope this
            // path already runs on the flash output. All other paths upload
            // q as a slice, so they keep the CPU rope here.
            let gpu_rope_q = n_rot > 0
                && std::env::var("DS4_GPU_ROPE_Q").ok().as_deref() != Some("0");
            let q_heads_rot: Vec<f32> = if bridge_slow_path && gpu_rope_q {
                Vec::new() // unused — bridge path ropes q on GPU
            } else {
                let mut q = outs.q_heads.clone();
                if n_rot > 0 {
                    let rope_q_fwd = precompute_rope_tail_table(
                        n_rot,
                        pos,
                        false,
                        p.rope_freq_base,
                        p.rope_freq_scale,
                        p.rope_ext_factor,
                        p.rope_attn_factor,
                        p.rope_orig_ctx,
                    );
                    for head in q.chunks_mut(head_dim) {
                        let tail = &mut head[head_dim - n_rot..];
                        apply_rope_tail_with_table(tail, &rope_q_fwd, n_rot);
                    }
                }
                q
            };

            if bridge_slow_path {
                let kv_comp = comp_rows_slice.expect("comp_rows_slice");
                let comp_sel = comp_sel_slice.expect("comp_sel_slice");
                let q_dim = (p.n_head as usize) * head_dim;
                let n_groups = p.n_out_group as usize;
                let group_dim = q_dim / n_groups;
                let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                let out_low_dim = n_groups * n_lora_o;
                let want_q80 =
                    std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");

                let mut scope = self.dispatcher.batch_scope();
                let heads_b = if gpu_rope_q {
                    // Upload raw q, rope forward on GPU in-scope, flash via
                    // the resident-q variant — no CPU rope, q never leaves
                    // the GPU between rope and attention.
                    let q_b = scope.upload_f32(&outs.q_heads);
                    scope.rope_tail_q_heads_in_place(
                        &q_b,
                        p.n_head as usize,
                        head_dim,
                        p,
                        pos,
                        false,
                    )?;
                    // single-cb step 9: gather comp_rows from the GPU ring (no
                    // CPU comp_kv_ring upload). `kv_comp` (the CPU slice) is
                    // unused on this path but kept by the non-gpu-rope branch.
                    let _ = kv_comp;
                    let comp_ring = self.dispatcher.comp_ring_or_alloc(
                        layer_idx as u32,
                        comp_ring_rows(raw_cap) * head_dim * std::mem::size_of::<f32>(),
                    );
                    scope.flash_attn_decode_persistent_compressor_qbuf_gpuring(
                        layer_idx as u32,
                        p,
                        &q_b,
                        n_raw,
                        raw_cap,
                        &comp_ring,
                        comp_sel,
                        n_sel,
                        &layer.attn.attn_sinks,
                    )?
                } else {
                    scope.flash_attn_decode_persistent_compressor(
                        layer_idx as u32,
                        p,
                        &q_heads_rot,
                        n_raw,
                        raw_cap,
                        kv_comp,
                        comp_sel,
                        n_sel,
                        &layer.attn.attn_sinks,
                    )?
                };
                if n_rot > 0 {
                    scope.rope_tail_q_heads_in_place(
                        &heads_b,
                        p.n_head as usize,
                        head_dim,
                        p,
                        pos,
                        true,
                    )?;
                }
                let w_o_a_b = scope.weight_f32_lean_opt(&layer.attn.w_o_a);
                let w_o_b_b = scope.weight_f32_lean_opt(&layer.attn.w_o_b);
                let cur_hc_b = scope.upload_f32(&state.cur_hc);
                let post_b = scope.upload_f32(hc_split_post_attn);
                let comb_b = scope.upload_f32(hc_split_comb_attn);
                let after_db = scope.encode_attn_output_proj(
                    &heads_b,
                    &w_o_a_b,
                    &w_o_b_b,
                    &layer.attn.w_o_a_q8,
                    &layer.attn.w_o_b_q8,
                    &cur_hc_b,
                    &post_b,
                    &comb_b,
                    n_groups,
                    n_lora_o,
                    group_dim,
                    out_low_dim,
                    n_hc_p,
                    p.d_embd as usize,
                )?;
                // M5 #100 — port FFN-half hc_collapse_norm onto GPU
                // inside this same scope. The CPU `decode_attn_ffn_pre_with`
                // path delegates to `CpuAttentionDispatcher::hc_collapse_norm`
                // today (macos.rs:5892); using `scope.hc_collapse_norm`
                // here matches the attn-half encoder path
                // (`BatchScope::encode_layer_attn_half`) and saves the
                // per-layer CPU rms_norm + matvec + sinkhorn compute.
                //
                // Weight slices go through `weight_f32` (identity-keyed
                // cache) so each is uploaded once per session, not per
                // layer per token. `unit_gamma_hc` is shared across all
                // layers (created in this fn's prologue) → one cached
                // upload total.
                let (hc_ffn_fn_db, hc_ffn_is_f16) = scope.weight_hc(&layer.attn.hc_ffn_fn, layer.attn.hc_ffn_fn_f16());
                let hc_ffn_scale_db = scope.weight_f32(&layer.attn.hc_ffn_scale);
                let hc_ffn_base_db = scope.weight_f32(&layer.attn.hc_ffn_base);
                let hc_ffn_norm_gamma_db =
                    scope.weight_f32(&layer.attn.hc_ffn_norm_gamma);
                let unit_gamma_hc_db = scope.weight_f32(&unit_gamma_hc);
                let (split_db, _cur_db, normed_db) = scope
                    .hc_collapse_norm(
                        &after_db,
                        &hc_ffn_fn_db,
                        &hc_ffn_scale_db,
                        &hc_ffn_base_db,
                        &hc_ffn_norm_gamma_db,
                        p.n_hc as usize,
                        p.d_embd as usize,
                        p.hc_sinkhorn_iter as i32,
                        p.hc_eps,
                        p.rms_eps,
                        &unit_gamma_hc_db,
                        hc_ffn_is_f16,
                    )
                    .expect("hc_collapse_norm");

                // M5 #100 — DeferredBuf path when !want_q80 keeps
                // `normed_db` GPU-resident across router_logits + moe +
                // shared (no readback + re-upload of ffn_normed). The
                // checkpoint now reads only after_attn_hc + hc_split_ffn
                // (24 floats), not the full 16K-element normed.
                //
                // For want_q80=true the q8_0 round-trip needs CPU
                // access to ffn_normed, so that path still reads it
                // back and uses the CPU-slice variant.
                let router_logits_chain = |scope: &mut crate::deferred::BatchScope,
                                            h_norm_db: &crate::deferred::DeferredBuf|
                 -> Result<(crate::deferred::DeferredBuf, crate::deferred::DeferredBuf)> {
                    let (w_router_db, w_router_r_f16) = scope.weight_hc(&layer.moe.w_router, layer.moe.w_router_f16());
                    let probs_db = scope.encode_router_logits(
                        &w_router_db,
                        h_norm_db,
                        layer.moe.n_experts,
                        w_router_r_f16,
                    )?;
                    let bias_db = scope.weight_f32(&layer.moe.router_bias);
                    scope.encode_router_finalize(&probs_db, &bias_db)
                };

                let after_ffn_hc: Vec<f32> = if !want_q80 {
                    // M5 #100 / single-cb step 4a: GPU ffn-half path with
                    // NO mid-layer checkpoint. `split_db` ([pre|post|comb],
                    // 2*n_hc+n_hc² floats) stays GPU-resident and the
                    // post/comb sub-ranges are byte-offset-bound directly
                    // by `hc_expand_add_split` (deferred.rs:1239). The old
                    // path committed here (`commit_wait_read_multi(&[split_db])`)
                    // only to read 24 floats back and re-upload them as
                    // separate post/comb DeferredBufs — that whole cb
                    // boundary is now gone. The entire bridge layer
                    // (flash_attn → output_proj → hc_collapse → router →
                    // moe+shared → hc_expand_add) is ONE command buffer,
                    // one commit. Saves ~1 cb/layer × ~20 bridge layers.
                    let (sel_db, w_db) = router_logits_chain(&mut scope, &normed_db)
                        .expect("router_logits+finalize");
                    // SSD-streaming: drain so the GPU expert ids are readable,
                    // ensure them in the cache, and rebind the ids buffer with
                    // SLOT ids (the _db call then binds the cache pool).
                    let sel_db = if ssd_stream {
                        let w_read = scope.commit_wait_read_multi(&[&w_db]);
                        let ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(
                                sel_db.buffer().contents() as *const i32,
                                6,
                            )
                        }
                        .to_vec();
                        if std::env::var("DS4_LAYER_TRACE").is_ok()
                            && (ids.iter().any(|&i| i < 0 || i >= 1024)
                                || w_read[0].iter().any(|v| !v.is_finite()))
                        {
                            eprintln!(
                                "[router-bad] L{layer_idx} pos={} ids={ids:?} w={:?}",
                                state.pos, w_read[0]
                            );
                        }
                        // DS4_SSD_STUB=0 probe: tensors fully bound — pass ids through.
                        match self.dispatcher.streaming_expert_bind(layer_idx as u32, &ids) {
                            Some((slots, _g, _u, _d)) => {
                                let slot_db = scope.alloc_i32(6);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        slots.as_ptr(),
                                        slot_db.buffer().contents() as *mut i32,
                                        6,
                                    );
                                }
                                slot_db
                            }
                            None => sel_db,
                        }
                    } else {
                        sel_db
                    };
                    let (moe_db, shared_db) = scope
                        .encode_moe_and_shared_chain_with_router_bufs_db(
                            layer_idx as u32,
                            &normed_db,
                            &sel_db,
                            &w_db,
                            layer.moe.d_ffn.map_or(0, |n| n.get()),
                            &normed_db,
                            &layer.attn.w_shared_gate,
                            &layer.attn.w_shared_up,
                            &layer.attn.w_shared_down,
                            layer.attn.shared_dim,
                            &layer.attn.w_shared_gate_q8,
                            &layer.attn.w_shared_up_q8,
                            &layer.attn.w_shared_down_q8,
                        )
                        .expect("encode_moe_and_shared_chain_with_router_bufs_db");

                    if ssd_stream && std::env::var("DS4_LAYER_TRACE").is_ok() && layer_idx >= 58 {
                        let r = scope.commit_wait_read_multi(&[&moe_db, &shared_db]);
                        let nn = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
                        eprintln!(
                            "[ffn-out] L{layer_idx} pos={} moe_nan={} shared_nan={}",
                            state.pos, nn(&r[0]), nn(&r[1])
                        );
                    }
                    let after_ffn_db = scope
                        .hc_expand_add_split(
                            &shared_db,
                            &moe_db,
                            &after_db,
                            &split_db,
                            n_hc_p,
                            p.d_embd as usize,
                        )
                        .expect("hc_expand_add_split");
                    scope.flush_and_read(&after_ffn_db)
                } else {
                    // want_q80 path: still read ffn_normed to CPU for
                    // the q8_0 round-trip; CPU hc_expand_add_only
                    // closes the FFN-half. Keeps q80 fidelity until a
                    // GPU q8_0 quantize lands.
                    let cp = scope.commit_wait_read_multi(&[
                        &after_db, &normed_db, &split_db,
                    ]);
                    let mut cp_it = cp.into_iter();
                    let after_attn_hc =
                        cp_it.next().expect("after_attn_hc readback");
                    let ffn_normed =
                        cp_it.next().expect("ffn_normed readback");
                    let hc_split_ffn =
                        cp_it.next().expect("hc_split_ffn readback");

                    let prefix = AttnPrefixOut {
                        after_attn_hc,
                        hc_split_attn: outs.split.clone(),
                        normed: outs.normed.clone(),
                    };
                    let ffn_pre = FfnPreOut {
                        ffn_normed,
                        hc_split_ffn,
                    };
                    let h_norm = &ffn_pre.ffn_normed;
                    let ffn_in =
                        ds4_engine::forward::q8_0_round_trip(h_norm);

                    let h_norm_db = scope.upload_f32(h_norm);
                    let (sel_db, w_db) = router_logits_chain(&mut scope, &h_norm_db)
                        .expect("router_logits+finalize");
                    let (moe_db, shared_db) = scope
                        .encode_moe_and_shared_chain_with_router_bufs(
                            layer_idx as u32,
                            h_norm,
                            &sel_db,
                            &w_db,
                            layer.moe.d_ffn.map_or(0, |n| n.get()),
                            &ffn_in,
                            &layer.attn.w_shared_gate,
                            &layer.attn.w_shared_up,
                            &layer.attn.w_shared_down,
                            layer.attn.shared_dim,
                            true,
                            &layer.attn.w_shared_gate_q8,
                            &layer.attn.w_shared_up_q8,
                            &layer.attn.w_shared_down_q8,
                        )
                        .expect("encode_moe_and_shared_chain_with_router_bufs");
                    let outs_ffn = scope.flush_and_read_multi(&[&moe_db, &shared_db]);
                    let mut it = outs_ffn.into_iter();
                    let routed_out = it.next().expect("routed_out");
                    let shared_out = it.next().expect("shared_out");

                    decode_attn_ffn_post_with(
                        self.dispatcher,
                        self.dispatcher,
                        &layer.attn,
                        &prefix,
                        &ffn_pre,
                        &routed_out,
                        pos,
                        Some(&shared_out),
                    )
                };
                state.cur_hc = after_ffn_hc;

                per_layer.push(outs);
                if stage_prof { t_ffn += _sp.elapsed().as_secs_f64() * 1000.0; n_bridge += 1; }
                continue;
            }

            let _sp_attnout = std::time::Instant::now();
            let after_attn_hc: Vec<f32> = if fast_path {
                let q_dim = (p.n_head as usize) * head_dim;
                let n_groups = p.n_out_group as usize;
                let group_dim = q_dim / n_groups;
                let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                let out_low_dim = n_groups * n_lora_o;

                let scope = self.dispatcher.batch_scope();
                let heads_b = scope.flash_attn_decode_persistent(
                    layer_idx as u32,
                    p,
                    &q_heads_rot,
                    n_raw,
                    raw_cap,
                    &layer.attn.attn_sinks,
                )?;
                if n_rot > 0 {
                    scope.rope_tail_q_heads_in_place(
                        &heads_b,
                        p.n_head as usize,
                        head_dim,
                        p,
                        pos,
                        true,
                    )?;
                }
                let w_o_a_b = scope.weight_f32_lean_opt(&layer.attn.w_o_a);
                let w_o_b_b = scope.weight_f32_lean_opt(&layer.attn.w_o_b);
                let cur_hc_b = scope.upload_f32(&state.cur_hc);
                let post_b = scope.upload_f32(hc_split_post_attn);
                let comb_b = scope.upload_f32(hc_split_comb_attn);
                let after_db = scope.encode_attn_output_proj(
                    &heads_b,
                    &w_o_a_b,
                    &w_o_b_b,
                    &layer.attn.w_o_a_q8,
                    &layer.attn.w_o_b_q8,
                    &cur_hc_b,
                    &post_b,
                    &comb_b,
                    n_groups,
                    n_lora_o,
                    group_dim,
                    out_low_dim,
                    n_hc_p,
                    p.d_embd as usize,
                )?;
                scope.flush_and_read(&after_db)
            } else if head_dim == 512 {
                // M5 — fused GPU path for the layers that previously fell
                // to the CPU flash_attn (~96ms/token over ~23 layers):
                //   * raw layers (no compressor) whose n_raw isn't
                //     32-aligned, so fast_path skipped them;
                //   * compressor layers with comp rows but no indexer
                //     selection (the bridge_slow_path above didn't fire).
                // The GPU compressor kernel pads the KV workspace to 32
                // and masks, and only handles an explicit selection — so
                // raw layers pass an empty selection (n_sel=0, just the
                // raw rows), and dense-comp layers synthesize a full
                // selection ([0..n_comp)) == "attend all compressed rows"
                // == "select every row" (identical math to the CPU path).
                // flash_attn + GPU rope_tail + attn_output_proj run in
                // ONE cb.
                let empty_comp: [f32; 0] = [];
                let kv_comp: &[f32] = comp_rows_slice.unwrap_or(&empty_comp);
                let full_sel: Vec<u32>;
                let comp_sel: &[u32] = match (comp_rows_slice.is_some(), comp_sel_slice) {
                    (false, _) => &[],
                    (true, Some(s)) => s,
                    (true, None) => {
                        full_sel = (0..n_comp_val).collect();
                        &full_sel
                    }
                };
                let sel_count = comp_sel.len() as u32;
                let q_dim = (p.n_head as usize) * head_dim;
                let n_groups = p.n_out_group as usize;
                let group_dim = q_dim / n_groups;
                let n_lora_o = (if layer.attn.w_o_a.is_empty() { (layer.attn.w_o_a_q8.len() / 34) * 32 } else { layer.attn.w_o_a.len() }) / (n_groups * group_dim);
                let out_low_dim = n_groups * n_lora_o;

                let scope = self.dispatcher.batch_scope();
                let heads_b = scope.flash_attn_decode_persistent_compressor(
                    layer_idx as u32,
                    p,
                    &q_heads_rot,
                    n_raw,
                    raw_cap,
                    kv_comp,
                    comp_sel,
                    sel_count,
                    &layer.attn.attn_sinks,
                )?;
                if n_rot > 0 {
                    scope.rope_tail_q_heads_in_place(
                        &heads_b,
                        p.n_head as usize,
                        head_dim,
                        p,
                        pos,
                        true,
                    )?;
                }
                let w_o_a_b = scope.weight_f32_lean_opt(&layer.attn.w_o_a);
                let w_o_b_b = scope.weight_f32_lean_opt(&layer.attn.w_o_b);
                let cur_hc_b = scope.upload_f32(&state.cur_hc);
                let post_b = scope.upload_f32(hc_split_post_attn);
                let comb_b = scope.upload_f32(hc_split_comb_attn);
                let after_db = scope.encode_attn_output_proj(
                    &heads_b,
                    &w_o_a_b,
                    &w_o_b_b,
                    &layer.attn.w_o_a_q8,
                    &layer.attn.w_o_b_q8,
                    &cur_hc_b,
                    &post_b,
                    &comb_b,
                    n_groups,
                    n_lora_o,
                    group_dim,
                    out_low_dim,
                    n_hc_p,
                    p.d_embd as usize,
                )?;
                scope.flush_and_read(&after_db)
            } else {
                // Slow path (pre-Phase-F-#87): 2 separate cbs.
                let mut heads_back = self.dispatcher.flash_attn_decode_metal_persistent(
                    layer_idx as u32,
                    p,
                    &q_heads_rot,
                    n_raw,
                    raw_cap,
                    comp_rows_slice,
                    n_comp_val,
                    comp_sel_slice,
                    n_sel,
                    &layer.attn.attn_sinks,
                );
                if n_rot > 0 {
                    let rope_q_back = precompute_rope_tail_table(
                        n_rot,
                        pos,
                        true,
                        p.rope_freq_base,
                        p.rope_freq_scale,
                        p.rope_ext_factor,
                        p.rope_attn_factor,
                        p.rope_orig_ctx,
                    );
                    for head in heads_back.chunks_mut(head_dim) {
                        let tail = &mut head[head_dim - n_rot..];
                        apply_rope_tail_with_table(tail, &rope_q_back, n_rot);
                    }
                }
                AttentionDispatcher::attn_output_proj(
                    self.dispatcher,
                    p,
                    &heads_back,
                    &layer.attn.w_o_a,
                    &layer.attn.w_o_b,
                    &state.cur_hc,
                    hc_split_post_attn,
                    hc_split_comb_attn,
                )
            };

            if stage_prof { t_attnout += _sp_attnout.elapsed().as_secs_f64() * 1000.0; }
            let _sp_moe = std::time::Instant::now();
            if with_ffn_half {
                // ── 6-10. FFN-half via the Metal scope-aware helper —
                // fuses encode_router_finalize + encode_moe_and_shared_chain
                // into ONE cb (M5 task #99). Falls back to run_ffn_half
                // internally for silu_fidelity and hash-routing layers.
                let prefix = AttnPrefixOut {
                    after_attn_hc,
                    hc_split_attn: outs.split.clone(),
                    normed: outs.normed.clone(),
                };
                let after_ffn_hc = run_ffn_half_metal_scoped(
                    self.dispatcher,
                    layer_idx,
                    layer,
                    &prefix,
                    pos,
                );
                state.cur_hc = after_ffn_hc;

            } else {
                // Attn-only mode (M5.4.5.3 contract): stop at after_attn_hc.
                state.cur_hc = after_attn_hc;
            }
            if stage_prof {
                t_moe += _sp_moe.elapsed().as_secs_f64() * 1000.0;
                t_ffn += _sp.elapsed().as_secs_f64() * 1000.0;
                n_split += 1;
            }
            // DS4_RESID_TRACE=1: per-layer staged residual L2 + first values
            // (diagnostic for the staged-garbage bisection; staged path only —
            // chain keeps cur_hc GPU-resident).
            if !chain_on && std::env::var("DS4_RESID_TRACE").is_ok() {
                let l2: f64 = state.cur_hc.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                eprintln!("[resid] L{layer_idx} pos={} l2={l2:.4} c0={:.5} c1={:.5}",
                    state.pos, state.cur_hc.first().copied().unwrap_or(0.0),
                    state.cur_hc.get(1).copied().unwrap_or(0.0));
            }
            per_layer.push(outs);
        }

        if chain_on && std::env::var("DS4_CHAIN_PROFILE").is_ok() {
            eprintln!(
                "[chain] prologue={}us encode_loop={}us (commit={}us)",
                (t_token.elapsed().as_micros()).saturating_sub(t_chain_loop.elapsed().as_micros()),
                t_chain_loop.elapsed().as_micros(), chain_commit_us
            );
        }
        // Chain token-end: wait the async-committed layer cbs, read the final
        // cur_hc back to CPU (for the lm-head / next token), run all deferred
        // compressor finishes. With DS4_LAYERS_PER_CB>1 the chain_scope may
        // still hold uncommitted batched layers — commit them first so the
        // final wait covers everything.
        if let Some(s) = chain_scope.take() {
            let _tc = std::time::Instant::now();
            chain_cbs.extend(s.commit_detach());
            chain_commit_us += _tc.elapsed().as_micros();
            layers_in_batch = 0;
        }
        // Resident no-drain mode (DS4_RESIDENT_TAIL): when the whole token
        // stayed chained (`chain_hc` is Some), hand the resident HC buffer +
        // its async-committed cbs + deferred compressor finishes back to the
        // caller INSTEAD of draining. The caller (decode_token_via_first_half_
        // argmax) folds the output-head + lm-head + argmax in-scope on the
        // resident HC, flushes ONCE (which waits the chain by commit order),
        // then runs the finishes. Skips the cur_hc → CPU readback entirely.
        if let (Some(out), true) = (resident_out, chain_on && chain_hc.is_some()) {
            *out = Some(ResidentChain {
                hc: chain_hc.take().unwrap(),
                cbs: std::mem::take(&mut chain_cbs),
                finishes: std::mem::take(&mut chain_finishes),
            });
            if stage_prof {
                eprintln!("[stage] pos={} attn={t_attn:.1} comp={t_comp:.1} ffn={t_ffn:.1} ms", state.pos);
            }
            return Ok(FirstHalfOutputs { per_layer });
        }
        if chain_on && chain_hc.is_some() {
            let t_fc = std::time::Instant::now();
            let hc = chain_hc.take();
            let cbs = std::mem::take(&mut chain_cbs);
            let finishes = std::mem::take(&mut chain_finishes);
            // Clone cb0 + last-cb handles so we can read their GPU timestamps
            // AFTER finish_chain waits (timestamps are only valid post-completion).
            let probe_cbs = if startup_probe && !cbs.is_empty() {
                Some((cbs[0].to_owned(), cbs[cbs.len() - 1].to_owned()))
            } else {
                None
            };
            self.finish_chain(model, state, hc, cbs, finishes);
            if let Some((first, last)) = probe_cbs {
                let (gpu_start, _) = cb_gpu_start_end_secs(&first);
                let (_, gpu_end) = cb_gpu_start_end_secs(&last);
                let token_end = mach_now_secs();
                let commit = first_commit_secs.unwrap_or(token_start_secs);
                let ms = |a: f64, b: f64| (b - a) * 1.0e3;
                eprintln!(
                    "[startup] encode→1st_commit={:.2}ms  1st_commit→GPU_start={:.2}ms  GPU_span={:.2}ms  GPU_end→token_end={:.2}ms  | token={:.2}ms",
                    ms(token_start_secs, commit),
                    ms(commit, gpu_start),
                    ms(gpu_start, gpu_end),
                    ms(gpu_end, token_end),
                    ms(token_start_secs, token_end),
                );
            }
            if std::env::var("DS4_CHAIN_PROFILE").is_ok() {
                eprintln!(
                    "[chain] finish_chain_wall={}us total={}us",
                    t_fc.elapsed().as_micros(), t_token.elapsed().as_micros()
                );
            }
        }

        if stage_prof {
            eprintln!("[stage] pos={} attn={t_attn:.1} comp={t_comp:.1} ffn={t_ffn:.1} ms", state.pos);
            eprintln!(
                "[ffn split] attn_out={t_attnout:.1} moe={t_moe:.1} ms \
                 (split_layers={n_split} bridge_layers={n_bridge})"
            );
        }

        Ok(FirstHalfOutputs { per_layer })
    }

    /// Phase C scaffold. Reads back selection indices, compressor
    /// state mutations, indexer bitmaps, and hash router state from
    /// buffer-A outputs in ONE host sync; updates `state` accordingly.
    #[allow(dead_code)]
    fn batched_cpu_branching(
        &self,
        _state: &mut AttnStepState,
        _plan: &BatchedBranchingPlan,
    ) -> Result<()> {
        Ok(())
    }

    /// Phase C scaffold. Encodes `layers[l_split..n_layers]` + lm_head +
    /// argmax-top1 into command buffer B, commits, waits, returns logits.
    #[allow(dead_code)]
    fn encode_second_half(
        &self,
        _model: &ComposedModelWeights,
        _state: &AttnStepState,
        _l_split: LayerCutpoint,
    ) -> Result<Vec<f32>> {
        bail!("encode_second_half: Phase C not yet implemented")
    }

    /// Phase C tail slice (M4 #330m). Bit-identical to running
    /// `rms_norm(γ)` → (optional `q8_0_round_trip` if `want_q80`) →
    /// `matvec_f32(lm_head)` through the trait dispatcher, but packed
    /// into one `MTLCommandBuffer` with a single readback.
    ///
    /// `final_hidden` is the post-`output_hc_head_one` residual that
    /// `decode_step_with_attn` produces just before the final norm.
    /// Future `encode_second_half` will call this after running the
    /// upper layer half; for now it's exposed as a standalone primitive
    /// validated by `tests/tail_lm_head_batched_smoke.rs`.
    pub fn tail_lm_head_batched(
        &self,
        final_hidden: &[f32],
        model: &ComposedModelWeights,
        want_q80: bool,
    ) -> Vec<f32> {
        // Phase 2 (lm-head): full logits via the q8 lm-head matvec (reads
        // output.weight at 1 B/weight from the mmap, no 2.1 GB f32 dequant). The
        // q8 matvec == dequant(q8) matvec, so logits are bit-identical to the f32
        // tail when there's no q8 activation round-trip (want_q80=false, the
        // default). DS4_Q8_LMHEAD=0 or DS4_Q8_0_ACT=1 fall back to the f32 tail.
        let use_q8 = !model.lm_head_q8.is_empty()
            && !want_q80
            && std::env::var("DS4_Q8_LMHEAD").ok().as_deref() != Some("0");
        if use_q8 {
            return self
                .dispatcher
                .tail_lm_head_full_q8(
                    final_hidden,
                    &model.final_norm_gamma,
                    self.cfg.eps_rms,
                    &model.lm_head_q8,
                    model.vocab_size,
                )
                .expect("ds4_metal::tail_lm_head_full_q8 encoding failed");
        }
        assert!(
            !model.lm_head.is_empty(),
            "f32 lm_head was dropped but the q8 full-logits tail is unavailable \
             (DS4_Q8_LMHEAD=0 or DS4_Q8_0_ACT=1) — unset one of them"
        );
        self.dispatcher.tail_lm_head_batched(
            final_hidden,
            &model.final_norm_gamma,
            self.cfg.eps_rms,
            want_q80,
            &model.lm_head,
            model.vocab_size,
        )
    }

    /// Greedy tail: returns the GPU-argmax token id (no full-logit readback).
    pub fn tail_lm_head_argmax(
        &self,
        final_hidden: &[f32],
        model: &ComposedModelWeights,
        want_q80: bool,
    ) -> i32 {
        // Q4_0 lm-head (DS4_LOW_RAM + DS4_Q4_LMHEAD): re-quantized at load to
        // ~281 MB at 4-bit; preferred when present (f32 + q8 were dropped).
        if !model.lm_head_q4.is_empty() {
            return self.dispatcher.tail_lm_head_argmax_q4(
                final_hidden,
                &model.final_norm_gamma,
                self.cfg.eps_rms,
                &model.lm_head_q4,
                model.vocab_size,
            );
        }
        // Q8_0 lm-head fast path (default-on when the raw q8 bytes are present):
        // reads output.weight at 1 byte/weight (~562 MB) instead of the f32
        // dequant (~2.1 GB) — token-identical. DS4_Q8_LMHEAD=0 reverts to f32.
        let use_q8 = !model.lm_head_q8.is_empty()
            && std::env::var("DS4_Q8_LMHEAD").ok().as_deref() != Some("0");
        if use_q8 {
            return self.dispatcher.tail_lm_head_argmax_q8(
                final_hidden,
                &model.final_norm_gamma,
                self.cfg.eps_rms,
                &model.lm_head_q8,
                model.vocab_size,
            );
        }
        assert!(
            !model.lm_head.is_empty(),
            "f32 lm_head was dropped (DS4_LOW_RAM) but DS4_Q8_LMHEAD=0 selected the f32 tail — \
             unset one of them"
        );
        self.dispatcher.tail_lm_head_argmax(
            final_hidden,
            &model.final_norm_gamma,
            self.cfg.eps_rms,
            want_q80,
            &model.lm_head,
            model.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase B contract: cutpoint helper returns n_layers / 2.
    #[test]
    fn cutpoint_middle_halves_layer_count() {
        assert_eq!(LayerCutpoint::middle(43).0, 21);
        assert_eq!(LayerCutpoint::middle(2).0, 1);
        assert_eq!(LayerCutpoint::middle(0).0, 0);
    }

    /// Plan starts empty — Phase C populates it.
    #[test]
    fn plan_starts_empty() {
        let p = BatchedBranchingPlan::default();
        assert!(p.compressor_layers.is_empty());
        assert!(p.indexer_layers.is_empty());
        assert!(p.hash_router_layers.is_empty());
    }
}
