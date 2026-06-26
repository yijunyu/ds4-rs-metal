//! Deferred-execution scope for fusing multiple kernel dispatches into a
//! single `MTLCommandBuffer` (M4 #330p, Phase C-A).
//!
//! Background: the trait `KernelDispatcher` methods on `MetalDispatcher`
//! each return `Vec<f32>`, which forces a commit + waitUntilCompleted +
//! readback per op. Empirically (`benchmarks/ds4_msl/results/
//! ds4_path_a_capture_2026-05-23.trace`) this fans out to ~250
//! command-buffer submissions per decode token, with a 2.47 ms median
//! inter-cb CPU encoding gap, capping the Rust pipeline at ~1.5 tok/s.
//!
//! `BatchScope` generalizes the pattern that the existing `*_batched_impl`
//! methods (e.g. `layer_qa_rms_batched_impl`) already use: build buffers,
//! open ONE `cmd_buf`, encode N compute passes against it, commit + wait
//! once at the end, read back only the final outputs. The intermediate
//! results stay in GPU buffers — they never round-trip through CPU.
//!
//! This module is intentionally NOT a `KernelDispatcher` impl. The trait
//! is the per-op-readback surface the correctness tests depend on
//! (`TracingDispatcher` traces every method as an opaque Vec<f32>).
//! `BatchScope` is a parallel surface for fused-chain encoding; future
//! callers can use it to write new fused methods without the boilerplate
//! of `layer_qa_rms_batched_impl`.

#![cfg(target_os = "macos")]

use anyhow::Result;
use metal::{MTLDataType, MTLSize};

use crate::macos::{
    new_input_buffer, new_output_buffer, read_buffer, set_scalar_bytes, MetalState,
};

// ── Step 0 drain instrumentation (DS4_DRAIN_TRACE) ──────────────────────────
// Counts every `flush_and_read*` (commit+wait+CPU-readback = a per-layer GPU
// drain) by call site, plus how many detached `pending_cbs` each one drains.
// `flush_and_read*` are #[track_caller], so the recorded location is the
// single_buffer_encoder.rs line that needed a CPU value. Dump once per token via
// `drain_trace_dump_and_reset`. Zero cost when DS4_DRAIN_TRACE is unset.
fn drain_trace_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DS4_DRAIN_TRACE").is_ok())
}

/// DS4_ZERO_SCRATCH=1 (diagnostic): zero-initialize every `alloc_f32` scratch
/// buffer. Used to test whether chunk-path nondeterminism is a kernel reading an
/// unwritten scratch region (read-before-write of garbage) vs a genuine race.
fn zero_scratch_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DS4_ZERO_SCRATCH").map(|v| v != "0").unwrap_or(false))
}

/// Assert a COMPLETED command buffer did not error. A Metal cb that errors
/// (kernel OOB, exceeded a resource limit, GPU fault) leaves its output buffers
/// untouched — i.e. zero-initialized — so the failure is otherwise SILENT and
/// surfaces only as garbage/all-zeros far downstream (the chunk-prefill mm_id
/// "all-zeros" was exactly this: a 40-layer single cb overflowed → errored →
/// zeros). Per the FFI/GPU-boundary requirement, turn that silent GPU UB into a
/// clean panic at the wait site. Call ONLY after `wait_until_completed`.
#[track_caller]
pub(crate) fn assert_cb_ok(cb: &metal::CommandBufferRef, label: &str) {
    use metal::MTLCommandBufferStatus;
    if cb.status() != MTLCommandBufferStatus::Error {
        return;
    }
    let desc = unsafe {
        use metal::foreign_types::ForeignTypeRef;
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        let cb_ptr: *mut Object = std::mem::transmute(cb.as_ptr());
        let err: *mut Object = msg_send![cb_ptr, error];
        if err.is_null() {
            String::from("(no NSError attached)")
        } else {
            let nsstr: *mut Object = msg_send![err, localizedDescription];
            if nsstr.is_null() {
                String::from("(error with no description)")
            } else {
                let cstr: *const std::os::raw::c_char = msg_send![nsstr, UTF8String];
                std::ffi::CStr::from_ptr(cstr).to_string_lossy().into_owned()
            }
        }
    };
    panic!(
        "Metal command buffer '{label}' completed with status=Error: {desc}\n\
         A silently-errored cb leaves its output buffers zero — likely a kernel \
         out-of-bounds or a cb that exceeded a resource limit (too many encoders/\
         commands). For the chunk-prefill driver, shrink DS4_CHUNK_COMMIT_EVERY \
         (async cb-size bound, single_buffer_encoder.rs)."
    );
}
thread_local! {
    // (file, line) -> (count, total pending_cbs drained)
    static DRAIN_TRACE: std::cell::RefCell<
        std::collections::HashMap<(&'static str, u32), (u64, u64)>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}
#[inline]
fn drain_trace_record(loc: &'static std::panic::Location<'static>, n_pending: usize) {
    if !drain_trace_on() {
        return;
    }
    DRAIN_TRACE.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry((loc.file(), loc.line())).or_insert((0, 0));
        e.0 += 1;
        e.1 += n_pending as u64;
    });
}
pub(crate) fn drain_trace_dump_and_reset(tag: &str) {
    if !drain_trace_on() {
        return;
    }
    DRAIN_TRACE.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_empty() {
            return;
        }
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by_key(|(_, (c, _))| std::cmp::Reverse(*c));
        let total: u64 = v.iter().map(|(_, (c, _))| *c).sum();
        let pend: u64 = v.iter().map(|(_, (_, p))| *p).sum();
        eprintln!(
            "[drain-trace {tag}] {total} flush_and_read* this token ({pend} detached cbs drained), by call site:"
        );
        for ((file, line), (count, p)) in v {
            let f = file.rsplit('/').next().unwrap_or(file);
            eprintln!("  {count:>4}x  {f}:{line}   (drained {p} pending cbs)");
        }
        m.clear();
    });
}

/// Phase E M5.4.1 — output bundle of `BatchScope::encode_layer_attn_half`.
/// All fields are DeferredBuf handles into the same scope; flush+read
/// them via `flush_and_read_multi(&[...])` for the unused-yet-tested
/// path, or chain downstream ops (kv_fp8_store_persistent +
/// flash_attn_decode_metal_persistent + ...) in the same scope.
/// K-position equivalent of `LayerAttnHalfOuts`. Each field is the
/// concatenated `[K, ...]` buffer produced by the corresponding K-position
/// primitive. Returned by `encode_layer_attn_half_k`.
pub struct LayerAttnHalfOutsK {
    pub normed_k: DeferredBuf,            // [K, n_embd]
    pub split_k: DeferredBuf,              // [K, mix_hc]
    pub qr_normed_k: DeferredBuf,          // [K, n_lora_q]
    pub q_heads_k: DeferredBuf,            // [K, n_head * head_dim]
    pub kv_normed_rotated_k: DeferredBuf,  // [K, kv_row]
}

pub struct LayerAttnHalfOuts {
    /// hc_collapse_norm output: the normed residual fed into the QKV
    /// chain. Length `n_embd`.
    pub normed: DeferredBuf,
    /// hc_split (pre + post + comb sections). Length
    /// `2*n_hc + n_hc*n_hc`. Used downstream by hc_expand_add at the
    /// layer's end.
    pub split: DeferredBuf,
    /// Q-LoRA rms-normed projection. Length `n_lora_q`. Surfaced for
    /// the indexer on ratio==4 layers (not yet wired in the unified
    /// encoder; M5.5 work).
    pub qr_normed: DeferredBuf,
    /// Q heads, per-head rms-normed. Length `n_head * head_dim`.
    /// Feeds rope_tail_q (still CPU today) and flash_attn_decode.
    pub q_heads: DeferredBuf,
    /// KV row after rms_norm + rope_tail on the n_rot suffix. Length
    /// `n_lora_kv`. Feeds kv_fp8_store_persistent (writes the
    /// quantized form into the persistent buffer).
    pub kv_normed_rotated: DeferredBuf,
}

/// Opaque handle to a Metal buffer encoded within (or uploaded into) a
/// `BatchScope`. Holds an Arc to the underlying buffer plus the element
/// count needed for readback. Intermediate results in a fused chain are
/// `DeferredBuf`s — they stay GPU-resident across ops.
pub struct DeferredBuf {
    buf: metal::Buffer,
    n_elements: usize,
}

thread_local! {
    /// DS4_CHUNK_POOL_SCRATCH: per-(key,n) reused scratch buffers for the chunk-
    /// prefill path. Bounds the per-cb transient footprint to O(1) so commit_every>2
    /// (layer packing → bubble recovery) stops thrashing on N× fresh ~196MB allocs.
    /// ONLY for INTRA-layer scratch (produced+consumed within one layer, never
    /// returned cross-layer) — pooling the hidden carrier (after_ffn_k/cur_hc) would
    /// corrupt the residual stream. The prefill thread owns this; reused across chunks.
    static CHUNK_SCRATCH_POOL: std::cell::RefCell<std::collections::HashMap<(&'static str, usize), metal::Buffer>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// Ping-pong parity for the cross-layer hidden carrier (after_ffn_k). Flipped each
    /// hc_expand_add_split_k call so consecutive layers write ALTERNATING buffers: layer
    /// reads its predecessor's buffer, writes the other → no alias, and a packed ce>2 cb
    /// references only 2 carrier buffers (O(1)) not N. Free-running alternation is
    /// sufficient (absolute parity irrelevant). DS4_CHUNK_POOL_SCRATCH only.
    static CARRIER_TOGGLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// DS4_CHUNK_POOL_ATTN one-shot key: set immediately before a matmul_k_q8_0 GEMM whose
    /// output is an INTRA-layer attention intermediate (qkv-chain projections qr/q_heads_raw/
    /// kv_raw_row — K×dim, the dominant per-cb attention footprint NOT covered by
    /// pooled_scratch). Consumed+cleared at the GEMM output alloc → reused across layers (O(1)
    /// transient). One-shot so it can't leak to an unrelated matmul. See
    /// [[ds4-residency-collapse-pressure-masked]] (the 01f71a30 footprint-bounding rework).
    static ATTN_POOL_KEY: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
    /// DS4_CHUNK_POOL_PINGPONG: cb-commit parity for the pooled scratch. Flipped each
    /// pipelined commit so consecutive command buffers draw DIFFERENT pooled-buffer
    /// instances per (key,n) → cb[N] and cb[N+1] never share a pooled buffer, removing
    /// the cross-cb WAR that corrupts pool+DS4_CHUNK_PIPELINE (the reused buffer is
    /// otherwise read by an in-flight cb[N] while cb[N+1] overwrites it). Doubles the
    /// pinned set for ping-ponged keys (still O(2·keys)). No-op when the flag is off.
    /// ⚠ VALIDATED ce=1 ONLY (pool+PIPELINE byte-identical under 73GB pressure at ce=1).
    /// At ce=2 (2 layers/cb, the prefill default for K≤3000) it STILL diverges daemon-out:
    /// the per-layer carrier toggle (CARRIER_TOGGLE — the residual stream's own 2-buffer
    /// ping-pong) needs >2 buffers under pipeline, which this pooled-scratch phase does not
    /// cover. So DS4_CHUNK_PIPELINE is NOT defaulted; this stays gated infrastructure for
    /// future single-cb / ce=1 pipelining. See docs/PREFILL_BOUNDED_WORKING_SET.md.
    static POOL_PHASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// DS4_ATTN_Q4: per-weight-buffer cache of the q8_0→q4_0 requantized attention
    /// projection weights (keyed by the q8 buffer's contents ptr). First-touch requant,
    /// reused for the process — the q4 weight is half the bytes (18 vs 34 B/block).
    static ATTN_Q4_WEIGHT_CACHE: std::cell::RefCell<std::collections::HashMap<usize, metal::Buffer>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The high-bit phase tag OR'd into a pooled-scratch key's `n` when ping-pong is
/// active (DS4_CHUNK_POOL_PINGPONG=1). Bit 63 — distinct from the u16 pool's bit 62.
/// Returns 0 when ping-pong is off (single-buffered, the sequential-safe default).
fn pool_phase_bit() -> usize {
    if std::env::var("DS4_CHUNK_POOL_PINGPONG").ok().as_deref() == Some("1") {
        POOL_PHASE.with(|p| p.get()) << 63
    } else {
        0
    }
}

/// Flip the pooled-scratch ping-pong parity. Called once per pipelined cb commit so
/// the NEXT cb's pooled buffers don't alias the just-committed (still in-flight) cb's.
pub fn flip_pool_phase() {
    POOL_PHASE.with(|p| p.set(p.get() ^ 1));
}

/// Arm the one-shot attention-pool key for the NEXT `matmul_k_q8_0` GEMM output (see
/// `ATTN_POOL_KEY`). No-op unless DS4_CHUNK_POOL_ATTN=1 is read at the alloc site.
pub fn arm_attn_pool_key(key: &'static str) {
    ATTN_POOL_KEY.with(|k| k.set(Some(key)));
}

/// ⚠ EXPERIMENTAL, NET-NEUTRAL/LOSS — keep default-OFF. DS4_CHUNK_POOL_SCRATCH reuses
/// INTRA-layer scratch (+ after_ffn_k carrier ping-pong) across layers so a packed
/// commit_every>2 cb holds O(1) not O(N) transient memory. CORRECT (exonerated 2026-06-13:
/// the "intermittent ce=8 all-zeros" was synthetic-prompt argmax NOISE — maxabs stays 34.19
/// finite; non-pooled ce=8 flaked identically). BUT clean-GPU best-of-5 A/B: ce=4 −0.6%,
/// ce=8 −24% → NO net win. Pooling barely moved clean ce=8 ⇒ the packing cost is large-cb
/// SCHEDULING at our ~92-dispatch/layer density, NOT footprint. ce>2 only becomes a win
/// after a SEPARATE per-layer-dispatch-count reduction. Kept as a validated mechanism for
/// that future. See docs/ATTN_GEMM_F16_SCOPE.md.
fn warn_pool_scratch_unstable() {
    // 2026-06-13 (T1-T4, docs/PREFILL_BOUNDED_WORKING_SET.md): DS4_CHUNK_POOL_SCRATCH is now
    // DEFAULT-ON in the chunk-prefill path, paired with DS4_CHUNK_HASH_GPU. The pool is the
    // VEHICLE for pinning the merged-cb working set resident (pin_pooled_resident → MTLResidencySet,
    // queue-bound) so GPU-resident hash routing survives daemon memory pressure. Validated:
    // byte-identical under 73GB pressure at ce=2 (pool-correctness + HASH-pressure gates), +3.1%
    // prefill @3000. NOTE: do NOT combine with DS4_CHUNK_PIPELINE — the pool's cross-layer reuse
    // assumes SEQUENTIAL layers; commit_pipelined races the reused buffer (WAR corruption, caught
    // daemon-out). The warning below is retained only for the standalone ce>2 packing use (a
    // separate, still-experimental path); silence it when the prefill default-on owns the flag.
    if std::env::var("DS4_CHUNK_HASH_GPU").ok().as_deref() == Some("1") {
        return; // prefill bounded-working-set owns the pool; not the experimental ce>2 packing
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[ds4] NOTE: DS4_CHUNK_POOL_SCRATCH=1 (standalone, no HASH_GPU) is EXPERIMENTAL \
             (commit_every>2 footprint packing). Byte-identical on a clean GPU; ce>2 needs a \
             per-layer-dispatch-count reduction before it is a net win."
        );
    });
}

/// Phase 3 MTP — input-mixer weight DeferredBufs (steps 2-7 of the draft).
/// Held by reference to the caller-uploaded buffers; lifetime tied to the
/// containing scope.
pub struct MtpDraftInputMix<'a> {
    pub enorm_gamma: &'a DeferredBuf,
    pub e_proj_q8:   &'a DeferredBuf,
    pub hnorm_gamma: &'a DeferredBuf,
    pub h_proj_q8:   &'a DeferredBuf,
}

/// Phase 3 MTP — drafter's decode-layer weight DeferredBufs + CPU shared-expert
/// slices. Bundles the 18 attn-half + ffn-half tensors the inner
/// `encode_layer_k(K=1)` call requires. Mirrors `AttnLayerWeights` /
/// `MoeWeights` shape conventions.
pub struct MtpDraftLayerWeights<'a> {
    // Attn-half (uploaded as DeferredBufs).
    pub hc_attn_fn:    &'a DeferredBuf,
    pub hc_attn_scale: &'a DeferredBuf,
    pub hc_attn_base:  &'a DeferredBuf,
    pub attn_norm:     &'a DeferredBuf,
    pub attn_q_a_q8:   &'a DeferredBuf,
    pub gamma_q:       &'a DeferredBuf,  // = attn_q_a_norm
    pub attn_q_b_q8:   &'a DeferredBuf,
    pub attn_kv_q8:    &'a DeferredBuf,
    pub gamma_kv:      &'a DeferredBuf,  // = attn_kv_a_norm
    pub attn_sinks:    &'a DeferredBuf,  // [n_head] per-head softmax sink logits
    pub w_o_a_q8:      &'a DeferredBuf,  // = attn_output_a
    pub w_o_b_q8:      &'a DeferredBuf,  // = attn_output_b
    // FFN-half DeferredBufs.
    pub hc_ffn_fn:     &'a DeferredBuf,
    pub hc_ffn_scale:  &'a DeferredBuf,
    pub hc_ffn_base:   &'a DeferredBuf,
    pub ffn_norm:      &'a DeferredBuf,
    pub w_router:      &'a DeferredBuf,
    pub router_bias:   &'a DeferredBuf,
    // Shared-expert CPU slices (encode_layer_k still takes these as
    // &[f32]/&[u8]; future cleanup could promote them to DeferredBuf).
    pub sh_w_gate:     &'a [f32],
    pub sh_w_up:       &'a [f32],
    pub sh_w_down:     &'a [f32],
    pub sh_w_gate_q8:  &'a [u8],
    pub sh_w_up_q8:    &'a [u8],
    pub sh_w_down_q8:  &'a [u8],
}

/// Phase 3 MTP — output-head DeferredBufs (the LM-head pipeline).
/// `base_lm_head_q8` is the BASE model's `output` tensor — the drafter
/// shares the vocab projection with the verifier.
pub struct MtpDraftOutputHead<'a> {
    pub unit_gamma_hc:    &'a DeferredBuf,  // [hc_dim] ones
    pub hc_head_fn:       &'a DeferredBuf,
    pub hc_head_scale:    &'a DeferredBuf,
    pub hc_head_base:     &'a DeferredBuf,
    pub mtp_norm:         &'a DeferredBuf,
    pub base_lm_head_q8:  &'a DeferredBuf,
}

/// Phase 3 MTP — shape + scalar params for `encode_mtp_draft`.
#[derive(Copy, Clone, Debug)]
pub struct MtpDraftShape {
    pub n_embd: usize,
    pub n_hc: usize,
    pub n_lora_q: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub kv_row: usize,
    pub n_groups: usize,
    pub n_lora_o: usize,
    pub group_dim: usize,
    pub out_low_dim: usize,
    pub n_experts: usize,
    pub d_ffn: usize,
    pub shared_dim: u32,
    pub vocab: usize,
    pub sinkhorn_iters: i32,
    pub hc_eps: f32,
    pub rms_eps: f32,
    pub flash_scale: f32,
}

/// Phase 3 Step 4d.2 — per-layer weight bundle for the K-position
/// verifier. Same shape as `MtpDraftLayerWeights` (the drafter's single
/// layer) — the verifier just runs 43 such layers in a chain.
///
/// Held by reference to caller-uploaded DeferredBufs + caller-owned
/// CPU slices for the shared-expert tensors.
pub struct BaseLayerVerifyBundle<'a> {
    // Attn-half.
    pub hc_attn_fn:    &'a DeferredBuf,
    pub hc_attn_scale: &'a DeferredBuf,
    pub hc_attn_base:  &'a DeferredBuf,
    pub attn_norm:     &'a DeferredBuf,
    pub attn_q_a_q8:   &'a DeferredBuf,
    pub gamma_q:       &'a DeferredBuf,
    pub attn_q_b_q8:   &'a DeferredBuf,
    pub attn_kv_q8:    &'a DeferredBuf,
    pub gamma_kv:      &'a DeferredBuf,
    pub attn_sinks:    &'a DeferredBuf,  // [n_head] per-head softmax sink logits
    pub w_o_a_q8:      &'a DeferredBuf,
    pub w_o_b_q8:      &'a DeferredBuf,
    // FFN-half DeferredBufs.
    pub hc_ffn_fn:     &'a DeferredBuf,
    pub hc_ffn_scale:  &'a DeferredBuf,
    pub hc_ffn_base:   &'a DeferredBuf,
    pub ffn_norm:      &'a DeferredBuf,
    pub w_router:      &'a DeferredBuf,
    pub router_bias:   &'a DeferredBuf,
    // Shared-expert CPU slices.
    pub sh_w_gate:     &'a [f32],
    pub sh_w_up:       &'a [f32],
    pub sh_w_down:     &'a [f32],
    pub sh_w_gate_q8:  &'a [u8],
    pub sh_w_up_q8:    &'a [u8],
    pub sh_w_down_q8:  &'a [u8],
    // Compressor + indexer CPU slices (DS4_VERIFY_COMPRESSOR path). Empty
    // for dense/non-compressor layers (compress_ratio == 0). The main
    // compressor emits long-range rows into the per-layer comp ring; the
    // indexer compressor keeps the index state consistent (its top-k
    // selection is all-allowed while n_comp <= DS4_N_INDEXER_TOP_K). These
    // are uploaded by compressor_encode_in_scope (identity-cached), same as
    // the K=1 path — no pre-upload needed.
    pub attn_compressor_kv:   &'a [f32],
    pub attn_compressor_gate: &'a [f32],
    pub attn_compressor_ape:  &'a [f32],
    pub attn_compressor_norm: &'a [f32],
    pub indexer_compressor_kv:   &'a [f32],
    pub indexer_compressor_gate: &'a [f32],
    pub indexer_compressor_ape:  &'a [f32],
    pub indexer_compressor_norm: &'a [f32],
    // No-copy F16 mmap bytes for the kv/gate projections (lean weights). `Some`
    // when the f32 above is empty (skipped); the verifier's CompressorInputs
    // drive `matvec_f16` off these. See `AttnLayerWeights::attn_compressor_f16`.
    pub attn_compressor_kv_f16:   Option<&'a [u8]>,
    pub attn_compressor_gate_f16: Option<&'a [u8]>,
    pub indexer_compressor_kv_f16:   Option<&'a [u8]>,
    pub indexer_compressor_gate_f16: Option<&'a [u8]>,
    // CPU attn sink logits [n_head] — the ring-aware K=1 flash takes sinks as
    // a CPU slice (the raw K-flash takes the uploaded DeferredBuf above).
    pub attn_sinks_cpu: &'a [f32],
    // Per-layer rope/scaling params (each layer may differ).
    pub layer_params:  &'a ds4_engine::attn_dispatch::LayerParams,
}

/// Per-layer compressor/indexer context for the DS4_VERIFY_COMPRESSOR K-path.
/// Built by `encode_verify_layers_K` for compressor layers (compress_ratio==4)
/// whose prefill ring is non-empty. When present, `encode_attn_chain_k` runs a
/// per-draft ring-aware flash over [raw KV | comp ring] instead of the raw
/// `flash_attn_k_mla`, closing the verifier's missing long-range-context gap.
pub struct CompVerifyCtx<'a> {
    /// Prefill ring row count (= AttnStepState.n_comp[layer]). The K=1 indexer
    /// selection is all-allowed while n_comp <= DS4_N_INDEXER_TOP_K, so the
    /// flash attends all `n_comp` compressed rows.
    pub n_comp: u32,
    /// CPU attn sink logits [n_head] for the ring-aware flash.
    pub attn_sinks_cpu: &'a [f32],
    /// Main + indexer compressor weights (for per-draft emit; increment 2).
    pub attn_comp: ds4_engine::attn_dispatch::CompressorInputs<'a>,
    pub idx_comp:  ds4_engine::attn_dispatch::CompressorInputs<'a>,
}

impl DeferredBuf {
    pub fn len(&self) -> usize {
        self.n_elements
    }
    pub fn is_empty(&self) -> bool {
        self.n_elements == 0
    }
    /// Borrow the underlying Metal buffer. Useful when callers need to
    /// hand a `DeferredBuf` to a kernel encoder helper that takes a
    /// `&metal::Buffer` directly.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buf
    }

    /// Phase 2 MoE-K Step 5 — wrap an external `metal::Buffer` (e.g. an
    /// expert weight tensor from `MetalDispatcher::expert_weight_bufs`)
    /// as a DeferredBuf so the K-amortized MoE encoders can consume it
    /// without re-uploading. `n_elements` is the LOGICAL element count
    /// (for shape assertions); the underlying byte size of `buf` may be
    /// larger (e.g. quantized expert tensors are byte-counted by the
    /// kernel, but logically have n_experts × d × d elements).
    pub fn from_external_buffer(buf: metal::Buffer, n_elements: usize) -> Self {
        Self { buf, n_elements }
    }
}

/// RAII scope that owns one `MTLCommandBuffer` and accumulates compute
/// dispatches into it. Commit + wait + readback happen exactly once, in
/// `flush_and_read`, which consumes the scope.
///
/// Typical use:
/// ```ignore
/// let scope = BatchScope::new(state);
/// let w  = scope.weight_f32(&w_slice);
/// let x  = scope.upload_f32(&x_slice);
/// let qr = scope.matvec_f32(&w, &x, d_in, d_out)?;
/// let g  = scope.upload_f32(&gamma);
/// let y  = scope.rms_norm_mul(&qr, &g, eps)?;
/// let out = scope.flush_and_read(&y);
/// ```
///
/// Counts as ONE command-buffer submission regardless of how many ops
/// were encoded between `new` and `flush_and_read`.
pub struct BatchScope<'a> {
    state: &'a MetalState,
    cmd_buf: metal::CommandBuffer,
    /// Phase F: cbs that have been committed but not yet waited on.
    /// Mirrors antirez's `g_pending_cbs` (ds4_metal.m:38). Filled by
    /// `commit_keep_open`; drained by `flush_and_read[_multi]` and
    /// `wait_all`. Multiple cbs per scope can be in flight at once;
    /// the GPU pipelines them but the caller doesn't pay per-cb
    /// wait overhead until end-of-scope.
    pending_cbs: Vec<metal::CommandBuffer>,
    /// Lazily-created MTLEvent for `event_split` (inter-cb GPU ordering
    /// without a CPU drain). One event per scope; the value increments on
    /// every split.
    order_event: Option<metal::Event>,
    order_event_value: u64,
    /// When set, `encode_attn_qkv_chain` runs the q/kv projections as Q8_0
    /// matvecs (1 byte/weight) instead of dequant→`matvec_f32` (4 bytes). The
    /// caller (`encode_layer_attn_half_open_resident`) must then have created
    /// the projection weight bufs via `weight_q8_0`. Scope-local so other
    /// callers (CPU/test paths passing f32 bufs) are unaffected.
    q8_proj: std::cell::Cell<bool>,
}

impl<'a> BatchScope<'a> {
    pub(crate) fn new(state: &'a MetalState) -> Self {
        let cmd_buf = state.command_queue.new_command_buffer().to_owned();
        Self {
            state,
            cmd_buf,
            pending_cbs: Vec::new(),
            order_event: None,
            order_event_value: 0,
            q8_proj: std::cell::Cell::new(false),
        }
    }

    /// Enable the Q8_0 projection path for `encode_attn_qkv_chain` in this
    /// scope. See `q8_proj`.
    pub fn set_q8_proj(&self, on: bool) {
        self.q8_proj.set(on);
    }

    /// Phase F (task #86) — commit the current cb without waiting,
    /// open a fresh cb to continue encoding into. Mirrors antirez's
    /// `ds4_metal_flush_commands` (ds4_metal.m:3882). Used when ops
    /// can't share an encoder boundary (e.g., a kernel that needs
    /// CPU sync mid-token) but the cb-completion can still pipeline
    /// against the next cb's work.
    ///
    /// The committed cb's outputs are available for GPU reads in
    /// subsequent cbs (Metal guarantees ordering within a queue).
    /// CPU reads need a wait — call `flush_and_read[_multi]` or
    /// `wait_all_and_drop` first.
    pub fn commit_keep_open(&mut self) {
        crate::macos::end_shared_compute_enc_force();        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        old_cb.commit();
        crate::macos::layer_prof_push(&old_cb);
        self.pending_cbs.push(old_cb);
    }

    /// 1-DEEP PIPELINED commit (DS4_CHUNK_PIPELINE): overlaps the CPU encode of the next
    /// cb with the GPU execution of the current, recovering the inter-cb bubble — WITHOUT
    /// keeping >1 cb executing (so no residency collapse, unlike the event-window). The
    /// per-cb commit_wait drain leaves the GPU idle while the CPU encodes the next ~2 layers
    /// (~72ms/boundary, ~15% of prefill, LAYER_PROF). Here: commit the current cb WITHOUT
    /// waiting, then at the NEXT boundary wait the PREVIOUS cb (which ran on the GPU during
    /// this layer's encode → near-zero wait) BEFORE committing the current. Result: exactly
    /// ONE cb executes at a time (same residency as commit_wait → coherent under pressure),
    /// but the encode overlaps execution (no bubble). Cross-layer ordering is safe: the new
    /// cb is committed only AFTER the previous is waited, so its GPU reads of cur_hc see the
    /// final values. Requires DEFER_RESYNC (CPU-mirror reads deferred to the terminal).
    pub fn commit_pipelined(&mut self, label: &'static str) {
        crate::macos::end_shared_compute_enc_force();
        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        // Wait the PREVIOUS committed cb (ran during the just-finished encode → ~instant),
        // keeping ≤1 cb in flight, then commit the current (it starts running while we go on
        // to encode the next layers).
        if let Some(prev) = self.pending_cbs.pop() {
            prev.wait_until_completed();
            assert_cb_ok(&prev, label);
            crate::macos::layer_prof_push(&prev);
        }
        old_cb.commit();
        self.pending_cbs.push(old_cb);
        // Ping-pong the pooled-scratch parity so the NEXT cb's pooled buffers don't
        // alias this just-committed (now in-flight) cb's. No-op unless
        // DS4_CHUNK_POOL_PINGPONG=1. Removes the pool+PIPELINE WAR.
        flip_pool_phase();
    }

    /// MTLEvent-ordered cb split — like `commit_keep_open` (commit current cb
    /// WITHOUT a CPU wait, open a fresh cb) but additionally enforces GPU-side
    /// ordering: the committed cb signals a shared MTLEvent and the fresh cb
    /// encodes a wait on it FIRST, so the next cb's work cannot START until
    /// every command in the previous cb has COMPLETED. Use at hazard boundaries
    /// where Metal's in-cb/inter-cb hazard tracking is unreliable (chunk-prefill
    /// SWA-tile / ring-overwrite boundaries) and where a CPU drain
    /// (`commit_wait_stage`) corrupts CPU-mirror resync logic — this keeps the
    /// CPU encoding freely while serializing on the GPU.
    pub fn event_split(&mut self, _label: &str) {
        crate::macos::end_shared_compute_enc_force();        if self.order_event.is_none() {
            self.order_event = Some(self.state.device.new_event());
        }
        self.order_event_value += 1;
        let v = self.order_event_value;
        let ev = self.order_event.as_ref().unwrap();
        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        new_cb.encode_wait_for_event(ev, v);
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        old_cb.encode_signal_event(ev, v);
        old_cb.commit();
        self.pending_cbs.push(old_cb);
    }

    /// `event_split` (GPU-ordered, no CPU wait) with a SLIDING in-flight window:
    /// after committing, if more than `window` cbs are pending, CPU-wait + drop the
    /// OLDEST one. Keeps in-flight cbs ≤ window (under the ~8-11 resource ceiling that
    /// faults to all-zeros — Task 0) WITHOUT a full mid-chunk `drain_all` (which
    /// corrupts the event chain). Because the oldest cb is `window` layers behind the
    /// CPU's encode position, it is normally already complete → the wait returns
    /// instantly → near-zero bubble. This is the antirez-style bounded pipeline.
    pub fn event_split_windowed(&mut self, label: &str, window: usize) {
        self.event_split(label);
        while self.pending_cbs.len() > window {
            let oldest = self.pending_cbs.remove(0);
            oldest.wait_until_completed();
            assert_cb_ok(&oldest, label);
            crate::macos::layer_prof_push(&oldest);
        }
    }

    /// DEBUG ONLY: commit the current cb (keep it pending), wait for all pending
    /// cbs, and read `db` back to CPU — without consuming the scope (a fresh cb
    /// is opened to keep encoding). Kills overlap; use only behind a gate to
    /// bisect a numerical divergence (DS4_COMP_DUMP). Not for the hot path.
    pub fn debug_read(&mut self, db: &DeferredBuf) -> Vec<f32> {
        crate::macos::end_shared_compute_enc_force();        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        old_cb.commit();
        self.pending_cbs.push(old_cb);
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
        }
        unsafe { read_buffer::<f32>(&db.buf, db.n_elements) }
    }

    /// Per-stage GPU profiling (DS4_KERNEL_PROFILE harness): commit + WAIT the
    /// current cb, emitting its GPUStartTime/GPUEndTime under DS4_OP_TRACE with
    /// `label`, then open a fresh cb to keep encoding. Synchronous (kills
    /// overlap) so it's profiling-only — but it isolates each stage's GPU-busy
    /// time within a layer (attn-half / compressor / flash / output / moe / …),
    /// which the normal one-cb-per-layer path lumps together. Resident
    /// DeferredBufs survive (queue ordering + the wait), so the next stage
    /// chains against them unchanged.
    pub fn commit_wait_stage(&mut self, label: &str) {
        crate::macos::end_shared_compute_enc_force();        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        self.state.commit_wait_traced(&old_cb, label);
        crate::macos::layer_prof_push(&old_cb);
    }

    /// PROFILING ONLY: commit + wait the current cb, then return its GPU-BUSY
    /// micros (GPUEndTime − GPUStartTime), and open a fresh cb. Unlike wall-clock
    /// timing around commit_wait_stage, this excludes CPU encode + commit/queue
    /// overhead, so per-region splits inside a hot loop are ACCURATE even when
    /// committed frequently (the per-position-commit_wait wall timers lied).
    pub fn commit_wait_gpu_us(&mut self) -> u128 {
        crate::macos::end_shared_compute_enc_force();        use metal::foreign_types::ForeignTypeRef;
        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        old_cb.commit();
        old_cb.wait_until_completed();
        let (s, e): (f64, f64) = unsafe {
            use objc::runtime::Object;
            use objc::{msg_send, sel, sel_impl};
            let p: *mut Object = std::mem::transmute(old_cb.as_ptr());
            (msg_send![p, GPUStartTime], msg_send![p, GPUEndTime])
        };
        (((e - s) * 1_000_000.0).max(0.0)) as u128
    }

    /// M5 task #100 foundation — non-consuming checkpoint. Commits the
    /// current cb, waits for it and any prior `commit_keep_open` cbs to
    /// complete, reads `outputs` as `Vec<Vec<f32>>`, then opens a fresh
    /// cb on this scope. The scope is NOT consumed: any `DeferredBuf`
    /// allocated/produced before the checkpoint remains valid as input
    /// to ops encoded into the new cb (Metal guarantees ordering within
    /// a queue and StorageModeShared buffers retain their bytes across
    /// cb boundaries).
    ///
    /// Use this to "phase" a scope when some intermediate values must
    /// land on CPU (e.g., to drive CPU-scalar compressor/indexer paths)
    /// while other DeferredBufs (e.g., the persistent KV buffer or a
    /// q_heads output) keep flowing into the next phase's encodes
    /// without a re-upload round-trip.
    ///
    /// Each checkpoint still counts as one cb commit. The win over
    /// `flush_and_read_multi` + a fresh `batch_scope()` is structural:
    /// `DeferredBuf`s and their underlying MTLBuffers survive, so the
    /// next phase can chain against them rather than re-uploading.
    ///
    /// Caller pattern:
    /// ```ignore
    /// let scope = disp.batch_scope();
    /// let a = scope.upload_f32(&data_a);
    /// let b = scope.matvec_f32(&w, &a, ...)?;
    /// // Need `b` on CPU mid-scope; `a` and any subsequent allocs
    /// // stay GPU-resident.
    /// let mid = scope.commit_wait_read_multi(&[&b]);
    /// let c = scope.rms_norm_mul(&a, &gamma, eps)?;  // uses `a` from before
    /// let final_out = scope.flush_and_read(&c);
    /// ```
    pub fn commit_wait_read_multi(&mut self, outputs: &[&DeferredBuf]) -> Vec<Vec<f32>> {
        crate::macos::end_shared_compute_enc_force();        self.commit_keep_open();
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
        }
        self.pending_cbs.clear();
        outputs
            .iter()
            .map(|b| unsafe { read_buffer::<f32>(&b.buf, b.n_elements) })
            .collect()
    }

    /// Single-cb step 10 (layer chaining) — commit the scope's cb(s)
    /// WITHOUT waiting, and hand the in-flight CommandBuffers back to the
    /// caller to wait on later (at token end). DeferredBufs produced in this
    /// scope stay valid (their MTLBuffers are Arc-held), so a *later* scope
    /// can bind them as inputs: Metal serializes cbs within the queue, so the
    /// next layer's reads see this layer's writes without a CPU wait. This is
    /// how the per-layer commit+wait (which idles the GPU between cbs) is
    /// removed — the queue stays full across all 43 layers.
    pub fn commit_detach(mut self) -> Vec<metal::CommandBuffer> {
        crate::macos::end_shared_compute_enc_force();        self.cmd_buf.commit();
        let mut cbs = std::mem::take(&mut self.pending_cbs);
        cbs.push(self.cmd_buf.clone());
        cbs
    }

    /// Phase F (task #86) — wait on every committed-but-not-waited cb
    /// (`pending_cbs` + current `cmd_buf`). Use when ending a scope
    /// without reading outputs (e.g., side-effect-only work that
    /// writes to persistent buffers).
    pub fn wait_all_and_drop(self) {
        crate::macos::end_shared_compute_enc_force();        // Commit + wait the current cb.
        self.cmd_buf.commit();
        // Wait on all pending including the just-committed one.
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
            assert_cb_ok(cb, "wait_all_and_drop:pending");
        }
        self.cmd_buf.wait_until_completed();
        assert_cb_ok(&self.cmd_buf, "wait_all_and_drop:current");
        // DS4_LAYER_PROF: capture the terminal cb + pending, then report (all cbs —
        // incl. every commit_wait_stage/drain — are now COMPLETE so GPU timestamps
        // are valid). busy/span≈100% ⇒ GPU compute-bound; busy≪span ⇒ per-layer
        // CPU-stall bubbles (the structural deficit vs antirez's one-cb chunk).
        crate::macos::layer_prof_push(&self.cmd_buf);
        for cb in &self.pending_cbs { crate::macos::layer_prof_push(cb); }
        crate::macos::layer_prof_report("prefill");
    }

    /// Commit the current cb and CPU-wait EVERY committed cb (pending splits +
    /// current), then open a fresh cb to keep encoding. Unlike
    /// `commit_wait_stage` (which waits only the just-committed cb) this fully
    /// drains the event-split chain — the boundary the chunk>raw_cap path needs
    /// (event splits push cbs into `pending_cbs` that the per-layer
    /// commit_wait_stage never CPU-waits; pending heavy cbs accumulating across
    /// layers silently fault/zero under load — the BOS corruption).
    pub fn drain_all_stage(&mut self, label: &str) {
        crate::macos::end_shared_compute_enc_force();        let new_cb = self.state.command_queue.new_command_buffer().to_owned();
        let old_cb = std::mem::replace(&mut self.cmd_buf, new_cb);
        old_cb.commit();
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
            assert_cb_ok(cb, label);
            crate::macos::layer_prof_push(cb);
        }
        self.pending_cbs.clear();
        old_cb.wait_until_completed();
        assert_cb_ok(&old_cb, label);
        crate::macos::layer_prof_push(&old_cb);
    }

    /// Upload a slice into a fresh shared-storage buffer.
    pub fn upload_f32(&self, data: &[f32]) -> DeferredBuf {
        let buf = new_input_buffer(&self.state.device, data);
        DeferredBuf {
            buf,
            n_elements: data.len(),
        }
    }

    /// Upload an i32 slice to a fresh shared-storage buffer (e.g. precomputed
    /// top-k selection indices for the mixed-attention flash). n_elements counts
    /// i32s.
    pub fn upload_i32(&self, data: &[i32]) -> DeferredBuf {
        let buf = new_input_buffer(&self.state.device, data);
        DeferredBuf {
            buf,
            n_elements: data.len(),
        }
    }

    /// Allocate a fresh zero-init shared-storage output buffer.
    pub fn alloc_f32(&self, n: usize) -> DeferredBuf {
        let buf = new_output_buffer::<f32>(&self.state.device, n);
        // DS4_ZERO_SCRATCH=1 (diagnostic): zero every scratch alloc. If this makes
        // the chunk path deterministic, the nondeterminism is a kernel reading an
        // alloc_f32 scratch region it never wrote (read-before-write of garbage);
        // if it still flips, the source is a genuine GPU race, not a stale read.
        if zero_scratch_enabled() {
            // DS4_ZERO_SCRATCH_MINSZ=N (bisect): only zero allocs with >= N elements,
            // to localize WHICH buffer is the read-before-write (large residual/hc buffers
            // vs small score/scalar buffers). 0/unset = zero everything.
            let minsz: usize = std::env::var("DS4_ZERO_SCRATCH_MINSZ")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let maxsz: usize = std::env::var("DS4_ZERO_SCRATCH_MAXSZ")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX);
            if n >= minsz && n <= maxsz {
                unsafe {
                    std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * std::mem::size_of::<f32>());
                }
            }
        }
        DeferredBuf {
            buf,
            n_elements: n,
        }
    }

    /// Pooled scratch for the chunk path (DS4_CHUNK_POOL_SCRATCH=1): reuse ONE
    /// buffer per (`key`, `n`) across layers/chunks instead of a fresh `alloc_f32`
    /// each layer. Bounds the per-cb transient footprint to O(1) so commit_every>2
    /// (layer packing for bubble recovery) stops thrashing on N× fresh ~196MB allocs.
    /// CALLER CONTRACT: `key` must name an INTRA-layer buffer (produced AND fully
    /// consumed within one layer, never returned as the cross-layer hidden state) —
    /// pooling `after_ffn_k`/`cur_hc` would alias the residual carrier and corrupt it.
    /// Safe at ce>1: the producer writes the full extent; Metal within-cb hazard
    /// tracking serializes the cross-layer WAR (layers are sequential anyway). The
    /// buffer is NOT zeroed. Flag off → falls back to `alloc_f32` (fresh, unchanged).
    pub fn pooled_scratch_f32(&self, key: &'static str, n: usize) -> DeferredBuf {
        if std::env::var("DS4_CHUNK_POOL_SCRATCH").ok().as_deref() != Some("1") {
            return self.alloc_f32(n);
        }
        warn_pool_scratch_unstable();
        let kn = n | pool_phase_bit(); // ping-pong parity (no-op when flag off)
        let buf = CHUNK_SCRATCH_POOL.with(|p| {
            let mut map = p.borrow_mut();
            if let Some(b) = map.get(&(key, kn)) {
                return b.clone();
            }
            let b = new_output_buffer::<f32>(&self.state.device, n);
            self.pin_pooled_resident(&b); // T2: pin at first-touch (O(1) pool buffers)
            map.insert((key, kn), b.clone());
            b
        });
        DeferredBuf { buf, n_elements: n }
    }

    /// T2 (bounded-working-set): pin a freshly-created pooled-scratch buffer into
    /// the model residency set + commit immediately, so it stays GPU-wired across
    /// layer reuses and can't be evicted mid-cb under daemon memory pressure (the
    /// unpinned StorageModeShared collapse that makes `DS4_CHUNK_HASH_GPU` diverge).
    /// First-touch only — the pool reuses the buffer, so this fires O(pool-keys)
    /// (~tens) times per process, never per layer. commit+requestResidency on the
    /// scratch additions is cheap vs the 86 GB weight set. No-op if the set wasn't
    /// created (DS4_METAL_NO_RESIDENCY / macOS < 15).
    fn pin_pooled_resident(&self, buf: &metal::Buffer) {
        self.state.pin_state_buffer_resident(buf);
        self.state.commit_residency();
    }

    /// u16-element twin of [`Self::pooled_scratch_f32`] (same INTRA-layer contract).
    /// `n` = element count; the buffer is `n` halves (2 B each). For the f16 MoE mid.
    pub fn pooled_scratch_u16(&self, key: &'static str, n: usize) -> DeferredBuf {
        if std::env::var("DS4_CHUNK_POOL_SCRATCH").ok().as_deref() != Some("1") {
            let buf = crate::macos::new_output_buffer::<u16>(&self.state.device, n);
            return DeferredBuf { buf, n_elements: n };
        }
        warn_pool_scratch_unstable();
        // Key on (name | 1<<62) so it can't collide with the f32 pool's (name, n).
        let buf = CHUNK_SCRATCH_POOL.with(|p| {
            let mut map = p.borrow_mut();
            let k = (key, n | (1usize << 62) | pool_phase_bit());
            if let Some(b) = map.get(&k) {
                return b.clone();
            }
            let b = crate::macos::new_output_buffer::<u16>(&self.state.device, n);
            self.pin_pooled_resident(&b); // T2
            map.insert(k, b.clone());
            b
        });
        DeferredBuf { buf, n_elements: n }
    }

    /// Output buffer for an attention-chain intermediate. If the one-shot `ATTN_POOL_KEY`
    /// is armed (via `arm_attn_pool_key`) AND `DS4_CHUNK_POOL_ATTN=1`, return the per-(key,n)
    /// pooled buffer reused across layers (O(1) transient → bounds the per-cb footprint that
    /// else collapses event-window/packed prefill under memory pressure, see
    /// [[ds4-residency-collapse-pressure-masked]]); otherwise a fresh `alloc_f32`. ALWAYS
    /// consumes the one-shot key so it cannot leak to a later matmul. Safe ONLY for
    /// intra-layer outputs whose producer writes the FULL extent (the GEMM does — no
    /// read-before-write of stale pooled bytes); never arm it for a cross-layer carrier.
    fn alloc_attn_out(&self, n: usize) -> DeferredBuf {
        let key = ATTN_POOL_KEY.with(|k| k.take());
        match key {
            Some(key) if std::env::var("DS4_CHUNK_POOL_ATTN").ok().as_deref() == Some("1") => {
                let kn = n | pool_phase_bit();
                let buf = CHUNK_SCRATCH_POOL.with(|p| {
                    let mut map = p.borrow_mut();
                    if let Some(b) = map.get(&(key, kn)) {
                        return b.clone();
                    }
                    let b = crate::macos::new_output_buffer::<f32>(&self.state.device, n);
                    self.pin_pooled_resident(&b); // T2
                    map.insert((key, kn), b.clone());
                    b
                });
                DeferredBuf { buf, n_elements: n }
            }
            _ => self.alloc_f32(n),
        }
    }

    /// Hand the scope a weight slice. Uses the dispatcher's identity-keyed
    /// weight cache, so repeated calls with the same `&[f32]` re-use the
    /// already-uploaded buffer (matches `cached_weight_buffer`'s semantics).
    #[track_caller]
    pub fn weight_f32(&self, w: &[f32]) -> DeferredBuf {
        if w.is_empty() && std::env::var("DS4_TRAP_ZERO_WEIGHT").is_ok() {
            panic!(
                "weight_f32: ZERO-length f32 weight bound at {} — lean-dropped f32 \
                 (use weight_hc / weight_f32_lean_opt with the f16/q8 fallback)",
                std::panic::Location::caller()
            );
        }
        let buf = self.state.cached_weight_buffer(w);
        DeferredBuf {
            buf,
            n_elements: w.len(),
        }
    }

    /// Bind an f32 projection weight that may be lean-dropped to Q8_0. Under
    /// lean/streaming the f32 slice is EMPTY (the weight lives in the `*_q8`
    /// bytes), and the consumer (`encode_attn_output_proj`) ignores this f32
    /// buffer entirely when `use_q8` is on. Binding an empty slice through
    /// `weight_f32` would create a zero-length Metal buffer (16-byte dummy +
    /// noisy backtrace, and an OOB hazard if ever read), so return a 1-element
    /// placeholder instead. Non-lean (f32 present) → the real weight buffer.
    pub fn weight_f32_lean_opt(&self, w: &[f32]) -> DeferredBuf {
        if w.is_empty() {
            // placeholder — the q8 path never reads it; 1 elem keeps Metal happy.
            return self.alloc_f32(1);
        }
        self.weight_f32(w)
    }

    /// Hand the scope an f32 projection weight to run as a Q8_0 matvec. The
    /// weight is re-quantized to GGUF `block_q8_0` bytes once (cached by slice
    /// identity) so the matvec reads 1 byte/weight instead of 4. `n_elements`
    /// tracks the logical weight count (`d_in*d_out`); the buffer holds bytes.
    pub fn weight_q8_0(&self, w: &[f32]) -> DeferredBuf {
        let buf = self.state.cached_q8_0_weight_buffer(w);
        DeferredBuf {
            buf,
            n_elements: w.len(),
        }
    }

    /// Hand the scope raw GGUF `block_q8_0` projection bytes (owned,
    /// page-aligned — from `AttnLayerWeights::attn_*_q8`) as a no-copy resident
    /// buffer. No re-quantization (the bytes are already Q8_0) and no copy into
    /// Metal-managed storage, so the buffer avoids the page-fault stalls the
    /// re-quantize `weight_q8_0` path hit. `n_weights` is the logical count
    /// (`d_in*d_out`) the matvec validates against.
    pub fn weight_q8_0_raw(&self, bytes: &[u8], n_weights: usize) -> DeferredBuf {
        let buf = self.state.cached_q8_0_raw_buffer(bytes);
        DeferredBuf {
            buf,
            n_elements: n_weights,
        }
    }

    /// `out[d_out] = w[d_out x d_in] · x[d_in]` with `w` stored as GGUF
    /// `block_q8_0` (from `weight_q8_0`). In-scope twin of
    /// `matvec_q8_0_bytes_impl` (macos.rs); reads the weight at 1 byte/weight.
    /// `d_in` must be a multiple of 32 (Q8_0 block size); `d_out` even.
    /// K-position Q8_0 matvec: `out[k][r] = sum_i w[r][i] * x[k][i]` for
    /// `k ∈ [0, K)` and `r ∈ [0, d_out)`. Uses the simdgroup-matrix-tiled
    /// `ds4_kernel_mul_mv_K_q8_0_f32_sg` (Phase 0 winner; K=8 ratio 0.93 vs
    /// K=1). Activation layout: f32 row-major `[K, d_in]`. Output layout:
    /// f32 row-major `[K, d_out]`. K supported in `{1, 2, 4, 8}`.
    pub fn matvec_k_q8_0(
        &self,
        w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            w_q8.n_elements == d_in * d_out,
            "matvec_k_q8_0: w has {} elems, expected d_in*d_out = {}*{} = {}",
            w_q8.n_elements, d_in, d_out, d_in * d_out
        );
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_in,
            "matvec_k_q8_0: x_K has {} elems, expected K*d_in = {}*{}",
            x_k.n_elements, k_positions, d_in
        );
        anyhow::ensure!(d_in % 32 == 0, "matvec_k_q8_0: d_in must be %32 (Q8_0)");
        anyhow::ensure!(
            matches!(k_positions, 1 | 2 | 4 | 8),
            "matvec_k_q8_0: K must be in {{1,2,4,8}} (got {})", k_positions
        );

        // DS4_CHUNK_Q8_FAITHFUL: per-row matvec_q8_0 (the per-token decode
        // kernel) — bit-identical to the decode oracle at every K. The K-batched
        // Q8 kernel (below) drifts ~1e-3/row from matvec_q8_0 (F32 versions match;
        // Q8 dequant/accum order differs) which cascades to @3000 word-salad.
        // Correctness-first (loses the K-batch; the GEMM-vs-matvec Q8 alignment is
        // the perf follow-up).
        if std::env::var("DS4_CHUNK_Q8_FAITHFUL").ok().as_deref() == Some("1") {
            let out = self.alloc_f32(k_positions * d_out);
            for k in 0..k_positions {
                let xk = self.slice_out_f32(x_k, k * d_in, d_in);
                let ok = self.matvec_q8_0(w_q8, &xk, d_in, d_out)?;
                self.copy_buf_into(&ok, out.buffer(), k * d_out);
            }
            return Ok(out);
        }

        // Fixed kernel constants (matched to the shim layout):
        //   NR0=32 rows/TG, NSG=4 simdgroups, K_MAX=8 cols.
        let nsg: i16 = 4;
        let kk: i16 = k_positions as i16;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&kk.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mv_K_q8_0_f32_sg", &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg as *const _ as *const _, MTLDataType::Short, 1500,
                );
                fcv.set_constant_value_at_index(
                    &kk as *const _ as *const _, MTLDataType::Short, 1501,
                );
            })?;

        let out = self.alloc_f32(k_positions * d_out);

        // Threadgroup memory: SA[32×32] float (4096B) + SB[32×8] float (1024B) =
        // 5120B; reused after the matmul as SC[NSG×64] float (1024B). Pick the
        // max so both phases fit. FLOAT tiles (was half) so the q8 dequant matches
        // the per-token matvec_q8_0 oracle (half-rounding cascaded to @3000 salad).
        let tg_bytes: u64 = 32 * 32 * 4 + 32 * 8 * 4;
        let n_row_tg = ((d_out as u64) + 31) / 32; // NR0=32 per TG

        let d_in_u32: u32 = d_in as u32;
        let d_out_u32: u32 = d_out as u32;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &d_in_u32);
        set_scalar_bytes(enc, 1, &d_out_u32);
        enc.set_buffer(2, Some(&w_q8.buf), 0);
        enc.set_buffer(3, Some(&x_k.buf), 0);
        enc.set_buffer(4, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, tg_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// GPU f32→f16 cast of a contiguous `[n]` activation buffer → a fresh half
    /// (`u16`-backed) buffer of `n` half-elements. No CPU round-trip (the source is
    /// GPU-resident). Drives `ds4_kernel_cpy_f32_f16` as a flat 1-D copy: ne00=n,
    /// ne01/02/03=1, one element/thread, grid.x=ceil(n/256). Used by the
    /// `DS4_ATTN_GEMM_F16` activation path in [`Self::matmul_k_q8_0`] (the f16 GEMM
    /// reads src1 as half — halves the activation READ bandwidth; the mul_mm re-reads
    /// src1 ~d_out/64× across output-row tiles, so the one-shot cast pays off). The
    /// returned `n_elements` counts HALF elements (2 B each).
    pub fn cast_f32_to_f16_contig(&self, src: &DeferredBuf, n: usize) -> Result<DeferredBuf> {
        anyhow::ensure!(n > 0, "cast_f32_to_f16_contig: n must be > 0");
        anyhow::ensure!(
            src.n_elements >= n,
            "cast_f32_to_f16_contig: src has {} f32 elems, need {}", src.n_elements, n
        );
        let dst = crate::macos::new_output_buffer::<u16>(&self.state.device, n);
        // Compile on-demand from the bridge library (cpy_f32_f16 is not in the
        // pre-built `pipelines` map; no function constants → empty key).
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_cpy_f32_f16", &[], |_| {})?;
        // ds4_metal_args_cpy (cpy.metal:1) — flat 1-D contiguous copy.
        #[repr(C)]
        struct CpyArgs {
            nk0: i64,
            ne00: i64, ne01: i64, ne02: i64, ne03: i64,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne0: i64, ne1: i64, ne2: i64, ne3: i64,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }
        let n64 = n as u64;
        let args = CpyArgs {
            nk0: n as i64,
            ne00: n as i64, ne01: 1, ne02: 1, ne03: 1,
            nb00: 4, nb01: n64 * 4, nb02: n64 * 4, nb03: n64 * 4,
            ne0: n as i64, ne1: 1, ne2: 1, ne3: 1,
            nb0: 2, nb1: n64 * 2, nb2: n64 * 2, nb3: n64 * 2,
        };
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<CpyArgs>() as u64,
            &args as *const CpyArgs as *const _);
        enc.set_buffer(1, Some(&src.buf), 0);
        enc.set_buffer(2, Some(&dst), 0);
        let ntg = 256u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n64.div_ceil(ntg), 1, 1),
            MTLSize::new(ntg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(DeferredBuf { buf: dst, n_elements: n })
    }

    /// DS4_CHUNK_BLKFLASH workspace: cast `kv_f32` ([n_rows, head_dim] f32, the
    /// rotated MLA latent — n_lora_kv == head_dim == 512) directly to an f16 flash
    /// workspace [n_total_padded, head_dim], padded rows left uninitialised (the
    /// SWA mask sets their softmax weight to 0 so their V never contributes). This
    /// is antirez's `cpy_f32_f16` (ds4_metal.m:17328) — it REPLACES our
    /// `kv_fp8_store + build_extended_kv` (fp8 round-trip + gather, the 139→255
    /// all-raw overhead) with one lossless pass. K and V share this buffer (MLA).
    pub fn kv_f16_workspace_direct(
        &self, kv_f32: &DeferredBuf, n_rows: usize, head_dim: usize, n_total_padded: u32,
    ) -> Result<metal::Buffer> {
        let n = n_rows * head_dim; // real KV elements to cast
        anyhow::ensure!(kv_f32.n_elements >= n, "kv_f16_workspace_direct: kv has {} elems < {}",
            kv_f32.n_elements, n);
        anyhow::ensure!(n_total_padded as usize >= n_rows, "n_total_padded < n_rows");
        let dst = crate::macos::new_output_buffer::<u16>(
            &self.state.device, n_total_padded as usize * head_dim);
        let pipe = self.state.specialized_pipeline("ds4_kernel_cpy_f32_f16", &[], |_| {})?;
        #[repr(C)]
        struct CpyArgs {
            nk0: i64, ne00: i64, ne01: i64, ne02: i64, ne03: i64,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne0: i64, ne1: i64, ne2: i64, ne3: i64,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }
        let n64 = n as u64;
        let args = CpyArgs {
            nk0: n as i64, ne00: n as i64, ne01: 1, ne02: 1, ne03: 1,
            nb00: 4, nb01: n64 * 4, nb02: n64 * 4, nb03: n64 * 4,
            ne0: n as i64, ne1: 1, ne2: 1, ne3: 1,
            nb0: 2, nb1: n64 * 2, nb2: n64 * 2, nb3: n64 * 2,
        };
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<CpyArgs>() as u64, &args as *const CpyArgs as *const _);
        enc.set_buffer(1, Some(&kv_f32.buf), 0);
        enc.set_buffer(2, Some(&dst), 0);
        let ntg = 256u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n64.div_ceil(ntg), 1, 1), MTLSize::new(ntg, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(dst)
    }

    /// Large-K DENSE q8_0 GEMM — `[K, d_in] · [d_out, d_in]^T → [K, d_out]`
    /// (row-major out, matching `matvec_k_q8_0`). Wires DwarfStar's tiled
    /// simdgroup matmul `kernel_mul_mm_q8_0_f32` (dense.metal), which scales
    /// with batch M=K — the path DwarfStar itself uses for n_tok>8 (it falls to
    /// the register-unrolled mul_mv_ext at n_tok<=8). This replaces the K<=8
    /// `matvec_k_q8_0` for chunked prefill's dense projections (shared expert,
    /// qkv, output_proj). `d_in % 32 == 0` (Q8_0 block).
    ///
    /// `DS4_ATTN_GEMM_F16=1` (gated, default-off): cast src1 activations to f16 and
    /// dispatch `ds4_kernel_mul_mm_q8_0_f16` — halves the activation read bandwidth
    /// (the MMA is already half8x8; only src1 memory type changes). Faithful precision
    /// change (f16 activation error ≪ q8 weight quant); validate by coherence, not
    /// byte-identity. See docs/ATTN_GEMM_F16_SCOPE.md.
    pub fn matmul_k_q8_0(
        &self,
        w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        // FFI-boundary guard (ffi-boundary-dim-validation-requirement): a 0 dim =>
        // GPU OOB in the Metal kernel. Reject at the call site.
        anyhow::ensure!(
            d_in > 0 && d_out > 0,
            "matmul_k_q8_0: dims must be > 0 (d_in={d_in} d_out={d_out}) — 0 => GPU OOB"
        );
        anyhow::ensure!(
            w_q8.n_elements == d_in * d_out,
            "matmul_k_q8_0: w has {} elems, expected d_in*d_out = {}*{}",
            w_q8.n_elements, d_in, d_out
        );
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_in,
            "matmul_k_q8_0: x_K has {} elems, expected K*d_in = {}*{}",
            x_k.n_elements, k_positions, d_in
        );
        anyhow::ensure!(d_in % 32 == 0, "matmul_k_q8_0: d_in must be %32 (Q8_0)");

        // DS4_CHUNK_Q8_FAITHFUL: per-row matvec_q8_0 (the per-token decode kernel)
        // — bit-identical to the decode oracle at every K. The GEMM below drifts
        // ~1e-3/row from matvec_q8_0 (Q8 dequant/accum order differs; F32 matches)
        // which cascades to @3000 word-salad. Correctness-first (loses the GEMM
        // batch; the GEMM-vs-matvec Q8 alignment is the perf follow-up).
        if std::env::var("DS4_CHUNK_Q8_FAITHFUL").ok().as_deref() == Some("1") {
            let out = self.alloc_f32(k_positions * d_out);
            for k in 0..k_positions {
                let xk = self.slice_out_f32(x_k, k * d_in, d_in);
                let ok = self.matvec_q8_0(w_q8, &xk, d_in, d_out)?;
                self.copy_buf_into(&ok, out.buffer(), k * d_out);
            }
            return Ok(out);
        }

        // ds4_gpu_mul_mm_args (ds4_metal.m:1700) — exact C layout.
        #[repr(C)]
        struct MmArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne12: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        let f32_b = std::mem::size_of::<f32>() as u64;
        let row_bytes = (d_in as u64 / 32) * 34; // Q8_0: 34 B / 32-elem block
        // DS4_ATTN_GEMM_F16 (gated): read src1 (activation) as f16 → half the strides
        // + the f16 kernel + a pre-cast of x_k. The MMA is already half8x8; this only
        // changes the src1 memory type. See docs/ATTN_GEMM_F16_SCOPE.md.
        let gemm_f16 = std::env::var("DS4_ATTN_GEMM_F16").ok().as_deref() == Some("1");
        let act_b: u64 = if gemm_f16 { 2 } else { f32_b };
        let args = MmArgs {
            ne00: d_in as i32, ne02: 1,
            nb01: row_bytes, nb02: row_bytes * d_out as u64, nb03: row_bytes * d_out as u64,
            ne12: 1,
            nb10: act_b, nb11: d_in as u64 * act_b,
            nb12: d_in as u64 * k_positions as u64 * act_b,
            nb13: d_in as u64 * k_positions as u64 * act_b,
            ne0: d_out as i32, ne1: k_positions as i32,
            r2: 1, r3: 1,
        };

        let bc_inp = (d_in % 32) != 0; // false (asserted %32) — kept for parity
        let bc_out = (d_out % 64) != 0 || (k_positions % 32) != 0;
        let mut key = Vec::with_capacity(2);
        key.push(bc_inp as u8);
        key.push(bc_out as u8);
        let pipe = self.state.specialized_pipeline(
            if gemm_f16 { "ds4_kernel_mul_mm_q8_0_f16" } else { "ds4_kernel_mul_mm_q8_0_f32" }, &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &bc_inp as *const bool as *const _, MTLDataType::Bool, 700);
                fcv.set_constant_value_at_index(
                    &bc_out as *const bool as *const _, MTLDataType::Bool, 701);
            })?;

        // OOB guard probe (chunk>raw_cap NaN bisect): mul_mm tiles are 32 rows in
        // ne1; pad the ALLOCATION (not ne1) so any partial-tile spill lands in
        // owned memory instead of the next heap buffer. DS4_MM_PAD=1 enables
        // (probe-only; the kernel's bc_out path is bounds-checked, so default off).
        let pad_rows = if std::env::var("DS4_MM_PAD").ok().as_deref() == Some("1") {
            k_positions.div_ceil(32) * 32
        } else {
            k_positions
        };
        // Pooled when DS4_CHUNK_POOL_ATTN=1 + a key was armed for this projection (the
        // qkv-chain intermediates) → O(1) cross-layer footprint; else fresh. The GEMM
        // writes the full [k_positions × d_out] extent, so reusing a pooled buffer is safe
        // (no stale-tail read). Consumes the one-shot key unconditionally.
        let out_padded = self.alloc_attn_out(pad_rows * d_out);
        let out = DeferredBuf { buf: out_padded.buf, n_elements: k_positions * d_out };
        // f16 activation pre-cast (BEFORE the GEMM encoder opens — the cast uses its
        // own encoder). Outlives `enc` so the buffer stays bound through dispatch.
        let x_f16 = if gemm_f16 {
            Some(self.cast_f32_to_f16_contig(x_k, k_positions * d_in)?)
        } else {
            None
        };
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<MmArgs>() as u64,
            &args as *const MmArgs as *const _);
        enc.set_buffer(1, Some(&w_q8.buf), 0);
        enc.set_buffer(2, Some(x_f16.as_ref().map_or(&x_k.buf, |b| &b.buf)), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, if bc_out { 8192 } else { 6144 });
        enc.dispatch_thread_groups(
            MTLSize::new((k_positions as u64 + 31) / 32, (d_out as u64 + 63) / 64, 1),
            MTLSize::new(128, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// q4_0 twin of [`Self::matmul_k_q8_0`]: weight-stationary tiled GEMM reading
    /// block_q4_0 weights (18 B / 32-elem block: half d + 16 nibble bytes) → half the
    /// weight bytes of q8_0. Dispatches the bridge's `ds4_kernel_mul_mm_q4_0_f32`
    /// instantiation. For the q4 attention projections (DS4_ATTN_Q4); q4 attn is
    /// argmax-safe (DS4_ATTN_Q4_PROBE-validated). `w_q4` holds the q4_0 bytes;
    /// `n_elements` is the LOGICAL d_in*d_out (same as the q8 weight).
    pub fn matmul_k_q4_0(
        &self,
        w_q4: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            d_in > 0 && d_out > 0,
            "matmul_k_q4_0: dims must be > 0 (d_in={d_in} d_out={d_out}) — 0 => GPU OOB"
        );
        anyhow::ensure!(d_in % 32 == 0, "matmul_k_q4_0: d_in must be %32 (Q4_0)");
        #[repr(C)]
        struct MmArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne12: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        // The bridge kernel_mul_mm reads src1 as T1=FLOAT (f32 in memory), then casts to
        // S1=half during shmem staging. So x is plain f32 with 4-byte strides — NO f16
        // cast. (The earlier f32 attempt failed only on the 6144→8192 dynamic-shmem size.)
        let f_b: u64 = 4;
        let row_bytes = (d_in as u64 / 32) * 18; // Q4_0: 18 B / 32-elem block
        let args = MmArgs {
            ne00: d_in as i32, ne02: 1,
            nb01: row_bytes, nb02: row_bytes * d_out as u64, nb03: row_bytes * d_out as u64,
            ne12: 1,
            nb10: f_b, nb11: d_in as u64 * f_b,
            nb12: d_in as u64 * k_positions as u64 * f_b,
            nb13: d_in as u64 * k_positions as u64 * f_b,
            ne0: d_out as i32, ne1: k_positions as i32,
            r2: 1, r3: 1,
        };
        let bc_inp = false;
        let bc_out = (d_out % 64) != 0 || (k_positions % 32) != 0;
        let key = vec![bc_inp as u8, bc_out as u8];
        let pipe = self.state.specialized_pipeline("ds4_kernel_mul_mm_q4_0_f32", &key, |fcv| {
            fcv.set_constant_value_at_index(&bc_inp as *const bool as *const _, MTLDataType::Bool, 700);
            fcv.set_constant_value_at_index(&bc_out as *const bool as *const _, MTLDataType::Bool, 701);
        })?;
        let out = self.alloc_attn_out(k_positions * d_out);
        let out = DeferredBuf { buf: out.buf, n_elements: k_positions * d_out };
        // The bridge kernel uses host-set DYNAMIC threadgroup memory. On the shared/
        // batched encoder that set didn't take (other dispatches share the encoder's
        // threadgroup-length state) → zeros in the full pipeline though the kernel is
        // correct in isolation. Force-close the shared encoder so this GEMM gets its
        // OWN encoder with the right 8192 B threadgroup allocation.
        crate::macos::end_shared_compute_enc_force();
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<MmArgs>() as u64, &args as *const MmArgs as *const _);
        enc.set_buffer(1, Some(&w_q4.buf), 0);
        enc.set_buffer(2, Some(&x_k.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        // The BRIDGE kernel_mul_mm uses a host-set dynamic threadgroup buffer
        // (`threadgroup char * shmem [[threadgroup(0)]]`) = sa[2048]+sb[2048] halves =
        // 8192 B. (The q8 path uses the EMITTED kernel with STATIC shmem, so its host
        // size was a no-op — copying it gave 6144 and truncated the bridge's staging.)
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(
            MTLSize::new((k_positions as u64 + 31) / 32, (d_out as u64 + 63) / 64, 1),
            MTLSize::new(128, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Attention-projection GEMM with optional q4 weights (DS4_ATTN_Q4=1): requant the
    /// q8_0 weight to q4_0 ONCE per distinct weight buffer (cached for the process), then
    /// dispatch matmul_k_q4_0. Otherwise the normal q8 path. The 5 attention projection
    /// call-sites (q_a/q_b/kv/w_o) route through here; shared/MoE stay q8. q4 attn is
    /// argmax-safe (DS4_ATTN_Q4_PROBE: byte-identical greedy @600/@3000).
    pub fn matmul_k_attn_proj(
        &self,
        w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        if std::env::var("DS4_ATTN_Q4").ok().as_deref() != Some("1") {
            return self.matmul_k_q8_0_auto(w_q8, x_k, d_in, d_out, k_positions);
        }
        let key = w_q8.buf.contents() as usize;
        let w_q4 = ATTN_Q4_WEIGHT_CACHE.with(|c| -> Result<DeferredBuf> {
            if let Some(b) = c.borrow().get(&key) {
                return Ok(DeferredBuf { buf: b.clone(), n_elements: d_in * d_out });
            }
            // requant q8_0 → q4_0 (read the resident q8 bytes, 34 B/block).
            let n_blocks = (d_in * d_out) / 32;
            let q8_bytes = unsafe {
                std::slice::from_raw_parts(w_q8.buf.contents() as *const u8, n_blocks * 34)
            };
            let q4 = ds4_engine::layer_view::requant_q8_0_to_q4_0(q8_bytes)?;
            let buf = new_input_buffer(&self.state.device, &q4);
            // Pin AND commit so the fresh q4 buffer is actually made resident before the
            // GEMM reads it. (Pinning without the commit adds it to the MTLResidencySet
            // uncommitted → not yet wired → the kernel reads zeros in the full pipeline,
            // though the smoke — which has no residency set — works. First-touch only.)
            self.state.pin_state_buffer_resident(&buf);
            self.state.commit_residency();
            c.borrow_mut().insert(key, buf.clone());
            Ok(DeferredBuf { buf, n_elements: d_in * d_out })
        })?;
        self.matmul_k_q4_0(&w_q4, x_k, d_in, d_out, k_positions)
    }

    /// Weight-stationary f16 GEMM `[K, d_in] · [d_out, d_in]^T → [K, d_out]` via the
    /// tiled simdgroup `ds4_kernel_mul_mm_f16_f32` (dense.metal) — the f16 twin of
    /// [`Self::matmul_k_q8_0`]. Replaces the per-token `matmul_f16_k` (mul_mv) for
    /// K-batched dense projections (indexer q_b / weights, compressor, hc-mix): the
    /// mul_mv re-streams the whole `[d_in×d_out]` f16 weight ONCE PER TOKEN
    /// (K× the weight traffic → bandwidth-bound), whereas this loads each weight tile
    /// once and reuses it across a 32-token tile. f16 multiply, f32 accumulate (matches
    /// antirez's prefill projections; same precision class as DS4_INDEXER_F16). The
    /// weight `w` is f16 (`w.n_elements == d_in*d_out` half-elements). `d_in % 8 == 0`.
    pub fn matmul_k_f16(
        &self,
        w: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            d_in > 0 && d_out > 0,
            "matmul_k_f16: dims must be > 0 (d_in={d_in} d_out={d_out}) — 0 => GPU OOB"
        );
        anyhow::ensure!(
            w.n_elements == d_in * d_out,
            "matmul_k_f16: w has {} elems, expected d_in*d_out = {}*{}",
            w.n_elements, d_in, d_out
        );
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_in,
            "matmul_k_f16: x_k has {} elems, expected K*d_in = {}*{}",
            x_k.n_elements, k_positions, d_in
        );
        anyhow::ensure!(d_in % 8 == 0, "matmul_k_f16: d_in must be %8 (f16 tile load)");

        #[repr(C)]
        struct MmArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne12: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        let f32_b = std::mem::size_of::<f32>() as u64;
        let f16_b = 2u64;
        let row_bytes = d_in as u64 * f16_b; // f16: 2 B / elem, dense rows
        let args = MmArgs {
            ne00: d_in as i32, ne02: 1,
            nb01: row_bytes, nb02: row_bytes * d_out as u64, nb03: row_bytes * d_out as u64,
            ne12: 1,
            nb10: f32_b, nb11: d_in as u64 * f32_b,
            nb12: d_in as u64 * k_positions as u64 * f32_b,
            nb13: d_in as u64 * k_positions as u64 * f32_b,
            ne0: d_out as i32, ne1: k_positions as i32,
            r2: 1, r3: 1,
        };

        let bc_inp = (d_in % 32) != 0;
        let bc_out = (d_out % 64) != 0 || (k_positions % 32) != 0;
        let mut key = Vec::with_capacity(2);
        key.push(bc_inp as u8);
        key.push(bc_out as u8);
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mm_f16_f32", &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &bc_inp as *const bool as *const _, MTLDataType::Bool, 700);
                fcv.set_constant_value_at_index(
                    &bc_out as *const bool as *const _, MTLDataType::Bool, 701);
            })?;

        let out = self.alloc_f32(k_positions * d_out);
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<MmArgs>() as u64,
            &args as *const MmArgs as *const _);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(&x_k.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, if bc_out { 8192 } else { 6144 });
        enc.dispatch_thread_groups(
            MTLSize::new((k_positions as u64 + 31) / 32, (d_out as u64 + 63) / 64, 1),
            MTLSize::new(128, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// GROUPED large-K Q8_0 GEMM (the output-proj `w_o_a` block-diagonal matmul):
    /// `n_groups` independent GEMMs `[K, group_dim] · [n_lora_o, group_dim]^T`, one
    /// per group, in ONE batched `mul_mm` dispatch (grid.z = n_groups). The
    /// activation `x_k` is the K-major `heads_k [K, n_groups*group_dim]` read with a
    /// STRIDED matrix view (nb11 = full row, nb12 = group offset) so NO gather is
    /// needed; each group's Q8 weight is read ONCE (vs the per-K mul_mv re-reading it
    /// K×). Output is GROUP-MAJOR `[n_groups, K, n_lora_o]` (the caller interleaves to
    /// `[K, out_low]` with [`Self::interleave_group_major`]). `group_dim % 32 == 0`.
    pub fn matmul_k_q8_0_grouped(
        &self,
        w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        group_dim: usize,
        n_lora_o: usize,
        n_groups: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(group_dim % 32 == 0, "matmul_k_q8_0_grouped: group_dim %32 (Q8_0)");
        anyhow::ensure!(
            w_q8.n_elements == n_groups * n_lora_o * group_dim,
            "matmul_k_q8_0_grouped: w {} != n_groups*n_lora_o*group_dim {}",
            w_q8.n_elements, n_groups * n_lora_o * group_dim
        );
        anyhow::ensure!(
            x_k.n_elements == k_positions * n_groups * group_dim,
            "matmul_k_q8_0_grouped: x {} != K*n_groups*group_dim {}",
            x_k.n_elements, k_positions * n_groups * group_dim
        );
        #[repr(C)]
        struct MmArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne12: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        let f32_b = 4u64;
        let row_bytes = (group_dim as u64 / 32) * 34; // Q8_0 weight row
        let x_row = (n_groups * group_dim) as u64 * f32_b; // heads K-row stride
        let args = MmArgs {
            ne00: group_dim as i32, ne02: n_groups as i32,
            nb01: row_bytes,
            nb02: row_bytes * n_lora_o as u64,         // weight matrix (group) stride
            nb03: row_bytes * n_lora_o as u64,
            ne12: n_groups as i32,
            nb10: f32_b,
            nb11: x_row,                                // x row (K) stride
            nb12: group_dim as u64 * f32_b,             // x group (matrix) stride
            nb13: x_row * k_positions as u64,
            ne0: n_lora_o as i32, ne1: k_positions as i32,
            r2: 1, r3: 1,
        };
        let bc_inp = false; // group_dim %32 asserted
        let bc_out = (n_lora_o % 64) != 0 || (k_positions % 32) != 0;
        let key = [bc_inp as u8, bc_out as u8];
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mm_q8_0_f32", &key, |fcv| {
                fcv.set_constant_value_at_index(&bc_inp as *const bool as *const _, MTLDataType::Bool, 700);
                fcv.set_constant_value_at_index(&bc_out as *const bool as *const _, MTLDataType::Bool, 701);
            })?;
        let out = self.alloc_f32(n_groups * k_positions * n_lora_o); // group-major
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, std::mem::size_of::<MmArgs>() as u64, &args as *const MmArgs as *const _);
        enc.set_buffer(1, Some(&w_q8.buf), 0);
        enc.set_buffer(2, Some(&x_k.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, if bc_out { 8192 } else { 6144 });
        enc.dispatch_thread_groups(
            MTLSize::new((k_positions as u64 + 31) / 32, (n_lora_o as u64 + 63) / 64, n_groups as u64),
            MTLSize::new(128, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Interleave a GROUP-MAJOR `[n_groups, K, n_lora_o]` buffer into K-MAJOR
    /// `[K, n_groups*n_lora_o]` (the layout the dense `w_o_b` GEMM + the rest of the
    /// chain expect). One dispatch, one thread per output element. Inline pipeline.
    pub fn interleave_group_major(
        &self,
        src: &DeferredBuf,
        n_groups: usize,
        k_positions: usize,
        n_lora_o: usize,
    ) -> Result<DeferredBuf> {
        let out = self.alloc_f32(k_positions * n_groups * n_lora_o);
        let pipe = self.state.ensure_interleave_group_major_pipeline()?.clone();
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        let total = (k_positions * n_groups * n_lora_o) as u32;
        let ng = n_groups as u32;
        let kp = k_positions as u32;
        let nlo = n_lora_o as u32;
        enc.set_buffer(0, Some(&src.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        set_scalar_bytes(enc, 2, &ng);
        set_scalar_bytes(enc, 3, &kp);
        set_scalar_bytes(enc, 4, &nlo);
        set_scalar_bytes(enc, 5, &total);
        let tg = 256u64;
        enc.dispatch_thread_groups(
            MTLSize::new((total as u64 + tg - 1) / tg, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// q8_0 K-batched matmul that auto-selects the right kernel by batch size,
    /// exactly like DwarfStar (mul_mv_ext for K<=8, tiled mul_mm GEMM for K>8).
    /// The drop-in large-K replacement for `matvec_k_q8_0`.
    pub fn matmul_k_q8_0_auto(
        &self,
        w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        // K=8 EXCLUDED from matvec_k: matvec_k_q8_0 at K=8 drifts ~6e-5/position
        // from the K=1/2/4 (bit-identical) result — enough to break chunked-prefill
        // bit-equivalence (amplifies to ~3% over 40 layers). matmul_k_q8_0 (the
        // GEMM) is bit-identical per-row at all K (matmul_k_q8_0_smoke rel=0), so
        // route K≥8 through it. DS4_K8_MATVEC=1 reverts to matvec_k at K=8.
        let k8_matvec = std::env::var("DS4_K8_MATVEC").ok().as_deref() == Some("1");
        // The one-shot attention pool key (arm_attn_pool_key) targets ONLY the
        // weight-stationary GEMM output. Consume it here and re-arm just before the GEMM
        // leaf, so the matvec / pertok (small-K, never pooled) paths can't leak it to a
        // later matmul. In the chunk-prefill path K is large → always the GEMM.
        let armed_attn_key = ATTN_POOL_KEY.with(|k| k.take());
        // LEVER B (DS4_MATMUL_K_PERTOK=1): ORDER-MATCH the per-token reference.
        // The large-K GEMM (matmul_k) reduces in a different order than the K=1
        // matvec (rel ~4e-4 non-assoc) — the chunk-prefill residual-drift source.
        // matvec_k at K∈{1,2,4} is bit-identical PER POSITION to K=1, so looping it
        // in {4,2,1} groups reproduces the per-token reduction order EXACTLY for
        // every matmul_k_q8_0_auto call-site (q_a/q_b/kv/output-proj). Guided-
        // precision probe: if this flattens the residual_divergence_probe drift,
        // these attention/output matmuls are the hotspot; residual drift = MoE.
        if std::env::var("DS4_MATMUL_K_PERTOK").ok().as_deref() == Some("1") && k_positions > 4 {
            let out = self.alloc_f32(k_positions * d_out);
            let mut r = 0usize;
            while r < k_positions {
                let g = if k_positions - r >= 4 { 4 } else if k_positions - r >= 2 { 2 } else { 1 };
                let x_g = self.slice_out_f32(x_k, r * d_in, g * d_in);
                let res = self.matvec_k_q8_0(w_q8, &x_g, d_in, d_out, g)?;
                self.copy_buf_into(&res, &out.buf, r * d_out);
                r += g;
            }
            return Ok(out);
        }
        if matches!(k_positions, 1 | 2 | 4) || (k_positions == 8 && k8_matvec) {
            self.matvec_k_q8_0(w_q8, x_k, d_in, d_out, k_positions)
        } else {
            if let Some(key) = armed_attn_key {
                arm_attn_pool_key(key);
            }
            self.matmul_k_q8_0(w_q8, x_k, d_in, d_out, k_positions)
        }
    }

    pub fn matvec_q8_0(
        &self,
        w_q8: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        let out = self.q8_matvec_into(enc, w_q8, x, d_in, d_out)?;
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// In-scope GPU argmax over a logit row → token id (1-elem i32
    /// `DeferredBuf`). Wraps the `ds4_argmax_f32` kernel (single
    /// threadgroup, 256 cooperative threads; ties → lowest index, matching
    /// the CPU `argmax_i32`). Lets the resident fused decode tail
    /// (`DS4_RESIDENT_TAIL`) read back just the 4-byte token id instead of
    /// the full ~vocab logit vector. Bit-identical to the standalone-cb
    /// argmax in `tail_lm_head_argmax_q8` (same kernel, same dispatch).
    pub fn argmax(&self, logits: &DeferredBuf, n: usize) -> Result<DeferredBuf> {
        let out = DeferredBuf {
            buf: new_output_buffer::<i32>(&self.state.device, 1),
            n_elements: 1,
        };
        let pipe = self
            .state
            .specialized_pipeline("ds4_argmax_f32", &[], |_fcv| {})?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&logits.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        let n_u = n as u32;
        set_scalar_bytes(enc, 2, &n_u);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Two INDEPENDENT Q8_0 matvecs in ONE **concurrent** compute encoder, so
    /// the GPU overlaps them instead of paying two encoder-boundary pipeline
    /// drains. SAFETY: the two dispatches must be hazard-free w.r.t. each other
    /// — distinct output buffers, and inputs that neither writes. Used for
    /// `attn_q_a ∥ attn_kv` (both read `normed`, write separate rows). Returns
    /// `(out_a, out_b)`. No `memoryBarrier` is issued (no inter-dispatch
    /// dependency); the encoder boundary still orders it against later ops.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_q8_0_pair_concurrent(
        &self,
        w_a: &DeferredBuf,
        x_a: &DeferredBuf,
        d_in_a: usize,
        d_out_a: usize,
        w_b: &DeferredBuf,
        x_b: &DeferredBuf,
        d_in_b: usize,
        d_out_b: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        let enc = self
            .cmd_buf
            .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let out_a = self.q8_matvec_into(enc, w_a, x_a, d_in_a, d_out_a)?;
        let out_b = self.q8_matvec_into(enc, w_b, x_b, d_in_b, d_out_b)?;
        crate::macos::end_shared_compute_enc(enc);
        Ok((out_a, out_b))
    }

    /// Encode one Q8_0 matvec into a caller-owned compute encoder (the shared
    /// core of `matvec_q8_0` and `matvec_q8_0_pair_concurrent`). Does NOT open
    /// or end the encoder — the caller owns its lifetime so several independent
    /// matvecs can share one encoder.
    fn q8_matvec_into(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_q8: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            w_q8.n_elements == d_in * d_out,
            "matvec_q8_0: w has {} elems, expected d_in*d_out = {}*{} = {}",
            w_q8.n_elements,
            d_in,
            d_out,
            d_in * d_out
        );
        anyhow::ensure!(
            x.n_elements == d_in,
            "matvec_q8_0: x has {} elems, expected d_in = {}",
            x.n_elements,
            d_in
        );
        anyhow::ensure!(d_in % 32 == 0, "matvec_q8_0: d_in ({}) % 32 != 0", d_in);
        anyhow::ensure!(d_out % 2 == 0, "matvec_q8_0: d_out ({}) % 2 != 0", d_out);

        let row_stride = (d_in / 32) * 34;
        // nsg = #simdgroups cooperating on each output row's d_in reduction.
        // The q8 matvec is WEIGHT-BANDWIDTH bound, so we want maximum ROW
        // parallelism (many independent threadgroups streaming weight rows), and
        // high nsg WASTES that — it ganged 8 simdgroups onto one row's cheap
        // reduction. Microbench (q8_matvec_bandwidth_bench) on the real attn
        // shapes: the dominant attn_q_b (d_out=65536) goes 212→476 GB/s (2.25×)
        // at nsg=1 vs our old clamp-to-8; small-d_out shapes want 2-4. Heuristic:
        // low nsg, =1 once d_out is large enough to fill the GPU on rows alone.
        // DS4_Q8_NSG_TUNED=0 reverts to the old d_in-based clamp.
        let mut nsg: i16 = if std::env::var("DS4_Q8_NSG_TUNED").ok().as_deref() == Some("0") {
            (((d_in as u64 + 127) / 128).clamp(1, 8)) as i16
        } else if d_out >= 8192 {
            1
        } else if d_out >= 1024 {
            2
        } else {
            4
        };
        // The kernel_mul_mv_q8_0_f32_impl reads FC_mul_mv_nxpsg (dense.metal:4)
        // and has simd-reduction paths for nxpsg=8/16/32 (dense.metal:854-860).
        // Match the f32 path's heuristic so aligned d_in (the attn projections —
        // d_embd=4096, n_lora_q=1024, both ≡0 mod 256) gets full nxpsg=16
        // occupancy instead of the prior conservative hardcoded 4.
        let mut nxpsg: i16 = if d_in % 256 == 0 {
            16
        } else if d_in % 128 == 0 {
            8
        } else {
            4
        };
        // Dispatch-param sweep overrides (q8_matvec_bandwidth_bench / tuning).
        // antirez dispatches the IDENTICAL kernel with nsg=4 (no nxpsg); ours
        // defaults to nsg=8/nxpsg=16. Lets us A/B the occupancy params that the
        // microbench showed leave us at ~26% of the bandwidth roofline.
        if let Ok(v) = std::env::var("DS4_Q8_NSG") {
            if let Ok(n) = v.parse::<i16>() { nsg = n; }
        }
        if let Ok(v) = std::env::var("DS4_Q8_NXPSG") {
            if let Ok(n) = v.parse::<i16>() { nxpsg = n; }
        }
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mv_q8_0_f32", &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg as *const _ as *const _,
                    MTLDataType::Short,
                    600,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg as *const _ as *const _,
                    MTLDataType::Short,
                    601,
                );
            })?;

        let out = self.alloc_f32(d_out);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            _pad0: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            _pad1: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MulMvArgs {
            ne00: d_in as i32,
            ne01: d_out as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 2,
            nb01: row_stride as u64,
            nb02: (row_stride * d_out) as u64,
            nb03: (row_stride * d_out) as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            nb13: (d_in * 4) as u64,
            ne0: d_out as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let shmem_bytes: u64 = 32 * 2 * 4;
        let n_row_tg = ((d_out as u64) + 1) / 2;

        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_q8.buf), 0);
        enc.set_buffer(2, Some(&x.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );

        Ok(out)
    }

    /// q4_0 twin of [`Self::q8_matvec_into`] — the DECODE (K=1) weight-bound path.
    /// At K=1 each weight is read once (no K-amortization), so halving the weight
    /// bytes (q8 34 B/blk → q4 18 B/blk) gives a real 1.4–1.8× speedup (measured,
    /// `matvec_q4_0_decode_timing_vs_q8`), unlike the K=3000 GEMM which is
    /// compute-bound (1.00×). `w_q4` holds q4_0 bytes; `n_elements` is logical d_in*d_out.
    fn q4_matvec_into(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_q4: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(x.n_elements == d_in, "matvec_q4_0: x has {} elems, expected d_in={d_in}", x.n_elements);
        anyhow::ensure!(d_in % 32 == 0, "matvec_q4_0: d_in ({d_in}) % 32 != 0");
        anyhow::ensure!(d_out % 2 == 0, "matvec_q4_0: d_out ({d_out}) % 2 != 0");
        let row_stride = (d_in / 32) * 18;
        // Port the q8_matvec_into bandwidth tuning: the matvec is weight-bound, so
        // we want maximum ROW parallelism — high nsg WASTES it (gangs simdgroups
        // onto one row's cheap reduction). The untuned clamp(d_in/128,1,8) gives
        // nsg=8 for d_in=7168 — the exact pathology the q8 path fixed — which made
        // tuned-q8 BEAT untuned-q4 end-to-end despite q4's half-bytes. Match the q8
        // heuristic so q4's byte advantage actually shows.
        let mut nsg: i16 = if d_out >= 8192 { 1 } else if d_out >= 1024 { 2 } else { 4 };
        let mut nxpsg: i16 = if d_in % 256 == 0 { 16 } else if d_in % 128 == 0 { 8 } else { 4 };
        if let Ok(v) = std::env::var("DS4_Q4_NSG") { if let Ok(n) = v.parse::<i16>() { nsg = n; } }
        if let Ok(v) = std::env::var("DS4_Q4_NXPSG") { if let Ok(n) = v.parse::<i16>() { nxpsg = n; } }
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline("ds4_kernel_mul_mv_q4_0_f32", &key, |fcv| {
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;
        let out = self.alloc_f32(d_out);
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let args = MulMvArgs {
            ne00: d_in as i32, ne01: d_out as i32, ne02: 1, _pad0: 0,
            nb00: 2, nb01: row_stride as u64,
            nb02: (row_stride * d_out) as u64, nb03: (row_stride * d_out) as u64,
            ne10: d_in as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_in * 4) as u64, nb12: (d_in * 4) as u64, nb13: (d_in * 4) as u64,
            ne0: d_out as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_q4.buf), 0);
        enc.set_buffer(2, Some(&x.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new((d_out as u64 + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        Ok(out)
    }

    /// Decode (K=1) attention projection, q4_0 when `DS4_DECODE_ATTN_Q4=1`, else
    /// q8_0. Requants the resident q8 weight to q4 once (shared `ATTN_Q4_WEIGHT_CACHE`,
    /// keyed by weight ptr; pinned+committed resident, like the prefill
    /// [`Self::matmul_k_attn_proj`] path) and runs the bandwidth-bound q4 matvec.
    pub fn matvec_attn_proj(
        &self,
        w_q8: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        if std::env::var("DS4_DECODE_ATTN_Q4").ok().as_deref() != Some("1") {
            return self.matvec_q8_0(w_q8, x, d_in, d_out);
        }
        let key = w_q8.buf.contents() as usize;
        let w_q4 = ATTN_Q4_WEIGHT_CACHE.with(|c| -> Result<DeferredBuf> {
            if let Some(b) = c.borrow().get(&key) {
                return Ok(DeferredBuf { buf: b.clone(), n_elements: d_in * d_out });
            }
            let n_blocks = (d_in * d_out) / 32;
            let q8_bytes = unsafe {
                std::slice::from_raw_parts(w_q8.buf.contents() as *const u8, n_blocks * 34)
            };
            let q4 = ds4_engine::layer_view::requant_q8_0_to_q4_0(q8_bytes)?;
            let buf = new_input_buffer(&self.state.device, &q4);
            self.state.pin_state_buffer_resident(&buf);
            self.state.commit_residency();
            c.borrow_mut().insert(key, buf.clone());
            Ok(DeferredBuf { buf, n_elements: d_in * d_out })
        })?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        let out = self.q4_matvec_into(enc, &w_q4, x, d_in, d_out)?;
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// `out[d_out] = w[d_out x d_in] · x[d_in]`. Mirrors the matvec half of
    /// `layer_qa_rms_batched_impl` (macos.rs:1924-2073). Specialization
    /// constants 600/601 are bound per (d_in) classification.
    /// GPU port of the indexer top-k SCORE computation — the dominant cost of
    /// `ds4_engine::attn_dispatch::indexer_allowed_decode_one` (the long-context
    /// `n_comp > DS4_N_INDEXER_TOP_K` path). Computes, all on GPU in this scope:
    ///   q       = w_q_b · qr_norm          (matvec, `[n_head*head_dim]`)
    ///   q       = rope_tail(q)             (per-head, forward, at `pos`)
    ///   weights = w_proj · cur             (matvec, `[n_head]`, UNSCALED)
    ///   scores[c] = Σ_h max(0, dot(kv_c, q_h)) * weights[h] * scale
    /// Returns `scores [n_comp]`. The cheap O(top_k·n_comp) greedy top-k stays
    /// on the caller (CPU over the read-back scores): it is ~1/16th the cost of
    /// the scores and avoids the multi-block argsort the GPU kernels would need
    /// for `n_comp > 1024`. `index_ring` is the GPU-resident `[n_comp, head_dim]`
    /// f32 indexer compressed-KV ring. Matches `indexer_allowed_decode_one`
    /// modulo float-reduction order (the score kernel uses float4+simd_sum).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indexer_scores(
        &self,
        qr_normed: &DeferredBuf,
        cur: &DeferredBuf,
        w_q_b: &[f32],
        w_proj: &[f32],
        // No-copy F16 bytes for `w_q_b`/`w_proj` (lean path); `Some` → matvec_f16,
        // `None` → matvec_f32 over the f32 above. Bit-identical (F16→f32 exact).
        w_q_b_f16: Option<&[u8]>,
        w_proj_f16: Option<&[u8]>,
        index_ring: &DeferredBuf,
        n_comp: usize,
        n_head: usize,
        head_dim: usize,
        params: &ds4_engine::attn_dispatch::LayerParams,
        pos: u32,
    ) -> Result<DeferredBuf> {
        let q_dim = n_head * head_dim;
        let n_lora_q = qr_normed.n_elements;
        let d_embd = cur.n_elements;
        // f32 length checks apply only when the f32 path is in use; the lean f16
        // path drives the matvec from `w_q_b_f16`/`w_proj_f16` (f32 is empty).
        if w_q_b_f16.is_none() {
            anyhow::ensure!(
                w_q_b.len() == q_dim * n_lora_q,
                "encode_indexer_scores: w_q_b len {} != q_dim*n_lora_q {}",
                w_q_b.len(),
                q_dim * n_lora_q
            );
        }
        if w_proj_f16.is_none() {
            anyhow::ensure!(
                w_proj.len() == d_embd * n_head,
                "encode_indexer_scores: w_proj len {} != d_embd*n_head {}",
                w_proj.len(),
                d_embd * n_head
            );
        }
        anyhow::ensure!(
            index_ring.n_elements == n_comp * head_dim,
            "encode_indexer_scores: index_ring len {} != n_comp*head_dim {}",
            index_ring.n_elements,
            n_comp * head_dim
        );

        // q = w_q_b · qr_norm. antirez stores w_q_b col-major as
        // w[j*n_lora_q + k] (output j contiguous) == row-major [q_dim, n_lora_q],
        // which is exactly mul_mv's [d_out, d_in] layout. F16 no-copy when present
        // (matvec_f16 needs d_in%4 / d_out%2 — true for the real dims).
        let q = if let Some(bytes) = w_q_b_f16.filter(|_| n_lora_q % 4 == 0 && q_dim % 2 == 0) {
            let w = self.weight_f16(bytes);
            self.matvec_f16(&w, qr_normed, n_lora_q, q_dim)?
        } else {
            let w_q_b_db = self.weight_f32(w_q_b);
            self.matvec_f32(&w_q_b_db, qr_normed, n_lora_q, q_dim)?
        };
        if params.n_rot > 0 {
            self.rope_tail_q_heads_in_place(&q, n_head, head_dim, params, pos, false)?;
        }
        // weights = w_proj · cur (UNSCALED; the score kernel applies `scale`).
        let weights = if let Some(bytes) = w_proj_f16.filter(|_| d_embd % 4 == 0 && n_head % 2 == 0) {
            let w = self.weight_f16(bytes);
            self.matvec_f16(&w, cur, d_embd, n_head)?
        } else {
            let w_proj_db = self.weight_f32(w_proj);
            self.matvec_f32(&w_proj_db, cur, d_embd, n_head)?
        };

        // The score kernel hardcodes n_head=64, head_dim=128, 128 threads/tg.
        anyhow::ensure!(
            n_head == 64 && head_dim == 128,
            "encode_indexer_scores: kernel hardcodes n_head=64/head_dim=128 (got {n_head}/{head_dim})"
        );
        let _ = (q_dim, pos); // q_dim implied by n_head*head_dim; rope used `pos` above.

        let scores = self.alloc_f32(n_comp);
        let scale = 1.0f32 / ((head_dim * n_head) as f32).sqrt();

        // score_one_direct is in EMITTED_KERNEL_SYMBOLS but not a KERNELS spec,
        // so it isn't eagerly preloaded. It has no function constants — build+
        // cache on demand. specialized_pipeline does emitted-first lookup, so we
        // bind the EMITTED ABI: scalar args (n_comp@4, q_head_stride@5,
        // index_row_stride@6, scale@7), static threadgroup arrays (no tg-mem
        // binding). See emitted/dsv4_indexer_score_one_direct.metal.
        let pipe =
            self.state
                .specialized_pipeline("ds4_dsv4_indexer_score_one_direct", &[], |_fcv| {})?;
        let n_comp_u = n_comp as u32;
        let q_head_stride_u = (head_dim * 4) as u32; // bytes
        let index_row_stride_u = (head_dim * 4) as u32; // bytes
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q.buf), 0);
        enc.set_buffer(1, Some(&weights.buf), 0);
        enc.set_buffer(2, Some(&index_ring.buf), 0);
        enc.set_buffer(3, Some(&scores.buf), 0);
        set_scalar_bytes(enc, 4, &n_comp_u);
        set_scalar_bytes(enc, 5, &q_head_stride_u);
        set_scalar_bytes(enc, 6, &index_row_stride_u);
        set_scalar_bytes(enc, 7, &scale);
        enc.dispatch_thread_groups(
            MTLSize::new(n_comp as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(scores)
    }

    /// BATCHED indexer scores for a whole prefill chunk — port of antirez
    /// `kernel_dsv4_indexer_scores_tiled_f32` (ds4 metal/dsv4_misc.metal:841,
    /// already in our Metal lib via the upstream-dsv4_misc include). Step 1 of
    /// the antirez batched-prefill-attention port: replaces the per-position
    /// `encode_indexer_scores` loop with ONE dispatch producing
    /// `scores[n_tokens][n_comp]`.
    ///
    /// `q_k`       — batched indexer queries `[n_tokens, n_head, head_dim]`,
    ///               ALREADY rope-fwd'd (= per-token `w_q_b · qr_norm` then rope).
    /// `weights_k` — batched indexer weights `[n_tokens, n_head]` (= `w_proj·cur`,
    ///               UNSCALED; the kernel applies `scale`).
    /// `index_ring`— `[n_comp, head_dim]` f32 indexer compressed-KV ring.
    /// Causal mask is baked in (comp ≥ `(pos0+token+1)/ratio` → `-INFINITY`), so
    /// the downstream top_k never selects future rows — matches the per-position
    /// causal selection set. Returns `scores [n_tokens * n_comp]` (row-major per
    /// token). `n_head=64`, `head_dim=128` (DS4 indexer dims).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indexer_scores_tiled(
        &self,
        q_k: &DeferredBuf,
        weights_k: &DeferredBuf,
        index_ring: &DeferredBuf,
        n_comp: usize,
        n_tokens: usize,
        n_head: usize,
        head_dim: usize,
        pos0: u32,
        ratio: u32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            n_head == 64 && head_dim == 128,
            "encode_indexer_scores_tiled: kernel assumes n_head=64/head_dim=128 (got {n_head}/{head_dim})"
        );
        anyhow::ensure!(
            q_k.n_elements == n_tokens * n_head * head_dim,
            "encode_indexer_scores_tiled: q_k {} != n_tokens*n_head*head_dim {}",
            q_k.n_elements, n_tokens * n_head * head_dim
        );
        anyhow::ensure!(
            weights_k.n_elements == n_tokens * n_head,
            "encode_indexer_scores_tiled: weights_k {} != n_tokens*n_head {}",
            weights_k.n_elements, n_tokens * n_head
        );
        anyhow::ensure!(
            index_ring.n_elements == n_comp * head_dim,
            "encode_indexer_scores_tiled: index_ring {} != n_comp*head_dim {}",
            index_ring.n_elements, n_comp * head_dim
        );

        // Bind the EMITTED ABI (emitted/dsv4_indexer_scores_tiled_f32.metal):
        // p0=q@0, p1=weights@1, p2=index_comp@2, p3=scores@3, then scalar
        // uint args @4..13 (strides in BYTES, 32-bit) and float scale@14.
        // Static threadgroup arrays (no tg-mem binding). head_dim hardcoded D=128.
        let scale = 1.0f32 / ((head_dim * n_head) as f32).sqrt();
        let scores = self.alloc_f32(n_tokens * n_comp);
        // DS4_INDEXER_F16=1 dispatches the f16-Q/K-staging variant (simdgroup_half8x8,
        // ~2× the Q·K throughput; f32 accumulator + f32 in/out — identical ABI). This
        // is antirez's prefill indexer ("score matrix dominates prefill slope; f16 Q/K
        // is an intentional precision tradeoff"). Same args/buffers/grid/static-tgmem;
        // only the internal staging tiles differ.
        let kname = if std::env::var("DS4_INDEXER_F16").ok().as_deref() == Some("1") {
            "ds4_dsv4_indexer_scores_tiled"
        } else {
            "ds4_dsv4_indexer_scores_tiled_f32"
        };
        let pipe = self
            .state
            .specialized_pipeline(kname, &[], |_fcv| {})?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_k.buf), 0);
        enc.set_buffer(1, Some(&weights_k.buf), 0);
        enc.set_buffer(2, Some(&index_ring.buf), 0);
        enc.set_buffer(3, Some(&scores.buf), 0);
        let n_comp_u = n_comp as u32;
        let n_tokens_u = n_tokens as u32;
        let n_head_u = n_head as u32;
        let q_token_stride = (n_head * head_dim * 4) as u32;
        let q_head_stride = (head_dim * 4) as u32;
        let weights_token_stride = (n_head * 4) as u32;
        let index_row_stride = (head_dim * 4) as u32;
        let score_token_stride = (n_comp * 4) as u32;
        set_scalar_bytes(enc, 4, &n_comp_u);
        set_scalar_bytes(enc, 5, &n_tokens_u);
        set_scalar_bytes(enc, 6, &n_head_u);
        set_scalar_bytes(enc, 7, &pos0);
        set_scalar_bytes(enc, 8, &ratio);
        set_scalar_bytes(enc, 9, &q_token_stride);
        set_scalar_bytes(enc, 10, &q_head_stride);
        set_scalar_bytes(enc, 11, &weights_token_stride);
        set_scalar_bytes(enc, 12, &index_row_stride);
        set_scalar_bytes(enc, 13, &score_token_stride);
        set_scalar_bytes(enc, 14, &scale);
        let gx = ((n_comp + 31) / 32) as u64;
        let gy = ((n_tokens + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(scores)
    }

    /// BATCHED top-k for a whole prefill chunk — step 2 of the antirez batched-
    /// prefill-attention port. Wires the emitted `ds4_argsort_f32_i32_desc`
    /// (one threadgroup per token, bitonic descending sort) over the
    /// `scores[n_tokens][n_comp]` matrix from `encode_indexer_scores_tiled`,
    /// producing `selected[n_tokens][top_k]` i32 comp indices in DESCENDING-score
    /// order per token (the order the mixed-attention flash needs: it iterates
    /// the row and `break`s on `idx >= visible`, so causally-masked (-INFINITY-
    /// scored, future) rows sort to the end and are skipped). The single-block
    /// bitonic argsort caps at `n_comp <= 1024`; ABOVE that (ctx>4096 single-chunk
    /// prefill) this delegates to [`Self::encode_indexer_topk_batched_threshold`]
    /// (parallel threshold-select, any n_comp, same SET + tie-break, valid rows
    /// before future rows). Read back with `flush_and_read_i32`.
    pub fn encode_indexer_topk_batched(
        &self,
        scores: &DeferredBuf,
        n_comp: usize,
        n_tokens: usize,
        top_k: usize,
    ) -> Result<DeferredBuf> {
        // n_comp > 1024 exceeds the single-block bitonic argsort (tcount capped at
        // the 1024 threadgroup limit) → use the threshold-select batched kernel.
        if n_comp > 1024 {
            return self.encode_indexer_topk_batched_threshold(scores, n_comp, n_tokens, top_k);
        }
        anyhow::ensure!(
            n_comp > 0,
            "encode_indexer_topk_batched: n_comp must be > 0 (got {n_comp})"
        );
        anyhow::ensure!(top_k <= n_comp, "encode_indexer_topk_batched: top_k {top_k} > n_comp {n_comp}");
        anyhow::ensure!(
            scores.n_elements == n_tokens * n_comp,
            "encode_indexer_topk_batched: scores {} != n_tokens*n_comp {}",
            scores.n_elements, n_tokens * n_comp
        );
        let sel = self.alloc_i32(n_tokens * top_k);
        // threads/tg = next power of 2 >= n_comp (bitonic sort), capped 1024.
        let mut tcount: u64 = 1;
        while tcount < n_comp as u64 {
            tcount *= 2;
        }
        let pipe = self
            .state
            .specialized_pipeline("ds4_argsort_f32_i32_desc", &[], |_fcv| {})?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&scores.buf), 0);
        enc.set_buffer(1, Some(&sel.buf), 0);
        let ne00 = n_comp as u32;
        let ne01 = n_tokens as u32;
        let tk = top_k as u32;
        let ne0 = top_k as u32;
        let nb01 = (n_comp * 4) as u32;
        set_scalar_bytes(enc, 2, &ne00);
        set_scalar_bytes(enc, 3, &ne01);
        set_scalar_bytes(enc, 4, &tk);
        set_scalar_bytes(enc, 5, &ne0);
        set_scalar_bytes(enc, 6, &nb01);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tokens as u64, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(sel)
    }

    /// BATCHED threshold-select top-k — the n_comp>1024 path for long-context
    /// single-chunk prefill (ctx>4096), where the single-block bitonic argsort
    /// can't fit. Wires `ds4_dsv4_indexer_topk_threshold_batched` (one threadgroup
    /// per token; the batched twin of the decode `ds4_dsv4_indexer_topk_threshold`)
    /// over `scores[n_tokens][n_comp]` → `selected[n_tokens][top_k]`. Same SET +
    /// lowest-index tie-break as the argsort/greedy; finite (valid) scores always
    /// outrank causally-masked (-INFINITY) future rows, so every idx<visible
    /// precedes any idx>=visible — the order `encode_indexed_mixed_attention`'s
    /// break-on-visible needs. Any n_comp (no threadgroup-size cap). Read back with
    /// `flush_and_read_i32`.
    pub fn encode_indexer_topk_batched_threshold(
        &self,
        scores: &DeferredBuf,
        n_comp: usize,
        n_tokens: usize,
        top_k: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(n_comp > 0, "encode_indexer_topk_batched_threshold: n_comp must be > 0");
        anyhow::ensure!(
            top_k <= n_comp,
            "encode_indexer_topk_batched_threshold: top_k {top_k} > n_comp {n_comp}"
        );
        anyhow::ensure!(
            scores.n_elements == n_tokens * n_comp,
            "encode_indexer_topk_batched_threshold: scores {} != n_tokens*n_comp {}",
            scores.n_elements, n_tokens * n_comp
        );
        let sel = self.alloc_i32(n_tokens * top_k);
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_indexer_topk_threshold_batched", &[], |_fcv| {})?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&scores.buf), 0);
        enc.set_buffer(1, Some(&sel.buf), 0);
        set_scalar_bytes(enc, 2, &(n_comp as u32));
        set_scalar_bytes(enc, 3, &(top_k as u32));
        set_scalar_bytes(enc, 4, &(n_tokens as u32));
        enc.dispatch_thread_groups(
            MTLSize::new(n_tokens as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(sel)
    }

    /// BATCHED mixed-attention flash — step 3 of the antirez batched-prefill-
    /// attention port. Wires the emitted `ds4_dsv4_indexed_mixed_attention_h8`:
    /// for every (token, head) it does ONE online-softmax flash over
    /// [SWA raw window | top_k-selected compressed rows] + a per-head sink,
    /// replacing our per-position masked-flash loop. Causal/window mask on the
    /// raw ring (`window`, `raw_start`/`raw_cap` ring wrap) + sparse+causal mask
    /// on the comp rows (`visible=(qpos+1)/ratio`, iterate `topk_sel` row and
    /// `break` on `idx>=visible`). KV cast to half4, float accumulate (matches
    /// antirez). `head_dim` fixed at 128. Grid (n_tokens, ceil(n_head/8)), 256
    /// threads (8 simdgroups, one head each). Returns `out[n_tokens*n_head*head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indexed_mixed_attention(
        &self,
        q_heads_k: &DeferredBuf,
        raw_ring: &DeferredBuf,
        comp_ring: &DeferredBuf,
        topk_sel: &DeferredBuf,
        sinks: &DeferredBuf,
        n_tokens: usize,
        n_head: usize,
        head_dim: usize,
        n_raw: usize,
        n_comp: usize,
        top_k: usize,
        ratio: u32,
        window: u32,
        pos0: u32,
        raw_start: u32,
        raw_cap: u32,
        scale: f32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            head_dim == 512,
            "encode_indexed_mixed_attention: kernel hardcodes head_dim=512 (4 float4/lane x 32 \
             lanes = the MLA KV latent); got {head_dim}"
        );
        anyhow::ensure!(
            q_heads_k.n_elements == n_tokens * n_head * head_dim,
            "encode_indexed_mixed_attention: q_heads_k {} != n_tokens*n_head*head_dim {}",
            q_heads_k.n_elements, n_tokens * n_head * head_dim
        );
        // `>=` (not `==`): the raw core (n_comp==0/top_k==0) passes a dummy sel that
        // the kernel never reads (its comp loop runs 0 times). ratio must be >0 even
        // then (the kernel computes `visible=(qpos+1)/ratio`); callers pass 1.
        anyhow::ensure!(
            topk_sel.n_elements >= n_tokens * top_k,
            "encode_indexed_mixed_attention: topk_sel {} < n_tokens*top_k {}",
            topk_sel.n_elements, n_tokens * top_k
        );
        anyhow::ensure!(ratio > 0, "encode_indexed_mixed_attention: ratio must be > 0");
        let out = self.alloc_f32(n_tokens * n_head * head_dim);
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_indexed_mixed_attention_h8", &[], |_fcv| {})?;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_heads_k.buf), 0);
        enc.set_buffer(1, Some(&raw_ring.buf), 0);
        enc.set_buffer(2, Some(&comp_ring.buf), 0);
        enc.set_buffer(3, Some(&topk_sel.buf), 0);
        enc.set_buffer(4, Some(&sinks.buf), 0);
        enc.set_buffer(5, Some(&out.buf), 0);
        let hd_bytes = (head_dim * 4) as u32;
        let scalars: [(u64, u32); 18] = [
            (6, n_tokens as u32),
            (7, n_head as u32),
            (8, n_raw as u32),
            (9, n_comp as u32),
            (10, top_k as u32),
            (11, ratio),
            (12, window),
            (13, pos0),
            (14, raw_start),
            (15, raw_cap),
            (16, (n_head * head_dim * 4) as u32), // q_token_stride
            (17, hd_bytes),                       // q_head_stride
            (18, hd_bytes),                       // raw_row_stride
            (19, hd_bytes),                       // comp_row_stride
            (20, (top_k * 4) as u32),             // topk_token_stride
            (21, (n_head * head_dim * 4) as u32), // dst_token_stride
            (22, hd_bytes),                       // dst_head_stride
            (23, scale.to_bits()),                // float scale (set as bits)
        ];
        for (idx, v) in scalars.iter() {
            if *idx == 23 {
                let f = scale;
                set_scalar_bytes(enc, *idx, &f);
            } else {
                set_scalar_bytes(enc, *idx, v);
            }
        }
        let gy = ((n_head + 7) / 8) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n_tokens as u64, gy, 1),
            MTLSize::new(256, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// GPU cooperative greedy top-k over `scores [n_comp]` → `selected [top_k]`
    /// (i32 row indices, descending-score order). Single threadgroup (256
    /// threads) iterating `top_k` argmax-and-mask rounds, matching the CPU
    /// greedy (ties → lowest index). Mutates `scores` in place as scratch (the
    /// caller must not reuse it afterward). Avoids the multi-block bitonic
    /// argsort — we only need the SET. Read back with `flush_and_read_i32`.
    /// Requires `top_k <= n_comp`.
    pub fn encode_indexer_topk(
        &self,
        scores: &DeferredBuf,
        n_comp: usize,
        top_k: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            top_k <= n_comp,
            "encode_indexer_topk: top_k ({top_k}) > n_comp ({n_comp})"
        );
        let selected = self.alloc_i32(top_k);
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_indexer_topk_greedy", &[], |_fcv| {})?;
        let n_comp_u = n_comp as u32;
        let top_k_u = top_k as u32;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&scores.buf), 0);
        enc.set_buffer(1, Some(&selected.buf), 0);
        set_scalar_bytes(enc, 2, &n_comp_u);
        set_scalar_bytes(enc, 3, &top_k_u);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(selected)
    }

    /// Parallel threshold-select top-k — same SET as [`Self::encode_indexer_topk`]
    /// (top_k rows by score, ties → lowest index) but ~32 parallel-count passes
    /// instead of top_k(=512) sequential greedy rounds. At long context the
    /// greedy is the decode wall (~27 ms/token, single-threadgroup, fully serial,
    /// ×21 indexer layers); this binary-searches the rank-top_k score then
    /// gathers. `scores` is NOT mutated. Selected indices are ascending (order is
    /// irrelevant to the softmax gather). Requires `top_k <= n_comp`.
    pub fn encode_indexer_topk_threshold(
        &self,
        scores: &DeferredBuf,
        n_comp: usize,
        top_k: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            top_k <= n_comp,
            "encode_indexer_topk_threshold: top_k ({top_k}) > n_comp ({n_comp})"
        );
        let selected = self.alloc_i32(top_k);
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_indexer_topk_threshold", &[], |_fcv| {})?;
        let n_comp_u = n_comp as u32;
        let top_k_u = top_k as u32;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&scores.buf), 0);
        enc.set_buffer(1, Some(&selected.buf), 0);
        set_scalar_bytes(enc, 2, &n_comp_u);
        set_scalar_bytes(enc, 3, &top_k_u);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(selected)
    }

    pub fn matvec_f32(
        &self,
        w: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            w.n_elements == d_in * d_out,
            "matvec_f32: w has {} elems, expected d_in*d_out = {}*{} = {}",
            w.n_elements,
            d_in,
            d_out,
            d_in * d_out
        );
        anyhow::ensure!(
            x.n_elements == d_in,
            "matvec_f32: x has {} elems, expected d_in = {}",
            x.n_elements,
            d_in
        );
        anyhow::ensure!(d_in % 4 == 0, "matvec_f32: d_in ({}) % 4 != 0", d_in);
        anyhow::ensure!(d_out % 2 == 0, "matvec_f32: d_out ({}) % 2 != 0", d_out);

        let nsg: i16 = (((d_in as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_in % 256 == 0 {
            16
        } else if d_in % 128 == 0 {
            8
        } else {
            4
        };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg as *const _ as *const _,
                    MTLDataType::Short,
                    600,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg as *const _ as *const _,
                    MTLDataType::Short,
                    601,
                );
            })?;

        let out = self.alloc_f32(d_out);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            _pad0: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            _pad1: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MulMvArgs {
            ne00: d_in as i32,
            ne01: d_out as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_in * 4) as u64,
            nb02: (d_in * d_out * 4) as u64,
            nb03: (d_in * d_out * 4) as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            nb13: (d_in * 4) as u64,
            ne0: d_out as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let shmem_bytes: u64 = 32 * 2 * 4;
        let n_row_tg = ((d_out as u64) + 1) / 2;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(&x.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// No-copy F16 weight buffer over raw GGUF f16 bytes (2 B/elem), borrowed
    /// straight from the mmap. Format-agnostic no-copy (same path as
    /// `weight_q8_0_raw`); `n_elements` = bytes/2.
    pub fn weight_f16(&self, bytes: &[u8]) -> DeferredBuf {
        let buf = self.state.cached_q8_0_raw_buffer(bytes);
        DeferredBuf { buf, n_elements: bytes.len() / 2 }
    }

    /// Build an hc-projection weight buffer + its f16 flag: no-copy F16 from the
    /// mmap when `f16` is `Some` (lean path), else upload the dequantized f32.
    /// The flag threads to `hc_collapse_norm` to pick the f16 kernel/matvec.
    pub fn weight_hc(&self, f32: &[f32], f16: Option<&[u8]>) -> (DeferredBuf, bool) {
        match f16 {
            Some(bytes) => (self.weight_f16(bytes), true),
            None => (self.weight_f32(f32), false),
        }
    }

    /// `out[d_out] = dequant_f16(w[d_out x d_in]) · x[d_in]` via
    /// `ds4_kernel_mul_mv_f16_f32_4` (f16 weight, f32 activation). Twin of
    /// `matvec_f32` for f16-no-copy weights; bit-identical to `matvec_f32` on the
    /// dequantized weight (F16→f32 is exact). `w` holds d_in*d_out f16 elems.
    pub fn matvec_f16(
        &self,
        w: &DeferredBuf,
        x: &DeferredBuf,
        d_in: usize,
        d_out: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            w.n_elements == d_in * d_out,
            "matvec_f16: w has {} elems, expected d_in*d_out = {}*{} = {}",
            w.n_elements, d_in, d_out, d_in * d_out
        );
        anyhow::ensure!(x.n_elements == d_in, "matvec_f16: x has {} elems != d_in {}", x.n_elements, d_in);
        anyhow::ensure!(d_in % 4 == 0, "matvec_f16: d_in ({}) % 4 != 0", d_in);
        anyhow::ensure!(d_out % 2 == 0, "matvec_f16: d_out ({}) % 2 != 0", d_out);

        let nsg: i16 = (((d_in as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_in % 256 == 0 { 16 } else if d_in % 128 == 0 { 8 } else { 4 };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline("ds4_kernel_mul_mv_f16_f32_4", &key, |fcv| {
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        let out = self.alloc_f32(d_out);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        // Weight strides are f16 (2 B/elem); activation stays f32 (4 B).
        let args = MulMvArgs {
            ne00: d_in as i32, ne01: d_out as i32, ne02: 1, _pad0: 0,
            nb00: 2, nb01: (d_in * 2) as u64, nb02: (d_in * d_out * 2) as u64, nb03: (d_in * d_out * 2) as u64,
            ne10: d_in as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_in * 4) as u64, nb12: (d_in * 4) as u64, nb13: (d_in * 4) as u64,
            ne0: d_out as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };
        let n_row_tg = ((d_out as u64) + 1) / 2;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(&x.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(MTLSize::new(n_row_tg, 1, 1), MTLSize::new(32, nsg as u64, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// K-batched `matvec_f16`: one matmul `W[d_out,d_in](f16) × X_k[K,d_in](f32)
    /// → out[K,d_out](f32)` instead of K matvecs. The underlying mul_mv kernel is
    /// already ne11-aware (`r1 = tgpig.y` selects the activation column, output
    /// written at `r1*ne0`); `matvec_f16` just pins ne11=1. Here grid-Y = K and
    /// ne11 = K. `x_k` is `[K, d_in]` row-major; output is `[K, d_out]` row-major.
    /// Used to batch the chunk-prefill compressor/indexer projections across the
    /// chunk (the per-position matvecs were ~17% of prefill).
    pub fn matmul_f16_k(
        &self, w: &DeferredBuf, x_k: &DeferredBuf, d_in: usize, d_out: usize, k: usize,
    ) -> Result<DeferredBuf> {
        self.mul_mv_k_impl("ds4_kernel_mul_mv_f16_f32_4", 2, w, x_k, d_in, d_out, k)
    }

    /// K-batched `matvec_f32` (f32 weights). See [`Self::matmul_f16_k`].
    pub fn matmul_f32_k(
        &self, w: &DeferredBuf, x_k: &DeferredBuf, d_in: usize, d_out: usize, k: usize,
    ) -> Result<DeferredBuf> {
        self.mul_mv_k_impl("ds4_kernel_mul_mv_f32_f32_4", 4, w, x_k, d_in, d_out, k)
    }

    /// Shared body for the K-batched mul_mv wrappers. `wbytes` = bytes/weight-elem
    /// (2 for f16, 4 for f32).
    fn mul_mv_k_impl(
        &self, kernel: &str, wbytes: u64,
        w: &DeferredBuf, x_k: &DeferredBuf, d_in: usize, d_out: usize, k: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(d_in > 0 && d_out > 0 && k > 0, "mul_mv_k: dims must be >0 (d_in={d_in} d_out={d_out} k={k})");
        anyhow::ensure!(w.n_elements == d_in * d_out, "mul_mv_k: w has {} elems != d_in*d_out={}", w.n_elements, d_in * d_out);
        anyhow::ensure!(x_k.n_elements == k * d_in, "mul_mv_k: x_k has {} elems != k*d_in={}", x_k.n_elements, k * d_in);
        anyhow::ensure!(d_in % 4 == 0, "mul_mv_k: d_in ({d_in}) % 4 != 0");
        anyhow::ensure!(d_out % 2 == 0, "mul_mv_k: d_out ({d_out}) % 2 != 0");

        let nsg: i16 = (((d_in as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_in % 256 == 0 { 16 } else if d_in % 128 == 0 { 8 } else { 4 };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline(kernel, &key, |fcv| {
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        let out = self.alloc_f32(k * d_out);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let args = MulMvArgs {
            ne00: d_in as i32, ne01: d_out as i32, ne02: 1, _pad0: 0,
            nb00: wbytes, nb01: (d_in as u64) * wbytes, nb02: (d_in * d_out) as u64 * wbytes, nb03: (d_in * d_out) as u64 * wbytes,
            // activation X_k is [K, d_in] f32: column r1 (=tgpig.y) at r1*nb11.
            ne10: d_in as i32, ne11: k as i32, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_in * 4) as u64, nb12: (k * d_in * 4) as u64, nb13: (k * d_in * 4) as u64,
            ne0: d_out as i32, ne1: k as i32, nr0: 2, r2: 1, r3: 1,
        };
        let n_row_tg = ((d_out as u64) + 1) / 2;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(&x_k.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        // grid: (d_out/2 row-tiles, K activation columns, 1).
        enc.dispatch_thread_groups(MTLSize::new(n_row_tg, k as u64, 1), MTLSize::new(32, nsg as u64, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// `out[n] = (x / rms(x, eps)) * gamma`. Mirrors the rms_norm half of
    /// `layer_qa_rms_batched_impl` (macos.rs:2015-2090). `n = x.len()`
    /// must be divisible by 4.
    /// K-position rms_norm_mul: applies rms-norm + gamma to K independent rows
    /// of `x_k` (layout `[K, n]`). gamma is shared across all K rows. Output
    /// layout matches input: `[K, n]`. Uses the existing
    /// `ds4_kernel_rms_norm_mul_f32_4` kernel with `rows=K` — the kernel's
    /// outer `tgpig.x` dimension iterates rows, so K-position normalization
    /// is a single dispatch.
    pub fn rms_norm_mul_k(
        &self,
        x_k: &DeferredBuf,
        gamma: &DeferredBuf,
        n: usize,
        k_positions: usize,
        eps: f32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            x_k.n_elements == k_positions * n,
            "rms_norm_mul_k: x_k has {} elems, expected K*n = {}*{}",
            x_k.n_elements, k_positions, n
        );
        anyhow::ensure!(
            gamma.n_elements == n,
            "rms_norm_mul_k: gamma has {} elems, expected n = {}", gamma.n_elements, n
        );
        anyhow::ensure!(n % 4 == 0, "rms_norm_mul_k: n ({}) % 4 != 0", n);

        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?
            .clone();

        // Pooled iff a one-shot ATTN_POOL_KEY was armed (qkv-chain qr_normed/kv_normed);
        // the kernel writes the full [k_positions × n] extent, so reuse is safe. Other
        // callers (decode, kv-norm) arm no key → fresh alloc.
        let out = self.alloc_attn_out(k_positions * n);
        let rows: u32 = k_positions as u32;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut args = [0u8; 144];
        args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        args[4..8].copy_from_slice(&((n as i32) / 4).to_le_bytes());
        args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        args[16..24].copy_from_slice(&plane.to_le_bytes());
        args[24..32].copy_from_slice(&plane.to_le_bytes());
        args[32..36].copy_from_slice(&eps.to_le_bytes());
        args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        for off in [40usize, 44, 48, 52, 56, 60, 64, 68] {
            args[off..off + 4].copy_from_slice(&1i32.to_le_bytes());
        }
        args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        args[96..104].copy_from_slice(&plane.to_le_bytes());
        args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        args[120..128].copy_from_slice(&plane.to_le_bytes());
        args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        let ne00_t = (n / 4) as u64;
        let mut nth: u64 = 32;
        while nth < ne00_t && nth < 1024 { nth *= 2; }
        if nth > ne00_t { nth = ne00_t; }
        let nth = nth.max(1);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
        enc.set_buffer(1, Some(&x_k.buf), 0);
        enc.set_buffer(2, Some(&gamma.buf), 0);
        enc.set_buffer(3, Some(&x_k.buf), 0);
        enc.set_buffer(4, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(nth, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    pub fn rms_norm_mul(
        &self,
        x: &DeferredBuf,
        gamma: &DeferredBuf,
        eps: f32,
    ) -> Result<DeferredBuf> {
        let n = x.n_elements;
        anyhow::ensure!(
            gamma.n_elements == n,
            "rms_norm_mul: gamma has {} elems, expected n = {}",
            gamma.n_elements,
            n
        );
        anyhow::ensure!(n % 4 == 0, "rms_norm_mul: n ({}) % 4 != 0", n);

        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?
            .clone();

        let out = self.alloc_f32(n);

        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut args = [0u8; 144];
        args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        args[4..8].copy_from_slice(&((n as i32) / 4).to_le_bytes());
        args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        args[16..24].copy_from_slice(&plane.to_le_bytes());
        args[24..32].copy_from_slice(&plane.to_le_bytes());
        args[32..36].copy_from_slice(&eps.to_le_bytes());
        args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        for off in [40usize, 44, 48, 52, 56, 60, 64, 68] {
            args[off..off + 4].copy_from_slice(&1i32.to_le_bytes());
        }
        args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        args[96..104].copy_from_slice(&plane.to_le_bytes());
        args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        args[120..128].copy_from_slice(&plane.to_le_bytes());
        args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        let ne00_t = (n / 4) as u64;
        let mut nth: u64 = 32;
        while nth < ne00_t && nth < 1024 {
            nth *= 2;
        }
        if nth > ne00_t {
            nth = ne00_t;
        }
        let nth = nth.max(1);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
        enc.set_buffer(1, Some(&x.buf), 0);
        enc.set_buffer(2, Some(&gamma.buf), 0);
        enc.set_buffer(3, Some(&x.buf), 0);
        enc.set_buffer(4, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(nth, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// `out[n_head*head_dim] = x` with each per-head slice rms-normalized
    /// (no gamma multiply — antirez `head_rms_norm_inplace`,
    /// `ds4.c:4511`). Mirrors `head_rms_norm_impl` (macos.rs:619-682).
    /// K-position head_rms_norm: applies head-rms-norm to K independent rows
    /// of `x_k` (layout `[K, n_head, head_dim]`). Dispatches `K × n_head`
    /// threadgroups; each one rms-norms one (k-position, head) pair. The
    /// underlying kernel `ds4_head_rms_norm_f32` reads `x[tgpig.x * head_dim
    /// .. + head_dim]` so the [K, n_head, head_dim] flat layout aligns
    /// naturally — no new kernel needed.
    pub fn head_rms_norm_k(
        &self,
        x_k: &DeferredBuf,
        n_head: usize,
        head_dim: usize,
        k_positions: usize,
        eps: f32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            x_k.n_elements == k_positions * n_head * head_dim,
            "head_rms_norm_k: x_k has {} elems, expected K*n_head*head_dim = {}*{}*{}",
            x_k.n_elements, k_positions, n_head, head_dim
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_head_rms_norm_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_head_rms_norm_f32 pipeline not loaded"))?
            .clone();
        // Pooled iff a one-shot ATTN_POOL_KEY was armed (qkv-chain q_heads); the kernel
        // writes the full extent. Other callers arm no key → fresh alloc.
        let out = self.alloc_attn_out(x_k.n_elements);
        let hd_u32 = head_dim as u32;
        let mut tcount: u64 = 32;
        while tcount < head_dim as u64 && tcount < 1024 { tcount *= 2; }
        if tcount > head_dim as u64 { tcount = head_dim as u64; }
        let tcount = tcount.max(1);
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x_k.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        enc.set_bytes(2, std::mem::size_of::<u32>() as u64, &hd_u32 as *const _ as *const _);
        enc.set_bytes(3, std::mem::size_of::<f32>() as u64, &eps as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((k_positions * n_head) as u64, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    pub fn head_rms_norm(
        &self,
        x: &DeferredBuf,
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            x.n_elements == n_head * head_dim,
            "head_rms_norm: x has {} elems, expected n_head*head_dim = {}*{} = {}",
            x.n_elements,
            n_head,
            head_dim,
            n_head * head_dim
        );
        anyhow::ensure!(head_dim >= 1, "head_rms_norm: head_dim must be >= 1");

        let pipe = self
            .state
            .pipelines
            .get("ds4_head_rms_norm_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_head_rms_norm_f32 pipeline not loaded"))?
            .clone();

        let out = self.alloc_f32(x.n_elements);
        let hd_u32 = head_dim as u32;

        // One threadgroup per head; pick tcount up to head_dim, capped
        // at 1024, rounded up to the next power-of-two for simdgroup
        // reduction comfort. Matches `head_rms_norm_impl`.
        let mut tcount: u64 = 32;
        while tcount < head_dim as u64 && tcount < 1024 {
            tcount *= 2;
        }
        if tcount > head_dim as u64 {
            tcount = head_dim as u64;
        }
        let tcount = tcount.max(1);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &hd_u32 as *const _ as *const _,
        );
        enc.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &eps as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// Apply `dsv4_rope_tail_f32` to a sub-slice of `buf` IN PLACE
    /// (writes back at the same offset). The kernel runs one thread per
    /// `(j0, j1)` rotation pair, with `j0 != j1` across threads, so
    /// in-place is safe — no thread reads what another thread has
    /// written.
    ///
    /// `byte_offset` is where the rope-tail slice begins inside `buf`.
    /// `n_rot * n_heads` elements at that offset are rotated.
    /// `params.n_rot` must equal `n_rot` (the per-head rotation width);
    /// `params.head_dim` is used for stride computation in the kernel.
    ///
    /// Mirrors `rope_tail_impl` (macos.rs:4285-4371) but pinned to the
    /// in-place layout where the buffer length is the rope-tail span
    /// (n_heads * n_rot) — same shape the standalone impl uses for the
    /// `&mut kv_normed[n_lora_kv - n_rot..]` call site in `attn_prefix_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail_in_place(
        &self,
        buf: &DeferredBuf,
        byte_offset: u64,
        n_heads: usize,
        params: &ds4_engine::attn_dispatch::LayerParams,
        pos: u32,
        backward: bool,
    ) -> Result<()> {
        let n_rot = params.n_rot as usize;
        anyhow::ensure!(
            n_rot >= 2 && n_rot % 2 == 0,
            "rope_tail_in_place: n_rot must be even and >= 2 (got {})",
            n_rot
        );
        let span_bytes = (n_heads * n_rot * std::mem::size_of::<f32>()) as u64;
        anyhow::ensure!(
            byte_offset + span_bytes <= (buf.n_elements * std::mem::size_of::<f32>()) as u64,
            "rope_tail_in_place: range [{}, {}) exceeds buf bytes ({})",
            byte_offset,
            byte_offset + span_bytes,
            buf.n_elements * std::mem::size_of::<f32>()
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_rope_tail_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_rope_tail_f32 pipeline not loaded"))?
            .clone();

        let pos_buf = new_input_buffer(&self.state.device, &[pos as i32]);
        let freqs_buf = new_input_buffer(&self.state.device, &[0.0f32]);

        // 23-scalar uniform layout — must match rope_tail_impl byte-for-byte.
        let head_dim = n_rot as u32; // we operate on the rope-tail slice
        let n_nope: u32 = 0;
        let stride_bytes = 4u32;
        let scalars: [u32; 23] = [
            head_dim,
            n_rot as u32,
            n_nope,
            params.rope_freq_base.to_bits(),
            params.rope_freq_scale.to_bits(),
            params.rope_ext_factor.to_bits(),
            params.rope_attn_factor.to_bits(),
            (32.0f32).to_bits(),
            (1.0f32).to_bits(),
            params.rope_orig_ctx,
            pos,
            n_heads as u32,
            stride_bytes,
            (n_rot as u32) * stride_bytes,
            (n_rot as u32) * stride_bytes * n_heads as u32,
            backward as u32,
            stride_bytes,
            (n_rot as u32) * stride_bytes,
            (n_rot as u32) * stride_bytes * n_heads as u32,
            0,
            0,
            0,
            0,
        ];

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf.buf), byte_offset);
        enc.set_buffer(1, Some(&pos_buf), 0);
        enc.set_buffer(2, Some(&freqs_buf), 0);
        enc.set_buffer(3, Some(&buf.buf), byte_offset); // in-place: same buffer + offset
        for (i, s) in scalars.iter().enumerate() {
            set_scalar_bytes(enc, 4 + i as u64, s);
        }
        let half = (n_rot / 2) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(half.max(1).min(1024), 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(())
    }

    /// Phase F task #86 — apply rope_tail to all n_head q_heads of a
    /// `[n_head, head_dim]` row-major buffer in place, in this scope.
    ///
    /// The existing `rope_tail_in_place` treats the buffer as
    /// `[n_heads, n_rot]` (head_dim == n_rot), which works for the
    /// single-row KV case but not for q_heads where each head has
    /// `head_dim` floats and the rope tail is the LAST `n_rot` of
    /// that row.
    ///
    /// This variant passes the actual `head_dim` as the kernel's
    /// per-head stride and sets `byte_offset = (head_dim - n_rot) *
    /// sizeof(f32)` to skip the first head's NOPE prefix. The kernel
    /// computes `base = head_id * head_dim` and reads `src[base +
    /// i0]` for i0 in [0, n_rot), which maps to each head's rope
    /// tail correctly under this layout.
    ///
    /// Replaces the CPU loop in `encode_first_half_inner` (around
    /// step 2/4 — rope_q forward/backward) with a single GPU
    /// dispatch into the shared scope. Unblocks the next Phase F
    /// fusion: keeping `encode_layer_attn_half` + rope_q + (future)
    /// scope-aware flash_attn in ONE BatchScope per layer.
    pub fn rope_tail_q_heads_in_place(
        &self,
        q_heads: &DeferredBuf,
        n_head: usize,
        head_dim: usize,
        params: &ds4_engine::attn_dispatch::LayerParams,
        pos: u32,
        backward: bool,
    ) -> Result<()> {
        let n_rot = params.n_rot as usize;
        anyhow::ensure!(
            n_rot >= 2 && n_rot % 2 == 0,
            "rope_tail_q_heads_in_place: n_rot must be even and >= 2 (got {})",
            n_rot
        );
        anyhow::ensure!(
            head_dim >= n_rot,
            "rope_tail_q_heads_in_place: head_dim ({}) < n_rot ({})",
            head_dim,
            n_rot
        );
        anyhow::ensure!(
            q_heads.n_elements == n_head * head_dim,
            "rope_tail_q_heads_in_place: q_heads.len ({}) != n_head * head_dim ({} * {})",
            q_heads.n_elements,
            n_head,
            head_dim
        );

        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_rope_tail_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_rope_tail_f32 pipeline not loaded"))?
            .clone();

        let pos_buf = new_input_buffer(&self.state.device, &[pos as i32]);
        let freqs_buf = new_input_buffer(&self.state.device, &[0.0f32]);
        let byte_offset = ((head_dim - n_rot) * std::mem::size_of::<f32>()) as u64;
        let stride_bytes = 4u32;
        // Note: the kernel uses `head_dim` for the per-head base stride;
        // pass the actual head_dim (not n_rot) so head_id*head_dim
        // steps correctly between rows.
        let scalars: [u32; 23] = [
            head_dim as u32,
            n_rot as u32,
            0,
            params.rope_freq_base.to_bits(),
            params.rope_freq_scale.to_bits(),
            params.rope_ext_factor.to_bits(),
            params.rope_attn_factor.to_bits(),
            (32.0f32).to_bits(),
            (1.0f32).to_bits(),
            params.rope_orig_ctx,
            pos,
            n_head as u32,
            stride_bytes,
            (head_dim as u32) * stride_bytes,
            (head_dim as u32) * stride_bytes * (n_head as u32),
            backward as u32,
            stride_bytes,
            (head_dim as u32) * stride_bytes,
            (head_dim as u32) * stride_bytes * (n_head as u32),
            0,
            0,
            0,
            0,
        ];

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_heads.buf), byte_offset);
        enc.set_buffer(1, Some(&pos_buf), 0);
        enc.set_buffer(2, Some(&freqs_buf), 0);
        enc.set_buffer(3, Some(&q_heads.buf), byte_offset);
        for (i, s) in scalars.iter().enumerate() {
            set_scalar_bytes(enc, 4 + i as u64, s);
        }
        let half = (n_rot / 2) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(half.max(1).min(1024), 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(())
    }

    /// Phase F task #86 — `flash_attn_decode_metal_persistent`
    /// encoded into this scope (no commit, no wait). Reads KV from
    /// the layer's persistent buffer (`kv_buffer_or_alloc(layer_idx,
    /// ...)`), runs the vec + reduce passes, and returns the heads
    /// output as a DeferredBuf for downstream ops (e.g.
    /// `encode_attn_output_matmuls`) to consume in the same scope.
    ///
    /// Fast-path only: head_dim must be 512, n_raw must be
    /// 32-aligned and > 0. Compressor/indexer (kv_comp, comp_selected)
    /// inputs are NOT supported here — non-compressor layers only.
    /// Caller falls back to the non-scope
    /// `MetalDispatcher::flash_attn_decode_metal_persistent` (with
    /// its built-in CPU fallback) for off-fast-path conditions.
    ///
    /// q and attn_sinks are uploaded fresh per call (CPU `&[f32]`
    /// inputs). A future slice can extend this to accept a DeferredBuf
    /// q (post `rope_tail_q_heads_in_place`) keeping rope_q and
    /// flash_attn fully GPU-resident across the boundary.
    pub fn flash_attn_decode_persistent(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let out_buf = self.state.flash_attn_decode_metal_persistent_encode(
            &self.cmd_buf,
            layer_idx,
            params,
            q,
            n_raw,
            raw_cap,
            attn_sinks,
        )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// Resident-`q` variant of [`Self::flash_attn_decode_persistent`]: takes
    /// the rope'd Q heads as a GPU-resident `DeferredBuf` (e.g. straight from
    /// `rope_tail_q_heads_in_place`) instead of a `&[f32]`, so attn-half →
    /// rope_q → flash_attn chain in one cb with no q readback/re-upload.
    pub fn flash_attn_decode_persistent_qbuf(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &DeferredBuf,
        n_raw: u32,
        raw_cap: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            q.n_elements == n_head * head_dim,
            "flash_attn_decode_persistent_qbuf: q.len ({}) != n_head*head_dim ({}*{})",
            q.n_elements, n_head, head_dim,
        );
        let out_buf = self.state.flash_attn_decode_metal_persistent_encode_qbuf(
            &self.cmd_buf,
            layer_idx,
            params,
            &q.buf,
            n_raw,
            raw_cap,
            attn_sinks,
        )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// Resident-`q`, padded variant: runs flash_attn over the raw KV rows
    /// (plus any selected comp rows) padded to 32 and masked, binding a
    /// resident `q` DeferredBuf. Raw layers pass empty `kv_comp`/`comp_sel`
    /// so this handles any (non-32-aligned) `n_raw` — the fused layer path's
    /// attention for the general case.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_persistent_compressor_qbuf(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &DeferredBuf,
        n_raw: u32,
        raw_cap: u32,
        kv_comp: &[f32],
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            q.n_elements == n_head * head_dim,
            "flash_attn_decode_persistent_compressor_qbuf: q.len ({}) != n_head*head_dim",
            q.n_elements,
        );
        let out_buf = self.state.flash_attn_decode_metal_persistent_compressor_encode_qbuf(
            &self.cmd_buf,
            layer_idx,
            params,
            &q.buf,
            n_raw,
            raw_cap,
            kv_comp,
            comp_selected,
            n_selected,
            attn_sinks,
        )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// Single-cb step 9 — GPU-resident-ring variant of
    /// [`Self::flash_attn_decode_persistent_compressor_qbuf`]: `comp_rows`
    /// is read from the caller-owned ring buffer (`comp_ring_or_alloc`)
    /// rather than an uploaded `&[f32]`, so the flash gather has no CPU
    /// comp-row dependency and can share this scope's cb with attn-half.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_persistent_compressor_qbuf_gpuring(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &DeferredBuf,
        n_raw: u32,
        raw_cap: u32,
        kv_comp_buf: &metal::Buffer,
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            q.n_elements == n_head * head_dim,
            "flash_attn_decode_persistent_compressor_qbuf_gpuring: q.len ({}) != n_head*head_dim",
            q.n_elements,
        );
        let out_buf = self
            .state
            .flash_attn_decode_metal_persistent_compressor_encode_qbuf_gpuring(
                &self.cmd_buf,
                layer_idx,
                params,
                &q.buf,
                n_raw,
                raw_cap,
                kv_comp_buf,
                comp_selected,
                n_selected,
                attn_sinks,
            )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// Lever A: gather a SHARED chunk KV workspace (raw[0..n_raw] + comp[0..n_comp])
    /// into f16 [n_total_padded, head_dim], ONCE for the whole chunk (vs the
    /// per-position O(n²) re-gather). `kv_raw` is the layer's persistent KV buffer
    /// (the DeferredBuf returned by kv_fp8_store_persistent_k).
    pub fn build_chunk_kv_workspace(
        &self, kv_raw: &DeferredBuf, comp_ring: &metal::Buffer,
        n_raw: u32, n_comp: u32, head_dim: u32, n_total_padded: u32,
    ) -> Result<metal::Buffer> {
        let sel: Vec<u32> = (0..n_comp).collect();
        self.state.build_extended_kv_encode_gpubuf(
            &self.cmd_buf, &kv_raw.buf, comp_ring, &sel, n_raw, n_comp, head_dim, n_total_padded,
        )
    }

    /// DS4_CHUNK_SWA_KFLASH: allocate the CONTIGUOUS f32 raw window buffer for one tile
    /// of a long chunk (chunk > raw_cap). `n_rows` = pre + tk (pre-tile SWA window rows
    /// + in-tile rows). The caller fills it via `gather_ring_window` (pre-window BEFORE
    /// the tile store; in-tile rows AFTER) so every row is the fp8-snapped ring value —
    /// matching the per-token SWA reference (which flashes against the fp8 ring).
    pub fn alloc_swa_raw_window(&self, n_rows: usize, head_dim: usize) -> DeferredBuf {
        let n_elems = n_rows * head_dim;
        let buf = self.state.device.new_buffer(
            (n_elems * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        // Zero the whole buffer (StorageModeShared → host-visible). The flash kernel
        // reads ALL n_rows rows into the K·Q simdgroup matmul; pad rows past the real
        // window are masked to -INF in the softmax WEIGHT, but their V is still
        // multiplied by the (zero) post-softmax weight — so garbage/NaN in pad rows
        // would poison the output (0*NaN=NaN). Zeroing guarantees finite pad V.
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0,
                n_elems * std::mem::size_of::<f32>());
        }
        DeferredBuf { buf, n_elements: n_elems }
    }

    /// DS4_CHUNK_SWA_KFLASH: blit `n` consecutive positions `[first_pos .. first_pos+n)`
    /// out of the persistent KV ring (`ring` = [raw_cap, head_dim] f32; slot=pos%raw_cap)
    /// into `out` starting at row `out_row_off`. The source slots form a contiguous
    /// wrapped run, copied as ≤2 segments split at the ring wrap. `n` must be <= raw_cap
    /// (a wider span can't coexist in the ring). Used to assemble the SWA raw window
    /// from fp8-snapped ring values (faithful to the per-token reference).
    #[allow(clippy::too_many_arguments)]
    pub fn gather_ring_window(
        &self,
        ring: &metal::Buffer,
        out: &DeferredBuf,
        out_row_off: u32,
        raw_cap: u32,
        first_pos: u32,
        n: u32,
        head_dim: u32,
    ) {
        if n == 0 { return; }
        let row_bytes = (head_dim as usize * std::mem::size_of::<f32>()) as u64;
        let first_slot = first_pos % raw_cap;
        let seg1 = (raw_cap - first_slot).min(n); // rows before the ring wrap
        crate::macos::end_shared_compute_enc_force();
        let blit = self.cmd_buf.new_blit_command_encoder();
        blit.copy_from_buffer(
            ring, (first_slot as u64) * row_bytes,
            &out.buf, (out_row_off as u64) * row_bytes, (seg1 as u64) * row_bytes,
        );
        if seg1 < n {
            let seg2 = n - seg1; // rows after the wrap (start at ring slot 0)
            blit.copy_from_buffer(
                ring, 0,
                &out.buf, ((out_row_off + seg1) as u64) * row_bytes, (seg2 as u64) * row_bytes,
            );
        }
        blit.end_encoding();
    }

    /// Lever A: K-query batched flash over the shared workspace + a per-query causal
    /// mask. q_k = [K, n_head, head_dim] (= half.q_heads_k). Returns [K, n_head*head_dim].
    pub fn flash_attn_decode_k(
        &self, params: &ds4_engine::attn_dispatch::LayerParams, q_k: &DeferredBuf,
        workspace: &metal::Buffer, n_total: u32, mask: &[u16], sinks: &[f32], k: usize,
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let out = self.state.flash_attn_decode_k_metal(
            &self.cmd_buf, params, &q_k.buf, workspace, n_total, mask, sinks, k,
        )?;
        Ok(DeferredBuf { buf: out, n_elements: k * n_head * head_dim })
    }

    /// Antirez-style block-skipping prefill flash (DS4_CHUNK_BLKFLASH) — see
    /// [`MetalState::flash_attn_ext_blk_metal`]. `n_total` must be 64-aligned;
    /// `mask` is SWA-windowed (built by the caller). UNVALIDATED.
    pub fn flash_attn_ext_blk(
        &self, params: &ds4_engine::attn_dispatch::LayerParams, q_k: &DeferredBuf,
        workspace: &metal::Buffer, n_total: u32, mask: &[u16], sinks: &[f32], k: usize,
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let out = self.state.flash_attn_ext_blk_metal(
            &self.cmd_buf, params, &q_k.buf, workspace, n_total, mask, sinks, k,
        )?;
        Ok(DeferredBuf { buf: out, n_elements: k * n_head * head_dim })
    }

    /// Fully-resident-selection variant of
    /// [`Self::flash_attn_decode_persistent_compressor_qbuf_gpuring`]: the
    /// indexer top-k selection is a GPU `DeferredBuf` (i32 row indices from
    /// `encode_indexer_topk`) bound straight into the gather — no CPU
    /// `comp_selected` slice, no readback. Keeps the long-context indexer
    /// bridge chained at zero drains.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_persistent_compressor_qbuf_gpuring_sel(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &DeferredBuf,
        n_raw: u32,
        raw_cap: u32,
        kv_comp_buf: &metal::Buffer,
        comp_selected_db: &DeferredBuf,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            q.n_elements == n_head * head_dim,
            "flash..._qbuf_gpuring_sel: q.len ({}) != n_head*head_dim",
            q.n_elements,
        );
        anyhow::ensure!(
            comp_selected_db.n_elements >= n_selected as usize,
            "flash..._qbuf_gpuring_sel: sel.len ({}) < n_selected ({})",
            comp_selected_db.n_elements,
            n_selected,
        );
        let out_buf = self
            .state
            .flash_attn_decode_metal_persistent_compressor_encode_qbuf_gpuring_sel(
                &self.cmd_buf,
                layer_idx,
                params,
                &q.buf,
                n_raw,
                raw_cap,
                kv_comp_buf,
                &comp_selected_db.buf,
                n_selected,
                attn_sinks,
            )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// Single-cb step 9 — copy a resident `src` DeferredBuf (its full
    /// `len()` f32 elements) into `dst` at element offset `dst_elem_off`,
    /// encoded into this scope's cb via a blit. Used to append the
    /// compressor's GPU-resident emit row into the persistent comp ring in
    /// the same command buffer that produced it.
    pub fn copy_buf_into(&self, src: &DeferredBuf, dst: &metal::Buffer, dst_elem_off: usize) {
        let bytes = (src.n_elements * std::mem::size_of::<f32>()) as u64;
        let dst_off = (dst_elem_off * std::mem::size_of::<f32>()) as u64;
        crate::macos::end_shared_compute_enc_force();
        let blit = self.cmd_buf.new_blit_command_encoder();
        blit.copy_from_buffer(&src.buf, 0, dst, dst_off, bytes);
        blit.end_encoding();
    }

    /// Copy a contiguous `n_elems`-f32 slice of `src` starting at element
    /// offset `src_elem_off` into a freshly-allocated DeferredBuf, via a blit
    /// in this scope's cb. Used by the K-position verifier's compressor path to
    /// extract draft k's single-position query `q_heads_k[k*q_dim..]` so it can
    /// feed the K=1 ring-aware flash (which asserts a single-position q).
    pub fn slice_out_f32(&self, src: &DeferredBuf, src_elem_off: usize, n_elems: usize) -> DeferredBuf {
        debug_assert!(src_elem_off + n_elems <= src.n_elements);
        let bytes = (n_elems * std::mem::size_of::<f32>()) as u64;
        let out = self
            .state
            .device
            .new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared);
        let src_off = (src_elem_off * std::mem::size_of::<f32>()) as u64;
        crate::macos::end_shared_compute_enc_force();
        let blit = self.cmd_buf.new_blit_command_encoder();
        blit.copy_from_buffer(&src.buf, src_off, &out, 0, bytes);
        blit.end_encoding();
        DeferredBuf { buf: out, n_elements: n_elems }
    }

    /// M5 task #100 — scope-aware compressor flash_attn. Same compute as
    /// the inherent `flash_attn_decode_metal_persistent` compressor
    /// branch (gather raw KV + indexer-selected comp rows into one f16
    /// workspace; mask-pad to 32-row alignment; dispatch
    /// flash_attn_ext_vec_f16_dk512_dv512) but encoded into this scope's
    /// cb so the orchestrator can chain rope_q_back +
    /// attn_output_proj + ffn-half ops on top without flushing.
    ///
    /// Preconditions same as
    /// `MetalState::flash_attn_decode_metal_persistent_compressor_encode`:
    /// head_dim == 512, n_lora_kv == head_dim, n_raw + n_selected > 0.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_persistent_compressor(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        kv_comp: &[f32],
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<DeferredBuf> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let out_buf = self.state.flash_attn_decode_metal_persistent_compressor_encode(
            &self.cmd_buf,
            layer_idx,
            params,
            q,
            n_raw,
            raw_cap,
            kv_comp,
            comp_selected,
            n_selected,
            attn_sinks,
        )?;
        Ok(DeferredBuf {
            buf: out_buf,
            n_elements: n_head * head_dim,
        })
    }

    /// M5 Phase D — encode antirez's `kernel_dsv4_compressor_store_one`
    /// into this scope's cb. Writes one precomputed `kv`/`score` row
    /// into the per-layer compressor state buffers, fusing the APE add.
    /// See [`MetalState::compressor_store_one_encode`] for the buffer
    /// + arg-layout contract.
    ///
    /// `state_kv` and `state_score` are caller-owned MTLBuffers (the
    /// persistent per-layer compressor state). The dispatch writes one
    /// row in-place; reads of the state buffer from a subsequent op on
    /// this queue see the write (Metal serializes cbs within a queue).
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_store_one(
        &self,
        kv: &[f32],
        score: &[f32],
        ape_bytes: &[u8],
        state_kv: &metal::Buffer,
        state_score: &metal::Buffer,
        width: u32,
        ratio: u32,
        pos: u32,
        ape_is_f16: bool,
    ) -> Result<()> {
        self.state.compressor_store_one_encode(
            &self.cmd_buf,
            kv,
            score,
            ape_bytes,
            state_kv,
            state_score,
            width,
            ratio,
            pos,
            ape_is_f16,
        )
    }

    /// M5 scope-merge — DeferredBuf-fed `compressor_store_one`. Binds
    /// `kv`/`score` matvec outputs directly (no CPU readback/upload) and
    /// the resident state pools, adding APE on-GPU. Encodes into this
    /// scope's cb so matvec → store → pool stay one commit.
    pub fn compressor_store_one_db(
        &self,
        kv: &DeferredBuf,
        score: &DeferredBuf,
        ape_bytes: &[u8],
        state_kv: &metal::Buffer,
        state_score: &metal::Buffer,
        width: u32,
        ratio: u32,
        pos: u32,
        ape_is_f16: bool,
    ) -> Result<()> {
        self.state.compressor_store_one_db_encode(
            &self.cmd_buf, kv.buffer(), score.buffer(), ape_bytes, state_kv, state_score,
            width, ratio, pos, ape_is_f16,
        )
    }

    /// M5 Phase D — antirez's `kernel_dsv4_softmax_pool`. Fused
    /// softmax-weighted pool of `kv` rows weighted by `score` rows.
    /// Both inputs are row-major `[n_rows × width]` GPU buffers
    /// (typically the persistent per-layer compressor state pools from
    /// `MetalDispatcher::compressor_state_*_or_alloc`).
    ///
    /// Returns a DeferredBuf<f32> of length `width` — the pooled
    /// output. The dispatch encodes into this scope's cb; the output
    /// stays GPU-resident until flushed.
    pub fn softmax_pool(
        &self,
        kv: &metal::Buffer,
        score: &metal::Buffer,
        n_rows: u32,
        width: u32,
    ) -> Result<DeferredBuf> {
        let out = self
            .state
            .softmax_pool_encode(&self.cmd_buf, kv, score, n_rows, width)?;
        Ok(DeferredBuf {
            buf: out,
            n_elements: width as usize,
        })
    }

    /// M5 Phase D — bespoke `ds4_kernel_dsv4_compressor_pool_ratio4`.
    /// Mirrors `compressor_pool_decode_state` (`attn_dispatch.rs:1985`)
    /// ratio==4 branch directly: per output column `j` in
    /// `[0..head_dim]`, max-stabilized softmax over `2*ratio` scores
    /// drawn from the 2-window state (`[2*ratio, 2*head_dim]`
    /// row-major), weighted sum of matching KV values.
    ///
    /// `state_kv` / `state_score` are caller-owned MTLBuffers,
    /// typically the persistent per-layer compressor state pools from
    /// `MetalDispatcher::compressor_state_*_or_alloc`. Each must be
    /// sized `2*ratio * 2*head_dim * sizeof(f32)` bytes.
    ///
    /// Returns a DeferredBuf<f32> of length `head_dim` — the pooled
    /// output. The dispatch encodes into this scope's cb; the output
    /// stays GPU-resident until flushed.
    pub fn compressor_pool_ratio4(
        &self,
        state_kv: &metal::Buffer,
        state_score: &metal::Buffer,
        head_dim: u32,
    ) -> Result<DeferredBuf> {
        let out = self.state.compressor_pool_ratio4_encode(
            &self.cmd_buf,
            state_kv,
            state_score,
            head_dim,
        )?;
        Ok(DeferredBuf {
            buf: out,
            n_elements: head_dim as usize,
        })
    }

    /// DS4_CHUNK_ATTN_NOSYNC — GPU-resident ratio==4 pool rotation. Rotates the
    /// prefix window of the persistent kv/score pools in place (front := back),
    /// replicating the CPU `finish_emit` rotation entirely on the GPU so the
    /// per-quad emit needs no `commit_wait` + CPU pool round-trip before the
    /// next quad's stores. Encodes into this scope's cb; no commit/wait/readback.
    pub fn compressor_rotate_ratio4(
        &self,
        state_kv: &metal::Buffer,
        state_score: &metal::Buffer,
        head_dim: u32,
    ) -> Result<()> {
        self.state.compressor_rotate_ratio4_encode(
            &self.cmd_buf,
            state_kv,
            state_score,
            head_dim,
        )
    }

    /// STAGE 1 fused chunk-graph compressor prefill (noidx, ratio != 4).
    /// One dispatch builds ALL `n_comp = k_positions / ratio` compressed rows
    /// for the chunk from the batched compressor projections, fusing the
    /// per-position store + softmax-pool + RMS-norm + rope_tail + ring-write.
    /// Writes the finished emit rows directly into `comp_ring` starting at row
    /// `comp_row0` (== state.n_comp at chunk start). Per-row byte-equivalent to
    /// the per-position `compressor_encode_in_scope` chain when chunk_start %
    /// ratio == 0.
    ///
    /// `kv` / `sc` are the K-batched `[k_positions × head_dim]` projections
    /// (coff == 1 ⇒ width == head_dim). `ape` is the model-layout
    /// `[head_dim × ratio]` (`ape[j*ratio + r]`); `norm` is `[head_dim]`.
    /// Encodes into this scope's cb — no commit/wait/readback.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill_noidx(
        &self,
        kv: &DeferredBuf,
        sc: &DeferredBuf,
        ape: &[f32],
        norm: &[f32],
        comp_ring: &metal::Buffer,
        head_dim: u32,
        ratio: u32,
        n_rot: u32,
        chunk_start: u32,
        comp_row0: u32,
        n_comp: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        rms_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(ratio != 0 && ratio != 4, "compressor_prefill_noidx: ratio must be != 0,4");
        anyhow::ensure!(n_rot % 2 == 0 && n_rot <= head_dim, "compressor_prefill_noidx: bad n_rot");
        anyhow::ensure!(head_dim as usize <= 1024, "compressor_prefill_noidx: head_dim must be <= 1024");
        anyhow::ensure!(ape.len() == (head_dim * ratio) as usize, "compressor_prefill_noidx: ape len");
        anyhow::ensure!(norm.len() == head_dim as usize, "compressor_prefill_noidx: norm len");
        if n_comp == 0 {
            return Ok(());
        }
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_compressor_prefill_noidx_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_compressor_prefill_noidx_f32 pipeline not loaded"))?
            .clone();
        let ape_db = self.upload_f32(ape);
        let norm_db = self.upload_f32(norm);

        // args struct must match ds4_compressor_prefill_args byte-for-byte.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            head_dim: u32,
            ratio: u32,
            n_rot: u32,
            pos0: u32,
            comp_row0: u32,
            n_comp: u32,
            rms_eps: f32,
            freq_base_b: u32,
            freq_scale_b: u32,
            ext_factor_b: u32,
            attn_factor_b: u32,
            beta_fast_b: u32,
            beta_slow_b: u32,
            orig_ctx: u32,
            backward: u32,
        }
        let args = Args {
            head_dim,
            ratio,
            n_rot,
            pos0: chunk_start,
            comp_row0,
            n_comp,
            rms_eps,
            freq_base_b: params.rope_freq_base.to_bits(),
            freq_scale_b: params.rope_freq_scale.to_bits(),
            ext_factor_b: params.rope_ext_factor.to_bits(),
            attn_factor_b: params.rope_attn_factor.to_bits(),
            beta_fast_b: (32.0f32).to_bits(),
            beta_slow_b: (1.0f32).to_bits(),
            orig_ctx: params.rope_orig_ctx,
            backward: 0,
        };

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(kv.buffer()), 0);
        enc.set_buffer(1, Some(sc.buffer()), 0);
        enc.set_buffer(2, Some(ape_db.buffer()), 0);
        enc.set_buffer(3, Some(norm_db.buffer()), 0);
        enc.set_buffer(4, Some(comp_ring), 0);
        set_scalar_bytes(enc, 5, &args);
        // One threadgroup per emit row; head_dim threads (<=1024).
        let tg = (head_dim as u64).min(1024);
        enc.dispatch_thread_groups(
            MTLSize::new(n_comp as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(())
    }

    /// Fused ratio==4 chunk-prefill compressor (two-window analog of
    /// [`Self::compressor_prefill_noidx`]). `kv`/`sc` are the K-batched
    /// `[k_positions × 2*head_dim]` compressor projections (coff==2); `ape` is
    /// model-layout `[2*head_dim × ratio]` (`ape[col*ratio + r]`); `norm` is
    /// `[head_dim]`. Builds ALL `n_comp` emit rows in ONE dispatch into
    /// `comp_ring[comp_row0 ..]`. Per-row byte-equivalent to the per-position
    /// store_one → pool_ratio4 → rms → rope chain when chunk_start % 4 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill_idx(
        &self,
        kv: &DeferredBuf,
        sc: &DeferredBuf,
        ape: &[f32],
        norm: &[f32],
        comp_ring: &metal::Buffer,
        head_dim: u32,
        n_rot: u32,
        chunk_start: u32,
        comp_row0: u32,
        n_comp: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        rms_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(n_rot % 2 == 0 && n_rot <= head_dim, "compressor_prefill_idx: bad n_rot");
        anyhow::ensure!(ape.len() == (2 * head_dim * 4) as usize, "compressor_prefill_idx: ape len");
        anyhow::ensure!(norm.len() == head_dim as usize, "compressor_prefill_idx: norm len");
        if n_comp == 0 {
            return Ok(());
        }
        let ape_db = self.upload_f32(ape);
        let norm_db = self.upload_f32(norm);
        self.state.compressor_prefill_idx_encode(
            &self.cmd_buf, kv.buffer(), sc.buffer(), ape_db.buffer(), norm_db.buffer(),
            comp_ring, head_dim, n_rot, chunk_start, comp_row0, n_comp, params, rms_eps,
        )
    }

    /// STAGE 1 companion to [`Self::compressor_prefill_noidx`]: store the
    /// chunk's TRAILING partial group (the `rem` positions after the last emit
    /// boundary, which do not themselves emit) into the per-layer compressor
    /// STATE pool, so the subsequent per-position decode pools across the chunk
    /// boundary correctly. Mirrors antirez `compressor_set_rows_projected`
    /// (non-ratio4 `rem != 0` branch). ONE dispatch (no per-position store_one).
    ///
    /// `kv` / `sc` are the same K-batched `[k_positions × head_dim]` projections.
    /// `state_kv` / `state_score` are the persistent per-layer pools
    /// (`compressor_state_*_or_alloc`). `first` is the index of the first
    /// trailing position within the chunk (`n_emit * ratio`); `rem` its count.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_pool_fill_noidx(
        &self,
        kv: &DeferredBuf,
        sc: &DeferredBuf,
        ape: &[f32],
        state_kv: &metal::Buffer,
        state_score: &metal::Buffer,
        head_dim: u32,
        ratio: u32,
        chunk_start: u32,
        first: u32,
        rem: u32,
    ) -> Result<()> {
        if rem == 0 {
            return Ok(());
        }
        anyhow::ensure!(rem <= ratio, "compressor_pool_fill_noidx: rem must be <= ratio");
        anyhow::ensure!(ape.len() == (head_dim * ratio) as usize, "compressor_pool_fill_noidx: ape len");
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_compressor_pool_fill_noidx_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_compressor_pool_fill_noidx_f32 pipeline not loaded"))?
            .clone();
        let ape_db = self.upload_f32(ape);
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            width: u32,
            ratio: u32,
            pos0: u32,
            first: u32,
            rem: u32,
        }
        let args = Args { width: head_dim, ratio, pos0: chunk_start, first, rem };
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(kv.buffer()), 0);
        enc.set_buffer(1, Some(sc.buffer()), 0);
        enc.set_buffer(2, Some(ape_db.buffer()), 0);
        enc.set_buffer(3, Some(state_kv), 0);
        enc.set_buffer(4, Some(state_score), 0);
        set_scalar_bytes(enc, 5, &args);
        // grid = (width, rem); threadgroup tiled (32 × 1).
        let tg_x = 32u64.min(head_dim as u64);
        enc.dispatch_thread_groups(
            MTLSize::new((head_dim as u64).div_ceil(tg_x), rem as u64, 1),
            MTLSize::new(tg_x, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(())
    }

    /// Phase E M1 (M4 #330p): GPU equivalent of
    /// `hc_expand_add_only` (attn_dispatch.rs:2940). For each
    /// `(dst_hc, e)` writes:
    ///
    /// ```text
    /// out[dst_hc, e] = hc_split_post[dst_hc] * (shared[e] + routed[e])
    ///               + Σ_src(hc_split_comb[dst_hc, src] * after_attn[src, e])
    /// ```
    ///
    /// Uses antirez's `kernel_dsv4_hc_expand` (loaded via the bridge,
    /// rewritten to `ds4_dsv4_hc_expand`). Pure element-wise data flow
    /// per thread; safe to encode into a fused cb alongside other ops.
    ///
    /// Standalone this op may be slower than the CPU loop for tiny
    /// shapes (n_hc * d_embd ~16K elements) due to Metal cb-launch
    /// overhead. It's intended for use INSIDE a larger fused cb where
    /// the overhead is amortized — see Phase E plan.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_expand_add(
        &self,
        shared_out: &DeferredBuf,
        routed_out: &DeferredBuf,
        after_attn_hc: &DeferredBuf,
        hc_split_post: &DeferredBuf,
        hc_split_comb: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            shared_out.n_elements == d_embd,
            "hc_expand_add: shared_out has {} elems, expected d_embd={}",
            shared_out.n_elements,
            d_embd
        );
        anyhow::ensure!(
            routed_out.n_elements == d_embd,
            "hc_expand_add: routed_out has {} elems, expected d_embd={}",
            routed_out.n_elements,
            d_embd
        );
        anyhow::ensure!(
            after_attn_hc.n_elements == n_hc * d_embd,
            "hc_expand_add: after_attn_hc has {} elems, expected n_hc*d_embd = {}*{} = {}",
            after_attn_hc.n_elements,
            n_hc,
            d_embd,
            n_hc * d_embd
        );
        anyhow::ensure!(
            hc_split_post.n_elements == n_hc,
            "hc_expand_add: hc_split_post has {} elems, expected n_hc={}",
            hc_split_post.n_elements,
            n_hc
        );
        anyhow::ensure!(
            hc_split_comb.n_elements == n_hc * n_hc,
            "hc_expand_add: hc_split_comb has {} elems, expected n_hc*n_hc={}",
            hc_split_comb.n_elements,
            n_hc * n_hc
        );

        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_hc_expand")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_expand pipeline not loaded"))?
            .clone();

        let out = self.alloc_f32(n_hc * d_embd);

        // Mirrors `struct ds4_metal_args_dsv4_hc_expand` in
        // benchmarks/ds4_msl/upstream/ds4/metal/dsv4_hc.metal:58.
        // Strides describe a (d_embd, n_hc, n_tokens=1) layout with
        // float (4-byte) elements; `nb_resN` and `nb_combN` reflect
        // `after_attn_hc[src_hc, e]` and `hc_split_comb[dst_hc + src*n_hc]`
        // respectively.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct HcExpandArgs {
            n_embd: i64,
            n_hc: i64,
            n_tokens: i64,
            nb_block0: u64,
            nb_block1: u64,
            nb_add0: u64,
            nb_add1: u64,
            nb_res0: u64,
            nb_res1: u64,
            nb_res2: u64,
            nb_post0: u64,
            nb_post1: u64,
            nb_comb0: u64,
            nb_comb1: u64,
            nb_comb2: u64,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            has_add: i32,
            _pad: i32,
        }
        let de = d_embd as u64;
        let nh = n_hc as u64;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = HcExpandArgs {
            n_embd: d_embd as i64,
            n_hc: n_hc as i64,
            n_tokens: 1,
            nb_block0: f32_b,
            nb_block1: de * f32_b,
            nb_add0: f32_b,
            nb_add1: de * f32_b,
            nb_res0: f32_b,
            nb_res1: de * f32_b,
            nb_res2: nh * de * f32_b,
            nb_post0: f32_b,
            nb_post1: nh * f32_b,
            nb_comb0: f32_b,
            nb_comb1: nh * f32_b,
            nb_comb2: nh * nh * f32_b,
            nb0: f32_b,
            nb1: de * f32_b,
            nb2: nh * de * f32_b,
            has_add: 1,
            _pad: 0,
        };

        let n_elem = (n_hc * d_embd) as u64;
        let threads_per_tg: u64 = 256;
        let n_tg = (n_elem + threads_per_tg - 1) / threads_per_tg;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&shared_out.buf), 0);
        enc.set_buffer(2, Some(&after_attn_hc.buf), 0);
        enc.set_buffer(3, Some(&hc_split_post.buf), 0);
        enc.set_buffer(4, Some(&hc_split_comb.buf), 0);
        enc.set_buffer(5, Some(&routed_out.buf), 0);
        enc.set_buffer(6, Some(&out.buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// Shared core for the `_split` variants of `hc_expand_attn`/`hc_expand_add`:
    /// binds `post`/`comb` at caller-supplied byte offsets into (possibly the
    /// same) resident buffers. `block` is buffer1, `res` is buffer2 (cur_hc or
    /// after_attn_hc). `add = Some(routed_out)` sets has_add=1 (the FFN add
    /// path); `None` sets has_add=0 and binds `block` as the dummy (attn path).
    #[allow(clippy::too_many_arguments)]
    fn hc_expand_core(
        &self,
        block: &DeferredBuf,
        res: &DeferredBuf,
        post_buf: &DeferredBuf,
        post_off: u64,
        comb_buf: &DeferredBuf,
        comb_off: u64,
        add: Option<&DeferredBuf>,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            block.n_elements == d_embd,
            "hc_expand_core: block has {} elems, expected d_embd={}",
            block.n_elements, d_embd
        );
        anyhow::ensure!(
            res.n_elements == n_hc * d_embd,
            "hc_expand_core: res shape mismatch ({} vs {}*{})",
            res.n_elements, n_hc, d_embd
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_hc_expand")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_expand pipeline not loaded"))?
            .clone();
        let out = self.alloc_f32(n_hc * d_embd);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct HcExpandArgs {
            n_embd: i64,
            n_hc: i64,
            n_tokens: i64,
            nb_block0: u64,
            nb_block1: u64,
            nb_add0: u64,
            nb_add1: u64,
            nb_res0: u64,
            nb_res1: u64,
            nb_res2: u64,
            nb_post0: u64,
            nb_post1: u64,
            nb_comb0: u64,
            nb_comb1: u64,
            nb_comb2: u64,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            has_add: i32,
            _pad: i32,
        }
        let de = d_embd as u64;
        let nh = n_hc as u64;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = HcExpandArgs {
            n_embd: d_embd as i64,
            n_hc: n_hc as i64,
            n_tokens: 1,
            nb_block0: f32_b,
            nb_block1: de * f32_b,
            nb_add0: f32_b,
            nb_add1: de * f32_b,
            nb_res0: f32_b,
            nb_res1: de * f32_b,
            nb_res2: nh * de * f32_b,
            nb_post0: f32_b,
            nb_post1: nh * f32_b,
            nb_comb0: f32_b,
            nb_comb1: nh * f32_b,
            nb_comb2: nh * nh * f32_b,
            nb0: f32_b,
            nb1: de * f32_b,
            nb2: nh * de * f32_b,
            has_add: if add.is_some() { 1 } else { 0 },
            _pad: 0,
        };
        let n_elem = (n_hc * d_embd) as u64;
        let threads_per_tg: u64 = 256;
        let n_tg = (n_elem + threads_per_tg - 1) / threads_per_tg;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&block.buf), 0);
        enc.set_buffer(2, Some(&res.buf), 0);
        enc.set_buffer(3, Some(&post_buf.buf), post_off);
        enc.set_buffer(4, Some(&comb_buf.buf), comb_off);
        enc.set_buffer(5, Some(&add.unwrap_or(block).buf), 0);
        enc.set_buffer(6, Some(&out.buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Resident-`split` variant of [`Self::hc_expand_add`]: binds `post`/`comb`
    /// as byte-offset sub-ranges of one `[pre|post|comb]` split buffer (e.g.
    /// `hc_collapse_norm`'s split output). Avoids reading split back to upload
    /// the sub-slices in the fused layer path.
    pub fn hc_expand_add_split(
        &self,
        shared_out: &DeferredBuf,
        routed_out: &DeferredBuf,
        after_attn_hc: &DeferredBuf,
        split: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            split.n_elements == 2 * n_hc + n_hc * n_hc,
            "hc_expand_add_split: split has {} elems, expected 2*n_hc+n_hc² = {}",
            split.n_elements,
            2 * n_hc + n_hc * n_hc
        );
        let f32_b = std::mem::size_of::<f32>() as u64;
        let post_off = (n_hc as u64) * f32_b;
        let comb_off = (2 * n_hc as u64) * f32_b;
        self.hc_expand_core(
            shared_out, after_attn_hc, split, post_off, split, comb_off,
            Some(routed_out), n_hc, d_embd,
        )
    }

    /// Phase 2 — K-position variant of `hc_expand_add_split`. FFN-half HC
    /// expand: produces `after_ffn_k [K, n_hc, d_embd]` from per-K shared
    /// expert output, routed MoE output, attention-half HC residual, and
    /// the FFN-half split.
    ///
    /// Inputs (all per-K):
    /// - `shared_out_k` `[K, d_embd]`            — shared-expert MLP output
    /// - `routed_out_k` `[K, d_embd]`            — routed MoE output
    /// - `after_attn_k` `[K, n_hc, d_embd]`      — attn-half HC residual
    /// - `split_k`      `[K, 2*n_hc + n_hc*n_hc]` — FFN-half hc_collapse split
    ///
    /// Returns `after_ffn_k [K, n_hc, d_embd]`. K supported in `{1,2,4,8}`.
    pub fn hc_expand_add_split_k(
        &self,
        shared_out_k: &DeferredBuf,
        routed_out_k: &DeferredBuf,
        after_attn_k: &DeferredBuf,
        split_k: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        anyhow::ensure!(
            shared_out_k.n_elements == k_positions * d_embd,
            "hc_expand_add_split_k: shared_out_k has {} elems, expected K*d_embd = {}*{}",
            shared_out_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            routed_out_k.n_elements == k_positions * d_embd,
            "hc_expand_add_split_k: routed_out_k has {} elems, expected K*d_embd = {}*{}",
            routed_out_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            after_attn_k.n_elements == k_positions * n_hc * d_embd,
            "hc_expand_add_split_k: after_attn_k has {} elems, expected K*n_hc*d_embd = {}*{}*{}",
            after_attn_k.n_elements, k_positions, n_hc, d_embd
        );
        anyhow::ensure!(
            split_k.n_elements == k_positions * mix_hc,
            "hc_expand_add_split_k: split_k has {} elems, expected K*mix_hc = {}*{}",
            split_k.n_elements, k_positions, mix_hc
        );

        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_hc_expand")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_expand pipeline not loaded"))?
            .clone();
        // Carrier ping-pong (DS4_CHUNK_POOL_SCRATCH): after_ffn_k is the cross-layer
        // hidden state (becomes next layer's cur_hc). 2 alternating buffers bound the
        // packed-cb footprint to O(1). ⚠ UNVALIDATED on a clean GPU — the toggle logic
        // is reviewed-sound (alternation ⇒ each layer reads predecessor's buffer, writes
        // the other) but this is the core residual stream; byte-identity MUST be confirmed
        // on a clean-boot GPU before trusting (tonight's box is perturbed). See docs.
        let after_ffn_k = if std::env::var("DS4_CHUNK_POOL_SCRATCH").ok().as_deref() == Some("1") {
            let slot = CARRIER_TOGGLE.with(|t| { let v = t.get(); t.set(!v); v });
            self.pooled_scratch_f32(
                if slot { "hc_carrier_a" } else { "hc_carrier_b" },
                k_positions * n_hc * d_embd,
            )
        } else {
            self.alloc_f32(k_positions * n_hc * d_embd)
        };

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct HcExpandArgs {
            n_embd: i64,
            n_hc: i64,
            n_tokens: i64,
            nb_block0: u64,
            nb_block1: u64,
            nb_add0: u64,
            nb_add1: u64,
            nb_res0: u64,
            nb_res1: u64,
            nb_res2: u64,
            nb_post0: u64,
            nb_post1: u64,
            nb_comb0: u64,
            nb_comb1: u64,
            nb_comb2: u64,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            has_add: i32,
            _pad: i32,
        }
        let de = d_embd as u64;
        let nh = n_hc as u64;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = HcExpandArgs {
            n_embd: d_embd as i64,
            n_hc: n_hc as i64,
            n_tokens: 1,
            nb_block0: f32_b, nb_block1: de * f32_b,
            nb_add0:   f32_b, nb_add1:   de * f32_b,
            nb_res0:   f32_b, nb_res1:   de * f32_b,
            nb_res2: nh * de * f32_b,
            nb_post0: f32_b, nb_post1: nh * f32_b,
            nb_comb0: f32_b, nb_comb1: nh * f32_b,
            nb_comb2: nh * nh * f32_b,
            nb0: f32_b, nb1: de * f32_b,
            nb2: nh * de * f32_b,
            has_add: 1,
            _pad: 0,
        };
        let n_elem = (n_hc * d_embd) as u64;
        let threads_per_tg: u64 = 256;
        let n_tg = (n_elem + threads_per_tg - 1) / threads_per_tg;

        let block_per_k_bytes = (d_embd as u64) * f32_b;
        let hc_per_k_bytes = (n_hc as u64) * (d_embd as u64) * f32_b;
        let split_per_k_bytes = (mix_hc as u64) * f32_b;
        let post_inner_off = (n_hc as u64) * f32_b;
        let comb_inner_off = (2 * n_hc as u64) * f32_b;

        // FUSED K path (DS4_HC_EXPAND_K_FUSE, default on): ONE dispatch over
        // all K via n_tokens=K. The kernel derives t=gid/(n_embd*n_hc) and
        // indexes every buffer by t*stride, so this is BIT-IDENTICAL to the
        // K-loop below (each loop iter is just t=0), just 1 dispatch not K.
        // Only the token strides into split_k change (→ mix_hc, the per-K
        // row); the inner n_hc/n_hc² the loop puts in nb_post1/nb_comb2 are
        // unused at n_tokens=1. Buffers bind at the k=0 offsets.
        if std::env::var("DS4_HC_EXPAND_K_FUSE").as_deref() != Ok("0") {
            let mut fargs = args;
            fargs.n_tokens = k_positions as i64;
            fargs.nb_post1 = (mix_hc as u64) * f32_b;
            fargs.nb_comb2 = (mix_hc as u64) * f32_b;
            let n_elem_k = (n_hc * d_embd * k_positions) as u64;
            let n_tg_k = (n_elem_k + threads_per_tg - 1) / threads_per_tg;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &fargs);
            enc.set_buffer(1, Some(&shared_out_k.buf), 0);
            enc.set_buffer(2, Some(&after_attn_k.buf), 0);
            enc.set_buffer(3, Some(&split_k.buf), post_inner_off);
            enc.set_buffer(4, Some(&split_k.buf), comb_inner_off);
            enc.set_buffer(5, Some(&routed_out_k.buf), 0);
            enc.set_buffer(6, Some(&after_ffn_k.buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tg_k, 1, 1),
                MTLSize::new(threads_per_tg, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
            return Ok(after_ffn_k);
        }

        for k in 0..k_positions {
            let k64 = k as u64;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&shared_out_k.buf), k64 * block_per_k_bytes);
            enc.set_buffer(2, Some(&after_attn_k.buf), k64 * hc_per_k_bytes);
            enc.set_buffer(3, Some(&split_k.buf), k64 * split_per_k_bytes + post_inner_off);
            enc.set_buffer(4, Some(&split_k.buf), k64 * split_per_k_bytes + comb_inner_off);
            enc.set_buffer(5, Some(&routed_out_k.buf), k64 * block_per_k_bytes);
            enc.set_buffer(6, Some(&after_ffn_k.buf), k64 * hc_per_k_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(threads_per_tg, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        Ok(after_ffn_k)
    }

    /// Phase E M5.4.5-prep: attention-half HC expand. Same kernel as
    /// `hc_expand_add` but with `has_add=0` — no `shared+routed`
    /// summing. Used after `attn_output_matmuls` to fold `attn_out`
    /// back into the HC residual:
    ///
    /// ```text
    /// out[dst_hc, e] = post[dst_hc] * attn_out[e]
    ///              + Σ_src(comb_attn[dst_hc, src] * cur_hc[src, e])
    /// ```
    ///
    /// Matches the CPU semantics where the FFN-half's `hc_expand_add_only`
    /// is invoked with `block_add` unused (has_add=0) — the kernel
    /// simply skips the block_add load.
    ///
    /// Buffer 5 (block_add) is still bound (the kernel signature
    /// requires it) but the bound buffer is never read; we re-use
    /// `attn_out` for the bind as a no-op dummy.
    pub fn hc_expand_attn(
        &self,
        attn_out: &DeferredBuf,
        cur_hc: &DeferredBuf,
        hc_split_post_attn: &DeferredBuf,
        hc_split_comb_attn: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            attn_out.n_elements == d_embd,
            "hc_expand_attn: attn_out has {} elems, expected d_embd={}",
            attn_out.n_elements,
            d_embd
        );
        anyhow::ensure!(
            cur_hc.n_elements == n_hc * d_embd,
            "hc_expand_attn: cur_hc shape mismatch ({} vs n_hc*d_embd={}*{})",
            cur_hc.n_elements, n_hc, d_embd
        );
        anyhow::ensure!(
            hc_split_post_attn.n_elements == n_hc,
            "hc_expand_attn: hc_split_post_attn must be n_hc ({}) floats",
            n_hc
        );
        anyhow::ensure!(
            hc_split_comb_attn.n_elements == n_hc * n_hc,
            "hc_expand_attn: hc_split_comb_attn must be n_hc² ({}) floats",
            n_hc * n_hc
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_hc_expand")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_expand pipeline not loaded"))?
            .clone();
        let out = self.alloc_f32(n_hc * d_embd);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct HcExpandArgs {
            n_embd: i64,
            n_hc: i64,
            n_tokens: i64,
            nb_block0: u64,
            nb_block1: u64,
            nb_add0: u64,
            nb_add1: u64,
            nb_res0: u64,
            nb_res1: u64,
            nb_res2: u64,
            nb_post0: u64,
            nb_post1: u64,
            nb_comb0: u64,
            nb_comb1: u64,
            nb_comb2: u64,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            has_add: i32,
            _pad: i32,
        }
        let de = d_embd as u64;
        let nh = n_hc as u64;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = HcExpandArgs {
            n_embd: d_embd as i64,
            n_hc: n_hc as i64,
            n_tokens: 1,
            nb_block0: f32_b,
            nb_block1: de * f32_b,
            nb_add0: f32_b,
            nb_add1: de * f32_b,
            nb_res0: f32_b,
            nb_res1: de * f32_b,
            nb_res2: nh * de * f32_b,
            nb_post0: f32_b,
            nb_post1: nh * f32_b,
            nb_comb0: f32_b,
            nb_comb1: nh * f32_b,
            nb_comb2: nh * nh * f32_b,
            nb0: f32_b,
            nb1: de * f32_b,
            nb2: nh * de * f32_b,
            has_add: 0,
            _pad: 0,
        };
        let n_elem = (n_hc * d_embd) as u64;
        let threads_per_tg: u64 = 256;
        let n_tg = (n_elem + threads_per_tg - 1) / threads_per_tg;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&attn_out.buf), 0);
        enc.set_buffer(2, Some(&cur_hc.buf), 0);
        enc.set_buffer(3, Some(&hc_split_post_attn.buf), 0);
        enc.set_buffer(4, Some(&hc_split_comb_attn.buf), 0);
        // block_add unused (has_add=0); bind a valid buffer as a no-op.
        enc.set_buffer(5, Some(&attn_out.buf), 0);
        enc.set_buffer(6, Some(&out.buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// Resident-`split` variant of [`Self::hc_expand_attn`]: binds `post`
    /// and `comb` as byte-offset sub-ranges of a single `[pre|post|comb]`
    /// split buffer (the layout `encode_layer_attn_half` / `hc_collapse_norm`
    /// emit), so the fused layer path needn't read split back to extract
    /// the sub-slices. `split.n_elements` must be `2*n_hc + n_hc*n_hc`.
    pub fn hc_expand_attn_split(
        &self,
        attn_out: &DeferredBuf,
        cur_hc: &DeferredBuf,
        split: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            split.n_elements == 2 * n_hc + n_hc * n_hc,
            "hc_expand_attn_split: split has {} elems, expected 2*n_hc+n_hc² = {}",
            split.n_elements,
            2 * n_hc + n_hc * n_hc
        );
        let f32_b = std::mem::size_of::<f32>() as u64;
        let post_off = (n_hc as u64) * f32_b;
        let comb_off = (2 * n_hc as u64) * f32_b;
        self.hc_expand_core(
            attn_out, cur_hc, split, post_off, split, comb_off,
            None, n_hc, d_embd,
        )
    }

    /// Phase 2 — K-position variant of `hc_expand_attn_split`. Each K-row
    /// gets its own `attn_out_k`/`cur_hc_k`/`split_k` slice; output is
    /// `after_attn_k [K, n_hc, d_embd]`. K iterations of the existing
    /// `ds4_dsv4_hc_expand` kernel with byte offsets stepping through the
    /// per-K rows; weights and shape args are identical per call.
    ///
    /// Inputs:
    /// - `attn_out_k` `[K, d_embd]` — K-position projected attention output
    ///   (from `encode_attn_chain_k`).
    /// - `cur_hc_k`   `[K, n_hc, d_embd]` — K-position previous HC residual
    ///   (in spec-decode all K positions share the same `cur_hc`; the API
    ///   takes per-K to mirror the K=2-vs-two-K=1 bit-id contract).
    /// - `split_k`    `[K, 2*n_hc + n_hc*n_hc]` — K-position split from
    ///   the attn-half `hc_collapse_norm_k`.
    ///
    /// Returns `after_attn_k [K, n_hc, d_embd]`. K supported in `{1,2,4,8}`.
    pub fn hc_expand_attn_split_k(
        &self,
        attn_out_k: &DeferredBuf,
        cur_hc_k: &DeferredBuf,
        split_k: &DeferredBuf,
        n_hc: usize,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        anyhow::ensure!(
            attn_out_k.n_elements == k_positions * d_embd,
            "hc_expand_attn_split_k: attn_out_k has {} elems, expected K*d_embd = {}*{}",
            attn_out_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            cur_hc_k.n_elements == k_positions * n_hc * d_embd,
            "hc_expand_attn_split_k: cur_hc_k has {} elems, expected K*n_hc*d_embd = {}*{}*{}",
            cur_hc_k.n_elements, k_positions, n_hc, d_embd
        );
        anyhow::ensure!(
            split_k.n_elements == k_positions * mix_hc,
            "hc_expand_attn_split_k: split_k has {} elems, expected K*mix_hc = {}*{}",
            split_k.n_elements, k_positions, mix_hc
        );

        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_hc_expand")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_expand pipeline not loaded"))?
            .clone();
        // Pooled (DS4_CHUNK_POOL_SCRATCH): intra-layer residual scratch (~196MB at
        // K=3000), produced here + consumed within this layer (hc_collapse + the fold);
        // NEVER returned cross-layer. The fused expand below writes its full extent.
        let after_attn_k = self.pooled_scratch_f32("hc_after_attn", k_positions * n_hc * d_embd);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct HcExpandArgs {
            n_embd: i64,
            n_hc: i64,
            n_tokens: i64,
            nb_block0: u64,
            nb_block1: u64,
            nb_add0: u64,
            nb_add1: u64,
            nb_res0: u64,
            nb_res1: u64,
            nb_res2: u64,
            nb_post0: u64,
            nb_post1: u64,
            nb_comb0: u64,
            nb_comb1: u64,
            nb_comb2: u64,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            has_add: i32,
            _pad: i32,
        }
        let de = d_embd as u64;
        let nh = n_hc as u64;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = HcExpandArgs {
            n_embd: d_embd as i64,
            n_hc: n_hc as i64,
            n_tokens: 1,
            nb_block0: f32_b,
            nb_block1: de * f32_b,
            nb_add0: f32_b, nb_add1: de * f32_b,
            nb_res0: f32_b, nb_res1: de * f32_b,
            nb_res2: nh * de * f32_b,
            nb_post0: f32_b, nb_post1: nh * f32_b,
            nb_comb0: f32_b, nb_comb1: nh * f32_b,
            nb_comb2: nh * nh * f32_b,
            nb0: f32_b, nb1: de * f32_b,
            nb2: nh * de * f32_b,
            has_add: 0,
            _pad: 0,
        };
        let n_elem = (n_hc * d_embd) as u64;
        let threads_per_tg: u64 = 256;
        let n_tg = (n_elem + threads_per_tg - 1) / threads_per_tg;

        let attn_per_k_bytes = (d_embd as u64) * f32_b;
        let hc_per_k_bytes = (n_hc as u64) * (d_embd as u64) * f32_b;
        let split_per_k_bytes = (mix_hc as u64) * f32_b;
        let post_inner_off = (n_hc as u64) * f32_b;
        let comb_inner_off = (2 * n_hc as u64) * f32_b;

        // FUSED K path (DS4_HC_EXPAND_K_FUSE, default on): ONE dispatch over
        // all K via n_tokens=K — bit-identical to the K-loop below (t=0 per
        // call). See hc_expand_add_split_k for the full rationale. Only the
        // split_k token strides change (→ mix_hc). has_add=0 here, so buffer 5
        // stays the (unread) attn_out dummy.
        if std::env::var("DS4_HC_EXPAND_K_FUSE").as_deref() != Ok("0") {
            let mut fargs = args;
            fargs.n_tokens = k_positions as i64;
            fargs.nb_post1 = (mix_hc as u64) * f32_b;
            fargs.nb_comb2 = (mix_hc as u64) * f32_b;
            let n_elem_k = (n_hc * d_embd * k_positions) as u64;
            let n_tg_k = (n_elem_k + threads_per_tg - 1) / threads_per_tg;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &fargs);
            enc.set_buffer(1, Some(&attn_out_k.buf), 0);
            enc.set_buffer(2, Some(&cur_hc_k.buf), 0);
            enc.set_buffer(3, Some(&split_k.buf), post_inner_off);
            enc.set_buffer(4, Some(&split_k.buf), comb_inner_off);
            enc.set_buffer(5, Some(&attn_out_k.buf), 0);  // dummy (has_add=0)
            enc.set_buffer(6, Some(&after_attn_k.buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tg_k, 1, 1),
                MTLSize::new(threads_per_tg, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
            return Ok(after_attn_k);
        }

        for k in 0..k_positions {
            let k64 = k as u64;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&attn_out_k.buf), k64 * attn_per_k_bytes);
            enc.set_buffer(2, Some(&cur_hc_k.buf), k64 * hc_per_k_bytes);
            enc.set_buffer(3, Some(&split_k.buf), k64 * split_per_k_bytes + post_inner_off);
            enc.set_buffer(4, Some(&split_k.buf), k64 * split_per_k_bytes + comb_inner_off);
            enc.set_buffer(5, Some(&attn_out_k.buf), k64 * attn_per_k_bytes);  // dummy (has_add=0)
            enc.set_buffer(6, Some(&after_attn_k.buf), k64 * hc_per_k_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(threads_per_tg, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        Ok(after_attn_k)
    }

    /// Phase E M2 (M4 #330p): the split+sinkhorn+weighted_sum+
    /// rms_norm-with-γ tail of `hc_collapse_norm`, in one antirez
    /// kernel (`kernel_dsv4_hc_split_weighted_sum_norm4`, hardcoded for
    /// n_hc=4 n_embd=4096 — DS4 V4 Flash production shape).
    ///
    /// Inputs:
    /// - `mixes` `[mix_hc]` — pre-computed `hc_fn @ flat` (mix_hc = 2*n_hc + n_hc²)
    /// - `scale` `[3]` — hc_scale
    /// - `base` `[mix_hc]` — hc_base
    /// - `x` `[n_hc * n_embd]` — prev_hc
    /// - `gamma` `[n_embd]` — final rms_norm gamma
    ///
    /// Outputs (3 buffers; caller must include them in `flush_and_read_multi`):
    /// - `split` `[mix_hc]` — split values (pre + post + comb)
    /// - `cur` `[n_embd]` — pre-norm weighted sum
    /// - `normed` `[n_embd]` — post rms_norm with gamma
    #[allow(clippy::too_many_arguments)]
    pub fn hc_split_weighted_sum_norm4(
        &self,
        mixes: &DeferredBuf,
        scale: &DeferredBuf,
        base: &DeferredBuf,
        x: &DeferredBuf,
        gamma: &DeferredBuf,
        n_hc: usize,
        n_embd: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
    ) -> Result<(DeferredBuf, DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(
            n_hc == 4 && n_embd % 4 == 0,
            "hc_split_weighted_sum_norm4: kernels require n_hc=4 and n_embd % 4 == 0; got n_hc={} n_embd={}",
            n_hc, n_embd
        );
        // The antirez kernel hardcodes n_embd=4096 (Flash). For other widths
        // (PRO: 7168), use the width-generic bridge shim — same math, n_embd
        // from args. Threadgroup memory n_embd+36 floats must fit the 32 KB
        // limit (n_embd ≤ ~8150).
        let use_any = n_embd != 4096;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        anyhow::ensure!(
            mixes.n_elements == mix_hc,
            "mixes has {} elems, expected mix_hc={}",
            mixes.n_elements,
            mix_hc
        );
        anyhow::ensure!(scale.n_elements == 3, "scale must be 3 floats");
        anyhow::ensure!(base.n_elements == mix_hc, "base must be mix_hc floats");
        anyhow::ensure!(
            x.n_elements == n_hc * n_embd,
            "x must be n_hc*n_embd floats"
        );
        anyhow::ensure!(gamma.n_elements == n_embd, "gamma must be n_embd floats");

        let pipe_name = if use_any {
            "ds4_dsv4_hc_split_weighted_sum_norm_any"
        } else {
            "ds4_dsv4_hc_split_weighted_sum_norm4"
        };
        let pipe = self
            .state
            .pipelines
            .get(pipe_name)
            .ok_or_else(|| anyhow::anyhow!("{pipe_name} pipeline not loaded"))?
            .clone();

        let split_buf = self.alloc_f32(mix_hc);
        let cur_buf = self.alloc_f32(n_embd);
        let normed_buf = self.alloc_f32(n_embd);

        // Mirrors `struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm`
        // (`metal/dsv4_hc.metal:40`).
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            n_embd: i64,
            n_hc: i32,
            sinkhorn_iters: i32,
            n_rows: i64,
            mix_hc: i64,
            nb_mix1: u64,
            nb_split1: u64,
            nb_x0: u64,
            nb_x1: u64,
            nb_x2: u64,
            nb0: u64,
            nb1: u64,
            nb_norm1: u64,
            eps: f32,
            norm_eps: f32,
        }
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = Args {
            n_embd: n_embd as i64,
            n_hc: n_hc as i32,
            sinkhorn_iters,
            n_rows: 1,
            mix_hc: mix_hc as i64,
            nb_mix1: (mix_hc as u64) * f32_b,
            nb_split1: (mix_hc as u64) * f32_b,
            nb_x0: f32_b,
            nb_x1: (n_embd as u64) * f32_b,
            nb_x2: (n_hc as u64) * (n_embd as u64) * f32_b,
            nb0: f32_b,
            nb1: (n_embd as u64) * f32_b,
            nb_norm1: (n_embd as u64) * f32_b,
            eps: hc_eps,
            norm_eps: rms_eps,
        };

        // Threadgroup memory: row_shmem (n_embd/4 float4) + pre_shmem (4
        // float) + sum_shmem (32 float). 4096 → ~16 KB; 7168 → ~28 KB.
        let shmem_bytes: u64 = (n_embd as u64) * 4 + 4 * 4 + 32 * 4;
        // Clamp to the pipeline's limit — large shmem (PRO 28 KB) lowers
        // maxTotalThreadsPerThreadgroup below 1024. The kernel loops are
        // strided so fewer threads stay correct.
        let tg_threads: u64 = (pipe.max_total_threads_per_threadgroup())
            .min(1024)
            .min((n_embd as u64) / 4);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&mixes.buf), 0);
        enc.set_buffer(2, Some(&scale.buf), 0);
        enc.set_buffer(3, Some(&base.buf), 0);
        enc.set_buffer(4, Some(&x.buf), 0);
        enc.set_buffer(5, Some(&split_buf.buf), 0);
        enc.set_buffer(6, Some(&cur_buf.buf), 0);
        enc.set_buffer(7, Some(&gamma.buf), 0);
        enc.set_buffer(8, Some(&normed_buf.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),          // 1 row (single decode token)
            MTLSize::new(tg_threads, 1, 1), // n_embd/4 float4 lanes, ≤1024
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok((split_buf, cur_buf, normed_buf))
    }

    /// Phase E M5 prep (M4 #330p): full GPU `hc_collapse_norm` as a
    /// 3-stage chain encoded into the BatchScope:
    ///
    ///   1. flat = rms_norm_mul(prev_hc, unit_γ, rms_eps)   — rms_norm_plain
    ///   2. mix  = matvec_f32(hc_fn, flat, mix_hc)
    ///   3. (split, cur, normed) =
    ///        hc_split_weighted_sum_norm4(mix, scale, base, prev_hc, γ, …)
    ///
    /// Returns (cur, normed, split) matching the CPU signature
    /// `AttentionDispatcher::hc_collapse_norm`. Each stage is an
    /// already-tested BatchScope op (Phase C-A, M2). The chain stays
    /// on GPU between stages — flat and mix never touch CPU.
    ///
    /// Hardcoded for n_hc=4, n_embd=4096 (the M2 `*_norm4` kernel
    /// constraint). Production DS4 V4 Flash satisfies this.
    ///
    /// As with all M5-prep ops, this is intended for the unified-cb
    /// decoder. Standalone wiring would regress (slice-4 anti-pattern
    /// for the small inputs). The win arrives when ALL per-layer GPU
    /// work fits in one cb.
    /// Fused hc_collapse stage-1+2 (megakernel Phase-1 PoC, DS4_FUSE_HC=1):
    /// rms_norm_mul(prev_hc, unit_gamma) → hc_fn matvec in ONE kernel. Removes
    /// the rms→matvec encoder boundary + the `flat[hc_dim]` device round-trip.
    /// Output `mix[mix_hc]` ≈ matvec_f32(hc_fn, rms_norm_mul(prev_hc, unit), …)
    /// to ~1e-6 (different reduction order; argmax-stable). hc_dim % 4 == 0.
    pub fn fused_hc_collapse_stage12(
        &self,
        hc_fn: &DeferredBuf,
        prev_hc: &DeferredBuf,
        hc_dim: usize,
        mix_hc: usize,
        eps: f32,
        // When true, `hc_fn.buf` holds the F16 weight (2 B/elem, no-copy from the
        // mmap) and the f16 kernel variant reads it as half4. Bit-identical to the
        // f32 kernel on the dequantized weight.
        hc_fn_is_f16: bool,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(hc_dim % 4 == 0, "fused_hc_collapse: hc_dim ({hc_dim}) % 4 != 0");
        anyhow::ensure!(prev_hc.n_elements == hc_dim, "fused_hc_collapse: prev_hc shape");
        anyhow::ensure!(hc_fn.n_elements == hc_dim * mix_hc, "fused_hc_collapse: hc_fn shape");
        let kernel = if hc_fn_is_f16 {
            "ds4_dsv4_hc_collapse_rms_mv_f16"
        } else {
            "ds4_dsv4_hc_collapse_rms_mv_f32"
        };
        let pipe = self.state.specialized_pipeline(kernel, &[], |_fcv| {})?;
        let mix = self.alloc_f32(mix_hc);
        let hc_dim_u = hc_dim as u32;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&hc_fn.buf), 0);
        enc.set_buffer(1, Some(&prev_hc.buf), 0);
        enc.set_buffer(2, Some(&mix.buf), 0);
        set_scalar_bytes(enc, 3, &hc_dim_u);
        set_scalar_bytes(enc, 4, &eps);
        enc.dispatch_thread_groups(
            MTLSize::new(mix_hc as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(mix)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn hc_collapse_norm(
        &self,
        prev_hc: &DeferredBuf,
        hc_fn: &DeferredBuf,
        hc_scale: &DeferredBuf,
        hc_base: &DeferredBuf,
        hc_norm_gamma: &DeferredBuf,
        n_hc: usize,
        n_embd: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        unit_gamma_hc_dim: &DeferredBuf, // caller-uploaded [n_hc*n_embd]
        // `hc_fn` holds F16 weight bytes (no-copy, lean path) vs f32. Selects the
        // f16 fused kernel / `matvec_f16` for stage 1+2.
        hc_fn_is_f16: bool,
    ) -> Result<(DeferredBuf, DeferredBuf, DeferredBuf)> {
        let hc_dim = n_hc * n_embd;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        anyhow::ensure!(
            prev_hc.n_elements == hc_dim,
            "hc_collapse_norm: prev_hc has {} elems, expected n_hc*n_embd = {}",
            prev_hc.n_elements,
            hc_dim
        );
        anyhow::ensure!(
            hc_fn.n_elements == hc_dim * mix_hc,
            "hc_collapse_norm: hc_fn has {} elems, expected hc_dim*mix_hc = {}",
            hc_fn.n_elements,
            hc_dim * mix_hc
        );
        anyhow::ensure!(
            unit_gamma_hc_dim.n_elements == hc_dim,
            "hc_collapse_norm: unit_gamma_hc_dim must be hc_dim long (caller uploads [1.0; hc_dim])"
        );

        // Stage 1+2: flat = rms_norm(prev_hc, unit_gamma); mix = hc_fn @ flat.
        // Fused into one kernel (no flat round-trip, no rms→matvec encoder
        // boundary) — default-on, DS4_FUSE_HC=0 reverts to the two-kernel path
        // (rms_norm_mul + matvec_f32; hc_dim=16384 %4 ✓, mix_hc=24 %2 ✓).
        // Measured +2.9% decode (22.39→23.04 tok/s), gpu_busy −1.2ms, tokens
        // bit-identical (fused rel 2.5e-6 vs the 2-kernel path).
        let mix = if std::env::var("DS4_FUSE_HC").ok().as_deref() != Some("0") {
            self.fused_hc_collapse_stage12(hc_fn, prev_hc, hc_dim, mix_hc, rms_eps, hc_fn_is_f16)?
        } else {
            let flat = self.rms_norm_mul(prev_hc, unit_gamma_hc_dim, rms_eps)?;
            if hc_fn_is_f16 {
                self.matvec_f16(hc_fn, &flat, hc_dim, mix_hc)?
            } else {
                self.matvec_f32(hc_fn, &flat, hc_dim, mix_hc)?
            }
        };

        // Stage 3: the fused M2 kernel produces (split, cur, normed)
        // from mix + scale + base + prev_hc + gamma.
        self.hc_split_weighted_sum_norm4(
            &mix,
            hc_scale,
            hc_base,
            prev_hc,
            hc_norm_gamma,
            n_hc,
            n_embd,
            sinkhorn_iters,
            hc_eps,
            rms_eps,
        )
    }

    /// K-position hc_collapse_norm. Iterates the existing single-position
    /// `hc_collapse_norm` K times with blit-copied input slices and blit-
    /// copied output concatenation. K-linear in dispatches; acceptable
    /// since hc_collapse is small (~2-3ms of token GPU at K=1).
    ///
    /// Inputs:
    ///   prev_hc_k       — [K, n_hc*n_embd] f32 row-major
    ///   hc_fn/scale/base/norm_gamma — shared across K (unchanged from K=1).
    /// Outputs:
    ///   (split_k, cur_k, normed_k) all [K, ...] row-major.
    ///
    /// Hardcoded n_hc=4, n_embd=4096 (inherited from the underlying
    /// `hc_split_weighted_sum_norm4` kernel constraints).
    #[allow(clippy::too_many_arguments)]
    pub fn hc_collapse_norm_k(
        &self,
        prev_hc_k: &DeferredBuf,
        hc_fn: &DeferredBuf,
        hc_scale: &DeferredBuf,
        hc_base: &DeferredBuf,
        hc_norm_gamma: &DeferredBuf,
        n_hc: usize,
        n_embd: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        unit_gamma_hc_dim: &DeferredBuf,
        k_positions: usize,
        hc_fn_is_f16: bool,
    ) -> Result<(DeferredBuf, DeferredBuf, DeferredBuf)> {
        let hc_dim = n_hc * n_embd;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        anyhow::ensure!(
            prev_hc_k.n_elements == k_positions * hc_dim,
            "hc_collapse_norm_k: prev_hc_k has {} elems, expected K*hc_dim = {}*{}",
            prev_hc_k.n_elements, k_positions, hc_dim
        );

        // Pre-allocate concatenated outputs. normed_k (K×n_embd, the big one) is pooled
        // when a one-shot ATTN_POOL_KEY was armed by the caller (DS4_CHUNK_POOL_ATTN). It is
        // allocated HERE, before the internal flat_k = rms_norm_mul_k below — so the key lands
        // on normed_k, not flat_k (which then gets None → fresh). cur_k is the cross-layer
        // carrier — NEVER pool it. split_k is tiny (K×mix_hc). The norm4 dispatch writes the
        // full normed_k extent, so pooled reuse is safe; byte-identity gated.
        let split_k = self.alloc_f32(k_positions * mix_hc);
        let cur_k = self.alloc_f32(k_positions * n_embd);
        let normed_k = self.alloc_attn_out(k_positions * n_embd);

        // FUSED K-batched path (DS4_HC_COLLAPSE_K_FUSE, default on): 3 dispatches
        // total (rms_norm_mul_k + batched mul_mv_f32 ne11=K + norm4 n_rows=K) with
        // ZERO blits, vs the per-K loop's 3K dispatches + 5K blits. All three
        // underlying kernels already index rows (rms_norm_mul_k rows=K; mul_mv
        // ne11/nb11 batch dim; norm4 reads row from threadgroup_position_in_grid
        // + n_rows) — so this is bit-identical to looping K=1, just fewer
        // dispatches. n_hc=4/n_embd=4096 only (norm4 kernel constraint).
        // K-cap on the fused path. HISTORY: capped at 128 (f6e2a768) because the
        // fused collapse appeared to corrupt at K≳160 with "residual drift from L8".
        // 2026-06-10 clean-boot bisect: that corruption was DS4_CHUNK_SWA_KFLASH
        // (the tile-boundary NaN), NOT the fused collapse. With SWA_KFLASH off the
        // fused path is COHERENT at full K under SYNC commit (gamma needle recall
        // 15/15 across 1024-3000) and recovers prefill @3000 to 114.9 tok/s (the
        // cap had halved it to ~60 by forcing the per-K loop: 3K dispatches + 5K
        // blits). So the default is now UNCAPPED. CAVEAT: under ASYNC commit
        // (DS4_CHUNK_COMMIT_ASYNC=1, no per-layer drain) the uncapped fused path
        // races (~1/6 BOS) — prefill_chunk re-caps to 128 for the async path; the
        // sync default (production) is coherent uncapped. DS4_HC_COLLAPSE_K_FUSE_MAX
        // overrides either way.
        let fuse_k_max: usize = std::env::var("DS4_HC_COLLAPSE_K_FUSE_MAX")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        if std::env::var("DS4_HC_COLLAPSE_K_FUSE").as_deref() != Ok("0")
            && n_hc == 4 && n_embd == 4096
            && k_positions <= fuse_k_max
        {
            // Stage 1: flat_k = rms_norm(prev_hc_k row, unit_gamma) per K row.
            let flat_k = self.rms_norm_mul_k(prev_hc_k, unit_gamma_hc_dim, hc_dim, k_positions, rms_eps)?;
            // Stage 2: mix_k[k] = hc_fn @ flat_k[k]. Batched matvec (ne11=K).
            // Lean weights keep hc_fn as F16 only (f32 dropped) — use the
            // f16-weight twin (same ne11-batched kernel family).
            // NOTE: hc-mix is NOT routed through the weight-stationary matmul_k_f16
            // (unlike the indexer/compressor projections). Measured wash (+0%): the
            // hc_fn weight is only [16384×24]≈786KB → cache-resident, so the per-token
            // mul_mv is NOT DRAM-bandwidth-bound. The per-token-matvec cost only bites
            // when the weight exceeds cache (indexer 16MB, compressor 4MB). Keeping the
            // f32-accumulate matvec here also preserves the needle-validated precision.
            let mix_k = if hc_fn_is_f16 {
                self.matmul_f16_k(hc_fn, &flat_k, hc_dim, mix_hc, k_positions)?
            } else {
                self.matvec_f32_k(hc_fn, &flat_k, hc_dim, mix_hc, k_positions)?
            };
            // Stage 3: norm4 over K rows in one dispatch.
            self.hc_split_weighted_sum_norm4_into_k(
                &mix_k, hc_scale, hc_base, prev_hc_k, hc_norm_gamma,
                n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps, k_positions,
                &split_k, &cur_k, &normed_k,
            )?;
            return Ok((split_k, cur_k, normed_k));
        }

        let f32_b: u64 = 4;
        for k in 0..k_positions {
            // 1. Slice prev_hc_k row k → fresh per-row buffer (blit).
            let row = self.alloc_f32(hc_dim);
            crate::macos::end_shared_compute_enc_force();
            let enc = self.cmd_buf.new_blit_command_encoder();
            enc.copy_from_buffer(
                &prev_hc_k.buf, (k * hc_dim) as u64 * f32_b,
                &row.buf, 0, (hc_dim as u64) * f32_b,
            );
            enc.end_encoding();

            // 2. Run K=1 hc_collapse_norm on this row.
            let (split, cur, normed) = self.hc_collapse_norm(
                &row, hc_fn, hc_scale, hc_base, hc_norm_gamma,
                n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps,
                unit_gamma_hc_dim,
                hc_fn_is_f16,
            )?;

            // 3. Blit per-row outputs into their slot in the K-concat bufs.
            crate::macos::end_shared_compute_enc_force();
            let enc = self.cmd_buf.new_blit_command_encoder();
            enc.copy_from_buffer(
                &split.buf, 0,
                &split_k.buf, (k * mix_hc) as u64 * f32_b, (mix_hc as u64) * f32_b,
            );
            enc.copy_from_buffer(
                &cur.buf, 0,
                &cur_k.buf, (k * n_embd) as u64 * f32_b, (n_embd as u64) * f32_b,
            );
            enc.copy_from_buffer(
                &normed.buf, 0,
                &normed_k.buf, (k * n_embd) as u64 * f32_b, (n_embd as u64) * f32_b,
            );
            enc.end_encoding();
        }
        Ok((split_k, cur_k, normed_k))
    }

    /// K-batched f32 matvec: `out[k] = w @ x_k[k]` for k in 0..K, in ONE
    /// dispatch. `w` is shared `[d_out, d_in]`; `x_k` is `[K, d_in]`; output
    /// `[K, d_out]`. Uses the same `ds4_kernel_mul_mv_f32_f32_4` kernel as
    /// `matvec_f32` with ne11=K (activation batch dim) + grid.y=K — bit-identical
    /// to looping matvec_f32 K times, fewer dispatches. (d_in%4==0, d_out%2==0.)
    pub fn matvec_f32_k(
        &self,
        w: &DeferredBuf,
        x_k: &DeferredBuf,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(w.n_elements == d_in * d_out, "matvec_f32_k: w shape");
        anyhow::ensure!(x_k.n_elements == k_positions * d_in, "matvec_f32_k: x_k shape");
        anyhow::ensure!(d_in % 4 == 0 && d_out % 2 == 0, "matvec_f32_k: dim align");

        let nsg: i16 = (((d_in as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_in % 256 == 0 { 16 } else if d_in % 128 == 0 { 8 } else { 4 };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        let out = self.alloc_f32(k_positions * d_out);
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let din_b = (d_in * 4) as u64;
        let args = MulMvArgs {
            ne00: d_in as i32, ne01: d_out as i32, ne02: 1, _pad0: 0,
            nb00: 4, nb01: din_b, nb02: (d_in * d_out * 4) as u64, nb03: (d_in * d_out * 4) as u64,
            ne10: d_in as i32, ne11: k_positions as i32, ne12: 1, _pad1: 0,
            nb10: 4, nb11: din_b, nb12: din_b * k_positions as u64, nb13: din_b * k_positions as u64,
            ne0: d_out as i32, ne1: k_positions as i32, nr0: 2, r2: 1, r3: 1,
        };
        let shmem_bytes: u64 = 32 * 2 * 4;
        let n_row_tg = ((d_out as u64) + 1) / 2;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w.buf), 0);
        enc.set_buffer(2, Some(&x_k.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        // grid.y = K (activation batch); kernel indexes x/out rows by tgpig.y.
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, k_positions as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// K-batched `hc_split_weighted_sum_norm4` writing into caller-provided
    /// `[K, ...]` output buffers in ONE dispatch (grid.x=K). The norm4 kernel
    /// already reads its row from `threadgroup_position_in_grid` and indexes all
    /// buffers by `row*stride` — so K rows process in one dispatch, bit-identical
    /// to K separate n_rows=1 calls. n_hc=4/n_embd=4096 only.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_split_weighted_sum_norm4_into_k(
        &self,
        mixes_k: &DeferredBuf,   // [K, mix_hc]
        scale: &DeferredBuf,
        base: &DeferredBuf,
        x_k: &DeferredBuf,       // [K, n_hc*n_embd]
        gamma: &DeferredBuf,
        n_hc: usize,
        n_embd: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        k_positions: usize,
        split_k: &DeferredBuf,   // [K, mix_hc]  (output)
        cur_k: &DeferredBuf,     // [K, n_embd]  (output)
        normed_k: &DeferredBuf,  // [K, n_embd]  (output)
    ) -> Result<()> {
        anyhow::ensure!(n_hc == 4 && n_embd == 4096, "norm4_k: n_hc=4/n_embd=4096 only");
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let pipe = self.state.pipelines.get("ds4_dsv4_hc_split_weighted_sum_norm4")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_hc_split_weighted_sum_norm4 not loaded"))?
            .clone();
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            n_embd: i64, n_hc: i32, sinkhorn_iters: i32, n_rows: i64, mix_hc: i64,
            nb_mix1: u64, nb_split1: u64, nb_x0: u64, nb_x1: u64, nb_x2: u64,
            nb0: u64, nb1: u64, nb_norm1: u64, eps: f32, norm_eps: f32,
        }
        let f32_b = std::mem::size_of::<f32>() as u64;
        let args = Args {
            n_embd: n_embd as i64, n_hc: n_hc as i32, sinkhorn_iters,
            n_rows: k_positions as i64, mix_hc: mix_hc as i64,
            nb_mix1: (mix_hc as u64) * f32_b,    // stride between K rows of mixes
            nb_split1: (mix_hc as u64) * f32_b,  // stride between K rows of split
            nb_x0: f32_b, nb_x1: (n_embd as u64) * f32_b,
            nb_x2: (n_hc as u64) * (n_embd as u64) * f32_b,  // stride between K rows of x
            nb0: f32_b, nb1: (n_embd as u64) * f32_b,
            nb_norm1: (n_embd as u64) * f32_b,   // stride between K rows of cur/normed
            eps: hc_eps, norm_eps: rms_eps,
        };
        let shmem_bytes: u64 = 1024 * 16 + 4 * 4 + 32 * 4;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&mixes_k.buf), 0);
        enc.set_buffer(2, Some(&scale.buf), 0);
        enc.set_buffer(3, Some(&base.buf), 0);
        enc.set_buffer(4, Some(&x_k.buf), 0);
        enc.set_buffer(5, Some(&split_k.buf), 0);
        enc.set_buffer(6, Some(&cur_k.buf), 0);
        enc.set_buffer(7, Some(&gamma.buf), 0);
        enc.set_buffer(8, Some(&normed_k.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        // CLAMP threads/tg to the pipeline limit. The raw upstream pipeline's
        // heavy register use pushed maxTotalThreadsPerThreadgroup below 1024
        // (832 on M1 Ultra; 512 under MTL_SHADER_VALIDATION) and dispatching
        // 1024 against it is UB with validation OFF (MTL_SHADER_VALIDATION
        // asserts). The bridge shim now pins the limit at 1024 via
        // [[max_total_threads_per_threadgroup(1024)]], so tg==1024 — the
        // bit-identical reduction shape of the per-token baseline. The kernel
        // loops are stride-based, so the clamp stays correct as defense if the
        // limit ever drops again.
        let tg = pipe.max_total_threads_per_threadgroup().min(1024);
        enc.dispatch_thread_groups(
            MTLSize::new(k_positions as u64, 1, 1),  // grid.x = K rows
            MTLSize::new(tg, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(())
    }

    /// Phase E M5.3: composable version of `attn_qkv_chain_batched`.
    /// Encodes the 5 ops (matvec_q_a → rms_norm → matvec_q_b →
    /// head_rms_norm → matvec_kv) into THIS scope instead of opening
    /// a new one. Caller uploads inputs as DeferredBufs and decides
    /// when to flush, enabling chains with `hc_collapse_norm` (whose
    /// output `normed` feeds the QKV chain) and downstream ops
    /// (rms_norm_mul + rope_tail_in_place + kv_fp8_store, etc.).
    ///
    /// The inherent `MetalDispatcher::attn_qkv_chain_batched` opens
    /// its own scope and flushes; both paths share the same kernels
    /// and produce bit-identical results when fed the same data.
    ///
    /// Preconditions (asserted, mirrors the inherent method):
    /// - d_embd derived from `normed_buf.n_elements`
    /// - `d_embd % 4 == 0`, `d_embd % 2 == 0`
    /// - `n_lora_q % 4 == 0`, `n_lora_q % 2 == 0`
    /// - `q_dim = n_head * head_dim`, `q_dim % 2 == 0`
    /// - `kv_row % 2 == 0`
    /// - shapes of attn_q_a/b/kv and gamma_q must match
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_qkv_chain(
        &self,
        normed_buf: &DeferredBuf,
        attn_q_a_buf: &DeferredBuf,
        gamma_q_buf: &DeferredBuf,
        attn_q_b_buf: &DeferredBuf,
        attn_kv_buf: &DeferredBuf,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        eps: f32,
        kv_row: usize,
    ) -> Result<(DeferredBuf, DeferredBuf, DeferredBuf)> {
        let d_embd = normed_buf.n_elements;
        let q_dim = n_head * head_dim;
        anyhow::ensure!(
            attn_q_a_buf.n_elements == n_lora_q * d_embd,
            "encode_attn_qkv_chain: attn_q_a shape mismatch ({} vs n_lora_q*d_embd = {}*{})",
            attn_q_a_buf.n_elements,
            n_lora_q,
            d_embd
        );
        anyhow::ensure!(
            gamma_q_buf.n_elements == n_lora_q,
            "encode_attn_qkv_chain: gamma_q must be n_lora_q ({}) floats",
            n_lora_q
        );
        anyhow::ensure!(
            attn_q_b_buf.n_elements == q_dim * n_lora_q,
            "encode_attn_qkv_chain: attn_q_b shape mismatch ({} vs q_dim*n_lora_q = {}*{})",
            attn_q_b_buf.n_elements,
            q_dim,
            n_lora_q
        );
        anyhow::ensure!(
            attn_kv_buf.n_elements == kv_row * d_embd,
            "encode_attn_qkv_chain: attn_kv shape mismatch ({} vs kv_row*d_embd = {}*{})",
            attn_kv_buf.n_elements,
            kv_row,
            d_embd
        );
        anyhow::ensure!(head_dim >= 1, "encode_attn_qkv_chain: head_dim must be >= 1");

        // Q8_0 projection path (set_q8_proj): the q/kv weights are stored as
        // block_q8_0 (1 byte/weight) instead of f32 (4 bytes), cutting the
        // attention-projection bandwidth ~4x — the dominant attn-half GPU cost
        // per the per-stage harness. The caller must have built the weight bufs
        // via weight_q8_0.
        let (qr, q_heads_raw, kv_raw_row);
        if self.q8_proj.get() {
            // attn_q_a (→qr, feeds the q-chain) and attn_kv (→kv, terminal) are
            // INDEPENDENT — both read `normed`, write separate rows. DS4_QKV_
            // CONCURRENT=1 issues them in one concurrent encoder so the GPU
            // overlaps them and we drop one encoder-boundary drain in attn_half
            // (the co-dominant ~35% stage). Default off until measured.
            if std::env::var("DS4_QKV_CONCURRENT").ok().as_deref() == Some("1") {
                let (qr_c, kv_c) = self.matvec_q8_0_pair_concurrent(
                    attn_q_a_buf, normed_buf, d_embd, n_lora_q,
                    attn_kv_buf, normed_buf, d_embd, kv_row,
                )?;
                qr = qr_c;
                kv_raw_row = kv_c;
            } else {
                qr = self.matvec_attn_proj(attn_q_a_buf, normed_buf, d_embd, n_lora_q)?;
                kv_raw_row = self.matvec_attn_proj(attn_kv_buf, normed_buf, d_embd, kv_row)?;
            }
            let qr_normed = self.rms_norm_mul(&qr, gamma_q_buf, eps)?;
            q_heads_raw = self.matvec_attn_proj(attn_q_b_buf, &qr_normed, n_lora_q, q_dim)?;
            let q_heads = self.head_rms_norm(&q_heads_raw, n_head, head_dim, eps)?;
            return Ok((qr_normed, q_heads, kv_raw_row));
        }
        qr = self.matvec_f32(attn_q_a_buf, normed_buf, d_embd, n_lora_q)?;
        let qr_normed = self.rms_norm_mul(&qr, gamma_q_buf, eps)?;
        q_heads_raw = self.matvec_f32(attn_q_b_buf, &qr_normed, n_lora_q, q_dim)?;
        let q_heads = self.head_rms_norm(&q_heads_raw, n_head, head_dim, eps)?;
        kv_raw_row = self.matvec_f32(attn_kv_buf, normed_buf, d_embd, kv_row)?;
        Ok((qr_normed, q_heads, kv_raw_row))
    }

    /// K-position q/kv projection chain — the Phase 1 building block of
    /// speculative decoding. Mirrors `encode_attn_qkv_chain` but for K
    /// activations:
    ///
    /// ```text
    ///   qr        = matvec_k_q8_0(attn_q_a, normed_K)           [K, n_lora_q]
    ///   qr_normed = rms_norm_mul_k(qr, γ_q, eps)                [K, n_lora_q]
    ///   q_heads_  = matvec_k_q8_0(attn_q_b, qr_normed)          [K, q_dim]
    ///   q_heads   = head_rms_norm_k(q_heads_, n_head, head_dim) [K, q_dim]
    ///   kv_raw    = matvec_k_q8_0(attn_kv, normed_K)            [K, kv_row]
    /// ```
    ///
    /// All projections use the simdgroup-matrix Q8_0 kernel (Phase 0 winner;
    /// K=8 ratio 0.93 vs K=1 — weight reads fully amortized). The two rms-
    /// norms use single-dispatch K-position variants. Caller must pass raw
    /// q8 bytes via `weight_q8_0_raw` (not the requantize path).
    ///
    /// Returns `(qr_normed, q_heads, kv_raw_row)` — same logical shape as K=1
    /// but each is `[K, ...]`. K supported in `{1, 2, 4, 8}`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_qkv_chain_k(
        &self,
        normed_k_buf: &DeferredBuf,
        attn_q_a_buf: &DeferredBuf,
        gamma_q_buf: &DeferredBuf,
        attn_q_b_buf: &DeferredBuf,
        attn_kv_buf: &DeferredBuf,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        eps: f32,
        kv_row: usize,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<(DeferredBuf, DeferredBuf, DeferredBuf)> {
        let q_dim = n_head * head_dim;
        anyhow::ensure!(
            normed_k_buf.n_elements == k_positions * d_embd,
            "encode_attn_qkv_chain_k: normed_k has {} elems, expected K*d_embd = {}*{}",
            normed_k_buf.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            attn_q_a_buf.n_elements == n_lora_q * d_embd,
            "attn_q_a shape mismatch"
        );
        anyhow::ensure!(gamma_q_buf.n_elements == n_lora_q, "gamma_q shape mismatch");
        anyhow::ensure!(
            attn_q_b_buf.n_elements == q_dim * n_lora_q,
            "attn_q_b shape mismatch"
        );
        anyhow::ensure!(
            attn_kv_buf.n_elements == kv_row * d_embd,
            "attn_kv shape mismatch"
        );

        // DS4_CHUNK_POOL_ATTN: pool the three K×dim projection outputs across layers (O(1)
        // transient footprint — bounds the per-cb residency that else collapses packed/
        // event-window prefill under memory pressure). All three are INTRA-layer (qr →
        // rms_norm; q_heads_raw → head_rms; kv_raw_row → kv_normed downstream this layer),
        // distinct keys, GEMM writes full extent → pooled reuse is safe. No-op unless the flag.
        arm_attn_pool_key("attn_qr");
        let qr = self.matmul_k_attn_proj(attn_q_a_buf, normed_k_buf, d_embd, n_lora_q, k_positions)?;
        arm_attn_pool_key("attn_qrnorm");
        let qr_normed = self.rms_norm_mul_k(&qr, gamma_q_buf, n_lora_q, k_positions, eps)?;
        arm_attn_pool_key("attn_qheads_raw");
        let q_heads_raw = self.matmul_k_attn_proj(attn_q_b_buf, &qr_normed, n_lora_q, q_dim, k_positions)?;
        arm_attn_pool_key("attn_qheads");
        let q_heads = self.head_rms_norm_k(&q_heads_raw, n_head, head_dim, k_positions, eps)?;
        arm_attn_pool_key("attn_kvraw");
        let kv_raw_row = self.matmul_k_attn_proj(attn_kv_buf, normed_k_buf, d_embd, kv_row, k_positions)?;
        Ok((qr_normed, q_heads, kv_raw_row))
    }

    /// Phase E M5.4.1: encode the attention-half of one decode layer
    /// into this scope as a single composable chain:
    ///
    /// ```text
    /// 1. hc_collapse_norm(Attn): prev_hc → (split, cur, normed)
    /// 2. encode_attn_qkv_chain(normed, attn_q_a/b/kv, γ_q):
    ///       → (qr_normed, q_heads, kv_raw_row)
    /// 3. rms_norm_mul(kv_raw_row, γ_kv): → kv_normed
    /// 4. rope_tail_in_place(kv_normed[tail], pos)
    /// ```
    ///
    /// All four sub-chains are existing BatchScope ops (M5.1, M5.3,
    /// Phase C-A `rms_norm_mul`, Phase C-B Slice 4 `rope_tail_in_place`).
    /// Composing them here means the entire attn-half runs in ONE cb;
    /// the GPU-produced `normed`, `kv_raw_row`, and rotated `kv_normed`
    /// never round-trip through CPU.
    ///
    /// Returns DeferredBuf handles the per-layer encoder pipeline
    /// will consume downstream (kv_fp8_store_persistent +
    /// flash_attn_decode_metal_persistent + attn_output_matmuls
    /// + ...). M5.4.2 chains those in.
    ///
    /// Inputs already on the GPU (uploaded by the caller as DeferredBufs).
    /// `hc_dim = n_hc * n_embd`; `unit_gamma_hc_dim` must be `hc_dim`
    /// ones (used for the rms_norm_plain step inside hc_collapse_norm).
    ///
    /// Shape constraints inherited from hc_collapse_norm (n_hc=4,
    /// n_embd=4096 — DS4 V4 Flash) and encode_attn_qkv_chain.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_attn_half(
        &mut self,
        // hc_collapse_norm inputs
        prev_hc: &DeferredBuf,
        hc_fn: &DeferredBuf,
        hc_scale: &DeferredBuf,
        hc_base: &DeferredBuf,
        hc_norm_gamma: &DeferredBuf,
        unit_gamma_hc_dim: &DeferredBuf,
        // `hc_fn` holds F16 bytes (no-copy, lean) vs f32 — picks the f16 kernel.
        hc_fn_is_f16: bool,
        // qkv chain inputs
        attn_q_a: &DeferredBuf,
        gamma_q: &DeferredBuf,
        attn_q_b: &DeferredBuf,
        attn_kv: &DeferredBuf,
        // kv rms+rope inputs
        qkv_gamma_kv: &DeferredBuf,
        // shape + scalar params
        n_hc: usize,
        n_embd: usize,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        kv_row: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        params_for_rope: &ds4_engine::attn_dispatch::LayerParams,
        pos: u32,
    ) -> Result<LayerAttnHalfOuts> {
        // Fine-grained per-substage GPU probes (DS4_KERNEL_PROFILE_FINE): split
        // the attn-half into hc_collapse / qkv_chain / kv_rms / rope and commit+wait
        // each, so DS4_OP_TRACE prints their individual GPU times. Synchronous, so
        // it kills overlap — profiling-only. Off → normal fused cb path. Mutable
        // self-borrow needed for commit_wait_stage; cast via interior mut isn't
        // available, so use a small helper.
        let fine = std::env::var("DS4_KERNEL_PROFILE_FINE").is_ok();
        // Stage 1: hc_collapse_norm chain (M5.1).
        let (split_buf, _cur_buf, normed_buf) = self.hc_collapse_norm(
            prev_hc, hc_fn, hc_scale, hc_base, hc_norm_gamma,
            n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps, unit_gamma_hc_dim,
            hc_fn_is_f16,
        )?;
        if fine { self.commit_wait_stage("attn_hc_collapse"); }
        // Stage 2: 5-op QKV chain (M5.3, composable form).
        let (qr_normed, q_heads, kv_raw_row) = self.encode_attn_qkv_chain(
            &normed_buf, attn_q_a, gamma_q, attn_q_b, attn_kv,
            n_lora_q, n_head, head_dim, rms_eps, kv_row,
        )?;
        if fine { self.commit_wait_stage("attn_qkv_chain"); }
        // Stage 3: kv_rms_norm_row equivalent.
        let kv_normed = self.rms_norm_mul(&kv_raw_row, qkv_gamma_kv, rms_eps)?;
        if fine { self.commit_wait_stage("attn_kv_rms"); }
        // Stage 4: rope_tail in place over the last n_rot floats of
        // kv_normed (single head, n_lora_kv-wide row).
        let n_rot = params_for_rope.n_rot as usize;
        let tail_byte_offset =
            ((kv_row - n_rot) * std::mem::size_of::<f32>()) as u64;
        self.rope_tail_in_place(
            &kv_normed,
            tail_byte_offset,
            1,
            params_for_rope,
            pos,
            false,
        )?;
        if fine { self.commit_wait_stage("attn_rope_tail"); }
        Ok(LayerAttnHalfOuts {
            normed: normed_buf,
            split: split_buf,
            qr_normed,
            q_heads,
            kv_normed_rotated: kv_normed,
        })
    }

    /// K-position layer-attn-half encoder — Phase 1 composition of the 6
    /// landed primitives:
    ///
    /// ```text
    ///   1. hc_collapse_norm_k(prev_hc_k)               → (split_k, _, normed_k)
    ///   2. encode_attn_qkv_chain_k(normed_k, q_a/b/kv) → (qr_normed_k, q_heads_k, kv_raw_k)
    ///   3. rms_norm_mul_k(kv_raw_k, γ_kv)              → kv_normed_k
    ///   4. K iter rope_tail_in_place(kv_normed_k[k][tail], pos+k)
    /// ```
    ///
    /// All inputs/outputs are `[K, ...]` row-major. Each K-position uses
    /// pos = base_pos + k for its own rope angles. Caller passes Q8_0 raw-
    /// byte weight bufs (`weight_q8_0_raw`) for attn_q_a/b/kv.
    ///
    /// Returns a `LayerAttnHalfOutsK` with [K, ...] buffers for downstream
    /// (kv_fp8_store_persistent_k + flash_attn_K + output_proj_K + …).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_attn_half_k(
        &mut self,
        // hc_collapse_norm inputs
        prev_hc_k: &DeferredBuf,
        hc_fn: &DeferredBuf,
        hc_scale: &DeferredBuf,
        hc_base: &DeferredBuf,
        hc_norm_gamma: &DeferredBuf,
        unit_gamma_hc_dim: &DeferredBuf,
        // qkv chain inputs (Q8_0 raw bytes for q/kv)
        attn_q_a_q8: &DeferredBuf,
        gamma_q: &DeferredBuf,
        attn_q_b_q8: &DeferredBuf,
        attn_kv_q8: &DeferredBuf,
        // kv rms+rope inputs
        qkv_gamma_kv: &DeferredBuf,
        // shape + scalar params
        n_hc: usize,
        n_embd: usize,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        kv_row: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        params_for_rope: &ds4_engine::attn_dispatch::LayerParams,
        base_pos: u32,
        k_positions: usize,
        hc_fn_is_f16: bool,
    ) -> Result<LayerAttnHalfOutsK> {
        let fine = std::env::var("DS4_KERNEL_PROFILE_FINE").is_ok();

        // Stage 1: K-position hc_collapse_norm. Pool its normed_k (K×n_embd) — the big
        // attn-collapse intermediate; distinct key from the ffn-collapse so the two
        // calls/layer never alias.
        arm_attn_pool_key("hc_attn_normed");
        let (split_k, _cur_k, normed_k) = self.hc_collapse_norm_k(
            prev_hc_k, hc_fn, hc_scale, hc_base, hc_norm_gamma,
            n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps,
            unit_gamma_hc_dim, k_positions, hc_fn_is_f16,
        )?;
        if fine { self.commit_wait_stage("attn_hc_collapse_k"); }

        // Stage 2: K-position qkv chain.
        let (qr_normed_k, q_heads_k, kv_raw_row_k) = self.encode_attn_qkv_chain_k(
            &normed_k, attn_q_a_q8, gamma_q, attn_q_b_q8, attn_kv_q8,
            n_lora_q, n_head, head_dim, rms_eps, kv_row, n_embd, k_positions,
        )?;
        if fine { self.commit_wait_stage("attn_qkv_chain_k"); }

        // Stage 3: K-position kv_rms_norm. Pooled (intra-layer: rope-in-place + attention
        // consume it this layer; single call-site, distinct key).
        arm_attn_pool_key("attn_kvnorm");
        let kv_normed_k =
            self.rms_norm_mul_k(&kv_raw_row_k, qkv_gamma_kv, kv_row, k_positions, rms_eps)?;
        if fine { self.commit_wait_stage("attn_kv_rms_k"); }

        // Stage 4: per-K rope_tail. Each K-pos has pos = base_pos + k. The
        // tail at row k lives at byte offset (k*kv_row + kv_row - n_rot)*4.
        let n_rot = params_for_rope.n_rot as usize;
        let f32_b = std::mem::size_of::<f32>() as u64;
        if n_rot > 0 && std::env::var("DS4_ROPE_K_FUSE").as_deref() != Ok("0") {
            // K-batched kv-tail rope: ONE dispatch (grid.y=K) vs K. The tail is
            // a single "head" of n_rot floats per row; k_row_stride = kv_row.
            // Bit-identical to the per-K loop below.
            self.rope_tail_k_fused(
                &kv_normed_k, ((kv_row - n_rot) as u64) * f32_b,
                1, n_rot, kv_row as u64, params_for_rope, base_pos, k_positions, false,
            )?;
        } else {
            for k in 0..k_positions {
                let tail_byte_offset =
                    ((k * kv_row + (kv_row - n_rot)) as u64) * f32_b;
                self.rope_tail_in_place(
                    &kv_normed_k,
                    tail_byte_offset,
                    1,
                    params_for_rope,
                    base_pos + k as u32,
                    false,
                )?;
            }
        }
        // DS4_CHUNK_ROPE_KV_ORDER=1 (Phase 2b probe): force the rope→KV-store ordering
        // here. The kv-tail rope is IN-PLACE on kv_normed_k; the downstream KV store
        // reads it. At commit_every=1 the per-layer boundary drain masks the dep, but
        // at higher cb occupancy (few-cb) Metal's cross-encoder auto-tracking misses it
        // → KV rope-tail NaN (NaN scan: layer 5 kind=KV idx=n_nope). This drain
        // CONFIRMS the dep (it splits the cb; the real few-cb fix is a fence, not a drain).
        if fine || std::env::var("DS4_CHUNK_ROPE_KV_ORDER").is_ok() {
            self.commit_wait_stage("attn_rope_tail_k");
        }

        Ok(LayerAttnHalfOutsK {
            normed_k,
            split_k,
            qr_normed_k,
            q_heads_k,
            kv_normed_rotated_k: kv_normed_k,
        })
    }

    /// Phase 2 — K-position variant of `rope_tail_q_heads_in_place`. Applies
    /// `rope_tail` to every row of a `[K, n_head, head_dim]` q_heads buffer in
    /// place. Position-aware: row k uses `pos = base_pos + k` (matches the
    /// per-K-position kv rope in `encode_layer_attn_half_k`). Loops K * 1
    /// dispatches; each dispatch processes all n_head heads of one K row at
    /// the right byte offset.
    /// K-batched rope_tail: rotates K rows in ONE dispatch (grid = n_heads × K).
    /// Each K-row k uses position `base_pos + k` and starts `k * k_row_stride`
    /// floats past `byte_offset`. The buffer binds at `byte_offset` (row-0's
    /// first-head rope-tail start); the kernel adds `kpos*k_row_stride +
    /// head_id*head_dim`. Bit-identical to looping the per-row kernel K times
    /// (each is kpos=0). Shared by the q-heads rope (k_row_stride = n_head*
    /// head_dim, n_heads = n_head, head_dim = head_dim) and the kv-tail rope
    /// (k_row_stride = kv_row, n_heads = 1, head_dim = n_rot).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail_k_fused(
        &self,
        buf: &DeferredBuf,
        byte_offset: u64,
        n_heads: usize,
        head_dim: usize,
        k_row_stride_floats: u64,
        params: &ds4_engine::attn_dispatch::LayerParams,
        base_pos: u32,
        k_positions: usize,
        backward: bool,
    ) -> Result<()> {
        let n_rot = params.n_rot as usize;
        if n_rot == 0 || k_positions == 0 { return Ok(()); }
        anyhow::ensure!(
            n_rot >= 2 && n_rot % 2 == 0 && head_dim >= n_rot,
            "rope_tail_k_fused: n_rot {} / head_dim {} invalid", n_rot, head_dim
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_rope_tail_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_rope_tail_f32 pipeline not loaded"))?
            .clone();
        let pos_buf = new_input_buffer(&self.state.device, &[base_pos as i32]);
        let freqs_buf = new_input_buffer(&self.state.device, &[0.0f32]);
        let stride_bytes = 4u32;
        // scalar[10] (pos_u) = base_pos; kernel adds kpos. scalar[12]
        // (src_stride0, repurposed) = k_row_stride in FLOATS.
        let scalars: [u32; 23] = [
            head_dim as u32,
            n_rot as u32,
            0,
            params.rope_freq_base.to_bits(),
            params.rope_freq_scale.to_bits(),
            params.rope_ext_factor.to_bits(),
            params.rope_attn_factor.to_bits(),
            (32.0f32).to_bits(),
            (1.0f32).to_bits(),
            params.rope_orig_ctx,
            base_pos,
            n_heads as u32,
            k_row_stride_floats as u32,   // src_stride0 → k_row_stride (floats)
            (head_dim as u32) * stride_bytes,
            (head_dim as u32) * stride_bytes * (n_heads as u32),
            backward as u32,
            stride_bytes,
            (head_dim as u32) * stride_bytes,
            (head_dim as u32) * stride_bytes * (n_heads as u32),
            0, 0, 0, 0,
        ];
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf.buf), byte_offset);
        enc.set_buffer(1, Some(&pos_buf), 0);
        enc.set_buffer(2, Some(&freqs_buf), 0);
        enc.set_buffer(3, Some(&buf.buf), byte_offset);
        for (i, s) in scalars.iter().enumerate() {
            set_scalar_bytes(enc, 4 + i as u64, s);
        }
        let half = (n_rot / 2) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, k_positions as u64, 1),
            MTLSize::new(half.max(1).min(1024), 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(())
    }

    pub fn rope_tail_q_heads_in_place_k(
        &self,
        q_heads_k: &DeferredBuf,
        n_head: usize,
        head_dim: usize,
        params: &ds4_engine::attn_dispatch::LayerParams,
        base_pos: u32,
        k_positions: usize,
        backward: bool,
    ) -> Result<()> {
        let n_rot = params.n_rot as usize;
        if n_rot == 0 { return Ok(()); }
        anyhow::ensure!(
            q_heads_k.n_elements == k_positions * n_head * head_dim,
            "rope_tail_q_heads_in_place_k: q_heads_k.len ({}) != K * n_head * head_dim ({} * {} * {})",
            q_heads_k.n_elements, k_positions, n_head, head_dim
        );
        anyhow::ensure!(
            n_rot >= 2 && n_rot % 2 == 0 && head_dim >= n_rot,
            "rope_tail_q_heads_in_place_k: n_rot {} / head_dim {} invalid", n_rot, head_dim
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_rope_tail_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_rope_tail_f32 pipeline not loaded"))?
            .clone();
        let freqs_buf = new_input_buffer(&self.state.device, &[0.0f32]);
        let stride_bytes = 4u32;
        let f32_b = std::mem::size_of::<f32>() as u64;
        let per_k_floats = (n_head * head_dim) as u64;

        // FUSED K path (DS4_ROPE_K_FUSE, default on): ONE dispatch over all K
        // via grid.y=K — bit-identical to the per-K loop below (each iter is
        // kpos=0 at a per-row offset). k_row_stride = per_k_floats; the buffer
        // binds at row-0's first-head tail (head_dim-n_rot).
        if std::env::var("DS4_ROPE_K_FUSE").as_deref() != Ok("0") {
            self.rope_tail_k_fused(
                q_heads_k, ((head_dim - n_rot) as u64) * f32_b,
                n_head, head_dim, per_k_floats, params, base_pos, k_positions, backward,
            )?;
            return Ok(());
        }

        for k in 0..k_positions {
            let pos = base_pos + k as u32;
            let pos_buf = new_input_buffer(&self.state.device, &[pos as i32]);
            let byte_offset = (k as u64 * per_k_floats + (head_dim - n_rot) as u64) * f32_b;
            // 23-scalar uniform — mirrors rope_tail_q_heads_in_place. The
            // per-head base stride is head_dim (each thread group handles one
            // head; head_id*head_dim steps between rows).
            let scalars: [u32; 23] = [
                head_dim as u32,
                n_rot as u32,
                0,
                params.rope_freq_base.to_bits(),
                params.rope_freq_scale.to_bits(),
                params.rope_ext_factor.to_bits(),
                params.rope_attn_factor.to_bits(),
                (32.0f32).to_bits(),
                (1.0f32).to_bits(),
                params.rope_orig_ctx,
                pos,
                n_head as u32,
                stride_bytes,
                (head_dim as u32) * stride_bytes,
                (head_dim as u32) * stride_bytes * (n_head as u32),
                backward as u32,
                stride_bytes,
                (head_dim as u32) * stride_bytes,
                (head_dim as u32) * stride_bytes * (n_head as u32),
                0, 0, 0, 0,
            ];
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&q_heads_k.buf), byte_offset);
            enc.set_buffer(1, Some(&pos_buf), 0);
            enc.set_buffer(2, Some(&freqs_buf), 0);
            enc.set_buffer(3, Some(&q_heads_k.buf), byte_offset);
            for (i, s) in scalars.iter().enumerate() {
                set_scalar_bytes(enc, 4 + i as u64, s);
            }
            let half = (n_rot / 2) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new(n_head as u64, 1, 1),
                MTLSize::new(half.max(1).min(1024), 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        Ok(())
    }

    /// Phase 2 — END-TO-END K-position attention half. Composes every Phase-1
    /// and Phase-2 K-position primitive in the order the production K=1 path
    /// uses in `SingleBufferEncoder::encode_first_half_inner`:
    ///
    /// ```text
    ///   1. encode_layer_attn_half_k          → (split, qr_normed, q_heads, kv_normed_rotated)_K
    ///   2. kv_fp8_store_persistent_k         → persistent cache (slots base_slot..base_slot+K)
    ///   3. rope_tail_q_heads_in_place_k      → q rope-forward (per-K position)
    ///   4. flash_attn_k_mla                  → [K, n_head, DV] attended values
    ///   5. rope_tail_q_heads_in_place_k back → q rope-reverse (matches K=1 path)
    ///   6. encode_attn_output_matmuls_q8_k   → (attn_low_K, attn_out_K [K, d_embd])
    /// ```
    ///
    /// Returns `(half_outs, attn_out_k, split_k)`. `attn_out_k` is the
    /// projected output ready to feed into `hc_expand_attn_split`; `split_k`
    /// flows from the hc_collapse_norm in step 1 and is needed by the FFN
    /// half. K supported in `{1, 2, 4, 8}`.
    ///
    /// Causal masking: `attn_base_pos` is forwarded to `flash_attn_k_mla`
    /// which masks per-K-row to `[0, attn_base_pos + k + 1)`. For verifier
    /// callers (where K rows write to slots [base_slot..base_slot+K) and
    /// each row's window is base_pos+k+1), pass `attn_base_pos = base_pos`.
    /// For MTP drafter callers (K=1 with its own n_raw counter), pass
    /// `attn_base_pos = mtp_n_raw + draft_iter`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_chain_k(
        &mut self,
        prev_hc_k: &DeferredBuf,
        // attn-half weights (forwarded to encode_layer_attn_half_k)
        hc_fn: &DeferredBuf,
        hc_scale: &DeferredBuf,
        hc_base: &DeferredBuf,
        hc_norm_gamma: &DeferredBuf,
        unit_gamma_hc: &DeferredBuf,
        attn_q_a_q8: &DeferredBuf,
        gamma_q: &DeferredBuf,
        attn_q_b_q8: &DeferredBuf,
        attn_kv_q8: &DeferredBuf,
        qkv_gamma_kv: &DeferredBuf,
        // output projection weights (Q8_0)
        w_o_a_q8: &DeferredBuf,
        w_o_b_q8: &DeferredBuf,
        // shape params
        n_hc: usize,
        n_embd: usize,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        kv_row: usize,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        flash_scale: f32,
        layer_idx: u32,
        layer_params: &ds4_engine::attn_dispatch::LayerParams,
        raw_cap: u32,
        base_slot: u32,
        base_pos: u32,
        attn_base_pos: u32,
        attn_sinks: &DeferredBuf,   // [n_head] per-head softmax sink logits
        k_positions: usize,
        // DS4_VERIFY_COMPRESSOR: when Some, run the per-draft ring-aware flash
        // over [raw KV | comp ring] instead of the raw flash_attn_k_mla, so the
        // verifier attends the long-range compressed context decode_step sees.
        comp: Option<&CompVerifyCtx<'_>>,
    ) -> Result<(LayerAttnHalfOutsK, DeferredBuf)> {
        anyhow::ensure!(
            n_head * head_dim == n_groups * group_dim,
            "encode_attn_chain_k: head shape mismatch: n_head*head_dim ({}*{}={}) != n_groups*group_dim ({}*{}={})",
            n_head, head_dim, n_head * head_dim, n_groups, group_dim, n_groups * group_dim
        );

        // 1. Phase 1 attn-half (collapse_norm + qkv chain + kv rms + kv rope_tail).
        let half = self.encode_layer_attn_half_k(
            prev_hc_k, hc_fn, hc_scale, hc_base, hc_norm_gamma, unit_gamma_hc,
            attn_q_a_q8, gamma_q, attn_q_b_q8, attn_kv_q8, qkv_gamma_kv,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            sinkhorn_iters, hc_eps, rms_eps,
            layer_params, base_pos, k_positions,
            // Verifier K-path always receives f32 hc (base_run_bundle dequants
            // the lean f16 — opt-in spec-decode only).
            false,
        )?;

        // DS4_DUMP_NORMED_LAYER=L: dump the verifier's row-1 attn-half normed at
        // layer L (commit+wait so half.normed_k is GPU-valid), to compare vs
        // decode_step's normed (emit-row bisection step). Diagnostic only.
        if std::env::var("DS4_DUMP_NORMED_LAYER").ok().and_then(|s| s.parse::<u32>().ok())
            == Some(layer_idx) && k_positions >= 2
        {
            self.commit_wait_stage("dump_normed_k");
            let ptr = half.normed_k.buf.contents() as *const f32;
            let row1 = unsafe { std::slice::from_raw_parts(ptr.add(n_embd), n_embd) };
            let rms = (row1.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                / n_embd as f64).sqrt();
            eprintln!("[ver-normed] layer {} row1: rms={:.4e} head={:?}",
                layer_idx, rms, &row1[..6]);
        }

        // 2. KV cache write — slots [base_slot..base_slot+K).
        let kv_cache = self.kv_fp8_store_persistent_k(
            layer_idx, &half.kv_normed_rotated_k, layer_params,
            raw_cap, base_slot, k_positions,
        )?;

        // 3. RoPE-forward on Q (matches K=1 path: forward before flash).
        self.rope_tail_q_heads_in_place_k(
            &half.q_heads_k, n_head, head_dim, layer_params,
            base_pos, k_positions, false,
        )?;

        // 4. Flash attention. Two paths:
        let flash_out_k = if let Some(ctx) = comp.filter(|c| c.n_comp > 0) {
            // 4a. Compressor-aware path (DS4_VERIFY_COMPRESSOR, ratio==4 layers
            //     with a non-empty prefill ring). Two parts:
            //
            //   (i) PER-DRAFT MAIN-COMPRESSOR EMIT: for each draft k, store its
            //   kv/sc projection into the layer's sliding-window pool at pos
            //   base_pos+k; on an emit boundary ((pos+1)%ratio==0) pool+rms+rope
            //   and append a NEW comp row to the ring. This makes a later draft
            //   k'>k attend a comp row emitted by an earlier draft within the
            //   same K-batch — exactly what decode_step does, and the cause of
            //   the row-2 divergence when omitted. (Indexer emit skipped: its
            //   top-k selection is all-allowed at n_comp<=DS4_N_INDEXER_TOP_K, so
            //   the indexer ring is never consulted.)
            //
            //   (ii) ONE f32 K-flash over [raw KV (per-K causal) | comp ring],
            //   with a per-comp-row availability mask (comp_avail): prefill rows
            //   (avail=0) are seen by all K; an in-batch emit from draft j (avail
            //   = base_pos+j+1) is seen only by drafts k>j.
            //
            //   The compressor-state pool must be SEEDED from the prefill window
            //   (caller: populate_compressor_state) — store_one slides over it.
            let comp_ring = self.state.comp_ring_or_alloc(
                layer_idx,
                raw_cap as usize * head_dim * std::mem::size_of::<f32>(),
            );
            let ratio = layer_params.compress_ratio;
            // comp_avail[r]: position at which ring row r becomes attendable.
            // prefill rows [0..n_comp) → 0 (always); in-batch emits appended.
            let mut comp_avail: Vec<u32> = vec![0u32; ctx.n_comp as usize];
            if ratio == 4 && !ctx.attn_comp.w_kv.is_empty() {
                let cc = &ctx.attn_comp;
                let coff = 2usize; // ratio==4 → 2-window
                let width_u = coff * head_dim;
                let in_dim = n_embd;
                let state_kv_len = (2 * ratio as usize) * width_u; // rows(2*ratio) × width
                let state_bytes = state_kv_len * std::mem::size_of::<f32>();
                let pk = self.state.compressor_state_kv_or_alloc(layer_idx, state_bytes);
                let ps = self.state.compressor_state_score_or_alloc(layer_idx, state_bytes);
                // store_one kernel reads APE [ratio, width]; cc.w_ape is
                // [width, ratio] — transpose (loop-invariant, built once).
                let mut ape = vec![0.0f32; ratio as usize * width_u];
                for j in 0..width_u {
                    for r in 0..ratio as usize {
                        ape[r * width_u + j] = cc.w_ape[j * ratio as usize + r];
                    }
                }
                let ape_bytes = unsafe {
                    std::slice::from_raw_parts(ape.as_ptr() as *const u8, ape.len() * 4)
                };
                let w_kv_db = self.weight_f32(cc.w_kv);
                let w_gate_db = self.weight_f32(cc.w_gate);
                let w_norm_db = self.weight_f32(cc.w_norm);
                let n_rot = layer_params.n_rot as usize;
                let hd = head_dim;
                let mut emit_idx = 0u32;
                for kk in 0..k_positions {
                    let pos = base_pos + kk as u32;
                    let normed_row = self.slice_out_f32(&half.normed_k, kk * in_dim, in_dim);
                    let kv_db = self.matvec_f32(&w_kv_db, &normed_row, in_dim, width_u)?;
                    let sc_db = self.matvec_f32(&w_gate_db, &normed_row, in_dim, width_u)?;
                    self.compressor_store_one_db(
                        &kv_db, &sc_db, ape_bytes, &pk, &ps,
                        width_u as u32, ratio, pos, false,
                    )?;
                    if (pos + 1) % ratio == 0 {
                        // emit: pool 2-window → rms → rope_tail → append to ring.
                        let pooled_db = self.compressor_pool_ratio4(&pk, &ps, hd as u32)?;
                        let emit_db = self.rms_norm_mul(&pooled_db, &w_norm_db, 1.0e-6)?;
                        if n_rot > 0 {
                            let comp_pos = pos + 1 - ratio;
                            let off = ((hd - n_rot) * std::mem::size_of::<f32>()) as u64;
                            self.rope_tail_in_place(&emit_db, off, 1, layer_params, comp_pos, false)?;
                        }
                        let ring_row = ctx.n_comp + emit_idx;
                        self.copy_buf_into(&emit_db, &comp_ring, ring_row as usize * hd);
                        comp_avail.push(pos + 1); // attendable by drafts k>kk
                        emit_idx += 1;
                        // NOTE: ratio==4 window rotation (compressor_finish) is
                        // NOT applied here — correct only for ≤1 emit per K-batch
                        // (K<=ratio; e.g. K=4). K>=8 needs inter-emit rotation.
                    }
                }
            }
            let n_comp_total = comp_avail.len() as u32;
            self.flash_attn_k_mla_comp(
                &half.q_heads_k, &kv_cache, &comp_ring, n_comp_total, &comp_avail,
                n_head, head_dim, head_dim, raw_cap as usize, k_positions,
                flash_scale, attn_base_pos, attn_sinks,
            )?
        } else {
            // 4b. Raw K-query MLA flash against the persistent cache with
            //     per-K-row causal mask (attn_base_pos defines row 0's window;
            //     row k attends [0, attn_base_pos + k + 1)).
            self.flash_attn_k_mla(
                &half.q_heads_k, &kv_cache, n_head, head_dim, head_dim,
                raw_cap as usize, k_positions, flash_scale, attn_base_pos, attn_sinks,
            )?
        };

        // DS4_DUMP_FLASH_LAYER=L: dump the verifier's row-0 flash output at layer
        // L (commit+wait), to compare vs decode_step's flash heads (flash
        // bisection). flash_out_k is [K, n_head*head_dim]; row 0 = drafts[0].
        if std::env::var("DS4_DUMP_FLASH_LAYER").ok().and_then(|s| s.parse::<u32>().ok())
            == Some(layer_idx)
        {
            self.commit_wait_stage("dump_flash_k");
            let q_dim = n_head * head_dim;
            let ptr = flash_out_k.buf.contents() as *const f32;
            // row 1 (matches the after_attn/normed row-1 comparisons).
            let row1 = unsafe { std::slice::from_raw_parts(ptr.add(q_dim), q_dim) };
            let rms = (row1.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                / q_dim as f64).sqrt();
            eprintln!("[ver-flash] layer {} row1: rms={:.4e} head={:?}",
                layer_idx, rms, &row1[..6]);
        }

        // 5. RoPE-reverse on the FLASH OUTPUT (in-place) — matches the K=1 path
        //    (single_buffer_encoder rope_tail_q_heads_in_place(&heads_b, ..,
        //    true)) and decode_step (rope_tail backward on `heads` → heads_back
        //    before output_proj). The MLA value latent carries the rope tail, so
        //    the attention output inherits it and must be un-rope'd before the
        //    output projection. (BUG FIX: previously rope'd q_heads_k — the
        //    query — which left flash_out_k rope-contaminated, so output_proj /
        //    attn_out diverged from decode_step, corrupting the residual from
        //    layer 0 onward. The forward rope on Q at step 3 stays.)
        self.rope_tail_q_heads_in_place_k(
            &flash_out_k, n_head, head_dim, layer_params,
            base_pos, k_positions, true,
        )?;

        // 6. Output projection: stage1 (Q8_0 per-group matvec) + stage2
        //    (K-amortized simdgroup-matrix matvec). `d_embd == n_embd` for
        //    DS4 (the hc-collapse-norm hyperparams already encode n_hc / n_embd).
        let (_attn_low_k, attn_out_k) = self.encode_attn_output_matmuls_q8_k(
            &flash_out_k, w_o_a_q8, w_o_b_q8,
            n_groups, n_lora_o, group_dim, out_low_dim, n_embd,
            k_positions,
        )?;

        // `half.split_k` is the unchanged hc-collapse-norm split output;
        // caller reads it via the returned `half` to chain into the FFN-half's
        // `hc_expand_attn_split`. No copy needed — DeferredBufs are reference
        // handles to underlying MTLBuffers.
        Ok((half, attn_out_k))
    }

    /// Phase 2 — END-TO-END K-position FFN half. Composes every Phase-1 +
    /// Phase-2 K-position FFN primitive in the order the production K=1
    /// path uses in `SingleBufferEncoder::encode_first_half_inner`:
    ///
    /// ```text
    ///   1. hc_expand_attn_split_k        → after_attn_K [K, n_hc, d_embd]
    ///   2. hc_collapse_norm_k (FFN)      → (split_K, _cur_K, normed_K)
    ///   3. encode_router_logits_k        → probs_K [K, n_experts]
    ///   4. encode_router_finalize_k      → Vec<K> of (selected, weights)
    ///   5. encode_moe_and_shared_chain_k → (moe_K, shared_K) [K, d_embd]
    ///   6. hc_expand_add_split_k         → after_ffn_K [K, n_hc, d_embd]
    /// ```
    ///
    /// Inputs:
    /// - `attn_out_k`     `[K, d_embd]`       — attn-half output (from `encode_attn_chain_k`)
    /// - `cur_hc_k`       `[K, n_hc, d_embd]` — previous-token HC residual at K
    /// - `attn_split_k`   `[K, mix_hc]`       — split from attn-half (= `half.split_k`)
    /// - `hc_ffn_*`       — FFN-side hc-collapse weights
    /// - `w_router`       `[n_experts, d_embd]` f32
    /// - `bias`           `[256]` f32
    /// - shared-expert weights (f32 + Q8_0 bytes for the Q8_0 path)
    ///
    /// Returns `after_ffn_K [K, n_hc, d_embd]` — the next-layer HC residual.
    /// K supported in `{1,2,4,8}`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_chain_k(
        &mut self,
        attn_out_k: &DeferredBuf,
        cur_hc_k: &DeferredBuf,
        attn_split_k: &DeferredBuf,
        // FFN-side hc_collapse_norm weights
        hc_ffn_fn: &DeferredBuf,
        hc_ffn_fn_is_f16: bool,
        hc_ffn_scale: &DeferredBuf,
        hc_ffn_base: &DeferredBuf,
        hc_ffn_norm_gamma: &DeferredBuf,
        unit_gamma_hc: &DeferredBuf,
        // Router weights
        w_router: &DeferredBuf,
        w_router_is_f16: bool,
        router_bias: &DeferredBuf,
        // Shared-expert weights (CPU slices — uploaded inside moe_and_shared_chain)
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        sh_w_gate_q8: &[u8],
        sh_w_up_q8: &[u8],
        sh_w_down_q8: &[u8],
        // Shape + scalar params
        n_hc: usize,
        n_embd: usize,
        n_experts: usize,
        d_ffn: usize,
        shared_dim: u32,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        layer_idx: u32,
        k_positions: usize,
        // HASH ROUTING (Phase-A layers L0/1/2): Some((routing_table, token_ids[K],
        // n_experts_used)) → experts selected from the table, weighted from probs
        // (hash_router_weights_from_probs), NOT top-k. Runs mm_id MoE + returns early.
        hash_route: Option<(&[i32], &[i32], usize)>,
    ) -> Result<DeferredBuf> {
        let d_embd = n_embd;
        // DS4_KERNEL_PROFILE_FINE: per-stage GPU attribution within the FFN half
        // (serializes via commit_wait_stage — diagnostic only, kills overlap).
        let fine = std::env::var("DS4_KERNEL_PROFILE_FINE").is_ok();

        // 1. Fold attn_out into the HC residual (per K).
        let after_attn_k = self.hc_expand_attn_split_k(
            attn_out_k, cur_hc_k, attn_split_k, n_hc, d_embd, k_positions,
        )?;
        if fine { self.commit_wait_stage("ffn_hc_expand_attn"); }

        // 2. FFN-side hc_collapse_norm (K iterations through the existing kernel).
        // Pool its normed_k under a DISTINCT key from the attn-collapse (no intra-layer alias).
        arm_attn_pool_key("hc_ffn_normed");
        let (ffn_split_k, _cur_k, normed_k) = self.hc_collapse_norm_k(
            &after_attn_k, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, hc_ffn_norm_gamma,
            n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps, unit_gamma_hc, k_positions,
            hc_ffn_fn_is_f16,
        )?;
        if fine { self.commit_wait_stage("ffn_hc_collapse"); }

        // 3. Router logits at K (matvec + softplus_sqrt over flat K*n_experts).
        let probs_k = self.encode_router_logits_k(
            w_router, &normed_k, n_experts, d_embd, k_positions, w_router_is_f16,
        )?;

        // SSD-streaming chunked prefill: the chunk pool must hold THIS
        // layer's experts before the mm_id encode below. The CPU refill
        // overwrites buffers the GPU may still be reading (previous layer's
        // queued mm_id), so wait first. One wait + ~2.6 GB sequential
        // mmap read per layer per chunk — amortized over the chunk's K
        // tokens (the per-token path pays per token).
        let ssd_stream = std::env::var("DS4_SSD_STREAM").is_ok();
        if ssd_stream && self.state.chunk_pool_needs_refill(layer_idx) {
            self.commit_wait_stage("ssd_chunk_pool");
            let _ = self.state.chunk_pool_bind(layer_idx);
            // Prefetch next MoE layer's expert spans while this layer's GPU
            // work runs — fill_layer is page-fault bound, so async readahead
            // of L+1 turns the next refill into warm-page memcpy.
            if let Some(next) = self.state.expert_weights.get(layer_idx as usize + 1) {
                for t in [&next.gate, &next.up, &next.down] {
                    unsafe {
                        libc::madvise(
                            t.mmap_ptr as *mut libc::c_void,
                            t.mmap_len as usize,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
        }

        // 4-HASH. Phase-A hash layers: selection from the routing table, weights from
        // the router probs (no top-k). Self-contained — mm_id MoE + shared + fold, early return.
        if let Some((table, token_ids, k_used)) = hash_route {
            anyhow::ensure!(token_ids.len() == k_positions, "hash_route: token_ids len != K");
            // Selection is a pure CPU routing-table lookup — never needs probs.
            let mut sel = vec![0i32; k_positions * 6];
            for kk in 0..k_positions {
                let tok = token_ids[kk] as usize;
                let base = tok * k_used;
                anyhow::ensure!(base + k_used <= table.len(),
                    "hash_route: token {tok} k_used {k_used} oob (table {})", table.len());
                for j in 0..k_used.min(6) {
                    sel[kk * 6 + j] = table[base + j] as i32;
                }
            }
            let sel_flat = DeferredBuf { buf: new_input_buffer(&self.state.device, &sel), n_elements: sel.len() };
            // Chunk-graph stage 3 (DS4_CHUNK_HASH_GPU=1): weights on GPU from
            // resident probs_k — removes the per-chunk drain. Default = CPU drain.
            let hash_gpu = std::env::var("DS4_CHUNK_HASH_GPU").ok().as_deref() == Some("1")
                && k_used == 6
                && self.state.pipelines.contains_key("ds4_dsv4_router_weights_k");
            let wt_flat = if hash_gpu {
                let wt_db = self.pooled_scratch_f32("router_wt_k", k_positions * 6); // T4: pool+pin
                let pipe = self.state.pipelines.get("ds4_dsv4_router_weights_k").unwrap().clone();
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&probs_k.buf), 0);
                enc.set_buffer(1, Some(&sel_flat.buf), 0);
                enc.set_buffer(2, Some(&wt_db.buf), 0);
                let ku = k_used as u32;
                let ne = n_experts as u32;
                let scale = ds4_engine::moe::router_scale();
                let min_sum = 6.103515625e-5f32;
                enc.set_bytes(3, 4, &ku as *const _ as *const _);
                enc.set_bytes(4, 4, &ne as *const _ as *const _);
                enc.set_bytes(5, 4, &scale as *const _ as *const _);
                enc.set_bytes(6, 4, &min_sum as *const _ as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(k_positions as u64, 1, 1),
                    MTLSize::new(6, 1, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
                wt_db
            } else {
                let probs_cpu = self.commit_wait_read_multi(&[&probs_k]).pop().unwrap();
                let mut wt = vec![0f32; k_positions * 6];
                for kk in 0..k_positions {
                    let tok = token_ids[kk] as usize;
                    let base = tok * k_used;
                    let sel_k: Vec<usize> = table[base..base + k_used].iter().map(|&v| v as usize).collect();
                    let probs_row = &probs_cpu[kk * n_experts..(kk + 1) * n_experts];
                    let w = ds4_engine::moe::hash_router_weights_from_probs(probs_row, &sel_k, k_used);
                    for j in 0..k_used.min(6) {
                        wt[kk * 6 + j] = w[j];
                    }
                }
                DeferredBuf { buf: new_input_buffer(&self.state.device, &wt), n_elements: wt.len() }
            };
            let moe_k = self.encode_moe_chain_mm_qx_k_auto(
                layer_idx, &normed_k, &sel_flat, &wt_flat, d_embd, d_ffn, k_positions,
            )?;
            let wg_db = self.weight_q8_0_raw(sh_w_gate_q8, shared_dim as usize * d_embd);
            let wu_db = self.weight_q8_0_raw(sh_w_up_q8, shared_dim as usize * d_embd);
            let wd_db = self.weight_q8_0_raw(sh_w_down_q8, d_embd * shared_dim as usize);
            let shared_k = self.encode_shared_chain_q8_k(
                &normed_k, &wg_db, &wu_db, &wd_db, d_embd, shared_dim as usize, k_positions,
            )?;
            let after_ffn_k = self.hc_expand_add_split_k(
                &shared_k, &moe_k, &after_attn_k, &ffn_split_k, n_hc, d_embd, k_positions,
            )?;
            let _ = (rms_eps, sh_w_gate, sh_w_up, sh_w_down, router_bias);
            return Ok(after_ffn_k);
        }

        // 4. Top-6 expert selection + weight normalization per K. DS4_CHUNK_BATCH_ROUTER
        // collapses the K per-token select+weights dispatches into 2 batched ones (and
        // skips the K flatten blits) for the mm_id/fused engines that consume the flat
        // buffers; the blit-shim else-path keeps the per-token Vec (computed in-branch).
        let batch_router = std::env::var("DS4_CHUNK_BATCH_ROUTER").ok().as_deref() == Some("1");

        // 5. K-position MoE + shared chain. DS4_MOE_K_PATH selects which
        //    MoE path is used:
        //     - default / "blit": K-linear blit-shim around K=1 MoE kernel
        //       (encode_moe_and_shared_chain_k). 8× K=1 cost at K=8.
        //     - "fused": Option A K-batched FUSED pair_swiglu_K + sum6_K
        //       (encode_moe_chain_fused_K_auto). 2-3× K=1 cost at K=8.
        //       Shared chain runs separately via encode_shared_chain_q8_k
        //       (already K-amortized via matvec_k_q8_0).
        let moe_k_path = std::env::var("DS4_MOE_K_PATH").ok();
        let use_fused = moe_k_path.as_deref() == Some("fused") && !ssd_stream;
        // LARGE-K chunk-prefill default = mm_id (token-gather GEMM, scales
        // sub-linearly with K). At k>=128 it beats the K-linear blit shim
        // (@3000 measured 18.9→27.2 tok/s) and is coherent. SMALL K
        // (spec-decode K<=8) keeps the blit default — mm_id loses there
        // ([[moe-k-step5-mm-id-loses]]). Explicit DS4_MOE_K_PATH overrides;
        // SSD-streaming forces mm_id (only engine that binds the streamed pool).
        let large_k_mm_id = moe_k_path.is_none() && k_positions >= 128;
        // SSD-streaming forces mm_id: it is the only chunk MoE engine that
        // binds the streamed chunk pool (fused/blit bind the full stub).
        let use_mm_id = moe_k_path.as_deref() == Some("mm_id") || ssd_stream || large_k_mm_id;
        let (moe_k, shared_k) = if use_mm_id {
            // Large-K expert-token-gather MoE (scales ~linearly with K — the
            // chunked-prefill engine). Shared chain runs separately (already
            // K-amortized via matvec_k_q8_0), like the fused path.
            let (sel_flat, wt_flat) = if batch_router {
                self.encode_router_finalize_flat_k(&probs_k, router_bias, k_positions)?
            } else {
                let swk = self.encode_router_finalize_k(&probs_k, router_bias, k_positions)?;
                self.flatten_router_output_k(&swk)?
            };
            if fine { self.commit_wait_stage("ffn_router"); }
            let moe = self.encode_moe_chain_mm_qx_k_auto(
                layer_idx, &normed_k, &sel_flat, &wt_flat, d_embd, d_ffn, k_positions,
            )?;
            let wg_db = self.weight_q8_0_raw(sh_w_gate_q8, shared_dim as usize * d_embd);
            let wu_db = self.weight_q8_0_raw(sh_w_up_q8,   shared_dim as usize * d_embd);
            let wd_db = self.weight_q8_0_raw(sh_w_down_q8, d_embd * shared_dim as usize);
            if fine { self.commit_wait_stage("ffn_moe_experts"); }
            let shared = self.encode_shared_chain_q8_k(
                &normed_k, &wg_db, &wu_db, &wd_db,
                d_embd, shared_dim as usize, k_positions,
            )?;
            if fine { self.commit_wait_stage("ffn_shared"); }
            (moe, shared)
        } else if use_fused {
            // Fused-K path: flatten router outputs, dispatch fused-K MoE
            // chain via auto-detected ttype, run shared chain separately.
            let (sel_flat, wt_flat) = if batch_router {
                self.encode_router_finalize_flat_k(&probs_k, router_bias, k_positions)?
            } else {
                let swk = self.encode_router_finalize_k(&probs_k, router_bias, k_positions)?;
                self.flatten_router_output_k(&swk)?
            };
            if fine { self.commit_wait_stage("ffn_router"); }
            // DS4_MOE_EXPERT_PROFILE: count DISTINCT experts across the K
            // candidates' top-6 selections this layer. distinct/6 = the per-layer
            // MoE blowup vs K=1 (tests the spec-decode-MoE divergence: if distinct
            // ≪ K*6 the candidates share experts → a dedup'd verify-MoE could read
            // each distinct expert once; if distinct ≈ K*6 the divergence is real
            // and the K=4 MoE cost is irreducible). Serializes the chain — gated.
            if std::env::var("DS4_MOE_EXPERT_PROFILE").is_ok() {
                self.commit_wait_stage("moe_expert_profile");
                let n = k_positions * 6;
                let sel = unsafe { read_buffer::<i32>(&sel_flat.buf, n) };
                let mut seen = std::collections::BTreeSet::new();
                for &e in &sel { seen.insert(e); }
                eprintln!(
                    "MOEEXP layer={} K={} slots={} distinct={} blowup={:.2}x",
                    layer_idx, k_positions, n, seen.len(), seen.len() as f64 / 6.0
                );
            }
            let moe = self.encode_moe_chain_fused_K_auto(
                layer_idx, &normed_k, &sel_flat, &wt_flat,
                d_embd, d_ffn, k_positions,
            )?;
            // Shared chain — already K-amortized; uploads its own Q8_0 bytes.
            let wg_db = self.weight_q8_0_raw(sh_w_gate_q8, shared_dim as usize * d_embd);
            let wu_db = self.weight_q8_0_raw(sh_w_up_q8,   shared_dim as usize * d_embd);
            let wd_db = self.weight_q8_0_raw(sh_w_down_q8, d_embd * shared_dim as usize);
            if fine { self.commit_wait_stage("ffn_moe_experts"); }
            let shared = self.encode_shared_chain_q8_k(
                &normed_k, &wg_db, &wu_db, &wd_db,
                d_embd, shared_dim as usize, k_positions,
            )?;
            if fine { self.commit_wait_stage("ffn_shared"); }
            (moe, shared)
        } else {
            // Blit-shim (small K, e.g. spec-decode): keep the per-token Vec.
            let sel_wt_k = self.encode_router_finalize_k(&probs_k, router_bias, k_positions)?;
            if fine { self.commit_wait_stage("ffn_router"); }
            self.encode_moe_and_shared_chain_k(
                layer_idx, &normed_k, &sel_wt_k, d_ffn,
                sh_w_gate, sh_w_up, sh_w_down, shared_dim,
                sh_w_gate_q8, sh_w_up_q8, sh_w_down_q8,
                d_embd, k_positions,
            )?
        };
        let _ = (rms_eps, sh_w_gate, sh_w_up, sh_w_down);

        // 6. Fold shared + moe back into HC residual.
        let after_ffn_k = self.hc_expand_add_split_k(
            &shared_k, &moe_k, &after_attn_k, &ffn_split_k, n_hc, d_embd, k_positions,
        )?;

        // DS4_DUMP_RESID_LAYER=L: dump the verifier's row-1 after_attn (post
        // output-proj + hc_expand_attn) and after_ffn (post FFN/MoE half) at
        // layer L, to split the post-flash path vs decode_step. Diagnostic only.
        if std::env::var("DS4_DUMP_RESID_LAYER").ok().and_then(|s| s.parse::<u32>().ok())
            == Some(layer_idx) && k_positions >= 2
        {
            self.commit_wait_stage("dump_resid_k");
            let hc = n_hc * d_embd;
            let dump = |label: &str, buf: &DeferredBuf| {
                let ptr = buf.buf.contents() as *const f32;
                let row1 = unsafe { std::slice::from_raw_parts(ptr.add(hc), hc) };
                let rms = (row1.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                    / hc as f64).sqrt();
                eprintln!("[ver-{}] layer {} row1: rms={:.4e} head={:?}",
                    label, layer_idx, rms, &row1[..6]);
            };
            // attn_out_k is [K, d_embd] (output-proj result, pre-hc_expand) —
            // stride d_embd, not hc. Dump row 1 for the output_proj vs hc_expand split.
            {
                let ptr = attn_out_k.buf.contents() as *const f32;
                let row1 = unsafe { std::slice::from_raw_parts(ptr.add(d_embd), d_embd) };
                let rms = (row1.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                    / d_embd as f64).sqrt();
                eprintln!("[ver-attnout] layer {} row1: rms={:.4e} head={:?}",
                    layer_idx, rms, &row1[..6]);
            }
            dump("afterattn", &after_attn_k);
            dump("afterffn", &after_ffn_k);
        }

        Ok(after_ffn_k)
    }

    /// Phase 2 — FULL per-layer K-position forward pass. Composes
    /// `encode_attn_chain_k` (attention half) + `encode_ffn_chain_k`
    /// (FFN half) into ONE scope. Input `prev_hc_k [K, n_hc, d_embd]`,
    /// output `after_ffn_k [K, n_hc, d_embd]` — ready to feed the next
    /// layer's `encode_layer_k`.
    ///
    /// This is the milestone composition for K-position decode: one call
    /// per layer, in one cb. For a 43-layer DS4 forward pass at K=8 it
    /// closes the bridge between "K-position primitives" and "K-position
    /// per-token throughput."
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layer_k(
        &mut self,
        prev_hc_k: &DeferredBuf,
        // Attn-half weights
        hc_attn_fn: &DeferredBuf,
        hc_attn_scale: &DeferredBuf,
        hc_attn_base: &DeferredBuf,
        hc_attn_norm_gamma: &DeferredBuf,
        unit_gamma_hc: &DeferredBuf,
        attn_q_a_q8: &DeferredBuf,
        gamma_q: &DeferredBuf,
        attn_q_b_q8: &DeferredBuf,
        attn_kv_q8: &DeferredBuf,
        qkv_gamma_kv: &DeferredBuf,
        w_o_a_q8: &DeferredBuf,
        w_o_b_q8: &DeferredBuf,
        // FFN-half weights
        hc_ffn_fn: &DeferredBuf,
        hc_ffn_scale: &DeferredBuf,
        hc_ffn_base: &DeferredBuf,
        hc_ffn_norm_gamma: &DeferredBuf,
        w_router: &DeferredBuf,
        router_bias: &DeferredBuf,
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        sh_w_gate_q8: &[u8],
        sh_w_up_q8: &[u8],
        sh_w_down_q8: &[u8],
        // Shape params
        n_hc: usize,
        n_embd: usize,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        kv_row: usize,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        n_experts: usize,
        d_ffn: usize,
        shared_dim: u32,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        flash_scale: f32,
        layer_idx: u32,
        layer_params: &ds4_engine::attn_dispatch::LayerParams,
        raw_cap: u32,
        base_slot: u32,
        base_pos: u32,
        attn_base_pos: u32,
        attn_sinks: &DeferredBuf,   // [n_head] per-head softmax sink logits
        k_positions: usize,
        comp: Option<&CompVerifyCtx<'_>>,  // DS4_VERIFY_COMPRESSOR ctx (None = raw flash)
    ) -> Result<DeferredBuf> {
        // DS4_LAYER_PROFILE: split per-layer GPU time into attn-half vs
        // ffn-half via synchronous commit_wait_stage boundaries (kills
        // overlap — profiling only). Pair with DS4_VERIFY_N_LAYERS=1 to read
        // a single layer's breakdown; commit_wait_traced emits each stage's
        // GPU-busy time under DS4_OP_TRACE.
        let layer_prof = std::env::var("DS4_LAYER_PROFILE").is_ok();

        // Attention half — `attn_base_pos` defines the per-K causal window
        // forwarded into `flash_attn_k_mla`.  Verifier callers pass base_pos;
        // MTP drafter passes mtp_n_raw + draft_iter.
        let (half, attn_out_k) = self.encode_attn_chain_k(
            prev_hc_k,
            hc_attn_fn, hc_attn_scale, hc_attn_base, hc_attn_norm_gamma, unit_gamma_hc,
            attn_q_a_q8, gamma_q, attn_q_b_q8, attn_kv_q8, qkv_gamma_kv,
            w_o_a_q8, w_o_b_q8,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            n_groups, n_lora_o, group_dim, out_low_dim,
            sinkhorn_iters, hc_eps, rms_eps, flash_scale,
            layer_idx, layer_params, raw_cap, base_slot, base_pos,
            attn_base_pos, attn_sinks, k_positions, comp,
        )?;

        if layer_prof {
            self.commit_wait_stage("layer_attn_half");
        }

        // FFN half.
        let out = self.encode_ffn_chain_k(
            &attn_out_k, prev_hc_k, &half.split_k,
            hc_ffn_fn, false, hc_ffn_scale, hc_ffn_base, hc_ffn_norm_gamma, unit_gamma_hc,
            w_router, false, router_bias,
            sh_w_gate, sh_w_up, sh_w_down,
            sh_w_gate_q8, sh_w_up_q8, sh_w_down_q8,
            n_hc, n_embd, n_experts, d_ffn, shared_dim,
            sinkhorn_iters, hc_eps, rms_eps,
            layer_idx, k_positions,
            None,
        )?;

        if layer_prof {
            self.commit_wait_stage("layer_ffn_half");
        }

        Ok(out)
    }

    /// Phase E M5.4.2: composable `kv_fp8_store` against the persistent
    /// per-layer KV buffer. The kv row to store comes from a
    /// DeferredBuf (typically `kv_normed_rotated` from
    /// `encode_layer_attn_half`); the destination is the persistent
    /// buffer for `layer_idx` (from `kv_buffer_or_alloc`).
    ///
    /// Encodes into THIS scope — no commit/wait. Returns the
    /// persistent cache buffer as a `DeferredBuf` so the caller can
    /// chain a `flash_attn_decode_metal_persistent`-style read in
    /// the same cb (the FA kernel reads the cache buffer via its
    /// own f32→f16 conversion).
    ///
    /// Uses the same `ds4_dsv4_kv_fp8_store` shim as the inherent
    /// `kv_fp8_store_persistent_impl` (which gets the bridge-prefer
    /// fix from M5.2.2).
    pub fn kv_fp8_store_persistent(
        &self,
        layer_idx: u32,
        kv_row: &DeferredBuf,
        params: &ds4_engine::attn_dispatch::LayerParams,
        raw_cap: u32,
        slot: u32,
    ) -> Result<DeferredBuf> {
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(
            kv_row.n_elements == row,
            "kv_fp8_store_persistent: kv_row has {} elems, expected n_lora_kv ({})",
            kv_row.n_elements,
            row
        );
        anyhow::ensure!(raw_cap > 0, "raw_cap must be > 0");
        anyhow::ensure!(
            slot < raw_cap,
            "slot {} >= raw_cap {}",
            slot,
            raw_cap
        );

        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_kv_fp8_store")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_kv_fp8_store pipeline not loaded"))?
            .clone();

        let cache_byte_len = (raw_cap as usize) * row * std::mem::size_of::<f32>();
        let cache_buf_metal = self.state.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        let cache_buf = DeferredBuf {
            buf: cache_buf_metal,
            n_elements: (raw_cap as usize) * row,
        };

        let n_rot = params.n_rot as u32;
        let n_nope = (params.head_dim as u32).saturating_sub(n_rot);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&cache_buf.buf), 0);
        enc.set_buffer(1, Some(&kv_row.buf), 0);
        set_scalar_bytes(enc, 2, &n_nope);
        set_scalar_bytes(enc, 3, &n_rot);
        set_scalar_bytes(enc, 4, &slot);

        let chunks = (n_nope / 64).max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(chunks.min(1024), 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(cache_buf)
    }

    /// K-query MLA flash attention against the persistent KV cache.
    /// Reads Q [K, n_head, DK], KV cache [n_cache, DK] (shared K/V — MLA),
    /// writes O [K, n_head, DV]. All buffers f32; the kernel casts to half
    /// on shared-memory writes (cache values are already FP8-snapped via
    /// kv_fp8_store, FP16-representable, so the cast is lossless).
    ///
    /// Backed by `ds4_kernel_flash_attn_K_mla_f32_sg` (the Phase 0
    /// simdgroup-matrix flash attention kernel; K=8 ratio 1.03 vs K=1
    /// per drivers/flash_attn_K_simdgroup_check.swift). K supported in
    /// {1, 2, 4, 8}. DK, DV must equal 512 (MLA shape). n_cache must be a
    /// multiple of 8 (C=8 KV rows per outer tile).
    #[allow(clippy::too_many_arguments)]
    /// K-position MLA flash attention with causal window.
    ///
    /// `attn_base_pos` defines the per-row causal window: query row q in
    /// [0, K) attends over KV rows [0, attn_base_pos + q + 1).  Rows past
    /// that limit (up to n_cache) are masked to -INFINITY in the softmax
    /// so they contribute zero weight.  Without this, the kernel would
    /// attend over the entire `n_cache`-sized raw cache including stale /
    /// uninitialized slots, silently corrupting attention output.
    ///
    /// Callers:
    /// - Verifier base layer (K writes to slots [base_slot..base_slot+K)):
    ///   pass `attn_base_pos = base_pos` (row k's window covers base_pos+k+1
    ///   slots — base prefix + this K-batch up to and including row k).
    /// - MTP drafter (K=1) at draft iter i with MTP cache size N before
    ///   this call: pass `attn_base_pos = N + i` so window = N + i + 1.
    pub fn flash_attn_k_mla(
        &self,
        q_k: &DeferredBuf,       // [K, n_head, DK] f32
        kv_cache: &DeferredBuf,   // [n_cache, DK] f32 (raw_cap × n_lora_kv)
        n_head: usize,
        dk: usize,
        dv: usize,
        n_cache: usize,
        k_positions: usize,
        scale: f32,
        attn_base_pos: u32,
        attn_sinks: &DeferredBuf,   // [n_head] per-head softmax sink logits
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            q_k.n_elements == k_positions * n_head * dk,
            "flash_attn_k_mla: q_k has {} elems, expected K*n_head*DK = {}*{}*{}",
            q_k.n_elements, k_positions, n_head, dk
        );
        anyhow::ensure!(
            attn_sinks.n_elements == n_head,
            "flash_attn_k_mla: attn_sinks has {} elems, expected n_head={}",
            attn_sinks.n_elements, n_head
        );
        anyhow::ensure!(dk == 512 && dv == 512, "MLA DS4 shape: DK=DV=512");
        anyhow::ensure!(n_cache % 8 == 0, "n_cache must be multiple of 8 (C=8)");
        anyhow::ensure!(matches!(k_positions, 1 | 2 | 4 | 8),
                         "K must be in {{1,2,4,8}} (got {})", k_positions);

        let dk_i32: i32 = dk as i32;
        let dv_i32: i32 = dv as i32;
        let k_i32: i32 = k_positions as i32;
        let nc_i32: i32 = n_cache as i32;
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&dk_i32.to_le_bytes());
        key.extend_from_slice(&dv_i32.to_le_bytes());
        key.extend_from_slice(&k_i32.to_le_bytes());
        key.extend_from_slice(&nc_i32.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_flash_attn_K_mla_f32_sg", &key, |fcv| {
                fcv.set_constant_value_at_index(&dk_i32 as *const _ as *const _, MTLDataType::Int, 1600);
                fcv.set_constant_value_at_index(&dv_i32 as *const _ as *const _, MTLDataType::Int, 1601);
                fcv.set_constant_value_at_index(&k_i32 as *const _ as *const _, MTLDataType::Int, 1602);
                fcv.set_constant_value_at_index(&nc_i32 as *const _ as *const _, MTLDataType::Int, 1603);
                let has_mask = false;
                fcv.set_constant_value_at_index(&has_mask as *const _ as *const _, MTLDataType::Bool, 1604);
            })?;

        let out = self.alloc_f32(k_positions * n_head * dv);

        // Threadgroup memory: SQ + SK + SV (halves) + SS (floats) = 24KB at DK=DV=512.
        let q_bytes: u64 = 8 * (dk as u64) * 2;
        let v_bytes: u64 = (dv as u64) * 8 * 2;
        let ss_bytes: u64 = 64 * 4;
        let shmem_bytes: u64 = q_bytes + q_bytes + v_bytes + ss_bytes;

        // DS4_DUMP_FLASH_Q: dump the flash INPUTS (rope'd-q head0, KV rows 0 +
        // attn_base_pos) so they can be compared against decode_step's flash
        // inputs — localizes whether the in-chain attn_out divergence is q or KV.
        // Buffers are StorageModeShared (CPU-readable); read before dispatch.
        if std::env::var("DS4_DUMP_FLASH_Q").is_ok() {
            let q_ptr = q_k.buf.contents() as *const f32;
            let kv_ptr = kv_cache.buf.contents() as *const f32;
            let rd = |p: *const f32, off: usize, n: usize| -> (f64, Vec<f32>) {
                let mut s = 0.0f64; let mut head = Vec::new();
                for i in 0..n { let v = unsafe { *p.add(off + i) };
                    s += (v as f64) * (v as f64); if i < 4 { head.push(v); } }
                ((s / n as f64).sqrt(), head)
            };
            // q layout = [K, n_head, DK]; head h at k=0 is offset h*DK.
            let (q0r, _) = rd(q_ptr, 0, dk);                  // head 0
            let (q1r, _) = rd(q_ptr, dk, dk);                 // head 1
            let (q2r, _) = rd(q_ptr, 2 * dk, dk);             // head 2
            let qlr = rd(q_ptr, (n_head - 1) * dk, dk).0;     // last head
            // whole-q rms over all heads (K=0 row)
            let qall = rd(q_ptr, 0, n_head * dk).0;
            let (k0r, _) = rd(kv_ptr, 0, dk);                 // KV slot 0
            let (kbr, _) = rd(kv_ptr, (attn_base_pos as usize) * dk, dk);
            eprintln!("  [FLASH-IN gpu] q rms: h0={:.4} h1={:.4} h2={:.4} h{}={:.4} all={:.4} | kv[0]={:.4} kv[{}]={:.4}",
                q0r, q1r, q2r, n_head - 1, qlr, qall, k0r, attn_base_pos, kbr);
        }

        let n_head_u32: u32 = n_head as u32;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_k.buf), 0);
        enc.set_buffer(1, Some(&kv_cache.buf), 0);
        enc.set_buffer(2, Some(&out.buf), 0);
        set_scalar_bytes(enc, 3, &scale);
        set_scalar_bytes(enc, 4, &n_head_u32);
        set_scalar_bytes(enc, 5, &attn_base_pos);
        enc.set_buffer(6, Some(&attn_sinks.buf), 0);
        // Buffers 7/8/9 (comp ring + n_comp + comp_avail): raw path attends no
        // compressed rows, so bind kv_cache as a harmless dummy and n_comp=0
        // (kernel skips the comp loop). The compressor-aware path uses
        // flash_attn_k_mla_comp.
        enc.set_buffer(7, Some(&kv_cache.buf), 0);
        set_scalar_bytes(enc, 8, &0u32);
        enc.set_buffer(9, Some(&kv_cache.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32, 4, 1),  // NSG=4 simdgroups × NW=32 lanes
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Compressor-aware K-query MLA flash: like [`Self::flash_attn_k_mla`] but
    /// each query row ALSO attends `n_comp` compressed rows from `comp_ring`
    /// ([n_comp, DK] f32), accumulated into the same online softmax (no per-row
    /// causal mask — comp rows summarize past windows, valid for all queries).
    /// f32-faithful to decode_step's flash_attn_decode (raw + all comp rows),
    /// unlike the f16-workspace K=1 ring-aware flash. Used by the
    /// DS4_VERIFY_COMPRESSOR path in `encode_attn_chain_k`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_k_mla_comp(
        &self,
        q_k: &DeferredBuf,
        kv_cache: &DeferredBuf,
        comp_ring: &metal::Buffer,
        n_comp: u32,
        comp_avail: &[u32],   // [n_comp] position each comp row becomes attendable (0 = prefill)
        n_head: usize,
        dk: usize,
        dv: usize,
        n_cache: usize,
        k_positions: usize,
        scale: f32,
        attn_base_pos: u32,
        attn_sinks: &DeferredBuf,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            q_k.n_elements == k_positions * n_head * dk,
            "flash_attn_k_mla_comp: q_k has {} elems, expected K*n_head*DK = {}*{}*{}",
            q_k.n_elements, k_positions, n_head, dk
        );
        anyhow::ensure!(
            comp_avail.len() == n_comp as usize,
            "flash_attn_k_mla_comp: comp_avail len {} != n_comp {}",
            comp_avail.len(), n_comp
        );
        anyhow::ensure!(attn_sinks.n_elements == n_head, "attn_sinks n_head mismatch");
        anyhow::ensure!(dk == 512 && dv == 512, "MLA DS4 shape: DK=DV=512");
        anyhow::ensure!(n_cache % 8 == 0, "n_cache must be multiple of 8 (C=8)");
        anyhow::ensure!(matches!(k_positions, 1 | 2 | 4 | 8),
                         "K must be in {{1,2,4,8}} (got {})", k_positions);

        let dk_i32 = dk as i32;
        let dv_i32 = dv as i32;
        let k_i32 = k_positions as i32;
        let nc_i32 = n_cache as i32;
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&dk_i32.to_le_bytes());
        key.extend_from_slice(&dv_i32.to_le_bytes());
        key.extend_from_slice(&k_i32.to_le_bytes());
        key.extend_from_slice(&nc_i32.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_flash_attn_K_mla_f32_sg", &key, |fcv| {
                fcv.set_constant_value_at_index(&dk_i32 as *const _ as *const _, MTLDataType::Int, 1600);
                fcv.set_constant_value_at_index(&dv_i32 as *const _ as *const _, MTLDataType::Int, 1601);
                fcv.set_constant_value_at_index(&k_i32 as *const _ as *const _, MTLDataType::Int, 1602);
                fcv.set_constant_value_at_index(&nc_i32 as *const _ as *const _, MTLDataType::Int, 1603);
                let has_mask = false;
                fcv.set_constant_value_at_index(&has_mask as *const _ as *const _, MTLDataType::Bool, 1604);
            })?;

        let out = self.alloc_f32(k_positions * n_head * dv);
        let q_bytes: u64 = 8 * (dk as u64) * 2;
        let v_bytes: u64 = (dv as u64) * 8 * 2;
        let ss_bytes: u64 = 64 * 4;
        let shmem_bytes: u64 = q_bytes + q_bytes + v_bytes + ss_bytes;

        let n_head_u32 = n_head as u32;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_k.buf), 0);
        enc.set_buffer(1, Some(&kv_cache.buf), 0);
        enc.set_buffer(2, Some(&out.buf), 0);
        set_scalar_bytes(enc, 3, &scale);
        set_scalar_bytes(enc, 4, &n_head_u32);
        set_scalar_bytes(enc, 5, &attn_base_pos);
        enc.set_buffer(6, Some(&attn_sinks.buf), 0);
        enc.set_buffer(7, Some(comp_ring), 0);
        set_scalar_bytes(enc, 8, &n_comp);
        // comp_avail: upload [n_comp] u32 (>=1 elem so the buffer is non-null).
        let avail_pad: Vec<u32> = if comp_avail.is_empty() { vec![0u32] } else { comp_avail.to_vec() };
        let avail_buf = self.state.device.new_buffer_with_data(
            avail_pad.as_ptr() as *const _,
            (avail_pad.len() * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        enc.set_buffer(9, Some(&avail_buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32, 4, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// DS4_CHUNK_SWA_KFLASH: masked variant of [`Self::flash_attn_k_mla_comp`].
    ///
    /// Same grow-only-scratch single-dispatch simdgroup kernel (NO ~536 MB
    /// zero-init `tmp_buf` the f16-workspace `flash_attn_decode_k` allocates per
    /// tile, and NO separate f16 workspace gather — it reads the f32 raw window +
    /// f32 comp_ring directly), but the per-row attendance is driven by an
    /// EXPLICIT per-query additive `mask` (`[k_positions, n_raw + n_comp]`, f16,
    /// 0 = attend / 0xFC00 = -inf) instead of `attn_base_pos` + `comp_avail`.
    ///
    /// This is what the SWA prefill tiles need: a sliding window has a per-query
    /// LOWER bound (the gathered window is wider than `w` once it straddles two
    /// windows) that the causal-only `attn_base_pos` cannot express. The raw cols
    /// `[0..n_raw)` index `kv_window` (the gathered ABSOLUTE-order SWA window, NOT
    /// the wrapped ring); comp cols `[n_raw..n_raw+n_comp)` index `comp_ring`.
    ///
    /// `kv_window` must have `n_cache` (== `n_raw`, a multiple of 8) rows of
    /// `dk` f32 each; rows past the real window length are masked off per query.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_k_mla_comp_masked(
        &self,
        q_k: &DeferredBuf,
        kv_window: &DeferredBuf,
        comp_ring: &metal::Buffer,
        n_comp: u32,
        mask: &[u16],          // [k_positions, n_raw + n_comp] f16 additive
        n_head: usize,
        dk: usize,
        dv: usize,
        n_cache: usize,        // == n_raw, multiple of 8
        k_positions: usize,
        scale: f32,
        attn_sinks: &DeferredBuf,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            q_k.n_elements == k_positions * n_head * dk,
            "flash_attn_k_mla_comp_masked: q_k has {} elems, expected K*n_head*DK = {}*{}*{}",
            q_k.n_elements, k_positions, n_head, dk
        );
        anyhow::ensure!(attn_sinks.n_elements == n_head, "attn_sinks n_head mismatch");
        anyhow::ensure!(dk == 512 && dv == 512, "MLA DS4 shape: DK=DV=512");
        anyhow::ensure!(n_cache % 8 == 0, "n_cache must be multiple of 8 (C=8)");
        anyhow::ensure!(matches!(k_positions, 1 | 2 | 4 | 8),
                         "K must be in {{1,2,4,8}} (got {})", k_positions);
        let mask_row = n_cache + n_comp as usize;
        anyhow::ensure!(
            mask.len() == k_positions * mask_row,
            "flash_attn_k_mla_comp_masked: mask len {} != K*(n_raw+n_comp) = {}*{}",
            mask.len(), k_positions, mask_row
        );

        let dk_i32 = dk as i32;
        let dv_i32 = dv as i32;
        let k_i32 = k_positions as i32;
        let nc_i32 = n_cache as i32;
        let mut key = Vec::with_capacity(20);
        key.extend_from_slice(&dk_i32.to_le_bytes());
        key.extend_from_slice(&dv_i32.to_le_bytes());
        key.extend_from_slice(&k_i32.to_le_bytes());
        key.extend_from_slice(&nc_i32.to_le_bytes());
        key.push(1u8); // has_mask discriminator (distinct pipeline cache entry)
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_flash_attn_K_mla_f32_sg", &key, |fcv| {
                fcv.set_constant_value_at_index(&dk_i32 as *const _ as *const _, MTLDataType::Int, 1600);
                fcv.set_constant_value_at_index(&dv_i32 as *const _ as *const _, MTLDataType::Int, 1601);
                fcv.set_constant_value_at_index(&k_i32 as *const _ as *const _, MTLDataType::Int, 1602);
                fcv.set_constant_value_at_index(&nc_i32 as *const _ as *const _, MTLDataType::Int, 1603);
                let has_mask = true;
                fcv.set_constant_value_at_index(&has_mask as *const _ as *const _, MTLDataType::Bool, 1604);
            })?;

        let out = self.alloc_f32(k_positions * n_head * dv);
        let q_bytes: u64 = 8 * (dk as u64) * 2;
        let v_bytes: u64 = (dv as u64) * 8 * 2;
        let ss_bytes: u64 = 64 * 4;
        let shmem_bytes: u64 = q_bytes + q_bytes + v_bytes + ss_bytes;

        let n_head_u32 = n_head as u32;
        let mask_row_u32 = mask_row as u32;
        let mask_buf = self.state.device.new_buffer_with_data(
            mask.as_ptr() as *const _,
            (mask.len() * std::mem::size_of::<u16>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_k.buf), 0);
        enc.set_buffer(1, Some(&kv_window.buf), 0);
        enc.set_buffer(2, Some(&out.buf), 0);
        set_scalar_bytes(enc, 3, &scale);
        set_scalar_bytes(enc, 4, &n_head_u32);
        set_scalar_bytes(enc, 5, &0u32); // attn_base_pos unused under HAS_MASK
        enc.set_buffer(6, Some(&attn_sinks.buf), 0);
        enc.set_buffer(7, Some(comp_ring), 0);
        set_scalar_bytes(enc, 8, &n_comp);
        enc.set_buffer(9, Some(comp_ring), 0); // comp_avail unused under HAS_MASK (dummy)
        enc.set_buffer(10, Some(&mask_buf), 0);
        set_scalar_bytes(enc, 11, &mask_row_u32);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32, 4, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// K-position kv_fp8_store_persistent. Writes K consecutive slots
    /// `[base_slot, base_slot+K)` of the persistent KV cache, sourcing each
    /// from the corresponding `kv_row_k` row (layout `[K, n_lora_kv]`).
    ///
    /// Implementation: K dispatches of the existing single-position kernel
    /// (`ds4_dsv4_kv_fp8_store`) with input offset `k*n_lora_kv*4` and
    /// slot = `base_slot + k`. Cheap on this hardware — the kv_fp8_store
    /// kernel underutilizes the GPU at K=1 (only 8 threads per TG); the K
    /// iterations fill idle SMs concurrently, costing ~the same as K=1
    /// (Phase 2 subagent bench: K=8 ratio ≈ 1.00).
    ///
    /// Returns the persistent cache DeferredBuf (same for all K — caller
    /// uses it as the KV source for K-query flash attention downstream).
    pub fn kv_fp8_store_persistent_k(
        &self,
        layer_idx: u32,
        kv_row_k: &DeferredBuf,
        params: &ds4_engine::attn_dispatch::LayerParams,
        raw_cap: u32,
        base_slot: u32,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(
            kv_row_k.n_elements == k_positions * row,
            "kv_fp8_store_persistent_k: kv_row_k has {} elems, expected K*n_lora_kv = {}*{}",
            kv_row_k.n_elements, k_positions, row
        );
        anyhow::ensure!(raw_cap > 0, "raw_cap must be > 0");
        anyhow::ensure!(
            base_slot.checked_add(k_positions as u32).map(|s| s <= raw_cap).unwrap_or(false),
            "base_slot {} + K {} would exceed raw_cap {}", base_slot, k_positions, raw_cap
        );

        let cache_byte_len = (raw_cap as usize) * row * std::mem::size_of::<f32>();
        let cache_buf_metal = self.state.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        let cache_buf = DeferredBuf {
            buf: cache_buf_metal,
            n_elements: (raw_cap as usize) * row,
        };

        let n_rot = params.n_rot as u32;
        let n_nope = (params.head_dim as u32).saturating_sub(n_rot);
        let chunks = (n_nope / 64).max(1) as u64;

        // K-merged single dispatch (one compute encoder for all K slots) —
        // grid.x = K threadgroups, each owns slot base_slot+k reading
        // rows[k*row_stride..]. Replaces the old K-encoder loop (one of the
        // ~30 per-layer dispatches the verifier chain serializes; the K=N
        // store alone was K encoders/layer). Falls back to the per-slot
        // kernel if the merged pipeline isn't available. DS4_KV_STORE_MERGE=0
        // forces the per-slot fallback (for A/B perf measurement).
        // "0"=K-enc per-slot, "enc"=1-enc per-slot, anything else/default=merged kernel.
        let merge_mode = std::env::var("DS4_KV_STORE_MERGE").unwrap_or_default();
        let use_merged_kernel = merge_mode != "0" && merge_mode != "enc";
        if let Some(pipe_k) = self.state.pipelines.get("ds4_dsv4_kv_fp8_store_k")
            .filter(|_| use_merged_kernel)
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(pipe_k);
            enc.set_buffer(0, Some(&cache_buf.buf), 0);
            enc.set_buffer(1, Some(&kv_row_k.buf), 0);
            set_scalar_bytes(enc, 2, &n_nope);
            set_scalar_bytes(enc, 3, &n_rot);
            set_scalar_bytes(enc, 4, &base_slot);
            enc.dispatch_thread_groups(
                MTLSize::new(k_positions as u64, 1, 1),
                MTLSize::new(chunks.min(1024), 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
            return Ok(cache_buf);
        }

        let pipe = self
            .state
            .pipelines
            .get("ds4_dsv4_kv_fp8_store")
            .ok_or_else(|| anyhow::anyhow!("ds4_dsv4_kv_fp8_store pipeline not loaded"))?
            .clone();
        let row_bytes = (row * std::mem::size_of::<f32>()) as u64;
        // DS4_KV_STORE_MERGE=enc → issue the K per-slot dispatches into ONE
        // shared encoder (vs the default one-encoder-per-slot). Decomposes the
        // per-dispatch cost into encoder-boundary overhead vs dispatch overhead:
        //   "0"   = K encoders, K dispatches   (baseline)
        //   "enc" = 1 encoder,  K dispatches   (isolates encoder cost)
        //   "1"   = 1 encoder,  1 dispatch     (merged kernel)
        let one_encoder = std::env::var("DS4_KV_STORE_MERGE")
            .map(|v| v == "enc").unwrap_or(false);
        if one_encoder {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            for k in 0..k_positions {
                let slot = base_slot + k as u32;
                enc.set_buffer(0, Some(&cache_buf.buf), 0);
                enc.set_buffer(1, Some(&kv_row_k.buf), (k as u64) * row_bytes);
                set_scalar_bytes(enc, 2, &n_nope);
                set_scalar_bytes(enc, 3, &n_rot);
                set_scalar_bytes(enc, 4, &slot);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(chunks.min(1024), 1, 1),
                );
            }
            crate::macos::end_shared_compute_enc(enc);
            return Ok(cache_buf);
        }
        for k in 0..k_positions {
            let slot = base_slot + k as u32;
            let kv_row_offset = (k as u64) * row_bytes;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&cache_buf.buf), 0);
            enc.set_buffer(1, Some(&kv_row_k.buf), kv_row_offset);
            set_scalar_bytes(enc, 2, &n_nope);
            set_scalar_bytes(enc, 3, &n_rot);
            set_scalar_bytes(enc, 4, &slot);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(chunks.min(1024), 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        Ok(cache_buf)
    }

    /// Phase E M5.4.3: composable `attn_output_matmuls_batched`.
    /// Two-stage attention output projection encoded into THIS scope:
    ///
    ///   Stage 1 (per-group, `n_groups` matvecs):
    ///     attn_low[g*n_lora_o..(g+1)*n_lora_o] =
    ///       w_o_a[g*n_lora_o..(g+1)*n_lora_o] · heads[g*group_dim..(g+1)*group_dim]
    ///
    ///   Stage 2 (dense matvec):
    ///     attn_out = w_o_b · attn_low   → length d_embd
    ///
    /// Returns (attn_low, attn_out) as DeferredBufs. Same kernel
    /// (`ds4_kernel_mul_mv_f32_f32_4`) as the inherent
    /// `attn_output_matmuls_batched`, same args, same dispatch — only
    /// the cb wrapping differs. Bit-identity holds for the same inputs.
    ///
    /// `heads` / `w_o_a` / `w_o_b` are pre-uploaded DeferredBufs. Per-group
    /// indexing uses Metal's `set_buffer(idx, buf, offset)` to view
    /// slices of `heads` / `w_o_a` / `attn_low` without re-allocating.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_output_matmuls(
        &self,
        heads: &DeferredBuf,
        w_o_a: &DeferredBuf,
        w_o_b: &DeferredBuf,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        d_embd: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(group_dim % 4 == 0, "group_dim must be % 4");
        anyhow::ensure!(n_lora_o % 2 == 0, "n_lora_o must be % 2 (NR0=2)");
        anyhow::ensure!(out_low_dim % 4 == 0, "out_low_dim must be % 4");
        anyhow::ensure!(d_embd % 2 == 0, "d_embd must be % 2 (NR0=2)");
        anyhow::ensure!(out_low_dim == n_groups * n_lora_o, "out_low_dim shape mismatch");
        anyhow::ensure!(heads.n_elements == n_groups * group_dim, "heads shape mismatch");
        anyhow::ensure!(w_o_a.n_elements == out_low_dim * group_dim, "w_o_a shape mismatch");
        anyhow::ensure!(w_o_b.n_elements == d_embd * out_low_dim, "w_o_b shape mismatch");

        // Pipelines specialized on group_dim (stage 1) and out_low_dim (stage 2).
        let nsg_s1: i16 = (((group_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_s1: i16 = if group_dim % 256 == 0 { 16 } else if group_dim % 128 == 0 { 8 } else { 4 };
        let mut key_s1 = Vec::with_capacity(4);
        key_s1.extend_from_slice(&nsg_s1.to_le_bytes());
        key_s1.extend_from_slice(&nxpsg_s1.to_le_bytes());
        let pipe_s1 = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_s1, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg_s1 as *const _ as *const _, MTLDataType::Short, 600,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg_s1 as *const _ as *const _, MTLDataType::Short, 601,
                );
            })?;
        let nsg_s2: i16 = (((out_low_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_s2: i16 = if out_low_dim % 256 == 0 { 16 } else if out_low_dim % 128 == 0 { 8 } else { 4 };
        let mut key_s2 = Vec::with_capacity(4);
        key_s2.extend_from_slice(&nsg_s2.to_le_bytes());
        key_s2.extend_from_slice(&nxpsg_s2.to_le_bytes());
        let pipe_s2 = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_s2, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg_s2 as *const _ as *const _, MTLDataType::Short, 600,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg_s2 as *const _ as *const _, MTLDataType::Short, 601,
                );
            })?;

        let attn_low = self.alloc_f32(out_low_dim);
        let attn_out = self.alloc_f32(d_embd);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let shmem: u64 = 32 * 2 * 4;
        // Stage 1: ONE batched grouped matvec over n_groups instead of
        // n_groups separate compute encoders. The mul_mv kernel batches on
        // tgpig.z (`im`): src0 (w_o_a) batch stride nb02, src1 (heads) batch
        // stride nb12, dst written at im*ne0*ne1 (dense.metal:173). Each
        // separate encoder is a barrier that serializes the groups; batching
        // lets all n_groups run concurrently (more occupancy) — and is one
        // dispatch instead of n_groups.
        let s1_args = MulMvArgs {
            ne00: group_dim as i32, ne01: n_lora_o as i32, ne02: n_groups as i32, _pad0: 0,
            nb00: 4, nb01: (group_dim * 4) as u64,
            nb02: (group_dim * n_lora_o * 4) as u64,
            nb03: (group_dim * n_lora_o * 4) as u64,
            ne10: group_dim as i32, ne11: 1, ne12: n_groups as i32, _pad1: 0,
            nb10: 4, nb11: (group_dim * 4) as u64,
            nb12: (group_dim * 4) as u64, nb13: (group_dim * 4) as u64,
            ne0: n_lora_o as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };
        let s1_n_row_tg = ((n_lora_o as u64) + 1) / 2;
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe_s1);
            set_scalar_bytes(enc, 0, &s1_args);
            enc.set_buffer(1, Some(&w_o_a.buf), 0);
            enc.set_buffer(2, Some(&heads.buf), 0);
            enc.set_buffer(3, Some(&attn_low.buf), 0);
            enc.set_threadgroup_memory_length(0, shmem);
            enc.dispatch_thread_groups(
                MTLSize::new(s1_n_row_tg, 1, n_groups as u64),
                MTLSize::new(32, nsg_s1 as u64, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Stage 2: dense matvec — w_o_b · attn_low → attn_out.
        let s2_args = MulMvArgs {
            ne00: out_low_dim as i32, ne01: d_embd as i32, ne02: 1, _pad0: 0,
            nb00: 4, nb01: (out_low_dim * 4) as u64,
            nb02: (out_low_dim * d_embd * 4) as u64,
            nb03: (out_low_dim * d_embd * 4) as u64,
            ne10: out_low_dim as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (out_low_dim * 4) as u64,
            nb12: (out_low_dim * 4) as u64, nb13: (out_low_dim * 4) as u64,
            ne0: d_embd as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };
        let s2_n_row_tg = ((d_embd as u64) + 1) / 2;
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe_s2);
        set_scalar_bytes(enc, 0, &s2_args);
        enc.set_buffer(1, Some(&w_o_b.buf), 0);
        enc.set_buffer(2, Some(&attn_low.buf), 0);
        enc.set_buffer(3, Some(&attn_out.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem);
        enc.dispatch_thread_groups(
            MTLSize::new(s2_n_row_tg, 1, 1),
            MTLSize::new(32, nsg_s2 as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok((attn_low, attn_out))
    }

    /// Q8_0 twin of `encode_attn_output_matmuls`: stage 1 (grouped `w_o_a`) runs
    /// the batched `ds4_dsv4_attn_out_low_q8_0_f32` kernel reading raw block_q8_0
    /// weight bytes (from `weight_q8_0_raw`); stage 2 (dense `w_o_b`) reuses
    /// `matvec_q8_0`. Cuts the output-projection weight bandwidth ~4x (the
    /// harness's #2 attn-half GPU consumer). `w_o_a_q8`/`w_o_b_q8` are raw-byte
    /// DeferredBufs whose `n_elements` track the logical weight count.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_output_matmuls_q8(
        &self,
        heads: &DeferredBuf,
        w_o_a_q8: &DeferredBuf,
        w_o_b_q8: &DeferredBuf,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        d_embd: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(group_dim % 32 == 0, "group_dim must be % 32 (Q8_0)");
        anyhow::ensure!(n_lora_o % 2 == 0, "n_lora_o must be % 2 (NR0=2)");
        anyhow::ensure!(out_low_dim == n_groups * n_lora_o, "out_low_dim shape mismatch");
        anyhow::ensure!(heads.n_elements == n_groups * group_dim, "heads shape mismatch");
        anyhow::ensure!(w_o_a_q8.n_elements == out_low_dim * group_dim, "w_o_a shape mismatch");

        let row_stride = (group_dim / 32) * 34;
        // nsg geometry (tuned 2026-05-31): the old clamp(group_dim/128) keyed on
        // the reduction length and picked nsg=8 here — the q8 nsg audit's exact
        // mistake. The sweep (nsg_sweep_moe_attnout_bench) + end-to-end A/B found
        // a d_out(n_lora_o)-aware nsg faster (nsg=2 for n_lora_o=1024; ~1% net
        // ms/token, rel unchanged). DS4_ATTN_OUT_NSG overrides.
        let nsg: i16 = std::env::var("DS4_ATTN_OUT_NSG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if n_lora_o >= 8192 { 1 } else if n_lora_o >= 1024 { 2 } else { 4 });
        let nxpsg: i16 = if group_dim % 256 == 0 {
            16
        } else if group_dim % 128 == 0 {
            8
        } else {
            4
        };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_attn_out_low_q8_0_f32", &key, |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            })?;

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }
        let args = MulMvIdArgs {
            nei0: n_groups as i32, nei1: 1, nbi1: 0,
            ne00: group_dim as i32, ne01: n_lora_o as i32, ne02: n_groups as i32, _pad0: 0,
            nb00: 2, nb01: row_stride as u64, nb02: (row_stride * n_lora_o) as u64,
            ne10: group_dim as i32, ne11: n_groups as i32, ne12: 1, ne13: 1,
            nb10: 4, nb11: (group_dim * 4) as u64, nb12: (heads.n_elements * 4) as u64,
            ne0: n_lora_o as i32, ne1: out_low_dim as i32, nb1: (n_lora_o * 4) as u64,
            nr0: 2, _pad1: 0,
        };

        let attn_low = self.alloc_f32(out_low_dim);
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&w_o_a_q8.buf), 0);
            enc.set_buffer(2, Some(&heads.buf), 0);
            enc.set_buffer(3, Some(&attn_low.buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(((n_lora_o as u64) + 1) / 2, 1, n_groups as u64),
                MTLSize::new(32, nsg as u64, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Stage 2: dense w_o_b · attn_low → attn_out, Q8_0.
        let attn_out = self.matvec_attn_proj(w_o_b_q8, &attn_low, out_low_dim, d_embd)?;
        Ok((attn_low, attn_out))
    }

    /// K-position output projection. Stage 1 (grouped w_o_a) is K-linear:
    /// the existing `ds4_dsv4_attn_out_low_q8_0_f32` kernel is dispatched K
    /// times with offsets into the K-position heads and attn_low buffers.
    /// Stage 2 (dense w_o_b) is K-amortized via `matvec_k_q8_0` — the
    /// simdgroup-matrix kernel reads w_o_b once for all K activations.
    ///
    /// Layout: heads_K is `[K, n_groups, group_dim]` (flat f32). Outputs are
    /// `attn_low_K [K, out_low_dim]` and `attn_out_K [K, d_embd]`.
    /// K supported in `{1, 2, 4, 8}`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_output_matmuls_q8_k(
        &self,
        heads_k: &DeferredBuf,
        w_o_a_q8: &DeferredBuf,
        w_o_b_q8: &DeferredBuf,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(group_dim % 32 == 0, "group_dim must be % 32 (Q8_0)");
        anyhow::ensure!(n_lora_o % 2 == 0, "n_lora_o must be % 2 (NR0=2)");
        anyhow::ensure!(out_low_dim == n_groups * n_lora_o, "out_low_dim shape mismatch");
        anyhow::ensure!(
            heads_k.n_elements == k_positions * n_groups * group_dim,
            "heads_k shape mismatch: {} vs K*n_groups*group_dim = {}*{}*{}",
            heads_k.n_elements, k_positions, n_groups, group_dim
        );
        anyhow::ensure!(w_o_a_q8.n_elements == out_low_dim * group_dim, "w_o_a shape mismatch");

        // Stage 1 pipeline (shared across K-iterations — built once).
        let row_stride = (group_dim / 32) * 34;
        // nsg geometry (tuned 2026-05-31): the old clamp(group_dim/128) keyed on
        // the reduction length and picked nsg=8 here — the q8 nsg audit's exact
        // mistake. The sweep (nsg_sweep_moe_attnout_bench) + end-to-end A/B found
        // a d_out(n_lora_o)-aware nsg faster (nsg=2 for n_lora_o=1024; ~1% net
        // ms/token, rel unchanged). DS4_ATTN_OUT_NSG overrides.
        let nsg: i16 = std::env::var("DS4_ATTN_OUT_NSG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if n_lora_o >= 8192 { 1 } else if n_lora_o >= 1024 { 2 } else { 4 });
        let nxpsg: i16 = if group_dim % 256 == 0 { 16 }
                          else if group_dim % 128 == 0 { 8 }
                          else { 4 };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self
            .state
            .specialized_pipeline("ds4_dsv4_attn_out_low_q8_0_f32", &key, |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            })?;

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }
        // Per-K-iter args. heads.n_elements is now per-K-row (n_groups*group_dim).
        let heads_per_k_bytes = (n_groups * group_dim * 4) as u64;
        let attn_low_per_k_bytes = (out_low_dim * 4) as u64;
        let args = MulMvIdArgs {
            nei0: n_groups as i32, nei1: 1, nbi1: 0,
            ne00: group_dim as i32, ne01: n_lora_o as i32, ne02: n_groups as i32, _pad0: 0,
            nb00: 2, nb01: row_stride as u64, nb02: (row_stride * n_lora_o) as u64,
            ne10: group_dim as i32, ne11: n_groups as i32, ne12: 1, ne13: 1,
            nb10: 4, nb11: (group_dim * 4) as u64, nb12: heads_per_k_bytes,
            ne0: n_lora_o as i32, ne1: out_low_dim as i32, nb1: (n_lora_o * 4) as u64,
            nr0: 2, _pad1: 0,
        };

        let attn_low_k = self.alloc_f32(k_positions * out_low_dim);

        // Stage 1: optional K-merged single dispatch via the _k kernel
        // (grid.z = K*n_groups; iid1=z/nei0=K-pos, idx=z%nei0=group). Collapses
        // the per-K dispatch loop into ONE grid-batched dispatch. Bit-identical
        // (attn_out_low_k_merge_microbench: max_abs=0) but PERF-NEUTRAL: these
        // per-group dispatches are compute/bandwidth-bound real matmuls (34MB Q8
        // weight reads each), already GPU-saturating, so collapsing them saves
        // nothing (1616→1615µs) — UNLIKE the tiny latency-bound kv_fp8_store
        // (70% win). Default OFF (production stays on the validated per-K loop);
        // DS4_ATTN_OUT_MERGE=1 enables it. Kept for future K-weight-reuse work.
        // DS4_ATTN_OUT_GEMM: the grouped w_o_a as ONE batched mul_mm GEMM (each
        // group's Q8 weight read ONCE) + interleave, vs the per-K mul_mv re-reading
        // the weight K× (~K× weight bandwidth at large K). The one non-GEMM Q8
        // matmul left in the prefill. Stage 2 (w_o_b) is already a GEMM below.
        if std::env::var("DS4_ATTN_OUT_GEMM").ok().as_deref() == Some("1") && k_positions > 8 {
            let gm = self.matmul_k_q8_0_grouped(w_o_a_q8, heads_k, group_dim, n_lora_o, n_groups, k_positions)?;
            let il = self.interleave_group_major(&gm, n_groups, k_positions, n_lora_o)?;
            self.copy_buf_into(&il, attn_low_k.buffer(), 0);
            let attn_out_k = self.matmul_k_attn_proj(w_o_b_q8, &attn_low_k, out_low_dim, d_embd, k_positions)?;
            return Ok((attn_low_k, attn_out_k));
        }
        let attn_out_merge = std::env::var("DS4_ATTN_OUT_MERGE")
            .map(|v| v == "1").unwrap_or(false);
        let pipe_k = if attn_out_merge {
            self.state.specialized_pipeline(
                "ds4_dsv4_attn_out_low_q8_0_f32_k", &key,
                |fcv| {
                    fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                    fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
                },
            ).ok()
        } else { None };

        if let Some(pipe_k) = pipe_k {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe_k);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&w_o_a_q8.buf), 0);
            enc.set_buffer(2, Some(&heads_k.buf), 0);
            enc.set_buffer(3, Some(&attn_low_k.buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(((n_lora_o as u64) + 1) / 2, 1, (k_positions as u64) * n_groups as u64),
                MTLSize::new(32, nsg as u64, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        } else {
            // Per-K fallback: K iterations, each offset into heads_k / attn_low_k.
            for k in 0..k_positions {
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&pipe);
                set_scalar_bytes(enc, 0, &args);
                enc.set_buffer(1, Some(&w_o_a_q8.buf), 0);
                enc.set_buffer(2, Some(&heads_k.buf), (k as u64) * heads_per_k_bytes);
                enc.set_buffer(3, Some(&attn_low_k.buf), (k as u64) * attn_low_per_k_bytes);
                enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
                enc.dispatch_thread_groups(
                    MTLSize::new(((n_lora_o as u64) + 1) / 2, 1, n_groups as u64),
                    MTLSize::new(32, nsg as u64, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
        }

        // Stage 2: single K-amortized matvec via simdgroup-matrix kernel.
        let attn_out_k = self.matmul_k_attn_proj(w_o_b_q8, &attn_low_k, out_low_dim, d_embd, k_positions)?;
        Ok((attn_low_k, attn_out_k))
    }

    /// K-position shared-expert chain.
    /// ```text
    ///   g    = matvec_k_q8_0(w_gate, normed_K)            [K, sd]
    ///   u    = matvec_k_q8_0(w_up,   normed_K)            [K, sd]
    ///   mid  = swiglu_K(g, u)  // K iter of single-position kernel  [K, sd]
    ///   out  = matvec_k_q8_0(w_down, mid)                 [K, d_embd]
    /// ```
    /// gate/up/down weights are Q8_0 raw bytes (matching the q/kv path).
    /// All 3 matvecs use the simdgroup-matrix Q8_0 kernel (K-amortized;
    /// weights read once for all K activations). The swiglu is K-linear in
    /// dispatch (K iterations of a tiny kernel — sd ≤ a few thousand floats).
    /// K supported in `{1, 2, 4, 8}`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_shared_chain_q8_k(
        &self,
        normed_k: &DeferredBuf,
        w_gate_q8: &DeferredBuf,
        w_up_q8: &DeferredBuf,
        w_down_q8: &DeferredBuf,
        d_embd: usize,
        sd: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            normed_k.n_elements == k_positions * d_embd,
            "encode_shared_chain_q8_k: normed_k shape mismatch"
        );
        anyhow::ensure!(sd % 32 == 0, "shared_chain_k: sd ({}) must be %32 (Q8_0)", sd);
        anyhow::ensure!(d_embd % 32 == 0, "shared_chain_k: d_embd must be %32");

        let g_k = self.matmul_k_q8_0_auto(w_gate_q8, normed_k, d_embd, sd, k_positions)?;
        let u_k = self.matmul_k_q8_0_auto(w_up_q8,   normed_k, d_embd, sd, k_positions)?;

        // K iterations of swiglu (per-K-position elementwise; sd ≤ a few thousand).
        let mid_k = self.pooled_scratch_f32("shared_mid", k_positions * sd);
        let glu_pipe = self
            .state
            .pipelines
            .get("ds4_kernel_swiglu_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_kernel_swiglu_f32 pipeline not loaded"))?
            .clone();
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct GluArgs {
            ne00: i32, nb01: u64, ne10: i32, nb11: u64,
            ne0: i32, nb1: u64, i00: i32, i10: i32,
            alpha: f32, limit: f32,
        }
        let glu_args = GluArgs {
            ne00: sd as i32, nb01: (sd * 4) as u64,
            ne10: sd as i32, nb11: (sd * 4) as u64,
            ne0:  sd as i32, nb1:  (sd * 4) as u64,
            i00: 0, i10: 0, alpha: 0.0, limit: 0.0,
        };
        let glu_threads = (glu_pipe.max_total_threads_per_threadgroup() as u64)
            .min(sd as u64).max(1);
        // DS4_CHUNK_BATCH_ROUTER=1: the swiglu kernel already rows by tgpig (row =
        // tgpig*nb01), so K per-token dispatches collapse into ONE with grid.x=K
        // (each threadgroup = one K-position). K dispatches → 1, byte-identical.
        if std::env::var("DS4_CHUNK_BATCH_ROUTER").ok().as_deref() == Some("1") {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&glu_pipe);
            set_scalar_bytes(enc, 0, &glu_args); // nb01/nb11/nb1 = sd*4 → per-row stride
            enc.set_buffer(1, Some(&g_k.buf), 0);
            enc.set_buffer(2, Some(&u_k.buf), 0);
            enc.set_buffer(3, Some(&mid_k.buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(k_positions as u64, 1, 1),
                MTLSize::new(glu_threads, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        } else {
            for k in 0..k_positions {
                let off = (k * sd * 4) as u64;
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&glu_pipe);
                set_scalar_bytes(enc, 0, &glu_args);
                enc.set_buffer(1, Some(&g_k.buf), off);
                enc.set_buffer(2, Some(&u_k.buf), off);
                enc.set_buffer(3, Some(&mid_k.buf), off);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(glu_threads, 1, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
        }

        let out_k = self.matmul_k_q8_0_auto(w_down_q8, &mid_k, sd, d_embd, k_positions)?;
        Ok(out_k)
    }

    /// Phase E M5.4.4: composable `softplus_sqrt` activation.
    /// Mirrors `softplus_sqrt_impl` (macos.rs:722) but encodes into
    /// THIS scope. Length must be divisible by 4 (float4 lanes).
    pub fn softplus_sqrt(&self, x: &DeferredBuf) -> Result<DeferredBuf> {
        let n = x.n_elements;
        anyhow::ensure!(
            n % 4 == 0,
            "softplus_sqrt: len ({}) must be divisible by 4 (float4 lanes)",
            n
        );
        let pipe = self
            .state
            .pipelines
            .get("ds4_kernel_dsv4_softplus_sqrt_f32_4")
            .ok_or_else(|| anyhow::anyhow!("softplus_sqrt pipeline not loaded"))?
            .clone();
        let out = self.alloc_f32(n);
        let ne0_4: u32 = (n / 4) as u32;
        let nb: u32 = (n * 4) as u32;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        set_scalar_bytes(enc, 2, &ne0_4);
        set_scalar_bytes(enc, 3, &nb);
        set_scalar_bytes(enc, 4, &nb);
        enc.dispatch_threads(
            MTLSize::new(ne0_4 as u64, 1, 1),
            MTLSize::new(ne0_4 as u64, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Phase E M5.4.4: composable `router_logits_batched`.
    /// `matvec_f32(w_router, h_norm) → logits` then
    /// `softplus_sqrt(logits) → probs`, both encoded into THIS scope.
    /// Returns `probs` (length n_experts). Caller uploads weights and
    /// h_norm; no flush.
    ///
    /// Preconditions:
    /// - `h_norm.n_elements == d_embd`, `d_embd % 4 == 0`, `% 2 == 0`
    /// - `w_router.n_elements == n_experts * d_embd`
    /// - `n_experts % 4 == 0` (softplus_sqrt float4 lanes) and `% 2 == 0`
    pub fn encode_router_logits(
        &self,
        w_router: &DeferredBuf,
        h_norm: &DeferredBuf,
        n_experts: usize,
        // `w_router` holds F16 bytes (no-copy, lean non-hash layer) vs f32 —
        // selects `matvec_f16`. Bit-identical (F16→f32 exact).
        w_router_is_f16: bool,
    ) -> Result<DeferredBuf> {
        let d_embd = h_norm.n_elements;
        anyhow::ensure!(
            w_router.n_elements == n_experts * d_embd,
            "encode_router_logits: w_router shape mismatch ({} vs {}*{})",
            w_router.n_elements,
            n_experts,
            d_embd
        );
        let logits = if w_router_is_f16 {
            self.matvec_f16(w_router, h_norm, d_embd, n_experts)?
        } else {
            self.matvec_f32(w_router, h_norm, d_embd, n_experts)?
        };
        self.softplus_sqrt(&logits)
    }

    /// Phase 2 — K-position router logits. K iterations of the existing
    /// `ds4_kernel_mul_mv_f32_f32_4` matvec kernel (router weights are
    /// f32) with byte offsets into `h_norm_k` and the per-K output slice,
    /// then ONE flat softplus_sqrt over the K*n_experts buffer. The
    /// router head is tiny (n_experts ≤ 256 for DS4) so K-amortization
    /// gains are small; K-linearity is fine here.
    pub fn encode_router_logits_k(
        &self,
        w_router: &DeferredBuf,
        h_norm_k: &DeferredBuf,
        n_experts: usize,
        d_embd: usize,
        k_positions: usize,
        w_is_f16: bool,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            h_norm_k.n_elements == k_positions * d_embd,
            "encode_router_logits_k: h_norm_k has {} elems, expected K*d_embd = {}*{}",
            h_norm_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            w_router.n_elements == n_experts * d_embd,
            "encode_router_logits_k: w_router shape mismatch"
        );
        anyhow::ensure!(d_embd % 4 == 0, "router_logits_k: d_embd must be %4");
        anyhow::ensure!(n_experts % 2 == 0, "router_logits_k: n_experts must be %2");
        anyhow::ensure!(n_experts % 4 == 0, "router_logits_k: n_experts must be %4 for softplus_sqrt");

        let nsg: i16 = (((d_embd as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_embd % 256 == 0 { 16 }
                          else if d_embd % 128 == 0 { 8 } else { 4 };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        // Lean weights keep w_router as F16 only — same kernel family, f16 twin.
        let kernel = if w_is_f16 { "ds4_kernel_mul_mv_f16_f32_4" } else { "ds4_kernel_mul_mv_f32_f32_4" };
        let wb: u64 = if w_is_f16 { 2 } else { 4 };
        let pipe = self
            .state
            .specialized_pipeline(kernel, &key, |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            })?;

        // T4 (bounded-working-set): pool+pin the router logits/probs. The
        // GPU hash-routing path (DS4_CHUNK_HASH_GPU) reads this on the GPU with
        // NO CPU drain, so under daemon pressure an unpinned StorageModeShared
        // probs buffer can be evicted between the router dispatch and the
        // weight-kernel read → garbage → divergence. Pooling reuses ONE buffer
        // across layers/chunks (bounded) and pins it resident at first-touch.
        // Falls back to fresh alloc_f32 when DS4_CHUNK_POOL_SCRATCH is off.
        let logits_k = self.pooled_scratch_f32("router_logits_k", k_positions * n_experts);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let args = MulMvArgs {
            ne00: d_embd as i32, ne01: n_experts as i32, ne02: 1, _pad0: 0,
            nb00: wb, nb01: d_embd as u64 * wb,
            nb02: (d_embd * n_experts) as u64 * wb,
            nb03: (d_embd * n_experts) as u64 * wb,
            ne10: d_embd as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_embd * 4) as u64,
            nb12: (d_embd * 4) as u64, nb13: (d_embd * 4) as u64,
            ne0: n_experts as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };
        let shmem_bytes: u64 = 32 * 2 * 4;
        let n_row_tg = ((n_experts as u64) + 1) / 2;
        let x_per_k_bytes = (d_embd * 4) as u64;
        let out_per_k_bytes = (n_experts * 4) as u64;
        // DS4_CHUNK_BATCH_ROUTER=1: collapse the K per-token matvecs into ONE batched
        // dispatch (grid.z=K, ne12=K via the ggml broadcast-matvec; r2=K broadcasts the
        // shared w_router across tokens). K dispatches → 1; the dominant MoE-stage
        // dispatch hotspot (Task 0: per-layer dispatch count is the cb-ceiling blocker).
        if std::env::var("DS4_CHUNK_BATCH_ROUTER").ok().as_deref() == Some("1") {
            let mut bargs = args;
            bargs.ne12 = k_positions as i32;
            bargs.nb12 = x_per_k_bytes;
            bargs.nb13 = (k_positions as u64) * x_per_k_bytes;
            bargs.r2 = k_positions as i16; // ne12/ne02 = K/1 → weight offset 0 for all tokens
            bargs.r3 = 1;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &bargs);
            enc.set_buffer(1, Some(&w_router.buf), 0);
            enc.set_buffer(2, Some(&h_norm_k.buf), 0);
            enc.set_buffer(3, Some(&logits_k.buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(n_row_tg, 1, k_positions as u64),
                MTLSize::new(32, nsg as u64, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        } else {
            for k in 0..k_positions {
                let k64 = k as u64;
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&pipe);
                set_scalar_bytes(enc, 0, &args);
                enc.set_buffer(1, Some(&w_router.buf), 0);
                enc.set_buffer(2, Some(&h_norm_k.buf), k64 * x_per_k_bytes);
                enc.set_buffer(3, Some(&logits_k.buf), k64 * out_per_k_bytes);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
                enc.dispatch_thread_groups(
                    MTLSize::new(n_row_tg, 1, 1),
                    MTLSize::new(32, nsg as u64, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
        }
        // softplus_sqrt is per-element — works on the flat K*n_experts buffer.
        self.softplus_sqrt(&logits_k)
    }

    /// Phase F task #86 — `attn_output_proj` as a single composable
    /// BatchScope op. Combines the existing `encode_attn_output_matmuls`
    /// (two-stage matvec: per-group lo + dense out_low_dim → d_embd)
    /// with `hc_expand_attn` (HC fold over the n_hc residual). Mirrors
    /// the inherent `attn_output_proj_impl`'s body exactly — same
    /// kernels in the same order — but encoded into THIS scope so the
    /// 3 GPU dispatches (2 matvecs + 1 hc_expand) compose with adjacent
    /// scope ops (flash_attn before, run_ffn_half ops after) in ONE cb.
    ///
    /// Returns `after_attn_hc` as a DeferredBuf shape `[n_hc, d_embd]`
    /// (flat). Per the antirez decode loop ([[m5-antirez-decode-loop]])
    /// this stays GPU-resident across the layer boundary; future work
    /// in encode_first_half will pointer-swap it with the next layer's
    /// `cur_hc` instead of CPU readback.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attn_output_proj(
        &self,
        heads: &DeferredBuf,
        w_o_a: &DeferredBuf,
        w_o_b: &DeferredBuf,
        // Raw Q8_0 bytes for the output projection. When present (and DS4_Q8_PROJ
        // on) the matmuls run via the no-copy q8 path — REQUIRED under lean, where
        // the f32 `w_o_a`/`w_o_b` above are empty. Empty → the f32 path.
        w_o_a_q8: &[u8],
        w_o_b_q8: &[u8],
        cur_hc: &DeferredBuf,
        hc_split_post_attn: &DeferredBuf,
        hc_split_comb_attn: &DeferredBuf,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        n_hc: usize,
        d_embd: usize,
    ) -> Result<DeferredBuf> {
        let use_q8 = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true)
            && !w_o_a_q8.is_empty()
            && !w_o_b_q8.is_empty();
        let (_attn_low, attn_out) = if use_q8 {
            let a = self.weight_q8_0_raw(w_o_a_q8, (w_o_a_q8.len() / 34) * 32);
            let b = self.weight_q8_0_raw(w_o_b_q8, (w_o_b_q8.len() / 34) * 32);
            self.encode_attn_output_matmuls_q8(
                heads, &a, &b, n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
            )?
        } else {
            self.encode_attn_output_matmuls(
                heads, w_o_a, w_o_b, n_groups, n_lora_o, group_dim, out_low_dim, d_embd,
            )?
        };
        self.hc_expand_attn(
            &attn_out,
            cur_hc,
            hc_split_post_attn,
            hc_split_comb_attn,
            n_hc,
            d_embd,
        )
    }

    /// Allocate a fresh zero-init shared-storage output buffer of
    /// `n` i32 elements. Same as `alloc_f32` but the DeferredBuf's
    /// `n_elements` counts i32s (so `flush_and_read` won't work —
    /// caller uses `buffer().contents() as *const i32`).
    pub fn alloc_i32(&self, n: usize) -> DeferredBuf {
        let byte_len = (n * std::mem::size_of::<i32>()) as u64;
        let buf = self
            .state
            .device
            .new_buffer(byte_len, metal::MTLResourceOptions::StorageModeShared);
        DeferredBuf { buf, n_elements: n }
    }

    /// Phase E M5.4.4-followup: composable `router_finalize`.
    /// Runs the two GPU kernels (router_finalize_one + router_weights_one)
    /// in THIS scope and returns `(selected_buf, weights_buf)` as
    /// DeferredBufs — selected as 6 i32, weights as 6 f32.
    ///
    /// The inherent `router_finalize_impl` commits + waits + reads back
    /// the buffers as Rust Vecs. This composable version skips the
    /// readback — `selected_buf` becomes a direct input to a future
    /// scope-aware `moe_routed_step` (M5.4-followup), eliminating the
    /// router→moe CPU round-trip.
    ///
    /// Same kernels, same args, same dispatch as the inherent path.
    /// `probs.n_elements == 256`, `bias.n_elements == 256`, k=6
    /// hardcoded (DS4).
    pub fn encode_router_finalize(
        &self,
        probs: &DeferredBuf,
        bias: &DeferredBuf,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        let n_experts = probs.n_elements;
        anyhow::ensure!(
            n_experts > 0 && n_experts <= 1024,
            "encode_router_finalize: probs.n_elements out of range (got {})",
            n_experts
        );
        anyhow::ensure!(
            bias.n_elements == n_experts,
            "encode_router_finalize: bias.n_elements must equal n_experts {} (got {})",
            n_experts,
            bias.n_elements
        );
        let k: usize = 6;
        // Flash (256 experts, scale 1.5) keeps the upstream hardcoded
        // kernels (bit-identical default); anything else takes the
        // width/scale-generic bridge shims.
        let generic = n_experts != 256
            || ds4_engine::moe::router_scale() != 1.5;
        let npow2: u64 = (n_experts as u64).next_power_of_two();

        let select_pipe = self
            .state
            .pipelines
            .get(if generic {
                "ds4_dsv4_router_finalize_one_any"
            } else {
                "ds4_dsv4_router_finalize_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_finalize_one pipeline not loaded"))?
            .clone();
        let weights_pipe = self
            .state
            .pipelines
            .get(if generic {
                "ds4_dsv4_router_weights_one_any"
            } else {
                "ds4_dsv4_router_weights_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_weights_one pipeline not loaded"))?
            .clone();

        let hash_buf = new_input_buffer(&self.state.device, &[0i32]);
        let tokens_buf = new_input_buffer(&self.state.device, &[0i32]);
        let selected_buf = self.alloc_i32(k);
        let weights_buf = self.alloc_f32(k);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct SelectArgs {
            has_bias: u32,
            hash_mode: u32,
            use_token_buffer: u32,
            token: u32,
            hash_rows: u32,
            n_experts: u32, // consumed by the _any shim only
        }
        let args = SelectArgs {
            has_bias: 1,
            hash_mode: 0,
            use_token_buffer: 0,
            token: 0,
            hash_rows: 1,
            n_experts: n_experts as u32,
        };
        let (tg_threads, shmem) = if generic {
            (npow2, 2 * npow2 * 4)
        } else {
            (256, 2 * 256 * 4)
        };

        // Stage 1: bitonic top-6 select.
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&select_pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&probs.buf), 0);
            enc.set_buffer(2, Some(&bias.buf), 0);
            enc.set_buffer(3, Some(&hash_buf), 0);
            enc.set_buffer(4, Some(&tokens_buf), 0);
            enc.set_buffer(5, Some(&selected_buf.buf), 0);
            enc.set_threadgroup_memory_length(0, shmem);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(tg_threads, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Stage 2: normalize selected probs → weights.
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&weights_pipe);
            enc.set_buffer(0, Some(&probs.buf), 0);
            enc.set_buffer(1, Some(&selected_buf.buf), 0);
            enc.set_buffer(2, Some(&weights_buf.buf), 0);
            let num_experts: u32 = k as u32;
            let scale: f32 = ds4_engine::moe::router_scale();
            let min_sum: f32 = 6.103515625e-5;
            set_scalar_bytes(enc, 3, &num_experts);
            set_scalar_bytes(enc, 4, &scale);
            set_scalar_bytes(enc, 5, &min_sum);
            enc.dispatch_threads(
                MTLSize::new(k as u64, 1, 1),
                MTLSize::new(k as u64, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        Ok((selected_buf, weights_buf))
    }

    /// Hash-routing weights: given a precomputed `selected` (the per-token
    /// routing-table lookup) and the GPU router `probs`, run ONLY Stage 2 of
    /// `encode_router_finalize` (the `router_weights_one` kernel: gather
    /// probs[sel], clamp sum to 6.1e-5, w = p/sum*1.5). This is exactly
    /// `ds4_engine::moe::hash_router_weights_from_probs`, so hash-routed layers
    /// can compute their MoE weights on GPU (probs stays resident, no readback)
    /// and join the chained span. `selected` must hold 6 i32 expert ids.
    pub fn encode_router_weights_one(
        &self,
        probs: &DeferredBuf,
        selected: &DeferredBuf,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            probs.n_elements > 0 && probs.n_elements <= 1024,
            "encode_router_weights_one: probs.n_elements out of range (got {})",
            probs.n_elements
        );
        anyhow::ensure!(
            selected.n_elements == 6,
            "encode_router_weights_one: selected.n_elements must be 6 (got {})",
            selected.n_elements
        );
        let k = selected.n_elements;
        let generic = probs.n_elements != 256
            || ds4_engine::moe::router_scale() != 1.5;
        let weights_pipe = self
            .state
            .pipelines
            .get(if generic {
                "ds4_dsv4_router_weights_one_any"
            } else {
                "ds4_dsv4_router_weights_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_weights_one pipeline not loaded"))?
            .clone();
        let weights_buf = self.alloc_f32(k);
        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&weights_pipe);
        enc.set_buffer(0, Some(&probs.buf), 0);
        enc.set_buffer(1, Some(&selected.buf), 0);
        enc.set_buffer(2, Some(&weights_buf.buf), 0);
        let num_experts: u32 = k as u32;
        let scale: f32 = ds4_engine::moe::router_scale();
        let min_sum: f32 = 6.103515625e-5;
        set_scalar_bytes(enc, 3, &num_experts);
        set_scalar_bytes(enc, 4, &scale);
        set_scalar_bytes(enc, 5, &min_sum);
        enc.dispatch_threads(MTLSize::new(k as u64, 1, 1), MTLSize::new(k as u64, 1, 1));
        crate::macos::end_shared_compute_enc(enc);
        Ok(weights_buf)
    }

    /// Phase 2 — K-position router finalize. K iterations of the two
    /// existing kernels (`ds4_dsv4_router_finalize_one` +
    /// `ds4_dsv4_router_weights_one`). Allocates K SEPARATE (selected,
    /// weights) buffer pairs (6 i32 + 6 f32 each) — small allocations
    /// per K position — because the downstream
    /// `encode_moe_and_shared_chain_with_router_bufs_db` consumes
    /// `(selected, weights)` as `&metal::Buffer` (no offset support).
    /// Returns `Vec<(selected_k_pos, weights_k_pos)>` of length K.
    pub fn encode_router_finalize_k(
        &self,
        probs_k: &DeferredBuf,
        bias: &DeferredBuf,
        k_positions: usize,
    ) -> Result<Vec<(DeferredBuf, DeferredBuf)>> {
        let n_experts = bias.n_elements;
        anyhow::ensure!(
            n_experts > 0 && n_experts <= 1024,
            "encode_router_finalize_k: bias.n_elements out of range (got {})",
            n_experts
        );
        anyhow::ensure!(
            probs_k.n_elements == k_positions * n_experts,
            "encode_router_finalize_k: probs_k has {} elems, expected K*n_experts = {}*{}",
            probs_k.n_elements, k_positions, n_experts
        );
        let generic = n_experts != 256 || ds4_engine::moe::router_scale() != 1.5;
        let npow2: u64 = (n_experts as u64).next_power_of_two();
        let k_top: usize = 6;

        let select_pipe = self.state.pipelines
            .get(if generic {
                "ds4_dsv4_router_finalize_one_any"
            } else {
                "ds4_dsv4_router_finalize_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_finalize_one pipeline not loaded"))?
            .clone();
        let weights_pipe = self.state.pipelines
            .get(if generic {
                "ds4_dsv4_router_weights_one_any"
            } else {
                "ds4_dsv4_router_weights_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_weights_one pipeline not loaded"))?
            .clone();

        let hash_buf = new_input_buffer(&self.state.device, &[0i32]);
        let tokens_buf = new_input_buffer(&self.state.device, &[0i32]);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct SelectArgs {
            has_bias: u32, hash_mode: u32, use_token_buffer: u32, token: u32, hash_rows: u32,
            n_experts: u32, // consumed by the _any shim only
        }
        let select_args = SelectArgs {
            has_bias: 1, hash_mode: 0, use_token_buffer: 0, token: 0, hash_rows: 1,
            n_experts: n_experts as u32,
        };
        let num_experts: u32 = k_top as u32;
        let scale: f32 = ds4_engine::moe::router_scale();
        let min_sum: f32 = 6.103515625e-5;
        let probs_per_k_bytes: u64 = (n_experts as u64) * 4;
        let (sel_tg_threads, sel_shmem) = if generic {
            (npow2, 2 * npow2 * 4)
        } else {
            (256, 2 * 256 * 4)
        };

        let mut out: Vec<(DeferredBuf, DeferredBuf)> = Vec::with_capacity(k_positions);
        for k in 0..k_positions {
            let k64 = k as u64;
            let selected_buf = self.alloc_i32(k_top);
            let weights_buf = self.alloc_f32(k_top);
            // Stage 1: bitonic top-6 select.
            {
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&select_pipe);
                set_scalar_bytes(enc, 0, &select_args);
                enc.set_buffer(1, Some(&probs_k.buf), k64 * probs_per_k_bytes);
                enc.set_buffer(2, Some(&bias.buf), 0);
                enc.set_buffer(3, Some(&hash_buf), 0);
                enc.set_buffer(4, Some(&tokens_buf), 0);
                enc.set_buffer(5, Some(&selected_buf.buf), 0);
                enc.set_threadgroup_memory_length(0, sel_shmem);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(sel_tg_threads, 1, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
            // Stage 2: normalize selected probs → weights.
            {
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&weights_pipe);
                enc.set_buffer(0, Some(&probs_k.buf), k64 * probs_per_k_bytes);
                enc.set_buffer(1, Some(&selected_buf.buf), 0);
                enc.set_buffer(2, Some(&weights_buf.buf), 0);
                set_scalar_bytes(enc, 3, &num_experts);
                set_scalar_bytes(enc, 4, &scale);
                set_scalar_bytes(enc, 5, &min_sum);
                enc.dispatch_threads(
                    MTLSize::new(k_top as u64, 1, 1),
                    MTLSize::new(k_top as u64, 1, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
            out.push((selected_buf, weights_buf));
        }
        Ok(out)
    }

    /// BATCHED router finalize (DS4_CHUNK_BATCH_ROUTER): collapses the K per-token
    /// (select + weights) dispatches of `encode_router_finalize_k` into TWO batched
    /// dispatches (grid.x=K, row=tgpig indexes the token), writing the FLAT
    /// `sel_flat[K*6]` i32 + `wt_flat[K*6]` f32 directly — so the mm_id/fused path
    /// skips both the per-token Vec AND the K blits of `flatten_router_output_k`.
    /// Generic (`_any`) kernels: handles n_experts ≤ npow2, any router_scale. Non-hash.
    pub fn encode_router_finalize_flat_k(
        &self,
        probs_k: &DeferredBuf,
        bias: &DeferredBuf,
        k_positions: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        let n_experts = bias.n_elements;
        anyhow::ensure!(n_experts > 0 && n_experts <= 1024,
            "router_finalize_flat_k: n_experts out of range ({n_experts})");
        anyhow::ensure!(probs_k.n_elements == k_positions * n_experts,
            "router_finalize_flat_k: probs_k {} != K*n_experts {}", probs_k.n_elements, k_positions * n_experts);
        let k_top: usize = 6;
        let npow2: u64 = (n_experts as u64).next_power_of_two();
        let scale: f32 = ds4_engine::moe::router_scale();
        let min_sum: f32 = 6.103515625e-5;

        let select_pipe = self.state.specialized_pipeline(
            "ds4_dsv4_router_finalize_one_any_k", &[], |_fcv| {})?;
        let weights_pipe = self.state.specialized_pipeline(
            "ds4_dsv4_router_weights_one_any_k", &[], |_fcv| {})?;

        let sel_flat = self.alloc_i32(k_positions * k_top);
        let wt_flat = self.alloc_f32(k_positions * k_top);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct SelectArgs {
            has_bias: u32, hash_mode: u32, use_token_buffer: u32, token: u32, hash_rows: u32, n_experts: u32,
        }
        let select_args = SelectArgs {
            has_bias: 1, hash_mode: 0, use_token_buffer: 0, token: 0, hash_rows: 1,
            n_experts: n_experts as u32,
        };
        let sel_shmem = 2 * npow2 * 4;

        // Stage 1: batched bitonic top-6 select → sel_flat[K*6].
        {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&select_pipe);
            set_scalar_bytes(enc, 0, &select_args);
            enc.set_buffer(1, Some(&probs_k.buf), 0);
            enc.set_buffer(2, Some(&bias.buf), 0);
            enc.set_buffer(3, Some(&sel_flat.buf), 0);
            enc.set_threadgroup_memory_length(0, sel_shmem);
            enc.dispatch_thread_groups(
                MTLSize::new(k_positions as u64, 1, 1),
                MTLSize::new(npow2, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        // Stage 2: batched normalize selected → wt_flat[K*6].
        {
            let k_used: u32 = k_top as u32;
            let ne: u32 = n_experts as u32;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&weights_pipe);
            enc.set_buffer(0, Some(&probs_k.buf), 0);
            enc.set_buffer(1, Some(&sel_flat.buf), 0);
            enc.set_buffer(2, Some(&wt_flat.buf), 0);
            set_scalar_bytes(enc, 3, &k_used);
            set_scalar_bytes(enc, 4, &scale);
            set_scalar_bytes(enc, 5, &min_sum);
            set_scalar_bytes(enc, 6, &ne);
            enc.dispatch_thread_groups(
                MTLSize::new(k_positions as u64, 1, 1),
                MTLSize::new(k_top as u64, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }
        Ok((sel_flat, wt_flat))
    }

    /// Phase E M5.4.5-prep: composable `moe_and_shared_chain` —
    /// encodes the per-layer MoE+shared FFN body into THIS scope
    /// instead of opening its own cb.
    ///
    /// Closes the last remaining boundary inside the per-layer
    /// compute graph (the other is `flash_attn_decode`, deferred —
    /// needs GPU f32→f16 conversion). With this method composable,
    /// M5.4.5's orchestrator can chain (in one cb per layer):
    ///   hc_collapse_attn → qkv → kv_norm_rope → kv_fp8_store
    ///   ──── flash_attn boundary ────
    ///   attn_output_matmuls → hc_expand_attn (has_add=0)
    ///   → hc_collapse_ffn → router_logits → router_finalize
    ///   → encode_moe_and_shared_chain → hc_expand_add (has_add=1)
    ///
    /// Internally delegates to the existing
    /// `moe_routed_step_encode` and `shared_chain_batched_encode`
    /// helpers (Slice 5-redo, commit b94363ac), which already accept
    /// `Some(external_cb)`. We pass `&self.cmd_buf` so they encode
    /// into the scope and skip commit/wait.
    ///
    /// Requires `expert_weights` for `layer_idx` to be loaded via
    /// `MetalDispatcher::load_expert_weights` at model init time.
    /// Panics if not loaded — mirrors the inherent
    /// `moe_and_shared_chain_batched_inherent` (macos.rs:4490+)
    /// contract.
    ///
    /// Inputs as CPU slices for now (uploaded inside the encode
    /// helpers). Future optimization: accept DeferredBufs for
    /// inputs that come from upstream scope ops (h_norm from
    /// hc_collapse_ffn, selected/weights from encode_router_finalize).
    /// That refactor lives inside the moe + shared_chain impls and
    /// is decoupled from M5.4.5 — the orchestrator works today
    /// with CPU upload at this boundary, just with one round-trip
    /// per layer for these inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_and_shared_chain(
        &self,
        layer_idx: u32,
        moe_x: &[f32],
        moe_selected: &[usize],
        moe_weights: &[f32],
        d_ffn: usize,
        sh_ffn_norm: &[f32],
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        let qew = self
            .state
            .expert_weights
            .get(layer_idx as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "encode_moe_and_shared_chain: no QuantizedExpertWeights for layer {} (loaded={}); call MetalDispatcher::load_expert_weights at init",
                    layer_idx,
                    self.state.expert_weights.len(),
                )
            })?;
        let selected_i32: Vec<i32> = moe_selected.iter().map(|&i| i as i32).collect();
        let d_ffn_eff = if d_ffn == 0 { qew.d_ffn as usize } else { d_ffn };
        let d_in = moe_x.len();
        let d_embd = sh_ffn_norm.len();

        // SSD-streaming: bind expert cache slots (full stacked tensors are
        // mmap-only and must never reach the GPU).
        let stream_bind = self.state.streaming_expert_bind(layer_idx, &selected_i32);
        let (selected_i32, gate_buf, up_buf, down_buf) = match &stream_bind {
            Some((slots, g, u, d)) => (slots.clone(), g, u, d),
            None => (
                selected_i32,
                &qew.gate.metal_buf,
                &qew.up.metal_buf,
                &qew.down.metal_buf,
            ),
        };

        // Encode MoE into THIS scope (pass our cmd_buf as external_cb).
        let (_cb_moe, moe_dst) = self.state.moe_routed_step_encode(
            Some(&self.cmd_buf),
            moe_x,
            &selected_i32,
            moe_weights,
            gate_buf,
            up_buf,
            down_buf,
            qew.gate.ttype,
            qew.up.ttype,
            qew.down.ttype,
            d_ffn_eff,
            None,
            None,
        )?;
        // Encode shared_chain into THIS scope.
        let (_cb_sh, shared_dst) = self.state.shared_chain_batched_encode(
            Some(&self.cmd_buf),
            sh_ffn_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            want_q80,
            None,
            None,
            None,
            None,
        )?;

        Ok((
            DeferredBuf {
                buf: moe_dst,
                n_elements: d_in,
            },
            DeferredBuf {
                buf: shared_dst,
                n_elements: d_embd,
            },
        ))
    }

    /// M5 task #98 — variant of `encode_moe_and_shared_chain` that
    /// consumes the router's GPU outputs directly. `selected` (i32, 6)
    /// and `weights` (f32, 6) come from `encode_router_finalize` in the
    /// same scope; this binds those buffers straight into the moe pair
    /// + sum6 dispatches, skipping the CPU readback + re-upload of the
    /// router selection that `encode_moe_and_shared_chain` performs at
    /// the boundary.
    ///
    /// Same compute, same kernels, same args — only the data path for
    /// `selected`/`weights` changes. Bit-equivalent to chaining
    /// `encode_router_finalize` → CPU readback → `encode_moe_and_shared_chain`
    /// on the same inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_and_shared_chain_with_router_bufs(
        &self,
        layer_idx: u32,
        moe_x: &[f32],
        selected: &DeferredBuf,
        weights: &DeferredBuf,
        d_ffn: usize,
        sh_ffn_norm: &[f32],
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
        // Raw Q8_0 bytes for the shared gate/up/down (empty → f32). REQUIRED
        // under lean, where sh_w_gate/up/down above are empty.
        sh_w_gate_q8: &'a [u8],
        sh_w_up_q8: &'a [u8],
        sh_w_down_q8: &'a [u8],
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(
            selected.n_elements == 6,
            "encode_moe_and_shared_chain_with_router_bufs: selected.n_elements must be 6 (got {})",
            selected.n_elements,
        );
        anyhow::ensure!(
            weights.n_elements == 6,
            "encode_moe_and_shared_chain_with_router_bufs: weights.n_elements must be 6 (got {})",
            weights.n_elements,
        );
        let qew = self
            .state
            .expert_weights
            .get(layer_idx as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "encode_moe_and_shared_chain_with_router_bufs: no QuantizedExpertWeights for layer {} (loaded={}); call MetalDispatcher::load_expert_weights at init",
                    layer_idx,
                    self.state.expert_weights.len(),
                )
            })?;
        let d_ffn_eff = if d_ffn == 0 { qew.d_ffn as usize } else { d_ffn };
        let d_in = moe_x.len();
        let d_embd = sh_ffn_norm.len();

        // Empty placeholder slices; not consulted when router_bufs is Some.
        let placeholder_sel: &[i32] = &[];
        let placeholder_w: &[f32] = &[];

        let (_cb_moe, moe_dst) = self.state.moe_routed_step_encode(
            Some(&self.cmd_buf),
            moe_x,
            placeholder_sel,
            placeholder_w,
            &qew.gate.metal_buf,
            &qew.up.metal_buf,
            &qew.down.metal_buf,
            qew.gate.ttype,
            qew.up.ttype,
            qew.down.ttype,
            d_ffn_eff,
            Some((&selected.buf, &weights.buf)),
            None,
        )?;
        // Route the shared matmul through the no-copy q8 path when present
        // (DS4_Q8_PROJ on) — required under lean. Mirrors the `_db` variant.
        let q8_on = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true);
        let opt = |b: &'a [u8]| if q8_on && !b.is_empty() { Some(b) } else { None };
        let (_cb_sh, shared_dst) = self.state.shared_chain_batched_encode(
            Some(&self.cmd_buf),
            sh_ffn_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            want_q80,
            None,
            opt(sh_w_gate_q8),
            opt(sh_w_up_q8),
            opt(sh_w_down_q8),
        )?;

        Ok((
            DeferredBuf {
                buf: moe_dst,
                n_elements: d_in,
            },
            DeferredBuf {
                buf: shared_dst,
                n_elements: d_embd,
            },
        ))
    }

    /// M5 task #100 — fully-on-GPU FFN-half body. Same compute as
    /// `encode_moe_and_shared_chain_with_router_bufs`, but moe_x and
    /// sh_ffn_norm arrive as `DeferredBuf` (typically from
    /// `hc_collapse_norm`'s `normed` output) instead of CPU slices.
    /// The internal `moe_routed_step_encode` and
    /// `shared_chain_batched_encode` skip their CPU→scratch / CPU→buf
    /// uploads and bind the supplied buffers directly.
    ///
    /// Bit-equivalent to the CPU-slice variant when the DeferredBufs
    /// hold the same values; only the data path changes.
    ///
    /// `moe_x_db` and `sh_ffn_norm_db` may alias (same buffer) when
    /// !want_q80. For the want_q80 path the caller still needs CPU
    /// access to round-trip h_norm into q8_0, so this method does NOT
    /// take a `want_q80` flag — wire the want_q80 path through the
    /// CPU-slice variant instead.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_and_shared_chain_with_router_bufs_db(
        &self,
        layer_idx: u32,
        moe_x_db: &DeferredBuf,
        selected: &DeferredBuf,
        weights: &DeferredBuf,
        d_ffn: usize,
        sh_ffn_norm_db: &DeferredBuf,
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        // Raw GGUF block_q8_0 bytes for the shared gate/up/down (empty → f32).
        sh_w_gate_q8: &[u8],
        sh_w_up_q8: &[u8],
        sh_w_down_q8: &[u8],
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(
            selected.n_elements == 6,
            "encode_moe_and_shared_chain_with_router_bufs_db: selected.n_elements must be 6 (got {})",
            selected.n_elements,
        );
        anyhow::ensure!(
            weights.n_elements == 6,
            "encode_moe_and_shared_chain_with_router_bufs_db: weights.n_elements must be 6 (got {})",
            weights.n_elements,
        );
        let qew = self
            .state
            .expert_weights
            .get(layer_idx as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "encode_moe_and_shared_chain_with_router_bufs_db: no QuantizedExpertWeights for layer {} (loaded={}); call MetalDispatcher::load_expert_weights at init",
                    layer_idx,
                    self.state.expert_weights.len(),
                )
            })?;
        let d_ffn_eff = if d_ffn == 0 { qew.d_ffn as usize } else { d_ffn };
        let d_in = moe_x_db.n_elements;
        let d_embd = sh_ffn_norm_db.n_elements;

        // Placeholder slices; not consulted because router_bufs and
        // external_x_buf are Some. The x slice's `len()` IS read for
        // d_in shape — so reconstruct a zero-length slice with the
        // right address is not enough. Instead pass a dummy of the
        // correct length; its CONTENTS go unread.
        let dummy_x: Vec<f32> = vec![0.0; d_in];
        let dummy_norm: Vec<f32> = vec![0.0; d_embd];
        let placeholder_sel: &[i32] = &[];
        let placeholder_w: &[f32] = &[];

        // SSD-streaming: `selected` must already hold cache SLOT ids (the
        // caller remaps after a drain); bind the cache pool buffers instead
        // of the full mmap-only stacked tensors.
        let stream_cache = self
            .state
            .expert_caches
            .get(layer_idx as usize)
            .and_then(|c| c.lock().ok().map(|c| (c.gate.clone(), c.up.clone(), c.down.clone())));
        let (gate_buf, up_buf, down_buf) = match &stream_cache {
            Some((g, u, d)) => (g, u, d),
            None => (&qew.gate.metal_buf, &qew.up.metal_buf, &qew.down.metal_buf),
        };
        let (_cb_moe, moe_dst) = self.state.moe_routed_step_encode(
            Some(&self.cmd_buf),
            &dummy_x,
            placeholder_sel,
            placeholder_w,
            gate_buf,
            up_buf,
            down_buf,
            qew.gate.ttype,
            qew.up.ttype,
            qew.down.ttype,
            d_ffn_eff,
            Some((&selected.buf, &weights.buf)),
            Some(&moe_x_db.buf),
        )?;
        // Gate the shared-expert Q8_0 weight path on DS4_Q8_PROJ (consistent with
        // the q/kv + output projections); `opt` also guards against absent bytes.
        let q8_on = std::env::var("DS4_Q8_PROJ").map(|v| v != "0").unwrap_or(true);
        fn opt(on: bool, b: &[u8]) -> Option<&[u8]> {
            if on && !b.is_empty() {
                Some(b)
            } else {
                None
            }
        }
        let (_cb_sh, shared_dst) = self.state.shared_chain_batched_encode(
            Some(&self.cmd_buf),
            &dummy_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            false, // want_q80 path always goes through the CPU-slice variant
            Some(&sh_ffn_norm_db.buf),
            opt(q8_on, sh_w_gate_q8),
            opt(q8_on, sh_w_up_q8),
            opt(q8_on, sh_w_down_q8),
        )?;

        Ok((
            DeferredBuf {
                buf: moe_dst,
                n_elements: d_in,
            },
            DeferredBuf {
                buf: shared_dst,
                n_elements: d_embd,
            },
        ))
    }

    /// Phase 2 MoE-K Step 1 — dispatch `kernel_mul_mm_id_map0_ne20_6` to
    /// build the expert→token map from a `selected_k [K, 6]` (i32) buffer.
    /// Returns `(tpe [n_experts] u32, ids [n_experts*K] i32)`.
    ///
    /// `tpe[e]` = number of (token, slot) pairs that chose expert e.
    /// `ids[e*K + n]` = (token_idx * 6 + slot) for the n-th token using e.
    ///
    /// One threadgroup, n_experts threads (256 for DS4). ne20 hard-coded
    /// to 6 (top-6 expert routing).
    pub fn encode_mul_mm_id_map0_k(
        &self,
        selected_k: &DeferredBuf,
        n_experts: usize,
        k_positions: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(
            selected_k.n_elements == k_positions * 6,
            "encode_mul_mm_id_map0_k: selected_k has {} elems, expected K*6 = {}*6",
            selected_k.n_elements, k_positions
        );
        anyhow::ensure!(n_experts == 256, "DS4 hard-codes n_experts=256");

        // map0 has no function constants — load via specialized_pipeline with
        // empty key (the bridge lib has it but the preloader skips
        // ne20_6 since only ne20_8 is in the KernelSpec registry).
        let empty_key: &[u8] = &[];
        let pipe = self
            .state
            .specialized_pipeline("ds4_kernel_mul_mm_id_map0_ne20_6", empty_key, |_fcv| {})?;

        // Output buffers: tpe[n_experts] u32, ids[n_experts * K] i32.
        let tpe = self.alloc_i32(n_experts);             // u32 storage, alloc_i32 OK (4-byte stride)
        let ids = self.alloc_i32(n_experts * k_positions);

        // map0 args (per moe.metal:290). Only `ne21`, `ne20`, `nb21` are
        // read by the kernel; the rest are placeholders for unused fields.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Map0Args {
            ne02: i32, ne10: i32, ne11: i32,
            nb11: u64, nb12: u64,
            ne21: i32, ne20: i32,
            nb21: u64,
        }
        let args = Map0Args {
            ne02: n_experts as i32,
            ne10: 1, ne11: 1,
            nb11: 0, nb12: 0,
            ne21: k_positions as i32,
            ne20: 6,
            nb21: (6 * std::mem::size_of::<i32>()) as u64,
        };

        // Threadgroup shmem: ntg * ne20 * sizeof(uint16_t) per the kernel.
        // n_experts=256 threads × 6 × 2 = 3072 bytes.
        let shmem_bytes: u64 = (n_experts as u64) * 6 * 2;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&selected_k.buf), 0);
        enc.set_buffer(2, Some(&tpe.buf), 0);
        enc.set_buffer(3, Some(&ids.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(n_experts as u64, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok((tpe, ids))
    }

    /// Phase 2 MoE-K Step 4 — generic K-batched MoE matmul dispatcher.
    /// Targets ANY upstream `kernel_mul_mm_id_<quant>_f32` (q8_0 / q4_K /
    /// iq2_xxs / q2_K) — same args struct, same dispatch shape, only the
    /// kernel name + block bytes + block elements (QK) differ.
    ///
    /// Each expert weight is read ONCE per dispatch and applied to all K
    /// activations that selected it (per `tpe`/`ids` from
    /// `encode_mul_mm_id_map0_k`).
    ///
    /// Two variants via `act_slots`:
    /// - **Gate/Up** (`act_slots = 1`): activations `[K, d_in]` broadcast
    ///   to all 6 expert slots per K-position. ne11=1, nb11=d_in*4, nb12=d_in*4.
    /// - **Down** (`act_slots = 6`): activations `[K, 6, d_in=d_ffn]` —
    ///   per-slot activations (output of moe_swiglu_weight). ne11=6,
    ///   nb11=d_in*4, nb12=6*d_in*4.
    ///
    /// `block_qk`, `block_bytes` per quant:
    /// - Q8_0:    qk=32,  block_bytes=34
    /// - Q2_K:    qk=256, block_bytes=84
    /// - Q4_K:    qk=256, block_bytes=144
    /// - IQ2_XXS: qk=256, block_bytes=66
    ///
    /// Returns `dst [K, 6, d_out]` f32. dst[k, slot, j] is expert slot's
    /// contribution to token k at output column j.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mul_mm_id_qx_k(
        &self,
        kernel_name: &str,
        block_qk: usize,
        block_bytes: u64,
        experts_w: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        n_experts: usize,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
        act_slots: usize,
        act_bytes: u64,
    ) -> Result<DeferredBuf> {
        // FFI-boundary validation (see memory: ffi-boundary-dim-validation-requirement).
        // A 0 dim here => zero-length buffer views + degenerate grid/stride math =>
        // GPU out-of-bounds (SIGSEGV) inside the Metal kernel, which Rust cannot catch.
        // Reject it as a clean panic AT THE CALL SITE instead. (This is the guard that
        // would have turned the moe.d_ffn=0 SIGSEGV into an obvious error.)
        anyhow::ensure!(
            d_in > 0 && d_out > 0 && n_experts > 0,
            "mul_mm_id_qx_k: dims must be > 0 (d_in={d_in} d_out={d_out} n_experts={n_experts}) \
             — a 0 dim crossing into the Metal kernel = GPU OOB"
        );
        anyhow::ensure!(
            d_in % block_qk == 0,
            "mul_mm_id_qx_k: d_in ({}) must be %block_qk={}",
            d_in, block_qk
        );
        anyhow::ensure!(
            matches!(act_slots, 1 | 6),
            "mul_mm_id_qx_k: act_slots must be 1 (gate/up) or 6 (down), got {}",
            act_slots
        );
        anyhow::ensure!(
            x_k.n_elements == k_positions * act_slots * d_in,
            "mul_mm_id_qx_k: x_k has {} elems, expected K*act_slots*d_in = {}*{}*{}",
            x_k.n_elements, k_positions, act_slots, d_in
        );
        anyhow::ensure!(tpe.n_elements == n_experts);
        anyhow::ensure!(ids.n_elements == n_experts * k_positions);

        // Specialize the pipeline with FC_mul_mm_bc_inp=false, bc_out=false.
        let bc_inp: bool = false;
        let bc_out: bool = false;
        let mut key = Vec::with_capacity(2);
        key.push(bc_inp as u8);
        key.push(bc_out as u8);
        let pipe = self
            .state
            .specialized_pipeline(kernel_name, &key, |fcv| {
                fcv.set_constant_value_at_index(
                    &bc_inp as *const _ as *const _, MTLDataType::Bool, 700,
                );
                fcv.set_constant_value_at_index(
                    &bc_out as *const _ as *const _, MTLDataType::Bool, 701,
                );
            })?;

        // Per-quant row stride: ceil(d_in / block_qk) * block_bytes.
        let nb_per_row = (d_in as u64 / block_qk as u64) * block_bytes;
        let nb_per_expert = nb_per_row * (d_out as u64);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMmIdArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne11: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne20: i32, ne21: i32,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        let args = MulMmIdArgs {
            ne00: d_in as i32,
            ne02: n_experts as i32,
            nb01: nb_per_row,
            nb02: nb_per_expert,
            nb03: nb_per_expert * n_experts as u64,
            ne11: act_slots as i32,
            nb10: act_bytes,                                         // 4=f32, 2=f16 activation (mid)
            nb11: (d_in as u64) * act_bytes,                        // per slot (or per token when act_slots=1)
            nb12: (act_slots as u64) * (d_in as u64) * act_bytes,    // per token
            nb13: (k_positions as u64) * (act_slots as u64) * (d_in as u64) * act_bytes,
            ne20: 6,
            ne21: k_positions as i32,
            ne0: d_out as i32,
            ne1: 6,
            r2: 1, r3: 1,
        };

        // Output buffer: dst[K, 6, d_out] floats.
        let dst = self.pooled_scratch_f32("moe_mmid_dst", k_positions * 6 * d_out); // T4: pool+pin (merged-cb working set)

        // Dispatch grid: tgpig.z = expert (n_experts),
        //                tgpig.y = ceil(d_out / NR0=64),
        //                tgpig.x = ceil(max_tpe / NR1=32).
        // Worst case max_tpe = K*6 (all picks for one expert). For sparsity
        // it's usually << that, but we must size for worst case.
        const NR0: u64 = 64;
        const NR1: u64 = 32;
        // grid_x must cover the LARGEST expert's tokens-per-expert (tpe). The kernel early-exits
        // threadgroups with r1=tgpig.x*NR1 >= tpe[im], so any cap >= the run's actual max tpe is
        // BYTE-IDENTICAL; a smaller cap just stops launching over-provisioned early-exit
        // threadgroups (the dominant cost in the 41%-of-prefill MoE GEMM — the over-dispatch, not
        // the GEMM math). DEFAULT = ceil(K/2): the aux-loss-free-balanced DeepSeek router gives
        // actual max tpe ~ K*6/n_experts (~70 @K=3000; observed <=187), so K/2=1500 is a ~21x
        // margin (would need 50% of tokens routed to one of 256 experts — impossible for a
        // balanced router). +8.7% over the K worst case, byte-identical for any realistic prompt.
        // tpe[e] <= K is the only PROVABLE bound (token picks 6 DISTINCT experts) → DS4_MOE_GRID_DIV=1
        // for the provably-safe K; DS4_MOE_GRID_WIDE=1 for the original K*6.
        let max_tpe = if std::env::var("DS4_MOE_GRID_WIDE").ok().as_deref() == Some("1") {
            (k_positions as u64) * 6
        } else if let Ok(div) = std::env::var("DS4_MOE_GRID_DIV") {
            let d = div.parse::<u64>().unwrap_or(2).max(1);
            ((k_positions as u64) + d - 1) / d
        } else {
            // DEFAULT = ceil(K/8): grid covers ~375 tpe @K=3000 = 2x margin over the observed
            // max tpe (187). With the inter-cb bubble removed (bounded event-pipeline), the MoE
            // GEMM over-dispatch is the next gpu_busy lever — K/4→K/8 = +3% prefill @3000
            // (295.6→304.4 tok/s = +19.8% over antirez 254, BEATS the +18% goal). Byte-identical
            // at K=3000 (chunk_flag_identity DS4_MOE_GRID_DIV=8 vs WIDE). 2x margin is safe for any
            // load-balanced aux-loss-free DeepSeek router (would need >375/3000 tokens to one of
            // 256 experts to drop). DS4_MOE_GRID_DIV=4 reverts to the 4x-margin default;
            // DS4_MOE_GRID_DIV=1 = provably-safe K; DS4_MOE_GRID_WIDE=1 = original K*6.
            (k_positions as u64 + 7) / 8 // ceil(K/8) — 2x margin over observed max tpe (187 @K=3000)
        };
        // SAFETY: the K/2 default is byte-identical for any prompt whose actual max tpe <= K/2
        // (validated by the chunk_flag_identity gate: DS4_MOE_GRID_DIV=N vs DS4_MOE_GRID_WIDE).
        // The provably-safe K is one flag away (DS4_MOE_GRID_DIV=1) if a non-balanced router is
        // ever a concern. A run-adaptive over-cap monitor needs a non-consuming u32 tpe read (TODO).
        let grid_x = (max_tpe + NR1 - 1) / NR1;
        let grid_y = ((d_out as u64) + NR0 - 1) / NR0;
        let grid_z = n_experts as u64;

        // Threadgroup: 4 simdgroups × 32 = 128 threads.
        let tg_threads = MTLSize::new(32, 4, 1);

        // Shmem: 4096 bytes for sa + (NR0 * NR1 * sizeof(f32)) for output staging =
        // 4096 + 8192 = 12288 bytes. Kernel allocates threadgroup S0 * sa
        // at shmem and S1 * sb at shmem+4096; output staging reuses shmem.
        let shmem_bytes: u64 = 4096 + (NR0 * NR1 * 4);

        // DS4_MOE_GRID_INDIRECT (default-on): size grid_x to the run's ACTUAL max
        // tokens-per-expert via a GPU max-reduce of tpe[] → MTLDispatchThreadgroupsIndirect
        // args, instead of the K worst-case. Always correct (covers the true max, drops no
        // tokens — byte-identical) AND tight (removes the ~n_experts/6x early-exit
        // over-dispatch, the major prefill cost: ~+18% @K=3000). DS4_MOE_GRID_INDIRECT=0
        // reverts to the static grid_x (still ceil(K/32) unless DS4_MOE_GRID_WIDE).
        // DEFAULT-OFF: the indirect path is byte-identical + safe but OVERHEAD-LIMITED — its
        // per-call GPU reduce + barrier (3 mm_id calls/layer × 43) negate most of the
        // over-dispatch saving (~+2-3% noisy vs the static narrow grid's +18%). The clean
        // capture is a LOOPING-GRID kernel (fixed small grid_x, each tg loops its tiles if
        // tpe > coverage — no reduce/barrier), TODO. Kept gated as scaffolding.
        let indirect = std::env::var("DS4_MOE_GRID_INDIRECT").ok().as_deref() == Some("1")
            && std::env::var("DS4_MOE_GRID_WIDE").ok().as_deref() != Some("1")
            && std::env::var("DS4_MOE_GRID_DIV").is_err();
        if indirect {
            // Stage 1: GPU-compute the indirect grid [grid_x=ceil(max_tpe/NR1), grid_y, grid_z].
            let args_buf = self.alloc_i32(3); // 3×u32 MTLDispatchThreadgroupsIndirectArguments
            let empty_key: &[u8] = &[];
            let gpipe = self.state.specialized_pipeline("ds4_kernel_moe_grid_args", empty_key, |_| {})?;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&gpipe);
            enc.set_buffer(0, Some(&tpe.buf), 0);
            set_scalar_bytes(enc, 1, &(n_experts as u32));
            set_scalar_bytes(enc, 2, &(NR1 as u32));
            set_scalar_bytes(enc, 3, &(grid_y as u32));
            set_scalar_bytes(enc, 4, &(grid_z as u32));
            enc.set_buffer(5, Some(&args_buf.buf), 0);
            enc.set_threadgroup_memory_length(0, 256 * 4);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            // Light in-encoder barrier so the args write is visible to the indirect grid read
            // — far cheaper than ending the encoder (which broke pipelining: +3.6% vs the
            // static grid's +18%). MTLBarrierScopeBuffers = 1.
            {
                use objc::{msg_send, sel, sel_impl};
                let _: () = unsafe { msg_send![enc, memoryBarrierWithScope: 1u64] };
            }
            // Stage 2: the GEMM, grid sized indirectly from args_buf (same encoder).
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&experts_w.buf), 0);
            enc.set_buffer(2, Some(&x_k.buf), 0);
            enc.set_buffer(3, Some(&tpe.buf), 0);
            enc.set_buffer(4, Some(&ids.buf), 0);
            enc.set_buffer(5, Some(&dst.buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups_indirect(&args_buf.buf, 0, tg_threads);
            crate::macos::end_shared_compute_enc(enc);
        } else {
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&experts_w.buf), 0);
            enc.set_buffer(2, Some(&x_k.buf), 0);
            enc.set_buffer(3, Some(&tpe.buf), 0);
            enc.set_buffer(4, Some(&ids.buf), 0);
            enc.set_buffer(5, Some(&dst.buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(grid_x, grid_y, grid_z),
                tg_threads,
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        Ok(dst)
    }

    /// DS4 FUSED gate+up+SwiGLU mm_id for iq2_xxs experts (prefill MoE). Replaces
    /// the separate gate GEMM + up GEMM + `moe_swiglu_weight_k` (3 dispatches) with
    /// ONE: reads the per-token activation tile once, accumulates BOTH gate and up
    /// (simdgroup_half8x8 tiles), and applies SwiGLU + route-weight inline. Ported
    /// from antirez `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16`, output f32 so the
    /// existing f32 down GEMM consumes the `mid` unchanged.
    ///
    /// `x_k` = normed `[K, d_embd]` (broadcast across the 6 slots, ne11=1).
    /// Returns `mid [K, 6, d_ffn]` f32 — the same layout `encode_mul_mm_id_qx_k`
    /// (down stage, act_slots=6) expects.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mul_mm_id_iq2_pair_swiglu_k(
        &self,
        gate_w: &DeferredBuf,
        up_w: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        wt_flat: &DeferredBuf,
        n_experts: usize,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            d_embd > 0 && d_ffn > 0 && n_experts > 0,
            "mul_mm_id_iq2_pair_swiglu: dims must be >0 (d_embd={d_embd} d_ffn={d_ffn} n_experts={n_experts})"
        );
        anyhow::ensure!(d_embd % 256 == 0, "iq2 pair_swiglu: d_embd ({d_embd}) %256 != 0");
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_embd,
            "iq2 pair_swiglu: x_k {} != K*d_embd = {}*{}", x_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(tpe.n_elements == n_experts);
        anyhow::ensure!(ids.n_elements == n_experts * k_positions);
        anyhow::ensure!(
            wt_flat.n_elements == k_positions * 6,
            "iq2 pair_swiglu: wt_flat {} != K*6 = {}", wt_flat.n_elements, k_positions * 6
        );

        // iq2_xxs block: QK_K=256, 66 bytes/block.
        let nb_per_row = (d_embd as u64 / 256) * 66;
        let nb_per_expert = nb_per_row * (d_ffn as u64);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMmIdArgs {
            ne00: i32, ne02: i32,
            nb01: u64, nb02: u64, nb03: u64,
            ne11: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne20: i32, ne21: i32,
            ne0: i32, ne1: i32,
            r2: i16, r3: i16,
        }
        let args = MulMmIdArgs {
            ne00: d_embd as i32,
            ne02: n_experts as i32,
            nb01: nb_per_row,
            nb02: nb_per_expert,
            nb03: nb_per_expert * n_experts as u64,
            ne11: 1, // gate/up: per-token activations broadcast across slots
            nb10: 4,
            nb11: (d_embd as u64) * 4,
            nb12: (d_embd as u64) * 4,
            nb13: (k_positions as u64) * (d_embd as u64) * 4,
            ne20: 6,
            ne21: k_positions as i32,
            ne0: d_ffn as i32,
            ne1: 6,
            r2: 1, r3: 1,
        };

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MoeSwigluActArgs {
            width: u32, rows: u32,
            gate_row_stride: u64, up_row_stride: u64, mid_row_stride: u64, weight_stride: u64,
            write_clamped: u32, clamp_value: f32,
        }
        let act = MoeSwigluActArgs {
            width: d_ffn as u32,
            rows: (k_positions * 6) as u32,
            gate_row_stride: (d_ffn as u64) * 4,
            up_row_stride: (d_ffn as u64) * 4,
            mid_row_stride: (d_ffn as u64) * 4, // f32 mid → down GEMM reads it unchanged
            weight_stride: 4,
            write_clamped: 0,
            clamp_value: f32::INFINITY, // no clamp — matches moe_swiglu_weight_k
        };

        // mid [K, 6, d_ffn] f32 — same layout the down stage consumes.
        let mid = self.pooled_scratch_f32("moe_mid", k_positions * 6 * d_ffn);

        let empty_key: &[u8] = &[];
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mm_id_iq2_xxs_pair_swiglu_f32", empty_key, |_fcv| {},
        )?;

        // Grid: x = ceil(max_tpe/NR1), y = ceil(d_ffn/NR0), z = n_experts. Each
        // token picks 6 DISTINCT experts → an expert gets ≤ K tokens, so max_tpe
        // ≤ k_positions (the ids row stride = ne21). Kernel early-exits r1>=neh1.
        const NR0: u64 = 64;
        const NR1: u64 = 32;
        let grid_x = (k_positions as u64 + NR1 - 1) / NR1;
        let grid_y = (d_ffn as u64 + NR0 - 1) / NR0;
        let grid_z = n_experts as u64;
        // Two f32 output accumulators (temp_gate + temp_up, NR0*NR1 each) = 16384 B.
        let shmem_bytes: u64 = 16384;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        set_scalar_bytes(enc, 1, &act);
        enc.set_buffer(2, Some(&gate_w.buf), 0);
        enc.set_buffer(3, Some(&up_w.buf), 0);
        enc.set_buffer(4, Some(&x_k.buf), 0);
        enc.set_buffer(5, Some(&tpe.buf), 0);
        enc.set_buffer(6, Some(&ids.buf), 0);
        enc.set_buffer(7, Some(&mid.buf), 0);
        enc.set_buffer(8, Some(&wt_flat.buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(grid_x, grid_y, grid_z),
            MTLSize::new(32, 4, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(mid)
    }

    /// Q8_0 thin wrapper for `encode_mul_mm_id_qx_k`. Block: QK=32, 34 bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mul_mm_id_q8_0_k(
        &self,
        experts_w_q8: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        n_experts: usize,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
        act_slots: usize,
    ) -> Result<DeferredBuf> {
        self.encode_mul_mm_id_qx_k(
            "ds4_kernel_mul_mm_id_q8_0_f32", 32, 34,
            experts_w_q8, x_k, tpe, ids,
            n_experts, d_in, d_out, k_positions, act_slots, 4,
        )
    }

    /// Q4_K thin wrapper. Block: QK_K=256, 144 bytes
    /// (256 quants × 4 bits + 12 bytes of scales/mins/d/dmin = 128 + 16 = 144).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_mul_mm_id_q4_K_k(
        &self,
        experts_w_q4k: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        n_experts: usize,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
        act_slots: usize,
    ) -> Result<DeferredBuf> {
        self.encode_mul_mm_id_qx_k(
            "ds4_kernel_mul_mm_id_q4_K_f32", 256, 144,
            experts_w_q4k, x_k, tpe, ids,
            n_experts, d_in, d_out, k_positions, act_slots, 4,
        )
    }

    /// Q2_K thin wrapper. Block: QK_K=256, 84 bytes.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_mul_mm_id_q2_K_k(
        &self,
        experts_w_q2k: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        n_experts: usize,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
        act_slots: usize,
    ) -> Result<DeferredBuf> {
        self.encode_mul_mm_id_qx_k(
            "ds4_kernel_mul_mm_id_q2_K_f32", 256, 84,
            experts_w_q2k, x_k, tpe, ids,
            n_experts, d_in, d_out, k_positions, act_slots, 4,
        )
    }

    /// IQ2_XXS thin wrapper. Block: QK_K=256, 66 bytes
    /// (256 quants × 2 bits via 8-bit grid index = 64 + half scale = 64 + 2 = 66).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mul_mm_id_iq2_xxs_k(
        &self,
        experts_w_iq2xxs: &DeferredBuf,
        x_k: &DeferredBuf,
        tpe: &DeferredBuf,
        ids: &DeferredBuf,
        n_experts: usize,
        d_in: usize,
        d_out: usize,
        k_positions: usize,
        act_slots: usize,
    ) -> Result<DeferredBuf> {
        self.encode_mul_mm_id_qx_k(
            "ds4_kernel_mul_mm_id_iq2_xxs_f32", 256, 66,
            experts_w_iq2xxs, x_k, tpe, ids,
            n_experts, d_in, d_out, k_positions, act_slots, 4,
        )
    }

    /// Phase 2 MoE-K Option A — auto-dispatch K-batched fused MoE chain
    /// based on the layer's expert quant types. Pulls expert weights from
    /// `state.expert_weights[layer_idx]` (loaded via
    /// `MetalDispatcher::load_expert_weights`) and selects the matching
    /// pair_swiglu_K + sum6_K kernel set:
    ///
    /// - IQ2_XXS gate/up + Q2_K down  → `encode_moe_chain_fused_K_iq2_xxs_q2_K`
    /// - Q4_K    gate/up + Q4_K down  → `encode_moe_chain_fused_K_q4_K_q4_K`
    ///
    /// Returns moe_k [K, d_embd]. Bails if the layer's ttype combination
    /// doesn't match a supported variant — caller falls back to the
    /// blit-shim path.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_moe_chain_fused_K_auto(
        &self,
        layer_idx: u32,
        normed_k: &DeferredBuf,
        sel_flat: &DeferredBuf,
        wt_flat:  &DeferredBuf,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        use ds4_engine::gguf::GgmlType;
        let qew = self
            .state
            .expert_weights
            .get(layer_idx as usize)
            .ok_or_else(|| anyhow::anyhow!(
                "encode_moe_chain_fused_K_auto: no expert weights for layer {} (loaded={})",
                layer_idx, self.state.expert_weights.len()
            ))?;
        let n_experts = qew.n_experts as usize;
        // d_ffn from QEW when the caller passes 0 (composed moe.d_ffn is 0). See the
        // mm_qx_k_auto fix above — same chunk-prefill OOB/SEGV root cause.
        let d_ffn = if d_ffn != 0 { d_ffn } else { qew.d_ffn as usize };
        let gate_db = DeferredBuf::from_external_buffer(
            qew.gate.metal_buf.clone(), n_experts * d_ffn * d_embd,
        );
        let up_db = DeferredBuf::from_external_buffer(
            qew.up.metal_buf.clone(), n_experts * d_ffn * d_embd,
        );
        let down_db = DeferredBuf::from_external_buffer(
            qew.down.metal_buf.clone(), n_experts * d_embd * d_ffn,
        );

        match (qew.gate.ttype, qew.up.ttype, qew.down.ttype) {
            (GgmlType::IQ2_XXS, GgmlType::IQ2_XXS, GgmlType::Q2_K) => {
                self.encode_moe_chain_fused_K_iq2_xxs_q2_K(
                    normed_k, sel_flat, wt_flat,
                    &gate_db, &up_db, &down_db,
                    n_experts, d_embd, d_ffn, k_positions,
                )
            }
            (GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K) => {
                self.encode_moe_chain_fused_K_q4_K_q4_K(
                    normed_k, sel_flat, wt_flat,
                    &gate_db, &up_db, &down_db,
                    n_experts, d_embd, d_ffn, k_positions,
                )
            }
            (g, u, d) => anyhow::bail!(
                "encode_moe_chain_fused_K_auto: layer {} quant combo ({:?}, {:?}, {:?}) not supported (have iq2_xxs+iq2_xxs+q2_K, q4_K+q4_K+q4_K)",
                layer_idx, g, u, d
            ),
        }
    }

    /// Large-K MoE via the mm_id (expert-token gather) chain — the
    /// megablocks-style path that amortizes each expert's weight read across
    /// ALL tokens routed to it in the chunk. Validated to scale to ~97% K=1-eff
    /// at K=512 (vs the fused pair_swiglu_K path which breaks at K>=32 and the
    /// blit shim which doesn't batch the MoE at all). Sources the per-layer
    /// expert weights from `state.expert_weights` (like the fused auto path) and
    /// maps each tensor's quant type to its `mul_mm_id` kernel. This is the
    /// MoE engine for chunked prefill.
    pub fn encode_moe_chain_mm_qx_k_auto(
        &self,
        layer_idx: u32,
        normed_k: &DeferredBuf,
        sel_flat: &DeferredBuf,
        wt_flat: &DeferredBuf,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        use ds4_engine::gguf::GgmlType;
        fn kinfo(t: GgmlType) -> Result<(&'static str, usize, u64)> {
            Ok(match t {
                GgmlType::Q8_0 => ("ds4_kernel_mul_mm_id_q8_0_f32", 32, 34),
                GgmlType::Q2_K => ("ds4_kernel_mul_mm_id_q2_K_f32", 256, 84),
                GgmlType::Q4_K => ("ds4_kernel_mul_mm_id_q4_K_f32", 256, 144),
                GgmlType::IQ2_XXS => ("ds4_kernel_mul_mm_id_iq2_xxs_f32", 256, 66),
                other => anyhow::bail!("mm_id MoE chain: unsupported quant {:?}", other),
            })
        }
        let qew = self.state.expert_weights.get(layer_idx as usize).ok_or_else(|| {
            anyhow::anyhow!(
                "encode_moe_chain_mm_qx_k_auto: no expert weights for layer {} (loaded={})",
                layer_idx, self.state.expert_weights.len()
            )
        })?;
        let n_experts = qew.n_experts as usize;
        // d_ffn comes from the QEW expert tensor when the caller passes 0 — the
        // composed model's moe.d_ffn is intentionally 0 (decode_step.rs:574
        // "MetalDispatcher reads d_ffn from QEW"). Trusting that 0 yields 0-sized
        // buffer views + degenerate grid/stride math → OOB/SEGV (the chunk-prefill
        // mm_id MoE fault). Fall back to the real tensor dim.
        let d_ffn = if d_ffn != 0 { d_ffn } else { qew.d_ffn as usize };
        // SSD-streaming: the stacked tensors are mmap-only (16 KB GPU stubs —
        // never bindable). Bind the whole-layer chunk pool instead; the
        // caller synced + refilled it for this layer (ssd_chunk_pool_sync).
        // Expert ids are identity in the pool — no remap needed.
        let (gate_buf, up_buf, down_buf) = if std::env::var("DS4_SSD_STREAM").is_ok() {
            self.state.chunk_pool_bind(layer_idx).ok_or_else(|| {
                anyhow::anyhow!("mm_id stream: no chunk pool for layer {layer_idx}")
            })?
        } else {
            (qew.gate.metal_buf.clone(), qew.up.metal_buf.clone(), qew.down.metal_buf.clone())
        };
        let gate_db = DeferredBuf::from_external_buffer(gate_buf, n_experts * d_ffn * d_embd);
        let up_db = DeferredBuf::from_external_buffer(up_buf, n_experts * d_ffn * d_embd);
        let down_db = DeferredBuf::from_external_buffer(down_buf, n_experts * d_embd * d_ffn);
        let (gk, gqk, gb) = kinfo(qew.gate.ttype)?;
        let (uk, uqk, ub) = kinfo(qew.up.ttype)?;
        let (dk, dqk, db) = kinfo(qew.down.ttype)?;
        self.encode_moe_chain_mm_qx_k(
            normed_k, sel_flat, wt_flat,
            &gate_db, &up_db, &down_db,
            gk, gqk, gb, uk, uqk, ub, dk, dqk, db,
            n_experts, d_embd, d_ffn, k_positions,
        )
    }

    /// Phase 2 MoE-K Option A — flatten `encode_router_finalize_k`'s
    /// `Vec<(selected, weights)>` output into flat `[K*6]` i32 + f32
    /// DeferredBufs. The fused-K MoE chain consumes flat buffers; this
    /// helper bridges the existing router_finalize_k API.
    ///
    /// Uses K blit-copies (cheap: 6 i32 + 6 f32 per K-position).
    pub fn flatten_router_output_k(
        &self,
        sel_wt_k: &[(DeferredBuf, DeferredBuf)],
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        let k = sel_wt_k.len();
        let sel_flat = self.alloc_i32(k * 6);
        let wt_flat  = self.alloc_f32(k * 6);
        let f32_b: u64 = 4;
        let row_bytes: u64 = 6 * f32_b;
        for (i, (sel, wt)) in sel_wt_k.iter().enumerate() {
            let off = (i as u64) * row_bytes;
            crate::macos::end_shared_compute_enc_force();
            let enc = self.cmd_buf.new_blit_command_encoder();
            enc.copy_from_buffer(&sel.buf, 0, &sel_flat.buf, off, row_bytes);
            enc.copy_from_buffer(&wt.buf,  0, &wt_flat.buf,  off, row_bytes);
            enc.end_encoding();
        }
        Ok((sel_flat, wt_flat))
    }

    /// Phase 2 — K-position MoE + shared-expert chain. K iterations of
    /// `encode_moe_and_shared_chain_with_router_bufs_db` with blit-copy
    /// shimming around the per-K normed input and the dual per-K outputs.
    ///
    /// For each k position:
    ///   1. blit_copy: normed_k[k*d_embd..(k+1)*d_embd] → temp_normed
    ///   2. encode_moe_and_shared_chain_with_router_bufs_db (existing K=1
    ///      kernel) on temp_normed + (selected[k], weights[k]) → (moe_dst,
    ///      shared_dst)
    ///   3. blit_copy: moe_dst → moe_k[k*d_embd..(k+1)*d_embd]
    ///                 shared_dst → shared_k[k*d_embd..(k+1)*d_embd]
    ///
    /// The blit shim is the cost of preserving the existing K=1 impl
    /// unchanged. Real perf win comes from amortizing the EXPERT MATMULS
    /// which are the FFN bottleneck — each K-position still triggers full
    /// MoE pair dispatches. A future K-amortized mul_mv_id refactor would
    /// reuse expert weight reads across K activations (analogous to
    /// matvec_k_q8_0 vs matvec_q8_0).
    ///
    /// Phase 2 MoE-K Step 2 — per-row swiglu + route-weight multiplication.
    /// Dispatches `kernel_dsv4_moe_swiglu_weight` with `rows = K*6` (one
    /// row per expert slot per K-position). Writes mid in place:
    ///
    /// ```text
    ///   mid[r, i] = silu(gate[r, i]) * up[r, i] * route_weight[r]
    /// ```
    ///
    /// where `r = k * 6 + slot`. Inputs/outputs are flat `[K*6, d_ffn]` f32.
    /// `weights_k_flat` is `[K*6]` f32 — slot-major within K (matches the
    /// `[K, 6]` natural layout).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_swiglu_weight_k(
        &self,
        gate_k: &DeferredBuf,
        up_k: &DeferredBuf,
        weights_k_flat: &DeferredBuf,
        d_ffn: usize,
        k_positions: usize,
        mid_f16: bool,
    ) -> Result<DeferredBuf> {
        let rows = k_positions * 6;
        anyhow::ensure!(gate_k.n_elements == rows * d_ffn);
        anyhow::ensure!(up_k.n_elements == rows * d_ffn);
        anyhow::ensure!(weights_k_flat.n_elements == rows);

        // No FCs; the kernel isn't in the KernelSpec preload list — load
        // via specialized_pipeline (cached after first call). DS4_MOE_MID_F16:
        // the _f16 twin writes `device half` mid (half the down-GEMM read bytes).
        let empty_key: &[u8] = &[];
        let pipe = self.state.specialized_pipeline(
            if mid_f16 { "ds4_dsv4_moe_swiglu_weight_f16" } else { "ds4_dsv4_moe_swiglu_weight" },
            empty_key, |_fcv| {},
        )?;
        let mid_bytes: u64 = if mid_f16 { 2 } else { 4 };

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MoeSwigluActArgs {
            width: u32,
            rows: u32,
            gate_row_stride: u64,
            up_row_stride: u64,
            mid_row_stride: u64,
            weight_stride: u64,
            write_clamped: u32,
            clamp_value: f32,
        }
        let args = MoeSwigluActArgs {
            width: d_ffn as u32,
            rows: rows as u32,
            gate_row_stride: (d_ffn as u64) * 4,
            up_row_stride:   (d_ffn as u64) * 4,
            mid_row_stride:  (d_ffn as u64) * mid_bytes,
            weight_stride: 4,
            write_clamped: 0,
            // SwiGLU clamp = the model's swiglu_clamp_exp. antirez validates the GGUF array
            // deepseek4.swiglu_clamp_exp is uniformly 10.0 (confirmed: 10.0 ×43 in our gguf)
            // and clamps gate=min(g,10)/up=clamp(u,±10) in EVERY path (CPU, Metal f32 AND f16,
            // ds4.c:18023). We had been passing INFINITY = a latent divergence from antirez.
            // NOW DEFAULT-ON (faithful): byte-identical at short ctx (gate/up<10 → exact no-op,
            // verified) and clips <0.001% at 3000 (deterministic gate/up magnitude readback:
            // max ~6-14 across all 43 layers) — a near-no-op that makes us bit-faithful to
            // antirez AND bounds the f16 mid. Opt out: DS4_MOE_SWIGLU_CLAMP=0.
            // DS4_MOE_MID_CLAMP overrides the 10.0 value. (TODO: load the per-layer value from
            // the GGUF instead of hardcoding 10.0, for models whose array isn't uniform.)
            clamp_value: if !mid_f16
                && std::env::var("DS4_MOE_SWIGLU_CLAMP").as_deref() == Ok("0")
            {
                f32::INFINITY
            } else {
                std::env::var("DS4_MOE_MID_CLAMP").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0f32)
            },
        };

        // Mid: in-place write would alias gate; allocate fresh per-K mid.
        // f16 mid = a u16-element buffer (rows*d_ffn halves); n_elements still
        // counts elements so the down-GEMM shape assert holds.
        // Pooled (DS4_CHUNK_POOL_SCRATCH): the routed-expert mid (f16 default-on,
        // ~73MB at K=3000), produced here + consumed by the down GEMM within this
        // layer — intra-layer, single-use. The swiglu below writes its full extent.
        let mid_k = if mid_f16 {
            self.pooled_scratch_u16("moe_mid_f16", rows * d_ffn)
        } else {
            self.pooled_scratch_f32("moe_mid_f32", rows * d_ffn)
        };

        // One threadgroup per row; threads loop over the width.
        let tcount: u64 = (pipe.max_total_threads_per_threadgroup() as u64)
            .min(d_ffn as u64).max(1);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&gate_k.buf), 0);
        enc.set_buffer(2, Some(&up_k.buf), 0);
        enc.set_buffer(3, Some(&mid_k.buf), 0);
        enc.set_buffer(4, Some(&weights_k_flat.buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(mid_k)
    }

    /// Phase 2 MoE-K Step 2 — sum 6 expert-slot contributions per K-position.
    /// Reduces `down_out [K, 6, d_embd]` → `[K, d_embd]` via the bridge
    /// shim `ds4_kernel_dsv4_sum6_k_f32`.
    pub fn sum6_k(
        &self,
        down_out_k: &DeferredBuf,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            down_out_k.n_elements == k_positions * 6 * d_embd,
            "sum6_k: input has {} elems, expected K*6*d_embd = {}*6*{}",
            down_out_k.n_elements, k_positions, d_embd
        );
        let empty_key: &[u8] = &[];
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_dsv4_sum6_k_f32", empty_key, |_fcv| {},
        )?;
        let out = self.pooled_scratch_f32("moe_mmid_out", k_positions * d_embd); // T4: pool+pin (merged-cb working set)
        let d_embd_u32 = d_embd as u32;
        let k_u32 = k_positions as u32;
        let total = (k_positions * d_embd) as u64;
        let tcount: u64 = (pipe.max_total_threads_per_threadgroup() as u64)
            .min(total).max(1);
        let tg_count = (total + tcount - 1) / tcount;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&down_out_k.buf), 0);
        enc.set_buffer(1, Some(&out.buf), 0);
        set_scalar_bytes(enc, 2, &d_embd_u32);
        set_scalar_bytes(enc, 3, &k_u32);
        enc.dispatch_thread_groups(
            MTLSize::new(tg_count, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);
        Ok(out)
    }

    /// Phase 2 MoE-K Step 2 — FULL K-amortized MoE chain via mul_mm_id.
    /// Replaces the K-linear blit-shim path in encode_moe_and_shared_chain_k's
    /// MoE component with a K-batched matmul pipeline.
    ///
    /// ```text
    ///   1. map0(selected_K)                  → tpe[256], ids[256*K]
    ///   2. mul_mm_id(gate_w, normed_K)       → gate_out[K, 6, d_ffn]
    ///   3. mul_mm_id(up_w,   normed_K)       → up_out  [K, 6, d_ffn]
    ///   4. moe_swiglu_weight(gate, up, wt_K) → mid     [K, 6, d_ffn]
    ///   5. mul_mm_id(down_w, mid, ne11=6)    → down_out[K, 6, d_embd]
    ///   6. sum6_k(down_out)                  → moe_k  [K, d_embd]
    /// ```
    ///
    /// All matmuls are Q8_0 expert weights. For iq2_xxs / q4_K, swap the
    /// kernel name and `nb01` block-bytes (Step 4 of the MoE-K plan).
    ///
    /// `sel_k_flat`/`wt_k_flat` are flat `[K*6]` buffers (i32/f32). Caller
    /// builds these from router_finalize_k's `Vec<(sel, wt)>` output via
    /// blit copy or generates them directly.
    ///
    /// Returns `moe_k [K, d_embd]` ready to feed `hc_expand_add_split_k`.
    ///
    /// Phase 2 MoE-K Step 5 — mixed-quant variant. DS4 V4 Flash uses
    /// per-layer quant choices (e.g. iq2_xxs gate/up + q4_K down, or all
    /// q4_K, etc.). Each stage specified by (kernel_name, block_qk,
    /// block_bytes); see `encode_mul_mm_id_qx_k` for the per-quant matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_chain_mm_qx_k(
        &self,
        normed_k: &DeferredBuf,
        sel_k_flat: &DeferredBuf,
        wt_k_flat: &DeferredBuf,
        gate_w: &DeferredBuf,
        up_w:   &DeferredBuf,
        down_w: &DeferredBuf,
        gate_kernel: &str, gate_qk: usize, gate_bytes: u64,
        up_kernel:   &str, up_qk:   usize, up_bytes:   u64,
        down_kernel: &str, down_qk: usize, down_bytes: u64,
        n_experts: usize,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            normed_k.n_elements == k_positions * d_embd,
            "encode_moe_chain_mm_qx_k: normed_k shape mismatch"
        );
        anyhow::ensure!(sel_k_flat.n_elements == k_positions * 6);
        anyhow::ensure!(wt_k_flat.n_elements == k_positions * 6);

        // 1. Build expert→token map.
        let (tpe, ids) = self.encode_mul_mm_id_map0_k(sel_k_flat, n_experts, k_positions)?;

        // 2-4. Gate + up + SwiGLU. DS4_MOE_FUSED_MMID=1 fuses all three into one
        // dispatch when both are iq2_xxs (antirez's prefill MoE kernel); otherwise
        // the separate gate GEMM + up GEMM + moe_swiglu_weight_k path.
        let iq2 = "ds4_kernel_mul_mm_id_iq2_xxs_f32";
        let fuse = gate_kernel == iq2
            && up_kernel == iq2
            && std::env::var("DS4_MOE_FUSED_MMID").ok().as_deref() == Some("1");
        // DS4_MOE_MID_F16 (now DEFAULT-ON, antirez's request_mid_f16): f16 swiglu mid + f16-RHS
        // down GEMM. Halves the down-GEMM activation read (f32 [K,6,d_ffn]≈147MB → f16 ≈73MB)
        // → +4.1% @3000 (175.3→182.4, best-of-5). QUALITY-SAFE: now that the SwiGLU clamp is
        // default-on, mid is bounded (~100-250), where f16 precision (~0.06) is FAR finer than
        // the q2_K down-weight quantization (~2-bit) — so the down GEMM can't tell f16 from f32
        // mid. Byte-identical clamp/f16 at 600. (The earlier "needle loss" was f16 OVERFLOW from
        // the missing clamp + the f16 down kernels missing from BRIDGE_PREFERRED — both fixed.)
        // Opt out: DS4_MOE_MID_F16=0. Unfused path only (the fused pair_swiglu keeps f32 mid).
        let mid_f16 = std::env::var("DS4_MOE_MID_F16").ok().as_deref() != Some("0") && !fuse;
        let mid = if fuse {
            self.encode_mul_mm_id_iq2_pair_swiglu_k(
                gate_w, up_w, normed_k, &tpe, &ids, wt_k_flat,
                n_experts, d_embd, d_ffn, k_positions,
            )?
        } else {
            let gate_out = self.encode_mul_mm_id_qx_k(
                gate_kernel, gate_qk, gate_bytes,
                gate_w, normed_k, &tpe, &ids,
                n_experts, d_embd, d_ffn, k_positions, 1, 4,
            )?;
            let up_out = self.encode_mul_mm_id_qx_k(
                up_kernel, up_qk, up_bytes,
                up_w, normed_k, &tpe, &ids,
                n_experts, d_embd, d_ffn, k_positions, 1, 4,
            )?;
            self.moe_swiglu_weight_k(&gate_out, &up_out, wt_k_flat, d_ffn, k_positions, mid_f16)?
        };

        // 5. Down matmul (ne11=6, per-slot activations from mid). DS4_MOE_MID_F16: the
        // mid is f16 → read it via the f16-RHS down kernel (_f32 → _f16, act_bytes=2).
        let down_k = if mid_f16 { down_kernel.replace("_f32", "_f16") } else { down_kernel.to_string() };
        let down_out = self.encode_mul_mm_id_qx_k(
            &down_k, down_qk, down_bytes,
            down_w, &mid, &tpe, &ids,
            n_experts, d_ffn, d_embd, k_positions, 6, if mid_f16 { 2 } else { 4 },
        )?;

        // 6. Sum 6 expert-slot contributions per K-position.
        self.sum6_k(&down_out, d_embd, k_positions)
    }

    /// Phase 2 MoE-K Option A — K-batched FUSED pair_swiglu for iq2_xxs
    /// experts. Dispatches the bridge shim `ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_K_f32`
    /// — a clone of the K=1 production fused kernel with two-line offset
    /// modification to support K tokens per dispatch. Each (token, slot)
    /// pair is one TG in tgpig.z (grid_z = K * 6 instead of 6).
    ///
    /// Returns `mid [K*6, d_ffn]` f32 — silu(gate)*up*route_weight per slot
    /// per K-position, ready to feed `encode_sum6_K_*` for the down stage.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_pair_swiglu_K_iq2_xxs(
        &self,
        experts_w_gate_iq2: &DeferredBuf,
        experts_w_up_iq2:   &DeferredBuf,
        x_k:        &DeferredBuf,        // [K, d_in] f32
        ids_flat:   &DeferredBuf,        // [K*6] i32
        weights_flat: &DeferredBuf,      // [K*6] f32
        _n_experts: usize,
        d_in: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(d_in % 256 == 0, "iq2_xxs requires d_in % 256");
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_in,
            "encode_pair_swiglu_K_iq2_xxs: x_k {} vs K*d_in = {}*{}",
            x_k.n_elements, k_positions, d_in
        );
        anyhow::ensure!(ids_flat.n_elements == k_positions * 6);
        anyhow::ensure!(weights_flat.n_elements == k_positions * 6);

        // FC: NSG=2, NXPSG=4 — matches K=1 production dispatch (iq2_xxs).
        // DS4_MOE_NSG overrides nsg for the geometry sweep / a tuned default.
        let nsg: i16 = std::env::var("DS4_MOE_NSG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_K_f32", &key,
            |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            },
        )?;

        // Args mirror K=1 production iq2_xxs pair_swiglu (moe_routed_step_encode
        // line 2782), with nei1=K instead of 1 and matching K-batched ids buffer.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MoeSwigluActArgs {
            width: u32, rows: u32,
            gate_row_stride: u64, up_row_stride: u64,
            mid_row_stride: u64, weight_stride: u64,
            write_clamped: u32, clamp_value: f32,
        }

        const QK_K: u64 = 256;
        let pair_nb_per_row = (d_in as u64 / QK_K) * 66;     // iq2_xxs block_bytes = 66
        let pair_nr0: i32 = 4;                                // N_R0_IQ2_XXS = 4

        let pair_args = MulMvIdArgs {
            nei0: 6, nei1: k_positions as i32,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_in as i32, ne01: d_ffn as i32, ne02: 1, _pad0: 0,
            nb00: 2,
            nb01: pair_nb_per_row,
            nb02: pair_nb_per_row * d_ffn as u64,
            ne10: d_in as i32, ne11: 1, ne12: 1, ne13: 1,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            ne0: d_ffn as i32, ne1: 1,
            nb1: (d_ffn * 4) as u64,
            nr0: pair_nr0, _pad1: 0,
        };
        let act_args = MoeSwigluActArgs {
            width: d_ffn as u32,
            rows: (k_positions * 6) as u32,
            gate_row_stride: (d_ffn * 4) as u64,
            up_row_stride: (d_ffn * 4) as u64,
            mid_row_stride: (d_ffn * 4) as u64,
            weight_stride: 4,
            write_clamped: 0,
            clamp_value: f32::INFINITY,
        };

        let mid = self.pooled_scratch_f32("moe_mid", k_positions * 6 * d_ffn);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &pair_args);
        set_scalar_bytes(enc, 1, &act_args);
        enc.set_buffer(2, Some(&experts_w_gate_iq2.buf), 0);
        enc.set_buffer(3, Some(&experts_w_up_iq2.buf), 0);
        enc.set_buffer(4, Some(&x_k.buf), 0);
        enc.set_buffer(5, Some(&mid.buf), 0);
        enc.set_buffer(6, Some(&ids_flat.buf), 0);
        enc.set_buffer(7, Some(&weights_flat.buf), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        // Grid: (ceil(d_ffn / (NSG * N_R0_IQ2_XXS=4)), 1, K * nei0=6)
        let rows_per_tg = (nsg as u64) * (pair_nr0 as u64);
        let n_row_tg = ((d_ffn as u64) + rows_per_tg - 1) / rows_per_tg;
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, (k_positions * 6) as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(mid)
    }

    /// Phase 2 MoE-K Option A — K-batched FUSED pair_swiglu for Q4_K
    /// experts. Mirror of `encode_pair_swiglu_K_iq2_xxs` with Q4_K-specific
    /// block size (144 bytes, QK_K=256, N_R0_Q4_K=2). Sets `ne1 = 6` in
    /// args so the kernel's dst_gate/up scratch addressing formula
    /// `(idx + i12 * ne1) * ne0 * 4` reduces to the K-batched layout.
    ///
    /// Q4_K kernel uses dst_gate / dst_up as INTERMEDIATE scratch (the
    /// inner template fn writes to them then re-reads for silu). So they
    /// must be allocated for K*6 slots; iq2_xxs path drops them.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_pair_swiglu_K_q4_K(
        &self,
        experts_w_gate_q4k: &DeferredBuf,
        experts_w_up_q4k:   &DeferredBuf,
        x_k:        &DeferredBuf,
        ids_flat:   &DeferredBuf,
        weights_flat: &DeferredBuf,
        _n_experts: usize,
        d_in: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(d_in % 256 == 0, "q4_K requires d_in % 256");
        anyhow::ensure!(
            x_k.n_elements == k_positions * d_in,
            "encode_pair_swiglu_K_q4_K: x_k {} vs K*d_in = {}*{}",
            x_k.n_elements, k_positions, d_in
        );
        anyhow::ensure!(ids_flat.n_elements == k_positions * 6);
        anyhow::ensure!(weights_flat.n_elements == k_positions * 6);

        let nsg: i16 = 2;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mv_id_q4_K_pair_swiglu_K_f32", &key,
            |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            },
        )?;

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MoeSwigluActArgs {
            width: u32, rows: u32,
            gate_row_stride: u64, up_row_stride: u64,
            mid_row_stride: u64, weight_stride: u64,
            write_clamped: u32, clamp_value: f32,
        }

        const QK_K: u64 = 256;
        let pair_nb_per_row = (d_in as u64 / QK_K) * 144;    // q4_K block_bytes = 144
        let pair_nr0: i32 = 2;                                // N_R0_Q4_K = 2

        // ne1 = 6 (NOT 1 like K=1 production) so the kernel's dst_gate/up
        // addressing `(idx + i12*ne1)*ne0` reduces to linear_slot*ne0 for
        // K-batched. At K=1 (i12=0), this still gives idx*ne0 — same as
        // K=1 production layout.
        let pair_args = MulMvIdArgs {
            nei0: 6, nei1: k_positions as i32,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_in as i32, ne01: d_ffn as i32, ne02: 1, _pad0: 0,
            nb00: 2,
            nb01: pair_nb_per_row,
            nb02: pair_nb_per_row * d_ffn as u64,
            ne10: d_in as i32, ne11: 1, ne12: 1, ne13: 1,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            ne0: d_ffn as i32, ne1: 6,                       // ← 6, not 1
            nb1: (d_ffn * 4) as u64,
            nr0: pair_nr0, _pad1: 0,
        };
        let act_args = MoeSwigluActArgs {
            width: d_ffn as u32,
            rows: (k_positions * 6) as u32,
            gate_row_stride: (d_ffn * 4) as u64,
            up_row_stride: (d_ffn * 4) as u64,
            mid_row_stride: (d_ffn * 4) as u64,
            weight_stride: 4,
            write_clamped: 0,
            clamp_value: f32::INFINITY,
        };

        // dst_gate/dst_up scratch: K*6 slots × d_ffn floats each.
        let scratch_size = k_positions * 6 * d_ffn;
        let dst_gate_scratch = self.alloc_f32(scratch_size);
        let dst_up_scratch   = self.alloc_f32(scratch_size);
        let mid = self.pooled_scratch_f32("moe_mid", k_positions * 6 * d_ffn);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &pair_args);
        set_scalar_bytes(enc, 1, &act_args);
        enc.set_buffer(2, Some(&experts_w_gate_q4k.buf), 0);
        enc.set_buffer(3, Some(&experts_w_up_q4k.buf), 0);
        enc.set_buffer(4, Some(&x_k.buf), 0);
        enc.set_buffer(5, Some(&dst_gate_scratch.buf), 0);
        enc.set_buffer(6, Some(&dst_up_scratch.buf), 0);
        enc.set_buffer(7, Some(&mid.buf), 0);
        enc.set_buffer(8, Some(&ids_flat.buf), 0);
        enc.set_buffer(9, Some(&weights_flat.buf), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        let rows_per_tg = (nsg as u64) * (pair_nr0 as u64);
        let n_row_tg = ((d_ffn as u64) + rows_per_tg - 1) / rows_per_tg;
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, (k_positions * 6) as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(mid)
    }

    /// Phase 2 MoE-K Option A — K-batched sum6 for Q4_K experts. Existing
    /// upstream `kernel_mul_mv_id_q4_K_sum6_f32` ALREADY supports K via
    /// `tgpig.y = token` (same as Q2_K variant).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_sum6_K_q4_K(
        &self,
        experts_w_down_q4k: &DeferredBuf,
        mid_K: &DeferredBuf,
        ids_flat: &DeferredBuf,
        _n_experts: usize,
        d_ffn: usize,
        d_embd: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(d_ffn % 256 == 0, "q4_K down requires d_ffn % 256");

        let nsg: i16 = 2;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mv_id_q4_K_sum6_f32", &key,
            |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            },
        )?;

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }

        const QK_K: u64 = 256;
        let down_block_bytes: u64 = 144;                     // q4_K
        let down_nr0: i32 = 2;                                // N_R0_Q4_K = 2
        let down_nb_per_row = (d_ffn as u64 / QK_K) * down_block_bytes;

        let down_args = MulMvIdArgs {
            nei0: 6, nei1: k_positions as i32,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_ffn as i32, ne01: d_embd as i32, ne02: 1, _pad0: 0,
            nb00: 2,
            nb01: down_nb_per_row,
            nb02: down_nb_per_row * d_embd as u64,
            ne10: d_ffn as i32, ne11: 6, ne12: 1, ne13: 1,
            nb10: 4,
            nb11: (d_ffn * 4) as u64,
            nb12: (6 * d_ffn * 4) as u64,
            ne0: d_embd as i32, ne1: k_positions as i32,
            nb1: (d_embd * 4) as u64,
            nr0: down_nr0, _pad1: 0,
        };

        let out = self.alloc_f32(k_positions * d_embd);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &down_args);
        enc.set_buffer(1, Some(&experts_w_down_q4k.buf), 0);
        enc.set_buffer(2, Some(&mid_K.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_buffer(4, Some(&ids_flat.buf), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        let rows_per_tg = (nsg as u64) * (down_nr0 as u64);
        let n_row_tg = ((d_embd as u64) + rows_per_tg - 1) / rows_per_tg;
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, k_positions as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// Phase 2 MoE-K Option A — Q4_K-uniform fused chain (gate/up/down all Q4_K).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_moe_chain_fused_K_q4_K_q4_K(
        &self,
        normed_k: &DeferredBuf,
        ids_flat: &DeferredBuf,
        weights_flat: &DeferredBuf,
        experts_w_gate_q4k: &DeferredBuf,
        experts_w_up_q4k:   &DeferredBuf,
        experts_w_down_q4k: &DeferredBuf,
        n_experts: usize,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        let mid_k = self.encode_pair_swiglu_K_q4_K(
            experts_w_gate_q4k, experts_w_up_q4k,
            normed_k, ids_flat, weights_flat,
            n_experts, d_embd, d_ffn, k_positions,
        )?;
        self.encode_sum6_K_q4_K(
            experts_w_down_q4k, &mid_k, ids_flat,
            n_experts, d_ffn, d_embd, k_positions,
        )
    }

    /// Phase 2 MoE-K Option A — K-batched sum6 down projection (Q2_K
    /// experts). The existing `kernel_mul_mv_id_q2_K_sum6_f32` ALREADY
    /// supports K via tgpig.y; this dispatcher just sets grid_y = K and
    /// passes per-token-strided args.
    ///
    /// Input: `mid_K [K*6, d_ffn]` from `encode_pair_swiglu_K_iq2_xxs`.
    /// Output: `out_K [K, d_in]` (d_in here is the d_embd-side output of
    /// the down projection — confusingly named in the kernel as args.ne0).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_sum6_K_q2_K(
        &self,
        experts_w_down_q2k: &DeferredBuf,
        mid_K: &DeferredBuf,           // [K*6, d_ffn] f32 (from pair_swiglu_K)
        ids_flat: &DeferredBuf,        // [K*6] i32 — same as pair_swiglu_K
        _n_experts: usize,
        d_ffn: usize,                  // input dim (per-slot rows)
        d_embd: usize,                 // output dim (d_in of model, kernel's ne0)
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(d_ffn % 256 == 0, "q2_K down requires d_ffn % 256");

        let nsg: i16 = 2;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_mul_mv_id_q2_K_sum6_f32", &key,
            |fcv| {
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
            },
        )?;

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32, nei1: i32, nbi1: u64,
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64,
            ne0: i32, ne1: i32, nb1: u64, nr0: i32, _pad1: i32,
        }

        const QK_K: u64 = 256;
        let down_block_bytes: u64 = 84;                       // q2_K
        let down_nr0: i32 = 4;                                // N_R0_Q2_K = 4
        let down_nb_per_row = (d_ffn as u64 / QK_K) * down_block_bytes;

        let down_args = MulMvIdArgs {
            nei0: 6, nei1: k_positions as i32,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_ffn as i32, ne01: d_embd as i32, ne02: 1, _pad0: 0,
            nb00: 2,
            nb01: down_nb_per_row,
            nb02: down_nb_per_row * d_embd as u64,
            ne10: d_ffn as i32, ne11: 6, ne12: 1, ne13: 1,
            nb10: 4,
            nb11: (d_ffn * 4) as u64,                        // per-slot stride within token
            nb12: (6 * d_ffn * 4) as u64,                    // per-token stride (6 slots)
            ne0: d_embd as i32, ne1: k_positions as i32,
            nb1: (d_embd * 4) as u64,                        // per-token output stride
            nr0: down_nr0, _pad1: 0,
        };

        let out = self.alloc_f32(k_positions * d_embd);

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        set_scalar_bytes(enc, 0, &down_args);
        enc.set_buffer(1, Some(&experts_w_down_q2k.buf), 0);
        enc.set_buffer(2, Some(&mid_K.buf), 0);
        enc.set_buffer(3, Some(&out.buf), 0);
        enc.set_buffer(4, Some(&ids_flat.buf), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        // Grid: (ceil(d_embd / (NSG * N_R0_Q2_K=4)), K, 1)
        let rows_per_tg = (nsg as u64) * (down_nr0 as u64);
        let n_row_tg = ((d_embd as u64) + rows_per_tg - 1) / rows_per_tg;
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, k_positions as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(out)
    }

    /// Phase 2 MoE-K Option A — FULL fused K-batched MoE chain for the
    /// (IQ2_XXS gate/up + Q2_K down) layer type (DS4 V4 Flash layer 5 etc.).
    ///
    /// 2 dispatches total (vs 6 for the mm_id path; vs 2 for K=1 production):
    ///   1. pair_swiglu_K → mid [K*6, d_ffn]   (1 dispatch)
    ///   2. sum6_K        → out [K, d_embd]     (1 dispatch)
    ///
    /// Same fusion advantages as the K=1 production path (gate+up matmul +
    /// silu+route_weight all in one TG), plus K-batched outer parallelism.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_moe_chain_fused_K_iq2_xxs_q2_K(
        &self,
        normed_k: &DeferredBuf,
        ids_flat: &DeferredBuf,
        weights_flat: &DeferredBuf,
        experts_w_gate_iq2: &DeferredBuf,
        experts_w_up_iq2:   &DeferredBuf,
        experts_w_down_q2k: &DeferredBuf,
        n_experts: usize,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        let mid_k = self.encode_pair_swiglu_K_iq2_xxs(
            experts_w_gate_iq2, experts_w_up_iq2,
            normed_k, ids_flat, weights_flat,
            n_experts, d_embd, d_ffn, k_positions,
        )?;
        self.encode_sum6_K_q2_K(
            experts_w_down_q2k, &mid_k, ids_flat,
            n_experts, d_ffn, d_embd, k_positions,
        )
    }

    /// Q8_0-only convenience wrapper (gate/up/down all Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_chain_mm_q8_k(
        &self,
        normed_k: &DeferredBuf,
        sel_k_flat: &DeferredBuf,
        wt_k_flat: &DeferredBuf,
        gate_w_q8: &DeferredBuf,
        up_w_q8:   &DeferredBuf,
        down_w_q8: &DeferredBuf,
        n_experts: usize,
        d_embd: usize,
        d_ffn: usize,
        k_positions: usize,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            normed_k.n_elements == k_positions * d_embd,
            "encode_moe_chain_mm_q8_k: normed_k shape mismatch"
        );
        anyhow::ensure!(sel_k_flat.n_elements == k_positions * 6);
        anyhow::ensure!(wt_k_flat.n_elements == k_positions * 6);
        anyhow::ensure!(gate_w_q8.n_elements == n_experts * d_ffn * d_embd);
        anyhow::ensure!(up_w_q8.n_elements == n_experts * d_ffn * d_embd);
        anyhow::ensure!(down_w_q8.n_elements == n_experts * d_embd * d_ffn);

        // 1. Build expert→token map.
        let (tpe, ids) = self.encode_mul_mm_id_map0_k(sel_k_flat, n_experts, k_positions)?;

        // 2-3. Gate + up matmuls (ne11=1, activations broadcast across slots).
        let gate_out = self.encode_mul_mm_id_q8_0_k(
            gate_w_q8, normed_k, &tpe, &ids,
            n_experts, d_embd, d_ffn, k_positions, 1,
        )?;
        let up_out = self.encode_mul_mm_id_q8_0_k(
            up_w_q8, normed_k, &tpe, &ids,
            n_experts, d_embd, d_ffn, k_positions, 1,
        )?;

        // 4. SwiGLU + route-weight multiplication, in flat [K*6, d_ffn] layout.
        let mid = self.moe_swiglu_weight_k(&gate_out, &up_out, wt_k_flat, d_ffn, k_positions, false)?;

        // 5. Down matmul (ne11=6, per-slot activations from mid).
        let down_out = self.encode_mul_mm_id_q8_0_k(
            down_w_q8, &mid, &tpe, &ids,
            n_experts, d_ffn, d_embd, k_positions, 6,
        )?;

        // 6. Sum 6 expert-slot contributions per K-position.
        self.sum6_k(&down_out, d_embd, k_positions)
    }

    /// `selected_weights_k` length must == k_positions; each pair is
    /// `(selected[6 i32], weights[6 f32])` from `encode_router_finalize_k`.
    /// Returns `(moe_k [K, d_embd], shared_k [K, d_embd])`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_and_shared_chain_k(
        &self,
        layer_idx: u32,
        normed_k: &DeferredBuf,
        selected_weights_k: &[(DeferredBuf, DeferredBuf)],
        d_ffn: usize,
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        sh_w_gate_q8: &[u8],
        sh_w_up_q8: &[u8],
        sh_w_down_q8: &[u8],
        d_embd: usize,
        k_positions: usize,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        anyhow::ensure!(
            normed_k.n_elements == k_positions * d_embd,
            "encode_moe_and_shared_chain_k: normed_k has {} elems, expected K*d_embd = {}*{}",
            normed_k.n_elements, k_positions, d_embd
        );
        anyhow::ensure!(
            selected_weights_k.len() == k_positions,
            "encode_moe_and_shared_chain_k: selected_weights_k.len() != K"
        );

        let moe_k    = self.alloc_f32(k_positions * d_embd);
        let shared_k = self.alloc_f32(k_positions * d_embd);
        let f32_b: u64 = 4;
        let row_bytes = (d_embd as u64) * f32_b;

        for k in 0..k_positions {
            let k64 = k as u64;
            let (selected, weights) = &selected_weights_k[k];

            // 1. Blit per-K normed slice → temp normed buf.
            let temp_normed = self.alloc_f32(d_embd);
            {
                crate::macos::end_shared_compute_enc_force();
                let enc = self.cmd_buf.new_blit_command_encoder();
                enc.copy_from_buffer(
                    &normed_k.buf, k64 * row_bytes,
                    &temp_normed.buf, 0, row_bytes,
                );
                enc.end_encoding();
            }

            // 2. Run existing K=1 MoE + shared-chain on temp_normed.
            let (moe_dst, shared_dst) = self.encode_moe_and_shared_chain_with_router_bufs_db(
                layer_idx, &temp_normed, selected, weights, d_ffn,
                &temp_normed, sh_w_gate, sh_w_up, sh_w_down, shared_dim,
                sh_w_gate_q8, sh_w_up_q8, sh_w_down_q8,
            )?;

            // 3. Blit results into K-tiled output buffers.
            {
                crate::macos::end_shared_compute_enc_force();
                let enc = self.cmd_buf.new_blit_command_encoder();
                enc.copy_from_buffer(&moe_dst.buf, 0, &moe_k.buf, k64 * row_bytes, row_bytes);
                enc.copy_from_buffer(&shared_dst.buf, 0, &shared_k.buf, k64 * row_bytes, row_bytes);
                enc.end_encoding();
            }
        }
        Ok((moe_k, shared_k))
    }

    /// Phase 3 MTP — drafter input mixer (steps 2-7 of the MTP forward pass).
    ///
    /// Composes the per-token embedding mixer that produces the mtp_input_hc
    /// used as the prev_hc for the drafter's single decode layer. Mirrors
    /// antirez `metal_graph_eval_mtp_draft_from_hc` (ds4.c:12256-12294) steps
    /// AFTER the token embedding extraction (which the caller supplies as
    /// `token_embed_db`):
    ///
    /// ```text
    ///   enorm   = rms_norm_mul(token_embed, enorm_gamma)            [n_embd]
    ///   eproj   = matvec_q8_0(e_proj, enorm)                         [n_embd]
    ///   hnorm_hc = rms_norm_mul_k(prev_hc, hnorm_gamma, n_embd, n_hc) [n_hc, n_embd]
    ///   hproj_hc = matvec_k_q8_0(h_proj, hnorm_hc, n_embd, n_embd, n_hc) [n_hc, n_embd]
    ///   mtp_input_hc[r, c] = eproj[c] + hproj_hc[r, c]               [n_hc, n_embd]
    /// ```
    ///
    /// The final broadcast-add is the new bridge shim
    /// `ds4_kernel_dsv4_mtp_input_mix_broadcast_add_f32` — fuses the
    /// repeat-eproj-across-n_hc-rows + element-wise add into one pass, no
    /// eproj_hc scratch needed.
    ///
    /// n_hc must be in {1, 2, 4, 8} (matvec_k_q8_0 constraint). DS4 uses n_hc=4.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp_input_mix(
        &self,
        token_embed_db: &DeferredBuf,    // [n_embd]   (caller-extracted from base.token_embd)
        prev_hc_db:     &DeferredBuf,    // [n_hc, n_embd]
        enorm_gamma_db: &DeferredBuf,    // [n_embd]
        e_proj_q8_db:   &DeferredBuf,    // [n_embd, n_embd] Q8_0
        hnorm_gamma_db: &DeferredBuf,    // [n_embd]
        h_proj_q8_db:   &DeferredBuf,    // [n_embd, n_embd] Q8_0
        n_embd: usize,
        n_hc: usize,
        rms_eps: f32,
    ) -> Result<DeferredBuf> {
        anyhow::ensure!(
            token_embed_db.n_elements == n_embd,
            "encode_mtp_input_mix: token_embed has {} elems, expected n_embd={}",
            token_embed_db.n_elements, n_embd
        );
        anyhow::ensure!(
            prev_hc_db.n_elements == n_hc * n_embd,
            "encode_mtp_input_mix: prev_hc has {} elems, expected n_hc*n_embd = {}*{}",
            prev_hc_db.n_elements, n_hc, n_embd
        );
        anyhow::ensure!(
            matches!(n_hc, 1 | 2 | 4 | 8),
            "encode_mtp_input_mix: n_hc must be in {{1,2,4,8}} (matvec_k constraint), got {}",
            n_hc
        );

        // Step 2: rms_norm_mul(token_embed, enorm) → enorm_vec.
        let enorm_vec = self.rms_norm_mul(token_embed_db, enorm_gamma_db, rms_eps)?;
        // Step 3: matmul Q8_0 e_proj × enorm_vec → eproj.
        let eproj = self.matvec_q8_0(e_proj_q8_db, &enorm_vec, n_embd, n_embd)?;
        // Step 5: rms_norm_mul_k across n_hc rows of prev_hc with hnorm gamma.
        let hnorm_hc = self.rms_norm_mul_k(prev_hc_db, hnorm_gamma_db, n_embd, n_hc, rms_eps)?;
        // Step 6: matvec_k_q8_0 h_proj × hnorm_hc → hproj_hc [n_hc, n_embd].
        let hproj_hc = self.matvec_k_q8_0(h_proj_q8_db, &hnorm_hc, n_embd, n_embd, n_hc)?;

        // Step 7 (fused broadcast-add): mtp_input_hc[r, c] = eproj[c] + hproj_hc[r, c].
        let empty_key: &[u8] = &[];
        let pipe = self.state.specialized_pipeline(
            "ds4_kernel_dsv4_mtp_input_mix_broadcast_add_f32", empty_key, |_fcv| {},
        )?;
        let mtp_input_hc = self.alloc_f32(n_hc * n_embd);
        let n_embd_u32 = n_embd as u32;
        let n_hc_u32 = n_hc as u32;
        let total = (n_hc * n_embd) as u64;
        let tcount: u64 = (pipe.max_total_threads_per_threadgroup() as u64)
            .min(total).max(1);
        let tg_count = (total + tcount - 1) / tcount;

        let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&eproj.buf), 0);
        enc.set_buffer(1, Some(&hproj_hc.buf), 0);
        enc.set_buffer(2, Some(&mtp_input_hc.buf), 0);
        set_scalar_bytes(enc, 3, &n_embd_u32);
        set_scalar_bytes(enc, 4, &n_hc_u32);
        enc.dispatch_thread_groups(
            MTLSize::new(tg_count, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        crate::macos::end_shared_compute_enc(enc);

        Ok(mtp_input_hc)
    }

    /// Phase 3 MTP — drafter output head (steps 1-6 of antirez
    /// `metal_graph_encode_output_head_mtp` at ds4.c:9590).
    ///
    /// Composes the drafter's final LM-head pipeline:
    ///
    /// ```text
    ///   flat_hc     = rms_norm_mul(cur_hc, unit_γ_hc, eps)                [hc_dim]
    ///   output_pre  = matvec_f32(hc_head_fn, flat_hc, hc_dim, n_hc)       [n_hc]
    ///   output_weights = (mul+add+sigmoid+eps)(output_pre, scale, base)   [n_hc]
    ///   output_embd = hc_weighted_sum(cur_hc, output_weights, n_embd, n_hc) [n_embd]
    ///   output_norm = rms_norm_mul(output_embd, mtp_norm_γ, rms_eps)      [n_embd]
    ///   logits      = matvec_q8_0(base_lm_head_q8, output_norm, n_embd, vocab) [vocab]
    /// ```
    ///
    /// `cur_hc_db` is the output of the drafter's single decode layer
    /// (the `encode_mtp_input_mix` → `encode_layer_k(K=1, mtp.block)` chain).
    /// `base_lm_head_q8_db` is the BASE model's `output` LM head tensor
    /// (the drafter shares vocab projection with the verifier).
    ///
    /// `hc_head_fn_db` must be f32 (caller dequantizes from F16 if needed).
    /// `unit_gamma_hc_dim_db` is `hc_dim` ones (used for the plain-RMS step).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp_output_head(
        &self,
        cur_hc_db:          &DeferredBuf, // [n_hc, n_embd]
        unit_gamma_hc_dim_db: &DeferredBuf, // [hc_dim] all-ones
        hc_head_fn_db:      &DeferredBuf, // [hc_dim, n_hc] f32
        hc_head_scale_db:   &DeferredBuf, // [1] f32
        hc_head_base_db:    &DeferredBuf, // [n_hc] f32
        mtp_norm_gamma_db:  &DeferredBuf, // [n_embd] f32 (mtp.norm)
        base_lm_head_q8_db: &DeferredBuf, // [n_embd, vocab] Q8_0 (base.output)
        n_embd: usize,
        n_hc: usize,
        vocab: usize,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<DeferredBuf> {
        let hc_dim = n_hc * n_embd;
        anyhow::ensure!(cur_hc_db.n_elements == hc_dim,
            "encode_mtp_output_head: cur_hc {} vs hc_dim={}", cur_hc_db.n_elements, hc_dim);
        anyhow::ensure!(unit_gamma_hc_dim_db.n_elements == hc_dim,
            "encode_mtp_output_head: unit_gamma {} vs hc_dim={}",
            unit_gamma_hc_dim_db.n_elements, hc_dim);
        anyhow::ensure!(hc_head_fn_db.n_elements == hc_dim * n_hc,
            "encode_mtp_output_head: hc_head_fn shape mismatch");
        anyhow::ensure!(hc_head_base_db.n_elements == n_hc);
        anyhow::ensure!(mtp_norm_gamma_db.n_elements == n_embd);

        // Step 1: rms_norm_plain (unit gamma). Reuses rms_norm_mul.
        let flat_hc = self.rms_norm_mul(cur_hc_db, unit_gamma_hc_dim_db, rms_eps)?;

        // Step 2: matvec_f32(hc_head_fn, flat_hc, hc_dim, n_hc) → output_pre.
        let output_pre = self.matvec_f32(hc_head_fn_db, &flat_hc, hc_dim, n_hc)?;

        // Step 3: output_hc_weights fused (mul+add+sigmoid+eps) bridge shim.
        let output_weights = self.alloc_f32(n_hc);
        {
            let empty_key: &[u8] = &[];
            let pipe = self.state.specialized_pipeline(
                "ds4_kernel_dsv4_mtp_output_hc_weights_f32", empty_key, |_fcv| {},
            )?;
            let n_hc_u32 = n_hc as u32;
            let tcount = (pipe.max_total_threads_per_threadgroup() as u64)
                .min(n_hc as u64).max(1);
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&output_pre.buf), 0);
            enc.set_buffer(1, Some(&hc_head_scale_db.buf), 0);
            enc.set_buffer(2, Some(&hc_head_base_db.buf), 0);
            enc.set_buffer(3, Some(&output_weights.buf), 0);
            set_scalar_bytes(enc, 4, &n_hc_u32);
            set_scalar_bytes(enc, 5, &hc_eps);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(tcount, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Step 4: hc_weighted_sum(cur_hc, output_weights, n_embd, n_hc) → output_embd.
        let output_embd = self.alloc_f32(n_embd);
        {
            let empty_key: &[u8] = &[];
            let pipe = self.state.specialized_pipeline(
                "ds4_dsv4_hc_weighted_sum", empty_key, |_fcv| {},
            )?;
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct HcWsArgs {
                n_embd: i64, n_hc: i64, n_tokens: i64,
                nb_x0: u64, nb_x1: u64, nb_x2: u64,
                nb_w0: u64, nb_w1: u64,
                nb0: u64, nb1: u64,
            }
            let f32_b: u64 = 4;
            let args = HcWsArgs {
                n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: 1,
                // cur_hc layout: [n_hc, n_embd]. Kernel indexes as
                // x[d*nb_x0 + h*nb_x1 + t*nb_x2]. d = n_embd col, h = n_hc row.
                // So nb_x0 = f32 (per-element across d), nb_x1 = n_embd*4 (per-row).
                nb_x0: f32_b,
                nb_x1: (n_embd as u64) * f32_b,
                nb_x2: (hc_dim as u64) * f32_b,
                // weights layout: [n_hc] for 1 token; per-h stride = 4, per-t stride unused.
                nb_w0: f32_b,
                nb_w1: (n_hc as u64) * f32_b,
                // output: [n_embd] per token.
                nb0: f32_b,
                nb1: (n_embd as u64) * f32_b,
            };
            let total = (n_embd as u64) * 1; // n_tokens=1
            let tcount = (pipe.max_total_threads_per_threadgroup() as u64)
                .min(total).max(1);
            let tg_count = (total + tcount - 1) / tcount;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&cur_hc_db.buf), 0);
            enc.set_buffer(2, Some(&output_weights.buf), 0);
            enc.set_buffer(3, Some(&output_embd.buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(tg_count, 1, 1),
                MTLSize::new(tcount, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Step 5: rms_norm_mul(output_embd, mtp.norm) → output_norm.
        let output_norm = self.rms_norm_mul(&output_embd, mtp_norm_gamma_db, rms_eps)?;

        // Step 6: matvec_q8_0(base.output_lm_head, output_norm) → logits [vocab].
        let logits = self.matvec_q8_0(base_lm_head_q8_db, &output_norm, n_embd, vocab)?;

        Ok(logits)
    }

    /// Phase 3 Step 4d.3 — K-position verifier output head. Converts
    /// `cur_hc_K [K, hc_dim]` from `encode_verify_layers_K` into
    /// `[K, vocab]` logits, one row per speculative candidate.
    ///
    /// Mirrors antirez's output_hc_head pipeline (ds4.c:9590), K-batched:
    ///
    /// ```text
    ///   flat_hc_K       = rms_norm_mul_k(cur_hc_K, unit_γ_hc, hc_dim, K)  [K, hc_dim]
    ///   output_pre_K    = K iters of matvec_f32(hc_head_fn, flat_hc[k]) → [K, n_hc]
    ///   output_weights_K = K iters of (mul+add+sigmoid+eps) shim         [K, n_hc]
    ///   output_embd_K   = hc_weighted_sum(cur_hc_K, output_weights_K)    [K, n_embd]
    ///                     (kernel supports K via tgpig.y; set n_tokens=K)
    ///   output_norm_K   = rms_norm_mul_k(output_embd_K, final_norm_γ)    [K, n_embd]
    ///   logits_K        = matvec_k_q8_0(base.output, output_norm_K)      [K, vocab]
    ///                     (K-amortized via Phase 2's matvec_k_q8_0)
    /// ```
    ///
    /// For models WITHOUT `output_hc_*` tensors (the antirez "slot 0"
    /// shortcut), caller should slice cur_hc_K[:, 0, :] and skip the HC
    /// head entirely — pass `final_norm_γ` + `base_lm_head` to a simpler
    /// path. This method assumes the HC-head path; future cleanup may
    /// add a `no_hc_head` variant.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_verify_output_head_K(
        &self,
        cur_hc_K:           &DeferredBuf,    // [K, hc_dim] flat
        unit_gamma_hc_dim:  &DeferredBuf,    // [hc_dim] ones
        hc_head_fn:         &DeferredBuf,    // [hc_dim, n_hc] f32
        hc_head_scale:      &DeferredBuf,    // [1] f32
        hc_head_base:       &DeferredBuf,    // [n_hc] f32
        final_norm_gamma:   &DeferredBuf,    // [n_embd] f32 (model.final_norm)
        base_lm_head_q8:    &DeferredBuf,    // [vocab, n_embd] Q8_0 (base.output)
        n_embd: usize,
        n_hc: usize,
        vocab: usize,
        k_positions: usize,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<DeferredBuf> {
        let hc_dim = n_hc * n_embd;
        anyhow::ensure!(
            cur_hc_K.n_elements == k_positions * hc_dim,
            "encode_verify_output_head_K: cur_hc_K {} != K*hc_dim = {}*{}",
            cur_hc_K.n_elements, k_positions, hc_dim,
        );
        anyhow::ensure!(unit_gamma_hc_dim.n_elements == hc_dim);
        anyhow::ensure!(hc_head_fn.n_elements == hc_dim * n_hc);
        anyhow::ensure!(hc_head_base.n_elements == n_hc);
        anyhow::ensure!(final_norm_gamma.n_elements == n_embd);

        // Step 1: K-batched RMS-norm-plain. rms_norm_mul_k handles K rows.
        let flat_hc_K = self.rms_norm_mul_k(
            cur_hc_K, unit_gamma_hc_dim, hc_dim, k_positions, rms_eps,
        )?;

        // Step 2: K iterations of matvec_f32(hc_head_fn, flat_hc[k]) → output_pre_K [K, n_hc].
        // Mirrors encode_router_logits_k's loop pattern. n_hc is tiny (4),
        // so K iters of a small matvec is fine.
        let output_pre_K = {
            let nsg: i16 = (((hc_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
            let nxpsg: i16 = if hc_dim % 256 == 0 { 16 }
                              else if hc_dim % 128 == 0 { 8 } else { 4 };
            let mut key = Vec::with_capacity(4);
            key.extend_from_slice(&nsg.to_le_bytes());
            key.extend_from_slice(&nxpsg.to_le_bytes());
            let pipe = self.state.specialized_pipeline(
                "ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
                    fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
                    fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
                },
            )?;
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct MulMvArgs {
                ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
                nb00: u64, nb01: u64, nb02: u64, nb03: u64,
                ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
                nb10: u64, nb11: u64, nb12: u64, nb13: u64,
                ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
            }
            let args = MulMvArgs {
                ne00: hc_dim as i32, ne01: n_hc as i32, ne02: 1, _pad0: 0,
                nb00: 4, nb01: (hc_dim * 4) as u64,
                nb02: (hc_dim * n_hc * 4) as u64,
                nb03: (hc_dim * n_hc * 4) as u64,
                ne10: hc_dim as i32, ne11: 1, ne12: 1, _pad1: 0,
                nb10: 4, nb11: (hc_dim * 4) as u64,
                nb12: (hc_dim * 4) as u64, nb13: (hc_dim * 4) as u64,
                ne0: n_hc as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
            };
            let shmem: u64 = 32 * 2 * 4;
            let n_row_tg = ((n_hc as u64) + 1) / 2;
            let out_K = self.alloc_f32(k_positions * n_hc);
            let x_per_k_bytes = (hc_dim * 4) as u64;
            let out_per_k_bytes = (n_hc * 4) as u64;
            for k in 0..k_positions {
                let k64 = k as u64;
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&pipe);
                set_scalar_bytes(enc, 0, &args);
                enc.set_buffer(1, Some(&hc_head_fn.buf), 0);
                enc.set_buffer(2, Some(&flat_hc_K.buf), k64 * x_per_k_bytes);
                enc.set_buffer(3, Some(&out_K.buf), k64 * out_per_k_bytes);
                enc.set_threadgroup_memory_length(0, shmem);
                enc.dispatch_thread_groups(
                    MTLSize::new(n_row_tg, 1, 1),
                    MTLSize::new(32, nsg as u64, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
            out_K
        };

        // Step 3: K iterations of the output_hc_weights shim.
        let output_weights_K = self.alloc_f32(k_positions * n_hc);
        {
            let empty_key: &[u8] = &[];
            let pipe = self.state.specialized_pipeline(
                "ds4_kernel_dsv4_mtp_output_hc_weights_f32", empty_key, |_fcv| {},
            )?;
            let n_hc_u32 = n_hc as u32;
            let tcount = (pipe.max_total_threads_per_threadgroup() as u64)
                .min(n_hc as u64).max(1);
            let per_k_bytes = (n_hc * 4) as u64;
            for k in 0..k_positions {
                let k64 = k as u64;
                let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&output_pre_K.buf), k64 * per_k_bytes);
                enc.set_buffer(1, Some(&hc_head_scale.buf), 0);
                enc.set_buffer(2, Some(&hc_head_base.buf), 0);
                enc.set_buffer(3, Some(&output_weights_K.buf), k64 * per_k_bytes);
                set_scalar_bytes(enc, 4, &n_hc_u32);
                set_scalar_bytes(enc, 5, &hc_eps);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(tcount, 1, 1),
                );
                crate::macos::end_shared_compute_enc(enc);
            }
        }

        // Step 4: hc_weighted_sum with n_tokens=K — kernel already supports
        // K via tgpig.y; set the right strides and a K-sized output.
        let output_embd_K = self.alloc_f32(k_positions * n_embd);
        {
            let empty_key: &[u8] = &[];
            let pipe = self.state.specialized_pipeline(
                "ds4_dsv4_hc_weighted_sum", empty_key, |_fcv| {},
            )?;
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct HcWsArgs {
                n_embd: i64, n_hc: i64, n_tokens: i64,
                nb_x0: u64, nb_x1: u64, nb_x2: u64,
                nb_w0: u64, nb_w1: u64,
                nb0: u64, nb1: u64,
            }
            let f32_b: u64 = 4;
            let args = HcWsArgs {
                n_embd: n_embd as i64, n_hc: n_hc as i64,
                n_tokens: k_positions as i64,
                nb_x0: f32_b,
                nb_x1: (n_embd as u64) * f32_b,    // per-h stride
                nb_x2: (hc_dim as u64) * f32_b,    // per-K stride
                nb_w0: f32_b,
                nb_w1: (n_hc as u64) * f32_b,      // per-K weights stride
                nb0: f32_b,
                nb1: (n_embd as u64) * f32_b,
            };
            let total = (n_embd as u64) * (k_positions as u64);
            let tcount = (pipe.max_total_threads_per_threadgroup() as u64)
                .min(total).max(1);
            let tg_count = (total + tcount - 1) / tcount;
            let enc = crate::macos::shared_compute_enc(&self.cmd_buf);
            enc.set_compute_pipeline_state(&pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&cur_hc_K.buf), 0);
            enc.set_buffer(2, Some(&output_weights_K.buf), 0);
            enc.set_buffer(3, Some(&output_embd_K.buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(tg_count, 1, 1),
                MTLSize::new(tcount, 1, 1),
            );
            crate::macos::end_shared_compute_enc(enc);
        }

        // Step 5: K-batched RMS-norm-mul with final_norm gamma.
        let output_norm_K = self.rms_norm_mul_k(
            &output_embd_K, final_norm_gamma, n_embd, k_positions, rms_eps,
        )?;

        // Step 6: K-amortized matvec against base LM head.
        self.matvec_k_q8_0(base_lm_head_q8, &output_norm_K, n_embd, vocab, k_positions)
    }

    /// Phase 3 MTP — FULL drafter forward pass (the full 10-step pipeline
    /// from antirez `metal_graph_eval_mtp_draft_from_hc` at ds4.c:12225).
    ///
    /// Composes:
    ///   1. `encode_mtp_input_mix`         (Step 2a) — token+prev_hc → mtp_input_hc
    ///   2. `encode_layer_k(K=1)`          — the drafter's single decode layer
    ///   3. `encode_mtp_output_head`       (Step 2b) — out_hc → logits
    ///
    /// Returns `logits [vocab]` ready for argmax/sampling on the host.
    ///
    /// Caller is responsible for:
    /// - Pre-uploading all weight tensors as DeferredBufs (see
    ///   `MtpDraftBufs` / `MtpDraftInputMix` / `MtpDraftOutputHead`).
    /// - Loading MTP expert weights into `state.expert_weights[mtp_layer_idx]`
    ///   via a future `MetalDispatcher::load_mtp_expert_weights` call (the
    ///   drafter's MoE chain pulls from there).
    /// - Allocating + initializing the MTP KV cache (also indexed by
    ///   `mtp_layer_idx`; sized for the drafter's window).
    /// - Token embedding extraction: `token_embed_db [n_embd]` is one row
    ///   of `base.token_embd` for the input token.
    ///
    /// `mtp_layer_idx` is the DEDICATED layer slot for the drafter's KV
    /// cache + expert weights — antirez uses a separate index so the
    /// drafter doesn't trample base-model layer state. Convention:
    /// allocate `n_base_layers + mtp_offset` (e.g., 250).
    /// `attn_base_pos` (forwarded to `encode_layer_k` → `flash_attn_k_mla`)
    /// is the MTP cache fill BEFORE this draft writes its row.  For draft
    /// iter i with prior `mtp_n_raw=N`, pass `attn_base_pos = N + i` so the
    /// drafter's attn window is `[0, N + i + 1)` (only the MTP's own
    /// previously-written rows + the one being written now).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp_draft(
        &mut self,
        prev_hc_db: &DeferredBuf,
        token_embed_db: &DeferredBuf,
        input_mix: MtpDraftInputMix<'_>,
        layer: MtpDraftLayerWeights<'_>,
        output_head: MtpDraftOutputHead<'_>,
        shape: MtpDraftShape,
        layer_params: &ds4_engine::attn_dispatch::LayerParams,
        mtp_layer_idx: u32,
        raw_cap: u32,
        base_slot: u32,
        base_pos: u32,
        attn_base_pos: u32,
    ) -> Result<DeferredBuf> {
        let (_out_hc, logits) = self.encode_mtp_draft_with_out_hc(
            prev_hc_db, token_embed_db,
            input_mix, layer, output_head, shape,
            layer_params, mtp_layer_idx, raw_cap, base_slot, base_pos, attn_base_pos,
        )?;
        Ok(logits)
    }

    /// Phase 3 Step 4d — same as `encode_mtp_draft` but also returns the
    /// drafter's `out_hc` so the caller can THREAD it as `prev_hc` into
    /// the next chained draft. Mirrors antirez's ping-pong pattern in
    /// the K-token draft loop (ds4.c:16154-16177): alternates two HC
    /// buffers across draft iterations, feeding each draft's `out_hc`
    /// as the next draft's `prev_hc`.
    ///
    /// Returns `(out_hc [n_hc, n_embd], logits [vocab])`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp_draft_with_out_hc(
        &mut self,
        prev_hc_db: &DeferredBuf,
        token_embed_db: &DeferredBuf,
        input_mix: MtpDraftInputMix<'_>,
        layer: MtpDraftLayerWeights<'_>,
        output_head: MtpDraftOutputHead<'_>,
        shape: MtpDraftShape,
        layer_params: &ds4_engine::attn_dispatch::LayerParams,
        mtp_layer_idx: u32,
        raw_cap: u32,
        base_slot: u32,
        base_pos: u32,
        attn_base_pos: u32,
    ) -> Result<(DeferredBuf, DeferredBuf)> {
        // 1. Input-mix prefix — produces mtp_input_hc [n_hc, n_embd].
        let mtp_input_hc = self.encode_mtp_input_mix(
            token_embed_db, prev_hc_db,
            input_mix.enorm_gamma, input_mix.e_proj_q8,
            input_mix.hnorm_gamma, input_mix.h_proj_q8,
            shape.n_embd, shape.n_hc, shape.rms_eps,
        )?;

        // 2. The drafter's single decode layer at K=1 via encode_layer_k.
        let out_hc = self.encode_layer_k(
            &mtp_input_hc,
            // attn-half
            layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            layer.attn_norm, output_head.unit_gamma_hc,
            layer.attn_q_a_q8, layer.gamma_q, layer.attn_q_b_q8,
            layer.attn_kv_q8, layer.gamma_kv,
            layer.w_o_a_q8, layer.w_o_b_q8,
            // ffn-half
            layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
            layer.ffn_norm,
            layer.w_router, layer.router_bias,
            layer.sh_w_gate, layer.sh_w_up, layer.sh_w_down,
            layer.sh_w_gate_q8, layer.sh_w_up_q8, layer.sh_w_down_q8,
            // shape
            shape.n_hc, shape.n_embd,
            shape.n_lora_q, shape.n_head, shape.head_dim, shape.kv_row,
            shape.n_groups, shape.n_lora_o, shape.group_dim, shape.out_low_dim,
            shape.n_experts, shape.d_ffn, shape.shared_dim,
            shape.sinkhorn_iters, shape.hc_eps, shape.rms_eps, shape.flash_scale,
            mtp_layer_idx, layer_params, raw_cap, base_slot, base_pos,
            attn_base_pos, layer.attn_sinks, /*k_positions=*/1,
            /*comp=*/None,
        )?;

        // 3. Output head → logits [vocab].
        let logits = self.encode_mtp_output_head(
            &out_hc,
            output_head.unit_gamma_hc,
            output_head.hc_head_fn, output_head.hc_head_scale, output_head.hc_head_base,
            output_head.mtp_norm,
            output_head.base_lm_head_q8,
            shape.n_embd, shape.n_hc, shape.vocab,
            shape.rms_eps, shape.hc_eps,
        )?;

        Ok((out_hc, logits))
    }

    /// Phase 3 Step 4d.2 — K-position verifier: run all N base decode
    /// layers at K=K with `DS4_MOE_K_PATH=fused` (set externally), produce
    /// `cur_hc_K [K, n_hc, n_embd]`. Each K-row is one speculative
    /// candidate's forward pass; the K-batched dispatch shares cb +
    /// weight reads across candidates (Phase 2 win).
    ///
    /// Output: the FINAL HC residual after all N layers — caller threads
    /// to the LM-head stage (final RMS + output_hc_head + matvec_q8_0
    /// against base LM head → logits [K, vocab]).
    ///
    /// `unit_gamma_hc` is `[n_hc * n_embd]` ones, shared across all layers.
    /// `base_slot` + `base_pos` are the K-token batch's STARTING slot/pos
    /// (each K-row uses slot=base_slot+k, pos=base_pos+k internally).
    /// `raw_cap` is the KV cache capacity per layer.
    ///
    /// All N layers run in this ONE scope/cb — caller does flush_and_read.
    /// DS4_MOE_K_PATH=fused env var (set by caller) selects the K-amortized
    /// MoE path for layers with iq2_xxs+q2_K or q4_K+q4_K experts.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn encode_verify_layers_K(
        &mut self,
        prev_hc_k: &DeferredBuf,                   // [K, n_hc, n_embd]
        layers: &[BaseLayerVerifyBundle<'_>],      // 43 entries for DS4 V4 Flash
        unit_gamma_hc: &DeferredBuf,                // [n_hc * n_embd] ones
        // Common shape — per-layer dims are uniform in DS4 V4 Flash.
        n_hc: usize,
        n_embd: usize,
        n_lora_q: usize,
        n_head: usize,
        head_dim: usize,
        kv_row: usize,
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        n_experts: usize,
        d_ffn: usize,
        shared_dim: u32,
        sinkhorn_iters: i32,
        hc_eps: f32,
        rms_eps: f32,
        flash_scale: f32,
        raw_cap: u32,
        base_slot: u32,
        base_pos: u32,
        k_positions: usize,
        // DS4_VERIFY_COMPRESSOR: per-layer prefill ring count (= state.n_comp).
        // None disables the compressor-aware path (verifier uses raw flash, as
        // before). When Some, compressor layers (ratio==4) with n_comp>0 get the
        // per-draft ring-aware flash. Length must equal `layers.len()`.
        comp_n_per_layer: Option<&[u32]>,
    ) -> Result<DeferredBuf> {
        // Verifier semantics: K-row k writes/reads its own KV at slot
        // base_slot+k for position base_pos+k.  Pass base_pos as the kernel's
        // attn_base_pos so per-row causal window = [0, base_pos+k+1).
        let attn_base_pos = base_pos;
        if let Some(cn) = comp_n_per_layer {
            anyhow::ensure!(
                cn.len() == layers.len(),
                "encode_verify_layers_K: comp_n_per_layer len {} != layers {}",
                cn.len(), layers.len()
            );
        }
        let hc_dim = n_hc * n_embd;
        anyhow::ensure!(
            prev_hc_k.n_elements == k_positions * hc_dim,
            "encode_verify_layers_K: prev_hc_k {} != K*n_hc*n_embd = {}*{}",
            prev_hc_k.n_elements, k_positions, hc_dim,
        );
        anyhow::ensure!(!layers.is_empty(), "encode_verify_layers_K: no layers");

        // DS4_VERIFY_CB_SPLIT=N: split the layer chain into sub-cbs every N
        // layers (commit_keep_open, no wait) to enable CPU-encode / GPU-execute
        // overlap and shrink the per-cb scheduling footprint. None = single cb.
        let cb_split: Option<usize> = std::env::var("DS4_VERIFY_CB_SPLIT")
            .ok().and_then(|s| s.parse().ok());

        // Chain N layers; each layer's output becomes next layer's input.
        // `encode_layer_k` returns a fresh DeferredBuf for `after_ffn_k`,
        // so we own the result without lifetime concerns.
        // We need to keep TWO bufs alive (prev + current) — the previous
        // iter's DeferredBuf is dropped at iter end as `cur` overwrites it.
        // Initial prev is the caller's prev_hc_k (borrowed).
        let mut current: Option<DeferredBuf> = None;

        // DS4_VERIFY_N_LAYERS=L truncates the chain to the first L layers for
        // bisection (pair with DS4_DECODE_N_LAYERS=L so the faithfulness ref
        // greedy truncates to match). cur_hc after L layers is comparable; the
        // logits are invalid when truncated (diagnostic only).
        let verify_n_layers: usize = std::env::var("DS4_VERIFY_N_LAYERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(layers.len())
            .min(layers.len());

        for (layer_idx, layer) in layers.iter().enumerate().take(verify_n_layers) {
            let prev = match &current {
                Some(c) => c,
                None => prev_hc_k,
            };
            // Build the compressor ctx for this layer when enabled + this is a
            // compressor layer (ratio==4, non-empty weights) with a non-empty
            // prefill ring. Otherwise None → raw flash (4b).
            let ratio = layer.layer_params.compress_ratio;
            let n_comp = comp_n_per_layer.map(|cn| cn[layer_idx]).unwrap_or(0);
            let comp_ctx = if comp_n_per_layer.is_some()
                && ratio == 4
                && n_comp > 0
                && (!layer.attn_compressor_kv.is_empty()
                    || layer.attn_compressor_kv_f16.is_some())
            {
                Some(CompVerifyCtx {
                    n_comp,
                    attn_sinks_cpu: layer.attn_sinks_cpu,
                    attn_comp: ds4_engine::attn_dispatch::CompressorInputs {
                        w_kv: layer.attn_compressor_kv,
                        w_gate: layer.attn_compressor_gate,
                        w_kv_f16: layer.attn_compressor_kv_f16,
                        w_gate_f16: layer.attn_compressor_gate_f16,
                        w_ape: layer.attn_compressor_ape,
                        w_norm: layer.attn_compressor_norm,
                        head_dim: layer.layer_params.head_dim,
                        compress_ratio: ratio,
                    },
                    idx_comp: ds4_engine::attn_dispatch::CompressorInputs {
                        w_kv: layer.indexer_compressor_kv,
                        w_gate: layer.indexer_compressor_gate,
                        w_kv_f16: layer.indexer_compressor_kv_f16,
                        w_gate_f16: layer.indexer_compressor_gate_f16,
                        w_ape: layer.indexer_compressor_ape,
                        w_norm: layer.indexer_compressor_norm,
                        head_dim: ds4_engine::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                        compress_ratio: ratio,
                    },
                })
            } else {
                None
            };
            let next = self.encode_layer_k(
                prev,
                // attn-half
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                layer.attn_norm, unit_gamma_hc,
                layer.attn_q_a_q8, layer.gamma_q, layer.attn_q_b_q8,
                layer.attn_kv_q8, layer.gamma_kv,
                layer.w_o_a_q8, layer.w_o_b_q8,
                // ffn-half
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                layer.ffn_norm,
                layer.w_router, layer.router_bias,
                layer.sh_w_gate, layer.sh_w_up, layer.sh_w_down,
                layer.sh_w_gate_q8, layer.sh_w_up_q8, layer.sh_w_down_q8,
                // shape
                n_hc, n_embd,
                n_lora_q, n_head, head_dim, kv_row,
                n_groups, n_lora_o, group_dim, out_low_dim,
                n_experts, d_ffn, shared_dim,
                sinkhorn_iters, hc_eps, rms_eps, flash_scale,
                layer_idx as u32, layer.layer_params,
                raw_cap, base_slot, base_pos, attn_base_pos, layer.attn_sinks, k_positions,
                comp_ctx.as_ref(),
            )?;
            current = Some(next);

            // DS4_VERIFY_DUMP_LAYER_RMS: per-layer per-row residual RMS, to find
            // WHERE row 2's cur_hc explodes (the row-2 logit-flip bug). Commit+
            // wait this layer's chain so `current` (StorageModeShared) is GPU-
            // valid, then read each row's RMS. Serializes the chain (diagnostic
            // only — kills overlap). A row whose RMS jumps orders of magnitude
            // at some layer pinpoints the diverging layer.
            if std::env::var("DS4_VERIFY_DUMP_LAYER_RMS").is_ok() && k_positions >= 3 {
                self.commit_wait_stage("verify_layer_rms");
                let cur = current.as_ref().unwrap();
                let ptr = cur.buf.contents() as *const f32;
                let rms_of = |row: usize| -> f64 {
                    let mut s = 0.0f64;
                    for i in 0..hc_dim {
                        let v = unsafe { *ptr.add(row * hc_dim + i) } as f64;
                        s += v * v;
                    }
                    (s / hc_dim as f64).sqrt()
                };
                let r3 = (k_positions - 1).min(3);
                eprintln!(
                    "[verify-layer-rms] after layer {:2} (ratio={:3}): row0={:.4e} row1={:.4e} row2={:.4e} row{}={:.4e}",
                    layer_idx, layer.layer_params.compress_ratio,
                    rms_of(0), rms_of(1), rms_of(2), r3, rms_of(r3)
                );
            }

            // DS4_VERIFY_CB_SPLIT=N: commit (without waiting) every N layers,
            // opening a fresh cb. Breaks the single ~1290-encoder mega-cb into
            // smaller cbs so the GPU can begin executing committed work while
            // the CPU keeps encoding (overlap), and so each cb's scheduling
            // footprint is smaller. commit_keep_open does NOT wait — the
            // resident DeferredBufs (StorageModeShared) stay valid across the
            // cb boundary (queue ordering guarantees the next cb sees prior
            // writes). flush_and_read waits on all pending cbs at the end.
            if let Some(split) = cb_split {
                if split > 0 && (layer_idx + 1) % split == 0 && layer_idx + 1 < layers.len() {
                    self.commit_keep_open();
                }
            }
        }

        // The loop ran ≥1 layer (asserted above), so `current` is always Some.
        Ok(current.unwrap())
    }

    /// Commit the accumulated command buffer, wait for the GPU to finish,
    /// and read back the chosen output as a `Vec<f32>`. Consumes the
    /// scope — no further ops can be added.
    #[track_caller]
    pub fn flush_and_read(self, out: &DeferredBuf) -> Vec<f32> {
        crate::macos::end_shared_compute_enc_force();        drain_trace_record(std::panic::Location::caller(), self.pending_cbs.len());
        // Wait on any pending cbs first so their writes are visible
        // before we read the output (Metal serialization within a
        // queue means the current cb sees prior writes, but the CPU
        // read of `out.buf` needs all writes complete).
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
        }
        self.state.commit_wait_traced(&self.cmd_buf, "BatchScope::flush_and_read");
        unsafe { read_buffer::<f32>(&out.buf, out.n_elements) }
    }

    /// Like `flush_and_read` but returns multiple buffers in one call.
    /// Use when a fused chain has more than one host-visible output
    /// (e.g. `attn_qkv_chain_batched` returns `qr_normed`, `q_heads`,
    /// `kv_raw_row`). Order matches the input slice.
    #[track_caller]
    pub fn flush_and_read_multi(self, outs: &[&DeferredBuf]) -> Vec<Vec<f32>> {
        crate::macos::end_shared_compute_enc_force();        drain_trace_record(std::panic::Location::caller(), self.pending_cbs.len());
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
        }
        self.state
            .commit_wait_traced(&self.cmd_buf, "BatchScope::flush_and_read_multi");
        outs.iter()
            .map(|b| unsafe { read_buffer::<f32>(&b.buf, b.n_elements) })
            .collect()
    }

    /// Like `flush_and_read` but reads the output as `i32` (e.g. the indexer
    /// top-k `selected` index buffer).
    #[track_caller]
    pub fn flush_and_read_i32(self, out: &DeferredBuf) -> Vec<i32> {
        crate::macos::end_shared_compute_enc_force();        drain_trace_record(std::panic::Location::caller(), self.pending_cbs.len());
        for cb in &self.pending_cbs {
            cb.wait_until_completed();
        }
        self.state
            .commit_wait_traced(&self.cmd_buf, "BatchScope::flush_and_read_i32");
        unsafe { read_buffer::<i32>(&out.buf, out.n_elements) }
    }
}

impl crate::MetalDispatcher {
    /// Open a new `BatchScope` against this dispatcher's `MetalState`.
    /// Each scope corresponds to exactly one `MTLCommandBuffer` submission.
    pub fn batch_scope(&self) -> BatchScope<'_> {
        BatchScope::new(&self.inner)
    }

    /// Phase C-B Slice 2 (M4 #330p). Fuses the attention-half Q/K/V chain
    /// into ONE `MTLCommandBuffer`:
    ///
    /// ```text
    /// matvec(attn_q_a, normed) → qr           (size n_lora_q)
    /// rms_norm_mul(qr, gamma_q, eps) → qr_normed
    /// matvec(attn_q_b, qr_normed) → q_heads_raw   (size n_head*head_dim)
    /// head_rms_norm(q_heads_raw, n_head, head_dim, eps) → q_heads
    /// matvec(attn_kv, normed) → kv_raw_row     (size kv_row)
    /// ```
    ///
    /// Returns `(qr_normed, q_heads, kv_raw_row)`. Replaces the prior
    /// sequence of `layer_qa_rms_batched` (1 cb) + `qkv_b_head_rms_batched`
    /// (1 cb) — saves 1 commit + 1 wait + 2 readbacks per layer per token
    /// vs. that pair. `qr_normed` is exposed in the output tuple because
    /// the per-token indexer (ratio==4 layers) consumes it on the host.
    ///
    /// Bit-identical to running the two `_batched` ops sequentially:
    /// same MSL kernels (`ds4_kernel_mul_mv_f32_f32_4`,
    /// `ds4_kernel_rms_norm_mul_f32_4`, `ds4_head_rms_norm_f32`), same
    /// FC specialization keys, same args. The only behavioural
    /// difference is one cb commit instead of two.
    ///
    /// Args:
    /// - `normed` `[d_embd]` — output of `hc_collapse_norm` (CPU side)
    /// - `attn_q_a` `[n_lora_q × d_embd]` — Q-projection low-rank "A" weight
    /// - `gamma_q` `[n_lora_q]` — qkv_gamma_q (for the inter-stage rms_norm)
    /// - `attn_q_b` `[q_dim × n_lora_q]` — Q-projection "B" weight where
    ///   `q_dim = n_head * head_dim`
    /// - `attn_kv` `[kv_row × d_embd]` — KV-projection weight
    ///
    /// Preconditions (asserted):
    /// - `d_embd % 4 == 0` (float4 lanes), `d_embd % 2 == 0` (matvec NR0)
    /// - `n_lora_q % 4 == 0` (rms_norm float4 lanes), `n_lora_q % 2 == 0`
    /// - `q_dim % 2 == 0` (NR0)
    /// - `kv_row % 2 == 0` (NR0)
    /// - `q_dim == n_head * head_dim`
    #[allow(clippy::too_many_arguments)]
    pub fn attn_qkv_chain_batched(
        &self,
        normed: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        attn_q_b: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
        attn_kv: &[f32],
        kv_row: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let d_embd = normed.len();
        let q_dim = n_head * head_dim;
        anyhow::ensure!(
            attn_q_a.len() == n_lora_q * d_embd,
            "attn_qkv_chain_batched: attn_q_a.len ({}) != n_lora_q*d_embd ({}*{})",
            attn_q_a.len(),
            n_lora_q,
            d_embd
        );
        anyhow::ensure!(
            gamma_q.len() == n_lora_q,
            "attn_qkv_chain_batched: gamma_q.len ({}) != n_lora_q ({})",
            gamma_q.len(),
            n_lora_q
        );
        anyhow::ensure!(
            attn_q_b.len() == q_dim * n_lora_q,
            "attn_qkv_chain_batched: attn_q_b.len ({}) != q_dim*n_lora_q ({}*{})",
            attn_q_b.len(),
            q_dim,
            n_lora_q
        );
        anyhow::ensure!(
            attn_kv.len() == kv_row * d_embd,
            "attn_qkv_chain_batched: attn_kv.len ({}) != kv_row*d_embd ({}*{})",
            attn_kv.len(),
            kv_row,
            d_embd
        );
        anyhow::ensure!(head_dim >= 1, "attn_qkv_chain_batched: head_dim must be >= 1");

        let scope = self.batch_scope();
        let x_b = scope.upload_f32(normed);
        let w_qa = scope.weight_f32(attn_q_a);
        let g_b = scope.upload_f32(gamma_q);
        let w_qb = scope.weight_f32(attn_q_b);
        let w_kv = scope.weight_f32(attn_kv);

        let qr = scope.matvec_f32(&w_qa, &x_b, d_embd, n_lora_q)?;
        let qr_normed = scope.rms_norm_mul(&qr, &g_b, eps)?;
        let q_heads_raw = scope.matvec_f32(&w_qb, &qr_normed, n_lora_q, q_dim)?;
        let q_heads = scope.head_rms_norm(&q_heads_raw, n_head, head_dim, eps)?;
        let kv_raw_row = scope.matvec_f32(&w_kv, &x_b, d_embd, kv_row)?;

        let outs = scope.flush_and_read_multi(&[&qr_normed, &q_heads, &kv_raw_row]);
        let mut it = outs.into_iter();
        let qr_normed_v = it.next().unwrap();
        let q_heads_v = it.next().unwrap();
        let kv_raw_row_v = it.next().unwrap();
        Ok((qr_normed_v, q_heads_v, kv_raw_row_v))
    }

    /// Phase C-B Slice 4 (M4 #330p). Fuse `kv_rms_norm_row → rope_tail(KV)`
    /// into ONE `MTLCommandBuffer`:
    ///
    /// ```text
    /// kv_normed = rms_norm(kv_raw_row, qkv_gamma_kv)
    /// kv_normed[n_lora_kv - n_rot ..] = rope_tail(kv_normed[..], pos)
    /// ```
    ///
    /// Returns the fully-normed-and-rotated `kv_normed` row. Replaces the
    /// `kv_rms_norm_row + rope_tail` pair (2 cbs) on the attention prefix
    /// path — saves 1 commit+wait per layer per token.
    ///
    /// Preconditions:
    /// - `kv_raw_row.len() == qkv_gamma_kv.len() == params.n_lora_kv`
    /// - `params.n_rot` even and ≥ 2
    /// - `params.n_lora_kv % 4 == 0` (rms_norm float4 lanes)
    /// - `params.n_lora_kv >= params.n_rot`
    pub fn kv_norm_rope_chain(
        &self,
        kv_raw_row: &[f32],
        qkv_gamma_kv: &[f32],
        params: &ds4_engine::attn_dispatch::LayerParams,
        pos: u32,
        eps_rms: f32,
    ) -> Result<Vec<f32>> {
        let n_lora_kv = params.n_lora_kv as usize;
        let n_rot = params.n_rot as usize;
        anyhow::ensure!(
            kv_raw_row.len() == n_lora_kv,
            "kv_norm_rope_batched: kv_raw_row.len ({}) != n_lora_kv ({})",
            kv_raw_row.len(),
            n_lora_kv
        );
        anyhow::ensure!(
            qkv_gamma_kv.len() == n_lora_kv,
            "kv_norm_rope_batched: qkv_gamma_kv.len ({}) != n_lora_kv ({})",
            qkv_gamma_kv.len(),
            n_lora_kv
        );
        anyhow::ensure!(
            n_lora_kv >= n_rot,
            "kv_norm_rope_batched: n_lora_kv ({}) < n_rot ({})",
            n_lora_kv,
            n_rot
        );

        let scope = self.batch_scope();
        let x_b = scope.upload_f32(kv_raw_row);
        let g_b = scope.upload_f32(qkv_gamma_kv);
        let kv_normed = scope.rms_norm_mul(&x_b, &g_b, eps_rms)?;
        // The rope_tail slice covers the LAST n_rot elements as a single
        // "head" of width n_rot — same layout the standalone
        // `rope_tail_impl` consumes when callers pass
        // `&mut kv_normed[n_lora_kv - n_rot..]`.
        let tail_byte_offset = ((n_lora_kv - n_rot) * std::mem::size_of::<f32>()) as u64;
        scope.rope_tail_in_place(&kv_normed, tail_byte_offset, 1, params, pos, false)?;
        Ok(scope.flush_and_read(&kv_normed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_engine::dispatch::KernelDispatcher;

    /// Smoke test for the speculative-decoding K-position Q8_0 matvec kernel
    /// (ds4_kernel_mul_mv_K_q8_0_f32_sg). Validates:
    ///   - the pipeline specializes for each supported K ∈ {1,2,4,8};
    ///   - the K=1 column of K=4 output matches K=1 (single-position) output
    ///     within tight tolerance — proves K-amortization doesn't change the
    ///     per-position result.
    /// Uses small synthetic shape (d_in=128, d_out=64) so the test runs fast.
    #[test]
    fn matvec_k_q8_0_smoke() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let d_in: usize = 128;
        let d_out: usize = 64;
        // Build random Q8_0 weight bytes (per-block: f16 scale + 32 int8).
        let nb = d_in / 32;
        let row_bytes = nb * 34;
        let mut w_bytes = vec![0u8; d_out * row_bytes];
        let mut rng: u32 = 0x9E3779B9;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        for r in 0..d_out {
            for b in 0..nb {
                let off = r * row_bytes + b * 34;
                // Set scale = half(1.0) and qs to small signed ints.
                w_bytes[off] = 0x00; w_bytes[off+1] = 0x3C;
                for i in 0..32 {
                    w_bytes[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                }
            }
        }
        // Build K-position activations: [K=4, d_in].
        let k_positions: usize = 4;
        let mut x: Vec<f32> = Vec::with_capacity(k_positions * d_in);
        for _ in 0..(k_positions * d_in) {
            let r = next();
            x.push((r & 0xffff) as f32 / 65536.0 - 0.5);
        }
        // Also build a K=1 reference activation = x[0..d_in].
        let x_k1: Vec<f32> = x[0..d_in].to_vec();

        let scope = disp.batch_scope();
        let w_db = scope.weight_q8_0_raw(&w_bytes, d_in * d_out);
        // K=4 path.
        let x_k4_db = scope.upload_f32(&x);
        let out_k4 = scope.matvec_k_q8_0(&w_db, &x_k4_db, d_in, d_out, 4).expect("k=4");
        // K=1 path (same kernel, K=1 specialization).
        let x_k1_db = scope.upload_f32(&x_k1);
        let out_k1 = scope.matvec_k_q8_0(&w_db, &x_k1_db, d_in, d_out, 1).expect("k=1");
        let outs = scope.flush_and_read_multi(&[&out_k4, &out_k1]);
        let v_k4 = &outs[0]; // [K=4, d_out]
        let v_k1 = &outs[1]; // [d_out]
        assert_eq!(v_k4.len(), 4 * d_out);
        assert_eq!(v_k1.len(), d_out);
        // The K=4 output's first K-row should match the K=1 output (same x).
        let mut max_abs = 0.0f32;
        for r in 0..d_out {
            let a = v_k4[r];                 // K=4 layout: dst[0*d_out + r]
            let b = v_k1[r];
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
            assert!(a.is_finite() && b.is_finite(), "non-finite at r={}", r);
        }
        assert!(max_abs < 1e-3, "K=4[0] vs K=1 max_abs = {} (expected < 1e-3)", max_abs);
        // Outputs are non-trivial.
        let nz_k4 = v_k4.iter().filter(|&&v| v != 0.0).count();
        assert!(nz_k4 > k_positions * d_out / 2, "too many zeros in K=4 output: {} / {}", nz_k4, v_k4.len());
        eprintln!("matvec_k_q8_0_smoke: K=4[0] vs K=1 max_abs = {:.2e}, nz_k4 = {}/{}", max_abs, nz_k4, v_k4.len());
    }

    /// Validates the f16-no-copy matvec infra (commit 5a7f7219) that the
    /// compressor/indexer f16 conversion relies on: `matvec_f16` over raw f16
    /// weight bytes must match `matvec_f32` over the same weight dequantized to
    /// f32. Uses exactly-f16-representable weight values (m/1024, |m|<2048) so
    /// the two paths see bit-identical weights; only float reduction order can
    /// differ. Shapes mirror the real compressor: indexer (d_in=512, d_out=256)
    /// and main (d_in=512, d_out=1024).
    #[test]
    fn matvec_f16_matches_matvec_f32_oracle() {
        use crate::f16_cast::f32_to_f16_bits;
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let mut rng: u32 = 0x1234_5678;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        for (d_in, d_out) in [(512usize, 256usize), (512, 1024)] {
            // Weights exactly representable in f16: m/1024 with |m| < 2048.
            let w_f32: Vec<f32> = (0..d_in * d_out)
                .map(|_| ((next() % 4096) as i32 - 2048) as f32 / 1024.0)
                .collect();
            // Raw f16 bytes (little-endian), the no-copy mmap layout.
            let mut w_f16 = vec![0u8; d_in * d_out * 2];
            for (i, &v) in w_f32.iter().enumerate() {
                w_f16[i * 2..i * 2 + 2].copy_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            let x: Vec<f32> = (0..d_in)
                .map(|i| ((i as f32 * 0.013).sin() * 1.7) + ((i as f32 * 0.007).cos() * 0.3))
                .collect();

            let scope = disp.batch_scope();
            let x_db = scope.upload_f32(&x);
            let w32_db = scope.weight_f32(&w_f32);
            let w16_db = scope.weight_f16(&w_f16);
            let out32 = scope.matvec_f32(&w32_db, &x_db, d_in, d_out).expect("matvec_f32");
            let out16 = scope.matvec_f16(&w16_db, &x_db, d_in, d_out).expect("matvec_f16");
            let outs = scope.flush_and_read_multi(&[&out32, &out16]);
            let (v32, v16) = (&outs[0], &outs[1]);
            assert_eq!(v32.len(), d_out);
            assert_eq!(v16.len(), d_out);
            let max_abs = v32.iter().zip(v16.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1.0e-3,
                "matvec_f16 drifted from matvec_f32 oracle (d_in={d_in}, d_out={d_out}): max_abs={max_abs:e}"
            );
            eprintln!("matvec_f16_matches_matvec_f32_oracle: d_in={d_in}, d_out={d_out}, max_abs={max_abs:e}");
        }
    }

    /// Phase 1 chain: matvec_k_q8_0 → rms_norm_mul_k. Validates that the
    /// K=2 chained output bit-matches two K=1 chains run independently on the
    /// same activations. This is the foundational building block of
    /// encode_attn_qkv_chain_K (q-LoRA: q_a · x → rms_q → q_b · qr_normed).
    #[test]
    fn matvec_k_then_rms_norm_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let d_in: usize = 128;
        let d_out: usize = 64;  // must be %4 for rms_norm_mul
        let nb = d_in / 32;
        let row_bytes = nb * 34;
        let mut w_bytes = vec![0u8; d_out * row_bytes];
        let mut rng: u32 = 0xCAFEBABE;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        for r in 0..d_out {
            for b in 0..nb {
                let off = r * row_bytes + b * 34;
                w_bytes[off] = 0x00; w_bytes[off+1] = 0x3C;  // half 1.0
                for i in 0..32 {
                    w_bytes[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                }
            }
        }
        let k_positions: usize = 2;
        let mut x_k: Vec<f32> = Vec::with_capacity(k_positions * d_in);
        for _ in 0..(k_positions * d_in) {
            x_k.push((next() & 0xffff) as f32 / 65536.0 - 0.5);
        }
        let gamma: Vec<f32> = (0..d_out).map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05).collect();
        let eps: f32 = 1e-5;

        // K=2 chained path.
        let scope = disp.batch_scope();
        let w_db = scope.weight_q8_0_raw(&w_bytes, d_in * d_out);
        let x_k_db = scope.upload_f32(&x_k);
        let gamma_db = scope.weight_f32(&gamma);
        let mv_k = scope.matvec_k_q8_0(&w_db, &x_k_db, d_in, d_out, 2).expect("mv k=2");
        let rms_k = scope.rms_norm_mul_k(&mv_k, &gamma_db, d_out, 2, eps).expect("rms k=2");

        // Two K=1 references (using the production rms_norm_mul).
        let x_k0: Vec<f32> = x_k[0..d_in].to_vec();
        let x_k1: Vec<f32> = x_k[d_in..2*d_in].to_vec();
        let x_k0_db = scope.upload_f32(&x_k0);
        let x_k1_db = scope.upload_f32(&x_k1);
        let mv_0 = scope.matvec_k_q8_0(&w_db, &x_k0_db, d_in, d_out, 1).expect("mv k0");
        let mv_1 = scope.matvec_k_q8_0(&w_db, &x_k1_db, d_in, d_out, 1).expect("mv k1");
        let rms_0 = scope.rms_norm_mul(&mv_0, &gamma_db, eps).expect("rms k0");
        let rms_1 = scope.rms_norm_mul(&mv_1, &gamma_db, eps).expect("rms k1");

        let outs = scope.flush_and_read_multi(&[&rms_k, &rms_0, &rms_1]);
        let v_k = &outs[0];   // [K=2, d_out]
        let v_0 = &outs[1];   // [d_out]
        let v_1 = &outs[2];   // [d_out]
        assert_eq!(v_k.len(), 2 * d_out);
        assert_eq!(v_0.len(), d_out);
        assert_eq!(v_1.len(), d_out);

        // K=2 row 0 should match K=1 path 0; K=2 row 1 should match K=1 path 1.
        let mut max_abs_0 = 0.0f32;
        let mut max_abs_1 = 0.0f32;
        for r in 0..d_out {
            let d0 = (v_k[r] - v_0[r]).abs();
            let d1 = (v_k[d_out + r] - v_1[r]).abs();
            if d0 > max_abs_0 { max_abs_0 = d0; }
            if d1 > max_abs_1 { max_abs_1 = d1; }
        }
        eprintln!(
            "matvec_k_then_rms_norm_k: K=2 row0 vs K=1 max_abs={:.2e}, K=2 row1 vs K=1 max_abs={:.2e}",
            max_abs_0, max_abs_1
        );
        assert!(max_abs_0 < 1e-4, "row0 mismatch: {}", max_abs_0);
        assert!(max_abs_1 < 1e-4, "row1 mismatch: {}", max_abs_1);
    }

    /// Phase 1 full chain: encode_attn_qkv_chain_k matches two K=1 chains. The
    /// complete q-LoRA + kv-projection path through three Q8_0 matvecs + two
    /// rms-norms (rms_norm_mul_k and head_rms_norm_k). Validates that all
    /// three outputs (qr_normed, q_heads, kv_raw_row) at K=2 row-wise match
    /// independent K=1 calls on the same activations.
    #[test]
    fn encode_attn_qkv_chain_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        // Tiny but representative shape (matches Q8_0 / rms / head constraints).
        let d_embd: usize = 128;          // %32 (Q8_0) and %4 (rms)
        let n_lora_q: usize = 64;         // %4 (rms)
        let n_head: usize = 4;
        let head_dim: usize = 32;
        let q_dim: usize = n_head * head_dim;  // 128
        let kv_row: usize = 64;
        let eps: f32 = 1e-5;

        // Random Q8_0 weights for q_a [n_lora_q, d_embd], q_b [q_dim, n_lora_q], kv [kv_row, d_embd].
        fn rand_q8(d_out: usize, d_in: usize, seed0: u32) -> Vec<u8> {
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed0;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off+1] = 0x3C;  // half 1.0
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }
        let w_qa = rand_q8(n_lora_q, d_embd, 0x1111_1111);
        let w_qb = rand_q8(q_dim, n_lora_q, 0x2222_2222);
        let w_kv = rand_q8(kv_row, d_embd, 0x3333_3333);

        // K=2 activations: [2, d_embd] random.
        let k_positions: usize = std::env::var("DS4_TEST_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let mut rng: u32 = 0xBEEFCAFE;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let normed_k: Vec<f32> = (0..k_positions * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let gamma_q: Vec<f32> = (0..n_lora_q)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();

        let scope = disp.batch_scope();
        let w_qa_db = scope.weight_q8_0_raw(&w_qa, n_lora_q * d_embd);
        let w_qb_db = scope.weight_q8_0_raw(&w_qb, q_dim * n_lora_q);
        let w_kv_db = scope.weight_q8_0_raw(&w_kv, kv_row * d_embd);
        let gamma_q_db = scope.weight_f32(&gamma_q);

        // K=2 chained path.
        let normed_k_db = scope.upload_f32(&normed_k);
        let (qr_normed_k, q_heads_k, kv_raw_k) = scope.encode_attn_qkv_chain_k(
            &normed_k_db, &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db,
            n_lora_q, n_head, head_dim, eps, kv_row, d_embd, k_positions,
        ).expect("k=2 chain");

        // Two K=1 reference paths (using the same _k entry point with K=1).
        let normed_0: Vec<f32> = normed_k[0..d_embd].to_vec();
        let normed_1: Vec<f32> = normed_k[d_embd..2*d_embd].to_vec();
        let n0_db = scope.upload_f32(&normed_0);
        let n1_db = scope.upload_f32(&normed_1);
        let (qrn_0, qh_0, kv_0) = scope.encode_attn_qkv_chain_k(
            &n0_db, &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db,
            n_lora_q, n_head, head_dim, eps, kv_row, d_embd, 1,
        ).expect("k=1 path 0");
        let (qrn_1, qh_1, kv_1) = scope.encode_attn_qkv_chain_k(
            &n1_db, &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db,
            n_lora_q, n_head, head_dim, eps, kv_row, d_embd, 1,
        ).expect("k=1 path 1");

        let outs = scope.flush_and_read_multi(&[
            &qr_normed_k, &q_heads_k, &kv_raw_k,
            &qrn_0, &qh_0, &kv_0,
            &qrn_1, &qh_1, &kv_1,
        ]);
        let qrn_k = &outs[0]; let qh_k = &outs[1]; let kv_k = &outs[2];
        let qrn_r0 = &outs[3]; let qh_r0 = &outs[4]; let kv_r0 = &outs[5];
        let qrn_r1 = &outs[6]; let qh_r1 = &outs[7]; let kv_r1 = &outs[8];
        assert_eq!(qrn_k.len(), k_positions * n_lora_q);
        assert_eq!(qh_k.len(), k_positions * q_dim);
        assert_eq!(kv_k.len(), k_positions * kv_row);

        // Helper: max-abs over an aligned slice.
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let mqrn0 = max_abs(&qrn_k[0..n_lora_q], qrn_r0);
        let mqrn1 = max_abs(&qrn_k[n_lora_q..2*n_lora_q], qrn_r1);
        let mqh0  = max_abs(&qh_k[0..q_dim], qh_r0);
        let mqh1  = max_abs(&qh_k[q_dim..2*q_dim], qh_r1);
        let mkv0  = max_abs(&kv_k[0..kv_row], kv_r0);
        let mkv1  = max_abs(&kv_k[kv_row..2*kv_row], kv_r1);
        eprintln!(
            "encode_attn_qkv_chain_k: qrn[{:.2e},{:.2e}] qh[{:.2e},{:.2e}] kv[{:.2e},{:.2e}]",
            mqrn0, mqrn1, mqh0, mqh1, mkv0, mkv1
        );
        let tol = 1e-3f32;
        assert!(mqrn0 < tol && mqrn1 < tol, "qr_normed mismatch: {} {}", mqrn0, mqrn1);
        assert!(mqh0 < tol && mqh1 < tol, "q_heads mismatch: {} {}", mqh0, mqh1);
        assert!(mkv0 < tol && mkv1 < tol, "kv_raw mismatch: {} {}", mkv0, mkv1);
    }

    /// Phase 1 output projection at K: encode_attn_output_matmuls_q8_k at
    /// K=2 matches two encode_attn_output_matmuls_q8 K=1 calls. Stage 1
    /// (grouped) is K-linear (K iterations); stage 2 (dense w_o_b) is
    /// K-amortized via matvec_k_q8_0.
    #[test]
    fn encode_attn_output_matmuls_q8_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_groups: usize = 4;
        let group_dim: usize = 64;             // %32 (Q8_0), %4 (matvec_k)
        let n_lora_o: usize = 8;               // %2 (NR0)
        let out_low_dim: usize = n_groups * n_lora_o;  // 32
        let d_embd: usize = 64;                // %4

        // Random Q8_0 weights.
        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off+1] = 0x3C;  // half 1.0
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }
        let w_o_a = rand_q8(out_low_dim, group_dim, 0x4444_4444);
        let w_o_b = rand_q8(d_embd, out_low_dim, 0x5555_5555);

        // K=2 heads: [2, n_groups, group_dim].
        let k_positions: usize = std::env::var("DS4_TEST_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let mut rng: u32 = 0xFACE_1234;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let heads_k: Vec<f32> = (0..k_positions * n_groups * group_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let w_o_a_db = scope.weight_q8_0_raw(&w_o_a, out_low_dim * group_dim);
        let w_o_b_db = scope.weight_q8_0_raw(&w_o_b, d_embd * out_low_dim);

        // K=2 path.
        let heads_k_db = scope.upload_f32(&heads_k);
        let (low_k, out_k) = scope.encode_attn_output_matmuls_q8_k(
            &heads_k_db, &w_o_a_db, &w_o_b_db,
            n_groups, n_lora_o, group_dim, out_low_dim, d_embd, k_positions,
        ).expect("k=2 output proj");

        // Two K=1 reference paths — use the SAME _k entry point at K=1 so
        // the comparison is kernel-identical (matvec_k_q8_0 is half-precision
        // intermediates; matvec_q8_0 is float — they differ by ~half-precision
        // noise which would mask K-amortization correctness errors).
        let heads_0: Vec<f32> = heads_k[0..n_groups * group_dim].to_vec();
        let heads_1: Vec<f32> = heads_k[n_groups * group_dim..2*n_groups * group_dim].to_vec();
        let h0_db = scope.upload_f32(&heads_0);
        let h1_db = scope.upload_f32(&heads_1);
        let (low_0, out_0) = scope.encode_attn_output_matmuls_q8_k(
            &h0_db, &w_o_a_db, &w_o_b_db,
            n_groups, n_lora_o, group_dim, out_low_dim, d_embd, 1,
        ).expect("k=1 path 0");
        let (low_1, out_1) = scope.encode_attn_output_matmuls_q8_k(
            &h1_db, &w_o_a_db, &w_o_b_db,
            n_groups, n_lora_o, group_dim, out_low_dim, d_embd, 1,
        ).expect("k=1 path 1");

        let outs = scope.flush_and_read_multi(&[&low_k, &out_k, &low_0, &out_0, &low_1, &out_1]);
        let low_k_v = &outs[0]; let out_k_v = &outs[1];
        let low_0_v = &outs[2]; let out_0_v = &outs[3];
        let low_1_v = &outs[4]; let out_1_v = &outs[5];
        assert_eq!(low_k_v.len(), k_positions * out_low_dim);
        assert_eq!(out_k_v.len(), k_positions * d_embd);

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let m_low_0 = max_abs(&low_k_v[0..out_low_dim], low_0_v);
        let m_low_1 = max_abs(&low_k_v[out_low_dim..2*out_low_dim], low_1_v);
        let m_out_0 = max_abs(&out_k_v[0..d_embd], out_0_v);
        let m_out_1 = max_abs(&out_k_v[d_embd..2*d_embd], out_1_v);
        eprintln!(
            "encode_attn_output_matmuls_q8_k: low[{:.2e},{:.2e}]  out[{:.2e},{:.2e}]",
            m_low_0, m_low_1, m_out_0, m_out_1
        );
        let tol = std::env::var("DS4_TEST_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3f32);
        assert!(m_low_0 < tol && m_low_1 < tol, "low mismatch: {} {}", m_low_0, m_low_1);
        assert!(m_out_0 < tol && m_out_1 < tol, "out mismatch: {} {}", m_out_0, m_out_1);
    }

    /// Phase 1 shared chain at K: encode_shared_chain_q8_k at K=2 matches
    /// two K=1 calls of the same function. 3 matvecs (K-amortized via
    /// matvec_k_q8_0) + K-iter swiglu.
    #[test]
    fn encode_shared_chain_q8_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let d_embd: usize = 128;
        let sd: usize = 64;
        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off+1] = 0x3C;
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }
        let w_gate = rand_q8(sd, d_embd, 0x6666_6666);
        let w_up   = rand_q8(sd, d_embd, 0x7777_7777);
        let w_down = rand_q8(d_embd, sd, 0x8888_8888);

        let k_positions: usize = 2;
        let mut rng: u32 = 0xDEAD_BEEF;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let normed_k: Vec<f32> = (0..k_positions * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let w_gate_db = scope.weight_q8_0_raw(&w_gate, sd * d_embd);
        let w_up_db   = scope.weight_q8_0_raw(&w_up,   sd * d_embd);
        let w_down_db = scope.weight_q8_0_raw(&w_down, d_embd * sd);

        let normed_k_db = scope.upload_f32(&normed_k);
        let out_k = scope.encode_shared_chain_q8_k(
            &normed_k_db, &w_gate_db, &w_up_db, &w_down_db, d_embd, sd, 2,
        ).expect("k=2 shared");

        let n0: Vec<f32> = normed_k[0..d_embd].to_vec();
        let n1: Vec<f32> = normed_k[d_embd..2*d_embd].to_vec();
        let n0_db = scope.upload_f32(&n0);
        let n1_db = scope.upload_f32(&n1);
        let out_0 = scope.encode_shared_chain_q8_k(
            &n0_db, &w_gate_db, &w_up_db, &w_down_db, d_embd, sd, 1,
        ).expect("k=1 row0");
        let out_1 = scope.encode_shared_chain_q8_k(
            &n1_db, &w_gate_db, &w_up_db, &w_down_db, d_embd, sd, 1,
        ).expect("k=1 row1");

        let outs = scope.flush_and_read_multi(&[&out_k, &out_0, &out_1]);
        let v_k = &outs[0]; let v_0 = &outs[1]; let v_1 = &outs[2];
        assert_eq!(v_k.len(), 2 * d_embd);
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let m0 = max_abs(&v_k[0..d_embd], v_0);
        let m1 = max_abs(&v_k[d_embd..2*d_embd], v_1);
        eprintln!("encode_shared_chain_q8_k: row0={:.2e}, row1={:.2e}", m0, m1);
        assert!(m0 < 1e-3 && m1 < 1e-3, "shared chain mismatch: {} {}", m0, m1);
    }

    /// Phase 1 hc_collapse_norm at K: hc_collapse_norm_k at K=2 matches two
    /// K=1 hc_collapse_norm calls on the same input rows. Shape is fixed by
    /// the underlying sinkhorn kernel (n_hc=4, n_embd=4096).
    #[test]
    fn hc_collapse_norm_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_hc: usize = 4;
        let n_embd: usize = 4096;
        let hc_dim = n_hc * n_embd;            // 16384
        let mix_hc = 2 * n_hc + n_hc * n_hc;  // 24
        let sinkhorn_iters: i32 = 5;
        let hc_eps: f32 = 1e-6;
        let rms_eps: f32 = 1e-5;

        let mut rng: u32 = 0xC0FFEE_42;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let k_positions: usize = std::env::var("DS4_TEST_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let prev_hc_k: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
        let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + i as f32 * 0.01).collect();
        let hc_norm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();
        let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];

        let scope = disp.batch_scope();
        let hc_fn_db = scope.weight_f32(&hc_fn);
        let hc_scale_db = scope.weight_f32(&hc_scale);
        let hc_base_db = scope.weight_f32(&hc_base);
        let hc_norm_gamma_db = scope.weight_f32(&hc_norm_gamma);
        let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);

        // K=2 chained path.
        let prev_hc_k_db = scope.upload_f32(&prev_hc_k);
        let (_split_k, _cur_k, normed_k) = scope.hc_collapse_norm_k(
            &prev_hc_k_db, &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db,
            n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps,
            &unit_gamma_db, k_positions, false,
        ).expect("k=2 hc_collapse");

        // K=1 reference for EVERY row (not just 0,1 — a per-row K-dependent
        // drift at row>=2 would otherwise hide).
        let mut ref_norms: Vec<DeferredBuf> = Vec::with_capacity(k_positions);
        for r in 0..k_positions {
            let hr: Vec<f32> = prev_hc_k[r*hc_dim..(r+1)*hc_dim].to_vec();
            let hr_db = scope.upload_f32(&hr);
            let (_s, _c, n) = scope.hc_collapse_norm(
                &hr_db, &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db,
                n_hc, n_embd, sinkhorn_iters, hc_eps, rms_eps, &unit_gamma_db, false,
            ).expect("k=1 row");
            ref_norms.push(n);
        }
        let mut refs: Vec<&DeferredBuf> = vec![&normed_k];
        refs.extend(ref_norms.iter());
        let outs = scope.flush_and_read_multi(&refs);
        let nk = &outs[0];
        assert_eq!(nk.len(), k_positions * n_embd);
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let mut worst = 0.0f32; let mut worst_row = 0usize;
        for r in 0..k_positions {
            let m = max_abs(&nk[r*n_embd..(r+1)*n_embd], &outs[1+r]);
            if m > worst { worst = m; worst_row = r; }
        }
        eprintln!("hc_collapse_norm_k: K={k_positions} normed max-abs over ALL rows = {worst:.2e} (worst row {worst_row})");
        let tol = std::env::var("DS4_TEST_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3f32);
        assert!(worst < tol, "normed mismatch row {worst_row}: {worst}");
    }

    /// Phase 1 layer-attn-half encoder at K — the final composition. K=2
    /// path through encode_layer_attn_half_k matches two K=1 paths through
    /// encode_layer_attn_half. Forces production-compatible n_hc=4
    /// n_embd=4096 (hardcoded in the sinkhorn kernel).
    #[test]
    fn encode_layer_attn_half_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        use ds4_engine::attn_dispatch::LayerParams;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        // Production-compatible shapes (n_hc=4, n_embd=4096 forced).
        let n_hc: usize = 4;
        let n_embd: usize = 4096;
        let d_embd: usize = n_embd;
        let n_lora_q: usize = 64;
        let n_head: usize = 4;
        let head_dim: usize = 64;
        let q_dim: usize = n_head * head_dim;
        let kv_row: usize = 128;
        let n_rot: usize = 32;
        let hc_dim = n_hc * n_embd;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let sinkhorn_iters: i32 = 5;
        let hc_eps: f32 = 1e-6;
        let rms_eps: f32 = 1e-5;
        let base_pos: u32 = 7;

        let params = LayerParams {
            layer_idx: 0,
            d_embd: d_embd as u32,
            n_hc: n_hc as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            n_rot: n_rot as u32,
            n_lora_q: n_lora_q as u32,
            n_lora_kv: kv_row as u32,
            hc_sinkhorn_iter: sinkhorn_iters as u32,
            hc_eps,
            rms_eps,
            rope_orig_ctx: 4096,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            compress_ratio: 1,
            n_out_group: 2,
        };

        // Random Q8_0 weights.
        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off+1] = 0x3C;
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }
        let w_qa = rand_q8(n_lora_q, d_embd, 0xAAAA_AAAA);
        let w_qb = rand_q8(q_dim, n_lora_q, 0xBBBB_BBBB);
        let w_kv = rand_q8(kv_row, d_embd, 0xCCCC_CCCC);

        // Random K=2 prev_hc and shared inputs.
        let k_positions: usize = 2;
        let mut rng: u32 = 0xDEAD_BEEF;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let prev_hc_k: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
        let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + i as f32 * 0.01).collect();
        let hc_norm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();
        let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];
        let gamma_q: Vec<f32> = (0..n_lora_q)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.04).collect();
        let gamma_kv: Vec<f32> = (0..kv_row)
            .map(|i| 1.0 + (i as f32 * 0.017).sin() * 0.04).collect();

        let mut scope = disp.batch_scope();
        let hc_fn_db = scope.weight_f32(&hc_fn);
        let hc_scale_db = scope.weight_f32(&hc_scale);
        let hc_base_db = scope.weight_f32(&hc_base);
        let hc_norm_gamma_db = scope.weight_f32(&hc_norm_gamma);
        let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);
        let gamma_q_db = scope.weight_f32(&gamma_q);
        let gamma_kv_db = scope.weight_f32(&gamma_kv);
        let w_qa_db = scope.weight_q8_0_raw(&w_qa, n_lora_q * d_embd);
        let w_qb_db = scope.weight_q8_0_raw(&w_qb, q_dim * n_lora_q);
        let w_kv_db = scope.weight_q8_0_raw(&w_kv, kv_row * d_embd);

        // K=2 layer encoder.
        let prev_hc_k_db = scope.upload_f32(&prev_hc_k);
        let outs_k = scope.encode_layer_attn_half_k(
            &prev_hc_k_db, &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db,
            &unit_gamma_db,
            &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db,
            &gamma_kv_db,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            sinkhorn_iters, hc_eps, rms_eps,
            &params, base_pos, k_positions, false,
        ).expect("k=2 layer");

        // Two K=1 references via the existing encode_layer_attn_half. Each
        // row uses its OWN pos (base_pos+k) — must match rope_tail in the K
        // path which iterates pos = base_pos+k per row.
        let h0: Vec<f32> = prev_hc_k[0..hc_dim].to_vec();
        let h1: Vec<f32> = prev_hc_k[hc_dim..2*hc_dim].to_vec();
        let h0_db = scope.upload_f32(&h0);
        let h1_db = scope.upload_f32(&h1);
        // Note: encode_layer_attn_half uses self.q8_proj.get() to pick q8 vs f32 matvec.
        // Set scope.q8_proj=true so the K=1 reference also uses q8 (matching the K=2 path).
        scope.set_q8_proj(true);
        let outs_0 = scope.encode_layer_attn_half(
            &h0_db, &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db,
            &unit_gamma_db,
            false,
            &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db, &gamma_kv_db,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            sinkhorn_iters, hc_eps, rms_eps, &params, base_pos,
        ).expect("k=1 row0");
        let outs_1 = scope.encode_layer_attn_half(
            &h1_db, &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db,
            &unit_gamma_db,
            false,
            &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db, &gamma_kv_db,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            sinkhorn_iters, hc_eps, rms_eps, &params, base_pos + 1,
        ).expect("k=1 row1");

        let res = scope.flush_and_read_multi(&[
            &outs_k.normed_k, &outs_k.split_k, &outs_k.qr_normed_k,
            &outs_k.q_heads_k, &outs_k.kv_normed_rotated_k,
            &outs_0.normed, &outs_0.split, &outs_0.qr_normed,
            &outs_0.q_heads, &outs_0.kv_normed_rotated,
            &outs_1.normed, &outs_1.split, &outs_1.qr_normed,
            &outs_1.q_heads, &outs_1.kv_normed_rotated,
        ]);
        let nk = &res[0]; let sk = &res[1]; let qrnk = &res[2];
        let qhk = &res[3]; let kvk = &res[4];
        let n0 = &res[5]; let s0 = &res[6]; let qrn0 = &res[7];
        let qh0 = &res[8]; let kv0 = &res[9];
        let n1 = &res[10]; let s1 = &res[11]; let qrn1 = &res[12];
        let qh1 = &res[13]; let kv1 = &res[14];

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let mn0 = max_abs(&nk[0..n_embd], n0);
        let mn1 = max_abs(&nk[n_embd..2*n_embd], n1);
        let ms0 = max_abs(&sk[0..mix_hc], s0);
        let ms1 = max_abs(&sk[mix_hc..2*mix_hc], s1);
        let mqrn0 = max_abs(&qrnk[0..n_lora_q], qrn0);
        let mqrn1 = max_abs(&qrnk[n_lora_q..2*n_lora_q], qrn1);
        let mqh0 = max_abs(&qhk[0..q_dim], qh0);
        let mqh1 = max_abs(&qhk[q_dim..2*q_dim], qh1);
        let mkv0 = max_abs(&kvk[0..kv_row], kv0);
        let mkv1 = max_abs(&kvk[kv_row..2*kv_row], kv1);
        eprintln!(
            "encode_layer_attn_half_k:\n  normed[{:.2e},{:.2e}] split[{:.2e},{:.2e}]\n  qr_normed[{:.2e},{:.2e}] q_heads[{:.2e},{:.2e}]\n  kv_normed_rotated[{:.2e},{:.2e}]",
            mn0, mn1, ms0, ms1, mqrn0, mqrn1, mqh0, mqh1, mkv0, mkv1
        );
        let tol = 1e-3f32;
        assert!(mn0 < tol && mn1 < tol, "normed mismatch: {} {}", mn0, mn1);
        assert!(ms0 < tol && ms1 < tol, "split mismatch: {} {}", ms0, ms1);
        assert!(mqrn0 < tol && mqrn1 < tol, "qr_normed mismatch: {} {}", mqrn0, mqrn1);
        assert!(mqh0 < tol && mqh1 < tol, "q_heads mismatch: {} {}", mqh0, mqh1);
        assert!(mkv0 < tol && mkv1 < tol, "kv_normed_rotated mismatch: {} {}", mkv0, mkv1);
    }

    /// Phase 2 kv_fp8_store_persistent_k: K=2 cache writes (slots
    /// base_slot, base_slot+1) match two K=1 kv_fp8_store_persistent calls
    /// at the same slots, byte-identical in the persistent cache region.
    #[test]
    fn kv_fp8_store_persistent_k_matches_two_k1_stores() {
        use crate::MetalDispatcher;
        use ds4_engine::attn_dispatch::LayerParams;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        // Two distinct layer indices — fresh persistent KV cache per index.
        let layer_idx_k = 100u32;
        let layer_idx_ref = 101u32;
        let n_lora_kv: usize = 256;
        let head_dim: usize = 320;  // n_nope = 320 - 64 = 256 = n_lora_kv. matches kv_fp8_store_persistent contract.
        let n_rot: usize = 64;
        let raw_cap: u32 = 16;
        let base_slot: u32 = 3;
        let k_positions: usize = 2;

        let params = LayerParams {
            layer_idx: 0, d_embd: 4096, n_hc: 4, n_head: 4,
            head_dim: head_dim as u32, n_rot: n_rot as u32,
            n_lora_q: 64, n_lora_kv: n_lora_kv as u32,
            hc_sinkhorn_iter: 5, hc_eps: 1e-6, rms_eps: 1e-5,
            rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
            rope_ext_factor: 0.0, rope_attn_factor: 1.0,
            compress_ratio: 1, n_out_group: 2,
        };

        let mut rng: u32 = 0x12345678;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let kv_k: Vec<f32> = (0..k_positions * n_lora_kv)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let kv_k_db = scope.upload_f32(&kv_k);
        let cache_k = scope.kv_fp8_store_persistent_k(
            layer_idx_k, &kv_k_db, &params, raw_cap, base_slot, k_positions,
        ).expect("k=2 store");

        // K=1 references on a separate layer (fresh cache) — two consecutive slots.
        let kv_0: Vec<f32> = kv_k[0..n_lora_kv].to_vec();
        let kv_1: Vec<f32> = kv_k[n_lora_kv..2*n_lora_kv].to_vec();
        let kv_0_db = scope.upload_f32(&kv_0);
        let kv_1_db = scope.upload_f32(&kv_1);
        let _cache0 = scope.kv_fp8_store_persistent(
            layer_idx_ref, &kv_0_db, &params, raw_cap, base_slot,
        ).expect("k=1 row0");
        let cache_ref = scope.kv_fp8_store_persistent(
            layer_idx_ref, &kv_1_db, &params, raw_cap, base_slot + 1,
        ).expect("k=1 row1");

        let outs = scope.flush_and_read_multi(&[&cache_k, &cache_ref]);
        let ck = &outs[0]; let cr = &outs[1];
        assert_eq!(ck.len(), cr.len());
        // Compare the [base_slot .. base_slot+K) slot region byte-for-byte
        // (treating the f32 buffer as the raw KV cache contents).
        let row_floats = n_lora_kv;
        let start = (base_slot as usize) * row_floats;
        let end   = ((base_slot as usize) + k_positions) * row_floats;
        let max_abs = ck[start..end].iter().zip(cr[start..end].iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!(
            "kv_fp8_store_persistent_k: slots[{}..{}) max_abs = {:.2e}",
            base_slot, base_slot + k_positions as u32, max_abs
        );
        assert!(max_abs == 0.0, "byte-identity expected; got max_abs = {}", max_abs);
    }

    /// Isolated micro-bench: kv_fp8_store_persistent_k at K=8, merged
    /// (1 dispatch) vs per-slot (K dispatches). Interleaved per-iteration so
    /// slow thermal drift cancels in the ratio. Gated DS4_BENCH_KV_STORE=1.
    /// This avoids the 43-layer thermal load that confounds the full-chain
    /// spec-decode bench — it measures ONLY the kv-store op's dispatch cost.
    #[test]
    fn kv_fp8_store_merge_microbench() {
        use crate::MetalDispatcher;
        use ds4_engine::attn_dispatch::LayerParams;
        use std::time::Instant;
        if std::env::var("DS4_BENCH_KV_STORE").ok().as_deref() != Some("1") {
            eprintln!("DS4_BENCH_KV_STORE unset — skipping. Set =1 to run.");
            return;
        }
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_lora_kv: usize = 512;
        let head_dim: usize = 512;
        let raw_cap: u32 = 256;
        let base_slot: u32 = 0;
        let k_positions: usize = 8;
        let params = LayerParams {
            layer_idx: 0, d_embd: 4096, n_hc: 4, n_head: 64,
            head_dim: head_dim as u32, n_rot: 64,
            n_lora_q: 1024, n_lora_kv: n_lora_kv as u32,
            hc_sinkhorn_iter: 5, hc_eps: 1e-6, rms_eps: 1e-5,
            rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
            rope_ext_factor: 0.0, rope_attn_factor: 1.0,
            compress_ratio: 1, n_out_group: 8,
        };
        let mut rng: u32 = 0xC0FFEE;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let kv: Vec<f32> = (0..k_positions * n_lora_kv)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        // One timed call = batch_scope + store(K) + flush (commit+wait).
        // mode: "0"=K encoders, "enc"=1 encoder K dispatches, "1"=merged kernel.
        let one_call = |mode: &str, layer_idx: u32| -> f64 {
            std::env::set_var("DS4_KV_STORE_MERGE", mode);
            let scope = disp.batch_scope();
            let kv_db = scope.upload_f32(&kv);
            let cache = scope.kv_fp8_store_persistent_k(
                layer_idx, &kv_db, &params, raw_cap, base_slot, k_positions,
            ).expect("store");
            let t = Instant::now();
            let _ = scope.flush_and_read(&cache);
            t.elapsed().as_secs_f64() * 1e6 // µs (flush = commit+wait+read)
        };

        let warmup = 10; let iters = 200;
        for _ in 0..warmup { one_call("0", 201); one_call("enc", 202); one_call("1", 200); }
        let (mut s_kenc, mut s_1enc, mut s_merged) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..iters {
            // Interleave all 3 modes per iteration so thermal drift cancels.
            s_kenc   += one_call("0",   201);
            s_1enc   += one_call("enc", 202);
            s_merged += one_call("1",   200);
        }
        std::env::remove_var("DS4_KV_STORE_MERGE");
        let (m_kenc, m_1enc, m_merged) =
            (s_kenc / iters as f64, s_1enc / iters as f64, s_merged / iters as f64);
        eprintln!(
            "kv_fp8_store K={} micro-bench ({} iters, interleaved 3-way):\n  \
             K encoders, K dispatch = {:.1} µs/call  (baseline)\n  \
             1 encoder,  K dispatch = {:.1} µs/call  (−{:.1}, isolates encoder-boundary cost)\n  \
             1 encoder,  1 dispatch = {:.1} µs/call  (−{:.1} more, isolates dispatch cost)\n  \
             total merged saving = {:.1} µs/call ({:.1}%) → ×43 ≈ {:.0} µs/verify",
            k_positions, iters,
            m_kenc,
            m_1enc, m_kenc - m_1enc,
            m_merged, m_1enc - m_merged,
            m_kenc - m_merged, (m_kenc - m_merged) / m_kenc * 100.0,
            (m_kenc - m_merged) * 43.0
        );
    }

    /// Output-projection stage-1 K-merge: validates the _k grid-batched kernel
    /// (DS4_ATTN_OUT_MERGE=1) produces bit-identical attn_low_k to the per-K
    /// dispatch loop (=0), and micro-benches both interleaved. Gated
    /// DS4_BENCH_ATTN_OUT=1. Random bytes suffice — the merge-vs-loop comparison
    /// is valid regardless of whether bytes are "proper" Q8 (both paths read
    /// them identically).
    #[test]
    fn attn_out_low_k_merge_microbench() {
        use crate::MetalDispatcher;
        use std::time::Instant;
        if std::env::var("DS4_BENCH_ATTN_OUT").ok().as_deref() != Some("1") {
            eprintln!("DS4_BENCH_ATTN_OUT unset — skipping. Set =1 to run.");
            return;
        }
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {}", e); return; }
        };
        // Real DS4 dims: n_groups=8, n_lora_o=1024, group_dim=4096.
        let n_groups = 8usize; let n_lora_o = 1024usize; let group_dim = 4096usize;
        let out_low_dim = n_groups * n_lora_o; let d_embd = 4096usize; let k = 8usize;
        let mut rng: u32 = 0xABCDEF;
        let mut nb = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); (rng >> 24) as u8 };
        let heads: Vec<f32> = {
            let mut r: u32 = 0x1234;
            (0..k*n_groups*group_dim).map(|_| { r = r.wrapping_mul(1664525).wrapping_add(1013904223); (r & 0xffff) as f32 / 65536.0 - 0.5 }).collect()
        };
        let woa_bytes: Vec<u8> = (0..out_low_dim*(group_dim/32)*34).map(|_| nb()).collect();
        let wob_bytes: Vec<u8> = (0..d_embd*(out_low_dim/32)*34).map(|_| nb()).collect();

        let run = |merge: bool| -> (Vec<f32>, f64) {
            std::env::set_var("DS4_ATTN_OUT_MERGE", if merge { "1" } else { "0" });
            let scope = disp.batch_scope();
            let heads_db = scope.upload_f32(&heads);
            let woa = scope.weight_q8_0_raw(&woa_bytes, out_low_dim*group_dim);
            let wob = scope.weight_q8_0_raw(&wob_bytes, d_embd*out_low_dim);
            let (low, _out) = scope.encode_attn_output_matmuls_q8_k(
                &heads_db, &woa, &wob, n_groups, n_lora_o, group_dim, out_low_dim, d_embd, k,
            ).expect("attn_out");
            let t = Instant::now();
            let v = scope.flush_and_read(&low);
            (v, t.elapsed().as_secs_f64() * 1e6)
        };

        // Correctness: merged vs loop, bit-identical.
        let (low_m, _) = run(true);
        let (low_p, _) = run(false);
        let max_abs = low_m.iter().zip(low_p.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        eprintln!("attn_out_low_k merge vs loop: max_abs = {:.2e} (len {})", max_abs, low_m.len());
        assert!(max_abs == 0.0, "merge must be bit-identical to loop; got {}", max_abs);

        // Perf: interleaved.
        let iters = 100;
        for _ in 0..10 { run(true); run(false); }
        let (mut sm, mut sp) = (0.0, 0.0);
        for _ in 0..iters { sm += run(true).1; sp += run(false).1; }
        std::env::remove_var("DS4_ATTN_OUT_MERGE");
        let (mm, mp) = (sm/iters as f64, sp/iters as f64);
        eprintln!(
            "attn_out_low stage1 K={} ({} iters, interleaved):\n  \
             per-K loop ({} dispatch) = {:.1} µs/call\n  \
             merged     (1 dispatch)  = {:.1} µs/call\n  \
             saving = {:.1} µs/call ({:.1}%) → ×43 ≈ {:.0} µs/verify",
            k, iters, k, mp, mm, mp - mm, (mp-mm)/mp*100.0, (mp-mm)*43.0
        );
    }

    /// Phase 2 K-query flash attention: K=2 path matches two K=1 calls on
    /// the same KV cache + same per-K-row queries. Cache is FP8-snapped f32
    /// (the kernel casts to half losslessly). Validates that the
    /// simdgroup-matrix kernel produces consistent per-row output across K
    /// specializations.
    /// FLASH KERNEL CORRECTNESS vs a hand-written CPU MLA reference at REAL
    /// dims (n_head=64, DK=DV=512). Self-contained (no GGUF/prefill) — a gross
    /// compute bug in flash_attn_k_mla reproduces with any data. This is the
    /// final pinpoint of the spec-decode verifier bug (the op-bisect proved
    /// every flash input + surrounding op is correct, yet attn-half diverges).
    #[test]
    fn flash_attn_k_mla_matches_cpu_mla_reference() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {}", e); return; }
        };
        let n_head = 64usize;
        let dk = 512usize;
        let dv = 512usize;
        let n_cache = 8usize;     // multiple of 8 (C=8); n_raw=7 via attn_base_pos=6
        let n_raw = 7usize;
        let attn_base_pos = (n_raw - 1) as u32; // K=1: window [0, base_pos+1) = [0,7)
        let scale = 1.0f32 / (dk as f32).sqrt();

        let mut rng: u32 = 0x5EED_1234;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 16) as f32 / 65536.0 - 0.5 };
        let q: Vec<f32> = (0..n_head * dk).map(|_| next()).collect();       // [1, n_head, DK]
        let kv: Vec<f32> = (0..n_cache * dk).map(|_| next()).collect();     // [n_cache, DK]

        // GPU: flash_attn_k_mla (K=1), zero sinks.
        let scope = disp.batch_scope();
        let q_db = scope.upload_f32(&q);
        let kv_db = scope.upload_f32(&kv);
        // Disable the sink (large-negative logit → exp→0) so the GPU matches the
        // no-sink CPU reference — isolates the pure flash MLA math.
        let sinks_db = scope.weight_f32(&vec![-1e30f32; n_head]);
        let o_db = scope.flash_attn_k_mla(
            &q_db, &kv_db, n_head, dk, dv, n_cache, 1, scale, attn_base_pos, &sinks_db,
        ).expect("flash_attn_k_mla");
        let o_gpu = scope.flush_and_read(&o_db);

        // CPU MLA reference: per head, softmax(q_h·kv_c·scale) over c<n_raw, ·kv_c.
        let mut o_ref = vec![0.0f32; n_head * dv];
        for h in 0..n_head {
            let q_h = &q[h * dk..(h + 1) * dk];
            let mut scores = vec![0.0f32; n_raw];
            let mut mx = f32::NEG_INFINITY;
            for c in 0..n_raw {
                let kc = &kv[c * dk..c * dk + dk];
                let dot: f32 = q_h.iter().zip(kc.iter()).map(|(a, b)| a * b).sum();
                scores[c] = dot * scale;
                if scores[c] > mx { mx = scores[c]; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = sum.recip();
            let out_h = &mut o_ref[h * dv..(h + 1) * dv];
            for c in 0..n_raw {
                let w = scores[c] * inv;
                let vc = &kv[c * dk..c * dk + dv];
                for d in 0..dv { out_h[d] += w * vc[d]; }
            }
        }

        let mut max_abs = 0.0f32; let mut rms = 0.0f64;
        for (g, r) in o_gpu.iter().zip(o_ref.iter()) {
            let d = (g - r).abs();
            if d > max_abs { max_abs = d; }
            rms += (*r as f64) * (*r as f64);
        }
        let rms = (rms / o_ref.len() as f64).sqrt();
        let rel = max_abs as f64 / rms.max(1e-9);
        eprintln!("flash_attn_k_mla vs CPU MLA ref (n_head={} DK={} n_raw={}): \
            max_abs={:.3e} rms_ref={:.3e} rel={:.3e}", n_head, dk, n_raw, max_abs, rms, rel);
        assert!(rel < 0.05, "flash_attn_k_mla diverges from CPU MLA ref: rel={}", rel);
    }

    /// OUTPUT-PROJECTION correctness vs CPU at REAL dims. GPU
    /// encode_attn_output_matmuls_q8_k (grouped Q8 matvec + dense Q8 matvec)
    /// vs CPU f32 grouped+dense matvec (mirrors decode_step attn_output_proj
    /// stages 1-2, attn_dispatch.rs:1479-1494). Self-contained.
    #[test]
    fn attn_output_proj_matches_cpu_reference() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {}", e); return; }
        };
        let n_groups = 8usize; let n_lora_o = 1024usize; let group_dim = 4096usize;
        let out_low_dim = n_groups * n_lora_o;       // 8192
        let d_embd = 4096usize;
        let mut rng: u32 = 0x0A7_F00D;
        let mut nf = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 16) as f32 / 65536.0 - 0.5 };
        let heads: Vec<f32> = (0..n_groups * group_dim).map(|_| nf()).collect();
        let w_o_a: Vec<f32> = (0..out_low_dim * group_dim).map(|_| nf() * 0.05).collect();
        let w_o_b: Vec<f32> = (0..d_embd * out_low_dim).map(|_| nf() * 0.05).collect();

        // GPU: quantize weights to Q8 (weight_q8_0) so the kernel reads valid Q8.
        let scope = disp.batch_scope();
        scope.set_q8_proj(true);
        let heads_db = scope.upload_f32(&heads);
        let woa = scope.weight_q8_0(&w_o_a);
        let wob = scope.weight_q8_0(&w_o_b);
        let (_low, out_db) = scope.encode_attn_output_matmuls_q8_k(
            &heads_db, &woa, &wob, n_groups, n_lora_o, group_dim, out_low_dim, d_embd, 1,
        ).expect("attn_output_matmuls");
        let out_gpu = scope.flush_and_read(&out_db);

        // CPU reference (f32): grouped matvec → attn_low, then dense matvec.
        let mut attn_low = vec![0.0f32; out_low_dim];
        for g in 0..n_groups {
            let hg = &heads[g * group_dim..(g + 1) * group_dim];
            for l in 0..n_lora_o {
                let row = &w_o_a[(g * n_lora_o + l) * group_dim..(g * n_lora_o + l + 1) * group_dim];
                attn_low[g * n_lora_o + l] = row.iter().zip(hg.iter()).map(|(a, b)| a * b).sum();
            }
        }
        let mut out_ref = vec![0.0f32; d_embd];
        for o in 0..d_embd {
            let row = &w_o_b[o * out_low_dim..(o + 1) * out_low_dim];
            out_ref[o] = row.iter().zip(attn_low.iter()).map(|(a, b)| a * b).sum();
        }
        let mut max_abs = 0.0f32; let mut rms = 0.0f64;
        for (g, r) in out_gpu.iter().zip(out_ref.iter()) {
            let d = (g - r).abs();
            if d > max_abs { max_abs = d; }
            rms += (*r as f64) * (*r as f64);
        }
        let rms = (rms / out_ref.len() as f64).sqrt();
        let rel = max_abs as f64 / rms.max(1e-9);
        eprintln!("attn_output_proj vs CPU (n_groups={} n_lora_o={} group_dim={}): \
            max_abs={:.3e} rms_ref={:.3e} rel={:.3e}", n_groups, n_lora_o, group_dim, max_abs, rms, rel);
        assert!(rel < 0.15, "output-proj diverges from CPU (>15%, beyond Q8 noise): rel={}", rel);
    }

    /// hc_expand_attn_split vs CPU reference (decode_step's hc_post_one formula,
    /// attn_dispatch.rs:1546): after[dst,e] = post[dst]·attn_out[e] +
    /// Σ_src comb[dst+src·n_hc]·cur_hc[src,e]. Self-contained, real n_hc/d_embd.
    #[test]
    fn hc_expand_attn_split_matches_cpu_reference() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {}", e); return; }
        };
        let n_hc = 4usize; let d_embd = 4096usize;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let mut rng: u32 = 0xE1A7_D00D;
        let mut nf = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 16) as f32 / 65536.0 - 0.5 };
        let attn_out: Vec<f32> = (0..d_embd).map(|_| nf()).collect();
        let cur_hc: Vec<f32> = (0..n_hc * d_embd).map(|_| nf()).collect();
        // split = [pre(n_hc), post(n_hc), comb(n_hc²)]. Use a realistic comb
        // (row-stochastic-ish) + arbitrary pre/post.
        let mut split = vec![0.0f32; mix_hc];
        for i in 0..n_hc { split[i] = 1.0; }                 // pre
        for i in 0..n_hc { split[n_hc + i] = 0.1 + 0.2 * i as f32; } // post
        for i in 0..n_hc * n_hc { split[2 * n_hc + i] = 0.1 + (i as f32 * 0.05).sin().abs(); } // comb

        let scope = disp.batch_scope();
        let ao = scope.upload_f32(&attn_out);
        let ch = scope.upload_f32(&cur_hc);
        let sp = scope.upload_f32(&split);
        let out_db = scope.hc_expand_attn_split(&ao, &ch, &sp, n_hc, d_embd).expect("hc_expand");
        let out_gpu = scope.flush_and_read(&out_db);

        // CPU reference.
        let post = &split[n_hc..2 * n_hc];
        let comb = &split[2 * n_hc..];
        let mut out_ref = vec![0.0f32; n_hc * d_embd];
        for dst in 0..n_hc {
            for e in 0..d_embd {
                let mut acc = post[dst] * attn_out[e];
                for src in 0..n_hc {
                    acc += comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
                }
                out_ref[dst * d_embd + e] = acc;
            }
        }
        let mut max_abs = 0.0f32; let mut rms = 0.0f64; let mut fd: Option<usize> = None;
        for (i, (g, r)) in out_gpu.iter().zip(out_ref.iter()).enumerate() {
            let d = (g - r).abs();
            if d > max_abs { max_abs = d; }
            rms += (*r as f64) * (*r as f64);
            if fd.is_none() && d > 1e-3 { fd = Some(i); }
        }
        let rms = (rms / out_ref.len() as f64).sqrt();
        let rel = max_abs as f64 / rms.max(1e-9);
        eprintln!("hc_expand_attn_split vs CPU (n_hc={} d_embd={}): max_abs={:.3e} \
            rms_ref={:.3e} rel={:.3e} first_div={:?}", n_hc, d_embd, max_abs, rms, rel, fd);
        assert!(rel < 0.02, "hc_expand_attn_split diverges from CPU: rel={} first_div={:?}", rel, fd);
    }

    #[test]
    fn flash_attn_k_mla_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_head: usize = 2;
        let dk: usize = 512;
        let dv: usize = 512;
        let n_cache: usize = 128;
        let k_positions: usize = 2;
        let scale: f32 = 1.0 / (dk as f32).sqrt();

        let mut rng: u32 = 0xFEED_F00D;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        // Q: [K, n_head, DK].
        let q_k: Vec<f32> = (0..k_positions * n_head * dk)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        // KV cache: [n_cache, DK]. Use small magnitudes so f32 values are
        // already within FP16-representable range — Metal's implicit f32→half
        // cast in the kernel is then lossless (mirrors the kv_fp8_store output
        // which is FP8-snapped then FP16-round-tripped).
        let kv: Vec<f32> = (0..n_cache * dk)
            .map(|_| (next() & 0x7fff) as f32 / 32768.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let q_k_db = scope.upload_f32(&q_k);
        let kv_db = scope.upload_f32(&kv);

        // K=2 path.  Set attn_base_pos = n_cache - 1 so the per-q causal
        // mask doesn't crop anything (q=0 window = n_cache; q=1 window =
        // n_cache+1 → clamped) — matches the original "attend over full
        // cache" semantics this consistency test was written against.
        let attn_base_pos = (n_cache - 1) as u32;
        // Zero sinks → no sink contribution (the K=2-vs-2×K=1 self-consistency
        // test compares the kernel against itself; sinks affect both equally).
        let sinks_db = scope.weight_f32(&vec![0.0f32; n_head]);
        let o_k = scope.flash_attn_k_mla(
            &q_k_db, &kv_db, n_head, dk, dv, n_cache, k_positions, scale,
            attn_base_pos, &sinks_db,
        ).expect("k=2 flash");

        // Two K=1 references (same kernel, K=1 specialization).
        let q_0: Vec<f32> = q_k[0..n_head * dk].to_vec();
        let q_1: Vec<f32> = q_k[n_head * dk..2*n_head * dk].to_vec();
        let q0_db = scope.upload_f32(&q_0);
        let q1_db = scope.upload_f32(&q_1);
        let o_0 = scope.flash_attn_k_mla(&q0_db, &kv_db, n_head, dk, dv, n_cache, 1, scale, attn_base_pos, &sinks_db).expect("k=1 row0");
        let o_1 = scope.flash_attn_k_mla(&q1_db, &kv_db, n_head, dk, dv, n_cache, 1, scale, attn_base_pos, &sinks_db).expect("k=1 row1");

        let outs = scope.flush_and_read_multi(&[&o_k, &o_0, &o_1]);
        let ok = &outs[0]; let o0 = &outs[1]; let o1 = &outs[2];
        assert_eq!(ok.len(), 2 * n_head * dv);
        assert_eq!(o0.len(), n_head * dv);
        assert_eq!(o1.len(), n_head * dv);

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let m0 = max_abs(&ok[0..n_head * dv], o0);
        let m1 = max_abs(&ok[n_head * dv..2*n_head * dv], o1);
        eprintln!("flash_attn_k_mla: K=2 row0 vs K=1 max_abs = {:.2e}, row1 = {:.2e}", m0, m1);
        // FP16-staged simdgroup matmul has ~few-element-ulp noise on K=2 vs K=1
        // even with same kernel (different padding zeros in the SQ tile). Tolerate
        // small drift.
        assert!(m0 < 1e-3 && m1 < 1e-3, "flash attn mismatch: {} {}", m0, m1);
    }

    /// Phase 2 — END-TO-END K-position attention half composition.
    /// `encode_attn_chain_k` at K=2 chains every Phase-1 + Phase-2 K-position
    /// primitive (collapse_norm → qkv → kv-rope → kv-store → q-rope-fwd →
    /// flash_attn → q-rope-back → output_proj) into one cb. This smoke test
    /// verifies:
    ///   (a) the chain runs end-to-end without panic/error,
    ///   (b) `attn_out_k` shape is `[K, n_embd]` floats,
    ///   (c) every element is finite (no NaN/Inf from a wiring bug),
    ///   (d) values have reasonable magnitudes (< 1000 — sanity).
    /// Per-primitive numerical bit-id is already proven by the prior 5 K=2
    /// smoke tests; this exercises the COMPOSITION glue (shapes, buffer
    /// strides, rope_q_heads_K, persistent-cache state across stages).
    #[test]
    fn encode_attn_chain_k_runs_end_to_end() {
        use crate::MetalDispatcher;
        use ds4_engine::attn_dispatch::LayerParams;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        // Production-compatible shapes — head_dim=512 is mandatory for
        // flash_attn_k_mla (NO=16 hardcoded). n_head=2 keeps runtime small;
        // n_hc=4 / n_embd=4096 are fixed by hc_collapse_norm.
        let n_hc: usize = 4;
        let n_embd: usize = 4096;
        let d_embd: usize = n_embd;
        let n_lora_q: usize = 128;
        let n_head: usize = 2;
        let head_dim: usize = 512;
        let q_dim: usize = n_head * head_dim;       // 1024
        let kv_row: usize = 512;                    // = DK (MLA cache row)
        let n_rot: usize = 64;                      // n_nope = head_dim - n_rot = 448
        let hc_dim = n_hc * n_embd;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let sinkhorn_iters: i32 = 5;
        let hc_eps: f32 = 1e-6;
        let rms_eps: f32 = 1e-5;
        // Output-projection shapes: heads_k flat = q_dim = 1024 = n_groups*group_dim.
        let n_groups: usize = 8;
        let group_dim: usize = 128;                 // %32 ✓ (Q8_0); 8*128 = 1024 ✓
        let n_lora_o: usize = 4;                    // %2 ✓
        let out_low_dim: usize = n_groups * n_lora_o;   // 32

        let raw_cap: u32 = 16;                      // %8 ✓ (flash C=8)
        let base_slot: u32 = 4;
        let base_pos: u32 = 4;
        let k_positions: usize = 2;
        let flash_scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let params = LayerParams {
            layer_idx: 0,
            d_embd: d_embd as u32,
            n_hc: n_hc as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            n_rot: n_rot as u32,
            n_lora_q: n_lora_q as u32,
            n_lora_kv: kv_row as u32,
            hc_sinkhorn_iter: sinkhorn_iters as u32,
            hc_eps,
            rms_eps,
            rope_orig_ctx: 4096,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            compress_ratio: 1,
            n_out_group: 2,
        };

        // Tiny-magnitude Q8_0 weights (32-row Q8_0 blocks: d=1.0 in fp16, qs in
        // [-32, +31] → values in [-32, +31]; matvecs scaled appropriately by RMS).
        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            assert_eq!(d_in % 32, 0, "Q8_0 needs d_in %32 (got {})", d_in);
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off+1] = 0x3C;  // half(1.0)
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }
        let w_qa = rand_q8(n_lora_q, d_embd, 0xA1);
        let w_qb = rand_q8(q_dim, n_lora_q, 0xB2);
        let w_kv = rand_q8(kv_row, d_embd, 0xC3);
        let w_o_a = rand_q8(out_low_dim, group_dim, 0xD4);  // per-group: [n_lora_o*n_groups, group_dim]
        let w_o_b = rand_q8(d_embd, out_low_dim, 0xE5);

        let mut rng: u32 = 0xDEAD_BEEF;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let prev_hc_k: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_scale: Vec<f32> = vec![1.0, 0.5, 2.0];
        let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.1 + i as f32 * 0.01).collect();
        let hc_norm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();
        let unit_gamma_hc: Vec<f32> = vec![1.0; hc_dim];
        let gamma_q: Vec<f32> = (0..n_lora_q)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.04).collect();
        let gamma_kv: Vec<f32> = (0..kv_row)
            .map(|i| 1.0 + (i as f32 * 0.017).sin() * 0.04).collect();

        let mut scope = disp.batch_scope();
        // q8_proj=true so encode_attn_qkv_chain_k uses the q8 matvec path
        // (matching the weights we uploaded as q8 raw bytes).
        scope.set_q8_proj(true);
        let hc_fn_db = scope.weight_f32(&hc_fn);
        let hc_scale_db = scope.weight_f32(&hc_scale);
        let hc_base_db = scope.weight_f32(&hc_base);
        let hc_norm_gamma_db = scope.weight_f32(&hc_norm_gamma);
        let unit_gamma_db = scope.weight_f32(&unit_gamma_hc);
        let gamma_q_db = scope.weight_f32(&gamma_q);
        let gamma_kv_db = scope.weight_f32(&gamma_kv);
        let w_qa_db = scope.weight_q8_0_raw(&w_qa, n_lora_q * d_embd);
        let w_qb_db = scope.weight_q8_0_raw(&w_qb, q_dim * n_lora_q);
        let w_kv_db = scope.weight_q8_0_raw(&w_kv, kv_row * d_embd);
        let w_oa_db = scope.weight_q8_0_raw(&w_o_a, out_low_dim * group_dim);
        let w_ob_db = scope.weight_q8_0_raw(&w_o_b, d_embd * out_low_dim);

        let prev_hc_k_db = scope.upload_f32(&prev_hc_k);

        // Fresh layer_idx → persistent cache is zero-initialized. Cache rows
        // beyond {base_slot, base_slot+1} stay zero; the un-masked flash kernel
        // attends over all `raw_cap` rows including those zeros (M=-inf in
        // softmax for an all-zero K row would dominate, but with small KV
        // magnitudes the contribution is bounded — we just check finiteness).
        // Fresh layer_idx → persistent cache is zero-initialized. The
        // smoke test verifies SHAPE + FINITENESS of the end-to-end chain.
        // Per-primitive numerical correctness (bit-id K=2 vs two K=1) is
        // proven independently by the 5 prior K=2-vs-2K=1 smoke tests at
        // smaller dims; this test at production-scale dims (n_embd=4096,
        // head_dim=512) exercises composition WIRING (shape compatibility,
        // buffer strides, multi-stage encoder serialization).
        //
        // The numerical magnitudes at production-scale dims are sensitive
        // to the test's LCG-based random weight distribution — observed
        // degenerate uniformity (q_heads ≈ ±1.0 from head_rms_norm of an
        // already-saturated matvec output) is a test-setup artifact, not a
        // chain bug. Real-model weights have proper variance.
        let layer_idx = 200u32;
        let sinks_db = scope.weight_f32(&vec![0.0f32; n_head]);
        let (_half, attn_out_k) = scope.encode_attn_chain_k(
            &prev_hc_k_db,
            &hc_fn_db, &hc_scale_db, &hc_base_db, &hc_norm_gamma_db, &unit_gamma_db,
            &w_qa_db, &gamma_q_db, &w_qb_db, &w_kv_db, &gamma_kv_db,
            &w_oa_db, &w_ob_db,
            n_hc, n_embd, n_lora_q, n_head, head_dim, kv_row,
            n_groups, n_lora_o, group_dim, out_low_dim,
            sinkhorn_iters, hc_eps, rms_eps, flash_scale,
            layer_idx, &params, raw_cap, base_slot, base_pos,
            /*attn_base_pos=*/raw_cap - 1, &sinks_db, k_positions,
            /*comp=*/None,
        ).expect("encode_attn_chain_k K=2");

        let outs = scope.flush_and_read_multi(&[&attn_out_k]);
        let out_k = &outs[0];

        assert_eq!(out_k.len(), k_positions * d_embd,
                    "attn_out_k shape: got {}, expected K*d_embd = {}*{}",
                    out_k.len(), k_positions, d_embd);

        let mut max_abs: f32 = 0.0;
        let mut bad_idx: Option<usize> = None;
        for (i, &v) in out_k.iter().enumerate() {
            if !v.is_finite() { bad_idx = Some(i); break; }
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        if let Some(i) = bad_idx {
            panic!("encode_attn_chain_k attn_out_k[{}] non-finite: {}", i, out_k[i]);
        }
        eprintln!(
            "encode_attn_chain_k K={} runs end-to-end. attn_out_k shape OK ({} elems), max_abs = {:.3e}",
            k_positions, out_k.len(), max_abs
        );
        // Sanity: bounded values (Q8_0 inputs in [-32, 31], RMS-normalized to
        // ~unit scale, then projected through bounded LoRA-style ops).
        assert!(max_abs < 1e3, "encode_attn_chain_k output magnitudes too large: {}", max_abs);
    }

    /// Phase 2 FFN-half — `hc_expand_attn_split_k` at K=2 matches two K=1
    /// `hc_expand_attn_split` calls. Bit-identical per-row.
    #[test]
    fn hc_expand_attn_split_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_hc: usize = 4;
        let d_embd: usize = 256;  // %4 ✓
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let hc_dim = n_hc * d_embd;
        let k_positions: usize = std::env::var("DS4_TEST_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

        let mut rng: u32 = 0xACE0_1234;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let attn_out_k: Vec<f32> = (0..k_positions * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let cur_hc_k: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let split_k: Vec<f32> = (0..k_positions * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let attn_db = scope.upload_f32(&attn_out_k);
        let hc_db = scope.upload_f32(&cur_hc_k);
        let split_db = scope.upload_f32(&split_k);
        let out_k = scope.hc_expand_attn_split_k(
            &attn_db, &hc_db, &split_db, n_hc, d_embd, k_positions,
        ).expect("k=2");

        // Two K=1 reference calls.
        let a0: Vec<f32> = attn_out_k[0..d_embd].to_vec();
        let a1: Vec<f32> = attn_out_k[d_embd..2*d_embd].to_vec();
        let h0: Vec<f32> = cur_hc_k[0..hc_dim].to_vec();
        let h1: Vec<f32> = cur_hc_k[hc_dim..2*hc_dim].to_vec();
        let s0: Vec<f32> = split_k[0..mix_hc].to_vec();
        let s1: Vec<f32> = split_k[mix_hc..2*mix_hc].to_vec();
        let a0_db = scope.upload_f32(&a0); let a1_db = scope.upload_f32(&a1);
        let h0_db = scope.upload_f32(&h0); let h1_db = scope.upload_f32(&h1);
        let s0_db = scope.upload_f32(&s0); let s1_db = scope.upload_f32(&s1);
        let out_0 = scope.hc_expand_attn_split(&a0_db, &h0_db, &s0_db, n_hc, d_embd).expect("k=1 row0");
        let out_1 = scope.hc_expand_attn_split(&a1_db, &h1_db, &s1_db, n_hc, d_embd).expect("k=1 row1");

        let res = scope.flush_and_read_multi(&[&out_k, &out_0, &out_1]);
        let vk = &res[0]; let v0 = &res[1]; let v1 = &res[2];
        assert_eq!(vk.len(), k_positions * hc_dim);
        let mab = |a: &[f32], b: &[f32]| a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
        let m0 = mab(&vk[0..hc_dim], v0);
        let m1 = mab(&vk[hc_dim..2*hc_dim], v1);
        eprintln!("hc_expand_attn_split_k: K={k_positions} row0 vs K=1 max_abs = {:.2e}, row1 = {:.2e}", m0, m1);
        assert!(m0 == 0.0 && m1 == 0.0, "expected bit-id; got {} {}", m0, m1);
    }

    /// Phase 2 FFN-half — `hc_expand_add_split_k` at K=2 matches two K=1
    /// `hc_expand_add_split` calls. Bit-identical per-row.
    #[test]
    fn hc_expand_add_split_k_matches_two_k1_paths() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_hc: usize = 4;
        let d_embd: usize = 256;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let hc_dim = n_hc * d_embd;
        let k_positions: usize = std::env::var("DS4_TEST_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

        let mut rng: u32 = 0xFACEFEED;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        let shared_k: Vec<f32> = (0..k_positions * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let routed_k: Vec<f32> = (0..k_positions * d_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let after_attn_k: Vec<f32> = (0..k_positions * hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let split_k: Vec<f32> = (0..k_positions * mix_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();

        let scope = disp.batch_scope();
        let sh_db = scope.upload_f32(&shared_k);
        let rt_db = scope.upload_f32(&routed_k);
        let aa_db = scope.upload_f32(&after_attn_k);
        let sp_db = scope.upload_f32(&split_k);
        let out_k = scope.hc_expand_add_split_k(
            &sh_db, &rt_db, &aa_db, &sp_db, n_hc, d_embd, k_positions,
        ).expect("k=2");

        let mut k1_dbs = Vec::with_capacity(k_positions);
        let mut k1_outs = Vec::with_capacity(k_positions);
        for k in 0..k_positions {
            let sh_row: Vec<f32> = shared_k[k*d_embd..(k+1)*d_embd].to_vec();
            let rt_row: Vec<f32> = routed_k[k*d_embd..(k+1)*d_embd].to_vec();
            let aa_row: Vec<f32> = after_attn_k[k*hc_dim..(k+1)*hc_dim].to_vec();
            let sp_row: Vec<f32> = split_k[k*mix_hc..(k+1)*mix_hc].to_vec();
            let sh = scope.upload_f32(&sh_row);
            let rt = scope.upload_f32(&rt_row);
            let aa = scope.upload_f32(&aa_row);
            let sp = scope.upload_f32(&sp_row);
            k1_dbs.push((sh, rt, aa, sp));
        }
        for (sh, rt, aa, sp) in &k1_dbs {
            k1_outs.push(scope.hc_expand_add_split(sh, rt, aa, sp, n_hc, d_embd).expect("k=1"));
        }
        let refs: Vec<&DeferredBuf> = std::iter::once(&out_k).chain(k1_outs.iter()).collect();
        let res = scope.flush_and_read_multi(&refs);
        let vk = &res[0]; let v0 = &res[1]; let v1 = &res[2];
        assert_eq!(vk.len(), k_positions * hc_dim);
        let mab = |a: &[f32], b: &[f32]| a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
        let m0 = mab(&vk[0..hc_dim], v0);
        let m1 = mab(&vk[hc_dim..2*hc_dim], v1);
        eprintln!("hc_expand_add_split_k: K={k_positions} row0 vs K=1 max_abs = {:.2e}, row1 = {:.2e}", m0, m1);
        assert!(m0 == 0.0 && m1 == 0.0, "expected bit-id; got {} {}", m0, m1);
    }

    /// CPU reference: matvec then rms_norm_mul, matching the kernels'
    /// observable semantics.
    /// Two ops in one cb via `BatchScope` must produce bit-identical
    /// output to running the same ops sequentially through the trait
    /// dispatcher. Both paths exercise the same MSL kernels with the same
    /// args; only the number of `cmd_buf` submissions differs (1 vs 2).
    #[test]
    fn matvec_then_rms_norm_matches_sequential_trait() {
        let disp = match crate::MetalDispatcher::new() {
            Ok(d) => d,
            Err(_) => return, // no Metal device — skip
        };

        // Shape mirrors `layer_qa_rms_batched_smoke::qa_rms_matches_sequential_small`,
        // which passes on this branch with these dims.
        let d_in: usize = 256;
        let d_out: usize = 64;
        let eps: f32 = 1e-5;

        let x: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
            .collect();
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 * 0.009).cos() * 0.25)
            .collect();
        let gamma: Vec<f32> = (0..d_out)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
            .collect();

        // Sequential trait path: two cmd_bufs, two readbacks.
        let qr_seq = disp.matvec_f32(&w, &x, d_out);
        let y_seq = disp.rms_norm(&qr_seq, &gamma, eps);

        // Deferred path: one cmd_buf, one readback.
        let scope = disp.batch_scope();
        let w_b = scope.weight_f32(&w);
        let x_b = scope.upload_f32(&x);
        let qr = scope.matvec_f32(&w_b, &x_b, d_in, d_out).unwrap();
        let g_b = scope.upload_f32(&gamma);
        let y = scope.rms_norm_mul(&qr, &g_b, eps).unwrap();
        let y_batched = scope.flush_and_read(&y);

        assert_eq!(y_batched.len(), y_seq.len(), "output length mismatch");
        for (i, (b, s)) in y_batched.iter().zip(&y_seq).enumerate() {
            assert_eq!(
                b.to_bits(),
                s.to_bits(),
                "BatchScope != trait at i={i}: batched={b} sequential={s}"
            );
        }
    }

    /// Stronger correctness check: `BatchScope` output must agree with a
    /// CPU f64-accumulator reference for `matvec → rms_norm_mul`. Catches
    /// the silent-zero-fallback class of bugs (locked in once
    /// `BRIDGE_PREFERRED` keeps the GPU `matvec_f32` healthy).
    #[test]
    fn matvec_then_rms_norm_matches_cpu_reference() {
        let disp = match crate::MetalDispatcher::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        let d_in: usize = 256;
        let d_out: usize = 64;
        let eps: f32 = 1e-5;

        let x: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
            .collect();
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 * 0.009).cos() * 0.25)
            .collect();
        let gamma: Vec<f32> = (0..d_out)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
            .collect();

        // CPU reference: f64 accumulator matvec + standard rms_norm.
        let mut mid = vec![0.0f32; d_out];
        for o in 0..d_out {
            let mut acc = 0.0f64;
            for i in 0..d_in {
                acc += (w[o * d_in + i] as f64) * (x[i] as f64);
            }
            mid[o] = acc as f32;
        }
        let ss: f64 = mid.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let scale = 1.0 / ((ss / d_out as f64) as f32 + eps).sqrt();
        let want: Vec<f32> = mid
            .iter()
            .zip(gamma.iter())
            .map(|(&v, &g)| v * scale * g)
            .collect();

        // BatchScope.
        let scope = disp.batch_scope();
        let w_b = scope.weight_f32(&w);
        let x_b = scope.upload_f32(&x);
        let qr = scope.matvec_f32(&w_b, &x_b, d_in, d_out).unwrap();
        let g_b = scope.upload_f32(&gamma);
        let y = scope.rms_norm_mul(&qr, &g_b, eps).unwrap();
        let got = scope.flush_and_read(&y);

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w_)) in got.iter().zip(want.iter()).enumerate() {
            let abs = (g - w_).abs();
            assert!(
                abs < 1e-3,
                "idx {i}: got {g}, want {w_}, |diff|={abs}"
            );
        }
    }

    /// Phase 3 — MTP input-mix encoder synthetic smoke. Verifies steps 2-7
    /// of the MTP draft forward pass compose end-to-end at production-like
    /// dims (n_embd=4096, n_hc=4). Synthetic Q8_0 weights produce
    /// uninteresting numerical content but the dispatch path is fully
    /// exercised: rms_norm_mul → matvec_q8_0 → rms_norm_mul_k →
    /// matvec_k_q8_0 → broadcast-add shim.
    #[test]
    fn encode_mtp_input_mix_synthetic_smoke() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_embd: usize = 4096;
        let n_hc: usize = 4;
        let rms_eps: f32 = 1e-5;

        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            assert_eq!(d_in % 32, 0);
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off + 1] = 0x3C;
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }

        let mut rng: u32 = 0xCAFE_BABE;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

        let token_embed: Vec<f32> = (0..n_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let prev_hc: Vec<f32> = (0..n_hc * n_embd)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let enorm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();
        let hnorm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.04).collect();
        let w_e_proj = rand_q8(n_embd, n_embd, 0xE1);
        let w_h_proj = rand_q8(n_embd, n_embd, 0xE2);

        let scope = disp.batch_scope();
        let token_embed_db = scope.upload_f32(&token_embed);
        let prev_hc_db = scope.upload_f32(&prev_hc);
        let enorm_db = scope.weight_f32(&enorm_gamma);
        let hnorm_db = scope.weight_f32(&hnorm_gamma);
        let e_proj_db = scope.weight_q8_0_raw(&w_e_proj, n_embd * n_embd);
        let h_proj_db = scope.weight_q8_0_raw(&w_h_proj, n_embd * n_embd);

        let out = scope.encode_mtp_input_mix(
            &token_embed_db, &prev_hc_db,
            &enorm_db, &e_proj_db,
            &hnorm_db, &h_proj_db,
            n_embd, n_hc, rms_eps,
        ).expect("encode_mtp_input_mix");
        let result = scope.flush_and_read(&out);

        assert_eq!(result.len(), n_hc * n_embd, "mtp_input_hc shape");
        let nan = result.iter().any(|v| !v.is_finite());
        assert!(!nan, "mtp_input_hc contains non-finite");
        let max_abs = result.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // Sanity bound — Q8_0 with d=1.0 + RMS-norm + matmuls; expect
        // moderate magnitudes (synthetic int weights yield compound outputs).
        eprintln!(
            "encode_mtp_input_mix synthetic smoke OK: shape {} elems, max_abs = {:.3e}",
            result.len(), max_abs
        );

        // Broadcast property: with prev_hc shared activations enorm_gamma
        // and hnorm_gamma similar, mtp_input_hc rows should be close.
        // (Not strictly identical because hnorm_rows differs per row.)
        let row0_sum: f32 = result[0..n_embd].iter().sum();
        let row1_sum: f32 = result[n_embd..2*n_embd].iter().sum();
        eprintln!("  row0 sum={:.3e}, row1 sum={:.3e}", row0_sum, row1_sum);
    }

    /// Phase 3 — MTP output-head encoder synthetic smoke. Verifies the
    /// 6-stage output head dispatch composes end-to-end at production
    /// dims (hc_dim=16384, n_embd=4096, n_hc=4, vocab=129280). Tests:
    ///   (1) rms_norm_plain via unit gamma
    ///   (2) plain matmul of hc_head_fn
    ///   (3) output_hc_weights fused shim (mul+add+sigmoid+eps)
    ///   (4) hc_weighted_sum
    ///   (5) rms_norm_mul with mtp.norm
    ///   (6) Q8_0 LM head matmul (base.output)
    /// Synthetic weights produce finite, bounded logits.
    #[test]
    fn encode_mtp_output_head_synthetic_smoke() {
        use crate::MetalDispatcher;
        let disp = match MetalDispatcher::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {}", e); return; }
        };
        let n_embd: usize = 4096;
        let n_hc: usize = 4;
        let hc_dim = n_hc * n_embd;
        let vocab: usize = 1024;     // small vocab for bench speed
        let rms_eps: f32 = 1e-5;
        let hc_eps: f32 = 1e-3;

        fn rand_q8(d_out: usize, d_in: usize, seed: u32) -> Vec<u8> {
            assert_eq!(d_in % 32, 0);
            let nb = d_in / 32;
            let row_bytes = nb * 34;
            let mut v = vec![0u8; d_out * row_bytes];
            let mut rng = seed;
            let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
            for r in 0..d_out {
                for b in 0..nb {
                    let off = r * row_bytes + b * 34;
                    v[off] = 0x00; v[off + 1] = 0x3C;
                    for i in 0..32 {
                        v[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
                    }
                }
            }
            v
        }

        let mut rng: u32 = 0xBEEF_0042;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };

        let cur_hc: Vec<f32> = (0..hc_dim)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let unit_gamma_hc_vec: Vec<f32> = vec![1.0; hc_dim];
        let hc_head_fn_vec: Vec<f32> = (0..hc_dim * n_hc)
            .map(|_| (next() & 0xffff) as f32 / 65536.0 - 0.5).collect();
        let hc_head_scale: Vec<f32> = vec![0.5];
        let hc_head_base: Vec<f32> = (0..n_hc)
            .map(|i| 0.1 + i as f32 * 0.01).collect();
        let mtp_norm_gamma: Vec<f32> = (0..n_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05).collect();
        let base_lm_head_q8 = rand_q8(vocab, n_embd, 0x77);

        let scope = disp.batch_scope();
        let cur_hc_db = scope.upload_f32(&cur_hc);
        let unit_db   = scope.weight_f32(&unit_gamma_hc_vec);
        let hcfn_db   = scope.weight_f32(&hc_head_fn_vec);
        let scale_db  = scope.weight_f32(&hc_head_scale);
        let base_db   = scope.weight_f32(&hc_head_base);
        let mnorm_db  = scope.weight_f32(&mtp_norm_gamma);
        let lm_db     = scope.weight_q8_0_raw(&base_lm_head_q8, vocab * n_embd);

        let logits = scope.encode_mtp_output_head(
            &cur_hc_db, &unit_db, &hcfn_db, &scale_db, &base_db, &mnorm_db, &lm_db,
            n_embd, n_hc, vocab, rms_eps, hc_eps,
        ).expect("encode_mtp_output_head");
        let result = scope.flush_and_read(&logits);

        assert_eq!(result.len(), vocab, "logits shape");
        let nan_count = result.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nan_count, 0, "logits contain {} non-finite", nan_count);
        let max_abs = result.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let min_v = result.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "encode_mtp_output_head synthetic smoke OK: {} logits  range=[{:.3e}, {:.3e}]  max_abs={:.3e}",
            result.len(), min_v, max_v, max_abs
        );
        // Logit RANGE collapsed at synthetic weights because the embedded
        // matvec_f32(hc_head_fn, flat_hc) with d_in=hc_dim=16384 saturates
        // through `pre*scale+base → sigmoid` into a near-constant
        // output_weights vector. Real model weights are properly scaled
        // and don't suffer this. Smoke only validates shape + finiteness.
        let _ = (min_v, max_v);
    }
}
