//! macOS-only Metal state + per-method kernel encoding stubs.
//!
//! Each `KernelDispatcher` method has its own helper here. Initially
//! they all `unimplemented!()` so the crate compiles on Mac with a
//! clear surface; per-method encoding lands incrementally once Mac
//! capacity opens.

use anyhow::Result;
use ds4_engine::gguf::GgmlType;
use metal::{Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;

use crate::quantized_experts::QuantizedExpertWeights;

// ── Phase 2c: antirez single-encoder model (DS4_CHUNK_SHARED_ENC) ──────────────
// Reuse ONE serial compute encoder across dispatches on the scope's command buffer
// (mirrors antirez g_batch_enc, ds4_metal.m:225) so Metal's robust within-encoder
// hazard tracking orders them — vs our default of a fresh encoder per op (cross-
// encoder tracking breaks at high cb occupancy = the few-cb KV-rope-tail NaN).
//
// THREAD-LOCAL (single Metal worker thread) because metal encoders aren't Send and
// MetalState is Sync (Mutex-based). SINGLE-SLOT: routed encoders are all on the
// scope's one cmd_buf at a time; Group-B helpers that make their OWN cb
// (moe_routed_step_encode, shared_chain_batched_encode) are intentionally NOT routed
// through this (their cb-commit needs their own encoder ended). Ended only at blit /
// commit / cb-rotation boundaries (end_shared_compute_enc_force), where no &ref is live.
thread_local! {
    static SHARED_ENC: std::cell::RefCell<Option<metal::ComputeCommandEncoder>> =
        const { std::cell::RefCell::new(None) };
    // Active ONLY inside one chunk prefill (set by shared_active_guard) so the
    // decode chain (DS4_CHAIN, encode_first_half_inner) keeps its own encoder
    // handling and never sees the global shared encoder.
    static SHARED_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn shared_env_on() -> bool {
    std::env::var("DS4_CHUNK_SHARED_ENC").ok().as_deref() == Some("1")
}

pub(crate) fn shared_enc_on() -> bool {
    SHARED_ACTIVE.with(|a| a.get()) && shared_env_on()
}

/// RAII guard: activate the shared compute encoder for the duration of one chunk
/// prefill. On drop it ends any open shared encoder and deactivates, so the
/// following decode path falls back to per-dispatch (legacy) encoders.
pub(crate) struct SharedActiveGuard(bool);
impl Drop for SharedActiveGuard {
    fn drop(&mut self) {
        if self.0 {
            end_shared_compute_enc_force();
            SHARED_ACTIVE.with(|a| a.set(false));
        }
    }
}
pub(crate) fn shared_active_guard() -> SharedActiveGuard {
    if shared_env_on() {
        SHARED_ACTIVE.with(|a| a.set(true));
        SharedActiveGuard(true)
    } else {
        SharedActiveGuard(false)
    }
}

// ── DS4_LAYER_PROF: non-perturbing structural profiler. Captures EVERY committed
// cb (commit_keep_open / commit_wait_stage / drain_all / commit_detach / terminal),
// then at the terminal reads GPUStartTime/GPUEndTime to split GPU-busy from idle
// bubbles. Reveals whether prefill is GPU-compute-bound or per-layer-stall-bound.
thread_local! {
    static LAYER_PROF_CBS: std::cell::RefCell<Vec<metal::CommandBuffer>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
pub(crate) fn layer_prof_on() -> bool {
    std::env::var("DS4_LAYER_PROF").ok().as_deref() == Some("1")
}
pub(crate) fn layer_prof_push(cb: &metal::CommandBufferRef) {
    if layer_prof_on() {
        LAYER_PROF_CBS.with(|v| v.borrow_mut().push(cb.to_owned()));
    }
}
/// Read every captured cb's GPU window (all must be COMPLETE — call after the
/// terminal wait), print busy/span/bubble + per-cb dur+gap, then clear.
pub(crate) fn layer_prof_report(tag: &str) {
    if !layer_prof_on() { return; }
    use metal::foreign_types::ForeignTypeRef;
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};
    LAYER_PROF_CBS.with(|v| {
        let cbs = std::mem::take(&mut *v.borrow_mut());
        if cbs.len() < 8 { return; }
        let mut spans: Vec<(f64, f64)> = cbs.iter().map(|cb| unsafe {
            let p: *mut Object = std::mem::transmute(cb.as_ptr());
            (msg_send![p, GPUStartTime], msg_send![p, GPUEndTime])
        }).collect();
        spans.retain(|(s, e)| *s > 0.0 && *e >= *s);
        spans.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if spans.is_empty() { return; }
        let t0 = spans[0].0;
        let tend = spans.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
        let busy: f64 = spans.iter().map(|(s, e)| (e - s).max(0.0)).sum();
        let span = tend - t0;
        eprintln!("[LAYER_PROF {tag}] cbs={} gpu_busy={:.1}ms gpu_span={:.1}ms bubble={:.1}ms busy/span={:.0}%",
            spans.len(), busy * 1e3, span * 1e3, (span - busy) * 1e3, busy / span * 100.0);
        let mut prev_end = t0;
        let mut bubble_n = 0;
        let mut bubble_ms = 0.0f64;
        for (s, e) in &spans {
            let gap = s - prev_end;
            if gap > 5.0e-3 { bubble_n += 1; bubble_ms += gap * 1e3; }
            prev_end = prev_end.max(*e);
        }
        eprintln!("[LAYER_PROF {tag}]   inter-cb idle gaps >5ms: {bubble_n} ({bubble_ms:.0}ms total)");
    });
}

// DS4_DISPATCH_COUNT: count compute-encoder acquisitions (≈ one Metal dispatch each,
// since legacy mode = one fresh encoder per dispatch). Read per-stage by the KPROF
// dump to find the redundant-dispatch hotspots (the per-layer cb-command-count ceiling
// — Task 0 — is ~5x antirez's; this localizes WHERE).
static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Get the compute encoder for a dispatch on `cb`. Shared mode reuses one serial
/// encoder; legacy mode returns a fresh one (caller ends via `end_shared_compute_enc`).
pub(crate) fn shared_compute_enc(cb: &metal::CommandBufferRef) -> &metal::ComputeCommandEncoderRef {
    DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !shared_enc_on() {
        return cb.new_compute_command_encoder();
    }
    SHARED_ENC.with(|s| {
        if s.borrow().is_none() {
            *s.borrow_mut() = Some(cb.new_compute_command_encoder().to_owned());
        }
        let slot = s.borrow();
        let r: &metal::ComputeCommandEncoderRef = slot.as_ref().unwrap();
        // SAFETY: the encoder lives in the thread-local until end_shared_compute_enc_force
        // (called only between dispatches at blit/commit/rotation — no &ref outstanding).
        unsafe { std::mem::transmute::<&metal::ComputeCommandEncoderRef, &metal::ComputeCommandEncoderRef>(r) }
    })
}

/// End the per-dispatch encoder. Legacy mode: end the encoder (Metal inserts an
/// implicit full barrier at the encoder boundary). Shared mode: keep the one
/// encoder open but emit an explicit buffer-scope barrier so the next dispatch
/// sees this one's writes — replicates legacy's boundary barrier (incl. aliased
/// buffers Metal's automatic hazard tracking can miss), keeping output identical
/// while collapsing ~72 encoders/layer into one. MTLBarrierScopeBuffers = 1.
pub(crate) fn end_shared_compute_enc(enc: &metal::ComputeCommandEncoderRef) {
    if shared_enc_on() {
        // DS4_SHARED_NOBARRIER=1 is a PERF-CEILING probe ONLY (max pipelining, but
        // INCORRECT — Metal's auto hazard tracking misses aliased-buffer deps like
        // rope→KV). Default emits memoryBarrierWithScope: MTLBarrierScopeBuffers (=1),
        // which replicates legacy's encoder-boundary barrier → byte-identical.
        if std::env::var("DS4_SHARED_NOBARRIER").ok().as_deref() == Some("1") {
            return;
        }
        use objc::{msg_send, sel, sel_impl};
        let _: () = unsafe { msg_send![enc, memoryBarrierWithScope: 1u64] };
    } else {
        enc.end_encoding()
    }
}

/// End the shared compute encoder if open (before a blit / commit / cb-rotation).
pub(crate) fn end_shared_compute_enc_force() {
    SHARED_ENC.with(|s| {
        if let Some(enc) = s.borrow_mut().take() { enc.end_encoding() }
    });
}

pub(crate) struct MetalState {
    pub(crate) device: Device,
    pub(crate) command_queue: metal::CommandQueue,
    /// Compiled MSL library from the generated kernel source. Lazily
    /// populated once a real .metallib source is wired up.
    pub(crate) library: Option<metal::Library>,
    /// Compiled MSL library from emitter output (benchmarks/ds4_msl/emitted/).
    /// Preferred lookup target for function-constant specialization.
    pub(crate) emitted_lib: Option<metal::Library>,
    /// Compiled MSL library from the antirez bridge (concatenated
    /// metal/*.metal with host_name prefix-renamed). Fallback when the
    /// kernel isn't in the emitted library.
    pub(crate) bridge_lib: Option<metal::Library>,
    /// Per-kernel compiled pipeline state, keyed by `KernelSpec.metal_fn`
    /// (the actual Metal symbol name, e.g. `ds4_kernel_dsv4_softplus_sqrt_f32_4`).
    pub(crate) pipelines: HashMap<String, metal::ComputePipelineState>,
    /// Cache of specialized (function-constant-bound) pipelines, keyed
    /// by `(symbol, constants_key)` where `constants_key` is a stable
    /// byte representation of the bound constant values. Populated
    /// lazily by `specialized_pipeline()` on first call per (sym, key).
    pub(crate) specialized:
        std::sync::Mutex<HashMap<(String, Vec<u8>), metal::ComputePipelineState>>,
    /// Per-MoE-layer quantized expert weight tables. Populated at model
    /// load time via `MetalState::load_expert_weights`. Empty until then,
    /// at which point `moe_routed_step` will `bail!`.
    pub(crate) expert_weights: Vec<QuantizedExpertWeights>,
    /// Identity-keyed cache mapping a weight slice's `(host_ptr,
    /// byte_len)` to its already-uploaded `MTLBuffer`. Populated lazily
    /// on first `cached_weight_buffer` call; entries live for the
    /// lifetime of `MetalState`. Activations MUST NOT use this cache —
    /// their pointer/length pair is reused across calls with different
    /// contents. See `cached_weight_buffer`.
    pub(crate) weight_buf_cache: std::sync::Mutex<HashMap<(usize, usize), metal::Buffer>>,
    /// Identity-keyed cache mapping an f32 weight slice's `(host_ptr, byte_len)`
    /// to its re-quantized GGUF `block_q8_0` MTLBuffer. Populated lazily by
    /// `cached_q8_0_weight_buffer`: the already-dequantized projection weight is
    /// quantized to Q8_0 bytes once at first use, then the GPU matvec reads it at
    /// 1 byte/weight instead of dequant→`matvec_f32` at 4 bytes/weight. Entries
    /// live for the lifetime of `MetalState`.
    pub(crate) q8_weight_buf_cache: std::sync::Mutex<HashMap<(usize, usize), metal::Buffer>>,
    /// Lazy MoE per-call scratch pool keyed by (d_in, d_ffn). Each entry
    /// owns the 7 transient buffers (x, ids, weights, dst_gate, dst_up,
    /// dst_mid, dst_final) that were previously freshly allocated on
    /// every `moe_routed_step_impl` call (43 layers × N decode tokens ⇒
    /// 7 × 43 × N allocations). With pooling, allocation happens once
    /// per unique (d_in, d_ffn) shape and the inputs are re-loaded via
    /// `contents()` memcpy each call. See M4 #362.
    pub(crate) moe_scratch: std::sync::Mutex<HashMap<(usize, usize), MoeScratch>>,
    /// SSD-streaming expert caches, one per MoE layer (DS4_SSD_STREAM only;
    /// empty otherwise). See `quantized_experts::ExpertCache`.
    pub(crate) expert_caches: Vec<std::sync::Mutex<crate::quantized_experts::ExpertCache>>,
    /// SSD-streaming chunked-prefill scratch pool: ONE whole-layer expert
    /// pool (slots == n_experts) reused across layers — `fill_layer`
    /// streams a layer's full expert set sequentially from the mmap so the
    /// chunk mm_id path runs with identity expert ids (no remap, no sel
    /// readback). The i32 tracks the resident layer (-1 = none). Caller
    /// MUST wait for in-flight GPU work using the pool before refilling.
    pub(crate) chunk_pool:
        Option<(std::sync::Mutex<crate::quantized_experts::ExpertCache>, std::sync::atomic::AtomicI32)>,
    /// Second chunk pool for double-buffered prefill (layer L on pool[L%2]):
    /// the background fill thread loads L+1 into the other pool while the
    /// GPU computes L.
    pub(crate) chunk_pool2:
        Option<(std::sync::Mutex<crate::quantized_experts::ExpertCache>, std::sync::atomic::AtomicI32)>,
    /// Pending prefetch (layer L+1) running on a background thread.
    pub(crate) chunk_prefetch: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Prefill-first budget (DS4_POOL_PIN_LAYERS=N): per-layer pools for the
    /// first N MoE layers, filled once at load and never refilled — the
    /// decode LRU budget reassigned to whole-layer prefill residency.
    pub(crate) pinned_pool:
        Vec<(std::sync::Mutex<crate::quantized_experts::ExpertCache>, std::sync::atomic::AtomicI32)>,
    /// MTLResidencySet pinning the model's mmap-backed expert weight
    /// buffers as GPU-resident (macOS 15+). Null if the OS predates 15 or
    /// `DS4_METAL_NO_RESIDENCY` is set. Stored here so the Obj-C object
    /// stays retained for the lifetime of `MetalState` — releasing it
    /// would un-pin the pages.
    ///
    /// Without this, the GPU page-faults waiting for OS to swap in the
    /// expert weight pages on every `moe_routed_step` call — Apple's
    /// mmap residency policy isn't aggressive enough for an 80+ GB model
    /// with sparse access patterns (only 6 of 256 experts touched per
    /// token). Antirez's C runtime (ds4_metal.m:352-387) does the same.
    pub(crate) model_residency_set: std::sync::Mutex<Option<*mut objc::runtime::Object>>,
    /// Set when a weight buffer was `addAllocation`'d to the residency set but
    /// not yet committed. Batched: `commit_residency` (called once per token)
    /// does the single `commit`+`requestResidency` for all buffers added since,
    /// instead of paying that O(set) cost per buffer (which thrashed). After the
    /// first decode populates the weight caches this stays false (no new adds).
    pub(crate) residency_dirty: std::sync::atomic::AtomicBool,
    /// Phase E M5.2: persistent per-layer KV cache buffers. Keyed by
    /// layer index (u32). Created lazily on first access via
    /// `kv_buffer_or_alloc`; lives for the lifetime of `MetalState`.
    ///
    /// Each buffer is `raw_cap * n_lora_kv * sizeof(f32)` bytes, sized
    /// to match the f32 storage view in `KvCacheView::raw`. The actual
    /// FP8-quantized payload uses 1/4 of that capacity (kv_fp8_store
    /// kernel writes byte-sized FP8 values into the f32-sized buffer);
    /// the over-allocation keeps the byte layout compatible with
    /// callers that interpret the buffer as f32.
    ///
    /// Once this is wired through `kv_fp8_store_impl` and the Metal
    /// flash_attn path, the buffer never round-trips through CPU on
    /// the fused-cb decode path. That eliminates the
    /// `device.new_buffer_with_data(view.raw, ...)` + readback on every
    /// kv_fp8_store call — the dominant cost of the current CPU-oracle
    /// fallback (lib.rs:341).
    pub(crate) kv_cache_buffers:
        std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    /// Per-(layer, role) reusable flash-attention SCRATCH buffers (the f16 KV
    /// workspace + the per-workgroup `tmp` reduction buffer). These are written
    /// then consumed within a single layer's flash dispatch and never read
    /// across layers, so a per-layer buffer (grow-only) can be reused every
    /// token instead of `new_buffer`'d on every call (43 layers × every token).
    /// Keyed (layer_idx, role): role 0 = flash tmp, 1 = extended-KV workspace.
    /// Only the persistent DecodeSession chain (which drains per token) opts in;
    /// other callers pass `None` and allocate fresh (unchanged).
    pub(crate) flash_scratch_buffers:
        std::sync::Mutex<HashMap<(u32, u8), metal::Buffer>>,
    /// The layer the persistent chain is currently encoding (-1 = not in the
    /// chain → flash buffers are allocated fresh, the unchanged behavior). Set
    /// around the chain's flash call so the shared `build_extended_kv_encode` /
    /// flash encoders can reuse this layer's scratch buffers. Single Metal
    /// worker thread → no contention.
    pub(crate) flash_scratch_layer: std::sync::atomic::AtomicI64,
    /// When true, the per-layer flash-scratch REUSE is suppressed (each flash
    /// allocates fresh scratch) even when `flash_scratch_reuse_enabled()`. Set
    /// by the chunk-prefill driver around its Phase-B loop: under async cb
    /// chaining (no per-layer GPU wait), reusing a layer's grow-only scratch
    /// across the K per-position flashes RACES (the next use overwrites scratch
    /// the prior, still-in-flight, flash is reading) — proven via the
    /// DS4_FLASH_SCRATCH_REUSE=0 A/B (async run1==run2, byte-identical to sync).
    /// Fresh alloc per flash removes the alias. The hot per-token decode chain
    /// (K=1, one flash per layer) never aliases, so it keeps reuse. Single Metal
    /// worker thread → no contention on this flag.
    pub(crate) flash_scratch_suppress: std::sync::atomic::AtomicBool,
    /// When true, the pooled MoE scratch (`moe_scratch`) is suppressed:
    /// `moe_routed_step_impl` allocates the 7 transient buffers FRESH per call
    /// instead of reusing the per-(d_in,d_ffn) pool. Set by the chunk-prefill
    /// driver around its async Phase-B loop. The pool is shared across ALL MoE
    /// layers (key (d_in,d_ffn) is layer-independent) and its CPU `.contents()`
    /// re-load relies on a prior `wait_until_completed`; under async cb chaining
    /// (no per-layer wait) that guarantee is gone → cross-call read/write race
    /// on the shared scratch → NaN. Fresh per-call buffers remove the alias. The
    /// hot sync/decode path keeps the pool (no regression). Single Metal worker
    /// thread → no contention.
    pub(crate) moe_scratch_suppress: std::sync::atomic::AtomicBool,
    /// Phase F task #90 — lazy-initialized pipeline for the
    /// `kernel_touch_u8_stride` warm-up kernel. Compiled from an inline
    /// MSL source (independent of the main library) so the warm-up path
    /// doesn't depend on the kernel registry or the antirez bridge.
    /// Used by `MetalDispatcher::warm_up_expert_pages_gpu`.
    pub(crate) touch_u8_stride_pipeline: std::sync::OnceLock<metal::ComputePipelineState>,
    /// M5 task #97 — lazy-initialized pipeline for
    /// `kernel_build_extended_kv`. Gathers `kv_raw` rows and indexer-
    /// selected `kv_comp` rows into one contiguous workspace buffer so
    /// the existing `kernel_flash_attn_ext_vec_f16_dk512_dv512` fast
    /// path can serve compressor/indexer layers. Same inline-MSL pattern
    /// as `touch_u8_stride_pipeline` so it does not depend on the
    /// emitted/bridge libraries.
    pub(crate) build_extended_kv_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// M5 Phase D — antirez's `kernel_dsv4_compressor_store_one`
    /// (dsv4_kv.metal:201-227). Per-token APE add + state write for
    /// the compressor frontier. The other compressor steps (matvec
    /// projections, pool, RMS norm, rope_tail, FP8) reuse existing
    /// kernels; only this one's APE+state-write fusion needs a
    /// dedicated kernel.
    pub(crate) compressor_store_one_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// M5 Phase D — antirez's `kernel_dsv4_softmax_pool`
    /// (dsv4_misc.metal:1012-1043). Fused softmax-weighted pooling of
    /// compressed KV rows; fires on the compressor's emit step
    /// (`(pos+1) % ratio == 0`) to collapse the n_rows × width state
    /// into one width-long pooled output. Inline-MSL port behind a
    /// OnceLock, same pattern as `compressor_store_one_pipeline`.
    pub(crate) softmax_pool_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// M5 Phase D — bespoke `ds4_kernel_dsv4_compressor_pool_ratio4`
    /// matching the CPU oracle `compressor_pool_decode_state`
    /// (`attn_dispatch.rs:1985`) ratio==4 branch directly. The fused
    /// softmax-weighted pool above can't express the cross-column
    /// two-window read pattern (prev window in `[0..hd]`, current
    /// window in `[hd..2*hd]`) in a single dispatch. This kernel does.
    /// One dispatch per emit, one thread per output column, bit-
    /// equivalent to the CPU oracle including the `-1e9` mask
    /// sentinel.
    pub(crate) compressor_pool_ratio4_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// DS4_CHUNK_ATTN_NOSYNC — GPU-resident ratio==4 two-window pool
    /// rotation. Replicates the CPU `finish_emit` rotation (compressor.rs
    /// ~598-613) net effect — `front_row[k][0..width] := back_row[k][0..width]`
    /// for k in [0..ratio] — entirely on the GPU so the per-quad emit no
    /// longer needs a `commit_wait` + CPU `sync_pool_to_mirror`/`write_pool_all`
    /// round-trip before the next quad's stores. One dispatch per emit, one
    /// thread per (k, column) element. The CPU-mirror finish (state resync +
    /// comp_kv_ring append + n_comp bump) is then deferred to layer end.
    pub(crate) compressor_rotate_ratio4_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// Fused ratio==4 chunk-prefill compressor (the two-window analog of
    /// `compressor_prefill_noidx`): builds ALL emit rows for the chunk in ONE
    /// dispatch (one threadgroup per emit), replacing the per-position
    /// store_one→pool→rms→rope→ring-write→rotate loop.
    pub(crate) compressor_prefill_idx_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// Group-major → K-major interleave for the grouped output-proj GEMM.
    pub(crate) interleave_group_major_pipeline:
        std::sync::OnceLock<metal::ComputePipelineState>,
    /// M5 Phase D — persistent per-layer compressor + indexer state
    /// buffers. Mirror of the CPU `state.comp_state_kv[layer_idx]` /
    /// `state.comp_state_score[layer_idx]` / `state.index_state_kv[…]`
    /// / `state.index_state_score[…]` Vec<f32>s, lifted to GPU-resident
    /// MTLBuffers so `compressor_store_one` (and the future pool +
    /// rope + fp8 chain) can read/write them directly without per-call
    /// upload/readback.
    ///
    /// Sized by the caller (typically `ratio*width` for ratio≠4 and
    /// `2*ratio*width` for ratio==4, matching the kernel's dst_row
    /// arithmetic). Allocated zero-initialized on first access
    /// (Metal's `new_buffer(length, StorageModeShared)` returns
    /// zero-filled pages on Apple Silicon — same assumption as
    /// `kv_cache_buffers`).
    pub(crate) compressor_state_kv_buffers:
        std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    pub(crate) compressor_state_score_buffers:
        std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    pub(crate) indexer_state_kv_buffers:
        std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    pub(crate) indexer_state_score_buffers:
        std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    /// Single-cb step 9 — GPU-resident compressed-KV rings, the on-device
    /// mirror of the CPU `state.comp_kv_ring[layer_idx]` /
    /// `state.index_comp_kv_ring[layer_idx]` Vec<f32>s. The compressor's
    /// emit row (already GPU-resident from the in-scope rms+rope) is copied
    /// into row `n_comp` here, so the flash gather reads `comp_rows` straight
    /// off the GPU — breaking the CPU comp_kv_ring → flash dependency that
    /// forced the attn-half scope to flush before flash. Row-major
    /// `[max_rows × head_dim]`; sized by the caller (raw_cap × head_dim).
    pub(crate) comp_ring_buffers: std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    /// Per-layer GPU-resident indexer compressed-KV ring ([rows, 128] f32),
    /// the GPU mirror of `state.index_comp_kv_ring[layer]`. Populated one row
    /// per emit (DS4_GPU_INDEXER) so the GPU indexer score op reads it without
    /// re-uploading the whole ring each token.
    pub(crate) index_comp_ring_buffers: std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    /// Phase F task #93 — counters tracking how many `cached_weight_buffer`
    /// calls took the zero-copy path (page-aligned source → `newBufferWithBytesNoCopy`)
    /// vs the memcpy fallback. Surfaced via `BufferAuditReport`.
    pub(crate) weight_no_copy_count: std::sync::atomic::AtomicUsize,
    pub(crate) weight_memcpy_count: std::sync::atomic::AtomicUsize,
    pub(crate) weight_no_copy_bytes: std::sync::atomic::AtomicU64,
    pub(crate) weight_memcpy_bytes: std::sync::atomic::AtomicU64,
}

unsafe impl Send for MetalState {}
unsafe impl Sync for MetalState {}

/// Reusable scratch buffers for one `(d_in, d_ffn)` MoE call shape.
pub(crate) struct MoeScratch {
    pub(crate) x_buf: metal::Buffer,
    pub(crate) ids_buf: metal::Buffer,
    pub(crate) weights_buf: metal::Buffer,
    pub(crate) dst_gate: metal::Buffer,
    pub(crate) dst_up: metal::Buffer,
    pub(crate) dst_mid: metal::Buffer,
    pub(crate) dst_final: metal::Buffer,
}

// `metal::Buffer` is an ARC handle to a Metal resource. We never share
// individual buffers across threads concurrently — the Mutex around the
// pool serializes access — but Rust's auto-trait inference doesn't see
// through the FFI handle, so we assert Send/Sync explicitly for the
// pool entry. Justified because:
//   1. The buffers are only accessed while the pool's Mutex is held.
//   2. Metal itself supports CPU access to shared-storage buffers from
//      any thread (Apple docs: MTLStorageModeShared is thread-safe).
unsafe impl Send for MoeScratch {}
unsafe impl Sync for MoeScratch {}

// Generated by build.rs from benchmarks/ds4_msl/emitted/*.metal.
// Provides KERNEL_SOURCES: &[(&str, &str)] keyed by file stem.
include!(concat!(env!("OUT_DIR"), "/kernel_sources.rs"));

/// Returns true if the function has unbound `function_constant`
/// specialization parameters (e.g. flash_attn templates). Such
/// functions abort the process at pipeline-state creation when
/// constants are not supplied; callers must use
/// `newFunctionWithName:constantValues:` instead.
fn function_needs_constants(func: &metal::FunctionRef) -> bool {
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};
    let _ = (class!(NSObject), sel!(count));
    let dict: *mut Object = func.function_constants_dictionary();
    if dict.is_null() {
        return false;
    }
    let count: usize = unsafe { msg_send![dict, count] };
    count > 0
}

impl MetalState {
    pub(crate) fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("no default Metal device"))?;
        let command_queue = device.new_command_queue();

        // Compile every emitted MSL file into one MTLLibrary. The
        // KERNEL_SOURCES table is baked at compile time by build.rs.
        let mut source = String::new();
        for (_name, src) in KERNEL_SOURCES {
            source.push_str(src);
            source.push('\n');
        }

        let options = metal::CompileOptions::new();
        let emitted_lib = if source.trim().is_empty() {
            None
        } else {
            Some(
                device
                    .new_library_with_source(&source, &options)
                    .map_err(|e| anyhow::anyhow!("emitted MSL compile failed: {}", e))?,
            )
        };

        // Bridge library: antirez upstream concatenated + host_name
        // prefix-renamed at build time. Compiled separately so a
        // failure in one library doesn't take the other down.
        // q4_0 dense prefill GEMM: a q4 instantiation of the bridge's kernel_mul_mm
        // template, appended so it shares the template + mul_mm_t typedef defined in
        // ANTIREZ_BRIDGE_SOURCE. 18-byte std ggml block_q4_0 (half d + 16 nibble bytes,
        // matching requant_q8_0_to_q4_0) → half the weight bytes of q8_0 on the
        // attention projections (q4 attn is argmax-safe, DS4_ATTN_Q4_PROBE-validated).
        const Q4_MM_ADDENDUM: &str = concat!(
            "\n\n// ── ds4 q4_0 dense prefill GEMM ──\n",
            "typedef struct { half d; uint8_t qs[16]; } ds4_block_q4_0_mm;\n",
            "template <typename type4x4>\n",
            "void ds4_dequantize_q4_0_mm(device const ds4_block_q4_0_mm *xb, short il, thread type4x4 & reg) {\n",
            "    device const uint8_t * qs = (device const uint8_t *)xb->qs;\n",
            "    const float d = (float)xb->d;\n",
            "    float4x4 reg_f;\n",
            "    for (int i = 0; i < 16; i++) {\n",
            "        const int nib = (il == 0) ? (int)(qs[i] & 0x0F) : (int)(qs[i] >> 4);\n",
            "        reg_f[i/4][i%4] = ((float)(nib - 8) * d);\n",
            "    }\n",
            "    reg = (type4x4) reg_f;\n",
            "}\n",
            "template [[host_name(\"ds4_kernel_mul_mm_q4_0_f32\")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, ds4_block_q4_0_mm, 2, ds4_dequantize_q4_0_mm, float, float4x4, float, float2x4>;\n",
        );
        let bridge_lib = if ANTIREZ_BRIDGE_SOURCE.trim().is_empty() {
            None
        } else {
            let bridge_src = format!("{}{}", ANTIREZ_BRIDGE_SOURCE, Q4_MM_ADDENDUM);
            match device.new_library_with_source(&bridge_src, &options) {
                Ok(l) => Some(l),
                Err(e) => {
                    eprintln!("ds4_metal: antirez bridge MSL compile failed: {}", e);
                    None
                }
            }
        };

        // Walk every registered kernel symbol; resolve from emitted
        // first, fall back to bridge. Symbols found in neither are
        // reported but don't fail the load.
        //
        // Function-constant templates (flash_attn etc.) require
        // `newFunctionWithName:constantValues:` to specialize; calling
        // plain `get_function` returns a Function but Metal aborts
        // with SIGABRT at `new_compute_pipeline_state_with_function`.
        // Detect them by inspecting `functionConstantsDictionary` and
        // skip — their composers will build pipelines via
        // constantValues at dispatch time.
        // Symbols whose emitted-lib version has an ABI INCOMPATIBLE
        // with the Rust callers (which set args struct + named buffers
        // per antirez's layout). Both the registry walk below and the
        // explicit-load loop further down consult this allowlist.
        // Same class as `BRIDGE_PREFERRED` in `specialized_pipeline`
        // (see commits e1ccf360, d9f18b8b, 310067be, this commit).
        const BRIDGE_PREFERRED_LOAD: &[&str] = &[
            // M1 fix
            "ds4_dsv4_hc_expand",
            // M4 fix: router selection + weights kernels — emitted variants
            // take individual uint buffers; Rust caller sets a packed
            // SelectArgs struct at buffer 0 per antirez `dsv4_misc.metal:82`.
            "ds4_dsv4_router_finalize_one",
            "ds4_dsv4_router_weights_one",
            // M5.2.2 fix: emitted variant swaps buffers 0/1 (p0=row,
            // p1=cache) AND uses `head_dim` scalar at buffer 2;
            // bridge shim has p0=cache, p1=row, n_nope at buffer 2.
            // The Rust caller follows the shim layout, so always
            // pick the shim. (Resolves the long-standing kv_fp8_store
            // "mis-binding" comment in lib.rs:341.)
            "ds4_dsv4_kv_fp8_store",
            // K-merged variant: stores K slots in one dispatch (bridge-only).
            "ds4_dsv4_kv_fp8_store_k",
            // STAGE 1 fused chunk-prefill compressor (noidx, ratio != 4):
            // builds the whole chunk's compressed rows in one dispatch. Bridge
            // shim only.
            "ds4_kernel_dsv4_compressor_prefill_noidx_f32",
            // STAGE 1 companion: batched trailing-partial-group pool fill.
            "ds4_kernel_dsv4_compressor_pool_fill_noidx_f32",
        ];

        let mut pipelines: HashMap<String, metal::ComputePipelineState> =
            HashMap::with_capacity(ds4_engine::kernel_registry::KERNELS.len());
        let mut missing: Vec<&str> = Vec::new();
        let mut needs_constants: Vec<&str> = Vec::new();
        for spec in ds4_engine::kernel_registry::KERNELS {
            let bridge_first = BRIDGE_PREFERRED_LOAD.contains(&spec.metal_fn);
            let func_opt = if bridge_first {
                bridge_lib
                    .as_ref()
                    .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                    .or_else(|| {
                        emitted_lib
                            .as_ref()
                            .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                    })
            } else {
                emitted_lib
                    .as_ref()
                    .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                    .or_else(|| {
                        bridge_lib
                            .as_ref()
                            .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                    })
            };
            match func_opt {
                Some(func) => {
                    if function_needs_constants(&func) {
                        needs_constants.push(spec.metal_fn);
                        continue;
                    }
                    match device.new_compute_pipeline_state_with_function(&func) {
                        Ok(pso) => {
                            if pso.max_total_threads_per_threadgroup() < 1024 {
                                eprintln!(
                                    "ds4_metal: pipeline {} maxThreads={} (<1024)",
                                    spec.metal_fn,
                                    pso.max_total_threads_per_threadgroup()
                                );
                            }
                            pipelines.insert(spec.metal_fn.to_string(), pso);
                        }
                        Err(e) => {
                            eprintln!(
                                "ds4_metal: pipeline build failed for {}: {} (skipped)",
                                spec.metal_fn, e
                            );
                            missing.push(spec.metal_fn);
                        }
                    }
                }
                None => missing.push(spec.metal_fn),
            }
        }
        if !missing.is_empty() {
            eprintln!(
                "ds4_metal: {} kernel symbol(s) not found in emitted or bridged MSL: {:?}",
                missing.len(),
                missing
            );
        }
        if !needs_constants.is_empty() {
            eprintln!(
                "ds4_metal: {} kernel symbol(s) require function-constant \
                 specialization (composer must use constantValues): {:?}",
                needs_constants.len(),
                needs_constants
            );
        }

        // Pick one library to retain for legacy `library` field; keep
        // both libs separately for specialized_pipeline() lookups.
        let library = emitted_lib.clone().or_else(|| bridge_lib.clone());

        // M4 #330e: register `ds4_head_rms_norm_f32` outside the registry
        // walk so it doesn't have to live in `kernel_registry::KERNELS`
        // (which is gated on an MLIR-emitter fixture in print_msl.rs).
        // It's a plain hand-written kernel — load it the same way the
        // registry walk loads everything else.
        // (Bridge-preferred list is `BRIDGE_PREFERRED_LOAD`, declared
        // above and shared with the registry walk.)
        for sym in [
            "ds4_head_rms_norm_f32",
            "ds4_q8_0_round_trip_f32",
            "ds4_silu_default_f32",
            "ds4_silu_fidelity_f32",
            "ds4_sigmoid_default_f32",
            "ds4_sigmoid_fidelity_f32",
            "ds4_softplus_sqrt_default_f32",
            "ds4_softplus_sqrt_fidelity_f32",
            "ds4_kernel_swiglu_f32",
            // Phase E M1: antirez hc_expand brought in via the bridge.
            // Not on the production hot path yet (CPU `hc_expand_add_only`
            // still runs); preload it so `BatchScope::hc_expand_add` (and
            // future fused-cb users) can find the pipeline.
            "ds4_dsv4_hc_expand",
            // Phase E M2: antirez hc_split_weighted_sum_norm4 brought in
            // via the bridge. Implements the full split+sinkhorn+
            // weighted_sum+(rms_norm with gamma) tail of hc_collapse_norm
            // in one kernel. Hardcoded for n_hc=4, n_embd=4096 (DS4 prod).
            "ds4_dsv4_hc_split_weighted_sum_norm4",
            // Width-generic variant (bridge shim) for non-Flash n_embd
            // (PRO: 7168). Same math, n_embd from args.
            "ds4_dsv4_hc_split_weighted_sum_norm_any",
            // Expert-count/scale-generic router shims (PRO: 384 experts,
            // scale 2.5).
            "ds4_dsv4_router_finalize_one_any",
            "ds4_dsv4_router_weights_one_any",
            // K-merged KV FP8 store — writes K slots in one dispatch
            // (eliminates the per-slot encoder loop in
            // kv_fp8_store_persistent_k). Bridge-only; not in the registry.
            "ds4_dsv4_kv_fp8_store_k",
            // K-batched hash-router weights (chunk-graph stage 3): kills the
            // per-chunk probs_k drain for hash layers. Bridge-only.
            "ds4_dsv4_router_weights_k",
            // STAGE 1 fused chunk-prefill compressor (noidx, ratio != 4).
            "ds4_kernel_dsv4_compressor_prefill_noidx_f32",
            "ds4_kernel_dsv4_compressor_pool_fill_noidx_f32",
        ] {
            let bridge_first = BRIDGE_PREFERRED_LOAD.contains(&sym);
            let func_opt = if bridge_first {
                bridge_lib
                    .as_ref()
                    .and_then(|l| l.get_function(sym, None).ok())
                    .or_else(|| {
                        emitted_lib
                            .as_ref()
                            .and_then(|l| l.get_function(sym, None).ok())
                    })
            } else {
                emitted_lib
                    .as_ref()
                    .and_then(|l| l.get_function(sym, None).ok())
                    .or_else(|| {
                        bridge_lib
                            .as_ref()
                            .and_then(|l| l.get_function(sym, None).ok())
                    })
            };
            match func_opt {
                Some(func) => {
                    if function_needs_constants(&func) {
                        eprintln!("ds4_metal: preload symbol {sym} skipped (needs function constants)");
                    }
                    if !function_needs_constants(&func) {
                        match device.new_compute_pipeline_state_with_function(&func) {
                            Ok(pso) => {
                                if pso.max_total_threads_per_threadgroup() < 1024 {
                                    eprintln!(
                                        "ds4_metal: pipeline {sym} maxThreads={} (<1024)",
                                        pso.max_total_threads_per_threadgroup()
                                    );
                                }
                                pipelines.insert(sym.to_string(), pso);
                            }
                            Err(e) => eprintln!("ds4_metal: pipeline build failed for {sym}: {e}"),
                        }
                    }
                }
                None => eprintln!("ds4_metal: preload symbol {sym} not found in emitted or bridge lib"),
            }
        }

        Ok(Self {
            device,
            command_queue,
            library,
            emitted_lib,
            bridge_lib,
            pipelines,
            specialized: std::sync::Mutex::new(HashMap::new()),
            expert_weights: Vec::new(),
            weight_buf_cache: std::sync::Mutex::new(HashMap::new()),
            q8_weight_buf_cache: std::sync::Mutex::new(HashMap::new()),
            moe_scratch: std::sync::Mutex::new(HashMap::new()),
            expert_caches: Vec::new(),
            chunk_pool: None,
            chunk_pool2: None,
            chunk_prefetch: std::sync::Mutex::new(None),
            pinned_pool: Vec::new(),
            model_residency_set: std::sync::Mutex::new(None),
            residency_dirty: std::sync::atomic::AtomicBool::new(false),
            kv_cache_buffers: std::sync::Mutex::new(HashMap::new()),
            flash_scratch_buffers: std::sync::Mutex::new(HashMap::new()),
            flash_scratch_layer: std::sync::atomic::AtomicI64::new(-1),
            flash_scratch_suppress: std::sync::atomic::AtomicBool::new(false),
            moe_scratch_suppress: std::sync::atomic::AtomicBool::new(false),
            touch_u8_stride_pipeline: std::sync::OnceLock::new(),
            build_extended_kv_pipeline: std::sync::OnceLock::new(),
            compressor_store_one_pipeline: std::sync::OnceLock::new(),
            softmax_pool_pipeline: std::sync::OnceLock::new(),
            compressor_pool_ratio4_pipeline: std::sync::OnceLock::new(),
            compressor_rotate_ratio4_pipeline: std::sync::OnceLock::new(),
            compressor_prefill_idx_pipeline: std::sync::OnceLock::new(),
            interleave_group_major_pipeline: std::sync::OnceLock::new(),
            compressor_state_kv_buffers: std::sync::Mutex::new(HashMap::new()),
            compressor_state_score_buffers: std::sync::Mutex::new(HashMap::new()),
            indexer_state_kv_buffers: std::sync::Mutex::new(HashMap::new()),
            indexer_state_score_buffers: std::sync::Mutex::new(HashMap::new()),
            comp_ring_buffers: std::sync::Mutex::new(HashMap::new()),
            index_comp_ring_buffers: std::sync::Mutex::new(HashMap::new()),
            weight_no_copy_count: std::sync::atomic::AtomicUsize::new(0),
            weight_memcpy_count: std::sync::atomic::AtomicUsize::new(0),
            weight_no_copy_bytes: std::sync::atomic::AtomicU64::new(0),
            weight_memcpy_bytes: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Build (or fetch cached) the `kernel_touch_u8_stride` pipeline.
    /// Mirrors antirez `ds4_metal.m:1155-1165`. Compiled from an inline
    /// MSL source so warm-up doesn't depend on the kernel registry.
    pub(crate) fn ensure_touch_u8_stride_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.touch_u8_stride_pipeline.get() {
            return Ok(p);
        }
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void ds4_kernel_touch_u8_stride(\n\
        device const uchar    *src        [[buffer(0)]],\n\
        device uchar          *dst        [[buffer(1)]],\n\
        constant ulong        &stride     [[buffer(2)]],\n\
        constant ulong        &bytes      [[buffer(3)]],\n\
        constant ulong        &dst_offset [[buffer(4)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    ulong off = (ulong)gid * stride;\n\
    if (off >= bytes) return;\n\
    dst[dst_offset + (ulong)gid] = src[off];\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("touch_u8_stride library compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_touch_u8_stride", None)
            .map_err(|e| anyhow::anyhow!("touch_u8_stride function lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("touch_u8_stride pipeline build failed: {}", e))?;
        let _ = self.touch_u8_stride_pipeline.set(pso);
        Ok(self.touch_u8_stride_pipeline.get().unwrap())
    }

    /// M5 task #97 — build (or fetch cached) the
    /// `ds4_kernel_build_extended_kv` pipeline. Compiled from inline MSL
    /// so it does not depend on the emitted/bridge libraries (same
    /// pattern as `ensure_touch_u8_stride_pipeline`).
    ///
    /// Kernel assumes `kv_raw` rows already share stride `head_dim` with
    /// `kv_comp` (the standard V4 Flash layout where
    /// `n_lora_kv == head_dim`). The caller-side gate in
    /// `flash_attn_decode_impl` enforces that precondition.
    ///
    /// Step 2 (task #97): writes `half` (f16) into the workspace so the
    /// output can feed `flash_attn_decode_metal_encode_with_kv_buf`
    /// directly — no CPU readback, no per-call f32→f16 staging.
    ///
    /// Step 3 (task #97): the kernel writes (half)0 for rows in
    /// `[n_raw + n_selected, n_workspace_rows)` so the caller can pad
    /// the workspace up to the kernel's 32-row alignment without leaving
    /// uninitialized lanes that could produce NaN in the flash_attn dot.
    /// Padded rows must still be masked out via the flash_attn mask
    /// buffer (see `n_raw_valid` in `_with_kv_buf`).
    pub(crate) fn ensure_build_extended_kv_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.build_extended_kv_pipeline.get() {
            return Ok(p);
        }
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void ds4_kernel_build_extended_kv(\n\
        device const float    *kv_raw         [[buffer(0)]],\n\
        device const float    *kv_comp        [[buffer(1)]],\n\
        device const uint     *comp_selected  [[buffer(2)]],\n\
        device half           *workspace      [[buffer(3)]],\n\
        constant uint         &n_raw          [[buffer(4)]],\n\
        constant uint         &n_selected     [[buffer(5)]],\n\
        constant uint         &head_dim       [[buffer(6)]],\n\
        uint2 gid [[thread_position_in_grid]]) {\n\
    uint row = gid.y;\n\
    uint col = gid.x;\n\
    if (col >= head_dim) return;\n\
    uint n_total = n_raw + n_selected;\n\
    if (row < n_raw) {\n\
        workspace[row * head_dim + col] = (half)kv_raw[row * head_dim + col];\n\
    } else if (row < n_total) {\n\
        uint si = row - n_raw;\n\
        uint comp_idx = comp_selected[si];\n\
        workspace[row * head_dim + col] = (half)kv_comp[comp_idx * head_dim + col];\n\
    } else {\n\
        workspace[row * head_dim + col] = (half)0.0;\n\
    }\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("build_extended_kv library compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_build_extended_kv", None)
            .map_err(|e| anyhow::anyhow!("build_extended_kv function lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("build_extended_kv pipeline build failed: {}", e))?;
        let _ = self.build_extended_kv_pipeline.set(pso);
        Ok(self.build_extended_kv_pipeline.get().unwrap())
    }

    /// M5 task #97 — encode `ds4_kernel_build_extended_kv` into a
    /// caller-provided cb. Gathers `n_raw` rows from `kv_raw` (stride
    /// `head_dim`) followed by `n_selected` rows from `kv_comp` (indexed
    /// by `comp_selected`) into a freshly allocated f16 workspace buffer
    /// of `n_workspace_rows * head_dim` half-precision elements. Returns
    /// the workspace buffer; caller owns commit + wait + read.
    ///
    /// `n_workspace_rows` must be >= `n_raw + n_selected`; the kernel
    /// zero-fills rows in `[n_raw + n_selected, n_workspace_rows)` so
    /// the caller can pad up to the flash_attn kernel's 32-row alignment
    /// (step 3) without leaving uninitialized lanes. Padded rows must
    /// still be masked via the flash_attn mask buffer.
    ///
    /// Output is f16 so it can feed
    /// `flash_attn_decode_metal_encode_with_kv_buf` directly inside the
    /// same cb (the flash_attn kernel reads KV as f16). All input arrays
    /// are uploaded via `new_input_buffer` (StorageMode Shared).
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn build_extended_kv_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv_raw: &[f32],
        kv_comp: &[f32],
        comp_selected: &[u32],
        n_raw: u32,
        n_selected: u32,
        head_dim: u32,
        n_workspace_rows: u32,
    ) -> Result<metal::Buffer> {
        let head_dim_usz = head_dim as usize;
        let n_total = (n_raw as usize) + (n_selected as usize);
        let n_workspace_usz = n_workspace_rows as usize;
        anyhow::ensure!(
            kv_raw.len() >= (n_raw as usize) * head_dim_usz,
            "build_extended_kv: kv_raw too small ({} < {} * {})",
            kv_raw.len(),
            n_raw,
            head_dim_usz
        );
        anyhow::ensure!(
            kv_comp.len() % head_dim_usz == 0,
            "build_extended_kv: kv_comp length ({}) not a multiple of head_dim ({})",
            kv_comp.len(),
            head_dim_usz
        );
        anyhow::ensure!(
            comp_selected.len() >= n_selected as usize,
            "build_extended_kv: comp_selected too short ({} < {})",
            comp_selected.len(),
            n_selected
        );
        anyhow::ensure!(
            n_workspace_usz >= n_total,
            "build_extended_kv: n_workspace_rows ({}) < n_raw + n_selected ({})",
            n_workspace_rows,
            n_total
        );

        let pipeline = self.ensure_build_extended_kv_pipeline()?;

        // Guard empty inputs: new_buffer_with_data with byte_len 0 returns a
        // NULL MTLBuffer (panics on bind). When n_selected==0 (no compressed
        // rows — e.g. early prefill before the comp ring fills), kv_comp and
        // comp_selected are empty; bind a 1-element dummy instead (the kernel
        // never reads comp rows when n_selected==0).
        let kv_raw_buf = new_input_buffer(&self.device, kv_raw);
        let kv_comp_dummy = [0.0f32];
        let kv_comp_buf = new_input_buffer(
            &self.device,
            if kv_comp.is_empty() { &kv_comp_dummy[..] } else { kv_comp },
        );
        let sel_dummy = [0u32];
        let sel_buf = new_input_buffer(
            &self.device,
            if comp_selected.is_empty() { &sel_dummy[..] } else { comp_selected },
        );
        // Workspace holds u16-sized half values (one per f16 element). Reuse
        // this layer's (grow-only) across tokens when the persistent chain set a
        // layer; the gather rewrites all n_workspace rows (incl. zero-padding)
        // every call, so a larger backing buffer is harmless. Else fresh alloc.
        let workspace = {
            let l = self.flash_scratch_layer.load(std::sync::atomic::Ordering::Relaxed);
            if l >= 0 {
                self.flash_scratch_or_alloc(
                    l as u32,
                    1,
                    n_workspace_usz * head_dim_usz * std::mem::size_of::<u16>(),
                )
            } else {
                new_output_buffer::<u16>(&self.device, n_workspace_usz * head_dim_usz)
            }
        };

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&kv_raw_buf), 0);
        enc.set_buffer(1, Some(&kv_comp_buf), 0);
        enc.set_buffer(2, Some(&sel_buf), 0);
        enc.set_buffer(3, Some(&workspace), 0);
        set_scalar_bytes(enc, 4, &n_raw);
        set_scalar_bytes(enc, 5, &n_selected);
        set_scalar_bytes(enc, 6, &head_dim);

        let tg_w: u64 = 32;
        let groups_x = head_dim_usz.div_ceil(tg_w as usize) as u64;
        let groups_y = n_workspace_usz.max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups_x, groups_y, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);

        Ok(workspace)
    }

    /// Single-cb step 9 — GPU-resident-`kv_comp` variant of
    /// [`Self::build_extended_kv_encode`]: binds the caller-owned
    /// compressed-KV ring buffer (`comp_ring_or_alloc`) for buffer(1) instead
    /// of uploading a `&[f32]`, so the gather reads `comp_rows` straight off
    /// the GPU. `kv_raw`/`comp_selected` stay as before. The ring must hold at
    /// least `max(comp_selected[..n_selected]) + 1` rows of `head_dim` f32.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_extended_kv_encode_gpubuf(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv_raw_buf: &metal::Buffer,
        kv_comp_buf: &metal::Buffer,
        comp_selected: &[u32],
        n_raw: u32,
        n_selected: u32,
        head_dim: u32,
        n_workspace_rows: u32,
    ) -> Result<metal::Buffer> {
        let head_dim_usz = head_dim as usize;
        let n_total = (n_raw as usize) + (n_selected as usize);
        let n_workspace_usz = n_workspace_rows as usize;
        anyhow::ensure!(
            comp_selected.len() >= n_selected as usize,
            "build_extended_kv_gpubuf: comp_selected too short ({} < {})",
            comp_selected.len(),
            n_selected
        );
        anyhow::ensure!(
            n_workspace_usz >= n_total,
            "build_extended_kv_gpubuf: n_workspace_rows ({}) < n_raw + n_selected ({})",
            n_workspace_rows,
            n_total
        );

        let pipeline = self.ensure_build_extended_kv_pipeline()?;

        // Bind the persistent KV buffer DIRECTLY (not a CPU snapshot): when
        // this gather shares a cb with the kv_fp8_store_persistent that wrote
        // this token's slot (merged single-cb bridge path), Metal hazard
        // tracking orders the store before this read. A buf.contents() CPU
        // read at encode time would miss the not-yet-executed GPU write.
        // Guard empty comp_selected (n_selected==0, e.g. early prefill before
        // the comp ring fills): new_buffer_with_data on a 0-byte slice returns a
        // NULL MTLBuffer that panics on bind. Bind a 1-elem dummy; the kernel
        // reads no comp rows when n_selected==0.
        let sel_dummy = [0u32];
        let sel_buf = new_input_buffer(
            &self.device,
            if comp_selected.is_empty() { &sel_dummy[..] } else { comp_selected },
        );
        // Reuse this layer's grow-only workspace across tokens (gpuring path).
        let workspace = {
            let l = self.flash_scratch_layer.load(std::sync::atomic::Ordering::Relaxed);
            if l >= 0 {
                self.flash_scratch_or_alloc(
                    l as u32,
                    1,
                    n_workspace_usz * head_dim_usz * std::mem::size_of::<u16>(),
                )
            } else {
                new_output_buffer::<u16>(&self.device, n_workspace_usz * head_dim_usz)
            }
        };

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(kv_raw_buf), 0);
        enc.set_buffer(1, Some(kv_comp_buf), 0);
        enc.set_buffer(2, Some(&sel_buf), 0);
        enc.set_buffer(3, Some(&workspace), 0);
        set_scalar_bytes(enc, 4, &n_raw);
        set_scalar_bytes(enc, 5, &n_selected);
        set_scalar_bytes(enc, 6, &head_dim);

        let tg_w: u64 = 32;
        let groups_x = head_dim_usz.div_ceil(tg_w as usize) as u64;
        let groups_y = n_workspace_usz.max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups_x, groups_y, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);

        Ok(workspace)
    }

    /// Fully-resident variant of [`Self::build_extended_kv_encode_gpubuf`]:
    /// the indexer top-k selection lives in a GPU buffer (`sel_buf`, i32 row
    /// indices, produced by `encode_indexer_topk` in the SAME cb) instead of a
    /// CPU `&[u32]`. Binds `sel_buf` directly at buffer(2) — no readback, no
    /// re-upload — so the long-context bridge stays chained with zero drains.
    /// `sel_buf` must hold at least `n_selected` i32 entries; each must index a
    /// valid row of the comp ring.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_extended_kv_encode_gpubuf_sel(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv_raw_buf: &metal::Buffer,
        kv_comp_buf: &metal::Buffer,
        sel_buf: &metal::Buffer,
        n_raw: u32,
        n_selected: u32,
        head_dim: u32,
        n_workspace_rows: u32,
    ) -> Result<metal::Buffer> {
        let head_dim_usz = head_dim as usize;
        let n_total = (n_raw as usize) + (n_selected as usize);
        let n_workspace_usz = n_workspace_rows as usize;
        anyhow::ensure!(
            n_workspace_usz >= n_total,
            "build_extended_kv_gpubuf_sel: n_workspace_rows ({}) < n_raw + n_selected ({})",
            n_workspace_rows,
            n_total
        );

        let pipeline = self.ensure_build_extended_kv_pipeline()?;
        let workspace = {
            let l = self.flash_scratch_layer.load(std::sync::atomic::Ordering::Relaxed);
            if l >= 0 {
                self.flash_scratch_or_alloc(
                    l as u32,
                    1,
                    n_workspace_usz * head_dim_usz * std::mem::size_of::<u16>(),
                )
            } else {
                new_output_buffer::<u16>(&self.device, n_workspace_usz * head_dim_usz)
            }
        };

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(kv_raw_buf), 0);
        enc.set_buffer(1, Some(kv_comp_buf), 0);
        enc.set_buffer(2, Some(sel_buf), 0);
        enc.set_buffer(3, Some(&workspace), 0);
        set_scalar_bytes(enc, 4, &n_raw);
        set_scalar_bytes(enc, 5, &n_selected);
        set_scalar_bytes(enc, 6, &head_dim);

        let tg_w: u64 = 32;
        let groups_x = head_dim_usz.div_ceil(tg_w as usize) as u64;
        let groups_y = n_workspace_usz.max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups_x, groups_y, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);

        Ok(workspace)
    }

    /// M5 Phase D — build (or fetch cached) the
    /// `ds4_kernel_dsv4_compressor_store_one` pipeline. Port of antirez's
    /// kernel in `dsv4_kv.metal:201-227` — per-token APE add + state
    /// write for the compressor frontier.
    ///
    /// Inline-MSL so the kernel is independent of the
    /// emitted/bridge libraries (same pattern as
    /// `ensure_touch_u8_stride_pipeline` and
    /// `ensure_build_extended_kv_pipeline`). Identical algorithm and
    /// buffer ABI to the antirez kernel so the dispatch + correctness
    /// match upstream.
    pub(crate) fn ensure_compressor_store_one_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.compressor_store_one_pipeline.get() {
            return Ok(p);
        }
        // Verbatim from `dsv4_kv.metal:201-227`, renamed to a `ds4_`
        // symbol so it doesn't collide with the antirez-bridge library's
        // identically-named kernel. Same args struct layout, same buffer
        // bindings (kv, score, ape, state_kv, state_score, args), same
        // gid → element mapping.
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ds4_compressor_store_one_args {\n\
    uint width;\n\
    uint ratio;\n\
    uint pos;\n\
    uint ape_type;\n\
};\n\
kernel void ds4_kernel_dsv4_compressor_store_one(\n\
        constant ds4_compressor_store_one_args & args [[buffer(0)]],\n\
        device const float * kv          [[buffer(1)]],\n\
        device const float * score       [[buffer(2)]],\n\
        device const char  * ape         [[buffer(3)]],\n\
        device       float * state_kv    [[buffer(4)]],\n\
        device       float * state_score [[buffer(5)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {\n\
        return;\n\
    }\n\
    uint pos_mod = args.pos % args.ratio;\n\
    uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;\n\
    uint dst = dst_row * args.width + gid;\n\
    uint ape_i = pos_mod * args.width + gid;\n\
    float ape_v;\n\
    if (args.ape_type == 1u) {\n\
        ape_v = (float)(((device const half *)ape)[ape_i]);\n\
    } else {\n\
        ape_v = ((device const float *)ape)[ape_i];\n\
    }\n\
    state_kv[dst] = kv[gid];\n\
    state_score[dst] = score[gid] + ape_v;\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| {
                anyhow::anyhow!("compressor_store_one library compile failed: {}", e)
            })?;
        let func = lib
            .get_function("ds4_kernel_dsv4_compressor_store_one", None)
            .map_err(|e| {
                anyhow::anyhow!("compressor_store_one function lookup failed: {}", e)
            })?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                anyhow::anyhow!("compressor_store_one pipeline build failed: {}", e)
            })?;
        let _ = self.compressor_store_one_pipeline.set(pso);
        Ok(self.compressor_store_one_pipeline.get().unwrap())
    }

    /// M5 Phase D — encode `ds4_kernel_dsv4_compressor_store_one` into a
    /// caller-provided cb. Writes one row of the precomputed `kv` +
    /// `score` projections into the layer's compressor state buffers at
    /// slot `dst_row` (derived from `pos % ratio`), adding the position-
    /// dependent APE value to `score` along the way.
    ///
    /// `ape_is_f16`: when true, the `ape` slice is reinterpreted as
    /// f16 (`half`) values; when false, as f32. Mirrors the kernel's
    /// `ape_type` switch. The slice's `len()` must be at least
    /// `ratio * width` half-precision or f32 elements respectively.
    ///
    /// State buffers (`state_kv_buf`, `state_score_buf`) are caller-owned
    /// MTLBuffers — typically the persistent per-layer compressor state
    /// buffers (sized 2*ratio*width floats for ratio==4, ratio*width
    /// otherwise). The kernel writes the dst slot in-place; no commit,
    /// no wait, no readback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_store_one_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv: &[f32],
        score: &[f32],
        ape_bytes: &[u8],
        state_kv_buf: &metal::Buffer,
        state_score_buf: &metal::Buffer,
        width: u32,
        ratio: u32,
        pos: u32,
        ape_is_f16: bool,
    ) -> Result<()> {
        anyhow::ensure!(width > 0, "compressor_store_one: width must be > 0");
        anyhow::ensure!(ratio > 0, "compressor_store_one: ratio must be > 0");
        let w = width as usize;
        anyhow::ensure!(
            kv.len() >= w,
            "compressor_store_one: kv.len ({}) < width ({})",
            kv.len(),
            w
        );
        anyhow::ensure!(
            score.len() >= w,
            "compressor_store_one: score.len ({}) < width ({})",
            score.len(),
            w
        );
        let ape_elem_bytes = if ape_is_f16 { 2 } else { 4 };
        let ape_needed = ratio as usize * w * ape_elem_bytes;
        anyhow::ensure!(
            ape_bytes.len() >= ape_needed,
            "compressor_store_one: ape_bytes.len ({}) < ratio*width*{} ({})",
            ape_bytes.len(),
            ape_elem_bytes,
            ape_needed
        );

        let pipeline = self.ensure_compressor_store_one_pipeline()?;

        // Uniforms: 4×u32. Match the MSL struct layout exactly.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            width: u32,
            ratio: u32,
            pos: u32,
            ape_type: u32,
        }
        let args = Args {
            width,
            ratio,
            pos,
            ape_type: if ape_is_f16 { 1 } else { 0 },
        };

        let kv_buf = new_input_buffer(&self.device, kv);
        let score_buf = new_input_buffer(&self.device, score);
        let ape_buf = self.device.new_buffer_with_data(
            ape_bytes.as_ptr() as *const _,
            ape_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&kv_buf), 0);
        enc.set_buffer(2, Some(&score_buf), 0);
        enc.set_buffer(3, Some(&ape_buf), 0);
        enc.set_buffer(4, Some(state_kv_buf), 0);
        enc.set_buffer(5, Some(state_score_buf), 0);

        // 1 thread per element. Threadgroup width 32; grid = ceil(w/32).
        let tg_w: u64 = 32;
        let groups = (w as u64).div_ceil(tg_w);
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);
        Ok(())
    }

    /// M5 scope-merge — DeferredBuf-fed variant of
    /// `compressor_store_one_encode`: kv/score are already GPU-resident
    /// (matvec outputs), so bind them directly instead of uploading.
    /// Width ≥ buf elem counts; ape uploaded fresh (small).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_store_one_db_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv_buf: &metal::Buffer,
        score_buf: &metal::Buffer,
        ape_bytes: &[u8],
        state_kv_buf: &metal::Buffer,
        state_score_buf: &metal::Buffer,
        width: u32,
        ratio: u32,
        pos: u32,
        ape_is_f16: bool,
    ) -> Result<()> {
        anyhow::ensure!(width > 0 && ratio > 0, "store_one_db: width/ratio > 0");
        let pipeline = self.ensure_compressor_store_one_pipeline()?;
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args { width: u32, ratio: u32, pos: u32, ape_type: u32 }
        let args = Args { width, ratio, pos, ape_type: if ape_is_f16 { 1 } else { 0 } };
        let ape_buf = self.device.new_buffer_with_data(
            ape_bytes.as_ptr() as *const _,
            ape_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(kv_buf), 0);
        enc.set_buffer(2, Some(score_buf), 0);
        enc.set_buffer(3, Some(&ape_buf), 0);
        enc.set_buffer(4, Some(state_kv_buf), 0);
        enc.set_buffer(5, Some(state_score_buf), 0);
        let tg_w: u64 = 32;
        let groups = (width as u64).div_ceil(tg_w);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg_w, 1, 1));
        end_shared_compute_enc(enc);
        Ok(())
    }

    /// M5 Phase D — build (or fetch cached) the `ds4_kernel_dsv4_softmax_pool`
    /// pipeline. Port of antirez's `kernel_dsv4_softmax_pool` (dsv4_misc
    /// .metal:1012-1043) verbatim.
    ///
    /// One thread per output element. For each (id, ic) in [ne0]×[ne1]:
    ///   - max_s = max over ir in [ne00] of score[ir,id,ic]
    ///   - sum,acc = Σ exp(score-max_s), Σ kv*exp(...)
    ///   - dst[id,ic] = acc / sum
    pub(crate) fn ensure_softmax_pool_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.softmax_pool_pipeline.get() {
            return Ok(p);
        }
        // Args struct + kernel body verbatim from `dsv4_misc.metal:
        // 30-44, 1012-1043`. Renamed to a `ds4_` symbol so it can't
        // collide with the antirez-bridge library's identically-named
        // kernel.
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ds4_softmax_pool_args {\n\
    long  ne00;\n\
    long  ne01;\n\
    long  ne02;\n\
    ulong nb00;\n\
    ulong nb01;\n\
    ulong nb02;\n\
    ulong nb10;\n\
    ulong nb11;\n\
    ulong nb12;\n\
    long  ne0;\n\
    long  ne1;\n\
    ulong nb0;\n\
    ulong nb1;\n\
};\n\
kernel void ds4_kernel_dsv4_softmax_pool(\n\
        constant ds4_softmax_pool_args & args [[buffer(0)]],\n\
        device const char * kv    [[buffer(1)]],\n\
        device const char * score [[buffer(2)]],\n\
        device       char * dst   [[buffer(3)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    long n = args.ne0 * args.ne1;\n\
    if ((long)gid >= n) {\n\
        return;\n\
    }\n\
    long id = (long)gid % args.ne0;\n\
    long ic = (long)gid / args.ne0;\n\
    float max_s = -INFINITY;\n\
    for (long ir = 0; ir < args.ne00; ++ir) {\n\
        float s = *((device const float *)(score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));\n\
        max_s = max(max_s, s);\n\
    }\n\
    float sum = 0.0f;\n\
    float acc = 0.0f;\n\
    for (long ir = 0; ir < args.ne00; ++ir) {\n\
        float s = *((device const float *)(score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));\n\
        float w = exp(s - max_s);\n\
        float v = *((device const float *)(kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));\n\
        sum += w;\n\
        acc += v * w;\n\
    }\n\
    *((device float *)(dst + id*args.nb0 + ic*args.nb1)) = acc / sum;\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("softmax_pool library compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_dsv4_softmax_pool", None)
            .map_err(|e| anyhow::anyhow!("softmax_pool function lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("softmax_pool pipeline build failed: {}", e))?;
        let _ = self.softmax_pool_pipeline.set(pso);
        Ok(self.softmax_pool_pipeline.get().unwrap())
    }

    /// M5 Phase D — encode `ds4_kernel_dsv4_softmax_pool` into a
    /// caller-provided cb. Reads `n_rows` rows of KV and score from
    /// row-major `[n_rows × width]` buffers, computes the
    /// max-stabilized softmax of score along the row dim, and writes
    /// the weighted sum of KV into a fresh `width`-long output buffer
    /// (returned).
    ///
    /// `kv_buf` and `score_buf` are caller-owned MTLBuffers (typically
    /// the persistent per-layer compressor state pools from
    /// `compressor_state_*_or_alloc`). The kernel assumes row-major
    /// layout with element stride `sizeof(f32)`; row stride is
    /// `width * sizeof(f32)` (computed from `width`).
    ///
    /// No commit, no wait, no readback.
    pub(crate) fn softmax_pool_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv_buf: &metal::Buffer,
        score_buf: &metal::Buffer,
        n_rows: u32,
        width: u32,
    ) -> Result<metal::Buffer> {
        anyhow::ensure!(n_rows > 0, "softmax_pool: n_rows must be > 0");
        anyhow::ensure!(width > 0, "softmax_pool: width must be > 0");
        let w = width as usize;
        let pipeline = self.ensure_softmax_pool_pipeline()?;

        let elem = std::mem::size_of::<f32>() as u64;
        let row_bytes = (w as u64) * elem;
        // Row-major [n_rows × width] layout, single ic.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            ne00: i64,
            ne01: i64,
            ne02: i64,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            ne0: i64,
            ne1: i64,
            nb0: u64,
            nb1: u64,
        }
        let args = Args {
            ne00: n_rows as i64,
            ne01: 1,
            ne02: 1,
            // kv strides: ir × row, id × elem, ic × 0
            nb00: row_bytes,
            nb01: elem,
            nb02: 0,
            // score strides: same layout as kv
            nb10: row_bytes,
            nb11: elem,
            nb12: 0,
            // output shape + strides: [width × 1], dst[id] = …
            ne0: w as i64,
            ne1: 1,
            nb0: elem,
            nb1: 0,
        };

        let out_buf = new_output_buffer::<f32>(&self.device, w);

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(kv_buf), 0);
        enc.set_buffer(2, Some(score_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);

        // One thread per output element.
        let tg_w: u64 = 32;
        let groups = (w as u64).div_ceil(tg_w);
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);
        Ok(out_buf)
    }

    /// M5 Phase D — build (or fetch cached) the
    /// `ds4_kernel_dsv4_compressor_pool_ratio4` pipeline. Bespoke
    /// kernel that mirrors `compressor_pool_decode_state`
    /// (`attn_dispatch.rs:1985`) ratio==4 branch exactly: per output
    /// column `j` in `[0..head_dim]`, scan the 2-window state
    /// (`[2*ratio, 2*head_dim]` row-major) reading `state[r, j]` from
    /// the prev-window rows and `state[ratio+r, head_dim+j]` from the
    /// current-window rows. Max-stabilized softmax over `2*ratio`
    /// scores; weighted sum over `2*ratio` KV values; preserves the
    /// `-1e9` masked-slot sentinel (out=0 when no valid slot).
    ///
    /// Inline-MSL, same pattern as `ensure_softmax_pool_pipeline`.
    pub(crate) fn ensure_compressor_pool_ratio4_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.compressor_pool_ratio4_pipeline.get() {
            return Ok(p);
        }
        // Mirrors `compressor_pool_decode_state` (ratio==4 branch) at
        // `attn_dispatch.rs:1985`. Layout assumption:
        //   - state_kv, state_score: row-major `[2*ratio, 2*head_dim]`
        //   - width = 2 * head_dim
        //   - prev-window rows in `[0..ratio]`, cols `[0..head_dim]`
        //   - current-window rows in `[ratio..2*ratio]`, cols `[head_dim..2*head_dim]`
        //   - dst: `head_dim` floats (one per thread).
        // The -1e9 sentinel comes from `decode_step.rs:493` initializing
        // the CPU state buffer; on GPU we rely on the caller initializing
        // state buffers to -1e9 on first alloc (see compressor_state_*_or_alloc).
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ds4_compressor_pool_ratio4_args {\n\
    uint head_dim;\n\
    uint ratio;\n\
    uint width;\n\
};\n\
kernel void ds4_kernel_dsv4_compressor_pool_ratio4(\n\
        constant ds4_compressor_pool_ratio4_args & args [[buffer(0)]],\n\
        device const float * state_kv    [[buffer(1)]],\n\
        device const float * state_score [[buffer(2)]],\n\
        device       float * dst         [[buffer(3)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    uint j = gid;\n\
    if (j >= args.head_dim) {\n\
        return;\n\
    }\n\
    uint hd    = args.head_dim;\n\
    uint ratio = args.ratio;\n\
    uint width = args.width;\n\
    float max_score = -1.0e9f;\n\
    for (uint r = 0; r < ratio; ++r) {\n\
        float sp = state_score[r * width + j];\n\
        float sc = state_score[(ratio + r) * width + hd + j];\n\
        max_score = max(max_score, sp);\n\
        max_score = max(max_score, sc);\n\
    }\n\
    if (max_score <= -5.0e8f) {\n\
        dst[j] = 0.0f;\n\
        return;\n\
    }\n\
    float denom = 0.0f;\n\
    float sum   = 0.0f;\n\
    for (uint r = 0; r < ratio; ++r) {\n\
        float wp = exp(state_score[r * width + j]                   - max_score);\n\
        float wc = exp(state_score[(ratio + r) * width + hd + j]    - max_score);\n\
        denom += wp + wc;\n\
        sum   += wp * state_kv[r * width + j];\n\
        sum   += wc * state_kv[(ratio + r) * width + hd + j];\n\
    }\n\
    dst[j] = denom > 0.0f ? sum / denom : 0.0f;\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| {
                anyhow::anyhow!("compressor_pool_ratio4 library compile failed: {}", e)
            })?;
        let func = lib
            .get_function("ds4_kernel_dsv4_compressor_pool_ratio4", None)
            .map_err(|e| {
                anyhow::anyhow!("compressor_pool_ratio4 function lookup failed: {}", e)
            })?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                anyhow::anyhow!("compressor_pool_ratio4 pipeline build failed: {}", e)
            })?;
        let _ = self.compressor_pool_ratio4_pipeline.set(pso);
        Ok(self.compressor_pool_ratio4_pipeline.get().unwrap())
    }

    /// M5 Phase D — encode `ds4_kernel_dsv4_compressor_pool_ratio4`
    /// into a caller-provided cb. Pools the persistent
    /// `[2*ratio, 2*head_dim]` state into a fresh `head_dim`-long
    /// output buffer (returned) using the cross-column two-window
    /// read pattern from `compressor_pool_decode_state`.
    ///
    /// `state_kv_buf` and `state_score_buf` are caller-owned
    /// MTLBuffers, typically the persistent per-layer compressor
    /// state pools from `compressor_state_*_or_alloc`. Each must be at
    /// least `2*ratio * 2*head_dim * sizeof(f32)` bytes.
    ///
    /// No commit, no wait, no readback.
    pub(crate) fn compressor_pool_ratio4_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        state_kv_buf: &metal::Buffer,
        state_score_buf: &metal::Buffer,
        head_dim: u32,
    ) -> Result<metal::Buffer> {
        anyhow::ensure!(head_dim > 0, "compressor_pool_ratio4: head_dim must be > 0");
        let hd = head_dim as usize;
        let pipeline = self.ensure_compressor_pool_ratio4_pipeline()?;

        // The kernel hard-codes the 2-window layout: ratio is 4 and
        // width is 2*head_dim. The CPU oracle only takes this branch
        // for compress_ratio == 4 (see `attn_dispatch.rs:1992`).
        let ratio: u32 = 4;
        let width: u32 = 2 * head_dim;

        // Sanity-check the buffer sizes match the layout assumption.
        let elem = std::mem::size_of::<f32>() as u64;
        let need_bytes = (2 * ratio as u64) * (width as u64) * elem;
        anyhow::ensure!(
            state_kv_buf.length() >= need_bytes,
            "compressor_pool_ratio4: state_kv buf too small ({} < {})",
            state_kv_buf.length(),
            need_bytes
        );
        anyhow::ensure!(
            state_score_buf.length() >= need_bytes,
            "compressor_pool_ratio4: state_score buf too small ({} < {})",
            state_score_buf.length(),
            need_bytes
        );

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            head_dim: u32,
            ratio: u32,
            width: u32,
        }
        let args = Args {
            head_dim,
            ratio,
            width,
        };

        let out_buf = new_output_buffer::<f32>(&self.device, hd);

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(state_kv_buf), 0);
        enc.set_buffer(2, Some(state_score_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);

        // One thread per output column j in [0..head_dim].
        let tg_w: u64 = 32;
        let groups = (hd as u64).div_ceil(tg_w);
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        end_shared_compute_enc(enc);
        Ok(out_buf)
    }

    /// DS4_CHUNK_ATTN_NOSYNC — build the GPU ratio==4 pool-rotation pipeline.
    /// Replicates the net effect of the CPU `finish_emit` rotation
    /// (compressor.rs ~598-613): `front_row[k][0..width] := back_row[k][0..width]`
    /// for k in [0..ratio]. The CPU code's two `copy_within` loops collapse to
    /// this single forward copy (the second loop writes the just-overwritten
    /// front back into the back window — a no-op). Operates in-place on the
    /// persistent `[2*ratio, width]` pool, so the next quad's `pool_ratio4`
    /// reads the rotated prefix window without any CPU round-trip.
    pub(crate) fn ensure_compressor_rotate_ratio4_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.compressor_rotate_ratio4_pipeline.get() {
            return Ok(p);
        }
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ds4_compressor_rotate_ratio4_args {\n\
    uint ratio;\n\
    uint width;\n\
};\n\
kernel void ds4_kernel_dsv4_compressor_rotate_ratio4(\n\
        constant ds4_compressor_rotate_ratio4_args & args [[buffer(0)]],\n\
        device float * state_kv    [[buffer(1)]],\n\
        device float * state_score [[buffer(2)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    uint total = args.ratio * args.width;\n\
    if (gid >= total) { return; }\n\
    uint k = gid / args.width;\n\
    uint j = gid % args.width;\n\
    uint front = k * args.width + j;\n\
    uint back  = (args.ratio + k) * args.width + j;\n\
    state_kv[front]    = state_kv[back];\n\
    state_score[front] = state_score[back];\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("compressor_rotate_ratio4 compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_dsv4_compressor_rotate_ratio4", None)
            .map_err(|e| anyhow::anyhow!("compressor_rotate_ratio4 lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("compressor_rotate_ratio4 build failed: {}", e))?;
        let _ = self.compressor_rotate_ratio4_pipeline.set(pso);
        Ok(self.compressor_rotate_ratio4_pipeline.get().unwrap())
    }

    /// DS4_CHUNK_ATTN_NOSYNC — encode the GPU ratio==4 pool rotation into a
    /// caller-provided cb (no commit, no wait, no readback). `width = 2*head_dim`.
    /// Rotates the prefix window of both the kv and score pools in place.
    pub(crate) fn compressor_rotate_ratio4_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        state_kv_buf: &metal::Buffer,
        state_score_buf: &metal::Buffer,
        head_dim: u32,
    ) -> Result<()> {
        anyhow::ensure!(head_dim > 0, "compressor_rotate_ratio4: head_dim must be > 0");
        let pipeline = self.ensure_compressor_rotate_ratio4_pipeline()?;
        let ratio: u32 = 4;
        let width: u32 = 2 * head_dim;
        let elem = std::mem::size_of::<f32>() as u64;
        let need_bytes = (2 * ratio as u64) * (width as u64) * elem;
        anyhow::ensure!(
            state_kv_buf.length() >= need_bytes && state_score_buf.length() >= need_bytes,
            "compressor_rotate_ratio4: state buf too small"
        );
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            ratio: u32,
            width: u32,
        }
        let args = Args { ratio, width };
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(state_kv_buf), 0);
        enc.set_buffer(2, Some(state_score_buf), 0);
        let total = (ratio * width) as u64;
        let tg_w: u64 = 64;
        let groups = total.div_ceil(tg_w);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg_w, 1, 1));
        end_shared_compute_enc(enc);
        Ok(())
    }

    /// Inline pipeline for the grouped-output-proj interleave (group-major →
    /// K-major). out[k*(ng*nlo) + g*nlo + j] = src[g*(K*nlo) + k*nlo + j].
    pub(crate) fn ensure_interleave_group_major_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.interleave_group_major_pipeline.get() {
            return Ok(p);
        }
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void ds4_kernel_interleave_group_major(\n\
        device const float * src [[buffer(0)]],\n\
        device       float * dst [[buffer(1)]],\n\
        constant uint & n_groups [[buffer(2)]],\n\
        constant uint & k_pos    [[buffer(3)]],\n\
        constant uint & nlo      [[buffer(4)]],\n\
        constant uint & total    [[buffer(5)]],\n\
        uint gid [[thread_position_in_grid]]) {\n\
    if (gid >= total) return;\n\
    uint j = gid % nlo;\n\
    uint rem = gid / nlo;\n\
    uint g = rem % n_groups;\n\
    uint k = rem / n_groups;\n\
    dst[gid] = src[(g * k_pos + k) * nlo + j];\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("interleave_group_major compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_interleave_group_major", None)
            .map_err(|e| anyhow::anyhow!("interleave_group_major lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("interleave_group_major build failed: {}", e))?;
        let _ = self.interleave_group_major_pipeline.set(pso);
        Ok(self.interleave_group_major_pipeline.get().unwrap())
    }

    /// Build the fused ratio==4 chunk-prefill compressor pipeline (inline source,
    /// like the rotate kernel). The two-window analog of
    /// `ds4_kernel_dsv4_compressor_prefill_noidx_f32`: per emit row e, pool over
    /// the CURRENT window (positions [e*4 .. e*4+4), 2nd-half columns hd+j of the
    /// `width=2*hd` projection) AND the PREVIOUS window (positions [e*4-4 .. e*4),
    /// 1st-half columns j; zeros when <0, matching the fresh-sequence init), with
    /// the per-member APE bias (`ape[col*ratio + r]`); softmax-weighted column
    /// pool → RMS-norm × `norm` → partial RoPE at comp position `pos0 + e*4` →
    /// write to comp[(comp_row0 + e)*hd]. Bit-equivalent to the per-position
    /// compressor_store_one → compressor_pool_ratio4 → rms_norm_mul →
    /// rope_tail_in_place chain when chunk_start % 4 == 0.
    pub(crate) fn ensure_compressor_prefill_idx_pipeline(
        &self,
    ) -> Result<&metal::ComputePipelineState> {
        if let Some(p) = self.compressor_prefill_idx_pipeline.get() {
            return Ok(p);
        }
        let src = "\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ds4_compressor_prefill_args {\n\
    uint head_dim; uint ratio; uint n_rot; uint pos0;\n\
    uint comp_row0; uint n_comp; float rms_eps;\n\
    uint freq_base_b; uint freq_scale_b; uint ext_factor_b; uint attn_factor_b;\n\
    uint beta_fast_b; uint beta_slow_b; uint orig_ctx; uint backward;\n\
};\n\
static inline float yarn_ramp_mask_cp(float low, float high, int i0) {\n\
    float y = (float(i0) - low) / max(0.001f, high - low);\n\
    return 1.0f - clamp(y, 0.0f, 1.0f);\n\
}\n\
static inline float yarn_get_mscale_cp(float scale, float mscale) {\n\
    if (scale <= 1.0f) return 1.0f;\n\
    return 0.1f * mscale * log(scale) + 1.0f;\n\
}\n\
kernel void ds4_kernel_dsv4_compressor_prefill_idx_f32(\n\
    device const float * kv   [[buffer(0)]],\n\
    device const float * sc   [[buffer(1)]],\n\
    device const float * ape  [[buffer(2)]],\n\
    device const float * norm [[buffer(3)]],\n\
    device       float * comp [[buffer(4)]],\n\
    constant ds4_compressor_prefill_args & args [[buffer(5)]],\n\
    uint e [[threadgroup_position_in_grid]],\n\
    uint tid [[thread_position_in_threadgroup]],\n\
    uint tg_size [[threads_per_threadgroup]]) {\n\
    const uint hd = args.head_dim;\n\
    const uint ratio = args.ratio;\n\
    if (e >= args.n_comp) return;\n\
    threadgroup float pooled_sh[1024];\n\
    threadgroup float red[32];\n\
    const uint W = 2u * hd;\n\
    const uint base_row = e * ratio;\n\
    for (uint j = tid; j < hd; j += tg_size) {\n\
        float max_score = -1.0e30f;\n\
        for (uint r = 0; r < ratio; ++r) {\n\
            uint cur = base_row + r;\n\
            float scc = sc[cur * W + hd + j] + ape[(hd + j) * ratio + r];\n\
            max_score = max(max_score, scc);\n\
            int prev = int(base_row) - int(ratio) + int(r);\n\
            if (prev >= 0) {\n\
                float scp = sc[uint(prev) * W + j] + ape[j * ratio + r];\n\
                max_score = max(max_score, scp);\n\
            } else {\n\
                max_score = max(max_score, 0.0f);\n\
            }\n\
        }\n\
        float denom = 0.0f; float sum = 0.0f;\n\
        for (uint r = 0; r < ratio; ++r) {\n\
            uint cur = base_row + r;\n\
            float scc = sc[cur * W + hd + j] + ape[(hd + j) * ratio + r];\n\
            float wc = exp(scc - max_score);\n\
            denom += wc; sum += wc * kv[cur * W + hd + j];\n\
            int prev = int(base_row) - int(ratio) + int(r);\n\
            if (prev >= 0) {\n\
                float scp = sc[uint(prev) * W + j] + ape[j * ratio + r];\n\
                float wp = exp(scp - max_score);\n\
                denom += wp; sum += wp * kv[uint(prev) * W + j];\n\
            } else {\n\
                denom += exp(0.0f - max_score);\n\
            }\n\
        }\n\
        pooled_sh[j] = denom > 0.0f ? sum / denom : 0.0f;\n\
    }\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    float local_ss = 0.0f;\n\
    for (uint j = tid; j < hd; j += tg_size) { float v = pooled_sh[j]; local_ss += v * v; }\n\
    local_ss = simd_sum(local_ss);\n\
    uint sg_id = tid / 32u; uint lane = tid % 32u;\n\
    if (lane == 0u) red[sg_id] = local_ss;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    float ss = 0.0f; uint n_sg = (tg_size + 31u) / 32u;\n\
    for (uint g = 0; g < n_sg; ++g) ss += red[g];\n\
    float rms = 1.0f / sqrt(ss / float(hd) + args.rms_eps);\n\
    for (uint j = tid; j < hd; j += tg_size) { pooled_sh[j] = pooled_sh[j] * rms * norm[j]; }\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    const uint n_rot = args.n_rot;\n\
    if (n_rot > 0u) {\n\
        float freq_base = as_type<float>(args.freq_base_b);\n\
        float freq_scale = as_type<float>(args.freq_scale_b);\n\
        float ext_factor = as_type<float>(args.ext_factor_b);\n\
        float attn_factor = as_type<float>(args.attn_factor_b);\n\
        float beta_fast = as_type<float>(args.beta_fast_b);\n\
        float beta_slow = as_type<float>(args.beta_slow_b);\n\
        int pos = int(args.pos0 + base_row);\n\
        int backward = int(args.backward);\n\
        const uint tail0 = hd - n_rot;\n\
        float low = floor(log(float(args.orig_ctx) / (2.0f * M_PI_F * beta_fast)) / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;\n\
        float high = ceil(log(float(args.orig_ctx) / (2.0f * M_PI_F * beta_slow)) / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;\n\
        float mscale = yarn_get_mscale_cp(freq_scale, attn_factor);\n\
        for (uint pair = tid; pair * 2u < n_rot; pair += tg_size) {\n\
            uint i0 = pair * 2u;\n\
            float exponent = float(i0) / float(n_rot);\n\
            float freq_full = 1.0f / pow(freq_base, exponent);\n\
            float freq_inter = freq_full / freq_scale;\n\
            float ramp = yarn_ramp_mask_cp(low, high, int(i0));\n\
            float freq = freq_inter * (1.0f - ramp * ext_factor) + freq_full * ramp * ext_factor;\n\
            float theta = float(pos) * freq;\n\
            float cos_t = cos(theta) * mscale;\n\
            float sin_t = sin(theta) * mscale;\n\
            if (backward) sin_t = -sin_t;\n\
            float x0 = pooled_sh[tail0 + i0];\n\
            float x1 = pooled_sh[tail0 + i0 + 1u];\n\
            pooled_sh[tail0 + i0] = x0 * cos_t - x1 * sin_t;\n\
            pooled_sh[tail0 + i0 + 1u] = x0 * sin_t + x1 * cos_t;\n\
        }\n\
        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    }\n\
    device float * out = comp + (args.comp_row0 + e) * hd;\n\
    for (uint j = tid; j < hd; j += tg_size) { out[j] = pooled_sh[j]; }\n\
}\n";
        let options = metal::CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(src, &options)
            .map_err(|e| anyhow::anyhow!("compressor_prefill_idx compile failed: {}", e))?;
        let func = lib
            .get_function("ds4_kernel_dsv4_compressor_prefill_idx_f32", None)
            .map_err(|e| anyhow::anyhow!("compressor_prefill_idx lookup failed: {}", e))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("compressor_prefill_idx build failed: {}", e))?;
        let _ = self.compressor_prefill_idx_pipeline.set(pso);
        Ok(self.compressor_prefill_idx_pipeline.get().unwrap())
    }

    /// Encode the fused ratio==4 compressor for the whole chunk (one dispatch).
    /// `kv`/`sc` are the K-batched `[k_positions × 2*head_dim]` projections;
    /// `ape` is model-layout `[2*head_dim × ratio]` (`ape[col*ratio + r]`);
    /// `norm` is `[head_dim]`. Writes emit rows into `comp_ring` at
    /// `[comp_row0 .. comp_row0+n_comp)`. No commit/wait/readback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_prefill_idx_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        kv: &metal::Buffer,
        sc: &metal::Buffer,
        ape_buf: &metal::Buffer,
        norm_buf: &metal::Buffer,
        comp_ring: &metal::Buffer,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        comp_row0: u32,
        n_comp: u32,
        params: &LayerParams,
        rms_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(head_dim as usize <= 1024, "compressor_prefill_idx: head_dim <= 1024");
        if n_comp == 0 {
            return Ok(());
        }
        let pipe = self.ensure_compressor_prefill_idx_pipeline()?.clone();
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Args {
            head_dim: u32, ratio: u32, n_rot: u32, pos0: u32,
            comp_row0: u32, n_comp: u32, rms_eps: f32,
            freq_base_b: u32, freq_scale_b: u32, ext_factor_b: u32, attn_factor_b: u32,
            beta_fast_b: u32, beta_slow_b: u32, orig_ctx: u32, backward: u32,
        }
        let args = Args {
            head_dim, ratio: 4, n_rot, pos0, comp_row0, n_comp, rms_eps,
            freq_base_b: params.rope_freq_base.to_bits(),
            freq_scale_b: params.rope_freq_scale.to_bits(),
            ext_factor_b: params.rope_ext_factor.to_bits(),
            attn_factor_b: params.rope_attn_factor.to_bits(),
            beta_fast_b: (32.0f32).to_bits(),
            beta_slow_b: (1.0f32).to_bits(),
            orig_ctx: params.rope_orig_ctx,
            backward: 0,
        };
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(kv), 0);
        enc.set_buffer(1, Some(sc), 0);
        enc.set_buffer(2, Some(ape_buf), 0);
        enc.set_buffer(3, Some(norm_buf), 0);
        enc.set_buffer(4, Some(comp_ring), 0);
        set_scalar_bytes(enc, 5, &args);
        let tg = (head_dim as u64).min(1024);
        enc.dispatch_thread_groups(
            MTLSize::new(n_comp as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        end_shared_compute_enc(enc);
        Ok(())
    }

    /// Phase E M5.2: get-or-allocate the persistent KV cache buffer for
    /// `layer_idx`, sized `byte_size`. Re-uses the buffer on every
    /// subsequent call regardless of `byte_size` (debug-asserts the
    /// size doesn't change once allocated).
    ///
    /// Returns a clone of the buffer (Arc-counted Metal foreign type).
    /// Holding the clone keeps the buffer alive even if the pool is
    /// cleared.
    #[allow(dead_code)]
    pub(crate) fn kv_buffer_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        let mut map = self.kv_cache_buffers.lock().expect("kv_cache_buffers mutex");
        let entry = map.entry(layer_idx).or_insert_with(|| {
            self.device.new_buffer(
                byte_size as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });
        debug_assert!(
            entry.length() == byte_size as u64,
            "kv_buffer_or_alloc({layer_idx}): allocated len {} != requested {}",
            entry.length(),
            byte_size
        );
        entry.clone()
    }

    /// Get-or-allocate a reusable per-(layer, role) flash SCRATCH buffer of at
    /// least `min_bytes` (grow-only: reallocates iff the cached one is too
    /// small). Unlike `kv_buffer_or_alloc` this does NOT assert an exact size —
    /// the f16 KV workspace grows monotonically with context, and the flash
    /// kernels read only the active `n_raw`/`n_total` prefix (passed via args),
    /// so a larger backing buffer is harmless. Dropping a grown-out buffer is
    /// safe: any in-flight command buffer retains it until completion.
    pub(crate) fn flash_scratch_or_alloc(
        &self,
        layer: u32,
        role: u8,
        min_bytes: usize,
    ) -> metal::Buffer {
        let mut map = self
            .flash_scratch_buffers
            .lock()
            .expect("flash_scratch_buffers mutex");
        let entry = map.entry((layer, role)).or_insert_with(|| {
            self.device
                .new_buffer(min_bytes as u64, MTLResourceOptions::StorageModeShared)
        });
        if (entry.length() as usize) < min_bytes {
            *entry = self
                .device
                .new_buffer(min_bytes as u64, MTLResourceOptions::StorageModeShared);
        }
        entry.clone()
    }

    /// M5 Phase D — get-or-allocate the persistent per-layer compressor
    /// `state_kv` buffer. Same Arc-clone-on-return contract as
    /// `kv_buffer_or_alloc`. Zero-initialized on first alloc; subsequent
    /// calls return the same buffer (writes from prior dispatches
    /// persist).
    pub(crate) fn compressor_state_kv_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        state_buffer_or_alloc_pin(
            &self.device,
            &self.compressor_state_kv_buffers,
            "compressor_state_kv",
            layer_idx,
            byte_size,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    /// M5 Phase D — get-or-allocate the persistent per-layer compressor
    /// `state_score` buffer. Filled with `-1e9` on first alloc to match
    /// the CPU mirror's softmax-mask sentinel (`decode_step.rs:493`).
    /// See [`Self::compressor_state_kv_or_alloc`] for the kv counterpart.
    pub(crate) fn compressor_state_score_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        state_buffer_or_alloc_filled_f32_pin(
            &self.device,
            &self.compressor_state_score_buffers,
            "compressor_state_score",
            layer_idx,
            byte_size,
            -1.0e9,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    /// M5 Phase D — get-or-allocate the persistent per-layer indexer
    /// `state_kv` buffer. Mirror of compressor_state_kv but for the
    /// indexer's separate compressor on ratio==4 layers.
    pub(crate) fn indexer_state_kv_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        state_buffer_or_alloc_pin(
            &self.device,
            &self.indexer_state_kv_buffers,
            "indexer_state_kv",
            layer_idx,
            byte_size,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    /// M5 Phase D — get-or-allocate the persistent per-layer indexer
    /// `state_score` buffer. Filled with `-1e9` on first alloc to
    /// match the CPU mirror's softmax-mask sentinel
    /// (`decode_step.rs:504`).
    pub(crate) fn indexer_state_score_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        state_buffer_or_alloc_filled_f32_pin(
            &self.device,
            &self.indexer_state_score_buffers,
            "indexer_state_score",
            layer_idx,
            byte_size,
            -1.0e9,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    /// Re-initialise the GPU-resident compressor/indexer rolling-window state
    /// pools to their fresh-sequence values (kv → 0, score → DS4_NEG_INF=-1e9).
    /// These pools persist on the dispatcher across sequences and are filled
    /// only on FIRST alloc (see `state_buffer_or_alloc_filled_f32`), so a new
    /// sequence that reuses the same dispatcher inherits the PRIOR sequence's
    /// window. A stale `score != -1e9` un-masks not-yet-written slots in the
    /// pooled softmax → wrong compression → run-to-run divergence. Call at the
    /// start of every decode sequence (pos==0). Safe: the previous sequence's
    /// GPU work is flushed+waited by then, and this one hasn't dispatched yet.
    pub(crate) fn reset_decode_state_pools(&self) {
        fn fill(pool: &std::sync::Mutex<HashMap<u32, metal::Buffer>>, v: f32) {
            if let Ok(map) = pool.lock() {
                for buf in map.values() {
                    let n = buf.length() as usize / std::mem::size_of::<f32>();
                    unsafe {
                        let p = buf.contents() as *mut f32;
                        for i in 0..n {
                            *p.add(i) = v;
                        }
                    }
                }
            }
        }
        fill(&self.compressor_state_kv_buffers, 0.0);
        fill(&self.compressor_state_score_buffers, -1.0e9);
        fill(&self.indexer_state_kv_buffers, 0.0);
        fill(&self.indexer_state_score_buffers, -1.0e9);
        // Also zero the persistent EMITTED-row rings and the raw KV ring. A fresh
        // sequence starts with n_comp = n_index_comp = kv_pos = 0, so these must be
        // empty — but they live on the dispatcher and survive across sequences. If
        // left stale, a chunk prefill's gather/flash can read a prior sequence's
        // ring rows (the pool reset above only covers the rolling pool window, not
        // the emitted rings), giving run-to-run divergence even on the fully-sync
        // chunk path. Byte-zero is correct here: at a fresh sequence nothing valid
        // has been written yet.
        fn zero(pool: &std::sync::Mutex<HashMap<u32, metal::Buffer>>) {
            if let Ok(map) = pool.lock() {
                for buf in map.values() {
                    unsafe {
                        std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize);
                    }
                }
            }
        }
        zero(&self.comp_ring_buffers);
        zero(&self.index_comp_ring_buffers);
        zero(&self.kv_cache_buffers);
    }

    /// Single-cb step 9 — get-or-allocate the GPU-resident compressed-KV
    /// ring for `layer_idx`. Row-major `[max_rows × head_dim]`; the
    /// compressor's emit row is copied into row `n_comp`, and the flash
    /// gather reads `comp_rows` from here instead of an uploaded CPU slice.
    pub(crate) fn comp_ring_or_alloc(&self, layer_idx: u32, byte_size: usize) -> metal::Buffer {
        state_buffer_or_alloc_pin(
            &self.device,
            &self.comp_ring_buffers,
            "comp_ring",
            layer_idx,
            byte_size,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    pub(crate) fn index_comp_ring_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        state_buffer_or_alloc_pin(
            &self.device,
            &self.index_comp_ring_buffers,
            "index_comp_ring",
            layer_idx,
            byte_size,
            |b| self.pin_state_buffer_resident(b),
        )
    }

    /// Build (or fetch from cache) a `ComputePipelineState` for a
    /// function-constant-specialized kernel. The composer passes a
    /// builder closure that populates `FunctionConstantValues` and a
    /// stable byte-key uniquely identifying that set of constants
    /// (a hash collision would map two different specializations to the
    /// same pipeline, so encode every constant the kernel actually
    /// reads).
    ///
    /// Returns an error if the symbol isn't present in either library.
    /// On Metal compile failure, the error message includes the symbol
    /// and the underlying NSError, so composer code can wrap it with
    /// extra context.
    #[allow(dead_code)]
    pub(crate) fn specialized_pipeline<F>(
        &self,
        symbol: &str,
        constants_key: &[u8],
        populate: F,
    ) -> Result<metal::ComputePipelineState>
    where
        F: Fn(&metal::FunctionConstantValuesRef),
    {
        let cache_key = (symbol.to_string(), constants_key.to_vec());
        if let Ok(guard) = self.specialized.lock() {
            if let Some(pso) = guard.get(&cache_key) {
                return Ok(pso.clone());
            }
        }

        // FCV is consumed by get_function so we re-build it for the
        // fallback path; the `populate` closure is therefore `Fn`
        // (callable repeatedly).
        let make_values = || {
            let v = metal::FunctionConstantValues::new();
            populate(&v);
            v
        };

        // Some symbols exist in BOTH `emitted_lib` and `bridge_lib` with
        // INCOMPATIBLE Metal ABIs — same kernel name, different buffer
        // layouts. For these the emitted version was produced by the
        // codegen with its own ABI (individual `uint` buffers, hardcoded
        // `constexpr` workgroup sizes) while every Rust caller still
        // encodes the antirez-bridge args (struct at buffer 0 + function
        // constants 600/601). Without an override the default
        // emitted-first lookup binds the wrong kernel and the dispatch
        // silently writes nothing (kernel reads `ne00` from `dst_buf`,
        // sees 0, loops zero times). See:
        // `benchmarks/ds4_msl/emitted/mul_mv_f32_f32_4.metal` vs
        // `benchmarks/ds4_msl/upstream/ds4/metal/dense.metal:547`.
        //
        // Until the emitted ABI is unified with the bridge ABI (or the
        // Rust callers are migrated to the emitted ABI), force-bridge
        // these symbols.
        const BRIDGE_PREFERRED: &[&str] = &[
            "ds4_kernel_mul_mv_f32_f32_4",
            // Emitted flash_attn variant hardcodes DK=512/DV=512 and lacks
            // the function constants (has_mask/sinks/bias/softcap/kvpad +
            // ns10/ns20/nsg/nwg) the Rust caller binds; emitted-first
            // lookup produces ~0.5 absolute error in `flash_attn_decode_metal`.
            "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512",
            // Tiled f16 weight-stationary matmul (matmul_k_f16). Exists in BOTH
            // libs: the emitted variant has the codegen ABI (reads ne00 from the
            // dst buffer → sees 0 → writes nothing → all-zeros), while the Rust
            // caller binds the antirez-bridge mm_args struct at buffer 0 + fc
            // 700/701. Force the bridge kernel. (The q8_0 twin needs no override
            // because the emitted lib has no q8_0 mul_mm, so it auto-falls through.)
            "ds4_kernel_mul_mm_f16_f32",
            // f16-RHS grouped MoE GEMMs (DS4_MOE_MID_F16 down path): same emitted-vs-
            // bridge ABI collision as the f16 mul_mm above → force the bridge kernel.
            "ds4_kernel_mul_mm_id_q2_K_f16",
            "ds4_kernel_mul_mm_id_q8_0_f16",
            "ds4_kernel_mul_mm_id_q4_K_f16",
            "ds4_kernel_mul_mm_id_iq2_xxs_f16",
        ];
        let bridge_first = BRIDGE_PREFERRED.contains(&symbol);

        let function = if bridge_first {
            match self
                .bridge_lib
                .as_ref()
                .and_then(|l| l.get_function(symbol, Some(make_values())).ok())
            {
                Some(f) => f,
                None => self
                    .emitted_lib
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "specialized_pipeline: bridge-preferred symbol {} \
                             not in bridge lib and emitted lib unavailable",
                            symbol
                        )
                    })?
                    .get_function(symbol, Some(make_values()))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "specialized_pipeline: get_function({}) failed in both libs: {}",
                            symbol,
                            e
                        )
                    })?,
            }
        } else {
            match self
                .emitted_lib
                .as_ref()
                .and_then(|l| l.get_function(symbol, Some(make_values())).ok())
            {
                Some(f) => f,
                None => self
                    .bridge_lib
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "specialized_pipeline: symbol {} not in emitted lib \
                             and bridge lib unavailable",
                            symbol
                        )
                    })?
                    .get_function(symbol, Some(make_values()))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "specialized_pipeline: get_function({}) failed in both libs: {}",
                            symbol,
                            e
                        )
                    })?,
            }
        };

        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                anyhow::anyhow!(
                    "specialized_pipeline: pipeline build for {} failed: {}",
                    symbol,
                    e
                )
            })?;
        if pso.max_total_threads_per_threadgroup() < 1024 {
            eprintln!(
                "ds4_metal: pipeline {symbol} maxThreads={} (<1024)",
                pso.max_total_threads_per_threadgroup()
            );
        }

        if let Ok(mut guard) = self.specialized.lock() {
            guard.insert(cache_key, pso.clone());
        }
        Ok(pso)
    }

    /// Get-or-create a cached `MTLBuffer` for the given f32 weight slice,
    /// keyed by its `(host_ptr, byte_len)`.
    ///
    /// Phase F task #93 — picks ONE of two backing strategies per
    /// (host_ptr, byte_len) on first call:
    /// - **Zero-copy** (`newBufferWithBytesNoCopy`) when the slice base
    ///   is page-aligned AND the byte length is ≥ a page. Saves the
    ///   host→GPU memcpy AND, more importantly, the 25 GB of duplicated
    ///   Storage:Shared RAM that mirrors `ComposedModelWeights`. Counts
    ///   tracked in `weight_no_copy_*`.
    /// - **Memcpy fallback** (`new_buffer_with_data`) otherwise — keeps
    ///   the old behavior for small/unaligned slices. Counts tracked
    ///   in `weight_memcpy_*`.
    ///
    /// On macOS Apple Silicon the page size is 16 KiB; the system
    /// allocator returns page-aligned addresses for sufficiently-large
    /// `Vec<T>` allocations (typically ≥ 1 MiB), which covers the
    /// dominant attn/lm_head/router weight slices but not tiny gammas.
    ///
    /// Set `DS4_WEIGHT_CACHE=0` to disable (per-call upload, matches
    /// the legacy slow path). Set `DS4_WEIGHT_NO_COPY=0` to force the
    /// memcpy fallback for everything (diagnostic; restores pre-#93
    /// behavior).
    ///
    /// CALLER CONTRACT: `data` must be an immutable, lifetime-stable
    /// weight slice (e.g., a borrow from `ComposedModelWeights`). For
    /// the zero-copy path the source `Vec` MUST outlive every cached
    /// `MTLBuffer` and MUST NOT be reallocated (no `push`/`shrink`/
    /// `resize`) after caching — otherwise the buffer points at freed
    /// memory. Passing an activation slice (whose contents change
    /// across calls at the same address) will return stale GPU data.
    /// Activations must use `new_input_buffer` directly.
    #[allow(dead_code)]
    pub(crate) fn cached_weight_buffer(&self, data: &[f32]) -> metal::Buffer {
        use std::sync::atomic::Ordering;
        // DS4_TRAP_ZERO_WEIGHT=1: hard-panic on a zero-length weight slice so
        // the full backtrace pins the lean-dropped f32 bind (diagnostic).
        if data.is_empty() && std::env::var("DS4_TRAP_ZERO_WEIGHT").is_ok() {
            panic!("cached_weight_buffer: ZERO-length weight slice");
        }
        let bypass = std::env::var("DS4_WEIGHT_CACHE")
            .map(|v| v == "0")
            .unwrap_or(false);
        if bypass {
            return new_input_buffer(&self.device, data);
        }
        let byte_len = std::mem::size_of_val(data);
        let key = (data.as_ptr() as usize, byte_len);
        if let Ok(guard) = self.weight_buf_cache.lock() {
            if let Some(buf) = guard.get(&key) {
                return buf.clone();
            }
        }
        let no_copy_enabled = std::env::var("DS4_WEIGHT_NO_COPY")
            .map(|v| v != "0")
            .unwrap_or(true);
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let ptr_usize = data.as_ptr() as usize;
        let buf = if no_copy_enabled
            && byte_len >= page
            && ptr_usize % page == 0
        {
            // Zero-copy view over the caller-owned heap allocation.
            // Metal treats the source as device-resident Shared memory;
            // the source Vec MUST outlive every `metal::Buffer` we hand out.
            let mb = self.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const std::ffi::c_void,
                byte_len as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            self.weight_no_copy_count.fetch_add(1, Ordering::Relaxed);
            self.weight_no_copy_bytes
                .fetch_add(byte_len as u64, Ordering::Relaxed);
            mb
        } else {
            let mb = new_input_buffer(&self.device, data);
            self.weight_memcpy_count.fetch_add(1, Ordering::Relaxed);
            self.weight_memcpy_bytes
                .fetch_add(byte_len as u64, Ordering::Relaxed);
            mb
        };
        if let Ok(mut guard) = self.weight_buf_cache.lock() {
            guard.entry(key).or_insert_with(|| buf.clone());
        }
        self.pin_weight_buffer_resident(&buf);
        buf
    }

    /// Re-quantize an already-dequantized f32 projection weight to GGUF
    /// `block_q8_0` bytes and upload it, caching by the f32 slice's
    /// `(host_ptr, byte_len)`. Returns `(buffer, n_weights)`. Subsequent calls
    /// with the same slice reuse the uploaded Q8_0 buffer (the quantization
    /// runs once). `DS4_WEIGHT_CACHE=0` bypasses the cache.
    pub(crate) fn cached_q8_0_weight_buffer(&self, data: &[f32]) -> metal::Buffer {
        let bypass = std::env::var("DS4_WEIGHT_CACHE")
            .map(|v| v == "0")
            .unwrap_or(false);
        let key = (data.as_ptr() as usize, std::mem::size_of_val(data));
        if !bypass {
            if let Ok(guard) = self.q8_weight_buf_cache.lock() {
                if let Some(buf) = guard.get(&key) {
                    return buf.clone();
                }
            }
        }
        if std::env::var("DS4_Q8_DBG").is_ok() {
            eprintln!("[q8 cache MISS] quantizing {} weights", data.len());
        }
        let bytes = crate::quantized_experts::quantize_q8_0_to_bytes(data);
        let buf = new_input_buffer(&self.device, &bytes);
        if !bypass {
            if let Ok(mut guard) = self.q8_weight_buf_cache.lock() {
                guard.entry(key).or_insert_with(|| buf.clone());
            }
        }
        self.pin_weight_buffer_resident(&buf);
        buf
    }

    /// Upload raw GGUF `block_q8_0` projection bytes (owned, page-aligned —
    /// carried in `AttnLayerWeights::attn_*_q8`) as a no-copy MTLBuffer, caching
    /// by `(host_ptr, byte_len)`. Mirrors `cached_weight_buffer`'s no-copy path
    /// so the buffer shares the same warm-resident treatment as the f32
    /// projections (no copy into Metal-managed storage = no page-fault stalls,
    /// the failure mode of the re-quantize path). `DS4_WEIGHT_CACHE=0` bypasses.
    pub(crate) fn cached_q8_0_raw_buffer(&self, bytes: &[u8]) -> metal::Buffer {
        use std::sync::atomic::Ordering;
        let bypass = std::env::var("DS4_WEIGHT_CACHE")
            .map(|v| v == "0")
            .unwrap_or(false);
        let byte_len = bytes.len();
        let key = (bytes.as_ptr() as usize, byte_len);
        if !bypass {
            if let Ok(guard) = self.q8_weight_buf_cache.lock() {
                if let Some(buf) = guard.get(&key) {
                    return buf.clone();
                }
            }
        }
        let no_copy_enabled = std::env::var("DS4_WEIGHT_NO_COPY")
            .map(|v| v != "0")
            .unwrap_or(true);
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        // Phase 2: no-copy even when the slice ptr isn't page-aligned. The q8
        // weights are now borrowed straight from the model mmap (Cow::Borrowed),
        // whose tensor offsets are only 32-byte aligned — but the underlying
        // allocation (the mmap) IS page-aligned, and Metal tolerates intra-page
        // offsets on newBufferWithBytesNoCopy (the experts' load_quant_tensor
        // path relies on exactly this and is bit-identical). The old
        // `ptr % page == 0` gate forced these into a memcpy (anonymous heap),
        // defeating the no-copy. Owned page-aligned Vecs still take this path too.
        let buf = if no_copy_enabled && byte_len >= page {
            let mb = self.device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr() as *const std::ffi::c_void,
                byte_len as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            self.weight_no_copy_count.fetch_add(1, Ordering::Relaxed);
            self.weight_no_copy_bytes
                .fetch_add(byte_len as u64, Ordering::Relaxed);
            mb
        } else {
            new_input_buffer(&self.device, bytes)
        };
        if !bypass {
            if let Ok(mut guard) = self.q8_weight_buf_cache.lock() {
                guard.entry(key).or_insert_with(|| buf.clone());
            }
        }
        self.pin_weight_buffer_resident(&buf);
        buf
    }

    /// Load per-layer quantized expert weight tables from a GGUF. Call
    /// once at model-load time before the first `moe_routed_step`. The
    /// `gguf_bytes` slice must back the same file `gguf` was parsed from
    /// (we copy bytes out, so the slice lifetime ends with this call).
    #[allow(dead_code)]
    pub(crate) fn load_expert_weights(
        &mut self,
        gguf: &ds4_engine::gguf::GgufFile,
        gguf_bytes: &[u8],
        n_moe_layers: u32,
    ) -> Result<()> {
        self.expert_weights.clear();
        self.expert_weights.reserve(n_moe_layers as usize);
        for layer in 0..n_moe_layers {
            let qew = QuantizedExpertWeights::from_gguf(gguf, gguf_bytes, layer, &self.device)?;
            self.expert_weights.push(qew);
        }
        // SSD-streaming: build the fixed-slot expert caches. The stacked
        // tensors stay mmap-only (never bound — GPU faults are fatal); MoE
        // dispatch binds the caches after a CPU ensure+remap.
        if std::env::var("DS4_SSD_STREAM").is_ok() && !self.expert_weights.is_empty() {
            let budget_gb: f64 = std::env::var("DS4_SSD_CACHE_GB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40.0);
            let per_expert: u64 = {
                let q = &self.expert_weights[0];
                q.gate.expert_stride + q.up.expert_stride + q.down.expert_stride
            };
            let total_slots = ((budget_gb * 1e9) as u64 / per_expert.max(1)) as u32;
            let per_layer = (total_slots / n_moe_layers.max(1)).clamp(8, 384);
            eprintln!(
                "ds4_metal: SSD-stream expert cache {per_layer} slots/layer × {n_moe_layers} layers ({:.1} GB)",
                per_layer as f64 * n_moe_layers as f64 * per_expert as f64 / 1e9,
            );
            for qew in &self.expert_weights {
                self.expert_caches.push(std::sync::Mutex::new(
                    crate::quantized_experts::ExpertCache::new(&self.device, qew, per_layer),
                ));
            }
            // Chunked-prefill whole-layer pool (slots == n_experts, ~2.6 GB):
            // one pool reused across layers so the chunk mm_id MoE runs with
            // identity expert ids. Allocated up front under streaming —
            // prefill always wants it and 2.6 GB is within the cache budget's
            // accounting slack. -1 = no layer resident yet.
            let q0 = &self.expert_weights[0];
            self.chunk_pool = Some((
                std::sync::Mutex::new(crate::quantized_experts::ExpertCache::new(
                    &self.device, q0, q0.n_experts as u32,
                )),
                std::sync::atomic::AtomicI32::new(-1),
            ));
            if std::env::var("DS4_POOL_DOUBLE").ok().as_deref() == Some("1") {
                self.chunk_pool2 = Some((
                    std::sync::Mutex::new(crate::quantized_experts::ExpertCache::new(
                        &self.device, q0, q0.n_experts as u32,
                    )),
                    std::sync::atomic::AtomicI32::new(-1),
                ));
            }
            eprintln!(
                "ds4_metal: SSD-stream chunk pools 2 × {} slots ({:.2} GB)",
                q0.n_experts,
                2.0 * q0.n_experts as f64 * per_expert as f64 / 1e9
            );
            // Prefill-first: pin the first N layers' whole-layer pools
            // permanently (filled lazily on first chunk; owner sticks).
            let pin_n = std::env::var("DS4_POOL_PIN_LAYERS")
                .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0)
                .min(self.expert_weights.len());
            for li in 0..pin_n {
                let q = &self.expert_weights[li];
                self.pinned_pool.push((
                    std::sync::Mutex::new(crate::quantized_experts::ExpertCache::new(
                        &self.device, q, q.n_experts as u32,
                    )),
                    std::sync::atomic::AtomicI32::new(-1),
                ));
            }
            if pin_n > 0 {
                eprintln!(
                    "ds4_metal: SSD-stream pinned pools {} layers ({:.1} GB)",
                    pin_n,
                    pin_n as f64 * q0.n_experts as f64 * per_expert as f64 / 1e9
                );
            }
        }
        self.request_model_residency();
        Ok(())
    }

    /// SSD-streaming MoE bind: ensure the selected experts are cached and
    /// return (slot ids, gate, up, down) clones to bind in place of the
    /// full stacked tensors. None when streaming is off.
    pub(crate) fn streaming_expert_bind(
        &self,
        layer_idx: u32,
        selected: &[i32],
    ) -> Option<(Vec<i32>, metal::Buffer, metal::Buffer, metal::Buffer)> {
        // DS4_SSD_STUB=0 probe: tensors fully bound — skip the cache pool.
        if std::env::var("DS4_SSD_STUB").map(|v| v == "0").unwrap_or(false) {
            return None;
        }
        let cache = self.expert_caches.get(layer_idx as usize)?;
        let qew = self.expert_weights.get(layer_idx as usize)?;
        let mut c = cache.lock().ok()?;
        let slots = c.ensure(qew, selected);
        Some((slots, c.gate.clone(), c.up.clone(), c.down.clone()))
    }

    /// Chunked-prefill pool: does layer `layer_idx` need a (CPU) refill?
    /// When true the caller must wait for in-flight GPU work referencing the
    /// pool buffers before calling `chunk_pool_fill`.
    pub(crate) fn chunk_pool_needs_refill(&self, layer_idx: u32) -> bool {
        self.pool_for(layer_idx)
            .map(|(_, owner)| owner.load(std::sync::atomic::Ordering::Acquire) != layer_idx as i32)
            .unwrap_or(false)
    }

    /// Pinned layers (prefill-first budget): layers < DS4_POOL_PIN_LAYERS get
    /// their own pool, filled once, never refilled.
    pub(crate) fn pinned_pools(&self) -> usize {
        std::env::var("DS4_POOL_PIN_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
    }

    /// Double-buffer: layer L uses pool[L % 2].
    fn pool_for(
        &self,
        layer_idx: u32,
    ) -> Option<&(std::sync::Mutex<crate::quantized_experts::ExpertCache>, std::sync::atomic::AtomicI32)>
    {
        if (layer_idx as usize) < self.pinned_pools() {
            return self.pinned_pool.get(layer_idx as usize);
        }
        if std::env::var("DS4_POOL_DOUBLE").ok().as_deref() != Some("1") {
            return self.chunk_pool.as_ref();
        }
        if layer_idx % 2 == 0 { self.chunk_pool.as_ref() } else { self.chunk_pool2.as_ref() }
    }

    /// Chunked-prefill pool: fill with layer `layer_idx`'s full expert set
    /// (no-op when already resident) and return (gate, up, down) buffers.
    pub(crate) fn chunk_pool_bind(
        &self,
        layer_idx: u32,
    ) -> Option<(metal::Buffer, metal::Buffer, metal::Buffer)> {
        // Wait for any in-flight background prefetch (it may be filling THIS
        // layer's pool, and at most one prefetch runs at a time).
        if let Ok(mut g) = self.chunk_prefetch.lock() {
            if let Some(h) = g.take() {
                let _ = h.join();
            }
        }
        let (pool, owner) = self.pool_for(layer_idx)?;
        let qew = self.expert_weights.get(layer_idx as usize)?;
        let mut p = pool.lock().ok()?;
        if owner.load(std::sync::atomic::Ordering::Acquire) != layer_idx as i32 {
            let t0 = std::time::Instant::now();
            match self.expert_caches.get(layer_idx as usize).and_then(|c| c.lock().ok()) {
                Some(lru) => p.fill_layer_from_lru(qew, &lru),
                None => p.fill_layer(qew),
            }
            if std::env::var("DS4_POOL_TIMING").is_ok() {
                let gb = (qew.gate.expert_stride + qew.up.expert_stride + qew.down.expert_stride)
                    as f64 * qew.n_experts as f64 / 1e9;
                eprintln!(
                    "[pool] L{layer_idx} fill {:.2} GB in {:.2}s = {:.2} GB/s",
                    gb, t0.elapsed().as_secs_f64(), gb / t0.elapsed().as_secs_f64()
                );
            }
            owner.store(layer_idx as i32, std::sync::atomic::Ordering::Release);
        }
        let out = (p.gate.clone(), p.up.clone(), p.down.clone());
        drop(p);
        // Kick a background fill of layer L+1 into the OTHER pool — it is not
        // referenced by in-flight GPU work (per-layer commit_wait precedes
        // every refill), so the copy overlaps this layer's GPU compute.
        // DS4_POOL_DOUBLE=1 opt-in: the second 1.8GB pool puts a 64GB box over
        // budget at long ctx (@3000 prefill 266→578s) — net loss by default.
        let next = layer_idx + 1;
        if std::env::var("DS4_POOL_DOUBLE").ok().as_deref() == Some("1")
            && (next as usize) < self.expert_weights.len()
        {
            if let Some((npool, nowner)) = self.pool_for(next) {
                if nowner.load(std::sync::atomic::Ordering::Acquire) != next as i32 {
                    // SAFETY: MetalState outlives all decode/prefill calls and
                    // every chunk_pool_bind joins the previous prefetch first.
                    let npool: &'static std::sync::Mutex<crate::quantized_experts::ExpertCache> =
                        unsafe { std::mem::transmute(npool) };
                    let nowner: &'static std::sync::atomic::AtomicI32 =
                        unsafe { std::mem::transmute(nowner) };
                    let nqew: &'static crate::quantized_experts::QuantizedExpertWeights =
                        unsafe { std::mem::transmute(self.expert_weights.get(next as usize)?) };
                    let nlru: Option<&'static std::sync::Mutex<crate::quantized_experts::ExpertCache>> =
                        self.expert_caches.get(next as usize).map(|c| unsafe { std::mem::transmute(c) });
                    let h = std::thread::spawn(move || {
                        if let Ok(mut p) = npool.lock() {
                            match nlru.and_then(|c| c.lock().ok()) {
                                Some(lru) => p.fill_layer_from_lru(nqew, &lru),
                                None => p.fill_layer(nqew),
                            }
                            nowner.store(next as i32, std::sync::atomic::Ordering::Release);
                        }
                    });
                    if let Ok(mut g) = self.chunk_prefetch.lock() {
                        *g = Some(h);
                    }
                }
            }
        }
        Some(out)
    }

    /// Phase 3 — load MTP drafter expert weights from a SEPARATE GGUF file
    /// (antirez DeepSeek-V4-Flash-MTP-*.gguf). Appends one slot to
    /// `state.expert_weights` and returns its index — the caller passes
    /// this back as `mtp_layer_idx` to `encode_mtp_draft` so the drafter's
    /// MoE chain pulls from the right slot.
    ///
    /// MUST be called AFTER `load_expert_weights` (which clears the table).
    /// The returned index is `n_base_moe_layers` (current vec length).
    pub(crate) fn load_mtp_expert_weights(
        &mut self,
        mtp_gguf: &ds4_engine::gguf::GgufFile,
        mtp_gguf_bytes: &[u8],
    ) -> Result<u32> {
        let allocated_idx = self.expert_weights.len() as u32;
        let qew = QuantizedExpertWeights::from_mtp_gguf(
            mtp_gguf, mtp_gguf_bytes, allocated_idx, &self.device,
        )?;
        self.expert_weights.push(qew);
        // Re-pin residency to include the new MTP slot buffers.
        self.request_model_residency();
        Ok(allocated_idx)
    }

    /// Pin all expert weight buffers as GPU-resident via `MTLResidencySet`
    /// (macOS 15+). Without this, the GPU page-faults waiting for the OS
    /// to swap in expert weight pages on every `moe_routed_step` call —
    /// the model is mmap-backed and only 6 of 256 experts are touched per
    /// token, so Apple's default residency policy evicts cold pages and
    /// reloads them each access. Mirrors antirez `ds4_metal.m:352-387`.
    ///
    /// No-op if `DS4_METAL_NO_RESIDENCY` is set, on macOS < 15, or if any
    /// step of the residency-set creation fails (logged once to stderr).
    fn request_model_residency(&self) {
        if std::env::var("DS4_METAL_NO_RESIDENCY").is_ok() {
            return;
        }
        // SSD-streaming mode: expert weights exceed RAM, so pinning them is
        // impossible (over-budget residency wedges the GPU) — they are also
        // never bound (cache slots are bound instead). The residency set
        // itself is still created so dense weights pin via first-touch
        // (`pin_weight_buffer_resident`); only the expert addAllocation loop
        // is skipped.
        let ssd_stream = std::env::var("DS4_SSD_STREAM").is_ok();
        if ssd_stream {
            eprintln!("ds4_metal: DS4_SSD_STREAM — residency set without expert pinning");
        }
        if self.expert_weights.is_empty() {
            return;
        }
        // Already requested? Don't double-add.
        if let Ok(guard) = self.model_residency_set.lock() {
            if guard.is_some() {
                return;
            }
        }
        use metal::foreign_types::ForeignType;
        use objc::runtime::Object;
        use objc::{class, msg_send, sel, sel_impl};
        unsafe {
            // [[MTLResidencySetDescriptor alloc] init]
            let desc_class = match objc::runtime::Class::get("MTLResidencySetDescriptor") {
                Some(c) => c,
                None => {
                    eprintln!(
                        "ds4_metal: MTLResidencySetDescriptor class not available — \
                         skipping residency pinning (requires macOS 15+)"
                    );
                    return;
                }
            };
            let desc: *mut Object = msg_send![desc_class, alloc];
            let desc: *mut Object = msg_send![desc, init];

            // [device newResidencySetWithDescriptor:desc error:&error]
            let device_ptr: *mut Object = std::mem::transmute(self.device.as_ptr());
            let mut error: *mut Object = std::ptr::null_mut();
            let set: *mut Object = msg_send![
                device_ptr,
                newResidencySetWithDescriptor: desc
                error: &mut error
            ];
            // Release the descriptor (we hold it via init; the residency
            // set has its own reference if it needs the descriptor).
            let _: () = msg_send![desc, release];
            if set.is_null() {
                let msg: *mut Object = if !error.is_null() {
                    msg_send![error, localizedDescription]
                } else {
                    std::ptr::null_mut()
                };
                let utf8: *const std::os::raw::c_char = if !msg.is_null() {
                    msg_send![msg, UTF8String]
                } else {
                    std::ptr::null()
                };
                let descr = if !utf8.is_null() {
                    std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
                } else {
                    "(no description)".to_string()
                };
                eprintln!(
                    "ds4_metal: newResidencySetWithDescriptor failed: {} — skipping pinning",
                    descr
                );
                return;
            }

            // Add every expert weight buffer (skip under SSD streaming —
            // the cache buffers pin via first-touch instead).
            let mut n_added: usize = 0;
            if !ssd_stream {
                for qew in &self.expert_weights {
                    for tensor in [&qew.gate, &qew.up, &qew.down] {
                        let buf_ptr: *mut Object = std::mem::transmute(tensor.metal_buf.as_ptr());
                        let _: () = msg_send![set, addAllocation: buf_ptr];
                        n_added += 1;
                    }
                }
            }
            // [set commit]; [set requestResidency]
            let _: () = msg_send![set, commit];
            let _: () = msg_send![set, requestResidency];
            // T1 (bounded-working-set): bind the set to the command queue so
            // EVERY cb submitted on it keeps the set resident under pressure —
            // not just the one-shot requestResidency above. antirez does this
            // (ds4_metal.m:1170 `addResidencySet:`) + per-cb `useResidencySet:`.
            // `requestResidency` alone makes the set resident NOW but doesn't
            // tie it to the queue's per-cb residency, so transient scratch added
            // later (T2) can still be evicted mid-cb under daemon pressure.
            // Guarded by respondsToSelector for pre-macOS-15 / non-Apple GPUs.
            let queue_ptr: *mut Object = std::mem::transmute(self.command_queue.as_ptr());
            let binds: bool =
                msg_send![queue_ptr, respondsToSelector: sel!(addResidencySet:)];
            if binds {
                let _: () = msg_send![queue_ptr, addResidencySet: set];
            }
            eprintln!(
                "ds4_metal: residency pinned {} expert weight buffers via MTLResidencySet \
                 (queue-bound={})",
                n_added, binds
            );

            if let Ok(mut guard) = self.model_residency_set.lock() {
                *guard = Some(set);
            }
            // Suppress unused-var on `class!`/`sel!` macros if absent.
            let _ = (class!(NSObject), sel!(release));
        }
    }

    /// Add a freshly-cached non-expert weight buffer to the model residency
    /// set so it stays GPU-resident (wired) and is NOT compressed when the
    /// daemon idles. `request_model_residency` pins only the experts; the
    /// ~31 GB of anonymous projection / q8 / shared-expert weight buffers
    /// (`ComposedModelWeights` mirrors) were left unpinned, so macOS swapped
    /// them to the compressor between requests — the next decode then paid a
    /// multi-second decompress and throughput collapsed (0.2 tok/s) until the
    /// pages faulted back. antirez avoids this by requesting residency for its
    /// whole 80 GB model. Mirror that for our weight buffers.
    ///
    /// Called once per distinct weight buffer (on cache MISS) from the
    /// `cached_*_buffer` helpers; the buffers are cached for the process
    /// lifetime so the add/commit/requestResidency cost is one-time (first
    /// decode). No-op if the set wasn't created (DS4_METAL_NO_RESIDENCY /
    /// macOS < 15) or if `DS4_PIN_WEIGHTS=0`.
    /// Pin a persistent per-layer STATE buffer (KV ring, compressor/indexer
    /// pools, comp/index emit rings) GPU-resident, UNGATED (a few hundred MB
    /// total vs the 86 GB weights — no commit-cost concern). The decode reads
    /// these every token; if macOS evicts/compresses a ring while a flash reads
    /// it, the read silently yields zeros (no cb error) — a binary,
    /// load-dependent corruption signature. DS4_NO_PIN_RINGS=1 disables (A/B).
    pub(crate) fn pin_state_buffer_resident(&self, buf: &metal::Buffer) {
        if std::env::var("DS4_NO_PIN_RINGS").ok().as_deref() == Some("1") {
            return;
        }
        use metal::foreign_types::ForeignType;
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        if let Ok(guard) = self.model_residency_set.lock() {
            if let Some(set) = *guard {
                unsafe {
                    let buf_ptr: *mut Object = std::mem::transmute(buf.as_ptr());
                    let _: () = msg_send![set, addAllocation: buf_ptr];
                }
                self.residency_dirty
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn pin_weight_buffer_resident(&self, buf: &metal::Buffer) {
        // Opt-in (DS4_PIN_WEIGHTS=1). Default OFF: pins the ~31 GB of non-expert
        // weight buffers resident to stop idle-compression, but the
        // commit/requestResidency cost on a 100 GB+ set is unvalidated on a
        // memory-clean box (this session's box was too churned to benchmark),
        // so it stays opt-in until a clean-reboot A/B confirms it's a net win
        // and doesn't stall first decode. Default keeps the known-good daemon.
        if std::env::var("DS4_PIN_WEIGHTS").map(|v| v != "1").unwrap_or(true) {
            return;
        }
        use metal::foreign_types::ForeignType;
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        if let Ok(guard) = self.model_residency_set.lock() {
            if let Some(set) = *guard {
                unsafe {
                    let buf_ptr: *mut Object = std::mem::transmute(buf.as_ptr());
                    let _: () = msg_send![set, addAllocation: buf_ptr];
                }
                // Defer the (O(set)) commit+requestResidency to `commit_residency`
                // at the token boundary — paying it per buffer thrashed.
                self.residency_dirty
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Flush pending residency-set additions: one `commit` + `requestResidency`
    /// for all weight buffers `addAllocation`'d since the last flush. Call once
    /// per decode token (cheap no-op once the weight caches are warm and the
    /// dirty flag stays false). Splitting the per-buffer `addAllocation` from
    /// this batched commit avoids the O(buffers × set) thrash of committing on
    /// every cache miss. No-op if the set is absent or `DS4_PIN_WEIGHTS=0`.
    pub(crate) fn commit_residency(&self) {
        if !self
            .residency_dirty
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        if let Ok(guard) = self.model_residency_set.lock() {
            if let Some(set) = *guard {
                unsafe {
                    let _: () = msg_send![set, commit];
                    let _: () = msg_send![set, requestResidency];
                }
            }
        }
    }

    /// Speculative softplus_sqrt encoding. Not wired into the trait yet
    /// because it requires a compiled `metal::ComputePipelineState` for
    /// `ds4_kernel_dsv4_softplus_sqrt_f32_4`, which in turn needs the
    /// emitted MSL library loaded. Cross-compiles cleanly for
    /// aarch64-apple-darwin so the encoding scaffolding is checked.
    ///
    /// Kernel signature (from the registry + emitter):
    ///   - p0=src (char* const)
    ///   - p1=dst (char* writable)
    ///   - ne0_4 (u32) = N/4 (number of float4 lanes per row)
    ///   - nb_src (u32) = row stride in bytes (=4*N for a 1D row)
    ///   - nb_dst (u32) = row stride in bytes
    /// Dispatch: 1 threadgroup (row=0), ne0_4 threads.
    #[allow(dead_code)]
    /// Commit + wait the given cb. When `DS4_OP_TRACE=1`, also log a
    /// line of the shape:
    ///
    ///   `DS4_OP_TRACE,op=<name>,commit=<us>,wait=<us>,gpu=<us>`
    ///
    /// This lets us identify which ops have the highest per-call wait
    /// cost — the dominant factor in dispatch-bound throughput on the
    /// Metal backend (see project_m23_path_c_a_batchscope memory).
    /// Call sites: every `*_impl` that owns a single cmd_buf + commit
    /// + wait pattern. Existing `DS4_MOE_TRACE` / `DS4_FFN_TRACE`
    /// remain as richer per-op channels for moe and shared_chain.
    pub(crate) fn commit_wait_traced(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        op_name: &str,
    ) {
        let trace = std::env::var("DS4_OP_TRACE").is_ok();
        let t_commit = std::time::Instant::now();
        cmd_buf.commit();
        let commit_us = t_commit.elapsed().as_micros();
        let t_wait = std::time::Instant::now();
        cmd_buf.wait_until_completed();
        let wait_us = t_wait.elapsed().as_micros();
        if trace {
            let (gpu_start, gpu_end): (f64, f64) = unsafe {
                use metal::foreign_types::ForeignTypeRef;
                use objc::runtime::Object;
                use objc::{msg_send, sel, sel_impl};
                let cb_ptr: *mut Object = std::mem::transmute(cmd_buf.as_ptr());
                let s: f64 = msg_send![cb_ptr, GPUStartTime];
                let e: f64 = msg_send![cb_ptr, GPUEndTime];
                (s, e)
            };
            let gpu_us = ((gpu_end - gpu_start) * 1_000_000.0).max(0.0) as u64;
            eprintln!(
                "DS4_OP_TRACE,op={},commit={},wait={},gpu={}",
                op_name, commit_us, wait_us, gpu_us
            );
        }
    }

    pub(crate) fn softplus_sqrt_impl(&self, logits: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            logits.len() % 4 == 0,
            "softplus_sqrt requires len divisible by 4 (float4 lanes)"
        );
        let pipeline = self
            .pipelines
            .get("ds4_kernel_dsv4_softplus_sqrt_f32_4")
            .ok_or_else(|| anyhow::anyhow!("softplus_sqrt pipeline not loaded"))?;

        let in_buf = new_input_buffer(&self.device, logits);
        let out_buf = new_output_buffer::<f32>(&self.device, logits.len());

        let ne0_4: u32 = (logits.len() / 4) as u32;
        let nb: u32 = (logits.len() * 4) as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        set_scalar_bytes(enc, 2, &ne0_4);
        set_scalar_bytes(enc, 3, &nb);
        set_scalar_bytes(enc, 4, &nb);
        enc.dispatch_threads(
            MTLSize::new(ne0_4 as u64, 1, 1),
            MTLSize::new(ne0_4 as u64, 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "softplus_sqrt_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, logits.len()) })
    }

    /// rms_norm encoding via the float4-vectorized
    /// `ds4_kernel_rms_norm_mul_f32_4` kernel (RMSNorm × learned weight).
    ///
    /// Bindings match antirez `ds4_metal.m` (~line 5527):
    ///   buffer(0) = `ds4_metal_args_norm` struct (set_bytes)
    ///   buffer(1) = src0 (x, f32, float4-cast in-kernel)
    ///   buffer(2) = src1_0 (gamma, f32)
    ///   buffer(3) = src1_1 (unused for F=2 multiply path; bind a
    ///               placeholder)
    ///   buffer(4) = dst (f32)
    /// Threadgroup memory 0: 32 floats (per-simdgroup partial sums).
    /// Dispatch: 1 threadgroup per row × `rms_norm_threads(n)` threads.
    pub(crate) fn rms_norm_impl(&self, x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == gamma.len(),
            "rms_norm: x.len ({}) != gamma.len ({})",
            x.len(),
            gamma.len()
        );
        anyhow::ensure!(
            x.len() % 4 == 0,
            "rms_norm requires len divisible by 4 (float4 lanes)"
        );
        let pipeline = self
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?;

        let n = x.len() as u32;
        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;

        // Build the uniform buffer as a raw 144-byte byte vector to match
        // the exact C layout antirez computes. Offsets per `ds4_metal.m:1928`:
        //   off 0: ne00        i32
        //   off 4: ne00_t      i32
        //   off 8: nb1         u64
        //   off 16: nb2        u64
        //   off 24: nb3        u64
        //   off 32: eps        f32
        //   off 36: nef1[3]    i32 × 3
        //   off 48: nef2[3]    i32 × 3
        //   off 60: nef3[3]    i32 × 3
        //   off 72: nbf1[3]    u64 × 3   (8-byte aligned; 72%8==0)
        //   off 96: nbf2[3]    u64 × 3
        //   off 120: nbf3[3]   u64 × 3
        //   total: 144 bytes
        let mut args = [0u8; 144];
        args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        args[4..8].copy_from_slice(&((n / 4) as i32).to_le_bytes());
        args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        args[16..24].copy_from_slice(&(row_bytes * rows as u64).to_le_bytes());
        args[24..32].copy_from_slice(&(row_bytes * rows as u64).to_le_bytes());
        args[32..36].copy_from_slice(&eps.to_le_bytes());
        // nef1 = [rows, 1, 1]
        args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        args[40..44].copy_from_slice(&1i32.to_le_bytes());
        args[44..48].copy_from_slice(&1i32.to_le_bytes());
        // nef2 = [1, 1, 1]
        args[48..52].copy_from_slice(&1i32.to_le_bytes());
        args[52..56].copy_from_slice(&1i32.to_le_bytes());
        args[56..60].copy_from_slice(&1i32.to_le_bytes());
        // nef3 = [1, 1, 1]
        args[60..64].copy_from_slice(&1i32.to_le_bytes());
        args[64..68].copy_from_slice(&1i32.to_le_bytes());
        args[68..72].copy_from_slice(&1i32.to_le_bytes());
        // nbf1 = [row_bytes; 3]
        args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        // nbf2 = [row_bytes*rows, row_bytes, row_bytes]
        let plane = row_bytes * rows as u64;
        args[96..104].copy_from_slice(&plane.to_le_bytes());
        args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        // nbf3 = [plane, row_bytes, row_bytes]
        args[120..128].copy_from_slice(&plane.to_le_bytes());
        args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        let x_buf = new_input_buffer(&self.device, x);
        let w_buf = new_input_buffer(&self.device, gamma);
        let out_buf = new_output_buffer::<f32>(&self.device, x.len());

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        // src1_1 is unused for F=2 (mul) but bound for safety.
        enc.set_buffer(3, Some(&x_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        // 32 floats threadgroup scratch — antirez `ds4_metal.m:5533`.
        enc.set_threadgroup_memory_length(0, 32 * 4);

        // Threads per group: antirez `ds4_metal_rms_norm_threads(n)`:
        //   ne00_t = n / 4
        //   nth starts at 32, doubles while < ne00_t and < 1024
        //   then clamped to ne00_t
        let ne00_t = (n / 4) as u64;
        let mut nth: u64 = 32;
        while nth < ne00_t && nth < 1024 {
            nth *= 2;
        }
        if nth > ne00_t {
            nth = ne00_t;
        }
        let nth = nth.max(1);

        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(nth, 1, 1));
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "rms_norm_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, x.len()) })
    }

    /// M4 #330e — per-head RMS norm matching decode_step.rs:656-663.
    ///
    /// Layout: `x` is `[n_head * head_dim]`. Each head gets an independent
    /// RMS normalization (no learned gamma) with `eps` added after the
    /// per-head mean. One threadgroup per head; simdgroup reduction within
    /// the head.
    ///
    /// Fidelity gate: this Metal kernel uses f32 throughout, matching the
    /// default-OFF behaviour of DS4_HEAD_RMS_F64_FIDELITY (M4 #314). When
    /// the gate is ON the caller should fall back to the CPU f64 oracle.
    pub(crate) fn head_rms_norm_impl(
        &self,
        x: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == n_head * head_dim,
            "head_rms_norm: x.len ({}) != n_head*head_dim ({}*{}={})",
            x.len(),
            n_head,
            head_dim,
            n_head * head_dim,
        );
        anyhow::ensure!(
            head_dim >= 1,
            "head_rms_norm: head_dim must be >= 1",
        );
        let pipeline = self
            .pipelines
            .get("ds4_head_rms_norm_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_head_rms_norm_f32 pipeline not loaded"))?;

        let x_buf = new_input_buffer(&self.device, x);
        let out_buf = new_output_buffer::<f32>(&self.device, x.len());

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        let hd_u32 = head_dim as u32;
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

        // One threadgroup per head; pick tcount up to head_dim, capped at 1024,
        // rounded to next power-of-two for the simd reduction comfort zone.
        let mut tcount: u64 = 32;
        while tcount < head_dim as u64 && tcount < 1024 {
            tcount *= 2;
        }
        if tcount > head_dim as u64 {
            tcount = head_dim as u64;
        }
        let tcount = tcount.max(1);

        enc.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(tcount, 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "head_rms_norm_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, x.len()) })
    }

    /// M4 #330f — q8_0 activation round-trip matching
    /// `ds4_engine::forward::q8_0_round_trip` byte-exactly.
    ///
    /// Layout: input `[n_elts]`, output `[n_elts]`. Each 32-element block is
    /// quantized to i8 against its own amax/127, then dequantized back to
    /// f32. Tail (n_elts % 32) is quantized against its own amax.
    ///
    /// One threadgroup per block (32 threads). `rint()` matches C `lrintf`
    /// under default FE_TONEAREST = round-half-to-even, which is also what
    /// Rust `round_ties_even` produces — required for byte-exact match
    /// against the CPU oracle used by every gated q8_0 site (M4 #299/#302/
    /// #303/#304/#306).
    pub(crate) fn q8_0_round_trip_impl(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let pipeline = self
            .pipelines
            .get("ds4_q8_0_round_trip_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_q8_0_round_trip_f32 pipeline not loaded"))?;

        let in_buf = new_input_buffer(&self.device, x);
        let out_buf = new_output_buffer::<f32>(&self.device, x.len());

        let n_elts: u32 = x.len() as u32;
        let n_full_blocks: u32 = (x.len() / 32) as u32;
        let last_block_len: u32 = (x.len() % 32) as u32;
        let n_blocks_total: u32 = n_full_blocks + (last_block_len != 0) as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_elts as *const _ as *const _,
        );
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_full_blocks as *const _ as *const _,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &last_block_len as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(n_blocks_total as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "q8_0_round_trip_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, x.len()) })
    }

    /// M4 #330i — silu (elementwise) on Metal.
    ///
    /// `fidelity = false` ⇒ `ds4_silu_default_f32` (positive-branch
    /// always: `x / (1 + exp(-x))`).
    /// `fidelity = true`  ⇒ `ds4_silu_fidelity_f32` (antirez
    /// `sigmoid_stable` branched on sign — M4 #311 gate).
    ///
    /// One thread per element. Byte-exact against
    /// `ds4_engine::forward::silu` at the corresponding gate setting.
    pub(crate) fn silu_impl(&self, x: &[f32], fidelity: bool) -> Result<Vec<f32>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let sym = if fidelity {
            "ds4_silu_fidelity_f32"
        } else {
            "ds4_silu_default_f32"
        };
        let pipeline = self
            .pipelines
            .get(sym)
            .ok_or_else(|| anyhow::anyhow!("{} pipeline not loaded", sym))?;

        let in_buf = new_input_buffer(&self.device, x);
        let out_buf = new_output_buffer::<f32>(&self.device, x.len());
        let n: u32 = x.len() as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n as *const _ as *const _,
        );
        // One thread per element; let Metal pick the threadgroup size.
        let tcount = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(tcount.max(1), 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "silu_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, x.len()) })
    }

    /// M4 #330j — sigmoid (elementwise) on Metal.
    ///
    /// `fidelity = false` ⇒ `ds4_sigmoid_default_f32` (positive-branch
    /// always: `1 / (1 + exp(-x))`).
    /// `fidelity = true`  ⇒ `ds4_sigmoid_fidelity_f32` (antirez
    /// `sigmoid_stable` branched on sign).
    ///
    /// One thread per element. Used by `output_hc_head_one` (M4 #315)
    /// between the per-head `pre*scale + base` linear and the
    /// n_hc-wide weighted sum.
    pub(crate) fn sigmoid_impl(&self, x: &[f32], fidelity: bool) -> Result<Vec<f32>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let sym = if fidelity {
            "ds4_sigmoid_fidelity_f32"
        } else {
            "ds4_sigmoid_default_f32"
        };
        let pipeline = self
            .pipelines
            .get(sym)
            .ok_or_else(|| anyhow::anyhow!("{} pipeline not loaded", sym))?;

        let in_buf = new_input_buffer(&self.device, x);
        let out_buf = new_output_buffer::<f32>(&self.device, x.len());
        let n: u32 = x.len() as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n as *const _ as *const _,
        );
        let tcount = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(tcount.max(1), 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "sigmoid_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, x.len()) })
    }

    /// M4 #330k — softplus_sqrt (elementwise) on Metal, fidelity-aware.
    ///
    /// `fidelity = false` ⇒ `ds4_softplus_sqrt_default_f32` (stable softplus
    /// identity `max(x,0) + log(1 + exp(-|x|))` then sqrt).
    /// `fidelity = true`  ⇒ `ds4_softplus_sqrt_fidelity_f32` (antirez
    /// ds4.c:4867 piecewise: x>20 ⇒ sqrt(x); x<-20 ⇒ sqrt(exp(x));
    /// else ⇒ sqrt(log1p(exp(x)))).
    ///
    /// One thread per element. Used by the router-logit transform whose
    /// output Σ=1.5 multiplies INTO every routed-MoE expert's down output.
    pub(crate) fn softplus_sqrt_fidelity_impl(
        &self,
        logits: &[f32],
        fidelity: bool,
    ) -> Result<Vec<f32>> {
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        let sym = if fidelity {
            "ds4_softplus_sqrt_fidelity_f32"
        } else {
            "ds4_softplus_sqrt_default_f32"
        };
        let pipeline = self
            .pipelines
            .get(sym)
            .ok_or_else(|| anyhow::anyhow!("{} pipeline not loaded", sym))?;

        let in_buf = new_input_buffer(&self.device, logits);
        let out_buf = new_output_buffer::<f32>(&self.device, logits.len());
        let n: u32 = logits.len() as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n as *const _ as *const _,
        );
        let tcount = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(tcount.max(1), 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "softplus_sqrt_fidelity_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, logits.len()) })
    }

    /// Speculative router_finalize encoding via the two-kernel pipeline:
    ///
    ///   1. `kernel_dsv4_router_finalize_one` — 256-thread bitonic top-6 over
    ///      `(probs + bias)` writing 6 i32 selected indices.
    ///   2. `kernel_dsv4_router_weights_one` — 6-thread normalize step writing
    ///      `weights[i] = probs[selected[i]] / Σ_selected * 1.5`.
    ///
    /// Antirez splits this so denominator-order changes don't ripple through
    /// 43 MoE layers; we mirror the same split.
    ///
    /// Hash-mode short-circuit (`args.hash_mode=true`) is not exercised here —
    /// we always use the `false` branch (full bitonic). The struct uniform
    /// `ds4_metal_args_dsv4_router_select_one` is encoded inline via
    /// `set_bytes`; placeholder hash/tokens buffers are 1-element each.
    #[allow(dead_code)]
    pub(crate) fn router_finalize_impl(
        &self,
        probs: &[f32],
        bias: &[f32],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let n_experts = probs.len();
        anyhow::ensure!(
            n_experts > 0 && n_experts <= 1024,
            "router_finalize: n_experts out of range (got {})",
            n_experts
        );
        anyhow::ensure!(
            probs.len() == bias.len(),
            "router_finalize: probs.len ({}) != bias.len ({})",
            probs.len(),
            bias.len()
        );
        anyhow::ensure!(k == 6, "router_finalize: DS4 hard-codes k=6 (got {})", k);
        // Flash (256, 1.5) keeps the upstream hardcoded kernels (bit-identical
        // default); anything else takes the width/scale-generic shims.
        let generic =
            n_experts != 256 || ds4_engine::moe::router_scale() != 1.5;
        let npow2: u64 = (n_experts as u64).next_power_of_two();

        let select_pipe = self
            .pipelines
            .get(if generic {
                "ds4_dsv4_router_finalize_one_any"
            } else {
                "ds4_dsv4_router_finalize_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_finalize_one pipeline not loaded"))?;
        let weights_pipe = self
            .pipelines
            .get(if generic {
                "ds4_dsv4_router_weights_one_any"
            } else {
                "ds4_dsv4_router_weights_one"
            })
            .ok_or_else(|| anyhow::anyhow!("router_weights_one pipeline not loaded"))?;

        let probs_buf = new_input_buffer(&self.device, probs);
        let bias_buf = new_input_buffer(&self.device, bias);
        let hash_buf = new_input_buffer(&self.device, &[0i32]);
        let tokens_buf = new_input_buffer(&self.device, &[0i32]);
        let selected_buf = new_output_buffer::<i32>(&self.device, 6);
        let weights_buf = new_output_buffer::<f32>(&self.device, 6);

        // ds4_metal_args_dsv4_router_select_one — 5 scalars per registry.
        // Field order matches antirez: { has_bias, hash_mode, use_token_buffer,
        // token, hash_rows } as four uint32 + one uint32. We pack as 5×u32.
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
        let (sel_tg_threads, sel_shmem) = if generic {
            (npow2, 2 * npow2 * 4)
        } else {
            (256, 2 * 256 * 4)
        };

        let cmd_buf = self.command_queue.new_command_buffer();

        // ── Stage 1: bitonic top-6 selection ──────────────────────────────
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(select_pipe);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&probs_buf), 0);
            enc.set_buffer(2, Some(&bias_buf), 0);
            enc.set_buffer(3, Some(&hash_buf), 0);
            enc.set_buffer(4, Some(&tokens_buf), 0);
            enc.set_buffer(5, Some(&selected_buf), 0);
            // threadgroup<float, npow2> scratch + threadgroup<i32, npow2> idx.
            enc.set_threadgroup_memory_length(0, sel_shmem);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(sel_tg_threads, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        // ── Stage 2: normalize selected probs → weights ──────────────────
        // Emitted MSL signature (per emitter): probs, selected, weights at
        // buffer(0..2) + 3 uniform scalars (num_experts, scale, min_sum) at
        // buffer(3..5). Antirez fixes scale=1.5, min_sum=6.103515625e-5 (= 2^-14).
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(weights_pipe);
            enc.set_buffer(0, Some(&probs_buf), 0);
            enc.set_buffer(1, Some(&selected_buf), 0);
            enc.set_buffer(2, Some(&weights_buf), 0);
            let num_experts: u32 = k as u32;
            let scale: f32 = ds4_engine::moe::router_scale();
            let min_sum: f32 = 6.103515625e-5;
            set_scalar_bytes(enc, 3, &num_experts);
            set_scalar_bytes(enc, 4, &scale);
            set_scalar_bytes(enc, 5, &min_sum);
            enc.dispatch_threads(MTLSize::new(k as u64, 1, 1), MTLSize::new(k as u64, 1, 1));
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "router_finalize_impl");

        let selected_i32 = unsafe { read_buffer::<i32>(&selected_buf, 6) };
        let weights = unsafe { read_buffer::<f32>(&weights_buf, 6) };
        let selected: Vec<usize> = selected_i32.into_iter().map(|i| i as usize).collect();
        Ok((selected, weights))
    }

    pub(crate) fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.rms_norm_impl(x, gamma, eps)
            .expect("ds4_metal::rms_norm encoding failed")
    }

    /// Per-head RMS norm (no learned gamma). See `head_rms_norm_impl` for the
    /// semantics — mirrors decode_step.rs:656-663. CPU oracle path under
    /// DS4_HEAD_RMS_F64_FIDELITY=1 (caller's responsibility).
    pub(crate) fn head_rms_norm(
        &self,
        x: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        self.head_rms_norm_impl(x, n_head, head_dim, eps)
            .expect("ds4_metal::head_rms_norm encoding failed")
    }

    /// Per-32-elt-block activation quantization round-trip. See
    /// `q8_0_round_trip_impl` for the semantics — byte-exact against
    /// `ds4_engine::forward::q8_0_round_trip`.
    pub(crate) fn q8_0_round_trip(&self, x: &[f32]) -> Vec<f32> {
        self.q8_0_round_trip_impl(x)
            .expect("ds4_metal::q8_0_round_trip encoding failed")
    }

    /// Elementwise silu (Metal). `fidelity=true` uses the antirez
    /// `sigmoid_stable` branched kernel (M4 #311 gate); otherwise uses
    /// the positive-branch identity (`x / (1 + exp(-x))`).
    pub(crate) fn silu(&self, x: &[f32], fidelity: bool) -> Vec<f32> {
        self.silu_impl(x, fidelity)
            .expect("ds4_metal::silu encoding failed")
    }

    /// Elementwise sigmoid (Metal). `fidelity=true` uses the antirez
    /// `sigmoid_stable` branched kernel; otherwise uses the positive-
    /// branch identity (`1 / (1 + exp(-x))`).
    pub(crate) fn sigmoid(&self, x: &[f32], fidelity: bool) -> Vec<f32> {
        self.sigmoid_impl(x, fidelity)
            .expect("ds4_metal::sigmoid encoding failed")
    }

    /// Elementwise softplus_sqrt (Metal). `fidelity=true` uses the
    /// antirez ds4.c:4867 piecewise kernel (M4 #308 gate); otherwise
    /// uses the stable-softplus identity. Distinct from the legacy
    /// `softplus_sqrt` shim above which only exposes the default
    /// stable form and operates on float4 lanes.
    pub(crate) fn softplus_sqrt_fidelity(&self, logits: &[f32], fidelity: bool) -> Vec<f32> {
        self.softplus_sqrt_fidelity_impl(logits, fidelity)
            .expect("ds4_metal::softplus_sqrt_fidelity encoding failed")
    }

    /// Speculative moe_routed_step encoding — two-kernel q4_K pipeline:
    ///
    ///   1. `kernel_mul_mv_id_q4_K_pair_swiglu_f32` (M131): paired gate+up
    ///      matvec fused with SwiGLU; writes `dst_mid[slot,row] =
    ///      silu(clamp(gate)) * clamp(up) * route_weight` for all 6 expert
    ///      slots in one dispatch. Reads quantized expert weights, residual
    ///      activation `x`, ids (selected expert per slot), and weights
    ///      (route weights per slot).
    ///
    ///   2. `kernel_mul_mv_id_q4_K_sum6_f32` (M133): per-token sum-of-6
    ///      down-projection. Reads `dst_mid` from stage 1, projects each
    ///      slot's expert FFN output back into d_model space, accumulates
    ///      the 6 contributions into `dst` (the routed residual).
    ///
    /// **Important impedance mismatch with the trait:** `moe_routed_step`'s
    /// signature takes `&[f32]` expert weights, but the registry kernels
    /// take quantized weights (q4_K blocks). This impl therefore ignores
    /// the trait's `experts_w_*` arguments — they would need to come from
    /// a `QuantizedExpertWeights` field on `MetalState`, populated during
    /// model load from a GGUF tensor. The signature here matches what the
    /// real wire-up will look like, but the inputs are accepted as opaque
    /// `&metal::Buffer` references instead of slices.
    ///
    /// Returns `bail!` unless both expert-buffer + pipeline tables are
    /// populated; the production path will replace this with a real call.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn moe_routed_step_impl(
        &self,
        x: &[f32],
        selected: &[i32],
        weights: &[f32],
        experts_w_gate_buf: &metal::Buffer,
        experts_w_up_buf: &metal::Buffer,
        experts_w_down_buf: &metal::Buffer,
        gate_ttype: GgmlType,
        up_ttype: GgmlType,
        down_ttype: GgmlType,
        d_ffn: usize,
    ) -> Result<Vec<f32>> {
        let d_in = x.len();
        let (_cb, dst_final) = self.moe_routed_step_encode(
            None,
            x,
            selected,
            weights,
            experts_w_gate_buf,
            experts_w_up_buf,
            experts_w_down_buf,
            gate_ttype,
            up_ttype,
            down_ttype,
            d_ffn,
            None,
            None,
        )?;
        // _cb is already committed AND waited inside moe_routed_step_encode
        // when called with cmd_buf=None (the legacy single-cb path).
        Ok(unsafe { read_buffer::<f32>(&dst_final, d_in) })
    }

    /// Phase C-B Slice 5-redo (M4 #330p): the encoder body of
    /// `moe_routed_step_impl`, factored to optionally encode into a
    /// caller-provided command buffer. Returns `(cmd_buf, dst_final)` —
    /// the command buffer that the work was encoded into, and an Arc-
    /// clone of the dst_final scratch buffer for late readback.
    ///
    /// - `cmd_buf=None` (legacy path): creates a fresh cb, commits AND
    ///   waits. The caller can read `dst_final` immediately.
    /// - `cmd_buf=Some(cb)`: encodes into the supplied cb, does NOT
    ///   commit or wait. The caller is responsible for committing the cb
    ///   and waiting before reading `dst_final`.
    ///
    /// In the Some path the moe_scratch pool lock is released before the
    /// commit happens (in the caller's frame). This is safe because the
    /// scratch buffers are Arc-counted and retained by the cb's encoder
    /// references; the pool entry's outer reference can be re-used by
    /// the next call without dropping the GPU-visible buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_routed_step_encode(
        &self,
        external_cb: Option<&metal::CommandBufferRef>,
        x: &[f32],
        selected: &[i32],
        weights: &[f32],
        experts_w_gate_buf: &metal::Buffer,
        experts_w_up_buf: &metal::Buffer,
        experts_w_down_buf: &metal::Buffer,
        gate_ttype: GgmlType,
        up_ttype: GgmlType,
        down_ttype: GgmlType,
        d_ffn: usize,
        // M5 task #98: when `Some((ids_buf, weights_buf))`, those GPU
        // buffers (typically produced by `BatchScope::encode_router_finalize`)
        // are bound directly to the moe encoders and the CPU
        // `selected`/`weights` slices are ignored. When `None`, the
        // legacy CPU→scratch copy runs. Eliminates one router→moe
        // CPU round-trip per layer when used.
        router_bufs: Option<(&metal::Buffer, &metal::Buffer)>,
        // M5 task #100: when `Some(buf)`, the moe x input is read
        // directly from this GPU buffer (typically a DeferredBuf from
        // an upstream scope op like `hc_collapse_norm`) and the CPU
        // `x` slice is ignored EXCEPT for its `len()` (d_in). When
        // `None`, the legacy CPU→scratch.x_buf copy runs.
        external_x_buf: Option<&metal::Buffer>,
    ) -> Result<(metal::CommandBuffer, metal::Buffer)> {
        if router_bufs.is_none() {
            anyhow::ensure!(
                selected.len() == 6 && weights.len() == 6,
                "moe_routed_step: DS4 hard-codes 6 active experts"
            );
        }
        let d_in = x.len();
        anyhow::ensure!(
            d_in % 256 == 0,
            "moe_routed_step: d_in ({}) must be divisible by QK_K=256",
            d_in
        );

        let (pair_symbol, pair_block_bytes, pair_nr0) = match (gate_ttype, up_ttype) {
            (GgmlType::IQ2_XXS, GgmlType::IQ2_XXS) => (
                "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32",
                66_u64,
                4_u64,
            ),
            (GgmlType::Q4_K, GgmlType::Q4_K) => {
                ("ds4_kernel_mul_mv_id_q4_K_pair_swiglu_f32", 144_u64, 2_u64)
            }
            _ => anyhow::bail!(
                "moe_routed_step: unsupported gate/up quant types for Metal fast path"
            ),
        };
        let (sum6_symbol, down_block_bytes, down_nr0) = match down_ttype {
            GgmlType::Q2_K => ("ds4_kernel_mul_mv_id_q2_K_sum6_f32", 84_u64, 4_u64),
            GgmlType::Q4_K => ("ds4_kernel_mul_mv_id_q4_K_sum6_f32", 144_u64, 2_u64),
            _ => anyhow::bail!("moe_routed_step: unsupported down quant type for Metal fast path"),
        };

        // MoE kernels are function-constant templates. They are
        // intentionally skipped by the plain pipeline preload and must be
        // specialized at the dispatch shape used below.
        // nsg geometry (tuned 2026-05-31): the nsg sweep
        // (nsg_sweep_moe_attnout_bench) found nsg=4 ~17% faster than the old 2 on
        // the iq2_xxs pair (gate+up) matvec at the real 4096→2048 shape, and the
        // end-to-end decode A/B confirmed nsg=4 net −3.3% ms/token across the real
        // iq2_xxs + q4_K + down mix (rel residual unchanged at 0.236<0.50).
        // DS4_MOE_NSG=2 reverts. The shared value also drives the sum6 down stage.
        let nsg: i16 = std::env::var("DS4_MOE_NSG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let nxpsg: i16 = 4;
        // FC_moe_write_pre_swiglu (FC_MUL_MV + 2): when false (default), the
        // IQ2_XXS pair-swiglu kernel skips the dead dst_gate/dst_up stores
        // (gate/up are consumed in-register by the SwiGLU; only dst_mid is
        // read by sum6). DS4_MOE_WRITE_PRE_SWIGLU=1 re-enables the stores for
        // debug. The Q4_K pair kernel ignores this constant (its matvec
        // template uses dst_gate/up as genuine intermediate scratch).
        let write_pre_swiglu: bool =
            std::env::var("DS4_MOE_WRITE_PRE_SWIGLU").ok().as_deref() == Some("1");
        let mut fc_key = Vec::with_capacity(5);
        fc_key.extend_from_slice(&nsg.to_le_bytes());
        fc_key.extend_from_slice(&nxpsg.to_le_bytes());
        fc_key.push(write_pre_swiglu as u8);
        let populate_mul_mv_fcv = |fcv: &metal::FunctionConstantValuesRef| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(
                &nsg as *const _ as *const _,
                MTLDataType::Short,
                600, // FC_MUL_MV + 0 = FC_mul_mv_nsg
            );
            fcv.set_constant_value_at_index(
                &nxpsg as *const _ as *const _,
                MTLDataType::Short,
                601, // FC_MUL_MV + 1 = FC_mul_mv_nxpsg
            );
            fcv.set_constant_value_at_index(
                &write_pre_swiglu as *const _ as *const _,
                MTLDataType::Bool,
                602, // FC_MUL_MV + 2 = FC_moe_write_pre_swiglu
            );
        };
        let _ds4_moe_trace = std::env::var("DS4_MOE_TRACE").is_ok();

        #[cfg(feature = "metal_capture")]
        let _capture_guard = {
            static MOE_CALL_COUNTER: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let this_call =
                MOE_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let capture_target_call = std::env::var("DS4_MOE_CAPTURE_CALL_N")
                .ok()
                .and_then(|s| s.parse::<u64>().ok());
            if Some(this_call) == capture_target_call {
                let capture_path = std::env::var("DS4_MOE_CAPTURE_PATH")
                    .unwrap_or_else(|_| "/tmp/moe_capture.gputrace".to_string());
                let mgr = metal::CaptureManager::shared();
                let desc = metal::CaptureDescriptor::new();
                desc.set_capture_device(&self.device);
                desc.set_output_url(std::path::PathBuf::from(&capture_path));
                desc.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
                match mgr.start_capture(&desc) {
                    Ok(()) => {
                        eprintln!(
                            "DS4_MOE_CAPTURE: started capture for call #{} → {}",
                            this_call, capture_path
                        );
                        Some((mgr, this_call))
                    }
                    Err(e) => {
                        eprintln!("DS4_MOE_CAPTURE: start_capture failed: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        };

        let t_pipe = std::time::Instant::now();
        let pair_pipe = self.specialized_pipeline(pair_symbol, &fc_key, populate_mul_mv_fcv)?;
        let sum6_pipe = self.specialized_pipeline(sum6_symbol, &fc_key, populate_mul_mv_fcv)?;
        let pipe_us = t_pipe.elapsed().as_micros();

        // ds4_metal_args_mul_mv_id (80 bytes with natural alignment).
        // Field order matches antirez moe.metal:267.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32,
            nei1: i32,
            nbi1: u64,
            ne00: i32,
            ne01: i32,
            ne02: i32,
            _pad0: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            ne0: i32,
            ne1: i32,
            nb1: u64,
            nr0: i32,
            _pad1: i32,
        }
        // ds4_metal_dsv4_moe_swiglu_weight_args (48 bytes).
        // Field order matches antirez moe.metal:113.
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

        const QK_K: i32 = 256;
        let pair_nb_per_row = (d_in as i32 / QK_K) as u64 * pair_block_bytes;

        let t_alloc = std::time::Instant::now();

        // Pooled MoE scratch (M4 #362): the 7 transient buffers per call
        // are allocated lazily once per (d_in, d_ffn) shape and re-used
        // across all 43 layers × N decode tokens. Inputs are re-loaded
        // each call via the buffer's StorageModeShared `contents()` ptr.
        // ⚠ The pool's correctness depends on a per-call wait_until_completed
        // (the GPU must finish reading before the next call's CPU re-load /
        // GPU overwrite). Under async chunk-prefill (no per-layer wait) that
        // guarantee is gone → cross-call race on the shared scratch → NaN. When
        // `moe_scratch_suppress` is set (async path), allocate FRESH per call so
        // each in-flight MoE has private scratch; sync/decode keep the pool.
        let mk = || MoeScratch {
            x_buf: self
                .device
                .new_buffer((d_in * 4) as u64, MTLResourceOptions::StorageModeShared),
            ids_buf: self
                .device
                .new_buffer((6 * std::mem::size_of::<i32>()) as u64, MTLResourceOptions::StorageModeShared),
            weights_buf: self
                .device
                .new_buffer((6 * 4) as u64, MTLResourceOptions::StorageModeShared),
            dst_gate: self
                .device
                .new_buffer((6 * d_ffn * 4) as u64, MTLResourceOptions::StorageModeShared),
            dst_up: self
                .device
                .new_buffer((6 * d_ffn * 4) as u64, MTLResourceOptions::StorageModeShared),
            dst_mid: self
                .device
                .new_buffer((6 * d_ffn * 4) as u64, MTLResourceOptions::StorageModeShared),
            dst_final: self
                .device
                .new_buffer((d_in * 4) as u64, MTLResourceOptions::StorageModeShared),
        };
        let suppress_pool = self
            .moe_scratch_suppress
            .load(std::sync::atomic::Ordering::Relaxed);
        // Both `pool_guard` (the MutexGuard) and `fresh` live to end of function
        // so the &metal::Buffer refs taken from `scratch` below stay valid.
        let mut pool_guard;
        let mut fresh;
        let scratch: &mut MoeScratch = if suppress_pool {
            fresh = mk();
            &mut fresh
        } else {
            pool_guard = self.moe_scratch.lock().expect("moe_scratch mutex");
            pool_guard.entry((d_in, d_ffn)).or_insert_with(mk)
        };
        // Re-load inputs into the pooled buffers (StorageModeShared ⇒
        // CPU and GPU see the same memory; the prior wait_until_completed
        // on the previous call's command buffer guarantees the GPU is no
        // longer reading these buffers).
        unsafe {
            // M5 task #100: only copy x into scratch when the caller
            // didn't provide a GPU buffer for x (DeferredBuf path).
            if external_x_buf.is_none() {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr(),
                    scratch.x_buf.contents() as *mut f32,
                    d_in,
                );
            }
            // M5 task #98: only copy selected/weights into scratch when
            // the caller didn't provide GPU buffers from the router.
            if router_bufs.is_none() {
                std::ptr::copy_nonoverlapping(
                    selected.as_ptr(),
                    scratch.ids_buf.contents() as *mut i32,
                    6,
                );
                std::ptr::copy_nonoverlapping(
                    weights.as_ptr(),
                    scratch.weights_buf.contents() as *mut f32,
                    6,
                );
            }
        }
        let x_buf: &metal::Buffer = external_x_buf.unwrap_or(&scratch.x_buf);
        let (ids_buf, weights_buf): (&metal::Buffer, &metal::Buffer) = match router_bufs {
            Some((ids, w)) => (ids, w),
            None => (&scratch.ids_buf, &scratch.weights_buf),
        };
        let dst_gate = &scratch.dst_gate;
        let dst_up = &scratch.dst_up;
        let dst_mid = &scratch.dst_mid;
        let dst_final = &scratch.dst_final;

        let alloc_us = t_alloc.elapsed().as_micros();

        let pair_args = MulMvIdArgs {
            nei0: 6,
            nei1: 1,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_in as i32,
            ne01: d_ffn as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 2,
            nb01: pair_nb_per_row,
            nb02: pair_nb_per_row * d_ffn as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            ne13: 1,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            ne0: d_ffn as i32,
            ne1: 1,
            nb1: (d_ffn * 4) as u64,
            nr0: pair_nr0 as i32,
            _pad1: 0,
        };
        let act_args = MoeSwigluActArgs {
            width: d_ffn as u32,
            rows: 6,
            gate_row_stride: (d_ffn * 4) as u64,
            up_row_stride: (d_ffn * 4) as u64,
            mid_row_stride: (d_ffn * 4) as u64,
            weight_stride: 4,
            write_clamped: 0,
            clamp_value: f32::INFINITY,
        };

        let t_cmdbuf = std::time::Instant::now();
        // External cb (Some): caller-owned; we encode into it and skip
        // commit/wait/read here (caller's responsibility).
        // None: legacy single-cb path — fresh cb committed + waited here.
        let owned_cb: metal::CommandBuffer;
        let cmd_buf: &metal::CommandBufferRef = if let Some(cb) = external_cb {
            cb
        } else {
            owned_cb = self.command_queue.new_command_buffer().to_owned();
            &owned_cb
        };
        let cmdbuf_us = t_cmdbuf.elapsed().as_micros();

        let t_enc1 = std::time::Instant::now();
        // ── Stage 1: paired gate+up matvec + SwiGLU (6 expert slots) ─────
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pair_pipe);
            set_scalar_bytes(enc, 0, &pair_args);
            set_scalar_bytes(enc, 1, &act_args);
            enc.set_buffer(2, Some(experts_w_gate_buf), 0);
            enc.set_buffer(3, Some(experts_w_up_buf), 0);
            enc.set_buffer(4, Some(x_buf), 0);
            enc.set_buffer(5, Some(dst_gate), 0);
            enc.set_buffer(6, Some(dst_up), 0);
            enc.set_buffer(7, Some(dst_mid), 0);
            enc.set_buffer(8, Some(ids_buf), 0);
            enc.set_buffer(9, Some(weights_buf), 0);
            // threadgroup shmem: simdgroup-matrix scratch (~8 KiB)
            enc.set_threadgroup_memory_length(0, 8192);
            // Grid: (ceil(d_ffn / (NSG*nr0)), 1, nei1*nei0).
            let rows_per_tg = nsg as u64 * pair_nr0;
            let n_row_tg = (d_ffn as u64 + rows_per_tg - 1) / rows_per_tg;
            enc.dispatch_thread_groups(
                MTLSize::new(n_row_tg, 1, 6),
                MTLSize::new(32, nsg as u64, 1), // NW=32
            );
            enc.end_encoding();
        }
        let enc1_us = t_enc1.elapsed().as_micros();

        // ── Stage 2: per-token sum-of-6 down-projection ──────────────────
        let down_args = MulMvIdArgs {
            nei0: 6,
            nei1: 1,
            nbi1: (6 * std::mem::size_of::<i32>()) as u64,
            ne00: d_ffn as i32,
            ne01: d_in as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 2,
            nb01: (d_ffn as i32 / QK_K) as u64 * down_block_bytes,
            nb02: (d_ffn as i32 / QK_K) as u64 * down_block_bytes * d_in as u64,
            ne10: d_ffn as i32,
            ne11: 6,
            ne12: 1,
            ne13: 1,
            nb10: 4,
            nb11: (d_ffn * 4) as u64,
            nb12: (6 * d_ffn * 4) as u64,
            ne0: d_in as i32,
            ne1: 1,
            nb1: (d_in * 4) as u64,
            nr0: down_nr0 as i32,
            _pad1: 0,
        };
        let t_enc2 = std::time::Instant::now();
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&sum6_pipe);
            set_scalar_bytes(enc, 0, &down_args);
            enc.set_buffer(1, Some(experts_w_down_buf), 0);
            enc.set_buffer(2, Some(dst_mid), 0);
            enc.set_buffer(3, Some(dst_final), 0);
            enc.set_buffer(4, Some(ids_buf), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            let rows_per_tg = nsg as u64 * down_nr0;
            let n_row_tg = (d_in as u64 + rows_per_tg - 1) / rows_per_tg;
            enc.dispatch_thread_groups(
                MTLSize::new(n_row_tg, 1, 1), // 1 token, all slots summed inside kernel
                MTLSize::new(32, nsg as u64, 1),
            );
            enc.end_encoding();
        }
        let enc2_us = t_enc2.elapsed().as_micros();

        // Clone the dst_final buffer reference (Arc-bump on the Metal
        // foreign type) so the caller can read it after wait, even after
        // the pool MutexGuard drops at end-of-function. The cb retains
        // its internal reference to the scratch buffers for the duration
        // of execution.
        let dst_final_cloned = dst_final.clone();
        let dst_cb_ref = cmd_buf.to_owned();

        // Legacy path (external_cb=None): commit + wait here so the
        // caller can read immediately. Fused path (Some): caller is
        // responsible for committing + waiting; we just leave the cb
        // populated.
        let mut commit_us: u128 = 0;
        let mut wait_us: u128 = 0;
        let read_us: u128 = 0;
        if external_cb.is_none() {
            let t_commit = std::time::Instant::now();
            cmd_buf.commit();
            commit_us = t_commit.elapsed().as_micros();
            let t_wait = std::time::Instant::now();
            cmd_buf.wait_until_completed();
            wait_us = t_wait.elapsed().as_micros();
        }
        // The `pool_guard` MutexGuard (pooled path only) is held through the
        // whole call and drops at function end: the GPU is reading/writing the
        // cached buffers and another thread mustn't swap entries while a dispatch
        // is in flight. Decode is single-threaded so this is not a perf concern.
        // The suppress path (fresh per-call scratch) holds no lock; `fresh` also
        // drops at function end after the GPU work is committed/waited.

        if _ds4_moe_trace {
            // Query MTLCommandBuffer GPUStartTime / GPUEndTime for the
            // ACTUAL GPU execution span. The difference between
            // wait_us and gpu_us is the scheduling overhead (queue wait,
            // CPU-side completion handling).
            let (gpu_start, gpu_end): (f64, f64) = unsafe {
                use metal::foreign_types::ForeignTypeRef;
                use objc::runtime::Object;
                use objc::{msg_send, sel, sel_impl};
                let cb_ptr: *mut Object = std::mem::transmute(cmd_buf.as_ptr());
                let s: f64 = msg_send![cb_ptr, GPUStartTime];
                let e: f64 = msg_send![cb_ptr, GPUEndTime];
                (s, e)
            };
            let gpu_us = ((gpu_end - gpu_start) * 1_000_000.0).max(0.0) as u64;
            eprintln!(
                "DS4_MOE_TRACE,d_in={},d_ffn={},pipe={},alloc={},cmdbuf={},enc1={},enc2={},commit={},wait={},read={},gpu={},total={}",
                d_in,
                d_ffn,
                pipe_us,
                alloc_us,
                cmdbuf_us,
                enc1_us,
                enc2_us,
                commit_us,
                wait_us,
                read_us,
                gpu_us,
                pipe_us + alloc_us + cmdbuf_us + enc1_us + enc2_us + commit_us + wait_us + read_us,
            );
        }

        // dst_final already contains the weighted sum; apply route weights
        // happens inside the SwiGLU kernel (act_args.weight_stride field
        // makes weights[slot] multiplicative at mid-tensor write-back).

        #[cfg(feature = "metal_capture")]
        if let Some((mgr, this_call)) = _capture_guard {
            mgr.stop_capture();
            eprintln!("DS4_MOE_CAPTURE: stopped capture for call #{}", this_call);
        }

        Ok((dst_cb_ref, dst_final_cloned))
    }

    /// Speculative matvec_f32 encoding via the half-weight registry kernel
    /// `ds4_kernel_mul_mv_f16_f32_4` (M116).
    ///
    /// **f32→f16 cast strategy:** The trait method's `&[f32]` weight slice
    /// is converted to half via `half::f16::from_f32` once per call. In
    /// production this cast must be cached on `MetalState` at model load
    /// time — every layer's projection weights stay resident as f16
    /// buffers for the lifetime of the dispatcher. Doing the cast inside
    /// each call here is wasteful by design (it makes the impl
    /// trait-compatible without a refactor); a follow-up commit will
    /// thread `WeightCache` through and accept `&metal::Buffer` instead.
    ///
    /// **Signature:** registry says 3 bufs + 14 scalars. The struct
    /// `ds4_metal_args_mul_mv` (antirez dense.metal:6) is 19 fields with
    /// padding; we encode it as `#[repr(C)]` and let Rust handle padding.
    /// Kernel signature requires `ne00 % 4 == 0` for the float4 path.
    #[allow(dead_code)]
    pub(crate) fn matvec_f32_impl(&self, w: &[f32], x: &[f32], d_out: usize) -> Result<Vec<f32>> {
        let d_in = x.len();
        anyhow::ensure!(
            w.len() == d_out * d_in,
            "matvec_f32: w.len ({}) != d_out * d_in ({} * {})",
            w.len(),
            d_out,
            d_in
        );
        anyhow::ensure!(
            d_in % 4 == 0,
            "matvec_f32: d_in ({}) must be divisible by 4 (float4 path)",
            d_in
        );

        anyhow::ensure!(
            d_out % 2 == 0,
            "matvec_f32: d_out ({}) must be divisible by NR0=2",
            d_out
        );

        // FC values per antirez ds4_metal_mv_select / mv_ext_nxpsg for n_tok=1:
        //   nsg   = clamp(ceil(d_in/128), 1, 8)
        //   nxpsg = 16 if d_in%256==0, 8 if d_in%128==0, else 4
        let nsg: i16 = {
            let n = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
            n
        };
        let nxpsg: i16 = if d_in % 256 == 0 {
            16
        } else if d_in % 128 == 0 {
            8
        } else {
            4
        };

        // FC cache key = (FC_MUL_MV+0, FC_MUL_MV+1) packed little-endian.
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());

        let pipeline = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(
                &nsg as *const _ as *const _,
                MTLDataType::Short,
                600, // FC_MUL_MV + 0 = FC_mul_mv_nsg
            );
            fcv.set_constant_value_at_index(
                &nxpsg as *const _ as *const _,
                MTLDataType::Short,
                601, // FC_MUL_MV + 1 = FC_mul_mv_nxpsg
            );
        })?;

        // Weight buffer is cached on `MetalState` keyed by (ptr, len) so
        // dense projection weights upload once at first use. Activations
        // stay per-call (their address may be reused with new contents).
        // See `MetalState::cached_weight_buffer` and DS4_WEIGHT_CACHE=0
        // to disable.
        let w_buf = self.cached_weight_buffer(w);
        let x_buf = new_input_buffer(&self.device, x);
        let dst_buf = new_output_buffer::<f32>(&self.device, d_out);

        // ds4_metal_args_mul_mv (dense.metal:6). nb* are byte strides;
        // f32 weights → nb00=4, nb01=d_in*4. ne11=ne12=1 for decode-time
        // single-token matvec; r2/r3 are batch broadcast ratios (both 1).
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
            nb00: 4, // sizeof(float)
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
            nr0: 2, // NR0 — switch picks the case in kernel_mul_mv_t_t_4_disp
            r2: 1,
            r3: 1,
        };

        // Threadgroup memory: helper_mv_reduce_and_write writes
        // NW=32 lanes × NR0=2 × float = 256 B; round up for safety.
        let shmem_bytes: u64 = 32 * 2 * 4;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_bytes);
        // Grid: (ceil(d_out / NR0), 1, 1) groups; threads per group = (NW, NSG, 1).
        let n_row_tg = ((d_out as u64) + 1) / 2;
        enc.dispatch_thread_groups(
            MTLSize::new(n_row_tg, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "matvec_f32_impl");

        Ok(unsafe { read_buffer::<f32>(&dst_buf, d_out) })
    }

    pub(crate) fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        self.matvec_f32_impl(w, x, d_out)
            .expect("ds4_metal::matvec_f32 encoding failed")
    }

    /// M4 #330m — Phase C tail slice. Batch the post-layer-loop tail
    /// (final `rms_norm(γ)` + optional `q8_0_round_trip` +
    /// `matvec_f32(lm_head)`) into ONE `MTLCommandBuffer`. Eliminates 2
    /// commits + 2 waits + 2 readbacks per token vs. the trait path.
    ///
    /// `x` = post-`output_hc_head_one` residual `[d_embd]`.
    /// `gamma` = `final_norm_gamma` `[d_embd]`.
    /// `lm_head` = `[vocab_size * d_embd]` row-major.
    /// `want_q80` = mirrors `DS4_Q8_0_ACT=1` (M4 #299).
    ///
    /// Bit-identical to `rms_norm` → (`q8_0_round_trip`?) → `matvec_f32`
    /// run sequentially; the only difference is fewer GPU↔CPU sync points.
    pub(crate) fn tail_lm_head_batched_impl(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        want_q80: bool,
        lm_head: &[f32],
        vocab_size: usize,
        // When false (greedy decode), only the GPU-argmax token id is read back
        // (4 bytes); the full vocab_size logit vector is NOT copied to CPU. When
        // true (sampling / diagnostics), the full logits are also read back.
        // Returns (argmax_token_id, logits_or_empty).
        want_full_logits: bool,
    ) -> Result<(i32, Vec<f32>)> {
        let d_embd = x.len();
        anyhow::ensure!(
            gamma.len() == d_embd,
            "tail_lm_head_batched: gamma.len ({}) != d_embd ({})",
            gamma.len(),
            d_embd,
        );
        anyhow::ensure!(
            lm_head.len() == vocab_size * d_embd,
            "tail_lm_head_batched: lm_head.len ({}) != vocab_size * d_embd ({} * {})",
            lm_head.len(),
            vocab_size,
            d_embd,
        );
        anyhow::ensure!(
            d_embd % 4 == 0,
            "tail_lm_head_batched: d_embd ({}) must be divisible by 4 (float4)",
            d_embd,
        );
        anyhow::ensure!(
            vocab_size % 2 == 0,
            "tail_lm_head_batched: vocab_size ({}) must be divisible by NR0=2",
            vocab_size,
        );

        // ---- Pipelines ----
        let rms_pipe = self
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?;
        let q80_pipe = if want_q80 {
            Some(
                self.pipelines
                    .get("ds4_q8_0_round_trip_f32")
                    .ok_or_else(|| anyhow::anyhow!("q8_0_round_trip pipeline not loaded"))?,
            )
        } else {
            None
        };

        // matvec_f32 specialization (same logic as matvec_f32_impl).
        let nsg: i16 = (((d_embd as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if d_embd % 256 == 0 {
            16
        } else if d_embd % 128 == 0 {
            8
        } else {
            4
        };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let mv_pipe = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
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

        // ---- Buffers (allocated once) ----
        let x_buf = new_input_buffer(&self.device, x);
        let gamma_buf = new_input_buffer(&self.device, gamma);
        let normed_buf = new_output_buffer::<f32>(&self.device, d_embd);
        // q8_0 round-trip needs a separate output buffer (kernel does not
        // write in-place). If !want_q80, `lm_input_buf` aliases `normed_buf`.
        let q80_out_buf = if want_q80 {
            Some(new_output_buffer::<f32>(&self.device, d_embd))
        } else {
            None
        };
        let logits_buf = new_output_buffer::<f32>(&self.device, vocab_size);
        let token_buf = new_output_buffer::<i32>(&self.device, 1);
        let argmax_pipe =
            self.specialized_pipeline("ds4_argmax_f32", &[], |_fcv| {})?;
        let w_buf = self.cached_weight_buffer(lm_head);

        // ---- Build args for rms_norm pass (verbatim from rms_norm_impl) ----
        let n = d_embd as u32;
        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut rms_args = [0u8; 144];
        rms_args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        rms_args[4..8].copy_from_slice(&((n / 4) as i32).to_le_bytes());
        rms_args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[16..24].copy_from_slice(&(row_bytes * rows as u64).to_le_bytes());
        rms_args[24..32].copy_from_slice(&(row_bytes * rows as u64).to_le_bytes());
        rms_args[32..36].copy_from_slice(&eps.to_le_bytes());
        rms_args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        rms_args[40..44].copy_from_slice(&1i32.to_le_bytes());
        rms_args[44..48].copy_from_slice(&1i32.to_le_bytes());
        rms_args[48..52].copy_from_slice(&1i32.to_le_bytes());
        rms_args[52..56].copy_from_slice(&1i32.to_le_bytes());
        rms_args[56..60].copy_from_slice(&1i32.to_le_bytes());
        rms_args[60..64].copy_from_slice(&1i32.to_le_bytes());
        rms_args[64..68].copy_from_slice(&1i32.to_le_bytes());
        rms_args[68..72].copy_from_slice(&1i32.to_le_bytes());
        rms_args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[96..104].copy_from_slice(&plane.to_le_bytes());
        rms_args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[120..128].copy_from_slice(&plane.to_le_bytes());
        rms_args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        // Threads per group for rms (same as rms_norm_impl).
        let ne00_t = (n / 4) as u64;
        let mut rms_nth: u64 = 32;
        while rms_nth < ne00_t && rms_nth < 1024 {
            rms_nth *= 2;
        }
        if rms_nth > ne00_t {
            rms_nth = ne00_t;
        }
        let rms_nth = rms_nth.max(1);

        // ---- q8_0 dispatch params (only used if want_q80) ----
        let q80_n_elts: u32 = d_embd as u32;
        let q80_n_full_blocks: u32 = (d_embd / 32) as u32;
        let q80_last_block_len: u32 = (d_embd % 32) as u32;
        let q80_n_blocks_total: u32 = q80_n_full_blocks + (q80_last_block_len != 0) as u32;

        // ---- matvec args (verbatim from matvec_f32_impl) ----
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
        let mv_args = MulMvArgs {
            ne00: d_embd as i32,
            ne01: vocab_size as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_embd * 4) as u64,
            nb02: (d_embd * vocab_size * 4) as u64,
            nb03: (d_embd * vocab_size * 4) as u64,
            ne10: d_embd as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_embd * 4) as u64,
            nb12: (d_embd * 4) as u64,
            nb13: (d_embd * 4) as u64,
            ne0: vocab_size as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_shmem_bytes: u64 = 32 * 2 * 4;
        let mv_n_row_tg = ((vocab_size as u64) + 1) / 2;

        // ---- Build ONE command buffer with 2 or 3 encoder passes ----
        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: rms_norm — x, gamma → normed_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(rms_pipe);
            enc.set_bytes(0, rms_args.len() as u64, rms_args.as_ptr() as *const _);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&gamma_buf), 0);
            enc.set_buffer(3, Some(&x_buf), 0); // src1_1 unused for F=2
            enc.set_buffer(4, Some(&normed_buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rms_nth, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2 (optional): q8_0 round-trip — normed_buf → q80_out_buf
        let lm_input_buf: &metal::Buffer = if let (Some(q80_pipe), Some(q80_out_buf)) =
            (q80_pipe, q80_out_buf.as_ref())
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(q80_pipe);
            enc.set_buffer(0, Some(&normed_buf), 0);
            enc.set_buffer(1, Some(q80_out_buf), 0);
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &q80_n_elts as *const _ as *const _,
            );
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &q80_n_full_blocks as *const _ as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &q80_last_block_len as *const _ as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(q80_n_blocks_total as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            end_shared_compute_enc(enc);
            q80_out_buf
        } else {
            &normed_buf
        };

        // Pass 3: matvec_f32 — lm_head × lm_input → logits_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe);
            set_scalar_bytes(enc, 0, &mv_args);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(lm_input_buf), 0);
            enc.set_buffer(3, Some(&logits_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 4: GPU argmax — logits_buf → token_buf (single token id). Always
        // run (one threadgroup, negligible); lets greedy decode skip the full
        // ~vocab_size logit readback.
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&argmax_pipe);
            enc.set_buffer(0, Some(&logits_buf), 0);
            enc.set_buffer(1, Some(&token_buf), 0);
            let n_u = vocab_size as u32;
            set_scalar_bytes(enc, 2, &n_u);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "tail_lm_head_batched_impl");

        let token = unsafe { read_buffer::<i32>(&token_buf, 1) }[0];
        let logits = if want_full_logits {
            unsafe { read_buffer::<f32>(&logits_buf, vocab_size) }
        } else {
            Vec::new()
        };
        Ok((token, logits))
    }

    pub(crate) fn tail_lm_head_batched(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        want_q80: bool,
        lm_head: &[f32],
        vocab_size: usize,
    ) -> Vec<f32> {
        self.tail_lm_head_batched_impl(x, gamma, eps, want_q80, lm_head, vocab_size, true)
            .expect("ds4_metal::tail_lm_head_batched encoding failed")
            .1
    }

    /// Greedy-decode tail: rms_norm → (q8_0?) → lm_head matvec → GPU argmax,
    /// returning just the token id (no full-logit readback). The argmax kernel
    /// matches the CPU `argmax_i32` (lowest index on ties).
    pub(crate) fn tail_lm_head_argmax(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        want_q80: bool,
        lm_head: &[f32],
        vocab_size: usize,
    ) -> i32 {
        self.tail_lm_head_batched_impl(x, gamma, eps, want_q80, lm_head, vocab_size, false)
            .expect("ds4_metal::tail_lm_head_argmax encoding failed")
            .0
    }

    /// Q8_0 greedy-decode tail: `rms_norm → mul_mv_q8_0(lm_head) → GPU argmax`.
    /// `output.weight` is natively Q8_0 in DS4 V4 Flash, so reading the raw q8
    /// bytes (1 byte/weight, ~562 MB) instead of the dequantized f32 (~2.1 GB)
    /// cuts the tail weight read ~3.7×. Token-identical to the f32 tail (the f32
    /// lm_head IS dequant(q8); the q8 matvec computes the same dot). The huge
    /// `d_out=vocab` also gets the correct row-parallel nsg=1 (the f32 path used
    /// the d_in-keyed clamp = nsg=8). No q8_0 activation round-trip: the q8
    /// weight matvec reads the f32 normed activation directly.
    pub(crate) fn tail_lm_head_argmax_q8(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q8: &[u8],
        vocab_size: usize,
    ) -> Result<i32> {
        self.tail_lm_head_argmax_quant(
            x, gamma, eps, lm_head_q8, vocab_size,
            "ds4_kernel_mul_mv_q8_0_f32", 34, false,
        )
        .map(|(t, _)| t)
    }

    /// Full-logits Q8_0 lm-head tail: `rms_norm → mul_mv_q8_0(lm_head_q8) →`
    /// read back all `vocab_size` logits. Lets the sampling decode path drop the
    /// 2.1 GB f32 lm_head (the q8 matvec == dequant(q8) matvec, so the logits
    /// are bit-identical to `tail_lm_head_batched` with want_q80=false — which is
    /// the default, `DS4_Q8_0_ACT` unset).
    pub(crate) fn tail_lm_head_full_q8(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q8: &[u8],
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        self.tail_lm_head_argmax_quant(
            x, gamma, eps, lm_head_q8, vocab_size,
            "ds4_kernel_mul_mv_q8_0_f32", 34, true,
        )
        .map(|(_, logits)| logits)
    }

    /// Q4_0 lm-head tail (DS4_LOW_RAM): identical to the Q8_0 tail but the
    /// weights are `block_q4_0` (18 B/32w, re-quantized from the model's Q8_0
    /// at load) read by `ds4_kernel_mul_mv_q4_0_f32`. Half the resident weight
    /// bytes; token-quality validated by the q4 token-agreement bench.
    pub(crate) fn tail_lm_head_argmax_q4(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q4: &[u8],
        vocab_size: usize,
    ) -> Result<i32> {
        self.tail_lm_head_argmax_quant(
            x, gamma, eps, lm_head_q4, vocab_size,
            "ds4_kernel_mul_mv_q4_0_f32", 18, false,
        )
        .map(|(t, _)| t)
    }

    /// Shared lm-head greedy tail for a block-quantized weight (rms_norm →
    /// `mv_kernel` matvec → GPU argmax → token id, one cb). `block_bytes` is
    /// the on-disk byte size of one 32-weight block (34 for Q8_0, 18 for
    /// Q4_0); `w_bytes` are the raw GGUF/requantized block bytes (no-copy
    /// resident). Token-identical to running rms + matvec + argmax separately.
    pub(crate) fn tail_lm_head_argmax_quant(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        w_bytes: &[u8],
        vocab_size: usize,
        mv_kernel: &str,
        block_bytes: u64,
        // When true, skip the GPU argmax and read back the full vocab logits
        // (returned in `.1`); `.0` is 0. When false, return the argmax token in
        // `.0` and an empty `.1` (no full-vocab readback).
        want_full: bool,
    ) -> Result<(i32, Vec<f32>)> {
        let d_embd = x.len();
        anyhow::ensure!(gamma.len() == d_embd, "tail_quant: gamma len");
        anyhow::ensure!(d_embd % 32 == 0, "tail_quant: d_embd % 32");
        anyhow::ensure!(vocab_size % 2 == 0, "tail_quant: vocab % 2");
        anyhow::ensure!(
            w_bytes.len() as u64 == vocab_size as u64 * (d_embd as u64 / 32) * block_bytes,
            "tail_quant: w bytes {} != vocab*({}/32)*{}",
            w_bytes.len(),
            d_embd,
            block_bytes,
        );

        // Row-parallel nsg (q8 d_out heuristic): vocab is huge → nsg=1.
        let nsg: i16 = if vocab_size >= 8192 {
            1
        } else if vocab_size >= 1024 {
            2
        } else {
            4
        };
        let nxpsg: i16 = if d_embd % 256 == 0 {
            16
        } else if d_embd % 128 == 0 {
            8
        } else {
            4
        };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let mv_pipe = self.specialized_pipeline(mv_kernel, &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;
        let argmax_pipe = self.specialized_pipeline("ds4_argmax_f32", &[], |_fcv| {})?;
        let rms_pipe = self
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?;

        let x_buf = new_input_buffer(&self.device, x);
        let gamma_buf = new_input_buffer(&self.device, gamma);
        let normed_buf = new_output_buffer::<f32>(&self.device, d_embd);
        let logits_buf = new_output_buffer::<f32>(&self.device, vocab_size);
        let token_buf = new_output_buffer::<i32>(&self.device, 1);
        // cached_q8_0_raw_buffer is format-agnostic: a no-copy resident MTLBuffer
        // over the raw block bytes (works for Q4_0 blocks too).
        let w_buf = self.cached_q8_0_raw_buffer(w_bytes);

        // rms_norm args (verbatim from tail_lm_head_batched_impl).
        let n = d_embd as u32;
        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut rms_args = [0u8; 144];
        rms_args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        rms_args[4..8].copy_from_slice(&((n / 4) as i32).to_le_bytes());
        rms_args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[16..24].copy_from_slice(&plane.to_le_bytes());
        rms_args[24..32].copy_from_slice(&plane.to_le_bytes());
        rms_args[32..36].copy_from_slice(&eps.to_le_bytes());
        rms_args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        rms_args[40..44].copy_from_slice(&1i32.to_le_bytes());
        rms_args[44..48].copy_from_slice(&1i32.to_le_bytes());
        rms_args[48..52].copy_from_slice(&1i32.to_le_bytes());
        rms_args[52..56].copy_from_slice(&1i32.to_le_bytes());
        rms_args[56..60].copy_from_slice(&1i32.to_le_bytes());
        rms_args[60..64].copy_from_slice(&1i32.to_le_bytes());
        rms_args[64..68].copy_from_slice(&1i32.to_le_bytes());
        rms_args[68..72].copy_from_slice(&1i32.to_le_bytes());
        rms_args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[96..104].copy_from_slice(&plane.to_le_bytes());
        rms_args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[120..128].copy_from_slice(&plane.to_le_bytes());
        rms_args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[136..144].copy_from_slice(&row_bytes.to_le_bytes());
        let ne00_t = (n / 4) as u64;
        let mut rms_nth: u64 = 32;
        while rms_nth < ne00_t && rms_nth < 1024 {
            rms_nth *= 2;
        }
        if rms_nth > ne00_t {
            rms_nth = ne00_t;
        }
        let rms_nth = rms_nth.max(1);

        // q8 matvec args (same MulMvArgs layout as matvec_q8_0).
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvArgs {
            ne00: i32, ne01: i32, ne02: i32, _pad0: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, _pad1: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
        }
        let row_stride = (d_embd as u64 / 32) * block_bytes;
        let mv_args = MulMvArgs {
            ne00: d_embd as i32, ne01: vocab_size as i32, ne02: 1, _pad0: 0,
            nb00: 2, nb01: row_stride, nb02: row_stride * vocab_size as u64, nb03: row_stride * vocab_size as u64,
            ne10: d_embd as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_embd * 4) as u64, nb12: (d_embd * 4) as u64, nb13: (d_embd * 4) as u64,
            ne0: vocab_size as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };

        let cmd_buf = self.command_queue.new_command_buffer();
        // Pass 1: rms_norm
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(rms_pipe);
            enc.set_bytes(0, rms_args.len() as u64, rms_args.as_ptr() as *const _);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&gamma_buf), 0);
            enc.set_buffer(3, Some(&x_buf), 0);
            enc.set_buffer(4, Some(&normed_buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 4);
            enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(rms_nth, 1, 1));
            end_shared_compute_enc(enc);
        }
        // Pass 2: q8 matvec lm_head × normed → logits
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe);
            set_scalar_bytes(enc, 0, &mv_args);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&normed_buf), 0);
            enc.set_buffer(3, Some(&logits_buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(((vocab_size as u64) + 1) / 2, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }
        // Pass 3: GPU argmax → token id (skipped on the full-logits path).
        if !want_full {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&argmax_pipe);
            enc.set_buffer(0, Some(&logits_buf), 0);
            enc.set_buffer(1, Some(&token_buf), 0);
            let n_u = vocab_size as u32;
            set_scalar_bytes(enc, 2, &n_u);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            end_shared_compute_enc(enc);
        }
        self.commit_wait_traced(cmd_buf, "tail_lm_head_quant");
        if want_full {
            Ok((0, unsafe { read_buffer::<f32>(&logits_buf, vocab_size) }))
        } else {
            Ok((unsafe { read_buffer::<i32>(&token_buf, 1) }[0], Vec::new()))
        }
    }

    /// M4 #330o — Phase C.3a layer-head slice. Batch `matvec_f32(w_qa, x)` →
    /// `rms_norm(qr, gamma_q)` into ONE `MTLCommandBuffer`, returning the
    /// post-rms `qr_normed[n_lora_q]`. Eliminates 1 commit + 1 wait +
    /// 1 readback per layer vs. running `k.matvec_f32` then `k.rms_norm`
    /// sequentially through the trait.
    ///
    /// Bit-identical to the sequential trait calls — same kernels, same
    /// args, same dispatch dimensions — so the 18 fidelity gates and
    /// `--correctness` mode keep their byte-for-byte semantics. The
    /// intermediate `qr` is not read back; the second pass reads it
    /// directly from device memory.
    ///
    /// Inputs:
    /// - `x[d_in]` — `normed_q` (post-`hc_collapse_norm` + optional q80
    ///   round-trip on host); identical layout to what `matvec_f32`
    ///   already accepts.
    /// - `w_qa[n_lora_q * d_in]` — the `attn_q_a` projection weights.
    /// - `gamma_q[n_lora_q]` — `qkv_gamma_q`.
    /// - `eps_rms` — `params.rms_eps`.
    ///
    /// Returns `qr_normed[n_lora_q]`.
    pub(crate) fn layer_qa_rms_batched_impl(
        &self,
        x: &[f32],
        w_qa: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        eps_rms: f32,
    ) -> Result<Vec<f32>> {
        let d_in = x.len();
        anyhow::ensure!(
            w_qa.len() == n_lora_q * d_in,
            "layer_qa_rms_batched: w_qa.len ({}) != n_lora_q * d_in ({} * {})",
            w_qa.len(),
            n_lora_q,
            d_in,
        );
        anyhow::ensure!(
            gamma_q.len() == n_lora_q,
            "layer_qa_rms_batched: gamma_q.len ({}) != n_lora_q ({})",
            gamma_q.len(),
            n_lora_q,
        );
        anyhow::ensure!(
            d_in % 4 == 0,
            "layer_qa_rms_batched: d_in ({}) must be divisible by 4 (float4)",
            d_in,
        );
        anyhow::ensure!(
            n_lora_q % 2 == 0,
            "layer_qa_rms_batched: n_lora_q ({}) must be divisible by NR0=2",
            n_lora_q,
        );
        anyhow::ensure!(
            n_lora_q % 4 == 0,
            "layer_qa_rms_batched: n_lora_q ({}) must be divisible by 4 (rms_norm float4 lanes)",
            n_lora_q,
        );

        // ---- Matvec pipeline + specialization (same as matvec_f32_impl) ----
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
        let mv_pipe = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
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

        // ---- RMS pipeline ----
        let rms_pipe = self
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?;

        // ---- Buffers ----
        let w_buf = self.cached_weight_buffer(w_qa);
        let x_buf = new_input_buffer(&self.device, x);
        let qr_buf = new_output_buffer::<f32>(&self.device, n_lora_q);
        let gamma_buf = new_input_buffer(&self.device, gamma_q);
        let out_buf = new_output_buffer::<f32>(&self.device, n_lora_q);

        // ---- Matvec args (same as matvec_f32_impl) ----
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
        let mv_args = MulMvArgs {
            ne00: d_in as i32,
            ne01: n_lora_q as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_in * 4) as u64,
            nb02: (d_in * n_lora_q * 4) as u64,
            nb03: (d_in * n_lora_q * 4) as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            nb13: (d_in * 4) as u64,
            ne0: n_lora_q as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_shmem_bytes: u64 = 32 * 2 * 4;
        let mv_n_row_tg = ((n_lora_q as u64) + 1) / 2;

        // ---- RMS args (same as rms_norm_impl, sized for n_lora_q) ----
        let n = n_lora_q as u32;
        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut rms_args = [0u8; 144];
        rms_args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        rms_args[4..8].copy_from_slice(&((n / 4) as i32).to_le_bytes());
        rms_args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[16..24].copy_from_slice(&plane.to_le_bytes());
        rms_args[24..32].copy_from_slice(&plane.to_le_bytes());
        rms_args[32..36].copy_from_slice(&eps_rms.to_le_bytes());
        rms_args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        rms_args[40..44].copy_from_slice(&1i32.to_le_bytes());
        rms_args[44..48].copy_from_slice(&1i32.to_le_bytes());
        rms_args[48..52].copy_from_slice(&1i32.to_le_bytes());
        rms_args[52..56].copy_from_slice(&1i32.to_le_bytes());
        rms_args[56..60].copy_from_slice(&1i32.to_le_bytes());
        rms_args[60..64].copy_from_slice(&1i32.to_le_bytes());
        rms_args[64..68].copy_from_slice(&1i32.to_le_bytes());
        rms_args[68..72].copy_from_slice(&1i32.to_le_bytes());
        rms_args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[96..104].copy_from_slice(&plane.to_le_bytes());
        rms_args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[120..128].copy_from_slice(&plane.to_le_bytes());
        rms_args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        let ne00_t = (n / 4) as u64;
        let mut rms_nth: u64 = 32;
        while rms_nth < ne00_t && rms_nth < 1024 {
            rms_nth *= 2;
        }
        if rms_nth > ne00_t {
            rms_nth = ne00_t;
        }
        let rms_nth = rms_nth.max(1);

        // ---- Single command buffer, two passes ----
        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: matvec_f32 — w_qa × x → qr_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe);
            set_scalar_bytes(enc, 0, &mv_args);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&qr_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2: rms_norm_mul — qr_buf, gamma_buf → out_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(rms_pipe);
            enc.set_bytes(0, rms_args.len() as u64, rms_args.as_ptr() as *const _);
            enc.set_buffer(1, Some(&qr_buf), 0);
            enc.set_buffer(2, Some(&gamma_buf), 0);
            enc.set_buffer(3, Some(&qr_buf), 0); // src1_1 unused for F=2
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rms_nth, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "layer_qa_rms_batched_impl");

        Ok(unsafe { read_buffer::<f32>(&out_buf, n_lora_q) })
    }

    pub(crate) fn layer_qa_rms_batched(
        &self,
        x: &[f32],
        w_qa: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.layer_qa_rms_batched_impl(x, w_qa, gamma_q, n_lora_q, eps_rms)
            .expect("ds4_metal::layer_qa_rms_batched encoding failed")
    }

    /// M4 #330o — Phase C.3b post-attention chain batched primitive.
    ///
    /// Packs the shared-expert FFN body — `matvec_f32(w_gate)`,
    /// `matvec_f32(w_up)`, fused SwiGLU (silu(g)*u), optional `q8_0`
    /// activation round-trip, and `matvec_f32(w_down)` — into ONE
    /// MTLCommandBuffer with a single readback. Saves 3-4 commit+wait+
    /// readback round-trips per layer relative to the trait path.
    ///
    /// `want_q80 = true` mirrors `DS4_Q8_0_ACT=1` on the trait path:
    /// `shared_mid` is round-tripped through q8_0 before the down matvec.
    ///
    /// Algebraically equivalent to the trait-path sequence (matvec_f32
    /// gate / matvec_f32 up / `shared_expert_swiglu(g,u)` /
    /// optional `q8_0_round_trip` / matvec_f32 down) under
    /// DS4_SILU_FIDELITY=0 (the default — the fused SwiGLU pass uses
    /// the default-branch silu identity `g / (1 + exp(-g))`, matching
    /// `kernel_swiglu_f32` in antirez `metal/glu.metal`).
    ///
    /// NOT bit-identical to the trait reference: the trait path runs
    /// `shared_expert_swiglu` on CPU (Rust `f32::exp`), whereas this
    /// fused chain runs the GPU `kernel_swiglu_f32` (Metal `exp`).
    /// Rust and Metal don't ship the same `expf` implementation, so the
    /// SwiGLU output diverges in the last mantissa bit; the down-matvec
    /// then amplifies that. Smoke tests compare with absolute tolerance
    /// rather than `to_bits`.
    ///
    /// **Not yet wired into `decode_attn_ffn_post_with`.** A separate
    /// landing (C.3b.2) routes the helper through this primitive when
    /// DS4_SILU_FIDELITY is off; the fidelity path stays on the
    /// sequential trait calls.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shared_chain_batched_impl(
        &self,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Result<Vec<f32>> {
        let d_embd = ffn_norm.len();
        let (_cb, out_buf) = self.shared_chain_batched_encode(
            None, ffn_norm, w_gate, w_up, w_down, shared_dim, want_q80, None, None, None, None,
        )?;
        Ok(unsafe { read_buffer::<f32>(&out_buf, d_embd) })
    }

    /// Phase C-B Slice 5-redo: encoder body of `shared_chain_batched_impl`,
    /// factored to optionally encode into a caller-provided cb.
    /// Same Some/None semantics as `moe_routed_step_encode`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shared_chain_batched_encode(
        &self,
        external_cb: Option<&metal::CommandBufferRef>,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
        // M5 task #100: when `Some(buf)`, read ffn_norm directly from this
        // GPU buffer (typically a DeferredBuf from upstream
        // `hc_collapse_norm`) instead of uploading the CPU `ffn_norm`
        // slice fresh each call. The slice's `len()` is still used for
        // shape (d_embd); its contents are ignored when external_norm_buf
        // is Some.
        external_norm_buf: Option<&metal::Buffer>,
        // Raw GGUF block_q8_0 bytes for gate/up/down (owned, page-aligned). When
        // all three are Some, the gate/up/down matvecs read Q8_0 weights at
        // 1 byte/weight (no-copy resident buffer) instead of dequant→f32 — the
        // shared-expert slice of the harness's router_moe_shared cost. swiglu and
        // structure are unchanged. Orthogonal to `want_q80` (activation requant).
        w_gate_q8: Option<&[u8]>,
        w_up_q8: Option<&[u8]>,
        w_down_q8: Option<&[u8]>,
    ) -> Result<(metal::CommandBuffer, metal::Buffer)> {
        let d_embd = ffn_norm.len();
        let sd = shared_dim as usize;
        let use_w_q8 = w_gate_q8.is_some() && w_up_q8.is_some() && w_down_q8.is_some();
        if use_w_q8 {
            anyhow::ensure!(
                d_embd % 32 == 0 && sd % 32 == 0,
                "shared_chain_batched: Q8_0 weights need d_embd ({}) and shared_dim ({}) divisible by 32",
                d_embd,
                sd,
            );
        }
        anyhow::ensure!(
            sd > 0,
            "shared_chain_batched: shared_dim ({}) must be > 0",
            sd
        );
        // f32 length checks apply only to the f32 path; on the q8 path the f32
        // slices may be empty (freed by `free_dead_f32_weights` — the q8 buffers
        // carry the data). Skip them when `use_w_q8`.
        if !use_w_q8 {
            anyhow::ensure!(
                w_gate.len() == sd * d_embd,
                "shared_chain_batched: w_gate.len ({}) != shared_dim * d_embd ({} * {})",
                w_gate.len(),
                sd,
                d_embd
            );
            anyhow::ensure!(
                w_up.len() == sd * d_embd,
                "shared_chain_batched: w_up.len ({}) != shared_dim * d_embd ({} * {})",
                w_up.len(),
                sd,
                d_embd
            );
            anyhow::ensure!(
                w_down.len() == d_embd * sd,
                "shared_chain_batched: w_down.len ({}) != d_embd * shared_dim ({} * {})",
                w_down.len(),
                d_embd,
                sd
            );
        }
        anyhow::ensure!(
            d_embd % 4 == 0,
            "shared_chain_batched: d_embd ({}) must be divisible by 4 (float4)",
            d_embd
        );
        anyhow::ensure!(
            sd % 4 == 0,
            "shared_chain_batched: shared_dim ({}) must be divisible by 4 (float4 down-side)",
            sd
        );
        anyhow::ensure!(
            sd % 2 == 0,
            "shared_chain_batched: shared_dim ({}) must be divisible by NR0=2",
            sd
        );
        anyhow::ensure!(
            d_embd % 2 == 0,
            "shared_chain_batched: d_embd ({}) must be divisible by NR0=2",
            d_embd
        );
        if want_q80 {
            anyhow::ensure!(
                sd % 32 == 0,
                "shared_chain_batched: shared_dim ({}) must be divisible by QK8_0=32 when want_q80",
                sd
            );
        }

        // ---- Pipelines ----
        // matvec specialization for the gate/up pair (ne00 = d_embd).
        let nsg_gu: i16 = (((d_embd as u64 + 127) / 128).clamp(1, 8)) as i16;
        // Same nxpsg heuristic for both kernels — the q8 path supports
        // FC_mul_mv_nxpsg=16/8/4 via dense.metal's simd-reduction branches.
        let nxpsg_gu: i16 = if d_embd % 256 == 0 {
            16
        } else if d_embd % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_gu = Vec::with_capacity(4);
        key_gu.extend_from_slice(&nsg_gu.to_le_bytes());
        key_gu.extend_from_slice(&nxpsg_gu.to_le_bytes());
        let gu_kernel = if use_w_q8 {
            "ds4_kernel_mul_mv_q8_0_f32"
        } else {
            "ds4_kernel_mul_mv_f32_f32_4"
        };
        let mv_pipe_gu =
            self.specialized_pipeline(gu_kernel, &key_gu, |fcv| {
                use metal::MTLDataType;
                fcv.set_constant_value_at_index(
                    &nsg_gu as *const _ as *const _,
                    MTLDataType::Short,
                    600,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg_gu as *const _ as *const _,
                    MTLDataType::Short,
                    601,
                );
            })?;

        // matvec specialization for the down pair (ne00 = sd).
        let nsg_d: i16 = (((sd as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_d: i16 = if sd % 256 == 0 {
            16
        } else if sd % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_d = Vec::with_capacity(4);
        key_d.extend_from_slice(&nsg_d.to_le_bytes());
        key_d.extend_from_slice(&nxpsg_d.to_le_bytes());
        let down_kernel = if use_w_q8 {
            "ds4_kernel_mul_mv_q8_0_f32"
        } else {
            "ds4_kernel_mul_mv_f32_f32_4"
        };
        let mv_pipe_d = self.specialized_pipeline(down_kernel, &key_d, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(
                &nsg_d as *const _ as *const _,
                MTLDataType::Short,
                600,
            );
            fcv.set_constant_value_at_index(
                &nxpsg_d as *const _ as *const _,
                MTLDataType::Short,
                601,
            );
        })?;

        let glu_pipe = self
            .pipelines
            .get("ds4_kernel_swiglu_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_kernel_swiglu_f32 pipeline not loaded"))?
            .clone();
        let q80_pipe_opt = if want_q80 {
            Some(
                self.pipelines
                    .get("ds4_q8_0_round_trip_f32")
                    .ok_or_else(|| anyhow::anyhow!("ds4_q8_0_round_trip_f32 pipeline not loaded"))?
                    .clone(),
            )
        } else {
            None
        };

        // ---- Buffers ----
        // Q8_0 weights bind a no-copy resident buffer over the raw GGUF bytes;
        // the f32 path uploads/caches the dequantized weights.
        let w_gate_buf = match w_gate_q8 {
            Some(b) if use_w_q8 => self.cached_q8_0_raw_buffer(b),
            _ => self.cached_weight_buffer(w_gate),
        };
        let w_up_buf = match w_up_q8 {
            Some(b) if use_w_q8 => self.cached_q8_0_raw_buffer(b),
            _ => self.cached_weight_buffer(w_up),
        };
        let w_down_buf = match w_down_q8 {
            Some(b) if use_w_q8 => self.cached_q8_0_raw_buffer(b),
            _ => self.cached_weight_buffer(w_down),
        };
        // M5 task #100: when the caller supplies a GPU norm buffer
        // (DeferredBuf-fed bridge), skip the CPU upload and bind it
        // directly. The owned_norm holds the fresh buffer in the
        // None path so its lifetime covers the encoder dispatch.
        let owned_norm: Option<metal::Buffer> = if external_norm_buf.is_none() {
            Some(new_input_buffer(&self.device, ffn_norm))
        } else {
            None
        };
        let ffn_buf: &metal::Buffer = match (external_norm_buf, owned_norm.as_ref()) {
            (Some(b), _) => b,
            (None, Some(owned)) => owned,
            (None, None) => unreachable!(),
        };
        let g_buf = new_output_buffer::<f32>(&self.device, sd);
        let u_buf = new_output_buffer::<f32>(&self.device, sd);
        let mid_buf = new_output_buffer::<f32>(&self.device, sd);
        let mid_q80_buf = if want_q80 {
            Some(new_output_buffer::<f32>(&self.device, sd))
        } else {
            None
        };
        let out_buf = new_output_buffer::<f32>(&self.device, d_embd);

        // ---- Matvec args ----
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
        // gate / up: w (sd × d_embd) · ffn_norm (d_embd) → out (sd)
        let gu_args = MulMvArgs {
            ne00: d_embd as i32,
            ne01: sd as i32,
            ne02: 1,
            _pad0: 0,
            nb00: if use_w_q8 { 2 } else { 4 },
            nb01: if use_w_q8 { ((d_embd / 32) * 34) as u64 } else { (d_embd * 4) as u64 },
            nb02: (d_embd * sd * 4) as u64,
            nb03: (d_embd * sd * 4) as u64,
            ne10: d_embd as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_embd * 4) as u64,
            nb12: (d_embd * 4) as u64,
            nb13: (d_embd * 4) as u64,
            ne0: sd as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let gu_shmem_bytes: u64 = 32 * 2 * 4;
        let gu_n_row_tg = ((sd as u64) + 1) / 2;

        // down: w (d_embd × sd) · mid (sd) → out (d_embd)
        let down_args = MulMvArgs {
            ne00: sd as i32,
            ne01: d_embd as i32,
            ne02: 1,
            _pad0: 0,
            nb00: if use_w_q8 { 2 } else { 4 },
            nb01: if use_w_q8 { ((sd / 32) * 34) as u64 } else { (sd * 4) as u64 },
            nb02: (sd * d_embd * 4) as u64,
            nb03: (sd * d_embd * 4) as u64,
            ne10: sd as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (sd * 4) as u64,
            nb12: (sd * 4) as u64,
            nb13: (sd * 4) as u64,
            ne0: d_embd as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let down_shmem_bytes: u64 = 32 * 2 * 4;
        let down_n_row_tg = ((d_embd as u64) + 1) / 2;

        // ---- SwiGLU args (antirez `ds4_metal_args_glu`) ----
        // Single row, sd elements, no offsets. nb01/nb11/nb1 are zero so
        // tgpig=0 picks the only row; ne0 is sd; i00/i10 zero.
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct GluArgs {
            ne00: i32,
            nb01: u64,
            ne10: i32,
            nb11: u64,
            ne0: i32,
            nb1: u64,
            i00: i32,
            i10: i32,
            alpha: f32,
            limit: f32,
        }
        let glu_args = GluArgs {
            ne00: sd as i32,
            nb01: (sd * 4) as u64,
            ne10: sd as i32,
            nb11: (sd * 4) as u64,
            ne0: sd as i32,
            nb1: (sd * 4) as u64,
            i00: 0,
            i10: 0,
            alpha: 0.0,
            limit: 0.0,
        };
        let glu_threads = (glu_pipe.max_total_threads_per_threadgroup() as u64)
            .min(sd as u64)
            .max(1);

        // ---- Q8_0 round-trip args (if requested) ----
        let q80_dispatch = if want_q80 {
            let n_elts: u32 = sd as u32;
            let n_full_blocks: u32 = (sd / 32) as u32;
            let last_block_len: u32 = (sd % 32) as u32;
            let n_blocks_total: u32 = n_full_blocks + (last_block_len != 0) as u32;
            Some((n_elts, n_full_blocks, last_block_len, n_blocks_total))
        } else {
            None
        };

        // ---- Command buffer: caller-supplied (fused) or fresh (legacy) ----
        let owned_cb: metal::CommandBuffer;
        let cmd_buf: &metal::CommandBufferRef = if let Some(cb) = external_cb {
            cb
        } else {
            owned_cb = self.command_queue.new_command_buffer().to_owned();
            &owned_cb
        };

        // Pass 1: matvec — w_gate × ffn_norm → g_buf
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&mv_pipe_gu);
            set_scalar_bytes(enc, 0, &gu_args);
            enc.set_buffer(1, Some(&w_gate_buf), 0);
            enc.set_buffer(2, Some(ffn_buf), 0);
            enc.set_buffer(3, Some(&g_buf), 0);
            enc.set_threadgroup_memory_length(0, gu_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(gu_n_row_tg, 1, 1),
                MTLSize::new(32, nsg_gu as u64, 1),
            );
            enc.end_encoding();
        }

        // Pass 2: matvec — w_up × ffn_norm → u_buf
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&mv_pipe_gu);
            set_scalar_bytes(enc, 0, &gu_args);
            enc.set_buffer(1, Some(&w_up_buf), 0);
            enc.set_buffer(2, Some(ffn_buf), 0);
            enc.set_buffer(3, Some(&u_buf), 0);
            enc.set_threadgroup_memory_length(0, gu_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(gu_n_row_tg, 1, 1),
                MTLSize::new(32, nsg_gu as u64, 1),
            );
            enc.end_encoding();
        }

        // Pass 3: SwiGLU — silu(g) * u → mid_buf
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&glu_pipe);
            set_scalar_bytes(enc, 0, &glu_args);
            enc.set_buffer(1, Some(&g_buf), 0);
            enc.set_buffer(2, Some(&u_buf), 0);
            enc.set_buffer(3, Some(&mid_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(glu_threads, 1, 1),
            );
            enc.end_encoding();
        }

        // Pass 4 (optional): q8_0 round-trip — mid_buf → mid_q80_buf
        let down_src = if let (Some(q80_pipe), Some((n_elts, n_full_blocks, last_block_len, n_blocks_total)), Some(mid_q80)) =
            (q80_pipe_opt.as_ref(), q80_dispatch.as_ref(), mid_q80_buf.as_ref())
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(q80_pipe);
            enc.set_buffer(0, Some(&mid_buf), 0);
            enc.set_buffer(1, Some(mid_q80), 0);
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                n_elts as *const u32 as *const _,
            );
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                n_full_blocks as *const u32 as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                last_block_len as *const u32 as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(*n_blocks_total as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            enc.end_encoding();
            mid_q80
        } else {
            &mid_buf
        };

        // Pass 5: matvec — w_down × down_src → out_buf
        {
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&mv_pipe_d);
            set_scalar_bytes(enc, 0, &down_args);
            enc.set_buffer(1, Some(&w_down_buf), 0);
            enc.set_buffer(2, Some(down_src), 0);
            enc.set_buffer(3, Some(&out_buf), 0);
            enc.set_threadgroup_memory_length(0, down_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(down_n_row_tg, 1, 1),
                MTLSize::new(32, nsg_d as u64, 1),
            );
            enc.end_encoding();
        }

        let _ds4_ffn_trace = std::env::var("DS4_FFN_TRACE").is_ok();
        let out_buf_cloned = out_buf.clone();
        let cb_cloned = cmd_buf.to_owned();
        if external_cb.is_none() {
            let t_commit = std::time::Instant::now();
            cmd_buf.commit();
            let commit_us = t_commit.elapsed().as_micros();
            let t_wait = std::time::Instant::now();
            cmd_buf.wait_until_completed();
            let wait_us = t_wait.elapsed().as_micros();
            if _ds4_ffn_trace {
                eprintln!(
                    "DS4_FFN_TRACE,d_embd={},sd={},q80={},commit={},wait={}",
                    d_embd, sd, want_q80 as i32, commit_us, wait_us
                );
            }
        }
        Ok((cb_cloned, out_buf_cloned))
    }

    /// Phase C-B Slice 5-redo (M4 #330p): encode `moe_routed_step` and
    /// `shared_chain_batched` into the SAME `MTLCommandBuffer`, with one
    /// commit and one `wait_until_completed` total. Saves one
    /// commit+wait+read round-trip per layer per token vs running the
    /// two ops sequentially (each was 1 cb + 1 wait under the trait
    /// path). The new understanding from `DS4_MOE_TRACE` is that the
    /// dominant cost is the per-cb `wait_until_completed` latency
    /// (~25-100 ms median, GPU compute itself is ~300-400 us).
    ///
    /// Both ops share the same input `ffn_normed` in production but
    /// the signature accepts them separately for flexibility.
    ///
    /// Returns `(moe_out, shared_out)` — algebraically identical to
    /// `(moe_routed_step_impl(...), shared_chain_batched_impl(...))`,
    /// bit-identical at the kernel level (same MSL, same FCs, same
    /// args; only the cb wrapping differs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_shared_chain_batched_impl(
        &self,
        // moe inputs
        moe_x: &[f32],
        moe_selected: &[i32],
        moe_weights: &[f32],
        experts_w_gate_buf: &metal::Buffer,
        experts_w_up_buf: &metal::Buffer,
        experts_w_down_buf: &metal::Buffer,
        gate_ttype: GgmlType,
        up_ttype: GgmlType,
        down_ttype: GgmlType,
        d_ffn: usize,
        // shared_chain inputs
        sh_ffn_norm: &[f32],
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d_in = moe_x.len();
        let d_embd = sh_ffn_norm.len();

        // Open ONE cmd_buf, encode both ops into it, commit + wait once.
        let cmd_buf = self.command_queue.new_command_buffer();
        let (_cb_moe, moe_dst) = self.moe_routed_step_encode(
            Some(cmd_buf),
            moe_x,
            moe_selected,
            moe_weights,
            experts_w_gate_buf,
            experts_w_up_buf,
            experts_w_down_buf,
            gate_ttype,
            up_ttype,
            down_ttype,
            d_ffn,
            None,
            None,
        )?;
        let (_cb_sh, shared_dst) = self.shared_chain_batched_encode(
            Some(cmd_buf),
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
        self.commit_wait_traced(cmd_buf, "moe_shared_chain_batched_impl");
        let moe_out = unsafe { read_buffer::<f32>(&moe_dst, d_in) };
        let shared_out = unsafe { read_buffer::<f32>(&shared_dst, d_embd) };
        Ok((moe_out, shared_out))
    }

    pub(crate) fn shared_chain_batched(
        &self,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Vec<f32> {
        self.shared_chain_batched_impl(ffn_norm, w_gate, w_up, w_down, shared_dim, want_q80)
            .expect("ds4_metal::shared_chain_batched encoding failed")
    }

    /// M4 #330o — Phase C.3c output-head batched primitive.
    ///
    /// Packs `rms_norm(unit gamma, eps_rms)` on the full `hc_dim = n_hc *
    /// d_embd` input followed by `matvec_f32(fn_w, flat, n_hc)` into ONE
    /// `MTLCommandBuffer` with a single readback. Returns the `pre[n_hc]`
    /// vector that callers feed into the sigmoid + weighted-sum post-amble
    /// implemented in `decode_step::output_hc_head_one`. Saves one
    /// commit+wait+readback round-trip per token (rms_norm + matvec collapse
    /// from two cmd buffers into one).
    ///
    /// Bit-identical to running `rms_norm(inp_hc, &[1.0; hc_dim], eps_rms)`
    /// then `matvec_f32(fn_w, &flat, n_hc)` sequentially through THIS
    /// dispatcher — same kernels, same args, same dispatch dimensions.
    /// The intermediate `flat` stays resident in GPU memory and is never
    /// read back.
    ///
    /// Inputs:
    /// - `inp_hc[n_hc * d_embd]` — un-normalised HC residual.
    /// - `fn_w[(n_hc * d_embd) * n_hc]` — `output_hc_fn` weights.
    ///
    /// Returns `pre[n_hc]`.
    pub(crate) fn output_hc_head_batched_impl(
        &self,
        inp_hc: &[f32],
        fn_w: &[f32],
        n_hc: usize,
        d_embd: usize,
        eps_rms: f32,
    ) -> Result<Vec<f32>> {
        let hc_dim = n_hc * d_embd;
        anyhow::ensure!(
            inp_hc.len() == hc_dim,
            "output_hc_head_batched: inp_hc.len ({}) != n_hc * d_embd ({} * {})",
            inp_hc.len(),
            n_hc,
            d_embd
        );
        anyhow::ensure!(
            fn_w.len() == hc_dim * n_hc,
            "output_hc_head_batched: fn_w.len ({}) != hc_dim * n_hc ({} * {})",
            fn_w.len(),
            hc_dim,
            n_hc
        );
        anyhow::ensure!(
            hc_dim % 4 == 0,
            "output_hc_head_batched: hc_dim ({}) must be divisible by 4 (float4)",
            hc_dim
        );
        anyhow::ensure!(
            n_hc % 2 == 0,
            "output_hc_head_batched: n_hc ({}) must be divisible by NR0=2",
            n_hc
        );

        // ---- Matvec pipeline + specialization (ne00 = hc_dim) ----
        let nsg: i16 = (((hc_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg: i16 = if hc_dim % 256 == 0 {
            16
        } else if hc_dim % 128 == 0 {
            8
        } else {
            4
        };
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let mv_pipe = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
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

        // ---- RMS pipeline ----
        let rms_pipe = self
            .pipelines
            .get("ds4_kernel_rms_norm_mul_f32_4")
            .ok_or_else(|| anyhow::anyhow!("rms_norm_mul pipeline not loaded"))?;

        // ---- Buffers ----
        let w_buf = self.cached_weight_buffer(fn_w);
        let inp_buf = new_input_buffer(&self.device, inp_hc);
        let unit_gamma: Vec<f32> = vec![1.0f32; hc_dim];
        let gamma_buf = new_input_buffer(&self.device, &unit_gamma);
        let flat_buf = new_output_buffer::<f32>(&self.device, hc_dim);
        let pre_buf = new_output_buffer::<f32>(&self.device, n_hc);

        // ---- RMS args (mirrors rms_norm_impl; sized for hc_dim) ----
        let n = hc_dim as u32;
        let rows: u32 = 1;
        let row_bytes = (n as u64) * 4;
        let plane = row_bytes * rows as u64;
        let mut rms_args = [0u8; 144];
        rms_args[0..4].copy_from_slice(&(n as i32).to_le_bytes());
        rms_args[4..8].copy_from_slice(&((n / 4) as i32).to_le_bytes());
        rms_args[8..16].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[16..24].copy_from_slice(&plane.to_le_bytes());
        rms_args[24..32].copy_from_slice(&plane.to_le_bytes());
        rms_args[32..36].copy_from_slice(&eps_rms.to_le_bytes());
        rms_args[36..40].copy_from_slice(&(rows as i32).to_le_bytes());
        rms_args[40..44].copy_from_slice(&1i32.to_le_bytes());
        rms_args[44..48].copy_from_slice(&1i32.to_le_bytes());
        rms_args[48..52].copy_from_slice(&1i32.to_le_bytes());
        rms_args[52..56].copy_from_slice(&1i32.to_le_bytes());
        rms_args[56..60].copy_from_slice(&1i32.to_le_bytes());
        rms_args[60..64].copy_from_slice(&1i32.to_le_bytes());
        rms_args[64..68].copy_from_slice(&1i32.to_le_bytes());
        rms_args[68..72].copy_from_slice(&1i32.to_le_bytes());
        rms_args[72..80].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[80..88].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[88..96].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[96..104].copy_from_slice(&plane.to_le_bytes());
        rms_args[104..112].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[112..120].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[120..128].copy_from_slice(&plane.to_le_bytes());
        rms_args[128..136].copy_from_slice(&row_bytes.to_le_bytes());
        rms_args[136..144].copy_from_slice(&row_bytes.to_le_bytes());

        let ne00_t = (n / 4) as u64;
        let mut rms_nth: u64 = 32;
        while rms_nth < ne00_t && rms_nth < 1024 {
            rms_nth *= 2;
        }
        if rms_nth > ne00_t {
            rms_nth = ne00_t;
        }
        let rms_nth = rms_nth.max(1);

        // ---- Matvec args (fn_w (n_hc × hc_dim) · flat (hc_dim) → pre (n_hc)) ----
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
        let mv_args = MulMvArgs {
            ne00: hc_dim as i32,
            ne01: n_hc as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (hc_dim * 4) as u64,
            nb02: (hc_dim * n_hc * 4) as u64,
            nb03: (hc_dim * n_hc * 4) as u64,
            ne10: hc_dim as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (hc_dim * 4) as u64,
            nb12: (hc_dim * 4) as u64,
            nb13: (hc_dim * 4) as u64,
            ne0: n_hc as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_shmem_bytes: u64 = 32 * 2 * 4;
        let mv_n_row_tg = ((n_hc as u64) + 1) / 2;

        // ---- Single command buffer, two passes ----
        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: rms_norm_mul — inp_buf, gamma_buf (unit) → flat_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(rms_pipe);
            enc.set_bytes(0, rms_args.len() as u64, rms_args.as_ptr() as *const _);
            enc.set_buffer(1, Some(&inp_buf), 0);
            enc.set_buffer(2, Some(&gamma_buf), 0);
            enc.set_buffer(3, Some(&inp_buf), 0); // src1_1 unused for F=2
            enc.set_buffer(4, Some(&flat_buf), 0);
            enc.set_threadgroup_memory_length(0, 32 * 4);
            enc.dispatch_thread_groups(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rms_nth, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2: matvec_f32 — fn_w × flat_buf → pre_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe);
            set_scalar_bytes(enc, 0, &mv_args);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&flat_buf), 0);
            enc.set_buffer(3, Some(&pre_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "output_hc_head_batched_impl");

        Ok(unsafe { read_buffer::<f32>(&pre_buf, n_hc) })
    }

    pub(crate) fn output_hc_head_batched(
        &self,
        inp_hc: &[f32],
        fn_w: &[f32],
        n_hc: usize,
        d_embd: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.output_hc_head_batched_impl(inp_hc, fn_w, n_hc, d_embd, eps_rms)
            .expect("ds4_metal::output_hc_head_batched encoding failed")
    }

    /// M4 #330o Phase C.3d.2: packs the per-layer Q/K/V split chain
    ///   q_heads_raw = matvec_f32(w_q_b,  qr_normed_q,  q_dim)
    ///   q_heads     = head_rms_norm(q_heads_raw, n_head, head_dim, eps)
    ///   kv_raw_row  = matvec_f32(w_kv,   normed_kv,    kv_row)
    /// into ONE `MTLCommandBuffer` with two readbacks (`q_heads`,
    /// `kv_raw_row`). Passes 1 and 3 are independent (different inputs,
    /// weights, outputs); the Metal command buffer schedules them on
    /// the GPU. Pass 2 depends on pass 1's output. Bit-identical to
    /// the sequential trait path under default-OFF
    /// `DS4_HEAD_RMS_F64_FIDELITY=0` (the f32 head_rms_norm path —
    /// callers under fid=1 must NOT use this primitive).
    pub(crate) fn qkv_b_head_rms_batched_impl(
        &self,
        qr_normed_q: &[f32],
        w_q_b: &[f32],
        q_dim: usize,
        n_head: usize,
        head_dim: usize,
        eps_rms: f32,
        normed_kv: &[f32],
        w_kv: &[f32],
        kv_row: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(
            q_dim == n_head * head_dim,
            "qkv_b_head_rms_batched: q_dim ({}) != n_head*head_dim ({}*{}={})",
            q_dim,
            n_head,
            head_dim,
            n_head * head_dim,
        );
        let d_qb = qr_normed_q.len();
        anyhow::ensure!(
            w_q_b.len() == q_dim * d_qb,
            "qkv_b_head_rms_batched: w_q_b.len ({}) != q_dim*d_qb ({}*{})",
            w_q_b.len(),
            q_dim,
            d_qb,
        );
        let d_kv = normed_kv.len();
        anyhow::ensure!(
            w_kv.len() == kv_row * d_kv,
            "qkv_b_head_rms_batched: w_kv.len ({}) != kv_row*d_kv ({}*{})",
            w_kv.len(),
            kv_row,
            d_kv,
        );
        anyhow::ensure!(
            d_qb % 4 == 0,
            "qkv_b_head_rms_batched: d_qb ({}) must be divisible by 4 (float4)",
            d_qb,
        );
        anyhow::ensure!(
            d_kv % 4 == 0,
            "qkv_b_head_rms_batched: d_kv ({}) must be divisible by 4 (float4)",
            d_kv,
        );
        anyhow::ensure!(
            q_dim % 2 == 0,
            "qkv_b_head_rms_batched: q_dim ({}) must be divisible by NR0=2",
            q_dim,
        );
        anyhow::ensure!(
            kv_row % 2 == 0,
            "qkv_b_head_rms_batched: kv_row ({}) must be divisible by NR0=2",
            kv_row,
        );
        anyhow::ensure!(head_dim >= 1, "qkv_b_head_rms_batched: head_dim must be >= 1");

        // ---- Pipelines ----
        // Matvec for Q-side (specialized on d_qb).
        let nsg_qb: i16 = (((d_qb as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_qb: i16 = if d_qb % 256 == 0 {
            16
        } else if d_qb % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_qb = Vec::with_capacity(4);
        key_qb.extend_from_slice(&nsg_qb.to_le_bytes());
        key_qb.extend_from_slice(&nxpsg_qb.to_le_bytes());
        let mv_pipe_qb = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_qb, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg_qb as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg_qb as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        // Matvec for KV-side (specialized on d_kv).
        let nsg_kv: i16 = (((d_kv as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_kv: i16 = if d_kv % 256 == 0 {
            16
        } else if d_kv % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_kv = Vec::with_capacity(4);
        key_kv.extend_from_slice(&nsg_kv.to_le_bytes());
        key_kv.extend_from_slice(&nxpsg_kv.to_le_bytes());
        let mv_pipe_kv = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_kv, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg_kv as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg_kv as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        // Head-RMS pipeline (Phase A.1).
        let head_rms_pipe = self
            .pipelines
            .get("ds4_head_rms_norm_f32")
            .ok_or_else(|| anyhow::anyhow!("ds4_head_rms_norm_f32 pipeline not loaded"))?;

        // ---- Buffers ----
        let w_qb_buf = self.cached_weight_buffer(w_q_b);
        let qr_in_buf = new_input_buffer(&self.device, qr_normed_q);
        let q_raw_buf = new_output_buffer::<f32>(&self.device, q_dim);
        let q_out_buf = new_output_buffer::<f32>(&self.device, q_dim);

        let w_kv_buf = self.cached_weight_buffer(w_kv);
        let kv_in_buf = new_input_buffer(&self.device, normed_kv);
        let kv_out_buf = new_output_buffer::<f32>(&self.device, kv_row);

        // ---- Matvec args ----
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

        let mv_args_qb = MulMvArgs {
            ne00: d_qb as i32,
            ne01: q_dim as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_qb * 4) as u64,
            nb02: (d_qb * q_dim * 4) as u64,
            nb03: (d_qb * q_dim * 4) as u64,
            ne10: d_qb as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_qb * 4) as u64,
            nb12: (d_qb * 4) as u64,
            nb13: (d_qb * 4) as u64,
            ne0: q_dim as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_args_kv = MulMvArgs {
            ne00: d_kv as i32,
            ne01: kv_row as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_kv * 4) as u64,
            nb02: (d_kv * kv_row * 4) as u64,
            nb03: (d_kv * kv_row * 4) as u64,
            ne10: d_kv as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_kv * 4) as u64,
            nb12: (d_kv * 4) as u64,
            nb13: (d_kv * 4) as u64,
            ne0: kv_row as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_shmem_bytes: u64 = 32 * 2 * 4;
        let mv_n_row_tg_qb = ((q_dim as u64) + 1) / 2;
        let mv_n_row_tg_kv = ((kv_row as u64) + 1) / 2;

        // ---- head_rms_norm threadgroup sizing (matches head_rms_norm_impl) ----
        let mut tcount: u64 = 32;
        while tcount < head_dim as u64 && tcount < 1024 {
            tcount *= 2;
        }
        if tcount > head_dim as u64 {
            tcount = head_dim as u64;
        }
        let tcount = tcount.max(1);
        let hd_u32 = head_dim as u32;

        // ---- Single command buffer, three passes ----
        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: matvec_f32 w_q_b × qr_normed_q → q_raw_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe_qb);
            set_scalar_bytes(enc, 0, &mv_args_qb);
            enc.set_buffer(1, Some(&w_qb_buf), 0);
            enc.set_buffer(2, Some(&qr_in_buf), 0);
            enc.set_buffer(3, Some(&q_raw_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg_qb, 1, 1),
                MTLSize::new(32, nsg_qb as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2: head_rms_norm q_raw_buf → q_out_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(head_rms_pipe);
            enc.set_buffer(0, Some(&q_raw_buf), 0);
            enc.set_buffer(1, Some(&q_out_buf), 0);
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &hd_u32 as *const _ as *const _,
            );
            enc.set_bytes(
                3,
                std::mem::size_of::<f32>() as u64,
                &eps_rms as *const _ as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(n_head as u64, 1, 1),
                MTLSize::new(tcount, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 3: matvec_f32 w_kv × normed_kv → kv_out_buf (independent of 1/2)
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe_kv);
            set_scalar_bytes(enc, 0, &mv_args_kv);
            enc.set_buffer(1, Some(&w_kv_buf), 0);
            enc.set_buffer(2, Some(&kv_in_buf), 0);
            enc.set_buffer(3, Some(&kv_out_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg_kv, 1, 1),
                MTLSize::new(32, nsg_kv as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        let _ds4_qkv_trace = std::env::var("DS4_QKV_TRACE").is_ok();
        let t_commit = std::time::Instant::now();
        cmd_buf.commit();
        let commit_us = t_commit.elapsed().as_micros();
        let t_wait = std::time::Instant::now();
        cmd_buf.wait_until_completed();
        let wait_us = t_wait.elapsed().as_micros();

        let q_heads = unsafe { read_buffer::<f32>(&q_out_buf, q_dim) };
        let kv_raw_row = unsafe { read_buffer::<f32>(&kv_out_buf, kv_row) };

        if _ds4_qkv_trace {
            eprintln!(
                "DS4_QKV_TRACE,d_qb={},q_dim={},d_kv={},kv_row={},commit={},wait={}",
                d_qb, q_dim, d_kv, kv_row, commit_us, wait_us
            );
        }

        Ok((q_heads, kv_raw_row))
    }

    pub(crate) fn qkv_b_head_rms_batched(
        &self,
        qr_normed_q: &[f32],
        w_q_b: &[f32],
        q_dim: usize,
        n_head: usize,
        head_dim: usize,
        eps_rms: f32,
        normed_kv: &[f32],
        w_kv: &[f32],
        kv_row: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.qkv_b_head_rms_batched_impl(
            qr_normed_q, w_q_b, q_dim, n_head, head_dim, eps_rms,
            normed_kv, w_kv, kv_row,
        )
        .expect("ds4_metal::qkv_b_head_rms_batched encoding failed")
    }

    /// Phase C.3e (M4 #330o). Pack the per-layer MoE router head
    /// `matvec_f32(w_router, h_norm, n_experts)` + `softplus_sqrt` into
    /// ONE `MTLCommandBuffer` with a single readback. Saves 1 cwr/layer =
    /// 43 cwr/token on DS4 V4 Flash. Bit-identical to the sequential
    /// trait path (uses `ds4_kernel_mul_mv_f32_f32_4` for matvec and
    /// `ds4_kernel_dsv4_softplus_sqrt_f32_4` for the activation — same
    /// kernels the trait `matvec_f32` and `softplus_sqrt` methods use).
    pub(crate) fn router_logits_batched_impl(
        &self,
        w_router: &[f32],
        h_norm: &[f32],
        n_experts: usize,
    ) -> Result<Vec<f32>> {
        let d_in = h_norm.len();
        anyhow::ensure!(
            w_router.len() == n_experts * d_in,
            "router_logits_batched: w_router.len ({}) != n_experts*d_in ({}*{})",
            w_router.len(),
            n_experts,
            d_in,
        );
        anyhow::ensure!(
            d_in % 4 == 0,
            "router_logits_batched: d_in ({}) must be divisible by 4 (float4)",
            d_in,
        );
        anyhow::ensure!(
            n_experts % 4 == 0,
            "router_logits_batched: n_experts ({}) must be divisible by 4 (softplus_sqrt float4)",
            n_experts,
        );
        anyhow::ensure!(
            n_experts % 2 == 0,
            "router_logits_batched: n_experts ({}) must be divisible by NR0=2",
            n_experts,
        );

        // Matvec pipeline (specialized on d_in).
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
        let mv_pipe = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        // softplus_sqrt pipeline (same kernel as `softplus_sqrt_impl`).
        let sps_pipe = self
            .pipelines
            .get("ds4_kernel_dsv4_softplus_sqrt_f32_4")
            .ok_or_else(|| anyhow::anyhow!("softplus_sqrt pipeline not loaded"))?
            .clone();

        // ---- Buffers ----
        let w_buf = self.cached_weight_buffer(w_router);
        let h_buf = new_input_buffer(&self.device, h_norm);
        let logits_buf = new_output_buffer::<f32>(&self.device, n_experts);
        let probs_buf = new_output_buffer::<f32>(&self.device, n_experts);

        // ---- Matvec args ----
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
        let mv_args = MulMvArgs {
            ne00: d_in as i32,
            ne01: n_experts as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_in * 4) as u64,
            nb02: (d_in * n_experts * 4) as u64,
            nb03: (d_in * n_experts * 4) as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            nb13: (d_in * 4) as u64,
            ne0: n_experts as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let mv_shmem_bytes: u64 = 32 * 2 * 4;
        let mv_n_row_tg = ((n_experts as u64) + 1) / 2;

        // softplus_sqrt args (matches softplus_sqrt_impl).
        let ne0_4: u32 = (n_experts / 4) as u32;
        let nb: u32 = (n_experts * 4) as u32;

        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: matvec_f32 w_router × h_norm → logits_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&mv_pipe);
            set_scalar_bytes(enc, 0, &mv_args);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&h_buf), 0);
            enc.set_buffer(3, Some(&logits_buf), 0);
            enc.set_threadgroup_memory_length(0, mv_shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(mv_n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2: softplus_sqrt logits_buf → probs_buf
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&sps_pipe);
            enc.set_buffer(0, Some(&logits_buf), 0);
            enc.set_buffer(1, Some(&probs_buf), 0);
            set_scalar_bytes(enc, 2, &ne0_4);
            set_scalar_bytes(enc, 3, &nb);
            set_scalar_bytes(enc, 4, &nb);
            enc.dispatch_threads(
                MTLSize::new(ne0_4 as u64, 1, 1),
                MTLSize::new(ne0_4 as u64, 1, 1),
            );
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "router_logits_batched_impl");

        Ok(unsafe { read_buffer::<f32>(&probs_buf, n_experts) })
    }

    pub(crate) fn router_logits_batched(
        &self,
        w_router: &[f32],
        h_norm: &[f32],
        n_experts: usize,
    ) -> Vec<f32> {
        self.router_logits_batched_impl(w_router, h_norm, n_experts)
            .expect("ds4_metal::router_logits_batched encoding failed")
    }

    #[allow(dead_code)]
    pub(crate) fn matvec_q8_0_bytes_impl(
        &self,
        w_q8_0: &[u8],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let d_in = x.len();
        anyhow::ensure!(
            d_in % 32 == 0,
            "matvec_q8_0: d_in ({}) must be divisible by QK8_0=32",
            d_in
        );
        let row_stride = (d_in / 32) * 34;
        anyhow::ensure!(
            w_q8_0.len() == d_out * row_stride,
            "matvec_q8_0: w bytes ({}) != d_out * row_stride ({} * {})",
            w_q8_0.len(),
            d_out,
            row_stride
        );
        anyhow::ensure!(
            d_out % 2 == 0,
            "matvec_q8_0: d_out ({}) must be divisible by NR0=2",
            d_out
        );

        let nsg: i16 = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline = self.specialized_pipeline("ds4_kernel_mul_mv_q8_0_f32", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(
                &nxpsg as *const _ as *const _,
                MTLDataType::Short,
                601,
            );
        })?;

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

        let w_buf = new_input_buffer(&self.device, w_q8_0);
        let x_buf = new_input_buffer(&self.device, x);
        let dst_buf = new_output_buffer::<f32>(&self.device, d_out);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((d_out as u64) + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "matvec_q8_0_bytes_impl");

        Ok(unsafe { read_buffer::<f32>(&dst_buf, d_out) })
    }

    /// Dense Q4_0 matvec (`out[d_out] = dequant_q4_0(w) · x[d_in]`) via
    /// `ds4_kernel_mul_mv_q4_0_f32`. Twin of `matvec_q8_0_bytes_impl` for
    /// `block_q4_0` (18 B/32w). Used by the q4_0 parity test; the lm-head tail
    /// shares the same kernel through `tail_lm_head_argmax_quant`.
    #[allow(dead_code)]
    pub(crate) fn matvec_q4_0_bytes_impl(
        &self,
        w_q4_0: &[u8],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let d_in = x.len();
        anyhow::ensure!(d_in % 32 == 0, "matvec_q4_0: d_in ({}) % 32", d_in);
        anyhow::ensure!(d_out % 2 == 0, "matvec_q4_0: d_out ({}) % NR0=2", d_out);
        let row_stride = (d_in / 32) * 18;
        anyhow::ensure!(
            w_q4_0.len() == d_out * row_stride,
            "matvec_q4_0: w bytes ({}) != d_out*row_stride ({}*{})",
            w_q4_0.len(), d_out, row_stride
        );

        let nsg: i16 = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline = self.specialized_pipeline("ds4_kernel_mul_mv_q4_0_f32", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

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
            nb00: 2, nb01: row_stride as u64, nb02: (row_stride * d_out) as u64, nb03: (row_stride * d_out) as u64,
            ne10: d_in as i32, ne11: 1, ne12: 1, _pad1: 0,
            nb10: 4, nb11: (d_in * 4) as u64, nb12: (d_in * 4) as u64, nb13: (d_in * 4) as u64,
            ne0: d_out as i32, ne1: 1, nr0: 2, r2: 1, r3: 1,
        };

        let w_buf = new_input_buffer(&self.device, w_q4_0);
        let x_buf = new_input_buffer(&self.device, x);
        let dst_buf = new_output_buffer::<f32>(&self.device, d_out);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((d_out as u64) + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "matvec_q4_0_bytes_impl");

        Ok(unsafe { read_buffer::<f32>(&dst_buf, d_out) })
    }

    #[allow(dead_code)]
    pub(crate) fn q8_hc_expand4_bytes_impl(
        &self,
        w_q8_0: &[u8],
        input: &[f32],
        residual_dh: &[f32],
        post: &[f32],
        comb: &[f32],
        d_out: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d_in = input.len();
        anyhow::ensure!(
            d_in % 32 == 0,
            "q8_hc_expand4: d_in ({}) must be divisible by QK8_0=32",
            d_in
        );
        anyhow::ensure!(
            residual_dh.len() == d_out * 4,
            "q8_hc_expand4: residual_dh.len ({}) != d_out * 4 ({} * 4)",
            residual_dh.len(),
            d_out
        );
        anyhow::ensure!(
            post.len() == 4,
            "q8_hc_expand4: post.len ({}) != 4",
            post.len()
        );
        anyhow::ensure!(
            comb.len() == 16,
            "q8_hc_expand4: comb.len ({}) != 16",
            comb.len()
        );
        let row_stride = (d_in / 32) * 34;
        anyhow::ensure!(
            w_q8_0.len() == d_out * row_stride,
            "q8_hc_expand4: w bytes ({}) != d_out * row_stride ({} * {})",
            w_q8_0.len(),
            d_out,
            row_stride
        );

        let nsg: i16 = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline = self.specialized_pipeline("ds4_dsv4_q8_hc_expand4_q8_0", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(
                &nxpsg as *const _ as *const _,
                MTLDataType::Short,
                601,
            );
        })?;

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
            _pad0: i32,
        }

        let mv = MulMvArgs {
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
        let hc = HcExpandArgs {
            n_embd: d_out as i64,
            n_hc: 4,
            n_tokens: 1,
            nb_block0: 4,
            nb_block1: (d_out * 4) as u64,
            nb_add0: 0,
            nb_add1: 0,
            nb_res0: 16,
            nb_res1: 4,
            nb_res2: (d_out * 4 * 4) as u64,
            nb_post0: 4,
            nb_post1: 0,
            nb_comb0: 16,
            nb_comb1: 4,
            nb_comb2: 0,
            nb0: 16,
            nb1: 4,
            nb2: (d_out * 4 * 4) as u64,
            has_add: 0,
            _pad0: 0,
        };

        let w_buf = new_input_buffer(&self.device, w_q8_0);
        let input_buf = new_input_buffer(&self.device, input);
        let block_out_buf = new_output_buffer::<f32>(&self.device, d_out);
        let residual_buf = new_input_buffer(&self.device, residual_dh);
        let post_buf = new_input_buffer(&self.device, post);
        let comb_buf = new_input_buffer(&self.device, comb);
        let dst_buf = new_output_buffer::<f32>(&self.device, d_out * 4);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &mv);
        set_scalar_bytes(enc, 1, &hc);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&input_buf), 0);
        enc.set_buffer(4, Some(&block_out_buf), 0);
        enc.set_buffer(5, Some(&residual_buf), 0);
        enc.set_buffer(6, Some(&post_buf), 0);
        enc.set_buffer(7, Some(&comb_buf), 0);
        enc.set_buffer(8, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((d_out as u64) + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "q8_hc_expand4_bytes_impl");

        Ok((
            unsafe { read_buffer::<f32>(&block_out_buf, d_out) },
            unsafe { read_buffer::<f32>(&dst_buf, d_out * 4) },
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn attn_out_low_q8_0_bytes_impl(
        &self,
        w_q8_0: &[u8],
        heads: &[f32],
        n_groups: usize,
        n_lora_o: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(n_groups > 0, "attn_out_low_q8_0: n_groups must be > 0");
        anyhow::ensure!(
            heads.len() % n_groups == 0,
            "attn_out_low_q8_0: heads.len ({}) must be divisible by n_groups ({})",
            heads.len(),
            n_groups
        );
        anyhow::ensure!(n_lora_o > 0, "attn_out_low_q8_0: n_lora_o must be > 0");
        let group_dim = heads.len() / n_groups;
        anyhow::ensure!(
            group_dim % 32 == 0,
            "attn_out_low_q8_0: group_dim ({}) must be divisible by QK8_0=32",
            group_dim
        );
        let row_stride = (group_dim / 32) * 34;
        let out_low_dim = n_groups * n_lora_o;
        anyhow::ensure!(
            w_q8_0.len() == out_low_dim * row_stride,
            "attn_out_low_q8_0: w bytes ({}) != out_low_dim * row_stride ({} * {})",
            w_q8_0.len(),
            out_low_dim,
            row_stride
        );

        let nsg: i16 = ((group_dim as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline =
            self.specialized_pipeline("ds4_dsv4_attn_out_low_q8_0_f32", &key, |fcv| {
                use metal::MTLDataType;
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

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MulMvIdArgs {
            nei0: i32,
            nei1: i32,
            nbi1: u64,
            ne00: i32,
            ne01: i32,
            ne02: i32,
            _pad0: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            ne0: i32,
            ne1: i32,
            nb1: u64,
            nr0: i32,
            _pad1: i32,
        }

        let args = MulMvIdArgs {
            nei0: n_groups as i32,
            nei1: 1,
            nbi1: 0,
            ne00: group_dim as i32,
            ne01: n_lora_o as i32,
            ne02: n_groups as i32,
            _pad0: 0,
            nb00: 2,
            nb01: row_stride as u64,
            nb02: (row_stride * n_lora_o) as u64,
            ne10: group_dim as i32,
            ne11: n_groups as i32,
            ne12: 1,
            ne13: 1,
            nb10: 4,
            nb11: (group_dim * 4) as u64,
            nb12: (heads.len() * 4) as u64,
            ne0: n_lora_o as i32,
            ne1: out_low_dim as i32,
            nb1: (n_lora_o * 4) as u64,
            nr0: 2,
            _pad1: 0,
        };

        let w_buf = new_input_buffer(&self.device, w_q8_0);
        let heads_buf = new_input_buffer(&self.device, heads);
        let dst_buf = new_output_buffer::<f32>(&self.device, out_low_dim);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&heads_buf), 0);
        enc.set_buffer(3, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((n_lora_o as u64) + 1) / 2, 1, n_groups as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "attn_out_low_q8_0_bytes_impl");

        Ok(unsafe { read_buffer::<f32>(&dst_buf, out_low_dim) })
    }

    #[allow(dead_code)]
    pub(crate) fn shared_down_hc_expand4_q8_0_bytes_impl(
        &self,
        w_q8_0: &[u8],
        shared_mid: &[f32],
        routed_out: &[f32],
        residual_dh: &[f32],
        post: &[f32],
        comb: &[f32],
        d_out: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d_in = shared_mid.len();
        anyhow::ensure!(
            d_in % 32 == 0,
            "shared_down_hc_expand4_q8_0: d_in ({}) must be divisible by QK8_0=32",
            d_in
        );
        anyhow::ensure!(
            routed_out.len() == d_out,
            "shared_down_hc_expand4_q8_0: routed_out.len ({}) != d_out ({})",
            routed_out.len(),
            d_out
        );
        anyhow::ensure!(
            residual_dh.len() == d_out * 4,
            "shared_down_hc_expand4_q8_0: residual_dh.len ({}) != d_out * 4 ({} * 4)",
            residual_dh.len(),
            d_out
        );
        anyhow::ensure!(
            post.len() == 4,
            "shared_down_hc_expand4_q8_0: post.len ({}) != 4",
            post.len()
        );
        anyhow::ensure!(
            comb.len() == 16,
            "shared_down_hc_expand4_q8_0: comb.len ({}) != 16",
            comb.len()
        );
        let row_stride = (d_in / 32) * 34;
        anyhow::ensure!(
            w_q8_0.len() == d_out * row_stride,
            "shared_down_hc_expand4_q8_0: w bytes ({}) != d_out * row_stride ({} * {})",
            w_q8_0.len(),
            d_out,
            row_stride
        );

        let nsg: i16 = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline =
            self.specialized_pipeline("ds4_dsv4_shared_down_hc_expand4_q8_0", &key, |fcv| {
                use metal::MTLDataType;
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
            _pad0: i32,
        }

        let mv = MulMvArgs {
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
        let hc = HcExpandArgs {
            n_embd: d_out as i64,
            n_hc: 4,
            n_tokens: 1,
            nb_block0: 4,
            nb_block1: (d_out * 4) as u64,
            nb_add0: 0,
            nb_add1: 0,
            nb_res0: 16,
            nb_res1: 4,
            nb_res2: (d_out * 4 * 4) as u64,
            nb_post0: 4,
            nb_post1: 0,
            nb_comb0: 16,
            nb_comb1: 4,
            nb_comb2: 0,
            nb0: 16,
            nb1: 4,
            nb2: (d_out * 4 * 4) as u64,
            has_add: 1,
            _pad0: 0,
        };

        let w_buf = new_input_buffer(&self.device, w_q8_0);
        let shared_mid_buf = new_input_buffer(&self.device, shared_mid);
        let shared_out_buf = new_output_buffer::<f32>(&self.device, d_out);
        let routed_buf = new_input_buffer(&self.device, routed_out);
        let residual_buf = new_input_buffer(&self.device, residual_dh);
        let post_buf = new_input_buffer(&self.device, post);
        let comb_buf = new_input_buffer(&self.device, comb);
        let dst_buf = new_output_buffer::<f32>(&self.device, d_out * 4);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &mv);
        set_scalar_bytes(enc, 1, &hc);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&shared_mid_buf), 0);
        enc.set_buffer(4, Some(&shared_out_buf), 0);
        enc.set_buffer(5, Some(&routed_buf), 0);
        enc.set_buffer(6, Some(&residual_buf), 0);
        enc.set_buffer(7, Some(&post_buf), 0);
        enc.set_buffer(8, Some(&comb_buf), 0);
        enc.set_buffer(9, Some(&dst_buf), 0);
        enc.set_threadgroup_memory_length(0, 32 * 2 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((d_out as u64) + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "shared_down_hc_expand4_q8_0_bytes_impl");

        Ok((
            unsafe { read_buffer::<f32>(&shared_out_buf, d_out) },
            unsafe { read_buffer::<f32>(&dst_buf, d_out * 4) },
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn shared_gate_up_swiglu_q8_0_bytes_impl(
        &self,
        gate_q8_0: &[u8],
        up_q8_0: &[u8],
        x: &[f32],
        d_out: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let d_in = x.len();
        anyhow::ensure!(
            d_in % 32 == 0,
            "shared_gate_up_swiglu_q8_0: d_in ({}) must be divisible by QK8_0=32",
            d_in
        );
        let row_stride = (d_in / 32) * 34;
        anyhow::ensure!(
            gate_q8_0.len() == d_out * row_stride,
            "shared_gate_up_swiglu_q8_0: gate bytes ({}) != d_out * row_stride ({} * {})",
            gate_q8_0.len(),
            d_out,
            row_stride
        );
        anyhow::ensure!(
            up_q8_0.len() == d_out * row_stride,
            "shared_gate_up_swiglu_q8_0: up bytes ({}) != d_out * row_stride ({} * {})",
            up_q8_0.len(),
            d_out,
            row_stride
        );

        let nsg: i16 = ((d_in as u64 + 127) / 128).clamp(1, 8) as i16;
        let nxpsg: i16 = 4;
        let mut key = Vec::with_capacity(4);
        key.extend_from_slice(&nsg.to_le_bytes());
        key.extend_from_slice(&nxpsg.to_le_bytes());
        let pipeline =
            self.specialized_pipeline("ds4_dsv4_shared_gate_up_swiglu_q8_0", &key, |fcv| {
                use metal::MTLDataType;
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

        let gate_buf = new_input_buffer(&self.device, gate_q8_0);
        let up_buf = new_input_buffer(&self.device, up_q8_0);
        let x_buf = new_input_buffer(&self.device, x);
        let gate_out = new_output_buffer::<f32>(&self.device, d_out);
        let up_out = new_output_buffer::<f32>(&self.device, d_out);
        let mid_out = new_output_buffer::<f32>(&self.device, d_out);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&pipeline);
        set_scalar_bytes(enc, 0, &args);
        enc.set_buffer(1, Some(&gate_buf), 0);
        enc.set_buffer(2, Some(&up_buf), 0);
        enc.set_buffer(3, Some(&x_buf), 0);
        enc.set_buffer(4, Some(&gate_out), 0);
        enc.set_buffer(5, Some(&up_out), 0);
        enc.set_buffer(6, Some(&mid_out), 0);
        enc.set_threadgroup_memory_length(0, 32 * 4 * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(((d_out as u64) + 1) / 2, 1, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "shared_gate_up_swiglu_q8_0_bytes_impl");

        Ok((
            unsafe { read_buffer::<f32>(&gate_out, d_out) },
            unsafe { read_buffer::<f32>(&up_out, d_out) },
            unsafe { read_buffer::<f32>(&mid_out, d_out) },
        ))
    }

    pub(crate) fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.softplus_sqrt_impl(logits)
            .expect("ds4_metal::softplus_sqrt encoding failed")
    }

    pub(crate) fn router_finalize(
        &self,
        probs: &[f32],
        bias: &[f32],
        k: usize,
    ) -> (Vec<usize>, Vec<f32>) {
        self.router_finalize_impl(probs, bias, k)
            .expect("ds4_metal::router_finalize encoding failed")
    }

    /// Trait-facing `moe_routed_step` shim. Routes through the preloaded
    /// `QuantizedExpertWeights` table for `layer_idx`; the `experts_w_*`
    /// `&[f32]` slices from the trait are intentionally ignored — they
    /// are the CPU reference's dequantized view, while the Metal kernel
    /// consumes the quantized bytes directly from `metal::Buffer`s on
    /// `MetalState`. Falls back to `bail!` (via `.expect`) if the layer's
    /// table was never loaded with `load_expert_weights`.
    /// Phase C-B Slice 5-redo: fused `moe + shared_chain` entry point.
    /// Mirrors `moe_routed_step`'s production wiring (looks up the
    /// per-layer QuantizedExpertWeights for the moe side) but bundles
    /// the shared_chain call into the same cb so we pay only one
    /// `wait_until_completed`. Falls back to two-step path when MoE
    /// CPU-via-dequant is forced via env var.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_and_shared_chain_batched_inherent(
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
    ) -> (Vec<f32>, Vec<f32>) {
        // Fall back to the two-step path under the same env-var gates
        // moe_routed_step honours, so debug/correctness modes keep
        // working unchanged.
        let force_cpu_moe = std::env::var("DS4_MOE_CPU_VIA_DEQUANT").is_ok()
            || std::env::var("DS4_ZERO_ROUTED_MOE").is_ok();
        if force_cpu_moe {
            let moe_out = self.moe_routed_step(
                layer_idx, moe_x, moe_selected, moe_weights,
                &[], &[], &[], d_ffn,
            );
            let sh_out = self.shared_chain_batched(
                sh_ffn_norm, sh_w_gate, sh_w_up, sh_w_down, shared_dim, want_q80,
            );
            return (moe_out, sh_out);
        }
        let qew = self.expert_weights.get(layer_idx as usize).unwrap_or_else(|| {
            panic!(
                "ds4_metal::moe_and_shared_chain_batched: no QuantizedExpertWeights for layer {} (loaded={})",
                layer_idx,
                self.expert_weights.len(),
            )
        });
        let selected_i32: Vec<i32> = moe_selected.iter().map(|&i| i as i32).collect();
        let d_ffn_eff = if d_ffn == 0 { qew.d_ffn as usize } else { d_ffn };
        // SSD-streaming: bind the expert cache slots instead of the full
        // (mmap-only, GPU-unsafe) stacked tensors.
        let stream_bind = self.streaming_expert_bind(layer_idx, &selected_i32);
        let (selected_i32, gate_buf, up_buf, down_buf) = match &stream_bind {
            Some((slots, g, u, d)) => (slots.clone(), g, u, d),
            None => (
                selected_i32,
                &qew.gate.metal_buf,
                &qew.up.metal_buf,
                &qew.down.metal_buf,
            ),
        };
        self.moe_shared_chain_batched_impl(
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
            sh_ffn_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            want_q80,
        )
        .expect("ds4_metal::moe_and_shared_chain_batched encoding failed")
    }

    pub(crate) fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        _experts_w_gate: &[f32],
        _experts_w_up: &[f32],
        _experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        if std::env::var("DS4_ZERO_ROUTED_MOE").is_ok() {
            return vec![0.0_f32; x.len()];
        }
        let qew = self
            .expert_weights
            .get(layer_idx as usize)
            .unwrap_or_else(|| {
                panic!(
                    "ds4_metal::moe_routed_step: no QuantizedExpertWeights for layer {} (loaded={})",
                    layer_idx,
                    self.expert_weights.len(),
                )
            });
        if std::env::var("DS4_MOE_CPU_VIA_DEQUANT").is_ok() {
            // Caller's d_ffn is 0 for production (from_views skips it since
            // Metal reads d_ffn from QEW); use qew.d_ffn directly for the
            // CPU oracle's shape check.
            let d_ffn_eff = if d_ffn == 0 {
                qew.d_ffn as usize
            } else {
                d_ffn
            };
            return crate::quantized_experts::moe_routed_step_cpu_via_dequant(
                qew, x, selected, weights, d_ffn_eff,
            )
            .unwrap_or_else(|e| {
                panic!("ds4_metal::moe_routed_step layer {layer_idx} cpu-via-dequant fallback: {e}")
            });
        }
        // `selected` from the trait is `&[usize]`, but the kernel reads
        // i32 expert ids — convert once here.
        let selected_i32: Vec<i32> = selected.iter().map(|&i| i as i32).collect();
        let d_ffn_eff = if d_ffn == 0 {
            qew.d_ffn as usize
        } else {
            d_ffn
        };
        self.moe_routed_step_impl(
            x,
            &selected_i32,
            weights,
            &qew.gate.metal_buf,
            &qew.up.metal_buf,
            &qew.down.metal_buf,
            qew.gate.ttype,
            qew.up.ttype,
            qew.down.ttype,
            d_ffn_eff,
        )
        .expect("ds4_metal::moe_routed_step encoding failed")
    }
}

// Compile-time use of metal types so unused-import warnings don't fire
// before the per-method encoding stubs are filled in.
#[allow(dead_code)]
fn _touch_metal_types(state: &MetalState) {
    let _: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
    let _: MTLSize = MTLSize::new(1, 1, 1);
    let _ = &state.command_queue;
    let _ = &state.device;
}

// ───────────────────────────────────────────────────────────────────────
// AttentionDispatcher (M4 Option 3, task #214) encoding stubs.
//
// Each helper follows the same shape as the KernelDispatcher helpers
// above: look up the registered pipeline by `metal_fn`, allocate input
// buffers from f32 slices, encode + dispatch, read back. Heavyweight
// composers (flash_attn_decode, hc_collapse_norm, attn_output_proj,
// shared_expert, shared_down_hc_expand_add) wire the pipeline lookups
// but bail at dispatch time until the macOS validation pass lands;
// simple per-row kernels (qkv_rms_norm_rows, rope_tail, kv_fp8_store)
// encode their kernel directly.
// ───────────────────────────────────────────────────────────────────────

use ds4_engine::attn_dispatch::{AttnHeadsOut, KvCacheView, LayerParams};

impl MetalState {
    /// `dsv4_qkv_rms_norm_f32_4` — joint Q + KV RMSNorm.
    ///
    /// Buffers (registry: ConstIn ConstIn Writable ConstIn ConstIn Writable):
    ///   p0=qr (f32), p1=gamma_q (f32), p2=qr_out (f32),
    ///   p3=kv_raw (f32, n_lora_kv), p4=gamma_kv (f32), p5=kv_out (f32).
    /// Uniforms (Scalars 7): n_q4 (u32), n_kv4 (u32), n_lora_q (u32),
    ///   n_lora_kv (u32), n_rot (u32), eps_q (f32), eps_kv (f32).
    /// Dispatch: 1 threadgroup, max(n_q4, n_kv4) threads.
    #[allow(dead_code)]
    pub(crate) fn qkv_rms_norm_rows_impl(
        &self,
        params: &LayerParams,
        qr: &[f32],
        kv_raw: &[f32],
        gamma_q: &[f32],
        gamma_kv: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n_lora_q = params.n_lora_q as usize;
        let n_lora_kv = params.n_lora_kv as usize;
        let n_rot = params.n_rot as usize;
        anyhow::ensure!(qr.len() == n_lora_q, "qkv_rms_norm: qr.len mismatch");
        anyhow::ensure!(kv_raw.len() == n_lora_kv, "qkv_rms_norm: kv.len mismatch");
        anyhow::ensure!(
            gamma_q.len() == n_lora_q,
            "qkv_rms_norm: gamma_q.len mismatch"
        );
        anyhow::ensure!(
            gamma_kv.len() == n_lora_kv,
            "qkv_rms_norm: gamma_kv.len mismatch"
        );
        anyhow::ensure!(
            n_lora_q % 4 == 0 && n_lora_kv % 4 == 0,
            "qkv_rms_norm: dims must be %4 (float4 path)"
        );
        let pipeline = self
            .pipelines
            .get("ds4_dsv4_qkv_rms_norm_f32_4")
            .ok_or_else(|| anyhow::anyhow!("dsv4_qkv_rms_norm pipeline not loaded"))?;

        let qr_in = new_input_buffer(&self.device, qr);
        let gq_in = new_input_buffer(&self.device, gamma_q);
        let qr_out = new_output_buffer::<f32>(&self.device, n_lora_q);
        let kv_in = new_input_buffer(&self.device, kv_raw);
        let gkv_in = new_input_buffer(&self.device, gamma_kv);
        let kv_out = new_output_buffer::<f32>(&self.device, n_lora_kv);

        let n_q4: u32 = (n_lora_q / 4) as u32;
        let n_kv4: u32 = (n_lora_kv / 4) as u32;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&qr_in), 0);
        enc.set_buffer(1, Some(&gq_in), 0);
        enc.set_buffer(2, Some(&qr_out), 0);
        enc.set_buffer(3, Some(&kv_in), 0);
        enc.set_buffer(4, Some(&gkv_in), 0);
        enc.set_buffer(5, Some(&kv_out), 0);
        set_scalar_bytes(enc, 6, &n_q4);
        set_scalar_bytes(enc, 7, &n_kv4);
        set_scalar_bytes(enc, 8, &(n_lora_q as u32));
        set_scalar_bytes(enc, 9, &(n_lora_kv as u32));
        set_scalar_bytes(enc, 10, &(n_rot as u32));
        set_scalar_bytes(enc, 11, &params.rms_eps);
        set_scalar_bytes(enc, 12, &params.rms_eps);

        let tg = (n_q4.max(n_kv4) as u64).max(1).min(1024);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "qkv_rms_norm_rows_impl");

        let qr_o = unsafe { read_buffer::<f32>(&qr_out, n_lora_q) };
        let kv_o = unsafe { read_buffer::<f32>(&kv_out, n_lora_kv) };
        Ok((qr_o, kv_o))
    }

    /// `dsv4_rope_tail_f32` (M122) — partial RoPE on the trailing `n_rot`
    /// floats of every head, with YaRN scaling. Mutates `x` in place.
    ///
    /// Registry uniforms = 23 scalars; we pack the antirez `ds4_metal_args_rope`
    /// layout (`ds4.c::ds4_metal_args_rope`). The kernel reads p0=src (f32),
    /// p1=pos (i32 array of length 1 here), p2=freqs (f32, unused for
    /// inline-base mode), p3=dst (f32, in-place when src==dst).
    /// Dispatch: 1 threadgroup per head, n_rot/2 threads.
    #[allow(dead_code)]
    pub(crate) fn rope_tail_impl(
        &self,
        params: &LayerParams,
        x: &mut [f32],
        pos: u32,
        backward: bool,
    ) -> Result<()> {
        let n_rot = params.n_rot as usize;
        anyhow::ensure!(
            n_rot >= 2 && n_rot % 2 == 0,
            "rope_tail: n_rot must be even and ≥2"
        );
        anyhow::ensure!(
            x.len() % n_rot == 0,
            "rope_tail: x.len ({}) must be a multiple of n_rot ({})",
            x.len(),
            n_rot
        );
        let pipeline = self
            .pipelines
            .get("ds4_kernel_dsv4_rope_tail_f32")
            .ok_or_else(|| anyhow::anyhow!("dsv4_rope_tail_f32 pipeline not loaded"))?;

        let src_buf = new_input_buffer(&self.device, x);
        let pos_buf = new_input_buffer(&self.device, &[pos as i32]);
        let freqs_buf = new_input_buffer(&self.device, &[0.0f32]);
        let dst_buf = new_output_buffer::<f32>(&self.device, x.len());

        // 23-scalar uniform layout: see `mlir_to_msl::emit_rope_tail`. Matches
        // antirez `ds4_metal_args_rope` (head_dim, n_rot, n_nope, freq_base,
        // freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
        // orig_ctx, pos, head_count, byte strides...).
        let n_heads = (x.len() / n_rot) as u32;
        let head_dim = n_rot as u32; // we pass only the rope-tail slice
        let n_nope: u32 = 0;
        let stride_bytes = 4u32; // f32

        let scalars: [u32; 23] = [
            head_dim,
            n_rot as u32,
            n_nope,
            params.rope_freq_base.to_bits(),
            params.rope_freq_scale.to_bits(),
            params.rope_ext_factor.to_bits(),
            params.rope_attn_factor.to_bits(),
            (32.0f32).to_bits(), // beta_fast
            (1.0f32).to_bits(),  // beta_slow
            params.rope_orig_ctx,
            pos,
            n_heads,
            stride_bytes,
            (n_rot as u32) * stride_bytes,
            (n_rot as u32) * stride_bytes * n_heads,
            backward as u32,
            stride_bytes,
            (n_rot as u32) * stride_bytes,
            (n_rot as u32) * stride_bytes * n_heads,
            0,
            0,
            0,
            0,
        ];

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&pos_buf), 0);
        enc.set_buffer(2, Some(&freqs_buf), 0);
        enc.set_buffer(3, Some(&dst_buf), 0);
        for (i, s) in scalars.iter().enumerate() {
            set_scalar_bytes(enc, 4 + i as u64, s);
        }
        let half = (n_rot / 2) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(half.max(1).min(1024), 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "rope_tail_impl");

        // Write back into caller-owned slice.
        let out = unsafe { read_buffer::<f32>(&dst_buf, x.len()) };
        x.copy_from_slice(&out);
        Ok(())
    }

    /// Phase E M5.2.2 — kv_fp8_store against the persistent per-layer
    /// buffer (M5.2.1).
    ///
    /// Differs from `kv_fp8_store_impl` in three ways:
    ///   1. The cache buffer comes from `kv_buffer_or_alloc(layer_idx, _)`
    ///      and PERSISTS across calls — no per-call upload, no per-call
    ///      readback.
    ///   2. Caller passes `slot: u32` and `cache_byte_len: usize`
    ///      directly (no `KvCacheView`); cache lifetime is tied to
    ///      `MetalDispatcher`, not a per-call CPU slice.
    ///   3. Returns `metal::Buffer` (cloned handle) so the caller (e.g.
    ///      a fused-cb encoder) can chain a subsequent op that reads
    ///      from the same persistent buffer without round-trip.
    ///
    /// The CPU read-back from the old impl is removed. If a CPU view
    /// of the KV cache is needed (correctness mode, debug probes),
    /// fetch it explicitly via `read_buffer::<f32>(&buf, n_elements)`
    /// after committing. The fused-cb decoder won't need it.
    ///
    /// Uses the SAME `ds4_dsv4_kv_fp8_store` bridge shim as the
    /// existing impl, so semantics are identical at the kernel level
    /// (e4m3 round-trip on NOPE prefix, f16 half-cast on n_rot tail).
    #[allow(dead_code)]
    pub(crate) fn kv_fp8_store_persistent_impl(
        &self,
        layer_idx: u32,
        params: &LayerParams,
        kv_row_f32: &[f32],
        raw_cap: u32,
        slot: u32,
    ) -> Result<metal::Buffer> {
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(
            kv_row_f32.len() == row,
            "kv_fp8_store_persistent: kv_row.len ({}) != n_lora_kv ({})",
            kv_row_f32.len(),
            row
        );
        anyhow::ensure!(raw_cap > 0, "kv_fp8_store_persistent: raw_cap must be > 0");
        anyhow::ensure!(
            slot < raw_cap,
            "kv_fp8_store_persistent: slot {} >= raw_cap {}",
            slot,
            raw_cap
        );

        let pipeline = self
            .pipelines
            .get("ds4_dsv4_kv_fp8_store")
            .ok_or_else(|| anyhow::anyhow!("dsv4_kv_fp8_store pipeline not loaded"))?;

        // Persistent per-layer cache buffer (M5.2.1). Size matches the
        // f32 footprint of the full ring (raw_cap × n_lora_kv floats).
        let cache_byte_len = (raw_cap as usize) * row * std::mem::size_of::<f32>();
        let cache_buf = self.kv_buffer_or_alloc(layer_idx, cache_byte_len);
        // Per-call input row stays a fresh upload — this is small
        // (~2 KB for n_lora_kv=512) and avoids cross-call aliasing.
        let row_buf = new_input_buffer(&self.device, kv_row_f32);

        let n_rot = params.n_rot as u32;
        let n_nope = (params.head_dim as u32).saturating_sub(n_rot);

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&cache_buf), 0);
        enc.set_buffer(1, Some(&row_buf), 0);
        set_scalar_bytes(enc, 2, &n_nope);
        set_scalar_bytes(enc, 3, &n_rot);
        set_scalar_bytes(enc, 4, &slot);

        let chunks = (n_nope / 64).max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(chunks.min(1024), 1, 1),
        );
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "kv_fp8_store_persistent_impl");
        Ok(cache_buf)
    }

    /// `dsv4_kv_fp8_store` — per-row FP8 round-trip + n_rot tail half-cast,
    /// appended to the raw KV cache.
    ///
    /// Buffers (registry: Writable, Writable): p0=kv_cache, p1=row (mutable
    /// scratch view in antirez's API). Uniforms (Scalars 3): n_nope (u32),
    /// n_rot (u32), slot (u32). Dispatch: 1 threadgroup, n_nope/64 threads.
    #[allow(dead_code)]
    pub(crate) fn kv_fp8_store_impl(
        &self,
        params: &LayerParams,
        kv_row_f32: &[f32],
        view: &mut KvCacheView<'_>,
    ) -> Result<()> {
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(kv_row_f32.len() == row, "kv_fp8_store: row.len mismatch");
        anyhow::ensure!(view.raw_cap > 0, "kv_fp8_store: raw_cap must be > 0");
        anyhow::ensure!(
            view.raw.len() >= (view.raw_cap as usize) * row,
            "kv_fp8_store: view.raw smaller than raw_cap * row"
        );
        let pipeline = self
            .pipelines
            .get("ds4_dsv4_kv_fp8_store")
            .ok_or_else(|| anyhow::anyhow!("dsv4_kv_fp8_store pipeline not loaded"))?;

        // Stage the f32 source through a writable buffer; the kernel
        // FP8-quantizes in place. A real production wire-up will mmap
        // the existing cache buffer instead of copying every step.
        let row_buf = new_input_buffer(&self.device, kv_row_f32);
        let cache_bytes = view.raw.len() * std::mem::size_of::<f32>();
        let cache_buf = self.device.new_buffer_with_data(
            view.raw.as_ptr() as *const _,
            cache_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // FP8 quantization runs over the NOPE prefix only — the last
        // `n_rot` slots of the row are the rope tail and stay verbatim.
        // Antirez `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1486-1504)
        // uses `n_nope = head_dim - n_rot`. With `n_lora_kv == head_dim`
        // under the new layout, the same definition holds here.
        let n_rot = params.n_rot as u32;
        let n_nope = (params.head_dim as u32).saturating_sub(n_rot);
        let slot = view.pos % view.raw_cap;

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&cache_buf), 0);
        enc.set_buffer(1, Some(&row_buf), 0);
        set_scalar_bytes(enc, 2, &n_nope);
        set_scalar_bytes(enc, 3, &n_rot);
        set_scalar_bytes(enc, 4, &slot);

        let chunks = (n_nope / 64).max(1) as u64;
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(chunks.min(1024), 1, 1));
        end_shared_compute_enc(enc);
        self.commit_wait_traced(cmd_buf, "kv_fp8_store_impl");

        // Read the updated cache back into the caller's slice + bump pos.
        let updated = unsafe { read_buffer::<f32>(&cache_buf, view.raw.len()) };
        view.raw.copy_from_slice(&updated);
        view.pos = view.pos.saturating_add(1);
        Ok(())
    }

    /// `flash_attn_ext_vec_f16_dk512_dv512` (M124) — decode-shape flash
    /// attention. Heavyweight: 32-field `ds4_metal_args_flash_attn_ext_vec`
    /// + 7 buffers (uniform, Q, K, V, mask, sinks, pad, tmp) and a
    /// `kernel_flash_attn_ext_vec_reduce` second pass.
    ///
    /// Routes to Metal when the antirez vec-kernel preconditions hold:
    ///   - `kv_comp.is_none()` (vec only handles contiguous raw KV)
    ///   - `head_dim == 512` (kernel symbol bakes dk=dv=512)
    ///   - `raw_start == 0` (skip ring-buffer rotation prelude — for
    ///     non-zero starts we still hit the CPU oracle; production
    ///     decode flushes raw_start≠0 rarely).
    ///   - `n_raw % 32 == 0` (avoid the optional pad kernel for the
    ///     first wire; partial-block tails take the CPU path).
    /// Otherwise CPU fallback (CPU oracle, bit-equal to algorithm spec).
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_impl(
        &self,
        params: &LayerParams,
        q: &[f32],
        kv_raw: &[f32],
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        kv_comp: Option<&[f32]>,
        n_comp: u32,
        comp_selected: Option<&[u32]>,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<AttnHeadsOut> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(
            q.len() == n_head * head_dim,
            "flash_attn_decode: q.len ({}) != n_head * head_dim ({} * {})",
            q.len(),
            n_head,
            head_dim
        );
        anyhow::ensure!(
            attn_sinks.len() == n_head,
            "flash_attn_decode: attn_sinks.len ({}) != n_head ({})",
            attn_sinks.len(),
            n_head
        );

        // Real Metal vec+reduce wire when antirez preconditions hold.
        if kv_comp.is_none()
            && head_dim == 512
            && raw_start == 0
            && n_raw % 32 == 0
            && n_raw > 0
            && row == head_dim
        {
            return self.flash_attn_decode_metal(params, q, kv_raw, n_raw, attn_sinks);
        }

        // M5 task #97 — GPU path for compressor/indexer layers. Builds
        // an extended f16 KV (raw rows + indexer-selected comp rows) into
        // a workspace via `build_extended_kv_encode` and chains the
        // existing dk512 flash_attn fast path into the same cb via
        // `flash_attn_decode_metal_encode_with_kv_buf` (step 2: one cb
        // per compressor flash_attn call, no CPU readback).
        //
        // Step 3 (task #97): non-32-aligned `n_total` is handled by
        // rounding the workspace and the flash_attn `n_raw` up to the
        // next multiple of 32. The gather kernel zero-fills the padded
        // rows; the flash_attn mask buffer carries f16 -inf at padded
        // positions so they drop out of the softmax.
        //
        // Preconditions mirror the dense fast path plus:
        //   - `comp_selected.is_some()` (indexer layers; dense-comp
        //     iteration over all n_comp rows is left to CPU for now)
        if let (Some(comp), Some(sel)) = (kv_comp, comp_selected) {
            let n_total = n_raw + n_selected;
            if head_dim == 512 && raw_start == 0 && row == head_dim && n_total > 0 {
                let n_total_padded = (n_total + 31) & !31;
                let cmd_buf = self.command_queue.new_command_buffer();
                let kv_workspace = self.build_extended_kv_encode(
                    cmd_buf,
                    kv_raw,
                    comp,
                    sel,
                    n_raw,
                    n_selected,
                    head_dim as u32,
                    n_total_padded,
                )?;
                let out_buf = self.flash_attn_decode_metal_encode_with_kv_buf(
                    cmd_buf,
                    params,
                    q,
                    &kv_workspace,
                    n_total_padded,
                    attn_sinks,
                    Some(n_total),
                )?;
                self.commit_wait_traced(cmd_buf, "flash_attn_decode_metal_compressor");
                return Ok(unsafe { read_buffer::<f32>(&out_buf, n_head * head_dim) });
            }
        }

        flash_attn_decode_cpu_fallback(
            n_head, head_dim, row,
            q, kv_raw, n_raw, raw_cap, raw_start,
            kv_comp, n_comp, comp_selected, n_selected,
            attn_sinks,
        )
    }

    /// Real Metal vec+reduce wire of `kernel_flash_attn_ext_vec_f16_dk512_dv512`.
    /// Called by `flash_attn_decode_impl` when antirez preconditions hold:
    /// raw KV only, head_dim=512, raw_start=0, n_raw%ncpsg==0, row==head_dim.
    ///
    /// Two passes via `specialized_pipeline`:
    ///   - vec: 9 FCs (5 bool at 400-404 + 4 int at 420-423), 7 buffers,
    ///     dispatch (1, n_head, nwg) groups × (32, nsg, 1) threads.
    ///   - reduce: 2 FCs (DV at 500, NWG at 501), 3 buffers,
    ///     dispatch (n_head, 1, 1) × (32*nwg, 1, 1).
    ///
    /// KV is f32→f16 staged once and bound to both K and V buffers
    /// (antirez: same buffer; vec kernel reads via `ns10`/`ns20` strides).
    #[allow(dead_code, clippy::too_many_arguments)]
    /// Phase E M5.2.3: `flash_attn_decode` that reads KV cache from the
    /// persistent per-layer buffer (`kv_buffer_or_alloc`) instead of a
    /// CPU `kv_raw: &[f32]` slice.
    ///
    /// Delegates to `flash_attn_decode_impl` after reinterpreting the
    /// persistent buffer's contents as a `&[f32]` — same kernel routing
    /// (Metal fast path when antirez preconditions hold, CPU fallback
    /// otherwise) and same outputs. The fallback is what lets the M5.4
    /// unified-cb decoder run at pos < 31 (n_raw < 32) without bailing.
    ///
    /// Safety: the persistent buffer is StorageModeShared (CPU-visible).
    /// Unwritten slots are zero-initialised by Metal. After
    /// `kv_fp8_store_persistent` writes a slot, that slot holds valid
    /// f32 values. Re-interpreting as `&[f32]` is well-defined.
    #[allow(dead_code, clippy::too_many_arguments)]
    /// Phase F task #86 — encode `flash_attn_decode_metal` fast-path
    /// into a caller-provided cb (no commit, no wait, no readback).
    /// Returns the output metal::Buffer. Only honors the fast-path
    /// preconditions (head_dim=512, n_raw 32-aligned, no kv_comp,
    /// no comp_selected, raw_start=0); caller must use the non-scope
    /// `flash_attn_decode_metal_persistent` for the general case.
    pub(crate) fn flash_attn_decode_metal_persistent_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let row = params.n_lora_kv as usize;
        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        let kv_raw: &[f32] = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const f32, total_elements)
        };
        self.flash_attn_decode_metal_encode(cmd_buf, params, q, kv_raw, n_raw, attn_sinks)
    }

    /// Resident-`q` variant of [`Self::flash_attn_decode_metal_persistent_encode`]:
    /// reads the persistent f32 KV buffer, f16-stages it, and binds a
    /// caller-owned resident `q_buf` (no `&[f32]` upload). Same compute.
    ///
    /// WIP (single-cb decode rewrite, step 1): wired through
    /// `BatchScope::flash_attn_decode_persistent_qbuf` but not yet used by
    /// the layer loop — the fused raw-layer path that consumes it still
    /// needs split offset-binding + cur_hc residency. Allow dead_code until
    /// then.
    #[allow(dead_code)]
    pub(crate) fn flash_attn_decode_metal_persistent_encode_qbuf(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        n_raw: u32,
        raw_cap: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        let kv_raw: &[f32] = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const f32, total_elements)
        };
        let kv_n = (n_raw as usize) * head_dim;
        let mut kv_f16 = vec![0u16; kv_n];
        for (i, &v) in kv_raw.iter().take(kv_n).enumerate() {
            kv_f16[i] = crate::f16_cast::f32_to_f16_bits(v);
        }
        let kv_buf = new_input_buffer(&self.device, &kv_f16);
        self.flash_attn_decode_metal_encode_with_kv_qbuf(
            cmd_buf, params, q_buf, &kv_buf, n_raw, attn_sinks, None,
        )
    }

    /// M5 task #100 — scope-aware compressor flash_attn. Mirrors the
    /// compressor branch of `flash_attn_decode_impl` (build_extended_kv
    /// + flash_attn_decode_metal_encode_with_kv_buf), but encodes into
    /// a caller-provided `cmd_buf` and returns the output buffer without
    /// committing/waiting/reading. Enables the slow_path post-attn
    /// pipeline to compose into one BatchScope per layer.
    ///
    /// Preconditions (asserted): head_dim == 512, row == head_dim,
    /// n_raw + n_selected > 0, head_dim==512 plus the kernel's
    /// 32-aligned-total requirement (handled internally by padding to
    /// 32 with f16-(-inf) mask, same as the inherent compressor branch).
    /// Caller must ensure `n_raw < raw_cap` so the persistent buffer
    /// slice is within bounds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_metal_persistent_compressor_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        kv_comp: &[f32],
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(
            head_dim == 512,
            "flash_attn_decode_persistent_compressor: head_dim must be 512"
        );
        anyhow::ensure!(
            row == head_dim,
            "flash_attn_decode_persistent_compressor: n_lora_kv ({}) must equal head_dim ({})",
            row,
            head_dim
        );
        let n_total = n_raw + n_selected;
        anyhow::ensure!(
            n_total > 0,
            "flash_attn_decode_persistent_compressor: n_raw + n_selected must be > 0"
        );
        let n_total_padded = (n_total + 31) & !31;

        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        let kv_raw: &[f32] = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const f32, total_elements)
        };

        let kv_workspace = self.build_extended_kv_encode(
            cmd_buf,
            kv_raw,
            kv_comp,
            comp_selected,
            n_raw,
            n_selected,
            head_dim as u32,
            n_total_padded,
        )?;
        self.flash_attn_decode_metal_encode_with_kv_buf(
            cmd_buf,
            params,
            q,
            &kv_workspace,
            n_total_padded,
            attn_sinks,
            Some(n_total),
        )
    }

    /// Resident-`q` variant of
    /// [`Self::flash_attn_decode_metal_persistent_compressor_encode`]: binds a
    /// caller-owned `q_buf` instead of uploading `&[f32]`. Used by the fused
    /// layer path — raw layers pass empty `kv_comp`/`comp_selected` (n_sel=0),
    /// so the padded workspace is just the raw KV rows (any n_raw, padded to
    /// 32 + masked), letting one resident-q flash run for non-32-aligned n_raw.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_metal_persistent_compressor_encode_qbuf(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        n_raw: u32,
        raw_cap: u32,
        kv_comp: &[f32],
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(head_dim == 512, "compressor_qbuf: head_dim must be 512");
        anyhow::ensure!(row == head_dim, "compressor_qbuf: n_lora_kv must equal head_dim");
        let n_total = n_raw + n_selected;
        anyhow::ensure!(n_total > 0, "compressor_qbuf: n_raw + n_selected must be > 0");
        let n_total_padded = (n_total + 31) & !31;

        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        let kv_raw: &[f32] = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const f32, total_elements)
        };
        // Reuse this layer's flash scratch (workspace + tmp) across tokens
        // instead of allocating fresh every call. Reset to -1 afterward so the
        // shared encoders fall back to fresh allocation for all other callers.
        use std::sync::atomic::Ordering;
        self.flash_scratch_layer.store(if flash_scratch_reuse_enabled() && !self.flash_scratch_suppress.load(Ordering::Relaxed) { layer_idx as i64 } else { -1 }, Ordering::Relaxed);
        let out = (|| -> Result<metal::Buffer> {
            let kv_workspace = self.build_extended_kv_encode(
                cmd_buf, kv_raw, kv_comp, comp_selected, n_raw, n_selected,
                head_dim as u32, n_total_padded,
            )?;
            self.flash_attn_decode_metal_encode_with_kv_qbuf(
                cmd_buf, params, q_buf, &kv_workspace, n_total_padded, attn_sinks, Some(n_total),
            )
        })();
        self.flash_scratch_layer.store(-1, Ordering::Relaxed);
        out
    }

    /// Single-cb step 9 — GPU-resident-ring variant of
    /// [`Self::flash_attn_decode_metal_persistent_compressor_encode_qbuf`]:
    /// `comp_rows` is read from the caller-owned ring buffer
    /// (`comp_ring_or_alloc`) instead of an uploaded `&[f32]`, so the whole
    /// flash gather has no CPU comp-row dependency.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_metal_persistent_compressor_encode_qbuf_gpuring(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        n_raw: u32,
        raw_cap: u32,
        kv_comp_buf: &metal::Buffer,
        comp_selected: &[u32],
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(head_dim == 512, "compressor_qbuf_gpuring: head_dim must be 512");
        anyhow::ensure!(row == head_dim, "compressor_qbuf_gpuring: n_lora_kv must equal head_dim");
        let n_total = n_raw + n_selected;
        anyhow::ensure!(n_total > 0, "compressor_qbuf_gpuring: n_raw + n_selected must be > 0");
        let n_total_padded = (n_total + 31) & !31;

        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        use std::sync::atomic::Ordering;
        self.flash_scratch_layer.store(if flash_scratch_reuse_enabled() && !self.flash_scratch_suppress.load(Ordering::Relaxed) { layer_idx as i64 } else { -1 }, Ordering::Relaxed);
        let out = (|| -> Result<metal::Buffer> {
            let kv_workspace = self.build_extended_kv_encode_gpubuf(
                cmd_buf, &buf, kv_comp_buf, comp_selected, n_raw, n_selected,
                head_dim as u32, n_total_padded,
            )?;
            self.flash_attn_decode_metal_encode_with_kv_qbuf(
                cmd_buf, params, q_buf, &kv_workspace, n_total_padded, attn_sinks, Some(n_total),
            )
        })();
        self.flash_scratch_layer.store(-1, Ordering::Relaxed);
        out
    }

    /// Fully-resident-selection variant of
    /// [`Self::flash_attn_decode_metal_persistent_compressor_encode_qbuf_gpuring`]:
    /// the indexer top-k `sel_buf` (GPU i32 indices from `encode_indexer_topk`)
    /// is bound straight into the extended-KV gather (no CPU `comp_selected`
    /// slice). Used by the long-context bridge so indexer layers (post_n_index >
    /// top_k) stay chained with zero per-token drains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_metal_persistent_compressor_encode_qbuf_gpuring_sel(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        layer_idx: u32,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        n_raw: u32,
        raw_cap: u32,
        kv_comp_buf: &metal::Buffer,
        sel_buf: &metal::Buffer,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        let row = params.n_lora_kv as usize;
        anyhow::ensure!(head_dim == 512, "compressor_qbuf_gpuring_sel: head_dim must be 512");
        anyhow::ensure!(row == head_dim, "compressor_qbuf_gpuring_sel: n_lora_kv must equal head_dim");
        let n_total = n_raw + n_selected;
        anyhow::ensure!(n_total > 0, "compressor_qbuf_gpuring_sel: n_raw + n_selected must be > 0");
        let n_total_padded = (n_total + 31) & !31;

        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        use std::sync::atomic::Ordering;
        self.flash_scratch_layer.store(if flash_scratch_reuse_enabled() && !self.flash_scratch_suppress.load(Ordering::Relaxed) { layer_idx as i64 } else { -1 }, Ordering::Relaxed);
        let out = (|| -> Result<metal::Buffer> {
            let kv_workspace = self.build_extended_kv_encode_gpubuf_sel(
                cmd_buf, &buf, kv_comp_buf, sel_buf, n_raw, n_selected,
                head_dim as u32, n_total_padded,
            )?;
            self.flash_attn_decode_metal_encode_with_kv_qbuf(
                cmd_buf, params, q_buf, &kv_workspace, n_total_padded, attn_sinks, Some(n_total),
            )
        })();
        self.flash_scratch_layer.store(-1, Ordering::Relaxed);
        out
    }

    pub(crate) fn flash_attn_decode_metal_persistent(
        &self,
        layer_idx: u32,
        params: &LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        kv_comp: Option<&[f32]>,
        n_comp: u32,
        comp_selected: Option<&[u32]>,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> Result<AttnHeadsOut> {
        let row = params.n_lora_kv as usize;
        let total_elements = (raw_cap as usize) * row;
        let byte_size = total_elements * std::mem::size_of::<f32>();
        let buf = self.kv_buffer_or_alloc(layer_idx, byte_size);
        // The slice borrows from `buf` for the duration of this fn call.
        // Both the Metal fast path (f32→f16 staged once) and the CPU
        // fallback consume `kv_raw` synchronously, so the borrow ends
        // before this function returns. `buf` (Arc clone) outlives it.
        let kv_raw: &[f32] = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const f32, total_elements)
        };
        // Phase E M5.4.5.4: route through flash_attn_decode_impl (not
        // directly to flash_attn_decode_metal) so the CPU fallback fires
        // when n_raw < 32 or head_dim != 512 — unblocks pos < 31 and
        // synthetic test shapes.
        //
        // Phase E M5.4.5.6: compressor/indexer layers feed kv_comp
        // (the compressed-KV rolling cache) + comp_selected (indexer
        // top-k bitmap) here. Non-compressor layers pass None / 0.
        self.flash_attn_decode_impl(
            params,
            q,
            kv_raw,
            n_raw,
            raw_cap,
            0,
            kv_comp,
            n_comp,
            comp_selected,
            n_selected,
            attn_sinks,
        )
    }

    fn flash_attn_decode_metal(
        &self,
        params: &LayerParams,
        q: &[f32],
        kv_raw: &[f32],
        n_raw: u32,
        attn_sinks: &[f32],
    ) -> Result<AttnHeadsOut> {
        let cmd_buf = self.command_queue.new_command_buffer();
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let out_buf = self.flash_attn_decode_metal_encode(
            cmd_buf, params, q, kv_raw, n_raw, attn_sinks,
        )?;
        self.commit_wait_traced(cmd_buf, "flash_attn_decode_metal");
        Ok(unsafe { read_buffer::<f32>(&out_buf, n_head * head_dim) })
    }

    /// Phase F task #86 — extract the encoding portion of
    /// `flash_attn_decode_metal` so it can be invoked into an
    /// external command buffer (no commit, no wait, no readback).
    /// Returns the output metal::Buffer. Caller owns commit+wait+read.
    ///
    /// All constraints from `flash_attn_decode_metal` still apply:
    /// head_dim == 512, n_raw 32-aligned, kv_raw at least n_raw*head_dim.
    ///
    /// Thin wrapper over `flash_attn_decode_metal_encode_with_kv_buf`:
    /// stages `kv_raw` f32 → f16 on CPU into a fresh MTLBuffer, then
    /// delegates. Compressor/indexer callers that already have a GPU-
    /// resident f16 KV workspace (built by `build_extended_kv_encode`)
    /// should call `_with_kv_buf` directly to keep gather+flash_attn in
    /// the same cb.
    pub(crate) fn flash_attn_decode_metal_encode(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        params: &LayerParams,
        q: &[f32],
        kv_raw: &[f32],
        n_raw: u32,
        attn_sinks: &[f32],
    ) -> Result<metal::Buffer> {
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            kv_raw.len() >= (n_raw as usize) * head_dim,
            "flash_attn_decode_metal: kv_raw too small"
        );
        let kv_n = (n_raw as usize) * head_dim;
        let mut kv_f16 = vec![0u16; kv_n];
        for (i, &v) in kv_raw.iter().take(kv_n).enumerate() {
            kv_f16[i] = crate::f16_cast::f32_to_f16_bits(v);
        }
        let kv_buf = new_input_buffer(&self.device, &kv_f16);
        self.flash_attn_decode_metal_encode_with_kv_buf(
            cmd_buf, params, q, &kv_buf, n_raw, attn_sinks, None,
        )
    }

    /// M5 task #97 step 2 — encode the flash_attn dk512 fast path with
    /// a caller-provided f16 KV `metal::Buffer`. Same kernel constraints
    /// as `flash_attn_decode_metal_encode` (head_dim == 512, n_raw 32-
    /// aligned). The buffer must hold at least `n_raw * head_dim` f16
    /// elements at offset 0.
    ///
    /// `n_raw_valid` (step 3): when `Some(v)`, mask entries in
    /// `[v..n_raw]` are set to f16 `-inf` so the kernel's softmax drops
    /// those rows. The caller uses this to round n_raw up to the
    /// kernel's 32-row alignment (the gather kernel zero-fills
    /// `[v..n_raw]` in the workspace). `None` keeps the dense-path
    /// behavior of an all-zero mask.
    ///
    /// This is what makes one-cb compressor flash_attn possible: the
    /// gather workspace built by `build_extended_kv_encode` (f16) is fed
    /// straight in, no CPU readback, no per-call f32→f16 staging.
    pub(crate) fn flash_attn_decode_metal_encode_with_kv_buf(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        params: &LayerParams,
        q: &[f32],
        kv_buf: &metal::Buffer,
        n_raw: u32,
        attn_sinks: &[f32],
        n_raw_valid: Option<u32>,
    ) -> Result<metal::Buffer> {
        let q_buf = new_input_buffer(&self.device, q);
        self.flash_attn_decode_metal_encode_with_kv_qbuf(
            cmd_buf, params, &q_buf, kv_buf, n_raw, attn_sinks, n_raw_valid,
        )
    }

    /// Resident-`q` variant of [`Self::flash_attn_decode_metal_encode_with_kv_buf`]:
    /// binds a caller-owned `q_buf` (f32, `n_head*head_dim` elems) instead of
    /// uploading a `&[f32]`. Lets the orchestrator keep rope_q's output
    /// GPU-resident and chain flash_attn in the same cb without a readback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_metal_encode_with_kv_qbuf(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        kv_buf: &metal::Buffer,
        n_raw: u32,
        attn_sinks: &[f32],
        n_raw_valid: Option<u32>,
    ) -> Result<metal::Buffer> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        anyhow::ensure!(
            head_dim == 512,
            "flash_attn_decode_metal: head_dim must be 512"
        );
        anyhow::ensure!(
            n_raw % 32 == 0,
            "flash_attn_decode_metal: n_raw must be 32-aligned"
        );

        let ncpsg: u32 = 32;
        let nwg: u32 = 32;
        let nsg: u32 = {
            // antirez ds4_metal_flash_attn_vec_nsg
            let mut n = 1u32;
            while 2 * nwg * n * ncpsg < n_raw && n < 4 {
                n *= 2;
            }
            n
        };
        let row_bytes_f32 = head_dim * std::mem::size_of::<f32>();
        let row_bytes_f16 = head_dim * std::mem::size_of::<u16>();
        let mask_bytes = (n_raw as usize) * std::mem::size_of::<u16>();

        // Mask: zero at valid positions, f16 -inf at padded tail (step 3).
        // The kernel adds mask to scaled Q·K scores; -inf drops the row
        // from softmax. Caller passes Some(v) when n_raw was rounded up
        // for the 32-row alignment requirement.
        let mut mask_data = vec![0u16; n_raw as usize];
        if let Some(v) = n_raw_valid {
            let v_usz = v as usize;
            debug_assert!(
                v_usz <= n_raw as usize,
                "n_raw_valid ({}) must be <= n_raw ({})",
                v,
                n_raw,
            );
            const F16_NEG_INF: u16 = 0xFC00;
            for slot in &mut mask_data[v_usz..] {
                *slot = F16_NEG_INF;
            }
        }
        let mask_buf = new_input_buffer(&self.device, &mask_data);

        let sinks_buf = new_input_buffer(&self.device, attn_sinks);

        // pad buffer required by FC has_kvpad path; we only run when
        // n_raw%ncpsg==0 so has_kvpad=false, but bind a dummy anyway
        // because the kernel signature still references it.
        let pad_dummy = vec![0u16; (2 * ncpsg as usize) * head_dim + ncpsg as usize];
        let pad_buf = new_input_buffer(&self.device, &pad_dummy);

        // tmp buffer holds per-workgroup partial heads + (m,l) reduction.
        //   tmp_bytes = nrows*head_dim*nwg*f32 + nrows*(2*nwg)*f32
        let tmp_elems = n_head * head_dim * (nwg as usize) + n_head * 2 * (nwg as usize);
        // Reuse this layer's per-workgroup reduction scratch across tokens when
        // the persistent chain set a layer (else fresh alloc — unchanged). The
        // flash vec kernel writes every workgroup slot before the reduce reads,
        // so no zero-init is relied on.
        let tmp_buf = {
            let l = self.flash_scratch_layer.load(std::sync::atomic::Ordering::Relaxed);
            if l >= 0 {
                self.flash_scratch_or_alloc(l as u32, 0, tmp_elems * std::mem::size_of::<f32>())
            } else {
                new_output_buffer::<f32>(&self.device, tmp_elems)
            }
        };

        // ── vec pipeline: 5 bool FC + 4 int32 FC.
        let has_mask: bool = true;
        let has_sinks: bool = true;
        let has_bias: bool = false;
        let has_scap: bool = false;
        let has_kvpad: bool = false; // n_raw % ncpsg == 0 ⇒ no pad needed
        let ns10: i32 = head_dim as i32;
        let ns20: i32 = head_dim as i32;
        let nsg_i: i32 = nsg as i32;
        let nwg_i: i32 = nwg as i32;

        let mut vec_key = Vec::with_capacity(21);
        vec_key.push(has_mask as u8);
        vec_key.push(has_sinks as u8);
        vec_key.push(has_bias as u8);
        vec_key.push(has_scap as u8);
        vec_key.push(has_kvpad as u8);
        vec_key.extend_from_slice(&ns10.to_le_bytes());
        vec_key.extend_from_slice(&ns20.to_le_bytes());
        vec_key.extend_from_slice(&nsg_i.to_le_bytes());
        vec_key.extend_from_slice(&nwg_i.to_le_bytes());

        let vec_pipeline = self.specialized_pipeline(
            "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512",
            &vec_key,
            |fcv| {
                use metal::MTLDataType;
                fcv.set_constant_value_at_index(
                    &has_mask as *const _ as *const _,
                    MTLDataType::Bool,
                    400,
                );
                fcv.set_constant_value_at_index(
                    &has_sinks as *const _ as *const _,
                    MTLDataType::Bool,
                    401,
                );
                fcv.set_constant_value_at_index(
                    &has_bias as *const _ as *const _,
                    MTLDataType::Bool,
                    402,
                );
                fcv.set_constant_value_at_index(
                    &has_scap as *const _ as *const _,
                    MTLDataType::Bool,
                    403,
                );
                fcv.set_constant_value_at_index(
                    &has_kvpad as *const _ as *const _,
                    MTLDataType::Bool,
                    404,
                );
                fcv.set_constant_value_at_index(
                    &ns10 as *const _ as *const _,
                    MTLDataType::Int,
                    420,
                );
                fcv.set_constant_value_at_index(
                    &ns20 as *const _ as *const _,
                    MTLDataType::Int,
                    421,
                );
                fcv.set_constant_value_at_index(
                    &nsg_i as *const _ as *const _,
                    MTLDataType::Int,
                    422,
                );
                fcv.set_constant_value_at_index(
                    &nwg_i as *const _ as *const _,
                    MTLDataType::Int,
                    423,
                );
            },
        )?;

        // ── reduce pipeline.
        let dv: i32 = head_dim as i32;
        let mut red_key = Vec::with_capacity(8);
        red_key.extend_from_slice(&dv.to_le_bytes());
        red_key.extend_from_slice(&nwg_i.to_le_bytes());
        let reduce_pipeline =
            self.specialized_pipeline("ds4_kernel_flash_attn_ext_vec_reduce", &red_key, |fcv| {
                use metal::MTLDataType;
                fcv.set_constant_value_at_index(&dv as *const _ as *const _, MTLDataType::Int, 500);
                fcv.set_constant_value_at_index(
                    &nwg_i as *const _ as *const _,
                    MTLDataType::Int,
                    501,
                );
            })?;

        // ── vec_args: 192 bytes, byte-explicit per ds4_metal.m:8681.
        let mut vec_args = [0u8; 192];
        // ne01=1, ne02=n_head, ne03=1
        vec_args[0..4].copy_from_slice(&1i32.to_le_bytes());
        vec_args[4..8].copy_from_slice(&(n_head as i32).to_le_bytes());
        vec_args[8..12].copy_from_slice(&1i32.to_le_bytes());
        // nb01 @16 = n_head*row_bytes, nb02 @24 = row_bytes, nb03 @32 = n_head*row_bytes
        vec_args[16..24].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());
        vec_args[24..32].copy_from_slice(&(row_bytes_f32 as u64).to_le_bytes());
        vec_args[32..40].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());
        // ne11=n_raw @40, ne_12_2=1 @44, ne_12_3=1 @48, ns10=head_dim @52
        vec_args[40..44].copy_from_slice(&(n_raw as i32).to_le_bytes());
        vec_args[44..48].copy_from_slice(&1i32.to_le_bytes());
        vec_args[48..52].copy_from_slice(&1i32.to_le_bytes());
        vec_args[52..56].copy_from_slice(&(head_dim as i32).to_le_bytes());
        // nb11 @56 = row_bytes_f16, nb12 @64 = n_raw*row_bytes_f16, nb13 @72 = same
        vec_args[56..64].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());
        vec_args[64..72].copy_from_slice(&((n_raw as usize * row_bytes_f16) as u64).to_le_bytes());
        vec_args[72..80].copy_from_slice(&((n_raw as usize * row_bytes_f16) as u64).to_le_bytes());
        // ns20 @80 = head_dim
        vec_args[80..84].copy_from_slice(&(head_dim as i32).to_le_bytes());
        // nb21 @88, nb22 @96, nb23 @104 — V strides (same as K)
        vec_args[88..96].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());
        vec_args[96..104].copy_from_slice(&((n_raw as usize * row_bytes_f16) as u64).to_le_bytes());
        vec_args[104..112]
            .copy_from_slice(&((n_raw as usize * row_bytes_f16) as u64).to_le_bytes());
        // ne31=1 @112, ne32=1 @116, ne33=1 @120
        vec_args[112..116].copy_from_slice(&1i32.to_le_bytes());
        vec_args[116..120].copy_from_slice(&1i32.to_le_bytes());
        vec_args[120..124].copy_from_slice(&1i32.to_le_bytes());
        // nb31 @128, nb32 @136, nb33 @144 — mask byte strides
        vec_args[128..136].copy_from_slice(&(mask_bytes as u64).to_le_bytes());
        vec_args[136..144].copy_from_slice(&(mask_bytes as u64).to_le_bytes());
        vec_args[144..152].copy_from_slice(&(mask_bytes as u64).to_le_bytes());
        // ne1=n_head @152, ne2=1 @156, ne3=1 @160
        vec_args[152..156].copy_from_slice(&(n_head as i32).to_le_bytes());
        vec_args[156..160].copy_from_slice(&1i32.to_le_bytes());
        vec_args[160..164].copy_from_slice(&1i32.to_le_bytes());
        // scale @164 = 1/sqrt(head_dim)
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        vec_args[164..168].copy_from_slice(&scale.to_le_bytes());
        // max_bias @168, m0 @172, m1 @176 = 0
        // n_head_log2 @180 = 0
        // logit_softcap @184 = 0
        // (remaining 4 bytes = trailing pad to 192, already zero)

        // ── threadgroup memory for vec pass.
        //   shared_elems = (align_up(head_dim, 128) + 4*ncpsg + 2*align_up(head_dim, 128)) * nsg
        //   shared_bytes = align_up(shared_elems * sizeof(half), 16)
        let align_up = |x: usize, a: usize| ((x + a - 1) / a) * a;
        let shared_elems =
            (align_up(head_dim, 128) + 4 * ncpsg as usize + 2 * align_up(head_dim, 128))
                * (nsg as usize);
        let shared_bytes = align_up(shared_elems * 2, 16) as u64;

        // ── vec dispatch (uses caller-provided cmd_buf).
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&vec_pipeline);
        enc.set_bytes(0, vec_args.len() as u64, vec_args.as_ptr() as *const _);
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(kv_buf), 0);
        enc.set_buffer(3, Some(kv_buf), 0);
        enc.set_buffer(4, Some(&mask_buf), 0);
        enc.set_buffer(5, Some(&sinks_buf), 0);
        enc.set_buffer(6, Some(&pad_buf), 0);
        enc.set_buffer(7, Some(&tmp_buf), 0);
        enc.set_threadgroup_memory_length(0, shared_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(1, n_head as u64, nwg as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);

        // ── reduce dispatch.
        let mut reduce_args = [0u8; 4];
        reduce_args[0..4].copy_from_slice(&(n_head as i32).to_le_bytes());
        let out_buf = new_output_buffer::<f32>(&self.device, n_head * head_dim);
        let enc2 = shared_compute_enc(cmd_buf);
        enc2.set_compute_pipeline_state(&reduce_pipeline);
        enc2.set_bytes(
            0,
            reduce_args.len() as u64,
            reduce_args.as_ptr() as *const _,
        );
        enc2.set_buffer(1, Some(&tmp_buf), 0);
        enc2.set_buffer(2, Some(&out_buf), 0);
        enc2.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32 * nwg as u64, 1, 1),
        );
        enc2.end_encoding();

        Ok(out_buf)
    }

    /// Lever A: K-QUERY batched flash over a SHARED f16 KV workspace + a per-query
    /// causal mask. The same efficient `flash_attn_ext_vec` kernel as the per-position
    /// path, dispatched with ne01=K (iq1=tgpig[0] indexes the query) and a [K, n_total]
    /// mask (FC_has_mask). q_buf = [K, n_head, head_dim] f32; kv_workspace = [n_total,
    /// head_dim] f16 (n_total % 32 == 0); mask = [K * n_total] f16 (0 = visible, -inf =
    /// masked, incl. the padded tail). Returns out [K, n_head, head_dim] f32. Replaces K
    /// per-position flashes (each of which re-gathers an O(n) workspace → O(n²)) with ONE
    /// shared-workspace gather + ONE batched dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_decode_k_metal(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        kv_workspace: &metal::Buffer,
        n_total: u32,
        mask: &[u16],
        attn_sinks: &[f32],
        k_positions: usize,
    ) -> Result<metal::Buffer> {
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let k = k_positions;
        anyhow::ensure!(head_dim == 512, "flash_attn_decode_k: head_dim must be 512");
        anyhow::ensure!(n_total % 32 == 0, "flash_attn_decode_k: n_total must be 32-aligned");
        anyhow::ensure!(mask.len() == k * n_total as usize,
            "flash_attn_decode_k: mask len {} != K*n_total = {}*{}", mask.len(), k, n_total);

        let ncpsg: u32 = 32;
        let nwg: u32 = 32;
        let nsg: u32 = { let mut n = 1u32; while 2 * nwg * n * ncpsg < n_total && n < 4 { n *= 2; } n };
        let row_bytes_f32 = head_dim * std::mem::size_of::<f32>();
        let row_bytes_f16 = head_dim * std::mem::size_of::<u16>();

        let mask_buf = new_input_buffer(&self.device, mask);
        let sinks_buf = new_input_buffer(&self.device, attn_sinks);
        let pad_dummy = vec![0u16; (2 * ncpsg as usize) * head_dim + ncpsg as usize];
        let pad_buf = new_input_buffer(&self.device, &pad_dummy);

        // tmp: nrows = K*n_head partials.
        let nrows = k * n_head;
        let tmp_elems = nrows * head_dim * (nwg as usize) + nrows * 2 * (nwg as usize);
        let tmp_buf = new_output_buffer::<f32>(&self.device, tmp_elems);

        let has_mask = true; let has_sinks = true;
        let has_bias = false; let has_scap = false; let has_kvpad = false;
        let ns10: i32 = head_dim as i32; let ns20: i32 = head_dim as i32;
        let nsg_i: i32 = nsg as i32; let nwg_i: i32 = nwg as i32;
        let mut vec_key = Vec::with_capacity(21);
        for b in [has_mask, has_sinks, has_bias, has_scap, has_kvpad] { vec_key.push(b as u8); }
        vec_key.extend_from_slice(&ns10.to_le_bytes());
        vec_key.extend_from_slice(&ns20.to_le_bytes());
        vec_key.extend_from_slice(&nsg_i.to_le_bytes());
        vec_key.extend_from_slice(&nwg_i.to_le_bytes());
        let vec_pipeline = self.specialized_pipeline(
            "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512", &vec_key, |fcv| {
                use metal::MTLDataType;
                fcv.set_constant_value_at_index(&has_mask as *const _ as *const _, MTLDataType::Bool, 400);
                fcv.set_constant_value_at_index(&has_sinks as *const _ as *const _, MTLDataType::Bool, 401);
                fcv.set_constant_value_at_index(&has_bias as *const _ as *const _, MTLDataType::Bool, 402);
                fcv.set_constant_value_at_index(&has_scap as *const _ as *const _, MTLDataType::Bool, 403);
                fcv.set_constant_value_at_index(&has_kvpad as *const _ as *const _, MTLDataType::Bool, 404);
                fcv.set_constant_value_at_index(&ns10 as *const _ as *const _, MTLDataType::Int, 420);
                fcv.set_constant_value_at_index(&ns20 as *const _ as *const _, MTLDataType::Int, 421);
                fcv.set_constant_value_at_index(&nsg_i as *const _ as *const _, MTLDataType::Int, 422);
                fcv.set_constant_value_at_index(&nwg_i as *const _ as *const _, MTLDataType::Int, 423);
            })?;
        let dv: i32 = head_dim as i32;
        let mut red_key = Vec::with_capacity(8);
        red_key.extend_from_slice(&dv.to_le_bytes());
        red_key.extend_from_slice(&nwg_i.to_le_bytes());
        let reduce_pipeline = self.specialized_pipeline("ds4_kernel_flash_attn_ext_vec_reduce", &red_key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&dv as *const _ as *const _, MTLDataType::Int, 500);
            fcv.set_constant_value_at_index(&nwg_i as *const _ as *const _, MTLDataType::Int, 501);
        })?;

        let mut a = [0u8; 192];
        // ne01=K, ne02=n_head, ne03=1
        a[0..4].copy_from_slice(&(k as i32).to_le_bytes());
        a[4..8].copy_from_slice(&(n_head as i32).to_le_bytes());
        a[8..12].copy_from_slice(&1i32.to_le_bytes());
        // q strides: nb01 (per query) = n_head*row_f32, nb02 (per head) = row_f32
        a[16..24].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());
        a[24..32].copy_from_slice(&(row_bytes_f32 as u64).to_le_bytes());
        a[32..40].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());
        // ne11 = n_total (KV rows), shared across queries
        a[40..44].copy_from_slice(&(n_total as i32).to_le_bytes());
        a[44..48].copy_from_slice(&1i32.to_le_bytes());
        a[48..52].copy_from_slice(&1i32.to_le_bytes());
        a[52..56].copy_from_slice(&(head_dim as i32).to_le_bytes());
        a[56..64].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());
        a[64..72].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes());
        a[72..80].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes());
        a[80..84].copy_from_slice(&(head_dim as i32).to_le_bytes());
        a[88..96].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());
        a[96..104].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes());
        a[104..112].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes());
        // mask: ne31=1 ne32=1 ne33=1; nb31 = per-query row = n_total*2
        a[112..116].copy_from_slice(&1i32.to_le_bytes());
        a[116..120].copy_from_slice(&1i32.to_le_bytes());
        a[120..124].copy_from_slice(&1i32.to_le_bytes());
        let mask_row = (n_total as usize * 2) as u64;
        a[128..136].copy_from_slice(&mask_row.to_le_bytes());
        a[136..144].copy_from_slice(&mask_row.to_le_bytes());
        a[144..152].copy_from_slice(&mask_row.to_le_bytes());
        // output dims: ne1=n_head, ne2=K, ne3=1 → nrows=ne1*ne2*ne3=K*n_head;
        // rid = iq2 + iq1*ne1 = iq2 + iq1*n_head (query-major [K,n_head]).
        a[152..156].copy_from_slice(&(n_head as i32).to_le_bytes());
        a[156..160].copy_from_slice(&(k as i32).to_le_bytes());
        a[160..164].copy_from_slice(&1i32.to_le_bytes());
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        a[164..168].copy_from_slice(&scale.to_le_bytes());

        let align_up = |x: usize, al: usize| ((x + al - 1) / al) * al;
        let shared_elems = (align_up(head_dim, 128) + 4 * ncpsg as usize + 2 * align_up(head_dim, 128)) * (nsg as usize);
        let shared_bytes = align_up(shared_elems * 2, 16) as u64;

        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&vec_pipeline);
        enc.set_bytes(0, a.len() as u64, a.as_ptr() as *const _);
        enc.set_buffer(1, Some(q_buf), 0);
        enc.set_buffer(2, Some(kv_workspace), 0);
        enc.set_buffer(3, Some(kv_workspace), 0);
        enc.set_buffer(4, Some(&mask_buf), 0);
        enc.set_buffer(5, Some(&sinks_buf), 0);
        enc.set_buffer(6, Some(&pad_buf), 0);
        enc.set_buffer(7, Some(&tmp_buf), 0);
        enc.set_threadgroup_memory_length(0, shared_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(k as u64, n_head as u64, nwg as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);

        let mut reduce_args = [0u8; 4];
        reduce_args[0..4].copy_from_slice(&(nrows as i32).to_le_bytes());
        let out_buf = new_output_buffer::<f32>(&self.device, k * n_head * head_dim);
        let enc2 = shared_compute_enc(cmd_buf);
        enc2.set_compute_pipeline_state(&reduce_pipeline);
        enc2.set_bytes(0, reduce_args.len() as u64, reduce_args.as_ptr() as *const _);
        enc2.set_buffer(1, Some(&tmp_buf), 0);
        enc2.set_buffer(2, Some(&out_buf), 0);
        enc2.dispatch_thread_groups(
            MTLSize::new(nrows as u64, 1, 1),
            MTLSize::new(32 * nwg as u64, 1, 1),
        );
        enc2.end_encoding();
        Ok(out_buf)
    }

    /// Antirez-style NON-VEC block-skipping prefill flash (`DS4_CHUNK_BLKFLASH`).
    /// 2-kernel pipeline (blk mask-scan → ext block-skip flash) — with an SWA-128
    /// mask this skips fully-out-of-window key blocks → O(N·128) instead of the vec
    /// kernel's O(N²), matching antirez's lean all-raw prefill (255.6 vs 192.5 t/s).
    /// `kv_workspace` = [n_total, head_dim] f16; `n_total` MUST be 64-aligned (so
    /// has_kvpad=false, no pad kernel). `mask` = [k_positions, n_total] f16 additive
    /// (0 attend / 0xFC00 -inf) — caller builds it SWA-windowed. Mirrors antirez
    /// ds4_metal.m:17261-17463; args struct is byte-identical to the vec kernel.
    /// UNVALIDATED until needle-checked on a real prompt — a partial flash → NaN.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attn_ext_blk_metal(
        &self,
        cmd_buf: &metal::CommandBufferRef,
        params: &LayerParams,
        q_buf: &metal::Buffer,
        kv_workspace: &metal::Buffer,
        n_total: u32,
        mask: &[u16],
        attn_sinks: &[f32],
        k_positions: usize,
    ) -> Result<metal::Buffer> {
        use metal::MTLDataType;
        let n_head = params.n_head as usize;
        let head_dim = params.head_dim as usize;
        let k = k_positions;
        anyhow::ensure!(head_dim == 512, "flash_attn_ext_blk: head_dim must be 512");
        anyhow::ensure!(n_total % 64 == 0, "flash_attn_ext_blk: n_total must be 64-aligned (got {n_total})");
        anyhow::ensure!(mask.len() == k * n_total as usize,
            "flash_attn_ext_blk: mask len {} != K*n_total = {}*{}", mask.len(), k, n_total);

        // Antirez prefill block-flash tuning (ds4_metal.m:17261).
        let nqptg: u32 = 8;
        let ncpsg: u32 = 64;
        let nsg: i32 = 8; // head_dim>=512
        let bc_mask: bool = (k as u32) % nqptg != 0;
        let nblk0 = n_total.div_ceil(ncpsg);            // ceil(n_raw/64)
        let nblk1 = (k as u32).div_ceil(nqptg);          // ceil(n_tok/8)
        let row_bytes_f32 = head_dim * std::mem::size_of::<f32>();
        let row_bytes_f16 = head_dim * std::mem::size_of::<u16>();

        let mask_buf = new_input_buffer(&self.device, mask);
        let sinks_buf = new_input_buffer(&self.device, attn_sinks);
        let pad_dummy = vec![0u16; 64];
        let pad_buf = new_input_buffer(&self.device, &pad_dummy);
        let align_up = |x: usize, al: usize| x.div_ceil(al) * al;
        let blk_bytes = align_up((nblk0 * nblk1) as usize, 32);
        let blk_buf = new_output_buffer::<u8>(&self.device, blk_bytes);

        // ── blk: mask-block scan → blk_buf markers (FC 224=nqptg, 225=ncpsg). ──
        let nqptg_i = nqptg as i32; let ncpsg_i = ncpsg as i32;
        let mut blk_key = Vec::with_capacity(8);
        blk_key.extend_from_slice(&nqptg_i.to_le_bytes());
        blk_key.extend_from_slice(&ncpsg_i.to_le_bytes());
        let blk_pipeline = self.specialized_pipeline(
            "ds4_kernel_flash_attn_ext_blk", &blk_key, |fcv| {
                fcv.set_constant_value_at_index(&nqptg_i as *const _ as *const _, MTLDataType::Int, 224);
                fcv.set_constant_value_at_index(&ncpsg_i as *const _ as *const _, MTLDataType::Int, 225);
            })?;
        // ds4_metal_args_flash_attn_ext_blk: ne01,ne30,ne31,ne32,ne33 (i32) + nb31,nb32,nb33 (u64).
        let mut bargs = [0u8; 48];
        bargs[0..4].copy_from_slice(&(k as i32).to_le_bytes());            // ne01 = n_tokens
        bargs[4..8].copy_from_slice(&(n_total as i32).to_le_bytes());       // ne30 = n_raw
        bargs[8..12].copy_from_slice(&(k as i32).to_le_bytes());            // ne31
        bargs[12..16].copy_from_slice(&1i32.to_le_bytes());                 // ne32
        bargs[16..20].copy_from_slice(&1i32.to_le_bytes());                 // ne33
        let mask_row = (n_total as usize * 2) as u64;
        let mask_bytes = (k * n_total as usize * 2) as u64;
        bargs[24..32].copy_from_slice(&mask_row.to_le_bytes());             // nb31 (u64, 8-aligned)
        bargs[32..40].copy_from_slice(&mask_bytes.to_le_bytes());           // nb32
        bargs[40..48].copy_from_slice(&mask_bytes.to_le_bytes());           // nb33
        let benc = shared_compute_enc(cmd_buf);
        benc.set_compute_pipeline_state(&blk_pipeline);
        benc.set_bytes(0, bargs.len() as u64, bargs.as_ptr() as *const _);
        benc.set_buffer(1, Some(&mask_buf), 0);
        benc.set_buffer(2, Some(&blk_buf), 0);
        benc.dispatch_thread_groups(
            MTLSize::new(nblk0 as u64, nblk1 as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        end_shared_compute_enc(benc);

        // ── ext: block-skip flash (FC 300 has_mask,301 has_sinks,304 has_kvpad,
        //    310 bc_mask, 320 ns10, 321 ns20, 322 nsg). Args byte-identical to vec. ──
        let has_mask = true; let has_sinks = true; let has_bias = false;
        let has_scap = false; let has_kvpad = false;
        let ns10: i32 = head_dim as i32; let ns20: i32 = head_dim as i32;
        let mut ek = Vec::with_capacity(24);
        for b in [has_mask, has_sinks, has_bias, has_scap, has_kvpad, bc_mask] { ek.push(b as u8); }
        ek.extend_from_slice(&ns10.to_le_bytes());
        ek.extend_from_slice(&ns20.to_le_bytes());
        ek.extend_from_slice(&nsg.to_le_bytes());
        let ext_pipeline = self.specialized_pipeline(
            "ds4_kernel_flash_attn_ext_f16_dk512_dv512", &ek, |fcv| {
                fcv.set_constant_value_at_index(&has_mask as *const _ as *const _, MTLDataType::Bool, 300);
                fcv.set_constant_value_at_index(&has_sinks as *const _ as *const _, MTLDataType::Bool, 301);
                fcv.set_constant_value_at_index(&has_bias as *const _ as *const _, MTLDataType::Bool, 302);
                fcv.set_constant_value_at_index(&has_scap as *const _ as *const _, MTLDataType::Bool, 303);
                fcv.set_constant_value_at_index(&has_kvpad as *const _ as *const _, MTLDataType::Bool, 304);
                fcv.set_constant_value_at_index(&bc_mask as *const _ as *const _, MTLDataType::Bool, 310);
                fcv.set_constant_value_at_index(&ns10 as *const _ as *const _, MTLDataType::Int, 320);
                fcv.set_constant_value_at_index(&ns20 as *const _ as *const _, MTLDataType::Int, 321);
                fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Int, 322);
            })?;
        // ds4_metal_args_flash_attn_ext — same 192-byte layout as the vec kernel.
        let mut a = [0u8; 192];
        a[0..4].copy_from_slice(&(k as i32).to_le_bytes());                // ne01 = K
        a[4..8].copy_from_slice(&(n_head as i32).to_le_bytes());            // ne02 = n_head
        a[8..12].copy_from_slice(&1i32.to_le_bytes());                      // ne03
        a[16..24].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());  // nb01
        a[24..32].copy_from_slice(&(row_bytes_f32 as u64).to_le_bytes());             // nb02
        a[32..40].copy_from_slice(&((n_head * row_bytes_f32) as u64).to_le_bytes());  // nb03
        a[40..44].copy_from_slice(&(n_total as i32).to_le_bytes());         // ne11 = KV rows
        a[44..48].copy_from_slice(&1i32.to_le_bytes());
        a[48..52].copy_from_slice(&1i32.to_le_bytes());
        a[52..56].copy_from_slice(&(head_dim as i32).to_le_bytes());        // ns10
        a[56..64].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());   // nb11
        a[64..72].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes()); // nb12
        a[72..80].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes()); // nb13
        a[80..84].copy_from_slice(&(head_dim as i32).to_le_bytes());        // ns20
        a[88..96].copy_from_slice(&(row_bytes_f16 as u64).to_le_bytes());   // nb21
        a[96..104].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes()); // nb22
        a[104..112].copy_from_slice(&((n_total as usize * row_bytes_f16) as u64).to_le_bytes()); // nb23
        a[112..116].copy_from_slice(&1i32.to_le_bytes());                  // ne31
        a[116..120].copy_from_slice(&1i32.to_le_bytes());
        a[120..124].copy_from_slice(&1i32.to_le_bytes());
        a[128..136].copy_from_slice(&mask_row.to_le_bytes());              // nb31
        a[136..144].copy_from_slice(&mask_row.to_le_bytes());
        a[144..152].copy_from_slice(&mask_row.to_le_bytes());
        a[152..156].copy_from_slice(&(n_head as i32).to_le_bytes());        // ne1
        a[156..160].copy_from_slice(&(k as i32).to_le_bytes());             // ne2
        a[160..164].copy_from_slice(&1i32.to_le_bytes());                   // ne3
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        a[164..168].copy_from_slice(&scale.to_le_bytes());                 // scale
        // shared = nqptg*(head_dim + 2*padded_v + 2*2*ncpsg) halves, padded_v=align(head_dim,64).
        let padded_v = align_up(head_dim, 64);
        let shared_elems = nqptg as usize * (head_dim + 2 * padded_v + 2 * (2 * ncpsg as usize));
        let shared_bytes = align_up(shared_elems * 2, 16) as u64;
        let out_buf = new_output_buffer::<f32>(&self.device, k * n_head * head_dim);
        let enc = shared_compute_enc(cmd_buf);
        enc.set_compute_pipeline_state(&ext_pipeline);
        enc.set_bytes(0, a.len() as u64, a.as_ptr() as *const _);
        enc.set_buffer(1, Some(q_buf), 0);
        enc.set_buffer(2, Some(kv_workspace), 0);
        enc.set_buffer(3, Some(kv_workspace), 0);
        enc.set_buffer(4, Some(&mask_buf), 0);
        enc.set_buffer(5, Some(&sinks_buf), 0);
        enc.set_buffer(6, Some(&pad_buf), 0);
        enc.set_buffer(7, Some(&blk_buf), 0);
        enc.set_buffer(8, Some(&out_buf), 0);
        enc.set_threadgroup_memory_length(0, shared_bytes);
        enc.dispatch_thread_groups(
            MTLSize::new(nblk1 as u64, n_head as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        end_shared_compute_enc(enc);
        Ok(out_buf)
    }

    /// TEST-ONLY: run [`Self::flash_attn_decode_k_metal`] standalone (build cb, commit,
    /// wait, read back). q=[K,n_head,head_dim] f32, ws=[n_total,head_dim] f16,
    /// mask=[K*n_total] f16. Returns [K,n_head,head_dim] f32.
    pub(crate) fn debug_flash_k(
        &self, params: &LayerParams, q: &[f32], ws: &[u16], mask: &[u16],
        sinks: &[f32], k: usize, n_total: u32,
    ) -> Vec<f32> {
        let nh = params.n_head as usize; let hd = params.head_dim as usize;
        let q_buf = new_input_buffer(&self.device, q);
        let ws_buf = new_input_buffer(&self.device, ws);
        let cb = self.command_queue.new_command_buffer().to_owned();
        let out = self.flash_attn_decode_k_metal(&cb, params, &q_buf, &ws_buf, n_total, mask, sinks, k)
            .expect("flash_attn_decode_k_metal");
        cb.commit(); cb.wait_until_completed();
        unsafe { read_buffer::<f32>(&out, k * nh * hd) }
    }

    /// TEST-ONLY: per-position flash over a given f16 workspace (the K=1 reference for
    /// [`Self::debug_flash_k`]). Reads ws rows [0..n_raw] with [n_raw_valid..n_raw] masked.
    pub(crate) fn debug_flash_1(
        &self, params: &LayerParams, q: &[f32], ws: &[u16], n_raw: u32, n_raw_valid: u32, sinks: &[f32],
    ) -> Vec<f32> {
        let nh = params.n_head as usize; let hd = params.head_dim as usize;
        let q_buf = new_input_buffer(&self.device, q);
        let ws_buf = new_input_buffer(&self.device, ws);
        let cb = self.command_queue.new_command_buffer().to_owned();
        let out = self.flash_attn_decode_metal_encode_with_kv_qbuf(
            &cb, params, &q_buf, &ws_buf, n_raw, sinks, Some(n_raw_valid)).expect("flash_1");
        cb.commit(); cb.wait_until_completed();
        unsafe { read_buffer::<f32>(&out, nh * hd) }
    }

    /// `dsv4_softmax_pool` — indexer attention pool. Stub pending macOS.
    #[allow(dead_code)]
    pub(crate) fn softmax_pool_impl(&self, _scores: &[f32], _kv: &[f32]) -> Result<Vec<f32>> {
        let _pipeline = self
            .pipelines
            .get("ds4_dsv4_softmax_pool")
            .ok_or_else(|| anyhow::anyhow!("dsv4_softmax_pool pipeline not loaded"))?;
        anyhow::bail!(
            "ds4_metal::softmax_pool_impl not yet wired — only used on indexer \
             layers (compress_ratio==4), deferred to M5"
        )
    }

    /// `dsv4_compressor_store_one` — compressor-cache append. Stub pending.
    #[allow(dead_code)]
    pub(crate) fn compressor_store_one_impl(
        &self,
        _kv: &[f32],
        _scores: &[f32],
        _pos_embd: &[f32],
    ) -> Result<()> {
        let _pipeline = self
            .pipelines
            .get("ds4_dsv4_compressor_store_one")
            .ok_or_else(|| anyhow::anyhow!("dsv4_compressor_store_one pipeline not loaded"))?;
        anyhow::bail!(
            "ds4_metal::compressor_store_one_impl not yet wired — only used \
             on compressed layers (compress_ratio>1), deferred to M5"
        )
    }

    /// `hc_collapse_norm` composes 3 stages (sinkhorn iter, weighted
    /// collapse, RMSNorm). Sinkhorn (n_hc ≤ 4, hc_sinkhorn_iter ≤ 3) and
    /// collapse (n_hc × d_embd) are CPU-side — small enough that a
    /// dedicated Metal kernel would not amortize encoding cost. The RMS
    /// norm leg routes through `rms_norm_impl` when gamma is Some;
    /// gamma=None falls back to the CPU path (no Metal kernel for that
    /// signature in the registry).
    /// hc_collapse_norm pivoted post-#242 to match antirez
    /// `kernel_dsv4_hc_split_weighted_sum_norm4`. Until the Metal matvec
    /// for `hc_fn @ rms_norm_plain(prev_hc)` lands, this delegates to the
    /// CPU oracle (`CpuAttentionDispatcher`). The RMS-with-gamma final
    /// leg already has Metal wiring but is unreachable from this path
    /// until the matvec leg is encoded.
    #[allow(dead_code)]
    pub(crate) fn hc_collapse_norm_impl(
        &self,
        params: &LayerParams,
        kind: ds4_engine::attn_dispatch::HcKind,
        hc_fn: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        prev_hc: &[f32],
        use_gamma: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        use ds4_engine::attn_dispatch::{AttentionDispatcher, CpuAttentionDispatcher};
        let cpu = CpuAttentionDispatcher;
        Ok(cpu.hc_collapse_norm(params, kind, hc_fn, hc_scale, hc_base, prev_hc, use_gamma))
    }

    /// M4 #330o C.3f — pack stage-1 (`n_groups` grouped matvecs against
    /// `w_o_a`) and stage-2 (one dense matvec against `w_o_b`) into
    /// ONE `MTLCommandBuffer`. Returns `(attn_low, attn_out)` — both are
    /// read back at the end of the buffer; `attn_out` is what the CPU
    /// post-amble needs, `attn_low` is returned so the DS4_DUMP_ATTN_OUT
    /// probe in `attn_output_proj_impl` can compute its `attn_low_rms`
    /// field without re-running the matvec on the host.
    ///
    /// Bit-identical to the previous `(matvec_f32_impl × n_groups)
    /// + matvec_f32_impl` sequence: same `ds4_kernel_mul_mv_f32_f32_4`
    /// pipeline, same FC constants (`nsg`, `nxpsg`), same args struct,
    /// same threadgroup memory. The only behavioural difference is the
    /// fused `commit() + wait_until_completed() + read_buffer()` round-trip
    /// — saves `n_groups` (typically 8 on DS4 V4 Flash) commit+wait+readback
    /// cycles per layer = 8 * 43 = 344 cwr per token.
    ///
    /// Preconditions (asserted by `attn_output_proj_impl` before calling):
    /// - `group_dim % 4 == 0` (float4 path on stage 1)
    /// - `n_lora_o % 2 == 0` (NR0=2 on stage 1)
    /// - `out_low_dim % 4 == 0` (float4 path on stage 2 — out_low_dim = n_groups * n_lora_o)
    /// - `d_embd % 2 == 0` (NR0=2 on stage 2)
    fn attn_output_matmuls_batched(
        &self,
        heads: &[f32],
        w_o_a: &[f32],
        w_o_b: &[f32],
        n_groups: usize,
        n_lora_o: usize,
        group_dim: usize,
        out_low_dim: usize,
        d_embd: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(
            group_dim % 4 == 0,
            "attn_output_matmuls_batched: group_dim ({}) must be divisible by 4",
            group_dim
        );
        anyhow::ensure!(
            n_lora_o % 2 == 0,
            "attn_output_matmuls_batched: n_lora_o ({}) must be divisible by NR0=2",
            n_lora_o
        );
        anyhow::ensure!(
            out_low_dim % 4 == 0,
            "attn_output_matmuls_batched: out_low_dim ({}) must be divisible by 4",
            out_low_dim
        );
        anyhow::ensure!(
            d_embd % 2 == 0,
            "attn_output_matmuls_batched: d_embd ({}) must be divisible by NR0=2",
            d_embd
        );

        // ---- Pipeline (stage 1): nsg/nxpsg keyed on group_dim ----
        let nsg_s1: i16 = (((group_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_s1: i16 = if group_dim % 256 == 0 {
            16
        } else if group_dim % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_s1 = Vec::with_capacity(4);
        key_s1.extend_from_slice(&nsg_s1.to_le_bytes());
        key_s1.extend_from_slice(&nxpsg_s1.to_le_bytes());
        let pipe_s1 = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_s1, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(
                &nsg_s1 as *const _ as *const _,
                MTLDataType::Short,
                600,
            );
            fcv.set_constant_value_at_index(
                &nxpsg_s1 as *const _ as *const _,
                MTLDataType::Short,
                601,
            );
        })?;

        // ---- Pipeline (stage 2): nsg/nxpsg keyed on out_low_dim ----
        let nsg_s2: i16 = (((out_low_dim as u64 + 127) / 128).clamp(1, 8)) as i16;
        let nxpsg_s2: i16 = if out_low_dim % 256 == 0 {
            16
        } else if out_low_dim % 128 == 0 {
            8
        } else {
            4
        };
        let mut key_s2 = Vec::with_capacity(4);
        key_s2.extend_from_slice(&nsg_s2.to_le_bytes());
        key_s2.extend_from_slice(&nxpsg_s2.to_le_bytes());
        let pipe_s2 = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key_s2, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(
                &nsg_s2 as *const _ as *const _,
                MTLDataType::Short,
                600,
            );
            fcv.set_constant_value_at_index(
                &nxpsg_s2 as *const _ as *const _,
                MTLDataType::Short,
                601,
            );
        })?;

        // ---- Buffers ----
        // w_o_a and w_o_b live as cached weight buffers (one upload per
        // unique pointer for the lifetime of MetalState). attn_low_buf is
        // an on-device transient — written by stage 1, read by stage 2,
        // read back at the end of the buffer for the dump probe.
        let heads_buf = new_input_buffer(&self.device, heads);
        let w_o_a_buf = self.cached_weight_buffer(w_o_a);
        let w_o_b_buf = self.cached_weight_buffer(w_o_b);
        let attn_low_buf = new_output_buffer::<f32>(&self.device, out_low_dim);
        let attn_out_buf = new_output_buffer::<f32>(&self.device, d_embd);

        // ---- Args (verbatim layout from matvec_f32_impl) ----
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
        let shmem_bytes: u64 = 32 * 2 * 4;

        // Stage-1 args (one per group; only the `w_o_a` row offset
        // differs across groups, encoded via buffer offset, so args
        // are identical for every group).
        let s1_args = MulMvArgs {
            ne00: group_dim as i32,
            ne01: n_lora_o as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (group_dim * 4) as u64,
            nb02: (group_dim * n_lora_o * 4) as u64,
            nb03: (group_dim * n_lora_o * 4) as u64,
            ne10: group_dim as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (group_dim * 4) as u64,
            nb12: (group_dim * 4) as u64,
            nb13: (group_dim * 4) as u64,
            ne0: n_lora_o as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let s1_n_row_tg = ((n_lora_o as u64) + 1) / 2;

        let s2_args = MulMvArgs {
            ne00: out_low_dim as i32,
            ne01: d_embd as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (out_low_dim * 4) as u64,
            nb02: (out_low_dim * d_embd * 4) as u64,
            nb03: (out_low_dim * d_embd * 4) as u64,
            ne10: out_low_dim as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (out_low_dim * 4) as u64,
            nb12: (out_low_dim * 4) as u64,
            nb13: (out_low_dim * 4) as u64,
            ne0: d_embd as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let s2_n_row_tg = ((d_embd as u64) + 1) / 2;

        // ---- ONE command buffer ----
        let cmd_buf = self.command_queue.new_command_buffer();

        // Stage 1: n_groups grouped matvec dispatches.
        for g in 0..n_groups {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&pipe_s1);
            set_scalar_bytes(enc, 0, &s1_args);
            // w_o_a row `g * n_lora_o` ... (g+1) * n_lora_o — offset in BYTES.
            let w_off = (g * n_lora_o * group_dim * 4) as u64;
            enc.set_buffer(1, Some(&w_o_a_buf), w_off);
            // heads slice `g * group_dim` ... (g+1) * group_dim — offset in BYTES.
            let h_off = (g * group_dim * 4) as u64;
            enc.set_buffer(2, Some(&heads_buf), h_off);
            // attn_low slot `g * n_lora_o` ... (g+1) * n_lora_o — offset in BYTES.
            let dst_off = (g * n_lora_o * 4) as u64;
            enc.set_buffer(3, Some(&attn_low_buf), dst_off);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(s1_n_row_tg, 1, 1),
                MTLSize::new(32, nsg_s1 as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Stage 2: one dense matvec — w_o_b · attn_low ⇒ attn_out.
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&pipe_s2);
            set_scalar_bytes(enc, 0, &s2_args);
            enc.set_buffer(1, Some(&w_o_b_buf), 0);
            enc.set_buffer(2, Some(&attn_low_buf), 0);
            enc.set_buffer(3, Some(&attn_out_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(s2_n_row_tg, 1, 1),
                MTLSize::new(32, nsg_s2 as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "attn_output_matmuls_batched");

        let attn_low = unsafe { read_buffer::<f32>(&attn_low_buf, out_low_dim) };
        let attn_out = unsafe { read_buffer::<f32>(&attn_out_buf, d_embd) };
        Ok((attn_low, attn_out))
    }

    /// `attn_output_proj` — grouped output projection (DS4 N_OUT_GROUP=8) +
    /// dense down-projection + HC expand-add. heads[q_dim] is reshaped to
    /// [n_groups, group_dim]; w_o_a [out_low_dim=n_groups*n_lora_o, group_dim]
    /// row-major; group g consumes rows [g*n_lora_o .. (g+1)*n_lora_o].
    /// Stage 1 (per-group): attn_low[g*n_lora_o + l] = Σ_d w_o_a[(g*n_lora_o+l)*group_dim + d] * heads[g*group_dim + d]
    /// Stage 2 (dense):     attn_out = w_o_b · attn_low  → [d_embd]
    /// Stage 3:             after[h*d_embd + e] = cur_hc[..] + hc_split_post[h] * attn_out[e]
    ///
    /// M4 #330o C.3f: stages 1 + 2 packed into one MTLCommandBuffer.
    #[allow(dead_code)]
    pub(crate) fn attn_output_proj_impl(
        &self,
        params: &LayerParams,
        heads: &[f32],
        w_o_a: &[f32],
        w_o_b: &[f32],
        cur_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Result<Vec<f32>> {
        let d_embd = params.d_embd as usize;
        let n_hc = params.n_hc as usize;
        let q_dim = heads.len();
        let n_groups = params.n_out_group as usize;
        anyhow::ensure!(
            n_groups > 0 && q_dim % n_groups == 0,
            "attn_output_proj: q_dim ({}) must be divisible by n_out_group ({})",
            q_dim,
            n_groups
        );
        let group_dim = q_dim / n_groups;
        anyhow::ensure!(
            !w_o_a.is_empty() && w_o_a.len() % (n_groups * group_dim) == 0,
            "attn_output_proj: w_o_a.len ({}) must be a positive multiple of n_groups*group_dim ({}*{})",
            w_o_a.len(),
            n_groups,
            group_dim
        );
        let n_lora_o = w_o_a.len() / (n_groups * group_dim);
        let out_low_dim = n_groups * n_lora_o;
        anyhow::ensure!(
            w_o_b.len() == d_embd * out_low_dim,
            "attn_output_proj: w_o_b.len ({}) != d_embd * out_low_dim ({} * {})",
            w_o_b.len(),
            d_embd,
            out_low_dim
        );
        anyhow::ensure!(
            hc_split_post.len() == n_hc,
            "attn_output_proj: hc_split_post.len ({}) != n_hc ({})",
            hc_split_post.len(),
            n_hc
        );
        anyhow::ensure!(
            hc_split_comb.len() == n_hc * n_hc,
            "attn_output_proj: hc_split_comb.len ({}) != n_hc² ({})",
            hc_split_comb.len(),
            n_hc * n_hc
        );
        anyhow::ensure!(
            cur_hc.len() == n_hc * d_embd,
            "attn_output_proj: cur_hc.len ({}) != n_hc * d_embd ({} * {})",
            cur_hc.len(),
            n_hc,
            d_embd
        );

        // M4 #330o C.3f — pack stage-1 (n_groups grouped matvecs) and
        // stage-2 (one dense matvec) into ONE MTLCommandBuffer. Saves
        // n_groups commit+wait+readback per layer (8 cwr/layer for
        // n_out_group=8 = 344 cwr/token on DS4 V4 Flash). Bit-identical
        // to the prior `(self.matvec_f32_impl × n_groups) + self.matvec_f32_impl`
        // sequence — same kernel + same FC constants. Internal refactor
        // only; no trait/CPU change.
        let (attn_low, attn_out) = self.attn_output_matmuls_batched(
            heads,
            w_o_a,
            w_o_b,
            n_groups,
            n_lora_o,
            group_dim,
            out_low_dim,
            d_embd,
        )?;

        // Verifier-bisection output-proj tap: capture attn_out (W_o projection,
        // pre-hc_expand) so a test can split output_proj vs hc_expand_attn.
        ds4_engine::attn_dispatch::ATTN_OUT_CAPTURE.with(|c| {
            let mut cc = c.borrow_mut();
            if cc.0 != usize::MAX
                && cc.0 == ds4_engine::attn_dispatch::CURRENT_LAYER_HINT.with(|h| h.get())
            {
                cc.1 = attn_out.clone();
            }
        });

        // M4 #287: env-gated dump for `attn_out` (post-stage-2, pre-HC-expand)
        // mirrored from the CPU oracle so the Metal path emits the same probe.
        // DS4_DUMP_ATTN_OUT=POS fires only at incoming state.pos == POS. We do
        // NOT have the `CURRENT_POS_HINT` thread-local in ds4_metal (it lives
        // in ds4_engine::attn_dispatch), so this probe reads its own env var
        // and fires for every layer at every position — caller filters via
        // grep on the `il=` field.
        {
            if std::env::var("DS4_DUMP_ATTN_OUT").ok().is_some() {
                let ss: f64 = attn_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let rms = (ss / attn_out.len() as f64).sqrt();
                let mut ranked: Vec<(usize, f32)> = attn_out.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top3: Vec<(usize, f32)> = ranked.iter().take(3).copied().collect();
                let post: Vec<f32> = hc_split_post.to_vec();
                let attn_low_rms = {
                    let ssl: f64 = attn_low.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    (ssl / attn_low.len() as f64).sqrt()
                };
                // M4 #287 follow-up: also dump comb row sums per dst slot
                // (column sums of comb matrix grouped by src), to localize
                // whether slot-3 divergence comes from the post leg or comb leg.
                let mut comb_row_sums = [0.0f32; 4];
                for dst in 0..n_hc.min(4) {
                    let mut s = 0.0f32;
                    for src in 0..n_hc {
                        s += hc_split_comb[dst + src * n_hc];
                    }
                    comb_row_sums[dst] = s;
                }
                eprintln!(
                    "ATTN_OUT il={} attn_low_rms={attn_low_rms:.4} attn_out_rms={rms:.4} attn_out_top3={top3:?} hc_split_post={post:?} comb_row_sums={comb_row_sums:?}",
                    params.layer_idx,
                );
            }
        }

        // Antirez `hc_post_one` (ds4.c:4217-4237):
        //   after[dst, e] = attn_out[e] * hc_split_post[dst]
        //                 + Σ_src hc_split_comb[dst+src*n_hc] * cur_hc[src, e]
        let mut after = vec![0.0f32; n_hc * d_embd];
        for dst in 0..n_hc {
            let base = dst * d_embd;
            let w_post = hc_split_post[dst];
            for e in 0..d_embd {
                let mut acc = w_post * attn_out[e];
                for src in 0..n_hc {
                    acc += hc_split_comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
                }
                after[base + e] = acc;
            }
        }
        Ok(after)
    }

    /// `shared_expert` — gate + up matvec + SwiGLU·clamp.
    /// Two `matvec_f32` calls (w_gate, w_up) then per-element
    /// `silu(clamp(g, ±c)) · clamp(u, ±c)`. The SwiGLU stitch is CPU-side
    /// (shared_dim is small enough — typically ≤ 2048 — that a dedicated
    /// Metal kernel would not win over the matvec cost).
    #[allow(dead_code)]
    pub(crate) fn shared_expert_impl(
        &self,
        params: &LayerParams,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        shared_dim: u32,
        _clamp: f32,
    ) -> Result<Vec<f32>> {
        let d_embd = params.d_embd as usize;
        let sd = shared_dim as usize;
        anyhow::ensure!(
            ffn_norm.len() == d_embd,
            "shared_expert: ffn_norm.len ({}) != d_embd ({})",
            ffn_norm.len(),
            d_embd
        );
        anyhow::ensure!(
            w_gate.len() == sd * d_embd,
            "shared_expert: w_gate.len ({}) != shared_dim * d_embd ({} * {})",
            w_gate.len(),
            sd,
            d_embd
        );
        anyhow::ensure!(
            w_up.len() == sd * d_embd,
            "shared_expert: w_up.len ({}) != shared_dim * d_embd ({} * {})",
            w_up.len(),
            sd,
            d_embd
        );

        // M4 #330o C.3g — pack gate + up matvecs into ONE MTLCommandBuffer.
        // Same kernel + same FC constants for both matvecs (`ffn_norm` is
        // the d_in vector, `sd` is the d_out — identical specialization).
        // Bit-identical to `(self.matvec_f32_impl × 2)`. Active only on
        // the `DS4_SILU_FIDELITY=1` path; default-OFF uses `shared_chain_
        // batched` (C.3b) which already bundles gate+up+silu+down. Saves
        // 2 cwr/layer × 43 = 86 cwr/token on the fid=1 path.
        let (g, u) = self.shared_expert_gate_up_batched(w_gate, w_up, ffn_norm, sd)?;

        Ok(shared_expert_swiglu(&g, &u))
    }

    /// M4 #330o C.3g helper: encode two matvecs (`w_gate · ffn_norm`,
    /// `w_up · ffn_norm`) into ONE `MTLCommandBuffer`. Both share the
    /// same input vector (`ffn_norm`) and the same output dim (`sd`),
    /// so a single specialization of `ds4_kernel_mul_mv_f32_f32_4`
    /// works for both. Returns `(g, u)` — two readbacks from one
    /// commit+wait.
    ///
    /// Bit-identical to running `matvec_f32_impl(w_gate, ffn_norm, sd)`
    /// followed by `matvec_f32_impl(w_up, ffn_norm, sd)`: same args,
    /// same FC constants, same threadgroup memory. Only the cwr cost
    /// changes — 1 instead of 2.
    fn shared_expert_gate_up_batched(
        &self,
        w_gate: &[f32],
        w_up: &[f32],
        ffn_norm: &[f32],
        sd: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d_in = ffn_norm.len();
        anyhow::ensure!(
            d_in % 4 == 0,
            "shared_expert_gate_up_batched: d_in ({}) must be divisible by 4",
            d_in
        );
        anyhow::ensure!(
            sd % 2 == 0,
            "shared_expert_gate_up_batched: sd ({}) must be divisible by NR0=2",
            sd
        );

        // FC constants (verbatim from matvec_f32_impl) — both matvecs
        // share them since they share d_in.
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
        let pipeline = self.specialized_pipeline("ds4_kernel_mul_mv_f32_f32_4", &key, |fcv| {
            use metal::MTLDataType;
            fcv.set_constant_value_at_index(&nsg as *const _ as *const _, MTLDataType::Short, 600);
            fcv.set_constant_value_at_index(&nxpsg as *const _ as *const _, MTLDataType::Short, 601);
        })?;

        let w_gate_buf = self.cached_weight_buffer(w_gate);
        let w_up_buf = self.cached_weight_buffer(w_up);
        let x_buf = new_input_buffer(&self.device, ffn_norm);
        let g_buf = new_output_buffer::<f32>(&self.device, sd);
        let u_buf = new_output_buffer::<f32>(&self.device, sd);

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
            ne01: sd as i32,
            ne02: 1,
            _pad0: 0,
            nb00: 4,
            nb01: (d_in * 4) as u64,
            nb02: (d_in * sd * 4) as u64,
            nb03: (d_in * sd * 4) as u64,
            ne10: d_in as i32,
            ne11: 1,
            ne12: 1,
            _pad1: 0,
            nb10: 4,
            nb11: (d_in * 4) as u64,
            nb12: (d_in * 4) as u64,
            nb13: (d_in * 4) as u64,
            ne0: sd as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };
        let shmem_bytes: u64 = 32 * 2 * 4;
        let n_row_tg = ((sd as u64) + 1) / 2;

        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 1: gate matvec.
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&pipeline);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&w_gate_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&g_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        // Pass 2: up matvec.
        {
            let enc = shared_compute_enc(cmd_buf);
            enc.set_compute_pipeline_state(&pipeline);
            set_scalar_bytes(enc, 0, &args);
            enc.set_buffer(1, Some(&w_up_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&u_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.dispatch_thread_groups(
                MTLSize::new(n_row_tg, 1, 1),
                MTLSize::new(32, nsg as u64, 1),
            );
            end_shared_compute_enc(enc);
        }

        self.commit_wait_traced(cmd_buf, "shared_expert_gate_up_batched");

        let g = unsafe { read_buffer::<f32>(&g_buf, sd) };
        let u = unsafe { read_buffer::<f32>(&u_buf, sd) };
        Ok((g, u))
    }

    /// `shared_down_hc_expand_add` — down-projection + HC expand + add.
    /// Wire: `shared_out = w_down · shared_mid` via `matvec_f32_impl`,
    /// then `after[h*d_embd+e] = after_attn_hc[..] + hc_split[h] *
    /// (shared_out[e] + routed_out[e])`. HC expand-add is CPU-side
    /// (n_hc ≤ 4 in production, fully bandwidth-bound, not worth a
    /// dedicated kernel encode).
    #[allow(dead_code)]
    pub(crate) fn shared_down_hc_expand_add_impl(
        &self,
        params: &LayerParams,
        shared_mid: &[f32],
        w_down: &[f32],
        routed_out: &[f32],
        after_attn_hc: &[f32],
        hc_split: &[f32],
        hc_split_comb: &[f32],
    ) -> Result<Vec<f32>> {
        let d_embd = params.d_embd as usize;
        let n_hc = params.n_hc as usize;
        let sd = shared_mid.len();
        anyhow::ensure!(
            w_down.len() == d_embd * sd,
            "shared_down: w_down.len ({}) != d_embd * sd ({} * {})",
            w_down.len(),
            d_embd,
            sd
        );
        anyhow::ensure!(
            routed_out.len() == d_embd,
            "shared_down: routed_out.len ({}) != d_embd ({})",
            routed_out.len(),
            d_embd
        );
        anyhow::ensure!(
            hc_split.len() == n_hc,
            "shared_down: hc_split.len ({}) != n_hc ({})",
            hc_split.len(),
            n_hc
        );
        anyhow::ensure!(
            hc_split_comb.len() == n_hc * n_hc,
            "shared_down: hc_split_comb.len ({}) != n_hc² ({})",
            hc_split_comb.len(),
            n_hc * n_hc
        );
        anyhow::ensure!(
            after_attn_hc.len() == n_hc * d_embd,
            "shared_down: after_attn_hc.len ({}) != n_hc * d_embd ({} * {})",
            after_attn_hc.len(),
            n_hc,
            d_embd
        );

        let shared_out = self.matvec_f32_impl(w_down, shared_mid, d_embd)?;

        // M4 #287 follow-up: probe FFN-half `hc_post_one`. Dumps ffn_out
        // (== shared_out + routed_out) RMS + top3 + hc_split (post-ffn) +
        // comb row sums. This is the second `hc_post_one` per layer, so
        // any slot-3 magnitude amplification at L0 must come from here.
        if std::env::var("DS4_DUMP_FFN_HC").ok().is_some() {
            let mut ffn_out = vec![0.0f32; d_embd];
            for e in 0..d_embd {
                ffn_out[e] = shared_out[e] + routed_out[e];
            }
            let ss: f64 = ffn_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rms = (ss / ffn_out.len() as f64).sqrt();
            let mut ranked: Vec<(usize, f32)> = ffn_out.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top3: Vec<(usize, f32)> = ranked.iter().take(3).copied().collect();
            let post: Vec<f32> = hc_split.to_vec();
            let mut comb_row_sums = [0.0f32; 4];
            for dst in 0..n_hc.min(4) {
                let mut s = 0.0f32;
                for src in 0..n_hc {
                    s += hc_split_comb[dst + src * n_hc];
                }
                comb_row_sums[dst] = s;
            }
            let shared_rms = {
                let ss: f64 = shared_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
                (ss / shared_out.len() as f64).sqrt()
            };
            let routed_rms = {
                let ss: f64 = routed_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
                (ss / routed_out.len() as f64).sqrt()
            };
            eprintln!(
                "FFN_HC il={} shared_rms={shared_rms:.4} routed_rms={routed_rms:.4} ffn_out_rms={rms:.4} ffn_out_top3={top3:?} hc_split_post_ffn={post:?} comb_row_sums={comb_row_sums:?}",
                params.layer_idx,
            );
        }

        // Antirez `hc_post_one` (ds4.c:4225-4236):
        //   ffn_out[e]    = shared_out[e] + routed_out[e]
        //   after[dst, e] = ffn_out[e] * hc_split[dst]
        //                 + Σ_src hc_split_comb[dst+src*n_hc] * after_attn_hc[src, e]
        let mut after = vec![0.0f32; n_hc * d_embd];
        for dst in 0..n_hc {
            let base = dst * d_embd;
            let w_post = hc_split[dst];
            for e in 0..d_embd {
                let mut acc = w_post * (shared_out[e] + routed_out[e]);
                for src in 0..n_hc {
                    acc += hc_split_comb[dst + src * n_hc] * after_attn_hc[src * d_embd + e];
                }
                after[base + e] = acc;
            }
        }
        Ok(after)
    }
}

/// M4 #319 — `shared_expert` SwiGLU stitch.
///
/// Pulled out of `MetalState::shared_expert_impl` so it is testable without
/// constructing a `MetalState` (which requires a live Metal device).
///
/// Antirez `swiglu` (ds4.c:4873-4877) is `silu(g) * u` with no clamp. Antirez
/// `silu` (ds4.c:4863) is `x * sigmoid_stable(x)`, where `sigmoid_stable`
/// (ds4.c:4736) sign-branches: `x>=0 → 1/(1+exp(-x))`, `x<0 → exp(x)/(1+exp(x))`.
///
/// Previous inline form here used the positive-branch-only identity
/// `g[i] / (1 + (-g[i]).exp())` which is algebraically equal but f32-ULP
/// different for `g[i] < 0` — the same divergence class as M4 #311 in
/// `ds4_engine::forward::silu`, but bypassing the existing
/// `DS4_SILU_FIDELITY=1` gate because the call was inlined.
///
/// Delegating to `ds4_engine::forward::silu` lets a single env toggle cover
/// both the engine-side silu (M4 #311) and the Metal-side shared_expert
/// silu (this site, M4 #319).
pub(crate) fn shared_expert_swiglu(g: &[f32], u: &[f32]) -> Vec<f32> {
    debug_assert_eq!(g.len(), u.len());
    let sd = g.len();
    let mut out = vec![0.0f32; sd];
    for i in 0..sd {
        out[i] = ds4_engine::forward::silu(g[i]) * u[i];
    }
    out
}

/// M4 #318 — CPU fallback body of `MetalState::flash_attn_decode_impl`.
///
/// Pulled out as a free function so it is testable without constructing a
/// `MetalState` (which requires a live Metal device). Mirrors the inline
/// implementation that was here previously, but gates two divergences
/// from antirez `layer_attention_rows_one` (ds4.c:4748-4786) behind
/// `DS4_MATVEC_F32_FIDELITY=1`:
///
/// 1. **Q·K dot reduction tree** — antirez uses `dot_f32` (2×4-lane f32
///    FMA pair-reduce). Default path is `Iterator::sum::<f32>()`
///    (left-fold). Same divergence class as M4 #312.
/// 2. **Output normalize order** — antirez accumulates raw `weight*kv`
///    via `axpy_f32`, then does ONE `scale_f32(oh, 1/denom, ...)` at end.
///    Default path pre-divides each weight by sum before V accumulation.
///    Same divergence class as M4 #305.
///
/// When `DS4_MATVEC_F32_FIDELITY=1`:
///   - dots flow through `ds4_engine::forward::dot_f32_antirez`;
///   - V accumulation uses raw weights and a single post-divide.
///
/// This path is dormant under production DS4 preconditions
/// (head_dim=512 → `flash_attn_decode_metal` takes the fast path
/// upstream), but exercised whenever those preconditions fail (e.g.
/// compressed-KV heads with `kv_comp.is_some()`, smaller test rigs, or
/// future model variants).
pub(crate) fn flash_attn_decode_cpu_fallback(
    n_head: usize,
    head_dim: usize,
    row: usize,
    q: &[f32],
    kv_raw: &[f32],
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    kv_comp: Option<&[f32]>,
    n_comp: u32,
    comp_selected: Option<&[u32]>,
    n_selected: u32,
    attn_sinks: &[f32],
) -> Result<AttnHeadsOut> {
    let antirez_dot_fidelity =
        std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");

    let mut out = vec![0.0f32; n_head * head_dim];
    let scale = (head_dim as f32).sqrt().recip();

    enum Src {
        Raw(u32),
        Comp(u32),
    }
    let mut idx: Vec<Src> = (0..n_raw)
        .map(|i| Src::Raw((raw_start + i) % raw_cap))
        .collect();
    if let (Some(sel), Some(_)) = (comp_selected, kv_comp) {
        for i in 0..n_selected as usize {
            idx.push(Src::Comp(sel[i]));
        }
    } else if kv_comp.is_some() {
        for i in 0..n_comp {
            idx.push(Src::Comp(i));
        }
    }

    let scores_buf_len = idx.len() + 1;
    for h in 0..n_head {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let k_dim_raw = head_dim.min(row);
        let mut scores = Vec::with_capacity(scores_buf_len);
        scores.push(attn_sinks[h]);
        for s_src in &idx {
            let (src, k_dim): (&[f32], usize) = match s_src {
                Src::Raw(ki) => {
                    let k_off = (*ki as usize) * row;
                    (&kv_raw[k_off..k_off + row], k_dim_raw)
                }
                Src::Comp(ci) => {
                    let comp = kv_comp.expect("Src::Comp requires kv_comp");
                    let c_off = (*ci as usize) * head_dim;
                    (&comp[c_off..c_off + head_dim], head_dim)
                }
            };
            let dot: f32 = if antirez_dot_fidelity {
                ds4_engine::forward::dot_f32_antirez(q_h, &src[..k_dim])
            } else {
                q_h.iter()
                    .zip(src.iter().take(k_dim))
                    .map(|(a, b)| a * b)
                    .sum()
            };
            scores.push(dot * scale);
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        if antirez_dot_fidelity {
            // Antirez `layer_attention_rows_one` (ds4.c:4774-4782):
            // accumulate raw weight*kv via axpy_f32, then ONE scale_f32 at end.
            for (j, s_src) in idx.iter().enumerate() {
                let w = scores[j + 1];
                let (src, v_dim): (&[f32], usize) = match s_src {
                    Src::Raw(ki) => {
                        let k_off = (*ki as usize) * row;
                        (&kv_raw[k_off..k_off + row], head_dim.min(row))
                    }
                    Src::Comp(ci) => {
                        let comp = kv_comp.expect("Src::Comp requires kv_comp");
                        let c_off = (*ci as usize) * head_dim;
                        (&comp[c_off..c_off + head_dim], head_dim)
                    }
                };
                for d in 0..v_dim {
                    out_h[d] += w * src[d];
                }
            }
            let inv = sum.recip();
            for v in out_h.iter_mut() {
                *v *= inv;
            }
        } else {
            let inv = sum.recip();
            for (j, s_src) in idx.iter().enumerate() {
                let w = scores[j + 1] * inv;
                let (src, v_dim): (&[f32], usize) = match s_src {
                    Src::Raw(ki) => {
                        let k_off = (*ki as usize) * row;
                        (&kv_raw[k_off..k_off + row], head_dim.min(row))
                    }
                    Src::Comp(ci) => {
                        let comp = kv_comp.expect("Src::Comp requires kv_comp");
                        let c_off = (*ci as usize) * head_dim;
                        (&comp[c_off..c_off + head_dim], head_dim)
                    }
                };
                for d in 0..v_dim {
                    out_h[d] += w * src[d];
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod flash_attn_fallback_tests {
    //! M4 #318 — failing-first discrimination test for the CPU fallback of
    //! `MetalState::flash_attn_decode_impl`.
    //!
    //! Same env-mutation risk as the ds4_engine fidelity tests, so we
    //! serialize on a local `Mutex` to avoid races with other tests in
    //! this binary that might toggle the same env.
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn left_fold_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn fallback_dot_and_normalize_antirez_fidelity_gate_switches() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

        // n_head=1, head_dim=256 → 32 iterations through the 8-wide FMA
        // partial loop. n_raw=2 with V-row magnitudes engineered so
        // softmax weights stay non-trivial (gap stays inside softmax's
        // linear region), so reduction-tree ULP delta propagates into
        // `out`.
        let n_head: usize = 1;
        let head_dim: usize = 256;
        let row: usize = head_dim;
        let raw_cap: u32 = 4;
        let n_raw: u32 = 2;
        let raw_start: u32 = 0;

        // q: alternating ±-magnitude (1.0 / -1e-3) — amplifies dot ULP delta.
        let q: Vec<f32> = (0..head_dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1e-3 })
            .collect();

        // kv_raw: row 0 is 0.5·q + tiny noise; row 1 is 0.45·q + tiny noise.
        // Dots ≈ 0.5·||q||² and 0.45·||q||². With head_dim=256 the squared
        // norm is ≈128, post-scale scores ≈ 4.0 and 3.6 — softmax weights
        // both ≈ 0.4-0.6, NOT saturating, so the ULP delta survives.
        let mut kv_raw = vec![0.0f32; (raw_cap as usize) * row];
        for i in 0..head_dim {
            kv_raw[i] = q[i] * 0.5 + ((i as f32) * 1e-4).sin() * 0.01;
            kv_raw[row + i] = q[i] * 0.45 + ((i as f32) * 2e-4).cos() * 0.01;
        }

        // attn_sinks: small so the sink term doesn't dominate softmax.
        let attn_sinks = vec![0.1f32];

        // Pre-flight: confirm the left-fold dot and antirez dot disagree
        // bit-wise on at least one row of this setup. If they don't, the
        // test is trivial and would pass with either implementation.
        let left = left_fold_dot(&q, &kv_raw[..head_dim]);
        let anti = ds4_engine::forward::dot_f32_antirez(&q, &kv_raw[..head_dim]);
        assert_ne!(
            left.to_bits(),
            anti.to_bits(),
            "test setup is trivial: left-fold and antirez dot bit-match"
        );

        // Gate OFF — capture default-path output.
        std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
        let out_off = flash_attn_decode_cpu_fallback(
            n_head, head_dim, row,
            &q, &kv_raw, n_raw, raw_cap, raw_start,
            None, 0, None, 0,
            &attn_sinks,
        ).expect("fallback off");

        // Gate ON — capture antirez-fidelity output.
        std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
        let out_on = flash_attn_decode_cpu_fallback(
            n_head, head_dim, row,
            &q, &kv_raw, n_raw, raw_cap, raw_start,
            None, 0, None, 0,
            &attn_sinks,
        ).expect("fallback on");
        std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

        // Discriminator: at least one output element must differ bit-wise
        // between gate-off and gate-on. The combined effect of M4 #312
        // (dot reduction tree) and M4 #305 (late-normalize) is small but
        // non-zero on adversarial input; one bit of f32-ULP difference is
        // enough.
        let any_diff = out_off
            .iter()
            .zip(out_on.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "DS4_MATVEC_F32_FIDELITY=1 did NOT change flash_attn_decode_cpu_fallback output — gate is a no-op"
        );
    }
}

#[cfg(test)]
mod shared_expert_swiglu_tests {
    //! M4 #319 — failing-first discrimination test for the SwiGLU stitch of
    //! `MetalState::shared_expert_impl`.
    //!
    //! Uses the same env-mutation hygiene as the M4 #318 test: local
    //! `Mutex` serializes against any other test in this binary that might
    //! toggle `DS4_SILU_FIDELITY`.
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn positive_branch_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn sigmoid_stable_antirez(x: f32) -> f32 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    #[test]
    fn shared_expert_swiglu_antirez_fidelity_gate_switches() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("DS4_SILU_FIDELITY");

        // Strongly negative gate input is where the two silu forms diverge:
        // positive-branch computes `exp(7) ≈ 1097` as an intermediate, antirez
        // computes `exp(-7) ≈ 9.12e-4` — the rounding cascade differs by ULPs.
        let g = vec![-7.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0];
        let u = vec![1.0f32; g.len()];

        // Pre-flight: confirm the two silu forms bit-differ on at least one
        // entry. If they don't, the test is trivial.
        let mut probe_diff = false;
        for &x in &g {
            if positive_branch_silu(x).to_bits() != (x * sigmoid_stable_antirez(x)).to_bits() {
                probe_diff = true;
                break;
            }
        }
        assert!(
            probe_diff,
            "test setup is trivial: positive-branch and antirez silu bit-match on all probes"
        );

        // Gate OFF — captures default-path output (positive-branch silu inside
        // ds4_engine::forward::silu).
        std::env::remove_var("DS4_SILU_FIDELITY");
        let out_off = shared_expert_swiglu(&g, &u);

        // Gate ON — antirez sigmoid_stable sign-branched form.
        std::env::set_var("DS4_SILU_FIDELITY", "1");
        let out_on = shared_expert_swiglu(&g, &u);
        std::env::remove_var("DS4_SILU_FIDELITY");

        let any_diff = out_off
            .iter()
            .zip(out_on.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "DS4_SILU_FIDELITY=1 did NOT change shared_expert_swiglu output — gate is a no-op"
        );
    }
}

// ── Encoding helpers ───────────────────────────────────────────────────
//
// These reduce the per-method boilerplate to:
//   1. allocate input buffers
//   2. allocate output buffer
//   3. encode + dispatch
//   4. read back

/// M5 Phase D — shared helper for per-layer compressor/indexer state
/// pools. Mirrors `kv_buffer_or_alloc`'s contract: zero-init on first
/// access (Metal returns zero-filled pages for `new_buffer` on Apple
/// Silicon Shared memory), debug-assert that the requested size matches
/// the cached size, return an Arc-clone of the buffer.
fn state_buffer_or_alloc(
    device: &Device,
    pool: &std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    name: &'static str,
    layer_idx: u32,
    byte_size: usize,
) -> metal::Buffer {
    state_buffer_or_alloc_pin(device, pool, name, layer_idx, byte_size, |_| {})
}

fn state_buffer_or_alloc_pin(
    device: &Device,
    pool: &std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    name: &'static str,
    layer_idx: u32,
    byte_size: usize,
    on_new: impl FnOnce(&metal::Buffer),
) -> metal::Buffer {
    let mut map = pool.lock().unwrap_or_else(|e| {
        panic!("{name} mutex poisoned: {e}");
    });
    let mut created = false;
    let entry = map.entry(layer_idx).or_insert_with(|| {
        created = true;
        device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared)
    });
    if created {
        on_new(entry);
    }
    debug_assert!(
        entry.length() == byte_size as u64,
        "{name}({layer_idx}): allocated len {} != requested {}",
        entry.length(),
        byte_size,
    );
    entry.clone()
}

/// M5 Phase D — variant of `state_buffer_or_alloc` that fills the
/// buffer with `init_value: f32` on first allocation. Used by the
/// compressor/indexer `state_score` pools: the CPU mirror seeds
/// `state_score` with `DS4_NEG_INF` (`decode_step.rs:493`,
/// `decode_step.rs:504`) so that not-yet-written rows contribute ~0
/// to the pooled softmax. Without the matching GPU init, at pos=3
/// (first emit) the prev-window rows would hold `0.0` rather than
/// `-1e9`, weighting masked slots by `exp(0 - max_score)` instead of
/// `exp(-1e9 - max_score) ≈ 0`.
///
/// The fill runs once per `(pool, layer_idx)` — subsequent accesses
/// see the persisted contents of prior dispatches.
fn state_buffer_or_alloc_filled_f32_pin(
    device: &Device,
    pool: &std::sync::Mutex<HashMap<u32, metal::Buffer>>,
    name: &'static str,
    layer_idx: u32,
    byte_size: usize,
    init_value: f32,
    on_new: impl FnOnce(&metal::Buffer),
) -> metal::Buffer {
    let mut map = pool.lock().unwrap_or_else(|e| {
        panic!("{name} mutex poisoned: {e}");
    });
    let mut created = false;
    let entry = map.entry(layer_idx).or_insert_with(|| {
        created = true;
        let buf =
            device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared);
        // Shared-storage CPU fill. byte_size must be a multiple of f32.
        debug_assert_eq!(byte_size % std::mem::size_of::<f32>(), 0);
        let n = byte_size / std::mem::size_of::<f32>();
        unsafe {
            let p = buf.contents() as *mut f32;
            for i in 0..n {
                *p.add(i) = init_value;
            }
        }
        buf
    });
    if created {
        on_new(entry);
    }
    debug_assert!(
        entry.length() == byte_size as u64,
        "{name}({layer_idx}): allocated len {} != requested {}",
        entry.length(),
        byte_size,
    );
    entry.clone()
}

/// Allocate a shared-storage buffer initialised from a slice of POD values.
#[allow(dead_code)]
pub(crate) fn new_input_buffer<T: Copy>(device: &Device, data: &[T]) -> metal::Buffer {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len == 0 {
        // Metal rejects zero-length buffers (debug layer asserts; release
        // returns nil → downstream null deref). A zero-length upload is a
        // caller bug (e.g. binding a lean-dropped f32 weight) — make it loud.
        eprintln!(
            "ds4_metal: new_input_buffer called with ZERO length — caller bug; backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
        // 16-byte dummy keeps Metal happy long enough to surface the trace.
        return device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Allocate a shared-storage buffer of the given element count, zero-init.
#[allow(dead_code)]
pub(crate) fn new_output_buffer<T>(device: &Device, n_elements: usize) -> metal::Buffer {
    let byte_len = (n_elements * std::mem::size_of::<T>()) as u64;
    device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
}

/// Whether the persistent chain reuses per-layer flash scratch buffers across
/// tokens (default ON). `DS4_FLASH_SCRATCH_REUSE=0` reverts to fresh allocation
/// every flash call — the A/B knob and a safety escape hatch.
pub(crate) fn flash_scratch_reuse_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DS4_FLASH_SCRATCH_REUSE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Read back `n_elements` of type `T` from a shared-storage buffer.
///
/// Safety: caller must ensure `buf` has `n_elements * size_of::<T>()` bytes
/// of valid contents and that the GPU has finished writing.
#[allow(dead_code)]
pub(crate) unsafe fn read_buffer<T: Copy + Default>(buf: &metal::Buffer, n_elements: usize) -> Vec<T> {
    let mut out = vec![T::default(); n_elements];
    let src = buf.contents() as *const T;
    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_elements);
    out
}

/// Encode a single POD value as inline bytes at the given buffer index.
#[allow(dead_code)]
pub(crate) fn set_scalar_bytes<T: Copy>(enc: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        (value as *const T) as *const _,
    );
}

#[cfg(test)]
mod specialized_pipeline_tests {
    use super::*;
    use metal::foreign_types::ForeignType;
    use metal::MTLDataType;

    /// FC offsets — see `metal/unary.metal:1`: `#define FC_UNARY 1200`.
    const FC_UNARY_OP: u64 = 1200;
    const FC_UNARY_CNT: u64 = 1201;

    const FC_MUL_MV_NSG: u64 = 600;
    const FC_MUL_MV_NXPSG: u64 = 601;
    const FC_MOE_WRITE_PRE_SWIGLU: u64 = 602;
    const FC_FLASH_VEC_HAS_MASK: u64 = 400;
    const FC_FLASH_VEC_HAS_SINKS: u64 = 401;
    const FC_FLASH_VEC_HAS_BIAS: u64 = 402;
    const FC_FLASH_VEC_HAS_SCAP: u64 = 403;
    const FC_FLASH_VEC_HAS_KVPAD: u64 = 404;
    const FC_FLASH_VEC_NS10: u64 = 420;
    const FC_FLASH_VEC_NS20: u64 = 421;
    const FC_FLASH_VEC_NSG: u64 = 422;
    const FC_FLASH_VEC_NWG: u64 = 423;

    /// Smoke-tests `MetalState::specialized_pipeline`:
    /// 1. builds two pipelines for `kernel_unary_f32_f32` with distinct
    ///    constants (op=0, op=27) and verifies the cache hands back the
    ///    same handle on the second call with identical constants_key.
    /// 2. verifies that two different constants_keys produce distinct
    ///    cache entries (no aliasing on symbol alone).
    ///
    /// We don't dispatch — just check the helper's lookup + caching
    /// contract. Constant values must be encoded (every constant the
    /// kernel reads), so we set both FC_unary_op and FC_unary_cnt even
    /// when picking op=0.
    #[test]
    fn cache_hit_and_distinct_keys() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };

        let sym = "kernel_unary_f32_f32";

        let populate = |op: i16| {
            move |v: &metal::FunctionConstantValuesRef| {
                let op_v = op;
                let cnt_v: bool = false;
                v.set_constant_value_at_index(
                    &op_v as *const _ as *const _,
                    MTLDataType::Short,
                    FC_UNARY_OP,
                );
                v.set_constant_value_at_index(
                    &cnt_v as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_UNARY_CNT,
                );
            }
        };

        let key_op0 = b"op=0,cnt=0".to_vec();
        let key_op27 = b"op=27,cnt=0".to_vec();

        let p0_a = match state.specialized_pipeline(sym, &key_op0, populate(0)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skip: symbol {} not in libs: {}", sym, e);
                return;
            }
        };
        let p0_b = state
            .specialized_pipeline(sym, &key_op0, populate(0))
            .expect("second lookup with same key must succeed");
        let p27 = state
            .specialized_pipeline(sym, &key_op27, populate(27))
            .expect("distinct-key lookup must succeed");

        assert!(
            std::ptr::eq(p0_a.as_ptr(), p0_b.as_ptr()),
            "cache must return identical PSO for repeated key"
        );
        assert!(
            !std::ptr::eq(p0_a.as_ptr(), p27.as_ptr()),
            "distinct constants_key must produce distinct PSO"
        );
    }

    #[test]
    fn all_registered_plain_kernels_are_loaded() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };

        let mut missing = Vec::new();
        for spec in ds4_engine::kernel_registry::KERNELS {
            let func = state
                .emitted_lib
                .as_ref()
                .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                .or_else(|| {
                    state
                        .bridge_lib
                        .as_ref()
                        .and_then(|l| l.get_function(spec.metal_fn, None).ok())
                });

            let Some(func) = func else {
                missing.push(format!("{} ({})", spec.name, spec.metal_fn));
                continue;
            };
            if function_needs_constants(&func) {
                continue;
            }
            if !state.pipelines.contains_key(spec.metal_fn) {
                missing.push(format!("{} ({})", spec.name, spec.metal_fn));
            }
        }

        assert!(
            missing.is_empty(),
            "registered non-specialized kernel(s) not loaded: {missing:?}"
        );
    }

    #[test]
    fn registered_function_constant_kernels_specialize() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };

        let mul_mv_symbols = [
            "ds4_kernel_mul_mv_id_q2_K_f32",
            "ds4_kernel_mul_mv_id_q4_K_f32",
            "ds4_kernel_mul_mv_id_iq2_xxs_f32",
            "ds4_kernel_mul_mv_id_iq2_xxs_pair_f32",
            "ds4_kernel_mul_mv_id_q4_K_pair_f32",
            "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32",
            "ds4_kernel_mul_mv_id_q4_K_pair_swiglu_f32",
            "ds4_kernel_mul_mv_id_q2_K_sum6_f32",
            "ds4_kernel_mul_mv_id_q4_K_sum6_f32",
            "ds4_dsv4_q8_hc_expand4_q8_0",
            "ds4_dsv4_shared_down_hc_expand4_q8_0",
            "ds4_dsv4_shared_gate_up_swiglu_q8_0",
            "ds4_dsv4_attn_out_low_q8_0_f32",
            "ds4_kernel_mul_mv_f16_f32_4",
            "ds4_kernel_mul_mv_q8_0_f32",
        ];

        let nsg: i16 = 1;
        let nxpsg: i16 = 4;
        let mut mul_mv_key = Vec::with_capacity(4);
        mul_mv_key.extend_from_slice(&nsg.to_le_bytes());
        mul_mv_key.extend_from_slice(&nxpsg.to_le_bytes());

        let mut failures = Vec::new();
        for sym in mul_mv_symbols {
            let result = state.specialized_pipeline(sym, &mul_mv_key, |fcv| {
                fcv.set_constant_value_at_index(
                    &nsg as *const _ as *const _,
                    MTLDataType::Short,
                    FC_MUL_MV_NSG,
                );
                fcv.set_constant_value_at_index(
                    &nxpsg as *const _ as *const _,
                    MTLDataType::Short,
                    FC_MUL_MV_NXPSG,
                );
                // The IQ2_XXS pair-swiglu kernel references FC 602; harmless
                // for the other symbols (extra FCV values are ignored).
                let write_pre = false;
                fcv.set_constant_value_at_index(
                    &write_pre as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_MOE_WRITE_PRE_SWIGLU,
                );
            });
            if let Err(e) = result {
                failures.push(format!("{sym}: {e}"));
            }
        }

        let has_mask = true;
        let has_sinks = true;
        let has_bias = false;
        let has_scap = false;
        let has_kvpad = false;
        let ns10: i32 = 512;
        let ns20: i32 = 512;
        let flash_nsg: i32 = 1;
        let nwg: i32 = 32;
        let mut flash_key = Vec::with_capacity(21);
        flash_key.push(has_mask as u8);
        flash_key.push(has_sinks as u8);
        flash_key.push(has_bias as u8);
        flash_key.push(has_scap as u8);
        flash_key.push(has_kvpad as u8);
        flash_key.extend_from_slice(&ns10.to_le_bytes());
        flash_key.extend_from_slice(&ns20.to_le_bytes());
        flash_key.extend_from_slice(&flash_nsg.to_le_bytes());
        flash_key.extend_from_slice(&nwg.to_le_bytes());

        if let Err(e) = state.specialized_pipeline(
            "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512",
            &flash_key,
            |fcv| {
                fcv.set_constant_value_at_index(
                    &has_mask as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_FLASH_VEC_HAS_MASK,
                );
                fcv.set_constant_value_at_index(
                    &has_sinks as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_FLASH_VEC_HAS_SINKS,
                );
                fcv.set_constant_value_at_index(
                    &has_bias as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_FLASH_VEC_HAS_BIAS,
                );
                fcv.set_constant_value_at_index(
                    &has_scap as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_FLASH_VEC_HAS_SCAP,
                );
                fcv.set_constant_value_at_index(
                    &has_kvpad as *const _ as *const _,
                    MTLDataType::Bool,
                    FC_FLASH_VEC_HAS_KVPAD,
                );
                fcv.set_constant_value_at_index(
                    &ns10 as *const _ as *const _,
                    MTLDataType::Int,
                    FC_FLASH_VEC_NS10,
                );
                fcv.set_constant_value_at_index(
                    &ns20 as *const _ as *const _,
                    MTLDataType::Int,
                    FC_FLASH_VEC_NS20,
                );
                fcv.set_constant_value_at_index(
                    &flash_nsg as *const _ as *const _,
                    MTLDataType::Int,
                    FC_FLASH_VEC_NSG,
                );
                fcv.set_constant_value_at_index(
                    &nwg as *const _ as *const _,
                    MTLDataType::Int,
                    FC_FLASH_VEC_NWG,
                );
            },
        ) {
            failures.push(format!(
                "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512: {e}"
            ));
        }

        assert!(
            failures.is_empty(),
            "registered function-constant kernel(s) failed to specialize:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn real_gguf_q8_0_matvec_matches_dequant_oracle() {
        use ds4_engine::gguf::{validate_ds4_layout, GgmlType};
        use ds4_engine::layer_view::LayerViews;

        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("DS4_GGUF unset — skipping real-GGUF Q8_0 matvec execution check.");
                return;
            }
        };
        if !path.is_file() {
            eprintln!(
                "DS4_GGUF={} is not a regular file — skipping.",
                path.display()
            );
            return;
        }

        let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
        let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };
        for role in [
            "mla_q_a",
            "moe_gate_shared",
            "moe_up_shared",
            "moe_down_shared",
        ] {
            let h = views
                .layer(0)
                .require(role)
                .unwrap_or_else(|e| panic!("layer 0 {role} Q8_0 tensor: {e}"));
            assert_eq!(h.ttype, GgmlType::Q8_0);
            assert_eq!(h.dims.len(), 2);
            let d_in = h.dims[0] as usize;
            let d_out = h.dims[1] as usize;
            let x: Vec<f32> = (0..d_in)
                .map(|i| ((i as f32 * 0.011).sin() * 1.3) + ((i as f32 * 0.017).cos() * 0.2))
                .collect();
            let w_f32 = views
                .dequant_f32_simple(h)
                .unwrap_or_else(|e| panic!("dequant layer 0 {role}: {e}"));
            let mut oracle = vec![0.0f32; d_out];
            for row in 0..d_out {
                let w_row = &w_f32[row * d_in..(row + 1) * d_in];
                oracle[row] = w_row.iter().zip(&x).map(|(&a, &b)| a * b).sum();
            }

            let got = state
                .matvec_q8_0_bytes_impl(views.bytes_for(h).expect("raw Q8_0 bytes"), &x, d_out)
                .unwrap_or_else(|e| panic!("execute Q8_0 Metal matvec for {role}: {e}"));
            let max_abs_diff = got
                .iter()
                .zip(oracle.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_diff <= 2.0e-3,
                "real-GGUF Q8_0 Metal matvec drifted from dequant oracle for {role}: max_abs_diff={max_abs_diff:e}"
            );
            eprintln!(
                "real-GGUF Q8_0 Metal matvec PASS: role={role}, tensor={}, d_in={d_in}, d_out={d_out}, max_abs_diff={max_abs_diff:e}",
                h.name
            );
        }
    }

    #[test]
    fn real_gguf_attn_out_low_q8_0_matches_dequant_oracle() {
        use ds4_engine::gguf::{validate_ds4_layout, GgmlType};
        use ds4_engine::layer_view::LayerViews;

        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("DS4_GGUF unset — skipping real-GGUF attn_out_low execution check.");
                return;
            }
        };
        if !path.is_file() {
            eprintln!(
                "DS4_GGUF={} is not a regular file — skipping.",
                path.display()
            );
            return;
        }

        let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
        let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };

        let h = views
            .layer(0)
            .require("attn_output_a")
            .expect("layer 0 attn_output_a Q8_0 tensor");
        assert_eq!(h.ttype, GgmlType::Q8_0);
        assert_eq!(h.dims.len(), 2);
        let group_dim = h.dims[0] as usize;
        let out_low_dim = h.dims[1] as usize;
        let n_groups = 8usize;
        assert_eq!(
            out_low_dim % n_groups,
            0,
            "attn_output_a rows must be divisible by n_out_group"
        );
        let n_lora_o = out_low_dim / n_groups;
        let heads: Vec<f32> = (0..(n_groups * group_dim))
            .map(|i| ((i as f32 * 0.009).sin() * 0.75) + ((i as f32 * 0.013).cos() * 0.15))
            .collect();

        let w_f32 = views
            .dequant_f32_simple(h)
            .expect("dequant layer 0 attn_output_a");
        let mut oracle = vec![0.0f32; out_low_dim];
        for g in 0..n_groups {
            let x = &heads[g * group_dim..(g + 1) * group_dim];
            for l in 0..n_lora_o {
                let row = g * n_lora_o + l;
                let w_row = &w_f32[row * group_dim..(row + 1) * group_dim];
                oracle[row] = w_row.iter().zip(x).map(|(&a, &b)| a * b).sum();
            }
        }

        let got = state
            .attn_out_low_q8_0_bytes_impl(
                views.bytes_for(h).expect("raw attn_output_a Q8_0 bytes"),
                &heads,
                n_groups,
                n_lora_o,
            )
            .expect("execute attn_out_low Q8_0 Metal kernel");
        let max_abs_diff = got
            .iter()
            .zip(oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= 2.0e-3,
            "real-GGUF attn_out_low Q8_0 drifted from dequant oracle: max_abs_diff={max_abs_diff:e}"
        );
        eprintln!(
            "real-GGUF attn_out_low Q8_0 PASS: tensor={}, group_dim={group_dim}, n_groups={n_groups}, n_lora_o={n_lora_o}, max_abs_diff={max_abs_diff:e}",
            h.name
        );
    }

    #[test]
    fn real_gguf_q8_hc_expand4_matches_dequant_oracle() {
        use ds4_engine::gguf::{validate_ds4_layout, GgmlType};
        use ds4_engine::layer_view::LayerViews;

        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("DS4_GGUF unset — skipping real-GGUF Q8_0 HC expand execution check.");
                return;
            }
        };
        if !path.is_file() {
            eprintln!(
                "DS4_GGUF={} is not a regular file — skipping.",
                path.display()
            );
            return;
        }

        let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
        let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
        let h = views
            .layer(0)
            .require("moe_down_shared")
            .expect("layer 0 shared down Q8_0 tensor");
        assert_eq!(h.ttype, GgmlType::Q8_0);
        assert_eq!(h.dims.len(), 2);
        let d_in = h.dims[0] as usize;
        let d_out = h.dims[1] as usize;
        let input: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.019).sin() * 0.9) - ((i as f32 * 0.005).cos() * 0.1))
            .collect();
        let residual_dh: Vec<f32> = (0..d_out * 4)
            .map(|i| ((i as f32 * 0.007).sin() * 0.4) + ((i as f32 * 0.013).cos() * 0.2))
            .collect();
        let post: Vec<f32> = vec![0.35, -0.20, 0.15, 0.42];
        let comb: Vec<f32> = (0..16)
            .map(|i| ((i as f32 * 0.23).sin() * 0.3) + if i % 5 == 0 { 0.5 } else { 0.0 })
            .collect();

        let w_f32 = views
            .dequant_f32_simple(h)
            .expect("dequant layer 0 shared down");
        let mut block_oracle = vec![0.0f32; d_out];
        for row in 0..d_out {
            let w_row = &w_f32[row * d_in..(row + 1) * d_in];
            block_oracle[row] = w_row.iter().zip(&input).map(|(&a, &b)| a * b).sum();
        }
        let mut dst_oracle = vec![0.0f32; d_out * 4];
        for d in 0..d_out {
            for dst_hc in 0..4 {
                let mut acc = block_oracle[d] * post[dst_hc];
                for src_hc in 0..4 {
                    acc += comb[dst_hc * 4 + src_hc] * residual_dh[d * 4 + src_hc];
                }
                dst_oracle[d * 4 + dst_hc] = acc;
            }
        }

        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };
        let (block_got, dst_got) = state
            .q8_hc_expand4_bytes_impl(
                views.bytes_for(h).expect("raw Q8_0 bytes"),
                &input,
                &residual_dh,
                &post,
                &comb,
                d_out,
            )
            .expect("execute Q8_0 HC expand Metal kernel");
        let block_diff = block_got
            .iter()
            .zip(block_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let dst_diff = dst_got
            .iter()
            .zip(dst_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            block_diff <= 2.0e-3,
            "real-GGUF Q8_0 HC expand block_out drift: max_abs_diff={block_diff:e}"
        );
        assert!(
            dst_diff <= 2.0e-3,
            "real-GGUF Q8_0 HC expand dst drift: max_abs_diff={dst_diff:e}"
        );
        eprintln!(
            "real-GGUF Q8_0 HC expand PASS: tensor={}, d_in={d_in}, d_out={d_out}, block_diff={block_diff:e}, dst_diff={dst_diff:e}",
            h.name
        );
    }

    #[test]
    fn real_gguf_shared_down_hc_expand4_matches_dequant_oracle() {
        use ds4_engine::gguf::{validate_ds4_layout, GgmlType};
        use ds4_engine::layer_view::LayerViews;

        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!(
                    "DS4_GGUF unset — skipping real-GGUF shared-down HC expand execution check."
                );
                return;
            }
        };
        if !path.is_file() {
            eprintln!(
                "DS4_GGUF={} is not a regular file — skipping.",
                path.display()
            );
            return;
        }

        let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
        let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
        let h = views
            .layer(0)
            .require("moe_down_shared")
            .expect("layer 0 shared down Q8_0 tensor");
        assert_eq!(h.ttype, GgmlType::Q8_0);
        assert_eq!(h.dims.len(), 2);
        let d_in = h.dims[0] as usize;
        let d_out = h.dims[1] as usize;
        let shared_mid: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.017).sin() * 0.7) + ((i as f32 * 0.009).cos() * 0.15))
            .collect();
        let routed_out: Vec<f32> = (0..d_out)
            .map(|i| ((i as f32 * 0.003).sin() * 0.25) - ((i as f32 * 0.021).cos() * 0.05))
            .collect();
        let residual_dh: Vec<f32> = (0..d_out * 4)
            .map(|i| ((i as f32 * 0.007).sin() * 0.4) + ((i as f32 * 0.013).cos() * 0.2))
            .collect();
        let post: Vec<f32> = vec![0.31, -0.22, 0.19, 0.41];
        let comb: Vec<f32> = (0..16)
            .map(|i| ((i as f32 * 0.19).cos() * 0.25) + if i % 5 == 0 { 0.45 } else { 0.0 })
            .collect();

        let w_f32 = views
            .dequant_f32_simple(h)
            .expect("dequant layer 0 shared down");
        let mut shared_oracle = vec![0.0f32; d_out];
        for row in 0..d_out {
            let w_row = &w_f32[row * d_in..(row + 1) * d_in];
            shared_oracle[row] = w_row.iter().zip(&shared_mid).map(|(&a, &b)| a * b).sum();
        }
        let mut dst_oracle = vec![0.0f32; d_out * 4];
        for d in 0..d_out {
            let block_v = shared_oracle[d] + routed_out[d];
            for dst_hc in 0..4 {
                let mut acc = block_v * post[dst_hc];
                for src_hc in 0..4 {
                    acc += comb[dst_hc * 4 + src_hc] * residual_dh[d * 4 + src_hc];
                }
                dst_oracle[d * 4 + dst_hc] = acc;
            }
        }

        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };
        let (shared_got, dst_got) = state
            .shared_down_hc_expand4_q8_0_bytes_impl(
                views.bytes_for(h).expect("raw Q8_0 bytes"),
                &shared_mid,
                &routed_out,
                &residual_dh,
                &post,
                &comb,
                d_out,
            )
            .expect("execute shared-down HC expand Metal kernel");
        let shared_diff = shared_got
            .iter()
            .zip(shared_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let dst_diff = dst_got
            .iter()
            .zip(dst_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            shared_diff <= 2.0e-3,
            "real-GGUF shared-down HC expand shared_out drift: max_abs_diff={shared_diff:e}"
        );
        assert!(
            dst_diff <= 2.0e-3,
            "real-GGUF shared-down HC expand dst drift: max_abs_diff={dst_diff:e}"
        );
        eprintln!(
            "real-GGUF shared-down HC expand PASS: tensor={}, d_in={d_in}, d_out={d_out}, shared_diff={shared_diff:e}, dst_diff={dst_diff:e}",
            h.name
        );
    }

    #[test]
    fn real_gguf_shared_gate_up_swiglu_q8_0_matches_dequant_oracle() {
        use ds4_engine::gguf::{validate_ds4_layout, GgmlType};
        use ds4_engine::layer_view::LayerViews;

        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!(
                    "DS4_GGUF unset — skipping real-GGUF shared gate/up SwiGLU execution check."
                );
                return;
            }
        };
        if !path.is_file() {
            eprintln!(
                "DS4_GGUF={} is not a regular file — skipping.",
                path.display()
            );
            return;
        }

        let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
        let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
        let gate_h = views
            .layer(0)
            .require("moe_gate_shared")
            .expect("layer 0 shared gate Q8_0 tensor");
        let up_h = views
            .layer(0)
            .require("moe_up_shared")
            .expect("layer 0 shared up Q8_0 tensor");
        assert_eq!(gate_h.ttype, GgmlType::Q8_0);
        assert_eq!(up_h.ttype, GgmlType::Q8_0);
        assert_eq!(gate_h.dims, up_h.dims);
        assert_eq!(gate_h.dims.len(), 2);
        let d_in = gate_h.dims[0] as usize;
        let d_out = gate_h.dims[1] as usize;
        let x: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.015).sin() * 0.8) + ((i as f32 * 0.011).cos() * 0.2))
            .collect();

        let gate_f32 = views
            .dequant_f32_simple(gate_h)
            .expect("dequant shared gate");
        let up_f32 = views.dequant_f32_simple(up_h).expect("dequant shared up");
        let mut gate_oracle = vec![0.0f32; d_out];
        let mut up_oracle = vec![0.0f32; d_out];
        let mut mid_oracle = vec![0.0f32; d_out];
        for row in 0..d_out {
            let gate_row = &gate_f32[row * d_in..(row + 1) * d_in];
            let up_row = &up_f32[row * d_in..(row + 1) * d_in];
            let gate: f32 = gate_row.iter().zip(&x).map(|(&a, &b)| a * b).sum();
            let up: f32 = up_row.iter().zip(&x).map(|(&a, &b)| a * b).sum();
            gate_oracle[row] = gate;
            up_oracle[row] = up;
            mid_oracle[row] = gate / (1.0 + (-gate).exp()) * up;
        }

        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };
        let (gate_got, up_got, mid_got) = state
            .shared_gate_up_swiglu_q8_0_bytes_impl(
                views.bytes_for(gate_h).expect("raw gate Q8_0 bytes"),
                views.bytes_for(up_h).expect("raw up Q8_0 bytes"),
                &x,
                d_out,
            )
            .expect("execute shared gate/up SwiGLU Metal kernel");
        let gate_diff = gate_got
            .iter()
            .zip(gate_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let up_diff = up_got
            .iter()
            .zip(up_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mid_diff = mid_got
            .iter()
            .zip(mid_oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            gate_diff <= 2.0e-3,
            "real-GGUF shared gate Q8_0 drift: max_abs_diff={gate_diff:e}"
        );
        assert!(
            up_diff <= 2.0e-3,
            "real-GGUF shared up Q8_0 drift: max_abs_diff={up_diff:e}"
        );
        assert!(
            mid_diff <= 2.0e-3,
            "real-GGUF shared SwiGLU drift: max_abs_diff={mid_diff:e}"
        );
        eprintln!(
            "real-GGUF shared gate/up SwiGLU PASS: gate={}, up={}, d_in={d_in}, d_out={d_out}, gate_diff={gate_diff:e}, up_diff={up_diff:e}, mid_diff={mid_diff:e}",
            gate_h.name,
            up_h.name
        );
    }

    /// Bit-parity gate for the rewritten `ds4_dsv4_kv_fp8_store` shim: it
    /// must produce the exact persistent-slot bytes that the CPU "slot
    /// correction" writes (`ds4_fp8_kv_quantize_row_inplace` +
    /// `f16_round_trip_f32`, single_buffer_encoder.rs:974-999). The old
    /// shim did an unscaled per-element e4m3 round-trip and diverged, which
    /// is why the CPU pass existed; this proves the GPU write is now correct
    /// on its own so that pass can be dropped.
    #[test]
    fn kv_fp8_store_persistent_matches_cpu_correction() {
        use ds4_engine::attn_dispatch::{
            ds4_fp8_kv_quantize_row_inplace, f16_round_trip_f32, LayerParams,
        };

        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {}", e);
                return;
            }
        };

        let raw_cap: u32 = 4;
        let slot: u32 = 2;

        // Shapes: V4-Flash production (n_nope=448, 64-aligned), a 64-aligned
        // full row (n_nope=512), and two non-aligned n_nope (n_nope=200, 444)
        // to exercise the kernel's partial tail-block path.
        let shapes: &[(usize, usize)] = &[(512, 64), (512, 0), (208, 8), (512, 68)];

        for (li, &(head_dim, n_rot)) in shapes.iter().enumerate() {
            // Distinct layer_idx per shape — the persistent KV buffer is
            // cached by layer_idx and sized by n_lora_kv, which varies here.
            let layer_idx = li as u32;
            let n_lora_kv = head_dim; // DS4: kv-down row == head_dim
            let n_nope = head_dim - n_rot;

            // Deterministic pseudo-random row spanning several magnitudes so
            // each 64-block lands on a different amax/scale.
            let mut seed: u32 = 0x1234_5678 ^ (head_dim as u32) << 3 ^ n_rot as u32;
            let mut next = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 16_777_216.0) - 0.5 // [-0.5, 0.5)
            };
            let kv_row: Vec<f32> = (0..n_lora_kv)
                .map(|i| next() * (1.0 + (i / 64) as f32 * 7.0))
                .collect();

            let params = LayerParams {
                layer_idx,
                d_embd: 1,
                n_hc: 1,
                n_head: 1,
                head_dim: head_dim as u32,
                n_rot: n_rot as u32,
                n_lora_q: 1,
                n_lora_kv: n_lora_kv as u32,
                hc_sinkhorn_iter: 0,
                hc_eps: 1e-6,
                rms_eps: 1e-6,
                rope_orig_ctx: 4096,
                rope_freq_base: 10000.0,
                rope_freq_scale: 1.0,
                rope_ext_factor: 0.0,
                rope_attn_factor: 1.0,
                compress_ratio: 0,
                n_out_group: 1,
            };

            let cache_buf = state
                .kv_fp8_store_persistent_impl(layer_idx, &params, &kv_row, raw_cap, slot)
                .expect("kv_fp8_store_persistent_impl");

            let off = (slot as usize) * n_lora_kv;
            let gpu_slot: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    (cache_buf.contents() as *const f32).add(off),
                    n_lora_kv,
                )
            };

            // CPU reference — identical to the slot-correction block.
            let mut cpu = kv_row.clone();
            ds4_fp8_kv_quantize_row_inplace(&mut cpu, head_dim, n_rot);
            for d in cpu.iter_mut() {
                *d = f16_round_trip_f32(*d);
            }

            let mut n_diff = 0usize;
            let mut max_ulp = 0u32;
            for (i, (&g, &c)) in gpu_slot.iter().zip(cpu.iter()).enumerate() {
                if g.to_bits() != c.to_bits() {
                    n_diff += 1;
                    let d = (g.to_bits() as i64 - c.to_bits() as i64).unsigned_abs() as u32;
                    max_ulp = max_ulp.max(d);
                    if n_diff <= 4 {
                        eprintln!("  [hd={head_dim},nrot={n_rot}] mismatch[{i}]: gpu={g} cpu={c} (ulp={d})");
                    }
                }
            }
            assert_eq!(
                n_diff, 0,
                "GPU kv_fp8_store diverges from CPU (head_dim={head_dim}, n_rot={n_rot}, n_nope={n_nope}): {n_diff}/{n_lora_kv} differ, max_ulp={max_ulp}"
            );
        }
    }

    /// Isolated warmed microbench for the K=1 production routed-MoE
    /// (`pair_swiglu` + `sum6`) on real layer-5 expert weights. Built to
    /// measure the `FC_moe_write_pre_swiglu` dead-store removal: run once at
    /// the default (stores SKIPPED) and once with `DS4_MOE_WRITE_PRE_SWIGLU=1`
    /// (stores RESTORED), then compare. Prints mean/median/min ms plus an
    /// output checksum so the two runs can be confirmed bit-identical.
    ///
    /// Gated by `DS4_BENCH_MOE_DEADSTORE=1` + a valid `DS4_GGUF`. macOS-only,
    /// heavy (full expert-weight load). Pair with `DS4_MOE_TRACE=1` for the
    /// per-call GPUStart→End span (the legacy K=1 path commits+waits per call).
    #[test]
    fn moe_routed_step_dead_store_microbench() {
        if std::env::var("DS4_BENCH_MOE_DEADSTORE").ok().as_deref() != Some("1") {
            eprintln!(
                "DS4_BENCH_MOE_DEADSTORE unset — skipping. Set =1 (and DS4_GGUF=/path) to run."
            );
            return;
        }
        let path = match std::env::var("DS4_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("DS4_GGUF unset — skipping");
                return;
            }
        };
        if !path.is_file() {
            eprintln!("DS4_GGUF={} is not a file — skipping", path.display());
            return;
        }

        let manifest =
            ds4_engine::gguf::validate_ds4_layout(&path).expect("validate_ds4_layout");
        let views = ds4_engine::layer_view::LayerViews::open(&path, manifest.n_layers)
            .expect("LayerViews::open");
        let gguf = ds4_engine::gguf::GgufFile::open(&path).expect("GgufFile::open");

        let mut state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {e}");
                return;
            }
        };
        state
            .load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers)
            .expect("load_expert_weights");

        let layer_idx: u32 = std::env::var("DS4_BENCH_MOE_LAYER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let qew = &state.expert_weights[layer_idx as usize];
        let (n_experts, d_in, d_ffn) =
            (qew.n_experts as usize, qew.d_in as usize, qew.d_ffn as usize);
        eprintln!(
            "layer {layer_idx}: n_experts={n_experts} d_in={d_in} d_ffn={d_ffn}  \
             gate={:?} up={:?} down={:?}",
            qew.gate.ttype, qew.up.ttype, qew.down.ttype
        );
        // Dead-store write volume removed when SKIPPED: 2 (gate+up) × 6 experts
        // × d_ffn × 4 bytes per call.
        eprintln!(
            "  gate/up store volume = {} KiB/call (skipped when off)",
            2 * 6 * d_ffn * 4 / 1024
        );

        // Deterministic activation + fixed top-6 routing (same 6 experts every
        // iter ⇒ fully page-warm after the warmup iters).
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.013).sin() * 1.7).collect();
        let selected: Vec<usize> = (0..6).map(|i| (i * 37 + 5) % n_experts).collect();
        let weights: Vec<f32> = (0..6).map(|i| 0.1 + 0.05 * i as f32).collect();

        let iters: usize = std::env::var("DS4_BENCH_MOE_DEADSTORE_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        // `moe_routed_step_encode` reads DS4_MOE_WRITE_PRE_SWIGLU per call and
        // keys the specialized pipeline on it, so we can toggle modes in ONE
        // process (same thermal/clock state) by flipping the env var. Both
        // pipelines specialize on first use; we warm both, then interleave
        // A/B/A/B so any clock drift hits both modes equally.
        let run = |state: &MetalState, write_pre: bool| -> (f64, (f64, f32)) {
            std::env::set_var("DS4_MOE_WRITE_PRE_SWIGLU", if write_pre { "1" } else { "0" });
            let out =
                state.moe_routed_step(layer_idx, &x, &selected, &weights, &[], &[], &[], 0);
            let t0 = std::time::Instant::now();
            let _ = state.moe_routed_step(layer_idx, &x, &selected, &weights, &[], &[], &[], 0);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let sum: f64 = out.iter().map(|&v| v as f64).sum();
            let maxabs = out.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            (ms, (sum, maxabs))
        };

        // Warm both specializations + pages.
        let (_, cs_skip) = run(&state, false);
        let (_, cs_keep) = run(&state, true);
        for _ in 0..5 {
            run(&state, false);
            run(&state, true);
        }

        let mut skip_ms: Vec<f64> = Vec::with_capacity(iters);
        let mut keep_ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            skip_ms.push(run(&state, false).0);
            keep_ms.push(run(&state, true).0);
        }
        std::env::remove_var("DS4_MOE_WRITE_PRE_SWIGLU");

        let stats = |v: &mut Vec<f64>| -> (f64, f64, f64) {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (mean, v[v.len() / 2], v[0])
        };
        let (sk_mean, sk_med, sk_min) = stats(&mut skip_ms);
        let (kp_mean, kp_med, kp_min) = stats(&mut keep_ms);

        eprintln!("\ninterleaved A/B, {iters} samples each (same process/clock):");
        eprintln!(
            "  SKIPPED  (opt on, default): mean={sk_mean:.4} ms  median={sk_med:.4} ms  min={sk_min:.4} ms"
        );
        eprintln!(
            "  RESTORED (stores kept)    : mean={kp_mean:.4} ms  median={kp_med:.4} ms  min={kp_min:.4} ms"
        );
        eprintln!(
            "  delta (RESTORED-SKIPPED): mean={:+.4} ms ({:+.2}%)  min={:+.4} ms",
            kp_mean - sk_mean,
            (kp_mean - sk_mean) / kp_mean * 100.0,
            kp_min - sk_min
        );
        eprintln!(
            "  checksums: skipped sum={:.6} max_abs={:.6} | restored sum={:.6} max_abs={:.6}",
            cs_skip.0, cs_skip.1, cs_keep.0, cs_keep.1
        );
        assert_eq!(
            cs_skip, cs_keep,
            "dead-store removal changed MoE output — must be bit-identical"
        );
    }

    /// GPU argmax kernel (ds4_argmax_f32) vs the CPU `argmax_i32` (lowest index
    /// on ties), over several sizes incl. the real vocab (129280) and tie cases.
    /// No model load.
    #[test]
    fn gpu_argmax_matches_cpu() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: MetalState::new failed: {e}");
                return;
            }
        };
        fn cpu_argmax(v: &[f32]) -> i32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > bv {
                    bv = x;
                    bi = i;
                }
            }
            bi as i32
        }
        let mut rng: u32 = 0x1357_9bdf;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for &n in &[2usize, 256, 4000, 129280] {
            let mut logits: Vec<f32> = (0..n).map(|_| next()).collect();
            // Place the max at a low index and TIE it at a higher index so the
            // lowest-index tie-break is exercised.
            let lo = 5.min(n - 1);
            logits[lo] = 9.0;
            if n > 50 {
                logits[50] = 9.0; // tie with `lo` → both must pick `lo`
            }
            let lbuf = new_input_buffer(&state.device, &logits);
            let tbuf = new_output_buffer::<i32>(&state.device, 1);
            let pipe = state
                .specialized_pipeline("ds4_argmax_f32", &[], |_fcv| {})
                .expect("argmax pipeline");
            let cb = state.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&lbuf), 0);
            enc.set_buffer(1, Some(&tbuf), 0);
            let nu = n as u32;
            set_scalar_bytes(enc, 2, &nu);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            end_shared_compute_enc(enc);
            cb.commit();
            cb.wait_until_completed();
            let gpu = unsafe { read_buffer::<i32>(&tbuf, 1) }[0];
            let cpu = cpu_argmax(&logits);
            eprintln!("argmax n={n}: gpu={gpu} cpu={cpu}");
            assert_eq!(gpu, cpu, "GPU argmax != CPU argmax at n={n}");
        }
    }

    /// `ds4_kernel_mul_mv_q4_0_f32` (bridge_shims/dsv4_mul_mv_q4_0_f32.metal)
    /// must match the CPU q4_0 dequant matvec on identical block bytes —
    /// proving the kernel's nibble unpack / scale / layout are correct. Random
    /// `block_q4_0` weights (any nibble, small f16 scale) + a smooth activation;
    /// GPU vs `dequant_q4_0_block` + dot. (No requant — this isolates the kernel.)
    #[test]
    fn q4_0_matvec_parity() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip q4_0_matvec_parity: MetalState::new failed: {e}");
                return;
            }
        };
        let d_in = 256usize;
        let d_out = 64usize;
        let nb = d_in / 32; // blocks per row

        let mut s: u32 = 0x1234_5678;
        let mut rng = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        };
        let mut w = Vec::<u8>::with_capacity(d_out * nb * 18);
        for _ in 0..d_out * nb {
            // small positive f16 scale in ~[0.005, 0.07]
            let d = 0.005f32 + (rng() & 0xff) as f32 * 2.5e-4;
            let dbits = crate::f16_cast::f32_to_f16_bits(d);
            w.push((dbits & 0xff) as u8);
            w.push((dbits >> 8) as u8);
            for _ in 0..16 {
                w.push((rng() & 0xff) as u8); // two random nibbles
            }
        }
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();

        let gpu = state
            .matvec_q4_0_bytes_impl(&w, &x, d_out)
            .expect("matvec_q4_0_bytes_impl");

        // CPU reference: dequant each block, dot with the matching x slice.
        let mut cpu = vec![0.0f32; d_out];
        for (r, cpu_r) in cpu.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for b in 0..nb {
                let blk = &w[(r * nb + b) * 18..(r * nb + b + 1) * 18];
                let wt = ds4_engine::layer_view::dequant_q4_0_block(blk);
                for k in 0..32 {
                    acc += wt[k] * x[b * 32 + k];
                }
            }
            *cpu_r = acc;
        }

        let max_abs = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("q4_0 matvec parity: max_abs={max_abs:.3e} (d_in={d_in} d_out={d_out})");
        assert!(max_abs < 1.0e-3, "q4_0 GPU vs CPU matvec max_abs={max_abs}");
    }
}
