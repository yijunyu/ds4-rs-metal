//! Metal back-end for `ds4_engine::dispatch::KernelDispatcher`.
//!
//! On macOS, `MetalDispatcher::new(device)` returns an instance that
//! dispatches each `KernelDispatcher` method through a generated MSL
//! kernel (looked up by `KernelSpec.metal_fn` in the kernel registry).
//!
//! On Linux, the crate still compiles — `MetalDispatcher::new()`
//! returns an error so callers never receive a non-functional
//! dispatcher. The trait impl is therefore macOS-only.
//!
//! ## Validation
//!
//! Bit-tolerance correctness is checked against the CPU reference via
//! `ds4_engine::dispatch::TracingDispatcher` + `check_traces_close`:
//!
//! ```text
//! let cpu = TracingDispatcher::new(&CpuDispatcher);
//! let metal = TracingDispatcher::new(&MetalDispatcher::new(device)?);
//! decode_step_with(&cpu,   x.clone(), &model, &cfg)?;
//! decode_step_with(&metal, x,         &model, &cfg)?;
//! check_traces_close(&cpu.events(), &metal.events(), 1e-5)?;
//! ```
//!
//! Per-kernel µs are measured via `PerKernelTimingDispatcher`.

use anyhow::Result;

pub mod cpu_via_dequant;
#[cfg(target_os = "macos")]
pub mod decode_runner;
mod f16_cast;
mod iq2_xxs_tables;
pub mod quantized_experts;
#[cfg(target_os = "macos")]
pub mod single_buffer_encoder;

#[cfg(target_os = "macos")]
pub mod deferred;

#[cfg(target_os = "macos")]
pub mod mtp_bundle;

#[cfg(target_os = "macos")]
pub mod base_run_bundle;

pub mod spec_decode;

/// CPU logits sampling (temperature/top_k/top_p/min_p + seedable RNG),
/// ported from antirez ds4.c. Used by the ds4_server HTTP layer.
pub mod sampling;

#[cfg(target_os = "macos")]
pub mod compressor;

#[cfg(target_os = "macos")]
mod macos;

/// Counts collected by [`MetalDispatcher::buffer_audit`]. Used to compare
/// our per-tensor MTLBuffer pattern against antirez's chunked-views
/// design described in [[m5-antirez-insights]] §(2).
#[derive(Default, Debug, Clone)]
pub struct BufferAuditReport {
    /// One `new_buffer_with_bytes_no_copy` per (layer, gate|up|down).
    pub n_expert_buffers: usize,
    pub expert_bytes_total: u64,
    /// One `new_buffer_with_data` per cached f32 weight slice.
    pub n_weight_cache_buffers: usize,
    pub weight_cache_bytes_total: u64,
    /// One persistent KV buffer per layer (allocated lazily).
    pub n_kv_cache_buffers: usize,
    pub kv_cache_bytes_total: u64,
    /// 7 reusable scratch buffers per unique `(d_in, d_ffn)` MoE shape.
    pub n_moe_scratch_pools: usize,
    pub n_moe_scratch_buffers: usize,
    /// Phase F task #93: weight cache zero-copy vs memcpy stats.
    pub n_weight_no_copy: usize,
    pub n_weight_memcpy: usize,
    pub weight_no_copy_bytes: u64,
    pub weight_memcpy_bytes: u64,
}

impl BufferAuditReport {
    pub fn total_buffers(&self) -> usize {
        self.n_expert_buffers
            + self.n_weight_cache_buffers
            + self.n_kv_cache_buffers
            + self.n_moe_scratch_buffers
    }

    pub fn total_bytes(&self) -> u64 {
        self.expert_bytes_total + self.weight_cache_bytes_total + self.kv_cache_bytes_total
    }
}

/// Metal back-end dispatcher. Construct via `MetalDispatcher::new`.
///
/// Phase 2 MoE-K Step 5 — view of a layer's expert weights for K-amortized
/// MoE dispatch. Returned by `MetalDispatcher::expert_weight_bufs`. Held
/// by reference into the dispatcher; lifetime tied to the dispatcher.
#[cfg(target_os = "macos")]
pub struct ExpertWeightBufs<'a> {
    pub gate: &'a metal::Buffer,
    pub up: &'a metal::Buffer,
    pub down: &'a metal::Buffer,
    pub gate_ttype: ds4_engine::gguf::GgmlType,
    pub up_ttype: ds4_engine::gguf::GgmlType,
    pub down_ttype: ds4_engine::gguf::GgmlType,
    pub n_experts: usize,
    pub d_in: usize,
    pub d_ffn: usize,
}

/// On non-macOS platforms the struct still exists but cannot be
/// instantiated (constructors return `Err`).
pub struct MetalDispatcher {
    #[cfg(target_os = "macos")]
    inner: macos::MetalState,
    #[cfg(not(target_os = "macos"))]
    _phantom: std::marker::PhantomData<()>,
}

impl MetalDispatcher {
    /// Initialise the Metal back end. On Linux this always returns
    /// `Err`; on macOS it discovers the system default device and
    /// compiles the kernel library.
    #[cfg(target_os = "macos")]
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: macos::MetalState::new()?,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Result<Self> {
        anyhow::bail!("ds4_metal::MetalDispatcher requires macOS")
    }

    /// Dense Q8_0 matvec (`out[d_out] = dequant_q8_0(w)·x`), own command
    /// buffer. Test-only passthrough for the decode-q4 bandwidth probe.
    #[cfg(target_os = "macos")]
    pub fn matvec_q8_0_dense(&self, w: &[u8], x: &[f32], d_out: usize) -> Result<Vec<f32>> {
        self.inner.matvec_q8_0_bytes_impl(w, x, d_out)
    }

    /// Dense Q4_0 matvec twin of `matvec_q8_0_dense` (half the weight bytes).
    #[cfg(target_os = "macos")]
    pub fn matvec_q4_0_dense(&self, w: &[u8], x: &[f32], d_out: usize) -> Result<Vec<f32>> {
        self.inner.matvec_q4_0_bytes_impl(w, x, d_out)
    }

    /// SSD-streaming MoE bind (DS4_SSD_STREAM): ensure the selected experts
    /// are in the layer's cache and return (slot ids, gate, up, down)
    /// buffers. None when streaming is off. macOS only.
    #[cfg(target_os = "macos")]
    pub fn streaming_expert_bind(
        &self,
        layer_idx: u32,
        selected: &[i32],
    ) -> Option<(Vec<i32>, metal::Buffer, metal::Buffer, metal::Buffer)> {
        self.inner.streaming_expert_bind(layer_idx, selected)
    }

    /// Load per-layer quantized expert weight tables from a GGUF.
    /// Must be called before the first `moe_routed_step`. macOS only.
    #[cfg(target_os = "macos")]
    pub fn load_expert_weights(
        &mut self,
        gguf: &ds4_engine::gguf::GgufFile,
        gguf_bytes: &[u8],
        n_moe_layers: u32,
    ) -> Result<()> {
        self.inner
            .load_expert_weights(gguf, gguf_bytes, n_moe_layers)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn load_expert_weights(
        &mut self,
        _gguf: &ds4_engine::gguf::GgufFile,
        _gguf_bytes: &[u8],
        _n_moe_layers: u32,
    ) -> Result<()> {
        anyhow::bail!("ds4_metal::MetalDispatcher requires macOS")
    }

    /// Phase 3 — Load MTP drafter expert weights from a separate GGUF
    /// (antirez DeepSeek-V4-Flash-MTP-*.gguf). Appends one slot to the
    /// expert weight table and returns its index — the caller threads
    /// this `mtp_layer_idx` to `BatchScope::encode_mtp_draft` and the
    /// drafter's MoE chain pulls from the matching slot.
    ///
    /// MUST be called AFTER `load_expert_weights` (the base call clears
    /// the table). Returned index = `n_base_moe_layers` for a fresh load.
    #[cfg(target_os = "macos")]
    pub fn load_mtp_expert_weights(
        &mut self,
        mtp_gguf: &ds4_engine::gguf::GgufFile,
        mtp_gguf_bytes: &[u8],
    ) -> Result<u32> {
        self.inner.load_mtp_expert_weights(mtp_gguf, mtp_gguf_bytes)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn load_mtp_expert_weights(
        &mut self,
        _mtp_gguf: &ds4_engine::gguf::GgufFile,
        _mtp_gguf_bytes: &[u8],
    ) -> Result<u32> {
        anyhow::bail!("ds4_metal::MetalDispatcher requires macOS")
    }

    /// Phase 2 MoE-K Step 5 — public accessor for a layer's expert weight
    /// metal buffers. Used by the K-amortized MoE chain bench to dispatch
    /// `encode_mul_mm_id_*_k` against real-model expert weights without
    /// re-uploading. Returns `(gate_buf, up_buf, down_buf, gate_ttype,
    /// up_ttype, down_ttype, d_in, d_ffn, n_experts)`.
    ///
    /// Panics if layer_idx is out of range; `load_expert_weights` must
    /// have been called first.
    #[cfg(target_os = "macos")]
    pub fn expert_weight_bufs(
        &self,
        layer_idx: u32,
    ) -> Result<ExpertWeightBufs<'_>> {
        let qew = self
            .inner
            .expert_weights
            .get(layer_idx as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "expert_weight_bufs: no QuantizedExpertWeights for layer {} (loaded={})",
                    layer_idx,
                    self.inner.expert_weights.len(),
                )
            })?;
        Ok(ExpertWeightBufs {
            gate: &qew.gate.metal_buf,
            up: &qew.up.metal_buf,
            down: &qew.down.metal_buf,
            gate_ttype: qew.gate.ttype,
            up_ttype: qew.up.ttype,
            down_ttype: qew.down.ttype,
            n_experts: qew.n_experts as usize,
            d_in: qew.d_in as usize,
            d_ffn: qew.d_ffn as usize,
        })
    }

    /// Phase E task #82 — force every expert-weight page resident in
    /// physical memory before timed decode work.
    ///
    /// `request_model_residency` adds the buffers to an `MTLResidencySet`
    /// at load time (macOS 15+), but the DS4_OP_TRACE profile in
    /// `[[m5-moe-wait-diagnosis]]` shows per-call `moe_shared_chain`
    /// wait_us has huge variance (median 789ms, max 5.86s) while gpu_us
    /// is a flat ~700us — strongly suggesting cold-page faults on the
    /// mmap-backed 10GB model even with the residency set in place.
    ///
    /// This method walks all loaded `QuantizedExpertWeights` and reads
    /// every byte of every gate/up/down tensor via the StorageModeShared
    /// `metal::Buffer::contents()` pointer. The volatile-load loop forces
    /// the OS to page in everything (and the Metal driver to keep it
    /// resident if it's going to honor MTLResidencySet at all).
    ///
    /// Cost: ~10 GB sequential read = ~1-2 seconds on M1 Ultra. Run
    /// once after `load_expert_weights`, before any timed bench.
    /// Returns the total bytes touched (sanity check).
    #[cfg(target_os = "macos")]
    pub fn warm_up_expert_pages(&self) -> usize {
        let mut total: usize = 0;
        let mut acc: u64 = 0;
        for qew in &self.inner.expert_weights {
            for tensor in [&qew.gate, &qew.up, &qew.down] {
                let n = tensor.bytes.len();
                if n == 0 {
                    // mmap-backed: read via the metal buffer's contents
                    // pointer. metal_buf wraps the same mmap region.
                    let len = tensor.mmap_len;
                    let ptr = tensor.metal_buf.contents() as *const u8;
                    if !ptr.is_null() && len > 0 {
                        unsafe {
                            // 64KB stride is enough to touch every 16KB
                            // page (and macOS uses 16KB pages on Apple
                            // Silicon). Read one byte per page.
                            let mut off = 0usize;
                            while off < len {
                                acc = acc.wrapping_add(*ptr.add(off) as u64);
                                off = off.saturating_add(4096);
                            }
                        }
                        total += len;
                    }
                } else {
                    // Owned bytes — already in heap, no page-in needed,
                    // but touch anyway to match semantics.
                    let mut off = 0usize;
                    while off < n {
                        acc = acc.wrapping_add(tensor.bytes[off] as u64);
                        off = off.saturating_add(4096);
                    }
                    total += n;
                }
            }
        }
        // Black-box the accumulator so the compiler can't optimize the
        // loop away.
        std::hint::black_box(acc);
        total
    }

    #[cfg(not(target_os = "macos"))]
    pub fn warm_up_expert_pages(&self) -> usize {
        0
    }

    /// Flush pending residency-set additions: a single batched
    /// `commit`+`requestResidency` for all weight buffers pinned (via
    /// `addAllocation`) during this token's first-touch. Call once per decode
    /// token; a no-op once the weight caches are warm (nothing new to add).
    /// Keeps the ~31 GB of non-expert weight buffers GPU-resident so they
    /// aren't idle-compressed between requests (the cause of post-idle
    /// throughput collapse). Delegates to the inner state.
    #[cfg(target_os = "macos")]
    pub fn commit_residency(&self) {
        self.inner.commit_residency();
    }

    #[cfg(not(target_os = "macos"))]
    pub fn commit_residency(&self) {}

    /// Phase F task #90 — GPU-side page warm-up via `kernel_touch_u8_stride`,
    /// mirroring antirez `ds4_metal_warm_model_views` (`ds4_metal.m:657`).
    /// Dispatches one byte-touch per `stride` bytes for every loaded expert
    /// weight buffer in a single command buffer. Forces the GPU's first-touch
    /// page fault for every page up front so subsequent decode dispatches
    /// don't pay it.
    ///
    /// Antirez's CPU-side equivalent (our existing `warm_up_expert_pages`)
    /// regressed wall-time per task #82 — CPU reads forced eviction of other
    /// pages from the unified-memory cache. The GPU-side variant uses the
    /// GPU's own access path so the residency hint is the one Metal cares
    /// about for compute work.
    ///
    /// `stride_bytes` defaults to 1 MiB to match antirez. Returns the total
    /// bytes touched (sum of per-tensor sizes).
    #[cfg(target_os = "macos")]
    pub fn warm_up_expert_pages_gpu(&self, stride_bytes: u64) -> Result<u64> {
        if self.inner.expert_weights.is_empty() {
            return Ok(0);
        }
        let mut views: Vec<(&metal::Buffer, u64)> = Vec::new();
        for qew in &self.inner.expert_weights {
            for tensor in [&qew.gate, &qew.up, &qew.down] {
                let bytes = if tensor.bytes.is_empty() {
                    tensor.mmap_len as u64
                } else {
                    tensor.bytes.len() as u64
                };
                if bytes > 0 {
                    views.push((&tensor.metal_buf, bytes));
                }
            }
        }
        self.warm_up_buffers_gpu(&views, stride_bytes)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn warm_up_expert_pages_gpu(&self, _stride_bytes: u64) -> Result<u64> {
        Ok(0)
    }

    /// Phase F task #90 — dispatch `kernel_touch_u8_stride` over an
    /// arbitrary set of `(buffer, byte_len)` views in a single command
    /// buffer. Public surface so callers (and the smoke test) can warm
    /// any buffer set, not just `expert_weights`. See `warm_up_expert_pages_gpu`
    /// for the rationale.
    #[cfg(target_os = "macos")]
    pub fn warm_up_buffers_gpu(
        &self,
        views: &[(&metal::Buffer, u64)],
        stride_bytes: u64,
    ) -> Result<u64> {
        use metal::{MTLResourceOptions, MTLSize};
        if views.is_empty() {
            return Ok(0);
        }
        let stride: u64 = if stride_bytes == 0 { 1024 * 1024 } else { stride_bytes };
        let pipeline = self.inner.ensure_touch_u8_stride_pipeline()?;

        let mut total_touches: u64 = 0;
        let mut total_bytes: u64 = 0;
        for (_, bytes) in views {
            if *bytes == 0 {
                continue;
            }
            total_touches += (bytes + stride - 1) / stride;
            total_bytes += *bytes;
        }
        if total_touches == 0 {
            return Ok(0);
        }

        let dst = self
            .inner
            .device
            .new_buffer(total_touches, MTLResourceOptions::StorageModeShared);

        let cmd_buf = self.inner.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        let mut dst_offset: u64 = 0;
        for (buf, bytes) in views {
            if *bytes == 0 {
                continue;
            }
            let n = (bytes + stride - 1) / stride;
            enc.set_buffer(0, Some(buf), 0);
            enc.set_buffer(1, Some(&dst), 0);
            enc.set_bytes(2, std::mem::size_of::<u64>() as u64, &stride as *const u64 as *const _);
            enc.set_bytes(3, std::mem::size_of::<u64>() as u64, bytes as *const u64 as *const _);
            enc.set_bytes(
                4,
                std::mem::size_of::<u64>() as u64,
                &dst_offset as *const u64 as *const _,
            );
            let groups = MTLSize::new((n + 255) / 256, 1, 1);
            let tpg = MTLSize::new(256, 1, 1);
            enc.dispatch_thread_groups(groups, tpg);
            dst_offset += n;
        }
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        Ok(total_bytes)
    }

    // Non-macOS: no stub for `warm_up_buffers_gpu` — its only caller
    // (`warm_up_expert_pages_gpu`) is itself `#[cfg(target_os = "macos")]`,
    // and the signature would otherwise have to name the macOS-only
    // `metal::Buffer` type. The non-macOS `warm_up_expert_pages_gpu` stub
    // returns `Ok(0)` directly.

    /// Phase F task #89 — snapshot the number and size of MTLBuffer
    /// objects this dispatcher has materialized. Includes the expert
    /// weight no-copy views, the f32 weight upload cache, the persistent
    /// KV cache buffers, and the per-shape MoE scratch pools. Used to
    /// quantify our buffer-count vs antirez's 1-2 huge views pattern
    /// (`ds4_metal.m:413-472`).
    #[cfg(target_os = "macos")]
    pub fn buffer_audit(&self) -> BufferAuditReport {
        let mut r = BufferAuditReport::default();
        for qew in &self.inner.expert_weights {
            for tensor in [&qew.gate, &qew.up, &qew.down] {
                let bytes = if tensor.bytes.is_empty() {
                    tensor.mmap_len as u64
                } else {
                    tensor.bytes.len() as u64
                };
                r.n_expert_buffers += 1;
                r.expert_bytes_total += bytes;
            }
        }
        if let Ok(guard) = self.inner.weight_buf_cache.lock() {
            r.n_weight_cache_buffers = guard.len();
            for ((_ptr, byte_len), _buf) in guard.iter() {
                r.weight_cache_bytes_total += *byte_len as u64;
            }
        }
        if let Ok(guard) = self.inner.kv_cache_buffers.lock() {
            r.n_kv_cache_buffers = guard.len();
            for (_layer, buf) in guard.iter() {
                r.kv_cache_bytes_total += buf.length();
            }
        }
        if let Ok(guard) = self.inner.moe_scratch.lock() {
            r.n_moe_scratch_pools = guard.len();
            r.n_moe_scratch_buffers = guard.len() * 7;
        }
        use std::sync::atomic::Ordering;
        r.n_weight_no_copy = self.inner.weight_no_copy_count.load(Ordering::Relaxed);
        r.n_weight_memcpy = self.inner.weight_memcpy_count.load(Ordering::Relaxed);
        r.weight_no_copy_bytes = self.inner.weight_no_copy_bytes.load(Ordering::Relaxed);
        r.weight_memcpy_bytes = self.inner.weight_memcpy_bytes.load(Ordering::Relaxed);
        r
    }

    #[cfg(not(target_os = "macos"))]
    pub fn buffer_audit(&self) -> BufferAuditReport {
        BufferAuditReport::default()
    }

    /// Phase E M5.2.3: `flash_attn_decode` reading KV cache from the
    /// persistent per-layer buffer (paired with `kv_fp8_store_persistent`).
    ///
    /// The two ops together close the loop: kv_fp8_store_persistent
    /// WRITES into the persistent buffer; this method READS from it.
    /// No CPU intermediary needed. M5.4's unified-cb decoder calls
    /// both via this surface.
    ///
    /// Returns the per-head attention output (n_head × head_dim) — the
    /// same shape `flash_attn_decode_metal` already returns. The
    /// caller-provided `q` and `attn_sinks` remain CPU slices.
    /// Eventually those may also become persistent buffers; for now
    /// they're per-call and small (~2 KB and ~128 B respectively).
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_metal_persistent(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        q: &[f32],
        n_raw: u32,
        raw_cap: u32,
        kv_comp: Option<&[f32]>,
        n_comp: u32,
        comp_selected: Option<&[u32]>,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> ds4_engine::attn_dispatch::AttnHeadsOut {
        self.inner
            .flash_attn_decode_metal_persistent(
                layer_idx,
                params,
                q,
                n_raw,
                raw_cap,
                kv_comp,
                n_comp,
                comp_selected,
                n_selected,
                attn_sinks,
            )
            .expect("ds4_metal::flash_attn_decode_metal_persistent failed")
    }

    /// Phase E M5.2.2: `kv_fp8_store` against the persistent KV buffer.
    /// Writes the quantized row into slot `slot` of the layer's
    /// persistent buffer (from `kv_buffer_or_alloc`). The buffer is
    /// returned so callers can chain a read or a downstream op.
    ///
    /// The existing trait `kv_fp8_store` still delegates to the CPU
    /// oracle for now (lib.rs:341); switching it to this path is part
    /// of M5.2.3/4 once `flash_attn_decode_metal` is migrated to read
    /// from the persistent buffer too. Until then, this entry point is
    /// the API M5.4 (unified-cb decoder) will call.
    #[cfg(target_os = "macos")]
    pub fn kv_fp8_store_persistent(
        &self,
        layer_idx: u32,
        params: &ds4_engine::attn_dispatch::LayerParams,
        kv_row_f32: &[f32],
        raw_cap: u32,
        slot: u32,
    ) -> metal::Buffer {
        self.inner
            .kv_fp8_store_persistent_impl(layer_idx, params, kv_row_f32, raw_cap, slot)
            .expect("ds4_metal::kv_fp8_store_persistent encoding failed")
    }

    /// Overwrite the GPU persistent KV cache for `layer_idx` with `kv_storage`
    /// (`[raw_cap, n_lora_kv]` row-major, same layout the K-position verifier
    /// flash reads). The CPU decode path (decode_step) keeps KV in host
    /// `AttnStepState::kv_storage` and the K-position verifier reads the GPU
    /// persistent buffer — these were NOT synced, so the verifier attended over
    /// a zero prefix. Call this after prefill (and after any decode_step that
    /// advances the cache) to seed the verifier's KV with the real prefix.
    #[cfg(target_os = "macos")]
    pub fn populate_persistent_kv(&self, layer_idx: u32, kv_storage: &[f32]) {
        let byte_size = std::mem::size_of_val(kv_storage);
        let buf = self.inner.kv_buffer_or_alloc(layer_idx, byte_size);
        unsafe {
            std::ptr::copy_nonoverlapping(
                kv_storage.as_ptr(),
                buf.contents() as *mut f32,
                kv_storage.len(),
            );
        }
    }

    /// Read `n_slots` KV rows (each `row` f32) starting at `slot_start` from
    /// the GPU persistent KV buffer for `layer_idx`, into `dst` at the SAME
    /// slot offsets (`dst` is the CPU `[raw_cap, row]` kv_storage for the
    /// layer). This is the inverse of [`populate_persistent_kv`] for a slot
    /// range — used by spec-decode to pull the verifier's already-computed KV
    /// for ACCEPTED draft tokens back into host state, so accepted tokens
    /// need no base re-run (the verifier wrote slots base_slot..base_slot+K
    /// during verification via `kv_fp8_store_persistent_k`).
    ///
    /// The buffer is `StorageModeShared` f32 (same dtype/layout the verifier
    /// flash reads). The caller MUST ensure the verifier cb that wrote these
    /// slots has been committed+waited — `flush_and_read` on the verify scope
    /// does this before the accept decision, so the slots are current here.
    #[cfg(target_os = "macos")]
    pub fn read_persistent_kv_slots(
        &self,
        layer_idx: u32,
        raw_cap: u32,
        row: usize,
        slot_start: u32,
        n_slots: u32,
        dst: &mut [f32],
    ) {
        let byte_size = raw_cap as usize * row * std::mem::size_of::<f32>();
        let buf = self.inner.kv_buffer_or_alloc(layer_idx, byte_size);
        let start = slot_start as usize * row;
        let count = n_slots as usize * row;
        debug_assert!(
            start + count <= dst.len() && start + count <= raw_cap as usize * row,
            "read_persistent_kv_slots out of range: start={} count={} dst={} cap_rows={}",
            start, count, dst.len(), raw_cap as usize * row
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                (buf.contents() as *const f32).add(start),
                dst.as_mut_ptr().add(start),
                count,
            );
        }
    }

    /// DS4_VERIFY_COMPRESSOR sync primitives. The K-position verifier reads
    /// the GPU-resident compressor ring + state pools, but the CPU decode path
    /// (decode_step, prefill) keeps the compressor's emitted rows + sliding
    /// window in host `AttnStepState` (comp_kv_ring / comp_state_kv /
    /// comp_state_score, and the indexer analogues). Without syncing, the
    /// verifier's compressor would start from a ZERO ring + zero window — the
    /// same class of bug as the KV zero-prefix. These copy the prefill-built
    /// CPU state into the GPU buffers so the verifier attends the real
    /// long-range context.

    /// Upload the CPU compressor ring (`comp_kv_ring[layer]`, `[n_comp, head_dim]`
    /// row-major f32) into the GPU `comp_ring` buffer for `layer_idx`. The ring
    /// is sized `[raw_cap, head_dim]`; only the first `comp_rows.len()` floats
    /// (= n_comp * head_dim) are written. Pass `ring_cap_rows` so the buffer is
    /// allocated to the same capacity the flash gather expects.
    #[cfg(target_os = "macos")]
    pub fn populate_comp_ring(
        &self,
        layer_idx: u32,
        comp_rows: &[f32],
        ring_cap_rows: usize,
        head_dim: usize,
    ) {
        let byte_size = ring_cap_rows * head_dim * std::mem::size_of::<f32>();
        let buf = self.inner.comp_ring_or_alloc(layer_idx, byte_size);
        if comp_rows.is_empty() {
            return;
        }
        debug_assert!(comp_rows.len() <= ring_cap_rows * head_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                comp_rows.as_ptr(),
                buf.contents() as *mut f32,
                comp_rows.len(),
            );
        }
    }

    /// Upload the CPU compressor sliding-window state (`comp_state_kv[layer]` +
    /// `comp_state_score[layer]`, each `[rows, width]` row-major f32) into the
    /// GPU compressor state pools for `layer_idx`. With `is_indexer=true`,
    /// targets the indexer's separate pools instead. Needed for per-draft emit
    /// (the verifier continues the sliding window from the prefill state).
    #[cfg(target_os = "macos")]
    pub fn populate_compressor_state(
        &self,
        layer_idx: u32,
        state_kv: &[f32],
        state_score: &[f32],
        is_indexer: bool,
    ) {
        debug_assert_eq!(state_kv.len(), state_score.len());
        let bytes = std::mem::size_of_val(state_kv);
        let (kv_buf, sc_buf) = if is_indexer {
            (
                self.inner.indexer_state_kv_or_alloc(layer_idx, bytes),
                self.inner.indexer_state_score_or_alloc(layer_idx, bytes),
            )
        } else {
            (
                self.inner.compressor_state_kv_or_alloc(layer_idx, bytes),
                self.inner.compressor_state_score_or_alloc(layer_idx, bytes),
            )
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                state_kv.as_ptr(), kv_buf.contents() as *mut f32, state_kv.len(),
            );
            std::ptr::copy_nonoverlapping(
                state_score.as_ptr(), sc_buf.contents() as *mut f32, state_score.len(),
            );
        }
    }

    /// Read the GPU `comp_ring` for `layer_idx` back into `dst` (`[n_rows,
    /// head_dim]` row-major f32). Inverse of [`populate_comp_ring`] — used after
    /// speculative K verification to recover the rows the verifier emitted for
    /// the ACCEPTED drafts (so the CPU mirror stays consistent for the next
    /// decode_step). Caller must ensure the verify cb has been committed+waited.
    #[cfg(target_os = "macos")]
    pub fn read_comp_ring(
        &self,
        layer_idx: u32,
        ring_cap_rows: usize,
        head_dim: usize,
        n_rows: usize,
        dst: &mut [f32],
    ) {
        let byte_size = ring_cap_rows * head_dim * std::mem::size_of::<f32>();
        let buf = self.inner.comp_ring_or_alloc(layer_idx, byte_size);
        let count = n_rows * head_dim;
        debug_assert!(count <= dst.len() && count <= ring_cap_rows * head_dim);
        if count == 0 {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const f32,
                dst.as_mut_ptr(),
                count,
            );
        }
    }

    /// Phase E M5.4.5 entry point: decode one token via the unified-cb
    /// encoder.
    ///
    /// Routes through `SingleBufferEncoder::decode_token_via_first_half`,
    /// which runs `encode_first_half(l_split=n_layers)` end-to-end + the
    /// `output_hc_head_one` + `tail_lm_head_batched` tail. Replaces the
    /// trait-dispatch `decode_step_with_attn_to_residual` for the
    /// per-layer loop while keeping the same external contract (same
    /// state mutations, same logit shape).
    ///
    /// Returns logits over the vocabulary. `sample_argmax` is the
    /// caller's responsibility.
    #[cfg(target_os = "macos")]
    pub fn decode_token_unified(
        &self,
        x: Vec<f32>,
        model: &ds4_engine::decode_step::ComposedModelWeights,
        state: &mut ds4_engine::decode_step::AttnStepState,
        raw_cap: u32,
    ) -> anyhow::Result<Vec<f32>> {
        let encoder = crate::single_buffer_encoder::SingleBufferEncoder::new(self, raw_cap);
        encoder.decode_token_via_first_half(&x, model, state)
    }

    /// Phase E M5.2: get-or-allocate the persistent KV cache buffer for
    /// the given `layer_idx`. First call allocates a fresh
    /// `StorageModeShared` `metal::Buffer` of `byte_size` bytes; later
    /// calls return a clone of the same buffer (debug-asserts the size
    /// doesn't change). The buffer lives for the lifetime of
    /// `MetalDispatcher`.
    ///
    /// This is the foundation for M5.2's KV-cache migration: once
    /// `kv_fp8_store_impl` is rewritten to write into this persistent
    /// buffer (instead of recreating one each call from
    /// `KvCacheView::raw`), the GPU stops paying the
    /// `new_buffer_with_data + read_buffer` round-trip per
    /// kv_fp8_store call, unblocking fused-cb decode for the attn
    /// prefix.
    #[cfg(target_os = "macos")]
    pub fn kv_buffer_or_alloc(&self, layer_idx: u32, byte_size: usize) -> metal::Buffer {
        self.inner.kv_buffer_or_alloc(layer_idx, byte_size)
    }

    /// M5 Phase D — persistent per-layer compressor `state_kv` buffer.
    /// See [`crate::deferred::BatchScope::compressor_store_one`].
    #[cfg(target_os = "macos")]
    pub fn compressor_state_kv_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        self.inner.compressor_state_kv_or_alloc(layer_idx, byte_size)
    }

    /// M5 Phase D — persistent per-layer compressor `state_score` buffer.
    #[cfg(target_os = "macos")]
    pub fn compressor_state_score_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        self.inner.compressor_state_score_or_alloc(layer_idx, byte_size)
    }

    /// M5 Phase D — persistent per-layer indexer `state_kv` buffer
    /// (ratio==4 layers' separate compressor for indexer scoring).
    #[cfg(target_os = "macos")]
    pub fn indexer_state_kv_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        self.inner.indexer_state_kv_or_alloc(layer_idx, byte_size)
    }

    /// M5 Phase D — persistent per-layer indexer `state_score` buffer.
    #[cfg(target_os = "macos")]
    pub fn indexer_state_score_or_alloc(
        &self,
        layer_idx: u32,
        byte_size: usize,
    ) -> metal::Buffer {
        self.inner.indexer_state_score_or_alloc(layer_idx, byte_size)
    }

    /// Single-cb step 9 — GPU-resident compressed-KV ring (main compressor).
    #[cfg(target_os = "macos")]
    pub fn comp_ring_or_alloc(&self, layer_idx: u32, byte_size: usize) -> metal::Buffer {
        self.inner.comp_ring_or_alloc(layer_idx, byte_size)
    }

    /// Suppress per-layer flash-scratch REUSE (each flash allocates fresh
    /// scratch) for the duration of a chunk-prefill Phase-B loop. Required for
    /// async cb chaining: reused scratch aliases across the K per-position
    /// flashes when there is no per-layer GPU wait. See
    /// `MetalState::flash_scratch_suppress`. No-op off macOS.
    #[cfg(target_os = "macos")]
    pub fn set_flash_scratch_suppress(&self, v: bool) {
        self.inner
            .flash_scratch_suppress
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_flash_scratch_suppress(&self, _v: bool) {}

    /// Suppress the pooled MoE scratch (fresh per-call buffers) for the duration
    /// of an async chunk-prefill Phase-B loop. Required because the pool is shared
    /// across all MoE layers and relies on a per-call GPU wait that async removes.
    /// See `MetalState::moe_scratch_suppress`. No-op off macOS.
    #[cfg(target_os = "macos")]
    pub fn set_moe_scratch_suppress(&self, v: bool) {
        self.inner
            .moe_scratch_suppress
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_moe_scratch_suppress(&self, _v: bool) {}

    /// TEST-ONLY (Lever A K-flash isolation): see `MetalState::debug_flash_k`.
    #[cfg(target_os = "macos")]
    pub fn debug_flash_k(
        &self, params: &ds4_engine::attn_dispatch::LayerParams, q: &[f32], ws: &[u16],
        mask: &[u16], sinks: &[f32], k: usize, n_total: u32,
    ) -> Vec<f32> {
        self.inner.debug_flash_k(params, q, ws, mask, sinks, k, n_total)
    }

    /// TEST-ONLY (Lever A K-flash reference): see `MetalState::debug_flash_1`.
    #[cfg(target_os = "macos")]
    pub fn debug_flash_1(
        &self, params: &ds4_engine::attn_dispatch::LayerParams, q: &[f32], ws: &[u16],
        n_raw: u32, n_raw_valid: u32, sinks: &[f32],
    ) -> Vec<f32> {
        self.inner.debug_flash_1(params, q, ws, n_raw, n_raw_valid, sinks)
    }

    #[cfg(target_os = "macos")]
    pub fn index_comp_ring_or_alloc(&self, layer_idx: u32, byte_size: usize) -> metal::Buffer {
        self.inner.index_comp_ring_or_alloc(layer_idx, byte_size)
    }

    /// Re-init the persistent compressor/indexer state pools for a fresh decode
    /// sequence (call at pos==0). See `MetalState::reset_decode_state_pools`.
    #[cfg(target_os = "macos")]
    pub fn reset_decode_state_pools(&self) {
        self.inner.reset_decode_state_pools();
    }

    /// Per-head RMS norm (M4 #330e). Mirrors decode_step.rs:656-663 with f32
    /// accumulator. Not part of `KernelDispatcher` because the existing decode
    /// path runs this inline on CPU; this entry point feeds the future
    /// single-buffer encoder (M4 #330h) which needs everything on GPU.
    ///
    /// Layout: `x` is `[n_head * head_dim]`. Output is same length.
    #[cfg(target_os = "macos")]
    pub fn head_rms_norm(
        &self,
        x: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        self.inner.head_rms_norm(x, n_head, head_dim, eps)
    }

    /// Per-32-elt-block activation quantization round-trip (M4 #330f).
    /// Byte-exact against `ds4_engine::forward::q8_0_round_trip`. Not part
    /// of `KernelDispatcher`; feeds the future single-buffer encoder
    /// (M4 #330h) which needs everything on GPU.
    #[cfg(target_os = "macos")]
    pub fn q8_0_round_trip(&self, x: &[f32]) -> Vec<f32> {
        self.inner.q8_0_round_trip(x)
    }

    /// Elementwise silu on Metal (M4 #330i). `fidelity=true` selects the
    /// antirez `sigmoid_stable`-branched kernel matching
    /// `DS4_SILU_FIDELITY=1`; `false` selects the positive-branch identity.
    /// Byte-exact against `ds4_engine::forward::silu` at the matching
    /// gate setting.
    #[cfg(target_os = "macos")]
    pub fn silu(&self, x: &[f32], fidelity: bool) -> Vec<f32> {
        self.inner.silu(x, fidelity)
    }

    /// Elementwise sigmoid on Metal (M4 #330j). `fidelity=true` selects
    /// the antirez `sigmoid_stable`-branched kernel; `false` selects the
    /// positive-branch identity. Used by `output_hc_head_one`
    /// (M4 #315) — same env (`DS4_SILU_FIDELITY`) governs both sites.
    #[cfg(target_os = "macos")]
    pub fn sigmoid(&self, x: &[f32], fidelity: bool) -> Vec<f32> {
        self.inner.sigmoid(x, fidelity)
    }

    /// Elementwise softplus_sqrt on Metal (M4 #330k). `fidelity=true`
    /// selects the antirez ds4.c:4867 piecewise kernel
    /// (DS4_MOE_ROUTER_FIDELITY gate); `false` selects the stable
    /// softplus identity. Distinct entry point from the legacy
    /// `softplus_sqrt` which only exposes the default kernel.
    #[cfg(target_os = "macos")]
    pub fn softplus_sqrt_fidelity(&self, logits: &[f32], fidelity: bool) -> Vec<f32> {
        self.inner.softplus_sqrt_fidelity(logits, fidelity)
    }

    /// Phase C tail slice (M4 #330m). Batch `rms_norm(γ)` →
    /// (optional `q8_0_round_trip`) → `matvec_f32(lm_head)` into ONE
    /// `MTLCommandBuffer` with a single readback. Bit-identical to
    /// running the three ops sequentially through this dispatcher.
    /// Used only by `SingleBufferEncoder::decode_token_batched`
    /// (Phase C+); the trait path keeps calling rms_norm/q8_0/matvec
    /// individually so the 18 fidelity gates and `--correctness` mode
    /// stay bit-for-bit unchanged.
    #[cfg(target_os = "macos")]
    pub fn tail_lm_head_batched(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        want_q80: bool,
        lm_head: &[f32],
        vocab_size: usize,
    ) -> Vec<f32> {
        self.inner
            .tail_lm_head_batched(x, gamma, eps, want_q80, lm_head, vocab_size)
    }

    /// Greedy-decode tail: same lm_head path as `tail_lm_head_batched` but
    /// returns just the GPU-argmax token id (no full-logit readback). See
    /// `MetalState::tail_lm_head_argmax`.
    #[cfg(target_os = "macos")]
    pub fn tail_lm_head_argmax(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        want_q80: bool,
        lm_head: &[f32],
        vocab_size: usize,
    ) -> i32 {
        self.inner
            .tail_lm_head_argmax(x, gamma, eps, want_q80, lm_head, vocab_size)
    }

    /// Q8_0 greedy-decode tail — reads the raw q8 `output.weight` bytes
    /// (~562 MB) instead of the dequantized f32 (~2.1 GB). Token-identical to
    /// `tail_lm_head_argmax`. See `MetalState::tail_lm_head_argmax_q8`.
    #[cfg(target_os = "macos")]
    pub fn tail_lm_head_argmax_q8(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q8: &[u8],
        vocab_size: usize,
    ) -> i32 {
        self.inner
            .tail_lm_head_argmax_q8(x, gamma, eps, lm_head_q8, vocab_size)
            .expect("ds4_metal::tail_lm_head_argmax_q8 encoding failed")
    }

    /// Full-logits Q8_0 lm-head tail: `rms_norm → mul_mv_q8_0(lm_head_q8) →`
    /// all `vocab_size` logits. Lets the sampling decode path drop the 2.1 GB
    /// f32 lm_head; bit-identical to the f32 full-logits tail (want_q80=false).
    #[cfg(target_os = "macos")]
    pub fn tail_lm_head_full_q8(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q8: &[u8],
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        self.inner
            .tail_lm_head_full_q8(x, gamma, eps, lm_head_q8, vocab_size)
    }

    /// Q4_0 greedy-decode tail (DS4_LOW_RAM + DS4_Q4_LMHEAD) — reads the
    /// re-quantized `block_q4_0` lm_head (~281 MB) at 4-bit. See
    /// `MetalState::tail_lm_head_argmax_q4`.
    #[cfg(target_os = "macos")]
    pub fn tail_lm_head_argmax_q4(
        &self,
        x: &[f32],
        gamma: &[f32],
        eps: f32,
        lm_head_q4: &[u8],
        vocab_size: usize,
    ) -> i32 {
        self.inner
            .tail_lm_head_argmax_q4(x, gamma, eps, lm_head_q4, vocab_size)
            .expect("ds4_metal::tail_lm_head_argmax_q4 encoding failed")
    }

    /// Phase C.3a layer-head slice (M4 #330o). Batch `matvec_f32(w_qa)` →
    /// `rms_norm(gamma_q)` for the per-layer Q-projection head into ONE
    /// `MTLCommandBuffer` with a single readback. Bit-identical to running
    /// the two ops sequentially through this dispatcher. The intermediate
    /// `qr` tensor stays resident in a GPU buffer and is never read back.
    #[cfg(target_os = "macos")]
    pub fn layer_qa_rms_batched(
        &self,
        x: &[f32],
        w_qa: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.inner
            .layer_qa_rms_batched(x, w_qa, gamma_q, n_lora_q, eps_rms)
    }

    /// Phase C.3b post-attention chain slice (M4 #330o). Batch
    /// `matvec_f32(w_gate) → matvec_f32(w_up) → swiglu → (optional
    /// q8_0_round_trip) → matvec_f32(w_down)` into ONE
    /// `MTLCommandBuffer` with a single readback. Saves 3-4
    /// commit+wait+readbacks per layer relative to the trait path.
    ///
    /// `want_q80` mirrors `DS4_Q8_0_ACT=1`. The SwiGLU pass uses
    /// the default-branch silu identity; with `DS4_SILU_FIDELITY=1`
    /// callers must fall back to the trait-path sequential chain.
    #[cfg(target_os = "macos")]
    pub fn shared_chain_batched(
        &self,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Vec<f32> {
        self.inner
            .shared_chain_batched(ffn_norm, w_gate, w_up, w_down, shared_dim, want_q80)
    }

    /// Phase C.3c output-head slice (M4 #330o). Batch `rms_norm(unit gamma)`
    /// on the full `hc_dim = n_hc * d_embd` HC residual into the intermediate
    /// `flat`, then `matvec_f32(fn_w, flat, n_hc)` into ONE
    /// `MTLCommandBuffer` with a single readback. Returns `pre[n_hc]`;
    /// callers run the sigmoid + weighted-sum post-amble on CPU (cheap;
    /// `n_hc * d_embd` mults at `n_hc = 8`).
    ///
    /// Bit-identical to running `rms_norm(inp_hc, &[1.0; hc_dim], eps_rms)`
    /// then `matvec_f32(fn_w, &flat, n_hc)` sequentially through this
    /// dispatcher. Saves one commit+wait+readback per token.
    #[cfg(target_os = "macos")]
    pub fn output_hc_head_batched(
        &self,
        inp_hc: &[f32],
        fn_w: &[f32],
        n_hc: usize,
        d_embd: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.inner
            .output_hc_head_batched(inp_hc, fn_w, n_hc, d_embd, eps_rms)
    }

    /// M4 #330o Phase C.3d.2 — packs `matvec(w_q_b) + head_rms_norm +
    /// matvec(w_kv)` into one `MTLCommandBuffer`. Returns
    /// `(q_heads, kv_raw_row)`. Bit-identical to sequential trait path
    /// under default-OFF `DS4_HEAD_RMS_F64_FIDELITY=0`. Saves 2
    /// commit+wait+readback per layer per token (one cwr instead of
    /// three).
    #[cfg(target_os = "macos")]
    pub fn qkv_b_head_rms_batched(
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
        self.inner.qkv_b_head_rms_batched(
            qr_normed_q, w_q_b, q_dim, n_head, head_dim, eps_rms,
            normed_kv, w_kv, kv_row,
        )
    }

    /// Phase C.3e (M4 #330o). Pack the per-layer MoE router head
    /// `matvec_f32(w_router, h_norm, n_experts) + softplus_sqrt` into ONE
    /// `MTLCommandBuffer` (one readback). Saves 1 cwr/layer = 43 cwr/token.
    /// Uses the same `ds4_kernel_mul_mv_f32_f32_4` +
    /// `ds4_kernel_dsv4_softplus_sqrt_f32_4` kernels as the trait methods,
    /// so this is bit-identical to the sequential path.
    #[cfg(target_os = "macos")]
    pub fn router_logits_batched(
        &self,
        w_router: &[f32],
        h_norm: &[f32],
        n_experts: usize,
    ) -> Vec<f32> {
        self.inner.router_logits_batched(w_router, h_norm, n_experts)
    }
}

// ---------------------------------------------------------------------------
// KernelDispatcher impl — macOS only. On Linux we still build but the
// impl is omitted (the struct is not instantiable anyway).
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
impl ds4_engine::attn_dispatch::AttentionDispatcher for MetalDispatcher {
    fn hc_collapse_norm(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        kind: ds4_engine::attn_dispatch::HcKind,
        hc_fn: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        prev_hc: &[f32],
        use_gamma: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner
            .hc_collapse_norm_impl(params, kind, hc_fn, hc_scale, hc_base, prev_hc, use_gamma)
            .expect("ds4_metal::hc_collapse_norm encoding failed")
    }

    fn qkv_rms_norm_rows(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        qr: &[f32],
        kv_raw: &[f32],
        gamma_q: &[f32],
        gamma_kv: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        self.inner
            .qkv_rms_norm_rows_impl(params, qr, kv_raw, gamma_q, gamma_kv)
            .expect("ds4_metal::qkv_rms_norm_rows encoding failed")
    }

    fn rope_tail(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        x: &mut [f32],
        pos: u32,
        backward: bool,
    ) {
        self.inner
            .rope_tail_impl(params, x, pos, backward)
            .expect("ds4_metal::rope_tail encoding failed");
    }

    /// Phase C-B Slice 4 (M4 #330p) override: pack `kv_rms_norm_row` +
    /// `rope_tail(KV tail)` into ONE `MTLCommandBuffer`. Saves 1 commit+
    /// wait per layer per token vs the trait default's sequential calls.
    /// Bit-identity gated by
    /// `tests/kv_norm_rope_batched_smoke.rs`.
    fn kv_norm_rope_batched(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        kv_raw_row: &[f32],
        qkv_gamma_kv: &[f32],
        pos: u32,
    ) -> Vec<f32> {
        // Inherent method is `kv_norm_rope_chain` to avoid colliding with
        // this trait method's name; both surface the same fused encoder.
        self.kv_norm_rope_chain(kv_raw_row, qkv_gamma_kv, params, pos, params.rms_eps)
            .expect("ds4_metal::kv_norm_rope_batched encoding failed")
    }

    fn kv_fp8_store(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        kv_row_f32: &[f32],
        view: &mut ds4_engine::attn_dispatch::KvCacheView<'_>,
    ) {
        // Phase E M5.2.4: unified path — match the GPU shim
        // (`ds4_dsv4_kv_fp8_store`) byte-for-byte. The shim does:
        //   1. per-64-block max-scaled FP8-E4M3FN on the NOPE prefix
        //      (`head_dim - n_rot` floats at row start),
        //   2. IEEE-754 f16 round-trip on the whole row.
        //
        // The earlier override delegated to `CpuAttentionDispatcher`,
        // which env-gated the FP8 step behind `DS4_FP8_KV_QUANT=1`.
        // Production decode (antirez `ds4.c:7608-7609`) always applies
        // both quantizations; the env gate was a Phase B opt-in. We
        // now apply both unconditionally so both the trait-dispatch
        // KV path (writes `view.raw`) and the encode_first_half path
        // (writes the persistent buffer) end up with byte-identical
        // bytes for production shape `n_lora_kv == head_dim`.
        //
        // Asymmetric shapes (`n_lora_kv != head_dim`) are not used by
        // production decode and are no longer matched by the trait —
        // see `tests/attn_smoke.rs` migration in this commit.
        let row = params.n_lora_kv as usize;
        debug_assert_eq!(kv_row_f32.len(), row);
        let slot = (view.pos % view.raw_cap) as usize;
        let off = slot * row;
        debug_assert!(off + row <= view.raw.len());
        let dest = &mut view.raw[off..off + row];
        dest.copy_from_slice(kv_row_f32);
        let head_dim_u = params.head_dim as usize;
        let n_rot_u = params.n_rot as usize;
        if head_dim_u > n_rot_u {
            ds4_engine::attn_dispatch::ds4_fp8_kv_quantize_row_inplace(
                dest, head_dim_u, n_rot_u,
            );
        }
        for d in dest.iter_mut() {
            *d = ds4_engine::attn_dispatch::f16_round_trip_f32(*d);
        }
        view.pos = view.pos.saturating_add(1);
    }

    fn flash_attn_decode(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
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
    ) -> ds4_engine::attn_dispatch::AttnHeadsOut {
        self.inner
            .flash_attn_decode_impl(
                params,
                q,
                kv_raw,
                n_raw,
                raw_cap,
                raw_start,
                kv_comp,
                n_comp,
                comp_selected,
                n_selected,
                attn_sinks,
            )
            .expect("ds4_metal::flash_attn_decode encoding failed")
    }

    fn attn_output_proj(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        heads: &[f32],
        w_o_a: &[f32],
        w_o_b: &[f32],
        cur_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        self.inner
            .attn_output_proj_impl(
                params,
                heads,
                w_o_a,
                w_o_b,
                cur_hc,
                hc_split_post,
                hc_split_comb,
            )
            .expect("ds4_metal::attn_output_proj encoding failed")
    }

    fn shared_expert(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        shared_dim: u32,
        clamp: f32,
    ) -> Vec<f32> {
        self.inner
            .shared_expert_impl(params, ffn_norm, w_gate, w_up, shared_dim, clamp)
            .expect("ds4_metal::shared_expert encoding failed")
    }

    fn shared_down_hc_expand_add(
        &self,
        params: &ds4_engine::attn_dispatch::LayerParams,
        shared_mid: &[f32],
        w_down: &[f32],
        routed_out: &[f32],
        after_attn_hc: &[f32],
        hc_split: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        self.inner
            .shared_down_hc_expand_add_impl(
                params,
                shared_mid,
                w_down,
                routed_out,
                after_attn_hc,
                hc_split,
                hc_split_comb,
            )
            .expect("ds4_metal::shared_down_hc_expand_add encoding failed")
    }
}

#[cfg(target_os = "macos")]
impl ds4_engine::dispatch::KernelDispatcher for MetalDispatcher {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.inner.rms_norm(x, gamma, eps)
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        self.inner.matvec_f32(w, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.inner.softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        self.inner.router_finalize(probs, bias, k)
    }

    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        self.inner.moe_routed_step(
            layer_idx,
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        )
    }

    /// Phase C.3a (M4 #330o). Overrides the trait default to pack
    /// `matvec_f32(w_qa, x, n_lora_q) → rms_norm(qr, gamma_q, eps)` into ONE
    /// `MTLCommandBuffer` with a single readback. Bit-identical to the
    /// default sequential path (smoke-gated by
    /// `tests/layer_qa_rms_batched_smoke.rs`).
    fn layer_qa_rms_batched(
        &self,
        x: &[f32],
        w_qa: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.inner
            .layer_qa_rms_batched(x, w_qa, gamma_q, n_lora_q, eps_rms)
    }

    /// Phase C.3b (M4 #330o). Overrides the trait default to pack the
    /// shared-expert FFN body (`matvec(w_gate) → matvec(w_up) → swiglu →
    /// (optional q8_0_round_trip) → matvec(w_down)`) into ONE
    /// `MTLCommandBuffer` with a single readback. Bit-identical to the
    /// default sequential path under default-branch silu; smoke-gated by
    /// `tests/shared_chain_batched_smoke.rs`. Callers under
    /// `DS4_SILU_FIDELITY=1` must take the trait-default path instead
    /// (the override only knows the antirez `kernel_swiglu_f32` form).
    fn shared_chain_batched(
        &self,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Vec<f32> {
        self.inner
            .shared_chain_batched(ffn_norm, w_gate, w_up, w_down, shared_dim, want_q80)
    }

    /// C.3c override: pack `rms_norm(unit gamma) → matvec_f32(fn_w)` into
    /// ONE `MTLCommandBuffer`. Bit-identical to the default sequential
    /// path; smoke-gated by `tests/output_hc_head_batched_smoke.rs`. No
    /// fidelity branch needed — the sigmoid/weighted-sum post-amble stays
    /// on CPU in `output_hc_head_one`.
    fn output_hc_head_batched(
        &self,
        inp_hc: &[f32],
        fn_w: &[f32],
        n_hc: usize,
        d_embd: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        self.inner
            .output_hc_head_batched(inp_hc, fn_w, n_hc, d_embd, eps_rms)
    }

    /// C.3d.1 override: dispatch through the Metal `ds4_head_rms_norm_f32`
    /// pipeline (Phase A.1, M4 #330e). Matches the trait default's f32
    /// chain bit-for-bit on this backend; callers under
    /// `DS4_HEAD_RMS_F64_FIDELITY=1` must NOT take this path — the helper
    /// at `decode_step.rs` runs an inline f64 reduction in that branch.
    fn head_rms_norm(&self, x: &[f32], n_head: usize, head_dim: usize, eps: f32) -> Vec<f32> {
        self.inner.head_rms_norm(x, n_head, head_dim, eps)
    }

    /// C.3d.2 override: pack `matvec(attn_q_b) → head_rms_norm(unit γ) →
    /// matvec(attn_kv)` into ONE `MTLCommandBuffer` with TWO readbacks
    /// (`q_heads`, `kv_raw_row`). Saves 2 cwr/layer = 86 cwr/token vs the
    /// sequential trait path. Smoke-gated by
    /// `tests/qkv_b_head_rms_batched_smoke.rs`. Callers under
    /// `DS4_HEAD_RMS_F64_FIDELITY=1` must NOT take this path — the helper
    /// at `decode_step.rs` keeps the legacy inline f64 reduction and two
    /// separate matvecs in that branch.
    fn qkv_b_head_rms_batched(
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
        self.inner.qkv_b_head_rms_batched(
            qr_normed_q,
            w_q_b,
            q_dim,
            n_head,
            head_dim,
            eps_rms,
            normed_kv,
            w_kv,
            kv_row,
        )
    }

    /// C.3e override: pack `matvec_f32(w_router, h_norm, n_experts)` +
    /// `softplus_sqrt` into ONE `MTLCommandBuffer` with a single readback.
    /// Bit-identical to the trait default (same Metal kernels under the
    /// hood — `ds4_kernel_mul_mv_f32_f32_4` + `ds4_kernel_dsv4_softplus_
    /// sqrt_f32_4` — just merged into one command buffer). Smoke-gated
    /// by `tests/router_logits_batched_smoke.rs`.
    fn router_logits_batched(
        &self,
        w_router: &[f32],
        h_norm: &[f32],
        n_experts: usize,
    ) -> Vec<f32> {
        self.inner.router_logits_batched(w_router, h_norm, n_experts)
    }

    /// Phase C-B Slice 5-redo (M4 #330p) override: pack
    /// `moe_routed_step + shared_chain_batched` into ONE
    /// `MTLCommandBuffer` with one `commit` + one `wait_until_completed`.
    /// The trait default runs the two methods sequentially (2 cbs, 2
    /// waits); this override saves ~25-100 ms per layer per token
    /// (the per-cb wait latency dominated by Metal scheduling, NOT
    /// kernel compute — see `DS4_MOE_TRACE` profiling notes).
    fn moe_and_shared_chain_batched(
        &self,
        layer_idx: u32,
        moe_x: &[f32],
        moe_selected: &[usize],
        moe_weights: &[f32],
        _moe_w_gate: &[f32],
        _moe_w_up: &[f32],
        _moe_w_down: &[f32],
        d_ffn: usize,
        sh_ffn_norm: &[f32],
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        // moe weights come from `expert_weights` (loaded from GGUF via
        // load_expert_weights), not the trait-passed slices. The trait
        // signature carries them only for backend symmetry; Metal looks
        // them up by `layer_idx`. The underscore prefixes mark the
        // unused-by-Metal slice parameters.
        self.inner.moe_and_shared_chain_batched_inherent(
            layer_idx,
            moe_x,
            moe_selected,
            moe_weights,
            d_ffn,
            sh_ffn_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            want_q80,
        )
    }

    /// Phase C-B (M4 #330p) override: pack the 5-op attention-half QKV
    /// chain into ONE `MTLCommandBuffer`. Saves 1 commit+wait + 2
    /// readbacks vs running `layer_qa_rms_batched + qkv_b_head_rms_batched`
    /// sequentially (the trait default). Smoke-gated by
    /// `tests/attn_qkv_chain_batched_smoke.rs`.
    fn attn_qkv_chain_batched(
        &self,
        normed: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        attn_q_b: &[f32],
        n_head: usize,
        head_dim: usize,
        eps_rms: f32,
        attn_kv: &[f32],
        kv_row: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // The inherent method `Self::attn_qkv_chain_batched` (in
        // `deferred.rs`) returns `Result`; the trait method returns the
        // unwrapped triple. Encoding failures here mirror the panic
        // semantics of `Self::matvec_f32` (`"ds4_metal::matvec_f32
        // encoding failed"`) — all input shape constraints are documented
        // and enforced in the underlying impl.
        Self::attn_qkv_chain_batched(
            self, normed, attn_q_a, gamma_q, n_lora_q, attn_q_b, n_head, head_dim,
            eps_rms, attn_kv, kv_row,
        )
        .expect("ds4_metal::attn_qkv_chain_batched encoding failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn new_returns_err_on_linux() {
        match MetalDispatcher::new() {
            Ok(_) => panic!("Linux MetalDispatcher::new must error"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("macOS"), "unexpected error: {msg}");
            }
        }
    }

    #[test]
    fn lib_compiles_on_both_platforms() {
        // Compile-only smoke test: confirms the crate's public types
        // are usable from a downstream test on any platform.
        let _ = std::marker::PhantomData::<MetalDispatcher>;
    }

    /// Compile-time assertion that `MetalDispatcher` implements both
    /// `KernelDispatcher` and `AttentionDispatcher` on macOS. This test
    /// is a no-op at runtime; it only matters that the file compiles.
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_dispatcher_implements_both_traits() {
        fn assert_kd<T: ds4_engine::dispatch::KernelDispatcher>() {}
        fn assert_ad<T: ds4_engine::attn_dispatch::AttentionDispatcher>() {}
        assert_kd::<MetalDispatcher>();
        assert_ad::<MetalDispatcher>();
    }
}
