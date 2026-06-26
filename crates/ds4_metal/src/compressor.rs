//! M5 Phase D — GPU orchestrator port of antirez `compressor_decode_one`
//! (`attn_dispatch.rs:2058`).
//!
//! Drop-in replacement for the CPU function: same inputs, same
//! outputs, same external semantics. State buffers (`state_kv`,
//! `state_score`) remain caller-owned CPU slices — the GPU dispatch
//! uploads them, runs the pool kernel, and reads the pooled row back.
//! This is wasteful (~64 KB upload per emit at V4 Flash main dims)
//! but keeps the CPU/GPU state mirrors bit-equivalent so the unit
//! tests can compare against the CPU oracle row-for-row. A later
//! milestone migrates state to the persistent per-layer GPU pools
//! (`compressor_state_*_or_alloc`) and elides the readback.
//!
//! Per-call GPU work (when `should_compress == false`):
//!   1 BatchScope, 2 matvecs, 1 cb. Matvec FLOPs move off CPU.
//!
//! Per-call GPU work (on emit, `(pos+1) % ratio == 0`):
//!   2 BatchScopes (matvecs + pool), pool readback (~hd floats).
//!   Matvec FLOPs move off CPU; pool collapses on-GPU.
//!
//! The remaining steps (APE add, state row copy, RMS norm, rope_tail,
//! optional FP8, ratio==4 rotation) stay on CPU so this v1 stays small
//! and obviously correct. `rope_tail` is dispatched via the existing
//! `AttentionDispatcher::rope_tail` trait method (which is already
//! GPU on `MetalDispatcher`).

use ds4_engine::attn_dispatch::{
    ds4_fp8_kv_quantize_row_inplace, AttentionDispatcher, CompressorInputs, LayerParams,
};
use ds4_engine::forward::q8_0_round_trip;

use crate::deferred::{BatchScope, DeferredBuf};
use crate::MetalDispatcher;

/// `out = W · x` for a compressor projection: `matvec_f16` over the no-copy
/// F16 mmap bytes when present (lean server path), else `matvec_f32` over the
/// dequantized f32. Bit-identical (F16→f32 is exact) — see the
/// `matvec_f16_matches_matvec_f32_oracle` test. `w_f16` is `Some` exactly when
/// the lean weights skipped the f32 dequant.
fn comp_matvec(
    scope: &BatchScope<'_>,
    w_f32: &[f32],
    w_f16: Option<&[u8]>,
    x_db: &DeferredBuf,
    in_dim: usize,
    width_u: usize,
) -> anyhow::Result<DeferredBuf> {
    if let Some(bytes) = w_f16 {
        let w = scope.weight_f16(bytes);
        scope.matvec_f16(&w, x_db, in_dim, width_u)
    } else {
        let w = scope.weight_f32(w_f32);
        scope.matvec_f32(&w, x_db, in_dim, width_u)
    }
}

/// K-batched [`comp_matvec`]: project ALL K chunk positions in ONE matmul.
/// `x_k` is `[K, in_dim]`; output is `[K, width_u]` row-major. Row r is
/// byte-identical to `comp_matvec(x_k[r])` (same ne11-aware kernel). Used by the
/// chunk-prefill cores to batch the compressor/indexer projections.
pub fn comp_proj_k(
    scope: &BatchScope<'_>,
    w_f32: &[f32],
    w_f16: Option<&[u8]>,
    x_k: &DeferredBuf,
    in_dim: usize,
    width_u: usize,
    k: usize,
) -> anyhow::Result<DeferredBuf> {
    // DS4_COMP_MM (default on): weight-stationary tiled matmul (matmul_k_f16)
    // instead of the per-token mul_mv (matmul_f16_k), which re-streams the whole
    // [in_dim × width] f16 weight ONCE PER TOKEN. Same fix as the indexer
    // projections (DS4_IDX_MM). PRECISION: f16 multiply casts the activation to
    // half (vs f32-accumulate matvec) — the compressor output feeds the compressed
    // KV cache → attention, so this is higher-risk than the indexer; validate
    // coherence. Opt out with DS4_COMP_MM=0.
    let comp_mm = std::env::var("DS4_COMP_MM").ok().as_deref() != Some("0");
    if let Some(bytes) = w_f16 {
        let w = scope.weight_f16(bytes);
        if comp_mm && in_dim % 8 == 0 {
            scope.matmul_k_f16(&w, x_k, in_dim, width_u, k)
        } else {
            scope.matmul_f16_k(&w, x_k, in_dim, width_u, k)
        }
    } else {
        let w = scope.weight_f32(w_f32);
        scope.matmul_f32_k(&w, x_k, in_dim, width_u, k)
    }
}

/// GPU port of antirez `compressor_decode_one` (`attn_dispatch.rs:2058`).
/// See the module docs for the contract and the parts that remain on
/// CPU.
///
/// Returns `Some(pooled_row)` (head_dim floats) when this position
/// emits a compressed row (every `compress_ratio` positions), else
/// `None`. Updates `state_kv` / `state_score` in place; on emit and
/// `ratio == 4`, also rotates the state's two-window halves so the
/// next quad re-uses the just-finished window as its prev-window
/// context.
pub fn compressor_decode_one_metal(
    dispatcher: &MetalDispatcher,
    params: &LayerParams,
    comp: &CompressorInputs<'_>,
    x: &[f32],
    state_kv: &mut [f32],
    state_score: &mut [f32],
    pos: u32,
    layer_idx: u32,
    is_indexer: bool,
) -> Option<Vec<f32>> {
    // Keep compressor state in persistent per-layer GPU pools, eliding the
    // full-state upload per emit and (with FUSE) the matvec readback. The
    // CPU mirror is kept in lock-step so unit tests still compare against the
    // oracle. Default-on (DS4_COMPRESSOR_POOL=0 reverts).
    let use_pool = std::env::var("DS4_COMPRESSOR_POOL").ok().as_deref() != Some("0");
    let ratio = comp.compress_ratio;
    let head_dim = comp.head_dim;
    let coff = if ratio == 4 { 2u32 } else { 1u32 } as usize;
    let width_u = coff * head_dim as usize;
    let pos_mod = (pos % ratio) as usize;
    let row = if ratio == 4 {
        ratio as usize + pos_mod
    } else {
        pos_mod
    };
    let should_compress = ((pos + 1) % ratio) == 0;

    let in_dim = x.len();
    // f32 length checks only when the f32 path is in use; lean mode empties the
    // f32 and drives the matvec from `comp.w_kv_f16` instead.
    if comp.w_kv_f16.is_none() {
        debug_assert_eq!(comp.w_kv.len(), in_dim * width_u);
        debug_assert_eq!(comp.w_gate.len(), in_dim * width_u);
    }

    // Q8_0 round-trip is gated on DS4_Q8_0_ACT, matching the CPU path
    // (attn_dispatch.rs:2088). Off by default.
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let x_owned;
    let x_in: &[f32] = if want_q80 {
        x_owned = q8_0_round_trip(x);
        &x_owned
    } else {
        x
    };

    // ── GPU: kv_cur = x @ w_kv, sc_cur = x @ w_gate ──────────────
    // `scope.matvec_f32` requires d_in % 4 == 0 and d_out % 2 == 0.
    // For V4 Flash main (in_dim=DS4_N_LORA_KV=512, width=2*512=1024)
    // and indexer (in_dim=DS4_N_LORA_KV=512, width=2*128=256) both
    // hold. Fall back to CPU when they don't (rare; covers any
    // hypothetical layer config where in_dim is not f4-aligned).
    let matvec_ok = in_dim % 4 == 0 && width_u % 2 == 0;
    let fuse = use_pool && matvec_ok && ratio == 4
        && std::env::var("DS4_COMPRESSOR_FUSE").ok().as_deref() != Some("0");
    if fuse {
        // One cb: matvec(kv,sc) → store_db(+APE) into resident pools →
        // pool_ratio4 → read pooled. No matvec readback, no full-state
        // upload. Pools resync the CPU mirror for rotation + tests.
        let bytes = state_kv.len() * std::mem::size_of::<f32>();
        let (pk, ps) = if is_indexer {
            (dispatcher.indexer_state_kv_or_alloc(layer_idx, bytes),
             dispatcher.indexer_state_score_or_alloc(layer_idx, bytes))
        } else {
            (dispatcher.compressor_state_kv_or_alloc(layer_idx, bytes),
             dispatcher.compressor_state_score_or_alloc(layer_idx, bytes))
        };
        // store_one kernel reads APE [ratio, width]; comp.w_ape is
        // [width, ratio] — transpose into kernel layout.
        let mut ape = vec![0.0f32; ratio as usize * width_u];
        for j in 0..width_u {
            for r in 0..ratio as usize {
                ape[r * width_u + j] = comp.w_ape[j * ratio as usize + r];
            }
        }
        let ape_bytes =
            unsafe { std::slice::from_raw_parts(ape.as_ptr() as *const u8, ape.len() * 4) };
        let scope = dispatcher.batch_scope();
        let x_db = scope.upload_f32(x_in);
        let kv_db = comp_matvec(&scope, comp.w_kv, comp.w_kv_f16, &x_db, in_dim, width_u)
            .expect("fuse matvec kv");
        let sc_db = comp_matvec(&scope, comp.w_gate, comp.w_gate_f16, &x_db, in_dim, width_u)
            .expect("fuse matvec sc");
        scope.compressor_store_one_db(&kv_db, &sc_db, ape_bytes, &pk, &ps,
            width_u as u32, ratio, pos, false).expect("fuse store");
        let emit = if should_compress {
            let pooled_db = scope.compressor_pool_ratio4(&pk, &ps, head_dim).expect("fuse pool");
            let pooled = scope.flush_and_read(&pooled_db);
            sync_pool_to_mirror(&pk, state_kv);
            sync_pool_to_mirror(&ps, state_score);
            Some(finish_emit(dispatcher, params, comp, pooled, state_kv, state_score,
                width_u, ratio, pos, Some(&pk), Some(&ps), None))
        } else {
            scope.wait_all_and_drop();
            sync_pool_to_mirror(&pk, state_kv);
            sync_pool_to_mirror(&ps, state_score);
            None
        };
        return emit;
    }
    let (kv_cur, mut sc_cur) = if matvec_ok {
        let scope = dispatcher.batch_scope();
        let x_db = scope.upload_f32(x_in);
        let kv_db = comp_matvec(&scope, comp.w_kv, comp.w_kv_f16, &x_db, in_dim, width_u)
            .expect("compressor matvec (w_kv)");
        let sc_db = comp_matvec(&scope, comp.w_gate, comp.w_gate_f16, &x_db, in_dim, width_u)
            .expect("compressor matvec (w_gate)");
        let outs = scope.flush_and_read_multi(&[&kv_db, &sc_db]);
        (outs[0].clone(), outs[1].clone())
    } else {
        // Defensive CPU fallback for unusual layer configs. Matches
        // the inline loop in `compressor_decode_one`
        // (attn_dispatch.rs:2107-2126). The lean f16 path only skips the f32
        // when the GPU matvec applies (matvec_ok), so f32 is present here.
        assert!(
            comp.w_kv_f16.is_none(),
            "compressor CPU fallback reached with f16-only weights (in_dim={in_dim} width={width_u} not f4/f2-aligned)"
        );
        let mut kv_cur = vec![0.0f32; width_u];
        let mut sc_cur = vec![0.0f32; width_u];
        for j in 0..width_u {
            let base = j * in_dim;
            let row_kv = &comp.w_kv[base..base + in_dim];
            let row_sc = &comp.w_gate[base..base + in_dim];
            let mut acc_kv = 0.0f32;
            let mut acc_sc = 0.0f32;
            for k in 0..in_dim {
                acc_kv += row_kv[k] * x_in[k];
                acc_sc += row_sc[k] * x_in[k];
            }
            kv_cur[j] = acc_kv;
            sc_cur[j] = acc_sc;
        }
        (kv_cur, sc_cur)
    };

    // ── APE add (CPU): sc_cur[j] += ape[j, pos_mod] ──────────────
    // antirez `w_ape` is row-major [width, ratio] so element (j, pos_mod)
    // sits at `w_ape[j*ratio + pos_mod]`.
    debug_assert_eq!(comp.w_ape.len(), width_u * ratio as usize);
    for j in 0..width_u {
        sc_cur[j] += comp.w_ape[j * ratio as usize + pos_mod];
    }

    // ── Commit current row into state (CPU) ──────────────────────
    state_kv[row * width_u..(row + 1) * width_u].copy_from_slice(&kv_cur);
    state_score[row * width_u..(row + 1) * width_u].copy_from_slice(&sc_cur);

    // ── Persistent pools: mirror the just-written row into the
    //    resident per-layer pool so the emit pool can read it without
    //    re-uploading all rows. APE already folded into sc_cur above.
    let pool_kv;
    let pool_sc;
    if use_pool {
        let bytes = state_kv.len() * std::mem::size_of::<f32>();
        let (pk, ps) = if is_indexer {
            (
                dispatcher.indexer_state_kv_or_alloc(layer_idx, bytes),
                dispatcher.indexer_state_score_or_alloc(layer_idx, bytes),
            )
        } else {
            (
                dispatcher.compressor_state_kv_or_alloc(layer_idx, bytes),
                dispatcher.compressor_state_score_or_alloc(layer_idx, bytes),
            )
        };
        write_pool_row(&pk, row, &kv_cur);
        write_pool_row(&ps, row, &sc_cur);
        pool_kv = Some(pk);
        pool_sc = Some(ps);
    } else {
        pool_kv = None;
        pool_sc = None;
    }

    if !should_compress {
        return None;
    }

    // ── GPU: pool → pooled[head_dim] ─────────────────────────────
    // DS4_COMPRESSOR_FINISH_GPU=1: chain the emit-row RMS-norm + rope_tail
    // on the resident pooled DeferredBuf inside this same cb (composing the
    // existing rms_norm_mul + rope_tail_in_place kernels), so the emit row
    // is produced GPU-side with no separate rope cb. Returns the finished
    // row directly; finish_emit then only does fp8 + the state rotation.
    // Default-on (set =0 to revert to the CPU rms + separate-cb rope path).
    let finish_gpu = std::env::var("DS4_COMPRESSOR_FINISH_GPU").ok().as_deref() != Some("0");
    let hd = head_dim as usize;
    let n_rot = params.n_rot as usize;
    let mut gpu_out_comp: Option<Vec<f32>> = None;
    let pooled = {
        let scope = dispatcher.batch_scope();
        // Pooled path reads the resident pools (already row-synced
        // above); CPU-slice path uploads the whole mirror per emit.
        let kv_db;
        let sc_db;
        let (kv_buf, sc_buf): (&metal::Buffer, &metal::Buffer) =
            if let (Some(pk), Some(ps)) = (pool_kv.as_ref(), pool_sc.as_ref()) {
                (pk, ps)
            } else {
                kv_db = scope.upload_f32(state_kv);
                sc_db = scope.upload_f32(state_score);
                (kv_db.buffer(), sc_db.buffer())
            };
        let pooled_db = if ratio == 4 {
            scope
                .compressor_pool_ratio4(kv_buf, sc_buf, head_dim)
                .expect("compressor_pool_ratio4")
        } else {
            // ratio != 4: simple softmax-weighted pool over `ratio`
            // rows of `[ratio, head_dim]` state — exactly what the
            // existing `softmax_pool` port handles.
            scope
                .softmax_pool(kv_buf, sc_buf, ratio, head_dim)
                .expect("softmax_pool")
        };
        if finish_gpu && hd % 4 == 0 {
            // out_comp = rms_norm_mul(pooled, w_norm, 1e-6); rope_tail on the
            // last n_rot floats at comp_pos (forward). Matches finish_emit's
            // CPU path bit-for-bit modulo f32-vs-f64 rms accumulation.
            let w_norm_db = scope.weight_f32(comp.w_norm);
            let normed_db = scope
                .rms_norm_mul(&pooled_db, &w_norm_db, 1.0e-6)
                .expect("compressor emit rms_norm_mul");
            if n_rot > 0 {
                let comp_pos = pos + 1 - ratio;
                let off = ((hd - n_rot) * std::mem::size_of::<f32>()) as u64;
                scope
                    .rope_tail_in_place(&normed_db, off, 1, params, comp_pos, false)
                    .expect("compressor emit rope_tail_in_place");
            }
            gpu_out_comp = Some(scope.flush_and_read(&normed_db));
            // pooled itself is no longer needed on CPU when finish_gpu.
            Vec::new()
        } else {
            scope.flush_and_read(&pooled_db)
        }
    };

    Some(finish_emit(
        dispatcher, params, comp, pooled, state_kv, state_score, width_u, ratio, pos,
        pool_kv.as_ref(), pool_sc.as_ref(), gpu_out_comp,
    ))
}

/// True iff the attn-half-fused in-scope compressor path applies: the same
/// `fuse` fast path `compressor_decode_one_metal` takes (ratio==4, matvec
/// f4/f2-aligned, POOL+FUSE env on) AND no Q8_0 activation round-trip (which
/// the resident-`normed` path can't apply without re-materializing x). When
/// false the caller falls back to `compressor_decode_one_metal` (own cb).
pub fn compressor_can_fuse_in_scope(comp: &CompressorInputs<'_>, normed_len: usize) -> bool {
    let use_pool = std::env::var("DS4_COMPRESSOR_POOL").ok().as_deref() != Some("0");
    let fuse_env = std::env::var("DS4_COMPRESSOR_FUSE").ok().as_deref() != Some("0");
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let width_u = 2 * comp.head_dim as usize; // ratio==4 → coff=2
    let matvec_ok = normed_len % 4 == 0 && width_u % 2 == 0;
    use_pool && fuse_env && !want_q80 && comp.compress_ratio == 4 && matvec_ok
}

/// Threads the in-scope compressor encode to its post-flush finish.
pub struct CompressorScopeHandle {
    pk: metal::Buffer,
    ps: metal::Buffer,
    should_compress: bool,
    pub(crate) width_u: usize,
    pub(crate) ratio: u32,
    /// When true, the emit-row RMS-norm + rope_tail were chained on-GPU in
    /// the caller's scope, so the flushed buffer IS the finished emit row
    /// (passed to `finish_emit` as `gpu_out_comp`); the separate-cb CPU rope
    /// is skipped. Mirrors DS4_COMPRESSOR_FINISH_GPU on the non-fuse path.
    pub(crate) finish_gpu: bool,
}

/// Phase 1 of the attn-half-fused compressor: encode matvec(kv,sc) →
/// store_one(+APE) into the resident per-layer pools → (on emit positions)
/// pool_ratio4, all into the caller's already-open `scope`, reading the
/// resident `normed_db` instead of re-uploading `normed`. No commit happens
/// here — the caller flushes the scope (including the returned `pooled_db`,
/// when `Some`) together with its attn-half outputs in one cb, then calls
/// [`compressor_finish_in_scope`]. This is the per-compressor command buffer
/// that `compressor_decode_one_metal`'s `fuse` branch would otherwise commit
/// on its own; folding it here removes one cb/layer.
///
/// Precondition: `compressor_can_fuse_in_scope(comp, normed_db.len())`.
#[allow(clippy::too_many_arguments)]
pub fn compressor_encode_in_scope(
    scope: &BatchScope<'_>,
    dispatcher: &MetalDispatcher,
    params: &LayerParams,
    comp: &CompressorInputs<'_>,
    normed_db: &DeferredBuf,
    state_kv_len: usize,
    pos: u32,
    layer_idx: u32,
    is_indexer: bool,
    // When `Some((ring, row))` and this is an emit position with finish_gpu,
    // the GPU-resident emit row is blit-copied into `ring[row*head_dim]` in
    // THIS scope — making the compressed-KV ring self-sufficient on the GPU
    // so the flash gather has no CPU comp_kv_ring dependency (step 9b).
    ring: Option<(&metal::Buffer, u32)>,
) -> anyhow::Result<(CompressorScopeHandle, Option<DeferredBuf>)> {
    let ratio = comp.compress_ratio;
    let coff = if ratio == 4 { 2usize } else { 1usize };
    let width_u = coff * comp.head_dim as usize;
    let in_dim = normed_db.len();
    let kv_db = comp_matvec(scope, comp.w_kv, comp.w_kv_f16, normed_db, in_dim, width_u)?;
    let sc_db = comp_matvec(scope, comp.w_gate, comp.w_gate_f16, normed_db, in_dim, width_u)?;
    compressor_encode_with_proj(
        scope, dispatcher, params, comp, &kv_db, &sc_db, state_kv_len, pos, layer_idx,
        is_indexer, ring,
    )
}

/// Like [`compressor_encode_in_scope`] but with the kv + score projections
/// PRE-COMPUTED (each `[width_u]` for THIS position) — skips the two per-position
/// `comp_matvec`s. The chunk-prefill cores batch those projections across the K
/// chunk positions in ONE `matmul_f16_k`/`matmul_f32_k` (the per-position matvecs
/// were ~17% of prefill) and call this per position with sliced rows. Everything
/// after the projection (state store, pool, emit/finish) is byte-identical to
/// `compressor_encode_in_scope`.
#[allow(clippy::too_many_arguments)]
pub fn compressor_encode_with_proj(
    scope: &BatchScope<'_>,
    dispatcher: &MetalDispatcher,
    params: &LayerParams,
    comp: &CompressorInputs<'_>,
    kv_db: &DeferredBuf,
    sc_db: &DeferredBuf,
    state_kv_len: usize,
    pos: u32,
    layer_idx: u32,
    is_indexer: bool,
    ring: Option<(&metal::Buffer, u32)>,
) -> anyhow::Result<(CompressorScopeHandle, Option<DeferredBuf>)> {
    let ratio = comp.compress_ratio;
    let head_dim = comp.head_dim;
    // ratio==4 keeps a 2-window state (coff=2); other ratios a single window.
    let coff = if ratio == 4 { 2usize } else { 1usize };
    let width_u = coff * head_dim as usize;
    let should_compress = ((pos + 1) % ratio) == 0;

    let bytes = state_kv_len * std::mem::size_of::<f32>();
    let (pk, ps) = if is_indexer {
        (
            dispatcher.indexer_state_kv_or_alloc(layer_idx, bytes),
            dispatcher.indexer_state_score_or_alloc(layer_idx, bytes),
        )
    } else {
        (
            dispatcher.compressor_state_kv_or_alloc(layer_idx, bytes),
            dispatcher.compressor_state_score_or_alloc(layer_idx, bytes),
        )
    };

    // store_one kernel reads APE [ratio, width]; comp.w_ape is [width, ratio]
    // — transpose into kernel layout (same as the fuse branch).
    let mut ape = vec![0.0f32; ratio as usize * width_u];
    for j in 0..width_u {
        for r in 0..ratio as usize {
            ape[r * width_u + j] = comp.w_ape[j * ratio as usize + r];
        }
    }
    let ape_bytes =
        unsafe { std::slice::from_raw_parts(ape.as_ptr() as *const u8, ape.len() * 4) };

    scope.compressor_store_one_db(
        kv_db, sc_db, ape_bytes, &pk, &ps, width_u as u32, ratio, pos, false,
    )?;
    // On emit, pool the 2-window state → pooled[head_dim], and (default-on,
    // DS4_COMPRESSOR_FINISH_GPU) chain the emit-row RMS-norm + rope_tail in
    // THIS scope so the finished row comes back in the caller's single flush
    // — no separate-cb dispatcher.rope_tail in finish_emit. Mirrors the
    // non-fuse finish_gpu path (rms_norm_mul + rope_tail_in_place).
    let hd = head_dim as usize;
    let finish_gpu = should_compress
        && std::env::var("DS4_COMPRESSOR_FINISH_GPU").ok().as_deref() != Some("0")
        && hd % 4 == 0;
    let emit_db = if should_compress {
        // ratio==4 → two-window pool; other ratios → softmax-weighted pool over
        // `ratio` rows. Mirrors the staged `compressor_decode_one_metal` emit
        // (compressor.rs ~272) exactly, so the resident emit is bit-identical.
        let pooled_db = if ratio == 4 {
            scope.compressor_pool_ratio4(&pk, &ps, head_dim)?
        } else {
            scope.softmax_pool(&pk, &ps, ratio, head_dim)?
        };
        if finish_gpu {
            let w_norm_db = scope.weight_f32(comp.w_norm);
            let normed_db = scope.rms_norm_mul(&pooled_db, &w_norm_db, 1.0e-6)?;
            let n_rot = params.n_rot as usize;
            if n_rot > 0 {
                let comp_pos = pos + 1 - ratio;
                let off = ((hd - n_rot) * std::mem::size_of::<f32>()) as u64;
                scope.rope_tail_in_place(&normed_db, off, 1, params, comp_pos, false)?;
            }
            // GPU-side ring append: copy the finished emit row into ring[row].
            if let Some((ring_buf, ring_row)) = ring {
                scope.copy_buf_into(&normed_db, ring_buf, ring_row as usize * hd);
            }
            Some(normed_db)
        } else {
            Some(pooled_db)
        }
    } else {
        None
    };
    Ok((
        CompressorScopeHandle { pk, ps, should_compress, width_u, ratio, finish_gpu },
        emit_db,
    ))
}

/// Phase 2: after the caller flushed the scope (so `store_one` and any
/// `pool_ratio4` completed, and `pooled` holds the read-back pooled row),
/// resync the CPU state mirrors from the resident pools and, on emit, run
/// the [`finish_emit`] tail (RMS + rope_tail + optional FP8 + ratio==4
/// rotation). Returns the emitted compressed row on emit positions, else
/// `None`. Mirrors the post-flush tail of the `fuse` branch.
pub fn compressor_finish_in_scope(
    handle: CompressorScopeHandle,
    dispatcher: &MetalDispatcher,
    params: &LayerParams,
    comp: &CompressorInputs<'_>,
    pooled: Vec<f32>,
    state_kv: &mut [f32],
    state_score: &mut [f32],
    pos: u32,
) -> Option<Vec<f32>> {
    sync_pool_to_mirror(&handle.pk, state_kv);
    sync_pool_to_mirror(&handle.ps, state_score);
    if handle.should_compress {
        // When finish_gpu, `pooled` IS the GPU-finished emit row (RMS+rope
        // already chained in-scope) — hand it to finish_emit as gpu_out_comp
        // so it skips the CPU rms + separate-cb rope. Otherwise it's the raw
        // pooled row and finish_emit does rms+rope on CPU.
        let (pooled_arg, gpu_out) = if handle.finish_gpu {
            (Vec::new(), Some(pooled))
        } else {
            (pooled, None)
        };
        Some(finish_emit(
            dispatcher,
            params,
            comp,
            pooled_arg,
            state_kv,
            state_score,
            handle.width_u,
            handle.ratio,
            pos,
            Some(&handle.pk),
            Some(&handle.ps),
            gpu_out,
        ))
    } else {
        None
    }
}

/// Emit tail shared by the CPU-mirror and fused paths: RMS norm, rope
/// tail, optional FP8, and the ratio==4 two-window rotation (with pool
/// resync). Returns the compressed row.
#[allow(clippy::too_many_arguments)]
fn finish_emit(
    dispatcher: &MetalDispatcher,
    params: &LayerParams,
    comp: &CompressorInputs<'_>,
    pooled: Vec<f32>,
    state_kv: &mut [f32],
    state_score: &mut [f32],
    width_u: usize,
    ratio: u32,
    pos: u32,
    pool_kv: Option<&metal::Buffer>,
    pool_sc: Option<&metal::Buffer>,
    // When `Some`, the RMS-norm + rope-tail emit row was already computed
    // on-GPU in the caller's scope (DS4_COMPRESSOR_FINISH_GPU) — skip the
    // CPU rms + separate-cb dispatcher.rope_tail and use it directly.
    gpu_out_comp: Option<Vec<f32>>,
) -> Vec<f32> {
    let head_dim = comp.head_dim;
    let hd = head_dim as usize;
    let n_rot = params.n_rot as usize;
    let mut out_comp = if let Some(o) = gpu_out_comp {
        debug_assert_eq!(o.len(), hd);
        o
    } else {
        let mut ss: f64 = 0.0;
        for v in pooled.iter() {
            ss += (*v as f64) * (*v as f64);
        }
        let rms = 1.0_f32 / ((ss / hd as f64) as f32 + 1.0e-6).sqrt();
        debug_assert_eq!(comp.w_norm.len(), hd);
        let mut out_comp = vec![0.0f32; hd];
        for i in 0..hd {
            out_comp[i] = pooled[i] * rms * comp.w_norm[i];
        }
        if n_rot > 0 {
            let comp_pos = pos + 1 - ratio;
            let tail = &mut out_comp[hd - n_rot..hd];
            dispatcher.rope_tail(params, tail, comp_pos, false);
        }
        out_comp
    };
    let want_fp8 = std::env::var("DS4_FP8_KV_QUANT").ok().as_deref() == Some("1");
    if hd == 512 && n_rot > 0 && want_fp8 {
        ds4_fp8_kv_quantize_row_inplace(&mut out_comp, hd, n_rot);
    }
    if ratio == 4 {
        let r = ratio as usize;
        for k in 0..r {
            let src = ((r + k) * width_u)..((r + k + 1) * width_u);
            state_kv.copy_within(src.clone(), k * width_u);
            state_score.copy_within(src, k * width_u);
        }
        for k in 0..r {
            let src = (k * width_u)..((k + 1) * width_u);
            state_kv.copy_within(src.clone(), (r + k) * width_u);
            state_score.copy_within(src, (r + k) * width_u);
        }
        if let (Some(pk), Some(ps)) = (pool_kv, pool_sc) {
            write_pool_all(pk, state_kv);
            write_pool_all(ps, state_score);
        }
    }
    out_comp
}

/// Write one row (`width` floats) into a resident Shared-storage pool
/// at row index `row`. Pools are MTLResourceStorageModeShared so the
/// CPU sees the same memory the next dispatch reads — no copy/upload.
#[inline]
fn write_pool_row(buf: &metal::Buffer, row: usize, vals: &[f32]) {
    let n = vals.len();
    unsafe {
        let p = (buf.contents() as *mut f32).add(row * n);
        std::ptr::copy_nonoverlapping(vals.as_ptr(), p, n);
    }
}

/// Copy a resident Shared-storage pool back into the CPU mirror so the
/// fused path can run rotation + oracle compares on host data.
#[inline]
fn sync_pool_to_mirror(buf: &metal::Buffer, dst: &mut [f32]) {
    unsafe {
        let p = buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(p, dst.as_mut_ptr(), dst.len());
    }
}

/// Overwrite the whole pool with the CPU mirror (used after the
/// ratio==4 rotation to resync the persistent rows).
#[inline]
fn write_pool_all(buf: &metal::Buffer, vals: &[f32]) {
    unsafe {
        let p = buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(vals.as_ptr(), p, vals.len());
    }
}
