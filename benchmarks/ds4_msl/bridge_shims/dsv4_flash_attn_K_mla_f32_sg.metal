// Phase 2 — K-query flash attention for the persistent FP8-snapped KV cache
// (MLA: shared K/V latent). Validated in Phase 0 at K=8 ratio 1.03 vs K=1
// (drivers/flash_attn_K_simdgroup_check.swift); this shim adapts the kernel
// for our cache format.
//
// Inputs (all f32 from the host pipeline):
//   Qbuf  : [K, n_head, DK] — query, per-position per-head.
//   KVbuf : [N_cache, DK]   — persistent KV cache (raw_cap × n_lora_kv f32 from
//                              kv_fp8_store_persistent: values are FP8 e4m3
//                              quantized then FP16-round-tripped, stored as f32).
//                              MLA: same buffer used for both K and V.
//   Obuf  : [K, n_head, DV] — output, f32.
//
// The kernel reads Qbuf/KVbuf as f32 and casts to half on the shared-memory
// write — lossless because all values are already FP16-representable.
//
// FC indices: 1600-1603 (above all upstream bases: FLASH_ATTN_EXT_*=100..500,
// MUL_MV=600, MUL_MM=700, UNARY=1200, BIN=1300, SUM_ROWS=1400, plus 1500/1501
// for the K-position Q8_0 matvec).
//
// Note: appended to ANTIREZ_BRIDGE_SOURCE after preamble + dense.metal + the
// upstream metal files, so metal_stdlib and using namespace metal are in scope.
// metal_simdgroup_matrix is included here (independent header).

#include <metal_simdgroup_matrix>

constant int FC_flash_K_DK   [[function_constant(1600)]];
constant int FC_flash_K_DV   [[function_constant(1601)]];
constant int FC_flash_K_K    [[function_constant(1602)]];
constant int FC_flash_K_NC   [[function_constant(1603)]];
// DS4_CHUNK_SWA_KFLASH: when set, an explicit per-query additive mask
// (buffer 10, half, [K, mask_row_] with mask_row_ >= NC + n_comp_) replaces
// the attn_base_pos / comp_avail derivation. Mask col layout: [0..NC) = raw
// rows (the gathered absolute-order SWA window), [NC..NC+n_comp_) = comp rows.
// 0 = attend, -INF (0xFC00) = masked. Lets the grow-only-scratch comp kernel
// serve the sliding-window prefill tiles (per-query lower+upper bound) that
// attn_base_pos (causal upper-bound only) cannot express. Default false.
constant bool FC_flash_K_HAS_MASK [[function_constant(1604)]];

// Causal-mask semantics: for query row q in [0, K), the attention window is
// [0, attn_base_pos_ + q + 1).  Rows >= that limit get -INFINITY in the
// softmax (zero post-softmax weight).  Callers:
//   - Verifier (writes new K/V at slots [base_slot..base_slot+K)): pass
//     attn_base_pos = base_pos (= position of K-row 0).
//   - MTP drafter (K=1) at draft iter i with prior mtp_n_raw=N:
//     pass attn_base_pos = N + i, so window = N + i + 1 rows.
kernel void ds4_kernel_flash_attn_K_mla_f32_sg(
        device const float * Qbuf            [[buffer(0)]],
        device const float * KVbuf           [[buffer(1)]],
        device       float * Obuf            [[buffer(2)]],
        constant     float & scale_          [[buffer(3)]],
        constant     uint  & n_head_         [[buffer(4)]],
        constant     uint  & attn_base_pos_  [[buffer(5)]],
        device const float * sinks_          [[buffer(6)]],
        device const float * comp_ring_      [[buffer(7)]],   // [n_comp, DK] f32 (compressed-KV ring)
        constant     uint  & n_comp_         [[buffer(8)]],   // # compressed rows (0 = no comp)
        device const uint  * comp_avail_     [[buffer(9)]],   // [n_comp] position each comp row becomes attendable (0 = prefill)
        device const half  * mask_           [[buffer(10)]],  // [K, mask_row_] additive mask (only read if HAS_MASK)
        constant     uint  & mask_row_       [[buffer(11)]],  // mask row stride (>= NC + n_comp_); only used if HAS_MASK
        threadgroup  char  * shmem           [[threadgroup(0)]],
        uint3   tgpig   [[threadgroup_position_in_grid]],
        ushort  tiitg   [[thread_index_in_threadgroup]],
        ushort  tiisg   [[thread_index_in_simdgroup]],
        ushort  sgitg   [[simdgroup_index_in_threadgroup]]) {

    constexpr short Q  = 8;
    constexpr short C  = 8;
    constexpr short NSG = 4;
    constexpr short NW = 32;
    const     int   DK = FC_flash_K_DK;
    const     int   DV = FC_flash_K_DV;
    const     int   K  = FC_flash_K_K;
    const     int   NC = FC_flash_K_NC;
    const     uint  n_head = n_head_;
    const     uint  h = tgpig.x;

    const int DK8 = DK / 8;
    const int DV8 = DV / 8;
    constexpr short NO = 16;  // DV8/NSG tiles per simdgroup (16 when DK=DV=512, NSG=4)

    threadgroup half  * SQ = (threadgroup half *)(shmem);
    threadgroup half  * SK = (threadgroup half *)(shmem + 2 * Q * DK);
    threadgroup half  * SV = (threadgroup half *)(shmem + 2 * Q * DK + 2 * Q * DK);
    threadgroup float * SS = (threadgroup float *)(shmem + 2 * Q * DK + 2 * Q * DK + 2 * DV * Q);

    // 1. Load Q (f32 → half) into SQ[Q × DK], zero rows for k≥K.
    {
        const int nThreads = NSG * NW;
        const int total    = Q * DK;
        for (int t = tiitg; t < total; t += nThreads) {
            const int k = t / DK;
            const int d = t % DK;
            half v = (half)0;
            if (k < K) {
                v = (half)Qbuf[k * (n_head * DK) + h * DK + d];
            }
            SQ[k * DK + d] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online-softmax state seeded with the per-head ATTENTION SINK: the sink
    // is an extra (unscaled) logit in the softmax denominator with NO V
    // contribution (matches decode_step flash_attn_decode + the K=1 production
    // paths, which prepend attn_sinks[h]). Init M = sink, S = exp(sink-M) = 1,
    // and lo (output accumulator) = 0; the sink thus contributes to the
    // denominator but not the numerator. Active queries (k<K) only.
    const float sink_h = sinks_[h];
    float M[8];
    float S[8];
    for (short k = 0; k < Q; ++k) {
        if (k < K) { M[k] = sink_h; S[k] = 1.0f; }
        else       { M[k] = -INFINITY; S[k] = 0.0f; }
    }

    simdgroup_float8x8 lo[NO];
    for (short ii = 0; ii < NO; ++ii) {
        lo[ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    // Chunk-level early exit: row K-1 has the largest window
    // (attn_base_pos + K).  Any chunk whose base row index already exceeds it
    // contributes -INFINITY for ALL queries, so its softmax contribution is
    // zero — skip the chunk entirely as a perf optimization.
    // Under an explicit mask the attendable set is arbitrary (sliding window),
    // so the attn_base_pos-derived early exit is invalid — scan all NC rows and
    // let the per-query mask zero the out-of-window ones.
    const int max_window = FC_flash_K_HAS_MASK ? NC : ((int)attn_base_pos_ + K);
    const int NC_BLK = NC / C;
    for (int ic = 0; ic < NC_BLK; ++ic) {
        const int kv_row_base = ic * C;
        if (kv_row_base >= max_window) break;

        // Load K_chunk (f32 → half) — MLA: K from shared KV cache.
        {
            const int nThreads = NSG * NW;
            const int total    = C * DK;
            for (int t = tiitg; t < total; t += nThreads) {
                const int r = t / DK;
                const int d = t % DK;
                SK[r * DK + d] = (half)KVbuf[(kv_row_base + r) * DK + d];
            }
        }
        // Load V_chunk (f32 → half) — MLA: V from the same shared cache.
        {
            const int nThreads = NSG * NW;
            const int total    = C * DV;
            for (int t = tiitg; t < total; t += nThreads) {
                const int c = t / DV;
                const int d = t % DV;
                SV[c * DV + d] = (half)KVbuf[(kv_row_base + c) * DV + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
        const short i_per_sg = DK8 / NSG;
        const short i_start  = (short)sgitg * i_per_sg;
        const short i_end    = i_start + i_per_sg;
        for (short i = i_start; i < i_end; ++i) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, SQ + 8 * i, DK, 0, false);
            simdgroup_load(mk, SK + 8 * i, DK, 0, true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }

        threadgroup float * mqk_scratch = (threadgroup float *)SK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(mqk, mqk_scratch + sgitg * 64, 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg < 64) {
            float s = 0.0f;
            for (short g = 0; g < NSG; ++g) {
                s += mqk_scratch[g * 64 + tiitg];
            }
            SS[tiitg] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * alpha_sh = mqk_scratch;
        if (sgitg == 0) {
            if (tiisg < (ushort)Q) {
                const short q = (short)tiisg;
                // Causal window for this Q-row: rows [0, attn_base_pos + q + 1).
                // Queries with q >= K are zero-padded Q-rows (see SQ load) and
                // contribute nothing to Obuf, so their mask doesn't matter.
                const int n_raw_q = (int)attn_base_pos_ + (int)q + 1;
                float m_old = M[q];
                float m_max = -INFINITY;
                float row[8];
                for (short c = 0; c < C; ++c) {
                    const int kv_row = kv_row_base + (int)c;
                    bool ok;
                    if (FC_flash_K_HAS_MASK) {
                        // Explicit per-query mask col `kv_row` (raw cols [0..NC)).
                        // Snaps the sliding-window lower+upper bound the gathered
                        // absolute-order SWA window needs. -INF half == masked.
                        ok = (kv_row < NC)
                          && ((float)mask_[(int)q * (int)mask_row_ + kv_row] > -1.0e30f);
                    } else {
                        ok = (kv_row < n_raw_q);
                    }
                    row[c] = ok ? (SS[q * 8 + c] * scale_) : -INFINITY;
                    if (row[c] > m_max) m_max = row[c];
                }
                float m_new = max(m_old, m_max);
                float alpha = exp(m_old - m_new);
                float sum_v = 0.0f;
                for (short c = 0; c < C; ++c) {
                    float v = exp(row[c] - m_new);
                    SS[q * 8 + c] = v;
                    sum_v += v;
                }
                S[q] = alpha * S[q] + sum_v;
                M[q] = m_new;
                alpha_sh[q] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * diag_buf = mqk_scratch + 64;
        if (tiitg < 64) {
            const short r = (short)tiitg / 8;
            const short c = (short)tiitg % 8;
            diag_buf[r * 8 + c] = (r == c) ? alpha_sh[r] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 alpha_mat;
        simdgroup_load(alpha_mat, diag_buf, 8, 0, false);

        for (short ii = 0; ii < NO; ++ii) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, alpha_mat, lo[ii]);
            lo[ii] = tmp;
        }

        simdgroup_float8x8 vs;
        simdgroup_load(vs, SS, 8, 0, false);

        const int d_base_sg = sgitg * (DV / NSG);
        for (short ii = 0; ii < NO; ++ii) {
            const int d_base_tile = d_base_sg + ii * 8;
            simdgroup_half8x8 mv;
            simdgroup_load(mv, SV + d_base_tile, DV, 0, false);
            simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compressed-row attendance (DS4_VERIFY_COMPRESSOR). The n_comp_ rows of
    // comp_ring_ summarize past windows (all at positions < base_pos), so they
    // are causally valid for EVERY query row q — no per-q window mask, only a
    // partial-last-chunk mask for c >= n_comp_. Accumulated into the SAME
    // online-softmax state (M/S/lo) as the raw rows above, matching decode_step
    // attending [raw 0..n_raw | all comp rows]. Skipped when n_comp_ == 0.
    const int NCMP = (int)n_comp_;
    const int NCMP_BLK = (NCMP + C - 1) / C;
    for (int icc = 0; icc < NCMP_BLK; ++icc) {
        const int comp_row_base = icc * C;

        // Load K_chunk + V_chunk from comp_ring (MLA: same buffer for K and V).
        // Rows >= NCMP are zero-filled and masked to -INF below.
        {
            const int nThreads = NSG * NW;
            const int total    = C * DK;
            for (int t = tiitg; t < total; t += nThreads) {
                const int r = t / DK;
                const int d = t % DK;
                const int cr = comp_row_base + r;
                SK[r * DK + d] = (cr < NCMP) ? (half)comp_ring_[cr * DK + d] : (half)0;
            }
        }
        {
            const int nThreads = NSG * NW;
            const int total    = C * DV;
            for (int t = tiitg; t < total; t += nThreads) {
                const int c = t / DV;
                const int d = t % DV;
                const int cr = comp_row_base + c;
                SV[c * DV + d] = (cr < NCMP) ? (half)comp_ring_[cr * DV + d] : (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
        const short i_per_sg = DK8 / NSG;
        const short i_start  = (short)sgitg * i_per_sg;
        const short i_end    = i_start + i_per_sg;
        for (short i = i_start; i < i_end; ++i) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, SQ + 8 * i, DK, 0, false);
            simdgroup_load(mk, SK + 8 * i, DK, 0, true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }

        threadgroup float * mqk_scratch = (threadgroup float *)SK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(mqk, mqk_scratch + sgitg * 64, 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg < 64) {
            float s = 0.0f;
            for (short g = 0; g < NSG; ++g) {
                s += mqk_scratch[g * 64 + tiitg];
            }
            SS[tiitg] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * alpha_sh = mqk_scratch;
        if (sgitg == 0) {
            if (tiisg < (ushort)Q) {
                const short q = (short)tiisg;
                float m_old = M[q];
                float m_max = -INFINITY;
                float row[8];
                for (short c = 0; c < C; ++c) {
                    const int cr = comp_row_base + (int)c;
                    // Per-row comp-causal mask: row q (pos attn_base_pos+q)
                    // attends comp row cr only if cr was emitted at an earlier
                    // position — comp_avail_[cr] <= attn_base_pos+q. Prefill rows
                    // have comp_avail=0 (always attendable); in-batch emits (from
                    // earlier draft rows) have comp_avail = base_pos+j+1 so only
                    // rows k>j see them. Also mask the partial last chunk (cr>=NCMP).
                    bool ok;
                    if (FC_flash_K_HAS_MASK) {
                        // Comp cols sit at [NC .. NC+NCMP) in the per-query mask.
                        ok = (cr < NCMP)
                          && ((float)mask_[(int)q * (int)mask_row_ + NC + cr] > -1.0e30f);
                    } else {
                        ok = (cr < NCMP)
                          && ((int)comp_avail_[cr] <= (int)attn_base_pos_ + (int)q);
                    }
                    row[c] = ok ? (SS[q * 8 + c] * scale_) : -INFINITY;
                    if (row[c] > m_max) m_max = row[c];
                }
                // Fully-masked chunk → m_max=-INF → m_new=m_old, alpha=1,
                // sum_v=0 (exp(-INF)=0): state unchanged, no NaN.
                float m_new = max(m_old, m_max);
                float alpha = exp(m_old - m_new);
                float sum_v = 0.0f;
                for (short c = 0; c < C; ++c) {
                    float v = exp(row[c] - m_new);
                    SS[q * 8 + c] = v;
                    sum_v += v;
                }
                S[q] = alpha * S[q] + sum_v;
                M[q] = m_new;
                alpha_sh[q] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * diag_buf = mqk_scratch + 64;
        if (tiitg < 64) {
            const short r = (short)tiitg / 8;
            const short c = (short)tiitg % 8;
            diag_buf[r * 8 + c] = (r == c) ? alpha_sh[r] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 alpha_mat;
        simdgroup_load(alpha_mat, diag_buf, 8, 0, false);

        for (short ii = 0; ii < NO; ++ii) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, alpha_mat, lo[ii]);
            lo[ii] = tmp;
        }

        simdgroup_float8x8 vs;
        simdgroup_load(vs, SS, 8, 0, false);

        const int d_base_sg = sgitg * (DV / NSG);
        for (short ii = 0; ii < NO; ++ii) {
            const int d_base_tile = d_base_sg + ii * 8;
            simdgroup_half8x8 mv;
            simdgroup_load(mv, SV + d_base_tile, DV, 0, false);
            simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float * S_sh = (threadgroup float *)SV;
    if (sgitg == 0 && tiisg < (ushort)Q) {
        S_sh[tiisg] = S[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * SO_out = (threadgroup float *)SQ;
    {
        const int d_base_sg = sgitg * (DV / NSG);
        for (short ii = 0; ii < NO; ++ii) {
            simdgroup_store(lo[ii], SO_out + (d_base_sg + ii * 8), DV, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        const int nThreads = NSG * NW;
        for (int t = tiitg; t < K * DV; t += nThreads) {
            const int k = t / DV;
            const int d = t % DV;
            const float inv = (S_sh[k] > 0.0f) ? (1.0f / S_sh[k]) : 0.0f;
            Obuf[k * (n_head * DV) + h * DV + d] = SO_out[k * DV + d] * inv;
        }
    }
}
