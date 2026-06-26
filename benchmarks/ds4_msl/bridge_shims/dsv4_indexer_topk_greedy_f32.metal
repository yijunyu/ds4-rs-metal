// GPU cooperative greedy top-k for the DS4 indexer selection.
//
// Matches the CPU `indexer_allowed_decode_one` greedy: top_k rounds, each
// picking the highest-score not-yet-selected row, ties broken by LOWEST index
// (the CPU uses a strict `>` scan from c=0, so equal scores keep the earlier
// index). One threadgroup of 256 threads iterates; `scores` is mutated in place
// as scratch (the picked row is set to -INF so the next round skips it). Output
// `selected[top_k]` holds the chosen row indices in descending-score order
// (the host sorts ascending to match the CPU filter, but order is irrelevant to
// the flash gather).
//
// Avoids the multi-block bitonic argsort+merge (overkill for top-512 of a few
// thousand rows) — we only need the SET, not a full sort. Single-TG, any n_comp
// (reads scores from device memory; no threadgroup-size cap on n_comp).
//
// Precondition (decode path): top_k < n_comp (the GPU indexer only runs when
// n_comp > top_k), so every round finds a valid pick.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_indexer_topk_greedy(
        device       float * scores   [[buffer(0)]], // [n_comp], mutated scratch
        device       int   * selected [[buffer(1)]], // [top_k] out
        constant     uint  & n_comp   [[buffer(2)]],
        constant     uint  & top_k    [[buffer(3)]],
        uint tid [[thread_position_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    threadgroup float tmax[256];
    threadgroup int   tidx[256];

    for (uint r = 0; r < top_k; ++r) {
        // Per-thread local argmax over a strided slice. Scanning ascending c
        // with strict `>` keeps the lowest index among this thread's ties.
        float best = -INFINITY;
        int   bi   = -1;
        for (uint c = tid; c < n_comp; c += nt) {
            float v = scores[c];
            if (v > best) { best = v; bi = (int)c; }
        }
        tmax[tid] = best;
        tidx[tid] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction: higher value wins; on equal value the lower index
        // wins (invalid index -1 treated as "never preferred").
        for (uint s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = tmax[tid + s];
                int   oi = tidx[tid + s];
                float cv = tmax[tid];
                int   ci = tidx[tid];
                bool take = (ov > cv) ||
                    (ov == cv && oi >= 0 && (ci < 0 || oi < ci));
                if (take) { tmax[tid] = ov; tidx[tid] = oi; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            int idx = tidx[0];
            selected[r] = idx;
            if (idx >= 0) {
                scores[idx] = -INFINITY; // mask so the next round skips it
            }
        }
        // Device-memory barrier so the masked score is visible next round.
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

// ── Parallel threshold-select top-k (replaces the top_k-round greedy) ─────────
//
// Produces the SAME SET as ds4_dsv4_indexer_topk_greedy — the top_k rows by
// score, ties broken by LOWEST index — but in ~32 parallel-count passes instead
// of top_k(=512) sequential argmax-and-mask rounds. At @3000 the greedy is the
// decode wall (~27 ms/token = 1.3 ms × 21 layers, single-threadgroup, fully
// serial); this binary-searches the rank-`top_k` score value, then gathers.
//
// `scores` is READ-ONLY (not mutated — unlike the greedy scratch). Output
// `selected[top_k]` is in ASCENDING index order (order is irrelevant to the
// softmax gather; deterministic so lean f16 vs non-lean f32 stay byte-identical
// when their scores agree, exactly as the greedy did).
//
// IEEE float → orderable uint key: monotonic across the full float range
// (handles negatives / ±INF), so `count(key >= K)` is monotonically
// non-increasing in K and a binary search on K converges to the threshold.
static inline uint ds4_orderable_f32(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void ds4_dsv4_indexer_topk_threshold(
        device const float * scores   [[buffer(0)]], // [n_comp], read-only
        device       int   * selected [[buffer(1)]], // [top_k] out (ascending idx)
        constant     uint  & n_comp   [[buffer(2)]],
        constant     uint  & top_k    [[buffer(3)]],
        uint tid [[thread_position_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    threadgroup uint tcnt[256];

    // Binary search for Kt = the largest key K with count(key >= K) >= top_k
    // (i.e. the rank-top_k key). 32 fixed iterations → exact uint convergence;
    // fixed count keeps every barrier uniform across the threadgroup.
    uint lo = 0u, hi = 0xFFFFFFFFu;
    for (int it = 0; it < 32; ++it) {
        // Overflow-safe ceil-mid = lo + ceil((hi-lo)/2): pulls toward hi so a
        // feasible mid makes progress upward (max-feasible search).
        uint span = hi - lo;
        uint mid = lo + (span - (span >> 1));
        uint c = 0u;
        for (uint i = tid; i < n_comp; i += nt) {
            if (ds4_orderable_f32(scores[i]) >= mid) c++;
        }
        tcnt[tid] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) tcnt[tid] += tcnt[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint total = tcnt[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (total >= top_k) lo = mid; else hi = mid - 1u;
    }
    if (tid == 0) {
        uint Kt = lo;
        // Gather exactly top_k indices: first all rows with key > Kt (there are
        // < top_k of them), then rows with key == Kt in ascending index order
        // until top_k is reached (lowest-index tie-break == the greedy's).
        uint w = 0u;
        for (uint c = 0u; c < n_comp && w < top_k; ++c) {
            if (ds4_orderable_f32(scores[c]) > Kt) selected[w++] = (int)c;
        }
        for (uint c = 0u; c < n_comp && w < top_k; ++c) {
            if (ds4_orderable_f32(scores[c]) == Kt) selected[w++] = (int)c;
        }
    }
}

// ── BATCHED parallel threshold-select top-k (whole-chunk prefill) ─────────────
//
// One threadgroup per token over scores[n_tokens][n_comp] (row-major) →
// selected[n_tokens][top_k]. Identical math + tie-break to the single-token
// ds4_dsv4_indexer_topk_threshold above, just batched over the grid. Handles ANY
// n_comp (no threadgroup-size cap), so it replaces the single-block bitonic
// argsort (capped n_comp<=1024) for long-context single-chunk prefill (ctx>4096).
//
// ORDER: the gather emits all rows with key > Kt (ascending index) then rows with
// key == Kt (ascending index). Causally-masked future rows are scored -INFINITY
// (the minimum orderable key), so finite (valid, idx<visible) rows ALWAYS rank in
// the `> Kt` group and future (idx>=visible) rows can only appear in the `== Kt`
// group — hence every idx<visible precedes any idx>=visible, exactly what
// dsv4_indexed_mixed_attention_h8's `break (idx>=visible)` requires. Unfilled
// slots (n_comp < top_k) are padded -1 (the flash skips idx<0).
kernel void ds4_dsv4_indexer_topk_threshold_batched(
        device const float * scores   [[buffer(0)]], // [n_tokens, n_comp] row-major, read-only
        device       int   * selected [[buffer(1)]], // [n_tokens, top_k] out
        constant     uint  & n_comp   [[buffer(2)]],
        constant     uint  & top_k    [[buffer(3)]],
        constant     uint  & n_tokens [[buffer(4)]],
        uint  tgid [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  nt   [[threads_per_threadgroup]]) {
    if (tgid >= n_tokens) return;
    device const float * srow = scores   + (uint64_t)tgid * n_comp;
    device       int   * drow = selected + (uint64_t)tgid * top_k;
    threadgroup uint tcnt[256];

    uint lo = 0u, hi = 0xFFFFFFFFu;
    for (int it = 0; it < 32; ++it) {
        uint span = hi - lo;
        uint mid = lo + (span - (span >> 1));
        uint c = 0u;
        for (uint i = tid; i < n_comp; i += nt) {
            if (ds4_orderable_f32(srow[i]) >= mid) c++;
        }
        tcnt[tid] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) tcnt[tid] += tcnt[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint total = tcnt[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (total >= top_k) lo = mid; else hi = mid - 1u;
    }
    if (tid == 0) {
        uint Kt = lo;
        uint w = 0u;
        for (uint c = 0u; c < n_comp && w < top_k; ++c) {
            if (ds4_orderable_f32(srow[c]) > Kt) drow[w++] = (int)c;
        }
        for (uint c = 0u; c < n_comp && w < top_k; ++c) {
            if (ds4_orderable_f32(srow[c]) == Kt) drow[w++] = (int)c;
        }
        for (; w < top_k; ++w) drow[w] = -1;
    }
}
