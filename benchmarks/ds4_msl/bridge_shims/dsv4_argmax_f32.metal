// GPU argmax over a logit row → token id (greedy decode).
//
// Single threadgroup (256 threads) cooperative reduction: each thread scans a
// strided slice for its local max, then a tree reduction finds the global max.
// Ties break to the LOWEST index, matching the CPU greedy `argmax_i32`
// (strict `>` scan from index 0). Lets the lm_head tail return just the token
// id (4 bytes) instead of reading back the full ~129k-element logit vector.
//
// `out[0]` = argmax index. Any `n` (reads logits from device memory; no
// threadgroup-size cap on n).

#include <metal_stdlib>
using namespace metal;

kernel void ds4_argmax_f32(
        device const float * logits [[buffer(0)]], // [n]
        device       int   * out    [[buffer(1)]], // [1] token id
        constant     uint  & n      [[buffer(2)]],
        uint tid [[thread_position_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    threadgroup float tmax[256];
    threadgroup int   tidx[256];

    float best = -INFINITY;
    int   bi   = 0;
    for (uint c = tid; c < n; c += nt) {
        float v = logits[c];
        if (v > best) { best = v; bi = (int)c; } // strict > → lowest index in-thread
    }
    tmax[tid] = best;
    tidx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = tmax[tid + s];
            int   oi = tidx[tid + s];
            // higher value wins; equal value → lower index wins
            if (ov > tmax[tid] || (ov == tmax[tid] && oi < tidx[tid])) {
                tmax[tid] = ov;
                tidx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[0] = tidx[0];
    }
}
