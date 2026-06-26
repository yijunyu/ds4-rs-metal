// Phase 2 MoE-K Step 2 — sum6_k reduction kernel.
//
// After the K-amortized down matmul (kernel_mul_mm_id_q8_0/q4_K/iq2_xxs_f32),
// the output is `down_out[K, 6, d_embd]`: per-token, per-expert-slot
// contributions. moe_swiglu_weight already baked the per-slot route weight
// into mid, so the final FFN-MoE output is just the slot-dim sum:
//
//   out[k, j] = Σ_slot down_out[k, slot, j]   for slot in 0..6
//
// This kernel does that reduction. One thread per (k, d_embd) output cell;
// reads 6 contiguous slot contributions, sums, writes one float.
//
// Input  layout: [K, 6, d_embd] f32 — slot-major within K.
// Output layout: [K, d_embd] f32.
//
// Grid: 1D, dispatch K*d_embd threads.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_kernel_dsv4_sum6_k_f32(
        device const float * src    [[buffer(0)]],
        device       float * dst    [[buffer(1)]],
        constant     uint  & d_embd [[buffer(2)]],
        constant     uint  & k_pos  [[buffer(3)]],
        uint                 tid    [[thread_position_in_grid]]) {
    const uint total = k_pos * d_embd;
    if (tid >= total) { return; }

    const uint k = tid / d_embd;
    const uint d = tid % d_embd;
    const uint base = k * 6 * d_embd + d;

    float s = 0.0f;
    s += src[base + 0u * d_embd];
    s += src[base + 1u * d_embd];
    s += src[base + 2u * d_embd];
    s += src[base + 3u * d_embd];
    s += src[base + 4u * d_embd];
    s += src[base + 5u * d_embd];
    dst[tid] = s;
}
