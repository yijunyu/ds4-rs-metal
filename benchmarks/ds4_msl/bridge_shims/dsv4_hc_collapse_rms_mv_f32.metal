// Fused hc_collapse stage-1+2: rms_norm_mul(prev_hc, unit_gamma) -> hc_fn matvec.
//
// The two-kernel path is:
//   flat[i] = prev_hc[i] * rsqrt(mean(prev_hc^2) + eps)      (rms, unit gamma)
//   mix[o]  = sum_i hc_fn[o,i] * flat[i]                     (f32 matvec, 24 rows)
// Because gamma is unit, the rms scale factors out of the dot product:
//   mix[o] = scale * dot(hc_fn_row_o, prev_hc),  scale = rsqrt(mean(prev_hc^2)+eps)
// so we compute the 24 raw dots AND the sum-of-squares in ONE pass over prev_hc,
// then apply `scale` — removing the rms->matvec encoder boundary and the 64KB
// `flat` round-trip. One threadgroup per output row `o`; the rms reduction is
// recomputed per row (prev_hc is read once per TG and reused for both the dot
// and the sum-of-squares, so no extra prev_hc traffic within a TG).
//
// NOT bit-identical to the 2-kernel path (the dot/rms reduction order differs),
// but argmax-stable (rel ~1e-6), like the q8 nsg change. eps formula matches
// norm.metal kernel_rms_norm_fuse_impl: mean = sum(x^2)/ne00; scale = 1/sqrt(mean+eps).
#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_hc_collapse_rms_mv_f32(
        device const float4 * hc_fn   [[buffer(0)]], // [mix_hc, hc_dim/4] row-major
        device const float4 * prev_hc [[buffer(1)]], // [hc_dim/4]
        device       float  * mix     [[buffer(2)]], // [mix_hc]
        constant     uint   & hc_dim  [[buffer(3)]],
        constant     float  & eps     [[buffer(4)]],
        uint   o     [[threadgroup_position_in_grid]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    threadgroup float sh_d[32];
    threadgroup float sh_s[32];

    const uint n4 = hc_dim / 4;
    device const float4 * wrow = hc_fn + (uint64_t)o * n4;

    float ldot = 0.0f;
    float lss  = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 p = prev_hc[i];
        ldot += dot(wrow[i], p);
        lss  += dot(p, p);
    }
    ldot = simd_sum(ldot);
    lss  = simd_sum(lss);
    if (tiisg == 0) {
        sh_d[sgitg] = ldot;
        sh_s[sgitg] = lss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const uint nsg = ntg / 32;
        float d = (tiisg < nsg) ? sh_d[tiisg] : 0.0f;
        float s = (tiisg < nsg) ? sh_s[tiisg] : 0.0f;
        d = simd_sum(d);
        s = simd_sum(s);
        if (tiisg == 0) {
            const float mean  = s / (float)hc_dim;
            const float scale = 1.0f / sqrt(mean + eps);
            mix[o] = d * scale;
        }
    }
}
