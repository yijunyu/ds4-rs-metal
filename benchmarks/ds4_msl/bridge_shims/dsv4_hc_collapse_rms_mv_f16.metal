// F16 twin of dsv4_hc_collapse_rms_mv_f32: the no-copy-f16 hc-collapse path.
// Identical math, but the hc_fn projection weight is read as half4 (2 B/elem,
// borrowed straight from the model mmap) instead of float4. The activation
// (prev_hc) and the accumulation stay f32, so this is bit-identical to the f32
// kernel on the dequantized weight (F16->f32 is exact) modulo the same
// reduction order. See deferred.rs fused_hc_collapse_stage12 (f16 branch).
#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_hc_collapse_rms_mv_f16(
        device const half4  * hc_fn   [[buffer(0)]], // [mix_hc, hc_dim/4] row-major, f16
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
    device const half4 * wrow = hc_fn + (uint64_t)o * n4;

    float ldot = 0.0f;
    float lss  = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 p = prev_hc[i];
        ldot += dot(float4(wrow[i]), p);
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
