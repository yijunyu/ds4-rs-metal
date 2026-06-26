// Bridge shim for `ds4_kernel_dsv4_softplus_sqrt_f32_4`.
//
// Decode-time router-logit transform: f(x) = sqrt(softplus(x)).
// softplus(x) = log(1 + exp(x)) — use the numerically stable form
// `max(x,0) + log1p(exp(-|x|))` to avoid overflow on large x.
//
// Encoder contract:
//   buffer(0): src     const float* (n floats, must be %4 == 0)
//   buffer(1): dst     float*       (n floats)
//   buffer(2): ne0_4   uint32  = n/4 (number of float4 lanes)
//   buffer(3): nb_src  uint32  = row stride bytes (unused here)
//   buffer(4): nb_dst  uint32  = row stride bytes (unused here)
//
// Dispatch: 1 threadgroup, ne0_4 threads (≤1024); each thread
// transforms one float4 lane.

#include <metal_stdlib>
using namespace metal;

// MSL has no log1p; expand `log(1 + e^-ax)` directly. For ax ≥ 0,
// `1 + e^-ax` is in (1, 2], so log() is well-behaved (no catastrophic
// cancellation in the magnitudes we expect at decode time).
static inline float stable_softplus(float x) {
    float ax = fabs(x);
    return max(x, 0.0f) + log(1.0f + exp(-ax));
}

kernel void ds4_kernel_dsv4_softplus_sqrt_f32_4(
    device const float * src   [[buffer(0)]],
    device float       * dst   [[buffer(1)]],
    constant uint      & ne0_4 [[buffer(2)]],
    constant uint      & nb_src [[buffer(3)]],
    constant uint      & nb_dst [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    (void)nb_src; (void)nb_dst;
    if (tid >= ne0_4) return;
    float4 v = ((device const float4*)src)[tid];
    float4 out;
    out.x = sqrt(stable_softplus(v.x));
    out.y = sqrt(stable_softplus(v.y));
    out.z = sqrt(stable_softplus(v.z));
    out.w = sqrt(stable_softplus(v.w));
    ((device float4*)dst)[tid] = out;
}
