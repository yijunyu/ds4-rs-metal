// Bridge shim for `ds4_kernel_silu_f32_4`.
//
// SiLU activation: f(x) = x / (1 + exp(-x)) = x * sigmoid(x).
// Vectorized over float4. Antirez fuses this into glu.metal; we
// expose a standalone kernel because the registry expects one for
// the M4 "no-antirez-at-runtime" surface.
//
// Encoder contract (consistent with other *_f32_4 unaries):
//   buffer(0): src     const float*
//   buffer(1): dst     float*
//   buffer(2): n4      uint32  = n/4
//
// Dispatch: 1 threadgroup, n4 threads (≤1024).

#include <metal_stdlib>
using namespace metal;

kernel void ds4_kernel_silu_f32_4(
    device const float * src [[buffer(0)]],
    device float       * dst [[buffer(1)]],
    constant uint      & n4  [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid >= n4) return;
    float4 v = ((device const float4*)src)[tid];
    float4 sig;
    sig.x = 1.0f / (1.0f + exp(-v.x));
    sig.y = 1.0f / (1.0f + exp(-v.y));
    sig.z = 1.0f / (1.0f + exp(-v.z));
    sig.w = 1.0f / (1.0f + exp(-v.w));
    ((device float4*)dst)[tid] = v * sig;
}
