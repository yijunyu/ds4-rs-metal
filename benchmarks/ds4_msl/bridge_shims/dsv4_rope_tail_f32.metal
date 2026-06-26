// Bridge shim for `ds4_kernel_dsv4_rope_tail_f32` (M122).
//
// Per-head partial RoPE: rotates the n_rot tail of each head in place.
// The encoder in `crates/ds4_metal/src/macos.rs::rope_tail_impl` passes
// only the rope-tail slice — so head_dim == n_rot and n_nope == 0.
// We therefore implement the rotation only; the n_nope passthrough
// branch is a no-op here.
//
// Encoder contract (buffers + 23 scalar uniforms):
//   buffer(0):  src      const float* (n_heads * n_rot)
//   buffer(1):  pos      const int*   (length 1, pos[0] = position)
//   buffer(2):  freqs    const float* (unused — inline-base mode)
//   buffer(3):  dst      float*       (n_heads * n_rot)
//   buffer(4..26): 23 packed uniforms (see rope_tail_impl):
//     [0]  head_dim       (= n_rot)
//     [1]  n_rot          (u32)
//     [2]  n_nope         (= 0)
//     [3]  freq_base      (f32 bits)
//     [4]  freq_scale     (f32 bits)
//     [5]  ext_factor     (f32 bits)
//     [6]  attn_factor    (f32 bits)
//     [7]  beta_fast      (= 32.0)
//     [8]  beta_slow      (= 1.0)
//     [9]  orig_ctx       (u32)
//     [10] pos            (u32, mirror of pos[0])
//     [11] n_heads        (u32)
//     [12..14]  src strides  (12 = lane, 13 = head row, 14 = whole tensor)
//     [15] backward        (u32 bool)
//     [16..18]  dst strides
//     [19..22]  pad zeros
//
// Dispatch: 1 threadgroup per head, n_rot/2 threads each.

#include <metal_stdlib>
using namespace metal;

// Helper: YaRN linear ramp mask (per antirez ds4_metal/.../rope).
static inline float yarn_ramp_mask(float low, float high, int i0) {
    float y = (float(i0) - low) / max(0.001f, high - low);
    return 1.0f - clamp(y, 0.0f, 1.0f);
}

static inline float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * log(scale) + 1.0f;
}

kernel void ds4_kernel_dsv4_rope_tail_f32(
    device const float * src         [[buffer(0)]],
    device const int   * pos_arr     [[buffer(1)]],
    device const float * freqs       [[buffer(2)]],
    device float       * dst         [[buffer(3)]],
    constant uint      & head_dim    [[buffer(4)]],
    constant uint      & n_rot       [[buffer(5)]],
    constant uint      & n_nope      [[buffer(6)]],
    constant uint      & freq_base_b [[buffer(7)]],
    constant uint      & freq_scale_b[[buffer(8)]],
    constant uint      & ext_factor_b[[buffer(9)]],
    constant uint      & attn_fact_b [[buffer(10)]],
    constant uint      & beta_fast_b [[buffer(11)]],
    constant uint      & beta_slow_b [[buffer(12)]],
    constant uint      & orig_ctx    [[buffer(13)]],
    constant uint      & pos_u       [[buffer(14)]],
    constant uint      & n_heads     [[buffer(15)]],
    constant uint      & src_stride0 [[buffer(16)]],
    constant uint      & src_stride1 [[buffer(17)]],
    constant uint      & src_stride2 [[buffer(18)]],
    constant uint      & backward_u  [[buffer(19)]],
    constant uint      & dst_stride0 [[buffer(20)]],
    constant uint      & dst_stride1 [[buffer(21)]],
    constant uint      & dst_stride2 [[buffer(22)]],
    constant uint      & _pad0       [[buffer(23)]],
    constant uint      & _pad1       [[buffer(24)]],
    constant uint      & _pad2       [[buffer(25)]],
    constant uint      & _pad3       [[buffer(26)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    uint3 tid3                         [[thread_position_in_threadgroup]])
{
    // Metal requires the grid/threadgroup position attributes to be all-scalar
    // or all-vector; tgpig is uint3 (we use .y for the K-row), so tid must also
    // be a vector. Only the x lane indexes the rope pair.
    uint tid = tid3.x;
    (void)freqs; (void)pos_arr; (void)_pad0; (void)_pad1; (void)_pad2; (void)_pad3;
    (void)src_stride1; (void)src_stride2;
    (void)dst_stride0; (void)dst_stride1; (void)dst_stride2;

    // K-batch support: grid.x = head_id, grid.y = K-row (kpos). All historical
    // callers dispatch grid.y=1 → kpos=0, so the kpos terms below vanish and the
    // result is bit-identical to the pre-K kernel. The K-batched rope helpers
    // dispatch grid.y=K, pass pos_u = base_pos (per-row pos = base_pos + kpos),
    // and set src_stride0 = the per-K-row stride IN FLOATS (k_row_stride).
    uint head_id = tgpig.x;
    uint kpos    = tgpig.y;

    float freq_base   = as_type<float>(freq_base_b);
    float freq_scale  = as_type<float>(freq_scale_b);
    float ext_factor  = as_type<float>(ext_factor_b);
    float attn_factor = as_type<float>(attn_fact_b);
    float beta_fast   = as_type<float>(beta_fast_b);
    float beta_slow   = as_type<float>(beta_slow_b);
    int   pos         = int(pos_u) + int(kpos);
    int   backward    = int(backward_u);
    (void)n_nope;

    // We process pairs (2*tid, 2*tid+1) within the n_rot tail of head_id.
    uint pair = tid;
    if (pair * 2u >= n_rot) return;

    // src_stride0 repurposed as k_row_stride (floats); kpos=0 → term vanishes.
    uint base = kpos * src_stride0 + head_id * head_dim;
    uint i0 = pair * 2u;

    // Inline-base frequency: freq = 1 / (freq_base^(i0/n_rot)).
    float exponent = float(i0) / float(n_rot);
    float freq_full = 1.0f / pow(freq_base, exponent);
    // Apply YaRN: freq_scale interpolates between extrapolation (freq_full)
    // and interpolation (freq_full/freq_scale).
    float freq_inter = freq_full / freq_scale;
    float low  = floor(log(float(orig_ctx) / (2.0f * 3.14159265358979323846f * beta_fast))
                       / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;
    float high = ceil(log(float(orig_ctx) / (2.0f * 3.14159265358979323846f * beta_slow))
                      / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;
    float ramp = yarn_ramp_mask(low, high, int(i0));
    float freq = freq_inter * (1.0f - ramp * ext_factor)
               + freq_full  * ramp * ext_factor;

    float theta = float(pos) * freq;
    float mscale = yarn_get_mscale(freq_scale, attn_factor);
    float cos_t = cos(theta) * mscale;
    float sin_t = sin(theta) * mscale;
    if (backward) sin_t = -sin_t;

    float x0 = src[base + i0];
    float x1 = src[base + i0 + 1u];

    dst[base + i0]      = x0 * cos_t - x1 * sin_t;
    dst[base + i0 + 1u] = x0 * sin_t + x1 * cos_t;
}
