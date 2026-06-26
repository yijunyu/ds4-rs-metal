// Bridge shim for `ds4_kernel_dsv4_compressor_prefill_noidx_f32`.
//
// STAGE 1 of the fused chunk-graph prefill rewrite. Fuses the per-position
// noidx (ratio != 4, coff == 1, width == head_dim) compressor build into ONE
// dispatch: for the whole chunk it emits all `n_comp = n_tokens / ratio`
// compressed rows at once. Replaces the ~ratio per-position
// store+pool+rms+rope+ring-copy dispatches (× n_comp) with a single grid.
//
// Mirrors antirez `ds4_gpu_compressor_prefill_tensor` (ds4_metal.m:13925) for
// the `else` (non-ratio4) branch, and is per-row byte-equivalent to the
// per-position chain `compressor_store_one_db -> softmax_pool -> rms_norm_mul
// -> rope_tail_in_place` (compressor.rs) when chunk_start % ratio == 0.
//
// Per emit row e in [0, n_comp):
//   pool over the `ratio` projection rows at kv/sc[(e*ratio + r)], with the
//   per-member APE bias added to the score (pos_mod = (pos0 + e*ratio + r)
//   % ratio); softmax-weighted column pool -> pooled[head_dim]; RMS-norm ×
//   norm weight; partial RoPE on the trailing n_rot floats at comp position
//   (pos0 + e*ratio); write to comp_cache[(comp_row0 + e) * head_dim].
//
// Encoder contract (one threadgroup per emit row, `head_dim` threads):
//   buffer(0): kv        const float* [n_tokens × head_dim]  (batched proj)
//   buffer(1): sc        const float* [n_tokens × head_dim]  (batched proj)
//   buffer(2): ape       const float* [head_dim × ratio]     (model layout:
//                          ape[j*ratio + r], row-major width-major)
//   buffer(3): norm      const float* [head_dim]
//   buffer(4): comp      float*       [(comp_row0 + n_comp) × head_dim]
//   buffer(5): args      constant ds4_compressor_prefill_args
//
// args carries the rope parameters in the SAME packed form as
// dsv4_rope_tail_f32 so the rope math is bit-identical to the proven shim.

#include <metal_stdlib>
using namespace metal;

struct ds4_compressor_prefill_args {
    uint head_dim;     // == width (coff == 1)
    uint ratio;        // compress period (!= 4)
    uint n_rot;        // rope tail length (even, <= head_dim)
    uint pos0;         // chunk_start (absolute position of token 0 in the chunk)
    uint comp_row0;    // first comp_cache row to write (n_comp at chunk start)
    uint n_comp;       // number of emit rows = n_tokens / ratio
    float rms_eps;     // 1e-6
    // rope params (mirror dsv4_rope_tail_f32):
    uint freq_base_b;
    uint freq_scale_b;
    uint ext_factor_b;
    uint attn_factor_b;
    uint beta_fast_b;
    uint beta_slow_b;
    uint orig_ctx;
    uint backward;     // 0 (forward) on the emit path
};

static inline float yarn_ramp_mask_cp(float low, float high, int i0) {
    float y = (float(i0) - low) / max(0.001f, high - low);
    return 1.0f - clamp(y, 0.0f, 1.0f);
}

static inline float yarn_get_mscale_cp(float scale, float mscale) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * log(scale) + 1.0f;
}

// Threadgroup scratch: pooled column values (so the RMS reduction can read
// the whole row) + a reduction buffer for sum-of-squares.
kernel void ds4_kernel_dsv4_compressor_prefill_noidx_f32(
    device const float * kv   [[buffer(0)]],
    device const float * sc   [[buffer(1)]],
    device const float * ape  [[buffer(2)]],
    device const float * norm [[buffer(3)]],
    device       float * comp [[buffer(4)]],
    constant ds4_compressor_prefill_args & args [[buffer(5)]],
    uint  e   [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    const uint hd    = args.head_dim;
    const uint ratio = args.ratio;
    if (e >= args.n_comp) return;

    threadgroup float pooled_sh[1024];   // head_dim <= 1024
    threadgroup float red[32];           // one slot per simdgroup (<=32)

    const uint base_row = e * ratio;     // first source row for this emit
    const uint width = hd;               // coff == 1

    // ── 1. softmax-weighted column pool over `ratio` rows ──────────────
    // Each thread owns column j == tid (head_dim threads). Max-stabilized
    // softmax over the `ratio` score values (with APE bias), weighted sum of
    // the matching kv values — exactly kernel_dsv4_softmax_pool / the antirez
    // store_score+softmax_pool chain.
    for (uint j = tid; j < hd; j += tg_size) {
        float max_score = -1.0e30f;
        for (uint r = 0; r < ratio; ++r) {
            uint pos = args.pos0 + base_row + r;
            uint pos_mod = pos % ratio;
            float s = sc[(base_row + r) * width + j] + ape[j * ratio + pos_mod];
            max_score = max(max_score, s);
        }
        float denom = 0.0f;
        float sum   = 0.0f;
        for (uint r = 0; r < ratio; ++r) {
            uint pos = args.pos0 + base_row + r;
            uint pos_mod = pos % ratio;
            float s = sc[(base_row + r) * width + j] + ape[j * ratio + pos_mod];
            float w = exp(s - max_score);
            denom += w;
            sum   += w * kv[(base_row + r) * width + j];
        }
        pooled_sh[j] = denom > 0.0f ? sum / denom : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. RMS norm over the pooled row (× norm weight) ────────────────
    // ss = Σ pooled[j]^2 ; rms = 1/sqrt(ss/hd + eps).
    float local_ss = 0.0f;
    for (uint j = tid; j < hd; j += tg_size) {
        float v = pooled_sh[j];
        local_ss += v * v;
    }
    // simdgroup reduce, then a tiny threadgroup reduce across simdgroups.
    local_ss = simd_sum(local_ss);
    uint sg_id   = tid / 32u;
    uint lane    = tid % 32u;
    if (lane == 0u) red[sg_id] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    uint n_sg = (tg_size + 31u) / 32u;
    for (uint g = 0; g < n_sg; ++g) ss += red[g];
    float rms = 1.0f / sqrt(ss / float(hd) + args.rms_eps);

    for (uint j = tid; j < hd; j += tg_size) {
        pooled_sh[j] = pooled_sh[j] * rms * norm[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. partial RoPE on the trailing n_rot floats (bit-identical to
    //       dsv4_rope_tail_f32) at comp position pos0 + e*ratio ──────────
    const uint n_rot = args.n_rot;
    if (n_rot > 0u) {
        float freq_base   = as_type<float>(args.freq_base_b);
        float freq_scale  = as_type<float>(args.freq_scale_b);
        float ext_factor  = as_type<float>(args.ext_factor_b);
        float attn_factor = as_type<float>(args.attn_factor_b);
        float beta_fast   = as_type<float>(args.beta_fast_b);
        float beta_slow   = as_type<float>(args.beta_slow_b);
        int   pos         = int(args.pos0 + base_row);
        int   backward    = int(args.backward);
        const uint tail0  = hd - n_rot;   // rope acts on [tail0 .. hd)

        float low  = floor(log(float(args.orig_ctx) / (2.0f * M_PI_F * beta_fast))
                           / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;
        float high = ceil(log(float(args.orig_ctx) / (2.0f * M_PI_F * beta_slow))
                          / log(freq_base) * float(n_rot) * 0.5f) * 2.0f;
        float mscale = yarn_get_mscale_cp(freq_scale, attn_factor);

        // one thread per pair (2*pair, 2*pair+1) within the tail.
        for (uint pair = tid; pair * 2u < n_rot; pair += tg_size) {
            uint i0 = pair * 2u;
            float exponent  = float(i0) / float(n_rot);
            float freq_full = 1.0f / pow(freq_base, exponent);
            float freq_inter = freq_full / freq_scale;
            float ramp = yarn_ramp_mask_cp(low, high, int(i0));
            float freq = freq_inter * (1.0f - ramp * ext_factor)
                       + freq_full  * ramp * ext_factor;
            float theta = float(pos) * freq;
            float cos_t = cos(theta) * mscale;
            float sin_t = sin(theta) * mscale;
            if (backward) sin_t = -sin_t;
            float x0 = pooled_sh[tail0 + i0];
            float x1 = pooled_sh[tail0 + i0 + 1u];
            pooled_sh[tail0 + i0]      = x0 * cos_t - x1 * sin_t;
            pooled_sh[tail0 + i0 + 1u] = x0 * sin_t + x1 * cos_t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 4. write the finished emit row to comp_cache ──────────────────
    device float * out = comp + (args.comp_row0 + e) * hd;
    for (uint j = tid; j < hd; j += tg_size) {
        out[j] = pooled_sh[j];
    }
}
