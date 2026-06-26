// Bridge shim for `ds4_dsv4_qkv_rms_norm_f32_4`.
//
// Composes two RMSNorms (q-lora row + kv-lora row) into one dispatch.
// Matches the encoder contract in
// `crates/ds4_metal/src/macos.rs::qkv_rms_norm_rows_impl`:
//
//   buffer(0): qr      const float* (n_lora_q)
//   buffer(1): gamma_q const float* (n_lora_q)
//   buffer(2): qr_out  float*       (n_lora_q)
//   buffer(3): kv_raw  const float* (n_lora_kv)
//   buffer(4): gamma_kv const float*(n_lora_kv)
//   buffer(5): kv_out  float*       (n_lora_kv)
//   buffer(6): n_q4    uint32      = n_lora_q / 4
//   buffer(7): n_kv4   uint32      = n_lora_kv / 4
//   buffer(8): n_lora_q  uint32
//   buffer(9): n_lora_kv uint32
//   buffer(10): n_rot   uint32
//   buffer(11): eps_q   float
//   buffer(12): eps_kv  float
//
// Dispatch: 1 threadgroup, max(n_q4, n_kv4) threads (≤1024).
//
// Each thread t with t < n_q4 contributes 4 lanes to the qr partial
// sum; each thread t with t < n_kv4 contributes 4 lanes to the kv
// partial sum. The encoder passes only the KV row, so n_rot is
// accepted for ABI compatibility but is not written as a tail here.
//
// Scratch uses two threadgroup floats for the row sums-of-squares.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_qkv_rms_norm_f32_4(
    device const float * qr        [[buffer(0)]],
    device const float * gamma_q   [[buffer(1)]],
    device float       * qr_out    [[buffer(2)]],
    device const float * kv_raw    [[buffer(3)]],
    device const float * gamma_kv  [[buffer(4)]],
    device float       * kv_out    [[buffer(5)]],
    constant uint      & n_q4      [[buffer(6)]],
    constant uint      & n_kv4     [[buffer(7)]],
    constant uint      & n_lora_q  [[buffer(8)]],
    constant uint      & n_lora_kv [[buffer(9)]],
    constant uint      & n_rot     [[buffer(10)]],
    constant float     & eps_q     [[buffer(11)]],
    constant float     & eps_kv    [[buffer(12)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    threadgroup float sumsq_q;
    threadgroup float sumsq_kv;
    threadgroup float inv_q;
    threadgroup float inv_kv;

    // Partial sums into private accumulators.
    float pq = 0.0f;
    float pkv = 0.0f;
    if (tid < n_q4) {
        float4 v = ((device const float4*)qr)[tid];
        pq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    if (tid < n_kv4) {
        float4 v = ((device const float4*)kv_raw)[tid];
        pkv = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Single-threadgroup reduction via atomic-free threadgroup mem:
    // pile partials into shmem array, barrier, then thread 0 sums.
    threadgroup float scratch_q[1024];
    threadgroup float scratch_kv[1024];
    scratch_q[tid]  = pq;
    scratch_kv[tid] = pkv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sq = 0.0f, skv = 0.0f;
        for (uint i = 0; i < tg_size; ++i) {
            sq  += scratch_q[i];
            skv += scratch_kv[i];
        }
        float mean_q  = sq  / (float)n_lora_q;
        float mean_kv = skv / (float)n_lora_kv;
        sumsq_q  = sq;
        sumsq_kv = skv;
        inv_q  = 1.0f / sqrt(mean_q  + eps_q);
        inv_kv = 1.0f / sqrt(mean_kv + eps_kv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply normalization * gamma.
    if (tid < n_q4) {
        float4 v = ((device const float4*)qr)[tid];
        float4 g = ((device const float4*)gamma_q)[tid];
        ((device float4*)qr_out)[tid] = v * inv_q * g;
    }
    if (tid < n_kv4) {
        float4 v = ((device const float4*)kv_raw)[tid];
        float4 g = ((device const float4*)gamma_kv)[tid];
        ((device float4*)kv_out)[tid] = v * inv_kv * g;
    }

    (void)n_rot; (void)sumsq_q; (void)sumsq_kv;
}
