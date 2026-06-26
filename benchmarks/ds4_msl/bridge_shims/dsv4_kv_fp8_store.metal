// Bridge shim for `ds4_dsv4_kv_fp8_store`.
//
// Writes one slot of the raw KV cache: per-64-block max-scaled E4M3FN
// quantize on the n_nope prefix followed by an FP16 round-trip, plus an
// FP16 round-trip on the n_rot rope tail. This matches antirez
// `kernel_dsv4_kv_fp8_store_f32` (upstream dsv4_kv.metal) and the CPU
// reference `ds4_fp8_kv_quantize_row_inplace` + `f16_round_trip_f32`
// (ds4_engine::attn_dispatch) bit-for-bit, so the encoder's persistent KV
// write is correct on its own — the CPU "slot correction" pass that used to
// overwrite this output is no longer needed.
//
// The earlier shim did an *unscaled* per-element e4m3 round-trip with no
// FP16 step, which diverged from antirez and compounded to ~25% rel error
// in cur_hc across 43 layers; that is why the CPU correction existed.
//
// Reuses `dsv4_e4m3fn_dequant` (defined in dsv4_kv.metal, concatenated
// ahead of this shim in ANTIREZ_BRIDGE_SOURCE — same translation unit).
//
// Encoder contract:
//   buffer(0): cache    float*       (raw_cap * row floats)
//   buffer(1): row      const float* (row floats)
//   buffer(2): n_nope   uint32
//   buffer(3): n_rot    uint32
//   buffer(4): slot     uint32
//
// Dispatch: 1 threadgroup, max(n_nope/64, 1) threads (<=1024). Each thread
// owns whole 64-element blocks (strided by tg_size) and computes that
// block's amax serially — no threadgroup reduction or barriers needed, and
// the strided loop covers a non-64-aligned tail block as well.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_kv_fp8_store(
    device float       * cache     [[buffer(0)]],
    device const float * row       [[buffer(1)]],
    constant uint      & n_nope    [[buffer(2)]],
    constant uint      & n_rot     [[buffer(3)]],
    constant uint      & slot      [[buffer(4)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    uint row_stride = n_nope + n_rot;
    uint base = slot * row_stride;

    // Per-64-block scaled E4M3FN on the n_nope prefix, then FP16 round-trip.
    // scale = 2^ceil(log2(amax / 448)); amax floored at 1e-4. Matches the
    // CPU `ds4_fp8_kv_quantize_row_inplace` block loop.
    for (uint off = tid * 64u; off < n_nope; off += tg_size * 64u) {
        uint end = min(off + 64u, n_nope);
        float amax = 1.0e-4f;
        for (uint i = off; i < end; ++i) {
            amax = max(amax, fabs(row[i]));
        }
        float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        for (uint i = off; i < end; ++i) {
            float q = dsv4_e4m3fn_dequant(clamp(row[i] / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            cache[base + i] = (float)((half)q);
        }
    }

    // FP16 round-trip on the n_rot tail.
    for (uint i = tid; i < n_rot; i += tg_size) {
        cache[base + n_nope + i] = (float)((half)row[n_nope + i]);
    }
}

// K-merged variant: stores K consecutive slots [base_slot, base_slot+K) in a
// SINGLE dispatch, eliminating the K separate compute encoders the host loop
// used to issue (one per slot). Each threadgroup (grid.x = k) owns one slot;
// identical per-slot math to `ds4_dsv4_kv_fp8_store`. The K rows are packed
// contiguously in `rows` at stride row_stride (= n_nope + n_rot).
//
// Encoder contract:
//   buffer(0): cache      float*        (raw_cap * row_stride floats)
//   buffer(1): rows       const float*  (K * row_stride floats)
//   buffer(2): n_nope     uint32
//   buffer(3): n_rot      uint32
//   buffer(4): base_slot  uint32
//
// Dispatch: grid (K, 1, 1) threadgroups × (max(n_nope/64,1), 1, 1) threads.
kernel void ds4_dsv4_kv_fp8_store_k(
    device float       * cache     [[buffer(0)]],
    device const float * rows      [[buffer(1)]],
    constant uint      & n_nope    [[buffer(2)]],
    constant uint      & n_rot     [[buffer(3)]],
    constant uint      & base_slot [[buffer(4)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint k                         [[threadgroup_position_in_grid]])
{
    uint row_stride = n_nope + n_rot;
    uint base = (base_slot + k) * row_stride;        // destination slot
    device const float * row = rows + k * row_stride; // this k-position's row

    for (uint off = tid * 64u; off < n_nope; off += tg_size * 64u) {
        uint end = min(off + 64u, n_nope);
        float amax = 1.0e-4f;
        for (uint i = off; i < end; ++i) {
            amax = max(amax, fabs(row[i]));
        }
        float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        for (uint i = off; i < end; ++i) {
            float q = dsv4_e4m3fn_dequant(clamp(row[i] / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            cache[base + i] = (float)((half)q);
        }
    }

    for (uint i = tid; i < n_rot; i += tg_size) {
        cache[base + n_nope + i] = (float)((half)row[n_nope + i]);
    }
}
