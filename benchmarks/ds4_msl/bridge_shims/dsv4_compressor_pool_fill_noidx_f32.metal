// Bridge shim for `ds4_kernel_dsv4_compressor_pool_fill_noidx_f32`.
//
// STAGE 1 companion to `compressor_prefill_noidx`. After the fused prefill
// kernel emits the chunk's compressed rows, the TRAILING partial group (the
// `rem = k_positions - n_emit*ratio` positions after the last emit, which do
// NOT themselves emit) must still land in the per-layer compressor STATE pool
// so the SUBSEQUENT per-position decode (the `feed` token + the next chunk's
// first group) pools across the chunk boundary correctly. This mirrors
// antirez `ds4_gpu_encode_compressor_set_rows_projected` (ds4_metal.m:14068,
// the non-ratio4 `rem != 0` branch): write kv[r] → state_kv[pos%ratio] and
// score[r] + APE[pos%ratio] → state_score[pos%ratio] for the rem rows, in ONE
// dispatch (no per-position store_one).
//
// coff == 1 ⇒ width == head_dim, dst_row == pos % ratio (the ratio==4 two-window
// offset does NOT apply — this kernel is noidx only).
//
// Encoder contract (grid = rem rows × width threads):
//   buffer(0): kv     const float* [k_positions × width]  (batched proj)
//   buffer(1): sc     const float* [k_positions × width]  (batched proj)
//   buffer(2): ape    const float* [width × ratio]        (ape[j*ratio + r])
//   buffer(3): state_kv    float*  [ratio × width]  (persistent pool)
//   buffer(4): state_score float*  [ratio × width]
//   buffer(5): args   constant ds4_compressor_pool_fill_args

#include <metal_stdlib>
using namespace metal;

struct ds4_compressor_pool_fill_args {
    uint width;     // == head_dim
    uint ratio;
    uint pos0;      // chunk_start
    uint first;     // index of first trailing position within the chunk (cutoff)
    uint rem;       // number of trailing positions to store
};

kernel void ds4_kernel_dsv4_compressor_pool_fill_noidx_f32(
    device const float * kv          [[buffer(0)]],
    device const float * sc          [[buffer(1)]],
    device const float * ape         [[buffer(2)]],
    device       float * state_kv    [[buffer(3)]],
    device       float * state_score [[buffer(4)]],
    constant ds4_compressor_pool_fill_args & args [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y;        // which trailing position (0..rem)
    uint j = gid.x;        // column
    if (r >= args.rem || j >= args.width) return;
    uint width = args.width;
    uint src_row = args.first + r;            // row in the batched projections
    uint pos = args.pos0 + src_row;
    uint pos_mod = pos % args.ratio;
    uint dst = pos_mod * width + j;           // pool row pos%ratio (coff==1)
    float ape_v = ape[j * args.ratio + pos_mod];
    state_kv[dst]    = kv[src_row * width + j];
    state_score[dst] = sc[src_row * width + j] + ape_v;
}
