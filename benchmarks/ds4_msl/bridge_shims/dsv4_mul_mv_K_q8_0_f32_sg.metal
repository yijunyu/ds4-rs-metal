// Phase 1 — K-position Q8_0 matvec kernel (simdgroup-matrix tiled) used by the
// speculative-decoding path. Validated in Phase 0 (drivers/mul_mv_K_simdgroup_check.swift):
//   K=1: 154us  K=8: 143us (ratio 0.93)  CORRECTNESS: max_abs=6.4e-5 vs naive K=1
//
// Structure: NR0=32 rows × K cols output tile per TG. NSG=4 simdgroups, each
// owns 8 rows × 8 K-cols simdgroup_float8x8 accumulator. Per NK=32 contraction
// (one Q8_0 block): dequant weights → SA[32][32] half tile; pack activations →
// SB[32][8] half tile; 4 × simdgroup_multiply_accumulate. Final tile staged to
// threadgroup, then strided write to dst[K][d_out].
//
// FC indices: 1500 (NSG, unused but reserved) and 1501 (K, the spec-decode K
// positions). Chosen above all upstream FC bases (FLASH_ATTN_EXT_*=100..500,
// MUL_MV=600, MUL_MM=700, UNARY=1200, BIN=1300, SUM_ROWS=1400). K is supported
// in {1, 2, 4, 8} (output tile has K-cols up to 8 padded; lanes beyond K are zero).
//
// Note: this shim is appended to ANTIREZ_BRIDGE_SOURCE after preamble + dense.metal
// so QK8_0, struct block_q8_0, and the metal_stdlib include are already in scope.
// Do NOT redefine — that triggers MSL compile errors.

#include <metal_simdgroup_matrix>

constant short FC_mul_mv_K_nsg [[function_constant(1500)]];
constant short FC_mul_mv_K_K   [[function_constant(1501)]];

kernel void ds4_kernel_mul_mv_K_q8_0_f32_sg(
        constant uint & d_in        [[buffer(0)]],
        constant uint & d_out       [[buffer(1)]],
        device const char * src0_w  [[buffer(2)]],
        device const float * src1_x [[buffer(3)]],
        device float * dst          [[buffer(4)]],
        threadgroup char * shmem    [[threadgroup(0)]],
        uint3  tgpig                [[threadgroup_position_in_grid]],
        ushort tiitg                [[thread_index_in_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {

    constexpr short NR0 = 32;
    constexpr short NK  = 32;
    constexpr short NSG = 4;
    const short K = FC_mul_mv_K_K;
    (void)FC_mul_mv_K_nsg;
    (void)tiisg;

    // FLOAT tiles (not half): half-rounding the dequantized weight/activation
    // drifts ~1e-3/row from the per-token float matvec_q8_0, which cascades to
    // chunk-prefill @3000 word-salad. Float matches the matvec oracle.
    threadgroup float * SA = (threadgroup float *)(shmem);
    threadgroup float * SB = (threadgroup float *)(shmem + NR0 * NK * sizeof(float));

    const uint nb = d_in / QK8_0;
    const uint r0 = tgpig.x * NR0;

    simdgroup_float8x8 mc = make_filled_simdgroup_matrix<float, 8>(0.f);

    const short row_sa = tiitg / 4;
    const short colg_sa = tiitg % 4;

    for (uint ib = 0; ib < nb; ++ib) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint row_global = r0 + (uint)row_sa;
            device const block_q8_0 * blk =
                (device const block_q8_0 *)(src0_w) + (uint64_t)row_global * nb + ib;
            const half d = blk->d;
            device const int8_t * qs = blk->qs + (short)colg_sa * 8;
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                SA[row_sa * NK + colg_sa * 8 + i] = (float)qs[i] * (float)d;
            }
        }

        {
            const short row_sb = tiitg / 4;
            const short col0   = (tiitg % 4) * 2;
            #pragma unroll
            for (short dc = 0; dc < 2; ++dc) {
                short kk = col0 + dc;
                float v = 0.f;
                if (kk < K) {
                    const uint base = (uint)kk * d_in + ib * QK8_0 + (uint)row_sb;
                    v = src1_x[base];
                }
                SB[row_sb * 8 + kk] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * lsma = SA + (short)sgitg * 8 * NK;
        threadgroup const float * lsmb = SB;
        simdgroup_float8x8 ma;
        simdgroup_float8x8 mb;
        #pragma unroll
        for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_load(ma, lsma + ik * 8, NK, 0, false);
            simdgroup_load(mb, lsmb + ik * 8 * 8, 8, 0, false);
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * SC = (threadgroup float *)shmem;
    simdgroup_store(mc, SC + (short)sgitg * 8 * 8, 8, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_count = (uint)NR0 * (uint)K;
    for (uint t = tiitg; t < out_count; t += NSG * 32u) {
        const uint row = t / (uint)K;
        const uint kk  = t % (uint)K;
        const uint sg  = row / 8u;
        const uint lr  = row % 8u;
        const float v  = SC[sg * 64u + lr * 8u + kk];
        if (r0 + row < d_out) {
            dst[kk * d_out + r0 + row] = v;
        }
    }
}
