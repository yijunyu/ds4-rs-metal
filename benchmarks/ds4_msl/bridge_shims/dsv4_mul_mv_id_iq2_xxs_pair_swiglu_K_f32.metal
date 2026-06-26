// Phase 2 MoE-K Option A — K-batched FUSED pair_swiglu for iq2_xxs experts.
//
// Mirror of upstream `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (moe.metal:959)
// with TWO targeted modifications to support K tokens in one dispatch:
//
//   1. Dispatch grid z-dim runs K * nei0 (instead of nei0=6). Each TG handles
//      one (iid1=token, idx=expert_slot) pair. iid1 ranges 0..K-1; idx 0..5.
//   2. dst_mid and route_w offsets use (iid1 * nei0 + idx) instead of just idx,
//      so K tokens × 6 slots each get their own output row + route weight.
//
// All compute (iq2_xxs grid/sign dequant, fused gate+up matvec, silu*up*weight)
// is byte-identical to the K=1 kernel. The K-batching comes from outer
// parallelism (K× more TGs) without changing inner-tile work per TG.
//
// dst_gate / dst_up writes from the K=1 kernel are DROPPED here — they're
// unused downstream in production (the sum6 kernel reads only dst_mid).
//
// Reuses moe.metal-declared types (concatenated before this shim in
// ANTIREZ_BRIDGE_SOURCE — same translation unit):
//   ds4_metal_args_mul_mv_id           (struct)
//   ds4_metal_dsv4_moe_swiglu_weight_args (struct)
//   block_iq2_xxs                      (struct)
//   ds4_metal_iq2xxs_grid              (array)
//   ds4_metal_ksigns_iq2xs             (array)
//   ds4_metal_kmask_iq2xs              (array)
//   FC_mul_mv_nsg                      (function_constant)
//   DS4_N_R0_IQ2_XXS, QK_K                 (macros)

#include <metal_stdlib>
using namespace metal;

// moe.metal #undefs QK_K and DS4_N_R0_IQ2_XXS at end-of-file. The `static
// constant` arrays (ds4_metal_iq2xxs_grid etc.) remain accessible by their
// fully-prefixed names — moe.metal's macro aliases (iq2xxs_grid etc.) are
// gone, but the underlying symbols persist in this translation unit.
#define DS4_QK_K            256
#define DS4_N_R0_IQ2_XXS    4

kernel void ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_K_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig                [[threadgroup_position_in_grid]],
        ushort tiitg                [[thread_index_in_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;

    // Decode (iid1=token, idx=slot) from tgpig.z. For K=1, iid1 always = 0.
    // For K-batched, iid1 ∈ 0..K-1, idx ∈ 0..nei0-1 (5).
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    // Linear row index in the K-batched dst_mid / weights buffers.
    const uint linear_slot = (uint)(iid1 * args.nei0 + idx);

    // Selected expert id for this (token, slot) pair.
    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];

    // Activation row for THIS token. gate/up share the activation across the
    // 6 slots of the token (`i11 = idx % args.ne11`; for ne11=1 broadcast).
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / DS4_QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * DS4_N_R0_IQ2_XXS;
    const int nb32 = nb * (DS4_QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[DS4_N_R0_IQ2_XXS] = {0.f};
    float sumu[DS4_N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (DS4_QK_K / 32);
        const int ib  = ib32 % (DS4_QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < DS4_N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }
        y4 += 32 * 32;
    }

    // K-BATCHED OUTPUT ADDRESSING: incorporate iid1 (token) into the linear
    // slot offset. K=1 (iid1==0) reduces to the K=1 production layout.
    device float *dst_mid_f32 =
        (device float *)(dst_mid + (uint64_t)linear_slot * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + (uint64_t)linear_slot * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < DS4_N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }
    (void)tiitg;
}
