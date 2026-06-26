// K=1 production override for `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`
// (upstream moe.metal:959). This shim shadows the upstream kernel: because it
// defines the final `ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` host_name,
// build.rs skips renaming the upstream raw kernel (which keeps its
// `kernel_…` name and goes unused), and the host dispatch in
// `moe_routed_step_encode` binds THIS kernel.
//
// Only one change vs upstream: the `dst_gate` / `dst_up` device-memory stores
// are gated behind the `FC_moe_write_pre_swiglu` function constant (false by
// default). Those writes are DEAD in the decode path — `gate`/`up` are
// consumed in-register by the SwiGLU and the sum6 down-projection reads only
// `dst_mid`. Skipping them removes 2 × 6 × d_ffn × 4 B of pure write traffic
// per call. Host sets the constant via `DS4_MOE_WRITE_PRE_SWIGLU` (=1 restores
// the stores for debugging the pre-swiglu gate/up activations).
//
// The signature is byte-for-byte the upstream one (dst_gate at buffer 5,
// dst_up at 6, dst_mid at 7) so no host binding change is required; the
// unused dst_gate/dst_up buffers stay bound and harmless.
//
// Reuses moe.metal-declared types (concatenated before this shim in the same
// translation unit): ds4_metal_args_mul_mv_id, ds4_metal_dsv4_moe_swiglu_weight_args,
// block_iq2_xxs, ds4_metal_iq2xxs_grid, ds4_metal_ksigns_iq2xs,
// ds4_metal_kmask_iq2xs, FC_mul_mv_nsg, and the FC_MUL_MV preamble macro.

#include <metal_stdlib>
using namespace metal;

// moe.metal #undefs QK_K / N_R0_IQ2_XXS at end-of-file; re-define the prefixed
// constants this shim uses (mirrors the K-batched shim).
#define DS4_QK_K            256
#define DS4_N_R0_IQ2_XXS    4

// When false (decode default) the gate/up dead stores below are dropped. Only
// this shim references FC_MUL_MV + 2; the host (`moe_routed_step_encode`) sets
// it and keys the specialized pipeline on it.
constant bool FC_moe_write_pre_swiglu [[function_constant(FC_MUL_MV + 2)]];

kernel void ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
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
            // Vectorized dequant: replace the 8 scalar FMAs + 8 sign branches per
            // l (×2 gate/up) with float4 ops — the same win the q4_0 experiment
            // measured. Each l covers 8 weights = two float4 lanes. The ±1 sign is
            // derived branchlessly from the bits of the sign byte (bit j ⇔ the old
            // `sign & kmask[j]`, kmask = {1,2,4,…,128}). Same arithmetic, reassoc.
            const uint4 bit_lo = uint4(0u, 1u, 2u, 3u);
            const uint4 bit_hi = uint4(4u, 5u, 6u, 7u);
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint signu = ssigns[(aux32u >> 7 * l) & 127];

                const float4 v0 = float4(yl[8*l+0], yl[8*l+1], yl[8*l+2], yl[8*l+3]);
                const float4 v1 = float4(yl[8*l+4], yl[8*l+5], yl[8*l+6], yl[8*l+7]);
                const float4 g0 = float4(gridg[0], gridg[1], gridg[2], gridg[3]);
                const float4 g1 = float4(gridg[4], gridg[5], gridg[6], gridg[7]);
                const float4 u0 = float4(gridu[0], gridu[1], gridu[2], gridu[3]);
                const float4 u1 = float4(gridu[4], gridu[5], gridu[6], gridu[7]);

                const float4 sg0 = select(float4(1.f), float4(-1.f), ((uint4(signg) >> bit_lo) & 1u) != 0u);
                const float4 sg1 = select(float4(1.f), float4(-1.f), ((uint4(signg) >> bit_hi) & 1u) != 0u);
                const float4 su0 = select(float4(1.f), float4(-1.f), ((uint4(signu) >> bit_lo) & 1u) != 0u);
                const float4 su1 = select(float4(1.f), float4(-1.f), ((uint4(signu) >> bit_hi) & 1u) != 0u);

                sg += dot(v0 * g0, sg0) + dot(v1 * g1, sg1);
                su += dot(v0 * u0, su0) + dot(v1 * u1, su1);
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

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + (uint64_t)idx * act.weight_stride);

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
            // Dead stores (default-off): unused downstream — sum6 reads dst_mid.
            if (FC_moe_write_pre_swiglu) {
                dst_gate_f32[out_row] = gate;
                dst_up_f32[out_row] = up;
            }
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}
