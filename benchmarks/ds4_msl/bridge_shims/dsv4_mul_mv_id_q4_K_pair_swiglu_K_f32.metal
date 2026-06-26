// Phase 2 MoE-K Option A — K-batched FUSED pair_swiglu for Q4_K experts.
//
// Mirror of upstream `kernel_mul_mv_id_q4_K_pair_swiglu_f32` (moe.metal:1160)
// with TWO targeted modifications to support K tokens in one dispatch:
//
//   1. Dispatch grid z-dim runs K * nei0 (instead of nei0=6). Each TG handles
//      one (iid1=token, idx=expert_slot) pair.
//   2. dst_mid and route_w offsets use (iid1 * nei0 + idx) instead of just idx,
//      so K tokens × 6 slots each get their own output row + route weight.
//
// dst_gate / dst_up scratch use the EXISTING address formula
// `(idx * ne0 + i12 * ne1 * ne0)`, which is already K-aware IF the host
// sets `ne1 = nei0 = 6` (instead of 1 in the K=1 production path). For K=1,
// ne1=6 reduces to idx*ne0 (i12=0) — backwards-compatible.
//
// dst_gate/dst_up MUST be allocated for K*6 slots (K times larger than
// K=1 production); they're transient scratch consumed only by the silu
// loop at the end.
//
// Reuses moe.metal-declared symbols (same translation unit):
//   ds4_metal_args_mul_mv_id, ds4_metal_dsv4_moe_swiglu_weight_args (structs)
//   ds4_metal_args_mul_mv (struct, used inside the q4_K impl)
//   block_q4_K, kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K> (template fn)
//   FC_mul_mv_nsg (function constant)
//
// moe.metal #undefs QK_K and N_R0_Q4_K at end-of-file so we redefine here.

#include <metal_stdlib>
using namespace metal;

#define DS4_QK_K        256
#define DS4_N_R0_Q4_K   2

kernel void ds4_kernel_mul_mv_id_q4_K_pair_swiglu_K_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,        // [K*6*ne0] scratch
        device       char * dst_up,          // [K*6*ne0] scratch
        device       char * dst_mid,         // [K*6, d_ffn] output
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig                [[threadgroup_position_in_grid]],
        ushort tiitg                [[thread_index_in_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    const uint linear_slot = (uint)(iid1 * args.nei0 + idx);

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    // dst_gate/dst_up scratch addressing — with host setting ne1=6, this
    // reduces to (i12*6 + idx)*ne0 = linear_slot*ne0. At K=1, i12=0 →
    // idx*ne0 (matches K=1 production layout).
    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<DS4_N_R0_Q4_K>(
        args0, src0_gate_cur, src1_cur, dst_gate_cur,
        shmem, tgpig, tiisg, sgitg);
    kernel_mul_mv_q4_K_f32_impl<DS4_N_R0_Q4_K>(
        args0, src0_up_cur, src1_cur, dst_up_cur,
        shmem, tgpig, tiisg, sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * DS4_N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32   = (device float *)dst_up_cur;
    // K-BATCHED: dst_mid + route_w use linear_slot (= iid1*nei0 + idx).
    device float *mid_f32  = (device float *)(dst_mid + (uint64_t)linear_slot * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)linear_slot * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < DS4_N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }
    (void)tiitg;
}
