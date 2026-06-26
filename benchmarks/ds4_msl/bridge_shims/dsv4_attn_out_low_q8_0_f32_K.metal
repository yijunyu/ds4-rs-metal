// K-batched attention output-projection stage-1 (per-group Q8_0 matvec).
//
// Mirror of upstream `kernel_dsv4_attn_out_low_q8_0_f32` (moe.metal:842) that
// processes K speculative-decode positions in ONE dispatch instead of the host
// issuing K separate dispatches (one per K-position). On the K-position
// verifier path this collapses K dispatches/layer → 1; the per-dispatch cost
// (~150µs each, measured via kv_fp8_store_merge_microbench) is the dominant
// verify-chain overhead, and only a single grid-batched dispatch removes it
// (consolidating encoders does nothing — same-buffer write hazards serialize).
//
// K-batching uses the SAME tgpig.z / nei0 decomposition the existing
// mul_mv_id_*_pair_swiglu_K shims use:
//   - Host dispatches grid.z = K * nei0 (nei0 = n_groups) instead of nei0.
//   - iid1 = tgpig.z / nei0  → K-position k   (was always 0 in the K=1 kernel)
//   - idx  = tgpig.z % nei0  → group index
//
// Input/output addressing per (k, group):
//   src1 (heads_k [K, n_groups, group_dim]): + i11*nb11 + i12*nb12, where
//     i11=group, nb11=group_dim*4, i12=k, nb12=heads_per_k_bytes
//     (= n_groups*group_dim*4) → element k*n_groups*group_dim + group*group_dim.
//     Identical to the upstream formula (i12*nb12 already supplies the k offset).
//   dst (attn_low_k [K, out_low_dim]): the ONLY change vs upstream — the
//     per-K stride is out_low_dim (= ne1), NOT ne1*ne0. So the k term is
//     iid1*ne1 instead of upstream's generic-batch i12*ne1*ne0.
//
// All inner compute (kernel_mul_mv_q8_0_f32_impl) is byte-identical to the
// K=1 kernel; the K-batching is pure outer parallelism (K× more threadgroups).
//
// Reuses moe.metal/dense.metal-declared types + impl (concatenated before this
// shim in ANTIREZ_BRIDGE_SOURCE — same translation unit):
//   ds4_metal_args_mul_mv_id, ds4_metal_args_mul_mv (structs)
//   kernel_mul_mv_q8_0_f32_impl (template), N_R0_Q8_0 (macro)

#include <metal_stdlib>
using namespace metal;

kernel void ds4_dsv4_attn_out_low_q8_0_f32_k(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;   // K-position k (z runs 0..K*nei0)
    const int idx  = tgpig.z%args.nei0;   // group index

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    // K-batched dst: per-K stride is ne1 (= out_low_dim), per-group is ne0.
    device       char * dst_cur  = dst   + ((int64_t)idx*args.ne0 + (int64_t)iid1*args.ne1)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}
