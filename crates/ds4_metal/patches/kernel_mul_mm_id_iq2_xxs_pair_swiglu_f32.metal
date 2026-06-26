// ── DS4 port: fused gate+up+SwiGLU mm_id for iq2_xxs experts (prefill MoE) ──
// Ported from antirez kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16; the ONE
// difference is the output `dst_mid` is f32 (not f16) so our existing f32 down
// GEMM consumes it unchanged. Fuses what we run as 3 dispatches (gate mm_id +
// up mm_id + moe_swiglu_weight) into ONE: loads the activation tile once into
// `sb`, accumulates into BOTH mc_gate and mc_up (simdgroup_half8x8 tiles), and
// applies SwiGLU + route-weight inline, writing the [token,slot,d_ffn] mid.
kernel void kernel_mul_mm_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/QK_NL;

    device const block_iq2_xxs * xg =
        (device const block_iq2_xxs *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_iq2_xxs * xu =
        (device const block_iq2_xxs *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        const short sx_b = (tiitg%NL1);
        const short sy_b = (tiitg/NL1)/8;
        const short ly_b = (tiitg/NL1)%8;
        const short ib_b = 4*sx_b + sy_b;
        *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
            (half2x4)(*((device float2x4 *) y));

        half4x4 temp_gate;
        dequantize_iq2_xxs(xg, il, temp_gate);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_gate + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma[i%4], mc_gate[i]);
            }

            lsma_gate += 8*64;
            lsmb += 4*64;
        }

        half4x4 temp_up;
        dequantize_iq2_xxs(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_up = (sa + 4*64*(sgitg%2));
        lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_up + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_up[i], mb[i/4], ma[i%4], mc_up[i]);
            }

            lsma_up += 8*64;
            lsmb += 4*64;
        }

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + QK_NL - 1)/QK_NL : xg;
        xu = (il < 2) ? xu + (2 + QK_NL - 1)/QK_NL : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float *D = (device float *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = silu * u * route_weight;
        }
    }
}
