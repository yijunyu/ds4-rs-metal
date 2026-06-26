// Dense Q4_0 matrix-vector multiply (DS4_LOW_RAM variant).
//
// Same structure as `kernel_mul_mv_q8_0_f32` (dense.metal): same
// `ds4_metal_args_mul_mv` arg struct, same `helper_mv_reduce_and_write`
// reduction, same `FC_mul_mv_nsg` simdgroup geometry — but reads a
// `block_q4_0` (18 B / 32 weights: a half scale + 16 nibble bytes) at ~half
// the weight bytes of Q8_0 (34 B / 32). Used for the low-RAM decode path,
// where the model's Q8_0 projection / lm_head tensors are re-quantized to
// Q4_0 at load (drop the Q8_0 copy → ~half the resident weight bytes).
//
// Layout (ggml q4_0): a 32-weight block stores the low nibbles as weights
// [0,16) and the high nibbles as weights [16,32); weight value =
// ((nibble & 0xF) - 8) * d. The host re-quantizer (macos.rs requant_q8_to_q4_0)
// writes exactly this layout, so the kernel and requantizer must stay in sync.
//
// NOT a throughput win — the nibble unpack is ALU-bound, so it can't beat the
// nsg-tuned Q8_0 kernel despite reading fewer bytes (see the decode notes). It
// exists purely to cut resident RAM. The shim is concatenated after
// dense.metal, so it reuses that file's `ds4_metal_args_mul_mv`,
// `helper_mv_reduce_and_write`, `FC_mul_mv_nsg`, `N_SIMDWIDTH`, `N_R0_Q8_0`
// and `FOR_UNROLL` definitions.

#define QK4_0 32

typedef struct {
    half    d;                 // delta (scale)
    uint8_t qs[QK4_0 / 2];     // 16 bytes: 32 packed 4-bit weights
} block_q4_0;

template<short NR0, typename args_t>
void ds4_kernel_mul_mv_q4_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK4_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q4_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q4_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    // 4 simdgroup lanes per block (NW/NQ); each lane owns 4 low nibbles
    // (weights [il*4, il*4+4)) and 4 high nibbles (weights [16+il*4, 16+il*4+4)).
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float ylo[4];
    float yhi[4];

    device const float * yb_lo = y + ib0*QK4_0 + il*4;
    device const float * yb_hi = y + ib0*QK4_0 + 16 + il*4;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < 4; ++i) {
            ylo[i] = yb_lo[i];
            yhi[i] = yb_hi[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const uint8_t * qs = ax[row][ib].qs + il*4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint8_t b = qs[i];
                sumq += ((float)(b & 0x0F) - 8.0f) * ylo[i];
                sumq += ((float)(b >>   4) - 8.0f) * yhi[i];
            }

            sumf[row] += sumq*(float)ax[row][ib].d;
        }

        yb_lo += NSG*NQ*QK4_0;
        yb_hi += NSG*NQ*QK4_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

kernel void ds4_kernel_mul_mv_q4_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_kernel_mul_mv_q4_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
