// Phase 3 — MTP drafter input-mix broadcast-add.
//
// After (eproj = matmul(e_proj, enorm(token_embed))) and (hproj_hc =
// matvec_k(h_proj, hnorm_rows(prev_hc))), the final step of the MTP
// draft input mixer is:
//
//   mtp_input_hc[r, c] = eproj[c] + hproj_hc[r, c]
//
// for r in 0..n_hc, c in 0..n_embd. One thread per (r, c) cell. The
// kernel broadcasts the same eproj row across all n_hc HC rows.
//
// Reference: antirez `ds4_metal_add_tensor(mtp_input_hc, eproj_hc,
// hproj_hc)` at ds4.c:12291, plus the implicit eproj→eproj_hc repeat
// (ds4_metal_repeat_hc_tensor at ds4.c:12271). This kernel fuses the
// repeat + add into one pass; no eproj_hc scratch buffer needed.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_kernel_dsv4_mtp_input_mix_broadcast_add_f32(
        device const float * eproj       [[buffer(0)]],   // [n_embd]
        device const float * hproj_hc    [[buffer(1)]],   // [n_hc * n_embd]
        device       float * mtp_input_hc[[buffer(2)]],   // [n_hc * n_embd]
        constant     uint  & n_embd      [[buffer(3)]],
        constant     uint  & n_hc        [[buffer(4)]],
        uint                 tid         [[thread_position_in_grid]]) {
    const uint total = n_hc * n_embd;
    if (tid >= total) { return; }
    const uint c = tid % n_embd;
    mtp_input_hc[tid] = eproj[c] + hproj_hc[tid];
}
