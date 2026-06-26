// Phase 3 — MTP drafter output_hc_weights (step 3 of the output head).
//
// Computes the per-HC-row output weights from the matmul-of-hc_head_fn
// projection:
//
//   for h in 0..n_hc:
//     v = pre[h] * scale + base[h]
//     v = sigmoid(v)
//     out[h] = v + eps        (additive eps trust adjustment)
//
// Mirrors antirez `ds4_metal_output_hc_weights_tensor` (ds4_metal.m:13835),
// which dispatches mul/add/sigmoid/scale as separate kernels with the same
// underlying math. Fuses them into one kernel since the inputs are tiny
// (n_hc = 4 floats for DS4 V4 Flash).
//
// Reference: antirez output head step 3 in metal_graph_encode_output_head_mtp
// (ds4.c:9601-9608).

#include <metal_stdlib>
using namespace metal;

kernel void ds4_kernel_dsv4_mtp_output_hc_weights_f32(
        device const float * pre   [[buffer(0)]],   // [n_hc] hc_head_fn @ flat_hc
        device const float * scale [[buffer(1)]],   // [1] hc_head_scale scalar
        device const float * base  [[buffer(2)]],   // [n_hc] hc_head_base
        device       float * out   [[buffer(3)]],   // [n_hc] output_weights
        constant     uint  & n_hc  [[buffer(4)]],
        constant     float & eps   [[buffer(5)]],
        uint                 tid   [[thread_position_in_grid]]) {
    if (tid >= n_hc) { return; }
    float v = pre[tid] * scale[0] + base[tid];
    v = 1.0f / (1.0f + exp(-v));     // sigmoid
    out[tid] = v + eps;
}
