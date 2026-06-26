// MoE mm_id indirect-dispatch grid builder.
//
// The mm_id GEMM dispatches grid = [grid_x, grid_y, grid_z=n_experts] where grid_x must
// cover the LARGEST expert's tokens-per-expert (tpe). The static worst case is tpe<=K, but
// the ACTUAL max for a balanced top-6 router is ~K*6/n_experts (<<K), so a static grid_x=
// ceil(K/32) over-provisions ~n_experts/6x in early-exit threadgroups (a major prefill cost,
// measured ~+18% @K=3000). This kernel computes grid_x = ceil(max_e tpe[e] / NR1) on the GPU
// so the GEMM's indirect dispatch is sized to the run's REAL distribution — always correct
// (covers the true max, never drops tokens) AND tight (no over-dispatch). Run as ONE
// threadgroup of 256 threads. Writes a MTLDispatchThreadgroupsIndirectArguments triple.
#include <metal_stdlib>
using namespace metal;

kernel void ds4_kernel_moe_grid_args(
    device const uint* tpe        [[buffer(0)]],   // [n_experts] tokens-per-expert (from map0)
    constant uint&     n_experts  [[buffer(1)]],
    constant uint&     nr1        [[buffer(2)]],   // NR1 = 32 (token tile width)
    constant uint&     grid_y     [[buffer(3)]],   // ceil(d_out / NR0), passed through
    constant uint&     grid_z     [[buffer(4)]],   // n_experts, passed through
    device uint*       args       [[buffer(5)]],   // out: [grid_x, grid_y, grid_z]
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup uint sh[256];
    uint mx = 0u;
    for (uint e = tid; e < n_experts; e += tgsz) { mx = max(mx, tpe[e]); }
    sh[min(tid, 255u)] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // tree reduction over up to 256 lanes
    for (uint s = tgsz >> 1; s > 0u; s >>= 1) {
        if (tid < s) { sh[tid] = max(sh[tid], sh[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        uint gx = (sh[0] + nr1 - 1u) / nr1;
        args[0] = max(gx, 1u);   // grid_x — never 0 (a fully-empty MoE still dispatches 1)
        args[1] = grid_y;
        args[2] = grid_z;
    }
}
