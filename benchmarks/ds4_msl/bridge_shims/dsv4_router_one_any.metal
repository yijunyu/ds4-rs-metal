// Expert-count- and scale-generic variants of the one-token decode router
// kernels (dsv4_misc.metal:154/179). The host has always bound num_experts /
// scale / min_sum at buffers 3-5 of router_weights_one — the upstream kernel
// ignores them (Flash 256/1.5 hardcode). These shims honor them so PRO
// (n_experts=384, expert_weights_scale=2.5) routes correctly.
kernel void ds4_dsv4_router_weights_one_any(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        constant uint  &k_used,
        constant float &scale,
        constant float &min_sum,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= k_used) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < k_used; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, min_sum);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * scale;
}

// Bitonic top-6 select over n_experts (any count ≤ threadgroup width; pad
// lanes score -inf). Dispatch with npow2 = next_pow2(n_experts) threads and
// 2*npow2*4 bytes threadgroup memory. args layout matches
// ds4_metal_args_dsv4_router_select_one with n_experts appended.
struct ds4_args_router_select_one_any {
    uint has_bias;
    uint hash_mode;
    uint use_token_buffer;
    uint token;
    uint hash_rows;
    uint n_experts;
};

kernel void ds4_dsv4_router_finalize_one_any(
        constant ds4_args_router_select_one_any & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    const uint n = ntg; // power of two ≥ args.n_experts

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + n);
    if (tid < args.n_experts) {
        const float p = probs[tid];
        sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    } else {
        sel_scores[tid] = -INFINITY;
    }
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= n; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── Batched (K-token) router finalize: row = tgpig indexes the token, so grid.x=K
// processes all K tokens in ONE dispatch (collapses the per-token router_finalize
// loop — the dominant MoE-stage dispatch hotspot). probs [K,n_experts] and the flat
// selected [K,6] are offset by row; bias [n_experts] is shared. Non-hash (router) only.
kernel void ds4_dsv4_router_finalize_one_any_k(
        constant ds4_args_router_select_one_any & args,
        device const float *probs,
        device const float *bias,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]],
        uint ntg [[threads_per_threadgroup]]) {
    const uint n = ntg; // power of two ≥ args.n_experts
    device const float *probs_row = probs + (uint64_t)row * (uint64_t)args.n_experts;
    device int32_t     *sel_row   = selected + (uint64_t)row * 6u;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + n);
    if (tid < args.n_experts) {
        const float p = probs_row[tid];
        sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    } else {
        sel_scores[tid] = -INFINITY;
    }
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                if ((tid & k) == 0) {
                    if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                        const int32_t tmp = idx[tid]; idx[tid] = idx[other]; idx[other] = tmp;
                    }
                } else {
                    if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                        const int32_t tmp = idx[tid]; idx[tid] = idx[other]; idx[other] = tmp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < 6) {
        sel_row[tid] = idx[tid];
    }
}

// Batched weights twin: grid.x=K threadgroups, k_used threads each. probs [K,n_experts]
// offset by row*n_experts; selected/weights [K,k_used] offset by row*k_used.
kernel void ds4_dsv4_router_weights_one_any_k(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        constant uint  &k_used,
        constant float &scale,
        constant float &min_sum,
        constant uint  &n_experts,
        uint tid [[thread_position_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
    if (tid >= k_used) return;
    device const float *p = (device const float *)probs + (uint64_t)row * (uint64_t)n_experts;
    device const int   *s = (device const int *)selected + (uint64_t)row * (uint64_t)k_used;

    float sum = 0.0f;
    for (uint i = 0; i < k_used; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, min_sum);

    device float *w = (device float *)weights + (uint64_t)row * (uint64_t)k_used;
    w[tid] = p[s[tid]] / sum * scale;
}
