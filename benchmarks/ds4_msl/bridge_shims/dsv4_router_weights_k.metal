// K-batched hash-router weights (chunk-graph stage 3): one dispatch computes
// weights for K positions — replaces the per-chunk drain of probs_k + CPU
// hash_router_weights_from_probs in the chunk hash branch. Row r computes
// w[r,t] = p[r, sel[r,t]] / max(Σ_t p[r, sel[r,t]], min_sum) * scale.
// Dispatch: grid (k_used threads, K rows).
kernel void ds4_dsv4_router_weights_k(
        device const float *probs,
        device const int   *selected,
        device       float *weights,
        constant uint  &k_used,
        constant uint  &n_experts,
        constant float &scale,
        constant float &min_sum,
        uint tid [[thread_position_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
    if (tid >= k_used) return;
    device const float *p = probs + row * n_experts;
    device const int   *s = selected + row * k_used;
    float sum = 0.0f;
    for (uint i = 0; i < k_used; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, min_sum);
    weights[row * k_used + tid] = p[s[tid]] / sum * scale;
}
