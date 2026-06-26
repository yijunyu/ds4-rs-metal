# ds4_metal Metal kernel patches

Kernels we add to the **synced antirez upstream** Metal sources. The upstream dir
`benchmarks/ds4_msl/upstream/ds4/metal/` is gitignored and re-synced from antirez, so
edits there are lost on a re-sync — these tracked copies are the source of truth.

These are **not** auto-applied by `build.rs` (that source-patch mechanism was reverted
for resync-PANIC fragility, commit `5fca874d`). Re-apply manually after an upstream sync.

## `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f32.metal`

Fused gate+up+SwiGLU `mm_id` for iq2_xxs MoE experts (prefill). Ported from antirez's
`kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16`; the one change is f32 `dst_mid` output so
our existing f32 down-GEMM consumes it unchanged.

**Apply:** insert the kernel into `upstream/ds4/metal/moe.metal` immediately **before**
the line `typedef decltype(kernel_mul_mm_id<half, half4x4, ...>) mul_mm_id;` — i.e. after
`kernel_mul_mm_id`'s closing brace, while `QK_NL` / `dequantize_iq2_xxs` / `block_iq2_xxs`
/ `ds4_metal_args_mul_mm_id` / `ds4_metal_dsv4_moe_swiglu_weight_args` /
`make_filled_simdgroup_matrix` / `FOR_UNROLL` are still in scope (they are `#undef`'d
further down). `build.rs` renames the bare `kernel void kernel_*` to the `ds4_kernel_*`
host name the dispatcher (`encode_mul_mm_id_iq2_pair_swiglu_k`) looks up.

**Gate:** `DS4_MOE_FUSED_MMID=1`. Result: +0.8% prefill @3000, byte-identical decode
(`chunk_moe_fused_identity`). Marginal — the prefill MoE is weight-bandwidth-bound, so
fusing the non-weight overhead recovers little. Kept for the record + the indexer-f16
sibling lever. See memory `ds4-moe-compute-frontier`.
