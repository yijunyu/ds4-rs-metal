# Metal Attention Composer Blueprint (#216)

Source-of-truth references extracted from
`benchmarks/ds4_msl/upstream/ds4/ds4_metal.m` (pinned commit `d615ab0`)
for the 5 heavyweight composers in `ds4_metal/src/macos.rs`.

## 1. `flash_attn_decode_impl`

**Antirez reference:** `ds4_metal.m:8550-8748` (the `ds4_metal_encode_flash_attn` body).

**Struct (ds4_metal.m:2265-2298):** `ds4_metal_flash_attn_vec_args` — 30 fields:

```c
struct ds4_metal_flash_attn_vec_args {
    int32_t  ne01, ne02, ne03;                  // q dims:  1, n_head, 1
    uint64_t nb01, nb02, nb03;                  // q strides (bytes)
    int32_t  ne11, ne_12_2, ne_12_3;            // k dims:  n_raw, 1, 1
    int32_t  ns10;                              // k row dim (head_dim)
    uint64_t nb11, nb12, nb13;                  // k strides
    int32_t  ns20;                              // v row dim
    uint64_t nb21, nb22, nb23;                  // v strides
    int32_t  ne31, ne32, ne33;                  // mask dims (1,1,1)
    uint64_t nb31, nb32, nb33;                  // mask strides
    int32_t  ne1, ne2, ne3;                     // output dims (n_head,1,1)
    float    scale, max_bias, m0, m1;           // 1.0/sqrt(dk), 0,0,0 for DS4
    int32_t  n_head_log2;                       // 0 for DS4 (no ALiBi)
    float    logit_softcap;                     // 0 for DS4
};
```

**Pipeline:** `ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512`
(pipeline lookup already wired in current stub).

**Required scratch buffers** (each persistent across calls — cache on `MetalState`):
- `kv_buffer`:  `n_raw * head_dim * sizeof(uint16_t)` (f16 k/v after cpy_f32_f16)
- `pad_buffer`: `2 * ncpsg * head_dim * sizeof(uint16_t) + ncpsg * sizeof(uint16_t)` (only if `n_raw % ncpsg != 0`)
- `tmp_buffer`: `n_head * head_dim * nwg * sizeof(float) + n_head * 2 * nwg * sizeof(float)`
- `ring_buffer`: `n_raw * head_dim * sizeof(float)` (only if `raw_start != 0`)
- `mask_buffer`: transient, zero-init, `n_raw * sizeof(uint16_t)`

**Constants:** `ncpsg=32`, `nwg=32`, `nsg = ds4_metal_flash_attn_vec_nsg(n_raw, 32, 32)`.

**Sub-pipelines required:**
1. `cpy_f32_f32_1d`  (only when raw_start != 0; ring-buffer rotation)
2. `cpy_f32_f16_1d`  (always; quantize kv to f16)
3. `flash_attn_pad`  (only when n_raw % 32 != 0)
4. `flash_attn_vec_f16_dk512_dv512` (the main pipeline)
5. `flash_attn_reduce` (final 32-way wg reduce → heads)

**Encode shape (main pass):** `dispatchThreadgroups(MTLSize(1, n_head, nwg), MTLSize(32, nsg, 1))`.
Threadgroup memory: `align_up(head_dim, 128) + 4*ncpsg + 2*align_up(head_dim, 128)) * nsg * sizeof(half)`, aligned 16.

**Encode shape (reduce pass):** `dispatchThreadgroups(MTLSize(n_head, 1, 1), MTLSize(32*nwg, 1, 1))`.

**Risk:** scratch-buffer caching policy. Antirez has `ds4_metal_ensure_scratch_buffer` (grows
or recreates if undersized). Mirror this with a `BufferPool` on `MetalState`.

## 2. `hc_collapse_norm_impl`

**Antirez reference:** chains 3 kernels in `ds4_metal.m` decode_layer body. Pipelines:
- `dsv4_sinkhorn` (or registry equivalent)
- HC collapse: 1 fused kernel (registry name `ds4_kernel_hc_collapse_f32`?)
- `rms_norm_mul_f32_4`

**Input signature:**
```rust
fn hc_collapse_norm(
    &self,
    params: &LayerParams,
    hc_mix: &[f32],              // [n_hc * n_kv_heads]
    after_attn_hc: &[f32],       // [n_hc * d_model]
    hc_split_in: &[f32],         // [n_hc * d_model]
    use_gamma: Option<&[f32]>,   // [d_model] rms gamma
) -> (Vec<f32>, Vec<f32>, Vec<f32>)
//     ^new_hc_mix, ^after_attn_hc_normed, ^hc_split_out
```

**Pending:** confirm exact registry names + buffer signatures by grepping
the kernel registry for `hc_collapse` / `sinkhorn`.

## 3. `attn_output_proj_impl`

**REVISED 2026-05-13** after reading `kernel_registry.rs:260-273`:
The registry has FUSED kernels — `dsv4_q8_hc_expand4_q8_0` (M126) is a single
"fused q8_0 matvec + 4-channel HC expand" kernel. So `attn_output_proj` is
likely ONE registry call after a single low-rank matvec, not two matvecs +
separate expand. Confirm against `ds4.c:9417` decode-layer body once on Mac.

**Pipeline candidates:**
- `ds4_kernel_dsv4_q8_hc_expand4_q8_0` (fused matvec + hc_expand, 7 bufs, 2 uniform structs)
- Possibly `ds4_kernel_mul_mv_f16_f32_4` for the low-rank a-projection if not fused.

**Buffer signature (M126):** `[ConstIn, ConstIn, Writable, ConstIn, ConstIn, ConstIn, Writable]`.
**Uniforms:** `Struct(2)` — HcMvUniforms + HcExpandUniforms (need source check).

**Encoding strategy (provisional):**
1. matvec_f16_f32_4(W_o_a, heads) → tmp_a  [low-rank intermediate]
2. dsv4_q8_hc_expand4_q8_0(W_o_b, tmp_a, cur_hc, hc_split, hc_post, hc_comb) → out

## 4. `shared_expert_impl`

**Antirez reference:** `mul_mv_id_q4_K_pair_swiglu_f32` with single ID = shared-expert index.

**Pipeline:** `ds4_kernel_mul_mv_id_q4_K_pair_swiglu_f32`.

**Encoding:** standard paired matvec with shared gate/up weights, SwiGLU activation
fused in the kernel. Single ID input = `&[shared_expert_id]`.

## 5. `shared_down_hc_expand_add_impl`

**REVISED 2026-05-13:** the registry has the fused kernel:
- `dsv4_shared_down_hc_expand4_q8_0` (M127): "add-sibling of M126 with
  extra residual-add input". 8 buffers, 2 uniform structs.

**Pipeline:** `ds4_kernel_dsv4_shared_down_hc_expand4_q8_0`.

**Buffer signature:** `[ConstIn, ConstIn, Writable, ConstIn, ConstIn, ConstIn, ConstIn, Writable]`.

**Encoding:** single dispatch — kernel folds the W_down matvec + HC expand
+ residual-add into one pass. No multi-stage encoding needed.

## Shared infrastructure to add to MetalState

```rust
struct ScratchBuffer {
    buffer: metal::Buffer,
    bytes: usize,
}

impl MetalState {
    fn ensure_scratch(&mut self, key: &str, min_bytes: usize) -> &metal::Buffer { ... }
    fn transient_buffer(&self, bytes: usize, label: &str) -> metal::Buffer { ... }
}
```

`ensure_scratch` grows the buffer in-place when an inner attention call needs
more space than the last; `transient_buffer` allocates per-call (mask, etc.)
and is freed when the command buffer completes.

## Bring-up order on Mac (recommended)

1. `qkv_rms_norm_rows` / `rope_tail` / `kv_fp8_store` smoke tests (already encoded).
2. `attn_output_proj_impl` — 2 matvecs + 1 hc_expand, no scratch caching needed.
3. `shared_expert_impl` — single paired matvec.
4. `shared_down_hc_expand_add_impl` — matvec + expand_add.
5. `hc_collapse_norm_impl` — 3-kernel chain.
6. `flash_attn_decode_impl` — last, hardest, needs ScratchBuffer infrastructure.
