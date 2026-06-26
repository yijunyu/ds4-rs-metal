//! Kernel registry — semantic name → Metal function name + buffer-layout metadata.
//!
//! Each `KernelSpec` describes one entry in a Metal `library` that was generated
//! by `rustc_codegen_tile`'s MSL backend (M104b-M150). The DS4 decode forward
//! graph (E4-E6) dispatches kernels by looking up specs here, allocating
//! buffers according to `BufRole`, and binding uniforms in the documented slot
//! order.
//!
//! macOS-only at runtime — the actual `MTLComputePipelineState` lives in a
//! sibling module behind `#[cfg(target_os = "macos")]`. The registry data
//! itself compiles on any host so we can author and unit-test it on Linux.
//!
//! # Naming convention
//!
//! `metal_fn` matches the MLIR `func.name` from `crates/codegen_tests/src/bin/print_msl.rs`
//! (the test fixtures that drive `mlir_to_msl.rs`). The emitter writes this
//! verbatim into `kernel void {func.name}(` at `mlir_to_msl.rs:691`, so this
//! is the exact symbol Metal's `make_compute_pipeline_state(name=…)` resolves.
//! Antirez upstream uses `kernel_*_f32`-style names; tile-rs prefixes with
//! `ds4_` (sometimes `ds4_kernel_` where the fixture preserves the upstream
//! prefix). The test `every_metal_fn_matches_an_emitted_fixture` enforces this.
//!
//! # Buffer-layout conventions (mirrors `mlir_to_msl.rs`)
//!
//! - `device char*` byte-stride buffers follow the antirez convention; the
//!   emitter binds them at buffer slots `0..n_bufs`.
//! - Uniforms follow at slot `n_bufs`. For kernels with ≤22 individual
//!   `constant uint& xxx [[buffer(N)]]`, each scalar takes one slot. For
//!   kernels exceeding Metal's 31-slot cap, uniforms are packed into a
//!   `constant Struct&` at slot `n_bufs` (see FaUniforms / HcMvUniforms /
//!   HcExpandUniforms rule from M123 / M126 / M127).
//! - `BufRole::ConstIn` = read-only input; `BufRole::Writable` = output (or
//!   read-modify-write). The split mirrors the writability predicates that
//!   `mlir_to_msl.rs` uses to emit `device` vs `device const`.

#![allow(dead_code)]

use std::collections::HashMap;

/// Role of a buffer at a Metal binding slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufRole {
    /// Read-only input. Emitter writes `device const char*`.
    ConstIn,
    /// Writable output (or in-place). Emitter writes `device char*`.
    Writable,
}

/// Shape of the uniform block following the buffer slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformLayout {
    /// `n` scalar `constant uint&` slots (≤22 by Metal practice).
    Scalars(u8),
    /// One packed `constant Struct&` slot. Caller is responsible for the
    /// struct layout — see `FaUniforms` / `HcExpandUniforms`. The `u8` is the
    /// total uniform-slot count (always 1 with this variant, but kept for
    /// symmetry with future multi-struct kernels like M126 with 2 structs).
    Struct(u8),
}

/// Static description of one kernel.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// Stable semantic name used by the engine forward graph.
    pub name: &'static str,
    /// Metal function name (the `[[kernel]]` symbol in the generated `.metal`).
    pub metal_fn: &'static str,
    /// One entry per buffer slot, in binding order.
    pub bufs: &'static [BufRole],
    /// Uniform-slot layout immediately following the buffers.
    pub uniforms: UniformLayout,
    /// Optional human-readable note (which milestone landed it, etc.).
    pub note: &'static str,
}

impl KernelSpec {
    pub const fn n_bufs(&self) -> usize {
        self.bufs.len()
    }
}

// ---------------------------------------------------------------------------
// DS4 decode forward-graph registry.
// ---------------------------------------------------------------------------
//
// Curated subset of the M104b-M150 surface. Order = rough dispatch order in
// one decode step: norm → attention → MoE FFN → final norm + lm_head.

use BufRole::*;

pub const KERNELS: &[KernelSpec] = &[
    // ── RMSNorm + element-wise utilities ────────────────────────────────
    KernelSpec {
        name: "rms_norm_f32_4",
        metal_fn: "ds4_kernel_rms_norm_f32_4",
        bufs: &[ConstIn, Writable],
        uniforms: UniformLayout::Scalars(4),
        note: "M-series RmsNormF32_4: per-row float4 plain RMSNorm",
    },
    KernelSpec {
        name: "rms_norm_mul_f32_4",
        metal_fn: "ds4_kernel_rms_norm_mul_f32_4",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(4),
        note: "RmsNormMulF32_4: per-row RMSNorm × learned weight",
    },
    KernelSpec {
        name: "dsv4_qkv_rms_norm_f32_4",
        metal_fn: "ds4_dsv4_qkv_rms_norm_f32_4",
        bufs: &[ConstIn, ConstIn, Writable, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(7),
        note: "Q + KV RMSNorm fused into one dispatch (MLA q-lora + kv-lora rows)",
    },

    // ── Embedding lookup (get_rows) ──────────────────────────────────────
    KernelSpec {
        name: "get_rows_f16",
        metal_fn: "ds4_kernel_get_rows_f16",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(12),
        note: "Vocabulary embedding: f16 table → f32 output, int32 ids",
    },
    KernelSpec {
        name: "get_rows_f32",
        metal_fn: "ds4_kernel_get_rows_f32",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(12),
        note: "f32 sibling; rarely used in DS4 (most rows are quantized)",
    },

    // ── DS4 RoPE (partial + YaRN) ────────────────────────────────────────
    KernelSpec {
        name: "dsv4_rope_tail_f32",
        metal_fn: "ds4_kernel_dsv4_rope_tail_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(23),
        note: "M122: byte-stride partial RoPE; copies n_nope prefix, rotates n_dims tail with YaRN",
    },

    // ── FlashAttention (decode-shape) ────────────────────────────────────
    KernelSpec {
        name: "flash_attn_ext_vec_f16_dk512_dv512",
        metal_fn: "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512",
        bufs: &[ConstIn, ConstIn, ConstIn, ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Struct(1),
        note: "M124: decode shape (Q=1), 32-field FaVecUniforms packed struct",
    },

    // ── MoE matvec — single-expert decode (M111/M112/M113) ───────────────
    KernelSpec {
        name: "mul_mv_id_q2_K_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q2_K_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(11),
        note: "M111: q2_K routed matvec, NSG=2 NR0=4",
    },
    KernelSpec {
        name: "mul_mv_id_q4_K_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q4_K_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(11),
        note: "M112: q4_K routed matvec, NSG=2 NR0=2, 6-bit scale unpack",
    },
    KernelSpec {
        name: "mul_mv_id_iq2_xxs_f32",
        metal_fn: "ds4_kernel_mul_mv_id_iq2_xxs_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(11),
        note: "M113: iq2_xxs routed matvec, NSG=2 NR0=4, grid+ksigns shmem",
    },

    // ── MoE paired matvec (M128/M130) ────────────────────────────────────
    KernelSpec {
        name: "mul_mv_id_iq2_xxs_pair_f32",
        metal_fn: "ds4_kernel_mul_mv_id_iq2_xxs_pair_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable, ConstIn],
        uniforms: UniformLayout::Scalars(11),
        note: "M128: paired gate+up iq2_xxs, shared y + table",
    },
    KernelSpec {
        name: "mul_mv_id_q4_K_pair_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q4_K_pair_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable, ConstIn],
        uniforms: UniformLayout::Scalars(11),
        note: "M130: paired gate+up q4_K, NR0=2 shared y",
    },

    // ── MoE SwiGLU-fused tri-output (M129/M131) ──────────────────────────
    KernelSpec {
        name: "mul_mv_id_iq2_xxs_pair_swiglu_f32",
        metal_fn: "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable, Writable, ConstIn, ConstIn],
        uniforms: UniformLayout::Scalars(14),
        note: "M129: paired iq2_xxs + SwiGLU writeback to dst_mid",
    },
    KernelSpec {
        name: "mul_mv_id_q4_K_pair_swiglu_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q4_K_pair_swiglu_f32",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable, Writable, ConstIn, ConstIn],
        uniforms: UniformLayout::Scalars(14),
        note: "M131: paired q4_K + SwiGLU writeback",
    },

    // ── MoE sum-of-6-experts (M132/M133) ─────────────────────────────────
    KernelSpec {
        name: "mul_mv_id_q2_K_sum6_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q2_K_sum6_f32",
        bufs: &[ConstIn, ConstIn, Writable, ConstIn],
        uniforms: UniformLayout::Scalars(8),
        note: "M132: q2_K sum-over-6-fixed-experts per token",
    },
    KernelSpec {
        name: "mul_mv_id_q4_K_sum6_f32",
        metal_fn: "ds4_kernel_mul_mv_id_q4_K_sum6_f32",
        bufs: &[ConstIn, ConstIn, Writable, ConstIn],
        uniforms: UniformLayout::Scalars(8),
        note: "M133: q4_K sum-over-6-fixed-experts per token",
    },

    // ── MoE routing helpers (argsort, topk_mask, router_finalize) ────────
    KernelSpec {
        name: "argsort_f32_i32_desc_full",
        metal_fn: "ds4_kernel_argsort_f32_i32_desc_full",
        bufs: &[ConstIn, Writable],
        uniforms: UniformLayout::Scalars(13),
        note: "M134: full 4-D bitonic sort, top-k indices descending",
    },
    KernelSpec {
        name: "argsort_merge_f32_i32_desc_full",
        metal_fn: "ds4_kernel_argsort_merge_f32_i32_desc_full",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(14),
        note: "M135: merge two pre-sorted runs into top-k",
    },
    KernelSpec {
        name: "dsv4_topk_mask",
        metal_fn: "ds4_dsv4_topk_mask",
        bufs: &[ConstIn, Writable],
        uniforms: UniformLayout::Scalars(8),
        note: "M125: -INFINITY mask fill for indexer attention (served by antirez dsv4_misc.metal raw kernel post bridge rewrite)",
    },
    KernelSpec {
        name: "dsv4_router_finalize_one",
        metal_fn: "ds4_dsv4_router_finalize_one",
        bufs: &[ConstIn, ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(5),
        note: "256-thread bitonic top-6 over (probs + bias) with hash_mode short-circuit",
    },
    KernelSpec {
        name: "dsv4_router_weights_one",
        metal_fn: "ds4_dsv4_router_weights_one",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "k-thread normalize step: weights[i] = probs[selected[i]] / max(Σ_selected, min_sum) * scale (uniforms: num_experts, scale, min_sum)",
    },

    // ── MoE map0 (expert dispatch table builder) ─────────────────────────
    KernelSpec {
        name: "dsv4_mul_mm_id_map0_ne20_8_full",
        metal_fn: "ds4_kernel_mul_mm_id_map0_ne20_8",
        bufs: &[ConstIn, Writable, Writable],
        uniforms: UniformLayout::Scalars(8),
        note: "M136: full host_name surface, ne20=8 fanout",
    },

    // ── HC expand (decode KV recurrent) ──────────────────────────────────
    KernelSpec {
        name: "dsv4_q8_hc_expand4_q8_0",
        metal_fn: "ds4_dsv4_q8_hc_expand4_q8_0",
        bufs: &[ConstIn, ConstIn, Writable, ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Struct(2),
        note: "M126: fused q8_0 matvec + 4-channel HC expand; served by antirez dsv4_hc.metal raw kernel post bridge rewrite",
    },
    KernelSpec {
        name: "dsv4_shared_down_hc_expand4_q8_0",
        metal_fn: "ds4_dsv4_shared_down_hc_expand4_q8_0",
        bufs: &[ConstIn, ConstIn, Writable, ConstIn, ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Struct(2),
        note: "M127: add-sibling of M126 with extra residual-add input; served by antirez dsv4_hc.metal raw kernel post bridge rewrite",
    },
    KernelSpec {
        name: "dsv4_shared_gate_up_swiglu_q8_0",
        metal_fn: "ds4_dsv4_shared_gate_up_swiglu_q8_0",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable, Writable],
        uniforms: UniformLayout::Struct(1),
        note: "Fused shared-expert q8_0 gate/up matvec plus SwiGLU mid; served by antirez dense.metal raw kernel post bridge rewrite",
    },
    KernelSpec {
        name: "dsv4_attn_out_low_q8_0_f32",
        metal_fn: "ds4_dsv4_attn_out_low_q8_0_f32",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Struct(1),
        note: "Grouped Q8_0 attention output-A projection; antirez direct path uses group id from the z-grid",
    },

    // ── Softmax (DS4 full host_name) ─────────────────────────────────────
    KernelSpec {
        name: "soft_max_f32_4",
        metal_fn: "ds4_kernel_soft_max_f32_4",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(22),
        note: "M117: unified softmax with mask/sink/ALiBi runtime branches",
    },

    // ── Dense matvec (lm_head + projection layers) ───────────────────────
    KernelSpec {
        name: "mul_mv_f16_f32_4",
        metal_fn: "ds4_kernel_mul_mv_f16_f32_4",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(14),
        note: "M116: float4-vectorized dense matvec, half src0",
    },
    KernelSpec {
        name: "mul_mv_q8_0_f32",
        metal_fn: "ds4_kernel_mul_mv_q8_0_f32",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(13),
        note: "M91: q8_0 dense matvec",
    },

    // ── Activations (SiLU/GELU/Sigmoid) ──────────────────────────────────
    KernelSpec {
        name: "silu_f32_4",
        metal_fn: "ds4_kernel_silu_f32_4",
        bufs: &[ConstIn, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "M120: float4 SiLU (sub-op of UnaryF32F32_4)",
    },

    // ── DS4 misc: softplus+sqrt, softmax pool, hc weighted sum ───────────
    KernelSpec {
        name: "dsv4_softplus_sqrt_f32_4",
        metal_fn: "ds4_kernel_dsv4_softplus_sqrt_f32_4",
        bufs: &[ConstIn, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "Decode-time router-logit transform: softplus then sqrt",
    },
    KernelSpec {
        name: "dsv4_softmax_pool",
        metal_fn: "ds4_dsv4_softmax_pool",
        bufs: &[ConstIn, ConstIn, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "Indexer attention pool: weighted softmax over scores → kv mix",
    },

    // ── KV cache store helpers ───────────────────────────────────────────
    KernelSpec {
        name: "dsv4_compressor_store_one",
        metal_fn: "ds4_dsv4_compressor_store_one",
        bufs: &[ConstIn, ConstIn, ConstIn, Writable, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "5-buffer KV+score store with positional encoding add",
    },
    KernelSpec {
        name: "dsv4_kv_fp8_store",
        metal_fn: "ds4_dsv4_kv_fp8_store",
        bufs: &[Writable, Writable],
        uniforms: UniformLayout::Scalars(3),
        note: "Per-row n_nope chunked-64 fp8 round-trip + n_rot tail half-cast",
    },
    KernelSpec {
        name: "dsv4_ratio4_shift",
        metal_fn: "ds4_dsv4_ratio4_shift",
        bufs: &[Writable, Writable],
        uniforms: UniformLayout::Scalars(1),
        note: "KV ratio-4 recurrent-state shift over two buffers",
    },
];

/// Build a lookup index by semantic name.
///
/// Returns a `HashMap<&'static str, &'static KernelSpec>`. The map is small
/// (≤40 entries) so a sequential scan would also work; `HashMap` is used so
/// `lookup(name)` reads idiomatically in the engine.
pub fn index() -> HashMap<&'static str, &'static KernelSpec> {
    KERNELS.iter().map(|k| (k.name, k)).collect()
}

/// Direct lookup by semantic name.
pub fn lookup(name: &str) -> Option<&'static KernelSpec> {
    KERNELS.iter().find(|k| k.name == name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_non_empty() {
        assert!(!KERNELS.is_empty());
    }

    #[test]
    fn semantic_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for k in KERNELS {
            assert!(
                seen.insert(k.name),
                "duplicate semantic name in registry: {}",
                k.name
            );
        }
    }

    #[test]
    fn metal_fn_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for k in KERNELS {
            assert!(
                seen.insert(k.metal_fn),
                "duplicate metal_fn in registry: {}",
                k.metal_fn
            );
        }
    }

    #[test]
    fn lookup_finds_known_kernels() {
        assert!(lookup("rms_norm_f32_4").is_some());
        assert!(lookup("flash_attn_ext_vec_f16_dk512_dv512").is_some());
        assert!(lookup("mul_mv_id_q2_K_sum6_f32").is_some());
        assert!(lookup("nonexistent_kernel").is_none());
    }

    #[test]
    fn index_size_matches_kernels() {
        assert_eq!(index().len(), KERNELS.len());
    }

    #[test]
    fn decode_critical_path_kernels_present() {
        // The minimum set required to dispatch one DS4 decode step:
        //   embedding → RMSNorm → QKV RMSNorm → RoPE → FlashAttn →
        //   MoE router (argsort+softmax_pool) → 6 experts × matvec_pair_swiglu →
        //   HC expand → final RMSNorm → lm_head matvec
        let must_have = [
            "get_rows_f16",
            "rms_norm_f32_4",
            "rms_norm_mul_f32_4",
            "dsv4_qkv_rms_norm_f32_4",
            "dsv4_rope_tail_f32",
            "flash_attn_ext_vec_f16_dk512_dv512",
            "dsv4_router_finalize_one",
            "mul_mv_id_q2_K_sum6_f32",
            "mul_mv_id_q4_K_sum6_f32",
            "mul_mv_id_q4_K_pair_swiglu_f32",
            "dsv4_q8_hc_expand4_q8_0",
            "soft_max_f32_4",
            "mul_mv_f16_f32_4",
        ];
        for name in must_have {
            assert!(
                lookup(name).is_some(),
                "decode-critical kernel missing: {name}"
            );
        }
    }

    struct KernelCoverage {
        name: &'static str,
        evidence: &'static [&'static str],
    }

    const COVERAGE: &[KernelCoverage] = &[
        KernelCoverage {
            name: "rms_norm_f32_4",
            evidence: &["engine synthetic CPU reference", "macOS attention smoke"],
        },
        KernelCoverage {
            name: "rms_norm_mul_f32_4",
            evidence: &["engine synthetic CPU reference", "registry decode-critical"],
        },
        KernelCoverage {
            name: "dsv4_qkv_rms_norm_f32_4",
            evidence: &[
                "macOS attention smoke",
                "engine synthetic attention reference",
            ],
        },
        KernelCoverage {
            name: "get_rows_f16",
            evidence: &[
                "engine decode synthetic",
                "real GGUF layout: token_embd F16",
            ],
        },
        KernelCoverage {
            name: "get_rows_f32",
            evidence: &["registry/fixture only: uncommon DS4 path"],
        },
        KernelCoverage {
            name: "dsv4_rope_tail_f32",
            evidence: &[
                "macOS attention smoke",
                "engine synthetic attention reference",
            ],
        },
        KernelCoverage {
            name: "flash_attn_ext_vec_f16_dk512_dv512",
            evidence: &[
                "engine synthetic attention reference",
                "registry decode-critical",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_q2_K_f32",
            evidence: &[
                "synthetic quantized expert oracle",
                "real GGUF layout: ffn_down_exps Q2_K",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_q4_K_f32",
            evidence: &["synthetic quantized expert oracle"],
        },
        KernelCoverage {
            name: "mul_mv_id_iq2_xxs_f32",
            evidence: &[
                "synthetic quantized expert oracle",
                "real GGUF layout: ffn_gate/up_exps IQ2_XXS",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_iq2_xxs_pair_f32",
            evidence: &[
                "synthetic quantized expert oracle",
                "real GGUF layout: ffn_gate/up_exps IQ2_XXS",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_q4_K_pair_f32",
            evidence: &["synthetic quantized expert oracle"],
        },
        KernelCoverage {
            name: "mul_mv_id_iq2_xxs_pair_swiglu_f32",
            evidence: &[
                "synthetic quantized expert oracle",
                "real GGUF sparse MoE layer-0",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_q4_K_pair_swiglu_f32",
            evidence: &["synthetic quantized expert oracle"],
        },
        KernelCoverage {
            name: "mul_mv_id_q2_K_sum6_f32",
            evidence: &[
                "synthetic quantized expert oracle",
                "real GGUF sparse MoE layer-0",
            ],
        },
        KernelCoverage {
            name: "mul_mv_id_q4_K_sum6_f32",
            evidence: &["synthetic quantized expert oracle"],
        },
        KernelCoverage {
            name: "argsort_f32_i32_desc_full",
            evidence: &["engine router synthetic reference"],
        },
        KernelCoverage {
            name: "argsort_merge_f32_i32_desc_full",
            evidence: &["registry/fixture only: not yet runtime-smoked"],
        },
        KernelCoverage {
            name: "dsv4_topk_mask",
            evidence: &["engine attention synthetic reference"],
        },
        KernelCoverage {
            name: "dsv4_router_finalize_one",
            evidence: &[
                "engine router synthetic reference",
                "math-vs-antirez contract tests",
            ],
        },
        KernelCoverage {
            name: "dsv4_router_weights_one",
            evidence: &[
                "engine hash-router synthetic reference",
                "math-vs-antirez contract tests",
            ],
        },
        KernelCoverage {
            name: "dsv4_mul_mm_id_map0_ne20_8_full",
            evidence: &["registry/fixture only: dispatch-map builder needs runtime smoke"],
        },
        KernelCoverage {
            name: "dsv4_q8_hc_expand4_q8_0",
            evidence: &[
                "engine synthetic HC reference",
                "real GGUF layout: Q8_0 projections",
            ],
        },
        KernelCoverage {
            name: "dsv4_shared_down_hc_expand4_q8_0",
            evidence: &[
                "engine synthetic shared-expert/HC reference",
                "real GGUF layout: Q8_0 shared expert",
            ],
        },
        KernelCoverage {
            name: "dsv4_shared_gate_up_swiglu_q8_0",
            evidence: &[
                "real GGUF shared-expert gate/up Q8_0 oracle",
                "macOS function-constant specialization sentinel",
            ],
        },
        KernelCoverage {
            name: "dsv4_attn_out_low_q8_0_f32",
            evidence: &[
                "real GGUF attention output-A Q8_0 grouped oracle",
                "macOS function-constant specialization sentinel",
            ],
        },
        KernelCoverage {
            name: "soft_max_f32_4",
            evidence: &["engine softmax synthetic reference"],
        },
        KernelCoverage {
            name: "mul_mv_f16_f32_4",
            evidence: &[
                "engine synthetic matvec reference",
                "real GGUF layout: F16 HC tensors",
            ],
        },
        KernelCoverage {
            name: "mul_mv_q8_0_f32",
            evidence: &[
                "engine/layer-view q8_0 reference",
                "real GGUF layout: Q8_0 projections",
            ],
        },
        KernelCoverage {
            name: "silu_f32_4",
            evidence: &["engine synthetic SiLU/SwiGLU reference"],
        },
        KernelCoverage {
            name: "dsv4_softplus_sqrt_f32_4",
            evidence: &["engine router synthetic reference"],
        },
        KernelCoverage {
            name: "dsv4_softmax_pool",
            evidence: &["engine attention compressor synthetic reference"],
        },
        KernelCoverage {
            name: "dsv4_compressor_store_one",
            evidence: &["engine attention compressor synthetic reference"],
        },
        KernelCoverage {
            name: "dsv4_kv_fp8_store",
            evidence: &["macOS attention smoke", "engine synthetic FP8 reference"],
        },
        KernelCoverage {
            name: "dsv4_ratio4_shift",
            evidence: &["engine attention compressor synthetic reference"],
        },
    ];

    #[test]
    fn every_registered_kernel_has_declared_test_coverage() {
        let covered: std::collections::HashMap<&str, &[&str]> =
            COVERAGE.iter().map(|c| (c.name, c.evidence)).collect();
        for k in KERNELS {
            let evidence = covered
                .get(k.name)
                .unwrap_or_else(|| panic!("kernel {} has no coverage declaration", k.name));
            assert!(
                !evidence.is_empty(),
                "kernel {} has an empty coverage declaration",
                k.name
            );
        }
    }

    #[test]
    fn coverage_declarations_refer_to_registered_kernels() {
        let registered: std::collections::HashSet<&str> = KERNELS.iter().map(|k| k.name).collect();
        for c in COVERAGE {
            assert!(
                registered.contains(c.name),
                "coverage declaration references non-registered kernel {}",
                c.name
            );
        }
    }

    #[test]
    fn real_gguf_coverage_is_declared_for_released_quantized_paths() {
        for name in [
            "mul_mv_id_iq2_xxs_f32",
            "mul_mv_id_iq2_xxs_pair_f32",
            "mul_mv_id_iq2_xxs_pair_swiglu_f32",
            "mul_mv_id_q2_K_f32",
            "mul_mv_id_q2_K_sum6_f32",
            "mul_mv_q8_0_f32",
            "dsv4_q8_hc_expand4_q8_0",
            "dsv4_shared_down_hc_expand4_q8_0",
        ] {
            let coverage = COVERAGE
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("missing coverage declaration for {name}"));
            assert!(
                coverage.evidence.iter().any(|e| e.contains("real GGUF")),
                "{name} should have real-GGUF coverage evidence, got {:?}",
                coverage.evidence
            );
        }
    }

    /// Snapshot of every `ds4_*` MLIR `func.name` declared in
    /// `crates/codegen_tests/src/bin/print_msl.rs` at HEAD on 2026-05-12.
    /// These are the names the MSL emitter (`mlir_to_msl.rs:691`) writes into
    /// `kernel void {func.name}(` — i.e. the symbols Metal's
    /// `make_compute_pipeline_state(name=…)` actually looks up.
    /// Refresh by grepping `llvm.func @ds4_*` in `print_msl.rs`.
    /// Antirez upstream uses different names (`kernel_*_f32`); this registry
    /// no longer claims antirez parity — the dispatcher consumes tile-rs's
    /// emitted names directly.
    const EMITTED_KERNEL_SYMBOLS: &[&str] = &[
        "ds4_argsort_f32_i32_desc",
        "ds4_argsort_merge_f32_i32_desc",
        "ds4_attention",
        "ds4_bin_add",
        "ds4_bin_div",
        "ds4_bin_mul",
        "ds4_bin_sub",
        "ds4_concat",
        "ds4_cpy",
        "ds4_dsv4_compressor_store_one",
        "ds4_dsv4_fp8_kv_quantize",
        "ds4_dsv4_hc_expand",
        "ds4_dsv4_hc_expand4",
        "ds4_dsv4_hc_split_sinkhorn_hc4",
        "ds4_dsv4_hc_split_weighted_sum_hc4",
        "ds4_dsv4_hc_split_weighted_sum_norm4",
        "ds4_dsv4_hc_weighted_sum",
        "ds4_dsv4_indexed_mixed_attention_h8",
        "ds4_dsv4_indexed_mixed_attention_h8_rb4",
        "ds4_dsv4_indexer_score_one_direct",
        "ds4_dsv4_indexer_scores_tiled",
        "ds4_dsv4_indexer_scores_tiled_f32",
        "ds4_dsv4_indexer_weighted_sum",
        "ds4_dsv4_kv_fp8_store",
        "ds4_dsv4_moe_swiglu_weight",
        "ds4_dsv4_moe_swiglu_weight_f16",
        "ds4_dsv4_mul_mm_id_map0",
        "ds4_dsv4_qkv_rms_norm_f32_4",
        "ds4_dsv4_ratio4_shift",
        "ds4_dsv4_router_finalize_one",
        "ds4_dsv4_router_weights_one",
        "ds4_dsv4_softmax_pool",
        "ds4_dsv4_sort_i32_rows_asc",
        "ds4_dsv4_topk_mask_scatter",
        "ds4_flash_attn_ext_blk",
        "ds4_flash_attn_ext_out",
        "ds4_flash_attn_ext_out_ms",
        "ds4_flash_attn_ext_pad",
        "ds4_flash_attn_ext_score",
        "ds4_flash_attn_ext_setup",
        "ds4_flash_attn_ext_vec_out",
        "ds4_flash_attn_ext_vec_out_ms",
        "ds4_flash_attn_ext_vec_reduce",
        "ds4_flash_attn_ext_vec_score",
        "ds4_flash_attn_ext_vec_setup",
        "ds4_get_rows",
        "ds4_glu",
        "ds4_kernel_abs_f16",
        "ds4_kernel_abs_f32_4",
        "ds4_kernel_abs_f32_scalar",
        "ds4_kernel_argsort_f32_i32_desc_full",
        "ds4_kernel_argsort_merge_f32_i32_desc_full",
        "ds4_kernel_bin_fuse_f32_f32_f32",
        "ds4_kernel_concat",
        "ds4_kernel_cpy_f16_f32",
        "ds4_kernel_cpy_f32_f16",
        "ds4_kernel_cpy_f32_f32",
        "ds4_dsv4_attn_out_low_q8_0_f32",
        "ds4_dsv4_q8_hc_expand4_q8_0",
        "ds4_kernel_dsv4_rope_tail_f32",
        "ds4_dsv4_shared_down_hc_expand4_q8_0",
        "ds4_dsv4_shared_gate_up_swiglu_q8_0",
        "ds4_kernel_dsv4_softplus_sqrt_f32_4",
        "ds4_dsv4_topk_mask",
        "ds4_kernel_exp_f16",
        "ds4_kernel_exp_f32_4",
        "ds4_kernel_exp_f32_scalar",
        "ds4_kernel_flash_attn_ext_f16_dk512_dv512",
        "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512",
        "ds4_kernel_gelu_f16",
        "ds4_kernel_gelu_f32_4",
        "ds4_kernel_gelu_f32_scalar",
        "ds4_kernel_get_rows_f16",
        "ds4_kernel_get_rows_f32",
        "ds4_kernel_get_rows_i32",
        "ds4_kernel_hardsigmoid_f16",
        "ds4_kernel_hardsigmoid_f32_4",
        "ds4_kernel_hardsigmoid_f32_scalar",
        "ds4_kernel_hardswish_f16",
        "ds4_kernel_hardswish_f32_4",
        "ds4_kernel_hardswish_f32_scalar",
        "ds4_kernel_log_f16",
        "ds4_kernel_log_f32_4",
        "ds4_kernel_log_f32_scalar",
        "ds4_kernel_mul_mm_f16_f32",
        "ds4_kernel_mul_mm_id_iq2_xxs_f16",
        "ds4_kernel_mul_mm_id_iq2_xxs_f32",
        "ds4_kernel_mul_mm_id_map0_ne20_1",
        "ds4_kernel_mul_mm_id_map0_ne20_10",
        "ds4_kernel_mul_mm_id_map0_ne20_16",
        "ds4_kernel_mul_mm_id_map0_ne20_2",
        "ds4_kernel_mul_mm_id_map0_ne20_22",
        "ds4_kernel_mul_mm_id_map0_ne20_4",
        "ds4_kernel_mul_mm_id_map0_ne20_5",
        "ds4_kernel_mul_mm_id_map0_ne20_6",
        "ds4_kernel_mul_mm_id_map0_ne20_8",
        "ds4_kernel_mul_mm_id_q2_K_f16",
        "ds4_kernel_mul_mm_id_q2_K_f32",
        "ds4_kernel_mul_mm_id_q4_K_f16",
        "ds4_kernel_mul_mm_id_q4_K_f32",
        "ds4_kernel_mul_mm_id_q8_0_f16",
        "ds4_kernel_mul_mm_id_q8_0_f32",
        "ds4_kernel_mul_mm_q8_0_f32",
        "ds4_kernel_mul_mv_ext_f16_f32_r1_2",
        "ds4_kernel_mul_mv_ext_f16_f32_r1_3",
        "ds4_kernel_mul_mv_ext_f16_f32_r1_4",
        "ds4_kernel_mul_mv_ext_f16_f32_r1_5",
        "ds4_kernel_mul_mv_ext_q8_0_f32_r1_2",
        "ds4_kernel_mul_mv_ext_q8_0_f32_r1_3",
        "ds4_kernel_mul_mv_ext_q8_0_f32_r1_4",
        "ds4_kernel_mul_mv_ext_q8_0_f32_r1_5",
        "ds4_kernel_mul_mv_f16_f32",
        "ds4_kernel_mul_mv_f16_f32_4",
        "ds4_kernel_mul_mv_f16_f32_4_reduce",
        "ds4_kernel_mul_mv_f16_f32_pair_4",
        "ds4_kernel_mul_mv_f16_f32_reduce",
        "ds4_kernel_mul_mv_f16_f32_short",
        "ds4_kernel_mul_mv_f32_f32",
        "ds4_kernel_mul_mv_f32_f32_4",
        "ds4_kernel_mul_mv_f32_f32_4_reduce",
        "ds4_kernel_mul_mv_f32_f32_acc",
        "ds4_kernel_mul_mv_f32_f32_reduce",
        "ds4_kernel_mul_mv_f32_f32_setup",
        "ds4_kernel_mul_mv_f32_f32_short",
        "ds4_kernel_mul_mv_id_iq2_xxs_f32",
        "ds4_kernel_mul_mv_id_iq2_xxs_pair_f32",
        "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32",
        "ds4_kernel_mul_mv_id_q2_K_f32",
        "ds4_kernel_mul_mv_id_q2_K_sum6_f32",
        "ds4_kernel_mul_mv_id_q4_K_f32",
        "ds4_kernel_mul_mv_id_q4_K_pair_f32",
        "ds4_kernel_mul_mv_id_q4_K_pair_swiglu_f32",
        "ds4_kernel_mul_mv_id_q4_K_sum6_f32",
        "ds4_kernel_mul_mv_id_q8_0_f32",
        "ds4_kernel_mul_mv_q8_0_f32",
        "ds4_kernel_neg_f16",
        "ds4_kernel_neg_f32_4",
        "ds4_kernel_neg_f32_scalar",
        "ds4_kernel_relu_f16",
        "ds4_kernel_relu_f32_4",
        "ds4_kernel_relu_f32_scalar",
        "ds4_kernel_repeat_f32",
        "ds4_kernel_rms_norm_f32_4",
        "ds4_kernel_rms_norm_mul_f32_4",
        "ds4_kernel_set_rows_f32_i32",
        "ds4_kernel_sigmoid_f16",
        "ds4_kernel_sigmoid_f32_4",
        "ds4_kernel_sigmoid_f32_scalar",
        "ds4_kernel_silu_f16",
        "ds4_kernel_silu_f32_4",
        "ds4_kernel_silu_f32_scalar",
        "ds4_kernel_soft_max_f32",
        "ds4_kernel_soft_max_f32_4",
        "ds4_kernel_soft_max_f32_4_alibi_f16",
        "ds4_kernel_soft_max_f32_4_alibi_f16_sink",
        "ds4_kernel_soft_max_f32_4_alibi_f32",
        "ds4_kernel_soft_max_f32_4_alibi_f32_sink",
        "ds4_kernel_soft_max_f32_4_mask_f16",
        "ds4_kernel_soft_max_f32_4_mask_f16_sink",
        "ds4_kernel_soft_max_f32_4_mask_f32",
        "ds4_kernel_soft_max_f32_4_mask_f32_sink",
        "ds4_kernel_soft_max_f32_4_sink",
        "ds4_kernel_soft_max_f32_scalar",
        "ds4_kernel_soft_max_f32_scalar_alibi_f16",
        "ds4_kernel_soft_max_f32_scalar_alibi_f16_sink",
        "ds4_kernel_soft_max_f32_scalar_alibi_f32",
        "ds4_kernel_soft_max_f32_scalar_alibi_f32_sink",
        "ds4_kernel_soft_max_f32_scalar_mask_f16",
        "ds4_kernel_soft_max_f32_scalar_mask_f16_sink",
        "ds4_kernel_soft_max_f32_scalar_mask_f32",
        "ds4_kernel_soft_max_f32_scalar_mask_f32_sink",
        "ds4_kernel_soft_max_f32_scalar_sink",
        "ds4_kernel_sqr_f16",
        "ds4_kernel_sqr_f32_4",
        "ds4_kernel_sqr_f32_scalar",
        "ds4_kernel_step_f16",
        "ds4_kernel_step_f32_4",
        "ds4_kernel_step_f32_scalar",
        "ds4_kernel_sum_rows_f32_f32",
        "ds4_kernel_swiglu_f32",
        "ds4_kernel_tanh_f16",
        "ds4_kernel_tanh_f32_4",
        "ds4_kernel_tanh_f32_scalar",
        "ds4_kernel_unary_f16_f16",
        "ds4_kernel_unary_f32_f32",
        "ds4_kernel_unary_f32_f32_4",
        "ds4_matmul_transposed",
        "ds4_repeat",
        "ds4_rms_norm",
        "ds4_rope",
        "ds4_rope_dsv4",
        "ds4_set_rows",
        "ds4_softmax",
        "ds4_sum_rows",
        "ds4_unary_clamp",
        "ds4_unary_fill",
        "ds4_unary_scale",
        "ds4_unary_sigmoid",
        "ds4_unary_softplus",
        "ds4_unary_sqrt",
    ];

    #[test]
    fn every_metal_fn_matches_an_emitted_fixture() {
        let emitted: std::collections::HashSet<&str> =
            EMITTED_KERNEL_SYMBOLS.iter().copied().collect();
        for k in KERNELS {
            assert!(
                emitted.contains(k.metal_fn),
                "kernel registry metal_fn '{}' (semantic '{}') has no matching \
                 MLIR `func.name` in `crates/codegen_tests/src/bin/print_msl.rs` — \
                 typo, stale rename, or missing fixture? \
                 Refresh `EMITTED_KERNEL_SYMBOLS` by re-running \
                 `grep -hoE 'llvm.func @ds4_[a-zA-Z0-9_]+' print_msl.rs`.",
                k.metal_fn,
                k.name
            );
        }
    }

    #[test]
    fn buf_role_matches_uniform_layout() {
        // Cross-check: kernels with Struct(n) uniforms always have n≥1.
        // Total binding count (bufs + uniforms) must fit Metal's 31-slot
        // argument-table cap; the M123 packing rule kicks in only when
        // individual scalars would overflow that cap.
        for k in KERNELS {
            match k.uniforms {
                UniformLayout::Scalars(n) => {
                    let total = k.bufs.len() + (n as usize);
                    assert!(
                        total <= 31,
                        "kernel {} has bufs={} + scalars={} > 31 Metal cap \
                         — should pack into Struct (see M123)",
                        k.name,
                        k.bufs.len(),
                        n
                    );
                }
                UniformLayout::Struct(n) => {
                    assert!(n >= 1, "kernel {} has Struct(0) uniforms", k.name);
                }
            }
        }
    }
}
