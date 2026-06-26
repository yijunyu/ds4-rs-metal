//! Phase 3 Step 5 — base-model per-layer weight bundles for the
//! K-position verifier.
//!
//! `encode_verify_layers_K` takes `&[BaseLayerVerifyBundle<'_>]` — 43
//! reference-bundles, one per base decode layer. Each bundle has 17
//! DeferredBuf refs (norms + gammas + Q8_0 weights) + 6 CPU slice refs
//! (shared-expert tensors) + 1 LayerParams ref.
//!
//! This module bridges `ds4_engine::decode_step::ComposedModelWeights`
//! → the per-scope DeferredBuf uploads + the reference-struct
//! construction the verifier needs.
//!
//! ## Pattern
//!
//! ```ignore
//! // Once per scope:
//! let scope = disp.batch_scope();
//! let owned_layers = upload_base_layers_to_scope(&scope, &model);
//!
//! // Per spec-decode iter (within the same scope):
//! let layer_bundles: Vec<BaseLayerVerifyBundle> = (0..n_layers)
//!     .map(|i| owned_layers[i].as_verify_bundle(&model.layers[i]))
//!     .collect();
//!
//! let cur_hc_K = scope.encode_verify_layers_K(&prev_hc_K_db, &layer_bundles, ...)?;
//! ```

#![cfg(target_os = "macos")]

use ds4_engine::decode_step::{ComposedModelWeights, ComposedLayerWeights};

use crate::deferred::{BaseLayerVerifyBundle, BatchScope, DeferredBuf};

/// Per-layer OWNED DeferredBufs uploaded to a scope. Constructed by
/// `upload_base_layers_to_scope`; consumed by `as_verify_bundle` which
/// builds the reference-bundle for `encode_verify_layers_K`.
pub struct BaseLayerOwned {
    // Attn-half DeferredBufs (11).
    pub hc_attn_fn:    DeferredBuf,
    pub hc_attn_scale: DeferredBuf,
    pub hc_attn_base:  DeferredBuf,
    pub attn_norm:     DeferredBuf,
    pub attn_q_a_q8:   DeferredBuf,
    pub gamma_q:       DeferredBuf,
    pub attn_q_b_q8:   DeferredBuf,
    pub attn_kv_q8:    DeferredBuf,
    pub gamma_kv:      DeferredBuf,
    pub attn_sinks:    DeferredBuf,
    pub w_o_a_q8:      DeferredBuf,
    pub w_o_b_q8:      DeferredBuf,
    // FFN-half DeferredBufs (6).
    pub hc_ffn_fn:     DeferredBuf,
    pub hc_ffn_scale:  DeferredBuf,
    pub hc_ffn_base:   DeferredBuf,
    pub ffn_norm:      DeferredBuf,
    pub w_router:      DeferredBuf,
    pub router_bias:   DeferredBuf,
}

impl BaseLayerOwned {
    /// Construct the verifier reference-bundle from this OWNED layer's
    /// DeferredBufs + the corresponding `ComposedLayerWeights` (for the
    /// shared-expert CPU slices + layer_params).
    pub fn as_verify_bundle<'a>(
        &'a self,
        composed_layer: &'a ComposedLayerWeights,
    ) -> BaseLayerVerifyBundle<'a> {
        BaseLayerVerifyBundle {
            hc_attn_fn:    &self.hc_attn_fn,
            hc_attn_scale: &self.hc_attn_scale,
            hc_attn_base:  &self.hc_attn_base,
            attn_norm:     &self.attn_norm,
            attn_q_a_q8:   &self.attn_q_a_q8,
            gamma_q:       &self.gamma_q,
            attn_q_b_q8:   &self.attn_q_b_q8,
            attn_kv_q8:    &self.attn_kv_q8,
            gamma_kv:      &self.gamma_kv,
            attn_sinks:    &self.attn_sinks,
            w_o_a_q8:      &self.w_o_a_q8,
            w_o_b_q8:      &self.w_o_b_q8,
            hc_ffn_fn:     &self.hc_ffn_fn,
            hc_ffn_scale:  &self.hc_ffn_scale,
            hc_ffn_base:   &self.hc_ffn_base,
            ffn_norm:      &self.ffn_norm,
            w_router:      &self.w_router,
            router_bias:   &self.router_bias,
            // Shared-expert CPU slices live in the engine's layer weights.
            sh_w_gate:    &composed_layer.attn.w_shared_gate,
            sh_w_up:      &composed_layer.attn.w_shared_up,
            sh_w_down:    &composed_layer.attn.w_shared_down,
            sh_w_gate_q8: &composed_layer.attn.w_shared_gate_q8,
            sh_w_up_q8:   &composed_layer.attn.w_shared_up_q8,
            sh_w_down_q8: &composed_layer.attn.w_shared_down_q8,
            // Compressor + indexer CPU slices (DS4_VERIFY_COMPRESSOR path).
            attn_compressor_kv:      &composed_layer.attn.attn_compressor_kv,
            attn_compressor_gate:    &composed_layer.attn.attn_compressor_gate,
            attn_compressor_ape:     &composed_layer.attn.attn_compressor_ape,
            attn_compressor_norm:    &composed_layer.attn.attn_compressor_norm,
            indexer_compressor_kv:   &composed_layer.attn.indexer_compressor_kv,
            indexer_compressor_gate: &composed_layer.attn.indexer_compressor_gate,
            indexer_compressor_ape:  &composed_layer.attn.indexer_compressor_ape,
            indexer_compressor_norm: &composed_layer.attn.indexer_compressor_norm,
            // No-copy F16 kv/gate bytes (lean weights skip the f32 above).
            attn_compressor_kv_f16:      composed_layer.attn.attn_compressor_f16().0,
            attn_compressor_gate_f16:    composed_layer.attn.attn_compressor_f16().1,
            indexer_compressor_kv_f16:   composed_layer.attn.indexer_compressor_f16().0,
            indexer_compressor_gate_f16: composed_layer.attn.indexer_compressor_f16().1,
            attn_sinks_cpu: &composed_layer.attn.attn_sinks,
            // Layer-specific rope/scaling params.
            layer_params: &composed_layer.attn.params,
        }
    }
}

/// Upload ALL base-model layer weights to the given scope. Returns a
/// `Vec<BaseLayerOwned>` of length `model.layers.len()` (=43 for DS4
/// V4 Flash). First upload pays the bytes; subsequent calls reuse via
/// `cached_weight_buffer` (identity-keyed).
///
/// The shared-expert weights live as CPU slices in
/// `model.layers[i].attn.w_shared_*` and are NOT uploaded here — they
/// flow into the verifier via `as_verify_bundle`'s ref handoff.
///
/// Caller usage:
///
/// ```ignore
/// let owned = upload_base_layers_to_scope(&scope, &model);
/// let bundles: Vec<_> = (0..model.layers.len())
///     .map(|i| owned[i].as_verify_bundle(&model.layers[i]))
///     .collect();
/// scope.encode_verify_layers_K(&prev_hc_K_db, &bundles, ...)
/// ```
pub fn upload_base_layers_to_scope(
    scope: &BatchScope<'_>,
    model: &ComposedModelWeights,
) -> Vec<BaseLayerOwned> {
    model
        .layers
        .iter()
        .map(|layer| upload_one_layer(scope, layer))
        .collect()
}

fn upload_one_layer(scope: &BatchScope<'_>, layer: &ComposedLayerWeights) -> BaseLayerOwned {
    let attn = &layer.attn;
    let moe = &layer.moe;
    let params = &attn.params;

    let n_lora_q = params.n_lora_q as usize;
    let d_embd = params.d_embd as usize;
    let q_dim = (params.n_head as usize) * (params.head_dim as usize);
    let n_lora_kv = params.n_lora_kv as usize;
    let n_groups = params.n_out_group as usize;
    let n_lora_o = 1024usize; // DS4_N_LORA_O — antirez ds4.c:93
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let group_dim = head_dim * (n_head / n_groups);
    let out_low_dim = n_groups * n_lora_o;

    BaseLayerOwned {
        // Verifier has no f16 hc-collapse kernel — reconstruct f32 from the
        // lean f16 bytes (no-op Cow::Borrowed when the f32 is present).
        hc_attn_fn:    scope.weight_f32(&attn.hc_attn_fn_as_f32()),
        hc_attn_scale: scope.weight_f32(&attn.hc_attn_scale),
        hc_attn_base:  scope.weight_f32(&attn.hc_attn_base),
        attn_norm:     scope.weight_f32(&attn.hc_norm_gamma),
        attn_q_a_q8:   scope.weight_q8_0_raw(&attn.attn_q_a_q8, n_lora_q * d_embd),
        gamma_q:       scope.weight_f32(&attn.qkv_gamma_q),
        attn_q_b_q8:   scope.weight_q8_0_raw(&attn.attn_q_b_q8, q_dim * n_lora_q),
        attn_kv_q8:    scope.weight_q8_0_raw(&attn.attn_kv_q8, n_lora_kv * d_embd),
        gamma_kv:      scope.weight_f32(&attn.qkv_gamma_kv),
        attn_sinks:    scope.weight_f32(&attn.attn_sinks),
        w_o_a_q8:      scope.weight_q8_0_raw(&attn.w_o_a_q8, out_low_dim * group_dim),
        w_o_b_q8:      scope.weight_q8_0_raw(&attn.w_o_b_q8, d_embd * out_low_dim),
        hc_ffn_fn:     scope.weight_f32(&attn.hc_ffn_fn),
        hc_ffn_scale:  scope.weight_f32(&attn.hc_ffn_scale),
        hc_ffn_base:   scope.weight_f32(&attn.hc_ffn_base),
        ffn_norm:      scope.weight_f32(&attn.hc_ffn_norm_gamma),
        w_router:      scope.weight_f32(&moe.w_router_as_f32()),
        router_bias:   scope.weight_f32(&moe.router_bias),
    }
}
