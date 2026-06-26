//! CPU dispatcher that routes MoE through the dequant oracle.
//!
//! `CpuViaDequantDispatcher` exists for quantized expert validation: it forwards
//! rms_norm / matvec_f32 / softplus_sqrt / router_finalize to the plain
//! `CpuDispatcher`, but overrides `moe_routed_step` to dispatch through
//! `moe_routed_step_cpu_via_dequant`, which dequantizes a preloaded
//! `QuantizedExpertWeights` table on every call. By default, routed-MoE
//! activations are Q8_K-rounded to match antirez's quantized matvec path.
//!
//! The point is to exercise the same byte stream the Metal dispatcher
//! consumes (`QuantizedExpertWeights::bytes`) without needing a Mac. When
//! AWS mac2-m1ultra capacity opens, we'll add a `MetalDispatcher` arm to
//! the same harness; until then this gives a Linux-buildable oracle for the
//! quant-byte plumbing and antirez activation rounding.

use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};

use crate::quantized_experts::{
    moe_routed_step_cpu_via_dequant_with_activation, ActivationQuant, QuantizedExpertWeights,
};

/// Dispatcher whose `moe_routed_step` dequantizes a preloaded per-layer
/// quantized expert table, optionally Q8_K-rounds activations, and dispatches
/// through the CPU implementation. All other methods are CPU pass-through.
pub struct CpuViaDequantDispatcher<'a> {
    cpu: CpuDispatcher,
    expert_weights: &'a [QuantizedExpertWeights],
    activation_quant: ActivationQuant,
}

impl<'a> CpuViaDequantDispatcher<'a> {
    /// `expert_weights[i]` is consulted whenever `moe_routed_step` fires for
    /// `layer_idx == i`. Callers must size the slice to cover every MoE
    /// layer the decoded model has.
    pub fn new(expert_weights: &'a [QuantizedExpertWeights]) -> Self {
        Self::with_activation_quant(expert_weights, ActivationQuant::default())
    }

    pub fn with_activation_quant(
        expert_weights: &'a [QuantizedExpertWeights],
        activation_quant: ActivationQuant,
    ) -> Self {
        Self {
            cpu: CpuDispatcher,
            expert_weights,
            activation_quant,
        }
    }

    /// Compatibility mode for tests that compare dequantized bytes against
    /// `ds4_engine`'s f32 MoE reference rather than antirez's Q8_K path.
    pub fn f32_reference(expert_weights: &'a [QuantizedExpertWeights]) -> Self {
        Self::with_activation_quant(expert_weights, ActivationQuant::F32)
    }
}

impl<'a> KernelDispatcher for CpuViaDequantDispatcher<'a> {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.cpu.rms_norm(x, gamma, eps)
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        self.cpu.matvec_f32(w, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.cpu.softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        self.cpu.router_finalize(probs, bias, k)
    }

    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        _experts_w_gate: &[f32],
        _experts_w_up: &[f32],
        _experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        if std::env::var("DS4_ZERO_ROUTED_MOE").is_ok() {
            return vec![0.0_f32; x.len()];
        }
        let qew = self
            .expert_weights
            .get(layer_idx as usize)
            .unwrap_or_else(|| {
                panic!(
                    "CpuViaDequantDispatcher::moe_routed_step: no QuantizedExpertWeights \
                 for layer_idx {layer_idx} (have {} entries)",
                    self.expert_weights.len()
                )
            });
        moe_routed_step_cpu_via_dequant_with_activation(
            qew,
            x,
            selected,
            weights,
            d_ffn,
            self.activation_quant,
        )
        .unwrap_or_else(|e| {
            panic!("CpuViaDequantDispatcher::moe_routed_step layer {layer_idx}: {e}")
        })
    }
}
