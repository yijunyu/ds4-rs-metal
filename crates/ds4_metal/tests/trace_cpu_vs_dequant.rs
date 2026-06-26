//! End-to-end contract tests for `CpuViaDequantDispatcher`.
//!
//! Builds a synthetic 2-layer DS4-shaped model whose MoE expert weights
//! come from synthetic q4_K bytes, then runs `decode_step_with` twice:
//!   1. With `TracingDispatcher::new(&CpuDispatcher)` — the f32 reference.
//!   2. With `TracingDispatcher::new(&CpuViaDequantDispatcher::f32_reference(&qews))`
//!      — same f32 outputs but the MoE leg goes through the dequant oracle.
//!
//! The f32-reference trace must agree bit-for-bit. The default dispatcher path
//! is antirez-parity mode and intentionally includes Q8_K activation rounding.

#![cfg(not(target_os = "macos"))]

use ds4_engine::decode_step::{decode_step_with, DecodeConfig, LayerWeights, ModelWeights};
use ds4_engine::dispatch::{check_traces_close, CpuDispatcher, TracingDispatcher};
use ds4_engine::gguf::GgmlType;
use ds4_metal::cpu_via_dequant::CpuViaDequantDispatcher;
use ds4_metal::quantized_experts::{QuantTensor, QuantizedExpertWeights};

/// One synthetic q4_K block (144 B / 256 weights). Layout matches
/// `dequant_q4_k_blocks`'s expectation: half d (0x3C00 = 1.0), half dmin
/// (0x3800 = 0.5), 12 scale bytes, 128 nibble bytes.
fn make_q4_k_block(seed: u8) -> Vec<u8> {
    let mut blk = vec![0u8; 144];
    blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    for i in 0..12 {
        blk[4 + i] = seed.wrapping_add(i as u8).wrapping_mul(7);
    }
    for i in 0..128 {
        blk[16 + i] = seed.wrapping_add(i as u8).wrapping_mul(13);
    }
    blk
}

/// One synthetic q2_K block (84 B / 256 weights). Layout: scales[16],
/// qs[64], half d (0x3C00 = 1.0), half dmin (0x3800 = 0.5) — d/dmin at END,
/// opposite of q4_K.
fn make_q2_k_block(seed: u8) -> Vec<u8> {
    let mut blk = vec![0u8; 84];
    for i in 0..16 {
        blk[i] = seed.wrapping_add(i as u8).wrapping_mul(11);
    }
    for i in 0..64 {
        blk[16 + i] = seed.wrapping_add(i as u8).wrapping_mul(17);
    }
    blk[80..82].copy_from_slice(&0x3C00u16.to_le_bytes());
    blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
    blk
}

/// One synthetic iq2_xxs block (66 B / 256 weights). Layout: half d, qs[32]
/// as little-endian uint16 stream (64 bytes).
fn make_iq2_xxs_block(seed: u8) -> Vec<u8> {
    let mut blk = vec![0u8; 66];
    blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 0..64 {
        blk[2 + i] = seed.wrapping_add(i as u8).wrapping_mul(19);
    }
    blk
}

fn make_block_for(ttype: GgmlType, seed: u8) -> Vec<u8> {
    match ttype {
        GgmlType::Q4_K => make_q4_k_block(seed),
        GgmlType::Q2_K => make_q2_k_block(seed),
        GgmlType::IQ2_XXS => make_iq2_xxs_block(seed),
        other => panic!("unsupported synth block type: {other:?}"),
    }
}

fn bytes_per_row_for(ttype: GgmlType) -> u64 {
    match ttype {
        GgmlType::Q4_K => 144,
        GgmlType::Q2_K => 84,
        GgmlType::IQ2_XXS => 66,
        other => panic!("unsupported synth block type: {other:?}"),
    }
}

/// Build a synthetic `QuantizedExpertWeights` from concatenated blocks of
/// `ttype`. One block per row (so `d_in == d_ffn == 256`).
fn synth_qew(layer_idx: u32, n_experts: u32, salt: u8, ttype: GgmlType) -> QuantizedExpertWeights {
    let d_in: u32 = 256;
    let d_ffn: u32 = 256;
    let rows = d_ffn;
    let bytes_per_row = bytes_per_row_for(ttype);
    let stride = rows as u64 * bytes_per_row;
    let mk_stream = |s: u8| -> Vec<u8> {
        let n = (n_experts * rows) as usize;
        let mut v = Vec::with_capacity(n * bytes_per_row as usize);
        for i in 0..n {
            v.extend_from_slice(&make_block_for(ttype, s.wrapping_add(i as u8)));
        }
        v
    };
    QuantizedExpertWeights {
        layer_idx,
        n_experts,
        d_in,
        d_ffn,
        gate: QuantTensor {
            ttype,
            dims: [n_experts as u64, d_ffn as u64, d_in as u64],
            bytes: mk_stream(salt.wrapping_add(0x10)),
            expert_stride: stride,
        },
        up: QuantTensor {
            ttype,
            dims: [n_experts as u64, d_ffn as u64, d_in as u64],
            bytes: mk_stream(salt.wrapping_add(0x20)),
            expert_stride: stride,
        },
        down: QuantTensor {
            ttype,
            dims: [n_experts as u64, d_in as u64, d_ffn as u64],
            bytes: mk_stream(salt.wrapping_add(0x30)),
            expert_stride: stride,
        },
    }
}

/// Build a `LayerWeights` whose expert slabs are the f32 dequant of `qew`.
/// This makes the CPU path see the same numerical inputs the dequant path
/// sees, so the trace comparison can be bit-exact.
fn layer_from_qew(qew: &QuantizedExpertWeights) -> LayerWeights {
    let d = qew.d_in as usize;
    let n_experts = qew.n_experts as usize;
    let n_used = 2;
    let gate = qew.dequant_gate_f32().expect("dequant gate");
    let up = qew.dequant_up_f32().expect("dequant up");
    let down = qew.dequant_down_f32().expect("dequant down");
    LayerWeights {
        d_model: d,
        d_ffn: qew.d_ffn as usize,
        n_experts,
        n_experts_used: n_used,
        attn_norm_gamma: vec![1.0; d],
        w_attn: vec![0.0; d * d], // residual passthrough
        ffn_norm_gamma: vec![1.0; d],
        w_router: vec![0.001; n_experts * d],
        w_router_f16: Vec::new().into(),
        router_bias: vec![0.0; n_experts],
        w_gate_exps: gate,
        w_up_exps: up,
        w_down_exps: down,
        routing_table: None,
    }
}

fn run_trace_equality_for(ttype: GgmlType) {
    // Two synthetic MoE layers, each with 4 experts × (256→256).
    let qew0 = synth_qew(0, 4, 0x10, ttype);
    let qew1 = synth_qew(1, 4, 0x80, ttype);
    let qews = vec![qew0, qew1];
    let layer0 = layer_from_qew(&qews[0]);
    let layer1 = layer_from_qew(&qews[1]);
    let d = layer0.d_model;
    let vocab = 8;
    let mut lm_head = vec![0.0f32; vocab * d];
    for v in 0..vocab {
        lm_head[v * d + (v % d)] = 1.0;
    }
    let model = ModelWeights {
        layers: vec![layer0, layer1],
        final_norm_gamma: vec![1.0; d],
        lm_head,
        vocab_size: vocab,
        d_model: d,
    };

    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013) - 0.7).collect();
    let cfg = DecodeConfig::default();

    let cpu = CpuDispatcher;
    let tracer_cpu = TracingDispatcher::new(&cpu);
    let logits_cpu = decode_step_with(&tracer_cpu, x.clone(), &model, &cfg).unwrap();

    let dequant = CpuViaDequantDispatcher::f32_reference(&qews);
    let tracer_dq = TracingDispatcher::new(&dequant);
    let logits_dq = decode_step_with(&tracer_dq, x.clone(), &model, &cfg).unwrap();

    // Logits agree bit-for-bit.
    assert_eq!(logits_cpu.len(), logits_dq.len());
    for (i, (a, b)) in logits_cpu.iter().zip(logits_dq.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{ttype:?} logit drift at idx {i}: cpu={a} dequant={b}"
        );
    }

    // Traces are byte-identical (tol=0). The CPU and dequant paths see the
    // same f32 expert slabs (since LayerWeights got them from the same
    // dequant call); the only difference is the call shape. With tol=0
    // any mismatch flags a regression in the dequant oracle's bit-exactness.
    let trace_cpu = tracer_cpu.events();
    let trace_dq = tracer_dq.events();
    check_traces_close(&trace_cpu, &trace_dq, 0.0)
        .unwrap_or_else(|e| panic!("{ttype:?} traces diverge: {e}"));

    // Sanity: 2 layers → 7N+2 = 16 events.
    assert_eq!(trace_cpu.len(), 7 * 2 + 2);
    assert_eq!(trace_dq.len(), 7 * 2 + 2);
}

fn run_trace_inequality_for(ttype: GgmlType) {
    let qew0 = synth_qew(0, 4, 0x10, ttype);
    let qew1 = synth_qew(1, 4, 0x80, ttype);
    let layer0 = layer_from_qew(&qew0);
    let layer1 = layer_from_qew(&qew1);
    let d = layer0.d_model;
    let model = ModelWeights {
        layers: vec![layer0, layer1],
        final_norm_gamma: vec![1.0; d],
        lm_head: {
            let vocab = 8;
            let mut h = vec![0.0f32; vocab * d];
            for v in 0..vocab {
                h[v * d + (v % d)] = 1.0;
            }
            h
        },
        vocab_size: 8,
        d_model: d,
    };

    // Swap salts so the dispatcher's bytes != the layer's f32 slabs.
    let qew0_swapped = synth_qew(0, 4, 0x11, ttype);
    let qew1_swapped = synth_qew(1, 4, 0x81, ttype);
    let qews = vec![qew0_swapped, qew1_swapped];

    let cfg = DecodeConfig::default();
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013) - 0.7).collect();

    let cpu = CpuDispatcher;
    let tracer_cpu = TracingDispatcher::new(&cpu);
    let _ = decode_step_with(&tracer_cpu, x.clone(), &model, &cfg).unwrap();

    let dequant = CpuViaDequantDispatcher::f32_reference(&qews);
    let tracer_dq = TracingDispatcher::new(&dequant);
    let _ = decode_step_with(&tracer_dq, x, &model, &cfg).unwrap();

    let trace_cpu = tracer_cpu.events();
    let trace_dq = tracer_dq.events();
    let res = check_traces_close(&trace_cpu, &trace_dq, 0.0);
    let err = res.err().unwrap_or_else(|| {
        panic!("{ttype:?}: expected trace divergence with swapped expert bytes")
    });
    // The first divergence must be a moe_routed_step output (the only thing
    // that depends on `qews`); rms_norm/matvec/softplus_sqrt/router_finalize
    // are unchanged because the layer's f32 inputs are identical.
    assert!(
        err.contains("moe_routed_step"),
        "{ttype:?}: expected moe_routed_step divergence, got: {err}"
    );
}

fn run_default_antirez_q8k_differs_from_f32_for(ttype: GgmlType, min_diff: f32) {
    let qew0 = synth_qew(0, 4, 0x10, ttype);
    let qew1 = synth_qew(1, 4, 0x80, ttype);
    let qews = vec![qew0, qew1];
    let layer0 = layer_from_qew(&qews[0]);
    let layer1 = layer_from_qew(&qews[1]);
    let d = layer0.d_model;
    let vocab = 8;
    let mut lm_head = vec![0.0f32; vocab * d];
    for v in 0..vocab {
        lm_head[v * d + (v % d)] = 1.0;
    }
    let model = ModelWeights {
        layers: vec![layer0, layer1],
        final_norm_gamma: vec![1.0; d],
        lm_head,
        vocab_size: vocab,
        d_model: d,
    };

    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013) - 0.7).collect();
    let cfg = DecodeConfig::default();

    let cpu = CpuDispatcher;
    let logits_cpu = decode_step_with(&cpu, x.clone(), &model, &cfg).unwrap();

    let dequant = CpuViaDequantDispatcher::new(&qews);
    let logits_dq = decode_step_with(&dequant, x, &model, &cfg).unwrap();

    let max_abs_diff = logits_cpu
        .iter()
        .zip(logits_dq.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff > min_diff,
        "{ttype:?}: default antirez Q8_K activation path unexpectedly matched f32 reference; max_abs_diff={max_abs_diff}"
    );
}

#[test]
fn math_reference_trace_equality_cpu_vs_dequant_q4_k() {
    run_trace_equality_for(GgmlType::Q4_K);
}

#[test]
fn math_reference_trace_equality_cpu_vs_dequant_q2_k() {
    run_trace_equality_for(GgmlType::Q2_K);
}

#[test]
fn math_reference_trace_equality_cpu_vs_dequant_iq2_xxs() {
    run_trace_equality_for(GgmlType::IQ2_XXS);
}

#[test]
fn antirez_parity_q8k_activation_path_differs_from_f32_iq2_xxs() {
    run_default_antirez_q8k_differs_from_f32_for(GgmlType::IQ2_XXS, 0.01);
}

#[test]
fn math_reference_trace_inequality_flagged_when_dequant_table_swapped_q4_k() {
    run_trace_inequality_for(GgmlType::Q4_K);
}

#[test]
fn math_reference_trace_inequality_flagged_when_dequant_table_swapped_q2_k() {
    run_trace_inequality_for(GgmlType::Q2_K);
}

#[test]
fn math_reference_trace_inequality_flagged_when_dequant_table_swapped_iq2_xxs() {
    run_trace_inequality_for(GgmlType::IQ2_XXS);
}
