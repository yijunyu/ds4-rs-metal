//! M5 Phase D — `compressor_decode_one_metal` end-to-end smoke.
//!
//! Compares the GPU orchestrator port against the CPU oracle
//! (`ds4_engine::attn_dispatch::compressor_decode_one`) for a
//! multi-position decode sequence covering at least two emit windows
//! (so the ratio==4 rotation logic and the cross-window pool layout
//! both get exercised).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    compressor_decode_one, CompressorInputs, CpuAttentionDispatcher, LayerParams,
};
use ds4_metal::compressor::compressor_decode_one_metal;
use ds4_metal::MetalDispatcher;

fn params_for(head_dim: u32, compress_ratio: u32, n_rot: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 64,
        n_hc: 1,
        n_head: 4,
        head_dim,
        n_rot,
        n_lora_q: 16,
        n_lora_kv: 16,
        hc_sinkhorn_iter: 0,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio,
        n_out_group: 2,
    }
}

fn synth_vec(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.013 + seed) * 1.3).sin() * scale)
        .collect()
}

/// Per-position bit-equivalence: run the CPU oracle and the GPU port
/// in lock-step over `n_positions` with identical inputs, identical
/// state slices. After every position, both state buffers must match,
/// and emits must agree.
fn run_sequence(
    n_positions: u32,
    in_dim: usize,
    head_dim: u32,
    compress_ratio: u32,
    n_rot: u32,
) {
    let disp_metal = MetalDispatcher::new().expect("MetalDispatcher::new");
    let disp_cpu = CpuAttentionDispatcher;
    let params = params_for(head_dim, compress_ratio, n_rot);

    let hd = head_dim as usize;
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * hd;
    let rows = if compress_ratio == 4 {
        2 * compress_ratio as usize
    } else {
        compress_ratio as usize
    };
    let state_len = rows * width;
    let ratio = compress_ratio as usize;

    // Weights and APE — shared by CPU + GPU paths.
    let w_kv = synth_vec(width * in_dim, 0.10, 0.05);
    let w_gate = synth_vec(width * in_dim, 0.27, 0.05);
    let w_ape = synth_vec(width * ratio, 0.41, 0.10);
    let w_norm: Vec<f32> = (0..hd).map(|i| 1.0 + 0.003 * (i as f32)).collect();

    let comp = CompressorInputs {
        w_kv: &w_kv,
        w_gate: &w_gate,
        w_kv_f16: None,
        w_gate_f16: None,
        w_ape: &w_ape,
        w_norm: &w_norm,
        head_dim,
        compress_ratio,
    };

    // State buffers — initial CPU semantics: kv all-zero, score all -1e9
    // (matches `decode_step.rs:493`).
    let mut cpu_state_kv = vec![0.0f32; state_len];
    let mut cpu_state_score = vec![-1.0e9f32; state_len];
    let mut gpu_state_kv = vec![0.0f32; state_len];
    let mut gpu_state_score = vec![-1.0e9f32; state_len];

    for pos in 0..n_positions {
        let x = synth_vec(in_dim, 0.07 * (pos as f32 + 1.0), 0.3);

        let cpu_emit = compressor_decode_one(
            &disp_cpu,
            &params,
            &comp,
            &x,
            &mut cpu_state_kv,
            &mut cpu_state_score,
            pos,
        );
        let gpu_emit = compressor_decode_one_metal(
            &disp_metal,
            &params,
            &comp,
            &x,
            &mut gpu_state_kv,
            &mut gpu_state_score,
            pos,
            0,
            false,
        );

        // States must stay in lock-step after every position. Use
        // tolerance because the GPU matvec uses float4 + threadgroup
        // reductions; the CPU oracle uses scalar left-fold.
        let mut max_abs_kv: f32 = 0.0;
        let mut max_abs_sc: f32 = 0.0;
        for i in 0..state_len {
            // Skip the unwritten "prev window" rows for ratio==4 at
            // pos < ratio — both sides still hold initial values
            // (-1e9 score, 0 kv) so they're trivially equal, but the
            // sentinel itself differs by 0 ULP so no need to special-case.
            max_abs_kv = max_abs_kv.max((gpu_state_kv[i] - cpu_state_kv[i]).abs());
            max_abs_sc = max_abs_sc.max((gpu_state_score[i] - cpu_state_score[i]).abs());
        }
        assert!(
            max_abs_kv < 1e-3,
            "state_kv divergence at pos={pos}: max_abs={max_abs_kv:.3e}",
        );
        assert!(
            max_abs_sc < 1e-3,
            "state_score divergence at pos={pos}: max_abs={max_abs_sc:.3e}",
        );

        match (cpu_emit, gpu_emit) {
            (Some(cpu_row), Some(gpu_row)) => {
                assert_eq!(cpu_row.len(), gpu_row.len(), "emit row len mismatch at pos={pos}");
                let mut max_abs: f32 = 0.0;
                let mut max_ref: f32 = 0.0;
                for i in 0..cpu_row.len() {
                    max_abs = max_abs.max((cpu_row[i] - gpu_row[i]).abs());
                    max_ref = max_ref.max(cpu_row[i].abs());
                }
                let rel = max_abs / max_ref.max(1e-6);
                assert!(
                    max_abs < 5e-4 && rel < 5e-4,
                    "emit row drift at pos={pos}: max_abs={max_abs:.3e} \
                     max_ref={max_ref:.3e} rel={rel:.3e}",
                );
            }
            (None, None) => {} // both agree this position is a no-emit
            (cpu, gpu) => {
                panic!(
                    "emit-or-not disagreement at pos={pos}: cpu={} gpu={}",
                    cpu.is_some(),
                    gpu.is_some()
                );
            }
        }
    }
}

#[test]
fn compressor_decode_one_metal_matches_cpu_oracle_ratio4() {
    // V4 Flash main compressor dims: head_dim=DS4_N_HEAD_DIM=512,
    // ratio=4, in_dim=DS4_N_LORA_KV=512. Cover 8 positions so the
    // emit at pos=3 and pos=7 both run (and the rotation between
    // them exercises the two-window pool layout).
    run_sequence(
        /*n_positions=*/ 8, /*in_dim=*/ 512, /*head_dim=*/ 512,
        /*compress_ratio=*/ 4, /*n_rot=*/ 64,
    );
}

#[test]
fn compressor_decode_one_metal_matches_cpu_oracle_indexer_dims() {
    // Indexer compressor dims: head_dim=DS4_N_INDEXER_HEAD_DIM=128,
    // ratio=4, in_dim=DS4_N_LORA_KV=512.
    run_sequence(
        /*n_positions=*/ 8, /*in_dim=*/ 512, /*head_dim=*/ 128,
        /*compress_ratio=*/ 4, /*n_rot=*/ 64,
    );
}

#[test]
fn compressor_decode_one_metal_matches_cpu_oracle_ratio2() {
    // ratio!=4 path: coff=1, width=head_dim. Uses the standard
    // `softmax_pool` kernel instead of `compressor_pool_ratio4`.
    // No rotation at the end.
    run_sequence(
        /*n_positions=*/ 6, /*in_dim=*/ 256, /*head_dim=*/ 64,
        /*compress_ratio=*/ 2, /*n_rot=*/ 16,
    );
}

#[test]
fn compressor_decode_one_metal_handles_no_rope_tail() {
    // n_rot==0 skips the rope_tail call. Verify the path still works.
    run_sequence(
        /*n_positions=*/ 8, /*in_dim=*/ 256, /*head_dim=*/ 128,
        /*compress_ratio=*/ 4, /*n_rot=*/ 0,
    );
}
