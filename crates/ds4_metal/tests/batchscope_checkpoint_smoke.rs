//! M5 task #100 foundation — `BatchScope::commit_wait_read_multi` smoke.
//!
//! Demonstrates the "checkpoint" semantic:
//!   - encode some ops → checkpoint (read one intermediate to CPU)
//!   - keep encoding more ops in the SAME scope, using DeferredBufs
//!     allocated BEFORE the checkpoint as inputs
//!   - final flush
//!
//! Asserts:
//!   1. The checkpoint readback matches a single-shot baseline.
//!   2. The post-checkpoint output matches a single-shot baseline that
//!      ran the same op sequence with no intermediate flush.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use ds4_metal::MetalDispatcher;

const D_IN: usize = 512;
const D_OUT: usize = 256;

fn deterministic_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..D_IN).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
    let w: Vec<f32> = (0..D_IN * D_OUT)
        .map(|i| ((i as f32) * 0.007).cos() * 0.05)
        .collect();
    let gamma: Vec<f32> = (0..D_IN).map(|i| 1.0 + (i as f32) * 0.0011).collect();
    (x, w, gamma)
}

#[test]
fn commit_wait_read_multi_keeps_scope_alive_across_checkpoint() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuDispatcher;
    let (x, w, gamma) = deterministic_inputs();
    let eps = 1e-5_f32;

    // ── Baseline: single-shot scope, both ops, separate readbacks via
    //   flush_and_read_multi.
    let baseline_mv;
    let baseline_rms;
    {
        let scope = disp.batch_scope();
        let x_db = scope.upload_f32(&x);
        let w_db = scope.upload_f32(&w);
        let g_db = scope.upload_f32(&gamma);
        let mv = scope.matvec_f32(&w_db, &x_db, D_IN, D_OUT).expect("mv");
        let rms = scope.rms_norm_mul(&x_db, &g_db, eps).expect("rms");
        let outs = scope.flush_and_read_multi(&[&mv, &rms]);
        baseline_mv = outs[0].clone();
        baseline_rms = outs[1].clone();
    }

    // ── Checkpoint path: encode matvec, checkpoint to read it, then
    //   encode rms_norm_mul using x_db allocated BEFORE the checkpoint.
    let (cp_mv, cp_rms) = {
        let mut scope = disp.batch_scope();
        let x_db = scope.upload_f32(&x);
        let w_db = scope.upload_f32(&w);
        let g_db = scope.upload_f32(&gamma);
        let mv = scope.matvec_f32(&w_db, &x_db, D_IN, D_OUT).expect("mv");
        // Checkpoint: read mv, keep scope alive.
        let mid = scope.commit_wait_read_multi(&[&mv]);
        let mv_read = mid.into_iter().next().expect("mv readback");
        // Encode another op using x_db (allocated pre-checkpoint).
        let rms = scope.rms_norm_mul(&x_db, &g_db, eps).expect("rms");
        let rms_read = scope.flush_and_read(&rms);
        (mv_read, rms_read)
    };

    // ── Correctness checks.
    let cpu_mv = cpu.matvec_f32(&w, &x, D_OUT);
    assert_eq!(cp_mv.len(), D_OUT);
    assert_eq!(baseline_mv.len(), D_OUT);
    for i in 0..D_OUT {
        let r = cpu_mv[i];
        assert!(
            (cp_mv[i] - r).abs() < 1e-3,
            "checkpoint mv[{i}] {} vs cpu {r}",
            cp_mv[i]
        );
        assert!(
            (baseline_mv[i] - r).abs() < 1e-3,
            "baseline mv[{i}] {} vs cpu {r}",
            baseline_mv[i]
        );
    }

    // rms_norm_mul: (x / rms(x, eps)) * gamma. CpuDispatcher exposes
    // rms_norm directly; compare element-wise.
    let cpu_rms = cpu.rms_norm(&x, &gamma, eps);
    assert_eq!(cp_rms.len(), D_IN);
    assert_eq!(baseline_rms.len(), D_IN);
    for i in 0..D_IN {
        let r = cpu_rms[i];
        assert!(
            (cp_rms[i] - r).abs() < 1e-4,
            "checkpoint rms[{i}] {} vs cpu {r}",
            cp_rms[i]
        );
        assert!(
            (baseline_rms[i] - r).abs() < 1e-4,
            "baseline rms[{i}] {} vs cpu {r}",
            baseline_rms[i]
        );
    }

    // Checkpoint and baseline must agree numerically (same kernels,
    // only the cb boundary differs).
    for i in 0..D_OUT {
        assert!(
            (cp_mv[i] - baseline_mv[i]).abs() < 1e-6,
            "checkpoint mv differs from baseline at [{i}]: cp={} base={}",
            cp_mv[i],
            baseline_mv[i]
        );
    }
    for i in 0..D_IN {
        assert!(
            (cp_rms[i] - baseline_rms[i]).abs() < 1e-6,
            "checkpoint rms differs from baseline at [{i}]: cp={} base={}",
            cp_rms[i],
            baseline_rms[i]
        );
    }
}
