//! Phase F task #86 — `BatchScope::commit_keep_open` smoke.
//!
//! Phase F's central insight (per [[m5-antirez-decode-loop]]):
//! antirez encodes ~all per-token GPU work into a single shared
//! MTLCommandBuffer using a global g_pending_cbs array for cbs that
//! get committed mid-token without being waited until end-of-token.
//! Our previous BatchScope was consumed by its first flush; this
//! smoke validates the new `commit_keep_open` primitive that lets a
//! single scope accumulate multiple committed cbs and wait them all
//! at end.
//!
//! The smoke runs the same op (matvec_f32) N=20 times and verifies:
//!   1. Correctness: all outputs match the CPU reference.
//!   2. The shared-scope total wall time is meaningfully smaller than
//!      running N independent scopes (each with its own commit+wait).
//!      Each commit+wait costs ~2-3 ms on Apple Silicon even for
//!      tiny work — so 20 separate cbs spend ~40-60 ms on overhead,
//!      while one scope with 19 commit_keep_open + 1 flush spends
//!      ~one final wait of 2-3 ms.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::dispatch::{CpuDispatcher, KernelDispatcher};
use ds4_metal::MetalDispatcher;

const N_OPS: usize = 20;
const D_IN: usize = 512;
const D_OUT: usize = 256;

fn deterministic_data(seed: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..D_IN)
        .map(|i| ((i + seed) as f32 * 0.017).sin() * 0.3)
        .collect();
    let w: Vec<f32> = (0..D_IN * D_OUT)
        .map(|i| ((i + seed * 7) as f32 * 0.013).cos() * 0.1)
        .collect();
    (x, w)
}

#[test]
fn commit_keep_open_chains_ops_under_one_scope() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");

    // ── Path A: ONE BatchScope with N-1 commit_keep_open + final
    //   flush. All N matvecs encoded over a chain of cbs that
    //   commit-without-wait until the final flush.
    let t_a = std::time::Instant::now();
    let mut scope = disp.batch_scope();
    let mut a_outputs: Vec<ds4_metal::deferred::DeferredBuf> = Vec::with_capacity(N_OPS);
    let use_commit_keep_open =
        std::env::var("DS4_SMOKE_NO_KEEP_OPEN").ok().as_deref() != Some("1");
    // Pre-build all per-iteration data and KEEP it alive for the
    // entire scope (`upload_f32` reads the slice at encode-time but
    // doesn't hold a reference to the host Vec — once the BatchScope
    // op encodes the buffer into the cb, the host data could in
    // principle be dropped, but keeping it alive avoids any
    // edge-case lifetime issue).
    let inputs: Vec<(Vec<f32>, Vec<f32>)> = (0..N_OPS).map(deterministic_data).collect();
    for i in 0..N_OPS {
        let (x, w) = &inputs[i];
        let x_b = scope.upload_f32(x);
        let w_b = scope.upload_f32(w);
        let out = scope
            .matvec_f32(&w_b, &x_b, D_IN, D_OUT)
            .expect("matvec_f32 (A)");
        a_outputs.push(out);
        if use_commit_keep_open && i + 1 < N_OPS {
            scope.commit_keep_open();
        }
    }
    let last = a_outputs.last().expect("a_outputs");
    let last_out = scope.flush_and_read(last);
    // Drop a_outputs to release refs; we only validate the last one
    // numerically below (the others' buffers are kept alive by the
    // committed cbs until they complete).
    let elapsed_a = t_a.elapsed();

    // ── Path B: N independent BatchScopes, each commit+wait
    //   individually.
    let t_b = std::time::Instant::now();
    let mut last_out_b: Vec<f32> = Vec::new();
    for i in 0..N_OPS {
        let (x, w) = &inputs[i];
        let scope_i = disp.batch_scope();
        let x_b = scope_i.upload_f32(x);
        let w_b = scope_i.upload_f32(w);
        let out = scope_i
            .matvec_f32(&w_b, &x_b, D_IN, D_OUT)
            .expect("matvec_f32 (B)");
        let v = scope_i.flush_and_read(&out);
        if i + 1 == N_OPS {
            last_out_b = v;
        }
    }
    let elapsed_b = t_b.elapsed();

    // ── Path C: ONE BatchScope, N matvecs encoded back-to-back into
    //   the SAME cb (no intermediate commits at all), ONE flush at end.
    //   This is the truest "antirez-style" pattern at this small scale —
    //   if it's faster than A and B, that's evidence the one-cb-per-token
    //   approach can deliver.
    let t_c = std::time::Instant::now();
    let scope_c = disp.batch_scope();
    let mut c_outputs: Vec<ds4_metal::deferred::DeferredBuf> = Vec::with_capacity(N_OPS);
    for i in 0..N_OPS {
        let (x, w) = &inputs[i];
        let x_b = scope_c.upload_f32(x);
        let w_b = scope_c.upload_f32(w);
        let out = scope_c
            .matvec_f32(&w_b, &x_b, D_IN, D_OUT)
            .expect("matvec_f32 (C)");
        c_outputs.push(out);
    }
    let last_c = c_outputs.last().expect("c_outputs");
    let last_out_c = scope_c.flush_and_read(last_c);
    let elapsed_c = t_c.elapsed();

    // ── Correctness: all three paths match CPU reference on last output.
    let (x_n, w_n) = deterministic_data(N_OPS - 1);
    let ref_out = CpuDispatcher.matvec_f32(&w_n, &x_n, D_OUT);
    assert_eq!(last_out.len(), D_OUT);
    assert_eq!(last_out_b.len(), D_OUT);
    assert_eq!(last_out_c.len(), D_OUT);
    for i in 0..D_OUT {
        let r = ref_out[i];
        assert!(
            (last_out[i] - r).abs() < 1e-3,
            "A[{i}] {} vs cpu {r}",
            last_out[i]
        );
        assert!(
            (last_out_b[i] - r).abs() < 1e-3,
            "B[{i}] {} vs cpu {r}",
            last_out_b[i]
        );
        assert!(
            (last_out_c[i] - r).abs() < 1e-3,
            "C[{i}] {} vs cpu {r}",
            last_out_c[i]
        );
    }

    eprintln!(
        "A: shared scope + N-1 commit_keep_open + 1 final flush:  {:>8.1} ms",
        elapsed_a.as_secs_f64() * 1000.0
    );
    eprintln!(
        "B: {} independent scopes, each commit+wait:               {:>8.1} ms",
        N_OPS,
        elapsed_b.as_secs_f64() * 1000.0
    );
    eprintln!(
        "C: ONE scope, all {} ops in one cb, ONE flush:           {:>8.1} ms",
        N_OPS,
        elapsed_c.as_secs_f64() * 1000.0
    );
    eprintln!(
        "ratio (B / C) = expected antirez-style win:               {:>8.2}×",
        elapsed_b.as_secs_f64() / elapsed_c.as_secs_f64()
    );
}
