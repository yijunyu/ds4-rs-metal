//! GUARDRAIL TEST — `matmul_k_q8_0` (the large-K dense q8_0 GEMM used by the
//! chunk-prefill K-batched projections). This kernel's K=1-vs-K-batch behaviour
//! is the ROOT-CAUSE class of the chunk "@3000 generic" bisection: a K-batched
//! projection that deviates from the K=1 path feeds a broad residual drift that
//! discrete MoE routing then amplifies. These guardrails catch it in seconds on
//! synthetic data (no 86 GB model), at the kernel — not 40 layers downstream.
//!
//! Wraps the suspect kernel in the three guardrails:
//!   • K-batch≡K=1  (Close — it's a reduction, fp32 non-associative)
//!   • determinism  (bit-identical across runs)
//!   • bandwidth    (microbench vs roofline)
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_metal::MetalDispatcher;

/// Random Q8_0 weight bytes [d_out, d_in]: per 32-elem block = f16 scale + 32 i8.
fn random_q8_0(d_in: usize, d_out: usize, seed: u32) -> Vec<u8> {
    let nb = d_in / 32;
    let row_bytes = nb * 34;
    let mut w = vec![0u8; d_out * row_bytes];
    let mut rng = seed;
    let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
    for r in 0..d_out {
        for b in 0..nb {
            let off = r * row_bytes + b * 34;
            let scale = 0.01f32 + (next() & 0xff) as f32 / 4096.0;
            // minimal f32→f16 for the block scale
            let bits = scale.to_bits();
            let h = (((bits >> 16) & 0x8000)
                | ((((bits >> 23) & 0xff).wrapping_sub(112) & 0x1f) << 10)
                | ((bits >> 13) & 0x3ff)) as u16;
            w[off] = (h & 0xff) as u8;
            w[off + 1] = (h >> 8) as u8;
            for i in 0..32 {
                w[off + 2 + i] = ((next() & 0xff) as i32 - 128) as i8 as u8;
            }
        }
    }
    w
}

#[test]
fn matmul_k_q8_0_guardrails() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let (d_in, d_out, k) = (512usize, 256usize, 256usize);
    let w = random_q8_0(d_in, d_out, 0x1234);
    let x = rand_f32(k * d_in, 0x9abc);

    // (2) K-batch ≡ K=1-looped. Reduction kernel ⇒ Close, not Exact.
    assert_k_batch_equiv(
        "matmul_k_q8_0",
        k,
        d_out,
        Equiv::Close { rel_tol: 2e-2 },
        || {
            let s = disp.batch_scope();
            let w_db = s.weight_q8_0_raw(&w, d_in * d_out);
            let x_db = s.upload_f32(&x);
            let mm = s.matmul_k_q8_0(&w_db, &x_db, d_in, d_out, k).expect("matmul_k");
            s.flush_and_read(&mm)
        },
        |r| {
            let s = disp.batch_scope();
            let w1 = s.weight_q8_0_raw(&w, d_in * d_out);
            let xr = s.upload_f32(&x[r * d_in..(r + 1) * d_in]);
            let mv = s.matvec_k_q8_0(&w1, &xr, d_in, d_out, 1).expect("matvec K=1");
            s.flush_and_read(&mv)
        },
    );

    // (1) Determinism — same inputs ⇒ byte-identical output every run.
    assert_deterministic("matmul_k_q8_0", 3, || {
        let s = disp.batch_scope();
        let w_db = s.weight_q8_0_raw(&w, d_in * d_out);
        let x_db = s.upload_f32(&x);
        let mm = s.matmul_k_q8_0(&w_db, &x_db, d_in, d_out, k).expect("matmul_k");
        s.flush_and_read(&mm)
    });

    // (5) Bandwidth microbench — DIAGNOSTIC (reported, not asserted).
    // MICROBENCH PITFALL the framework teaches: this closure re-creates the
    // scope + re-uploads the 17 MB of q8 weights EVERY iteration, so the median
    // is dominated by dispatch+upload overhead (+ any co-resident GPU load), not
    // the kernel — it reads ~2 GB/s, far below roofline. An ABSOLUTE-roofline
    // `assert_bandwidth_frac` is only valid when setup is hoisted out of the
    // timed loop (upload weights once, dispatch N times) AND the GPU is idle;
    // otherwise use it for RELATIVE regression vs a recorded baseline. Reported
    // here so a regression is visible without a flaky hard gate.
    let bytes = d_out * (d_in / 32) * 34 + k * d_in * 4 + k * d_out * 4;
    let _gbps = bench_bandwidth("matmul_k_q8_0", bytes, 20, || {
        let s = disp.batch_scope();
        let w_db = s.weight_q8_0_raw(&w, d_in * d_out);
        let x_db = s.upload_f32(&x);
        let mm = s.matmul_k_q8_0(&w_db, &x_db, d_in, d_out, k).expect("matmul_k");
        s.flush_and_read(&mm)
    });
}
