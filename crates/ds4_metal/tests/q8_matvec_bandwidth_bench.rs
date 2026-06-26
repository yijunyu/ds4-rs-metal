//! Phase 0 (decode gap): is our Q8_0 attn-projection matvec at the bandwidth
//! roofline? Our `ds4_kernel_mul_mv_q8_0_f32` is antirez's `kernel_mul_mv_q8_0_f32`
//! renamed (build.rs concat + host_name rewrite) — byte-identical source. So the
//! only thing that could differ from antirez is OUR dispatch (nsg/nxpsg/
//! threadgroup occupancy). This times `matvec_q8_0` at the three attention
//! projection shapes with resident synthetic Q8_0 weights and reports achieved
//! read bandwidth vs the M1 Ultra ~800 GB/s peak.
//!
//!   near 800 GB/s  → bandwidth-saturated; our dispatch is optimal and antirez
//!                    CANNOT be faster on the same kernel → the decode gap is
//!                    NOT here (diffuse / per-dispatch overhead).
//!   well below     → occupancy-bound; tuning nsg/nxpsg/tg could recover it.
//!
//! Opt-in: DS4_BENCH_Q8MV=1. No model load (synthetic weights) → safe to run
//! alongside the ds4-server. macOS-only.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use std::time::Instant;

/// Build random block_q8_0 bytes for a [d_out, d_in] weight (per 32-block:
/// f16 scale + 32 int8). scale = half(1.0).
fn rand_q8_0(d_in: usize, d_out: usize, seed: u32) -> Vec<u8> {
    let nb = d_in / 32;
    let row_bytes = nb * 34;
    let mut w = vec![0u8; d_out * row_bytes];
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };
    for r in 0..d_out {
        for b in 0..nb {
            let off = r * row_bytes + b * 34;
            w[off] = 0x00;
            w[off + 1] = 0x3C; // half 1.0
            for i in 0..32 {
                w[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
            }
        }
    }
    w
}

#[test]
fn q8_matvec_attn_bandwidth() {
    if std::env::var("DS4_BENCH_Q8MV").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_Q8MV unset — skipping. Set =1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };

    // (name, d_in, d_out) — the three production attn projections (Q8_0).
    // q_b is the dominant one (~71 MB weight read per token).
    let shapes: &[(&str, usize, usize)] = &[
        ("attn_q_a", 4096, 1024),
        ("attn_q_b", 1024, 65536),
        ("attn_kv ", 4096, 512),
    ];
    let n: usize = 64; // matvecs per timed batch (amortize commit/readback)

    eprintln!("\n  shape       d_in  d_out   wbytes(MB)  µs/matvec   GB/s   %800");
    for &(name, d_in, d_out) in shapes {
        let w_bytes = rand_q8_0(d_in, d_out, 0x9E3779B9 ^ (d_out as u32));
        let wb_mb = w_bytes.len() as f64 / 1e6;
        let x: Vec<f32> = (0..d_in).map(|i| ((i as f32) * 0.013).sin()).collect();

        let run_once = || {
            let scope = disp.batch_scope();
            let w_db = scope.weight_q8_0_raw(&w_bytes, d_in * d_out);
            let x_db = scope.upload_f32(&x);
            let mut outs = Vec::with_capacity(n);
            for _ in 0..n {
                outs.push(scope.matvec_q8_0(&w_db, &x_db, d_in, d_out).expect("matvec"));
            }
            let refs: Vec<&_> = outs.iter().collect();
            let _ = scope.flush_and_read_multi(&refs);
        };
        run_once(); // warmup (pipeline build + first-touch)
        run_once();
        let t0 = Instant::now();
        let reps = 3;
        for _ in 0..reps {
            run_once();
        }
        let secs = t0.elapsed().as_secs_f64() / reps as f64;
        let per_matvec_us = secs / n as f64 * 1e6;
        let gbps = (w_bytes.len() as f64 * n as f64) / secs / 1e9;
        eprintln!(
            "  {:<9} {:>5} {:>6}   {:>8.1}   {:>8.1}   {:>5.0}   {:>3.0}%",
            name,
            d_in,
            d_out,
            wb_mb,
            per_matvec_us,
            gbps,
            gbps / 800.0 * 100.0
        );
    }
    // ── Dispatch-param sweep on the dominant attn_q_b shape. The kernel source
    //    is identical to antirez; only nsg/nxpsg differ (ours 8/16, antirez 4/—).
    //    Find the params that maximize bandwidth.
    let (d_in, d_out) = (1024usize, 65536usize);
    let w_bytes = rand_q8_0(d_in, d_out, 0xBADC0DE);
    let x: Vec<f32> = (0..d_in).map(|i| ((i as f32) * 0.013).sin()).collect();
    eprintln!("\n  === attn_q_b (1024→65536, {:.1} MB) nsg/nxpsg sweep ===", w_bytes.len() as f64 / 1e6);
    eprintln!("  nsg  nxpsg   µs/matvec   GB/s   %800   note");
    let _ = (d_in, d_out, &w_bytes, &x); // (q_b dims; per-shape sweep below)
    // Sweep nsg ∈ {1,2,3,4,8} (nxpsg=16 fixed — it barely moved) across ALL
    // THREE attn shapes to derive the right nsg heuristic.
    for &(name, d_in, d_out) in shapes {
        let w_bytes = rand_q8_0(d_in, d_out, 0xBADC0DE ^ (d_out as u32));
        let x: Vec<f32> = (0..d_in).map(|i| ((i as f32) * 0.013).sin()).collect();
        eprintln!("  -- {} (d_in={} d_out={}, {:.1} MB) --", name.trim(), d_in, d_out, w_bytes.len() as f64 / 1e6);
        std::env::set_var("DS4_Q8_NXPSG", "16");
        for nsg in [1i16, 2, 3, 4, 8] {
            std::env::set_var("DS4_Q8_NSG", nsg.to_string());
            let run_once = || {
                let scope = disp.batch_scope();
                let w_db = scope.weight_q8_0_raw(&w_bytes, d_in * d_out);
                let x_db = scope.upload_f32(&x);
                let mut outs = Vec::with_capacity(n);
                for _ in 0..n {
                    outs.push(scope.matvec_q8_0(&w_db, &x_db, d_in, d_out).expect("matvec"));
                }
                let refs: Vec<&_> = outs.iter().collect();
                let _ = scope.flush_and_read_multi(&refs);
            };
            run_once();
            run_once();
            let t0 = Instant::now();
            let reps = 3;
            for _ in 0..reps {
                run_once();
            }
            let secs = t0.elapsed().as_secs_f64() / reps as f64;
            let per_us = secs / n as f64 * 1e6;
            let gbps = (w_bytes.len() as f64 * n as f64) / secs / 1e9;
            let tag = if nsg == 8 { "(ours)" } else if nsg == 4 { "(antirez)" } else { "" };
            eprintln!(
                "    nsg={:>2}   {:>8.1} µs   {:>5.0} GB/s   {:>3.0}%  {}",
                nsg, per_us, gbps, gbps / 800.0 * 100.0, tag
            );
        }
    }
    std::env::remove_var("DS4_Q8_NSG");
    std::env::remove_var("DS4_Q8_NXPSG");
    eprintln!(
        "\n  Interpretation: if a combo beats OURS by a wide margin, our q8 dispatch\n  is suboptimal → a free win (just change nsg/nxpsg). If all ~equal & low,\n  the kernel itself is occupancy-bound at this shape (deeper kernel work)."
    );
}
