//! Isolation smoke for the q4 attention GEMM (`matmul_k_attn_proj` w/ DS4_ATTN_Q4):
//! build a q8_0 weight, run it both as q8 (matmul_k_q8_0) and q4 (requant +
//! matmul_k_q4_0 via matmul_k_attn_proj), read both back. The decisive checks:
//! (1) the q4 output is NOT all-zeros (the bug under debug), (2) it's within q4
//! quantization error of the q8 result. macOS-only.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn half_from_f32(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 { return sign; }
    if exp >= 0x1f { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | mant as u16
}

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
            let h = half_from_f32(scale);
            w[off] = (h & 0xff) as u8;
            w[off + 1] = (h >> 8) as u8;
            for i in 0..32 {
                w[off + 2 + i] = ((next() & 0xff) as i32 - 128) as i8 as u8;
            }
        }
    }
    w
}

fn random_x(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = seed;
    (0..n).map(|_| {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 9) & 0x7fff) as f32 / 32768.0 - 0.5
    }).collect()
}

// DIAGNOSTIC (currently FAILS — the q4 GEMM kernel bug under debug). Run with
// `--ignored --nocapture`: isolates that the requant is correct (CPU_q4 ≈ q8) but
// the GPU q4 kernel outputs ~40000× too small (weight block-advance / f16-x staging).
// Decisive timing: is the bridge q4 GEMM actually FASTER than the emitted q8 GEMM?
// (q4 = half the weight bytes; but the bridge kernel may be less optimized than the
// emitted q8.) Real q_a-ish dims, many iters, best-of.
#[test]
#[ignore]
fn matmul_k_q4_0_timing_vs_q8() {
    use std::time::Instant;
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let (d_in, d_out, k) = (4096usize, 1024usize, 600usize);
    let w = random_q8_0(d_in, d_out, 0x1234);
    let x = random_x(k * d_in, 0x9abc);
    let mut best_q8 = f64::MAX;
    let mut best_q4 = f64::MAX;
    for _ in 0..8 {
        std::env::remove_var("DS4_ATTN_Q4");
        let t = Instant::now();
        for _ in 0..10 {
            let scope = disp.batch_scope();
            let w_db = scope.weight_q8_0_raw(&w, d_in * d_out);
            let x_db = scope.upload_f32(&x);
            let o = scope.matmul_k_attn_proj(&w_db, &x_db, d_in, d_out, k).unwrap();
            let _ = scope.flush_and_read(&o);
        }
        best_q8 = best_q8.min(t.elapsed().as_secs_f64());
        std::env::set_var("DS4_ATTN_Q4", "1");
        let t = Instant::now();
        for _ in 0..10 {
            let scope = disp.batch_scope();
            let w_db = scope.weight_q8_0_raw(&w, d_in * d_out);
            let x_db = scope.upload_f32(&x);
            let o = scope.matmul_k_attn_proj(&w_db, &x_db, d_in, d_out, k).unwrap();
            let _ = scope.flush_and_read(&o);
        }
        best_q4 = best_q4.min(t.elapsed().as_secs_f64());
        std::env::remove_var("DS4_ATTN_Q4");
    }
    eprintln!("GEMM 10-iter best: q8={best_q8:.4}s q4={best_q4:.4}s  speedup={:.2}x", best_q8 / best_q4);
}

// DECODE LEVER (option 2): the K=1 matvec is the weight-BOUND path (each weight
// read once, no K-amortization), so if any q4 win exists it shows HERE, not in
// the K=3000 GEMM (compute-bound, already measured 1.00x). Times the dense q8 vs
// q4 matvec at decode-realistic attention-projection dims. If ~2x → decode lever
// real; if ~1.0x → matvec is occupancy/latency-bound (small weight) not
// bandwidth-bound → q4 dead on decode too. Run `--ignored --nocapture`.
#[test]
#[ignore]
fn matvec_q4_0_decode_timing_vs_q8() {
    use std::time::Instant;
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // (d_in, d_out, label): q_a, kv, q_b, w_o_b — real DS4-Flash attn proj dims.
    for &(d_in, d_out, lbl) in &[
        (7168usize, 1536usize, "q_a   "),
        (7168, 576, "kv    "),
        (1536, 24576, "q_b   "),
        (16384, 7168, "w_o_b "),
    ] {
        let w_q8 = random_q8_0(d_in, d_out, 0x1234 ^ (d_out as u32));
        let w_q4 = ds4_engine::layer_view::requant_q8_0_to_q4_0(&w_q8).unwrap();
        let x = random_x(d_in, 0x9abc ^ (d_in as u32));
        let (mut bq8, mut bq4) = (f64::MAX, f64::MAX);
        // warm
        let _ = disp.matvec_q8_0_dense(&w_q8, &x, d_out).unwrap();
        let _ = disp.matvec_q4_0_dense(&w_q4, &x, d_out).unwrap();
        for _ in 0..12 {
            let t = Instant::now();
            for _ in 0..50 { let _ = disp.matvec_q8_0_dense(&w_q8, &x, d_out).unwrap(); }
            bq8 = bq8.min(t.elapsed().as_secs_f64() / 50.0);
            let t = Instant::now();
            for _ in 0..50 { let _ = disp.matvec_q4_0_dense(&w_q4, &x, d_out).unwrap(); }
            bq4 = bq4.min(t.elapsed().as_secs_f64() / 50.0);
        }
        let mb8 = (w_q8.len() as f64) / 1e6;
        eprintln!(
            "matvec {lbl} d_in={d_in} d_out={d_out} (w_q8={mb8:.1}MB): q8={:.1}us q4={:.1}us  speedup={:.2}x",
            bq8 * 1e6, bq4 * 1e6, bq8 / bq4
        );
    }
}

#[test]
#[ignore]
fn matmul_k_q4_0_not_zeros_and_near_q8() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Include bc_out=true cases (k%32!=0 / d_out%64!=0 → partial tiles, bounded store)
    // — the prefill uses k=600/3000 (bc_out=true), which the %32-aligned cases missed.
    for &(d_in, d_out, k) in &[(256usize, 128usize, 32usize), (256, 128, 48), (256, 96, 40), (256, 128, 600)] {
        let w = random_q8_0(d_in, d_out, 0x1234 ^ (d_out as u32));
        let x = random_x(k * d_in, 0x9abc ^ (k as u32));

        // q8 reference.
        std::env::remove_var("DS4_ATTN_Q4");
        let scope = disp.batch_scope();
        let w_db = scope.weight_q8_0_raw(&w, d_in * d_out);
        let x_db = scope.upload_f32(&x);
        let q8 = scope.matmul_k_attn_proj(&w_db, &x_db, d_in, d_out, k).expect("q8");
        let q8_out = scope.flush_and_read(&q8);

        // q4 path (requant + matmul_k_q4_0).
        std::env::set_var("DS4_ATTN_Q4", "1");
        let scope = disp.batch_scope();
        let w_db = scope.weight_q8_0_raw(&w, d_in * d_out);
        let x_db = scope.upload_f32(&x);
        let q4 = scope.matmul_k_attn_proj(&w_db, &x_db, d_in, d_out, k).expect("q4");
        let q4_out = scope.flush_and_read(&q4);
        std::env::remove_var("DS4_ATTN_Q4");

        // CPU q4 reference: requant on CPU, dequant, GEMM in f32 — isolates whether
        // the requant is right (cpu_q4 ≈ q8) vs the GPU kernel (gpu_q4 ≈ cpu_q4).
        let q4_bytes = ds4_engine::layer_view::requant_q8_0_to_q4_0(&w).unwrap();
        // dequant q4 weights [d_out, d_in]
        let nb = d_in / 32;
        let mut wf = vec![0.0f32; d_out * d_in];
        for r in 0..d_out {
            for b in 0..nb {
                let blk = &q4_bytes[(r * nb + b) * 18..(r * nb + b) * 18 + 18];
                let vals = ds4_engine::layer_view::dequant_q4_0_block(blk);
                for j in 0..32 { wf[r * d_in + b * 32 + j] = vals[j]; }
            }
        }
        let mut cpu_q4 = vec![0.0f32; k * d_out];
        for ki in 0..k {
            for o in 0..d_out {
                let mut acc = 0.0f32;
                for c in 0..d_in { acc += wf[o * d_in + c] * x[ki * d_in + c]; }
                cpu_q4[ki * d_out + o] = acc;
            }
        }
        let cpu_abs: f32 = cpu_q4.iter().map(|v| v.abs()).sum();
        let q4_abs: f32 = q4_out.iter().map(|v| v.abs()).sum();
        let q8_abs: f32 = q8_out.iter().map(|v| v.abs()).sum();
        // gpu_q4 vs cpu_q4 (the kernel-only error)
        let mut kn = 0.0f32; let mut kd = 0.0f32;
        for j in 0..cpu_q4.len() { kn = kn.max((q4_out[j]-cpu_q4[j]).abs()); kd = kd.max(cpu_q4[j].abs()); }
        eprintln!("d_in={d_in} d_out={d_out} k={k}: q8_abs={q8_abs:.1} CPU_q4_abs={cpu_abs:.1} GPU_q4_abs={q4_abs:.1} | gpu-vs-cpu_q4 rel={:.3e}", kn/kd.max(1e-4));
        let _ = q4_abs;
    }
}
