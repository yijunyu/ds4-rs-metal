//! Correctness smoke for the large-K dense q8_0 GEMM (`matmul_k_q8_0`, the
//! tiled `kernel_mul_mm_q8_0_f32` path) vs the proven K<=8 `matvec_k_q8_0`.
//!
//! The decisive check: at large K, row `r` of `matmul_k_q8_0([K, d_in])` must
//! equal `matvec_k_q8_0(x[r], K=1)` — i.e. the GEMM computes the same per-token
//! q8_0 matmul as the validated K=1 kernel. Also checks the matched-K (K=8)
//! case directly. Exercises both bc_out=false (d_out%64==0 && K%32==0) and
//! bc_out=true (padded) tiles. macOS-only.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

/// Build random Q8_0 weight bytes [d_out, d_in]: per 32-elem block = f16 scale
/// (2 B) + 32 int8.
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

/// Minimal f32 → IEEE half (round-to-nearest-even ignored; good enough for a
/// random-weight smoke).
fn half_from_f32(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 { return sign; }
    if exp >= 0x1f { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | mant as u16
}

fn random_x(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = seed;
    (0..n).map(|_| {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 9) & 0x7fff) as f32 / 32768.0 - 0.5
    }).collect()
}

fn run_case(disp: &MetalDispatcher, d_in: usize, d_out: usize, k: usize) {
    let w = random_q8_0(d_in, d_out, 0x1234 ^ (d_out as u32));
    let x = random_x(k * d_in, 0x9abc ^ (k as u32));

    let scope = disp.batch_scope();
    let w_db = scope.weight_q8_0_raw(&w, d_in * d_out);
    let x_db = scope.upload_f32(&x);
    let mm = scope.matmul_k_q8_0(&w_db, &x_db, d_in, d_out, k).expect("matmul_k_q8_0");
    let mm_out = scope.flush_and_read(&mm);
    assert_eq!(mm_out.len(), k * d_out, "matmul_k output shape");

    // Per-row: matmul row r must equal matvec_k_q8_0(x[r], K=1).
    let mut max_rel = 0.0f32;
    for &r in &[0usize, 1, k / 2, k - 1] {
        let scope = disp.batch_scope();
        let w1 = scope.weight_q8_0_raw(&w, d_in * d_out);
        let xr = scope.upload_f32(&x[r * d_in..(r + 1) * d_in]);
        let ref_r = scope.matvec_k_q8_0(&w1, &xr, d_in, d_out, 1).expect("matvec K=1");
        let ref_out = scope.flush_and_read(&ref_r);
        let row = &mm_out[r * d_out..(r + 1) * d_out];
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for j in 0..d_out {
            assert!(row[j].is_finite(), "nan at d_out={d_in}/{d_out} k={k} r={r} j={j}");
            num = num.max((row[j] - ref_out[j]).abs());
            den = den.max(ref_out[j].abs());
        }
        max_rel = max_rel.max(num / den.max(1e-4));
    }
    assert!(
        max_rel < 2e-2,
        "matmul_k_q8_0 vs matvec K=1 rel={max_rel:.3e} (d_in={d_in} d_out={d_out} K={k})"
    );
    eprintln!("matmul_k_q8_0 d_in={d_in} d_out={d_out} K={k}: per-row rel={max_rel:.2e} OK");
}

#[test]
fn matmul_k_q8_0_matches_k1_largeK() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // bc_out=false: d_out%64==0 && K%32==0
    run_case(&disp, 256, 128, 32);
    run_case(&disp, 512, 256, 64);
    run_case(&disp, 256, 128, 256);
    // bc_out=true: K not %32 / d_out not %64 → padded tiles
    run_case(&disp, 256, 128, 48);
    run_case(&disp, 256, 96, 64);
    run_case(&disp, 128, 64, 40);
}
