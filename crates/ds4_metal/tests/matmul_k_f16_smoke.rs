//! Correctness smoke for the K-batched dense f16/f32 mul_mv wrappers
//! (`matmul_f16_k` / `matmul_f32_k`) vs the per-row `matvec_f16` / `matvec_f32`.
//!
//! These reuse the SAME ne11-aware kernel (`kernel_mul_mv_{f16,f32}_f32_4`) as the
//! single-row matvec — batched just sets grid-Y = K and ne11 = K, so each output
//! column is computed by exactly the same code path as the K=1 matvec. Row r of
//! the batched output must therefore be BYTE-IDENTICAL to matvec(x[r]). Used to
//! batch the chunk-prefill compressor projections. macOS-only.
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

fn rand_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = seed;
    (0..n).map(|_| {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 9) & 0x7fff) as f32 / 32768.0 - 0.5
    }).collect()
}

fn run_f16(disp: &MetalDispatcher, d_in: usize, d_out: usize, k: usize) {
    let w_f32 = rand_f32(d_in * d_out, 0x1111 ^ d_out as u32);
    let w_bytes: Vec<u8> = w_f32.iter().flat_map(|&v| half_from_f32(v).to_le_bytes()).collect();
    let x = rand_f32(k * d_in, 0x2222 ^ k as u32);

    let scope = disp.batch_scope();
    let w_db = scope.weight_f16(&w_bytes);
    let x_db = scope.upload_f32(&x);
    let mm = scope.matmul_f16_k(&w_db, &x_db, d_in, d_out, k).expect("matmul_f16_k");
    let mm_out = scope.flush_and_read(&mm);
    assert_eq!(mm_out.len(), k * d_out);

    for &r in &[0usize, 1, k / 2, k - 1] {
        let scope = disp.batch_scope();
        let w1 = scope.weight_f16(&w_bytes);
        let xr = scope.upload_f32(&x[r * d_in..(r + 1) * d_in]);
        let ref_r = scope.matvec_f16(&w1, &xr, d_in, d_out).expect("matvec_f16");
        let ref_out = scope.flush_and_read(&ref_r);
        let row = &mm_out[r * d_out..(r + 1) * d_out];
        for j in 0..d_out {
            assert!(row[j].is_finite(), "nan f16 d_in={d_in} d_out={d_out} k={k} r={r} j={j}");
            assert_eq!(row[j].to_bits(), ref_out[j].to_bits(),
                "matmul_f16_k row {r} col {j} != matvec_f16 (d_in={d_in} d_out={d_out} k={k})");
        }
    }
    eprintln!("matmul_f16_k d_in={d_in} d_out={d_out} K={k}: byte-identical to matvec_f16 OK");
}

fn run_f32(disp: &MetalDispatcher, d_in: usize, d_out: usize, k: usize) {
    let w = rand_f32(d_in * d_out, 0x3333 ^ d_out as u32);
    let x = rand_f32(k * d_in, 0x4444 ^ k as u32);

    let scope = disp.batch_scope();
    let w_db = scope.weight_f32(&w);
    let x_db = scope.upload_f32(&x);
    let mm = scope.matmul_f32_k(&w_db, &x_db, d_in, d_out, k).expect("matmul_f32_k");
    let mm_out = scope.flush_and_read(&mm);
    assert_eq!(mm_out.len(), k * d_out);

    for &r in &[0usize, 1, k / 2, k - 1] {
        let scope = disp.batch_scope();
        let w1 = scope.weight_f32(&w);
        let xr = scope.upload_f32(&x[r * d_in..(r + 1) * d_in]);
        let ref_r = scope.matvec_f32(&w1, &xr, d_in, d_out).expect("matvec_f32");
        let ref_out = scope.flush_and_read(&ref_r);
        let row = &mm_out[r * d_out..(r + 1) * d_out];
        for j in 0..d_out {
            assert!(row[j].is_finite(), "nan f32 r={r} j={j}");
            assert_eq!(row[j].to_bits(), ref_out[j].to_bits(),
                "matmul_f32_k row {r} col {j} != matvec_f32 (d_in={d_in} d_out={d_out} k={k})");
        }
    }
    eprintln!("matmul_f32_k d_in={d_in} d_out={d_out} K={k}: byte-identical to matvec_f32 OK");
}

#[test]
fn matmul_f16_k_matches_matvec() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // compressor-ish shapes: d_in = n_embd-like (%256), d_out = coff*head_dim.
    run_f16(&disp, 512, 256, 8);
    run_f16(&disp, 512, 256, 64);
    run_f16(&disp, 1024, 128, 128);
    run_f16(&disp, 256, 64, 512);
}

#[test]
fn matmul_f32_k_matches_matvec() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    run_f32(&disp, 512, 256, 8);
    run_f32(&disp, 512, 256, 64);
    run_f32(&disp, 1024, 128, 128);
    run_f32(&disp, 256, 64, 512);
}
