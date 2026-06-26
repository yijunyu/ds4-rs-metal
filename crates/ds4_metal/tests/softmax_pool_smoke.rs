//! M5 Phase D — `kernel_dsv4_softmax_pool` MSL port smoke.
//!
//! Compares our inline-MSL port (verbatim from antirez
//! `dsv4_misc.metal:1012-1043`) against a CPU oracle of the same
//! algorithm. The kernel uses `exp` + reductions, so GPU vs CPU
//! agreement is within float-arithmetic tolerance (~1e-5 typical
//! for max-stabilized softmax + weighted sum over modest n_rows).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

/// CPU oracle for `kernel_dsv4_softmax_pool` over a row-major
/// `[n_rows × width]` layout: per output column `id`, computes
/// max_s = max over rows of score[ir,id]; then weights w_ir = exp
/// (score[ir,id] - max_s); returns Σ w*kv / Σ w.
fn cpu_oracle(kv: &[f32], score: &[f32], n_rows: usize, width: usize) -> Vec<f32> {
    assert_eq!(kv.len(), n_rows * width);
    assert_eq!(score.len(), n_rows * width);
    let mut out = vec![0.0f32; width];
    for id in 0..width {
        let mut max_s = f32::NEG_INFINITY;
        for ir in 0..n_rows {
            let s = score[ir * width + id];
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum = 0.0f32;
        let mut acc = 0.0f32;
        for ir in 0..n_rows {
            let s = score[ir * width + id];
            let v = kv[ir * width + id];
            let w = (s - max_s).exp();
            sum += w;
            acc += v * w;
        }
        out[id] = acc / sum;
    }
    out
}

fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
    }
    out
}

fn run_case(disp: &MetalDispatcher, n_rows: u32, width: u32, seed: u32) {
    let n = (n_rows as usize) * (width as usize);
    let kv: Vec<f32> = (0..n)
        .map(|i| (((i + seed as usize) as f32) * 0.013).sin() * 0.4)
        .collect();
    let score: Vec<f32> = (0..n)
        .map(|i| (((i + seed as usize) as f32) * 0.017).cos() * 0.3)
        .collect();

    let ref_out = cpu_oracle(&kv, &score, n_rows as usize, width as usize);

    let mut scope = disp.batch_scope();
    let kv_db = scope.upload_f32(&kv);
    let score_db = scope.upload_f32(&score);
    let out_db = scope
        .softmax_pool(kv_db.buffer(), score_db.buffer(), n_rows, width)
        .expect("softmax_pool");
    // Read by flushing — the pool produces a fresh DeferredBuf.
    let gpu_out = scope.flush_and_read(&out_db);
    // Touch the read closure so the helper stays referenced for future
    // tests below that read state directly.
    let _ = read_f32;

    assert_eq!(gpu_out.len(), width as usize);
    let mut max_abs: f32 = 0.0;
    let mut max_ref: f32 = 0.0;
    for i in 0..width as usize {
        max_abs = max_abs.max((gpu_out[i] - ref_out[i]).abs());
        max_ref = max_ref.max(ref_out[i].abs());
    }
    let rel = max_abs / max_ref.max(1e-6);
    assert!(
        max_abs < 5e-5 && rel < 5e-5,
        "softmax_pool drift exceeds tolerance: n_rows={n_rows} width={width} seed={seed} \
         max_abs={max_abs:.3e} max_ref={max_ref:.3e} rel={rel:.3e}",
    );
}

#[test]
fn softmax_pool_matches_cpu_oracle_ratio4_width128() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // V4 Flash compressor: ratio=4, head_dim=128. The pool typically
    // runs over the n_rows = ratio window on emit tokens.
    for seed in 0..4 {
        run_case(&disp, 4, 128, seed);
    }
}

#[test]
fn softmax_pool_matches_cpu_oracle_emit_8_rows() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // ratio==4 path may pool over the full 2*ratio = 8 rows of state
    // depending on how the host orchestrates emit.
    run_case(&disp, 8, 128, 11);
}

#[test]
fn softmax_pool_matches_cpu_oracle_small_width() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Width below threadgroup width (32) exercises the bounds check
    // in the kernel.
    run_case(&disp, 4, 17, 99);
}

#[test]
fn softmax_pool_matches_cpu_oracle_single_row() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Degenerate n_rows=1: softmax over one element gives w=1 trivially;
    // output = kv directly. Useful invariant check.
    run_case(&disp, 1, 64, 7);
}
