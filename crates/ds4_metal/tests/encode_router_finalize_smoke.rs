//! Phase E M5.4.4-followup — composable `encode_router_finalize` smoke test.
//!
//! Reads (selected, weights) from the GPU buffers directly via
//! `contents()` after flushing the scope; asserts they match the
//! inherent `router_finalize`'s CPU readback for the same inputs.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::dispatch::KernelDispatcher;
use ds4_metal::MetalDispatcher;

fn build_inputs() -> (Vec<f32>, Vec<f32>) {
    let n = 256;
    let probs: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.0271).sin() * 0.4 + 0.5).max(0.01))
        .collect();
    let bias: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.011).cos() * 0.3))
        .collect();
    (probs, bias)
}

fn read_i32(buf: &metal::Buffer, n: usize) -> Vec<i32> {
    let mut out = vec![0i32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const i32,
            out.as_mut_ptr(),
            n,
        );
    }
    out
}

fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            out.as_mut_ptr(),
            n,
        );
    }
    out
}

#[test]
fn encode_router_finalize_matches_inherent() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let (probs, bias) = build_inputs();

    // Inherent path (returns CPU vecs).
    let (sel_inherent, w_inherent) = disp.router_finalize(&probs, &bias, 6);

    // Composable path.
    let scope = disp.batch_scope();
    let probs_b = scope.upload_f32(&probs);
    let bias_b = scope.upload_f32(&bias);
    let (sel_buf, w_buf) = scope
        .encode_router_finalize(&probs_b, &bias_b)
        .expect("encode_router_finalize");
    // flush_and_read_multi reads as f32; we want i32 for selected, so
    // we flush by reading a no-op buffer then access via contents().
    // Simpler: clone the buffers and read after the scope flushes.
    let sel_clone = sel_buf.buffer().clone();
    let w_clone = w_buf.buffer().clone();
    // Flush by calling flush_and_read on weights (this commits + waits).
    let _ = scope.flush_and_read(&w_buf);

    let sel_scope_i32 = read_i32(&sel_clone, 6);
    let w_scope = read_f32(&w_clone, 6);

    // Compare sorted SET (bitonic + CPU sort may give different order
    // for ties), but here probs are distinct so the order is stable.
    let mut sel_inh_sorted: Vec<i32> = sel_inherent.iter().map(|&i| i as i32).collect();
    let mut sel_scope_sorted = sel_scope_i32.clone();
    sel_inh_sorted.sort();
    sel_scope_sorted.sort();
    assert_eq!(
        sel_inh_sorted, sel_scope_sorted,
        "selected set differs: inherent={sel_inherent:?} scope={sel_scope_i32:?}"
    );

    // For each selected index, weights must be bit-close.
    use std::collections::HashMap;
    let inh_map: HashMap<i32, f32> = sel_inherent
        .iter()
        .zip(w_inherent.iter())
        .map(|(&i, &w)| (i as i32, w))
        .collect();
    for (i, &sel) in sel_scope_i32.iter().enumerate() {
        let inh_w = *inh_map.get(&sel).expect("missing expert in inherent map");
        let sc_w = w_scope[i];
        let abs = (inh_w - sc_w).abs();
        assert!(
            abs < 1e-5,
            "weight for expert {sel}: inherent={inh_w} scope={sc_w} |diff|={abs}"
        );
    }
}
