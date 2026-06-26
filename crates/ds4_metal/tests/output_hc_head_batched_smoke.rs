//! M4 #330o — Phase C.3c output-head batched encoder smoke tests.
//!
//! `output_hc_head_batched` packs `rms_norm(unit gamma) → matvec_f32(fn_w)`
//! for the once-per-token post-layer-loop HC fold (antirez
//! `output_hc_head_one`, ds4.c:7654-7681) into ONE `MTLCommandBuffer`.
//! These tests verify it produces bit-identical `pre[n_hc]` to running
//! the two ops sequentially through the same dispatcher. Saves one
//! commit+wait+readback per token once C.3c.2 wires it into the encoder.
//!
//! macOS-only because we need a real Metal device.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(n_hc: usize, d_embd: usize) -> (Vec<f32>, Vec<f32>) {
    let hc_dim = n_hc * d_embd;
    let inp_hc: Vec<f32> = (0..hc_dim)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5 + 0.07).clamp(-2.0, 2.0))
        .collect();
    let fn_w: Vec<f32> = (0..hc_dim * n_hc)
        .map(|i| ((i as f32 * 0.0071).cos() * 0.18))
        .collect();
    (inp_hc, fn_w)
}

fn reference_pre(
    disp: &MetalDispatcher,
    inp_hc: &[f32],
    fn_w: &[f32],
    n_hc: usize,
    d_embd: usize,
    eps_rms: f32,
) -> Vec<f32> {
    use ds4_engine::dispatch::KernelDispatcher;
    let hc_dim = n_hc * d_embd;
    let unit_gamma = vec![1.0f32; hc_dim];
    let flat = disp.rms_norm(inp_hc, &unit_gamma, eps_rms);
    disp.matvec_f32(fn_w, &flat, n_hc)
}

#[test]
fn output_hc_head_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 8;
    let d_embd = 64;
    let eps = 1e-5_f32;
    let (inp_hc, fn_w) = build_inputs(n_hc, d_embd);

    let batched = disp.output_hc_head_batched(&inp_hc, &fn_w, n_hc, d_embd, eps);
    let reference = reference_pre(&disp, &inp_hc, &fn_w, n_hc, d_embd, eps);

    assert_eq!(batched.len(), reference.len(), "pre length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "output_hc_head batched != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

#[test]
fn output_hc_head_matches_sequential_ds4_shape() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // DS4 V4 Flash production shape: n_hc=8, d_embd=7168 (hc_dim=57344).
    // Use a scaled-down but layout-faithful variant: n_hc=8, d_embd=512
    // (hc_dim=4096) so the test stays fast while exercising the same
    // matvec-output-row count and rms_norm threadgroup-mul-tile path.
    let n_hc = 8;
    let d_embd = 512;
    let eps = 1e-5_f32;
    let (inp_hc, fn_w) = build_inputs(n_hc, d_embd);

    let batched = disp.output_hc_head_batched(&inp_hc, &fn_w, n_hc, d_embd, eps);
    let reference = reference_pre(&disp, &inp_hc, &fn_w, n_hc, d_embd, eps);

    assert_eq!(batched.len(), reference.len(), "pre length mismatch");
    for (i, (b, r)) in batched.iter().zip(&reference).enumerate() {
        assert_eq!(
            b.to_bits(),
            r.to_bits(),
            "output_hc_head batched (ds4 shape) != sequential at i={i}: batched={b} reference={r}"
        );
    }
}

/// Different `fn_w` weights produce different `pre` — locks the test
/// against silent fallback (a future regression that drops `fn_w` would
/// still pass against an all-zero output were the rms-only path
/// returned).
#[test]
fn output_hc_head_fn_w_affects_output() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 8;
    let d_embd = 64;
    let eps = 1e-5_f32;
    let (inp_hc, fn_w_a) = build_inputs(n_hc, d_embd);
    let fn_w_b: Vec<f32> = (0..n_hc * d_embd * n_hc)
        .map(|i| ((i as f32 * 0.0091).sin() * 0.31))
        .collect();

    let out_a = disp.output_hc_head_batched(&inp_hc, &fn_w_a, n_hc, d_embd, eps);
    let out_b = disp.output_hc_head_batched(&inp_hc, &fn_w_b, n_hc, d_embd, eps);

    let diff: f32 = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "different fn_w produced bit-identical output — matvec pass likely skipped (max |diff| = {diff})"
    );
}
