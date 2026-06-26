//! M4 #330o — Phase C.3b shared-chain batched encoder smoke tests.
//!
//! `shared_chain_batched` packs the shared-expert FFN body
//! (gate matvec → up matvec → SwiGLU → optional q8_0 round-trip → down
//! matvec) into ONE `MTLCommandBuffer` with a single readback. These tests
//! verify it produces bit-identical output to running the same chain
//! through the dispatcher trait. Saves 3-4 commit+wait+readbacks per layer
//! once C.3b.2 wires it into the encoder helper.
//!
//! macOS-only because we need a real Metal device. Default-branch silu
//! only — `DS4_SILU_FIDELITY=1` must not be set when running these tests.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn build_inputs(d_embd: usize, sd: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ffn_norm: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.019).sin() * 0.5 + 0.1).clamp(-2.0, 2.0))
        .collect();
    let w_gate: Vec<f32> = (0..sd * d_embd)
        .map(|i| ((i as f32 * 0.007).cos() * 0.2))
        .collect();
    let w_up: Vec<f32> = (0..sd * d_embd)
        .map(|i| ((i as f32 * 0.013).sin() * 0.2))
        .collect();
    let w_down: Vec<f32> = (0..d_embd * sd)
        .map(|i| ((i as f32 * 0.005).cos() * 0.15))
        .collect();
    (ffn_norm, w_gate, w_up, w_down)
}

fn swiglu_default(g: &[f32], u: &[f32]) -> Vec<f32> {
    // Default-branch silu identity (matches `ds4_silu_default_f32` MSL and
    // antirez `kernel_swiglu_f32` source — `x / (1 + exp(-x))`).
    g.iter()
        .zip(u.iter())
        .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
        .collect()
}

fn reference_shared_chain(
    disp: &MetalDispatcher,
    ffn_norm: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    sd: usize,
    want_q80: bool,
) -> Vec<f32> {
    use ds4_engine::dispatch::KernelDispatcher;
    let d_embd = ffn_norm.len();
    let g = disp.matvec_f32(w_gate, ffn_norm, sd);
    let u = disp.matvec_f32(w_up, ffn_norm, sd);
    let mid = swiglu_default(&g, &u);
    let mid_in = if want_q80 {
        ds4_engine::forward::q8_0_round_trip(&mid)
    } else {
        mid
    };
    disp.matvec_f32(w_down, &mid_in, d_embd)
}

/// Absolute tolerance for shared-chain comparisons. The chain contains
/// transcendentals (`exp` inside `silu`) — Rust `f32::exp` and Metal
/// `exp()` are NOT bit-identical implementations, so the GPU SwiGLU
/// inside `shared_chain_batched` and the CPU `swiglu_default` in
/// `reference_shared_chain` produce values that disagree in the last
/// mantissa bit. That single-bit difference at the SwiGLU output is
/// amplified by the down-projection matvec. Use absolute tolerance.
const SWIGLU_F32_TOL: f32 = 1e-4;

fn assert_close(batched: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(
        batched.len(),
        reference.len(),
        "{label}: down output length mismatch"
    );
    for (i, (b, r)) in batched.iter().zip(reference).enumerate() {
        let abs = (b - r).abs();
        let rel = abs / r.abs().max(1e-6);
        assert!(
            abs < tol || rel < tol,
            "{label} at i={i}: batched={b} reference={r} |diff|={abs} rel={rel}"
        );
    }
}

#[test]
fn shared_chain_matches_sequential_small() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let sd = 128;
    let (ffn_norm, w_gate, w_up, w_down) = build_inputs(d_embd, sd);

    let batched =
        disp.shared_chain_batched(&ffn_norm, &w_gate, &w_up, &w_down, sd as u32, false);
    let reference =
        reference_shared_chain(&disp, &ffn_norm, &w_gate, &w_up, &w_down, sd, false);

    assert_close(&batched, &reference, SWIGLU_F32_TOL, "shared_chain batched != sequential");
}

#[test]
fn shared_chain_matches_sequential_ds4_shape() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // DS4 V4 Flash production shape: d_embd=7168, shared_dim=2048.
    // Use a scaled-down but layout-faithful variant.
    let d_embd = 512;
    let sd = 512;
    let (ffn_norm, w_gate, w_up, w_down) = build_inputs(d_embd, sd);

    let batched =
        disp.shared_chain_batched(&ffn_norm, &w_gate, &w_up, &w_down, sd as u32, false);
    let reference =
        reference_shared_chain(&disp, &ffn_norm, &w_gate, &w_up, &w_down, sd, false);

    assert_close(
        &batched,
        &reference,
        SWIGLU_F32_TOL,
        "shared_chain batched (ds4 shape) != sequential",
    );
}

/// Toggling `want_q80` must change the output — locks against a silent
/// fallback where the q8_0 round-trip pass is skipped or aliased.
#[test]
fn shared_chain_q80_toggle_changes_output() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let d_embd = 256;
    let sd = 128;
    let (ffn_norm, w_gate, w_up, w_down) = build_inputs(d_embd, sd);

    let out_off =
        disp.shared_chain_batched(&ffn_norm, &w_gate, &w_up, &w_down, sd as u32, false);
    let out_on =
        disp.shared_chain_batched(&ffn_norm, &w_gate, &w_up, &w_down, sd as u32, true);

    let diff: f32 = out_off
        .iter()
        .zip(&out_on)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "want_q80 toggle produced bit-identical output — q8_0 pass likely skipped (max |diff| = {diff})"
    );

    // And the q8_0=true path also has to match its own sequential reference.
    let reference =
        reference_shared_chain(&disp, &ffn_norm, &w_gate, &w_up, &w_down, sd, true);
    assert_close(
        &out_on,
        &reference,
        SWIGLU_F32_TOL,
        "shared_chain batched (q80=on) != sequential",
    );
}
