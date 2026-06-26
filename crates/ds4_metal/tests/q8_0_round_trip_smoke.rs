//! M4 #330f — macOS-only Metal q8_0_round_trip smoke test.
//!
//! Validates `MetalDispatcher::q8_0_round_trip` against the CPU oracle in
//! `ds4_engine::forward::q8_0_round_trip`. The kernel MUST be byte-exact
//! because antirez uses q8_0 round-trip as the int8-dot pre-image at six
//! gated callsites (M4 #299/#302/#303/#304/#306). A 1-ULP drift here
//! breaks every downstream gate.
//!
//! Coverage:
//!   - exact 32-elt block (fits in one threadgroup)
//!   - exact multi-block (32, 64, 320 — exercises block independence)
//!   - partial tail (40 elts: 1 full block + 8-elt tail)
//!   - zero-block (all zeros → output all zeros, amax-zero short-circuit)
//!   - mixed-sign with banker's-rounding boundaries

#![cfg(target_os = "macos")]

use ds4_engine::forward::q8_0_round_trip;
use ds4_metal::MetalDispatcher;

fn assert_byte_exact(metal: &[f32], cpu: &[f32], label: &str) {
    assert_eq!(metal.len(), cpu.len(), "{label}: length mismatch");
    for (i, (&m, &c)) in metal.iter().zip(cpu).enumerate() {
        assert!(
            m.to_bits() == c.to_bits(),
            "{label}: byte-exact failed at i={i}: metal={m} ({:#x}) cpu={c} ({:#x})",
            m.to_bits(),
            c.to_bits()
        );
    }
}

#[test]
fn q8_0_round_trip_single_block_byte_exact() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Adversarial: mixed magnitudes + half-quantum boundaries.
    let mut x = vec![0.0f32; 32];
    x[0] = 127.0; // anchors amax → d = 1.0
    x[1] = 0.5; // banker's-round → 0
    x[2] = 1.5; // banker's-round → 2
    x[3] = 2.5; // banker's-round → 2 (even)
    x[4] = 3.5; // banker's-round → 4
    x[5] = -2.5; // banker's-round → -2
    x[7] = 12.0;
    x[15] = -45.0;
    x[31] = 0.001;
    let m = d.q8_0_round_trip(&x);
    let c = q8_0_round_trip(&x);
    assert_byte_exact(&m, &c, "single_block_byte_exact");
}

#[test]
fn q8_0_round_trip_multi_block_independent_scale() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    // 320 elts = 10 full blocks. Block-i amax scales with i so each block
    // gets its own (very different) d, exercising per-block-amax fan-out.
    let mut x = vec![0.0f32; 320];
    for b in 0..10 {
        for j in 0..32 {
            let phase = (b as f32 * 0.31 + j as f32 * 0.17).sin();
            x[b * 32 + j] = phase * (1.0 + b as f32 * 10.0);
        }
    }
    let m = d.q8_0_round_trip(&x);
    let c = q8_0_round_trip(&x);
    assert_byte_exact(&m, &c, "multi_block_independent_scale");
}

#[test]
fn q8_0_round_trip_partial_tail_block_byte_exact() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    // 40 elts = 1 full block + 8-elt tail. Tail must quantize against its
    // own amax.
    let mut x = vec![0.0f32; 40];
    x[3] = 0.5;
    x[7] = -1.2;
    x[15] = 100.0;
    x[31] = 0.001;
    x[32] = 2.0; // tail peak
    x[34] = 1.0;
    x[37] = -0.5;
    let m = d.q8_0_round_trip(&x);
    let c = q8_0_round_trip(&x);
    assert_byte_exact(&m, &c, "partial_tail_byte_exact");
}

#[test]
fn q8_0_round_trip_zero_block_returns_zeros() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let x = vec![0.0f32; 64]; // two full zero blocks
    let m = d.q8_0_round_trip(&x);
    assert!(m.iter().all(|&v| v == 0.0), "zero block must yield zero");
}

#[test]
fn q8_0_round_trip_empty_input() {
    let d = MetalDispatcher::new().expect("MetalDispatcher::new");
    let m = d.q8_0_round_trip(&[]);
    assert!(m.is_empty());
}
