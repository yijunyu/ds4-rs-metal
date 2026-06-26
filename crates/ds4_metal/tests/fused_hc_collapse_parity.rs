//! Megakernel Phase-1 PoC parity: the fused hc_collapse stage-1+2 kernel
//! (ds4_dsv4_hc_collapse_rms_mv_f32) must match the two-kernel path
//! rms_norm_mul(prev_hc, unit) → matvec_f32(hc_fn) to ~1e-4 rel (different
//! reduction order → not bit-identical, but argmax-stable). Synthetic data,
//! no model load. macOS-only. Opt-in: DS4_TEST_FUSE_HC=1.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

#[test]
fn fused_hc_collapse_matches_two_kernel() {
    if std::env::var("DS4_TEST_FUSE_HC").ok().as_deref() != Some("1") {
        eprintln!("DS4_TEST_FUSE_HC unset — skipping. Set =1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_hc = 4usize;
    let n_embd = 4096usize;
    let hc_dim = n_hc * n_embd; // 16384
    let mix_hc = 2 * n_hc + n_hc * n_hc; // 24
    let eps = 1e-5f32;

    // Deterministic synthetic prev_hc + hc_fn.
    let mut rng: u32 = 0x12345678;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let prev_hc: Vec<f32> = (0..hc_dim).map(|_| next() * 2.0).collect();
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc).map(|_| next() * 0.05).collect();
    let unit_gamma: Vec<f32> = vec![1.0; hc_dim];

    let scope = disp.batch_scope();
    let prev_db = scope.upload_f32(&prev_hc);
    let hcfn_db = scope.weight_f32(&hc_fn);
    let unit_db = scope.weight_f32(&unit_gamma);

    // Reference: two-kernel path.
    let flat = scope.rms_norm_mul(&prev_db, &unit_db, eps).expect("rms");
    let mix_ref = scope.matvec_f32(&hcfn_db, &flat, hc_dim, mix_hc).expect("matvec");
    // Fused.
    let mix_fused = scope
        .fused_hc_collapse_stage12(&hcfn_db, &prev_db, hc_dim, mix_hc, eps, false)
        .expect("fused");

    let outs = scope.flush_and_read_multi(&[&mix_ref, &mix_fused]);
    let r = &outs[0];
    let f = &outs[1];
    assert_eq!(r.len(), mix_hc);
    assert_eq!(f.len(), mix_hc);

    let mut max_rel = 0.0f32;
    for i in 0..mix_hc {
        let denom = r[i].abs().max(1e-6);
        let rel = (r[i] - f[i]).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(f[i].is_finite(), "fused[{i}] not finite");
    }
    eprintln!("fused vs ref: max_rel = {max_rel:.2e}");
    eprintln!("  ref[0..4]   = {:?}", &r[0..4]);
    eprintln!("  fused[0..4] = {:?}", &f[0..4]);
    assert!(max_rel < 1e-4, "fused hc_collapse rel {max_rel:.2e} >= 1e-4");
}
