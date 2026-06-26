//! Phase E M5.4.5-prep — `BatchScope::hc_expand_attn` (has_add=0) smoke test.
//!
//! Computes the attention-half HC expand:
//!   out[dst, e] = post[dst] * attn_out[e]
//!              + Σ_src(comb_attn[dst, src] * cur_hc[src, e])
//!
//! and verifies bit-close match with a CPU reference.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

fn cpu_hc_expand_attn(
    attn_out: &[f32],
    cur_hc: &[f32],
    hc_split_post: &[f32],
    hc_split_comb: &[f32],
    n_hc: usize,
    d_embd: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_hc * d_embd];
    for dst in 0..n_hc {
        let base = dst * d_embd;
        let w_post = hc_split_post[dst];
        for e in 0..d_embd {
            let mut acc = w_post * attn_out[e];
            for src in 0..n_hc {
                acc += hc_split_comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
            }
            out[base + e] = acc;
        }
    }
    out
}

#[test]
fn hc_expand_attn_matches_cpu() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n_hc = 4;
    let d_embd = 4096;

    let attn_out: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.017).sin() * 0.4 + 0.05).clamp(-2.0, 2.0))
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| ((i as f32 * 0.013).cos() * 0.25))
        .collect();
    let hc_split_post: Vec<f32> = (0..n_hc).map(|i| 0.5 + (i as f32) * 0.125).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| 0.25 + (i as f32 * 0.05) - ((i % 3) as f32) * 0.1)
        .collect();

    let want = cpu_hc_expand_attn(&attn_out, &cur_hc, &hc_split_post, &hc_split_comb, n_hc, d_embd);

    let scope = disp.batch_scope();
    let a_b = scope.upload_f32(&attn_out);
    let c_b = scope.upload_f32(&cur_hc);
    let p_b = scope.upload_f32(&hc_split_post);
    let comb_b = scope.upload_f32(&hc_split_comb);
    let out = scope
        .hc_expand_attn(&a_b, &c_b, &p_b, &comb_b, n_hc, d_embd)
        .expect("hc_expand_attn");
    let got = scope.flush_and_read(&out);

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let abs = (g - w).abs();
        assert!(
            abs < 1e-5,
            "hc_expand_attn[{i}]: gpu={g} cpu={w} |diff|={abs}"
        );
    }
}
