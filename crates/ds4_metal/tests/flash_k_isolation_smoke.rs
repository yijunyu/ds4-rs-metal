//! Lever A isolation smoke: the K-query batched flash (`flash_attn_decode_k_metal`,
//! ne01=K + shared f16 workspace + per-query causal mask) must match, per query, the
//! proven per-position flash over the SAME workspace prefix.
//!
//! Each query r attends a DIFFERENT visible prefix v_r (32*(r+1)) — this catches any
//! query/mask/output mis-indexing in the ne01=K args (if query 0 read query 3's mask or
//! wrote the wrong output row, the per-query compare would fail). Same kernel + the
//! masked rows contribute exp(-inf)=0, so the batched query r ≈ the K=1 flash over
//! ws[0..v_r] (rel ~ fp32). macOS-only.
#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

fn params(n_head: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0, d_embd: 16, n_hc: 4, n_head, head_dim: 512, n_rot: 0,
        n_lora_q: 16, n_lora_kv: 512, hc_sinkhorn_iter: 2, hc_eps: 1e-6, rms_eps: 1e-5,
        rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
        rope_ext_factor: 0.0, rope_attn_factor: 1.0, compress_ratio: 1, n_out_group: 2,
    }
}

fn half_from_f32(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = (b >> 13) & 0x3ff;
    if e <= 0 { return s; }
    if e >= 0x1f { return s | 0x7c00; }
    s | ((e as u16) << 10) | m as u16
}

fn randf(n: usize, seed: u32) -> Vec<f32> {
    let mut r = seed;
    (0..n).map(|_| { r = r.wrapping_mul(1664525).wrapping_add(1013904223); ((r >> 9) & 0x7fff) as f32 / 32768.0 - 0.5 }).collect()
}

#[test]
fn flash_k_matches_per_position() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let nh: u32 = 4;
    let hd = 512usize;
    let p = params(nh);
    let k = 4usize;
    let n_total = 128u32; // 32-aligned
    let nt = n_total as usize;

    let ws_f32 = randf(nt * hd, 0xA11);
    let ws: Vec<u16> = ws_f32.iter().map(|&v| half_from_f32(v)).collect();
    let q = randf(k * nh as usize * hd, 0xB22);
    let sinks = vec![0.25f32; nh as usize];

    // per-query visible prefix v_r = 32*(r+1) (each distinct, all 32-aligned).
    let vis: Vec<u32> = (0..k).map(|r| (32 * (r as u32 + 1)).min(n_total)).collect();
    const NEG: u16 = 0xFC00;
    let mut mask = vec![0u16; k * nt];
    for r in 0..k {
        for j in 0..nt {
            mask[r * nt + j] = if (j as u32) < vis[r] { 0 } else { NEG };
        }
    }

    let out_k = disp.debug_flash_k(&p, &q, &ws, &mask, &sinks, k, n_total);
    assert_eq!(out_k.len(), k * nh as usize * hd);

    let mut max_rel = 0.0f32;
    for r in 0..k {
        let qr = &q[r * nh as usize * hd..(r + 1) * nh as usize * hd];
        let refr = disp.debug_flash_1(&p, qr, &ws, vis[r], vis[r], &sinks);
        let row = &out_k[r * nh as usize * hd..(r + 1) * nh as usize * hd];
        let mut num = 0.0f32; let mut den = 0.0f32;
        for j in 0..nh as usize * hd {
            assert!(row[j].is_finite() && refr[j].is_finite(), "nan r={r} j={j}");
            num = num.max((row[j] - refr[j]).abs());
            den = den.max(refr[j].abs());
        }
        let rel = num / den.max(1e-4);
        max_rel = max_rel.max(rel);
        eprintln!("flash_k r={r} vis={} rel={rel:.3e}", vis[r]);
    }
    assert!(max_rel < 5e-3, "flash_k vs per-position rel={max_rel:.3e} — ne01=K dispatch wrong");
    eprintln!("flash_k_matches_per_position OK: K={k} n_total={n_total} max_rel={max_rel:.2e}");
}
