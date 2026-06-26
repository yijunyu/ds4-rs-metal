//! STAGE 1 model-free closeness test for the fused chunk-prefill compressor
//! kernel (`compressor_prefill_noidx`). Builds synthetic batched projections
//! kv/sc + APE + norm, runs the fused kernel for a noidx (ratio != 4) layer,
//! and compares each emitted comp row against a CPU oracle that replicates the
//! per-position chain softmax-pool → rms-norm → rope_tail exactly.
//!
//! This is the "kernel math" gate. The cross-chunk INTEGRATION (comp ring
//! offsets, n_comp, full-model output) is covered by
//! `comp_prefill_integration.rs` (needs the real model).
#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

/// CPU oracle: build comp row `e` for a noidx layer (coff==1, width==head_dim).
fn cpu_emit_row(
    kv: &[f32],
    sc: &[f32],
    ape: &[f32], // [head_dim * ratio], ape[j*ratio + r]
    norm: &[f32],
    head_dim: usize,
    ratio: usize,
    n_rot: usize,
    pos0: u32,
    e: usize,
    params: &LayerParams,
    rms_eps: f32,
) -> Vec<f32> {
    let width = head_dim;
    let base_row = e * ratio;
    // softmax-weighted column pool over `ratio` rows.
    let mut pooled = vec![0.0f32; head_dim];
    for j in 0..head_dim {
        let mut max_s = f32::NEG_INFINITY;
        for r in 0..ratio {
            let pos = pos0 as usize + base_row + r;
            let pos_mod = pos % ratio;
            let s = sc[(base_row + r) * width + j] + ape[j * ratio + pos_mod];
            if s > max_s {
                max_s = s;
            }
        }
        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        for r in 0..ratio {
            let pos = pos0 as usize + base_row + r;
            let pos_mod = pos % ratio;
            let s = sc[(base_row + r) * width + j] + ape[j * ratio + pos_mod];
            let wgt = (s - max_s).exp();
            denom += wgt;
            sum += wgt * kv[(base_row + r) * width + j];
        }
        pooled[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }
    // RMS norm × norm weight.
    let mut ss = 0.0f32;
    for &v in &pooled {
        ss += v * v;
    }
    let rms = 1.0f32 / (ss / head_dim as f32 + rms_eps).sqrt();
    for j in 0..head_dim {
        pooled[j] = pooled[j] * rms * norm[j];
    }
    // partial RoPE on the trailing n_rot floats at comp pos = pos0 + e*ratio.
    if n_rot > 0 {
        let pos = (pos0 as usize + base_row) as f32;
        let freq_base = params.rope_freq_base;
        let freq_scale = params.rope_freq_scale;
        let ext_factor = params.rope_ext_factor;
        let attn_factor = params.rope_attn_factor;
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;
        let orig_ctx = params.rope_orig_ctx as f32;
        let pi = std::f32::consts::PI;
        let low = ((orig_ctx / (2.0 * pi * beta_fast)).ln() / freq_base.ln()
            * n_rot as f32
            * 0.5)
            .floor()
            * 2.0;
        let high = ((orig_ctx / (2.0 * pi * beta_slow)).ln() / freq_base.ln()
            * n_rot as f32
            * 0.5)
            .ceil()
            * 2.0;
        let mscale = if freq_scale <= 1.0 {
            1.0
        } else {
            0.1 * attn_factor * freq_scale.ln() + 1.0
        };
        let tail0 = head_dim - n_rot;
        let mut pair = 0;
        while pair * 2 < n_rot {
            let i0 = pair * 2;
            let exponent = i0 as f32 / n_rot as f32;
            let freq_full = 1.0 / freq_base.powf(exponent);
            let freq_inter = freq_full / freq_scale;
            let y = (i0 as f32 - low) / (high - low).max(0.001);
            let ramp = 1.0 - y.clamp(0.0, 1.0);
            let freq = freq_inter * (1.0 - ramp * ext_factor) + freq_full * ramp * ext_factor;
            let theta = pos * freq;
            let cos_t = theta.cos() * mscale;
            let sin_t = theta.sin() * mscale;
            let x0 = pooled[tail0 + i0];
            let x1 = pooled[tail0 + i0 + 1];
            pooled[tail0 + i0] = x0 * cos_t - x1 * sin_t;
            pooled[tail0 + i0 + 1] = x0 * sin_t + x1 * cos_t;
            pair += 1;
        }
    }
    pooled
}

fn mk_params(head_dim: u32, n_rot: u32, ratio: u32) -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 1024,
        n_hc: 4,
        n_head: 8,
        head_dim,
        n_rot,
        n_lora_q: 64,
        n_lora_kv: head_dim,
        hc_sinkhorn_iter: 20,
        hc_eps: 1e-6,
        rms_eps: 1.0e-6,
        rope_orig_ctx: 4096,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: ratio,
        n_out_group: 1,
    }
}

#[test]
fn comp_prefill_noidx_matches_cpu() {
    let disp = MetalDispatcher::new().expect("disp");
    let head_dim = 512usize;
    let n_rot = 64usize;

    // Sweep a couple of ratios incl. production 128, and chunk_start offsets.
    for &(ratio, n_tokens, chunk_start) in
        &[(128usize, 256usize, 0u32), (128, 384, 128), (32, 96, 0), (32, 128, 64)]
    {
        let params = mk_params(head_dim as u32, n_rot as u32, ratio as u32);
        let width = head_dim;
        // deterministic synthetic projections + weights.
        let mut seed = 0x9E37_79B9u32 ^ (ratio as u32) ^ (chunk_start << 3);
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / 16_777_216.0) - 0.5
        };
        let kv: Vec<f32> = (0..n_tokens * width).map(|_| rnd() * 0.4).collect();
        let sc: Vec<f32> = (0..n_tokens * width).map(|_| rnd() * 0.6).collect();
        let ape: Vec<f32> = (0..head_dim * ratio).map(|_| rnd() * 0.3).collect();
        let norm: Vec<f32> = (0..head_dim).map(|_| 0.8 + rnd() * 0.4).collect();

        let n_emit = (n_tokens / ratio) as u32;
        assert!(n_emit > 0);

        // comp ring buffer — fixed size across configs (comp_ring_or_alloc
        // caches one buffer per layer key and asserts a stable size, so a
        // per-config size would trip the debug_assert). 64 rows of headroom.
        const RING_ROWS: usize = 64;
        let comp_ring = disp.comp_ring_or_alloc(7, RING_ROWS * head_dim * 4);
        assert!((n_emit as usize) < RING_ROWS);
        // zero it
        unsafe {
            std::ptr::write_bytes(comp_ring.contents() as *mut u8, 0, RING_ROWS * head_dim * 4);
        }

        let scope = disp.batch_scope();
        let kv_db = scope.upload_f32(&kv);
        let sc_db = scope.upload_f32(&sc);
        scope
            .compressor_prefill_noidx(
                &kv_db, &sc_db, &ape, &norm, &comp_ring,
                head_dim as u32, ratio as u32, n_rot as u32, chunk_start, 0, n_emit,
                &params, params.rms_eps,
            )
            .expect("fused prefill");
        scope.wait_all_and_drop();

        // compare each emit row.
        let gpu = unsafe {
            std::slice::from_raw_parts(
                comp_ring.contents() as *const f32,
                n_emit as usize * head_dim,
            )
        };
        let mut max_abs = 0.0f32;
        let mut worst = (0usize, 0usize);
        for e in 0..n_emit as usize {
            let cpu = cpu_emit_row(
                &kv, &sc, &ape, &norm, head_dim, ratio, n_rot, chunk_start, e, &params,
                params.rms_eps,
            );
            for j in 0..head_dim {
                let d = (gpu[e * head_dim + j] - cpu[j]).abs();
                if d > max_abs {
                    max_abs = d;
                    worst = (e, j);
                }
            }
        }
        eprintln!(
            "[comp_prefill_smoke] ratio={ratio} n_tok={n_tokens} cs={chunk_start} n_emit={n_emit} \
             max_abs_diff={max_abs:.3e} at (row={},col={})",
            worst.0, worst.1
        );
        assert!(
            max_abs < 5.0e-4,
            "fused comp-prefill diverges from CPU oracle: max_abs={max_abs:.3e} \
             (ratio={ratio} chunk_start={chunk_start})"
        );
    }
    eprintln!("[comp_prefill_smoke] PASS: fused kernel matches CPU oracle (4 configs)");
}
