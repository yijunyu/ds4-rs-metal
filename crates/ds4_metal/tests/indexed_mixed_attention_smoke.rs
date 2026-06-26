//! Validates the BATCHED mixed-attention flash `encode_indexed_mixed_attention`
//! (step 3 of the antirez batched-prefill-attention port; wires emitted
//! ds4_dsv4_indexed_mixed_attention_h8) against a CPU online-softmax oracle:
//! for each (token, head), attend [SWA raw window | top_k comp rows (idx<visible)]
//! + per-head sink. KV is cast to half in the kernel, so we compare with a
//! relative tolerance.
//!
//! macOS-only.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

// Identity: pure-f32 oracle. The kernel half-casts q/k (~1e-3 error), which the
// 3e-2 relative tolerance below easily absorbs — this validates the STRUCTURE
// (correct rows attended, masking, online softmax), not bit-exactness.
fn h(x: f32) -> f32 {
    x
}

#[test]
fn indexed_mixed_attention_matches_cpu_oracle() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n_tokens = 6usize;
    let n_head = 8usize;
    let head_dim = 512usize;
    let n_raw = 6usize; // = available positions (pos0..pos0+n_tokens); <= raw_cap
    let n_comp = 16usize;
    let top_k = 4usize;
    let ratio = 4u32;
    let window = 8u32;
    let pos0 = 0u32;
    let raw_cap = 8usize;
    let raw_start = 0u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut seed = 0xBEEFu64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let q = (0..n_tokens * n_head * head_dim).map(|_| rnd()).collect::<Vec<f32>>();
    let raw = (0..raw_cap * head_dim).map(|_| rnd()).collect::<Vec<f32>>();
    let comp = (0..n_comp * head_dim).map(|_| rnd()).collect::<Vec<f32>>();
    let sinks = (0..n_head).map(|_| rnd()).collect::<Vec<f32>>();
    // topk_sel[token] = [0,1,2,3] (kernel masks idx>=visible per token).
    let topk: Vec<i32> = (0..n_tokens).flat_map(|_| (0..top_k as i32)).collect();

    let gpu: Vec<f32> = {
        let scope = disp.batch_scope();
        let q_db = scope.upload_f32(&q);
        let raw_db = scope.upload_f32(&raw);
        let comp_db = scope.upload_f32(&comp);
        let sel_db = scope.upload_i32(&topk);
        let sink_db = scope.upload_f32(&sinks);
        let o = scope
            .encode_indexed_mixed_attention(
                &q_db, &raw_db, &comp_db, &sel_db, &sink_db, n_tokens, n_head, head_dim,
                n_raw, n_comp, top_k, ratio, window, pos0, raw_start, raw_cap as u32, scale,
            )
            .expect("encode_indexed_mixed_attention");
        scope.flush_and_read(&o)
    };

    // CPU online-softmax oracle (half-cast q/k to match the kernel).
    let mut worst = 0.0f64;
    for t in 0..n_tokens {
        let qpos = pos0 as usize + t;
        let last_pos = pos0 as usize + n_tokens - 1;
        let first_raw_pos = last_pos + 1 - n_raw;
        let raw_last_pos = first_raw_pos + n_raw - 1;
        let window_first = if window != 0 && qpos + 1 > window as usize { qpos + 1 - window as usize } else { 0 };
        let first = first_raw_pos.max(window_first);
        let last = qpos.min(raw_last_pos);
        let visible = ((qpos + 1) / ratio as usize).min(n_comp);
        for hd in 0..n_head {
            let qh: Vec<f32> = q[(t * n_head + hd) * head_dim..(t * n_head + hd) * head_dim + head_dim]
                .iter().map(|&x| h(x)).collect();
            let mut m = f32::MIN / 2.0;
            let mut s = 0.0f32;
            let mut o = vec![0.0f32; head_dim];
            let mut attend = |k: &[f32], m: &mut f32, s: &mut f32, o: &mut [f32]| {
                let kh: Vec<f32> = k.iter().map(|&x| h(x)).collect();
                let score: f32 = qh.iter().zip(&kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                let nm = m.max(score);
                let os = (*m - nm).exp();
                let rs = (score - nm).exp();
                *s = *s * os + rs;
                for (oi, ki) in o.iter_mut().zip(&kh) {
                    *oi = *oi * os + *ki * rs;
                }
                *m = nm;
            };
            if first <= last {
                for pos in first..=last {
                    let logical = pos - first_raw_pos;
                    let row = (raw_start as usize + logical) % raw_cap;
                    attend(&raw[row * head_dim..row * head_dim + head_dim], &mut m, &mut s, &mut o);
                }
            }
            for i in 0..top_k {
                let idx = topk[t * top_k + i];
                if idx < 0 { continue; }
                if idx as usize >= visible { break; }
                let c = idx as usize;
                attend(&comp[c * head_dim..c * head_dim + head_dim], &mut m, &mut s, &mut o);
            }
            // sink (no value).
            {
                let nm = m.max(sinks[hd]);
                let os = (m - nm).exp();
                let rs = (sinks[hd] - nm).exp();
                s = s * os + rs;
                for oi in o.iter_mut() { *oi *= os; }
                m = nm;
            }
            let inv = if s == 0.0 { 0.0 } else { 1.0 / s };
            for d in 0..head_dim {
                let g = gpu[(t * n_head + hd) * head_dim + d];
                let want = o[d] * inv;
                let rel = ((g - want).abs() / want.abs().max(1e-2)) as f64;
                worst = worst.max(rel);
            }
        }
    }
    eprintln!("[indexed_mixed_attention] worst_rel={worst:.3e}");
    assert!(worst < 3e-2, "mixed attention diverges from CPU oracle: worst_rel={worst:.3e}");
}
