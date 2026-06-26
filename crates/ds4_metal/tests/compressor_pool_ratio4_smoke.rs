//! M5 Phase D — `ds4_kernel_dsv4_compressor_pool_ratio4` smoke.
//!
//! Bespoke kernel that mirrors `compressor_pool_decode_state`
//! (`attn_dispatch.rs:1985`) ratio==4 branch directly: per output
//! column `j` in `[0..head_dim]`, max-stabilized softmax over `2*ratio`
//! scores drawn from the 2-window state (`[2*ratio, 2*head_dim]`
//! row-major), weighted sum of matching KV values. Includes the
//! `-1e9` masked-slot sentinel handled identically to the CPU oracle.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

const NEG_INF: f32 = -1.0e9;

/// CPU oracle — verbatim copy of the ratio==4 branch of
/// `compressor_pool_decode_state` (`attn_dispatch.rs:1985`). Kept here
/// because that function is private to `ds4_engine`. Layout:
/// state buffers are row-major `[2*ratio, 2*head_dim]`, width = 2*hd.
fn cpu_oracle_ratio4(
    state_kv: &[f32],
    state_score: &[f32],
    head_dim: usize,
) -> Vec<f32> {
    let ratio = 4usize;
    let width = 2 * head_dim;
    assert_eq!(state_kv.len(), 2 * ratio * width);
    assert_eq!(state_score.len(), 2 * ratio * width);
    let mut out = vec![0.0f32; head_dim];
    for j in 0..head_dim {
        let mut max_score = NEG_INF;
        for r in 0..ratio {
            let sp = state_score[r * width + j];
            let sc = state_score[(ratio + r) * width + head_dim + j];
            if sp > max_score {
                max_score = sp;
            }
            if sc > max_score {
                max_score = sc;
            }
        }
        if max_score <= NEG_INF * 0.5 {
            out[j] = 0.0;
            continue;
        }
        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        for r in 0..ratio {
            let wp = (state_score[r * width + j] - max_score).exp();
            let wc = (state_score[(ratio + r) * width + head_dim + j] - max_score).exp();
            denom += wp + wc;
            sum += wp * state_kv[r * width + j];
            sum += wc * state_kv[(ratio + r) * width + head_dim + j];
        }
        out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }
    out
}

fn synth_state(head_dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let ratio = 4usize;
    let width = 2 * head_dim;
    let n = 2 * ratio * width;
    let kv: Vec<f32> = (0..n)
        .map(|i| (((i + seed as usize) as f32) * 0.013).sin() * 0.4)
        .collect();
    let score: Vec<f32> = (0..n)
        .map(|i| (((i + seed as usize) as f32) * 0.017).cos() * 0.3)
        .collect();
    (kv, score)
}

fn run_case(disp: &MetalDispatcher, head_dim: u32, kv: &[f32], score: &[f32]) {
    let ref_out = cpu_oracle_ratio4(kv, score, head_dim as usize);

    let scope = disp.batch_scope();
    let kv_db = scope.upload_f32(kv);
    let score_db = scope.upload_f32(score);
    let out_db = scope
        .compressor_pool_ratio4(kv_db.buffer(), score_db.buffer(), head_dim)
        .expect("compressor_pool_ratio4");
    let gpu_out = scope.flush_and_read(&out_db);

    assert_eq!(gpu_out.len(), head_dim as usize);
    let mut max_abs: f32 = 0.0;
    let mut max_ref: f32 = 0.0;
    for i in 0..head_dim as usize {
        max_abs = max_abs.max((gpu_out[i] - ref_out[i]).abs());
        max_ref = max_ref.max(ref_out[i].abs());
    }
    let rel = max_abs / max_ref.max(1e-6);
    assert!(
        max_abs < 5e-5 && rel < 5e-5,
        "compressor_pool_ratio4 drift exceeds tolerance: head_dim={head_dim} \
         max_abs={max_abs:.3e} max_ref={max_ref:.3e} rel={rel:.3e}",
    );
}

#[test]
fn compressor_pool_ratio4_matches_cpu_oracle_v4_flash_main() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // V4 Flash main compressor: ratio=4, head_dim=DS4_N_HEAD_DIM=512.
    for seed in 0..3 {
        let (kv, score) = synth_state(512, seed);
        run_case(&disp, 512, &kv, &score);
    }
}

#[test]
fn compressor_pool_ratio4_matches_cpu_oracle_indexer_dims() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // Indexer compressor: ratio=4, head_dim=DS4_N_INDEXER_HEAD_DIM=128.
    for seed in 0..3 {
        let (kv, score) = synth_state(128, seed);
        run_case(&disp, 128, &kv, &score);
    }
}

#[test]
fn compressor_pool_ratio4_handles_small_head_dim() {
    // Below threadgroup width (32) exercises the bounds check.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let (kv, score) = synth_state(17, 42);
    run_case(&disp, 17, &kv, &score);
}

#[test]
fn compressor_pool_ratio4_returns_zero_when_all_slots_masked() {
    // -1e9 sentinel matches `decode_step.rs:493` CPU init. When every
    // score for column j is the sentinel, max_score <= -5e8 and the
    // output collapses to 0 — same as the CPU oracle's early return.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let head_dim = 64usize;
    let ratio = 4usize;
    let width = 2 * head_dim;
    let n = 2 * ratio * width;
    let kv = vec![3.14f32; n]; // value should not leak through
    let score = vec![NEG_INF; n];

    let ref_out = cpu_oracle_ratio4(&kv, &score, head_dim);
    assert!(
        ref_out.iter().all(|v| *v == 0.0),
        "oracle invariant: all-masked input → output must be 0"
    );

    let scope = disp.batch_scope();
    let kv_db = scope.upload_f32(&kv);
    let score_db = scope.upload_f32(&score);
    let out_db = scope
        .compressor_pool_ratio4(kv_db.buffer(), score_db.buffer(), head_dim as u32)
        .expect("compressor_pool_ratio4");
    let gpu_out = scope.flush_and_read(&out_db);

    assert_eq!(gpu_out.len(), head_dim);
    for v in &gpu_out {
        assert_eq!(*v, 0.0, "all-masked input on GPU must produce 0, got {v}");
    }
}

/// The state_score pools must seed with -1e9 (matching
/// `decode_step.rs:493`). Without this, pos=3 (first emit) reads the
/// untouched prev-window slots as 0, weighting them by `exp(0-max)`
/// instead of `exp(-1e9-max) ≈ 0`, and the GPU diverges from the CPU
/// mirror.
#[test]
fn state_score_pools_seed_neg_inf_on_first_alloc() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let head_dim = 32usize;
    let ratio = 4usize;
    let width = 2 * head_dim;
    let state_bytes = 2 * ratio * width * std::mem::size_of::<f32>();

    // Use very high layer indices to avoid colliding with other tests
    // in this binary (the pool persists across tests sharing a
    // MetalDispatcher).
    let li_a = 9001u32;
    let li_b = 9002u32;
    let comp_score = disp.compressor_state_score_or_alloc(li_a, state_bytes);
    let idx_score = disp.indexer_state_score_or_alloc(li_b, state_bytes);

    let n = state_bytes / std::mem::size_of::<f32>();
    let read = |buf: &metal::Buffer| -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const f32,
                out.as_mut_ptr(),
                n,
            );
        }
        out
    };
    for v in read(&comp_score) {
        assert_eq!(v, -1.0e9, "compressor_state_score must seed with -1e9");
    }
    for v in read(&idx_score) {
        assert_eq!(v, -1.0e9, "indexer_state_score must seed with -1e9");
    }

    // The kv pools should still seed with 0 (matches the CPU mirror's
    // `vec![0.0; rows*width]` initializer at decode_step.rs:490).
    let li_c = 9003u32;
    let li_d = 9004u32;
    let comp_kv = disp.compressor_state_kv_or_alloc(li_c, state_bytes);
    let idx_kv = disp.indexer_state_kv_or_alloc(li_d, state_bytes);
    for v in read(&comp_kv) {
        assert_eq!(v, 0.0, "compressor_state_kv must seed with 0");
    }
    for v in read(&idx_kv) {
        assert_eq!(v, 0.0, "indexer_state_kv must seed with 0");
    }
}

#[test]
fn compressor_pool_ratio4_handles_partial_mask() {
    // Realistic decode-emit warm-up: some prev-window slots still hold
    // the -1e9 sentinel while the current window has been written. The
    // CPU oracle absorbs the masked rows via exp(-1e9 - max) ≈ 0; the
    // GPU should produce the same result.
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let head_dim = 64usize;
    let ratio = 4usize;
    let width = 2 * head_dim;
    let n = 2 * ratio * width;
    let mut kv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 0.3).collect();
    let mut score: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019).cos() * 0.4).collect();
    // Mask the first 2 prev-window rows entirely.
    for r in 0..2 {
        for j in 0..width {
            kv[r * width + j] = 7.7; // value should be down-weighted to ~0
            score[r * width + j] = NEG_INF;
        }
    }
    run_case(&disp, head_dim as u32, &kv, &score);
}
