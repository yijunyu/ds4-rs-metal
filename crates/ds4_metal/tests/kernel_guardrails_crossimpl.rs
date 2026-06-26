//! GUARDRAIL TEST — CROSS-IMPLEMENTATION equivalence (a NEW guardrail category).
//!
//! The chunk-prefill bisection cleared every individual K-batched kernel (each is
//! faithful to K=1), leaving a CROSS-KERNEL mismatch: the chunk noidx core and the
//! per-token reference run DIFFERENT attention IMPLEMENTATIONS. This test pins
//! whether those two implementations actually agree on IDENTICAL inputs:
//!   A = `flash_attn_k_mla_comp_masked`         — f32 raw + comp, half-cast inside
//!   B = `build_chunk_kv_workspace` (f16 ws) + `flash_attn_decode_k` — the
//!       per-token-family path (gather raw+comp → f16 workspace → flash)
//! Both attend [raw window | comp rows] + per-head sink over the SAME synthetic
//! raw/comp/q. If A≡B (close), the two flash kernels are equivalent ⇒ the
//! production chunk-vs-pertoken divergence is NOT the kernel choice but the RAW-KV
//! PRECISION fed to them (chunk f32 vs per-token fp8→f16). If A≠B, the kernels
//! themselves disagree — the bug is in one of them.
#![cfg(target_os = "macos")]

mod guardrails;
use guardrails::*;

use ds4_engine::attn_dispatch::LayerParams;
use ds4_metal::MetalDispatcher;

const NEG: u16 = 0xFC00;

fn mla_params(n_head: usize) -> LayerParams {
    LayerParams {
        layer_idx: 0, d_embd: 4096, n_hc: 4,
        n_head: n_head as u32, head_dim: 512, n_rot: 0,
        n_lora_q: 1, n_lora_kv: 512,
        hc_sinkhorn_iter: 20, hc_eps: 1e-6, rms_eps: 1e-5,
        rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
        rope_ext_factor: 0.0, rope_attn_factor: 1.0,
        compress_ratio: 128, n_out_group: 1,
    }
}

#[test]
fn flash_comp_kernel_vs_f16_workspace_crossimpl() {
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: MetalDispatcher::new failed: {e}"); return; }
    };
    let (n_head, dk, n_cache, n_comp, k) = (8usize, 512usize, 8usize, 8usize, 8usize);
    let n_rows = n_cache + n_comp;
    let scale = 1.0f32 / (dk as f32).sqrt(); // matches flash_attn_decode_k's internal 1/√dk
    let p = mla_params(n_head);

    let q = rand_f32(k * n_head * dk, 0xA1);
    let raw = rand_f32(n_cache * dk, 0xB2);
    let comp = rand_f32(n_comp * dk, 0xC3);
    let sinks = rand_f32(n_head, 0xD4);

    // A — comp-kernel (f32 raw + comp). Mask: attend all n_rows for every query.
    let a = {
        let s = disp.batch_scope();
        let q_db = s.upload_f32(&q);
        let raw_db = s.upload_f32(&raw);
        let comp_db = s.upload_f32(&comp);
        let sink_db = s.upload_f32(&sinks);
        let mask = vec![0u16; k * n_rows]; // all attend
        let out = s.flash_attn_k_mla_comp_masked(
            &q_db, &raw_db, comp_db.buffer(), n_comp as u32, &mask,
            n_head, dk, dk, n_cache, k, scale, &sink_db,
        ).expect("flash_attn_k_mla_comp_masked");
        s.flush_and_read(&out)
    };

    // B — f16-workspace path (build [raw|comp] f16 → flash_attn_decode_k).
    let b = {
        let s = disp.batch_scope();
        let q_db = s.upload_f32(&q);
        let raw_db = s.upload_f32(&raw);
        let comp_db = s.upload_f32(&comp);
        let n_total = n_rows as u32;
        let n_total_padded = n_total.div_ceil(32) * 32;
        let ws = s.build_chunk_kv_workspace(&raw_db, comp_db.buffer(), n_cache as u32, n_comp as u32, dk as u32, n_total_padded)
            .expect("build_chunk_kv_workspace");
        // mask [K, n_total_padded]: cols [0..n_rows) attend, pad -inf.
        let ntp = n_total_padded as usize;
        let mut mask = vec![NEG; k * ntp];
        for t in 0..k {
            for c in 0..n_rows { mask[t * ntp + c] = 0; }
        }
        let out = s.flash_attn_decode_k(&p, &q_db, &ws, n_total_padded, &mask, &sinks, k)
            .expect("flash_attn_decode_k");
        s.flush_and_read(&out)
    };

    // CROSS-IMPL equivalence: do the two flash implementations agree per query?
    let row_dim = n_head * dk;
    let (flip, pert, minc) = row_divergence(&a, &b, row_dim);
    eprintln!(
        "[guardrail] crossimpl comp-kernel vs f16-workspace: rel_L2={:.3e} cos={:.6} \
         rows: flip%={:.1} pert%={:.1} minRowCos={:.4}",
        rel_l2(&a, &b), cosine(&a, &b), flip * 100.0, pert * 100.0, minc,
    );
    // Both are f16-KV attention over identical rows ⇒ should agree to f16/reduction
    // tolerance. A FAIL here means the two implementations genuinely disagree (kernel
    // bug); a PASS means the production chunk-vs-pertoken drift is the RAW-KV precision
    // (f32 vs fp8→f16) fed to them, not the kernel choice.
    // measured rel_L2 ≈ 1.2e-7 (fp32-epsilon) ⇒ the two implementations are
    // EQUIVALENT. 1e-5 = a tight gate that still tolerates reduction-order noise;
    // a real kernel divergence would blow past it.
    assert_close_to_oracle("crossimpl flash comp-kernel ≡ f16-workspace", &a, &b, 1e-5);
}
