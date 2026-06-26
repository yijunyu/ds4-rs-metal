//! Phase E M5.4.5.3 — `SingleBufferEncoder::encode_first_half` smoke test.
//!
//! Drives a synthetic 1-layer dense `ComposedModelWeights` through both:
//!   - `SingleBufferEncoder::encode_first_half(l_split=1)` (the new
//!     unified-cb path; writes persistent KV via GPU shim, runs the
//!     attn-half + post-attn-half through `attn_output_proj`)
//!   - an inline reference that composes the SAME kernels via inherent
//!     helpers (BatchScope::hc_collapse_norm → disp.attn_qkv_chain_batched
//!     → disp.kv_norm_rope_chain → trait kv_fp8_store [CPU oracle] →
//!     disp.flash_attn_decode → CPU rope_q passes → AttentionDispatcher::
//!     attn_output_proj)
//!
//! Both paths share every kernel and operand EXCEPT `kv_fp8_store`:
//! the new path uses the GPU shim (`ds4_dsv4_kv_fp8_store`), the
//! reference uses the CPU oracle (delegated via the trait at
//! `lib.rs:443`). For the production shape `n_lora_kv == head_dim`,
//! both implement the antirez FP8 round-trip but are not byte-identical
//! (per `kv_fp8_store_persistent_smoke`, the persistent write is within
//! e4m3 tolerance of the input — same bar applies vs the CPU oracle).
//!
//! Asserts `state.cur_hc` is **bit-close** between the two paths
//! (within fp8 e4m3 propagation tolerance through one attn_output_proj
//! stage). Strict bit-equivalence requires unifying the kv_fp8_store
//! paths — deferred work (M5.2.4 from memory) blocked behind the
//! `attn_smoke` n_lora_kv != head_dim asymmetric layout.
//!
//! FFN-half threading (steps 6-10) lands in M5.4.5.4 with the GGUF-backed
//! bench harness — those production-shape MoE constraints (n_experts=256,
//! d_embd % 256 == 0, loaded `QuantizedExpertWeights`) can't be
//! exercised on synthetic weights.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    apply_rope_tail_with_table, precompute_rope_tail_table, AttentionDispatcher,
    AttnLayerWeights, KvCacheView, LayerParams,
};
use ds4_engine::decode_step::{
    AttnStepState, ComposedLayerWeights, ComposedModelWeights, LayerWeights,
};
use ds4_metal::single_buffer_encoder::{LayerCutpoint, SingleBufferEncoder};
use ds4_metal::MetalDispatcher;

fn ds4_params() -> LayerParams {
    // Production shape for V4 Flash attn-half kernels:
    //   - `hc_split_weighted_sum_norm4` hardcodes n_hc=4 and d_embd=4096
    //   - `flash_attn_decode_metal` requires head_dim=512 + n_lora_kv=512
    LayerParams {
        layer_idx: 0,
        d_embd: 4096,
        n_hc: 4,
        n_head: 2,
        head_dim: 512,
        n_rot: 64,
        n_lora_q: 64,
        n_lora_kv: 512,
        hc_sinkhorn_iter: 5,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 0,
        n_out_group: 2,
    }
}

struct AttnWeights {
    hc_attn_fn: Vec<f32>,
    hc_attn_scale: Vec<f32>,
    hc_attn_base: Vec<f32>,
    hc_norm_gamma: Vec<f32>,
    qkv_gamma_q: Vec<f32>,
    qkv_gamma_kv: Vec<f32>,
    attn_q_a: Vec<f32>,
    attn_q_b: Vec<f32>,
    attn_kv: Vec<f32>,
    w_o_a: Vec<f32>,
    w_o_b: Vec<f32>,
    attn_sinks: Vec<f32>,
}

fn build_synthetic_attn(p: &LayerParams) -> AttnWeights {
    let n_hc = p.n_hc as usize;
    let d_embd = p.d_embd as usize;
    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let n_lora_q = p.n_lora_q as usize;
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let q_dim = n_head * head_dim;
    let kv_row = p.n_lora_kv as usize;
    let n_groups = p.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o: usize = 4;
    let out_low_dim = n_groups * n_lora_o;
    AttnWeights {
        hc_attn_fn: (0..hc_dim * mix_hc)
            .map(|i| (i as f32 * 0.0011).sin() * 0.05)
            .collect(),
        hc_attn_scale: vec![1.0, 0.5, 2.0],
        hc_attn_base: (0..mix_hc).map(|i| 0.1 + (i as f32) * 0.01).collect(),
        hc_norm_gamma: (0..d_embd)
            .map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.05)
            .collect(),
        qkv_gamma_q: (0..n_lora_q)
            .map(|i| 1.0 + (i as f32 * 0.011).sin() * 0.05)
            .collect(),
        qkv_gamma_kv: (0..kv_row)
            .map(|i| 1.0 + (i as f32 * 0.021).sin() * 0.04)
            .collect(),
        attn_q_a: (0..n_lora_q * d_embd)
            .map(|i| (i as f32 * 0.009).cos() * 0.25)
            .collect(),
        attn_q_b: (0..q_dim * n_lora_q)
            .map(|i| (i as f32 * 0.013).sin() * 0.2)
            .collect(),
        attn_kv: (0..kv_row * d_embd)
            .map(|i| (i as f32 * 0.019).cos() * 0.15)
            .collect(),
        w_o_a: (0..out_low_dim * group_dim)
            .map(|i| (i as f32 * 0.007).sin() * 0.1)
            .collect(),
        w_o_b: (0..d_embd * out_low_dim)
            .map(|i| (i as f32 * 0.012).cos() * 0.15)
            .collect(),
        attn_sinks: (0..n_head).map(|h| -0.5 - (h as f32) * 0.1).collect(),
    }
}

fn build_attn_layer_weights(p: LayerParams, w: &AttnWeights) -> AttnLayerWeights {
    AttnLayerWeights {
        params: p,
        hc_attn_fn: w.hc_attn_fn.clone(),
        hc_attn_scale: w.hc_attn_scale.clone(),
        hc_attn_base: w.hc_attn_base.clone(),
        hc_ffn_fn: Vec::new(),
        hc_ffn_scale: Vec::new(),
        hc_ffn_base: Vec::new(),
        hc_attn_fn_f16: Vec::new().into(),
        hc_ffn_fn_f16: Vec::new().into(),
        hc_norm_gamma: w.hc_norm_gamma.clone(),
        hc_ffn_norm_gamma: Vec::new(),
        qkv_gamma_q: w.qkv_gamma_q.clone(),
        qkv_gamma_kv: w.qkv_gamma_kv.clone(),
        attn_q_a: w.attn_q_a.clone(),
        attn_q_b: w.attn_q_b.clone(),
        attn_kv: w.attn_kv.clone(),
        attn_q_a_q8: Vec::new().into(),
        attn_q_b_q8: Vec::new().into(),
        attn_kv_q8: Vec::new().into(),
        w_o_a_q8: Vec::new().into(),
        w_o_b_q8: Vec::new().into(),
        w_shared_gate_q8: Vec::new().into(),
        w_shared_up_q8: Vec::new().into(),
        w_shared_down_q8: Vec::new().into(),
        w_o_a: w.w_o_a.clone(),
        w_o_b: w.w_o_b.clone(),
        attn_sinks: w.attn_sinks.clone(),
        w_shared_gate: Vec::new(),
        w_shared_up: Vec::new(),
        w_shared_down: Vec::new(),
        shared_dim: 0,
        shared_clamp: 0.0,
        attn_compressor_kv: Vec::new(),
        attn_compressor_gate: Vec::new(),
        attn_compressor_ape: Vec::new(),
        attn_compressor_norm: Vec::new(),
        indexer_compressor_kv: Vec::new(),
        indexer_compressor_gate: Vec::new(),
        indexer_compressor_ape: Vec::new(),
        indexer_compressor_norm: Vec::new(),
        attn_compressor_kv_f16: Vec::new().into(),
        attn_compressor_gate_f16: Vec::new().into(),
        indexer_compressor_kv_f16: Vec::new().into(),
        indexer_compressor_gate_f16: Vec::new().into(),
        indexer_attn_q_b: Vec::new(),
        indexer_proj: Vec::new(),
        indexer_attn_q_b_f16: Vec::new().into(),
        indexer_proj_f16: Vec::new().into(),
    }
}

fn build_moe_stub(d_embd: usize) -> LayerWeights {
    LayerWeights {
        n_experts: 1,
        n_experts_used: 1,
        d_model: d_embd,
        d_ffn: std::num::NonZeroUsize::new(1),
        attn_norm_gamma: vec![1.0; d_embd],
        w_attn: Vec::new(),
        ffn_norm_gamma: vec![1.0; d_embd],
        w_router: vec![0.0; d_embd],
        w_router_f16: Vec::new().into(),
        router_bias: vec![0.0; 1],
        routing_table: None,
        w_gate_exps: Vec::new(),
        w_up_exps: Vec::new(),
        w_down_exps: Vec::new(),
    }
}

fn cur_hc_from_x(x: &[f32], n_hc: usize) -> Vec<f32> {
    let d_embd = x.len();
    let mut v = vec![0.0f32; n_hc * d_embd];
    for h in 0..n_hc {
        v[h * d_embd..(h + 1) * d_embd].copy_from_slice(x);
    }
    v
}

#[test]
fn encode_first_half_after_attn_hc_matches_inline_reference() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let p = ds4_params();
    let d_embd = p.d_embd as usize;
    let n_hc = p.n_hc as usize;
    let hc_dim = n_hc * d_embd;
    let n_head = p.n_head as usize;
    let head_dim = p.head_dim as usize;
    let n_lora_q = p.n_lora_q as usize;
    let kv_row = p.n_lora_kv as usize;
    let n_rot = p.n_rot as usize;
    // raw_cap=32, pos=31 → n_raw=32, satisfies flash_attn_decode_metal
    // 32-alignment + nonzero requirements. Slots [0..31] pre-filled
    // with deterministic data byte-identical between paths.
    let raw_cap: u32 = 32;
    let pos: u32 = 31;
    let n_prefilled: u32 = 31;

    let w = build_synthetic_attn(&p);
    let attn_layer = build_attn_layer_weights(p.clone(), &w);
    let moe = build_moe_stub(d_embd);
    let model = ComposedModelWeights {
        layers: vec![ComposedLayerWeights {
            attn: attn_layer.clone(),
            moe,
        }],
        final_norm_gamma: vec![1.0; d_embd],
        lm_head: vec![0.0; d_embd],
        lm_head_q8: vec![].into(),
        lm_head_q4: vec![],
        vocab_size: 1,
        d_model: d_embd,
        output_hc_base: None,
        output_hc_fn: None,
        output_hc_scale: None,
    };
    let x: Vec<f32> = (0..d_embd)
        .map(|i| ((i as f32 * 0.023).sin() * 0.5).clamp(-1.5, 1.5))
        .collect();

    // Deterministic 31-row KV cache prefill (raw f32 — semantically
    // arbitrary, just needs to be byte-identical between both KV
    // backings so flash_attn reads agree).
    let prefill: Vec<f32> = (0..(n_prefilled as usize) * kv_row)
        .map(|i| (i as f32 * 0.0017 - 0.4).sin() * 0.05)
        .collect();

    // ── New path: encode_first_half (attn-half + post-attn-half).
    let mut state_new = AttnStepState::new(&model, raw_cap);
    state_new.pos = pos;
    state_new.kv_pos[0] = n_prefilled;
    // Pre-fill the persistent KV buffer for layer 0 slots [0..31]
    // with the deterministic prefill bytes (direct memory write).
    let cache_byte_len = (raw_cap as usize) * kv_row * std::mem::size_of::<f32>();
    {
        let buf = disp.kv_buffer_or_alloc(0, cache_byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                prefill.as_ptr(),
                buf.contents() as *mut f32,
                prefill.len(),
            );
        }
    }
    let encoder = SingleBufferEncoder::new(&disp, raw_cap);
    // attn_only variant: stops at after_attn_hc, doesn't invoke MoE
    // (this smoke's build_moe_stub doesn't satisfy production-shape
    // MoE constraints — n_experts=256, loaded QuantizedExpertWeights —
    // those are exercised by the GGUF-gated integration smoke).
    let _ = encoder
        .encode_first_half_attn_only(&x, &model, &mut state_new, LayerCutpoint(1))
        .expect("encode_first_half_attn_only");
    assert_eq!(state_new.kv_pos[0], n_prefilled + 1);
    let new_cur_hc = state_new.cur_hc.clone();

    // ── Reference: compose the same kernels via inherent helpers.
    let initial_cur_hc = cur_hc_from_x(&x, n_hc);

    // 1. hc_collapse_norm → normed + hc_split.
    let scope = disp.batch_scope();
    let prev_hc_b = scope.upload_f32(&initial_cur_hc);
    let hc_fn_b = scope.upload_f32(&w.hc_attn_fn);
    let scale_b = scope.upload_f32(&w.hc_attn_scale);
    let base_b = scope.upload_f32(&w.hc_attn_base);
    let hc_gamma_b = scope.upload_f32(&w.hc_norm_gamma);
    let unit_gamma_b = scope.upload_f32(&vec![1.0f32; hc_dim]);
    let (split_buf, _cur_buf, normed_buf) = scope
        .hc_collapse_norm(
            &prev_hc_b,
            &hc_fn_b,
            &scale_b,
            &base_b,
            &hc_gamma_b,
            n_hc,
            d_embd,
            p.hc_sinkhorn_iter as i32,
            p.hc_eps,
            p.rms_eps,
            &unit_gamma_b,
            false,
        )
        .expect("hc_collapse_norm");
    let normed_b = scope.flush_and_read_multi(&[&split_buf, &normed_buf]);
    let hc_split = &normed_b[0];
    let normed = &normed_b[1];

    // 2. attn_qkv_chain_batched → q_heads + kv_raw_row.
    let (_qr, q_heads, kv_raw_row) = disp
        .attn_qkv_chain_batched(
            normed,
            &w.attn_q_a,
            &w.qkv_gamma_q,
            n_lora_q,
            &w.attn_q_b,
            n_head,
            head_dim,
            p.rms_eps,
            &w.attn_kv,
            kv_row,
        )
        .expect("attn_qkv_chain_batched");

    // 3. kv_norm_rope_chain → kv_normed_rotated.
    let kv_normed_rot = disp
        .kv_norm_rope_chain(&kv_raw_row, &w.qkv_gamma_kv, &p, pos, p.rms_eps)
        .expect("kv_norm_rope_chain");

    // 4. Pre-fill state.kv_storage[0][0..31*row] with the same bytes
    //    the persistent buffer was pre-filled with, then write slot 31
    //    via trait kv_fp8_store (CPU oracle). For n_lora_kv == head_dim
    //    the CPU oracle agrees byte-for-byte with the GPU shim used
    //    by the new path.
    let mut state_ref = AttnStepState::new(&model, raw_cap);
    state_ref.pos = pos;
    state_ref.kv_pos[0] = n_prefilled;
    state_ref.kv_storage[0][..prefill.len()].copy_from_slice(&prefill);
    {
        let mut view = KvCacheView {
            raw: &mut state_ref.kv_storage[0],
            raw_cap,
            pos: state_ref.kv_pos[0],
        };
        AttentionDispatcher::kv_fp8_store(&disp, &p, &kv_normed_rot, &mut view);
        state_ref.kv_pos[0] = view.pos;
    }

    // 5. rope_q forward on q_heads.
    let mut q_heads_rot = q_heads.clone();
    if n_rot > 0 {
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            false,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in q_heads_rot.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }
    }

    // 6. flash_attn_decode (trait — reads kv from state.kv_storage).
    let n_raw = (pos + 1).min(raw_cap);
    let mut heads_back = AttentionDispatcher::flash_attn_decode(
        &disp,
        &p,
        &q_heads_rot,
        &state_ref.kv_storage[0],
        n_raw,
        raw_cap,
        0,
        None,
        0,
        None,
        0,
        &w.attn_sinks,
    );

    // 7. rope_q backward on heads.
    if n_rot > 0 {
        let table = precompute_rope_tail_table(
            n_rot,
            pos,
            true,
            p.rope_freq_base,
            p.rope_freq_scale,
            p.rope_ext_factor,
            p.rope_attn_factor,
            p.rope_orig_ctx,
        );
        for head in heads_back.chunks_mut(head_dim) {
            let tail = &mut head[head_dim - n_rot..];
            apply_rope_tail_with_table(tail, &table, n_rot);
        }
    }

    // 8. attn_output_proj → after_attn_hc.
    let hc_split_post = &hc_split[n_hc..2 * n_hc];
    let hc_split_comb = &hc_split[2 * n_hc..2 * n_hc + n_hc * n_hc];
    let after_attn_hc_ref = AttentionDispatcher::attn_output_proj(
        &disp,
        &p,
        &heads_back,
        &w.w_o_a,
        &w.w_o_b,
        &initial_cur_hc,
        hc_split_post,
        hc_split_comb,
    );

    // ── Bit-close on after_attn_hc (fp8 e4m3 propagation tolerance).
    //
    // The two paths' KV writes to slot 31 differ by the e4m3 round-trip
    // error (CPU oracle vs GPU shim — not byte-identical, per
    // `kv_fp8_store_persistent_smoke`'s e4m3-tolerance bar). That error
    // propagates through flash_attn × attn_output_proj into after_attn_hc.
    // Strict bit-equivalence is gated on M5.2.4 (kv_fp8_store CPU/GPU
    // unification).
    assert_eq!(new_cur_hc.len(), after_attn_hc_ref.len());
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (a, b) in new_cur_hc.iter().zip(&after_attn_hc_ref) {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if b.abs() > max_ref {
            max_ref = b.abs();
        }
    }
    // Absolute bound: e4m3 has 3-bit mantissa → ~13% per-value drift on
    // the single slot 31 write, attenuated by the 1-in-32 slot mass
    // through flash_attn × attn_output_proj. Observed ~1.9e-4 on the
    // synthetic fixture; 2e-3 is the generous algorithmic-correctness
    // bound (catches misordered ops, wrong indices, etc. while
    // permitting the documented kv_fp8 quantization drift).
    assert!(
        max_abs < 2e-3,
        "after_attn_hc divergence too large: max_abs={max_abs:.6} (ref max |b|={max_ref:.6})"
    );
}

// M5.4.5.6: removed `encode_first_half_bails_on_compressor_layer` — both
// hash-routing and compressor/indexer layers are now supported, no
// happy-path bail remains for the synthetic smoke to exercise. The
// remaining bail (`l_split > n_layers`) is structurally improbable
// and not worth a dedicated test. End-to-end correctness on real
// compressor/indexer layers is validated by `encode_first_half_gguf_smoke`.
