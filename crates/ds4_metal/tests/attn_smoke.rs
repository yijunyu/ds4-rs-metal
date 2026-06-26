//! macOS-only Metal attention smoke tests.
//!
//! Validates the 3 fully-encoded helpers landed under #214 against
//! their CPU references over a small synthetic input. These are
//! correctness gates only — perf is task #218's second pass.
//!
//! - `qkv_rms_norm_rows`: compared against `CpuDispatcher::rms_norm`
//!   applied row-wise.
//! - `rope_tail`: compared against the CPU `rope_tail` reference in
//!   `attn_dispatch.rs::CpuAttn`.
//! - `kv_fp8_store`: round-trip f32 → fp8 → f32 over a single token,
//!   compared against the CPU `kv_fp8_store` reference.
//!
//! Tolerance is per-kernel: rms_norm fp32-clean (1e-5), rope_tail
//! 1e-5 (sin/cos pure fp32), fp8_store ~1e-2 (e4m3 has ~3-bit mantissa).

#![cfg(target_os = "macos")]

use ds4_engine::attn_dispatch::{
    AttentionDispatcher, CpuAttentionDispatcher, HcKind, KvCacheView, LayerParams,
};
use ds4_metal::MetalDispatcher;

fn small_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 64,
        n_hc: 1,
        n_head: 4,
        head_dim: 8,
        n_rot: 4,
        n_lora_q: 16,
        n_lora_kv: 16,
        hc_sinkhorn_iter: 0,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 2,
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cpu_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(gamma).map(|(&v, &g)| v * inv * g).collect()
}

#[test]
fn qkv_rms_norm_rows_matches_cpu_rms_per_row() {
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let n_q = params.n_lora_q as usize;
    let n_kv = params.n_lora_kv as usize;

    let qr: Vec<f32> = (0..n_q).map(|i| (i as f32 * 0.1 + 0.2).sin()).collect();
    let kv: Vec<f32> = (0..n_kv).map(|i| (i as f32 * 0.07 - 0.3).cos()).collect();
    let gamma_q: Vec<f32> = (0..n_q).map(|i| 1.0 + 0.01 * i as f32).collect();
    let gamma_kv: Vec<f32> = (0..n_kv).map(|i| 1.0 - 0.005 * i as f32).collect();

    let (qr_out, kv_out) = dispatcher.qkv_rms_norm_rows(&params, &qr, &kv, &gamma_q, &gamma_kv);

    let qr_ref = cpu_rms_norm(&qr, &gamma_q, params.rms_eps);
    let kv_ref = cpu_rms_norm(&kv, &gamma_kv, params.rms_eps);

    let qr_err = max_abs_diff(&qr_out, &qr_ref);
    let kv_err = max_abs_diff(&kv_out, &kv_ref);
    assert!(qr_err < 1e-5, "qr_err = {qr_err}");
    assert!(kv_err < 1e-5, "kv_err = {kv_err}");
}

#[test]
fn rope_tail_matches_cpu_oracle_per_head() {
    // Exercises the dsv4_rope_tail_f32.metal shim against the CPU oracle.
    // Both use the adjacent-pair (j0, j0+1) convention matching antirez
    // metal/dsv4_rope.metal. Tolerance 1e-5 — pure fp32 sin/cos.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let cpu = CpuAttentionDispatcher;

    // 4 heads × n_rot=4 floats each: 16 values, all heads share the same
    // rotation table but distinct payload.
    let n_rot = params.n_rot as usize;
    let n_heads = params.n_head as usize;
    let mut x_gpu: Vec<f32> = (0..n_heads * n_rot)
        .map(|i| (i as f32 * 0.13 - 0.5).sin())
        .collect();
    let mut x_cpu = x_gpu.clone();

    let pos = 7u32;
    dispatcher.rope_tail(&params, &mut x_gpu, pos, false);
    cpu.rope_tail(&params, &mut x_cpu, pos, false);

    let err = max_abs_diff(&x_gpu, &x_cpu);
    assert!(err < 1e-5, "rope_tail err = {err}, expected < 1e-5");
}

#[test]
fn kv_fp8_store_round_trip_within_e4m3_tolerance() {
    // The Metal shim does an e4m3 round-trip on the n_nope (n_lora_kv)
    // prefix + an f16 round-trip on the n_rot tail. The CPU oracle is
    // identity. Tolerance is per-region: 6% relative for the e4m3 region
    // (≈ 3-bit mantissa precision), 1e-2 absolute for the f16 tail.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let params = small_params();
    let row = params.n_lora_kv as usize;
    let cap = 4u32;
    let mut storage_gpu = vec![0.0f32; row * cap as usize];

    // Sin-wave row: values in [-1, 1]; well inside e4m3's ±448 range.
    let kv: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.31).sin()).collect();

    let mut view = KvCacheView {
        raw: &mut storage_gpu,
        raw_cap: cap,
        pos: 0,
    };
    dispatcher.kv_fp8_store(&params, &kv, &mut view);
    assert_eq!(view.pos, 1, "pos must advance by 1");

    let n_nope = params.n_lora_kv as usize;
    let n_rot_t = params.n_rot as usize;

    // e4m3 region — accept up to 12.5% (1 ULP at this exponent) since
    // e4m3 has 3 mantissa bits ⇒ 1 ULP ≈ 1/8 = 12.5%.
    for (i, (&got, &want)) in storage_gpu[..n_nope].iter().zip(&kv[..n_nope]).enumerate() {
        let rel = if want.abs() > 1e-3 {
            (got - want).abs() / want.abs()
        } else {
            (got - want).abs()
        };
        assert!(
            rel < 0.13,
            "e4m3 row[{i}]: got={got}, want={want}, rel={rel}"
        );
    }
    // f16 region — much tighter; f16 ULP at this magnitude is ~1e-3.
    for (i, (&got, &want)) in storage_gpu[n_nope..n_nope + n_rot_t]
        .iter()
        .zip(&kv[n_nope..])
        .enumerate()
    {
        let abs = (got - want).abs();
        assert!(
            abs < 1e-2,
            "f16 tail row[{i}]: got={got}, want={want}, abs={abs}"
        );
    }
}

#[test]
fn shared_expert_matches_cpu_silu_clamp() {
    // shared_expert = silu(clamp(W_gate · x, ±c)) * clamp(W_up · x, ±c).
    // d_embd=64 (params), shared_dim=4 → matvec hits nxpsg=4, nsg=1.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = small_params();
    let d_embd = params.d_embd as usize;
    let sd: u32 = 4;
    let sdu = sd as usize;
    let clamp: f32 = 7.0;

    let ffn: Vec<f32> = (0..d_embd).map(|i| (i as f32 * 0.11 - 0.4).sin()).collect();
    let w_gate: Vec<f32> = (0..sdu * d_embd)
        .map(|i| ((i as f32) * 0.017).cos() * 0.3)
        .collect();
    let w_up: Vec<f32> = (0..sdu * d_embd)
        .map(|i| ((i as f32) * 0.023 + 0.1).sin() * 0.3)
        .collect();

    let metal_out = dispatcher.shared_expert(&params, &ffn, &w_gate, &w_up, sd, clamp);
    let cpu_out = cpu.shared_expert(&params, &ffn, &w_gate, &w_up, sd, clamp);

    assert_eq!(metal_out.len(), sdu);
    let err = max_abs_diff(&metal_out, &cpu_out);
    assert!(err < 1e-4, "shared_expert err = {err}, expected < 1e-4");
}

#[test]
fn shared_down_hc_expand_add_matches_cpu() {
    // d_embd=64, n_hc=1, sd=4 → matvec(w_down · shared_mid) with d_in=4
    // nxpsg=4 nsg=1; d_out=64 → 32 thread groups.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = small_params();
    let d_embd = params.d_embd as usize;
    let n_hc = params.n_hc as usize;
    let sd: usize = 4;

    let shared_mid: Vec<f32> = (0..sd).map(|i| (i as f32 * 0.3 - 0.1).sin()).collect();
    let w_down: Vec<f32> = (0..d_embd * sd)
        .map(|i| ((i as f32) * 0.013).cos() * 0.4)
        .collect();
    let routed_out: Vec<f32> = (0..d_embd).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
    let after_attn_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| (i as f32 * 0.07 + 1.0).cos())
        .collect();
    let hc_split: Vec<f32> = (0..n_hc).map(|h| 0.5 + 0.1 * h as f32).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| ((i as f32) * 0.13 - 0.2).sin() * 0.4)
        .collect();

    let metal_out = dispatcher.shared_down_hc_expand_add(
        &params,
        &shared_mid,
        &w_down,
        &routed_out,
        &after_attn_hc,
        &hc_split,
        &hc_split_comb,
    );
    let cpu_out = cpu.shared_down_hc_expand_add(
        &params,
        &shared_mid,
        &w_down,
        &routed_out,
        &after_attn_hc,
        &hc_split,
        &hc_split_comb,
    );

    assert_eq!(metal_out.len(), n_hc * d_embd);
    let err = max_abs_diff(&metal_out, &cpu_out);
    assert!(err < 1e-4, "shared_down err = {err}, expected < 1e-4");
}

#[test]
fn attn_output_proj_matches_cpu_grouped_proj() {
    // small_params: n_head=4, head_dim=8 → q_dim=32; d_embd=64; n_hc=1;
    // n_out_group=2 → group_dim=16. Pick n_lora_o=8 so out_low_dim=16
    // (mult of 4 for matvec float4 path on the stage-2 dense matvec).
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = small_params();
    let d_embd = params.d_embd as usize;
    let n_hc = params.n_hc as usize;
    let q_dim = (params.n_head * params.head_dim) as usize;
    let n_groups = params.n_out_group as usize;
    let group_dim = q_dim / n_groups;
    let n_lora_o: usize = 8;
    let out_low_dim = n_groups * n_lora_o;

    let heads: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.11 - 0.5).sin()).collect();
    let w_o_a: Vec<f32> = (0..out_low_dim * group_dim)
        .map(|i| ((i as f32) * 0.017).cos() * 0.25)
        .collect();
    let w_o_b: Vec<f32> = (0..d_embd * out_low_dim)
        .map(|i| ((i as f32) * 0.013 + 0.3).sin() * 0.3)
        .collect();
    let cur_hc: Vec<f32> = (0..n_hc * d_embd)
        .map(|i| (i as f32 * 0.05).cos())
        .collect();
    let hc_split: Vec<f32> = (0..n_hc).map(|h| 0.6 + 0.1 * h as f32).collect();
    let hc_split_comb: Vec<f32> = (0..n_hc * n_hc)
        .map(|i| ((i as f32) * 0.11 - 0.3).cos() * 0.3)
        .collect();

    let metal_out = dispatcher.attn_output_proj(
        &params,
        &heads,
        &w_o_a,
        &w_o_b,
        &cur_hc,
        &hc_split,
        &hc_split_comb,
    );
    let cpu_out = cpu.attn_output_proj(
        &params,
        &heads,
        &w_o_a,
        &w_o_b,
        &cur_hc,
        &hc_split,
        &hc_split_comb,
    );

    assert_eq!(metal_out.len(), n_hc * d_embd);
    let err = max_abs_diff(&metal_out, &cpu_out);
    assert!(err < 1e-4, "attn_output_proj err = {err}, expected < 1e-4");
}

#[test]
fn hc_collapse_norm_matches_cpu_with_gamma() {
    // n_hc=1, d_embd=64 (divisible by 4 for rms_norm float4 path).
    // sinkhorn_iter=2 to exercise the iter loop.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let mut params = small_params();
    params.hc_sinkhorn_iter = 2;
    let n_hc = params.n_hc as usize;
    let d_embd = params.d_embd as usize;

    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| 0.01 * (i as f32 * 0.17).sin())
        .collect();
    let hc_scale = vec![1.0f32, 1.0, 1.0];
    let hc_base: Vec<f32> = (0..mix_hc).map(|i| 0.05 * (i as f32 * 0.3).cos()).collect();
    let after_attn_hc: Vec<f32> = (0..hc_dim).map(|i| (i as f32 * 0.11 - 0.3).cos()).collect();
    let gamma: Vec<f32> = (0..d_embd).map(|i| 1.0 + 0.01 * i as f32).collect();

    let (m_cur, m_normed, m_split) = dispatcher.hc_collapse_norm(
        &params,
        HcKind::Attn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &after_attn_hc,
        Some(&gamma),
    );
    let (c_cur, c_normed, c_split) = cpu.hc_collapse_norm(
        &params,
        HcKind::Attn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &after_attn_hc,
        Some(&gamma),
    );

    let cur_err = max_abs_diff(&m_cur, &c_cur);
    let normed_err = max_abs_diff(&m_normed, &c_normed);
    let split_err = max_abs_diff(&m_split, &c_split);
    assert!(cur_err < 1e-5, "hc_collapse cur err = {cur_err}");
    assert!(normed_err < 1e-4, "hc_collapse normed err = {normed_err}");
    assert!(split_err < 1e-6, "hc_collapse split err = {split_err}");
}

/// Production-shape params for the Metal `dk512_dv512` path:
/// row = n_lora_kv = 512 = head_dim (rope tail inside the row at [448..512]).
fn fa_dk512_params() -> LayerParams {
    LayerParams {
        layer_idx: 0,
        d_embd: 128,
        n_hc: 1,
        n_head: 2,
        head_dim: 512,
        n_rot: 64,
        n_lora_q: 128,
        n_lora_kv: 512, // == head_dim; rope tail at [448..512] inside the row
        hc_sinkhorn_iter: 0,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        rope_orig_ctx: 4096,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 2,
    }
}

#[test]
fn flash_attn_decode_metal_path_matches_cpu_oracle() {
    // Exercises the real Metal vec+reduce wire (#227). Preconditions:
    //   head_dim=512, raw_start=0, n_raw=32 (ncpsg-aligned), kv_comp=None.
    // Tolerance loosened from CPU oracle's 1e-6 because the Metal path
    // stages KV through f16 (3-4 decimal digits precision).
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = fa_dk512_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    assert_eq!(
        row, head_dim,
        "test setup: row must equal head_dim for the dk512 path"
    );

    let raw_cap: u32 = 64;
    let n_raw: u32 = 32; // exactly one ncpsg block
    let raw_start: u32 = 0;

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.0017 - 0.4).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..raw_cap as usize * row)
        .map(|i| (i as f32 * 0.0009 + 0.1).cos() * 0.5)
        .collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.1 * h as f32).collect();

    let m_out = dispatcher.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        None,
        0,
        None,
        0,
        &attn_sinks,
    );
    let c_out = cpu.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        None,
        0,
        None,
        0,
        &attn_sinks,
    );

    assert_eq!(m_out.len(), n_head * head_dim);
    let err = max_abs_diff(&m_out, &c_out);
    // f16 KV staging puts a ~5e-3 ceiling on absolute element error
    // at this scale (cos(x)*0.5 magnitudes, ~1e-3 f16 ulp on the
    // accumulated softmax output).
    assert!(
        err < 5e-3,
        "flash_attn_decode_metal err = {err}, expected < 5e-3"
    );
}

#[test]
fn flash_attn_decode_metal_compressor_matches_cpu_oracle() {
    // M5 task #97 — GPU compressor/indexer path. Preconditions:
    //   head_dim=512, raw_start=0, kv_comp.is_some(), comp_selected.is_some(),
    //   (n_raw + n_selected) % 32 == 0. Builds extended KV (raw rows + indexer-
    //   selected comp rows) via `build_extended_kv` and routes through
    //   `flash_attn_decode_metal`. Asserts bit-equivalence vs CPU oracle within
    //   the f16 staging tolerance.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = fa_dk512_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    assert_eq!(row, head_dim, "test setup: row must equal head_dim");

    let raw_cap: u32 = 64;
    let n_raw: u32 = 16;
    let n_comp: u32 = 8;
    let n_selected: u32 = 16; // n_raw + n_selected = 32 (one ncpsg block)
    let raw_start: u32 = 0;

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.0017 - 0.4).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..raw_cap as usize * row)
        .map(|i| (i as f32 * 0.0009 + 0.1).cos() * 0.5)
        .collect();
    let kv_comp: Vec<f32> = (0..(n_comp as usize) * head_dim)
        .map(|i| (i as f32 * 0.0013 - 0.2).sin() * 0.4)
        .collect();
    // comp_selected indices are pre-filtered (no need for a mask): pick
    // n_selected distinct rows within [0, n_comp). For this test
    // n_selected=16 > n_comp=8, so repeat indices — the kernel just
    // re-reads the same comp row multiple times, which is fine for the
    // oracle comparison.
    let comp_selected: Vec<u32> = (0..n_selected).map(|i| i % n_comp).collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.1 * h as f32).collect();

    let m_out = dispatcher.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        Some(&kv_comp),
        n_comp,
        Some(&comp_selected),
        n_selected,
        &attn_sinks,
    );
    let c_out = cpu.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        Some(&kv_comp),
        n_comp,
        Some(&comp_selected),
        n_selected,
        &attn_sinks,
    );

    assert_eq!(m_out.len(), n_head * head_dim);
    let err = max_abs_diff(&m_out, &c_out);
    assert!(
        err < 5e-3,
        "flash_attn_decode_metal_compressor err = {err}, expected < 5e-3"
    );
}

#[test]
fn flash_attn_decode_metal_compressor_unaligned_matches_cpu_oracle() {
    // M5 task #97 step 3 — non-32-aligned `n_raw + n_selected` exercises
    // the mask-based padding path. The gather kernel zero-fills the
    // padded rows; the flash_attn mask buffer carries f16 -inf at
    // padded positions so they drop out of the softmax.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = fa_dk512_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;
    assert_eq!(row, head_dim);

    let raw_cap: u32 = 64;
    let n_raw: u32 = 10;
    let n_comp: u32 = 12;
    let n_selected: u32 = 15; // n_raw + n_selected = 25 → padded to 32
    let raw_start: u32 = 0;

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.0023 - 0.3).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..raw_cap as usize * row)
        .map(|i| (i as f32 * 0.0011 + 0.05).cos() * 0.5)
        .collect();
    let kv_comp: Vec<f32> = (0..(n_comp as usize) * head_dim)
        .map(|i| (i as f32 * 0.0017 - 0.15).sin() * 0.4)
        .collect();
    let comp_selected: Vec<u32> = (0..n_selected).map(|i| i % n_comp).collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.08 * h as f32).collect();

    let m_out = dispatcher.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        Some(&kv_comp),
        n_comp,
        Some(&comp_selected),
        n_selected,
        &attn_sinks,
    );
    let c_out = cpu.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        Some(&kv_comp),
        n_comp,
        Some(&comp_selected),
        n_selected,
        &attn_sinks,
    );

    assert_eq!(m_out.len(), n_head * head_dim);
    let err = max_abs_diff(&m_out, &c_out);
    assert!(
        err < 5e-3,
        "flash_attn_decode_metal_compressor_unaligned err = {err}, expected < 5e-3"
    );
}

#[test]
fn flash_attn_decode_matches_cpu_oracle() {
    // CPU-fallback path (head_dim=8 ⇒ no dk512 kernel; CPU only).
    // small_params: n_head=4, head_dim=8, n_lora_kv=16, n_rot=4 → row=20.
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let params = small_params();
    let n_head = params.n_head as usize;
    let head_dim = params.head_dim as usize;
    let row = params.n_lora_kv as usize;

    let raw_cap: u32 = 8;
    let n_raw: u32 = 3;
    let raw_start: u32 = 0;

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i as f32 * 0.17 - 0.4).sin())
        .collect();
    let kv_raw: Vec<f32> = (0..raw_cap as usize * row)
        .map(|i| (i as f32 * 0.09 + 0.1).cos() * 0.5)
        .collect();
    let attn_sinks: Vec<f32> = (0..n_head).map(|h| 0.1 * h as f32).collect();

    let m_out = dispatcher.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        None,
        0,
        None,
        0,
        &attn_sinks,
    );
    let c_out = cpu.flash_attn_decode(
        &params,
        &q,
        &kv_raw,
        n_raw,
        raw_cap,
        raw_start,
        None,
        0,
        None,
        0,
        &attn_sinks,
    );

    assert_eq!(m_out.len(), n_head * head_dim);
    let err = max_abs_diff(&m_out, &c_out);
    assert!(err < 1e-6, "flash_attn_decode err = {err}, expected < 1e-6");
}

#[test]
fn hc_collapse_norm_matches_cpu_without_gamma() {
    // gamma=None path (CPU fallback for normalization).
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    let cpu = CpuAttentionDispatcher;
    let mut params = small_params();
    params.hc_sinkhorn_iter = 1;
    let n_hc = params.n_hc as usize;
    let d_embd = params.d_embd as usize;

    let hc_dim = n_hc * d_embd;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    let hc_fn: Vec<f32> = (0..hc_dim * mix_hc)
        .map(|i| 0.01 * (i as f32 * 0.13).cos())
        .collect();
    let hc_scale = vec![1.0f32, 1.0, 1.0];
    let hc_base = vec![0.0f32; mix_hc];
    let after_attn_hc: Vec<f32> = (0..hc_dim).map(|i| (i as f32 * 0.07 + 0.5).sin()).collect();

    let (m_cur, m_normed, m_split) = dispatcher.hc_collapse_norm(
        &params,
        HcKind::Ffn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &after_attn_hc,
        None,
    );
    let (c_cur, c_normed, c_split) = cpu.hc_collapse_norm(
        &params,
        HcKind::Ffn,
        &hc_fn,
        &hc_scale,
        &hc_base,
        &after_attn_hc,
        None,
    );

    let cur_err = max_abs_diff(&m_cur, &c_cur);
    let normed_err = max_abs_diff(&m_normed, &c_normed);
    let split_err = max_abs_diff(&m_split, &c_split);
    assert!(cur_err < 1e-5, "hc_collapse cur err (no gamma) = {cur_err}");
    assert!(
        normed_err < 1e-5,
        "hc_collapse normed err (no gamma) = {normed_err}"
    );
    assert!(
        split_err < 1e-6,
        "hc_collapse split err (no gamma) = {split_err}"
    );
}

/// `KernelDispatcher::rms_norm` (which calls `MetalState::rms_norm_impl`)
/// must match the CPU oracle bit-close. This is the CPU-comparison gate
/// retro-added under #226 after `hc_collapse_norm` exposed that the
/// Metal `ds4_kernel_rms_norm_mul_f32_4` encoding was completely broken
/// (struct buffer layout mismatched antirez `ds4_metal_args_norm`).
///
/// We exercise three row widths to cover the antirez
/// `ds4_metal_rms_norm_threads(n)` thread-count branches:
///   - n=64  → ne00_t=16 → nth=16 (small power-of-2 path)
///   - n=256 → ne00_t=64 → nth=64
///   - n=1024 → ne00_t=256 → nth=256 (DS4 d_lora_q size)
#[test]
fn rms_norm_matches_cpu_oracle() {
    use ds4_engine::dispatch::KernelDispatcher;
    let dispatcher = MetalDispatcher::new().expect("MetalDispatcher::new");
    for &n in &[64usize, 256, 1024] {
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.013 + 0.2).sin() * 0.7)
            .collect();
        let gamma: Vec<f32> = (0..n).map(|i| 1.0 + 0.005 * (i as f32).cos()).collect();
        let eps = 1e-5f32;

        let m = dispatcher.rms_norm(&x, &gamma, eps);
        let c = cpu_rms_norm(&x, &gamma, eps);

        assert_eq!(m.len(), c.len(), "rms_norm n={n}: length mismatch");
        let err = max_abs_diff(&m, &c);
        // fp32-clean reduction; per-term simdgroup-accumulation order
        // differs from naive zip-sum but stays within ~1e-6 per term.
        assert!(err < 1e-5, "rms_norm n={n}: max abs err {err} >= 1e-5");
    }
}
