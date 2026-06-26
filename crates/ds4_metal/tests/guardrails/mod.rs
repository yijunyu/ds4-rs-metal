//! KERNEL GUARDRAILS — a small, synthetic-input (no-model, CI-able) unit-test
//! harness distilling the failure classes that cost a ~17-run model-level
//! bisection of the chunk-prefill "@3000 generic content" gap. Each guardrail
//! catches one class CHEAPLY (seconds, random data, Metal-only — no 86 GB GGUF),
//! so a regression is caught at the kernel, not 40 layers downstream.
//!
//! Include from a test binary with `mod guardrails;` then `use guardrails::*;`.
//!
//! ## The guardrails (and the lesson each encodes)
//!
//! 1. `assert_deterministic` — RUN-TO-RUN BIT-IDENTITY. A kernel that reads
//!    uninitialized scratch / races a shared buffer / depends on cb-ordering
//!    produces different bytes across runs. Lesson: the intermittent all-BOS
//!    output looked like a "race" until measured — measure it directly.
//!
//! 2. `assert_k_batch_equiv` — K-BATCHED == K=1-LOOPED. THE root-cause class:
//!    a K-batched kernel (chunk prefill) must produce, for each row r, what the
//!    K=1 kernel produces for that row alone. `Exact` for gather/elementwise/
//!    non-reduction ops (bit-identical REQUIRED); `Close{rel}` for reductions/
//!    matmuls (fp32 non-associativity makes byte-identity unachievable — assert
//!    closeness, NOT equality). On failure it prints the divergence SIGNATURE so
//!    you instantly know broad-drift vs sparse-flip.
//!
//! 3. `assert_close_to_oracle` — vs a CPU reference. Generalises the existing
//!    `*_smoke` pattern + attaches the divergence classifier.
//!
//! 4. `assert_selection_matches` / `selection_sensitivity` — DISCRETE-SELECTION
//!    guardrail (top-k / argmax / routing). Discrete ops AMPLIFY tiny upstream
//!    drift discontinuously — a ~1e-6 logit delta flips an expert and the FFN
//!    output changes by O(1). Assert the selection kernel is deterministic +
//!    matches CPU exactly, and quantify how many selections flip under an ε
//!    perturbation (the amplification risk the bisection traced to incoherence).
//!
//! 5. `bench_bandwidth` / `assert_bandwidth_frac` — EFFICIENCY. A correct kernel
//!    at the wrong dispatch params (nsg/nr0) runs at a fraction of roofline.
//!    Microbench GB/s vs device peak; flag regressions.
//!
//! 6. Precision helpers `fp8_snap` / `f16_round` — MATCH-THE-REFERENCE-PRECISION.
//!    Lesson: making a kernel MORE accurate than the path the model is calibrated
//!    for (f32 KV where decode reads fp8/f16) is a fidelity BUG, not a fix. Build
//!    oracles at the reference precision, not idealised f32.
//!
//! ## Wrapping a suspect kernel (the recipe)
//! 1. Has a K-batched variant? → `assert_k_batch_equiv` vs the K=1 kernel.
//!    `Exact` if it's gather/elementwise/non-reduction; `Close` if it reduces.
//! 2. Always → `assert_deterministic` (cheap; catches races/uninit scratch).
//! 3. Has a CPU reference? → `assert_close_to_oracle` (oracle at REFERENCE
//!    precision via `fp8_snap`/`f16_round`, not idealised f32).
//! 4. Produces a discrete pick (top-k/argmax/route)? → `assert_selection_matches`
//!    + report `selection_sensitivity` (flags it as an amplifier suspect).
//! 5. Perf-sensitive? → `bench_bandwidth` (hoist setup out of the loop for an
//!    absolute floor; else use it as a relative-regression signal).
//!
//! ## Suspect kernels to wrap (from the chunk-prefill bisection)
//! DONE:
//!   • `matmul_k_q8_0` (K-batch) — K≡K=1 rel 4e-4 ✓.
//!   • `encode_indexer_topk_batched` (selection) — CPU-exact ✓; ε flips 7%.
//!   • `flash_attn_k_mla_comp_masked` (noidx chunk flash) — K≡K=1 BIT-IDENTICAL
//!     + oracle-close 1.8e-4 at f16 reference precision. ⇒ this kernel is CLEAN /
//!     K-stable (NOT the chunk broad-drift source) — matches the bisection's
//!     f16-fallback no-op. (An f32 oracle would have shown a spurious ~1e-2 f16
//!     gap; build oracles at the kernel's precision.)
//!   • `hc_collapse_norm_k` (FFN-half glue / router input) — K-batch(fused)≡K=1
//!     rel 4e-7 ✓ (the kernel's "bit-identical" comment overclaims, but it's
//!     fp32-epsilon — CLEAN, not the broad-drift source).
//!
//! ★ CONCLUSION from these guardrails: EVERY individual K-batched kernel the
//! chunk path uses is faithful to K=1 (matmul 4e-4, flash 0.0, glue 4e-7). So the
//! cos-0.34 broad drift is NOT a K-batch artifact of any single kernel — it's a
//! CROSS-KERNEL mismatch: the chunk path and the per-token path run DIFFERENT
//! attention implementations (chunk noidx `flash_attn_k_mla_comp_masked` f32-raw
//! vs per-token `flash_attn_decode_persistent_compressor` / build_extended_kv
//! f16-workspace; + fp8→f16 vs f32→f16 raw KV). The needed guardrail is a NEW
//! category — CROSS-IMPLEMENTATION equivalence (kernel A ≡ kernel B on identical
//! inputs at matched precision), not K-batch≡K=1.
//!   • cross-impl DONE (kernel_guardrails_crossimpl.rs): `flash_attn_k_mla_comp_masked`
//!     (f32-raw comp-kernel) ≡ `build_chunk_kv_workspace`+`flash_attn_decode_k`
//!     (f16-workspace, per-token family) on identical raw/comp/q → rel_L2 1.2e-7,
//!     cos 1.0. The two flash impls are EQUIVALENT ⇒ the production divergence is
//!     NOT the kernel, it's the INPUTS: which raw rows + raw-KV PRECISION. (At idx
//!     layers the chunk feeds f32 `kv_normed` raw vs the per-token fp8-snapped ring
//!     — same kernel math, different raw precision.)
//!   • router DONE (kernel_guardrails_router.rs): `encode_router_logits_k`
//!     K≡K=1 BIT-IDENTICAL + `encode_router_finalize` CPU-exact ⇒ router kernel
//!     faithful (flips come from INPUT drift, not the kernel). Sensitivity sweep:
//!     onset ε≈3e-2 (a ~3% input drift flips 1/3 of top-6 experts, plateau 50%) —
//!     quantifies the discrete amplifier that maps to the bisection's depth-growing
//!     flip%. The whole kernel layer is now guardrailed & cleared.
//! TODO (input-side, needs model-level or input-assembly probes, NOT kernel tests):
//!   • the FLASH INPUTS the chunk vs per-token paths assemble (which raw window
//!     rows + comp set + raw-KV precision) — the cross-impl showed the kernels
//!     agree, so the divergence is here. Best pinned by per-layer residual capture.
#![allow(dead_code)]

// ── metrics ────────────────────────────────────────────────────────────────

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Per-ROW divergence signature: treat a/b as `[n_rows, row_dim]`. Returns
/// (flip_frac = rows with cos<0.5, pert_frac = rows with cos<0.99, min_row_cos).
/// SPARSE (small flip_frac, low min) ⇒ discrete per-row events (selection/routing
/// flips). BROAD (pert_frac≈1, no flips) ⇒ uniform numeric drift (matmul/precision).
pub fn row_divergence(a: &[f32], b: &[f32], row_dim: usize) -> (f64, f64, f64) {
    if row_dim == 0 || a.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = a.len() / row_dim;
    let (mut flip, mut pert, mut min_c) = (0usize, 0usize, 1.0f64);
    for r in 0..n {
        let c = cosine(&a[r * row_dim..(r + 1) * row_dim], &b[r * row_dim..(r + 1) * row_dim]);
        if c < 0.5 { flip += 1; }
        if c < 0.99 { pert += 1; }
        if c < min_c { min_c = c; }
    }
    (flip as f64 / n as f64, pert as f64 / n as f64, min_c)
}

fn signature(a: &[f32], b: &[f32], row_dim: usize) -> String {
    let (flip, pert, minc) = row_divergence(a, b, row_dim);
    let kind = if flip > 0.02 {
        "SPARSE/FLIP (discrete per-row events — selection/routing-class)"
    } else if pert > 0.5 {
        "BROAD (uniform numeric drift — matmul/precision-class)"
    } else {
        "minor"
    };
    format!(
        "rel_L2={:.3e} maxΔ={:.3e} cos={:.6} | rows: flip%={:.1} pert%={:.1} minRowCos={:.4} ⇒ {kind}",
        rel_l2(a, b), max_abs_diff(a, b), cosine(a, b), flip * 100.0, pert * 100.0, minc,
    )
}

// ── precision helpers (build oracles at the REFERENCE precision) ─────────────

/// Round f32 → IEEE half → f32 (what an f16-KV path stores). Self-contained
/// (no `half` dep); round-to-nearest-even on the mantissa, flush tiny subnormals.
pub fn f16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let half: u16 = if exp >= 0x1f {
        sign | 0x7c00 // inf/overflow
    } else if exp <= 0 {
        sign // flush subnormal → ±0
    } else {
        let mant10 = (bits >> 13) & 0x3ff;
        let round = (bits >> 12) & 1; // guard bit → round-half-up (approx RNE)
        let mut h = sign | ((exp as u16) << 10) | mant10 as u16;
        h += round as u16;
        h
    };
    // half → f32
    let s = ((half & 0x8000) as u32) << 16;
    let e = ((half >> 10) & 0x1f) as u32;
    let m = (half & 0x3ff) as u32;
    let f = if e == 0 {
        if m == 0 { s } else {
            // subnormal half → normalized f32
            let mut e2 = -1i32;
            let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            m2 &= 0x3ff;
            s | (((e2 + 127 - 15) as u32) << 23) | (m2 << 13)
        }
    } else if e == 0x1f {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(f)
}

/// FP8 E4M3-ish snap: matches the magnitude of `ds4_dsv4_kv_fp8_store`'s ~6%
/// quantization. (Approximate — use for sensitivity bounds, not bit-exact refs.)
pub fn fp8_snap(x: f32) -> f32 {
    if x == 0.0 { return 0.0; }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mant = bits & 0x007f_ffff;
    // keep top 3 mantissa bits (E4M3 has 3), round-to-nearest.
    let kept = (mant + 0x0008_0000) & 0x0070_0000;
    f32::from_bits(sign | (bits & 0x7f80_0000) | kept)
}

// ── guardrails ───────────────────────────────────────────────────────────────

/// (1) Determinism: run `f` `runs` times; every run must be byte-identical.
/// Catches races / uninitialized scratch / cb-ordering nondeterminism.
pub fn assert_deterministic<F: FnMut() -> Vec<f32>>(label: &str, runs: usize, mut f: F) {
    let base = f();
    assert!(base.iter().all(|v| v.is_finite()), "{label}: run 0 has non-finite values");
    for i in 1..runs {
        let r = f();
        assert_eq!(r.len(), base.len(), "{label}: run {i} length differs");
        if r != base {
            let (_, pert, minc) = row_divergence(&base, &r, base.len().max(1));
            panic!(
                "{label}: NON-DETERMINISTIC across runs (run {i} ≠ run 0): {} \
                 [pert%={:.1} minCos={:.4}] — a race / uninitialized scratch / cb-order dep.",
                signature(&base, &r, base.len().max(1)), pert * 100.0, minc,
            );
        }
    }
}

#[derive(Clone, Copy)]
pub enum Equiv {
    /// Bit-identical required (gather / elementwise / non-reduction kernels).
    Exact,
    /// Closeness only (reductions / matmuls — fp32 non-associativity).
    Close { rel_tol: f64 },
}

/// (2) K-batch equivalence: the batched output's row `r` must match the K=1
/// kernel run on row `r` alone. `batched` returns `[k*row_dim]`; `per_row(r)`
/// returns `[row_dim]`. Samples a spread of rows (or all, if k small).
pub fn assert_k_batch_equiv<B, P>(
    label: &str, k: usize, row_dim: usize, mode: Equiv, batched: B, per_row: P,
) where
    B: FnOnce() -> Vec<f32>,
    P: Fn(usize) -> Vec<f32>,
{
    let out = batched();
    assert_eq!(out.len(), k * row_dim, "{label}: batched output shape");
    assert!(out.iter().all(|v| v.is_finite()), "{label}: batched output has non-finite");
    let rows: Vec<usize> = if k <= 8 {
        (0..k).collect()
    } else {
        vec![0, 1, k / 3, k / 2, (2 * k) / 3, k - 1]
    };
    let mut worst_rel = 0f64;
    for &r in &rows {
        let want = per_row(r);
        assert_eq!(want.len(), row_dim, "{label}: per_row({r}) shape");
        let got = &out[r * row_dim..(r + 1) * row_dim];
        match mode {
            Equiv::Exact => {
                if got != want.as_slice() {
                    panic!(
                        "{label}: K-batched row {r} NOT bit-identical to K=1 (Exact mode) — \
                         {}. A non-reduction kernel must be exact; the K-batch path differs.",
                        signature(got, &want, row_dim),
                    );
                }
            }
            Equiv::Close { rel_tol } => {
                let rl = rel_l2(got, &want);
                worst_rel = worst_rel.max(rl);
                assert!(
                    rl < rel_tol,
                    "{label}: K-batched row {r} diverges from K=1 beyond rel_tol={rel_tol:.1e} — {}",
                    signature(got, &want, row_dim),
                );
            }
        }
    }
    eprintln!("[guardrail] {label}: K-batch≡K=1 OK (k={k}, worst_rel={worst_rel:.2e})");
}

/// (3) Closeness vs a CPU oracle (build the oracle at the REFERENCE precision —
/// see `f16_round`/`fp8_snap`). Reports the divergence signature on failure.
pub fn assert_close_to_oracle(label: &str, gpu: &[f32], oracle: &[f32], rel_tol: f64) {
    assert_eq!(gpu.len(), oracle.len(), "{label}: oracle shape mismatch");
    assert!(gpu.iter().all(|v| v.is_finite()), "{label}: gpu output has non-finite");
    let rl = rel_l2(gpu, oracle);
    assert!(
        rl < rel_tol,
        "{label}: GPU diverges from oracle beyond rel_tol={rel_tol:.1e} — {}",
        signature(gpu, oracle, oracle.len().max(1)),
    );
    eprintln!("[guardrail] {label}: oracle-close OK (rel_L2={rl:.2e} < {rel_tol:.1e})");
}

/// (4a) Discrete selection must match a reference exactly (top-k / argmax /
/// routing). The reference is usually a CPU top-k over the SAME logits.
pub fn assert_selection_matches(label: &str, got: &[i32], want: &[i32]) {
    assert_eq!(
        got, want,
        "{label}: discrete SELECTION differs from reference — got {got:?} want {want:?}. \
         A selection kernel must match CPU exactly; a mismatch here flips a downstream \
         expert/token and amplifies discontinuously."
    );
    eprintln!("[guardrail] {label}: selection matches reference OK ({} ids)", got.len());
}

/// (4b) Selection sensitivity DIAGNOSTIC (not an assertion): how many of `base`
/// selections flip when logits are perturbed by `eps`. High sensitivity = this
/// op will amplify upstream numeric drift into divergence; choose it as a prime
/// suspect when a downstream-coherence bug appears. Returns flipped-fraction.
pub fn selection_sensitivity(base: &[i32], perturbed: &[i32]) -> f64 {
    let n = base.len().min(perturbed.len()).max(1);
    let diff = base.iter().zip(perturbed).filter(|(a, b)| a != b).count();
    diff as f64 / n as f64
}

// ── efficiency ───────────────────────────────────────────────────────────────

/// (5) Microbench: median GB/s of `f` over `runs` (1 warm-up dropped).
/// `bytes_moved` = the kernel's dominant traffic (weights + activations read).
pub fn bench_bandwidth<F: FnMut() -> Vec<f32>>(
    label: &str, bytes_moved: usize, runs: usize, mut f: F,
) -> f64 {
    let _ = f(); // warm-up (pipeline compile / residency)
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = std::time::Instant::now();
        let out = f();
        std::hint::black_box(&out);
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let gbps = (bytes_moved as f64) / med / 1e9;
    eprintln!("[guardrail] {label}: {gbps:.1} GB/s (median over {runs}, {:.3} ms)", med * 1e3);
    gbps
}

/// Assert the kernel reaches at least `min_frac` of device peak bandwidth.
/// M1 Ultra ≈ 800 GB/s; pass the right peak for the box.
///
/// ⚠ ONLY valid when the timed closure hoists SETUP out of the loop (upload
/// weights ONCE, dispatch N times) AND the GPU is otherwise idle. With per-call
/// scope-create + weight upload, or a co-resident process (e.g. the ds4-server
/// daemon), the median is overhead/contention-dominated and reads far below
/// roofline — use `bench_bandwidth` as a RELATIVE regression signal vs a
/// recorded baseline instead of an absolute floor.
pub fn assert_bandwidth_frac(label: &str, gbps: f64, peak_gbps: f64, min_frac: f64) {
    let frac = gbps / peak_gbps;
    assert!(
        frac >= min_frac,
        "{label}: {gbps:.1} GB/s = {:.0}% of {peak_gbps:.0} GB/s peak < {:.0}% floor — \
         likely a bad dispatch geometry (nsg/nr0/tg size).",
        frac * 100.0, min_frac * 100.0,
    );
    eprintln!("[guardrail] {label}: bandwidth OK ({:.0}% of peak)", frac * 100.0);
}

// ── shared RNG for synthetic inputs ──────────────────────────────────────────

pub fn rand_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 9) & 0x7fff) as f32 / 32768.0 - 0.5
        })
        .collect()
}
