//! nsg geometry sweep for the two largest un-audited decode kernels, following
//! the q8_matvec_bandwidth_bench methodology that found the +7.5% q8 nsg win.
//!
//!   1. attn output-proj stage-1 (`ds4_dsv4_attn_out_low_q8_0_f32`): grouped
//!      Q8_0 matvec, group_dim=4096 → n_lora_o=1024 over n_groups=8. Production
//!      nsg = clamp(group_dim/128) = 8 — the exact high-nsg bias the q8 audit
//!      proved suboptimal. Swept via DS4_ATTN_OUT_NSG.
//!   2. MoE expert fused pair_swiglu (`..._iq2_xxs_pair_swiglu_K_f32`, K=1):
//!      d_embd=4096 → inter=2048, top-6 experts, iq2_xxs. Production nsg=2,
//!      N_R0=4. Swept via DS4_MOE_NSG. This is ~40-45% of the GPU-bound token
//!      and runs at ~65 GB/s = 8% of peak (occupancy-bound → nsg should bite).
//!
//! Reports µs/dispatch per nsg; the lowest wins. Bit-identical output (geometry
//! only), so any win is free + quality-neutral. Synthetic weights at REAL DS4
//! V4-Flash shapes, no GGUF → safe alongside ds4-server. Opt-in: DS4_BENCH_NSG=1.
#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use std::time::Instant;

/// Random block_q8_0 bytes for a [d_out, d_in] weight (per 32-block: f16 scale
/// 1.0 + 32 int8). Layout matches `weight_q8_0_raw`.
fn rand_q8_0(d_in: usize, d_out: usize, seed: u32) -> Vec<u8> {
    let nb = d_in / 32;
    let row = nb * 34;
    let mut w = vec![0u8; d_out * row];
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };
    for r in 0..d_out {
        for b in 0..nb {
            let off = r * row + b * 34;
            w[off + 1] = 0x3C; // half 1.0
            for i in 0..32 {
                w[off + 2 + i] = (((next() & 0x3f) as i32 - 32) as i8) as u8;
            }
        }
    }
    w
}

/// Random iq2_xxs bytes for [d_out, d_in] (66 bytes / 256-block). The kernel's
/// grid-table lookups are valid for any byte pattern, so random bytes give a
/// faithful read/occupancy profile for a *timing* sweep (values irrelevant).
fn rand_iq2_xxs(d_in: usize, d_out: usize, seed: u32) -> Vec<u8> {
    let nb = d_in / 256;
    let row = nb * 66;
    let mut w = vec![0u8; d_out * row];
    let mut rng = seed;
    for b in w.iter_mut() {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (rng >> 16) as u8;
    }
    w
}

fn time_sweep<F: Fn()>(run: F, n: usize, weight_bytes: usize) -> (f64, f64) {
    run();
    run(); // warmup: pipeline build + first-touch residency
    let reps = 4;
    let t0 = Instant::now();
    for _ in 0..reps {
        run();
    }
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    let per_us = secs / n as f64 * 1e6;
    let gbps = (weight_bytes as f64 * n as f64) / secs / 1e9;
    (per_us, gbps)
}

#[test]
fn nsg_sweep_moe_and_attn_out() {
    if std::env::var("DS4_BENCH_NSG").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_NSG unset — skipping. Set =1 to run.");
        return;
    }
    let disp = match MetalDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: MetalDispatcher::new failed: {e}");
            return;
        }
    };
    let n = 32usize; // dispatches per timed batch
    let nsgs = [1i16, 2, 3, 4, 8];

    // ── 1. attn output-proj (stage-1 nsg swept; stage-2 matvec_q8_0 constant) ──
    let (n_groups, n_lora_o, group_dim) = (8usize, 1024usize, 4096usize);
    let out_low_dim = n_groups * n_lora_o; // 8192
    let d_embd = 4096usize;
    let w_o_a = rand_q8_0(group_dim, out_low_dim, 0xA11CE); // [out_low_dim, group_dim]
    let w_o_b = rand_q8_0(out_low_dim, d_embd, 0xB0B); // [d_embd, out_low_dim]
    let heads: Vec<f32> = (0..n_groups * group_dim).map(|i| ((i as f32) * 0.001).sin()).collect();
    // stage-1 reads w_o_a once per dispatch:
    let s1_bytes = w_o_a.len();
    eprintln!(
        "\n=== attn output-proj stage-1  (group_dim={} → n_lora_o={} × {} groups, {:.1} MB q8) ===",
        group_dim, n_lora_o, n_groups, s1_bytes as f64 / 1e6
    );
    eprintln!("  nsg   µs/dispatch   stage1-GB/s   note");
    let mut best = (f64::MAX, 0i16);
    for &nsg in &nsgs {
        std::env::set_var("DS4_ATTN_OUT_NSG", nsg.to_string());
        let run = || {
            let scope = disp.batch_scope();
            let a = scope.weight_q8_0_raw(&w_o_a, out_low_dim * group_dim);
            let b = scope.weight_q8_0_raw(&w_o_b, d_embd * out_low_dim);
            let h = scope.upload_f32(&heads);
            let mut outs = Vec::with_capacity(n);
            for _ in 0..n {
                let (_low, out) = scope
                    .encode_attn_output_matmuls_q8(&h, &a, &b, n_groups, n_lora_o, group_dim, out_low_dim, d_embd)
                    .expect("attn out");
                outs.push(out);
            }
            let refs: Vec<&_> = outs.iter().collect();
            let _ = scope.flush_and_read_multi(&refs);
        };
        let (us, gb) = time_sweep(run, n, s1_bytes);
        if us < best.0 {
            best = (us, nsg);
        }
        let tag = if nsg == 8 { "(production)" } else { "" };
        eprintln!("  {:>3}   {:>9.1}    {:>9.0}     {}", nsg, us, gb, tag);
    }
    std::env::remove_var("DS4_ATTN_OUT_NSG");
    eprintln!("  → best nsg={} ({:.1} µs)  vs production nsg=8", best.1, best.0);

    // ── 2. MoE expert fused pair_swiglu (iq2_xxs, K=1) ──
    let (d_in, d_ffn) = (4096usize, 2048usize);
    let n_exp = 6usize; // allocate exactly the routed experts; ids = 0..5
    let w_gate = rand_iq2_xxs(d_in, d_ffn * n_exp, 0x6A7E);
    let w_up = rand_iq2_xxs(d_in, d_ffn * n_exp, 0x12);
    let x: Vec<f32> = (0..d_in).map(|i| ((i as f32) * 0.002).cos()).collect();
    // ids_flat = [0,1,2,3,4,5] i32, uploaded as raw bytes; weights_flat = 1.0
    let mut ids_bytes = vec![0u8; 6 * 4];
    for (e, chunk) in ids_bytes.chunks_mut(4).enumerate() {
        chunk.copy_from_slice(&(e as i32).to_le_bytes());
    }
    let wts: Vec<f32> = vec![1.0; 6];
    let moe_bytes = w_gate.len() + w_up.len(); // both fully read (6 experts × gate+up)
    eprintln!(
        "\n=== MoE expert pair_swiglu  (d_embd={} → inter={}, 6 experts iq2_xxs, {:.1} MB) ===",
        d_in, d_ffn, moe_bytes as f64 / 1e6
    );
    eprintln!("  nsg   µs/dispatch   GB/s   %800   note");
    let mut bestm = (f64::MAX, 0i16);
    for &nsg in &nsgs {
        std::env::set_var("DS4_MOE_NSG", nsg.to_string());
        let run = || {
            let scope = disp.batch_scope();
            let g = scope.weight_q8_0_raw(&w_gate, w_gate.len()); // byte-agnostic resident upload
            let u = scope.weight_q8_0_raw(&w_up, w_up.len());
            let xk = scope.upload_f32(&x);
            let ids = scope.weight_q8_0_raw(&ids_bytes, 6);
            let wf = scope.upload_f32(&wts);
            let mut outs = Vec::with_capacity(n);
            for _ in 0..n {
                let mid = scope
                    .encode_pair_swiglu_K_iq2_xxs(&g, &u, &xk, &ids, &wf, n_exp, d_in, d_ffn, 1)
                    .expect("pair_swiglu");
                outs.push(mid);
            }
            let refs: Vec<&_> = outs.iter().collect();
            let _ = scope.flush_and_read_multi(&refs);
        };
        let (us, gb) = time_sweep(run, n, moe_bytes);
        if us < bestm.0 {
            bestm = (us, nsg);
        }
        let tag = if nsg == 2 { "(production)" } else { "" };
        eprintln!("  {:>3}   {:>9.1}    {:>5.0}   {:>3.0}%  {}", nsg, us, gb, gb / 800.0 * 100.0, tag);
    }
    std::env::remove_var("DS4_MOE_NSG");
    eprintln!("  → best nsg={} ({:.1} µs)  vs production nsg=2", bestm.1, bestm.0);

    eprintln!(
        "\nInterpretation: a best-nsg well below the production row at either kernel\n\
         is a free, bit-identical decode win (bake the winner into the dispatch).\n\
         If all ~equal, that kernel is reduction/dequant-bound, not occupancy-bound."
    );
}
