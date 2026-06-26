//! Task #8 — isolate the MoE 12× bandwidth gap: pure contiguous-read ceiling.
//!
//! The production fused MoE reads 42.5 MB of quantized expert weight per token
//! per layer (top-6 experts, iq2_xxs+q2_K) and takes ~655 µs → only ~65 GB/s =
//! ~8% of M1 Ultra's 800 GB/s peak. Is that gap (a) recoverable DMA/parallelism
//! underutilization, or (b) irreducible dequant compute + GEMV occupancy?
//!
//! This microbench measures the ACHIEVABLE byte-read ceiling on this GPU: a
//! trivial kernel that bulk-reads N bytes (float4 grid-stride) and sums them
//! (write-back prevents dead-code elimination). No dequant, no matmul, max
//! parallelism. Achieved GB/s here is the ceiling the MoE *could* approach.
//!   - ceiling ≈ 500-600 GB/s  → MoE's 65 GB/s is dequant/occupancy-bound
//!     (MLSys "improve DMA" lever does NOT directly apply).
//!   - ceiling ≈ 65 GB/s too   → genuine memory-parallelism underutilization
//!     (the MLSys lever applies; a real multi-× MoE win exists).
//!
//! Gated by DS4_BENCH_MOE_BW=1. macOS-only; self-contained (no GGUF).
#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bw_read_sum(
    device const float4 * src   [[buffer(0)]],
    device       float  * out   [[buffer(1)]],
    constant uint       & n_vec4   [[buffer(2)]],
    constant uint       & n_threads[[buffer(3)]],
    constant uint       & n_pass   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    float4 acc = float4(0.0f);
    // n_pass passes over the buffer per dispatch: amplifies GPU read work so
    // command-buffer launch+sync overhead becomes negligible vs sustained BW.
    for (uint p = 0u; p < n_pass; p++) {
        for (uint i = gid; i < n_vec4; i += n_threads) {
            acc += src[i];
        }
    }
    out[gid & 1023u] = acc.x + acc.y + acc.z + acc.w;
}
"#;

#[test]
fn moe_bandwidth_ceiling() {
    if std::env::var("DS4_BENCH_MOE_BW").ok().as_deref() != Some("1") {
        eprintln!("DS4_BENCH_MOE_BW unset — skipping. Set =1 to run.");
        return;
    }
    let device = match Device::system_default() {
        Some(d) => d,
        None => { eprintln!("no default Metal device — skipping"); return; }
    };
    eprintln!("device: {}", device.name());

    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(SRC, &opts).expect("compile bw kernel");
    let func = lib.get_function("bw_read_sum", None).expect("get_function");
    let pipe = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline");
    let queue = device.new_command_queue();
    let out = device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);
    let tg_size: u64 = 256;
    let n_tg: u64 = 4096; // 1,048,576 threads, grid-strided
    let n_threads: u32 = (tg_size * n_tg) as u32;

    // (working-set bytes, passes/dispatch, iters): n_pass amplifies GPU read
    // work so the per-dispatch launch+sync (~hundreds of µs) is negligible.
    let configs: &[(usize, u32, usize)] = &[
        (42_467_328, 1, 200),   // single-pass 42.5MB = MoE working set + launch overhead
        (42_467_328, 128, 30),  // 42.5MB resident, launch amortized → small-WS sustained BW
        (1_073_741_824, 4, 20), // 1 GB → large-WS sustained BW = true peak ceiling
    ];
    eprintln!("\n  bytes/iter      passes  µs/iter     GB/s   %800   note");
    for &(bytes, n_pass, iters) in configs {
        let n_vec4: u32 = (bytes / 16) as u32;
        let src = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let dispatch = || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&src), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 4, &n_vec4 as *const u32 as *const _);
            enc.set_bytes(3, 4, &n_threads as *const u32 as *const _);
            enc.set_bytes(4, 4, &n_pass as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg_size, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        for _ in 0..3 { dispatch(); } // warmup
        let t0 = Instant::now();
        for _ in 0..iters { dispatch(); }
        let secs = t0.elapsed().as_secs_f64();
        let read_per_iter = bytes as f64 * n_pass as f64;
        let per_iter_us = secs / iters as f64 * 1e6;
        let gbps = read_per_iter * iters as f64 / secs / 1e9;
        let note = if n_pass == 1 { "1-pass (incl. launch)" } else { "sustained" };
        eprintln!(
            "  {:>10.1} MB   {:>4}    {:>8.1}   {:>6.1}   {:>3.0}%   {}",
            bytes as f64 / 1e6, n_pass, per_iter_us, gbps, gbps / 800.0 * 100.0, note
        );
    }
    eprintln!(
        "\n  Production fused MoE K=1 = 655 µs to read 42.5 MB = 65 GB/s.\n  → sustained ceiling >> 65 GB/s ⇒ MoE gap is dequant/occupancy (MLSys DMA lever N/A);\n    sustained ~ 65 GB/s ⇒ memory-parallelism underutilization (MLSys lever applies)."
    );
}
