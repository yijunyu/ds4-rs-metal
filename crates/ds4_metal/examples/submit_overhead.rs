//! Submit-overhead microbenchmark for Apple Silicon Metal queue.
//!
//! Measures how `commit() + wait_until_completed()` latency degrades the
//! tok/s ceiling under three encoding patterns. The "per-op submit+wait"
//! pattern is what `MetalDispatcher` does today; the "batched commit,
//! single wait" pattern is what `SingleBufferEncoder` aims at; the
//! "multi-cmdbuf, single wait" pattern is the cheap middle ground.
//!
//! Run on a Mac (mini or M1 Ultra). No model required.
//!
//!   cargo run --release -p ds4_metal --example submit_overhead
//!
//! Outputs a CSV to stdout. Each row:
//!   pattern,submits_per_batch,total_ops,wall_ms,ns_per_op,ops_per_s

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("submit_overhead: macOS-only (Metal). Skipping.");
}

#[cfg(target_os = "macos")]
use anyhow::{anyhow, Result};
#[cfg(target_os = "macos")]
use metal::{Device, MTLResourceOptions, MTLSize};
#[cfg(target_os = "macos")]
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Trivial no-op-ish kernel: write a counter into output[0]. We don't
// care about the value — we want to isolate driver submit-overhead, not
// arithmetic throughput. One thread, one workgroup.
kernel void noop_writeback(
    device float* out [[ buffer(0) ]],
    constant uint& tick [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid == 0) {
        out[0] = float(tick);
    }
}
"#;

#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    let device = Device::system_default().ok_or_else(|| anyhow!("no default Metal device"))?;
    let queue = device.new_command_queue();

    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(KERNEL_SRC, &opts)
        .map_err(|e| anyhow!("compile failed: {}", e))?;
    let func = lib
        .get_function("noop_writeback", None)
        .map_err(|e| anyhow!("function: {}", e))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow!("pso: {}", e))?;

    let out_buf = device.new_buffer(
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Warm: drive a few thousand submits so the queue is hot.
    for tick in 0u32..2000 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&out_buf), 0);
        enc.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &tick as *const u32 as *const _,
        );
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    println!("pattern,submits_per_batch,total_ops,wall_ms,ns_per_op,ops_per_s");

    // Each row gets ~5000 ops worth of work so timing is stable.
    let total_ops_target: u32 = 5000;

    // Sweep batch sizes. submits_per_batch ∈ {1,4,16,64,256,1024} —
    // 1 = current per-op submit+wait baseline; 1024 ≈ "one cmdbuf per token".
    for &batch in &[1u32, 4, 16, 64, 256, 1024] {
        let batches = total_ops_target.div_ceil(batch);
        let total_ops = batches * batch;

        // --- Pattern A: per-op submit+wait (today's MetalDispatcher) ---
        // Here `batch` is meaningless — every op is its own cmdbuf with its
        // own wait. We just run total_ops dispatches. Same row repeated for
        // every batch size as a reference; only run once at batch=1.
        if batch == 1 {
            let t = Instant::now();
            for tick in 0..total_ops {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&out_buf), 0);
                enc.set_bytes(
                    1,
                    std::mem::size_of::<u32>() as u64,
                    &tick as *const u32 as *const _,
                );
                enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let ns_per_op = wall_ms * 1.0e6 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "per_op_submit_wait,1,{},{:.3},{:.0},{:.1}",
                total_ops, wall_ms, ns_per_op, ops_per_s
            );
        }

        // --- Pattern B: N ops per cmdbuf, one commit + one wait per batch ---
        // This is what SingleBufferEncoder approaches in the limit. Each
        // cmdbuf encodes `batch` dispatches, then commit+wait once.
        {
            let t = Instant::now();
            let mut tick: u32 = 0;
            for _b in 0..batches {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pso);
                for _ in 0..batch {
                    enc.set_buffer(0, Some(&out_buf), 0);
                    enc.set_bytes(
                        1,
                        std::mem::size_of::<u32>() as u64,
                        &tick as *const u32 as *const _,
                    );
                    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    tick += 1;
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let ns_per_op = wall_ms * 1.0e6 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "batched_in_cmdbuf,{},{},{:.3},{:.0},{:.1}",
                batch, total_ops, wall_ms, ns_per_op, ops_per_s
            );
        }

        // --- Pattern C: separate cmdbufs per op, BUT only wait on the last ---
        // The Metal queue serializes implicitly, so committing batch cmdbufs
        // and waiting only on the last is correct. This isolates "wait" cost
        // from "commit" cost — the trade-off if we just drop per-op waits.
        {
            let t = Instant::now();
            let mut tick: u32 = 0;
            for _b in 0..batches {
                let mut last_cb: Option<metal::CommandBuffer> = None;
                for _ in 0..batch {
                    let cb = queue.new_command_buffer().to_owned();
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&pso);
                    enc.set_buffer(0, Some(&out_buf), 0);
                    enc.set_bytes(
                        1,
                        std::mem::size_of::<u32>() as u64,
                        &tick as *const u32 as *const _,
                    );
                    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    enc.end_encoding();
                    cb.commit();
                    last_cb = Some(cb);
                    tick += 1;
                }
                if let Some(cb) = last_cb {
                    cb.wait_until_completed();
                }
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let ns_per_op = wall_ms * 1.0e6 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "commit_no_wait_until_last,{},{},{:.3},{:.0},{:.1}",
                batch, total_ops, wall_ms, ns_per_op, ops_per_s
            );
        }
    }

    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  per_op_submit_wait      = today's MetalDispatcher path.");
    eprintln!("  batched_in_cmdbuf       = SingleBufferEncoder direction.");
    eprintln!("  commit_no_wait_until_last = cheap alternative if waits dominate.");
    eprintln!();
    eprintln!("DS4 today: ~347 submits per generated token.");
    eprintln!("Ceiling tok/s = ops_per_s(per_op_submit_wait) / 347.");
    eprintln!("Antirez observed: 26.35 tok/s = ~9145 ops/s effective.");

    Ok(())
}
