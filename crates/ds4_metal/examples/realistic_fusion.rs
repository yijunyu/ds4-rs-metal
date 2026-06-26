//! M4 #365 Phase D Slice 2 — realistic-workload fusion microbench.
//!
//! M4 #348's `submit_overhead.rs` measured per-op submit+wait latency with
//! trivial noop kernels (1-thread writeback) and found ~145 µs/op. That
//! is the *driver* floor. Real DS4 decode ops (rms_norm over d_embd=7168,
//! matvec_f32 over [7168 × 7168], etc.) do enough compute that the wait
//! is partially hidden by execution time, so the per-op overhead in the
//! decode hot path may be either lower (compute dominates) or higher
//! (cache/dispatch sync penalty).
//!
//! This bench quantifies the **fusion win**: how much wall time do we
//! save by packing N realistic ops into ONE cmdbuf vs N cmdbufs? That
//! directly tells us how much Phase D's Span-A (6-op fuse) can win.
//!
//! Patterns:
//!  A. per_op_submit_wait      — N cmdbufs, each with one rms_norm or
//!                                matvec_f32 dispatch + commit + wait.
//!                                This is today's `MetalDispatcher` path.
//!  B. batched_in_one_cmdbuf   — 1 cmdbuf with N dispatches, one commit,
//!                                one wait. This is Phase D's destination.
//!  C. multi_cmdbuf_one_wait   — N cmdbufs each committed but only the
//!                                last is waited on. Cheap middle ground.
//!
//! Workload: an `rms_norm`-shaped scalar reduce + write of d=7168 (matches
//! DS4 d_embd). The kernel is hand-written here to keep the bench
//! self-contained (no model/pipelines table dependency).
//!
//! Run on mini (M4 base) or M1 Ultra:
//!   cargo run --release -p ds4_metal --example realistic_fusion
//!
//! Outputs CSV to stdout:
//!   pattern,d,ops_per_batch,total_ops,wall_ms,us_per_op,ops_per_s

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("realistic_fusion: macOS-only (Metal). Skipping.");
}

#[cfg(target_os = "macos")]
use anyhow::{anyhow, Result};
#[cfg(target_os = "macos")]
use metal::{Device, MTLResourceOptions, MTLSize};
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// rms_norm-shaped kernel: read d floats, do an N-thread reduce, write d
// floats back. We don't care about numerical correctness — we want the
// realistic memory traffic + simdgroup reduce pattern.
//
// One threadgroup, 256 threads. Reduces in threadgroup memory then each
// thread writes its strided lane back.
kernel void rms_norm_shaped(
    device const float* x   [[ buffer(0) ]],
    device float*       out [[ buffer(1) ]],
    constant uint&      d   [[ buffer(2) ]],
    threadgroup float*  smem [[ threadgroup(0) ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    // Partial sum-of-squares (each thread strides through d).
    float local_ss = 0.0f;
    for (uint i = tid; i < d; i += 256u) {
        float v = x[i];
        local_ss += v * v;
    }
    smem[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over 256 lanes.
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { smem[tid] += smem[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(smem[0] / float(d) + 1.0e-5f);

    // Write back x * inv_rms.
    for (uint i = tid; i < d; i += 256u) {
        out[i] = x[i] * inv_rms;
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
        .get_function("rms_norm_shaped", None)
        .map_err(|e| anyhow!("function: {}", e))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow!("pso: {}", e))?;

    // DS4 d_embd. Single-row vector workload.
    let d: u32 = 7168;
    let bytes = (d as usize) * std::mem::size_of::<f32>();

    // Shared input/output. We don't care about race conditions across
    // dispatches — every dispatch reads and writes the full d floats.
    let in_buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let out_buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);

    // Seed input with non-zero data so the reduce isn't trivial.
    unsafe {
        let p = in_buf.contents() as *mut f32;
        for i in 0..(d as usize) {
            *p.add(i) = (i as f32) * 0.001 - 0.5;
        }
    }

    // Warmup: hot the driver + I-cache + page-fault output.
    for _ in 0..200 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &d as *const u32 as *const _,
        );
        enc.set_threadgroup_memory_length(0, 256 * 4);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    println!("pattern,d,ops_per_batch,total_ops,wall_ms,us_per_op,ops_per_s");

    // Target ~3000 total ops at each batch size — sub-second wall on mini.
    let total_ops_target: u32 = 3000;

    for &batch in &[1u32, 2, 4, 6, 8, 16, 32, 43, 64] {
        let batches = total_ops_target.div_ceil(batch);
        let total_ops = batches * batch;

        // Pattern A: per-op submit+wait. Run once (batch=1).
        if batch == 1 {
            let t = Instant::now();
            for _ in 0..total_ops {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&in_buf), 0);
                enc.set_buffer(1, Some(&out_buf), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &d as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, 256 * 4);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let us_per_op = wall_ms * 1000.0 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "per_op_submit_wait,{},1,{},{:.3},{:.2},{:.1}",
                d, total_ops, wall_ms, us_per_op, ops_per_s
            );
        }

        // Pattern B: batched_in_one_cmdbuf — batch dispatches per cmdbuf.
        {
            let t = Instant::now();
            for _b in 0..batches {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pso);
                for _ in 0..batch {
                    enc.set_buffer(0, Some(&in_buf), 0);
                    enc.set_buffer(1, Some(&out_buf), 0);
                    enc.set_bytes(
                        2,
                        std::mem::size_of::<u32>() as u64,
                        &d as *const u32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, 256 * 4);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let us_per_op = wall_ms * 1000.0 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "batched_in_one_cmdbuf,{},{},{},{:.3},{:.2},{:.1}",
                d, batch, total_ops, wall_ms, us_per_op, ops_per_s
            );
        }

        // Pattern C: multi_cmdbuf_one_wait — N cmdbufs, only last waited.
        {
            let t = Instant::now();
            for _b in 0..batches {
                let mut last_cb: Option<metal::CommandBuffer> = None;
                for _ in 0..batch {
                    let cb = queue.new_command_buffer().to_owned();
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&pso);
                    enc.set_buffer(0, Some(&in_buf), 0);
                    enc.set_buffer(1, Some(&out_buf), 0);
                    enc.set_bytes(
                        2,
                        std::mem::size_of::<u32>() as u64,
                        &d as *const u32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, 256 * 4);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                    cb.commit();
                    last_cb = Some(cb);
                }
                if let Some(cb) = last_cb {
                    cb.wait_until_completed();
                }
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let us_per_op = wall_ms * 1000.0 / total_ops as f64;
            let ops_per_s = total_ops as f64 / (wall_ms / 1000.0);
            println!(
                "multi_cmdbuf_one_wait,{},{},{},{:.3},{:.2},{:.1}",
                d, batch, total_ops, wall_ms, us_per_op, ops_per_s
            );
        }
    }

    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  per_op_submit_wait   = today's MetalDispatcher TERMINAL_READBACK floor.");
    eprintln!("  batched_in_one_cmdbuf = Phase D Span-A destination (6-op fuse → batch=6 row).");
    eprintln!("  multi_cmdbuf_one_wait = cheap alternative if encoder-reuse isn't possible.");
    eprintln!();
    eprintln!("DS4 today: 1333 commit+wait per token (M4 #348/#350).");
    eprintln!("With Phase D Span-A fuse: ~1333 - 5*43 = 1118 commit+wait per token.");
    eprintln!("Tok/s gain estimate = ops_per_s(batched@6) / ops_per_s(per_op_submit_wait).");

    Ok(())
}
