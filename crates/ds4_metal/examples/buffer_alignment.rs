//! Buffer-alignment microbenchmark for Apple Silicon Metal.
//!
//! Validates M4 #360 hypothesis: that wrapping mmap regions via
//! `newBufferWithBytesNoCopy` with a NON-page-aligned base pointer
//! (offset by GGUF's 32-byte tensor alignment) is materially slower
//! than wrapping a page-aligned base — even though Apple's docs say
//! the pointer MUST be page-aligned (16384 bytes on Apple silicon).
//!
//! Two patterns compared:
//!   - unaligned_per_tensor    — what `quantized_experts.rs:842` does
//!                                today: per-tensor `newBufferWithBytesNoCopy`
//!                                with `ptr = mmap_base + tensor_start` where
//!                                `tensor_start` is 32-byte (GGUF) aligned.
//!   - aligned_shared_view     — what antirez does: ONE big page-aligned view
//!                                covering the whole region, tensors accessed
//!                                via byte-offset.
//!
//! Run on a Mac. No model required — we synthesize a fake 2 GB region.
//!
//!   cargo run --release -p ds4_metal --example buffer_alignment
//!
//! Outputs CSV to stdout:
//!   pattern,run,bytes,wall_ms,gb_per_s
//!
//! If hypothesis #360 holds, `aligned_shared_view` should be O(10×) faster
//! than `unaligned_per_tensor` on the first few runs (when pages are cold)
//! AND on steady-state if Metal silently copies the unaligned buffer.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("buffer_alignment: macOS-only (Metal). Skipping.");
}

#[cfg(target_os = "macos")]
use anyhow::{anyhow, Result};
#[cfg(target_os = "macos")]
use metal::{Device, MTLResourceOptions, MTLSize};
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
extern "C" {
    fn getpagesize() -> std::os::raw::c_int;
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: std::os::raw::c_int,
        flags: std::os::raw::c_int,
        fd: std::os::raw::c_int,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> std::os::raw::c_int;
}

#[cfg(target_os = "macos")]
const PROT_READ: std::os::raw::c_int = 0x01;
#[cfg(target_os = "macos")]
const PROT_WRITE: std::os::raw::c_int = 0x02;
#[cfg(target_os = "macos")]
const MAP_PRIVATE: std::os::raw::c_int = 0x0002;
#[cfg(target_os = "macos")]
const MAP_ANON: std::os::raw::c_int = 0x1000;
#[cfg(target_os = "macos")]
fn map_failed() -> *mut std::ffi::c_void {
    usize::MAX as *mut std::ffi::c_void
}

// Trivial bandwidth-bound kernel: sum the buffer. We want to stress the
// memory subsystem, not arithmetic. Output is one scalar.
#[cfg(target_os = "macos")]
const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sum_f32(
    device const float* in [[ buffer(0) ]],
    device atomic_uint* out [[ buffer(1) ]],
    constant uint& n_elems [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]],
    uint nthreads [[ threads_per_grid ]]
) {
    float acc = 0.0f;
    for (uint i = tid; i < n_elems; i += nthreads) {
        acc += in[i];
    }
    // Sink to atomic so the compiler can't elide the loop. We don't read
    // the sum back; we just need the bytes to actually be touched.
    uint acc_bits = as_type<uint>(acc);
    atomic_fetch_add_explicit(out, acc_bits, memory_order_relaxed);
}
"#;

#[cfg(target_os = "macos")]
fn page_size() -> usize {
    unsafe { getpagesize() as usize }
}

#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    let device = Device::system_default().ok_or_else(|| anyhow!("no default Metal device"))?;
    let queue = device.new_command_queue();
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(KERNEL_SRC, &opts)
        .map_err(|e| anyhow!("compile failed: {}", e))?;
    let func = lib
        .get_function("sum_f32", None)
        .map_err(|e| anyhow!("function: {}", e))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow!("pso: {}", e))?;

    let page = page_size();
    eprintln!("getpagesize() = {} bytes", page);
    eprintln!(
        "Apple silicon expectation: 16384. If different, hypothesis #360 still applies."
    );

    // Synthesize a 2 GiB region via mmap so we have realistic page-table
    // pressure. Fill with non-zero junk so the kernel doesn't get clever.
    let region_bytes: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
    let region: *mut std::ffi::c_void = unsafe {
        mmap(
            std::ptr::null_mut(),
            region_bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0,
        )
    };
    if region == map_failed() {
        return Err(anyhow!("mmap 2 GiB failed: {}", std::io::Error::last_os_error()));
    }
    // Fill so pages get committed and we measure GPU read cost, not first-fault.
    unsafe {
        let p = region as *mut u32;
        let n = region_bytes / 4;
        for i in 0..n {
            std::ptr::write(p.add(i), (i as u32) ^ 0xDEAD_BEEF);
        }
    }

    // Per-tensor unaligned offsets: simulate GGUF tensor table — every
    // tensor starts at a multiple of 32 bytes, NOT page. We carve N
    // "tensors" of size T from the region.
    let tensor_bytes: usize = 16 * 1024 * 1024; // 16 MiB each — biggish expert slice
    let n_tensors: usize = 64;
    let gguf_align: usize = 32;
    assert!(tensor_bytes % 4 == 0);
    assert!(tensor_bytes >= page);
    let mut tensor_starts: Vec<usize> = Vec::with_capacity(n_tensors);
    {
        let mut off: usize = 0;
        for _ in 0..n_tensors {
            // GGUF-style 32-byte align (NOT page-aligned for most tensors).
            let pad = (gguf_align - (off % gguf_align)) % gguf_align;
            off += pad;
            // Bump by 32 so the start is NOT page-aligned (assuming region is page-aligned).
            // First tensor: off=0 → page-aligned. Skip it by adding 32.
            if off % page == 0 {
                off += gguf_align;
            }
            assert!(off + tensor_bytes <= region_bytes);
            tensor_starts.push(off);
            off += tensor_bytes;
        }
    }
    eprintln!(
        "Region: {} GiB. Tensors: {} × {} MiB. First tensor start mod page = {} (should be != 0).",
        region_bytes / (1024 * 1024 * 1024),
        n_tensors,
        tensor_bytes / (1024 * 1024),
        tensor_starts[0] % page
    );

    let n_elems_per_tensor = (tensor_bytes / 4) as u32;
    let out_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Warm: one dispatch per pattern to compile/cache anything lazy.
    let aligned_region_buf = device.new_buffer_with_bytes_no_copy(
        region as *const _,
        region_bytes as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&aligned_region_buf), tensor_starts[0] as u64);
        enc.set_buffer(1, Some(&out_buf), 0);
        let n = n_elems_per_tensor;
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    println!("pattern,run,bytes,wall_ms,gb_per_s");

    let runs: u32 = 3;

    // --- Pattern A: unaligned_per_tensor ---
    //
    // Mirrors `quantized_experts.rs:842` exactly: ONE buffer per tensor,
    // base pointer at `region + tensor_start` where `tensor_start` is
    // 32-byte aligned (not page-aligned). Issues n_tensors dispatches per run.
    for run in 0..runs {
        let mut buffers: Vec<metal::Buffer> = Vec::with_capacity(n_tensors);
        let t_alloc = Instant::now();
        for &start in &tensor_starts {
            let ptr = unsafe { (region as *const u8).add(start) };
            let b = device.new_buffer_with_bytes_no_copy(
                ptr as *const _,
                tensor_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            buffers.push(b);
        }
        let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for b in &buffers {
            enc.set_buffer(0, Some(b), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            let n = n_elems_per_tensor;
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(64, 1, 1));
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_ms = t.elapsed().as_secs_f64() * 1000.0;

        let total_bytes = (n_tensors * tensor_bytes) as f64;
        let gb_per_s = total_bytes / (wall_ms / 1000.0) / 1.0e9;
        println!(
            "unaligned_per_tensor,{},{},{:.3},{:.2}",
            run, total_bytes as u64, wall_ms, gb_per_s
        );
        eprintln!(
            "  unaligned_per_tensor run {}: alloc {:.2} ms + dispatch {:.2} ms = {:.2} GB/s",
            run, alloc_ms, wall_ms, gb_per_s
        );
    }

    // --- Pattern B: aligned_shared_view ---
    //
    // Mirrors antirez ds4_metal.m:407-472: ONE page-aligned buffer covering
    // the whole region; tensors accessed via byte-offset. Issues n_tensors
    // dispatches per run, same as pattern A — only the buffer wrapping differs.
    for run in 0..runs {
        let t = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for &start in &tensor_starts {
            enc.set_buffer(0, Some(&aligned_region_buf), start as u64);
            enc.set_buffer(1, Some(&out_buf), 0);
            let n = n_elems_per_tensor;
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(64, 1, 1));
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_ms = t.elapsed().as_secs_f64() * 1000.0;

        let total_bytes = (n_tensors * tensor_bytes) as f64;
        let gb_per_s = total_bytes / (wall_ms / 1000.0) / 1.0e9;
        println!(
            "aligned_shared_view,{},{},{:.3},{:.2}",
            run, total_bytes as u64, wall_ms, gb_per_s
        );
        eprintln!(
            "  aligned_shared_view run {}: dispatch {:.2} ms = {:.2} GB/s",
            run, wall_ms, gb_per_s
        );
    }

    // --- Pattern C: aligned_per_tensor ---
    //
    // Isolates "alignment" from "buffer count". Same N=n_tensors buffer
    // count as pattern A, but each buffer base is page-aligned (we shift
    // each tensor's start down to its enclosing page and adjust length).
    // The kernel reads bytes [page_floor(start)..page_floor(start)+len),
    // which over-reads up to one page on each side — but every base is
    // page-aligned, so we satisfy the documented requirement.
    //
    // If C ≈ B and both ≫ A: alignment is the load-bearing factor (refactor wins).
    // If C ≈ A and only B is fast: buffer COUNT is the load-bearing factor
    //   (must consolidate into shared views, can't just align per-tensor).
    // If A ≈ B ≈ C: neither matters; revisit #358 (encoder fan-out).
    for run in 0..runs {
        let mut buffers: Vec<metal::Buffer> = Vec::with_capacity(n_tensors);
        let t_alloc = Instant::now();
        for &start in &tensor_starts {
            let page_floor = start & !(page - 1);
            // Length: round (start - page_floor + tensor_bytes) up to a page.
            let body_with_lead = (start - page_floor) + tensor_bytes;
            let aligned_len = (body_with_lead + page - 1) & !(page - 1);
            // Don't exceed region.
            let aligned_len = aligned_len.min(region_bytes - page_floor);
            let ptr = unsafe { (region as *const u8).add(page_floor) };
            let b = device.new_buffer_with_bytes_no_copy(
                ptr as *const _,
                aligned_len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            buffers.push(b);
        }
        let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in buffers.iter().enumerate() {
            let start = tensor_starts[i];
            let page_floor = start & !(page - 1);
            let lead = start - page_floor;
            enc.set_buffer(0, Some(b), lead as u64);
            enc.set_buffer(1, Some(&out_buf), 0);
            let n = n_elems_per_tensor;
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(64, 1, 1));
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_ms = t.elapsed().as_secs_f64() * 1000.0;

        let total_bytes = (n_tensors * tensor_bytes) as f64;
        let gb_per_s = total_bytes / (wall_ms / 1000.0) / 1.0e9;
        println!(
            "aligned_per_tensor,{},{},{:.3},{:.2}",
            run, total_bytes as u64, wall_ms, gb_per_s
        );
        eprintln!(
            "  aligned_per_tensor run {}: alloc {:.2} ms + dispatch {:.2} ms = {:.2} GB/s",
            run, alloc_ms, wall_ms, gb_per_s
        );
    }

    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  If B ≫ A and C ≫ A: alignment is the load-bearing factor (any aligned base wins).");
    eprintln!("  If B ≫ A and C ≈ A: buffer COUNT matters too (must consolidate into shared views).");
    eprintln!("  If A ≈ B ≈ C: neither matters; revisit #358 (encoder fan-out).");

    unsafe {
        munmap(region, region_bytes);
    }
    Ok(())
}
