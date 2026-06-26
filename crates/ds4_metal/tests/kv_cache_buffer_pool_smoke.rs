//! Phase E M5.2 — persistent KV cache buffer pool smoke test.
//!
//! Verifies `MetalDispatcher::kv_buffer_or_alloc(layer_idx, byte_size)`:
//!   1. allocates a buffer of the requested size on first call
//!   2. returns the SAME buffer (pointer-equal underlying NSObject) on
//!      subsequent calls for the same layer_idx
//!   3. distinct layer_idxs map to distinct buffers
//!   4. data written via `contents()` persists across get-or-alloc
//!      calls — the buffer isn't reset
//!
//! These are the properties `kv_fp8_store_impl` will rely on once it's
//! migrated (M5.2-followup) to write into the persistent buffer
//! instead of creating a fresh one from `KvCacheView::raw` each call.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use metal::foreign_types::ForeignType;

#[test]
fn kv_buffer_or_alloc_returns_same_buffer_for_same_layer() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let bytes = 64 * 512 * 4; // raw_cap=64, n_lora_kv=512, f32 sized
    let a = disp.kv_buffer_or_alloc(0, bytes);
    let b = disp.kv_buffer_or_alloc(0, bytes);
    // The two clones must wrap the same NSObject.
    assert_eq!(a.as_ptr() as usize, b.as_ptr() as usize, "layer-0 pointer mismatch");
}

#[test]
fn kv_buffer_or_alloc_distinct_per_layer() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let bytes = 64 * 512 * 4;
    let a = disp.kv_buffer_or_alloc(0, bytes);
    let b = disp.kv_buffer_or_alloc(1, bytes);
    let c = disp.kv_buffer_or_alloc(2, bytes);
    let pa = a.as_ptr() as usize;
    let pb = b.as_ptr() as usize;
    let pc = c.as_ptr() as usize;
    assert_ne!(pa, pb, "layer 0 == layer 1");
    assert_ne!(pa, pc, "layer 0 == layer 2");
    assert_ne!(pb, pc, "layer 1 == layer 2");
}

#[test]
fn kv_buffer_or_alloc_data_persists_across_get_calls() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let n = 256usize;
    let bytes = n * 4;
    let buf = disp.kv_buffer_or_alloc(42, bytes);

    // Write a known pattern.
    let pattern: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 - 1.0).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            pattern.as_ptr(),
            buf.contents() as *mut f32,
            n,
        );
    }

    // Drop the clone, re-fetch, verify same data.
    drop(buf);
    let buf2 = disp.kv_buffer_or_alloc(42, bytes);
    let mut readback = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf2.contents() as *const f32,
            readback.as_mut_ptr(),
            n,
        );
    }
    assert_eq!(readback, pattern, "buffer data did not persist");
}
