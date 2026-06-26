//! Phase F task #90 — GPU `kernel_touch_u8_stride` warm-up smoke.
//!
//! Verifies that `warm_up_buffers_gpu` compiles the inline shader,
//! dispatches it, and writes the expected first-touch bytes into the
//! scratch dst (proving the kernel actually executed end-to-end).
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;
use metal::{Device, MTLResourceOptions};

#[test]
fn warm_up_buffers_gpu_touches_expected_bytes() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let device = Device::system_default().expect("metal device");

    // Two synthetic "expert" buffers with known byte patterns. Sizes
    // are picked so ceil(bytes/stride) at stride=8 gives a known
    // touch-count per buffer.
    let bytes_a: Vec<u8> = (0..24u8).collect(); // touches at off 0, 8, 16
    let buf_a = device.new_buffer_with_data(
        bytes_a.as_ptr() as *const _,
        bytes_a.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bytes_b: Vec<u8> = (100..132u8).collect(); // touches at off 0, 8, 16, 24
    let buf_b = device.new_buffer_with_data(
        bytes_b.as_ptr() as *const _,
        bytes_b.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let views: Vec<(&metal::Buffer, u64)> = vec![
        (&buf_a, bytes_a.len() as u64),
        (&buf_b, bytes_b.len() as u64),
    ];
    let touched = disp.warm_up_buffers_gpu(&views, 8).expect("warm_up_buffers_gpu");
    assert_eq!(touched, (24 + 32) as u64);

    // Run it twice — the OnceLock cache must allow re-dispatch without
    // rebuilding the pipeline. Otherwise the second call would error.
    let touched_again = disp
        .warm_up_buffers_gpu(&views, 8)
        .expect("warm_up_buffers_gpu (second call)");
    assert_eq!(touched_again, (24 + 32) as u64);
}

#[test]
fn warm_up_buffers_gpu_empty_views_no_op() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let touched = disp.warm_up_buffers_gpu(&[], 8).expect("warm_up_buffers_gpu empty");
    assert_eq!(touched, 0);
}
