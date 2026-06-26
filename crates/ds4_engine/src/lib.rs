//! ds4_engine library surface.
//!
//! The crate is primarily a binary (`ds4-infer`). This `lib.rs` exposes the
//! decode-critical kernel registry to integration tests and external crates
//! (e.g. `codegen_tests` for the emit-vs-registry drift sentinel).
//!
//! The `#[path]` attribute reuses the same source file as the binary, so
//! there is one source of truth for `KERNELS` and the drift-sentinel tests.

#[path = "kernel_registry.rs"]
pub mod kernel_registry;

#[path = "forward.rs"]
pub mod forward;

#[path = "moe.rs"]
pub mod moe;

#[path = "dispatch.rs"]
pub mod dispatch;

#[path = "attn_dispatch.rs"]
pub mod attn_dispatch;

#[path = "layer_view.rs"]
pub mod layer_view;

#[path = "gguf.rs"]
pub mod gguf;

#[path = "kv_cache.rs"]
pub mod kv_cache;

#[path = "decode_step.rs"]
pub mod decode_step;

#[path = "batched_branching.rs"]
pub mod batched_branching;

#[path = "prefill_ffi.rs"]
pub mod prefill_ffi;

#[path = "bench.rs"]
pub mod bench;

#[path = "tokenizer.rs"]
pub mod tokenizer;

#[path = "op_timer.rs"]
pub mod op_timer;

#[path = "mtp.rs"]
pub mod mtp;
