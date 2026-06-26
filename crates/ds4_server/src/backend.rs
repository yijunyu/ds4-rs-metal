//! Backend cfg-alias seam.
//!
//! `ds4_server` is backend-agnostic: it drives a decode engine through a tiny,
//! stable surface (`DecodeRunner` + `DecodeSession` + the CPU `sampling`
//! module). The concrete backend crate is selected at compile time by a Cargo
//! feature and re-exported here so the rest of the server only ever names
//! `crate::backend::*`:
//!
//! * `--features metal` (default) → [`ds4_metal`] — macOS / Apple Metal.
//! * `--features cuda`            → `ds4_cuda` — aarch64-linux / NVIDIA DGX Spark.
//!
//! To add the CUDA backend: (1) create `crates/ds4_cuda` exposing
//! `decode_runner::{DecodeRunner, DecodeSession}` and `sampling` with the same
//! public signatures as `ds4_metal`; (2) uncomment the `ds4_cuda` dep + `cuda`
//! feature in `Cargo.toml`; (3) uncomment the `cuda` arm below. No other server
//! file changes. See docs/MIG_DS4_SERVER_DECOUPLE.md for the exact surface a
//! backend must implement.

#[cfg(feature = "metal")]
pub use ds4_metal::{decode_runner, sampling};

#[cfg(feature = "cuda")]
pub use ds4_cuda::{decode_runner, sampling};

// Guard: exactly one backend feature must be active. Without this, a
// `--no-default-features` build would compile an empty `backend` module and
// fail with confusing "unresolved import" errors elsewhere.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
compile_error!(
    "ds4_server requires exactly one backend feature: \
     `--features metal` (macOS) or `--features cuda` (aarch64-linux). \
     The default is `metal`."
);
