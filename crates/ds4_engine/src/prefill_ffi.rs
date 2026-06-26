//! Hand-written FFI bindings to vendored antirez/ds4 (commit `d615ab0`).
//!
//! M4 uses antirez as the prefill bridge AND as the correctness oracle for
//! decode-token agreement. Both pipelines share the same GGUF and the same
//! tokeniser, so the only thing under test in M4's binary is whether our
//! `MetalDispatcher` produces the same argmax stream as antirez over N
//! decode steps.
//!
//! This file is compiled only on macOS — see `build.rs`. The non-macOS path
//! exposes a stub module so the rest of `ds4_engine` can `pub use` it from
//! `lib.rs` without `#[cfg(...)]` everywhere.

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(not(target_os = "macos"))]
pub use stub::*;

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};
    use std::path::Path;
    use std::ptr;

    /// Opaque handle to antirez `ds4_engine`.
    #[repr(C)]
    pub struct Ds4Engine {
        _private: [u8; 0],
    }

    /// Opaque handle to antirez `ds4_session`.
    #[repr(C)]
    pub struct Ds4Session {
        _private: [u8; 0],
    }

    /// Matches `ds4_backend` in ds4.h.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    #[allow(non_camel_case_types)]
    pub enum Ds4Backend {
        Metal = 0,
        Cpu = 1,
    }

    /// Matches `ds4_tokens` in ds4.h. Owned by antirez when returned from
    /// `ds4_tokenize_text`; freed via `ds4_tokens_free`.
    #[repr(C)]
    pub struct Ds4Tokens {
        pub v: *mut c_int,
        pub len: c_int,
        pub cap: c_int,
    }

    impl Ds4Tokens {
        pub fn empty() -> Self {
            Self {
                v: ptr::null_mut(),
                len: 0,
                cap: 0,
            }
        }

        pub fn as_slice(&self) -> &[c_int] {
            if self.v.is_null() || self.len <= 0 {
                &[]
            } else {
                // SAFETY: antirez guarantees v points to len valid c_ints
                // once tokens have been pushed.
                unsafe { std::slice::from_raw_parts(self.v, self.len as usize) }
            }
        }
    }

    /// Matches `ds4_engine_options` in ds4.h.
    #[repr(C)]
    pub struct Ds4EngineOptions {
        pub model_path: *const c_char,
        pub mtp_path: *const c_char,
        pub backend: Ds4Backend,
        pub n_threads: c_int,
        pub mtp_draft_tokens: c_int,
        pub mtp_margin: f32,
        pub warm_weights: bool,
        pub quality: bool,
    }

    extern "C" {
        pub fn ds4_engine_open(out: *mut *mut Ds4Engine, opt: *const Ds4EngineOptions) -> c_int;
        pub fn ds4_engine_close(e: *mut Ds4Engine);

        pub fn ds4_tokenize_text(e: *mut Ds4Engine, text: *const c_char, out: *mut Ds4Tokens);
        pub fn ds4_encode_chat_prompt(
            e: *mut Ds4Engine,
            system: *const c_char,
            prompt: *const c_char,
            think_mode: c_int,
            out: *mut Ds4Tokens,
        );
        pub fn ds4_tokens_push(tv: *mut Ds4Tokens, token: c_int);
        pub fn ds4_tokens_free(tv: *mut Ds4Tokens);

        pub fn ds4_session_create(
            out: *mut *mut Ds4Session,
            e: *mut Ds4Engine,
            ctx_size: c_int,
        ) -> c_int;
        pub fn ds4_session_free(s: *mut Ds4Session);

        pub fn ds4_session_sync(
            s: *mut Ds4Session,
            prompt: *const Ds4Tokens,
            err: *mut c_char,
            errlen: usize,
        ) -> c_int;
        pub fn ds4_session_argmax(s: *mut Ds4Session) -> c_int;
        pub fn ds4_session_eval(
            s: *mut Ds4Session,
            token: c_int,
            err: *mut c_char,
            errlen: usize,
        ) -> c_int;
        pub fn ds4_session_pos(s: *mut Ds4Session) -> c_int;
        pub fn ds4_token_eos(e: *mut Ds4Engine) -> c_int;
    }

    /// Safe wrapper: load a GGUF model and return an engine handle.
    pub struct Engine {
        ptr: *mut Ds4Engine,
        // Keep the model_path CString alive for the lifetime of the engine.
        _model_path: CString,
    }

    impl Engine {
        /// Open `path` with the Metal backend, mtp disabled, warm_weights on.
        pub fn open_metal(path: &Path) -> Result<Self, String> {
            Self::open(path, Ds4Backend::Metal, true)
        }

        /// Open `path` with the CPU backend and NO weight warming — only the
        /// vocab/tokenizer is needed. Use this when the antirez engine is wanted
        /// purely as the tokenizer/eos oracle (e.g. `--skip-antirez` benches):
        /// it skips antirez's Metal init (no `metal/*.metal` source lookup, no
        /// second 86 GB GPU residency that would wedge Metal alongside our own
        /// DecodeRunner) while still producing the chat-template token stream
        /// required for argmax-equality. See `tokenize()`.
        pub fn open_cpu_tokenizer(path: &Path) -> Result<Self, String> {
            Self::open(path, Ds4Backend::Cpu, false)
        }

        fn open(path: &Path, backend: Ds4Backend, warm_weights: bool) -> Result<Self, String> {
            let model_path = CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| "model path contains NUL byte".to_string())?;
            let opts = Ds4EngineOptions {
                model_path: model_path.as_ptr(),
                mtp_path: ptr::null(),
                backend,
                n_threads: 0,
                mtp_draft_tokens: 0,
                mtp_margin: 0.0,
                warm_weights,
                quality: false,
            };
            let mut raw: *mut Ds4Engine = ptr::null_mut();
            let rc = unsafe { ds4_engine_open(&mut raw, &opts) };
            if rc != 0 || raw.is_null() {
                return Err(format!("ds4_engine_open returned {rc}"));
            }
            Ok(Self {
                ptr: raw,
                _model_path: model_path,
            })
        }

        pub fn as_ptr(&self) -> *mut Ds4Engine {
            self.ptr
        }

        /// Tokenize `text` exactly the way `ds4_cli` does for a `-p` argument:
        /// wraps in the chat template via `ds4_encode_chat_prompt` with no system
        /// message and think_mode = DS4_THINK_NONE (0). This is REQUIRED for
        /// argmax-equality vs antirez baseline (M4 #278). Raw `ds4_tokenize_text`
        /// drops the `<|im_start|>user...<|im_end|>` wrapping and produces a
        /// different (shorter) token sequence.
        pub fn tokenize(&self, text: &str) -> Result<Vec<i32>, String> {
            // ds4_cli defaults (ds4_cli.c:1164,1170): system="You are a helpful assistant",
            // think_mode=DS4_THINK_HIGH(=1). Matching both is required for argmax-equality
            // against the antirez baseline — system contributes ~5 tokens of difference.
            let sys = CString::new("You are a helpful assistant")
                .map_err(|_| "system contains NUL byte".to_string())?;
            let c = CString::new(text).map_err(|_| "prompt contains NUL byte".to_string())?;
            let mut out = Ds4Tokens::empty();
            unsafe { ds4_encode_chat_prompt(self.ptr, sys.as_ptr(), c.as_ptr(), 1, &mut out) };
            let slice = out.as_slice().to_vec();
            unsafe { ds4_tokens_free(&mut out) };
            Ok(slice)
        }

        /// Raw tokenizer escape hatch (no chat-template wrap). Mostly for tests.
        pub fn tokenize_raw(&self, text: &str) -> Result<Vec<i32>, String> {
            let c = CString::new(text).map_err(|_| "prompt contains NUL byte".to_string())?;
            let mut out = Ds4Tokens::empty();
            unsafe { ds4_tokenize_text(self.ptr, c.as_ptr(), &mut out) };
            let slice = out.as_slice().to_vec();
            unsafe { ds4_tokens_free(&mut out) };
            Ok(slice)
        }

        pub fn eos_token(&self) -> i32 {
            unsafe { ds4_token_eos(self.ptr) }
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { ds4_engine_close(self.ptr) };
                self.ptr = ptr::null_mut();
            }
        }
    }

    /// Safe wrapper around `ds4_session`. One sequence at a time.
    pub struct Session {
        ptr: *mut Ds4Session,
        // Engines must outlive the sessions they spawn; PhantomData enforces
        // the borrow at compile time.
        _engine: std::marker::PhantomData<*mut Ds4Engine>,
    }

    impl Session {
        pub fn new(engine: &Engine, ctx_size: i32) -> Result<Self, String> {
            let mut raw: *mut Ds4Session = ptr::null_mut();
            let rc = unsafe { ds4_session_create(&mut raw, engine.ptr, ctx_size) };
            if rc != 0 || raw.is_null() {
                return Err(format!("ds4_session_create returned {rc}"));
            }
            Ok(Self {
                ptr: raw,
                _engine: std::marker::PhantomData,
            })
        }

        /// Prefill: replay the prompt tokens through antirez's Metal graph.
        pub fn sync(&mut self, tokens: &[i32]) -> Result<(), String> {
            let mut owned = Ds4Tokens::empty();
            for &t in tokens {
                unsafe { ds4_tokens_push(&mut owned, t) };
            }
            let mut err = [0u8; 256];
            let rc = unsafe {
                ds4_session_sync(self.ptr, &owned, err.as_mut_ptr() as *mut c_char, err.len())
            };
            unsafe { ds4_tokens_free(&mut owned) };
            if rc != 0 {
                let msg = unsafe { CStr::from_ptr(err.as_ptr() as *const c_char) }
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("ds4_session_sync failed (rc={rc}): {msg}"));
            }
            Ok(())
        }

        /// Read the current top-1 logit id without advancing position.
        pub fn argmax(&self) -> i32 {
            unsafe { ds4_session_argmax(self.ptr) }
        }

        /// Advance one decode step: feed `token` (typically the previous
        /// argmax), antirez runs a single forward, position advances.
        pub fn eval(&mut self, token: i32) -> Result<(), String> {
            let mut err = [0u8; 256];
            let rc = unsafe {
                ds4_session_eval(self.ptr, token, err.as_mut_ptr() as *mut c_char, err.len())
            };
            if rc != 0 {
                let msg = unsafe { CStr::from_ptr(err.as_ptr() as *const c_char) }
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("ds4_session_eval failed (rc={rc}): {msg}"));
            }
            Ok(())
        }

        pub fn pos(&self) -> i32 {
            unsafe { ds4_session_pos(self.ptr) }
        }

        // Pointer leak escape hatch for tests that need to call raw symbols.
        pub fn as_ptr(&mut self) -> *mut Ds4Session {
            self.ptr
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { ds4_session_free(self.ptr) };
                self.ptr = ptr::null_mut();
            }
        }
    }

    // Sessions and engines must stay on the main thread per the M4 plan.
    // Don't implement Send / Sync.
    #[allow(dead_code)]
    fn _assert_no_send_sync(_e: &Engine, _s: &Session, _: *const c_void) {}
}

#[cfg(not(target_os = "macos"))]
mod stub {
    use std::path::Path;

    pub struct Engine;
    pub struct Session;

    impl Engine {
        pub fn open_metal(_path: &Path) -> Result<Self, String> {
            Err("ds4_engine prefill_ffi: Metal bridge is macOS-only".to_string())
        }
        pub fn tokenize(&self, _text: &str) -> Result<Vec<i32>, String> {
            unreachable!("Engine cannot be constructed on non-macOS")
        }
        pub fn eos_token(&self) -> i32 {
            unreachable!("Engine cannot be constructed on non-macOS")
        }
    }

    impl Session {
        pub fn new(_engine: &Engine, _ctx_size: i32) -> Result<Self, String> {
            Err("ds4_engine prefill_ffi: Metal bridge is macOS-only".to_string())
        }
        pub fn sync(&mut self, _tokens: &[i32]) -> Result<(), String> {
            unreachable!()
        }
        pub fn argmax(&self) -> i32 {
            unreachable!()
        }
        pub fn eval(&mut self, _token: i32) -> Result<(), String> {
            unreachable!()
        }
        pub fn pos(&self) -> i32 {
            unreachable!()
        }
    }
}
