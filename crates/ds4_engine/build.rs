// build.rs — Compile vendored antirez/ds4 (`ds4.c` + `ds4_metal.m`) into a
// static library and emit linker flags so `ds4-infer` can call the prefill /
// argmax / eval public API via FFI from `src/prefill_ffi.rs`.
//
// macOS-only. On any other target this is a no-op.
//
// Pinned upstream: antirez/ds4 commit d615ab0 (vendored at
// ../../../ascend-rs-priv/benchmarks/ds4_msl/upstream/ds4). If those files
// drift, the sha256 check below fails fast with a diagnostic.

use std::env;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
const UPSTREAM_COMMIT: &str = "d615ab08c8bce9b8242963ecece5aed6b5a79367";

// SHA-256 of the four upstream files at commit d615ab0. If any drifts, the
// build fails so we can re-audit the prefill_ffi binding before shipping.
#[allow(dead_code)]
const PINNED_HASHES: &[(&str, &str)] = &[
    (
        "ds4.c",
        "c5b8477ac3f2a542fcdcbc076d67e4ad13851e6b294fd3be2533713bdba856fa",
    ),
    (
        "ds4_metal.m",
        "a922e3dac18b7780bf8a046c6c92af1aba6b4b72397348c55666692a708bf02d",
    ),
    (
        "ds4.h",
        "299c65d14980226069c4f89c4379041c9380e4f36e9997cf08b4cb11fb8ea071",
    ),
    (
        "ds4_metal.h",
        "7637f3c1cf208a63434f9800c8b1a0a91c2e90e7523c3553e24f7fe489a36082",
    ),
];

fn main() {
    // Re-run if our build script changes.
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        // Linux / others: prefill_ffi.rs is a stub; nothing to compile.
        println!("cargo:warning=ds4_engine: skipping antirez build on {target_os}");
        return;
    }

    build_macos();
}

#[cfg(target_os = "macos")]
fn build_macos() {
    let upstream = locate_upstream();
    verify_pinned(&upstream);

    let mut build = cc::Build::new();
    build
        .file(upstream.join("ds4.c"))
        .file(upstream.join("ds4_metal.m"))
        .include(&upstream)
        .flag("-fobjc-arc")
        .flag("-std=c11")
        .define("DS4_NO_LINENOISE", None)
        .define("DS4_NO_MAIN", None)
        .opt_level(3)
        .warnings(false);

    build.compile("ds4_upstream");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");

    for (name, _) in PINNED_HASHES {
        println!("cargo:rerun-if-changed={}", upstream.join(name).display());
    }
}

#[cfg(not(target_os = "macos"))]
fn build_macos() {
    // unreachable: caller checks CARGO_CFG_TARGET_OS == "macos" before
    // dispatching here, and `cc` is only present as a build-dep on macOS.
    unreachable!("ds4_engine build.rs: macOS path invoked from non-macOS host");
}

#[allow(dead_code)]
fn locate_upstream() -> PathBuf {
    // build.rs sits at .../ascend-rs-ds4/crates/ds4_engine/build.rs.
    // Upstream lives at .../ascend-rs-priv/benchmarks/ds4_msl/upstream/ds4.
    if let Ok(p) = env::var("DS4_UPSTREAM_DIR") {
        return PathBuf::from(p);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Self-contained / OSS layout: antirez vendored in-repo at
    // <repo>/benchmarks/ds4_msl/upstream/ds4 (crates/ds4_engine -> crates -> repo).
    if let Some(repo) = manifest.parent().and_then(Path::parent) {
        let vendored = repo.join("benchmarks/ds4_msl/upstream/ds4");
        if vendored.join("ds4.c").exists() {
            return vendored;
        }
    }
    // Monorepo dev fallback: sibling ascend-rs-priv checkout.
    // crates/ds4_engine -> crates -> ascend-rs-ds4 -> $HOME
    let home = manifest
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("manifest path has 3 parents")
        .to_path_buf();
    home.join("ascend-rs-priv/benchmarks/ds4_msl/upstream/ds4")
}

#[allow(dead_code)]
fn verify_pinned(upstream: &Path) {
    use std::fs::File;
    use std::io::Read;

    // M4 #273: allow temporarily patched upstream for the per-layer cur_hc
    // dump bisect. Set DS4_ALLOW_PATCHED_UPSTREAM=1 in env to bypass.
    println!("cargo:rerun-if-env-changed=DS4_ALLOW_PATCHED_UPSTREAM");
    if env::var("DS4_ALLOW_PATCHED_UPSTREAM").ok().as_deref() == Some("1") {
        println!("cargo:warning=ds4_engine: DS4_ALLOW_PATCHED_UPSTREAM=1 — sha256 pin SKIPPED");
        return;
    }

    for (name, expected) in PINNED_HASHES {
        let path = upstream.join(name);
        let mut f = match File::open(&path) {
            Ok(f) => f,
            Err(e) => panic!(
                "ds4_engine/build.rs: missing upstream file {} ({e})\n\
                 Expected commit {UPSTREAM_COMMIT} at {}",
                path.display(),
                upstream.display()
            ),
        };
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).expect("read upstream");
        let got = sha256_hex(&buf);
        if got != *expected {
            panic!(
                "ds4_engine/build.rs: upstream file {} drifted.\n\
                 expected: {}\n\
                 got:      {}\n\
                 Pinned upstream commit is {UPSTREAM_COMMIT}. If you bumped\n\
                 upstream, update PINNED_HASHES in build.rs and re-audit\n\
                 prefill_ffi.rs against ds4.h before shipping.",
                path.display(),
                expected,
                got,
            );
        }
    }
}

// Minimal SHA-256 implementation to avoid a build-time dep. ~40 LOC.
#[allow(dead_code)]
fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut s = String::with_capacity(64);
    for byte in digest {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[allow(dead_code)]
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes(b.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}
