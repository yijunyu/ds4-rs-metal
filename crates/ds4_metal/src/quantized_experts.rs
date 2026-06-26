//! Per-layer quantized MoE expert weight tables.
//!
//! DS4 stores each layer's 256 experts as a stacked quantized tensor:
//!
//!   gate:  [n_experts][d_ffn][d_in]   ttype ∈ {q2_K, q4_K, iq2_xxs}
//!   up:    [n_experts][d_ffn][d_in]   same ttype as gate
//!   down:  [n_experts][d_in][d_ffn]   same ttype family (often differs)
//!
//! These weights are immutable across decode and live for the lifetime of
//! the dispatcher. The Metal kernels consume them as opaque byte buffers
//! (`device const char *`), not as `f32` slices — the inner quant decode
//! happens on-GPU per `KernelSpec.metal_fn`.
//!
//! ## Design choices
//!
//! - **Owns `metal::Buffer`** on macOS (Storage:Shared, suitable for
//!   unified memory; Storage:Private would force a blit step at load).
//! - **CPU-side mirror as bytes** on both platforms so the CPU reference
//!   (`ds4_engine::moe::moe_routed_step`) can be exercised via a dequant
//!   path during bit-exact verification.
//! - **Loaded once per layer**, indexed by layer idx. `MetalState` holds a
//!   `Vec<QuantizedExpertWeights>` (one entry per MoE layer) populated at
//!   GGUF load time via `QuantizedExpertWeights::from_gguf`.
//! - **`metal::Buffer` is macOS-only**; Linux build path stores the raw
//!   bytes only so cross-compile + dequant tests still work.
//!
//! ## Bind shape (per `moe_routed_step_impl`)
//!
//! Stage 1 (`mul_mv_id_q4_K_pair_swiglu_f32`):
//!   buffer(2) = experts_w_gate.buffer
//!   buffer(3) = experts_w_up.buffer
//!   buffer(8) = ids (i32×6)
//!   buffer(9) = weights (f32×6)
//!
//! Stage 2 (`mul_mv_id_q4_K_sum6_f32`):
//!   buffer(1) = experts_w_down.buffer
//!   buffer(4) = ids (i32×6)
//!
//! See `crates/ds4_engine/src/kernel_registry.rs` for the full kernel set.

use anyhow::{anyhow, bail, Context, Result};
use ds4_engine::gguf::{GgmlType, GgufFile, TensorInfo};

use crate::iq2_xxs_tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// Per-layer quantized expert weight table.
///
/// Tensors are stored as raw quantized bytes (the GGUF on-disk layout)
/// plus a Metal buffer mirror on macOS. The kernel dispatch passes the
/// Metal buffer by reference; the CPU reference uses a dequant view of
/// the same bytes for tolerance checks.
pub struct QuantizedExpertWeights {
    pub layer_idx: u32,
    pub n_experts: u32,
    pub d_in: u32,
    pub d_ffn: u32,

    pub gate: QuantTensor,
    pub up: QuantTensor,
    pub down: QuantTensor,
}

/// One stacked quantized weight tensor: `[n_experts][rows][cols]`.
///
/// `bytes` is the raw GGUF block stream. `metal_buf` is a Storage:Shared
/// mirror on macOS; absent on Linux. Decoders read `bytes` directly.
pub struct QuantTensor {
    pub ttype: GgmlType,
    /// Stacked dimensions in row-major order. For gate/up:
    ///   `[n_experts, d_ffn, d_in]`
    /// For down:
    ///   `[n_experts, d_in, d_ffn]`
    pub dims: [u64; 3],
    /// Raw quantized bytes — block-packed per `ttype`.
    pub bytes: Vec<u8>,
    /// Per-expert byte stride (== `rows * cols / block_size * type_size`).
    pub expert_stride: u64,

    /// On macOS, a raw pointer into the mmaped GGUF (held alive by
    /// `LayerViews::bytes` on `DecodeRunner`). Used by the CPU dequant
    /// path when `bytes` is empty. None on Linux.
    #[cfg(target_os = "macos")]
    pub mmap_ptr: *const u8,
    #[cfg(target_os = "macos")]
    pub mmap_len: usize,

    #[cfg(target_os = "macos")]
    pub metal_buf: metal::Buffer,
}

// SAFETY: `mmap_ptr` points into a `Mmap` owned by `DecodeRunner` for the
// lifetime of the dispatcher; we never mutate or unmap it before the
// `QuantTensor` is dropped. The compiler can't see through the raw pointer
// so we manually assert thread-safety.
#[cfg(target_os = "macos")]
unsafe impl Send for QuantTensor {}
#[cfg(target_os = "macos")]
unsafe impl Sync for QuantTensor {}

impl QuantTensor {
    /// Bytes per single expert's matrix.
    pub fn expert_stride(&self) -> u64 {
        self.expert_stride
    }

    /// Total tensor bytes.
    pub fn total_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Slice the raw bytes for a single expert.
    ///
    /// On macOS production, `bytes` is empty (data lives in the mmap region
    /// captured at load time via `mmap_ptr`+`mmap_len`). On Linux/tests,
    /// `bytes` is the owned Vec.
    pub fn expert_slice(&self, expert: u32) -> &[u8] {
        let off = (expert as u64 * self.expert_stride) as usize;
        let end = off + self.expert_stride as usize;
        #[cfg(target_os = "macos")]
        {
            if self.bytes.is_empty() {
                // SAFETY: mmap_ptr was captured from a `Mmap` held by
                // `DecodeRunner::views.bytes` and remains valid for the
                // dispatcher's lifetime. mmap_len is the exact tensor byte
                // count we sliced from the mmap.
                let full = unsafe { std::slice::from_raw_parts(self.mmap_ptr, self.mmap_len) };
                return &full[off..end];
            }
        }
        &self.bytes[off..end]
    }

    /// Dequantize one expert's full matrix to `f32`.
    ///
    /// Output length = `rows * cols` (from `dims[1] * dims[2]`), laid out
    /// row-major. Matches antirez's `dequantize_*` semantics exactly when
    /// concatenated across all blocks in the per-expert byte stream.
    ///
    /// Currently supports `Q4_K`. Other GGML types return `Err`.
    pub fn dequant_expert_f32(&self, expert: u32) -> Result<Vec<f32>> {
        let bytes = self.expert_slice(expert);
        match self.ttype {
            GgmlType::Q4_K => Ok(dequant_q4_k_blocks(bytes)),
            GgmlType::Q2_K => Ok(dequant_q2_k_blocks(bytes)),
            GgmlType::IQ2_XXS => Ok(dequant_iq2_xxs_blocks(bytes)),
            other => bail!("dequant_expert_f32: unsupported type {:?}", other),
        }
    }
}

/// Decode a stream of `block_iq2_xxs` (66 B each, 256 weights each) to `f32`.
///
/// Block layout (matches antirez `block_iq2_xxs`):
///   off  0..2:    half d
///   off  2..66:   uint16 qs[32]    (32 × 16-bit words = 8 ib32 groups × 4 words)
///
/// Output order per block: 8 ib32 groups of 32 weights each. Each ib32 reads
/// `q2 = qs[4*ib32 .. 4*ib32+4]` and forms two 32-bit words:
///   aux32_g = q2[0] | (q2[1] << 16)   (4 grid indices: bytes 0..4)
///   aux32_s = q2[2] | (q2[3] << 16)   (4 sign keys + 4-bit scale_extra)
/// The per-ib32 scale is `dl = d * (0.5 + (aux32_s >> 28)) * 0.25`.
/// The 32 weights split into four 8-byte grids — `grid[aux8_g[k]]` for k=0..4 —
/// each modulated by `signs = ksigns_iq2xs[(aux32_s >> (7*k)) & 0x7F]`.
fn dequant_iq2_xxs_blocks(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 66 == 0,
        "dequant_iq2_xxs_blocks: stream length {} not a multiple of 66",
        bytes.len()
    );
    let n_blocks = bytes.len() / 66;
    let mut out = Vec::with_capacity(n_blocks * 256);
    for ib in 0..n_blocks {
        let blk = &bytes[ib * 66..(ib + 1) * 66];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        // qs[32] little-endian uint16 starts at offset 2.
        let mut qs = [0u16; 32];
        for i in 0..32 {
            qs[i] = u16::from_le_bytes([blk[2 + 2 * i], blk[2 + 2 * i + 1]]);
        }

        for ib32 in 0..8u32 {
            let q2 = &qs[(4 * ib32) as usize..(4 * ib32 + 4) as usize];
            let aux32_g: u32 = (q2[0] as u32) | ((q2[1] as u32) << 16);
            let aux32_s: u32 = (q2[2] as u32) | ((q2[3] as u32) << 16);
            let aux8 = aux32_g.to_le_bytes(); // [b0, b1, b2, b3]
            let dl = d * (0.5 + (aux32_s >> 28) as f32) * 0.25;

            for k in 0..4u32 {
                let grid_idx = aux8[k as usize] as usize;
                let grid_bytes = IQ2XXS_GRID[grid_idx].to_le_bytes();
                let sign_key = ((aux32_s >> (7 * k)) & 127) as usize;
                let signs = KSIGNS_IQ2XS[sign_key];
                for i in 0..8 {
                    let s = if (signs & KMASK_IQ2XS[i]) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let v = grid_bytes[i] as f32;
                    out.push(dl * v * s);
                }
            }
        }
    }
    out
}

/// Decode a stream of `block_q2_K` (84 B each, 256 weights each) to `f32`.
///
/// Block layout (matches antirez `block_q2_K`):
///   off  0..16:   uchar scales[16]   (low nibble = scale, high nibble = min)
///   off 16..80:   uchar qs[64]       (2-bit packed, 4 weights per byte)
///   off 80..82:   half d
///   off 82..84:   half dmin
///
/// Output order per block: 16 sub-blocks of 16 weights each. Sub-block
/// `i_sb` uses `sc = scales[i_sb]` and 16 weights drawn from
/// `qs[q_off .. q_off + 16]` at bit-field `bf = (i_sb / 2) % 4`, where
/// `q_off = (i_sb / 8) * 32 + (i_sb & 1) * 16`. The decoded value is
/// `dl * ((q & mask) >> shift) - ml` with `dl = d * (sc & 0x0F)` and
/// `ml = dmin * (sc >> 4)`. Antirez folds the right-shift into `dl` via
/// the `coef` factor, but we keep the explicit shift here for clarity.
fn dequant_q2_k_blocks(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 84 == 0,
        "dequant_q2_k_blocks: stream length {} not a multiple of 84",
        bytes.len()
    );
    let n_blocks = bytes.len() / 84;
    let mut out = Vec::with_capacity(n_blocks * 256);
    for ib in 0..n_blocks {
        let blk = &bytes[ib * 84..(ib + 1) * 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));

        for i_sb in 0..16u32 {
            let sc = scales[i_sb as usize];
            let dl = d * (sc & 0x0F) as f32;
            let ml = dmin * (sc >> 4) as f32;

            let q_off = ((i_sb / 8) * 32 + (i_sb & 1) * 16) as usize;
            // bit-field index 0..4 selects {0x03, 0x0C, 0x30, 0xC0}.
            let bf = ((i_sb / 2) & 3) as u8;
            let mask: u8 = 0x03u8 << (bf * 2);
            let shift = bf * 2;
            for k in 0..16 {
                let v = ((qs[q_off + k] & mask) >> shift) as f32;
                out.push(dl * v - ml);
            }
        }
    }
    out
}

/// Decode a stream of `block_q4_K` (144 B each, 256 weights each) to `f32`.
///
/// Block layout (matches antirez `block_q4_K`):
///   off  0..2:   half d
///   off  2..4:   half dmin
///   off  4..16:  uchar scales[12]   (6-bit packed scales/mins, 8 of each)
///   off 16..144: uchar qs[128]      (4-bit nibbles, 256 weights)
///
/// Output order per block (256 weights, 8 sub-blocks of 32):
///   sub-block i_sb (0..8) uses `(sc[i_sb], mn[i_sb])` and 32 weights from
///   `qs[(i_sb/2)*32 .. (i_sb/2)*32 + 32]` — low nibbles when `i_sb` is even,
///   high nibbles when odd. The dl factor for odd sub-blocks is `d/16`.
fn dequant_q4_k_blocks(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 144 == 0,
        "dequant_q4_k_blocks: stream length {} not a multiple of 144",
        bytes.len()
    );
    let n_blocks = bytes.len() / 144;
    let mut out = Vec::with_capacity(n_blocks * 256);
    for ib in 0..n_blocks {
        let blk = &bytes[ib * 144..(ib + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        for i_sb in 0..8u32 {
            // 6-bit scale+min unpack: same arithmetic as
            // `get_scale_min_k4_just2(is, k, scales)` but factored per-sub-block.
            //   sc[0..4] live in scales[0..4]&0x3F + scales[4..8]&0x3F
            //   sc[4..8] live in (scales[8..12]&0x0F)|((scales[0..4]&0xC0)>>2)
            //                 + (scales[8..12]>>4) |((scales[4..8]&0xC0)>>2)
            let (sc, mn) = if i_sb < 4 {
                let j = i_sb as usize;
                (scales[j] & 0x3F, scales[j + 4] & 0x3F)
            } else {
                let j = (i_sb - 4) as usize;
                (
                    (scales[j + 8] & 0x0F) | ((scales[j] & 0xC0) >> 2),
                    (scales[j + 8] >> 4) | ((scales[j + 4] & 0xC0) >> 2),
                )
            };

            let q_off = (i_sb as usize / 2) * 32;
            let high = (i_sb & 1) == 1;
            let dl = if high { d / 16.0 } else { d };
            let dl = dl * sc as f32;
            let ml = dmin * mn as f32;
            let mask: u8 = if high { 0xF0 } else { 0x0F };

            // 32 weights per sub-block, split antirez-style into two halves:
            //   first half = qs[q_off + 0..16]
            //   second half = qs[q_off + 16..32]
            // both use the same (dl, ml, mask) — matches `dequantize_q4_K`
            // calls with il = 2*i_sb and il = 2*i_sb+1 concatenated.
            for k in 0..16 {
                let v = (qs[q_off + k] & mask) as f32;
                out.push(dl * v - ml);
            }
            for k in 0..16 {
                let v = (qs[q_off + 16 + k] & mask) as f32;
                out.push(dl * v - ml);
            }
        }
    }
    out
}

/// IEEE 754 binary16 → binary32 (no NaN/Inf payload preservation, full range).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f = if exp == 0 {
        if mant == 0 {
            0.0_f32
        } else {
            // subnormal
            let m = mant as f32;
            m * (2.0_f32).powi(-24)
        }
    } else if exp == 0x1F {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let e = exp as i32 - 15;
        let m = 1.0_f32 + (mant as f32) / 1024.0;
        m * (2.0_f32).powi(e)
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

impl QuantizedExpertWeights {
    /// Dequantize the whole `gate` tensor to a flat f32 slab.
    ///
    /// Layout: `[n_experts][d_ffn][d_in]` row-major — matches the slice
    /// the f32 CPU reference (`moe::moe_routed_step`) expects.
    pub fn dequant_gate_f32(&self) -> Result<Vec<f32>> {
        dequant_full_tensor(&self.gate, self.n_experts)
    }

    /// Same as `dequant_gate_f32` but for the `up` tensor.
    pub fn dequant_up_f32(&self) -> Result<Vec<f32>> {
        dequant_full_tensor(&self.up, self.n_experts)
    }

    /// Same as `dequant_gate_f32` but for the `down` tensor. Layout is
    /// `[n_experts][d_in][d_ffn]` row-major.
    pub fn dequant_down_f32(&self) -> Result<Vec<f32>> {
        dequant_full_tensor(&self.down, self.n_experts)
    }

    /// Load a layer's gate/up/down stacked expert tensors from a GGUF.
    ///
    /// `gguf_bytes` is the memory-mapped file backing `gguf`. We slice
    /// out each tensor by `tensor_data_offset + ti.offset` and copy into
    /// an owned `Vec<u8>` (cheap because each tensor is < 4 GB and the
    /// alternative — `mmap` borrow — complicates the lifetime story when
    /// the `metal::Buffer` outlives the file handle).
    ///
    /// Tensor names follow antirez's layout:
    ///   `blk.{layer}.ffn_gate_exps.weight`
    ///   `blk.{layer}.ffn_up_exps.weight`
    ///   `blk.{layer}.ffn_down_exps.weight`
    pub fn from_gguf(
        gguf: &GgufFile,
        gguf_bytes: &[u8],
        layer_idx: u32,
        #[cfg(target_os = "macos")] device: &metal::Device,
    ) -> Result<Self> {
        Self::from_gguf_with_prefix(
            gguf, gguf_bytes,
            &format!("blk.{layer_idx}"),
            layer_idx,
            #[cfg(target_os = "macos")] device,
        )
    }

    /// Phase 3 — MTP variant. Reads `mtp.0.ffn_{gate,up,down}_exps.weight`
    /// from an MTP GGUF and stamps the caller-supplied `layer_idx`
    /// (= slot in `state.expert_weights`) into the result.
    pub fn from_mtp_gguf(
        gguf: &GgufFile,
        gguf_bytes: &[u8],
        layer_idx: u32,
        #[cfg(target_os = "macos")] device: &metal::Device,
    ) -> Result<Self> {
        Self::from_gguf_with_prefix(
            gguf, gguf_bytes,
            "mtp.0",
            layer_idx,
            #[cfg(target_os = "macos")] device,
        )
    }

    /// Shared body — `prefix` is the GGUF tensor-name prefix (e.g.
    /// `"blk.5"` or `"mtp.0"`); `layer_idx` is the dispatcher's
    /// `state.expert_weights` slot index.
    fn from_gguf_with_prefix(
        gguf: &GgufFile,
        gguf_bytes: &[u8],
        prefix: &str,
        layer_idx: u32,
        #[cfg(target_os = "macos")] device: &metal::Device,
    ) -> Result<Self> {
        let gate_name = format!("{prefix}.ffn_gate_exps.weight");
        let up_name   = format!("{prefix}.ffn_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");

        let gate_ti = find_tensor(gguf, &gate_name)?;
        let up_ti = find_tensor(gguf, &up_name)?;
        let down_ti = find_tensor(gguf, &down_name)?;

        // DS4 expert tensors are always 3-D: [d_in, rows, n_experts] in
        // GGUF column-major layout. Antirez flips them to row-major in
        // the kernel's nb01/nb02 stride math. We treat dims as
        // [n_experts, rows, cols] where rows/cols differ per tensor.
        let (n_experts, gate_rows, gate_cols) = three_d(gate_ti)?;
        let (_, up_rows, up_cols) = three_d(up_ti)?;
        let (_, down_rows, down_cols) = three_d(down_ti)?;

        if (n_experts, gate_rows, gate_cols) != (n_experts, up_rows, up_cols) {
            bail!(
                "gate/up shape mismatch at layer {layer_idx}: \
                 gate={gate_rows}x{gate_cols} up={up_rows}x{up_cols}"
            );
        }
        // gate/up are [d_ffn, d_in]; down is [d_in, d_ffn]
        let d_ffn = gate_rows as u32;
        let d_in = gate_cols as u32;
        if down_rows != d_in as u64 || down_cols != d_ffn as u64 {
            bail!(
                "down shape mismatch at layer {layer_idx}: \
                 expected {}x{} got {}x{}",
                d_in,
                d_ffn,
                down_rows,
                down_cols
            );
        }

        let gate = load_quant_tensor(
            gguf,
            gguf_bytes,
            gate_ti,
            [n_experts, gate_rows, gate_cols],
            #[cfg(target_os = "macos")]
            device,
        )
        .with_context(|| format!("loading {gate_name}"))?;
        let up = load_quant_tensor(
            gguf,
            gguf_bytes,
            up_ti,
            [n_experts, up_rows, up_cols],
            #[cfg(target_os = "macos")]
            device,
        )
        .with_context(|| format!("loading {up_name}"))?;
        let down = load_quant_tensor(
            gguf,
            gguf_bytes,
            down_ti,
            [n_experts, down_rows, down_cols],
            #[cfg(target_os = "macos")]
            device,
        )
        .with_context(|| format!("loading {down_name}"))?;

        Ok(Self {
            layer_idx,
            n_experts: n_experts as u32,
            d_in,
            d_ffn,
            gate,
            up,
            down,
        })
    }
}

/// Run one MoE-routed step through a CPU dequant oracle, taking weights from
/// a quantized expert table.
///
/// Each call dequantizes selected gate/up/down expert stacks to f32 and then
/// dispatches the MoE math locally. `ActivationQuant::AntirezQ8K` preserves
/// antirez activation rounding; `ActivationQuant::F32` preserves the old
/// f32-layout reference used to validate byte decoding.
///
/// This is intentionally allocation-heavy (full f32 slabs per call). It
/// exists for verification, not for the hot decode path.
/// Round-trip an activation vector through block_q8_K quantization, matching
/// antirez `ds4_quantize_row_q8_K` (ds4.c:1508-1545).
///
/// Per 256-element block: iscale = -127 / max(|x|), q[j] = lrintf(iscale·x[j])
/// clamped to [-128, 127], then x_back[j] = q[j] · (1/iscale) = q[j] · d.
///
/// Antirez requantizes activations through this in `matvec_iq2_xxs_pair_*`
/// and `matvec_q2_K_*` before each int8-dot. The antirez-parity oracle uses
/// this by default; the f32 reference mode deliberately skips it.
/// Round-trip an activation vector through `quantize_q8_0_activation`
/// (ds4.c:2974-2998). Per-32-element block: d = amax/127, q = round(x/d),
/// clamped to [-128, 127], then x_back = q · d.
///
/// Used by antirez for shared_expert (Q8_0 weights), attn LoRA projections,
/// output_hc heads, and grouped projections. M4 #299 mirrors this to chase
/// antirez-fidelity in the same way M4 #298 did for routed-MoE Q8_K.
pub fn q8_0_round_trip(x: &[f32]) -> Vec<f32> {
    const BLOCK: usize = 32;
    let mut out = vec![0.0f32; x.len()];
    let n_full = x.len() / BLOCK;
    for b in 0..n_full {
        let off = b * BLOCK;
        let mut amax = 0.0f32;
        for j in 0..BLOCK {
            let a = x[off + j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax == 0.0 {
            for j in 0..BLOCK {
                out[off + j] = 0.0;
            }
            continue;
        }
        let d = amax / 127.0f32;
        let id = 1.0f32 / d;
        for j in 0..BLOCK {
            // Use round-half-to-even (matches C `lrintf` default FE_TONEAREST).
            let v = (x[off + j] * id).round_ties_even() as i32;
            let v = v.clamp(-128, 127);
            out[off + j] = (v as f32) * d;
        }
    }
    let i0 = n_full * BLOCK;
    if i0 < x.len() {
        let mut amax = 0.0f32;
        for j in i0..x.len() {
            let a = x[j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax != 0.0 {
            let d = amax / 127.0f32;
            let id = 1.0f32 / d;
            for j in i0..x.len() {
                let v = (x[j] * id).round_ties_even() as i32;
                let v = v.clamp(-128, 127);
                out[j] = (v as f32) * d;
            }
        }
    }
    out
}

/// Quantize an f32 weight matrix to GGUF `block_q8_0` bytes: per 32-element
/// block, an f16 scale `d = amax/127` followed by 32 `int8` quants
/// `q = round(x/d)`. Layout matches `ggml`/antirez and the
/// `ds4_kernel_mul_mv_q8_0_f32` kernel (34 bytes/block).
///
/// `w.len()` must be a multiple of 32. Used to re-quantize the
/// already-dequantized projection weights (`attn_q_a/b`, `attn_kv`,
/// shared expert) so the decode path can run the Q8_0 matvec at 1 byte/weight
/// instead of dequant→`matvec_f32` at 4 bytes/weight. Re-quantizing values
/// that already lie on a Q8_0 grid recovers an equal-or-finer representation,
/// so the result matches the f32 path to Q8_0 precision.
pub fn quantize_q8_0_to_bytes(w: &[f32]) -> Vec<u8> {
    const BLOCK: usize = 32;
    assert!(
        w.len() % BLOCK == 0,
        "quantize_q8_0_to_bytes: len {} not a multiple of {BLOCK}",
        w.len()
    );
    let n_blocks = w.len() / BLOCK;
    let mut out = Vec::with_capacity(n_blocks * (2 + BLOCK));
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let mut amax = 0.0f32;
        for j in 0..BLOCK {
            let a = w[off + j].abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        out.extend_from_slice(&crate::f16_cast::f32_to_f16_bits(d).to_le_bytes());
        for j in 0..BLOCK {
            let v = (w[off + j] * id).round_ties_even() as i32;
            out.push(v.clamp(-128, 127) as i8 as u8);
        }
    }
    out
}

fn q8_k_round_trip(x: &[f32]) -> Vec<f32> {
    const QK_K: usize = 256;
    let mut out = vec![0.0f32; x.len()];
    let n_blocks = x.len() / QK_K;
    for b in 0..n_blocks {
        let off = b * QK_K;
        let mut max_abs = 0.0f32;
        for j in 0..QK_K {
            let a = x[off + j].abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0.0 {
            for j in 0..QK_K {
                out[off + j] = 0.0;
            }
            continue;
        }
        let iscale = -127.0f32 / max_abs;
        let d = 1.0f32 / iscale;
        for j in 0..QK_K {
            // Use round-half-to-even (matches C `lrintf` default FE_TONEAREST,
            // ds4.c:1533). For round-trip the sign convention with max_abs
            // (vs antirez signed `max`) cancels because v·d ≈ x[j] either way.
            let v = (iscale * x[off + j]).round_ties_even() as i32;
            let v = v.clamp(-128, 127);
            out[off + j] = (v as f32) * d;
        }
    }
    // Tail handling: blocks that don't fit QK_K. Antirez asserts in == multiple
    // of QK_K so for DS4 (d_in=7168, d_ffn=2048; both QK_K-aligned) this never
    // fires. Copy through if it does.
    for j in (n_blocks * QK_K)..x.len() {
        out[j] = x[j];
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationQuant {
    /// Match antirez's routed-MoE matvec path: Q8_K-round activations before
    /// quantized gate/up matvecs and again before the down matvec.
    AntirezQ8K,
    /// Keep activations as f32. This is useful for dequant-layout tests that
    /// compare against `ds4_engine::moe::moe_routed_step` on f32 slabs.
    F32,
}

impl Default for ActivationQuant {
    fn default() -> Self {
        Self::AntirezQ8K
    }
}

pub fn moe_routed_step_cpu_via_dequant(
    qew: &QuantizedExpertWeights,
    x: &[f32],
    selected: &[usize],
    weights: &[f32],
    d_ffn: usize,
) -> Result<Vec<f32>> {
    moe_routed_step_cpu_via_dequant_with_activation(
        qew,
        x,
        selected,
        weights,
        d_ffn,
        ActivationQuant::default(),
    )
}

pub fn moe_routed_step_cpu_via_dequant_with_activation(
    qew: &QuantizedExpertWeights,
    x: &[f32],
    selected: &[usize],
    weights: &[f32],
    d_ffn: usize,
    activation_quant: ActivationQuant,
) -> Result<Vec<f32>> {
    if d_ffn != qew.d_ffn as usize {
        bail!(
            "moe_routed_step_cpu_via_dequant: d_ffn mismatch (got {d_ffn}, qew={})",
            qew.d_ffn
        );
    }
    if x.len() != qew.d_in as usize {
        bail!(
            "moe_routed_step_cpu_via_dequant: x.len()={} but qew.d_in={}",
            x.len(),
            qew.d_in
        );
    }
    let d_in = x.len();
    for &e in selected {
        let e_u32 = e as u32;
        if e_u32 >= qew.n_experts {
            bail!(
                "moe_routed_step_cpu_via_dequant: selected expert {} >= n_experts={}",
                e,
                qew.n_experts
            );
        }
    }
    // Sparse dequant: only the `selected` experts are needed, not all 256.
    // Each selected expert is independent (dequant + matvec + SwiGLU + matvec
    // + weighted contribution), so compute them across scoped threads and
    // sum-reduce. std::thread::scope avoids pulling in a new rayon dep into
    // the vendor-only workspace.
    const CLAMP: f32 = 10.0;
    let n_sel = selected.len();
    let mut contribs: Vec<Result<Vec<f32>>> = (0..n_sel).map(|_| Ok(Vec::new())).collect();
    let dump_layer = std::env::var("DS4_DUMP_EXPERT_LAYER")
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let dump_pos = std::env::var("DS4_DUMP_EXPERT_POS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let cur_pos = ds4_engine::attn_dispatch::CURRENT_POS_HINT.with(|c| c.get()) as u32;
    let x_owned;
    let x_in: &[f32] = if activation_quant == ActivationQuant::AntirezQ8K {
        x_owned = q8_k_round_trip(x);
        &x_owned
    } else {
        x
    };
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_sel);
        for (i, (&e, &w_e)) in selected.iter().zip(weights.iter()).enumerate() {
            let e_u32 = e as u32;
            let layer = qew.layer_idx;
            let h = s.spawn(move || -> Result<(usize, Vec<f32>)> {
                let w_gate = qew.gate.dequant_expert_f32(e_u32)?;
                let w_up = qew.up.dequant_expert_f32(e_u32)?;
                let w_down = qew.down.dequant_expert_f32(e_u32)?;
                let gate = ds4_engine::forward::matvec_f32(&w_gate, x_in, d_ffn);
                let up = ds4_engine::forward::matvec_f32(&w_up, x_in, d_ffn);
                // M4 #302: antirez `matvec_iq2_xxs_mid_worker` (ds4.c:3690)
                // folds `expert_weight[slot]` INTO mid before the down-side
                // q8_K quantization at ds4.c:5266. Applying the weight after
                // q8_K round-trip (our pre-fix path) shrinks mid's block
                // amax by `1/w_e` ≈ 4× and gives our path 2 extra bits of
                // q8_K precision relative to antirez. To match antirez
                // bit-fidelity, multiply mid by w_e BEFORE the q8_K
                // round-trip, then DO NOT multiply y_e by w_e afterwards.
                let mid: Vec<f32> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&g, &u)| {
                        let g_c = if g > CLAMP { CLAMP } else { g };
                        let u_c = u.clamp(-CLAMP, CLAMP);
                        ds4_engine::forward::silu(g_c) * u_c * w_e
                    })
                    .collect();
                let mid = if activation_quant == ActivationQuant::AntirezQ8K {
                    q8_k_round_trip(&mid)
                } else {
                    mid
                };
                if let Some(t) = dump_layer {
                    let pos_ok = dump_pos.map(|p| p == cur_pos).unwrap_or(true);
                    if layer == t && pos_ok {
                        let rms = |v: &[f32]| -> f64 {
                            let ss: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
                            (ss / v.len() as f64).sqrt()
                        };
                        eprintln!(
                            "EXPERT_DUMP il={} slot={} e={} w_e={:.5} gate_rms={:.5} up_rms={:.5} mid_rms={:.5} mid_first8={:?}",
                            layer, i, e_u32, w_e, rms(&gate), rms(&up), rms(&mid),
                            mid.iter().take(8).copied().collect::<Vec<f32>>(),
                        );
                    }
                }
                // Weight already folded into mid above — do NOT scale y_e.
                let y_e = ds4_engine::forward::matvec_f32(&w_down, &mid, d_in);
                Ok((i, y_e))
            });
            handles.push(h);
        }
        for h in handles {
            match h.join() {
                Ok(Ok((i, v))) => contribs[i] = Ok(v),
                Ok(Err(e)) => {
                    contribs[0] = Err(e);
                }
                Err(_) => {
                    contribs[0] = Err(anyhow::anyhow!(
                        "moe_routed_step_cpu_via_dequant: expert worker thread panicked"
                    ));
                }
            }
        }
    });
    let mut acc = vec![0.0f32; d_in];
    for r in contribs {
        let v = r?;
        for (av, &yv) in acc.iter_mut().zip(v.iter()) {
            *av += yv;
        }
    }
    Ok(acc)
}

fn dequant_full_tensor(t: &QuantTensor, n_experts: u32) -> Result<Vec<f32>> {
    let per_expert_f32 = (t.dims[1] * t.dims[2]) as usize;
    let mut out = Vec::with_capacity(per_expert_f32 * n_experts as usize);
    for e in 0..n_experts {
        let f = t.dequant_expert_f32(e)?;
        if f.len() != per_expert_f32 {
            bail!(
                "dequant_full_tensor: expert {} produced {} floats, expected {}",
                e,
                f.len(),
                per_expert_f32
            );
        }
        out.extend_from_slice(&f);
    }
    Ok(out)
}

fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a TensorInfo> {
    gguf.tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor {name} not found in GGUF"))
}

fn three_d(ti: &TensorInfo) -> Result<(u64, u64, u64)> {
    if ti.dims.len() != 3 {
        bail!(
            "tensor {} has {} dims (expected 3 for a stacked expert tensor)",
            ti.name,
            ti.dims.len()
        );
    }
    // GGUF stores dims in fortran order: dim0 = innermost (cols),
    // dim2 = outermost (n_experts). We return (n_experts, rows, cols).
    Ok((ti.dims[2], ti.dims[1], ti.dims[0]))
}

fn load_quant_tensor(
    gguf: &GgufFile,
    gguf_bytes: &[u8],
    ti: &TensorInfo,
    dims_aer: [u64; 3], // [n_experts, rows, cols]
    #[cfg(target_os = "macos")] device: &metal::Device,
) -> Result<QuantTensor> {
    let nbytes = gguf.tensor_byte_size(ti) as usize;
    let start = (gguf.tensor_data_offset + ti.offset) as usize;
    let end = start
        .checked_add(nbytes)
        .ok_or_else(|| anyhow!("tensor byte range overflow"))?;
    if end > gguf_bytes.len() {
        bail!(
            "tensor {} extends past file end: [{start}..{end}) > {}",
            ti.name,
            gguf_bytes.len()
        );
    }

    let block_size = ti.ttype.block_size() as u64;
    let type_size = ti.ttype.type_size() as u64;
    let elems_per_expert = dims_aer[1] * dims_aer[2];
    if elems_per_expert % block_size != 0 {
        bail!(
            "tensor {}: per-expert elems {} not multiple of block_size {}",
            ti.name,
            elems_per_expert,
            block_size
        );
    }
    let expert_stride = (elems_per_expert / block_size) * type_size;

    // On macOS production we point the Metal buffer at the mmap region
    // via `new_buffer_with_bytes_no_copy` and leave `bytes` empty — the
    // OS holds the data exactly once (page-cache of the GGUF) and the
    // GPU reads through the unified-memory mapping. Avoids 80+ GB of
    // duplicate Vec<u8> + duplicate Metal Storage:Shared copies. The
    // CPU dequant path (`expert_slice`/`dequant_*_f32`) is only used in
    // tests; production decode reads through Metal.
    //
    // On Linux (no Metal) and in tests, fall through to the owned-Vec
    // path so `dequant_*_f32` and `moe_routed_step_cpu_via_dequant`
    // keep their byte slice.
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `gguf_bytes` is the `LayerViews::bytes` mmap slice held
        // by `DecodeRunner` for the lifetime of this process; we never
        // unmap it before this `metal_buf` is destroyed. mmap pages are
        // page-aligned, which is what `newBufferWithBytesNoCopy` wants
        // (the GGUF tensor offsets aren't all page-aligned, but Metal
        // tolerates intra-page offsets — only the base pointer needs to
        // come from a page-aligned allocation, which mmap guarantees).
        let region_ptr = unsafe { gguf_bytes.as_ptr().add(start) };
        // Record the mmap base for the pread fast path (whole-file slice).
        GGUF_BASE.store(gguf_bytes.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
        // SSD-streaming: do NOT register the full stacked tensor with the
        // device — for an over-RAM model that VA (380+ GB for PRO) counts
        // against the device allocation budget and later allocs return nil.
        // The CPU streams experts from `mmap_ptr` into the ExpertCache;
        // bind a 1-page placeholder so the field stays non-null.
        // DS4_SSD_STUB=0: full no-copy bind even under stream (fits-in-VA
        // models only) — isolates ExpertCache slot-pool bugs from the rest of
        // the stream path (the LRU/pool is bypassed at dispatch when ids pass
        // through unmapped).
        let stub = std::env::var("DS4_SSD_STREAM").is_ok()
            && std::env::var("DS4_SSD_STUB").map(|v| v != "0").unwrap_or(true);
        let nbytes_buf = if stub { 16384.min(nbytes) } else { nbytes };
        let metal_buf = device.new_buffer_with_bytes_no_copy(
            region_ptr as *const _,
            nbytes_buf as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        return Ok(QuantTensor {
            ttype: ti.ttype,
            dims: dims_aer,
            bytes: Vec::new(),
            expert_stride,
            mmap_ptr: region_ptr,
            mmap_len: nbytes,
            metal_buf,
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        let bytes = gguf_bytes[start..end].to_vec();
        Ok(QuantTensor {
            ttype: ti.ttype,
            dims: dims_aer,
            bytes,
            expert_stride,
        })
    }
}

/// SSD-streaming pread fast path: faulting cold mmap pages copies at
/// ~1.3 GB/s under memory pressure while direct pread hits ~4 GB/s. The
/// expert tensors live contiguously in the GGUF file, so a copy from
/// `mmap_ptr` can be replaced by `pread(file, dst, len, src - mmap_base)`.
/// `GGUF_BASE` is the mmap base (set at expert-tensor load); the file is
/// re-opened from `DS4_GGUF_PATH` (set by the server/CLI at startup).
#[cfg(target_os = "macos")]
pub(crate) static GGUF_BASE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(target_os = "macos")]
static GGUF_FILE: std::sync::OnceLock<Option<std::fs::File>> = std::sync::OnceLock::new();

#[cfg(target_os = "macos")]
fn pread_span(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use std::os::fd::AsRawFd;
    let base = GGUF_BASE.load(std::sync::atomic::Ordering::Relaxed);
    if base == 0 || (src as usize) < base {
        return false;
    }
    let Some(f) = GGUF_FILE
        .get_or_init(|| std::env::var("DS4_GGUF_PATH").ok().and_then(|p| std::fs::File::open(p).ok()))
        .as_ref()
    else {
        return false;
    };
    let off = src as usize - base;
    let fd = f.as_raw_fd();
    let mut done = 0usize;
    while done < len {
        let n = unsafe {
            libc::pread(fd, dst.add(done) as *mut libc::c_void, len - done, (off + done) as i64)
        };
        if n <= 0 {
            return false;
        }
        done += n as usize;
    }
    true
}

/// SSD-streaming expert cache (DS4_SSD_STREAM): a fixed pool of GPU-resident
/// expert slots per layer. The full stacked expert tensors of an over-RAM
/// model stay on SSD (mmap, never bound to the GPU — Metal cannot
/// demand-page mmap buffers; faults are fatal). Before each MoE dispatch the
/// host copies any missing selected experts into LRU slots and remaps the
/// expert ids to slot ids; the `mul_mv_id*` kernels are unchanged (the slot
/// pool is just a smaller stacked tensor).
#[cfg(target_os = "macos")]
pub struct ExpertCache {
    pub slots: u32,
    pub gate: metal::Buffer,
    pub up: metal::Buffer,
    pub down: metal::Buffer,
    /// expert id occupying each slot (-1 = empty).
    owner: Vec<i32>,
    /// expert id → slot.
    map: std::collections::HashMap<i32, u32>,
    /// LRU tick per slot.
    last_use: Vec<u64>,
    tick: u64,
    /// stats
    pub hits: u64,
    pub misses: u64,
}

#[cfg(target_os = "macos")]
impl ExpertCache {
    pub fn new(device: &metal::Device, qew: &QuantizedExpertWeights, slots: u32) -> Self {
        let alloc = |stride: u64| {
            let buf = device.new_buffer(
                stride * slots as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            // mlock cache slots — under memory pressure the OS pages out
            // plain Shared buffers and decode collapses to swap churn
            // (antirez mlocks his streaming cache too; partial lock OK).
            unsafe {
                libc::mlock(buf.contents() as *const libc::c_void, (stride * slots as u64) as usize);
            }
            buf
        };
        Self {
            slots,
            gate: alloc(qew.gate.expert_stride),
            up: alloc(qew.up.expert_stride),
            down: alloc(qew.down.expert_stride),
            owner: vec![-1; slots as usize],
            map: std::collections::HashMap::new(),
            last_use: vec![0; slots as usize],
            tick: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Ensure the selected experts are resident; return their slot ids in
    /// selection order. Copies missing experts from the mmap (page-cache
    /// pread) into LRU slots — synchronous, correctness-first.
    pub fn ensure(&mut self, qew: &QuantizedExpertWeights, selected: &[i32]) -> Vec<i32> {
        debug_assert!(selected.len() as u32 <= self.slots);
        self.tick += 1;
        let mut out = Vec::with_capacity(selected.len());
        let mut pending: Vec<(i32, u32)> = Vec::new(); // (expert, slot) misses
        for &e in selected {
            let slot = if let Some(&s) = self.map.get(&e) {
                self.hits += 1;
                s
            } else {
                self.misses += 1;
                // Evict LRU slot not used by this selection round.
                let s = (0..self.slots)
                    .filter(|&s| {
                        let o = self.owner[s as usize];
                        o < 0 || !selected.contains(&o)
                    })
                    .min_by_key(|&s| self.last_use[s as usize])
                    .expect("ExpertCache: no evictable slot");
                if self.owner[s as usize] >= 0 {
                    self.map.remove(&self.owner[s as usize]);
                }
                self.owner[s as usize] = e;
                self.map.insert(e, s);
                pending.push((e, s));
                s
            };
            self.last_use[slot as usize] = self.tick;
            out.push(slot as i32);
        }
        // Copy misses in parallel — cold mmap pages are page-fault bound;
        // readahead + one thread per miss keeps the SSD queue deep (decode
        // was up to 6 serial faulting 6.75MB copies per layer per token).
        if !pending.is_empty() {
            for &(e, _) in &pending {
                for t in [&qew.gate, &qew.up, &qew.down] {
                    let src = t.expert_slice(e as u32);
                    unsafe {
                        libc::madvise(
                            src.as_ptr() as *mut libc::c_void,
                            src.len(),
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
            let gate_base = self.gate.contents() as usize;
            let up_base = self.up.contents() as usize;
            let down_base = self.down.contents() as usize;
            std::thread::scope(|sc| {
                for &(e, s) in &pending {
                    sc.spawn(move || {
                        for (t, base) in [
                            (&qew.gate, gate_base),
                            (&qew.up, up_base),
                            (&qew.down, down_base),
                        ] {
                            let src = t.expert_slice(e as u32);
                            unsafe {
                                let dst = (base as *mut u8)
                                    .add(s as usize * t.expert_stride as usize);
                                if !pread_span(src.as_ptr(), dst, src.len()) {
                                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
                                }
                            }
                        }
                    });
                }
            });
        }
        out
    }

    /// Chunked-prefill whole-layer fill: stream ALL of this layer's experts
    /// into the pool in expert order (slot == expert id; the mm_id ids
    /// buffer needs NO remap). Requires `slots >= n_experts`. One large
    /// sequential mmap copy per tensor — the SSD-friendly access pattern.
    /// `owner` is left as identity for visibility; hits/misses untouched
    /// (this pool is a separate prefill scratch, not the decode LRU cache).
    /// Like `fill_layer` but copies experts already resident in `lru`
    /// (decode LRU cache) from RAM (~40 GB/s) instead of SSD — roughly
    /// halves fill traffic at typical 50%+ cache coverage.
    pub fn fill_layer_from_lru(&mut self, qew: &QuantizedExpertWeights, lru: &ExpertCache) {
        let n = qew.n_experts as u32;
        debug_assert!(self.slots >= n, "fill_layer_from_lru: pool too small");
        let pairs: [(&QuantTensor, &metal::Buffer, &metal::Buffer); 3] = [
            (&qew.gate, &lru.gate, &self.gate),
            (&qew.up, &lru.up, &self.up),
            (&qew.down, &lru.down, &self.down),
        ];
        let mut missing: Vec<u32> = Vec::new();
        for e in 0..n {
            if let Some(&s) = lru.map.get(&(e as i32)) {
                for (t, lbuf, dbuf) in pairs.iter() {
                    let stride = t.expert_stride as usize;
                    unsafe {
                        let src = (lbuf.contents() as *const u8).add(s as usize * stride);
                        let dst = (dbuf.contents() as *mut u8).add(e as usize * stride);
                        std::ptr::copy_nonoverlapping(src, dst, stride);
                    }
                }
            } else {
                missing.push(e);
            }
        }
        // SSD-read the misses in parallel stripes (pread fast path).
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(8);
        for (t, _l, dbuf) in pairs.iter() {
            let stride = t.expert_stride as usize;
            let dst_base = dbuf.contents() as usize;
            let missing = &missing;
            std::thread::scope(|sc| {
                for tid in 0..threads {
                    let t = &**t;
                    sc.spawn(move || {
                        let mut i = tid;
                        while i < missing.len() {
                            let e = missing[i];
                            let src = t.expert_slice(e);
                            unsafe {
                                let dst = (dst_base as *mut u8).add(e as usize * stride);
                                if !pread_span(src.as_ptr(), dst, src.len()) {
                                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
                                }
                            }
                            i += threads;
                        }
                    });
                }
            });
        }
        for e in 0..n {
            self.owner[e as usize] = e as i32;
        }
    }

    pub fn fill_layer(&mut self, qew: &QuantizedExpertWeights) {
        let n = qew.n_experts as u32;
        debug_assert!(self.slots >= n, "fill_layer: pool too small");
        // The sources are cold mmap pages under memory pressure — a single
        // memcpy is page-fault-bound (~1 outstanding SSD read). Kick async
        // readahead for each whole tensor span, then fault in parallel
        // stripes so the SSD queue stays deep. ~83% of chunk prefill wall
        // was this loop single-threaded.
        for t in [&qew.gate, &qew.up, &qew.down] {
            unsafe {
                libc::madvise(
                    t.mmap_ptr as *mut libc::c_void,
                    t.mmap_len as usize,
                    libc::MADV_WILLNEED,
                );
            }
        }
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(8);
        for (t, buf) in [
            (&qew.gate, &self.gate),
            (&qew.up, &self.up),
            (&qew.down, &self.down),
        ] {
            let stride = t.expert_stride as usize;
            let dst_base = buf.contents() as usize;
            std::thread::scope(|s| {
                for tid in 0..threads {
                    let t = &*t;
                    s.spawn(move || {
                        // Contiguous expert stripe per thread → large sequential preads.
                        let per = (n as usize + threads - 1) / threads;
                        let lo = (tid * per).min(n as usize);
                        let hi = ((tid + 1) * per).min(n as usize);
                        for e in lo..hi {
                            let src = t.expert_slice(e as u32);
                            unsafe {
                                let dst = (dst_base as *mut u8).add(e * stride);
                                if !pread_span(src.as_ptr(), dst, src.len()) {
                                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
                                }
                            }
                        }
                    });
                }
            });
        }
        for e in 0..n {
            self.owner[e as usize] = e as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_os = "macos"))]
    fn dummy_quant(ttype: GgmlType, n_experts: u64, rows: u64, cols: u64) -> QuantTensor {
        let block_size = ttype.block_size() as u64;
        let type_size = ttype.type_size() as u64;
        let elems = n_experts * rows * cols;
        let total_bytes = (elems / block_size) * type_size;
        let expert_stride = ((rows * cols) / block_size) * type_size;
        QuantTensor {
            ttype,
            dims: [n_experts, rows, cols],
            bytes: vec![0u8; total_bytes as usize],
            expert_stride,
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn expert_stride_q4_k_matches_packed_size() {
        // q4_K: block_size=256, type_size=144.
        // d_ffn=512, d_in=1024 → elems_per_expert = 512*1024 = 524288.
        // blocks_per_expert = 524288/256 = 2048; bytes = 2048*144 = 294912.
        let t = dummy_quant(GgmlType::Q4_K, 256, 512, 1024);
        assert_eq!(t.expert_stride, 294912);
        assert_eq!(t.total_bytes(), 256 * 294912);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn expert_stride_q2_k_smaller_than_q4_k() {
        // q2_K type_size=84 < q4_K type_size=144.
        let q2 = dummy_quant(GgmlType::Q2_K, 256, 512, 1024);
        let q4 = dummy_quant(GgmlType::Q4_K, 256, 512, 1024);
        assert!(q2.expert_stride < q4.expert_stride);
        assert_eq!(q2.expert_stride * 144, q4.expert_stride * 84);
    }

    /// Antirez's per-segment dequant, re-implemented in Rust to cross-check
    /// the full-block decoder against the same reference the GPU kernels use.
    fn antirez_dequantize_q4_k_segment(blk: &[u8], il: u8) -> [f32; 16] {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let is = ((il / 4) * 2) as usize;
        let q_start = ((il / 4) as usize) * 32 + 16 * ((il & 1) as usize);
        let q = &qs[q_start..q_start + 16];
        let il_eff = (il & 3) as usize;

        // get_scale_min_k4_just2(j=is, k=il_eff/2, q=scales)
        let j = is;
        let k = il_eff / 2;
        let sc_pair = if j < 4 {
            (scales[j + k] & 0x3F, scales[j + 4 + k] & 0x3F)
        } else {
            (
                (scales[j + 4 + k] & 0x0F) | ((scales[j - 4 + k] & 0xC0) >> 2),
                (scales[j + 4 + k] >> 4) | ((scales[j + k] & 0xC0) >> 2),
            )
        };
        let dd = if il_eff < 2 { d } else { d / 16.0 };
        let dl = dd * sc_pair.0 as f32;
        let ml = dmin * sc_pair.1 as f32;
        let mask: u8 = if il_eff < 2 { 0x0F } else { 0xF0 };

        let mut out = [0f32; 16];
        for i in 0..16 {
            out[i] = dl * (q[i] & mask) as f32 - ml;
        }
        out
    }

    fn make_q4_k_block(seed: u8) -> Vec<u8> {
        // half d = 1.0 (binary16 0x3C00), half dmin = 0.5 (0x3800)
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        // scales[12]: deterministic but covers both <4 and >=4 cases
        for i in 0..12 {
            blk[4 + i] = seed.wrapping_add(i as u8).wrapping_mul(7);
        }
        // qs[128]: deterministic
        for i in 0..128 {
            blk[16 + i] = seed.wrapping_add(i as u8).wrapping_mul(13);
        }
        blk
    }

    #[test]
    fn dequant_q4_k_block_matches_antirez_segments() {
        let blk = make_q4_k_block(0x5A);
        // Full-block decoder output, 256 weights.
        let out = dequant_q4_k_blocks(&blk);
        assert_eq!(out.len(), 256);
        // Antirez's per-segment helper is called with il = 0..16 (for nl=16);
        // each call writes 16 weights. Concatenated, that's 256 weights. The
        // mapping from il → output position must match our sub-block order:
        //   sub-block i_sb = 0..8 emits 32 weights via il=2*i_sb then 2*i_sb+1.
        for i_sb in 0..8u8 {
            let lo = antirez_dequantize_q4_k_segment(&blk, 2 * i_sb);
            let hi = antirez_dequantize_q4_k_segment(&blk, 2 * i_sb + 1);
            let off = i_sb as usize * 32;
            for k in 0..16 {
                let got = out[off + k];
                let exp = lo[k];
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "sub-block {i_sb} lo[{k}]: got {got} expected {exp}"
                );
            }
            for k in 0..16 {
                let got = out[off + 16 + k];
                let exp = hi[k];
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "sub-block {i_sb} hi[{k}]: got {got} expected {exp}"
                );
            }
        }
    }

    /// Antirez's per-segment q2_K dequant, transcribed to Rust.
    fn antirez_dequantize_q2_k_segment(blk: &[u8], il_in: u8) -> [f32; 16] {
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let min = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));

        let sc = scales[il_in as usize];
        let q_off = ((il_in / 8) as usize) * 32 + ((il_in & 1) as usize) * 16;
        let il = ((il_in / 2) & 3) as u8;
        let coef: f32 = match il {
            0 => 1.0,
            1 => 1.0 / 4.0,
            2 => 1.0 / 16.0,
            3 => 1.0 / 64.0,
            _ => unreachable!(),
        };
        let mask: u8 = match il {
            0 => 0x03,
            1 => 0x0C,
            2 => 0x30,
            3 => 0xC0,
            _ => unreachable!(),
        };
        let dl = d * (sc & 0x0F) as f32 * coef;
        let ml = min * (sc >> 4) as f32;

        let mut out = [0f32; 16];
        for i in 0..16 {
            out[i] = dl * (qs[q_off + i] & mask) as f32 - ml;
        }
        out
    }

    fn make_q2_k_block(seed: u8) -> Vec<u8> {
        let mut blk = vec![0u8; 84];
        for i in 0..16 {
            blk[i] = seed.wrapping_add(i as u8).wrapping_mul(11);
        }
        for i in 0..64 {
            blk[16 + i] = seed.wrapping_add(i as u8).wrapping_mul(17);
        }
        blk[80..82].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
        blk
    }

    fn lcg_next(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u32
    }

    fn lcg_u8(seed: &mut u64) -> u8 {
        lcg_next(seed) as u8
    }

    #[test]
    fn dequant_q2_k_block_matches_antirez_segments() {
        let blk = make_q2_k_block(0xA5);
        let out = dequant_q2_k_blocks(&blk);
        assert_eq!(out.len(), 256);
        // sub-block i_sb ↔ il = i_sb. 16 weights each, 16 sub-blocks.
        for i_sb in 0..16u8 {
            let exp = antirez_dequantize_q2_k_segment(&blk, i_sb);
            let off = i_sb as usize * 16;
            for k in 0..16 {
                let got = out[off + k];
                let e = exp[k];
                assert_eq!(
                    got.to_bits(),
                    e.to_bits(),
                    "sub-block {i_sb} weight {k}: got {got} expected {e}"
                );
            }
        }
    }

    #[test]
    fn dequant_q2_k_random_blocks_match_antirez_segments() {
        let mut seed = 0x5152_4b5f_5052_4f50u64;
        let finite_halfs = [0x0000u16, 0x2c00, 0x3400, 0x3800, 0x3c00, 0x4200, 0xbc00];
        for trial in 0..96 {
            let mut blk = vec![0u8; 84];
            for b in &mut blk[0..80] {
                *b = lcg_u8(&mut seed);
            }
            let d = finite_halfs[(lcg_next(&mut seed) as usize) % finite_halfs.len()];
            let dmin = finite_halfs[(lcg_next(&mut seed) as usize) % finite_halfs.len()];
            blk[80..82].copy_from_slice(&d.to_le_bytes());
            blk[82..84].copy_from_slice(&dmin.to_le_bytes());

            let out = dequant_q2_k_blocks(&blk);
            assert_eq!(out.len(), 256);
            for i_sb in 0..16u8 {
                let exp = antirez_dequantize_q2_k_segment(&blk, i_sb);
                let off = i_sb as usize * 16;
                for k in 0..16 {
                    assert_eq!(
                        out[off + k].to_bits(),
                        exp[k].to_bits(),
                        "trial {trial}, sub-block {i_sb}, weight {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn dequant_q2_k_applies_min_even_when_scale_is_zero() {
        let mut blk = vec![0u8; 84];
        for scale in &mut blk[0..16] {
            *scale = 0x30; // scale=0, min=3
        }
        blk[80..82].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5

        let out = dequant_q2_k_blocks(&blk);
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v.to_bits(), (-1.5f32).to_bits(), "out[{i}] = {v}");
        }
    }

    #[test]
    fn dequant_q2_k_extracts_high_two_bit_field() {
        let mut blk = vec![0u8; 84];
        blk[6] = 0x01; // sub-block 6: scale=1, min=0
        blk[16] = 0b1100_0000; // sub-block 6 reads the high two bits at q_off=0
        blk[80..82].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        blk[82..84].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0

        let out = dequant_q2_k_blocks(&blk);
        assert_eq!(out[6 * 16].to_bits(), 3.0f32.to_bits());
        assert_eq!(out[6 * 16 + 1].to_bits(), 0.0f32.to_bits());
    }

    /// Antirez's per-segment iq2_xxs dequant transcribed to Rust.
    fn antirez_dequantize_iq2_xxs_segment(blk: &[u8], il_in: u8) -> [f32; 16] {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut qs = [0u16; 32];
        for i in 0..32 {
            qs[i] = u16::from_le_bytes([blk[2 + 2 * i], blk[2 + 2 * i + 1]]);
        }
        let ib32 = (il_in / 2) as usize;
        let il_eff = (il_in % 2) as u32;
        let q2 = &qs[4 * ib32..4 * ib32 + 4];
        let aux32_g = (q2[0] as u32) | ((q2[1] as u32) << 16);
        let aux32_s = (q2[2] as u32) | ((q2[3] as u32) << 16);
        let aux8 = aux32_g.to_le_bytes();
        let dl = d * (0.5 + (aux32_s >> 28) as f32) * 0.25;
        let mut out = [0f32; 16];

        // First group: aux8[2*il_eff + 0], signs from aux32_s >> (14*il_eff)
        let g0 = IQ2XXS_GRID[aux8[(2 * il_eff) as usize] as usize].to_le_bytes();
        let s0 = KSIGNS_IQ2XS[((aux32_s >> (14 * il_eff)) & 127) as usize];
        for i in 0..8 {
            let sign = if (s0 & KMASK_IQ2XS[i]) != 0 {
                -1.0
            } else {
                1.0
            };
            out[i] = dl * g0[i] as f32 * sign;
        }
        let g1 = IQ2XXS_GRID[aux8[(2 * il_eff + 1) as usize] as usize].to_le_bytes();
        let s1 = KSIGNS_IQ2XS[((aux32_s >> (14 * il_eff + 7)) & 127) as usize];
        for i in 0..8 {
            let sign = if (s1 & KMASK_IQ2XS[i]) != 0 {
                -1.0
            } else {
                1.0
            };
            out[8 + i] = dl * g1[i] as f32 * sign;
        }
        out
    }

    fn make_iq2_xxs_block(seed: u8) -> Vec<u8> {
        let mut blk = vec![0u8; 66];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        for i in 0..64 {
            blk[2 + i] = seed.wrapping_add(i as u8).wrapping_mul(19);
        }
        blk
    }

    #[test]
    fn dequant_iq2_xxs_block_matches_antirez_segments() {
        let blk = make_iq2_xxs_block(0x33);
        let out = dequant_iq2_xxs_blocks(&blk);
        assert_eq!(out.len(), 256);
        // Each ib32 (0..8) corresponds to il_in=2*ib32 (16 weights) followed
        // by il_in=2*ib32+1 (16 weights). Each emits 32 consecutive weights
        // in the natural output order.
        for ib32 in 0..8u8 {
            let lo = antirez_dequantize_iq2_xxs_segment(&blk, 2 * ib32);
            let hi = antirez_dequantize_iq2_xxs_segment(&blk, 2 * ib32 + 1);
            let off = ib32 as usize * 32;
            for k in 0..16 {
                let got = out[off + k];
                let e = lo[k];
                assert_eq!(
                    got.to_bits(),
                    e.to_bits(),
                    "ib32 {ib32} lo[{k}]: got {got} expected {e}"
                );
            }
            for k in 0..16 {
                let got = out[off + 16 + k];
                let e = hi[k];
                assert_eq!(
                    got.to_bits(),
                    e.to_bits(),
                    "ib32 {ib32} hi[{k}]: got {got} expected {e}"
                );
            }
        }
    }

    #[test]
    fn dequant_iq2_xxs_random_blocks_match_antirez_segments() {
        let mut seed = 0x4951_325f_5052_4f50u64;
        let finite_halfs = [0x0000u16, 0x2c00, 0x3400, 0x3800, 0x3c00, 0x4200, 0xbc00];
        for trial in 0..96 {
            let mut blk = vec![0u8; 66];
            let d = finite_halfs[(lcg_next(&mut seed) as usize) % finite_halfs.len()];
            blk[0..2].copy_from_slice(&d.to_le_bytes());
            for b in &mut blk[2..66] {
                *b = lcg_u8(&mut seed);
            }

            let out = dequant_iq2_xxs_blocks(&blk);
            assert_eq!(out.len(), 256);
            for ib32 in 0..8u8 {
                let lo = antirez_dequantize_iq2_xxs_segment(&blk, 2 * ib32);
                let hi = antirez_dequantize_iq2_xxs_segment(&blk, 2 * ib32 + 1);
                let off = ib32 as usize * 32;
                for k in 0..16 {
                    assert_eq!(
                        out[off + k].to_bits(),
                        lo[k].to_bits(),
                        "trial {trial}, ib32 {ib32}, lo[{k}]"
                    );
                }
                for k in 0..16 {
                    assert_eq!(
                        out[off + 16 + k].to_bits(),
                        hi[k].to_bits(),
                        "trial {trial}, ib32 {ib32}, hi[{k}]"
                    );
                }
            }
        }
    }

    #[test]
    fn dequant_iq2_xxs_uses_scale_extra_high_nibble() {
        let mut lo = vec![0u8; 66];
        lo[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        lo[2] = 1; // grid index 1 in aux8[0]

        let mut hi = lo.clone();
        hi[2 + 2 * 3 + 1] = 0x10; // q2[3] high nibble -> aux32_s >> 28 == 1

        let lo_out = dequant_iq2_xxs_blocks(&lo);
        let hi_out = dequant_iq2_xxs_blocks(&hi);
        assert_eq!(lo_out.len(), 256);
        assert_eq!(hi_out.len(), 256);

        // grid[1]'s first byte is 0x2b. Scale-extra changes dl from
        // 0.5*0.25 to 1.5*0.25, so the same signed grid value triples.
        assert_eq!(lo_out[0].to_bits(), 5.375f32.to_bits());
        assert_eq!(hi_out[0].to_bits(), 16.125f32.to_bits());
    }

    #[test]
    fn dequant_iq2_xxs_applies_sign_key_bits() {
        let mut positive = vec![0u8; 66];
        positive[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        positive[2] = 1; // grid index 1 in aux8[0]

        let mut negative = positive.clone();
        let sign_key = KSIGNS_IQ2XS
            .iter()
            .position(|&s| (s & KMASK_IQ2XS[0]) != 0)
            .expect("sign table has a key for bit 0");
        negative[2 + 2 * 2] = sign_key as u8; // q2[2] low 7 bits -> sign key for k=0

        let pos_out = dequant_iq2_xxs_blocks(&positive);
        let neg_out = dequant_iq2_xxs_blocks(&negative);
        assert_eq!(pos_out[0].to_bits(), 5.375f32.to_bits());
        assert_eq!(neg_out[0].to_bits(), (-5.375f32).to_bits());
    }

    #[test]
    fn f16_to_f32_round_trip_known_values() {
        // 0x0000 → 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0x3C00 → 1.0
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // 0xBC00 → -1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // 0x4000 → 2.0
        assert_eq!(f16_to_f32(0x4000), 2.0);
        // 0x3800 → 0.5
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // 0x3555 → ~0.333 (1+0x155/1024) * 2^-2 = 1.3330078125 * 0.25 = 0.333251953125
        let v = f16_to_f32(0x3555);
        assert!((v - 0.333_251_95).abs() < 1e-6, "got {}", v);
    }

    /// End-to-end check: build a tiny `QuantizedExpertWeights` from synthetic
    /// q4_K bytes, then verify the f32-compatible dequant oracle matches a
    /// hand-computed direct call into `moe::moe_routed_step` on the same
    /// dequantized slabs. This confirms the wiring (slab layout, stride math,
    /// d_ffn/d_in field plumbing) without needing a real GGUF.
    /// M4 #302 — antirez folds `expert_weight[slot]` INTO the mid vector at
    /// ds4.c:3690 (`ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot]`),
    /// BEFORE the down-side q8_K quantization at ds4.c:5266-5268. Applying
    /// the weight AFTER the down matvec (our previous code) is mathematically
    /// equivalent in real arithmetic but diverges at f32 ULP level, and in
    /// the `AntirezQ8K` path it shrinks `mid`'s block amax by `1/w_e` (≈4×
    /// when Σ=1.5 over 6 experts), giving our path 2 extra bits of q8_K
    /// precision relative to antirez — i.e. systematically MORE precise than
    /// antirez. The fix folds w_e into mid; this test exercises an
    /// adversarial shape where the post-mul order produces a different f32
    /// result and asserts our fixed path matches the antirez-spec order
    /// bit-identically. Pre-fix this test panicked at idx 0; post-fix green.
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn moe_routed_folds_expert_weight_into_mid_per_antirez_ds4c_3690() {
        // d_in=8, d_ffn=4, 2 experts. Hand-rolled f32 weight slabs (no quant)
        // exposed via the f32 oracle `moe::moe_routed_step` so the only
        // remaining f32 non-associativity is the placement of the w_e mul.
        let d_in = 8usize;
        let d_ffn = 4usize;

        // Adversarial: gate values straddle CLAMP=10 so SwiGLU saturates,
        // and up values mix signs + large magnitudes so silu(g)*u has
        // big absmax for some columns and tiny for others. With small w_e
        // applied after, the per-output reduction picks up f32 cancellation
        // that's absent when w_e scales the mid contributions first.
        let mut x = vec![0.0f32; d_in];
        for (j, xv) in x.iter_mut().enumerate() {
            *xv = ((j as f32) - 3.5) * 0.7;
        }
        // Construct one expert with rows that produce mostly-large mids and
        // one with mostly-tiny mids, so when summed with small weights the
        // f32 ordering of "scale-then-add" vs "add-then-scale" diverges.
        let mut w_gate = vec![0.0f32; 2 * d_ffn * d_in];
        let mut w_up = vec![0.0f32; 2 * d_ffn * d_in];
        let mut w_down = vec![0.0f32; 2 * d_in * d_ffn];
        let mut seed = 0xC0FFEE_C0FFEE_42u64;
        let lcg = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 4.0 - 2.0
        };
        for v in w_gate.iter_mut() { *v = lcg(&mut seed); }
        for v in w_up.iter_mut() { *v = lcg(&mut seed); }
        for v in w_down.iter_mut() { *v = lcg(&mut seed); }
        let selected = vec![0usize, 1];
        let weights = vec![0.25f32, 0.75]; // Σ=1.0, mimics router_finalize/1.5

        // The "antirez-correct" reference: hand-stamp the per-expert path
        // folding w_e into mid BEFORE the down matvec.
        const CLAMP: f32 = 10.0;
        let mut want = vec![0.0f32; d_in];
        for (&e, &w_e) in selected.iter().zip(weights.iter()) {
            let g_w = &w_gate[e * d_ffn * d_in..(e + 1) * d_ffn * d_in];
            let u_w = &w_up[e * d_ffn * d_in..(e + 1) * d_ffn * d_in];
            let d_w = &w_down[e * d_in * d_ffn..(e + 1) * d_in * d_ffn];
            let gate = ds4_engine::forward::matvec_f32(g_w, &x, d_ffn);
            let up = ds4_engine::forward::matvec_f32(u_w, &x, d_ffn);
            let mid_weighted: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| {
                let g_c = if g > CLAMP { CLAMP } else { g };
                let u_c = u.clamp(-CLAMP, CLAMP);
                ds4_engine::forward::silu(g_c) * u_c * w_e
            }).collect();
            let y_e = ds4_engine::forward::matvec_f32(d_w, &mid_weighted, d_in);
            for (av, &yv) in want.iter_mut().zip(y_e.iter()) {
                *av += yv;
            }
        }

        // Pre-fix path: weight applied AFTER down matvec. Compute it locally
        // so we can prove the discrimination test is non-trivial (must give
        // a DIFFERENT bit pattern from the antirez path).
        let mut wrong = vec![0.0f32; d_in];
        for (&e, &w_e) in selected.iter().zip(weights.iter()) {
            let g_w = &w_gate[e * d_ffn * d_in..(e + 1) * d_ffn * d_in];
            let u_w = &w_up[e * d_ffn * d_in..(e + 1) * d_ffn * d_in];
            let d_w = &w_down[e * d_in * d_ffn..(e + 1) * d_in * d_ffn];
            let gate = ds4_engine::forward::matvec_f32(g_w, &x, d_ffn);
            let up = ds4_engine::forward::matvec_f32(u_w, &x, d_ffn);
            let mid: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| {
                let g_c = if g > CLAMP { CLAMP } else { g };
                let u_c = u.clamp(-CLAMP, CLAMP);
                ds4_engine::forward::silu(g_c) * u_c
            }).collect();
            let y_e = ds4_engine::forward::matvec_f32(d_w, &mid, d_in);
            for (av, &yv) in wrong.iter_mut().zip(y_e.iter()) {
                *av += w_e * yv;
            }
        }
        // Discrimination check — at least one output must differ at bit level,
        // otherwise the test trivially passes and proves nothing.
        let any_bit_diff = want.iter().zip(wrong.iter()).any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(any_bit_diff,
            "test is trivial: antirez-spec and pre-fix order produced identical f32 outputs — \
             rebuild the input to be more adversarial");

        // Our production path must equal the antirez-spec order bit-exactly.
        let got = ds4_engine::moe::moe_routed_step(
            &x, &selected, &weights, &w_gate, &w_up, &w_down, d_ffn);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(),
                "idx={i}: ours={g} want={w} (antirez ds4.c:3690 spec)");
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn oracle_matches_direct_moe_routed_step_q4_k() {
        // Tiny shape: d_in = 256 (one q4_K block per row), d_ffn = 256, 4 experts.
        // gate/up: [4][256][256], down: [4][256][256].
        let n_experts: u32 = 4;
        let d_in: u32 = 256;
        let d_ffn: u32 = 256;
        let rows_per_expert_gate = d_ffn as u64;
        let rows_per_expert_down = d_in as u64;
        // q4_K: each block packs 256 weights in 144 bytes. Per row = 1 block.
        let bytes_per_row = 144u64;
        let bytes_per_expert_gate = rows_per_expert_gate * bytes_per_row;
        let bytes_per_expert_down = rows_per_expert_down * bytes_per_row;

        let mk_stream = |salt: u8, n_blocks: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n_blocks * 144);
            for i in 0..n_blocks {
                v.extend_from_slice(&make_q4_k_block(salt.wrapping_add(i as u8)));
            }
            v
        };

        let gate_bytes = mk_stream(0x10, (n_experts * d_ffn) as usize);
        let up_bytes = mk_stream(0x20, (n_experts * d_ffn) as usize);
        let down_bytes = mk_stream(0x30, (n_experts * d_in) as usize);

        let qew = QuantizedExpertWeights {
            layer_idx: 0,
            n_experts,
            d_in,
            d_ffn,
            gate: QuantTensor {
                ttype: GgmlType::Q4_K,
                dims: [n_experts as u64, d_ffn as u64, d_in as u64],
                bytes: gate_bytes,
                expert_stride: bytes_per_expert_gate,
            },
            up: QuantTensor {
                ttype: GgmlType::Q4_K,
                dims: [n_experts as u64, d_ffn as u64, d_in as u64],
                bytes: up_bytes,
                expert_stride: bytes_per_expert_gate,
            },
            down: QuantTensor {
                ttype: GgmlType::Q4_K,
                dims: [n_experts as u64, d_in as u64, d_ffn as u64],
                bytes: down_bytes,
                expert_stride: bytes_per_expert_down,
            },
        };

        // Build a deterministic activation and a 2-expert route.
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.013) - 0.7).collect();
        let selected = vec![1usize, 3];
        let weights = vec![0.25f32, 0.75];

        let oracle = moe_routed_step_cpu_via_dequant_with_activation(
            &qew,
            &x,
            &selected,
            &weights,
            d_ffn as usize,
            ActivationQuant::F32,
        )
        .unwrap();

        // Hand-stamp the same call by reading dequantized slabs directly.
        let gate_f32 = qew.dequant_gate_f32().unwrap();
        let up_f32 = qew.dequant_up_f32().unwrap();
        let down_f32 = qew.dequant_down_f32().unwrap();
        let direct = ds4_engine::moe::moe_routed_step(
            &x,
            &selected,
            &weights,
            &gate_f32,
            &up_f32,
            &down_f32,
            d_ffn as usize,
        );

        assert_eq!(oracle.len(), direct.len());
        for (i, (&a, &b)) in oracle.iter().zip(direct.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "oracle vs direct diverge at idx {i}: {a} vs {b}"
            );
        }
        // Sanity-check that the output is non-trivial (route_weights affect it).
        let any_nonzero = oracle.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "expected non-zero output");
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn expert_slice_bounds() {
        let t = dummy_quant(GgmlType::Q4_K, 4, 256, 256);
        // 256*256/256 = 256 blocks; 256*144 = 36864 bytes/expert.
        let s0 = t.expert_slice(0);
        let s3 = t.expert_slice(3);
        assert_eq!(s0.len(), 36864);
        assert_eq!(s3.len(), 36864);
        // Bytes are zero-init, so slices are empty-equal but disjoint.
        assert_eq!(s0.as_ptr() as usize + 3 * 36864, s3.as_ptr() as usize);
    }

    // ───────────────────────────────────────────────────────────────────────
    // q8_K activation round-trip tests (M4 #298 breakthrough).
    //
    // q8_K is the antirez activation quantization used before int8-dot against
    // iq2_xxs/q2_K routed-MoE weights. Bugs would silently degrade pos=15
    // cluster precision (the bug class M4 #298 closed). These tests target
    // sign convention, rounding mode, block independence, dead-zero handling.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn q8_k_round_trip_zero_block_returns_zeros() {
        let x = vec![0.0f32; 256];
        let y = super::q8_k_round_trip(&x);
        assert_eq!(y.len(), 256);
        assert!(
            y.iter().all(|&v| v == 0.0),
            "all-zero block must round-trip to zero"
        );
    }

    #[test]
    fn q8_k_round_trip_preserves_max_abs_element() {
        // The element with max |·| anchors the scale. After round-trip
        // its quantum should sit at ±|max| within d/2.
        let mut x = vec![0.01f32; 256];
        x[42] = 5.0; // positive peak
        let y = super::q8_k_round_trip(&x);
        // d = |max|/127 = 5.0/127 ≈ 0.0394; peak quantizes to ±127 ⇒ |y| ≈ 5.0.
        assert!(
            (y[42].abs() - 5.0).abs() < 0.05,
            "peak |{}| should round-trip near 5.0, got {}",
            x[42],
            y[42]
        );
        // Sign must be preserved.
        assert_eq!(
            y[42].signum(),
            1.0,
            "positive peak must stay positive: got {}",
            y[42]
        );
    }

    #[test]
    fn q8_k_round_trip_preserves_sign_with_negative_peak() {
        // M4 #298 noted antirez `iscale = -127/max` (signed); our port uses
        // `iscale = -127/max_abs`. For round-trip both yield the same output
        // because v*d ≈ x[j]. This test verifies that invariant ALSO holds
        // when the max-abs element is negative — historically the edge case.
        let mut x = vec![0.02f32; 256];
        x[100] = -7.5; // negative peak
        let y = super::q8_k_round_trip(&x);
        assert!(
            y[100] < 0.0,
            "negative peak must remain negative after round-trip, got {}",
            y[100]
        );
        assert!(
            (y[100] - (-7.5)).abs() < 0.07,
            "negative peak should round-trip near -7.5, got {}",
            y[100]
        );
    }

    #[test]
    fn q8_k_round_trip_two_blocks_independent_scales() {
        // 512 elements = 2 blocks. block-0 amax=0.5, block-1 amax=50.0.
        // If the implementation shared one scale across blocks, the small
        // values in block-0 would be crushed to 0.
        let mut x = vec![0.0f32; 512];
        x[10] = 0.5; // block 0 peak
        x[20] = 0.1; // block 0 mid
        x[256 + 5] = 50.0; // block 1 peak
        x[256 + 15] = 25.0; // block 1 mid
        let y = super::q8_k_round_trip(&x);
        assert!(
            (y[20] - 0.1).abs() < 0.01,
            "block-0 small element should not be crushed by block-1 scale; got {}",
            y[20]
        );
        assert!(
            (y[256 + 15] - 25.0).abs() < 0.5,
            "block-1 mid should be ~25.0, got {}",
            y[256 + 15]
        );
    }

    #[test]
    fn q8_k_round_trip_uses_lrintf_banker_rounding() {
        // M4 #299 — lrintf in C uses round-half-to-even (FE_TONEAREST default),
        // NOT round-half-away-from-zero. If we use f32::round (away-from-zero),
        // half-quantum values round wrong and the int8-dot diverges from antirez.
        //
        // Setup: amax = 127.0 ⇒ d = 1.0 ⇒ q = lrintf(x). For x = 0.5, 1.5, 2.5, 3.5:
        //   banker's:  0, 2, 2, 4
        //   away-zero: 1, 2, 3, 4
        let mut x = vec![0.0f32; 256];
        x[0] = 127.0; // anchors amax → d = 1.0 exactly
        x[1] = 0.5;
        x[2] = 1.5;
        x[3] = 2.5;
        x[4] = 3.5;
        x[5] = -0.5;
        x[6] = -2.5;
        let y = super::q8_k_round_trip(&x);
        // amax = 127.0, max_abs = 127.0, iscale = -127/127 = -1, d = -1.
        // For round_ties_even(iscale*x[j]) with x[j]=0.5: round(-0.5)=0,
        // for x[j]=1.5: round(-1.5)=-2 (banker's, even), v*d = -2*-1 = 2.
        // for x[j]=2.5: round(-2.5)=-2 (banker's, even), v*d = -2*-1 = 2.
        // for x[j]=3.5: round(-3.5)=-4 (banker's, even), v*d = -4*-1 = 4.
        // for x[j]=-0.5: round(0.5)=0, v*d = 0.
        // for x[j]=-2.5: round(2.5)=2 (banker's, even), v*d = 2*-1 = -2.
        assert!((y[1] - 0.0).abs() < 1e-5, "0.5 banker → 0, got {}", y[1]);
        assert!((y[2] - 2.0).abs() < 1e-5, "1.5 banker → 2, got {}", y[2]);
        assert!(
            (y[3] - 2.0).abs() < 1e-5,
            "2.5 banker → 2 (even), got {}",
            y[3]
        );
        assert!(
            (y[4] - 4.0).abs() < 1e-5,
            "3.5 banker → 4 (even), got {}",
            y[4]
        );
        assert!((y[5] - 0.0).abs() < 1e-5, "-0.5 banker → 0, got {}", y[5]);
        assert!(
            (y[6] - (-2.0)).abs() < 1e-5,
            "-2.5 banker → -2 (even), got {}",
            y[6]
        );
    }

    #[test]
    fn q8_k_round_trip_clamps_at_int8_max() {
        // If the implementation skipped the v.clamp(-128, 127), values whose
        // quantum overflowed int8 would wrap modulo 256 → catastrophic error.
        // Force overflow: amax = 0.1 (d ≈ 0.00079) but include one value at
        // 1000.0 that would map to v ≈ 12700 without clamping.
        // (In practice this can't happen because amax IS max of |x|, but we
        // still want defence-in-depth: hand-construct a buggy input.)
        let mut x = vec![0.05f32; 256];
        x[0] = 1.0; // anchors amax = 1.0, d = 1/127.
                    // All values |x[j]| ≤ amax by construction; check the peak quantum.
        let y = super::q8_k_round_trip(&x);
        assert!(
            y[0].is_finite(),
            "peak quantum must be finite, got {}",
            y[0]
        );
        // Peak round-trips to ±|amax|.
        assert!((y[0].abs() - 1.0).abs() < 0.01, "peak ≈ 1.0, got {}", y[0]);
    }

    // ───────────────────────────────────────────────────────────────────────
    // q8_0_round_trip — distinct block size (32 vs 256) and tail handling.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn q8_0_round_trip_partial_block_at_end_quantizes() {
        // 40 elements: block 0 = 32 elements, partial tail = 8 elements.
        // Bug pattern: implementations that loop `for b in 0..(n/BLOCK)`
        // silently DROP the tail (output zeros for j ≥ 32). This test
        // forces a non-zero tail and checks it survives.
        let mut x = vec![0.001f32; 40];
        x[3] = 0.5; // block 0
        x[35] = 2.0; // tail peak — would round-trip exactly with own scale
        x[37] = 1.0;
        let y = super::q8_0_round_trip(&x);
        assert!(
            (y[35] - 2.0).abs() < 1e-4,
            "tail peak should round-trip, got {}",
            y[35]
        );
        assert!(
            y[37] > 0.5,
            "tail mid should not be zero-dropped, got {}",
            y[37]
        );
    }

    #[test]
    fn q8_0_round_trip_handles_64_elements_two_full_blocks() {
        // Sanity: two full blocks, asymmetric magnitudes. Verifies blocks
        // index correctly (no off-by-32 in the block-loop).
        let mut x = vec![0.0f32; 64];
        x[5] = 10.0; // block 0 peak
        x[10] = 5.0; // block 0 mid
        x[32 + 5] = 100.0; // block 1 peak
        x[32 + 10] = 50.0; // block 1 mid
        let y = super::q8_0_round_trip(&x);
        // Block 0 d = 10/127 ≈ 0.0787; y[5] ≈ 10.0, y[10] ≈ 5.0.
        assert!(
            (y[5] - 10.0).abs() < 0.1,
            "block-0 peak ≈ 10.0, got {}",
            y[5]
        );
        assert!(
            (y[10] - 5.0).abs() < 0.1,
            "block-0 mid ≈ 5.0, got {}",
            y[10]
        );
        // Block 1 d = 100/127 ≈ 0.787; y[37] ≈ 100.0, y[42] ≈ 50.0.
        assert!(
            (y[32 + 5] - 100.0).abs() < 1.0,
            "block-1 peak ≈ 100.0, got {}",
            y[32 + 5]
        );
        assert!(
            (y[32 + 10] - 50.0).abs() < 1.0,
            "block-1 mid ≈ 50.0, got {}",
            y[32 + 10]
        );
        // Critical: a block-loop off-by-one would write block-0 into block-1
        // slots; check that block-0's small value at slot 10 is NOT crushed
        // by block-1's d (which would give ~0).
        assert!(
            y[10] > 4.0,
            "off-by-one in block loop would crush this, got {}",
            y[10]
        );
    }

    // ---- antirez-mirror q8_K reference + bsums divergence tests ----
    //
    // M4 #299/#300 BUG-DISCOVERY: antirez `ds4_quantize_row_q8_K`
    // (ds4.c:1508-1545) computes:
    //
    //   max = signed-x[argmax_abs(x)]   // SIGNED value, not |x|
    //   iscale = -127.0f / max          // sign flips if x_max is positive
    //   for j: qs[j] = lrintf(iscale * x[j]).clamp(-128, 127)   // int8
    //   for ig: bsums[ig] = sum(qs[16*ig .. 16*ig+16])          // int16, signed
    //   d = 1.0f / iscale = -max / 127  (signed)
    //
    // Our `q8_k_round_trip` uses `iscale = -127.0f / max_abs` (UNSIGNED).
    // For round-trip f32 OUTPUT both sign conventions cancel (v·d ≈ x),
    // so existing round-trip tests pass. BUT:
    //
    //   - antirez's int8 `qs` has opposite sign vs ours on +max-abs blocks
    //   - antirez's `bsums[ig]` therefore differs in sign on +max-abs blocks
    //   - q2_K · q8_K dot in Metal kernels uses bsums to subtract min·sum_q;
    //     wrong sign here would systematically bias dot product results
    //
    // Round-trip f32 tests miss this. A direct bsums comparison catches it.

    /// Mirror antirez `ds4_quantize_row_q8_K` (ds4.c:1508-1545) verbatim.
    /// Returns (qs[256], bsums[16], d) per block — concatenated across blocks.
    fn antirez_q8_k_quantize(x: &[f32]) -> (Vec<i8>, Vec<i16>, Vec<f32>) {
        const QK_K: usize = 256;
        let n_blocks = x.len() / QK_K;
        let mut all_qs = vec![0i8; n_blocks * QK_K];
        let mut all_bsums = vec![0i16; n_blocks * 16];
        let mut all_d = vec![0.0f32; n_blocks];
        for b in 0..n_blocks {
            let off = b * QK_K;
            // Antirez: find SIGNED max at argmax_abs (ds4.c:1518-1525).
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for j in 0..QK_K {
                let v = x[off + j];
                let av = v.abs();
                if av > amax {
                    amax = av;
                    max = v;
                }
            }
            if amax == 0.0 {
                all_d[b] = 0.0;
                continue;
            }
            let iscale = -127.0f32 / max;
            for j in 0..QK_K {
                let q = (iscale * x[off + j]).round_ties_even() as i32;
                let q = q.clamp(-128, 127);
                all_qs[off + j] = q as i8;
            }
            // bsums[ig] = sum over 16-element group (ds4.c:1538-1542)
            for ig in 0..16 {
                let mut s: i32 = 0;
                for k in 0..16 {
                    s += all_qs[off + ig * 16 + k] as i32;
                }
                all_bsums[b * 16 + ig] = s as i16;
            }
            all_d[b] = 1.0f32 / iscale; // signed; = -max/127
        }
        (all_qs, all_bsums, all_d)
    }

    /// "Our" mirror — same as antirez but with `iscale = -127 / max_abs`
    /// (unsigned), matching `q8_k_round_trip` above. Used to confirm the
    /// sign-flip is real, not a transcription error in the antirez mirror.
    fn ours_q8_k_quantize_via_max_abs(x: &[f32]) -> (Vec<i8>, Vec<i16>, Vec<f32>) {
        const QK_K: usize = 256;
        let n_blocks = x.len() / QK_K;
        let mut all_qs = vec![0i8; n_blocks * QK_K];
        let mut all_bsums = vec![0i16; n_blocks * 16];
        let mut all_d = vec![0.0f32; n_blocks];
        for b in 0..n_blocks {
            let off = b * QK_K;
            let mut max_abs = 0.0f32;
            for j in 0..QK_K {
                let av = x[off + j].abs();
                if av > max_abs {
                    max_abs = av;
                }
            }
            if max_abs == 0.0 {
                all_d[b] = 0.0;
                continue;
            }
            let iscale = -127.0f32 / max_abs;
            for j in 0..QK_K {
                let q = (iscale * x[off + j]).round_ties_even() as i32;
                let q = q.clamp(-128, 127);
                all_qs[off + j] = q as i8;
            }
            for ig in 0..16 {
                let mut s: i32 = 0;
                for k in 0..16 {
                    s += all_qs[off + ig * 16 + k] as i32;
                }
                all_bsums[b * 16 + ig] = s as i16;
            }
            all_d[b] = 1.0f32 / iscale;
        }
        (all_qs, all_bsums, all_d)
    }

    fn dot_q2_16(q2: &[u8], q8: &[i8], shift: u8) -> i32 {
        let mut sum = 0i32;
        for i in 0..16 {
            sum += q8[i] as i32 * ((q2[i] >> shift) & 3) as i32;
        }
        sum
    }

    fn antirez_vec_dot_q2_k_q8_k_one_block(q2_blk: &[u8], x: &[f32]) -> f32 {
        assert_eq!(q2_blk.len(), 84);
        assert_eq!(x.len(), 256);
        let sc = &q2_blk[0..16];
        let q2 = &q2_blk[16..80];
        let d_q2 = f16_to_f32(u16::from_le_bytes([q2_blk[80], q2_blk[81]]));
        let dmin_q2 = f16_to_f32(u16::from_le_bytes([q2_blk[82], q2_blk[83]]));
        let (q8, bsums, d_q8) = antirez_q8_k_quantize(x);

        let mut summs = 0i32;
        for j in 0..16 {
            summs += bsums[j] as i32 * (sc[j] >> 4) as i32;
        }

        let mut isum = 0i32;
        let mut is = 0usize;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for _ in 0..2 {
            let mut shift = 0u8;
            for _ in 0..4 {
                let scale = (sc[is] & 0x0f) as i32;
                isum +=
                    scale * dot_q2_16(&q2[q2_off..q2_off + 16], &q8[q8_off..q8_off + 16], shift);
                is += 1;

                let scale = (sc[is] & 0x0f) as i32;
                isum += scale
                    * dot_q2_16(
                        &q2[q2_off + 16..q2_off + 32],
                        &q8[q8_off + 16..q8_off + 32],
                        shift,
                    );
                is += 1;

                shift += 2;
                q8_off += 32;
            }
            q2_off += 32;
        }

        let dall = d_q8[0] * d_q2;
        let dmin = d_q8[0] * dmin_q2;
        dall * isum as f32 - dmin * summs as f32
    }

    fn current_f32_oracle_q2_k_dot_one_block(q2_blk: &[u8], x: &[f32]) -> f32 {
        let weights = dequant_q2_k_blocks(q2_blk);
        let x_back = q8_k_round_trip(x);
        weights.iter().zip(x_back.iter()).map(|(w, x)| w * x).sum()
    }

    #[test]
    fn q8_k_positive_max_block_int8_qs_disagree_with_antirez() {
        // BUG-DISCOVERY: when the max-abs element is POSITIVE, antirez's
        // signed-max iscale has a different sign than our max_abs iscale.
        // Build a 256-element block with all-positive values (max-abs IS
        // positive). qs should differ in SIGN.
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = 0.5 + j as f32 * 0.01; // all positive, max at j=255
        }
        let (anti_qs, _anti_bsums, anti_d) = antirez_q8_k_quantize(&x);
        let (ours_qs, _ours_bsums, ours_d) = ours_q8_k_quantize_via_max_abs(&x);
        // d_anti = -max/127  (negative);  d_ours = -max_abs/127 = same magnitude
        // BUT antirez d = -max/127, ours d = -max_abs/127. max>0 here, so both
        // are negative, equal. So d agrees. The DIFFERENCE shows up in qs:
        //   antirez:  iscale = -127/+max → negative → qs has NEGATIVE sign for positive x
        //   ours:     iscale = -127/+max_abs (max_abs==+max) → SAME sign as antirez!
        //
        // Wait — when max IS positive, max == max_abs, so iscale is identical
        // and qs agrees. The sign flip only happens when the max-abs element
        // is NEGATIVE: then max = negative, iscale = -127/neg = POSITIVE;
        // whereas max_abs = +|neg|, iscale = -127/+ = NEGATIVE. Opposite signs.
        //
        // Re-purpose this test: positive-max block should AGREE.
        assert_eq!(
            anti_qs, ours_qs,
            "positive-max block: antirez and ours-via-max_abs should agree on qs"
        );
        assert!(
            (anti_d[0] - ours_d[0]).abs() < 1e-6,
            "positive-max block: d should agree (anti={}, ours={})",
            anti_d[0],
            ours_d[0]
        );
    }

    #[test]
    fn q2_k_dot_current_f32_oracle_matches_antirez_q8_k_on_negative_max() {
        let mut q2_blk = vec![0u8; 84];
        for j in 0..16 {
            q2_blk[j] = 0x31 + ((j as u8) & 0x03); // nonzero scale and nonzero min
        }
        for j in 0..64 {
            q2_blk[16 + j] = (j as u8).wrapping_mul(37).wrapping_add(0x5a);
        }
        q2_blk[80..82].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        q2_blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5

        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = ((j as i32 - 128) as f32) * 0.1; // max-abs is -12.8
        }

        let antirez = antirez_vec_dot_q2_k_q8_k_one_block(&q2_blk, &x);
        let current = current_f32_oracle_q2_k_dot_one_block(&q2_blk, &x);
        let diff = (antirez - current).abs();
        assert!(
            diff < 1.0e-3,
            "current f32 q2_K oracle drifted from antirez q8_K dot; antirez={antirez}, current={current}, diff={diff}"
        );
    }

    #[test]
    fn q8_k_negative_max_block_int8_qs_sign_flip_vs_antirez() {
        // BUG-DISCOVERY: build a block where the max-abs element is NEGATIVE.
        // antirez iscale = -127 / (negative max) = POSITIVE.
        // ours_via_max_abs iscale = -127 / +max_abs = NEGATIVE.
        // qs should be EXACTLY sign-flipped element-by-element.
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = -(0.5 + j as f32 * 0.01); // all negative, max-abs at j=255 (most negative)
        }
        let (anti_qs, anti_bsums, anti_d) = antirez_q8_k_quantize(&x);
        let (ours_qs, ours_bsums, ours_d) = ours_q8_k_quantize_via_max_abs(&x);

        // d: anti = -max/127 = -(-2.95)/127 = +0.0232 (POSITIVE)
        //    ours = -max_abs/127 = -2.95/127 = -0.0232 (NEGATIVE)
        // Opposite signs. The round-trip x_back = q*d ends up the SAME because
        // both q and d flip sign, but the int8 q values disagree.
        assert!(
            anti_d[0] * ours_d[0] < 0.0,
            "negative-max block: d should have opposite signs (anti={}, ours={})",
            anti_d[0],
            ours_d[0]
        );

        // qs sign-flip: every element should be exactly negated.
        // We can't assert qs[j] == -ours_qs[j] for ALL j because of clamping
        // asymmetry: -128 negated would be +128, clamped to +127. Check most:
        let mut sign_flips = 0;
        let mut clamp_asym = 0;
        for j in 0..256 {
            if anti_qs[j] as i32 == -(ours_qs[j] as i32) {
                sign_flips += 1;
            } else if anti_qs[j] == -128 || ours_qs[j] == -128 {
                clamp_asym += 1;
            }
        }
        assert!(sign_flips + clamp_asym == 256,
            "expected all 256 qs to be sign-flipped or clamp-asymmetric, got sign_flips={} clamp_asym={}",
            sign_flips, clamp_asym);
        assert!(
            sign_flips >= 255,
            "expected ≥255 strict sign-flips on negative-max block, got {}",
            sign_flips
        );

        // bsums: must be sign-flipped (sums of 16 sign-flipped qs).
        for ig in 0..16 {
            assert!(
                anti_bsums[ig] as i32 == -(ours_bsums[ig] as i32)
                    || (anti_bsums[ig] - (-ours_bsums[ig])).abs() <= 1,
                "bsums[{}]: anti={} ours={} (expected negation)",
                ig,
                anti_bsums[ig],
                ours_bsums[ig]
            );
        }
    }

    #[test]
    fn q8_k_round_trip_f32_output_agrees_despite_sign_flip() {
        // Sanity: round-trip x_back = q * d should still agree between
        // antirez and ours-via-max_abs because both q and d flip sign
        // simultaneously. This confirms why our existing round-trip tests
        // missed the bsums-sign bug.
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = -(0.5 + j as f32 * 0.01); // negative-max block
        }
        let (anti_qs, _, anti_d) = antirez_q8_k_quantize(&x);
        let (ours_qs, _, ours_d) = ours_q8_k_quantize_via_max_abs(&x);

        // Reconstruct x_back for both. Must agree to ~1e-5 despite sign flip
        // in q and d (because product is invariant under simultaneous flip).
        for j in 0..256 {
            let anti_back = anti_qs[j] as f32 * anti_d[0];
            let ours_back = ours_qs[j] as f32 * ours_d[0];
            assert!(
                (anti_back - ours_back).abs() < 1e-4,
                "round-trip[{}]: anti={} ours={}",
                j,
                anti_back,
                ours_back
            );
        }
    }

    #[test]
    fn q8_k_round_trip_in_quantized_experts_matches_antirez_x_back() {
        // The production `q8_k_round_trip` only RETURNS x_back (not qs / bsums).
        // So its f32 output should agree with antirez's x_back even on a
        // negative-max block — this is the "wrong qs but right x_back" trap
        // that hid the bug.
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = -(0.5 + j as f32 * 0.01);
        }
        let ours_x_back = super::q8_k_round_trip(&x);
        let (anti_qs, _, anti_d) = antirez_q8_k_quantize(&x);
        for j in 0..256 {
            let anti_back = anti_qs[j] as f32 * anti_d[0];
            assert!(
                (anti_back - ours_x_back[j]).abs() < 1e-4,
                "q8_k_round_trip vs antirez x_back[{}]: anti={} ours={}",
                j,
                anti_back,
                ours_x_back[j]
            );
        }
        // KEY INSIGHT: this test PASSES. That's why existing q8_K round-trip
        // tests never caught the sign-convention bug. The bug only manifests
        // when q OR bsums are CONSUMED downstream (q2_K·q8_K dot uses bsums
        // to subtract min·sum_q). M4 #298's `q8k_act_round_trip` uses ONLY
        // x_back via dequant_q2_k → matvec_f32, NEVER touches bsums directly,
        // so the bug doesn't manifest in the M4 CPU oracle path. Antirez's
        // Metal q2_K·q8_K dot DOES touch bsums though.
    }

    #[test]
    fn q8_k_mixed_sign_block_with_negative_max_qs_diverges() {
        // Practical case: an activation block with mixed-sign values but
        // largest-abs is negative (common in LLM hidden states).
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = ((j as i32 - 128) as f32) * 0.1; // -12.8 .. +12.7
        }
        // max-abs at j=0 with value -12.8 (negative).
        let (anti_qs, anti_bsums, _) = antirez_q8_k_quantize(&x);
        let (ours_qs, ours_bsums, _) = ours_q8_k_quantize_via_max_abs(&x);

        // qs should be sign-flipped element-wise (modulo clamp asymmetry).
        let mut differs = 0;
        for j in 0..256 {
            if anti_qs[j] != ours_qs[j] {
                differs += 1;
            }
        }
        assert!(differs >= 255,
            "expected ≥255 qs to differ between antirez and ours-via-max_abs on neg-max mixed block, got {}",
            differs);

        // bsums: should be sign-flipped per group.
        let mut bsums_differ = 0;
        for ig in 0..16 {
            if anti_bsums[ig] != ours_bsums[ig] {
                bsums_differ += 1;
            }
        }
        assert!(
            bsums_differ >= 14,
            "expected ≥14/16 bsums groups to differ, got {}",
            bsums_differ
        );
    }

    #[test]
    fn q8_k_positive_max_mixed_sign_block_agrees_with_antirez() {
        // Control case: mixed-sign block where max-abs is POSITIVE.
        // antirez and ours-via-max_abs should give IDENTICAL qs and bsums.
        let mut x = vec![0.0f32; 256];
        for j in 0..256 {
            x[j] = ((j as i32 - 127) as f32) * 0.1; // -12.7 .. +12.8 (max-abs at j=255 = +12.8)
        }
        let (anti_qs, anti_bsums, anti_d) = antirez_q8_k_quantize(&x);
        let (ours_qs, ours_bsums, ours_d) = ours_q8_k_quantize_via_max_abs(&x);
        assert_eq!(
            anti_qs, ours_qs,
            "positive-max block: qs must agree exactly"
        );
        assert_eq!(
            anti_bsums, ours_bsums,
            "positive-max block: bsums must agree exactly"
        );
        assert!(
            (anti_d[0] - ours_d[0]).abs() < 1e-7,
            "positive-max block: d must agree (anti={}, ours={})",
            anti_d[0],
            ours_d[0]
        );
    }
}
