//! Minimal GGUF v3 reader for DS4 q2 model headers + tensor index.
//!
//! Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! Scope: read-only. Parses magic, version, metadata KV pairs, and tensor info
//! table (name, dims, type, offset). Does NOT yet dequantize tensor data — that
//! belongs to E3 (Metal-side decode kernels).
//!
//! Vendor-friendly: zero external crates beyond `anyhow`. All LE reads are
//! hand-rolled over a Vec<u8> + position cursor.

use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // 'GGUF' little-endian
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];
/// Hard cap on a tensor's declared dimension count. GGML itself uses 4; 8 is a
/// generous bound that still rejects a forged/corrupt `n_dims` before it can
/// drive a large `Vec::with_capacity`.
const MAX_TENSOR_DIMS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => bail!("unknown GGUF type tag {v}"),
        })
    }
}

// Variant names mirror llama.cpp ggml_type constants verbatim for grep-ability.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => bail!("unknown GGML type tag {v}"),
        })
    }

    /// The GGML type tag (inverse of `from_u32`) — what the GPU kernels take as
    /// `gate_type`/`down_type`.
    pub fn ggml_id(self) -> u32 {
        match self {
            Self::F32 => 0, Self::F16 => 1, Self::Q4_0 => 2, Self::Q4_1 => 3,
            Self::Q5_0 => 6, Self::Q5_1 => 7, Self::Q8_0 => 8, Self::Q8_1 => 9,
            Self::Q2_K => 10, Self::Q3_K => 11, Self::Q4_K => 12, Self::Q5_K => 13,
            Self::Q6_K => 14, Self::Q8_K => 15, Self::IQ2_XXS => 16, Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18, Self::IQ1_S => 19, Self::IQ4_NL => 20, Self::IQ3_S => 21,
            Self::IQ2_S => 22, Self::IQ4_XS => 23, Self::I8 => 24, Self::I16 => 25,
            Self::I32 => 26, Self::I64 => 27, Self::F64 => 28, Self::IQ1_M => 29, Self::BF16 => 30,
        }
    }

    /// Elements per block. Used together with `type_size` (bytes per block)
    /// in `(n_elems / block_size) * type_size` to compute on-disk bytes.
    /// Primitives are 1 element per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::I32
            | Self::F16
            | Self::I16
            | Self::BF16
            | Self::F64
            | Self::I64
            | Self::I8 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            _ => 256,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::I16 | Self::BF16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 => 1,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 32,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // GGUF spec enumerants — parser emits all, consumers only read a subset today.
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array {
        elem_type: GgufType,
        len: u64,
        values: Vec<MetaValue>,
    },
}

impl MetaValue {
    pub fn short(&self) -> String {
        match self {
            Self::U32(v) => v.to_string(),
            Self::U64(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::F32(v) => format!("{v}"),
            Self::F64(v) => format!("{v}"),
            Self::Bool(v) => v.to_string(),
            Self::String(s) => {
                if s.len() <= 80 {
                    format!("{s:?}")
                } else {
                    format!("{:?}…({} bytes)", &s[..80], s.len())
                }
            }
            Self::Array { elem_type, len, .. } => format!("[{elem_type:?}; {len}]"),
            other => format!("{other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ttype: GgmlType,
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub n_tensors: u64,
    pub n_meta_kv: u64,
    pub meta: Vec<(String, MetaValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset of the tensor-info table (after metadata, before tensor 0).
    pub tensor_table_offset: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
}

struct Cur<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn position(&self) -> u64 {
        self.pos as u64
    }
    /// Bytes left to read from the current position. Used to bound an untrusted
    /// declared length (string/array) against the file before allocating, so a
    /// forged length can never drive an oversized allocation.
    fn remaining(&self) -> u64 {
        (self.buf.len() - self.pos.min(self.buf.len())) as u64
    }
    fn read_exact(&mut self, dst: &mut [u8]) -> Result<()> {
        let end = self
            .pos
            .checked_add(dst.len())
            .ok_or_else(|| anyhow!("read overflow"))?;
        if end > self.buf.len() {
            bail!(
                "unexpected EOF at pos {} (want {} more)",
                self.pos,
                dst.len()
            );
        }
        dst.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(())
    }
    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn read_f32(&mut self) -> Result<f32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn read_f64(&mut self) -> Result<f64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        // GGUF metadata + tensor index live at the *head* of the file
        // (typically a few MB). Mmap the file `PROT_READ`/`MAP_PRIVATE`
        // so only the header pages are demand-faulted — avoids slurping
        // the 81 GB DS4 GGUF into a `Vec<u8>` just to parse the index.
        let mmap = crate::layer_view::MmapSource::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let bytes = mmap.as_slice();
        let file_size = bytes.len() as u64;
        let mut cur = Cur::new(bytes);

        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})");
        }
        let version = cur.read_u32()?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            bail!(
                "unsupported GGUF version {version} (supported: {:?})",
                SUPPORTED_VERSIONS
            );
        }
        let n_tensors = cur.read_u64()?;
        let n_meta_kv = cur.read_u64()?;

        // Harden against a forged/corrupt header: these counts come straight from
        // the untrusted file and would otherwise drive an unbounded
        // `Vec::with_capacity` (a `n_tensors = 2^60` aborts the process before the
        // EOF check ever fires). Bound each against the smallest on-disk size a
        // single entry can occupy, so an absurd count is rejected before any large
        // allocation. A metadata KV is >= 12 bytes (8-byte key len + 4-byte value
        // tag); a tensor-info entry is >= 24 bytes (name len + n_dims + type +
        // offset). These are deliberate under-estimates — they never reject a
        // well-formed file, only impossible counts.
        if n_meta_kv > file_size / 12 {
            bail!("GGUF metadata count {n_meta_kv} exceeds the {file_size}-byte file's capacity — corrupt or hostile");
        }
        if n_tensors > file_size / 24 {
            bail!("GGUF tensor count {n_tensors} exceeds the {file_size}-byte file's capacity — corrupt or hostile");
        }

        // Pre-reserve only a sane amount: the count is already bounded vs
        // `file_size` above, but on a genuinely huge file that bound can still be
        // billions, so `with_capacity(count)` would itself be a large speculative
        // allocation. Reserve a small floor and let the Vec grow — a forged count
        // then just hits EOF in `read_string`/`read_value` and errors cleanly.
        let mut meta = Vec::with_capacity((n_meta_kv as usize).min(4096));
        for i in 0..n_meta_kv {
            let key =
                read_string(&mut cur).with_context(|| format!("reading metadata key #{i}"))?;
            let v = read_value(&mut cur)
                .with_context(|| format!("reading metadata value for key {key:?}"))?;
            meta.push((key, v));
        }

        // Byte offset where the tensor-info table begins (right after the
        // metadata KVs). The offline q4 re-quantizer copies [0, here) verbatim
        // and rewrites only the (same-length) table in place.
        let tensor_table_offset = cur.position();

        let mut tensors = Vec::with_capacity((n_tensors as usize).min(4096));
        for i in 0..n_tensors {
            let name =
                read_string(&mut cur).with_context(|| format!("reading tensor name #{i}"))?;
            let n_dims = cur.read_u32()?;
            if n_dims > MAX_TENSOR_DIMS {
                bail!("tensor #{i} declares {n_dims} dims (max {MAX_TENSOR_DIMS}) — corrupt");
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let ttype = GgmlType::from_u32(cur.read_u32()?)?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ttype,
                offset,
            });
        }

        // `general.alignment` is attacker-controlled metadata. A forged value of 0
        // would make `pos % alignment` a divide-by-zero panic; a non-power-of-two
        // is meaningless. Reject both and fall back to the GGUF default of 32 only
        // for a missing key (a present-but-bogus value is treated as hostile).
        let alignment = match meta.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, MetaValue::U32(v))) => {
                let a = *v as u64;
                if a == 0 || !a.is_power_of_two() {
                    bail!("GGUF general.alignment {a} is not a positive power of two — corrupt or hostile");
                }
                a
            }
            _ => 32,
        };
        let pos = cur.position();
        let pad = (alignment - (pos % alignment)) % alignment;
        let tensor_data_offset = pos.checked_add(pad).ok_or_else(|| {
            anyhow!("tensor data offset overflow (pos {pos} + pad {pad}) — corrupt or hostile")
        })?;

        Ok(Self {
            version,
            n_tensors,
            n_meta_kv,
            meta,
            tensors,
            tensor_table_offset,
            tensor_data_offset,
            file_size,
        })
    }

    #[allow(dead_code)] // Used by tests + future decode flow when reading model hyperparameters.
    pub fn get_meta(&self, key: &str) -> Option<&MetaValue> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensor_byte_size(&self, t: &TensorInfo) -> u64 {
        // Overflow-safe: dims come from the file, so the element product and the
        // byte-size multiply must not wrap. Saturating to u64::MAX makes any
        // impossible tensor fail the downstream `bytes_for` bounds-check against
        // the file size, rather than wrapping into a small in-range value.
        let n_elems: u64 = t.dims.iter().copied().fold(1u64, |a, d| a.saturating_mul(d));
        let bs = (t.ttype.block_size() as u64).max(1);
        let ts = t.ttype.type_size() as u64;
        (n_elems / bs).saturating_mul(ts)
    }
}

fn read_string(cur: &mut Cur) -> Result<String> {
    let len = cur.read_u64()?;
    // `len` is an untrusted u64. Bound it BEFORE allocating, on two independent
    // axes, so neither a giant value nor a merely-larger-than-the-file value can
    // drive an oversized `vec![0u8; len]` (a `len = 2^60` aborts the process long
    // before the EOF check in `read_exact` ever fires):
    //   1. A hard 1 MiB ceiling — no legitimate GGUF key/name/string value is
    //      anywhere near that, so this rejects gross corruption cheaply.
    //   2. The bytes actually remaining in the file — a string can never be
    //      longer than what is left to read, so this catches a forged length in a
    //      small file before any allocation, independent of the 1 MiB ceiling.
    if len > 1024 * 1024 {
        bail!("string length {len} suspiciously large — likely corrupt offset");
    }
    let remaining = cur.remaining();
    if len > remaining {
        bail!("string length {len} exceeds {remaining} bytes remaining in file — corrupt or hostile");
    }
    let mut buf = vec![0u8; len as usize];
    cur.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("non-UTF8 string in metadata: {e}"))
}

/// Cap on nested-array depth. GGUF arrays are flat in every real model; a
/// hostile file can declare array-of-array-of-array… with no natural bound,
/// which would recurse `read_value_of` until the thread stack overflows
/// (a process abort, not a catchable `Err`). 64 is far beyond anything a real
/// GGUF uses and keeps recursion well inside the stack.
const MAX_ARRAY_DEPTH: u32 = 64;

fn read_value(cur: &mut Cur) -> Result<MetaValue> {
    let tag = GgufType::from_u32(cur.read_u32()?)?;
    read_value_of(cur, tag, 0)
}

fn read_value_of(cur: &mut Cur, tag: GgufType, depth: u32) -> Result<MetaValue> {
    Ok(match tag {
        GgufType::U8 => MetaValue::U8(cur.read_u8()?),
        GgufType::I8 => MetaValue::I8(cur.read_i8()?),
        GgufType::U16 => MetaValue::U16(cur.read_u16()?),
        GgufType::I16 => MetaValue::I16(cur.read_i16()?),
        GgufType::U32 => MetaValue::U32(cur.read_u32()?),
        GgufType::I32 => MetaValue::I32(cur.read_i32()?),
        GgufType::F32 => MetaValue::F32(cur.read_f32()?),
        GgufType::Bool => MetaValue::Bool(cur.read_u8()? != 0),
        GgufType::String => MetaValue::String(read_string(cur)?),
        GgufType::U64 => MetaValue::U64(cur.read_u64()?),
        GgufType::I64 => MetaValue::I64(cur.read_i64()?),
        GgufType::F64 => MetaValue::F64(cur.read_f64()?),
        GgufType::Array => {
            if depth >= MAX_ARRAY_DEPTH {
                bail!("array nesting exceeds depth {MAX_ARRAY_DEPTH} — corrupt or hostile");
            }
            let elem = GgufType::from_u32(cur.read_u32()?)?;
            let len = cur.read_u64()?;
            if len > 2_000_000 {
                bail!("array length {len} too large — refusing to allocate");
            }
            // Even under the 2M cap, `len` can exceed the bytes left in the file
            // (every element consumes >= 1 byte), so bound it vs remaining BEFORE
            // reserving — a forged length then errors without a speculative
            // `with_capacity(len)` allocation. Reserve a small floor and grow.
            let remaining = cur.remaining();
            if len > remaining {
                bail!("array length {len} exceeds {remaining} bytes remaining — corrupt or hostile");
            }
            let mut values = Vec::with_capacity((len as usize).min(4096));
            for _ in 0..len {
                values.push(read_value_of(cur, elem, depth + 1)?);
            }
            MetaValue::Array {
                elem_type: elem,
                len,
                values,
            }
        }
    })
}

// ---------- subcommand drivers ----------

pub fn cmd_info(path: &Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    println!("path: {}", path.display());
    println!(
        "size: {} bytes ({:.2} GB)",
        g.file_size,
        g.file_size as f64 / 1e9
    );
    println!(
        "version: {}  n_tensors: {}  n_meta_kv: {}",
        g.version, g.n_tensors, g.n_meta_kv
    );
    println!(
        "tensor_data_offset: {} (0x{:x})",
        g.tensor_data_offset, g.tensor_data_offset
    );
    println!();
    println!("--- key metadata ---");
    // antirez reads keys like deepseek4.expert_count / expert_used_count / expert_group_count
    // / expert_group_used_count (ds4.c:1256-1259). Mirror that vocabulary here.
    let interesting_prefixes = [
        "general.",
        "deepseek",
        "dsv4",
        "llama.",
        "qwen",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        ".block_count",
        ".embedding_length",
        ".feed_forward_length",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".attention.key_length",
        ".attention.value_length",
        ".expert_count",
        ".expert_used_count",
        ".expert_group_count",
        ".expert_group_used_count",
        ".rope.",
    ];
    for (k, v) in &g.meta {
        if interesting_prefixes.iter().any(|p| k.contains(p)) {
            println!("  {k} = {}", v.short());
        }
    }
    println!();
    println!("--- tensor type histogram ---");
    let mut hist: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in &g.tensors {
        let e = hist.entry(format!("{:?}", t.ttype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += g.tensor_byte_size(t);
    }
    for (ty, (count, bytes)) in &hist {
        println!(
            "  {ty:10}  {count:6} tensors  {:.2} GiB",
            *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    Ok(())
}

pub fn cmd_manifest(path: &Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    println!("name,dtype,n_dims,shape,offset,bytes,role");
    for t in &g.tensors {
        let shape = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let role = classify_tensor(&t.name);
        println!(
            "{},{:?},{},{},{},{},{}",
            t.name,
            t.ttype,
            t.dims.len(),
            shape,
            t.offset,
            g.tensor_byte_size(t),
            role
        );
    }
    Ok(())
}

/// Offline GGUF re-quantizer: write `dst` identical to `src` except that every
/// Q8_0 tensor whose name satisfies `should_requant` is re-quantized to Q4_0
/// (18 B/32w vs 34 B/32w) via [`crate::layer_view::requant_q8_0_to_q4_0`]. This
/// is the production half of the low-RAM variant: the q4 weights live ON DISK,
/// so the model mmaps them directly with no load-time requant transient (the
/// load-time path holds both the q8 and q4 copies; see DS4_Q4_LMHEAD notes).
///
/// Header + metadata are copied byte-for-byte; only the tensor-info table is
/// rewritten in place (same byte length — only the `type` u32 and `offset` u64
/// VALUES change), so `tensor_data_offset` is unchanged. The data section is
/// re-laid-out in table order at `alignment`-aligned offsets (re-quantized
/// tensors shrink, shifting everything after them).
pub fn requant_gguf_q8_0_to_q4_0(
    src: &Path,
    dst: &Path,
    should_requant: &dyn Fn(&str) -> bool,
) -> Result<()> {
    use std::io::Write;

    let g = GgufFile::open(src)?;
    let mmap = crate::layer_view::MmapSource::open(src)
        .with_context(|| format!("mmap {}", src.display()))?;
    let bytes = mmap.as_slice();

    let alignment = match g.meta.iter().find(|(k, _)| k == "general.alignment") {
        Some((_, MetaValue::U32(v))) => *v as u64,
        _ => 32,
    };
    let align_up = |x: u64| -> u64 { (x + alignment - 1) / alignment * alignment };

    // Plan the new data layout (table order = data order in GGUF).
    struct Plan {
        requant: bool,
        old_data_off: u64, // relative to tensor_data_offset (src)
        old_size: u64,
        new_off: u64, // relative to tensor_data_offset (dst)
        new_size: u64,
    }
    let q4_tag: u32 = 2; // GgmlType::Q4_0
    let mut plans = Vec::with_capacity(g.tensors.len());
    let mut cursor = 0u64;
    let mut n_requant = 0usize;
    for t in &g.tensors {
        let old_size = g.tensor_byte_size(t);
        let do_req = t.ttype == GgmlType::Q8_0 && should_requant(&t.name);
        let n_elems: u64 = t.dims.iter().product();
        let new_size = if do_req { (n_elems / 32) * 18 } else { old_size };
        let new_off = cursor;
        if do_req {
            n_requant += 1;
        }
        plans.push(Plan {
            requant: do_req,
            old_data_off: t.offset,
            old_size,
            new_off,
            new_size,
        });
        cursor = align_up(new_off + new_size);
    }
    anyhow::ensure!(n_requant > 0, "requant_gguf: no Q8_0 tensor matched the predicate");

    // Copy the header region [0, tensor_data_offset) and patch the table in place.
    let data_off = g.tensor_data_offset;
    let mut header = bytes[..data_off as usize].to_vec();
    let mut p = g.tensor_table_offset as usize;
    for (t, plan) in g.tensors.iter().zip(&plans) {
        // entry layout: name_len(u64) name(bytes) n_dims(u32) dims(u64*) type(u32) offset(u64)
        p += 8 + t.name.len();
        p += 4; // n_dims
        p += 8 * t.dims.len(); // dims
        if plan.requant {
            header[p..p + 4].copy_from_slice(&q4_tag.to_le_bytes());
        }
        p += 4; // type
        header[p..p + 8].copy_from_slice(&plan.new_off.to_le_bytes());
        p += 8; // offset
    }
    debug_assert_eq!(p as u64, /* end of table */ {
        let mut e = g.tensor_table_offset;
        for t in &g.tensors {
            e += 8 + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
        }
        e
    });

    // Stream the new file.
    let f = std::fs::File::create(dst).with_context(|| format!("create {}", dst.display()))?;
    let mut out = std::io::BufWriter::with_capacity(8 << 20, f);
    out.write_all(&header)?;
    let mut written: u64 = 0; // bytes after data_off
    let zeros = [0u8; 64];
    for (t, plan) in g.tensors.iter().zip(&plans) {
        while written < plan.new_off {
            let pad = (plan.new_off - written).min(zeros.len() as u64) as usize;
            out.write_all(&zeros[..pad])?;
            written += pad as u64;
        }
        let src0 = (data_off + plan.old_data_off) as usize;
        let src_slice = &bytes[src0..src0 + plan.old_size as usize];
        if plan.requant {
            let q4 = crate::layer_view::requant_q8_0_to_q4_0(src_slice)?;
            anyhow::ensure!(q4.len() as u64 == plan.new_size, "requant size mismatch for {}", t.name);
            out.write_all(&q4)?;
            written += q4.len() as u64;
        } else {
            out.write_all(src_slice)?;
            written += plan.old_size;
        }
    }
    out.flush()?;
    eprintln!(
        "requant_gguf: {} → {} ({} tensor(s) Q8_0→Q4_0, data {} → {} bytes)",
        src.display(),
        dst.display(),
        n_requant,
        bytes.len() as u64 - data_off,
        written,
    );
    Ok(())
}

/// CLI: `requant <src.gguf> <dst.gguf>` — re-quantize the LM head (output.weight)
/// Q8_0→Q4_0 on disk. Extend the predicate to cover the attention/shared
/// projections once their q4 decode paths land.
pub fn cmd_requant(src: &Path, dst: &Path) -> Result<()> {
    requant_gguf_q8_0_to_q4_0(src, dst, &|name| name == "output.weight")
}

/// Per-role expected ggml type for DS4 Q2-imatrix quantisation.
///
/// Source: antirez/ds4 Q2 imatrix conversion docs (README) — routed
/// gate/up = IQ2_XXS, routed down = Q2_K, everything else = F16. If a
/// future Q4_K or Q8_0 GGUF is loaded, the table needs extending.
const DS4_Q2_LAYOUT: &[(&str, GgmlType)] = &[
    ("embed", GgmlType::F16),
    // Out = Q8 (per `OutQ8` in the antirez DS4 IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 layout)
    ("lm_head", GgmlType::Q8_0),
    ("final_norm", GgmlType::F32),
    // Output-side HC fold (antirez ds4.c:2143-2145):
    //   output_hc_base  : F32 [n_hc]
    //   output_hc_fn    : F16 [hc_dim, n_hc]   (hc_dim = d_model * n_hc)
    //   output_hc_scale : F32 [1]
    ("output_hc_base", GgmlType::F32),
    ("output_hc_fn", GgmlType::F16),
    ("output_hc_scale", GgmlType::F32),
    // Per-layer
    ("attn_norm", GgmlType::F32),
    ("ffn_norm", GgmlType::F32),
    ("mla_q_a_norm", GgmlType::F32),
    ("mla_kv_a_norm", GgmlType::F32),
    ("attn_compressor_norm", GgmlType::F32),
    ("indexer_compressor_norm", GgmlType::F32),
    // Compressor + indexer projection tensors are F16 per antirez
    // tensor_expect_layout (ds4.c:2169-2182). Only present on compressed
    // layers (il >= 2); LayerView::handles.get returns None for il in {0,1}
    // and the AttnLayerWeights::from_layer_view path keeps the field empty.
    ("attn_compressor_kv", GgmlType::F16),
    ("attn_compressor_gate", GgmlType::F16),
    ("attn_compressor_ape", GgmlType::F16),
    ("indexer_compressor_kv", GgmlType::F16),
    ("indexer_compressor_gate", GgmlType::F16),
    ("indexer_compressor_ape", GgmlType::F16),
    // Indexer projection weights (ds4.c:2177-2178). F16, only present on
    // compressed layers with ratio==4 (every 4th compressed layer in the
    // 43-layer DS4 stack); LayerView::handles.get returns None elsewhere.
    ("indexer_attn_q_b", GgmlType::F16),
    ("indexer_proj", GgmlType::F16),
    // MLA Q/KV projections are Q8 in the AProjQ8 layout (ground-truthed
    // against DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 GGUF).
    ("mla_q_a", GgmlType::Q8_0),
    ("mla_q_b", GgmlType::Q8_0),
    ("mla_kv", GgmlType::Q8_0),
    // AProj = Q8 (attention output projection)
    ("attn_o", GgmlType::Q8_0),
    ("attn_output_a", GgmlType::Q8_0),
    ("attn_output_b", GgmlType::Q8_0),
    ("attn_sinks", GgmlType::F32),
    ("moe_router", GgmlType::F16),
    ("moe_router_bias", GgmlType::F32),
    ("moe_routing_table", GgmlType::I32),
    ("moe_gate_experts", GgmlType::IQ2_XXS),
    ("moe_up_experts", GgmlType::IQ2_XXS),
    ("moe_down_experts", GgmlType::Q2_K),
    // SExp = Q8 (shared MoE experts)
    ("moe_gate_shared", GgmlType::Q8_0),
    ("moe_up_shared", GgmlType::Q8_0),
    ("moe_down_shared", GgmlType::Q8_0),
];

/// Quant types for which `cpu_via_dequant` has a block-level oracle.
/// Lifted from `ds4_metal/src/cpu_via_dequant.rs` coverage.
const DEQUANT_ORACLE_TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_K,
    GgmlType::Q2_K,
    GgmlType::IQ2_XXS,
    GgmlType::Q8_0,
    GgmlType::I32,
];

/// 88 GB ceiling = 96 GB unified memory minus headroom for KV cache, Metal
/// scratch, and the OS page cache for the GGUF mmap.
pub const DS4_TENSOR_BYTES_LIMIT: u64 = 88 * 1024 * 1024 * 1024;

/// Read an integer metadata value, trying multiple keys in order. Accepts
/// U32/I32/U64/I64; coerces signed via `>= 0`. Returns `None` if no key
/// matches.
pub fn meta_u32(g: &GgufFile, keys: &[&str]) -> Option<u32> {
    for k in keys {
        if let Some(v) = g.get_meta(k) {
            match v {
                MetaValue::U32(x) => return Some(*x),
                MetaValue::I32(x) if *x >= 0 => return Some(*x as u32),
                MetaValue::U64(x) => return Some(*x as u32),
                MetaValue::I64(x) if *x >= 0 => return Some(*x as u32),
                _ => {}
            }
        }
    }
    None
}

/// Read a float metadata value, trying multiple keys in order. Accepts F32
/// directly and integer types coerced to f32. Returns `None` if no key
/// matches.
pub fn meta_f32(g: &GgufFile, keys: &[&str]) -> Option<f32> {
    for k in keys {
        if let Some(v) = g.get_meta(k) {
            match v {
                MetaValue::F32(x) => return Some(*x),
                MetaValue::F64(x) => return Some(*x as f32),
                MetaValue::U32(x) => return Some(*x as f32),
                MetaValue::I32(x) => return Some(*x as f32),
                _ => {}
            }
        }
    }
    None
}

/// Outcome of a DS4-Q2 GGUF layout check: enough to drive `MetalDispatcher`
/// setup without re-walking the tensor index.
#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub path: std::path::PathBuf,
    pub n_layers: u32,
    pub d_model: u32,
    pub vocab_size: u32,
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub total_tensor_bytes: u64,
    pub per_type_bytes: std::collections::BTreeMap<String, u64>,
    pub roles_seen: std::collections::BTreeSet<String>,
}

/// One-shot DS4-Q2 GGUF validator. Returns a `ModelManifest` on success or
/// the first violation as an error.
///
/// Checks performed:
///   1. Every classified-role tensor has the expected `GgmlType` per
///      `DS4_Q2_LAYOUT`.
///   2. Every tensor's `GgmlType` is in `DEQUANT_ORACLE_TYPES` (so the
///      block-level oracle in `cpu_via_dequant.rs` can dequantise it for
///      the trace-equality test).
///   3. `Σ tensor_byte_size ≤ DS4_TENSOR_BYTES_LIMIT` (88 GB).
pub fn validate_ds4_layout(path: &Path) -> Result<ModelManifest> {
    let g = GgufFile::open(path)?;

    let layout: std::collections::BTreeMap<&str, GgmlType> =
        DS4_Q2_LAYOUT.iter().copied().collect();

    let mut per_type_bytes: std::collections::BTreeMap<String, u64> = Default::default();
    let mut roles_seen: std::collections::BTreeSet<String> = Default::default();
    let mut total: u64 = 0;

    for t in &g.tensors {
        let role = classify_tensor(&t.name);
        roles_seen.insert(role.to_string());
        let bytes = g.tensor_byte_size(t);
        // `tensor_byte_size` saturates to u64::MAX on a forged dim product, so the
        // running sum must saturate too — a plain `+=` would wrap in release and
        // under-report the total, letting an over-budget/forged file slip the
        // DS4_TENSOR_BYTES_LIMIT check below.
        total = total.saturating_add(bytes);
        let e = per_type_bytes.entry(format!("{:?}", t.ttype)).or_insert(0);
        *e = e.saturating_add(bytes);

        // Low-RAM variant: the lm_head (output.weight) may be Q4_0 (offline
        // re-quant, requant_gguf_q8_0_to_q4_0). It's read by the dedicated q4
        // decode tail, not the generic cpu_via_dequant oracle, so skip both the
        // oracle requirement and the "layout expects Q8_0" check for that role.
        let is_q4_lm_head = role == "lm_head" && t.ttype == GgmlType::Q4_0;
        if !is_q4_lm_head {
            if !DEQUANT_ORACLE_TYPES.contains(&t.ttype) {
                bail!(
                    "tensor {:?} (role={}) has type {:?} which has no dequant oracle in cpu_via_dequant.rs; extend the oracle before loading",
                    t.name, role, t.ttype,
                );
            }
            if role != "other" {
                if let Some(&expected) = layout.get(role) {
                    if t.ttype != expected {
                        bail!(
                            "tensor {:?} (role={}) has type {:?}, DS4-Q2 layout expects {:?}",
                            t.name,
                            role,
                            t.ttype,
                            expected,
                        );
                    }
                }
            }
        }
    }

    if total > DS4_TENSOR_BYTES_LIMIT {
        // SSD-streaming opt-in: weights stay mmap-backed and page in on
        // demand (expert residency pinning is skipped — see
        // request_model_residency), so models far larger than RAM can run
        // at SSD-bound throughput.
        if std::env::var("DS4_SSD_STREAM").is_ok() {
            eprintln!(
                "ds4: tensor data {:.2} GB exceeds resident budget {:.2} GB — SSD-streaming mode (DS4_SSD_STREAM)",
                total as f64 / 1e9,
                DS4_TENSOR_BYTES_LIMIT as f64 / 1e9,
            );
        } else {
            bail!(
                "tensor data {:.2} GB exceeds DS4 budget {:.2} GB (96 GB - 8 GB headroom); refusing to load (set DS4_SSD_STREAM=1 to stream from SSD)",
                total as f64 / 1e9,
                DS4_TENSOR_BYTES_LIMIT as f64 / 1e9,
            );
        }
    }

    let n_layers = meta_u32(
        &g,
        &[
            "deepseek4.block_count",
            "deepseek2.block_count",
            "llama.block_count",
        ],
    )
    .ok_or_else(|| anyhow!("missing block_count metadata"))?;
    let d_model = meta_u32(
        &g,
        &[
            "deepseek4.embedding_length",
            "deepseek2.embedding_length",
            "llama.embedding_length",
        ],
    )
    .ok_or_else(|| anyhow!("missing embedding_length metadata"))?;
    let vocab_size = meta_u32(&g, &["deepseek4.vocab_size", "tokenizer.ggml.tokens.len"])
        .or_else(|| {
            g.tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.last().map(|d| *d as u32))
        })
        .ok_or_else(|| {
            anyhow!("missing vocab_size metadata and could not infer from token_embd.weight")
        })?;
    let n_experts =
        meta_u32(&g, &["deepseek4.expert_count", "deepseek2.expert_count"]).unwrap_or(0);
    let n_experts_used = meta_u32(
        &g,
        &["deepseek4.expert_used_count", "deepseek2.expert_used_count"],
    )
    .unwrap_or(0);

    Ok(ModelManifest {
        path: path.to_path_buf(),
        n_layers,
        d_model,
        vocab_size,
        n_experts,
        n_experts_used,
        total_tensor_bytes: total,
        per_type_bytes,
        roles_seen,
    })
}

pub fn cmd_check(path: &Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    let mut total_tensor_bytes: u64 = 0;
    for t in &g.tensors {
        // Saturating: tensor_byte_size can be u64::MAX on a forged tensor; a plain
        // `+=` would wrap (release) and make `expected_min` small, wrongly passing
        // the truncation check below.
        total_tensor_bytes = total_tensor_bytes.saturating_add(g.tensor_byte_size(t));
    }
    let expected_min = g
        .tensor_data_offset
        .saturating_add(total_tensor_bytes);
    if g.file_size < expected_min {
        bail!(
            "file truncated: size {} < expected min {}",
            g.file_size,
            expected_min
        );
    }
    let slack = g.file_size - expected_min;
    println!(
        "OK: {} tensors, {:.2} GiB tensor data, slack {} bytes",
        g.n_tensors,
        total_tensor_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        slack
    );
    Ok(())
}

pub(crate) fn classify_tensor(name: &str) -> &'static str {
    // Names refined against antirez ds4.c (benchmarks/ds4_msl/upstream/ds4/ds4.c) —
    // grep `"blk\.%u\.` for the authoritative list. Order matters: most-specific
    // substrings first, since the classifier uses `contains`.
    if name == "token_embd.weight" {
        return "embed";
    }
    if name == "output.weight" || name == "lm_head.weight" {
        return "lm_head";
    }
    if name == "output_norm.weight" {
        return "final_norm";
    }
    // Output-side HC head (antirez `output_hc_head_one`, ds4.c:7654-7681):
    // sigmoid-weighted sum across the 8-way HC residual before the final
    // RMSNorm + lm_head matvec. Three tensors, all global (no `blk.*` prefix).
    if name == "output_hc_base.weight" {
        return "output_hc_base";
    }
    if name == "output_hc_fn.weight" {
        return "output_hc_fn";
    }
    if name == "output_hc_scale.weight" {
        return "output_hc_scale";
    }
    let n = name;
    // Indexer attention (DS4-specific) — must come before generic attn_* matches
    // because indexer.attn_q_b.weight contains ".attn_q_b" as a substring.
    if n.contains(".indexer_compressor_ape") {
        return "indexer_compressor_ape";
    }
    if n.contains(".indexer_compressor_gate") {
        return "indexer_compressor_gate";
    }
    if n.contains(".indexer_compressor_kv") {
        return "indexer_compressor_kv";
    }
    if n.contains(".indexer_compressor_norm") {
        return "indexer_compressor_norm";
    }
    if n.contains(".indexer.attn_q_b") {
        return "indexer_attn_q_b";
    }
    if n.contains(".indexer.proj") {
        return "indexer_proj";
    }
    // MLA compressor machinery (DS4-specific, KV compression)
    if n.contains(".attn_compressor_ape") {
        return "attn_compressor_ape";
    }
    if n.contains(".attn_compressor_gate") {
        return "attn_compressor_gate";
    }
    if n.contains(".attn_compressor_kv") {
        return "attn_compressor_kv";
    }
    if n.contains(".attn_compressor_norm") {
        return "attn_compressor_norm";
    }
    // Norms
    if n.contains(".attn_norm") {
        return "attn_norm";
    }
    if n.contains(".ffn_norm") {
        return "ffn_norm";
    }
    if n.contains(".attn_kv_a_norm") {
        return "mla_kv_a_norm";
    }
    if n.contains(".attn_q_a_norm") {
        return "mla_q_a_norm";
    }
    // MLA latent-compressed projections
    if n.contains(".attn_kv_a") {
        return "mla_kv_a";
    }
    if n.contains(".attn_kv_b") {
        return "mla_kv_b";
    }
    if n.contains(".attn_q_a") {
        return "mla_q_a";
    }
    if n.contains(".attn_q_b") {
        return "mla_q_b";
    }
    if n.contains(".attn_kv") {
        return "mla_kv";
    }
    if n.contains(".attn_q.") {
        return "attn_q";
    }
    if n.contains(".attn_k.") {
        return "attn_k";
    }
    if n.contains(".attn_v.") {
        return "attn_v";
    }
    // Output projection — DS4 splits into _a / _b lora-style, but pre-V4 has a single output
    if n.contains(".attn_output_a") {
        return "attn_output_a";
    }
    if n.contains(".attn_output_b") {
        return "attn_output_b";
    }
    if n.contains(".attn_output") || n.contains(".attn_o.") {
        return "attn_o";
    }
    // FlashAttention sinks (per-head learned softmax sink scalars)
    if n.contains(".attn_sinks") {
        return "attn_sinks";
    }
    // Hash Conjecture machinery — DS4-specific (M126/M127 hc_expand4)
    if n.contains(".hc_attn_base") {
        return "hc_attn_base";
    }
    if n.contains(".hc_attn_fn") {
        return "hc_attn_fn";
    }
    if n.contains(".hc_attn_scale") {
        return "hc_attn_scale";
    }
    if n.contains(".hc_ffn_base") {
        return "hc_ffn_base";
    }
    if n.contains(".hc_ffn_fn") {
        return "hc_ffn_fn";
    }
    if n.contains(".hc_ffn_scale") {
        return "hc_ffn_scale";
    }
    // MoE FFN — antirez names (ds4.c:1865-1873)
    if n.contains(".ffn_gate_tid2eid") {
        return "moe_routing_table";
    }
    if n.contains(".ffn_gate_inp") {
        return "moe_router";
    }
    if n.contains(".ffn_gate_exps") {
        return "moe_gate_experts";
    }
    if n.contains(".ffn_up_exps") {
        return "moe_up_experts";
    }
    if n.contains(".ffn_down_exps") {
        return "moe_down_experts";
    }
    if n.contains(".ffn_gate_shexp") {
        return "moe_gate_shared";
    }
    if n.contains(".ffn_up_shexp") {
        return "moe_up_shared";
    }
    if n.contains(".ffn_down_shexp") {
        return "moe_down_shared";
    }
    if n.contains(".exp_probs_b") {
        return "moe_router_bias";
    }
    // Dense FFN fallback (early layers may use dense before switching to MoE).
    if n.contains(".ffn_gate.") {
        return "ffn_gate_dense";
    }
    if n.contains(".ffn_up.") {
        return "ffn_up_dense";
    }
    if n.contains(".ffn_down.") {
        return "ffn_down_dense";
    }
    if n.contains("rope_freqs") {
        return "rope";
    }
    "other"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_type_sizes_match_emitter_constants() {
        // Duplicated from the mlir_to_msl emitter as a tripwire — if these change there,
        // this fires.
        assert_eq!(GgmlType::Q8_0.type_size(), 34, "M111 BLK_Q8_0");
        assert_eq!(GgmlType::Q2_K.type_size(), 84, "M111 block_q2_K");
        assert_eq!(GgmlType::Q4_K.type_size(), 144, "M112 block_q4_K");
        assert_eq!(GgmlType::IQ2_XXS.type_size(), 66, "M113 block_iq2_xxs");
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    // ---- hostile-header hardening (R1): a malformed/forged GGUF must Err,
    //      never panic or drive an unbounded allocation. ----

    fn gguf_header(n_tensors: u64, n_meta_kv: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // a supported version
        b.extend_from_slice(&n_tensors.to_le_bytes());
        b.extend_from_slice(&n_meta_kv.to_le_bytes());
        b
    }

    fn open_temp_gguf(bytes: &[u8], tag: &str) -> Result<GgufFile> {
        let mut p = std::env::temp_dir();
        p.push(format!("ds4_gguf_harden_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        let r = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        r
    }

    #[test]
    fn hostile_tensor_count_rejected_not_ooming() {
        // n_tensors = 2^60 in a 24-byte file: must error via the file-size bound,
        // not attempt to pre-allocate ~exabytes (which aborts before EOF).
        assert!(open_temp_gguf(&gguf_header(1u64 << 60, 0), "ntensors").is_err());
    }

    #[test]
    fn hostile_meta_count_rejected_not_ooming() {
        assert!(open_temp_gguf(&gguf_header(0, 1u64 << 60), "nmeta").is_err());
    }

    #[test]
    fn hostile_n_dims_rejected() {
        // Plausible header (1 tensor, 0 meta), then a tensor entry whose n_dims is
        // forged huge — must hit the MAX_TENSOR_DIMS cap, not pre-allocate.
        let mut b = gguf_header(1, 0);
        write_string(&mut b, "blk.0.x.weight");
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_dims
        assert!(open_temp_gguf(&b, "ndims").is_err());
    }

    #[test]
    fn truncated_header_errors_cleanly() {
        // Only the magic, nothing after — must Err, never panic.
        assert!(open_temp_gguf(&GGUF_MAGIC.to_le_bytes(), "trunc").is_err());
    }

    #[test]
    fn hostile_string_length_rejected_not_ooming() {
        // R1 remaining gap: a metadata KEY declares a u64 string length far larger
        // than the file. `read_string` MUST bound `len` against the bytes remaining
        // (and the 1 MiB ceiling) BEFORE `vec![0u8; len]`, so this errors cleanly
        // instead of attempting a multi-exabyte allocation that aborts the process.
        let mut b = gguf_header(0, 1); // 0 tensors, 1 metadata KV
        b.extend_from_slice(&(1u64 << 60).to_le_bytes()); // key string length = 2^60
        // (no key bytes follow — but we must never get far enough to read them)
        assert!(open_temp_gguf(&b, "strlen60").is_err());

        // Also a length just over the 1 MiB ceiling but plausibly "in file" if the
        // file were padded — still rejected by the ceiling, never allocated blindly.
        let mut b2 = gguf_header(0, 1);
        b2.extend_from_slice(&(2 * 1024 * 1024u64).to_le_bytes());
        assert!(open_temp_gguf(&b2, "strlen2m").is_err());

        // And a modest length that simply exceeds the bytes actually left in a small
        // file (under the 1 MiB ceiling) — caught by the remaining-bytes bound.
        let mut b3 = gguf_header(0, 1);
        b3.extend_from_slice(&4096u64.to_le_bytes()); // 4 KiB string in a ~20-byte file
        assert!(open_temp_gguf(&b3, "strlenrem").is_err());
    }

    #[test]
    fn hostile_string_length_in_metadata_value_rejected() {
        // Same gap, reached via a String-typed metadata VALUE (not just the key):
        // valid key, then value tag=String(8) with a forged 2^60 length.
        let mut b = gguf_header(0, 1);
        write_string(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // GgufType::String
        b.extend_from_slice(&(1u64 << 60).to_le_bytes()); // value string length
        assert!(open_temp_gguf(&b, "valstrlen").is_err());
    }

    #[test]
    fn hostile_array_length_rejected_not_ooming() {
        // A metadata array VALUE declares a length larger than the file. Must be
        // bounded vs remaining bytes (and the 2M cap) before `with_capacity`.
        let mut b = gguf_header(0, 1);
        write_string(&mut b, "some.array");
        b.extend_from_slice(&9u32.to_le_bytes()); // GgufType::Array
        b.extend_from_slice(&4u32.to_le_bytes()); // elem = U32
        b.extend_from_slice(&(1u64 << 60).to_le_bytes()); // array length = 2^60
        assert!(open_temp_gguf(&b, "arrlen").is_err());
    }

    #[test]
    fn hostile_nested_arrays_do_not_overflow_stack() {
        // array-of-array-of-array… with no natural depth bound would recurse
        // `read_value_of` until the stack overflows (a process abort). The
        // MAX_ARRAY_DEPTH cap must turn it into a clean Err. Build a value that is
        // `MAX_ARRAY_DEPTH + 8` arrays deep, each of length 1 wrapping the next.
        let mut b = gguf_header(0, 1);
        write_string(&mut b, "deep.array");
        let depth = (MAX_ARRAY_DEPTH + 8) as usize;
        for _ in 0..depth {
            b.extend_from_slice(&9u32.to_le_bytes()); // elem type = Array
            b.extend_from_slice(&1u64.to_le_bytes()); // length 1
        }
        // innermost terminal element would be a U8; never reached past the cap.
        b.push(0u8);
        // First read_value reads the outer tag separately, so prepend it:
        let mut full = gguf_header(0, 1);
        write_string(&mut full, "deep.array");
        full.extend_from_slice(&9u32.to_le_bytes()); // outer value tag = Array
        full.extend_from_slice(&9u32.to_le_bytes()); // outer elem type = Array
        full.extend_from_slice(&1u64.to_le_bytes());
        for _ in 0..depth {
            full.extend_from_slice(&9u32.to_le_bytes());
            full.extend_from_slice(&1u64.to_le_bytes());
        }
        full.push(0u8);
        // Must NOT abort with a stack overflow — must return Err.
        assert!(open_temp_gguf(&full, "deeparr").is_err());
    }

    #[test]
    fn hostile_zero_alignment_rejected_not_panicking() {
        // general.alignment = 0 would make `pos % alignment` a divide-by-zero
        // panic. A valid 1-tensor/1-meta file that sets alignment=0 must Err.
        let mut b = gguf_header(1, 1);
        write_string(&mut b, "general.alignment");
        b.extend_from_slice(&4u32.to_le_bytes()); // GgufType::U32
        b.extend_from_slice(&0u32.to_le_bytes()); // alignment = 0  (hostile)
        write_string(&mut b, "blk.0.x.weight");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        assert!(open_temp_gguf(&b, "align0").is_err());

        // A non-power-of-two alignment is equally bogus.
        let mut b2 = gguf_header(1, 1);
        write_string(&mut b2, "general.alignment");
        b2.extend_from_slice(&4u32.to_le_bytes());
        b2.extend_from_slice(&7u32.to_le_bytes()); // alignment = 7 (not pow2)
        write_string(&mut b2, "blk.0.x.weight");
        b2.extend_from_slice(&1u32.to_le_bytes());
        b2.extend_from_slice(&4u64.to_le_bytes());
        b2.extend_from_slice(&0u32.to_le_bytes());
        b2.extend_from_slice(&0u64.to_le_bytes());
        assert!(open_temp_gguf(&b2, "align7").is_err());
    }

    // ---- Property/fuzz harness (R1): the parser must ALWAYS return Ok/Err and
    //      NEVER panic, OOM, or abort on ANY byte stream. Self-contained
    //      (deterministic xorshift PRNG) to honour the crate's "zero crates
    //      beyond anyhow" rule — no proptest/arbitrary/cargo-fuzz dependency.
    //      Run under `cargo +nightly miri test` for UB coverage of the slice/
    //      from_raw_parts paths exercised on the valid inputs. ----

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
    }

    /// Build a "structurally plausible but hostile" GGUF: a real magic + version,
    /// then random/forged counts and bytes. Every field is drawn from the PRNG so
    /// the fuzzer covers forged counts, lengths, tags, dims, offsets, and
    /// truncation simultaneously.
    fn fuzz_one(rng: &mut XorShift) -> Vec<u8> {
        let mut b = Vec::new();
        // 25% of the time, a totally random blob (also exercises the magic check).
        if rng.next() % 4 == 0 {
            let n = (rng.next() % 256) as usize;
            for _ in 0..n {
                b.push(rng.byte());
            }
            return b;
        }
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Sometimes a valid version, sometimes garbage.
        let ver = if rng.next() % 2 == 0 { 3u32 } else { rng.next() as u32 };
        b.extend_from_slice(&ver.to_le_bytes());
        // Forged counts: small, or absurd (the hardening must reject absurd ones).
        let nt = if rng.next() % 2 == 0 { rng.next() % 8 } else { rng.next() };
        let nm = if rng.next() % 2 == 0 { rng.next() % 8 } else { rng.next() };
        b.extend_from_slice(&nt.to_le_bytes());
        b.extend_from_slice(&nm.to_le_bytes());
        // Append a random tail of forged metadata/tensor bytes.
        let tail = (rng.next() % 512) as usize;
        for _ in 0..tail {
            b.push(rng.byte());
        }
        // Occasionally inject a forged-huge length token somewhere in the tail.
        if rng.next() % 3 == 0 && b.len() >= 8 {
            let at = (rng.next() as usize) % (b.len() - 7);
            b[at..at + 8].copy_from_slice(&(1u64 << 60).to_le_bytes());
        }
        b
    }

    #[test]
    fn fuzz_parser_never_panics_or_ooms() {
        // Deterministic seed → reproducible. Each iteration writes a hostile byte
        // stream and asserts GgufFile::open returns Ok/Err (the assertion is
        // implicit: a panic/abort/OOM would fail the test process). Miri runs a
        // reduced count for speed.
        let iters = if cfg!(miri) { 64 } else { 20_000 };
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        let dir = std::env::temp_dir();
        let p = dir.join(format!("ds4_gguf_fuzz_{}.gguf", std::process::id()));
        for i in 0..iters {
            let bytes = fuzz_one(&mut rng);
            std::fs::write(&p, &bytes).unwrap();
            // The result is intentionally ignored: we only require that `open`
            // RETURNS (Ok or Err) rather than panicking/aborting. A forged input
            // that happens to parse Ok is fine — downstream bounds checks
            // (bytes_for) guard actual data access.
            let _ = std::panic::catch_unwind(|| {
                let _ = GgufFile::open(&p);
            })
            .map_err(|_| panic!("GgufFile::open PANICKED on fuzz iteration {i} (bytes={bytes:?})"));
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn tensor_byte_size_saturates_on_overflow() {
        // A tensor whose dim product overflows u64 must saturate (→ huge, caught
        // downstream), not wrap to a small in-range value.
        let g = GgufFile {
            version: 3,
            n_tensors: 1,
            n_meta_kv: 0,
            meta: vec![],
            tensors: vec![],
            tensor_table_offset: 0,
            tensor_data_offset: 0,
            file_size: 1024,
        };
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![u64::MAX, u64::MAX],
            ttype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(g.tensor_byte_size(&t), u64::MAX);
    }

    #[test]
    fn tensor_classifier_matches_antirez_names() {
        // Names sourced from antirez ds4.c (the layer struct + tensor_by_namef call sites).
        assert_eq!(classify_tensor("token_embd.weight"), "embed");
        assert_eq!(classify_tensor("output.weight"), "lm_head");
        assert_eq!(classify_tensor("output_norm.weight"), "final_norm");
        assert_eq!(classify_tensor("blk.0.attn_norm.weight"), "attn_norm");
        assert_eq!(classify_tensor("blk.0.ffn_norm.weight"), "ffn_norm");
        assert_eq!(
            classify_tensor("blk.0.attn_kv_a_norm.weight"),
            "mla_kv_a_norm"
        );
        assert_eq!(classify_tensor("blk.0.attn_q_a.weight"), "mla_q_a");
        assert_eq!(classify_tensor("blk.0.attn_q_b.weight"), "mla_q_b");
        assert_eq!(classify_tensor("blk.0.attn_kv_a.weight"), "mla_kv_a");
        assert_eq!(classify_tensor("blk.0.attn_kv_b.weight"), "mla_kv_b");
        assert_eq!(classify_tensor("blk.5.attn_kv.weight"), "mla_kv");
        assert_eq!(classify_tensor("blk.5.attn_output.weight"), "attn_o");
        assert_eq!(classify_tensor("blk.5.ffn_gate_inp.weight"), "moe_router");
        assert_eq!(
            classify_tensor("blk.5.ffn_gate_tid2eid"),
            "moe_routing_table"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_gate_exps.weight"),
            "moe_gate_experts"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_up_exps.weight"),
            "moe_up_experts"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_down_exps.weight"),
            "moe_down_experts"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_gate_shexp.weight"),
            "moe_gate_shared"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_up_shexp.weight"),
            "moe_up_shared"
        );
        assert_eq!(
            classify_tensor("blk.5.ffn_down_shexp.weight"),
            "moe_down_shared"
        );
        // MLA compressor machinery (DS4-specific, KV compression)
        assert_eq!(
            classify_tensor("blk.0.attn_compressor_ape.weight"),
            "attn_compressor_ape"
        );
        assert_eq!(
            classify_tensor("blk.0.attn_compressor_gate.weight"),
            "attn_compressor_gate"
        );
        assert_eq!(
            classify_tensor("blk.0.attn_compressor_kv.weight"),
            "attn_compressor_kv"
        );
        assert_eq!(
            classify_tensor("blk.0.attn_compressor_norm.weight"),
            "attn_compressor_norm"
        );
        // Split output projection (DS4 _a / _b lora-style)
        assert_eq!(
            classify_tensor("blk.0.attn_output_a.weight"),
            "attn_output_a"
        );
        assert_eq!(
            classify_tensor("blk.0.attn_output_b.weight"),
            "attn_output_b"
        );
        // FlashAttention sinks
        assert_eq!(classify_tensor("blk.0.attn_sinks.weight"), "attn_sinks");
        // Hash Conjecture machinery
        assert_eq!(classify_tensor("blk.0.hc_attn_base.weight"), "hc_attn_base");
        assert_eq!(classify_tensor("blk.0.hc_attn_fn.weight"), "hc_attn_fn");
        assert_eq!(
            classify_tensor("blk.0.hc_attn_scale.weight"),
            "hc_attn_scale"
        );
        assert_eq!(classify_tensor("blk.0.hc_ffn_base.weight"), "hc_ffn_base");
        assert_eq!(classify_tensor("blk.0.hc_ffn_fn.weight"), "hc_ffn_fn");
        assert_eq!(classify_tensor("blk.0.hc_ffn_scale.weight"), "hc_ffn_scale");
        // Indexer attention — must NOT collide with plain attn_q_b
        assert_eq!(
            classify_tensor("blk.0.indexer.attn_q_b.weight"),
            "indexer_attn_q_b"
        );
        assert_eq!(classify_tensor("blk.0.indexer.proj.weight"), "indexer_proj");
        assert_eq!(
            classify_tensor("blk.0.indexer_compressor_ape.weight"),
            "indexer_compressor_ape"
        );
        assert_eq!(
            classify_tensor("blk.0.indexer_compressor_gate.weight"),
            "indexer_compressor_gate"
        );
        assert_eq!(
            classify_tensor("blk.0.indexer_compressor_kv.weight"),
            "indexer_compressor_kv"
        );
        assert_eq!(
            classify_tensor("blk.0.indexer_compressor_norm.weight"),
            "indexer_compressor_norm"
        );
        // MoE router bias (exp_probs_b)
        assert_eq!(
            classify_tensor("blk.0.exp_probs_b.weight"),
            "moe_router_bias"
        );
        // Order-dependency sanity: indexer.attn_q_b must NOT classify as mla_q_b,
        // and attn_compressor_norm must NOT classify as attn_norm.
        assert_ne!(classify_tensor("blk.0.indexer.attn_q_b.weight"), "mla_q_b");
        assert_ne!(
            classify_tensor("blk.0.attn_compressor_norm.weight"),
            "attn_norm"
        );
    }

    /// Build a minimal valid GGUF v3 with one F32 metadata + one F32 tensor.
    fn write_minimal_gguf(path: &Path) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_meta_kv
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type = String
        write_string(&mut buf, "ds4");
        write_string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn round_trip_minimal_gguf() {
        let tmp = std::env::temp_dir().join("ds4_engine_test_gguf_min.gguf");
        write_minimal_gguf(&tmp);
        let g = GgufFile::open(&tmp).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.n_tensors, 1);
        assert_eq!(g.tensors[0].name, "token_embd.weight");
        assert_eq!(g.tensors[0].ttype, GgmlType::F32);
        assert_eq!(g.tensors[0].dims, vec![4]);
        match g.get_meta("general.architecture") {
            Some(MetaValue::String(s)) => assert_eq!(s, "ds4"),
            other => panic!("unexpected meta: {other:?}"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// Offline q4 re-quant writer: output.weight Q8_0 → Q4_0, a trailing F32
    /// tensor must keep its data at a correctly-shifted offset, and the file
    /// must re-open cleanly.
    #[test]
    fn requant_gguf_lm_head_q8_to_q4() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_meta_kv
        // tensor 0: output.weight Q8_0 [32] @ off 0
        write_string(&mut buf, "output.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // tensor 1: after.weight F32 [4] @ off align_up(34)=64
        write_string(&mut buf, "after.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&64u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_off = buf.len();
        // output.weight q8 block: f16 d=0.125 (0x3000) + qs[i]=i  → x[i]=i*0.125
        buf.extend_from_slice(&0x3000u16.to_le_bytes());
        for k in 0..32u8 {
            buf.push(k);
        }
        while buf.len() - data_off < 64 {
            buf.push(0);
        }
        for v in [10.0f32, 20.0, 30.0, 40.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let src = std::env::temp_dir().join("ds4_requant_src.gguf");
        let dst = std::env::temp_dir().join("ds4_requant_dst.gguf");
        std::fs::write(&src, &buf).unwrap();
        requant_gguf_q8_0_to_q4_0(&src, &dst, &|n| n == "output.weight").unwrap();

        let g = GgufFile::open(&dst).unwrap();
        let ow = g.tensors.iter().find(|t| t.name == "output.weight").unwrap();
        assert_eq!(ow.ttype, GgmlType::Q4_0, "output.weight type");
        assert_eq!(g.tensor_byte_size(ow), 18, "q4 block bytes");
        assert_eq!(ow.offset, 0, "lm_head stays at offset 0");
        let aw = g.tensors.iter().find(|t| t.name == "after.weight").unwrap();
        assert_eq!(aw.ttype, GgmlType::F32);
        assert_eq!(aw.offset, 32, "after.weight shifted to align_up(18)=32");

        let mmap = crate::layer_view::MmapSource::open(&dst).unwrap();
        let b = mmap.as_slice();
        let a0 = (g.tensor_data_offset + aw.offset) as usize;
        let vals: Vec<f32> = b[a0..a0 + 16]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![10.0, 20.0, 30.0, 40.0], "trailing tensor intact");

        // q4 dequant approximates the q8 ramp within one q4 step.
        let q4 = &b[g.tensor_data_offset as usize..g.tensor_data_offset as usize + 18];
        let w = crate::layer_view::dequant_q4_0_block(q4);
        for i in 0..32 {
            let exp = i as f32 * 0.125;
            assert!((w[i] - exp).abs() < 0.6, "q4[{i}]={} exp={exp}", w[i]);
        }
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dst).ok();
    }

    #[test]
    fn validate_ds4_layout_accepts_minimal_gguf() {
        // The minimal GGUF has only an F32 token_embd tensor and zero
        // metadata — vocab_size is inferred from the tensor's last dim.
        // We patch in block_count + embedding_length metadata via a
        // hand-rolled writer below.
        let tmp = std::env::temp_dir().join("ds4_engine_validate_layout_min.gguf");
        write_minimal_gguf_with_meta(&tmp);
        let m = validate_ds4_layout(&tmp).expect("validate_ds4_layout on patched minimal");
        assert_eq!(m.n_layers, 1);
        assert_eq!(m.d_model, 4);
        assert_eq!(m.vocab_size, 4);
        assert_eq!(m.total_tensor_bytes, 8); // 4 × f16
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn validate_ds4_layout_rejects_wrong_quant_for_role() {
        // Write a GGUF where moe_gate_experts is F16 (instead of IQ2_XXS).
        let tmp = std::env::temp_dir().join("ds4_engine_validate_layout_bad.gguf");
        write_gguf_with_wrong_moe_quant(&tmp);
        let err = validate_ds4_layout(&tmp).unwrap_err().to_string();
        assert!(err.contains("moe_gate_experts"), "err = {err}");
        assert!(
            err.contains("F16") && err.contains("IQ2_XXS"),
            "err = {err}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    fn write_minimal_gguf_with_meta(path: &Path) {
        // GGUF v3, three metadata keys (architecture + block_count + embedding_length),
        // one F16 tensor [4] called token_embd.weight (DS4-Q2 layout demands F16 for embed).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_meta_kv
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // String
        write_string(&mut buf, "deepseek4");
        write_string(&mut buf, "deepseek4.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&1u32.to_le_bytes());
        write_string(&mut buf, "deepseek4.embedding_length");
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&4u32.to_le_bytes());
        write_string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0 → also serves as vocab_size
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for _ in 0..8 {
            buf.push(0);
        } // 4 × f16 = 8 bytes
        std::fs::write(path, &buf).unwrap();
    }

    fn write_gguf_with_wrong_moe_quant(path: &Path) {
        // Two tensors: token_embd F16 (correct), blk.0.ffn_gate_exps F16 (WRONG — should be IQ2_XXS).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_meta_kv
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut buf, "deepseek4");
        write_string(&mut buf, "deepseek4.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        write_string(&mut buf, "deepseek4.embedding_length");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        // token_embd.weight: F16, [4]
        write_string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&0u64.to_le_bytes());
        // blk.0.ffn_gate_exps.weight: F16 (wrong — expected IQ2_XXS), [4]
        write_string(&mut buf, "blk.0.ffn_gate_exps.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&8u64.to_le_bytes()); // offset (after first tensor's 8 bytes)
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for _ in 0..8 {
            buf.push(0);
        } // 4 f16 = 8 bytes
        for _ in 0..8 {
            buf.push(0);
        }
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn cli_subcommands_run_on_minimal_gguf() {
        // Cover the CLI surface (info / manifest / check) end-to-end on a real
        // on-disk GGUF. These subcommands print to stdout — we only assert
        // they return Ok(()). Catches regressions in cmd_* wiring without
        // requiring a real 81 GB ds4flash.gguf locally.
        let tmp = std::env::temp_dir().join("ds4_engine_cli_test.gguf");
        write_minimal_gguf(&tmp);
        cmd_info(&tmp).expect("cmd_info failed on minimal GGUF");
        cmd_manifest(&tmp).expect("cmd_manifest failed on minimal GGUF");
        cmd_check(&tmp).expect("cmd_check failed on minimal GGUF");
        std::fs::remove_file(&tmp).ok();
    }
}
