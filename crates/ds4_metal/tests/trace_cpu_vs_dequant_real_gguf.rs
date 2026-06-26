//! Sparse real-GGUF analogue of `trace_cpu_vs_dequant.rs`.
//!
//! Opt-in: set `DS4_GGUF=/path/to/model.gguf` to run. Without it the test
//! prints a one-line skip and passes — keeps CI green on machines without
//! a DS4 GGUF on disk.
//!
//! What it proves beyond the synthetic harness:
//!   - GGUF offset math (`tensor_data_offset + ti.offset` + expert stride)
//!     yields selected expert bytes that round-trip through the dequant oracle
//!     to the same f32 slabs an independent direct computation consumes.
//!   - The 3-D tensor classifier (gate/up/down naming convention
//!     `blk.{layer}.ffn_{gate,up,down}_exps.weight`) matches a real DS4
//!     GGUF layer 0.
//!
//! The test intentionally reads only a few selected experts. A full DS4 GGUF
//! is too large to slurp into memory, and full f32 expert slabs are larger
//! still.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use ds4_engine::gguf::{GgmlType, GgufFile, TensorInfo};
use ds4_metal::quantized_experts::{
    moe_routed_step_cpu_via_dequant_with_activation, ActivationQuant, QuantTensor,
    QuantizedExpertWeights,
};

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
    Ok((ti.dims[2], ti.dims[1], ti.dims[0]))
}

fn read_exact_at(file: &mut File, offset: u64, nbytes: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; nbytes];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn sparse_quant_tensor(
    ttype: GgmlType,
    dims: [u64; 3],
    bytes: Vec<u8>,
    expert_stride: u64,
) -> QuantTensor {
    #[cfg(target_os = "macos")]
    {
        let device =
            metal::Device::system_default().expect("Metal device for sparse real-GGUF test");
        let metal_buf = device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared);
        QuantTensor {
            ttype,
            dims,
            bytes,
            expert_stride,
            mmap_ptr: std::ptr::null(),
            mmap_len: 0,
            metal_buf,
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        QuantTensor {
            ttype,
            dims,
            bytes,
            expert_stride,
        }
    }
}

fn sparse_quant_tensor_from_gguf(
    gguf: &GgufFile,
    file: &mut File,
    ti: &TensorInfo,
    dims_aer: [u64; 3],
    selected: &[u32],
) -> Result<QuantTensor> {
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
    let tensor_start = gguf.tensor_data_offset + ti.offset;
    let tensor_nbytes = gguf.tensor_byte_size(ti);
    let mut bytes = Vec::with_capacity(selected.len() * expert_stride as usize);
    for &expert in selected {
        let expert_start = tensor_start + expert as u64 * expert_stride;
        let expert_end = expert_start
            .checked_add(expert_stride)
            .ok_or_else(|| anyhow!("expert byte range overflow"))?;
        let tensor_end = tensor_start
            .checked_add(tensor_nbytes)
            .ok_or_else(|| anyhow!("tensor byte range overflow"))?;
        if expert_end > tensor_end {
            bail!(
                "expert {} of tensor {} extends past tensor range",
                expert,
                ti.name
            );
        }
        bytes.extend_from_slice(&read_exact_at(file, expert_start, expert_stride as usize)?);
    }
    Ok(sparse_quant_tensor(
        ti.ttype,
        [selected.len() as u64, dims_aer[1], dims_aer[2]],
        bytes,
        expert_stride,
    ))
}

fn sparse_qew_from_gguf(
    gguf: &GgufFile,
    file: &mut File,
    layer_idx: u32,
    selected: &[u32],
) -> Result<QuantizedExpertWeights> {
    let gate_name = format!("blk.{layer_idx}.ffn_gate_exps.weight");
    let up_name = format!("blk.{layer_idx}.ffn_up_exps.weight");
    let down_name = format!("blk.{layer_idx}.ffn_down_exps.weight");
    let gate_ti = find_tensor(gguf, &gate_name)?;
    let up_ti = find_tensor(gguf, &up_name)?;
    let down_ti = find_tensor(gguf, &down_name)?;

    let (n_experts, gate_rows, gate_cols) = three_d(gate_ti)?;
    let (_, up_rows, up_cols) = three_d(up_ti)?;
    let (_, down_rows, down_cols) = three_d(down_ti)?;
    for &expert in selected {
        if expert as u64 >= n_experts {
            bail!("selected expert {expert} >= n_experts={n_experts}");
        }
    }
    if (gate_rows, gate_cols) != (up_rows, up_cols) {
        bail!("gate/up shape mismatch: gate={gate_rows}x{gate_cols} up={up_rows}x{up_cols}");
    }
    let d_ffn = gate_rows as u32;
    let d_in = gate_cols as u32;
    if down_rows != d_in as u64 || down_cols != d_ffn as u64 {
        bail!("down shape mismatch: expected {d_in}x{d_ffn}, got {down_rows}x{down_cols}");
    }

    Ok(QuantizedExpertWeights {
        layer_idx,
        n_experts: selected.len() as u32,
        d_in,
        d_ffn,
        gate: sparse_quant_tensor_from_gguf(
            gguf,
            file,
            gate_ti,
            [n_experts, gate_rows, gate_cols],
            selected,
        )
        .with_context(|| format!("loading sparse {gate_name}"))?,
        up: sparse_quant_tensor_from_gguf(
            gguf,
            file,
            up_ti,
            [n_experts, up_rows, up_cols],
            selected,
        )
        .with_context(|| format!("loading sparse {up_name}"))?,
        down: sparse_quant_tensor_from_gguf(
            gguf,
            file,
            down_ti,
            [n_experts, down_rows, down_cols],
            selected,
        )
        .with_context(|| format!("loading sparse {down_name}"))?,
    })
}

fn assert_tensor_type_and_dims(
    gguf: &GgufFile,
    name: &str,
    ttype: GgmlType,
    dims: &[u64],
) -> Result<()> {
    let ti = find_tensor(gguf, name)?;
    if ti.ttype != ttype || ti.dims != dims {
        bail!(
            "tensor {name}: expected type={ttype:?} dims={dims:?}, got type={:?} dims={:?}",
            ti.ttype,
            ti.dims
        );
    }
    Ok(())
}

fn open_real_gguf_or_skip(test_name: &str) -> Option<(PathBuf, GgufFile, File)> {
    let path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "DS4_GGUF unset — skipping {test_name}. Set DS4_GGUF=/path/to/ds4.gguf to run."
            );
            return None;
        }
    };
    if !path.is_file() {
        eprintln!(
            "DS4_GGUF={} is not a regular file — skipping {test_name}.",
            path.display()
        );
        return None;
    }

    let gguf = GgufFile::open(&path).expect("parse GGUF header");
    let file = File::open(&path).expect("open GGUF for sparse expert byte reads");
    Some((path, gguf, file))
}

fn deterministic_activation(d: usize, layer_idx: u32) -> Vec<f32> {
    (0..d)
        .map(|i| {
            let x = i as f32;
            ((x * 0.013 + layer_idx as f32 * 0.071).sin() * 1.7) + ((x * 0.007 - 0.31).cos() * 0.3)
        })
        .collect()
}

fn run_sparse_math_reference_case(
    gguf: &GgufFile,
    file: &mut File,
    layer_idx: u32,
    selected_real: &[u32],
    route_weights: &[f32],
) {
    assert_eq!(
        selected_real.len(),
        route_weights.len(),
        "selected experts and route weights must align"
    );
    let qew = sparse_qew_from_gguf(gguf, file, layer_idx, selected_real)
        .unwrap_or_else(|e| panic!("load sparse layer {layer_idx} quantized expert weights: {e}"));
    let x = deterministic_activation(qew.d_in as usize, layer_idx);
    let selected: Vec<usize> = (0..selected_real.len()).collect();

    let oracle = moe_routed_step_cpu_via_dequant_with_activation(
        &qew,
        &x,
        &selected,
        route_weights,
        qew.d_ffn as usize,
        ActivationQuant::F32,
    )
    .expect("sparse CPU-via-dequant oracle");

    let gate_f32 = qew.dequant_gate_f32().expect("sparse dequant gate");
    let up_f32 = qew.dequant_up_f32().expect("sparse dequant up");
    let down_f32 = qew.dequant_down_f32().expect("sparse dequant down");
    let direct = ds4_engine::moe::moe_routed_step(
        &x,
        &selected,
        route_weights,
        &gate_f32,
        &up_f32,
        &down_f32,
        qew.d_ffn as usize,
    );

    assert_eq!(oracle.len(), direct.len());
    for (i, (a, b)) in oracle.iter().zip(direct.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sparse real-GGUF output drift at layer {layer_idx} idx {i}: oracle={a} direct={b}"
        );
    }

    eprintln!(
        "sparse real-GGUF dequant PASS: layer {layer_idx}, real_experts={selected_real:?}, d_in={}, d_ffn={}",
        qew.d_in, qew.d_ffn
    );
}

fn run_sparse_antirez_activation_case(
    gguf: &GgufFile,
    file: &mut File,
    layer_idx: u32,
    selected_real: &[u32],
    route_weights: &[f32],
    min_diff: f32,
) {
    assert_eq!(
        selected_real.len(),
        route_weights.len(),
        "selected experts and route weights must align"
    );
    let qew = sparse_qew_from_gguf(gguf, file, layer_idx, selected_real)
        .unwrap_or_else(|e| panic!("load sparse layer {layer_idx} quantized expert weights: {e}"));
    let x = deterministic_activation(qew.d_in as usize, layer_idx);
    let selected: Vec<usize> = (0..selected_real.len()).collect();

    let f32_out = moe_routed_step_cpu_via_dequant_with_activation(
        &qew,
        &x,
        &selected,
        route_weights,
        qew.d_ffn as usize,
        ActivationQuant::F32,
    )
    .expect("sparse f32 CPU-via-dequant oracle");

    let antirez_out = moe_routed_step_cpu_via_dequant_with_activation(
        &qew,
        &x,
        &selected,
        route_weights,
        qew.d_ffn as usize,
        ActivationQuant::AntirezQ8K,
    )
    .expect("sparse antirez-q8k CPU-via-dequant oracle");

    let max_abs_diff = f32_out
        .iter()
        .zip(antirez_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff > min_diff,
        "real-GGUF antirez Q8_K activation path unexpectedly matched f32 at layer {layer_idx}; max_abs_diff={max_abs_diff:e}, min_diff={min_diff:e}"
    );
    eprintln!(
        "sparse real-GGUF antirez parity PASS: layer {layer_idx}, real_experts={selected_real:?}, max_abs_diff={max_abs_diff:e}"
    );
}

#[test]
fn sparse_math_reference_cpu_vs_dequant_on_real_gguf_layer0() {
    let Some((_path, gguf, mut file)) = open_real_gguf_or_skip("sparse real-GGUF dequant check")
    else {
        return;
    };
    let selected_real = [0u32, 1u32];
    let route_weights = vec![0.25f32, 0.75];
    run_sparse_math_reference_case(&gguf, &mut file, 0, &selected_real, &route_weights);
}

#[test]
fn real_gguf_layer0_covers_released_kernel_quant_families() {
    let path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "DS4_GGUF unset — skipping real-GGUF kernel-family layout check. \
                 Set DS4_GGUF=/path/to/ds4.gguf to run."
            );
            return;
        }
    };
    if !path.is_file() {
        eprintln!(
            "DS4_GGUF={} is not a regular file — skipping.",
            path.display()
        );
        return;
    }

    let gguf = GgufFile::open(&path).expect("parse GGUF header");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.ffn_gate_exps.weight",
        GgmlType::IQ2_XXS,
        &[4096, 2048, 256],
    )
    .expect("layer0 gate experts must exercise IQ2_XXS kernels");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.ffn_up_exps.weight",
        GgmlType::IQ2_XXS,
        &[4096, 2048, 256],
    )
    .expect("layer0 up experts must exercise IQ2_XXS kernels");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.ffn_down_exps.weight",
        GgmlType::Q2_K,
        &[2048, 4096, 256],
    )
    .expect("layer0 down experts must exercise Q2_K kernels");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.attn_q_a.weight",
        GgmlType::Q8_0,
        &[4096, 1024],
    )
    .expect("layer0 q_a must exercise Q8_0 matvec kernels");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.ffn_gate_shexp.weight",
        GgmlType::Q8_0,
        &[4096, 2048],
    )
    .expect("layer0 shared gate must exercise Q8_0 shared-expert kernels");
    assert_tensor_type_and_dims(
        &gguf,
        "blk.0.hc_attn_fn.weight",
        GgmlType::F16,
        &[16384, 24],
    )
    .expect("layer0 HC attention function must exercise F16 matvec kernels");
    assert_tensor_type_and_dims(&gguf, "blk.0.attn_norm.weight", GgmlType::F32, &[4096])
        .expect("layer0 norms must exercise F32 RMS kernels");
    eprintln!("real-GGUF kernel-family layout PASS: layer 0 covers IQ2_XXS, Q2_K, Q8_0, F16, F32");
}

#[test]
fn sparse_antirez_q8k_activation_differs_from_f32_on_real_gguf_layer0() {
    let Some((_path, gguf, mut file)) =
        open_real_gguf_or_skip("sparse real-GGUF antirez parity check")
    else {
        return;
    };
    let selected_real = [0u32, 1u32];
    let route_weights = vec![0.25f32, 0.75];
    run_sparse_antirez_activation_case(&gguf, &mut file, 0, &selected_real, &route_weights, 1.0e-4);
}

#[test]
fn sparse_math_reference_cpu_vs_dequant_on_real_gguf_multiple_layers() {
    let Some((_path, gguf, mut file)) =
        open_real_gguf_or_skip("multi-layer sparse real-GGUF dequant check")
    else {
        return;
    };
    let cases: &[(u32, &[u32], &[f32])] = &[
        (0, &[0, 1], &[0.25, 0.75]),
        (7, &[3, 17, 255], &[0.2, 0.3, 0.5]),
        (21, &[0, 64, 128, 255], &[0.1, 0.2, 0.3, 0.4]),
        (42, &[5, 127, 200], &[0.15, 0.35, 0.5]),
    ];

    for &(layer_idx, selected_real, route_weights) in cases {
        run_sparse_math_reference_case(&gguf, &mut file, layer_idx, selected_real, route_weights);
    }
}

#[test]
fn sparse_antirez_q8k_activation_differs_from_f32_on_real_gguf_multiple_layers() {
    let Some((_path, gguf, mut file)) =
        open_real_gguf_or_skip("multi-layer sparse real-GGUF antirez parity check")
    else {
        return;
    };
    let cases: &[(u32, &[u32], &[f32])] = &[
        (0, &[0, 1], &[0.25, 0.75]),
        (7, &[3, 17, 255], &[0.2, 0.3, 0.5]),
        (21, &[0, 64, 128, 255], &[0.1, 0.2, 0.3, 0.4]),
        (42, &[5, 127, 200], &[0.15, 0.35, 0.5]),
    ];

    for &(layer_idx, selected_real, route_weights) in cases {
        run_sparse_antirez_activation_case(
            &gguf,
            &mut file,
            layer_idx,
            selected_real,
            route_weights,
            1.0e-4,
        );
    }
}

#[test]
#[cfg(target_os = "macos")]
fn metal_moe_fast_path_matches_sparse_dequant_on_real_gguf_layer0() {
    run_metal_moe_fast_path_cases(&[(
        0,
        &[0, 1, 2, 3, 4, 5],
        &[0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
    )]);
}

#[test]
#[cfg(target_os = "macos")]
fn metal_moe_fast_path_matches_sparse_dequant_on_real_gguf_multiple_layers() {
    run_metal_moe_fast_path_cases(&[
        (
            0,
            &[0, 1, 2, 3, 4, 5],
            &[0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        ),
        (
            7,
            &[3, 17, 64, 127, 200, 255],
            &[0.04, 0.08, 0.12, 0.16, 0.24, 0.36],
        ),
        (
            21,
            &[0, 1, 64, 65, 128, 255],
            &[0.03, 0.07, 0.11, 0.19, 0.23, 0.37],
        ),
        (
            42,
            &[5, 31, 127, 128, 200, 255],
            &[0.06, 0.09, 0.13, 0.17, 0.22, 0.33],
        ),
    ]);
}

#[cfg(target_os = "macos")]
fn run_metal_moe_fast_path_cases(cases: &[(u32, &[u32], &[f32])]) {
    use ds4_engine::dispatch::KernelDispatcher;
    use ds4_engine::gguf::validate_ds4_layout;
    use ds4_engine::layer_view::LayerViews;
    use ds4_metal::MetalDispatcher;

    let Some((path, gguf, mut file)) =
        open_real_gguf_or_skip("real-GGUF Metal MoE fast-path check")
    else {
        return;
    };
    let manifest = validate_ds4_layout(&path).expect("validate DS4 layout");
    let views = LayerViews::open(&path, manifest.n_layers).expect("mmap GGUF views");
    let mut metal = MetalDispatcher::new().expect("Metal dispatcher");
    let max_layer = cases
        .iter()
        .map(|(layer_idx, _, _)| *layer_idx)
        .max()
        .unwrap_or(0);
    metal
        .load_expert_weights(&gguf, views.bytes.as_ref(), max_layer + 1)
        .expect("load sampled expert weights into Metal dispatcher");

    for &(layer_idx, selected_real, route_weights) in cases {
        assert_eq!(
            selected_real.len(),
            6,
            "Metal MoE path requires 6 active experts"
        );
        assert_eq!(
            selected_real.len(),
            route_weights.len(),
            "selected experts and route weights must align"
        );
        let qew =
            sparse_qew_from_gguf(&gguf, &mut file, layer_idx, selected_real).unwrap_or_else(|e| {
                panic!("load sparse layer {layer_idx} quantized expert weights: {e}")
            });
        let x = deterministic_activation(qew.d_in as usize, layer_idx);
        let sparse_selected: Vec<usize> = (0..selected_real.len()).collect();
        let oracle = moe_routed_step_cpu_via_dequant_with_activation(
            &qew,
            &x,
            &sparse_selected,
            route_weights,
            qew.d_ffn as usize,
            ActivationQuant::F32,
        )
        .expect("sparse CPU-via-dequant oracle");

        let selected_for_metal: Vec<usize> = selected_real.iter().map(|&x| x as usize).collect();
        let got = metal.moe_routed_step(
            layer_idx,
            &x,
            &selected_for_metal,
            route_weights,
            &[],
            &[],
            &[],
            0,
        );

        assert_eq!(got.len(), oracle.len());
        let max_abs_diff = got
            .iter()
            .zip(oracle.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= 1.0e-2,
            "real-GGUF Metal MoE fast path drifted from sparse dequant oracle at layer {layer_idx}: max_abs_diff={max_abs_diff:e}"
        );
        eprintln!(
            "real-GGUF Metal MoE fast path PASS: layer {layer_idx}, real_experts={selected_real:?}, max_abs_diff={max_abs_diff:e}"
        );
    }
}
