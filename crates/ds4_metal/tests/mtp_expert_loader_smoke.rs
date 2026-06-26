//! Phase 3 Step 3 — MTP expert-weight loader smoke.
//!
//! Verifies `MetalDispatcher::load_mtp_expert_weights` against a real
//! MTP GGUF. Loads base expert weights first, then MTP, and checks:
//!   - load_mtp_expert_weights returns the expected slot (= n_base_layers).
//!   - state.expert_weight_bufs(mtp_idx) is accessible after the load.
//!   - The returned MTP slot has matching shape/ttype to MtpWeights.
//!
//! Gated by `DS4_GGUF` (base model) + `DS4_MTP_GGUF`. Skips gracefully
//! if either is missing.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_engine::mtp::MtpWeights;
use ds4_metal::MetalDispatcher;

#[test]
fn mtp_expert_loader_smoke() {
    let base_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => { eprintln!("DS4_GGUF unset — skipping"); return; }
    };
    let mtp_path = match std::env::var("DS4_MTP_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("DS4_MTP_GGUF unset — skipping. Download via:\n  \
                 cd benchmarks/ds4_msl/upstream/ds4 && ./download_model.sh mtp");
            return;
        }
    };
    if !base_path.is_file() || !mtp_path.is_file() {
        eprintln!("DS4_GGUF or DS4_MTP_GGUF not a file — skipping");
        return;
    }

    eprintln!("loading base GGUF: {}", base_path.display());
    let manifest = validate_ds4_layout(&base_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&base_path, manifest.n_layers).expect("LayerViews");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let _model = ComposedModelWeights::from_views(&views, &manifest, defaults)
        .expect("ComposedModelWeights");
    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let base_gguf = GgufFile::open(&base_path).expect("GgufFile base");
    disp.load_expert_weights(&base_gguf, views.bytes.as_ref(), manifest.n_layers)
        .expect("load_expert_weights base");
    eprintln!("loaded {} base layers", manifest.n_layers);

    eprintln!("loading MTP GGUF: {}", mtp_path.display());
    let mtp_gguf = GgufFile::open(&mtp_path).expect("GgufFile MTP");
    // Read the MTP file into memory (3.5 GB). We need a &[u8] for the
    // expert-tensor offset arithmetic; production drivers should mmap.
    let mtp_bytes = std::fs::read(&mtp_path).expect("read MTP GGUF bytes");

    let mtp_layer_idx = disp
        .load_mtp_expert_weights(&mtp_gguf, &mtp_bytes)
        .expect("load_mtp_expert_weights");
    eprintln!("loaded MTP at slot {}", mtp_layer_idx);
    assert_eq!(
        mtp_layer_idx, manifest.n_layers,
        "MTP slot should be n_base_layers after fresh load"
    );

    // Validate the slot is queryable via expert_weight_bufs.
    let info = disp
        .expert_weight_bufs(mtp_layer_idx)
        .expect("expert_weight_bufs(mtp_idx)");
    eprintln!(
        "MTP slot: n_experts={} d_in={} d_ffn={}  quants: gate={:?} up={:?} down={:?}",
        info.n_experts, info.d_in, info.d_ffn,
        info.gate_ttype, info.up_ttype, info.down_ttype
    );

    // Cross-check with MtpWeights::from_gguf — same expert tensors via
    // the engine-side loader should report identical shapes/ttypes.
    let mtp = MtpWeights::from_gguf(&mtp_gguf).expect("MtpWeights::from_gguf");
    eprintln!(
        "MtpWeights: n_experts={} d_ffn={}  ffn_gate_exps ttype={:?}",
        mtp.n_experts, mtp.d_ffn, mtp.ffn_gate_exps.info.ttype
    );
    assert_eq!(info.n_experts as u64, mtp.n_experts);
    assert_eq!(info.d_ffn as u64, mtp.d_ffn);
    assert_eq!(info.gate_ttype, mtp.ffn_gate_exps.info.ttype);
    assert_eq!(info.up_ttype,   mtp.ffn_up_exps.info.ttype);
    assert_eq!(info.down_ttype, mtp.ffn_down_exps.info.ttype);
    eprintln!("MTP expert loader smoke OK.");
}
