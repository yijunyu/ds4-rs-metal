//! Quick diagnostic: dump per-layer expert quant types for the loaded GGUF.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use ds4_engine::attn_dispatch::DefaultsDs4;
use ds4_engine::decode_step::ComposedModelWeights;
use ds4_engine::gguf::{validate_ds4_layout, GgufFile};
use ds4_engine::layer_view::LayerViews;
use ds4_metal::MetalDispatcher;

#[test]
fn dump_layer_quants() {
    if std::env::var("DS4_DUMP_QUANTS").ok().as_deref() != Some("1") {
        eprintln!("DS4_DUMP_QUANTS unset — skipping");
        return;
    }
    let gguf_path = match std::env::var("DS4_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => return,
    };
    if !gguf_path.is_file() { return; }

    let manifest = validate_ds4_layout(&gguf_path).expect("validate_ds4_layout");
    let views = LayerViews::open(&gguf_path, manifest.n_layers).expect("LayerViews::open");
    let defaults = DefaultsDs4::ds4_v4_flash();
    let _model = ComposedModelWeights::from_views(&views, &manifest, defaults).expect("model");
    let mut disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    let gguf = GgufFile::open(&gguf_path).expect("GgufFile::open");
    disp.load_expert_weights(&gguf, views.bytes.as_ref(), manifest.n_layers).expect("load");

    if let Ok(i0) = disp.expert_weight_bufs(0) {
        let n = (manifest.n_experts as u64).max(1);
        eprintln!(
            "MOEDIMS d_ffn={} n_experts={} gate_total={} up_total={} down_total={} | per_expert gate={} up={} down={} sum={}",
            i0.d_ffn, manifest.n_experts,
            i0.gate.length(), i0.up.length(), i0.down.length(),
            i0.gate.length()/n, i0.up.length()/n, i0.down.length()/n,
            (i0.gate.length()+i0.up.length()+i0.down.length())/n,
        );
    }
    eprintln!("\nlayer  gate         up           down");
    for layer in 0..manifest.n_layers {
        if let Ok(info) = disp.expert_weight_bufs(layer) {
            eprintln!(
                "{:>4}   {:<12?} {:<12?} {:<12?}",
                layer, info.gate_ttype, info.up_ttype, info.down_ttype
            );
        }
    }
}
