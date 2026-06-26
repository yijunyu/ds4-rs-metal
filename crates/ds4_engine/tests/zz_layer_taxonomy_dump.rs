//! Throwaway: classify each DS4 layer by ratio (4=indexer, 128=compressed
//! non-indexer, 1=uncompressed) and whether it is hash-routed. Metadata-only
//! (GgufFile::open demand-faults just the header), so it does NOT page the
//! 86 GB of weights. Run: DS4_GGUF=~/models/ds4flash-q2.gguf cargo test -p
//! ds4_engine --test zz_layer_taxonomy_dump -- --nocapture
use ds4_engine::gguf::GgufFile;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn parse_blk(name: &str) -> Option<(u32, String)> {
    let s = name.strip_prefix("blk.")?;
    let dot = s.find('.')?;
    let idx: u32 = s[..dot].parse().ok()?;
    Some((idx, s[dot + 1..].to_string()))
}

#[test]
fn dump_layer_taxonomy() {
    let Ok(p) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping taxonomy dump");
        return;
    };
    let g = GgufFile::open(&PathBuf::from(&p)).expect("open gguf");
    // per layer: (has_compressor, has_indexer, has_routing_table)
    let mut layers: BTreeMap<u32, (bool, bool, bool)> = BTreeMap::new();
    for t in &g.tensors {
        if let Some((idx, role)) = parse_blk(&t.name) {
            let e = layers.entry(idx).or_insert((false, false, false));
            if role.starts_with("attn_compressor_kv") {
                e.0 = true;
            }
            if role.starts_with("indexer_attn_q_b") {
                e.1 = true;
            }
            // hash routing table — antirez ffn_gate_tid2eid
            if role.contains("tid2eid") || role.contains("routing_table") {
                e.2 = true;
            }
        }
    }
    let (mut r1, mut r4, mut r128, mut hash) = (0u32, 0u32, 0u32, 0u32);
    let mut r128_hash = Vec::new();
    for (idx, (comp, indexer, hashr)) in &layers {
        let ratio = if !comp {
            1
        } else if *indexer {
            4
        } else {
            128
        };
        match ratio {
            1 => r1 += 1,
            4 => r4 += 1,
            128 => r128 += 1,
            _ => {}
        }
        if *hashr {
            hash += 1;
        }
        if ratio == 128 && *hashr {
            r128_hash.push(*idx);
        }
        eprintln!(
            "layer {:2}: ratio={:3} hash={}",
            idx,
            ratio,
            if *hashr { "YES" } else { "no" }
        );
    }
    eprintln!(
        "\nSUMMARY: {} layers | ratio1={} ratio4(indexer)={} ratio128={} | hash-routed={} | ratio128+hash layers={:?}",
        layers.len(),
        r1,
        r4,
        r128,
        hash,
        r128_hash
    );
}
