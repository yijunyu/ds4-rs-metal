//! M4 trace-plumbing smoke test (Linux-friendly).
//!
//! The real M4 gate runs CpuDispatcher × MetalDispatcher on a 96 GB Mac
//! Studio with the actual DS4 GGUF. That hardware is not on the bench yet,
//! so this file exercises the same plumbing — `decode_step_with` driven by
//! two `TracingDispatcher`s feeding `check_traces_close(.., 1e-5)` — using
//! `CpuDispatcher` on both sides. If anything in `TraceEvent` /
//! `check_traces_close` regresses, this test catches it before Mac time is
//! spent on a green-field failure.
//!
//! When `DS4_GGUF=/path/to/model.gguf` is set the test additionally runs
//! `gguf::validate_ds4_layout` on that file, surfacing layout / quant-type
//! regressions on whichever box has a DS4 GGUF locally.

use ds4_engine::decode_step::{decode_step_with, DecodeConfig, LayerWeights, ModelWeights};
use ds4_engine::dispatch::{check_traces_close, CpuDispatcher, TracingDispatcher};

fn identity_matrix(d: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

fn tiny_model(d: usize, n_layers: usize, vocab: usize) -> ModelWeights {
    let n_experts = 4;
    let n_used = 2;
    let exp_w = identity_matrix(d);
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let mut stacked = Vec::with_capacity(n_experts * d * d);
        for _ in 0..n_experts {
            stacked.extend_from_slice(&exp_w);
        }
        layers.push(LayerWeights {
            d_model: d,
            d_ffn: std::num::NonZeroUsize::new(d),
            n_experts,
            n_experts_used: n_used,
            attn_norm_gamma: vec![1.0; d],
            w_attn: vec![0.0; d * d],
            ffn_norm_gamma: vec![1.0; d],
            w_router: vec![0.5; n_experts * d],
            w_router_f16: Vec::new().into(),
            router_bias: vec![0.0; n_experts],
            w_gate_exps: stacked.clone(),
            w_up_exps: stacked.clone(),
            w_down_exps: stacked,
            routing_table: None,
        });
    }
    let mut lm_head = vec![0.0f32; vocab * d];
    for v in 0..vocab {
        lm_head[v * d + (v % d)] = 1.0;
    }
    ModelWeights {
        layers,
        final_norm_gamma: vec![1.0; d],
        lm_head,
        vocab_size: vocab,
        d_model: d,
    }
}

#[test]
fn cpu_x_cpu_trace_agreement_under_tracing_dispatcher() {
    // Two independent dispatchers, two traces, identical inputs. The Mac
    // harness will swap dispatcher B for `MetalDispatcher`; the plumbing
    // around it stays exactly this.
    let d = 8;
    let n_layers = 3;
    let vocab = 16;
    let model = tiny_model(d, n_layers, vocab);
    let cfg = DecodeConfig::default();

    let cpu_a = CpuDispatcher;
    let cpu_b = CpuDispatcher;
    let trace_a = TracingDispatcher::new(&cpu_a);
    let trace_b = TracingDispatcher::new(&cpu_b);

    let x: Vec<f32> = (0..d).map(|i| (i as f32) * 0.125 - 0.5).collect();

    let logits_a = decode_step_with(&trace_a, x.clone(), &model, &cfg).expect("decode_step A ok");
    let logits_b = decode_step_with(&trace_b, x, &model, &cfg).expect("decode_step B ok");

    assert_eq!(logits_a.len(), vocab);
    assert_eq!(logits_b.len(), vocab);
    assert_eq!(logits_a, logits_b, "CPU vs CPU logits must be bit-equal");

    let events_a = trace_a.events();
    let events_b = trace_b.events();

    // decode_step = N layers × (rms, matvec, rms, matvec_router, softplus,
    //                            router_finalize, moe_routed_step) + final rms + lm_head matvec
    //             = N * 7 + 2 events.
    let expected_events = n_layers * 7 + 2;
    assert_eq!(
        events_a.len(),
        expected_events,
        "trace shape regression: expected {expected_events} events, got {}",
        events_a.len()
    );
    assert_eq!(events_a.len(), events_b.len());

    check_traces_close(&events_a, &events_b, 1e-5).expect("Cpu×Cpu traces must close within 1e-5");
}

#[test]
fn trace_mismatch_is_caught_by_check_traces_close() {
    // Sanity: hand-perturb one event's output and verify check_traces_close
    // refuses to accept it at 1e-5 tolerance.
    use ds4_engine::dispatch::TraceEvent;

    let a = vec![TraceEvent::RmsNorm {
        x: vec![1.0, 2.0],
        gamma_len: 2,
        eps: 1e-6,
        output: vec![1.0, 2.0],
    }];
    let b = vec![TraceEvent::RmsNorm {
        x: vec![1.0, 2.0],
        gamma_len: 2,
        eps: 1e-6,
        output: vec![1.0, 2.0 + 1e-3],
    }];
    let err = check_traces_close(&a, &b, 1e-5).expect_err("1e-3 perturbation must exceed 1e-5 tol");
    assert!(
        err.contains("rms_norm"),
        "error must name the failing event: {err}"
    );
}

/// Opportunistic real-GGUF layout check. Skips cleanly when `DS4_GGUF` is
/// not set. Catches DS4-Q2 layout drift before the M4 binary boots.
#[test]
fn validate_ds4_layout_on_real_gguf_when_present() {
    let Ok(path) = std::env::var("DS4_GGUF") else {
        eprintln!("DS4_GGUF unset — skipping real-GGUF validate");
        return;
    };
    let p = std::path::PathBuf::from(&path);
    if !p.exists() {
        eprintln!("DS4_GGUF={path} does not exist — skipping");
        return;
    }
    let manifest = ds4_engine::gguf::validate_ds4_layout(&p)
        .unwrap_or_else(|e| panic!("validate_ds4_layout on {path} failed: {e:#}"));
    assert!(manifest.n_layers > 0, "n_layers must be positive");
    assert!(manifest.d_model > 0, "d_model must be positive");
    assert!(
        manifest.total_tensor_bytes <= ds4_engine::gguf::DS4_TENSOR_BYTES_LIMIT,
        "tensor bytes {} > limit {}",
        manifest.total_tensor_bytes,
        ds4_engine::gguf::DS4_TENSOR_BYTES_LIMIT
    );
    eprintln!(
        "OK DS4 layout: n_layers={} d_model={} vocab={} bytes={:.2} GB",
        manifest.n_layers,
        manifest.d_model,
        manifest.vocab_size,
        manifest.total_tensor_bytes as f64 / 1e9
    );
}
