//! `ds4-server` — OpenAI/Anthropic-compatible HTTP server for DS4-Flash that
//! drives our pure-Rust Metal engine (`ds4_metal::DecodeRunner`). Drop-in for
//! the antirez C `ds4-server` (`:8000 --cors --warm-weights`). Hand-rolled
//! `std::net` HTTP/1.1 + SSE — single-threaded serial (the GPU session is
//! single-threaded; `MetalDispatcher` is `!Send`).

mod anthropic;
mod backend;
mod gen;
mod http;
mod openai;
mod responses;
mod tools_dsml;

use std::net::{TcpListener, TcpStream};
use std::path::Path;

use ds4_engine::gguf::GgufFile;
use ds4_engine::tokenizer::{SpecialTokens, Vocab};
use backend::decode_runner::DecodeRunner;
use serde_json::{json, Value};

use gen::{generate, GenEvent};

const MODEL_ID: &str = "deepseek-v4-flash";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8000;
    let mut ctx: u32 = 32768;
    // raw_cap = the raw-attention (SWA) window, decoupled from `ctx`. None until
    // resolved below (flag → env → default), then clamped to ctx.
    let mut raw_cap_arg: Option<u32> = None;
    let mut warm = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--host" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    host = v.clone();
                }
            }
            "--port" => {
                i += 1;
                if let Some(p) = args.get(i).and_then(|s| s.parse().ok()) {
                    port = p;
                }
            }
            "-c" | "--ctx" => {
                i += 1;
                if let Some(c) = args.get(i).and_then(|s| s.parse().ok()) {
                    ctx = c;
                }
            }
            "--raw-cap" | "--raw_cap" => {
                i += 1;
                if let Some(r) = args.get(i).and_then(|s| s.parse().ok()) {
                    raw_cap_arg = Some(r);
                }
            }
            "--warm-weights" => warm = true,
            "--cors" => { /* CORS is always sent (see http.rs) */ }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4-server -m MODEL [--host H] [--port P] [-c CTX] \
                     [--raw-cap N] [--cors] [--warm-weights]"
                );
                return;
            }
            other => {
                // Tolerate the daemon's extra flags so we are arg-compatible.
                eprintln!("ds4-server: ignoring unrecognized arg `{other}`");
                if other.starts_with('-') && args.get(i + 1).map(|s| !s.starts_with('-')).unwrap_or(false) {
                    i += 1;
                }
            }
        }
        i += 1;
    }

    let model = model.unwrap_or_else(|| {
        eprintln!("ds4-server: error — -m/--model <path-to-gguf> is required");
        std::process::exit(2);
    });
    // SSD-streaming pread fast path (quantized_experts::pread_span) re-opens
    // the GGUF by path — cold mmap faults copy at ~1.3 GB/s vs ~4 GB/s pread.
    std::env::set_var("DS4_GGUF_PATH", &model);

    // raw_cap = the raw-attention (SWA) SLIDING window, NOT the full context.
    // DS4-Flash's architectural window is DS4_N_SWA=128 (antirez's own default,
    // ds4.c:103): the model attends the most-recent 128 raw rows + the
    // compressor/indexer for long range. Attending MORE raw rows (the old 8192
    // "full-raw" default) is OUT-OF-DISTRIBUTION and collapses to garbage past
    // ~2k tokens (measured: full-raw @3000 emits a word-salad; SWA=128 stays
    // coherent and recalls long-range facts via the compressor). The SWA circular
    // ring (single_buffer_encoder: slot = pos % raw_cap) makes a small raw_cap a
    // true sliding window — eviction matches antirez — and the compressed/indexer
    // rings are sized by ctx/ratio independently (comp_ring_rows), so a small
    // raw_cap no longer thrashes. Resolve from `--raw-cap`, else $DS4_RAW_CAP,
    // else DS4_N_SWA — then clamp to ctx.
    const RAW_CAP_DEFAULT: u32 = 128; // = DS4_N_SWA
    let raw_cap = raw_cap_arg
        .or_else(|| std::env::var("DS4_RAW_CAP").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(RAW_CAP_DEFAULT)
        .min(ctx)
        .max(1);
    eprintln!("ds4-server: loading {model} (ctx={ctx}, raw_cap={raw_cap}, warm={warm}) …");
    let mut runner = DecodeRunner::open(Path::new(&model), raw_cap)
        .unwrap_or_else(|e| panic!("ds4-server: model load failed: {e}"));
    let gguf = GgufFile::open(Path::new(&model))
        .unwrap_or_else(|e| panic!("ds4-server: gguf metadata: {e}"));
    let vocab = Vocab::from_gguf(&gguf).unwrap_or_else(|e| panic!("ds4-server: tokenizer: {e}"));
    let st = vocab.special_tokens().unwrap_or_else(|e| panic!("ds4-server: special tokens: {e}"));
    if warm {
        let g = runner.dispatcher.warm_up_expert_pages();
        eprintln!("ds4-server: warmed {:.2} GB of expert pages", g as f64 / 1e9);
    }
    // Phase 1 no-copy refactor: the encoder/DecodeSession path (what we serve)
    // reads shared-expert + lm_head weights via their q8 buffers, so the f32
    // duplicates are dead — free them (~6.5 GB of idle-compressed anonymous RAM).
    // Safe here because the server never uses the trait/run_argmax f32 path.
    // DS4_LEAN_WEIGHTS=0 escape hatch keeps the f32 (some staged/chain encoder
    // branches still read them on long prompts) — don't free in that mode.
    let lean = std::env::var("DS4_LEAN_WEIGHTS").map(|v| v != "0").unwrap_or(true);
    if lean {
        let freed = runner.free_dead_f32_weights();
        if freed > 0 {
            eprintln!("ds4-server: freed {:.2} GB of dead f32 weight duplicates", freed as f64 / 1e9);
        }
    }
    eprintln!("ds4-server: model ready (vocab={}, eos={})", vocab.n_vocab(), st.eos);

    let listener = TcpListener::bind((host.as_str(), port))
        .unwrap_or_else(|e| panic!("ds4-server: bind {host}:{port}: {e}"));
    eprintln!("ds4-server: listening on http://{host}:{port}  (model id: {MODEL_ID})");

    // Concurrency: the Metal runner is single-threaded (not Send), so it stays
    // on THIS thread, which becomes the inference WORKER. A detached acceptor
    // thread accepts connections and spawns a parser thread per connection; each
    // parser thread answers NON-inference requests (OPTIONS, /v1/models) directly
    // — so a long in-flight decode can never head-of-line-block a health check —
    // and forwards inference (POST) to this worker over a channel. Inference
    // still serializes (one runner / one KV cache), but the accept loop and
    // health checks stay responsive (an orchestrator no longer drops the backend
    // mid-decode). A small fix; true parallel decode would need an N-session pool.
    let (tx, rx) = std::sync::mpsc::channel::<(http::Request, TcpStream)>();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let tx = tx.clone();
                    std::thread::spawn(move || parse_conn(stream, tx));
                }
                Err(e) => eprintln!("ds4-server: accept error: {e}"),
            }
        }
    });
    // Worker loop: process inference requests serially on the Metal thread.
    while let Ok((req, mut stream)) = rx.recv() {
        if let Err(e) = serve_inference(&mut stream, &req, &runner, &vocab, &st, ctx) {
            eprintln!("ds4-server: connection error: {e}");
        }
    }
}

fn err_body(msg: &str) -> String {
    json!({ "error": { "message": msg } }).to_string()
}

/// Per-connection parser thread: read the request and answer non-inference
/// requests (CORS preflight, model discovery) DIRECTLY — these need no runner,
/// so they respond instantly even while the worker is mid-decode. Inference
/// (POST) is forwarded to the worker over `tx` (the worker writes the response).
fn parse_conn(mut stream: TcpStream, tx: std::sync::mpsc::Sender<(http::Request, TcpStream)>) {
    let req = match http::read_request(&stream) {
        Ok(Some(r)) => r,
        _ => return,
    };
    let handled = match (req.method.as_str(), req.path.as_str()) {
        ("OPTIONS", _) => Some(http::write_no_content(&mut stream)),
        ("GET", "/v1/models") => {
            Some(http::write_json(&mut stream, 200, &openai::models_json(MODEL_ID)))
        }
        ("GET", p) if p.starts_with("/v1/models/") => {
            Some(http::write_json(&mut stream, 200, &openai::model_json(MODEL_ID)))
        }
        _ => None, // inference → worker
    };
    match handled {
        Some(Ok(())) => {}
        Some(Err(e)) => eprintln!("ds4-server: connection error: {e}"),
        None => {
            let _ = tx.send((req, stream));
        }
    }
}

/// Worker-side inference dispatch (runs on the single Metal thread, serialized).
fn serve_inference(
    stream: &mut TcpStream,
    req: &http::Request,
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    ctx: u32,
) -> std::io::Result<()> {
    match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/v1/chat/completions") => handle_chat(stream, &req.body, runner, vocab, st, ctx),
        ("POST", "/v1/messages") => handle_messages(stream, &req.body, runner, vocab, st, ctx),
        ("POST", "/v1/responses") => handle_responses(stream, &req.body, runner, vocab, st, ctx),
        ("POST", "/v1/completions") => handle_completions(stream, &req.body, runner, vocab, st, ctx),
        _ => http::write_json(stream, 404, &err_body("unknown endpoint")),
    }
}

fn handle_chat(
    stream: &mut TcpStream,
    body: &[u8],
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    ctx: u32,
) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&format!("invalid JSON: {e}"))),
    };
    let (genreq, stream_mode) = match openai::parse_chat_request(&req, ctx) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&e)),
    };
    let id = openai::gen_id();
    let created = openai::now_secs();

    if stream_mode {
        http::write_sse_headers(stream)?;
        http::write_sse_data(
            stream,
            &openai::chunk_json(&id, created, MODEL_ID, json!({ "role": "assistant" }), None),
        )?;
        let mut emit = |ev: GenEvent| -> std::io::Result<()> {
            match ev {
                GenEvent::Reasoning(t) => http::write_sse_data(
                    stream,
                    &openai::chunk_json(&id, created, MODEL_ID, json!({ "reasoning_content": t }), None),
                ),
                GenEvent::Content(t) => http::write_sse_data(
                    stream,
                    &openai::chunk_json(&id, created, MODEL_ID, json!({ "content": t }), None),
                ),
                GenEvent::ToolCalls(calls) => http::write_sse_data(
                    stream,
                    &openai::chunk_json(&id, created, MODEL_ID, openai::tool_calls_delta(&calls), None),
                ),
                GenEvent::Done { finish_reason, .. } => {
                    http::write_sse_data(
                        stream,
                        &openai::chunk_json(&id, created, MODEL_ID, json!({}), Some(&finish_reason)),
                    )?;
                    http::write_sse_data(stream, "[DONE]")
                }
                GenEvent::Error(e) => {
                    http::write_sse_data(stream, &err_body(&e))
                }
            }
        };
        generate(runner, vocab, st, &genreq, &mut emit)
    } else {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<tools_dsml::ToolCall> = Vec::new();
        let mut finish = String::from("stop");
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut errored: Option<String> = None;
        {
            let mut emit = |ev: GenEvent| -> std::io::Result<()> {
                match ev {
                    GenEvent::Reasoning(t) => reasoning.push_str(&t),
                    GenEvent::Content(t) => content.push_str(&t),
                    GenEvent::ToolCalls(c) => tool_calls = c,
                    GenEvent::Done { finish_reason, prompt_tokens: p, completion_tokens: c } => {
                        finish = finish_reason;
                        prompt_tokens = p;
                        completion_tokens = c;
                    }
                    GenEvent::Error(e) => errored = Some(e),
                }
                Ok(())
            };
            generate(runner, vocab, st, &genreq, &mut emit)?;
        }
        if let Some(e) = errored {
            return http::write_json(stream, 500, &err_body(&e));
        }
        http::write_json(
            stream,
            200,
            &openai::final_completion_json(
                &id, created, MODEL_ID, &content, &reasoning, &tool_calls, &finish, prompt_tokens, completion_tokens,
            ),
        )
    }
}

/// Accumulated generation result (for non-streaming responses).
#[derive(Default)]
struct Collected {
    content: String,
    reasoning: String,
    tool_calls: Vec<tools_dsml::ToolCall>,
    finish: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    error: Option<String>,
}

/// Run a generation to completion, accumulating all events.
fn collect(
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    genreq: &gen::GenReq,
) -> std::io::Result<Collected> {
    let mut c = Collected { finish: "stop".to_string(), ..Default::default() };
    {
        let mut emit = |ev: GenEvent| -> std::io::Result<()> {
            match ev {
                GenEvent::Reasoning(t) => c.reasoning.push_str(&t),
                GenEvent::Content(t) => c.content.push_str(&t),
                GenEvent::ToolCalls(tc) => c.tool_calls = tc,
                GenEvent::Done { finish_reason, prompt_tokens, completion_tokens } => {
                    c.finish = finish_reason;
                    c.prompt_tokens = prompt_tokens;
                    c.completion_tokens = completion_tokens;
                }
                GenEvent::Error(e) => c.error = Some(e),
            }
            Ok(())
        };
        generate(runner, vocab, st, genreq, &mut emit)?;
    }
    Ok(c)
}

fn handle_messages(
    stream: &mut TcpStream,
    body: &[u8],
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    ctx: u32,
) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&format!("invalid JSON: {e}"))),
    };
    let (genreq, stream_mode) = match anthropic::parse_messages_request(&req, ctx) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&e)),
    };
    if stream_mode {
        http::write_sse_headers(stream)?;
        let mut sink = anthropic::AnthSink::new(MODEL_ID);
        let mut emit = |ev: GenEvent| -> std::io::Result<()> { sink.on(stream, ev) };
        generate(runner, vocab, st, &genreq, &mut emit)
    } else {
        let c = collect(runner, vocab, st, &genreq)?;
        if let Some(e) = c.error {
            return http::write_json(stream, 500, &err_body(&e));
        }
        http::write_json(
            stream,
            200,
            &anthropic::final_message_json(
                &anthropic::gen_id(), MODEL_ID, &c.content, &c.reasoning, &c.tool_calls, &c.finish,
                c.prompt_tokens, c.completion_tokens,
            ),
        )
    }
}

fn handle_responses(
    stream: &mut TcpStream,
    body: &[u8],
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    ctx: u32,
) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&format!("invalid JSON: {e}"))),
    };
    let (genreq, stream_mode) = match responses::parse_responses_request(&req, ctx) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&e)),
    };
    if stream_mode {
        http::write_sse_headers(stream)?;
        let mut sink = responses::RespSink::new(MODEL_ID);
        let mut emit = |ev: GenEvent| -> std::io::Result<()> { sink.on(stream, ev) };
        generate(runner, vocab, st, &genreq, &mut emit)
    } else {
        let c = collect(runner, vocab, st, &genreq)?;
        if let Some(e) = c.error {
            return http::write_json(stream, 500, &err_body(&e));
        }
        http::write_json(
            stream,
            200,
            &responses::final_response_json(
                &responses::gen_id(), MODEL_ID, &c.content, &c.reasoning, &c.tool_calls,
                c.prompt_tokens, c.completion_tokens,
            ),
        )
    }
}

fn handle_completions(
    stream: &mut TcpStream,
    body: &[u8],
    runner: &DecodeRunner,
    vocab: &Vocab,
    st: &SpecialTokens,
    ctx: u32,
) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&format!("invalid JSON: {e}"))),
    };
    let (genreq, stream_mode) = match openai::parse_completion_request(&req, ctx) {
        Ok(v) => v,
        Err(e) => return http::write_json(stream, 400, &err_body(&e)),
    };
    let id = openai::gen_id();
    let created = openai::now_secs();
    if stream_mode {
        http::write_sse_headers(stream)?;
        let mut emit = |ev: GenEvent| -> std::io::Result<()> {
            match ev {
                GenEvent::Content(t) => {
                    http::write_sse_data(stream, &openai::text_completion_chunk(&id, created, MODEL_ID, &t, None))
                }
                GenEvent::Done { finish_reason, .. } => {
                    http::write_sse_data(stream, &openai::text_completion_chunk(&id, created, MODEL_ID, "", Some(&finish_reason)))?;
                    http::write_sse_data(stream, "[DONE]")
                }
                GenEvent::Error(e) => http::write_sse_data(stream, &err_body(&e)),
                // completions has no reasoning/tool fields → ignore.
                _ => Ok(()),
            }
        };
        generate(runner, vocab, st, &genreq, &mut emit)
    } else {
        let c = collect(runner, vocab, st, &genreq)?;
        if let Some(e) = c.error {
            return http::write_json(stream, 500, &err_body(&e));
        }
        http::write_json(
            stream,
            200,
            &openai::final_text_completion(&id, created, MODEL_ID, &c.content, &c.finish, c.prompt_tokens, c.completion_tokens),
        )
    }
}
