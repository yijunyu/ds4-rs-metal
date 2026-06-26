# ds4-rs-metal

**An open-source Rust + Metal inference engine for DeepSeek‑V4‑Flash on Apple Silicon
— at or ahead of the original [antirez/ds4](https://github.com/antirez/ds4)
("DwarfStar") C reference on a Mac Studio.**

> ds4-rs-metal is a **derivative work of [antirez/ds4](https://github.com/antirez/ds4)**
> (MIT). It would not exist without antirez's reference engine and the ggml lineage
> behind it. The antirez C/Metal sources are **vendored in‑tree** under
> `benchmarks/ds4_msl/upstream/` (pinned at commit `d615ab0`) so the project builds
> with no external fetch. See [`NOTICE`](NOTICE) and [`PROVENANCE.md`](PROVENANCE.md).

---

## Headline

On an **Apple M1 Ultra, 128 GB Mac Studio**, running **DeepSeek‑V4‑Flash** quantized
to **~86 GB (iq2 imatrix)**:

| Phase (ctx = 3000) | antirez `ds4` (C reference) | **ds4-rs-metal** (Rust) | Δ |
|---|---|---|---|
| **Prefill** | ~255 tok/s | **~300–304 tok/s** | **+18–19 %** |
| **Decode** | ~22 tok/s | **~22 tok/s** | at parity / within run‑variance |

Each engine in its **own faithful/default configuration** at ctx 3000 (best‑of‑3).
See [How the comparison is measured](#how-the-comparison-is-measured) for the honest
details, and [Documentation](#documentation) for everything else.

---

## Build from source

ds4-rs-metal is a normal Cargo workspace — **no prebuilt anything required**.

**Requirements:** macOS on Apple Silicon, **Xcode Command Line Tools** (`xcode-select
--install` — for Metal + the C compiler that builds the vendored antirez engine), and
**Rust** (the pinned toolchain in `rust-toolchain.toml` is fetched automatically).

```sh
git clone https://github.com/yijunyu/ds4-rs-metal
cd ds4-rs-metal
cargo build --release -p ds4_server --features metal
```

That produces `target/release/ds4-server`. (A signed + notarized binary is also
attached to each [Release](https://github.com/yijunyu/ds4-rs-metal/releases) if you'd
rather not build.)

## Run

`ds4-server` is an OpenAI/Anthropic‑compatible HTTP server — a drop‑in for antirez's
own `ds4-server`. Bring any DeepSeek‑V4‑Flash GGUF (the headline uses the ~86 GB q2
imatrix build; needs a 128 GB machine).

```sh
./target/release/ds4-server --model /path/to/ds4flash-q2.gguf --port 8000 --warm-weights
```

Then talk to it with any OpenAI/Anthropic client, or curl:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"The capital of France is"}],
       "max_tokens":64}'
```

| Flag | Meaning | Default |
|---|---|---|
| `--model <path>` / `-m` | GGUF model file (required) | — |
| `--port <N>` | listen port | 8000 |
| `--host <addr>` | bind address | 127.0.0.1 |
| `--ctx <N>` / `-c` | context length | 32768 |
| `--raw-cap <N>` | raw (SWA) attention window | model default (128) |
| `--warm-weights` | pre‑touch weights at startup | off |

Endpoints: `/v1/chat/completions`, `/v1/messages` (Anthropic), `/v1/completions`,
`/v1/responses`, `/v1/models`. Each request logs a `ds4-profile:` line with prefill +
per‑token timings. See [`USAGE.md`](USAGE.md) for the full flag list + memory budget.

## Layout

```
crates/ds4_engine/   model engine: GGUF, tokenizer, decode loop, MLA + MoE dispatch
crates/ds4_metal/    Metal backend: residency + event-pipeline scheduler, encoders
crates/ds4_server/   the ds4-server HTTP binary (OpenAI/Anthropic compatible)
benchmarks/ds4_msl/  Metal shader sources: emitted/ (tile-rs codegen),
                     bridge_shims/, upstream/ds4/ (vendored antirez @ d615ab0, MIT)
```

## Reproduce the benchmark

[`bench/run_bench.sh`](bench/run_bench.sh) starts a server, sends a long‑prompt
request, and reads the engine's own `ds4-profile:` timings — for ds4-server and,
optionally, your own antirez `ds4-server` build:

```sh
DS4_RS_BIN=./target/release/ds4-server \
DS4_C_BIN=/path/to/antirez/ds4-server \
DS4_GGUF=/path/to/ds4flash-q2.gguf \
  bash bench/run_bench.sh
```

## How the comparison is measured

- **The headline is production‑vs‑production.** ds4-rs-metal's faithful config runs the
  model's trained **SWA‑128 + compressed long‑range** path (`raw_cap = 128`); the
  antirez reference at ctx 3000 runs all‑raw. That ~+19 % is a real **end‑to‑end
  user‑throughput** difference — **not** a "fewer FLOPs" claim.
- **Same all‑raw mode, both engines are at parity or we're a hair behind** — same MLA +
  MoE math. The advantage is in **how the work is scheduled on the GPU** (now open in
  `crates/ds4_metal` — the event‑pipeline scheduler), plus the cheaper trained
  long‑range path by default.
- **Decode** is at parity within run variance (~22 tok/s); no decode win claimed.

## Documentation

- **[`USAGE.md`](USAGE.md)** — full flag list, endpoints, throughput readout, memory budget.
- **[`bench/run_bench.sh`](bench/run_bench.sh)** — reproduce the comparison table.
- **[`NOTICE`](NOTICE)** / **[`PROVENANCE.md`](PROVENANCE.md)** — antirez/ggml attribution
  (MIT); which Metal kernels are upstream‑verbatim vs original.
- **Engineering write‑up** — the prefill‑scheduling story:
  [English](docs/ds4-rs-metal-apple-silicon-en.md) · [中文](docs/ds4-rs-metal-apple-silicon-zh.md).

## Credits

- **[Salvatore Sanfilippo (antirez)](https://github.com/antirez/ds4)** — the `ds4` /
  DwarfStar C engine this is built on. The prefill‑path Metal kernels (and the C engine
  we FFI for prefill) are his, vendored verbatim under MIT (see `PROVENANCE.md`). Thank
  you for making it, and for making it MIT.
- **The ggml authors** — the quantization + tensor lineage underneath.
- **[tile-rs](https://github.com/yijunyu/tile-rs)** — the Rust kernel‑codegen framework;
  its [Metal backend](https://github.com/yijunyu/tile-rs-metal) emits part of the MSL
  here (`benchmarks/ds4_msl/emitted/`).

## License

Original work in ds4-rs-metal is **MIT OR Apache‑2.0** ([`LICENSE-MIT`](LICENSE-MIT) /
[`LICENSE-APACHE`](LICENSE-APACHE)). Vendored antirez/ds4 + ggml portions remain under
their upstream **MIT** license — reproduced in [`NOTICE`](NOTICE).
