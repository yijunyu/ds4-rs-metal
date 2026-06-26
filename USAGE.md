# ds4-rs-metal — usage

A single signed macOS binary, **`ds4-server`**: an OpenAI/Anthropic‑compatible HTTP
server that runs DeepSeek‑V4‑Flash GGUFs on Apple Silicon via Metal. No build step,
no toolchain. A drop‑in for antirez's own `ds4-server`.

## Install

Download the latest release for Apple Silicon (arm64) from
[Releases](https://github.com/yijunyu/ds4-rs-metal/releases):

```sh
tar xzf ds4-rs-metal-macos-arm64.tar.gz
cd ds4-rs-metal
./ds4-server --help
# If Gatekeeper blocks it despite signing/notarization:
xattr -dr com.apple.quarantine ds4-server
```

## Run

```sh
./ds4-server --model /path/to/ds4flash-q2.gguf --port 8000 --warm-weights
```

Then use any OpenAI or Anthropic client, or curl:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Once upon a time"}],
       "max_tokens":128,"stream":true}'
```

### Flags

| Flag | Meaning | Default |
|---|---|---|
| `--model <path>` / `-m` | GGUF model file (required) | — |
| `--port <N>` | listen port | 8000 |
| `--host <addr>` | bind address | 127.0.0.1 |
| `--ctx <N>` / `-c` | context length | 32768 |
| `--raw-cap <N>` | raw (SWA) attention window; the trained value is 128 | model default |
| `--warm-weights` | pre‑touch all weights at startup (avoids first‑request stall) | off |
| `--cors` | (accepted; CORS headers are always sent) | — |

### Endpoints

`/v1/chat/completions` · `/v1/messages` (Anthropic) · `/v1/completions` ·
`/v1/responses` · `/v1/models`. Streaming (SSE) and non‑streaming both supported.

### Throughput readout

Each request logs a line to stderr:

```
ds4-profile: prompt=3000tok gen=128tok | tokenize=..ms prefill=..ms | per-tok: step(engine)=..ms ...
```

- **prefill tok/s** = `prompt_tokens / (prefill_ms / 1000)`
- **decode tok/s**  = `1000 / step(engine)_ms`

`bench/run_bench.sh` parses exactly this line.

## Memory budget per quant

The model must fit in unified memory alongside a few GB of working set:

| Quant | Model size | Min machine |
|---|---|---|
| q2 (iq2 imatrix) | ~86 GB | 128 GB |
| q4 | ~150 GB+ | 192 GB+ |

On a 128 GB Mac Studio, **boot out any large background daemon** before timing —
memory pressure perturbs GPU scheduling and can mask the headline throughput.

> ⚠️ **Never force‑kill the server mid‑request.** It drives the Metal GPU; killing a
> process with command buffers in flight can wedge the GPU until reboot. Stop it with
> Ctrl‑C / SIGTERM between requests, or let the request finish first.

## Licensing note

ds4-rs-metal is a derivative work of [antirez/ds4](https://github.com/antirez/ds4)
(MIT). The `NOTICE` and `PROVENANCE.md` shipped in the release tarball MUST be
retained in any redistribution, per the MIT terms.
