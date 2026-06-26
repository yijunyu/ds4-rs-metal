# Beating a C reference at its own game: DeepSeek‑V4‑Flash in Rust on Apple Silicon

*How a Rust + Metal port of [antirez](https://github.com/antirez)'s `ds4` engine
came to sustain ~300 tok/s of prefill on a Mac Studio — and what that taught us
about feeding a GPU.*

---

## A tribute first

This post is, before anything else, a thank‑you to **Salvatore Sanfilippo
([antirez](https://github.com/antirez))**.

When he released **`ds4`** — "DwarfStar" — he did the hard, unglamorous thing:
he wrote a small, legible, MIT‑licensed C engine that runs a frontier‑class
Mixture‑of‑Experts model, **DeepSeek‑V4‑Flash**, on a single Apple Silicon machine,
fast. No cluster. No Python tower. Just C, Metal, and a deep understanding of the
model. Every number in this post is measured **against his engine**, and most of
the Metal kernels our engine runs on the prefill path are **his kernels, verbatim**,
used under the MIT license he chose. The provenance file in the repo marks exactly
which shaders are his.

We didn't beat antirez by being cleverer about the math. We beat the *clock* by a
margin, in one specific place, by being fussier about **how the GPU is fed**. This
is the story of that margin.

## The machine and the model

- **Hardware:** Apple **M1 Ultra**, **128 GB** unified memory, Mac Studio.
- **Model:** **DeepSeek‑V4‑Flash**, quantized to **~86 GB** with an iq2 imatrix
  build. This is a tight fit — the model alone is two‑thirds of the machine's RAM —
  which is exactly what makes scheduling matter.
- **Reference:** antirez's `ds4`, same machine, same GGUF, same context length.

DeepSeek‑V4‑Flash is not a toy. It has Multi‑head Latent Attention (MLA), a sparse
MoE feed‑forward with hundreds of experts, a compressed long‑range memory path, and
a sliding‑window (SWA‑128) raw attention band. The prefill — ingesting a 3000‑token
prompt — is a parade of large matmuls and attention kernels, thousands of GPU
command‑buffer submissions deep.

## The headline

| Phase (ctx = 3000) | antirez `ds4` (C) | ds4-rs-metal (Rust) | Δ |
|---|---|---|---|
| **Prefill** | ~255 tok/s | **~300–304 tok/s** | **+18–19 %** |
| **Decode** | ~22 tok/s | ~22 tok/s | parity |

Two honest caveats, up front, because a benchmark you can't trust is worse than no
benchmark:

1. The prefill numbers compare **each engine in its own faithful configuration**.
   Ours runs the model's trained **compressed long‑range + SWA‑128** path by
   default; antirez's reference at this context runs all‑raw. This is a real
   end‑to‑end *user‑throughput* difference — it's what you'd actually see — but it
   is **not** a claim that we do fewer FLOPs. In identical all‑raw mode we are at
   parity or a hair behind. The arithmetic is the same arithmetic.

2. The part we'll spend the rest of this post on is a **separate, stronger** result:
   a **+13.6 % prefill speedup that is byte‑for‑byte identical** to the baseline
   output (logit cosine = 1.0, max‑diff = 0.0 at ctx 600 *and* 3000). No precision
   traded, no token changed. That one holds in **any** configuration, and it's pure
   GPU scheduling.

## The bubble

Here is the thing we stared at for a long time.

When you run prefill as a sequence of Metal **command buffers** — submit a chunk of
work, wait for it, submit the next — the GPU is not actually busy the whole time.
We measured it: across *every* commit strategy we tried, the GPU's own busy time
was **rock steady at 10.46 s**. But the wall‑clock was ~11.8 s. That ~1.3 s gap —
about **11 %** — was the GPU sitting idle *between* command buffers, ~21 times,
~62 ms each, while the CPU noticed the previous buffer had finished and queued the
next.

We called it "the bubble," and for most of a development cycle we believed it was
**non‑recoverable** — an intrinsic per‑command‑buffer flush latency. We tried the
obvious things and they all failed:

- **Overlap the CPU encode with a pipelined commit?** No effect. The CPU wasn't the
  bottleneck.
- **Go fully asynchronous, never block?** Catastrophe — 53 tok/s. With 86 GB of
  weights pinned, letting command buffers pile up thrashed residency and the GPU
  busy time *ballooned* from 10 s to 54 s.
- **Pack bigger command buffers (fewer, fatter submissions)?** The large‑buffer cost
  ate the savings.
- **Share encoders across chunks?** −0.9 %. Noise.

Every one of these treated the bubble as a *CPU* problem or a *batching* problem. It
was neither.

## Feeding the GPU from the GPU side

The fix, when it finally came, was to stop thinking about the CPU at all.

The bubble exists because each command buffer's completion is observed on the
**host**, and only *then* is the next one queued. The GPU finishes, raises its hand,
and waits for the CPU to hand it the next page. What if the next page were already
on the GPU's desk, ordered to start the instant the current one finishes — with the
CPU never in the loop?

That's an **`MTLEvent`**. We split prefill into command buffers as before, but each
buffer **signals** an event on completion and the next buffer **waits** on that
event *on the GPU's own timeline*. The next buffer is pre‑queued, GPU‑ordered
behind the current one. The GPU flows from buffer to buffer back‑to‑back; the host
is no longer the pacemaker.

The result:

- GPU **busy / span** went from **88 % to 100 %**.
- The bubble collapsed from **1.3 s to 1.1 ms**.
- Prefill went **247 → 280 tok/s** (+13.6 %) — and the logits were **identical**.
  We verified with a cosine/max‑diff harness (not a noisy argmax), cos = 1.0,
  max‑diff = 0.0, at both short and full context.

One subtlety earned its own scar tissue. An early version of this idea had been
*reverted* because it collapsed under memory pressure — under a 77 GB background
daemon it started emitting zeros. The reason was not the events; it was that the
in‑flight command buffers each held **fresh, non‑resident scratch** memory, which
faulted to zeros when the system was tight. The cure was a **bounded, pinned scratch
pool** — O(1) resident buffers, reused — plus a **window of 1** (never more than two
command buffers in flight). With the pool pinned and the window bounded, the win is
both real *and* pressure‑safe. The earlier revert had been right to revert; it was
missing the pool.

> The lesson we keep relearning on this hardware: **an idle GPU is almost never a GPU
> problem.** It's a feeding problem. Deterministic silicon does deterministic work;
> if it's idle, *you* left it idle.

## Two percent more, for free

With the bubble gone, prefill became genuinely **GPU‑compute‑bound** — busy/span at
100 %. So the next lever had to remove *work*, not waiting.

The MoE router dispatches a grid sized to the worst‑case number of tokens any single
expert might receive. The reference is conservative (`ceil(K/4)`). But this model's
router is the aux‑loss‑free *balanced* kind: across a 3000‑token prefill the observed
maximum tokens‑per‑expert was 187, and `ceil(K/8)` already covers 375 — a **2×
safety margin**. Tightening the grid from `K/4` to `K/8` removed the over‑dispatch
and took prefill **295.6 → 304.3 tok/s** (+3 %), still **byte‑identical**. We stopped
there: `K/16` would have been faster but zero‑margin, and a silent token drop is not
worth two percent.

That's the **+18–19 %** headline, stacked: the scheduling win, plus the cheaper
trained long‑range path, plus the grid tightening. The honest, config‑independent,
no‑caveats‑needed number is the **+13.6 % byte‑identical** scheduling result.

## Why Rust, and where the kernels come from

The engine is Rust. The MLA + MoE + attention orchestration, the residency
management, the event‑pipeline scheduler, the bounded scratch pool — all Rust, no GC,
no surprises about *when* memory moves. The **Metal kernels** are a mix: antirez's
verbatim shaders for the prefill path (his bodies, MIT, marked in `PROVENANCE.md`),
plus kernels emitted by **[tile-rs](https://github.com/yijunyu/tile-rs)**, our Rust
kernel‑codegen framework, whose **[Metal backend](https://github.com/yijunyu/tile-rs-metal)**
lowers a shape‑checked tile DSL to MSL. That codegen layer is already open source.

## What we are *not* claiming

- We are **not** smarter than antirez about the model. We learned the model *from*
  his engine, and his C is excellent.
- We do **not** do fewer FLOPs. Same MLA, same MoE, same math.
- We do **not** win decode; it's a tie.
- The headline's biggest single contributor is **scheduling**, and that one we'll
  defend byte‑for‑byte.

What we *will* claim is narrow and true: **on a 128 GB Mac Studio, you can run
DeepSeek‑V4‑Flash through a Rust engine and get noticeably more prefill throughput
than the C reference, with identical outputs from the part that matters most** —
because we got fussier about feeding the GPU.

Thank you, antirez. We're standing on your shoulders, and we tried to leave the view
a little better.

---

*Try it: the hardened macOS binary and a one‑command reproducible benchmark are at
[github.com/yijunyu/ds4-rs-metal](https://github.com/yijunyu/ds4-rs-metal). Bring
your own GGUF; run our `bench/run_bench.sh` against your own build of antirez `ds4`
and check our table on your own silicon.*

*Also please submit your results for us to list in the README — thank you!*
