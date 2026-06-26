#!/bin/bash
# run_bench.sh — prefill + decode throughput for ds4-server (Rust) and, optionally,
# antirez ds4-server (C), on the SAME machine / model / context.
#
# It starts each server, sends one long-prompt request, and reads the engine's own
# `ds4-profile:` stderr line (prefill ms + per-token step ms) — no guessing. Servers
# are stopped with SIGTERM between runs; a Metal process is NEVER SIGKILLed
# mid-request (that can wedge the GPU until reboot).
#
#   DS4_RS_BIN=./ds4-server \
#   DS4_C_BIN=/path/to/antirez/ds4-server \   # optional
#   DS4_GGUF=/path/to/ds4flash-q2.gguf \
#     bash bench/run_bench.sh
#
# Optional: DS4_CTX (3000), DS4_GEN (64 tokens to generate), DS4_PORT (8123),
#           DS4_PROMPT_FILE, DS4_WARM_SECS (max secs to wait for the port).
set -uo pipefail

: "${DS4_RS_BIN:?set DS4_RS_BIN to the ds4-server binary}"
: "${DS4_GGUF:?set DS4_GGUF to a DeepSeek-V4-Flash GGUF}"
C_BIN="${DS4_C_BIN:-}"
CTX="${DS4_CTX:-3000}"
GEN="${DS4_GEN:-64}"
PORT="${DS4_PORT:-8123}"
WARM_SECS="${DS4_WARM_SECS:-600}"
PROMPT_FILE="${DS4_PROMPT_FILE:-}"
[ -f "$DS4_GGUF" ] || { echo "model not found: $DS4_GGUF" >&2; exit 1; }

# A long deterministic prompt (~ a few hundred tokens) if none supplied.
if [ -z "$PROMPT_FILE" ]; then
  PROMPT_FILE="$(mktemp)"
  yes "The memory hierarchy separates storage into levels by latency and capacity." \
    | head -n 40 | tr '\n' ' ' > "$PROMPT_FILE"
fi
PROMPT_JSON="$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' "$PROMPT_FILE")"

bench_one () {                       # $1=binary  $2=label  -> "prefill_tps decode_tps"
  local bin="$1" label="$2" log; log="$(mktemp)"
  "$bin" --model "$DS4_GGUF" --port "$PORT" --ctx "$CTX" --warm-weights >"$log" 2>&1 &
  local pid=$!
  # Wait for the port to accept connections.
  local waited=0
  until nc -z 127.0.0.1 "$PORT" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "  $label: server exited early; see $log" >&2; return 1; }
    sleep 2; waited=$((waited+2)); [ "$waited" -ge "$WARM_SECS" ] && { echo "  $label: warm timeout" >&2; break; }
  done
  # One request — body parsed but we only need the engine's stderr profile line.
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":$PROMPT_JSON}],\"max_tokens\":$GEN}" \
    >/dev/null
  # Graceful stop (SIGTERM, then wait — never SIGKILL a Metal process).
  kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

  # ds4-profile: prompt=Ntok ... prefill=X.Xms | per-tok: step(engine)=Y.Yms ...
  local line ptok pf step
  line="$(grep -a 'ds4-profile:' "$log" | tail -1)"
  ptok="$(echo "$line" | grep -oE 'prompt=[0-9]+' | grep -oE '[0-9]+')"
  pf="$(echo "$line"   | grep -oE 'prefill=[0-9.]+' | grep -oE '[0-9.]+')"
  step="$(echo "$line" | grep -oE 'step\(engine\)=[0-9.]+' | grep -oE '[0-9.]+')"
  local pf_tps dec_tps
  pf_tps="$(awk -v p="${ptok:-0}" -v ms="${pf:-0}" 'BEGIN{print (ms>0)?p/(ms/1000):0}')"
  dec_tps="$(awk -v ms="${step:-0}" 'BEGIN{print (ms>0)?1000/ms:0}')"
  printf '  %-14s prefill %.1f tok/s  decode %.1f tok/s  (prompt=%s, prefill=%sms, step=%sms)\n' \
    "$label" "$pf_tps" "$dec_tps" "${ptok:-?}" "${pf:-?}" "${step:-?}" >&2
  echo "$pf_tps $dec_tps"
}

echo "== ds4 throughput bench (ctx=$CTX, gen=$GEN) ==" >&2
read -r RS_PF RS_DEC < <(bench_one "$DS4_RS_BIN" "ds4-server")
if [ -n "$C_BIN" ] && [ -x "$C_BIN" ]; then
  read -r C_PF C_DEC < <(bench_one "$C_BIN" "antirez(C)")
else
  echo "  (DS4_C_BIN unset — skipping the antirez reference column)" >&2; C_PF=""; C_DEC=""
fi

echo "--------------------------------------------------------------"
printf '%-18s %14s %14s\n' "phase" "antirez ds4(C)" "ds4-rs-metal"
if [ -n "$C_PF" ]; then
  printf '%-18s %11.1f/s %11.1f/s  (%+.1f%%)\n' "prefill @ctx$CTX" "$C_PF" "$RS_PF" \
    "$(awk -v r="$RS_PF" -v c="$C_PF" 'BEGIN{print (c>0)?(r-c)/c*100:0}')"
  printf '%-18s %11.1f/s %11.1f/s  (%+.1f%%)\n' "decode" "$C_DEC" "$RS_DEC" \
    "$(awk -v r="$RS_DEC" -v c="$C_DEC" 'BEGIN{print (c>0)?(r-c)/c*100:0}')"
else
  printf '%-18s %14s %11.1f/s\n' "prefill @ctx$CTX" "-" "$RS_PF"
  printf '%-18s %14s %11.1f/s\n' "decode" "-" "$RS_DEC"
fi
echo "--------------------------------------------------------------"
