#!/usr/bin/env bash
# Local RTX 5070 Ti serve for Maple-Preview TQ2 via the maple llama.cpp fork.
# Usage:
#   ./launch-server.sh              # default C16
#   NP=8 ./launch-server.sh         # C8 capacity
#   ./launch-server.sh stop
set -euo pipefail

MODE="${1:-launch}"
UNIT_NAME="${UNIT_NAME:-parallelhue-maple-preview-tq2.service}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/src/llama.cpp-maple/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/maple-preview-GGUF/maple-preview-TQ2_0-head-Q4_K.gguf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-maple-preview-TQ2_0-head-Q4_K.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Keep CUDA runtime libs next to the maple fork binary when needed.
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/miniforge3/envs/llamacpp-cu131/lib:$HOME/src/llama.cpp-maple/build/bin}"
LOG_FILE="${LOG_FILE:-${XDG_RUNTIME_DIR:-/tmp}/parallelhue-maple-preview-tq2.log}"

# Demo defaults. Override without editing this file:
#   NP=8 CTX=16384 ./launch-server.sh
NP="${NP:-16}"
CTX="${CTX:-32768}"
NGL="${NGL:-99}"

free_port_holders() {
  command -v ss >/dev/null 2>&1 || return 0
  local pid
  while read -r pid; do
    [[ -n "${pid:-}" ]] || continue
    kill "$pid" 2>/dev/null || true
  done < <(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '
    index($0, p) {
      if (match($0, /pid=[0-9]+/)) {
        print substr($0, RSTART + 4, RLENGTH - 4)
      }
    }')
}

port_in_use() {
  command -v ss >/dev/null 2>&1 || return 1
  ss -ltn 2>/dev/null | awk -v p=":${PORT}" 'index($0, p) { found = 1 } END { exit !found }'
}

case "$MODE" in
  stop)
    systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
    free_port_holders
    printf 'stopped %s (port %s)\n' "$UNIT_NAME" "$PORT"
    exit 0
    ;;
  launch|start) ;;
  *)
    printf 'usage: %s [launch|stop]\n' "$0" >&2
    exit 64
    ;;
esac

[[ -x "$LLAMA_SERVER_BIN" ]] || { printf 'missing llama-server: %s\n' "$LLAMA_SERVER_BIN" >&2; exit 66; }
[[ -f "$MODEL_PATH" ]] || { printf 'missing model: %s\n' "$MODEL_PATH" >&2; exit 66; }
command -v systemd-run >/dev/null || { printf 'systemd-run is required\n' >&2; exit 69; }

if ! [[ "$NP" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NP must be a positive integer\n' >&2
  exit 64
fi
if ! [[ "$CTX" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CTX must be a positive integer\n' >&2
  exit 64
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'PORT must be an integer in 1..65535\n' >&2
  exit 64
fi

systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
# legacy scope name from the first draft
systemctl --user stop parallelhue-maple-preview-tq2.scope 2>/dev/null || true
free_port_holders
sleep 1
if port_in_use; then
  printf 'port %s still in use; stop the holder or choose PORT=...\n' "$PORT" >&2
  exit 69
fi

: >"$LOG_FILE"

# Transient user service so launch returns immediately (scope stays attached).
# maple llama.cpp: continuous batching + flash-attn, reasoning off.
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --same-dir \
  --working-directory="$HOME" \
  --property=Type=exec \
  --property=Restart=no \
  --property=MemoryMax=32G \
  --property=MemorySwapMax=0 \
  --property="StandardOutput=append:${LOG_FILE}" \
  --property="StandardError=append:${LOG_FILE}" \
  --setenv=CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --alias "$SERVED_MODEL_NAME" \
    --host "$HOST" --port "$PORT" \
    -ngl "$NGL" \
    -c "$CTX" \
    -np "$NP" \
    -fa on \
    --cont-batching \
    --reasoning off \
    --metrics

printf 'started %s port=%s np=%s ctx=%s\n' "$UNIT_NAME" "$PORT" "$NP" "$CTX"
printf 'log: %s\n' "$LOG_FILE"
printf 'stop: %s stop\n' "$0"
