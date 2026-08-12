#!/usr/bin/env bash
# ParallelHue client for local Nemotron 3.5 Lightning NVFP4 + DSpark server.
# Same operation as the Qwen / dspark demos:
#   one command → tmux session with N panes → auto-attach → live colored streams
#
# Usage (from a normal terminal, NOT already deep inside a tiny pane):
#   ./run-c16.sh
#   CONCURRENCY=32 ./run-c16.sh
#   MODE=chunk MAX_TOKENS=2000 ./run-c16.sh
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PARALLELHUE_BIN="${PARALLELHUE_BIN:-$ROOT/.venv/bin/parallelhue}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
# Client-side profile only (colors / accepted-token counters). Server-side DSpark
# is enabled by launch-server.sh --speculative-config.
BACKEND="${BACKEND:-dspark}"
# chunk default: DSpark uses vLLM V2; ParallelHue exact is V1-proven only so far.
MODE="${MODE:-chunk}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
PROMPT="${PROMPT:-Implement a production-quality concurrent LRU cache in Python with TTL, size limits, thread safety, typed APIs, unit tests, and clear module structure. Keep writing complete code and tests until the token limit.}"
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"

[[ -x "$PARALLELHUE_BIN" ]] || {
  printf 'missing parallelhue bin: %s (pip install -e . in the repo)\n' "$PARALLELHUE_BIN" >&2
  exit 66
}

install -d -m 700 "$SOCKET_DIR"
export PARALLELHUE_SOCKET_DIR="$SOCKET_DIR"

exec "$PARALLELHUE_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --mode "$MODE" \
  --tmux \
  --prompt "$PROMPT"
