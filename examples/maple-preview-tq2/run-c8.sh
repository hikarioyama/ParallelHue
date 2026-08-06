#!/usr/bin/env bash
# ParallelHue client for the local Maple-Preview TQ2 llama-server.
# Same operation as the other examples:
#   one command → tmux session with N panes → auto-attach → live streams
#
# Usage (from a normal terminal, NOT already deep inside a tiny pane):
#   ./run-c8.sh
#   CONCURRENCY=16 ./run-c8.sh
#   MAX_TOKENS=1000 ./run-c8.sh
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PARALLELHUE_BIN="${PARALLELHUE_BIN:-$ROOT/.venv/bin/parallelhue}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8899/v1/chat/completions}"
MODEL="${MODEL:-maple-preview-TQ2_0-head-Q4_K.gguf}"
BACKEND="${BACKEND:-generic}"
MODE="${MODE:-chunk}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
PROMPT="${PROMPT:-Implement a production-quality concurrent LRU cache in Python with TTL, size limits, thread safety, typed APIs, unit tests, and clear module structure. Keep writing complete code and tests until the token limit. Do not stop early.}"

[[ -x "$PARALLELHUE_BIN" ]] || {
  printf 'missing parallelhue bin: %s (pip install -e . in the repo)\n' "$PARALLELHUE_BIN" >&2
  exit 66
}

# High-concurrency Maple TQ2 continuous batching collapses into word loops
# without mild anti-repetition. Defaults can be overridden or cleared:
#   PARALLELHUE_FREQUENCY_PENALTY= PARALLELHUE_REPEAT_PENALTY= ./run-c8.sh
export PARALLELHUE_FREQUENCY_PENALTY="${PARALLELHUE_FREQUENCY_PENALTY-0.3}"
export PARALLELHUE_REPEAT_PENALTY="${PARALLELHUE_REPEAT_PENALTY-1.2}"

extra=(--tmux)
# NO_ATTACH=1 keeps the tmux session but does not attach (scripts/CI).
if [[ "${NO_ATTACH:-0}" == "1" ]]; then
  extra+=(--no-attach)
fi

# --tmux creates the N-pane session and attaches when stdout is a TTY.
exec "$PARALLELHUE_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --mode "$MODE" \
  --prompt "$PROMPT" \
  "${extra[@]}"
