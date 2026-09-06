#!/usr/bin/env bash
# Usage:
#   ./run-parallelhue.example.sh
#   PARALLELHUE_PROMPT_FILE=/path/to/prompts.json CONCURRENCY=32 TMUX_FLAG=1 ./run-parallelhue.example.sh  # custom bank must contain at least 32 distinct entries

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PROMPT_FILE="${PARALLELHUE_PROMPT_FILE:-$ROOT/examples/glm-5.3-flash-2x-rtxpro6000/prompts.json}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-DeepSeek-V4-Flash-0731}"
BACKEND="${BACKEND:-dspark}"
MODE="${MODE:-chunk}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
PARALLELHUE_BIN="${PARALLELHUE_BIN:-parallelhue}"

if [[ "${MODE}" != "chunk" && "${MODE}" != "auto" && "${MODE}" != "exact" ]]; then
  printf 'MODE must be chunk, auto, or exact\n' >&2
  exit 64
fi

extra=()
if [[ "${TMUX_FLAG:-0}" == "1" || "${USE_TMUX:-0}" == "1" ]]; then
  extra+=(--tmux)
fi

exec "$PARALLELHUE_BIN" \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL}" \
  --backend "${BACKEND}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --mode "${MODE}" \
  --prompt-file "${PROMPT_FILE}" \
  "${extra[@]}"
