#!/usr/bin/env bash
# Usage:
#   ./run-parallelhue.example.sh
#   CONCURRENCY=32 TMUX_FLAG=1 ./run-parallelhue.example.sh

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-DeepSeek-V4-Flash-0731}"
BACKEND="${BACKEND:-dspark}"
MODE="${MODE:-chunk}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
PROMPT="${PROMPT:-Write a continuous 300-word production-quality Python LRU cache implementation. Keep writing until the token limit.}"
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
  --prompt "${PROMPT}" \
  "${extra[@]}"
