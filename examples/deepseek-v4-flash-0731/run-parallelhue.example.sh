#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-DeepSeek-V4-Flash-0731}"
MODE="${MODE:-chunk}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
PROMPT='Write a continuous 300-word production-quality Python LRU cache implementation. Keep writing until the token limit.'

if [[ "${MODE}" != "chunk" ]]; then
  printf 'MODE must be chunk; exact and auto are unsupported for this profile\n' >&2
  exit 64
fi

parallelhue \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --mode "${MODE}" \
  --prompt "${PROMPT}"
