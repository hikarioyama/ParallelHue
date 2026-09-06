#!/usr/bin/env bash
# Public-safe exact/chunk client for the guarded example server.
# C1 keeps the direct prompt below; for C>1, set PARALLELHUE_PROMPT_FILE
# to a bank with at least that many distinct entries.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PROMPT_FILE="${PARALLELHUE_PROMPT_FILE:-$ROOT/examples/glm-5.3-flash-2x-rtxpro6000/prompts.json}"

MODE="${1:-exact}"
PARALLELHUE_BIN="${PARALLELHUE_BIN:-parallelhue}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-qwen2.5-0.5b}"
PROMPT="${PROMPT:-Write sixteen short color names separated by spaces.}"
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"
CONCURRENCY="${CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-120}"

case "$MODE" in
  exact)
    MAX_TOKENS="${MAX_TOKENS:-32}"
    ;;
  chunk)
    MAX_TOKENS="${MAX_TOKENS:-16}"
    ;;
  *)
    printf 'usage: %s [exact|chunk]\n' "$0" >&2
    exit 64
    ;;
esac
prompt_args=(--prompt "$PROMPT")
if (( CONCURRENCY > 1 )); then
  prompt_args=(--prompt-file "$PROMPT_FILE")
fi

install -d -m 700 "$SOCKET_DIR"
export PARALLELHUE_SOCKET_DIR="$SOCKET_DIR"
exec "$PARALLELHUE_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --mode "$MODE" \
  --timeout "$TIMEOUT" \
  "${prompt_args[@]}"
