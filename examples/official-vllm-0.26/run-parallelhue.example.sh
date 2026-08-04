#!/usr/bin/env bash
# Public-safe exact/chunk client for the guarded example server.
set -euo pipefail

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

install -d -m 700 "$SOCKET_DIR"
export PARALLELHUE_SOCKET_DIR="$SOCKET_DIR"
exec "$PARALLELHUE_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --mode "$MODE" \
  --timeout "$TIMEOUT" \
  --prompt "$PROMPT"
