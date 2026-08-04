#!/usr/bin/env bash
# Intended client request for a server that has reached health.
# The Gemma compatibility attempts documented with this profile did not.
set -euo pipefail

: "${VENV:?set VENV to the vLLM/ParallelHue virtual-environment directory}"
: "${SOCKET_DIR:?set SOCKET_DIR to the private socket directory used by the server}"

PORT="${PORT:-18082}"
MODEL="${MODEL:-gemma-3-12b-it-awq}"
MODE="${MODE:-exact}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT='Write sixteen short color names separated by spaces.'
CLIENT_BIN="$VENV/bin/parallelhue"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:$PORT/v1/chat/completions}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'PORT must be an integer in 1..65535\n' >&2
  exit 64
fi
if [[ "$MODE" != exact && "$MODE" != auto ]]; then
  printf 'MODE must be exact or auto\n' >&2
  exit 64
fi
if [[ ! "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_TOKENS must be a positive integer\n' >&2
  exit 64
fi
[[ -x "$CLIENT_BIN" ]] || { printf 'missing executable: %s\n' "$CLIENT_BIN" >&2; exit 66; }
[[ -d "$SOCKET_DIR" ]] || { printf 'missing socket directory: %s\n' "$SOCKET_DIR" >&2; exit 66; }

# In the recorded Gemma run, both exact and auto were intended requests only;
# neither was executed because no backend reached /health.
exec "$CLIENT_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  --mode "$MODE" \
  --socket-dir "$SOCKET_DIR" \
  --prompt "$PROMPT"
