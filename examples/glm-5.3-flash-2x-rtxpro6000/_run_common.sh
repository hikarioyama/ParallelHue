#!/usr/bin/env bash
# Shared ParallelHue launcher body for the local GLM-5.3 Flash serve
# (2x RTX PRO 6000). Sourced by run-c16.sh / run-c1.sh after setting the
# CONCURRENCY default. Assumes the recipe serve is ALREADY listening on
# 127.0.0.1:8000 (vLLM, model glm-5.3-flash-local) — this script does NOT
# launch vLLM. One command → tmux session with CONCURRENCY panes → live
# colored streams.
#
# Usage (from a normal terminal, NOT already deep inside a tiny pane):
#   NO_ATTACH=1 ./run-c16.sh        # create session, do not attach (scripts/CI)
#   MAX_TOKENS=2000 ./run-c16.sh    # override the 1024 default
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PROMPT_FILE="$ROOT/examples/glm-5.3-flash-2x-rtxpro6000/prompts.json"
if [[ -x "${PARALLELHUE_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv/bin/parallelhue" ]]; then
  PARALLELHUE_BIN="$ROOT/.venv/bin/parallelhue"
elif command -v parallelhue >/dev/null 2>&1; then
  PARALLELHUE_BIN="$(command -v parallelhue)"
else
  PARALLELHUE_BIN="$ROOT/.venv/bin/parallelhue"
fi

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-glm-5.3-flash-local}"
BACKEND="${BACKEND:-mtp}"   # GLM-5.3 Flash has MTP/ReplaySSM → color + spec counters
MODE="${MODE:-chunk}"       # docker image has no ParallelHue vLLM exact plugin
CONCURRENCY="${CONCURRENCY:-16}"   # default only if unset; run-cN.sh sets it first
MAX_TOKENS="${MAX_TOKENS:-1024}"
TIMEOUT="${TIMEOUT:-180}"   # 16-way GLM can exceed the 60s CLI default
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"

[[ -x "$PARALLELHUE_BIN" ]] || {
  printf 'missing parallelhue bin: %s (pip install -e . in the repo, or set PARALLELHUE_BIN)\n' "$PARALLELHUE_BIN" >&2
  exit 66
}

install -d -m 700 "$SOCKET_DIR"
export PARALLELHUE_SOCKET_DIR="$SOCKET_DIR"

extra=(--tmux)
# NO_ATTACH=1 keeps the tmux session but does not attach (scripts/CI).
if [[ "${NO_ATTACH:-0}" == "1" ]]; then
  extra+=(--no-attach)
fi

# Each pane picks its own prompt from prompts.json via --worker-index.
exec "$PARALLELHUE_BIN" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --mode "$MODE" \
  --timeout "$TIMEOUT" \
  --prompt-file "$PROMPT_FILE" \
  "${extra[@]}"
