#!/usr/bin/env bash
# Public-safe negative reproduction for Gemma 3 12B AWQ on one GPU.
# This is not a claim that any backend starts a healthy server.
set -euo pipefail

usage() {
  printf 'usage: %s [launch|stop]\n' "$0" >&2
}

ACTION="${1:-launch}"
if [[ "$ACTION" != launch && "$ACTION" != stop ]]; then
  usage
  exit 64
fi

SCOPE_UNIT="${SCOPE_UNIT:-parallelhue-gemma3-12b.scope}"

if [[ "$ACTION" == stop ]]; then
  # Confirm that the scope is absent before treating stop as successful.
  load_state="$(systemctl --user show "$SCOPE_UNIT" -p LoadState --value)"
  if [[ "$load_state" == not-found ]]; then
    exit 0
  fi
  systemctl --user stop "$SCOPE_UNIT"
  exit $?
fi

: "${VENV:?set VENV to the vLLM 0.26 virtual-environment directory}"
: "${MODEL_DIR:?set MODEL_DIR to the local Gemma 3 12B AWQ directory}"
: "${SOCKET_DIR:?set SOCKET_DIR to a private writable ParallelHue socket directory}"
: "${GPU_INDEX:?set GPU_INDEX to the GPU index to expose}"
: "${PORT:?set PORT to the localhost server port}"

BACKEND="${BACKEND:-TRITON_ATTN}"
VLLM_BIN="$VENV/bin/vllm"

if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  printf 'GPU_INDEX must be a non-negative integer\n' >&2
  exit 64
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'PORT must be an integer in 1..65535\n' >&2
  exit 64
fi
case "$BACKEND" in
  FLASH_ATTN|FLASHINFER|TRITON_ATTN|auto) ;;
  *)
    printf 'BACKEND must be FLASH_ATTN, auto, FLASHINFER, or TRITON_ATTN\n' >&2
    exit 64
    ;;
esac

[[ -x "$VLLM_BIN" ]] || { printf 'missing executable: %s\n' "$VLLM_BIN" >&2; exit 66; }
[[ -d "$MODEL_DIR" ]] || { printf 'missing model directory: %s\n' "$MODEL_DIR" >&2; exit 66; }
command -v systemd-run >/dev/null || { printf 'systemd-run is required\n' >&2; exit 69; }
command -v systemctl >/dev/null || { printf 'systemctl is required\n' >&2; exit 69; }

install -d -m 700 "$SOCKET_DIR"

server_args=(
  serve "$MODEL_DIR"
  --served-model-name gemma-3-12b-it-awq
  --host 127.0.0.1
  --port "$PORT"
  --max-model-len 2048
  --max-num-seqs 1
  --gpu-memory-utilization 0.88
  --enforce-eager
  --language-model-only
)
if [[ "$BACKEND" != auto ]]; then
  server_args+=(--attention-backend "$BACKEND")
fi

env_args=(
  "CUDA_VISIBLE_DEVICES=$GPU_INDEX"
  TORCHINDUCTOR_COMPILE_THREADS=6
  MAX_JOBS=6
  NVCC_THREADS=2
  HF_HUB_OFFLINE=1
  VLLM_PLUGINS=parallelhue
  PARALLELHUE_VLLM_EXACT=1
  "PARALLELHUE_SOCKET_DIR=$SOCKET_DIR"
)
if [[ "$BACKEND" == FLASHINFER ]]; then
  env_args+=(FLASHINFER_DISABLE_VERSION_CHECK=1)
fi

# The measured attempts all exited before /health became ready.  Keep this
# command explicit so each BACKEND value can be reproduced and recorded.
exec systemd-run --user --scope --unit="$SCOPE_UNIT" \
  -p MemoryMax=100G -p MemorySwapMax=0 \
  env "${env_args[@]}" "$VLLM_BIN" "${server_args[@]}"
