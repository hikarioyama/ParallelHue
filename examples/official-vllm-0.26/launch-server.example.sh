#!/usr/bin/env bash
# Public-safe, guarded official vLLM 0.26 example.
set -euo pipefail

MODE="${1:-launch}"
SCOPE_NAME="${SCOPE_NAME:-parallelhue-vllm026-example.scope}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-0.5b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
MEMORY_MAX="${MEMORY_MAX:-100G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
COMPILE_THREADS="${COMPILE_THREADS:-6}"
MAX_JOBS="${MAX_JOBS:-6}"
NVCC_THREADS="${NVCC_THREADS:-2}"
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"
# These defaults reproduce the measured Qwen environment. Set
# FLASHINFER_WORKAROUND=0 when FlashInfer Python and cubin packages are matched;
# this is not a core vLLM need.
FLASHINFER_WORKAROUND="${FLASHINFER_WORKAROUND:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"

case "$MODE" in
  stop)
    systemctl --user stop "$SCOPE_NAME"
    exit 0
    ;;
  launch)
    ;;
  *)
    printf 'usage: %s [launch|stop]\n' "$0" >&2
    exit 64
    ;;
esac

if [[ "$FLASHINFER_WORKAROUND" != 0 && "$FLASHINFER_WORKAROUND" != 1 ]]; then
  printf 'FLASHINFER_WORKAROUND must be 0 or 1\n' >&2
  exit 64
fi
install -d -m 700 "$SOCKET_DIR"

vllm_args=(
  serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enforce-eager
)
systemd_env=(
  "--setenv=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "--setenv=VLLM_PLUGINS=parallelhue"
  "--setenv=PARALLELHUE_VLLM_EXACT=1"
  "--setenv=PARALLELHUE_SOCKET_DIR=$SOCKET_DIR"
  "--setenv=TORCHINDUCTOR_COMPILE_THREADS=$COMPILE_THREADS"
  "--setenv=MAX_JOBS=$MAX_JOBS"
  "--setenv=NVCC_THREADS=$NVCC_THREADS"
)
if [[ "$FLASHINFER_WORKAROUND" == 1 ]]; then
  # Measured-only workaround for a FlashInfer Python/cubin version mismatch.
  systemd_env+=("--setenv=FLASHINFER_DISABLE_VERSION_CHECK=1")
  vllm_args+=(--attention-backend "$ATTENTION_BACKEND")
fi

systemd-run --user --scope --no-block --unit "$SCOPE_NAME" \
  --property="MemoryMax=$MEMORY_MAX" \
  --property="MemorySwapMax=$MEMORY_SWAP_MAX" \
  "${systemd_env[@]}" \
  "$VLLM_BIN" "${vllm_args[@]}"
printf 'started %s; stop cleanly with: %s stop\n' "$SCOPE_NAME" "$0"
