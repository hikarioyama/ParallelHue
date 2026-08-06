#!/usr/bin/env bash
# Local single-GPU serve for Qwen3.6-35B-A3B-NVFP4 + ParallelHue exact plugin.
# Usage:
#   ./launch-server.sh          # baseline
#   ./launch-server.sh mtp      # MTP speculative (num_speculative_tokens=3)
#   MAX_NUM_SEQS=32 ./launch-server.sh mtp   # C32 capacity (default remains 16)
#   ./launch-server.sh stop
set -euo pipefail

MODE="${1:-baseline}"
SCOPE_NAME="${SCOPE_NAME:-parallelhue-qwen35a3b-nvfp4.scope}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm-env/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-35B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-35b-a3b-nvfp4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$MODEL_DIR/chat_template.jinja}"

# Demo defaults (tweet parity = C16). Override for C32 without editing this file:
#   MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 ./launch-server.sh mtp
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"

case "$MODE" in
  stop)
    systemctl --user stop "$SCOPE_NAME" 2>/dev/null || true
    sleep 2
    for p in $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
      kill "$p" 2>/dev/null || true
    done
    printf 'stopped %s\n' "$SCOPE_NAME"
    exit 0
    ;;
  baseline|mtp|launch) ;;
  *)
    printf 'usage: %s [baseline|mtp|stop]\n' "$0" >&2
    exit 64
    ;;
esac

[[ -x "$VLLM_BIN" ]] || { printf 'missing vllm: %s\n' "$VLLM_BIN" >&2; exit 66; }
[[ -d "$MODEL_DIR" ]] || { printf 'missing model: %s\n' "$MODEL_DIR" >&2; exit 66; }

systemctl --user stop "$SCOPE_NAME" 2>/dev/null || true
sleep 1
install -d -m 700 "$SOCKET_DIR"

extra=()
if [[ "$MODE" == "mtp" ]]; then
  # Default K=3 from checkpoint README; override with MTP_K=1..N
  MTP_K="${MTP_K:-3}"
  extra+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K},\"moe_backend\":\"triton\"}")
fi

# Eager mode is intentionally disabled for throughput; set ENFORCE_EAGER=1 to re-enable.
eager=()
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  eager+=(--enforce-eager)
fi

systemd-run --user --scope --no-block --unit="$SCOPE_NAME" \
  -p MemoryMax=100G -p MemorySwapMax=0 \
  --setenv=CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --setenv=VLLM_PLUGINS=parallelhue \
  --setenv=PARALLELHUE_VLLM_EXACT=1 \
  --setenv=PARALLELHUE_SOCKET_DIR="$SOCKET_DIR" \
  --setenv=FLASHINFER_DISABLE_VERSION_CHECK=1 \
  --setenv=TORCHINDUCTOR_COMPILE_THREADS=6 \
  --setenv=MAX_JOBS=6 \
  --setenv=NVCC_THREADS=2 \
  --setenv=HF_HUB_OFFLINE=1 \
  "$VLLM_BIN" serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype fp8 \
    --stream-interval 1 \
    --quantization compressed-tensors \
    --dtype bfloat16 \
    --trust-remote-code \
    --chat-template "$CHAT_TEMPLATE" \
    "${eager[@]}" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    "${extra[@]}"

printf 'started %s mode=%s port=%s (wait for Application startup complete)\n' "$SCOPE_NAME" "$MODE" "$PORT"
printf 'stop: %s stop\n' "$0"
