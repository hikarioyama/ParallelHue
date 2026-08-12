#!/usr/bin/env bash
# Local single-GPU serve for Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark.
#
# Requires vLLM >= 0.27.1 for this DSpark checkpoint (NVFP4 packed markov_w2).
# Default binary: $HOME/vllm027-env/bin/vllm  (NOT $HOME/vllm-env which is 0.26.0)
#
# Note: docker on this host does not expose physical GPU1 (PRO 6000 WS) via
# nvidia-container-toolkit (only GPU0 + GPU2 appear). Native CUDA_VISIBLE_DEVICES=1.
#
# Detach model: setsid+nohup (not bare systemd-run --scope). Scope units keep the
# child in the invoker's cgroup/session; tool timeouts then kill warmup mid-flight.
#
# Usage:
#   ./launch-server.sh              # DSpark speculative (default)
#   ./launch-server.sh dspark       # same
#   ./launch-server.sh baseline     # no speculative decoding
#   MAX_NUM_SEQS=32 ./launch-server.sh
#   ./launch-server.sh stop
set -euo pipefail

MODE="${1:-dspark}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm027-env/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
DSPARK_MODEL_DIR="${DSPARK_MODEL_DIR:-$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SOCKET_DIR="${PARALLELHUE_SOCKET_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue}"
LOG_DIR="${LOG_DIR:-$HOME/logs/parallelhue}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/nemotron35-lightning-nvfp4-dspark.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/nemotron35-lightning-nvfp4-dspark.pid}"

# Demo defaults (C16 parity). Override without editing:
#   MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 ./launch-server.sh
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DSPARK_K="${DSPARK_K:-3}"

stop_server() {
  # Prefer recorded pid, then any vllm on the target GPU.
  if [[ -f "$PID_FILE" ]]; then
    local old
    old="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
      kill "$old" 2>/dev/null || true
      sleep 2
      kill -9 "$old" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
  systemctl --user stop parallelhue-nemotron35-lightning-nvfp4-dspark.scope 2>/dev/null || true
  docker rm -f parallelhue-nemotron35-lightning-nvfp4-dspark >/dev/null 2>&1 || true
  for p in $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    if [[ -r "/proc/$p/cmdline" ]] && tr '\0' ' ' <"/proc/$p/cmdline" | grep -Eqi 'vllm'; then
      kill "$p" 2>/dev/null || true
    fi
  done
  sleep 1
  printf 'stopped lightning server on gpu=%s\n' "$CUDA_VISIBLE_DEVICES"
}

case "$MODE" in
  stop)
    stop_server
    exit 0
    ;;
  baseline|dspark|launch) ;;
  *)
    printf 'usage: %s [dspark|baseline|stop]\n' "$0" >&2
    exit 64
    ;;
esac

[[ -x "$VLLM_BIN" ]] || { printf 'missing vllm: %s\n' "$VLLM_BIN" >&2; exit 66; }
if "$VLLM_BIN" --version 2>/dev/null | grep -qE '0\.26\.'; then
  printf 'refusing vLLM 0.26.x (%s): Lightning DSpark NVFP4 needs >=0.27.1\n' "$VLLM_BIN" >&2
  exit 66
fi
[[ -d "$MODEL_DIR" ]] || { printf 'missing model: %s\n' "$MODEL_DIR" >&2; exit 66; }
[[ -f "$MODEL_DIR/config.json" ]] || { printf 'incomplete model (no config.json): %s\n' "$MODEL_DIR" >&2; exit 66; }
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  printf 'incomplete model (no weight index): %s\n' "$MODEL_DIR" >&2
  exit 66
}

extra=()
if [[ "$MODE" == "dspark" || "$MODE" == "launch" ]]; then
  [[ -d "$DSPARK_MODEL_DIR" ]] || { printf 'missing DSpark draft: %s\n' "$DSPARK_MODEL_DIR" >&2; exit 66; }
  [[ -f "$DSPARK_MODEL_DIR/config.json" ]] || {
    printf 'incomplete DSpark draft (no config.json): %s\n' "$DSPARK_MODEL_DIR" >&2
    exit 66
  }
  [[ -f "$DSPARK_MODEL_DIR/model.safetensors" ]] || {
    printf 'incomplete DSpark draft (no model.safetensors): %s\n' "$DSPARK_MODEL_DIR" >&2
    exit 66
  }
  # External NVIDIA DSpark draft (Qwen3DSparkModel). Not DeepSeek InstantTensor.
  extra+=(--speculative-config "{\"model\":\"${DSPARK_MODEL_DIR}\",\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K}}")
fi

stop_server >/dev/null
install -d -m 700 "$SOCKET_DIR"
install -d -m 755 "$LOG_DIR"
: >"$LOG_FILE"

# Fully detached: new session, stdin closed, logs appended, PID recorded.
setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PARALLELHUE_SOCKET_DIR="$SOCKET_DIR" \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  TORCHINDUCTOR_COMPILE_THREADS=6 \
  MAX_JOBS=6 \
  NVCC_THREADS=2 \
  HF_HUB_OFFLINE=1 \
  "$VLLM_BIN" serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --moe-backend marlin \
    --mamba-backend flashinfer \
    --mamba-cache-mode align \
    --stream-interval 1 \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    "${extra[@]}" \
  </dev/null >>"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
sleep 1
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  printf 'server exited immediately; see %s\n' "$LOG_FILE" >&2
  exit 1
fi

printf 'started pid=%s mode=%s port=%s gpu=%s\n' "$(cat "$PID_FILE")" "$MODE" "$PORT" "$CUDA_VISIBLE_DEVICES"
printf 'bin: %s\n' "$VLLM_BIN"
printf 'log: %s\n' "$LOG_FILE"
printf 'wait for /v1/models then: %s/run-c16.sh\n' \
  "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
printf 'stop: %s stop\n' "$0"
