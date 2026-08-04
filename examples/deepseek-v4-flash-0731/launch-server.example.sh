#!/usr/bin/env bash
# Public-safe example: supply your own publicly permitted image, model, and cache.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-deepseek-v4-flash-0731}"
HOST_PORT="${HOST_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
GPU_DEVICE_REQUEST="\"device=${GPU_DEVICES}\""
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.975}"
CPU_KV_CACHE_GIB="${CPU_KV_CACHE_GIB:-48.5}"

if [[ "${1:-launch}" == "stop" ]]; then
  if container_id="$(docker container ls -a --filter "name=^/${CONTAINER_NAME}$" --quiet)"; then
    if [[ -z "${container_id}" ]]; then
      printf 'container %q is already absent\n' "${CONTAINER_NAME}" >&2
      exit 0
    fi
  else
    docker_status=$?
    printf 'failed to query Docker for container %q\n' "${CONTAINER_NAME}" >&2
    exit "${docker_status}"
  fi
  if docker stop --time "${STOP_TIMEOUT}" "${CONTAINER_NAME}"; then
    exit 0
  else
    docker_status=$?
    printf 'failed to stop container %q\n' "${CONTAINER_NAME}" >&2
    exit "${docker_status}"
  fi
fi
: "${IMAGE:?set IMAGE to a vLLM-compatible container image (no digest required)}"
: "${MODEL_PATH:?set MODEL_PATH to the local model directory to mount read-only}"
: "${HOST_CACHE:?set HOST_CACHE to a writable host cache directory}"

if [[ "${1:-launch}" != "launch" ]]; then
  printf 'usage: %s [launch|stop]\n' "$0" >&2
  exit 64
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  printf 'container %q already exists; stop it with: %q stop\n' "${CONTAINER_NAME}" "$0" >&2
  exit 1
fi

# These guards keep optional compile/memory tuning deliberate and portable.
if [[ -n "${TORCHINDUCTOR_COMPILE_THREADS:-}" && ! "${TORCHINDUCTOR_COMPILE_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TORCHINDUCTOR_COMPILE_THREADS must be a positive integer\n' >&2
  exit 64
fi
if [[ ! "${GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]+$ && "${GPU_MEMORY_UTILIZATION}" != "1" && "${GPU_MEMORY_UTILIZATION}" != "1.0" ]]; then
  printf 'GPU_MEMORY_UTILIZATION must be a decimal in (0,1]\n' >&2
  exit 64
fi
if [[ ! "${CPU_KV_CACHE_GIB}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf 'CPU_KV_CACHE_GIB must be a positive decimal\n' >&2
  exit 64
fi
if [[ ! "${STOP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STOP_TIMEOUT must be a positive integer\n' >&2
  exit 64
fi

vllm_args=(
  --model /model
  --host 0.0.0.0
  --port 8000
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size 2
  --max-model-len 524288
  --max-num-seqs 16
  --max-num-batched-tokens 4096
  --block-size 256
  --enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --cpu-kvcache-space "${CPU_KV_CACHE_GIB}"
  --kv-cache-dtype fp8
  --stream-interval 1
  --seed 0
  --kv-transfer-config '{"connector":"instanttensor","mode":"buffered","backend":"dspark","backend_config":"b12x-a8","fixed_k":5}'
)

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  vllm_args+=(--enforce-eager)
fi
if [[ -n "${VLLM_COMPILATION_CONFIG:-}" ]]; then
  vllm_args+=(--compilation-config "${VLLM_COMPILATION_CONFIG}")
fi

docker_args=(
  run --detach --rm
  --name "${CONTAINER_NAME}"
  --gpus "${GPU_DEVICE_REQUEST}"
  --publish "127.0.0.1:${HOST_PORT}:8000"
  --mount "type=bind,src=${MODEL_PATH},dst=/model,readonly"
  --mount "type=bind,src=${HOST_CACHE},dst=/cache"
  --env HF_HOME=/cache/huggingface
  --env "VLLM_CPU_KVCACHE_SPACE=${CPU_KV_CACHE_GIB}"
)

if [[ -n "${TORCHINDUCTOR_COMPILE_THREADS:-}" ]]; then
  docker_args+=(--env "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}")
fi

docker "${docker_args[@]}" "${IMAGE}" vllm serve "${vllm_args[@]}"
printf 'server started as %s; clean stop: %s stop\n' "${CONTAINER_NAME}" "$0"
