# Gemma 3 12B AWQ on RTX 5070 Ti: blocked reproduction

This profile is a **negative compatibility reproduction**, not an advertised
working setup. Every guarded vLLM 0.26 attempt below exited before the server
reported health. Do not convert the `n=0` request result into PASS.

## Model identity

The requested `Gemma 4 12B` model was not present locally. The model actually
measured was Gemma 3 12B AWQ:

- repository: `gaunernst/gemma-3-12b-it-int4-awq`
- base model: `google/gemma-3-12b-it`
- architecture: `Gemma3ForConditionalGeneration`
- quantization: AWQ INT4, group size 32, zero point enabled
- served name: `gemma-3-12b-it-awq`

This result therefore says nothing about Gemma 4 compatibility.

## Parameterized guarded launch

Supply local paths and the GPU index explicitly. The placeholders below are
not paths embedded in the example:

```sh
export VENV='/path/to/vllm-0.26-venv'
export MODEL_DIR='/path/to/gaunernst-gemma-3-12b-it-int4-awq'
export SOCKET_DIR='/path/to/private/parallelhue-gemma3-socket'
export GPU_INDEX=2
export PORT=18082
```

The launcher uses a user systemd scope with `MemoryMax=100G` and
`MemorySwapMax=0`, `CUDA_VISIBLE_DEVICES=$GPU_INDEX`, six TorchInductor compile
threads, six build jobs, two NVCC threads, offline Hugging Face access,
`VLLM_PLUGINS=parallelhue`, `PARALLELHUE_VLLM_EXACT=1`, and a mode-0700 socket
directory. It serves on `127.0.0.1:$PORT` with the measured flags: maximum
model length 2048 tokens, one sequence, GPU memory fraction 0.88, eager mode,
and language-model-only mode.

Each backend can be reproduced independently. `auto` deliberately omits the
attention-backend flag; the other values pass the vLLM 0.26 flag:

```sh
BACKEND=FLASH_ATTN  bash examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh
BACKEND=auto        bash examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh
BACKEND=FLASHINFER bash examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh
BACKEND=TRITON_ATTN bash examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh
```

For `FLASHINFER`, the launcher additionally sets
`FLASHINFER_DISABLE_VERSION_CHECK=1`, matching the recorded attempt.

### Observed blockers

The following are measured launch attempts, not hypotheses. Each has
`exit_status=1`, `health=false`, and one attempt (`n=1`); no server reached
health in any backend. No client request was therefore possible
(`requests_n=0`, `same_condition=N/A`).

| `BACKEND` | blocker |
| --- | --- |
| `FLASH_ATTN` | Gemma3 partial multimodal-token full attention / `mm_prefix` requires FlashAttention v4, unavailable for this head size. |
| `auto` | `flashinfer-python 0.6.14` and `flashinfer-cubin 0.6.8.post1` version mismatch. |
| `FLASHINFER` (with `FLASHINFER_DISABLE_VERSION_CHECK=1`) | FLASHINFER does not support partial multimodal-token full attention for this Gemma3 configuration. |
| `TRITON_ATTN` | Triton supports `mm_prefix`, but vLLM sampler initialization still imports FlashInfer and hits the same package version mismatch. |

The evidence environment was vLLM `0.26.0`, torch `2.11.0+cu130`,
FlashInfer Python `0.6.14`, FlashInfer cubin `0.6.8.post1`, and
ParallelHue `0.1.0` on an RTX 5070 Ti with 16303 MiB VRAM and driver
`595.71.05` / CUDA `13.2`. These are conditions, not a successful server
claim.

To cleanly stop the named scope, use systemd; it performs the normal graceful
teardown. Do not use `pkill` or `SIGKILL`:

```sh
bash examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh stop
```

## Intended client commands (not executed)

The following are the exact and truthful-fallback commands that would have
been used only after `/health` became ready. They are **intended, not
executed**, because all four server attempts were blocked. The client example
defaults to `max_tokens=16` and the prompt below; select `MODE=exact` or
`MODE=auto`:

```sh
MODE=exact VENV="$VENV" SOCKET_DIR="$SOCKET_DIR" PORT="$PORT" \
  bash examples/gemma-3-12b-awq-rtx5070ti/run-parallelhue.example.sh

MODE=auto VENV="$VENV" SOCKET_DIR="$SOCKET_DIR" PORT="$PORT" \
  bash examples/gemma-3-12b-awq-rtx5070ti/run-parallelhue.example.sh
```

Prompt: `Write sixteen short color names separated by spaces.` Since neither
request ran, there are zero events, zero sequences, zero colors, zero finished
markers, and no terminal completion for this Gemma profile. `n=0` is a
blocked-before-server-health result, never PASS.

## Separate positive official-vLLM exact control

A separate, positive control is not Gemma evidence. It used official vLLM
`0.26` V1 with `Qwen/Qwen2.5-0.5B-Instruct` on an RTX PRO 6000 Blackwell
Max-Q (driver `595.71.05`, CUDA `13.2`) with torch `2.11+cu130` and the same
style of process guardrails.

- Exact CLI, `max_tokens=32`: exit code 0, label `exact` (`n=1`).
- Independent exact capture, `max_tokens=16`: 16 events with sequence IDs
  `0..15`, step IDs `32..47`, four colors, `finished=1`, byte-identical text
  and token IDs, and `terminal=true` (`n=1` capture).
- Chunk CLI, `max_tokens=16`: exit code 0, label `chunk` (`n=1`).

These are validation observations only. No throughput claim is made, and the
positive control does not change the Gemma verdict.
