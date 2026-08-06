# Maple-Preview TQ2 example

Public-safe recipe for serving Maple-Preview TQ2 through a maple llama.cpp fork
and rendering it with ParallelHue in chunk mode.

This example does **not** claim official Maple speculative-decoding support.
No draft / MTP / EAGLE / DSpark head is required or included. ParallelHue stays
monochrome here because the backend profile is `generic` (non-speculative).

The launcher and client are source-repository examples; this documentation does
not claim that they are included in wheel or sdist artifacts.

## Prerequisites

1. Build the maple llama.cpp fork so `llama-server` exists, either on `PATH` or
   at a path you pass as `LLAMA_SERVER_BIN`.
2. Obtain a Maple-Preview TQ2 GGUF you are allowed to use, e.g.
   `maple-preview-TQ2_0-head-Q4_K.gguf`.
3. Install ParallelHue in this repository:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

4. Linux user systemd (`systemd-run --user`) and `tmux` are required for the
   default launch/run flow.

## Launch

From the repository root, supply your local model path explicitly:

```sh
MODEL_PATH=/path/to/maple-preview-TQ2_0-head-Q4_K.gguf \
  ./examples/maple-preview-tq2/launch-server.sh
```

Optional overrides:

```sh
MODEL_PATH=/path/to/model.gguf \
LLAMA_SERVER_BIN=/path/to/llama-server \
CUDA_VISIBLE_DEVICES=0 \
PORT=8899 \
NP=16 \
CTX=32768 \
  ./examples/maple-preview-tq2/launch-server.sh
```

Wait until health succeeds:

```sh
curl --fail http://127.0.0.1:8899/health
```

Stop only the named user service:

```sh
./examples/maple-preview-tq2/launch-server.sh stop
```

### What the launcher enables

- continuous batching (`--cont-batching`)
- flash-attn (`-fa on`)
- reasoning off
- Prometheus metrics (`--metrics`) so ParallelHue summary can read
  `llamacpp:tokens_predicted_total`
- default capacity: `NP=16`, `CTX=32768` (2048 tokens/slot)

No machine-specific model path is embedded. `MODEL_PATH` is required.

## Start ParallelHue

```sh
# 16 panes
./examples/maple-preview-tq2/run-c16.sh

# 8 panes
./examples/maple-preview-tq2/run-c8.sh
```

Defaults:

- backend: `generic`
- mode: `chunk`
- max tokens: `1000`
- prompt: long code-generation text so decode stays busy
- mild anti-repetition for Maple TQ2 continuous batching:
  `PARALLELHUE_FREQUENCY_PENALTY=0.3`, `PARALLELHUE_REPEAT_PENALTY=1.2`

Override or clear anti-repetition if needed:

```sh
PARALLELHUE_FREQUENCY_PENALTY= PARALLELHUE_REPEAT_PENALTY= \
  ./examples/maple-preview-tq2/run-c16.sh
```

One command → tmux session with N panes → auto-attach → live streams → stop at
`max_tokens` → summary with aggregate tok/s when metrics are available.

## Notes for reproducers

- Colors stay off on purpose: speculative backends (`mtp`, `dspark`) are the
  only profiles that paint stream colors.
- Aggregate tok/s needs the launcher `--metrics` flag and a generic backend that
  understands `llamacpp:tokens_predicted_total` (already wired in ParallelHue).
- High concurrency without anti-repetition can produce word loops on this TQ2
  checkpoint; that is a sampling/model behavior, not a ParallelHue display bug.
