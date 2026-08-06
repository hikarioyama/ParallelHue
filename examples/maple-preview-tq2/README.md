# Maple-Preview TQ2 on RTX 5070 Ti

ParallelHue example for the maple llama.cpp fork + Maple-Preview TQ2 GGUF.
Chunk mode only (no exact scheduler telemetry on llama-server).

## Launch

```sh
# start (server allows 16 concurrent sequences by default)
NP=16 CTX=32768 \
  bash /home/hikari/projects/ParallelHue/examples/maple-preview-tq2/launch-server.sh

# stop
bash /home/hikari/projects/ParallelHue/examples/maple-preview-tq2/launch-server.sh stop
```

Defaults:

- GPU: `CUDA_VISIBLE_DEVICES=2` (RTX 5070 Ti)
- port: `8899`
- model: `/mnt/data/models/maple-preview-GGUF/maple-preview-TQ2_0-head-Q4_K.gguf`
- binary: `$HOME/src/llama.cpp-maple/build/bin/llama-server`
- metrics: enabled (`--metrics`) so ParallelHue summary can read
  `llamacpp:tokens_predicted_total`

## Start ParallelHue

```sh
# 16 panes
bash /home/hikari/projects/ParallelHue/examples/maple-preview-tq2/run-c16.sh

# 8 panes
bash /home/hikari/projects/ParallelHue/examples/maple-preview-tq2/run-c8.sh
```

`run-c16.sh` defaults mild anti-repetition for Maple TQ2 continuous batching:
`PARALLELHUE_FREQUENCY_PENALTY=0.3` and `PARALLELHUE_REPEAT_PENALTY=1.2`.
Override or clear them if needed:

```sh
PARALLELHUE_FREQUENCY_PENALTY= PARALLELHUE_REPEAT_PENALTY= \
  bash /home/hikari/projects/ParallelHue/examples/maple-preview-tq2/run-c16.sh
```

One command → tmux session with N panes → auto-attach → live streams → stop at
`max_tokens` → summary with aggregate tok/s when metrics are available.
