# Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark

Local single-GPU (default **GPU1**) recipe for NVIDIA Nemotron 3.5 Lightning with:

- target: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
- draft: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` (external NVIDIA DSpark)
- KV cache: FP8
- runtime: **native `$HOME/vllm027-env` = vLLM 0.27.1** (required)
- viewer: ParallelHue `--backend dspark --mode chunk`

This is **not** the DeepSeek InstantTensor DSpark path. Server-side speculation is plain vLLM `--speculative-config method=dspark` with the external draft checkpoint. ParallelHue's `--backend dspark` only selects the client metrics/color profile.

## Why 0.27.1 (not 0.26 / not Docker here)

- `$HOME/vllm-env` is **0.26.0** and dies loading this draft:
  `RuntimeError: tensor a (512) vs b (256)` on `markov_head.markov_w2` (NVFP4 packed `[V,256] uint8` for rank 512).
- Official card pins `vllm/vllm-openai:v0.27.1`, but on this host Docker only exposes GPU0+GPU2 (physical GPU1 / PRO 6000 WS is missing from nvidia-container). Native `CUDA_VISIBLE_DEVICES=1` is used instead.

## Prerequisites

```bash
$HOME/vllm027-env/bin/vllm --version   # 0.27.1
ls "$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/model.safetensors.index.json"
ls "$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark/model.safetensors"
```

## Launch (server)

```bash
./launch-server.sh          # DSpark ON, GPU1, KV FP8, :8000
./launch-server.sh baseline # no speculative decoding
./launch-server.sh stop
```

Wait until healthy **before** starting ParallelHue:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
# log: ~/logs/parallelhue/nemotron35-lightning-nvfp4-dspark.log
```

If panes show `Connection refused`, the server is down — fix serve first.

## Start (ParallelHue client)

```bash
./run-c16.sh
# if launched detached / non-TTY:
tmux ls | grep parallelhue
tmux attach -t parallelhue-<id>
```

Defaults: `--backend dspark --mode chunk --concurrency 16`.

## Notes

- First boot does torch.compile + FlashInfer autotune (~1–3 min). Later boots reuse cache.
- Do not copy DeepSeek `--kv-transfer-config` / InstantTensor flags here.
- K=3 is the NVIDIA serve default; SPEED-Bench tables use draft length 7 (different condition).
