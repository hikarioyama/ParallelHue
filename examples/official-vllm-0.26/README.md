# Official vLLM 0.26 example

This example launches the official vLLM 0.26 V1 engine with the ParallelHue
plugin and runs either exact telemetry reconciliation or ordinary SSE chunk
rendering. It is public-safe: use a model path or identifier that you are
allowed to access, and keep the endpoint on localhost unless you add your own
network controls.

The launcher and client are source-repository examples; this documentation does
not claim that they are included in wheel or sdist artifacts.

## Measured defaults

The example defaults are based on the positive live validation on one RTX PRO
6000 Blackwell Max-Q: `Qwen/Qwen2.5-0.5B-Instruct`, served as `qwen2.5-0.5b`,
`CUDA_VISIBLE_DEVICES=1`, example default port `8000` (the observed capture used
port `18080`), `--max-model-len 2048`, `--max-num-seqs 1`,
`--gpu-memory-utilization 0.50`, and eager execution.
The systemd user scope applies `MemoryMax=100G`, `MemorySwapMax=0`, compile
thread limits, a private mode-0700 socket directory, and the explicit
`VLLM_PLUGINS=parallelhue` / `PARALLELHUE_VLLM_EXACT=1` opt-ins.

The measured host used official vLLM `0.26.0` V1, PyTorch `2.11.0+cu130`,
and driver `595.71.05` (CUDA `13.2`). These values describe that observation,
not a promise that every installation has the same packages.

## Launch

Install the project and an official vLLM 0.26 environment first. Then set
`MODEL` to a local model directory or permitted model identifier and launch from
the repository root:

```sh
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  ./examples/official-vllm-0.26/launch-server.example.sh launch
curl --fail http://127.0.0.1:8000/health
```

All launch settings are variables. For example, choose another local model,
GPU, port, scope, or endpoint without editing the script:

```sh
MODEL=/path/to/model CUDA_VISIBLE_DEVICES=0 PORT=8001 \
  ./examples/official-vllm-0.26/launch-server.example.sh launch
```

Stop only the named user systemd scope; the script does not use broad process
matching or kill commands:

```sh
./examples/official-vllm-0.26/launch-server.example.sh stop
```

### FlashInfer note

The measured machine had a `flashinfer-python`/`flashinfer-cubin` package
version mismatch. The launch script therefore defaults to the measured
workaround `FLASHINFER_DISABLE_VERSION_CHECK=1` together with
`--attention-backend FLASH_ATTN`. This is **environment-specific**, not a
universal vLLM or ParallelHue requirement. On an installation with matched
FlashInfer packages, use `FLASHINFER_WORKAROUND=0`; if an administrator has a
different supported backend, set `ATTENTION_BACKEND` explicitly. Do not copy
the workaround as a substitute for diagnosing package compatibility.

## Exact and chunk commands

The client uses the same user and mode-0700 socket directory as the server.
The measured exact CLI request used 32 completion tokens:

```sh
./examples/official-vllm-0.26/run-parallelhue.example.sh exact
```

For an independent telemetry/SSE capture, use the same server and set the
request to 16 tokens:

```sh
MAX_TOKENS=16 ./examples/official-vllm-0.26/run-parallelhue.example.sh exact
```

The fallback chunk request uses 16 completion tokens and never consumes
scheduler telemetry:

```sh
./examples/official-vllm-0.26/run-parallelhue.example.sh chunk
```

The positive live result was `PASS` for both modes (`n=1` each); each is a
single-run compatibility/exactness observation, not a repeated same-condition
comparison. The exact CLI exited 0 and printed the `[EXACT SCHEDULER STEP]`
label. Its independent 16-token capture had 16 events with sequences `0..15`,
step IDs `32..47`, four color slots, one finished event, gap-free sequencing,
byte-identical SSE text/token IDs, and exactly one terminal completion. Chunk
exited 0 with the `[SSE CHUNK MODE]` label. No throughput claim is made.
