# DeepSeek-V4-Flash-0731 example profile

These are public-safe, parameterized examples, not a benchmark recipe. Set
`IMAGE`, `MODEL_PATH`, and `HOST_CACHE` before launching; no image digest or
private path is embedded. `launch-server.example.sh stop` cleanly stops and
removes its named container.

The launcher, client, and this profile are source-repository examples; this
documentation does not claim that they are included in wheel or sdist
artifacts.

The launch profile requests two GPUs with TP=2, a 524288-token context,
16 sequences, 4096 batched tokens, block size 256, prefix caching, 0.975 GPU
memory utilization, 48.5 GiB CPU KV cache, FP8 GPU KV cache, and stream
interval 1. It expresses the dataset-specific InstantTensor buffered / DSpark
`b12x-a8` / fixed-K=5 settings through `--kv-transfer-config`. Those connector
arguments require a vLLM build that implements them; standard vLLM builds may
reject them. Remove or port only after checking that build's public interface.

```sh
export IMAGE='your-public-vllm-image:tag'
export MODEL_PATH='/path/to/permitted/model'
export HOST_CACHE='/path/to/writable/cache'
bash examples/deepseek-v4-flash-0731/launch-server.example.sh
```

## C16 / 2000-token workload template

The ParallelHue example defaults to `--mode chunk`, C16, and 2000 completion
tokens. It uses this exact prompt:

```text
Write a continuous 300-word production-quality Python LRU cache implementation. Keep writing until the token limit.
```

```sh
CONCURRENCY=16 MAX_TOKENS=2000 MODE=chunk \
  bash examples/deepseek-v4-flash-0731/run-parallelhue.example.sh
```

For a direct OpenAI-compatible request, preserve the dataset workload controls:

```sh
curl --no-buffer --fail-with-body \
  -H 'content-type: application/json' \
  http://127.0.0.1:8000/v1/chat/completions \
  --data '{"model":"DeepSeek-V4-Flash-0731","stream":true,"stream_options":{"include_usage":true},"temperature":0,"seed":0,"ignore_eos":true,"max_tokens":2000,"messages":[{"role":"user","content":"Write a continuous 300-word production-quality Python LRU cache implementation. Keep writing until the token limit."}]}'
```

`ignore_eos: true`, temperature 0, seed 0, and exactly 2000 requested
completion tokens are request controls; server-side `--stream-interval 1` is
required by the shown profile. The direct request is one request; use a safe
client-side concurrency harness if reproducing C16 outside ParallelHue.

## Exact-mode boundary and provenance

Do not select `MODE=exact` or `MODE=auto` for this profile. ParallelHue exact
v0.1 supports only the official vLLM 0.26.x private telemetry contract, while
this DSpark/InstantTensor configuration is not evidence of that contract. The
DeepSeek client therefore accepts `MODE=chunk` only and rejects exact and auto
with exit status 64. Chunk mode is truthful for ordinary SSE and is the
default.

The aggregate `1425.721430518942 tok/s` is a precursor result from a
dspark8 renderer/custom plugin, **not** a ParallelHue exact-mode run. It is
included only as context and is not a claim of measured performance here.

Sanitized source pins: dataset label `DeepSeek-V4-Flash-0731`; TP `2`; C16;
completion limit `2000`; context `524288`; batching `4096`; block `256`;
GPU-memory fraction `0.975`; CPU KV `48.5 GiB`; GPU KV `FP8`; connector
`instanttensor` buffered; backend `dspark b12x-a8`; fixed K `5`; temperature
`0`; seed `0`; `ignore_eos=true`; stream interval `1`. Supply an approved
public image and model separately.

The model artifact has no available immutable revision in this record. This is
context-only precursor evidence, so it makes no public reproducibility claim.


The verified precursor record was `n=1` with no repeated same-condition
comparison: 2× RTX PRO 6000 Blackwell, TP2, C16, 32000 total requested
completion tokens, aggregate `1425.721430518942 tok/s`, makespan
`22.444777300115675 s`, accepted `24987`, draft `35060`, and sampled `7013`.
It was valid with `reasons=[]`; renderer lag was median
`0.04182748651348114 s` over `n=16` (range
`0.0012483990285545588..0.04265878605656326 s`). These numbers came from the
precursor `dspark8` renderer/custom plugin, not ParallelHue exact mode.

Sanitized source-tree pins for that precursor are retained as user-specified
provenance: vLLM `1e9c9c...552bf` and SparkInfer `eec30f...fddb62089`. They are
not a claim that those trees or the model artifact are publicly resolvable; an
image digest is intentionally omitted because it is not publicly obtainable.
