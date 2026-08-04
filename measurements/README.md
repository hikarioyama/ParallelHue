# Measurements

This directory contains public-safe, machine-readable performance records. The base measurement schema is
[`schema-v1.json`](./schema-v1.json); compatibility records use
[`compatibility-schema-v1.json`](./compatibility-schema-v1.json). Each record keeps conditions, units,
sample counts, and whether a value is measured, observed, derived, or a run condition.

These records are source-repository material; this directory does not claim
that measurements or examples are included in wheel or sdist artifacts.

## Record index

| Record | Model | Workload | Result |
| --- | --- | --- | --- |
| [`deepseek-v4-flash-0731-c16.json`](./deepseek-v4-flash-0731-c16.json) | DeepSeek-V4-Flash-0731 | 2× RTX PRO 6000 Blackwell, TP2, C16, exactly 2,000 completion tokens/request | 1,425.721430518942 tok/s aggregate precursor (`dspark8`/custom plugin), not ParallelHue exact; n=1 |
| [`gemma-3-12b-awq-rtx5070ti.json`](./gemma-3-12b-awq-rtx5070ti.json) | Gemma 3 12B AWQ (requested Gemma 4 12B unavailable) | RTX 5070 Ti; four backend launch attempts | BLOCKED_BEFORE_SERVER_HEALTH; exact/fallback n=0 |
| [`parallelhue-vllm026-qwen25-05b.json`](./parallelhue-vllm026-qwen25-05b.json) | Qwen/Qwen2.5-0.5B-Instruct | Official vLLM 0.26 V1 exact compatibility capture (observed port `18080`) | PASS; exact compatibility capture; n=1 |

The DeepSeek record is a measured precursor `dspark8` renderer/custom-plugin run. It
predates ParallelHue exact, is marked `parallelhue_exact: false`, and must not be presented
as ParallelHue exact evidence. Its render-lag summary is over 16 requests; the benchmark
itself has one run and no repeated same-condition comparison.

The DeepSeek model artifact has no available immutable revision. This is
context-only precursor evidence with no public reproducibility claim. The
record retains the existing source-tree pins as user-specified provenance, not
as a claim that those trees are publicly resolvable.

## Honest-number policy

- Report the workload and server conditions beside every number. A throughput number
  without hardware, software, request shape, and server configuration is incomplete.
- Preserve units and precision from the measurement. Do not silently round, convert, or
  mix tokens/s with requests/s.
- `kind: measured` is directly observed; `kind: derived` is calculated from recorded
  conditions or observations; `kind: condition` describes the setup, not an outcome.
- Always state `n`. `n=1` is one run, not a repeated comparison. `same_condition` is
  `false` unless repeated observations were actually made under the same comparison
  conditions.
- Keep validity and reasons explicit. A valid record is not evidence of a comparison it
  did not perform, and a precursor/custom-plugin result is not ParallelHue exact.
- Records contain no raw model output or token streams, private identifiers, credentials,
  internal hosts, or private image names/digests.
