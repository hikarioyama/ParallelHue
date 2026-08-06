# ParallelHue

ParallelHue renders a local, OpenAI-compatible streaming response with a truthful
label: `EXACT SCHEDULER STEP` when the supported telemetry plugin proves scheduler
steps, or `SSE CHUNK MODE` when it only has ordinary streaming chunks.

## Design philosophy

Local LLM serving is increasingly a multi-agent, parallel, and batched
workload. Aggregate throughput across concurrent requests is therefore
increasingly a first-class measure. But a single aggregate tok/s number hides
the mechanism: concurrent contribution, stalls, speculative accepted bursts,
and interleaving between requests. ParallelHue therefore places each stream's
observed scheduler-step boundaries and accepted-token bursts side by side,
using the same color vocabulary within each stream to make those contributions
and gaps visible. The colors are not decorative and do not assert a global
scheduler iteration shared across streams; they are a causal view of the
events the telemetry actually proves.

Color is only used when the selected backend profile actually runs speculative
decoding (`mtp`, `dspark`). Ordinary non-speculative backends (`generic`, and
`auto` when it resolves there) stay monochrome: without draft/accept structure
there is nothing for the palette to encode. `NO_COLOR` still forces monochrome
for every backend.

That distinction is why the client keeps an explicit semantic boundary between
`exact` and SSE `chunk` mode. Exact colors are emitted only after scheduler
telemetry, token IDs, sequence, and raw text reconcile; chunk colors describe
transport chunks and never pretend to be scheduler steps. The visualization
should make aggregate behavior legible without overstating what the evidence
supports.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
parallelhue --model my-model --mode auto "Explain this function"
```

The package has no runtime dependencies beyond Python 3.10+ standard library.
The `parallelhue` entry point and `python -m parallelhue` invoke the same CLI.

## Configuration and examples

```sh
parallelhue \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model my-model --max-tokens 256 --concurrency 2 \
  --mode auto --prompt "Write a small parser"
```

`--endpoint`, `--model`, `--prompt`, `--max-tokens`, `--concurrency`,
`--api-key`, `--mode`, `--socket-dir`, and `--timeout` are configurable. The
same values can be supplied with `PARALLELHUE_ENDPOINT`, `PARALLELHUE_MODEL`,
`PARALLELHUE_PROMPT`, `PARALLELHUE_MAX_TOKENS`, `PARALLELHUE_CONCURRENCY`,
`PARALLELHUE_API_KEY`, `PARALLELHUE_MODE`, and `PARALLELHUE_SOCKET_DIR`.
`OPENAI_API_KEY` is also accepted. `--tmux` creates one pane per worker when
`tmux` is installed; otherwise the CLI explicitly uses the single-terminal
worker path.

`chunk` mode never consumes telemetry and always says `SSE CHUNK MODE`.
`exact` mode fails closed if telemetry is absent, has sequence gaps, or cannot
be reconciled byte-for-byte with the SSE text and token IDs. `auto` starts with
exact reconciliation when telemetry is available and visibly downgrades to
`SSE CHUNK MODE` when it is not. Chunk colors are never described as scheduler
steps.

## vLLM exact telemetry (opt in)

Exact mode depends on the private, version-gated plugin contract for
`vllm>=0.26,<0.27`. The package registers the vLLM general plugin entry point
`parallelhue`; enable that loader and exact mode explicitly in the same user
account that runs vLLM and ParallelHue:

```sh
export VLLM_PLUGINS=parallelhue
export PARALLELHUE_VLLM_EXACT=1
export PARALLELHUE_SOCKET_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue"
install -d -m 700 "$PARALLELHUE_SOCKET_DIR"
```

The vLLM 0.26 default `stream_interval` is `1`, and exact mode requires that
value to remain `1`; do not override it. ParallelHue requests token IDs with
`return_token_ids: true`. After starting vLLM with the variables above, run an
exact client request against its OpenAI-compatible chat endpoint:

```sh
parallelhue \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model my-model --max-tokens 256 --mode exact \
  --prompt "Write a small parser"
```

The vLLM process and client must use the same private socket directory and
same uid. Reasoning or tool parsers can transform or split the raw text
exposed through SSE versus the hook; because exact reconciliation compares raw
text and token IDs byte-for-byte, such a mismatch downgrades `auto` to
`SSE CHUNK MODE` and makes `exact` fail closed.

The plugin is never enabled implicitly. Unsupported vLLM versions or missing
capabilities fail loudly when exact telemetry is explicitly requested. Without
the plugin, use `chunk` or `auto`; an ordinary OpenAI-compatible SSE server is
never called a scheduler-step backend.

## Tested matrix / Measurements

The [measurement index](measurements/README.md) links the machine-readable
records and their conditions. The current tested matrix is intentionally narrow:

| Case | Result and caveat |
| --- | --- |
| Official vLLM 0.26.0 V1 + `Qwen/Qwen2.5-0.5B-Instruct` on an NVIDIA RTX PRO 6000 Blackwell Max-Q | Exact telemetry `PASS`, `n=1` (observed server port `18080`); this is compatibility/exactness evidence only, not a throughput claim. See the [record](measurements/parallelhue-vllm026-qwen25-05b.json). |
| DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 Blackwell, TP2, C16 | `~1425.721 tok/s` aggregate (rounded, approximate), `n=1` precursor `dspark8`/custom-plugin result, **not ParallelHue exact evidence**. See the [record](measurements/deepseek-v4-flash-0731-c16.json). |
| Requested Gemma 4 12B (not present locally); actual Gemma 3 12B AWQ on RTX 5070 Ti | `BLOCKED_BEFORE_SERVER_HEALTH` across four backends, `n=0`; negative compatibility evidence, **not proof that ParallelHue requires a specific GPU**. See the [record](measurements/gemma-3-12b-awq-rtx5070ti.json). |

ParallelHue contains no CUDA kernels or device checks. Only the tested matrix is
recorded: exact compatibility hinges on the vLLM 0.26 frontend plus the
selected model and backend support. A blocked model/backend combination does
not establish a general GPU requirement.

## Examples and source recipes

The examples keep launch commands in source-repository scripts rather than
duplicating large blocks here. The examples and measurement records are
source-repository material; this documentation does not claim that they are
included in wheel or sdist artifacts:

- [Official vLLM 0.26 exact/chunk example](examples/official-vllm-0.26/README.md):
  [server launcher](examples/official-vllm-0.26/launch-server.example.sh) and
  [ParallelHue client command](examples/official-vllm-0.26/run-parallelhue.example.sh).
- [DeepSeek-V4-Flash-0731 precursor profile](examples/deepseek-v4-flash-0731/README.md):
  [server launcher](examples/deepseek-v4-flash-0731/launch-server.example.sh) and
  [ParallelHue client command](examples/deepseek-v4-flash-0731/run-parallelhue.example.sh).
  Its measured throughput remains precursor/custom-plugin evidence, not
  ParallelHue exact evidence.
- [Gemma 3 12B AWQ blocked profile](examples/gemma-3-12b-awq-rtx5070ti/README.md):
  [server launcher](examples/gemma-3-12b-awq-rtx5070ti/launch-server.example.sh) and
  [intended client command](examples/gemma-3-12b-awq-rtx5070ti/run-parallelhue.example.sh).
  All four server attempts were blocked before health, so the client command
  was not executed.
- [Qwen3.6-35B-A3B-NVFP4 + MTP recipe](examples/qwen36-35b-a3b-nvfp4/):
  [server launcher](examples/qwen36-35b-a3b-nvfp4/launch-server.sh) and
  [c16 client](examples/qwen36-35b-a3b-nvfp4/run-c16.sh). Uses the shared
  viewer with `--backend mtp`; model/path defaults are `$HOME/...` and
  overridable by env.
- [Maple-Preview TQ2 chunk recipe](examples/maple-preview-tq2/README.md):
  [server launcher](examples/maple-preview-tq2/launch-server.sh),
  [c8 client](examples/maple-preview-tq2/run-c8.sh), and
  [c16 client](examples/maple-preview-tq2/run-c16.sh). llama.cpp metrics +
  generic monochrome rendering; `MODEL_PATH` is required and no host-local
  model path is embedded.


## Validation

Validation is the test suite (`python -m pytest`) plus a live `--mode chunk`
smoke against a local OpenAI-compatible endpoint. The official vLLM 0.26
exact-mode capture linked above is a positive compatibility observation
(`n=1`), not a throughput claim; exact mode remains scoped to the private
vLLM 0.26.x hook contract.

## Architecture and security boundary

ParallelHue is a **shared viewer** plus **swappable inference recipes**:
the tmux/color/summary UX stays common, while speculative-decoding metric
profiles live under `src/parallelhue/backends/` (`generic` / `mtp` /
`dspark`) and per-model launch/run scripts live under `examples/`. Plug a
backend in; do not fork the viewer for each engine.


Each client run generates a 32-character lowercase hexadecimal `run_id` and
request IDs of the form `ph1_<run_id>_<stream index>`. The exact transport is a
run-scoped AF_UNIX datagram at `$PARALLELHUE_SOCKET_DIR/<run_id>.sock`; the
parent directory is mode 0700 and the socket mode 0600, owned by the current
uid. Telemetry is decoded into the shared `StepEvent` schema and reconciled by
request, sequence, token IDs, and raw text. The receiver and multi-stream
renderer use bounded queues and do not retain a persistent token log.

This v1 package is intentionally single-user, same-host, Linux software. The
socket is a local trust boundary, not a remote transport, authentication
system, or multi-tenant isolation mechanism. Keep the socket directory private.
Terminal text is sanitized before ANSI rendering to prevent control, ANSI, and
bidirectional-text injection.

## Limitations

Exact support is intentionally limited to the tested private vLLM 0.26.x hook
contract and requires the plugin's capability checks. Other vLLM releases,
other servers, non-streaming requests, and remote deployments use chunk mode or
are rejected by exact mode. ParallelHue makes no throughput, latency, or
quality claim; it is a visualization client.

The design is informed by public work on streaming token visualization and
OpenAI-compatible serving APIs.
The repository provides sanitized measurement records and public examples
without private infrastructure details or raw model/token streams; these
source files are not a wheel or sdist contents claim.

## Roadmap and community

ParallelHue is intended to grow beyond its current vLLM integration. The
roadmap includes adapters for more inference engines and richer visibility into
speculative decoding systems, including MTP and DFlash-style approaches. The
long-term goal is to build an open ecosystem around truthful inference
visualization and make ParallelHue a broadly used, interoperable industry
standard rather than a viewer tied to one engine.

The current maintainers can test CUDA deployments on NVIDIA SM120 hardware, but
do not have access to MLX environments or non-SM120 hardware. Compatibility
reports are therefore especially valuable. Reports of working and blocked
configurations, detailed issues, adapter proposals, and pull requests are all
welcome. Please include the inference engine and version, model family,
hardware, operating mode, and the observed result, while excluding credentials
and raw private model output.

## License

MIT; see [LICENSE](LICENSE).
