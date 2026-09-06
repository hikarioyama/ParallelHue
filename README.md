# ParallelHue

ParallelHue renders a local, OpenAI-compatible streaming response with a truthful
label: `EXACT TOKEN PROVENANCE` when scheduler and detokenizer telemetry prove
each emitted token's role, or `SSE CHUNK MODE` for ordinary streaming chunks.

## Design philosophy

Local LLM serving is increasingly a multi-agent, parallel, and batched
workload. Aggregate throughput across concurrent requests is therefore
increasingly a first-class measure. But a single aggregate tok/s number hides
the mechanism: concurrent contribution, stalls, speculative accepted bursts,
and interleaving between requests. ParallelHue therefore places each stream's
observed accepted-draft, target, and bonus tokens side by side. Exact colors
mark verified generation steps, not token roles, a numeric step-ID modulo
calculation, or the size of a transport chunk.

Color is only used when the selected backend profile actually runs speculative
decoding (`mtp`, `dspark`, `dflash`). Ordinary non-speculative backends (`generic`, and
`auto` when it resolves there) stay monochrome: without verified speculative-step
telemetry there is no step progression for the palette to encode. `NO_COLOR`
still forces monochrome
for every backend.

That distinction is why the client keeps an explicit semantic boundary between
`exact` and SSE `chunk` mode. Exact colors require scheduler token provenance,
native detokenizer byte traces, token IDs, and SSE text to reconcile. For each
request, the first newly observed verified step uses the first palette entry;
each later new step advances the shared four-color cycle in observation order,
wrapping after four. All fragments of one step retain that step's color even
when their token roles differ. Chunk colors describe transport chunks and never
pretend to identify token roles or verified scheduler steps.
The visualization should make aggregate behavior legible without overstating
what the evidence supports.

In exact mode, colors indicate verified step progression. Independent token
provenance still identifies accepted-draft, target, and bonus output, and those
roles remain available to exact-mode counters and metadata. Rejected draft
proposals are not emitted output and are not displayed as tokens. A Unicode span
that crosses steps and cannot be assigned truthfully stays neutral.

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
  --mode auto --prompt-file examples/glm-5.3-flash-2x-rtxpro6000/prompts.json
```

`--endpoint`, `--model`, `--prompt`, `--prompt-file`, `--max-tokens`,
`--concurrency`, `--api-key`, `--mode`, `--socket-dir`, and `--timeout` are
configurable. The same values can be supplied with `PARALLELHUE_ENDPOINT`,
`PARALLELHUE_MODEL`, `PARALLELHUE_PROMPT`, `PARALLELHUE_PROMPT_FILE`,
`PARALLELHUE_MAX_TOKENS`, `PARALLELHUE_CONCURRENCY`, `PARALLELHUE_API_KEY`,
`PARALLELHUE_MODE`, and `PARALLELHUE_SOCKET_DIR`. `OPENAI_API_KEY` is also
accepted. `--tmux` creates one pane per worker when `tmux` is installed;
otherwise the CLI explicitly uses the single-terminal worker path.

Every parallel stream must receive a distinct, non-empty prompt. A direct
`--prompt` or positional prompt is valid for C1; C>1 requires
`--prompt-file` (or `PARALLELHUE_PROMPT_FILE`) with at least N exact-distinct
entries. The first N entries are selected in file order; prompts are never
cycled, modulo-selected, synthesized, or normalized. An insufficient or
duplicate bank is rejected before any tmux session or model request.
`PARALLELHUE_PROMPT_FILE` can point to a custom bank when a run needs more
entries than the shipped examples provide.

`chunk` mode never consumes telemetry and always says `SSE CHUNK MODE`. It
retains its four-color cycle, assigning the next palette entry to each emitted
transport chunk.
`exact` mode fails closed if telemetry is absent, has sequence gaps, or cannot
be reconciled byte-for-byte with the SSE text and token IDs. `auto` starts with
exact reconciliation when telemetry is available and visibly downgrades to
`SSE CHUNK MODE` when it is not. Chunk colors are never described as scheduler
steps.

## vLLM exact telemetry (opt in)

Exact mode depends on a private, version- and capability-gated plugin contract:
vLLM 0.26.x, or the pinned GLM image's `0.1.dev20051+g487ecf187` fork. The
package registers the vLLM general plugin entry point `parallelhue`; enable
exact telemetry explicitly in the server environment. vLLM loads registered
plugins by default. If a deployment restricts `VLLM_PLUGINS`, add `parallelhue`
to its existing allowlist rather than disabling the model backend's plugins:

```sh
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

The server and client must share the same private socket directory and uid by
default. A root-run container can explicitly set `PARALLELHUE_SOCKET_UID` to
the host client's uid for the bind-mounted directory.

Text and token IDs need not share SSE boundaries: for example, GLM can hold
`<` until the next delta while already returning its token ID. The client
buffers these partial prefixes and verifies the complete text before coloring
it. Actual text removal or rewriting by reasoning/tool parsers is not treated
as a match: `exact` fails closed, while `auto` visibly uses `SSE CHUNK MODE`.

The plugin is never enabled implicitly. Unsupported vLLM versions or missing
capabilities fail loudly when exact telemetry is explicitly requested. Without
the plugin, use `chunk` or `auto`; an ordinary OpenAI-compatible SSE server is
never called a token-provenance backend.

## Tested matrix / Measurements

The [measurement index](measurements/README.md) links the machine-readable
records and their conditions. The current tested matrix is intentionally narrow:

| Case | Result and caveat |
| --- | --- |
| Official vLLM 0.26.0 V1 + `Qwen/Qwen2.5-0.5B-Instruct` on an NVIDIA RTX PRO 6000 Blackwell Max-Q | Historical scheduler-step telemetry `PASS`, `n=1` (observed server port `18080`); not validation of the current token-role protocol or a throughput claim. See the [record](measurements/parallelhue-vllm026-qwen25-05b.json). |
| GLM-5.3-Flash EXL3 4 bpw + DFlash on 2× RTX PRO 6000, TP2, C16, pinned v0.6.0 image | Current exact token provenance `PASS`, one C16 run: all 16 requests completed with 1024 verified tokens each. This is exactness evidence, not a throughput comparison. See the [recipe](examples/glm-5.3-flash-2x-rtxpro6000/). |
| DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 Blackwell, TP2, C16 | `~1425.721 tok/s` aggregate (rounded, approximate), `n=1` precursor `dspark8`/custom-plugin result, **not ParallelHue exact evidence**. See the [record](measurements/deepseek-v4-flash-0731-c16.json). |
| Requested Gemma 4 12B (not present locally); actual Gemma 3 12B AWQ on RTX 5070 Ti | `BLOCKED_BEFORE_SERVER_HEALTH` across four backends, `n=0`; negative compatibility evidence, **not proof that ParallelHue requires a specific GPU**. See the [record](measurements/gemma-3-12b-awq-rtx5070ti.json). |

ParallelHue contains no CUDA kernels or device checks. Only the tested matrix is
recorded: exact compatibility hinges on the supported scheduler/detokenizer
hooks plus the selected model and backend support. A blocked model/backend combination does
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
- [Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark](examples/nemotron35-lightning-30b-a3b-nvfp4-dspark/):
  [server launcher](examples/nemotron35-lightning-30b-a3b-nvfp4-dspark/launch-server.sh) and
  [c16 client](examples/nemotron35-lightning-30b-a3b-nvfp4-dspark/run-c16.sh). Native vLLM on
  GPU1 with external NVIDIA DSpark draft + KV FP8; viewer uses `--backend dspark`
  in chunk mode.
- [Maple-Preview TQ2 chunk recipe](examples/maple-preview-tq2/README.md):
  [server launcher](examples/maple-preview-tq2/launch-server.sh),
  [c8 client](examples/maple-preview-tq2/run-c8.sh), and
  [c16 client](examples/maple-preview-tq2/run-c16.sh). llama.cpp metrics +
  generic monochrome rendering; `MODEL_PATH` is required and no host-local
  model path is embedded.
- [GLM-5.3-Flash EXL3 4 bpw + DFlash](examples/glm-5.3-flash-2x-rtxpro6000/):
  [telemetry image](examples/glm-5.3-flash-2x-rtxpro6000/Dockerfile.exact),
  [c8 client](examples/glm-5.3-flash-2x-rtxpro6000/run-c8.sh), and
  [c16 client](examples/glm-5.3-flash-2x-rtxpro6000/run-c16.sh). The client
  defaults to `MODE=exact BACKEND=dflash` and requires the server setup below.

Eight-stream tmux runs use four aligned columns and two rows instead of
tmux's uneven tiled arrangement. The grid rebalances on terminal resize,
with at most one cell of rounding difference; pane zoom is preserved.

The existing GLM C8 recipe selects eight different tasks from the first eight
entries of its 16-entry prompt bank: Python concurrent TTL LRU cache, Rust
lock-free MPSC queue, Go token-bucket HTTP middleware, TypeScript/React
virtualized data grid, Haskell parser-combinator JSON, C11 SIMD Gaussian blur,
Kotlin Room/Flow offline-first cache, and SQL org-chart recursive CTE/window
rollups.

### GLM exact server setup

Build the telemetry image from the repository root. It preserves the pinned
model-serving image and installs ParallelHue without changing its model,
GPU, quantization, or DFlash settings:

```sh
docker build --network=none \
  -f examples/glm-5.3-flash-2x-rtxpro6000/Dockerfile.exact \
  -t parallelhue-glm53-exact:local .
export PARALLELHUE_SOCKET_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/parallelhue"
install -d -m 700 "$PARALLELHUE_SOCKET_DIR"
```

Use `parallelhue-glm53-exact:local` in the existing model launcher, retaining
its model mounts and serving arguments. Add these Docker options before the
image name; the plugin must be enabled when vLLM starts:

```sh
--mount "type=bind,src=$PARALLELHUE_SOCKET_DIR,dst=/run/parallelhue"
--env PARALLELHUE_VLLM_EXACT=1
--env PARALLELHUE_SOCKET_DIR=/run/parallelhue
--env "PARALLELHUE_SOCKET_UID=$(id -u)"
```

Once the server is ready, run either the eight-way recipe
`bash examples/glm-5.3-flash-2x-rtxpro6000/run-c8.sh` or the existing
16-way recipe `bash examples/glm-5.3-flash-2x-rtxpro6000/run-c16.sh`.
An unmodified upstream image does not provide these frames: use `MODE=chunk`
explicitly if running it without the plugin.


## Validation

Run the CPU regression suite with `python -m pytest`. Live exact validation
also exercises the real server, Unix transport, and terminal viewer: the GLM
C16 run above verified 16,384 output tokens without falling back to chunk
mode. A separate 256-token live request confirmed that delayed `<` text
reconciles correctly. The earlier official vLLM capture uses the historical
scheduler-step protocol and is retained as such.

## Architecture and security boundary

ParallelHue is a **shared viewer** plus **swappable inference recipes**:
the tmux/color/summary UX stays common, while speculative-decoding metric
profiles live under `src/parallelhue/backends/` (`generic` / `mtp` /
`dspark` / `dflash`) and per-model launch/run scripts live under `examples/`. Plug a
backend in; do not fork the viewer for each engine.


Each client run generates a 32-character lowercase hexadecimal `run_id` and
request IDs of the form `ph1_<run_id>_<stream index>`. The exact transport is a
run-scoped AF_UNIX datagram at `$PARALLELHUE_SOCKET_DIR/<run_id>.sock`; the
parent directory is mode 0700 and the socket mode 0600, owned by the current
uid. Schema version 2 uses two independent frame streams: `ProvenanceFrame`
for scheduler token IDs and roles, and `TextFrame` for native detokenizer text
and UTF-8 byte spans. They join by request, absolute token offset, exact token
identity, and independent source sequence, then reconcile against SSE.
Scheduler-step IDs remain annotations, not token-role evidence. The receiver
and renderer use bounded queues and do not retain a persistent token log.

This v1 package is intentionally single-user, same-host, Linux software. The
socket is a local trust boundary, not a remote transport, authentication
system, or multi-tenant isolation mechanism. Keep the socket directory private.
Terminal text is sanitized before ANSI rendering to prevent control, ANSI, and
bidirectional-text injection.

## Limitations

Exact support requires the supported private hook contracts and capability
checks. DFlash role classification requires single-sample verification and
non-adaptive drafts; other speculative algorithms do not gain exact support
merely by selecting a metric profile. Unsupported versions, missing frames,
sequence gaps, token mismatches, and incomplete text fail closed. Other
servers and remote deployments use chunk mode. ParallelHue makes no general
throughput, latency, or quality claim; it is a visualization client.

The design is informed by public work on streaming token visualization and
OpenAI-compatible serving APIs.
The repository provides sanitized measurement records and public examples
without private infrastructure details or raw model/token streams; these
source files are not a wheel or sdist contents claim.

## Roadmap and community

ParallelHue is intended to grow beyond its current vLLM integration. The
roadmap includes adapters for more inference engines and speculative algorithms
beyond the current DFlash token-provenance integration. The
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
