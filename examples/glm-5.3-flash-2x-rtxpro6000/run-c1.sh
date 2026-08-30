#!/usr/bin/env bash
# Single-stream ParallelHue client for the local GLM-5.3 Flash serve
# (2x RTX PRO 6000): one tmux pane + per-stream summary. Does NOT launch
# vLLM — the recipe serve must ALREADY be up on 127.0.0.1:8000.
#
# Usage (from a normal terminal):
#   ./run-c1.sh
#   NO_ATTACH=1 ./run-c1.sh             # create session, do not attach (scripts/CI)
#   MAX_TOKENS=2000 ./run-c1.sh
set -euo pipefail
CONCURRENCY="${CONCURRENCY:-1}"
# shellcheck source=_run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_run_common.sh"
