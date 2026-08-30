#!/usr/bin/env bash
# 16-way ParallelHue client for the local GLM-5.3 Flash serve (2x RTX PRO 6000).
# Assumes the recipe serve is ALREADY listening on 127.0.0.1:8000 — this
# script does NOT launch vLLM. One command → tmux session with 16 panes →
# live colored streams.
#
# Usage (from a normal terminal, NOT already deep inside a tiny pane):
#   ./run-c16.sh
#   CONCURRENCY=32 ./run-c16.sh          # C32 (server must allow max-num-seqs>=32)
#   NO_ATTACH=1 ./run-c16.sh             # create session, do not attach (scripts/CI)
#   MAX_TOKENS=2000 ./run-c16.sh
set -euo pipefail
CONCURRENCY="${CONCURRENCY:-16}"
# shellcheck source=_run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_run_common.sh"
