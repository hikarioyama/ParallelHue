#!/usr/bin/env bash
# 16-way ParallelHue client for Maple-Preview TQ2.
# Usage:
#   ./run-c16.sh
#   MAX_TOKENS=1000 ./run-c16.sh
set -euo pipefail
CONCURRENCY="${CONCURRENCY:-16}"
# shellcheck source=_run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_run_common.sh"
