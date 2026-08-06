#!/usr/bin/env bash
# 8-way ParallelHue client for Maple-Preview TQ2.
# Usage:
#   ./run-c8.sh
#   MAX_TOKENS=1000 ./run-c8.sh
set -euo pipefail
CONCURRENCY="${CONCURRENCY:-8}"
# shellcheck source=_run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_run_common.sh"
