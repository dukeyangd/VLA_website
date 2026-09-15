#!/usr/bin/env bash
# Compat alias → tools/eval/launch_gt_sonic_replay.sh
# Prefer: UNIFIED_ROOT=... bash launch/eval_gt_sonic.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-$(cd "${ROOT}/.." && pwd)}"
export UNIFIED_ROOT="${UNIFIED_ROOT:-${WS}/810short_horizon_v2_egypt_layout}"
export MAX_FRAMES="${MAX_FRAMES:-2703}"
exec bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh" "$@"
