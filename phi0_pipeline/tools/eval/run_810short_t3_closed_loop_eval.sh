#!/usr/bin/env bash
# Compat alias → tools/eval/run_t3_closed_loop_eval.sh（正式通用入口）。
# Prefer: bash launch/eval_t3_closed_loop.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# 历史默认：810short 数据 + 配置（可被环境变量覆盖）。
export CONFIG_NAME="${CONFIG_NAME:-train_810short_egypt_vlm_offset_sonic_revo2}"
WS="${WS:-$(cd "${ROOT}/.." && pwd)}"
export UNIFIED_ROOT="${UNIFIED_ROOT:-${WS}/810short_horizon_v2_egypt_layout}"
export GT_REPO_ID="${GT_REPO_ID:-810short_horizon_v2_egypt_layout}"
exec bash "${ROOT}/tools/eval/run_t3_closed_loop_eval.sh" "$@"
