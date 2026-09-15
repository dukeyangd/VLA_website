#!/usr/bin/env bash
# 单独打开空 Isaac Sim GUI（可选；日常 QA 用 run_qa.sh + Viser）
set -euo pipefail
LAUNCH="${ISAACSIM_LAUNCH:-/home/neotix/noetix/isaacsim/launch_isaac_sim.sh}"
if [[ ! -f "${LAUNCH}" ]]; then
  echo "[skill2_qa] FATAL: ${LAUNCH} 不存在" >&2
  exit 1
fi
exec bash "${LAUNCH}" "$@"
