#!/usr/bin/env bash
# Side-by-side GT replay: teleop template (unified[346:360]) vs measured Dex3 state hands.
#
#   EP=1 bash tools/eval/run_830_walk_gt_sonic_hand_compare.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EP="${EP:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/830_walk_gt_hand_compare_ep${EP}_${STAMP}}"
mkdir -p "${OUT_ROOT}"

_run() {
  local src="$1"
  local work="${OUT_ROOT}/${src}"
  local log="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/logs/830_walk_gt_hand_compare_ep${EP}_${src}_${STAMP}.log"
  echo "=== launch ${src} ep=${EP} ==="
  env \
    EP="${EP}" HAND_SOURCE="${src}" \
    WORK_DIR="${work}" LOG="${log}" STAMP="${STAMP}" \
    bash "${ROOT}/tools/eval/run_830_walk_gt_sonic_replay.sh"
  for _ in $(seq 1 120); do
    if grep -q '\[sonic_latent\] done work_dir=' "${log}" 2>/dev/null; then
      echo "done ${src} mp4=${work}/gt_sonic_replay.mp4"
      return 0
    fi
    sleep 3
  done
  echo "timeout ${src} log=${log}" >&2
  return 1
}

_run unified_gripper
_run measured_state

echo "OUT_ROOT=${OUT_ROOT}"
echo "teleop_template=${OUT_ROOT}/unified_gripper/gt_sonic_replay.mp4"
echo "measured_state=${OUT_ROOT}/measured_state/gt_sonic_replay.mp4"
