#!/usr/bin/env bash
# GT replay: 830demo_skill2_pick_toy_unified — unified_slice token + sonic_v1_1 decode → MuJoCo mp4.
# Hands from unified Dex3 (WBC) slice; no VLA.
#
#   bash tools/eval/run_830_pick_toy_gt_sonic_replay.sh
#   EP=1 bash tools/eval/run_830_pick_toy_gt_sonic_replay.sh
#   EP=0 MAX_FRAMES=300 bash tools/eval/run_830_pick_toy_gt_sonic_replay.sh
#
# Note: eval stack uses Phi_0_wpy/subpackages/gear_sonic_deploy → policy dir name is
# ``sonic_v1_1`` (not WBC-tree ``v1_1``). First run may rebuild TRT engines.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REF="${REF_ROOT:-${UNIFIED_ROOT:-/home/user/workspace/datasets/830/830demo_skill2_pick_toy_unified}}"
EP="${EP:-${UNIFIED_EP:-0}}"
HAND_SOURCE="${HAND_SOURCE:-auto}"
TAG="${HAND_SOURCE}"
WORK="${WORK_DIR:-${ROOT}/experiments/830_pick_toy_gt_sonic_replay_ep${EP}_${TAG}_${STAMP}}"
EVAL_LOG="${LOG:-${ROOT}/experiments/830_pick_toy_gt_sonic_replay_ep${EP}_${TAG}_${STAMP}/launch.log}"
mkdir -p "${WORK}"

echo "=== 830 pick_toy GT replay ep=${EP} hand_source=${HAND_SOURCE} deploy=${DEPLOY_POLICY_DIR:-sonic_v1_1} sim=dex3 ==="
env \
  UNIFIED_ROOT="${REF}" UNIFIED_EP="${EP}" EP="${EP}" \
  DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}" \
  TOKEN_SOURCE=unified_slice \
  HAND_SOURCE="${HAND_SOURCE}" \
  REVO2_HAND="${REVO2_HAND:-0}" \
  HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES:-0}" \
  GT_DECODER_PROPRIO_FROM_DATASET="${GT_DECODER_PROPRIO_FROM_DATASET:-0}" \
  LIVE_SIM_HISTORY_FRAMES="${LIVE_SIM_HISTORY_FRAMES:-10}" \
  WORK_DIR="${WORK}" \
  CUDA_DEVICES="${CUDA_DEVICES:-0}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  LOG="${EVAL_LOG}" \
  MAX_FRAMES="${MAX_FRAMES:-}" \
  bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh"

MP4="${WORK}/gt_sonic_replay.mp4"
echo "log=${EVAL_LOG} mp4=${MP4} pid=$(cat "${WORK}/main.pid" 2>/dev/null || echo '?')"
