#!/usr/bin/env bash
# pick-yellow-box b288: GT closed-loop eval -> sim mp4 only (no npz).
#
# Usage:
#   bash tools/eval/run_pick_yellow_b288_closed_loop_gt_eval.sh [episode_idx]
#   EPISODE_IDX=10 MOTION_SECONDS=0 bash tools/eval/run_pick_yellow_b288_closed_loop_gt_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

EPISODE_IDX="${1:-${EPISODE_IDX:-10}}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/experiments/pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_32k_ddp4/pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_ddp4_32k}"
CUDA_DEVICES="${CUDA_DEVICES:-4}"
MOTION_SECONDS="${MOTION_SECONDS:-0}"
GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-top}"
PROMPT="${PROMPT:-pick yellow box}"

UNIFIED_ROOT="${PHI0_WORKSPACE:-$(cd "${ROOT}/.." && pwd)}/Isaac-GR00T/data/pick_yellow_box_xperience_unified"
WORK_DIR="${WORK_DIR:-${ROOT}/logs/pick_yellow_b288_ep${EPISODE_IDX}_cl_gt_$(date +%Y%m%d_%H%M%S)}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/closed_loop_gt.mp4}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export CHECKPOINT CONFIG_NAME PROMPT MOTION_SECONDS GT_PANEL_LAYOUT
# Async GT closed-loop re-infer -> sim mp4 only (no npz).
export USE_CLOSED_LOOP=1
export GT_REPO_ID=pick_yellow_box_xperience_unified
export UNIFIED_ROOT UNIFIED_EP="${EPISODE_IDX}"
export EGO_MP4="${UNIFIED_ROOT}/videos/chunk-000/observation.images.ego_view/episode_$(printf '%06d' "${EPISODE_IDX}").mp4"
export WRIST_MP4=""
export ROBOT_ONLY=0
export ENABLE_G1_DEBUG_OVERLAY=0
export HAND_RAMP_FRAMES=40
export WORK_DIR OUT_MP4
export INFERENCE_RATE="${INFERENCE_RATE:-0}"
export PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"

# shellcheck source=/dev/null
source "${ROOT}/tools/env/setup_env.sh"

if [[ "${MOTION_SECONDS}" == "0" || "${MOTION_SECONDS}" == "0.0" ]]; then
  PARQUET="${UNIFIED_ROOT}/data/chunk-000/episode_$(printf '%06d' "${EPISODE_IDX}").parquet"
  MAX_FRAMES="$("${PHI0_PY}" - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("${PARQUET}").num_rows)
PY
  )"
  export MAX_FRAMES
fi

echo "==> pick-yellow b288 GT closed-loop eval (no npz)"
echo "    episode=${EPISODE_IDX} max_frames=${MAX_FRAMES:-auto} motion_seconds=${MOTION_SECONDS}"
echo "    checkpoint=${CHECKPOINT}"
echo "    ego=${EGO_MP4}"
echo "    out=${OUT_MP4}"
echo "    work_dir=${WORK_DIR}"

exec bash "${ROOT}/tools/eval/run_sonic_latent_sim_eval.sh"
