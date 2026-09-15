#!/usr/bin/env bash
# 810demo step30k → train-aligned GT closed-loop → MuJoCo sim mp4.
#
# Usage:
#   bash tools/eval/run_810demo_closed_loop_eval.sh [episode_idx] [max_frames]
#   bash tools/eval/run_810demo_closed_loop_eval.sh 0 0          # full episode
#   PROPRIO_SOURCE=gt bash tools/eval/run_810demo_closed_loop_eval.sh 0 400
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EP="${1:-${UNIFIED_EP:-0}}"
MAX_FRAMES="${2:-${MAX_FRAMES:-0}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-${ROOT}/experiments/810demo_closed_loop_ep${EP}_${STAMP}}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/closed_loop_gt.mp4}"

CHECKPOINT="${CHECKPOINT:-/mnt/data2/wpy/workspace/phi-0-810/experiments/810demo_offline_vlm_b16_ddp4_noqpos_30k_20260729_172752/phi0_step30000.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_810demo_egypt_vlm_b512}"
UNIFIED_ROOT="${UNIFIED_ROOT:-/mnt/data2/wpy/workspace/810demo_egypt_layout}"
GT_REPO_ID="${GT_REPO_ID:-810demo_egypt_layout}"
PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
CUDA_DEVICES="${CUDA_DEVICES:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export CHECKPOINT CONFIG_NAME UNIFIED_ROOT UNIFIED_EP="${EP}"
export GT_REPO_ID USE_CLOSED_LOOP=1 GT_PANEL_LAYOUT=top
# max_frames<=0 → full episode (closed-loop runs until GT exhausted).
if [[ "${MAX_FRAMES}" -le 0 ]]; then
  export MAX_FRAMES=0
  export MOTION_SECONDS=0
else
  export MAX_FRAMES
  export MOTION_SECONDS="$(python3 -c "print(round(int('${MAX_FRAMES}')/50.0, 2))")"
fi
export EGO_MP4="${UNIFIED_ROOT}/videos/chunk-000/observation.images.ego_view/episode_$(printf '%06d' "${EP}").mp4"
export WRIST_MP4="${UNIFIED_ROOT}/videos/chunk-000/observation.images.left_wrist/episode_$(printf '%06d' "${EP}").mp4"
export UNIFIED_PARQUET="${UNIFIED_PARQUET:-${UNIFIED_ROOT}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet}"
export WORK_DIR OUT_MP4 ROBOT_ONLY=0 ENABLE_G1_DEBUG_OVERLAY="${ENABLE_G1_DEBUG_OVERLAY:-0}" HAND_RAMP_FRAMES=40
export INFERENCE_RATE="${INFERENCE_RATE:-0}"
export PROPRIO_SOURCE
# Match CONTROL_FPS so GT panel + sim aren't stretched by 30fps record catch-up.
export RECORD_FPS="${RECORD_FPS:-50}"
export PHI0_WORKSPACE=/mnt/data2/wpy/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DEPLOY_POLICY_DIR=low_latency
export DEPLOY_OBS_CONFIG=policy/low_latency/observation_config_gt_v4.yaml
export DEPLOY_SKIP_ENCODER=1
# Gold LOCKED 810demo CL was recorded without deploy RTC soft-blend.
# 810short T3 path must export USE_RTC=1 via run_810short_t3_closed_loop_eval.sh.
export USE_RTC="${USE_RTC:-0}"
# Passthrough: T3 / 810short set PHI0_VLM_REFRESH_EVERY_STEP=1 (default off in session).
export PHI0_VLM_REFRESH_EVERY_STEP="${PHI0_VLM_REFRESH_EVERY_STEP:-0}"
# Clean mp4: no HUD text / debug spheres on the recorded frame.
export GT_PANEL_LABELS="${GT_PANEL_LABELS:-0}"
# Episode instruction from dataset (closed-loop script default); override with PROMPT if needed.
export PROMPT="${PROMPT:-}"

mkdir -p "${WORK_DIR}"
echo "==> 810demo train-aligned closed-loop sim"
echo "    checkpoint=${CHECKPOINT}"
echo "    config=${CONFIG_NAME}"
echo "    episode=${EP} max_frames=${MAX_FRAMES} proprio=${PROPRIO_SOURCE} rtc=${USE_RTC} record_fps=${RECORD_FPS}"
echo "    PHI0_VLM_REFRESH_EVERY_STEP=${PHI0_VLM_REFRESH_EVERY_STEP}"
echo "    deploy_policy=${DEPLOY_POLICY_DIR:-low_latency}"
echo "    ego=${EGO_MP4}"
echo "    chest=${WRIST_MP4}"
echo "    out=${OUT_MP4}"

exec bash "${ROOT}/tools/eval/run_sonic_latent_sim_eval.sh"
