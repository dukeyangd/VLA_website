#!/usr/bin/env bash
# 810-final step30k train-aligned closed-loop full ep0.
# Env:
#   PROPRIO_SOURCE=robot|gt|robot_gt_hand   (default robot)
#   CHECKPOINT / UNIFIED_ROOT / REF_ROOT / CUDA_DEVICES / COPY_H264_TO_ARTIFACTS=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

STAMP=$(date +%Y%m%d_%H%M%S)
PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
case "$PROPRIO_SOURCE" in
  robot|gt|robot_gt_hand) ;;
  *) echo "PROPRIO_SOURCE must be robot|gt|robot_gt_hand" >&2; exit 1 ;;
esac

TAG="full"
[[ "$PROPRIO_SOURCE" == "gt" ]] && TAG="full_gtproprio"
[[ "$PROPRIO_SOURCE" == "robot_gt_hand" ]] && TAG="full_robot_gt_hand"

CKPT="${CHECKPOINT:-${PHI0_ROOT}/experiments/810-final/phi0_step30000.pt}"
REF="${UNIFIED_ROOT:-${REF_ROOT:-${PHI0_WORKSPACE}/810short_horizon_v2_egypt_layout}}"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/experiments/810-final_cl_ep0_${TAG}_${STAMP}}"
LOG_DIR="${LOG_DIR:-${PHI0_ROOT}/experiments/logs}"
mkdir -p "$WORK_DIR/logs" "$LOG_DIR"
LOG="${LOG_DIR}/vla_810final_cl_ep0_${TAG}_${STAMP}.log"
H264="${H264_OUT:-${PHI0_ROOT}/docs/artifacts/lab_logs/videos/vla_810final_cl_ep0_${TAG}_h264.mp4}"
mkdir -p "$(dirname "$H264")"

PY="${PHI0_PY:-python}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing ckpt: $CKPT" >&2
  exit 1
fi
if [[ ! -f "${REF}/data/chunk-000/file-000.parquet" ]]; then
  echo "ERROR: missing ref parquet under UNIFIED_ROOT/REF_ROOT=$REF" >&2
  exit 1
fi

RSI_NPZ="${WORK_DIR}/qpos_rsi_frame0.npz"
"$PY" "${PHI0_ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
  "${REF}/data/chunk-000/file-000.parquet" \
  --out "$RSI_NPZ" --frame 0 --episode-index 0 \
  >"${WORK_DIR}/logs/export_qpos_rsi.log" 2>&1

{
  echo "ckpt=$CKPT step=30000 proprio=$PROPRIO_SOURCE"
  echo "align: CONFIG=train_810short_egypt_vlm_offset_sonic_revo2"
  echo "align: PHI0_ADALN_MODE=progress_vlm_age PHI0_ACTION_CROSS_ATTN_MODE=interleave_vlm"
  echo "align: PHI0_VLM_REFRESH_EVERY_STEP=1 USE_RTC=1 QPOS_RSI=1 full ep0"
  echo "work=$WORK_DIR h264=$H264"
} | tee "$LOG.launch"

nohup env \
  CHECKPOINT="$CKPT" \
  CONFIG_NAME=train_810short_egypt_vlm_offset_sonic_revo2 \
  PHI0_ADALN_MODE=progress_vlm_age \
  PHI0_ACTION_CROSS_ATTN_MODE=interleave_vlm \
  PHI0_VLM_REFRESH_EVERY_STEP=1 \
  USE_RTC=1 \
  USE_CLOSED_LOOP=1 \
  PROPRIO_SOURCE="$PROPRIO_SOURCE" \
  UNIFIED_ROOT="$REF" \
  UNIFIED_EP=0 \
  GT_REPO_ID=810short_horizon_v2_egypt_layout \
  MAX_FRAMES=0 \
  MOTION_SECONDS=0 \
  CONTROL_FPS=50 \
  RECORD_FPS=50 \
  CUDA_DEVICES="${CUDA_DEVICES:-0}" \
  WORK_DIR="$WORK_DIR" \
  OUT_MP4="${WORK_DIR}/closed_loop_gt.mp4" \
  QPOS_RSI=1 \
  QPOS_RSI_NPZ="$RSI_NPZ" \
  HAND_RAMP_FRAMES=40 \
  GT_PANEL_LAYOUT=top \
  GT_PANEL_LABELS=0 \
  ENABLE_G1_DEBUG_OVERLAY=0 \
  ROBOT_ONLY=0 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  bash "${PHI0_ROOT}/tools/eval/run_810demo_closed_loop_eval.sh" 0 0 \
  >"$LOG" 2>&1 &
echo $! | tee "$WORK_DIR/main.pid"
echo "log=$LOG"

OUT_MP4_NAME=closed_loop_gt.mp4 \
  nohup bash "${PHI0_ROOT}/tools/eval/remux_gt_sonic_when_done.sh" "$WORK_DIR" "$LOG" "$H264" \
  >"$WORK_DIR/logs/remux_watcher.log" 2>&1 &
echo "remux_watcher=$!"
