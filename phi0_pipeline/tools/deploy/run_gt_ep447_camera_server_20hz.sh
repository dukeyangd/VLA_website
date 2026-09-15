#!/usr/bin/env bash
# pick_yellow_box_valid GT ego video -> ZMQ camera server @127.0.0.1:5555 @ 20 Hz.
#
# Terminal 1:
#   bash tools/deploy/run_gt_ep447_camera_server_20hz.sh
#   EPISODE=18 bash tools/deploy/run_gt_ep447_camera_server_20hz.sh
#   bash tools/deploy/run_gt_ep447_camera_server_20hz.sh 18
#
# Terminal 2 (closed loop @ 20 Hz — match CONTROL_FPS):
#   bash tools/eval/run_pick_tissue_rtc_sim_closed_loop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PORT="${PORT:-5555}"
export CONTROL_FPS="${CONTROL_FPS:-20}"
export GT_ROOT="${GT_ROOT:-${HOME}/Isaac-GR00T/data}"
DATASET="${DATASET:-pick_yellow_box_valid}"
EPISODE="${EPISODE:-18}"

if [[ "${1:-}" == *.mp4 ]]; then
  EGO_MP4="$1"
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  EPISODE="$1"
  chunk=$((EPISODE / 1000))
  EGO_MP4="${GT_ROOT}/${DATASET}/videos/chunk-$(printf '%03d' "$chunk")/observation.images.ego_view/episode_$(printf '%06d' "$EPISODE").mp4"
else
  chunk=$((EPISODE / 1000))
  EGO_MP4="${GT_ROOT}/${DATASET}/videos/chunk-$(printf '%03d' "$chunk")/observation.images.ego_view/episode_$(printf '%06d' "$EPISODE").mp4"
fi

if [[ ! -f "${EGO_MP4}" ]]; then
  echo "[gt_camera] missing ego mp4: ${EGO_MP4}" >&2
  exit 1
fi

echo "[gt_camera] dataset=${DATASET} episode=${EPISODE} mp4=${EGO_MP4}"

exec bash "${SCRIPT_DIR}/run_gt_video_camera_server.sh" 0 \
  --ego-mp4 "${EGO_MP4}" --no-preload
