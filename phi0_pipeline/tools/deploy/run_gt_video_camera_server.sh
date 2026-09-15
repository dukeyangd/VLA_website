#!/usr/bin/env bash
# GT dataset video -> ZMQ camera server @5555 (ComposedCameraClientSensor protocol).
#
# Usage:
#   bash tools/deploy/run_gt_video_camera_server.sh
#   bash tools/deploy/run_gt_video_camera_server.sh 447
#   EPISODE=447 PORT=5555 bash tools/deploy/run_gt_video_camera_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"
cd "${PHI0_ROOT}"

EPISODE="${1:-447}"
shift || true
PORT="${PORT:-5555}"
CONTROL_FPS="${CONTROL_FPS:-50}"
REPO_ID="${REPO_ID:-pick_tissue_xperience_unified}"
GT_ROOT="${GT_ROOT:-${PHI0_WORKSPACE}/Isaac-GR00T/data}"

echo "[gt_camera] episode=${EPISODE} repo=${REPO_ID} port=${PORT} fps=${CONTROL_FPS}"

exec "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/publish_gt_video_zmq_camera.py" \
  --episode "${EPISODE}" \
  --repo-id "${REPO_ID}" \
  --root "${GT_ROOT}" \
  --port "${PORT}" \
  --control-fps "${CONTROL_FPS}" \
  "$@"
