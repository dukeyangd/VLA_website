#!/usr/bin/env bash
# Sim closed loop: local camera @127.0.0.1:5555 + g1_debug bootstrap + token roll-forward body.
#   IN  camera  tcp://127.0.0.1:5555  (local GT video server or composed_camera)
#   IN  proprio tcp://127.0.0.1:5557  (first frame only, then model roll-forward)
#   OUT tokens tcp://127.0.0.1:5556
#
# Terminal 1 — local GT ep447 camera @20Hz (or any camera publisher on :5555):
#   bash tools/deploy/run_gt_ep447_camera_server_20hz.sh
#
# Terminal 2 — deploy + keyboard (k -> p -> i -> p before bootstrap can read g1_debug):
#   g1_deploy_onnx_ref ... --zmq-port 5556
#   python tools/deploy/publish_deploy_keyboard_zmq.py
#
# Terminal 3:
#   bash tools/eval/run_pick_tissue_rtc_sim_closed_loop_local_bootstrap.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

export CAMERA_SOURCE="${CAMERA_SOURCE:-live}"
export CAMERA_HOST="${CAMERA_HOST:-127.0.0.1}"
export CAMERA_PORT="${CAMERA_PORT:-5555}"
export CONTROL_FPS="${CONTROL_FPS:-20}"
export PROPRIO_SOURCE="${PROPRIO_SOURCE:-bootstrap-roll-forward}"
export CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_valid_wholebody_rtc_b256_30k_deploy_align/pick_tissue_valid_wholebody_rtc_b256_act_step20000.pt}"
export CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_finetune_rtc_b256_30k}"

exec bash "${SCRIPT_DIR}/run_pick_tissue_rtc_live_closed_loop.sh" "$@"
