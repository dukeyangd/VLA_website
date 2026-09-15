#!/usr/bin/env bash
# Live RTC closed loop: remote camera + deploy g1_debug only (no local GT).
#   IN  camera  tcp://CAMERA_HOST:5555
#   IN  proprio tcp://STATE_ZMQ_HOST:5557  (g1_debug)
#   IN  keyboard tcp://KEYBOARD_ZMQ_HOST:5580  (VLA-style publisher pane)
#   OUT tokens tcp://ZMQ_HOST:5556
#
# Workflow (separate keyboard pane — publish_deploy_keyboard_zmq.py):
#   1. Start deploy + camera server externally
#   2. Run keyboard publisher: python tools/deploy/publish_deploy_keyboard_zmq.py
#   3. Run this script — loads model, binds ZMQ, waits
#   4. Press k  -> start C++ control loop (planner)
#   5. Press p  -> resume Phi-0 inference
#   6. Press i  -> initial pose + POSE mode
#   7. Press p  -> resume token streaming to deploy
#
# Logs io_rates every IO_LOG_INTERVAL_S (camera / g1_debug / zmq_out Hz).
# Writes outputs.npz + observations.npz when RECORD_DIR is set (default: auto).
#
# Usage:
#   bash tools/eval/run_pick_tissue_rtc_live_closed_loop.sh
#   CAMERA_HOST=192.168.123.165 bash tools/eval/run_pick_tissue_rtc_live_closed_loop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

export CAMERA_SOURCE="${CAMERA_SOURCE:-live}"
export CAMERA_HOST="${CAMERA_HOST:-192.168.123.165}"
export CAMERA_PORT="${CAMERA_PORT:-5555}"
export PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
export STATE_ZMQ_HOST="${STATE_ZMQ_HOST:-127.0.0.1}"
export STATE_ZMQ_PORT="${STATE_ZMQ_PORT:-5557}"
export ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
export ZMQ_PORT="${ZMQ_PORT:-5556}"
export CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_valid_wholebody_rtc_b256_30k_deploy_align/pick_tissue_valid_wholebody_rtc_b256_act_latest.pt}"
export CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_finetune_rtc_b256_30k}"
export PROMPT="${PROMPT:-pick tissue}"
export INFERENCE_RATE="${INFERENCE_RATE:-0}"
export STREAM_NOW="${STREAM_NOW:-0}"
export DEPLOY_KEYBOARD="${DEPLOY_KEYBOARD:-0}"
export KEYBOARD_ZMQ_HOST="${KEYBOARD_ZMQ_HOST:-127.0.0.1}"
export KEYBOARD_ZMQ_PORT="${KEYBOARD_ZMQ_PORT:-5580}"
export WAIT_DEPLOY_P="${WAIT_DEPLOY_P:-1}"
export WAIT_ROBOT_PROPRIO="${WAIT_ROBOT_PROPRIO:-0}"
export IO_LOG_INTERVAL_S="${IO_LOG_INTERVAL_S:-2.0}"
export RECORD_DIR="${RECORD_DIR:-${PHI0_ROOT}/logs/live_closed_loop_$(date +%Y%m%d_%H%M%S)}"

EXTRA_ARGS=()
if [[ "${WAIT_DEPLOY_P}" == "1" ]]; then
  EXTRA_ARGS+=(--wait-deploy-p)
fi
if [[ "${DEPLOY_KEYBOARD}" == "1" ]]; then
  EXTRA_ARGS+=(--deploy-keyboard)
else
  EXTRA_ARGS+=(--no-deploy-keyboard)
fi
if [[ "${KEYBOARD_ZMQ_PORT}" != "0" ]]; then
  EXTRA_ARGS+=(--keyboard-zmq-host "${KEYBOARD_ZMQ_HOST}" --keyboard-zmq-port "${KEYBOARD_ZMQ_PORT}")
fi

exec bash "${SCRIPT_DIR}/run_phi0_sonic_closed_loop.sh" \
  --no-gt-fallback \
  --io-log-interval-s "${IO_LOG_INTERVAL_S}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
