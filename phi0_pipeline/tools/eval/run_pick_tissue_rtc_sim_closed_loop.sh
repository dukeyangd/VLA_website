#!/usr/bin/env bash
# Sim closed loop: remote SONIC camera + local deploy g1_debug @5557.
#   IN  camera  tcp://CAMERA_HOST:5555  (default robot 192.168.123.165)
#   IN  proprio tcp://127.0.0.1:5557
#
# Default checkpoint: wholebody RTC b64@8k deploy_align (no body root in proprio).
# Checkpoint cfg merges deploy_align_proprio_prefix + unified_supervision_dataset
# g1_sonic_deploy:
#   proprio prefix [512-d input]: gripper14 [346:360] + body dof29 [367:396]
#   model output → ZMQ: above + sonic motion_token64 [396:460] (64-d forward motion)
#   zeroed: SMPL [0:346], root xyz/quat [360:367], reserved [460:512]
#
# Prerequisite: composed_camera running on the robot/host at :5555
#   (no local GT video server needed)
#
# Local GT video instead:
#   CAMERA_HOST=127.0.0.1 CONTROL_FPS=20 bash tools/deploy/run_gt_ep447_camera_server_20hz.sh
#   CAMERA_HOST=127.0.0.1 CONTROL_FPS=20 bash tools/eval/run_pick_tissue_rtc_sim_closed_loop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

export CAMERA_SOURCE="${CAMERA_SOURCE:-live}"
export CAMERA_HOST="${CAMERA_HOST:-192.168.123.165}"
export CAMERA_PORT="${CAMERA_PORT:-5555}"
export CONTROL_FPS="${CONTROL_FPS:-50}"
export PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
export CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_32k_ddp4/pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_act_latest.pt}"
export CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_finetune_rtc_b256_30k}"
# yellow_box ckpt needs matching config:
#   CHECKPOINT=.../pick_yellow_box_..._act_latest.pt \
#   CONFIG_NAME=train_pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_ddp4_32k PROMPT="pick yellow box" \
#   bash tools/eval/run_pick_tissue_rtc_sim_closed_loop.sh

exec bash "${SCRIPT_DIR}/run_pick_tissue_rtc_live_closed_loop.sh" "$@"
