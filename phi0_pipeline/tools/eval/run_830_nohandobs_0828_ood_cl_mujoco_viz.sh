#!/usr/bin/env bash
# nohandobs ckpt × 0828 skill2 OOD vision (dataset_video) × body=sim.
# Hand proprio zeroed to match train PHI0_ZERO_PROPRIO_HAND=1.
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830demo_skill2_pico_pick_toy_nohandobs_vlm_cache_h32_b32_ddp6_e10_20260904_031525/phi0_student_last.pt}"
export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/0828_skill2_unified}"
export EP="${EP:-0}"
export HORIZON="${HORIZON:-32}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export PHI0_HAND_MODE=dex3
export PHI0_NEWTON_REVO2=0
# pick_toy / pico：真 WBC 序（t3_final §0）；勿用旧 teleop ORDER=0
export PHI0_DEX3_HAND_POLICY_ORDER="${PHI0_DEX3_HAND_POLICY_ORDER:-1}"
export PHI0_DEX3_SCALE_TO_UNITREE="${PHI0_DEX3_SCALE_TO_UNITREE:-0}"
export TAG="${TAG:-830_nohandobs_0828_ood_cl_ep${EP}_$(date +%Y%m%d_%H%M%S)}"

export PHI0_USE_VLM_FRAME_LATENT_CACHE=0
export PHI0_CL_VLM_SOURCE=dataset_video
export PHI0_VLM_ENCODE_MIN_BATCH="${PHI0_VLM_ENCODE_MIN_BATCH:-2}"
export USE_VLM=1
export PHI0_ZERO_PROPRIO_HAND=1
export PHI0_CL_PROPRIO_BODY=live
export PHI0_CL_HAND_OBS="${PHI0_CL_HAND_OBS:-commanded}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
export MAX_FRAMES="${MAX_FRAMES:-1500}"
# live VLM replan can exceed default 0.4s between frame_index ticks
export GT_ZMQ_RECORD_STALL_S="${GT_ZMQ_RECORD_STALL_S:-30}"
export PROMPT="${PROMPT:-抓起黄色玩具放到篮子里。}"
export TASK_PROMPT="${TASK_PROMPT:-${PROMPT}}"

test -f "${REF_ROOT}/meta/stats.json"
test -d "${REF_ROOT}/videos"
test -f "${STUDENT_CKPT}"

echo "[nohandobs_0828] ckpt=${STUDENT_CKPT}"
echo "[nohandobs_0828] REF=${REF_ROOT} EP=${EP} vlm=dataset_video body=live zero_hand=1 order=${PHI0_DEX3_HAND_POLICY_ORDER} scale=${PHI0_DEX3_SCALE_TO_UNITREE}"
exec bash "${PHI0_ROOT}/tools/eval/run_830_walk_student_cl_mujoco_viz.sh"
