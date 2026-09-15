#!/usr/bin/env bash
# Default Newton-GL closed-loop viz for 820demo_skill_1_new (LL / low_latency student).
# Live sim proprio: body IL→MuJoCo remap (train tape order) + live Revo2.
#
#   STUDENT_CKPT=.../phi0_student_step020000.pt \
#   bash tools/eval/run_skill1_new_student_newton_viz.sh
#
# Optional: REF_START=58496 MAX_REF_FRAMES=600 NUM_STEPS=600 (skill2 segment)
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT}"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_skill_1_new_unified}"
REF_START="${REF_START:-0}"
MAX_REF_FRAMES="${MAX_REF_FRAMES:-477}"
NUM_STEPS="${NUM_STEPS:-${MAX_REF_FRAMES}}"
CONTROL="${CONTROL:-student}"
USE_VLM="${USE_VLM:-1}"
DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-low_latency}"
TAG="${TAG:-skill1_new_${CONTROL}_newton_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"

export STUDENT_CKPT REF_ROOT REF_START MAX_REF_FRAMES NUM_STEPS CONTROL TAG OUT
export USE_VLM PHI0_TRAIN_MODE=online_vlm TEACHER_Z_SOURCE=disk
export DEPLOY_POLICY_DIR
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-0}"
export HORIZON="${HORIZON:-32}"
export PHI0_CHUNK_EXEC="${PHI0_CHUNK_EXEC:-open_loop}"
# Newton G1 + BrainCo Revo2 composed USD (live hand joint_pos).
export PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2:-1}"

echo "[skill1_newton] ckpt=${STUDENT_CKPT}"
echo "[skill1_newton] ref=${REF_ROOT} start=${REF_START} steps=${NUM_STEPS} control=${CONTROL}"
echo "[skill1_newton] deploy=${DEPLOY_POLICY_DIR} use_vlm=${USE_VLM} revo2=${PHI0_NEWTON_REVO2} out=${OUT}"
echo "[skill1_newton] mp4 → ${OUT}/infer_${CONTROL}_newton_gl.mp4"

bash "${PHI0_ROOT}/tools/eval/run_qpos_student_newton_isaac_viz.sh"
