#!/usr/bin/env bash
# v4_ll ChunkStudent closed-loop infer → **Newton-GL** (default) or optional MuJoCo pack.
# NOT offline_student_rtc_infer (tape FK body).
#
# Stage 1 (default deliverable): newton_qpos_student_isaac_viz CONTROL=student --use_vlm
#   - body29 IL→MuJoCo remap + revo2_12 from sim (not dataset tape)
#   - dual VLM from episode mp4 synced to frame
# Stage 2–3 (optional): pack tokens → MuJoCo Revo2 replay when SKIP_MUJOCO=0
#
#   STUDENT_CKPT=.../phi0_student_step230000.pt \
#   bash tools/eval/run_v4_ll_student_cl_mujoco_viz.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"
# Newton infer must use Phi-0-wbc-newton-wpy (setup_env defaults Phi-0-wpy).
NEWTON_PY="/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python"
# MuJoCo deploy stack (tyro, etc.) uses Phi-0-wpy from setup_env.
MUJOCO_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
PHI0_PY="${NEWTON_PY}"

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT}"
TAG="${TAG:-v4_ll_student_cl_mujoco_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_v4_ll_unified}"
export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-low_latency}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_TRAIN_MODE=online_vlm
export USE_VLM=1
export TEACHER_Z_SOURCE=disk
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX=0
export SIMULATOR="${SIMULATOR:-newton}"
export NUM_ENVS=1
SKIP_MUJOCO="${SKIP_MUJOCO:-1}"

mkdir -p "${OUT}"
echo "[v4-ll-cl] ckpt=${STUDENT_CKPT}"
echo "[v4-ll-cl] out=${OUT} H=${HORIZON} deploy=${DEPLOY_POLICY_DIR} ref=${REF}"
echo "[v4-ll-cl] infer: Newton-GL default; skip_mujoco=${SKIP_MUJOCO}"
echo "[v4-ll-cl] live sim proprio: body IL→MuJoCo + GT dual camera (not tape FK body)"

run_one() {
  local start="$1" n="$2" ep="$3" tag="$4"
  local SD="${OUT}/${tag}"
  mkdir -p "${SD}/logs"
  local INFER_OUT="${SD}/isaac_infer"
  local INFER_NPZ="${INFER_OUT}/infer_qpos_traj_student.npz"
  local NEWTON_MP4="${INFER_OUT}/infer_student_newton_gl.mp4"
  local TOKENS="${SD}/tokens.npz"
  local MOTION="${SD}/motion_v4.npz"
  local MP4="${SD}/gt_latent_revo2/student_sonic_revo2.mp4"
  local PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${ep}").parquet"

  echo "[v4-ll-cl] === ${tag} Newton CL infer start=${start} frames=${n} ep=${ep} ==="
  mkdir -p "${INFER_OUT}"
  # Kit AppLauncher (eval_agent_trl) breaks on Newton; use launch_simulation path
  # (newton_qpos_student_isaac_viz → run_online_infer_eval, live sim proprio).
  PHI0_ISAAC_BACKEND=newton \
  PHI0_TRAIN_MODE=online_vlm \
  USE_VLM=1 \
  TEACHER_Z_SOURCE=disk \
  PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}" \
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/newton_qpos_student_isaac_viz.py" \
    --student_ckpt "${STUDENT_CKPT}" \
    --ref_root "${REF}" \
    --ref_start "${start}" \
    --max_ref_frames "${n}" \
    --num_steps "${n}" \
    --horizon "${HORIZON}" \
    --control student \
    --use_vlm \
    --out_dir "${INFER_OUT}" \
    >"${SD}/logs/isaac_infer.log" 2>&1

  if [[ ! -f "${INFER_NPZ}" ]]; then
    echo "[v4-ll-cl] missing ${INFER_NPZ}" >&2
    tail -40 "${SD}/logs/isaac_infer.log" >&2 || true
    exit 1
  fi
  echo "[v4-ll-cl] ${tag} Newton mp4 → ${NEWTON_MP4}"
  ls -lh "${NEWTON_MP4}" 2>/dev/null || ls -lh "${INFER_OUT}/infer_"*"_newton_gl.mp4" 2>/dev/null || true

  if [[ "${SKIP_MUJOCO}" == "1" ]]; then
    echo "[v4-ll-cl] ${tag} SKIP_MUJOCO=1 (Newton-GL is primary deliverable)"
    return 0
  fi

  echo "[v4-ll-cl] === ${tag} pack (optional MuJoCo对照) ==="
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_gt_replay_tokens_from_infer_npz.py" \
    "${INFER_NPZ}" --out "${TOKENS}" --fps 50.0 \
    >"${SD}/logs/pack_tokens.log" 2>&1

  "${PHI0_PY}" - <<PY
import numpy as np
t = np.load("${TOKENS}")
np.savez(
    "${MOTION}",
    tokens=np.asarray(t["motion_token"], dtype=np.float32),
    left=np.asarray(t["left_hand"], dtype=np.float32),
    right=np.asarray(t["right_hand"], dtype=np.float32),
)
print("wrote ${MOTION}", np.load("${MOTION}")["tokens"].shape)
PY

  echo "[v4-ll-cl] === ${tag} MuJoCo replay (CL trajectory) ==="
  WORK_DIR="${SD}/gt_latent_revo2" \
  OUT_MP4="${MP4}" \
  MOTION_NPZ="${MOTION}" \
  UNIFIED_ROOT="${REF}" \
  UNIFIED_PARQUET="${PARQUET}" \
  UNIFIED_EP="${ep}" \
  QPOS_RSI=1 \
  DEPLOY_POLICY_DIR=low_latency \
  DEPLOY_OBS_CONFIG=policy/low_latency/observation_config.yaml \
  DEPLOY_SKIP_ENCODER=1 \
  REVO2_HAND=1 \
  ROBOT_ONLY=1 \
  GT_PANEL_LAYOUT=robot \
  ENABLE_G1_DEBUG_OVERLAY=0 \
  HAND_RAMP_FRAMES=0 \
  MAX_FRAMES="${n}" \
  MOTION_SECONDS=0 \
  CONTROL_FPS=50 \
  RECORD_FPS=50 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PHI0_PY="${MUJOCO_PY}" \
  bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
    >"${SD}/logs/mujoco_replay.log" 2>&1

  echo "[v4-ll-cl] ${tag} mp4 → ${MP4}"
  ls -lh "${MP4}" || true
}

run_one 0 477 0 skill1_walk
run_one 58496 600 118 skill2_basket
run_one 334999 600 395 skill3_place

echo "[v4-ll-cl] DONE out=${OUT}"
ls -lh "${OUT}"/*/isaac_infer/infer_*_newton_gl.mp4 2>/dev/null || true
if [[ "${SKIP_MUJOCO}" != "1" ]]; then
  ls -lh "${OUT}"/*/gt_latent_revo2/*.mp4 2>/dev/null || true
fi
