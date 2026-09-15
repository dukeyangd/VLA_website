#!/usr/bin/env bash
# MuJoCo viz for stand+egypt+skill2 smoke ckpt. Not Newton. Not efs replay_gt_latent.
#
#   STUDENT_CKPT=.../phi0_student_last.pt bash tools/eval/run_stand_egypt_skill2_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT}"
TAG="${TAG:-stand_egypt_s2_viz_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"

mkdir -p "${OUT}"
echo "[s2-viz] ckpt=${STUDENT_CKPT}"
echo "[s2-viz] out=${OUT} H=${HORIZON}"

run_one() {
  local ref="$1" start="$2" n="$3" parquet="$4" ep="$5" tag="$6"
  local SD="${OUT}/${tag}"
  mkdir -p "${SD}/logs"
  local INFER_NPZ="${SD}/infer_qpos_traj_student.npz"
  local TOKENS="${SD}/tokens.npz"
  local MOTION="${SD}/motion_v4.npz"
  local MP4="${SD}/gt_latent_revo2/student_sonic_revo2.mp4"

  echo "[s2-viz] === ${tag} infer start=${start} frames=${n} ==="
  PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}" \
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/offline_student_rtc_infer.py" \
    --ckpt "${STUDENT_CKPT}" \
    --ref-root "${ref}" \
    --ref-start "${start}" \
    --max-frames "${n}" \
    --horizon "${HORIZON}" \
    --device "cuda:0" \
    --out "${INFER_NPZ}" \
    >"${SD}/logs/offline_infer.log" 2>&1

  echo "[s2-viz] === ${tag} pack ==="
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

  echo "[s2-viz] === ${tag} MuJoCo Revo2 ==="
  WORK_DIR="${SD}/gt_latent_revo2" \
  OUT_MP4="${MP4}" \
  MOTION_NPZ="${MOTION}" \
  UNIFIED_ROOT="${ref}" \
  UNIFIED_PARQUET="${parquet}" \
  UNIFIED_EP="${ep}" \
  QPOS_RSI=1 \
  DEPLOY_POLICY_DIR=release \
  DEPLOY_OBS_CONFIG=policy/release/observation_config.yaml \
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
  PHI0_PY="${PHI0_PY}" \
  bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
    >"${SD}/logs/mujoco_replay.log" 2>&1

  echo "[s2-viz] ${tag} mp4 → ${MP4}"
  ls -lh "${MP4}" || true
}

D5="/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_release_unified"
S2="/mnt/data2/wpy/workspace/820demo/820demo_skill_2_release_unified"

run_one "${D5}" 0 474 "${D5}/data/chunk-000/file-000.parquet" 0 stand
run_one "${D5}" 474 279 "${D5}/data/chunk-000/file-001.parquet" 1 egypt
run_one "${S2}" 0 1028 "${S2}/data/chunk-000/file-000.parquet" 0 skill2

echo "[s2-viz] DONE out=${OUT}"
ls -lh "${OUT}"/*/gt_latent_revo2/*.mp4 2>/dev/null || true
