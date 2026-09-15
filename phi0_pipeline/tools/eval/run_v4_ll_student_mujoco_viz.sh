#!/usr/bin/env bash
# v4_ll student ẑ + hand_pred → MuJoCo Revo2 (low_latency decode).
# Not Newton-GL. Not efs replay_gt_latent.sh.
#
#   STUDENT_CKPT=.../phi0_student_step004000.pt \
#   bash tools/eval/run_v4_ll_student_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT}"
TAG="${TAG:-v4_ll_student_mujoco_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_v4_ll_unified}"
export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-low_latency}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
# H=32 RTC: delay=6, exec<=H-d
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"

mkdir -p "${OUT}"
echo "[v4-ll-viz] ckpt=${STUDENT_CKPT}"
echo "[v4-ll-viz] out=${OUT} H=${HORIZON} deploy=${DEPLOY_POLICY_DIR} ref=${REF}"

run_one() {
  local start="$1" n="$2" ep="$3" tag="$4"
  local SD="${OUT}/${tag}"
  mkdir -p "${SD}/logs"
  local INFER_NPZ="${SD}/infer_qpos_traj_student.npz"
  local TOKENS="${SD}/tokens.npz"
  local MOTION="${SD}/motion_v4.npz"
  local MP4="${SD}/gt_latent_revo2/student_sonic_revo2.mp4"
  # episode parquet: chunk-000/file-{ep:03d}.parquet for ep<1000
  local PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${ep}").parquet"

  echo "[v4-ll-viz] === ${tag} infer start=${start} frames=${n} ep=${ep} ==="
  PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}" \
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/offline_student_rtc_infer.py" \
    --ckpt "${STUDENT_CKPT}" \
    --ref-root "${REF}" \
    --ref-start "${start}" \
    --max-frames "${n}" \
    --horizon "${HORIZON}" \
    --device "cuda:0" \
    --out "${INFER_NPZ}" \
    >"${SD}/logs/offline_infer.log" 2>&1

  echo "[v4-ll-viz] === ${tag} pack ==="
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_gt_replay_tokens_from_infer_npz.py" \
    "${INFER_NPZ}" --out "${TOKENS}" --fps 50.0 \
    >"${SD}/logs/pack_tokens.log" 2>&1

  "${PHI0_PY}" - <<PY
import numpy as np
t = np.load("${TOKENS}")
assert "left_hand" in t.files and "right_hand" in t.files, t.files
np.savez(
    "${MOTION}",
    tokens=np.asarray(t["motion_token"], dtype=np.float32),
    left=np.asarray(t["left_hand"], dtype=np.float32),
    right=np.asarray(t["right_hand"], dtype=np.float32),
)
print("wrote ${MOTION}", np.load("${MOTION}")["tokens"].shape)
PY

  echo "[v4-ll-viz] === ${tag} MuJoCo Revo2 (low_latency) ==="
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
  PHI0_PY="${PHI0_PY}" \
  bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
    >"${SD}/logs/mujoco_replay.log" 2>&1

  echo "[v4-ll-viz] ${tag} mp4 → ${MP4}"
  ls -lh "${MP4}" || true
}

# ep0 / first of skill2 / first of skill3 in 820demo_v4_ll_unified
run_one 0 477 0 skill1_walk
run_one 58496 600 118 skill2_basket
run_one 334999 600 395 skill3_place

echo "[v4-ll-viz] DONE out=${OUT}"
ls -lh "${OUT}"/*/gt_latent_revo2/*.mp4 2>/dev/null || true
