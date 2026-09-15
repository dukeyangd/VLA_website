#!/usr/bin/env bash
# Student ẑ → in-tree Revo2 MuJoCo for 820 demo5skill (text-only).
# Not Newton-GL. Not efs replay_gt_latent.sh.
# Prompt: episode instruction from 820demo_demo5skill_release_unified
# (DEMO5_PROMPTS short Chinese). Refuses leftover long Chinese or English TASK tags.
#
#   STUDENT_CKPT=.../phi0_student_last.pt bash tools/eval/run_student_demo5_mujoco_viz.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT}"
TAG="${TAG:-student_demo5_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
REF="${REF:-/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_release_unified}"
export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"

mkdir -p "${OUT}"
echo "[demo5] ckpt=${STUDENT_CKPT}"
echo "[demo5] out=${OUT} H=${HORIZON} ref=${REF}"

# Concatenated 5-ep tape (idle/egypt/spin/wave/bow). Same offsets as mix ep0–4.
SKILLS=(
  "file-000.parquet idle 0 0 474 静止假人"
  "file-001.parquet egypt 1 474 279 埃及肚皮舞"
  "file-002.parquet spin 2 753 363 华丽单脚转"
  "file-003.parquet wave 3 1116 264 告别挥手"
  "file-004.parquet bow 4 1380 243 正式鞠躬"
)

run_one() {
  local file="$1" tag="$2" ep="$3" start="$4" n="$5" prefix="$6"
  local SD="${OUT}/demo5/${tag}"
  mkdir -p "${SD}/logs"
  local INFER_NPZ="${SD}/infer_qpos_traj_student.npz"
  local TOKENS="${SD}/tokens.npz"
  local MOTION="${SD}/motion_v4.npz"
  local MP4="${SD}/gt_latent_revo2/student_sonic_revo2.mp4"
  local PARQUET="${REF}/data/chunk-000/${file}"

  echo "[demo5] === ${tag} offline infer start=${start} frames=${n} ==="
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

  if ! grep -q "prompt0=${prefix}" "${SD}/logs/offline_infer.log"; then
    echo "[demo5] FAIL ${tag}: expected prompt0=${prefix} (short Chinese)"
    grep -E 'prompt0=|no mp4' "${SD}/logs/offline_infer.log" | tail -5 || true
    return 1
  fi

  echo "[demo5] === ${tag} pack ==="
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

  echo "[demo5] === ${tag} MuJoCo Revo2 replay ==="
  WORK_DIR="${SD}/gt_latent_revo2" \
  OUT_MP4="${MP4}" \
  MOTION_NPZ="${MOTION}" \
  UNIFIED_ROOT="${REF}" \
  UNIFIED_PARQUET="${PARQUET}" \
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

  echo "[demo5] ${tag} mp4 → ${MP4}"
  ls -lh "${MP4}" || true
}

for row in "${SKILLS[@]}"; do
  read -r file tag ep start n prefix <<<"${row}"
  run_one "${file}" "${tag}" "${ep}" "${start}" "${n}" "${prefix}"
done

echo "[demo5] DONE out=${OUT}"
ls -lh "${OUT}"/demo5/*/gt_latent_revo2/*.mp4 2>/dev/null || true
