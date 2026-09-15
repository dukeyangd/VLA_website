#!/usr/bin/env bash
# Gate: MuJoCo sonic latent GT replay of precomputed egypt onnx_g1 z*.
# Defaults: Phi_0_wpy subpackages deploy + REVO2_HAND=1 (not efs Dex3 gt_replay).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/tools/env/setup_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/tools/eval/_assert_no_efs_gt_latent.sh"
# Refuse accidental REPLAY_SH override to efs Dex3 stack.
if [[ -n "${REPLAY_SH:-}" ]]; then
  echo "[egypt-z-gt] ERROR: REPLAY_SH is set (${REPLAY_SH}); efs replay_gt_latent.sh is forbidden." >&2
  echo "[egypt-z-gt] unset REPLAY_SH; this script only uses run_sonic_latent_sim_eval.sh + subpackages." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/egypt_smplsem_clip}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/egypt_offline_zonly_${STAMP}}"
TOKENS="${TOKENS:-${OUT_DIR}/tokens_zstar.npz}"
MOTION_NPZ="${MOTION_NPZ:-${OUT_DIR}/motion_zstar_v4.npz}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
# Hard defaults for this gate (override only if you know why).
export REVO2_HAND="${REVO2_HAND:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
export DEPLOY_SKIP_ENCODER="${DEPLOY_SKIP_ENCODER:-1}"
export QPOS_RSI="${QPOS_RSI:-1}"
export ROBOT_ONLY="${ROBOT_ONLY:-1}"
export GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-robot}"
export HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES:-40}"

mkdir -p "${OUT_DIR}/logs"
if [[ ! -f "${TOKENS}" ]]; then
  echo "[egypt-z-gt] precompute z* + tokens → ${OUT_DIR}"
  "${PHI0_PY}" "${ROOT}/tools/data/precompute_egypt_zstar_onnx_g1.py" \
    --ref-root "${REF_ROOT}" --max-frames 278 --out-dir "${OUT_DIR}"
  TOKENS="${OUT_DIR}/tokens_zstar.npz"
fi

# MOTION_NPZ contract for replay_sonic_latent_npz_zmq_v4: tokens/left/right
"${PHI0_PY}" - <<PY
import numpy as np
from pathlib import Path
src = Path("${TOKENS}")
dst = Path("${MOTION_NPZ}")
d = np.load(src)
z = np.asarray(d["motion_token"] if "motion_token" in d.files else d["tokens"], dtype=np.float32)
t = int(z.shape[0])
left = np.asarray(d["left_hand"], dtype=np.float32) if "left_hand" in d.files else np.zeros((t, 7), np.float32)
right = np.asarray(d["right_hand"], dtype=np.float32) if "right_hand" in d.files else np.zeros((t, 7), np.float32)
if left.shape[0] != t:
    left = np.zeros((t, 7), np.float32)
if right.shape[0] != t:
    right = np.zeros((t, 7), np.float32)
np.savez(dst, tokens=z, left=left, right=right)
print(f"[egypt-z-gt] motion_npz T={t} -> {dst}", flush=True)
PY

WORK_DIR="${WORK_DIR:-${OUT_DIR}/gt_latent_revo2_${STAMP}}"
# Refuse stale WORK_DIR/OUT_MP4 from prior 820mix shells.
case "${WORK_DIR}" in
  *egypt_offline_zonly*|*"${OUT_DIR}"*) ;;
  *)
    echo "[egypt-z-gt] ignoring stale WORK_DIR=${WORK_DIR}; using under OUT_DIR" >&2
    WORK_DIR="${OUT_DIR}/gt_latent_revo2_${STAMP}"
    ;;
esac
mkdir -p "${WORK_DIR}/logs"
PARQUET="${PARQUET:-${REF_ROOT}/data/chunk-000/file-000.parquet}"
RSI_NPZ="${WORK_DIR}/qpos_rsi_frame0.npz"
LOG="${LOG:-${ROOT}/logs/egypt_zstar_gt_revo2_${STAMP}.log}"
H264="${H264_OUT:-${ROOT}/logs/egypt_zstar_gt_revo2_${STAMP}_h264.mp4}"
OUT_MP4="${WORK_DIR}/gt_sonic_replay.mp4"
export GT_PANEL_LAYOUT=robot
export ROBOT_ONLY=1

"${PHI0_PY}" "${ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
  "${PARQUET}" --out "${RSI_NPZ}" --frame 0 \
  >"${WORK_DIR}/logs/export_qpos_rsi.log" 2>&1

echo "[egypt-z-gt] GR00T_ROOT=${GR00T_ROOT}"
echo "[egypt-z-gt] GEAR_SONIC_DEPLOY=${GEAR_SONIC_DEPLOY}"
echo "[egypt-z-gt] REVO2_HAND=${REVO2_HAND} policy=${DEPLOY_POLICY_DIR}"
echo "[egypt-z-gt] motion=${MOTION_NPZ}"
echo "[egypt-z-gt] work=${WORK_DIR}"
echo "[egypt-z-gt] video → ${WORK_DIR}/gt_sonic_replay.mp4"
echo "[egypt-z-gt] log=${LOG}"

# Foreground so caller can nohup; uses in-tree subpackages + Revo2 scene.
# Clear stale OUT_MP4/WORK_DIR from other eval shells.
env -u CHECKPOINT -u USE_CLOSED_LOOP -u CONFIG_NAME -u OUT_MP4 -u H264_OUT \
  UNIFIED_ROOT="${REF_ROOT}" UNIFIED_EP=0 \
  UNIFIED_PARQUET="${PARQUET}" \
  MOTION_NPZ="${MOTION_NPZ}" \
  WORK_DIR="${WORK_DIR}" OUT_MP4="${OUT_MP4}" \
  MAX_FRAMES=278 MOTION_SECONDS=0 CONTROL_FPS=50 RECORD_FPS=50 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR}" \
  DEPLOY_SKIP_ENCODER="${DEPLOY_SKIP_ENCODER}" \
  QPOS_RSI=1 QPOS_RSI_NPZ="${RSI_NPZ}" \
  REVO2_HAND="${REVO2_HAND}" \
  HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES}" \
  GT_PANEL_LAYOUT=robot GT_PANEL_LABELS=0 \
  ENABLE_G1_DEBUG_OVERLAY=0 ROBOT_ONLY=1 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  bash "${ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
  2>&1 | tee "${LOG}"

echo "[egypt-z-gt] done video=${WORK_DIR}/gt_sonic_replay.mp4"
ls -lh "${WORK_DIR}/gt_sonic_replay.mp4" 2>/dev/null || true
# best-effort remux copy for chat click
if [[ -f "${WORK_DIR}/gt_sonic_replay.mp4" ]]; then
  cp -f "${WORK_DIR}/gt_sonic_replay.mp4" "${H264}" || true
  echo "[egypt-z-gt] h264_copy=${H264}"
fi
