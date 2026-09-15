#!/usr/bin/env bash
# 830 walk student ẑ + hand_pred → MuJoCo Dex3 + sonic_v1_1 decode.
# **Open-loop** body proprio from tape FK (offline_student_rtc_infer).
# For sim g1_debug closed-loop use run_830_walk_student_cl_mujoco_viz.sh.
#
#   EP=125 bash tools/eval/run_830_walk_student_mujoco_viz.sh
#   STUDENT_CKPT=.../phi0_student_step032960.pt EP=125 bash tools/eval/run_830_walk_student_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830_walk_nosim_h14_b32_ddp8_e10_20260826_115006/phi0_student_last.pt}"
EP="${EP:-125}"
TAG="${TAG:-830_walk_student_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"
VALID_PARQUET="${VALID_HAND_ROOT:-/mnt/data2/wpy/workspace/Isaac-GR00T/data/skill_walk_to_black_box_new_v1_1_dex3_obs_hand_valid}/data/chunk-000/episode_$(printf '%06d' "${EP}").parquet"

export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_HAND_MODE="${PHI0_HAND_MODE:-dex3}"
export PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2:-0}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "[830_walk_student] missing ckpt ${STUDENT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[830_walk_student] missing parquet ${PARQUET}" >&2
  exit 1
fi

NFRAMES="${MAX_FRAMES:-$("${PHI0_PY}" - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("${PARQUET}").num_rows)
PY
)}"
REF_START="${REF_START:-$("${PHI0_PY}" - <<PY
import pyarrow.parquet as pq
from pathlib import Path
root = Path("${REF}") / "data/chunk-000"
ep = int("${EP}")
off = 0
for i in range(ep):
    off += pq.read_metadata(root / f"file-{i:03d}.parquet").num_rows
print(off)
PY
)}"

mkdir -p "${OUT}/logs"
INFER_NPZ="${OUT}/infer_qpos_traj_student.npz"
TOKENS="${OUT}/tokens.npz"
MOTION="${OUT}/motion_v4.npz"
MP4="${OUT}/student_sonic_dex3.mp4"

echo "[830_walk_student] ckpt=${STUDENT_CKPT}"
echo "[830_walk_student] ep=${EP} ref_start=${REF_START} frames=${NFRAMES} H=${HORIZON} deploy=${DEPLOY_POLICY_DIR}"
echo "[830_walk_student] out=${OUT}"

echo "[830_walk_student] === offline RTC infer ==="
PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}" \
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/offline_student_rtc_infer.py" \
  --ckpt "${STUDENT_CKPT}" \
  --ref-root "${REF}" \
  --ref-start "${REF_START}" \
  --max-frames "${NFRAMES}" \
  --horizon "${HORIZON}" \
  --device "cuda:0" \
  --out "${INFER_NPZ}" \
  >"${OUT}/logs/offline_infer.log" 2>&1
tail -5 "${OUT}/logs/offline_infer.log"

echo "[830_walk_student] === pack tokens ==="
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_gt_replay_tokens_from_infer_npz.py" \
  "${INFER_NPZ}" --out "${TOKENS}" --fps 50.0 \
  >"${OUT}/logs/pack_tokens.log" 2>&1

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

echo "[830_walk_student] === MuJoCo sonic_v1_1 Dex3 ==="
WORK_DIR="${OUT}/mujoco" \
OUT_MP4="${MP4}" \
MOTION_NPZ="${MOTION}" \
UNIFIED_ROOT="${REF}" \
UNIFIED_PARQUET="${PARQUET}" \
UNIFIED_EP="${EP}" \
VALID_PARQUET="${VALID_PARQUET}" \
QPOS_RSI=1 \
DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR}" \
DEPLOY_OBS_CONFIG="policy/${DEPLOY_POLICY_DIR}/observation_config.yaml" \
DEPLOY_SKIP_ENCODER=1 \
REVO2_HAND=0 \
ROBOT_ONLY=0 \
GT_PANEL_LAYOUT=top \
ENABLE_G1_DEBUG_OVERLAY=0 \
HAND_RAMP_FRAMES=0 \
MAX_FRAMES="${NFRAMES}" \
MOTION_SECONDS=0 \
CONTROL_FPS=50 \
RECORD_FPS=50 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PHI0_PY="${PHI0_PY}" \
bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
  >"${OUT}/logs/mujoco_replay.log" 2>&1 &

echo "mujoco_pid=$! log=${OUT}/logs/mujoco_replay.log"
echo "mp4=${MP4}"
