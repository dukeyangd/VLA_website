#!/usr/bin/env bash
# 830 walk ChunkStudent closed-loop → Newton-GL mp4 (train-aligned frame cache).
#
#   EP=125 bash tools/eval/run_830_walk_student_cl_newton_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/newton_hand_env.sh"
NEWTON_PY="/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python"
PHI0_PY="${PHI0_PY:-${NEWTON_PY}}"
if [[ "${PHI0_PY}" != *newton* ]]; then
  PHI0_PY="${NEWTON_PY}"
fi

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830_walk_nosim_h14_b32_ddp8_e10_20260826_115006/phi0_student_last.pt}"
EP="${EP:-125}"
# student | ref (disk z_ref→ATM→joint_pd GT) | qpos_ref | …
CONTROL="${CONTROL:-student}"
TAG="${TAG:-830_walk_${CONTROL}_cl_newton_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"

export CUDA_VISIBLE_DEVICES
export PHI0_ISAAC_BACKEND=newton
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
# NEWTON_HAND=rubber (default) | dex3 (composed USD opt-in) | revo2
export NEWTON_HAND="${NEWTON_HAND:-rubber}"
apply_newton_hand_env
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_TRAIN_MODE=online_vlm
export USE_VLM=0
export PHI0_USE_VLM_FRAME_LATENT_CACHE=1
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX=0
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"
export TEACHER_Z_SOURCE=disk

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "[830_walk_newton] missing ckpt ${STUDENT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[830_walk_newton] missing parquet ${PARQUET}" >&2
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

MP4="${OUT}/infer_${CONTROL}_newton_gl.mp4"
mkdir -p "${OUT}/logs"

# GT disk-z: no student/RTC (CONTROL=ref → z_ref→ATM→joint_pd).
if [[ "${CONTROL}" == "ref" || "${CONTROL}" == "ref_g1" || "${CONTROL}" == "qpos_ref" || "${CONTROL}" == "direct_latent_ref" ]]; then
  export USE_RTC="${USE_RTC:-0}"
fi

echo "[830_walk_newton] control=${CONTROL} ckpt=${STUDENT_CKPT}"
echo "[830_walk_newton] ep=${EP} ref_start=${REF_START} frames=${NFRAMES} H=${HORIZON}"
echo "[830_walk_newton] deploy=${DEPLOY_POLICY_DIR} vlm_cache=1 newton_hand=${NEWTON_HAND} mode=${PHI0_HAND_MODE} revo2_usd=${PHI0_NEWTON_REVO2}"
echo "[830_walk_newton] rtc=${USE_RTC} teacher_z=${TEACHER_Z_SOURCE}"
echo "[830_walk_newton] out=${OUT} mp4=${MP4}"

nohup env \
  PHI0_ISAAC_BACKEND=newton \
  NEWTON_HAND="${NEWTON_HAND}" \
  PHI0_HAND_MODE="${PHI0_HAND_MODE}" \
  PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2}" \
  PHI0_NEWTON_DEX3_USD="${PHI0_NEWTON_DEX3_USD:-0}" \
  TEACHER_Z_SOURCE=disk \
  USE_RTC="${USE_RTC}" \
  DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR}" \
  PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}" \
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/newton_qpos_student_isaac_viz.py" \
    --newton_hand "${NEWTON_HAND}" \
    --student_ckpt "${STUDENT_CKPT}" \
    --ref_root "${REF}" \
    --ref_start "${REF_START}" \
    --max_ref_frames "${NFRAMES}" \
    --num_steps "${NFRAMES}" \
    --horizon "${HORIZON}" \
    --control "${CONTROL}" \
    --gt_panel_layout top \
    --out_dir "${OUT}" \
  >"${OUT}/logs/newton_infer.log" 2>&1 &

echo "pid=$! log=${OUT}/logs/newton_infer.log mp4=${MP4}"
