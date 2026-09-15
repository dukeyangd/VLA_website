#!/usr/bin/env bash
# 830 walk ChunkStudent **closed-loop** MuJoCo viz (primary deliverable).
# Body: live sim; VLM: dataset mp4 + min_batch pad (aligned with frame cache).
# Hand proprio: commanded (last hand_pred), matching train hand_ref semantics.
#
#   EP=194 bash tools/eval/run_830_walk_student_cl_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830_walk_nosim_h14_b32_ddp8_e10_20260826_115006/phi0_student_last.pt}"
EP="${EP:-125}"
TAG="${TAG:-830_walk_student_cl_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"
VALID_PARQUET="${VALID_HAND_ROOT:-/mnt/data2/wpy/workspace/Isaac-GR00T/data/skill_walk_to_black_box_new_v1_1_dex3_obs_hand_valid}/data/chunk-000/episode_$(printf '%06d' "${EP}").parquet"

export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
# unified[396:460] = sonic v1.1 teleop tokens (see run_830_walk_gt_sonic_replay.sh).
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_HAND_MODE="${PHI0_HAND_MODE:-dex3}"
export PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2:-0}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_TRAIN_MODE="${PHI0_TRAIN_MODE:-online_vlm}"
# cache-trained ckpt default: frame_cache (LOCKED). dataset_video only for live/OOD ablation.
export PHI0_USE_VLM_FRAME_LATENT_CACHE="${PHI0_USE_VLM_FRAME_LATENT_CACHE:-1}"
if [[ -z "${PHI0_CL_VLM_SOURCE+x}" ]]; then
  if [[ "${PHI0_USE_VLM_FRAME_LATENT_CACHE}" == "1" || "${PHI0_USE_VLM_FRAME_LATENT_CACHE}" == "true" ]]; then
    export PHI0_CL_VLM_SOURCE=frame_cache
  else
    export PHI0_CL_VLM_SOURCE=dataset_video
  fi
else
  export PHI0_CL_VLM_SOURCE
fi
# Train pack hand obs from observation.state → align CL with sim measured Dex3.
export PHI0_CL_HAND_OBS="${PHI0_CL_HAND_OBS:-g1_debug}"
export PHI0_VLM_ENCODE_MIN_BATCH="${PHI0_VLM_ENCODE_MIN_BATCH:-8}"
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO="${PHI0_VLM_FRAME_CACHE_SKIP_VIDEO:-1}"
export USE_VLM="${USE_VLM:-1}"
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-0}"
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"
export TEACHER_Z_SOURCE=disk
# TRT↔ATM parity bootstrap: keep RSI on first LowCmd, no fall→stand reset,
# hold RSI until student publishes first token (then touch release flag).
export PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD="${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD:-1}"
export PHI0_MUJOCO_DISABLE_FALL_RESET="${PHI0_MUJOCO_DISABLE_FALL_RESET:-1}"
export PHI0_MUJOCO_DEPLOY_FALL_GRACE_S="${PHI0_MUJOCO_DEPLOY_FALL_GRACE_S:-0}"
export QPOS_RSI="${QPOS_RSI:-1}"
export QPOS_RSI_SNAP_ON_RECORD="${QPOS_RSI_SNAP_ON_RECORD:-1}"
export RECORD_SETTLE_S="${RECORD_SETTLE_S:-0}"
export RECORD_STABLE_S="${RECORD_STABLE_S:-1}"
export PHI0_CL_EXPECT_BODY0_MIN="${PHI0_CL_EXPECT_BODY0_MIN:-0.05}"
export PHI0_CL_PROPRIO_SYNC_S="${PHI0_CL_PROPRIO_SYNC_S:-8}"
export PHI0_CL_RSI_HIST_FILL_S="${PHI0_CL_RSI_HIST_FILL_S:-0.30}"

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "[830_walk_cl] missing ckpt ${STUDENT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[830_walk_cl] missing parquet ${PARQUET}" >&2
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

MP4="${OUT}/student_sonic_dex3_cl.mp4"
mkdir -p "${OUT}/logs"

echo "[830_walk_cl] ckpt=${STUDENT_CKPT}"
echo "[830_walk_cl] ep=${EP} ref_start=${REF_START} frames=${NFRAMES} H=${HORIZON} CLOSED_LOOP=1"
echo "[830_walk_cl] deploy=${DEPLOY_POLICY_DIR} vlm=${PHI0_CL_VLM_SOURCE} vlm_enc_min_b=${PHI0_VLM_ENCODE_MIN_BATCH} hand_obs=${PHI0_CL_HAND_OBS} rtc=${PHI0_RTC_INFERENCE_DELAY}/${PHI0_RTC_EXECUTION_HORIZON}"
echo "[830_walk_cl] out=${OUT}"

WORK_DIR="${OUT}/mujoco" \
OUT_MP4="${MP4}" \
UNIFIED_ROOT="${REF}" \
UNIFIED_PARQUET="${PARQUET}" \
UNIFIED_EP="${EP}" \
REF_START="${REF_START}" \
VALID_PARQUET="${VALID_PARQUET}" \
CHUNK_STUDENT_CKPT="${STUDENT_CKPT}" \
USE_CLOSED_LOOP=1 \
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
VLA_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PHI0_PY="${PHI0_PY}" \
bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
  >"${OUT}/logs/mujoco_cl.log" 2>&1 &

echo "pid=$! log=${OUT}/logs/mujoco_cl.log mp4=${MP4}"
