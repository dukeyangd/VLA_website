#!/usr/bin/env bash
# Humanoid Data Studio interactive CL entry.
# Same pipeline as run_830_walk_student_cl_mujoco_viz.sh, but
# STUDIO_INTERACTIVE=1 so run_sonic_latent_sim_eval.sh pauses for web buttons:
#   sim_ready → deploy → init_done → stand(]) → policy(arm) → stream(Enter/p)
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:?STUDENT_CKPT required}"
EP="${EP:-0}"
TAG="${TAG:-studio_cl_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python}"
REF="${REF_ROOT:?REF_ROOT required}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"
VALID_PARQUET="${VALID_HAND_ROOT:-}/data/chunk-000/episode_$(printf '%06d' "${EP}").parquet"

export CUDA_VISIBLE_DEVICES
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export USE_RTC="${USE_RTC:-1}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
export PHI0_HAND_MODE="${PHI0_HAND_MODE:-dex3}"
export PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2:-0}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_TRAIN_MODE="${PHI0_TRAIN_MODE:-online_vlm}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE="${PHI0_USE_VLM_FRAME_LATENT_CACHE:-1}"
export PHI0_CL_VLM_SOURCE="${PHI0_CL_VLM_SOURCE:-frame_cache}"
export PHI0_CL_HAND_OBS="${PHI0_CL_HAND_OBS:-g1_debug}"
export PHI0_VLM_ENCODE_MIN_BATCH="${PHI0_VLM_ENCODE_MIN_BATCH:-8}"
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO="${PHI0_VLM_FRAME_CACHE_SKIP_VIDEO:-1}"
export USE_VLM="${USE_VLM:-1}"
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-0}"
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"
export TEACHER_Z_SOURCE=disk
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
export STUDIO_INTERACTIVE=1

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "[studio_cl] missing ckpt ${STUDENT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[studio_cl] missing parquet ${PARQUET}" >&2
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
mkdir -p "${OUT}/logs" "${OUT}/mujoco"
# Seed control files early so Studio can write commands before engine creates them.
: > "${OUT}/mujoco/studio_cmd"
echo "starting" > "${OUT}/mujoco/studio_phase"
printf '{"phase":"starting","message":"launching","ts":"%s"}\n' "$(date -Iseconds)" > "${OUT}/mujoco/studio_status.json"

echo "[studio_cl] interactive=1 ckpt=${STUDENT_CKPT}"
echo "[studio_cl] ep=${EP} frames=${NFRAMES} H=${HORIZON} out=${OUT}"
echo "[studio_cl] control: ${OUT}/mujoco/studio_cmd  phase: ${OUT}/mujoco/studio_phase"

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
STUDIO_INTERACTIVE=1 \
VLA_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PHI0_PY="${PHI0_PY}" \
bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
  >"${OUT}/logs/mujoco_cl.log" 2>&1

echo "[studio_cl] engine exit=$?"
