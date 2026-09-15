#!/usr/bin/env bash
# 正式 eval 可视化：GT sonic（数据集 unified_slice token）→ deploy decode → MuJoCo mp4。
# 无 VLA / 无 CHECKPOINT。先验「数据 token + 当前 viz 栈」是否站住。
#
# 红线：REVO2_HAND=1 QPOS_RSI=1 TOKEN_SOURCE=unified_slice
# 非真机默认：不开 sim camera server；ego/wrist 对照条直接读数据集 mp4。
# 若要 live ZMQ 出图：CAMERA_SOURCE=live 或 ENABLE_IMAGE_PUBLISH=1。
#
# Usage:
#   UNIFIED_ROOT=/path/to/unified bash tools/eval/launch_gt_sonic_replay.sh
#   UNIFIED_ROOT=... MAX_FRAMES=579 EP=0 bash launch/eval_gt_sonic.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/tools/env/setup_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/tools/eval/_assert_no_efs_gt_latent.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
EP="${UNIFIED_EP:-${EP:-0}}"
REF="${UNIFIED_ROOT:-}"
if [[ -z "${REF}" ]]; then
  echo "error: set UNIFIED_ROOT=/path/to/*_unified (lerobot layout)" >&2
  exit 1
fi
REF="$(cd "${REF}" && pwd)"

WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/experiments/gt_sonic_replay_ep${EP}_${STAMP}}"
LOG_DIR="${LOG_DIR:-${PHI0_WORKSPACE}/logs}"
mkdir -p "${WORK_DIR}/logs" "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/gt_sonic_replay_ep${EP}_${STAMP}.log}"
H264="${H264_OUT:-${LOG_DIR}/gt_sonic_replay_ep${EP}_${STAMP}_h264.mp4}"

PY="${PHI0_PY:-python}"
# ponytail: ep0 → file-000; multi-ep packs may need UNIFIED_PARQUET override.
PARQUET="${UNIFIED_PARQUET:-${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet}"
if [[ ! -f "${PARQUET}" ]]; then
  # common overfit layout: all rows in file-000
  PARQUET="${REF}/data/chunk-000/file-000.parquet"
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "ERROR: missing parquet under ${REF}" >&2
  exit 1
fi

EP_PAD="$(printf '%06d' "${EP}")"
EGO_MP4="${EGO_MP4:-${REF}/videos/chunk-000/observation.images.ego_view/episode_${EP_PAD}.mp4}"
WRIST_MP4="${WRIST_MP4:-${REF}/videos/chunk-000/observation.images.left_wrist/episode_${EP_PAD}.mp4}"
# text-only / no-video packs: skip HE GT ego panel (else sim crashes on missing mp4)
ROBOT_ONLY="${ROBOT_ONLY:-0}"
GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-top}"
if [[ ! -f "${EGO_MP4}" ]]; then
  ROBOT_ONLY=1
  GT_PANEL_LAYOUT=robot
  EGO_MP4=""
  WRIST_MP4=""
  echo "note: no ego video under ${REF}; ROBOT_ONLY=1" | tee -a "${LOG:-/dev/stderr}"
fi

# Policy: default LL (810); release tokens need DEPLOY_POLICY_DIR=release.
DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-low_latency}"
if [[ -z "${DEPLOY_OBS_CONFIG:-}" ]]; then
  if [[ "${DEPLOY_POLICY_DIR}" == "low_latency" ]]; then
    DEPLOY_OBS_CONFIG="policy/low_latency/observation_config_gt_v4.yaml"
  else
    # release has no gt_v4; decoder-side config is enough with SKIP_ENCODER=1
    DEPLOY_OBS_CONFIG="policy/${DEPLOY_POLICY_DIR}/observation_config.yaml"
  fi
fi
echo "deploy_policy=${DEPLOY_POLICY_DIR} obs_config=${DEPLOY_OBS_CONFIG}" | tee -a "${LOG:-/dev/stderr}"

RSI_NPZ="${WORK_DIR}/qpos_rsi_frame0.npz"
"${PY}" "${PHI0_ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
  "${PARQUET}" --out "${RSI_NPZ}" --frame 0 --episode-index "${EP}" \
  >"${WORK_DIR}/logs/export_qpos_rsi.log" 2>&1 || \
"${PY}" "${PHI0_ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
  "${PARQUET}" --out "${RSI_NPZ}" --frame 0 \
  >"${WORK_DIR}/logs/export_qpos_rsi.log" 2>&1

# Multi-ep file-000 packs: slice one episode so GT replay / MAX_FRAMES align with EP.
# (replay_pick_tissue loads whole parquet then [:max_frames] — no episode filter.)
SLICE_PQ="${WORK_DIR}/episode_${EP_PAD}_slice.parquet"
"${PY}" - <<PY
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
src = "${PARQUET}"
ep = int("${EP}")
out = "${SLICE_PQ}"
t = pq.read_table(src)
if "episode_index" not in t.column_names:
    t2 = t
else:
    e = np.asarray(t.column("episode_index").to_numpy()).reshape(-1)
    idx = np.where(e == ep)[0]
    if idx.size == 0:
        raise SystemExit(f"episode_index={ep} missing in {src}")
    t2 = t.take(pa.array(idx.tolist()))
pq.write_table(t2, out)
print(f"sliced ep={ep} rows={t2.num_rows} -> {out}")
PY
PARQUET="${SLICE_PQ}"
if [[ -z "${MAX_FRAMES:-}" || "${MAX_FRAMES}" -le 0 ]]; then
  MAX_FRAMES="$("${PY}" - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("${PARQUET}").num_rows)
PY
)"
fi

printf 'stamp=%s work=%s ref=%s ep=%s frames=%s robot_only=%s\n' \
  "${STAMP}" "${WORK_DIR}" "${REF}" "${EP}" "${MAX_FRAMES}" "${ROBOT_ONLY}" | tee "${LOG}.launch"

nohup env -u CHECKPOINT -u USE_CLOSED_LOOP -u CONFIG_NAME \
  UNIFIED_ROOT="${REF}" UNIFIED_EP="${EP}" \
  UNIFIED_PARQUET="${PARQUET}" \
  EGO_MP4="${EGO_MP4}" \
  WRIST_MP4="${WRIST_MP4}" \
  TOKEN_SOURCE=unified_slice \
  WORK_DIR="${WORK_DIR}" OUT_MP4="${WORK_DIR}/gt_sonic_replay.mp4" \
  MAX_FRAMES="${MAX_FRAMES}" MOTION_SECONDS=0 CONTROL_FPS=50 RECORD_FPS=50 \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}" \
  DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-low_latency}" \
  DEPLOY_OBS_CONFIG="${DEPLOY_OBS_CONFIG:-}" \
  DEPLOY_SKIP_ENCODER=1 \
  QPOS_RSI=1 QPOS_RSI_NPZ="${RSI_NPZ}" \
  REVO2_HAND="${REVO2_HAND:-1}" \
  HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES:-40}" GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT}" GT_PANEL_LABELS=0 \
  HAND_SOURCE="${HAND_SOURCE:-auto}" \
  ENABLE_G1_DEBUG_OVERLAY=0 ROBOT_ONLY="${ROBOT_ONLY}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  bash "${PHI0_ROOT}/tools/eval/run_sonic_latent_sim_eval.sh" \
  >"${LOG}" 2>&1 &
echo $! | tee "${WORK_DIR}/main.pid"
echo "log=${LOG}"
echo "h264=${H264}"

nohup bash "${PHI0_ROOT}/tools/eval/remux_gt_sonic_when_done.sh" "${WORK_DIR}" "${LOG}" "${H264}" \
  >"${WORK_DIR}/logs/remux_watcher.log" 2>&1 &
echo "remux_watcher=$!"
