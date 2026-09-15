#!/usr/bin/env bash
# GT replay: 830 skill_walk_to_black_box_new_unified — unified_slice token + sonic_v1_1 decode.
#
#   bash tools/eval/run_830_walk_gt_sonic_replay.sh
#   EP=1 HAND_SOURCE=measured_state bash tools/eval/run_830_walk_gt_sonic_replay.sh
#   EP=1 HAND_SOURCE=unified_gripper bash tools/eval/run_830_walk_gt_sonic_replay.sh  # teleop template
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REF="${REF_ROOT:-${UNIFIED_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}}"
EP="${EP:-${UNIFIED_EP:-0}}"
HAND_SOURCE="${HAND_SOURCE:-measured_state}"
VALID_HAND_ROOT="${VALID_HAND_ROOT:-/mnt/data2/wpy/workspace/Isaac-GR00T/data/skill_walk_to_black_box_new_v1_1_dex3_obs_hand_valid}"
VALID_PARQUET="${VALID_PARQUET:-${VALID_HAND_ROOT}/data/chunk-000/episode_$(printf '%06d' "${EP}").parquet}"
# The converted hand-valid parquet intentionally omits action.wbc.  Resolve the
# original raw episode so dataset-history mode can seed real previous actions
# instead of silently writing zeros.
HISTORY_ACTION_PARQUET="${GT_DECODER_HISTORY_ACTION_PARQUET:-}"
if [[ -z "${HISTORY_ACTION_PARQUET}" && -f "${REF}/meta.json" ]]; then
  RAW_ROOT="$(jq -r '.raw_root // empty' "${REF}/meta.json")"
  SOURCE_SESSION="$(jq -r --argjson ep "${EP}" '.sources[] | select(.out_episode_index == $ep) | .session' "${REF}/meta.json")"
  SOURCE_EP="$(jq -r --argjson ep "${EP}" '.sources[] | select(.out_episode_index == $ep) | .source_episode_index' "${REF}/meta.json")"
  if [[ -n "${RAW_ROOT}" && -n "${SOURCE_SESSION}" && -n "${SOURCE_EP}" ]]; then
    CANDIDATE="${RAW_ROOT}/${SOURCE_SESSION}/data/chunk-000/episode_$(printf '%06d' "${SOURCE_EP}").parquet"
    [[ -f "${CANDIDATE}" ]] && HISTORY_ACTION_PARQUET="${CANDIDATE}"
    if [[ -z "${HISTORY_ACTION_PARQUET}" ]]; then
      LOCAL_RAW_ROOT="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/g1_dex3_data/skill_walk_to_black_box_new"
      CANDIDATE="${LOCAL_RAW_ROOT}/${SOURCE_SESSION}/data/chunk-000/episode_$(printf '%06d' "${SOURCE_EP}").parquet"
      [[ -f "${CANDIDATE}" ]] && HISTORY_ACTION_PARQUET="${CANDIDATE}"
    fi
  fi
fi
HISTORY_ACTION_PARQUET="${HISTORY_ACTION_PARQUET:-${VALID_PARQUET}}"
TAG="${HAND_SOURCE}"
WORK="${WORK_DIR:-${ROOT}/experiments/830_walk_gt_sonic_replay_ep${EP}_${TAG}_${STAMP}}"
EVAL_LOG="${LOG:-${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/logs/830_walk_gt_sonic_replay_ep${EP}_${TAG}_${STAMP}.log}"
mkdir -p "${WORK}"

if [[ "${HAND_SOURCE}" == "measured_state" && ! -f "${VALID_PARQUET}" ]]; then
  echo "[830_walk_gt] missing measured hand parquet: ${VALID_PARQUET}" >&2
  exit 1
fi

# Decoder history comes from real MuJoCo LowState, never dataset overlay.  The
# sim restores RSI when deploy lowcmd becomes active, runs 10 live control
# frames to fill StateLogger, then starts token streaming without a second snap.
echo "=== 830 walk GT replay ep=${EP} hand_source=${HAND_SOURCE} deploy=sonic_v1_1 sim=dex3 live_sim_history=${LIVE_SIM_HISTORY_FRAMES:-10} ==="
env \
  UNIFIED_ROOT="${REF}" UNIFIED_EP="${EP}" EP="${EP}" \
  DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}" \
  GT_DECODER_PROPRIO_FROM_DATASET="${GT_DECODER_PROPRIO_FROM_DATASET:-0}" \
  GT_DECODER_HISTORY_PREFILL="${GT_DECODER_HISTORY_PREFILL:-0}" \
  GT_DECODER_HISTORY_ACTION_PARQUET="${HISTORY_ACTION_PARQUET}" \
  LIVE_SIM_HISTORY_FRAMES="${LIVE_SIM_HISTORY_FRAMES:-10}" \
  QPOS_RSI_SNAP_ON_RECORD="${QPOS_RSI_SNAP_ON_RECORD:-0}" \
  PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD="${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD:-1}" \
  RECORD_SETTLE_S="${RECORD_SETTLE_S:-0}" \
  TOKEN_SOURCE=unified_slice \
  HAND_SOURCE="${HAND_SOURCE}" \
  VALID_PARQUET="${VALID_PARQUET}" \
  REVO2_HAND="${REVO2_HAND:-0}" \
  HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES:-0}" \
  WORK_DIR="${WORK}" \
  CUDA_DEVICES="${CUDA_DEVICES:-0}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  LOG="${EVAL_LOG}" \
  MAX_FRAMES="${MAX_FRAMES:-}" \
  bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh"

MP4="${WORK}/gt_sonic_replay.mp4"
echo "log=${EVAL_LOG} mp4=${MP4} pid=$(cat "${WORK}/main.pid" 2>/dev/null || echo '?')"
