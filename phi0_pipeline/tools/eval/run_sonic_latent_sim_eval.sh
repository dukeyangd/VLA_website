#!/usr/bin/env bash
# SONIC latent (GT / model / closed-loop) ZMQ v4 -> deploy -> MuJoCo mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/tools/eval/_assert_no_efs_gt_latent.sh"
# Hard prefer in-tree subpackages (refuse efs MixCorpus GR00T as default deploy root).
case "${GR00T_ROOT}" in
  */Phi_0_wpy/subpackages|*/Phi_0_wpy/subpackages/*) ;;
  *)
    if [[ "${GR00T_ROOT}" == */Phi0-MixCorpus/* ]] || [[ "${GR00T_ROOT}" == /mnt/efs_1/* ]]; then
      echo "[sonic_latent] ERROR: GR00T_ROOT=${GR00T_ROOT} looks like efs/MixCorpus" >&2
      echo "[sonic_latent] unset GR00T_ROOT / PHI0_SUBPACKAGES_OVERRIDE; use Phi_0_wpy/subpackages" >&2
      exit 1
    fi
    ;;
esac
GR00T_ROOT="$(cd "${GR00T_ROOT}" && pwd)"
VENV_SIM="${VENV_SIM:-${PHI0_ROOT}/.venv_sim}"
DEPLOY="${GEAR_SONIC_DEPLOY:-${GR00T_ROOT}/gear_sonic_deploy}"
ROBOT_MOTION="${ROBOT_MOTION:-${GR00T_ROOT}/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl}"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/logs/sonic_latent_sim_eval/$([ -n "${CHECKPOINT:-}" ] && echo model || echo gt)_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
VALID_ROOT="${VALID_ROOT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_valid}"
UNIFIED_ROOT="${UNIFIED_ROOT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_xperience_unified}"
MANIFEST_PATH="${MANIFEST_PATH:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissues.json}"
TOKEN_SOURCE="${TOKEN_SOURCE:-unified_slice}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Closed-loop VLA + TensorRT on one GPU stalls LowState (age 100ms+). Split by default.
VLA_CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES:-}"
DEPLOY_CUDA_VISIBLE_DEVICES="${DEPLOY_CUDA_VISIBLE_DEVICES:-}"
RECORD_FPS="${RECORD_FPS:-30}"
CONTROL_FPS="${CONTROL_FPS:-50}"
MOTION_SECONDS="${MOTION_SECONDS:-8}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RECORD_SETTLE_S="${RECORD_SETTLE_S:-8}"
RECORD_STABLE_S="${RECORD_STABLE_S:-5}"
GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-inset}"
ENABLE_G1_DEBUG_OVERLAY="${ENABLE_G1_DEBUG_OVERLAY:-1}"
CHECKPOINT="${CHECKPOINT:-}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_xperience_unified_ddp4_3k}"
export USE_RTC="${USE_RTC:-}"
# Non-real deploy: images from dataset (GT panel mp4 / CL --camera-source gt).
# Live ZMQ SensorServer (:5555) only when CAMERA_SOURCE=live or ENABLE_IMAGE_PUBLISH=1.
CAMERA_SOURCE="${CAMERA_SOURCE:-gt}"
if [[ -z "${ENABLE_IMAGE_PUBLISH:-}" ]]; then
  if [[ "${CAMERA_SOURCE}" == "live" ]]; then
    ENABLE_IMAGE_PUBLISH=1
  else
    ENABLE_IMAGE_PUBLISH=0
  fi
fi
ENABLE_IMAGE_PUBLISH="${ENABLE_IMAGE_PUBLISH}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GR00T_ROOT}:${PHI0_ROOT}/src:${PHI0_SUBPACKAGES:-${PHI0_ROOT}/subpackages}:${PYTHONPATH:-}"
# TensorRT / onnxruntime / unitree DDS: setup_env.sh
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
SIM_WARMUP_S="${SIM_WARMUP_S:-10}"
DEPLOY_INIT_TIMEOUT_S="${DEPLOY_INIT_TIMEOUT_S:-300}"
REPLAY_READY_TIMEOUT_S="${REPLAY_READY_TIMEOUT_S:-900}"

# Default GPU split for live closed-loop (VLA vs TensorRT deploy).
if [[ -n "${CHECKPOINT}" && "${USE_CLOSED_LOOP:-0}" == "1" ]]; then
  _cvd_first="${CUDA_VISIBLE_DEVICES%%,*}"
  VLA_CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES:-${_cvd_first}}"
  if [[ -z "${DEPLOY_CUDA_VISIBLE_DEVICES}" ]]; then
    if [[ "${_cvd_first}" == "0" ]]; then
      DEPLOY_CUDA_VISIBLE_DEVICES=7
    else
      DEPLOY_CUDA_VISIBLE_DEVICES=0
    fi
  fi
else
  VLA_CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  DEPLOY_CUDA_VISIBLE_DEVICES="${DEPLOY_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
fi
export VLA_CUDA_VISIBLE_DEVICES DEPLOY_CUDA_VISIBLE_DEVICES

# Resolve episode parquets (manifest ep2 -> unified idx 447, valid src ep524)
UNIFIED_EP="${UNIFIED_EP:-}"
VALID_EP="${VALID_EP:-}"
if [[ -n "${MANIFEST_SESSION:-}" && -n "${MANIFEST_EP:-}" ]]; then
  read -r UNIFIED_EP VALID_EP <<<"$(
    PHI0_ROOT="${PHI0_ROOT}" MANIFEST_PATH="${MANIFEST_PATH}" VALID_ROOT="${VALID_ROOT}" \
    MANIFEST_SESSION="${MANIFEST_SESSION}" MANIFEST_EP="${MANIFEST_EP}" "${PHI0_PY}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["PHI0_ROOT"], "src"))
from phi0.data.pick_tissue_episode_map import (
    manifest_ep_to_dst_ep,
    manifest_ep_to_unified_episode_index,
)
m, v, s, e = (
    os.environ["MANIFEST_PATH"],
    os.environ["VALID_ROOT"],
    os.environ["MANIFEST_SESSION"],
    int(os.environ["MANIFEST_EP"]),
)
ui = manifest_ep_to_unified_episode_index(m, v, s, e)
di = manifest_ep_to_dst_ep(m, s, e)
print(ui, di)
PY
  )"
  echo "[sonic_latent] manifest ${MANIFEST_SESSION} ep${MANIFEST_EP} -> unified=${UNIFIED_EP} valid=${VALID_EP}"
fi
UNIFIED_EP="${UNIFIED_EP:-447}"
if [[ -z "${VALID_EP:-}" ]]; then
  VALID_EP="$("${PHI0_PY}" - <<PY
import sys
sys.path.insert(0, "${PHI0_ROOT}/src")
from phi0.data.pick_tissue_episode_map import _sorted_valid_parquets
files = _sorted_valid_parquets("${VALID_ROOT}")
idx = int("${UNIFIED_EP}")
name = files[idx].stem  # episode_000544
print(int(name.split("_")[-1]))
PY
)"
fi

UNIFIED_PARQUET="${UNIFIED_PARQUET:-${UNIFIED_ROOT}/data/chunk-000/episode_$(printf '%06d' "${UNIFIED_EP}").parquet}"
VALID_PARQUET="${VALID_PARQUET:-${VALID_ROOT}/data/chunk-000/episode_$(printf '%06d' "${VALID_EP}").parquet}"
EGO_MP4="${EGO_MP4:-${UNIFIED_ROOT}/videos/chunk-000/observation.images.ego_view/episode_$(printf '%06d' "${UNIFIED_EP}").mp4}"
WRIST_MP4="${WRIST_MP4:-${UNIFIED_ROOT}/videos/chunk-000/observation.images.left_wrist/episode_$(printf '%06d' "${UNIFIED_EP}").mp4}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/pick_tissue_ep${UNIFIED_EP}_sonic_latent_$([ -n "${CHECKPOINT}" ] && echo model || echo gt).mp4}"
OUT_MP4="$(python3 -c "import os; print(os.path.abspath(os.path.expanduser('${OUT_MP4}')))")"
mkdir -p "$(dirname "${OUT_MP4}")"

if [[ -n "${MOTION_NPZ:-}" && -f "${MOTION_NPZ}" ]]; then
  _NPZ_FRAMES="$("${PHI0_PY}" - <<PY
import numpy as np
print(int(np.load("${MOTION_NPZ}")["tokens"].shape[0]))
PY
  )"
  if [[ "${MAX_FRAMES}" -le 0 ]]; then
    MAX_FRAMES="${_NPZ_FRAMES}"
  else
    MAX_FRAMES="$("${PHI0_PY}" - <<PY
print(min(int("${MAX_FRAMES}"), int("${_NPZ_FRAMES}")))
PY
)"
  fi
  MOTION_SECONDS="$("${PHI0_PY}" - <<PY
print(round(int("${MAX_FRAMES}") / float("${CONTROL_FPS}"), 2))
PY
  )"
  echo "[sonic_latent] MOTION_NPZ=${MOTION_NPZ} frames=${MAX_FRAMES} (~${MOTION_SECONDS}s)"
elif [[ "${MAX_FRAMES}" -le 0 ]]; then
  MAX_FRAMES="$("${PHI0_PY}" - <<PY
import math
print(int(math.ceil(float("${MOTION_SECONDS}") * float("${CONTROL_FPS}"))))
PY
)"
fi
PARQUET_ROWS="$("${PHI0_PY}" - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("${UNIFIED_PARQUET}").num_rows)
PY
)"
REQUESTED="${MAX_FRAMES}"
if [[ -z "${MOTION_NPZ:-}" && "${PARQUET_ROWS}" -lt "${MAX_FRAMES}" ]]; then
  MAX_FRAMES="${PARQUET_ROWS}"
  EP_DUR="$("${PHI0_PY}" - <<PY
print(round(${PARQUET_ROWS}/float("${CONTROL_FPS}"), 2))
PY
)"
  echo "[sonic_latent] WARN: episode has ${PARQUET_ROWS} frames (~${EP_DUR}s @ ${CONTROL_FPS}Hz), less than MOTION_SECONDS=${MOTION_SECONDS} (${REQUESTED} frames)"
fi

RECORD_START="${WORK_DIR}/.record_start"
RECORD_STOP="${WORK_DIR}/.record_stop"
ARM_FLAG="${WORK_DIR}/.arm_deploy"
REPLAY_READY="${WORK_DIR}/.replay_go"
STREAM_FLAG="${WORK_DIR}/.stream_deploy"
DEPLOY_FIFO="${DEPLOY_FIFO:-/tmp/pick_tissue_sonic_deploy_$$.fifo}"
# Align with phi-0-wbc-newton: panel/clock arms with REPLAY_READY (tokens first).
export RECORD_GO_FLAG="${RECORD_GO_FLAG:-${REPLAY_READY}}"
export CONTROL_FPS

# Optional exact SONIC decoder history seed.  Unlike the legacy LowState
# overlay, this is dormant during deploy warmup and is applied atomically only
# after the simulator has snapped to START_FRAME and the first token is ready.
GT_DECODER_HISTORY_PREFILL="${GT_DECODER_HISTORY_PREFILL:-0}"
if [[ "${GT_DECODER_HISTORY_PREFILL}" == "1" ]]; then
  GT_DECODER_HISTORY_START_FRAME="${GT_DECODER_HISTORY_START_FRAME:-9}"
  GT_DECODER_HISTORY_FILE="${GT_DECODER_HISTORY_FILE:-${WORK_DIR}/gt_decoder_history.csv}"
  GT_DECODER_HISTORY_GO="${GT_DECODER_HISTORY_GO:-${WORK_DIR}/.gt_history_seed_go}"
  GT_DECODER_HISTORY_ACK="${GT_DECODER_HISTORY_ACK:-${WORK_DIR}/.gt_history_seed_ack}"
  GT_DECODER_HISTORY_SOURCE_PARQUET="${GT_DECODER_HISTORY_SOURCE_PARQUET:-${UNIFIED_PARQUET}}"
  GT_DECODER_HISTORY_ACTION_PARQUET="${GT_DECODER_HISTORY_ACTION_PARQUET:-${VALID_PARQUET:-${UNIFIED_PARQUET}}}"
  GT_HISTORY_RSI_NPZ="${GT_HISTORY_RSI_NPZ:-${WORK_DIR}/qpos_rsi_history_start.npz}"
  HISTORY_EXPORT_ARGS=()
  if [[ -n "${GT_DECODER_HISTORY_EPISODE_INDEX:-}" ]]; then
    HISTORY_EXPORT_ARGS+=(--episode-index "${GT_DECODER_HISTORY_EPISODE_INDEX}")
  fi
  if [[ -n "${GT_DECODER_HISTORY_ACTION_EPISODE_INDEX:-}" ]]; then
    HISTORY_EXPORT_ARGS+=(--action-episode-index "${GT_DECODER_HISTORY_ACTION_EPISODE_INDEX}")
  fi
  "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/export_gt_decoder_history_seed.py" \
    "${GT_DECODER_HISTORY_SOURCE_PARQUET}" \
    --action-parquet "${GT_DECODER_HISTORY_ACTION_PARQUET}" \
    --out "${GT_DECODER_HISTORY_FILE}" \
    --rsi-out "${GT_HISTORY_RSI_NPZ}" \
    --start-frame "${GT_DECODER_HISTORY_START_FRAME}" \
    --history-frames 10 --fps "${CONTROL_FPS}" \
    "${HISTORY_EXPORT_ARGS[@]}" \
    > "${LOG_DIR}/export_gt_decoder_history.log" 2>&1
  rm -f "${GT_DECODER_HISTORY_GO}" "${GT_DECODER_HISTORY_ACK}"
  export GT_DECODER_HISTORY_FILE GT_DECODER_HISTORY_GO GT_DECODER_HISTORY_ACK
  QPOS_RSI=1
  QPOS_RSI_NPZ="${GT_HISTORY_RSI_NPZ}"
  echo "[sonic_latent] GT decoder history prefill frame=${GT_DECODER_HISTORY_START_FRAME} file=${GT_DECODER_HISTORY_FILE}"
fi

SIM_PID=""
DEPLOY_PID=""
REPLAY_PID=""

log_step() {
  echo "[sonic_latent][$(date '+%H:%M:%S')] $*"
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

require_pid() {
  local pid="$1" name="$2" logfile="${3:-}"
  if pid_alive "${pid}"; then
    return 0
  fi
  log_step "ERROR: ${name} (pid=${pid}) exited unexpectedly"
  if [[ -n "${logfile}" && -f "${logfile}" ]]; then
    log_step "--- tail ${logfile} ---"
    tail -25 "${logfile}" || true
  fi
  exit 1
}

require_replay() {
  local label="replay"
  if [[ -n "${MOTION_NPZ:-}" ]]; then
    label="npz replay"
  elif [[ -n "${CHECKPOINT:-}" ]]; then
    label="model publisher"
  fi
  require_pid "${REPLAY_PID}" "${label}" "${LOG_DIR}/replay.log"
  if grep -qE "Traceback \(most recent call last\)" "${LOG_DIR}/replay.log" 2>/dev/null; then
    log_step "model publisher crashed"
    tail -30 "${LOG_DIR}/replay.log"
    exit 1
  fi
}

_fall_count() {
  local n=0
  if [[ -f "$1" ]]; then
    n=$(grep -cF '[sim_health] FALL' "$1" 2>/dev/null) || n=0
  fi
  echo "${n}"
}

cleanup() {
  for pid in "${REPLAY_PID}" "${DEPLOY_PID}" "${SIM_PID}"; do
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && kill "${pid}" 2>/dev/null || true
  done
  rm -f "${DEPLOY_FIFO}" "${RECORD_START}" "${RECORD_STOP}" "${ARM_FLAG}" "${REPLAY_READY}" "${STREAM_FLAG}"
  if [[ -n "${GT_DECODER_HISTORY_GO:-}" ]]; then rm -f "${GT_DECODER_HISTORY_GO}"; fi
  if [[ -n "${GT_DECODER_HISTORY_ACK:-}" ]]; then rm -f "${GT_DECODER_HISTORY_ACK}"; fi
  if [[ "${STUDIO_INTERACTIVE:-0}" == "1" ]]; then
    echo "stopped" > "${WORK_DIR}/studio_phase" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Studio web UI gates (STUDIO_INTERACTIVE=1): pause between operator steps.
# Terminal / buttons write one line to ${WORK_DIR}/studio_cmd (y / ] / enter / deploy / …).
STUDIO_CMD_FILE="${WORK_DIR}/studio_cmd"
STUDIO_PHASE_FILE="${WORK_DIR}/studio_phase"
STUDIO_STATUS_FILE="${WORK_DIR}/studio_status.json"
studio_set_phase() {
  local phase="$1" msg="${2:-}"
  mkdir -p "${WORK_DIR}"
  echo "${phase}" > "${STUDIO_PHASE_FILE}"
  printf '{"phase":"%s","message":"%s","ts":"%s"}\n' \
    "${phase}" "${msg}" "$(date -Iseconds)" > "${STUDIO_STATUS_FILE}"
  log_step "STUDIO phase=${phase} ${msg}"
}
studio_wait_cmd() {
  # Usage: studio_wait_cmd expected_cmd [aliases...]
  local expected="$1"; shift || true
  STUDIO_LAST_CMD=""
  if [[ "${STUDIO_INTERACTIVE:-0}" != "1" ]]; then
    STUDIO_LAST_CMD="${expected}"
    return 0
  fi
  mkdir -p "${WORK_DIR}"
  : >> "${STUDIO_CMD_FILE}"
  log_step "STUDIO waiting for cmd=${expected} (file=${STUDIO_CMD_FILE})"
  while true; do
    local cmd
    cmd="$(head -n1 "${STUDIO_CMD_FILE}" 2>/dev/null | tr -d '\r' || true)"
    if [[ -n "${cmd}" ]]; then
      : > "${STUDIO_CMD_FILE}"
      case "${cmd}" in
        stop|quit|exit)
          studio_set_phase "stopped" "stop requested"
          exit 0
          ;;
        key:*)
          # Raw deploy FIFO key (y / ] / Enter) — also may satisfy expected aliases.
          local raw="${cmd#key:}"
          if declare -F send_deploy_key >/dev/null 2>&1; then
            if [[ "${raw}" == "\\n" || "${raw}" == "enter" || "${raw}" == "ENTER" ]]; then
              printf '\n' >&3 2>/dev/null || send_deploy_key $'\n'
            else
              send_deploy_key "${raw}"
            fi
          else
            log_step "STUDIO key queued (deploy fifo not open yet): ${cmd}"
          fi
          # Treat key:y / key:] / key:\n as stage advances when matching.
          if [[ "${raw}" == "y" || "${raw}" == "Y" ]]; then cmd="y"
          elif [[ "${raw}" == "]" ]]; then cmd="]"
          elif [[ "${raw}" == "\\n" || "${raw}" == "enter" || "${raw}" == "ENTER" || "${raw}" == "p" ]]; then
            cmd="${raw}"
            [[ "${cmd}" == "\\n" || "${cmd}" == "ENTER" ]] && cmd="enter"
          else
            continue
          fi
          ;;
      esac
      if [[ "${cmd}" == "${expected}" ]]; then
        STUDIO_LAST_CMD="${cmd}"
        log_step "STUDIO got cmd=${cmd}"
        return 0
      fi
      local alias
      for alias in "$@"; do
        if [[ "${cmd}" == "${alias}" ]]; then
          STUDIO_LAST_CMD="${cmd}"
          log_step "STUDIO got cmd=${cmd} (alias of ${expected})"
          return 0
        fi
      done
      log_step "STUDIO ignore unexpected cmd=${cmd} (want ${expected})"
    fi
    sleep 0.25
  done
}

wait_log() {
  local file="$1" pattern="$2" timeout="$3"
  local label="${4:-pattern}"
  log_step "wait_log: ${label} (timeout=${timeout}s) -> ${file}"
  for i in $(seq 1 "${timeout}"); do
    if grep -qE "${pattern}" "${file}" 2>/dev/null; then
      log_step "wait_log: OK ${label} (${i}s)"
      return 0
    fi
    if (( i % 10 == 0 )); then
      log_step "wait_log: still waiting ${label} (${i}/${timeout}s)"
      tail -2 "${file}" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
    fi
    sleep 1
  done
  log_step "wait_log: TIMEOUT ${label} after ${timeout}s"
  tail -20 "${file}" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
  return 1
}

send_deploy_key() { printf '%s' "$1" >&3; }

echo "[sonic_latent] unified=${UNIFIED_PARQUET} valid_hands=${VALID_PARQUET}"
echo "[sonic_latent] deploy_policy=${DEPLOY_POLICY_DIR:-release}"
echo "[sonic_latent] token_source=${TOKEN_SOURCE} frames=${MAX_FRAMES} out=${OUT_MP4}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "[sonic_latent] MODEL checkpoint=${CHECKPOINT} config=${CONFIG_NAME}"
elif [[ -n "${MOTION_NPZ:-}" ]]; then
  echo "[sonic_latent] NPZ replay motion=${MOTION_NPZ} hand_ramp=${HAND_RAMP_FRAMES:-0}"
fi
log_step "work_dir=${WORK_DIR} gpu=${CUDA_VISIBLE_DEVICES} vla_gpu=${VLA_CUDA_VISIBLE_DEVICES} deploy_gpu=${DEPLOY_CUDA_VISIBLE_DEVICES} sim_warmup=${SIM_WARMUP_S}s deploy_timeout=${DEPLOY_INIT_TIMEOUT_S}s"

pkill -f "run_sim_loop_vla_record" 2>/dev/null || true
pkill -f "g1_deploy_onnx_ref.*zmq_manager" 2>/dev/null || true
pkill -f "replay_pick_tissue_sonic_latent_zmq_v4" 2>/dev/null || true
pkill -f "replay_sonic_latent_npz_zmq_v4" 2>/dev/null || true
for port in 5555 5556 5557; do
  fuser -k "${port}/tcp" >/dev/null 2>&1 || true
done
sleep 2
rm -f "${RECORD_START}" "${RECORD_STOP}" "${ARM_FLAG}" "${REPLAY_READY}" "${STREAM_FLAG}"
if [[ -n "${GT_DECODER_HISTORY_GO:-}" ]]; then rm -f "${GT_DECODER_HISTORY_GO}"; fi
if [[ -n "${GT_DECODER_HISTORY_ACK:-}" ]]; then rm -f "${GT_DECODER_HISTORY_ACK}"; fi

PRECOMPUTE_NPZ="${PRECOMPUTE_IN:-${WORK_DIR}/sonic_latent_precompute.npz}"
USE_PRECOMPUTE_NPZ=0
if [[ -n "${CHECKPOINT}" ]]; then
  if [[ "${FORCE_PRECOMPUTE:-}" == "1" ]]; then
    if [[ ! -f "${PRECOMPUTE_NPZ}" ]]; then
      log_step "optional offline precompute -> ${PRECOMPUTE_NPZ} (set FORCE_PRECOMPUTE=0 for inline infer)"
      (
        cd "${PHI0_ROOT}"
        "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/phi0_sonic_latent_zmq_publisher.py" \
          --checkpoint "${CHECKPOINT}" \
          --config-name "${CONFIG_NAME}" \
          --episode-idx "${UNIFIED_EP}" \
          --control-fps "${CONTROL_FPS}" \
          --motion-seconds "${MOTION_SECONDS}" \
          --max-frames "${MAX_FRAMES}" \
          --precompute-out "${PRECOMPUTE_NPZ}"
      ) > "${LOG_DIR}/precompute.log" 2>&1 || {
        log_step "precompute failed"
        tail -40 "${LOG_DIR}/precompute.log"
        exit 1
      }
      log_step "precompute done ($(wc -l < "${LOG_DIR}/precompute.log" | tr -d ' ') log lines)"
    else
      log_step "reusing precompute ${PRECOMPUTE_NPZ}"
    fi
    USE_PRECOMPUTE_NPZ=1
  elif [[ -n "${PRECOMPUTE_IN:-}" ]]; then
    if [[ ! -f "${PRECOMPUTE_NPZ}" ]]; then
      log_step "ERROR: PRECOMPUTE_IN=${PRECOMPUTE_IN} not found"
      exit 1
    fi
    log_step "reuse precompute ${PRECOMPUTE_NPZ}"
    USE_PRECOMPUTE_NPZ=1
  else
    log_step "inline infer at publisher (dataset clip + VLM, no offline precompute)"
  fi
fi

# 1) MuJoCo sim + mp4 (defaults match sonic_latent_gt_20260628_030836)
SIM_EXTRA_ARGS=()
SIM_EGO_GT="${EGO_MP4}"
# 810short / T3 train is Revo2; Dex3 scene mismatches proprio41 + hand ZMQ packing.
# Override: REVO2_HAND=0 for legacy Dex3-only evals.
if [[ "${REVO2_HAND:-1}" == "1" || "${REVO2_HAND:-}" == "true" ]]; then
  SIM_EXTRA_ARGS+=(--revo2-hand)
  echo "[sonic_latent] sim hand=revo2 (scene_43dof_revo2)"
else
  SIM_EXTRA_ARGS+=(--no-revo2-hand)
  echo "[sonic_latent] sim hand=dex3 (--no-revo2-hand)"
fi
if [[ "${ROBOT_ONLY:-}" == "1" || "${GT_PANEL_LAYOUT}" == "robot" ]]; then
  SIM_EGO_GT=""
  SIM_EXTRA_ARGS+=(--camera-host "")
elif [[ "${GT_PANEL_LAYOUT}" == "top" ]]; then
  SIM_EXTRA_ARGS+=(--gt-panel-layout top)
  if [[ -n "${WRIST_MP4:-}" && -f "${WRIST_MP4}" ]]; then
    SIM_EXTRA_ARGS+=(--wrist-gt-video "${WRIST_MP4}")
  fi
elif [[ "${GT_PANEL_LAYOUT}" == "sim" ]]; then
  SIM_EGO_GT=""
  SIM_EXTRA_ARGS+=(--camera-host 127.0.0.1)
fi
if [[ "${ENABLE_G1_DEBUG_OVERLAY}" == "1" ]]; then
  SIM_EXTRA_ARGS+=(--enable-g1-debug-overlay)
else
  SIM_EXTRA_ARGS+=(--g1-debug-snap --no-enable-g1-debug-overlay)
fi
QPOS_RSI="${QPOS_RSI:-0}"
QPOS_RSI_NPZ="${QPOS_RSI_NPZ:-}"
if [[ "${QPOS_RSI}" == "1" ]]; then
  if [[ -z "${QPOS_RSI_NPZ}" ]]; then
    QPOS_RSI_NPZ="${WORK_DIR}/qpos_rsi_frame0.npz"
  fi
  if [[ ! -f "${QPOS_RSI_NPZ}" && -n "${UNIFIED_PARQUET:-}" && -f "${UNIFIED_PARQUET}" ]]; then
    "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
      "${UNIFIED_PARQUET}" --out "${QPOS_RSI_NPZ}" --frame "${QPOS_RSI_FRAME:-0}" \
      ${UNIFIED_EP:+--episode-index "${UNIFIED_EP}"} \
      >> "${LOG_DIR}/export_qpos_rsi.log" 2>&1
  fi
  if [[ ! -f "${QPOS_RSI_NPZ}" ]]; then
    echo "[sonic_latent] ERROR: QPOS_RSI=1 but missing ${QPOS_RSI_NPZ}" >&2
    exit 1
  fi
  log_step "qpos RSI npz=${QPOS_RSI_NPZ} (snap sim to dataset frame0 before latent)"
  SIM_EXTRA_ARGS+=(--qpos-snap-npz "${QPOS_RSI_NPZ}" --qpos-snap-frame "${QPOS_RSI_FRAME:-0}")
  if [[ "${QPOS_RSI_SNAP_ON_RECORD:-1}" == "1" ]]; then
    SNAP_ON_RECORD=--snap-on-record-start
  else
    # Live-SIM decoder history is collected after the initial RSI/lowcmd
    # restore.  A second teleport at record start would invalidate that tape.
    SNAP_ON_RECORD=--no-snap-on-record-start
  fi
  # Closed-loop: freeze RSI after snap until student first token (shared flag path).
  if [[ "${USE_CLOSED_LOOP:-0}" == "1" ]]; then
    export PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD="${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD:-1}"
    export PHI0_MUJOCO_DISABLE_FALL_RESET="${PHI0_MUJOCO_DISABLE_FALL_RESET:-1}"
    export PHI0_MUJOCO_DEPLOY_FALL_GRACE_S="${PHI0_MUJOCO_DEPLOY_FALL_GRACE_S:-0}"
    RSI_HOLD_RELEASE="${WORK_DIR}/.release_rsi_hold"
    rm -f "${RSI_HOLD_RELEASE}"
    export PHI0_MUJOCO_HOLD_RSI_UNTIL_FLAG="${RSI_HOLD_RELEASE}"
    export PHI0_CL_RELEASE_RSI_HOLD_FLAG="${RSI_HOLD_RELEASE}"
    log_step "RSI hold-until-flag=${RSI_HOLD_RELEASE} keep_pose=${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD} no_fall_reset=${PHI0_MUJOCO_DISABLE_FALL_RESET}"
  fi
else
  SNAP_ON_RECORD=--no-snap-on-record-start
fi
# Cluster-rsync'd .venv_sim needs local python + VIRTUAL_ENV paths.
# Dataset GT LowState → decoder his_* (joints/IMU/gravity via quat), not live sim.
if [[ "${GT_DECODER_PROPRIO_FROM_DATASET:-0}" == "1" ]]; then
  GT_DECODER_LOWSTATE_NPZ="${GT_DECODER_LOWSTATE_NPZ:-${WORK_DIR}/gt_decoder_lowstate.npz}"
  GT_DECODER_FRAME_PATH="${GT_DECODER_FRAME_PATH:-${WORK_DIR}/.gt_decoder_frame}"
  if [[ ! -f "${GT_DECODER_LOWSTATE_NPZ}" && -n "${UNIFIED_PARQUET:-}" && -f "${UNIFIED_PARQUET}" ]]; then
    "${PHI0_PY}" "${PHI0_ROOT}/tools/eval/export_gt_decoder_lowstate_npz.py" \
      "${UNIFIED_PARQUET}" --out "${GT_DECODER_LOWSTATE_NPZ}" \
      ${UNIFIED_EP:+--episode-index "${UNIFIED_EP}"} \
      >> "${LOG_DIR}/export_gt_decoder_lowstate.log" 2>&1
  fi
  if [[ ! -f "${GT_DECODER_LOWSTATE_NPZ}" ]]; then
    echo "[sonic_latent] ERROR: GT_DECODER_PROPRIO_FROM_DATASET=1 but missing ${GT_DECODER_LOWSTATE_NPZ}" >&2
    exit 1
  fi
  export GT_DECODER_LOWSTATE_NPZ GT_DECODER_FRAME_PATH
  echo 0 > "${GT_DECODER_FRAME_PATH}"
  log_step "GT decoder LowState overlay npz=${GT_DECODER_LOWSTATE_NPZ} frame=${GT_DECODER_FRAME_PATH}"
fi

bash "${PHI0_ROOT}/tools/env/fix_venv_sim.sh" >> "${LOG_DIR}/fix_venv_sim.log" 2>&1 || true
SIM_IMAGE_PUB_ARGS=(--no-enable-image-publish)
if [[ "${ENABLE_IMAGE_PUBLISH}" == "1" || "${ENABLE_IMAGE_PUBLISH}" == "true" ]]; then
  SIM_IMAGE_PUB_ARGS=(--enable-image-publish)
fi
log_step "camera_source=${CAMERA_SOURCE} enable_image_publish=${ENABLE_IMAGE_PUBLISH}"
(
  cd "${GR00T_ROOT}"
  source "${VENV_SIM}/bin/activate"
  python -u experiments/sonic_vla_overfit/scripts/run_sim_loop_vla_record.py \
    --no-enable-onscreen \
    "${SIM_IMAGE_PUB_ARGS[@]}" --enable-offscreen \
    --disable-elastic-band \
    --camera-port 5555 \
    --g1-debug-host 127.0.0.1 \
    --g1-debug-port 5557 \
    ${SNAP_ON_RECORD} \
    ${SIM_EGO_GT:+--ego-gt-video "${SIM_EGO_GT}"} \
    ${GT_TOKENS_NPZ:+--gt-tokens-npz "${GT_TOKENS_NPZ}"} \
    --record-mp4 "${OUT_MP4}" \
    --record-start-flag "${RECORD_START}" \
    --record-stop-flag "${RECORD_STOP}" \
    --record-fps "${RECORD_FPS}" \
    "${SIM_EXTRA_ARGS[@]}"
) > "${LOG_DIR}/sim.log" 2>&1 &
SIM_PID=$!
log_step "sim pid=${SIM_PID} log=${LOG_DIR}/sim.log"
# Dataset GT panels / CL camera-source=gt: no SensorServer. Live cam waits for it.
wait_log "${LOG_DIR}/sim.log" "sim loop starting" 120 "sim loop" || {
  if [[ "${ENABLE_IMAGE_PUBLISH}" == "1" || "${ENABLE_IMAGE_PUBLISH}" == "true" ]]; then
    wait_log "${LOG_DIR}/sim.log" "Sensor server running" 30 "sim Sensor server" || {
      if ss -tlnp 2>/dev/null | grep -qE ":${CAMERA_PORT:-5555}\\b"; then
        log_step "wait_log: OK sim Sensor server (port ${CAMERA_PORT:-5555} listening)"
      else
        require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
        tail -30 "${LOG_DIR}/sim.log"; exit 1
      fi
    }
  else
    require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
    tail -30 "${LOG_DIR}/sim.log"; exit 1
  fi
}
# Hard check: Revo2 scene so g1_debug hands match proprio41 / VLA train contract.
if [[ "${REVO2_HAND:-1}" == "1" || "${REVO2_HAND:-}" == "true" ]]; then
  if ! grep -qE '\[revo2_hand\]|scene_43dof_revo2|g1_41dof_revo2|with_revo2' "${LOG_DIR}/sim.log" 2>/dev/null; then
    log_step "ERROR: expected Revo2 sim (marker [revo2_hand] / scene_43dof_revo2) in ${LOG_DIR}/sim.log"
    tail -40 "${LOG_DIR}/sim.log" || true
    exit 1
  fi
  log_step "sim revo2 confirmed — g1_debug hands feed VLA proprio41"
fi
log_step "sim up; LowState bridge warmup ${SIM_WARMUP_S}s..."
for ((w=SIM_WARMUP_S; w>0; w-=5)); do
  require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
  log_step "warmup ${w}s remaining..."
  sleep 5
done

# Studio: after Sim init, wait for Deploy / terminal「y」before publisher+deploy.
if [[ "${STUDIO_INTERACTIVE:-0}" == "1" ]]; then
  studio_set_phase "sim_ready" "Sim initialized — Deploy 或 terminal 输入 y"
  studio_wait_cmd "deploy" "go" "continue" "y" "Y"
  studio_set_phase "deploy_starting" "starting publisher + deploy"
fi

# 2) ZMQ v4 publisher: arm_flag -> command start; replay_go -> pose stream
if [[ -n "${MOTION_NPZ:-}" ]]; then
  log_step "starting motion npz replay (${MOTION_NPZ})..."
  (
    "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/replay_sonic_latent_npz_zmq_v4.py" \
      --npz "${MOTION_NPZ}" \
      --zmq-port 5556 \
      --fps "${CONTROL_FPS}" \
      --max-frames "${MAX_FRAMES}" \
      --start-delay-s 0.5 \
      --hand-ramp-frames "${HAND_RAMP_FRAMES:-0}" \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
elif [[ -n "${CHUNK_STUDENT_CKPT:-}" && "${USE_CLOSED_LOOP:-0}" == "1" ]]; then
  log_step "starting distill ChunkStudent closed-loop (g1_debug body + commanded hand)..."
  (
    cd "${PHI0_ROOT}"
    CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES}" \
    "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/phi0_chunk_student_sonic_closed_loop_zmq.py" \
      --ckpt "${CHUNK_STUDENT_CKPT}" \
      --ref-root "${UNIFIED_ROOT}" \
      --ref-start "${REF_START:-0}" \
      --max-frames "${MAX_FRAMES}" \
      --zmq-host 127.0.0.1 \
      --zmq-port 5556 \
      --state-zmq-host 127.0.0.1 \
      --state-zmq-port 5557 \
      --fps "${CONTROL_FPS}" \
      --start-delay-s 0.5 \
      --hand-ramp-frames "${HAND_RAMP_FRAMES:-0}" \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
elif [[ -n "${CHECKPOINT}" && "${USE_CLOSED_LOOP:-0}" == "1" ]]; then
  log_step "starting Phi-0 GT closed-loop (async re-infer; RECORD_DIR=${RECORD_DIR:-off})..."
  CL_RTC_ARGS=()
  if [[ "${USE_RTC}" == "1" ]]; then
    CL_RTC_ARGS+=(--rtc)
  elif [[ "${USE_RTC}" == "0" ]]; then
    CL_RTC_ARGS+=(--no-rtc)
  fi
  CL_RECORD_ARGS=()
  if [[ -n "${RECORD_DIR:-}" ]]; then
    mkdir -p "${RECORD_DIR}"
    CL_RECORD_ARGS+=(--record-dir "${RECORD_DIR}")
  fi
  CL_PROPRIO_ARGS=()
  # T3 / 810short: body qpos + revo2 hands from sim g1_debug only (no dataset proprio).
  # Do NOT --wait-robot-proprio here: publisher starts before deploy publishes g1_debug;
  # --seed-proprio covers the gap until LowState/g1_debug is live.
  if [[ "${PROPRIO_SOURCE:-robot}" == "robot" ]]; then
    CL_PROPRIO_ARGS+=(--no-gt-proprio-fallback)
  fi
  (
    cd "${PHI0_ROOT}"
    CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES}" \
    "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/phi0_sonic_closed_loop_zmq.py" \
      --checkpoint "${CHECKPOINT}" \
      --config-name "${CONFIG_NAME}" \
      --prompt "${PROMPT:-pick tissue}" \
      --camera-source "${CAMERA_SOURCE}" \
      --gt-camera-episode "${UNIFIED_EP}" \
      --gt-repo-id "${GT_REPO_ID:-pick_tissue_xperience_unified}" \
      --control-fps "${CONTROL_FPS}" \
      --inference-rate "${INFERENCE_RATE:-0}" \
      --motion-seconds "${MOTION_SECONDS}" \
      --proprio-source "${PROPRIO_SOURCE:-robot}" \
      --seed-proprio \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}" \
      --stream-flag "${STREAM_FLAG}" \
      --no-deploy-keyboard \
      --zmq-host 127.0.0.1 \
      --zmq-port 5556 \
      --state-zmq-host 127.0.0.1 \
      --state-zmq-port 5557 \
      ${NO_EPISODE_PROMPT:+--no-episode-prompt} \
      "${CL_PROPRIO_ARGS[@]}" \
      "${CL_RTC_ARGS[@]}" \
      "${CL_RECORD_ARGS[@]}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
elif [[ -n "${CHECKPOINT}" ]]; then
  log_step "starting Phi-0 model publisher..."
  PUBLISHER_ARGS=(
    --config-name "${CONFIG_NAME}"
    --episode-idx "${UNIFIED_EP}"
    --zmq-port 5556
    --control-fps "${CONTROL_FPS}"
    --motion-seconds "${MOTION_SECONDS}"
    --max-frames "${MAX_FRAMES}"
    --start-delay-s 0.5
    --arm-flag "${ARM_FLAG}"
    --ready-flag "${REPLAY_READY}"
    --ready-timeout-s "${REPLAY_READY_TIMEOUT_S}"
  )
  if [[ "${USE_PRECOMPUTE_NPZ}" == "1" ]]; then
    PUBLISHER_ARGS=(--precompute-in "${PRECOMPUTE_NPZ}" "${PUBLISHER_ARGS[@]}")
  else
    PUBLISHER_ARGS=(--checkpoint "${CHECKPOINT}" "${PUBLISHER_ARGS[@]}")
  fi
  # USE_RTC unset → follow model.rtc.enabled (unified yaml default on).
  # USE_RTC=0/1 force --no-rtc / --rtc for LOCKED gold repro etc.
  if [[ "${USE_RTC}" == "1" ]]; then
    PUBLISHER_ARGS+=(--rtc)
    log_step "publisher RTC forced on (config=${CONFIG_NAME})"
  elif [[ "${USE_RTC}" == "0" ]]; then
    PUBLISHER_ARGS+=(--no-rtc)
    log_step "publisher RTC forced off (config=${CONFIG_NAME})"
  else
    log_step "publisher RTC follow model cfg (config=${CONFIG_NAME})"
  fi
  (
    cd "${PHI0_ROOT}"
    "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/phi0_sonic_latent_zmq_publisher.py" \
      "${PUBLISHER_ARGS[@]}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
else
  log_step "starting GT replay publisher..."
  (
    "${PHI0_PY}" "${PHI0_ROOT}/tools/data/replay_pick_tissue_sonic_latent_zmq_v4.py" \
      --parquet "${UNIFIED_PARQUET}" \
      --token-source "${TOKEN_SOURCE}" \
      --hand-source "${HAND_SOURCE:-auto}" \
      --valid-parquet-for-hands "${VALID_PARQUET}" \
      --zmq-port 5556 \
      --fps "${CONTROL_FPS}" \
      --max-frames "${MAX_FRAMES}" \
      --start-frame "${GT_DECODER_HISTORY_START_FRAME:-0}" \
      --start-delay-s 0.5 \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}" \
      ${GT_DECODER_HISTORY_GO:+--history-seed-go-path "${GT_DECODER_HISTORY_GO}"} \
      ${GT_DECODER_HISTORY_ACK:+--history-seed-ack-path "${GT_DECODER_HISTORY_ACK}"} \
      ${GT_DECODER_FRAME_PATH:+--gt-decoder-frame-path "${GT_DECODER_FRAME_PATH}"}
  ) > "${LOG_DIR}/replay.log" 2>&1 &
fi
REPLAY_PID=$!
log_step "replay pid=${REPLAY_PID} log=${LOG_DIR}/replay.log"
if [[ -n "${MOTION_NPZ:-}" ]]; then
  wait_log "${LOG_DIR}/replay.log" "bound tcp://" 30 "npz replay bound" || require_pid "${REPLAY_PID}" "npz replay" "${LOG_DIR}/replay.log"
elif [[ -n "${CHECKPOINT}" ]] || [[ -n "${CHUNK_STUDENT_CKPT:-}" ]]; then
  # Precomputed path binds tcp in ~1s; inline VLM+inference may take minutes.
  wait_timeout=600
  if [[ "${USE_PRECOMPUTE_NPZ}" == "1" ]]; then
    wait_timeout=60
  fi
  for i in $(seq 1 "${wait_timeout}"); do
    require_pid "${REPLAY_PID}" "model publisher" "${LOG_DIR}/replay.log"
    if grep -qE "bound tcp://" "${LOG_DIR}/replay.log" 2>/dev/null; then
      log_step "model publisher ready (${i}s)"
      break
    fi
    if grep -qE "Traceback \(most recent call last\)" "${LOG_DIR}/replay.log" 2>/dev/null; then
      log_step "model publisher failed during load"
      tail -30 "${LOG_DIR}/replay.log"
      exit 1
    fi
    if (( i % 15 == 0 )); then
      log_step "model publisher loading... (${i}/${wait_timeout}s)"
      tail -3 "${LOG_DIR}/replay.log" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
    fi
    sleep 1
  done
else
  wait_log "${LOG_DIR}/replay.log" "bound tcp://" 30 "replay bound" || require_pid "${REPLAY_PID}" "replay" "${LOG_DIR}/replay.log"
fi

# 3) C++ deploy zmq_manager (TensorRT init ~1–3 min — looks idle in terminal)
log_step "starting deploy TensorRT (pid pending, log=${LOG_DIR}/deploy.log)..."
BIN="${DEPLOY}/target/release/g1_deploy_onnx_ref"
POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
OBS_CONFIG="${DEPLOY_OBS_CONFIG:-policy/${POLICY_DIR}/observation_config.yaml}"
ENCODER_FILE="${DEPLOY_ENCODER_FILE:-policy/${POLICY_DIR}/model_encoder.onnx}"
rm -f "${DEPLOY_FIFO}"; mkfifo "${DEPLOY_FIFO}"
DEPLOY_ENCODER_ARGS=()
if [[ "${DEPLOY_SKIP_ENCODER:-0}" != "1" && -f "${DEPLOY}/${ENCODER_FILE}" ]]; then
  DEPLOY_ENCODER_ARGS=(--encoder-file "${ENCODER_FILE}")
fi
(
  cd "${DEPLOY}"
  CUDA_VISIBLE_DEVICES="${DEPLOY_CUDA_VISIBLE_DEVICES}" \
  "${BIN}" lo "policy/${POLICY_DIR}/model_decoder.onnx" reference/example/ \
    --obs-config "${OBS_CONFIG}" \
    "${DEPLOY_ENCODER_ARGS[@]}" \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all \
    --zmq-host 127.0.0.1 --zmq-port 5556 --disable-crc-check < "${DEPLOY_FIFO}"
) > "${LOG_DIR}/deploy.log" 2>&1 &
DEPLOY_PID=$!
exec 3>"${DEPLOY_FIFO}"
log_step "deploy pid=${DEPLOY_PID} cuda=${DEPLOY_CUDA_VISIBLE_DEVICES}"
if ! wait_log "${LOG_DIR}/deploy.log" "Init Done" "${DEPLOY_INIT_TIMEOUT_S}" "deploy Init Done"; then
  require_pid "${DEPLOY_PID}" "deploy" "${LOG_DIR}/deploy.log"
  exit 1
fi

# Align with phi-0-wbc-newton:
# ARM(+start_streamed) → ENTER → settle → READY → first token → RECORD.
# Studio interactive (web buttons / terminal): y → ] → Enter
log_step "deploy Init Done; arming via replay/publisher..."
if [[ "${STUDIO_INTERACTIVE:-0}" == "1" ]]; then
  studio_set_phase "init_done" "Init Done — terminal 输入 ] 站立"
  studio_wait_cmd "stand" "]" "stand_up"
  send_deploy_key ']'
  # terminal 流程为 y → ] → Enter；Policy 按钮仍可发 policy（此处自动进入 arm）
  studio_set_phase "standing" "sent ] — arming publisher / CONTROL"
  studio_set_phase "policy_starting" "arming publisher / CONTROL"
fi

if [[ "${USE_CLOSED_LOOP:-0}" == "1" ]]; then
  touch "${STREAM_FLAG}"
  sleep 0.5
fi

touch "${ARM_FLAG}"
if ! wait_log "${LOG_DIR}/deploy.log" "transitioning to CONTROL state|ZMQManager.*Planner enabled" 90 "deploy arm/CONTROL"; then
  require_pid "${REPLAY_PID}" "replay/publisher" "${LOG_DIR}/replay.log"
  log_step "deploy did not arm; tails:"
  tail -30 "${LOG_DIR}/deploy.log"
  tail -15 "${LOG_DIR}/replay.log"
  exit 1
fi
if ! wait_log "${LOG_DIR}/deploy.log" "transitioning to CONTROL state" 60 "deploy CONTROL"; then
  log_step "deploy never entered CONTROL"
  tail -40 "${LOG_DIR}/deploy.log"
  exit 1
fi
# Wait for publisher start_streamed before ENTER (CL/GT log this after arm).
if ! wait_log "${LOG_DIR}/replay.log" "sent ZMQ command start|planner -> streamed" 30 "publisher start_streamed"; then
  log_step "WARN: no publisher start_streamed yet; trying ENTER anyway"
  tail -15 "${LOG_DIR}/replay.log" || true
fi

if [[ "${STUDIO_INTERACTIVE:-0}" == "1" ]]; then
  studio_set_phase "policy_ready" "Policy armed — terminal 回车 / Enter 开流"
  studio_wait_cmd "stream" "enter" "stream_p" "p" "go"
  if [[ "${STUDIO_LAST_CMD}" == "stream_p" || "${STUDIO_LAST_CMD}" == "p" ]]; then
    send_deploy_key 'p'
    sleep 0.2
  fi
  studio_set_phase "streaming" "enabling ZMQ stream (Enter)"
fi

log_step "deploy in CONTROL; enabling ZMQ streaming (ENTER)..."
for attempt in $(seq 1 12); do
  grep -q "ZMQ STREAMING MODE: ENABLED" "${LOG_DIR}/deploy.log" 2>/dev/null && break
  printf '\n' >&3
  sleep 2
done
if ! grep -q "ZMQ STREAMING MODE: ENABLED" "${LOG_DIR}/deploy.log" 2>/dev/null; then
  echo "[sonic_latent] ZMQ streaming not enabled"
  tail -20 "${LOG_DIR}/deploy.log"
  exit 1
fi
send_deploy_key 'I'
sleep 0.5
if [[ "${STUDIO_INTERACTIVE:-0}" == "1" ]]; then
  studio_set_phase "streaming" "ZMQ STREAMING MODE ENABLED"
fi

log_step "wait deploy lowcmd active (sim standing under deploy)..."
if ! wait_log "${LOG_DIR}/sim.log" "deploy lowcmd active" 120 "deploy lowcmd"; then
  tail -30 "${LOG_DIR}/sim.log"
  exit 1
fi
require_replay
LIVE_SIM_HISTORY_FRAMES="${LIVE_SIM_HISTORY_FRAMES:-0}"
if [[ "${LIVE_SIM_HISTORY_FRAMES}" -gt 0 ]]; then
  LIVE_SIM_HISTORY_SECONDS="$(${PHI0_PY} - <<PY
print(${LIVE_SIM_HISTORY_FRAMES} / float(${CONTROL_FPS}))
PY
)"
  log_step "collecting ${LIVE_SIM_HISTORY_FRAMES} live SIM LowState frames (${LIVE_SIM_HISTORY_SECONDS}s) for SONIC decoder history..."
  sleep "${LIVE_SIM_HISTORY_SECONDS}"
  require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
  require_pid "${DEPLOY_PID}" "deploy" "${LOG_DIR}/deploy.log"
  log_step "live SIM decoder history ready (${LIVE_SIM_HISTORY_FRAMES} frames)"
else
  log_step "wait sim stable (no new FALL for ${RECORD_STABLE_S}s)..."
  stable_deadline=$(( $(date +%s) + 90 ))
  while (( $(date +%s) < stable_deadline )); do
    require_replay
    fall_count="$(_fall_count "${LOG_DIR}/sim.log")"
    sleep "${RECORD_STABLE_S}"
    fall_after="$(_fall_count "${LOG_DIR}/sim.log")"
    if [[ "${fall_after}" -le "${fall_count}" ]]; then
      log_step "sim stable (${RECORD_STABLE_S}s without new FALL, total_fall=${fall_after})"
      break
    fi
    log_step "sim still settling (FALL ${fall_count} -> ${fall_after}), waiting..."
  done
fi
log_step "settle ${RECORD_SETTLE_S}s after deploy active before record..."
sleep "${RECORD_SETTLE_S}"

# When RSI snaps on record-start, recording MUST arm before first student token
# (else proprio/body0 is still post-settle stand while video later snaps to walk).
# Same ordering as GT_DECODER_HISTORY_PREFILL.
_RECORD_BEFORE_REPLAY=0
if [[ "${GT_DECODER_HISTORY_PREFILL}" == "1" ]]; then
  _RECORD_BEFORE_REPLAY=1
elif [[ "${QPOS_RSI}" == "1" && "${QPOS_RSI_SNAP_ON_RECORD:-1}" == "1" ]]; then
  _RECORD_BEFORE_REPLAY=1
fi
if [[ "${_RECORD_BEFORE_REPLAY}" == "1" ]]; then
  touch "${RECORD_START}"
  if ! wait_log "${LOG_DIR}/sim.log" "sim_record] started" 60 "sim record start"; then
    log_step "record flag paths: start=${RECORD_START} stop=${RECORD_STOP}"
    tail -20 "${LOG_DIR}/sim.log"; exit 1
  fi
  log_step "record armed before replay (RSI snap-on-record / GT hist prefill)"
fi
touch "${REPLAY_READY}"
log_step "replay go; wait first token tx..."
_first_tx=0
for _ in $(seq 1 300); do
  if grep -qE 'first output frame=|tx frame=1 |frame 1/' "${LOG_DIR}/replay.log" 2>/dev/null; then
    _first_tx=1
    break
  fi
  require_replay
  sleep 0.2
done
if [[ "${_first_tx}" -ne 1 ]]; then
  log_step "WARN: no first tx within 60s; continuing"
fi
if [[ "${_RECORD_BEFORE_REPLAY}" != "1" ]]; then
  touch "${RECORD_START}"
  if ! wait_log "${LOG_DIR}/sim.log" "sim_record] started" 60 "sim record start"; then
    log_step "record flag paths: start=${RECORD_START} stop=${RECORD_STOP}"
    ls -la "${RECORD_START}" "${REPLAY_READY}" 2>&1 || true
    tail -20 "${LOG_DIR}/sim.log"; exit 1
  fi
fi
log_step "recording + streaming ${MAX_FRAMES} frames @ ${CONTROL_FPS}Hz..."
require_replay

while kill -0 "${REPLAY_PID}" 2>/dev/null; do
  require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
  require_pid "${DEPLOY_PID}" "deploy" "${LOG_DIR}/deploy.log"
  sleep 0.5
done
wait "${REPLAY_PID}" || true
touch "${RECORD_STOP}"
for _ in $(seq 1 60); do
  grep -q "sim_record] saved" "${LOG_DIR}/sim.log" 2>/dev/null && break
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    log_step "WARN: sim exited before record save; see ${LOG_DIR}/sim.log"
    break
  fi
  sleep 0.5
done
if [[ -n "${MOTION_NPZ:-}" ]]; then
  token_rx="$(grep -c "Received 64D token" "${LOG_DIR}/deploy.log" 2>/dev/null || echo 0)"
  log_step "deploy received ${token_rx} x 64D token frames"
fi

# ponytail: OpenCV mp4v is unreadable on many players; remux to H.264
_remux_mp4_h264() {
  local src="$1"
  local dst="${src%.mp4}_h264.mp4"
  if command -v ffmpeg >/dev/null 2>&1 && [[ -f "${src}" ]] && [[ $(stat -c%s "${src}") -gt 1000 ]]; then
    if ffmpeg -y -loglevel error -i "${src}" \
      -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart \
      "${dst}"; then
      mv "${dst}" "${src}"
      echo "[sonic_latent] remuxed to H.264 (libx264): ${src}"
      return 0
    fi
  fi
  return 1
}
OUT_H264="${OUT_MP4%.mp4}_h264.mp4"
if [[ ! -f "${OUT_MP4}" ]]; then
  _rel="${OUT_MP4#${PHI0_ROOT}/}"
  if [[ -f "${GR00T_ROOT}/${_rel}" ]]; then
    cp "${GR00T_ROOT}/${_rel}" "${OUT_MP4}"
    echo "[sonic_latent] copied mp4 from GR00T cwd -> ${OUT_MP4}"
  fi
fi
_remux_mp4_h264 "${OUT_MP4}" || true

echo "[sonic_latent] replay tail:"
tail -8 "${LOG_DIR}/replay.log" || true
echo "[sonic_latent] deploy token/hand:"
grep -E "64D token|hand joints set" "${LOG_DIR}/deploy.log" | tail -6 || true
echo "[sonic_latent] video: ${OUT_MP4}"
ls -lh "${OUT_MP4}" 2>/dev/null || true
echo "[sonic_latent] done work_dir=${WORK_DIR}"
