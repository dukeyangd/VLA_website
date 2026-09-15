#!/usr/bin/env bash
# 830 walk MuJoCo CL: ChunkStudent → ONNX ATM decode → DDS LowCmd PD (Newton-style).
# No C++ TRT deploy / latent ZMQ. RSI to frame0, prime hist, control ASAP.
#
#   EP=125 bash tools/eval/run_830_walk_student_cl_mujoco_atm_pd_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830_walk_nosim_h14_b32_ddp8_e10_20260826_115006/phi0_student_last.pt}"
EP="${EP:-125}"
PHI0_CL_Z_SOURCE="${PHI0_CL_Z_SOURCE:-student}"  # student | disk (dataset z_ref GT)
TAG="${TAG:-830_walk_${PHI0_CL_Z_SOURCE}_cl_mujoco_atm_pd_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy-py310-bak/bin/python}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"
CONTROL_FPS="${CONTROL_FPS:-50}"
RECORD_FPS="${RECORD_FPS:-50}"
SIM_WARMUP_S="${SIM_WARMUP_S:-0}"
RECORD_SETTLE_S="${RECORD_SETTLE_S:-0}"
DOMAIN_ID="${DOMAIN_ID:-0}"
DDS_IFACE="${DDS_IFACE:-eth0}"
GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
VENV_SIM="${VENV_SIM:-${PHI0_ROOT}/.venv_sim}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export GR00T_ROOT
export PHI0_HAND_MODE="${PHI0_HAND_MODE:-dex3}"
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_CL_Z_SOURCE
if [[ "${PHI0_CL_Z_SOURCE}" == "disk" || "${PHI0_CL_Z_SOURCE}" == "gt" ]]; then
  export USE_RTC="${USE_RTC:-0}"
  export PHI0_CL_HAND_OBS="${PHI0_CL_HAND_OBS:-tape}"
else
  export USE_RTC="${USE_RTC:-1}"
  export PHI0_CL_HAND_OBS="${PHI0_CL_HAND_OBS:-commanded}"
fi
export PHI0_TRAIN_MODE="${PHI0_TRAIN_MODE:-online_vlm}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE="${PHI0_USE_VLM_FRAME_LATENT_CACHE:-1}"
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO="${PHI0_VLM_FRAME_CACHE_SKIP_VIDEO:-1}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-0}"
export PHI0_RTC_INFERENCE_DELAY="${PHI0_RTC_INFERENCE_DELAY:-6}"
export PHI0_RTC_EXECUTION_HORIZON="${PHI0_RTC_EXECUTION_HORIZON:-26}"
# Keep RSI walk pose on first LowCmd; do not 8s stand-settle re-pin (Newton ATM-PD parity).
export PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD="${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD:-1}"
export PHI0_MUJOCO_DEPLOY_FALL_GRACE_S="${PHI0_MUJOCO_DEPLOY_FALL_GRACE_S:-0}"
export PHI0_MUJOCO_DISABLE_FALL_RESET="${PHI0_MUJOCO_DISABLE_FALL_RESET:-1}"

PHI0_CL_DECODE="${PHI0_CL_DECODE:-atm}"  # atm (default) | trt (legacy ZMQ latent)

if [[ "${PHI0_CL_DECODE}" == "trt" ]]; then
  echo "[830_atm_pd] PHI0_CL_DECODE=trt: use tools/eval/run_830_walk_student_cl_mujoco_viz.sh (legacy TRT path)" >&2
  exit 2
fi

export PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}"

if [[ "${PHI0_CL_Z_SOURCE}" != "disk" && "${PHI0_CL_Z_SOURCE}" != "gt" && ! -f "${STUDENT_CKPT}" ]]; then
  echo "[830_atm_pd] missing ckpt ${STUDENT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[830_atm_pd] missing parquet ${PARQUET}" >&2
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

WORK_DIR="${OUT}/mujoco"
LOG_DIR="${OUT}/logs"
MP4="${OUT}/atm_pd_cl.mp4"
NPZ="${OUT}/atm_pd_traj.npz"
RECORD_START="${WORK_DIR}/.record_start"
RECORD_STOP="${WORK_DIR}/.record_stop"
mkdir -p "${WORK_DIR}" "${LOG_DIR}"
rm -f "${RECORD_START}" "${RECORD_STOP}"

echo "[830_atm_pd] z_source=${PHI0_CL_Z_SOURCE} ckpt=${STUDENT_CKPT}"
echo "[830_atm_pd] ep=${EP} ref_start=${REF_START} frames=${NFRAMES} (ATM→LowCmd PD, no TRT)"
echo "[830_atm_pd] hand_obs=${PHI0_CL_HAND_OBS} rtc=${USE_RTC} out=${OUT}"

QPOS_RSI_NPZ="${WORK_DIR}/qpos_rsi_frame0.npz"
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/export_qpos_rsi_npz_from_unified.py" \
  "${PARQUET}" --out "${QPOS_RSI_NPZ}" --frame 0 \
  >"${LOG_DIR}/export_qpos_rsi.log" 2>&1

cleanup() {
  [[ -n "${CTRL_PID:-}" ]] && kill "${CTRL_PID}" 2>/dev/null || true
  [[ -n "${SIM_PID:-}" ]] && kill "${SIM_PID}" 2>/dev/null || true
  touch "${RECORD_STOP}" 2>/dev/null || true
}
trap cleanup EXIT

bash "${PHI0_ROOT}/tools/env/fix_venv_sim.sh" >>"${LOG_DIR}/fix_venv_sim.log" 2>&1 || true

# 1) Sim only (Dex3 / no Revo2 to match 830 walk + Newton gold).
(
  cd "${GR00T_ROOT}"
  # shellcheck source=/dev/null
  source "${VENV_SIM}/bin/activate"
  python -u experiments/sonic_vla_overfit/scripts/run_sim_loop_vla_record.py \
    --no-enable-onscreen \
    --enable-offscreen \
    --no-enable-image-publish \
    --no-revo2-hand \
    --camera-port 5555 \
    --g1-debug-host 127.0.0.1 \
    --g1-debug-port 5557 \
    --g1-debug-snap --no-enable-g1-debug-overlay \
    --qpos-snap-npz "${QPOS_RSI_NPZ}" \
    --qpos-snap-frame 0 \
    --domain-id "${DOMAIN_ID}" \
    --interface "${DDS_IFACE}" \
    --record-mp4 "${MP4}" \
    --record-start-flag "${RECORD_START}" \
    --record-stop-flag "${RECORD_STOP}" \
    --record-fps "${RECORD_FPS}"
) >"${LOG_DIR}/sim.log" 2>&1 &
SIM_PID=$!
echo "[830_atm_pd] sim pid=${SIM_PID} log=${LOG_DIR}/sim.log"

for i in $(seq 1 180); do
  if grep -qE "sim loop starting|qpos RSI|SimWrapper" "${LOG_DIR}/sim.log" 2>/dev/null; then
    echo "[830_atm_pd] sim up (${i}s)"
    break
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "[830_atm_pd] sim died" >&2
    tail -50 "${LOG_DIR}/sim.log" >&2
    exit 1
  fi
  sleep 1
done
sleep "${SIM_WARMUP_S}"

# 2) ATM-PD controller (student or dataset GT z).
CTRL_ARGS=(
  --ref-root "${REF}"
  --ref-start "${REF_START}"
  --max-frames "${NFRAMES}"
  --fps "${CONTROL_FPS}"
  --start-delay-s 0.2
  --domain-id "${DOMAIN_ID}"
  --dds-interface "${DDS_IFACE}"
  --out-npz "${NPZ}"
  --publish-hands
)
if [[ "${PHI0_CL_Z_SOURCE}" != "disk" && "${PHI0_CL_Z_SOURCE}" != "gt" ]]; then
  CTRL_ARGS+=(--ckpt "${STUDENT_CKPT}")
fi
(
  cd "${PHI0_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/phi0_chunk_student_atm_lowcmd_cl.py" \
    "${CTRL_ARGS[@]}"
) >"${LOG_DIR}/atm_pd_cl.log" 2>&1 &
CTRL_PID=$!
echo "[830_atm_pd] controller pid=${CTRL_PID} log=${LOG_DIR}/atm_pd_cl.log"

# Wait for first LowCmd / first output, then record (no long stand settle).
for i in $(seq 1 120); do
  if grep -qE "first output|atm_pd_cl.*tx frame=1|release sim stand" "${LOG_DIR}/atm_pd_cl.log" 2>/dev/null; then
    echo "[830_atm_pd] controller streaming (${i}s)"
    break
  fi
  if ! kill -0 "${CTRL_PID}" 2>/dev/null; then
    echo "[830_atm_pd] controller died early" >&2
    tail -40 "${LOG_DIR}/atm_pd_cl.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${RECORD_SETTLE_S}" != "0" ]]; then
  sleep "${RECORD_SETTLE_S}"
fi
touch "${RECORD_START}"
echo "[830_atm_pd] recording ${MP4}"

TIMEOUT_S=$(( NFRAMES / CONTROL_FPS + 180 ))
for i in $(seq 1 "${TIMEOUT_S}"); do
  if ! kill -0 "${CTRL_PID}" 2>/dev/null; then
    break
  fi
  if (( i % 30 == 0 )); then
    echo "[830_atm_pd] still running (${i}/${TIMEOUT_S}s)"
    tail -2 "${LOG_DIR}/atm_pd_cl.log" 2>/dev/null | sed 's/^/[830_atm_pd]   /' || true
  fi
  sleep 1
done
touch "${RECORD_STOP}"
sleep 2
wait "${CTRL_PID}" 2>/dev/null || true
kill "${SIM_PID}" 2>/dev/null || true
wait "${SIM_PID}" 2>/dev/null || true
trap - EXIT

# OpenCV mp4v is unplayable in many viewers; rewrite as H.264 in place.
if [[ -f "${MP4}" ]] && command -v ffmpeg >/dev/null 2>&1; then
  _tmp_h264="${MP4}.h264.tmp.mp4"
  if ffmpeg -y -i "${MP4}" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast -movflags +faststart "${_tmp_h264}" >/dev/null 2>&1; then
    mv -f "${_tmp_h264}" "${MP4}"
    echo "[830_atm_pd] re-encoded ${MP4} -> H.264"
  else
    rm -f "${_tmp_h264}"
    echo "[830_atm_pd] WARN: H.264 re-encode failed; left mp4v" >&2
  fi
fi

echo "[830_atm_pd] done"
echo "mp4=${MP4}"
echo "npz=${NPZ}"
echo "log=${LOG_DIR}/atm_pd_cl.log"
if [[ -f "${MP4}" ]]; then
  ls -lh "${MP4}"
else
  echo "[830_atm_pd] WARN: mp4 missing; check logs" >&2
  tail -40 "${LOG_DIR}/sim.log" >&2 || true
  tail -40 "${LOG_DIR}/atm_pd_cl.log" >&2 || true
  exit 1
fi
