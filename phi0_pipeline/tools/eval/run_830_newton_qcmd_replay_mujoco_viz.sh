#!/usr/bin/env bash
# Isolation: Newton q_cmd (IL→MJ) → MuJoCo LowCmd PD. No student / ATM.
#   EP=125 bash tools/eval/run_830_newton_qcmd_replay_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

EP="${EP:-125}"
TAG="${TAG:-830_newton_qcmd_replay_mujoco_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy-py310-bak/bin/python}"
NEWTON_NPZ="${NEWTON_NPZ:-${PHI0_ROOT}/experiments/830_walk_student_cl_newton_ep125_20260827_004942/infer_qpos_traj_student.npz}"
REF="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified}"
PARQUET="${REF}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet"
CONTROL_FPS="${CONTROL_FPS:-50}"
RECORD_FPS="${RECORD_FPS:-50}"
DOMAIN_ID="${DOMAIN_ID:-0}"
DDS_IFACE="${DDS_IFACE:-eth0}"
GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/subpackages}"
VENV_SIM="${VENV_SIM:-${PHI0_ROOT}/.venv_sim}"
SIM_WARMUP_S="${SIM_WARMUP_S:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export GR00T_ROOT
export PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD="${PHI0_MUJOCO_KEEP_POSE_ON_LOWCMD:-1}"
export PHI0_MUJOCO_DEPLOY_FALL_GRACE_S="${PHI0_MUJOCO_DEPLOY_FALL_GRACE_S:-0}"
export PHI0_MUJOCO_DISABLE_FALL_RESET="${PHI0_MUJOCO_DISABLE_FALL_RESET:-1}"
# Isolation defaults: band off, deploy KP/KD unscaled (override via env only if needed).
export PHI0_MUJOCO_KEEP_BAND_ON_LOWCMD="${PHI0_MUJOCO_KEEP_BAND_ON_LOWCMD:-0}"
export PHI0_MUJOCO_BAND_RELEASE_AFTER_S="${PHI0_MUJOCO_BAND_RELEASE_AFTER_S:-}"
export PHI0_ATM_KP_SCALE="${PHI0_ATM_KP_SCALE:-1.0}"
export PHI0_ATM_KD_SCALE="${PHI0_ATM_KD_SCALE:-1.0}"
export PYTHONPATH="${PHI0_ROOT}/src:${GR00T_ROOT}:${PYTHONPATH:-}"

[[ -f "${NEWTON_NPZ}" ]] || { echo "[newton_replay] missing ${NEWTON_NPZ}" >&2; exit 1; }
[[ -f "${PARQUET}" ]] || { echo "[newton_replay] missing ${PARQUET}" >&2; exit 1; }

WORK_DIR="${OUT}/mujoco"
LOG_DIR="${OUT}/logs"
MP4="${OUT}/newton_qcmd_replay_cl.mp4"
NPZ="${OUT}/newton_qcmd_replay_traj.npz"
RECORD_START="${WORK_DIR}/.record_start"
RECORD_STOP="${WORK_DIR}/.record_stop"
mkdir -p "${WORK_DIR}" "${LOG_DIR}"
rm -f "${RECORD_START}" "${RECORD_STOP}"

echo "[newton_replay] newton=${NEWTON_NPZ}"
echo "[newton_replay] ep=${EP} out=${OUT} keep_band=${PHI0_MUJOCO_KEEP_BAND_ON_LOWCMD} release_after=${PHI0_MUJOCO_BAND_RELEASE_AFTER_S:-none}"

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
echo "[newton_replay] sim pid=${SIM_PID}"

for i in $(seq 1 180); do
  if grep -qE "sim loop starting|qpos RSI|SimWrapper" "${LOG_DIR}/sim.log" 2>/dev/null; then
    echo "[newton_replay] sim up (${i}s)"
    break
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "[newton_replay] sim died" >&2
    tail -50 "${LOG_DIR}/sim.log" >&2
    exit 1
  fi
  sleep 1
done
sleep "${SIM_WARMUP_S}"

(
  cd "${PHI0_ROOT}"
  "${PHI0_PY}" "${PHI0_ROOT}/tools/deploy/replay_newton_qcmd_lowcmd.py" \
    --newton-npz "${NEWTON_NPZ}" \
    --rsi-npz "${QPOS_RSI_NPZ}" \
    --fps "${CONTROL_FPS}" \
    --start-delay-s 0.3 \
    --rsi-hold-frames 10 \
    --domain-id "${DOMAIN_ID}" \
    --dds-interface "${DDS_IFACE}" \
    --out-npz "${NPZ}" \
    --kp-scale "${PHI0_ATM_KP_SCALE}" \
    --kd-scale "${PHI0_ATM_KD_SCALE}"
) >"${LOG_DIR}/newton_replay.log" 2>&1 &
CTRL_PID=$!
echo "[newton_replay] controller pid=${CTRL_PID}"

for i in $(seq 1 120); do
  if grep -qE "\[newton_replay\] t=0 " "${LOG_DIR}/newton_replay.log" 2>/dev/null; then
    echo "[newton_replay] streaming (${i}s)"
    break
  fi
  if ! kill -0 "${CTRL_PID}" 2>/dev/null; then
    echo "[newton_replay] controller died early" >&2
    tail -40 "${LOG_DIR}/newton_replay.log" >&2
    exit 1
  fi
  sleep 1
done

touch "${RECORD_START}"
echo "[newton_replay] recording ${MP4}"

while kill -0 "${CTRL_PID}" 2>/dev/null; do sleep 1; done
wait "${CTRL_PID}" || true
sleep 1
touch "${RECORD_STOP}"
sleep 2
kill "${SIM_PID}" 2>/dev/null || true
wait "${SIM_PID}" 2>/dev/null || true

if [[ -f "${MP4}" ]]; then
  TMP="${MP4}.h264.mp4"
  if ffmpeg -y -i "${MP4}" -c:v libx264 -pix_fmt yuv420p -an "${TMP}" >/dev/null 2>&1; then
    mv -f "${TMP}" "${MP4}"
    echo "[newton_replay] re-encoded ${MP4} -> H.264"
  fi
fi

echo "[newton_replay] done"
echo "mp4=${MP4}"
echo "npz=${NPZ}"
echo "log=${LOG_DIR}/newton_replay.log"
