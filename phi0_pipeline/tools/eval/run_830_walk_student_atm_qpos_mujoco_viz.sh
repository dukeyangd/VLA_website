#!/usr/bin/env bash
# 830 walk: MuJoCo driven by Newton-same **Python ATM decode** joint targets
# (q_cmd), NOT sonic TRT decoder. Packs IsaacLab q_cmd → MuJoCo observation_qpos
# and replays via trusted qpos ZMQ PD loop (egypt LOCKED deploy path).
#
#   INFER_NPZ=.../infer_qpos_traj_student.npz EP=125 \
#     bash tools/eval/run_830_walk_student_atm_qpos_mujoco_viz.sh
#   SOURCE=q_star bash tools/eval/run_830_walk_student_atm_qpos_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
DEPLOY_PY="${DEPLOY_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python3}"
GT_REPLAY="${GT_REPLAY_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/global_unittest/sonic_latent_gt_replay}"

EP="${EP:-125}"
SOURCE="${SOURCE:-q_cmd}"  # q_cmd = student ATM; q_star = ATM(z*); q_act = tracked
INFER_NPZ="${INFER_NPZ:-${PHI0_ROOT}/experiments/830_walk_student_cl_newton_ep${EP}_20260827_015707/infer_qpos_traj_student.npz}"
TAG="${TAG:-830_walk_student_atm_qpos_${SOURCE}_ep${EP}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
WORK_DIR="${WORK_DIR:-${OUT}/mujoco_qpos}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

if [[ ! -f "${INFER_NPZ}" ]]; then
  echo "[830_atm_qpos] missing INFER_NPZ=${INFER_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${GT_REPLAY}/replay_gt_qpos_loop.sh" ]]; then
  echo "[830_atm_qpos] missing ${GT_REPLAY}/replay_gt_qpos_loop.sh" >&2
  exit 1
fi

mkdir -p "${OUT}/logs" "${WORK_DIR}"
QPOS_NPZ="${WORK_DIR}/qpos.npz"
MP4="${WORK_DIR}/gt_qpos_replay.mp4"

echo "[830_atm_qpos] infer=${INFER_NPZ}"
echo "[830_atm_qpos] source=${SOURCE} (Python ATM joints → MuJoCo qpos PD; no TRT)"
echo "[830_atm_qpos] out=${OUT} gpu=${CUDA_VISIBLE_DEVICES}"

echo "[830_atm_qpos] === pack ${SOURCE} + root_act → qpos.npz ==="
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_qpos_npz_from_infer.py" \
  "${INFER_NPZ}" \
  --out "${QPOS_NPZ}" \
  --source "${SOURCE}" \
  --root-from-infer \
  >"${OUT}/logs/pack_qpos.log" 2>&1
tail -5 "${OUT}/logs/pack_qpos.log"

FK_MP4="${OUT}/fk_${SOURCE}_mujoco.mp4"
echo "[830_atm_qpos] === FK render ${SOURCE} (no physics) ==="
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/render_qact_mujoco.py" \
  "${INFER_NPZ}" --out "${FK_MP4}" --source "${SOURCE}" \
  >"${OUT}/logs/fk_render.log" 2>&1 &
FK_PID=$!

echo "[830_atm_qpos] === MuJoCo qpos PD replay ==="
nohup env \
  SKIP_NPZ_EXTRACT=1 \
  EGO_VIDEO= \
  GT_PANEL_LAYOUT=inset \
  PROTO_MOCAP_REPLAY=1 \
  REPLAY_SETTLE="${REPLAY_SETTLE:-120}" \
  WORK_DIR="${WORK_DIR}" \
  QPOS_NPZ="${QPOS_NPZ}" \
  PHI0_PY="${DEPLOY_PY}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  KILL_EXISTING="${KILL_EXISTING:-1}" \
  bash "${GT_REPLAY}/replay_gt_qpos_loop.sh" \
  >"${OUT}/logs/mujoco_qpos.log" 2>&1 &

echo "pid=$! fk_pid=${FK_PID} log=${OUT}/logs/mujoco_qpos.log"
echo "mp4=${MP4}"
echo "fk_mp4=${FK_MP4}"
