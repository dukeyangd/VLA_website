#!/usr/bin/env bash
# Trusted qpos eval: student q_cmd → same ZMQ v1 deploy as GT qpos replay.
# See .cursor/rules/phi0-qpos-eval.mdc
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
sonic_resolve_python
PHI0_PY="${PHI0_PY:-${PYTHON_BIN}}"
GT_REPLAY="${GT_REPLAY_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus}/global_unittest/sonic_latent_gt_replay"

STUDENT_CKPT="${STUDENT_CKPT:?}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_g1csv_aligned_phi0}"
REF_START="${REF_START:-80282}"
MAX_REF_FRAMES="${MAX_REF_FRAMES:-278}"
NUM_STEPS="${NUM_STEPS:-${MAX_REF_FRAMES}}"
GT_QPOS_NPZ="${GT_QPOS_NPZ:-${GT_REPLAY}/runs/gt_qpos_egypt_ep199_20260722_110640/qpos.npz}"
TAG="${TAG:-vla_qpos_deploy_$(date +%Y%m%d_%H%M%S)}"
INFER_OUT="${INFER_OUT:-${PHI0_ROOT}/experiments/${TAG}_infer}"
RUN_DIR="${RUN_DIR:-${GT_REPLAY}/runs/${TAG}}"

mkdir -p "${INFER_OUT}" "${RUN_DIR}"

echo "[qpos_deploy_eval] 1/3 infer qpos_student"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PHI0_DISTILL_OUT="${INFER_OUT}" STUDENT_CKPT="${STUDENT_CKPT}" \
NUM_ENVS=1 NUM_STEPS="${NUM_STEPS}" HORIZON="${HORIZON:-8}" \
CONTROL=qpos_student MAX_REF_FRAMES="${MAX_REF_FRAMES}" \
REF_START="${REF_START}" REF_ROOT="${REF_ROOT}" RECORD_VIDEO=0 \
bash "${PHI0_ROOT}/tools/eval/run_sonic_online_infer_isaac.sh"

NPZ="${INFER_OUT}/infer_qpos_traj_qpos_student.npz"
echo "[qpos_deploy_eval] 2/3 pack q_cmd (+ GT root_action)"
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_qpos_npz_from_infer.py" \
  "${NPZ}" --out "${RUN_DIR}/qpos.npz" --source q_cmd \
  --gt-qpos-npz "${GT_QPOS_NPZ}"

echo "[qpos_deploy_eval] 3/3 deploy replay"
SKIP_NPZ_EXTRACT=1 EGO_VIDEO= GT_PANEL_LAYOUT=inset \
PROTO_MOCAP_REPLAY=1 REPLAY_SETTLE="${REPLAY_SETTLE:-120}" \
WORK_DIR="${RUN_DIR}" QPOS_NPZ="${RUN_DIR}/qpos.npz" \
PHI0_PY="${PHI0_PY}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
KILL_EXISTING="${KILL_EXISTING:-1}" \
bash "${GT_REPLAY}/replay_gt_qpos_loop.sh"

echo "[qpos_deploy_eval] done: ${RUN_DIR}/gt_qpos_replay.mp4"
