#!/usr/bin/env bash
# Visualize a trained online-distill checkpoint's *own* closed-loop output
# (the 64D sonic latent it predicts) through the real deploy binary
# (g1_deploy_onnx_ref) + MuJoCo sim.
#
# Replaces the old scripts/render_infer_qpos_mp4.py path: that was a bespoke
# kinematic-only MuJoCo replay (no physics, hand-rolled dof name remapping)
# that produced ambiguous/misleading results while debugging this pipeline —
# it has been removed.
#
# Uses vla_deploy_replay/replay_vla_deploy.sh — a dedicated fork of the
# dataset-validation tool (sonic_latent_gt_replay/replay_gt_latent.sh), kept
# as a separate entry point/lock file/port range on purpose so GT-parquet
# validation runs and VLA-deployment runs never share state or interfere
# with each other.
#
# Three stages:
#   1. run_sonic_online_infer_isaac.sh (RECORD_VIDEO=0) -> infer_qpos_traj_*.npz
#      (has z_pred[T,B,64] logged every step)
#   2. build_gt_replay_tokens_from_infer_npz.py -> tokens.npz
#   3. replay_vla_deploy.sh TOKENS=tokens.npz -> vla_deploy_replay.mp4
#
# Known limitation (see build_gt_replay_tokens_from_infer_npz.py docstring):
# root translation/heading isn't reconstructed from our own output yet
# (nav_yaw=0, robot doesn't travel) — only joint articulation is judged.

set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GT_REPLAY_ROOT="${GT_REPLAY_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus}"
VLA_REPLAY_SH="${GT_REPLAY_ROOT}/global_unittest/vla_deploy_replay/replay_vla_deploy.sh"
[[ -f "${VLA_REPLAY_SH}" ]] || {
  echo "[run_vla_gt_replay] missing ${VLA_REPLAY_SH} (GT_REPLAY_ROOT wrong?)" >&2
  exit 1
}

CONDA_ENV="${CONDA_ENV:-Phi-0-wpy}"
if [[ -x "/mnt/data/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
  PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/${CONDA_ENV}/bin/python}"
else
  PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/${CONDA_ENV}/bin/python}"
fi

STUDENT_CKPT="${STUDENT_CKPT:?set STUDENT_CKPT=/path/to/phi0_student_last.pt}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_g1csv_aligned_phi0}"
REF_START="${REF_START:-0}"
MAX_REF_FRAMES="${MAX_REF_FRAMES:-320}"
HORIZON="${HORIZON:-8}"
NUM_STEPS="${NUM_STEPS:-300}"
NUM_ENVS="${NUM_ENVS:-1}"
CONTROL="${CONTROL:-student}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

INFER_OUT="${INFER_OUT:-${PHI0_ROOT}/experiments/vla_gt_replay/${TAG}}"
mkdir -p "${INFER_OUT}"

echo "[run_vla_gt_replay] stage 1/3: Isaac closed-loop infer -> ${INFER_OUT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PHI0_DISTILL_OUT="${INFER_OUT}" \
STUDENT_CKPT="${STUDENT_CKPT}" \
NUM_ENVS="${NUM_ENVS}" \
NUM_STEPS="${NUM_STEPS}" \
HORIZON="${HORIZON}" \
CONTROL="${CONTROL}" \
MAX_REF_FRAMES="${MAX_REF_FRAMES}" \
REF_START="${REF_START}" \
REF_ROOT="${REF_ROOT}" \
bash "${PHI0_ROOT}/tools/eval/run_sonic_online_infer_isaac.sh"

TAG_SUFFIX="${CONTROL}"
[[ "${TAG_SUFFIX}" == "qpos" ]] && TAG_SUFFIX="qpos_student"
INFER_NPZ="${INFER_OUT}/infer_qpos_traj_${TAG_SUFFIX}.npz"
[[ -f "${INFER_NPZ}" ]] || {
  echo "[run_vla_gt_replay] missing ${INFER_NPZ} — infer stage did not produce it" >&2
  exit 1
}

TOKENS="${INFER_OUT}/tokens.npz"
echo "[run_vla_gt_replay] stage 2/3: build tokens.npz from VLA z_pred"
"${PHI0_PY}" "${PHI0_ROOT}/tools/eval/build_gt_replay_tokens_from_infer_npz.py" \
  "${INFER_NPZ}" --out "${TOKENS}" --fps 50.0

echo "[run_vla_gt_replay] stage 3/3: replay through dedicated VLA-deploy pipeline"
RUN_DIR="${RUN_DIR:-${GT_REPLAY_ROOT}/global_unittest/vla_deploy_replay/runs/${TAG}}"

TOKENS="${TOKENS}" \
RUN_DIR="${RUN_DIR}" \
PHI0_PY="${PHI0_PY}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
REPLAY_SETTLE="${REPLAY_SETTLE:-120}" \
ENABLE_DEPLOY_CSV_LOGS=1 \
KILL_EXISTING="${KILL_EXISTING:-1}" \
bash "${VLA_REPLAY_SH}"

echo "[run_vla_gt_replay] done: ${RUN_DIR}/vla_deploy_replay.mp4"
