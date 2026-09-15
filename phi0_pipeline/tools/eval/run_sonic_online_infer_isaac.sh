#!/usr/bin/env bash
# Closed-loop Isaac infer for distilled Phi0 student.
# CONTROL=qpos_student (alias qpos): one forward → open-loop q_head[0:H] via joint PD.
# CONTROL=qpos_ref: live q*(z*) → joint PD (expert oracle).
# CONTROL=student|ref|ref_g1|mix: decode(z) → joint PD (same drive as train expert).
# CONTROL=direct_latent|direct_latent_ref: legacy ATM action_mode (Newton-unstable).
# Newton-GL video: tools/eval/newton_qpos_student_isaac_viz.py (PHI0_ISAAC_BACKEND=newton).
# Writes infer_qpos_traj_<tag>.npz every control step.

set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
# physx = Lab2 Kit; newton = Lab3 (default). Must bind before resolve_python.
sonic_apply_simulator "${SIMULATOR:-physx}"
sonic_resolve_python
sonic_export_isaac_env
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/sonic_online_distill_isaac}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # shellcheck source=/dev/null
  source "${GR00T_ROOT}/install_scripts/pick_cuda_device.sh"
fi

mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"
sonic_pythonpath

NUM_ENVS="${NUM_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-200}"
HORIZON="${HORIZON:-8}"
CONTROL="${CONTROL:-student}"  # student | ref | ref_g1 | mix | qpos | qpos_ref
TEACHER_ENCODER="${TEACHER_ENCODER:-onnx_g1}"  # Sonic qpos/GMR encoder only
STUDENT_CKPT="${STUDENT_CKPT:-${PHI0_DISTILL_OUT}/phi0_student_last.pt}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_qpos_phi0_protomotions_full}"


MAX_REF_FRAMES="${MAX_REF_FRAMES:-1000000}"
REF_START="${REF_START:-0}"
REF_ARGS=(
  "++callbacks.online_infer.phi0_full_v3_root=${REF_ROOT}"
  "++callbacks.online_infer.max_ref_frames=${MAX_REF_FRAMES}"
  "++callbacks.online_infer.ref_start=${REF_START}"
  "++callbacks.online_infer.teacher_encoder=${TEACHER_ENCODER}"
)
echo "[infer] ref_root=${REF_ROOT} start=${REF_START} max_frames=${MAX_REF_FRAMES}"
echo "[infer] teacher_encoder=${TEACHER_ENCODER}"

echo "[infer] ckpt=${STUDENT_CKPT}"
echo "[infer] control=${CONTROL} num_envs=${NUM_ENVS} steps=${NUM_STEPS}"

RECORD_VIDEO="${RECORD_VIDEO:-0}"
RENDER_ARGS=()
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "true" || "${RECORD_VIDEO}" == "True" ]]; then
  RENDER_DIR="${RENDER_DIR:-${PHI0_DISTILL_OUT}/renderings_${CONTROL}}"
  # Gear Sonic 4.5 working viz (same as hybrid RECORD_TRAIN_MP4): eval_camera +
  # use_fabric=false. Do NOT use manager_env/recorders=render (TiledCamera → usdrt crash).
  export ISAACLAB_ASSET_ROOT="${ISAACLAB_ASSET_ROOT:-${PHI0_ROOT}/assets/isaac_nucleus}"
  RENDER_ARGS+=(
    "++manager_env.config.render_results=true"
    "++manager_env.config.render_width=${RENDER_WIDTH:-960}"
    "++manager_env.config.render_height=${RENDER_HEIGHT:-540}"
    "++manager_env.config.enable_cameras=false"
    "++manager_env.sim.use_fabric=false"
    "++manager_env.commands.motion.debug_vis=false"
    "++manager_env.config.save_rendering_dir=${RENDER_DIR}"
  )
  echo "[infer] record_video=1 render_dir=${RENDER_DIR} (eval_camera use_fabric=false)"
  echo "[infer] ISAACLAB_ASSET_ROOT=${ISAACLAB_ASSET_ROOT}"
fi

"${PYTHON_BIN}" gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++run_eval_loop=False \
  "++num_envs=${NUM_ENVS}" \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.terminations.anchor_pos.params.threshold=100.0" \
  "++manager_env.terminations.anchor_ori_full.params.threshold=100.0" \
  "++manager_env.terminations.ee_body_pos.params.threshold=100.0" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered" \
  "+callbacks.online_infer._target_=phi0.online.isaac_callback.OnlineInferCallback" \
  "++callbacks.online_infer.student_ckpt=${STUDENT_CKPT}" \
  "++callbacks.online_infer.num_steps=${NUM_STEPS}" \
  "++callbacks.online_infer.action_horizon=${HORIZON}" \
  "++callbacks.online_infer.control=${CONTROL}" \
  "++callbacks.online_infer.out_dir=${PHI0_DISTILL_OUT}" \
  "++eval_callbacks=online_infer" \
  "${REF_ARGS[@]}" \
  "${RENDER_ARGS[@]}" \
  "$@"

# Filenames: qpos → qpos_student; latent modes keep their name.
TAG="${CONTROL}"
[[ "${TAG}" == "qpos" ]] && TAG="qpos_student"
NPZ="${PHI0_DISTILL_OUT}/infer_qpos_traj_${TAG}.npz"
echo "[infer] wrote ${NPZ}"
if [[ "${RECORD_VIDEO}" == "1" || "${RECORD_VIDEO}" == "true" || "${RECORD_VIDEO}" == "True" ]]; then
  echo "[infer] video (if produced): ${PHI0_DISTILL_OUT}/infer_${TAG}.mp4"
elif [[ "${TAG}" == "qpos_student" || "${TAG}" == "qpos_ref" ]]; then
  echo "[infer] for a video of this qpos-space control mode, rerun with RECORD_VIDEO=1"
else
  echo "[infer] for a trustworthy video of this z-space control mode: STUDENT_CKPT=${STUDENT_CKPT} bash ${PHI0_ROOT}/tools/eval/run_vla_gt_replay_visualize.sh"
fi
