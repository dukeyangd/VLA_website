#!/usr/bin/env bash
# Phi0 <- Sonic online distill on REAL Isaac ManagerEnv.
# Expert: online encode (GMR+live root) → direct_latent under DR/push; student BC
# on z* + decode q* (q_head). Deploy: CONTROL=qpos_student (q_head joint PD, no Sonic).
# checkpoint=sonic_release inherits DR events; push_robot re-enabled via train_only_events=[].
# Conda: Phi-0-wpy. Workdir code: phi-0-wbc; sim/weights: GR00T-WholeBodyControl.

set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
sonic_resolve_python
sonic_export_isaac_env
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/sonic_online_distill_isaac}"

# Prefer GPU >=1 (leave 0 free); override with CUDA_VISIBLE_DEVICES
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # shellcheck source=/dev/null
  source "${GR00T_ROOT}/install_scripts/pick_cuda_device.sh"
fi

mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"
sonic_pythonpath

echo "[run] python=${PYTHON_BIN}"
echo "[run] conda_env=${CONDA_ENV}"
echo "[run] out=${PHI0_DISTILL_OUT}"
"${PYTHON_BIN}" -c "import sys; print(sys.executable); import isaaclab, phi0; print('phi0', phi0.__file__)"

HORIZON="${HORIZON:-8}"
NUM_ENVS="${NUM_ENVS:-64}"
RESUME_CKPT="${RESUME_CKPT:-}"
# Optional LeRobot v3 root with action.smpl_* + RSI qpos (default: full protomotions-matched).
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_qpos_phi0_protomotions_full}"

# Full 47M: RG_SCAN=1 loads one parquet row-group (~1M frames) at a time, trains, drops.
# Override RG_SCAN=0 MAX_REF_FRAMES=200000 for small eager.
MAX_REF_FRAMES="${MAX_REF_FRAMES:-0}"
LAZY_REF="${LAZY_REF:-0}"
RG_SCAN="${RG_SCAN:-1}"
SHARD_RANK="${SHARD_RANK:-0}"
SHARD_WORLD="${SHARD_WORLD:-1}"
# EPOCHS: if set, NUM_CHUNKS ≈ epochs * ceil(shard_starts / H).
# (each grad step: cursor += H per env; wrap/fall resample uniformly + RSI)
EPOCHS="${EPOCHS:-}"

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_ARGS+=("++callbacks.online_distill.student_ckpt=${RESUME_CKPT}")
  echo "[run] resume=${RESUME_CKPT}"
fi

REF_ARGS=("++callbacks.online_distill.phi0_full_v3_root=${REF_ROOT}")
echo "[run] ref_root=${REF_ROOT} (SMPL+RSI; z* live-encoded; lazy=${LAZY_REF} rg_scan=${RG_SCAN})"

if [[ -n "${EPOCHS}" ]]; then
  T="$(sonic_count_ref_frames "${REF_ROOT}" "${MAX_REF_FRAMES}")"
  H="${HORIZON}"
  STARTS=$(( T > H ? T - H + 1 : 1 ))
  # Each rank only sees 1/SHARD_WORLD of starts.
  STARTS_LOCAL=$(( (STARTS + SHARD_WORLD - 1) / SHARD_WORLD ))
  STEPS_PER_EPOCH=$(( (STARTS_LOCAL + H - 1) / H ))
  NUM_CHUNKS=$(( EPOCHS * STEPS_PER_EPOCH ))
  echo "[run] epochs=${EPOCHS} T=${T} starts_global=${STARTS} starts_local~${STARTS_LOCAL} H=${H} steps/epoch=${STEPS_PER_EPOCH} → chunks=${NUM_CHUNKS}"
else
  NUM_CHUNKS="${NUM_CHUNKS:-256}"
fi

REF_ARGS+=("++callbacks.online_distill.max_ref_frames=${MAX_REF_FRAMES}")
if [[ "${LAZY_REF}" == "1" || "${LAZY_REF}" == "true" || "${LAZY_REF}" == "True" ]]; then
  REF_ARGS+=("++callbacks.online_distill.lazy_ref=true")
else
  REF_ARGS+=("++callbacks.online_distill.lazy_ref=false")
fi
if [[ "${RG_SCAN}" == "1" || "${RG_SCAN}" == "true" || "${RG_SCAN}" == "True" ]]; then
  REF_ARGS+=("++callbacks.online_distill.rg_scan=true")
else
  REF_ARGS+=("++callbacks.online_distill.rg_scan=false")
fi
REF_ARGS+=("++callbacks.online_distill.shard_rank=${SHARD_RANK}")
REF_ARGS+=("++callbacks.online_distill.shard_world=${SHARD_WORLD}")
REF_ARGS+=("++callbacks.online_distill.ref_start=${REF_START:-0}")
TEACHER_ENCODER="${TEACHER_ENCODER:-onnx_g1}"
REF_ARGS+=("++callbacks.online_distill.teacher_encoder=${TEACHER_ENCODER}")
EXPERT_FALL_Z_ERR="${EXPERT_FALL_Z_ERR:-0.25}"
REF_ARGS+=("++callbacks.online_distill.expert_fall_z_err=${EXPERT_FALL_Z_ERR}")
if [[ "${DUMP_EXPERT_TRAJ:-0}" == "1" || "${DUMP_EXPERT_TRAJ:-}" == "true" ]]; then
  REF_ARGS+=("++callbacks.online_distill.dump_expert_traj=true")
fi
Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-${PHI0_ROOT}/meta/z_star_stats.json}"
REF_ARGS+=("++callbacks.online_distill.z_star_stats_path=${Z_STAR_STATS_PATH}")
CKPT_EVERY="${CKPT_EVERY:-10000}"
REF_ARGS+=("++callbacks.online_distill.ckpt_every=${CKPT_EVERY}")
export PHI0_CKPT_STEP_KEEP="${PHI0_CKPT_STEP_KEEP:-0}"
# RSI phase on init/wrap/fall: first_frame (clip t=0) | random (uniform window start)
RSI_START="${RSI_START:-first_frame}"
REF_ARGS+=("++callbacks.online_distill.rsi_start=${RSI_START}")

echo "[run] num_envs=${NUM_ENVS} chunks(steps)=${NUM_CHUNKS} horizon=${HORIZON} stride=1 shard=${SHARD_RANK}/${SHARD_WORLD} lazy=${LAZY_REF} rg_scan=${RG_SCAN}"
echo "[run] teacher_encoder=${TEACHER_ENCODER} z_star_stats=${Z_STAR_STATS_PATH} expert_fall_z_err=${EXPERT_FALL_Z_ERR} ckpt_every=${CKPT_EVERY} rsi_start=${RSI_START}"

exec "${PYTHON_BIN}" gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++run_eval_loop=False \
  "++num_envs=${NUM_ENVS}" \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.terminations.anchor_pos.params.threshold=100.0" \
  "++manager_env.terminations.anchor_ori_full.params.threshold=100.0" \
  "++manager_env.terminations.ee_body_pos.params.threshold=100.0" \
  "++manager_env.config.train_only_events=[]" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered" \
  "+callbacks.online_distill._target_=phi0.online.isaac_callback.OnlineDistillCallback" \
  "++callbacks.online_distill.num_chunks=${NUM_CHUNKS}" \
  "++callbacks.online_distill.action_horizon=${HORIZON}" \
  "++callbacks.online_distill.clip_stride=1" \
  "++callbacks.online_distill.out_dir=${PHI0_DISTILL_OUT}" \
  "++eval_callbacks=online_distill" \
  "${RESUME_ARGS[@]}" \
  "${REF_ARGS[@]}" \
  "$@"
