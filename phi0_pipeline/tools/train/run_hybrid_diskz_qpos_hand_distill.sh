#!/usr/bin/env bash
# Hybrid disk-z distill (BoneSEED demo5 + 810demo egypt):
#   BoneSEED: onnx_g1 encode + sonic_release ATM decode
#   810:      disk unified[396:460] + low_latency ATM decode
# Expert drive: always joint_pd (dual ATM → q*). Works on SIMULATOR=newton|physx.
#
# Env/Manager cfg stays sonic_release. Does NOT change run_boneseed_full defaults.
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths

SIMULATOR="${SIMULATOR:-newton}"
sim_lc="$(echo "${SIMULATOR}" | tr '[:upper:]' '[:lower:]')"
case "${sim_lc}" in
  physx|isaaclab|lab2) SIMULATOR=physx ;;
  newton|lab3|isaaclab3|"") SIMULATOR=newton ;;
  *)
    echo "[hybrid] SIMULATOR=${SIMULATOR} unsupported (want physx|newton)" >&2
    exit 1
    ;;
esac
sonic_apply_simulator "${SIMULATOR}"
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath
export PHI0_FABRIC=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29571}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Hybrid forces joint_pd (shell hint; Python resolve_expert_drive also enforces).
export PHI0_DISTILL_EXPERT_DRIVE=joint_pd
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE:-hybrid}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NGPU="${NGPU:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# Env + BoneSEED ATM = sonic_release. low_latency ATM is dual-loaded in fabric
# for 810 disk-z decode only — do NOT set DEPLOY_POLICY_DIR=low_latency here.
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0}"
REF_ROOT_810="${REF_ROOT_810:-/mnt/data2/wpy/workspace/810demo_egypt_layout}"
export REF_ROOT_810
ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT_810}/meta/stats.json}"
export ACTION_STATS_PATH
EPISODE_ALLOWLIST="${EPISODE_ALLOWLIST:-${PHI0_ROOT}/meta/boneseed_allowlist_demo5skill.json}"
export W_Z="${W_Z:-0}"
export W_SMPL="${W_SMPL:-0}"
export W_Q_HEAD="${W_Q_HEAD:-1}"
export W_HAND="${W_HAND:-1}"
export RESIDENT_REF="${RESIDENT_REF:-1}"
HORIZON="${HORIZON:-32}"
NUM_ENVS="${NUM_ENVS:-64}"
EPOCHS="${EPOCHS:-1}"
export EPOCHS
RSI_START="${RSI_START:-random_episode}"
LR="${LR:-1e-4}"
CKPT_EVERY="${CKPT_EVERY:-500}"
USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-1}"
EXPERT_FALL_Z_ERR="${EXPERT_FALL_Z_ERR:-0.35}"
Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-${PHI0_ROOT}/meta/z_star_stats_boneseed.json}"
export Z_STAR_STATS_PATH

PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/hybrid_diskz_${SIMULATOR}_fabric${NGPU}_e${EPOCHS}_h${HORIZON}_b${NUM_ENVS}_${STAMP}}"
export PHI0_DISTILL_OUT

if [[ "${HORIZON}" -lt 1 ]]; then
  echo "[hybrid] HORIZON must be >=1 (got ${HORIZON})" >&2
  exit 1
fi
if [[ ! -f "${ACTION_STATS_PATH}" ]]; then
  echo "[hybrid] missing ACTION_STATS_PATH=${ACTION_STATS_PATH}" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT_810}" ]]; then
  echo "[hybrid] missing REF_ROOT_810=${REF_ROOT_810}" >&2
  exit 1
fi
if [[ ! -f "${EPISODE_ALLOWLIST}" ]]; then
  echo "[hybrid] missing EPISODE_ALLOWLIST=${EPISODE_ALLOWLIST}" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT}/meta/lang_latents_qwen3vl" ]]; then
  echo "[hybrid] missing ${REF_ROOT}/meta/lang_latents_qwen3vl" >&2
  exit 1
fi

mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"

export PHI0_RESIDENT_SHARE="${PHI0_RESIDENT_SHARE:-1}"
export PHI0_RESIDENT_MMAP="${PHI0_RESIDENT_MMAP:-1}"
export PHI0_RESIDENT_LOAD_WORKERS="${PHI0_RESIDENT_LOAD_WORKERS:-16}"
export PHI0_RESIDENT_CACHE_ROOT="${PHI0_RESIDENT_CACHE_ROOT:-/mnt/data3/wpy/cache/resident}"

echo "[hybrid] simulator=${SIMULATOR} expert_drive=joint_pd"
echo "[hybrid] env_policy=${DEPLOY_POLICY_DIR} (BoneSEED ATM=release; 810 ATM=low_latency dual-load)"
echo "[hybrid] teacher_z=${TEACHER_Z_SOURCE}"
echo "[hybrid] boneseed=${REF_ROOT} egypt810=${REF_ROOT_810}"
echo "[hybrid] stats=${ACTION_STATS_PATH} allow=${EPISODE_ALLOWLIST}"
echo "[hybrid] W_Z=${W_Z} W_SMPL=${W_SMPL} W_Q_HEAD=${W_Q_HEAD} W_HAND=${W_HAND}"
RECORD_TRAIN_MP4="${RECORD_TRAIN_MP4:-0}"
# Avoid leftover env opening RECORD without kit/rgb_array patch.
if [[ "${RECORD_TRAIN_MP4}" != "1" && "${RECORD_TRAIN_MP4}" != "true" ]]; then
  unset RECORD_TRAIN_MP4 PHI0_DISTILL_RECORD_MP4 || true
  RECORD_TRAIN_MP4=0
fi
# RECORD defaults (only when recording; non-RECORD keeps RECORD_EVERY=10).
if [[ "${RECORD_TRAIN_MP4}" == "1" || "${RECORD_TRAIN_MP4}" == "true" ]]; then
  RECORD_EVERY="${RECORD_EVERY:-2}"
  # Mid-episode RSI so expert joint_pd shows motion (ep frame0 is often stand).
  RSI_START="${PHI0_DISTILL_MP4_RSI:-random}"
else
  RECORD_EVERY="${RECORD_EVERY:-10}"
fi

echo "[hybrid] out=${PHI0_DISTILL_OUT} ngpu=${NGPU} B=${NUM_ENVS} rsi=${RSI_START}"
echo "[hybrid] record_mp4=${RECORD_TRAIN_MP4} record_every=${RECORD_EVERY} no_dr=${NO_DR:-0}"
echo "[hybrid] python=${PYTHON_BIN}"

if [[ "${SIMULATOR}" == "newton" ]]; then
  EXTRA_ARGS=()
  if [[ "${RECORD_TRAIN_MP4}" == "1" || "${RECORD_TRAIN_MP4}" == "true" ]]; then
    export PHI0_DISTILL_RECORD_MP4=1
    export RECORD_TRAIN_MP4=1
    export PHI0_DISTILL_MP4_MAX_FRAMES="${PHI0_DISTILL_MP4_MAX_FRAMES:-100}"
    export PHI0_DISTILL_MP4_FLUSH="${PHI0_DISTILL_MP4_FLUSH:-0}"
    EXTRA_ARGS+=(--record_train_mp4 --record_every "${RECORD_EVERY}")
    echo "[hybrid] newton RECORD_TRAIN_MP4=1 every=${RECORD_EVERY} max_frames=${PHI0_DISTILL_MP4_MAX_FRAMES} flush=${PHI0_DISTILL_MP4_FLUSH}"
  fi
  if [[ "${NO_DR:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no_dr)
  fi
  # shellcheck disable=SC2086
  exec "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    "${PHI0_ROOT}/tools/train/newton_boneseed_distill_fabric.py" \
    --ref_root "${REF_ROOT}" \
    --ref_root_810 "${REF_ROOT_810}" \
    --teacher_z_source hybrid \
    --action_stats_path "${ACTION_STATS_PATH}" \
    --episode_allowlist "${EPISODE_ALLOWLIST}" \
    --num_envs "${NUM_ENVS}" \
    --horizon "${HORIZON}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --ckpt_every "${CKPT_EVERY}" \
    --rsi_start "${RSI_START}" \
    --expert_fall_z_err "${EXPERT_FALL_Z_ERR}" \
    --w_z "${W_Z}" \
    --w_smpl "${W_SMPL}" \
    --w_q_head "${W_Q_HEAD}" \
    --w_hand "${W_HAND}" \
    --resident_ref \
    --out_dir "${PHI0_DISTILL_OUT}" \
    $([ "${USE_LANG_LATENT_CACHE}" = "1" ] && echo --use_lang_latent_cache || true) \
    "${EXTRA_ARGS[@]}"
fi

# --- physx: gear sonic AppLauncher + OnlineDistillCallback (hybrid params) ---
export PHI0_ISAAC_LOCK_HOLD_S="${PHI0_ISAAC_LOCK_HOLD_S:-0}"
export PHI0_FABRIC_STAGGER_S="${PHI0_FABRIC_STAGGER_S:-0}"

# Safety valve for callback (epoch-cover stops via EPOCHS; num_chunks unused when resident).
NUM_CHUNKS="${NUM_CHUNKS:-999999}"

REF_ARGS=(
  "++callbacks.online_distill.phi0_full_v3_root=${REF_ROOT}"
  "++callbacks.online_distill.ref_root_810=${REF_ROOT_810}"
  "++callbacks.online_distill.teacher_z_source=hybrid"
  "++callbacks.online_distill.resident_ref=true"
  "++callbacks.online_distill.max_ref_frames=0"
  "++callbacks.online_distill.epochs=${EPOCHS}"
  "++callbacks.online_distill.episode_allowlist=${EPISODE_ALLOWLIST}"
  "++callbacks.online_distill.action_stats_path=${ACTION_STATS_PATH}"
  "++callbacks.online_distill.z_star_stats_path=${Z_STAR_STATS_PATH}"
  "++callbacks.online_distill.teacher_encoder=onnx_g1"
  "++callbacks.online_distill.rsi_start=${RSI_START}"
  "++callbacks.online_distill.lr=${LR}"
  "++callbacks.online_distill.ckpt_every=${CKPT_EVERY}"
  "++callbacks.online_distill.expert_fall_z_err=${EXPERT_FALL_Z_ERR}"
  "++callbacks.online_distill.w_z=${W_Z}"
  "++callbacks.online_distill.w_smpl=${W_SMPL}"
  "++callbacks.online_distill.w_q_head=${W_Q_HEAD}"
  "++callbacks.online_distill.w_hand=${W_HAND}"
  "++callbacks.online_distill.expert_drive=joint_pd"
  "++callbacks.online_distill.simulator=physx"
  "++callbacks.online_distill.lazy_ref=false"
  "++callbacks.online_distill.rg_scan=false"
)
if [[ "${USE_LANG_LATENT_CACHE}" == "1" ]]; then
  REF_ARGS+=("++callbacks.online_distill.use_lang_latent_cache=true")
else
  REF_ARGS+=("++callbacks.online_distill.use_lang_latent_cache=false")
fi
if [[ "${RECORD_TRAIN_MP4}" == "1" || "${RECORD_TRAIN_MP4}" == "true" ]]; then
  export PHI0_DISTILL_RECORD_MP4=1
  export RECORD_TRAIN_MP4=1
  # Hard cap: write once. Do NOT flush every few frames (that re-encoded GB junk).
  export PHI0_DISTILL_MP4_MAX_FRAMES="${PHI0_DISTILL_MP4_MAX_FRAMES:-120}"
  export PHI0_DISTILL_MP4_FLUSH="${PHI0_DISTILL_MP4_FLUSH:-0}"
  export PHI0_DISTILL_MP4_CAM="${PHI0_DISTILL_MP4_CAM:-fixed}"
  export PHI0_DISTILL_MP4_SKIP_STEPS="${PHI0_DISTILL_MP4_SKIP_STEPS:-80}"
  # onnx2torch ConstantOfShape breaks dynamo; keep logs clean for viz jobs.
  export PHI0_ONNX_ENCODE_COMPILE="${PHI0_ONNX_ENCODE_COMPILE:-0}"
  # Project-local Isaac nucleus (copied USD; no symlink / no S3).
  export ISAACLAB_ASSET_ROOT="${ISAACLAB_ASSET_ROOT:-${PHI0_ROOT}/assets/isaac_nucleus}"
  if [[ ! -f "${ISAACLAB_ASSET_ROOT}/Isaac/Environments/Grid/default_environment.usd" ]]; then
    echo "[hybrid] missing ${ISAACLAB_ASSET_ROOT}/Isaac/Environments/Grid/default_environment.usd" >&2
    exit 1
  fi
  REF_ARGS+=(
    "++callbacks.online_distill.record_train_mp4=true"
    "++callbacks.online_distill.record_every=${RECORD_EVERY}"
    # Pure expert joint_pd so RECORD shows ref motion (not early student stand/fall).
    "++callbacks.online_distill.dagger_beta=1.0"
    # Gear Sonic 4.5 viz path: eval_camera + use_fabric=false (not viewport rgb_array).
    "++manager_env.config.render_results=true"
    "++manager_env.config.render_width=960"
    "++manager_env.config.render_height=540"
    "++manager_env.config.enable_cameras=false"
    "++manager_env.sim.use_fabric=false"
    # Fabric rewrites terrain → MeshBox+PreviewSurface (not GroundPlane USD).
    # Yellow goal markers drown the robot silhouette.
    "++manager_env.commands.motion.debug_vis=false"
  )
  echo "[hybrid] physx RECORD_TRAIN_MP4=1 every=${RECORD_EVERY} max_frames=${PHI0_DISTILL_MP4_MAX_FRAMES} flush=${PHI0_DISTILL_MP4_FLUSH}"
  echo "[hybrid] viz=eval_camera cam=${PHI0_DISTILL_MP4_CAM} skip=${PHI0_DISTILL_MP4_SKIP_STEPS} rsi=${RSI_START} beta=1 debug_vis=0 MeshBox"
  echo "[hybrid] ISAACLAB_ASSET_ROOT=${ISAACLAB_ASSET_ROOT}"
  echo "[hybrid] WARN: if rgb is red-noise, recorder auto-aborts (no multi-GB junk)"
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${PHI0_ROOT}/tools/train/train_online_distill_fabric.py" \
  --ngpu "${NGPU}" \
  --groot-root "${GR00T_ROOT}" \
  --stagger-s "${PHI0_FABRIC_STAGGER_S}" \
  -- \
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
  "${REF_ARGS[@]}"
