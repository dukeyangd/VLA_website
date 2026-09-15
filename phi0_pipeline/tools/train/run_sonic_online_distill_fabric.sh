#!/usr/bin/env bash
# CANONICAL multi-GPU online distill — Fabric DDP + gradient AllReduce.
# Default NGPU=2 smoke. BoneSEED full wrapper: run_boneseed_full_distill_4gpu.sh
# Mirrors ProtoMotions: AppLauncher-before-torch → Fabric.launch → AppLauncher(device=…).
# Multi-GPU GPU model = ProtoMotions SLURM (1 GPU visible per rank): torchrun + CVD remap.
# Shard-independent multi-process scripts are legacy/bug — see pipeline §4.0.
#
# Example smoke (egypt clip):
#   NGPU=2 NUM_ENVS=8 NUM_CHUNKS=40 REF_START=80282 MAX_REF_FRAMES=278 RG_SCAN=0 \
#     bash tools/train/run_sonic_online_distill_fabric.sh

set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
# Hydra+AppLauncher path → gear sonic native PhysX unless SIMULATOR=newton.
sonic_apply_simulator "${SIMULATOR:-physx}"
sonic_resolve_python
sonic_export_isaac_env
export PHI0_FABRIC=1
# ProtoMotions has neither Kit lock-hold nor AppLauncher stagger; keep both off.
export PHI0_ISAAC_LOCK_HOLD_S="${PHI0_ISAAC_LOCK_HOLD_S:-0}"
export PHI0_FABRIC_STAGGER_S="${PHI0_FABRIC_STAGGER_S:-0}"
# Force local TCPStore/NCCL (cluster DNS/host aliases have broken c10d before).
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29551}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

NGPU="${NGPU:-2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/sonic_online_distill_fabric_${STAMP}}"

# Visible GPUs for this job (per-rank remap happens in train_online_distill_fabric.py).
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mapfile -t _gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  if ((${#_gpus[@]} >= NGPU + 1)); then
    CVD=()
    for ((i = 1; i <= NGPU; i++)); do CVD+=("${_gpus[$i]}"); done
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${CVD[*]}")"
  elif ((${#_gpus[@]} >= NGPU)); then
    CVD=()
    for ((i = 0; i < NGPU; i++)); do CVD+=("${_gpus[$i]}"); done
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${CVD[*]}")"
  fi
fi

mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"
sonic_pythonpath

echo "[fabric] python=${PYTHON_BIN} ngpu=${NGPU} cvd=${CUDA_VISIBLE_DEVICES:-} out=${PHI0_DISTILL_OUT}"
"${PYTHON_BIN}" -c "from lightning.fabric import Fabric; import phi0; print('fabric+phi0 ok', phi0.__file__)"

HORIZON="${HORIZON:-8}"
NUM_ENVS="${NUM_ENVS:-8}"
RESUME_CKPT="${RESUME_CKPT:-}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_g1csv_aligned_phi0}"
MAX_REF_FRAMES="${MAX_REF_FRAMES:-278}"
LAZY_REF="${LAZY_REF:-0}"
RG_SCAN="${RG_SCAN:-0}"
REF_START="${REF_START:-80282}"
NUM_CHUNKS="${NUM_CHUNKS:-40}"
USE_VLM="${USE_VLM:-0}"
USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-0}"
RSI_START="${RSI_START:-first_frame}"
LR="${LR:-1e-4}"
# If CKPT_EVERY is explicitly exported by caller, keep it; else EPOCHS may set per-epoch.
if [[ -n "${CKPT_EVERY+x}" && -n "${CKPT_EVERY}" ]]; then
  CKPT_EVERY_SET=1
else
  CKPT_EVERY_SET=
  CKPT_EVERY="${CKPT_EVERY:-1000}"
fi
ACTION_STATS_PATH="${ACTION_STATS_PATH:-}"
# EPOCHS: if set, primary stop for random_episode (RG-list passes / ep-queue drain).
# NUM_CHUNKS = max_steps safety ≈ 1.5 * epochs * ceil(n_frames / GLOBAL_B).
# - else: stride-1 tape ≈ ceil(starts_per_rank / H)
EPOCHS="${EPOCHS:-}"
EPISODE_ALLOWLIST="${EPISODE_ALLOWLIST:-}"

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_ARGS+=("++callbacks.online_distill.student_ckpt=${RESUME_CKPT}")
  echo "[fabric] resume=${RESUME_CKPT}"
fi

if [[ -n "${EPOCHS}" ]]; then
  H="${HORIZON}"
  # random_episode: 1 epoch ≈ slide through allowlisted frames once (full-ep).
  if [[ "${RSI_START}" == "random_episode" ]]; then
    GLOBAL_B=$(( NUM_ENVS * NGPU ))
    ALLOW="${EPISODE_ALLOWLIST:-}"
    FRAME_CACHE=""
    if [[ -n "${ALLOW}" && -f "${ALLOW}" ]]; then
      FRAME_CACHE="${ALLOW%.json}_frame_count.json"
    fi
    read -r N_EP N_FRAMES < <("${PYTHON_BIN}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, r"""${PHI0_ROOT}/src""")
from phi0.online.isaac_loop import load_episode_allowlist
from phi0.online.lazy_ref import count_allowlist_frames, list_boneseed_row_groups
allow = r"""${ALLOW}""".strip()
root = Path(r"""${REF_ROOT}""")
if allow and allow.lower() not in ("", "none", "full", "-", "0", "all") and Path(allow).is_file():
    allow_p = Path(allow)
    ids = load_episode_allowlist(allow_p)
    cache = r"""${FRAME_CACHE}""" or None
    info = count_allowlist_frames(
        root, ids, cache_path=cache or None, allowlist_path=allow_p,
    )
    print(int(info["n_ep"]), int(info["n_frames"]))
else:
    # No allowlist: approximate with full tape rows.
    specs = list_boneseed_row_groups(root)
    print(0, int(sum(s.num_rows for s in specs)))
PY
)
    STEPS_PER_EPOCH=$(( (N_FRAMES + GLOBAL_B - 1) / GLOBAL_B ))
    [[ "${STEPS_PER_EPOCH}" -lt 1 ]] && STEPS_PER_EPOCH=1
    NUM_CHUNKS=$(( (EPOCHS * STEPS_PER_EPOCH * 3 + 1) / 2 ))
    echo "[fabric] epochs=${EPOCHS} mode=ep_queue_drain n_ep=${N_EP} n_frames=${N_FRAMES} Bglobal=${GLOBAL_B} eta_steps/epoch~${STEPS_PER_EPOCH} max_steps=${NUM_CHUNKS}"
    echo "[fabric] WARN NUM_CHUNKS is safety valve; stop after EPOCHS RG passes (+ drain)"
  else
    T="$(sonic_count_ref_frames "${REF_ROOT}" "${MAX_REF_FRAMES}")"
    STARTS=$(( T > H ? T - H + 1 : 1 ))
    # Fabric online loop shards RGs by world_size (=NGPU).
    STARTS_LOCAL=$(( (STARTS + NGPU - 1) / NGPU ))
    STEPS_PER_EPOCH=$(( (STARTS_LOCAL + H - 1) / H ))
    NUM_CHUNKS=$(( EPOCHS * STEPS_PER_EPOCH ))
    echo "[fabric] epochs=${EPOCHS} T=${T} starts=${STARTS} starts/rank~${STARTS_LOCAL} H=${H} steps/epoch=${STEPS_PER_EPOCH} → chunks=${NUM_CHUNKS}"
  fi
  # Save once per epoch when caller did not override CKPT_EVERY.
  if [[ -z "${CKPT_EVERY_SET:-}" ]]; then
    CKPT_EVERY="${STEPS_PER_EPOCH}"
  fi
fi

REF_ARGS=(
  "++callbacks.online_distill.phi0_full_v3_root=${REF_ROOT}"
  "++callbacks.online_distill.max_ref_frames=${MAX_REF_FRAMES}"
  "++callbacks.online_distill.ref_start=${REF_START}"
  # Ignored when Fabric is on (rank/world from Fabric); kept for single-process fallback.
  "++callbacks.online_distill.shard_rank=0"
  "++callbacks.online_distill.shard_world=1"
)
if [[ -n "${EPISODE_ALLOWLIST:-}" ]]; then
  case "${EPISODE_ALLOWLIST}" in
    none|full|-|0|ALL|all) ;;
    *)
      REF_ARGS+=("++callbacks.online_distill.episode_allowlist=${EPISODE_ALLOWLIST}")
      echo "[fabric] episode_allowlist=${EPISODE_ALLOWLIST}"
      ;;
  esac
fi
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
if [[ "${USE_VLM}" == "1" || "${USE_VLM}" == "true" || "${USE_VLM}" == "True" ]]; then
  REF_ARGS+=("++callbacks.online_distill.use_vlm=true")
else
  REF_ARGS+=("++callbacks.online_distill.use_vlm=false")
fi
if [[ "${USE_LANG_LATENT_CACHE}" == "1" || "${USE_LANG_LATENT_CACHE}" == "true" || "${USE_LANG_LATENT_CACHE}" == "True" ]]; then
  REF_ARGS+=("++callbacks.online_distill.use_lang_latent_cache=true")
else
  REF_ARGS+=("++callbacks.online_distill.use_lang_latent_cache=false")
fi
REF_ARGS+=("++callbacks.online_distill.rsi_start=${RSI_START}")
REF_ARGS+=("++callbacks.online_distill.lr=${LR}")
REF_ARGS+=("++callbacks.online_distill.ckpt_every=${CKPT_EVERY}")
if [[ -n "${ACTION_STATS_PATH}" ]]; then
  REF_ARGS+=("++callbacks.online_distill.action_stats_path=${ACTION_STATS_PATH}")
elif [[ -f "${REF_ROOT}/meta/stats.json" ]]; then
  REF_ARGS+=("++callbacks.online_distill.action_stats_path=${REF_ROOT}/meta/stats.json")
fi
TEACHER_ENCODER="${TEACHER_ENCODER:-onnx_g1}"
REF_ARGS+=("++callbacks.online_distill.teacher_encoder=${TEACHER_ENCODER}")
Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-${PHI0_ROOT}/meta/z_star_stats.json}"
REF_ARGS+=("++callbacks.online_distill.z_star_stats_path=${Z_STAR_STATS_PATH}")
EXPERT_FALL_Z_ERR="${EXPERT_FALL_Z_ERR:-0.25}"
REF_ARGS+=("++callbacks.online_distill.expert_fall_z_err=${EXPERT_FALL_Z_ERR}")
if [[ "${DUMP_EXPERT_TRAJ:-0}" == "1" || "${DUMP_EXPERT_TRAJ:-}" == "true" ]]; then
  REF_ARGS+=("++callbacks.online_distill.dump_expert_traj=true")
fi

echo "[fabric] num_envs(per-gpu)=${NUM_ENVS} chunks=${NUM_CHUNKS} H=${HORIZON} lr=${LR} rsi=${RSI_START} ref_start=${REF_START} max_frames=${MAX_REF_FRAMES} rg_scan=${RG_SCAN} use_vlm=${USE_VLM} lang_cache=${USE_LANG_LATENT_CACHE}"
echo "[fabric] torchrun nproc=${NGPU} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[fabric] action_stats=${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
echo "[fabric] teacher_encoder=${TEACHER_ENCODER} z_star_stats=${Z_STAR_STATS_PATH} expert_fall_z_err=${EXPERT_FALL_Z_ERR}"

# ProtoMotions SLURM: ntasks-per-node=ngpu. torchrun is the non-SLURM equivalent.
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
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
  "${RESUME_ARGS[@]}" \
  "${REF_ARGS[@]}" \
  "$@"
