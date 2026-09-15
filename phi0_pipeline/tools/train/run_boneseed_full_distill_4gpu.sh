#!/usr/bin/env bash
# CANONICAL multi-GPU BoneSEED distill — Fabric DDP AllReduce on Newton kit-less.
# Uses newton_boneseed_distill_fabric.py (launch_simulation), NOT AppLauncher /
# SimulationApp (broken on Lab3+Newton).
# Policy: meta/NEWTON_train_deploy_pipeline.md §4.0
# Stop = EPOCHS × per-RG allowlist frame cover (no NUM_CHUNKS hard stop).
# DR ON by default = obs corruption only (Event DR off).
# Opt out: --no_dr. Full Event DR: PHI0_NEWTON_DR_EVENTS=1.
# Push-only: PHI0_NEWTON_DR_PUSH_ONLY=1.
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
# BoneSEED full defaults to Newton; override with SIMULATOR=physx via run_distill_fabric.sh.
sonic_apply_simulator "${SIMULATOR:-newton}"
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath
export PHI0_FABRIC=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29561}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# BoneSEED distill default: progress_only + Fourier cap=64 (override if needed).
export PHI0_ADALN_MODE="${PHI0_ADALN_MODE:-progress_only}"
export PHI0_FOURIER_MAX_CYCLES="${PHI0_FOURIER_MAX_CYCLES:-64}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NGPU="${NGPU:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0}"
Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-${PHI0_ROOT}/meta/z_star_stats_boneseed.json}"
export Z_STAR_STATS_PATH
NUM_ENVS="${NUM_ENVS:-256}"
HORIZON="${HORIZON:-32}"
if [[ "${HORIZON}" -lt 1 ]]; then
  echo "[boneseed_full] HORIZON must be >=1 (got ${HORIZON})" >&2
  exit 1
fi
EPOCHS="${EPOCHS:-4}"
RSI_START="${RSI_START:-random}"
LR="${LR:-1e-4}"
# ACT DiT depth (phi0_student _PHI0_ACT_DIT / PHI0_DISTILL_ACT_LAYERS).
ACT_LAYERS="${ACT_LAYERS:-6}"
export PHI0_DISTILL_ACT_LAYERS="${ACT_LAYERS}"
USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-1}"
# Train-only proprio dropout (zeros obs_hist prefix). Opt out: =0.
export PHI0_DISTILL_STATE_DROPOUT="${PHI0_DISTILL_STATE_DROPOUT:-0.4}"
# Default: expert-only slide-1 BC (student 1×H loss; expert steps sim).
# Opt-in DAgger: STUDENT_DRIVE=1 (+ optional DAGGER_BETA_* anneal).
STUDENT_DRIVE="${STUDENT_DRIVE:-0}"
# Proprio history prefix: 0=K=1 (default, align VLA past_w=1), 1=K=10.
OBS_HIST="${OBS_HIST:-0}"
export PHI0_DISTILL_OBS_HIST="${OBS_HIST}"
# RSI fall: pelvis |Δz|>EXPERT_FALL_Z_ERR for all drive modes (see distill_pelvis_dz_fall).
TRACK_BODY_ERR="${TRACK_BODY_ERR:-0.25}"
# RSI hybrid: rsi_start=random + 50% episode frame0.
export PHI0_RSI_EP0_PROB="${PHI0_RSI_EP0_PROB:-0.5}"
DAGGER_BETA_START="${DAGGER_BETA_START:-1.0}"
DAGGER_BETA_END="${DAGGER_BETA_END:-0.5}"
DAGGER_BETA_SCHEDULE="${DAGGER_BETA_SCHEDULE:-linear}"
# Default anneal=0 → expert-only (no β blend). Ablation: set ANNEAL_STEPS>0 + STUDENT_DRIVE=1.
# Host-resident allowlist + global random ep (disables rg_scan). Required for
# BoneSEED full distill on this branch (no rg_scan / dual_rg).
RESIDENT_REF="${RESIDENT_REF:-${PHI0_DISTILL_RESIDENT:-1}}"
export RESIDENT_REF
# Same thr as wl3 expert gate (fall=0.35 + xy_thr=1.0).
EXPERT_FALL_Z_ERR="${EXPERT_FALL_Z_ERR:-0.35}"
EPISODE_ALLOWLIST="${EPISODE_ALLOWLIST:-${PHI0_ROOT}/meta/boneseed_wl3_xy1m_from_wl2_B8192_ngpu8_20260804_122014.json}"
# demo5skill: distinct-prompt lang cache (all_vlm every-layer cross-attn).
if [[ "${EPISODE_ALLOWLIST}" == *demo5skill* ]]; then
  export PHI0_LANG_LATENTS_DIRNAME="${PHI0_LANG_LATENTS_DIRNAME:-lang_latents_qwen3vl_demo5skill}"
fi
_MODE_TAG="rg"
if [[ "${RESIDENT_REF}" == "1" || "${RESIDENT_REF}" == "true" || "${RESIDENT_REF}" == "yes" ]]; then
  _MODE_TAG="resident"
fi
_HIST_TAG=""
if [[ "${OBS_HIST}" == "0" || "${OBS_HIST}" == "false" || "${OBS_HIST}" == "no" ]]; then
  _HIST_TAG="_nohist"
fi
PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/newton_boneseed_wl_fabric${NGPU}_e${EPOCHS}_h${HORIZON}_b${NUM_ENVS}_l${ACT_LAYERS}_${_MODE_TAG}${_HIST_TAG}_${STAMP}}"
export PHI0_DISTILL_OUT

if [[ ! -f "${REF_ROOT}/meta/stats.json" ]]; then
  echo "[boneseed_full] missing ${REF_ROOT}/meta/stats.json" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT}/meta/lang_latents_qwen3vl" ]]; then
  echo "[boneseed_full] missing ${REF_ROOT}/meta/lang_latents_qwen3vl" >&2
  exit 1
fi
if [[ ! -f "${EPISODE_ALLOWLIST}" ]]; then
  echo "[boneseed_full] missing allowlist ${EPISODE_ALLOWLIST}" >&2
  exit 1
fi

# EPOCHS = RG-list passes; each RG runs ceil(allowlist_sum_len / B_per_gpu) sim steps
# on every rank (DDP replicate). Cover/ETA/CKPT must use NUM_ENVS (= loop B), NOT GLOBAL_B.
GLOBAL_B=$((NUM_ENVS * NGPU))
FRAME_CACHE="${EPISODE_ALLOWLIST%.json}_frame_count.json"
read -r N_EP N_FRAMES N_RG FRAME_CACHED < <("${PYTHON_BIN}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, r"""${PHI0_ROOT}/src""")
from phi0.online.isaac_loop import load_episode_allowlist
from phi0.online.lazy_ref import count_allowlist_frames
allow_p = Path(r"""${EPISODE_ALLOWLIST}""")
allow = load_episode_allowlist(allow_p)
info = count_allowlist_frames(
    r"""${REF_ROOT}""",
    allow,
    cache_path=r"""${FRAME_CACHE}""",
    allowlist_path=allow_p,
)
print(int(info["n_ep"]), int(info["n_frames"]), int(info["n_rg"]), int(info.get("cached", 0)))
PY
)
if [[ "${N_FRAMES}" -lt 1 ]]; then
  echo "[boneseed_full] n_frames=0 for allowlist — abort" >&2
  exit 1
fi
# Match isaac_loop: steps_per_epoch = ceil(n_frames / B) per rank.
STEPS_PER_EPOCH=$(( (N_FRAMES + NUM_ENVS - 1) / NUM_ENVS ))
[[ "${STEPS_PER_EPOCH}" -lt 1 ]] && STEPS_PER_EPOCH=1
ETA_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
: "${DAGGER_BETA_ANNEAL_STEPS:=0}"
export DAGGER_BETA_START DAGGER_BETA_END DAGGER_BETA_ANNEAL_STEPS DAGGER_BETA_SCHEDULE
export PHI0_DAGGER_BETA_START="${DAGGER_BETA_START}"
export PHI0_DAGGER_BETA_END="${DAGGER_BETA_END}"
export PHI0_DAGGER_BETA_ANNEAL_STEPS="${DAGGER_BETA_ANNEAL_STEPS}"
export PHI0_DAGGER_BETA_SCHEDULE="${DAGGER_BETA_SCHEDULE}"
# Default: every ½ per-rank cover epoch; overwrites phi0_student_last.pt
# (weights) + phi0_student_last_optim.pt (AdamW sidecar).
# Tiny allowlists (demo5: spe≈26) make half-epoch=13 → spam ~1GB saves; floor to
# ~5% of job (min 500) on long runs. Override with CKPT_EVERY=N.
if [[ -z "${CKPT_EVERY:-}" ]]; then
  CKPT_EVERY=$(( (STEPS_PER_EPOCH + 1) / 2 ))
  if [[ "${ETA_STEPS}" -ge 5000 && "${CKPT_EVERY}" -lt 500 ]]; then
    CKPT_EVERY=$(( ETA_STEPS / 20 ))
    [[ "${CKPT_EVERY}" -lt 500 ]] && CKPT_EVERY=500
  fi
fi
[[ "${CKPT_EVERY}" -lt 1 ]] && CKPT_EVERY=1

mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"

LANG_ARGS=(--use_lang_latent_cache)
if [[ "${USE_LANG_LATENT_CACHE}" == "0" || "${USE_LANG_LATENT_CACHE}" == "false" ]]; then
  LANG_ARGS=(--no_lang_latent_cache)
fi

RG_ARGS=(--rg_scan)
EST_CPU_GB=$(awk -v n="${N_FRAMES}" 'BEGIN{printf "%.1f", n*1600/1024/1024/1024}')
if [[ "${RESIDENT_REF}" == "1" || "${RESIDENT_REF}" == "true" || "${RESIDENT_REF}" == "yes" ]]; then
  RG_ARGS=(--resident_ref --no_rg_scan)
  # Rank0-only parquet → local NVMe cache; followers attach (single-node).
  export PHI0_RESIDENT_SHARE="${PHI0_RESIDENT_SHARE:-1}"
  export PHI0_RESIDENT_MMAP="${PHI0_RESIDENT_MMAP:-1}"
  export PHI0_RESIDENT_LOAD_WORKERS="${PHI0_RESIDENT_LOAD_WORKERS:-16}"
  export PHI0_RESIDENT_CACHE_ROOT="${PHI0_RESIDENT_CACHE_ROOT:-${PHI0_ROOT}/cache/resident}"
  echo "[boneseed_full] resident_ref=1 est_cpu_GB≈${EST_CPU_GB} (mmap share; was ×${NGPU} private pin) — no rg_scan/dual_rg"
  echo "[boneseed_full] PHI0_RESIDENT_SHARE=${PHI0_RESIDENT_SHARE} mmap=${PHI0_RESIDENT_MMAP} workers=${PHI0_RESIDENT_LOAD_WORKERS} cache_root=${PHI0_RESIDENT_CACHE_ROOT}"
fi

# No legacy action_encoder on learnable-query student → DDP find_unused off.
export PHI0_DISTILL_FIND_UNUSED="${PHI0_DISTILL_FIND_UNUSED:-0}"
echo "[boneseed_full] PHI0_DISTILL_FIND_UNUSED=${PHI0_DISTILL_FIND_UNUSED}"

# Sonic ONNX encode is ~half of step; torch.compile (fallback to eager on fail).
export PHI0_ONNX_ENCODE_COMPILE="${PHI0_ONNX_ENCODE_COMPILE:-1}"
echo "[boneseed_full] PHI0_ONNX_ENCODE_COMPILE=${PHI0_ONNX_ENCODE_COMPILE}"

echo "[boneseed_full] out=${PHI0_DISTILL_OUT}"
echo "[boneseed_full] ref=${REF_ROOT} z_star_stats=${Z_STAR_STATS_PATH} ngpu=${NGPU} B/gpu=${NUM_ENVS} H=${HORIZON} act_layers=${ACT_LAYERS} epochs=${EPOCHS} lr=${LR} rsi=${RSI_START}"
echo "[boneseed_full] mode=epoch_cover allowlist=${EPISODE_ALLOWLIST} n_ep=${N_EP} n_frames=${N_FRAMES} n_rg=${N_RG} frame_cache=${FRAME_CACHED}"
echo "[boneseed_full] B/gpu=${NUM_ENVS} Bglobal=${GLOBAL_B} cover≈${STEPS_PER_EPOCH}/ep/rank eta≈${ETA_STEPS} (epochs×cover) ckpt_every=${CKPT_EVERY}"
echo "[boneseed_full] lang_cache=${USE_LANG_LATENT_CACHE} fabric=newton_kitless slide_gt=1 stride=1"
echo "[boneseed_full] obs_hist=${OBS_HIST} (1=K10, 0=K1)"
echo "[boneseed_full] zero_action=${PHI0_DISTILL_ZERO_ACTION:-0} (1=act[64:93]≡0)"
if [[ "${RSI_START}" == "random" ]]; then
  echo "[boneseed_full] rsi_ep0_prob=${PHI0_RSI_EP0_PROB:-0.5}"
fi
echo "[boneseed_full] adaln_mode=${PHI0_ADALN_MODE} fourier_max_cycles=${PHI0_FOURIER_MAX_CYCLES}"
if [[ "${PHI0_DISTILL_STUDENT_DRIVE:-0}" == "1" || "${STUDENT_DRIVE:-0}" == "1" || "${STUDENT_DRIVE:-}" == "true" ]]; then
  export PHI0_DISTILL_STUDENT_DRIVE=1
  STUDENT_DRIVE=1
  if [[ "${DAGGER_BETA_ANNEAL_STEPS}" -gt 0 ]] || [[ "${DAGGER_BETA_START}" != "1.0" && "${DAGGER_BETA_START}" != "1" ]]; then
    echo "[boneseed_full] student_drive=1 dagger_β=${DAGGER_BETA_START}→${DAGGER_BETA_END} anneal=${DAGGER_BETA_ANNEAL_STEPS} sched=${DAGGER_BETA_SCHEDULE}"
  else
    echo "[boneseed_full] student_drive=1 (β=0 fixed) track_body_err=${TRACK_BODY_ERR:-0.25}"
  fi
else
  export PHI0_DISTILL_STUDENT_DRIVE=0
  STUDENT_DRIVE=0
  echo "[boneseed_full] expert_drive fall_z=${EXPERT_FALL_Z_ERR:-0.35}"
fi
echo "[boneseed_full] RSI_fall=pelvis_dz thr=${EXPERT_FALL_Z_ERR:-0.35} rsi=${RSI_START}"
if [[ "${RESIDENT_REF}" == "1" || "${RESIDENT_REF}" == "true" || "${RESIDENT_REF}" == "yes" ]]; then
  echo "[boneseed_full] dual_rg=0 resident_ref=1 (host pin; rsi=random|random_episode; root_goal_cond=off)"
else
  echo "[boneseed_full] dual_rg=1 (peak RAM≈2 RG; retire+drain across RG boundary; root_goal_cond=off)"
fi
echo "[boneseed_full] torchrun nproc=${NGPU} master=${MASTER_ADDR}:${MASTER_PORT} cvd=${CUDA_VISIBLE_DEVICES}"

SD_ARGS=()
if [[ "${STUDENT_DRIVE:-0}" == "1" || "${STUDENT_DRIVE:-}" == "true" || "${PHI0_DISTILL_STUDENT_DRIVE:-0}" == "1" ]]; then
  SD_ARGS=(--student_drive)
fi
HIST_ARGS=(--obs_hist)
if [[ "${OBS_HIST}" == "0" || "${OBS_HIST}" == "false" || "${OBS_HIST}" == "no" ]]; then
  HIST_ARGS=(--no-obs_hist)
fi
DAGGER_ARGS=(
  --dagger_beta_start "${DAGGER_BETA_START}"
  --dagger_beta_end "${DAGGER_BETA_END}"
  --dagger_beta_anneal_steps "${DAGGER_BETA_ANNEAL_STEPS}"
  --dagger_beta_schedule "${DAGGER_BETA_SCHEDULE}"
)

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PHI0_ROOT}/tools/train/newton_boneseed_distill_fabric.py" \
  --groot-root "${GR00T_ROOT}" \
  --num_envs "${NUM_ENVS}" \
  --epochs "${EPOCHS}" \
  --horizon "${HORIZON}" \
  --ckpt_every "${CKPT_EVERY}" \
  --lr "${LR}" \
  --rsi_start "${RSI_START}" \
  --ref_root "${REF_ROOT}" \
  "${RG_ARGS[@]}" \
  "${LANG_ARGS[@]}" \
  --episode_allowlist "${EPISODE_ALLOWLIST}" \
  --expert_fall_z_err "${EXPERT_FALL_Z_ERR}" \
  "${SD_ARGS[@]}" \
  "${HIST_ARGS[@]}" \
  "${DAGGER_ARGS[@]}" \
  --track_body_err "${TRACK_BODY_ERR:-0.25}" \
  --out_dir "${PHI0_DISTILL_OUT}" \
  --headless \
  "$@"
