#!/usr/bin/env bash
# Unified online_vlm mix distill (820demo_mix_release_unified).
# Fuses teleop (has_video) + BoneSEED demo5skill text-only in ONE loop / one student.
#
# Default recipe (≠ LOCKED H=1 gold / ≠ LOCKED video-DAgger / ≠ Isaac-hold vision):
#   H=32, AdaLN progress_only, PHI0_FOURIER_MAX_CYCLES=64, interleave_vlm,
#   prompt_max_length=256, disk unified sonic z*.
#   Cross-attn: PHI0_ACTION_CROSS_ATTN_MODE=interleave_vlm（even；odd 无实质差、all_vlm 无优势：
#     docs/report/training/interleave_vs_all_vlm_stand_egypt_s2.md）.
#   PHI0_VISION_DATALOADER=1 (yjh-style PickTissue DataLoader dual feed):
#     has_video  → DataLoader images → freeze VLM → ChunkStudent BC (no Isaac phys)
#     no_video   → Newton sim body + text VLM + β=0.8 DAgger → ATM decode → PD
#   PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY=1 (default when vision_dl=1): Isaac resident
#     demo5 no_video only; has_video → DataLoader (not Isaac resident).
#   LOCKED recover: PHI0_VISION_DATALOADER=0 PHI0_MODALITY_SKIP_SIM_VIDEO=0
#     PHI0_DAGGER_ON_NO_VIDEO=0 (+ optional interleave via run_820mix_interleave_distill.sh).
#
#   NGPU=8 NUM_ENVS=8 bash tools/train/run_online_vlm_mix_distill.sh
#   VISION_ONLY=1 …   # regress: vision_episode_allowlist only (no text Isaac)
#   DRY_RUN=1 bash tools/train/run_online_vlm_mix_distill.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
sonic_apply_simulator "${SIMULATOR:-newton}"
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath
export PHI0_FABRIC=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29581}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NGPU="${NGPU:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

export PHI0_TRAIN_MODE=online_vlm
export USE_VLM=1
# disk (default: 820 release_unified[396:460) GT) | online (onnx_g1 encode)
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE:-disk}"
export PHI0_ADALN_MODE="${PHI0_ADALN_MODE:-progress_only}"
# none|progress_vlm_age|progress_only
# aliases offset_vlm_age|offset_only|offset → progress_*
export PHI0_FOURIER_MAX_CYCLES="${PHI0_FOURIER_MAX_CYCLES:-64}"
# Vision samples: hard-zero AdaLN modulation (t_mod=0). Text keeps cap=64 Fourier.
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
# Even layers cross→VLM. all_vlm is opt-in only.
export PHI0_ACTION_CROSS_ATTN_MODE="${PHI0_ACTION_CROSS_ATTN_MODE:-interleave_vlm}"
# VLM hold: 1 = every-step current-frame encode (no latent cache). Legacy 5Hz: 10.
export PHI0_VLM_HOLD_PERIOD="${PHI0_VLM_HOLD_PERIOD:-1}"
export PHI0_VLM_REFRESH_EVERY_STEP="${PHI0_VLM_REFRESH_EVERY_STEP:-1}"
# Mix recipe: keep left thumb_aux from tape (do not force proprio41[30]=0).
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-0}"
# Multimodal input_ids cap (vision pads + text). Mix tasks shortened for dual≤256.
export PHI0_PROMPT_MAX_LENGTH="${PHI0_PROMPT_MAX_LENGTH:-256}"
# Vision DataLoader feed (≠ LOCKED Isaac-hold dual VLM).
export PHI0_VISION_DATALOADER="${PHI0_VISION_DATALOADER:-1}"
# Psi0 sonic/finetune DataLoader hardcodes num_workers=12.
export PHI0_VISION_DL_WORKERS="${PHI0_VISION_DL_WORKERS:-12}"
# Bernoulli: vision_dl vs text Isaac. Default 0.8 → ~20% no-video Isaac steps.
export PHI0_P_VISION="${PHI0_P_VISION:-0.8}"
# Closed-loop DAgger on has_video Isaac (LOCKED). Old mix recipe was
# skip_sim_video=1 + dagger_on_no_video=1 (text-only mix, 5 BoneSEED eps).
export PHI0_MODALITY_SKIP_SIM_VIDEO="${PHI0_MODALITY_SKIP_SIM_VIDEO:-0}"
export PHI0_DAGGER_ON_NO_VIDEO="${PHI0_DAGGER_ON_NO_VIDEO:-0}"
# vision_dl=1: Isaac resident = no_video demo5 only (has_video → DataLoader).
if [[ "${PHI0_VISION_DATALOADER}" == "1" ]]; then
  export PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY="${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY:-1}"
else
  export PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY="${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY:-0}"
fi
# demo5 BoneSEED Isaac has no hand_ref; explicit warn path (≠ silent zero on teleop).
if [[ "${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY}" == "1" ]]; then
  export PHI0_ALLOW_ZERO_HAND="${PHI0_ALLOW_ZERO_HAND:-1}"
else
  export PHI0_ALLOW_ZERO_HAND="${PHI0_ALLOW_ZERO_HAND:-0}"
fi
# Disable random interleave as primary path (vision_dl + modality gate replace it).
export PHI0_INTERLEAVE_P_PROPRIO="${PHI0_INTERLEAVE_P_PROPRIO:-1}"
# Quiet train logs: skip onnx compile spam; short FA fallback line; mute warnings.
export PHI0_DISTILL_QUIET="${PHI0_DISTILL_QUIET:-1}"
export PHI0_ONNX_ENCODE_COMPILE="${PHI0_ONNX_ENCODE_COMPILE:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
# Train+infer VLM: flash_attention_2 (Newton env has flash-attn). Opt-in sdpa only.
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_HAND_MODE="${PHI0_HAND_MODE:-revo2}"
export PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2:-$([[ "${PHI0_HAND_MODE}" == revo2 ]] && echo 1 || echo 0)}"
# Throughput knobs (defaults keep semantics / safety):
#   PHI0_VIDEO_DECODER_LRU=32  — reuse torchcodec handles across RSI ep jumps (0=off)
#   PHI0_DISTILL_NO_BARRIER=1  — skip pre-bwd barrier (default on for vision_dl throughput)
#   PHI0_VLM_REFRESH_PAD=0     — pad VLM refresh to world-max (adds idle-rank work; opt-in)
#   PHI0_DISTILL_PROFILE=0     — set 1 to print [enc/isaac/student/barrier %]
export PHI0_VIDEO_DECODER_LRU="${PHI0_VIDEO_DECODER_LRU:-32}"
export PHI0_DISTILL_NO_BARRIER="${PHI0_DISTILL_NO_BARRIER:-1}"
export PHI0_VLM_REFRESH_PAD="${PHI0_VLM_REFRESH_PAD:-0}"
# RSI_START=random: PHI0_RSI_EP0_PROB = P(frame0) vs mid-ep (default 0.5).
# RSI_START=random_episode: always frame0 (ignores ep0_prob).
export PHI0_RSI_EP0_PROB="${PHI0_RSI_EP0_PROB:-0.5}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
export PHI0_DISTILL_OBS_HIST="${OBS_HIST:-0}"
export PHI0_DISTILL_STATE_DROPOUT="${PHI0_DISTILL_STATE_DROPOUT:-0.2}"
# DAgger: β = teacher weight in z_exec = β·z* + (1-β)·ẑ → decode (default 0.8→80/20).
STUDENT_DRIVE="${STUDENT_DRIVE:-1}"
if [[ "${STUDENT_DRIVE}" == "0" || "${STUDENT_DRIVE}" == "false" || "${STUDENT_DRIVE}" == "no" || "${STUDENT_DRIVE}" == "off" ]]; then
  export PHI0_DISTILL_STUDENT_DRIVE=0
  STUDENT_DRIVE=0
else
  export PHI0_DISTILL_STUDENT_DRIVE=1
  STUDENT_DRIVE=1
fi
DAGGER_BETA_START="${DAGGER_BETA_START:-0.8}"
DAGGER_BETA_END="${DAGGER_BETA_END:-${DAGGER_BETA_START}}"
DAGGER_BETA_ANNEAL_STEPS="${DAGGER_BETA_ANNEAL_STEPS:-0}"
DAGGER_BETA_SCHEDULE="${DAGGER_BETA_SCHEDULE:-linear}"
export PHI0_DAGGER_BETA_START="${DAGGER_BETA_START}"
export PHI0_DAGGER_BETA_END="${DAGGER_BETA_END}"
export PHI0_DAGGER_BETA_ANNEAL_STEPS="${DAGGER_BETA_ANNEAL_STEPS}"
export PHI0_DAGGER_BETA_SCHEDULE="${DAGGER_BETA_SCHEDULE}"
TRACK_BODY_ERR="${TRACK_BODY_ERR:-0.25}"
export PHI0_ONNX_ENCODE_COMPILE="${PHI0_ONNX_ENCODE_COMPILE:-1}"
export PHI0_DISTILL_FIND_UNUSED="${PHI0_DISTILL_FIND_UNUSED:-0}"
# live (default deploy) | tape (= offline reencode fk_root anchor); ignored when TEACHER_Z=disk
export PHI0_ZSTAR_ANCHOR="${PHI0_ZSTAR_ANCHOR:-live}"
# joint_pd (Newton default) | direct_latent (ATM hist decoder → lowcmd-style PD; closer to deploy)
export PHI0_DISTILL_EXPERT_DRIVE="${EXPERT_DRIVE:-${PHI0_DISTILL_EXPERT_DRIVE:-}}"
# Default: obs corruption + push_robot (opt out: NO_DR=1).
export NO_DR="${NO_DR:-0}"

REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_mix_release_unified}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
export ACTION_STATS_PATH
# Disk GT (820 release sonic in action.unified): default no boneseed z overlay.
if [[ "${TEACHER_Z_SOURCE}" == "disk" ]]; then
  Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-none}"
else
  Z_STAR_STATS_PATH="${Z_STAR_STATS_PATH:-${PHI0_ROOT}/meta/z_star_stats_boneseed.json}"
fi
export Z_STAR_STATS_PATH
# Default: all 613 ep (demo5skill text + skill dual). VISION_ONLY=1 → 608 video-only.
if [[ -z "${EPISODE_ALLOWLIST:-}" ]]; then
  if [[ "${VISION_ONLY:-0}" == "1" || "${VISION_ONLY:-}" == "true" ]]; then
    EPISODE_ALLOWLIST="${REF_ROOT}/meta/vision_episode_allowlist.json"
  else
    EPISODE_ALLOWLIST="${REF_ROOT}/meta/all_episode_allowlist.json"
  fi
fi
NUM_ENVS="${NUM_ENVS:-8}"
export NUM_ENVS
HORIZON="${HORIZON:-32}"
EPOCHS="${EPOCHS:-1}"
# Default: 50% ep frame0 / 50% mid. Always-frame0: RSI_START=random_episode
RSI_START="${RSI_START:-random}"
LR="${LR:-1e-4}"
# HF Trainer LR: cosine + warmup_ratio (GR00T default 0.05). warmup_steps>0 overrides ratio.
export PHI0_LR_SCHEDULER="${PHI0_LR_SCHEDULER:-cosine}"
export PHI0_LR_WARMUP_STEPS="${PHI0_LR_WARMUP_STEPS:-0}"
export PHI0_LR_WARMUP_RATIO="${PHI0_LR_WARMUP_RATIO:-0.05}"
W_Z="${W_Z:-1}"
# Default: sonic latent BC only (no q_head / no q_head bar). Opt-in: W_Q_HEAD=1.
W_Q_HEAD="${W_Q_HEAD:-0}"
W_SMPL="${W_SMPL:-0}"
W_HAND="${W_HAND:-1}"
EXPERT_FALL_Z_ERR="${EXPERT_FALL_Z_ERR:-0.35}"
RESIDENT_REF="${RESIDENT_REF:-1}"
export RESIDENT_REF
export W_Z W_SMPL W_Q_HEAD W_HAND
# Train Newton-GL viz (optional). STUDENT_CKPT resumes weights into a new OUT.
RECORD_TRAIN_MP4="${RECORD_TRAIN_MP4:-0}"
# 2 → mp4 fps = 50/2 = 25 (near realtime). Sparse A/B: RECORD_EVERY=25 → 2fps.
RECORD_EVERY="${RECORD_EVERY:-2}"
STUDENT_CKPT="${STUDENT_CKPT:-}"
# Resume: train this many more steps past ckpt ``steps_done`` (else epoch×cover stop).
if [[ -n "${EXTRA_STEPS:-}" ]]; then
  export PHI0_EXTRA_STEPS="${EXTRA_STEPS}"
fi

PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/online_vlm_820mix_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
export PHI0_DISTILL_OUT
LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/online_vlm_820mix_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"

if [[ "${HORIZON}" -lt 1 ]]; then
  echo "[online_vlm] HORIZON must be >=1 (got ${HORIZON})" >&2
  exit 1
fi
if [[ ! -f "${REF_ROOT}/meta/stats.json" ]]; then
  echo "[online_vlm] missing ${REF_ROOT}/meta/stats.json" >&2
  exit 1
fi
if [[ ! -f "${EPISODE_ALLOWLIST}" ]]; then
  echo "[online_vlm] missing allowlist ${EPISODE_ALLOWLIST}" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT}/videos" ]]; then
  echo "[online_vlm] missing dual videos under ${REF_ROOT}/videos" >&2
  exit 1
fi

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
  echo "[online_vlm] n_frames=0 for allowlist — abort" >&2
  exit 1
fi
STEPS_PER_EPOCH=$(( (N_FRAMES + NUM_ENVS - 1) / NUM_ENVS ))
[[ "${STEPS_PER_EPOCH}" -lt 1 ]] && STEPS_PER_EPOCH=1
ETA_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
# Default: overwrite ``phi0_student_last.pt`` every 2000 steps (no step snapshots).
# Overwrite-only ``phi0_student_last.pt`` every N steps (no step snapshots unless KEEP>0).
CKPT_EVERY="${CKPT_EVERY:-10000}"
[[ "${CKPT_EVERY}" -lt 1 ]] && CKPT_EVERY=1
export PHI0_CKPT_STEP_KEEP="${PHI0_CKPT_STEP_KEEP:-0}"

mkdir -p "${PHI0_DISTILL_OUT}" "$(dirname "${LOG_FILE}")"
cd "${GR00T_ROOT}"

export PHI0_RESIDENT_SHARE="${PHI0_RESIDENT_SHARE:-1}"
export PHI0_RESIDENT_MMAP="${PHI0_RESIDENT_MMAP:-1}"
export PHI0_RESIDENT_LOAD_WORKERS="${PHI0_RESIDENT_LOAD_WORKERS:-16}"
export PHI0_RESIDENT_CACHE_ROOT="${PHI0_RESIDENT_CACHE_ROOT:-${PHI0_ROOT}/cache/resident}"

cat > "${PHI0_DISTILL_OUT}/RESOLVED_TRAIN_SETTINGS.yaml" <<EOF
repo: Phi_0_wpy
train_mode: online_vlm
fuse_has_video: true
vision_only: ${VISION_ONLY:-0}
ref_root: ${REF_ROOT}
episode_allowlist: ${EPISODE_ALLOWLIST}
action_stats_path: ${ACTION_STATS_PATH}
z_star_stats_path: ${Z_STAR_STATS_PATH}
batch_size_per_gpu: ${NUM_ENVS}
ngpu: ${NGPU}
effective_batch: $((NUM_ENVS * NGPU))
horizon: ${HORIZON}
epochs: ${EPOCHS}
learning_rate: ${LR}
adaln_mode: ${PHI0_ADALN_MODE}
fourier_max_cycles: ${PHI0_FOURIER_MAX_CYCLES:-64}
adaln_zero_vision: ${PHI0_ADALN_ZERO_VISION:-1}
action_cross_attn_mode: ${PHI0_ACTION_CROSS_ATTN_MODE:-interleave_vlm}
vlm_hold_period: ${PHI0_VLM_HOLD_PERIOD}
vlm_refresh_every_step: ${PHI0_VLM_REFRESH_EVERY_STEP}
zero_proprio_left_thumb_aux: ${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX}
prompt_max_length: ${PHI0_PROMPT_MAX_LENGTH}
rsi_start: ${RSI_START}
rsi_ep0_prob: ${PHI0_RSI_EP0_PROB}
teacher_z_source: ${TEACHER_Z_SOURCE}
zstar_anchor: ${PHI0_ZSTAR_ANCHOR:-live}
expert_drive: ${PHI0_DISTILL_EXPERT_DRIVE:-recipe_default}
deploy_policy_dir: ${DEPLOY_POLICY_DIR}
hand_mode: ${PHI0_HAND_MODE}
train_hand_obs: ${PHI0_TRAIN_HAND_OBS:-}
hand_proprio_lag: ${PHI0_HAND_PROPRIO_LAG:-1}
zero_proprio_hand: ${PHI0_ZERO_PROPRIO_HAND:-0}
newton_revo2: ${PHI0_NEWTON_REVO2}
w_z: ${W_Z}
w_q_head: ${W_Q_HEAD}
w_smpl: ${W_SMPL}
w_hand: ${W_HAND}
vlm_attn: ${PHI0_VLM_ATTN}
allow_sdpa_fallback: ${PHI0_ALLOW_SDPA_FALLBACK}
video_decoder_lru: ${PHI0_VIDEO_DECODER_LRU}
distill_no_barrier: ${PHI0_DISTILL_NO_BARRIER}
vlm_refresh_pad: ${PHI0_VLM_REFRESH_PAD}
obs_hist: ${PHI0_DISTILL_OBS_HIST}
state_dropout: ${PHI0_DISTILL_STATE_DROPOUT}
student_drive: ${STUDENT_DRIVE}
dagger_beta_start: ${DAGGER_BETA_START}
dagger_beta_end: ${DAGGER_BETA_END}
dagger_beta_anneal_steps: ${DAGGER_BETA_ANNEAL_STEPS}
dagger_beta_schedule: ${DAGGER_BETA_SCHEDULE}
track_body_err: ${TRACK_BODY_ERR}
student_ckpt: ${STUDENT_CKPT:-}
record_train_mp4: ${RECORD_TRAIN_MP4:-0}
record_every: ${RECORD_EVERY:-2}
no_dr: ${NO_DR:-0}
n_ep: ${N_EP}
n_frames: ${N_FRAMES}
steps_per_epoch: ${STEPS_PER_EPOCH}
interleave_p_proprio: ${PHI0_INTERLEAVE_P_PROPRIO:-1}
vision_dataloader: ${PHI0_VISION_DATALOADER:-1}
vision_dl_workers: ${PHI0_VISION_DL_WORKERS:-12}
p_vision: ${PHI0_P_VISION:-0.8}
lr_scheduler: ${PHI0_LR_SCHEDULER:-cosine}
lr_warmup_steps: ${PHI0_LR_WARMUP_STEPS:-0}
lr_warmup_ratio: ${PHI0_LR_WARMUP_RATIO:-0.05}
extra_steps: ${PHI0_EXTRA_STEPS:-}
modality_skip_sim_video: ${PHI0_MODALITY_SKIP_SIM_VIDEO:-0}
dagger_on_no_video: ${PHI0_DAGGER_ON_NO_VIDEO:-0}
isaac_no_video_only: ${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY}
allow_zero_hand: ${PHI0_ALLOW_ZERO_HAND}
stand_episode_index: ${PHI0_STAND_EPISODE_INDEX:--1}
stand_rsi_prob: ${PHI0_STAND_RSI_PROB:-0.5}
stand_policy: rsi_frame0_p_stand
vlm_frame_latent_cache: ${PHI0_USE_VLM_FRAME_LATENT_CACHE:-0}
vlm_frame_cache_skip_video: ${PHI0_VLM_FRAME_CACHE_SKIP_VIDEO:-1}
use_vlm: ${USE_VLM:-1}
EOF

echo "================================================================"
echo "[online_vlm] fuse has_video DataLoader + text Isaac + teacher_z=${TEACHER_Z_SOURCE} + H=${HORIZON}"
echo "vision_dl=${PHI0_VISION_DATALOADER} p_vision=${PHI0_P_VISION} dagger_on_no_video=${PHI0_DAGGER_ON_NO_VIDEO} modality_skip_sim_video=${PHI0_MODALITY_SKIP_SIM_VIDEO} isaac_no_video_only=${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY} allow_zero_hand=${PHI0_ALLOW_ZERO_HAND}"
if [[ "${TEACHER_Z_SOURCE}" == "disk" ]]; then
  echo "teacher_z=disk → unified[396:460] (no onnx_g1 encode on hot path)"
else
  echo "teacher_z=online → onnx_g1 SonicMotionEncoder"
fi
echo "REF=${REF_ROOT}"
echo "ALLOW=${EPISODE_ALLOWLIST}"
echo "OUT=${PHI0_DISTILL_OUT}"
echo "LOG=${LOG_FILE}"
echo "B=${NUM_ENVS} × NGPU=${NGPU} epochs=${EPOCHS} rsi=${RSI_START} ep0=${PHI0_RSI_EP0_PROB} drive=${PHI0_DISTILL_EXPERT_DRIVE:-recipe}"
if [[ "${STUDENT_DRIVE}" == "1" ]]; then
  if [[ "${PHI0_DAGGER_ON_NO_VIDEO}" == "1" ]]; then
    echo "dagger: student_drive=1 β=${DAGGER_BETA_START}→${DAGGER_BETA_END} (text Isaac only when vision_dl=1)"
  else
    echo "dagger: student_drive=1 β=${DAGGER_BETA_START}→${DAGGER_BETA_END} anneal=${DAGGER_BETA_ANNEAL_STEPS}"
  fi
else
  echo "dagger: expert-only (STUDENT_DRIVE=0)"
fi
echo "n_ep=${N_EP} n_frames=${N_FRAMES} cover≈${STEPS_PER_EPOCH}/ep ckpt_every=${CKPT_EVERY}"
if [[ -n "${PHI0_EXTRA_STEPS:-}" ]]; then
  echo "extra_steps=${PHI0_EXTRA_STEPS} (stop at resume_steps + extra)"
fi
echo "================================================================"

VLM_CLI_FLAG=(--use_vlm)
if [[ "${PHI0_USE_VLM_FRAME_LATENT_CACHE:-0}" == "1" || "${PHI0_USE_VLM_FRAME_LATENT_CACHE:-}" == "true" ]]; then
  VLM_CLI_FLAG=(--no_use_vlm)
  export USE_VLM=0
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NGPU}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  "${PHI0_ROOT}/tools/train/newton_boneseed_distill_fabric.py"
  --groot-root "${GR00T_ROOT}"
  --num_envs "${NUM_ENVS}"
  --epochs "${EPOCHS}"
  --horizon "${HORIZON}"
  --ckpt_every "${CKPT_EVERY}"
  --lr "${LR}"
  --rsi_start "${RSI_START}"
  --ref_root "${REF_ROOT}"
  --episode_allowlist "${EPISODE_ALLOWLIST}"
  --resident_ref
  --no_rg_scan
  "${VLM_CLI_FLAG[@]}"
  --no_lang_latent_cache
  --teacher_z_source "${TEACHER_Z_SOURCE}"
  --action_stats_path "${ACTION_STATS_PATH}"
  --w_z "${W_Z}"
  --w_smpl "${W_SMPL}"
  --w_q_head "${W_Q_HEAD}"
  --w_hand "${W_HAND}"
  --expert_fall_z_err "${EXPERT_FALL_Z_ERR}"
  --dagger_beta_start "${DAGGER_BETA_START}"
  --dagger_beta_end "${DAGGER_BETA_END}"
  --dagger_beta_anneal_steps "${DAGGER_BETA_ANNEAL_STEPS}"
  --dagger_beta_schedule "${DAGGER_BETA_SCHEDULE}"
  --track_body_err "${TRACK_BODY_ERR}"
  --out_dir "${PHI0_DISTILL_OUT}"
  --simulator newton
  --headless
)

if [[ "${STUDENT_DRIVE}" == "1" ]]; then
  CMD+=(--student_drive)
fi

# Newton-GL train viz → ${PHI0_DISTILL_OUT}/distill_train_viz.mp4 (follows env0).
if [[ "${RECORD_TRAIN_MP4}" == "1" || "${RECORD_TRAIN_MP4}" == "true" ]]; then
  export PHI0_DISTILL_MP4_MAX_FRAMES="${PHI0_DISTILL_MP4_MAX_FRAMES:-200}"
  export PHI0_DISTILL_MP4_FLUSH="${PHI0_DISTILL_MP4_FLUSH:-25}"
  CMD+=(--record_train_mp4 --record_every "${RECORD_EVERY}")
  echo "[online_vlm] RECORD_TRAIN_MP4=1 every=${RECORD_EVERY} "\
"max_frames=${PHI0_DISTILL_MP4_MAX_FRAMES} flush=${PHI0_DISTILL_MP4_FLUSH}"
fi

if [[ -n "${STUDENT_CKPT}" ]]; then
  CMD+=(--student_ckpt "${STUDENT_CKPT}")
  echo "[online_vlm] resume student_ckpt=${STUDENT_CKPT}"
fi

# Default NO_DR=0 → DR on (corruption + push). Opt out: NO_DR=1.
if [[ "${NO_DR}" == "1" || "${NO_DR}" == "true" ]]; then
  CMD+=(--no_dr)
  echo "[online_vlm] NO_DR=1 (obs corruption + push off)"
else
  echo "[online_vlm] NO_DR=0 (obs corruption + push_robot on)"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
