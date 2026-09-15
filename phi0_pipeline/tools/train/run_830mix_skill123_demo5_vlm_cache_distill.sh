#!/usr/bin/env bash
# 830mix skill1+2+3 teleop + 820demo demo5skill (BoneSEED text) → VLM frame cache mix.
#
#   EPOCHS=3 PHI0_P_VISION=0.9 bash tools/train/run_830mix_skill123_demo5_vlm_cache_distill.sh
#
# Expects merged pack with dual VLM cache on vision eps + lang_latents for demo5.
# Build via: bash tools/data/run_830mix_skill123_demo5_pipeline.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

REF_ROOT="${REF_ROOT:-/mnt/data3/wpy/datasets/830/830mix_skill123_demo5_unified}"
NGPU="${NGPU:-8}"
NUM_ENVS="${NUM_ENVS:-32}"
EPOCHS="${EPOCHS:-3}"
HORIZON="${HORIZON:-32}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export REF_ROOT CKPT_EVERY CUDA_VISIBLE_DEVICES
export PHI0_CKPT_STEP_KEEP="${PHI0_CKPT_STEP_KEEP:-0}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE=1
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1
export USE_VLM=0
# 90% vision_dl / 10% text Isaac (no-vision fraction 0.1)
export PHI0_P_VISION="${PHI0_P_VISION:-0.9}"
export PHI0_VISION_DATALOADER=1
export PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY=1
export PHI0_MODALITY_SKIP_SIM_VIDEO=1
export PHI0_DAGGER_ON_NO_VIDEO=1
export VISION_ONLY=0
export PHI0_VISION_DL_ONLY=0
export PHI0_RESIDENT_CACHE_FORCE=1
export TEACHER_Z_SOURCE=disk
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export PHI0_HAND_MODE=dex3
export PHI0_NEWTON_REVO2=0
export PHI0_TRAIN_HAND_OBS="${PHI0_TRAIN_HAND_OBS:-commanded}"
export PHI0_HAND_PROPRIO_LAG="${PHI0_HAND_PROPRIO_LAG:-1}"
export PHI0_ZERO_PROPRIO_HAND="${PHI0_ZERO_PROPRIO_HAND:-0}"
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export W_Q_HEAD="${W_Q_HEAD:-0}"
export PHI0_DISTILL_NO_BARRIER="${PHI0_DISTILL_NO_BARRIER:-1}"
export W_HAND="${W_HAND:-1}"
# demo5 BoneSEED Isaac has no hand_ref
export PHI0_ALLOW_ZERO_HAND=1
export STUDENT_DRIVE="${STUDENT_DRIVE:-1}"
export DAGGER_BETA_START="${DAGGER_BETA_START:-0.8}"
export DAGGER_BETA_END="${DAGGER_BETA_END:-0.8}"
export PHI0_DISTILL_RTC="${PHI0_DISTILL_RTC:-1}"
export PHI0_DISTILL_RTC_MAX_DELAY="${PHI0_DISTILL_RTC_MAX_DELAY:-8}"

# Always stamp a fresh OUT unless caller sets MIX_DISTILL_OUT explicitly
# (bare PHI0_DISTILL_OUT from prior skill launches is ignored).
if [[ -n "${MIX_DISTILL_OUT:-}" ]]; then
  PHI0_DISTILL_OUT="${MIX_DISTILL_OUT}"
else
  PHI0_DISTILL_OUT="/mnt/data3/wpy/830mix_skill123_demo5_handcmd_lag1_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_pv${PHI0_P_VISION}_${STAMP}"
fi
LOG_FILE="${MIX_LOG_FILE:-${PHI0_ROOT}/logs/830mix_skill123_demo5_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_pv${PHI0_P_VISION}_${STAMP}.log}"
export PHI0_DISTILL_OUT LOG_FILE NGPU NUM_ENVS EPOCHS HORIZON STAMP

_VLM_CACHE_DIRNAME="${PHI0_VLM_FRAME_LATENTS_DIRNAME:-vlm_frame_latents_qwen3vl_dual}"
for f in \
  "${REF_ROOT}/meta/stats.json" \
  "${REF_ROOT}/meta/vision_episode_allowlist.json" \
  "${REF_ROOT}/meta/all_episode_allowlist.json" \
  "${REF_ROOT}/meta/${_VLM_CACHE_DIRNAME}/meta.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "[830mix_s123_d5] missing ${f}" >&2
    echo "[830mix_s123_d5] build first: bash tools/data/run_830mix_skill123_demo5_pipeline.sh" >&2
    exit 1
  fi
done

echo "[830mix_s123_d5] REF=${REF_ROOT}"
echo "[830mix_s123_d5] p_vision=${PHI0_P_VISION} isaac_no_video_only=${PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY} allow_zero_hand=${PHI0_ALLOW_ZERO_HAND}"
echo "[830mix_s123_d5] deploy=${DEPLOY_POLICY_DIR} hand=${PHI0_HAND_MODE} lag=${PHI0_HAND_PROPRIO_LAG} rtc=${PHI0_DISTILL_RTC}"
echo "[830mix_s123_d5] B=${NUM_ENVS}×NGPU=${NGPU} H=${HORIZON} epochs=${EPOCHS} CUDA=${CUDA_VISIBLE_DEVICES}"
echo "[830mix_s123_d5] OUT=${PHI0_DISTILL_OUT}"

# Avoid polluted LOG_DIR from prior viz jobs.
exec env -u LOG_DIR bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
