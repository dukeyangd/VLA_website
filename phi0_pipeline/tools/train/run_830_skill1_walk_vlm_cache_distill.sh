#!/usr/bin/env bash
# 830demo skill_1_walk_to_black_box → 830demo_skill1_walk_unified
# vision_dl + dual VLM frame cache (ego + chest/left_wrist alias).
#
#   EPOCHS=2 bash tools/train/run_830_skill1_walk_vlm_cache_distill.sh
#   # if cache missing first:
#   DATASET_ROOT=.../830demo_skill1_walk_unified PROMPT='机器人朝黑箱子走过去。' \
#     bash tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified}"
NGPU="${NGPU:-8}"
NUM_ENVS="${NUM_ENVS:-32}"
EPOCHS="${EPOCHS:-2}"
HORIZON="${HORIZON:-32}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export REF_ROOT CKPT_EVERY CUDA_VISIBLE_DEVICES
export PHI0_CKPT_STEP_KEEP="${PHI0_CKPT_STEP_KEEP:-0}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE=1
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1
export USE_VLM=0
export PHI0_P_VISION=1
export PHI0_VISION_DATALOADER=1
export PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY=0
export PHI0_MODALITY_SKIP_SIM_VIDEO=1
export PHI0_DAGGER_ON_NO_VIDEO=0
export VISION_ONLY=1
export PHI0_VISION_DL_ONLY=1
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
export PHI0_ALLOW_ZERO_HAND=0
export STUDENT_DRIVE="${STUDENT_DRIVE:-1}"
export DAGGER_BETA_START="${DAGGER_BETA_START:-0.8}"
export DAGGER_BETA_END="${DAGGER_BETA_END:-0.8}"
export PHI0_DISTILL_RTC="${PHI0_DISTILL_RTC:-1}"
export PHI0_DISTILL_RTC_MAX_DELAY="${PHI0_DISTILL_RTC_MAX_DELAY:-8}"
export PROMPT="${PROMPT:-机器人朝黑箱子走过去。}"
export TASK_PROMPT="${TASK_PROMPT:-${PROMPT}}"

PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/830demo_skill1_walk_handcmd_lag1_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/830demo_skill1_walk_handcmd_lag1_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"
export PHI0_DISTILL_OUT LOG_FILE NGPU NUM_ENVS EPOCHS HORIZON STAMP

_VLM_CACHE_DIRNAME="${PHI0_VLM_FRAME_LATENTS_DIRNAME:-vlm_frame_latents_qwen3vl_dual}"
for f in \
  "${REF_ROOT}/meta/stats.json" \
  "${REF_ROOT}/meta/vision_episode_allowlist.json" \
  "${REF_ROOT}/meta/${_VLM_CACHE_DIRNAME}/meta.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "[830_skill1_walk] missing ${f}" >&2
    echo "[830_skill1_walk] encode first: DATASET_ROOT=${REF_ROOT} PROMPT='${PROMPT}' bash tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh" >&2
    exit 1
  fi
done

echo "[830_skill1_walk] REF=${REF_ROOT}"
echo "[830_skill1_walk] VLM cache=on deploy=${DEPLOY_POLICY_DIR} hand=${PHI0_HAND_MODE} lag=${PHI0_HAND_PROPRIO_LAG} rtc=${PHI0_DISTILL_RTC} W_HAND=${W_HAND}"
echo "[830_skill1_walk] B=${NUM_ENVS}×NGPU=${NGPU} H=${HORIZON} epochs=${EPOCHS} CUDA=${CUDA_VISIBLE_DEVICES}"
echo "[830_skill1_walk] OUT=${PHI0_DISTILL_OUT}"

exec bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
