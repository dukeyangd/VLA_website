#!/usr/bin/env bash
# 830demo skill_2_pico_pick_the_toy — vision_dl + VLM frame cache (same recipe as skill3).
# Hand train proprio: motor cmd tape (hand_ref), not measured hand_obs — aligns
# deploy PHI0_CL_HAND_OBS=commanded / last hand_pred.
# Ablation: PHI0_ZERO_PROPRIO_HAND=1 zeros hand slots in student obs; W_HAND / hand
# GT from action.unified unchanged (still predict dex3).
#
#   bash tools/train/run_830_skill2_pico_pick_toy_vlm_cache_distill.sh
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NGPU=6 bash tools/train/run_830_skill2_pico_pick_toy_vlm_cache_distill.sh
#   PHI0_ZERO_PROPRIO_HAND=1 bash tools/train/run_830_skill2_pico_pick_toy_vlm_cache_distill.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill2_pico_pick_toy_unified}"
NGPU="${NGPU:-6}"
NUM_ENVS="${NUM_ENVS:-32}"
EPOCHS="${EPOCHS:-10}"
HORIZON="${HORIZON:-32}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

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
# Train hand proprio = teleop motor cmd (unified[346:360)); not measured obs[475:489).
# Lag 1 = tape cmd @ t−1 in obs (matches deploy last hand_pred); GT still [t,t+H).
export PHI0_TRAIN_HAND_OBS="${PHI0_TRAIN_HAND_OBS:-commanded}"
export PHI0_HAND_PROPRIO_LAG="${PHI0_HAND_PROPRIO_LAG:-1}"
# Ablation: zero hand proprio input; keep dex3 hand head + loss (W_HAND).
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
export PROMPT="${PROMPT:-抓起黄色玩具放到篮子里。}"
export TASK_PROMPT="${TASK_PROMPT:-${PROMPT}}"

_HAND_TAG="handcmd"
if [[ "${PHI0_ZERO_PROPRIO_HAND}" == "1" || "${PHI0_ZERO_PROPRIO_HAND}" == "true" || "${PHI0_ZERO_PROPRIO_HAND}" == "yes" || "${PHI0_ZERO_PROPRIO_HAND}" == "on" ]]; then
  _HAND_TAG="nohandobs"
elif [[ "${PHI0_HAND_PROPRIO_LAG}" != "0" ]]; then
  _HAND_TAG="handcmd_lag${PHI0_HAND_PROPRIO_LAG}"
fi
PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/830demo_skill2_pico_pick_toy_${_HAND_TAG}_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/830demo_skill2_pico_pick_toy_${_HAND_TAG}_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"
export PHI0_DISTILL_OUT LOG_FILE NGPU NUM_ENVS EPOCHS HORIZON STAMP

_VLM_CACHE_DIRNAME="${PHI0_VLM_FRAME_LATENTS_DIRNAME:-vlm_frame_latents_qwen3vl_dual}"
for f in \
  "${REF_ROOT}/meta/stats.json" \
  "${REF_ROOT}/meta/vision_episode_allowlist.json" \
  "${REF_ROOT}/meta/${_VLM_CACHE_DIRNAME}/meta.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "[830_pico_pick] missing ${f}" >&2
    exit 1
  fi
done

echo "[830_pico_pick] REF=${REF_ROOT}"
echo "[830_pico_pick] VLM cache=on deploy=${DEPLOY_POLICY_DIR} hand=${PHI0_HAND_MODE} train_hand_obs=${PHI0_TRAIN_HAND_OBS} hand_proprio_lag=${PHI0_HAND_PROPRIO_LAG} zero_proprio_hand=${PHI0_ZERO_PROPRIO_HAND} W_HAND=${W_HAND} ckpt_every=${CKPT_EVERY}"
echo "[830_pico_pick] B=${NUM_ENVS}×NGPU=${NGPU} H=${HORIZON} epochs=${EPOCHS} CUDA=${CUDA_VISIBLE_DEVICES}"
echo "[830_pico_pick] OUT=${PHI0_DISTILL_OUT}"

exec bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
