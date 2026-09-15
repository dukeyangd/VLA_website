#!/usr/bin/env bash
# 830 skill_walk_to_black_box_new_unified — vision_dl + dual VLM frame cache.
#
#   bash tools/train/run_830_walk_blackbox_vlm_cache_distill.sh
#   EPOCHS=20 NGPU=8 bash tools/train/run_830_walk_blackbox_vlm_cache_distill.sh
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill_walk_to_black_box_new_unified}"
NGPU="${NGPU:-8}"
NUM_ENVS="${NUM_ENVS:-32}"
EPOCHS="${EPOCHS:-10}"
HORIZON="${HORIZON:-32}"

export REF_ROOT
export PHI0_USE_VLM_FRAME_LATENT_CACHE=1
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1
export USE_VLM=0
export PHI0_P_VISION=1
export PHI0_VISION_DATALOADER=1
# Pure vision corpus: no text-only Isaac eps.
export PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY=0
export PHI0_MODALITY_SKIP_SIM_VIDEO=1
export PHI0_DAGGER_ON_NO_VIDEO=0
export VISION_ONLY=1
# Pure vision_dl: skip 8× Isaac Sim boot (DataLoader + frame cache only).
export PHI0_VISION_DL_ONLY=1
export PHI0_RESIDENT_CACHE_FORCE=1
# Raw action.motion_token in unified[396:460) — v1.1 teleop domain; decode with sonic_v1_1.
export TEACHER_Z_SOURCE=disk
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export PHI0_HAND_MODE=dex3
export PHI0_NEWTON_REVO2=0
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export W_Q_HEAD="${W_Q_HEAD:-0}"
export PHI0_DISTILL_NO_BARRIER="${PHI0_DISTILL_NO_BARRIER:-1}"
# Dex3 hand BC: unified[346:360] full 14-d student in/out (walk ≈ constant open pose).
export W_HAND="${W_HAND:-1}"
export PHI0_ALLOW_ZERO_HAND=0
export STUDENT_DRIVE="${STUDENT_DRIVE:-1}"
export DAGGER_BETA_START="${DAGGER_BETA_START:-0.8}"
export DAGGER_BETA_END="${DAGGER_BETA_END:-0.8}"

PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/830_walk_blackbox_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/830_walk_blackbox_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"
export PHI0_DISTILL_OUT LOG_FILE NGPU NUM_ENVS EPOCHS HORIZON STAMP

for f in \
  "${REF_ROOT}/meta/stats.json" \
  "${REF_ROOT}/meta/vision_episode_allowlist.json" \
  "${REF_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/meta.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "[830_walk] missing ${f}" >&2
    exit 1
  fi
done

echo "[830_walk] REF=${REF_ROOT}"
echo "[830_walk] VLM cache=on deploy=${DEPLOY_POLICY_DIR} hand=${PHI0_HAND_MODE} W_HAND=${W_HAND} no_barrier=${PHI0_DISTILL_NO_BARRIER} OUT=${PHI0_DISTILL_OUT}"
echo "[830_walk] B=${NUM_ENVS}×NGPU=${NGPU} H=${HORIZON} epochs=${EPOCHS}"

exec bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
