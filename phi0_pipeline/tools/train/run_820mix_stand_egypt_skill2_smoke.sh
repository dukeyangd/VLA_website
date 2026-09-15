#!/usr/bin/env bash
# Smoke: mix ep0 stand + ep1 egypt (text Isaac) + mix ep226 = skill2 origin ep0 (vision_dl).
# 8 GPU × 8 envs. AdaLN: vision hard-off, text cap=64.
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PHI0_ROOT
export PHI0_SUBPACKAGES="${PHI0_ROOT}/subpackages"
export GR00T_ROOT="${PHI0_SUBPACKAGES}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_mix_release_unified}"
OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/820mix_stand_egypt_s2_smoke_${STAMP}}"
mkdir -p "${OUT}"
ALLOW="${EPISODE_ALLOWLIST:-${OUT}/allowlist_stand_egypt_s2.json}"
if [[ ! -f "${ALLOW}" ]]; then
  echo '{"episode_index":[0,1,226]}' >"${ALLOW}"
fi

export PHI0_DISTILL_OUT="${OUT}"
export REF_ROOT
export EPISODE_ALLOWLIST="${ALLOW}"
export NGPU="${NGPU:-8}"
export NUM_ENVS="${NUM_ENVS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29611}"
export HORIZON="${HORIZON:-32}"
export EPOCHS="${EPOCHS:-1}"
# Cover of 3 short eps is tiny; run a fixed step budget.
export EXTRA_STEPS="${EXTRA_STEPS:-4000}"
export CKPT_EVERY="${CKPT_EVERY:-1000}"
export PHI0_P_VISION="${PHI0_P_VISION:-0.5}"
export W_Z="${W_Z:-1}"
export W_HAND="${W_HAND:-1}"
export W_Q_HEAD="${W_Q_HEAD:-0}"
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE:-disk}"
export PHI0_ADALN_MODE="${PHI0_ADALN_MODE:-progress_only}"
export PHI0_FOURIER_MAX_CYCLES="${PHI0_FOURIER_MAX_CYCLES:-64}"
export PHI0_ADALN_ZERO_VISION="${PHI0_ADALN_ZERO_VISION:-1}"
export PHI0_ACTION_CROSS_ATTN_MODE="${PHI0_ACTION_CROSS_ATTN_MODE:-interleave_vlm}"
export PHI0_STAND_EPISODE_INDEX="${PHI0_STAND_EPISODE_INDEX:-0}"
export PHI0_STAND_RSI_PROB="${PHI0_STAND_RSI_PROB:-0.5}"
export PHI0_RSI_EP0_PROB="${PHI0_RSI_EP0_PROB:-0.5}"
export RSI_START="${RSI_START:-random}"
export RECORD_TRAIN_MP4="${RECORD_TRAIN_MP4:-0}"
export LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/820mix_stand_egypt_s2_smoke_${STAMP}.log}"

echo "[stand-egypt-s2] out=${OUT}"
echo "[stand-egypt-s2] allow=${ALLOW} H=${HORIZON} extra=${EXTRA_STEPS} p_vision=${PHI0_P_VISION} xattn=${PHI0_ACTION_CROSS_ATTN_MODE}"
exec bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
