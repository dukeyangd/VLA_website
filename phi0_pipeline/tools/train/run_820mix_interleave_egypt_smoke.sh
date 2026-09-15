#!/usr/bin/env bash
# Egypt smoke overfit on 820 mix interleave: ep0 stand + ep1 埃及肚皮舞.
# ep1 has_video=false → vision coin falls back to proprio_online (onnx_g1).
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PHI0_ROOT
# Parent shells sometimes export a broken PHI0_SUBPACKAGES under tools/ — force repo root.
export PHI0_SUBPACKAGES="${PHI0_ROOT}/subpackages"
export GR00T_ROOT="${PHI0_SUBPACKAGES}"
export SIMULATOR="${SIMULATOR:-newton}"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/eval/_assert_no_efs_gt_latent.sh"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_mix_release_unified}"
OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/820mix_interleave_egypt_smoke_${STAMP}}"
mkdir -p "${OUT}"
ALLOW="${EPISODE_ALLOWLIST:-${OUT}/allowlist_egypt_stand.json}"
if [[ ! -f "${ALLOW}" ]]; then
  echo '{"episode_index":[0,1]}' >"${ALLOW}"
fi

export PHI0_DISTILL_OUT="${OUT}"
export REF_ROOT
export EPISODE_ALLOWLIST="${ALLOW}"
export PHI0_STAND_EPISODE_INDEX="${PHI0_STAND_EPISODE_INDEX:-0}"
export PHI0_INTERLEAVE_P_PROPRIO="${PHI0_INTERLEAVE_P_PROPRIO:-0.1}"
export PHI0_STAND_RSI_PROB="${PHI0_STAND_RSI_PROB:-0.5}"
export PHI0_RSI_EP0_PROB="${PHI0_RSI_EP0_PROB:-1.0}"
export RSI_START="${RSI_START:-random}"
export NGPU="${NGPU:-1}"
export NUM_ENVS="${NUM_ENVS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EPOCHS="${EPOCHS:-2}"
export HORIZON="${HORIZON:-1}"
export W_Z="${W_Z:-1}"
export W_HAND="${W_HAND:-0}"
export W_Q_HEAD="${W_Q_HEAD:-0}"
export USE_VLM="${USE_VLM:-1}"
# Force online onnx_g1 for egypt smoke (parent env may have TEACHER_Z_SOURCE=disk).
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE_FORCE:-online}"
export RESIDENT_REF=1
export RECORD_TRAIN_MP4="${RECORD_TRAIN_MP4:-1}"
export RECORD_EVERY="${RECORD_EVERY:-2}"
export LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/820mix_interleave_egypt_smoke_${STAMP}.log}"

echo "[interleave-egypt] out=${OUT}"
echo "[interleave-egypt] allow=${ALLOW} stand_ep=${PHI0_STAND_EPISODE_INDEX} p_proprio=${PHI0_INTERLEAVE_P_PROPRIO}"
exec bash "${PHI0_ROOT}/tools/train/run_820mix_interleave_distill.sh"
