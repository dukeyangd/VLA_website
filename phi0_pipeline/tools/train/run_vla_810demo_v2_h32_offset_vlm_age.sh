#!/usr/bin/env bash
# 810demo/810short VLA: no hist vision, H=32, offset_vlm_age AdaLN.
# Defaults match prior b8×ddp8 50k recipe; override REF_ROOT for 810short.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NGPU="${NGPU:-8}"
export MASTER_PORT="${MASTER_PORT:-29643}"
export CONFIG=train_810demo_egypt_vlm_h32_offset_vlm_age
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-50000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export STAMP
export PHI0_TRAIN_MODE=vla
export PHI0_ADALN_MODE=offset_vlm_age
# Align deploy/train: left Revo2 thumb_aux (proprio41 index 30) = 0.
export ZERO_PROPRIO_LEFT_THUMB_AUX="${ZERO_PROPRIO_LEFT_THUMB_AUX:-1}"

export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/810demo_v2_egypt_layout}"
export PICK_TISSUE_REPO_ID="${PICK_TISSUE_REPO_ID:-$(basename "${REF_ROOT}")}"
export ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
# Name out by dataset basename (810short vs 810demo_v2).
_ds="$(basename "${REF_ROOT}")"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/vla_${_ds}_h32_offset_vlm_age_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-/mnt/data2/wpy/workspace/logs/vla_${_ds}_h32_offset_vlm_age_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}.log}"

echo "================================================================"
echo "[vla_h32_offset_vlm_age] VLA: current-frame dual + H=32 + offset_vlm_age"
echo "================================================================"
echo "repo          = phi-0-wbc-newton"
echo "mode          = vla"
echo "CONFIG=${CONFIG}"
echo "PHI0_ADALN_MODE=${PHI0_ADALN_MODE}"
echo "REF_ROOT=${REF_ROOT}"
echo "PICK_TISSUE_REPO_ID=${PICK_TISSUE_REPO_ID}"
echo "ACTION_STATS_PATH=${ACTION_STATS_PATH}  (z-score action)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NGPU=${NGPU}  BATCH_SIZE=${BATCH_SIZE}  effective_batch=$((BATCH_SIZE * NGPU))"
echo "MAX_STEPS=${MAX_STEPS}  SAVE_EVERY=${SAVE_EVERY}"
echo "learning_rate=1e-4  mixed_precision=bf16  distributed=true"
echo "vlm           = freeze Qwen3-VL dual current-frame (ego+chest)  NO hist"
echo "horizon       = past_w=1  H=32  seq_len=33"
echo "adaln         = offset_vlm_age  bins=6000  vlm_age_period=10"
echo "thumb_aux0    = ${ZERO_PROPRIO_LEFT_THUMB_AUX}"
echo "OUT=${PHI0_DISTILL_OUT}"
echo "LOG=${LOG_FILE}"
echo "================================================================"

exec bash "${ROOT}/run_vla_diskz_810demo_distill.sh"
