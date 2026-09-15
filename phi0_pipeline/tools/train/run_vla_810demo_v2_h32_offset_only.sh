#!/usr/bin/env bash
# 810demo_v2_egypt_layout VLA: no hist vision, H=32, offset_only AdaLN, 80k steps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NGPU="${NGPU:-8}"
export MASTER_PORT="${MASTER_PORT:-29642}"
export CONFIG=train_810demo_egypt_vlm_h32_offset_only
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-80000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export STAMP
export PHI0_TRAIN_MODE=vla
export PHI0_ADALN_MODE=offset_only

export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/810demo_v2_egypt_layout}"
export PICK_TISSUE_REPO_ID="${PICK_TISSUE_REPO_ID:-810demo_v2_egypt_layout}"
export ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
export PHI0_DISTILL_OUT="/mnt/data3/wpy/vla_810demo_v2_h32_offset_only_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}"
export LOG_FILE="/mnt/data2/wpy/workspace/logs/vla_810demo_v2_h32_offset_only_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}.log"

echo "================================================================"
echo "[vla_h32_offset_only] 810demo_v2 VLA: current-frame dual + H=32 + offset_only"
echo "================================================================"
echo "repo          = $PHI0_ROOT (Phi_0_clean)"
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
echo "vlm_video_delta_indices=(none, current frame only)"
echo "horizon       = past_w=1  H=32  seq_len=33"
echo "adaln         = offset_only  bins=6000  (no vlm_age phase)"
echo "proprio       = proprio41 = body29(qpos) + revo2_12  raw  deploy_align=false"
echo "action_loss_exclude=g1_body_qpos_36,projected_gravity_xyz"
echo "prompt        = episode instruction (no override)"
echo "rtc           = enabled (train postfix max_delay=8)"
echo "OUT=${PHI0_DISTILL_OUT}"
echo "LOG=${LOG_FILE}"
echo "================================================================"

exec bash "${ROOT}/run_vla_diskz_810demo_distill.sh"
