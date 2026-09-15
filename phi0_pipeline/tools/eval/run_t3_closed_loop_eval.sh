#!/usr/bin/env bash
# 正式 eval 可视化：T3 train-aligned 闭环 → 单进程 mp4（不起三终端）。
#
# 通用入口：任意 unified 数据集 + ckpt，用环境变量切换，不绑死某一 demo。
#
# 红线（2026-08-14 验证）：
#   CONFIG 必须带 hand_model=revo2 + unified_ds=g1_sonic_zmq_revo2（否则手被抹零）
#   USE_RTC=1  PROPRIO_SOURCE=robot  PHI0_VLM_REFRESH_EVERY_STEP=1
#   REVO2_HAND=1 → MuJoCo scene_43dof_revo2；proprio = sim g1_debug body29+revo2_12
#   PHI0_ADALN_MODE=offset_vlm_age  unset PROMPT（episode instruction）
#   QPOS_RSI 默认 1（与 GT sonic 验收一致）；不要用旧 experiments/*/launch.sh
#
# Usage:
#   CHECKPOINT=... CONFIG_NAME=... UNIFIED_ROOT=... GT_REPO_ID=... \
#     bash tools/eval/run_t3_closed_loop_eval.sh [episode_idx] [max_frames]
#   bash launch/eval_t3_closed_loop.sh 0 400
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WS="${WS:-$(cd "${ROOT}/.." && pwd)}"
PKG_T3="${ROOT}/real_deploy/t3_phi-0_deploy"

EP="${1:-${UNIFIED_EP:-0}}"
MAX_FRAMES="${2:-${MAX_FRAMES:-400}}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Prefer package gold ckpt; else require CHECKPOINT.
if [[ -z "${CHECKPOINT:-}" ]]; then
  for cand in \
    "${PKG_T3}/meta/phi0_step30000.pt" \
    "${ROOT}/experiments/810-final/phi0_step30000.pt"
  do
    if [[ -f "${cand}" ]]; then
      CHECKPOINT="${cand}"
      break
    fi
  done
fi
if [[ -z "${CHECKPOINT:-}" || ! -f "${CHECKPOINT}" ]]; then
  echo "error: set CHECKPOINT=/path/to/phi0_*.pt" >&2
  exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-train_810short_egypt_vlm_offset_sonic_revo2}"
UNIFIED_ROOT="${UNIFIED_ROOT:-${WS}/810short_horizon_v2_egypt_layout}"
GT_REPO_ID="${GT_REPO_ID:-810short_horizon_v2_egypt_layout}"
PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
case "${PROPRIO_SOURCE}" in
  robot) ;;
  robot_gt_hand|gt)
    echo "error: PROPRIO_SOURCE=${PROPRIO_SOURCE} replaces/ignores sim Revo2 hand for VLA input" >&2
    echo "  default is PROPRIO_SOURCE=robot (body29+revo2_12 from g1_debug :5557)" >&2
    exit 1
    ;;
  *)
    echo "error: unsupported PROPRIO_SOURCE=${PROPRIO_SOURCE} (want robot)" >&2
    exit 1
    ;;
esac
USE_RTC="${USE_RTC:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
WORK_DIR="${WORK_DIR:-${ROOT}/experiments/vla_t3_cl_ep${EP}_${STAMP}}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/closed_loop_gt.mp4}"

if [[ -n "${PHI0_ADALN_MODE:-}" ]]; then
  case "${PHI0_ADALN_MODE}" in
    progress_vlm_age|offset_vlm_age) ;;
    *)
      echo "error: PHI0_ADALN_MODE=${PHI0_ADALN_MODE} incompatible; want progress_vlm_age" >&2
      exit 1
      ;;
  esac
fi
export PHI0_ADALN_MODE=progress_vlm_age
export PHI0_ACTION_CROSS_ATTN_MODE="${PHI0_ACTION_CROSS_ATTN_MODE:-interleave_vlm}"
export PHI0_VLM_REFRESH_EVERY_STEP="${PHI0_VLM_REFRESH_EVERY_STEP:-1}"
export PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX="${PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX:-1}"
export REVO2_HAND="${REVO2_HAND:-1}"
# 与 GT sonic 验收对齐：默认 snap 到 dataset frame0；显式 QPOS_RSI=0 可关。
export QPOS_RSI="${QPOS_RSI:-1}"

unset PROMPT
export PROMPT=""

export CHECKPOINT CONFIG_NAME UNIFIED_ROOT GT_REPO_ID PROPRIO_SOURCE USE_RTC
export CUDA_DEVICES WORK_DIR OUT_MP4 QPOS_RSI REVO2_HAND
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-${WS}}"
export PHI0_ROOT="${PHI0_ROOT:-${ROOT}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "==> T3 train-aligned closed-loop eval (one-shot mp4)"
echo "    CHECKPOINT=${CHECKPOINT}"
echo "    CONFIG_NAME=${CONFIG_NAME}"
echo "    PHI0_ADALN_MODE=${PHI0_ADALN_MODE}"
echo "    PHI0_VLM_REFRESH_EVERY_STEP=${PHI0_VLM_REFRESH_EVERY_STEP}"
echo "    PROPRIO_SOURCE=${PROPRIO_SOURCE} USE_RTC=${USE_RTC} REVO2_HAND=${REVO2_HAND} QPOS_RSI=${QPOS_RSI}"
echo "    proprio: sim g1_debug body29+revo2_12 only (no dataset GT proprio)"
echo "    UNIFIED_ROOT=${UNIFIED_ROOT} ep=${EP} max_frames=${MAX_FRAMES}"
echo "    WORK_DIR=${WORK_DIR}"
echo "    OUT_MP4=${OUT_MP4}"

exec bash "${ROOT}/tools/eval/run_810demo_closed_loop_eval.sh" "${EP}" "${MAX_FRAMES}"
