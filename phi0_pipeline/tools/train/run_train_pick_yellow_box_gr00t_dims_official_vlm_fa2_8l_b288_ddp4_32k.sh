#!/usr/bin/env bash
# 4-GPU DDP: pick-yellow-box GR00T dims + official Qwen3-VL + FA2, 8L DiT, batch=72/GPU, 32k.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

EXP="${EXP:-experiments/pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_32k_ddp4}"
CONFIG="${CONFIG:-train_pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_ddp4_32k}"
CKPT_NAME="${CKPT_NAME:-pick_yellow_box_gr00t_dims_official_vlm_fa2_8l_b288_act}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
AUTO_RESUME="${AUTO_RESUME:-false}"
USE_TEE="${USE_TEE:-0}"
MAX_STEPS="${MAX_STEPS:-32000}"
BATCH_SIZE="${BATCH_SIZE:-72}"

export EXP CONFIG CKPT_NAME CUDA_DEVICES AUTO_RESUME USE_TEE MAX_STEPS BATCH_SIZE
exec bash tools/train/run_train_pick_yellow_box_xperience_unified_ddp4_16k.sh
