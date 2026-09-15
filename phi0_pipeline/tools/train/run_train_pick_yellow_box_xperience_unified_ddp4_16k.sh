#!/usr/bin/env bash
# 4-GPU DDP: pick-yellow-box xperience unified (512-d), 16k steps, batch=18/GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

NGPUS="${NGPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
EXP="${EXP:-experiments/pick_yellow_box_xperience_unified_16k_ddp4}"
CONFIG="${CONFIG:-train_pick_yellow_box_xperience_unified_ddp4_16k}"
CKPT_NAME="${CKPT_NAME:-pick_yellow_box_xperience_unified_act}"
BATCH_SIZE="${BATCH_SIZE:-18}"
MAX_STEPS="${MAX_STEPS:-16000}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
LEARNING_RATE_ACTION="${LEARNING_RATE_ACTION:-2.0e-4}"
AUTO_RESUME="${AUTO_RESUME:-false}"

export PHI0_WORKSPACE="${PHI0_WORKSPACE:-$(cd "${ROOT}/.." && pwd)}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/subpackages:${ROOT}/../vggt-omega:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}"
LOG_FILE="${LOG_FILE:-${EXP}/train.log}"

PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"

echo "==> Pick-yellow-box xperience unified DDP: ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS))"
echo "    max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} lr_action=${LEARNING_RATE_ACTION}"
echo "    log=${LOG_FILE}"
USE_NOHUP="${USE_NOHUP:-1}"
USE_TEE="${USE_TEE:-1}"
PID_FILE="${PID_FILE:-${EXP}/train.pid}"

echo "    auto_resume=${AUTO_RESUME} use_nohup=${USE_NOHUP} use_tee=${USE_TEE}"

_TRAIN_CMD=(
  "${PHI0_PY}" -m torch.distributed.run
  --standalone
  --nnodes=1
  --nproc_per_node="${NGPUS}"
  tools/train/train.py
  --config-name "${CONFIG}"
  output_dir="${EXP}"
  checkpoint_name="${CKPT_NAME}"
  batch_size="${BATCH_SIZE}"
  max_steps="${MAX_STEPS}"
  save_every_steps="${SAVE_EVERY}"
  learning_rate_action="${LEARNING_RATE_ACTION}"
  distributed=true
  auto_resume="${AUTO_RESUME}"
  save_action_expert_only=true
)

if [[ "${USE_NOHUP}" == "1" ]]; then
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "ERROR: training already running (pid=$(cat "${PID_FILE}"))" >&2
    echo "       stop: kill \$(cat ${PID_FILE})" >&2
    exit 1
  fi
  nohup "${_TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "==> Detached via nohup, pid=$(cat "${PID_FILE}")"
  echo "    tail -f ${LOG_FILE}"
  exit 0
fi

if [[ "${USE_TEE}" == "1" ]]; then
  exec "${_TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  exec "${_TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1
fi
