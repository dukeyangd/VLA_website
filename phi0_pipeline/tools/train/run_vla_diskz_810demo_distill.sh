#!/usr/bin/env bash
# mode=vla: 810demo gold-aligned offline VLA (dual RGB + episode instruction).
# Gold-compare knobs (match phi-0-810 810demo_offline_vlm_b16_ddp4_noqpos_30k):
#   Steps  : max_steps=30000, save_every_steps=10000
#   LR     : 1e-4 · mixed_precision=bf16
#   Batch  : per-GPU 16 × 4 DDP → effective 64
#   Stats  : 810demo_egypt_layout/meta/stats.json (z-score norm-space MSE)
#   Logs   : loss_action + supervised slices only (skip excluded); ETA wall-ms
#   Vision : ego + chest_forward · prompt=episode instruction · past_w=1
#   ACT    : num_root_goals=0 (no BoneSEED goal slots)
# NOT ChunkStudent / NOT Isaac. Deploy proof = 810 LOCKED closed loop.
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PHI0_ROOT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NGPU="${NGPU:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
CONFIG="${CONFIG:-train_810demo_egypt_vlm_b512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29594}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/data3/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/data3/hf_home}"
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}"
export PHI0_TRAIN_MODE="${PHI0_TRAIN_MODE:-vla}"

# Gold VLM train env (not Isaac newton conda).
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
FASTWAM_SRC="${FASTWAM_SRC:-${PHI0_ROOT}/subpackages}"
VGGT_SRC="${VGGT_SRC:-/mnt/data2/wpy/workspace/vggt-omega}"
export PYTHONPATH="${PHI0_ROOT}/src:${FASTWAM_SRC}:${VGGT_SRC}:${PYTHONPATH:-}"

REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/810demo_egypt_layout}"
# Default repo id = basename(REF_ROOT) so v2 layout works without editing yaml.
PICK_TISSUE_REPO_ID="${PICK_TISSUE_REPO_ID:-$(basename "${REF_ROOT}")}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/vla_810demo_dual_vlm_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}}"
export PHI0_DISTILL_OUT
OUT_LOG="${PHI0_DISTILL_OUT}/train.log"
LOG_FILE="${LOG_FILE:-/mnt/data2/wpy/workspace/logs/vla_810demo_dual_vlm_${NGPU}gpu_${STAMP}.log}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[vla_810] missing PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT}" ]]; then
  echo "[vla_810] missing REF_ROOT=${REF_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${ACTION_STATS_PATH}" ]]; then
  echo "[vla_810] missing ACTION_STATS_PATH=${ACTION_STATS_PATH}" >&2
  exit 1
fi

mkdir -p "${PHI0_DISTILL_OUT}" "$(dirname "${LOG_FILE}")"

echo "[vla_810] mode=vla gold dual-VLM offline (no Isaac, no ChunkStudent)"
echo "[vla_810] config=${CONFIG} train_vlm_view=dual prompt=cfg instruction_override|episode task num_root_goals=0"
echo "[vla_810] ref=${REF_ROOT} repo_id=${PICK_TISSUE_REPO_ID} stats=${ACTION_STATS_PATH} (z-score norm MSE)"
echo "[vla_810] out=${PHI0_DISTILL_OUT} ngpu=${NGPU} B=${BATCH_SIZE} eff=$((BATCH_SIZE * NGPU)) max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} lr=1e-4 bf16"
EXTRA_OVERRIDES=()
if [[ "${ZERO_PROPRIO_LEFT_THUMB_AUX:-0}" == "1" || "${ZERO_PROPRIO_LEFT_THUMB_AUX:-}" == "true" ]]; then
  EXTRA_OVERRIDES+=(data.zero_proprio_left_thumb_aux=true)
fi

echo "[vla_810] python=${PYTHON_BIN}"
echo "[vla_810] log=${LOG_FILE}  out_log=${OUT_LOG}"
echo "[vla_810] extra_overrides=${EXTRA_OVERRIDES[*]:-}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PHI0_ROOT}/tools/train/train.py" \
  --config-name "${CONFIG}" \
  output_dir="${PHI0_DISTILL_OUT}" \
  batch_size="${BATCH_SIZE}" \
  max_steps="${MAX_STEPS}" \
  save_every_steps="${SAVE_EVERY}" \
  learning_rate=1.0e-4 \
  mixed_precision=bf16 \
  distributed=true \
  save_action_expert_only=true \
  data.pick_tissue_root=/mnt/data2/wpy/workspace \
  data.pick_tissue_repo_id="${PICK_TISSUE_REPO_ID}" \
  data.action_stats_path="${ACTION_STATS_PATH}" \
  data.train_vlm_view=dual \
  "${EXTRA_OVERRIDES[@]}" \
  2>&1 | tee -a "${OUT_LOG}" "${LOG_FILE}"
