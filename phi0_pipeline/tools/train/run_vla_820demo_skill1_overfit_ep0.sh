#!/usr/bin/env bash
# 820demo skill_1 unified — overfit ep0 with default VLA knobs: B=8 × 8 GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NGPU="${NGPU:-8}"
export MASTER_PORT="${MASTER_PORT:-29711}"
export CONFIG="${CONFIG:-train_820demo_skill1_unified_overfit}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export STAMP
export PHI0_TRAIN_MODE=vla

export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_skill_1_unified_overfit1}"
export PICK_TISSUE_REPO_ID="${PICK_TISSUE_REPO_ID:-820demo_skill_1_unified_overfit1}"
export ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/vla_820demo_skill1_overfit_ep0_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820demo_skill1_unified_overfit_ep0_b${BATCH_SIZE}_ddp${NGPU}.log}"

# diskz launcher hardcodes pick_tissue_root=workspace; our repo lives under 820demo/.
# Override via EXTRA_OVERRIDES after sourcing pattern — call torchrun directly.
source "${ROOT}/tools/env/setup_env.sh"
PYTHON_BIN="${PHI0_PY}"
mkdir -p "${PHI0_DISTILL_OUT}" "$(dirname "${LOG_FILE}")"

echo "================================================================"
echo "[820demo_skill1_overfit_ep0] B=${BATCH_SIZE} × NGPU=${NGPU} (eff=$((BATCH_SIZE * NGPU)))"
echo "REF_ROOT=${REF_ROOT}"
echo "OUT=${PHI0_DISTILL_OUT}"
echo "LOG=${LOG_FILE}"
echo "================================================================"

# Persist knobs for audit
cat > "${PHI0_DISTILL_OUT}/RESOLVED_TRAIN_SETTINGS.txt" <<EOF
batch_size_per_gpu=${BATCH_SIZE}
ngpu=${NGPU}
effective_batch=$((BATCH_SIZE * NGPU))
max_steps=${MAX_STEPS}
save_every_steps=${SAVE_EVERY}
learning_rate=1e-4
mixed_precision=bf16
distributed=true
config=${CONFIG}
data_root=/mnt/data2/wpy/workspace/820demo
repo_id=${PICK_TISSUE_REPO_ID}
stats=${ACTION_STATS_PATH}
note=VLA default is per-GPU 8 with 8-GPU DDP; not batch_size=512 from b512 yaml name
EOF

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT}/tools/train/train.py" \
  --config-name "${CONFIG}" \
  output_dir="${PHI0_DISTILL_OUT}" \
  batch_size="${BATCH_SIZE}" \
  max_steps="${MAX_STEPS}" \
  save_every_steps="${SAVE_EVERY}" \
  learning_rate=1.0e-4 \
  mixed_precision=bf16 \
  distributed=true \
  save_action_expert_only=true \
  data.pick_tissue_root=/mnt/data2/wpy/workspace/820demo \
  data.pick_tissue_repo_id="${PICK_TISSUE_REPO_ID}" \
  data.action_stats_path="${ACTION_STATS_PATH}" \
  data.train_vlm_view=dual \
  2>&1 | tee "${LOG_FILE}" | tee "${PHI0_DISTILL_OUT}/train.log"
