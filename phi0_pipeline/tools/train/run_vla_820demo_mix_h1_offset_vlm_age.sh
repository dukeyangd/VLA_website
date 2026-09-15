#!/usr/bin/env bash
# LEGACY — 820demo_mix offline Hydra VLA（磁盘 sonic 当 GT，无仿真 hold）。
# 混训主路径：bash tools/train/run_online_vlm_mix_distill.sh
# 820demo_mix_release_unified — full mix VLA: H=1, current-frame dual,
# align_vlm_refresh (hold until age=0), no hist vision, B=8 × 8 GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

echo "[LEGACY] offline VLA 820 mix — prefer run_online_vlm_mix_distill.sh" >&2

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NGPU="${NGPU:-8}"
export MASTER_PORT="${MASTER_PORT:-29721}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/data3/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/data3/hf_home}"
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}"
export PHI0_TRAIN_MODE=vla
export PHI0_ADALN_MODE=offset_vlm_age

export CONFIG="${CONFIG:-train_810demo_egypt_vlm_h1_offset_vlm_age}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
# ~539k frames / eff64 ≈ 8.4k steps/epoch; 50k ≈ ~6 epochs
export MAX_STEPS="${MAX_STEPS:-50000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export STAMP

export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_mix_release_unified}"
export PICK_TISSUE_REPO_ID="${PICK_TISSUE_REPO_ID:-820demo_mix_release_unified}"
export ACTION_STATS_PATH="${ACTION_STATS_PATH:-${REF_ROOT}/meta/stats.json}"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/vla_820mix_h1_age_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-/mnt/data2/wpy/workspace/Phi_0_wpy/logs/vla_820mix_h1_age_b${BATCH_SIZE}_ddp${NGPU}_s${MAX_STEPS}_${STAMP}.log}"

source "${ROOT}/tools/env/setup_env.sh"
PYTHON_BIN="${PHI0_PY}"
mkdir -p "${PHI0_DISTILL_OUT}" "$(dirname "${LOG_FILE}")"

cat > "${PHI0_DISTILL_OUT}/RESOLVED_TRAIN_SETTINGS.yaml" <<EOF
repo: Phi_0_wpy
train_mode: vla
config: ${CONFIG}
ref_root: ${REF_ROOT}
pick_tissue_root: /mnt/data2/wpy/workspace/820demo
repo_id: ${PICK_TISSUE_REPO_ID}
action_stats_path: ${ACTION_STATS_PATH}
batch_size_per_gpu: ${BATCH_SIZE}
ngpu: ${NGPU}
effective_batch: $((BATCH_SIZE * NGPU))
max_steps: ${MAX_STEPS}
save_every_steps: ${SAVE_EVERY}
learning_rate: 1.0e-4
mixed_precision: bf16
distributed: true
adaln_mode: offset_vlm_age
vlm_age_period: 10
align_vlm_refresh: true
train_obs_only_video: true
action_future_horizon: 1
seq_len: 2
past_w: 1
train_vlm_view: dual
instruction: episode (instruction_override=null)
hand_model: revo2
action_loss_exclude:
  - g1_body_qpos_36
  - projected_gravity_xyz
EOF

echo "================================================================"
echo "[820mix_h1_age] dual current-frame + hold@age0 + H=1 + offset_vlm_age"
echo "REF=${REF_ROOT}"
echo "OUT=${PHI0_DISTILL_OUT}"
echo "LOG=${LOG_FILE}"
echo "B=${BATCH_SIZE} × NGPU=${NGPU} eff=$((BATCH_SIZE * NGPU)) steps=${MAX_STEPS}"
echo "================================================================"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
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
  +data.hand_model=revo2 \
  data.unified_supervision_dataset=g1_sonic_zmq_revo2 \
  data.instruction_override=null \
  data.align_vlm_refresh=true \
  data.vlm_refresh_period=10 \
  data.train_obs_only_video=true \
  data.seq_len=2 \
  model.action_future_horizon=1 \
  model.action_dit_config.action_future_horizon=1 \
  model.action_dit_config.num_action_queries=1 \
  model.action_dit_config.adaln_mode=offset_vlm_age \
  model.action_dit_config.vlm_age_period=10 \
  2>&1 | tee "${LOG_FILE}" | tee "${PHI0_DISTILL_OUT}/train.log"
