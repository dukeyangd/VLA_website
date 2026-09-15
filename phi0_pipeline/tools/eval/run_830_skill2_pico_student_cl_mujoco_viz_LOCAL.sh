#!/usr/bin/env bash
# Local single-GPU repro for skill2 pico ep0 CL (sonic_v1_1 + Dex3 + frame_cache).
# Prerequisite: assets under datasets/830/skill_2_pico_new826_unified + ckpt under experiments/...
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PHI0_ROOT}"

# kill stale TRT deploy / sim that holds :5556 and GPU
pkill -9 -f 'g1_deploy_onnx_ref' 2>/dev/null || true
pkill -9 -f 'run_sim_loop_vla_record' 2>/dev/null || true
pkill -9 -f 'phi0_chunk_student_sonic_closed_loop' 2>/dev/null || true
sleep 1

export PHI0_PY="${PHI0_PY:-/home/user/miniconda3/envs/Phi-0-wbc-newton-wpy/bin/python}"
export STUDENT_CKPT="${STUDENT_CKPT:-${PHI0_ROOT}/experiments/830_skill2_pico_nosim_h32_b32_ddp8_e10_20260827_081246/phi0_student_last.pt}"
export REF_ROOT="${REF_ROOT:-/home/user/workspace/datasets/830/skill_2_pico_new826_unified}"
export VALID_HAND_ROOT="${VALID_HAND_ROOT:-/home/user/workspace/datasets/830/skill_2_pico_new826/2026-08-26-15-40-11}"
export EP="${EP:-0}"
export HORIZON="${HORIZON:-32}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export PHI0_DEX3_HAND_POLICY_ORDER="${PHI0_DEX3_HAND_POLICY_ORDER:-0}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE="${PHI0_USE_VLM_FRAME_LATENT_CACHE:-1}"
export PHI0_CL_VLM_SOURCE="${PHI0_CL_VLM_SOURCE:-frame_cache}"
# single 4090: do NOT let walk script default deploy→GPU7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLA_CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES:-0}"
export DEPLOY_CUDA_VISIBLE_DEVICES="${DEPLOY_CUDA_VISIBLE_DEVICES:-0}"
export TAG="${TAG:-830_skill2_pico_student_cl_ep${EP}_local_$(date +%Y%m%d_%H%M%S)}"

test -f "${STUDENT_CKPT}" || { echo "missing ckpt ${STUDENT_CKPT}" >&2; exit 1; }
test -f "${REF_ROOT}/data/chunk-000/file-$(printf '%03d' "${EP}").parquet" || { echo "missing parquet" >&2; exit 1; }
test -f "${REF_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/ep/$(printf '%06d' "${EP}")/latents.fp16.dat" || { echo "missing vlm cache ep${EP}" >&2; exit 1; }

echo "[local_skill2_cl] ckpt=${STUDENT_CKPT}"
echo "[local_skill2_cl] ref=${REF_ROOT} ep=${EP} deploy=${DEPLOY_POLICY_DIR} hand_order=${PHI0_DEX3_HAND_POLICY_ORDER} vlm=${PHI0_CL_VLM_SOURCE}"
echo "[local_skill2_cl] gpu VLA=${VLA_CUDA_VISIBLE_DEVICES} deploy=${DEPLOY_CUDA_VISIBLE_DEVICES}"
exec bash "${PHI0_ROOT}/tools/eval/run_830_skill2_pico_student_cl_mujoco_viz.sh"
