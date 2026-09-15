#!/usr/bin/env bash
# 8-GPU per-frame dual ego+chest + prompt VLM latent cache.
#
#   DATASET_ROOT=.../skill_walk_to_black_box_new_unified \
#     bash tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/tools/env/setup_env.sh"

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
PROMPT="${PROMPT:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# ponytail: chunked decode+encode so GPU is not idle during whole-ep CPU decode.
DECODE_BATCH="${DECODE_BATCH:-64}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
# 0 → RefVideoFrameSource auto: 16 on cpu decode / 2 on cuda NVDEC
DECODE_WORKERS="${DECODE_WORKERS:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-cpu}"
NGPU="${NGPU:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
RESUME="${RESUME:-1}"
# Avoid 8× unrestricted BLAS/OMP thrashing with co-tenant Isaac (tdcon).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
LOG_DIR="${LOG_DIR:-${DATASET_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/logs}"
mkdir -p "${LOG_DIR}"

export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
# keep env in sync when launcher sets DECODE_DEVICE
export PHI0_VIDEO_DECODE_DEVICE="${DECODE_DEVICE}"

EXTRA=()
if [[ -n "${PROMPT}" ]]; then
  EXTRA+=(--prompt "${PROMPT}")
fi
RESUME_FLAG=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_FLAG=(--resume)
fi

echo "[dual_vlm_8gpu] init meta root=${DATASET_ROOT} decode=${DECODE_DEVICE} workers=${DECODE_WORKERS} encode_bs=${BATCH_SIZE} decode_bs=${DECODE_BATCH} omp=${OMP_NUM_THREADS}"
"${PHI0_PY}" "${ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
  --dataset-root "${DATASET_ROOT}" \
  --init-only \
  --max-seq-len "${MAX_SEQ_LEN}" \
  "${EXTRA[@]}"

PIDS=()
for i in $(seq 0 $((NGPU - 1))); do
  LOG="${LOG_DIR}/shard_${i}_of_${NGPU}.log"
  echo "[dual_vlm_8gpu] launch shard ${i}/${NGPU} → ${LOG}"
  CUDA_VISIBLE_DEVICES="${i}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    TORCH_NUM_THREADS="${TORCH_NUM_THREADS}" \
    "${PHI0_PY}" "${ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
      --dataset-root "${DATASET_ROOT}" \
      --shard "${i}/${NGPU}" \
      --batch-size "${BATCH_SIZE}" \
      --decode-batch "${DECODE_BATCH}" \
      --video-backend "${VIDEO_BACKEND}" \
      --decode-workers "${DECODE_WORKERS}" \
      --decode-device "${DECODE_DEVICE}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      "${RESUME_FLAG[@]}" \
      "${EXTRA[@]}" \
      >"${LOG}" 2>&1 &
  PIDS+=("$!")
done

echo "[dual_vlm_8gpu] pids=${PIDS[*]} logs=${LOG_DIR}"
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[dual_vlm_8gpu] pid ${pid} failed" >&2
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[dual_vlm_8gpu] FAILED; check ${LOG_DIR}" >&2
  exit 1
fi
echo "[dual_vlm_8gpu] all shards done → ${DATASET_ROOT}/meta/vlm_frame_latents_qwen3vl_dual"
