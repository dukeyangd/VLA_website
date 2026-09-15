#!/usr/bin/env bash
# Pack + VLM cache + launch skill3 distill (GPUs 4-7).
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPE_LOG="${PIPE_LOG:-${PHI0_ROOT}/logs/830_skill3_place_basket_pipeline_${STAMP}.log}"

{
  echo "[pipeline] start $(date -Is) stamp=${STAMP}"
  STAMP="${STAMP}" bash "${PHI0_ROOT}/tools/data/run_830_skill3_place_basket_pack_and_cache.sh"
  echo "[pipeline] pack+cache done; launching training"
  STAMP="${STAMP}" bash "${PHI0_ROOT}/tools/train/run_830_skill3_place_basket_vlm_cache_distill.sh"
  echo "[pipeline] done $(date -Is)"
} 2>&1 | tee "${PIPE_LOG}"
