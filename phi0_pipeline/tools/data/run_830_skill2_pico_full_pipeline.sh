#!/usr/bin/env bash
# Pack + VLM cache + NVMe symlink + launch skill2 nosim training.
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPE_LOG="${PIPE_LOG:-/mnt/data2/wpy/workspace/logs/830_skill2_pico_pipeline_${STAMP}.log}"

{
  echo "[pipeline] start $(date -Is) stamp=${STAMP}"
  bash "${PHI0_ROOT}/tools/data/run_830_skill2_pico_pack_and_cache.sh"
  echo "[pipeline] pack+cache done; launching training"
  STAMP="${STAMP}" bash "${PHI0_ROOT}/tools/train/run_830_skill2_pico_vlm_cache_distill.sh"
  echo "[pipeline] done $(date -Is)"
} 2>&1 | tee "${PIPE_LOG}"
