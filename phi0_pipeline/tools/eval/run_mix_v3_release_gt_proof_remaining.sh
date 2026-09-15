#!/usr/bin/env bash
# Remaining mix_v3 GT replays (serial — ZMQ port conflict if parallel).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="/mnt/data2/wpy/workspace/820demo/820demo_mix_v3_release_unified"
OUT_ROOT="/mnt/data2/wpy/workspace/Phi_0_wpy/experiments/mix_v3_release_gt_proof_20260819_114057"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/mnt/data2/wpy/workspace/Phi_0_wpy/logs"
WS_LOG_ROOT="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/logs"
PICKS=(
  "skill_1_walk:2:0"
  "skill_2:123:0"
  "skill_3:400:0"
)
for spec in "${PICKS[@]}"; do
  IFS=: read -r tag ep gpu <<<"${spec}"
  WORK="${OUT_ROOT}/${tag}_ep${ep}"
  LAUNCH_LOG="${LOG_ROOT}/mix_v3_release_gt_${tag}_ep${ep}_${STAMP}.log"
  EVAL_LOG="${WS_LOG_ROOT}/gt_sonic_replay_ep${ep}_${STAMP}.log"
  MP4="${WORK}/gt_sonic_replay.mp4"
  if [[ -f "${MP4}" ]] && [[ $(stat -c%s "${MP4}") -gt 200000 ]]; then
    echo "skip ${tag} ep=${ep} existing ${MP4}"
    continue
  fi
  rm -f "${MP4}" "${WORK}/.record_start" "${WORK}/.record_stop" 2>/dev/null || true
  mkdir -p "${WORK}"
  echo "=== launch ${tag} ep=${ep} eval_log=${EVAL_LOG} ==="
  env \
    UNIFIED_ROOT="${REF}" UNIFIED_EP="${ep}" EP="${ep}" \
    DEPLOY_POLICY_DIR=release TOKEN_SOURCE=unified_slice \
    WORK_DIR="${WORK}" CUDA_DEVICES="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
    LOG="${EVAL_LOG}" \
    bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh" \
    >"${LAUNCH_LOG}" 2>&1
  for _ in $(seq 1 600); do
    if grep -q '\[sonic_latent\] done work_dir=' "${EVAL_LOG}" 2>/dev/null; then
      sz=$(stat -c%s "${MP4}" 2>/dev/null || echo 0)
      echo "done ${tag} size=${sz} mp4=${MP4}"
      break
    fi
    sleep 3
  done
done
echo "all remaining done under ${OUT_ROOT}"
