#!/usr/bin/env bash
# GT replay: skill_0 (rotate to person) + skill_5 (pick rubbish).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/skill0_skill5_gt_replay_${STAMP}}"
WS_LOG_ROOT="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/logs"
mkdir -p "${OUT_ROOT}"

PICKS=(
  "skill_0:/mnt/data2/wpy/workspace/820demo/820demo_skill_0_release_unified:0"
  "skill_5_pick_rubbish:/mnt/data2/wpy/workspace/820demo/820demo_skill_5_pick_rubbish_release_unified:0"
)

for spec in "${PICKS[@]}"; do
  IFS=: read -r tag ref ep <<<"${spec}"
  WORK="${OUT_ROOT}/${tag}_ep${ep}"
  EVAL_LOG="${WS_LOG_ROOT}/gt_sonic_replay_${tag}_ep${ep}_${STAMP}.log"
  MP4="${WORK}/gt_sonic_replay.mp4"
  mkdir -p "${WORK}"
  echo "=== ${tag} ep=${ep} ref=${ref} ==="
  env \
    UNIFIED_ROOT="${ref}" UNIFIED_EP="${ep}" EP="${ep}" \
    DEPLOY_POLICY_DIR=release TOKEN_SOURCE=unified_slice \
    WORK_DIR="${WORK}" CUDA_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
    LOG="${EVAL_LOG}" \
    bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh"
  for _ in $(seq 1 600); do
    if grep -q '\[sonic_latent\] done work_dir=' "${EVAL_LOG}" 2>/dev/null; then
      echo "done ${tag} mp4=${MP4} size=$(stat -c%s "${MP4}" 2>/dev/null || echo 0)"
      break
    fi
    sleep 3
  done
done
echo "OUT_ROOT=${OUT_ROOT}"
