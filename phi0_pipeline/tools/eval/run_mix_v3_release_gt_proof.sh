#!/usr/bin/env bash
# One GT sonic replay per mix_v3 skill/task (release decoder + unified_slice).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${UNIFIED_ROOT:-/mnt/data2/wpy/workspace/820demo/820demo_mix_v3_release_unified}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PHI0_ROOT:-${ROOT}}/experiments/mix_v3_release_gt_proof_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}/Phi_0_wpy/logs}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# skill_tag:ep:gpu  (one ep per mix v3 task)
PICKS=(
  "demo5_idle:0:0"
  "demo5_bow:1:1"
  "skill_1_walk:2:2"
  "skill_2:123:3"
  "skill_3:400:4"
)

echo "ref=${REF}"
echo "out=${OUT_ROOT}"

# ponytail: ZMQ ports 5555/5556/5557 are fixed in run_sonic_latent_sim_eval.sh — serial only.
SEQUENTIAL="${SEQUENTIAL:-1}"
for spec in "${PICKS[@]}"; do
  IFS=: read -r tag ep gpu <<<"${spec}"
  WORK="${OUT_ROOT}/${tag}_ep${ep}"
  LOG="${LOG_ROOT}/mix_v3_release_gt_${tag}_ep${ep}_${STAMP}.log"
  mkdir -p "${WORK}"
  echo "launch ${tag} ep=${ep} gpu=${gpu} sequential=${SEQUENTIAL}"
  if [[ "${SEQUENTIAL}" == "1" ]]; then
    env \
      UNIFIED_ROOT="${REF}" \
      UNIFIED_EP="${ep}" \
      EP="${ep}" \
      DEPLOY_POLICY_DIR=release \
      TOKEN_SOURCE=unified_slice \
      WORK_DIR="${WORK}" \
      CUDA_DEVICES="${gpu}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh" \
      >"${LOG}" 2>&1
    echo "done ${tag} mp4=${WORK}/gt_sonic_replay.mp4"
  else
    nohup env \
      UNIFIED_ROOT="${REF}" \
      UNIFIED_EP="${ep}" \
      EP="${ep}" \
      DEPLOY_POLICY_DIR=release \
      TOKEN_SOURCE=unified_slice \
      WORK_DIR="${WORK}" \
      CUDA_DEVICES="${gpu}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      bash "${ROOT}/tools/eval/launch_gt_sonic_replay.sh" \
      >"${LOG}" 2>&1 &
    echo "${tag} pid=$! log=${LOG} work=${WORK}/gt_sonic_replay.mp4"
  fi
done

echo "OUT_ROOT=${OUT_ROOT}"
echo "See ${OUT_ROOT}/*/gt_sonic_replay.mp4 when done"
