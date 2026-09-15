#!/usr/bin/env bash
# After smoke train finishes, run MuJoCo viz on GPU 0.
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_OUT="${1:?train out dir}"
LOG="${2:?train log}"
CKPT="${TRAIN_OUT}/phi0_student_last.pt"
VIZ_OUT="${TRAIN_OUT}/mujoco_viz"
echo "[wait-viz] watching ${LOG}"
for i in $(seq 1 720); do
  if grep -qE '\[DISTILL_FABRIC\] done|distill.*done steps=' "${LOG}" 2>/dev/null; then
    break
  fi
  if ! pgrep -af 'run_online_vlm_mix_distill.sh|newton_boneseed_distill_fabric' >/dev/null; then
    echo "[wait-viz] train process gone at poll ${i}"
    break
  fi
  sleep 30
done
if grep -qE '\[DISTILL_FABRIC\] done|distill.*done steps=' "${LOG}" 2>/dev/null; then
  :
else
  echo "[wait-viz] train did not finish cleanly ${LOG}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[wait-viz] missing ckpt ${CKPT}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 1
fi
echo "[wait-viz] ckpt=${CKPT}"
export STUDENT_CKPT="${CKPT}"
export OUT="${VIZ_OUT}"
export HORIZON="${HORIZON:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash "${PHI0_ROOT}/tools/eval/run_stand_egypt_skill2_mujoco_viz.sh"
echo "[wait-viz] DONE ${VIZ_OUT}"
