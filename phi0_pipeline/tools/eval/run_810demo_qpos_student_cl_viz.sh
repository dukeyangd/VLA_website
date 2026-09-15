#!/usr/bin/env bash
# 810demo closed-loop Newton-GL viz for mode=vla (disk-z) / distill students.
# Uses CONTROL=qpos_student (obs_hist+lang → q_head), NOT 810 dual-VLM ZMQ CL.
#
# Usage:
#   bash tools/eval/run_810demo_qpos_student_cl_viz.sh <CKPT> <OUT_BASE> [ep0|all|both]
# Env:
#   CUDA_VISIBLE_DEVICES  — default 0
#   EPISODES              — space-separated ep ids (default: 0 for smoke; all=0..27)
#   NUM_STEPS_CAP         — max frames per ep (default 600)
#   WAVE                  — qpos|latent|both (default qpos)
set -euo pipefail

CKPT="${1:?ckpt}"
OUT_BASE="${2:?out_base}"
WAVE="${3:-${WAVE:-qpos}}"
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/810demo_egypt_layout}"
LOG="${LOG:-${OUT_BASE}/810demo_qpos_student_cl_viz.log}"

export HORIZON="${HORIZON:-32}"
export PHI0_CHUNK_EXEC="${PHI0_CHUNK_EXEC:-first}"
# first-token A/B is incompatible with RTC open-loop play; keep legacy 810 default.
export USE_RTC="${USE_RTC:-0}"
export USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-0}"
# VLA train default K=1 — must match ckpt history_len.
export PHI0_DISTILL_OBS_HIST="${PHI0_DISTILL_OBS_HIST:-0}"
export FRAME_SKIP="${FRAME_SKIP:-2}"
NUM_STEPS_CAP="${NUM_STEPS_CAP:-600}"

mkdir -p "$OUT_BASE" "$(dirname "$LOG")"
: >"$LOG"

# tag ep REF_START NUM_STEPS (absolute tape index after file-000..027 concat)
ALL_EPS=(
  "ep0 0 0 8630"
  "ep1 1 8630 7047"
  "ep2 2 15677 6832"
  "ep3 3 22509 6271"
  "ep4 4 28780 6188"
  "ep5 5 34968 6670"
  "ep6 6 41638 5378"
  "ep7 7 47016 6074"
  "ep8 8 53090 5975"
  "ep9 9 59065 5892"
  "ep10 10 64957 6174"
  "ep11 11 71131 6345"
  "ep12 12 77476 6266"
  "ep13 13 83742 6717"
  "ep14 14 90459 6523"
  "ep15 15 96982 7289"
  "ep16 16 104271 6786"
  "ep17 17 111057 7307"
  "ep18 18 118364 7657"
  "ep19 19 126021 7042"
  "ep20 20 133063 6591"
  "ep21 21 139654 6030"
  "ep22 22 145684 5829"
  "ep23 23 151513 7040"
  "ep24 24 158553 7233"
  "ep25 25 165786 6146"
  "ep26 26 171932 6064"
  "ep27 27 177996 6691"
)

if [[ -n "${EPISODES:-}" ]]; then
  WANT=()
  for e in ${EPISODES}; do WANT+=("$e"); done
else
  WANT=(0)
fi

SKILLS=()
for row in "${ALL_EPS[@]}"; do
  read -r tag ep ref_start num_steps <<<"${row}"
  keep=0
  for w in "${WANT[@]}"; do
    if [[ "${w}" == "all" || "${w}" == "${ep}" ]]; then
      keep=1
      break
    fi
  done
  if [[ "${keep}" == "1" ]]; then
    n="${num_steps}"
    if [[ "${n}" -gt "${NUM_STEPS_CAP}" ]]; then
      n="${NUM_STEPS_CAP}"
    fi
    SKILLS+=("${tag} ${ep} ${ref_start} ${n}")
  fi
done

if [[ "${#SKILLS[@]}" -eq 0 ]]; then
  echo "[810_cl] no episodes selected (EPISODES=${EPISODES:-0})" >&2
  exit 2
fi

run_one() {
  local control="$1" tag="$2" ref_start="$3" num_steps="$4" gpu="${5:-0}"
  local out="${OUT_BASE}/${control}/${tag}"
  mkdir -p "$out"
  echo "===== $(date -Iseconds) control=${control} ${tag} ref_start=${ref_start} steps=${num_steps} gpu=${gpu} =====" | tee -a "$LOG"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export CONTROL="${control}"
    export TAG="${tag}"
    export OUT="${out}"
    export STUDENT_CKPT="${CKPT}"
    export REF_ROOT="${REF_ROOT}"
    export REF_START="${ref_start}"
    export NUM_STEPS="${num_steps}"
    export MAX_REF_FRAMES="${num_steps}"
    export HORIZON="${HORIZON:-32}"
    export PHI0_CHUNK_EXEC="${PHI0_CHUNK_EXEC:-first}"
    export USE_RTC="${USE_RTC:-0}"
    export USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-0}"
    export PHI0_DISTILL_OBS_HIST="${PHI0_DISTILL_OBS_HIST:-0}"
    bash "${PHI0_ROOT}/tools/eval/run_qpos_student_newton_isaac_viz.sh"
  ) >>"$LOG" 2>&1
  local mp4="${out}/infer_${control}_newton_gl.mp4"
  if [[ -f "${mp4}" ]]; then
    echo "[done] ${control}/${tag} mp4=${mp4}" | tee -a "$LOG"
    if [[ -f "${out}/newton_isaac_viz_contract.json" ]]; then
      python3 -c "import json; d=json.load(open('${out}/newton_isaac_viz_contract.json')); print('  infer_ok=',d.get('infer_ok'),'root_z_end=',d.get('root_z_end'),'mean_abs_q=',d.get('mean_abs_q_cmd_vs_q_act'))" | tee -a "$LOG" || true
    fi
  else
    echo "[FAIL] ${control}/${tag} no mp4; tail:" | tee -a "$LOG"
    tail -n 40 "${out}/newton_isaac_viz.log" | tee -a "$LOG" || true
    return 1
  fi
}

launch_wave() {
  local control="$1"
  local i=0
  for row in "${SKILLS[@]}"; do
    read -r tag ep ref_start num_steps <<<"${row}"
    run_one "${control}" "${tag}" "${ref_start}" "${num_steps}" "${CUDA_VISIBLE_DEVICES:-0}"
    i=$((i + 1))
  done
}

echo "[810_cl] ckpt=${CKPT}" | tee -a "$LOG"
echo "[810_cl] out=${OUT_BASE} wave=${WAVE} H=${HORIZON} ref=${REF_ROOT} eps=${WANT[*]} cap=${NUM_STEPS_CAP}" | tee -a "$LOG"

ec=0
case "${WAVE}" in
  qpos)
    launch_wave qpos_student || ec=1
    ;;
  latent)
    launch_wave student || ec=1
    ;;
  both)
    launch_wave qpos_student || ec=1
    launch_wave student || ec=1
    ;;
  *)
    echo "unknown WAVE=${WAVE} (qpos|latent|both)" >&2
    exit 2
    ;;
esac
exit "$ec"
