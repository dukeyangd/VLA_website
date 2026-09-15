#!/usr/bin/env bash
# Demo5skill Newton-GL viz (infer only; does not change distill train).
# Latent/qpos both use VLA-style open-loop: exec H tokens then replan.
# Usage:
#   bash tools/eval/run_demo5skill_cl_viz.sh <CKPT> <OUT_BASE> [qpos|latent|both]
# Env:
#   CUDA_VISIBLE_DEVICES  — single-GPU sequential (default 0)
#   PARALLEL=1            — fan out one GPU per skill (uses GPUs 0..N-1)
#   WAVE=qpos|latent|both — same as 3rd arg (default both)
set -euo pipefail

CKPT="${1:?ckpt}"
OUT_BASE="${2:?out_base}"
WAVE="${3:-${WAVE:-both}}"
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0}"
LOG="${LOG:-${OUT_BASE}/demo5skill_cl_viz.log}"

export HORIZON="${HORIZON:-32}"
# Infer schedule only — same as prior distill infer / VLA chunk play.
export PHI0_CHUNK_EXEC="${PHI0_CHUNK_EXEC:-open_loop}"
export USE_LANG_LATENT_CACHE=1
# 820mix deploy viz is tools/eval/run_student_demo5_mujoco_viz.sh (live VLM + short Chinese).
# This Newton path still uses BoneSEED lang cache (same Chinese strings after json swap).
export PHI0_LANG_LATENTS_DIRNAME="${PHI0_LANG_LATENTS_DIRNAME:-lang_latents_qwen3vl_demo5skill}"
export PHI0_DISTILL_OBS_HIST="${PHI0_DISTILL_OBS_HIST:-1}"
export FRAME_SKIP="${FRAME_SKIP:-2}"

mkdir -p "$OUT_BASE" "$(dirname "$LOG")"
: >"$LOG"

# tag ep REF_START NUM_STEPS
SKILLS=(
  "ep291_idle 291 104202 948"
  "ep110825_egypt 110825 40619720 558"
  "ep81_spin 81 34568 726"
  "ep28_wave 28 9227 528"
  "ep37561_bow 37561 13787216 486"
)

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
    export PHI0_CHUNK_EXEC="${PHI0_CHUNK_EXEC:-open_loop}"
    export USE_LANG_LATENT_CACHE=1
    export PHI0_DISTILL_OBS_HIST="${PHI0_DISTILL_OBS_HIST:-1}"
    bash "${PHI0_ROOT}/tools/eval/run_qpos_student_newton_isaac_viz.sh"
  ) >>"$LOG" 2>&1
  local mp4="${out}/infer_${control}_newton_gl.mp4"
  local pid
  pid=$(pgrep -f "newton_qpos_student_isaac_viz.py.*${out}" | head -1 || true)
  if [[ -n "${pid}" ]]; then
    while kill -0 "${pid}" 2>/dev/null; do sleep 15; done
  fi
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
  local parallel="${PARALLEL:-0}"
  local i=0
  local pids=()
  for row in "${SKILLS[@]}"; do
    read -r tag ep ref_start num_steps <<<"${row}"
    local gpu=0
    if [[ "${parallel}" == "1" ]]; then
      gpu="${i}"
    fi
    if [[ "${parallel}" == "1" ]]; then
      run_one "${control}" "${tag}" "${ref_start}" "${num_steps}" "${gpu}" &
      pids+=($!)
    else
      run_one "${control}" "${tag}" "${ref_start}" "${num_steps}" "${gpu}"
    fi
    i=$((i + 1))
  done
  if [[ "${parallel}" == "1" ]]; then
    local ec=0
    for p in "${pids[@]}"; do
      wait "$p" || ec=1
    done
    return "$ec"
  fi
}

echo "[demo5_cl] ckpt=${CKPT}" | tee -a "$LOG"
echo "[demo5_cl] out=${OUT_BASE} wave=${WAVE} H=${HORIZON} chunk_exec=${PHI0_CHUNK_EXEC} lang_cache=1 obs_hist=${PHI0_DISTILL_OBS_HIST}" | tee -a "$LOG"

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

echo "===== ALL DONE $(date -Iseconds) out=${OUT_BASE} ec=${ec} =====" | tee -a "$LOG"
exit "${ec}"
