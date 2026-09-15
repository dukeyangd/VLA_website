#!/usr/bin/env bash
# Watch an online_vlm OUT for the first ``phi0_student_last.pt`` (+ optim), then
# kill that job and relaunch mix distill with STUDENT_CKPT=that pt + current
# recipe defaults (PHI0_P_VISION=0.8, no RSI_NO_VIDEO weight, resume steps_done).
#
#   WATCH_OUT=/mnt/data3/wpy/online_vlm_...092620 \
#     bash tools/train/watch_ckpt_restart_online_vlm.sh
#
# Optional: DRY_RUN=1 (detect only), POLL_SEC=30, KILL_OLD=1, PHI0_P_VISION=0.8
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WATCH_OUT="${WATCH_OUT:?set WATCH_OUT to the running distill out_dir}"
POLL_SEC="${POLL_SEC:-30}"
STABLE_POLLS="${STABLE_POLLS:-2}"   # size unchanged this many polls
KILL_OLD="${KILL_OLD:-1}"
DRY_RUN="${DRY_RUN:-0}"
MIN_WEIGHT_BYTES="${MIN_WEIGHT_BYTES:-50000000}"  # ~50MB floor (weights are ~0.5G)

WEIGHTS="${WATCH_OUT}/phi0_student_last.pt"
OPTIM="${WATCH_OUT}/phi0_student_last_optim.pt"
WATCH_LOG="${WATCH_LOG:-${PHI0_ROOT}/logs/watch_ckpt_restart_$(basename "${WATCH_OUT}").log}"
mkdir -p "$(dirname "${WATCH_LOG}")"

log() {
  # ponytail: append-only; caller may also redirect stdout — don't tee (doubles lines).
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"${WATCH_LOG}"
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "watching ${WATCH_OUT} every ${POLL_SEC}s → restart with latest recipe"
log "weights=${WEIGHTS}"

prev_sz=-1
stable=0
while true; do
  if [[ -f "${WEIGHTS}" && -f "${OPTIM}" ]]; then
    sz=$(stat -c '%s' "${WEIGHTS}" 2>/dev/null || echo 0)
    osz=$(stat -c '%s' "${OPTIM}" 2>/dev/null || echo 0)
    if [[ "${sz}" -ge "${MIN_WEIGHT_BYTES}" && "${osz}" -gt 0 ]]; then
      if [[ "${sz}" -eq "${prev_sz}" ]]; then
        stable=$((stable + 1))
      else
        stable=0
        prev_sz="${sz}"
      fi
      log "seen weights=${sz}B optim=${osz}B stable=${stable}/${STABLE_POLLS}"
      if [[ "${stable}" -ge "${STABLE_POLLS}" ]]; then
        break
      fi
    else
      log "ckpt present but tiny/incomplete weights=${sz} optim=${osz}"
      stable=0
      prev_sz=-1
    fi
  else
    # progress hint from metrics if any
    if [[ -f "${WATCH_OUT}/distill_metrics.json" ]]; then
      steps=$(python3 -c "import json;print(json.load(open('${WATCH_OUT}/distill_metrics.json')).get('steps_done',0))" 2>/dev/null || echo '?')
      log "waiting... steps_done≈${steps} (no last.pt yet)"
    else
      log "waiting... no last.pt"
    fi
    stable=0
    prev_sz=-1
  fi
  sleep "${POLL_SEC}"
done

log "ckpt ready: ${WEIGHTS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1 — not killing / not relaunching"
  exit 0
fi

if [[ "${KILL_OLD}" == "1" ]]; then
  # Match fabric workers by --out_dir <WATCH_OUT>
  mapfile -t PIDS < <(pgrep -f -- "--out_dir ${WATCH_OUT}" || true)
  if ((${#PIDS[@]})); then
    log "killing old job pids: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
    sleep 5
    # escalate leftovers
    mapfile -t LEFT < <(pgrep -f -- "--out_dir ${WATCH_OUT}" || true)
    if ((${#LEFT[@]})); then
      log "SIGKILL leftovers: ${LEFT[*]}"
      kill -9 "${LEFT[@]}" 2>/dev/null || true
      sleep 2
    fi
  else
    log "no live pids for out_dir (already stopped?)"
  fi
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
# Distinct OUT so we do not overwrite the snapshot we resume from.
export STAMP
export STUDENT_CKPT="${WEIGHTS}"
export PHI0_P_VISION="${PHI0_P_VISION:-0.8}"
export NGPU="${NGPU:-8}"
export NUM_ENVS="${NUM_ENVS:-8}"
export EPOCHS="${EPOCHS:-2}"
export HORIZON="${HORIZON:-32}"
export W_HAND="${W_HAND:-1}"
export W_Q_HEAD="${W_Q_HEAD:-0}"
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE:-disk}"
export PHI0_VISION_DATALOADER="${PHI0_VISION_DATALOADER:-1}"
export STUDENT_DRIVE="${STUDENT_DRIVE:-1}"
# Tag in default OUT path via STAMP prefix note in PHI0_DISTILL_OUT
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/online_vlm_820mix_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_resume_pv${PHI0_P_VISION}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/820mix_h${HORIZON}_resume_pv${PHI0_P_VISION}_${STAMP}.log}"

log "relaunch STUDENT_CKPT=${STUDENT_CKPT}"
log "PHI0_P_VISION=${PHI0_P_VISION} OUT=${PHI0_DISTILL_OUT}"
log "new log → ${LOG_FILE}"

cd "${PHI0_ROOT}"
# shellcheck disable=SC2086
nohup bash tools/train/run_online_vlm_mix_distill.sh >>"${LOG_FILE}" 2>&1 &
echo $! >"${PHI0_DISTILL_OUT}.watch_relaunch.pid" 2>/dev/null || true
log "relaunch pid=$! — monitor ${LOG_FILE}"
log "done"
