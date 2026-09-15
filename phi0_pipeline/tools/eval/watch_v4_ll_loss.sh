#!/usr/bin/env bash
# ponytail: 60s refresh for 820mix_v4_ll loss png (z/hand fixed axes)
set -euo pipefail
PLOT="/mnt/data2/wpy/workspace/Phi_0_wpy/tools/eval/plot_v4_ll_loss.py"
TICK="/tmp/loss_plot_v4_ll_watcher.log"
LOG_ARG="${1:-}"
while true; do
  if [[ -n "${LOG_ARG}" ]]; then
    python3 "$PLOT" "$LOG_ARG" >>"$TICK" 2>&1 || true
  else
    python3 "$PLOT" >>"$TICK" 2>&1 || true
  fi
  date '+%F %T' >>"$TICK"
  sleep 60
done
