#!/usr/bin/env bash
# ponytail: 60s refresh for latest 820mix_v2 B8 e4 fresh0 loss png
set -euo pipefail
PLOT="/mnt/data2/wpy/workspace/Phi_0_wpy/tools/eval/plot_b8_aligned_resume_loss.py"
TICK="/tmp/loss_plot_e4_fresh0_watcher.log"
while true; do
  python3 "$PLOT" >>"$TICK" 2>&1 || true
  date '+%F %T' >>"$TICK"
  sleep 60
done
