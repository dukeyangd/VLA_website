#!/bin/bash
# ponytail: sequential demo5 GT sonic replay (sonic_v1_1 + dex3, robot-only)
set -euo pipefail
ROOT=/mnt/data2/wpy/workspace/Phi_0_wpy
UNIFIED=/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_sonic_v1_1_unified
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE=${OUT_BASE:-/mnt/data3/wpy/demo5_gt_sonic_replay_${STAMP}}
LOG_DIR=/mnt/data2/wpy/workspace/Phi_0_wpy/logs
MASTER_LOG=${MASTER_LOG:-${LOG_DIR}/demo5_gt_sonic_replay_${STAMP}.log}
mkdir -p "$OUT_BASE" "$LOG_DIR"
cd "$ROOT"
echo "OUT_BASE=$OUT_BASE" | tee "$MASTER_LOG"

for EP in 0 1 2 3 4; do
  echo "===== EP=${EP} start $(date -Is) =====" | tee -a "$MASTER_LOG"
  WORK_DIR="${OUT_BASE}/ep${EP}"
  LOG="${LOG_DIR}/demo5_gt_sonic_replay_${STAMP}_ep${EP}.log"
  H264="${LOG_DIR}/demo5_gt_sonic_replay_${STAMP}_ep${EP}_h264.mp4"
  mkdir -p "$WORK_DIR"
  # Never GPU0 (busy / user request). Default physical GPU2.
  GPU="${CUDA_DEVICES:-2}"
  UNIFIED_ROOT="$UNIFIED" \
  EP="$EP" \
  DEPLOY_POLICY_DIR=sonic_v1_1 \
  REVO2_HAND=0 \
  CUDA_DEVICES="$GPU" \
  WORK_DIR="$WORK_DIR" \
  LOG="$LOG" \
  H264_OUT="$H264" \
  bash tools/eval/launch_gt_sonic_replay.sh

  # Wait for a real mp4 (launch creates a tiny placeholder early — require >=500KB).
  ok=0
  for _ in $(seq 1 180); do
    sz=0
    if [[ -f "$H264" ]]; then
      sz=$(stat -c%s "$H264" 2>/dev/null || echo 0)
    elif [[ -f "${WORK_DIR}/gt_sonic_replay.mp4" ]]; then
      sz=$(stat -c%s "${WORK_DIR}/gt_sonic_replay.mp4" 2>/dev/null || echo 0)
    fi
    if [[ "$sz" -ge 500000 ]]; then
      echo "===== EP=${EP} mp4 ready size=${sz} $(date -Is) =====" | tee -a "$MASTER_LOG"
      ls -lh "${WORK_DIR}/gt_sonic_replay.mp4" "$H264" 2>/dev/null | tee -a "$MASTER_LOG" || true
      ok=1
      break
    fi
    if [[ -f "$LOG" ]] && rg -q 'Traceback \(most recent call last\)|ERROR: replay|TIMEOUT replay' "$LOG"; then
      # allow a few seconds for remux race; only fail if still tiny
      if [[ "$sz" -lt 500000 ]]; then
        echo "===== EP=${EP} FAILED =====" | tee -a "$MASTER_LOG"
        tail -60 "$LOG" | tee -a "$MASTER_LOG"
        exit 1
      fi
    fi
    sleep 10
  done
  if [[ "$ok" != 1 ]]; then
    echo "===== EP=${EP} TIMEOUT =====" | tee -a "$MASTER_LOG"
    tail -60 "$LOG" | tee -a "$MASTER_LOG" || true
    exit 1
  fi
  # ensure previous eval fully exited before next ep (ZMQ ports)
  sleep 5
done
echo DONE | tee -a "$MASTER_LOG"
