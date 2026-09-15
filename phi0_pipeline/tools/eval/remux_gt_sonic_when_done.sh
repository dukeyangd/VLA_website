#!/usr/bin/env bash
# Wait for GT sonic eval workdir to finish, then remux mp4v -> h264.
set -euo pipefail
WORK="${1:?work dir}"
LOG="${2:?log}"
H264="${3:?h264 out}"
OUT_NAME="${OUT_MP4_NAME:-gt_sonic_replay.mp4}"
OUT="${WORK}/${OUT_NAME}"
mkdir -p "${WORK}/logs"
for _ in $(seq 1 360); do
  if [[ -s "$OUT" ]] && ! pgrep -f "run_sonic_latent_sim_eval\.sh|run_pick_tissue_sonic_latent_eval\.sh" >/dev/null 2>&1 \
     && ! pgrep -f "run_810demo_closed_loop_eval.sh" >/dev/null 2>&1; then
    sleep 2
    if [[ -s "$OUT" ]]; then
      ffmpeg -y -i "$OUT" -c:v libx264 -pix_fmt yuv420p -crf 18 -an "$H264" \
        >"${WORK}/logs/ffmpeg_h264.log" 2>&1 || cp -f "$OUT" "$H264"
      echo "h264=$H264 bytes=$(stat -c%s "$H264")" | tee -a "$LOG"
      exit 0
    fi
  fi
  sleep 10
done
echo "remux watcher timeout work=$WORK" | tee -a "$LOG"
exit 1
