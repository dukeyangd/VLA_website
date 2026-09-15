#!/usr/bin/env bash
# One-shot: clear stuck SSH ControlMaster reads, mark dead skill1 job, restart Studio.
set -euo pipefail
ROOT=/mnt/data2/gzy/workspace/wb-vla-data-station-src
LOG=/mnt/data2/wpy/workspace/Phi_0_wpy/logs/skill1_walk_h32_b32_ddp8_e10_20260909_045529.log

pkill -f 'cluster_0 python3 -c .*skill1_walk_h32_b32_ddp8_e10_20260909_045529' 2>/dev/null || true
kill 70148 2>/dev/null || true
sleep 1
rm -f /tmp/humanoid-studio-c9d4f3bcb4aab49ed5e35ef4c7e138739ece8fd2

if [[ -f "$LOG" ]] && ! grep -q '\[studio\] train_exit=' "$LOG"; then
  echo "[studio] train_exit=1  # recovered: GPU distill gone, SSH zombie cleaned $(date -Iseconds)" >> "$LOG"
fi

python3 <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("/mnt/data2/gzy/workspace/wb-vla-data-station-src/state/studio.json")
d = json.loads(p.read_text())
for j in d.get("train_jobs") or []:
    if j.get("id") == "train_2d453f94":
        j["status"] = "error"
        j["message"] = "训练进程已退出（约 step 1701）；请重新启动（将走 tmux）"
        j["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")
print("marked", [(j["id"], j["status"]) for j in d.get("train_jobs") or []])
PY

FLASK=$(pgrep -f "$ROOT/.venv/bin/python app.py" | head -1 || true)
if [[ -n "${FLASK}" ]]; then kill "$FLASK" || true; fi
fuser -k 7890/tcp 2>/dev/null || true
sleep 1
cd "$ROOT"
nohup .venv/bin/python app.py --host 0.0.0.0 --port 7890 >/tmp/studio_gzy.log 2>&1 &
sleep 3
curl -s -m 5 -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:7890/
curl -s -m 8 http://127.0.0.1:7890/api/train/jobs/train_2d453f94/metrics | head -c 400; echo
