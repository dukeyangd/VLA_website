#!/usr/bin/env bash
# cluster_0 上一键启动 Studio（释放端口、确保 venv、前台或后台）。
# 用法:
#   bash scripts/start_on_cluster.sh           # 前台
#   bash scripts/start_on_cluster.sh --bg      # 后台，日志 /tmp/studio_gzy.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7890}"
BG=0
[[ "${1:-}" == "--bg" || "${1:-}" == "-d" ]] && BG=1

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "[setup] creating .venv ..."
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install -U pip
  "$ROOT/.venv/bin/python" -m pip install -r requirements.txt
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1
fi

export HOST PORT
export PYTHON_BIN="$ROOT/.venv/bin/python"

# Sync skill cards from shared 830demo (other hosts). SKIP_SKILL_SYNC=1 to skip.
if [[ "${SKIP_SKILL_SYNC:-0}" != "1" ]]; then
  echo "[start] sync skills from 830demo…"
  "$PYTHON_BIN" "$ROOT/scripts/sync_skills_from_830demo.py" || echo "[start] skill sync failed (continuing)" >&2
fi

echo "Starting Studio on http://0.0.0.0:${PORT}  (open via SSH tunnel or cluster browser)"
echo "  Doc: $ROOT/README.md"
echo "  Phi0: /mnt/data2/wpy/workspace/Phi_0_wpy"

if [[ "$BG" -eq 1 ]]; then
  LOG="${STUDIO_LOG:-/tmp/studio_gzy.log}"
  nohup "$PYTHON_BIN" app.py --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
  echo "PID $!  log: $LOG"
  sleep 1
  curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://127.0.0.1:${PORT}/" || true
else
  exec "$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
fi
