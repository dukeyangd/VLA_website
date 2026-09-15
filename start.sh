#!/usr/bin/env bash
# Start WB-VLA Data-station (Studio UI). Prefer local .venv when present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7890}"

# Free stale Studio on this port (Ctrl-C / crash often leaves 7890 occupied).
if command -v fuser >/dev/null 2>&1; then
  if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "[start] port ${PORT} busy — fuser -k ${PORT}/tcp"
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 0.5
  fi
elif command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${PORT}\\s"; then
  echo "Port ${PORT} is busy and fuser is unavailable." >&2
  echo "  Free it manually, or:  PORT=7891 bash start.sh" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  :
else
  PYTHON_BIN="python3"
fi

# Pull new skill cards / sessions from shared 830demo (other hosts).
# SKIP_SKILL_SYNC=1 bash start.sh  → skip
if [[ "${SKIP_SKILL_SYNC:-0}" != "1" ]]; then
  echo "[start] sync skills from 830demo…"
  "$PYTHON_BIN" "$ROOT/scripts/sync_skills_from_830demo.py" || echo "[start] skill sync failed (continuing)" >&2
fi

echo "[start] http://${HOST}:${PORT}  python=${PYTHON_BIN}"
exec "$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
