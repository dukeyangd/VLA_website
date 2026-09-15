#!/usr/bin/env python3
"""Sidecar: probe local/remote sizes for active Studio uploads → static/upload-progress.json.

Does not touch the Studio process or rsync. Safe to run while uploads are in flight.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "upload-progress.json"
SSH_CONFIG = Path(os.environ.get("SSH_CONFIG", str(Path.home() / ".ssh" / "config"))).expanduser()
STUDIO = os.environ.get("STUDIO_URL", "http://127.0.0.1:7890")


def ssh_du(target: str, remote_path: str) -> int:
    args = ["ssh"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    args += [
        "-o", "BatchMode=yes",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=8",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=60",
        "-o", "ControlPath=/tmp/humanoid-studio-%C",
        "--", target,
        f"du -sb -- {remote_path} 2>/dev/null | awk '{{print $1}}'",
    ]
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=30, check=False)
        return int((r.stdout or "").strip().splitlines()[0])
    except Exception:
        return 0


def local_du(path: str) -> int:
    try:
        r = subprocess.run(
            ["du", "-sb", "--", path],
            text=True, capture_output=True, timeout=30, check=False,
        )
        return int((r.stdout or "").strip().split()[0])
    except Exception:
        return 0


def host_target(state: dict, host_id: str) -> str:
    for h in state.get("hosts") or []:
        if h.get("id") == host_id:
            return str(h.get("target") or host_id)
    return host_id or "cluster_0"


def fetch_state() -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{STUDIO}/api/state", timeout=10) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    idle_rounds = 0
    while True:
        try:
            state = fetch_state()
        except Exception as exc:
            OUT.write_text(json.dumps({"ok": False, "error": str(exc), "items": {}}, ensure_ascii=False) + "\n")
            time.sleep(3)
            continue

        uploading = [c for c in (state.get("collections") or []) if c.get("status") == "uploading"]
        items: dict[str, dict] = {}
        for c in uploading:
            cid = str(c.get("id") or "")
            local = str(c.get("local_dir") or "")
            remote = str(c.get("remote_dir") or "")
            host_id = str(c.get("host_id") or "cluster_0")
            total = int(c.get("bytes_total") or 0) or (local_du(local) if local else 0)
            done = ssh_du(host_target(state, host_id), remote) if remote else 0
            pct = 0
            if total > 0:
                pct = max(0, min(99, int(done * 100 / total)))
            items[cid] = {
                "pct": pct,
                "done": done,
                "total": total,
                "local_dir": local,
                "remote_dir": remote,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        OUT.write_text(json.dumps({"ok": True, "items": items}, ensure_ascii=False, indent=2) + "\n")
        if uploading:
            idle_rounds = 0
            time.sleep(2.5)
        else:
            idle_rounds += 1
            if idle_rounds >= 8:
                break
            time.sleep(3)


if __name__ == "__main__":
    main()
