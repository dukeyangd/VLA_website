#!/usr/bin/env python3
"""Plot z/hand for 820mix_v4_ll B8 e4 (fixed axes; no total)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPAN = 30
WARM = 201
X_MAX = 240000
Y_Z_MAX = 0.1
Y_HAND_MAX = 0.1
BAR = re.compile(
    r"\[distill\]\s+(\d+)/\d+\s+.*?loss=([0-9.eE+-]+)"
    r"(?:\s+z=([0-9.eE+-]+))?(?:\s+hand=([0-9.eE+-]+))?"
)

LOG_DIR = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs")
GLOB = "online_vlm_820mix_h32_b8_ddp8_*.log"
# Prefer explicit run tag when present.
PREF = "820mix_v4_ll"


def _pick_log() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    tagged = sorted(LOG_DIR.glob(f"*{PREF}*.log"))
    if tagged:
        return tagged[-1]
    cands = sorted(LOG_DIR.glob(GLOB))
    if not cands:
        raise FileNotFoundError(f"no {GLOB} under {LOG_DIR}")
    return cands[-1]


def ewm(xs, span):
    a = 2.0 / (span + 1.0)
    out, s = [], None
    for x in xs:
        s = x if s is None else (a * x + (1.0 - a) * s)
        out.append(s)
    return out


def parse_log(path: Path) -> dict[int, tuple[float, float | None, float | None]]:
    d: dict[int, tuple[float, float | None, float | None]] = {}
    if not path.exists():
        return d
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        m = BAR.search(line)
        if not m:
            continue
        z = float(m.group(3)) if m.group(3) else None
        h = float(m.group(4)) if m.group(4) else None
        d[int(m.group(1))] = (float(m.group(2)), z, h)
    return d


def series(d, field: int):
    steps, raw = [], []
    for s in sorted(k for k in d if WARM <= k <= X_MAX):
        v = d[s][field]
        if v is None:
            continue
        steps.append(s)
        raw.append(v)
    return steps, ewm(raw, SPAN), raw


def main():
    log = _pick_log()
    png = log.with_name(log.stem + "_loss.png")
    if "online_vlm_820mix" in log.name:
        png = log.with_name(log.name.replace(".log", "_loss.png"))
    snap = png.with_suffix(".json")
    data = parse_log(log)
    sz, ez, rz = series(data, 1)
    sh, eh, rh = series(data, 2)
    if not sz:
        print(f"no bars yet in {log}")
        return 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(sz, ez, color="#2563eb", lw=2.0, label="z ema")
    axes[0].plot(sz, rz, color="#2563eb", alpha=0.22, lw=0.8)
    axes[0].set_ylabel("z")
    axes[0].set_ylim(0, Y_Z_MAX)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(sh, eh, color="#16a34a", lw=2.0, label="hand ema")
    axes[1].plot(sh, rh, color="#16a34a", alpha=0.22, lw=0.8)
    axes[1].set_ylabel("hand")
    axes[1].set_ylim(0, Y_HAND_MAX)
    axes[1].set_xlabel("step")
    axes[1].set_xlim(0, X_MAX)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"v4_ll loss  {log.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    meta = {
        "log": str(log),
        "png": str(png),
        "n_z": len(sz),
        "step_hi": sz[-1] if sz else None,
        "z_last": ez[-1] if ez else None,
        "hand_last": eh[-1] if eh else None,
    }
    snap.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
