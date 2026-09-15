#!/usr/bin/env python3
"""Plot z/hand for 820mix_v2 B8 e4 fresh0 (step counter from 0)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPAN = 30
WARM = 201
X_MAX = 120000
Y_Z_MAX = 0.06
Y_HAND_MAX = 0.02
BAR = re.compile(
    r"\[distill\]\s+(\d+)/\d+\s+.*?loss=([0-9.eE+-]+)"
    r"(?:\s+z=([0-9.eE+-]+))?(?:\s+hand=([0-9.eE+-]+))?"
)

LOG_DIR = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs")


def _latest_fresh0_log() -> Path:
    cands = sorted(LOG_DIR.glob("820mix_v2_full_pv09_e3_b8_e4_fresh0_*.log"))
    if not cands:
        raise FileNotFoundError("no 820mix_v2_full_pv09_e3_b8_e4_fresh0_*.log")
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


def stitch() -> tuple[dict[int, tuple[float, float | None, float | None]], list[dict]]:
    fresh0_log = _latest_fresh0_log()
    segments = (("B8 e4 fresh0 (0→)", fresh0_log, "#2563eb", 1, 10**9),)
    merged: dict[int, tuple[float, float | None, float | None]] = {}
    meta = []
    for label, path, color, lo, hi in segments:
        raw = parse_log(path)
        kept = {s: v for s, v in raw.items() if lo <= s <= hi}
        merged.update(kept)
        steps = sorted(kept)
        meta.append(
            {
                "label": label,
                "log": str(path),
                "color": color,
                "n": len(steps),
                "step_lo": steps[0] if steps else None,
                "step_hi": steps[-1] if steps else None,
            }
        )
    return merged, meta


def series(d, field: int):
    steps, raw = [], []
    for s in sorted(k for k in d if WARM <= k <= X_MAX):
        v = d[s][field]
        if v is None:
            continue
        steps.append(s)
        raw.append(v)
    return steps, ewm(raw, SPAN), raw


def draw(ax, steps, raw, ema, *, ylabel: str, color: str, show_raw: bool):
    ax.plot(steps, ema, color=color, lw=2.0, label=f"{ylabel} ema", zorder=3)
    if show_raw:
        ax.plot(
            steps,
            raw,
            color=color,
            alpha=0.22,
            lw=0.8,
            ls="-",
            zorder=2,
        )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def main():
    fresh0_log = _latest_fresh0_log()
    png = fresh0_log.with_name(
        fresh0_log.name.replace("820mix_v2_full_pv09_e3_b8_e4_fresh0", "820mix_v2_b8_e4_fresh0_loss").replace(".log", ".png")
    )
    snap = png.with_suffix(".json")
    data, seg_meta = stitch()
    sz, ez, rz = series(data, 1)
    sh, eh, rh = series(data, 2)
    if not sz:
        print("no bars stitched yet")
        return 1

    fig, (ax_z, ax_h) = plt.subplots(2, 1, figsize=(10.5, 7.2), dpi=130, sharex=True)
    draw(ax_z, sz, rz, ez, ylabel="z", color="#2563eb", show_raw=False)
    draw(ax_h, sh, rh, eh, ylabel="hand", color="#2563eb", show_raw=False)

    leg = [f"{m['label']} [{m['step_lo']}..{m['step_hi']}] n={m['n']}" for m in seg_meta if m["n"]]
    ax_z.set_title(
        "820mix_v2 B8 e4 fresh0  ema span="
        f"{SPAN}  warm>{WARM}  x∈[0,{X_MAX}]\n"
        + " | ".join(leg),
        fontsize=9,
    )
    ax_h.set_xlabel("step")
    ax_z.set_xlim(0, X_MAX)
    ax_h.set_xlim(0, X_MAX)
    ax_z.set_ylim(0, Y_Z_MAX)
    ax_h.set_ylim(0, Y_HAND_MAX)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    plt.close(fig)

    snap_payload = {
        "png": str(png),
        "log": str(fresh0_log),
        "segments": seg_meta,
        "last_step": sz[-1],
        "z_ema": round(float(ez[-1]), 4),
        "hand_ema": round(float(eh[-1]), 4),
    }
    snap.write_text(json.dumps(snap_payload, indent=2), encoding="utf-8")
    print(json.dumps(snap_payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
