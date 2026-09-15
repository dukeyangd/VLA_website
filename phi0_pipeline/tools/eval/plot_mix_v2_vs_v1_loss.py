#!/usr/bin/env python3
"""z/hand overlay: mix_v1 vs mix_v2 B8 vs mix_v2 B32."""
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
XMAX_MIN = 30000
BAR = re.compile(
    r"\[distill\]\s+(\d+)/\d+\s+.*?loss=([0-9.eE+-]+)"
    r"(?:\s+z=([0-9.eE+-]+))?(?:\s+hand=([0-9.eE+-]+))?"
)

PNG = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_v2_full_pv09_e3_20260818_160126_loss.png")
SNAP = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_v2_full_pv09_e3_20260818_160126_loss_overlay.json")
V2_B8_LOG = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_v2_full_pv09_e3_20260818_160126.log")
V2_B32_PTR = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_v2_full_pv09_e3_b32_CURRENT.log")
V2_RESUME_LOG = Path(
    "/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_v2_full_pv09_e3_b32_resume22k_nos2_s0_20260819_023851.log"
)
V2_B8_RESUME_LOG = Path(
    "/mnt/data2/wpy/workspace/Phi_0_wpy/logs/"
    "820mix_v2_full_pv09_e3_b8_resume39k_aligned_20260819_124123.log"
)
V1_LOGS = [
    Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_h16_vdl_clk_pmt_b8_e2_20260817_092620.log"),
    Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs/820mix_h16_resume_pv0.8_20260817_140740.log"),
]
RUNS = (
    ("mix_v1 H16 B8", "#c44e52", None),
    ("mix_v2 H32 B8 (1st)", "#4c72b0", V2_B8_LOG),
    ("mix_v2 H32 B32", "#55a868", V2_B32_PTR),
    ("resume@22k→38k", "#8172b3", V2_RESUME_LOG),
    ("B8 aligned resume@38k", "#2563eb", V2_B8_RESUME_LOG),
)


def ewm(xs, span):
    a = 2.0 / (span + 1.0)
    out, s = [], None
    for x in xs:
        s = x if s is None else (a * x + (1.0 - a) * s)
        out.append(s)
    return out


def parse_log(path: Path) -> dict[int, tuple[float, float | None, float | None]]:
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BAR.search(line)
        if not m:
            continue
        z = float(m.group(3)) if m.group(3) else None
        h = float(m.group(4)) if m.group(4) else None
        d[int(m.group(1))] = (float(m.group(2)), z, h)
    return d


def stitch_v1():
    a = parse_log(V1_LOGS[0])
    b = parse_log(V1_LOGS[1])
    cut = min(b) if b else 10**18
    out = {k: v for k, v in a.items() if k < cut}
    out.update(b)
    return out


def series(d, field):
    steps, raw = [], []
    for s in sorted(k for k in d if k >= WARM):
        v = d[s][field]
        if v is None:
            continue
        steps.append(s)
        raw.append(v)
    return steps, ewm(raw, SPAN), raw


def draw_run(ax, steps, raw, ema, color, label):
    if not steps:
        return
    short = len(steps) < 40
    ax.plot(steps, ema, color=color, lw=2.0, label=label, zorder=3)
    ax.plot(
        steps,
        raw,
        color=color,
        alpha=0.7 if short else 0.28,
        lw=1.2 if short else 0.8,
        ls="--" if short else "-",
        marker="o" if short else None,
        ms=4 if short else 0,
        zorder=4,
    )


def main():
    logs = {
        "mix_v1 H16 B8": stitch_v1(),
        "mix_v2 H32 B8 (1st)": parse_log(V2_B8_LOG),
        "mix_v2 H32 B32": parse_log(V2_B32_PTR.resolve() if V2_B32_PTR.exists() else V2_B32_PTR),
        "resume@22k→38k": parse_log(V2_RESUME_LOG),
        "B8 aligned resume@38k": parse_log(V2_B8_RESUME_LOG),
    }
    last = 0
    packed = {}
    for name, color, _ in RUNS:
        sz, ez, rz = series(logs[name], 1)
        sh, eh, rh = series(logs[name], 2)
        packed[name] = (color, sz, ez, rz, sh, eh, rh)
        if name != "mix_v1 H16 B8" and sz:
            last = max(last, sz[-1])
    if last == 0:
        print("no bars yet", file=sys.stderr)
        return 1
    xmax = max(float(last) * 1.05, float(XMAX_MIN))

    fig, (ax_z, ax_h) = plt.subplots(2, 1, figsize=(10.5, 7.2), dpi=120, sharex=True)
    for name, color, _ in RUNS:
        color, sz, ez, rz, sh, eh, rh = packed[name]
        draw_run(ax_z, sz, rz, ez, color, name)
        draw_run(ax_h, sh, rh, eh, color, name)
    ax_z.set_ylabel("z")
    ax_h.set_ylabel("hand")
    ax_z.set_ylim(0, 0.1)
    ax_h.set_ylim(0, 0.1)
    ax_z.set_xlim(WARM, xmax)
    ax_z.set_title(f"online_vlm  ema span={SPAN} after step {WARM}")
    ax_h.set_xlabel("step")
    ax_z.legend(loc="upper right", fontsize=8)
    ax_h.legend(loc="upper right", fontsize=8)
    ax_z.grid(True, alpha=0.25)
    ax_h.grid(True, alpha=0.25)
    fig.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    def ds(steps, ema, lo, hi, stride=2):
        pts = [(int(s), round(float(y), 4)) for s, y in zip(steps, ema) if lo <= s <= hi]
        return pts[::stride]

    b8 = packed["mix_v2 H32 B8 (1st)"]
    b32 = packed["mix_v2 H32 B32"]
    rs = packed["resume@22k→38k"]
    b8n = packed["B8 aligned resume@38k"]
    snap = {
        "xmax": int(xmax),
        "b8_step": b8[1][-1] if b8[1] else None,
        "b8_z": round(float(b8[2][-1]), 4) if b8[2] else None,
        "b8_hand": round(float(b8[5][-1]), 4) if b8[5] else None,
        "b32_step": b32[1][-1] if b32[1] else None,
        "b32_z": round(float(b32[2][-1]), 4) if b32[2] else None,
        "b32_hand": round(float(b32[5][-1]), 4) if b32[5] else None,
        "resume_step": rs[1][-1] if rs[1] else None,
        "resume_z": round(float(rs[2][-1]), 4) if rs[2] else None,
        "resume_hand": round(float(rs[5][-1]), 4) if rs[5] else None,
        "b8_now_step": b8n[1][-1] if b8n[1] else None,
        "b8_now_z": round(float(b8n[2][-1]), 4) if b8n[2] else None,
        "b8_now_hand": round(float(b8n[5][-1]), 4) if b8n[5] else None,
    }
    SNAP.write_text(json.dumps(snap), encoding="utf-8")
    print(
        f"png={PNG} xmax={int(xmax)} "
        f"b8={snap['b8_step']} z={snap['b8_z']} hand={snap['b8_hand']} "
        f"b32={snap['b32_step']} z={snap['b32_z']} hand={snap['b32_hand']} "
        f"resume={snap['resume_step']} z={snap['resume_z']} hand={snap['resume_hand']} "
        f"b8_now={snap['b8_now_step']} z={snap['b8_now_z']} hand={snap['b8_now_hand']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
