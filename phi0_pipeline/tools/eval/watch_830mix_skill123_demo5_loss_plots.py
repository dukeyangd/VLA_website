#!/usr/bin/env python3
"""Refresh 830mix skill123+demo5 loss plots every N seconds until train exits.

Writes (overwrite in place so IDE can refresh):
  logs/830mix_skill123_demo5_loss_0_0p1_20260907.png
  logs/830mix_skill123_demo5_loss_z_hand_20260907.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E3 = Path(
    "/mnt/data3/wpy/830mix_skill123_demo5_handcmd_lag1_vlm_cache_h32_b32_ddp8_e3_pv0.95_20260907_011524"
)
B64 = Path(
    "/mnt/data3/wpy/830mix_skill123_demo5_handcmd_lag1_vlm_cache_h32_b64_ddp8_e2_pv0.95_resume_20260907_035710"
)
RESUME_AT = 38883
LOG_DIR = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs")
OUT_LOSS = LOG_DIR / "830mix_skill123_demo5_loss_0_0p1_20260907.png"
OUT_ZH = LOG_DIR / "830mix_skill123_demo5_loss_z_hand_20260907.png"
TRAIN_MARK = "newton_boneseed_distill_fabric.*035710"


def _load(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps, loss, z, hand = [], [], [], []
    if not p.is_file():
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    with p.open() as f:
        for line in f:
            if '"loss"' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "loss" not in r or "chunk" not in r:
                continue
            steps.append(int(r["chunk"]))
            loss.append(float(r["loss"]))
            z.append(float(r["loss_z"]) if "loss_z" in r else np.nan)
            hand.append(float(r["loss_hand"]) if "loss_hand" in r else np.nan)
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(loss, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        np.asarray(hand, dtype=np.float64),
    )


# ponytail: heavier smooth for dense step logs; bump via --ema-span
EMA_SPAN_DEFAULT = 400


def _ewm(x: np.ndarray, span: int = EMA_SPAN_DEFAULT) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    s = None
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = np.nan if s is None else s
            continue
        s = v if s is None else a * v + (1.0 - a) * s
        out[i] = s
    return out


def _merge(parts):
    d: dict[int, tuple[float, float, float]] = {}
    for s, l, z, h in parts:
        for i in range(len(s)):
            d[int(s[i])] = (float(l[i]), float(z[i]), float(h[i]))
    steps = np.array(sorted(d), dtype=np.int64)
    loss = np.array([d[i][0] for i in steps], dtype=np.float64)
    zz = np.array([d[i][1] for i in steps], dtype=np.float64)
    hh = np.array([d[i][2] for i in steps], dtype=np.float64)
    return steps, loss, zz, hh


def _train_alive() -> bool:
    # ponytail: pgrep via /proc scan; ceiling = miss renamed argv
    needle = "035710"
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "ignore"
            )
        except OSError:
            continue
        if "newton_boneseed_distill_fabric" in cmd and needle in cmd:
            return True
    return False


def render(*, ema_span: int) -> tuple[int, int]:
    s1, l1, z1, h1 = _load(E3 / "distill_metrics.jsonl")
    s2, l2, z2, h2 = _load(B64 / "distill_metrics.jsonl")
    m2 = s2 > RESUME_AT
    steps, loss, zz, hh = _merge(
        [(s1, l1, z1, h1), (s2[m2], l2[m2], z2[m2], h2[m2])]
    )
    if steps.size == 0:
        return 0, -1

    # total loss 0-0.1
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(steps, loss, color="#111827", alpha=0.12, lw=0.5)
    ax.plot(
        steps,
        _ewm(loss, span=ema_span),
        color="#111827",
        lw=1.8,
        label=f"loss (ewm{ema_span})",
    )
    ax.axvline(RESUME_AT, color="#6b7280", ls="--", lw=1.0, label="B32→B64 resume")
    ax.set_ylim(0.0, 0.1)
    ax.set_xlabel("step (from 0)")
    ax.set_ylabel("loss")
    ax.set_title(
        f"830mix skill123+demo5 total loss (ylim 0–0.1)  last={int(steps[-1])}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    tmp = OUT_LOSS.with_name(OUT_LOSS.stem + ".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    tmp.replace(OUT_LOSS)
    (B64 / "loss_0_0p1.png").write_bytes(OUT_LOSS.read_bytes())

    # z + hand
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, y, name, color in (
        (axes[0], zz, "loss_z", "#2563eb"),
        (axes[1], hh, "loss_hand", "#dc2626"),
    ):
        ax.plot(steps, y, color=color, alpha=0.12, lw=0.5)
        ax.plot(
            steps,
            _ewm(y, span=ema_span),
            color=color,
            lw=1.8,
            label=f"{name} (ewm{ema_span})",
        )
        ax.axvline(RESUME_AT, color="#6b7280", ls="--", lw=1.0, label="B32→B64 resume")
        ax.set_ylabel(name)
        ax.set_ylim(0.0, 0.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step (from 0)")
    axes[0].set_title(
        f"830mix skill123+demo5 loss_z / loss_hand (ylim 0–0.1)  last={int(steps[-1])}"
    )
    fig.tight_layout()
    tmp = OUT_ZH.with_name(OUT_ZH.stem + ".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    tmp.replace(OUT_ZH)
    (B64 / "loss_z_hand.png").write_bytes(OUT_ZH.read_bytes())
    return int(steps.size), int(steps[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--ema-span", type=int, default=EMA_SPAN_DEFAULT)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        n, last = render(ema_span=int(args.ema_span))
        alive = _train_alive()
        print(
            f"[watch] n={n} last_step={last} ema={args.ema_span} "
            f"train_alive={int(alive)} loss={OUT_LOSS} zh={OUT_ZH}",
            flush=True,
        )
        if args.once or not alive:
            break
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
