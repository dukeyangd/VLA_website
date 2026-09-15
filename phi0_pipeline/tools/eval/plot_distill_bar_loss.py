#!/usr/bin/env python3
"""Parse ``[distill] step/N ... loss=`` bars and overlay curves."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

_BAR = re.compile(
    r"\[distill\]\s+(\d+)/\d+\s+.*?loss=([0-9.eE+-]+)"
    r"(?:\s+z=([0-9.eE+-]+))?(?:\s+hand=([0-9.eE+-]+))?"
)


def parse_log(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    steps: list[int] = []
    loss: list[float] = []
    z: list[float] = []
    hand: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _BAR.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        z.append(float(m.group(3)) if m.group(3) else float("nan"))
        hand.append(float(m.group(4)) if m.group(4) else float("nan"))
    return steps, loss, z, hand


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("series", nargs="+", help="label=logfile")
    args = p.parse_args()
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for spec in args.series:
        label, _, raw = spec.partition("=")
        path = Path(raw)
        s, lo, zz, hh = parse_log(path)
        if not s:
            raise SystemExit(f"no distill bars in {path}")
        ax[0].plot(s, lo, label=label, lw=1.4)
        ax[1].plot(s, zz, label=label, lw=1.4)
        ax[2].plot(s, hh, label=label, lw=1.4)
    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("z")
    ax[2].set_ylabel("hand")
    ax[2].set_xlabel("step")
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(loc="upper right")
        a.set_yscale("log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
