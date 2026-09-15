#!/usr/bin/env python3
"""Compare MuJoCo ATM-PD traj vs Newton gold (z_pred / q_cmd MSE)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_z_q(npz: Path) -> tuple[np.ndarray, np.ndarray | None]:
    d = np.load(npz)
    z = None
    for k in ("z_pred", "z_hat", "token", "z"):
        if k in d.files:
            z = np.asarray(d[k], dtype=np.float32)
            break
    if z is None:
        raise KeyError(f"no z_* in {npz}: {d.files}")
    if z.ndim == 3:
        z = z[:, 0]
    q = None
    for k in ("q_cmd", "q_pred", "q_hat", "joint_pos_target"):
        if k in d.files:
            q = np.asarray(d[k], dtype=np.float32)
            break
    if q is not None and q.ndim == 3:
        q = q[:, 0]
    return z, q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--atm", type=Path, required=True)
    p.add_argument("--newton", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    z_a, q_a = _load_z_q(args.atm)
    z_n, q_n = _load_z_q(args.newton)
    n = min(len(z_a), len(z_n))
    z_a, z_n = z_a[:n], z_n[:n]
    report = {
        "T": int(n),
        "z_mse": float(np.mean((z_a - z_n) ** 2)),
        "z_cos_mean": float(
            np.mean(
                [
                    float(
                        np.dot(z_a[t], z_n[t])
                        / (np.linalg.norm(z_a[t]) * np.linalg.norm(z_n[t]) + 1e-8)
                    )
                    for t in range(n)
                ]
            )
        ),
    }
    if q_a is not None and q_n is not None:
        m = min(len(q_a), len(q_n), n)
        report["q_mse"] = float(np.mean((q_a[:m] - q_n[:m]) ** 2))
        report["q_mae"] = float(np.mean(np.abs(q_a[:m] - q_n[:m])))
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **{k: np.asarray(v) for k, v in report.items()})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
