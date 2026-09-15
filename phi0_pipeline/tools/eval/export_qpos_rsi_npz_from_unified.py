#!/usr/bin/env python3
"""Export frame0 qpos RSI npz from unified parquet (810demo / pick-tissue)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from phi0.schema.g1_relroot import abs_q36_from_relroot_unified, looks_like_relative_root7
from phi0.schema.unified_action_schema import SLICES


def _stack_col(table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_numpy()).astype(np.float32, copy=False)


def export_qpos_rsi_npz(
    parquet: Path,
    out: Path,
    *,
    frame: int = 0,
    episode_row: int | None = None,
) -> Path:
    parquet = parquet.resolve()
    table = pq.read_table(parquet, columns=["action.unified", "timestamp", "episode_index"])
    if episode_row is not None:
        ep = np.asarray(table.column("episode_index").to_numpy()).reshape(-1)
        rows = np.where(ep == int(episode_row))[0]
        if not rows.size:
            raise ValueError(f"episode_index={episode_row} not in {parquet}")
        fi = int(rows[0]) + int(frame)
    else:
        fi = int(frame)
    if fi < 0 or fi >= table.num_rows:
        raise IndexError(f"frame {fi} out of range [0, {table.num_rows})")

    ua = _stack_col(table, "action.unified")
    u = ua[fi:fi + 1]
    qs, qe = SLICES["g1_body_qpos_36"]
    q36 = u[:, qs:qe]
    if looks_like_relative_root7(u[:, 360:367]):
        q36 = abs_q36_from_relroot_unified(u)
    root = np.asarray(q36[0, 0:3], dtype=np.float32)
    quat = np.asarray(q36[0, 3:7], dtype=np.float32)
    dof = np.asarray(q36[0, 7:36], dtype=np.float32)
    ts = np.asarray(table.column("timestamp")[fi].as_py(), dtype=np.float32)
    np.savez(
        out,
        observation_qpos=dof,
        reference_root_xyz=root,
        reference_root_quat_wxyz=quat,
        timestamp=np.asarray([ts], dtype=np.float32),
        dataset_fps=np.asarray([50.0], dtype=np.float32),
        frame_index=np.asarray([fi], dtype=np.int64),
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--episode-index", type=int, default=None)
    args = p.parse_args()
    out = export_qpos_rsi_npz(
        args.parquet,
        args.out,
        frame=args.frame,
        episode_row=args.episode_index,
    )
    z = np.load(out)
    print(
        f"[ok] {out} frame={int(z['frame_index'][0])} "
        f"root_z={float(z['reference_root_xyz'][2]):.3f} "
        f"dof0={float(z['observation_qpos'][0]):+.3f}"
    )


if __name__ == "__main__":
    main()
