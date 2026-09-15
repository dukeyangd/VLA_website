#!/usr/bin/env python3
"""Export dataset GT body/IMU streams for decoder LowState overlay.

C++ deploy builds his_* from LowState (not from VLA proprio). Overlaying these
fields makes joint/IMU/gravity history come from parquet GT while token still
arrives via ZMQ. last_actions stay previous policy outputs (not sim-derived).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from phi0.schema.unified_action_schema import SLICES


def _stack_col(table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_numpy()).astype(np.float32, copy=False)


def _quat_angvel(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity from successive wxyz quats (rad/s)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    n = len(q)
    out = np.zeros((n, 3), dtype=np.float32)
    if n < 2:
        return out
    dt = 1.0 / float(fps)
    rots = R.from_quat(q[:, [1, 2, 3, 0]])  # scipy xyzw
    for t in range(1, n):
        r_rel = rots[t - 1].inv() * rots[t]
        out[t] = (r_rel.as_rotvec() / dt).astype(np.float32)
    out[0] = out[1]
    return out


def export_gt_decoder_lowstate(
    parquet: Path,
    out: Path,
    *,
    episode_row: int | None = None,
    fps: float = 50.0,
) -> Path:
    parquet = parquet.resolve()
    table = pq.read_table(parquet, columns=["action.unified", "episode_index"])
    ua = _stack_col(table, "action.unified")
    if episode_row is not None:
        ep = np.asarray(table.column("episode_index").to_numpy()).reshape(-1)
        rows = np.where(ep == int(episode_row))[0]
        if not rows.size:
            raise ValueError(f"episode_index={episode_row} not in {parquet}")
        ua = ua[rows]
    qs, qe = SLICES["g1_body_qpos_36"]
    q36 = ua[:, qs:qe]
    body_q = q36[:, 7:36].astype(np.float32)
    root_xyz = q36[:, 0:3].astype(np.float32)
    root_quat = q36[:, 3:7].astype(np.float32)
    # finite-diff joint vel (MuJoCo / teleop abs order)
    body_dq = np.zeros_like(body_q)
    if len(body_q) > 1:
        body_dq[1:] = (body_q[1:] - body_q[:-1]) * float(fps)
        body_dq[0] = body_dq[1]
    ang_vel = _quat_angvel(root_quat, fps)
    gs, ge = SLICES["projected_gravity_xyz"]
    gravity = ua[:, gs:ge].astype(np.float32)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        body_q=body_q,
        body_dq=body_dq,
        root_xyz=root_xyz,
        root_quat_wxyz=root_quat,
        base_ang_vel=ang_vel,
        projected_gravity_xyz=gravity,
        fps=np.asarray([fps], dtype=np.float32),
        note=np.asarray(
            [
                "LowState overlay for decoder his_*; abs dof29+quat from unified; "
                "dq/angvel finite-diff; last_actions still policy"
            ]
        ),
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episode-index", type=int, default=None)
    p.add_argument("--fps", type=float, default=50.0)
    args = p.parse_args()
    out = export_gt_decoder_lowstate(
        args.parquet, args.out, episode_row=args.episode_index, fps=args.fps
    )
    z = np.load(out)
    print(
        f"[ok] {out} T={z['body_q'].shape[0]} "
        f"dof0={float(z['body_q'][0, 0]):+.3f} "
        f"quat={z['root_quat_wxyz'][0]} "
        f"g0={z['projected_gravity_xyz'][0]}"
    )


if __name__ == "__main__":
    main()
