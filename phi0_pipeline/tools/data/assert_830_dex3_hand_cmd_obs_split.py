#!/usr/bin/env python3
"""Assert 830 Dex3 unified: [346:360]=cmd, [475:489]=measured, and they differ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phi0.data.wbc43_io import hands_cmd_14, hands_obs_14  # noqa: E402
from phi0.deploy.robot_proprio import deploy_hand7_to_wbc  # noqa: E402


def _state43(row: pd.Series) -> np.ndarray:
    s = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
    return s[:43]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", type=Path, required=True)
    ap.add_argument("--raw-ep", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    df_u = pd.read_parquet(args.unified)
    df_r = pd.read_parquet(args.raw_ep)
    n = min(len(df_u), len(df_r), 200)
    u = np.stack(
        [np.asarray(df_u.iloc[i]["action.unified"], dtype=np.float32) for i in range(n)]
    )
    cmd_u = u[:, 346:360]
    obs_u = u[:, 475:489]
    cmds = []
    obss = []
    for i in range(n):
        row = df_r.iloc[i]
        cmds.append(hands_cmd_14(row))
        obss.append(hands_obs_14(row, _state43(row)))
    cmd = np.stack(cmds)
    obs = np.stack(obss)
    assert np.allclose(cmd_u, cmd, atol=args.tol), "unified[346:360] != action.dex3→WBC"
    assert np.allclose(obs_u, obs, atol=args.tol), "unified[475:489] != obs.dex3→WBC"
    mse = float(np.mean((cmd_u - obs_u) ** 2))
    assert mse > 1e-6, f"cmd≈obs (mse={mse}); split failed"
    # sanity: obs matches actuator→WBC of observation.dex3
    act_l = np.stack(
        [deploy_hand7_to_wbc(np.asarray(df_r.iloc[i]["observation.dex3.left.position"])) for i in range(n)]
    )
    assert np.allclose(obs_u[:, :7], act_l, atol=args.tol)
    print(
        f"OK n={n} mse(cmd,obs)={mse:.6g} "
        f"|cmd|max={float(np.abs(cmd_u).max()):.4g} "
        f"|obs|max={float(np.abs(obs_u).max()):.4g}"
    )


if __name__ == "__main__":
    main()
