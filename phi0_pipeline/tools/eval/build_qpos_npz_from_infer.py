#!/usr/bin/env python3
"""Pack student ``q_cmd`` (IsaacLab dof29) into GT ``observation_qpos`` tokens.npz.

Same schema as ``replay_gt_qpos.sh`` output, so ``replay_gt_qpos_loop.sh`` can
validate model joints on the trusted ZMQ v1 deploy+MuJoCo path.

``q_cmd`` is remapped IsaacLab→MuJoCo (dataset ``observation.qpos[0:29]`` is
MuJoCo order). Root uses GT ``root_action`` when ``--gt-qpos-npz`` is given
(so the video judges joint articulation on the same root path as the good GT
clip); otherwise falls back to zeros + standing height via the loop defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_GROOT = Path(__file__).resolve().parents[2] / "subpackages"


def _root_xyz_quat_from_infer(d: np.lib.npyio.NpzFile, *, env_index: int, t: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Newton/Isaac infer dumps ``root_act`` as absolute [T,B,7]=xyz+quat_wxyz."""
    key = "root_act" if "root_act" in d.files else ("root_action" if "root_action" in d.files else None)
    if key is None:
        return None
    root = np.asarray(d[key], dtype=np.float32)
    if root.ndim == 3:
        root = root[:, int(env_index), :]
    if root.ndim != 2 or root.shape[-1] < 7:
        raise ValueError(f"expected root [T,7+], got {root.shape}")
    n = min(t, int(root.shape[0]))
    xyz = root[:n, :3].copy()
    quat = root[:n, 3:7].copy()
    nrm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat /= np.maximum(nrm, 1e-8)
    return xyz, quat


def build(
    infer_npz: Path,
    *,
    source: str = "q_cmd",
    env_index: int = 0,
    gt_qpos_npz: Path | None = None,
    root_from_infer: bool = False,
    fps: float = 50.0,
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(_GROOT))
    from gear_sonic.utils.zmq_sim_diagnostics import isaaclab_to_mujoco_dof

    d = np.load(infer_npz)
    if source not in d.files:
        raise ValueError(f"{infer_npz} missing '{source}'; has {d.files}")
    dof_il = np.asarray(d[source], dtype=np.float64)
    if dof_il.ndim == 3:
        dof_il = dof_il[:, int(env_index), :]
    if dof_il.shape[-1] != 29:
        raise ValueError(f"expected [T,29], got {dof_il.shape}")
    t = int(dof_il.shape[0])
    dof_mj = isaaclab_to_mujoco_dof(dof_il).astype(np.float32)
    obs = np.zeros((t, 43), dtype=np.float32)
    obs[:, :29] = dof_mj

    payload: dict[str, np.ndarray] = {
        "observation_qpos": obs,
        "timestamp": (np.arange(t, dtype=np.float32) / float(fps)),
        "dataset_fps": np.array([fps], dtype=np.float32),
        "ego_video_offset": np.array([0], dtype=np.int64),
    }
    if root_from_infer:
        rq = _root_xyz_quat_from_infer(d, env_index=env_index, t=t)
        if rq is None:
            raise ValueError(f"{infer_npz} missing root_act/root_action for --root-from-infer")
        xyz, quat = rq
        n = int(xyz.shape[0])
        payload["reference_root_xyz"] = xyz
        payload["reference_root_quat_wxyz"] = quat
        payload["observation_qpos"] = obs[:n]
        payload["timestamp"] = payload["timestamp"][:n]
        print(f"[info] root from infer T={n} xyz0={xyz[0].tolist()} xyzT={xyz[-1].tolist()}")
    elif gt_qpos_npz is not None:
        g = np.load(gt_qpos_npz)
        n = t
        if "root_action" in g.files:
            ra = np.asarray(g["root_action"], dtype=np.float32)
            n = min(t, len(ra))
            payload["root_action"] = ra[:n]
            payload["observation_qpos"] = obs[:n]
            payload["timestamp"] = payload["timestamp"][:n]
        # Direct root preferred when present (qpos36-style packs).
        if "reference_root_xyz" in g.files and "reference_root_quat_wxyz" in g.files:
            xyz = np.asarray(g["reference_root_xyz"], dtype=np.float32)
            quat = np.asarray(g["reference_root_quat_wxyz"], dtype=np.float32)
            n = min(t, len(xyz), len(quat))
            payload["reference_root_xyz"] = xyz[:n]
            payload["reference_root_quat_wxyz"] = quat[:n]
            payload["observation_qpos"] = obs[:n]
            payload["timestamp"] = payload["timestamp"][:n]
            payload.pop("root_action", None)
        # ponytail: optional GT joint ref for quick MSE print
        if "observation_qpos" in g.files:
            gt = np.asarray(g["observation_qpos"], dtype=np.float32)[:n, :29]
            mse = float(np.mean((obs[:n, :29] - gt) ** 2))
            print(f"[info] dof29 MSE vs GT obs (mujoco order)={mse:.5f}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", choices=("q_cmd", "q_act", "q_star"), default="q_cmd")
    ap.add_argument("--gt-qpos-npz", type=Path, default=None, help="borrow root_action from GT clip")
    ap.add_argument(
        "--root-from-infer",
        action="store_true",
        help="use infer root_act xyz+quat_wxyz as reference_root_* (Newton ATM path)",
    )
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()
    payload = build(
        args.infer_npz,
        source=args.source,
        env_index=args.env,
        gt_qpos_npz=args.gt_qpos_npz,
        root_from_infer=bool(args.root_from_infer),
        fps=args.fps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"[ok] {args.out} frames={len(payload['observation_qpos'])} source={args.source}")


if __name__ == "__main__":
    main()
