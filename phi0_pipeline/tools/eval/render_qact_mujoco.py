#!/usr/bin/env python3
"""FK-render Isaac ``qpos_student`` infer traj through MuJoCo (no physics re-sim).

Isaac native ``RECORD_VIDEO=1`` is currently broken in this env (``usdrt.hierarchy``);
this replaces it by replaying logged ``q_act``/``root_act`` (or ``q_cmd``) with
``mj_forward`` only. IsaacLab dof29 → MuJoCo via ``isaaclab_to_mujoco_dof``.

Usage:
  python tools/eval/render_qact_mujoco.py INFER_NPZ --out out.mp4
  python tools/eval/render_qact_mujoco.py INFER_NPZ --out out.mp4 --source q_cmd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_GROOT = Path(__file__).resolve().parents[2] / "subpackages"
_DEFAULT_MJCF = _GROOT / "gear_sonic_deploy/g1/g1_29dof.xml"


def render(
    infer_npz: Path,
    out_mp4: Path,
    *,
    source: str = "q_act",
    env_index: int = 0,
    fps: float = 50.0,
    mjcf: Path = _DEFAULT_MJCF,
) -> Path:
    sys.path.insert(0, str(_GROOT))
    from gear_sonic.utils.zmq_sim_diagnostics import isaaclab_to_mujoco_dof

    d = np.load(infer_npz)
    if source not in d.files:
        raise ValueError(f"{infer_npz} missing '{source}'; has {d.files}")
    if "root_act" not in d.files:
        raise ValueError(f"{infer_npz} missing 'root_act'")

    dof = np.asarray(d[source], dtype=np.float64)
    root = np.asarray(d["root_act"], dtype=np.float64)
    if dof.ndim == 3:
        dof = dof[:, int(env_index), :]
    if root.ndim == 3:
        root = root[:, int(env_index), :]
    if dof.shape[-1] != 29 or root.shape[-1] != 7:
        raise ValueError(f"expected dof[...,29] root[...,7], got {dof.shape} {root.shape}")

    t = int(dof.shape[0])
    q_mj = isaaclab_to_mujoco_dof(dof)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.azimuth = 90.0

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w = 480, 640
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for i in range(t):
        q = model.qpos0.copy()
        q[0:3] = root[i, 0:3]
        q[3:7] = root[i, 3:7]
        q[7:36] = q_mj[i]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [float(root[i, 0]), float(root[i, 1]), 0.8]
        renderer.update_scene(data, camera=cam)
        proc.stdin.write(renderer.render().tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out_mp4}")
    print(f"[ok] {out_mp4} frames={t} source={source} fps={fps}")
    return out_mp4


def _self_check() -> None:
    # ponytail: dof remap length + mjcf exists
    assert _DEFAULT_MJCF.is_file(), _DEFAULT_MJCF
    sys.path.insert(0, str(_GROOT))
    from gear_sonic.utils.zmq_sim_diagnostics import isaaclab_to_mujoco_dof

    x = np.zeros((2, 29), dtype=np.float64)
    y = isaaclab_to_mujoco_dof(x)
    assert y.shape == (2, 29)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", choices=("q_act", "q_cmd", "q_star"), default="q_act")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF)
    args = ap.parse_args()
    render(
        args.infer_npz,
        args.out,
        source=args.source,
        env_index=args.env,
        fps=args.fps,
        mjcf=args.mjcf,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        print("[ok] self-check")
    else:
        main()
