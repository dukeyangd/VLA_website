#!/usr/bin/env python3
"""Egypt clip root layout: SMPL front [0:315) + GMR root7 (Δxy + abs z + abs quat) + dof29.

- ``[0:9]``: SMPL relative (do not overwrite with GMR)
- ``[360:367]``: GMR ``(Δx,Δy,z_abs)`` + **absolute** quat; no meta init
- ``[367:396]``: GMR dof29
- ``[396:460]``: sonic token slot; disk zero
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from phi0.schema.g1_relroot import (
    gmr_abs_to_relative_root7,
    integrate_g1_relative_root7,
)

G1_ROOT = slice(360, 367)
G1_DOF = slice(367, 396)
SONIC_TOKEN = slice(396, 460)


def apply_gmr_qpos_tail(
    unified: np.ndarray,
    *,
    q36_abs: np.ndarray,
    dim_mask: np.ndarray | None = None,
    check_roundtrip: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Write GMR ``[360:367]`` (Δxy+abs z+abs quat) + dof ``[367:396]``; clear sonic.

    Returns ``(unified, dim_mask, init_xyz, init_quat)``.
    ``init_*`` are diagnostic only (frame0); not written to meta.
    """
    u = np.asarray(unified, dtype=np.float32).copy()
    q = np.asarray(q36_abs, dtype=np.float32).reshape(-1, 36)
    if len(u) != len(q):
        raise ValueError(f"len unified {len(u)} != q36 {len(q)}")
    rel = gmr_abs_to_relative_root7(q[:, 0:3], q[:, 3:7])
    init_xyz = np.zeros(3, dtype=np.float32)
    init_quat = q[0, 3:7].copy()
    if check_roundtrip:
        pos_i, quat_i = integrate_g1_relative_root7(rel)
        if not np.allclose(pos_i, q[:, 0:3], atol=2e-4):
            err = float(np.abs(pos_i - q[:, 0:3]).max())
            raise AssertionError(f"relative root xyz roundtrip failed maxerr={err}")
        dots = np.abs(np.sum(quat_i * q[:, 3:7], axis=1))
        if float(dots.min()) < 1.0 - 5e-3:
            raise AssertionError(f"absolute root quat roundtrip failed min|dot|={dots.min()}")

    u[:, G1_ROOT] = rel
    u[:, G1_DOF] = q[:, 7:36]
    u[:, SONIC_TOKEN] = 0.0

    dm = None
    if dim_mask is not None:
        dm = np.asarray(dim_mask, dtype=np.bool_).copy()
        dm[:, G1_ROOT] = True
        dm[:, G1_DOF] = True
        dm[:, SONIC_TOKEN] = False
    return u, dm, init_xyz, init_quat


def assert_root_layout_consistent(unified: np.ndarray, *, atol: float = 2e-3) -> None:
    """Check GMR root7 + sonic disk zero; does not couple to SMPL [0:9]."""
    u = np.asarray(unified, dtype=np.float32)
    rel = u[:, G1_ROOT]
    if not np.isfinite(rel).all():
        raise AssertionError("[360:367] non-finite")
    qn = np.linalg.norm(rel[:, 3:7], axis=1)
    if float(np.abs(qn - 1.0).max()) > 5e-3:
        raise AssertionError(f"[363:367] quat not unit (max |n-1|={np.abs(qn-1).max()})")
    pos, quat = integrate_g1_relative_root7(rel)
    rel2 = gmr_abs_to_relative_root7(pos, quat)
    if not np.allclose(rel2, rel, atol=atol):
        err = float(np.abs(rel2 - rel).max())
        raise AssertionError(f"rel root not stable under integrate/encode maxerr={err}")
    if float(np.abs(u[:, SONIC_TOKEN]).max()) > 1e-8:
        raise AssertionError("[396:460] disk must be zero (sonic GT is online-only)")


def apply_relative_root_layout(*_a, **_k):  # noqa: ANN001
    raise RuntimeError(
        "apply_relative_root_layout removed: it overwrote SMPL [0:9] with GMR. "
        "Use apply_gmr_qpos_tail + write_smpl_semantic_from_pose_aa."
    )


if __name__ == "__main__":
    n = 8
    pos = np.zeros((n, 3), np.float64)
    pos[:, 0] = np.linspace(0, 0.14, n)
    pos[:, 2] = 0.78
    yaw = np.linspace(0.0, 0.55, n)
    quat = R.from_euler("z", yaw.reshape(-1, 1)).as_quat(scalar_first=True).astype(np.float32)
    q36 = np.zeros((n, 36), np.float32)
    q36[:, 0:3] = pos
    q36[:, 3:7] = quat
    q36[:, 7:] = 0.01
    u = np.zeros((n, 512), np.float32)
    u[:, 0:9] = 0.5
    u[:, 396:460] = 0.5
    out, dm, init_xyz, init_q = apply_gmr_qpos_tail(
        u, q36_abs=q36, dim_mask=np.ones((n, 512), dtype=bool)
    )
    assert np.allclose(out[:, 0:9], 0.5)
    assert np.allclose(init_xyz, 0.0)
    assert abs(float(out[0, 362]) - 0.78) < 1e-5
    assert float(np.abs(np.sum(out[0, 363:367] * init_q))) > 0.99
    assert_root_layout_consistent(out)
    pos2, quat2 = integrate_g1_relative_root7(out[:, 360:367])
    assert np.allclose(pos2, pos, atol=1e-4)
    assert dm is not None and not bool(dm[:, 396:460].any())
    print("egypt_clip_root_layout: ok")
