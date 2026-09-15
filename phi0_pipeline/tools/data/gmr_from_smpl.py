#!/usr/bin/env python3
"""Shared SMPL(Y-up) → GMR G1 q36 (egypt / BoneSEED same path).

Do **not** apply Sonic ``remove_smpl_base_rot``. Default height: ``None``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# G1 29dof (MuJoCo hinge order after freejoint)
WAIST_YAW = 12
L_HIP_YAW = 2
R_HIP_YAW = 8
WAIST_LIM = 1.57


def smpl_yup_to_zup(pose_aa: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Y-up SMPL → Z-up for GMR: Rx(90°) on global_orient + transl (x,y,z)→(x,-z,y)."""
    pose = np.asarray(pose_aa, dtype=np.float64).reshape(-1, 24, 3).copy()
    transl = np.asarray(transl, dtype=np.float64).reshape(-1, 3)
    rx = R.from_euler("x", 90.0, degrees=True)
    pose[:, 0] = (rx * R.from_rotvec(pose[:, 0])).as_rotvec()
    t_z = np.stack([transl[:, 0], -transl[:, 2], transl[:, 1]], axis=1)
    return pose.reshape(-1, 72).astype(np.float32), t_z.astype(np.float32)


def smpl_yaw_zup_rad(pose_aa_yup: np.ndarray) -> np.ndarray:
    """Horizontal heading (rad, unwrapped) of SMPL GO after Y-up→Z-up Rx90."""
    go = np.asarray(pose_aa_yup, dtype=np.float64).reshape(-1, 72)[:, :3]
    rx = R.from_euler("x", 90.0, degrees=True)
    yaw = (rx * R.from_rotvec(go)).as_euler("xyz")[:, 2]
    return np.unwrap(yaw)


def align_root_yaw_to_smpl(
    q36: np.ndarray,
    pose_aa_yup: np.ndarray,
    *,
    compensate_hip_yaw: bool = True,
) -> tuple[np.ndarray, dict]:
    """DEPRECATED post-hoc yaw snap (twists waist). Prefer lock_root in GMR."""
    q = np.asarray(q36, dtype=np.float32).copy()
    smpl = smpl_yaw_zup_rad(pose_aa_yup)
    root = np.unwrap(R.from_quat(q[:, 3:7], scalar_first=True).as_euler("xyz")[:, 2])
    waist = q[:, 7 + WAIST_YAW].astype(np.float64)
    eff = root + waist
    d_root = smpl - root
    waist_new = np.clip(eff - smpl, -WAIST_LIM, WAIST_LIM)
    q[:, 3:7] = R.from_euler("z", smpl).as_quat(scalar_first=True).astype(np.float32)
    q[:, 7 + WAIST_YAW] = waist_new.astype(np.float32)
    if compensate_hip_yaw:
        q[:, 7 + L_HIP_YAW] = (q[:, 7 + L_HIP_YAW] - d_root).astype(np.float32)
        q[:, 7 + R_HIP_YAW] = (q[:, 7 + R_HIP_YAW] - d_root).astype(np.float32)
    root2 = np.unwrap(R.from_quat(q[:, 3:7], scalar_first=True).as_euler("xyz")[:, 2])
    err = np.degrees(root2 - smpl)
    stats = {
        "root_vs_smpl_deg_max": float(np.max(np.abs(err))),
        "root_vs_smpl_deg_std": float(err.std()),
        "waist_before_deg_std": float(np.degrees(waist).std()),
        "waist_after_deg_std": float(np.degrees(waist_new).std()),
        "d_root_deg_std": float(np.degrees(d_root).std()),
    }
    if stats["root_vs_smpl_deg_max"] > 1e-3:
        raise AssertionError(f"align failed: {stats}")
    return q, stats


def _pin_root(configuration, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> None:
    q = configuration.data.qpos.copy()
    q[0:3] = np.asarray(root_pos, dtype=np.float64).reshape(3)
    q[3:7] = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4)
    configuration.update(q)


def retarget_with_root_locked(gmr, human_data, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """GMR IK with freejoint pinned (nv[0:6]=0 each step)."""
    import mink

    gmr.update_targets(human_data, offset_to_ground=False)
    _pin_root(gmr.configuration, root_pos, root_quat_wxyz)

    def _solve(tasks, error_fn) -> None:
        dt = gmr.configuration.model.opt.timestep
        curr = float(error_fn())
        for _ in range(int(gmr.max_iter)):
            vel = mink.solve_ik(
                gmr.configuration, tasks, dt, gmr.solver, gmr.damping, gmr.ik_limits
            )
            vel = np.asarray(vel, dtype=np.float64).copy()
            vel[:6] = 0.0
            gmr.configuration.integrate_inplace(vel, dt)
            _pin_root(gmr.configuration, root_pos, root_quat_wxyz)
            nxt = float(error_fn())
            if curr - nxt <= 0.001:
                break
            curr = nxt

    if gmr.use_ik_match_table1:
        _solve(gmr.tasks1, gmr.error1)
    if gmr.use_ik_match_table2:
        _solve(gmr.tasks2, gmr.error2)
    return gmr.configuration.data.qpos.copy().astype(np.float32)


def gmr_qpos_from_smpl(
    pose_aa_z: np.ndarray,
    transl_z: np.ndarray,
    *,
    lock_root: bool = False,
    progress_every: int = 0,
) -> np.ndarray:
    """Retarget Z-up SMPL → G1 q36. ``lock_root`` pins GMR scaled pelvis freejoint."""
    from phi0.deploy.gmr_retarget import (  # lazy: heavy
        GmrRetargetSession,
        unified_to_gmr_human_data,
    )
    from phi0.schema.unified_action_schema import write_smpl_semantic_from_pose_aa, zeros_unified

    sess = GmrRetargetSession()
    gmr = sess._retarget
    n = len(pose_aa_z)
    out = np.zeros((n, 36), dtype=np.float32)
    u = zeros_unified()
    for i in range(n):
        write_smpl_semantic_from_pose_aa(u, pose_aa_z[i], np.zeros(3, dtype=np.float32))
        if lock_root:
            human = unified_to_gmr_human_data(u, state_root_trans_world=transl_z[i])
            gmr.update_targets(human, offset_to_ground=False)
            ppos, pquat = gmr.scaled_human_data["pelvis"]
            out[i] = retarget_with_root_locked(gmr, human, ppos, pquat)
        else:
            out[i] = sess.retarget(u, state_root_trans_world=transl_z[i]).astype(np.float32)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[gmr] {i + 1}/{n} lock_root={lock_root}", flush=True)
    return out


def postprocess_gmr_root_xy0_z_match(q36: np.ndarray, *, proto_z0: float) -> np.ndarray:
    """egypt post: xy -= xy[0]; z += proto_z0 - gmr_z0 (scalar floor only)."""
    q = np.asarray(q36, dtype=np.float32).copy()
    q[:, 0] -= q[0, 0]
    q[:, 1] -= q[0, 1]
    q[:, 2] += float(proto_z0 - q[0, 2])
    return q


def absolute_root_to_vln09(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of integrate_vln (yaw-only); last-frame Δ=0."""
    from phi0.schema.unified_action_schema import quat_wxyz_to_rot6d

    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    yaws = np.unwrap(R.from_quat(quat, scalar_first=True).as_euler("xyz")[:, 2])
    n = len(pos)
    out = np.zeros((n, 9), dtype=np.float32)
    for t in range(n - 1):
        d = pos[t + 1] - pos[t]
        c, s = np.cos(yaws[t]), np.sin(yaws[t])
        out[t, 0] = np.float32(c * d[0] + s * d[1])
        out[t, 1] = np.float32(-s * d[0] + c * d[1])
        out[t, 2] = np.float32(d[2])
        q_next = R.from_euler("z", yaws[t + 1]).as_quat(scalar_first=True).astype(np.float32)
        out[t, 3:9] = quat_wxyz_to_rot6d(q_next)
    q_last = R.from_euler("z", yaws[-1]).as_quat(scalar_first=True).astype(np.float32)
    out[-1, 3:9] = quat_wxyz_to_rot6d(q_last)
    return out


def episode_smpl_to_gmr_q36(
    pose_aa_yup: np.ndarray,
    transl_yup: np.ndarray,
    *,
    proto_z0: float,
    lock_root: bool = False,
    progress_every: int = 0,
) -> np.ndarray:
    """Full egypt path: Y-up SMPL → GMR q36 with xy0/z-match postprocess."""
    pose_z, transl_z = smpl_yup_to_zup(pose_aa_yup, transl_yup)
    q = gmr_qpos_from_smpl(pose_z, transl_z, lock_root=lock_root, progress_every=progress_every)
    if not np.isfinite(q).all():
        raise RuntimeError("GMR produced non-finite qpos")
    return postprocess_gmr_root_xy0_z_match(q, proto_z0=proto_z0)


if __name__ == "__main__":
    n = 4
    pose = np.zeros((n, 72), np.float32)
    transl = np.zeros((n, 3), np.float32)
    transl[:, 1] = np.linspace(0, 0.1, n)
    pz, tz = smpl_yup_to_zup(pose, transl)
    assert pz.shape == (n, 72) and tz.shape == (n, 3)
    print("gmr_from_smpl: yup→zup ok (GMR not smoke-run here)")
