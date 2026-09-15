#!/usr/bin/env python3
"""Export a synchronized SONIC decoder history seed for GT replay.

The SONIC v1.1 decoder consumes 10 frames of robot history in addition to the
64-D token.  This tool exports those entries in the exact StateLogger layout
used by the C++ deploy process and an RSI pose for the newest seed frame.

CSV row layout (94 values):
  base_quat_wxyz[4], base_ang_vel[3], body_q_il_relative[29],
  body_dq_il[29], previous_policy_action_il[29]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from phi0.deploy.ref_traj_builder import DEFAULT_QPOS_FULL
from phi0.schema.unified_action_schema import SLICES


# MuJoCo/WBC index -> IsaacLab index, matching policy_parameters.hpp.
_MJ_TO_IL = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)

_DEFAULT_MJ = np.asarray(DEFAULT_QPOS_FULL[7:36], dtype=np.float32)

# Exact action scales from policy_parameters.hpp.
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_OMEGA = 10.0 * 2.0 * np.pi


def _scale(effort: float, armature: float) -> float:
    return 0.25 * effort / (armature * _OMEGA * _OMEGA)


_ACTION_SCALE_MJ = np.asarray(
    [
        _scale(139, _ARMATURE_7520_22), _scale(139, _ARMATURE_7520_22),
        _scale(88, _ARMATURE_7520_14), _scale(139, _ARMATURE_7520_22),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(139, _ARMATURE_7520_22), _scale(139, _ARMATURE_7520_22),
        _scale(88, _ARMATURE_7520_14), _scale(139, _ARMATURE_7520_22),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(88, _ARMATURE_7520_14), _scale(25, _ARMATURE_5020),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(5, _ARMATURE_4010), _scale(5, _ARMATURE_4010),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(25, _ARMATURE_5020), _scale(25, _ARMATURE_5020),
        _scale(25, _ARMATURE_5020), _scale(5, _ARMATURE_4010),
        _scale(5, _ARMATURE_4010),
    ],
    dtype=np.float32,
)


def _stack(table: pq.Table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_numpy()).astype(np.float32, copy=False)


def _select_episode(table: pq.Table, episode_index: int | None) -> pq.Table:
    if episode_index is None or "episode_index" not in table.column_names:
        return table
    ep = np.asarray(table.column("episode_index").to_numpy()).reshape(-1)
    rows = np.flatnonzero(ep == int(episode_index))
    if not rows.size:
        raise ValueError(f"episode_index={episode_index} is absent")
    return table.take(rows.tolist())


def _mj_to_il(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values)
    out[..., _MJ_TO_IL] = values
    return out


def _angular_velocity(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    out = np.zeros((len(quat), 3), dtype=np.float32)
    if len(quat) < 2:
        return out
    rotations = R.from_quat(quat[:, [1, 2, 3, 0]])
    for i in range(1, len(quat)):
        out[i] = ((rotations[i - 1].inv() * rotations[i]).as_rotvec() * fps).astype(
            np.float32
        )
    out[0] = out[1]
    return out


def _load_body(parquet: Path, episode_index: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = _select_episode(pq.read_table(parquet), episode_index)
    cols = set(table.column_names)
    if "action.unified" in cols or "unified_action" in cols:
        name = "action.unified" if "action.unified" in cols else "unified_action"
        unified = _stack(table, name)
        start, end = SLICES["g1_body_qpos_36"]
        q36 = unified[:, start:end]
        return q36[:, 7:36], q36[:, 3:7], q36[:, 0:3]
    if "observation.state" not in cols or "observation.root_orientation" not in cols:
        raise ValueError("parquet needs unified qpos or observation.state/root_orientation")
    state = _stack(table, "observation.state")
    body = np.concatenate([state[:, :22], state[:, 29:36]], axis=1)
    quat = _stack(table, "observation.root_orientation")
    root = np.zeros((len(body), 3), dtype=np.float32)
    root[:, 2] = 0.785
    return body, quat, root


def _load_actions(
    parquet: Path, episode_index: int | None, expected_rows: int
) -> np.ndarray:
    table = _select_episode(pq.read_table(parquet), episode_index)
    if "action.wbc" not in table.column_names:
        return np.zeros((expected_rows, 29), dtype=np.float32)
    wbc = _stack(table, "action.wbc")
    target_mj = np.concatenate([wbc[:, :22], wbc[:, 29:36]], axis=1)
    if len(target_mj) != expected_rows:
        raise ValueError(f"action rows {len(target_mj)} != body rows {expected_rows}")
    action_mj = (target_mj - _DEFAULT_MJ) / _ACTION_SCALE_MJ
    return _mj_to_il(action_mj)


def export_seed(
    parquet: Path,
    out: Path,
    rsi_out: Path,
    *,
    action_parquet: Path | None = None,
    episode_index: int | None = None,
    action_episode_index: int | None = None,
    start_frame: int = 9,
    history_frames: int = 10,
    fps: float = 50.0,
) -> None:
    body_mj, quat, root = _load_body(parquet, episode_index)
    n = len(body_mj)
    if not 0 <= start_frame < n:
        raise IndexError(f"start_frame={start_frame} outside [0,{n})")
    body_dq_mj = np.zeros_like(body_mj)
    if n > 1:
        body_dq_mj[1:] = np.diff(body_mj, axis=0) * float(fps)
        body_dq_mj[0] = body_dq_mj[1]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    ang_vel = _angular_velocity(quat, fps)
    action_il = _load_actions(
        action_parquet or parquet,
        action_episode_index if action_episode_index is not None else episode_index,
        n,
    )

    q_rel_il = _mj_to_il(body_mj - _DEFAULT_MJ)
    dq_il = _mj_to_il(body_dq_mj)
    indices = list(range(max(0, start_frame - history_frames + 1), start_frame + 1))
    while len(indices) < history_frames:
        indices.insert(0, indices[0])

    rows: list[np.ndarray] = []
    for frame in indices:
        previous_action = np.zeros(29, dtype=np.float32) if frame == 0 else action_il[frame - 1]
        rows.append(
            np.concatenate(
                [quat[frame], ang_vel[frame], q_rel_il[frame], dq_il[frame], previous_action]
            ).astype(np.float64)
        )
    matrix = np.stack(rows)
    if matrix.shape != (history_frames, 94) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid history matrix {matrix.shape}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out,
        matrix,
        delimiter=",",
        fmt="%.10g",
        header="quat_wxyz4,ang_vel3,q_il_rel29,dq_il29,previous_action_il29",
    )
    rsi_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        rsi_out,
        observation_qpos=body_mj[start_frame].astype(np.float32),
        reference_root_xyz=root[start_frame].astype(np.float32),
        reference_root_quat_wxyz=quat[start_frame].astype(np.float32),
        dataset_fps=np.asarray([fps], dtype=np.float32),
        frame_index=np.asarray([start_frame], dtype=np.int64),
    )
    print(
        f"[ok] history={out} shape={matrix.shape} frames={indices[0]}..{indices[-1]} "
        f"q_l2={np.linalg.norm(matrix[-1, 7:36]):.4f} "
        f"prev_action_l2={np.linalg.norm(matrix[-1, 65:94]):.4f}"
    )
    print(f"[ok] rsi={rsi_out} frame={start_frame} root_z={root[start_frame, 2]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rsi-out", type=Path, required=True)
    parser.add_argument("--action-parquet", type=Path, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--action-episode-index", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=9)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()
    export_seed(
        args.parquet,
        args.out,
        args.rsi_out,
        action_parquet=args.action_parquet,
        episode_index=args.episode_index,
        action_episode_index=args.action_episode_index,
        start_frame=args.start_frame,
        history_frames=args.history_frames,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
