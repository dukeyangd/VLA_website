#!/usr/bin/env python3
"""Capture ZMQ :5556 Dex3 hand output and compare to replay parquet gripper14 [346:360].

ZMQ left/right_hand_joints are actuator order (thumb,index,middle).
Parquet unified[346:360] is policy order (index,middle,thumb per hand).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
for root in (_PHI0 / "src", _PHI0 / "subpackages"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.utils.zmq_pose_unpack import unpack_pose_message  # noqa: E402
from phi0.deploy.dex3_gripper import (  # noqa: E402
    dex3_actuator_to_policy_hand,
    NUM_GRIPPER_EACH,
)
from phi0.schema.unified_action_schema import SLICES  # noqa: E402

POLICY14_LABELS = [
    "Li0", "Li1", "Lm0", "Lm1", "Lt0", "Lt1", "Lt2",
    "Ri0", "Ri1", "Rm0", "Rm1", "Rt0", "Rt1", "Rt2",
]
GRIP_SLICE = slice(*SLICES["g1_gripper_joints_14"])


def actuator14_to_policy14(left7: np.ndarray, right7: np.ndarray) -> np.ndarray:
    l_pol = dex3_actuator_to_policy_hand(np.asarray(left7, dtype=np.float32).reshape(7))
    r_pol = dex3_actuator_to_policy_hand(np.asarray(right7, dtype=np.float32).reshape(7))
    return np.concatenate([l_pol, r_pol]).astype(np.float32)


def load_gt_policy14(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(str(parquet_path))
    col = "action.unified" if "action.unified" in table.column_names else "unified"
    if col not in table.column_names:
        raise KeyError(f"no gripper column in {parquet_path.name}: {table.column_names}")
    unified = np.stack(table.column(col).to_pylist()).astype(np.float32)
    return unified[:, GRIP_SLICE]


def capture_zmq(host: str, port: int, duration_s: float) -> list[dict]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(f"tcp://{host}:{port}")
    deadline = time.monotonic() + duration_s
    rows: list[dict] = []
    while time.monotonic() < deadline:
        try:
            raw = sub.recv()
        except zmq.Again:
            continue
        msg = unpack_pose_message(raw, "pose")
        fi = int(np.asarray(msg.get("frame_index", [0])).reshape(-1)[0])
        l7 = np.asarray(msg["left_hand_joints"], dtype=np.float32).reshape(NUM_GRIPPER_EACH)
        r7 = np.asarray(msg["right_hand_joints"], dtype=np.float32).reshape(NUM_GRIPPER_EACH)
        rows.append({"frame": fi, "left_act7": l7, "right_act7": r7})
    sub.close()
    ctx.term()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Dex3 14D ZMQ vs parquet gripper14")
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gt = load_gt_policy14(args.parquet)
    n_gt = gt.shape[0]
    print(f"GT parquet: {args.parquet}  T={n_gt}  gripper14 policy order {GRIP_SLICE.start}:{GRIP_SLICE.stop}")

    cap = capture_zmq(args.host, args.port, args.duration_s)
    if not cap:
        raise SystemExit(f"no ZMQ messages on {args.host}:{args.port} in {args.duration_s}s")

    frames = np.array([r["frame"] for r in cap], dtype=np.int64)
    pol = np.stack(
        [actuator14_to_policy14(r["left_act7"], r["right_act7"]) for r in cap],
        axis=0,
    )
    act = np.stack(
        [np.concatenate([r["left_act7"], r["right_act7"]]) for r in cap],
        axis=0,
    )

    # align: frame index mod episode length (loop replay)
    fi_mod = (frames % n_gt).astype(np.int64)
    gt_aligned = gt[fi_mod]
    err = pol - gt_aligned
    abs_err = np.abs(err)

    print(f"Captured {len(cap)} ZMQ msgs  frame=[{frames.min()}, {frames.max()}]")
    print(f"Policy14 vs GT: max_abs={abs_err.max():.6f}  mean_abs={abs_err.mean():.6f}")
    print(f"All close atol=0.02? {np.allclose(pol, gt_aligned, atol=0.02)}")
    print(f"All close atol=0.001? {np.allclose(pol, gt_aligned, atol=0.001)}")

    per_dim_max = abs_err.max(axis=0)
    print("\nPer-dim max |err| (policy order):")
    for lab, v in zip(POLICY14_LABELS, per_dim_max):
        print(f"  {lab}: {v:.6f}")

    # show sample windows
    for label, idx in [("start", 0), ("mid", len(cap) // 2), ("end", -1)]:
        fi = int(frames[idx])
        ti = int(fi_mod[idx])
        print(f"\n--- {label} cap#{idx} frame={fi} gt_row={ti} ---")
        print(f"  ZMQ policy14: {np.round(pol[idx], 4).tolist()}")
        print(f"  GT  policy14: {np.round(gt[ti], 4).tolist()}")
        print(f"  ZMQ act14:    {np.round(act[idx], 4).tolist()}")

    out = args.out
    if out is None:
        out = args.parquet.parent.parent.parent / "dex14_zmq_vs_gt.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame", "gt_row"] + [f"zmq_{l}" for l in POLICY14_LABELS] + [f"gt_{l}" for l in POLICY14_LABELS]
    mat = np.column_stack([frames, fi_mod, pol, gt_aligned])
    np.savetxt(out, mat, delimiter=",", header=",".join(header), comments="")
    print(f"\nSaved CSV: {out.resolve()}")


if __name__ == "__main__":
    main()
