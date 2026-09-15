#!/usr/bin/env python3
"""Convert a ``run_online_infer_eval`` npz (has ``z_pred``) into the
``tokens.npz`` format ``replay_gt_latent.sh`` / ``replay_latent_zmq.py``
(the validated sonic_latent_gt_replay deploy+MuJoCo pipeline) consume.

Why this exists: earlier ad-hoc visualization here was a bespoke kinematic
MuJoCo replay (no physics, hand-rolled joint-name remapping) that repeatedly
produced ambiguous/misleading results during debugging — e.g. mapping
dataset-order dof29 through an IsaacLab-name remap it didn't need, which
silently scrambled the skeleton. Feeding the VLA's own 64D latent output
through the *real* deploy binary (``g1_deploy_onnx_ref``) + MuJoCo sim that
already produces trusted ground-truth replay videos removes that whole class
of bug: whatever the pipeline does with GT tokens, it does identically with
ours.

Usage:
    python build_gt_replay_tokens_from_infer_npz.py INFER_NPZ --out tokens.npz

Then feed the result to replay_gt_latent.sh:
    SKIP_PARQUET_EXTRACT=1 TOKENS=tokens.npz PROTO_MOCAP_REPLAY=1 \\
      bash .../sonic_latent_gt_replay/replay_gt_latent.sh

Known limitation: root translation/heading is not reconstructed from the
VLA's own output (no validated mapping from our smpl_pred to the replay
pipeline's ``root_action`` [N,9] format yet) — this writes ``nav_yaw=0``
(robot stays in place, no net turning) so only the *joint articulation*
(the 64D motion_token stream) is being judged, not root travel. Root path
fidelity is a separate, not-yet-solved problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _hand_to_zmq_hands7(hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Student hand_pred → deploy ZMQ left7/right7 (Dex3 actuator order)."""
    import os

    from phi0.deploy.dex3_gripper import gripper14_to_zmq_lr
    from phi0.hand.hand_mode import HAND_MODE_DEX3, hand_mode_from_env

    h = np.asarray(hand, dtype=np.float32)
    if h.ndim != 2 or h.shape[-1] not in (12, 14):
        raise ValueError(f"expected hand [T,12|14], got {h.shape}")
    t = h.shape[0]
    left = np.zeros((t, 7), dtype=np.float32)
    right = np.zeros((t, 7), dtype=np.float32)
    if h.shape[-1] == 14:
        policy_order = hand_mode_from_env() == HAND_MODE_DEX3 and os.environ.get(
            "PHI0_DEX3_HAND_POLICY_ORDER", "1"
        ).strip().lower() not in ("0", "false", "no", "deploy", "actuator")
        for i in range(t):
            l7, r7 = gripper14_to_zmq_lr(h[i], policy_order=policy_order)
            left[i], right[i] = l7, r7
        return left, right
    left[:, :6] = np.clip(h[:, :6], 0.0, 1.0)
    right[:, :6] = np.clip(h[:, 6:12], 0.0, 1.0)
    return left, right


def _revo2_12_to_zmq_hands7(hand12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _hand_to_zmq_hands7(hand12)


def build_tokens(
    infer_npz: Path,
    *,
    env_index: int = 0,
    fps: float = 50.0,
) -> dict[str, np.ndarray]:
    d = np.load(infer_npz)
    if "z_pred" not in d.files:
        raise ValueError(f"{infer_npz} has no 'z_pred' (run_online_infer_eval output expected)")
    z_pred = np.asarray(d["z_pred"], dtype=np.float32)
    if z_pred.ndim != 3 or z_pred.shape[-1] != 64:
        raise ValueError(f"expected z_pred [T,B,64], got {z_pred.shape}")
    motion_token = z_pred[:, env_index, :]
    t = motion_token.shape[0]
    if "hand_pred" in d.files:
        hand = np.asarray(d["hand_pred"], dtype=np.float32)
        if hand.ndim != 3 or hand.shape[-1] not in (12, 14):
            raise ValueError(f"expected hand_pred [T,B,12|14], got {hand.shape}")
        left, right = _hand_to_zmq_hands7(hand[:, env_index, :])
    else:
        # Legacy npz: hands were never logged → zeros (not model grasp).
        left = np.zeros((t, 7), dtype=np.float32)
        right = np.zeros((t, 7), dtype=np.float32)
    return {
        "motion_token": motion_token,
        "left_hand": left,
        "right_hand": right,
        "timestamp": (np.arange(t, dtype=np.float64) / float(fps)),
        "nav_yaw": np.zeros(t, dtype=np.float32),
        "dataset_fps": np.array([fps], dtype=np.float32),
        "ego_video_offset": np.array([0], dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_npz", type=Path, help="run_online_infer_eval output (infer_qpos_traj_*.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", type=int, default=0, help="which env/batch row to extract")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    payload = build_tokens(args.infer_npz, env_index=args.env, fps=args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(
        f"[ok] wrote {args.out} frames={payload['motion_token'].shape[0]} "
        f"motion_token mean={payload['motion_token'].mean():.4f} "
        f"min={payload['motion_token'].min():.4f} max={payload['motion_token'].max():.4f} "
        f"L_hand_max={float(np.abs(payload['left_hand']).max()):.4f} "
        f"R_hand_max={float(np.abs(payload['right_hand']).max()):.4f}"
    )


if __name__ == "__main__":
    main()
