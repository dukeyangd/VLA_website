#!/usr/bin/env python3
"""Offline onnx_g1 encode egypt clip → z* npy + GT-replay tokens.npz + stats.

Tape FK root (not live). Same encoder as LOCKED teacher / compute_z_star_stats.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")


def _pack_tokens(
    z: np.ndarray,
    *,
    fps: float,
    root_action: np.ndarray | None = None,
    observation_qpos: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    t = int(z.shape[0])
    out: dict[str, np.ndarray] = {
        "motion_token": np.asarray(z, dtype=np.float32),
        "left_hand": np.zeros((t, 7), dtype=np.float32),
        "right_hand": np.zeros((t, 7), dtype=np.float32),
        "timestamp": (np.arange(t, dtype=np.float64) / float(fps)),
        "nav_yaw": np.zeros(t, dtype=np.float32),
        "dataset_fps": np.array([fps], dtype=np.float32),
        "ego_video_offset": np.array([0], dtype=np.int64),
    }
    # Root from clip qpos pack — needed by PROTO_MOCAP / snap + heading integrate.
    if root_action is not None:
        ra = np.asarray(root_action, dtype=np.float32)
        if ra.shape[0] != t:
            raise ValueError(f"root_action T={ra.shape[0]} != z T={t}")
        out["root_action"] = ra
    if observation_qpos is not None:
        q = np.asarray(observation_qpos, dtype=np.float32)
        if q.shape[0] != t:
            raise ValueError(f"observation_qpos T={q.shape[0]} != z T={t}")
        out["observation_qpos"] = q
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ref-root",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/egypt_smplsem_clip"),
    )
    ap.add_argument("--ref-start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=278)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--qpos-npz",
        type=Path,
        default=None,
        help="Optional qpos_from_clip.npz for root_action / observation_qpos",
    )
    args = ap.parse_args()

    from phi0.online.latent_ref import load_sonic_latent_reference
    from phi0.online.onnx_encode import precompute_ref_z_onnx_g1

    ref = load_sonic_latent_reference(
        args.ref_root,
        max_frames=int(args.max_frames),
        start=int(args.ref_start),
        require_rsi=True,
        require_smpl=False,
    )
    ref.require_rsi_state()
    fps = float(getattr(ref, "fps", 50.0) or 50.0)
    z = np.asarray(precompute_ref_z_onnx_g1(ref, fps=fps), dtype=np.float32)
    if z.ndim != 2 or z.shape[-1] != 64:
        raise ValueError(f"expected z*[T,64], got {z.shape}")

    qpos_npz = args.qpos_npz
    if qpos_npz is None:
        cand = Path(args.ref_root) / "qpos_from_clip.npz"
        if cand.is_file():
            qpos_npz = cand
    root_action = None
    observation_qpos = None
    if qpos_npz is not None and Path(qpos_npz).is_file():
        qd = np.load(qpos_npz)
        if "root_action" in qd.files:
            root_action = qd["root_action"][: z.shape[0]]
        if "observation_qpos" in qd.files:
            observation_qpos = qd["observation_qpos"][: z.shape[0]]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z_path = out / "z_star_onnx_g1.npy"
    tok_path = out / "tokens_zstar.npz"
    stats_path = out / "z_star_stats.json"

    np.save(z_path, z)
    np.savez(
        tok_path,
        **_pack_tokens(
            z,
            fps=fps,
            root_action=root_action,
            observation_qpos=observation_qpos,
        ),
    )

    mean = z.mean(axis=0).astype(np.float64)
    std = np.maximum(z.std(axis=0).astype(np.float64), 0.01)
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n": int(z.shape[0]),
        "dim": 64,
        "encoder": "onnx_g1",
        "encode": "precompute_ref_z_onnx_g1",
        "anchor": "fk_root_tape",
        "ref_root": str(args.ref_root),
        "ref_start": int(args.ref_start),
        "max_frames": int(args.max_frames),
        "rsi_source": getattr(ref, "rsi_source", None),
        "z_mean": float(z.mean()),
        "z_absmax": float(np.abs(z).max()),
        "z_std": float(z.std()),
        "tokens_has_root_action": root_action is not None,
        "tokens_has_observation_qpos": observation_qpos is not None,
        "qpos_npz": str(qpos_npz) if qpos_npz is not None else None,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(
        f"[ok] T={z.shape[0]} mean={stats['z_mean']:.4f} absmax={stats['z_absmax']:.4f} "
        f"root_action={root_action is not None} qpos={observation_qpos is not None} "
        f"npy={z_path} tokens={tok_path} stats={stats_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
