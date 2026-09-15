#!/usr/bin/env python3
"""Compute live ``z*`` mean/std for online distill.

Deploy ``SonicMotionEncoder`` via ``encode_zstar_onnx_g1_online``
(same path as train; ``robot_anchor=fk_root`` ≡ offline tape when tracking).

Example (egypt LOCKED)::

  PYTHONPATH=src python tools/data/compute_z_star_stats.py \\
    --ref-root /mnt/data2/wpy/workspace/egypt_smplsem_clip \\
    --max-frames 278 --out meta/z_star_stats.json

BoneSEED (do not overwrite egypt LOCKED file)::

  PYTHONPATH=src python tools/data/compute_z_star_stats.py \\
    --ref-root /mnt/efs_1/.../smpl_gmr_relroot_phi0 \\
    --max-frames 16384 --stride 4 --out meta/z_star_stats_boneseed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ref-root",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/egypt_smplsem_clip"),
    )
    ap.add_argument("--ref-start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=278)
    ap.add_argument("--stride", type=int, default=1, help="Subsample frame starts")
    ap.add_argument("--batch-size", type=int, default=32, help="Online ONNX batch size")
    ap.add_argument(
        "--encoder",
        choices=("onnx_g1",),
        default="onnx_g1",
        help="Sonic qpos/GMR encoder only (legacy --encoder smpl removed)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "meta" / "z_star_stats.json",
    )
    args = ap.parse_args()

    from phi0.online.latent_ref import load_sonic_latent_reference
    from phi0.online.onnx_encode import encode_zstar_onnx_g1_online

    ref = load_sonic_latent_reference(
        args.ref_root,
        max_frames=int(args.max_frames),
        start=int(args.ref_start),
        require_rsi=True,
        require_smpl=False,
    )
    ref.require_rsi_state()
    t = int(len(ref))
    times = np.arange(0, t, max(1, int(args.stride)), dtype=np.int64)
    print(
        f"[z_stats] online ONNX G1 encode T={t} n_times={len(times)} "
        f"rsi={ref.rsi_source} anchor=fk_root",
        flush=True,
    )
    chunks: list[np.ndarray] = []
    bs = max(1, int(args.batch_size))
    tape_cache: dict = {}
    for i0 in range(0, len(times), bs):
        ts = times[i0 : i0 + bs]
        live = np.asarray(ref.fk_root_quat, dtype=np.float32)[ts]
        z = (
            encode_zstar_onnx_g1_online(
                fk_dof29_mj=ref.fk_dof29,
                fk_root_quat_wxyz=ref.fk_root_quat,
                robot_anchor_quat_wxyz=live,
                times=ts,
                horizon=1,
                fps=float(getattr(ref, "fps", 50.0) or 50.0),
                tape_cache=tape_cache,
            )[:, 0]
            .detach()
            .cpu()
            .numpy()
        )
        chunks.append(z)
        done = min(i0 + bs, len(times))
        if done == len(times) or done % 64 == 0:
            print(f"[z_stats] encoded {done}/{len(times)}", flush=True)
    z_all = np.concatenate(chunks, axis=0)

    mean = z_all.mean(axis=0).astype(np.float64)
    std = z_all.std(axis=0).astype(np.float64)
    std = np.maximum(std, 0.01)
    out = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n": int(z_all.shape[0]),
        "dim": 64,
        "encoder": "onnx_g1",
        "encode": "online_onnx_g1",
        "anchor": "fk_root",
        "ref_root": str(args.ref_root),
        "ref_start": int(args.ref_start),
        "max_frames": int(args.max_frames),
        "stride": int(args.stride),
        "rsi_source": getattr(ref, "rsi_source", None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[z_stats] wrote {args.out} n={out['n']} encoder=onnx_g1", flush=True)


if __name__ == "__main__":
    main()
