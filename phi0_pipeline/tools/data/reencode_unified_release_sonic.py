#!/usr/bin/env python3
"""Re-encode teleop unified ``[396:460)`` via **release** Sonic (qpos → z*).

Default: GPU ``encode_zstar_onnx_g1_online`` (``PHI0_ONNX_ENCODE_GPU=1``).
Supports 8-way episode sharding + merge for multi-GPU.

Videos are hardlinked from ``--in-root``. Does not overwrite in-root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")

from phi0.online.onnx_encode import (  # noqa: E402
    _ENCODER_OBS_DIM,
    _MOTION_TOKEN_DIM,
    encode_zstar_onnx_g1_online,
)
from phi0.schema.unified_action_schema import SLICES  # noqa: E402

Q36 = SLICES["g1_body_qpos_36"]
SONIC = SLICES["sonic_motion_token_64"]
FPS = 50.0
D_UNIFIED = 512
RELEASE_ONNX_DEFAULT = (
    _ROOT / "subpackages/gear_sonic_deploy/policy/release/model_encoder.onnx"
)
VIDEO_KEYS = (
    "observation.images.ego_view",
    "observation.images.left_wrist",
)


def _fsl_np(col: pa.Array | pa.ChunkedArray, width: int) -> np.ndarray:
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    flat = arr.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(len(arr), width)


def _bool_fsl(col: pa.Array | pa.ChunkedArray, width: int) -> np.ndarray:
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    flat = arr.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.bool_).reshape(len(arr), width)


def _np_fsl(mat: np.ndarray, pa_dtype: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(mat.reshape(-1), type=pa_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, mat.shape[1])


def _ep_parquet(root: Path, ep: int) -> Path:
    chunk = ep // 1000
    file_i = ep % 1000
    return root / f"data/chunk-{chunk:03d}/file-{file_i:03d}.parquet"


def _encode_episode_q36_gpu(
    q36: np.ndarray,
    *,
    fps: float,
    batch: int,
) -> np.ndarray:
    """q36 [T,36] → z* [T,64] via GPU release encoder (FK root as anchor)."""
    t_len = int(q36.shape[0])
    dof = np.asarray(q36[:, 7:36], dtype=np.float32)
    root_q = np.asarray(q36[:, 3:7], dtype=np.float32)
    z = np.zeros((t_len, _MOTION_TOKEN_DIM), dtype=np.float32)
    tape_cache: dict = {}
    for s in range(0, t_len, batch):
        e = min(s + batch, t_len)
        times = torch.arange(s, e, dtype=torch.long)
        tok = encode_zstar_onnx_g1_online(
            fk_dof29_mj=dof,
            fk_root_quat_wxyz=root_q,
            robot_anchor_quat_wxyz=root_q[s:e],
            times=times,
            horizon=1,
            fps=float(fps),
            tape_cache=tape_cache,
            episode_end_frame=float(t_len - 1),
            use_gpu=True,
            device="cuda",
        )
        z[s:e] = tok.reshape(-1, _MOTION_TOKEN_DIM).detach().float().cpu().numpy()
    return z


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_videos(in_root: Path, out_root: Path, eps: list[int]) -> int:
    n = 0
    for ep in eps:
        for key in VIDEO_KEYS:
            src = in_root / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
            if not src.is_file():
                continue
            dst = out_root / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
            _hardlink_or_copy(src, dst)
            n += 1
    return n


def _load_sources(in_root: Path, max_episodes: int | None) -> list[dict]:
    meta_in = json.loads((in_root / "meta.json").read_text(encoding="utf-8"))
    sources = list(meta_in.get("sources") or [])
    if not sources:
        files = sorted((in_root / "data").rglob("file-*.parquet"))
        sources = [{"out_episode_index": int(p.stem.split("-")[-1])} for p in files]
    sources = sorted(sources, key=lambda r: int(r["out_episode_index"]))
    if max_episodes is not None:
        sources = sources[: int(max_episodes)]
    return sources, meta_in


def _write_episode(
    *,
    in_root: Path,
    out_root: Path,
    ep: int,
    fps: float,
    encode_batch: int,
) -> tuple[int, float, float, float]:
    """Returns (T, sonic_abs_sum, old_abs_sum, cos_sum)."""
    src_pq = _ep_parquet(in_root, ep)
    table = pq.read_table(src_pq)
    u = _fsl_np(table.column("action.unified"), D_UNIFIED)
    m = _bool_fsl(table.column("action.dim_mask"), D_UNIFIED)
    q0, q1 = Q36
    s0, s1 = SONIC
    q36 = u[:, q0:q1].copy()
    old_z = u[:, s0:s1].copy()
    z = _encode_episode_q36_gpu(q36, fps=fps, batch=encode_batch)
    u2 = u.copy()
    m2 = m.copy()
    u2[:, s0:s1] = z
    m2[:, s0:s1] = True
    denom = (np.linalg.norm(old_z, axis=1) * np.linalg.norm(z, axis=1)) + 1e-8
    cos = np.sum(old_z * z, axis=1) / denom
    cols = {n: table.column(n) for n in table.column_names}
    cols["action.unified"] = _np_fsl(u2, pa.float32())
    cols["action.dim_mask"] = _np_fsl(m2, pa.bool_())
    dst_pq = _ep_parquet(out_root, ep)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst_pq, compression="zstd")
    print(
        f"[release] ep={ep:03d} T={len(z)} sonic_absmean={float(np.abs(z).mean()):.4f} "
        f"cos_vs_ll={float(cos.mean()):.3f} gpu={torch.cuda.current_device()}",
        flush=True,
    )
    return (
        int(len(z)),
        float(np.abs(z).sum()),
        float(np.abs(old_z).sum()),
        float(cos.sum()),
    )


def run_shard(
    *,
    in_root: Path,
    shard_dir: Path,
    shard_id: int,
    num_shards: int,
    max_episodes: int | None,
    fps: float,
    encode_batch: int,
    overwrite: bool,
) -> None:
    in_root = in_root.resolve()
    shard_dir = shard_dir.resolve()
    if shard_dir.exists() and any(shard_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{shard_dir} exists (pass --overwrite)")
    if shard_dir.exists() and overwrite:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    sources, _meta = _load_sources(in_root, max_episodes)
    mine = [r for r in sources if int(r["out_episode_index"]) % num_shards == shard_id]
    eps = [int(r["out_episode_index"]) for r in mine]
    print(
        f"[shard {shard_id}/{num_shards}] eps={len(eps)} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    sonic_abs = old_abs = cos_acc = 0.0
    sonic_n = cos_n = 0
    frames = 0
    for ep in eps:
        t, sa, oa, cs = _write_episode(
            in_root=in_root,
            out_root=shard_dir,
            ep=ep,
            fps=fps,
            encode_batch=encode_batch,
        )
        frames += t
        sonic_abs += sa
        old_abs += oa
        cos_acc += cs
        sonic_n += t * _MOTION_TOKEN_DIM
        cos_n += t
    manifest = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "episodes": eps,
        "frames": frames,
        "sonic_abs_sum": sonic_abs,
        "old_abs_sum": old_abs,
        "cos_sum": cos_acc,
        "sonic_n": sonic_n,
        "cos_n": cos_n,
    }
    (shard_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[shard {shard_id}] done frames={frames}", flush=True)


def merge_shards(
    *,
    in_root: Path,
    out_root: Path,
    shards_root: Path,
    num_shards: int,
    onnx: Path,
    overwrite: bool,
) -> None:
    in_root = in_root.resolve()
    out_root = out_root.resolve()
    shards_root = shards_root.resolve()
    if out_root.exists() and any(
        p.name != ".shards" for p in out_root.iterdir()
    ) and not overwrite:
        # allow out_root to contain only .shards
        raise FileExistsError(f"{out_root} exists (pass --overwrite)")

    sources, meta_in = _load_sources(in_root, None)
    eps_all = [int(r["out_episode_index"]) for r in sources]

    # move/link parquet from shards
    sonic_abs = old_abs = cos_acc = 0.0
    sonic_n = cos_n = 0
    frames = 0
    seen: set[int] = set()
    for k in range(num_shards):
        sd = shards_root / f"shard{k}"
        man_p = sd / "shard_manifest.json"
        if not man_p.is_file():
            raise FileNotFoundError(man_p)
        man = json.loads(man_p.read_text(encoding="utf-8"))
        sonic_abs += float(man["sonic_abs_sum"])
        old_abs += float(man["old_abs_sum"])
        cos_acc += float(man["cos_sum"])
        sonic_n += int(man["sonic_n"])
        cos_n += int(man["cos_n"])
        frames += int(man["frames"])
        for ep in man["episodes"]:
            ep = int(ep)
            src = _ep_parquet(sd, ep)
            dst = _ep_parquet(out_root, ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            seen.add(ep)
    missing = sorted(set(eps_all) - seen)
    if missing:
        raise RuntimeError(f"merge missing episodes: {missing[:20]}...")

    n_vid = _link_videos(in_root, out_root, eps_all)
    print(f"[merge] videos={n_vid} eps={len(seen)} frames={frames}", flush=True)

    # stats
    all_u = []
    for ep in eps_all:
        t = pq.read_table(_ep_parquet(out_root, ep), columns=["action.unified"])
        all_u.append(_fsl_np(t.column("action.unified"), D_UNIFIED))
    U = np.concatenate(all_u, axis=0)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "modality.json", "tasks.parquet", "episodes.jsonl"):
        src = in_root / "meta" / name
        if src.is_file():
            shutil.copy2(src, out_root / "meta" / name)
    stats = {
        "action.unified": {
            "mean": U.mean(axis=0).tolist(),
            "std": U.std(axis=0).tolist(),
            "min": U.min(axis=0).tolist(),
            "max": U.max(axis=0).tolist(),
        }
    }
    (out_root / "meta" / "stats.json").write_text(
        json.dumps(stats) + "\n", encoding="utf-8"
    )

    s0, s1 = SONIC
    meta = dict(meta_in)
    meta["episodes"] = len(eps_all)
    meta["frames"] = int(U.shape[0])
    meta["sources"] = sources
    meta["sonic_motion_token"] = {
        "slice": [s0, s1],
        "source": "qpos_release_reencode",
        "policy": "release",
        "onnx": str(onnx.resolve()),
        "obs_dim": _ENCODER_OBS_DIM,
        "mode_id": 0,
        "note": (
            "Re-encoded from g1_body_qpos_36 via release GPU encoder "
            "(1762-d, mode=0, 10×step5). Not low_latency record tokens."
        ),
        "abs_mean": sonic_abs / max(sonic_n, 1),
        "ll_token_abs_mean": old_abs / max(sonic_n, 1),
        "cos_vs_ll_mean": cos_acc / max(cos_n, 1),
        "in_root": str(in_root),
        "num_shards": num_shards,
    }
    meta["qpos_note"] = (
        str(meta.get("qpos_note", ""))
        + " | sonic[396:460)=release reencode from disk q36 (not LL motion_token)."
    )
    meta["layout_note"] = "release_sonic_reencode_from_qpos"
    (out_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    info_path = out_root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_episodes"] = len(eps_all)
        info["total_frames"] = int(U.shape[0])
        info["sonic_encoder"] = meta["sonic_motion_token"]
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    # cleanup shards
    if shards_root.is_dir():
        shutil.rmtree(shards_root)
    print(
        f"[merge] wrote {out_root} eps={len(eps_all)} frames={U.shape[0]} "
        f"cos_vs_ll={meta['sonic_motion_token']['cos_vs_ll_mean']:.3f}",
        flush=True,
    )


def run_single(
    *,
    in_root: Path,
    out_root: Path,
    onnx: Path,
    max_episodes: int | None,
    overwrite: bool,
    fps: float,
    encode_batch: int,
) -> None:
    """Non-sharded encode (smoke / single GPU)."""
    in_root = in_root.resolve()
    out_root = out_root.resolve()
    if out_root == in_root:
        raise ValueError("out_root must differ from in_root")
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_root} exists (pass --overwrite)")
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "modality.json", "tasks.parquet", "episodes.jsonl"):
        src = in_root / "meta" / name
        if src.is_file():
            shutil.copy2(src, out_root / "meta" / name)

    sources, meta_in = _load_sources(in_root, max_episodes)
    eps = [int(r["out_episode_index"]) for r in sources]
    sonic_abs = old_abs = cos_acc = 0.0
    sonic_n = cos_n = 0
    for ep in eps:
        t, sa, oa, cs = _write_episode(
            in_root=in_root,
            out_root=out_root,
            ep=ep,
            fps=fps,
            encode_batch=encode_batch,
        )
        sonic_abs += sa
        old_abs += oa
        cos_acc += cs
        sonic_n += t * _MOTION_TOKEN_DIM
        cos_n += t
    n_vid = _link_videos(in_root, out_root, eps)
    print(f"[release] videos={n_vid}", flush=True)

    all_u = [
        _fsl_np(
            pq.read_table(_ep_parquet(out_root, ep), columns=["action.unified"]).column(
                "action.unified"
            ),
            D_UNIFIED,
        )
        for ep in eps
    ]
    U = np.concatenate(all_u, axis=0)
    stats = {
        "action.unified": {
            "mean": U.mean(axis=0).tolist(),
            "std": U.std(axis=0).tolist(),
            "min": U.min(axis=0).tolist(),
            "max": U.max(axis=0).tolist(),
        }
    }
    (out_root / "meta" / "stats.json").write_text(
        json.dumps(stats) + "\n", encoding="utf-8"
    )
    s0, s1 = SONIC
    meta = dict(meta_in)
    meta["episodes"] = len(eps)
    meta["frames"] = int(U.shape[0])
    meta["sources"] = sources
    meta["sonic_motion_token"] = {
        "slice": [s0, s1],
        "source": "qpos_release_reencode",
        "policy": "release",
        "onnx": str(onnx.resolve()),
        "obs_dim": _ENCODER_OBS_DIM,
        "mode_id": 0,
        "note": "GPU release reencode from q36.",
        "abs_mean": sonic_abs / max(sonic_n, 1),
        "ll_token_abs_mean": old_abs / max(sonic_n, 1),
        "cos_vs_ll_mean": cos_acc / max(cos_n, 1),
        "in_root": str(in_root),
    }
    meta["layout_note"] = "release_sonic_reencode_from_qpos"
    (out_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[release] wrote {out_root} eps={len(eps)} frames={U.shape[0]} "
        f"cos_vs_ll={meta['sonic_motion_token']['cos_vs_ll_mean']:.3f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-root",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/820demo/820demo_skill_1_unified"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/mnt/data2/wpy/workspace/820demo/820demo_skill_1_release_unified"
        ),
    )
    p.add_argument("--onnx", type=Path, default=RELEASE_ONNX_DEFAULT)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Worker output dir (default OUT/.shards/shard{id})",
    )
    p.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge OUT/.shards/shard* into OUT_ROOT",
    )
    args = p.parse_args()

    if args.merge_shards:
        shards_root = args.out_root / ".shards"
        merge_shards(
            in_root=args.in_root,
            out_root=args.out_root,
            shards_root=shards_root,
            num_shards=int(args.num_shards),
            onnx=args.onnx,
            overwrite=bool(args.overwrite),
        )
        return

    if args.shard_id is not None:
        shard_dir = args.shard_dir or (
            args.out_root / ".shards" / f"shard{int(args.shard_id)}"
        )
        run_shard(
            in_root=args.in_root,
            shard_dir=shard_dir,
            shard_id=int(args.shard_id),
            num_shards=int(args.num_shards),
            max_episodes=args.max_episodes,
            fps=float(args.fps),
            encode_batch=int(args.encode_batch),
            overwrite=bool(args.overwrite),
        )
        return

    run_single(
        in_root=args.in_root,
        out_root=args.out_root,
        onnx=args.onnx,
        max_episodes=args.max_episodes,
        overwrite=bool(args.overwrite),
        fps=float(args.fps),
        encode_batch=int(args.encode_batch),
    )


if __name__ == "__main__":
    main()
