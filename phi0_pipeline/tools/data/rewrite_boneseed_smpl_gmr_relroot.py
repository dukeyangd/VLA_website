#!/usr/bin/env python3
"""BoneSEED smpl_g1csv_aligned_phi0 → smpl_gmr_relroot_phi0 (egypt gold GMR).

Per episode: SMPL sidecar → gmr_from_smpl (or _q36_cache) → Δxy+abs z+abs quat + dof.
Drops action sidecars and observation.qpos_frame0 (disk root is self-describing).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from egypt_clip_root_layout import apply_gmr_qpos_tail  # noqa: E402
from gmr_from_smpl import episode_smpl_to_gmr_q36  # noqa: E402

DEFAULT_SRC = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_g1csv_aligned_phi0"
)
DEFAULT_OUT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
DROP_COLS = (
    "action.smpl_pose_aa",
    "action.smpl_transl",
    "action.smpl_joints",
    "action.qpos_g1",
    "action.smpl_pkl_source",
    "observation.qpos_frame0",
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


def write_info_json(
    *,
    src: Path,
    out: Path,
    files_this_run: int,
    frames_this_run: int,
    elapsed_min: float,
) -> None:
    """Single-process meta finalize (call once after all shards)."""
    meta_out = out / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    src_tasks = src / "meta" / "tasks.parquet"
    if src_tasks.is_file():
        shutil.copy2(src_tasks, meta_out / "tasks.parquet")
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    feats = dict(info.get("features") or {})
    for k in DROP_COLS:
        feats.pop(k, None)
    info.pop("root_init_xyz", None)
    info.pop("root_init_quat_wxyz", None)
    info.pop("root_layout", None)
    info.update(
        {
            "dataset_type": "smpl_gmr_relroot_phi0",
            "qpos_source": "gmr_from_smpl_filtered",
            "description": (
                "BoneSEED true GMR (egypt gold): SMPL[0:315); "
                "GMR [360:367]=Δxy+abs z+abs quat; [367:396]=dof; "
                "sonic disk 0; no qpos_frame0 / meta root_init."
            ),
            "features": feats,
            "smpl_gmr_relroot": {
                "source": str(src),
                "layout": "delta_xy_abs_z_abs_quat",
                "files_this_run": files_this_run,
                "frames_this_run": frames_this_run,
                "elapsed_min": elapsed_min,
            },
        }
    )
    (meta_out / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _process_file(
    src_pq: Path,
    out_pq: Path,
    *,
    cache_dir: Path,
    bad_log: Path,
    max_episodes: int | None,
    resume_cache: bool,
) -> tuple[int, float, int]:
    """Returns (n_frames, elapsed_s, n_cache_miss)."""
    table = pq.read_table(src_pq)
    pose = _fsl_np(table.column("action.smpl_pose_aa"), 72)
    transl = _fsl_np(table.column("action.smpl_transl"), 3)
    unified = _fsl_np(table.column("action.unified"), 512).copy()
    dim_mask = _bool_fsl(table.column("action.dim_mask"), 512).copy()
    ep = np.asarray(table.column("episode_index"), dtype=np.int64)
    fi = np.asarray(table.column("frame_index"), dtype=np.int64)
    q36_src = unified[:, 360:396].copy()  # Proto abs for z0 only on cache miss

    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq = np.unique(ep)
    if max_episodes is not None:
        uniq = uniq[: max_episodes]

    n_frames = 0
    n_miss = 0
    t0 = time.time()

    for e in uniq:
        rows = np.where(ep == e)[0]
        order = rows[np.argsort(fi[rows])]
        cache_path = cache_dir / f"ep{int(e)}.npz"
        q_ep = None
        if resume_cache and cache_path.is_file():
            q_ep = np.load(cache_path)["q36"].astype(np.float32)
            if len(q_ep) != len(order):
                q_ep = None
        if q_ep is None:
            n_miss += 1
            try:
                proto_z0 = float(q36_src[order[0], 2])
                q_ep = episode_smpl_to_gmr_q36(
                    pose[order],
                    transl[order],
                    proto_z0=proto_z0,
                    progress_every=0,
                )
                np.savez_compressed(cache_path, q36=q_ep)
            except Exception as exc:  # noqa: BLE001
                with bad_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"episode": int(e), "error": str(exc)}) + "\n")
                dim_mask[order, 360:396] = False
                continue

        u_ep, dm_ep, _init_xyz, _init_q = apply_gmr_qpos_tail(
            unified[order],
            q36_abs=q_ep,
            dim_mask=dim_mask[order],
            check_roundtrip=False,
        )
        unified[order] = u_ep
        dim_mask[order] = dm_ep
        n_frames += len(order)

    if max_episodes is not None:
        done = set(int(x) for x in uniq)
        for e in np.unique(ep):
            if int(e) in done:
                continue
            rows = np.where(ep == e)[0]
            dim_mask[rows, 360:396] = False

    keep = [name for name in table.column_names if name not in DROP_COLS]
    arrays: list[pa.Array] = []
    for name in keep:
        if name == "action.unified":
            arrays.append(_np_fsl(unified, pa.float32()))
        elif name == "action.dim_mask":
            arrays.append(_np_fsl(dim_mask.astype(np.bool_), pa.bool_()))
        else:
            col = table.column(name)
            arrays.append(col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col)
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=keep), out_pq, compression="zstd")
    elapsed = time.time() - t0
    return n_frames, elapsed, n_miss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--file", type=str, default=None, help="e.g. file-000.parquet")
    ap.add_argument("--shard", type=str, default=None, help="i/N zero-based")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="ignored (compat)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip out parquet that already exists (do NOT use for absquat remigrate)",
    )
    ap.add_argument(
        "--finalize-meta-only",
        action="store_true",
        help="only rewrite meta/info.json then exit",
    )
    args = ap.parse_args()

    if args.finalize_meta_only:
        write_info_json(
            src=args.src,
            out=args.out,
            files_this_run=0,
            frames_this_run=0,
            elapsed_min=0.0,
        )
        print(f"[gmr] finalized meta → {args.out / 'meta' / 'info.json'}", flush=True)
        return

    src_pqs = sorted((args.src / "data").rglob("*.parquet"))
    if args.file:
        src_pqs = [p for p in src_pqs if p.name == args.file]
    if args.shard:
        i_s, n_s = args.shard.split("/")
        i_s, n_s = int(i_s), int(n_s)
        src_pqs = [p for j, p in enumerate(src_pqs) if j % n_s == i_s]
    if args.max_files:
        src_pqs = src_pqs[: args.max_files]
    if not src_pqs:
        raise SystemExit("no parquet selected")

    data_out = args.out / "data" / "chunk-000"
    cache_dir = args.out / "_q36_cache"
    data_out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)
    bad_log = args.out / "bad_episodes.jsonl"

    total_frames = 0
    total_miss = 0
    t_all = time.time()
    print(f"[gmr] src={args.src} out={args.out} n_files={len(src_pqs)}", flush=True)

    for fi, src_pq in enumerate(src_pqs):
        out_pq = data_out / src_pq.name
        if args.resume and out_pq.is_file() and args.max_episodes is None:
            print(f"  [{fi+1}/{len(src_pqs)}] {src_pq.name} skip", flush=True)
            continue
        print(f"  [{fi+1}/{len(src_pqs)}] {src_pq.name} …", flush=True)
        n_fr, elapsed, n_miss = _process_file(
            src_pq,
            out_pq,
            cache_dir=cache_dir,
            bad_log=bad_log,
            max_episodes=args.max_episodes,
            resume_cache=True,
        )
        total_frames += n_fr
        total_miss += n_miss
        hz = n_fr / max(elapsed, 1e-6)
        print(
            f"    → frames={n_fr:,d} {elapsed:.1f}s ({hz:.2f} Hz) "
            f"cache_miss={n_miss} → {out_pq.name}",
            flush=True,
        )

    # best-effort mid-run meta (finalize again after all shards)
    write_info_json(
        src=args.src,
        out=args.out,
        files_this_run=len(src_pqs),
        frames_this_run=total_frames,
        elapsed_min=(time.time() - t_all) / 60.0,
    )
    print(
        f"[gmr] done frames={total_frames:,d} cache_miss={total_miss} "
        f"elapsed={(time.time()-t_all)/60:.1f}min → {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
