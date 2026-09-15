#!/usr/bin/env python3
"""Merge teleop (+ optional BoneSEED no-video) unified packs; hardlink videos + VLM cache.

ponytail: 1 chunk, file-i = ep-i, dual video keys for vision eps.
``--allow-no-video`` skips video/cache for demo5-style eps; multi-task prompts preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from pack_teleop_qpos_unified_lerobot import (  # noqa: E402
    CHUNKS_SIZE,
    VIDEO_KEYS,
    VIDEO_PATH_TMPL,
    _VIDEO_FEATURE,
    _acc_unified_stats,
)

CACHE_DIRNAME = "vlm_frame_latents_qwen3vl_dual"


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _hardlink_tree_files(src_dir: Path, dst_dir: Path) -> int:
    """Hardlink all files under src_dir into dst_dir (flat or nested)."""
    n = 0
    if not src_dir.is_dir():
        return 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        _hardlink_or_copy(src, dst)
        n += 1
    return n


def _rewrite_data_parquet(
    src: Path,
    dst: Path,
    *,
    out_ep: int,
    global_index0: int,
    task_index: int | None = None,
) -> int:
    table = pq.read_table(src)
    n = table.num_rows
    names = table.column_names
    cols = {n: table.column(n) for n in names}
    cols["episode_index"] = pa.array([out_ep] * n, type=pa.int64())
    cols["index"] = pa.array(
        list(range(global_index0, global_index0 + n)), type=pa.int64()
    )
    if task_index is not None and "task_index" in cols:
        cols["task_index"] = pa.array([int(task_index)] * n, type=pa.int64())
    # frame_index unchanged
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return n


def _load_task_map(root: Path) -> dict[int, str]:
    tasks_p = root / "meta/tasks.parquet"
    if not tasks_p.is_file():
        return {}
    t = pq.read_table(tasks_p).to_pydict()
    idxs = t.get("task_index") or []
    texts = t.get("task") or []
    return {int(i): str(x) for i, x in zip(idxs, texts)}


def _peek_task_index_from_data(src_pq: Path) -> int | None:
    try:
        t = pq.read_table(src_pq, columns=["task_index"])
    except Exception:
        return None
    if t.num_rows < 1:
        return None
    return int(t.column(0)[0].as_py())


def _episode_task_text(
    root: Path,
    src_ep: int,
    *,
    task_map: dict[int, str],
    fallback: str,
    data_task_index: int | None = None,
) -> tuple[str, int]:
    ep_p = root / "meta/episodes/chunk-000/file-000.parquet"
    task_index = 0 if data_task_index is None else int(data_task_index)
    if ep_p.is_file():
        ep = pq.read_table(ep_p).to_pandas()
        row = ep[ep["episode_index"] == src_ep]
        if len(row):
            r = row.iloc[0]
            if "task_index" in row.columns:
                task_index = int(r["task_index"])
            if "instruction" in row.columns and str(r["instruction"]).strip():
                return str(r["instruction"]).strip(), task_index
            if "tasks" in row.columns:
                ts = r["tasks"]
                if hasattr(ts, "__iter__") and not isinstance(ts, (str, bytes)):
                    arr = list(ts)
                    if arr:
                        return str(arr[0]), task_index
    if task_index in task_map:
        return task_map[task_index], task_index
    if data_task_index is None and len(task_map) == 1:
        # single-task teleop packs
        only = next(iter(task_map.items()))
        return only[1], int(only[0])
    if task_map:
        # multi-task without episode row: prefer data parquet index
        if task_index in task_map:
            return task_map[task_index], task_index
        only = next(iter(task_map.items()))
        return only[1], int(only[0])
    return fallback, task_index


def merge_roots(
    roots: list[Path],
    tags: list[str],
    out: Path,
    *,
    overwrite: bool,
    task_prompt: str,
    allow_no_video: bool = False,
) -> None:
    if len(roots) != len(tags):
        raise ValueError("roots/tags length mismatch")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists (pass --overwrite)")
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    for vk in VIDEO_KEYS:
        (out / "videos/chunk-000" / vk).mkdir(parents=True, exist_ok=True)
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cache_out = meta_dir / CACHE_DIRNAME
    (cache_out / "ep").mkdir(parents=True, exist_ok=True)

    ep_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    task_union: dict[str, int] = {}  # text -> task_index
    cursor = 0
    out_ep = 0
    n_video_files = 0
    cache_n_eps = 0
    cache_n_frames = 0
    cache_meta_template: dict[str, Any] | None = None
    video_counts = {k: 0 for k in VIDEO_KEYS}
    vision_eps: list[int] = []

    for root, tag in zip(roots, tags):
        root = root.resolve()
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        meta_in = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        src_sources = list(meta_in.get("sources") or [])
        n_ep = int(info["total_episodes"])
        if n_ep != len(src_sources):
            ep_tbl = pq.read_table(
                root / "meta/episodes/chunk-000/file-000.parquet"
            ).to_pandas()
            src_sources = [
                {
                    "out_episode_index": int(r.episode_index),
                    "session": tag,
                    "source_episode_index": int(r.episode_index),
                    "length": int(r.length),
                    "has_video": bool(r["has_video"])
                    if "has_video" in ep_tbl.columns
                    else True,
                }
                for r in ep_tbl.itertuples()
            ]
        task_map = _load_task_map(root)
        cache_in = root / "meta" / CACHE_DIRNAME
        if (cache_in / "meta.json").is_file() and cache_meta_template is None:
            cache_meta_template = json.loads(
                (cache_in / "meta.json").read_text(encoding="utf-8")
            )

        for src_row in src_sources:
            src_ep = int(src_row["out_episode_index"])
            src_pq = root / "data/chunk-000" / f"file-{src_ep:03d}.parquet"
            if not src_pq.is_file():
                raise FileNotFoundError(src_pq)

            data_ti = _peek_task_index_from_data(src_pq)
            prompt_text, src_task_i = _episode_task_text(
                root,
                src_ep,
                task_map=task_map,
                fallback=task_prompt,
                data_task_index=data_ti,
            )
            if prompt_text not in task_union:
                task_union[prompt_text] = len(task_union)
            out_task_i = task_union[prompt_text]

            has_video = bool(src_row.get("has_video", True))
            # Detect missing videos even if metadata says True.
            ego = (
                root
                / "videos/chunk-000"
                / VIDEO_KEYS[0]
                / f"episode_{src_ep:06d}.mp4"
            )
            if not ego.is_file():
                if not allow_no_video:
                    raise FileNotFoundError(ego)
                has_video = False

            dst_pq = out / "data/chunk-000" / f"file-{out_ep:03d}.parquet"
            n = _rewrite_data_parquet(
                src_pq,
                dst_pq,
                out_ep=out_ep,
                global_index0=cursor,
                task_index=out_task_i,
            )

            if has_video:
                for vk in VIDEO_KEYS:
                    src_v = (
                        root
                        / "videos/chunk-000"
                        / vk
                        / f"episode_{src_ep:06d}.mp4"
                    )
                    dst_v = (
                        out
                        / "videos/chunk-000"
                        / vk
                        / f"episode_{out_ep:06d}.mp4"
                    )
                    if not src_v.is_file():
                        raise FileNotFoundError(src_v)
                    _hardlink_or_copy(src_v, dst_v)
                    video_counts[vk] += 1
                    n_video_files += 1

                src_ced = cache_in / "ep" / f"{src_ep:06d}"
                dst_ced = cache_out / "ep" / f"{out_ep:06d}"
                if not src_ced.is_dir():
                    raise FileNotFoundError(f"missing VLM cache ep dir: {src_ced}")
                n_linked = _hardlink_tree_files(src_ced, dst_ced)
                if n_linked < 1:
                    raise RuntimeError(f"empty VLM cache: {src_ced}")
                cache_n_eps += 1
                cache_n_frames += n
                vision_eps.append(out_ep)

            ep_rows.append(
                {
                    "episode_index": out_ep,
                    "length": n,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + n,
                    "tasks": [prompt_text],
                    "instruction": prompt_text,
                    "task_index": out_task_i,
                    "has_video": bool(has_video),
                    "data/chunk_index": out_ep // CHUNKS_SIZE,
                    "data/file_index": out_ep % CHUNKS_SIZE,
                }
            )
            sources.append(
                {
                    "out_episode_index": out_ep,
                    "origin_tag": tag,
                    "origin_root": str(root),
                    "origin_episode_index": src_ep,
                    "session": src_row.get("session"),
                    "source_episode_index": src_row.get("source_episode_index"),
                    "length": n,
                    "has_video": bool(has_video),
                    "task_index": out_task_i,
                }
            )
            cursor += n
            out_ep += 1
            if out_ep % 50 == 0:
                print(f"[merge] out_ep={out_ep} frames={cursor}", flush=True)

    total_eps = out_ep
    total_frames = cursor

    ep_dir = meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), ep_dir / "file-000.parquet")

    shutil.copy2(roots[0] / "meta/modality.json", meta_dir / "modality.json")
    tasks_rows = [
        {"task_index": i, "task": text}
        for text, i in sorted(task_union.items(), key=lambda kv: kv[1])
    ]
    pq.write_table(pa.Table.from_pylist(tasks_rows), meta_dir / "tasks.parquet")

    features = {
        "action.unified": {"dtype": "float32", "shape": [512]},
        "action.dim_mask": {"dtype": "bool", "shape": [512]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for key in VIDEO_KEYS:
        features[key] = dict(_VIDEO_FEATURE)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": 50.0,
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": n_video_files,
        "total_chunks": int(math.ceil(total_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": VIDEO_PATH_TMPL,
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    (meta_dir / "all_episode_allowlist.json").write_text(
        json.dumps({"episode_index": list(range(total_eps))}, indent=2) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "vision_episode_allowlist.json").write_text(
        json.dumps({"episode_index": vision_eps}, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = _acc_unified_stats(out)
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # Optional demo5 lang latents (text-only prompts).
    for root in roots:
        lang = root / "meta/lang_latents_qwen3vl_demo5skill"
        if lang.is_dir():
            n = _hardlink_tree_files(lang, meta_dir / "lang_latents_qwen3vl_demo5skill")
            print(f"[merge] lang_latents hardlinked n_files={n} from {lang}", flush=True)
            break

    meta0 = json.loads((roots[0] / "meta.json").read_text(encoding="utf-8"))
    meta = {
        "layout": meta0.get("layout", "phi0_unified_teleop_qpos"),
        "layout_note": "merge_teleop_unified_vlm_cache (multi-root; optional no-video)",
        "qpos_source": meta0.get("qpos_source"),
        "qpos_note": meta0.get("qpos_note"),
        "hand_mode": meta0.get("hand_mode"),
        "dex3_gripper": meta0.get("dex3_gripper"),
        "sonic_motion_token": meta0.get("sonic_motion_token"),
        "videos": {
            "keys": list(VIDEO_KEYS),
            "path": VIDEO_PATH_TMPL,
            "counts": video_counts,
            "note": "hardlinked from origin packs; no-video eps skipped",
        },
        "episodes": total_eps,
        "frames": total_frames,
        "sources": sources,
        "merge": {
            "roots": [str(r.resolve()) for r in roots],
            "tags": tags,
            "vlm_cache": CACHE_DIRNAME,
            "allow_no_video": bool(allow_no_video),
            "n_vision": len(vision_eps),
            "n_no_video": total_eps - len(vision_eps),
        },
        "train": {"REF_ROOT": str(out.resolve())},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if cache_n_eps < 1 or cache_meta_template is None:
        raise RuntimeError("no VLM cache found for any vision episode")
    cache_meta = dict(cache_meta_template)
    cache_meta.update(
        {
            "dataset_root": str(out.resolve()),
            "n_episodes": cache_n_eps,
            "n_frames": cache_n_frames,
            "merged_from": [str(r.resolve()) for r in roots],
        }
    )
    (cache_out / "meta.json").write_text(
        json.dumps(cache_meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[merge] wrote {out}: {total_eps} eps / {total_frames} frames / "
        f"vision={len(vision_eps)} no_video={total_eps - len(vision_eps)} "
        f"vlm_cache_eps={cache_n_eps} tasks={len(tasks_rows)}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="unified root (repeat)")
    ap.add_argument("--tag", action="append", required=True, help="origin tag (repeat)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-prompt", default="抓起黄色玩具放到篮子里。")
    ap.add_argument(
        "--allow-no-video",
        action="store_true",
        help="Allow BoneSEED/demo5 eps without dual video / VLM frame cache",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if len(args.root) != len(args.tag):
        raise SystemExit("--root and --tag must pair 1:1")
    merge_roots(
        [Path(r) for r in args.root],
        list(args.tag),
        Path(args.out_dir),
        overwrite=bool(args.overwrite),
        task_prompt=str(args.task_prompt),
        allow_no_video=bool(args.allow_no_video),
    )


if __name__ == "__main__":
    main()
