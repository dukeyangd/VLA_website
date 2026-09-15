#!/usr/bin/env python3
"""Remove episodes from a LeRobot dataset and reindex remaining episodes.

Supports:
  - LeRobot v3.x  (meta/episodes/chunk-000/file-000.parquet, data/file-XXX.parquet)
  - LeRobot v2.1  (meta/episodes.jsonl, data/episode_XXXXXX.parquet)

Usage::

    python prune_lerobot_v3_episodes.py --root /path/to/dataset --delete 63 64 69
    python prune_lerobot_v3_episodes.py --root /path/to/dataset --delete 63 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def detect_format(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    ver = ""
    if info_path.is_file():
        try:
            ver = str(json.loads(info_path.read_text("utf-8")).get("codebase_version") or "")
        except (OSError, json.JSONDecodeError, TypeError):
            ver = ""
    if (root / "meta/episodes.jsonl").is_file() or ver.startswith("v2"):
        return "v2.1"
    if (root / "meta/episodes/chunk-000/file-000.parquet").is_file() or ver.startswith("v3"):
        return "v3"
    raise FileNotFoundError(
        f"无法识别 LeRobot 格式（需要 v2.1 episodes.jsonl 或 v3 parquet）: {root}"
    )


def discover_video_keys(root: Path) -> list[str]:
    vid_root = root / "videos" / "chunk-000"
    if not vid_root.is_dir():
        return []
    return sorted(d.name for d in vid_root.iterdir() if d.is_dir())


def load_episode_rows_v3(root: Path) -> list[dict]:
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing episode meta: {path}")
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(cols.get("episode_index") or [])
    rows = []
    for i in range(n):
        row = {k: cols[k][i] for k in cols}
        rows.append(row)
    return rows


def load_episode_rows_v21(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing episode meta: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def update_allowlist(path: Path, old_to_new: dict[int, int], delete_eps: set[int]) -> None:
    if not path.is_file():
        return
    data = json.loads(path.read_text("utf-8"))
    if "episode_index" in data:
        kept = [old_to_new[i] for i in data["episode_index"] if i not in delete_eps and i in old_to_new]
        data["episode_index"] = sorted(kept)
    if "episodes" in data:
        kept = [old_to_new[i] for i in data["episodes"] if i not in delete_eps and i in old_to_new]
        data["episodes"] = sorted(kept)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def rewrite_data_parquet(src: Path, dst: Path, new_ep: int, global_offset: int) -> int:
    table = pq.read_table(src)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(cols["index"])
    cols["episode_index"] = [new_ep] * n
    cols["index"] = list(range(global_offset, global_offset + n))
    if "frame_index" in cols:
        cols["frame_index"] = list(range(n))
    new_table = pa.table({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(new_table, dst)
    return n


def _data_src(root: Path, fmt: str, old_ep: int) -> Path:
    data_dir = root / "data" / "chunk-000"
    if fmt == "v3":
        return data_dir / f"file-{old_ep:03d}.parquet"
    return data_dir / f"episode_{old_ep:06d}.parquet"


def _data_dst_name(fmt: str, new_ep: int) -> str:
    if fmt == "v3":
        return f"file-{new_ep:03d}.parquet"
    return f"episode_{new_ep:06d}.parquet"


def prune_dataset(root: Path, delete_eps: set[int], dry_run: bool = False) -> dict:
    root = root.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")

    info = json.loads(info_path.read_text("utf-8"))
    fmt = detect_format(root)
    rows = load_episode_rows_v3(root) if fmt == "v3" else load_episode_rows_v21(root)
    old_indices = [int(r.get("episode_index")) for r in rows]
    row_by_ep = {int(r.get("episode_index")): r for r in rows}

    missing = sorted(delete_eps - set(old_indices))
    if missing:
        # Stale UI marks (e.g. labels still list ep≥total) must not abort the whole prune.
        print(
            f"[warn] skip episode(s) not in dataset: {missing}",
            file=sys.stderr,
        )
        delete_eps = set(delete_eps) - set(missing)
    if not delete_eps:
        raise ValueError(
            "没有可删除的 Episode"
            + (f"（已跳过不存在: {missing}）" if missing else "")
        )

    keep_old = sorted(i for i in old_indices if i not in delete_eps)
    if not keep_old:
        raise ValueError("cannot delete all episodes")

    old_to_new = {old: new for new, old in enumerate(keep_old)}
    video_keys = discover_video_keys(root)
    data_dir = root / "data" / "chunk-000"
    staging = root / ".prune_staging"
    report = {
        "root": str(root),
        "format": fmt,
        "deleted": sorted(delete_eps),
        "skipped_missing": missing,
        "kept_before": len(old_indices),
        "kept_after": len(keep_old),
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "dry_run": dry_run,
    }

    if dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / f".prune_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info_path, backup / "info.json")
    if fmt == "v3":
        shutil.copy2(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            backup / "episodes_meta.parquet",
        )
    else:
        shutil.copy2(root / "meta" / "episodes.jsonl", backup / "episodes.jsonl")
        stats = root / "meta" / "episodes_stats.jsonl"
        if stats.is_file():
            shutil.copy2(stats, backup / "episodes_stats.jsonl")
    report["backup_dir"] = str(backup)

    if staging.exists():
        shutil.rmtree(staging)
    staging_data = staging / "data" / "chunk-000"
    staging_videos = staging / "videos" / "chunk-000"
    staging_data.mkdir(parents=True, exist_ok=True)
    for vk in video_keys:
        (staging_videos / vk).mkdir(parents=True, exist_ok=True)

    new_episodes: list[dict] = []
    global_offset = 0
    for old_ep in keep_old:
        new_ep = old_to_new[old_ep]
        src_data = _data_src(root, fmt, old_ep)
        if not src_data.is_file():
            raise FileNotFoundError(f"missing data file: {src_data}")
        dst_data = staging_data / _data_dst_name(fmt, new_ep)
        length = rewrite_data_parquet(src_data, dst_data, new_ep, global_offset)
        old_row = row_by_ep[old_ep]
        tasks = old_row.get("tasks") or []
        entry = {
            "episode_index": new_ep,
            "length": length,
            "tasks": tasks,
        }
        if fmt == "v3":
            entry.update(
                {
                    "dataset_from_index": global_offset,
                    "dataset_to_index": global_offset + length,
                    "data/chunk_index": 0,
                    "data/file_index": new_ep,
                }
            )
        new_episodes.append(entry)
        for vk in video_keys:
            src_vid = root / "videos" / "chunk-000" / vk / f"episode_{old_ep:06d}.mp4"
            dst_vid = staging_videos / vk / f"episode_{new_ep:06d}.mp4"
            if src_vid.is_file():
                shutil.copy2(src_vid, dst_vid)
            elif new_ep == old_to_new[keep_old[0]]:
                print(f"[warn] missing video {src_vid}", file=sys.stderr)
        global_offset += length

    # Remove deleted + old-index files from live tree
    for old_ep in old_indices:
        live_data = _data_src(root, fmt, old_ep)
        if live_data.is_file():
            live_data.unlink()
        for vk in video_keys:
            live_vid = root / "videos" / "chunk-000" / vk / f"episode_{old_ep:06d}.mp4"
            if live_vid.is_file():
                live_vid.unlink()

    # Move staged files into place
    for dst_data in sorted(staging_data.iterdir()):
        if dst_data.is_file():
            shutil.move(str(dst_data), str(data_dir / dst_data.name))
    for vk in video_keys:
        for dst_vid in (staging_videos / vk).glob("*.mp4"):
            shutil.move(str(dst_vid), str(root / "videos" / "chunk-000" / vk / dst_vid.name))

    # Write episodes meta
    if fmt == "v3":
        ep_out = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_meta = pa.table(
            {
                "episode_index": [r["episode_index"] for r in new_episodes],
                "length": [r["length"] for r in new_episodes],
                "dataset_from_index": [r["dataset_from_index"] for r in new_episodes],
                "dataset_to_index": [r["dataset_to_index"] for r in new_episodes],
                "tasks": [r["tasks"] for r in new_episodes],
                "data/chunk_index": [0] * len(new_episodes),
                "data/file_index": [r["data/file_index"] for r in new_episodes],
            }
        )
        pq.write_table(ep_meta, ep_out)
    else:
        ep_out = root / "meta" / "episodes.jsonl"
        with ep_out.open("w", encoding="utf-8") as f:
            for r in new_episodes:
                f.write(
                    json.dumps(
                        {
                            "episode_index": r["episode_index"],
                            "tasks": r.get("tasks") or [],
                            "length": r["length"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        # Drop stale per-episode stats; they will be inconsistent after reindex.
        stats = root / "meta" / "episodes_stats.jsonl"
        if stats.is_file():
            stats.unlink()

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_offset
    info["total_videos"] = len(new_episodes) * len(video_keys)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")

    meta_dir = root / "meta"
    for name in ("vision_episode_allowlist.json", "all_episode_allowlist.json", "allowlist.json"):
        update_allowlist(meta_dir / name, old_to_new, delete_eps)

    issues_path = root / ".sonic_qa_issues.json"
    if issues_path.is_file():
        issues = json.loads(issues_path.read_text("utf-8"))
        for ep in list(issues.get("issues", {})):
            if int(ep) in delete_eps:
                issues["issues"].pop(ep, None)
            elif int(ep) in old_to_new:
                note = issues["issues"].pop(ep)
                issues["issues"][str(old_to_new[int(ep)])] = note
        issues["pruned_at"] = stamp
        issues["deleted_episodes"] = sorted(delete_eps)
        issues_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2) + "\n", "utf-8")

    shutil.rmtree(staging, ignore_errors=True)
    # Do not leave Studio/prune scratch inside the dataset tree.
    shutil.rmtree(backup, ignore_errors=True)
    report.pop("backup_dir", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune LeRobot v2.1/v3 episodes and reindex")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--delete", type=int, nargs="+", required=True, help="Episode indices to delete")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prune_dataset(Path(args.root), set(args.delete), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
