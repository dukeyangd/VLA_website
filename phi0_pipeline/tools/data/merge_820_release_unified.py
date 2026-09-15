#!/usr/bin/env python3
"""Merge 820demo *_release_unified packs into one mix root.

v1 order: demo5skill (no video) → skill_1 → skill_2 → skill_3.
v2 order: demo5skill → skill_1_walk → skill_2 → skill_3_place_basket_new →
skill_4_give_me_five → skill_5_pick_rubbish → skill_0.
Prompt table keeps task_index 5=walk / 6=skill_2 so DistillNorm / task_index stay aligned.
v3 order: demo5 idle+bow only → skill_1_walk → skill_2 → skill_3.
Compact task table (new mix; no v1/v2 DistillNorm resume).
v4 order (LL, no release reencode): skill_1_new → skill_2 → skill_3_place_basket_new.
Uses ``*_unified`` packs (raw ``action.motion_token``); no demo5.

Writes ``meta/episodes`` with ``has_video``, corrected Chinese prompts, hardlinked videos.
Does not touch efs or raw ``820demo/skill_N`` sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D_UNIFIED = 512
CHUNKS_SIZE = 1000
FPS = 50.0
VIDEO_KEYS = (
    "observation.images.ego_view",
    "observation.images.left_wrist",
)
VIDEO_PATH_TMPL = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
_VIDEO_FEATURE = {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channel"],
    "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": 50,
        "video.channels": 3,
        "has_audio": False,
    },
}

# skill123: keep short Chinese (dual-VLM + image pads must fit prompt_max_length=256).
SKILL_PROMPTS = {
    1: "机器人朝黑箱子走过去。",
    2: "走到桌面，抓取篮中玩具并搬运篮子到另一区域。",
    3: "拿起左边的篮子并移到另一工作区域。",
}

# mix v2: new place-basket / high-five / pick-rubbish (short Chinese, dual-VLM 256).
V2_SKILL_PROMPTS = {
    1: SKILL_PROMPTS[1],
    2: SKILL_PROMPTS[2],
    3: "把篮子放到指定位置。",
    4: "走上前和人击掌。",
    5: "捡起地上的垃圾并放入垃圾桶。",
    # id 6 (not 0): append after 1..5 so task_index 5–9 stay put for 22k resume.
    6: "旋转朝向镜头中的人",
}

# demo5skill is text-only (no image pads). Short Chinese; fits prompt_max_length=256.
# BoneSEED ep ids in the same order (idle, egypt, spin, wave, bow).
DEMO5_BONESEED_EPS = (291, 110825, 81, 28, 37561)
DEMO5_PROMPTS = [
    "静止假人：完全静止站立，双臂贴腿，零移动。",
    "埃及肚皮舞：髋部左右隔离摆动，胸腹波浪；肘直角、腕轻弹；原地脚尖点地。",
    "华丽单脚转：单脚绕竖直轴连转，双臂伸展，落地急停直立。",
    "告别挥手：站稳，单臂过头大幅挥手，另一臂垂髋侧。",
    "正式鞠躬：髋折腰低头后回直立，双手贴大腿。",
]
MIX_TASK_PROMPTS = [*DEMO5_PROMPTS, SKILL_PROMPTS[1], SKILL_PROMPTS[2], SKILL_PROMPTS[3]]
MIX_V2_TASK_PROMPTS = [
    *DEMO5_PROMPTS,
    V2_SKILL_PROMPTS[1],
    V2_SKILL_PROMPTS[2],
    V2_SKILL_PROMPTS[3],
    V2_SKILL_PROMPTS[4],
    V2_SKILL_PROMPTS[5],
    V2_SKILL_PROMPTS[6],
]

# mix v3: idle+bow only (demo5skill src ep 0, 4). Compact task_index 0..4.
V3_DEMO5_KEEP = (
    (0, DEMO5_PROMPTS[0]),  # idle
    (4, DEMO5_PROMPTS[4]),  # bow
)
V3_DEMO5_SRC_EPS = tuple(i for i, _ in V3_DEMO5_KEEP)
V3_DEMO5_PROMPTS = [p for _, p in V3_DEMO5_KEEP]
V3_SKILL_PROMPTS = {
    1: SKILL_PROMPTS[1],
    2: SKILL_PROMPTS[2],
    3: SKILL_PROMPTS[3],
}
MIX_V3_TASK_PROMPTS = [
    *V3_DEMO5_PROMPTS,
    V3_SKILL_PROMPTS[1],
    V3_SKILL_PROMPTS[2],
    V3_SKILL_PROMPTS[3],
]

# mix v4: vision-only teleop LL sonic (no demo5, no release reencode).
V4_SKILL_PROMPTS = {
    1: SKILL_PROMPTS[1],  # walk-to-box
    2: SKILL_PROMPTS[2],  # skill_2 basket toys
    3: V2_SKILL_PROMPTS[3],  # place basket short
}
MIX_V4_TASK_PROMPTS = [
    V4_SKILL_PROMPTS[1],
    V4_SKILL_PROMPTS[2],
    V4_SKILL_PROMPTS[3],
]
V4_SONIC_META = {
    "slice": [396, 460],
    "source": "raw action.motion_token",
    "policy": "low_latency",
    "note": (
        "Copied as-is from teleop record-time Sonic GT; "
        "not onnx-reencoded to release. Decode with DEPLOY_POLICY_DIR=low_latency."
    ),
}

# (origin_tag, dirname under 820demo, has_video, skill_id or None for demo5)
V1_PACK_SPECS = (
    ("demo5skill", "820demo_demo5skill_release_unified", False, None),
    ("skill_1", "820demo_skill_1_release_unified", True, 1),
    ("skill_2", "820demo_skill_2_release_unified", True, 2),
    ("skill_3", "820demo_skill_3_release_unified", True, 3),
)
V2_PACK_SPECS = (
    ("demo5skill", "820demo_demo5skill_release_unified", False, None),
    ("skill_1_walk", "820demo_skill_1_walk_release_unified", True, 1),
    ("skill_2", "820demo_skill_2_release_unified", True, 2),
    (
        "skill_3_place_basket_new",
        "820demo_skill_3_place_basket_new_release_unified",
        True,
        3,
    ),
    (
        "skill_4_give_me_five",
        "820demo_skill_4_give_me_five_release_unified",
        True,
        4,
    ),
    (
        "skill_5_pick_rubbish",
        "820demo_skill_5_pick_rubbish_release_unified",
        True,
        5,
    ),
    (
        "skill_0",
        "820demo_skill_0_release_unified",
        True,
        6,
    ),
)
# optional 5th field: include_src_eps (demo5skill idle+bow)
V3_PACK_SPECS = (
    (
        "demo5skill",
        "820demo_demo5skill_release_unified",
        False,
        None,
        V3_DEMO5_SRC_EPS,
    ),
    ("skill_1_walk", "820demo_skill_1_walk_release_unified", True, 1),
    ("skill_2", "820demo_skill_2_release_unified", True, 2),
    ("skill_3", "820demo_skill_3_release_unified", True, 3),
)
# LL intermediate packs (no *_release_unified)
V4_PACK_SPECS = (
    ("skill_1_new", "820demo_skill_1_new_unified", True, 1),
    ("skill_2", "820demo_skill_2_unified", True, 2),
    (
        "skill_3_place_basket_new",
        "820demo_skill_3_place_basket_new_unified",
        True,
        3,
    ),
)
DEFAULT_SOURCES = V1_PACK_SPECS


def _unpack_spec(
    spec: tuple,
) -> tuple[str, str, bool, int | None, tuple[int, ...] | None]:
    tag, dirname, has_video, skill_id = spec[0], spec[1], spec[2], spec[3]
    include = spec[4] if len(spec) > 4 else None
    include_eps = tuple(int(x) for x in include) if include is not None else None
    return str(tag), str(dirname), bool(has_video), skill_id, include_eps


def _ep_parquet(root: Path, ep: int) -> Path:
    chunk = int(ep) // CHUNKS_SIZE
    file_i = int(ep) % CHUNKS_SIZE
    return root / "data" / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.parquet"


def _video_path(root: Path, ep: int, key: str) -> Path:
    chunk = int(ep) // CHUNKS_SIZE
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / key
        / f"episode_{int(ep):06d}.mp4"
    )


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ep_length(root: Path, src_row: dict[str, Any], src_ep: int) -> int:
    if "length" in src_row and src_row["length"] is not None:
        return int(src_row["length"])
    path = _ep_parquet(root, src_ep)
    return int(pq.read_table(path, columns=["frame_index"]).num_rows)


def merge(
    *,
    workspace_820: Path,
    out_root: Path,
    overwrite: bool,
    pack_specs: tuple = V1_PACK_SPECS,
    skill_prompts: dict[int, str] | None = None,
    demo5_prompts: list[str] | None = None,
    mix_note: str | None = None,
    layout_note: str | None = None,
    keep_stats: bool = False,
    sonic_override: dict[str, Any] | None = None,
) -> None:
    workspace_820 = workspace_820.resolve()
    out_root = out_root.resolve()
    skill_prompts = dict(skill_prompts or SKILL_PROMPTS)
    demo5_prompts = list(demo5_prompts if demo5_prompts is not None else DEMO5_PROMPTS)
    mix_note = mix_note or (
        "demo5skill(no video)+skill_1/2/3(release sonic); "
        "has_video on meta/episodes; Chinese task prompts"
    )
    layout_note = layout_note or (
        "mix: demo5skill (no video) + skill_1/2/3 release sonic; "
        "has_video in meta/episodes"
    )
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_root} exists (pass --overwrite)")
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # task table: demo5 prompts (full 0..4, or v3 idle+bow), then skills in id order
    task_rows: list[dict[str, Any]] = []
    for i, p in enumerate(demo5_prompts):
        task_rows.append({"task_index": i, "task": p})
    skill_task_index: dict[int, int] = {}
    for sid in sorted(skill_prompts):
        skill_task_index[int(sid)] = len(task_rows)
        task_rows.append(
            {"task_index": skill_task_index[int(sid)], "task": skill_prompts[sid]}
        )

    out_sources: list[dict[str, Any]] = []
    ep_rows: list[dict[str, Any]] = []
    vision_eps: list[int] = []
    all_eps: list[int] = []
    video_counts = {k: 0 for k in VIDEO_KEYS}
    cursor = 0
    out_ep = 0
    sum_u = None
    sum_u2 = None
    n_frames = 0
    u_min = None
    u_max = None
    sonic_meta: dict[str, Any] | None = None

    for spec in pack_specs:
        tag, dirname, has_video, skill_id, include_src_eps = _unpack_spec(spec)
        in_root = workspace_820 / dirname
        if not in_root.is_dir():
            raise FileNotFoundError(in_root)
        meta_in = json.loads((in_root / "meta.json").read_text(encoding="utf-8"))
        if sonic_meta is None:
            if sonic_override is not None:
                sonic_meta = dict(sonic_override)
            else:
                sonic_meta = dict(meta_in.get("sonic_motion_token") or {})
        sources = list(meta_in.get("sources") or [])
        if not sources:
            raise RuntimeError(f"{in_root}: empty sources")

        for src_row in sources:
            src_ep = int(src_row.get("out_episode_index", src_row.get("source_episode_index", 0)))
            # Prefer pack-local out index when present (always for our packs).
            if "out_episode_index" in src_row:
                src_ep = int(src_row["out_episode_index"])
            length = _ep_length(in_root, src_row, src_ep)
            src_pq = _ep_parquet(in_root, src_ep)
            if not src_pq.is_file():
                raise FileNotFoundError(src_pq)
            if include_src_eps is not None and src_ep not in include_src_eps:
                continue

            if skill_id is None:
                # demo5skill: one prompt per kept episode
                if include_src_eps is not None:
                    task_index = include_src_eps.index(src_ep)
                else:
                    task_index = int(src_ep)  # 0..4 in source
                prompt = demo5_prompts[task_index]
            else:
                task_index = int(skill_task_index[skill_id])
                prompt = skill_prompts[skill_id]

            table = pq.read_table(src_pq)
            pdf = table.to_pandas()
            pdf["episode_index"] = out_ep
            pdf["task_index"] = task_index
            pdf["index"] = np.arange(cursor, cursor + length, dtype=np.int64)
            if "frame_index" in pdf.columns:
                pdf["frame_index"] = np.arange(length, dtype=np.int64)
            out_pq = _ep_parquet(out_root, out_ep)
            out_pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), out_pq)

            u = np.stack(pdf["action.unified"].to_numpy()).astype(np.float64)
            if u.ndim != 2 or u.shape[1] != D_UNIFIED:
                u = np.asarray(list(pdf["action.unified"]), dtype=np.float64).reshape(
                    -1, D_UNIFIED
                )
            if sum_u is None:
                sum_u = u.sum(axis=0)
                sum_u2 = (u * u).sum(axis=0)
                u_min = u.min(axis=0)
                u_max = u.max(axis=0)
            else:
                sum_u += u.sum(axis=0)
                sum_u2 += (u * u).sum(axis=0)
                u_min = np.minimum(u_min, u.min(axis=0))
                u_max = np.maximum(u_max, u.max(axis=0))
            n_frames += int(u.shape[0])

            if has_video:
                for key in VIDEO_KEYS:
                    src_v = _video_path(in_root, src_ep, key)
                    if not src_v.is_file():
                        raise FileNotFoundError(src_v)
                    dst_v = _video_path(out_root, out_ep, key)
                    _hardlink_or_copy(src_v, dst_v)
                    video_counts[key] += 1
                vision_eps.append(out_ep)

            ep_rows.append(
                {
                    "episode_index": out_ep,
                    "length": length,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + length,
                    "tasks": [prompt],
                    "instruction": prompt,
                    "task_index": int(task_index),
                    "has_video": bool(has_video),
                    "data/chunk_index": out_ep // CHUNKS_SIZE,
                    "data/file_index": out_ep % CHUNKS_SIZE,
                }
            )
            out_sources.append(
                {
                    "out_episode_index": out_ep,
                    "origin_tag": tag,
                    "origin_root": str(in_root),
                    "origin_episode_index": src_ep,
                    "session": src_row.get("session"),
                    "source_episode_index": src_row.get("source_episode_index"),
                    "length": length,
                    "has_video": bool(has_video),
                    "task_index": task_index,
                }
            )
            all_eps.append(out_ep)
            cursor += length
            out_ep += 1
            if out_ep % 50 == 0:
                print(f"[merge] out_ep={out_ep} frames={n_frames}", flush=True)

    total_eps = out_ep
    assert sum_u is not None and sum_u2 is not None and u_min is not None and u_max is not None
    mean = sum_u / max(n_frames, 1)
    var = sum_u2 / max(n_frames, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    pq.write_table(
        pa.Table.from_pylist(task_rows),
        out_root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    features = {
        "action.unified": {"dtype": "float32", "shape": [D_UNIFIED]},
        "action.dim_mask": {"dtype": "bool", "shape": [D_UNIFIED]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for key in VIDEO_KEYS:
        features[key] = dict(_VIDEO_FEATURE)

    n_video_files = sum(video_counts.values())
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": FPS,
        "total_episodes": total_eps,
        "total_frames": n_frames,
        "total_tasks": len(task_rows),
        "total_videos": n_video_files,
        "total_chunks": int(math.ceil(total_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": VIDEO_PATH_TMPL,
        "features": features,
        "layout": "phi0_unified_teleop_qpos",
        "sonic_encoder": sonic_meta or {},
        "mix_note": mix_note,
    }
    (out_root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (out_root / "meta" / "modality.json").write_text(
        json.dumps(
            {
                "video": {
                    "ego_view": {"original_key": "observation.images.ego_view"},
                    "left_wrist": {
                        "original_key": "observation.images.left_wrist",
                        "alias": "chest_forward",
                        "note": "misnamed disk key; physically a chest-mounted forward camera",
                    },
                    "chest_forward": {
                        "original_key": "observation.images.left_wrist",
                        "note": "batch/VLM alias for left_wrist disk key",
                    },
                },
                "has_video": {
                    "note": (
                        "per-episode bool in meta/episodes; false → VLM text-only "
                        "(reuse vision_dropout keep/drop+merge)"
                    )
                },
            },
            indent=4,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not keep_stats:
        stats = {
            "action.unified": {
                "mean": mean.astype(np.float64).tolist(),
                "std": std.astype(np.float64).tolist(),
                "min": u_min.astype(np.float64).tolist(),
                "max": u_max.astype(np.float64).tolist(),
            }
        }
        (out_root / "meta" / "stats.json").write_text(
            json.dumps(stats) + "\n", encoding="utf-8"
        )

    allow_vision = {"episode_index": vision_eps}
    allow_all = {"episode_index": all_eps}
    (out_root / "meta" / "vision_episode_allowlist.json").write_text(
        json.dumps(allow_vision, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "meta" / "all_episode_allowlist.json").write_text(
        json.dumps(allow_all, indent=2) + "\n", encoding="utf-8"
    )

    meta = {
        "layout": "phi0_unified_teleop_qpos",
        "layout_note": layout_note,
        "qpos_source": "mixed",
        "sonic_motion_token": sonic_meta or {},
        "videos": {
            "keys": list(VIDEO_KEYS),
            "path": VIDEO_PATH_TMPL,
            "counts": video_counts,
            "note": (
                "hardlink from skill packs; demo5skill has_video=false (no mp4); "
                "left_wrist misnamed chest-forward"
            ),
        },
        "task_prompts": [r["task"] for r in task_rows],
        "episodes": total_eps,
        "frames": n_frames,
        "sources": out_sources,
        "train": {
            "vision_episode_allowlist": "meta/vision_episode_allowlist.json",
            "all_episode_allowlist": "meta/all_episode_allowlist.json",
            "require_video_default": True,
        },
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[merge] wrote {out_root} eps={total_eps} frames={n_frames} "
        f"vision_eps={len(vision_eps)} videos={n_video_files}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workspace-820",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/820demo"),
    )
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--preset", choices=("v1", "v2", "v3", "v4"), default="v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep-stats",
        action="store_true",
        help="Do not rewrite meta/stats.json (caller restores a frozen file).",
    )
    args = p.parse_args()
    ws = args.workspace_820
    if args.preset == "v2":
        out = args.out_root or (ws / "820demo_mix_v2_release_unified")
        merge(
            workspace_820=ws,
            out_root=out,
            overwrite=bool(args.overwrite),
            pack_specs=V2_PACK_SPECS,
            skill_prompts=V2_SKILL_PROMPTS,
            mix_note=(
                "mix v2: demo5skill(no video)+skill_1_walk+skill_2+"
                "skill_3_place_basket_new+skill_4_give_me_five+"
                "skill_5_pick_rubbish+skill_0 "
                "(no old skill_1 walk-to-box; skill_0 is task_index 10)"
            ),
            layout_note=(
                "mix v2: demo5skill (no video) + skill_1_walk + skill_2 + "
                "place_basket_new + give_me_five + pick_rubbish + skill_0; "
                "skill_0 is task_index 10; has_video in meta/episodes"
            ),
            keep_stats=bool(args.keep_stats),
        )
        return
    if args.preset == "v3":
        out = args.out_root or (ws / "820demo_mix_v3_release_unified")
        merge(
            workspace_820=ws,
            out_root=out,
            overwrite=bool(args.overwrite),
            pack_specs=V3_PACK_SPECS,
            skill_prompts=V3_SKILL_PROMPTS,
            demo5_prompts=V3_DEMO5_PROMPTS,
            mix_note=(
                "mix v3: demo5skill idle+bow (no video)+skill_1_walk+skill_2+skill_3 "
                "(no egypt/spin/wave; no place_basket_new/give_me_five/pick_rubbish/skill_0)"
            ),
            layout_note=(
                "mix v3: demo5 idle+bow (no video) + skill_1_walk + skill_2 + skill_3; "
                "compact task_index 0..4; has_video in meta/episodes"
            ),
            keep_stats=bool(args.keep_stats),
        )
        return
    if args.preset == "v4":
        out = args.out_root or (ws / "820demo_v4_ll_unified")
        merge(
            workspace_820=ws,
            out_root=out,
            overwrite=bool(args.overwrite),
            pack_specs=V4_PACK_SPECS,
            skill_prompts=V4_SKILL_PROMPTS,
            demo5_prompts=[],
            mix_note=(
                "mix v4 LL: skill_1_new+skill_2+skill_3_place_basket_new "
                "(raw motion_token; no release reencode; no demo5)"
            ),
            layout_note=(
                "mix v4 LL: skill_1_new + skill_2 + place_basket_new; "
                "task_index 0..2; sonic=record-time low_latency tokens; "
                "has_video in meta/episodes"
            ),
            keep_stats=bool(args.keep_stats),
            sonic_override=V4_SONIC_META,
        )
        return
    out = args.out_root or (ws / "820demo_mix_release_unified")
    merge(
        workspace_820=ws,
        out_root=out,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
