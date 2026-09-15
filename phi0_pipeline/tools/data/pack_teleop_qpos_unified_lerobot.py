#!/usr/bin/env python3
"""Pack multi-session teleop LeRobot → Phi0 unified 512-d train set (V3 + dual video).

Input: session dirs under ``--raw-root`` + valid-episode JSON ``--manifest``
(session_id → {valid:[ep,...]}). No SMPL / no GMR retarget.

``[360:396)`` from teleop qpos:
  - dof29 ← ``observation.state`` (legs+waist+arms)
  - quat ← ``observation.root_orientation``
  - xyz ← ``(0, 0, STAND_Z)`` every frame (dataset has no ``base_trans``)
  - disk root via ``Δxy + abs z + abs quat`` (``apply_gmr_qpos_tail``)

``[0:315)`` / ``[346:360)`` Dex3 unused (zero).
``[396:460)`` ← raw ``action.motion_token`` (record-time Sonic GT; not onnx-reencoded).
Gravity ``[460:463)`` from ``observation.projected_gravity``.
Revo2 / 强脑手 ``[463:475)`` ← left6+right6 ``action.revo2.*.position`` (rest of
``reserved_49`` stays 0).

``--patch-sonic``: in-place rewrite of an existing layout (parquets + stats only;
videos untouched).

Videos: hardlink/copy ``ego_view`` + ``left_wrist`` (misnamed chest-forward cam)
from raw sessions into ``videos/chunk-*/{video_key}/episode_{out_ep:06d}.mp4``
(episode index remapped). Batch/VLM alias for the second view is ``chest_forward``.
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_PHI810 = Path(__file__).resolve().parents[2]
_SCRIPTS_DATA = Path(__file__).resolve().parent
for p in (_SCRIPTS_DATA, _PHI810 / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from egypt_clip_root_layout import (  # noqa: E402
    apply_gmr_qpos_tail,
    assert_root_layout_consistent,
)

D_UNIFIED = 512
QPOS_SLICES = [(0, 15), (15, 22), (29, 36)]  # legs+waist, larm, rarm -> 29
STAND_Z = 0.785
CHUNKS_SIZE = 1000
FPS = 50.0
SONIC_SLICE = slice(396, 460)  # action.motion_token → sonic_motion_token_64
# reserved_49 = [463:512]; pack Revo2 L6+R6 at the front
REVO2_SLICE = slice(463, 475)
VIDEO_KEYS = (
    "observation.images.ego_view",
    "observation.images.left_wrist",
)
# Source teleop cameras (480x640 h264 @ 50fps); see 810demo/*/meta/info.json
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
VIDEO_PATH_TMPL = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
# Fixed language prompt for all 810demo episodes (VLA / distill task text).
TASK_PROMPT = (
    "机器人从第一人称视角移动至桌面，抓取并操作篮中的布艺玩具，"
    "随后提起并搬运整个篮子，移动至另一工作区域，"
    "与坐着的人员完成一次物品递交交互，最后离开操作区域。"
)


def take_slices(vec: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([vec[a:b] for a, b in slices])


def body_dof29_from_wbc43(wbc43: np.ndarray) -> np.ndarray:
    vec = np.asarray(wbc43, dtype=np.float32).reshape(43)
    return take_slices(vec, QPOS_SLICES).astype(np.float32)


def _revo2_12(row: pd.Series) -> np.ndarray:
    left = np.asarray(row["action.revo2.left.position"], dtype=np.float32).reshape(6)
    right = np.asarray(row["action.revo2.right.position"], dtype=np.float32).reshape(6)
    return np.concatenate([left, right]).astype(np.float32)


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = (q / n).astype(np.float32)
    if out[0] < 0:
        out = -out
    return out


def _load_manifest(path: Path) -> list[tuple[str, int]]:
    man = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for session_id, info in man.items():
        for ep in info.get("valid", []):
            out.append((session_id, int(ep)))
    return out


def _motion_token64(row: pd.Series) -> np.ndarray:
    if "action.motion_token" not in row.index:
        raise KeyError("action.motion_token missing in raw parquet")
    tok = np.asarray(row["action.motion_token"], dtype=np.float32).reshape(-1)
    if tok.shape != (64,):
        raise ValueError(f"motion_token dim {tok.shape} != (64,)")
    if not np.isfinite(tok).all():
        raise ValueError("non-finite action.motion_token")
    return tok


def _state_wbc43(row: pd.Series) -> np.ndarray:
    """obs.state is wbc43, or 61 = wbc43 + 18 dex3 touch (830demo+)."""
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
    if state.size == 61:
        state = state[:43]
    elif state.size != 43:
        raise ValueError(f"observation.state dim {state.size} not in {{43,61}}")
    return state


def _gripper_14(row: pd.Series) -> np.ndarray:
    from phi0.data.wbc43_io import resolve_hands

    # prefer state hands; teleop only if state degenerate (830demo teleop often all-zero)
    return resolve_hands(row, _state_wbc43(row))


def pack_episode(
    df: pd.DataFrame, *, stand_z: float, hand_mode: str = "revo2"
) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    q36 = np.zeros((n, 36), dtype=np.float32)
    unified = np.zeros((n, D_UNIFIED), dtype=np.float32)
    dim_mask = np.zeros((n, D_UNIFIED), dtype=bool)
    tokens = np.zeros((n, 64), dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]
        state = _state_wbc43(row)
        if not np.isfinite(state).all():
            raise ValueError(f"non-finite observation.state at frame {i}")
        quat = _norm_quat(row["observation.root_orientation"])
        q36[i, 0:3] = (0.0, 0.0, stand_z)
        q36[i, 3:7] = quat
        q36[i, 7:36] = body_dof29_from_wbc43(state)
        tokens[i] = _motion_token64(row)

        if hand_mode == "dex3":
            grip = _gripper_14(row)
            if not np.isfinite(grip).all():
                raise ValueError(f"non-finite dex3 gripper at frame {i}")
            unified[i, 346:360] = grip
            dim_mask[i, 346:360] = True
        else:
            revo = _revo2_12(row)
            if not np.isfinite(revo).all():
                raise ValueError(f"non-finite revo2 at frame {i}")
            unified[i, REVO2_SLICE] = revo
            dim_mask[i, REVO2_SLICE] = True

        if "observation.projected_gravity" in df.columns:
            g = np.asarray(row["observation.projected_gravity"], dtype=np.float32).reshape(3)
            unified[i, 460:463] = g
            dim_mask[i, 460:463] = True

    # apply_gmr zeros sonic; assert_root requires zero sonic — write token *after*.
    unified, dim_mask, _, _ = apply_gmr_qpos_tail(
        unified, q36_abs=q36, dim_mask=dim_mask, check_roundtrip=True
    )
    assert dim_mask is not None
    assert_root_layout_consistent(unified)
    unified[:, SONIC_SLICE] = tokens
    dim_mask[:, SONIC_SLICE] = True
    return unified, dim_mask


def _acc_unified_stats(out: Path) -> dict[str, Any]:
    """Mask-aware mean/std/min/max for ``action.unified`` over all layout parquets."""

    class _Acc:
        def __init__(self) -> None:
            self.count = np.zeros(D_UNIFIED, dtype=np.float64)
            self.sum = np.zeros(D_UNIFIED, dtype=np.float64)
            self.sumsq = np.zeros(D_UNIFIED, dtype=np.float64)
            self.min = np.full(D_UNIFIED, np.inf, dtype=np.float64)
            self.max = np.full(D_UNIFIED, -np.inf, dtype=np.float64)

        def add(self, x: np.ndarray, mask: np.ndarray) -> None:
            m = mask.astype(np.float64, copy=False)
            self.count += m.sum(0)
            xm = x * m
            self.sum += xm.sum(0)
            self.sumsq += (x * xm).sum(0)
            self.min = np.minimum(self.min, np.where(mask, x, np.inf).min(0))
            self.max = np.maximum(self.max, np.where(mask, x, -np.inf).max(0))

        def finalize(self) -> dict[str, list[float]]:
            c = np.maximum(self.count, 1.0)
            mean = self.sum / c
            var = np.maximum(self.sumsq / c - mean * mean, 0.0)
            std = np.sqrt(var)
            dead = self.count < 1.0
            mean = mean.copy()
            std = std.copy()
            mn = self.min.copy()
            mx = self.max.copy()
            mean[dead] = 0.0
            std[dead] = 0.0
            mn[dead] = 0.0
            mx[dead] = 0.0
            return {
                "mean": mean.astype(np.float64).tolist(),
                "std": std.astype(np.float64).tolist(),
                "min": mn.astype(np.float64).tolist(),
                "max": mx.astype(np.float64).tolist(),
                "count": self.count.astype(np.float64).tolist(),
            }

    def _fsl(col, width: int) -> np.ndarray:
        arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
        flat = arr.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float64).reshape(len(arr), width)

    def _bool_fsl(col, width: int) -> np.ndarray:
        arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
        flat = arr.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.bool_).reshape(len(arr), width)

    acc = _Acc()
    files = sorted((out / "data").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data/**/file-*.parquet under {out}")
    for path in files:
        table = pq.read_table(path, columns=["action.unified", "action.dim_mask"])
        u = _fsl(table.column("action.unified"), D_UNIFIED)
        m = _bool_fsl(table.column("action.dim_mask"), D_UNIFIED)
        acc.add(u, m)
    return {"action.unified": acc.finalize()}


def patch_sonic_into_layout(
    *,
    raw: Path,
    out: Path,
    manifest: Path,
    stand_z: float = STAND_Z,
) -> dict[str, Any]:
    """Rewrite sonic slice + dim_mask from raw motion_token; refresh stats.json."""
    del stand_z  # pack_episode still needs stand_z only on full pack
    meta_path = out / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sources = meta.get("sources")
    if not sources:
        # fall back to manifest order (out_ep = enumerate order)
        episodes = _load_manifest(manifest)
        sources = [
            {
                "out_episode_index": i,
                "session": sid,
                "source_episode_index": ep,
            }
            for i, (sid, ep) in enumerate(episodes)
        ]

    n_patched = 0
    sonic_abs = 0.0
    sonic_n = 0
    for row in sources:
        out_ep = int(row["out_episode_index"])
        session_id = str(row["session"])
        src_ep = int(row["source_episode_index"])
        src_pq = raw / session_id / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
        chunk = out_ep // CHUNKS_SIZE
        file_i = out_ep % CHUNKS_SIZE
        out_pq = out / f"data/chunk-{chunk:03d}/file-{file_i:03d}.parquet"
        if not src_pq.is_file():
            raise FileNotFoundError(src_pq)
        if not out_pq.is_file():
            raise FileNotFoundError(out_pq)
        df = pd.read_parquet(src_pq)
        tokens = np.stack([_motion_token64(df.iloc[i]) for i in range(len(df))])
        table = pq.read_table(out_pq)
        names = table.column_names
        cols = {n: table.column(n) for n in names}
        u = np.asarray(
            cols["action.unified"].combine_chunks().values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(len(table), D_UNIFIED)
        m = np.asarray(
            cols["action.dim_mask"].combine_chunks().values.to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        ).reshape(len(table), D_UNIFIED)
        if u.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"ep{out_ep}: layout T={u.shape[0]} != raw T={tokens.shape[0]}"
            )
        u = u.copy()
        m = m.copy()
        u[:, SONIC_SLICE] = tokens
        m[:, SONIC_SLICE] = True
        sonic_abs += float(np.abs(tokens).sum())
        sonic_n += int(tokens.size)
        new_cols = dict(cols)
        new_cols["action.unified"] = _fixed_list(u, pa.float32())
        new_cols["action.dim_mask"] = _fixed_list(m, pa.bool_())
        pq.write_table(pa.table(new_cols), out_pq, compression="zstd")
        n_patched += 1
        print(
            f"[810demo→egypt patch] ep{out_ep:03d} sonic absmean="
            f"{float(np.abs(tokens).mean()):.4f}",
            flush=True,
        )

    stats = _acc_unified_stats(out)
    # Preserve non-unified keys if present.
    stats_path = out / "meta" / "stats.json"
    if stats_path.is_file():
        prev = json.loads(stats_path.read_text(encoding="utf-8"))
        prev["action.unified"] = stats["action.unified"]
        stats_path.write_text(json.dumps(prev, indent=2) + "\n", encoding="utf-8")
    else:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    meta["qpos_note"] = (
        "dof29+quat from observation.state / root_orientation; "
        f"root xyz fixed (0,0,{STAND_Z}) — no observation.base_trans; "
        "disk [360:367]=Δxy+abs z+abs quat via apply_gmr_qpos_tail; "
        "no SMPL, no GMR retarget; [396:460]=raw action.motion_token; "
        "[346:360] Dex3 unused"
    )
    meta["sonic_motion_token"] = {
        "slice": [396, 460],
        "source": "raw action.motion_token",
        "note": "copied as-is; not onnx-reencoded",
        "patched_episodes": n_patched,
        "abs_mean": (sonic_abs / max(sonic_n, 1)),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta["sonic_motion_token"]


def patch_hand_into_layout(
    *,
    raw: Path,
    out: Path,
    manifest: Path,
    hand_mode: str = "dex3",
) -> dict[str, Any]:
    """Rewrite hand slice + dim_mask from raw teleop; refresh stats.json."""
    mode = str(hand_mode).strip().lower()
    if mode not in ("dex3", "revo2"):
        raise ValueError(f"hand_mode must be dex3|revo2, got {hand_mode!r}")
    meta_path = out / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sources = meta.get("sources")
    if not sources:
        episodes = _load_manifest(manifest)
        sources = [
            {
                "out_episode_index": i,
                "session": sid,
                "source_episode_index": ep,
            }
            for i, (sid, ep) in enumerate(episodes)
        ]

    grip_slice = slice(346, 360) if mode == "dex3" else REVO2_SLICE
    n_patched = 0
    hand_abs = 0.0
    hand_n = 0
    for row in sources:
        out_ep = int(row["out_episode_index"])
        session_id = str(row["session"])
        src_ep = int(row["source_episode_index"])
        src_pq = raw / session_id / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
        chunk = out_ep // CHUNKS_SIZE
        file_i = out_ep % CHUNKS_SIZE
        out_pq = out / f"data/chunk-{chunk:03d}/file-{file_i:03d}.parquet"
        if not src_pq.is_file():
            raise FileNotFoundError(src_pq)
        if not out_pq.is_file():
            raise FileNotFoundError(out_pq)
        df = pd.read_parquet(src_pq)
        if mode == "dex3":
            hands = np.stack([_gripper_14(df.iloc[i]) for i in range(len(df))])
        else:
            hands = np.stack([_revo2_12(df.iloc[i]) for i in range(len(df))])
        table = pq.read_table(out_pq)
        names = table.column_names
        cols = {n: table.column(n) for n in names}
        u = np.asarray(
            cols["action.unified"].combine_chunks().values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(len(table), D_UNIFIED)
        m = np.asarray(
            cols["action.dim_mask"].combine_chunks().values.to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        ).reshape(len(table), D_UNIFIED)
        if u.shape[0] != hands.shape[0]:
            raise ValueError(
                f"ep{out_ep}: layout T={u.shape[0]} != raw T={hands.shape[0]}"
            )
        u = u.copy()
        m = m.copy()
        u[:, grip_slice] = hands
        m[:, grip_slice] = True
        hand_abs += float(np.abs(hands).sum())
        hand_n += int(hands.size)
        new_cols = dict(cols)
        new_cols["action.unified"] = _fixed_list(u, pa.float32())
        new_cols["action.dim_mask"] = _fixed_list(m, pa.bool_())
        pq.write_table(pa.table(new_cols), out_pq, compression="zstd")
        n_patched += 1
        print(
            f"[patch-hand] ep{out_ep:03d} mode={mode} absmean="
            f"{float(np.abs(hands).mean()):.4f}",
            flush=True,
        )

    stats = _acc_unified_stats(out)
    stats_path = out / "meta" / "stats.json"
    if stats_path.is_file():
        prev = json.loads(stats_path.read_text(encoding="utf-8"))
        prev["action.unified"] = stats["action.unified"]
        stats_path.write_text(json.dumps(prev, indent=2) + "\n", encoding="utf-8")
    else:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    meta["hand_mode"] = mode
    if mode == "dex3":
        meta["dex3_gripper"] = {
            "slice": [346, 360],
            "source": "teleop.left/right_hand_joints or observation.state[22:29,36:43]",
            "patched_episodes": n_patched,
            "abs_mean": (hand_abs / max(hand_n, 1)),
        }
        meta["qpos_note"] = (
            "dof29+quat from observation.state / root_orientation; "
            f"root xyz fixed (0,0,{STAND_Z}); "
            "[346:360]=Dex3 gripper14; [396:460]=raw action.motion_token"
        )
    else:
        meta["revo2_hand"] = {
            "slice": [463, 475],
            "layout": "action.revo2.left.position[6] + action.revo2.right.position[6]",
            "patched_episodes": n_patched,
            "abs_mean": (hand_abs / max(hand_n, 1)),
        }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"hand_mode": mode, "patched_episodes": n_patched}


def _fixed_list(arr: np.ndarray, dtype: pa.DataType) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(arr).reshape(-1), type=dtype),
        arr.shape[-1],
    )


def _link_or_copy(src: Path, dst: Path) -> str:
    """Prefer hardlink (same FS); fall back to copy. Returns mode used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _src_video(raw: Path, session_id: str, src_ep: int, video_key: str) -> Path:
    return (
        raw
        / session_id
        / "videos"
        / "chunk-000"
        / video_key
        / f"episode_{src_ep:06d}.mp4"
    )


def _dst_video(out: Path, out_ep: int, video_key: str) -> Path:
    chunk = out_ep // CHUNKS_SIZE
    return (
        out
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{out_ep:06d}.mp4"
    )


def pack_videos(
    *,
    raw: Path,
    out: Path,
    sources: list[dict[str, Any]],
) -> dict[str, int]:
    """Hardlink/copy ego_view + left_wrist mp4s under out/videos (remapped ep idx)."""
    counts = {k: 0 for k in VIDEO_KEYS}
    for row in sources:
        out_ep = int(row["out_episode_index"])
        session_id = str(row["session"])
        src_ep = int(row["source_episode_index"])
        for key in VIDEO_KEYS:
            src = _src_video(raw, session_id, src_ep, key)
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = _dst_video(out, out_ep, key)
            _link_or_copy(src, dst)
            counts[key] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=Path("/mnt/data2/wpy/workspace/810demo"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/810demo/810_demo.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/810demo_egypt_layout"),
    )
    ap.add_argument("--stand-z", type=float, default=STAND_Z)
    ap.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="Override TASK_PROMPT written to tasks.parquet / meta.json",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--patch-sonic",
        action="store_true",
        help="In-place: copy raw action.motion_token into existing layout + refresh stats",
    )
    ap.add_argument(
        "--patch-hand",
        action="store_true",
        help="In-place: rewrite hand slice (dex3|revo2) from raw teleop + refresh stats",
    )
    ap.add_argument(
        "--hand-mode",
        choices=("dex3", "revo2"),
        default="revo2",
        help="Pack/patch hand: dex3→unified[346:360], revo2→unified[463:475]",
    )
    args = ap.parse_args()

    raw = args.raw_root.resolve()
    out = args.out_dir.resolve()
    man = args.manifest.resolve()
    task_prompt = str(args.task_prompt).strip() if args.task_prompt else TASK_PROMPT
    if not task_prompt:
        raise SystemExit("--task-prompt must be non-empty")

    if bool(args.patch_sonic):
        info = patch_sonic_into_layout(
            raw=raw, out=out, manifest=man, stand_z=float(args.stand_z)
        )
        print(f"[810demo→egypt patch] done {info}", flush=True)
        return

    if bool(args.patch_hand):
        info = patch_hand_into_layout(
            raw=raw, out=out, manifest=man, hand_mode=str(args.hand_mode)
        )
        print(f"[patch-hand] done {info}", flush=True)
        return

    hand_mode = str(args.hand_mode).strip().lower()
    episodes = _load_manifest(man)
    if not episodes:
        raise SystemExit("manifest has no valid episodes")

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    lengths: list[int] = []
    sources: list[dict[str, Any]] = []
    global_index = 0
    for out_ep, (session_id, src_ep) in enumerate(episodes):
        src_pq = raw / session_id / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
        if not src_pq.is_file():
            raise FileNotFoundError(src_pq)
        df = pd.read_parquet(src_pq)
        unified, dim_mask = pack_episode(
            df, stand_z=float(args.stand_z), hand_mode=hand_mode
        )
        n = len(unified)
        # rewrite global index into a fresh table
        chunk = out_ep // CHUNKS_SIZE
        file_i = out_ep % CHUNKS_SIZE
        out_pq = out / f"data/chunk-{chunk:03d}/file-{file_i:03d}.parquet"
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "timestamp": pa.array(
                    np.arange(n, dtype=np.float32) / FPS, type=pa.float32()
                ),
                "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                "next.done": pa.array([i == n - 1 for i in range(n)], type=pa.bool_()),
                "episode_index": pa.array(
                    np.full(n, out_ep, dtype=np.int64), type=pa.int64()
                ),
                "task_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
                "index": pa.array(
                    np.arange(global_index, global_index + n, dtype=np.int64),
                    type=pa.int64(),
                ),
                "action.unified": _fixed_list(unified, pa.float32()),
                "action.dim_mask": _fixed_list(dim_mask, pa.bool_()),
            }
        )
        pq.write_table(table, out_pq, compression="zstd")
        lengths.append(n)
        sources.append(
            {
                "out_episode_index": out_ep,
                "session": session_id,
                "source_episode_index": src_ep,
                "length": n,
            }
        )
        global_index += n
        print(
            f"[810demo→egypt] {session_id} ep{src_ep:03d} -> ep{out_ep:03d} ({n} frames)",
            flush=True,
        )

    video_counts = pack_videos(raw=raw, out=out, sources=sources)
    n_video_files = int(sum(video_counts.values()))
    print(
        f"[810demo→egypt] videos: {video_counts} ({n_video_files} files)",
        flush=True,
    )

    total_frames = int(sum(lengths))
    total_eps = len(lengths)
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": task_prompt}]),
        meta_dir / "tasks.parquet",
    )

    ep_rows = []
    cursor = 0
    for i, n in enumerate(lengths):
        ep_rows.append(
            {
                "episode_index": i,
                "length": n,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + n,
                "tasks": [task_prompt],
                "data/chunk_index": i // CHUNKS_SIZE,
                "data/file_index": i % CHUNKS_SIZE,
            }
        )
        cursor += n
    ep_dir = meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), ep_dir / "file-000.parquet")
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

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": FPS,
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_video_files,
        "total_chunks": int(math.ceil(total_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": VIDEO_PATH_TMPL,
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # Disk key left_wrist is a misnomer (chest-mounted forward cam); batch alias chest_forward.
    (meta_dir / "modality.json").write_text(
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
                }
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "layout": "phi0_unified_teleop_qpos",
        "layout_note": (
            "512-d unified; root packing = Δxy+abs z+abs quat "
            "(historical egypt_smplsem_clip-compatible)"
        ),
        "qpos_source": "teleop_observation_state",
        "qpos_note": (
            "dof29+quat from observation.state / root_orientation; "
            f"root xyz fixed (0,0,{args.stand_z}) — no observation.base_trans; "
            "disk [360:367]=Δxy+abs z+abs quat via apply_gmr_qpos_tail; "
            "no SMPL, no GMR retarget; [396:460]=raw action.motion_token; "
            + (
                "[346:360]=Dex3 gripper14 from teleop"
                if hand_mode == "dex3"
                else "[346:360] Dex3 unused; [463:475]=Revo2"
            )
        ),
        "hand_mode": hand_mode,
        **(
            {
                "dex3_gripper": {
                    "slice": [346, 360],
                    "source": "teleop.left/right_hand_joints or observation.state hands",
                }
            }
            if hand_mode == "dex3"
            else {
                "revo2_hand": {
                    "slice": [463, 475],
                    "layout": "action.revo2.left.position[6] + action.revo2.right.position[6]",
                    "note": "packed into reserved_49; does not use Dex3 gripper [346:360]",
                }
            }
        ),
        "sonic_motion_token": {
            "slice": [396, 460],
            "source": "raw action.motion_token",
            "note": "copied as-is; not onnx-reencoded",
        },
        "videos": {
            "keys": list(VIDEO_KEYS),
            "path": VIDEO_PATH_TMPL,
            "counts": video_counts,
            "note": (
                "hardlink/copy from raw; left_wrist disk key is misnamed chest-forward; "
                "VLM labels Ego/Chest forward + add_vision_id"
            ),
        },
        "task_prompt": task_prompt,
        "stand_z": float(args.stand_z),
        "manifest": str(args.manifest.resolve()),
        "raw_root": str(raw),
        "episodes": total_eps,
        "frames": total_frames,
        "sources": sources,
        "train": {
            "REF_ROOT": str(out),
            "note": "not egypt GMR gold; teleop qpos pass-through in egypt unified slots",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    stats = _acc_unified_stats(out)
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[810demo→egypt] wrote {out}: {total_eps} eps / {total_frames} frames / "
        f"{n_video_files} videos",
        flush=True,
    )


if __name__ == "__main__":
    main()
