#!/usr/bin/env python3
"""解歧映射 → 合并 SMPL pkl / protomotions qpos → LeRobot V3 蒸馏数据集（无 sonic latent）。

一个源 parquet 对应一个输出 parquet（文件级处理，共 24 个源文件）。

Pipeline (v3: motion-stem pkl + pack unified semantic):
  ProtoMotions episode meta (meta/episodes/**/*.parquet)
    → source_motion_file stem → smpl_filtered/{stem}.pkl（含 ``_M``；与机器人 retarget 同侧）
    → 后备: alignment_resolved.parquet ``smpl_path``
      （alignment 常给非 ``_M``，相对 qpos 会左右镜像，勿优先）

  alignment_resolved.parquet
    → 按 data_file 分组（对应 protomotions_g1_phi0 的 24 个 parquet）
    → 每文件: 筛选 usable episodes → 读源帧 → 读 SMPL pkl → 帧对齐
    → pose_aa + transl 差分 → write_smpl_semantic_from_pose_aa → unified[0:315)
      (contacts/tactile [315:346] 保留 Proto 原值；[346:396] qpos 不变)
    → 写出新列: action.smpl_* / observation.qpos_frame0
    → action.unified[396:460]=0, dim_mask[396:460]=False
    → 重映射 episode_index / task_index 为全局连续值

输出列:
  timestamp / frame_index / episode_index / index / task_index / next.done
  observation.qpos              [43]   ProtoMotions retarget 关节角
  observation.projected_gravity [3]
  observation.qpos_frame0       [36]   episode 第零帧 unified[360:396] = g1_body_qpos_36
  action.unified                [512]  pkl-packed SMPL(0:315)+contacts(315:346)+qpos(360:396)+sonic归零
  action.dim_mask               [512]  sonic dims = False
  action.smpl_pose_aa           [72]   SMPL pkl pose_aa
  action.smpl_transl            [3]    SMPL pkl transl
  action.smpl_joints            [72]   SMPL pkl smpl_joints (24×3 展平)
  action.smpl_pkl_source        str    "exact"(motion-stem，含_M) | "alignment"(后备)

Usage::

  conda run -n Phi-0-wpy python tools/data/build_boneseed_smpl_g1csv_aligned.py \\
    --alignment /mnt/efs_1/.../meta/smpl_to_protomotions_g1_phi0_alignment_resolved.parquet \\
    --src-lerobot /mnt/efs_1/.../bone_seed/train/protomotions_g1_phi0 \\
    --out /mnt/efs_1/.../bone_seed/train/smpl_g1csv_aligned_phi0 \\
    [--max-files 2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phi0.schema.unified_action_schema import (  # noqa: E402
    smpl_pose_aa_from_unified,
    write_smpl_semantic_from_pose_aa,
)

# ponytail: raise on first episode whose pack round-trip exceeds this (cheap per-ep check).
_PACK_BODY_MSE_TOL = 1e-3

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SONIC_LO, SONIC_HI     = 396, 460   # sonic token slice in action.unified [396:460]
QPOS36_LO, QPOS36_HI   = 360, 396   # g1_body_qpos_36 in action.unified [360:396]

USABLE_STATUSES = frozenset([
    "resolved_ambiguous_by_family_assignment",
    "matched_unique_by_frame",
])

DEFAULT_ALIGNMENT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/raw/"
    "gear_sonic_training/meta/smpl_to_protomotions_g1_phi0_alignment_resolved.parquet"
)
DEFAULT_SRC = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/protomotions_g1_phi0"
)
DEFAULT_OUT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_g1csv_aligned_phi0"
)
DEFAULT_SMPL_FILTERED = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/raw/"
    "gear_sonic_training/data/smpl_filtered"
)

# ---------------------------------------------------------------------------
# ProtoMotions episode meta → 精确 pkl 路径表
# ---------------------------------------------------------------------------

def build_exact_pkl_map(src_lerobot: Path, smpl_filtered: Path) -> dict[int, Path]:
    """episode_index → pkl matching ``source_motion_file`` stem (includes ``_M``).

    ProtoMotions ``*_M.motion`` retarget and ``*_M.pkl`` share handedness with
    robot qpos. ``alignment_resolved.smpl_path`` often points at the non-``_M``
    twin and will L/R-mirror vs qpos — use this map first.
    """
    meta_dir = src_lerobot / "meta" / "episodes"
    ep_map: dict[int, Path] = {}
    exact_count = 0
    miss_count  = 0

    for pq_file in sorted(meta_dir.rglob("*.parquet")):
        tbl = pq.read_table(pq_file, columns=["episode_index", "source_motion_file"])
        for ep_idx, motion_file in zip(
            tbl["episode_index"].to_pylist(),
            tbl["source_motion_file"].to_pylist(),
        ):
            if not motion_file:
                miss_count += 1
                continue
            stem = Path(str(motion_file)).stem          # e.g. "dancing_routine_V002_001__A421_M"
            pkl_path = smpl_filtered / f"{stem}.pkl"
            if pkl_path.exists():
                ep_map[int(ep_idx)] = pkl_path
                exact_count += 1
            else:
                miss_count += 1

    print(
        f"[pkl_map] exact={exact_count:,d}  missing(fallback)={miss_count:,d} "
        f"({100*exact_count/max(exact_count+miss_count,1):.1f}% exact)",
        flush=True,
    )
    return ep_map


# ---------------------------------------------------------------------------
# Arrow 工具函数
# ---------------------------------------------------------------------------

def _fsl_np(col: pa.Array | pa.ChunkedArray, width: int, dtype) -> np.ndarray:
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    flat = arr.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=dtype).reshape(len(arr), width)


def _np_fsl(mat: np.ndarray, pa_dtype: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(mat.reshape(-1), type=pa_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, mat.shape[1])


def _plain(col: pa.Array | pa.ChunkedArray) -> pa.Array:
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


# ---------------------------------------------------------------------------
# SMPL pkl 读取与帧对齐
# ---------------------------------------------------------------------------

def load_smpl_pkl(path: str | Path) -> dict[str, np.ndarray]:
    data = joblib.load(path)
    return {
        "pose_aa":     np.asarray(data["pose_aa"],     dtype=np.float32),
        "transl":      np.asarray(data["transl"],      dtype=np.float32),
        "smpl_joints": np.asarray(data["smpl_joints"], dtype=np.float32),
    }


def align_smpl_frames(
    smpl: dict[str, np.ndarray],
    frame_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """按 frame_index 映射 SMPL 帧（超出末尾则取末帧）。"""
    t = int(smpl["pose_aa"].shape[0])
    idx = np.minimum(frame_indices.astype(np.int64), t - 1)
    return {
        "action.smpl_pose_aa": smpl["pose_aa"][idx],
        "action.smpl_transl":  smpl["transl"][idx],
        "action.smpl_joints":  smpl["smpl_joints"][idx].reshape(-1, 72),
    }


def pack_unified_smpl_semantic_for_episode(
    rows: np.ndarray,
    frame_indices: np.ndarray,
    *,
    u_all: np.ndarray,
    m_all: np.ndarray,
    smpl_pose_aa: np.ndarray,
    smpl_transl: np.ndarray,
) -> float:
    """Pack pkl pose_aa + transl deltas into unified[0:315) for one episode. Returns mean body aa MSE."""
    order = np.argsort(frame_indices)
    rs = rows[order]
    transl = smpl_transl[rs]
    pose = smpl_pose_aa[rs]
    rtl = np.zeros_like(transl)
    if len(rs) > 1:
        rtl[1:] = transl[1:] - transl[:-1]
    for j, r in enumerate(rs):
        write_smpl_semantic_from_pose_aa(
            u_all[r], pose[j], rtl[j], dim_mask=m_all[r]
        )
    # ponytail: spot-check 3 frames/ep not all T (2M×decode/file is too slow).
    check_j = [0]
    if len(rs) > 2:
        check_j.append(len(rs) // 2)
        check_j.append(len(rs) - 1)
    body_err: list[float] = []
    for j in check_j:
        got = smpl_pose_aa_from_unified(u_all[rs[j]]).reshape(-1)
        body_err.append(float(np.mean((got - pose[j, 3:66]) ** 2)))
    return float(np.mean(body_err)) if body_err else 0.0


# ---------------------------------------------------------------------------
# 单文件处理核心
# ---------------------------------------------------------------------------

def process_file(
    src_pq: Path,
    *,
    file_align: pd.DataFrame,
    ep_base: int,
    task_base: int,
    idx_base: int,
    exact_pkl_map: dict[int, Path],   # episode_index → exact pkl (精确匹配)
) -> tuple[pa.Table | None, pd.DataFrame, int, int, int]:
    """处理单个源 parquet，返回 (output_table, tasks_df, next_ep, next_task, next_idx)。"""

    usable_eps = set(file_align["episode_index"].astype(int))
    src = pq.read_table(src_pq)
    src_ep = np.asarray(src.column("episode_index"), dtype=np.int64)
    row_mask = np.isin(src_ep, list(usable_eps))
    if not row_mask.any():
        return None, pd.DataFrame(), ep_base, task_base, idx_base

    src = src.filter(row_mask.tolist())
    src_ep = src_ep[row_mask]
    n = src.num_rows

    smpl_pose_aa = np.zeros((n, 72), dtype=np.float32)
    smpl_transl  = np.zeros((n,  3), dtype=np.float32)
    smpl_joints  = np.zeros((n, 72), dtype=np.float32)
    qpos_frame0  = np.zeros((n, 36), dtype=np.float32)
    smpl_source  = np.empty(n, dtype=object)   # "exact" | "fallback" | ""
    smpl_source[:] = ""

    u_all = _fsl_np(src.column("action.unified"), 512, np.float32).copy()
    m_all = _fsl_np(src.column("action.dim_mask"), 512, np.bool_).copy()
    u_all[:, SONIC_LO:SONIC_HI] = 0.0
    m_all[:, SONIC_LO:SONIC_HI] = False

    fi_arr = np.asarray(src.column("frame_index"), dtype=np.int64)
    ep_rows: dict[int, list[int]] = {}
    for r, ep in enumerate(src_ep):
        ep_rows.setdefault(int(ep), []).append(r)

    ep_remap: dict[int, int] = {}
    task_rows: list[dict] = []
    smpl_errors = 0
    n_exact = 0
    n_alignment = 0
    n_packed = 0
    pack_mse_sum = 0.0

    for row in file_align.sort_values("episode_index").itertuples():
        old_ep = int(row.episode_index)
        if old_ep not in ep_rows:
            continue

        rows  = np.array(ep_rows[old_ep], dtype=np.int64)
        frames = fi_arr[rows]

        f0_mask = frames == 0
        f0_row  = rows[f0_mask][0] if f0_mask.any() else rows[0]
        qpos_frame0[rows] = u_all[f0_row, QPOS36_LO:QPOS36_HI]

        # --- motion-stem pkl 优先（含 _M，与机器人 qpos 同侧）；alignment 仅后备 ---
        exact_pkl = exact_pkl_map.get(old_ep)
        align_pkl = Path(str(row.smpl_path)) if getattr(row, "smpl_path", None) else None
        if exact_pkl is not None and exact_pkl.is_file():
            pkl_path = exact_pkl
            src_label = "exact"
            n_exact += 1
        elif align_pkl is not None and align_pkl.is_file():
            pkl_path = align_pkl
            src_label = "alignment"
            n_alignment += 1
        else:
            raise FileNotFoundError(f"ep={old_ep}: no motion-stem pkl and no alignment smpl_path")

        try:
            smpl = load_smpl_pkl(pkl_path)
            aligned = align_smpl_frames(smpl, frames)
            smpl_pose_aa[rows] = aligned["action.smpl_pose_aa"]
            smpl_transl[rows]  = aligned["action.smpl_transl"]
            smpl_joints[rows]  = aligned["action.smpl_joints"]
            smpl_source[rows]  = src_label
            mse = pack_unified_smpl_semantic_for_episode(
                rows,
                frames,
                u_all=u_all,
                m_all=m_all,
                smpl_pose_aa=smpl_pose_aa,
                smpl_transl=smpl_transl,
            )
            n_packed += 1
            pack_mse_sum += mse
            if mse >= _PACK_BODY_MSE_TOL:
                raise AssertionError(
                    f"ep={old_ep} pack body aa round-trip mse={mse:.2e} >= {_PACK_BODY_MSE_TOL}"
                )
        except AssertionError:
            raise
        except Exception as exc:
            smpl_errors += 1
            smpl_source[rows] = "error"
            print(f"    [WARN] pkl fail ep={old_ep} ({src_label}): {exc}", flush=True)

        new_ep = ep_base + len(ep_remap)
        ep_remap[old_ep] = new_ep
        task_rows.append({
            "task_index":           task_base + len(task_rows),
            "task":                 str(row.task),
            "task_en":              str(row.task_en),
            "source_motion_key":    str(row.source_motion_key),
            "source_episode_index": old_ep,
            "smpl_relative_path":   str(row.smpl_relative_path),
            "smpl_exact_pkl":       str(exact_pkl) if exact_pkl else "",
            "alignment_status":     str(row.alignment_status),
            "smpl_pkl_source":      src_label,
        })

    if not ep_remap:
        return None, pd.DataFrame(), ep_base, task_base, idx_base

    valid = np.array([int(e) in ep_remap for e in src_ep], dtype=bool)
    if not valid.all():
        src          = src.filter(valid.tolist())
        src_ep       = src_ep[valid]
        u_all        = u_all[valid]
        m_all        = m_all[valid]
        smpl_pose_aa = smpl_pose_aa[valid]
        smpl_transl  = smpl_transl[valid]
        smpl_joints  = smpl_joints[valid]
        qpos_frame0  = qpos_frame0[valid]
        smpl_source  = smpl_source[valid]
        fi_arr       = fi_arr[valid]
        n            = src.num_rows

    new_ep_col   = np.vectorize(lambda e: ep_remap[int(e)], otypes=[np.int64])(src_ep)
    new_task_col = new_ep_col.copy()
    new_idx_col  = np.arange(n, dtype=np.int64) + idx_base

    n_eps_out = len(ep_remap)
    next_ep   = ep_base   + n_eps_out
    next_task = task_base + n_eps_out
    next_idx  = idx_base  + n

    SKIP = {"episode_index", "task_index", "index", "action.unified", "action.dim_mask"}
    out_arrays: list[pa.Array] = []
    out_names:  list[str]      = []

    for col_name in src.column_names:
        if col_name in SKIP:
            continue
        out_arrays.append(_plain(src.column(col_name)))
        out_names.append(col_name)

    qpos_43_out: np.ndarray | None = None
    if "observation.qpos" in src.column_names:
        qpos_43_out = _fsl_np(src.column("observation.qpos"), 43, np.float32)

    out_arrays += [
        pa.array(new_ep_col,   type=pa.int64()),
        pa.array(new_task_col, type=pa.int64()),
        pa.array(new_idx_col,  type=pa.int64()),
        _np_fsl(u_all,        pa.float32()),
        _np_fsl(m_all,        pa.bool_()),
        _np_fsl(smpl_pose_aa, pa.float32()),
        _np_fsl(smpl_transl,  pa.float32()),
        _np_fsl(smpl_joints,  pa.float32()),
        _np_fsl(qpos_frame0,  pa.float32()),
        pa.array(smpl_source.tolist(), type=pa.string()),
    ]
    out_names += [
        "episode_index", "task_index", "index",
        "action.unified", "action.dim_mask",
        "action.smpl_pose_aa", "action.smpl_transl", "action.smpl_joints",
        "observation.qpos_frame0",
        "action.smpl_pkl_source",
    ]
    if qpos_43_out is not None:
        out_arrays.append(_np_fsl(qpos_43_out, pa.float32()))
        out_names.append("action.qpos_g1")

    out_table = pa.Table.from_arrays(out_arrays, names=out_names)
    tasks_df  = pd.DataFrame(task_rows)

    if smpl_errors:
        print(f"    [INFO] {smpl_errors} episodes pkl failed (零列保留)", flush=True)
    print(
        f"      pkl: alignment={n_alignment} exact_stem={n_exact} packed={n_packed} err={smpl_errors}"
        + (
            f" pack_mse~{pack_mse_sum / max(n_packed, 1):.1e}"
            if n_packed
            else ""
        ),
        flush=True,
    )
    return out_table, tasks_df, next_ep, next_task, next_idx


# ---------------------------------------------------------------------------
# Meta 写出
# ---------------------------------------------------------------------------

def write_tasks(dst: Path, tasks_df: pd.DataFrame) -> None:
    out = dst / "meta" / "tasks.parquet"
    tasks_df.to_parquet(out, index=False, engine="pyarrow", compression="zstd")
    print(f"[meta] tasks.parquet → {len(tasks_df):,d} tasks", flush=True)


def write_info(dst: Path, *, total_episodes: int, total_frames: int) -> None:
    info = {
        "fps": 50.0,
        "total_episodes": total_episodes,
        "total_frames":   total_frames,
        "dataset_type":   "smpl_g1csv_aligned_phi0_v3",
        "description": (
            "BoneSEED SMPL + protomotions_g1 qpos (v3): motion-stem pkl preferred "
            "(includes _M, matches robot retarget handedness); alignment_resolved "
            "smpl_path only as fallback. pose_aa packed into unified[0:315). Sonic zeroed."
        ),
        "features": {
            "observation.qpos":              {"dtype": "float32", "shape": [43]},
            "observation.projected_gravity": {"dtype": "float32", "shape": [3]},
            "observation.qpos_frame0": {
                "dtype": "float32", "shape": [36],
                "description": (
                    "Episode frame-0 g1_body_qpos_36 = unified[360:396]: "
                    "[root_xyz(3), root_quat_wxyz(4), dof29(29)]. "
                    "Initial state anchor for offline distillation."
                ),
            },
            "action.unified":           {"dtype": "float32", "shape": [512]},
            "action.dim_mask":          {"dtype": "bool",    "shape": [512]},
            "action.smpl_pose_aa":      {"dtype": "float32", "shape": [72]},
            "action.smpl_transl":       {"dtype": "float32", "shape": [3]},
            "action.smpl_joints":       {"dtype": "float32", "shape": [72]},
            "action.smpl_pkl_source":   {"dtype": "string",  "shape": [],
                                         "description": "'alignment'=alignment_resolved smpl_path (preferred); 'exact'=motion-stem fallback"},
        },
    }
    (dst / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[meta] info.json 写出 ({total_episodes} eps, {total_frames:,d} frames)", flush=True)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alignment",      type=Path, default=DEFAULT_ALIGNMENT)
    p.add_argument("--src-lerobot",    type=Path, default=DEFAULT_SRC, dest="src_lerobot")
    p.add_argument("--smpl-filtered",  type=Path, default=DEFAULT_SMPL_FILTERED, dest="smpl_filtered")
    p.add_argument("--out",            type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-files",      type=int,  default=None)
    p.add_argument("--overwrite",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.alignment.is_file():
        raise FileNotFoundError(f"alignment 不存在: {args.alignment}")
    if not (args.src_lerobot / "data").is_dir():
        raise FileNotFoundError(f"src-lerobot 不存在: {args.src_lerobot}")
    if not args.smpl_filtered.is_dir():
        raise FileNotFoundError(f"smpl-filtered 不存在: {args.smpl_filtered}")

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} 已存在，加 --overwrite 覆盖")
        import subprocess
        print("[build] 清理旧数据集...", flush=True)
        try:
            shutil.rmtree(args.out)
        except (OSError, PermissionError):
            subprocess.run(f"rm -rf {args.out}", shell=True, check=False, timeout=30)
    (args.out / "data" / "chunk-000").mkdir(parents=True)
    (args.out / "meta").mkdir(parents=True)

    # ---- 构建精确 pkl 映射 ----
    print("[build] 构建精确 pkl 映射（source_motion_file → smpl_filtered/）...", flush=True)
    exact_pkl_map = build_exact_pkl_map(args.src_lerobot, args.smpl_filtered)

    # ---- 读 alignment ----
    align_all = pd.read_parquet(args.alignment)
    usable    = align_all[align_all["alignment_status"].isin(USABLE_STATUSES)].copy()
    print(
        f"[build] usable episodes: {len(usable):,d} / {len(align_all):,d} "
        f"({100*len(usable)/max(len(align_all),1):.1f}%)",
        flush=True,
    )

    src_pqs = sorted((args.src_lerobot / "data").rglob("*.parquet"))
    if args.max_files:
        src_pqs = src_pqs[: args.max_files]
    print(f"[build] 源 parquet: {len(src_pqs)} 个", flush=True)

    ep_base = task_base = idx_base = 0
    all_tasks: list[pd.DataFrame] = []
    total_rows = 0
    t0 = time.time()

    for fi, src_pq in enumerate(src_pqs):
        rel = src_pq.relative_to(args.src_lerobot).as_posix()
        file_align = usable[usable["data_file"] == rel]

        if file_align.empty:
            print(f"  [{fi+1:02d}/{len(src_pqs)}] {src_pq.name}: 无 usable eps，跳过", flush=True)
            continue

        t1 = time.time()
        print(f"  [{fi+1:02d}/{len(src_pqs)}] {src_pq.name}: {len(file_align)} eps …", end="", flush=True)

        out_table, tasks_df, ep_base, task_base, idx_base = process_file(
            src_pq,
            file_align=file_align,
            ep_base=ep_base,
            task_base=task_base,
            idx_base=idx_base,
            exact_pkl_map=exact_pkl_map,
        )

        if out_table is None or out_table.num_rows == 0:
            print(" → 0 行，跳过", flush=True)
            continue

        out_path = args.out / "data" / "chunk-000" / f"file-{fi:03d}.parquet"
        pq.write_table(out_table, out_path, compression="zstd")
        total_rows += out_table.num_rows

        if not tasks_df.empty:
            all_tasks.append(tasks_df)

        print(
            f" → {out_table.num_rows:,d} rows | {len(tasks_df)} eps | "
            f"{time.time()-t1:.1f}s (累计 {total_rows:,d} rows, {ep_base} eps)",
            flush=True,
        )

    if all_tasks:
        write_tasks(args.out, pd.concat(all_tasks, ignore_index=True))
    write_info(args.out, total_episodes=ep_base, total_frames=total_rows)

    # ---- 汇总 exact vs fallback ----
    if all_tasks:
        tasks_all = pd.concat(all_tasks, ignore_index=True)
        src_dist = tasks_all["smpl_pkl_source"].value_counts()
        print("\n[build] smpl_pkl_source 分布:", flush=True)
        print(src_dist.to_string(), flush=True)

    print(
        f"\n[build] ✓ 完成: {total_rows:,d} rows | {ep_base:,d} episodes | "
        f"{(time.time()-t0)/60:.1f} min → {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
