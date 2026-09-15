#!/usr/bin/env python3
"""Rewrite egypt_smplsem_clip qpos from clip SMPL via GMR (drop Bones/Proto G1 CSV qpos).

Pipeline:
  meta.smpl_pkl → write SMPL into unified[0:315) (Y-up pose + transl Δ)
  same pkl → Rx(90°) Z-up → GMR → absolute q36 (memory)
  → relative root7 + dof29 into unified[360:396] (no absolute root on disk)
  [0:9] stays SMPL — never overwrite with GMR.

Do **not** apply Sonic ``remove_smpl_base_rot``. Height: ``None``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
_WORK = Path("/mnt/data2/wpy/workspace")
sys.path.insert(0, str(_WORK / "phi-0-wbc" / "src"))
sys.path.insert(0, str(_HERE))

from egypt_clip_root_layout import (  # noqa: E402
    apply_gmr_qpos_tail,
    assert_root_layout_consistent,
)
from gmr_from_smpl import (  # noqa: E402
    WAIST_YAW as _WAIST_YAW,
    align_root_yaw_to_smpl,
    gmr_qpos_from_smpl,
    postprocess_gmr_root_xy0_z_match,
    smpl_yup_to_zup,
)
from phi0.schema.unified_action_schema import (  # noqa: E402
    write_smpl_semantic_from_pose_aa,
)

DEFAULT_CLIP = _WORK / "egypt_smplsem_clip"


def _fsl(col, w: int) -> np.ndarray:
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    flat = arr.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(len(arr), w)


def _write_parquet(pq_path: Path, table, u_new: np.ndarray, dm_new: np.ndarray) -> None:
    arrays = []
    names = []
    for name in table.column_names:
        if name == "action.unified":
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(u_new.reshape(-1)), 512))
        elif name == "action.dim_mask":
            arrays.append(
                pa.FixedSizeListArray.from_arrays(pa.array(dm_new.reshape(-1).astype(bool)), 512)
            )
        elif name == "action.qpos_g1":
            arrays.append(
                pa.FixedSizeListArray.from_arrays(pa.array(u_new[:, 360:396].reshape(-1)), 36)
            )
        else:
            arrays.append(table.column(name))
        names.append(name)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), pq_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    ap.add_argument(
        "--smpl-pkl",
        type=Path,
        default=None,
        help="defaults to meta.json smpl_pkl (need transl for full GMR)",
    )
    ap.add_argument(
        "--postprocess-only",
        action="store_true",
        help="DEPRECATED: post-hoc align (twists waist); prefer --lock-root",
    )
    ap.add_argument(
        "--lock-root",
        action="store_true",
        help="pin freejoint yaw to SMPL, re-solve IK for joints only",
    )
    ap.add_argument(
        "--align-root",
        action="store_true",
        help="DEPRECATED post-hoc root:=SMPL (do not use)",
    )
    ap.add_argument("--no-hip-compensate", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta_path = args.clip / "meta.json"
    meta = json.loads(meta_path.read_text())
    pkl_path = Path(args.smpl_pkl or meta["smpl_pkl"])
    pq_path = args.clip / "data" / "chunk-000" / "file-000.parquet"

    table = pq.read_table(pq_path)
    n = table.num_rows if args.max_frames is None else min(args.max_frames, table.num_rows)
    unified = _fsl(table.column("action.unified"), 512)[:n]
    dim_mask = _fsl(table.column("action.dim_mask"), 512).astype(bool)[:n]
    pkl = joblib.load(pkl_path)
    pose_pkl = np.asarray(pkl["pose_aa"][:n], dtype=np.float32)
    if "action.smpl_pose_aa" in table.column_names:
        pose_clip = _fsl(table.column("action.smpl_pose_aa"), 72)[:n]
        if not np.allclose(pose_clip, pose_pkl, atol=1e-5):
            raise AssertionError("clip smpl_pose_aa != pkl[:n]; refuse to retarget wrong motion")
    else:
        pose_clip = pose_pkl  # no sidecar — trust meta.smpl_pkl
    bak = pq_path.with_suffix(".parquet.bak_official_qpos")
    bak_gmr = pq_path.with_suffix(".parquet.bak_gmr_raw")
    if bak.is_file() and "action.qpos_g1" in pq.read_schema(bak).names:
        old_q = _fsl(pq.read_table(bak, columns=["action.qpos_g1"]).column(0), 36)[:n]
    elif "action.qpos_g1" in table.column_names:
        old_q = _fsl(table.column("action.qpos_g1"), 36)[:n]
    else:
        old_q = unified[:, 360:396].copy()

    if args.postprocess_only or args.align_root:
        if "action.qpos_g1" in table.column_names:
            q = _fsl(table.column("action.qpos_g1"), 36)[:n].copy()
        else:
            q = unified[:, 360:396].copy()
        print(f"[post] DEPRECATED align on {n} frames", flush=True)
        q, st = align_root_yaw_to_smpl(
            q, pose_clip, compensate_hip_yaw=not args.no_hip_compensate
        )
        print(f"[align] {st}", flush=True)
    else:
        transl = np.asarray(pkl["transl"][:n], dtype=np.float32)
        pose_z, transl_z = smpl_yup_to_zup(pose_pkl, transl)
        print(
            f"[gmr] retargeting {n} frames lock_root={args.lock_root} from {pkl_path.name} ...",
            flush=True,
        )
        q_gmr = gmr_qpos_from_smpl(
            pose_z,
            transl_z,
            lock_root=bool(args.lock_root),
            progress_every=50,
        )
        if not np.isfinite(q_gmr).all():
            raise RuntimeError("GMR produced non-finite qpos")
        q = postprocess_gmr_root_xy0_z_match(q_gmr, proto_z0=float(old_q[0, 2]))
        st = {}
        if args.lock_root:
            wy = np.degrees(q[:, 7 + _WAIST_YAW])
            wr = np.degrees(q[:, 7 + _WAIST_YAW + 1])
            wp = np.degrees(q[:, 7 + _WAIST_YAW + 2])
            root_y = np.unwrap(R.from_quat(q[:, 3:7], scalar_first=True).as_euler("xyz")[:, 2])
            st = {
                "lock_target": "gmr_scaled_pelvis_quat",
                "root_yaw_deg_range": float(np.degrees(np.ptp(root_y))),
                "waist_yaw_deg_std": float(wy.std()),
                "waist_yaw_deg_range": float(wy.ptp()),
                "waist_roll_deg_range": float(wr.ptp()),
                "waist_pitch_deg_range": float(wp.ptp()),
                "waist_abs_sum_max_deg": float(
                    np.max(np.abs(np.degrees(q[:, 7 + 12 : 7 + 15])).sum(axis=1))
                ),
            }
            print(f"[lock-root] {st}", flush=True)
            if st["waist_abs_sum_max_deg"] > 60:
                print("[warn] waist still large; check lock frame", flush=True)

    # when max-frames < full clip, only rewrite prefix — refuse partial write to parquet
    if n != table.num_rows:
        print(f"[dry-partial] n={n}/{table.num_rows}; use --dry-run metrics only or full n", flush=True)
        if not args.dry_run:
            raise SystemExit("refusing partial parquet write; pass --dry-run for smoke")
        print("[dry-run] skip write")
        return

    # 1) SMPL front from pkl (Y-up) — never GMR into [0:9]
    u_new = _fsl(table.column("action.unified"), 512).copy()
    dm_new = _fsl(table.column("action.dim_mask"), 512).astype(bool).copy()
    transl_y = np.asarray(pkl["transl"][:n], dtype=np.float32)
    rtl = np.zeros_like(transl_y)
    rtl[1:] = transl_y[1:] - transl_y[:-1]
    for i in range(n):
        write_smpl_semantic_from_pose_aa(
            u_new[i], pose_pkl[i], rtl[i], dim_mask=dm_new[i]
        )

    # 2) GMR tail: relative root7 + dof29
    u_new, dm_new, _init_xyz, _init_quat = apply_gmr_qpos_tail(
        u_new, q36_abs=q, dim_mask=dm_new
    )
    assert dm_new is not None
    assert_root_layout_consistent(u_new)

    yaw0 = float(R.from_quat(q[0, 3:7], scalar_first=True).as_euler("xyz")[2])
    yaw50 = float(R.from_quat(q[min(50, n - 1), 3:7], scalar_first=True).as_euler("xyz")[2])
    print(
        f"[qpos] z=[{q[:, 2].min():.3f},{q[:, 2].max():.3f}] "
        f"yaw0={np.degrees(yaw0):.1f}deg rel@50={np.degrees(yaw50 - yaw0):+.1f}deg "
        f"smpl_front=1 gmr_rel_root=1",
        flush=True,
    )

    if args.dry_run:
        print("[dry-run] skip write")
        return

    if not bak.exists():
        bak.write_bytes(pq_path.read_bytes())
        print(f"[bak] {bak}", flush=True)
    if not bak_gmr.exists():
        bak_gmr.write_bytes(pq_path.read_bytes())
        print(f"[bak] {bak_gmr} (pre-lock snapshot if first)", flush=True)

    _write_parquet(pq_path, table, u_new, dm_new)

    if args.lock_root:
        meta["qpos_source"] = "gmr_lock_root_gmr_pelvis"
        meta["qpos_note"] = (
            "GMR lock-root; unified[0:315)=SMPL; [360:367]=GMR Δxy+abs z+abs quat; "
            "[367:396]=dof. No meta init for RSI."
        )
    else:
        meta["qpos_source"] = "gmr_from_smpl_filtered"
        meta["qpos_note"] = (
            "unified[0:315)=SMPL from smpl_pkl; [360:367]=GMR Δxy+abs z+abs quat; "
            "[367:396]=GMR dof29. No meta init for RSI."
        )
    meta["note"] = meta["qpos_note"]
    meta["qpos_lock_root"] = st
    meta.pop("root_init_xyz", None)
    meta.pop("root_init_quat_wxyz", None)
    meta.pop("root_layout", None)
    meta["dropped_action_sidecars"] = [
        "action.smpl_pose_aa",
        "action.smpl_joints",
        "action.qpos_g1",
    ]
    meta["action_schema"] = "ACTION_SCHEMA.md"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[ok] wrote {pq_path} + meta.json", flush=True)


if __name__ == "__main__":
    main()
