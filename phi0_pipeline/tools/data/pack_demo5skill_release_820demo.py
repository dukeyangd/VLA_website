#!/usr/bin/env python3
"""Pack BoneSEED demo5skill → 820demo ``*_release_unified`` (skill_N layout).

Same on-disk contract as ``820demo_skill_3_release_unified``:
  data/chunk-000/file-{ep:03d}.parquet  (one episode per file)
  meta/{info,modality,stats,tasks}.json|parquet
  meta.json
  videos/ — none (BoneSEED has no ego/chest); modality notes that.

Sonic ``[396:460)`` from q36 via release GPU encoder (default path).
Revo2 ``[463:475)`` = 0. No SMPL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation as R

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tools" / "data"))

os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")

from egypt_clip_root_layout import apply_gmr_qpos_tail  # noqa: E402
from phi0.online.onnx_encode import (  # noqa: E402
    _ENCODER_OBS_DIM,
    _MOTION_TOKEN_DIM,
    encode_zstar_onnx_g1_online,
)
from phi0.schema.unified_action_schema import SLICES  # noqa: E402

from merge_820_release_unified import DEMO5_PROMPTS  # noqa: E402

ALLOW_DEFAULT = _ROOT / "meta" / "boneseed_allowlist_demo5skill.json"
RELROOT_DEFAULT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
OUT_DEFAULT = Path(
    "/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_release_unified"
)
RELEASE_ONNX = (
    _ROOT / "subpackages/gear_sonic_deploy/policy/release/model_encoder.onnx"
)

FPS = 50.0
D = 512
SONIC = SLICES["sonic_motion_token_64"]
GRAV = SLICES["projected_gravity_xyz"]
REVO2 = SLICES["revo2_hand_12"]
STAND_Z = 0.785


def _np_fsl(mat: np.ndarray, pa_dtype: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(mat.reshape(-1), type=pa_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, mat.shape[1])


def _gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-8)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return R.from_quat(q, scalar_first=True).inv().apply(g_world).astype(np.float32)


def _encode_z(q36: np.ndarray, *, batch: int = 128) -> np.ndarray:
    t_len = int(q36.shape[0])
    dof = np.asarray(q36[:, 7:36], dtype=np.float32)
    root_q = np.asarray(q36[:, 3:7], dtype=np.float32)
    z = np.zeros((t_len, _MOTION_TOKEN_DIM), dtype=np.float32)
    tape: dict = {}
    for s in range(0, t_len, batch):
        e = min(s + batch, t_len)
        times = torch.arange(s, e, dtype=torch.long)
        tok = encode_zstar_onnx_g1_online(
            fk_dof29_mj=dof,
            fk_root_quat_wxyz=root_q,
            robot_anchor_quat_wxyz=root_q[s:e],
            times=times,
            horizon=1,
            fps=FPS,
            tape_cache=tape,
            episode_end_frame=float(t_len - 1),
            use_gpu=True,
            device="cuda",
        )
        z[s:e] = tok.reshape(-1, _MOTION_TOKEN_DIM).detach().float().cpu().numpy()
    return z


def pack(*, allow: list[int], prompts: dict[str, str], relroot: Path, out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists (pass --overwrite)")
    if out.exists() and overwrite:
        import shutil

        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    sources = []
    ep_lengths = []
    all_u = []
    global_index = 0
    s0, s1 = SONIC
    gs, ge = GRAV
    rs, re = REVO2

    for local_i, src_ep in enumerate(allow):
        cache = relroot / "_q36_cache" / f"ep{src_ep}.npz"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        q36 = np.load(cache)["q36"].astype(np.float32)
        # match teleop: root xyz often stand_z; boneseed q36 already has xyz — keep as-is
        n = int(q36.shape[0])
        z = _encode_z(q36)
        u = np.zeros((n, D), dtype=np.float32)
        dm = np.zeros((n, D), dtype=np.bool_)
        u2, dm2, _, _ = apply_gmr_qpos_tail(u, q36_abs=q36, dim_mask=dm, check_roundtrip=True)
        u[:] = u2
        dm[:] = dm2
        u[:, s0:s1] = z
        dm[:, s0:s1] = True
        for i in range(n):
            u[i, gs:ge] = _gravity_from_quat(q36[i, 3:7])
        dm[:, gs:ge] = False
        u[:, rs:re] = 0.0
        dm[:, rs:re] = False
        u[:, 0:360] = 0.0
        dm[:, 0:360] = False

        ts = np.arange(n, dtype=np.float32) / FPS
        fi = np.arange(n, dtype=np.int64)
        ep = np.full(n, local_i, dtype=np.int64)
        task = np.full(n, local_i, dtype=np.int64)
        done = np.zeros(n, dtype=np.bool_)
        done[-1] = True
        idx = np.arange(global_index, global_index + n, dtype=np.int64)
        global_index += n

        table = pa.table(
            {
                "timestamp": pa.array(ts, type=pa.float32()),
                "frame_index": pa.array(fi, type=pa.int64()),
                "next.done": pa.array(done, type=pa.bool_()),
                "episode_index": pa.array(ep, type=pa.int64()),
                "task_index": pa.array(task, type=pa.int64()),
                "index": pa.array(idx, type=pa.int64()),
                "action.unified": _np_fsl(u, pa.float32()),
                "action.dim_mask": _np_fsl(dm, pa.bool_()),
            }
        )
        pq.write_table(
            table,
            out / "data" / "chunk-000" / f"file-{local_i:03d}.parquet",
            compression="zstd",
        )
        all_u.append(u)
        ep_lengths.append(n)
        sources.append(
            {
                "out_episode_index": local_i,
                "session": "boneseed_demo5skill",
                "source_episode_index": int(src_ep),
            }
        )
        print(
            f"[demo5skill-release] ep{src_ep}→{local_i} T={n} "
            f"sonic_absmean={float(np.abs(z).mean()):.4f}",
            flush=True,
        )

    U = np.concatenate(all_u, axis=0)
    task_rows = []
    for local_i, src_ep in enumerate(allow):
        task_rows.append(
            {
                "task_index": local_i,
                "task": prompts.get(str(src_ep), f"demo5skill episode {src_ep}"),
            }
        )
    pq.write_table(pa.Table.from_pylist(task_rows), out / "meta" / "tasks.parquet")

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
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": FPS,
        "total_episodes": len(allow),
        "total_frames": int(U.shape[0]),
        "total_tasks": len(allow),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(allow)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": features,
        "layout": "phi0_unified_teleop_qpos",
        "sonic_encoder": {
            "policy": "release",
            "onnx": str(RELEASE_ONNX),
            "obs_dim": _ENCODER_OBS_DIM,
            "mode": "g1",
            "mode_id": 0,
            "source": "qpos_release_reencode",
        },
    }
    (out / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "meta" / "modality.json").write_text(
        json.dumps(
            {
                "note": "no video modalities (BoneSEED demo5skill); layout matches skill_N_release_unified action schema"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stats = {
        "action.unified": {
            "mean": U.mean(axis=0).tolist(),
            "std": U.std(axis=0).tolist(),
            "min": U.min(axis=0).tolist(),
            "max": U.max(axis=0).tolist(),
        }
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats) + "\n", encoding="utf-8")

    meta = {
        "layout": "phi0_unified_teleop_qpos",
        "layout_note": "release_sonic_reencode_from_qpos (BoneSEED demo5skill; no teleop video)",
        "qpos_source": "boneseed_q36_cache",
        "qpos_note": (
            f"q36 from {relroot}/_q36_cache; disk root via apply_gmr_qpos_tail; "
            "sonic[396:460)=release reencode; revo2 zeros; no video"
        ),
        "sonic_motion_token": {
            "slice": [s0, s1],
            "source": "qpos_release_reencode",
            "policy": "release",
            "onnx": str(RELEASE_ONNX.resolve()),
            "obs_dim": _ENCODER_OBS_DIM,
            "mode_id": 0,
            "note": "Same default as skill_N_release_unified: qpos → release encoder.",
            "abs_mean": float(np.abs(U[:, s0:s1]).mean()),
        },
        "revo2_hand": {
            "slice": list(REVO2),
            "layout": "zeros (no hand in BoneSEED demo5skill)",
            "note": "forced 0; dim_mask false",
        },
        "videos": {
            "keys": [],
            "path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "counts": {},
            "note": "no ego/chest video for BoneSEED demo5skill; GT uses ROBOT_ONLY=1",
        },
        "task_prompt": None,
        "stand_z": STAND_Z,
        "manifest": str(ALLOW_DEFAULT),
        "raw_root": str(relroot),
        "episodes": len(allow),
        "frames": int(U.shape[0]),
        "sources": sources,
        "train": {"episode_lengths": ep_lengths},
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[demo5skill-release] wrote {out} eps={len(allow)} frames={U.shape[0]}", flush=True)


def _prompts_for_allow(allow: list[int], prompts_path: Path | None) -> dict[str, str]:
    if prompts_path is not None:
        return {
            str(k): str(v)
            for k, v in json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"].items()
        }
    if len(allow) != len(DEMO5_PROMPTS):
        raise SystemExit(
            f"demo5 allow n={len(allow)} != DEMO5_PROMPTS n={len(DEMO5_PROMPTS)}"
        )
    return {str(ep): DEMO5_PROMPTS[i] for i, ep in enumerate(allow)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allowlist", type=Path, default=ALLOW_DEFAULT)
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="optional {prompts: {ep: text}} json; default DEMO5_PROMPTS short Chinese",
    )
    p.add_argument("--relroot", type=Path, default=RELROOT_DEFAULT)
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    allow = [int(x) for x in json.loads(args.allowlist.read_text())["episode_index"]]
    prompts = _prompts_for_allow(allow, args.prompts)
    pack(
        allow=allow,
        prompts=prompts,
        relroot=args.relroot,
        out=args.out_root,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
