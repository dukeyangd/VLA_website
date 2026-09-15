#!/usr/bin/env python3
"""Pack BoneSEED demo5skill → 820demo unified (LL sonic from **qpos only**).

Canonical path: ``_q36_cache`` → Isaac DOF + root quat → LL ``encoder_mode=0`` (g1)
→ sonic 64. SMPL / mode=2 packing is abandoned (unstable on hard skills).

Output matches ``820demo_skill_1_unified`` **action** schema (no video features).
Revo2 ``[463:475)`` = 0 with dim_mask=False.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tools" / "data"))
sys.path.insert(0, str(_ROOT / "subpackages"))

from egypt_clip_root_layout import apply_gmr_qpos_tail  # noqa: E402
from phi0.online.student_obs import mujoco_to_isaaclab_dof  # noqa: E402
from phi0.schema.unified_action_schema import SLICES  # noqa: E402

from merge_820_release_unified import DEMO5_PROMPTS  # noqa: E402

ALLOW_DEFAULT = _ROOT / "meta" / "boneseed_allowlist_demo5skill.json"
RELROOT_DEFAULT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
LL_ONNX_DEFAULT = (
    _ROOT / "subpackages/gear_sonic_deploy/policy/low_latency/model_encoder.onnx"
)
OUT_ROOT_DEFAULT = Path("/mnt/data2/wpy/workspace/820demo")
OUT_NAME = "820demo_demo5skill_ll_g1_unified"

FPS = 50.0
LL_OBS_DIM = 1247
TOKEN_DIM = 64

# LL observation_config.yaml encoder_observations (g1 slots only filled).
OFF_MODE = 0  # 4
OFF_JPOS = 4  # 290
OFF_JVEL = 294  # 290
OFF_ANCHOR_MF = 584  # 60

SONIC = SLICES["sonic_motion_token_64"]
GRAV = SLICES["projected_gravity_xyz"]
REVO2 = SLICES["revo2_hand_12"]


def _np_fsl(mat: np.ndarray, pa_dtype: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(mat.reshape(-1), type=pa_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, mat.shape[1])


def _rot6d(base_quat_wxyz: np.ndarray, ref_quat_wxyz: np.ndarray) -> np.ndarray:
    base = R.from_quat(np.asarray(base_quat_wxyz, dtype=np.float64), scalar_first=True)
    ref = R.from_quat(np.asarray(ref_quat_wxyz, dtype=np.float64), scalar_first=True)
    mat = (base.inv() * ref).as_matrix()
    return mat[:, :2].reshape(6).astype(np.float32)


def _gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """World gravity [0,0,-1] in pelvis frame."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-8)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return R.from_quat(q, scalar_first=True).inv().apply(g_world).astype(np.float32)


def _clamp_idx(i: int, n: int) -> int:
    return int(min(max(i, 0), n - 1))


def build_ll_obs_g1(*, dof_isaac: np.ndarray, root_quat: np.ndarray, t: int) -> np.ndarray:
    """LL 1247-d obs for encoder_mode=g1 (0) from qpos/DOF."""
    n = int(dof_isaac.shape[0])
    obs = np.zeros(LL_OBS_DIM, dtype=np.float32)
    obs[OFF_MODE] = 0.0  # g1
    dt = 1.0 / FPS
    base_q = root_quat[_clamp_idx(t, n)]
    for k in range(10):
        i0 = _clamp_idx(t + k, n)
        i1 = _clamp_idx(t + k + 1, n)
        obs[OFF_JPOS + k * 29 : OFF_JPOS + (k + 1) * 29] = dof_isaac[i0]
        obs[OFF_JVEL + k * 29 : OFF_JVEL + (k + 1) * 29] = (dof_isaac[i1] - dof_isaac[i0]) / dt
        obs[OFF_ANCHOR_MF + k * 6 : OFF_ANCHOR_MF + (k + 1) * 6] = _rot6d(
            base_q, root_quat[i0]
        )
    return obs


class LlEncoder:
    def __init__(self, onnx_path: Path) -> None:
        self.path = onnx_path.resolve()
        self.sess = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inp = self.sess.get_inputs()[0]
        if inp.shape[-1] != LL_OBS_DIM:
            raise ValueError(f"expected obs dim {LL_OBS_DIM}, got {inp.shape}")
        self._in = inp.name
        self._out = self.sess.get_outputs()[0].name

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, LL_OBS_DIM)
        y = self.sess.run([self._out], {self._in: x})[0]
        return np.asarray(y, dtype=np.float32).reshape(TOKEN_DIM)


def _load_allowlist(path: Path) -> list[int]:
    d = json.loads(path.read_text())
    return [int(x) for x in d["episode_index"]]


def _load_prompts(allow: list[int], path: Path | None) -> dict[str, str]:
    if path is not None:
        d = json.loads(path.read_text())
        return {str(k): str(v) for k, v in d["prompts"].items()}
    if len(allow) != len(DEMO5_PROMPTS):
        raise SystemExit(
            f"demo5 allow n={len(allow)} != DEMO5_PROMPTS n={len(DEMO5_PROMPTS)}"
        )
    return {str(ep): DEMO5_PROMPTS[i] for i, ep in enumerate(allow)}


def _collect_episode_tables(
    root: Path, eps: list[int], columns: list[str]
) -> dict[int, pa.Table]:
    want = set(eps)
    ep_to_file: dict[int, Path] = {}
    for pqf in sorted((root / "data").rglob("*.parquet")):
        if ".bak" in str(pqf):
            continue
        ep = np.asarray(
            pq.read_table(pqf, columns=["episode_index"]).column(0).to_numpy()
        ).reshape(-1)
        for e in list(want - ep_to_file.keys()):
            if np.any(ep == e):
                ep_to_file[e] = pqf
        if len(ep_to_file) == len(want):
            break
    missing = want - set(ep_to_file)
    if missing:
        raise KeyError(f"episodes {sorted(missing)} not found under {root}")

    out: dict[int, pa.Table] = {}
    by_file: dict[Path, list[int]] = {}
    for e, f in ep_to_file.items():
        by_file.setdefault(f, []).append(e)
    for pqf, elist in by_file.items():
        t = pq.read_table(pqf, columns=columns)
        ep = np.asarray(t.column("episode_index").to_numpy()).reshape(-1)
        for e in elist:
            idx = np.where(ep == e)[0]
            tab = t.take(pa.array(idx.tolist()))
            fi = np.asarray(tab.column("frame_index").to_numpy()).reshape(-1)
            order = np.argsort(fi, kind="mergesort")
            out[e] = tab.take(pa.array(order.tolist()))
    return out


def _blank_unified(n: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros((n, 512), dtype=np.float32)
    m = np.zeros((n, 512), dtype=np.bool_)
    return u, m


def _write_episode_common(
    *,
    u: np.ndarray,
    dm: np.ndarray,
    q36: np.ndarray,
    z: np.ndarray,
) -> None:
    u2, dm2, _, _ = apply_gmr_qpos_tail(u, q36_abs=q36, dim_mask=dm, check_roundtrip=True)
    u[:] = u2
    dm[:] = dm2
    s, e = SONIC
    u[:, s:e] = z
    dm[:, s:e] = True
    gs, ge = GRAV
    for i in range(u.shape[0]):
        g = _gravity_from_quat(q36[i, 3:7])
        u[i, gs:ge] = g
    dm[:, gs:ge] = False
    rs, re = REVO2
    u[:, rs:re] = 0.0
    dm[:, rs:re] = False
    # clear SMPL semantic front (match teleop unified: zeros / unmasked)
    u[:, 0:360] = 0.0
    dm[:, 0:360] = False


def pack_qpos_ll(
    *,
    allow: list[int],
    prompts: dict[str, str],
    relroot: Path,
    encoder: LlEncoder,
    out_dir: Path,
    overwrite: bool,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} exists (pass --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    rel_cols = [
        "episode_index",
        "frame_index",
        "timestamp",
        "action.unified",
        "action.dim_mask",
        "observation.projected_gravity",
    ]
    rel_tables = _collect_episode_tables(relroot, allow, rel_cols)

    all_u: list[np.ndarray] = []
    all_m: list[np.ndarray] = []
    all_ts: list[np.ndarray] = []
    all_fi: list[np.ndarray] = []
    all_ep: list[np.ndarray] = []
    all_task: list[np.ndarray] = []
    all_done: list[np.ndarray] = []
    source_map: dict[str, int] = {}
    ep_lengths: list[int] = []

    for local_i, src_ep in enumerate(allow):
        source_map[str(local_i)] = int(src_ep)
        tab = rel_tables[src_ep]
        n = tab.num_rows
        cache = relroot / "_q36_cache" / f"ep{src_ep}.npz"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        q36 = np.load(cache)["q36"].astype(np.float32)
        if q36.shape[0] != n:
            raise ValueError(
                f"ep{src_ep}: q36 len {q36.shape[0]} != parquet rows {n}"
            )

        dof_isaac = mujoco_to_isaaclab_dof(q36[:, 7:36]).numpy()
        z = np.zeros((n, TOKEN_DIM), dtype=np.float32)
        for t in range(n):
            obs = build_ll_obs_g1(dof_isaac=dof_isaac, root_quat=q36[:, 3:7], t=t)
            z[t] = encoder.encode(obs)
            if t % 100 == 0:
                print(f"[qpos-ll] ep{src_ep}→{local_i} frame {t}/{n}", flush=True)

        u, dm = _blank_unified(n)
        _write_episode_common(u=u, dm=dm, q36=q36, z=z)

        ts = np.arange(n, dtype=np.float32) / FPS
        fi = np.arange(n, dtype=np.int64)
        ep = np.full(n, local_i, dtype=np.int64)
        task = np.full(n, local_i, dtype=np.int64)
        done = np.zeros(n, dtype=np.bool_)
        done[-1] = True

        all_u.append(u)
        all_m.append(dm)
        all_ts.append(ts)
        all_fi.append(fi)
        all_ep.append(ep)
        all_task.append(task)
        all_done.append(done)
        ep_lengths.append(n)
        print(
            f"[qpos-ll] packed local_ep={local_i} src={src_ep} T={n} "
            f"sonic_absmean={float(np.abs(z).mean()):.4f}",
            flush=True,
        )

    U = np.concatenate(all_u, axis=0)
    M = np.concatenate(all_m, axis=0)
    TS = np.concatenate(all_ts)
    FI = np.concatenate(all_fi)
    EP = np.concatenate(all_ep)
    TASK = np.concatenate(all_task)
    DONE = np.concatenate(all_done)
    INDEX = np.arange(len(U), dtype=np.int64)

    table = pa.table(
        {
            "timestamp": pa.array(TS, type=pa.float32()),
            "frame_index": pa.array(FI, type=pa.int64()),
            "next.done": pa.array(DONE, type=pa.bool_()),
            "episode_index": pa.array(EP, type=pa.int64()),
            "task_index": pa.array(TASK, type=pa.int64()),
            "index": pa.array(INDEX, type=pa.int64()),
            "action.unified": _np_fsl(U, pa.float32()),
            "action.dim_mask": _np_fsl(M, pa.bool_()),
        }
    )
    out_pq = out_dir / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(table, out_pq)

    task_rows = []
    for local_i, src_ep in enumerate(allow):
        prompt = prompts.get(str(src_ep), f"demo5skill episode {src_ep}")
        task_rows.append({"task_index": local_i, "task": prompt})
    pq.write_table(pa.Table.from_pylist(task_rows), out_dir / "meta" / "tasks.parquet")

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
        "total_frames": int(len(U)),
        "total_tasks": len(allow),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(allow)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": features,
        "layout": "phi0_unified_demo5skill_ll",
        "sonic_encoder": {
            "policy": "low_latency",
            "onnx": str(encoder.path),
            "obs_dim": LL_OBS_DIM,
            "mode": "g1",
            "mode_id": 0,
            "source": "q36_cache_qpos",
        },
        "revo2_hand": {
            "slice": list(REVO2),
            "values": "zeros",
            "dim_mask": False,
            "note": "no hand supervision; values forced to 0",
        },
        "source": {
            "allowlist": str(ALLOW_DEFAULT),
            "relroot": str(relroot),
            "q36_cache": str(relroot / "_q36_cache"),
            "episode_map_local_to_boneseed": source_map,
            "episode_lengths": ep_lengths,
        },
        "note": (
            "No camera videos; VLM train text-only via tasks/prompts. "
            "LL sonic from qpos only (SMPL mode=2 abandoned)."
        ),
    }
    (out_dir / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "meta" / "source_episode_map.json").write_text(
        json.dumps(source_map, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "meta" / "modality.json").write_text(
        json.dumps({"note": "no video modalities"}, indent=2) + "\n", encoding="utf-8"
    )

    stats = {
        "action.unified": {
            "mean": U.mean(axis=0).tolist(),
            "std": U.std(axis=0).tolist(),
            "min": U.min(axis=0).tolist(),
            "max": U.max(axis=0).tolist(),
        }
    }
    (out_dir / "meta" / "stats.json").write_text(
        json.dumps(stats) + "\n", encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "layout": info["layout"],
                "sonic_encoder": info["sonic_encoder"],
                "revo2_hand": info["revo2_hand"],
                "source": info["source"],
                "note": info["note"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[qpos-ll] wrote {out_pq} frames={len(U)} eps={len(allow)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pack demo5skill LL sonic from q36 qpos only (no SMPL)."
    )
    p.add_argument("--allowlist", type=Path, default=ALLOW_DEFAULT)
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="optional {prompts: {ep: text}} json; default DEMO5_PROMPTS short Chinese",
    )
    p.add_argument("--relroot", type=Path, default=RELROOT_DEFAULT)
    p.add_argument("--ll-onnx", type=Path, default=LL_ONNX_DEFAULT)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    # ponytail: old CLI; refuse so callers notice SMPL path is gone
    p.add_argument(
        "--version",
        choices=["g1"],
        default="g1",
        help="Only g1/qpos LL is supported (smpl abandoned).",
    )
    args = p.parse_args()

    allow = _load_allowlist(args.allowlist)
    prompts = _load_prompts(allow, args.prompts)
    enc = LlEncoder(args.ll_onnx)
    print(f"LL encoder {enc.path} (qpos mode=0 only)", flush=True)

    out = args.out_root / OUT_NAME
    pack_qpos_ll(
        allow=allow,
        prompts=prompts,
        relroot=args.relroot,
        encoder=enc,
        out_dir=out,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
