#!/usr/bin/env python3
"""Pack BoneSEED demo5skill → sonic_v1_1 unified with **LL-style explicit obs**.

Same structure as ``pack_demo5skill_ll_sonic_820demo.py``:
  ``_q36_cache`` → Isaac DOF + root quat → hand-built g1 encoder obs → ONNX encode.

sonic_v1_1 specifics (≠ release / ≠ ``onnx_encode`` 1762 layout):
  - obs dim **1751** (not 1762); g1 anchor at **584** (heading 10×step5)
  - ``motion_anchor_orientation_heading``: left = yaw-only(base), right = full ref
  - history step = 5 frames @ 50fps
  - one parquet per episode (skill_N layout) for EP=k GT replay
  - ``--hand-mode dex3`` zeros (match 830 mix train)

Not SMPL. Not ``encode_zstar_onnx_g1_online`` (that still builds release 1762@601).
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
from scipy.spatial.transform import Rotation as R

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tools" / "data"))

os.environ.setdefault("DEPLOY_POLICY_DIR", "sonic_v1_1")

from egypt_clip_root_layout import apply_gmr_qpos_tail  # noqa: E402
from phi0.online.student_obs import mujoco_to_isaaclab_dof  # noqa: E402
from phi0.schema.unified_action_schema import SLICES  # noqa: E402

from merge_820_release_unified import DEMO5_PROMPTS  # noqa: E402

ALLOW_DEFAULT = _ROOT / "meta" / "boneseed_allowlist_demo5skill.json"
RELROOT_DEFAULT = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
OUT_DEFAULT = Path(
    "/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_sonic_v1_1_llstyle_unified"
)
V11_ONNX_DEFAULT = (
    _ROOT / "subpackages/gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx"
)

FPS = 50.0
D = 512
TOKEN_DIM = 64
OBS_DIM = 1751  # sonic_v1_1 encoder (release is 1762)
NUM_HIST = 10
HIST_STEP = 5  # 0.1s @ 50fps
VEL_STEP = 1  # 0.02s
OFF_MODE = 0
OFF_JPOS = 4
OFF_JVEL = 294
OFF_ANCHOR_HEADING = 584  # motion_anchor_orientation_heading_10frame_step5

SONIC = SLICES["sonic_motion_token_64"]
GRAV = SLICES["projected_gravity_xyz"]
REVO2 = SLICES["revo2_hand_12"]
DEX3 = SLICES["g1_gripper_joints_14"]


def _np_fsl(mat: np.ndarray, pa_dtype: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(mat.reshape(-1), type=pa_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, mat.shape[1])


def _gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-8)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return R.from_quat(q, scalar_first=True).inv().apply(g_world).astype(np.float32)


def _heading_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion (wxyz). Matches deploy ``calc_heading_quat_d``."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-8)
    x = R.from_quat(q, scalar_first=True).as_matrix()[:, 0]
    yaw = float(np.arctan2(x[1], x[0]))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _rot6d(base_quat_wxyz: np.ndarray, ref_quat_wxyz: np.ndarray) -> np.ndarray:
    base = R.from_quat(np.asarray(base_quat_wxyz, dtype=np.float64), scalar_first=True)
    ref = R.from_quat(np.asarray(ref_quat_wxyz, dtype=np.float64), scalar_first=True)
    mat = (base.inv() * ref).as_matrix()
    return mat[:, :2].reshape(6).astype(np.float32)


def _clamp_idx(i: int, n: int) -> int:
    return int(min(max(i, 0), n - 1))


def build_v11_obs_g1(
    *, dof_isaac: np.ndarray, root_quat: np.ndarray, t: int
) -> np.ndarray:
    """sonic_v1_1 g1 obs (1751): jpos/jvel step5 + heading ori @584."""
    n = int(dof_isaac.shape[0])
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[OFF_MODE] = 0.0  # g1
    dt = float(VEL_STEP) / FPS
    head_base = _heading_quat_wxyz(root_quat[_clamp_idx(t, n)])
    for k in range(NUM_HIST):
        i0 = _clamp_idx(t + k * HIST_STEP, n)
        i1 = _clamp_idx(t + k * HIST_STEP + VEL_STEP, n)
        obs[OFF_JPOS + k * 29 : OFF_JPOS + (k + 1) * 29] = dof_isaac[i0]
        obs[OFF_JVEL + k * 29 : OFF_JVEL + (k + 1) * 29] = (
            dof_isaac[i1] - dof_isaac[i0]
        ) / dt
        obs[
            OFF_ANCHOR_HEADING + k * 6 : OFF_ANCHOR_HEADING + (k + 1) * 6
        ] = _rot6d(head_base, root_quat[i0])
    return obs


class V11Encoder:
    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        self.path = onnx_path.resolve()
        self.sess = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inp = self.sess.get_inputs()[0]
        dim = int(inp.shape[-1]) if inp.shape[-1] is not None else -1
        if dim != OBS_DIM:
            raise ValueError(f"expected obs dim {OBS_DIM}, got {inp.shape}")
        self._in = inp.name
        self._out = self.sess.get_outputs()[0].name

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM)
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


def pack(
    *,
    allow: list[int],
    prompts: dict[str, str],
    relroot: Path,
    encoder: V11Encoder,
    out: Path,
    overwrite: bool,
    hand_mode: str = "dex3",
) -> None:
    hand_mode = str(hand_mode).strip().lower()
    if hand_mode not in ("revo2", "dex3"):
        raise SystemExit(f"hand_mode must be revo2|dex3, got {hand_mode!r}")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists (pass --overwrite)")
    if out.exists() and overwrite:
        import shutil

        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    hs, he = (DEX3 if hand_mode == "dex3" else REVO2)
    sources = []
    ep_lengths: list[int] = []
    all_u: list[np.ndarray] = []

    for local_i, src_ep in enumerate(allow):
        cache = relroot / "_q36_cache" / f"ep{src_ep}.npz"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        q36 = np.load(cache)["q36"].astype(np.float32)
        n = int(q36.shape[0])
        dof_isaac = mujoco_to_isaaclab_dof(q36[:, 7:36])
        if hasattr(dof_isaac, "numpy"):
            dof_isaac = dof_isaac.numpy()
        dof_isaac = np.asarray(dof_isaac, dtype=np.float32)

        z = np.zeros((n, TOKEN_DIM), dtype=np.float32)
        for t in range(n):
            obs = build_v11_obs_g1(
                dof_isaac=dof_isaac, root_quat=q36[:, 3:7], t=t
            )
            z[t] = encoder.encode(obs)
            if t % 100 == 0:
                print(
                    f"[v11-llstyle] ep{src_ep}→{local_i} frame {t}/{n}",
                    flush=True,
                )

        u = np.zeros((n, D), dtype=np.float32)
        dm = np.zeros((n, D), dtype=np.bool_)
        u2, dm2, _, _ = apply_gmr_qpos_tail(
            u, q36_abs=q36, dim_mask=dm, check_roundtrip=True
        )
        u[:] = u2
        dm[:] = dm2
        s0, s1 = SONIC
        u[:, s0:s1] = z
        dm[:, s0:s1] = True
        gs, ge = GRAV
        for i in range(n):
            u[i, gs:ge] = _gravity_from_quat(q36[i, 3:7])
        dm[:, gs:ge] = False
        u[:, 0:346] = 0.0
        dm[:, 0:346] = False
        u[:, hs:he] = 0.0
        dm[:, hs:he] = hand_mode == "dex3"
        if hand_mode == "dex3":
            rs, re = REVO2
            u[:, rs:re] = 0.0
            dm[:, rs:re] = False

        ts = np.arange(n, dtype=np.float32) / FPS
        fi = np.arange(n, dtype=np.int64)
        ep = np.full(n, local_i, dtype=np.int64)
        task = np.full(n, local_i, dtype=np.int64)
        done = np.zeros(n, dtype=np.bool_)
        done[-1] = True
        idx = np.arange(n, dtype=np.int64)
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
        out_pq = out / "data" / "chunk-000" / f"file-{local_i:03d}.parquet"
        pq.write_table(table, out_pq)
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
            f"[v11-llstyle] packed local={local_i} src={src_ep} T={n} "
            f"sonic_absmean={float(np.abs(z).mean()):.4f} -> {out_pq.name}",
            flush=True,
        )

    U = np.concatenate(all_u, axis=0)
    task_rows = [
        {
            "task_index": local_i,
            "task": prompts.get(str(src_ep), f"demo5skill episode {src_ep}"),
        }
        for local_i, src_ep in enumerate(allow)
    ]
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
        "layout": "phi0_unified_demo5skill_sonic_v1_1_llstyle",
        "sonic_encoder": {
            "policy": "sonic_v1_1",
            "onnx": str(encoder.path),
            "obs_dim": OBS_DIM,
            "mode": "g1",
            "mode_id": 0,
            "anchor_offset": OFF_ANCHOR_HEADING,
            "anchor": "motion_anchor_orientation_heading_10frame_step5",
            "history_step_frames": HIST_STEP,
            "source": "q36_cache_qpos_llstyle_explicit_obs",
        },
        "hand_mode": hand_mode,
        "episode_lengths": ep_lengths,
    }
    (out / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "meta" / "modality.json").write_text(
        json.dumps({"note": "no video modalities (BoneSEED demo5skill)"}, indent=2)
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
        "layout": info["layout"],
        "layout_note": (
            "sonic_v1_1 from q36 via LL-style explicit g1 obs "
            f"(dim={OBS_DIM}, heading@ {OFF_ANCHOR_HEADING}, step{HIST_STEP}); "
            "not encode_zstar release-1762 path"
        ),
        "hand_mode": hand_mode,
        "qpos_source": "boneseed_q36_cache",
        "qpos_note": (
            f"q36 from {relroot}/_q36_cache; apply_gmr_qpos_tail; "
            f"sonic[396:460)=sonic_v1_1 llstyle explicit; {hand_mode} zeros"
        ),
        "sonic_motion_token": {
            "slice": list(SONIC),
            "source": "qpos_sonic_v1_1_llstyle_explicit",
            "policy": "sonic_v1_1",
            "onnx": str(encoder.path),
            "obs_dim": OBS_DIM,
            "mode_id": 0,
            "anchor_offset": OFF_ANCHOR_HEADING,
            "history_step_frames": HIST_STEP,
            "note": "LL-script structure; true sonic_v1_1 1751 heading obs.",
            "abs_mean": float(np.abs(U[:, SONIC[0] : SONIC[1]]).mean()),
        },
        "revo2_hand": {
            "slice": list(REVO2),
            "layout": "unused (hand_mode=dex3)" if hand_mode == "dex3" else "zeros",
            "note": "zeros; dim_mask false",
        },
        "dex3_hand": {
            "slice": list(DEX3),
            "layout": "zeros (no hand in BoneSEED; hand_mode=dex3)",
            "note": "forced 0; dim_mask True (W_HAND GT)"
            if hand_mode == "dex3"
            else "unused",
        },
        "videos": None,
        "episodes": len(allow),
        "frames": int(U.shape[0]),
        "sources": sources,
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[v11-llstyle] wrote {out} policy=sonic_v1_1 hand={hand_mode} "
        f"eps={len(allow)} frames={U.shape[0]} obs_dim={OBS_DIM} "
        f"hist_step={HIST_STEP}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allowlist", type=Path, default=ALLOW_DEFAULT)
    p.add_argument("--prompts", type=Path, default=None)
    p.add_argument("--relroot", type=Path, default=RELROOT_DEFAULT)
    p.add_argument("--onnx", type=Path, default=V11_ONNX_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--hand-mode", choices=("revo2", "dex3"), default="dex3")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    allow = _load_allowlist(args.allowlist)
    prompts = _load_prompts(allow, args.prompts)
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    enc = V11Encoder(onnx_path)
    print(
        f"[v11-llstyle] encoder={enc.path} obs_dim={OBS_DIM} "
        f"anchor@{OFF_ANCHOR_HEADING} hist_step={HIST_STEP}",
        flush=True,
    )
    pack(
        allow=allow,
        prompts=prompts,
        relroot=args.relroot,
        encoder=enc,
        out=args.out,
        overwrite=bool(args.overwrite),
        hand_mode=str(args.hand_mode),
    )


if __name__ == "__main__":
    main()
