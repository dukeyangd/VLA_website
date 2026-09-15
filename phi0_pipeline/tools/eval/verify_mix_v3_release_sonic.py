#!/usr/bin/env python3
"""Proof: mix_v3 release sonic[396:460) != raw LL unified; matches release re-encode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "data"))

from reencode_unified_release_sonic import _encode_episode_q36_gpu  # noqa: E402

SLICE = slice(396, 460)


def _parquet_for_ep(root: Path, ep: int) -> Path:
    p = root / "data" / "chunk-000" / f"file-{ep:03d}.parquet"
    return p if p.exists() else root / "data" / "chunk-000" / "file-000.parquet"


def load_slice(parquet: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    from phi0.schema.unified_action_schema import D_UNIFIED, unpack_g1_body_qpos_36

    t = pq.read_table(parquet)
    if "episode_index" in t.column_names:
        e = np.asarray(t.column("episode_index").to_numpy()).reshape(-1)
        idx = np.where(e == ep)[0]
        if idx.size:
            t = t.take(idx.tolist())
    col = t.column("action.unified")
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    flat = arr.values.to_numpy(zero_copy_only=False)
    act = np.asarray(flat, dtype=np.float32).reshape(len(arr), D_UNIFIED)
    q36 = np.stack([unpack_g1_body_qpos_36(act[i]) for i in range(act.shape[0])])
    return q36, act[:, SLICE]


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / (na * nb + 1e-8)
    return {
        "l2_mean": float(np.linalg.norm(d, axis=-1).mean()),
        "l2_max": float(np.linalg.norm(d, axis=-1).max()),
        "cos_mean": float(cos.mean()),
    }


def ll_root_from_release(origin_root: str) -> Path | None:
    p = Path(origin_root.replace("_release_unified", "_unified"))
    return p if (p / "data").is_dir() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=64)
    args = ap.parse_args()
    ref = args.ref_root.resolve()
    meta = json.loads((args.meta or ref / "meta.json").read_text())

    # one ep per origin_tag + demo5 bow
    picks: list[dict] = []
    seen: set[str] = set()
    for row in meta["sources"]:
        tag = row["origin_tag"]
        key = tag if row.get("task_index") != 1 else "demo5skill_bow"
        if key in seen:
            continue
        seen.add(key)
        picks.append(
            {
                "label": key,
                "ep": int(row["out_episode_index"]),
                "origin_root": row["origin_root"],
                "origin_ep": int(row["origin_episode_index"]),
                "task_index": row.get("task_index"),
            }
        )

    rows = []
    for pick in sorted(picks, key=lambda r: r["ep"]):
        ep = pick["ep"]
        pq = _parquet_for_ep(ref, ep)
        q36, z_release = load_slice(pq, ep)
        z_re_full = _encode_episode_q36_gpu(q36, fps=50.0, batch=64)
        n = min(args.max_frames, z_release.shape[0])
        idx = np.linspace(0, z_release.shape[0] - 1, n, dtype=int)
        q36 = q36[idx]
        z_release = z_release[idx]
        z_re = z_re_full[idx]
        rel = stats(z_release, z_re)
        row = {
            "label": pick["label"],
            "ep": ep,
            "task_index": pick["task_index"],
            "frames": int(n),
            "release_disk_vs_reencode": rel,
        }
        ll_root = ll_root_from_release(pick["origin_root"])
        if ll_root is not None:
            ll_pq = _parquet_for_ep(ll_root, pick["origin_ep"])
            if ll_pq.exists():
                _, z_ll_full = load_slice(ll_pq, pick["origin_ep"])
                z_ll = z_ll_full[idx]
                row["release_vs_ll_intermediate"] = stats(z_release, z_ll)
        rows.append(row)
        msg = (
            f"[{pick['label']}] ep={ep} disk↔reencode l2={rel['l2_mean']:.4f} cos={rel['cos_mean']:.4f}"
        )
        if "release_vs_ll_intermediate" in row:
            r2 = row["release_vs_ll_intermediate"]
            msg += f" | vs_LL l2={r2['l2_mean']:.4f} cos={r2['cos_mean']:.4f}"
        print(msg)

    out = ref / "meta" / "release_sonic_verify_mix_v3.json"
    out.write_text(
        json.dumps({"ref_root": str(ref), "policy": "release", "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
