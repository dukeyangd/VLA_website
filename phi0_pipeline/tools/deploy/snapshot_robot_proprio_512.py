#!/usr/bin/env python3
"""Capture one deploy g1_debug -> unified 512-d proprio; save zero/non-zero breakdown."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_GR00T = Path(os.environ.get("GR00T_ROOT", str(ROOT.parent / "GR00T-WholeBodyControl"))).expanduser()
for p in (ROOT / "src", _GR00T):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phi0.deploy.robot_proprio import unified_from_g1_debug  # noqa: E402
from phi0.schema.unified_action_schema import (  # noqa: E402
    D_UNIFIED,
    SLICES,
    deploy_robot_proprio_observable_dim_mask,
)


def _sample_g1_debug() -> dict[str, Any]:
    return {
        "base_trans_measured": [0.0, -1.0, 0.793],
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "body_q_measured": (np.arange(29, dtype=np.float32) * 0.01).tolist(),
        "left_hand_q_measured": (np.arange(7, dtype=np.float32) * 0.1).tolist(),
        "right_hand_q_measured": (np.arange(7, dtype=np.float32) * -0.1).tolist(),
        "token_state": np.linspace(-0.2, 0.2, 64, dtype=np.float32).tolist(),
    }


def _capture_g1_debug(host: str, port: int, wait_s: float) -> dict[str, Any]:
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

    sub = ZMQStateSubscriber(host=host, port=int(port))
    deadline = time.monotonic() + float(wait_s)
    try:
        while time.monotonic() < deadline:
            msg = sub.get_msg(clear=True)
            if msg is not None:
                return msg
            time.sleep(0.02)
    finally:
        sub.close()
    raise TimeoutError(f"no g1_debug on tcp://{host}:{port} within {wait_s:.0f}s")


def _slice_report(vec: np.ndarray) -> list[dict[str, Any]]:
    v = np.asarray(vec, dtype=np.float64).reshape(D_UNIFIED)
    rows: list[dict[str, Any]] = []
    for name, (s, e) in SLICES.items():
        chunk = v[s:e]
        nz = np.flatnonzero(np.abs(chunk) > 1e-8)
        rows.append(
            {
                "field": name,
                "start": int(s),
                "end": int(e),
                "size": int(e - s),
                "nonzero_count": int(nz.size),
                "all_zero": bool(nz.size == 0),
                "min": float(chunk.min()) if chunk.size else 0.0,
                "max": float(chunk.max()) if chunk.size else 0.0,
                "mean_abs": float(np.mean(np.abs(chunk))) if chunk.size else 0.0,
                "nonzero_indices_local": nz[:16].tolist(),
                "values_if_small": chunk.tolist() if chunk.size <= 14 else None,
            }
        )
    return rows


def build_snapshot(msg: dict[str, Any]) -> dict[str, Any]:
    d_raw, anchor_root = unified_from_g1_debug(msg)
    vec = np.asarray(d_raw, dtype=np.float32).reshape(D_UNIFIED)
    deploy_mask = deploy_robot_proprio_observable_dim_mask()
    model_visible = vec.copy()
    model_visible[~deploy_mask] = 0.0
    nz_global = np.flatnonzero(np.abs(vec) > 1e-8)
    nz_model = np.flatnonzero(np.abs(model_visible) > 1e-8)
    return {
        "source": "g1_debug",
        "d_unified": D_UNIFIED,
        "anchor_root_world": anchor_root.tolist(),
        "nonzero_count_full": int(nz_global.size),
        "nonzero_indices_full": nz_global.tolist(),
        "nonzero_count_deploy_visible": int(nz_model.size),
        "nonzero_indices_deploy_visible": nz_model.tolist(),
        "deploy_observable_dims": deploy_mask.nonzero()[0].tolist(),
        "slices": _slice_report(vec),
        "vector": vec.tolist(),
        "vector_deploy_visible_only": model_visible.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--wait-s", type=float, default=5.0)
    p.add_argument("--offline", action="store_true", help="Use synthetic g1_debug (no ZMQ).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs/robot_proprio_512_sample.json"),
    )
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        snap = build_snapshot(_sample_g1_debug())
        vec = np.asarray(snap["vector"], dtype=np.float32)
        assert np.allclose(vec[:346], 0.0)
        assert np.allclose(vec[396:], 0.0)
        assert np.any(vec[346:396] != 0)
        assert snap["nonzero_count_full"] == snap["nonzero_count_deploy_visible"]
        print("snapshot_robot_proprio_512 self-check ok")
        return

    msg = _sample_g1_debug() if args.offline else _capture_g1_debug(args.host, args.port, args.wait_s)
    snap = build_snapshot(msg)
    out_json = args.out.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz = out_json.with_suffix(".npz")

    np.savez_compressed(
        out_npz,
        vector=np.asarray(snap["vector"], dtype=np.float32),
        deploy_visible=np.asarray(snap["vector_deploy_visible_only"], dtype=np.float32),
        deploy_mask=deploy_robot_proprio_observable_dim_mask(),
        anchor_root=np.asarray(snap["anchor_root_world"], dtype=np.float32),
    )
    out_json.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_npz}")
    print(
        f"nonzero full={snap['nonzero_count_full']}/{D_UNIFIED} "
        f"deploy_visible={snap['nonzero_count_deploy_visible']}"
    )
    for row in snap["slices"]:
        flag = "ZERO" if row["all_zero"] else f"nnz={row['nonzero_count']}/{row['size']}"
        print(f"  [{row['start']:3d}:{row['end']:3d}] {row['field']:24s} {flag}")


if __name__ == "__main__":
    main()
