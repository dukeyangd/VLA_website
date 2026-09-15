#!/usr/bin/env python3
"""830 walk MuJoCo vs Newton gap diagnostics (ep125).

Quantifies:
  1) student z sensitivity to body proprio (tape vs live g1_debug)
  2) TRT decoder his_* mismatch (tape history vs post-settle standing)
  3) body joint order roundtrip (parquet MJ ↔ IL ↔ overlay npz)
  4) Newton q_act / q_star coordinate convention

Usage:
  PYTHONPATH=src python tools/eval/debug_830_walk_mujoco_gap.py
  PYTHONPATH=src python tools/eval/debug_830_walk_mujoco_gap.py --ep 125 --out /tmp/830_gap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from phi0.online.student_obs import isaaclab_to_mujoco_dof, mujoco_to_isaaclab_dof

DEFAULT_IL = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
G1_ACTION_SCALE = np.array(
    [
        0.25 * 139 / 25.101925,
        0.25 * 139 / 25.101925,
        0.25 * 88 / 10.17752,
        0.25 * 139 / 25.101925,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 139 / 25.101925,
        0.25 * 139 / 25.101925,
        0.25 * 88 / 10.17752,
        0.25 * 139 / 25.101925,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 88 / 10.17752,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 5 / 4.25,
        0.25 * 5 / 4.25,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 25 / 3.609725,
        0.25 * 5 / 4.25,
        0.25 * 5 / 4.25,
    ],
    dtype=np.float32,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified")
DECODER_ONNX = ROOT / "subpackages/gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx"

# Logged CL token0 from prior runs (chunk_cl replay.log).
CL_TOKENS = {
    "alltape_t0": 0.001,
    "tapehand2_t0": -0.066,
    "offline_openloop_t0": None,
}


def _hist_window(seq: np.ndarray, t: int, n: int = 10) -> np.ndarray:
    out = np.zeros((n, seq.shape[1]), np.float32)
    for k in range(n):
        ti = max(0, t - (n - 1 - k))
        out[k] = seq[ti]
    return out.reshape(-1)


def _il_rel_from_mj_abs(mj_q: np.ndarray) -> np.ndarray:
    rel = np.zeros(29, dtype=np.float32)
    for il_i in range(29):
        mj_i = int(MUJOCO_TO_ISAACLAB[il_i])
        rel[il_i] = mj_q[mj_i] - DEFAULT_IL[il_i]
    return rel


def _decoder_action(z64: np.ndarray, hist: dict[str, np.ndarray], session) -> np.ndarray:
    obs = np.concatenate(
        [
            z64.astype(np.float32),
            hist["ang"],
            hist["q"],
            hist["dq"],
            hist["act"],
            hist["grav"],
        ]
    )[None, :]
    inp = session.get_inputs()[0].name
    return session.run(None, {inp: obs})[0].reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ep", type=int, default=125)
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--newton-npz", type=Path, default=None)
    p.add_argument("--offline-z-npz", type=Path, default=Path("/tmp/830_ep125_offline_z.npz"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ep = int(args.ep)
    parquet = args.parquet or (DATASET / "data/chunk-000" / f"file-{ep}.parquet")
    newton_npz = args.newton_npz or (
        ROOT
        / "experiments/830_walk_student_cl_newton_ep125_20260827_015707/infer_qpos_traj_student.npz"
    )

    ua = np.stack(
        [
            np.asarray(x, np.float32)
            for x in pq.read_table(parquet, columns=["action.unified"]).column("action.unified").to_pylist()
        ]
    )
    body_mj = ua[:, 367:396]
    z_gt = ua[:, 396:460]
    grav = ua[:, 360:363]
    body_il_abs = mujoco_to_isaaclab_dof(body_mj).numpy()
    body_il_rel = body_il_abs - DEFAULT_IL
    body_dq = np.zeros_like(body_il_rel)
    body_dq[1:] = (body_il_abs[1:] - body_il_abs[:-1]) * 50.0
    body_dq[0] = body_dq[1]
    act_zeros = np.zeros((len(ua), 29), np.float32)
    ang_zeros = np.zeros((len(ua), 3), np.float32)

    report: dict = {"ep": ep, "T": int(len(ua)), "parquet": str(parquet)}

    # --- order roundtrip ---
    il0 = _il_rel_from_mj_abs(body_mj[0])
    report["body_order"] = {
        "parquet_mj0_j0": float(body_mj[0, 0]),
        "il_rel0_j0": float(body_il_rel[0, 0]),
        "il_rel_cpp_roundtrip_mse": float(np.mean((il0 - body_il_rel[0]) ** 2)),
        "standing_il_rel_l2_vs_tape_t0": float(np.linalg.norm(body_il_rel[0])),
    }

    # --- offline student z ---
    if args.offline_z_npz.is_file():
        ol = np.load(args.offline_z_npz)["z_pred"][:, 0, :]
        CL_TOKENS["offline_openloop_t0"] = float(ol[0, 0])
        report["student_z"] = {
            "offline_vs_gt_mse_all": float(np.mean((ol - z_gt) ** 2)),
            "offline_vs_gt_cos_mean": float(
                np.mean(
                    [
                        float(np.dot(ol[t], z_gt[t]) / (np.linalg.norm(ol[t]) * np.linalg.norm(z_gt[t]) + 1e-8))
                        for t in range(len(z_gt))
                    ]
                )
            ),
            "tokens_t0": {k: v for k, v in CL_TOKENS.items()},
        }

    # --- Newton coords ---
    if newton_npz.is_file():
        nw = np.load(newton_npz)
        q_act = nw["q_act"][:, 0, :]
        q_star = nw["q_star"][:, 0, :]
        z_pred = nw["z_pred"][:, 0, :]
        report["newton"] = {
            "q_act_vs_tape_il_mse": float(np.mean((q_act - body_il_abs) ** 2)),
            "q_star_vs_tape_il_mse": float(np.mean((q_star - body_il_abs) ** 2)),
            "z_pred_vs_gt_mse": float(np.mean((z_pred - z_gt) ** 2)),
            "coord_note": "q_act/q_star are IsaacLab order (not MJ)",
        }

    # --- live g1_debug from CL log ---
    live_mj0 = -0.331  # tapehand2 chunk_cl replan t=0
    report["live_vs_tape"] = {
        "tape_mj0": float(body_mj[0, 0]),
        "live_g1_debug_mj0_after_settle": live_mj0,
        "delta_mj0": float(live_mj0 - body_mj[0, 0]),
        "live_il_rel0_j0_approx": float(_il_rel_from_mj_abs(np.array([live_mj0] + body_mj[0, 1:].tolist(), np.float32))[0]),
    }

    # --- ONNX decoder counterfactual (same z_gt, different his_*) ---
    if DECODER_ONNX.is_file():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
        stand = {
            "ang": np.zeros(30, np.float32),
            "q": np.tile(np.zeros(29, np.float32), 10),
            "dq": np.zeros(290, np.float32),
            "act": np.zeros(290, np.float32),
            "grav": np.tile(np.array([0.0, 0.0, -1.0], np.float32), 10),
        }
        dec_rows = []
        for t in (0, 50, 100):
            tape = {
                "ang": _hist_window(ang_zeros, t),
                "q": _hist_window(body_il_rel, t),
                "dq": _hist_window(body_dq, t),
                "act": _hist_window(act_zeros, t),
                "grav": _hist_window(grav, t),
            }
            z = z_gt[t]
            a_tape = _decoder_action(z, tape, sess)
            a_stand = _decoder_action(z, stand, sess)
            q_tape_il = DEFAULT_IL + a_tape * G1_ACTION_SCALE
            q_stand_il = DEFAULT_IL + a_stand * G1_ACTION_SCALE
            dec_rows.append(
                {
                    "t": t,
                    "z0": float(z[0]),
                    "action_l2_tape_vs_stand": float(np.linalg.norm(a_tape - a_stand)),
                    "q_target_il_l2_tape_vs_stand": float(np.linalg.norm(q_tape_il - q_stand_il)),
                    "note": "last_actions=0 ponytail; gravity from parquet tape for tape arm",
                }
            )
        report["decoder_counterfactual"] = {
            "onnx": str(DECODER_ONNX),
            "obs_layout": "994=token64+ang30+q290+dq290+act290+grav30 (IL rel q in his_*)",
            "frames": dec_rows,
            "gt_replay_interpretation": (
                "GT streams correct z* but C++ state_logger his_* comes from live sim "
                "after RECORD_SETTLE standing (~IL rel≈0), not walk tape history."
            ),
        }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
