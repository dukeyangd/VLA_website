#!/usr/bin/env python3
"""Tap live 830 TRT closed-loop I/O: camera, g1_debug proprio, student :5556 output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import msgpack
import numpy as np
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
for root in (_PHI0 / "src", _PHI0 / "subpackages"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.camera.sensor_server import ImageMessageSchema  # noqa: E402
from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC  # noqa: E402
from gear_sonic.utils.zmq_pose_unpack import unpack_pose_message  # noqa: E402
from phi0.deploy.dex3_gripper import dex3_actuator_to_policy_hand  # noqa: E402
from phi0.deploy.robot_proprio import _body_q29, gripper14_wbc_from_g1_debug  # noqa: E402
from phi0.hand.hand_mode import student_proprio_dim  # noqa: E402


def _summ(x: np.ndarray, n: int = 6) -> dict:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    d = {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}
    d["head"] = a[:n].tolist()
    return d


def _policy14_from_zmq(l7: np.ndarray, r7: np.ndarray) -> np.ndarray:
    l = dex3_actuator_to_policy_hand(np.asarray(l7, dtype=np.float32).reshape(7))
    r = dex3_actuator_to_policy_hand(np.asarray(r7, dtype=np.float32).reshape(7))
    return np.concatenate([l, r]).astype(np.float32)


@dataclass
class IoSample:
    t_mono: float
    frame_index: int
    token0: float
    token_mean: float
    token_std: float
    left7: list[float]
    right7: list[float]
    body29_head: list[float]
    hand_proprio14_head: list[float]
    g1_hand_meas_head: list[float]
    ego_mean: float
    chest_mean: float
    ego_std: float


@dataclass
class MonitorState:
    commanded_hand14: np.ndarray = field(
        default_factory=lambda: np.zeros(14, dtype=np.float32)
    )
    samples: list[IoSample] = field(default_factory=list)
    n_cam: int = 0
    n_dbg: int = 0
    n_pose: int = 0
    last_ego: dict | None = None
    last_g1: dict | None = None


def _parse_g1_debug(raw: bytes) -> dict | None:
    if not raw.startswith(STATE_ZMQ_TOPIC.encode("utf-8")):
        return None
    payload = raw[len(STATE_ZMQ_TOPIC) :]
    msg = msgpack.unpackb(payload, raw=False)
    return msg if isinstance(msg, dict) else None


def _parse_camera(raw: bytes) -> dict | None:
    data = msgpack.unpackb(raw, raw=False)
    if not isinstance(data, dict):
        return None
    return ImageMessageSchema.deserialize(data).asdict()


def _parse_pose(raw: bytes) -> dict | None:
    if not raw.startswith(b"pose"):
        return None
    return unpack_pose_message(raw, topic="pose")


def run(
    *,
    camera_host: str,
    camera_port: int,
    debug_host: str,
    debug_port: int,
    pose_host: str,
    pose_port: int,
    duration_s: float,
    print_every: int,
    hand_obs: str,
    out: Path,
) -> None:
    ctx = zmq.Context()
    poller = zmq.Poller()

    cam = ctx.socket(zmq.SUB)
    cam.connect(f"tcp://{camera_host}:{camera_port}")
    cam.setsockopt_string(zmq.SUBSCRIBE, "")
    poller.register(cam, zmq.POLLIN)

    dbg = ctx.socket(zmq.SUB)
    dbg.connect(f"tcp://{debug_host}:{debug_port}")
    dbg.setsockopt_string(zmq.SUBSCRIBE, "")
    poller.register(dbg, zmq.POLLIN)

    pose = ctx.socket(zmq.SUB)
    pose.connect(f"tcp://{pose_host}:{pose_port}")
    pose.setsockopt_string(zmq.SUBSCRIBE, "pose")
    poller.register(pose, zmq.POLLIN)

    st = MonitorState()
    proprio_d = student_proprio_dim()
    lines: list[str] = []
    t_end = time.monotonic() + duration_s
    last_print_frame = -1

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    log(
        f"[io_mon] camera=tcp://{camera_host}:{camera_port} "
        f"g1_debug=tcp://{debug_host}:{debug_port} pose=tcp://{pose_host}:{pose_port} "
        f"hand_obs={hand_obs} proprio_dim={proprio_d} duration={duration_s}s"
    )

    while time.monotonic() < t_end:
        events = dict(poller.poll(timeout=100))
        if cam in events:
            raw = cam.recv(zmq.NOBLOCK)
            st.n_cam += 1
            cam_d = _parse_camera(raw)
            if cam_d:
                st.last_ego = cam_d
        if dbg in events:
            raw = dbg.recv(zmq.NOBLOCK)
            g1 = _parse_g1_debug(raw)
            if g1 is not None:
                st.n_dbg += 1
                st.last_g1 = g1
        if pose in events:
            raw = pose.recv(zmq.NOBLOCK)
            msg = _parse_pose(raw)
            if msg is None:
                continue
            st.n_pose += 1
            tok = np.asarray(msg.get("token_state", msg.get("tokens", [])), dtype=np.float32).reshape(-1)
            fi = int(np.asarray(msg.get("frame_index", [-1])).reshape(-1)[0])
            l7 = np.asarray(msg["left_hand_joints"], dtype=np.float32).reshape(7)
            r7 = np.asarray(msg["right_hand_joints"], dtype=np.float32).reshape(7)
            out_policy14 = _policy14_from_zmq(l7, r7)

            body29 = g1_hand_meas = None
            if st.last_g1 is not None:
                body29 = _body_q29(st.last_g1)
                try:
                    g1_hand_meas = gripper14_wbc_from_g1_debug(st.last_g1)
                except KeyError:
                    g1_hand_meas = None

            if hand_obs == "commanded":
                hand_in = st.commanded_hand14
            elif hand_obs == "live" and g1_hand_meas is not None:
                hand_in = g1_hand_meas
            else:
                hand_in = st.commanded_hand14

            ego_mean = chest_mean = ego_std = float("nan")
            if st.last_ego and st.last_ego.get("images"):
                imgs = st.last_ego["images"]
                if "ego_view" in imgs:
                    e = np.asarray(imgs["ego_view"], dtype=np.uint8)
                    ego_mean, ego_std = float(e.mean()), float(e.std())
                if "left_wrist" in imgs:
                    chest_mean = float(np.asarray(imgs["left_wrist"], dtype=np.uint8).mean())

            samp = IoSample(
                t_mono=time.monotonic(),
                frame_index=fi,
                token0=float(tok[0]) if tok.size else float("nan"),
                token_mean=float(tok.mean()) if tok.size else float("nan"),
                token_std=float(tok.std()) if tok.size else float("nan"),
                left7=l7.tolist(),
                right7=r7.tolist(),
                body29_head=body29[:6].tolist() if body29 is not None else [],
                hand_proprio14_head=hand_in[:6].tolist(),
                g1_hand_meas_head=g1_hand_meas[:6].tolist() if g1_hand_meas is not None else [],
                ego_mean=ego_mean,
                chest_mean=chest_mean,
                ego_std=ego_std,
            )
            st.samples.append(samp)
            st.commanded_hand14 = out_policy14.copy()

            if fi >= 0 and (fi - last_print_frame) >= print_every:
                last_print_frame = fi
                b0 = body29[0] if body29 is not None else float("nan")
                g1h0 = g1_hand_meas[0] if g1_hand_meas is not None else float("nan")
                log(
                    f"[io_mon] frame={fi} "
                    f"vision egoμ={ego_mean:.1f}σ={ego_std:.1f} chestμ={chest_mean:.1f} | "
                    f"proprio body0={b0:+.3f} hand_in0={hand_in[0]:+.3f} "
                    f"g1_hand0={g1h0:+.3f} | "
                    f"out token0={samp.token0:+.4f} tokμ={samp.token_mean:+.4f} "
                    f"L7={l7[:3].round(2).tolist()} R7={r7[:3].round(2).tolist()}"
                )

    # Analysis
    log("")
    log("=== 分析 ===")
    log(f"收包 rate: camera={st.n_cam} g1_debug={st.n_dbg} pose={st.n_pose} over {duration_s}s")
    if st.samples:
        fis = [s.frame_index for s in st.samples]
        t0s = [s.token0 for s in st.samples]
        log(f"pose frame_index: {min(fis)}..{max(fis)} (n={len(st.samples)})")
        log(f"token[0]: min={min(t0s):+.4f} max={max(t0s):+.4f} mean={np.mean(t0s):+.4f} std={np.std(t0s):.4f}")
        dt = np.diff([s.frame_index for s in st.samples[:200]])
        if dt.size:
            log(f"frame_index step: unique={sorted(set(dt.tolist()))[:8]}")

        # commanded vs g1 measured hand
        if hand_obs == "commanded":
            diffs = []
            for s in st.samples:
                if s.g1_hand_meas_head:
                    # full vectors stored partially; recompute from last sample fields insufficient
                    pass
            log("hand_obs=commanded: student proprio hand = 上一帧 model 输出 (非 g1_debug 实测)")

        last = st.samples[-1]
        log(f"末帧 frame={last.frame_index} token0={last.token0:+.4f} L7={last.left7} R7={last.right7}")

        # token drift
        if len(t0s) >= 10:
            drift = t0s[-1] - t0s[0]
            log(f"token[0] drift (first→last): {drift:+.4f}")

        # vision stability
        ego_means = [s.ego_mean for s in st.samples if not np.isnan(s.ego_mean)]
        if ego_means:
            log(f"ego mean pixel: {np.mean(ego_means):.1f} ± {np.std(ego_means):.1f} (scene stable if std small)")

    if st.last_ego and st.last_ego.get("images"):
        keys = sorted(st.last_ego["images"].keys())
        log(f"camera keys: {keys}")
        for k in keys:
            a = np.asarray(st.last_ego["images"][k])
            ch = a.shape[-1] if a.ndim == 3 else "?"
            log(f"  {k}: shape={a.shape} ch={ch} RGB={'yes' if ch==3 else 'no'}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "samples": [s.__dict__ for s in st.samples[-min(200, len(st.samples)):]],
        "counts": {"camera": st.n_cam, "g1_debug": st.n_dbg, "pose": st.n_pose},
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"\nWrote {out} and {out.with_suffix('.json')}")

    cam.close(); dbg.close(); pose.close(); ctx.term()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-host", default="192.168.123.165")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--debug-host", default="127.0.0.1")
    p.add_argument("--debug-port", type=int, default=5557)
    p.add_argument("--pose-host", default="127.0.0.1")
    p.add_argument("--pose-port", type=int, default=5556)
    p.add_argument("--duration-s", type=float, default=12.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--hand-obs", default="commanded", choices=("commanded", "live", "tape"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/user/workspace/Phi_0_wpy/experiments/cl_830_3term_ep125_local/io_monitor_report.txt"),
    )
    args = p.parse_args()
    run(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        debug_host=args.debug_host,
        debug_port=args.debug_port,
        pose_host=args.pose_host,
        pose_port=args.pose_port,
        duration_s=args.duration_s,
        print_every=args.print_every,
        hand_obs=args.hand_obs,
        out=args.out,
    )


if __name__ == "__main__":
    main()
