#!/usr/bin/env python3
"""Compare 830 walk vision: robot/T3 ZMQ camera vs disk RefVideoFrameSource.

Reports schema (keys/shapes), JPEG wire roundtrip loss (disk→ZMQ codec→RGB),
and optional live ZMQ capture stats. Live robot frames rarely pixel-match episode
mp4 (different scene/time); pipeline/format alignment is the main goal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PHI0 = Path(__file__).resolve().parents[2]
for root in (_PHI0 / "src", _PHI0 / "subpackages"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor  # noqa: E402
from gear_sonic.camera.sensor_server import ImageMessageSchema  # noqa: E402
from phi0.online.ref_video_vlm import RefVideoFrameSource  # noqa: E402


def _stats(name: str, rgb: np.ndarray) -> dict:
    arr = np.asarray(rgb, dtype=np.uint8)
    return {
        "name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        return {"shape_match": False, "a": x.shape, "b": y.shape}
    d = np.abs(x - y)
    mse = float(np.mean(d * d))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 * 255.0 / mse))
    return {
        "shape_match": True,
        "mae": float(d.mean()),
        "max_abs": float(d.max()),
        "rmse": float(np.sqrt(mse)),
        "psnr_db": psnr,
        "pct_px_zero_diff": float(100.0 * np.mean(d == 0)),
    }


def _jpeg_wire_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Same path as composed camera server/client (JPEG in msgpack)."""
    key = "ego_view"
    msg = ImageMessageSchema(
        timestamps={key: time.time()},
        images={key: np.asarray(rgb, dtype=np.uint8)},
    ).serialize()
    out = ImageMessageSchema.deserialize(msg).images[key]
    return np.asarray(out, dtype=np.uint8)


def _decode_disk_pair(
    src: RefVideoFrameSource,
    episode: int,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    ego_t, chest_t = src.frames_at(
        np.array([episode], dtype=np.int64),
        np.array([frame_idx], dtype=np.int64),
    )
    # [1,1,C,H,W] float → HWC uint8
    def _to_hwc(t) -> np.ndarray:
        x = t[0, 0].detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    return _to_hwc(ego_t), _to_hwc(chest_t)


def _capture_zmq(host: str, port: int, wait_s: float, n: int) -> list[dict]:
    client = ComposedCameraClientSensor(server_ip=host, port=int(port))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        msg = client.read(blocking=False)
        if msg and msg.get("images"):
            break
        time.sleep(0.02)
    else:
        raise TimeoutError(f"no ZMQ frames from tcp://{host}:{port} within {wait_s}s")

    frames: list[dict] = []
    for _ in range(n):
        msg = client.read(blocking=False)
        if msg and msg.get("images"):
            frames.append(msg)
        time.sleep(0.02)
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()
    return frames


def _pick(images: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for k in keys:
        if k in images:
            return np.asarray(images[k], dtype=np.uint8)
    return None


_EGO_KEYS = ("ego_view", "head", "observation.images.ego_view")
_CHEST_KEYS = ("chest_forward", "left_wrist", "observation.images.left_wrist")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ref-root",
        type=Path,
        default=Path("/home/user/workspace/datasets/830/skill_walk_to_black_box_new_unified"),
    )
    p.add_argument("--episode", type=int, default=125)
    p.add_argument(
        "--frame-indices",
        type=int,
        nargs="+",
        default=[0, 50, 100, 200, 426],
        help="Frame indices within episode mp4 (0..426 for EP125)",
    )
    p.add_argument("--zmq-host", default="192.168.123.165")
    p.add_argument("--zmq-port", type=int, default=5555)
    p.add_argument("--zmq-wait-s", type=float, default=15.0)
    p.add_argument("--zmq-samples", type=int, default=5)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/user/workspace/Phi_0_wpy/experiments/cl_830_vision_align_report.txt"),
    )
    args = p.parse_args()

    lines: list[str] = []
    ep = int(args.episode)

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    log("=== 830 vision align: ZMQ camera vs disk ===")
    log(f"ref_root={args.ref_root}")
    log(f"episode={ep} frame_indices={args.frame_indices}")
    log(f"zmq=tcp://{args.zmq_host}:{args.zmq_port}")
    log()

    # --- Disk path ---
    log("--- [1] Disk RefVideoFrameSource (mp4 h264 decode) ---")
    disk = RefVideoFrameSource(args.ref_root, fps=50.0, video_backend="opencv")
    for fi in args.frame_indices:
        ego, chest = _decode_disk_pair(disk, ep, fi)
        log(f"  frame={fi} ego {_stats('ego', ego)}")
        log(f"  frame={fi} chest(left_wrist) {_stats('chest', chest)}")
        ej = _jpeg_wire_roundtrip(ego)
        cj = _jpeg_wire_roundtrip(chest)
        de = _diff(ego, ej)
        dc = _diff(chest, cj)
        log(f"  frame={fi} ego disk vs JPEG-wire: {de}")
        log(f"  frame={fi} chest disk vs JPEG-wire: {dc}")
    log()

    # --- Live ZMQ ---
    log("--- [2] Live ZMQ composed camera ---")
    try:
        zmq_frames = _capture_zmq(
            args.zmq_host, args.zmq_port, args.zmq_wait_s, args.zmq_samples
        )
        all_keys: set[str] = set()
        depth_keys: list[str] = []
        for i, msg in enumerate(zmq_frames):
            imgs = msg.get("images") or {}
            keys = sorted(imgs.keys())
            all_keys.update(keys)
            depth_keys.extend([k for k in keys if "depth" in k.lower()])
            ts = msg.get("timestamps") or {}
            log(f"  sample={i} keys={keys} timestamps={list(ts.keys())}")
            ego = _pick(imgs, _EGO_KEYS)
            chest = _pick(imgs, _CHEST_KEYS)
            if ego is not None:
                log(f"    ego {_stats('ego', ego)}")
            else:
                log("    ego MISSING")
            if chest is not None:
                log(f"    chest {_stats('chest', chest)}")
            else:
                log("    chest(left_wrist) MISSING")
        log(f"  union_keys={sorted(all_keys)}")
        log(f"  depth_keys_seen={sorted(set(depth_keys)) or 'none'}")
        log("  note: live robot pixels != episode mp4 (different scene/time)")
        # Compare live ego to disk frame 0 (format only — expect large diff)
        if zmq_frames:
            imgs0 = zmq_frames[0].get("images") or {}
            live_ego = _pick(imgs0, _EGO_KEYS)
            live_chest = _pick(imgs0, _CHEST_KEYS)
            disk_ego, disk_chest = _decode_disk_pair(disk, ep, 0)
            if live_ego is not None:
                log(f"  live_ego vs disk_ep{ep}_frame0: {_diff(live_ego, disk_ego)}")
            if live_chest is not None:
                log(f"  live_chest vs disk_ep{ep}_frame0: {_diff(live_chest, disk_chest)}")
    except Exception as exc:
        log(f"  ZMQ capture FAILED: {exc}")
    log()

    # --- Student tensor path equivalence (disk vs zmq codec on same pixels) ---
    log("--- [3] Student input tensor path (float [0,1] B1CHW) ---")
    ego, chest = _decode_disk_pair(disk, ep, args.frame_indices[0])
    # ZmqDualCameraFrameSource equivalent
    ego_f = ego.astype(np.float32) / 255.0
    ego_b1chw = np.transpose(ego_f, (2, 0, 1))[None, None]
    ego_j = _jpeg_wire_roundtrip(ego)
    ego_j_f = ego_j.astype(np.float32) / 255.0
    ego_j_b1chw = np.transpose(ego_j_f, (2, 0, 1))[None, None]
    log(f"  disk tensor shape={ego_b1chw.shape} range=[{ego_b1chw.min():.4f},{ego_b1chw.max():.4f}]")
    log(f"  after JPEG-wire shape={ego_j_b1chw.shape} diff={_diff(ego_b1chw, ego_j_b1chw)}")
    log()

    log("--- Summary ---")
    log("  Views used by student: ego_view + left_wrist (chest_forward in VLM)")
    log("  Depth: NOT used by 830 student (RGB 3ch only)")
    log("  disk: indexed by ref episode_index + frame_index")
    log("  zmq: latest frame on wire (no ref index); JPEG codec on wire")
    log("  Same-content pipeline loss = disk vs JPEG-wire (section 1)")
    log("  Live robot vs disk frame0 = scene mismatch (section 2); large MAE expected")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
