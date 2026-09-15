#!/usr/bin/env python3
"""Subscribe ZMQ camera and write latest JPEG tiles for Studio MJPEG proxy.

Intended to run with GR00T-WholeBodyControl-hand .venv_sim (has zmq/cv2/msgpack).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

CAM_PREFER = (
    "ego_view",
    "head",
    "head_color_image",
    "left_wrist",
    "left_wrist_color_image",
    "right_wrist",
)


def decode_images(raw_images: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in (raw_images or {}).items():
        if isinstance(value, (bytes, bytearray)):
            mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            if mat is None:
                continue
            out[key] = mat[..., ::-1]  # BGR -> RGB
        elif isinstance(value, str):
            import base64

            mat = cv2.imdecode(
                np.frombuffer(base64.b64decode(value), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if mat is None:
                continue
            out[key] = mat[..., ::-1]
        elif isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, dict) and b"nd" in value:
            out[key] = m.decode(value)
    return out


def to_bgr_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    arr = np.asarray(img)
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = arr
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def pick_keys(images: dict[str, np.ndarray]) -> list[str]:
    keys = [k for k in images if "depth" not in k.lower()]
    chosen: list[str] = []
    for prefer in CAM_PREFER:
        for k in keys:
            if k == prefer or k.startswith(prefer + "_"):
                if k not in chosen:
                    chosen.append(k)
                break
        if len(chosen) >= 2:
            break
    for k in keys:
        if k not in chosen:
            chosen.append(k)
        if len(chosen) >= 2:
            break
    return chosen[:2] if chosen else list(images.keys())[:2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera-host", default="192.168.123.165")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE"
    if done.exists():
        done.unlink()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVHWM, 3)
    sock.connect(f"tcp://{args.camera_host}:{args.camera_port}")
    print(f"[studio_camera] connect tcp://{args.camera_host}:{args.camera_port}", flush=True)

    period = 1.0 / max(args.fps, 1.0)
    selected: list[str] = []
    while not done.exists():
        t0 = time.time()
        if sock.poll(50):
            packed = sock.recv()
            data = msgpack.unpackb(packed, object_hook=m.decode)
            images = decode_images((data or {}).get("images") or {})
            if images:
                if not selected:
                    selected = pick_keys(images)
                    (out / "cameras.json").write_text(
                        json.dumps({"keys": selected}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[studio_camera] keys={selected}", flush=True)
                for name in selected:
                    img = images.get(name)
                    if img is None:
                        continue
                    jpeg = to_bgr_jpeg(img)
                    tmp = out / f".{name}.jpg.tmp"
                    final = out / f"{name}.jpg"
                    tmp.write_bytes(jpeg)
                    tmp.replace(final)
        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    sock.close(0)
    print("[studio_camera] exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
