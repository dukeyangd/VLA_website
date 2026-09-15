#!/usr/bin/env python3
"""Studio deploy monitor: ZMQ 5555/5556/5557 rates + camera JPEG snapshot.

Writes:
  <out>/status.json   rolling Hz + last summaries
  <out>/camera.jpg    latest ego/wrist panel (if camera port has traffic)

Usage (on deploy host):
  python tools/deploy/studio_zmq_monitor.py --host 127.0.0.1 --out /tmp/studio_zmq_monitor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import msgpack
except ImportError:  # pragma: no cover
    msgpack = None


@dataclass
class PortRate:
    window_s: float = 2.0
    count: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    _times: deque[float] = field(default_factory=deque)

    def tick(self, t: float) -> None:
        self.count += 1
        if self.first_ts is None:
            self.first_ts = t
        self.last_ts = t
        self._times.append(t)
        cutoff = t - self.window_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    def window_hz(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0

    def avg_hz(self) -> float:
        if self.first_ts is None or self.last_ts is None or self.count < 2:
            return 0.0
        dt = self.last_ts - self.first_ts
        return (self.count - 1) / dt if dt > 0 else 0.0


def _decode_images(raw_images: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not raw_images:
        return out
    for key, value in raw_images.items():
        if isinstance(value, (bytes, bytearray)) and cv2 is not None:
            mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            if mat is not None:
                out[key] = mat[..., ::-1]
        elif isinstance(value, np.ndarray):
            out[key] = value
    return out


def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    if len(panels) == 1:
        return panels[0]
    target_h = max(p.shape[0] for p in panels)
    aligned = []
    for p in panels:
        if p.shape[0] == target_h:
            aligned.append(p)
            continue
        scale = target_h / float(p.shape[0])
        w = max(1, int(round(p.shape[1] * scale)))
        aligned.append(cv2.resize(p, (w, target_h), interpolation=cv2.INTER_AREA))
    return np.hstack(aligned)


def _maybe_camera_jpeg(raw: bytes, keys: list[str]) -> bytes | None:
    if msgpack is None or cv2 is None:
        return None
    try:
        data = msgpack.unpackb(raw, raw=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    images = _decode_images(data.get("images") or {})
    panels = []
    for k in keys:
        if k in images:
            rgb = np.asarray(images[k], dtype=np.uint8)
            panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not panels:
        for k, arr in images.items():
            if "depth" in k.lower():
                continue
            rgb = np.asarray(arr, dtype=np.uint8)
            if rgb.ndim >= 2:
                panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)
                break
    if not panels:
        return None
    frame = _stack_panels(panels)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else None


def _pose_hint(raw: bytes) -> str:
    if raw.startswith(b"command"):
        return "command"
    if raw.startswith(b"planner"):
        return "planner"
    if raw.startswith(b"pose"):
        return "pose"
    return raw[:12].decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--camera-port", type=int, default=5555)
    ap.add_argument("--pose-port", type=int, default=5556)
    ap.add_argument("--debug-port", type=int, default=5557)
    ap.add_argument("--out", default="/tmp/studio_zmq_monitor")
    ap.add_argument("--keys", default="ego_view,left_wrist")
    ap.add_argument("--write-interval", type=float, default=0.5)
    ap.add_argument("--no-camera-jpeg", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    cam_path = out / "camera.jpg"
    pid_path = out / "monitor.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()]
    ctx = zmq.Context.instance()
    poller = zmq.Poller()
    sockets: dict[zmq.Socket, tuple[int, str]] = {}

    def add_sub(port: int, label: str) -> None:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.connect(f"tcp://{args.host}:{port}")
        poller.register(sock, zmq.POLLIN)
        sockets[sock] = (port, label)

    add_sub(args.camera_port, "camera")
    add_sub(args.pose_port, "pose")
    add_sub(args.debug_port, "debug")

    rates = {
        args.camera_port: PortRate(),
        args.pose_port: PortRate(),
        args.debug_port: PortRate(),
    }
    labels = {
        args.camera_port: "camera",
        args.pose_port: "pose",
        args.debug_port: "g1_debug",
    }
    last_pose = ""
    last_cam_keys: list[str] = []
    last_write = 0.0

    print(
        f"[studio_monitor] SUB {args.host}:{args.camera_port}/{args.pose_port}/{args.debug_port} "
        f"out={out}",
        flush=True,
    )

    try:
        while True:
            events = dict(poller.poll(200))
            now = time.time()
            for sock, flag in events.items():
                if flag != zmq.POLLIN:
                    continue
                port, label = sockets[sock]
                raw = sock.recv()
                rates[port].tick(now)
                if port == args.pose_port:
                    last_pose = _pose_hint(raw)
                if port == args.camera_port and not args.no_camera_jpeg:
                    jpeg = _maybe_camera_jpeg(raw, keys)
                    if jpeg:
                        cam_path.write_bytes(jpeg)
                        try:
                            data = msgpack.unpackb(raw, raw=False) if msgpack else {}
                            last_cam_keys = list((data.get("images") or {}).keys())
                        except Exception:
                            pass

            if now - last_write >= args.write_interval:
                last_write = now
                payload = {
                    "ok": True,
                    "host": args.host,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "unix": now,
                    "pose_hint": last_pose,
                    "camera_keys": last_cam_keys,
                    "has_camera_jpeg": cam_path.is_file(),
                    "ports": {
                        str(port): {
                            "label": labels[port],
                            "hz": round(rates[port].window_hz(), 2),
                            "avg_hz": round(rates[port].avg_hz(), 2),
                            "count": rates[port].count,
                            "alive": rates[port].window_hz() > 0.05,
                        }
                        for port in sorted(rates)
                    },
                }
                status_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except KeyboardInterrupt:
        pass
    finally:
        for sock in list(sockets):
            sock.close(0)
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
