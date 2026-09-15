#!/usr/bin/env python3
"""Pull ZMQ camera (SensorServer) and serve live MJPEG in the browser.

Usage:
  # terminal A: camera server on :5555
  # terminal B:
  PYTHONPATH=/home/user/workspace/GR00T-WholeBodyControl:$PYTHONPATH \\
    python scripts/view_zmq_camera_mjpeg.py
  # open http://127.0.0.1:8080/
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_GR00T = ROOT.parent / "GR00T-WholeBodyControl"
for p in (ROOT / "src", ROOT, _GR00T):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor  # noqa: E402


def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Side-by-side; resize to common height (ego 480 + wrist 180)."""
    if len(panels) == 1:
        return panels[0]
    target_h = max(p.shape[0] for p in panels)
    aligned: list[np.ndarray] = []
    for p in panels:
        if p.shape[0] == target_h:
            aligned.append(p)
            continue
        scale = target_h / float(p.shape[0])
        w = max(1, int(round(p.shape[1] * scale)))
        aligned.append(cv2.resize(p, (w, target_h), interpolation=cv2.INTER_AREA))
    return np.hstack(aligned)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument(
        "--keys",
        default="ego_view,left_wrist",
        help="Comma-separated image keys to show side-by-side",
    )
    args = ap.parse_args()
    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()]

    lock = threading.Lock()
    jpeg: bytes | None = None

    def pull() -> None:
        nonlocal jpeg
        client = ComposedCameraClientSensor(server_ip=str(args.host), port=int(args.port))
        print(f"[view] SUB tcp://{args.host}:{args.port} keys={keys}")
        while True:
            msg = client.read(blocking=True)
            if not msg or not msg.get("images"):
                time.sleep(0.01)
                continue
            images = msg["images"]
            panels = []
            for k in keys:
                if k not in images:
                    continue
                rgb = np.asarray(images[k], dtype=np.uint8)
                panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not panels:
                continue
            frame = _stack_panels(panels)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            with lock:
                jpeg = buf.tobytes()

    threading.Thread(target=pull, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        data = jpeg
                    if data is not None:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return

    print(f"[view] open http://127.0.0.1:{int(args.http_port)}/")
    ThreadingHTTPServer(("127.0.0.1", int(args.http_port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
