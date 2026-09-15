#!/usr/bin/env python3
"""Grab one (or N) g1_debug msgpack messages from deploy ZMQ :5557 and save JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
_GR00T = ROOT.parent / "GR00T-WholeBodyControl"
for p in (ROOT / "src", ROOT, _GR00T):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC  # noqa: E402


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _recv_g1_debug(sock: zmq.Socket) -> dict[str, Any]:
    raw = sock.recv()
    msg = msgpack.unpackb(raw[len(STATE_ZMQ_TOPIC) :], raw=False)
    if not isinstance(msg, dict):
        raise TypeError(f"expected dict g1_debug payload, got {type(msg)}")
    out: dict[str, Any] = {}
    for key, val in msg.items():
        if isinstance(val, (list, tuple)):
            out[str(key)] = np.asarray(val)
        else:
            out[str(key)] = val
    return out


def capture_g1_debug(
    *,
    host: str,
    port: int,
    wait_s: float,
    num_frames: int,
) -> list[dict[str, Any]]:
    mnp.patch()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    time.sleep(0.2)

    frames: list[dict[str, Any]] = []
    deadline = time.monotonic() + float(wait_s)
    try:
        while len(frames) < int(num_frames) and time.monotonic() < deadline:
            try:
                frames.append(_recv_g1_debug(sock))
            except zmq.Again:
                continue
    finally:
        sock.close(linger=0)

    if not frames:
        raise TimeoutError(
            f"no g1_debug on tcp://{host}:{port} within {wait_s:.0f}s "
            f"(deploy in CONTROL? topic={STATE_ZMQ_TOPIC!r})"
        )
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--wait-s", type=float, default=30.0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs/g1_debug_snapshot.json"),
    )
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        assert _jsonify(np.array([1.0, 2.0])) == [1.0, 2.0]
        print("g1_debug json self-check ok")
        return

    frames = capture_g1_debug(
        host=str(args.host),
        port=int(args.port),
        wait_s=float(args.wait_s),
        num_frames=int(args.num_frames),
    )
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: Any
    if len(frames) == 1:
        payload = _jsonify(frames[0])
    else:
        payload = [_jsonify(f) for f in frames]

    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = sorted(frames[-1].keys())
    print(
        f"saved {out} frames={len(frames)} keys={len(keys)} "
        f"sample={keys[:8]}{'...' if len(keys) > 8 else ''}"
    )


if __name__ == "__main__":
    main()
