#!/usr/bin/env python3
"""Bind ZMQ PUB on port 5580 and forward stdin keys (gear_sonic launch_inference keyboard pane)."""

from __future__ import annotations

import argparse
import time

import zmq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: all interfaces via 127.0.0.1)")
    p.add_argument("--port", type=int, default=5580)
    args = p.parse_args()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.host}:{args.port}")
    time.sleep(0.5)
    print(
        f"Keyboard publisher ready on tcp://{args.host}:{args.port}\n"
        "Keys: p=pause/resume, k=start/stop control loop, i=initial pose, [/]=toggle hands, t <text>=prompt"
    )
    try:
        while True:
            key = input()
            if key.startswith("t "):
                pub.send_string("prompt:" + key[2:])
                print("Sent prompt:", key[2:])
            else:
                pub.send_string(key)
                print("Sent:", key)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pub.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
