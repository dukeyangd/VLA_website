#!/usr/bin/env python3
"""Replay pick-tissue GT SONIC motion_token (+ hands) over ZMQ pose v4 only."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
# Prefer the in-tree subpackages used by run_sonic_latent_sim_eval.sh.  The
# sibling GR00T checkout is only a legacy fallback and can carry an older ZMQ
# v4 helper without frame_offset/history-handshake support.  Remove existing
# PYTHONPATH occurrences before inserting so a fallback can never jump ahead.
_GEAR_ROOT = _PHI0 / "subpackages"
if not (_GEAR_ROOT / "gear_sonic").is_dir():
    _GEAR_ROOT = _GR00T
for root in reversed((_PHI0 / "src", _GEAR_ROOT)):
    root_str = str(root)
    while root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message  # noqa: E402

from phi0.deploy.sonic_latent_gt_replay import (  # noqa: E402
    build_replay_messages,
    load_sonic_latent_replay_arrays,
)


@dataclass
class ReplayConfig:
    parquet: Path
    token_source: str = "auto"
    """auto | valid_column | unified_slice"""
    valid_parquet_for_hands: Path | None = None
    """When parquet is unified-only, load teleop hands from valid GR00T parquet."""
    hand_source: str = "auto"
    """auto | unified_gripper | measured_state (deploy-order Dex3 from observation.state)."""
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    fps: float = 50.0
    max_frames: int | None = None
    start_frame: int = 0
    """Skip dataset rows before this frame; outgoing frame_index keeps the offset."""
    start_delay_s: float = 0.5
    ready_flag: str = ""
    """Wait for this flag before streaming pose frames (after RECORD_START for GT panel f0)."""
    arm_flag: str = ""
    """Wait for this flag, then arm-only + start_streamed (CONTROL before ENTER)."""
    hand_ramp_frames: int = 40
    gt_decoder_frame_path: str = ""
    """If set, write 0-based frame index each pose send (GT LowState overlay sync)."""
    history_seed_go_path: str = ""
    """Optional flag that requests an atomic C++ decoder-history seed."""
    history_seed_ack_path: str = ""
    """ACK written by C++ after the history seed has replaced live stand history."""
    history_seed_timeout_s: float = 30.0


def _wait_flag(path: Path, *, label: str, timeout_s: float = 240.0) -> None:
    print(f"[pick_tissue_sonic_latent] waiting for {label} {path}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{label} flag not found: {path}")


def _send_deploy_start_commands(pub: zmq.Socket) -> None:
    # Match phi-0-wbc-newton GT replay: planner start then streamed motion.
    pub.send(build_command_message(start=True, stop=False, planner=True))
    time.sleep(0.2)
    pub.send(build_command_message(start=True, stop=False, planner=False))
    time.sleep(0.2)
    print("[pick_tissue_sonic_latent] sent ZMQ command start (planner -> streamed motion)")


def main(config: ReplayConfig) -> None:
    tokens, left, right, token_source = load_sonic_latent_replay_arrays(
        config.parquet,
        token_source=config.token_source,
        hand_source=config.hand_source,
        valid_parquet_for_hands=config.valid_parquet_for_hands,
        max_frames=config.max_frames,
    )
    start = int(config.start_frame)
    if start < 0 or start >= len(tokens):
        raise ValueError(f"start_frame={start} outside [0,{len(tokens)})")
    tokens, left, right = tokens[start:], left[start:], right[start:]
    n = len(tokens)
    messages = build_replay_messages(
        tokens,
        left,
        right,
        hand_ramp_frames=config.hand_ramp_frames,
        frame_offset=start,
    )

    print(
        f"[pick_tissue_sonic_latent] parquet={config.parquet.name} frames={n} "
        f"start_frame={start} "
        f"token_source={token_source} "
        f"L_hand_max={float(abs(left).max()):.3f} R_hand_max={float(abs(right).max()):.3f}"
    )

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{config.zmq_host}:{config.zmq_port}")
    time.sleep(0.5)
    print(f"[pick_tissue_sonic_latent] bound tcp://{config.zmq_host}:{config.zmq_port}")

    # Match newton: start_streamed on arm and again on ready (READY still gates tokens).
    if config.arm_flag:
        _wait_flag(Path(config.arm_flag), label="arm")
        time.sleep(config.start_delay_s)
        _send_deploy_start_commands(pub)

    if config.ready_flag:
        _wait_flag(Path(config.ready_flag), label="ready")
        if not config.history_seed_go_path:
            time.sleep(config.start_delay_s)
            _send_deploy_start_commands(pub)

    if config.history_seed_go_path:
        go = Path(config.history_seed_go_path)
        ack = Path(config.history_seed_ack_path)
        if not config.history_seed_ack_path:
            raise ValueError("history_seed_ack_path is required with history_seed_go_path")
        ack.unlink(missing_ok=True)
        go.parent.mkdir(parents=True, exist_ok=True)
        go.write_text("seed\n", encoding="utf-8")
        print(f"[pick_tissue_sonic_latent] history seed requested -> {go}")
        deadline = time.monotonic() + float(config.history_seed_timeout_s)
        # Re-publish the first token until C++ observes token+GO in one control
        # snapshot, atomically seeds StateLogger, and writes ACK.
        while time.monotonic() < deadline and not ack.is_file():
            pub.send(messages[0])
            time.sleep(0.01)
        if not ack.is_file():
            raise TimeoutError(f"GT decoder history ACK not found: {ack}")
        print(f"[pick_tissue_sonic_latent] history seed ACK <- {ack}")

    frame_path = Path(config.gt_decoder_frame_path) if config.gt_decoder_frame_path else None
    if frame_path is not None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_text("0\n", encoding="utf-8")
        print(f"[pick_tissue_sonic_latent] gt_decoder_frame -> {frame_path}")

    period = 1.0 / config.fps
    for i, msg in enumerate(messages):
        t0 = time.monotonic()
        if frame_path is not None:
            tmp = frame_path.with_suffix(".tmp")
            tmp.write_text(f"{i + start}\n", encoding="utf-8")
            tmp.replace(frame_path)
        pub.send(msg)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == n:
            print(
                f"[pick_tissue_sonic_latent] frame {i + 1}/{n} "
                f"dataset_frame={i + start} "
                f"token[0]={tokens[i][0]:+.3f} R_hand[0]={right[i][0]:+.3f}"
            )
        elapsed = time.monotonic() - t0
        rem = period - elapsed
        if rem > 0:
            time.sleep(rem)

    print("[pick_tissue_sonic_latent] done")


if __name__ == "__main__":
    main(tyro.cli(ReplayConfig))
