#!/usr/bin/env python3
"""Publish 810short/pick-tissue GT dual camera (ego + left_wrist/chest) on ZMQ :5555.

Wire format matches ComposedCameraClientSensor: RGB ndarrays → ImageMessageSchema JPEG.
Keys: ``ego_view`` + ``left_wrist`` (disk chest-forward; train-aligned dual VLM).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.pick_tissue_unified import CHEST_FORWARD_DISK_KEY, EGO_IMAGE_KEY  # noqa: E402
from phi0.deploy.pick_tissue_gt import (  # noqa: E402
    PickTissueEpisodeSpan,
    control_index_to_global_frame,
    reader_from_data_cfg,
)
from phi0.deploy.pick_tissue_gt_images import PickTissuePredecodedReader  # noqa: E402
from phi0.paths import workspace_root  # noqa: E402


def _max_control_frames(span: PickTissueEpisodeSpan, *, native_fps: float, control_fps: float) -> int:
    if native_fps <= 0 or control_fps <= 0:
        return int(span.frame_count)
    return max(1, int(round(float(span.frame_count) * float(control_fps) / float(native_fps))))


def publish_gt_video(
    *,
    episode: int,
    root: str,
    repo_id: str,
    port: int,
    control_fps: float,
    start_control_idx: int,
    loop: bool,
    preload: bool,
) -> None:
    from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

    # Native HxW (810short ego+chest are both 480x640). Do not letterbox to 180x320.
    frame_reader = PickTissuePredecodedReader(
        root_dir=root,
        repo_id=repo_id,
        image_size=(480, 640),
        view_fit="letterbox_raw",
    )
    gt_reader = reader_from_data_cfg({"pick_tissue_root": root, "pick_tissue_repo_id": repo_id})
    span = gt_reader.episode_span(int(episode))
    native_fps = float(frame_reader.native_fps)
    chest_mp4 = frame_reader._mp4_path(int(episode), CHEST_FORWARD_DISK_KEY)
    publish_chest = chest_mp4.is_file()
    if preload:
        t0 = time.monotonic()
        frame_reader.preload_episode(span)
        print(
            f"[gt_camera] preloaded ep{episode} ({span.frame_count} native frames) "
            f"in {time.monotonic() - t0:.1f}s chest={'yes' if publish_chest else 'no'}"
        )

    max_ctrl = _max_control_frames(span, native_fps=native_fps, control_fps=control_fps)
    server = SensorServer()
    try:
        server.start_server(int(port))
    except Exception as e:
        # zmq.error.ZMQError: Address already in use — usually T1 still publishing sim ego
        msg = str(e).lower()
        if "address already in use" in msg or "address in use" in msg:
            raise SystemExit(
                f"[gt_camera] tcp://*:{port} busy (often T1 sim image_publish). "
                f"Restart T1 without --publish-dual-camera, or: "
                f"fuser -k {port}/tcp\n  original: {e}"
            ) from e
        raise
    time.sleep(0.3)
    keys = "ego_view+left_wrist" if publish_chest else "ego_view-only"
    print(
        f"[gt_camera] publishing ep{episode} repo={repo_id} tcp://*:{port} "
        f"keys={keys} control_fps={control_fps} start={start_control_idx} "
        f"max_ctrl={max_ctrl} loop={loop}"
    )

    period = 1.0 / float(control_fps)
    control_idx = int(start_control_idx)
    t_next = time.monotonic()
    sent = 0
    try:
        while True:
            rel = control_idx - int(start_control_idx)
            if rel >= max_ctrl:
                if not loop:
                    print(f"[gt_camera] reached end ({max_ctrl} control frames)")
                    break
                control_idx = int(start_control_idx)
                rel = 0

            global_frame = control_index_to_global_frame(
                span.frame_start,
                control_idx,
                native_fps=native_fps,
                control_fps=control_fps,
            )
            ego = frame_reader.read_camera_rgb(global_frame, span, key=EGO_IMAGE_KEY)
            ts = time.time()
            images = {"ego_view": np.asarray(ego, dtype=np.uint8)}
            timestamps = {"ego_view": ts}
            if publish_chest:
                chest = frame_reader.read_camera_rgb(
                    global_frame, span, key=CHEST_FORWARD_DISK_KEY
                )
                images["left_wrist"] = np.asarray(chest, dtype=np.uint8)
                timestamps["left_wrist"] = ts

            msg = ImageMessageSchema(timestamps=timestamps, images=images).serialize()
            server.send_message(msg)
            sent += 1
            if sent % int(max(1, control_fps)) == 0:
                shapes = {k: list(np.asarray(v).shape) for k, v in images.items()}
                print(f"[gt_camera] ctrl={control_idx} sent={sent} shapes={shapes}")

            control_idx += 1
            t_next += period
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                t_next = time.monotonic()
    except KeyboardInterrupt:
        print("[gt_camera] stopped")
    finally:
        frame_reader.close()
        server.stop_server()


def publish_ego_mp4(
    *,
    mp4_path: str,
    port: int,
    control_fps: float,
    loop: bool,
) -> None:
    """Publish a single ego mp4 over ZMQ (no LeRobot meta required)."""
    from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

    path = Path(mp4_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ego mp4 not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open mp4: {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = float(control_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_ctrl = max(1, int(round(frame_count * float(control_fps) / native_fps))) if frame_count > 0 else 1

    server = SensorServer()
    server.start_server(int(port))
    time.sleep(0.3)
    print(
        f"[gt_camera] publishing ego mp4={path} tcp://*:{port} "
        f"native_fps={native_fps:.2f} control_fps={control_fps} frames={frame_count} "
        f"max_ctrl={max_ctrl} loop={loop}"
    )

    period = 1.0 / float(control_fps)
    control_idx = 0
    t_next = time.monotonic()
    sent = 0
    try:
        while True:
            if control_idx >= max_ctrl:
                if not loop:
                    print(f"[gt_camera] reached end ({max_ctrl} control frames)")
                    break
                control_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_idx = int(round(control_idx * native_fps / float(control_fps)))
            if frame_count > 0:
                frame_idx = min(frame_idx, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                if not loop:
                    print(f"[gt_camera] read failed at frame {frame_idx}")
                    break
                control_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            ego = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = time.time()
            msg = ImageMessageSchema(
                timestamps={"ego_view": ts},
                images={"ego_view": np.asarray(ego, dtype=np.uint8)},
            ).serialize()
            server.send_message(msg)
            sent += 1
            if sent % int(max(1, control_fps)) == 0:
                print(f"[gt_camera] ctrl={control_idx} frame={frame_idx} sent={sent}")

            control_idx += 1
            t_next += period
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                t_next = time.monotonic()
    except KeyboardInterrupt:
        print("[gt_camera] stopped")
    finally:
        cap.release()
        server.stop_server()


def _self_check() -> None:
    assert control_index_to_global_frame(100, 0, native_fps=50, control_fps=50) == 100
    assert control_index_to_global_frame(100, 10, native_fps=50, control_fps=50) == 110
    max_ctrl = _max_control_frames(
        PickTissueEpisodeSpan(episode_index=0, frame_start=0, frame_count=500),
        native_fps=50,
        control_fps=50,
    )
    assert max_ctrl == 500
    # RGB roundtrip via ImageMessageSchema (encode expects RGB, client gets RGB).
    from gear_sonic.camera.sensor_server import ImageMessageSchema

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # R
    rgb[..., 2] = 10  # B
    packed = ImageMessageSchema(
        timestamps={"ego_view": 0.0},
        images={"ego_view": rgb},
    ).serialize()
    out = ImageMessageSchema.deserialize(packed).images["ego_view"]
    assert out.shape == (4, 4, 3)
    assert int(out[0, 0, 0]) > 100 and int(out[0, 0, 2]) < 50, out[0, 0]
    print("[gt_camera] self-check ok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=int, default=447)
    p.add_argument(
        "--ego-mp4",
        type=str,
        default="",
        help="Publish this ego mp4 directly (skips LeRobot episode lookup).",
    )
    p.add_argument("--repo-id", type=str, default="pick_tissue_xperience_unified")
    p.add_argument(
        "--root",
        type=str,
        default=f"{workspace_root()}/Isaac-GR00T/data",
    )
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--control-fps", type=float, default=50.0)
    p.add_argument("--start-control-idx", type=int, default=0)
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-preload", action="store_true")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
        return
    ego_mp4 = str(args.ego_mp4 or "").strip()
    if ego_mp4:
        publish_ego_mp4(
            mp4_path=ego_mp4,
            port=int(args.port),
            control_fps=float(args.control_fps),
            loop=bool(args.loop),
        )
        return
    publish_gt_video(
        episode=int(args.episode),
        root=str(args.root),
        repo_id=str(args.repo_id),
        port=int(args.port),
        control_fps=float(args.control_fps),
        start_control_idx=int(args.start_control_idx),
        loop=bool(args.loop),
        preload=not bool(args.no_preload),
    )


if __name__ == "__main__":
    main()
