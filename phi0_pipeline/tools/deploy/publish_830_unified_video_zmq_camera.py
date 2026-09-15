#!/usr/bin/env python3
"""Publish 830 unified LeRobot dual camera (ego + left_wrist) on ZMQ :5555.

New script for 3-terminal CL — does not modify publish_gt_video_zmq_camera.py.

Wire format matches ComposedCameraClientSensor / ImageMessageSchema JPEG.
Keys: ``ego_view`` + ``left_wrist`` (train-aligned dual VLM).

Usage:
  python tools/deploy/publish_830_unified_video_zmq_camera.py --episode 125
  python tools/deploy/publish_830_unified_video_zmq_camera.py --episode 125 --loop-pause-s 2
  python tools/deploy/publish_830_unified_video_zmq_camera.py --episode 125 --preview opencv
  python tools/deploy/publish_830_unified_video_zmq_camera.py --episode 125 --preview browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.lerobot_step_io import load_info, video_path  # noqa: E402

EGO_KEY = "observation.images.ego_view"
CHEST_KEY = "observation.images.left_wrist"
_PREVIEW_WINDOW = "830_walk_dual_replay"
# task_index order in 830mix_skill123_demo5 meta/tasks.parquet
_TASK_SKILLS = (
    "skill1",
    "skill2",
    "skill3",
    "idle",
    "egypt",
    "spin",
    "wave",
    "bow",
)


def _resolve_dual_mp4(ref_root: Path, episode: int) -> tuple[Path, Path]:
    info = load_info(ref_root)
    ego = video_path(ref_root, info, int(episode), video_key=EGO_KEY)
    chest = video_path(ref_root, info, int(episode), video_key=CHEST_KEY)
    if not ego.is_file():
        raise FileNotFoundError(f"missing ego mp4: {ego}")
    if not chest.is_file():
        raise FileNotFoundError(f"missing chest mp4: {chest}")
    return ego, chest


def _load_skill_episode_map(ref_root: Path) -> dict[str, int]:
    """First vision ``episode_index`` per skill from ``meta/episodes``."""
    import pyarrow.parquet as pq

    ep_dir = ref_root / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet")) if ep_dir.is_dir() else []
    if not files:
        return {}
    # has_video / task_index optional — pico_pure meta has tasks[] only (no task_index).
    schema_names = set(pq.read_schema(files[0]).names)
    if "task_index" not in schema_names or "episode_index" not in schema_names:
        return {}
    cols = ["episode_index", "task_index"]
    if "has_video" in schema_names:
        cols.append("has_video")
    table = pq.read_table(files[0], columns=cols)
    df = table.to_pandas()
    out: dict[str, int] = {}
    for task_i, name in enumerate(_TASK_SKILLS):
        rows = df[df["task_index"] == int(task_i)]
        if rows.empty:
            continue
        if "has_video" in rows.columns:
            vis = rows[rows["has_video"].astype(bool)]
            if vis.empty:
                continue
            rows = vis
        out[name] = int(rows["episode_index"].min())
    return out


def _open_caps(ego: Path, chest: Path):
    cap_ego = cv2.VideoCapture(str(ego))
    cap_chest = cv2.VideoCapture(str(chest))
    if not cap_ego.isOpened() or not cap_chest.isOpened():
        raise RuntimeError(f"failed to open mp4 ego={ego} chest={chest}")
    return cap_ego, cap_chest


def _clip_stats(cap_ego, cap_chest, control_fps: float) -> tuple[float, int, int]:
    native_fps = float(cap_ego.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = float(control_fps)
    frame_count = int(cap_ego.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    chest_frames = int(cap_chest.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if chest_frames > 0 and frame_count > 0:
        frame_count = min(frame_count, chest_frames)
    max_ctrl = (
        max(1, int(round(frame_count * float(control_fps) / native_fps)))
        if frame_count > 0
        else 1
    )
    return native_fps, frame_count, max_ctrl


def _stack_bgr_panels(bgr_e: np.ndarray, bgr_c: np.ndarray) -> np.ndarray:
    """Side-by-side preview: ego | chest, equal height."""
    h = max(bgr_e.shape[0], bgr_c.shape[0])

    def _resize_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / float(img.shape[0])
        w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    left = _resize_h(bgr_e)
    right = _resize_h(bgr_c)
    return np.hstack([left, right])


class _PreviewState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._label = ""

    def set(self, bgr: np.ndarray, *, label: str = "") -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._label = label

    def get(self) -> tuple[bytes | None, str]:
        with self._lock:
            return self._jpeg, self._label


def _start_browser_preview(state: _PreviewState, http_port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/"):
                jpeg, label = state.get()
                if jpeg is None:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"waiting for first frame")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    jpeg, label = state.get()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    hdr = (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    try:
                        self.wfile.write(hdr)
                        self.wfile.flush()
                    except BrokenPipeError:
                        break
                    time.sleep(1.0 / 30.0)

        def log_message(self, fmt: str, *args: object) -> None:
            del fmt, args

    srv = ThreadingHTTPServer(("127.0.0.1", int(http_port)), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(
        f"[830_camera] browser preview http://127.0.0.1:{http_port}/ "
        f"(ego|chest side-by-side)",
        flush=True,
    )
    return srv


def publish_830_dual_video(
    *,
    ref_root: str,
    episode: int,
    port: int,
    control_fps: float,
    start_control_idx: int,
    loop: bool,
    loop_pause_s: float,
    preview: str = "off",
    preview_http_port: int = 8088,
    skill_cmd_host: str = "127.0.0.1",
    skill_cmd_port: int = 0,
) -> None:
    from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

    root = Path(ref_root).expanduser().resolve()
    skill_ep = _load_skill_episode_map(root)
    episode = int(episode)
    ego_path, chest_path = _resolve_dual_mp4(root, episode)
    cap_ego, cap_chest = _open_caps(ego_path, chest_path)
    native_fps, frame_count, max_ctrl = _clip_stats(cap_ego, cap_chest, control_fps)

    server = SensorServer()
    try:
        server.start_server(int(port))
    except Exception as e:
        msg = str(e).lower()
        if "address already in use" in msg or "address in use" in msg:
            raise SystemExit(
                f"[830_camera] tcp://*:{port} busy. "
                f"Stop other publisher/sim image_publish, or: fuser -k {port}/tcp\n"
                f"  original: {e}"
            ) from e
        raise
    time.sleep(0.3)

    skill_pull = None
    if int(skill_cmd_port) > 0:
        import zmq

        skill_pull = zmq.Context.instance().socket(zmq.PULL)
        skill_pull.setsockopt(zmq.RCVHWM, 8)
        skill_pull.setsockopt(zmq.RCVTIMEO, 0)
        bind_ep = f"tcp://{skill_cmd_host}:{int(skill_cmd_port)}"
        try:
            skill_pull.bind(bind_ep)
        except zmq.ZMQError as exc:
            raise SystemExit(
                f"[830_camera] skill-cmd bind {bind_ep} failed: {exc} "
                f"(fuser -k {int(skill_cmd_port)}/tcp)"
            ) from exc
        print(
            f"[830_camera] skill-cmd PULL {bind_ep} map={skill_ep or '{}'} "
            f"(student PUSH → switch ep / reset t=0)",
            flush=True,
        )

    print(
        f"[830_camera] ep={episode} root={root}\n"
        f"  ego={ego_path.name} chest={chest_path.name}\n"
        f"  tcp://*:{port} native_fps={native_fps:.2f} control_fps={control_fps} "
        f"frames={frame_count} max_ctrl={max_ctrl} loop={loop} loop_pause_s={loop_pause_s} "
        f"preview={preview}",
        flush=True,
    )

    preview_mode = str(preview).strip().lower()
    preview_state = _PreviewState()
    browser_srv: ThreadingHTTPServer | None = None
    use_opencv = preview_mode in ("opencv", "window", "1", "true", "yes", "on")
    use_browser = preview_mode in ("browser", "http", "web")
    if use_opencv and not os.environ.get("DISPLAY"):
        print(
            "[830_camera] WARN: DISPLAY unset; opencv preview disabled, use --preview browser",
            flush=True,
        )
        use_opencv = False
    if use_browser:
        browser_srv = _start_browser_preview(preview_state, int(preview_http_port))
    elif use_opencv:
        print(
            f"[830_camera] opencv preview window '{_PREVIEW_WINDOW}' "
            f"(ego|left_wrist); press q in window to stop",
            flush=True,
        )

    period = 1.0 / float(control_fps)
    control_idx = int(start_control_idx)
    t_next = time.monotonic()
    sent = 0
    round_idx = 0
    last_frame_idx = -1
    cached_e = cached_c = None

    def _seek_zero() -> None:
        nonlocal control_idx, last_frame_idx, cached_e, cached_c, t_next
        control_idx = int(start_control_idx)
        cap_ego.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_chest.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last_frame_idx = -1
        cached_e = cached_c = None
        t_next = time.monotonic()

    def _reload_episode(ep: int, *, reason: str) -> None:
        nonlocal episode, ego_path, chest_path, cap_ego, cap_chest
        nonlocal native_fps, frame_count, max_ctrl
        ep = int(ep)
        new_ego, new_chest = _resolve_dual_mp4(root, ep)
        cap_ego.release()
        cap_chest.release()
        cap_ego, cap_chest = _open_caps(new_ego, new_chest)
        native_fps, frame_count, max_ctrl = _clip_stats(cap_ego, cap_chest, control_fps)
        ego_path, chest_path = new_ego, new_chest
        episode = ep
        _seek_zero()
        print(
            f"[830_camera] {reason} ep={episode} frames={frame_count} "
            f"ego={ego_path.name} chest={chest_path.name}",
            flush=True,
        )

    def _poll_skill_cmd() -> None:
        if skill_pull is None:
            return
        import zmq

        while True:
            try:
                raw_b = skill_pull.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            raw = raw_b.decode("utf-8", errors="replace").strip().lower()
            if not raw:
                continue
            # Vision skills with mapped ep → switch clip; else just reset t=0.
            if raw in skill_ep:
                want = int(skill_ep[raw])
                if want != int(episode):
                    try:
                        _reload_episode(want, reason=f"skill={raw}")
                    except FileNotFoundError as exc:
                        print(f"[830_camera] skill={raw} missing mp4: {exc}", flush=True)
                        _seek_zero()
                        print(f"[830_camera] reset t=0 (kept ep={episode})", flush=True)
                else:
                    _seek_zero()
                    print(f"[830_camera] skill={raw} reset t=0 ep={episode}", flush=True)
            elif raw in _TASK_SKILLS or raw in {"reset", "0"}:
                _seek_zero()
                print(f"[830_camera] skill={raw!r} reset t=0 ep={episode}", flush=True)
            else:
                print(f"[830_camera] ignore skill_cmd={raw!r}", flush=True)

    try:
        while True:
            _poll_skill_cmd()
            rel = control_idx - int(start_control_idx)
            if rel >= max_ctrl:
                if not loop:
                    print(f"[830_camera] reached end ({max_ctrl} control frames)", flush=True)
                    break
                round_idx += 1
                print(
                    f"[830_camera] round {round_idx} done; pause {loop_pause_s:.1f}s",
                    flush=True,
                )
                if use_opencv or use_browser:
                    pause_bgr = np.zeros((240, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        pause_bgr,
                        f"PAUSED {loop_pause_s:.0f}s  round {round_idx}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if use_opencv:
                        cv2.imshow(_PREVIEW_WINDOW, pause_bgr)
                        cv2.waitKey(1)
                    if use_browser:
                        preview_state.set(pause_bgr, label="paused")
                time.sleep(max(0.0, float(loop_pause_s)))
                control_idx = int(start_control_idx)
                cap_ego.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_chest.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame_idx = -1
                cached_e = cached_c = None
                t_next = time.monotonic()
                rel = 0

            frame_idx = int(round(rel * native_fps / float(control_fps)))
            if frame_count > 0:
                frame_idx = min(frame_idx, frame_count - 1)
            if frame_idx == last_frame_idx and cached_e is not None:
                bgr_e, bgr_c = cached_e, cached_c
                ok_e = ok_c = True
            else:
                if frame_idx != last_frame_idx + 1:
                    cap_ego.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    cap_chest.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok_e, bgr_e = cap_ego.read()
                ok_c, bgr_c = cap_chest.read()
                if ok_e and ok_c:
                    last_frame_idx = frame_idx
                    cached_e, cached_c = bgr_e, bgr_c
            if not ok_e or not ok_c:
                if not loop:
                    print(f"[830_camera] read failed at frame {frame_idx}", flush=True)
                    break
                round_idx += 1
                time.sleep(max(0.0, float(loop_pause_s)))
                control_idx = int(start_control_idx)
                cap_ego.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_chest.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame_idx = -1
                cached_e = cached_c = None
                t_next = time.monotonic()
                continue

            ego = cv2.cvtColor(bgr_e, cv2.COLOR_BGR2RGB)
            chest = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2RGB)
            ts = time.time()
            msg = ImageMessageSchema(
                timestamps={"ego_view": ts, "left_wrist": ts},
                images={
                    "ego_view": ego,
                    "left_wrist": chest,
                },
            ).serialize()
            server.send_message(msg)
            sent += 1

            if use_opencv or use_browser:
                panel = _stack_bgr_panels(bgr_e, bgr_c)
                label = f"ep{episode} ctrl={control_idx} frame={frame_idx} sent={sent}"
                cv2.putText(
                    panel,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if use_opencv:
                    cv2.imshow(_PREVIEW_WINDOW, panel)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[830_camera] preview quit (q)", flush=True)
                        break
                if use_browser:
                    preview_state.set(panel, label=label)

            if sent % int(max(1, control_fps)) == 0:
                print(
                    f"[830_camera] ctrl={control_idx} frame={frame_idx} sent={sent} "
                    f"ego={list(ego.shape)} chest={list(chest.shape)}",
                    flush=True,
                )

            control_idx += 1
            t_next += period
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                t_next = time.monotonic()
    except KeyboardInterrupt:
        print("[830_camera] stopped", flush=True)
    finally:
        if use_opencv:
            cv2.destroyAllWindows()
        if browser_srv is not None:
            browser_srv.shutdown()
        if skill_pull is not None:
            skill_pull.close(0)
        cap_ego.release()
        cap_chest.release()
        server.stop_server()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ref-root",
        type=str,
        default="/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_walk_to_black_box_new_unified",
    )
    p.add_argument("--episode", type=int, default=125)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--control-fps", type=float, default=50.0)
    p.add_argument("--start-control-idx", type=int, default=0)
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--loop-pause-s",
        type=float,
        default=2.0,
        help="Pause between loop rounds (default 2s).",
    )
    p.add_argument(
        "--preview",
        type=str,
        default="opencv",
        choices=("off", "opencv", "browser"),
        help="Local preview in this terminal process: opencv window or browser MJPEG.",
    )
    p.add_argument(
        "--preview-http-port",
        type=int,
        default=8088,
        help="HTTP port when --preview browser (default 8088).",
    )
    p.add_argument(
        "--skill-cmd-host",
        default=os.environ.get("CAMERA_SKILL_ZMQ_HOST", "127.0.0.1"),
    )
    p.add_argument(
        "--skill-cmd-port",
        type=int,
        default=int(os.environ.get("CAMERA_SKILL_ZMQ_PORT", "0") or 0),
        help="0=off. PULL-bind; student PUSH skill name → switch ep / reset t=0.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    publish_830_dual_video(
        ref_root=str(args.ref_root),
        episode=int(args.episode),
        port=int(args.port),
        control_fps=float(args.control_fps),
        start_control_idx=int(args.start_control_idx),
        loop=bool(args.loop),
        loop_pause_s=float(args.loop_pause_s),
        preview=str(args.preview),
        preview_http_port=int(args.preview_http_port),
        skill_cmd_host=str(args.skill_cmd_host),
        skill_cmd_port=int(args.skill_cmd_port),
    )


if __name__ == "__main__":
    main()
