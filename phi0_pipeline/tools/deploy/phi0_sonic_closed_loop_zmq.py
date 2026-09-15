#!/usr/bin/env python3
"""Closed-loop Phi-0 -> SONIC ZMQ v4: live or GT camera + robot or roll-forward proprio."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from typing import Callable, Protocol

import numpy as np
import torch
import zmq
from hydra import compose, initialize_config_dir

# tools/deploy/<this> → repo root (not tools/)
ROOT = Path(__file__).resolve().parents[2]
_GR00T = Path(
    os.environ.get(
        "GR00T_ROOT",
        str(Path.home() / "YZY" / "GR00T-WholeBodyControl"),
    )
).expanduser().resolve()
for p in (ROOT / "src", ROOT, _GR00T):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.utils.inference.vla_utils import (  # noqa: E402
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.zmq.v4_latent_replay import (  # noqa: E402
    hand_ramp_weights,
    pack_latent_action_message,
)

from phi0.checkpoint_utils import merge_saved_cfg  # noqa: E402
from phi0.deploy.pick_tissue_gt import (  # noqa: E402
    PickTissueEpisodeSpan,
    control_index_to_global_frame,
    reader_from_data_cfg,
)
from phi0.deploy.gt_io import (  # noqa: E402
    GtEpisodeProprioSource,
    PickTissueGtBackend,
    is_pick_tissue_unified_cfg,
)
from phi0.deploy.train_prompt import resolve_deploy_prompt  # noqa: E402
from phi0.deploy.deploy_keyboard import (  # noqa: E402
    DeployCommandState,
    DeployKeyboardListener,
    handle_vla_zmq_key,
    open_zmq_keyboard_subscriber,
    send_deploy_command,
    send_start_streamed,
)
from phi0.deploy.robot_proprio import (  # noqa: E402
    RobotProprioSource,
    log_frame0_proprio_raw,
    maybe_log_frame0_proprio_raw,
    normalize_proprio41,
    normalize_unified_proprio,
    pack_brainco_hand_cmd,
    proprio41_from_g1_debug,
    unified_from_g1_debug,
)
from phi0.deploy.sonic_zmq_io import (  # noqa: E402
    HAND_MODE_REVO2,
    hand_mode_from_unified_dataset,
    unified_action_denorm_to_zmq_arrays,
)
from phi0.schema.unified_action_schema import D_UNIFIED  # noqa: E402
from phi0.inference.closed_loop_sched import (
    chunk_index_when_result_arrives,
    inference_pipeline_idle,
    prefetch_trigger_steps,
    resolve_min_inference_interval,
    should_trigger_chunk_prefetch,
)
from phi0.inference.session import ActionInferenceSession, resolve_deploy_action_chunk_size
from phi0.inference.rtc import (
    resolve_rtc_deploy_cfg,
    rtc_play_horizon,
    shift_action_chunk_rtc,
    validate_rtc_params,
)
from phi0.models.vlm.preprocess import normalize_vlm_instruction
from phi0.runtime import (  # noqa: E402
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

logger = logging.getLogger(__name__)


class IoRateTracker:
    """Sliding-window Hz log for live ZMQ I/O (camera, g1_debug, zmq_out)."""

    def __init__(self, *, interval_s: float = 2.0) -> None:
        self.interval_s = max(0.1, float(interval_s))
        self._counts: dict[str, int] = {"camera": 0, "g1_debug": 0, "zmq_out": 0}
        self._totals: dict[str, int] = {"camera": 0, "g1_debug": 0, "zmq_out": 0}
        self._first_seen: set[str] = set()
        self._last_log = time.monotonic()

    def tick(self, channel: str, *, first_log: str | None = None) -> None:
        if channel not in self._counts:
            self._counts[channel] = 0
            self._totals[channel] = 0
        if channel not in self._first_seen:
            self._first_seen.add(channel)
            if first_log:
                logger.info(first_log)
        self._counts[channel] += 1
        self._totals[channel] += 1

    def poll_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < self.interval_s:
            return
        dt = now - self._last_log
        parts = []
        for ch in ("camera", "g1_debug", "zmq_out"):
            n = int(self._counts.get(ch, 0))
            if n > 0 or ch in self._first_seen:
                parts.append(f"{ch}={n / dt:.1f}Hz")
            self._counts[ch] = 0
        if parts:
            logger.info("io_rates [%s]", " ".join(parts))
        self._last_log = now

    def summary(self) -> dict[str, int]:
        return dict(self._totals)


def _io_rate_self_check() -> None:
    t = IoRateTracker(interval_s=0.05)
    t.tick("camera", first_log="io_self_check camera ok")
    t.tick("g1_debug")
    t.tick("zmq_out")
    t._last_log = time.monotonic() - 0.06
    t.poll_log()
    assert t.summary()["camera"] == 1


_EGO_KEYS = ("ego_view", "head", "observation.images.ego_view")
# Train dual: chest_forward on disk is left_wrist (see CHEST_FORWARD_DISK_KEY).
_CHEST_KEYS = (
    "chest_forward",
    "left_wrist",
    "observation.images.left_wrist",
    "observation.images.chest_forward",
)


@dataclass
class ActionChunk:
    tokens: np.ndarray
    left: np.ndarray
    right: np.ndarray
    horizon: int


@dataclass
class ObsSnapshot:
    control_idx: int
    ego_hwc: np.ndarray
    timestamp: float


@dataclass
class RtcInferState:
    """Normalized action chunk from the previous policy query (for deploy RTC blend)."""

    prev_chunk_norm: torch.Tensor | None = None


@dataclass
class PredictResult:
    chunk: ActionChunk
    obs: ObsSnapshot
    stage_s: dict[str, float] | None = None


def _cuda_sync(device: torch.device | str) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mono_elapsed(t0: float) -> float:
    return time.monotonic() - t0


class ClosedLoopRecorder:
    """Append-only trace; flushed to npz on save()."""

    def __init__(
        self,
        *,
        prompt: str,
        camera_source: str,
        control_fps: float,
        checkpoint: str,
    ):
        self._meta = {
            "prompt": prompt,
            "camera_source": camera_source,
            "control_fps": float(control_fps),
            "checkpoint": str(checkpoint),
        }
        self._obs_ego: list[np.ndarray] = []
        self._obs_control_idx: list[int] = []
        self._obs_timestamp: list[float] = []
        self._obs_inference_elapsed_s: list[float] = []
        self._obs_stage_timings: list[dict[str, float]] = []
        self._out_tokens: list[np.ndarray] = []
        self._out_left: list[np.ndarray] = []
        self._out_right: list[np.ndarray] = []
        self._out_control_idx: list[int] = []
        self._out_chunk_idx: list[int] = []
        self._out_frame_index: list[int] = []
        self._out_hand_ramp: list[float] = []
        self._out_timestamp: list[float] = []

    def record_observation(
        self,
        obs: ObsSnapshot,
        *,
        inference_elapsed_s: float | None = None,
        stage_s: dict[str, float] | None = None,
    ) -> None:
        self._obs_ego.append(np.asarray(obs.ego_hwc, dtype=np.uint8))
        self._obs_control_idx.append(int(obs.control_idx))
        self._obs_timestamp.append(float(obs.timestamp))
        if inference_elapsed_s is not None:
            self._obs_inference_elapsed_s.append(float(inference_elapsed_s))
        if stage_s is not None:
            self._obs_stage_timings.append({k: float(v) for k, v in stage_s.items()})

    def record_output(
        self,
        *,
        token: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        control_idx: int,
        chunk_idx: int,
        frame_index: int,
        hand_ramp: float,
        timestamp: float,
    ) -> None:
        self._out_tokens.append(np.asarray(token, dtype=np.float32).reshape(-1))
        self._out_left.append(np.asarray(left, dtype=np.float32).reshape(-1))
        self._out_right.append(np.asarray(right, dtype=np.float32).reshape(-1))
        self._out_control_idx.append(int(control_idx))
        self._out_chunk_idx.append(int(chunk_idx))
        self._out_frame_index.append(int(frame_index))
        self._out_hand_ramp.append(float(hand_ramp))
        self._out_timestamp.append(float(timestamp))

    def save(self, obs_path: Path, output_path: Path) -> None:
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = obs_path.parent / "record_meta.json"
        meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        logger.info("saved metadata %s", meta_path)
        if self._obs_stage_timings and len(self._obs_stage_timings) == len(self._obs_ego):
            stage_path = obs_path.parent / "inference_stage_timings.json"
            stage_path.write_text(
                json.dumps(
                    {
                        "control_idx": self._obs_control_idx,
                        "timestamp": self._obs_timestamp,
                        "inferences": [
                            {"i": i, "control_idx": int(self._obs_control_idx[i]), **t}
                            for i, t in enumerate(self._obs_stage_timings)
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("saved stage timings %s", stage_path)
        if self._obs_ego:
            obs_payload: dict = {
                "control_idx": np.asarray(self._obs_control_idx, dtype=np.int32),
                "timestamp": np.asarray(self._obs_timestamp, dtype=np.float64),
                "ego": np.stack(self._obs_ego, axis=0),
                "num_inferences": np.int32(len(self._obs_ego)),
            }
            if len(self._obs_inference_elapsed_s) == len(self._obs_ego):
                elapsed = np.asarray(self._obs_inference_elapsed_s, dtype=np.float64)
                obs_payload["inference_elapsed_s"] = elapsed
            np.savez(obs_path, **obs_payload)
            if "inference_elapsed_s" in obs_payload:
                e = obs_payload["inference_elapsed_s"]
                logger.info(
                    "saved observations %s (%d inferences, ego %s, infer %.3f/%.3f/%.3fs mean/p95/max)",
                    obs_path,
                    len(self._obs_ego),
                    obs_payload["ego"].shape,
                    float(np.mean(e)),
                    float(np.percentile(e, 95)),
                    float(np.max(e)),
                )
            else:
                logger.info(
                    "saved observations %s (%d inferences, ego %s)",
                    obs_path,
                    len(self._obs_ego),
                    obs_payload["ego"].shape,
                )
        else:
            logger.warning("no observations recorded; skip %s", obs_path)

        if self._out_tokens:
            np.savez(
                output_path,
                tokens=np.stack(self._out_tokens, axis=0).astype(np.float32),
                left=np.stack(self._out_left, axis=0).astype(np.float32),
                right=np.stack(self._out_right, axis=0).astype(np.float32),
                control_idx=np.asarray(self._out_control_idx, dtype=np.int32),
                chunk_idx=np.asarray(self._out_chunk_idx, dtype=np.int32),
                frame_index=np.asarray(self._out_frame_index, dtype=np.int32),
                hand_ramp=np.asarray(self._out_hand_ramp, dtype=np.float32),
                timestamp=np.asarray(self._out_timestamp, dtype=np.float64),
                num_frames=np.int32(len(self._out_tokens)),
            )
            logger.info(
                "saved outputs %s (%d frames x 64-d tokens)",
                output_path,
                len(self._out_tokens),
            )
        else:
            logger.warning("no outputs recorded; skip %s", output_path)


class CameraSource(Protocol):
    def read_ego_chw(self, control_idx: int) -> torch.Tensor: ...

    def close(self) -> None: ...


@dataclass
class LiveCameraSource:
    """Live ZMQ camera. Train-aligned dual: head/ego + left_wrist as chest_forward."""

    client: object
    io_tracker: IoRateTracker | None = None
    # ponytail: one msg → ego+chest; ceiling=stale if chest read without prior ego poll.
    _last_images: dict | None = None

    def _poll_images(self) -> dict:
        deadline = time.monotonic() + 2.0
        last_keys: list[str] = []
        while time.monotonic() < deadline:
            msg = self.client.read(blocking=False)
            if msg and msg.get("images"):
                images = msg["images"]
                last_keys = sorted(images.keys())
                self._last_images = images
                if self.io_tracker is not None:
                    ego = _pick_image(images, _EGO_KEYS, label="ego")
                    chest = None
                    try:
                        chest = _pick_image(images, _CHEST_KEYS, label="chest")
                    except KeyError:
                        pass
                    chest_s = (
                        f" chest={tuple(np.asarray(chest).shape)}" if chest is not None else " chest=missing"
                    )
                    self.io_tracker.tick(
                        "camera",
                        first_log=(
                            f"first camera frame keys={last_keys} "
                            f"ego={tuple(np.asarray(ego).shape)}{chest_s}"
                        ),
                    )
                return images
            time.sleep(0.01)
        hint = f" keys seen: {last_keys}" if last_keys else ""
        raise TimeoutError(f"no camera frame within 2s{hint}")

    def read_ego_chw(self, control_idx: int) -> torch.Tensor:
        del control_idx
        images = self._poll_images()
        return _rgb_to_chw(_pick_image(images, _EGO_KEYS, label="ego"))

    def read_chest_chw(self, control_idx: int) -> torch.Tensor:
        del control_idx
        # Same snapshot as the preceding read_ego_chw (predict always polls ego first).
        images = self._last_images if self._last_images is not None else self._poll_images()
        return _rgb_to_chw(_pick_image(images, _CHEST_KEYS, label="chest"))

    def close(self) -> None:
        self.client.close()


@dataclass
class GtCameraSource:
    reader: object
    span: PickTissueEpisodeSpan
    native_fps: float
    control_fps: float
    start_control_idx: int
    max_control_frames: int
    proprio_reader: object | None = None

    def _rel_global_frame(self, control_idx: int) -> int:
        rel = int(control_idx) - int(self.start_control_idx)
        rel = max(0, min(rel, self.max_control_frames - 1))
        return control_index_to_global_frame(
            self.span.frame_start,
            rel,
            native_fps=self.native_fps,
            control_fps=self.control_fps,
        )

    def read_ego_chw(self, control_idx: int) -> torch.Tensor:
        ego_rgb = self.reader.read_ego_rgb(self._rel_global_frame(control_idx), self.span)
        return _rgb_to_chw(ego_rgb)

    def read_chest_chw(self, control_idx: int) -> torch.Tensor:
        read_chest = getattr(self.reader, "read_chest_rgb", None)
        if not callable(read_chest):
            raise AttributeError("GT frame reader missing read_chest_rgb")
        chest_rgb = read_chest(self._rel_global_frame(control_idx), self.span)
        return _rgb_to_chw(chest_rgb)

    def close(self) -> None:
        close_fn = getattr(self.reader, "close", None)
        if callable(close_fn):
            close_fn()


def _summarize_camera_msg(msg: dict) -> str:
    images = msg.get("images") or {}
    parts = []
    for name, img in sorted(images.items()):
        arr = np.asarray(img)
        parts.append(f"{name}={list(arr.shape)}")
    ts = msg.get("timestamps") or {}
    ts_part = ", ".join(f"{k}={v:.3f}" for k, v in sorted(ts.items())[:4])
    return f"keys=[{', '.join(parts)}] ts=[{ts_part}]"


def _probe_live_camera(client, *, host: str, port: int, wait_s: float) -> dict:
    """Block until composed_camera on tcp://host:port returns at least one frame."""
    deadline = time.monotonic() + wait_s
    last_keys: list[str] = []
    while time.monotonic() < deadline:
        msg = client.read(blocking=False)
        if msg and msg.get("images"):
            summary = _summarize_camera_msg(msg)
            logger.info(
                "SONIC camera_server tcp://%s:%d ready %s",
                host,
                port,
                summary,
            )
            return msg
        if msg:
            last_keys = sorted((msg.get("images") or {}).keys())
        time.sleep(0.05)
    raise TimeoutError(
        f"no frames from SONIC composed_camera tcp://{host}:{port} within {wait_s:.0f}s"
        + (f" (keys seen: {last_keys})" if last_keys else "")
    )


def _preload_gt_episode(
    *,
    gt_root: str,
    gt_repo: str,
    episode: int,
) -> tuple[object, object, PickTissueEpisodeSpan]:
    """Load GT camera (mp4/npy) + parquet proprio once before the control loop."""
    from phi0.deploy.pick_tissue_gt import PickTissueGtReader, _cached_reader
    from phi0.deploy.pick_tissue_gt_images import PickTissuePredecodedReader, _cached_predecoded_reader

    gt_reader: PickTissueGtReader = _cached_reader(gt_root, gt_repo)
    # native H×W → VLM preprocess once (matches train dataloader). letterbox_raw
    # would Resize twice vs training and destroy alignment on 480×640 demos.
    frame_reader: PickTissuePredecodedReader = _cached_predecoded_reader(
        gt_root, gt_repo, "native"
    )
    span = gt_reader.episode_span(int(episode))
    t0 = time.monotonic()
    frame_reader.preload_episode(span)
    gt_reader.preload_span(span)
    logger.info(
        "preloaded GT ep%d (%d native frames) in %.2fs (camera + parquet in RAM)",
        int(episode),
        int(span.frame_count),
        time.monotonic() - t0,
    )
    return frame_reader, gt_reader, span


def _build_camera_source(
    args,
    *,
    data_cfg,
    io_tracker: IoRateTracker | None = None,
) -> CameraSource:
    source = str(args.camera_source).strip().lower()

    if source == "gt":
        ep = int(args.gt_camera_episode)
        gt_root = str(data_cfg.get("pick_tissue_root", "./data"))
        gt_repo = str(getattr(args, "gt_repo_id", None) or "pick_tissue_xperience_unified")
        frame_reader, gt_reader, span = _preload_gt_episode(
            gt_root=gt_root,
            gt_repo=gt_repo,
            episode=ep,
        )
        native_fps = float(gt_reader.native_fps)
        control_fps = float(args.control_fps)
        max_frames = int(span.frame_count)
        if float(args.motion_seconds) > 0:
            max_frames = min(
                max_frames,
                int(round(float(args.motion_seconds) * control_fps)),
            )
        src = GtCameraSource(
            reader=frame_reader,
            proprio_reader=gt_reader,
            span=span,
            native_fps=native_fps,
            control_fps=control_fps,
            start_control_idx=int(args.gt_camera_start_idx),
            max_control_frames=max(1, max_frames),
        )
        ego = src.read_ego_chw(int(args.gt_camera_start_idx))
        logger.info(
            "GT camera episode=%d frames=%d start_ctrl=%d ego=%s (preloaded)",
            ep,
            src.max_control_frames,
            int(args.gt_camera_start_idx),
            tuple(ego.shape),
        )
        return src

    host = args.camera_host.strip()
    port = int(args.camera_port)
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    logger.info("connecting SONIC composed_camera tcp://%s:%d", host, port)
    client = ComposedCameraClientSensor(server_ip=host, port=port)
    probe = _probe_live_camera(client, host=host, port=port, wait_s=float(args.camera_wait_s))
    images = probe.get("images") or {}
    missing_chest = not any(k in images for k in _CHEST_KEYS)
    if missing_chest:
        logger.warning(
            "live camera missing chest keys %s (have %s) — dual train-aligned VLM will fall back to ego-only",
            list(_CHEST_KEYS),
            sorted(images.keys()),
        )
    src = LiveCameraSource(client=client, io_tracker=io_tracker, _last_images=images)
    return src


def parse_args():
    p = argparse.ArgumentParser(
        description="Phi-0 closed-loop SONIC latent publisher (camera + roll-forward proprio)",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument(
        "--config-name",
        type=str,
        default="train_pick_yellow_box_xperience_unified_ddp4_16k",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument("--prompt", type=str, default="pick tissue")
    p.add_argument(
        "--camera-source",
        type=str,
        choices=("live", "gt"),
        default="live",
        help="live=SONIC composed_camera ZMQ :5555; gt=pick-tissue dataset episode.",
    )
    p.add_argument("--camera-host", type=str, default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--camera-wait-s", type=float, default=30.0)
    p.add_argument(
        "--gt-camera-episode",
        type=int,
        default=447,
        help=(
            "Dataset episode_index for GT camera and for train-aligned VLM prompt "
            "(live and gt). 810short launch scripts pass 0."
        ),
    )
    p.add_argument(
        "--gt-camera-start-idx",
        type=int,
        default=0,
        help="Control-frame offset into the GT episode (50 Hz timeline).",
    )
    p.add_argument(
        "--gt-repo-id",
        type=str,
        default="pick_tissue_xperience_unified",
        help="LeRobot repo for GT camera/proprio (default eval clip dataset).",
    )
    p.add_argument("--zmq-host", type=str, default="127.0.0.1")
    p.add_argument("--zmq-port", type=int, default=5556)
    p.add_argument("--state-zmq-host", type=str, default="127.0.0.1")
    p.add_argument("--state-zmq-port", type=int, default=5557)
    p.add_argument(
        "--hand-zmq-host",
        type=str,
        default="",
        help="Optional second SUB for robot hand proprio (0 port = disabled).",
    )
    p.add_argument("--hand-zmq-port", type=int, default=0)
    p.add_argument("--hand-zmq-topic", type=str, default="brainco_hand")
    p.add_argument(
        "--robot-zmq-host",
        type=str,
        default="",
        help="Optional SONIC token PUB mirror connect (unused if port=0).",
    )
    p.add_argument("--robot-zmq-port", type=int, default=0)
    p.add_argument(
        "--hand-cmd-zmq-host",
        type=str,
        default="",
        help="BrainCo hand command PUB connect host (robot binds :5570).",
    )
    p.add_argument("--hand-cmd-zmq-port", type=int, default=0)
    p.add_argument("--hand-cmd-zmq-topic", type=str, default="brainco_hand_cmd")
    p.add_argument("--control-fps", type=float, default=50.0)
    p.add_argument(
        "--scheduling",
        type=str,
        choices=("prefetch", "legacy"),
        default="prefetch",
        help=(
            "prefetch=trigger infer before chunk runs out (step-aligned index); "
            "legacy=time-based 2.5Hz gate + wall-clock latency index."
        ),
    )
    p.add_argument(
        "--inference-rate",
        type=float,
        default=0.0,
        help="Optional max re-infer rate (Hz). 0=no cap (prefetch mode default).",
    )
    p.add_argument(
        "--infer-latency-budget",
        type=float,
        default=0.30,
        help="Expected infer latency (s) for prefetch trigger margin @ control-fps.",
    )
    p.add_argument(
        "--infer-prefetch-safety-steps",
        type=int,
        default=2,
        help="Extra control steps added to prefetch trigger threshold.",
    )
    p.add_argument("--motion-seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    p.add_argument("--hand-ramp-frames", type=int, default=40)
    p.add_argument("--start-delay-s", type=float, default=0.5)
    p.add_argument("--arm-flag", type=str, default="", help="Wait for this file before arming deploy.")
    p.add_argument("--ready-flag", type=str, default="", help="Wait for this file before streaming tokens.")
    p.add_argument("--stream-flag", type=str, default="", help="Wait for this file before streamed-motion ZMQ arm.")
    p.add_argument("--arm-timeout-s", type=float, default=600.0)
    p.add_argument("--ready-timeout-s", type=float, default=900.0)
    p.add_argument(
        "--gt-proprio-episode",
        type=int,
        default=0,
        help=(
            "Fallback GT proprio episode_index when g1_debug is absent (live camera). "
            "Default 0: with --camera-source=gt, uses --gt-camera-episode automatically."
        ),
    )
    p.add_argument(
        "--proprio-source",
        type=str,
        choices=(
            "robot",
            "hybrid",
            "gt",
            "robot_gt_hand",
            "roll-forward",
            "bootstrap-roll-forward",
        ),
        default="robot",
        help=(
            "robot=g1_debug 5557 unified proprio; "
            "hybrid=SMPL semantic from last pred + robot tail; "
            "gt=dataset GT proprio @ control_idx (forward-align verify); "
            "robot_gt_hand=sim body29 from g1_debug + dataset revo2_12 @ control_idx; "
            "roll-forward=predicted proprio only; "
            "bootstrap-roll-forward=first g1_debug then roll-forward body from model tokens. "
            "When robot proprio is missing, dataset GT from the video episode is used if available."
        ),
    )
    p.add_argument(
        "--train-aligned-vlm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dual-view VLM + episode prompt refresh (810demo train-aligned forward).",
    )
    p.add_argument(
        "--no-episode-prompt",
        action="store_true",
        help="Use --prompt instead of dataset episode instruction.",
    )
    p.add_argument(
        "--seed-proprio",
        action="store_true",
        help="Use dataset-mean proprio before g1_debug arrives (robot/hybrid only). Default: off.",
    )
    p.add_argument(
        "--no-gt-fallback",
        action="store_true",
        help="Disable all dataset GT (no GT camera path, no GT proprio fallback). Live ZMQ only.",
    )
    p.add_argument(
        "--no-gt-proprio-fallback",
        action="store_true",
        help=(
            "Disable dataset GT proprio only (keep GT cameras). "
            "VLA proprio must come from g1_debug body29+revo2 (sim/robot)."
        ),
    )
    p.add_argument(
        "--io-log-interval-s",
        type=float,
        default=2.0,
        help="Log camera/g1_debug/zmq_out receive rates every N seconds (0=off).",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run IoRateTracker self-check and exit.",
    )
    p.add_argument(
        "--wait-deploy-p",
        action="store_true",
        help="Hold inference until keyboard p (ZMQ 5580 or stdin). VLA: p toggles pause; i then p streams.",
    )
    p.add_argument(
        "--wait-robot-proprio",
        action="store_true",
        help=(
            "Block until first g1_debug on state port before starting. "
            "Default: subscribe immediately and apply robot proprio once deploy control loop publishes (~50 Hz)."
        ),
    )
    p.add_argument(
        "--wait-deploy-state",
        action="store_true",
        help="Wait for first g1_debug frame on state port (roll-forward mode only).",
    )
    p.add_argument(
        "--stream-now",
        action="store_true",
        help="Send deploy start commands after the first action chunk is ready.",
    )
    p.add_argument(
        "--no-zmq",
        action="store_true",
        help="Do not bind/send ZMQ tokens or deploy commands; only save to npz.",
    )
    p.add_argument(
        "--deploy-keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read stdin keys for deploy ZMQ commands (legacy). Ignored with --no-zmq.",
    )
    p.add_argument(
        "--keyboard-zmq-port",
        type=int,
        default=5580,
        help="SUB to VLA keyboard publisher on this port (0=off). gear_sonic launch_inference uses 5580.",
    )
    p.add_argument(
        "--keyboard-zmq-host",
        type=str,
        default="127.0.0.1",
        help="Host for --keyboard-zmq-port SUB connect.",
    )
    p.add_argument(
        "--record-dir",
        type=str,
        default="",
        help="If set, save observations.npz (per inference) and outputs.npz (per tx frame).",
    )
    p.add_argument(
        "--record-obs",
        type=str,
        default="",
        help="Override observations npz path (default: RECORD_DIR/observations.npz).",
    )
    p.add_argument(
        "--record-output",
        type=str,
        default="",
        help="Override outputs npz path (default: RECORD_DIR/outputs.npz).",
    )
    p.add_argument(
        "--print-start-prefix",
        action="store_true",
        help="Log prefix tensor once on the first predict (eval debug).",
    )
    p.add_argument(
        "--print-start-prefix-full",
        action="store_true",
        help="With --print-start-prefix, log/save all raw_action_dim values (norm+denorm).",
    )
    p.add_argument(
        "--prefix-dump-path",
        type=str,
        default="",
        help="JSON path for full prefix dump (default: RECORD_DIR/start_prefix.json).",
    )
    p.add_argument(
        "--print-dataset-body-xyz",
        action="store_true",
        help="Log dataset unified_action[360:363] xyz at each inference control_idx.",
    )
    p.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deploy RTC hard frozen-prefix (default: follow model.rtc.enabled / USE_RTC).",
    )
    p.add_argument("--rtc-inference-delay", type=int, default=0, help="d (0=model cfg)")
    p.add_argument("--rtc-execution-horizon", type=int, default=0, help="s (0=model cfg)")
    p.add_argument(
        "--vlm-input",
        type=str,
        choices=("hwc", "clip"),
        default="hwc",
        help="VLM image path: hwc=uint8 PIL (fast); clip=GPU BCTHW tensor (legacy).",
    )
    p.add_argument(
        "--benchmark-vlm-input",
        action="store_true",
        help="On first infer, log preprocess timing for clip vs hwc paths.",
    )
    return p.parse_args()


def _resolve_rtc_cfg(cfg, args, *, horizon: int = 0) -> dict:
    import os

    rtc_flag = getattr(args, "rtc", None)
    if rtc_flag is None:
        env = os.environ.get("USE_RTC", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            rtc_flag = False
        elif env in ("1", "true", "yes", "on"):
            rtc_flag = True
    return resolve_rtc_deploy_cfg(
        cfg,
        rtc_flag=rtc_flag,
        inference_delay=int(getattr(args, "rtc_inference_delay", 0) or 0),
        execution_horizon=int(getattr(args, "rtc_execution_horizon", 0) or 0),
        schedule="hard",
        horizon=int(horizon),
    )


def _resolve_record_paths(args) -> tuple[Path | None, Path | None]:
    record_dir = args.record_dir.strip()
    obs_explicit = args.record_obs.strip()
    out_explicit = args.record_output.strip()
    if not record_dir and not obs_explicit and not out_explicit:
        return None, None
    if record_dir:
        base = Path(record_dir).expanduser().resolve()
        obs_path = (
            Path(obs_explicit).expanduser().resolve()
            if obs_explicit
            else base / "observations.npz"
        )
        out_path = (
            Path(out_explicit).expanduser().resolve()
            if out_explicit
            else base / "outputs.npz"
        )
    else:
        if not obs_explicit or not out_explicit:
            raise SystemExit(
                "--record-obs and --record-output required when --record-dir is omitted"
            )
        obs_path = Path(obs_explicit).expanduser().resolve()
        out_path = Path(out_explicit).expanduser().resolve()
    return obs_path, out_path


def _chw_to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().float().cpu().numpy()
    if x.ndim != 3:
        raise ValueError(f"expected CHW, got {x.shape}")
    if x.max() <= 1.0 + 1e-6:
        x = np.clip(x * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(x.transpose(1, 2, 0).astype(np.uint8))


def _pick_image(images: dict, keys: tuple[str, ...], *, label: str) -> np.ndarray:
    for key in keys:
        if key in images:
            return np.asarray(images[key], dtype=np.uint8)
    available = sorted(images.keys())
    if label == "ego" and available:
        logger.warning("%s missing; using %s", label, available[0])
        return np.asarray(images[available[0]], dtype=np.uint8)
    raise KeyError(f"{label} camera missing (have {available})")


def _rgb_to_chw(rgb: np.ndarray) -> torch.Tensor:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {arr.shape}")
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


def _frame_to_bcthw(frame_chw: torch.Tensor, *, device, dtype) -> torch.Tensor:
    clip = frame_chw.unsqueeze(0).unsqueeze(2)
    return (clip.to(device=device, dtype=dtype) * 2.0 - 1.0)


# ponytail: fails if dual live key order drifts from train CHEST_FORWARD_DISK_KEY.
def _selfcheck_dual_live_keys() -> None:
    images = {
        "head": np.zeros((2, 2, 3), np.uint8),
        "left_wrist": np.full((2, 2, 3), 7, np.uint8),
    }
    ego = _pick_image(images, _EGO_KEYS, label="ego")
    chest = _pick_image(images, _CHEST_KEYS, label="chest")
    assert ego.shape == (2, 2, 3) and int(chest[0, 0, 0]) == 7
    src = LiveCameraSource(client=object())
    src._last_images = images
    assert round(float(src.read_chest_chw(0)[0, 0, 0] * 255.0)) == 7


_selfcheck_dual_live_keys()


def _wait_for_deploy_state(host: str, port: int, timeout_s: float) -> None:
    import msgpack
    import msgpack_numpy as mnp

    from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC

    mnp.patch()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            msg = msgpack.unpackb(raw[len(STATE_ZMQ_TOPIC) :], raw=False)
            logger.info("deploy state ready keys=%s", sorted(msg.keys())[:12])
            return
    finally:
        sock.close(linger=0)
    raise TimeoutError(f"no g1_debug on tcp://{host}:{port} within {timeout_s:.0f}s")


def _send_deploy_start_commands(pub: zmq.Socket, send_lock: threading.Lock | None = None) -> None:
    send_start_streamed(pub, send_lock=send_lock)
    time.sleep(0.2)
    logger.info("sent ZMQ command start (planner -> streamed motion)")


def _send_deploy_arm_commands(pub: zmq.Socket, send_lock: threading.Lock | None = None) -> None:
    send_deploy_command(pub, start=True, stop=False, planner=True, send_lock=send_lock)
    time.sleep(0.2)
    logger.info("sent ZMQ command arm (planner start only)")


def _wait_flag(path: Path, *, label: str, timeout_s: float = 600.0) -> None:
    logger.info("waiting for %s %s", label, path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{label} flag not found: {path}")


def _load_model_bundle(args):
    ckpt_path = Path(args.checkpoint).resolve()
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    cfg.device = device

    logger.info("loading checkpoint %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ponytail: ckpt cfg wins in merge; keep yaml deploy recipe for hand/supervision/AdaLN
    # (810short gold ckpt still says g1_sonic_zmq — want g1_sonic_zmq_revo2 + hand_model=revo2).
    yaml_uds = str(cfg.data.get("unified_supervision_dataset") or "").strip()
    yaml_hand = cfg.data.get("hand_model")
    yaml_xattn = cfg.model.action_dit_config.get("action_cross_attn_mode")
    yaml_adaln = cfg.model.action_dit_config.get("adaln_mode")
    yaml_bins = cfg.model.action_dit_config.get("offset_bins")
    yaml_period = cfg.model.action_dit_config.get("vlm_age_period")
    yaml_exclude = cfg.model.get("action_loss_exclude_slices")
    if isinstance(payload, dict) and payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])
    if yaml_uds:
        cfg.data.unified_supervision_dataset = yaml_uds
    if yaml_hand not in (None, "", "null", "none"):
        cfg.data.hand_model = yaml_hand
    if yaml_xattn not in (None, "", "null", "none"):
        cfg.model.action_dit_config.action_cross_attn_mode = yaml_xattn
    if yaml_adaln not in (None, "", "null", "none"):
        cfg.model.action_dit_config.adaln_mode = yaml_adaln
    if yaml_bins is not None:
        cfg.model.action_dit_config.offset_bins = int(yaml_bins)
    if yaml_period is not None:
        cfg.model.action_dit_config.vlm_age_period = int(yaml_period)
    if yaml_exclude is not None:
        cfg.model.action_loss_exclude_slices = yaml_exclude
    logger.info(
        "deploy recipe after merge: unified_ds=%s hand_model=%s adaln=%s "
        "xattn=%s bins=%s period=%s exclude=%s",
        cfg.data.get("unified_supervision_dataset"),
        cfg.data.get("hand_model"),
        cfg.model.action_dit_config.get("adaln_mode"),
        cfg.model.action_dit_config.get("action_cross_attn_mode"),
        cfg.model.action_dit_config.get("offset_bins"),
        cfg.model.action_dit_config.get("vlm_age_period"),
        list(cfg.model.get("action_loss_exclude_slices") or []),
    )

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt_path))
    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()

    chunk_h = resolve_deploy_action_chunk_size(
        model, seq_len=int(cfg.data.get("seq_len", 33))
    )
    prompt = resolve_deploy_prompt(args, cfg.data)
    logger.info("deploy prompt: %r", prompt[:80])
    proprio_source = str(args.proprio_source).strip().lower()
    use_robot_proprio = proprio_source in {"robot", "hybrid", "robot_gt_hand"}
    bootstrap_roll_forward = proprio_source == "bootstrap-roll-forward"
    unified_ds = str(cfg.data.get("unified_supervision_dataset") or "g1_sonic").strip()
    train_vlm_view = str(cfg.data.get("train_vlm_view", "dual")).strip().lower()
    use_wrist_view = train_vlm_view in {"dual", "chest_forward"} or bool(
        cfg.data.get("use_wrist_view", getattr(processor, "use_wrist_view", False))
    )
    session = ActionInferenceSession(
        model,
        processor=processor,
        deploy_seq_len=int(cfg.data.get("seq_len", 33)),
        use_gt_proprio=use_robot_proprio,
        use_wrist_view=use_wrist_view,
        unified_supervision_dataset=unified_ds,
    )
    if bootstrap_roll_forward:
        logger.info(
            "proprio: bootstrap from g1_debug tcp://%s:%d then roll-forward from predicted body",
            args.state_zmq_host.strip(),
            int(args.state_zmq_port),
        )
    elif proprio_source == "roll-forward" or bool(getattr(args, "seed_proprio", False)):
        # proprio41 models need 41-d seed; processor.mean is action 512-d.
        expert = getattr(model, "action_expert", model)
        pdim = int(getattr(expert, "proprio_dim", 0) or 0)
        pmean = getattr(processor, "proprio_mean", None)
        if (
            pmean is not None
            and int(pmean.numel()) > 0
            and (pdim <= 0 or int(pmean.numel()) == pdim)
        ):
            seed = pmean.to(device=model.device, dtype=model.torch_dtype).reshape(-1)
            logger.info("proprio seeded from proprio_mean dim=%d", int(seed.numel()))
        elif pdim > 0:
            seed = torch.zeros(pdim, device=model.device, dtype=model.torch_dtype)
            logger.info("proprio seeded zeros dim=%d (no proprio_mean)", pdim)
        else:
            seed = processor.mean.to(device=model.device, dtype=model.torch_dtype).reshape(
                -1
            )
            logger.info("proprio seeded from action mean dim=%d", int(seed.numel()))
        session.seed_proprio_from_normalized(seed)
    elif proprio_source == "robot_gt_hand":
        logger.info(
            "proprio: sim body29 (g1_debug) + dataset GT revo2_12 @ control_idx"
        )
    elif use_robot_proprio:
        logger.info("proprio: no mean seed — inference waits for g1_debug")
    elif proprio_source == "gt":
        logger.info("proprio: dataset GT @ control_idx (train-aligned forward verify)")
    if bool(getattr(args, "print_start_prefix", False)):
        dump_path = str(getattr(args, "prefix_dump_path", "") or "").strip()
        if not dump_path and str(getattr(args, "record_dir", "") or "").strip():
            dump_path = str(Path(args.record_dir) / "start_prefix.json")
        session.configure_start_prefix_log(
            enabled=True,
            full_vector=bool(getattr(args, "print_start_prefix_full", False)),
            dump_path=dump_path or None,
        )
        logger.info(
            "will log eval start prefix on first predict (full=%s dump=%s)",
            bool(getattr(args, "print_start_prefix_full", False)),
            dump_path or "log only",
        )
    if bool(getattr(args, "print_dataset_body_xyz", False)):
        logger.info("will log dataset body_xyz[360:363] at each inference")
    from phi0.inference.train_aligned_forward import resolve_vlm_video_delta_indices
    from omegaconf import OmegaConf

    raw_deltas = cfg.data.get("vlm_video_delta_indices")
    if OmegaConf.is_config(raw_deltas):
        raw_deltas = OmegaConf.to_container(raw_deltas, resolve=True)
    vlm_video_delta_indices = resolve_vlm_video_delta_indices(raw_deltas)
    logger.info(
        "model ready chunk_h=%d past_w=%d use_wrist=%s prompt=%r proprio=%s unified_ds=%s "
        "deploy_align=%s train_aligned_vlm=%s vlm_video_delta_indices=%s",
        chunk_h,
        int(getattr(model, "past_action_window_size", 1)),
        use_wrist_view,
        prompt,
        proprio_source,
        unified_ds,
        bool(getattr(model, "deploy_align_proprio_prefix", False)),
        bool(getattr(args, "train_aligned_vlm", True)),
        vlm_video_delta_indices,
    )
    rtc_cfg = _resolve_rtc_cfg(cfg, args, horizon=int(chunk_h))
    if rtc_cfg.get("enabled"):
        validate_rtc_params(
            chunk_h,
            int(rtc_cfg["inference_delay"]),
            int(rtc_cfg["execution_horizon"]),
        )
        logger.info(
            "RTC deploy enabled d=%d s=%d schedule=%s",
            int(rtc_cfg["inference_delay"]),
            int(rtc_cfg["execution_horizon"]),
            rtc_cfg["schedule"],
        )
    return (
        model,
        processor,
        session,
        prompt,
        chunk_h,
        cfg,
        proprio_source,
        rtc_cfg,
        vlm_video_delta_indices,
    )




def _build_gt_proprio_source(
    args,
    data_cfg,
    camera: CameraSource,
) -> GtEpisodeProprioSource | None:
    if bool(getattr(args, "no_gt_fallback", False)):
        return None
    if bool(getattr(args, "no_gt_proprio_fallback", False)):
        return None
    if not is_pick_tissue_unified_cfg(data_cfg):
        return None
    control_fps = float(args.control_fps)
    if isinstance(camera, GtCameraSource):
        proprio_reader = camera.proprio_reader or camera.reader
        return GtEpisodeProprioSource(
            PickTissueGtBackend(
                reader=proprio_reader,
                span=camera.span,
                native_fps=float(camera.native_fps),
                control_fps=float(camera.control_fps),
            )
        )

    ep = int(getattr(args, "gt_proprio_episode", 0) or 0)
    if ep <= 0:
        return None
    reader = reader_from_data_cfg(data_cfg)
    span = reader.episode_span(ep)
    return GtEpisodeProprioSource(
        PickTissueGtBackend(
            reader=reader,
            span=span,
            native_fps=float(reader.native_fps),
            control_fps=control_fps,
        )
    )


def _apply_robot_proprio(
    *,
    session: ActionInferenceSession,
    processor,
    robot_proprio: RobotProprioSource | None,
    proprio_source: str,
) -> bool:
    del proprio_source
    if robot_proprio is None:
        return False
    robot_proprio.poll()
    if not robot_proprio.ready:
        return False
    norm = robot_proprio.build_normalized(processor)
    w = int(getattr(session.model, "past_action_window_size", 1) or 1)
    steps = norm.reshape(1, -1).expand(w, -1)
    session.set_proprio_gt(steps)
    return True


def _apply_proprio(
    *,
    session: ActionInferenceSession,
    processor,
    control_idx: int,
    robot_proprio: RobotProprioSource | None,
    gt_proprio: GtEpisodeProprioSource | None,
    proprio_source: str,
    gt_fallback_logged: list[bool] | None = None,
    frame0_prefix_logged: list[bool] | None = None,
) -> bool:
    if proprio_source == "robot_gt_hand" and robot_proprio is not None and gt_proprio is not None:
        robot_proprio.poll()
        if robot_proprio.ready and robot_proprio.last_msg is not None:
            maybe_log_frame0_proprio_raw(
                robot_proprio, frame0_prefix_logged, frame=0, log=logger
            )
            revo = gt_proprio.revo2_raw_at(int(control_idx))
            p41 = proprio41_from_g1_debug(robot_proprio.last_msg, revo2_12=revo)
            norm = normalize_proprio41(processor, p41)
            w = int(getattr(session.model, "past_action_window_size", 1) or 1)
            steps = norm.reshape(1, -1).expand(w, -1)
            session.set_proprio_gt(steps)
            return True
        # before first g1_debug: full dataset proprio (includes revo2)
        if gt_fallback_logged is not None and not gt_fallback_logged[0]:
            gt_fallback_logged[0] = True
            logger.warning(
                "robot_gt_hand: g1_debug not ready; temporary full GT proprio @ ctrl=%d",
                int(control_idx),
            )
        gt_proprio.apply_to_session(session, processor, int(control_idx))
        return True
    if proprio_source in {"robot", "hybrid"} and robot_proprio is not None:
        robot_proprio.poll()
        if robot_proprio.ready:
            maybe_log_frame0_proprio_raw(
                robot_proprio, frame0_prefix_logged, frame=0, log=logger
            )
            norm = robot_proprio.build_normalized(processor)
            w = int(getattr(session.model, "past_action_window_size", 1) or 1)
            steps = norm.reshape(1, -1).expand(w, -1)
            session.set_proprio_gt(steps)
            return True
    if gt_proprio is not None:
        if (
            proprio_source in {"robot", "hybrid", "robot_gt_hand"}
            and robot_proprio is not None
            and gt_fallback_logged is not None
            and not gt_fallback_logged[0]
        ):
            gt_fallback_logged[0] = True
            logger.warning(
                "robot proprio unavailable; using dataset GT proprio at control_idx=%d "
                "(pass --no-gt-fallback to disable)",
                int(control_idx),
            )
        gt_proprio.apply_to_session(session, processor, int(control_idx))
        return True
    return False


def _predict_chunk(
    *,
    session: ActionInferenceSession,
    model,
    processor,
    camera: CameraSource,
    get_control_idx: Callable[[], int],
    prompt: str,
    chunk_h: int,
    rtc_cfg: dict | None = None,
    rtc_state: RtcInferState | None = None,
    steps_since_query: int = 0,
    robot_proprio: RobotProprioSource | None = None,
    gt_proprio: GtEpisodeProprioSource | None = None,
    proprio_source: str = "roll-forward",
    gt_fallback_logged: list[bool] | None = None,
    print_dataset_body_xyz: bool = False,
    frame0_prefix_logged: list[bool] | None = None,
    vlm_input: str = "hwc",
    benchmark_vlm_input: bool = False,
    benchmark_done: list[bool] | None = None,
    train_aligned_vlm: bool = False,
    vlm_video_delta_indices: list[int] | None = None,
    hand_mode: str = "dex3",
) -> PredictResult:
    device = model.device
    stage: dict[str, float] = {}
    t_total = time.monotonic()

    t0 = time.monotonic()
    control_idx = int(get_control_idx())
    from phi0.models.adaln_exec import needs_exec_clocks, vlm_refresh_anchor

    vlm_refresh_period = session.vlm_refresh_period()
    vlm_needs_refresh = session.should_refresh_vlm(control_idx)
    # ponytail: period-gate uses refresh anchor; every-step (train no-align) uses current ctrl
    use_refresh_anchor = needs_exec_clocks(model) and not session.vlm_refresh_every_step()
    vlm_image_ctrl = (
        int(vlm_refresh_anchor(control_idx, period=vlm_refresh_period))
        if use_refresh_anchor
        else control_idx
    )
    if print_dataset_body_xyz and gt_proprio is not None:
        xyz = gt_proprio.dataset_body_xyz(control_idx)
        logger.info(
            "ctrl=%d dataset body_xyz[360:363]=[% .4f, % .4f, % .4f]",
            control_idx,
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
    deltas = list(vlm_video_delta_indices) if vlm_video_delta_indices else None
    use_hist = deltas is not None and len(deltas) > 1
    read_chest = getattr(camera, "read_chest_chw", None)
    ego_chw = camera.read_ego_chw(vlm_image_ctrl if vlm_needs_refresh else control_idx)
    ego_hwc: np.ndarray | list[np.ndarray] = _chw_to_hwc_uint8(ego_chw)
    chest_hwc: np.ndarray | list[np.ndarray] | None = None
    if callable(read_chest):
        try:
            if use_hist:
                ego_frames: list[np.ndarray] = []
                chest_frames: list[np.ndarray] = []
                for d in deltas:
                    c = int(control_idx) + int(d)
                    ego_frames.append(_chw_to_hwc_uint8(camera.read_ego_chw(c)))
                    chest_frames.append(_chw_to_hwc_uint8(read_chest(c)))
                ego_hwc = ego_frames
                chest_hwc = chest_frames
            else:
                chest_ctrl = vlm_image_ctrl if vlm_needs_refresh else control_idx
                chest_hwc = _chw_to_hwc_uint8(read_chest(chest_ctrl))
        except (FileNotFoundError, KeyError, AttributeError, OSError):
            chest_hwc = None
            if use_hist:
                ego_hwc = _chw_to_hwc_uint8(ego_chw)
    stage["camera"] = _mono_elapsed(t0)

    t0 = time.monotonic()
    ego_clip = _frame_to_bcthw(ego_chw, device=device, dtype=model.torch_dtype)
    _cuda_sync(device)
    stage["tensor"] = _mono_elapsed(t0)

    t0 = time.monotonic()
    ego_hwc_cur = ego_hwc[-1] if isinstance(ego_hwc, list) else ego_hwc
    if (
        benchmark_vlm_input
        and benchmark_done is not None
        and not benchmark_done[0]
        and session.model.uses_vlm_tower()
    ):
        bench = session.benchmark_vlm_preprocess(
            video=ego_clip,
            ego_hwc=ego_hwc_cur,
            instruction=prompt,
        )
        clip_ms = bench["clip_preprocess_s"] * 1000.0
        hwc_ms = bench["hwc_preprocess_s"] * 1000.0
        speedup = clip_ms / hwc_ms if hwc_ms > 0 else float("inf")
        logger.info(
            "vlm preprocess benchmark: clip=%.1fms hwc=%.1fms speedup=%.2fx",
            clip_ms,
            hwc_ms,
            speedup,
        )
        benchmark_done[0] = True

    vlm_path = str(vlm_input).strip().lower()
    use_dual_train_aligned = bool(train_aligned_vlm) and chest_hwc is not None
    hist_deltas = deltas if (use_hist and isinstance(ego_hwc, list)) else None
    if use_dual_train_aligned and proprio_source == "gt" and gt_proprio is not None:
        from phi0.inference.train_aligned_forward import refresh_train_aligned_context

        refresh_train_aligned_context(
            session,
            model,
            processor,
            ego_hwc=ego_hwc,
            chest_hwc=chest_hwc,
            instruction=prompt,
            gt_backend=gt_proprio.backend,
            control_idx=int(control_idx),
            vlm_video_delta_indices=hist_deltas,
        )
        stage["vlm_mode"] = 1.0 if session.action_ctx is not None else 0.0
        vlm_path = (
            "dual_train_aligned_gt_hist" if hist_deltas is not None else "dual_train_aligned_gt"
        )
    elif use_dual_train_aligned:
        from phi0.inference.train_aligned_forward import normalize_instruction

        if session.action_ctx is None:
            session.prefill_from_dual_hwc(
                ego_hwc,
                chest_hwc,
                normalize_instruction(prompt),
                vlm_video_delta_indices=hist_deltas,
            )
            stage["vlm_mode"] = 0.0
        elif vlm_needs_refresh:
            session.refresh_video_context_from_dual_hwc(
                ego_hwc,
                chest_hwc,
                prompt=normalize_instruction(prompt),
                vlm_video_delta_indices=hist_deltas,
            )
            stage["vlm_mode"] = 1.0
        else:
            stage["vlm_mode"] = 1.0
        vlm_path = (
            "dual_train_aligned_hist" if hist_deltas is not None else "dual_train_aligned"
        )
    elif vlm_path == "clip":
        if session.action_ctx is None:
            session.prefill_from_video_clip(ego_clip, prompt)
            stage["vlm_mode"] = 0.0  # ponytail: 0=prefill, 1=refresh (stored as float for json)
        elif vlm_needs_refresh:
            session.refresh_video_context_from_clip(ego_clip, prompt=prompt)
            stage["vlm_mode"] = 1.0
        else:
            stage["vlm_mode"] = 1.0
    else:
        if session.action_ctx is None:
            session.prefill_from_hwc(ego_hwc_cur, prompt)
            stage["vlm_mode"] = 0.0
        elif vlm_needs_refresh:
            session.refresh_video_context_from_hwc(ego_hwc_cur, prompt=prompt)
            stage["vlm_mode"] = 1.0
        else:
            stage["vlm_mode"] = 1.0
    _cuda_sync(device)
    stage["vlm"] = _mono_elapsed(t0)
    st = session.last_stage_timing
    stage["vlm_pre"] = st.vlm_preprocess_s
    stage["vlm_fwd"] = st.vlm_forward_s
    logger.info(
        "infer stages ms: vlm_pre=%.1f vlm_fwd=%.1f (path=%s)",
        st.vlm_preprocess_s * 1000.0,
        st.vlm_forward_s * 1000.0,
        vlm_path,
    )

    t0 = time.monotonic()
    use_dual_train_aligned = bool(train_aligned_vlm) and chest_hwc is not None
    if not (use_dual_train_aligned and proprio_source == "gt" and gt_proprio is not None):
        _apply_proprio(
            session=session,
            processor=processor,
            control_idx=control_idx,
            robot_proprio=robot_proprio,
            gt_proprio=gt_proprio,
            proprio_source=proprio_source,
            gt_fallback_logged=gt_fallback_logged,
            frame0_prefix_logged=frame0_prefix_logged,
        )
    stage["proprio"] = _mono_elapsed(t0)

    session.set_exec_step(int(control_idx))

    use_amp = device.type == "cuda"
    rtc_on = bool(rtc_cfg and rtc_cfg.get("enabled"))
    t0 = time.monotonic()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        if rtc_on and rtc_state is not None and rtc_state.prev_chunk_norm is not None:
            prev = rtc_state.prev_chunk_norm
            if int(steps_since_query) > 0:
                prev = shift_action_chunk_rtc(prev, int(steps_since_query))
            pred_norm = session.predict_rtc(
                int(chunk_h),
                prev,
                inference_delay=int(rtc_cfg["inference_delay"]),
                execution_horizon=int(rtc_cfg["execution_horizon"]),
                schedule=str(rtc_cfg["schedule"]),
                denormalize=False,
            )
        else:
            pred_norm = session.predict(int(chunk_h), denormalize=False)
    _cuda_sync(device)
    stage["act"] = _mono_elapsed(t0)
    session.last_stage_timing.act_predict_s = stage["act"]
    logger.info(
        "infer stages ms: act=%.1f (path=%s)",
        stage["act"] * 1000.0,
        vlm_path,
    )

    t0 = time.monotonic()
    if rtc_state is not None:
        rtc_state.prev_chunk_norm = pred_norm.detach().to(dtype=torch.float32)
    if pred_norm.ndim == 3:
        pred = processor.postprocess(pred_norm)
    elif pred_norm.ndim == 2:
        pred = processor.postprocess(pred_norm.unsqueeze(0)).squeeze(0)
    else:
        raise ValueError(f"unexpected pred shape {tuple(pred_norm.shape)}")
    action_denorm = pred.float().detach().cpu().numpy()
    if robot_proprio is not None and proprio_source == "hybrid":
        if action_denorm.ndim == 2:
            robot_proprio.set_semantic_base(action_denorm[-1])
        else:
            robot_proprio.set_semantic_base(action_denorm.reshape(-1)[:D_UNIFIED])
    tokens, left, right = unified_action_denorm_to_zmq_arrays(
        action_denorm, hand_mode=hand_mode
    )
    stage["post"] = _mono_elapsed(t0)
    stage["total"] = _mono_elapsed(t_total)

    chunk = ActionChunk(
        tokens=tokens,
        left=left,
        right=right,
        horizon=int(tokens.shape[0]),
    )
    obs = ObsSnapshot(
        control_idx=control_idx,
        ego_hwc=ego_hwc_cur,
        timestamp=time.monotonic(),
    )
    return PredictResult(chunk=chunk, obs=obs, stage_s=stage)


def _inference_worker(
    *,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    session: ActionInferenceSession,
    model,
    processor,
    camera: CameraSource,
    get_control_idx: Callable[[], int],
    prompt: str,
    chunk_h: int,
    rtc_cfg: dict | None = None,
    rtc_state: RtcInferState | None = None,
    steps_since_query: Callable[[], int] | None = None,
    recorder: ClosedLoopRecorder | None = None,
    robot_proprio: RobotProprioSource | None = None,
    gt_proprio: GtEpisodeProprioSource | None = None,
    proprio_source: str = "roll-forward",
    robot_proprio_ready_logged: list[bool] | None = None,
    gt_fallback_logged: list[bool] | None = None,
    defer_inference_without_robot: bool = False,
    defer_inference_logged: list[bool] | None = None,
    frame0_prefix_logged: list[bool] | None = None,
    vlm_input: str = "hwc",
    benchmark_vlm_input: bool = False,
    benchmark_done: list[bool] | None = None,
    print_dataset_body_xyz: bool = False,
    train_aligned_vlm: bool = False,
    vlm_video_delta_indices: list[int] | None = None,
    hand_mode: str = "dex3",
) -> None:
    while not stop_event.is_set():
        try:
            inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        trigger_steps = int(steps_since_query()) if steps_since_query is not None else 0
        busy_event.set()
        try:
            if defer_inference_without_robot and proprio_source in {
                "robot",
                "hybrid",
                "robot_gt_hand",
            }:
                has_proprio = False
                if robot_proprio is not None:
                    robot_proprio.poll()
                    has_proprio = robot_proprio.ready
                if not has_proprio and gt_proprio is None:
                    if defer_inference_logged is not None and not defer_inference_logged[0]:
                        defer_inference_logged[0] = True
                        logger.info(
                            "inference deferred until first g1_debug (seeded proprio disabled)"
                        )
                    continue
            t0 = time.monotonic()
            steps = int(steps_since_query()) if steps_since_query is not None else 0
            result = _predict_chunk(
                session=session,
                model=model,
                processor=processor,
                camera=camera,
                get_control_idx=get_control_idx,
                prompt=prompt,
                chunk_h=chunk_h,
                rtc_cfg=rtc_cfg,
                rtc_state=rtc_state,
                steps_since_query=steps,
                robot_proprio=robot_proprio,
                gt_proprio=gt_proprio,
                proprio_source=proprio_source,
                gt_fallback_logged=gt_fallback_logged,
                frame0_prefix_logged=frame0_prefix_logged,
                vlm_input=vlm_input,
                benchmark_vlm_input=benchmark_vlm_input,
                benchmark_done=benchmark_done,
                print_dataset_body_xyz=print_dataset_body_xyz,
                train_aligned_vlm=train_aligned_vlm,
                vlm_video_delta_indices=vlm_video_delta_indices,
                hand_mode=hand_mode,
            )
            if (
                robot_proprio is not None
                and robot_proprio.ready
                and robot_proprio_ready_logged is not None
                and not robot_proprio_ready_logged[0]
            ):
                robot_proprio_ready_logged[0] = True
                logger.info(
                    "robot proprio active (first g1_debug applied; deploy control loop running)"
                )
            chunk = result.chunk
            delay = time.monotonic() - t0
            stage = result.stage_s or {}
            if recorder is not None:
                recorder.record_observation(
                    result.obs,
                    inference_elapsed_s=delay,
                    stage_s=stage,
                )
            stage_ms = {k: stage[k] * 1000.0 for k in ("camera", "tensor", "vlm", "proprio", "act", "post", "total") if k in stage}
            vlm_tag = "prefill" if stage.get("vlm_mode", 1.0) < 0.5 else "refresh"
            logger.info(
                "inference done horizon=%d token0=%+.3f wall=%.3fs ctrl=%d | "
                "stage_ms camera=%.0f tensor=%.0f vlm(%s)=%.0f proprio=%.0f act=%.0f post=%.0f total=%.0f",
                chunk.horizon,
                float(chunk.tokens[0, 0]),
                delay,
                result.obs.control_idx,
                stage_ms.get("camera", 0.0),
                stage_ms.get("tensor", 0.0),
                vlm_tag,
                stage_ms.get("vlm", 0.0),
                stage_ms.get("proprio", 0.0),
                stage_ms.get("act", 0.0),
                stage_ms.get("post", 0.0),
                stage_ms.get("total", delay * 1000.0),
            )
            try:
                result_queue.put_nowait((chunk, t0, trigger_steps))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait((chunk, t0, trigger_steps))
        except Exception:
            logger.exception("inference worker failed")
        finally:
            busy_event.clear()


def _pipeline_idle(
    *,
    busy_event: threading.Event,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
) -> bool:
    return inference_pipeline_idle(
        worker_busy=busy_event.is_set(),
        inference_queue_pending=not inference_queue.empty(),
        result_queue_pending=not result_queue.empty(),
    )


def _try_enqueue_inference(
    *,
    inference_queue: queue.Queue,
    last_trigger_time: dict[str, float],
) -> bool:
    try:
        last_trigger_time["t"] = time.monotonic()
        inference_queue.put_nowait(None)
        return True
    except queue.Full:
        return False


def _sleep_remaining(t_start: float, period: float) -> None:
    rem = period - (time.monotonic() - t_start)
    if rem > 0:
        time.sleep(rem)


def run_closed_loop(args) -> None:
    os.chdir(ROOT)
    (
        model,
        processor,
        session,
        prompt,
        chunk_h,
        cfg,
        proprio_source,
        rtc_cfg,
        vlm_video_delta_indices,
    ) = _load_model_bundle(args)
    obs_path, output_path = _resolve_record_paths(args)
    recorder: ClosedLoopRecorder | None = None
    if obs_path is not None and output_path is not None:
        recorder = ClosedLoopRecorder(
            prompt=prompt,
            camera_source=str(args.camera_source),
            control_fps=float(args.control_fps),
            checkpoint=str(args.checkpoint),
        )
        logger.info("recording obs=%s output=%s", obs_path, output_path)

    io_tracker: IoRateTracker | None = None
    io_interval = float(getattr(args, "io_log_interval_s", 2.0))
    if io_interval > 0:
        io_tracker = IoRateTracker(interval_s=io_interval)

    camera = _build_camera_source(args, data_cfg=cfg.data, io_tracker=io_tracker)
    gt_proprio = _build_gt_proprio_source(args, cfg.data, camera)
    if bool(getattr(args, "no_gt_fallback", False)):
        logger.info(
            "GT disabled: camera=tcp://%s:%d proprio=g1_debug tcp://%s:%d only",
            args.camera_host.strip(),
            int(args.camera_port),
            args.state_zmq_host.strip(),
            int(args.state_zmq_port),
        )
    elif bool(getattr(args, "no_gt_proprio_fallback", False)):
        logger.info(
            "GT proprio disabled: VLA proprio = g1_debug body29+revo2_12 only "
            "(camera may still be dataset GT)"
        )
    elif gt_proprio is not None:
        session.use_gt_proprio = True
        logger.info(
            "dataset GT proprio enabled (episode fallback when g1_debug absent)"
        )
    total_frames_sent = 0
    robot_proprio: RobotProprioSource | None = None

    def _control_idx() -> int:
        if str(args.camera_source).strip().lower() == "gt":
            return int(args.gt_camera_start_idx) + total_frames_sent
        return total_frames_sent

    def _robot_proprio_kwargs() -> dict:
        # sim: hands from g1_debug on :5557; real: optional :5558 brainco_hand overlay
        hand_host = str(getattr(args, "hand_zmq_host", "") or "").strip()
        hand_port = int(getattr(args, "hand_zmq_port", 0) or 0)
        hand_topic = str(
            getattr(args, "hand_zmq_topic", "brainco_hand") or "brainco_hand"
        ).strip()
        kw: dict = {
            "host": args.state_zmq_host.strip(),
            "port": int(args.state_zmq_port),
        }
        if hand_host and hand_port > 0:
            kw["hand_host"] = hand_host
            kw["hand_port"] = hand_port
            kw["hand_topic"] = hand_topic
        return kw

    def _log_robot_proprio_sub() -> None:
        logger.info(
            "robot proprio SUB tcp://%s:%d — body29 from g1_debug; "
            "revo2 from same msg (sim) or hand_zmq overlay (real)",
            args.state_zmq_host,
            int(args.state_zmq_port),
        )
        hand_host = str(getattr(args, "hand_zmq_host", "") or "").strip()
        hand_port = int(getattr(args, "hand_zmq_port", 0) or 0)
        hand_topic = str(
            getattr(args, "hand_zmq_topic", "brainco_hand") or "brainco_hand"
        ).strip()
        if hand_host and hand_port > 0:
            logger.info(
                "robot hand proprio SUB tcp://%s:%d topic=%s (overrides g1_debug hands)",
                hand_host,
                hand_port,
                hand_topic,
            )
        else:
            logger.info(
                "robot hand proprio: from g1_debug left/right_hand_q (no HAND_ZMQ)"
            )

    try:
        frame0_prefix_logged = [False]
        if proprio_source == "bootstrap-roll-forward":
            bootstrap_src = RobotProprioSource(**_robot_proprio_kwargs())
            try:
                first = bootstrap_src.wait_first(timeout_s=float(args.camera_wait_s))
                d_raw, _anchor = unified_from_g1_debug(first)
                log_frame0_proprio_raw(d_raw, g1_msg=first, frame=0, log=logger)
                frame0_prefix_logged[0] = True
                norm0 = normalize_unified_proprio(processor, d_raw)
                session.set_proprio_gt(norm0.reshape(1, -1))
                logger.info(
                    "bootstrap proprio from g1_debug tcp://%s:%d keys=%s — "
                    "subsequent body proprio rolls forward from model tokens",
                    args.state_zmq_host,
                    int(args.state_zmq_port),
                    sorted(first.keys())[:12],
                )
            finally:
                bootstrap_src.close()
        elif proprio_source in {"robot", "hybrid", "robot_gt_hand"}:
            robot_proprio = RobotProprioSource(**_robot_proprio_kwargs())
            if args.wait_robot_proprio:
                first = robot_proprio.wait_first(timeout_s=float(args.camera_wait_s))
                d_raw, _anchor = unified_from_g1_debug(first)
                log_frame0_proprio_raw(d_raw, g1_msg=first, frame=0, log=logger)
                frame0_prefix_logged[0] = True
                # proprio41 path: include revo2 from g1_debug / brainco overlay
                pmean = getattr(processor, "proprio_mean", None)
                if pmean is not None and int(pmean.numel()) == 41:
                    norm0 = robot_proprio.build_normalized(processor, msg=first)
                else:
                    norm0 = normalize_unified_proprio(processor, d_raw)
                session.set_proprio_gt(norm0.reshape(1, -1))
                logger.info(
                    "robot proprio ready tcp://%s:%d keys=%s hand_ep=%s hand_ready=%s",
                    args.state_zmq_host,
                    int(args.state_zmq_port),
                    sorted(first.keys())[:12],
                    robot_proprio.hand_endpoint or "-",
                    robot_proprio.hand_ready,
                )
            else:
                _log_robot_proprio_sub()
        elif args.wait_deploy_state:
            _wait_for_deploy_state(
                args.state_zmq_host.strip(),
                int(args.state_zmq_port),
                float(args.camera_wait_s),
            )

        zmq_enabled = not args.no_zmq
        pub: zmq.Socket | None = None
        ctx: zmq.Context | None = None
        send_lock = threading.Lock()
        hand_cmd_pub: zmq.Socket | None = None
        hand_cmd_tap: zmq.Socket | None = None
        hand_cmd_seq = {"n": 0}
        hand_cmd_topic = str(getattr(args, "hand_cmd_zmq_topic", "brainco_hand_cmd") or "brainco_hand_cmd")
        if zmq_enabled:
            ctx = zmq.Context()
            pub = ctx.socket(zmq.PUB)
            pub.bind(f"tcp://{args.zmq_host}:{args.zmq_port}")
            time.sleep(0.5)
            logger.info("bound tcp://%s:%d", args.zmq_host, args.zmq_port)
            hand_cmd_host = str(getattr(args, "hand_cmd_zmq_host", "") or "").strip()
            hand_cmd_port = int(getattr(args, "hand_cmd_zmq_port", 0) or 0)
            if hand_cmd_port != 0 and hand_cmd_host:
                hand_cmd_pub = ctx.socket(zmq.PUB)
                hand_cmd_ep = f"tcp://{hand_cmd_host}:{hand_cmd_port}"
                hand_cmd_pub.connect(hand_cmd_ep)
                tap_bind = os.environ.get("HAND_CMD_TAP_BIND", "tcp://127.0.0.1:15571").strip()
                if tap_bind:
                    hand_cmd_tap = ctx.socket(zmq.PUB)
                    try:
                        hand_cmd_tap.bind(tap_bind)
                    except zmq.ZMQError as exc:
                        logger.warning("hand_cmd tap bind %s failed: %s", tap_bind, exc)
                        hand_cmd_tap.close(linger=0)
                        hand_cmd_tap = None
                time.sleep(0.2)
                logger.info(
                    "hand_cmd PUB connect %s topic=%s tap=%s",
                    hand_cmd_ep,
                    hand_cmd_topic,
                    tap_bind if hand_cmd_tap is not None else "off",
                )
            deploy_started = False
            if args.stream_now:
                time.sleep(float(args.start_delay_s))
                _send_deploy_start_commands(pub, send_lock=send_lock)
                deploy_started = True
            else:
                # Locked contract: STREAM → ARM(+start_streamed) → ENTER(by eval) → READY → tokens.
                # start_streamed must precede ENTER; READY only gates token tx (not CONTROL).
                if args.stream_flag:
                    _wait_flag(
                        Path(args.stream_flag),
                        label="stream",
                        timeout_s=float(args.arm_timeout_s),
                    )
                    time.sleep(float(args.start_delay_s))
                if args.arm_flag:
                    _wait_flag(
                        Path(args.arm_flag),
                        label="arm",
                        timeout_s=float(args.arm_timeout_s),
                    )
                    time.sleep(float(args.start_delay_s))
                    _send_deploy_arm_commands(pub, send_lock=send_lock)
                    _send_deploy_start_commands(pub, send_lock=send_lock)
                if args.ready_flag:
                    _wait_flag(
                        Path(args.ready_flag),
                        label="ready",
                        timeout_s=float(args.ready_timeout_s),
                    )
                    time.sleep(float(args.start_delay_s))
                deploy_started = bool(args.arm_flag or args.ready_flag or args.stream_flag)
        else:
            logger.info("ZMQ disabled — outputs will be saved to npz only")
            deploy_started = False

        inference_queue: queue.Queue = queue.Queue(maxsize=1)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        busy_event = threading.Event()

        deploy_keyboard: DeployKeyboardListener | None = None
        zmq_keyboard = None
        deploy_state: DeployCommandState | None = None
        inference_armed = threading.Event()
        wait_deploy_p = bool(getattr(args, "wait_deploy_p", False))
        if not wait_deploy_p:
            inference_armed.set()
        wait_p_logged = [False]

        def _arm_inference() -> None:
            if not inference_armed.is_set():
                inference_armed.set()
                logger.info(
                    "inference armed (p resume) — GPU prefetch starting; "
                    "press i for initial pose, then p to stream tokens"
                )

        def _on_pause_change(paused: bool) -> None:
            if paused:
                inference_armed.clear()
                logger.info("inference paused (p)")
            else:
                _arm_inference()

        if zmq_enabled:
            deploy_state = DeployCommandState(pause_loop=wait_deploy_p)
            kb_port = int(getattr(args, "keyboard_zmq_port", 0) or 0)
            if kb_port > 0:
                zmq_keyboard = open_zmq_keyboard_subscriber(
                    str(getattr(args, "keyboard_zmq_host", "127.0.0.1")),
                    kb_port,
                )
                logger.info(
                    "ZMQ keyboard SUB tcp://%s:%d (k/p/i/[/] — run publish_deploy_keyboard_zmq.py)",
                    getattr(args, "keyboard_zmq_host", "127.0.0.1"),
                    kb_port,
                )

        if zmq_enabled and args.deploy_keyboard:
            assert pub is not None
            assert deploy_state is not None
            deploy_keyboard = DeployKeyboardListener(
                pub=pub,
                stop_event=stop_event,
                send_lock=send_lock,
                state=deploy_state,
                on_quit=lambda: stop_event.set(),
                on_inference_arm=_arm_inference if wait_deploy_p and zmq_keyboard is None else None,
            )
            deploy_keyboard.start()
        elif wait_deploy_p and zmq_keyboard is None:
            logger.warning(
                "--wait-deploy-p ignored without keyboard (enable --keyboard-zmq-port or stdin --deploy-keyboard)"
            )
            inference_armed.set()

        robot_proprio_ready_logged = [False]
        gt_fallback_logged = [False]
        defer_inference_without_robot = proprio_source in {
            "robot",
            "hybrid",
        } and not bool(args.seed_proprio) and gt_proprio is None
        defer_inference_logged = [False]
        vlm_benchmark_done = [False]
        rtc_state = RtcInferState()
        steps_since_query = {"n": 0}
        last_trigger_time: dict[str, float] = {"t": 0.0}

        def _steps_since_query() -> int:
            return int(steps_since_query["n"])

        worker = threading.Thread(
            target=_inference_worker,
            kwargs={
                "inference_queue": inference_queue,
                "result_queue": result_queue,
                "stop_event": stop_event,
                "busy_event": busy_event,
                "session": session,
                "model": model,
                "processor": processor,
                "camera": camera,
                "get_control_idx": _control_idx,
                "prompt": prompt,
                "chunk_h": chunk_h,
                "rtc_cfg": rtc_cfg,
                "rtc_state": rtc_state,
                "steps_since_query": _steps_since_query,
                "recorder": recorder,
                "robot_proprio": robot_proprio,
                "gt_proprio": gt_proprio,
                "proprio_source": proprio_source,
                "robot_proprio_ready_logged": robot_proprio_ready_logged,
                "gt_fallback_logged": gt_fallback_logged,
                "frame0_prefix_logged": frame0_prefix_logged,
                "defer_inference_without_robot": defer_inference_without_robot,
                "defer_inference_logged": defer_inference_logged,
                "vlm_input": str(args.vlm_input).strip().lower(),
                "benchmark_vlm_input": bool(args.benchmark_vlm_input),
                "benchmark_done": vlm_benchmark_done,
                "print_dataset_body_xyz": bool(getattr(args, "print_dataset_body_xyz", False)),
                "train_aligned_vlm": bool(getattr(args, "train_aligned_vlm", True)),
                "vlm_video_delta_indices": vlm_video_delta_indices,
                "hand_mode": (
                    HAND_MODE_REVO2
                    if (
                        str(cfg.data.get("hand_model") or "").strip().lower() == "revo2"
                        or "revo2"
                        in str(
                            cfg.data.get("unified_supervision_dataset") or ""
                        ).lower()
                        or int(getattr(args, "hand_cmd_zmq_port", 0) or 0) != 0
                    )
                    else hand_mode_from_unified_dataset(
                        str(cfg.data.get("unified_supervision_dataset") or "g1_sonic").strip()
                    )
                ),
            },
            name="phi0_closed_loop_infer",
            daemon=True,
        )
        worker.start()

        control_fps = float(args.control_fps)
        loop_period = 1.0 / control_fps
        scheduling = str(getattr(args, "scheduling", "prefetch")).strip().lower()
        min_infer_interval = resolve_min_inference_interval(float(args.inference_rate))
        if scheduling == "legacy" and min_infer_interval is None:
            min_infer_interval = 0.4
        prefetch_margin = prefetch_trigger_steps(
            float(getattr(args, "infer_latency_budget", 0.35)),
            control_fps,
            safety_steps=int(getattr(args, "infer_prefetch_safety_steps", 2)),
        )
        # RTC on: play/replan on execution_horizon (s); still predict full chunk_h for soft-mask.
        play_horizon = rtc_play_horizon(rtc_cfg, int(chunk_h))
        action_horizon = int(play_horizon)
        rtc_on = bool(rtc_cfg.get("enabled"))
        if rtc_on:
            d_lead = max(1, int(rtc_cfg.get("inference_delay") or 6))
            prefetch_margin = min(
                int(prefetch_margin),
                int(d_lead),
                max(1, int(play_horizon) // 2),
            )

        cached_chunk: ActionChunk | None = None
        action_chunk_index = 0
        last_inference_time = 0.0
        zmq_frame_counter = 0
        gt_max_frames = (
            getattr(camera, "max_control_frames", None)
            if str(args.camera_source).strip().lower() == "gt"
            else None
        )
        # GT already sized via motion_seconds → max_control_frames; wall deadline
        # would cut short while waiting for arm/p/empty control ticks.
        deadline = (
            time.monotonic() + float(args.motion_seconds)
            if float(args.motion_seconds) > 0 and gt_max_frames is None
            else None
        )

        cam_mode = (
            f"GT ep{int(args.gt_camera_episode)}"
            if str(args.camera_source).strip().lower() == "gt"
            else f"SONIC live tcp://{args.camera_host}:{args.camera_port}"
        )
        logger.info(
            "closed loop: camera=%s control=%.1fHz sched=%s prefetch=%d infer_cap=%s "
            "chunk_h=%d play_h=%d proprio=%s rtc=%s zmq=%s infer_arm=%s vlm_input=%s",
            cam_mode,
            control_fps,
            scheduling,
            prefetch_margin,
            f"{float(args.inference_rate):.2f}Hz" if min_infer_interval is not None else "ASAP",
            chunk_h,
            play_horizon,
            proprio_source,
            "on" if rtc_on else "off",
            "on" if zmq_enabled else "off (npz only)",
            "p" if wait_deploy_p else "immediate",
            str(args.vlm_input).strip().lower(),
        )
        if wait_deploy_p:
            if zmq_keyboard is not None:
                logger.info(
                    "waiting for ZMQ keyboard p (5580) to resume inference; "
                    "k=control loop, i=initial pose, p=stream tokens"
                )
            else:
                logger.info("waiting for deploy keyboard p before inference (] to stream tokens after)")

        def _should_trigger() -> bool:
            if not inference_armed.is_set():
                return False
            pipeline_idle = _pipeline_idle(
                busy_event=busy_event,
                inference_queue=inference_queue,
                result_queue=result_queue,
            )
            if scheduling == "legacy":
                return should_trigger_new_inference(
                    cached_chunk_exists=(cached_chunk is not None),
                    inference_thread_running=busy_event.is_set(),
                    time_since_last_inference=time.monotonic() - last_inference_time,
                    inference_interval=float(min_infer_interval or 0.4),
                )
            return should_trigger_chunk_prefetch(
                cached_chunk_exists=(cached_chunk is not None),
                pipeline_idle=pipeline_idle,
                action_chunk_index=action_chunk_index,
                action_horizon=action_horizon,
                prefetch_steps=prefetch_margin,
                min_interval_s=min_infer_interval,
                time_since_last_trigger_s=time.monotonic() - last_trigger_time["t"],
                steps_since_chunk_consumed=int(steps_since_query["n"]),
                allow_early_overlap=not rtc_on,
            )

        try:
            while deadline is None or time.monotonic() < deadline:
                if gt_max_frames is not None and total_frames_sent >= int(gt_max_frames):
                    logger.info("GT episode exhausted (%d frames)", total_frames_sent)
                    break
                t_start = time.monotonic()
                if robot_proprio is not None:
                    g1_msg = robot_proprio.poll()
                    if g1_msg is not None and io_tracker is not None:
                        io_tracker.tick(
                            "g1_debug",
                            first_log=(
                                "first g1_debug tcp://"
                                f"{args.state_zmq_host}:{int(args.state_zmq_port)} "
                                f"keys={sorted(g1_msg.keys())[:12]}"
                            ),
                        )

                if io_tracker is not None:
                    io_tracker.poll_log()

                if zmq_keyboard is not None and pub is not None and deploy_state is not None:
                    kb_key = zmq_keyboard.read_msg()
                    if kb_key is not None:
                        kb_result = handle_vla_zmq_key(
                            kb_key,
                            pub,
                            deploy_state,
                            send_lock=send_lock,
                            on_quit=lambda: stop_event.set(),
                            on_pause_change=_on_pause_change if wait_deploy_p else None,
                            on_prompt_change=lambda p: logger.info(
                                '[zmq keyboard] prompt change ignored at runtime: "%s"', p
                            ),
                        )
                        if not kb_result.continue_running:
                            break
                        if kb_result.reset_action_cache:
                            cached_chunk = None
                            action_chunk_index = 0
                            zmq_frame_counter = 0
                            steps_since_query["n"] = 0
                            last_inference_time = 0.0

                if not inference_armed.is_set() and not wait_p_logged[0]:
                    wait_p_logged[0] = True
                    if zmq_keyboard is not None:
                        logger.info(
                            "inference idle — press p on keyboard publisher (tcp://%s:%d)",
                            getattr(args, "keyboard_zmq_host", "127.0.0.1"),
                            int(getattr(args, "keyboard_zmq_port", 5580) or 5580),
                        )
                    else:
                        logger.info(
                            "inference idle — press p in this terminal (CONTROL+planner) to start"
                        )

                try:
                    chunk, infer_t0, trigger_steps = result_queue.get_nowait()
                    delay = time.monotonic() - infer_t0
                    steps_now = int(steps_since_query["n"])
                    if scheduling == "legacy":
                        action_chunk_index = calculate_latency_compensated_index(
                            delay, control_fps, action_horizon
                        )
                        idx_mode = "wall"
                    else:
                        action_chunk_index = chunk_index_when_result_arrives(
                            steps_now,
                            trigger_steps,
                            action_horizon,
                        )
                        idx_mode = "steps"
                    wall_idx = calculate_latency_compensated_index(
                        delay, control_fps, action_horizon
                    )
                    chunk_cover_s = max(action_horizon - 1, 1) / control_fps
                    if delay > chunk_cover_s * 0.95:
                        logger.warning(
                            "infer latency %.3fs > chunk cover %.3fs — idx=%d; "
                            "speed up GPU infer or increase chunk_h",
                            delay,
                            chunk_cover_s,
                            action_chunk_index,
                        )
                    cached_chunk = chunk
                    last_inference_time = time.monotonic()
                    steps_since_query["n"] = 0
                    logger.info(
                        "new chunk idx=%d (%s=%d wall=%d trigger@%d latency=%.3fs) token0=%+.3f",
                        action_chunk_index,
                        idx_mode,
                        action_chunk_index,
                        wall_idx,
                        trigger_steps,
                        delay,
                        float(chunk.tokens[action_chunk_index, 0]),
                    )
                    if zmq_enabled and args.stream_now and not deploy_started:
                        assert pub is not None
                        time.sleep(float(args.start_delay_s))
                        _send_deploy_start_commands(pub, send_lock=send_lock)
                        deploy_started = True
                except queue.Empty:
                    pass

                if cached_chunk is None:
                    if _should_trigger():
                        _try_enqueue_inference(
                            inference_queue=inference_queue,
                            last_trigger_time=last_trigger_time,
                        )
                    _sleep_remaining(t_start, loop_period)
                    continue

                if _should_trigger():
                    _try_enqueue_inference(
                        inference_queue=inference_queue,
                        last_trigger_time=last_trigger_time,
                    )

                stream_tokens = True
                if wait_deploy_p and deploy_state is not None:
                    if zmq_keyboard is not None:
                        stream_tokens = (
                            not deploy_state.pause_loop and not deploy_state.planner_mode
                        )
                    elif deploy_state.planner_mode:
                        stream_tokens = False

                if not stream_tokens:
                    _sleep_remaining(t_start, loop_period)
                    continue

                idx = min(action_chunk_index, cached_chunk.horizon - 1)
                ramp = hand_ramp_weights(total_frames_sent + 1, int(args.hand_ramp_frames))[-1]
                tok = cached_chunk.tokens[idx]
                lh_raw = cached_chunk.left[idx]
                rh_raw = cached_chunk.right[idx]
                lh = lh_raw * ramp
                rh = rh_raw * ramp
                grip346_max = float(
                    np.max(np.abs(np.concatenate([lh_raw, rh_raw], axis=0)))
                )
                if zmq_enabled:
                    assert pub is not None
                    msg = pack_latent_action_message(
                        motion_token=tok,
                        frame_index=np.array([zmq_frame_counter], dtype=np.int64),
                        left_hand_joints=lh,
                        right_hand_joints=rh,
                    )
                    with send_lock:
                        pub.send(msg)
                    if hand_cmd_pub is not None:
                        hand_cmd_seq["n"] += 1
                        cmd = pack_brainco_hand_cmd(
                            lh,
                            rh,
                            sequence=hand_cmd_seq["n"],
                        )
                        parts = [
                            hand_cmd_topic.encode("utf-8"),
                            json.dumps(cmd).encode("utf-8"),
                        ]
                        with send_lock:
                            hand_cmd_pub.send_multipart(parts)
                            if hand_cmd_tap is not None:
                                hand_cmd_tap.send_multipart(parts)
                if recorder is not None:
                    recorder.record_output(
                        token=tok,
                        left=lh,
                        right=rh,
                        control_idx=_control_idx(),
                        chunk_idx=idx,
                        frame_index=zmq_frame_counter,
                        hand_ramp=float(ramp),
                        timestamp=time.monotonic(),
                    )
                if io_tracker is not None:
                    io_tracker.tick(
                        "zmq_out",
                        first_log=(
                            f"first output frame={zmq_frame_counter + 1} "
                            f"token0={float(tok[0]):+.3f} "
                            f"zmq={'on' if zmq_enabled else 'off'}"
                        ),
                    )
                zmq_frame_counter += 1
                total_frames_sent += 1
                steps_since_query["n"] += 1
                action_chunk_index = min(action_chunk_index + 1, action_horizon - 1)

                if total_frames_sent == 1 or total_frames_sent % 50 == 0:
                    verb = "tx" if zmq_enabled else "rec"
                    logger.info(
                        "%s frame=%d chunk_i=%d token0=%+.3f "
                        "L_hand_max=%+.3f R_hand_max=%+.3f grip346_max=%+.3f ramp=%.2f",
                        verb,
                        total_frames_sent,
                        idx,
                        float(cached_chunk.tokens[idx, 0]),
                        float(np.max(np.abs(lh))),
                        float(np.max(np.abs(rh))),
                        grip346_max,
                        float(ramp),
                    )

                _sleep_remaining(t_start, loop_period)
        except KeyboardInterrupt:
            logger.info("stopped by user")
        finally:
            stop_event.set()
            worker.join(timeout=2.0)
            if pub is not None:
                pub.close()
            if hand_cmd_pub is not None:
                hand_cmd_pub.close(linger=0)
            if hand_cmd_tap is not None:
                hand_cmd_tap.close(linger=0)
            if ctx is not None:
                ctx.term()
            logger.info(
                "%s %d frames",
                "sent" if zmq_enabled else "recorded",
                total_frames_sent,
            )
            if io_tracker is not None:
                totals = io_tracker.summary()
                logger.info(
                    "io_totals camera=%d g1_debug=%d zmq_out=%d",
                    totals.get("camera", 0),
                    totals.get("g1_debug", 0),
                    totals.get("zmq_out", 0),
                )
            if recorder is not None and obs_path is not None and output_path is not None:
                if io_tracker is not None:
                    recorder._meta["io_totals"] = io_tracker.summary()
                recorder.save(obs_path, output_path)
            if robot_proprio is not None:
                robot_proprio.close()
            if zmq_keyboard is not None:
                zmq_keyboard.close()
    finally:
        camera.close()


def main() -> None:
    if "--self-check" in sys.argv:
        _io_rate_self_check()
        print("io_rate self-check ok")
        return
    args = parse_args()
    if bool(getattr(args, "no_gt_fallback", False)) and str(args.camera_source).strip().lower() == "gt":
        raise SystemExit("--no-gt-fallback is incompatible with --camera-source=gt")
    if args.no_zmq and not args.record_dir.strip() and not (
        args.record_obs.strip() and args.record_output.strip()
    ):
        raise SystemExit("--no-zmq requires --record-dir (or --record-obs and --record-output)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_closed_loop(args)


if __name__ == "__main__":
    main()
