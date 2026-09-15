"""Live dual-camera ZMQ source for 830 walk 3-terminal closed-loop (new path only).

Compatible with ``DistillVlmHold`` via ``frames_from_ref`` — same tensor layout as
``RefVideoFrameSource`` ([B,1,C,H,W] float in [0,1]). Prefer ``frames_hwc_from_ref``
for deploy encode (uint8 HWC, no float roundtrip).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

_EGO_KEYS = ("ego_view", "head", "observation.images.ego_view")
_CHEST_KEYS = (
    "chest_forward",
    "left_wrist",
    "observation.images.left_wrist",
    "observation.images.chest_forward",
)


def _pick_image(images: dict, keys: tuple[str, ...], *, label: str) -> np.ndarray:
    for key in keys:
        if key in images:
            return np.asarray(images[key], dtype=np.uint8)
    available = sorted(images.keys())
    if label == "ego" and available:
        print(f"[cl_830_zmq] WARN: ego missing; using {available[0]}", flush=True)
        return np.asarray(images[available[0]], dtype=np.uint8)
    raise KeyError(f"{label} camera missing (have {available})")


def _rgb_nhwc_to_b1chw(rgb: np.ndarray) -> torch.Tensor:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    # float() allocates; avoid an extra uint8 copy when already contiguous.
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).unsqueeze(1)


@dataclass
class ZmqDualCameraFrameSource:
    """Read ego + chest from ``ComposedCameraClientSensor`` (tcp://host:port)."""

    host: str = "127.0.0.1"
    port: int = 5555
    wait_s: float = 30.0
    client: Any = field(default=None, repr=False)
    _last_images: dict | None = field(default=None, repr=False)
    _last_msg: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

            self.client = ComposedCameraClientSensor(
                server_ip=str(self.host), port=int(self.port)
            )
            self._probe()

    def _probe(self) -> None:
        deadline = time.monotonic() + float(self.wait_s)
        last_keys: list[str] = []
        while time.monotonic() < deadline:
            msg = self.client.read(blocking=False)
            if msg and msg.get("images"):
                self._last_msg = msg
                self._last_images = msg["images"]
                keys = sorted(self._last_images.keys())
                print(
                    f"[cl_830_zmq] camera tcp://{self.host}:{self.port} ready keys={keys}",
                    flush=True,
                )
                return
            if msg:
                last_keys = sorted((msg.get("images") or {}).keys())
            time.sleep(0.05)
        hint = f" keys seen: {last_keys}" if last_keys else ""
        raise TimeoutError(
            f"no frames from tcp://{self.host}:{self.port} within {self.wait_s:.0f}s{hint}"
        )

    def _poll_images(self) -> dict:
        """Non-blocking drain to newest frame; return stale immediately if available.

        ``ComposedCameraClientSensor.read(blocking=False)`` returns the *same*
        cached dict when no new ZMQ message arrives — must not treat that as
        a new frame or drain spins forever and never reaches VLM encode.
        """
        while True:
            msg = self.client.read(blocking=False)
            if not msg or not msg.get("images"):
                break
            if msg is self._last_msg:
                break
            self._last_msg = msg
            self._last_images = msg["images"]
        if self._last_images is not None:
            return self._last_images
        # Cold start only: wait briefly for first frame.
        deadline = time.monotonic() + 2.0
        last_keys: list[str] = []
        while time.monotonic() < deadline:
            msg = self.client.read(blocking=False)
            if msg and msg.get("images") and msg is not self._last_msg:
                self._last_msg = msg
                self._last_images = msg["images"]
                return self._last_images
            if msg:
                last_keys = sorted((msg.get("images") or {}).keys())
            time.sleep(0.01)
        hint = f" keys seen: {last_keys}" if last_keys else ""
        raise TimeoutError(f"no camera frame within 2s{hint}")

    def _read_dual_hwc(self) -> tuple[np.ndarray, np.ndarray]:
        images = self._poll_images()
        ego = _pick_image(images, _EGO_KEYS, label="ego")
        chest = _pick_image(images, _CHEST_KEYS, label="chest")
        return ego, chest

    def _read_dual_b1chw(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego, chest = self._read_dual_hwc()
        return _rgb_nhwc_to_b1chw(ego), _rgb_nhwc_to_b1chw(chest)

    def frames_hwc_from_ref(
        self,
        ref: Any,
        times: np.ndarray | torch.Tensor,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Latest ego+chest uint8 HWC, repeated for batch size of ``times``."""
        del ref
        ts = (
            times.detach().long().cpu().numpy()
            if isinstance(times, torch.Tensor)
            else np.asarray(times, dtype=np.int64)
        ).reshape(-1)
        b = int(ts.shape[0])
        ego, chest = self._read_dual_hwc()
        return [ego] * b, [chest] * b

    def frames_at(
        self,
        episode_index: np.ndarray | list[int],
        frame_index: np.ndarray | list[int],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del episode_index, frame_index
        ego, chest = self._read_dual_b1chw()
        b = 1
        ego_b = ego.expand(b, -1, -1, -1, -1)
        chest_b = chest.expand(b, -1, -1, -1, -1)
        if device is not None:
            ego_b = ego_b.to(device=device)
            chest_b = chest_b.to(device=device)
        return ego_b, chest_b

    def frames_from_ref(
        self,
        ref: Any,
        times: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ref
        ts = (
            times.detach().long().cpu().numpy()
            if isinstance(times, torch.Tensor)
            else np.asarray(times, dtype=np.int64)
        ).reshape(-1)
        b = int(ts.shape[0])
        egos: list[torch.Tensor] = []
        chests: list[torch.Tensor] = []
        for _ in range(b):
            ego, chest = self._read_dual_b1chw()
            egos.append(ego)
            chests.append(chest)
        ego_t = torch.cat(egos, dim=0)
        chest_t = torch.cat(chests, dim=0)
        if device is not None:
            ego_t = ego_t.to(device=device)
            chest_t = chest_t.to(device=device)
        return ego_t, chest_t

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()
