#!/usr/bin/env python3
"""830 walk 3-terminal TRT student (PUB :5556; deploy SUB).

Vision: ``--vision-source zmq`` needs T3 camera; ``cache`` skips T3.
Keeps T1 sim + T3 vision running — do not use run_cl830_trt_local.sh.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

_PHI0 = Path(__file__).resolve().parents[2]
for root in (_PHI0 / "subpackages", _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_ORIG_PATH = Path(__file__).resolve().parent / "phi0_chunk_student_sonic_closed_loop_zmq.py"
_spec = importlib.util.spec_from_file_location("_sonic_cl_orig", _ORIG_PATH)
assert _spec and _spec.loader
_orig = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _orig
_spec.loader.exec_module(_orig)


def _resolve_vision_source(explicit: str) -> str:
    raw = (explicit or os.environ.get("PHI0_CL_3TERM_VISION", "zmq")).strip().lower()
    if raw in ("cache", "frame_cache", "latent_cache"):
        return "cache"
    if raw in ("zmq", "live", "camera"):
        return "zmq"
    if raw in ("disk", "mp4", "ref"):
        return "disk"
    raise ValueError(f"unknown vision-source {raw!r} (cache|zmq|disk)")


@dataclass
class Cl3TermTrtConfig:
    ckpt: Path
    ref_root: Path
    ref_start: int = 0
    max_frames: int = 0
    vision_source: str = "zmq"
    camera_host: str = "127.0.0.1"
    camera_port: int = 5555
    camera_wait_s: float = 120.0
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    state_zmq_host: str = "127.0.0.1"
    state_zmq_port: int = 5557
    fps: float = 50.0
    start_delay_s: float = 0.2
    hand_ramp_frames: int = 0
    wait_g1_debug_s: float = 120.0
    auto_deploy_start: bool = True


def main(config: Cl3TermTrtConfig) -> None:
    mode = _resolve_vision_source(config.vision_source)
    os.environ.setdefault("PHI0_CL_PROPRIO_BODY", "live")
    os.environ.setdefault("PHI0_CL_HAND_OBS", "commanded")
    os.environ.setdefault("PHI0_HAND_MODE", "dex3")
    os.environ.setdefault("PHI0_NEWTON_REVO2", "0")
    os.environ.setdefault("USE_RTC", "1")
    os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
    os.environ.setdefault("PHI0_ADALN_ZERO_VISION", "1")
    os.environ.setdefault("PHI0_VLM_HOLD_PERIOD", "1")

    if mode == "cache":
        os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "1"
        os.environ["PHI0_VLM_FRAME_CACHE_SKIP_VIDEO"] = "1"
    else:
        os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "0"
        os.environ["PHI0_VLM_FRAME_CACHE_SKIP_VIDEO"] = "0"

    orig_cfg = _orig.ClConfig(
        ckpt=config.ckpt,
        ref_root=config.ref_root,
        ref_start=int(config.ref_start),
        max_frames=int(config.max_frames),
        zmq_host=config.zmq_host,
        zmq_port=int(config.zmq_port),
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=int(config.state_zmq_port),
        fps=float(config.fps),
        start_delay_s=float(config.start_delay_s),
        hand_ramp_frames=int(config.hand_ramp_frames),
        wait_g1_debug_s=float(config.wait_g1_debug_s),
        auto_deploy_start=bool(config.auto_deploy_start),
    )

    if mode == "zmq":
        _deploy_dir = Path(__file__).resolve().parent
        if str(_deploy_dir) not in sys.path:
            sys.path.insert(0, str(_deploy_dir))
        import cl_830_3term_runner  # noqa: WPS433

        cl_830_3term_runner.run_sonic_cl_zmq(
            orig_cfg,
            camera_host=str(config.camera_host),
            camera_port=int(config.camera_port),
            camera_wait_s=float(config.camera_wait_s),
        )
    else:
        _orig.main(orig_cfg)


if __name__ == "__main__":
    main(tyro.cli(Cl3TermTrtConfig))
