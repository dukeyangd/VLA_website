"""ZMQ-vision variant runner for 830 3-terminal CL (patches RefVideoFrameSource only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PHI0 = Path(__file__).resolve().parents[2]
for root in (_PHI0 / "subpackages", _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_deploy = Path(__file__).resolve().parent
if str(_deploy) not in sys.path:
    sys.path.insert(0, str(_deploy))

from cl_830_zmq_video_source import ZmqDualCameraFrameSource  # noqa: E402


def run_atm_pd_cl_zmq(
    config: Any,
    *,
    camera_host: str = "127.0.0.1",
    camera_port: int = 5555,
    camera_wait_s: float = 30.0,
) -> None:
    """Run original atm_pd_cl with live ZMQ dual camera instead of disk/cache vision."""
    import importlib.util

    import phi0.online.ref_video_vlm as ref_video_vlm

    zmq_src = ZmqDualCameraFrameSource(
        host=str(camera_host),
        port=int(camera_port),
        wait_s=float(camera_wait_s),
    )

    class _ZmqRefVideoShim:
        """Drop-in for RefVideoFrameSource; ignores dataset paths."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def frames_from_ref(self, ref: Any, times: Any, *, device: Any = None) -> Any:
            return zmq_src.frames_from_ref(ref, times, device=device)

        def frames_hwc_from_ref(self, ref: Any, times: Any) -> Any:
            return zmq_src.frames_hwc_from_ref(ref, times)

        def frames_at(self, *args: Any, **kwargs: Any) -> Any:
            return zmq_src.frames_at(*args, **kwargs)

    orig_cls = ref_video_vlm.RefVideoFrameSource
    ref_video_vlm.RefVideoFrameSource = _ZmqRefVideoShim  # type: ignore[misc,assignment]

    os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "0"
    os.environ["PHI0_VLM_FRAME_CACHE_SKIP_VIDEO"] = "0"

    orig_path = _deploy / "phi0_chunk_student_atm_lowcmd_cl.py"
    spec = importlib.util.spec_from_file_location("_atm_pd_cl_orig_run", orig_path)
    assert spec and spec.loader
    orig_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = orig_mod
    spec.loader.exec_module(orig_mod)

    print(
        f"[cl_830_3term] vision=zmq camera=tcp://{camera_host}:{camera_port} "
        f"proprio=live (VLM tower encode)",
        flush=True,
    )
    try:
        orig_mod.main(config)
    finally:
        ref_video_vlm.RefVideoFrameSource = orig_cls  # type: ignore[misc]
        zmq_src.close()


def run_sonic_cl_zmq(
    config: Any,
    *,
    camera_host: str = "127.0.0.1",
    camera_port: int = 5555,
    camera_wait_s: float = 30.0,
) -> None:
    """Run sonic TRT student CL with live ZMQ dual camera (T3 :15555)."""
    import importlib.util

    import phi0.online.ref_video_vlm as ref_video_vlm

    zmq_src = ZmqDualCameraFrameSource(
        host=str(camera_host),
        port=int(camera_port),
        wait_s=float(camera_wait_s),
    )

    class _ZmqRefVideoShim:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def frames_from_ref(self, ref: Any, times: Any, *, device: Any = None) -> Any:
            return zmq_src.frames_from_ref(ref, times, device=device)

        def frames_hwc_from_ref(self, ref: Any, times: Any) -> Any:
            return zmq_src.frames_hwc_from_ref(ref, times)

        def frames_at(self, *args: Any, **kwargs: Any) -> Any:
            return zmq_src.frames_at(*args, **kwargs)

    orig_cls = ref_video_vlm.RefVideoFrameSource
    ref_video_vlm.RefVideoFrameSource = _ZmqRefVideoShim  # type: ignore[misc,assignment]

    os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "0"
    os.environ["PHI0_VLM_FRAME_CACHE_SKIP_VIDEO"] = "0"
    # ponytail: real/sim ZMQ camera — RefVideoFrameSource shim even without ref_root/videos mp4
    os.environ["PHI0_CL_ZMQ_VISION"] = "1"

    orig_path = _deploy / "phi0_chunk_student_sonic_closed_loop_zmq.py"
    spec = importlib.util.spec_from_file_location("_sonic_cl_orig_run", orig_path)
    assert spec and spec.loader
    orig_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = orig_mod
    spec.loader.exec_module(orig_mod)

    print(
        f"[cl_830_3term] vision=zmq camera=tcp://{camera_host}:{camera_port} "
        f"proprio=live token=PUB :{int(getattr(config, 'zmq_port', 5556))} (TRT deploy SUB)",
        flush=True,
    )
    try:
        orig_mod.main(config)
    finally:
        ref_video_vlm.RefVideoFrameSource = orig_cls  # type: ignore[misc]
        zmq_src.close()
