#!/usr/bin/env python3
"""Distill ChunkStudent closed-loop → ZMQ v4 (proprio body/hand configurable).

Body default: live ``g1_debug``. Hand via ``PHI0_CL_HAND_OBS``:
``commanded`` (default; last ``hand_pred``, train ``hand_ref``),
``g1_debug|sim|measured`` (Dex3 LowState), ``tape`` (dataset ``hand_ref``),
or ``zero|nohandobs`` (zeros; nohandobs train).
``PHI0_CL_VLM_SOURCE=dataset_video``: live VLM on ref videos with
``PHI0_VLM_ENCODE_MIN_BATCH`` pad (default 8) so latent matches
``vlm_frame_latents_qwen3vl_dual`` cache — see
docs/report/deploy/vlm_frame_cache_infer_batch_LOCKED.md.
"""
from __future__ import annotations

import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
for root in (_GR00T, _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.utils.teleop.zmq.v4_latent_replay import pack_latent_action_message  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message  # noqa: E402

from phi0.deploy.dex3_gripper import (  # noqa: E402
    gripper14_batch_to_zmq_lr,
    gripper14_to_zmq_lr,
)
from phi0.deploy.robot_proprio import (  # noqa: E402
    RobotProprioSource,
    _body_q29,
    gripper14_actuator_from_g1_debug,
    gripper14_wbc_from_g1_debug,
)
from phi0.hand.hand_mode import HAND_MODE_DEX3, hand_mode_from_env, student_hand_dim  # noqa: E402
from phi0.inference.rtc import (  # noqa: E402
    blend_action_chunks_rtc,
    build_rtc_infer_prefix_kwargs,
    create_rtc_hard_mask,
    distill_rtc_deploy_cfg,
    rtc_deploy_params_ok,
    rtc_play_horizon,
    shift_action_chunk_rtc,
)
from phi0.models.adaln_exec import (  # noqa: E402
    adaln_mode_of,
    exec_clocks_for_infer,
    has_video_mask_from_ref,
    infer_zero_exec_clock_enabled,
    normalize_adaln_mode,
    student_adaln_forward_kwargs,
)
from phi0.online.distill_norm import DistillNorm, build_distill_processor  # noqa: E402
from phi0.online.exec_time import ep_relative_steps  # noqa: E402
from phi0.online.isaac_loop import (  # noqa: E402
    _attach_prompts_if_needed,
    _tape_hand_obs,
    episode_bounds_from_frame_index,
)
from phi0.online.latent_ref import load_sonic_latent_reference  # noqa: E402
from phi0.online.lazy_ref import attach_has_video_to_ref  # noqa: E402
from phi0.online.phi0_student import (  # noqa: E402
    Phi0ChunkStudent,
    build_phi0_student,
    load_student_act_state_dict,
)
from phi0.online.ref_video_vlm import RefVideoFrameSource  # noqa: E402
from phi0.online.student_obs import build_student_proprio41_hist, distill_obs_hist_len  # noqa: E402
from phi0.online.vlm_frame_latents import DualVlmFrameLatentCache  # noqa: E402
from phi0.online.vlm_hold_distill import DistillVlmHold, resolve_vlm_hold_period, vlm_encode_min_batch_from_env  # noqa: E402
from phi0.online.vision_dl_distill import (  # noqa: E402
    encode_dual_vlm_from_frame_cache,
    use_vlm_frame_latent_cache_from_env,
)


@dataclass
class ClConfig:
    ckpt: Path
    ref_root: Path
    ref_start: int = 0
    max_frames: int = 0
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    state_zmq_host: str = "127.0.0.1"
    state_zmq_port: int = 5557
    fps: float = 50.0
    start_delay_s: float = 0.5
    arm_flag: str = ""
    ready_flag: str = ""
    hand_ramp_frames: int = 0
    wait_g1_debug_s: float = 120.0
    auto_deploy_start: bool = False


def _wait_flag(path: Path, *, label: str, timeout_s: float = 240.0) -> None:
    print(f"[chunk_cl] waiting for {label} {path}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{label} flag not found: {path}")


def _send_deploy_start(pub: zmq.Socket) -> None:
    pub.send(build_command_message(start=True, stop=False, planner=True))
    time.sleep(0.2)
    pub.send(build_command_message(start=True, stop=False, planner=False))
    time.sleep(0.2)
    print("[chunk_cl] sent ZMQ command start (planner -> streamed motion)")


def _dex3_hand_policy_order_from_env() -> bool:
    """True when unified hand_pred / hand_ref use policy (WBC) joint order."""
    raw = os.environ.get("PHI0_DEX3_HAND_POLICY_ORDER", "1").strip().lower()
    return raw not in ("0", "false", "no", "deploy", "actuator")


def _async_infer_from_env() -> bool:
    """Always Psi0 RTC control. ``PHI0_CL_ASYNC_INFER=0`` is a removed bug — ignored."""
    raw = os.environ.get("PHI0_CL_ASYNC_INFER", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        print(
            "[chunk_cl] WARN: PHI0_CL_ASYNC_INFER=0 ignored "
            "(sync block-replan removed; Psi0 control only)",
            flush=True,
        )
    return True


def _psi0_rtc_rebase_selfcheck(h: int = 32, s: int = 15, d: int = 6) -> None:
    """ponytail: rebase math must match Psi0 RealTimeChunkController (ceiling: skip≠d)."""
    assert d <= s <= h - d, (d, s, h)
    t_at_done = s + d
    assert t_at_done - s == d


def _hand_zmq7(hand14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Student gripper14 → ZMQ left7/right7 in Dex3 actuator order."""
    policy_order = (
        hand_mode_from_env(default=HAND_MODE_DEX3) == HAND_MODE_DEX3
        and _dex3_hand_policy_order_from_env()
    )
    return gripper14_to_zmq_lr(hand14, policy_order=policy_order)


def _cl_proprio_body_tape() -> bool:
    """``PHI0_CL_PROPRIO=tape`` or ``PHI0_CL_PROPRIO_BODY=tape`` → fk_dof29 tape."""
    if os.environ.get("PHI0_CL_PROPRIO", "").strip().lower() in ("tape", "all"):
        return True
    return os.environ.get("PHI0_CL_PROPRIO_BODY", "live").strip().lower() in (
        "tape",
        "1",
        "true",
        "yes",
    )


def _cl_hand_obs_mode() -> str:
    """``PHI0_CL_HAND_OBS``: commanded | g1_debug/sim/measured | tape | zero."""
    raw = os.environ.get("PHI0_CL_HAND_OBS", "commanded").strip().lower()
    if raw in ("tape", "dataset"):
        return "tape"
    if raw in ("commanded", "pred", "last_pred", "hand_pred"):
        return "commanded"
    # nohandobs train: proprio hand slots = 0 (still predict hand cmds).
    if raw in ("zero", "none", "off", "nohand", "no_hand", "nohandobs"):
        return "zero"
    # sim / measured / live / g1_debug → LowState Dex3 joints
    return "g1_debug"


# 830mix_skill123_demo5 tasks.parquet — vision skills + demo5 text-only.
# Keys: 1 skill1 · 2 skill2 · 3 skill3 · 4 idle · 5 egypt · 6 spin · 7 wave · 8 bow
_CL830MIX_SKILLS: dict[str, tuple[str, bool]] = {
    "skill1": ("机器人朝黑箱子走过去。", True),
    "skill2": ("抓起黄色玩具放到篮子里。", True),
    "skill3": ("把篮子放到指定位置。", True),
    "idle": ("静止假人：完全静止站立，双臂贴腿，零移动。", False),
    "egypt": (
        "埃及肚皮舞：髋部左右隔离摆动，胸腹波浪；肘直角、腕轻弹；原地脚尖点地。",
        False,
    ),
    "spin": ("华丽单脚转：单脚绕竖直轴连转，双臂伸展，落地急停直立。", False),
    "wave": ("告别挥手：站稳，单臂过头大幅挥手，另一臂垂髋侧。", False),
    "bow": ("正式鞠躬：髋折腰低头后回直立，双手贴大腿。", False),
}
_CL830MIX_KEY_TO_SKILL = {
    "1": "skill1",
    "2": "skill2",
    "3": "skill3",
    "4": "idle",
    "i": "idle",
    "5": "egypt",
    "e": "egypt",
    "6": "spin",
    "7": "wave",
    "8": "bow",
    "b": "bow",
    "w": "skill1",
}
_CL830MIX_ALIASES = {
    "stand": "idle",
    "stay": "idle",
    "walk": "skill1",
    "walking": "skill1",
    "skill_1": "skill1",
    "pick": "skill2",
    "pick_toy": "skill2",
    "skill_2": "skill2",
    "place": "skill3",
    "place_basket": "skill3",
    "skill_3": "skill3",
    "egyptian": "egypt",
    "dance": "egypt",
    "fancy_spin": "spin",
    "fancyspin": "spin",
    "waver": "wave",
    "bowr": "bow",
    "站着": "idle",
    "站立": "idle",
    "静止": "idle",
    "走": "skill1",
    "走路": "skill1",
    "抓玩具": "skill2",
    "搬篮子": "skill3",
    "旋转": "spin",
    "转圈": "spin",
    "挥手": "wave",
    "鞠躬": "bow",
    "埃及": "egypt",
    "埃及舞": "egypt",
}


def _skill_keys_enabled() -> bool:
    """Opt-in keyboard skill switch (t3_830mix-style). Default off — gold path unchanged."""
    for key in ("CL830_SKILL_KEYS", "PHI0_SKILL_STDIN"):
        raw = os.environ.get(key, "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
    return False


def _resolve_cl830mix_skill(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    low = text.lower()
    if low in {"q", "quit", "exit"} or text in {"退出", "停"}:
        return "__quit__"
    if low in _CL830MIX_SKILLS:
        return low
    if low in _CL830MIX_ALIASES:
        return _CL830MIX_ALIASES[low]
    if text in _CL830MIX_ALIASES:
        return _CL830MIX_ALIASES[text]
    if len(low) == 1 and low in _CL830MIX_KEY_TO_SKILL:
        return _CL830MIX_KEY_TO_SKILL[low]
    return None


def _paint_skill_on_ref(ref: object, prompt: str, has_video: bool) -> None:
    n = int(len(ref))  # type: ignore[arg-type]
    ref.prompts = np.asarray([prompt] * n, dtype=object)  # type: ignore[attr-defined]
    ref.has_video = np.full(n, bool(has_video), dtype=np.bool_)  # type: ignore[attr-defined]


def _cl_vlm_dataset_video() -> bool:
    """Live VLM encode from dataset mp4 (``PHI0_CL_VLM_SOURCE=dataset_video``)."""
    raw = os.environ.get("PHI0_CL_VLM_SOURCE", "").strip().lower()
    if raw in ("dataset", "dataset_video", "video", "live"):
        return True
    if raw in ("frame_cache", "cache"):
        return False
    return os.environ.get("PHI0_USE_VLM_FRAME_LATENT_CACHE", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def main(config: ClConfig) -> None:
    # T2b CL830 default: RTC off (launch scripts may still export USE_RTC=1).
    os.environ.setdefault("USE_RTC", "0")
    os.environ.setdefault("PHI0_CHUNK_EXEC", "open_loop")
    os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
    os.environ.setdefault("PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX", "0")
    os.environ.setdefault("PHI0_HAND_MODE", "dex3")
    os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
    os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "15")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_root = str(config.ref_root.resolve())
    ref = load_sonic_latent_reference(
        ref_root,
        max_frames=int(config.max_frames) if int(config.max_frames) > 0 else 100000,
        start=int(config.ref_start),
        require_rsi=True,
        require_smpl=False,
    )
    _attach_prompts_if_needed(ref, ref_root)
    prompt_override = os.environ.get("PHI0_CL_PROMPT_OVERRIDE", "").strip()
    if prompt_override:
        ref.prompts = np.asarray([prompt_override] * len(ref), dtype=object)
        print(f"[chunk_cl] prompt override: {prompt_override[:80]!r}")
    attach_has_video_to_ref(ref, ref_root)
    skill_keys_on = _skill_keys_enabled()
    skill_name = ""
    if skill_keys_on and not prompt_override:
        # ponytail: mix skill table hardcoded from tasks.parquet; env CL830_SKILL picks start.
        init_sk = (
            os.environ.get("CL830_SKILL") or os.environ.get("SKILL") or "skill1"
        ).strip()
        resolved0 = _resolve_cl830mix_skill(init_sk)
        if resolved0 is None or resolved0 not in _CL830MIX_SKILLS:
            resolved0 = "skill1"
        p0, hv0 = _CL830MIX_SKILLS[resolved0]
        _paint_skill_on_ref(ref, p0, hv0)
        skill_name = resolved0
        print(
            f"[chunk_cl] skill_keys=on start={skill_name} has_video={int(hv0)} "
            f"prompt={p0[:40]!r}",
            flush=True,
        )
    t_len = int(len(ref))
    loop_forever = int(config.max_frames) <= 0
    episode_len = t_len
    if loop_forever:
        run_frames = episode_len
    else:
        run_frames = int(config.max_frames) if int(config.max_frames) > 0 else episode_len
        run_frames = min(run_frames, episode_len)
    max_frames = run_frames  # compat for tape shape checks / logging

    payload = torch.load(config.ckpt, map_location=device, weights_only=False)
    if payload.get("prompt_max_length") is not None:
        os.environ.setdefault("PHI0_PROMPT_MAX_LENGTH", str(int(payload["prompt_max_length"])))
    h = int(payload["horizon"])
    hist_len = int(payload.get("history_len") or distill_obs_hist_len())
    adaln = normalize_adaln_mode(str(payload.get("adaln_mode") or "progress_only"))
    vlm_dataset_video = _cl_vlm_dataset_video()
    frame_cache_on = (not vlm_dataset_video) and use_vlm_frame_latent_cache_from_env()
    frame_cache = DualVlmFrameLatentCache.open(ref_root) if frame_cache_on else None
    use_vlm_tower = not frame_cache_on
    student = Phi0ChunkStudent(
        build_phi0_student(
            device=str(device),
            use_vlm=use_vlm_tower,
            horizon=h,
            vlm_age_h=int(payload.get("vlm_age_h", 0) or 0),
            adaln_mode=adaln,
            history_len=hist_len,
        ),
        horizon=h,
        history_len=hist_len,
        require_lang_ctx=True,
    ).to(device)
    load_student_act_state_dict(student, payload, expected_horizon=h)
    student.eval()

    stats_p = config.ckpt.parent / "action_stats.json"
    proc, st = build_distill_processor(stats_p, phi0_full_v3_root=ref_root, require_z_stats=False)
    dnorm = DistillNorm(proc, st)
    dnorm.attach_to_student(student)

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(rtc.get("enabled"))
    play = int(rtc_play_horizon(rtc, h) if rtc_on else h)
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else 0

    vids = Path(ref_root) / "videos"
    zmq_vision = os.environ.get("PHI0_CL_ZMQ_VISION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    video_src = (
        None
        if frame_cache is not None
        else (
            RefVideoFrameSource(ref_root, fps=50.0)
            if zmq_vision or (vids.is_dir() and any(vids.rglob("*.mp4")))
            else None
        )
    )
    hold = DistillVlmHold(1, period=resolve_vlm_hold_period(default=1))
    fi = getattr(ref, "frame_index", None)
    if fi is None:
        ep_starts = torch.zeros(1, device=device, dtype=torch.long)
    else:
        ep_starts_np, _ = episode_bounds_from_frame_index(
            np.asarray(fi), horizon=1, n_ds=max(1, t_len - h + 1)
        )
        ep_starts = torch.as_tensor(ep_starts_np, device=device, dtype=torch.long)

    body_tape = bool(_cl_proprio_body_tape())
    hand_obs_mode = _cl_hand_obs_mode()
    hand_tape_mode = hand_obs_mode == "tape"
    # Live body+hand (+ live/ZMQ VLM): monotonic t, no episode wrap / chunk clear.
    # Tape or frame-cache still needs 0..episode_len rounds.
    live_continuous = (
        loop_forever
        and not body_tape
        and not hand_tape_mode
        and not frame_cache_on
    )
    if live_continuous:
        # ponytail: ceiling = forever wall-clock; use huge run_frames so inner loop never wraps.
        run_frames = 10**9
        max_frames = run_frames
    body_tape_np: np.ndarray | None = None
    if body_tape:
        fk = getattr(ref, "fk_dof29", None)
        if fk is None:
            raise RuntimeError("PHI0_CL_PROPRIO_BODY=tape requires ref.fk_dof29")
        body_tape_np = np.asarray(fk, dtype=np.float32)
        if body_tape_np.shape != (max_frames, 29):
            raise RuntimeError(
                f"fk_dof29 shape {body_tape_np.shape} != ({max_frames}, 29)"
            )

    robot = RobotProprioSource(host=config.state_zmq_host, port=int(config.state_zmq_port))
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{config.zmq_host}:{int(config.zmq_port)}")
    time.sleep(0.5)
    print(
        f"[chunk_cl] bound tcp://{config.zmq_host}:{int(config.zmq_port)} "
        f"state=tcp://{config.state_zmq_host}:{int(config.state_zmq_port)} "
        f"T={'∞' if live_continuous else run_frames}"
        f"{' live_continuous' if live_continuous else (' loop' if loop_forever else '')} "
        f"H={h} rtc={int(rtc_on)} "
        f"play={play} d={rtc_d} s={rtc_s} async={int(_async_infer_from_env())} "
        f"rtc_inpaint={int(rtc_on)} hard_freeze={int(rtc_on)} "
        f"adaln={adaln} "
        f"vlm={'dataset_video' if vlm_dataset_video else ('frame_cache' if frame_cache is not None else 'live')} "
        f"{'vlm_enc_min_b='+str(vlm_encode_min_batch_from_env())+' ' if vlm_dataset_video else ''}"
        f"body={'tape' if body_tape else 'live'} hand={hand_obs_mode} "
        f"dex3_policy_order={int(_dex3_hand_policy_order_from_env())}"
    )

    if config.arm_flag:
        _wait_flag(Path(config.arm_flag), label="arm")
        time.sleep(config.start_delay_s)
        _send_deploy_start(pub)
    elif config.ready_flag:
        _wait_flag(Path(config.ready_flag), label="ready")
        time.sleep(config.start_delay_s)
        _send_deploy_start(pub)
    elif config.auto_deploy_start:
        time.sleep(config.start_delay_s)
        _send_deploy_start(pub)

    deadline = time.monotonic() + float(config.wait_g1_debug_s)
    while time.monotonic() < deadline:
        if robot.poll() is not None:
            break
        time.sleep(0.02)
    else:
        raise TimeoutError(f"no g1_debug within {config.wait_g1_debug_s:.0f}s")

    # After sim RSI snap, wait until live body0 looks like walk (not stand ~-0.3).
    # Sim holds RSI until PHI0_CL_RELEASE_RSI_HOLD_FLAG is touched (see below).
    expect_body0_min = float(os.environ.get("PHI0_CL_EXPECT_BODY0_MIN", "0.05"))
    sync_s = float(os.environ.get("PHI0_CL_PROPRIO_SYNC_S", "8"))
    release_flag = (os.environ.get("PHI0_CL_RELEASE_RSI_HOLD_FLAG") or "").strip()
    if not body_tape and sync_s > 0:
        sync_deadline = time.monotonic() + sync_s
        body0 = float("nan")
        while time.monotonic() < sync_deadline:
            robot.poll()
            if robot.ready and robot.last_msg is not None:
                body0 = float(_body_q29(robot.last_msg)[0])
                if body0 >= expect_body0_min:
                    break
            time.sleep(0.02)
        print(
            f"[chunk_cl] proprio sync body0={body0:+.3f} "
            f"(need>={expect_body0_min:+.3f}) release_flag={release_flag or 'none'}",
            flush=True,
        )
        # Let C++ decoder hist fill with RSI LowState while sim still holds.
        hist_fill_s = float(os.environ.get("PHI0_CL_RSI_HIST_FILL_S", "0.30"))
        if hist_fill_s > 0:
            time.sleep(hist_fill_s)

    if hand_tape_mode and getattr(ref, "hand_ref", None) is None:
        raise RuntimeError(
            "PHI0_CL_HAND_OBS=tape requires ref.hand_ref (vision_dl dex3 / revo2 tape)"
        )
    prev_z: torch.Tensor | None = None
    prev_hand: torch.Tensor | None = None
    commanded_hand = torch.zeros(
        1, int(student_hand_dim()), device=device, dtype=torch.float32
    )
    chunk_tokens: list[np.ndarray] = []
    chunk_left: list[np.ndarray] = []
    chunk_right: list[np.ndarray] = []
    chunk_hand_wbc: list[np.ndarray] = []
    chunk_i = 0
    period = 1.0 / float(config.fps)
    ramp_denom = max(int(config.hand_ramp_frames), 1)
    t_start = time.monotonic()
    _async_infer_from_env()  # always Psi0; warns if env asks for 0
    # Non-RTC: s_min = H − prefetch. RTC: s_min = play (=s).
    prefetch_steps = max(1, int(rtc_d) if rtc_on else 6)

    # t3_830mix-style stdin skill switch (opt-in via CL830_SKILL_KEYS / PHI0_SKILL_STDIN).
    pending_skill: list[str | None] = [None]
    skill_stop = threading.Event()
    skill_lock = threading.Lock()
    cam_skill_push = None
    cam_skill_port = int(os.environ.get("CAMERA_SKILL_ZMQ_PORT", "0") or 0)
    if cam_skill_port > 0:
        # module-level zmq — do not re-import here (shadows → UnboundLocalError on ctx=).
        cam_skill_push = zmq.Context.instance().socket(zmq.PUSH)
        cam_skill_push.setsockopt(zmq.LINGER, 0)
        cam_skill_push.setsockopt(zmq.SNDHWM, 8)
        cam_host = os.environ.get("CAMERA_SKILL_ZMQ_HOST", "127.0.0.1").strip() or "127.0.0.1"
        cam_ep = f"tcp://{cam_host}:{cam_skill_port}"
        cam_skill_push.connect(cam_ep)
        print(f"[chunk_cl] camera-skill PUSH {cam_ep} (T3 PULL → reset/switch)", flush=True)

    def _notify_camera_skill(name: str) -> None:
        if cam_skill_push is None:
            return
        # ponytail: drop if T3 not up (real robot has no tape vision).
        try:
            cam_skill_push.send(str(name).encode("utf-8"), zmq.NOBLOCK)
            print(f"[chunk_cl] camera-skill notify → {name}", flush=True)
        except zmq.Again:
            print(
                f"[chunk_cl] camera-skill notify dropped (T3 not on :{cam_skill_port}?)",
                flush=True,
            )
        except zmq.ZMQError as exc:
            print(f"[chunk_cl] camera-skill notify failed: {exc}", flush=True)

    def _queue_skill(name: str) -> None:
        with skill_lock:
            pending_skill[0] = name

    def _apply_pending_skill(
        t_now: int,
        *,
        chunk_lock: threading.Lock | None = None,
        clear_async: tuple[queue.Queue, queue.Queue] | None = None,
    ) -> bool:
        """Paint prompt/has_video, clear VLM hold + RTC + chunk, reset AdaLN clock."""
        nonlocal prev_z, prev_hand, ep_starts, skill_name
        nonlocal chunk_tokens, chunk_left, chunk_right, chunk_hand_wbc, chunk_i
        with skill_lock:
            name = pending_skill[0]
            pending_skill[0] = None
        if name is None or name not in _CL830MIX_SKILLS:
            return False
        if name == skill_name:
            return False
        prompt, has_video = _CL830MIX_SKILLS[name]
        _paint_skill_on_ref(ref, prompt, has_video)
        hold.clear()
        prev_z = None
        prev_hand = None
        ep_starts = torch.tensor([int(t_now)], device=device, dtype=torch.long)
        if clear_async is not None:
            iq, rq = clear_async
            for q in (iq, rq):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        if chunk_lock is not None:
            with chunk_lock:
                chunk_tokens, chunk_left, chunk_right, chunk_hand_wbc = [], [], [], []
                chunk_i = 0
        else:
            chunk_tokens, chunk_left, chunk_right, chunk_hand_wbc = [], [], [], []
            chunk_i = 0
        skill_name = name
        _notify_camera_skill(name)
        print(
            f"[chunk_cl] switched skill={skill_name} has_video={int(has_video)} "
            f"t0={t_now} prompt={prompt[:48]!r}",
            flush=True,
        )
        return True

    def _stdin_skill_loop() -> None:
        allow_pipe = os.environ.get("PHI0_SKILL_STDIN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not sys.stdin.isatty() and not allow_pipe:
            print("[chunk_cl] skill keys: stdin not a TTY — disabled", flush=True)
            return
        print(
            "[chunk_cl] skill keys ready — Enter after: "
            "1=skill1 2=skill2 3=skill3 4=idle 5=egypt 6=spin 7=wave 8=bow | q=quit",
            flush=True,
        )
        while not skill_stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (ValueError, OSError):
                break
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                break
            got = _resolve_cl830mix_skill(line)
            if got is None:
                print(
                    f"[chunk_cl] unknown skill {line.strip()!r} — "
                    "use 1-8 / skill1|skill2|skill3|idle|egypt|spin|wave|bow / q",
                    flush=True,
                )
                continue
            if got == "__quit__":
                skill_stop.set()
                break
            _queue_skill(got)

    if skill_keys_on:
        threading.Thread(
            target=_stdin_skill_loop, name="cl830-skill-keys", daemon=True
        ).start()

    def _hand_obs_at(t: int, body_msg: dict | None) -> torch.Tensor:
        ts_t = torch.tensor([t], device=device, dtype=torch.long)
        if hand_tape_mode:
            return _tape_hand_obs(ref, ts_t, device=device)
        if hand_obs_mode == "commanded":
            return commanded_hand
        if hand_obs_mode == "zero":
            return commanded_hand  # stays zeros; never updated from preds
        assert body_msg is not None
        if _dex3_hand_policy_order_from_env():
            hand14 = gripper14_wbc_from_g1_debug(body_msg)
        else:
            hand14 = gripper14_actuator_from_g1_debug(body_msg, prefer_command=False)
        return torch.from_numpy(hand14).to(device=device, dtype=torch.float32).reshape(1, -1)

    def _replan_at(
        t: int,
        body: np.ndarray,
        hand_obs: torch.Tensor,
        *,
        rtc_shift: int | None = None,
        full_horizon: bool = False,
        inference_delay: int | None = None,
        prev_z_chunk: torch.Tensor | None = None,
        prev_hand_chunk: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        ts = torch.tensor([t], device=device, dtype=torch.long)
        obs_hist = build_student_proprio41_hist(
            body[None], hand_obs, k=hist_len
        ).to(device=device)
        steps_rel = ep_relative_steps(ts, ep_starts=ep_starts)
        lang_ctx = lang_mask = None
        if frame_cache is not None:
            if getattr(ref, "episode_index", None) is None or getattr(
                ref, "frame_index", None
            ) is None:
                raise RuntimeError(
                    "PHI0_USE_VLM_FRAME_LATENT_CACHE requires ref.episode_index "
                    "and ref.frame_index"
                )
            ti = min(int(ts[0].item()), len(ref) - 1)
            lang_ctx, lang_mask = encode_dual_vlm_from_frame_cache(
                cache=frame_cache,
                episode_indices=[int(ref.episode_index[ti])],
                frame_indices=[int(ref.frame_index[ti])],
                device=device,
                dtype=student.phi0.torch_dtype,
            )
        elif video_src is not None:
            lang_ctx, lang_mask = hold.get_or_refresh(
                student=student,
                ref=ref,
                video_src=video_src,
                ts_abs=ts,
                steps=steps_rel,
                device=device,
            )
        elif not frame_cache_on:
            raise RuntimeError(
                "PHI0_CL_VLM_SOURCE=dataset_video requires mp4 under "
                f"{ref_root}/videos (ego + wrist)"
            )
        steps_clk, _, _ = exec_clocks_for_infer(steps_rel)
        adaln_kw = student_adaln_forward_kwargs(
            exec_steps=steps_clk,
            has_video=has_video_mask_from_ref(ref, ts, device=device),
            adaln_mode=adaln_mode_of(student),
            batch_size=1,
            device=device,
        )
        # RTC: forward API is rtc_clean/rtc_delay (not prev_z/rtc_prefix_d).
        # Async Psi0: rtc_shift=plan_t (steps already played); else use train s.
        shift_n = int(0 if rtc_shift is None else rtc_shift)
        d_use = int(rtc_d if inference_delay is None else inference_delay)
        s_for_shift = shift_n if shift_n > 0 else int(rtc_s)
        rtc_kw = build_rtc_infer_prefix_kwargs(
            prev_z=prev_z_chunk,
            prev_hand=prev_hand_chunk,
            inference_delay=d_use,
            execution_horizon=s_for_shift,
            batch_size=1,
            device=device,
            norm_z=dnorm.norm_z,
            norm_hand=dnorm.norm_hand,
            enabled=bool(
                rtc_on and prev_z_chunk is not None and s_for_shift > 0 and d_use > 0
            ),
        )
        z_n, _, _, hand_n = student(
            obs_hist,
            lang_ctx=lang_ctx,
            lang_mask=lang_mask,
            **adaln_kw,
            **rtc_kw,
        )
        z_pred = dnorm.denorm_z(z_n)
        hand_pred = dnorm.denorm_hand(hand_n)
        # Train-aligned deploy hard freeze: out[:d] = shift(prev, s)[:d]
        # (offline_student_rtc_infer / act_distill_rtc_frozen_prefix.md).
        if (
            rtc_on
            and prev_z_chunk is not None
            and s_for_shift > 0
            and d_use > 0
            and rtc_deploy_params_ok(h, d_use, s_for_shift)
        ):
            prev_z_s = shift_action_chunk_rtc(prev_z_chunk, s_for_shift)
            rtc_mask = create_rtc_hard_mask(
                h, d_use, s_for_shift, device=device
            )
            z_pred = blend_action_chunks_rtc(z_pred, prev_z_s, rtc_mask)
            if prev_hand_chunk is not None:
                prev_h_s = shift_action_chunk_rtc(prev_hand_chunk, s_for_shift)
                hand_pred = blend_action_chunks_rtc(hand_pred, prev_h_s, rtc_mask)
        # Async/Psi0: always keep full H so H−s ≥ d buffer remains during infer.
        if full_horizon:
            n_exec = int(h)
        else:
            n_exec = int(play) if live_continuous else min(int(play), max(0, run_frames - t))
        if n_exec <= 0:
            return [], [], [], []
        z_np = z_pred[0, :n_exec].float().cpu().numpy()
        h_np = hand_pred[0, :n_exec].float().cpu().numpy()
        policy_order = (
            hand_mode_from_env(default=HAND_MODE_DEX3) == HAND_MODE_DEX3
            and _dex3_hand_policy_order_from_env()
        )
        lefts_a, rights_a = gripper14_batch_to_zmq_lr(h_np, policy_order=policy_order)
        if config.hand_ramp_frames > 0:
            scales = np.minimum(
                1.0, np.arange(n_exec, dtype=np.float32) / float(ramp_denom)
            )
            lefts_a = lefts_a * scales[:, None]
            rights_a = rights_a * scales[:, None]
        toks = [z_np[i] for i in range(n_exec)]
        lefts = [lefts_a[i] for i in range(n_exec)]
        rights = [rights_a[i] for i in range(n_exec)]
        hands = [h_np[i] for i in range(n_exec)]
        return toks, lefts, rights, hands

    with torch.no_grad():
        round_n = 0
        total_tx = 0
        loop_pause_s = float(os.environ.get("CL830_LOOP_PAUSE_S", "0"))
        while True:
            # Tape/cache: each episode round clears chunk + restarts t=0.
            # Live continuous: never enter this path (single unbounded inner loop).
            if loop_forever and round_n > 0 and not live_continuous:
                prev_z = None
                prev_hand = None
                chunk_tokens = []
                chunk_left = []
                chunk_right = []
                chunk_hand_wbc = []
                chunk_i = 0
                hold.clear()
                if config.auto_deploy_start:
                    _send_deploy_start(pub)
                if loop_pause_s > 0:
                    print(
                        f"[chunk_cl] loop round {round_n} pause {loop_pause_s:.1f}s",
                        flush=True,
                    )
                    time.sleep(loop_pause_s)
                print(f"[chunk_cl] loop round {round_n} restart t=0", flush=True)
            t = 0

            # Psi0 RealTimeChunkController only (sync block-replan removed).
            # Control never blocks on VLM/student; on new A, rebase t -= s.
            _psi0_rtc_rebase_selfcheck(
                h,
                int(play) if rtc_on else max(1, int(h) - prefetch_steps),
                max(1, int(rtc_d) if rtc_on else prefetch_steps),
            )
            s_min = int(play) if rtc_on else max(1, int(h) - prefetch_steps)
            d_init = max(1, int(rtc_d) if rtc_on else prefetch_steps)
            delay_q: deque[int] = deque([d_init], maxlen=6)
            stop_ev = threading.Event()
            ctrl_cv = threading.Condition()
            # plan_t == Psi0 t (increment-before-read). A_* == A_cur.
            plan_t = 0
            A_toks: list[np.ndarray] = []
            A_left: list[np.ndarray] = []
            A_right: list[np.ndarray] = []
            A_hand: list[np.ndarray] = []
            o_body: np.ndarray | None = None
            o_hand: torch.Tensor | None = None
            n_hold = 0
            infer_busy = False

            def _infer_loop() -> None:
                nonlocal plan_t, A_toks, A_left, A_right, A_hand, infer_busy
                while not stop_ev.is_set():
                    with ctrl_cv:
                        while not stop_ev.is_set() and (
                            not A_toks or plan_t < s_min or infer_busy
                        ):
                            ctrl_cv.wait(timeout=0.05)
                        if stop_ev.is_set():
                            break
                        s = int(plan_t)
                        d_lat = int(max(delay_q))
                        body_snap = None if o_body is None else o_body.copy()
                        hand_snap = None if o_hand is None else o_hand.detach().clone()
                        prev_z_chunk = prev_hand_chunk = None
                        if A_toks and A_hand:
                            prev_z_chunk = torch.as_tensor(
                                np.stack(A_toks, axis=0),
                                device=device,
                                dtype=torch.float32,
                            ).unsqueeze(0)
                            prev_hand_chunk = torch.as_tensor(
                                np.stack(A_hand, axis=0),
                                device=device,
                                dtype=torch.float32,
                            ).unsqueeze(0)
                        infer_busy = True
                    if body_snap is None or hand_snap is None:
                        with ctrl_cv:
                            infer_busy = False
                            ctrl_cv.notify_all()
                        continue
                    try:
                        with torch.no_grad():
                            toks, lefts, rights, hands = _replan_at(
                                # wall frame for AdaLN/logging; RTC uses s/d.
                                max(0, int(t)),
                                body_snap,
                                hand_snap,
                                rtc_shift=s,
                                full_horizon=True,
                                inference_delay=d_lat,
                                prev_z_chunk=prev_z_chunk,
                                prev_hand_chunk=prev_hand_chunk,
                            )
                        with ctrl_cv:
                            if toks:
                                A_toks, A_left, A_right, A_hand = (
                                    toks,
                                    lefts,
                                    rights,
                                    hands,
                                )
                                # Psi0: A_cur = A_new; t = t - s; Q ← t
                                plan_t = int(plan_t) - s
                                delay_q.append(max(0, int(plan_t)))
                                print(
                                    f"[chunk_cl] rtc_swap s={s} d={d_lat} "
                                    f"t'={plan_t} n={len(toks)} "
                                    f"token0={toks[0][0]:+.3f}",
                                    flush=True,
                                )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[chunk_cl] async infer failed: {exc!r}", flush=True)
                    finally:
                        with ctrl_cv:
                            infer_busy = False
                            ctrl_cv.notify_all()

            worker = threading.Thread(
                target=_infer_loop, name="chunk_cl_psi0_rtc", daemon=True
            )

            # Bootstrap: block first chunk (Psi0 warm A_first).
            while t < run_frames:
                t0 = time.monotonic()
                _apply_pending_skill(t)
                robot.poll()
                if not robot.ready or robot.last_msg is None:
                    time.sleep(0.01)
                    continue
                body = (
                    body_tape_np[t].astype(np.float32, copy=False)
                    if body_tape
                    else _body_q29(robot.last_msg).astype(np.float32)
                )
                hand_obs = _hand_obs_at(t, robot.last_msg)
                with torch.no_grad():
                    toks, lefts, rights, hands = _replan_at(
                        t, body, hand_obs, full_horizon=True, inference_delay=d_init
                    )
                if not toks:
                    rem = period - (time.monotonic() - t0)
                    if rem > 0:
                        time.sleep(rem)
                    continue
                with ctrl_cv:
                    A_toks, A_left, A_right, A_hand = toks, lefts, rights, hands
                    plan_t = 0
                    o_body, o_hand = body.copy(), hand_obs.detach().clone()
                    ctrl_cv.notify_all()
                print(
                    f"[chunk_cl] bootstrap n={len(toks)} token0={toks[0][0]:+.3f} "
                    f"s_min={s_min} d_init={d_init} psi0_rtc=1",
                    flush=True,
                )
                break

            worker.start()

            while t < run_frames:
                t0 = time.monotonic()
                if _apply_pending_skill(t):
                    with ctrl_cv:
                        A_toks, A_left, A_right, A_hand = [], [], [], []
                        plan_t = 0
                        infer_busy = False
                        ctrl_cv.notify_all()
                    # Refill A immediately (infer thread waits for non-empty A).
                    body0 = (
                        body_tape_np[t].astype(np.float32, copy=False)
                        if body_tape
                        else _body_q29(robot.last_msg).astype(np.float32)
                        if robot.last_msg is not None
                        else None
                    )
                    if body0 is not None:
                        hand0 = _hand_obs_at(t, robot.last_msg)
                        with torch.no_grad():
                            toks, lefts, rights, hands = _replan_at(
                                t,
                                body0,
                                hand0,
                                full_horizon=True,
                                inference_delay=d_init,
                            )
                        with ctrl_cv:
                            if toks:
                                A_toks, A_left, A_right, A_hand = (
                                    toks,
                                    lefts,
                                    rights,
                                    hands,
                                )
                                plan_t = 0
                                ctrl_cv.notify_all()
                robot.poll()
                if not robot.ready or robot.last_msg is None:
                    time.sleep(0.01)
                    continue
                body = (
                    body_tape_np[t].astype(np.float32, copy=False)
                    if body_tape
                    else _body_q29(robot.last_msg).astype(np.float32)
                )
                hand_obs = _hand_obs_at(t, robot.last_msg)

                with ctrl_cv:
                    o_body = body.copy()
                    o_hand = hand_obs.detach().clone()
                    plan_t += 1
                    ctrl_cv.notify_all()
                    idx = plan_t - 1
                    if A_toks and idx < len(A_toks):
                        tok = A_toks[idx]
                        l7 = A_left[idx]
                        r7 = A_right[idx]
                        h_wbc = A_hand[idx]
                        held = False
                    elif A_toks:
                        tok, l7, r7, h_wbc = (
                            A_toks[-1],
                            A_left[-1],
                            A_right[-1],
                            A_hand[-1],
                        )
                        held = True
                        n_hold += 1
                    else:
                        held = True
                        n_hold += 1
                        rem = period - (time.monotonic() - t0)
                        if rem > 0:
                            time.sleep(rem)
                        continue

                msg = pack_latent_action_message(
                    tok,
                    np.array([t], dtype=np.int64),
                    left_hand_joints=l7,
                    right_hand_joints=r7,
                )
                pub.send(msg)
                if hand_obs_mode == "commanded":
                    commanded_hand = torch.from_numpy(
                        h_wbc.astype(np.float32, copy=False)
                    ).to(device=device, dtype=torch.float32).reshape(1, -1)
                if t == 0 and round_n == 0:
                    print(
                        f"[chunk_cl] first output frame=1 token0={float(tok[0]):+.3f} "
                        f"dex_wbc_L7={np.round(h_wbc[:7], 3).tolist()} "
                        f"dex_zmq_L7={np.round(l7, 3).tolist()} "
                        f"dex_zmq_R7={np.round(r7, 3).tolist()} "
                        f"policy_order={int(_dex3_hand_policy_order_from_env())} "
                        f"async=1 psi0_rtc s_min={s_min} d_init={d_init}",
                        flush=True,
                    )
                    if release_flag:
                        Path(release_flag).parent.mkdir(parents=True, exist_ok=True)
                        Path(release_flag).touch()
                        print(
                            f"[chunk_cl] released RSI hold flag={release_flag} "
                            f"body0={float(body[0]):+.3f}",
                            flush=True,
                        )
                if t == 0 or (t + 1) % 50 == 0:
                    tag = "hold" if held else "tx"
                    print(
                        f"[chunk_cl] {tag} frame={t + 1} token0={float(tok[0]):+.3f} "
                        f"plan_t={plan_t} "
                        f"L7={np.round(l7, 2).tolist()} "
                        f"R7={np.round(r7, 2).tolist()}",
                        flush=True,
                    )
                t += 1
                total_tx += 1
                rem = period - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)

            stop_ev.set()
            with ctrl_cv:
                ctrl_cv.notify_all()
            worker.join(timeout=2.0)
            if n_hold > 0:
                print(f"[chunk_cl] async hold-last ticks={n_hold}", flush=True)

            round_n += 1
            if not loop_forever or live_continuous:
                break

    wall = time.monotonic() - t_start
    if not loop_forever:
        need = run_frames / float(config.fps)
        if wall < need:
            time.sleep(need - wall)
    robot.close()
    pub.close(linger=0)
    ctx.term()
    skill_stop.set()
    if live_continuous:
        print(f"[chunk_cl] stopped live_continuous tx={total_tx}", flush=True)
    elif loop_forever:
        print(f"[chunk_cl] stopped after {round_n} round(s) tx={total_tx}", flush=True)
    else:
        print(f"[chunk_cl] done frames={run_frames}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(ClConfig))
