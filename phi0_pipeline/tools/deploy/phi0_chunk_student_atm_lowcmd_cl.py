#!/usr/bin/env python3
"""ChunkStudent / GT-z closed-loop → ONNX ATM decode → DDS LowCmd (Newton-style PD).

No C++ TRT / latent ZMQ. Reads ``rt/lowstate``, gets ``z`` from student **or**
dataset ``z_ref`` (``PHI0_CL_Z_SOURCE=disk``), decodes ``z→q29`` with sonic_v1_1
ONNX, publishes ``rt/lowcmd`` (+ Dex3 hand).

Hand proprio default: previous commanded ``hand_pred`` (Newton parity).
``PHI0_CL_HAND_OBS=tape`` uses dataset ``hand_ref``. Body default: live LowState;
``PHI0_CL_PROPRIO_BODY=tape`` uses ``fk_dof29``.
GT open-loop: ``PHI0_CL_Z_SOURCE=disk`` + ``PHI0_CL_HAND_OBS=tape`` (+ optional
``USE_RTC=0`` for 1-token/step).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
for root in (_GR00T, _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


from phi0.deploy.dex3_gripper import split_gripper14_wbc_to_deploy  # noqa: E402
from phi0.deploy.robot_proprio import RobotProprioSource, _body_q29  # noqa: E402
from phi0.deploy.lowcmd_publisher import LowCmdPublisher, LowStateReader  # noqa: E402
from phi0.deploy.sonic_decoder_obs import (  # noqa: E402
    DEFAULT_DECODER_ONNX,
    DecoderHistBuffer,
    OnnxAtmDecoder,
    decoder_action_to_q_mj,
)
from phi0.inference.rtc import (  # noqa: E402
    blend_action_chunks_rtc,
    create_rtc_hard_mask,
    distill_rtc_deploy_cfg,
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
from phi0.online.vlm_hold_distill import DistillVlmHold, resolve_vlm_hold_period  # noqa: E402
from phi0.online.vision_dl_distill import (  # noqa: E402
    encode_dual_vlm_from_frame_cache,
    use_vlm_frame_latent_cache_from_env,
)


@dataclass
class ClConfig:
    ckpt: Path | None = None
    ref_root: Path = Path(".")
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
    out_npz: Path | None = None
    decoder_onnx: Path | None = None
    domain_id: int = 0
    dds_interface: str = ""
    publish_hands: bool = True
    horizon: int = 32  # used when PHI0_CL_Z_SOURCE=disk and no ckpt


def _cl_z_source_disk() -> bool:
    """``PHI0_CL_Z_SOURCE=disk|gt|tape|z_ref`` → play ``ref.z_ref`` (no student)."""
    return os.environ.get("PHI0_CL_Z_SOURCE", "student").strip().lower() in (
        "disk",
        "gt",
        "tape",
        "z_ref",
        "teacher",
    )


def _wait_flag(path: Path, *, label: str, timeout_s: float = 240.0) -> None:
    print(f"[atm_pd_cl] waiting for {label} {path}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{label} flag not found: {path}")



def _hand_zmq7(hand14_wbc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l7, r7 = split_gripper14_wbc_to_deploy(np.asarray(hand14_wbc, dtype=np.float32))
    return l7, r7


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


def _cl_hand_obs_tape() -> bool:
    """``PHI0_CL_HAND_OBS=tape`` → dataset hand_ref. Default: commanded hand_pred."""
    return os.environ.get("PHI0_CL_HAND_OBS", "commanded").strip().lower() in (
        "tape", "1", "true", "yes",
    )


def main(config: ClConfig) -> None:
    z_disk = _cl_z_source_disk()
    # GT open-loop: 1 token/step unless user explicitly enables RTC.
    if z_disk:
        os.environ.setdefault("USE_RTC", "0")
    else:
        os.environ.setdefault("USE_RTC", "1")
    os.environ.setdefault("PHI0_CHUNK_EXEC", "open_loop")
    os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
    os.environ.setdefault("PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX", "0")
    os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
    os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "26")

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
    attach_has_video_to_ref(ref, ref_root)
    t_len = int(len(ref))
    max_frames = int(config.max_frames) if int(config.max_frames) > 0 else t_len
    max_frames = min(max_frames, t_len)

    z_tape = getattr(ref, "z_ref", None)
    if z_disk:
        if z_tape is None:
            raise RuntimeError("PHI0_CL_Z_SOURCE=disk requires ref.z_ref")
        z_tape = np.asarray(z_tape, dtype=np.float32)
        if z_tape.ndim != 2 or int(z_tape.shape[-1]) != 64:
            raise RuntimeError(f"ref.z_ref shape {z_tape.shape} want [T,64]")
        hand_tape = getattr(ref, "hand_ref", None)
        if hand_tape is None:
            raise RuntimeError("PHI0_CL_Z_SOURCE=disk requires ref.hand_ref for hand cmds")
        hand_tape = np.asarray(hand_tape, dtype=np.float32)
        h = max(1, int(config.horizon))
        hist_len = 1
        student = None
        dnorm = None
        frame_cache = None
        video_src = None
        hold = None
        adaln = "none"
    else:
        if config.ckpt is None or not Path(config.ckpt).is_file():
            raise RuntimeError("--ckpt required when PHI0_CL_Z_SOURCE=student")
        payload = torch.load(config.ckpt, map_location=device, weights_only=False)
        if payload.get("prompt_max_length") is not None:
            os.environ.setdefault(
                "PHI0_PROMPT_MAX_LENGTH", str(int(payload["prompt_max_length"]))
            )
        h = int(payload["horizon"])
        hist_len = int(payload.get("history_len") or distill_obs_hist_len())
        adaln = normalize_adaln_mode(str(payload.get("adaln_mode") or "progress_only"))
        frame_cache_on = use_vlm_frame_latent_cache_from_env()
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
        stats_p = Path(config.ckpt).parent / "action_stats.json"
        proc, st = build_distill_processor(
            stats_p, phi0_full_v3_root=ref_root, require_z_stats=False
        )
        dnorm = DistillNorm(proc, st)
        dnorm.attach_to_student(student)
        vids = Path(ref_root) / "videos"
        video_src = (
            None
            if frame_cache is not None
            else (
                RefVideoFrameSource(ref_root, fps=50.0)
                if vids.is_dir() and any(vids.rglob("*.mp4"))
                else None
            )
        )
        hold = DistillVlmHold(1, period=resolve_vlm_hold_period(default=1))
        hand_tape = None
        z_tape = None

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(rtc.get("enabled"))
    play = int(rtc_play_horizon(rtc, h) if rtc_on else h)
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else 0
    rtc_mask = (
        create_rtc_hard_mask(h, rtc_d, rtc_s, device=device) if rtc_on else None
    )
    fi = getattr(ref, "frame_index", None)
    if fi is None:
        ep_starts = torch.zeros(1, device=device, dtype=torch.long)
    else:
        ep_starts_np, _ = episode_bounds_from_frame_index(
            np.asarray(fi), horizon=1, n_ds=max(1, t_len - h + 1)
        )
        ep_starts = torch.as_tensor(ep_starts_np, device=device, dtype=torch.long)

    body_tape = bool(_cl_proprio_body_tape())
    hand_tape_mode = bool(_cl_hand_obs_tape())
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
    if hand_tape_mode and getattr(ref, "hand_ref", None) is None:
        raise RuntimeError("PHI0_CL_HAND_OBS=tape requires ref.hand_ref")

    iface = (config.dds_interface or "").strip() or None
    lowstate = LowStateReader(domain_id=int(config.domain_id), interface=iface)
    lowcmd = LowCmdPublisher(
        domain_id=int(config.domain_id),
        interface=iface,
        init_factory=False,
        publish_hands=bool(config.publish_hands),
    )
    decoder = OnnxAtmDecoder(config.decoder_onnx or DEFAULT_DECODER_ONNX)
    hist = DecoderHistBuffer()

    print(
        f"[atm_pd_cl] decoder={decoder.path} T={max_frames} H={h} "
        f"rtc={int(rtc_on)} play={play} adaln={adaln} "
        f"z={'disk' if z_disk else 'student'} "
        f"vlm={'frame_cache' if frame_cache is not None else ('none' if z_disk else 'live')} "
        f"body={'tape' if body_tape else 'live'} "
        f"hand={'tape' if (hand_tape_mode or z_disk) else 'commanded'}",
        flush=True,
    )

    if config.arm_flag:
        _wait_flag(Path(config.arm_flag), label="arm")
        time.sleep(config.start_delay_s)
    if config.ready_flag:
        _wait_flag(Path(config.ready_flag), label="ready")
        time.sleep(config.start_delay_s)

    lowstate.wait_ready(timeout_s=float(config.wait_g1_debug_s))
    snap0 = lowstate.snapshot()
    assert snap0 is not None
    lowcmd.set_mode_machine(int(snap0["mode_machine"]))
    hist.prime_from_state(
        ang_vel=snap0["ang_vel"],
        q_mj=snap0["q_mj"],
        dq_mj=snap0["dq_mj"],
        quat_wxyz=snap0["quat_wxyz"],
    )
    # First LowCmd: MuJoCo KEEP_POSE restores RSI; then re-prime from live state.
    lowcmd.publish(np.asarray(snap0["q_mj"], dtype=np.float32))
    time.sleep(max(0.25, float(config.start_delay_s)))
    snap1 = lowstate.snapshot()
    if snap1 is not None:
        hist.prime_from_state(
            ang_vel=snap1["ang_vel"],
            q_mj=snap1["q_mj"],
            dq_mj=snap1["dq_mj"],
            quat_wxyz=snap1["quat_wxyz"],
        )
        print(
            f"[atm_pd_cl] reprime after RSI restore "
            f"q0={float(snap1['q_mj'][0]):+.3f} "
            f"qw={float(snap1['quat_wxyz'][0]):+.3f}",
            flush=True,
        )

    prev_z: torch.Tensor | None = None
    commanded_hand = torch.zeros(1, 14, device=device, dtype=torch.float32)
    z_log: list[np.ndarray] = []
    q_log: list[np.ndarray] = []
    hand_log: list[np.ndarray] = []
    prev_hand: torch.Tensor | None = None
    chunk_tokens: list[np.ndarray] = []
    chunk_left: list[np.ndarray] = []
    chunk_right: list[np.ndarray] = []
    chunk_hand_wbc: list[np.ndarray] = []
    chunk_i = 0
    period = 1.0 / float(config.fps)
    ramp_denom = max(int(config.hand_ramp_frames), 1)
    t_start = time.monotonic()

    with torch.no_grad():
        t = 0
        while t < max_frames:
            t0 = time.monotonic()
            snap = lowstate.snapshot()
            if snap is None:
                time.sleep(0.01)
                continue
            lowcmd.set_mode_machine(int(snap["mode_machine"]))
            body = (
                body_tape_np[t].astype(np.float32, copy=False)
                if body_tape
                else np.asarray(snap["q_mj"], dtype=np.float32)
            )
            ts = torch.tensor([t], device=device, dtype=torch.long)
            hand_obs = (
                _tape_hand_obs(ref, ts, device=device)
                if hand_tape_mode
                else commanded_hand
            )
            obs_hist = build_student_proprio41_hist(
                body[None], hand_obs, k=hist_len
            ).to(device=device)

            if chunk_i >= len(chunk_tokens):
                n_exec = min(int(play), max_frames - t)
                if z_disk:
                    assert z_tape is not None and hand_tape is not None
                    z_np = z_tape[t : t + n_exec]
                    h_np = hand_tape[t : t + n_exec]
                else:
                    assert student is not None and dnorm is not None
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
                        assert hold is not None
                        lang_ctx, lang_mask = hold.get_or_refresh(
                            student=student,
                            ref=ref,
                            video_src=video_src,
                            ts_abs=ts,
                            steps=steps_rel,
                            device=device,
                        )
                    steps_clk, _, _ = exec_clocks_for_infer(steps_rel)
                    adaln_kw = student_adaln_forward_kwargs(
                        exec_steps=steps_clk,
                        has_video=has_video_mask_from_ref(ref, ts, device=device),
                        adaln_mode=adaln_mode_of(student),
                        batch_size=1,
                        device=device,
                    )
                    z_n, _, _, hand_n = student(
                        obs_hist, lang_ctx=lang_ctx, lang_mask=lang_mask, **adaln_kw
                    )
                    z_pred = dnorm.denorm_z(z_n)
                    hand_pred = dnorm.denorm_hand(hand_n)
                    if rtc_on and rtc_mask is not None:
                        if prev_z is not None:
                            prev = shift_action_chunk_rtc(prev_z, int(rtc_s))
                            z_pred = blend_action_chunks_rtc(z_pred, prev, rtc_mask)
                        if prev_hand is not None:
                            prev_h = shift_action_chunk_rtc(prev_hand, int(rtc_s))
                            hand_pred = blend_action_chunks_rtc(
                                hand_pred, prev_h, rtc_mask
                            )
                        prev_z = z_pred.detach()
                        prev_hand = hand_pred.detach()
                    z_np = z_pred[0, :n_exec].float().cpu().numpy()
                    h_np = hand_pred[0, :n_exec].float().cpu().numpy()
                chunk_tokens, chunk_left, chunk_right, chunk_hand_wbc = [], [], [], []
                for i in range(n_exec):
                    l7, r7 = _hand_zmq7(h_np[i])
                    scale = (
                        min(1.0, float(i) / ramp_denom)
                        if config.hand_ramp_frames > 0
                        else 1.0
                    )
                    chunk_tokens.append(z_np[i])
                    chunk_left.append(l7 * scale)
                    chunk_right.append(r7 * scale)
                    chunk_hand_wbc.append(h_np[i].astype(np.float32, copy=True))
                chunk_i = 0
                if t == 0 or (t + 1) % 50 == 0:
                    src = "tape" if body_tape else "live"
                    print(
                        f"[atm_pd_cl] replan t={t} body0={body[0]:+.3f}({src}) "
                        f"hand0={float(hand_obs[0, 0].item()):+.3f} "
                        f"token0={chunk_tokens[0][0]:+.3f}",
                        flush=True,
                    )

            tok = chunk_tokens[chunk_i]
            hand_wbc = chunk_hand_wbc[chunk_i]
            l7, r7 = _hand_zmq7(hand_wbc)
            commanded_hand = torch.as_tensor(
                hand_wbc, device=device, dtype=torch.float32
            ).view(1, -1)
            obs994 = hist.build_obs994(tok)
            act_il = decoder(obs994)
            q_mj = decoder_action_to_q_mj(act_il)
            q_mj = np.clip(q_mj, -4.0, 4.0)

            if not np.isfinite(q_mj).all():
                raise RuntimeError(f"non-finite q_mj at t={t}")
            lowcmd.publish(q_mj, left_hand7=l7, right_hand7=r7)
            hist.push(
                ang_vel=snap["ang_vel"],
                q_mj=snap["q_mj"],
                dq_mj=snap["dq_mj"],
                last_act_il=act_il,
                quat_wxyz=snap["quat_wxyz"],
            )
            z_log.append(np.asarray(tok, dtype=np.float32).copy())
            q_log.append(q_mj.copy())
            hand_log.append(np.asarray(hand_wbc, dtype=np.float32).copy())
            if t == 0:
                print(
                    f"[atm_pd_cl] first output frame=1 token0={float(tok[0]):+.3f}",
                    flush=True,
                )
            if t == 0 or (t + 1) % 50 == 0:
                print(
                    f"[atm_pd_cl] tx frame={t + 1} token0={float(tok[0]):+.3f}",
                    flush=True,
                )
            chunk_i += 1
            t += 1

            elapsed = time.monotonic() - t0
            rem = period - elapsed
            if rem > 0:
                time.sleep(rem)

    wall = time.monotonic() - t_start
    need = max_frames / float(config.fps)
    if wall < need:
        time.sleep(need - wall)
    if config.out_npz is not None:
        config.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.out_npz,
            z_pred=np.stack(z_log, axis=0) if z_log else np.zeros((0, 64), np.float32),
            q_cmd=np.stack(q_log, axis=0) if q_log else np.zeros((0, 29), np.float32),
            hand_pred=np.stack(hand_log, axis=0) if hand_log else np.zeros((0, 14), np.float32),
        )
        print(f"[atm_pd_cl] wrote {config.out_npz}", flush=True)
    print(f"[atm_pd_cl] done frames={max_frames}")


if __name__ == "__main__":
    main(tyro.cli(ClConfig))
