#!/usr/bin/env python3
"""Open-loop student ẑ (+hand) dump with deploy RTC. No Isaac.

Body from tape FK; **hand obs is last commanded Revo2** (open at t=0), not
dataset tape — same contract as sim/robot closed loop. Freeze VLM → ChunkStudent
→ RTC blend → infer_qpos_traj_student.npz for
``build_gt_replay_tokens_from_infer_npz.py`` / MuJoCo MOTION_NPZ.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from phi0.inference.rtc import (
    blend_action_chunks_rtc,
    build_rtc_infer_prefix_kwargs,
    create_rtc_hard_mask,
    distill_rtc_deploy_cfg,
    rtc_play_horizon,
    shift_action_chunk_rtc,
)
from phi0.models.adaln_exec import (
    adaln_mode_of,
    exec_clocks_for_infer,
    has_video_mask_from_ref,
    infer_zero_exec_clock_enabled,
    normalize_adaln_mode,
    student_adaln_forward_kwargs,
)
from phi0.online.distill_norm import DistillNorm, build_distill_processor
from phi0.online.exec_time import ep_relative_steps
from phi0.online.isaac_loop import _attach_prompts_if_needed, episode_bounds_from_frame_index
from phi0.online.latent_ref import load_sonic_latent_reference
from phi0.online.lazy_ref import attach_has_video_to_ref
from phi0.online.phi0_student import (
    Phi0ChunkStudent,
    build_phi0_student,
    load_student_act_state_dict,
)
from phi0.online.ref_video_vlm import RefVideoFrameSource
from phi0.hand.hand_mode import student_hand_dim
from phi0.online.student_obs import build_student_proprio41_hist, distill_obs_hist_len
from phi0.online.vlm_hold_distill import DistillVlmHold, resolve_vlm_hold_period


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ref-root", type=Path, required=True)
    p.add_argument("--ref-start", type=int, default=0)
    p.add_argument("--max-frames", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = _parse()
    os.environ.setdefault("USE_RTC", "1")
    os.environ.setdefault("PHI0_CHUNK_EXEC", "open_loop")
    os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
    # Match vision_dl train (run_online_vlm_mix_distill.sh).
    os.environ.setdefault("PHI0_ZERO_PROPRIO_LEFT_THUMB_AUX", "0")
    # H=16: default d=6,s=15 violates d<=s<=H-d; keep d=6, s=10.
    os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
    os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "10")
    device = torch.device(args.device)
    ref_root = str(args.ref_root.resolve())
    ref = load_sonic_latent_reference(
        ref_root,
        max_frames=int(args.max_frames),
        start=int(args.ref_start),
        require_rsi=True,
        require_smpl=False,
    )
    _attach_prompts_if_needed(ref, ref_root)
    attach_has_video_to_ref(ref, ref_root)
    t_len = int(len(ref))
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    _pml = payload.get("prompt_max_length")
    if _pml is not None:
        os.environ.setdefault("PHI0_PROMPT_MAX_LENGTH", str(int(_pml)))
    os.environ.setdefault("PHI0_PROMPT_MAX_LENGTH", "256")
    h = int(payload["horizon"])
    if int(args.horizon) != h:
        raise SystemExit(f"horizon CLI {args.horizon} != ckpt {h}")
    hist_len = int(payload.get("history_len") or distill_obs_hist_len())
    if hist_len != 1:
        raise SystemExit(f"this dump assumes K=1, ckpt history_len={hist_len}")
    adaln = normalize_adaln_mode(str(payload.get("adaln_mode") or "progress_only"))
    student = Phi0ChunkStudent(
        build_phi0_student(
            device=str(device),
            use_vlm=True,
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

    stats_p = args.ckpt.parent / "action_stats.json"
    if not stats_p.is_file():
        raise SystemExit(f"missing {stats_p}")
    proc, st = build_distill_processor(
        stats_p, phi0_full_v3_root=ref_root, require_z_stats=False
    )
    dnorm = DistillNorm(proc, st)
    dnorm.attach_to_student(student)

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(rtc.get("enabled"))
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else h
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    play = rtc_play_horizon(rtc, h) if rtc_on else h
    rtc_prefix_cond = bool(rtc_on)
    rtc_mask = (
        create_rtc_hard_mask(h, rtc_d, rtc_s, device=device) if rtc_on else None
    )
    p0 = "?"
    try:
        if getattr(ref, "prompts", None) is not None:
            p0 = str(ref.prompts[0])
        elif hasattr(ref, "prompt_batch"):
            p0 = str(ref.prompt_batch(np.asarray([0], dtype=np.int64))[0])
    except Exception:
        pass
    zero_clk = infer_zero_exec_clock_enabled()
    print(
        f"[offline_rtc] T={t_len} H={h} rtc={int(rtc_on)} d={rtc_d} s={rtc_s} "
        f"prefix_cond={int(rtc_prefix_cond)} play={play} "
        f"adaln={adaln} zero_exec_clock={int(zero_clk)} prompt0={p0[:80]}",
        flush=True,
    )

    vids = Path(ref_root) / "videos"
    has_mp4 = vids.is_dir() and any(vids.rglob("*.mp4"))
    if not has_mp4:
        ref.has_video = np.zeros(t_len, dtype=np.bool_)
        video_src = None
        print("[offline_rtc] no mp4 — text-only VLM", flush=True)
    else:
        video_src = RefVideoFrameSource(ref_root, fps=50.0)
    hold = DistillVlmHold(1, period=resolve_vlm_hold_period(default=1))
    fi = getattr(ref, "frame_index", None)
    if fi is None:
        ep_starts = torch.zeros(1, device=device, dtype=torch.long)
    else:
        ep_starts_np, _ = episode_bounds_from_frame_index(
            np.asarray(fi), horizon=1, n_ds=max(1, t_len - h + 1)
        )
        ep_starts = torch.as_tensor(ep_starts_np, device=device, dtype=torch.long)

    body = np.asarray(ref.fk_dof29, dtype=np.float32)
    # Closed-loop hand: start open (sim default). No parquet on the robot.
    hand_live = np.zeros((1, int(student_hand_dim())), dtype=np.float32)
    # fk_dof29 == unified[367:396] on 820 release_unified (skill2 A/B).
    z_log: list[np.ndarray] = []
    hand_log: list[np.ndarray] = []
    prev_z: torch.Tensor | None = None
    prev_hand: torch.Tensor | None = None
    t = 0
    t_max = t_len - 1
    with torch.no_grad():
        while t <= t_max:
            ts = torch.tensor([min(t, t_max)], device=device, dtype=torch.long)
            ti = int(ts[0].item())
            obs_hist = build_student_proprio41_hist(
                body[ti : ti + 1], hand_live, k=hist_len
            ).to(device=device)
            steps_rel = ep_relative_steps(ts, ep_starts=ep_starts)
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
                batch_size=int(ts.shape[0]),
                device=device,
            )
            rtc_kw = build_rtc_infer_prefix_kwargs(
                prev_z=prev_z,
                prev_hand=prev_hand,
                inference_delay=rtc_d,
                execution_horizon=rtc_s,
                batch_size=int(ts.shape[0]),
                device=device,
                norm_z=dnorm.norm_z,
                norm_hand=dnorm.norm_hand,
                enabled=rtc_prefix_cond,
            )
            z_n, _q, _s, hand_n = student(
                obs_hist,
                lang_ctx=lang_ctx,
                lang_mask=lang_mask,
                **adaln_kw,
                **rtc_kw,
            )
            z_pred = dnorm.denorm_z(z_n)
            hand_pred = dnorm.denorm_hand(hand_n)
            if rtc_on and rtc_mask is not None:
                if prev_z is not None:
                    prev = shift_action_chunk_rtc(prev_z, int(rtc_s))
                    z_pred = blend_action_chunks_rtc(z_pred, prev, rtc_mask)
                if prev_hand is not None:
                    prev_h = shift_action_chunk_rtc(prev_hand, int(rtc_s))
                    hand_pred = blend_action_chunks_rtc(hand_pred, prev_h, rtc_mask)
                prev_z = z_pred.detach()
                prev_hand = hand_pred.detach()
            n_exec = min(int(play), t_max - t + 1)
            z_np = z_pred[0, :n_exec].float().cpu().numpy()
            h_np = hand_pred[0, :n_exec].float().cpu().numpy()
            for i in range(n_exec):
                z_log.append(z_np[i])
                hand_log.append(h_np[i])
            hand_live = h_np[-1:].astype(np.float32, copy=True)
            t += n_exec
            if t % 50 == 0 or t > t_max:
                print(f"[offline_rtc] t={min(t, t_len)}/{t_len}", flush=True)

    z = np.stack(z_log, axis=0)[:, None, :]
    hd = np.stack(hand_log, axis=0)[:, None, :]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        z_pred=z.astype(np.float32),
        hand_pred=hd.astype(np.float32),
        rtc_enabled=np.array([int(rtc_on)], dtype=np.int32),
        rtc_d=np.array([rtc_d], dtype=np.int32),
        rtc_s=np.array([rtc_s], dtype=np.int32),
    )
    print(
        f"[offline_rtc] wrote {args.out} T={z.shape[0]} rtc={int(rtc_on)} d={rtc_d} s={rtc_s}",
        flush=True,
    )


if __name__ == "__main__":
    main()
