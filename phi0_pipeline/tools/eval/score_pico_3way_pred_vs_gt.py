#!/usr/bin/env python3
"""Open-loop z/hand vs GT for today's 3 pico ckpts on one unified ep."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PHI0_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHI0_ROOT / "src"))

os.environ.setdefault("PHI0_HAND_MODE", "dex3")
os.environ.setdefault("PHI0_NEWTON_REVO2", "0")
os.environ.setdefault("USE_RTC", "1")
os.environ.setdefault("PHI0_CHUNK_EXEC", "open_loop")
os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "26")
os.environ.setdefault("PHI0_VLM_ATTN", "flash_attention_2")
os.environ.setdefault("PHI0_ALLOW_SDPA_FALLBACK", "0")
os.environ.setdefault("PHI0_PROMPT_MAX_LENGTH", "256")
os.environ.setdefault("PHI0_VLM_ENCODE_MIN_BATCH", "2")
os.environ.setdefault("USE_VLM", "1")

from phi0.hand.hand_mode import student_hand_dim
from phi0.inference.rtc import (
    blend_action_chunks_rtc,
    create_rtc_hard_mask,
    distill_rtc_deploy_cfg,
    rtc_play_horizon,
    shift_action_chunk_rtc,
)
from phi0.models.adaln_exec import (
    adaln_mode_of,
    exec_clocks_for_infer,
    has_video_mask_from_ref,
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
from phi0.online.student_obs import build_student_proprio41_hist, distill_obs_hist_len
from phi0.deploy.robot_proprio import deploy_hand7_to_wbc
from phi0.online.vlm_hold_distill import DistillVlmHold, resolve_vlm_hold_period


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _load_raw_dex3_hands14(
    ref_root: Path, *, ep: int = 0, max_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load WBC-order gripper14 from raw ``action/observation.dex3.*.position``.

    0828 pack left unified[346:360]=0 because ``resolve_hands`` only saw zero
    teleop/state hands; raw dex3 columns still hold GT (actuator order).
    """
    import pyarrow.parquet as pq

    meta_p = ref_root / "meta.json"
    if not meta_p.is_file():
        raise FileNotFoundError(f"missing {meta_p}")
    meta = json.loads(meta_p.read_text())
    sources = meta.get("sources") or []
    if not sources:
        raise RuntimeError(f"no sources in {meta_p}")
    src = next(s for s in sources if int(s["out_episode_index"]) == int(ep))
    raw_root = Path(meta.get("raw_root") or "")
    raw = (
        raw_root
        / str(src["session"])
        / "data"
        / "chunk-000"
        / f"episode_{int(src['source_episode_index']):06d}.parquet"
    )
    if not raw.is_file():
        raise FileNotFoundError(raw)
    cols = [
        "action.dex3.left.position",
        "action.dex3.right.position",
        "observation.dex3.left.position",
        "observation.dex3.right.position",
    ]
    table = pq.read_table(raw, columns=cols)
    al = np.stack(table.column(0).to_pylist()).astype(np.float32)
    ar = np.stack(table.column(1).to_pylist()).astype(np.float32)
    ol = np.stack(table.column(2).to_pylist()).astype(np.float32)
    orr = np.stack(table.column(3).to_pylist()).astype(np.float32)
    if max_frames is not None:
        al, ar, ol, orr = al[:max_frames], ar[:max_frames], ol[:max_frames], orr[:max_frames]

    def _to_wbc14(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.stack([deploy_hand7_to_wbc(x) for x in left], axis=0),
                np.stack([deploy_hand7_to_wbc(x) for x in right], axis=0),
            ],
            axis=1,
        ).astype(np.float32)

    hand_cmd = _to_wbc14(al, ar)
    hand_meas = _to_wbc14(ol, orr)
    return hand_cmd, hand_meas, str(raw)


def _run_one(
    *,
    name: str,
    ckpt: Path,
    ref_root: Path,
    max_frames: int,
    hand_mode: str,  # measured | commanded | zero
    device: torch.device,
    ep: int = 0,
) -> dict:
    os.environ["PHI0_ZERO_PROPRIO_HAND"] = "1" if hand_mode == "zero" else "0"
    ref = load_sonic_latent_reference(
        str(ref_root), max_frames=max_frames, start=0, require_rsi=True, require_smpl=False
    )
    _attach_prompts_if_needed(ref, str(ref_root))
    attach_has_video_to_ref(ref, str(ref_root))
    t_len = int(len(ref))
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    h = int(payload["horizon"])
    hist_len = int(payload.get("history_len") or distill_obs_hist_len())
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
    proc, st = build_distill_processor(
        ckpt.parent / "action_stats.json", phi0_full_v3_root=str(ref_root), require_z_stats=False
    )
    dnorm = DistillNorm(proc, st)
    dnorm.attach_to_student(student)

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(rtc.get("enabled"))
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else h
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    play = rtc_play_horizon(rtc, h) if rtc_on else h
    rtc_mask = (
        create_rtc_hard_mask(h, rtc_d, rtc_s, device=device) if rtc_on else None
    )

    video_src = RefVideoFrameSource(str(ref_root), fps=50.0)
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
    hd = int(student_hand_dim())
    # Prefer raw action/observation.dex3 (WBC-converted). Unified pack may be all-zero
    # when teleop/state hands were empty (0828).
    hand_cmd, hand_meas, raw_path = _load_raw_dex3_hands14(
        ref_root, ep=ep, max_frames=t_len
    )
    if hand_cmd.shape[0] < t_len or hand_cmd.shape[1] != hd:
        raise RuntimeError(
            f"raw dex3 shape {hand_cmd.shape} vs T={t_len} hd={hd} from {raw_path}"
        )
    hand_cmd = hand_cmd[:t_len]
    hand_meas = hand_meas[:t_len]
    print(
        f"[score] hand_gt from raw dex3 {raw_path} "
        f"cmd_abs={float(np.mean(np.abs(hand_cmd))):.4f} "
        f"meas_abs={float(np.mean(np.abs(hand_meas))):.4f}",
        flush=True,
    )
    if hand_mode == "measured":
        hand_obs_seq = hand_meas
    elif hand_mode == "commanded":
        hand_obs_seq = hand_cmd
    else:
        hand_obs_seq = np.zeros_like(hand_cmd)

    z_gt = np.asarray(ref.z_ref, dtype=np.float32)[:t_len]
    hand_gt = hand_cmd[:t_len]

    z_log, h_log = [], []
    prev_z = prev_hand = None
    t = 0
    t_max = t_len - 1
    with torch.no_grad():
        while t <= t_max:
            ts = torch.tensor([min(t, t_max)], device=device, dtype=torch.long)
            ti = int(ts[0].item())
            if hand_mode == "commanded" and t > 0 and h_log:
                # match CL: last commanded = last hand_pred
                hand_live = h_log[-1][None].astype(np.float32)
            else:
                hand_live = hand_obs_seq[ti : ti + 1]
            if os.environ.get("PHI0_ZERO_PROPRIO_HAND", "0") in ("1", "true", "True"):
                hand_live = np.zeros_like(hand_live)
            obs_hist = build_student_proprio41_hist(body[ti : ti + 1], hand_live, k=hist_len).to(
                device=device
            )
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
                batch_size=1,
                device=device,
            )
            z_n, _q, _s, hand_n = student(
                obs_hist, lang_ctx=lang_ctx, lang_mask=lang_mask, **adaln_kw
            )
            z_pred = dnorm.denorm_z(z_n)
            hand_pred = dnorm.denorm_hand(hand_n)
            if rtc_on and rtc_mask is not None:
                if prev_z is not None:
                    z_pred = blend_action_chunks_rtc(
                        z_pred, shift_action_chunk_rtc(prev_z, int(rtc_s)), rtc_mask
                    )
                if prev_hand is not None:
                    hand_pred = blend_action_chunks_rtc(
                        hand_pred, shift_action_chunk_rtc(prev_hand, int(rtc_s)), rtc_mask
                    )
                prev_z, prev_hand = z_pred.detach(), hand_pred.detach()
            n_exec = min(int(play), t_max - t + 1)
            z_np = z_pred[0, :n_exec].float().cpu().numpy()
            h_np = hand_pred[0, :n_exec].float().cpu().numpy()
            for i in range(n_exec):
                z_log.append(z_np[i])
                h_log.append(h_np[i])
            t += n_exec

    z_p = np.stack(z_log, 0)[:t_len]
    h_p = np.stack(h_log, 0)[:t_len]
    n = min(len(z_p), len(z_gt))
    z_p, z_gt, h_p, hand_gt = z_p[:n], z_gt[:n], h_p[:n], hand_gt[:n]
    hand_gt_abs = float(np.mean(np.abs(hand_gt)))
    out = {
        "name": name,
        "hand_mode": hand_mode,
        "hand_gt_source": "raw_action.dex3_wbc",
        "T": n,
        "z_mae": _mae(z_p, z_gt),
        "z_mse": _mse(z_p, z_gt),
        "z_mae_t0": _mae(z_p[:, :1], z_gt[:, :1]),
        "hand_gt_abs_mean": hand_gt_abs,
        "hand_mae": _mae(h_p, hand_gt) if hand_gt_abs > 1e-6 else None,
        "hand_mse": _mse(h_p, hand_gt) if hand_gt_abs > 1e-6 else None,
        "hand_pred_abs_mean": float(np.mean(np.abs(h_p))),
        "z_pred_abs_mean": float(np.mean(np.abs(z_p))),
        "z_gt_abs_mean": float(np.mean(np.abs(z_gt))),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    # free VLM
    del student, hold, video_src
    torch.cuda.empty_cache()
    return out


def main() -> None:
    device = torch.device("cuda:0")
    ref_root = Path(
        os.environ.get(
            "REF_ROOT",
            "/mnt/data2/wpy/workspace/local_nvme/datasets/830/0828_skill2_unified",
        )
    )
    max_frames = int(os.environ.get("MAX_FRAMES", "634"))
    runs = [
        (
            "measured",
            Path(
                "/mnt/data3/wpy/830demo_skill2_pico_pick_toy_vlm_cache_h32_b32_ddp6_e10_20260903_205811/phi0_student_last.pt"
            ),
            "measured",
        ),
        (
            "handcmd",
            Path(
                "/mnt/data3/wpy/830demo_skill2_pico_pick_toy_handcmd_vlm_cache_h32_b32_ddp6_e10_20260904_010502/phi0_student_last.pt"
            ),
            "commanded",
        ),
        (
            "nohandobs",
            Path(
                "/mnt/data3/wpy/830demo_skill2_pico_pick_toy_nohandobs_vlm_cache_h32_b32_ddp6_e10_20260904_031525/phi0_student_last.pt"
            ),
            "zero",
        ),
    ]
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            "/mnt/data2/wpy/workspace/Phi_0_wpy/experiments/830_pico_3way_0828_front_20260904_125523",
        )
    )
    results = []
    for name, ckpt, hm in runs:
        print(f"[score] === {name} ===", flush=True)
        results.append(
            _run_one(
                name=name,
                ckpt=ckpt,
                ref_root=ref_root,
                max_frames=max_frames,
                hand_mode=hm,
                device=device,
                ep=int(os.environ.get("EP", "0")),
            )
        )
    out_path = out_dir / os.environ.get("OUT_NAME", "pred_vs_gt_0828_ep0_raw_dex3.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"[score] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
