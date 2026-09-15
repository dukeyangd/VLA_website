#!/usr/bin/env python3
"""Skill3 place_basket ep0: GT vs 830mix student — hand closure + body ||z||."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

PHI0_ROOT = Path("/mnt/data2/wpy/workspace/Phi_0_wpy")
sys.path.insert(0, str(PHI0_ROOT / "src"))

os.environ.setdefault("PHI0_HAND_MODE", "dex3")
os.environ.setdefault("PHI0_NEWTON_REVO2", "0")
os.environ.setdefault("PHI0_CHUNK_EXEC", "open_loop")
os.environ.setdefault("PHI0_TRAIN_MODE", "online_vlm")
os.environ.setdefault("PHI0_VLM_ATTN", "flash_attention_2")
os.environ.setdefault("PHI0_ALLOW_SDPA_FALLBACK", "0")
os.environ.setdefault("PHI0_PROMPT_MAX_LENGTH", "256")
os.environ.setdefault("PHI0_ZERO_PROPRIO_HAND", "0")
os.environ.setdefault("PHI0_TRAIN_HAND_OBS", "commanded")
os.environ.setdefault("PHI0_HAND_PROPRIO_LAG", "1")
os.environ.setdefault("DEPLOY_POLICY_DIR", "sonic_v1_1")
os.environ.setdefault("PHI0_WORKSPACE", "/mnt/data2/wpy/workspace")
os.environ.setdefault("PHI0_USE_VLM_FRAME_LATENT_CACHE", "1")
os.environ.setdefault("PHI0_VLM_FRAME_CACHE_SKIP_VIDEO", "1")
os.environ.setdefault("USE_VLM", "0")
os.environ.setdefault("USE_RTC", "1")
os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "26")
os.environ.setdefault("PHI0_ADALN_ZERO_VISION", "1")

from phi0.deploy.dex3_gripper import gripper14_batch_to_zmq_lr  # noqa: E402
from phi0.deploy.robot_proprio import deploy_hand7_to_wbc  # noqa: E402
from phi0.inference.rtc import (  # noqa: E402
    blend_action_chunks_rtc,
    build_rtc_infer_prefix_kwargs,
    create_rtc_hard_mask,
    distill_rtc_deploy_cfg,
    rtc_play_horizon,
    shift_action_chunk_rtc,
)
from phi0.models.adaln_exec import (  # noqa: E402
    adaln_mode_of,
    exec_clocks_for_infer,
    has_video_mask_from_ref,
    normalize_adaln_mode,
    student_adaln_forward_kwargs,
)
from phi0.online.distill_norm import DistillNorm, build_distill_processor  # noqa: E402
from phi0.online.exec_time import ep_relative_steps  # noqa: E402
from phi0.online.isaac_loop import (  # noqa: E402
    _attach_prompts_if_needed,
    episode_bounds_from_frame_index,
)
from phi0.online.latent_ref import load_sonic_latent_reference  # noqa: E402
from phi0.online.lazy_ref import attach_has_video_to_ref  # noqa: E402
from phi0.online.phi0_student import (  # noqa: E402
    Phi0ChunkStudent,
    build_phi0_student,
    load_student_act_state_dict,
)
from phi0.online.student_obs import (  # noqa: E402
    build_student_proprio41_hist,
    distill_obs_hist_len,
)
from phi0.online.vision_dl_distill import encode_dual_vlm_from_frame_cache  # noqa: E402
from phi0.online.vlm_frame_latents import DualVlmFrameLatentCache  # noqa: E402

HZ = 50.0
REF = Path("/mnt/data3/wpy/datasets/830/830demo_skill3_place_basket_unified")
RAW = REF / "data/chunk-000/file-000.parquet"
CKPT = Path(
    "/mnt/data3/wpy/830mix_skill123_demo5_handcmd_lag1_vlm_cache_h32_b64_ddp8_e2_pv0.95_resume_20260907_035710/phi0_student_last.pt"
)
OUT = Path("/mnt/data2/wpy/workspace/Phi_0_wpy/logs")
PNG = OUT / "skill3_ep0_mix_vs_gt_hand_body_closure_20260907.png"
NPZ = OUT / "skill3_ep0_mix_vs_gt_hand_body_closure_20260907.npz"


def closure_from_zmq(left, right):
    return 0.5 * (np.mean(np.abs(left[:, 3:7]), 1) + np.mean(np.abs(right[:, 3:7]), 1))


def load_hand_gt_wbc14(raw: Path, t_len: int) -> np.ndarray:
    schema = set(pq.read_schema(raw).names)
    if "action.unified" in schema:
        U = np.stack(
            pq.read_table(raw, columns=["action.unified"]).column(0).to_pylist()
        ).astype(np.float32)[:t_len]
        return U[:, 346:360].copy()
    if "action.dex3.left.position" in schema:
        t = pq.read_table(
            raw, columns=["action.dex3.left.position", "action.dex3.right.position"]
        )
        al = np.stack(t.column(0).to_pylist()).astype(np.float32)[:t_len]
        ar = np.stack(t.column(1).to_pylist()).astype(np.float32)[:t_len]
        return np.concatenate(
            [
                np.stack([deploy_hand7_to_wbc(x) for x in al], 0),
                np.stack([deploy_hand7_to_wbc(x) for x in ar], 0),
            ],
            1,
        ).astype(np.float32)
    raise ValueError(f"no hand cols in {raw}")


def run_open_loop(ckpt: Path, device: torch.device):
    t_len = int(pq.read_metadata(RAW).num_rows)
    ref = load_sonic_latent_reference(
        str(REF), max_frames=t_len, start=0, require_rsi=True, require_smpl=False
    )
    _attach_prompts_if_needed(ref, str(REF))
    attach_has_video_to_ref(ref, str(REF))
    t_len = len(ref)
    z_gt = np.asarray(ref.z_ref, dtype=np.float32)[:t_len]
    ep_idx = np.asarray(ref.episode_index, dtype=np.int64)
    fr_idx = np.asarray(ref.frame_index, dtype=np.int64)
    body = np.asarray(ref.fk_dof29, dtype=np.float32)
    hand_gt = load_hand_gt_wbc14(RAW, t_len)

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    h = int(payload["horizon"])
    hist_len = int(payload.get("history_len") or distill_obs_hist_len())
    adaln = normalize_adaln_mode(str(payload.get("adaln_mode") or "progress_only"))
    student = Phi0ChunkStudent(
        build_phi0_student(
            device=str(device),
            use_vlm=False,
            use_lang_latent_cache=False,
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
        ckpt.parent / "action_stats.json",
        phi0_full_v3_root=str(REF),
        require_z_stats=False,
    )
    dnorm = DistillNorm(proc, st)
    dnorm.attach_to_student(student)

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(rtc.get("enabled"))
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else h
    play = int(rtc_play_horizon(rtc, h) if rtc_on else h)
    rtc_mask = create_rtc_hard_mask(h, rtc_d, rtc_s, device=device) if rtc_on else None
    print(
        f"[skill3_closure] H={h} play={play} rtc={int(rtc_on)} d={rtc_d} s={rtc_s} T={t_len}",
        flush=True,
    )

    fi = getattr(ref, "frame_index", None)
    if fi is None:
        ep_starts = torch.zeros(1, device=device, dtype=torch.long)
    else:
        ep_starts_np, _ = episode_bounds_from_frame_index(
            np.asarray(fi), horizon=1, n_ds=max(1, t_len - h + 1)
        )
        ep_starts = torch.as_tensor(ep_starts_np, device=device, dtype=torch.long)

    frame_cache = DualVlmFrameLatentCache.open(
        REF, dirname="vlm_frame_latents_qwen3vl_dual"
    )
    h_log: list[np.ndarray] = []
    z_log: list[np.ndarray] = []
    prev_z = prev_hand = None
    t, t_max = 0, t_len - 1
    with torch.no_grad():
        while t <= t_max:
            ts = torch.tensor([min(t, t_max)], device=device, dtype=torch.long)
            ti = int(ts[0].item())
            hand_live = (
                h_log[-1][None].astype(np.float32) if h_log else hand_gt[ti : ti + 1]
            )
            obs = build_student_proprio41_hist(
                body[ti : ti + 1], hand_live, k=hist_len
            ).to(device)
            steps_rel = ep_relative_steps(ts, ep_starts=ep_starts)
            lang_ctx, lang_mask = encode_dual_vlm_from_frame_cache(
                cache=frame_cache,
                episode_indices=[int(ep_idx[ti])],
                frame_indices=[int(fr_idx[ti])],
                device=device,
                dtype=torch.float32,
            )
            steps_clk, _, _ = exec_clocks_for_infer(steps_rel)
            adaln_kw = student_adaln_forward_kwargs(
                exec_steps=steps_clk,
                has_video=has_video_mask_from_ref(ref, ts, device=device),
                adaln_mode=adaln_mode_of(student),
                batch_size=1,
                device=device,
            )
            rtc_kw = build_rtc_infer_prefix_kwargs(
                prev_z=prev_z,
                prev_hand=prev_hand,
                inference_delay=rtc_d,
                execution_horizon=rtc_s,
                batch_size=1,
                device=device,
                norm_z=dnorm.norm_z,
                norm_hand=dnorm.norm_hand,
                enabled=rtc_on,
            )
            z_n, _q, _s, hand_n = student(
                obs, lang_ctx=lang_ctx, lang_mask=lang_mask, **adaln_kw, **rtc_kw
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
                        hand_pred,
                        shift_action_chunk_rtc(prev_hand, int(rtc_s)),
                        rtc_mask,
                    )
                prev_z = z_pred.detach()
                prev_hand = hand_pred.detach()
            n_exec = min(int(play), t_max - t + 1)
            z_np = z_pred[0, :n_exec].float().cpu().numpy()
            h_np = hand_pred[0, :n_exec].float().cpu().numpy()
            for i in range(n_exec):
                z_log.append(z_np[i])
                h_log.append(h_np[i])
            t += n_exec
            if t % 200 == 0 or t > t_max:
                print(f"  student t={min(t, t_len)}/{t_len}", flush=True)

    h_p = np.stack(h_log, 0)[:t_len]
    z_p = np.stack(z_log, 0)[:t_len]
    L, R = gripper14_batch_to_zmq_lr(h_p, policy_order=True)
    c_pred = closure_from_zmq(L, R)
    Lg, Rg = gripper14_batch_to_zmq_lr(hand_gt[:t_len], policy_order=True)
    c_gt = closure_from_zmq(Lg, Rg)
    return c_gt, c_pred, z_gt, z_p, play


def plot(c_gt, c_pred, z_gt, z_pred, play: int, png: Path):
    T = min(len(c_gt), len(c_pred), len(z_gt), len(z_pred))
    t = np.arange(T) / HZ
    zE = {"GT": np.linalg.norm(z_gt[:T], axis=1), "Mix": np.linalg.norm(z_pred[:T], axis=1)}
    hand = {"GT": c_gt[:T], "Mix": c_pred[:T]}
    hand_mae = float(np.mean(np.abs(hand["Mix"] - hand["GT"])))
    z_mae = float(np.mean(np.abs(z_pred[:T] - z_gt[:T])))

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), dpi=140)
    panels = [
        (axes[0], hand, None, f"hand closure  mae={hand_mae:.4f}", "closure"),
        (axes[1], zE, None, f"body sonic ||z||  mae={z_mae:.4f}", "||z||"),
        (axes[2], hand, (0, 8), "hand zoom 0–8s", "closure"),
        (axes[3], zE, (0, 8), "body ||z|| zoom 0–8s", "||z||"),
    ]
    colors = {"GT": "#9aa0a6", "Mix": "#2563eb"}
    for ax, series, xlim, title, ylab in panels:
        for name in ("GT", "Mix"):
            ax.plot(
                t,
                series[name],
                color=colors[name],
                lw=1.35 if name == "GT" else 1.1,
                label=name,
            )
        step = max(int(play), 1)
        for b in range(step, T, step):
            ax.axvline(b / HZ, color="#dadce0", lw=0.35, zorder=0)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        if xlim:
            ax.set_xlim(*xlim)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "skill3 place_basket ep0 — Mix B64 resume vs GT (frame_cache + RTC 6/26)",
        fontsize=11,
    )
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    fig.savefig(CKPT.parent / "skill3_ep0_mix_vs_gt_hand_body_closure.png")
    print(f"[skill3_closure] wrote {png}", flush=True)
    plt.close(fig)
    return hand_mae, z_mae


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert CKPT.is_file(), CKPT
    assert RAW.is_file(), RAW
    c_gt, c_pred, z_gt, z_pred, play = run_open_loop(CKPT, device)
    np.savez_compressed(NPZ, c_gt=c_gt, c_pred=c_pred, z_gt=z_gt, z_pred=z_pred, play=play)
    hand_mae, z_mae = plot(c_gt, c_pred, z_gt, z_pred, play, PNG)
    print(json.dumps({"hand_mae": hand_mae, "z_mae": z_mae, "T": int(len(c_pred)), "play": int(play)}, indent=2))


if __name__ == "__main__":
    main()
