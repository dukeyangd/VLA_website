#!/usr/bin/env python3
"""Compare baseline Instruct e2 vs RTC-prefix e2 open-loop hand/||z||.

Train-domain: mix pure session 2026-09-05-22-04-56 src_ep1 (frame cache).
OOD: 0828 session 2026-08-29-00-13-05 src_ep1 (live dual VLM).

Baseline eval matches prior report (USE_RTC=0, play=H).
RTC-prefix uses USE_RTC=1 + hard freeze + prefix cond (play=s).
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault("DEPLOY_POLICY_DIR", "sonic_v1_1")
os.environ.setdefault("PHI0_WORKSPACE", "/mnt/data2/wpy/workspace")

from phi0.deploy.dex3_gripper import gripper14_batch_to_zmq_lr
from phi0.deploy.robot_proprio import deploy_hand7_to_wbc
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
from phi0.online.vision_dl_distill import encode_dual_vlm_from_frame_cache
from phi0.online.vlm_frame_latents import DualVlmFrameLatentCache
from phi0.online.vlm_hold_distill import DistillVlmHold, resolve_vlm_hold_period

HZ = 50.0
OUT = PHI0_ROOT / "logs"
INSTRUCT_VLM = (
    "/mnt/data3/hf_home/hub/models--Qwen--Qwen3-VL-2B-Instruct/"
    "snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)


def closure_from_zmq(left, right):
    return 0.5 * (np.mean(np.abs(left[:, 3:7]), 1) + np.mean(np.abs(right[:, 3:7]), 1))


def load_hand_gt_wbc14(raw: Path, t_len: int) -> np.ndarray:
    schema = set(pq.read_schema(raw).names)
    if "action.dex3.left.position" in schema:
        cols = ["action.dex3.left.position", "action.dex3.right.position"]
        t = pq.read_table(raw, columns=cols)
        al = np.stack(t.column(0).to_pylist()).astype(np.float32)[:t_len]
        ar = np.stack(t.column(1).to_pylist()).astype(np.float32)[:t_len]
        return np.concatenate(
            [
                np.stack([deploy_hand7_to_wbc(x) for x in al], 0),
                np.stack([deploy_hand7_to_wbc(x) for x in ar], 0),
            ],
            1,
        ).astype(np.float32)
    if "teleop.left_hand_joints" in schema:
        cols = ["teleop.left_hand_joints", "teleop.right_hand_joints"]
        t = pq.read_table(raw, columns=cols)
        al = np.stack(t.column(0).to_pylist()).astype(np.float32)[:t_len]
        ar = np.stack(t.column(1).to_pylist()).astype(np.float32)[:t_len]
        return np.concatenate(
            [
                np.stack([deploy_hand7_to_wbc(x) for x in al], 0),
                np.stack([deploy_hand7_to_wbc(x) for x in ar], 0),
            ],
            1,
        ).astype(np.float32)
    if "action.unified" in schema:
        # teleop unified: dex3 gripper14 @ [346:360)
        U = np.stack(
            pq.read_table(raw, columns=["action.unified"]).column(0).to_pylist()
        ).astype(np.float32)[:t_len]
        return U[:, 346:360].copy()
    raise ValueError(f"no hand columns in {raw}: {sorted(schema)[:20]}")


def summarize_hand(name, c, gt=None):
    dc = np.diff(c)
    rev = int(np.sum((dc[1:] * dc[:-1]) < 0)) if len(dc) > 1 else 0
    out = {
        "kind": "hand",
        "name": name,
        "T": int(len(c)),
        "mean": float(c.mean()),
        "d_abs_mean": float(np.abs(dc).mean()) if len(dc) else 0.0,
        "reversals_per_sec": float(rev / (len(c) / HZ)),
    }
    if gt is not None:
        out["mae_vs_gt"] = float(np.mean(np.abs(c - gt)))
    # chunk junctions every PLAY later filled by caller
    return out


def summarize_z(name, z, gt=None):
    e = np.linalg.norm(z, axis=1)
    dz = np.linalg.norm(np.diff(z, axis=0), axis=1) if len(z) > 1 else np.zeros(0)
    out = {
        "kind": "z",
        "name": name,
        "z_norm_mean": float(e.mean()),
        "step_dz_mean": float(dz.mean()) if len(dz) else 0.0,
    }
    if gt is not None:
        out["mae_vs_gt"] = float(np.mean(np.abs(z - gt)))
    return out


def junction_mae(series: np.ndarray, play: int) -> float:
    """Mean |Δ| at chunk boundaries (indices play, 2play, ...)."""
    if play <= 0 or len(series) <= play:
        return float("nan")
    idxs = list(range(play, len(series), play))
    if not idxs:
        return float("nan")
    return float(np.mean([abs(series[i] - series[i - 1]) for i in idxs]))


def run_open_loop(
    *,
    name: str,
    ckpt: Path,
    ref_root: Path,
    start: int,
    length: int,
    out_ep: int,
    raw: Path,
    device: torch.device,
    use_frame_cache: bool,
    use_rtc: bool,
    vlm_path: str | None,
):
    print(
        f"[closure] === {name} rtc={int(use_rtc)} cache={int(use_frame_cache)} ===",
        flush=True,
    )
    os.environ["USE_RTC"] = "1" if use_rtc else "0"
    if use_rtc:
        os.environ.setdefault("PHI0_RTC_INFERENCE_DELAY", "6")
        # H=32: s=15 satisfies d<=s<=H-d
        os.environ.setdefault("PHI0_RTC_EXECUTION_HORIZON", "15")
    else:
        pass  # USE_RTC=0 already set; prefix follows enabled

    if use_frame_cache:
        os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "1"
        os.environ["USE_VLM"] = "0"
        os.environ["PHI0_VLM_FRAME_LATENTS_DIRNAME"] = "vlm_frame_latents_qwen3vl_dual"
    else:
        os.environ["PHI0_USE_VLM_FRAME_LATENT_CACHE"] = "0"
        os.environ["USE_VLM"] = "1"
        if vlm_path:
            os.environ["PHI0_VLM_MODEL_PATH"] = vlm_path

    ref = load_sonic_latent_reference(
        str(ref_root),
        max_frames=length,
        start=start,
        require_rsi=True,
        require_smpl=False,
    )
    _attach_prompts_if_needed(ref, str(ref_root))
    attach_has_video_to_ref(ref, str(ref_root))
    t_len = len(ref)
    assert int(np.asarray(ref.episode_index)[0]) == out_ep, (
        f"ep {int(np.asarray(ref.episode_index)[0])} != {out_ep}"
    )
    z_gt = np.asarray(ref.z_ref, dtype=np.float32)[:t_len]
    ep_idx = np.asarray(ref.episode_index, dtype=np.int64)
    fr_idx = np.asarray(ref.frame_index, dtype=np.int64)

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    h = int(payload["horizon"])
    hist_len = int(payload.get("history_len") or distill_obs_hist_len())
    adaln = normalize_adaln_mode(str(payload.get("adaln_mode") or "progress_only"))
    student = Phi0ChunkStudent(
        build_phi0_student(
            device=str(device),
            use_vlm=not use_frame_cache,
            use_lang_latent_cache=False,
            horizon=h,
            vlm_age_h=int(payload.get("vlm_age_h", 0) or 0),
            adaln_mode=adaln,
            history_len=hist_len,
            vlm_model_path=vlm_path if not use_frame_cache else None,
        ),
        horizon=h,
        history_len=hist_len,
        require_lang_ctx=True,
    ).to(device)
    load_student_act_state_dict(student, payload, expected_horizon=h)
    student.eval()
    proc, st = build_distill_processor(
        ckpt.parent / "action_stats.json",
        phi0_full_v3_root=str(ref_root),
        require_z_stats=False,
    )
    dnorm = DistillNorm(proc, st)
    dnorm.attach_to_student(student)

    rtc = distill_rtc_deploy_cfg(h)
    rtc_on = bool(use_rtc) and bool(rtc.get("enabled"))
    rtc_d = int(rtc["inference_delay"]) if rtc_on else 0
    rtc_s = int(rtc["execution_horizon"]) if rtc_on else h
    play = int(rtc_play_horizon(rtc, h) if rtc_on else h)
    rtc_prefix_cond = bool(rtc_on)
    rtc_mask = (
        create_rtc_hard_mask(h, rtc_d, rtc_s, device=device) if rtc_on else None
    )
    print(
        f"  H={h} play={play} rtc_on={int(rtc_on)} d={rtc_d} s={rtc_s} "
        f"prefix={int(rtc_prefix_cond)}",
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

    frame_cache = (
        DualVlmFrameLatentCache.open(ref_root, dirname="vlm_frame_latents_qwen3vl_dual")
        if use_frame_cache
        else None
    )
    video_src = None
    hold = None
    if not use_frame_cache:
        video_src = RefVideoFrameSource(str(ref_root), fps=50.0)
        hold = DistillVlmHold(1, period=resolve_vlm_hold_period(default=1))

    body = np.asarray(ref.fk_dof29, dtype=np.float32)
    hand_gt = load_hand_gt_wbc14(raw, t_len)
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
            if frame_cache is not None:
                lang_ctx, lang_mask = encode_dual_vlm_from_frame_cache(
                    cache=frame_cache,
                    episode_indices=[int(ep_idx[ti])],
                    frame_indices=[int(fr_idx[ti])],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                assert hold is not None and video_src is not None
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
            rtc_kw = build_rtc_infer_prefix_kwargs(
                prev_z=prev_z,
                prev_hand=prev_hand,
                inference_delay=rtc_d,
                execution_horizon=rtc_s,
                batch_size=1,
                device=device,
                norm_z=dnorm.norm_z,
                norm_hand=dnorm.norm_hand,
                enabled=rtc_prefix_cond,
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
            if t % 160 == 0 or t > t_max:
                print(f"  {name} t={min(t, t_len)}/{t_len}", flush=True)

    h_p = np.stack(h_log, 0)[:t_len]
    z_p = np.stack(z_log, 0)[:t_len]
    L, R = gripper14_batch_to_zmq_lr(h_p, policy_order=True)
    c = closure_from_zmq(L, R)
    del student
    if frame_cache is not None:
        del frame_cache
    torch.cuda.empty_cache()
    return c, z_p, z_gt, hand_gt[:t_len], play


def plot_compare(tag, hand_c, z_pred, z_gt, order, play_by_name, title_prefix):
    colors = {
        "GT": "#9aa0a6",
        "Instruct": "#2563eb",
        "RTC-prefix": "#16a34a",
        "RTC-off": "#2563eb",
        "RTC-on": "#16a34a",
    }
    # Fallback palette for any other series names.
    _fallback = ("#ea580c", "#7c3aed", "#0891b2", "#db2777")
    _fi = 0
    for name in order:
        if name not in colors:
            colors[name] = _fallback[_fi % len(_fallback)]
            _fi += 1
    T = min(len(hand_c["GT"]), len(z_gt), *(len(hand_c[k]) for k in order if k != "GT"))
    t = np.arange(T) / HZ
    zE = {"GT": np.linalg.norm(z_gt[:T], axis=1)}
    for k in order:
        if k == "GT":
            continue
        hand_c[k] = hand_c[k][:T]
        z_pred[k] = z_pred[k][:T]
        zE[k] = np.linalg.norm(z_pred[k], axis=1)
    hand_c["GT"] = hand_c["GT"][:T]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=140)
    panels = [
        (axes[0], hand_c, None, "hand closure", "closure"),
        (axes[1], zE, None, "body sonic ||z||", "||z||"),
        (axes[2], hand_c, (0, 4), "hand zoom 0–4s", "closure"),
        (axes[3], zE, (0, 4), "z zoom 0–4s", "||z||"),
    ]
    for ax, series, xlim, title, ylab in panels:
        for name in order:
            ax.plot(
                t,
                series[name],
                color=colors.get(name, "#333"),
                lw=1.3 if name == "GT" else 1.0,
                label=name,
                alpha=0.95,
            )
        # mark Instruct play=32 junctions
        for b in range(32, T, 32):
            ax.axvline(b / HZ, color="#dadce0", lw=0.4, zorder=0)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title_prefix} — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        if xlim:
            ax.set_xlim(*xlim)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    png = OUT / f"{tag}.png"
    fig.savefig(png)
    print(f"[closure] wrote {png}", flush=True)
    return png


def run_episode(
    *,
    tag: str,
    title_prefix: str,
    ref_root: Path,
    start: int,
    length: int,
    out_ep: int,
    raw: Path,
    use_frame_cache: bool,
    baseline: Path,
    rtc_ckpt: Path,
    device: torch.device,
):
    specs = [
        ("Instruct", baseline, False),
        ("RTC-prefix", rtc_ckpt, True),
    ]
    hand_c = {}
    z_pred = {}
    z_gt = hand_gt14 = None
    play_by = {}
    for name, ckpt, use_rtc in specs:
        assert ckpt.is_file(), ckpt
        c, zp, zg, hg, play = run_open_loop(
            name=name,
            ckpt=ckpt,
            ref_root=ref_root,
            start=start,
            length=length,
            out_ep=out_ep,
            raw=raw,
            device=device,
            use_frame_cache=use_frame_cache,
            use_rtc=use_rtc,
            vlm_path=None if use_frame_cache else INSTRUCT_VLM,
        )
        hand_c[name] = c
        z_pred[name] = zp
        z_gt = zg
        hand_gt14 = hg
        play_by[name] = play

    Lgt, Rgt = gripper14_batch_to_zmq_lr(hand_gt14, policy_order=True)
    hand_c["GT"] = closure_from_zmq(Lgt, Rgt)
    order = ("GT", "Instruct", "RTC-prefix")
    stats = [
        summarize_hand(k, hand_c[k], None if k == "GT" else hand_c["GT"]) for k in order
    ]
    for s in stats:
        name = s["name"]
        if name == "GT":
            continue
        s["junction_abs_mean"] = junction_mae(hand_c[name], int(play_by[name]))
        s["play"] = int(play_by[name])
        s["rtc"] = int(name == "RTC-prefix")
    stats += [
        summarize_z(k, z_gt if k == "GT" else z_pred[k], None if k == "GT" else z_gt)
        for k in order
    ]
    for s in stats:
        if s["kind"] != "z" or s["name"] == "GT":
            continue
        name = s["name"]
        e = np.linalg.norm(z_pred[name], axis=1)
        s["junction_abs_mean"] = junction_mae(e, int(play_by[name]))
        s["play"] = int(play_by[name])
        s["rtc"] = int(name == "RTC-prefix")
    stats.append(
        {
            "ep": {
                "tag": tag,
                "out_ep": out_ep,
                "T": int(len(hand_c["GT"])),
                "vlm": "frame_cache" if use_frame_cache else "live_dual",
                "baseline": str(baseline),
                "rtc_ckpt": str(rtc_ckpt),
            }
        }
    )
    (OUT / f"{tag}_stats.json").write_text(json.dumps(stats, indent=2))
    np.savez_compressed(
        OUT / f"{tag}.npz",
        **{f"hand_{k}": hand_c[k].astype(np.float32) for k in order},
        z_GT=z_gt.astype(np.float32),
        **{f"z_{k}": z_pred[k].astype(np.float32) for k in ("Instruct", "RTC-prefix")},
    )
    print(json.dumps(stats, indent=2), flush=True)
    plot_compare(tag, hand_c, z_pred, z_gt, order, play_by, title_prefix)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--rtc-ckpt", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", choices=("train", "ood", "both"), default="both")
    args = p.parse_args()
    device = torch.device(args.device)
    baseline = args.baseline
    rtc_ckpt = args.rtc_ckpt
    if baseline.is_dir():
        baseline = baseline / "phi0_student_last.pt"
    if rtc_ckpt.is_dir():
        rtc_ckpt = rtc_ckpt / "phi0_student_last.pt"

    if args.only in ("train", "both"):
        run_episode(
            tag="pico_pure_sess220456_ep1_instruct_vs_rtcprefix_closure",
            title_prefix=(
                "pure 2026-09-05-22-04-56 src_ep1 (mix ep356) — Instruct vs RTC-prefix"
            ),
            ref_root=Path(
                "/mnt/data3/wpy/datasets/830/830mix_pico_pick_toy_pure_unified"
            ),
            start=200970,
            length=674,
            out_ep=356,
            raw=Path(
                "/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy_pure/"
                "2026-09-05-22-04-56/data/chunk-000/episode_000001.parquet"
            ),
            use_frame_cache=True,
            baseline=baseline,
            rtc_ckpt=rtc_ckpt,
            device=device,
        )
    if args.only in ("ood", "both"):
        run_episode(
            tag="pico_0828_sess001305_ep1_instruct_vs_rtcprefix_closure",
            title_prefix=(
                "0828 OOD 2026-08-29-00-13-05 src_ep1 (ep40) — Instruct vs RTC-prefix"
            ),
            ref_root=Path("/mnt/data3/wpy/datasets/830/0828_skill2_unified"),
            start=25045,
            length=682,
            out_ep=40,
            raw=Path(
                "/mnt/data2/wpy/workspace/0828data/skill2_0828/"
                "2026-08-29-00-13-05/data/chunk-000/episode_000001.parquet"
            ),
            use_frame_cache=False,
            baseline=baseline,
            rtc_ckpt=rtc_ckpt,
            device=device,
        )


if __name__ == "__main__":
    main()
