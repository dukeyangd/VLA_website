#!/usr/bin/env python3
"""Micro-benchmark vision_dl BC step throughput (rank0 smoke)."""
from __future__ import annotations

import argparse
import os
import re
import time

import torch


def _parse_log_step_s(log_path: str) -> float | None:
    if not log_path or not os.path.isfile(log_path):
        return None
    vals: list[float] = []
    pat = re.compile(r"(\d+\.\d+) step/s")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "step/s" not in line or "[distill]" not in line:
            continue
        m = pat.search(line)
        if m:
            vals.append(float(m.group(1)))
    return sum(vals[-20:]) / len(vals[-20:]) if vals else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--baseline-log", default="")
    args = p.parse_args()

    os.environ.setdefault("PHI0_USE_VLM_FRAME_LATENT_CACHE", "1")
    os.environ.setdefault("PHI0_VLM_FRAME_CACHE_SKIP_VIDEO", "1")
    os.environ.setdefault("PHI0_HAND_MODE", "dex3")
    os.environ.setdefault("PHI0_ADALN_ZERO_VISION", "1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from phi0.online.distill_norm import DistillNorm, build_distill_processor
    from phi0.online.phi0_student import Phi0ChunkStudent, build_phi0_student
    from phi0.online.vision_dl_distill import (
        make_vision_dl_state,
        run_vision_dl_bc_step,
    )
    from phi0.models.adaln_exec import ADALN_MODE_PROGRESS_ONLY

    student = Phi0ChunkStudent(
        build_phi0_student(
            device=device,
            use_vlm=False,
            use_lang_latent_cache=False,
            horizon=args.horizon,
            adaln_mode=ADALN_MODE_PROGRESS_ONLY,
        ),
        horizon=args.horizon,
        require_lang_ctx=True,
    ).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
    student.train()

    stats_path = f"{args.ref}/meta/stats.json"
    proc, stats = build_distill_processor(
        stats_path,
        z_star_stats_path=None,
        require_z_stats=False,
        teacher_encoder="disk",
        auto_resolve_z_stats=False,
    )
    dnorm = DistillNorm(proc, stats)
    dnorm.attach_to_student(student)
    hand_mask = dnorm.hand_dim_mask(device=device)

    vdl = make_vision_dl_state(
        ref_root=args.ref,
        horizon=args.horizon,
        batch_size=args.batch,
        rank=0,
        world=1,
        seed=0,
    )
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    clip_params = list(student.trainable_parameters())

    it = vdl.it
    for _ in range(args.warmup):
        batch = next(it)
        run_vision_dl_bc_step(
            student=student,
            opt=opt,
            dnorm=dnorm,
            batch=batch,
            horizon=args.horizon,
            device=device,
            fabric=None,
            w_z=1.0,
            w_hand=1.0,
            w_q_head=0.0,
            hand_dim_mask=hand_mask,
            smpl_dim_mask=None,
            vlm_frame_cache=vdl.vlm_frame_cache,
            clip_params=clip_params,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        batch = next(it)
        run_vision_dl_bc_step(
            student=student,
            opt=opt,
            dnorm=dnorm,
            batch=batch,
            horizon=args.horizon,
            device=device,
            fabric=None,
            w_z=1.0,
            w_hand=1.0,
            w_q_head=0.0,
            hand_dim_mask=hand_mask,
            smpl_dim_mask=None,
            vlm_frame_cache=vdl.vlm_frame_cache,
            clip_params=clip_params,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sps = args.steps / elapsed
    baseline = _parse_log_step_s(args.baseline_log)
    print(f"benchmark steps={args.steps} batch={args.batch} elapsed={elapsed:.2f}s")
    print(f"step/s={sps:.2f} samples/s={sps * args.batch:.1f}")
    if baseline is not None:
        print(f"baseline_log_avg_step/s={baseline:.2f} delta={sps - baseline:+.2f}")


if __name__ == "__main__":
    main()
