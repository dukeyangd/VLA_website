#!/usr/bin/env python3
"""BoneSEED absquat full distill on Lab3+Newton (single-process / debug).

Same recipe as newton_egypt_distill, but:
  - REF = smpl_gmr_relroot_phi0
  - rg_scan=1, random_episode RSI, lang latent cache → interleave_vlm
  - Phi0 GR00T action head (via build_phi0_student)

Multi-GPU (canonical): Fabric DDP AllReduce via
``tools/train/run_boneseed_full_distill_4gpu.sh`` → ``run_sonic_online_distill_fabric.sh``.
Shard-independent multi-process (SHARD_RANK/WORLD, no grad sync) is a bug/legacy
path — do not use for real multi-GPU distill.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ["PHI0_ISAAC_BACKEND"] = os.environ.get("PHI0_ISAAC_BACKEND", "newton")
os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

_PHI0 = Path(__file__).resolve().parents[2]
_LIB = _PHI0 / "scripts" / "lib"
for _p in (_PHI0 / "src", _LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
from isaaclab_tasks.utils.sim_launcher import launch_simulation

from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.utils import config_utils

from newton_gate_common import (
    active_dr_event_names,
    fill_obs_dims,
    load_atm_policy,
    load_full_cfg,
)
from phi0.online.isaac_loop import run_online_distill
from phi0.online.isaac_sim import assert_env_control_hz, enable_direct_latent_atm

config_utils.register_rl_resolvers()

PHI0 = _PHI0
DEFAULT_REF = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
Z_STATS = PHI0 / "meta" / "z_star_stats_boneseed.json"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_chunks", type=int, default=50000)
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--ckpt_every", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--student_ckpt", type=str, default="")
parser.add_argument(
    "--rsi_start",
    type=str,
    default="random",
    choices=("first_frame", "random", "random_episode"),
    help="random=hybrid ep0 (PHI0_RSI_EP0_PROB, default 0.5) + mid-ep.",
)
parser.add_argument("--ref_root", type=str, default=str(DEFAULT_REF))
parser.add_argument("--rg_scan", action="store_true", default=True)
parser.add_argument("--no_rg_scan", action="store_false", dest="rg_scan")
parser.add_argument("--use_lang_latent_cache", action="store_true", default=True)
parser.add_argument("--no_lang_latent_cache", action="store_false", dest="use_lang_latent_cache")
parser.add_argument("--shard_rank", type=int, default=int(os.environ.get("SHARD_RANK", "0")))
parser.add_argument("--shard_world", type=int, default=int(os.environ.get("SHARD_WORLD", "1")))
parser.add_argument("--out_dir", type=str, default="")
_DEFAULT_ALLOW = PHI0 / "meta" / "boneseed_allowlist_demo5skill.json"
parser.add_argument(
    "--episode_allowlist",
    type=str,
    default=str(_DEFAULT_ALLOW),
    help="JSON with episode_index allowlist; default=egypt-only (ep 199).",
)
parser.add_argument(
    "--no_episode_allowlist",
    action="store_true",
    help="Train on full corpus (disable default egypt allowlist).",
)
parser.add_argument(
    "--episode_blacklist",
    type=str,
    default="",
    help="JSON deny-list of episode_index (same schema as allowlist). Empty=off.",
)
parser.add_argument(
    "--no_dr",
    action="store_true",
    help="Disable sonic_release Event DR + obs corruption (default: DR on).",
)
add_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = setup_preset_cli(parser)
if hasattr(args_cli, "visualizer"):
    args_cli.visualizer = []
sys.argv = [sys.argv[0]] + hydra_args


def _log_recipe_alignment(env_cfg, env) -> None:
    robot_cfg = env_cfg.scene.robot
    jd = getattr(robot_cfg.spawn, "joint_drive_props", None)
    ensure = bool(getattr(jd, "ensure_drives_exist", False)) if jd is not None else False
    cfg_ok = all(isinstance(v, ImplicitActuatorCfg) for v in robot_cfg.actuators.values())
    use_na = bool(getattr(env_cfg.sim, "use_newton_actuators", False))
    robot = env.env.scene["robot"]
    rt_ok = all(isinstance(v, ImplicitActuator) for v in robot.actuators.values())
    print(
        f"[DISTILL_RECIPE] ensure_drives_exist={ensure} use_newton_actuators={use_na} "
        f"cfg_Implicit={cfg_ok} runtime_Implicit={rt_ok}",
        flush=True,
    )
    if not (ensure and cfg_ok and rt_ok and use_na):
        raise RuntimeError(
            f"recipe misaligned: ensure={ensure} cfg_ok={cfg_ok} rt_ok={rt_ok} use_na={use_na}"
        )


def main() -> None:
    torch.manual_seed(42 + int(args_cli.shard_rank))
    ref = Path(args_cli.ref_root)
    assert ref.is_dir(), ref
    assert Z_STATS.is_file(), Z_STATS
    assert (ref / "meta" / "stats.json").is_file(), ref / "meta" / "stats.json"
    if args_cli.use_lang_latent_cache:
        assert (ref / "meta" / "lang_latents_qwen3vl").is_dir(), "lang_latents_qwen3vl missing"

    out = Path(
        args_cli.out_dir
        or os.environ.get(
            "PHI0_DISTILL_OUT",
            str(PHI0 / "experiments" / "newton_boneseed_distill"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    enable_dr = not bool(getattr(args_cli, "no_dr", False))
    conf = load_full_cfg(args_cli.num_envs, enable_dr=enable_dr)
    env_cfg = custom_instantiate(conf.manager_env)
    env_cfg.seed = 42 + int(args_cli.shard_rank)
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "config") and isinstance(env_cfg.config, dict):
        env_cfg.config["headless"] = True

    phys = type(getattr(env_cfg.sim, "physics", None)).__name__
    resume_ckpt = (args_cli.student_ckpt or "").strip() or None
    if resume_ckpt is not None and not Path(resume_ckpt).is_file():
        raise FileNotFoundError(f"--student_ckpt not found: {resume_ckpt}")
    dr_names = active_dr_event_names(conf.manager_env)
    print(
        f"[DISTILL] physics={phys} B={args_cli.num_envs} chunks={args_cli.num_chunks} "
        f"H={args_cli.horizon} lr={float(args_cli.lr):g} rsi={args_cli.rsi_start} "
        f"allowlist={'off' if args_cli.no_episode_allowlist else args_cli.episode_allowlist} "
        f"blacklist={args_cli.episode_blacklist or 'off'} "
        f"rg_scan={args_cli.rg_scan} lang_cache={args_cli.use_lang_latent_cache} "
        f"dr={'on' if enable_dr else 'off'} events={dr_names} "
        f"shard={args_cli.shard_rank}/{args_cli.shard_world} "
        f"ref={ref} resume={resume_ckpt or '-'} out={out}",
        flush=True,
    )
    assert "Newton" in phys, f"expected NewtonCfg, got {phys}"

    t0 = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        inner = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env = ManagerEnvWrapper(inner, env_cfg.config)
        device = str(env.device)
        fill_obs_dims(env)
        atm = load_atm_policy(conf, env, device)
        enable_direct_latent_atm(env, atm)
        assert_env_control_hz(env, expected_hz=50.0)
        _log_recipe_alignment(env_cfg, env)

        result = run_online_distill(
            env=env,
            atm_policy=atm,
            device=device,
            num_chunks=int(args_cli.num_chunks),
            action_horizon=int(args_cli.horizon),
            lr=float(args_cli.lr),
            max_ref_frames=0,
            phi0_full_v3_root=str(ref),
            seed=42 + int(args_cli.shard_rank),
            out_dir=out,
            student_ckpt=resume_ckpt,
            clip_stride=1,
            lazy_ref=False,
            rg_scan=bool(args_cli.rg_scan),
            shard_rank=int(args_cli.shard_rank),
            shard_world=int(args_cli.shard_world),
            ref_start=0,
            use_vlm=False,
            use_lang_latent_cache=bool(args_cli.use_lang_latent_cache),
            action_stats_path=str(ref / "meta" / "stats.json"),
            z_star_stats_path=str(Z_STATS),
            ckpt_every=int(args_cli.ckpt_every),
            teacher_encoder="onnx_g1",
            expert_fall_z_err=0.25,
            rsi_start=str(args_cli.rsi_start),
            episode_allowlist=(
                None
                if bool(args_cli.no_episode_allowlist)
                else str(args_cli.episode_allowlist)
            ),
            episode_blacklist=(
                str(args_cli.episode_blacklist).strip() or None
            ),
        )
        n = int(result.get("num_chunks") or 0)
        loss_last = result.get("loss_last")
        elapsed = time.perf_counter() - t0
        sps = (n / elapsed) if elapsed > 0 else float("nan")
        print(
            f"[DISTILL] result num_chunks={n} loss_last={loss_last} ok={result.get('ok')} "
            f"elapsed_s={elapsed:.1f} chunk/s={sps:.3f}",
            flush=True,
        )
        if n < int(args_cli.num_chunks):
            raise RuntimeError(f"chunks {n} < requested {args_cli.num_chunks}")
        if loss_last is None or not math.isfinite(float(loss_last)):
            raise RuntimeError(f"non-finite loss_last={loss_last}")
        print(
            f"[NEWTON_BONESEED_DISTILL_OK] chunks={n} loss_last={float(loss_last):.6f} "
            f"chunk/s={sps:.3f} shard={args_cli.shard_rank}/{args_cli.shard_world} out={out}",
            flush=True,
        )
        close = getattr(env, "close", None) or getattr(inner, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
