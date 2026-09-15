"""Egypt onnx_g1 distill on Lab3+Newton (qpos_ref recipe aligned).

Train expert drive = ``step_joint_qpos_batch(q*)`` (same as ``qpos_ref``).
Env: ImplicitPD + ensure_drives_exist + body-29 obs pin + Newton solver.

See ``meta/NEWTON_train_deploy_pipeline.md``.
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

from newton_gate_common import fill_obs_dims, load_atm_policy, load_full_cfg
from phi0.online.isaac_loop import run_online_distill
from phi0.online.isaac_sim import assert_env_control_hz, enable_direct_latent_atm

config_utils.register_rl_resolvers()

PHI0 = _PHI0
EGYPT = Path("/mnt/data2/wpy/workspace/egypt_smplsem_clip")
Z_STATS = PHI0 / "meta" / "z_star_stats.json"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_chunks", type=int, default=64)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--ckpt_every", type=int, default=1000)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Adam learning rate for student (default 1e-4)",
)
parser.add_argument(
    "--student_ckpt",
    type=str,
    default="",
    help="Resume ACT student weights (optional); metrics resume if out_dir has distill_metrics.json",
)
parser.add_argument(
    "--rsi_start",
    type=str,
    default="first_frame",
    choices=("first_frame", "random"),
    help="RSI phase on init/wrap/fall: first_frame | random",
)
parser.add_argument("--out_dir", type=str, default="")
add_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = setup_preset_cli(parser)
if hasattr(args_cli, "visualizer"):
    args_cli.visualizer = []
sys.argv = [sys.argv[0]] + hydra_args


def _log_recipe_alignment(env_cfg, env) -> None:
    """Print qpos_ref-recipe env + train-drive checks."""
    robot_cfg = env_cfg.scene.robot
    jd = getattr(robot_cfg.spawn, "joint_drive_props", None)
    ensure = bool(getattr(jd, "ensure_drives_exist", False)) if jd is not None else False
    cfg_types = {k: type(v).__name__ for k, v in robot_cfg.actuators.items()}
    cfg_ok = all(isinstance(v, ImplicitActuatorCfg) for v in robot_cfg.actuators.values())
    use_na = bool(getattr(env_cfg.sim, "use_newton_actuators", False))
    robot = env.env.scene["robot"]
    rt_ok = all(isinstance(v, ImplicitActuator) for v in robot.actuators.values())
    print(
        f"[DISTILL_RECIPE] ensure_drives_exist={ensure} use_newton_actuators={use_na} "
        f"cfg_Implicit={cfg_ok} runtime_Implicit={rt_ok} cfg_types={cfg_types}",
        flush=True,
    )
    print(
        "[DISTILL_RECIPE] train_control=joint_qpos(q*) "
        "(aligned to qpos_ref; decode→step_joint_qpos_batch).",
        flush=True,
    )
    if not (ensure and cfg_ok and rt_ok and use_na):
        raise RuntimeError(
            f"recipe misaligned: ensure={ensure} cfg_ok={cfg_ok} rt_ok={rt_ok} use_na={use_na}"
        )


def main():
    torch.manual_seed(42)
    assert EGYPT.is_dir(), EGYPT
    assert Z_STATS.is_file(), Z_STATS
    out = Path(
        args_cli.out_dir
        or os.environ.get(
            "PHI0_DISTILL_OUT",
            str(PHI0 / "experiments" / "newton_egypt_distill"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    conf = load_full_cfg(args_cli.num_envs)
    env_cfg = custom_instantiate(conf.manager_env)
    env_cfg.seed = 42
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "config") and isinstance(env_cfg.config, dict):
        env_cfg.config["headless"] = True

    phys = type(getattr(env_cfg.sim, "physics", None)).__name__
    resume_ckpt = (args_cli.student_ckpt or "").strip() or None
    if resume_ckpt is not None and not Path(resume_ckpt).is_file():
        raise FileNotFoundError(f"--student_ckpt not found: {resume_ckpt}")
    print(
        f"[DISTILL] physics={phys} B={args_cli.num_envs} chunks={args_cli.num_chunks} "
        f"H={args_cli.horizon} lr={float(args_cli.lr):g} "
        f"resume={resume_ckpt or '-'} out={out}",
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
            max_ref_frames=278,
            phi0_full_v3_root=str(EGYPT),
            seed=42,
            out_dir=out,
            student_ckpt=resume_ckpt,
            clip_stride=1,
            lazy_ref=False,
            rg_scan=False,
            ref_start=0,
            z_star_stats_path=str(Z_STATS),
            ckpt_every=int(args_cli.ckpt_every),
            teacher_encoder="onnx_g1",
            expert_fall_z_err=0.25,
            rsi_start=str(args_cli.rsi_start),
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
        # ``num_chunks`` = steps this call; with resume into same out_dir,
        # ``n`` is cumulative (prev + this) and still ``n >= num_chunks``.
        if n < int(args_cli.num_chunks):
            raise RuntimeError(f"chunks {n} < requested {args_cli.num_chunks}")
        if loss_last is None or not math.isfinite(float(loss_last)):
            raise RuntimeError(f"non-finite loss_last={loss_last}")
        metrics = out / "distill_metrics.json"
        if not metrics.is_file():
            raise RuntimeError(f"missing {metrics}")
        print(
            f"[NEWTON_EGYPT_DISTILL_OK] chunks={n} loss_last={float(loss_last):.6f} "
            f"chunk/s={sps:.3f} physics={phys} out={out}",
            flush=True,
        )
        close = getattr(env, "close", None) or getattr(inner, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
