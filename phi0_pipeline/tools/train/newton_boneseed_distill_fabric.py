#!/usr/bin/env python3
"""BoneSEED / online_vlm Newton distill — Fabric DDP + kit-less launch_simulation.

Canonical multi-GPU path for Lab3+Newton (no Isaac Sim ``SimulationApp``).
``torchrun`` remaps one GPU per rank → Fabric(devices=1, num_nodes=WORLD_SIZE)
→ ``launch_simulation`` → ``run_online_distill(..., fabric=)``.

Modes:
- ``PHI0_TRAIN_MODE=online_vlm`` / ``--use_vlm``: dual VLM hold + online onnx_g1
  (820 mix default via ``run_online_vlm_mix_distill.sh``).
- BoneSEED lang-cache: ``--use_lang_latent_cache`` (no live VLM).

Launcher: ``tools/train/run_online_vlm_mix_distill.sh`` /
``tools/train/run_boneseed_full_distill_4gpu.sh``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PHI0_ISAAC_BACKEND"] = os.environ.get("PHI0_ISAAC_BACKEND", "newton")
os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")
# onnx2torch + dynamo breaks; default off (override PHI0_ONNX_ENCODE_COMPILE=1).
os.environ.setdefault("PHI0_ONNX_ENCODE_COMPILE", "0")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
if os.environ.get("PHI0_DISTILL_QUIET", "1").lower() in ("1", "true", "yes"):
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

_PHI0 = Path(__file__).resolve().parents[2]
_LIB = _PHI0 / "scripts" / "lib"
_TOOLS_LIB = _PHI0 / "tools" / "lib"
for _p in (_PHI0 / "src", _LIB, _TOOLS_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _remap_cvd_one_gpu_per_rank() -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," not in cvd or "LOCAL_RANK" not in os.environ:
        return
    gpus = [g.strip() for g in cvd.split(",") if g.strip()]
    local = int(os.environ["LOCAL_RANK"])
    if local >= len(gpus):
        raise RuntimeError(f"LOCAL_RANK={local} but CVD has {len(gpus)} ids: {cvd}")
    os.environ["PHI0_FABRIC_PHYS_GPU"] = gpus[local]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[local]
    os.environ["LOCAL_RANK"] = "0"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="RG-list passes; stop after epochs×per-RG allowlist cover (required)",
    )
    p.add_argument(
        "--num_chunks",
        type=int,
        default=0,
        help=argparse.SUPPRESS,  # deprecated; BoneSEED ignores (epoch-cover stop)
    )
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument(
        "--ckpt_every",
        type=int,
        default=10000,
        help="Overwrite phi0_student_last.pt every N steps (PHI0_CKPT_STEP_KEEP=0).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--student_ckpt", type=str, default="")
    p.add_argument(
        "--rsi_start",
        type=str,
        default="random",
        choices=("first_frame", "random", "random_episode"),
        help="random=hybrid ep0 (PHI0_RSI_EP0_PROB, default 0.5) + mid-ep.",
    )
    p.add_argument(
        "--ref_root",
        type=str,
        default=str(
            Path(
                "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/"
                "train/smpl_gmr_relroot_phi0"
            )
        ),
    )
    p.add_argument("--rg_scan", action="store_true", default=True)
    p.add_argument("--no_rg_scan", action="store_false", dest="rg_scan")
    p.add_argument(
        "--resident_ref",
        action="store_true",
        default=False,
        help="Host-resident allowlist tape + global random ep (mutex with --rg_scan).",
    )
    p.add_argument("--use_lang_latent_cache", action="store_true", default=True)
    p.add_argument(
        "--no_lang_latent_cache", action="store_false", dest="use_lang_latent_cache"
    )
    p.add_argument(
        "--use_vlm",
        action="store_true",
        default=False,
        help="Online dual-VLM hold distill (PHI0_TRAIN_MODE=online_vlm). "
        "Mutex with lang latent cache; teacher_z=online.",
    )
    p.add_argument(
        "--no_use_vlm",
        action="store_false",
        dest="use_vlm",
        help="Disable dual VLM (BoneSEED lang-cache default).",
    )
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument(
        "--episode_allowlist",
        type=str,
        default=str(_PHI0 / "meta" / "boneseed_wl3_xy1m_from_wl2_B8192_ngpu8_20260804_122014.json"),
    )
    p.add_argument("--no_episode_allowlist", action="store_true")
    p.add_argument("--episode_blacklist", type=str, default="")
    p.add_argument(
        "--expert_fall_z_err",
        type=float,
        default=0.35,
        help="Pelvis |z_sim-z_fk| fall thr (expert-drive); default 0.35 = wl3 gate.",
    )
    p.add_argument(
        "--student_drive",
        action="store_true",
        default=False,
        help="Opt-in DAgger (default off). Expert-only slide-1 BC is default. "
        "PHI0_DISTILL_STUDENT_DRIVE=1.",
    )
    p.add_argument(
        "--obs_hist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Proprio history prefix K=10 when set. Default off → K=1 (align VLA). "
        "PHI0_DISTILL_OBS_HIST=0|1.",
    )
    p.add_argument(
        "--track_body_err",
        type=float,
        default=0.25,
        help="Relative body pos RSI thr (m) when --student_drive (default 0.25).",
    )
    p.add_argument(
        "--dagger_beta_start",
        type=float,
        default=1.0,
        help="DAgger teacher weight at start (β in β·q*+(1-β)·q_student).",
    )
    p.add_argument(
        "--dagger_beta_end",
        type=float,
        default=0.0,
        help="DAgger teacher weight end (ablation; default expert-only anneal=0).",
    )
    p.add_argument(
        "--dagger_beta_anneal_steps",
        type=int,
        default=0,
        help="Opt steps for β schedule length (0=off unless start≠1).",
    )
    p.add_argument(
        "--dagger_beta_schedule",
        type=str,
        default="linear",
        choices=("linear", "cosine"),
        help="β schedule for opt-in DAgger (default unused; anneal_steps=0).",
    )
    p.add_argument("--groot-root", type=str, default="")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--visualizer", nargs="*", default=[])
    p.add_argument(
        "--record_train_mp4",
        action="store_true",
        help="Record Newton-GL mp4 during train (follows env 0; works for any num_envs).",
    )
    p.add_argument("--record_every", type=int, default=50)
    p.add_argument(
        "--no_dr",
        action="store_true",
        help="Disable obs corruption (default: corruption on, Event DR off).",
    )
    p.add_argument(
        "--teacher_z_source",
        type=str,
        default="online",
        choices=("online", "hybrid", "disk"),
        help="online=onnx_g1 encode; hybrid=810 disk z + BoneSEED encode; "
        "disk=810-only disk z (mode=vla, no onnx encode).",
    )
    p.add_argument(
        "--ref_root_810",
        type=str,
        default="",
        help="810demo_egypt_layout root (required when --teacher_z_source=hybrid).",
    )
    p.add_argument(
        "--action_stats_path",
        type=str,
        default="",
        help="Override meta/stats.json (hybrid default: 810 stats).",
    )
    p.add_argument("--w_z", type=float, default=1.0)
    p.add_argument("--w_smpl", type=float, default=0.0)
    p.add_argument("--w_q_head", type=float, default=0.0)
    p.add_argument("--w_hand", type=float, default=0.0)
    p.add_argument(
        "--simulator",
        type=str,
        default=os.environ.get("PHI0_SIMULATOR", "newton"),
        choices=("newton", "physx"),
        help="newton=Lab3 kit-less (this entry); physx must use distill_fabric_entry / AppLauncher.",
    )
    return p.parse_args()


_SIM_HZ = 50.0


def _mp4_fps_for_record_every(record_every: int, *, sim_hz: float = _SIM_HZ) -> float:
    """Match playback speed to sim: each frame spans ``record_every`` steps at ``sim_hz``."""
    re = max(1, int(record_every))
    return float(sim_hz) / re


assert _mp4_fps_for_record_every(2) == 25.0 and _mp4_fps_for_record_every(10) == 5.0


def _write_distill_mp4(frames: list, path: Path, *, fps: float) -> None:
    """Write rgb frames to mp4. Atomic replace so players never see half-written moov."""
    if not frames:
        return
    import imageio
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep .mp4 suffix — imageio picks writer by extension (.mp4.tmp → TiffWriter).
    tmp = path.parent / f"{path.stem}.partial{path.suffix}"
    with imageio.get_writer(
        str(tmp), fps=fps, codec="libx264", quality=7, pixelformat="yuv420p"
    ) as w:
        for fr in frames:
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            if fr.ndim == 3 and fr.shape[-1] == 4:
                fr = fr[..., :3]
            w.append_data(fr)
    os.replace(tmp, path)


def main() -> None:
    _remap_cvd_one_gpu_per_rank()
    args = _parse()

    from phi0.online.sim_backend import (
        apply_recipe_env,
        assert_runtime_matches_recipe,
        get_distill_sim_recipe,
        resolve_expert_drive,
    )

    if str(args.simulator).lower() != "newton":
        raise SystemExit(
            "newton_boneseed_distill_fabric.py is newton-only; "
            "use tools/train/distill_fabric_entry.py --simulator physx "
            "(or run_hybrid_diskz_qpos_hand_distill.sh SIMULATOR=physx)"
        )
    recipe = get_distill_sim_recipe("newton")
    apply_recipe_env(recipe)
    assert_runtime_matches_recipe(recipe)

    groot = os.path.abspath(
        args.groot_root
        or os.environ.get("GR00T_ROOT", str(Path(__file__).resolve().parents[2] / "subpackages"))
    )
    os.chdir(groot)
    if groot not in sys.path:
        sys.path.insert(0, groot)

    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy

    import torch

    # A800 Tensor Cores: TF32 matmul (logs already warn without this).
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    # Learnable-query path no longer registers legacy action_encoder → default off.
    # Override with PHI0_DISTILL_FIND_UNUSED=1 if a new unused tower appears.
    find_unused = os.environ.get("PHI0_DISTILL_FIND_UNUSED", "0").lower() not in (
        "0",
        "false",
        "no",
    )
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
        num_nodes=world,
        strategy=DDPStrategy(
            find_unused_parameters=find_unused,
            broadcast_buffers=False,
        ),
        # ACT fwd/bwd in bf16; loss stays fp32 via .float() in distill losses.
        precision="bf16-mixed",
    )
    fabric.launch()

    os.environ["PHI0_FABRIC"] = "1"
    os.environ["SHARD_RANK"] = str(fabric.global_rank)
    os.environ["SHARD_WORLD"] = str(fabric.world_size)
    os.environ["RANK"] = str(fabric.global_rank)
    os.environ["WORLD_SIZE"] = str(fabric.world_size)
    os.environ["LOCAL_RANK"] = "0"

    from phi0.online.fabric_runtime import set_fabric

    set_fabric(fabric)
    from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    from isaaclab_tasks.utils.sim_launcher import launch_simulation

    from omegaconf import OmegaConf

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

    # launch_simulation needs AppLauncher-style args; rebuild via preset CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=args.num_envs)
    add_launcher_args(parser)
    parser.set_defaults(headless=True, num_envs=args.num_envs)
    args_cli, _hydra = setup_preset_cli(parser)
    args_cli.num_envs = int(args.num_envs)
    viz_enabled = bool(args.record_train_mp4)
    if args.record_train_mp4 and int(args.num_envs) != 1 and fabric.is_global_zero:
        print(
            f"[DISTILL_FABRIC] record_train_mp4: B={args.num_envs} → camera follows env 0",
            flush=True,
        )
    args_cli.headless = True
    if hasattr(args_cli, "visualizer"):
        args_cli.visualizer = []

    ref = Path(args.ref_root)
    assert ref.is_dir(), ref
    teacher_z = str(
        os.environ.get("TEACHER_Z_SOURCE", args.teacher_z_source) or "online"
    ).strip().lower()
    _mode = os.environ.get("PHI0_TRAIN_MODE", "").strip().lower()
    _env_vlm = os.environ.get("USE_VLM", os.environ.get("PHI0_USE_VLM", "")).strip().lower()
    _frame_cache = os.environ.get(
        "PHI0_USE_VLM_FRAME_LATENT_CACHE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    use_vlm = bool(args.use_vlm) or _mode in ("online_vlm", "online-vlm")
    if _env_vlm in ("1", "true", "yes", "on"):
        use_vlm = True
    elif _env_vlm in ("0", "false", "no", "off"):
        use_vlm = False
    if _frame_cache:
        # Per-frame dual cache injects lang_ctx in vision_dl; no live Qwen tower.
        use_vlm = False
        args.use_lang_latent_cache = False
        if fabric.is_global_zero:
            print(
                "[DISTILL_FABRIC] vlm_frame_latent_cache=1 → action-only student "
                "(skip Qwen tower; ctx from DualVlmFrameLatentCache)",
                flush=True,
            )
    if use_vlm:
        os.environ["PHI0_TRAIN_MODE"] = "online_vlm"
        # Default online onnx_g1; allow TEACHER_Z_SOURCE=disk (820 release_unified GT).
        if teacher_z not in ("disk", "hybrid"):
            teacher_z = "online"
        args.use_lang_latent_cache = False
        os.environ.setdefault("PHI0_ADALN_MODE", "progress_vlm_age")
        os.environ.setdefault("PHI0_RSI_EP0_PROB", "0.5")
    elif _mode == "vla" and teacher_z == "online":
        teacher_z = "disk"
    elif _mode == "distill" and teacher_z not in ("hybrid", "disk"):
        teacher_z = "online"
    if teacher_z not in ("online", "hybrid", "disk"):
        raise SystemExit(f"teacher_z_source must be online|hybrid|disk, got {teacher_z!r}")
    if use_vlm and teacher_z == "hybrid":
        raise SystemExit("online_vlm + teacher_z_source=hybrid is not supported")

    # disk: 820 release_unified already has sonic mean/std in action.unified →
    # default no boneseed z_star overlay (set Z_STAR_STATS_PATH to force).
    z_stats_raw = str(os.environ.get("Z_STAR_STATS_PATH", "")).strip()
    if teacher_z == "disk" and z_stats_raw.lower() in ("", "none", "0", "off"):
        z_stats: Path | None = None
        if fabric.is_global_zero:
            print(
                "[DISTILL_FABRIC] disk-z: Z_STAR_STATS_PATH=none "
                "(use dataset action.unified sonic stats)",
                flush=True,
            )
    else:
        z_stats = Path(
            z_stats_raw
            or str(_PHI0 / "meta" / "z_star_stats_boneseed.json")
        )
        assert z_stats.is_file(), z_stats
    assert (ref / "meta" / "stats.json").is_file()
    # disk/vla 810 layout has no lang_latents; allow zero-ctx student.
    # online_vlm: dual hold encodes live — no lang_latents dir required.
    if args.use_lang_latent_cache and teacher_z != "disk" and not use_vlm:
        assert (ref / "meta" / "lang_latents_qwen3vl").is_dir()
    elif teacher_z == "disk" and args.use_lang_latent_cache:
        if not (ref / "meta" / "lang_latents_qwen3vl").is_dir():
            if fabric.is_global_zero:
                print(
                    "[DISTILL_FABRIC] disk/vla: no lang_latents_qwen3vl → "
                    "forcing use_lang_latent_cache=0 (zero lang ctx)",
                    flush=True,
                )
            args.use_lang_latent_cache = False
    if use_vlm and args.use_lang_latent_cache:
        raise SystemExit("use_vlm/online_vlm is mutually exclusive with lang latent cache")

    ref_810 = (
        str(os.environ.get("REF_ROOT_810", args.ref_root_810) or "").strip()
        or None
    )
    if teacher_z == "hybrid":
        if not ref_810:
            raise SystemExit("hybrid requires --ref_root_810 / REF_ROOT_810")
        if not Path(ref_810).is_dir():
            raise SystemExit(f"ref_root_810 missing: {ref_810}")
    if teacher_z == "disk" and not bool(args.resident_ref):
        # Env may set RESIDENT_REF later; still require intent for disk.
        env_res0 = os.environ.get("RESIDENT_REF", os.environ.get("PHI0_DISTILL_RESIDENT", ""))
        if str(env_res0).lower() not in ("1", "true", "yes", "on"):
            raise SystemExit("teacher_z_source=disk requires --resident_ref / RESIDENT_REF=1")
    w_z = float(os.environ.get("W_Z", args.w_z))
    w_smpl = float(os.environ.get("W_SMPL", args.w_smpl))
    w_q_head = float(os.environ.get("W_Q_HEAD", args.w_q_head))
    w_hand = float(os.environ.get("W_HAND", args.w_hand))
    stats_override = (
        str(os.environ.get("ACTION_STATS_PATH", args.action_stats_path) or "").strip()
        or None
    )
    action_stats = stats_override or str(ref / "meta" / "stats.json")
    if not Path(action_stats).is_file():
        raise SystemExit(f"action_stats missing: {action_stats}")

    out = Path(
        args.out_dir
        or os.environ.get("PHI0_DISTILL_OUT")
        or str(_PHI0 / "experiments" / "newton_boneseed_fabric")
    )
    if fabric.is_global_zero:
        out.mkdir(parents=True, exist_ok=True)
    fabric.barrier()

    from phi0.online.vision_dl_distill import vision_dl_only_from_env

    vision_dl_only = vision_dl_only_from_env()
    os.environ["NUM_ENVS"] = str(int(args.num_envs))

    allow = None if args.no_episode_allowlist else str(args.episode_allowlist)
    resume = (args.student_ckpt or "").strip() or None
    if int(args.epochs) < 1:
        raise SystemExit("--epochs must be >=1 (BoneSEED epoch-cover stop; no NUM_CHUNKS)")
    # Env override: RESIDENT_REF=1 / PHI0_DISTILL_RESIDENT=1
    env_res = os.environ.get("RESIDENT_REF", os.environ.get("PHI0_DISTILL_RESIDENT", ""))
    if str(env_res).lower() in ("1", "true", "yes", "on"):
        args.resident_ref = True
    env_sd = os.environ.get("PHI0_DISTILL_STUDENT_DRIVE", "").strip().lower()
    if env_sd in ("1", "true", "yes", "on"):
        args.student_drive = True
    env_oh = os.environ.get("PHI0_DISTILL_OBS_HIST", "").strip().lower()
    if env_oh in ("0", "false", "no", "off"):
        args.obs_hist = False
    elif env_oh in ("1", "true", "yes", "on"):
        args.obs_hist = True
    elif teacher_z == "disk":
        # mode=vla / unset: K=1 (matches BoneSEED distill default).
        args.obs_hist = False
    os.environ["PHI0_DISTILL_OBS_HIST"] = "1" if args.obs_hist else "0"
    if teacher_z == "disk":
        os.environ.setdefault("PHI0_DISTILL_STATE_DROPOUT", "0.2")
    else:
        os.environ.setdefault("PHI0_DISTILL_STATE_DROPOUT", "0")
    dagger_anneal = int(
        os.environ.get(
            "PHI0_DAGGER_BETA_ANNEAL_STEPS",
            str(int(args.dagger_beta_anneal_steps)),
        ).strip()
        or 0
    )
    # Expert-only default: anneal alone does NOT enable student_drive.
    if (dagger_anneal > 0 or abs(float(args.dagger_beta_start) - 1.0) > 1e-9) and not bool(
        args.student_drive
    ):
        raise SystemExit(
            "DAgger β anneal requires --student_drive / PHI0_DISTILL_STUDENT_DRIVE=1 "
            "(default distill is expert-only)"
        )
    use_resident = bool(args.resident_ref)
    use_rg = bool(args.rg_scan) and not use_resident
    if use_resident and bool(args.rg_scan):
        if fabric.is_global_zero:
            print(
                "[DISTILL_FABRIC] resident_ref=1 → forcing rg_scan=0",
                flush=True,
            )
    if use_resident and allow is None:
        raise SystemExit("resident_ref requires --episode_allowlist")

    enable_dr = not bool(args.no_dr)
    conf = None
    env_cfg = None
    args_cli = None
    render_mode = None
    viz_frames: list = []
    viz_written = False
    dr_names: list[str] = []
    corr = False
    if not vision_dl_only:
        conf = load_full_cfg(args.num_envs, enable_dr=enable_dr)
        env_cfg = custom_instantiate(conf.manager_env)
        env_cfg.seed = 42 + int(fabric.global_rank)
        env_cfg.sim.device = "cuda:0"
        env_cfg.scene.num_envs = int(args.num_envs)
        if hasattr(env_cfg, "config") and isinstance(env_cfg.config, dict):
            env_cfg.config["headless"] = True
        if viz_enabled and fabric.is_global_zero:
            import numpy as np
            from isaaclab.envs.common import ViewerCfg

            env_cfg.viewer = ViewerCfg(eye=(2.8, 2.8, 1.4), lookat=(0.0, 0.0, 0.9))
            env_cfg.video_recorder.backend_source = "renderer"
            env_cfg.video_recorder.window_width = 960
            env_cfg.video_recorder.window_height = 544
            env_cfg.video_recorder.eye = env_cfg.viewer.eye
            env_cfg.video_recorder.lookat = env_cfg.viewer.lookat
            render_mode = "rgb_array"
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_envs", type=int, default=args.num_envs)
        add_launcher_args(parser)
        parser.set_defaults(headless=True, num_envs=args.num_envs)
        args_cli, _hydra = setup_preset_cli(parser)
        args_cli.num_envs = int(args.num_envs)
        args_cli.headless = True
        if hasattr(args_cli, "visualizer"):
            args_cli.visualizer = []
        dr_names = active_dr_event_names(conf.manager_env)
        corr = bool(OmegaConf.select(conf, "manager_env.observations.policy.enable_corruption"))

    if fabric.is_global_zero:
        print(
            f"[DISTILL_FABRIC] rank={fabric.global_rank}/{fabric.world_size} "
            f"phys_gpu={os.environ.get('PHI0_FABRIC_PHYS_GPU')} "
            f"B={args.num_envs} epochs={args.epochs} H={args.horizon} slide_gt=1 "
            f"ckpt_every={args.ckpt_every} lr={args.lr:g} rsi={args.rsi_start} "
            f"allowlist={allow or 'off'} lang_cache={args.use_lang_latent_cache} "
            f"use_vlm={int(use_vlm)} vision_dl_only={int(vision_dl_only)} "
            f"rg_scan={int(use_rg)} resident_ref={int(use_resident)} "
            f"drive={'student_first_token' if args.student_drive else 'expert'} "
            f"obs_hist={'on' if args.obs_hist else 'off'} "
            f"dagger_beta={args.dagger_beta_start:g}→{args.dagger_beta_end:g} "
            f"anneal={dagger_anneal} sched={args.dagger_beta_schedule} "
            f"track_thr={args.track_body_err:g} "
            f"dr={'on' if enable_dr and not vision_dl_only else 'off'} events={dr_names} corruption={int(corr)} "
            f"teacher_z={teacher_z} w_z={w_z:g} w_smpl={w_smpl:g} "
            f"w_q_head={w_q_head:g} w_hand={w_hand:g} "
            f"atm="
            f"{'release+low_latency' if teacher_z == 'hybrid' else ('DEPLOY_POLICY_DIR' if teacher_z == 'disk' else 'env_default')} "
            f"stats={action_stats} "
            f"out={out}",
            flush=True,
        )

    t0 = time.perf_counter()
    device = str(fabric.device)

    def _run_distill(*, env, atm, atm_ll, viz_capture=None) -> dict:
        return run_online_distill(
            env=env,
            atm_policy=atm,
            atm_low_latency=atm_ll,
            device=device,
            num_chunks=None,
            action_horizon=int(args.horizon),
            lr=float(args.lr),
            max_ref_frames=0,
            phi0_full_v3_root=str(ref),
            seed=42 + int(fabric.global_rank),
            out_dir=out,
            student_ckpt=resume,
            clip_stride=1,
            lazy_ref=False,
            rg_scan=bool(use_rg),
            resident_ref=bool(use_resident),
            shard_rank=int(fabric.global_rank),
            shard_world=int(fabric.world_size),
            ref_start=0,
            fabric=fabric,
            use_vlm=bool(use_vlm),
            use_lang_latent_cache=bool(args.use_lang_latent_cache),
            action_stats_path=str(action_stats),
            z_star_stats_path=(str(z_stats) if z_stats is not None else None),
            ckpt_every=int(args.ckpt_every),
            teacher_encoder="onnx_g1",
            teacher_z_source=str(teacher_z),
            ref_root_810=ref_810,
            w_z=float(w_z),
            w_smpl=float(w_smpl),
            w_q_head=float(w_q_head),
            w_hand=float(w_hand),
            expert_fall_z_err=float(args.expert_fall_z_err),
            student_drive=bool(args.student_drive),
            track_body_err=float(args.track_body_err),
            dagger_beta_start=float(args.dagger_beta_start),
            dagger_beta_end=float(args.dagger_beta_end),
            dagger_beta_anneal_steps=int(dagger_anneal),
            dagger_beta_schedule=str(args.dagger_beta_schedule),
            rsi_start=str(args.rsi_start),
            episode_allowlist=allow,
            episode_blacklist=(str(args.episode_blacklist).strip() or None),
            epochs=int(args.epochs),
            viz_capture=viz_capture,
            viz_every=int(args.record_every),
            expert_drive=resolve_expert_drive(
                simulator=str(recipe.name),
                teacher_z_source=str(teacher_z),
                expert_drive=None,
            ),
            simulator=str(recipe.name),
        )

    if vision_dl_only:
        if fabric.is_global_zero:
            print(
                "[DISTILL_FABRIC] vision_dl_only=1 → skip Isaac Sim (DataLoader BC only)",
                flush=True,
            )
        result = _run_distill(env=None, atm=None, atm_ll=None)
    else:
        with launch_simulation(env_cfg, args_cli):
            inner = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
            env = ManagerEnvWrapper(inner, env_cfg.config)
            device = str(fabric.device)
            fill_obs_dims(env)
            # BoneSEED online: sonic_release ATM. Hybrid: dual ATM.
            # Disk GT: ATM must match token family (820 release_unified → release).
            if teacher_z == "hybrid":
                atm = load_atm_policy(conf, env, device, policy_dir="release")
                atm_ll = load_atm_policy(conf, env, device, policy_dir="low_latency")
            elif teacher_z == "disk":
                disk_pol = str(os.environ.get("DEPLOY_POLICY_DIR", "release")).strip() or "release"
                atm = load_atm_policy(conf, env, device, policy_dir=disk_pol)
                atm_ll = None
                if fabric.is_global_zero:
                    print(
                        f"[DISTILL_FABRIC] disk-z ATM policy_dir={disk_pol}",
                        flush=True,
                    )
            else:
                atm = load_atm_policy(conf, env, device)
                atm_ll = None
            enable_direct_latent_atm(env, atm)
            assert_env_control_hz(env, expected_hz=50.0)

            robot_cfg = env_cfg.scene.robot
            jd = getattr(robot_cfg.spawn, "joint_drive_props", None)
            ensure = bool(getattr(jd, "ensure_drives_exist", False)) if jd is not None else False
            cfg_ok = all(isinstance(v, ImplicitActuatorCfg) for v in robot_cfg.actuators.values())
            use_na = bool(getattr(env_cfg.sim, "use_newton_actuators", False))
            robot = env.env.scene["robot"]
            rt_ok = all(isinstance(v, ImplicitActuator) for v in robot.actuators.values())
            if fabric.is_global_zero:
                print(
                    f"[DISTILL_RECIPE] ensure_drives_exist={ensure} use_newton_actuators={use_na} "
                    f"cfg_Implicit={cfg_ok} runtime_Implicit={rt_ok}",
                    flush=True,
                )
            if not (ensure and cfg_ok and rt_ok and use_na):
                raise RuntimeError(
                    f"recipe misaligned: ensure={ensure} cfg_ok={cfg_ok} rt_ok={rt_ok} use_na={use_na}"
                )

            viz_capture = None
            if viz_enabled and fabric.is_global_zero:
                import numpy as np

                cam_offset = np.array([2.8, 2.8, 1.4], dtype=np.float64)
                mp4_path = out / "distill_train_viz.mp4"
                record_every_i = max(1, int(args.record_every))
                mp4_fps = float(
                    os.environ.get(
                        "PHI0_DISTILL_MP4_FPS",
                        str(_mp4_fps_for_record_every(record_every_i)),
                    )
                )
                flush_every = int(os.environ.get("PHI0_DISTILL_MP4_FLUSH", "0") or "0")
                max_frames = max(
                    1, int(os.environ.get("PHI0_DISTILL_MP4_MAX_FRAMES", "100"))
                )

                def _viz_capture() -> None:
                    nonlocal viz_written
                    if viz_written or len(viz_frames) >= max_frames:
                        return
                    robot = inner.scene["robot"]
                    root = (
                        robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                    )
                    eye = root + cam_offset
                    look = root + np.array([0.0, 0.0, 0.35], dtype=np.float64)
                    cap = getattr(getattr(inner, "video_recorder", None), "_capture", None)
                    if cap is not None and hasattr(cap, "update_camera"):
                        cap.update_camera(tuple(eye.tolist()), tuple(look.tolist()))
                    rgb = inner.render()
                    if rgb is not None:
                        viz_frames.append(np.asarray(rgb))
                        if flush_every > 0 and len(viz_frames) % flush_every == 0:
                            _write_distill_mp4(viz_frames, mp4_path, fps=mp4_fps)
                            print(
                                f"[DISTILL_FABRIC] mp4 flush {mp4_path} "
                                f"frames={len(viz_frames)}/{max_frames} fps={mp4_fps:g} "
                                f"record_every={record_every_i} sim_hz={_SIM_HZ:g}",
                                flush=True,
                            )
                        if len(viz_frames) >= max_frames:
                            _write_distill_mp4(viz_frames, mp4_path, fps=mp4_fps)
                            viz_written = True
                            print(
                                f"[DISTILL_FABRIC] mp4 cap hit frames={len(viz_frames)} "
                                f"→ {mp4_path}",
                                flush=True,
                            )

                viz_capture = _viz_capture
                backend = getattr(getattr(inner, "video_recorder", None), "_backend", None)
                if backend != "newton_gl":
                    raise RuntimeError(f"--record_train_mp4 expected newton_gl, got {backend!r}")

            result = _run_distill(env=env, atm=atm, atm_ll=atm_ll, viz_capture=viz_capture)

    if fabric.is_global_zero and viz_frames and not viz_written:
        mp4 = out / "distill_train_viz.mp4"
        re_i = max(1, int(args.record_every))
        fps = float(
            os.environ.get("PHI0_DISTILL_MP4_FPS", str(_mp4_fps_for_record_every(re_i)))
        )
        _write_distill_mp4(viz_frames, mp4, fps=fps)
        print(
            f"[DISTILL_FABRIC] wrote {mp4} frames={len(viz_frames)} "
            f"fps={fps:g} record_every={re_i} sim_hz={_SIM_HZ:g}",
            flush=True,
        )

    if fabric.is_global_zero:
        n = int(result.get("steps_done") or result.get("num_chunks") or 0)
        elapsed = time.perf_counter() - t0
        print(
            f"[DISTILL_FABRIC] done steps={n} epochs={result.get('epochs')} "
            f"loss_last={result.get('loss_last')} "
            f"elapsed_s={elapsed:.1f} out={out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
