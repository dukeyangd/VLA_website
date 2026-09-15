#!/usr/bin/env python3
"""Fabric multi-GPU online distill — ProtoMotions launch order + SLURM GPU model.

ProtoMotions ``train_agent.py`` order::

    1. argparse (no torch)
    2. ``from isaaclab.app import AppLauncher``   # BEFORE torch
    3. import torch / Fabric
    4. ``Fabric(...).launch()``
    5. ``AppLauncher({headless, device=fabric.device, distributed=...})``
    6. train with ``fabric.setup`` / ``fabric.backward``

ProtoMotions multi-GPU is SLURM ``ntasks-per-node=ngpu`` — **one GPU visible
per process** (see ``train_slurm.py``). Non-SLURM adaptation: remap
``CUDA_VISIBLE_DEVICES`` to a single id from torchrun ``LOCAL_RANK`` before any
Isaac/torch import, then ``Fabric(devices=1)`` so each rank mirrors a SLURM task.

No Kit filelock / hold sleep — ProtoMotions does not use them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _remap_cvd_one_gpu_per_rank() -> None:
    """SLURM-equivalent: each process sees exactly one physical GPU as cuda:0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," not in cvd:
        return
    if "LOCAL_RANK" not in os.environ:
        return
    gpus = [g.strip() for g in cvd.split(",") if g.strip()]
    local = int(os.environ["LOCAL_RANK"])
    if local >= len(gpus):
        raise RuntimeError(f"LOCAL_RANK={local} but CVD has {len(gpus)} ids: {cvd}")
    os.environ["PHI0_FABRIC_PHYS_GPU"] = gpus[local]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[local]
    # After remap only cuda:0 exists; match SLURM one-GPU task before Fabric.
    os.environ["LOCAL_RANK"] = "0"


def _parse() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--groot-root",
        default=os.environ.get(
            "GR00T_ROOT", str(Path(__file__).resolve().parents[2] / "subpackages")
        ),
    )
    # Kept for CLI compat; default 0 (ProtoMotions has no stagger).
    parser.add_argument(
        "--stagger-s",
        type=float,
        default=float(os.environ.get("PHI0_FABRIC_STAGGER_S", "0") or "0"),
    )
    parser.add_argument("--headless", action="store_true", default=True)
    # --ngpu kept for CLI compat; real world size comes from torchrun WORLD_SIZE.
    parser.add_argument("--ngpu", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    return parser.parse_known_args()


def main() -> None:
    # Must run before isaaclab / torch (ProtoMotions + SLURM one-GPU-per-task).
    _remap_cvd_one_gpu_per_rank()

    args, hydra_argv = _parse()
    if hydra_argv and hydra_argv[0] == "--":
        hydra_argv = hydra_argv[1:]

    groot = os.path.abspath(args.groot_root)
    os.chdir(groot)
    if groot not in sys.path:
        sys.path.insert(0, groot)

    # --- ProtoMotions: AppLauncher class BEFORE torch ---
    from isaaclab.app import AppLauncher  # noqa: WPS433

    # Kit-less Lab3/Newton: AppLauncher.is_isaac_sim_version_5() references
    # ``isaacsim`` without importing it when the pip package is absent.
    def _is_sim_ver_5_safe(self) -> bool:  # noqa: ANN001
        if hasattr(self, "_is_sim_ver_5"):
            return bool(self._is_sim_ver_5)
        try:
            import isaacsim  # noqa: F401

            version_path = os.path.abspath(
                os.path.join(os.path.dirname(isaacsim.__file__), "../../VERSION")
            )
            if os.path.isfile(version_path):
                with open(version_path) as f:
                    ver = f.readline().strip()
                    self._is_sim_ver_5 = ver.startswith("5")
                    return bool(self._is_sim_ver_5)
            from importlib.metadata import version as pkg_version

            self._is_sim_ver_5 = pkg_version("isaacsim").startswith("5")
        except Exception:
            self._is_sim_ver_5 = False
        return bool(self._is_sim_ver_5)

    AppLauncher.is_isaac_sim_version_5 = _is_sim_ver_5_safe  # type: ignore[method-assign]

    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy

    # devices=1 per process (SLURM one-GPU task). Under torchrun, Lightning requires
    # devices * num_nodes == WORLD_SIZE — set num_nodes=WORLD_SIZE as the non-SLURM
    # stand-in for ntasks (ProtoMotions train_slurm: ntasks-per-node=ngpu).
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
        num_nodes=world,
        # broadcast_buffers=False: Isaac step time varies per rank; default
        # per-forward buffer broadcast lets a fast rank enter the next forward
        # while others are still in clamp/sim → NCCL spin (GPU util 100%, hang).
        # find_unused_parameters=True: ACT/lang-cache leaves unused towers some steps.
        strategy=DDPStrategy(
            find_unused_parameters=True,
            broadcast_buffers=False,
        ),
        precision="32-true",
    )
    fabric.launch()

    os.environ["PHI0_FABRIC"] = "1"
    os.environ["SHARD_RANK"] = str(fabric.global_rank)
    os.environ["SHARD_WORLD"] = str(fabric.world_size)

    from phi0.online.fabric_runtime import set_fabric, set_simulation_app

    set_fabric(fabric)

    # Optional stagger (default 0). ProtoMotions does not stagger AppLauncher.
    stagger = max(0.0, float(args.stagger_s))
    if stagger > 0 and fabric.global_rank > 0:
        import time

        time.sleep(stagger * float(fabric.global_rank))

    # --- ProtoMotions §AppLauncher after Fabric.launch (no filelock / hold) ---
    # ProtoMotions SLURM sets distributed=True when world_size>1. With torchrun +
    # one-GPU CVD remap, that path fails here (gpu.foundation "No device could be
    # created"). Each rank already has a private GPU → single-process Kit; student
    # AllReduce still goes through Fabric/DDP. Opt-in: PHI0_FABRIC_ISAAC_DISTRIBUTED=1.
    app_launcher_flags: dict = {
        "headless": bool(args.headless),
        "device": str(fabric.device),
    }
    # PhysX train mp4 needs camera extensions in Kit (see OnlineDistillCallback).
    record_mp4 = str(os.environ.get("PHI0_DISTILL_RECORD_MP4", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) or str(os.environ.get("RECORD_TRAIN_MP4", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Experience: only if PHI0_ISAAC_EXPERIENCE set. Otherwise AppLauncher picks
    # isaacsim_4_5/{headless,headless.rendering}.kit from headless/enable_cameras.
    # Do NOT hardcode apps/isaaclab.python*.kit (wrong tree; hangs at renderer).
    exp = str(os.environ.get("PHI0_ISAAC_EXPERIENCE", "")).strip()
    if exp:
        app_launcher_flags["experience"] = exp
    if record_mp4:
        # AppLauncher → …/isaacsim_4_5/isaaclab.python.headless.rendering.kit
        app_launcher_flags["enable_cameras"] = True
        os.environ["PHI0_DISTILL_RECORD_MP4"] = "1"
        os.environ["ENABLE_CAMERAS"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = str(fabric.global_rank)
    os.environ["WORLD_SIZE"] = str(fabric.world_size)
    if fabric.world_size > 1 and os.environ.get("PHI0_FABRIC_ISAAC_DISTRIBUTED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        app_launcher_flags["distributed"] = True
        os.environ["LOCAL_RANK"] = str(fabric.local_rank)

    print(  # noqa: T201
        f"[fabric] AppLauncher rank={fabric.global_rank}/{fabric.world_size} "
        f"device={fabric.device} phys_gpu={os.environ.get('PHI0_FABRIC_PHYS_GPU')} "
        f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} flags={app_launcher_flags}",
        flush=True,
    )
    app_launcher = AppLauncher(app_launcher_flags)
    set_simulation_app(app_launcher.app)
    os.environ["PHI0_FABRIC_APP_READY"] = "1"

    if record_mp4:
        # Headless AppLauncher still needs rgb_array + enable_cameras on env.
        # GR00T create_manager_env sets render_mode=None when headless — patch it.
        import gear_sonic.train_agent_trl as _tat  # noqa: WPS433
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.utils.common import custom_instantiate
        from isaaclab.envs import ManagerBasedRLEnv

        def _create_manager_env_rgb(config, device, args_cli):  # noqa: ANN001
            # Prefer hydra ``render_results=true`` so ModularTrackingEnvCfg spawns
            # eval_camera (Gear Sonic's working 4.5 path). Fall back if missing.
            me = getattr(config, "manager_env", None)
            cfg_dict = getattr(me, "config", None) if me is not None else None
            if isinstance(cfg_dict, dict):
                cfg_dict["render_results"] = True
                cfg_dict.setdefault("render_width", 960)
                cfg_dict.setdefault("render_height", 540)
                # Viewport-only ego cam not needed; eval_camera is the viz source.
                cfg_dict["enable_cameras"] = False
            # Keep PhysX robot USD/URDF path (do NOT set PHI0_ISAAC_BACKEND=newton).
            env_instance_cfg = custom_instantiate(config.manager_env)
            env_instance_cfg.seed = config.seed
            env_instance_cfg.sim.device = device
            env_instance_cfg.config["headless"] = args_cli.headless
            env_instance_cfg.config["render_results"] = True
            env_instance_cfg.config["enable_cameras"] = False
            env_instance_cfg.config.setdefault("render_width", 960)
            env_instance_cfg.config.setdefault("render_height", 540)
            # Replace GroundPlane USD grid with solid MeshBox + PreviewSurface.
            # (plane visual_material only tints grid color — still noisy Nucleus USD.)
            try:
                import isaaclab.sim as sim_utils
                from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
                from isaaclab.terrains.trimesh import MeshBoxTerrainCfg

                phys = None
                old = getattr(getattr(env_instance_cfg, "scene", None), "terrain", None)
                if old is not None:
                    phys = getattr(old, "physics_material", None)
                _gen = TerrainGeneratorCfg(
                    size=(8.0, 8.0),
                    border_width=20.0,
                    num_rows=1,
                    num_cols=1,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    slope_threshold=0.75,
                    use_cache=False,
                    sub_terrains={
                        "flat": MeshBoxTerrainCfg(
                            proportion=1.0,
                            box_height_range=(0.0, 0.0),
                            platform_width=8.0,
                            double_box=False,
                        )
                    },
                )
                mat = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.45, 0.45, 0.48),
                    roughness=1.0,
                    metallic=0.0,
                )
                kw = dict(
                    prim_path="/World/ground",
                    terrain_type="generator",
                    terrain_generator=_gen,
                    visual_material=mat,
                    max_init_terrain_level=10,
                )
                if phys is not None:
                    kw["physics_material"] = phys
                env_instance_cfg.scene.terrain = TerrainImporterCfg(**kw)
                env_instance_cfg.config["terrain_type"] = "generator"
                # Soften lights — MeshBox + DistantLight=3000 fireflies / white grain.
                try:
                    light = getattr(env_instance_cfg.scene, "light", None)
                    if light is not None and getattr(light, "spawn", None) is not None:
                        if hasattr(light.spawn, "intensity"):
                            light.spawn.intensity = 800.0
                    sky = getattr(env_instance_cfg.scene, "sky_light", None)
                    if sky is not None and getattr(sky, "spawn", None) is not None:
                        if hasattr(sky.spawn, "intensity"):
                            sky.spawn.intensity = 400.0
                except Exception as le:  # noqa: BLE001
                    print(f"[fabric] WARN light soften failed: {le}", flush=True)
                print(
                    f"[fabric] terrain=generator/MeshBox visual_material="
                    f"{type(env_instance_cfg.scene.terrain.visual_material).__name__} "
                    f"(expect PreviewSurfaceCfg; not GroundPlane USD)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[fabric] WARN flat MeshBox terrain not set: {e}", flush=True)
            # Prefer RayTracedLighting + disable noisy secondary effects for eval_camera.
            try:
                import carb

                s = carb.settings.get_settings()
                s.set("/rtx/rendermode", "RayTracedLighting")
                s.set("/rtx/pathtracing/spp", 1)
                s.set("/rtx/indirectDiffuse/enabled", False)
                s.set("/rtx/ambientOcclusion/enabled", False)
                s.set("/rtx/reflections/enabled", False)
                s.set("/rtx/raytracing/cached/enabled", True)
                s.set("/rtx/post/aa/op", 2)  # TAA
                print("[fabric] rtx rendermode=RayTracedLighting (AO/refl off, TAA)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[fabric] WARN rtx rendermode not set: {e}", flush=True)
            # Isaac Sim 4.5: Fabric camera poses are broken; GR00T forces this off
            # when render_results (see ModularTrackingEnvCfg + recorders.py).
            if hasattr(env_instance_cfg.sim, "use_fabric"):
                env_instance_cfg.sim.use_fabric = False
            # Still set ViewerCfg for kit; primary frames come from eval_camera.
            try:
                from isaaclab.envs.common import ViewerCfg

                env_instance_cfg.viewer = ViewerCfg(
                    eye=(2.8, 2.8, 1.4),
                    lookat=(0.0, 0.0, 0.9),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[fabric] WARN ViewerCfg not set: {e}", flush=True)
            # rgb_array kept so env.render works as fallback; prefer eval_camera.
            env = ManagerBasedRLEnv(cfg=env_instance_cfg, render_mode="rgb_array")
            return ManagerEnvWrapper(env, env_instance_cfg.config)

        _tat.create_manager_env = _create_manager_env_rgb  # type: ignore[method-assign]
        if fabric.is_global_zero:
            print(
                "[fabric] PHI0_DISTILL_RECORD_MP4=1 → rgb_array+eval_camera "
                f"terrain=MeshBox ISAACLAB_ASSET_ROOT={os.environ.get('ISAACLAB_ASSET_ROOT', '')}",
                flush=True,
            )

    import runpy

    eval_py = os.path.join(groot, "gear_sonic", "eval_agent_trl.py")
    sys.argv = [eval_py] + hydra_argv
    if fabric.is_global_zero:
        print(  # noqa: T201
            f"[fabric] → eval_agent hydra_argv={len(hydra_argv)} cwd={groot}",
            flush=True,
        )
    runpy.run_path(eval_py, run_name="__main__")


if __name__ == "__main__":
    main()
