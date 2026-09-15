#!/usr/bin/env python3
"""Unified Fabric distill entry — ``--simulator {physx,newton}``.

Shell must call ``sonic_apply_simulator`` before torchrun so conda/ISAACLAB match.
This process only dispatches to the existing backend entry + asserts recipe.

Phase-2: ``teacher_z_source=hybrid`` works on both sims (forces ``joint_pd``).
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

_PHI0 = Path(__file__).resolve().parents[2]
_LIB = _PHI0 / "scripts" / "lib"
for _p in (_PHI0 / "src", _LIB, _PHI0 / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _parse() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--simulator",
        type=str,
        default=os.environ.get("PHI0_SIMULATOR", os.environ.get("SIMULATOR", "newton")),
        choices=("physx", "newton", "isaaclab", "lab2", "lab3", "isaaclab3"),
        help="physx=gear sonic native (phi-0-wbc); newton=IsaacLab 3.0 (default).",
    )
    p.add_argument(
        "--teacher_z_source",
        type=str,
        default=os.environ.get("TEACHER_Z_SOURCE", "online"),
        help="online|hybrid; hybrid forces expert_drive=joint_pd on both sims.",
    )
    return p.parse_known_args()


def main() -> None:
    args, rest = _parse()
    from phi0.online.sim_backend import (
        apply_recipe_env,
        assert_runtime_matches_recipe,
        get_distill_sim_recipe,
        normalize_simulator,
        resolve_expert_drive,
    )

    sim = normalize_simulator(args.simulator)
    # teacher_z may also appear later in rest for newton fabric; peek env + flag.
    tz = str(args.teacher_z_source or "online")
    if "--teacher_z_source" in rest:
        i = rest.index("--teacher_z_source")
        if i + 1 < len(rest):
            tz = rest[i + 1]
    tz = str(tz or "online").strip().lower()

    recipe = get_distill_sim_recipe(sim)
    apply_recipe_env(recipe)
    assert_runtime_matches_recipe(recipe)
    drive = resolve_expert_drive(
        simulator=sim, teacher_z_source=tz, expert_drive=None
    )
    os.environ["PHI0_DISTILL_EXPERT_DRIVE"] = drive
    if tz == "hybrid":
        os.environ["TEACHER_Z_SOURCE"] = "hybrid"
    print(
        f"[distill_entry] simulator={recipe.name} expert_drive={drive} "
        f"teacher_z={tz} needs_kit={recipe.needs_kit}",
        flush=True,
    )

    # Strip our flags from rest if duplicated.
    cleaned: list[str] = []
    skip_next = False
    for i, a in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        if a in ("--simulator", "--teacher_z_source") and i + 1 < len(rest):
            skip_next = True
            continue
        if a.startswith("--simulator=") or a.startswith("--teacher_z_source="):
            continue
        cleaned.append(a)

    if recipe.name == "newton":
        argv = [str(_PHI0 / "scripts" / "newton_boneseed_distill_fabric.py")]
        if tz and tz != "online":
            argv += ["--teacher_z_source", tz]
        argv += cleaned
        sys.argv = argv
        runpy.run_path(str(_PHI0 / "scripts" / "newton_boneseed_distill_fabric.py"), run_name="__main__")
        return

    # physx: AppLauncher path (gear sonic native). Hybrid → joint_pd already set.
    argv = [str(_PHI0 / "scripts" / "train_online_distill_fabric.py")] + cleaned
    sys.argv = argv
    runpy.run_path(str(_PHI0 / "scripts" / "train_online_distill_fabric.py"), run_name="__main__")


if __name__ == "__main__":
    main()
