#!/usr/bin/env python3
"""Scan BoneSEED episodes → blacklist JSON (static and/or Isaac expert fall).

Examples:
  # Fast CPU: ep0 pelvis z / xy-reset / nonfinite
  python tools/data/scan_boneseed_episode_blacklist.py --mode static \\
    --out meta/boneseed_episode_blacklist.json

  # Isaac expert H-roll from each ep frame0 (needs free GPU; stop train first)
  python tools/data/scan_boneseed_episode_blacklist.py --mode isaac \\
    --num_envs 512 --horizon 8 --max_row_groups 1 \\
    --out meta/boneseed_episode_blacklist_isaac.json --headless --visualizer none

  # Multi-GPU: each rank scans specs[rank::world], then merge:
  python tools/data/scan_boneseed_episode_blacklist.py --mode merge \\
    --out meta/final_bl.json --whitelist_out meta/final_wl.json --shard_world 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PHI0_ISAAC_BACKEND"] = os.environ.get("PHI0_ISAAC_BACKEND", "newton")
os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

_PHI0 = Path(__file__).resolve().parents[1]
_LIB = _PHI0 / "scripts" / "lib"
for _p in (_PHI0 / "src", _LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_REF = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)


def _run_static(args: argparse.Namespace) -> None:
    from phi0.online.expert_fall_scan import scan_static_corpus, write_blacklist_json

    t0 = time.perf_counter()
    result = scan_static_corpus(
        args.ref_root,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        max_row_groups=args.max_row_groups,
    )
    note = (
        f"static ep0 filter z=[{args.z_min},{args.z_max}] xy_atol=1e-2; "
        f"ref={args.ref_root} elapsed_s={time.perf_counter() - t0:.1f}"
    )
    out = Path(args.out)
    write_blacklist_json(out, result, note=note)
    print(
        f"[OK] static scanned={result.n_scanned} blacklist={len(result.blacklist)} "
        f"→ {out}",
        flush=True,
    )


def _run_merge(args: argparse.Namespace) -> None:
    from phi0.online.expert_fall_scan import (
        merge_gate_shard_jsons,
        shard_out_path,
        write_blacklist_json,
        write_whitelist_json,
    )

    world = int(args.shard_world)
    if world < 1:
        raise SystemExit(f"--shard_world must be >=1, got {world}")
    out = Path(args.out)
    wl_out = Path(
        args.whitelist_out
        if args.whitelist_out
        else str(out.with_name(out.stem + "_whitelist.json"))
    )
    bl_shards = [shard_out_path(out, r, world) for r in range(world)]
    wl_shards = [shard_out_path(wl_out, r, world) for r in range(world)]
    missing = [str(p) for p in bl_shards + wl_shards if not p.is_file()]
    if missing:
        raise SystemExit("missing shard files:\n  " + "\n  ".join(missing))

    merged = merge_gate_shard_jsons(
        blacklist_shards=bl_shards, whitelist_shards=wl_shards
    )
    note = (
        f"merged shard_world={world} shards_bl={[p.name for p in bl_shards]} "
        f"n_scanned={merged.n_scanned} deny={len(merged.blacklist)} "
        f"allow={len(merged.whitelist)}"
    )
    parent = None
    tier = None
    for p in bl_shards:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("tier") is not None:
            tier = int(obj["tier"])
        if obj.get("parent_allowlist"):
            parent = str(obj["parent_allowlist"])
        if "tier=2" in str(obj.get("note") or ""):
            tier = tier or 2
        if parent:
            break
    if parent:
        note = f"tier={tier or 2} second_wash_from_whitelist parent_allowlist={parent}; " + note
    extra: dict = {
        "shards_merged": world,
        "shard_blacklist_files": [str(p) for p in bl_shards],
        "shard_whitelist_files": [str(p) for p in wl_shards],
    }
    if tier is not None:
        extra["tier"] = tier
    if parent:
        extra["parent_allowlist"] = parent
    write_blacklist_json(out, merged, note=note, extra=extra)
    write_whitelist_json(wl_out, merged, note=note, extra=extra)
    print(
        f"[OK] merge scanned={merged.n_scanned} "
        f"blacklist={len(merged.blacklist)} → {out} "
        f"whitelist={len(merged.whitelist)} → {wl_out}",
        flush=True,
    )


def _run_isaac(args: argparse.Namespace) -> None:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    from isaaclab_tasks.utils.sim_launcher import launch_simulation
    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.utils import config_utils

    from newton_gate_common import fill_obs_dims, load_atm_policy, load_full_cfg
    from phi0.online.expert_fall_scan import (
        BlacklistScanResult,
        scan_isaac_expert_falls_on_ref,
        scan_static_corpus,
        shard_out_path,
        write_blacklist_json,
        write_whitelist_json,
    )
    from phi0.online.isaac_sim import assert_env_control_hz, enable_direct_latent_atm
    from phi0.online.isaac_loop import load_episode_allowlist
    from phi0.online.lazy_ref import list_boneseed_row_groups, load_row_group_ref
    from phi0.online.lazy_ref import filter_row_groups_by_episode_allowlist

    config_utils.register_rl_resolvers()

    # re-parse with launcher args (AppLauncher)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="isaac")
    parser.add_argument("--ref_root", type=str, default=str(DEFAULT_REF))
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--roll_steps",
        type=int,
        default=None,
        help="expert-roll steps per ep; default=horizon; <=0 or --full_episode = entire ep",
    )
    parser.add_argument(
        "--full_episode",
        action="store_true",
        help="RSI frame0 and expert-roll to ep end (sets roll_steps=0)",
    )
    parser.add_argument("--fall_thr", type=float, default=0.35)
    parser.add_argument(
        "--xy_thr",
        type=float,
        default=1.0,
        help="planar L2 ||xy_sim_local-xy_fk||; <=0 disables (default 1.0m)",
    )
    parser.add_argument(
        "--fall_via",
        choices=("expert", "pelvis", "root_z"),
        default="pelvis",
        help="pelvis=|z_sim-z_fk| (dataset pelvis z); expert=+env_done; root_z=abs floor",
    )
    parser.add_argument("--z_min", type=float, default=0.40)
    parser.add_argument("--z_max", type=float, default=1.20)
    parser.add_argument("--max_row_groups", type=int, default=None)
    parser.add_argument(
        "--episode_allowlist",
        type=str,
        default=None,
        help="JSON allowlist; only scan these episode_index values",
    )
    parser.add_argument(
        "--whitelist_out",
        type=str,
        default=None,
        help="Write pass (no-fall to end) allowlist JSON; default=<out>_whitelist.json",
    )
    parser.add_argument("--merge_static", action="store_true", default=True)
    parser.add_argument("--no_merge_static", action="store_false", dest="merge_static")
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=0,
        help="RG shard rank in [0, shard_world); specs[rank::world]",
    )
    parser.add_argument(
        "--shard_world",
        type=int,
        default=1,
        help="Number of RG shards (1 = single-process, no .shard suffix)",
    )
    add_launcher_args(parser)
    parser.set_defaults(headless=True)
    args_cli, hydra_args = setup_preset_cli(parser)
    if hasattr(args_cli, "visualizer"):
        args_cli.visualizer = []
    sys.argv = [sys.argv[0]] + hydra_args

    shard_rank = int(getattr(args_cli, "shard_rank", 0) or 0)
    shard_world = int(getattr(args_cli, "shard_world", 1) or 1)
    if shard_world < 1:
        raise SystemExit(f"--shard_world must be >=1, got {shard_world}")
    if not (0 <= shard_rank < shard_world):
        raise SystemExit(
            f"--shard_rank {shard_rank} out of range for shard_world={shard_world}"
        )
    tag = f"r{shard_rank}/{shard_world}"

    if bool(getattr(args_cli, "full_episode", False)):
        args_cli.roll_steps = 0
    roll_steps = args_cli.roll_steps
    roll_desc = (
        "full_episode"
        if roll_steps is not None and int(roll_steps) <= 0
        else str(roll_steps if roll_steps is not None else args_cli.horizon)
    )

    conf = load_full_cfg(args_cli.num_envs)
    env_cfg = custom_instantiate(conf.manager_env)
    env_cfg.seed = int(shard_rank)
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if hasattr(env_cfg, "config") and isinstance(env_cfg.config, dict):
        env_cfg.config["headless"] = True

    merged = BlacklistScanResult()
    if bool(args_cli.merge_static):
        print(f"[isaac {tag}] prefilter static…", flush=True)
        st = scan_static_corpus(
            args_cli.ref_root,
            z_min=float(args_cli.z_min),
            z_max=float(args_cli.z_max),
            max_row_groups=args_cli.max_row_groups,
        )
        merged.blacklist.extend(st.blacklist)
        merged.records.extend(st.records)

    allow_eps = load_episode_allowlist(args_cli.episode_allowlist)
    deny_static = set(merged.blacklist)
    if str(args_cli.fall_via).lower().strip() != "pelvis":
        raise SystemExit(
            f"gate requires --fall_via pelvis (got {args_cli.fall_via!r})"
        )

    out_base = Path(args_cli.out)
    wl_base = Path(
        args_cli.whitelist_out
        if getattr(args_cli, "whitelist_out", None)
        else str(out_base.with_name(out_base.stem + "_whitelist.json"))
    )
    out = shard_out_path(out_base, shard_rank, shard_world)
    wl_out = shard_out_path(wl_base, shard_rank, shard_world)
    ckpt_path = out.with_name(out.stem + ".partial.json")

    def _atomic_write_json(path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _checkpoint(note: str) -> None:
        merged.finalize()
        write_blacklist_json(out, merged, note=note + " (partial)")
        write_whitelist_json(wl_out, merged, note=note + " (partial)")
        _atomic_write_json(
            ckpt_path,
            {
                "done_episode_index": sorted(
                    set(merged.blacklist) | set(merged.whitelist)
                ),
                "n_scanned": int(merged.n_scanned),
                "n_blacklisted": len(merged.blacklist),
                "n_whitelisted": len(merged.whitelist),
                "note": note,
                "shard_rank": shard_rank,
                "shard_world": shard_world,
            },
        )

    done_eps: set[int] = set(deny_static)
    if ckpt_path.is_file():
        try:
            prev = json.loads(ckpt_path.read_text(encoding="utf-8"))
            done_eps |= {int(x) for x in (prev.get("done_episode_index") or [])}
            if out.is_file():
                bl_prev = json.loads(out.read_text(encoding="utf-8"))
                merged.blacklist.extend(int(x) for x in (bl_prev.get("episode_index") or []))
                merged.records.extend(bl_prev.get("records") or [])
            if wl_out.is_file():
                wl_prev = json.loads(wl_out.read_text(encoding="utf-8"))
                merged.whitelist.extend(int(x) for x in (wl_prev.get("episode_index") or []))
            merged.n_scanned = int(prev.get("n_scanned") or merged.n_scanned)
            merged.finalize()
            print(
                f"[isaac {tag}] resume ckpt {ckpt_path.name} done={len(done_eps)} "
                f"deny={len(merged.blacklist)} allow={len(merged.whitelist)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[isaac {tag}] ckpt load failed ({exc}); starting fresh", flush=True)
            done_eps = set(deny_static)

    t0 = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        inner = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env = ManagerEnvWrapper(inner, env_cfg.config)
        device = str(env.device)
        fill_obs_dims(env)
        atm = load_atm_policy(conf, env, device)
        enable_direct_latent_atm(env, atm)
        assert_env_control_hz(env, expected_hz=50.0)

        specs = list_boneseed_row_groups(args_cli.ref_root)
        if allow_eps is not None:
            n_before = len(specs)
            specs = filter_row_groups_by_episode_allowlist(specs, allow_eps)
            print(
                f"[isaac {tag}] allowlist n_eps={len(allow_eps)} "
                f"RG {len(specs)}/{n_before} src={args_cli.episode_allowlist}",
                flush=True,
            )
            if not specs:
                raise RuntimeError(
                    f"allowlist matched 0 row-groups (src={args_cli.episode_allowlist})"
                )
        if args_cli.max_row_groups is not None:
            specs = specs[: int(args_cli.max_row_groups)]

        n_rg_all = len(specs)
        if shard_world > 1:
            specs = specs[shard_rank::shard_world]
        print(
            f"[isaac {tag}] RG shard {len(specs)}/{n_rg_all} "
            f"out={out.name} wl={wl_out.name}",
            flush=True,
        )

        for i, spec in enumerate(specs):
            print(
                f"[isaac {tag}] load {i + 1}/{len(specs)} "
                f"{spec.path.name} rg{spec.rg_index}",
                flush=True,
            )
            ref = load_row_group_ref(spec, root=args_cli.ref_root)
            part = scan_isaac_expert_falls_on_ref(
                env=env,
                device=device,
                ref=ref,
                horizon=int(args_cli.horizon),
                roll_steps=roll_steps,
                fall_thr=float(args_cli.fall_thr),
                fall_via=str(args_cli.fall_via),
                root_z_min=float(args_cli.z_min),
                xy_thr=float(getattr(args_cli, "xy_thr", 1.0)),
                episode_allowlist=allow_eps,
                episode_blacklist=done_eps or None,
            )
            for eid in part.blacklist:
                if eid not in done_eps:
                    merged.blacklist.append(eid)
            for eid in part.whitelist:
                if eid not in done_eps:
                    merged.whitelist.append(eid)
            merged.records.extend(part.records)
            merged.n_scanned += part.n_scanned
            done_eps |= set(part.blacklist) | set(part.whitelist)
            print(
                f"[isaac {tag}] rg done fall_new="
                f"{len(part.blacklist)} "
                f"ok_new={len(part.whitelist)} "
                f"total_deny={len(set(merged.blacklist))} "
                f"total_allow={len(set(merged.whitelist))}",
                flush=True,
            )
            note_partial = (
                f"isaac {tag} expert roll_steps={roll_desc} "
                f"H_bounds={args_cli.horizon} thr={args_cli.fall_thr} "
                f"xy_thr={getattr(args_cli, 'xy_thr', 1.0)} "
                f"via={args_cli.fall_via} num_envs={args_cli.num_envs} "
                f"merge_static={args_cli.merge_static} "
                f"z=[{args_cli.z_min},{args_cli.z_max}] "
                f"rg={i + 1}/{len(specs)} "
                f"elapsed_s={time.perf_counter() - t0:.1f}"
            )
            _checkpoint(note_partial)

        close = getattr(env, "close", None) or getattr(inner, "close", None)
        if callable(close):
            close()

    merged.finalize()
    note = (
        f"isaac {tag} expert roll_steps={roll_desc} "
        f"H_bounds={args_cli.horizon} thr={args_cli.fall_thr} "
        f"xy_thr={getattr(args_cli, 'xy_thr', 1.0)} via={args_cli.fall_via} "
        f"num_envs={args_cli.num_envs} merge_static={args_cli.merge_static} "
        f"z=[{args_cli.z_min},{args_cli.z_max}] elapsed_s={time.perf_counter() - t0:.1f}"
    )
    extra: dict = {"shard_rank": shard_rank, "shard_world": shard_world}
    if allow_eps is not None:
        note = (
            f"tier=3 third_wash_xy1m_from_wl2 "
            f"parent_allowlist={args_cli.episode_allowlist} n_parent={len(allow_eps)}; "
            + note
        )
        extra["tier"] = 3
        extra["parent_allowlist"] = str(args_cli.episode_allowlist)
    write_blacklist_json(out, merged, note=note, extra=extra)
    write_whitelist_json(wl_out, merged, note=note, extra=extra)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    print(
        f"[OK] isaac {tag} scanned={merged.n_scanned} "
        f"blacklist={len(merged.blacklist)} → {out} "
        f"whitelist={len(merged.whitelist)} → {wl_out}",
        flush=True,
    )


def main() -> None:
    # lightweight pre-parse for mode (isaac re-parses with launcher)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--mode", choices=("static", "isaac", "merge"), default="static"
    )
    pre.add_argument("--ref_root", type=str, default=str(DEFAULT_REF))
    pre.add_argument("--out", type=str, default="")
    pre.add_argument("--z_min", type=float, default=0.40)
    pre.add_argument("--z_max", type=float, default=1.20)
    pre.add_argument("--max_row_groups", type=int, default=None)
    pre.add_argument("--num_envs", type=int, default=512)
    pre.add_argument("--horizon", type=int, default=8)
    pre.add_argument("--roll_steps", type=int, default=None)
    pre.add_argument("--full_episode", action="store_true")
    pre.add_argument("--whitelist_out", type=str, default=None)
    pre.add_argument("--episode_allowlist", type=str, default=None)
    pre.add_argument("--fall_thr", type=float, default=0.35)
    pre.add_argument("--xy_thr", type=float, default=1.0)
    pre.add_argument("--shard_rank", type=int, default=0)
    pre.add_argument("--shard_world", type=int, default=1)
    args, _rest = pre.parse_known_args()
    if not args.out:
        raise SystemExit("--out is required")
    if args.mode == "static":
        ns = argparse.Namespace(**vars(args))
        _run_static(ns)
    elif args.mode == "merge":
        _run_merge(args)
    else:
        _run_isaac(args)


if __name__ == "__main__":
    main()
