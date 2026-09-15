#!/usr/bin/env python3
"""qpos_student closed-loop on Newton → Isaac Lab 3 Newton-GL video.

Proves action→sim / state←sim adapters: frames come from NewtonManager state
after ``step_joint_qpos_batch`` (scatter + PD), not MuJoCo FK / Phi-0-wpy.

Requires: Phi-0-wbc-newton-wpy + IsaacLab-3.0, PHI0_ISAAC_BACKEND=newton.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["PHI0_ISAAC_BACKEND"] = "newton"
os.environ.setdefault("PHI0_ONNX_ENCODE_GPU", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

# Refuse the old PhysX / MuJoCo conda by name.
_py = Path(sys.executable).resolve()
if "Phi-0-wpy" in str(_py) and "newton" not in str(_py).lower():
    raise SystemExit(
        f"refusing interpreter {_py}: use Phi-0-wbc-newton-wpy, not Phi-0-wpy"
    )

_PHI0 = Path(__file__).resolve().parents[2]
_LIB = _PHI0 / "scripts" / "lib"
_TOOLS_LIB = _PHI0 / "tools" / "lib"
for _p in (_PHI0 / "src", _LIB, _TOOLS_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _early_argv_value(*names: str) -> str | None:
    for name in names:
        flag = f"--{name.replace('_', '-')}"
        for i, a in enumerate(sys.argv):
            if a == flag and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
            if a.startswith(f"{flag}="):
                return a.split("=", 1)[1]
    return None


_hand_cli = _early_argv_value("newton_hand")
if _hand_cli:
    os.environ["NEWTON_HAND"] = _hand_cli

from phi0.hand.hand_mode import apply_newton_hand_env  # noqa: E402

_NEWTON_HAND = apply_newton_hand_env()
_usd = (
    "main_nodex_revo2_newton"
    if _NEWTON_HAND == "revo2"
    else ("main_nodex_dex3_newton" if os.environ.get("PHI0_NEWTON_DEX3_USD") == "1" else "main_nodex_newton")
)
print(
    f"[NEWTON_VIZ] newton_hand={_NEWTON_HAND} "
    f"PHI0_NEWTON_REVO2={os.environ.get('PHI0_NEWTON_REVO2')} "
    f"usd={_usd}",
    flush=True,
)

import imageio
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
from isaaclab_tasks.utils.sim_launcher import launch_simulation

from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.utils import config_utils

from newton_gate_common import fill_obs_dims, load_atm_policy, load_full_cfg
from phi0.online.isaac_loop import run_online_infer_eval
from phi0.online.isaac_sim import assert_env_control_hz, enable_direct_latent_atm
from phi0.online.sim_adapter import sim_uses_xyzw

config_utils.register_rl_resolvers()

PHI0 = _PHI0
EGYPT = Path("/mnt/data2/wpy/workspace/egypt_smplsem_clip")
DEFAULT_BONESEED = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
GOLD_CKPT = Path(
    "/mnt/data2/wpy/workspace/phi-0-wbc/experiments/"
    "egypt_overfit_onnx_gpu_b1024_20260725_110734/phi0_student_step002000.pt"
)

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt", type=str, default=str(GOLD_CKPT))
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--num_steps", type=int, default=278)
parser.add_argument("--ref_start", type=int, default=0)
parser.add_argument("--ref_root", type=str, default="")
parser.add_argument("--max_ref_frames", type=int, default=278)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--frame_skip", type=int, default=2)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=544)  # divisible by 16 for h264
parser.add_argument(
    "--use_lang_latent_cache",
    action="store_true",
    default=False,
    help="BoneSEED deploy: feed cached Qwen3-VL lang_ctx (matches train)",
)
parser.add_argument(
    "--use_vlm",
    action="store_true",
    default=False,
    help="online_vlm: dual/text DistillVlmHold (matches train use_vlm)",
)
parser.add_argument(
    "--control",
    type=str,
    default="qpos_student",
    choices=(
        "qpos_student",
        "qpos_ref",
        "ref",
        "student",
        "direct_latent",
        "direct_latent_ref",
    ),
)
parser.add_argument(
    "--infer-only",
    action="store_true",
    default=False,
    help="Skip Newton-GL capture/mp4; only run closed-loop infer npz (MuJoCo replay path).",
)
parser.add_argument(
    "--gt_panel_layout",
    type=str,
    default=os.environ.get("GT_PANEL_LAYOUT", "top"),
    choices=("top", "none", "robot"),
    help="Default top: dataset ego+wrist above Newton-GL at current tape frame. "
    "none/robot = sim-only.",
)
parser.add_argument(
    "--gt_panel_h",
    type=int,
    default=int(os.environ.get("GT_PANEL_H", "160")),
    help="Height of the top dataset vision strip (pixels).",
)
parser.add_argument(
    "--newton_hand",
    type=str,
    default=os.environ.get("NEWTON_HAND", "rubber").strip().lower(),
    choices=("rubber", "dex3", "revo2"),
    help="Newton viz mesh: rubber=main_nodex_newton (default), "
    "dex3=composed三指 USD (opt-in), revo2=强脑.",
)
add_launcher_args(parser)
# kit-less Newton GL; infer-only skips render (no visualizer).
if not any(a == "--infer-only" for a in sys.argv):
    parser.set_defaults(visualizer=["newton"], headless=True)
else:
    parser.set_defaults(headless=True)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


class _CaptureOnStep:
    """Wrap ManagerEnvWrapper.step → grab Newton-GL RGB after each control step."""

    def __init__(self, wrapper: ManagerEnvWrapper, inner: ManagerBasedRLEnv, every_n: int):
        self.wrapper = wrapper
        self.inner = inner
        self.every_n = max(1, int(every_n))
        self.frames: list[np.ndarray] = []
        # Control step_i (= _n-1) for each kept frame; aligns with infer rows[].
        self.step_ids: list[int] = []
        self._n = 0
        self._orig = wrapper.step
        wrapper.step = self._step  # type: ignore[method-assign]
        self.cam_offset = np.array([2.8, 2.8, 1.4], dtype=np.float64)

    def _follow_camera(self) -> None:
        robot = self.inner.scene["robot"]
        root = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        eye = root + self.cam_offset
        look = root + np.array([0.0, 0.0, 0.35], dtype=np.float64)
        cap = getattr(getattr(self.inner, "video_recorder", None), "_capture", None)
        if cap is not None and hasattr(cap, "update_camera"):
            cap.update_camera(tuple(eye.tolist()), tuple(look.tolist()))

    def _step(self, *a, **k):
        out = self._orig(*a, **k)
        self._n += 1
        if self._n % self.every_n != 0:
            return out
        try:
            self._follow_camera()
            rgb = self.inner.render()
        except Exception as exc:  # noqa: BLE001
            print(f"[NEWTON_VIZ] render skip step={self._n}: {exc}", flush=True)
            return out
        if rgb is not None:
            self.frames.append(np.asarray(rgb))
            self.step_ids.append(int(self._n - 1))
        return out

    def restore(self) -> None:
        self.wrapper.step = self._orig  # type: ignore[method-assign]


def _composite_top_dataset_panel(
    main_rgb: np.ndarray,
    panels: list[tuple[np.ndarray, str]],
    *,
    panel_h: int,
) -> np.ndarray:
    """Stack dataset RGB tiles above Newton-GL (same layout as MuJoCo GT_PANEL=top)."""
    import cv2

    main = np.asarray(main_rgb)
    if main.ndim == 3 and main.shape[-1] == 4:
        main = main[..., :3]
    if main.dtype != np.uint8:
        main = np.clip(main, 0, 255).astype(np.uint8)
    main_h, main_w = main.shape[:2]
    top = np.zeros((int(panel_h), main_w, 3), dtype=np.uint8)
    n = max(len(panels), 1)
    slot_w = main_w // n
    for i, (img, label) in enumerate(panels):
        tile = np.asarray(img)
        if tile.ndim == 3 and tile.shape[-1] == 4:
            tile = tile[..., :3]
        if tile.dtype != np.uint8:
            if tile.max() <= 1.0:
                tile = (np.clip(tile, 0, 1) * 255.0).astype(np.uint8)
            else:
                tile = np.clip(tile, 0, 255).astype(np.uint8)
        ih, iw = tile.shape[:2]
        scale = min((slot_w - 8) / max(iw, 1), (panel_h - 24) / max(ih, 1))
        tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
        small = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
        x0 = i * slot_w + (slot_w - tw) // 2
        y0 = 18 + (panel_h - 18 - th) // 2
        top[y0 : y0 + th, x0 : x0 + tw] = small
        if label:
            cv2.putText(
                top,
                label,
                (i * slot_w + 6, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
    return np.vstack([top, main])


def _overlay_dataset_vision(
    frames: list[np.ndarray],
    step_ids: list[int],
    *,
    ref_root: Path,
    ref_start: int,
    max_ref_frames: int,
    infer_rows: list[dict],
    panel_h: int,
) -> list[np.ndarray]:
    """Decode dataset ego+wrist at each captured tape time; composite on top."""
    if not frames:
        return frames
    if len(step_ids) != len(frames):
        raise RuntimeError(
            f"step_ids/frames length mismatch {len(step_ids)} vs {len(frames)}"
        )
    times: list[int] = []
    for sid in step_ids:
        if 0 <= sid < len(infer_rows):
            times.append(int(round(float(infer_rows[sid]["t_mean"]))))
        else:
            # Fallback: dense 1:1 with ref window (no RSI wrap).
            times.append(int(sid))
    times_np = np.asarray(times, dtype=np.int64)

    from phi0.online.latent_ref import load_sonic_latent_reference
    from phi0.online.ref_video_vlm import RefVideoFrameSource

    ref = load_sonic_latent_reference(
        str(ref_root),
        max_frames=int(max_ref_frames),
        start=int(ref_start),
        require_rsi=False,
        require_smpl=False,
    )
    ei = getattr(ref, "episode_index", None)
    fi = getattr(ref, "frame_index", None)
    if ei is None or fi is None:
        print(
            "[NEWTON_VIZ] gt panel skip: ref missing episode_index/frame_index",
            flush=True,
        )
        return frames
    ep0 = int(np.asarray(ei).reshape(-1)[int(times_np[0])])
    # Text-only / BoneSEED eps: no mp4 — leave sim-only.
    from phi0.data.lerobot_step_io import load_episodes

    ep_meta = {int(e["episode_index"]): e for e in load_episodes(ref_root)}
    if ep0 in ep_meta and not bool(ep_meta[ep0].get("has_video", True)):
        print(
            f"[NEWTON_VIZ] gt panel skip: episode {ep0} has_video=false",
            flush=True,
        )
        return frames

    show_labels = os.environ.get("GT_PANEL_LABELS", "0").strip() != "0"
    video_src = RefVideoFrameSource(ref_root, fps=50.0)
    ego_t, chest_t = video_src.frames_from_ref(ref, times_np)  # [N,1,C,H,W] float
    ego_u8 = (ego_t[:, 0].permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255.0).astype(
        np.uint8
    )
    chest_u8 = (
        chest_t[:, 0].permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255.0
    ).astype(np.uint8)
    fi_a = np.asarray(fi, dtype=np.int64).reshape(-1)
    out: list[np.ndarray] = []
    for i, main in enumerate(frames):
        t = int(np.clip(times_np[i], 0, len(fi_a) - 1))
        f_idx = int(fi_a[t])
        ego_lab = f"ego (f={f_idx})" if show_labels else "ego"
        wrist_lab = f"wrist (f={f_idx})" if show_labels else "wrist"
        out.append(
            _composite_top_dataset_panel(
                main,
                [(ego_u8[i], ego_lab), (chest_u8[i], wrist_lab)],
                panel_h=int(panel_h),
            )
        )
    print(
        f"[NEWTON_VIZ] gt panel top n={len(out)} ep={ep0} "
        f"t0={int(times_np[0])} tN={int(times_np[-1])} panel_h={panel_h}",
        flush=True,
    )
    return out


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
    ) as w:
        for fr in frames:
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            if fr.ndim == 3 and fr.shape[-1] == 4:
                fr = fr[..., :3]
            w.append_data(fr)


def main() -> None:
    ckpt = Path(args_cli.student_ckpt)
    assert ckpt.is_file(), ckpt
    from datetime import datetime

    env_tag = (os.environ.get("TAG") or "").strip()
    tag = env_tag or f"newton_isaac_qpos_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    control = str(args_cli.control)
    ref_root = Path(args_cli.ref_root) if str(args_cli.ref_root).strip() else EGYPT
    max_ref = int(args_cli.max_ref_frames) or int(args_cli.num_steps)
    out_dir = Path(args_cli.out_dir or str(PHI0 / "experiments" / tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"infer_{control}_newton_gl.mp4"

    conf = load_full_cfg(1)
    env_cfg = custom_instantiate(conf.manager_env)
    env_cfg.seed = 42
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, "config") and isinstance(env_cfg.config, dict):
        env_cfg.config["headless"] = True
    infer_only = bool(args_cli.infer_only)
    render_mode = None
    if not infer_only:
        # Lab3 VideoRecorder → Newton GL from physics stack (no Kit / no TiledCamera).
        env_cfg.viewer = ViewerCfg(eye=(2.8, 2.8, 1.4), lookat=(0.0, 0.0, 0.9))
        env_cfg.video_recorder.backend_source = "renderer"
        env_cfg.video_recorder.window_width = int(args_cli.width)
        env_cfg.video_recorder.window_height = int(args_cli.height)
        env_cfg.video_recorder.eye = env_cfg.viewer.eye
        env_cfg.video_recorder.lookat = env_cfg.viewer.lookat
        render_mode = "rgb_array"

    phys = type(getattr(env_cfg.sim, "physics", None)).__name__
    print(
        f"[NEWTON_VIZ] physics={phys} xyzw={sim_uses_xyzw()} control={control} "
        f"ref={ref_root} ref_start={int(args_cli.ref_start)} max_ref={max_ref} "
        f"ckpt={ckpt} out={out_dir} py={_py}",
        flush=True,
    )
    assert "Newton" in phys, phys

    capture: _CaptureOnStep | None = None
    with launch_simulation(env_cfg, args_cli):
        inner = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
        env = ManagerEnvWrapper(inner, env_cfg.config)
        device = str(env.device)
        fill_obs_dims(env)
        atm = load_atm_policy(
            conf,
            env,
            device,
            policy_dir=(os.environ.get("ATM_POLICY_DIR") or "").strip() or None,
        )
        enable_direct_latent_atm(env, atm)
        assert_env_control_hz(env, expected_hz=50.0)

        if not infer_only:
            backend = getattr(getattr(inner, "video_recorder", None), "_backend", None)
            print(f"[NEWTON_VIZ] video_backend={backend}", flush=True)
            if backend != "newton_gl":
                raise RuntimeError(f"expected newton_gl video backend, got {backend!r}")

        capture = None
        if not infer_only:
            capture = _CaptureOnStep(env, inner, every_n=int(args_cli.frame_skip))
        try:
            result = run_online_infer_eval(
                env=env,
                atm_policy=atm,
                device=device,
                student_ckpt=str(ckpt),
                num_steps=int(args_cli.num_steps),
                action_horizon=int(args_cli.horizon),
                max_ref_frames=max_ref,
                ref_start=int(args_cli.ref_start),
                phi0_full_v3_root=str(ref_root),
                seed=42,
                out_dir=out_dir,
                control=control,
                teacher_encoder="onnx_g1",
                use_lang_latent_cache=bool(args_cli.use_lang_latent_cache),
                use_vlm=bool(args_cli.use_vlm),
            )
        finally:
            if capture is not None:
                capture.restore()

        close = getattr(env, "close", None) or getattr(inner, "close", None)
        if callable(close):
            close()

    if infer_only:
        npz = out_dir / f"infer_qpos_traj_{control}.npz"
        print(f"[NEWTON_VIZ] infer-only ok npz={npz}", flush=True)
        return

    assert capture is not None
    n_frames = len(capture.frames)
    print(f"[NEWTON_VIZ] captured_frames={n_frames}", flush=True)
    if n_frames < 2:
        raise RuntimeError("Newton-GL capture produced <2 frames")

    frames_out = capture.frames
    gt_layout = str(args_cli.gt_panel_layout).strip().lower()
    if gt_layout == "top":
        rows = result.get("steps") or []
        if not isinstance(rows, list):
            rows = []
        try:
            frames_out = _overlay_dataset_vision(
                capture.frames,
                capture.step_ids,
                ref_root=ref_root,
                ref_start=int(args_cli.ref_start),
                max_ref_frames=max_ref,
                infer_rows=rows,
                panel_h=int(args_cli.gt_panel_h),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[NEWTON_VIZ] gt panel failed (sim-only mp4): {exc}", flush=True)
            frames_out = capture.frames
    else:
        print(f"[NEWTON_VIZ] gt panel layout={gt_layout} (sim-only)", flush=True)

    fps = 50.0 / float(args_cli.frame_skip)
    _write_mp4(frames_out, mp4, fps=fps)

    # Contract sidecar: action vs measured q, root height from logged traj.
    npz = out_dir / f"infer_qpos_traj_{control}.npz"
    steps_v = result.get("steps", 0)
    if isinstance(steps_v, list):
        steps_n = len(steps_v)
    else:
        try:
            steps_n = int(steps_v)
        except (TypeError, ValueError):
            steps_n = 0
    contract: dict = {
        "python": str(_py),
        "physics": phys,
        "video_backend": "newton_gl",
        "sim_quat_xyzw": True,
        "control": control,
        "student_ckpt": str(ckpt),
        "mp4": str(mp4),
        "n_frames": n_frames,
        "num_steps": int(args_cli.num_steps),
        "ref_start": int(args_cli.ref_start),
        "ref_root": str(ref_root),
        "max_ref_frames": max_ref,
        "gt_panel_layout": gt_layout,
        "gt_panel_h": int(args_cli.gt_panel_h) if gt_layout == "top" else 0,
        "infer_ok": bool(result.get("ok", steps_n > 0)),
        "infer_steps": steps_n,
    }
    if npz.is_file():
        data = np.load(npz)
        q_cmd = data["q_cmd"]
        q_act = data["q_act"]
        root_act = data["root_act"]
        # q_* are [T,B,36] or [T,B,29]; compare overlapping joint dims
        t = min(q_cmd.shape[0], q_act.shape[0])
        jc = q_cmd[:t, 0, -29:].astype(np.float64)
        ja = q_act[:t, 0, -29:].astype(np.float64)
        contract["mean_abs_q_cmd_vs_q_act"] = float(np.mean(np.abs(jc - ja)))
        contract["root_z_start"] = float(root_act[0, 0, 2])
        contract["root_z_end"] = float(root_act[t - 1, 0, 2])
        contract["root_z_min"] = float(np.min(root_act[:t, 0, 2]))

    (out_dir / "newton_isaac_viz_contract.json").write_text(json.dumps(contract, indent=2))
    print(f"[NEWTON_VIZ] mp4={mp4}", flush=True)
    print(f"[NEWTON_VIZ] contract={out_dir / 'newton_isaac_viz_contract.json'}", flush=True)
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
