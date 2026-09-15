#!/usr/bin/env python3
"""Offline check: training dataloader forward vs train_aligned deploy forward."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phi0.checkpoint_utils import merge_saved_cfg  # noqa: E402
from phi0.data.pick_tissue_unified import (  # noqa: E402
    CHEST_FORWARD_BATCH_KEY,
    PickTissueUnifiedClipDataset,
)
from phi0.deploy.gt_io import PickTissueGtBackend  # noqa: E402
from phi0.deploy.pick_tissue_gt import (  # noqa: E402
    control_index_to_global_frame,
    reader_from_data_cfg,
)
from phi0.deploy.pick_tissue_gt_images import _cached_predecoded_reader  # noqa: E402
from phi0.inference.session import (  # noqa: E402
    ActionInferenceSession,
    resolve_deploy_action_chunk_size,
)
from phi0.inference.train_aligned_forward import (  # noqa: E402
    history_window,
    predict_chunk_train_aligned,
)
from phi0.paths import resolve_workspace_path  # noqa: E402
from phi0.runtime import (  # noqa: E402
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    prepare_model_batch,
    resolve_inference_device,
    sync_model_action_norm,
)

logger = logging.getLogger(__name__)

SONIC_SLICE = slice(396, 460)
GRIPPER_SLICE = slice(346, 360)


def parse_args():
    p = argparse.ArgumentParser(description="810demo train vs train_aligned infer alignment")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            ROOT
            / "experiments/810demo_offline_vlm_b16_ddp4_noqpos_30k_20260729_172752/phi0_step30000.pt"
        ),
    )
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_810demo_egypt_vlm_b512")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--episodes", type=str, default="0,2")
    p.add_argument("--control-indices", type=str, default="0,32,128,256")
    p.add_argument("--sonic-cos-min", type=float, default=0.999)
    p.add_argument("--gripper-max-diff", type=float, default=1e-3)
    return p.parse_args()


def find_clip_index(ds: PickTissueUnifiedClipDataset, episode: int, base_index: int) -> int:
    for i, (ep, bi) in enumerate(ds._steps):
        if int(ep) == int(episode) and int(bi) == int(base_index):
            return i
    raise KeyError(f"no clip for episode={episode} base_index={base_index}")


def chw_to_hwc_uint8(chw: torch.Tensor) -> np.ndarray:
    t = chw[0] if chw.ndim == 4 else chw
    arr = t.detach().cpu().float()
    if float(arr.max()) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).permute(1, 2, 0).numpy().astype(np.uint8)


def sonic_cosine(a: np.ndarray, b: np.ndarray) -> float:
    sa = np.asarray(a[..., SONIC_SLICE], dtype=np.float64).reshape(-1)
    sb = np.asarray(b[..., SONIC_SLICE], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(sa) * np.linalg.norm(sb))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(sa, sb) / denom)


def gripper_max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(a[..., GRIPPER_SLICE]) - np.asarray(b[..., GRIPPER_SLICE])))
    )


@torch.no_grad()
def predict_dataloader_path(
    model,
    processor,
    ds: PickTissueUnifiedClipDataset,
    clip_idx: int,
    chunk_h: int,
    *,
    deploy_seq_len: int = 33,
) -> np.ndarray:
    sample = ds[clip_idx]
    batch = PickTissueUnifiedClipDataset.collate_fn([sample])
    mb = prepare_model_batch(model, processor, batch)
    inputs = model.build_inputs(mb)

    unified_ds = str(getattr(model, "unified_supervision_dataset", "g1_sonic"))
    session = ActionInferenceSession(
        model,
        processor=processor,
        deploy_seq_len=int(deploy_seq_len),
        use_gt_history=True,
        unified_supervision_dataset=unified_ds,
    )
    session.prefill_from_clip_inputs(inputs)

    w = history_window(model)
    if "proprio" in mb and mb["proprio"] is not None:
        # Training batch uses proprio41; do not feed 512-d action prefix as proprio_tokens.
        proprio = mb["proprio"].to(model.device)
        session.set_proprio_gt(proprio[0] if proprio.ndim == 3 else proprio)
    elif w > 0 and model.uses_history_action_input():
        action = mb["action"][:, :w, :].to(model.device)
        masked = model._mask_training_action_prefix(action)
        session.set_history_gt(masked[0])
    if "exec_step" in mb:
        session.set_exec_step(int(mb["exec_step"][0].item()))

    device = model.device
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = session.predict(int(chunk_h), denormalize=True)
    return pred.float().detach().cpu().numpy()


def build_gt_readers(cfg, episode: int, *, control_fps: float):
    reader = reader_from_data_cfg(cfg.data)
    span = reader.episode_span(int(episode))
    root = str(resolve_workspace_path(str(cfg.data.get("pick_tissue_root", "./data"))))
    repo = str(cfg.data.get("pick_tissue_repo_id", "810demo_egypt_layout"))
    frame_reader = _cached_predecoded_reader(root, repo, "letterbox_raw")
    frame_reader.preload_episode(span)
    backend = PickTissueGtBackend(
        reader,
        span,
        native_fps=float(reader.native_fps),
        control_fps=float(control_fps),
    )

    def global_frame(control_idx: int) -> int:
        return control_index_to_global_frame(
            span.frame_start,
            int(control_idx),
            native_fps=float(reader.native_fps),
            control_fps=float(control_fps),
        )

    def read_ego_hwc(control_idx: int) -> np.ndarray:
        return frame_reader.read_ego_rgb(global_frame(control_idx), span)

    def read_chest_hwc(control_idx: int) -> np.ndarray:
        return frame_reader.read_chest_rgb(global_frame(control_idx), span)

    instruction = str(ds_instruction(cfg, episode))
    return backend, read_ego_hwc, read_chest_hwc, instruction


def ds_instruction(cfg, episode: int) -> str:
    ds = PickTissueUnifiedClipDataset(
        root_dir=str(resolve_workspace_path(str(cfg.data.get("pick_tissue_root", "./data")))),
        repo_id=str(cfg.data.get("pick_tissue_repo_id", "810demo_egypt_layout")),
        seq_len=int(cfg.data.get("seq_len", 33)),
        val=False,
        val_ratio=0.0,
        train_obs_only_video=bool(cfg.data.get("train_obs_only_video", True)),
        video_backend="pyav",
        supervision_dataset=str(cfg.data.get("unified_supervision_dataset", "g1_sonic")),
        action_dim=int(cfg.data.get("action_dim", 512)),
        image_size=tuple(cfg.data.get("image_size", [480, 640])),
    )
    return ds._instructions.get(int(episode), "pick tissue")


def load_model_bundle(args):
    os.environ.setdefault("PHI0_WORKSPACE", "/mnt/data2/wpy/workspace")
    ckpt_path = Path(args.checkpoint).resolve()
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    cfg.device = device

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])

    cfg.data.pick_tissue_root = str(
        resolve_workspace_path(str(cfg.data.get("pick_tissue_root", "./data")))
    )
    stats_path = str(cfg.data.get("action_stats_path", "") or "").strip()
    if stats_path:
        cfg.data.action_stats_path = str(resolve_workspace_path(stats_path))

    # ponytail: verify runs on nodes without FA2 wheels
    if hasattr(cfg, "model") and hasattr(cfg.model, "vlm"):
        cfg.model.vlm.attn_implementation = "sdpa"

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt_path))
    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()
    return cfg, model, processor


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    os.chdir(ROOT)

    cfg, model, processor = load_model_bundle(args)
    chunk_h = resolve_deploy_action_chunk_size(model, seq_len=int(cfg.data.get("seq_len", 33)))
    control_fps = float(cfg.data.get("control_fps", 50.0))

    ds = PickTissueUnifiedClipDataset(
        root_dir=str(resolve_workspace_path(str(cfg.data.get("pick_tissue_root", "./data")))),
        repo_id=str(cfg.data.get("pick_tissue_repo_id", "810demo_egypt_layout")),
        seq_len=int(cfg.data.get("seq_len", 33)),
        val=False,
        val_ratio=0.0,
        train_obs_only_video=bool(cfg.data.get("train_obs_only_video", True)),
        video_backend="pyav",
        supervision_dataset=str(cfg.data.get("unified_supervision_dataset", "g1_sonic")),
        action_dim=int(cfg.data.get("action_dim", 512)),
        image_size=tuple(cfg.data.get("image_size", [480, 640])),
    )

    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    control_indices = [int(x) for x in args.control_indices.split(",") if x.strip()]

    failures: list[str] = []
    for ep in episodes:
        backend, _, _, instruction = build_gt_readers(cfg, ep, control_fps=control_fps)
        session = ActionInferenceSession(model, processor=processor, use_gt_history=True)

        for control_idx in control_indices:
            clip_idx = find_clip_index(ds, ep, control_idx)
            sample = ds[clip_idx]
            assert CHEST_FORWARD_BATCH_KEY in sample["images"], "missing chest in dataloader sample"
            ego_hwc = chw_to_hwc_uint8(sample["images"]["ego_view"])
            chest_hwc = chw_to_hwc_uint8(sample["images"][CHEST_FORWARD_BATCH_KEY])
            task_instruction = str(sample.get("task") or instruction)

            dl_pred = predict_dataloader_path(
                model,
                processor,
                ds,
                clip_idx,
                chunk_h,
                deploy_seq_len=int(cfg.data.get("seq_len", 33)),
            )
            aligned_pred = predict_chunk_train_aligned(
                session,
                model,
                processor,
                control_idx=int(control_idx),
                ego_hwc=ego_hwc,
                chest_hwc=chest_hwc,
                instruction=task_instruction,
                gt_backend=backend,
                chunk_len=chunk_h,
                denormalize=True,
            )

            cos = sonic_cosine(dl_pred, aligned_pred)
            gdiff = gripper_max_diff(dl_pred, aligned_pred)
            ok = cos >= float(args.sonic_cos_min) and gdiff <= float(args.gripper_max_diff)
            status = "PASS" if ok else "FAIL"
            logger.info(
                "%s ep=%d ctrl=%d clip=%d cos=%.6f gripper_max=%.6e",
                status,
                ep,
                control_idx,
                clip_idx,
                cos,
                gdiff,
            )
            if not ok:
                failures.append(
                    f"ep={ep} ctrl={control_idx}: cos={cos:.6f} gripper_max={gdiff:.6e}"
                )

    if failures:
        logger.error("alignment check failed:\n  %s", "\n  ".join(failures))
        return 1
    logger.info("all %d cases passed", len(episodes) * len(control_indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
