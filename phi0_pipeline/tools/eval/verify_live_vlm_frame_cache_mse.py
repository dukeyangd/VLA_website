#!/usr/bin/env python3
"""Compare live VLM encode vs ``vlm_frame_latents_qwen3vl_dual`` cache (MSE gate).

Uses the same pad path as ``DistillVlmHold`` (``PHI0_VLM_ENCODE_MIN_BATCH``).

Usage::

    CUDA_VISIBLE_DEVICES=0 PHI0_VLM_ENCODE_MIN_BATCH=2 \\
      python tools/eval/verify_live_vlm_frame_cache_mse.py \\
      --dataset-root .../skill_walk_to_black_box_new_unified --episode 194
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phi0.models.vlm.attn import (  # noqa: E402
    allow_sdpa_attn_fallback,
    resolve_vlm_attn_implementation,
)
from phi0.models.vlm.preprocess import build_deploy_vlm_inputs_from_pixels  # noqa: E402
from phi0.models.vlm.tower import Qwen3VLTower  # noqa: E402
from phi0.online.ref_video_vlm import RefVideoFrameSource  # noqa: E402
from phi0.online.task_prompts import load_task_prompt_table  # noqa: E402
from phi0.online.vlm_frame_latents import DualVlmFrameLatentCache  # noqa: E402
from phi0.online.vlm_hold_distill import (  # noqa: E402
    _pad_vision_encode_batch,
    vlm_encode_min_batch_from_env,
)


def _load_tower(vlm_path: str, device: str) -> Qwen3VLTower:
    attn = resolve_vlm_attn_implementation("flash_attention_2")
    try:
        tower = Qwen3VLTower(
            vlm_path, device=device, torch_dtype=torch.bfloat16, attn_implementation=attn
        )
    except Exception as exc:
        if not allow_sdpa_attn_fallback():
            raise
        tower = Qwen3VLTower(
            vlm_path, device=device, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    tower.eval()
    return tower


@torch.inference_mode()
def _encode_one_frame(
    tower: Qwen3VLTower,
    video_src: RefVideoFrameSource,
    *,
    episode: int,
    frame_index: int,
    prompt: str,
    min_batch: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ei = np.asarray([episode], dtype=np.int64)
    fi = np.asarray([frame_index], dtype=np.int64)
    ego, chest = video_src.frames_at(ei, fi)
    ego_p, chest_p, prompts_p, n_real = _pad_vision_encode_batch(
        ego.float(), chest.float(), [prompt], min_batch=int(min_batch)
    )
    assert n_real == 1
    batch = build_deploy_vlm_inputs_from_pixels(
        tower.processor,
        None,
        ego_p,
        prompts_p,
        model_max_length=max_len,
        chest_pixel=chest_p,
    )
    dev = next(tower.vlm_model.parameters()).device

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=dev, non_blocking=True)

    ctx, mask = tower.extract_action_context(
        _to(batch["input_ids"]),
        _to(batch["attention_mask"]),
        _to(batch["pixel_values"]),
        _to(batch["image_grid_thw"]),
        _to(batch["mm_token_type_ids"]) if batch.get("mm_token_type_ids") is not None else None,
    )
    L = int(mask[0].sum().item())
    return ctx[0, :L].float().cpu(), mask[0, :L].cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--encode-batch", type=int, default=0, help="0=use PHI0_VLM_ENCODE_MIN_BATCH")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    cache = DualVlmFrameLatentCache.open(root)
    meta = json.loads((cache.cache_root / "meta.json").read_text(encoding="utf-8"))
    prompt = load_task_prompt_table(root).get(0, cache.prompt)
    min_batch = int(args.encode_batch) if int(args.encode_batch) > 0 else vlm_encode_min_batch_from_env()
    ep_rows = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    t_len = int(ep_rows.loc[ep_rows.episode_index == int(args.episode), "length"].iloc[0])
    n_frames = t_len if int(args.max_frames) <= 0 else min(t_len, int(args.max_frames))

    tower = _load_tower(str(meta["vlm_path"]), args.device)
    video_src = RefVideoFrameSource(str(root), fps=50.0)
    max_len = int(meta.get("max_seq_len") or 256)

    mse_list: list[float] = []
    cos_list: list[float] = []
    cos_norm_list: list[float] = []
    norm_ratio_list: list[float] = []
    fis = list(range(n_frames))
    # Simulate hold B=1: one frame per encode call with min_batch pad.
    for fi in fis:
        live_ctx, live_mask = _encode_one_frame(
            tower,
            video_src,
            episode=int(args.episode),
            frame_index=int(fi),
            prompt=prompt,
            min_batch=min_batch,
            max_len=max_len,
        )
        cache_ctx, cache_mask = cache.get_frame(int(args.episode), int(fi))
        L = min(int(live_mask.sum()), int(cache_mask.sum()))
        live = live_ctx[:L].float()
        ref = cache_ctx[:L].float()
        diff = live - ref
        mse_list.append(float((diff**2).mean()))
        # ponytail: token-wise cosine; report mean over valid tokens
        live_n = live.norm(dim=-1).clamp_min(1e-8)
        ref_n = ref.norm(dim=-1).clamp_min(1e-8)
        cos_tok = (live * ref).sum(dim=-1) / (live_n * ref_n)
        cos_list.append(float(cos_tok.mean()))
        live_u = live / live_n.unsqueeze(-1)
        ref_u = ref / ref_n.unsqueeze(-1)
        cos_norm_list.append(float((live_u * ref_u).sum(dim=-1).mean()))
        norm_ratio_list.append(float((live_n / ref_n).mean()))

    arr = np.asarray(mse_list, dtype=np.float64)
    cos_arr = np.asarray(cos_list, dtype=np.float64)
    cos_n_arr = np.asarray(cos_norm_list, dtype=np.float64)
    nr_arr = np.asarray(norm_ratio_list, dtype=np.float64)
    print(f"dataset={root}")
    print(f"episode={args.episode} frames={n_frames} min_encode_batch={min_batch}")
    print(f"mse mean={arr.mean():.6e} std={arr.std():.6e} max={arr.max():.6e}")
    print(f"mse p99={np.percentile(arr, 99):.6e} near_zero={int((arr < 1e-4).sum())}/{len(arr)}")
    print(f"cos mean={cos_arr.mean():.6f} min={cos_arr.min():.6f} p01={np.percentile(cos_arr, 1):.6f}")
    print(f"cos_norm mean={cos_n_arr.mean():.6f} min={cos_n_arr.min():.6f}")
    print(f"norm_ratio mean={nr_arr.mean():.6f} std={nr_arr.std():.6f}")
    worst = int(arr.argmax())
    print(f"worst frame={worst} mse={arr[worst]:.6e} cos={cos_arr[worst]:.6f}")


if __name__ == "__main__":
    main()
