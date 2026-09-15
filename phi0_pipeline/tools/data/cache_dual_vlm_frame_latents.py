#!/usr/bin/env python3
"""Cache per-frame dual ego+chest + prompt VLM latents for a teleop unified dataset.

Matches train encode::

    build_deploy_vlm_inputs_from_pixels(ego, chest, prompt)
    → Qwen3VLTower.extract_action_context

Writes ``<dataset>/meta/vlm_frame_latents_qwen3vl_dual/ep/{ep:06d}/``.

Train reads this cache; live ``DistillVlmHold(1)`` (B=1) is **not** numerically
equivalent — see ``docs/report/deploy/vlm_frame_cache_infer_batch_LOCKED.md``.

Usage::

    # init meta once
    python tools/data/cache_dual_vlm_frame_latents.py --dataset-root ... --init-only
    # 8-GPU shard (see run_cache_dual_vlm_frame_latents_8gpu.sh)
    CUDA_VISIBLE_DEVICES=i python ... --shard i/8 --resume --batch-size 4

Dataloader::

    from phi0.online.vlm_frame_latents import DualVlmFrameLatentCache
    cache = DualVlmFrameLatentCache.open(REF_ROOT)
    ctx, mask = cache.get_batch(episode_indices, frame_indices)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from phi0.models.vlm.preprocess import build_deploy_vlm_inputs_from_pixels, normalize_vlm_instruction  # noqa: E402
from phi0.models.vlm.tower import (  # noqa: E402
    OFFICIAL_QWEN3VL_INSTRUCT,
    QWEN3VL_HIDDEN_DIM,
    Qwen3VLTower,
)
from phi0.online.ref_video_vlm import CHEST_KEY, EGO_KEY, RefVideoFrameSource  # noqa: E402
from phi0.online.vlm_frame_latents import (  # noqa: E402
    cache_dir_for_dataset,
    episode_done,
    init_episode_files,
    open_episode_writable,
    resolve_cache_dirname,
    write_cache_meta,
)

_DEFAULT_VLM_PATH = (
    "/mnt/data3/hf_home/hub/models--Qwen--Qwen3-VL-2B-Instruct/"
    "snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)


def _resolve_vlm_path(vlm_path: str | None) -> str:
    if vlm_path:
        return str(vlm_path)
    p = Path(_DEFAULT_VLM_PATH)
    if p.is_dir():
        return str(p)
    return OFFICIAL_QWEN3VL_INSTRUCT


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if not spec:
        return 0, 1
    i_s, n_s = spec.split("/")
    i_s, n_s = int(i_s), int(n_s)
    if n_s < 1 or i_s < 0 or i_s >= n_s:
        raise SystemExit(f"bad --shard {spec!r}; want i/N with 0<=i<N")
    return i_s, n_s


def _load_prompt(dataset_root: Path, override: str | None) -> str:
    if override and str(override).strip():
        return str(override).strip()
    tasks = dataset_root / "meta" / "tasks.parquet"
    if not tasks.is_file():
        raise FileNotFoundError(f"missing {tasks}; pass --prompt")
    df = pq.read_table(tasks).to_pandas()
    if "task" not in df.columns or len(df) < 1:
        raise RuntimeError(f"{tasks} missing task column")
    return str(df.loc[0, "task"]).strip()


def _load_episodes(dataset_root: Path) -> list[dict]:
    ep_pq = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_pq.is_file():
        raise FileNotFoundError(ep_pq)
    df = pq.read_table(ep_pq).to_pandas()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "episode_index": int(r["episode_index"]),
                "length": int(r["length"]),
            }
        )
    return rows


def _video_path(dataset_root: Path, episode_index: int, video_key: str) -> Path:
    # unified pack: videos/chunk-000/{key}/episode_{ep:06d}.mp4
    chunk = episode_index // 1000
    return (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def _encode_chunk(
    tower: Qwen3VLTower,
    video_src: RefVideoFrameSource,
    ep_i: int,
    frame_indices: list[int],
    prompt: str,
    *,
    max_seq_len: int,
    encode_bs: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Batch-decode + VLM-encode a list of frame indices within one episode."""
    if not frame_indices:
        return [], torch.empty(0), torch.empty(0)
    ei = np.asarray([ep_i] * len(frame_indices), dtype=np.int64)
    fi = np.asarray(frame_indices, dtype=np.int64)
    ego_t, chest_t = video_src.frames_at(ei, fi)
    outs_fi: list[int] = []
    ctx_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    bs = max(1, int(encode_bs))
    n_total = len(frame_indices)
    for start in range(0, n_total, bs):
        end = min(start + bs, n_total)
        n = end - start
        ctx, mask = _encode_batch(
            tower,
            ego_t[start:end],
            chest_t[start:end],
            [prompt] * n,
            max_seq_len=max_seq_len,
        )
        outs_fi.extend(frame_indices[start:end])
        ctx_parts.append(ctx)
        mask_parts.append(mask)
    return outs_fi, torch.cat(ctx_parts, dim=0), torch.cat(mask_parts, dim=0)


def _hwc_to_btchw(frames: list[np.ndarray]) -> torch.Tensor:
    # list of HWC uint8 → B,T=1,C,H,W float 0-1
    xs = []
    for arr in frames:
        t = torch.from_numpy(np.asarray(arr)).permute(2, 0, 1).float() / 255.0
        xs.append(t.unsqueeze(0))  # T=1,C,H,W
    return torch.stack(xs, dim=0)  # B,T,C,H,W


@torch.inference_mode()
def _encode_batch(
    tower: Qwen3VLTower,
    ego_btchw: torch.Tensor,
    chest_btchw: torch.Tensor,
    prompts: list[str],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = build_deploy_vlm_inputs_from_pixels(
        tower.processor,
        None,
        ego_btchw.cpu(),
        list(prompts),
        model_max_length=int(max_seq_len),
        chest_pixel=chest_btchw.cpu(),
        vlm_video_delta_indices=None,
    )
    device = next(tower.vlm_model.parameters()).device

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, non_blocking=True)

    return tower.extract_action_context(
        _to(batch["input_ids"]),
        _to(batch["attention_mask"]),
        _to(batch["pixel_values"]),
        _to(batch["image_grid_thw"]),
        _to(batch["mm_token_type_ids"])
        if batch.get("mm_token_type_ids") is not None
        else None,
    )


def _write_rows(
    latents: np.memmap,
    masks: np.memmap,
    lengths: np.memmap,
    frame_indices: list[int],
    ctx: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_seq: int,
) -> None:
    _b, s_cur, d = ctx.shape
    if s_cur > max_seq:
        ctx = ctx[:, :max_seq]
        mask = mask[:, :max_seq]
        s_cur = max_seq
    ctx_np = ctx.float().cpu().numpy()
    # ponytail: fp16 store saturates at ±65504; clip before cast (Psi0 HE can overflow).
    ctx_np = np.clip(ctx_np, -65504.0, 65504.0).astype(np.float16, copy=False)
    mask_np = mask.cpu().numpy().astype(np.uint8, copy=False)
    for j, fi in enumerate(frame_indices):
        latents[fi, :, :] = 0
        masks[fi, :] = 0
        latents[fi, :s_cur, :] = ctx_np[j]
        masks[fi, :s_cur] = mask_np[j]
        lengths[fi] = int(mask_np[j].sum())


def _load_tower(vlm_path: str, device: str, dtype_name: str) -> Qwen3VLTower:
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else device
    attn = resolve_vlm_attn_implementation("flash_attention_2")
    try:
        tower = Qwen3VLTower.from_pretrained(
            model_path=vlm_path,
            device=device,
            torch_dtype=dtype,
            freeze=True,
            attn_implementation=attn,
            local_files_only=Path(vlm_path).is_dir(),
        )
    except Exception as exc:  # noqa: BLE001
        if not allow_sdpa_attn_fallback():
            raise RuntimeError(
                f"VLM attn={attn} failed ({type(exc).__name__}: {exc}). "
                "Install flash-attn or set PHI0_ALLOW_SDPA_FALLBACK=1."
            ) from exc
        print(f"[dual_vlm_cache] {attn} → sdpa ({type(exc).__name__})", flush=True)
        tower = Qwen3VLTower.from_pretrained(
            model_path=vlm_path,
            device=device,
            torch_dtype=dtype,
            freeze=True,
            attn_implementation="sdpa",
            local_files_only=Path(vlm_path).is_dir(),
        )
        attn = "sdpa"
    print(f"[dual_vlm_cache] vlm_attn={attn} device={device}", flush=True)
    tower.eval()
    return tower


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--vlm-path", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=4, help="VLM encode micro-batch")
    ap.add_argument(
        "--decode-batch",
        type=int,
        default=0,
        help="frames per torchcodec decode call (0=whole ep missing span)",
    )
    ap.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        help="torchcodec|pyav|opencv (default torchcodec, matches train)",
    )
    ap.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help="RefVideoFrameSource thread pool (0=backend default: 16 cpu / 2 cuda)",
    )
    ap.add_argument(
        "--decode-device",
        type=str,
        default="cpu",
        help="torchcodec decode device (default cpu: avoid NVDEC fight with co-tenants; VLM still --device)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--shard", type=str, default=None, help="i/N over episodes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-only", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=0, help="0=all (debug)")
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    prompt = normalize_vlm_instruction(_load_prompt(root, args.prompt))
    episodes = _load_episodes(root)
    if int(args.max_episodes) > 0:
        episodes = episodes[: int(args.max_episodes)]
    shard_i, shard_n = _parse_shard(args.shard)
    cache_dirname = resolve_cache_dirname()
    out_dir = cache_dir_for_dataset(root)
    vlm_path = _resolve_vlm_path(args.vlm_path)
    hidden = QWEN3VL_HIDDEN_DIM
    max_seq = int(args.max_seq_len)
    n_frames_total = int(sum(int(e["length"]) for e in episodes))

    meta_blob = {
        "dataset_root": str(root),
        "cache_dirname": cache_dirname,
        "vlm_path": vlm_path,
        "prompt": prompt,
        "views": [EGO_KEY, CHEST_KEY],
        "encode": (
            "build_deploy_vlm_inputs_from_pixels(dual) + extract_action_context "
            "(last hidden); matches vision_dl_distill"
        ),
        "dtype_store": "float16",
        "max_seq_len": max_seq,
        "hidden_dim": hidden,
        "n_episodes": len(episodes),
        "n_frames": n_frames_total,
    }

    tag = f"shard={shard_i}/{shard_n}" if shard_n > 1 else "single"
    print(
        f"[dual_vlm_cache][{tag}] root={root} eps={len(episodes)} frames={n_frames_total} "
        f"out={out_dir} prompt={prompt!r} encode_bs={args.batch_size} "
        f"decode_bs={args.decode_batch or 'ep'} backend={args.video_backend}",
        flush=True,
    )

    if args.init_only or shard_i == 0:
        write_cache_meta(out_dir, meta_blob)
        print(f"[dual_vlm_cache][{tag}] wrote meta → {out_dir / 'meta.json'}", flush=True)
    if args.init_only:
        return

    # wait briefly if meta not yet visible (other ranks)
    for _ in range(60):
        if (out_dir / "meta.json").is_file():
            break
        time.sleep(1.0)
    else:
        raise SystemExit(f"[{tag}] missing meta; run --init-only first")

    mine = [e for e in episodes if int(e["episode_index"]) % shard_n == shard_i]
    if args.resume:
        todo = [
            e
            for e in mine
            if not episode_done(out_dir, int(e["episode_index"]), n_frames=int(e["length"]))
        ]
    else:
        todo = mine
    print(f"[dual_vlm_cache][{tag}] todo_eps={len(todo)}/{len(mine)}", flush=True)
    if not todo:
        print(f"[dual_vlm_cache][{tag}] nothing to encode", flush=True)
        return

    tower = _load_tower(vlm_path, args.device, args.dtype)
    encode_bs = max(1, int(args.batch_size))
    decode_bs = int(args.decode_batch)
    decode_workers = int(args.decode_workers) if int(args.decode_workers) > 0 else None
    # ponytail: default decode=cpu; co-tenant GPUs (e.g. Isaac) own NVDEC.
    # Ceiling: CPU decode + FA2 encode; upgrade = dedicated GPUs → --decode-device cuda.
    video_src = RefVideoFrameSource(
        root,
        fps=50.0,
        video_backend=str(args.video_backend),
        max_workers=decode_workers,
        decode_device=str(args.decode_device).strip() or "cpu",
    )
    t0 = time.time()
    frames_done = 0

    for ep in todo:
        ep_i = int(ep["episode_index"])
        T = int(ep["length"])
        ego_p = _video_path(root, ep_i, EGO_KEY)
        chest_p = _video_path(root, ep_i, CHEST_KEY)
        if not ego_p.is_file() or not chest_p.is_file():
            raise FileNotFoundError(f"ep{ep_i} missing videos: {ego_p} / {chest_p}")

        if args.resume and (out_dir / "ep" / f"{ep_i:06d}" / "latents.fp16.dat").is_file():
            latents, masks, lengths = open_episode_writable(
                out_dir, ep_i, n_frames=T, max_seq_len=max_seq, hidden_dim=hidden
            )
        else:
            latents, masks, lengths = init_episode_files(
                out_dir, ep_i, n_frames=T, max_seq_len=max_seq, hidden_dim=hidden
            )

        missing = [
            fi
            for fi in range(T)
            if not (args.resume and int(lengths[fi]) > 0)
        ]
        ep_t0 = time.time()
        chunk = decode_bs if decode_bs > 0 else max(len(missing), 1)
        for start in range(0, len(missing), chunk):
            fis = missing[start : start + chunk]
            outs_fi, ctx, mask = _encode_chunk(
                tower,
                video_src,
                ep_i,
                fis,
                prompt,
                max_seq_len=max_seq,
                encode_bs=encode_bs,
            )
            if int(ctx.shape[-1]) != hidden and ctx.numel() > 0:
                raise RuntimeError(f"ctx dim {ctx.shape[-1]} != {hidden}")
            _write_rows(latents, masks, lengths, outs_fi, ctx, mask, max_seq=max_seq)
            frames_done += len(outs_fi)
            latents.flush()
            masks.flush()
            lengths.flush()

        latents.flush()
        masks.flush()
        lengths.flush()
        n_new = len(missing)
        hz = n_new / max(time.time() - ep_t0, 1e-6) if n_new else 0.0
        global_hz = frames_done / max(time.time() - t0, 1e-6)
        print(
            f"  [{tag}] ep{ep_i:06d} T={T} new={n_new} done ({hz:.2f} f/s ep, {global_hz:.2f} f/s shard)",
            flush=True,
        )

    print(
        f"[dual_vlm_cache][{tag}] done frames={frames_done} elapsed={time.time()-t0:.1f}s → {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
