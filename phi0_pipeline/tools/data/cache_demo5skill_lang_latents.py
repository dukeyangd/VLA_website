#!/usr/bin/env python3
"""Encode demo5skill distinct prompts → ``meta/lang_latents_qwen3vl_demo5skill/``.

Reads ``meta/demo5skill_task_prompts.json`` (short Chinese = DEMO5_PROMPTS),
writes a sparse task_index→uid cache under BoneSEED (5 demo episodes).

Usage::

    CUDA_VISIBLE_DEVICES=0 python tools/data/cache_demo5skill_lang_latents.py
    PHI0_LANG_LATENTS_DIRNAME=lang_latents_qwen3vl_demo5skill ...  # train / viz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phi0.models.vlm.preprocess import build_text_only_vlm_chat_inputs  # noqa: E402
from phi0.models.vlm.tower import (  # noqa: E402
    OFFICIAL_QWEN3VL_INSTRUCT,
    QWEN3VL_HIDDEN_DIM,
    Qwen3VLTower,
)
from phi0.online.lang_latents import init_cache_files  # noqa: E402

DEFAULT_DATASET = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
)
DEFAULT_PROMPTS = _ROOT / "meta" / "demo5skill_task_prompts.json"
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


@torch.inference_mode()
def _encode(
    tower: Qwen3VLTower,
    prompts: list[str],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = build_text_only_vlm_chat_inputs(
        tower.processor,
        prompts,
        model_max_length=int(max_seq_len),
    )
    device = next(tower.vlm_model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device=device, non_blocking=True)
    return tower.extract_action_context(
        batch["input_ids"],
        batch["attention_mask"],
        None,
        None,
        batch.get("mm_token_type_ids"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--prompts-json", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--vlm-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16"))
    args = ap.parse_args()

    blob = json.loads(args.prompts_json.read_text(encoding="utf-8"))
    dirname = str(blob.get("cache_dirname") or "lang_latents_qwen3vl_demo5skill")
    prompts_map = {int(k): str(v).strip() for k, v in dict(blob["prompts"]).items()}
    if len(prompts_map) < 2:
        raise SystemExit("need ≥2 prompts in prompts-json")
    task_indices = sorted(prompts_map)
    uid_texts = [prompts_map[ti] for ti in task_indices]
    if len(set(uid_texts)) != len(uid_texts):
        raise SystemExit("demo5 prompts must be unique strings")

    max_ti = max(task_indices)
    task_to_uid = np.full((max_ti + 1,), -1, dtype=np.int32)
    for uid, ti in enumerate(task_indices):
        task_to_uid[ti] = uid

    out_dir = args.dataset_root / "meta" / dirname
    vlm_path = _resolve_vlm_path(args.vlm_path)
    max_seq = int(args.max_seq_len)
    hidden = QWEN3VL_HIDDEN_DIM
    print(
        f"[demo5_lang] out={out_dir} n={len(uid_texts)} max_ti={max_ti} vlm={vlm_path}",
        flush=True,
    )
    for ti, t in zip(task_indices, uid_texts):
        print(f"  task_index={ti}: {t[:60]}…", flush=True)

    latents, masks, lengths = init_cache_files(
        out_dir,
        num_unique=len(uid_texts),
        max_seq_len=max_seq,
        hidden_dim=hidden,
        max_task_index=max_ti,
        meta={
            "dataset_root": str(args.dataset_root),
            "vlm_path": vlm_path,
            "prompts_json": str(args.prompts_json.resolve()),
            "dtype_store": "float16",
            "encode": "build_text_only_vlm_chat_inputs + extract_action_context (last hidden)",
            "demo5skill": True,
        },
        uid_texts=uid_texts,
        task_index_to_uid=task_to_uid,
    )

    from phi0.models.vlm.attn import allow_sdpa_attn_fallback, resolve_vlm_attn_implementation

    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
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
        print(f"[demo5_lang] {attn} → sdpa ({type(exc).__name__})", flush=True)
        tower = Qwen3VLTower.from_pretrained(
            model_path=vlm_path,
            device=device,
            torch_dtype=dtype,
            freeze=True,
            attn_implementation="sdpa",
            local_files_only=Path(vlm_path).is_dir(),
        )
        attn = "sdpa"
    print(f"[demo5_lang] vlm_attn={attn}", flush=True)
    tower.eval()
    ctx, mask = _encode(tower, uid_texts, max_seq_len=max_seq)
    b, s_cur, d = ctx.shape
    if d != hidden:
        raise RuntimeError(f"ctx dim {d} != {hidden}")
    if s_cur > max_seq:
        ctx = ctx[:, :max_seq]
        mask = mask[:, :max_seq]
        s_cur = max_seq
    ctx_np = ctx.float().cpu().numpy().astype(np.float16, copy=False)
    mask_np = mask.cpu().numpy().astype(np.uint8, copy=False)
    for uid in range(b):
        latents[uid, :, :] = 0
        masks[uid, :] = 0
        latents[uid, :s_cur, :] = ctx_np[uid]
        masks[uid, :s_cur] = mask_np[uid]
        lengths[uid] = int(mask_np[uid].sum())
    latents.flush()
    masks.flush()
    lengths.flush()
    np.save(out_dir / "lengths.npy", np.asarray(lengths, dtype=np.int32))

    def _report(name: str, vecs: list[np.ndarray]) -> None:
        V = np.stack(vecs)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        sim = V @ V.T
        off = sim[np.triu_indices(b, k=1)]
        print(f"[demo5_lang] {name} cosine:\n{np.array2string(sim, precision=3)}", flush=True)
        print(
            f"[demo5_lang] {name} offdiag mean={float(off.mean()):.3f} "
            f"max={float(off.max()):.3f} min={float(off.min()):.3f}",
            flush=True,
        )

    mean_vecs, last_vecs = [], []
    for uid in range(b):
        m = mask_np[uid].astype(bool)
        idx = np.flatnonzero(m)
        mean_vecs.append(ctx_np[uid][m].astype(np.float32).mean(0))
        last_vecs.append(ctx_np[uid, int(idx[-1])].astype(np.float32))
        print(
            f"  uid{uid} len={int(m.sum())} ||mean||={float(np.linalg.norm(mean_vecs[-1])):.3f}",
            flush=True,
        )
    _report("mean-pool", mean_vecs)
    _report("last-token", last_vecs)
    print(
        f"[demo5_lang] done. export PHI0_LANG_LATENTS_DIRNAME={dirname} for train/viz",
        flush=True,
    )


if __name__ == "__main__":
    main()
