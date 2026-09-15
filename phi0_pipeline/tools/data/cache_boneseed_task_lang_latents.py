#!/usr/bin/env python3
"""Cache BoneSEED ``tasks.parquet`` language through Qwen3-VL (last-layer action context).

Reads the same dataset meta as ``rewrite_boneseed_smpl_gmr_relroot.py`` (default
``smpl_gmr_relroot_phi0``). Does **not** touch ``data/*.parquet``.

Encode path matches training::

    build_text_only_vlm_chat_inputs → Qwen3VLTower.extract_action_context
    → hidden_states[-1]

Multi-GPU (6 cards)::

    # once: create empty sidecar
    python tools/data/cache_boneseed_task_lang_latents.py --init-only ...
    # then parallel (see run_cache_boneseed_task_lang_latents_6gpu.sh)
    CUDA_VISIBLE_DEVICES=i python ... --shard i/6 --resume --batch-size 64

Dataloader::

    from phi0.online.lang_latents import TaskLangLatentCache
    cache = TaskLangLatentCache.open(REF_ROOT)
    ctx, mask = cache.get_batch(task_indices)
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

from phi0.models.vlm.preprocess import build_text_only_vlm_chat_inputs  # noqa: E402
from phi0.models.vlm.tower import (  # noqa: E402
    OFFICIAL_QWEN3VL_INSTRUCT,
    QWEN3VL_HIDDEN_DIM,
    Qwen3VLTower,
)
from phi0.online.lang_latents import (  # noqa: E402
    CACHE_DIRNAME,
    cache_dir_for_dataset,
    init_cache_files,
    open_cache_writable,
    save_lengths,
)
from phi0.online.task_prompts import DEFAULT_DISTILL_PROMPT  # noqa: E402

DEFAULT_DATASET = Path(
    "/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0"
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


def _load_task_rows(
    dataset_root: Path,
    *,
    text_col: str,
    max_tasks: int | None,
) -> tuple[list[int], list[str]]:
    path = dataset_root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    table = pq.read_table(path)
    names = set(table.column_names)
    if "task_index" not in names:
        raise RuntimeError(f"{path} missing task_index; cols={sorted(names)}")
    col = text_col if text_col in names else ("task" if "task" in names else "task_en")
    if col not in names:
        raise RuntimeError(f"{path} has no {text_col}/task/task_en; cols={sorted(names)}")
    idxs = table.column("task_index").to_pylist()
    texts = table.column(col).to_pylist()
    out_i: list[int] = []
    out_t: list[str] = []
    for i, t in zip(idxs, texts):
        s = str(t).strip() if t is not None else ""
        out_i.append(int(i))
        out_t.append(s if s else DEFAULT_DISTILL_PROMPT)
        if max_tasks is not None and len(out_i) >= int(max_tasks):
            break
    return out_i, out_t


def _unique_texts(texts: list[str]) -> tuple[list[str], dict[str, int]]:
    uid_texts: list[str] = []
    text_to_uid: dict[str, int] = {}
    for t in texts:
        if t not in text_to_uid:
            text_to_uid[t] = len(uid_texts)
            uid_texts.append(t)
    return uid_texts, text_to_uid


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if not spec:
        return 0, 1
    i_s, n_s = spec.split("/")
    i_s, n_s = int(i_s), int(n_s)
    if n_s < 1 or i_s < 0 or i_s >= n_s:
        raise SystemExit(f"bad --shard {spec!r}; want i/N with 0<=i<N")
    return i_s, n_s


@torch.inference_mode()
def _encode_batch(
    tower: Qwen3VLTower,
    prompts: list[str],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    processor = tower.processor
    batch = build_text_only_vlm_chat_inputs(
        processor,
        prompts,
        model_max_length=int(max_seq_len),
    )
    # Move to GPU once (processor returns CPU).
    device = next(tower.vlm_model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device=device, non_blocking=True)
    ctx, mask = tower.extract_action_context(
        batch["input_ids"],
        batch["attention_mask"],
        None,
        None,
        batch.get("mm_token_type_ids"),
    )
    return ctx, mask


def _write_batch(
    latents: np.memmap,
    masks: np.memmap,
    lengths: np.memmap,
    batch_uids: list[int],
    ctx: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_seq: int,
    hidden_dim: int,
) -> None:
    """Write one GPU batch into shared memmaps (per-row; safe for multi-writer)."""
    _b, s_cur, d = ctx.shape
    if d != hidden_dim:
        raise RuntimeError(f"ctx dim {d} != {hidden_dim}")
    if s_cur > max_seq:
        ctx = ctx[:, :max_seq]
        mask = mask[:, :max_seq]
        s_cur = max_seq
    # ponytail: row loop avoids advanced-index memmap write-through surprises
    ctx_np = ctx.float().cpu().numpy().astype(np.float16, copy=False)
    mask_np = mask.cpu().numpy().astype(np.uint8, copy=False)
    for j, uid in enumerate(batch_uids):
        latents[uid, :, :] = 0
        masks[uid, :] = 0
        latents[uid, :s_cur, :] = ctx_np[j]
        masks[uid, :s_cur] = mask_np[j]
        lengths[uid] = int(mask_np[j].sum())


def _prepare_tables(
    root: Path,
    *,
    text_col: str,
    max_tasks: int | None,
    max_unique: int | None,
) -> tuple[list[str], np.ndarray, list[int], int]:
    idxs, texts = _load_task_rows(root, text_col=text_col, max_tasks=max_tasks)
    uid_texts, text_to_uid = _unique_texts(texts)
    if max_unique is not None:
        uid_texts = uid_texts[: int(max_unique)]
        text_to_uid = {t: i for t, i in text_to_uid.items() if i < len(uid_texts)}
        keep = [(i, t) for i, t in zip(idxs, texts) if t in text_to_uid]
        idxs = [i for i, _ in keep]
        texts = [t for _, t in keep]
    max_ti = max(idxs) if idxs else 0
    task_to_uid = np.full((max_ti + 1,), -1, dtype=np.int32)
    for ti, t in zip(idxs, texts):
        task_to_uid[int(ti)] = int(text_to_uid[t])
    return uid_texts, task_to_uid, idxs, max_ti


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--vlm-path", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64, help="prompts per VLM forward")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--text-col", type=str, default="task")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--max-unique", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip uids with lengths>0")
    ap.add_argument("--init-only", action="store_true", help="create empty sidecar then exit")
    ap.add_argument("--shard", type=str, default=None, help="i/N over unique uid space")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--flush-every", type=int, default=4, help="flush memmap every N batches")
    args = ap.parse_args()

    shard_i, shard_n = _parse_shard(args.shard)
    root = args.dataset_root
    uid_texts, task_to_uid, idxs, max_ti = _prepare_tables(
        root,
        text_col=args.text_col,
        max_tasks=args.max_tasks,
        max_unique=args.max_unique,
    )
    out_dir = cache_dir_for_dataset(root)
    vlm_path = _resolve_vlm_path(args.vlm_path)
    hidden_dim = QWEN3VL_HIDDEN_DIM
    max_seq = int(args.max_seq_len)
    u = len(uid_texts)

    tag = f"shard={shard_i}/{shard_n}" if shard_n > 1 else "single"
    print(
        f"[lang_latents][{tag}] root={root} unique={u} task_rows={len(idxs)} "
        f"out={out_dir} vlm={vlm_path} batch={args.batch_size}",
        flush=True,
    )

    meta_blob = {
        "dataset_root": str(root),
        "vlm_path": vlm_path,
        "text_col": args.text_col,
        "dtype_store": "float16",
        "encode": "build_text_only_vlm_chat_inputs + extract_action_context (last hidden)",
    }

    have_cache = (out_dir / "meta.json").is_file() and (out_dir / "latents.fp16.dat").is_file()
    if args.init_only:
        if have_cache:
            print(f"[lang_latents][{tag}] already initialized at {out_dir}", flush=True)
            return
        init_cache_files(
            out_dir,
            num_unique=u,
            max_seq_len=max_seq,
            hidden_dim=hidden_dim,
            max_task_index=max_ti,
            meta=meta_blob,
            uid_texts=uid_texts,
            task_index_to_uid=task_to_uid,
        )
        print(f"[lang_latents][{tag}] initialized empty cache at {out_dir}", flush=True)
        return

    if not have_cache:
        if shard_i != 0:
            raise SystemExit(
                f"[{tag}] cache missing; run --init-only on rank0 first, or use the 6gpu launcher"
            )
        latents, masks, lengths = init_cache_files(
            out_dir,
            num_unique=u,
            max_seq_len=max_seq,
            hidden_dim=hidden_dim,
            max_task_index=max_ti,
            meta=meta_blob,
            uid_texts=uid_texts,
            task_index_to_uid=task_to_uid,
        )
        print(f"[lang_latents][{tag}] initialized empty cache at {out_dir}", flush=True)
    else:
        meta, latents, masks, lengths = open_cache_writable(out_dir)
        if int(meta.get("num_unique", -1)) != u or int(meta.get("max_seq_len", -1)) != max_seq:
            raise SystemExit(
                f"[{tag}] resume meta mismatch unique/seq "
                f"{meta.get('num_unique')}/{meta.get('max_seq_len')} vs {u}/{max_seq}"
            )
        # refresh map/texts (idempotent)
        np.save(out_dir / "task_index_to_uid.npy", task_to_uid.astype(np.int32))

    shard_uids = [i for i in range(u) if (i % shard_n) == shard_i]
    if not args.resume:
        for i in shard_uids:
            lengths[i] = 0
        lengths.flush()
        todo = shard_uids
    else:
        todo = [i for i in shard_uids if int(lengths[i]) <= 0]

    print(
        f"[lang_latents][{tag}] todo={len(todo)}/{u} "
        f"(done_global={int((np.asarray(lengths) > 0).sum())})",
        flush=True,
    )
    if not todo:
        print(f"[lang_latents][{tag}] nothing to encode", flush=True)
        return

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # Prefer visible device 0 when CUDA_VISIBLE_DEVICES isolates one GPU.
    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else device
    tower = Qwen3VLTower.from_pretrained(
        model_path=vlm_path,
        device=device,
        torch_dtype=torch_dtype,
        freeze=True,
        local_files_only=Path(vlm_path).is_dir(),
    )
    live_dim = int(getattr(tower, "action_context_dim", hidden_dim))
    if live_dim != hidden_dim:
        raise SystemExit(f"VLM action_context_dim={live_dim} != expected {hidden_dim}")

    bs = max(1, int(args.batch_size))
    flush_every = max(1, int(args.flush_every))
    t0 = time.time()
    done = 0
    for start in range(0, len(todo), bs):
        batch_uids = todo[start : start + bs]
        prompts = [uid_texts[i] for i in batch_uids]
        ctx, mask = _encode_batch(tower, prompts, max_seq_len=max_seq)
        _write_batch(
            latents,
            masks,
            lengths,
            batch_uids,
            ctx,
            mask,
            max_seq=max_seq,
            hidden_dim=hidden_dim,
        )
        done += len(batch_uids)
        if (start // bs + 1) % flush_every == 0 or done == len(todo):
            latents.flush()
            masks.flush()
            save_lengths(out_dir, lengths)
            hz = done / max(time.time() - t0, 1e-6)
            print(
                f"  [{tag}] encoded {done}/{len(todo)} unique ({hz:.2f} uid/s)",
                flush=True,
            )

    latents.flush()
    masks.flush()
    save_lengths(out_dir, lengths)
    if shard_i == 0:
        (out_dir / "uid_texts.json").write_text(
            json.dumps(uid_texts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_path = out_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "num_unique": u,
                "num_task_rows": len(idxs),
                "cache_dirname": CACHE_DIRNAME,
                "last_shard_elapsed_sec": time.time() - t0,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[lang_latents][{tag}] done → {out_dir} elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
