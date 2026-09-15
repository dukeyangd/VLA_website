"""Training catalog helpers + metrics parsing for Humanoid Data Studio."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
CATALOG_PATH = APP_DIR / "phi0_pipeline" / "train_catalog.json"
SCREENED_QA_REL = "meta/sonic_qa_valid_invalid.json"
VISION_ALLOWLIST_REL = "meta/vision_episode_allowlist.json"
ALL_ALLOWLIST_REL = "meta/all_episode_allowlist.json"
READY_FILES = (
    "meta/stats.json",
    "meta/vision_episode_allowlist.json",
    "meta/vlm_frame_latents_qwen3vl_dual/meta.json",
)
MIX_READY_FILES = READY_FILES + (ALL_ALLOWLIST_REL,)

# Single generic entry: historical run_830_* wrappers are documentation only.
LAUNCHER_SCRIPT = "tools/train/run_online_vlm_mix_distill.sh"
# Pack selected raw sessions (valid-only manifest) then call LAUNCHER_SCRIPT.
SELECTION_PACK_LAUNCHER = "tools/train/run_studio_selection_pack_and_distill.sh"
# Per-skill prompt+datasets → pack each → merge → distill (mix compose).
MIX_SLOTS_PACK_LAUNCHER = "tools/train/run_studio_mix_slots_pack_and_distill.sh"
MANIFEST_DIR = APP_DIR / "state" / "train_manifests"
# Legacy tmp packs (no longer written; cleanup may still refuse-delete outside this root).
EPHEMERAL_PACK_ROOT = Path("/mnt/data3/wpy/tmp/studio_train_packs")
# Student ckpts / distill metrics (not packs, not VLM cache).
DEFAULT_TRAIN_OUT_BASE = "/mnt/data2/wpy/workspace/Phi_0_model_zoo"
# Packed train datasets live here while training (same absolute path on train hosts).
DEFAULT_TRAIN_PACK_BASE = "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data"
# After a successful train: copy pack here, verify, then delete the data2 copy.
ARCHIVE_TRAIN_PACK_BASE = "/mnt/efs_1/gzy/workspace/Phi_0_train_data"
LEGACY_TRAIN_OUT_BASE = "/mnt/data3/wpy"
_TRAIN_OUT_MARKERS = (
    "phi0_student_last.pt",
    "distill_metrics.jsonl",
    "distill_metrics.json",
    "RESOLVED_TRAIN_SETTINGS.yaml",
)


def train_out_dir_occupied(path: str | Path) -> bool:
    """True if this folder already holds a previous distill run."""
    root = Path(str(path or "").strip())
    if not str(root) or not root.exists():
        return False
    if not root.is_dir():
        return True
    return any((root / name).exists() for name in _TRAIN_OUT_MARKERS)


def allocate_train_out_dir(
    *,
    out_base: str,
    skill_key: str,
    stamp: str,
    requested: str = "",
) -> str:
    """Always return a unique run folder; never reuse a directory with old ckpts."""
    fresh = f"{str(out_base).rstrip('/')}/{skill_key}_{stamp}"
    want = str(requested or "").strip()
    if not want:
        return fresh
    if train_out_dir_occupied(want):
        return fresh
    return want


def allocate_train_pack_dir(
    *,
    pack_base: str,
    skill_key: str,
    stamp: str,
) -> str:
    """Unique pack folder: ``<pack_base>/<skill>_<stamp>``, never reuse an existing dir."""
    fresh = f"{str(pack_base).rstrip('/')}/{skill_key}_{stamp}"
    if not Path(fresh).exists():
        return fresh
    for i in range(2, 50):
        cand = f"{fresh}_{i}"
        if not Path(cand).exists():
            return cand
    return f"{fresh}_{int(time.time())}"


# Modality / hand recipes applied as env overlays (skill-adaptive).
TRAIN_PROFILES: dict[str, dict[str, Any]] = {
    "vision_teleop": {
        "title": "纯视觉 teleop",
        "description": "nosim vision_dl + VLM frame cache；无 Isaac 文本轨。",
        "fixed_flags": {
            "PHI0_P_VISION": "1",
            "PHI0_VISION_DL_ONLY": "1",
            "PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY": "0",
            "PHI0_DAGGER_ON_NO_VIDEO": "0",
            "VISION_ONLY": "1",
            "PHI0_ALLOW_ZERO_HAND": "0",
            "PHI0_ZERO_PROPRIO_HAND": "0",
        },
    },
    "vision_teleop_handcmd": {
        "title": "视觉 teleop + handcmd",
        "description": "commanded hand proprio（lag=1）；无 RTC。",
        "fixed_flags": {
            "PHI0_P_VISION": "1",
            "PHI0_VISION_DL_ONLY": "1",
            "PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY": "0",
            "PHI0_DAGGER_ON_NO_VIDEO": "0",
            "VISION_ONLY": "1",
            "PHI0_ALLOW_ZERO_HAND": "0",
            "PHI0_TRAIN_HAND_OBS": "commanded",
            "PHI0_HAND_PROPRIO_LAG": "1",
            "PHI0_ZERO_PROPRIO_HAND": "0",
            "PHI0_DISTILL_RTC": "0",
        },
    },
    "vision_teleop_handcmd_rtc": {
        "title": "视觉 teleop + handcmd + RTC",
        "description": "skill1 walk 配方：commanded lag=1 + distill RTC。",
        "fixed_flags": {
            "PHI0_P_VISION": "1",
            "PHI0_VISION_DL_ONLY": "1",
            "PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY": "0",
            "PHI0_DAGGER_ON_NO_VIDEO": "0",
            "VISION_ONLY": "1",
            "PHI0_ALLOW_ZERO_HAND": "0",
            "PHI0_TRAIN_HAND_OBS": "commanded",
            "PHI0_HAND_PROPRIO_LAG": "1",
            "PHI0_ZERO_PROPRIO_HAND": "0",
            "PHI0_DISTILL_RTC": "1",
            "PHI0_DISTILL_RTC_MAX_DELAY": "8",
        },
    },
    "mix_vision_isaac": {
        "title": "混训 vision+Isaac",
        "description": "需已 pack 的 830mix_*_unified（含 demo5 文本轨 + all_episode_allowlist）。",
        "fixed_flags": {
            "PHI0_P_VISION": "0.9",
            "PHI0_VISION_DL_ONLY": "0",
            "PHI0_VISION_DL_ISAAC_NO_VIDEO_ONLY": "1",
            "PHI0_DAGGER_ON_NO_VIDEO": "1",
            "VISION_ONLY": "0",
            "PHI0_ALLOW_ZERO_HAND": "1",
            "PHI0_TRAIN_HAND_OBS": "commanded",
            "PHI0_HAND_PROPRIO_LAG": "1",
            "PHI0_ZERO_PROPRIO_HAND": "0",
            "PHI0_DISTILL_RTC": "1",
            "PHI0_DISTILL_RTC_MAX_DELAY": "8",
        },
    },
}


def _episode_list_from_payload(raw: Any) -> list[int]:
    if isinstance(raw, list):
        return sorted({int(x) for x in raw})
    if isinstance(raw, dict):
        eps = raw.get("episode_index") or raw.get("episodes") or raw.get("valid") or []
        if isinstance(eps, list):
            return sorted({int(x) for x in eps})
    return []


def read_screened_valid(dataset_path: str) -> dict[str, Any]:
    """Read 02-exported valid pool for train/infer.

    Prefer meta/sonic_qa_valid_invalid.json (legacy 02「导出 valid」allowlist).
    Then session labels.json / flat {valid,invalid} (current 02 default).
    Fall back to vision_episode_allowlist.json (may be pack full dump → screened=False).
    """
    root = Path(dataset_path)
    qa_path = root / SCREENED_QA_REL
    vision_path = root / VISION_ALLOWLIST_REL
    labels_path = root / "labels.json"
    out: dict[str, Any] = {
        "path": str(root),
        "valid": [],
        "invalid": [],
        "screened": False,
        "source": None,
        "qa_path": str(qa_path) if qa_path.is_file() else None,
        "allowlist_path": str(vision_path) if vision_path.is_file() else None,
        "labels_path": str(labels_path) if labels_path.is_file() else None,
        "valid_count": 0,
        "invalid_count": 0,
    }
    if qa_path.is_file():
        try:
            raw = json.loads(qa_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
        if isinstance(raw, dict):
            valid = _episode_list_from_payload(raw.get("valid") if "valid" in raw else raw)
            invalid = _episode_list_from_payload(raw.get("invalid") or [])
            invalid = [ep for ep in invalid if ep not in set(valid)]
            out.update(
                valid=valid,
                invalid=invalid,
                screened=bool(valid),
                source="sonic_qa",
                valid_count=len(valid),
                invalid_count=len(invalid),
            )
            return out
    if labels_path.is_file():
        try:
            raw = json.loads(labels_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
        if isinstance(raw, dict):
            session = root.name
            entry = raw.get(session) if isinstance(raw.get(session), dict) else raw
            if isinstance(entry, dict) and ("valid" in entry or "invalid" in entry):
                valid = _episode_list_from_payload(entry.get("valid") or [])
                invalid = _episode_list_from_payload(entry.get("invalid") or [])
                invalid = [ep for ep in invalid if ep not in set(valid)]
                out.update(
                    valid=valid,
                    invalid=invalid,
                    screened=bool(valid),
                    source="labels",
                    valid_count=len(valid),
                    invalid_count=len(invalid),
                )
                return out
    if vision_path.is_file():
        try:
            raw = json.loads(vision_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
        valid = _episode_list_from_payload(raw)
        out.update(
            valid=valid,
            invalid=[],
            screened=False,
            source="vision_allowlist",
            valid_count=len(valid),
            invalid_count=0,
        )
    return out


def require_screened_valid(dataset_path: str, *, what: str = "训练") -> dict[str, Any]:
    """Raise if dataset has not been filtered via 02 export."""
    info = read_screened_valid(dataset_path)
    if info.get("screened") and info.get("valid"):
        return info
    if info.get("valid") and not info.get("screened"):
        raise ValueError(
            f"数据集 allowlist 仍是打包全量，尚未经 02 筛选导出。"
            f"请先在 02 点「导出 valid」后再{what}"
            f"（需要 labels.json 或 meta/sonic_qa_valid_invalid.json）"
        )
    raise ValueError(
        f"未找到 02 筛选后的 valid 列表。请先在 02 标注并「导出 valid」后再{what}"
        f"（读 session/labels.json 或 meta/sonic_qa_valid_invalid.json）"
    )


def require_infer_episode_pool(dataset_path: str) -> dict[str, Any]:
    """Episode pool for 04 closed-loop.

    Prefer 02-screened labels / sonic_qa. If missing, accept pack
    vision/all allowlist (typical for already-trained *_unified REF).
    """
    info = read_screened_valid(dataset_path)
    if info.get("valid"):
        return info
    raise ValueError(
        "未找到可用 episode 列表。请确认 REF 有 meta/vision_episode_allowlist.json，"
        "或先在 02 导出 valid（labels.json / sonic_qa_valid_invalid.json）"
    )


def is_unified_path(path: str | None) -> bool:
    blob = str(path or "").lower().replace("\\", "/")
    return "unified" in blob or "/datasets/830/" in blob


def is_raw_session_path(path: str | None) -> bool:
    """830demo / V2.1 session folder (not a packed *_unified REF)."""
    p = str(path or "").strip().rstrip("/")
    if not p:
        return False
    blob = p.lower().replace("\\", "/")
    if is_unified_path(p) or is_mix_dataset_path(p):
        return False
    name = Path(p).name
    if name.endswith(".json"):
        return False
    # Typical session id: 2026-09-11-19-46-45
    if re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name):
        return True
    if "/830demo/" in blob or "/820demo/" in blob:
        return True
    # Heuristic: has episode parquet layout
    root = Path(p)
    if root.is_dir() and (root / "data").is_dir() and (root / "meta").is_dir():
        return "unified" not in blob
    return False


def _episode_ints(raw: Any) -> list[int]:
    return _episode_list_from_payload(raw)


def load_session_label_entry(session_path: str) -> dict[str, Any]:
    """Load {valid,invalid} for one session from labels.json or parent skill_N.json."""
    root = Path(str(session_path).rstrip("/"))
    session = root.name
    empty = {"valid": [], "invalid": [], "source": None, "path": str(root)}

    def _from_dict(raw: dict[str, Any], *, source: str) -> dict[str, Any] | None:
        entry = raw.get(session) if isinstance(raw.get(session), dict) else raw
        if not isinstance(entry, dict):
            return None
        if "valid" not in entry and "invalid" not in entry:
            return None
        valid = _episode_ints(entry.get("valid") or [])
        invalid = [ep for ep in _episode_ints(entry.get("invalid") or []) if ep not in set(valid)]
        return {
            "valid": valid,
            "invalid": invalid,
            "source": source,
            "path": str(root),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    labels = root / "labels.json"
    if labels.is_file():
        try:
            raw = json.loads(labels.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
        if isinstance(raw, dict):
            hit = _from_dict(raw, source="labels.json")
            if hit and hit["valid"]:
                return hit

    parent = root.parent
    for cand in sorted(parent.glob("skill*.json")) + sorted(parent.glob("*_labels.json")):
        if cand.name.endswith(".lock"):
            continue
        try:
            raw = json.loads(cand.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        hit = _from_dict(raw, source=cand.name)
        if hit and (hit["valid"] or hit["invalid"]):
            return hit

    # Fall back to screened helpers (sonic_qa / vision allowlist on session)
    screened = read_screened_valid(str(root))
    if screened.get("valid"):
        return {
            "valid": list(screened["valid"]),
            "invalid": list(screened.get("invalid") or []),
            "source": screened.get("source"),
            "path": str(root),
            "valid_count": len(screened["valid"]),
            "invalid_count": len(screened.get("invalid") or []),
        }
    return empty


def build_valid_only_manifest(session_paths: list[str]) -> dict[str, Any]:
    """Merge selected sessions into pack manifest: only ``valid`` episodes (drop invalid)."""
    manifest: dict[str, Any] = {}
    empty_valid: list[str] = []
    total_valid = 0
    total_invalid = 0
    for raw in session_paths:
        path = str(raw or "").strip().rstrip("/")
        if not path:
            continue
        entry = load_session_label_entry(path)
        session = Path(path).name
        valid = list(entry.get("valid") or [])
        invalid = list(entry.get("invalid") or [])
        if not valid:
            empty_valid.append(session)
            continue
        manifest[session] = {
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "source_path": path,
            "source": entry.get("source"),
        }
        total_valid += len(valid)
        total_invalid += len(invalid)
    if not manifest:
        detail = "、".join(empty_valid[:6]) or "无有效路径"
        raise ValueError(
            "勾选的数据集没有可用的 valid episode（invalid 不会进入训练）。"
            f"请先在 02 标注并导出 valid。空 valid：{detail}"
        )
    return {
        "manifest": manifest,
        "session_count": len(manifest),
        "valid_count": total_valid,
        "invalid_count": total_invalid,
        "skipped_empty_valid": empty_valid,
    }


def write_valid_only_manifest(
    session_paths: list[str],
    *,
    dest: str | Path,
) -> dict[str, Any]:
    built = build_valid_only_manifest(session_paths)
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pack script only reads session → {valid:[...]}; keep invalid for audit.
    payload = {
        sid: {
            "valid": row["valid"],
            "invalid": row.get("invalid") or [],
            "valid_count": row["valid_count"],
            "invalid_count": row.get("invalid_count") or 0,
        }
        for sid, row in built["manifest"].items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    built["path"] = str(path)
    built["fingerprint"] = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return built


def write_episode_allowlist(episodes: list[int], dest: str | Path) -> str:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"episode_index": sorted({int(x) for x in episodes})}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return str(path)


def is_skill_dataset_folder(path: str | None) -> bool:
    """Parent folder that contains timestamped session dirs (e.g. 830demo/skill5)."""
    root = Path(str(path or "").strip().rstrip("/"))
    if not root.is_dir() or is_unified_path(str(root)):
        return False
    if is_raw_session_path(str(root)) and re.match(
        r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", root.name
    ):
        return False
    kids = [
        p for p in root.iterdir()
        if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", p.name)
    ]
    return bool(kids)


def expand_train_selection_paths(paths: list[str]) -> list[str]:
    """Expand skill folders → child sessions; keep explicit session paths."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        p = str(raw or "").strip().rstrip("/")
        if not p:
            continue
        if is_skill_dataset_folder(p):
            for child in sorted(Path(p).iterdir()):
                if not child.is_dir():
                    continue
                cp = str(child)
                if not is_raw_session_path(cp):
                    continue
                key = str(child.resolve()) if child.exists() else cp
                if key in seen:
                    continue
                seen.add(key)
                out.append(cp)
            continue
        key = str(Path(p).resolve()) if Path(p).exists() else p
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def materialize_raw_stage(session_paths: list[str], stage_dir: str | Path) -> dict[str, Any]:
    """Symlink selected sessions into one RAW_ROOT for pack (supports multi skill folders)."""
    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    sources: dict[str, str] = {}
    for raw in session_paths:
        src = Path(str(raw).rstrip("/"))
        if not src.is_dir():
            raise ValueError(f"session 不存在: {src}")
        name = src.name
        resolved = str(src.resolve())
        if name in sources and sources[name] != resolved:
            raise ValueError(
                f"session 名冲突「{name}」来自不同目录，无法合并到同一临时 RAW_ROOT：\n"
                f"- {sources[name]}\n- {resolved}"
            )
        sources[name] = resolved
        dest = stage / name
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir() and not dest.is_symlink():
            raise ValueError(f"临时 RAW 舞台被非软链占用: {dest}")
        if not dest.exists():
            dest.symlink_to(src.resolve(), target_is_directory=True)
        linked.append(name)
    return {
        "raw_root": str(stage),
        "sessions": linked,
        "sources": sources,
        "parent_count": len({str(Path(p).parent) for p in session_paths}),
    }


def classify_train_selection(paths: list[str], *, train_profile: str = "") -> dict[str, Any]:
    """Decide how to turn UI multi-select into a single REF for distill."""
    cleaned = expand_train_selection_paths(paths)
    cleaned = list(dict.fromkeys(cleaned))
    if not cleaned:
        raise ValueError("请先勾选或填写至少一个数据集路径")
    profile = normalize_train_profile(train_profile)
    mix = is_mix_train_profile(profile)

    # Drop phantom / missing *_unified leftovers (e.g. never-packed studio_* paths).
    missing_unified = [
        p for p in cleaned
        if is_unified_path(p) and not Path(p).is_dir()
    ]
    if missing_unified:
        cleaned = [p for p in cleaned if p not in set(missing_unified)]
    if not cleaned:
        raise ValueError(
            "勾选的 *_unified 路径不存在（可能是未完成的临时 pack）。"
            "请清空勾选后只选 830demo 下的 raw session，或选一个真实存在的 unified"
        )

    unified = [p for p in cleaned if is_unified_path(p)]
    raw = [p for p in cleaned if is_raw_session_path(p)]
    other = [p for p in cleaned if p not in unified and p not in raw]

    if mix:
        if len(cleaned) != 1 or not is_mix_dataset_path(cleaned[0]):
            raise ValueError(
                "混训请只选一个已 pack 的 830mix_*_unified（不要多选 raw session）"
            )
        if not Path(cleaned[0]).is_dir():
            raise ValueError(f"混训 REF 不存在: {cleaned[0]}")
        return {
            "mode": "unified",
            "paths": cleaned,
            "ref_root": cleaned[0],
            "raw_sessions": [],
            "needs_pack": False,
            "dropped_missing_unified": missing_unified,
        }

    if other:
        raise ValueError("无法识别的数据集路径: " + ", ".join(Path(p).name for p in other[:4]))

    # If phantom unified was mixed with raw sessions, prefer packing the raw sessions.
    if unified and raw:
        existing_uni = [p for p in unified if Path(p).is_dir()]
        if not existing_uni:
            unified = []
        else:
            raise ValueError(
                "请不要同时勾选 raw session 与 *_unified。"
                "多选 session 会按 valid 重新 pack；或只选一个已有 unified。"
            )
    if len(unified) > 1:
        raise ValueError(
            "不能同时训练多个 *_unified。请改选 raw session / 技能文件夹，"
            "或使用「混合训练」卡片选择已 pack 的 830mix_*_unified。"
        )
    if len(unified) == 1:
        if not Path(unified[0]).is_dir():
            raise ValueError(f"*_unified 路径不存在: {unified[0]}")
        return {
            "mode": "unified",
            "paths": unified,
            "ref_root": unified[0],
            "raw_sessions": [],
            "needs_pack": False,
            "dropped_missing_unified": missing_unified,
        }

    if not raw:
        raise ValueError("没有可 pack 的 raw session")

    parents = sorted({str(Path(p).parent) for p in raw})
    # Multi skill folders OK: plan_selection_pack stages them into one temporary RAW_ROOT.
    return {
        "mode": "pack_sessions",
        "paths": raw,
        "ref_root": "",
        "raw_sessions": raw,
        "raw_root": parents[0] if len(parents) == 1 else "",
        "raw_parents": parents,
        "needs_pack": True,
        "multi_root": len(parents) > 1,
        "dropped_missing_unified": missing_unified,
    }


def plan_selection_pack(
    session_paths: list[str],
    *,
    skill_id: str = "",
    prompt: str = "",
    stamp: str = "",
    job_id: str = "",
    pack_base: str | Path | None = None,
) -> dict[str, Any]:
    """Build valid-only manifest + persistent pack paths (kept after train).

    Packs land under ``DEFAULT_TRAIN_PACK_BASE/<skill>_<stamp>/`` (same naming as
    ckpt dirs). Original 830demo sessions are never modified.

    Sessions from multiple skill folders (e.g. skill5 + skill_5_throw_the_rubbish)
    are symlinked into one RAW_ROOT for the official pack script.
    """
    classified = classify_train_selection(session_paths, train_profile="vision_teleop")
    if not classified.get("needs_pack"):
        raise ValueError("当前选择不需要 pack")
    sessions = list(classified["raw_sessions"])
    parents = list(classified.get("raw_parents") or [])
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    skill_hint = str(skill_id or (Path(parents[0]).name if parents else "skill"))
    skill = re.sub(r"[^A-Za-z0-9_-]+", "", skill_hint)[:40] or "skill"
    job_key = re.sub(r"[^A-Za-z0-9_-]+", "", str(job_id or ""))[:48] or f"{skill}_{stamp}"
    pack_root = Path(
        allocate_train_pack_dir(
            pack_base=str(pack_base or DEFAULT_TRAIN_PACK_BASE),
            skill_key=skill,
            stamp=stamp,
        )
    )
    stage = materialize_raw_stage(sessions, pack_root / "raw_stage")
    raw_root = str(stage["raw_root"])
    built = write_valid_only_manifest(
        sessions,
        dest=MANIFEST_DIR / f"{skill}_{stamp}_{len(sessions)}sess_valid.json",
    )
    out_dir = str(pack_root / "pack_out")
    nvme_dir = str(pack_root / "studio_tmp_unified")
    # Name must contain "unified" — validate_train_ref / distill scripts expect that.
    ws_link = str(pack_root / "studio_tmp_unified_link")
    prompt = str(prompt or "").strip() or "task"
    return {
        "raw_root": raw_root,
        "raw_parents": parents,
        "multi_root": bool(classified.get("multi_root")),
        "stage_sources": stage.get("sources") or {},
        "manifest_path": built["path"],
        "fingerprint": built["fingerprint"],
        "session_count": built["session_count"],
        "valid_count": built["valid_count"],
        "invalid_count": built["invalid_count"],
        "skipped_empty_valid": built.get("skipped_empty_valid") or [],
        "sessions": list(built["manifest"].keys()),
        "session_paths": list(sessions),
        "pack_root": str(pack_root),
        "out_dir": out_dir,
        "nvme_dir": nvme_dir,
        "ws_link": ws_link,
        "ref_root": ws_link,
        "task_prompt": prompt,
        "skip_pack": False,
        "ephemeral": False,
        "stamp": stamp,
        "job_key": job_key,
    }


def plan_mix_slots_pack(
    slots: list[dict[str, Any]],
    *,
    stamp: str = "",
    job_id: str = "",
    pack_base: str | Path | None = None,
) -> dict[str, Any]:
    """Plan per-skill packs + merged mix unified under Phi_0_train_data.

    Each slot: ``{skill_id, prompt, dataset_paths:[raw sessions…]}``.
    Packs are kept (not deleted). Shared distill REF is the merged ws_link.
    """
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    job_key = re.sub(r"[^A-Za-z0-9_-]+", "", str(job_id or ""))[:48] or f"mix_{stamp}"
    root = Path(
        allocate_train_pack_dir(
            pack_base=str(pack_base or DEFAULT_TRAIN_PACK_BASE),
            skill_key="mix",
            stamp=stamp,
        )
    )
    if not slots or len(slots) < 2:
        raise ValueError("混合训练至少需要 2 个技能槽（各含 prompt + 数据集）")
    slot_plans: list[dict[str, Any]] = []
    total_valid = 0
    total_invalid = 0
    total_sessions = 0
    all_session_names: list[str] = []
    for i, raw_slot in enumerate(slots):
        if not isinstance(raw_slot, dict):
            raise ValueError(f"mix slot[{i}] 无效")
        skill_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "",
            str(raw_slot.get("skill_id") or raw_slot.get("id") or f"skill{i}"),
        )[:40] or f"skill{i}"
        prompt = resolve_train_prompt(
            raw_slot.get("prompt"),
            raw_slot.get("default_prompt"),
            raw_slot.get("title"),
        ) or "task"
        paths = raw_slot.get("dataset_paths") or raw_slot.get("paths") or []
        if isinstance(paths, str):
            paths = [ln.strip() for ln in paths.replace(",", "\n").splitlines() if ln.strip()]
        paths = [str(p).strip() for p in paths if str(p).strip()]
        if not paths:
            raise ValueError(f"技能 {skill_id} 未选择数据集")
        classified = classify_train_selection(paths, train_profile="vision_teleop")
        if not classified.get("needs_pack"):
            raise ValueError(
                f"技能 {skill_id} 请勾选 raw session（不要直接选 mix unified）"
            )
        sub = plan_selection_pack(
            list(classified["raw_sessions"]),
            skill_id=skill_id,
            prompt=prompt,
            stamp=stamp,
            job_id=f"{job_key}_{skill_id}",
            pack_base=str(root / "skills"),
        )
        slot_plans.append({
            "skill_id": skill_id,
            "title": str(raw_slot.get("title") or skill_id),
            "prompt": prompt,
            "dataset_paths": list(classified["raw_sessions"]),
            "pack_root": sub["pack_root"],
            "raw_root": sub["raw_root"],
            "manifest_path": sub["manifest_path"],
            "out_dir": sub["out_dir"],
            "nvme_dir": sub["nvme_dir"],
            "ws_link": sub["ws_link"],
            "valid_count": sub["valid_count"],
            "invalid_count": sub["invalid_count"],
            "session_count": sub["session_count"],
            "sessions": list(sub.get("sessions") or []),
            "stage_sources": sub.get("stage_sources") or {},
            "fingerprint": sub.get("fingerprint"),
        })
        total_valid += int(sub.get("valid_count") or 0)
        total_invalid += int(sub.get("invalid_count") or 0)
        total_sessions += int(sub.get("session_count") or 0)
        all_session_names.extend(list(sub.get("sessions") or []))

    merged_out = str(root / "studio_tmp_unified")
    ws_link = str(root / "studio_tmp_unified_link")
    return {
        "mix_slots_mode": True,
        "ephemeral": False,
        "stamp": stamp,
        "job_key": job_key,
        "pack_root": str(root),
        "out_dir": str(root / "pack_out"),
        "nvme_dir": merged_out,
        "ws_link": ws_link,
        "ref_root": ws_link,
        "merged_out": merged_out,
        "slots": slot_plans,
        "session_count": total_sessions,
        "valid_count": total_valid,
        "invalid_count": total_invalid,
        "sessions": all_session_names,
        "session_paths": [
            p for s in slot_plans for p in (s.get("dataset_paths") or [])
        ],
        "stage_sources": {
            k: v
            for s in slot_plans
            for k, v in (s.get("stage_sources") or {}).items()
        },
        "task_prompt": " | ".join(
            f"{s['skill_id']}:{s['prompt']}" for s in slot_plans
        ),
        "fingerprint": hashlib.sha1(
            json.dumps(
                [
                    {
                        "skill_id": s["skill_id"],
                        "prompt": s["prompt"],
                        "paths": s["dataset_paths"],
                        "fp": s.get("fingerprint"),
                    }
                    for s in slot_plans
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12],
        "skip_pack": False,
    }


def cleanup_ephemeral_pack(pack: dict[str, Any] | None) -> dict[str, Any]:
    """Delete temporary pack tree after train completes/cancels. Never touches raw sessions.

    If ``pack.keep_vlm_cache`` / ``pack.retain`` is truthy, skip deletion so the
    unified pack + VLM frame cache can be reused by a later job.
    """
    import shutil

    out: dict[str, Any] = {"ok": True, "removed": [], "skipped": [], "errors": []}
    if not isinstance(pack, dict) or not pack.get("ephemeral"):
        out["ok"] = True
        out["skipped"].append("not_ephemeral")
        return out
    if pack.get("keep_vlm_cache") or pack.get("retain") or pack.get("retained_for_reuse"):
        out["ok"] = True
        out["skipped"].append("keep_vlm_cache")
        out["retained_root"] = pack.get("pack_root")
        return out
    roots: list[str] = []
    pack_root = str(pack.get("pack_root") or "").strip()
    if pack_root:
        roots.append(pack_root)
    for key in ("out_dir", "nvme_dir", "ws_link", "ref_root"):
        p = str(pack.get(key) or "").strip()
        if p and p not in roots:
            # Only delete paths under the ephemeral root for safety.
            if pack_root and (p == pack_root or p.startswith(pack_root.rstrip("/") + "/")):
                roots.append(p)
            elif "/tmp/studio_train_packs/" in p.replace("\\", "/"):
                roots.append(p)
    # Prefer deleting the top pack_root once.
    if pack_root:
        roots = [pack_root]
    ephemeral_prefix = str(EPHEMERAL_PACK_ROOT.resolve())
    for raw in roots:
        path = Path(raw)
        try:
            resolved = str(path.resolve()) if path.exists() or path.is_symlink() else str(path)
        except OSError:
            resolved = str(path)
        if not resolved.startswith(ephemeral_prefix):
            out["skipped"].append(raw)
            out["errors"].append(f"refuse_delete_outside_ephemeral_root:{raw}")
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
                out["removed"].append(str(path))
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
                out["removed"].append(str(path))
            else:
                out["skipped"].append(str(path))
        except Exception as exc:  # noqa: BLE001
            out["ok"] = False
            out["errors"].append(f"{path}: {exc}")
    # Manifest under studio state is small; remove with pack.
    man = str(pack.get("manifest_path") or "").strip()
    if man:
        try:
            mp = Path(man)
            if mp.is_file() and str(mp.resolve()).startswith(str(MANIFEST_DIR.resolve())):
                mp.unlink(missing_ok=True)
                out["removed"].append(str(mp))
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"manifest:{exc}")
    return out


def load_train_catalog() -> dict[str, Any]:
    raw = json.loads(CATALOG_PATH.read_text("utf-8"))
    return raw


def known_train_profiles() -> list[str]:
    return list(TRAIN_PROFILES.keys())


def normalize_train_profile(name: str | None) -> str:
    key = str(name or "").strip()
    if not key:
        raise ValueError(
            "未指定 train_profile。请在技能卡设置 train_profile，"
            f"或匹配内置配方。可选: {', '.join(known_train_profiles())}"
        )
    if key not in TRAIN_PROFILES:
        raise ValueError(
            f"未知 train_profile={key!r}。可选: {', '.join(known_train_profiles())}"
        )
    return key


def profile_fixed_flags(profile: str) -> dict[str, str]:
    key = normalize_train_profile(profile)
    flags = dict((TRAIN_PROFILES[key].get("fixed_flags") or {}))
    return {str(k): str(v) for k, v in flags.items()}


def is_mix_train_profile(profile: str | None) -> bool:
    return str(profile or "").strip() == "mix_vision_isaac"


def is_mix_dataset_path(path: str | None) -> bool:
    blob = str(path or "").lower().replace("\\", "/")
    if "unified" not in blob:
        return False
    return "830mix" in blob or "/mix_" in blob or "_mix_" in blob or "mix_skill" in blob


def launcher_script(catalog: dict[str, Any] | None = None) -> str:
    catalog = catalog or {}
    return str(catalog.get("launcher") or catalog.get("script") or LAUNCHER_SCRIPT).strip() or LAUNCHER_SCRIPT


def validate_train_ref(ref_root: str, *, train_profile: str, require_local_files: bool = True) -> dict[str, Any]:
    """Ensure REF matches profile: unified+cache for teleop; mix pack for mix_vision_isaac."""
    profile = normalize_train_profile(train_profile)
    root = Path(str(ref_root or "").strip())
    if not str(root).startswith("/"):
        raise ValueError("数据集路径必须是绝对路径")
    if is_mix_train_profile(profile):
        if require_local_files and root.is_dir():
            missing = [rel for rel in MIX_READY_FILES if not (root / rel).is_file()]
            if missing:
                raise ValueError(
                    "混训 REF 未就绪，缺少: "
                    + ", ".join(missing)
                    + "。请先跑 pack pipeline（如 tools/data/run_830mix_skill123_demo5_pipeline.sh "
                    "或 run_830mix_skill1234_demo5_pipeline.sh）"
                )
        if require_local_files and root.is_dir() and not is_mix_dataset_path(str(root)):
            # Allow non-"mix" names only when all_episode_allowlist exists (already checked).
            if not (root / ALL_ALLOWLIST_REL).is_file():
                raise ValueError(
                    "混训请选择已合并的 830mix_*_unified（含 all_episode_allowlist），"
                    "不能把单技能 raw session / 单技能 unified 当作混训 REF"
                )
        elif not require_local_files and not is_mix_dataset_path(str(root)):
            raise ValueError(
                "混训请选择已合并的 830mix_*_unified 路径"
                f"（当前: {root.name}）"
            )
    else:
        if is_mix_dataset_path(str(root)):
            raise ValueError(
                f"单技能 train_profile={profile} 不能使用混训包 {root.name}。"
                "请选该技能自己的 *_unified，或改用「混合训练」卡片"
            )
        pack_planned = (
            str(EPHEMERAL_PACK_ROOT) in str(root).replace("\\", "/")
            or str(DEFAULT_TRAIN_PACK_BASE) in str(root).replace("\\", "/")
        )
        if "unified" not in str(root).lower() and not pack_planned:
            raise ValueError(
                "训练 REF 必须是 pack 后的 *_unified 目录（含 stats + allowlist + VLM cache），"
                f"不能直接用原始 session：{root}。"
                "请在 03 勾选 raw session（将 pack 到 Phi_0_train_data），或改选已有 *_unified"
            )
        if require_local_files and root.is_dir():
            missing = [rel for rel in READY_FILES if not (root / rel).is_file()]
            if missing:
                raise ValueError(
                    "数据集未就绪，缺少: "
                    + ", ".join(missing)
                    + "。请先 pack 为 *_unified 并 encode VLM frame cache"
                )
    return {"ok": True, "ref_root": str(root), "train_profile": profile}


def script_default_params(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults from builtin walk card / cluster_0 wrapper."""
    catalog = catalog or load_train_catalog()
    for task in catalog.get("tasks") or []:
        if task.get("id") in {"skill1_walk", "walk_blackbox"} and task.get("defaults"):
            return dict(task["defaults"])
    return {
        "ngpu": 8,
        "num_envs": 32,
        "horizon": 32,
        "epochs": 4,
        "extra_steps": 0,
        "lr": "1e-4",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "warmup_steps": 0,
        "ckpt_every_epoch": 1,
        "ckpt_every": 0,
        "ckpt_step_keep": 3,
        "hand_mode": "dex3",
        "deploy_policy": "sonic_v1_1",
    }


def merge_catalog_tasks(catalog: dict[str, Any], custom_tasks: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Builtin catalog tasks + persisted custom skill cards."""
    out = dict(catalog)
    builtin = list(catalog.get("tasks") or [])
    builtin_ids = {t.get("id") for t in builtin}
    extras = []
    for task in custom_tasks or []:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "").strip()
        if not tid or tid in builtin_ids:
            continue
        extras.append({**task, "custom": True})
    out["tasks"] = builtin + extras
    return out


def normalize_custom_task(data: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    catalog = catalog or load_train_catalog()
    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError("请填写技能名称 / 卡片标题")
    skill = str(data.get("skill") or "").strip() or _slugify(title)
    badge = str(data.get("badge") or skill or "SKILL").strip().upper()[:12]
    prompt = str(data.get("default_prompt") or data.get("prompt") or "").strip()
    dataset_path = str(data.get("dataset_path") or data.get("path") or "").strip()
    if not dataset_path:
        raise ValueError("请填写数据集路径")
    if not dataset_path.startswith("/"):
        raise ValueError("数据集路径必须是绝对路径")
    remote_path = str(data.get("remote_path") or dataset_path).strip()
    if remote_path and not remote_path.startswith("/"):
        raise ValueError("远端数据集路径必须是绝对路径")
    tid = str(data.get("id") or "").strip()
    if not tid:
        tid = f"custom_{_slugify(skill)}_{int(time.time()) % 100000:05d}"
    if not tid.startswith("custom_"):
        tid = f"custom_{_slugify(tid)}"
    reserved = {t.get("id") for t in (catalog.get("tasks") or [])}
    if tid in reserved and not data.get("id"):
        tid = f"{tid}_{int(time.time()) % 1000}"
    defaults = script_default_params(catalog)
    user_defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    defaults.update({k: user_defaults[k] for k in user_defaults})
    if data.get("epochs") is not None:
        defaults["epochs"] = int(data["epochs"])
    if data.get("num_envs") is not None:
        defaults["num_envs"] = int(data["num_envs"])
    if data.get("ngpu") is not None:
        defaults["ngpu"] = int(data["ngpu"])
    ds_id = _slugify(Path(dataset_path).name) or "dataset"
    task = {
        "id": tid,
        "title": title,
        "skill": skill,
        "badge": badge,
        "subtitle": str(data.get("subtitle") or "自定义技能 · 同套 nosim 蒸馏配方").strip(),
        "description": str(data.get("description") or "用户保存的技能训练卡片，下次可直接选用。").strip(),
        "accent": str(data.get("accent") or "#4a6fa5").strip() or "#4a6fa5",
        "default_prompt": prompt,
        "custom": True,
        "datasets": [
            {
                "id": ds_id,
                "label": str(data.get("dataset_label") or Path(dataset_path).name),
                "path": dataset_path,
                "remote_path": remote_path or dataset_path,
            }
        ],
        "defaults": defaults,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return task


def _slugify(text: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "_", str(text or "").strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s[:48] or "skill"


def resolve_local_phi0_root(catalog: dict[str, Any] | None = None) -> str | None:
    catalog = catalog or load_train_catalog()
    for cand in catalog.get("phi0_root_local_candidates") or []:
        root = Path(cand)
        launcher = root / "tools" / "train" / "run_online_vlm_mix_distill.sh"
        if launcher.is_file():
            return str(root.resolve())
    # Fallback: use phi0_pipeline itself if it gains a launcher symlink later.
    pipeline = APP_DIR / "phi0_pipeline"
    if (pipeline / "tools" / "train" / "run_online_vlm_mix_distill.sh").is_file():
        return str(pipeline.resolve())
    return None


def read_dataset_prompt(dataset_path: str) -> str | None:
    """Read first task prompt from V3 tasks.parquet or V2.1 tasks.jsonl / episodes.jsonl."""
    root = Path(dataset_path)
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet, columns=["task"])
            for t in table.column("task").to_pylist():
                text = str(t or "").strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001
            pass
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.is_file():
        try:
            with tasks_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = str(row.get("task") or "").strip()
                    if text:
                        return text
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.is_file():
        try:
            with episodes_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tasks = row.get("tasks") or []
                    if isinstance(tasks, list):
                        for t in tasks:
                            text = str(t or "").strip()
                            if text:
                                return text
                    else:
                        text = str(tasks or "").strip()
                        if text:
                            return text
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return None


def is_placeholder_prompt(prompt: str | None) -> bool:
    """Collector often writes literal 'demo'; treat as empty for training."""
    text = str(prompt or "").strip()
    if not text:
        return True
    return text.lower() in {"demo", "task", "none", "null", "n/a", "-"}


def resolve_train_prompt(*candidates: str | None) -> str:
    for c in candidates:
        text = str(c or "").strip()
        if text and not is_placeholder_prompt(text):
            return text
    return ""


def probe_train_dataset(path: str) -> dict[str, Any]:
    root = Path(path)
    checks = {}
    for rel in READY_FILES:
        checks[rel] = (root / rel).is_file()
    checks[SCREENED_QA_REL] = (root / SCREENED_QA_REL).is_file()
    prompt = read_dataset_prompt(path) if root.is_dir() else None
    info: dict[str, Any] = {}
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            info = {}
    screened = read_screened_valid(str(root))
    n_eps = screened.get("valid_count") or None
    if n_eps is None:
        allow_path = root / VISION_ALLOWLIST_REL
        if allow_path.is_file():
            try:
                allow = json.loads(allow_path.read_text("utf-8"))
                eps = _episode_list_from_payload(allow)
                n_eps = len(eps)
            except (OSError, ValueError, TypeError):
                pass
    total_all = int(info.get("total_episodes") or 0)
    checks[ALL_ALLOWLIST_REL] = (root / ALL_ALLOWLIST_REL).is_file()
    ready = root.is_dir() and all(checks[rel] for rel in READY_FILES)
    mix_ready = ready and bool(checks.get(ALL_ALLOWLIST_REL))
    train_ready = ready and bool(screened.get("screened") and screened.get("valid"))
    return {
        "path": str(root),
        "name": root.name,
        "exists": root.is_dir(),
        "ready": ready,
        "mix_ready": mix_ready,
        "is_mix": is_mix_dataset_path(str(root)),
        "train_ready": train_ready,
        "checks": checks,
        "prompt": prompt,
        "total_episodes": int(n_eps or total_all or 0),
        "total_episodes_all": total_all,
        "valid_count": int(screened.get("valid_count") or 0),
        "invalid_count": int(screened.get("invalid_count") or 0),
        "screened": bool(screened.get("screened")),
        "allowlist_source": screened.get("source"),
        "valid": list(screened.get("valid") or []),
        "invalid": list(screened.get("invalid") or []),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "has_videos": (root / "videos").is_dir(),
        "total_frames": int(info.get("total_frames") or 0),
    }


def estimate_steps_per_epoch(dataset_path: str, num_envs: int = 32) -> int:
    """Align with launcher: spe ≈ ceil(n_frames / NUM_ENVS). Fallback 10000."""
    num_envs = max(1, int(num_envs or 32))
    root = Path(dataset_path)
    n_frames = 0
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text("utf-8"))
            n_frames = int(info.get("total_frames") or 0)
        except (OSError, ValueError, TypeError):
            n_frames = 0
    if n_frames <= 0:
        # crude fallback: episodes × 300 frames
        probe = probe_train_dataset(str(root)) if root.is_dir() else {}
        n_eps = int(probe.get("total_episodes") or 0)
        n_frames = n_eps * 300 if n_eps else 0
    if n_frames <= 0:
        return 10000
    return max(1, (n_frames + num_envs - 1) // num_envs)


def train_ckpt_data_meta(job: dict[str, Any]) -> dict[str, Any]:
    """Data paths + episode/session counts recorded into ckpt info.json."""
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    pack = job.get("pack") if isinstance(job.get("pack"), dict) else {}
    paths: list[str] = []
    for raw in params.get("dataset_paths") or []:
        s = str(raw or "").strip()
        if s:
            paths.append(s)
    sources = pack.get("stage_sources") if isinstance(pack.get("stage_sources"), dict) else {}
    if sources:
        paths = [str(v) for v in sources.values() if str(v).strip()] or paths
    episode_count = int(params.get("valid_count") or pack.get("valid_count") or 0)
    session_count = int(pack.get("session_count") or 0)
    ref_root = str(job.get("ref_root") or pack.get("ref_root") or "").strip()
    if episode_count <= 0 and ref_root and Path(ref_root).is_dir():
        try:
            episode_count = int(read_screened_valid(ref_root).get("valid_count") or 0)
        except Exception:  # noqa: BLE001
            episode_count = 0
    if session_count <= 0:
        session_count = len(paths)
    out: dict[str, Any] = {
        "paths": paths,
        "ref_root": ref_root,
        "episode_count": episode_count,
        "session_count": session_count,
    }
    if pack.get("manifest_path"):
        out["manifest"] = str(pack["manifest_path"])
    return out


def enrich_catalog_for_ui(catalog: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for task in catalog.get("tasks") or []:
        datasets = []
        for ds in task.get("datasets") or []:
            local = ds.get("path") or ""
            if local.startswith("/") and Path(local).exists():
                probe = probe_train_dataset(local)
            else:
                probe = {
                    "path": local,
                    "name": Path(local).name if local else "",
                    "exists": False,
                    "ready": False,
                    "checks": {},
                    "prompt": "",
                    "total_episodes": 0,
                }
            # Prefer skill card prompt over collector placeholder "demo".
            probe["prompt"] = resolve_train_prompt(
                ds.get("prompt"),
                probe.get("prompt"),
                task.get("default_prompt"),
            )
            datasets.append({**ds, "prompt": probe["prompt"] or ds.get("prompt") or "", "probe": probe})
        t = {**task, "datasets": datasets}
        # Drop legacy resume cards — Studio trains only (script-aligned, no STUDENT_CKPT).
        if t.get("requires_ckpt") or t.get("id") == "resume_extra":
            continue
        tasks.append(t)
    out = dict(catalog)
    out["tasks"] = tasks
    out["local_phi0_root"] = resolve_local_phi0_root(catalog)
    return out


# Train / pack / VLM-cache must use Newton torch+flash_attn (not legacy Phi-0-wpy).
NEWTON_PHI0_PY_CANDIDATES = (
    "/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python",
    "/mnt/data2/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python",
)


def resolve_train_phi0_py(explicit: str | None = None) -> str:
    """Prefer Newton python; never silently fall back to stale Phi-0-wpy."""
    candidates = [str(explicit or "").strip()] if explicit else []
    candidates.extend(NEWTON_PHI0_PY_CANDIDATES)
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    # Remote hosts may not share Studio's FS; still emit the canonical path so SSH
    # jobs get the right interpreter after setup_env (which keeps an executable PHI0_PY).
    return NEWTON_PHI0_PY_CANDIDATES[0]


def build_train_env(job: dict[str, Any], catalog: dict[str, Any]) -> dict[str, str]:
    params = job.get("params") or {}
    ngpu = int(params.get("ngpu") or 8)
    num_envs = int(params.get("num_envs") or 32)
    horizon = int(params.get("horizon") or 32)
    epochs = int(params.get("epochs") or 4)
    extra_steps = int(params.get("extra_steps") or params.get("train_steps") or 0)
    stamp = job.get("stamp") or time.strftime("%Y%m%d_%H%M%S")
    task = next((t for t in (catalog.get("tasks") or []) if t.get("id") == job.get("task_id")), None) or {}
    profile = str(
        job.get("train_profile")
        or params.get("train_profile")
        or task.get("train_profile")
        or ""
    ).strip()
    if not profile:
        profile = "vision_teleop"
    env: dict[str, str] = {str(k): str(v) for k, v in (catalog.get("fixed_flags") or {}).items()}
    env.update(profile_fixed_flags(profile))
    env.update({str(k): str(v) for k, v in (task.get("fixed_flags") or {}).items()})
    env.update({str(k): str(v) for k, v in (job.get("fixed_flags") or {}).items()})
    hand_obs = str(params.get("hand_obs") or job.get("hand_obs") or "").strip()
    if hand_obs:
        env["PHI0_TRAIN_HAND_OBS"] = hand_obs
    if params.get("hand_proprio_lag") is not None:
        env["PHI0_HAND_PROPRIO_LAG"] = str(int(params["hand_proprio_lag"]))
    if params.get("distill_rtc") is not None:
        env["PHI0_DISTILL_RTC"] = "1" if params.get("distill_rtc") else "0"
    if params.get("p_vision") is not None:
        env["PHI0_P_VISION"] = str(params["p_vision"])
    env.update({
        "REF_ROOT": str(job["ref_root"]),
        "NGPU": str(ngpu),
        "NUM_ENVS": str(num_envs),
        "HORIZON": str(horizon),
        "EPOCHS": str(max(epochs, 1)),
        "LR": str(params.get("lr") or "1e-4"),
        "PHI0_LR_SCHEDULER": str(params.get("lr_scheduler") or "cosine"),
        "PHI0_LR_WARMUP_RATIO": str(params.get("warmup_ratio") if params.get("warmup_ratio") is not None else 0.05),
        "PHI0_LR_WARMUP_STEPS": str(int(params.get("warmup_steps") or 0)),
        "STAMP": stamp,
        "PHI0_DISTILL_OUT": str(job["out_dir"]),
        "LOG_FILE": str(job["log_file"]),
        "CUDA_VISIBLE_DEVICES": str(params.get("cuda_devices") or ",".join(str(i) for i in range(ngpu))),
        "ACTION_STATS_PATH": f"{job['ref_root'].rstrip('/')}/meta/stats.json",
    })
    ckpt_every_epoch = int(
        params["ckpt_every_epoch"]
        if params.get("ckpt_every_epoch") is not None
        else 1
    )
    if ckpt_every_epoch > 0:
        env["PHI0_CKPT_EVERY_EPOCH"] = str(ckpt_every_epoch)
        env["CKPT_EVERY"] = "0"
    else:
        env["PHI0_CKPT_EVERY_EPOCH"] = "0"
        env["CKPT_EVERY"] = str(int(params.get("ckpt_every") or 2000))
    env["PHI0_CKPT_STEP_KEEP"] = str(
        params.get("ckpt_step_keep")
        if params.get("ckpt_step_keep") is not None
        else env.get("PHI0_CKPT_STEP_KEEP", "3")
    )
    env.update({
        "DEPLOY_POLICY_DIR": str(params.get("deploy_policy") or env.get("DEPLOY_POLICY_DIR") or "sonic_v1_1"),
        "PHI0_HAND_MODE": str(params.get("hand_mode") or env.get("PHI0_HAND_MODE") or "dex3"),
    })
    allow = str(job.get("episode_allowlist") or params.get("episode_allowlist") or "").strip()
    if allow:
        env["EPISODE_ALLOWLIST"] = allow
    pack = job.get("pack") if isinstance(job.get("pack"), dict) else None
    if pack:
        if pack.get("mix_slots_mode") and pack.get("slots"):
            env["MIX_SLOTS_JSON"] = json.dumps(pack.get("slots") or [], ensure_ascii=False)
            env["MERGED_OUT"] = str(pack.get("merged_out") or pack.get("nvme_dir") or "")
            env["WS_LINK"] = str(pack.get("ws_link") or job["ref_root"])
            env["SKIP_PACK"] = "1" if pack.get("skip_pack") else "0"
            env["REF_ROOT"] = env["WS_LINK"]
            env["ACTION_STATS_PATH"] = f"{env['REF_ROOT'].rstrip('/')}/meta/stats.json"
            env["TASK_PROMPT"] = str(pack.get("task_prompt") or params.get("prompt") or "task")
        else:
            env.update({
                "RAW_ROOT": str(pack.get("raw_root") or ""),
                "MANIFEST": str(pack.get("manifest_path") or ""),
                "OUT_DIR": str(pack.get("out_dir") or ""),
                "NVME_DIR": str(pack.get("nvme_dir") or ""),
                "WS_LINK": str(pack.get("ws_link") or job["ref_root"]),
                "TASK_PROMPT": str(pack.get("task_prompt") or params.get("prompt") or "task"),
                "SKIP_PACK": "1" if pack.get("skip_pack") else "0",
            })
            # Distill reads the workspace link created by pack.
            env["REF_ROOT"] = str(pack.get("ws_link") or job["ref_root"])
            env["ACTION_STATS_PATH"] = f"{env['REF_ROOT'].rstrip('/')}/meta/stats.json"
    if is_mix_train_profile(profile):
        env["MIX_DISTILL_OUT"] = str(job["out_dir"])
        env["MIX_LOG_FILE"] = str(job["log_file"])
    prompt = str(params.get("prompt") or "").strip()
    if prompt:
        env["PROMPT"] = prompt
        env["TASK_PROMPT"] = prompt
    if job.get("student_ckpt"):
        env["STUDENT_CKPT"] = str(job["student_ckpt"])
    if params.get("init_from_base"):
        env["PHI0_RESUME_STEP0"] = "1"
    if extra_steps > 0:
        env["EXTRA_STEPS"] = str(extra_steps)
    env["EFFECTIVE_BATCH"] = str(ngpu * num_envs)
    env.setdefault("PHI0_METRICS_FLUSH_EVERY", "50")
    skill_name = str(job.get("skill_id") or job.get("task_id") or "").strip()
    if skill_name:
        env["PHI0_CKPT_SKILL"] = skill_name
    data_meta = train_ckpt_data_meta(job)
    if data_meta:
        env["PHI0_TRAIN_DATA_META"] = json.dumps(data_meta, ensure_ascii=False)
        if data_meta.get("episode_count") is not None:
            env["PHI0_TRAIN_EPISODE_COUNT"] = str(int(data_meta.get("episode_count") or 0))
        if data_meta.get("session_count") is not None:
            env["PHI0_TRAIN_SESSION_COUNT"] = str(int(data_meta.get("session_count") or 0))
        paths = data_meta.get("paths") or []
        if isinstance(paths, list) and paths:
            env["PHI0_TRAIN_DATA_PATHS"] = "\n".join(str(p) for p in paths if str(p).strip())
    # Critical: setup_env.sh otherwise latches onto /mnt/data/.../Phi-0-wpy (torch 2.5),
    # which breaks flash_attn against Newton-built wheels (seen on h20 pack lang_latents).
    env["PHI0_PY"] = resolve_train_phi0_py(
        str(params.get("phi0_py") or job.get("phi0_py") or env.get("PHI0_PY") or "")
    )
    return env


# Progress bar lines from newton_boneseed_distill / isaac_loop, e.g.
# [distill]  1001/15290   6.5% │…│   3.6 step/s (w50=3.5)  ETA 1.1h  loss=0.1907  z=0.143  hand=0.047
_DISTILL_STEP_RE = re.compile(r"\[distill\]\s*(\d+)\s*/\s*(\d+)\b")
_DISTILL_SPEED_RE = re.compile(r"([\d.]+)\s*step/s")
_DISTILL_LOSS_RE = re.compile(r"\bloss=([-\d.eE+]+)")
_DISTILL_Z_RE = re.compile(r"\bz=([-\d.eE+]+)")
_DISTILL_Z_PHYS_RE = re.compile(r"\bz_phys=([-\d.eE+]+)")
_DISTILL_HAND_RE = re.compile(r"\bhand=([-\d.eE+]+)")
_DISTILL_Q_HEAD_RE = re.compile(r"\bq_head=([-\d.eE+]+)")
_DISTILL_SMPL_RE = re.compile(r"\bsmpl=([-\d.eE+]+)")
_DISTILL_ETA_RE = re.compile(
    r"ETA\s+(\d+(?:\.\d+)?)\s*h(?:\s*(\d+)\s*m)?|ETA\s+(\d+)\s*m(?:\s*(\d+)\s*s)?|ETA\s+(\d+)\s*s",
    re.IGNORECASE,
)


def _parse_eta_seconds(text: str) -> int | None:
    m = _DISTILL_ETA_RE.search(text or "")
    if not m:
        return None
    if m.group(1) is not None:
        hours = float(m.group(1))
        mins = float(m.group(2) or 0)
        return int(hours * 3600 + mins * 60)
    if m.group(3) is not None:
        mins = float(m.group(3))
        secs = float(m.group(4) or 0)
        return int(mins * 60 + secs)
    if m.group(5) is not None:
        return int(float(m.group(5)))
    return None


def parse_distill_log_metrics(
    text: str, limit: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``[distill] step/total … loss=…`` bars when distill_metrics.jsonl is not flushed yet.

    ``limit``: if set, keep an even span across the full history (not only the tail).
    """
    by_step: dict[int, dict[str, Any]] = {}
    bar_total: int | None = None
    for ln in (text or "").splitlines():
        if "[distill]" not in ln or "loss=" not in ln or "/" not in ln:
            continue
        sm = _DISTILL_STEP_RE.search(ln)
        lm = _DISTILL_LOSS_RE.search(ln)
        if not sm or not lm:
            continue
        step = int(sm.group(1))
        total = int(sm.group(2))
        if total > 0:
            bar_total = total
        speed_m = _DISTILL_SPEED_RE.search(ln)
        z_m = _DISTILL_Z_RE.search(ln)
        z_phys_m = _DISTILL_Z_PHYS_RE.search(ln)
        hand_m = _DISTILL_HAND_RE.search(ln)
        q_m = _DISTILL_Q_HEAD_RE.search(ln)
        smpl_m = _DISTILL_SMPL_RE.search(ln)
        row: dict[str, Any] = {
            "step": step,
            "loss": _f(lm.group(1)),
            "loss_z": _f(z_m.group(1)) if z_m else None,
            "loss_z_phys": _f(z_phys_m.group(1)) if z_phys_m else None,
            "loss_hand": _f(hand_m.group(1)) if hand_m else None,
            "loss_q_head": _f(q_m.group(1)) if q_m else None,
            "loss_smpl": _f(smpl_m.group(1)) if smpl_m else None,
            "grad_l2": None,
            "dagger_beta": None,
            "n_valid": None,
            "batch": None,
            "steps_per_sec": _f(speed_m.group(1)) if speed_m else None,
            "eta_seconds": _parse_eta_seconds(ln),
            "bar_total": total,
        }
        by_step[step] = row
    rows = [by_step[k] for k in sorted(by_step)]
    if limit is not None and limit > 0:
        rows = downsample_metric_rows(rows, int(limit))
    meta: dict[str, Any] = {}
    if bar_total:
        meta["max_steps"] = bar_total
        meta["steps_per_epoch"] = None
    return rows, meta


def _metrics_row_from_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    step = obj.get("chunk", obj.get("step", obj.get("global_step")))
    if step is None:
        return None
    return {
        "step": int(step),
        "loss": _f(obj.get("loss")),
        "loss_z": _f(obj.get("loss_z")),
        "loss_z_phys": _f(obj.get("loss_z_phys")),
        "loss_hand": _f(obj.get("loss_hand")),
        "loss_q_head": _f(obj.get("loss_q_head")),
        "loss_smpl": _f(obj.get("loss_smpl")),
        "grad_l2": _f(obj.get("grad_l2")),
        "dagger_beta": _f(obj.get("dagger_beta")),
        "n_valid": obj.get("n_valid"),
        "batch": obj.get("batch"),
    }


def parse_metrics_jsonl(text: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Parse distill_metrics.jsonl text into rows.

    ``limit``: if set, keep an even span across the full history (not only the tail).
    Default ``None`` keeps every parsed line so charts cover training from step 0.
    """
    rows: list[dict[str, Any]] = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        row = _metrics_row_from_obj(obj)
        if row is not None:
            rows.append(row)
    if limit is not None and limit > 0:
        rows = downsample_metric_rows(rows, int(limit))
    return rows


def parse_metrics_jsonl_file(
    path: str | Path,
    limit: int | None = None,
    *,
    byte_offset: int = 0,
) -> list[dict[str, Any]]:
    """Stream-parse a metrics jsonl file (full history, optional even downsample).

    ``byte_offset``: skip this many leading bytes so a reused ``out_dir`` can
    hide previous runs' metrics (Studio stores the size at job start).
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    off = max(0, int(byte_offset or 0))
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        if off > 0:
            fh.seek(min(off, p.stat().st_size))
            # Drop partial first line after a mid-file seek.
            if off < p.stat().st_size:
                fh.readline()
        for ln in fh:
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            row = _metrics_row_from_obj(obj)
            if row is not None:
                rows.append(row)
    if limit is not None and limit > 0:
        rows = downsample_metric_rows(rows, int(limit))
    return rows


def metrics_file_size(path: str | Path | None) -> int:
    """Bytes in ``distill_metrics.jsonl`` (0 if missing)."""
    if not path:
        return 0
    p = Path(path)
    try:
        return int(p.stat().st_size) if p.is_file() else 0
    except OSError:
        return 0


def parse_metrics_jsonl_bytes(
    text: str,
    limit: int | None = None,
    *,
    byte_offset: int = 0,
) -> list[dict[str, Any]]:
    """Parse metrics jsonl text, optionally skipping a leading byte offset (remote-safe)."""
    if not text:
        return []
    off = max(0, int(byte_offset or 0))
    if off > 0:
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= off:
            return []
        # Mid-line seek → drop partial line; exact line boundary → keep.
        at_line_start = raw[off - 1 : off] == b"\n"
        chunk = raw[off:]
        if not at_line_start:
            nl = chunk.find(b"\n")
            if nl < 0:
                return []
            chunk = chunk[nl + 1 :]
        text = chunk.decode("utf-8", errors="replace")
    return parse_metrics_jsonl(text, limit=limit)


def log_has_distill_progress(text: str) -> bool:
    """True when the train log already contains a distill loss bar."""
    if not text:
        return False
    return bool(_DISTILL_STEP_RE.search(text) and _DISTILL_LOSS_RE.search(text))


def downsample_metric_rows(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Evenly sample rows across the full series (always keeps first & last)."""
    n = len(rows)
    if max_points <= 0 or n <= max_points:
        return rows
    if max_points == 1:
        return [rows[-1]]
    idxs = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return [rows[i] for i in idxs]


def _f(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize_metrics(
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    *,
    created_at: str | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    params = params or {}
    train_steps = int(params.get("train_steps") or params.get("extra_steps") or 0)
    epochs_param = int(params.get("epochs") or 0) or None
    if not rows:
        return {
            "steps_done": int(summary.get("steps_done") or 0),
            "epochs": summary.get("epochs") or epochs_param,
            "steps_per_epoch": summary.get("steps_per_epoch"),
            "loss": summary.get("loss_last"),
            "loss_z": None,
            "loss_z_phys": None,
            "loss_hand": None,
            "loss_q_head": None,
            "loss_smpl": None,
            "grad_l2": None,
            "epoch_frac": None,
            "eta_steps": train_steps or None,
            "target_steps": train_steps or None,
            "eta_seconds": None,
            "steps_per_sec": None,
            "loss_smooth": None,
        }
    last = rows[-1]
    spe = int(summary.get("steps_per_epoch") or 0) or None
    epochs_cfg = summary.get("epochs") or epochs_param
    step = int(last["step"])
    epoch_frac = None
    if spe:
        epoch_frac = step / float(spe)
    eta_steps = None
    target = int(summary.get("max_steps") or 0) or None
    if spe and epochs_cfg:
        target = int(spe) * int(epochs_cfg)
        eta_steps = max(0, target - step)
    elif train_steps:
        target = train_steps
        if step > train_steps and rows:
            base = int(rows[0]["step"])
            if base > 1:
                target = max(int(target), base + train_steps - 1)
        eta_steps = max(0, int(target) - step)
    grad = None
    for r in reversed(rows):
        if r.get("grad_l2") is not None:
            grad = r["grad_l2"]
            break
    # EMA smooth of last losses for UI
    loss_smooth = None
    alpha = 0.08
    for r in rows:
        v = r.get("loss")
        if v is None:
            continue
        loss_smooth = float(v) if loss_smooth is None else (alpha * float(v) + (1 - alpha) * loss_smooth)

    steps_per_sec = None
    eta_seconds = None
    # Prefer live speed/ETA stamped on the last row (e.g. from log progress bars).
    if last.get("steps_per_sec") is not None:
        try:
            steps_per_sec = float(last["steps_per_sec"])
        except (TypeError, ValueError):
            steps_per_sec = None
    if last.get("eta_seconds") is not None:
        try:
            eta_seconds = int(last["eta_seconds"])
        except (TypeError, ValueError):
            eta_seconds = None
    if created_at and step > 0 and steps_per_sec is None:
        try:
            # Accept ISO with or without timezone
            from datetime import datetime

            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            elapsed = max(1.0, float(now_ts or time.time()) - created.timestamp())
            # Prefer delta over recent window if enough points
            if len(rows) >= 2:
                d_step = max(1, int(rows[-1]["step"]) - int(rows[0]["step"]))
                steps_per_sec = d_step / elapsed
            else:
                steps_per_sec = step / elapsed
            if eta_seconds is None and eta_steps is not None and steps_per_sec and steps_per_sec > 0:
                eta_seconds = int(eta_steps / steps_per_sec)
        except Exception:  # noqa: BLE001
            pass
    elif eta_seconds is None and eta_steps is not None and steps_per_sec and steps_per_sec > 0:
        eta_seconds = int(eta_steps / steps_per_sec)

    # If summary/max_steps came from log bar total, prefer that as target.
    if target is None and summary.get("max_steps"):
        try:
            target = int(summary["max_steps"])
            eta_steps = max(0, target - step)
            if eta_seconds is None and steps_per_sec and steps_per_sec > 0:
                eta_seconds = int(eta_steps / steps_per_sec)
        except (TypeError, ValueError):
            pass

    return {
        "steps_done": step,
        "epochs": epochs_cfg,
        "steps_per_epoch": spe,
        "loss": last.get("loss"),
        "loss_z": last.get("loss_z"),
        "loss_z_phys": last.get("loss_z_phys"),
        "loss_hand": last.get("loss_hand"),
        "loss_q_head": last.get("loss_q_head"),
        "loss_smpl": last.get("loss_smpl"),
        "grad_l2": grad,
        "dagger_beta": last.get("dagger_beta"),
        "epoch_frac": epoch_frac,
        "eta_steps": eta_steps,
        "target_steps": target,
        "loss_first": summary.get("loss_first"),
        "loss_last": summary.get("loss_last") or last.get("loss"),
        "loss_smooth": loss_smooth,
        "steps_per_sec": steps_per_sec,
        "eta_seconds": eta_seconds,
    }


_NAMED_CKPT_RE = re.compile(r"^.+_\d{8}_\d{6}(?:_s\d+)?\.pt$")


def _load_ckpt_info_blob(root: Path) -> dict[str, Any]:
    path = root / "info.json"
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return blob if isinstance(blob, dict) else {}


def list_student_ckpts(out_dir: str) -> list[dict[str, Any]]:
    """List last.pt, named skill+time snaps, and legacy step*.pt under out_dir."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    info_blob = _load_ckpt_info_blob(root)
    info_map: dict[str, dict[str, Any]] = {}
    for row in info_blob.get("checkpoints") or []:
        if isinstance(row, dict) and row.get("name"):
            info_map[str(row["name"])] = row
    data_info = info_blob.get("data") if isinstance(info_blob.get("data"), dict) else {}
    found: list[Path] = []
    last = root / "phi0_student_last.pt"
    if last.is_file():
        found.append(last)
    snaps: list[Path] = []
    for p in root.glob("*.pt"):
        if p.name.endswith("_optim.pt") or p == last:
            continue
        if p.name.startswith("phi0_student_step") or _NAMED_CKPT_RE.match(p.name):
            snaps.append(p)
    snaps.sort(key=lambda p: (p.stat().st_mtime, p.name))
    found.extend(snaps)
    out = []
    seen: set[str] = set()
    for p in found:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        try:
            st = p.stat()
        except OSError:
            continue
        item = {
            "path": str(p),
            "name": p.name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        }
        extra = info_map.get(p.name) or {}
        for k in ("loss", "loss_z", "loss_z_phys", "loss_hand", "steps", "epoch", "saved_at"):
            if extra.get(k) is not None:
                item[k] = extra[k]
        if data_info.get("episode_count") is not None:
            item["episode_count"] = data_info["episode_count"]
        out.append(item)
    return out


def list_pt_files(dir_path: str, *, recursive: bool = False) -> list[dict[str, Any]]:
    """List *.pt under a local directory (optionally one-level recursive)."""
    root = Path(dir_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"目录不存在: {root}")
    paths: list[Path] = []
    if recursive:
        paths = sorted(root.rglob("*.pt"))
    else:
        paths = sorted(root.glob("*.pt"))
        # also immediate subdirs (common: run_dir/*.pt)
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                paths.extend(sorted(sub.glob("*.pt")))
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({
            "path": str(p),
            "name": p.name,
            "rel": _rel_or_name(p, root),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        })
    out.sort(key=lambda x: (-int(x["mtime"]), x["name"]))
    return out


def _rel_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def smooth_metric_series(
    rows: list[dict[str, Any]],
    key: str | list[str] = "loss",
    alpha: float = 0.08,
) -> list[dict[str, Any]]:
    """Attach EMA field ``{key}_smooth`` for charting (one or many keys)."""
    keys = [key] if isinstance(key, str) else list(key)
    emas: dict[str, float | None] = {k: None for k in keys}
    out = []
    for r in rows:
        row = dict(r)
        for k in keys:
            v = r.get(k)
            if v is not None:
                cur = emas[k]
                emas[k] = float(v) if cur is None else alpha * float(v) + (1 - alpha) * cur
                row[f"{k}_smooth"] = emas[k]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# tmux-hosted train (survives Studio / browser crashes)
# ---------------------------------------------------------------------------

def train_tmux_session_name(job_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "", str(job_id or ""))[:48] or "job"
    return f"st_{safe}"


def build_train_inner_bash(*, phi0_root: str, script: str, env_map: dict[str, str], out_dir: str, log_file: str) -> str:
    """Inner shell that actually runs the distill wrapper and tees the log."""
    exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_map.items())
    return (
        f"set -o pipefail; "
        f"mkdir -p -- {shlex.quote(str(out_dir))} "
        f"{shlex.quote(str(Path(log_file).parent))} && "
        f"cd {shlex.quote(phi0_root)} && {exports} "
        f"bash {shlex.quote(script)} 2>&1 | tee -a {shlex.quote(log_file)}; "
        f"rc=${{PIPESTATUS[0]:-$?}}; "
        f"echo \"[studio] train_exit=$rc\" | tee -a {shlex.quote(log_file)}; "
        f"exit $rc"
    )


def build_tmux_launch_bash(session: str, inner_bash: str) -> str:
    """Create/replace a detached tmux session that runs inner_bash."""
    sess = shlex.quote(session)
    return (
        f"command -v tmux >/dev/null || {{ echo '[studio] tmux not found' >&2; exit 127; }}; "
        f"tmux has-session -t {sess} 2>/dev/null && tmux kill-session -t {sess} || true; "
        f"tmux new-session -d -s {sess} -- bash -lc {shlex.quote(inner_bash)}; "
        f"sleep 0.3; "
        f"tmux has-session -t {sess} || {{ echo '[studio] tmux session failed to start' >&2; exit 1; }}; "
        f"echo \"[studio] tmux_session={session}\""
    )


def build_tmux_kill_bash(session: str, *, out_dir: str = "", job_id: str = "") -> str:
    """Force-stop a studio train: kill tmux pane tree + orphans matching out_dir.

    Soft C-c / kill-session alone often leaves torchrun/DDP orphans holding GPUs.
    Always follow with SIGKILL on the pane process group, then kill -9 any
    remaining PIDs whose cmdline contains this job's unique out_dir / job id
    (skipping this kill shell itself).
    """
    sess = shlex.quote(session)
    out = str(out_dir or "").strip()
    jid = str(job_id or "").strip()
    # pkill -f would also match this kill shell (pattern appears in argv); kill by
    # enumerated PIDs and skip $$ / $PPID instead.
    marker_kills = (
        "_studio_kill_pat() { "
        "  _pat=$1; [ -n \"$_pat\" ] || return 0; "
        "  for _pid in $(pgrep -f -- \"$_pat\" 2>/dev/null || true); do "
        "    [ \"$_pid\" = \"$$\" ] && continue; "
        "    [ \"$_pid\" = \"$PPID\" ] && continue; "
        "    kill -9 \"$_pid\" 2>/dev/null || true; "
        "  done; "
        "}; "
    )
    if out.startswith("/") and len(out) >= 24:
        marker_kills += f"_studio_kill_pat {shlex.quote(out)}; "
    if jid and len(jid) >= 8:
        marker_kills += f"_studio_kill_pat {shlex.quote(jid)}; "
    return (
        f"pane_pid=''; "
        f"if tmux has-session -t {sess} 2>/dev/null; then "
        f"  pane_pid=$(tmux list-panes -t {sess} -F '#{{pane_pid}}' 2>/dev/null | head -n1); "
        f"  tmux send-keys -t {sess} C-c 2>/dev/null || true; "
        f"  sleep 0.5; "
        f"  if [ -n \"$pane_pid\" ]; then "
        f"    kill -TERM -\"$pane_pid\" 2>/dev/null || true; "
        f"    pkill -TERM -P \"$pane_pid\" 2>/dev/null || true; "
        f"    sleep 1; "
        f"    kill -KILL -\"$pane_pid\" 2>/dev/null || true; "
        f"    pkill -KILL -P \"$pane_pid\" 2>/dev/null || true; "
        f"    kill -9 \"$pane_pid\" 2>/dev/null || true; "
        f"  fi; "
        f"  tmux kill-session -t {sess} 2>/dev/null || true; "
        f"fi; "
        f"{marker_kills}"
        f"sleep 0.3; "
        f"echo \"[studio] force_killed session={session} out_dir={out} pane_pid=${{pane_pid:-}}\"; "
        f"exit 0"
    )


def build_tmux_has_bash(session: str) -> str:
    return f"tmux has-session -t {shlex.quote(session)}"


def parse_train_exit_code(log_text: str) -> int | None:
    for ln in reversed((log_text or "").splitlines()):
        if "[studio] train_exit=" not in ln:
            continue
        try:
            return int(ln.split("train_exit=", 1)[1].strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


def parse_tmux_session_from_log(log_text: str) -> str:
    for ln in (log_text or "").splitlines():
        if "[studio] tmux_session=" in ln:
            return ln.split("tmux_session=", 1)[1].strip().split()[0]
    return ""


# ---------------------------------------------------------------------------
# Multi-host train (cluster_0 local-SSH + off-box h20 / cluster_2)
# ---------------------------------------------------------------------------

LOCAL_TRAIN_HOST_ID = "cluster_0"
OFFBOX_TRAIN_HOST_IDS = ("h20-0", "h20-1", "cluster_2")

DEFAULT_HOST_TRAIN_PATHS: dict[str, str] = {
    "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
    "model_zoo": DEFAULT_TRAIN_OUT_BASE,
    "train_data": DEFAULT_TRAIN_PACK_BASE,
}


def builtin_train_hosts() -> list[dict[str, Any]]:
    """Canonical train hosts seeded into studio state."""
    rows = [
        {
            "id": "cluster_0",
            "name": "cluster_0",
            "target": "cluster_0",
            "train_ready": True,
            "gpus": 8,
        },
        {
            "id": "h20-0",
            "name": "h20-0",
            "target": "h20-0",
            "train_ready": False,
            "gpus": 8,
        },
        {
            "id": "h20-1",
            "name": "h20-1",
            "target": "h20-1",
            "train_ready": False,
            "gpus": 8,
        },
        {
            "id": "cluster_2",
            "name": "cluster_2",
            "target": "cluster_2",
            "train_ready": False,
            "gpus": 8,
        },
    ]
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(normalize_train_host(row))
    return out


def normalize_train_host(host: dict[str, Any] | None) -> dict[str, Any]:
    """Fill train path defaults on a host record."""
    h = dict(host or {})
    hid = str(h.get("id") or "").strip() or "host"
    h.setdefault("id", hid)
    h.setdefault("name", hid)
    h.setdefault("target", hid)
    for key, val in DEFAULT_HOST_TRAIN_PATHS.items():
        if not str(h.get(key) or "").strip():
            h[key] = val
    if "train_ready" not in h:
        h["train_ready"] = hid == LOCAL_TRAIN_HOST_ID
    if "gpus" not in h:
        h["gpus"] = 8
    return h


def is_offbox_train_host(host_id: str | None) -> bool:
    return str(host_id or "").strip() in OFFBOX_TRAIN_HOST_IDS


def merge_builtin_train_hosts(existing: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Ensure builtin train hosts exist; preserve custom hosts and known fields."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in existing or []:
        if not isinstance(row, dict):
            continue
        hid = str(row.get("id") or "").strip()
        if not hid:
            continue
        by_id[hid] = normalize_train_host(row)
    for builtin in builtin_train_hosts():
        hid = builtin["id"]
        if hid in by_id:
            merged = dict(builtin)
            merged.update({k: v for k, v in by_id[hid].items() if v not in (None, "")})
            # Keep explicit train_ready from stored state when present.
            if "train_ready" in by_id[hid]:
                merged["train_ready"] = bool(by_id[hid]["train_ready"])
            by_id[hid] = normalize_train_host(merged)
        else:
            by_id[hid] = builtin
    # Stable order: builtins first, then extras.
    order = [h["id"] for h in builtin_train_hosts()]
    out = [by_id[i] for i in order if i in by_id]
    for hid, row in by_id.items():
        if hid not in order:
            out.append(row)
    return out


def require_train_ready_host(host: dict[str, Any]) -> None:
    h = normalize_train_host(host)
    if h.get("id") == LOCAL_TRAIN_HOST_ID:
        return
    if not h.get("train_ready"):
        reason = str(h.get("train_ready_reason") or "完整性校验未通过").strip()
        raise ValueError(f"训练主机 {h.get('name') or h.get('id')} 未就绪：{reason}")


def collect_train_stage_paths(job: dict[str, Any]) -> list[str]:
    """Local absolute paths that must exist on the remote host before launch."""
    paths: list[str] = []
    pack = job.get("pack") if isinstance(job.get("pack"), dict) else None
    if pack:
        for key in ("manifest_path",):
            p = str(pack.get(key) or "").strip()
            if p:
                paths.append(p)
        for slot in pack.get("slots") or []:
            if not isinstance(slot, dict):
                continue
            mp = str(slot.get("manifest_path") or "").strip()
            if mp:
                paths.append(mp)
            for sess in slot.get("dataset_paths") or []:
                if str(sess).startswith("/"):
                    paths.append(str(sess))
        for sess in pack.get("sessions") or []:
            # Prefer absolute sources map if present.
            sources = pack.get("stage_sources") if isinstance(pack.get("stage_sources"), dict) else {}
            if isinstance(sources, dict) and sess in sources:
                paths.append(str(sources[sess]))
            elif str(sess).startswith("/"):
                paths.append(str(sess))
        # Also include original session list from pack plan.
        for sess in pack.get("session_paths") or pack.get("raw_sessions") or []:
            if str(sess).startswith("/"):
                paths.append(str(sess))
    else:
        ref = str(job.get("ref_root") or "").strip()
        if ref:
            paths.append(ref)
    allow = str(job.get("episode_allowlist") or "").strip()
    if allow:
        paths.append(allow)
    # Dedup preserve order
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        p = str(p).rstrip("/")
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def remote_pack_symlink_bash(pack: dict[str, Any]) -> str:
    """Bash that recreates pack raw_stage symlinks on the train host."""
    stage = str(pack.get("raw_root") or "").strip()
    sources = pack.get("stage_sources") if isinstance(pack.get("stage_sources"), dict) else {}
    if not stage:
        return "true"
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(stage)}",
    ]
    if sources:
        for name, src in sources.items():
            dest = f"{stage.rstrip('/')}/{name}"
            lines.append(f"rm -rf {shlex.quote(dest)}")
            lines.append(
                f"ln -s {shlex.quote(str(src))} {shlex.quote(dest)}"
            )
    else:
        for sess in pack.get("session_paths") or []:
            src = Path(str(sess))
            dest = f"{stage.rstrip('/')}/{src.name}"
            lines.append(f"rm -rf {shlex.quote(dest)}")
            lines.append(f"ln -s {shlex.quote(str(src))} {shlex.quote(dest)}")
    for key in ("out_dir", "nvme_dir", "pack_root"):
        p = str(pack.get(key) or "").strip()
        if p:
            lines.append(f"mkdir -p {shlex.quote(p)}")
    ws = str(pack.get("ws_link") or "").strip()
    nvme = str(pack.get("nvme_dir") or "").strip()
    if ws and nvme:
        lines.append(f"mkdir -p {shlex.quote(str(Path(ws).parent))}")
        lines.append(f"rm -f {shlex.quote(ws)}")
        lines.append(f"ln -sfn {shlex.quote(nvme)} {shlex.quote(ws)}")
    return "\n".join(lines)


def remap_local_train_data_path(path: str) -> str:
    """Identity helper kept for older call sites / tests."""
    return str(path or "").strip()


def archive_dest_for_pack_root(pack_root: str) -> str:
    """``.../Phi_0_train_data/<skill_stamp>`` → EFS archive sibling."""
    root = Path(str(pack_root or "").rstrip("/"))
    return str(Path(ARCHIVE_TRAIN_PACK_BASE) / root.name)


def _dir_file_stats(root: Path) -> dict[str, int]:
    """Count regular files + total bytes under root (follow neither symlink trees)."""
    n_files = 0
    n_bytes = 0
    if not root.is_dir():
        return {"files": 0, "bytes": 0}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.is_symlink():
                    continue
                if not p.is_file():
                    continue
                n_files += 1
                n_bytes += int(p.stat().st_size)
            except OSError:
                continue
    return {"files": n_files, "bytes": n_bytes}


def archive_train_pack_to_efs(
    pack: dict[str, Any] | None,
    *,
    archive_base: str | Path | None = None,
    delete_local: bool = True,
) -> dict[str, Any]:
    """Copy finished pack to EFS, verify file count/size, then delete local data2 copy.

    Training still uses ``DEFAULT_TRAIN_PACK_BASE`` (data2). Only call this after a
    successful train (+ pullback when off-box). Never deletes outside the local pack base.
    """
    import shutil

    out: dict[str, Any] = {
        "ok": False,
        "archived": False,
        "deleted_local": False,
        "src": "",
        "dest": "",
        "errors": [],
        "verify": {},
    }
    if not isinstance(pack, dict):
        out["errors"].append("no_pack")
        return out
    src_raw = str(pack.get("pack_root") or "").strip()
    if not src_raw:
        out["errors"].append("no_pack_root")
        return out
    src = Path(src_raw)
    pack_prefix = str(Path(DEFAULT_TRAIN_PACK_BASE).resolve())
    try:
        src_resolved = str(src.resolve()) if src.exists() else str(src)
    except OSError as exc:
        out["errors"].append(f"resolve_src:{exc}")
        return out
    if not (src_resolved == pack_prefix or src_resolved.startswith(pack_prefix + os.sep)):
        out["errors"].append(f"refuse_archive_outside_pack_base:{src_resolved}")
        return out
    if not src.is_dir():
        out["errors"].append(f"src_missing:{src}")
        return out

    base = Path(str(archive_base or ARCHIVE_TRAIN_PACK_BASE))
    dest = base / src.name
    out["src"] = str(src)
    out["dest"] = str(dest)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        out["errors"].append(f"mkdir_archive:{exc}")
        return out

    # rsync preserves hardlinks where possible; fallback to copytree on missing rsync.
    rsync_bin = shutil.which("rsync")
    if rsync_bin:
        proc = subprocess.run(
            [
                rsync_bin, "-aH", "--delete",
                f"{src}/",
                f"{dest}/",
            ],
            capture_output=True,
            text=True,
            timeout=86_400,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"rsync rc={proc.returncode}").strip()[-800:]
            out["errors"].append(f"rsync:{err}")
            return out
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, symlinks=True)

    src_stats = _dir_file_stats(src)
    dest_stats = _dir_file_stats(dest)
    out["verify"] = {"src": src_stats, "dest": dest_stats}
    if src_stats["files"] <= 0:
        out["errors"].append("src_empty")
        return out
    if src_stats != dest_stats:
        out["errors"].append(
            f"verify_mismatch src={src_stats} dest={dest_stats}"
        )
        return out
    out["archived"] = True
    out["ok"] = True

    if delete_local:
        try:
            # Drop workspace link first so we don't leave a dangling pointer under pack_root.
            ws = src / "studio_tmp_unified_link"
            if ws.is_symlink() or ws.is_file():
                ws.unlink(missing_ok=True)
            shutil.rmtree(src)
            out["deleted_local"] = True
        except OSError as exc:
            out["ok"] = False
            out["errors"].append(f"delete_local:{exc}")
    return out


def pullback_paths_for_job(job: dict[str, Any]) -> list[str]:
    """Remote dirs to rsync back to the studio host (same absolute paths)."""
    paths: list[str] = []
    out_dir = str(job.get("out_dir") or "").strip()
    if out_dir:
        paths.append(out_dir)
    pack = job.get("pack") if isinstance(job.get("pack"), dict) else None
    if pack:
        root = str(pack.get("pack_root") or "").strip()
        if root:
            paths.append(root)
    return paths


def train_preflight_remote_bash() -> str:
    """Remote bash that prints PASS/FAIL lines for train readiness."""
    return r'''
set -e
ok=0; fail=0
pass(){ echo "PASS $1"; ok=$((ok+1)); }
failf(){ echo "FAIL $1"; fail=$((fail+1)); }
PY=""
for p in /mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python /mnt/data2/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python; do
  [[ -x "$p" ]] && PY=$p && break
done
if [[ -n "$PY" ]]; then
  # Bound import: busy GPUs / hung driver can stall torch._C forever.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
  if timeout 90 env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
      "$PY" -c "import torch,flash_attn; assert torch.cuda.device_count()>=8; print(torch.__version__)" \
      2>/tmp/studio_newton_err; then
    pass "newton_import_gpus8"
  else
    # Fallback: env files present + nvidia-smi sees 8 GPUs (import may stall under load).
    TPKG="$("$PY" - <<'PY'
import sys, pathlib
print(pathlib.Path(sys.prefix)/"lib"/("python%d.%d"%sys.version_info[:2])/"site-packages"/"torch")
PY
)"
    FA=$(ls "$("$PY" -c 'import sys,glob; import pathlib; p=pathlib.Path(sys.prefix)/"lib"/("python%d.%d"%sys.version_info[:2])/"site-packages"; print(p)')"/flash_attn_2_cuda*.so 2>/dev/null | head -1)
    ng=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ -d "$TPKG" && -n "$FA" && "${ng:-0}" -ge 8 ]]; then
      pass "newton_pkgs_gpus8"
      echo "WARN newton_import_timeout (rc/timeout); pkgs+8gpus ok"
    else
      failf "newton_import"
      echo "WARN $(tail -c 240 /tmp/studio_newton_err 2>/dev/null | tr '\n' ' ')"
    fi
  fi
else
  failf "newton_python_missing"
fi
if [[ -d /mnt/data3/wpy/IsaacLab-3.0/source/isaaclab || -d /mnt/data2/wpy/IsaacLab-3.0/source/isaaclab ]]; then
  pass "isaaclab"
else
  failf "isaaclab"
fi
ROOT=/mnt/data2/wpy/workspace/Phi_0_wpy
if [[ -f $ROOT/tools/train/run_online_vlm_mix_distill.sh && -f $ROOT/tools/train/run_studio_selection_pack_and_distill.sh && -f $ROOT/tools/data/run_830_skill2_pico_pack_and_cache.sh ]]; then
  pass "phi0_train_scripts"
else
  failf "phi0_train_scripts"
fi
POL=$ROOT/subpackages/gear_sonic_deploy/policy/sonic_v1_1
sz=$(du -sb "$POL" 2>/dev/null | awk '{print $1}')
if [[ ${sz:-0} -gt 1400000000 && -f $POL/last.pt && -f $POL/model_encoder.onnx ]]; then
  pass "sonic_v1_1"
else
  failf "sonic_v1_1"
fi
qok=0
for q in /mnt/data3/hf_home/hub/models--Qwen--Qwen3-VL-2B-Instruct /mnt/data2/hf_home/hub/models--Qwen--Qwen3-VL-2B-Instruct; do
  nb=$(ls "$q/blobs" 2>/dev/null | wc -l)
  [[ ${nb:-0} -ge 5 ]] && qok=1
done
[[ $qok -eq 1 ]] && pass "qwen3vl" || failf "qwen3vl"
ZOO=/mnt/data2/wpy/workspace/Phi_0_model_zoo
TD=$ZOO/Phi_0_train_data
mkdir -p "$ZOO" "$TD"
if [[ -d "$ZOO" && -w "$ZOO" && -d "$TD" && -w "$TD" ]]; then
  pass "model_zoo_writable"
else
  failf "model_zoo_writable"
fi
command -v tmux >/dev/null && pass "tmux" || failf "tmux"
echo "SUMMARY ok=$ok fail=$fail"
[[ $fail -eq 0 ]]
'''
