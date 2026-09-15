"""Local skill-card registry: one skill = one folder with meta JSON + labels stubs.

Surface layout (scripts / remote publish can be wired later)::

    {skills_root}/{skill_id}/
      {skill_id}.json      # skill card meta
      valid.json           # aggregated valid list (stub until review merge)
      invalid.json         # aggregated invalid list
      sessions/            # uploaded / linked dataset sessions
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SKILLS_ROOT = APP_DIR / "data" / "skills"
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def skills_root(cfg: dict[str, Any] | None = None) -> Path:
    raw = ""
    if cfg:
        raw = str(cfg.get("skills_root") or "").strip()
    root = Path(raw) if raw else DEFAULT_SKILLS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("._-")
    return (s or f"skill_{int(time.time()) % 100000:05d}")[:96]


def skill_dir(root: Path, skill_id: str) -> Path:
    return root / skill_id


def meta_path(root: Path, skill_id: str) -> Path:
    return skill_dir(root, skill_id) / f"{skill_id}.json"


def ensure_skill_files(folder: Path, skill_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sessions").mkdir(exist_ok=True)
    valid_p = folder / "valid.json"
    invalid_p = folder / "invalid.json"
    if not valid_p.exists():
        valid_p.write_text(json.dumps({"valid": [], "updated_at": meta.get("updated_at")}, ensure_ascii=False, indent=2), "utf-8")
    if not invalid_p.exists():
        invalid_p.write_text(json.dumps({"invalid": [], "updated_at": meta.get("updated_at")}, ensure_ascii=False, indent=2), "utf-8")
    mp = folder / f"{skill_id}.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return meta


def list_session_dirs(folder: Path) -> list[dict[str, Any]]:
    sessions_root = folder / "sessions"
    if not sessions_root.is_dir():
        return []
    out = []
    for p in sorted(sessions_root.iterdir()):
        if p.is_dir():
            out.append({"id": p.name, "path": str(p), "label": p.name})
    return out


def _looks_like_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "meta" / "info.json").is_file() or (path / "data").is_dir()


def _read_session_marker(session_dir: Path) -> dict[str, Any]:
    marker = session_dir / "session.json"
    if not marker.is_file():
        return {}
    try:
        data = json.loads(marker.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def scan_skill_datasets(skill_or_id: dict[str, Any] | str, *, root: Path | None = None) -> list[dict[str, Any]]:
    """Discover filterable / trainable dataset roots under a skill card."""
    root = root or skills_root()
    if isinstance(skill_or_id, str):
        skill = load_skill(root, skill_or_id) or {"id": skill_or_id, "folder": str(skill_dir(root, skill_or_id))}
    else:
        skill = skill_or_id
    folder = Path(str(skill.get("folder") or skill_dir(root, str(skill.get("id") or ""))))
    found: dict[str, dict[str, Any]] = {}

    def add(entry: dict[str, Any]) -> None:
        path = str(entry.get("path") or "").strip()
        if not path:
            return
        key = path.rstrip("/")
        prev = found.get(key) or {}
        merged = {**prev, **{k: v for k, v in entry.items() if v not in (None, "")}}
        merged.setdefault("id", Path(key).name)
        merged.setdefault("label", merged["id"])
        found[key] = merged

    for d in skill.get("datasets") or []:
        if isinstance(d, dict):
            add(d)

    # sessions/<name> may be a dataset root or a marker pointing at local_dir
    sessions_root = folder / "sessions"
    if sessions_root.is_dir():
        for p in sorted(sessions_root.iterdir()):
            if not p.is_dir():
                continue
            marker = _read_session_marker(p)
            local_dir = str(marker.get("local_dir") or "").strip()
            remote_dir = str(marker.get("remote_dir") or "").strip()
            if local_dir and _looks_like_dataset(Path(local_dir)):
                add({"id": p.name, "label": p.name, "path": local_dir, "remote_path": remote_dir, "source": "session_local"})
            elif _looks_like_dataset(p):
                add({"id": p.name, "label": p.name, "path": str(p), "remote_path": remote_dir, "source": "session_dir"})
            else:
                add({"id": p.name, "label": f"{p.name}（待转换/无 meta）", "path": local_dir or str(p), "remote_path": remote_dir, "source": "session_stub", "ready": False})

    # optional datasets/ subfolder
    datasets_root = folder / "datasets"
    if datasets_root.is_dir():
        for p in sorted(datasets_root.iterdir()):
            if p.is_dir():
                add({"id": p.name, "label": p.name, "path": str(p), "source": "datasets_dir", "ready": _looks_like_dataset(p)})

    # top-level linked paths stored on skill meta (only if they look like a dataset root)
    for key in ("ref_root", "local_ref_root", "dataset_root", "unified_root"):
        path = str(skill.get(key) or "").strip()
        if path and _looks_like_dataset(Path(path)):
            add({"id": Path(path).name, "label": Path(path).name, "path": path, "source": key})
        # Parent 830demo skill dirs hold sessions + skill_N.json — not a dataset root.

    out = list(found.values())
    for d in out:
        p = Path(str(d.get("path") or ""))
        if "ready" not in d:
            d["ready"] = _looks_like_dataset(p)
        if d.get("ready") and p.is_dir():
            info = p / "meta" / "info.json"
            if info.is_file():
                try:
                    meta = json.loads(info.read_text("utf-8"))
                    d["total_episodes"] = meta.get("total_episodes")
                    d["total_frames"] = meta.get("total_frames")
                    d["robot_type"] = meta.get("robot_type")
                except (OSError, json.JSONDecodeError):
                    pass
    out.sort(
        key=lambda x: (
            not x.get("ready"),
            "unified" in str(x.get("path") or x.get("remote_path") or x.get("label") or "").lower(),
            str(x.get("label") or x.get("id") or ""),
        )
    )
    return out


def load_skill(root: Path, skill_id: str) -> dict[str, Any] | None:
    mp = meta_path(root, skill_id)
    if not mp.is_file():
        return None
    try:
        meta = json.loads(mp.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    folder = skill_dir(root, skill_id)
    meta.setdefault("id", skill_id)
    meta.setdefault("folder", str(folder))
    meta["sessions"] = list_session_dirs(folder)
    meta["valid_json"] = str(folder / "valid.json")
    meta["invalid_json"] = str(folder / "invalid.json")
    meta["meta_json"] = str(mp)
    # Live scan so 04/05/06 share the same dataset inventory.
    scanned = scan_skill_datasets({**meta, "id": skill_id, "folder": str(folder)}, root=root)
    if scanned:
        meta["datasets"] = scanned
    elif not isinstance(meta.get("datasets"), list):
        meta["datasets"] = []
    return meta


def list_skills(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or skills_root()
    items: list[dict[str, Any]] = []
    if not root.is_dir():
        return items
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill = load_skill(root, child.name)
        if skill:
            items.append(skill)
    items.sort(key=lambda s: str(s.get("updated_at") or s.get("created_at") or ""), reverse=True)
    return items


def default_collect_root(skill_id: str, root: Path | None = None) -> str:
    """Stable per-skill local capture root (outside the skill card meta folder)."""
    root = root or skills_root()
    return str((root.parent / "collections" / skill_id).resolve())


def create_skill(
    data: dict[str, Any],
    *,
    root: Path | None = None,
    host_id: str = "cluster_0",
    remote_base: str = "",
) -> dict[str, Any]:
    root = root or skills_root()
    title = str(data.get("title") or data.get("name") or "").strip()
    if not title:
        raise ValueError("请填写技能名称")
    skill_id = str(data.get("id") or data.get("skill") or data.get("name") or "").strip() or _slugify(title)
    if not SAFE.fullmatch(skill_id):
        raise ValueError("技能 ID 仅支持字母、数字、点、下划线和短横线")
    if meta_path(root, skill_id).is_file():
        raise ValueError(f"技能已存在: {skill_id}")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    badge = str(data.get("badge") or skill_id).strip().upper()[:12]
    prompt = str(data.get("prompt") or data.get("description") or "").strip()
    folder = skill_dir(root, skill_id)
    remote_dir = str(data.get("remote_dir") or "").strip()
    if not remote_dir and remote_base:
        base = remote_base.rstrip("/")
        remote_dir = base if base.endswith("/" + skill_id) else f"{base}/{skill_id}"
    collect_root = str(data.get("collect_root") or "").strip() or default_collect_root(skill_id, root)
    Path(collect_root).mkdir(parents=True, exist_ok=True)
    meta = {
        "id": skill_id,
        "title": title,
        "badge": badge,
        "prompt": prompt,
        "description": str(data.get("description") or prompt).strip(),
        "host_id": host_id or str(data.get("host_id") or "cluster_0"),
        "remote_dir": remote_dir,
        "collect_root": collect_root,
        "current_session": "",
        "current_local_dir": "",
        "current_collection_id": "",
        "folder": str(folder),
        "created_at": now,
        "updated_at": now,
        "datasets": [],
        "source": str(data.get("source") or "studio"),
    }
    ensure_skill_files(folder, skill_id, meta)
    return load_skill(root, skill_id) or meta


def update_skill_fields(skill_id: str, fields: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    root = root or skills_root()
    skill = load_skill(root, skill_id)
    if not skill:
        raise ValueError(f"技能不存在: {skill_id}")
    allowed = {
        "title", "badge", "prompt", "description", "host_id", "remote_dir",
        "collect_root", "current_session", "current_local_dir", "current_collection_id",
        "student_ckpt", "ref_root", "local_ref_root", "hand_obs",
        "defaults", "train_out_dir", "ckpts", "recipe_id", "preferred_dataset_id", "ep",
        "labels_path", "labels_manifest", "manifest_name", "task_name",
        "train_profile",
    }
    for key, value in fields.items():
        if key not in allowed:
            continue
        if key == "collect_root":
            path = str(value or "").strip()
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
                skill[key] = path
            continue
        skill[key] = value
    if not str(skill.get("collect_root") or "").strip():
        skill["collect_root"] = default_collect_root(skill_id, root)
        Path(skill["collect_root"]).mkdir(parents=True, exist_ok=True)
    return _persist_skill(root, skill)


def prepare_session_dir(skill: dict[str, Any], local_dir: str = "", *, root: Path | None = None) -> dict[str, str]:
    """Resolve/create a timestamped session under the skill collect_root.

    Returns paths + env hints for a future external capture launcher:
      STUDIO_SKILL_ID / STUDIO_SESSION_DIR / STUDIO_COLLECT_ROOT / STUDIO_SESSION_NAME
    """
    root = root or skills_root()
    skill_id = str(skill.get("id") or "").strip()
    collect_root = str(skill.get("collect_root") or "").strip() or default_collect_root(skill_id, root)
    Path(collect_root).mkdir(parents=True, exist_ok=True)
    local_dir = str(local_dir or "").strip()
    if local_dir:
        session_path = Path(local_dir)
        session_path.mkdir(parents=True, exist_ok=True)
    else:
        stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        session_path = Path(collect_root) / stamp
        session_path.mkdir(parents=True, exist_ok=True)
    session_name = session_path.name
    return {
        "skill_id": skill_id,
        "collect_root": collect_root,
        "session_name": session_name,
        "local_dir": str(session_path.resolve()),
        "STUDIO_SKILL_ID": skill_id,
        "STUDIO_COLLECT_ROOT": collect_root,
        "STUDIO_SESSION_NAME": session_name,
        "STUDIO_SESSION_DIR": str(session_path.resolve()),
    }


def _persist_skill(root: Path, skill: dict[str, Any]) -> dict[str, Any]:
    skill_id = str(skill.get("id") or "").strip()
    if not skill_id:
        raise ValueError("技能缺少 id")
    folder = skill_dir(root, skill_id)
    skill["folder"] = str(folder)
    skill["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    clean = {k: v for k, v in skill.items() if k not in {"sessions", "valid_json", "invalid_json", "meta_json"}}
    ensure_skill_files(folder, skill_id, clean)
    return load_skill(root, skill_id) or skill


def attach_session(
    skill_id: str,
    session_name: str,
    *,
    local_dir: str = "",
    remote_dir: str = "",
    collection_id: str = "",
    status: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or skills_root()
    skill = load_skill(root, skill_id)
    if not skill:
        raise ValueError(f"技能不存在: {skill_id}")
    folder = skill_dir(root, skill_id)
    sessions = folder / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    link = sessions / session_name
    link.mkdir(parents=True, exist_ok=True)
    marker = {
        "name": session_name,
        "local_dir": local_dir,
        "remote_dir": remote_dir,
        "collection_id": collection_id,
        "status": status or "collecting",
        "attached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {
            "STUDIO_SKILL_ID": skill_id,
            "STUDIO_SESSION_NAME": session_name,
            "STUDIO_SESSION_DIR": local_dir,
            "STUDIO_COLLECT_ROOT": skill.get("collect_root") or "",
        },
    }
    (link / "session.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), "utf-8")
    datasets = list(skill.get("datasets") or [])
    entry = {
        "id": session_name,
        "label": session_name,
        "path": local_dir or str(link),
        "remote_path": remote_dir or "",
        "local_dir": local_dir,
        "collection_id": collection_id,
        "status": marker["status"],
    }
    datasets = [d for d in datasets if d.get("id") != session_name]
    datasets.append(entry)
    skill["datasets"] = datasets
    skill["current_session"] = session_name
    skill["current_local_dir"] = local_dir
    if collection_id:
        skill["current_collection_id"] = collection_id
    if not str(skill.get("collect_root") or "").strip() and local_dir:
        skill["collect_root"] = str(Path(local_dir).parent)
    return _persist_skill(root, skill)


def add_dataset(
    skill_id: str,
    *,
    path: str,
    label: str = "",
    remote_path: str = "",
    dataset_id: str = "",
    ready: bool | None = None,
    total_episodes: Any = None,
    valid_count: Any = None,
    labels_path: str = "",
    prompt: str = "",
    source: str = "linked",
    root: Path | None = None,
) -> dict[str, Any]:
    """Attach an existing local/remote dataset path to a skill card (04 筛选页用)."""
    root = root or skills_root()
    skill = load_skill(root, skill_id)
    if not skill:
        raise ValueError(f"技能不存在: {skill_id}")
    path = str(path or "").strip()
    if not path:
        raise ValueError("请填写数据集路径")
    if not path.startswith("/"):
        raise ValueError("数据集路径须为绝对路径")
    remote_path = str(remote_path or "").strip() or path
    name = str(dataset_id or label or Path(path).name or "").strip() or Path(path).name
    if not SAFE.fullmatch(name.replace(" ", "_")):
        name = _slugify(name)
    local_ready = _looks_like_dataset(Path(path))
    if ready is None:
        ready = local_ready or bool(remote_path)
    entry = {
        "id": name,
        "label": str(label or Path(path).name or name).strip(),
        "path": path,
        "remote_path": remote_path,
        "source": source or "linked",
        "ready": bool(ready),
        "attached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if total_episodes is not None:
        try:
            entry["total_episodes"] = int(total_episodes)
        except (TypeError, ValueError):
            pass
    if valid_count is not None:
        try:
            entry["valid_count"] = int(valid_count)
        except (TypeError, ValueError):
            pass
    if labels_path:
        entry["labels_path"] = str(labels_path).strip()
    if prompt:
        entry["prompt"] = str(prompt).strip()
    datasets = [d for d in (skill.get("datasets") or []) if isinstance(d, dict)]
    key = path.rstrip("/")
    replaced = False
    for i, d in enumerate(datasets):
        if str(d.get("path") or "").rstrip("/") == key or str(d.get("id") or "") == name:
            datasets[i] = {**d, **entry}
            replaced = True
            break
    if not replaced:
        datasets.append(entry)
    skill["datasets"] = datasets
    return _persist_skill(root, skill)


def remove_dataset(
    skill_id: str,
    *,
    path: str = "",
    dataset_id: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """Remove a linked dataset entry from a skill card (does not delete files on disk)."""
    root = root or skills_root()
    skill = load_skill(root, skill_id)
    if not skill:
        raise ValueError(f"技能不存在: {skill_id}")
    path = str(path or "").strip().rstrip("/")
    dataset_id = str(dataset_id or "").strip()
    if not path and not dataset_id:
        raise ValueError("请指定要移除的数据集 path 或 id")
    before = [d for d in (skill.get("datasets") or []) if isinstance(d, dict)]
    after = []
    for d in before:
        d_path = str(d.get("path") or "").rstrip("/")
        d_id = str(d.get("id") or "").strip()
        if path and d_path == path:
            continue
        if dataset_id and d_id == dataset_id:
            continue
        after.append(d)
    if len(after) == len(before):
        raise ValueError("未找到要移除的数据集条目")
    skill["datasets"] = after
    return _persist_skill(root, skill)


def delete_skill(skill_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Delete a skill card folder and meta JSON. Does not delete remote datasets."""
    import shutil

    root = root or skills_root()
    skill_id = str(skill_id or "").strip()
    if not SAFE.fullmatch(skill_id):
        raise ValueError("无效的技能 ID")
    folder = skill_dir(root, skill_id)
    if not folder.is_dir() and not meta_path(root, skill_id).is_file():
        raise ValueError(f"技能不存在: {skill_id}")
    if folder.is_dir():
        shutil.rmtree(folder)
    return {"id": skill_id, "deleted": True, "folder": str(folder)}


def _path_blob(d: dict[str, Any]) -> str:
    return " ".join(str(d.get(k) or "") for k in ("path", "remote_path", "label", "id")).lower()


def is_unified_dataset(d: dict[str, Any]) -> bool:
    blob = _path_blob(d)
    return "unified" in blob or "/datasets/830/" in blob


def is_mix_unified_dataset(d: dict[str, Any]) -> bool:
    return train_backend_is_mix_path(_path_blob(d))


def train_backend_is_mix_path(path: str) -> bool:
    blob = str(path or "").lower().replace("\\", "/")
    if "unified" not in blob:
        return False
    return "830mix" in blob or "/mix_" in blob or "_mix_" in blob or "mix_skill" in blob


def is_v21_collect_dataset(d: dict[str, Any]) -> bool:
    blob = _path_blob(d)
    return ("/830demo/" in blob or "/820demo/" in blob) and "unified" not in blob


def prefer_train_dataset(
    datasets: list[dict[str, Any]],
    *,
    preferred_id: str = "",
    for_mix: bool = False,
) -> dict[str, Any] | None:
    """03/04 REF：优先技能指定 id；单技能跳过 mix 包；混训优先 830mix_*_unified。

    preferred_id 若指向 raw 830demo session（仅 labels），不能当作训练 REF，
    继续回落到该技能的 *_unified。
    """
    if not datasets:
        return None

    def _usable_ref(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if not d:
            return None
        if for_mix:
            return d if is_mix_unified_dataset(d) else None
        # Single-skill train REF must be packed unified, not a labeled raw session.
        if is_v21_collect_dataset(d) or (
            not is_unified_dataset(d) and "/830demo/" in _path_blob(d)
        ):
            return None
        if is_mix_unified_dataset(d):
            return None
        return d if is_unified_dataset(d) else None

    if preferred_id:
        hit = next((d for d in datasets if str(d.get("id") or "") == preferred_id), None)
        usable = _usable_ref(hit)
        if usable:
            return usable
    ready = [d for d in datasets if d.get("ready")]
    pool = ready or datasets
    if for_mix:
        mix = next((d for d in pool if is_mix_unified_dataset(d)), None)
        if mix:
            return mix
        uni = next((d for d in pool if is_unified_dataset(d)), None)
        return uni or pool[0]
    uni = next((d for d in pool if is_unified_dataset(d) and not is_mix_unified_dataset(d)), None)
    if uni:
        return uni
    # No packed single-skill unified: do not fall back to raw 830demo sessions as REF.
    return None


def resolve_train_ref_from_selection(
    *,
    selected: str,
    task: dict[str, Any],
    remote_dataset_path: str = "",
    train_profile: str = "",
) -> tuple[str, str, str]:
    """Map UI selection → (ref_root, labels_path, note).

    Labeled raw sessions only supply valid/invalid; packed task.ref_root / *_unified
    is the actual distill REF (no re-pack needed when unified already exists).
    """
    selected = str(selected or "").strip()
    remote = str(remote_dataset_path or "").strip()
    task_ref = str(task.get("ref_root") or "").strip()
    profile = str(train_profile or task.get("train_profile") or "").strip()
    for_mix = profile == "mix_vision_isaac" or bool(task.get("mix"))

    # Labels come from the selected dataset entry when present.
    labels = str(task.get("labels_path") or task.get("valid_json") or "").strip()
    selected_entry = None
    for d in task.get("datasets") or []:
        if not isinstance(d, dict):
            continue
        for key in ("path", "remote_path"):
            if str(d.get(key) or "").rstrip("/") == selected.rstrip("/"):
                selected_entry = d
                break
        if selected_entry:
            break
    if selected_entry and selected_entry.get("labels_path"):
        labels = str(selected_entry.get("labels_path") or labels).strip()

    cand = remote or selected or task_ref
    if for_mix:
        if cand and is_mix_unified_dataset({"path": cand}):
            return cand, labels, ""
        if task_ref and is_mix_unified_dataset({"path": task_ref}):
            return task_ref, labels, "selection_was_not_mix_pack"
        return cand, labels, ""

    # Single skill: never train on raw 830demo session folders.
    raw_selected = bool(cand) and (
        is_v21_collect_dataset({"path": cand})
        or ("/830demo/" in cand.replace("\\", "/").lower() and "unified" not in cand.lower())
    )
    if raw_selected and task_ref and (
        is_unified_dataset({"path": task_ref}) or "unified" in task_ref.lower()
    ):
        return task_ref, labels, f"labeled_session→ref_root:{Path(cand).name}"
    if cand and is_unified_dataset({"path": cand}) and not is_mix_unified_dataset({"path": cand}):
        return cand, labels, ""
    if task_ref:
        return task_ref, labels, "fallback_task_ref_root"
    return cand, labels, ""


def prefer_replay_dataset(datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """02 Isaac 正式回放优先 unified；否则 V2.1 采集会话。"""
    if not datasets:
        return None
    ready = [d for d in datasets if d.get("ready")]
    pool = ready or datasets
    uni = next((d for d in pool if is_unified_dataset(d)), None)
    if uni:
        return uni
    v21 = next((d for d in pool if is_v21_collect_dataset(d)), None)
    return v21 or pool[0]


def bind_train_artifacts(
    skill_id: str,
    *,
    out_dir: str = "",
    ckpts: list[dict[str, Any]] | None = None,
    student_ckpt: str = "",
    ref_root: str = "",
    defaults: dict[str, Any] | None = None,
    prompt: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """Write train outputs / params back onto the skill card."""
    fields: dict[str, Any] = {}
    out_dir = str(out_dir or "").strip()
    if out_dir:
        fields["train_out_dir"] = out_dir
    if ckpts is not None:
        fields["ckpts"] = list(ckpts)
    ckpt = str(student_ckpt or "").strip()
    if not ckpt and ckpts:
        last = next((c for c in ckpts if "last" in str(c.get("name") or "").lower()), None)
        pick = last or ckpts[0]
        ckpt = str(pick.get("path") or "")
    if ckpt:
        fields["student_ckpt"] = ckpt
    if ref_root:
        fields["ref_root"] = str(ref_root).strip()
    if defaults:
        fields["defaults"] = dict(defaults)
    if prompt:
        fields["prompt"] = str(prompt)
    if not fields:
        skill = load_skill(root or skills_root(), skill_id)
        if not skill:
            raise ValueError(f"技能不存在: {skill_id}")
        return skill
    return update_skill_fields(skill_id, fields, root=root)


def skill_as_train_task(skill: dict[str, Any], recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map a skill card into the train workspace task shape (surface)."""
    datasets = list(skill.get("datasets") or scan_skill_datasets(skill))
    recipe = recipe or {}
    # Align with cluster_0 online distill defaults (epochs/GPU shared by Studio cards).
    defaults = {
        "ngpu": 8,
        "num_envs": 32,
        "horizon": 32,
        "epochs": 4,
        "extra_steps": 0,
        "lr": "1e-4",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "warmup_steps": 0,
        "ckpt_every": 10000,
        "ckpt_every_epoch": 1,
        "ckpt_step_keep": 3,
        "hand_mode": "dex3",
        "deploy_policy": "sonic_v1_1",
        "hand_obs": "commanded",
        "cuda_devices": "0,1,2,3,4,5,6,7",
    }
    defaults.update(recipe.get("defaults") or {})
    if isinstance(skill.get("defaults"), dict):
        defaults.update({k: v for k, v in skill["defaults"].items() if v is not None and v != ""})
    recipe_ds = list(recipe.get("datasets") or [])
    # Cluster train REF is usually a packed *_unified under recipe; skill sessions are raw V2.1.
    # Merge recipe first so prefer_train_dataset() can pick unified for 03.
    if recipe_ds:
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for d in recipe_ds + datasets:
            key = str(d.get("remote_path") or d.get("path") or d.get("id") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(d)
        datasets = merged
    elif not datasets and recipe_ds:
        datasets = recipe_ds
    for_mix = str(recipe.get("train_profile") or skill.get("train_profile") or "").strip() == "mix_vision_isaac"
    preferred = prefer_train_dataset(
        datasets,
        preferred_id=str(skill.get("preferred_dataset_id") or ""),
        for_mix=for_mix,
    )
    skill_prompt = str(skill.get("prompt") or recipe.get("default_prompt") or "").strip()
    # Stamp skill prompt onto datasets when missing / collector "demo".
    stamped: list[dict[str, Any]] = []
    for d in datasets:
        if not isinstance(d, dict):
            continue
        row = dict(d)
        cur = str(row.get("prompt") or "").strip()
        if not cur or cur.lower() in {"demo", "task", "none", "null", "n/a", "-"}:
            if skill_prompt:
                row["prompt"] = skill_prompt
        stamped.append(row)
    datasets = stamped
    # Prefer cluster unified REF on the task surface when skill.ref_root still points at raw parent.
    ref_root = str(skill.get("ref_root") or "").strip()
    if preferred:
        pref_path = str(preferred.get("remote_path") or preferred.get("path") or "").strip()
        if pref_path and (
            not ref_root
            or (is_unified_dataset(preferred) and "unified" not in ref_root.lower())
            or (not for_mix and is_mix_unified_dataset({"path": ref_root}))
        ):
            ref_root = pref_path
    if not for_mix and ref_root and is_mix_unified_dataset({"path": ref_root}):
        # Drop accidental mix REF for single-skill cards.
        ref_root = str((preferred or {}).get("remote_path") or (preferred or {}).get("path") or "") if preferred else ""

    train_profile = resolve_train_profile_for_skill(skill, recipe)
    profile_flags = {}
    try:
        from train_backend import LAUNCHER_SCRIPT, profile_fixed_flags

        launcher = LAUNCHER_SCRIPT
        if train_profile:
            profile_flags = profile_fixed_flags(train_profile)
    except Exception:  # noqa: BLE001
        launcher = "tools/train/run_online_vlm_mix_distill.sh"

    out = {
        "id": skill["id"],
        "title": skill.get("title") or skill["id"],
        "skill": skill["id"],
        "badge": skill.get("badge") or recipe.get("badge") or "SKILL",
        "subtitle": recipe.get("subtitle") or "通用技能卡",
        "description": skill.get("description") or skill.get("prompt") or recipe.get("description") or "",
        "accent": recipe.get("accent") or "#2f6b4f",
        "default_prompt": skill_prompt or skill.get("prompt") or recipe.get("default_prompt") or "",
        "datasets": datasets,
        "preferred_dataset": preferred,
        "valid_json": skill.get("labels_path") or skill.get("valid_json"),
        "invalid_json": skill.get("invalid_json"),
        "labels_path": skill.get("labels_path") or "",
        "manifest_name": skill.get("manifest_name") or "",
        "task_name": skill.get("task_name") or "",
        "defaults": defaults,
        "train_out_dir": skill.get("train_out_dir") or "",
        "student_ckpt": skill.get("student_ckpt") or "",
        "ckpts": list(skill.get("ckpts") or []),
        "ref_root": ref_root
        or str((preferred or {}).get("remote_path") or "")
        or str((preferred or {}).get("path") or "")
        or "",
        "folder": skill.get("folder"),
        "custom": True,
        "from_skill": True,
        "script": launcher,
        "train_profile": train_profile or "",
        "recipe_matched": bool(recipe.get("id")),
    }
    if profile_flags:
        out["fixed_flags"] = dict(profile_flags)
    if recipe.get("fixed_flags"):
        out["fixed_flags"] = {
            **dict(out.get("fixed_flags") or {}),
            **dict(recipe.get("fixed_flags") or {}),
        }
    rid = skill.get("recipe_id") or recipe.get("id")
    if rid:
        out["recipe_id"] = rid
    if not train_profile:
        out["train_blocked"] = True
        out["train_block_reason"] = (
            "未解析到 train_profile：请在技能卡设置 train_profile，"
            "或确保能匹配内置配方（skill1/2/3/4/6 / mix）"
        )
    elif not out.get("ref_root"):
        # No pre-packed unified: still trainable if labeled raw sessions exist
        # (03 multi-select → valid-only pack → distill).
        labeled = [
            d for d in datasets
            if isinstance(d, dict)
            and int(d.get("valid_count") or 0) > 0
            and not is_unified_dataset(d)
            and not is_mix_unified_dataset(d)
        ]
        if labeled:
            out["train_pack_on_start"] = True
            out["train_hint"] = (
                f"无现成 *_unified；启动时按勾选 session 的 valid 自动 pack"
                f"（当前挂载 {len(labeled)} 个已标 session）"
            )
        else:
            out["train_blocked"] = True
            out["train_block_reason"] = (
                "无可用单技能 *_unified REF，也没有已标注 valid 的 raw session。"
                "请先在 02 标注导出 valid，或 pack/挂载该技能的 unified+VLM cache"
            )
    return out


def resolve_train_profile_for_skill(
    skill: dict[str, Any],
    recipe: dict[str, Any] | None = None,
) -> str:
    """Explicit skill/recipe profile, else adaptive default for unmatched teleop cards."""
    recipe = recipe or {}
    for src in (skill.get("train_profile"), recipe.get("train_profile")):
        key = str(src or "").strip()
        if key:
            return key
    # Adaptive: unmatched skill cards default to pure vision teleop (never silent skill1 RTC).
    if recipe.get("id"):
        return ""
    return "vision_teleop"


def match_train_recipe(skill: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Best-effort map a universal skill card onto a train_catalog recipe."""
    catalog = catalog or _load_json_catalog("train_catalog.json")
    tasks = list(catalog.get("tasks") or [])
    sid = str(skill.get("id") or "").lower()
    badge = str(skill.get("badge") or "").lower()
    title = str(skill.get("title") or "").lower()
    task_name = str(skill.get("task_name") or "").lower()
    remote = str(skill.get("remote_dir") or "").lower()
    blob = f"{sid} {badge} {title} {task_name} {remote}"
    explicit = str(skill.get("recipe_id") or "").strip()
    if explicit:
        hit = next((t for t in tasks if str(t.get("id") or "") == explicit), None)
        if hit:
            return hit

    def find(key: str) -> dict[str, Any] | None:
        return next((t for t in tasks if str(t.get("id") or "").lower() == key), None)

    if sid:
        hit = find(sid)
        if hit:
            return hit
    if "mix" in blob or "skill1234" in blob:
        return find("830mix_skill1234_demo5") or find("830mix_skill123_demo5")
    if "skill123" in blob:
        return find("830mix_skill123_demo5")
    if "pick" in blob and "toy" in blob:
        return find("skill2_pico_pick_toy")
    if "skill4" in blob or "flower" in blob or "put_the_flower" in blob:
        return find("skill4") or find("skill4_flower")
    if "skill6" in blob:
        return find("skill6")
    if "skill3" in blob or ("place" in blob and "basket" in blob):
        return find("skill3_place_basket") or find("skill3")
    if "pico" in blob or "skill2" in blob:
        return find("skill2_pico") or find("skill2")
    if "walk" in blob or "black" in blob or "skill1" in blob:
        return find("skill1_walk") or find("skill1")
    return None


def mix_train_task(skills: list[dict[str, Any]], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(skills) < 2:
        raise ValueError("混合训练请至少选择 2 个技能")
    catalog = catalog or _load_json_catalog("train_catalog.json")
    ids = [str(s.get("id") or "") for s in skills]
    blob = " ".join(ids).lower()
    prefer_1234 = any(
        x.startswith("skill4") or "flower" in x or x.startswith("skill6")
        for x in ids
    ) or "skill4" in blob
    mix_id_key = "830mix_skill1234_demo5" if prefer_1234 else "830mix_skill123_demo5"
    mix_recipe = next((t for t in (catalog.get("tasks") or []) if t.get("id") == mix_id_key), None)
    if not mix_recipe:
        mix_recipe = next(
            (t for t in (catalog.get("tasks") or []) if t.get("id") == "830mix_skill123_demo5"),
            None,
        ) or {}

    from train_backend import (
        LAUNCHER_SCRIPT,
        is_mix_dataset_path,
        is_raw_session_path,
        load_session_label_entry,
        profile_fixed_flags,
        resolve_train_prompt,
    )

    mix_slots: list[dict[str, Any]] = []
    prompts: list[str] = []
    for s in skills:
        recipe = match_train_recipe(s, catalog)
        t = skill_as_train_task(s, recipe)
        prompt = resolve_train_prompt(
            s.get("prompt"),
            t.get("default_prompt"),
            recipe.get("default_prompt") if recipe else "",
        )
        candidates: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        def _add_cand(path: str, *, label: str = "", valid_count: Any = None, total_episodes: Any = None, prompt_hint: str = "") -> None:
            path = str(path or "").strip()
            if not path or path in seen_paths:
                return
            if is_mix_dataset_path(path):
                return
            if "unified" in path.lower() and path.rstrip("/").endswith("unified"):
                return
            seen_paths.add(path)
            vc = valid_count
            te = total_episodes
            if vc is None and Path(path).is_dir():
                lab = load_session_label_entry(path)
                vc = lab.get("valid_count")
                if te is None and lab.get("valid"):
                    te = len(lab.get("valid") or []) + len(lab.get("invalid") or [])
            candidates.append({
                "id": Path(path).name,
                "label": label or Path(path).name,
                "path": path,
                "remote_path": path,
                "valid_count": vc,
                "total_episodes": te,
                "prompt": resolve_train_prompt(prompt_hint, prompt),
            })

        for d in t.get("datasets") or []:
            if not isinstance(d, dict):
                continue
            path = str(d.get("remote_path") or d.get("path") or "").strip()
            if not path or is_mix_dataset_path(path) or is_mix_unified_dataset(d):
                continue
            if is_unified_dataset(d) and "unified" in path.lower():
                continue
            _add_cand(
                path,
                label=str(d.get("label") or Path(path).name),
                valid_count=d.get("valid_count"),
                total_episodes=d.get("total_episodes"),
                prompt_hint=str(d.get("prompt") or ""),
            )

        # Also harvest timestamp sessions under skill remote_dir (830demo).
        remote = str(s.get("remote_dir") or "").strip()
        if remote and Path(remote).is_dir():
            for child in sorted(Path(remote).iterdir()):
                if child.is_dir() and is_raw_session_path(str(child)):
                    _add_cand(str(child))

        default_paths = [
            c["path"] for c in candidates
            if int(c.get("valid_count") or 0) > 0
        ]
        if not default_paths:
            default_paths = [c["path"] for c in candidates]
        if prompt:
            prompts.append(prompt)
        mix_slots.append({
            "skill_id": str(s.get("id") or ""),
            "title": str(s.get("title") or s.get("id") or ""),
            "badge": str(s.get("badge") or ""),
            "prompt": prompt or "task",
            "dataset_paths": default_paths,
            "candidates": candidates,
            "remote_dir": remote,
        })

    has_compose_data = any(slot.get("dataset_paths") for slot in mix_slots)

    def _mix_ref_from_recipe(recipe: dict[str, Any]) -> str:
        ds = list(recipe.get("datasets") or [])
        existing = []
        for d in ds:
            p = str(d.get("remote_path") or d.get("path") or "").strip()
            if p and is_mix_unified_dataset({"path": p}) and Path(p).is_dir():
                existing.append(p)
        if existing:
            return existing[0]
        pref = prefer_train_dataset(ds, for_mix=True)
        if not pref:
            return ""
        path = str(pref.get("remote_path") or pref.get("path") or "").strip()
        return path if path and is_mix_unified_dataset({"path": path}) else ""

    # Optional legacy prebuilt mix pack (only as fallback / reference dataset list).
    ref_root = ""
    if mix_recipe:
        ref_root = _mix_ref_from_recipe(mix_recipe)
        if ref_root and not Path(ref_root).is_dir():
            ref_root = ""

    if not has_compose_data and not ref_root:
        raise ValueError(
            "混合训练需要：各技能卡下有可勾选的 raw session，"
            "或机器上已有 830mix_*_unified 混训包。"
            "请先在技能卡挂载 830demo session（02 导出 valid），或跑 mix pipeline。"
        )

    datasets = list((mix_recipe or {}).get("datasets") or [])
    for slot in mix_slots:
        for d in slot.get("candidates") or []:
            datasets.append({
                **d,
                "label": f"[{slot.get('title') or slot.get('skill_id')}] {d.get('label') or d.get('id')}",
            })

    preferred = None
    if ref_root:
        preferred = next(
            (d for d in datasets if str(d.get("remote_path") or d.get("path") or "") == ref_root),
            {"id": Path(ref_root).name, "path": ref_root, "remote_path": ref_root},
        )

    mix_id = "mix_" + "_".join(ids)[:48] + f"_{uuid.uuid4().hex[:4]}"
    defaults = dict(skill_as_train_task(skills[0], match_train_recipe(skills[0], catalog))["defaults"])
    if mix_recipe:
        defaults.update(mix_recipe.get("defaults") or {})
    fixed = {
        **profile_fixed_flags("mix_vision_isaac"),
        **dict((mix_recipe or {}).get("fixed_flags") or {}),
    }
    return {
        "id": mix_id,
        "title": "混合训练 · " + " + ".join((s.get("title") or s["id"]) for s in skills),
        "skill": "mix",
        "badge": "MIX",
        "subtitle": f"{len(skills)} 技能 · 每技能独立 prompt/数据集 · 超参共用",
        "description": (
            "按技能分别指定 prompt 与 raw session；启动后分技能 pack 再合并，"
            "用 mix_vision_isaac 共用超参训练。不会误用单个总 prompt。"
        ),
        "accent": (mix_recipe or {}).get("accent") or "#4a6fa5",
        "default_prompt": " | ".join(prompts) if prompts else "",
        "datasets": datasets,
        "preferred_dataset": preferred,
        "ref_root": ref_root if not has_compose_data else "",
        "defaults": defaults,
        "script": LAUNCHER_SCRIPT,
        "train_profile": "mix_vision_isaac",
        "fixed_flags": fixed,
        "mix_skill_ids": ids,
        "mix_slots": mix_slots,
        "mix_compose": bool(has_compose_data),
        "custom": True,
        "from_skill": True,
        "mix": True,
        "recipe_id": (mix_recipe or {}).get("id") or mix_id_key,
        "recipe_matched": bool(mix_recipe),
    }


def _load_json_catalog(rel: str) -> dict[str, Any]:
    path = APP_DIR / "phi0_pipeline" / rel
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def skill_as_infer_skill(skill: dict[str, Any], recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map universal skill card → infer skill card shape."""
    datasets = list(skill.get("datasets") or scan_skill_datasets(skill))
    preferred = prefer_train_dataset(
        datasets, preferred_id=str(skill.get("preferred_dataset_id") or "")
    )
    first = preferred or (datasets[0] if datasets else {})
    recipe = recipe or {}
    local_path = str(
        skill.get("local_ref_root")
        or first.get("path")
        or recipe.get("local_ref_root")
        or ""
    ).strip()
    remote_path = str(
        skill.get("ref_root")
        or first.get("remote_path")
        or first.get("path")
        or skill.get("remote_dir")
        or recipe.get("ref_root")
        or local_path
    ).strip()
    return {
        "id": skill["id"],
        "title": skill.get("title") or skill["id"],
        "badge": skill.get("badge") or recipe.get("badge") or "SKILL",
        "prompt": skill.get("prompt") or skill.get("description") or recipe.get("prompt") or "",
        "student_ckpt": skill.get("student_ckpt") or recipe.get("student_ckpt") or "",
        "ref_root": remote_path,
        "local_ref_root": local_path,
        "train_out_dir": skill.get("train_out_dir") or "",
        "ckpts": list(skill.get("ckpts") or []),
        "ep": int(skill.get("ep") or recipe.get("ep") or 0),
        "hand_obs": skill.get("hand_obs") or recipe.get("hand_obs") or "",
        "script": skill.get("script") or recipe.get("script") or "tools/eval/studio_cl_orchestrator.sh",
        "datasets": datasets,
        "valid_json": skill.get("valid_json"),
        "folder": skill.get("folder"),
        "from_skill": True,
        "custom": True,
        "defaults": dict(skill.get("defaults") or {}),
    }


def match_infer_recipe(skill: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    catalog = catalog or _load_json_catalog("infer_catalog.json")
    skills = list(catalog.get("skills") or [])
    sid = str(skill.get("id") or "").lower()
    if sid:
        hit = next((s for s in skills if str(s.get("id") or "").lower() == sid), None)
        if hit:
            return hit
    aliases = [
        ("mix", "830mix_skill123_demo5"),
        ("skill123", "830mix_skill123_demo5"),
        ("pick_toy", "skill2_pico_pick_toy"),
        ("toy", "skill2_pico_pick_toy"),
        ("skill3", "skill3_place_basket"),
        ("place", "skill3_place_basket"),
        ("pico", "skill2_pico"),
        ("skill2", "skill2_pico"),
        ("walk", "skill1_walk"),
        ("skill1", "skill1_walk"),
        ("blackbox", "skill1_walk"),
        ("flower", "skill4"),
        ("skill4", "skill4"),
        ("skill6", "skill6"),
    ]
    blob = f"{sid} {skill.get('badge', '')} {skill.get('title', '')}".lower()
    for key, rid in aliases:
        if key in blob:
            return next((s for s in skills if s.get("id") == rid), None)
    return None
