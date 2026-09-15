#!/usr/bin/env python3
"""Sync local Studio skill cards from the shared 830demo tree.

Idempotent: creates missing cards and attaches new sessions; does not wipe
existing skill1–N cards. Safe to run on every ``bash start.sh``.

  SKIP_SKILL_SYNC=1 bash start.sh          # skip
  DEMO_ROOT=/path/to/830demo python scripts/sync_skills_from_830demo.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

import skill_backend  # noqa: E402

DEMO_ROOT = Path(
    os.environ.get("DEMO_ROOT")
    or os.environ.get("STUDIO_SHARED_ROOT")
    or "/mnt/data2/wpy/workspace/830demo"
).expanduser()
SKILLS_ROOT = Path(
    os.environ.get("SKILLS_ROOT") or str(APP / "data" / "skills")
).expanduser()

# Stable IDs already used by this Studio (do not create duplicate folders).
REMOTE_ALIASES: dict[str, str] = {
    "skill_1_walk_to_black_box": "skill1",
    "skill_2_pico_pick_and_place_pure2": "skill2",
    "skill_3_pico_place_the_basket": "skill3",
    "skill_4_put_the_flower": "skill4",
    "skill6": "skill6",
}

# Explicit skips (empty / superseded / ignored by team).
SKIP_REMOTE_NAMES = {
    "skill_2_pico_pick_the_toy",
    "skill_5_throw_the_rubbish",
    "skill_2_pick_the_toy",
}

SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _title_for(remote_name: str, skill_id: str) -> str:
    pretty = remote_name.replace("_", " ")
    if skill_id.startswith("skill") and skill_id[5:].isdigit():
        return f"Skill{skill_id[5:]} · {pretty}"
    return pretty[:80] or skill_id


def _find_labels_manifest(remote: Path) -> Path | None:
    preferred = [
        remote / f"{remote.name}.json",
        remote / "skill_1.json",
        remote / "skill_2.json",
        remote / "skill_3.json",
        remote / "skill_4.json",
        remote / "skill_5.json",
        remote / "skill6.json",
    ]
    for p in preferred:
        if p.is_file() and not p.name.endswith(".lock"):
            return p
    for p in sorted(remote.glob("skill*.json")):
        if p.is_file() and not p.name.endswith(".lock"):
            return p
    return None


def _read_session_prompt(session: Path) -> str:
    tj = session / "meta" / "tasks.jsonl"
    if not tj.is_file():
        return ""
    try:
        for line in tj.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = str(row.get("task") or row.get("prompt") or "").strip()
            if task:
                return task
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    return ""


def scan_remote_dir(remote_dir: Path, labels_manifest: Path | None) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    labels_path = ""
    if labels_manifest and labels_manifest.is_file():
        labels_path = str(labels_manifest)
        try:
            raw = json.loads(labels_manifest.read_text("utf-8"))
            if isinstance(raw, dict):
                labels = raw
        except (OSError, json.JSONDecodeError):
            labels = {}

    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_from_entry(name: str, entry: dict[str, Any] | None) -> None:
        nonlocal sessions
        s = remote_dir / name
        info_p = s / "meta" / "info.json"
        if not info_p.is_file() or name in seen:
            return
        try:
            info = json.loads(info_p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        entry = entry if isinstance(entry, dict) else {}
        valid = sorted({int(x) for x in (entry.get("valid") or []) if str(x).lstrip("-").isdigit()})
        invalid = sorted({int(x) for x in (entry.get("invalid") or []) if str(x).lstrip("-").isdigit()})
        sessions.append({
            "name": name,
            "path": str(s),
            "total_episodes": int(info.get("total_episodes") or (len(valid) + len(invalid))),
            "prompt": _read_session_prompt(s),
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid) if valid else int(entry.get("valid_count") or 0),
        })
        seen.add(name)

    for name, entry in labels.items():
        if isinstance(entry, dict):
            add_from_entry(str(name), entry)

    for child in sorted(remote_dir.iterdir()):
        if child.is_dir() and (child / "meta" / "info.json").is_file():
            add_from_entry(child.name, labels.get(child.name) if isinstance(labels.get(child.name), dict) else None)

    return {"labels_path": labels_path, "sessions": sessions}


def discover_remote_tasks(demo_root: Path) -> list[dict[str, str]]:
    """Return [{name, remote_dir, labels_manifest}] for 830demo task folders."""
    out: list[dict[str, str]] = []
    if not demo_root.is_dir():
        return out

    # Prefer shared studio.db task list when present (includes newly registered cards).
    db = demo_root / ".humanoid_data_studio" / "studio.db"
    names_from_db: set[str] = set()
    if db.is_file():
        try:
            import sqlite3

            con = sqlite3.connect(str(db))
            for name, remote_dir in con.execute(
                "SELECT name, remote_dir FROM tasks WHERE remote_dir LIKE ?",
                (f"{str(demo_root).rstrip('/')}/%",),
            ):
                names_from_db.add(str(name))
                rd = Path(str(remote_dir))
                if not rd.is_dir():
                    continue
                if rd.name in SKIP_REMOTE_NAMES or str(name) in SKIP_REMOTE_NAMES:
                    continue
                man = _find_labels_manifest(rd)
                out.append({
                    "name": str(name),
                    "remote_dir": str(rd),
                    "labels_manifest": man.name if man else "",
                })
            con.close()
        except Exception:  # noqa: BLE001
            names_from_db = set()

    for child in sorted(demo_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in SKIP_REMOTE_NAMES:
            continue
        if any(x["remote_dir"] == str(child) for x in out):
            continue
        # Need at least one LeRobot session or a labels json.
        man = _find_labels_manifest(child)
        has_session = any(
            (p / "meta" / "info.json").is_file()
            for p in child.iterdir() if p.is_dir()
        )
        if not man and not has_session:
            continue
        out.append({
            "name": child.name,
            "remote_dir": str(child),
            "labels_manifest": man.name if man else "",
        })
    return out


def resolve_skill_id(remote_name: str, existing: list[dict[str, Any]]) -> str:
    if remote_name in REMOTE_ALIASES:
        return REMOTE_ALIASES[remote_name]
    for sk in existing:
        if str(sk.get("remote_dir") or "").rstrip("/") == f"{DEMO_ROOT}/{remote_name}".rstrip("/"):
            return str(sk["id"])
        if str(sk.get("task_name") or "") == remote_name:
            return str(sk["id"])
    # New card: prefer short skillN if folder is skillN; else folder name.
    if SAFE.fullmatch(remote_name):
        return remote_name
    return skill_backend._slugify(remote_name)  # noqa: SLF001


def upsert_skill(task: dict[str, str], *, root: Path) -> dict[str, Any]:
    remote_dir = Path(task["remote_dir"])
    remote_name = remote_dir.name
    existing = skill_backend.list_skills(root)
    skill_id = resolve_skill_id(remote_name, existing)
    man_name = task.get("labels_manifest") or ""
    man = remote_dir / man_name if man_name else _find_labels_manifest(remote_dir)
    scanned = scan_remote_dir(remote_dir, man)
    sessions = scanned["sessions"]
    screened = [s for s in sessions if s.get("valid")]

    prompts = [str(s.get("prompt") or "").strip() for s in sessions if str(s.get("prompt") or "").strip()]
    prompt = next((p for p in prompts if p.lower() != "demo"), None)
    if not prompt:
        prompt = prompts[0] if prompts else remote_name

    skill = skill_backend.load_skill(root, skill_id)
    created = False
    if not skill:
        skill_backend.create_skill(
            {
                "id": skill_id,
                "title": _title_for(remote_name, skill_id),
                "badge": skill_id.upper()[:12],
                "prompt": prompt,
                "description": f"{_title_for(remote_name, skill_id)} · {remote_dir}",
                "remote_dir": str(remote_dir),
                "source": "830demo_autosync",
            },
            root=root,
            host_id="cluster_0",
            remote_base=str(DEMO_ROOT),
        )
        created = True
        skill = skill_backend.load_skill(root, skill_id)

    fields: dict[str, Any] = {
        "remote_dir": str(remote_dir),
        "task_name": remote_name,
        "labels_path": scanned.get("labels_path") or "",
        "manifest_name": man.name if man else "",
        "labels_manifest": man.name if man else "",
        "source": str((skill or {}).get("source") or "830demo_autosync"),
    }
    if prompt and prompt.lower() != "demo":
        fields["prompt"] = prompt
    skill_backend.update_skill_fields(skill_id, fields, root=root)

    # Attach screened sessions (and unscreened ones so 02 筛选可见).
    have = {str(d.get("id") or d.get("path") or "") for d in ((skill or {}).get("datasets") or [])}
    attached = 0
    for s in sessions:
        sid = s["name"]
        if sid in have or s["path"] in have:
            continue
        vc = int(s.get("valid_count") or 0)
        te = int(s.get("total_episodes") or 0)
        label = f"{sid} (valid {vc}/{te})" if vc else f"{sid} (未筛选 {te} eps)"
        skill_backend.add_dataset(
            skill_id,
            path=s["path"],
            remote_path=s["path"],
            dataset_id=sid,
            label=label,
            ready=True,
            total_episodes=te,
            valid_count=vc or None,
            labels_path=scanned.get("labels_path") or "",
            prompt=s.get("prompt") or prompt,
            source="830demo_parent_labels" if vc else "830demo_session",
            root=root,
        )
        attached += 1

    skill = skill_backend.load_skill(root, skill_id) or {}
    if screened and not skill.get("preferred_dataset_id"):
        best = max(screened, key=lambda x: int(x.get("valid_count") or 0))
        skill_backend.update_skill_fields(
            skill_id, {"preferred_dataset_id": best["name"]}, root=root
        )

    # Light allowlists for screened sessions (local path = cluster_0 mount).
    for s in screened:
        meta = Path(s["path"]) / "meta"
        try:
            meta.mkdir(parents=True, exist_ok=True)
            allow = {
                "episode_index": s.get("valid") or [],
                "valid": s.get("valid") or [],
                "invalid": s.get("invalid") or [],
            }
            (meta / "vision_episode_allowlist.json").write_text(
                json.dumps(allow, ensure_ascii=False, indent=2) + "\n", "utf-8"
            )
        except OSError:
            pass

    return {
        "id": skill_id,
        "remote": remote_name,
        "created": created,
        "sessions": len(sessions),
        "attached": attached,
        "valid_total": sum(int(s.get("valid_count") or 0) for s in screened),
    }


def main() -> int:
    if os.environ.get("SKIP_SKILL_SYNC", "").strip() in {"1", "true", "yes"}:
        print("[skill-sync] skipped (SKIP_SKILL_SYNC=1)")
        return 0
    if not DEMO_ROOT.is_dir():
        print(f"[skill-sync] demo root missing: {DEMO_ROOT}", file=sys.stderr)
        return 0

    root = skill_backend.skills_root({"skills_root": str(SKILLS_ROOT)})
    tasks = discover_remote_tasks(DEMO_ROOT)
    if not tasks:
        print(f"[skill-sync] no tasks under {DEMO_ROOT}")
        return 0

    rows = []
    for task in tasks:
        try:
            rows.append(upsert_skill(task, root=root))
        except Exception as exc:  # noqa: BLE001
            rows.append({"remote": task.get("name"), "error": str(exc)})
            print(f"[skill-sync] fail {task.get('name')}: {exc}", file=sys.stderr)

    created = [r for r in rows if r.get("created")]
    attached = sum(int(r.get("attached") or 0) for r in rows if "error" not in r)
    print(
        f"[skill-sync] {DEMO_ROOT.name}: tasks={len(tasks)} "
        f"cards={len([r for r in rows if 'error' not in r])} "
        f"new={len(created)} new_datasets={attached}"
    )
    for r in rows:
        if r.get("error"):
            print(f"  ! {r.get('remote')}: {r['error']}")
        elif r.get("created") or r.get("attached"):
            print(
                f"  + {r['id']} ← {r['remote']} "
                f"(sessions={r.get('sessions')} attached={r.get('attached')} valid={r.get('valid_total')})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
