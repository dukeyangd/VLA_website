#!/usr/bin/env python3
"""Wipe local skill cards and rebind skill1–4 to cluster_0 830demo roots.

Does NOT delete any remote datasets. Writes per-session allowlists from parent
skill_N.json valid lists so 03 train can gate on screened valid only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

import skill_backend  # noqa: E402

COLLECT_ROOT = "/home/user/DataCollection/GR00T-WholeBodyControl/outputs"
SKILLS_ROOT = APP / "data" / "skills"

SPECS = [
    {
        "id": "skill1",
        "title": "Skill1 · Walk to black box",
        "badge": "SKILL1",
        "remote_dir": "/mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box",
        "labels_manifest": "skill_1.json",
        "recipe_id": "skill1_walk",
        "fallback_prompt": "walk to the black box",
    },
    {
        "id": "skill2",
        "title": "Skill2 · Pico pick and place",
        "badge": "SKILL2",
        "remote_dir": "/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_and_place_pure2",
        "labels_manifest": "skill_2.json",
        "recipe_id": "skill2_pico",
        "fallback_prompt": "pick and place",
    },
    {
        "id": "skill3",
        "title": "Skill3 · Place the basket",
        "badge": "SKILL3",
        "remote_dir": "/mnt/data2/wpy/workspace/830demo/skill_3_pico_place_the_basket",
        "labels_manifest": "skill_3.json",
        "recipe_id": "skill3_place_basket",
        "fallback_prompt": "place the basket",
    },
    {
        "id": "skill4",
        "title": "Skill4 · Put the flower",
        "badge": "SKILL4",
        "remote_dir": "/mnt/data2/wpy/workspace/830demo/skill_4_put_the_flower",
        "labels_manifest": "skill_4.json",
        "recipe_id": "",
        "fallback_prompt": "put the flower",
    },
]


def wipe_local_skills() -> list[str]:
    removed = []
    if SKILLS_ROOT.is_dir():
        for child in list(SKILLS_ROOT.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
                removed.append(child.name)
    SKILLS_ROOT.mkdir(parents=True, exist_ok=True)
    return removed


def ssh_py(code: str) -> str:
    proc = subprocess.run(
        ["ssh", "cluster_0", "bash", "-s"],
        input="python3 - <<'PY'\n" + code + "\nPY\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout or "ssh failed")
    return proc.stdout


def scan_remote(remote_dir: str, labels_manifest: str) -> dict:
    code = f"""
import json
from pathlib import Path
root = Path({remote_dir!r})
labels_path = root / {labels_manifest!r}
labels = json.loads(labels_path.read_text()) if labels_path.is_file() else {{}}
sessions = []
for name, entry in labels.items():
    if not isinstance(entry, dict):
        continue
    s = root / name
    if not (s / 'meta' / 'info.json').is_file():
        continue
    info = json.loads((s / 'meta' / 'info.json').read_text())
    prompt = ''
    tj = s / 'meta' / 'tasks.jsonl'
    if tj.is_file():
        for line in tj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('task'):
                prompt = str(row['task'])
                break
    valid = sorted({{int(x) for x in (entry.get('valid') or [])}})
    invalid = sorted({{int(x) for x in (entry.get('invalid') or [])}})
    sessions.append({{
        'name': name,
        'path': str(s),
        'total_episodes': int(info.get('total_episodes') or 0),
        'codebase_version': info.get('codebase_version') or '',
        'prompt': prompt,
        'valid': valid,
        'invalid': invalid,
        'valid_count': len(valid),
    }})
for c in sorted(root.iterdir()):
    if not c.is_dir() or not (c / 'meta' / 'info.json').is_file():
        continue
    if any(s['name'] == c.name for s in sessions):
        continue
    info = json.loads((c / 'meta' / 'info.json').read_text())
    prompt = ''
    tj = c / 'meta' / 'tasks.jsonl'
    if tj.is_file():
        for line in tj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('task'):
                prompt = str(row['task'])
                break
    sessions.append({{
        'name': c.name,
        'path': str(c),
        'total_episodes': int(info.get('total_episodes') or 0),
        'codebase_version': info.get('codebase_version') or '',
        'prompt': prompt,
        'valid': [],
        'invalid': [],
        'valid_count': 0,
        'unscreened': True,
    }})
print(json.dumps({{'labels_path': str(labels_path), 'sessions': sessions}}))
"""
    out = ssh_py(code).strip().splitlines()
    return json.loads(out[-1])


def write_allowlists(remote_dir: str, sessions: list[dict]) -> None:
    payload = {
        "remote_dir": remote_dir,
        "sessions": [
            {
                "name": s["name"],
                "valid": s.get("valid") or [],
                "invalid": s.get("invalid") or [],
            }
            for s in sessions
            if s.get("valid")
        ],
    }
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        local = f.name
    remote = f"/tmp/studio_allowlist_{Path(remote_dir).name}.json"
    try:
        subprocess.run(["scp", "-o", "ConnectTimeout=20", local, f"cluster_0:{remote}"], check=True)
        code = f"""
import json
from pathlib import Path
data = json.loads(Path({remote!r}).read_text())
root = Path(data['remote_dir'])
for s in data['sessions']:
    sess = root / s['name']
    meta = sess / 'meta'
    meta.mkdir(parents=True, exist_ok=True)
    valid = sorted({{int(x) for x in (s.get('valid') or [])}})
    invalid = sorted({{int(x) for x in (s.get('invalid') or [])}})
    payload = {{
        'valid': valid,
        'invalid': invalid,
        'valid_count': len(valid),
        'source': 'skill_parent_labels',
    }}
    (meta / 'sonic_qa_valid_invalid.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\\n')
    allow = {{'episode_index': valid, 'valid': valid, 'invalid': invalid}}
    (meta / 'vision_episode_allowlist.json').write_text(json.dumps(allow, ensure_ascii=False, indent=2) + '\\n')
    (meta / 'all_episode_allowlist.json').write_text(json.dumps(allow, ensure_ascii=False, indent=2) + '\\n')
    print(s['name'], len(valid))
"""
        out = ssh_py(code)
        print("  allowlists:", out.strip().replace("\n", " | "))
    finally:
        Path(local).unlink(missing_ok=True)


def main() -> int:
    removed = wipe_local_skills()
    print("wiped local skills:", removed or "(none)")

    root = skill_backend.skills_root({"skills_root": str(SKILLS_ROOT)})
    summary = []
    for spec in SPECS:
        sid = spec["id"]
        scanned = scan_remote(spec["remote_dir"], spec["labels_manifest"])
        sessions = scanned["sessions"]
        # prompt: prefer non-demo task from any session, else fallback from folder semantics
        prompts = [str(s.get("prompt") or "").strip() for s in sessions]
        prompts = [p for p in prompts if p]
        prompt = next((p for p in prompts if p.lower() != "demo"), None)
        if not prompt:
            prompt = prompts[0] if prompts else spec["fallback_prompt"]
        if prompt.lower() == "demo":
            prompt = spec["fallback_prompt"]

        skill = skill_backend.create_skill(
            {
                "id": sid,
                "title": spec["title"],
                "badge": spec["badge"],
                "prompt": prompt,
                "description": f"{spec['title']} · remote {spec['remote_dir']}",
                "remote_dir": spec["remote_dir"],
                "collect_root": COLLECT_ROOT,
                "source": "830demo_rebind",
            },
            root=root,
            host_id="cluster_0",
            remote_base="/mnt/data2/wpy/workspace/830demo",
        )
        fields = {
            "labels_manifest": spec["labels_manifest"],
            "labels_path": scanned["labels_path"],
            "manifest_name": spec["labels_manifest"],
            "task_name": Path(spec["remote_dir"]).name,
            "ref_root": spec["remote_dir"],
            "prompt": prompt,
        }
        if spec.get("recipe_id"):
            fields["recipe_id"] = spec["recipe_id"]
        skill_backend.update_skill_fields(sid, fields, root=root)

        screened = [s for s in sessions if s.get("valid")]
        for s in screened:
            label = f"{s['name']} (valid {s['valid_count']}/{s['total_episodes']})"
            skill_backend.add_dataset(
                sid,
                path=s["path"],
                remote_path=s["path"],
                dataset_id=s["name"],
                label=label,
                ready=True,
                total_episodes=s["total_episodes"],
                valid_count=s["valid_count"],
                labels_path=scanned["labels_path"],
                prompt=s.get("prompt") or prompt,
                source="830demo_parent_labels",
                root=root,
            )
        # preferred = largest valid session
        if screened:
            best = max(screened, key=lambda x: int(x.get("valid_count") or 0))
            skill_backend.update_skill_fields(
                sid, {"preferred_dataset_id": best["name"]}, root=root
            )

        write_allowlists(spec["remote_dir"], screened)
        # Local skill stubs: aggregate valid counts for UI (not used as train gate)
        folder = skill_backend.skill_dir(root, sid)
        all_valid = []
        all_invalid = []
        for s in screened:
            all_valid.extend(s.get("valid") or [])
            all_invalid.extend(s.get("invalid") or [])
        (folder / "valid.json").write_text(
            json.dumps({"note": "per-session indices live in parent skill_N.json + session meta allowlists",
                        "labels_path": scanned["labels_path"],
                        "sessions": {s["name"]: {"valid_count": s["valid_count"]} for s in screened}},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (folder / "invalid.json").write_text(
            json.dumps({"sessions": {s["name"]: {"invalid_count": len(s.get("invalid") or [])} for s in screened}},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        skill = skill_backend.load_skill(root, sid)
        row = {
            "id": sid,
            "prompt": prompt,
            "labels": scanned["labels_path"],
            "sessions": len(screened),
            "valid_total": sum(int(s["valid_count"]) for s in screened),
            "datasets": [d.get("id") for d in (skill or {}).get("datasets") or []],
        }
        summary.append(row)
        print(f"created {sid}: sessions={row['sessions']} valid_eps={row['valid_total']} prompt={prompt!r}")

    print(json.dumps({"ok": True, "skills": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
