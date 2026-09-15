#!/usr/bin/env python3
"""Humanoid robot data collection, review and phi_0 conversion studio."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, stream_with_context

import collect_backend
import skill_backend
import train_backend


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATE_DIR = APP_DIR / "state"
STATE_FILE = STATE_DIR / "studio.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
DATA_EXTENSIONS = VIDEO_EXTENSIONS | {".h5", ".hdf5", ".npz", ".parquet", ".json"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SSH_CONFIG = Path(os.environ.get("SSH_CONFIG", str(Path.home() / ".ssh" / "config"))).expanduser()
DEFAULT_SHARED_ROOT = "/mnt/data2/wpy/workspace/830demo"
SHARED_DB_RELATIVE = ".humanoid_data_studio/studio.db"
MEDIA_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_QA_ROOT_CLUSTER = "/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill2_sonic_qa"
DEFAULT_DATASET_BASE_CLUSTER = "/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830"
LOCAL_DATAPROCESS_ROOT = Path("/home/neotix/noetix/dataprocess")
LOCAL_QA_ROOT = LOCAL_DATAPROCESS_ROOT / "skill2_sonic_qa"
LOCAL_DATASET_BASE = LOCAL_DATAPROCESS_ROOT
DEFAULT_QA_ROOT = str(LOCAL_QA_ROOT if LOCAL_QA_ROOT.is_dir() else APP_DIR / "sonic_qa")
DEFAULT_DATASET_BASE = str(LOCAL_DATASET_BASE if LOCAL_DATASET_BASE.is_dir() else DEFAULT_DATASET_BASE_CLUSTER)
LOCAL_QA_DIR = Path(DEFAULT_QA_ROOT)
PRUNE_SCRIPT = LOCAL_QA_DIR / "prune_lerobot_v3_episodes.py"
if not PRUNE_SCRIPT.is_file():
    PRUNE_SCRIPT = APP_DIR / "sonic_qa" / "prune_lerobot_v3_episodes.py"
REPLAY_REASON_ZH = {
    "fall": "机器人摔倒（高度或落差超过阈值）",
    "sim_twitch": "仿真关节抖动（建议结合 Viser 画面确认）",
    "data_twitch": "磁盘 SONIC latent 抖动提示（仅预扫描，不计入失败）",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def default_state() -> dict[str, Any]:
    return {
        "hosts": [
            {
                "id": "cluster_0",
                "name": "cluster_0",
                "target": "cluster_0",
                "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
                "model_zoo": "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
                "train_data": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data",
                "train_ready": True,
                "gpus": 8,
            },
            {
                "id": "h20-0",
                "name": "h20-0",
                "target": "h20-0",
                "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
                "model_zoo": "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
                "train_data": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data",
                "train_ready": False,
                "gpus": 8,
            },
            {
                "id": "h20-1",
                "name": "h20-1",
                "target": "h20-1",
                "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
                "model_zoo": "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
                "train_data": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data",
                "train_ready": False,
                "gpus": 8,
            },
            {
                "id": "cluster_2",
                "name": "cluster_2",
                "target": "cluster_2",
                "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
                "model_zoo": "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
                "train_data": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data",
                "train_ready": False,
                "gpus": 8,
            },
        ],
        "tasks": [],
        "collections": [],
        "conversions": [],
        "replay_config": {
            "execution_mode": "remote",
            "host_id": "cluster_0",
            "qa_root": DEFAULT_QA_ROOT_CLUSTER,
            "dataset_base": DEFAULT_DATASET_BASE_CLUSTER,
            # 02 筛选默认按 LeRobot V2.1（episodes.jsonl / episode_*.parquet）
            "preferred_format": "v2.1",
            "default_dataset_path": "/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy/2026-09-04-00-41-34",
            "last_dataset_path": "/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy/2026-09-04-00-41-34",
            "viser_port": 8081,
            "dataprocess_root": str(LOCAL_DATAPROCESS_ROOT),
        },
        "replay_jobs": [],
        "replay_issues": {},
        "train_config": {
            "execution_mode": "remote",
            "host_id": "cluster_0",
            "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
            "custom_tasks": [],
            "hidden_task_ids": [],
        },
        "train_jobs": [],
        "skills_config": {
            "skills_root": str(APP_DIR / "data" / "skills"),
            "remote_base": DEFAULT_SHARED_ROOT,
            "host_id": "cluster_0",
        },
        "infer_config": {
            "execution_mode": "remote",
            "host_id": "cluster_0",
            "phi0_root": "/mnt/data2/wpy/workspace/Phi_0_wpy",
            "skills": [],
            "monitor_host": "127.0.0.1",
            "monitor_out": "/tmp/studio_zmq_monitor",
            "last_ckpt_dir": "",
            "remote_ckpt_base": "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
            "local_ckpt_cache": str(APP_DIR / "cache" / "ckpts"),
            "remote_phi0_py": "/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python",
        },
        "infer_jobs": [],
        "collect_config": dict(collect_backend.DEFAULT_COLLECT_CONFIG),
        "shared": {"enabled": True, "host_id": "cluster_0", "root": DEFAULT_SHARED_ROOT},
        "shared_cache": {"tasks": [], "sessions": [], "updated_at": None},
    }


class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            loaded = json.loads(STATE_FILE.read_text("utf-8"))
            self.data = default_state() | loaded
        except (OSError, ValueError, TypeError):
            self.data = default_state()
        self.data["shared"] = default_state()["shared"] | (self.data.get("shared") or {})
        self.data.setdefault("shared_cache", default_state()["shared_cache"])
        for group in ("collections", "conversions", "replay_jobs", "train_jobs", "infer_jobs"):
            for item in self.data.get(group) or []:
                if item.get("status") in {"running", "syncing", "stopping", "importing",
                                          "collecting_offline", "stopping_offline", "uploading",
                                          "pruning"}:
                    # 03 训练挂在 tmux（或仍在写日志）：Studio 重启后由
                    # resume_running_train_monitors 重连，不要误标 interrupted。
                    if group == "train_jobs":
                        continue
                    item["status"] = "interrupted"
                    item["message"] = "服务重启，后台任务已中断，可重新启动"
        self.data.setdefault("replay_config", default_state()["replay_config"])
        self.data.setdefault("replay_issues", {})
        # Prune worker dies on restart; don't leave UI stuck on「清理中…」
        for path, entry in list((self.data.get("replay_issues") or {}).items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("prune_status") == "running":
                entry["prune_status"] = "pending"
                entry["prune_message"] = "上次清理被服务重启中断，可重新确认删除"
                entry["updated_at"] = now_iso()
        self.data.setdefault("replay_jobs", [])
        self.data.setdefault("train_config", default_state()["train_config"])
        self.data.setdefault("train_jobs", [])
        self.data.setdefault("infer_config", default_state()["infer_config"])
        self.data.setdefault("infer_jobs", [])
        self.data["collect_config"] = default_state()["collect_config"] | (
            self.data.get("collect_config") or {}
        )
        self.data["hosts"] = train_backend.merge_builtin_train_hosts(self.data.get("hosts") or [])
        self.save()

    def save(self) -> None:
        with self.lock:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8")
            os.replace(tmp, STATE_FILE)

    def find(self, group: str, item_id: str) -> dict[str, Any] | None:
        with self.lock:
            return next((x for x in self.data[group] if x.get("id") == item_id), None)


store = Store()
workers: dict[str, dict[str, Any]] = {}
worker_lock = threading.RLock()


def api_error(message: str, code: int = 400):
    return jsonify({"ok": False, "message": message}), code


def require_abs(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not value or not value.startswith("/") or "\x00" in value:
        raise ValueError(f"{label}必须是绝对路径")
    return value.rstrip("/") or "/"


def validate_session_destination(task: dict[str, Any], host_id: str, remote_dir: str) -> None:
    task_root = task["remote_dir"].rstrip("/")
    if remote_dir == task_root or not remote_dir.startswith(task_root + "/"):
        raise ValueError(f"每次数据必须保存在任务目录的独立子目录中，例如 {task_root}/2026-08-11-20-51-09")
    duplicate = next((x for x in store.data["collections"]
                      if x.get("host_id") == host_id and x.get("remote_dir") == remote_dir), None)
    if duplicate:
        raise ValueError(f"远端 session 路径已被 {duplicate['id']} 使用，请换一个时间戳子目录")


def host_by_id(host_id: str) -> dict[str, Any]:
    host = store.find("hosts", host_id)
    if not host:
        raise ValueError("远端 Host 不存在")
    return train_backend.normalize_train_host(host)


def update_host_fields(host_id: str, **fields: Any) -> dict[str, Any]:
    with store.lock:
        for row in store.data.get("hosts") or []:
            if str(row.get("id")) == str(host_id):
                row.update(fields)
                normalized = train_backend.normalize_train_host(row)
                row.clear()
                row.update(normalized)
                store.save()
                return dict(row)
    raise ValueError("远端 Host 不存在")


def rsync_path_to_host(local_path: str, host: dict[str, Any], *, timeout: int | None = None) -> None:
    """Mirror an absolute local path onto the remote host at the same absolute path."""
    src = Path(str(local_path).rstrip("/"))
    if not src.exists():
        raise FileNotFoundError(f"本地路径不存在，无法同步: {src}")
    target = str(host.get("target") or "").strip()
    if not target:
        raise ValueError("Host 缺少 SSH target")
    remote = f"{target}:{src}"
    parent = str(src.parent)
    mkdir = run_command(
        ssh_args(target, ["mkdir", "-p", parent]),
        timeout=30,
    )
    if mkdir.returncode:
        raise RuntimeError((mkdir.stderr or mkdir.stdout or "远端 mkdir 失败").strip())
    args = [
        "rsync", "-aH", "--info=progress2", "--partial",
        *rsync_ssh(),
        f"{src}/" if src.is_dir() else str(src),
        f"{remote}/" if src.is_dir() else remote,
    ]
    # For a single file, avoid trailing slash form.
    if src.is_file():
        args = [
            "rsync", "-aH", "--info=progress2", "--partial",
            *rsync_ssh(),
            str(src),
            remote,
        ]
    proc = rsync_with_progress(args)
    if proc.returncode:
        raise RuntimeError((proc.stdout or proc.stderr or f"rsync 失败 rc={proc.returncode}").strip()[-2000:])


def rsync_path_from_host(remote_path: str, host: dict[str, Any]) -> None:
    """Pull a remote absolute path back onto the studio host (same path)."""
    dest = Path(str(remote_path).rstrip("/"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = str(host.get("target") or "").strip()
    remote = f"{target}:{dest}"
    # Probe remote type.
    probe = run_command(
        ssh_args(target, ["bash", "-lc", f"if [[ -d {shlex.quote(str(dest))} ]]; then echo DIR; elif [[ -e {shlex.quote(str(dest))} ]]; then echo FILE; else echo MISSING; fi"]),
        timeout=30,
    )
    kind = (probe.stdout or "").strip().splitlines()[-1] if probe.stdout else "MISSING"
    if kind == "MISSING":
        raise FileNotFoundError(f"远端路径不存在: {dest}")
    if kind == "DIR":
        args = [
            "rsync", "-aH", "--info=progress2", "--partial",
            *rsync_ssh(),
            f"{remote}/",
            f"{dest}/",
        ]
    else:
        args = [
            "rsync", "-aH", "--info=progress2", "--partial",
            *rsync_ssh(),
            remote,
            str(dest),
        ]
    proc = rsync_with_progress(args)
    if proc.returncode:
        raise RuntimeError((proc.stdout or proc.stderr or f"pullback rsync 失败 rc={proc.returncode}").strip()[-2000:])


def run_train_host_preflight(host_id: str) -> dict[str, Any]:
    """SSH integrity check; persist train_ready on the host record."""
    host = host_by_id(host_id)
    if str(host.get("id")) == train_backend.LOCAL_TRAIN_HOST_ID:
        update_host_fields(host_id, train_ready=True, train_ready_reason="", train_ready_at=now_iso())
        return {"ok": True, "host_id": host_id, "train_ready": True, "checks": ["local_cluster_0"], "output": ""}
    bash = train_backend.train_preflight_remote_bash()
    proc = run_command(
        ssh_args(host["target"], ["bash", "-lc", bash]),
        timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    passes = [ln[5:].strip() for ln in out.splitlines() if ln.startswith("PASS ")]
    fails = [ln[5:].strip() for ln in out.splitlines() if ln.startswith("FAIL ")]
    ready = proc.returncode == 0 and not fails
    reason = "" if ready else ("缺失: " + ", ".join(fails[:8]) if fails else (out.strip()[-400:] or f"rc={proc.returncode}"))
    update_host_fields(
        host_id,
        train_ready=ready,
        train_ready_reason=reason,
        train_ready_at=now_iso(),
        train_ready_checks=passes,
    )
    return {
        "ok": ready,
        "host_id": host_id,
        "train_ready": ready,
        "checks": passes,
        "fails": fails,
        "message": "完整性校验通过" if ready else reason,
        "output": out[-4000:],
    }


def stage_train_job_to_host(item: dict[str, Any]) -> None:
    """Rsync selected data / manifests to an off-box train host and recreate pack stage."""
    host_id = str(item.get("host_id") or "")
    if not train_backend.is_offbox_train_host(host_id):
        return
    host = host_by_id(host_id)
    train_backend.require_train_ready_host(host)
    log_line(item, f"[stage] 同步训练输入 → {host.get('name') or host_id}")
    ensure_remote_train_entrypoints(host, item)
    # Ensure zoo dirs exist remotely.
    zoo = str(host.get("model_zoo") or train_backend.DEFAULT_TRAIN_OUT_BASE)
    td = str(host.get("train_data") or train_backend.DEFAULT_TRAIN_PACK_BASE)
    out_dir = str(item.get("out_dir") or "")
    log_file = str(item.get("log_file") or "")
    mkdir_parts = [zoo, td]
    if out_dir:
        mkdir_parts.append(out_dir)
        mkdir_parts.append(str(Path(out_dir).parent))
    if log_file:
        mkdir_parts.append(str(Path(log_file).parent))
    pack = item.get("pack") if isinstance(item.get("pack"), dict) else None
    if pack and pack.get("pack_root"):
        mkdir_parts.append(str(pack.get("pack_root")))
        for key in ("raw_root", "out_dir", "nvme_dir"):
            p = str(pack.get(key) or "").strip()
            if p:
                mkdir_parts.append(p)
    mkdir_bash = " && ".join(f"mkdir -p {shlex.quote(p)}" for p in dict.fromkeys(mkdir_parts))
    mk = run_command(ssh_args(host["target"], ["bash", "-lc", mkdir_bash]), timeout=60)
    if mk.returncode:
        raise RuntimeError((mk.stderr or mk.stdout or "远端创建 zoo 失败").strip())

    staged_paths = train_backend.collect_train_stage_paths(item)
    if not staged_paths:
        raise RuntimeError("没有可同步到远端的训练数据路径")
    for path in staged_paths:
        log_line(item, f"[stage] rsync {path}")
        item["message"] = f"同步到 {host.get('name')}: {Path(path).name}"
        store.save()
        rsync_path_to_host(path, host)

    if pack:
        # Recreate pack directory skeleton + symlinks on remote.
        setup = train_backend.remote_pack_symlink_bash(pack)
        proc = run_command(ssh_args(host["target"], ["bash", "-lc", setup]), timeout=120)
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout or "远端 pack stage 失败").strip()[-1500:])
        # Verify at least one staged session exists remotely.
        sources = pack.get("stage_sources") if isinstance(pack.get("stage_sources"), dict) else {}
        for src in list(sources.values())[:3]:
            chk = run_command(
                ssh_args(host["target"], ["bash", "-lc", f"test -d {shlex.quote(str(src))}"]),
                timeout=20,
            )
            if chk.returncode:
                raise RuntimeError(f"远端缺少 session 目录: {src}")
        log_line(item, f"[stage] 远端 pack stage 就绪 · {pack.get('pack_root')}")
    else:
        ref = str(item.get("ref_root") or "").strip()
        if ref:
            chk = run_command(
                ssh_args(host["target"], ["bash", "-lc", f"test -d {shlex.quote(ref)}"]),
                timeout=20,
            )
            if chk.returncode:
                raise RuntimeError(f"远端缺少 REF 目录: {ref}")
    item["message"] = f"已同步到 {host.get('name')}，准备启动训练"
    item["staged_to_host"] = True
    store.save()


def ensure_remote_train_entrypoints(host: dict[str, Any], item: dict[str, Any] | None = None) -> None:
    """Ensure pack/distill entry scripts exist on the train host; rsync if missing."""
    phi0 = str(host.get("phi0_root") or "/mnt/data2/wpy/workspace/Phi_0_wpy").rstrip("/")
    local_phi0 = "/mnt/data2/wpy/workspace/Phi_0_wpy"
    needed = [
        "tools/train/run_online_vlm_mix_distill.sh",
        "tools/train/run_studio_selection_pack_and_distill.sh",
        "tools/data/run_830_skill2_pico_pack_and_cache.sh",
    ]
    missing: list[str] = []
    for rel in needed:
        remote = f"{phi0}/{rel}"
        proc = run_command(
            ssh_args(host["target"], ["bash", "-lc", f"test -f {shlex.quote(remote)}"]),
            timeout=20,
        )
        if proc.returncode:
            missing.append(rel)
    if not missing:
        return
    if item is not None:
        log_line(item, f"[stage] 远端缺少入口脚本，正在补齐: {', '.join(Path(m).name for m in missing)}")
    for rel in missing:
        local = Path(local_phi0) / rel
        if not local.is_file():
            raise RuntimeError(f"本机也缺少训练入口，无法同步到远端: {local}")
        # Sync single file to mirrored path.
        rsync_path_to_host(str(local), host)
    # Re-check
    still = []
    for rel in needed:
        remote = f"{phi0}/{rel}"
        proc = run_command(
            ssh_args(host["target"], ["bash", "-lc", f"test -f {shlex.quote(remote)}"]),
            timeout=20,
        )
        if proc.returncode:
            still.append(rel)
    if still:
        raise RuntimeError("远端训练入口仍缺失: " + ", ".join(still))


def pullback_train_job_from_host(item: dict[str, Any]) -> None:
    """After off-box train, copy zoo out_dir + pack_root back to cluster_0."""
    host_id = str(item.get("host_id") or "")
    if not train_backend.is_offbox_train_host(host_id):
        return
    if not item.get("staged_to_host") and item.get("execution") == "local":
        return
    host = host_by_id(host_id)
    paths = train_backend.pullback_paths_for_job(item)
    if not paths:
        log_line(item, "[pullback] 无路径可回传")
        return
    log_line(item, f"[pullback] 从 {host.get('name')} 迁回 {len(paths)} 个路径")
    ok_n = 0
    hard_errors: list[str] = []
    for path in paths:
        try:
            item["message"] = f"回传 {Path(path).name} ← {host.get('name')}"
            store.save()
            rsync_path_from_host(path, host)
            log_line(item, f"[pullback] ok {path}")
            ok_n += 1
        except FileNotFoundError as exc:
            # Cancel / early fail: out_dir or pack may not exist yet — skip, don't fail whole job.
            log_line(item, f"[pullback] skip missing {path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            log_line(item, f"[pullback] 失败 {path}: {exc}")
            hard_errors.append(f"{path}: {exc}")
    if hard_errors:
        raise RuntimeError("; ".join(hard_errors))
    if ok_n:
        item["pulled_back"] = True
        store.save()


def schedule_pullback_train_job(item: dict[str, Any], *, reason: str = "pullback") -> None:
    """Best-effort async pullback so cancel API does not block on large rsync."""
    host_id = str(item.get("host_id") or "")
    if not train_backend.is_offbox_train_host(host_id):
        return
    if not item.get("staged_to_host"):
        return
    job_id = str(item.get("id") or "")

    def _run() -> None:
        row = store.find("train_jobs", job_id) if job_id else item
        if not row:
            return
        try:
            pullback_train_job_from_host(row)
            if row.get("status") == "cancelled":
                row["message"] = f"已取消 · 已尽量回传 ckpt/pack（{reason}）"
            elif row.get("status") == "error" and not str(row.get("message") or "").endswith("回传"):
                row["message"] = f"{row.get('message') or '训练失败'} · 已尽量回传产物"
            store.save()
        except Exception as exc:  # noqa: BLE001
            log_line(row, f"[pullback] {reason} failed: {exc}")
            store.save()

    threading.Thread(target=_run, daemon=True, name=f"train-pullback-{job_id or 'x'}").start()


def task_by_id(task_id: str) -> dict[str, Any]:
    task = store.find("tasks", task_id)
    if not task:
        raise ValueError("数采任务不存在")
    return task


def run_command(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)


def ssh_args(target: str, remote_args: list[str]) -> list[str]:
    # OpenSSH joins trailing argv through a remote shell; quote every remote token explicitly.
    args = ["ssh"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    return args + ["-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
                   "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
                   "-o", "ControlPath=/tmp/humanoid-studio-%C",
                   "-o", "ConnectTimeout=8", "--", target, shlex.join(remote_args)]


def ssh_args_x11(target: str, remote_args: list[str]) -> list[str]:
    """SSH with trusted X11 forward (like ssh -Y) for interactive MuJoCo glfw windows.

    Must NOT reuse the ClearAllForwardings ControlMaster socket — X11 has to be
    negotiated on the initial connection.
    """
    args = ["ssh", "-Y"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    return args + [
        "-o", "BatchMode=yes",
        "-o", "ForwardX11=yes",
        "-o", "ForwardX11Trusted=yes",
        "-o", "ControlMaster=no",
        "-o", "ConnectTimeout=12",
        "--", target, shlex.join(remote_args),
    ]


def resolve_local_display() -> str:
    """DISPLAY for X11 apps / ssh -Y. Empty if no usable local X server."""
    disp = (os.environ.get("DISPLAY") or "").strip()
    if disp:
        return disp
    # Common desktop fallback when Studio was started without inheriting DISPLAY.
    if Path("/tmp/.X11-unix/X0").exists():
        return ":0"
    for p in sorted(Path("/tmp/.X11-unix").glob("X*")) if Path("/tmp/.X11-unix").is_dir() else []:
        name = p.name  # X0, X1, …
        if name.startswith("X") and name[1:].isdigit():
            return f":{name[1:]}"
    return ""


def find_desktop_terminal_launcher(inner_bash: str) -> list[str]:
    """Build argv to open a visible desktop terminal running inner_bash."""
    # Prefer terminals that keep the window open until the command ends.
    if shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--", "bash", "-lc", inner_bash]
    if shutil.which("xfce4-terminal"):
        return ["xfce4-terminal", "--hold", "-e", f"bash -lc {shlex.quote(inner_bash)}"]
    if shutil.which("konsole"):
        return ["konsole", "-e", "bash", "-lc", inner_bash]
    if shutil.which("xterm"):
        return ["xterm", "-hold", "-e", "bash", "-lc", inner_bash]
    if shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", "bash", "-lc", inner_bash]
    raise RuntimeError(
        "本机未找到桌面终端（gnome-terminal / xfce4-terminal / konsole / xterm）。"
        "也可在系统终端手动执行：ssh -Y cluster_0"
    )


def open_desktop_terminal(inner_bash: str, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Open a GUI terminal on the Studio host (where DISPLAY lives)."""
    disp = resolve_local_display()
    if not disp:
        raise RuntimeError(
            "本机没有 DISPLAY，无法弹出桌面终端 / MuJoCo。"
            "请在跑 Studio 的这台电脑的图形桌面里启动服务（export DISPLAY=:0）。"
        )
    run_env = dict(env or os.environ)
    run_env["DISPLAY"] = disp
    launcher = find_desktop_terminal_launcher(inner_bash)
    proc = subprocess.Popen(
        launcher,
        env=run_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True, "display": disp, "launcher": launcher[0], "pid": proc.pid}


def rsync_ssh() -> list[str]:
    args = ["ssh"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    args += ["-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes"]
    args += ["-o", "ControlMaster=auto", "-o", "ControlPersist=600",
             "-o", "ControlPath=/tmp/humanoid-studio-%C"]
    return ["-e", shlex.join(args)]


# ---------------------------------------------------------------------------
# Viser SSH local-forward (auto tunnel for remote SONIC replay)
# ---------------------------------------------------------------------------
_viser_tunnel_lock = threading.Lock()
_viser_tunnel: dict[str, Any] = {
    "proc": None,
    "port": None,
    "target": None,
    "started_at": None,
    "auto": False,
}


def _local_port_open(port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def probe_viser_http(port: int, timeout: float = 0.45) -> tuple[bool, str]:
    """HTTP probe for Viser. Empty/reset responses mean the tunnel is dead (not just busy)."""
    port = int(port)
    if not _local_port_open(port, timeout=min(0.35, timeout)):
        return False, "本机端口未监听"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            req = (
                f"GET / HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
            sock.sendall(req)
            buf = b""
            while b"\r\n\r\n" not in buf and len(buf) < 8192:
                chunk = sock.recv(2048)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            # Listening SSH forward with nothing behind it → browser NS_ERROR_NET_EMPTY_RESPONSE
            return False, "empty_response"
        head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1", errors="replace")
        status_line = head.split("\r\n", 1)[0]
        parts = status_line.split()
        code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        if code in {502, 503, 504}:
            return False, f"HTTP {code}"
        if 200 <= code < 500:
            return True, ""
        return False, status_line or f"HTTP {code}"
    except OSError as exc:
        err = str(exc).lower()
        if "timed out" in err or "timeout" in err:
            return False, "http_slow"
        if "reset" in err or "104" in err:
            return False, "empty_response"
        return False, str(exc)


def reclaim_stale_viser_port(port: int) -> dict[str, Any]:
    """Kill orphan listeners on port when HTTP is dead (stale ssh -L)."""
    port = int(port)
    info: dict[str, Any] = {"port": port, "killed": False, "message": ""}
    if not _local_port_open(port):
        info["message"] = "端口未占用"
        return info
    ok, msg = probe_viser_http(port)
    if ok or msg == "http_slow":
        info["message"] = "端口健康，无需回收"
        return info
    # Only reclaim on hard failures (empty/reset/refused), not slow busy Viser
    if msg not in {"empty_response", "本机端口未监听"} and "reset" not in msg.lower() and "refused" not in msg.lower():
        if msg.startswith("HTTP 502") or msg.startswith("HTTP 503") or msg.startswith("HTTP 504"):
            pass  # reclaim — remote viser gone behind tunnel
        else:
            info["message"] = f"保留端口（{msg}）"
            return info
    with _viser_tunnel_lock:
        _stop_viser_tunnel_locked()
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        info["killed"] = True
    except Exception as exc:  # noqa: BLE001
        info["message"] = f"fuser 失败: {exc}"
        return info
    time.sleep(0.4)
    info["message"] = f"已回收僵死转发 :{port}（原因: {msg}）"
    return info


def _viser_tunnel_cmd(target: str, port: int) -> list[str]:
    """Dedicated forwarder — do NOT use ClearAllForwardings (would drop -L)."""
    args = ["ssh"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    args += [
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ConnectTimeout=8",
        "-N",
        "-L", f"{int(port)}:127.0.0.1:{int(port)}",
        "--", str(target),
    ]
    return args


def _stop_viser_tunnel_locked() -> dict[str, Any]:
    proc = _viser_tunnel.get("proc")
    info = {
        "stopped": False,
        "port": _viser_tunnel.get("port"),
        "target": _viser_tunnel.get("target"),
    }
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        info["stopped"] = True
    _viser_tunnel.update(proc=None, port=None, target=None, started_at=None, auto=False)
    return info


def stop_viser_tunnel() -> dict[str, Any]:
    with _viser_tunnel_lock:
        return _stop_viser_tunnel_locked()


def viser_tunnel_status() -> dict[str, Any]:
    with _viser_tunnel_lock:
        proc = _viser_tunnel.get("proc")
        alive = bool(proc is not None and proc.poll() is None)
        port = _viser_tunnel.get("port")
        return {
            "alive": alive,
            "auto": bool(_viser_tunnel.get("auto")),
            "port": port,
            "target": _viser_tunnel.get("target"),
            "started_at": _viser_tunnel.get("started_at"),
            "pid": proc.pid if alive and proc is not None else None,
            "local_open": _local_port_open(int(port)) if port else False,
        }


def is_loopback_ssh_target(target: str) -> bool:
    """True when SSH target is this machine — ssh -L would self-bind and steal Viser's port."""
    t = str(target or "").strip()
    if not t:
        return True
    if t in {"localhost", "127.0.0.1", "::1", "cluster_0", "0.0.0.0"}:
        return True
    # Resolve HostName from ssh config when possible (cluster_0 → 127.0.0.1).
    try:
        proc = subprocess.run(
            ["ssh", "-G", t],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        for line in (proc.stdout or "").splitlines():
            if line.lower().startswith("hostname "):
                host = line.split(None, 1)[1].strip().lower()
                if host in {"localhost", "127.0.0.1", "::1"}:
                    return True
                break
    except Exception:  # noqa: BLE001
        pass
    return False


def ensure_viser_tunnel(target: str, port: int) -> dict[str, Any]:
    """Ensure local port forwards to remote 127.0.0.1:port. Idempotent."""
    target = str(target).strip()
    port = int(port)
    if not target:
        raise ValueError("SSH target 为空，无法建立 Viser 隧道")
    # Studio on cluster_0: never ssh -L to self (binds local port, Viser falls to next port).
    if is_loopback_ssh_target(target):
        return {
            "ok": True,
            "alive": False,
            "reused": False,
            "auto": False,
            "skipped": True,
            "port": port,
            "target": target,
            "message": f"本机 Host（{target}）无需隧道，请直接访问 http://127.0.0.1:{port}/",
        }
    with _viser_tunnel_lock:
        proc = _viser_tunnel.get("proc")
        if (
            proc is not None
            and proc.poll() is None
            and _viser_tunnel.get("port") == port
            and _viser_tunnel.get("target") == target
        ):
            return {
                "ok": True,
                "alive": True,
                "reused": True,
                "auto": True,
                "port": port,
                "target": target,
                "pid": proc.pid,
                "message": "已复用自动隧道",
            }
        # External tunnel (manual ssh -L) already listening — reuse only if HTTP is alive.
        if _local_port_open(port) and not (
            proc is not None and proc.poll() is None and _viser_tunnel.get("port") == port
        ):
            ok, msg = probe_viser_http(port)
            if ok or msg == "http_slow":
                return {
                    "ok": True,
                    "alive": True,
                    "reused": True,
                    "auto": False,
                    "port": port,
                    "target": target,
                    "message": f"本机 {port} 已可访问（外部隧道/服务），跳过自动转发",
                }
            reclaim_stale_viser_port(port)
        _stop_viser_tunnel_locked()
        proc = subprocess.Popen(
            _viser_tunnel_cmd(target, port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # Wait briefly for bind / ExitOnForwardFailure
        deadline = time.time() + 2.5
        while time.time() < deadline and proc.poll() is None and not _local_port_open(port):
            time.sleep(0.15)
        if proc.poll() is not None:
            err = b""
            try:
                err = proc.stderr.read() if proc.stderr else b""
            except Exception:  # noqa: BLE001
                pass
            msg = (err or b"").decode("utf-8", errors="replace").strip()[:400]
            raise RuntimeError(f"Viser SSH 隧道启动失败: {msg or f'exit {proc.returncode}'}")
        if not _local_port_open(port):
            # still running but not open yet — keep it, frontend will poll
            pass
        _viser_tunnel.update(
            proc=proc,
            port=port,
            target=target,
            started_at=now_iso(),
            auto=True,
        )
        return {
            "ok": True,
            "alive": True,
            "reused": False,
            "auto": True,
            "port": port,
            "target": target,
            "pid": proc.pid,
            "message": f"已自动建立 ssh -L {port}:127.0.0.1:{port} {target}",
        }


atexit.register(stop_viser_tunnel)


def remote_mkdir(host: dict[str, Any], path: str) -> tuple[bool, str]:
    proc = run_command(ssh_args(host["target"], ["mkdir", "-p", "--", path]), timeout=20)
    msg = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, msg


def log_line(item: dict[str, Any], text: str) -> None:
    with store.lock:
        lines = item.setdefault("logs", [])
        lines.append(f"[{time.strftime('%H:%M:%S')}] {text.rstrip()}")
        del lines[:-160]
        store.save()


def scan_episodes(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    def is_studio_file(path: Path) -> bool:
        return (".humanoid_data" in path.parts or path.name == "labels.json"
                or path.name.startswith(".labels-"))

    videos = sorted(p for p in root.rglob("*")
                    if p.is_file() and not is_studio_file(p)
                    and p.suffix.lower() in VIDEO_EXTENSIONS)
    # LeRobot 的每个 episode 通常有多路相机视频。按文件名聚合，并优先选择 ego/head
    # 作为质检播放器来源，避免同一个 episode 在列表里出现多次。
    grouped: dict[str, list[Path]] = {}
    for path in videos:
        grouped.setdefault(path.stem, []).append(path)
    episode_views: list[tuple[str, list[Path]]] = []
    for key in sorted(grouped):
        views = grouped[key]
        def view_rank(path: Path) -> tuple[int, str]:
            text = path.as_posix().lower()
            if "ego" in text or "head" in text:
                return 0, text
            if "left_wrist" in text:
                return 1, text
            if "right_wrist" in text:
                return 2, text
            return 3, text
        views.sort(key=view_rank)
        episode_views.append((key, views))
    result = []
    def stable_episode_id(identity: str) -> str:
        return "ep_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]

    if episode_views:
        for index, (name, views) in enumerate(episode_views[:3000]):
            video_items = []
            for path in views:
                camera = path.parent.name
                if camera.startswith("observation.images."):
                    camera = camera.removeprefix("observation.images.")
                video_items.append({"label": camera, "relative_path": path.relative_to(root).as_posix()})
            primary = views[0]
            stat = primary.stat()
            result.append({
                "id": stable_episode_id(name), "name": name,
                "relative_path": primary.relative_to(root).as_posix(),
                "videos": video_items, "media": True,
                "size": stat.st_size, "modified": stat.st_mtime,
                "status": "unreviewed",
            })
    else:
        candidates = sorted(p for p in root.rglob("*")
                            if p.is_file() and not is_studio_file(p)
                            and p.suffix.lower() in DATA_EXTENSIONS)
        for index, path in enumerate(candidates[:3000]):
            stat = path.stat()
            rel = path.relative_to(root).as_posix()
            result.append({
                "id": stable_episode_id(rel), "name": path.stem, "relative_path": rel,
                "videos": [], "media": False,
                "size": stat.st_size, "modified": stat.st_mtime,
                "status": "unreviewed",
            })
    return result


def collect_config() -> dict[str, Any]:
    return collect_backend.merge_collect_config(store.data.get("collect_config"))


def collection_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload.pop("logs", None)
    with worker_lock:
        worker = workers.get(item["id"])
        alive = worker is not None
        stack = (worker or {}).get("collect_stack")
    payload["worker_alive"] = alive
    payload["stack"] = collect_backend.stack_status(stack)
    payload["work_dir"] = str(item.get("work_dir") or (stack or {}).get("work_dir") or "")
    payload["deploy_phase"] = str(item.get("deploy_phase") or "")
    return payload


def episode_number(episode: dict[str, Any]) -> int | None:
    match = re.search(r"(\d+)$", str(episode.get("name", "")))
    return int(match.group(1)) if match else None


def labels_path(collection: dict[str, Any]) -> Path:
    return Path(collection["local_dir"]) / "labels.json"


def local_labels_payload(collection: dict[str, Any]) -> dict[str, Any]:
    # 与 cluster_0 任务级 skill_N.json 完全同构：顶层为 session 名，
    # 每个 session 只保留 valid_count / valid / invalid。
    session_name = Path(collection["local_dir"]).name
    return {session_name: remote_manifest_entry(collection)}


def write_local_labels(collection: dict[str, Any]) -> str:
    path = labels_path(collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = local_labels_payload(collection)
    fd, tmp_name = tempfile.mkstemp(prefix=".labels-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as dst:
            json.dump(payload, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    collection["labels_file"] = str(path)
    return str(path)


def load_local_label_statuses(local_root: Path) -> dict[str, str]:
    path = local_root / "labels.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    statuses: dict[str, str] = {}
    # 兼容读取改版前包含 relative_path 的详细 labels.json；下一次写入时会自动
    # 迁移为与 skill_N.json 一致的新格式。
    if payload.get("version") == 1:
        for status in ("valid", "invalid", "unreviewed"):
            for episode in payload.get(status, []):
                if isinstance(episode, dict) and isinstance(episode.get("relative_path"), str):
                    statuses[episode["relative_path"]] = status
        return statuses
    entry = payload.get(local_root.name)
    if not isinstance(entry, dict):
        return {}
    for status in ("valid", "invalid"):
        values = entry.get(status, [])
        if not isinstance(values, list):
            continue
        for number in values:
            if isinstance(number, int) and not isinstance(number, bool):
                statuses[f"#{number}"] = status
    return statuses


def manifest_basename(task_name: str) -> str:
    # skill_1_walk_to_black_box 与现有转换链路约定使用 skill_1.json。
    match = re.match(r"^(skill_\d+)(?:_|$)", task_name, re.I)
    return (match.group(1) if match else task_name) + ".json"


def remote_manifest_entry(collection: dict[str, Any]) -> dict[str, Any]:
    labeled = [episode for episode in collection.get("episodes", [])
               if episode.get("status") in {"valid", "invalid"}]
    missing = [episode["name"] for episode in labeled if episode_number(episode) is None]
    if missing:
        raise ValueError("以下已标注 Episode 无数字后缀，无法写入远端任务 JSON: "
                         + ", ".join(missing))
    valid = sorted(episode_number(episode) for episode in labeled
                   if episode["status"] == "valid")
    invalid = sorted(episode_number(episode) for episode in labeled
                     if episode["status"] == "invalid")
    return {"valid_count": len(valid), "valid": valid, "invalid": invalid}


REMOTE_MANIFEST_MERGE = r'''
import fcntl, json, os, sys, tempfile
path, session = sys.argv[1], sys.argv[2]
entry = json.load(sys.stdin)
parent = os.path.dirname(path)
os.makedirs(parent, exist_ok=True)
lock_path = path + '.lock'
with open(lock_path, 'a+', encoding='utf-8') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    data = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as src:
            data = json.load(src)
        if not isinstance(data, dict):
            raise ValueError('manifest root must be a JSON object')
    data[session] = entry
    fd, tmp = tempfile.mkstemp(prefix='.manifest-', suffix='.json', dir=parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as dst:
            json.dump(data, dst, ensure_ascii=False, indent=2)
            dst.write('\n')
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
print(json.dumps(data, ensure_ascii=False))
'''


REMOTE_SHARED_DB = r'''
import fcntl, hashlib, json, mimetypes, os, sqlite3, sys, tempfile, time

db_path, action = sys.argv[1], sys.argv[2]
payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = sqlite3.connect(db_path, timeout=15)
db.row_factory = sqlite3.Row
db.execute('PRAGMA busy_timeout=15000')
db.execute('PRAGMA foreign_keys=ON')
db.executescript("""
CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER NOT NULL);
INSERT INTO schema_meta(version) SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_meta);
CREATE TABLE IF NOT EXISTS tasks(
  name TEXT PRIMARY KEY, remote_dir TEXT NOT NULL, manifest_name TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, task_name TEXT NOT NULL REFERENCES tasks(name), name TEXT NOT NULL,
  remote_path TEXT NOT NULL UNIQUE, source_type TEXT NOT NULL, source_hostname TEXT NOT NULL,
  source_ip TEXT NOT NULL, file_count INTEGER NOT NULL DEFAULT 0,
  bytes_total INTEGER NOT NULL DEFAULT 0, uploaded_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS episodes(
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
  name TEXT NOT NULL, relative_path TEXT NOT NULL, videos_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unreviewed', version INTEGER NOT NULL DEFAULT 0,
  labeled_by_hostname TEXT, labeled_by_ip TEXT, labeled_at TEXT,
  PRIMARY KEY(session_id,id)
);
CREATE TABLE IF NOT EXISTS episode_locks(
  session_id TEXT NOT NULL, episode_id TEXT NOT NULL, client_id TEXT NOT NULL,
  hostname TEXT NOT NULL, ip TEXT NOT NULL, acquired_at REAL NOT NULL, expires_at REAL NOT NULL,
  PRIMARY KEY(session_id,episode_id),
  FOREIGN KEY(session_id,episode_id) REFERENCES episodes(session_id,id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_name);
CREATE INDEX IF NOT EXISTS idx_locks_expiry ON episode_locks(expires_at);
""")
db.commit()

def output(value):
    print(json.dumps(value, ensure_ascii=False))

def atomic_json(path, value):
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.studio-', suffix='.json', dir=parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def episode_payload(row):
    item = dict(row)
    item['videos'] = json.loads(item.pop('videos_json') or '[]')
    lock_keys = ('client_id','lock_hostname','lock_ip','expires_at')
    if item.get('client_id'):
        item['lock'] = {"client_id": item.get('client_id'), "hostname": item.get('lock_hostname'),
                        "ip": item.get('lock_ip'), "expires_at": item.get('expires_at')}
    else:
        item['lock'] = None
    for key in lock_keys: item.pop(key, None)
    return item

def regenerate_files(session_id):
    session = db.execute('SELECT s.*,t.remote_dir,t.manifest_name FROM sessions s JOIN tasks t ON t.name=s.task_name WHERE s.id=?',(session_id,)).fetchone()
    rows = db.execute('SELECT * FROM episodes WHERE session_id=? ORDER BY name',(session_id,)).fetchall()
    groups = {k: [] for k in ('valid','invalid','unreviewed')}
    for row in rows:
        number_match = __import__('re').search(r'(\d+)$', row['name'])
        groups[row['status']].append({"name":row['name'],"relative_path":row['relative_path'],
                                      "episode_number":int(number_match.group(1)) if number_match else None})
    bad = [x['name'] for status in ('valid','invalid') for x in groups[status] if x['episode_number'] is None]
    if bad: raise ValueError('labeled episodes without numeric suffix: '+', '.join(bad))
    entry = {"valid_count":len(groups['valid']),
             "valid":sorted(x['episode_number'] for x in groups['valid']),
             "invalid":sorted(x['episode_number'] for x in groups['invalid'])}
    atomic_json(os.path.join(session['remote_path'],'labels.json'), {session['name']:entry})
    manifest_path = os.path.join(session['remote_dir'],session['manifest_name'])
    with open(manifest_path+'.lock','a+',encoding='utf-8') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path,encoding='utf-8') as src: manifest=json.load(src)
        if not isinstance(manifest,dict): raise ValueError('task manifest root must be an object')
        manifest[session['name']] = entry
        atomic_json(manifest_path,manifest)

def scan_remote_session(task_name, requested_path):
    task=db.execute('SELECT * FROM tasks WHERE name=?',(task_name,)).fetchone()
    if not task: raise ValueError('shared task not found')
    task_root=os.path.realpath(task['remote_dir']); root=os.path.realpath(requested_path)
    if root == task_root or not root.startswith(task_root+os.sep):
        raise ValueError('remote session must be a child of the selected task directory')
    if not os.path.isdir(root): raise ValueError('remote session directory does not exist')
    video_ext={'.mp4','.mov','.m4v','.webm','.avi','.mkv'}
    data_ext=video_ext|{'.h5','.hdf5','.npz','.parquet','.json'}
    videos=[]; candidates=[]; file_count=0; bytes_total=0
    for current, dirs, names in os.walk(root):
        dirs[:]=[name for name in dirs if name != '.humanoid_data']
        for name in names:
            if name == 'labels.json' or name.startswith('.labels-'): continue
            path=os.path.join(current,name)
            try:
                if not os.path.isfile(path): continue
                size=os.path.getsize(path)
            except OSError:
                continue
            file_count+=1; bytes_total+=size
            ext=os.path.splitext(name)[1].lower()
            if ext in video_ext: videos.append(path)
            if ext in data_ext: candidates.append(path)
    status_by_rel={}; status_by_num={}
    labels_path=os.path.join(root,'labels.json')
    if os.path.isfile(labels_path):
        try:
            with open(labels_path,encoding='utf-8') as src: labels=json.load(src)
            if isinstance(labels,dict) and labels.get('version') == 1:
                for status in ('valid','invalid','unreviewed'):
                    for item in labels.get(status,[]):
                        if isinstance(item,dict):
                            if isinstance(item.get('relative_path'),str): status_by_rel[item['relative_path']]=status
                            if isinstance(item.get('episode_number'),int): status_by_num[item['episode_number']]=status
                        elif isinstance(item,int): status_by_num[item]=status
            else:
                entry=labels.get(os.path.basename(root),{}) if isinstance(labels,dict) else {}
                for number in entry.get('valid',[]): status_by_num[int(number)]='valid'
                for number in entry.get('invalid',[]): status_by_num[int(number)]='invalid'
        except (OSError,ValueError,TypeError):
            pass
    manifest_path=os.path.join(task['remote_dir'],task['manifest_name'])
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path,encoding='utf-8') as src: manifest=json.load(src)
            entry=manifest.get(os.path.basename(root),{}) if isinstance(manifest,dict) else {}
            for number in entry.get('valid',[]): status_by_num[int(number)]='valid'
            for number in entry.get('invalid',[]): status_by_num[int(number)]='invalid'
        except (OSError,ValueError,TypeError):
            pass
    def rel(path): return os.path.relpath(path,root).replace(os.sep,'/')
    def status_for(name, relative_path):
        if relative_path in status_by_rel: return status_by_rel[relative_path]
        match=__import__('re').search(r'(\d+)$',name)
        return status_by_num.get(int(match.group(1)),'unreviewed') if match else 'unreviewed'
    episodes=[]
    if videos:
        grouped={}
        for path in videos: grouped.setdefault(os.path.splitext(os.path.basename(path))[0],[]).append(path)
        def view_rank(path):
            text=path.replace(os.sep,'/').lower()
            if 'ego' in text or 'head' in text: return (0,text)
            if 'left_wrist' in text: return (1,text)
            if 'right_wrist' in text: return (2,text)
            return (3,text)
        for name in sorted(grouped)[:3000]:
            views=sorted(grouped[name],key=view_rank); primary=views[0]; relative=rel(primary)
            video_items=[]
            for path in views:
                camera=os.path.basename(os.path.dirname(path))
                if camera.startswith('observation.images.'): camera=camera[len('observation.images.'):]
                video_items.append({'label':camera,'relative_path':rel(path)})
            episodes.append({'id':'ep_'+hashlib.sha256(name.encode()).hexdigest()[:12],
                             'name':name,'relative_path':relative,'videos':video_items,
                             'status':status_for(name,relative)})
    else:
        for path in sorted(candidates)[:3000]:
            relative=rel(path); name=os.path.splitext(os.path.basename(path))[0]
            episodes.append({'id':'ep_'+hashlib.sha256(relative.encode()).hexdigest()[:12],
                             'name':name,'relative_path':relative,'videos':[],
                             'status':status_for(name,relative)})
    counts={key:sum(1 for episode in episodes if episode['status']==key)
            for key in ('valid','invalid','unreviewed')}
    return {'ok':True,'task':dict(task),'remote_path':root,'name':os.path.basename(root),
            'file_count':file_count,'bytes_total':bytes_total,'episodes':episodes,'counts':counts}

if action == 'init':
    output({"ok":True,"schema_version":db.execute('SELECT version FROM schema_meta').fetchone()[0]})
elif action == 'register_task':
    now = payload['updated_at']
    known=db.execute('SELECT remote_dir,manifest_name FROM tasks WHERE name=?',(payload['name'],)).fetchone()
    if known and (known['remote_dir'] != payload['remote_dir'] or known['manifest_name'] != payload['manifest_name']):
        raise ValueError('shared task name already points to a different remote directory')
    db.execute("""INSERT INTO tasks(name,remote_dir,manifest_name,created_at,updated_at) VALUES(?,?,?,?,?)
                  ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at""",
               (payload['name'],payload['remote_dir'],payload['manifest_name'],payload.get('created_at',now),now))
    db.commit(); output({"ok":True})
elif action == 'scan_session':
    output(scan_remote_session(payload['task_name'],payload['remote_path']))
elif action == 'publish':
    task, session = payload['task'], payload['session']
    if not os.path.isdir(session['remote_path']): raise ValueError('remote session directory does not exist')
    db.execute('BEGIN IMMEDIATE')
    known_task=db.execute('SELECT remote_dir,manifest_name FROM tasks WHERE name=?',(task['name'],)).fetchone()
    if known_task and (known_task['remote_dir'] != task['remote_dir'] or known_task['manifest_name'] != task['manifest_name']):
        raise ValueError('shared task name already points to a different remote directory')
    db.execute("""INSERT INTO tasks(name,remote_dir,manifest_name,created_at,updated_at) VALUES(?,?,?,?,?)
                  ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at""",
               (task['name'],task['remote_dir'],task['manifest_name'],task['created_at'],task['updated_at']))
    clash = db.execute('SELECT id FROM sessions WHERE remote_path=?',(session['remote_path'],)).fetchone()
    if clash and clash['id'] != session['id']: raise ValueError('remote session path already published')
    db.execute("""INSERT INTO sessions(id,task_name,name,remote_path,source_type,source_hostname,source_ip,file_count,bytes_total,uploaded_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                  task_name=excluded.task_name,name=excluded.name,remote_path=excluded.remote_path,
                  source_type=excluded.source_type,source_hostname=excluded.source_hostname,
                  source_ip=excluded.source_ip,file_count=excluded.file_count,bytes_total=excluded.bytes_total""",
               (session['id'],task['name'],session['name'],session['remote_path'],session['source_type'],
                session['source_hostname'],session['source_ip'],session['file_count'],session['bytes_total'],session['uploaded_at']))
    for ep in payload['episodes']:
        db.execute("""INSERT INTO episodes(session_id,id,name,relative_path,videos_json,status)
                      VALUES(?,?,?,?,?,?) ON CONFLICT(session_id,id) DO UPDATE SET
                      name=excluded.name,relative_path=excluded.relative_path,videos_json=excluded.videos_json,
                      status=CASE WHEN episodes.version=0 THEN excluded.status ELSE episodes.status END""",
                   (session['id'],ep['id'],ep['name'],ep['relative_path'],json.dumps(ep.get('videos',[])),ep.get('status','unreviewed')))
    db.commit(); output({"ok":True,"session_id":session['id']})
elif action == 'list':
    now=time.time(); db.execute('DELETE FROM episode_locks WHERE expires_at<=?',(now,)); db.commit()
    detail=str(payload.get('detail') or 'summary')
    tasks=[dict(x) for x in db.execute('SELECT * FROM tasks ORDER BY name')]
    sessions=[]
    for srow in db.execute('SELECT * FROM sessions ORDER BY uploaded_at DESC'):
        s=dict(srow)
        count_rows=db.execute(
            'SELECT status, COUNT(*) AS n FROM episodes WHERE session_id=? GROUP BY status',
            (s['id'],),
        ).fetchall()
        counts={k:0 for k in ('valid','invalid','unreviewed')}
        for row in count_rows:
            if row['status'] in counts:
                counts[row['status']]=int(row['n'])
        s['counts']=counts
        s['episode_count']=sum(counts.values())
        if detail == 'full':
            rows=db.execute("""SELECT e.*,l.client_id,l.hostname lock_hostname,l.ip lock_ip,l.expires_at
                               FROM episodes e LEFT JOIN episode_locks l ON l.session_id=e.session_id AND l.episode_id=e.id
                               WHERE e.session_id=? ORDER BY e.name""",(s['id'],)).fetchall()
            s['episodes']=[episode_payload(x) for x in rows]
        else:
            s['episodes']=[]
        sessions.append(s)
    output({"ok":True,"tasks":tasks,"sessions":sessions,"server_time":now,"detail":detail})
elif action == 'get_session':
    now=time.time(); db.execute('DELETE FROM episode_locks WHERE expires_at<=?',(now,)); db.commit()
    sid=str(payload.get('session_id') or '')
    srow=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
    if not srow: raise ValueError('session not found')
    s=dict(srow)
    rows=db.execute("""SELECT e.*,l.client_id,l.hostname lock_hostname,l.ip lock_ip,l.expires_at
                       FROM episodes e LEFT JOIN episode_locks l ON l.session_id=e.session_id AND l.episode_id=e.id
                       WHERE e.session_id=? ORDER BY e.name""",(sid,)).fetchall()
    s['episodes']=[episode_payload(x) for x in rows]
    s['counts']={k:sum(1 for x in s['episodes'] if x['status']==k) for k in ('valid','invalid','unreviewed')}
    s['episode_count']=len(s['episodes'])
    output({"ok":True,"session":s,"server_time":now})
elif action == 'lock':
    now=time.time(); lease=max(30,min(int(payload.get('lease_seconds',90)),300))
    db.execute('BEGIN IMMEDIATE'); db.execute('DELETE FROM episode_locks WHERE expires_at<=?',(now,))
    ep=db.execute('SELECT version FROM episodes WHERE session_id=? AND id=?',(payload['session_id'],payload['episode_id'])).fetchone()
    if not ep: raise ValueError('episode not found')
    held=db.execute('SELECT * FROM episode_locks WHERE session_id=? AND episode_id=?',(payload['session_id'],payload['episode_id'])).fetchone()
    if held and held['client_id'] != payload['client_id']:
        db.rollback(); output({"ok":True,"acquired":False,"version":ep['version'],"holder":dict(held)}); sys.exit(0)
    expires=now+lease
    db.execute("""INSERT INTO episode_locks(session_id,episode_id,client_id,hostname,ip,acquired_at,expires_at)
                  VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_id,episode_id) DO UPDATE SET
                  client_id=excluded.client_id,hostname=excluded.hostname,ip=excluded.ip,expires_at=excluded.expires_at""",
               (payload['session_id'],payload['episode_id'],payload['client_id'],payload['hostname'],payload['ip'],now,expires))
    db.commit(); output({"ok":True,"acquired":True,"version":ep['version'],"expires_at":expires})
elif action == 'renew':
    now=time.time(); expires=now+max(30,min(int(payload.get('lease_seconds',90)),300))
    cur=db.execute("""UPDATE episode_locks SET expires_at=? WHERE session_id=? AND episode_id=?
                      AND client_id=? AND expires_at>?""",(expires,payload['session_id'],payload['episode_id'],payload['client_id'],now))
    db.commit(); output({"ok":True,"renewed":cur.rowcount==1,"expires_at":expires})
elif action == 'release':
    cur=db.execute('DELETE FROM episode_locks WHERE session_id=? AND episode_id=? AND client_id=?',
                   (payload['session_id'],payload['episode_id'],payload['client_id']))
    db.commit(); output({"ok":True,"released":cur.rowcount==1})
elif action == 'mark':
    if payload['status'] not in ('valid','invalid'): raise ValueError('invalid label status')
    now=time.time(); db.execute('BEGIN IMMEDIATE')
    held=db.execute('SELECT * FROM episode_locks WHERE session_id=? AND episode_id=? AND client_id=? AND expires_at>?',
                    (payload['session_id'],payload['episode_id'],payload['client_id'],now)).fetchone()
    if not held: raise PermissionError('episode lock is missing or expired')
    ep=db.execute('SELECT version FROM episodes WHERE session_id=? AND id=?',(payload['session_id'],payload['episode_id'])).fetchone()
    if not ep or ep['version'] != int(payload['version']): raise RuntimeError('episode version conflict')
    new_version=ep['version']+1
    db.execute("""UPDATE episodes SET status=?,version=?,labeled_by_hostname=?,labeled_by_ip=?,labeled_at=?
                  WHERE session_id=? AND id=?""",(payload['status'],new_version,payload['hostname'],payload['ip'],
                  time.strftime('%Y-%m-%dT%H:%M:%S%z'),payload['session_id'],payload['episode_id']))
    db.execute('DELETE FROM episode_locks WHERE session_id=? AND episode_id=?',(payload['session_id'],payload['episode_id']))
    try:
        regenerate_files(payload['session_id'])
    except Exception:
        db.rollback(); raise
    db.commit(); output({"ok":True,"status":payload['status'],"version":new_version})
elif action == 'media_info':
    row=db.execute("""SELECT s.remote_path,e.videos_json,e.relative_path FROM episodes e
                      JOIN sessions s ON s.id=e.session_id WHERE e.session_id=? AND e.id=?""",
                   (payload['session_id'],payload['episode_id'])).fetchone()
    if not row: raise ValueError('episode not found')
    videos=json.loads(row['videos_json'] or '[]'); index=int(payload.get('view_index',0))
    rel=(videos[index]['relative_path'] if videos and 0<=index<len(videos) else row['relative_path'] if index==0 else None)
    if not rel: raise ValueError('video view not found')
    root=os.path.realpath(row['remote_path']); path=os.path.realpath(os.path.join(root,rel))
    if not path.startswith(root+os.sep) or not os.path.isfile(path): raise ValueError('invalid media path')
    output({"ok":True,"path":path,"size":os.path.getsize(path),"mime":mimetypes.guess_type(path)[0] or 'application/octet-stream'})
else:
    raise ValueError('unknown shared action: '+action)
'''


def local_identity() -> dict[str, str]:
    hostname = socket.gethostname()
    try:
        addresses = [x for x in socket.gethostbyname_ex(hostname)[2] if not x.startswith("127.")]
        ip = addresses[0] if addresses else "127.0.0.1"
    except OSError:
        ip = "127.0.0.1"
    return {"hostname": hostname, "ip": ip}


def shared_config() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = store.data.get("shared") or {}
    if not config.get("enabled", True):
        raise ValueError("共享工作区未启用")
    host = host_by_id(str(config.get("host_id") or "cluster_0"))
    root = require_abs(config.get("root") or DEFAULT_SHARED_ROOT, "共享根目录")
    return config, host, root.rstrip("/") + "/" + SHARED_DB_RELATIVE


def shared_call(action: str, payload: dict[str, Any] | None = None,
                timeout: int = 60) -> dict[str, Any]:
    _, host, db_path = shared_config()
    proc = subprocess.run(
        ssh_args(host["target"], ["python3", "-c", REMOTE_SHARED_DB, db_path, action]),
        input=json.dumps(payload or {}, ensure_ascii=False), text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"共享操作 {action} 失败")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"共享操作 {action} 返回内容无法解析: {exc}") from exc


def shared_list_summary() -> dict[str, Any]:
    """Light shared index: tasks + session counts, no episode payloads."""
    return shared_call("list", {"detail": "summary"}, timeout=45)


def shared_get_session(session_id: str) -> dict[str, Any]:
    result = shared_call("get_session", {"session_id": session_id}, timeout=45)
    session = result.get("session")
    if not isinstance(session, dict):
        raise RuntimeError("共享会话详情返回异常")
    return session


def shared_session_id(remote_path: str) -> str:
    return "session_" + hashlib.sha256(remote_path.encode("utf-8")).hexdigest()[:20]


def sync_local_tasks_shared() -> dict[str, Any]:
    config = store.data.get("shared") or {}
    host_id = str(config.get("host_id") or "cluster_0")
    with store.lock:
        tasks = [dict(task) for task in store.data.get("tasks", [])
                 if str(task.get("host_id")) == host_id and not task.get("shared")]
    synced: list[str] = []
    warnings: list[str] = []
    for task in tasks:
        try:
            shared_call("register_task", {
                "name": task["name"], "remote_dir": task["remote_dir"],
                "manifest_name": manifest_basename(task["name"]),
                "created_at": task.get("created_at") or now_iso(), "updated_at": now_iso(),
            }, timeout=30)
            synced.append(task["name"])
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{task.get('name', '?')}: {exc}")
    return {"synced": synced, "warnings": warnings}


def publish_collection_shared(collection: dict[str, Any]) -> str:
    task = task_by_id(collection["task_id"])
    identity = local_identity()
    session_id = shared_session_id(collection["remote_dir"])
    payload = {
        "task": {"name": task["name"], "remote_dir": task["remote_dir"],
                 "manifest_name": manifest_basename(task["name"]),
                 "created_at": task.get("created_at") or now_iso(), "updated_at": now_iso()},
        "session": {"id": session_id, "name": Path(collection["remote_dir"]).name,
                    "remote_path": collection["remote_dir"],
                    "source_type": collection.get("source_type", "unknown"),
                    "source_hostname": identity["hostname"], "source_ip": identity["ip"],
                    "file_count": collection.get("file_count", 0),
                    "bytes_total": collection.get("bytes_total", 0),
                    "uploaded_at": collection.get("uploaded_at") or now_iso()},
        "episodes": [{"id": e["id"], "name": e["name"], "relative_path": e["relative_path"],
                      "videos": e.get("videos", []), "status": e.get("status", "unreviewed")}
                     for e in collection.get("episodes", [])],
    }
    result = shared_call("publish", payload, timeout=120)
    collection["shared_session_id"] = result["session_id"]
    collection["published_at"] = now_iso()
    return result["session_id"]


def write_manifest(collection: dict[str, Any], upload: bool = True) -> str:
    local_root = Path(collection["local_dir"])
    manifest_dir = local_root / ".humanoid_data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_name = manifest_basename(collection["task_name"])
    manifest_path = manifest_dir / task_manifest_name
    task = task_by_id(collection["task_id"])
    host = host_by_id(collection["host_id"])
    remote_task_manifest = task["remote_dir"].rstrip("/") + "/" + task_manifest_name
    session_name = Path(collection["remote_dir"]).name
    entry = remote_manifest_entry(collection)
    aggregate: dict[str, Any] = {}
    if upload:
        proc = subprocess.run(
            ssh_args(host["target"], ["python3", "-c", REMOTE_MANIFEST_MERGE,
                                      remote_task_manifest, session_name]),
            input=json.dumps(entry, ensure_ascii=False), text=True, capture_output=True,
            timeout=60, check=False,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip() or "远端 JSON 原子更新失败"
            log_line(collection, f"清单更新失败 {task_manifest_name}: {message}")
            raise RuntimeError(message)
        try:
            aggregate = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"远端清单已更新，但返回内容无法解析: {exc}") from exc
    elif manifest_path.is_file():
        try:
            aggregate = json.loads(manifest_path.read_text("utf-8")) or {}
        except (OSError, ValueError):
            aggregate = {}
        aggregate[session_name] = entry
    else:
        aggregate[session_name] = entry
    manifest_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), "utf-8")
    collection["manifest_file"] = str(manifest_path)
    collection.pop("manifest_files", None)
    store.save()
    return str(manifest_path)


def merge_remote_manifest(collection: dict[str, Any]) -> str:
    task = task_by_id(collection["task_id"])
    host = host_by_id(collection["host_id"])
    filename = str(task.get("manifest_name") or "").strip() or manifest_basename(task["name"])
    remote_path = task["remote_dir"].rstrip("/") + "/" + filename
    session_name = Path(collection["remote_dir"]).name
    labels_file = Path(collection.get("labels_file") or labels_path(collection))
    try:
        labels = json.loads(labels_file.read_text("utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"无法读取本地 Session labels.json: {exc}") from exc
    local_session_name = Path(collection["local_dir"]).name
    entry = labels.get(local_session_name) if isinstance(labels, dict) else None
    if not isinstance(entry, dict) or set(entry) != {"valid_count", "valid", "invalid"}:
        raise ValueError("本地 labels.json 与任务 JSON 格式不一致")
    valid, invalid = entry.get("valid"), entry.get("invalid")
    if (not isinstance(entry.get("valid_count"), int)
            or not isinstance(valid, list) or not isinstance(invalid, list)
            or any(not isinstance(number, int) or isinstance(number, bool)
                   for number in valid + invalid)
            or entry["valid_count"] != len(valid)):
        raise ValueError("本地 labels.json 的 valid_count / valid / invalid 内容无效")
    proc = subprocess.run(
        ssh_args(host["target"], ["python3", "-c", REMOTE_MANIFEST_MERGE,
                                  remote_path, session_name]),
        input=json.dumps(entry, ensure_ascii=False),
        text=True, capture_output=True, timeout=60, check=False,
    )
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "远端 JSON 原子更新失败")
    collection["remote_manifest_file"] = remote_path
    return remote_path


def refresh_collection_files(item: dict[str, Any]) -> None:
    root = Path(item["local_dir"])
    files = [p for p in root.rglob("*")
             if p.is_file() and ".humanoid_data" not in p.parts
             and ".studio_collect" not in p.parts
             and p.name != "labels.json" and not p.name.startswith(".labels-")]
    item["file_count"] = len(files)
    item["bytes_total"] = sum(p.stat().st_size for p in files)
    known = load_local_label_statuses(root)
    known.update({e["relative_path"]: e.get("status", "unreviewed")
                  for e in item.get("episodes", [])})
    episodes = scan_episodes(root)
    for ep in episodes:
        number = episode_number(ep)
        ep["status"] = known.get(ep["relative_path"],
                                 known.get(f"#{number}", "unreviewed"))
    item["episodes"] = episodes


def finalize_offline_collection(item: dict[str, Any]) -> None:
    with store.lock:
        refresh_collection_files(item)
        write_local_labels(item)
        item["status"] = "pending_upload"
        item["completed_at"] = now_iso()
        item["last_scan_at"] = now_iso()
        item["message"] = "离线数采已结束，labels.json 已生成，等待联网上传"
        skill_id = str(item.get("skill_id") or "").strip()
        session_name = str(item.get("session_name") or Path(item.get("local_dir") or "").name).strip()
        if skill_id and session_name:
            try:
                skill_backend.attach_session(
                    skill_id,
                    session_name,
                    local_dir=str(item.get("local_dir") or ""),
                    remote_dir=str(item.get("remote_dir") or ""),
                    collection_id=str(item.get("id") or ""),
                    status="pending_upload",
                    root=skills_root_path(),
                )
            except Exception:  # noqa: BLE001
                pass
        store.save()


def recover_stale_offline_collections() -> list[str]:
    """Studio restart can leave stop mid-flight (stopping_offline, worker gone)."""
    recovered: list[str] = []
    with store.lock:
        items = list(store.data.get("collections") or [])
    for item in items:
        if item.get("source_type") != "offline":
            continue
        if item.get("status") not in {"stopping_offline", "collecting_offline"}:
            continue
        with worker_lock:
            alive = item.get("id") in workers
        if alive:
            continue
        # collecting_offline without worker = interrupted restart; still finalize labels.
        try:
            finalize_offline_collection(item)
            recovered.append(str(item.get("id") or ""))
            log_line(item, "[studio] recovered stale offline session → pending_upload")
        except Exception as exc:  # noqa: BLE001
            item["status"] = "interrupted"
            item["message"] = f"重启后恢复失败，请再点结束: {exc}"
            store.save()
            recovered.append(str(item.get("id") or ""))
    return recovered


def offline_scan_worker(item_id: str, stop: threading.Event) -> None:
    item = store.find("collections", item_id)
    if not item:
        return
    try:
        while not stop.is_set():
            with store.lock:
                refresh_collection_files(item)
                write_local_labels(item)
                item["status"] = "collecting_offline"
                item["message"] = f"正在扫描本地数据（{item['file_count']} 个文件）"
                item["last_scan_at"] = now_iso()
                store.save()
            if stop.wait(3):
                break
        item["status"] = "stopping_offline"
        item["message"] = "正在执行最终扫描并生成 labels.json"
        store.save()
        finalize_offline_collection(item)
    except Exception as exc:  # noqa: BLE001
        item["status"] = "interrupted"
        item["message"] = f"本地扫描中断: {exc}"
        log_line(item, f"错误: {exc}")
    finally:
        with worker_lock:
            worker = workers.get(item_id) or {}
            stack = worker.get("collect_stack")
            if stack:
                try:
                    collect_backend.stop_collect_stack(stack, kill_deploy=True)
                except Exception:  # noqa: BLE001
                    pass
                worker["collect_stack"] = None
            workers.pop(item_id, None)
        store.save()


def sync_worker(item_id: str, stop: threading.Event) -> None:
    item = store.find("collections", item_id)
    if not item:
        return
    try:
        host = host_by_id(item["host_id"])
        ok, msg = remote_mkdir(host, item["remote_dir"])
        if not ok:
            raise RuntimeError(msg or "远端目录创建失败")
        log_line(item, f"远端目录已就绪: {host['target']}:{item['remote_dir']}")
        while not stop.is_set():
            refresh_collection_files(item)
            item["status"] = "syncing"
            item["message"] = f"正在增量同步 {item['file_count']} 个文件"
            item["last_scan_at"] = now_iso()
            store.save()
            src = item["local_dir"].rstrip("/") + "/"
            dst = f"{host['target']}:{item['remote_dir'].rstrip('/')}/"
            args = ["rsync", "-az", "--partial", "--append-verify", *rsync_ssh(),
                    "--exclude", ".humanoid_data/", "--", src, dst]
            proc = subprocess.Popen(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with worker_lock:
                if item_id in workers:
                    workers[item_id]["proc"] = proc
            out, _ = proc.communicate()
            if out.strip():
                log_line(item, out.strip()[-2000:])
            if proc.returncode and not stop.is_set():
                item["message"] = f"同步失败，3 秒后重试 (rc={proc.returncode})"
                log_line(item, item["message"])
            else:
                item["last_sync_at"] = now_iso()
                item["message"] = "本轮增量同步完成，等待新数据"
            store.save()
            stop.wait(3)
        # “结束数采”可能恰好打断一轮 rsync；结束前再跑一次完整增量同步，
        # 确保最后几秒落盘的数据也进入远端。
        item["status"] = "stopping"
        item["message"] = "正在执行最后一次增量同步"
        store.save()
        src = item["local_dir"].rstrip("/") + "/"
        dst = f"{host['target']}:{item['remote_dir'].rstrip('/')}/"
        final_sync = run_command(
            ["rsync", "-az", "--partial", *rsync_ssh(),
             "--exclude", ".humanoid_data/", "--", src, dst],
            timeout=None,
        )
        if final_sync.returncode:
            raise RuntimeError("最终同步失败: " + (final_sync.stderr or final_sync.stdout).strip())
        refresh_collection_files(item)
        item["status"] = "completed"
        item["completed_at"] = now_iso()
        item["message"] = "数采已结束，已生成标注清单"
        write_manifest(item, upload=True)
    except Exception as exc:  # noqa: BLE001
        item["status"] = "publish_error" if item.get("uploaded_at") else "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(item_id, None)


def import_worker(item_id: str, stop: threading.Event) -> None:
    """Upload a pre-existing dataset, then publish it before review is allowed."""
    item = store.find("collections", item_id)
    if not item:
        return
    try:
        host = host_by_id(item["host_id"])
        ok, msg = remote_mkdir(host, item["remote_dir"])
        if not ok:
            raise RuntimeError(msg or "远端目录创建失败")
        item["status"] = "importing"
        item["message"] = f"正在上传外部数据集（{item['file_count']} 个文件）"
        store.save()
        src = item["local_dir"].rstrip("/") + "/"
        dst = f"{host['target']}:{item['remote_dir'].rstrip('/')}/"
        args = ["rsync", "-az", "--partial", *rsync_ssh(),
                "--exclude", ".humanoid_data/", "--", src, dst]
        proc = subprocess.Popen(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        with worker_lock:
            if item_id in workers:
                workers[item_id]["proc"] = proc
        out, _ = proc.communicate()
        if out.strip():
            log_line(item, out.strip()[-3000:])
        if stop.is_set():
            item["status"] = "cancelled"
            item["message"] = "外部数据上传已取消，已扫描的数据仍可继续标注"
        elif proc.returncode:
            raise RuntimeError(f"外部数据上传失败 (rc={proc.returncode}): {out.strip()[-1000:]}")
        else:
            item["status"] = "publishing"
            item["completed_at"] = now_iso()
            item["last_sync_at"] = now_iso()
            item["uploaded_at"] = now_iso()
            item["message"] = "外部数据已上传，正在发布到共享工作区"
            store.save()
            publish_collection_shared(item)
            item["status"] = "published"
            item["message"] = "外部数据已上传并发布，可由任意电脑打标"
    except Exception as exc:  # noqa: BLE001
        item["status"] = "publish_error" if item.get("uploaded_at") else "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(item_id, None)


_RSYNC_PROGRESS_RE = re.compile(
    r"(?P<bytes>\d[\d,]*)\s+(?P<pct>\d+)%\s+(?P<speed>\S+)\s+(?P<eta>\S+)"
)


def fmt_bytes(n: int | float = 0) -> str:
    n = float(n or 0)
    if n < 1024:
        return f"{int(n)} B"
    if n < 1048576:
        return f"{n / 1024:.1f} KB"
    if n < 1073741824:
        return f"{n / 1048576:.1f} MB"
    return f"{n / 1073741824:.2f} GB"


def _parse_rsync_progress_line(line: str) -> dict[str, Any] | None:
    text = (line or "").replace("\r", "\n").strip()
    if not text:
        return None
    match = _RSYNC_PROGRESS_RE.search(text)
    if not match:
        return None
    try:
        done = int(match.group("bytes").replace(",", ""))
        pct = max(0, min(100, int(match.group("pct"))))
    except ValueError:
        return None
    return {
        "upload_bytes_done": done,
        "upload_progress": pct,
        "upload_speed": match.group("speed"),
        "upload_eta": match.group("eta"),
    }


def rsync_with_progress(
    args: list[str],
    *,
    on_progress: Any | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run rsync and stream --info=progress2 lines to an optional callback."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )
    chunks: list[str] = []
    buf = ""
    assert proc.stdout is not None
    while True:
        piece = proc.stdout.read(256)
        if not piece:
            break
        chunks.append(piece)
        buf += piece
        while True:
            cut = -1
            for sep in ("\r", "\n"):
                idx = buf.find(sep)
                if idx >= 0 and (cut < 0 or idx < cut):
                    cut = idx
            if cut < 0:
                break
            line, buf = buf[:cut], buf[cut + 1 :]
            parsed = _parse_rsync_progress_line(line)
            if parsed and on_progress:
                try:
                    on_progress(parsed)
                except Exception:  # noqa: BLE001
                    pass
    if buf.strip():
        parsed = _parse_rsync_progress_line(buf)
        if parsed and on_progress:
            try:
                on_progress(parsed)
            except Exception:  # noqa: BLE001
                pass
    code = proc.wait()
    out = "".join(chunks)
    return subprocess.CompletedProcess(args, code, out, "")


def deferred_upload_worker(item_id: str) -> None:
    item = store.find("collections", item_id)
    if not item:
        return
    try:
        item["status"] = "uploading"
        item["message"] = "正在补传本地原始数据和 labels.json"
        item["upload_progress"] = 0
        item["upload_bytes_done"] = 0
        store.save()
        write_local_labels(item)
        host = host_by_id(item["host_id"])
        ok, message = remote_mkdir(host, item["remote_dir"])
        if not ok:
            raise RuntimeError(message or "远端 session 目录创建失败")
        src = item["local_dir"].rstrip("/") + "/"
        dst = f"{host['target']}:{item['remote_dir'].rstrip('/')}/"
        total = int(item.get("bytes_total") or 0)
        last_save = 0.0

        def _on_progress(parsed: dict[str, Any]) -> None:
            nonlocal last_save
            item["upload_bytes_done"] = int(parsed.get("upload_bytes_done") or 0)
            pct = int(parsed.get("upload_progress") or 0)
            if total > 0 and not pct:
                pct = max(0, min(99, int(item["upload_bytes_done"] * 100 / total)))
            item["upload_progress"] = pct
            speed = str(parsed.get("upload_speed") or "")
            eta = str(parsed.get("upload_eta") or "")
            item["message"] = (
                f"上传中 {pct}% · {fmt_bytes(item['upload_bytes_done'])}"
                + (f" / {fmt_bytes(total)}" if total else "")
                + (f" · {speed}" if speed else "")
                + (f" · ETA {eta}" if eta and eta not in {"0:00:00", "--:--:--"} else "")
            )
            now = time.time()
            if now - last_save >= 1.0:
                last_save = now
                store.save()

        result = rsync_with_progress(
            ["rsync", "-az", "--partial", "--mkpath", "--info=progress2", *rsync_ssh(),
             "--exclude", ".humanoid_data/", "--", src, dst],
            on_progress=_on_progress,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip() or f"rsync exit {result.returncode}"
            raise RuntimeError("补传失败: " + (detail[:480] + ("…" if len(detail) > 480 else "")))
        item["upload_progress"] = 100
        item["upload_bytes_done"] = total or int(item.get("upload_bytes_done") or 0)
        merge_remote_manifest(item)
        item["last_sync_at"] = now_iso()
        item["uploaded_at"] = now_iso()
        item["status"] = "publishing"
        item["message"] = "数据已上传，正在发布到共享工作区"
        store.save()
        publish_collection_shared(item)
        item["status"] = "published"
        item["message"] = "原始数据、标注和共享索引均已发布"
        # Auto-link uploaded remote session onto the skill card for 02.
        skill_id = str(item.get("skill_id") or item.get("task_name") or "").strip()
        remote_dir = str(item.get("remote_dir") or "").strip()
        if skill_id and remote_dir:
            try:
                session_name = Path(item.get("local_dir") or remote_dir).name
                skill_backend.attach_session(
                    skill_id,
                    session_name,
                    local_dir=str(item.get("local_dir") or ""),
                    remote_dir=remote_dir,
                    collection_id=str(item.get("id") or ""),
                    status="uploaded",
                    root=skills_root_path(),
                )
                # Prefer remote path as primary for 02 remote probe/replay.
                skill_backend.add_dataset(
                    skill_id,
                    path=remote_dir,
                    remote_path=remote_dir,
                    dataset_id=session_name,
                    label=f"{session_name} (已上传)",
                    source="uploaded_session",
                    root=skills_root_path(),
                )
                item["message"] = f"已发布并自动挂到 02 技能卡：{remote_dir}"
                log_line(item, f"[studio] linked to 02 skill={skill_id} path={remote_dir}")
            except Exception as link_exc:  # noqa: BLE001
                log_line(item, f"[studio] 02 自动挂载失败: {link_exc}")
    except Exception as exc:  # noqa: BLE001
        item["status"] = "publish_error" if item.get("uploaded_at") else "upload_error"
        item["message"] = str(exc)
        log_line(item, f"补传错误: {exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(item_id, None)


def conversion_worker(conv_id: str) -> None:
    item = store.find("conversions", conv_id)
    if not item:
        return
    if item.get("status") == "cancelled":
        return
    try:
        host = host_by_id(item["host_id"])
        ok, msg = remote_mkdir(host, item["output_dir"])
        if not ok:
            raise RuntimeError(msg or "无法创建转换输出目录")
        mapping_json = json.dumps({"version": 1, "dimension": 512, "mappings": item["mappings"]}, ensure_ascii=False)
        remote_map = item["output_dir"].rstrip("/") + "/.phi512_field_map.json"
        upload = subprocess.run(
            ssh_args(host["target"], ["sh", "-c", 'cat > "$1"', "mapping-upload", remote_map]),
            input=mapping_json, text=True, capture_output=True, timeout=30, check=False,
        )
        if upload.returncode != 0:
            raise RuntimeError("字段映射上传失败: " + upload.stderr.strip())
        # 现有 convert_skill*_to_512.sh 使用环境变量读取路径。FIELD_MAP 是本工具
        # 增加的统一字段映射约定，新版转换脚本可直接消费它。
        command = ["env", f"RAW_ROOT={item['input_dir']}", f"OUT_DIR={item['output_dir']}",
                   f"MANIFEST={item['manifest']}", f"FIELD_MAP={remote_map}",
                   "bash", item["script"]]
        command.extend(item.get("extra_args", []))
        item["command"] = shlex.join(command)
        item["message"] = "正在远端转换"
        store.save()
        proc = subprocess.Popen(ssh_args(host["target"], command), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        with worker_lock:
            workers[conv_id] = {"proc": proc, "kind": "conversion"}
        assert proc.stdout is not None
        for line in proc.stdout:
            log_line(item, line)
        code = proc.wait()
        if item.get("status") != "cancelled":
            item["status"] = "completed" if code == 0 else "error"
            item["message"] = "512D 转换完成" if code == 0 else f"转换失败 (rc={code})"
        item["completed_at"] = now_iso()
    except Exception as exc:  # noqa: BLE001
        item["status"] = "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(conv_id, None)


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.get("/")
def index():
    return send_file(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state():
    try:
        recover_stale_offline_collections()
    except Exception:  # noqa: BLE001
        pass
    with store.lock:
        data = json.loads(json.dumps(store.data))
    for item in data["collections"]:
        item.pop("logs", None)
        with worker_lock:
            worker = workers.get(item["id"])
            item["worker_alive"] = worker is not None
            stack = (worker or {}).get("collect_stack")
        item["stack"] = collect_backend.stack_status(stack)
        if not item.get("work_dir") and stack:
            item["work_dir"] = str(stack.get("work_dir") or "")
    for item in data.get("replay_jobs", []):
        item.pop("logs", None)
        with worker_lock:
            item["worker_alive"] = item["id"] in workers
    for item in data.get("train_jobs", []):
        item.pop("logs", None)
        with worker_lock:
            item["worker_alive"] = item["id"] in workers
    for item in data.get("infer_jobs", []):
        item.pop("logs", None)
        with worker_lock:
            item["worker_alive"] = item["id"] in workers
    try:
        data["skills"] = skill_backend.list_skills(skills_root_path())
    except Exception:  # noqa: BLE001
        data["skills"] = []
    data["collect_config"] = collect_config()
    return jsonify({"ok": True, **data})


def sync_shared_tasks(tasks: list[dict[str, Any]]) -> None:
    config = store.data.get("shared") or {}
    host_id = str(config.get("host_id") or "cluster_0")
    changed = False
    with store.lock:
        for remote in tasks:
            local = next((x for x in store.data["tasks"] if x.get("name") == remote.get("name")), None)
            if local:
                if local.get("remote_dir") != remote.get("remote_dir"):
                    local["remote_dir"] = remote["remote_dir"]
                    changed = True
                continue
            store.data["tasks"].append({
                "id": "shared_" + hashlib.sha256(remote["name"].encode()).hexdigest()[:10],
                "name": remote["name"], "description": "来自共享工作区",
                "host_id": host_id, "remote_dir": remote["remote_dir"],
                "created_at": remote.get("created_at") or now_iso(), "shared": True,
            })
            changed = True
        if changed:
            store.save()


@app.get("/api/shared/state")
def get_shared_state():
    try:
        result = shared_list_summary()
        result["updated_at"] = now_iso()
        with store.lock:
            store.data["shared_cache"] = {"tasks": result.get("tasks", []),
                                           "sessions": result.get("sessions", []),
                                           "updated_at": result["updated_at"]}
            store.save()
        sync_shared_tasks(result.get("tasks", []))
        return jsonify({"ok": True, "connected": True, **result})
    except Exception as exc:  # noqa: BLE001
        cache = store.data.get("shared_cache") or {"tasks": [], "sessions": [], "updated_at": None}
        return jsonify({"ok": True, "connected": False, "error": str(exc), **cache})


@app.get("/api/shared/sessions/<session_id>")
def get_shared_session_detail(session_id: str):
    try:
        session = shared_get_session(session_id)
        return jsonify({"ok": True, "session": session})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


@app.route("/api/shared/config", methods=["GET", "POST"])
def shared_config_api():
    if request.method == "GET":
        return jsonify({"ok": True, "config": store.data.get("shared")})
    data = request.get_json(silent=True) or {}
    try:
        host_by_id(str(data.get("host_id", "")))
        root = require_abs(data.get("root", ""), "共享根目录")
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    with store.lock:
        store.data["shared"] = {"enabled": bool(data.get("enabled", True)),
                                "host_id": str(data["host_id"]), "root": root}
        store.save()
    try:
        initialized = shared_call("init", timeout=30) if data.get("enabled", True) else {"ok": True}
        task_sync = sync_local_tasks_shared() if data.get("enabled", True) else {"synced": [], "warnings": []}
    except Exception as exc:  # noqa: BLE001
        return api_error(f"设置已保存，但共享工作区连接失败: {exc}", 502)
    return jsonify({"ok": True, "config": store.data["shared"], "initialized": initialized,
                    "task_sync": task_sync})


@app.post("/api/shared/collections/<item_id>/publish")
def retry_shared_publish(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("本机会话不存在", 404)
    if not item.get("uploaded_at"):
        return api_error("原始数据尚未上传，不能发布")
    try:
        session_id = publish_collection_shared(item)
        item["status"] = "published"
        item["message"] = "已发布到共享工作区"
        store.save()
        return jsonify({"ok": True, "session_id": session_id})
    except Exception as exc:  # noqa: BLE001
        item["status"] = "publish_error"
        item["message"] = str(exc)
        store.save()
        return api_error(str(exc), 502)


def scan_existing_shared_session(data: dict[str, Any]) -> dict[str, Any]:
    task_name = str(data.get("task_name", "")).strip()
    if not SAFE_NAME.fullmatch(task_name):
        raise ValueError("请选择有效的共享任务")
    remote_path = require_abs(data.get("remote_path", ""), "远端 Session 目录")
    return shared_call("scan_session", {"task_name": task_name,
                                         "remote_path": remote_path}, timeout=180)


@app.post("/api/shared/remote-sessions/scan")
def scan_existing_remote_session():
    data = request.get_json(silent=True) or {}
    try:
        result = scan_existing_shared_session(data)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


@app.post("/api/shared/remote-sessions/publish")
def publish_existing_remote_session():
    data = request.get_json(silent=True) or {}
    try:
        scanned = scan_existing_shared_session(data)
        identity = local_identity()
        remote_path = scanned["remote_path"]
        session_id = shared_session_id(remote_path)
        task = scanned["task"]
        result = shared_call("publish", {
            "task": {"name": task["name"], "remote_dir": task["remote_dir"],
                     "manifest_name": task["manifest_name"],
                     "created_at": task["created_at"], "updated_at": now_iso()},
            "session": {"id": session_id, "name": scanned["name"],
                        "remote_path": remote_path, "source_type": "existing_remote",
                        "source_hostname": identity["hostname"], "source_ip": identity["ip"],
                        "file_count": scanned["file_count"],
                        "bytes_total": scanned["bytes_total"], "uploaded_at": now_iso()},
            "episodes": scanned["episodes"],
        }, timeout=180)
        return jsonify({"ok": True, "session_id": result["session_id"],
                        "name": scanned["name"], "counts": scanned["counts"],
                        "episode_count": len(scanned["episodes"])})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


def shared_lock_payload(data: dict[str, Any]) -> dict[str, Any]:
    identity = local_identity()
    return {"session_id": str(data.get("session_id", "")),
            "episode_id": str(data.get("episode_id", "")),
            "client_id": str(data.get("client_id", "")),
            "hostname": identity["hostname"], "ip": identity["ip"], "lease_seconds": 90}


@app.post("/api/shared/locks/acquire")
def acquire_shared_lock():
    data = request.get_json(silent=True) or {}
    if not data.get("client_id"):
        return api_error("缺少浏览器 client_id")
    try:
        return jsonify(shared_call("lock", shared_lock_payload(data), timeout=30))
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


@app.post("/api/shared/locks/renew")
def renew_shared_lock():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(shared_call("renew", shared_lock_payload(data), timeout=30))
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


@app.post("/api/shared/locks/release")
def release_shared_lock():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(shared_call("release", shared_lock_payload(data), timeout=30))
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 502)


@app.post("/api/shared/mark")
def mark_shared_episode():
    data = request.get_json(silent=True) or {}
    if data.get("status") not in {"valid", "invalid"}:
        return api_error("共享标注只能选择 valid 或 invalid")
    payload = shared_lock_payload(data)
    payload.update({"status": data["status"], "version": data.get("version")})
    try:
        return jsonify(shared_call("mark", payload, timeout=60))
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc), 409 if "conflict" in str(exc).lower() else 502)


REMOTE_MEDIA_READ = r'''
import sys
path,start,length=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
with open(path,'rb') as src:
    src.seek(start)
    remaining=length
    while remaining>0:
        chunk=src.read(min(262144,remaining))
        if not chunk: break
        sys.stdout.buffer.write(chunk); remaining-=len(chunk)
'''


@app.get("/api/shared/media/<session_id>/<episode_id>/<int:view_index>")
def shared_media(session_id: str, episode_id: str, view_index: int):
    try:
        info = shared_call("media_info", {"session_id": session_id, "episode_id": episode_id,
                                          "view_index": view_index}, timeout=30)
        _, host, _ = shared_config()
    except Exception as exc:  # noqa: BLE001
        return str(exc), 502
    size = int(info["size"])
    base_headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=30"}
    if request.method == "HEAD":
        return Response(status=200, headers=base_headers | {"Content-Length": str(size)},
                        mimetype=info["mime"])
    range_header = request.headers.get("Range")
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or (not match.group(1) and not match.group(2)):
            return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
        if match.group(1):
            start = int(match.group(1)); requested_end = int(match.group(2)) if match.group(2) else size - 1
        else:
            suffix = int(match.group(2)); start = max(0, size - suffix); requested_end = size - 1
        if start >= size or requested_end < start:
            return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
        end = min(size - 1, requested_end, start + MEDIA_CHUNK_SIZE - 1)
        length = end - start + 1
        proc = subprocess.run(ssh_args(host["target"], ["python3", "-c", REMOTE_MEDIA_READ,
                                                        info["path"], str(start), str(length)]),
                              capture_output=True, timeout=45, check=False)
        if proc.returncode:
            return (proc.stderr or b"remote media read failed"), 502
        headers = base_headers | {"Content-Range": f"bytes {start}-{start + len(proc.stdout) - 1}/{size}",
                                  "Content-Length": str(len(proc.stdout))}
        return Response(proc.stdout, status=206, headers=headers, mimetype=info["mime"])
    proc = subprocess.Popen(ssh_args(host["target"], ["cat", "--", info["path"]]),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def generate():
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(262144)
                if not chunk: break
                yield chunk
        finally:
            if proc.poll() is None: proc.terminate()
    return Response(stream_with_context(generate()), status=200,
                    headers=base_headers | {"Content-Length": str(size)}, mimetype=info["mime"])


@app.post("/api/hosts")
def add_host():
    data = request.get_json(silent=True) or {}
    name, target = str(data.get("name", "")).strip(), str(data.get("target", "")).strip()
    if not name or not target or any(c.isspace() for c in target):
        return api_error("请填写 Host 名称和有效 SSH target")
    item = train_backend.normalize_train_host({
        "id": uuid.uuid4().hex[:10],
        "name": name,
        "target": target,
        "phi0_root": str(data.get("phi0_root") or train_backend.DEFAULT_HOST_TRAIN_PATHS["phi0_root"]),
        "model_zoo": str(data.get("model_zoo") or train_backend.DEFAULT_HOST_TRAIN_PATHS["model_zoo"]),
        "train_data": str(data.get("train_data") or train_backend.DEFAULT_HOST_TRAIN_PATHS["train_data"]),
        "train_ready": bool(data.get("train_ready", False)),
        "gpus": int(data.get("gpus") or 8),
    })
    with store.lock:
        store.data["hosts"].append(item)
        store.data["hosts"] = train_backend.merge_builtin_train_hosts(store.data.get("hosts") or [])
        store.save()
    return jsonify({"ok": True, "host": item})


@app.post("/api/hosts/test")
def test_host():
    data = request.get_json(silent=True) or {}
    try:
        host = host_by_id(str(data.get("host_id", "")))
        proc = run_command(ssh_args(host["target"], ["printf", "connected"]), timeout=12)
        if proc.returncode:
            return api_error((proc.stderr or proc.stdout).strip() or "连接失败")
        return jsonify({"ok": True, "message": f"{host['name']} 连接正常"})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/hosts/<host_id>/train-preflight")
def host_train_preflight(host_id: str):
    try:
        result = run_train_host_preflight(str(host_id))
        if not result.get("ok"):
            return jsonify({"ok": False, **result}), 400
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


def skills_config() -> dict[str, Any]:
    with store.lock:
        defaults = default_state()["skills_config"]
        cfg = store.data.setdefault("skills_config", dict(defaults))
        cfg.setdefault("skills_root", defaults["skills_root"])
        cfg.setdefault("remote_base", defaults["remote_base"])
        cfg.setdefault("host_id", defaults.get("host_id") or "cluster_0")
        root = str(cfg.get("skills_root") or "").strip()
        # Heal accidental unit-test pollution (tests used to overwrite live studio.json).
        bad_root = (
            not root
            or root.endswith("/state/test_skills")
            or "/test_skills" in root
            or root.startswith("/tmp/")
        )
        if bad_root:
            cfg["skills_root"] = defaults["skills_root"]
        remote = str(cfg.get("remote_base") or "").strip()
        if (not remote) or remote.startswith("/tmp/"):
            cfg["remote_base"] = defaults["remote_base"]
        return dict(cfg)


def skills_root_path() -> Path:
    return skill_backend.skills_root(skills_config())


def ensure_task_for_skill(skill: dict[str, Any]) -> dict[str, Any]:
    """Map a skill card onto a studio task so upload/collection keep working.

    Task ``name`` prefers the remote folder basename (e.g. skill_1_walk_to_black_box)
    so ``manifest_basename`` resolves to the shared parent ``skill_1.json``.
    """
    skill_id = str(skill.get("id") or "").strip()
    if not skill_id:
        raise ValueError("技能缺少 id")
    remote_dir = str(skill.get("remote_dir") or "").strip()
    task_name = str(skill.get("task_name") or "").strip()
    if not task_name and remote_dir:
        task_name = Path(remote_dir).name
    if not task_name:
        task_name = skill_id
    with store.lock:
        existing = next(
            (
                t
                for t in store.data.get("tasks") or []
                if t.get("skill_id") == skill_id
                or t.get("name") == skill_id
                or t.get("name") == task_name
            ),
            None,
        )
        if existing:
            if remote_dir:
                existing["remote_dir"] = remote_dir
            existing["skill_id"] = skill_id
            existing["name"] = task_name
            if skill.get("manifest_name"):
                existing["manifest_name"] = skill["manifest_name"]
            else:
                existing["manifest_name"] = manifest_basename(task_name)
            store.save()
            return dict(existing)
        host_id = str(skill.get("host_id") or skills_config().get("host_id") or "cluster_0")
        if not remote_dir:
            remote_dir = str(skill.get("folder") or skill_backend.skill_dir(skills_root_path(), skill_id))
        item = {
            "id": uuid.uuid4().hex[:10],
            "name": task_name,
            "description": str(skill.get("description") or skill.get("title") or "").strip(),
            "host_id": host_id,
            "remote_dir": remote_dir,
            "skill_id": skill_id,
            "manifest_name": str(skill.get("manifest_name") or manifest_basename(task_name)),
            "created_at": now_iso(),
        }
        store.data.setdefault("tasks", []).append(item)
        store.save()
        return dict(item)


@app.get("/api/skills")
def list_skills_api():
    try:
        cfg = skills_config()
        skills = skill_backend.list_skills(skills_root_path())
        return jsonify({"ok": True, "skills": skills, "config": cfg})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/skills")
def create_skill_api():
    data = request.get_json(silent=True) or {}
    try:
        cfg = skills_config()
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        # Surface: do not require live remote mkdir; folder JSON is local source of truth.
        skill = skill_backend.create_skill(
            data,
            root=skills_root_path(),
            host_id=host_id,
            remote_base=str(data.get("remote_base") or cfg.get("remote_base") or ""),
        )
        task = ensure_task_for_skill(skill)
        return jsonify({"ok": True, "skill": skill, "task": task})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.patch("/api/skills/<skill_id>")
def patch_skill_api(skill_id: str):
    data = request.get_json(silent=True) or {}
    try:
        skill = skill_backend.update_skill_fields(skill_id, data, root=skills_root_path())
        return jsonify({"ok": True, "skill": skill})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/skills/<skill_id>")
def get_skill_api(skill_id: str):
    skill = skill_backend.load_skill(skills_root_path(), skill_id)
    if not skill:
        return api_error("技能不存在", 404)
    return jsonify({"ok": True, "skill": skill})


@app.get("/api/skills/<skill_id>/datasets")
def list_skill_datasets_api(skill_id: str):
    """Scan datasets under a universal skill card for replay / train / infer."""
    root = skills_root_path()
    skill = skill_backend.load_skill(root, skill_id)
    if not skill:
        return api_error("技能不存在", 404)
    datasets = skill_backend.scan_skill_datasets(skill, root=root)
    return jsonify({
        "ok": True,
        "skill": skill,
        "datasets": datasets,
        "valid_json": skill.get("valid_json"),
        "invalid_json": skill.get("invalid_json"),
        "preferred_replay": skill_backend.prefer_replay_dataset(datasets),
        "preferred_train": skill_backend.prefer_train_dataset(
            datasets, preferred_id=str(skill.get("preferred_dataset_id") or "")
        ),
    })


@app.get("/api/skills/<skill_id>/ckpts")
def list_skill_ckpts_api(skill_id: str):
    """List ckpts bound to a skill (stored list + scan train_out_dir)."""
    root = skills_root_path()
    skill = skill_backend.load_skill(root, skill_id)
    if not skill:
        return api_error("技能不存在", 404)
    stored = list(skill.get("ckpts") or [])
    out_dir = str(skill.get("train_out_dir") or "").strip()
    scanned: list[dict[str, Any]] = []
    if out_dir:
        host_id = str(skill.get("host_id") or skills_config().get("host_id") or "cluster_0")
        execution = "local" if Path(out_dir).is_dir() else "remote"
        try:
            scanned = _list_ckpts_at(out_dir, execution, host_id)
        except Exception as exc:  # noqa: BLE001
            return jsonify({
                "ok": True,
                "skill_id": skill_id,
                "train_out_dir": out_dir,
                "student_ckpt": skill.get("student_ckpt") or "",
                "ckpts": stored,
                "warning": str(exc),
            })
    # Merge by path, prefer scanned metadata
    by_path: dict[str, dict[str, Any]] = {}
    for c in stored + scanned:
        p = str(c.get("path") or "").strip()
        if p:
            by_path[p] = {**by_path.get(p, {}), **c}
    ckpts = list(by_path.values())
    ckpts.sort(key=lambda c: (0 if "last" in str(c.get("name") or "").lower() else 1, str(c.get("name") or "")))
    return jsonify({
        "ok": True,
        "skill_id": skill_id,
        "train_out_dir": out_dir,
        "student_ckpt": skill.get("student_ckpt") or "",
        "ref_root": skill.get("ref_root") or "",
        "ckpts": ckpts,
    })


@app.post("/api/skills/<skill_id>/datasets")
def add_skill_dataset_api(skill_id: str):
    """Attach an existing dataset path (often on cluster_0) to a skill card from 04."""
    data = request.get_json(silent=True) or {}
    try:
        dataset_path = require_abs(data.get("path") or data.get("dataset_path") or "", "数据集路径")
        label = str(data.get("label") or "").strip()
        remote_path = str(data.get("remote_path") or dataset_path).strip()
        probe_first = bool(data.get("probe", True))
        ready = data.get("ready")
        total_episodes = data.get("total_episodes")
        cfg = replay_config()
        host = None
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        if probe_first:
            try:
                try:
                    host = host_by_id(host_id)
                except ValueError:
                    host = None
                probed = probe_dataset(dataset_path, cfg, host)
                total_episodes = probed.get("total_episodes", total_episodes)
                if ready is None:
                    ready = True
                if not label:
                    label = str(probed.get("name") or Path(dataset_path).name)
            except Exception as probe_exc:  # noqa: BLE001
                # Still allow linking the path; mark not-ready so UI can show the reason.
                if ready is None:
                    ready = False
                if data.get("require_probe"):
                    raise ValueError(f"数据集探查失败: {probe_exc}") from probe_exc
        skill = skill_backend.add_dataset(
            skill_id,
            path=dataset_path,
            label=label,
            remote_path=remote_path,
            dataset_id=str(data.get("id") or "").strip(),
            ready=None if ready is None else bool(ready),
            total_episodes=total_episodes,
            source=str(data.get("source") or "linked_remote"),
            root=skills_root_path(),
        )
        datasets = skill_backend.scan_skill_datasets(skill, root=skills_root_path())
        return jsonify({"ok": True, "skill": skill, "datasets": datasets})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.delete("/api/skills/<skill_id>/datasets")
def remove_skill_dataset_api(skill_id: str):
    data = request.get_json(silent=True) or {}
    try:
        skill = skill_backend.remove_dataset(
            skill_id,
            path=str(data.get("path") or data.get("dataset_path") or "").strip(),
            dataset_id=str(data.get("id") or data.get("dataset_id") or "").strip(),
            root=skills_root_path(),
        )
        datasets = skill_backend.scan_skill_datasets(skill, root=skills_root_path())
        return jsonify({"ok": True, "skill": skill, "datasets": datasets})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.delete("/api/skills/<skill_id>")
def delete_skill_api(skill_id: str):
    try:
        result = skill_backend.delete_skill(skill_id, root=skills_root_path())
        with store.lock:
            tasks = store.data.get("tasks") or []
            store.data["tasks"] = [
                t for t in tasks
                if str(t.get("skill_id") or "") != skill_id and str(t.get("name") or "") != skill_id
            ]
            store.save()
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/tasks")
def create_task():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not SAFE_NAME.fullmatch(name):
        return api_error("任务标识仅支持字母、数字、点、下划线和短横线")
    if any(t["name"] == name for t in store.data["tasks"]):
        return api_error("该任务已经注册")
    try:
        host = host_by_id(str(data.get("host_id", "")))
        base_dir = require_abs(data.get("remote_base", ""), "远端根目录")
        task_dir = base_dir if base_dir.rstrip("/").endswith("/" + name) else base_dir + "/" + name
        ok, msg = remote_mkdir(host, task_dir)
        if not ok:
            return api_error(msg or "远端文件夹创建失败")
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    item = {"id": uuid.uuid4().hex[:10], "name": name,
            "description": str(data.get("description", "")).strip(),
            "host_id": host["id"], "remote_dir": task_dir, "created_at": now_iso()}
    with store.lock:
        store.data["tasks"].append(item)
        store.save()
    warning = None
    try:
        shared_call("register_task", {"name": name, "remote_dir": task_dir,
                                      "manifest_name": manifest_basename(name),
                                      "created_at": item["created_at"], "updated_at": now_iso()}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        warning = f"任务已在本机注册，但共享索引暂未更新: {exc}"
    return jsonify({"ok": True, "task": item, "warning": warning})


@app.post("/api/collections/start")
def start_collection():
    data = request.get_json(silent=True) or {}
    try:
        skill_id = str(data.get("skill_id") or "").strip()
        if not skill_id:
            raise ValueError("请先选择技能卡")
        skill = skill_backend.load_skill(skills_root_path(), skill_id)
        if not skill:
            raise ValueError(f"技能不存在: {skill_id}")
        # Ensure older skill cards get a collect_root on first capture.
        if not str(skill.get("collect_root") or "").strip():
            skill = skill_backend.update_skill_fields(
                skill_id,
                {"collect_root": skill_backend.default_collect_root(skill_id, skills_root_path())},
                root=skills_root_path(),
            )
        task = ensure_task_for_skill(skill)
        host = host_by_id(str(task.get("host_id", "") or skills_config().get("host_id") or "cluster_0"))
        prepared = skill_backend.prepare_session_dir(
            skill,
            str(data.get("local_dir") or "").strip(),
            root=skills_root_path(),
        )
        local_dir = prepared["local_dir"]
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    item = {
        "id": "collect_" + uuid.uuid4().hex[:8], "task_id": task["id"],
        "task_name": task["name"], "skill_id": task.get("skill_id") or task["name"],
        "host_id": host["id"], "local_dir": local_dir,
        "session_name": prepared["session_name"],
        "collect_root": prepared["collect_root"],
        "launch_env": {
            "STUDIO_SKILL_ID": prepared["STUDIO_SKILL_ID"],
            "STUDIO_COLLECT_ROOT": prepared["STUDIO_COLLECT_ROOT"],
            "STUDIO_SESSION_NAME": prepared["STUDIO_SESSION_NAME"],
            "STUDIO_SESSION_DIR": prepared["STUDIO_SESSION_DIR"],
        },
        "labels_file": str(local / "labels.json"),
        "status": "collecting_offline", "message": "正在启动采集栈…",
        "created_at": now_iso(), "source_type": "offline",
        "file_count": 0, "bytes_total": 0, "episodes": [], "logs": [],
        "work_dir": "",
        "deploy_phase": "starting",
        "auto_stack": True,
    }
    # Merge optional collect_config overrides from request
    req_cfg = data.get("collect_config") if isinstance(data.get("collect_config"), dict) else {}
    cfg = collect_backend.merge_collect_config(
        {**(store.data.get("collect_config") or {}), **req_cfg}
    )
    if data.get("auto_stack") is False or data.get("scan_only"):
        cfg["auto_start_stack"] = False
    try:
        skill_backend.attach_session(
            skill_id,
            prepared["session_name"],
            local_dir=local_dir,
            remote_dir="",
            collection_id=item["id"],
            status="collecting",
            root=skills_root_path(),
        )
    except Exception as exc:  # noqa: BLE001
        return api_error(f"挂载技能会话失败: {exc}")
    with store.lock:
        store.data["collections"].insert(0, item)
        store.data["collections"] = store.data["collections"][:50]
        store.save()
    stop = threading.Event()
    stack_handles = None
    if cfg.get("auto_start_stack", True):
        def _on_line(msg: str) -> None:
            log_line(item, msg)
            # Track deploy phase file if present
            phase_path = Path(str(item.get("work_dir") or "")) / "studio_phase"
            if phase_path.is_file():
                try:
                    item["deploy_phase"] = phase_path.read_text("utf-8").strip().splitlines()[0]
                except OSError:
                    pass

        try:
            stack_handles = collect_backend.start_collect_stack(item, cfg, on_line=_on_line)
            item["work_dir"] = str(stack_handles["work_dir"])
            item["tmux_session"] = str(stack_handles.get("tmux_session") or "")
            item["deploy_phase"] = "waiting_deploy"
            item["message"] = (
                f"tmux 已启动 {item['tmux_session']}（GMR / Deploy / Exporter）· "
                "相机请在机器人侧启动 · Deploy 网页按 y → ] → Enter"
            )
            item["auto_stack"] = True
            # Re-read phase after stack ready (work_dir now set).
            phase_path = Path(item["work_dir"]) / "studio_phase"
            if phase_path.is_file():
                try:
                    item["deploy_phase"] = phase_path.read_text("utf-8").strip().splitlines()[0]
                except OSError:
                    pass
            store.save()
        except Exception as exc:  # noqa: BLE001
            item["message"] = f"采集栈启动失败: {exc}"
            item["status"] = "interrupted"
            log_line(item, item["message"])
            store.save()
            return api_error(item["message"], 500)
    else:
        item["message"] = "仅扫描本地目录（未启动采集栈）"
        item["auto_stack"] = False
        store.save()
    thread = threading.Thread(target=offline_scan_worker, args=(item["id"], stop), daemon=True)
    with worker_lock:
        workers[item["id"]] = {
            "thread": thread,
            "stop": stop,
            "proc": None,
            "kind": "offline_scan",
            "collect_stack": stack_handles,
        }
    thread.start()
    return jsonify({
        "ok": True,
        "collection": collection_payload(item),
        "session": prepared,
        "launch_env": item["launch_env"],
        "collect_config": cfg,
    })


@app.post("/api/collections/import")
def import_collection():
    data = request.get_json(silent=True) or {}
    try:
        task = task_by_id(str(data.get("task_id", "")))
        host = host_by_id(str(data.get("host_id") or task["host_id"]))
        local_dir = require_abs(data.get("local_dir", ""), "本地数据目录")
        remote_dir = require_abs(data.get("remote_dir", ""), "远端保存目录")
        validate_session_destination(task, host["id"], remote_dir)
        local = Path(local_dir)
        if not local.is_dir():
            raise ValueError(f"本地数据目录不存在: {local_dir}")
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    item = {
        "id": "import_" + uuid.uuid4().hex[:8], "task_id": task["id"], "task_name": task["name"],
        "host_id": host["id"], "local_dir": local_dir, "remote_dir": remote_dir,
        "source_type": "imported", "status": "importing", "message": "正在扫描外部数据",
        "created_at": now_iso(), "file_count": 0, "bytes_total": 0, "episodes": [], "logs": [],
    }
    try:
        refresh_collection_files(item)
    except OSError as exc:
        return api_error(f"扫描本地数据失败: {exc}")
    if not item["episodes"]:
        return api_error("该目录下没有发现可标注的 Episode 视频或数据文件")
    with store.lock:
        store.data["collections"].insert(0, item)
        store.data["collections"] = store.data["collections"][:50]
        store.save()
    stop = threading.Event()
    thread = threading.Thread(target=import_worker, args=(item["id"], stop), daemon=True)
    with worker_lock:
        workers[item["id"]] = {"thread": thread, "stop": stop, "proc": None, "kind": "import"}
    thread.start()
    return jsonify({"ok": True, "collection": collection_payload(item)})


@app.post("/api/collections/<item_id>/upload")
def upload_collection(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    if item.get("source_type") != "offline":
        return api_error("仅离线数采会话支持补传")
    if item.get("status") not in {"pending_upload", "upload_error"}:
        return api_error("请先结束离线数采并生成 labels.json")
    with worker_lock:
        if item_id in workers:
            return api_error("该会话已有后台任务运行")
    data = request.get_json(silent=True) or {}
    try:
        skill_id = str(data.get("skill_id") or item.get("skill_id") or "").strip()
        if skill_id:
            skill = skill_backend.load_skill(skills_root_path(), skill_id)
            if not skill:
                raise ValueError(f"技能不存在: {skill_id}")
            task = ensure_task_for_skill(skill)
        else:
            requested_task_id = str(data.get("task_id", ""))
            bound_task_id = str(item.get("task_id") or "")
            if bound_task_id and requested_task_id and requested_task_id != bound_task_id:
                raise ValueError(f"该 Session 已在采集前绑定任务 {item.get('task_name')}，不能改到其他任务")
            task = task_by_id(bound_task_id or requested_task_id)
            skill_id = str(task.get("skill_id") or task.get("name") or "")
        requested_host_id = str(data.get("host_id") or task.get("host_id") or "")
        if requested_host_id and requested_host_id != str(task.get("host_id")):
            # Surface: allow host from skill card / modal without hard fail when aligning.
            task = dict(task)
            task["host_id"] = requested_host_id
        host = host_by_id(str(task["host_id"]))
        local_session_name = Path(item["local_dir"]).name
        remote_dir = str(data.get("remote_dir") or "").strip()
        if not remote_dir:
            remote_dir = f"{str(task['remote_dir']).rstrip('/')}/{local_session_name}"
        remote_dir = require_abs(remote_dir, "远端 session 目录")
        task_root = str(task["remote_dir"]).rstrip("/")
        if remote_dir == task_root or not remote_dir.startswith(task_root + "/"):
            raise ValueError(f"远端 session 必须位于任务目录下: {task_root}/<session>")
        if Path(remote_dir).name != local_session_name:
            raise ValueError(f"远端 Session 目录名必须与本地一致: {local_session_name}")
        duplicate = next((other for other in store.data["collections"]
                          if other.get("id") != item_id
                          and other.get("host_id") == host["id"]
                          and other.get("remote_dir") == remote_dir), None)
        if duplicate:
            raise ValueError(f"远端 session 路径已被 {duplicate['id']} 使用")
        # Surface: attach into skill folder when the skill card exists.
        if skill_id and skill_backend.load_skill(skills_root_path(), skill_id):
            skill_backend.attach_session(
                skill_id,
                local_session_name,
                local_dir=item.get("local_dir") or "",
                remote_dir=remote_dir,
                collection_id=item_id,
                status="uploading",
                root=skills_root_path(),
            )
        try:
            remote_manifest_entry(item)
        except Exception:
            pass  # surface flow: skill JSON is enough for now
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    item.update({
        "task_id": task["id"], "task_name": task["name"], "skill_id": skill_id or task.get("skill_id"),
        "host_id": host["id"], "remote_dir": remote_dir,
        "status": "uploading", "message": "正在准备补传",
    })
    store.save()
    thread = threading.Thread(target=deferred_upload_worker, args=(item_id,), daemon=True)
    with worker_lock:
        workers[item_id] = {"thread": thread, "stop": None, "proc": None,
                            "kind": "deferred_upload"}
    thread.start()
    return jsonify({"ok": True, "collection": collection_payload(item)})


@app.post("/api/collections/<item_id>/stop")
def stop_collection(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    worker = None
    with worker_lock:
        worker = workers.get(item_id)
        if worker:
            if worker.get("stop") is not None:
                worker["stop"].set()
            proc = worker.get("proc")
            if proc and proc.poll() is None:
                proc.terminate()
            stack = worker.get("collect_stack")
            if stack:
                try:
                    collect_backend.stop_collect_stack(stack, kill_deploy=True)
                except Exception as exc:  # noqa: BLE001
                    log_line(item, f"[studio] stop stack: {exc}")
                worker["collect_stack"] = None
    if item.get("source_type") == "offline":
        item["status"] = "stopping_offline"
        item["message"] = "正在结束采集栈与本地扫描并生成 labels.json"
        item["deploy_phase"] = "stopped"
        store.save()
        if not worker:
            try:
                finalize_offline_collection(item)
            except Exception as exc:  # noqa: BLE001
                item["status"] = "interrupted"
                item["message"] = f"生成本地 labels.json 失败: {exc}"
                store.save()
                return api_error(item["message"], 500)
        return jsonify({"ok": True})
    if item.get("source_type") == "imported":
        item["status"] = "stopping"
        item["message"] = "正在取消外部数据上传"
    else:
        item["status"] = "stopping"
        item["message"] = "正在结束数采并生成 JSON"
    store.save()
    return jsonify({"ok": True})


@app.get("/api/collect/config")
def get_collect_config():
    return jsonify({"ok": True, "config": collect_config()})


@app.post("/api/collect/config")
def set_collect_config():
    data = request.get_json(silent=True) or {}
    cfg = collect_backend.merge_collect_config({**(store.data.get("collect_config") or {}), **data})
    # Keep derived paths coherent when roots change
    groot = Path(str(cfg.get("groot_root") or ""))
    if groot.is_dir():
        if not data.get("data_venv"):
            cfg["data_venv"] = str(groot / ".venv_data_collection")
        if not data.get("deploy_root"):
            cfg["deploy_root"] = str(groot / "gear_sonic_deploy")
        xrt = groot / "external_dependencies" / "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
        if not data.get("xrt_pybind") and xrt.is_dir():
            cfg["xrt_pybind"] = str(xrt)
    cam_repo = Path(str(cfg.get("camera_repo") or ""))
    if cam_repo.is_dir() and not data.get("camera_venv"):
        cfg["camera_venv"] = str(cam_repo / ".venv_sim")
    with store.lock:
        store.data["collect_config"] = cfg
        store.save()
    return jsonify({"ok": True, "config": cfg})


@app.post("/api/collections/<item_id>/record-cmd")
def collection_record_cmd(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    data = request.get_json(silent=True) or {}
    cmd = str(data.get("cmd") or data.get("key") or "").strip().lower()
    try:
        port = int(collect_config().get("keyboard_zmq_port") or 5580)
        pub = collect_backend.get_record_publisher(port)
        pub.send(cmd)
        labels = {"t": "开始录制", "y": "停止并保存", "x": "丢弃当前条"}
        log_line(item, f"[record] sent '{cmd}' → {labels.get(cmd, cmd)}")
        store.save()
        return jsonify({"ok": True, "cmd": cmd, "message": labels.get(cmd, cmd)})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/collections/<item_id>/deploy-cmd")
def collection_deploy_cmd(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    data = request.get_json(silent=True) or {}
    cmd = str(data.get("cmd") or data.get("action") or "").strip()
    work = str(item.get("work_dir") or "").strip()
    tmux_session = str(item.get("tmux_session") or "")
    with worker_lock:
        stack = (workers.get(item_id) or {}).get("collect_stack") or {}
        if not work and stack.get("work_dir"):
            work = str(stack.get("work_dir") or "")
        if not tmux_session:
            tmux_session = str(stack.get("tmux_session") or "")
    if not work:
        return api_error("采集工作目录尚未就绪")
    try:
        # Re-ensure work dir (exporter may have wiped nested .studio_collect).
        ensured = collect_backend.work_dir_for(item)
        if str(ensured) != work:
            work = str(ensured)
            item["work_dir"] = work
            with worker_lock:
                stack = workers.get(item_id) or {}
                if stack.get("collect_stack"):
                    stack["collect_stack"]["work_dir"] = work
        path = collect_backend.write_deploy_cmd(Path(work), cmd, tmux_session=tmux_session)
        # Track stand/stream for UI even when log markers are ambiguous.
        raw = cmd.lower().strip()
        if raw in {"]", "stand", "stand_up"}:
            item["deploy_phase"] = "standing"
            try:
                (Path(work) / "studio_phase").write_text("standing\n", encoding="utf-8")
            except OSError:
                pass
        elif raw in {"enter", "stream", ""}:
            # Keep init_done until ZMQ ENABLED is seen; still hint UI.
            if str(item.get("deploy_phase") or "") in {"init_done", "standing", ""}:
                item["deploy_phase"] = "standing"
        log_line(item, f"[deploy-cmd] {cmd} → tmux:{tmux_session or '-'} {path}")
        store.save()
        return jsonify({"ok": True, "cmd": cmd, "tmux_session": tmux_session, "phase": item.get("deploy_phase")})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/collections/<item_id>/collect-logs")
def collection_collect_logs(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    work = str(item.get("work_dir") or "").strip()
    with worker_lock:
        stack = (workers.get(item_id) or {}).get("collect_stack")
        status = collect_backend.stack_status(stack)
        if not work and stack:
            work = str(stack.get("work_dir") or "")
    text = collect_backend.read_collect_logs(work) if work else ""
    # Prefer live stack log; fall back to item.logs tail
    if not text:
        text = "\n".join((item.get("logs") or [])[-200:])
    phase = ""
    if work:
        try:
            phase = collect_backend.infer_deploy_phase(work)
            item["deploy_phase"] = phase
        except Exception:  # noqa: BLE001
            phase = str(item.get("deploy_phase") or "")
    else:
        phase = str(item.get("deploy_phase") or "")
    return jsonify({
        "ok": True,
        "logs": text,
        "phase": phase,
        "stack": status,
        "work_dir": work,
    })


@app.post("/api/collect/kill-deploy")
def collect_kill_deploy():
    msg = collect_backend.kill_g1_deploy()
    return jsonify({"ok": True, "message": msg or '已执行 pkill -9 -f "g1_deploy"'})


def _active_collect_camera_work() -> str:
    work = ""
    with worker_lock:
        for _item_id, worker in workers.items():
            stack = worker.get("collect_stack") or {}
            wd = stack.get("work_dir")
            if wd:
                work = str(wd)
                break
    if not work:
        for item in store.data.get("collections") or []:
            if item.get("status") == "collecting_offline" and item.get("work_dir"):
                work = str(item["work_dir"])
                break
    return work


@app.get("/api/collect/camera/snapshot/<slot>")
def collect_camera_snapshot(slot: str):
    """Single JPEG frame — prefer this over MJPEG to avoid browser tab OOM/crashes."""
    work = _active_collect_camera_work()
    if not work:
        return api_error("没有运行中的相机代理", 404)
    jpeg, _name = collect_backend.read_camera_jpeg(work, slot)
    if not jpeg:
        return api_error("暂无画面", 404)
    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/collect/camera/stream/<slot>")
def collect_camera_stream(slot: str):
    """Legacy MJPEG stream (kept for compatibility; Studio UI uses snapshot polling)."""
    work = _active_collect_camera_work()
    if not work:
        return api_error("没有运行中的相机代理", 404)

    boundary = b"frame"

    def generate():
        last = b""
        idle = 0
        while True:
            jpeg, name = collect_backend.read_camera_jpeg(work, slot)
            if jpeg and jpeg != last:
                last = jpeg
                idle = 0
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            else:
                idle += 1
                if idle > 300:  # ~30s with no frames
                    break
            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        mimetype=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/collect/camera/meta")
def collect_camera_meta():
    work = ""
    with worker_lock:
        for _item_id, worker in workers.items():
            stack = worker.get("collect_stack") or {}
            if stack.get("work_dir"):
                work = str(stack["work_dir"])
                break
    keys = []
    if work:
        meta = Path(work) / "camera" / "cameras.json"
        if meta.is_file():
            try:
                keys = list(json.loads(meta.read_text("utf-8")).get("keys") or [])
            except Exception:  # noqa: BLE001
                keys = []
    return jsonify({"ok": True, "work_dir": work, "keys": keys})


@app.post("/api/collections/<item_id>/refresh")
def refresh_collection(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    with store.lock:
        refresh_collection_files(item)
        if item.get("source_type") == "offline":
            write_local_labels(item)
        store.save()
    return jsonify({"ok": True, "collection": collection_payload(item)})


@app.get("/api/collections/<item_id>")
def get_collection(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    return jsonify({"ok": True, "collection": item})


@app.delete("/api/collections/<item_id>")
def delete_collection(item_id: str):
    """Remove a local collection from Studio (+ optional skill detach / disk wipe)."""
    import shutil

    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    data = request.get_json(silent=True) or {}
    delete_files = bool(data.get("delete_files") or data.get("wipe") or False)
    skill_id = str(item.get("skill_id") or data.get("skill_id") or "").strip()
    local_dir = str(item.get("local_dir") or "").strip()
    session_name = str(item.get("session_name") or Path(local_dir).name if local_dir else "").strip()

    # Stop live workers / tmux if still attached.
    with worker_lock:
        worker = workers.pop(item_id, None)
    if worker:
        try:
            if worker.get("stop") is not None:
                worker["stop"].set()
            proc = worker.get("proc")
            if proc and proc.poll() is None:
                proc.terminate()
            stack = worker.get("collect_stack")
            if stack:
                collect_backend.stop_collect_stack(stack, kill_deploy=False)
        except Exception:  # noqa: BLE001
            pass

    with store.lock:
        store.data["collections"] = [
            c for c in (store.data.get("collections") or []) if c.get("id") != item_id
        ]
        store.save()

    detached = False
    if skill_id:
        try:
            skill_backend.remove_dataset(
                skill_id,
                path=local_dir,
                dataset_id=session_name,
                root=skills_root_path(),
            )
            detached = True
        except Exception:  # noqa: BLE001
            # Also try by collection_id match via path only already attempted.
            try:
                skill = skill_backend.load_skill(skills_root_path(), skill_id)
                if skill:
                    keep = []
                    for d in skill.get("datasets") or []:
                        if str(d.get("collection_id") or "") == item_id:
                            detached = True
                            continue
                        keep.append(d)
                    if detached:
                        skill_backend.update_skill_fields(
                            skill_id, {"datasets": keep}, root=skills_root_path()
                        )
            except Exception:  # noqa: BLE001
                pass

    wiped = False
    wipe_path = ""
    if delete_files and local_dir:
        path = Path(local_dir).resolve()
        # Only allow wiping under known local collect roots (never remote /mnt blindly).
        allowed_roots: list[Path] = []
        cfg = collect_config()
        groot = str(cfg.get("groot_root") or "").strip()
        if groot:
            allowed_roots.append((Path(groot) / "outputs").resolve())
        allowed_roots.append(
            Path("/home/user/DataCollection/GR00T-WholeBodyControl/outputs").resolve()
        )
        skill = skill_backend.load_skill(skills_root_path(), skill_id) if skill_id else None
        if skill and skill.get("collect_root"):
            allowed_roots.append(Path(str(skill["collect_root"])).resolve())
        # Dedup
        seen: set[str] = set()
        uniq: list[Path] = []
        for r in allowed_roots:
            key = str(r)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        ok = any(path != r and r in path.parents for r in uniq)
        # Block deleting collect root / drive root itself
        if ok and path.is_dir() and path.name and path != path.anchor:
            try:
                shutil.rmtree(path)
                wiped = True
                wipe_path = str(path)
            except OSError as exc:
                return api_error(f"已从列表移除，但删除磁盘失败: {exc}", 500)
            # companion studio control dir
            collect_root = path.parent
            studio_dir = collect_root / ".studio_collect" / path.name
            if studio_dir.is_dir():
                try:
                    shutil.rmtree(studio_dir)
                except OSError:
                    pass
        elif delete_files:
            return jsonify({
                "ok": True,
                "removed": True,
                "detached": detached,
                "wiped": False,
                "warning": f"已从 Studio 移除，但路径不在允许删除的本机采集根下，未删磁盘: {local_dir}",
            })

    return jsonify({
        "ok": True,
        "removed": True,
        "detached": detached,
        "wiped": wiped,
        "wipe_path": wipe_path,
        "id": item_id,
    })


@app.post("/api/collections/<item_id>/mark")
def mark_episode(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    data = request.get_json(silent=True) or {}
    status = str(data.get("status", ""))
    if status not in {"valid", "invalid", "unreviewed"}:
        return api_error("标注状态无效")
    if item.get("source_type") == "imported":
        return api_error("外部导入必须等待上传并发布完成，再从共享会话中打标")
    with store.lock:
        episode = next((x for x in item.get("episodes", [])
                        if x["id"] == data.get("episode_id")), None)
        if not episode:
            return api_error("episode 不存在", 404)
        old_status = episode.get("status", "unreviewed")
        episode["status"] = status
        episode.pop("reason", None)
        if item.get("source_type") == "offline":
            try:
                write_local_labels(item)
                store.save()
            except Exception as exc:  # noqa: BLE001
                episode["status"] = old_status
                return api_error(f"本地标注文件写入失败: {exc}", 500)
            return jsonify({"ok": True, "episode": episode})
        store.save()
    try:
        write_manifest(item, upload=True)
    except Exception as exc:  # noqa: BLE001
        return api_error(f"标注已保存在本地，但远端 JSON 更新失败: {exc}", 502)
    return jsonify({"ok": True, "episode": episode})


@app.get("/api/collections/<item_id>/manifest")
def get_manifest(item_id: str):
    item = store.find("collections", item_id)
    if not item:
        return api_error("数采记录不存在", 404)
    if item.get("source_type") == "offline":
        manifest_path = item.get("labels_file")
        if not manifest_path or not Path(manifest_path).is_file():
            manifest_path = write_local_labels(item)
            store.save()
        try:
            manifest = json.loads(Path(manifest_path).read_text("utf-8"))
        except (OSError, ValueError) as exc:
            return api_error(f"读取本地 labels.json 失败: {exc}", 500)
        return jsonify({"ok": True, "filename": "labels.json",
                        "path": manifest_path, "manifest": manifest})
    manifest_path = item.get("manifest_file")
    if not manifest_path or not Path(manifest_path).is_file():
        manifest_path = write_manifest(item, upload=False)
    try:
        manifest = json.loads(Path(manifest_path).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        return api_error(f"读取本地清单失败: {exc}", 500)
    return jsonify({"ok": True, "filename": Path(manifest_path).name,
                    "path": manifest_path, "manifest": manifest})


@app.get("/api/media/<item_id>/<episode_id>/<int:view_index>")
def media(item_id: str, episode_id: str, view_index: int):
    item = store.find("collections", item_id)
    if not item:
        return "not found", 404
    episode = next((x for x in item.get("episodes", []) if x["id"] == episode_id), None)
    if not episode or not episode.get("media"):
        return "not found", 404
    videos = episode.get("videos") or []
    if videos:
        if view_index < 0 or view_index >= len(videos):
            return "not found", 404
        relative_path = videos[view_index]["relative_path"]
    elif view_index == 0:
        relative_path = episode["relative_path"]
    else:
        return "not found", 404
    root = Path(item["local_dir"]).resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file():
        return "not found", 404
    return send_file(path, conditional=True)


@app.get("/api/media/<item_id>/<episode_id>")
def media_default(item_id: str, episode_id: str):
    return media(item_id, episode_id, 0)


REMOTE_PROBE = r'''
import json, os, sys
root=sys.argv[1]
files=[]
for base, _, names in os.walk(root):
  for n in names:
    if os.path.splitext(n)[1].lower() in ('.h5','.hdf5','.npz','.parquet','.json'):
      files.append(os.path.join(base,n))
  if len(files)>=20: break
fields={}
for p in files[:20]:
  ext=os.path.splitext(p)[1].lower()
  try:
    if ext in ('.h5','.hdf5'):
      import h5py
      with h5py.File(p,'r') as f:
        def visit(k,v):
          if hasattr(v,'shape'): fields.setdefault(k, list(v.shape))
        f.visititems(visit)
    elif ext=='.npz':
      import numpy as np
      with np.load(p) as f:
        for k in f.files: fields.setdefault(k,list(f[k].shape))
    elif ext=='.parquet':
      import pyarrow.parquet as pq
      for x in pq.read_schema(p).names: fields.setdefault(x,[])
    elif ext=='.json':
      with open(p) as f: x=json.load(f)
      if isinstance(x,dict):
        for k,v in x.items(): fields.setdefault(k,[len(v)] if isinstance(v,list) else [])
  except Exception: pass
print(json.dumps({'files':files,'fields':[{'name':k,'shape':v} for k,v in sorted(fields.items())]}))
'''


@app.post("/api/fields/probe")
def probe_fields():
    data = request.get_json(silent=True) or {}
    try:
        host = host_by_id(str(data.get("host_id", "")))
        path = require_abs(data.get("input_dir", ""), "原始数据目录")
        proc = run_command(ssh_args(host["target"], ["python3", "-c", REMOTE_PROBE, path]), timeout=90)
        if proc.returncode:
            return api_error((proc.stderr or proc.stdout).strip() or "字段探查失败")
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/conversions")
def create_conversion():
    data = request.get_json(silent=True) or {}
    try:
        host = host_by_id(str(data.get("host_id", "")))
        input_dir = require_abs(data.get("input_dir", ""), "原始数据目录")
        output_dir = require_abs(data.get("output_dir", ""), "输出目录")
        script = require_abs(data.get("script", ""), "转换脚本")
        manifest = require_abs(data.get("manifest", ""), "valid/invalid 聚合清单")
        mappings = data.get("mappings") or []
        if not mappings:
            raise ValueError("至少添加一个字段映射")
        occupied = [False] * 512
        clean = []
        for row in mappings:
            source = str(row.get("source", "")).strip()
            start, end = int(row.get("start")), int(row.get("end"))
            if not source or start < 0 or end <= start or end > 512:
                raise ValueError("字段映射必须满足 0 ≤ start < end ≤ 512")
            if any(occupied[start:end]):
                raise ValueError(f"字段 {source} 的目标区间与其他字段重叠")
            occupied[start:end] = [True] * (end - start)
            clean.append({"source": source, "target_start": start, "target_end": end})
        extra = shlex.split(str(data.get("extra_args", "")))
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    item = {
        "id": "convert_" + uuid.uuid4().hex[:8], "host_id": host["id"], "input_dir": input_dir,
        "output_dir": output_dir, "script": script, "mappings": clean, "extra_args": extra,
        "manifest": manifest,
        "status": "running", "message": "准备远端转换", "created_at": now_iso(), "logs": [],
    }
    with store.lock:
        store.data["conversions"].insert(0, item)
        store.data["conversions"] = store.data["conversions"][:50]
        store.save()
    threading.Thread(target=conversion_worker, args=(item["id"],), daemon=True).start()
    return jsonify({"ok": True, "conversion": item})


@app.get("/api/conversions/<item_id>")
def get_conversion(item_id: str):
    item = store.find("conversions", item_id)
    if not item:
        return api_error("转换任务不存在", 404)
    return jsonify({"ok": True, "conversion": item})


@app.post("/api/conversions/<item_id>/cancel")
def cancel_conversion(item_id: str):
    item = store.find("conversions", item_id)
    if not item:
        return api_error("转换任务不存在", 404)
    with worker_lock:
        worker = workers.get(item_id)
        proc = worker.get("proc") if worker else None
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    item["status"], item["message"] = "cancelled", "已取消转换"
    store.save()
    return jsonify({"ok": True})


def replay_config() -> dict[str, Any]:
    cfg = default_state()["replay_config"] | (store.data.get("replay_config") or {})
    # Prefer cluster MixCorpus paths when remote.
    if str(cfg.get("execution_mode") or "remote") == "remote":
        cfg.setdefault("qa_root", DEFAULT_QA_ROOT_CLUSTER)
        cfg.setdefault("dataset_base", DEFAULT_DATASET_BASE_CLUSTER)
        if not cfg.get("qa_root") or str(cfg.get("qa_root")).startswith("/home/"):
            cfg["qa_root"] = DEFAULT_QA_ROOT_CLUSTER
        if not cfg.get("dataset_base") or str(cfg.get("dataset_base")).startswith("/home/"):
            cfg["dataset_base"] = DEFAULT_DATASET_BASE_CLUSTER
    return cfg


REMOTE_PROBE_DATASET = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise FileNotFoundError(f"不是数据集根目录（缺少 meta/info.json）: {root}")
info = json.loads(info_path.read_text())
eps, lengths, tasks = [], [], []
ver = str(info.get("codebase_version") or "")
fmt = ""

parquet_ep = root / "meta/episodes/chunk-000/file-000.parquet"
jsonl_ep = root / "meta/episodes.jsonl"
# 02 筛选默认优先 V2.1：有 episodes.jsonl 且未显式声明 v3 时走 v2.1
prefer_v21 = ver.startswith("v2") or (jsonl_ep.is_file() and not ver.startswith("v3"))
if prefer_v21:
    if not jsonl_ep.is_file():
        raise FileNotFoundError(f"声明为 v2.x 但缺少 episode meta: {jsonl_ep}")
    with jsonl_ep.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eps.append(int(row.get("episode_index", len(eps))))
            lengths.append(int(row.get("length") or row.get("num_frames") or 0))
            row_tasks = row.get("tasks") or []
            if isinstance(row_tasks, list):
                for t in row_tasks:
                    if t and str(t) not in tasks:
                        tasks.append(str(t))
            elif row_tasks and str(row_tasks) not in tasks:
                tasks.append(str(row_tasks))
    tasks_jsonl = root / "meta/tasks.jsonl"
    if tasks_jsonl.is_file():
        with tasks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("task") and str(row["task"]) not in tasks:
                    tasks.append(str(row["task"]))
    fmt = "v2.1"
elif parquet_ep.is_file() or ver.startswith("v3"):
    if not parquet_ep.is_file():
        raise FileNotFoundError(f"声明为 v3 但缺少 episode meta: {parquet_ep}")
    import pyarrow.parquet as pq
    ep_meta = pq.read_table(parquet_ep)
    eps = [int(x) for x in ep_meta.column("episode_index").to_pylist()]
    lengths = [int(x) for x in ep_meta.column("length").to_pylist()]
    try:
        t = pq.read_table(root / "meta/tasks.parquet")
        tasks = t.column("task").to_pylist() if "task" in t.column_names else []
    except Exception:
        pass
    fmt = "v3"
else:
    raise FileNotFoundError(
        "未找到 episode 元数据：需要 meta/episodes.jsonl（LeRobot V2.1，02 筛选默认）"
        " 或 meta/episodes/chunk-000/file-000.parquet（V3）"
    )

total_eps = int(info.get("total_episodes") or len(eps) or 0)
if not eps and total_eps:
    eps = list(range(total_eps))
    lengths = [0] * total_eps

# Sonic QA：优先 V2.1 raw（episode_* + action.motion_token）；亦兼容旧 V3 unified
data_dir = root / "data" / "chunk-000"
file_pars = sorted(data_dir.glob("file-*.parquet")) if data_dir.is_dir() else []
ep_pars = sorted(data_dir.glob("episode_*.parquet")) if data_dir.is_dir() else []
# 部分 V2.1 把 episode parquet 放在 data/ 根目录
if not ep_pars and (root / "data").is_dir():
    ep_pars = sorted((root / "data").glob("episode_*.parquet"))
sample = ep_pars[0] if ep_pars else (file_pars[0] if file_pars else None)
cols = []
if sample is not None:
    try:
        import pyarrow.parquet as pq
        cols = list(pq.read_schema(sample).names)
    except Exception:
        cols = []
has_unified = "action.unified" in cols
has_motion = "action.motion_token" in cols
# V2.1 采集布局：EpisodeZStore.format=raw，读 motion_token64
if ep_pars and has_motion:
    sonic = {
        "ready": True,
        "layout": "raw_v21",
        "data_naming": "episode_*",
        "screening_ready": True,
        "message": "LeRobot V2.1（action.motion_token）：可做视频/人工筛选；也可快速 Sonic 回放。正式 Isaac 终极筛与 03 训练请用 unified REF",
    }
elif fmt == "v2.1" and ep_pars and not has_motion:
    sonic = {
        "ready": False,
        "layout": "raw_v21_incomplete",
        "data_naming": "episode_*",
        "screening_ready": True,
        "message": "LeRobot V2.1 有 episode_* 但缺少 action.motion_token，无法 Sonic 回放（仍可筛选）",
    }
elif file_pars and has_unified:
    sonic = {
        "ready": True,
        "layout": "unified_v3",
        "data_naming": "file-*",
        "screening_ready": True,
        "message": "V3 unified：可筛选，亦可 Sonic QA 仿真回放（兼容旧数据）",
    }
elif file_pars and not has_unified:
    sonic = {
        "ready": False,
        "layout": "v3_non_unified",
        "data_naming": "file-*",
        "screening_ready": True,
        "message": "找到 file-*.parquet 但缺少 action.unified；仍可筛选，无法 Sonic 仿真回放",
    }
elif fmt == "v2.1":
    sonic = {
        "ready": False,
        "layout": "raw_v21",
        "data_naming": "v2.1-meta",
        "screening_ready": True,
        "message": "LeRobot V2.1 meta 齐全但缺 data/episode_*.parquet；可做索引级筛选",
    }
else:
    sonic = {
        "ready": False,
        "layout": "unknown",
        "data_naming": "",
        "screening_ready": bool(total_eps),
        "message": "未找到 data 下 parquet；若 meta 齐全仍可做索引级筛选",
    }

print(json.dumps({
    "ok": True,
    "name": root.name,
    "path": str(root),
    "format": fmt,
    "codebase_version": ver or fmt,
    "total_episodes": total_eps,
    "total_frames": int(info.get("total_frames", 0) or 0),
    "fps": float(info.get("fps", 50) or 50),
    "robot_type": info.get("robot_type", "") or "",
    "task_prompt": tasks[0] if tasks else "",
    "sonic_replay": sonic,
    "episodes": [{"index": int(e), "length": int(l)} for e, l in zip(eps, lengths)],
}))
'''


def use_local_execution(cfg: dict[str, Any], dataset_path: str) -> bool:
    mode = str(cfg.get("execution_mode") or "auto").lower()
    if mode == "remote":
        return False
    if mode == "local":
        return True
    # auto: local only when the path exists on this machine
    return Path(dataset_path).expanduser().is_dir()


def probe_dataset_at(path: str) -> dict[str, Any]:
    proc = run_command(["python3", "-c", REMOTE_PROBE_DATASET, path], timeout=60)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "数据集探查失败")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def probe_dataset(path: str, cfg: dict[str, Any], host: dict[str, Any] | None) -> dict[str, Any]:
    if use_local_execution(cfg, path):
        return probe_dataset_at(path)
    if not host:
        raise ValueError("远端模式需要有效 Host")
    proc = run_command(ssh_args(host["target"], ["python3", "-c", REMOTE_PROBE_DATASET, path]), timeout=60)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "数据集探查失败")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def read_json_at(path: str, cfg: dict[str, Any], host: dict[str, Any] | None) -> dict[str, Any] | None:
    local_path = Path(path).expanduser()
    if local_path.is_file():
        try:
            return json.loads(local_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            return None
    if host:
        return remote_read_json(host, path)
    return None


def discover_dataset_presets(cfg: dict[str, Any]) -> list[dict[str, str]]:
    presets: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, path: str) -> None:
        if path in seen:
            return
        seen.add(path)
        presets.append({"label": label, "path": path})

    # Local bases (optional)
    for base in (str(LOCAL_DATAPROCESS_ROOT),):
        root = Path(base).expanduser()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").is_file():
                add(f"本机 {child.name}", str(child.resolve()))

    # Remote MixCorpus/830 — primary for cloud replay/train
    remote_base = str(cfg.get("dataset_base") or DEFAULT_DATASET_BASE_CLUSTER)
    try:
        host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
        py = (
            "import json,os,sys\n"
            "from pathlib import Path\n"
            f"root=Path({remote_base!r})\n"
            "out=[]\n"
            "if root.is_dir():\n"
            "  for c in sorted(root.iterdir()):\n"
            "    if c.is_dir() and (c/'meta'/'info.json').is_file():\n"
            "      out.append({'label':c.name,'path':str(c)})\n"
            "print(json.dumps(out))\n"
        )
        proc = run_command(ssh_args(host["target"], ["python3", "-c", py]), timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            rows = []
            for row in json.loads(proc.stdout.strip().splitlines()[-1]):
                if isinstance(row, dict) and row.get("path"):
                    rows.append(row)
            # 02 默认 V2.1：非 *_unified 排前面
            rows.sort(key=lambda r: (
                "unified" in str(r.get("label") or r.get("path") or "").lower(),
                str(r.get("label") or ""),
            ))
            for row in rows:
                add(f"云端 {row.get('label') or Path(row['path']).name}", str(row["path"]))
    except Exception:  # noqa: BLE001
        # Fall back：优先 V2.1 会话路径，其次 unified
        for name, path in (
            ("skill_2_pico_pick_the_toy (V2.1)", "/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy/2026-09-04-00-41-34"),
            ("skill_2_pico_new826_unified (V3)", f"{DEFAULT_DATASET_BASE_CLUSTER}/skill_2_pico_new826_unified"),
            ("skill_walk_to_black_box_new_unified (V3)", f"{DEFAULT_DATASET_BASE_CLUSTER}/skill_walk_to_black_box_new_unified"),
        ):
            add(f"云端 {name}", path)
    return presets


def list_replay_experiments(
    qa_root: str,
    dataset_name: str,
    *,
    cfg: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or replay_config()
    if use_local_execution(cfg, qa_root) and Path(qa_root).expanduser().is_dir():
        exp_dir = Path(qa_root).expanduser() / "experiments"
        if not exp_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for child in exp_dir.iterdir():
            if not child.is_dir() or not child.name.startswith(dataset_name):
                continue
            summary_path = child / "batch_replay_summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text("utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            rows.append({
                "name": child.name,
                "path": str(child.resolve()),
                "summary": summary,
                "n_ok": summary.get("n_ok", len(summary.get("ok") or [])),
                "n_fail": summary.get("n_fail", len(summary.get("fail") or [])),
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(summary_path.stat().st_mtime)),
            })
        rows.sort(key=lambda x: x["mtime"], reverse=True)
        return rows

    # Remote experiments listing
    if host is None:
        try:
            host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
        except ValueError:
            return []
    py = (
        "import json,os,time\n"
        "from pathlib import Path\n"
        f"exp=Path({qa_root!r})/'experiments'\n"
        f"name={dataset_name!r}\n"
        "rows=[]\n"
        "if exp.is_dir():\n"
        "  for child in exp.iterdir():\n"
        "    if not child.is_dir() or not child.name.startswith(name): continue\n"
        "    sp=child/'batch_replay_summary.json'\n"
        "    if not sp.is_file(): continue\n"
        "    try:\n"
        "      summary=json.loads(sp.read_text())\n"
        "    except Exception:\n"
        "      continue\n"
        "    rows.append({'name':child.name,'path':str(child),'summary':summary,"
        "'n_ok':summary.get('n_ok',len(summary.get('ok') or [])),"
        "'n_fail':summary.get('n_fail',len(summary.get('fail') or [])),"
        "'mtime':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(sp.stat().st_mtime))})\n"
        "rows.sort(key=lambda x:x['mtime'], reverse=True)\n"
        "print(json.dumps(rows))\n"
    )
    proc = run_command(ssh_args(host["target"], ["python3", "-c", py]), timeout=60)
    if proc.returncode or not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        return data if isinstance(data, list) else []
    except ValueError:
        return []


def build_replay_env(item: dict[str, Any]) -> dict[str, str]:
    env = {
        "DATASET": item["dataset_name"],
        "DATA_ROOT": item["dataset_path"],
        "EPISODES": item["episodes"],
        "LIVE_UI": str(item.get("live_ui", 1)),
        "VISER_PORT": str(item.get("viser_port", 8081)),
        "LOOP": str(item.get("loop", 1)),
        "LOOP_PAUSE": str(item.get("loop_pause", 1)),
        "OUT": item["out_dir"],
    }
    if item.get("cuda_device") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(item["cuda_device"])
    return env


def issues_for_dataset(dataset_path: str) -> dict[str, Any]:
    with store.lock:
        all_issues = store.data.setdefault("replay_issues", {})
        entry = all_issues.setdefault(dataset_path, {
            "issues": {}, "marked_delete": [], "prune_status": "pending", "prune_message": "",
            "updated_at": None,
        })
        return entry


def parse_episode_selection(text: str, total: int) -> list[int]:
    text = str(text or "").strip()
    if not text:
        return list(range(min(20, total)))
    eps: set[int] = set()
    for part in re.split(r"[\s,]+", text):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"区间无效: {part}")
            eps.update(range(start, end + 1))
        else:
            eps.add(int(part))
    bad = sorted(x for x in eps if x < 0 or x >= total)
    if bad:
        raise ValueError(f"Episode 超出范围 0..{total - 1}: {bad[:8]}")
    return sorted(eps)


def translate_replay_fail(entry: dict[str, Any]) -> str:
    reasons = entry.get("reasons") or []
    parts = [REPLAY_REASON_ZH.get(r, r) for r in reasons]
    ep = entry.get("episode")
    detail = []
    if entry.get("fall_step", -1) >= 0:
        detail.append(f"摔倒步数 {entry['fall_step']}")
    tw = entry.get("twitch") or {}
    if tw.get("worst_joint", -1) >= 0:
        detail.append(f"抖动关节 {tw['worst_joint']}")
    head = f"Episode {ep}：" + ("；".join(parts) if parts else "自动检测异常")
    if detail:
        head += "（" + "，".join(detail) + "）"
    return head


def remote_read_json(host: dict[str, Any], path: str) -> dict[str, Any] | None:
    proc = run_command(ssh_args(host["target"], ["python3", "-c", "import json,sys;print(open(sys.argv[1]).read())", path]),
                       timeout=20)
    if proc.returncode:
        return None
    try:
        return json.loads(proc.stdout.strip())
    except (ValueError, TypeError):
        return None


def merge_replay_summary(dataset_path: str, summary: dict[str, Any]) -> None:
    """Union auto-detected failures; keep historical fails even if a later run passes."""
    entry = issues_for_dataset(dataset_path)
    for fail in summary.get("fail") or []:
        reasons = [r for r in (fail.get("reasons") or []) if r != "data_twitch"]
        # Disk latent pre-scan alone is not a QA failure — skip / strip it.
        if not reasons and not fail.get("fell"):
            continue
        fail = {**fail, "reasons": reasons}
        ep = str(fail.get("episode"))
        current = entry["issues"].get(ep) or {}
        note = str(current.get("note") or "")
        if current.get("source") == "manual":
            entry["issues"][ep] = {
                **current,
                "reasons": sorted(set((current.get("reasons") or []) + reasons)),
                "status": "fail",
            }
            continue
        entry["issues"][ep] = {
            "episode": int(fail.get("episode", ep)),
            "reasons": reasons,
            "summary_zh": translate_replay_fail(fail),
            "note": note,
            "source": "auto",
            "status": "fail",
        }
    entry["updated_at"] = now_iso()
    store.save()


def reset_replay_session_state(dataset_path: str) -> dict[str, Any]:
    """Clear transient QA UI state but keep accumulated failure records."""
    entry = issues_for_dataset(dataset_path)
    entry["marked_delete"] = []
    entry["prune_status"] = "pending"
    entry["prune_message"] = ""
    entry["updated_at"] = now_iso()
    store.save()
    return entry


def clear_replay_issues_log(dataset_path: str) -> dict[str, Any]:
    """Clear all failure records and transient QA state for a dataset."""
    entry = issues_for_dataset(dataset_path)
    entry["issues"] = {}
    entry["marked_delete"] = []
    entry["prune_status"] = "pending"
    entry["prune_message"] = ""
    entry["updated_at"] = now_iso()
    store.save()
    return entry


def replay_worker(job_id: str) -> None:
    item = store.find("replay_jobs", job_id)
    if not item or item.get("kind") != "replay":
        return
    cfg = replay_config()
    dataset_path = item["dataset_path"].rstrip("/")
    host = None
    try:
        host = host_by_id(item["host_id"])
    except ValueError:
        pass
    local = use_local_execution(cfg, dataset_path)
    try:
        qa_root = item["qa_root"].rstrip("/")
        env = os.environ.copy()
        env.update(build_replay_env(item))
        if local:
            item["execution"] = "local"
            env_map = build_replay_env(item)
            data_root = str(item.get("dataset_path") or "").rstrip("/")
            episodes = str(item.get("episodes") or "").strip()
            qa_cmd = f"bash run_qa.sh {shlex.quote(data_root)}"
            if episodes:
                qa_cmd += f" {shlex.quote(episodes)}"
            item["command"] = (
                f"cd {shlex.quote(qa_root)} && "
                + " ".join(f"{k}={shlex.quote(v)}" for k, v in env_map.items())
                + " " + qa_cmd
            )
            item["message"] = "正在本机启动 SONIC 回放（dataprocess）"
            store.save()
            env = os.environ.copy()
            env.update(env_map)
            proc = subprocess.Popen(
                ["bash", "-lc", qa_cmd],
                cwd=qa_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            if not host:
                raise RuntimeError("远端模式需要有效 Host")
            item["execution"] = "remote"
            env_map = build_replay_env(item)
            env_parts = [f"{k}={shlex.quote(v)}" for k, v in env_map.items()]
            # Pass session path as CLI so run_qa.sh does not fall back to hardcoded DATA/SESSION.
            data_root = str(item.get("dataset_path") or "").rstrip("/")
            episodes = str(item.get("episodes") or "").strip()
            qa_cmd = f"bash {shlex.quote(qa_root + '/run_qa.sh')} {shlex.quote(data_root)}"
            if episodes:
                qa_cmd += f" {shlex.quote(episodes)}"
            command = f"cd {shlex.quote(qa_root)} && " + " ".join(env_parts) + " " + qa_cmd
            item["command"] = command
            item["message"] = "正在 cluster_0 上启动 SONIC 回放"
            store.save()
            proc = subprocess.Popen(
                ssh_args(host["target"], ["bash", "-lc", command]),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with worker_lock:
            workers[job_id] = {"proc": proc, "kind": "replay"}
        assert proc.stdout is not None
        for line in proc.stdout:
            log_line(item, line)
        code = proc.wait()
        summary = read_json_at(f"{item['out_dir']}/batch_replay_summary.json", cfg, host)
        if summary:
            item["summary"] = summary
            merge_replay_summary(dataset_path, summary)
        if item.get("status") != "cancelled":
            item["status"] = "completed" if code == 0 else "error"
            if code == 0 and summary:
                item["message"] = f"回放完成：{summary.get('n_ok', 0)} 合格 / {summary.get('n_fail', 0)} 异常"
            elif code == 0:
                item["message"] = "回放进程已结束"
            else:
                item["message"] = f"回放失败 (rc={code})"
        item["completed_at"] = now_iso()
    except Exception as exc:  # noqa: BLE001
        item["status"] = "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(job_id, None)


def cleanup_dataset_prune_artifacts(
    dataset_path: str,
    *,
    host: dict[str, Any] | None = None,
    local: bool = False,
) -> list[str]:
    """Remove Studio prune scratch files; keep only data/videos/meta/labels."""
    root = dataset_path.rstrip("/")
    removed: list[str] = []
    shell = (
        "cd \"$1\" || exit 0; "
        "rm -f .studio_prune_lerobot_v3.py .sonic_qa_issues.json "
        "meta/vision_episode_allowlist.json meta/all_episode_allowlist.json meta/allowlist.json "
        "vision_episode_allowlist.json all_episode_allowlist.json allowlist.json "
        "sonic_qa_valid_invalid.json .sonic_qa_valid_invalid.json 2>/dev/null; "
        "rm -rf .prune_staging .prune_backup_* 2>/dev/null; "
        "ls -la"
    )
    try:
        if local:
            proc = run_command(["bash", "-lc", shell, "cleanup", root], timeout=30)
        elif host:
            proc = run_command(
                ssh_args(host["target"], ["bash", "-lc", shell, "cleanup", root]),
                timeout=45,
            )
        else:
            return removed
        if proc.stdout:
            # Best-effort: callers may log; we just signal attempt happened.
            removed.append("prune_scratch")
        if proc.returncode:
            removed.append(f"cleanup_rc={proc.returncode}")
    except Exception:  # noqa: BLE001
        pass
    return removed


def prune_worker(job_id: str) -> None:
    item = store.find("replay_jobs", job_id)
    if not item or item.get("kind") != "prune":
        return
    dataset_path = item["dataset_path"].rstrip("/")
    entry = issues_for_dataset(dataset_path)
    cfg = replay_config()
    host = None
    try:
        host = host_by_id(item["host_id"])
    except ValueError:
        pass
    local = use_local_execution(cfg, dataset_path) or Path(dataset_path).expanduser().is_dir()
    if item.get("execution") == "local":
        local = True
    remote_script = ""
    try:
        if not PRUNE_SCRIPT.is_file():
            raise RuntimeError(f"缺少 prune 脚本: {PRUNE_SCRIPT}")
        command = ["python3", str(PRUNE_SCRIPT), "--root", dataset_path,
                   "--delete", *map(str, item["delete_episodes"])]
        item["command"] = shlex.join(command)
        if local:
            item["execution"] = "local"
            item["message"] = "正在本机删除并重新排列 Episode"
            store.save()
            proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        else:
            if not host:
                raise RuntimeError("远端模式需要有效 Host")
            # Keep scratch out of the dataset tree (tmp + post-clean).
            remote_script = f"/tmp/studio_prune_{uuid.uuid4().hex[:10]}.py"
            upload = subprocess.run(
                ssh_args(host["target"], ["sh", "-c", 'cat > "$1"', "prune-upload", remote_script]),
                input=PRUNE_SCRIPT.read_text("utf-8"), text=True, capture_output=True, timeout=30, check=False,
            )
            if upload.returncode:
                raise RuntimeError("上传 prune 脚本失败: " + upload.stderr.strip())
            command = ["python3", remote_script, "--root", dataset_path,
                       "--delete", *map(str, item["delete_episodes"])]
            item["command"] = shlex.join(command)
            item["execution"] = "remote"
            item["message"] = "正在远端删除并重新排列 Episode"
            store.save()
            proc = subprocess.Popen(ssh_args(host["target"], command), text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        with worker_lock:
            workers[job_id] = {"proc": proc, "kind": "prune"}
        assert proc.stdout is not None
        for line in proc.stdout:
            log_line(item, line)
        code = proc.wait()
        if code == 0:
            deleted = sorted(int(x) for x in (item.get("delete_episodes") or []))
            # Prefer remap from prune JSON report in logs; else reconstruct.
            old_to_new: dict[int, int] = {}
            report_obj: dict[str, Any] | None = None
            blob = "\n".join(item.get("logs") or [])
            # logs are prefixed with timestamps like [15:57:24] — strip for JSON hunt
            try:
                # Find last JSON object in output
                start = blob.rfind("{")
                while start >= 0:
                    try:
                        # strip leading "[hh:mm:ss] " fragments line-wise
                        cleaned = "\n".join(
                            re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s?", "", ln)
                            for ln in blob[start:].splitlines()
                        )
                        cand = json.loads(cleaned)
                        if isinstance(cand, dict) and "old_to_new" in cand:
                            report_obj = cand
                            break
                    except ValueError:
                        pass
                    start = blob.rfind("{", 0, start)
            except Exception:  # noqa: BLE001
                report_obj = None
            if report_obj and isinstance(report_obj.get("old_to_new"), dict):
                old_to_new = {int(k): int(v) for k, v in report_obj["old_to_new"].items()}
            else:
                # Reconstruct assuming contiguous 0..N-1 before delete
                prior_entry, _, _ = read_label_bundle(dataset_path, host=host)
                known = set(prior_entry.get("valid") or []) | set(prior_entry.get("invalid") or []) | set(deleted)
                before = max(known) + 1 if known else 0
                keep = [i for i in range(before) if i not in set(deleted)]
                old_to_new = {old: new for new, old in enumerate(keep)}

            try:
                prior_entry, _, parent = read_label_bundle(dataset_path, host=host)
                # First: record deletes as invalid (old indices), then remap survivors
                marked = move_eps_to_invalid(prior_entry, deleted)
                remapped = remap_label_entry(marked, old_to_new)
                written = write_session_and_parent_labels(
                    dataset_path, remapped, host=host, parent_path=parent
                )
                log_line(
                    item,
                    f"[studio] labels 已同步 valid={written['valid_count']} "
                    f"invalid={written['invalid_count']} → {written['labels_path']} + {written['parent_path']}",
                )
            except Exception as label_exc:  # noqa: BLE001
                log_line(item, f"[studio] labels 同步失败（数据已删）: {label_exc}")

            entry["marked_delete"] = []
            entry["prune_status"] = "completed"
            skipped = item.get("skipped_missing") or []
            skip_msg = f"；另跳过不存在 {skipped}" if skipped else ""
            entry["prune_message"] = (
                f"已删除 {len(deleted)} 条并重排索引；labels 已写回 session+父级{skip_msg}"
            )
            # Drop deleted issue keys and remap survivors to new indices.
            old_issues = dict(entry.get("issues") or {})
            entry["issues"] = {}
            for key, issue in old_issues.items():
                try:
                    old_ep = int(issue.get("episode", key))
                except (TypeError, ValueError):
                    continue
                if old_ep in set(deleted):
                    continue
                new_ep = old_to_new.get(old_ep)
                if new_ep is None:
                    continue
                issue = dict(issue)
                issue["episode"] = new_ep
                if issue.get("summary_zh"):
                    issue["summary_zh"] = re.sub(
                        rf"\bEpisode\s+{old_ep}\b",
                        f"Episode {new_ep}",
                        str(issue["summary_zh"]),
                    )
                entry["issues"][str(new_ep)] = issue
            for issue in entry["issues"].values():
                if issue.get("status") == "marked_delete":
                    issue["status"] = "resolved"
            item["status"] = "completed"
            item["message"] = entry["prune_message"]
        else:
            entry["prune_status"] = "error"
            entry["prune_message"] = f"删除失败 (rc={code})"
            item["status"] = "error"
            item["message"] = entry["prune_message"]
        entry["updated_at"] = now_iso()
        item["completed_at"] = now_iso()
    except Exception as exc:  # noqa: BLE001
        entry["prune_status"] = "error"
        entry["prune_message"] = str(exc)
        item["status"] = "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        try:
            if remote_script and host:
                run_command(ssh_args(host["target"], ["rm", "-f", "--", remote_script]), timeout=15)
            cleanup_dataset_prune_artifacts(dataset_path, host=host, local=local)
            log_line(item, "[studio] 已清理 prune 临时文件（仅保留 data/videos/meta/labels）")
        except Exception as clean_exc:  # noqa: BLE001
            log_line(item, f"[studio] prune 临时文件清理失败: {clean_exc}")
        store.save()
        with worker_lock:
            workers.pop(job_id, None)


@app.get("/api/replay/config")
def get_replay_config():
    cfg = replay_config()
    presets = discover_dataset_presets(cfg)
    return jsonify({
        "ok": True,
        "config": cfg,
        "presets": presets,
        "local_dataprocess": str(LOCAL_DATAPROCESS_ROOT),
        "qa_root_default": DEFAULT_QA_ROOT,
        "local_qa_exists": LOCAL_QA_ROOT.is_dir(),
    })


@app.post("/api/replay/config")
def save_replay_config():
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        if data.get("execution_mode") in {"local", "remote", "auto"}:
            cfg["execution_mode"] = data["execution_mode"]
        if data.get("host_id"):
            host_by_id(str(data["host_id"]))
            cfg["host_id"] = str(data["host_id"])
        if data.get("qa_root"):
            cfg["qa_root"] = require_abs(data["qa_root"], "QA 根目录")
        if data.get("dataset_base"):
            cfg["dataset_base"] = require_abs(data["dataset_base"], "数据集基目录")
        if data.get("viser_port"):
            cfg["viser_port"] = int(data["viser_port"])
        if data.get("default_dataset_path"):
            cfg["default_dataset_path"] = require_abs(data["default_dataset_path"], "默认数据集路径")
        if data.get("last_dataset_path"):
            cfg["last_dataset_path"] = require_abs(data["last_dataset_path"], "当前数据集路径")
        with store.lock:
            store.data["replay_config"] = cfg
            store.save()
        return jsonify({"ok": True, "config": cfg, "presets": discover_dataset_presets(cfg)})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/replay/experiments")
def list_replay_experiments_api():
    dataset_path = str(request.args.get("dataset_path") or "").strip()
    cfg = replay_config()
    qa_root = str(request.args.get("qa_root") or cfg.get("qa_root") or DEFAULT_QA_ROOT).strip()
    if not dataset_path:
        return api_error("缺少 dataset_path")
    name = Path(dataset_path).name
    host = None
    try:
        host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
    except ValueError:
        host = None
    return jsonify({
        "ok": True,
        "experiments": list_replay_experiments(qa_root, name, cfg=cfg, host=host),
    })


@app.post("/api/replay/experiments/import")
def import_replay_experiment():
    data = request.get_json(silent=True) or {}
    try:
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        exp_path = require_abs(data.get("experiment_path", ""), "实验目录")
        summary_path = Path(exp_path) / "batch_replay_summary.json"
        if not summary_path.is_file():
            raise ValueError("该实验目录没有 batch_replay_summary.json")
        summary = json.loads(summary_path.read_text("utf-8"))
        merge_replay_summary(dataset_path, summary)
        return jsonify({"ok": True, "summary": summary, "issues": issues_for_dataset(dataset_path)})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/datasets/probe")
def probe_replay_dataset():
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        host = None
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        try:
            host = host_by_id(host_id)
        except ValueError:
            if not use_local_execution(cfg, str(data.get("dataset_path", ""))):
                raise
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        result = probe_dataset(dataset_path, cfg, host)
        qa_root = str(data.get("qa_root") or cfg.get("qa_root") or DEFAULT_QA_ROOT)
        with store.lock:
            rc = store.data.setdefault("replay_config", {})
            rc["last_dataset_path"] = dataset_path
            store.save()
        result["issues"] = issues_for_dataset(dataset_path)
        result["experiments"] = list_replay_experiments(
            qa_root, result.get("name") or Path(dataset_path).name, cfg=cfg, host=host,
        )
        result["execution_mode"] = "local" if use_local_execution(cfg, dataset_path) else "remote"
        result["qa_root"] = qa_root
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/replay/issues")
def get_replay_issues():
    dataset_path = str(request.args.get("dataset_path") or "").strip()
    if not dataset_path:
        return api_error("缺少 dataset_path")
    return jsonify({"ok": True, "issues": issues_for_dataset(dataset_path)})


@app.post("/api/replay/issues/reset-session")
def reset_replay_issues_session():
    data = request.get_json(silent=True) or {}
    try:
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        entry = reset_replay_session_state(dataset_path)
        return jsonify({"ok": True, "issues": entry})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/issues/clear")
def clear_replay_issues():
    data = request.get_json(silent=True) or {}
    try:
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        entry = clear_replay_issues_log(dataset_path)
        return jsonify({"ok": True, "issues": entry})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/issues")
def update_replay_issues():
    data = request.get_json(silent=True) or {}
    try:
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        entry = issues_for_dataset(dataset_path)
        if "marked_delete" in data:
            entry["marked_delete"] = sorted({int(x) for x in data["marked_delete"]})
        if "issue_updates" in data:
            for ep, payload in (data["issue_updates"] or {}).items():
                key = str(ep)
                current = entry["issues"].setdefault(key, {"episode": int(ep), "source": "manual"})
                if "note" in payload:
                    current["note"] = str(payload["note"])
                if "status" in payload:
                    current["status"] = str(payload["status"])
                if payload.get("summary_zh"):
                    current["summary_zh"] = str(payload["summary_zh"])
                elif not current.get("summary_zh"):
                    current["summary_zh"] = f"Episode {ep}：人工标记异常"
                if payload.get("source"):
                    current["source"] = str(payload["source"])
        if data.get("remove_episodes"):
            remove = {int(x) for x in data["remove_episodes"]}
            for ep in remove:
                entry["issues"].pop(str(ep), None)
            entry["marked_delete"] = sorted(x for x in entry["marked_delete"] if x not in remove)
        entry["updated_at"] = now_iso()
        store.save()
        return jsonify({"ok": True, "issues": entry})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


def _fail_episodes_from_issues(entry: dict[str, Any]) -> set[int]:
    fails: set[int] = set()
    for key, payload in (entry.get("issues") or {}).items():
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "resolved":
            continue
        try:
            fails.add(int(payload.get("episode", key)))
        except (TypeError, ValueError):
            continue
    return fails


def compute_valid_invalid(total: int, entry: dict[str, Any]) -> dict[str, Any]:
    """Build valid/invalid lists; training/screening uses valid only."""
    total = max(0, int(total))
    fails = _fail_episodes_from_issues(entry)
    invalid = sorted(ep for ep in fails if 0 <= ep < total)
    valid = [i for i in range(total) if i not in fails]
    return {
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid": valid,
        "invalid": invalid,
        "total_episodes": total,
    }


def parent_skill_json_path(dataset_path: str, *, manifest_name: str = "") -> str:
    """Session dir's sibling skill_N.json, e.g. .../skill_2_.../skill_2.json."""
    root = Path(str(dataset_path or "").rstrip("/")).parent
    name = str(manifest_name or "").strip()
    if not name:
        name = manifest_basename(root.name)
    if not name.endswith(".json"):
        name = name + ".json"
    return str(root / name)


def session_labels_path(dataset_path: str) -> str:
    return str(Path(str(dataset_path or "").rstrip("/")) / "labels.json")


def normalize_label_entry(
    valid: list[int] | None = None,
    invalid: list[int] | None = None,
) -> dict[str, Any]:
    valid_set = {int(x) for x in (valid or [])}
    invalid_set = {int(x) for x in (invalid or [])} - valid_set
    valid_list = sorted(valid_set)
    invalid_list = sorted(invalid_set)
    return {
        "valid_count": len(valid_list),
        "valid": valid_list,
        "invalid": invalid_list,
    }


def move_eps_to_invalid(entry: dict[str, Any], eps: list[int] | set[int]) -> dict[str, Any]:
    """Move episode indices from valid → invalid (keep other invalids)."""
    move = {int(x) for x in eps}
    valid = [int(x) for x in (entry.get("valid") or []) if int(x) not in move]
    invalid = sorted({int(x) for x in (entry.get("invalid") or [])} | move)
    # Drop from invalid anything still listed valid (shouldn't happen)
    invalid = [x for x in invalid if x not in set(valid)]
    return normalize_label_entry(valid, invalid)


def remap_label_entry(
    entry: dict[str, Any],
    old_to_new: dict[int, int],
) -> dict[str, Any]:
    """Remap valid/invalid after prune reindex; dropped eps disappear."""
    valid = [old_to_new[int(x)] for x in (entry.get("valid") or []) if int(x) in old_to_new]
    invalid = [old_to_new[int(x)] for x in (entry.get("invalid") or []) if int(x) in old_to_new]
    return normalize_label_entry(valid, invalid)


def read_label_bundle(
    dataset_path: str,
    *,
    host: dict[str, Any] | None = None,
    parent_path: str = "",
) -> tuple[dict[str, Any], str, str]:
    """Return (session_entry, session_labels_path, parent_skill_json_path)."""
    cfg = replay_config()
    dataset_path = str(dataset_path).rstrip("/")
    session = Path(dataset_path).name
    labels_file = session_labels_path(dataset_path)
    parent = parent_path or parent_skill_json_path(dataset_path)
    local = Path(dataset_path).is_dir()
    raw = read_json_at(labels_file, cfg, None if (local and Path(labels_file).is_file()) else host)
    entry: dict[str, Any] = {"valid_count": 0, "valid": [], "invalid": []}
    if isinstance(raw, dict):
        try:
            parsed = parse_labels_valid_invalid(raw, session_name=session)
            entry = normalize_label_entry(parsed.get("valid"), parsed.get("invalid"))
        except ValueError:
            if "valid" in raw or "invalid" in raw:
                entry = normalize_label_entry(raw.get("valid"), raw.get("invalid"))
    return entry, labels_file, parent


def write_session_and_parent_labels(
    dataset_path: str,
    entry: dict[str, Any],
    *,
    host: dict[str, Any] | None = None,
    parent_path: str = "",
) -> dict[str, Any]:
    """Write session labels.json and merge the same entry into parent skill_N.json."""
    dataset_path = str(dataset_path).rstrip("/")
    session = Path(dataset_path).name
    entry = normalize_label_entry(entry.get("valid"), entry.get("invalid"))
    labels_file = session_labels_path(dataset_path)
    parent = parent_path or parent_skill_json_path(dataset_path)
    local = Path(dataset_path).is_dir()
    write_host = None if local else host
    # Session labels: single-session object (same shape as 01 upload)
    write_json_at(labels_file, {session: entry}, write_host)
    # Parent skill_N.json: atomic merge by session key
    if host and not local:
        proc = subprocess.run(
            ssh_args(host["target"], ["python3", "-c", REMOTE_MANIFEST_MERGE, parent, session]),
            input=json.dumps(entry, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(
                (proc.stderr or proc.stdout).strip() or f"父级 labels 更新失败: {parent}"
            )
    else:
        aggregate: dict[str, Any] = {}
        p = Path(parent)
        if p.is_file():
            try:
                aggregate = json.loads(p.read_text("utf-8")) or {}
            except (OSError, ValueError):
                aggregate = {}
        if not isinstance(aggregate, dict):
            aggregate = {}
        aggregate[session] = entry
        write_json_at(parent, aggregate, None)
    return {
        "session": session,
        "labels_path": labels_file,
        "parent_path": parent,
        "entry": entry,
        "valid_count": entry["valid_count"],
        "invalid_count": len(entry["invalid"]),
    }


def apply_invalids_to_dataset_labels(
    dataset_path: str,
    invalid_eps: list[int] | set[int],
    *,
    host: dict[str, Any] | None = None,
    total_episodes: int | None = None,
    replace_from_issues: bool = False,
    issues_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update session + parent labels: move eps valid→invalid (or rebuild from issues)."""
    dataset_path = str(dataset_path).rstrip("/")
    entry, _, parent = read_label_bundle(dataset_path, host=host)
    if replace_from_issues:
        total = int(total_episodes or 0)
        if total <= 0:
            # Prefer current valid∪invalid span
            known = set(entry.get("valid") or []) | set(entry.get("invalid") or []) | set(invalid_eps)
            total = (max(known) + 1) if known else 0
        split = compute_valid_invalid(total, issues_entry or {"issues": {}})
        # Also force explicit invalid_eps
        entry = move_eps_to_invalid(
            normalize_label_entry(split["valid"], split["invalid"]),
            invalid_eps,
        )
    else:
        entry = move_eps_to_invalid(entry, invalid_eps)
    return write_session_and_parent_labels(dataset_path, entry, host=host, parent_path=parent)


def parse_labels_valid_invalid(
    raw: dict[str, Any] | list[Any],
    *,
    session_name: str = "",
) -> dict[str, Any]:
    """Accept labels.json / skill_N.json / flat {valid,invalid} / allowlist shapes.

    For parent skill_N.json ({session: {valid, invalid}}), pass session_name to
    pick one session — episode indices are per-session and must not be merged.
    """
    if isinstance(raw, list):
        valid = sorted({int(x) for x in raw})
        return {"valid": valid, "invalid": [], "valid_count": len(valid)}
    if not isinstance(raw, dict):
        raise ValueError("标签 JSON 格式无效")
    if "valid" in raw or "invalid" in raw:
        valid = sorted({int(x) for x in (raw.get("valid") or [])})
        invalid = sorted({int(x) for x in (raw.get("invalid") or [])})
        return {"valid": valid, "invalid": invalid, "valid_count": len(valid)}
    if "episode_index" in raw or "episodes" in raw:
        eps = raw.get("episode_index") or raw.get("episodes") or []
        valid = sorted({int(x) for x in eps})
        return {"valid": valid, "invalid": [], "valid_count": len(valid)}
    # skill_N.json: {session: {valid, invalid, valid_count}, ...}
    session_name = str(session_name or "").strip()
    if session_name and isinstance(raw.get(session_name), dict):
        entry = raw[session_name]
        valid = sorted({int(x) for x in (entry.get("valid") or [])})
        invalid = sorted({int(x) for x in (entry.get("invalid") or [])})
        return {
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "session": session_name,
        }
    # No session specified: refuse to merge indices across sessions (would collide).
    sessions = [
        k for k, entry in raw.items()
        if isinstance(entry, dict) and ("valid" in entry or "invalid" in entry)
    ]
    if sessions:
        raise ValueError(
            "父级 skill_N.json 含多个 session，请指定 session_name="
            + sessions[0]
            + f" 等（共 {len(sessions)} 个）"
        )
    raise ValueError("未找到 valid / invalid / episode_index 字段")


def write_json_at(path: str, payload: dict[str, Any], host: dict[str, Any] | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return
    except OSError:
        if host is None:
            raise
    py = (
        "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
        "p.write_text(sys.argv[2],encoding='utf-8')"
    )
    proc = run_command(ssh_args(host["target"], ["python3", "-c", py, path, text]), timeout=30)
    if proc.returncode:
        raise ValueError(f"远端写入失败: {proc.stderr or proc.stdout or path}")


def resolve_830_dataset_path(dataset_path: str, remote_hint: str | None = None) -> str:
    """Map a dataset to cluster_0 MixCorpus/datasets/830/<name>."""
    base = DEFAULT_DATASET_BASE_CLUSTER.rstrip("/")
    hint = str(remote_hint or "").strip()
    if hint:
        if hint.startswith(base) or "/datasets/830/" in hint:
            return hint.rstrip("/")
        name = Path(hint).name
        return f"{base}/{name}"
    path = str(dataset_path or "").rstrip("/")
    if not path:
        raise ValueError("数据集路径为空")
    if path.startswith(base) or "/datasets/830/" in path:
        return path
    return f"{base}/{Path(path).name}"


ALLOWLIST_META_NAMES = (
    "vision_episode_allowlist.json",
    "all_episode_allowlist.json",
    "sonic_qa_valid_invalid.json",
)


def sync_allowlist_meta_to_830(
    *,
    dataset_path: str,
    host_id: str | None = None,
    remote_path: str | None = None,
    payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Push 02筛选 allowlist meta to cluster_0 datasets/830/<dataset>/meta/."""
    cfg = replay_config()
    dataset_path = require_abs(dataset_path, "数据集路径")
    host_id = str(host_id or cfg.get("host_id") or "cluster_0")
    host = host_by_id(host_id)
    remote_root = resolve_830_dataset_path(dataset_path, remote_path)
    remote_meta = f"{remote_root.rstrip('/')}/meta"
    local_root = Path(dataset_path)
    local = local_root.is_dir()

    # Ensure remote meta dir exists
    mkdir = run_command(
        ssh_args(host["target"], ["bash", "-lc", f"mkdir -p -- {shlex.quote(remote_meta)}"]),
        timeout=30,
    )
    if mkdir.returncode:
        raise ValueError(f"远端创建 meta 失败: {(mkdir.stderr or mkdir.stdout or '').strip()}")

    synced: list[str] = []
    if payloads:
        for name, payload in payloads.items():
            dest = f"{remote_meta}/{name}"
            write_json_at(dest, payload, host)
            synced.append(dest)
    elif local:
        for name in ALLOWLIST_META_NAMES:
            src = local_root / "meta" / name
            if not src.is_file():
                continue
            dest = f"{host['target']}:{remote_meta}/{name}"
            proc = run_command(
                ["rsync", "-az", *rsync_ssh(), str(src), dest],
                timeout=120,
            )
            if proc.returncode:
                raise ValueError(
                    f"同步 {name} 失败: {(proc.stderr or proc.stdout or '').strip()[-400:]}"
                )
            synced.append(f"{remote_meta}/{name}")
    else:
        # Dataset itself is remote — copy meta within cluster if paths differ
        src_meta = f"{dataset_path.rstrip('/')}/meta"
        if src_meta.rstrip("/") != remote_meta.rstrip("/"):
            for name in ALLOWLIST_META_NAMES:
                src = f"{src_meta}/{name}"
                dest = f"{remote_meta}/{name}"
                copy = run_command(
                    ssh_args(
                        host["target"],
                        [
                            "bash", "-lc",
                            f"test -f {shlex.quote(src)} && cp -f -- {shlex.quote(src)} {shlex.quote(dest)}",
                        ],
                    ),
                    timeout=60,
                )
                # non-zero if missing; skip quietly
                if copy.returncode == 0:
                    synced.append(dest)
        else:
            for name in ALLOWLIST_META_NAMES:
                synced.append(f"{remote_meta}/{name}")

    if not synced:
        raise ValueError("没有可同步的 allowlist meta（请先导出 valid）")

    return {
        "ok": True,
        "host_id": host_id,
        "dataset_path": dataset_path,
        "remote_path": remote_root,
        "remote_meta": remote_meta,
        "synced": synced,
        "synced_count": len(synced),
    }


def do_export_valid_allowlist(
    *,
    dataset_path: str,
    host_id: str | None = None,
    source: str = "issues",
    labels_path: str | None = None,
    labels: Any = None,
    remote_path: str | None = None,
    sync_cluster: bool = True,
    write_allowlists: bool = False,
) -> dict[str, Any]:
    """02 筛选落盘：把不合格 ep 从 valid 移到 invalid，写 session labels.json + 父级 skill_N.json。

    默认不再写 vision/all allowlist / sonic_qa_valid_invalid.json（训练直接读 labels）。
    """
    cfg = replay_config()
    dataset_path = require_abs(dataset_path, "数据集路径")
    host = None
    host_id = str(host_id or cfg.get("host_id") or "cluster_0")
    try:
        host = host_by_id(host_id)
    except ValueError:
        pass
    local = use_local_execution(cfg, dataset_path) or Path(dataset_path).is_dir()
    meta = probe_dataset(dataset_path, cfg, None if local else host)
    total = int(meta.get("total_episodes") or 0)
    if total <= 0:
        raise ValueError("数据集中没有 Episode")

    source = str(source or "issues").strip()
    issues_entry = issues_for_dataset(dataset_path)
    if source == "labels" or labels_path or labels is not None:
        raw = labels
        if labels_path and raw is None:
            labels_path = require_abs(labels_path, "标签 JSON")
            raw = read_json_at(
                labels_path, cfg,
                None if Path(labels_path).is_file() else host,
            )
            if raw is None:
                raise ValueError(f"无法读取标签 JSON: {labels_path}")
        if raw is None:
            raise ValueError("请提供 labels / labels_path")
        session_name = Path(dataset_path).name
        try:
            parsed = parse_labels_valid_invalid(raw, session_name=session_name)
        except ValueError:
            parsed = parse_labels_valid_invalid(raw)
        valid = [int(ep) for ep in parsed["valid"] if 0 <= int(ep) < total]
        invalid = [int(x) for x in (parsed.get("invalid") or []) if 0 <= int(x) < total]
        fails = _fail_episodes_from_issues(issues_entry) | {
            int(x) for x in (issues_entry.get("marked_delete") or []) if 0 <= int(x) < total
        }
        entry = move_eps_to_invalid(normalize_label_entry(valid, invalid), fails)
        source = "labels+issues"
    else:
        # Rebuild from issues against full episode span; keep prior invalids if any
        prior, _, _ = read_label_bundle(dataset_path, host=None if local else host)
        split = compute_valid_invalid(total, issues_entry)
        marked = {int(x) for x in (issues_entry.get("marked_delete") or []) if 0 <= int(x) < total}
        entry = move_eps_to_invalid(
            normalize_label_entry(
                split["valid"],
                list(set(split["invalid"]) | set(prior.get("invalid") or [])),
            ),
            marked,
        )
        source = "issues"

    if not entry["valid"] and total > 0:
        raise ValueError("valid 列表为空，请检查筛选结果")

    write_host = None if local else host
    written = write_session_and_parent_labels(
        dataset_path, entry, host=write_host, parent_path=parent_skill_json_path(dataset_path)
    )

    issues_entry["allowlist_exported_at"] = now_iso()
    issues_entry["allowlist_valid_count"] = entry["valid_count"]
    store.save()

    result: dict[str, Any] = {
        "ok": True,
        "dataset_path": dataset_path,
        "labels_path": written["labels_path"],
        "parent_path": written["parent_path"],
        "allowlist_path": written["labels_path"],  # back-compat for UI
        "qa_json_path": written["parent_path"],
        "valid_count": entry["valid_count"],
        "invalid_count": len(entry["invalid"]),
        "valid": entry["valid"],
        "invalid": entry["invalid"],
        "source": source,
        "train_hint": {
            "dataset_path": dataset_path,
            "message": (
                f"已写回 labels：session={written['labels_path']} · "
                f"父级={written['parent_path']}（valid→invalid）"
            ),
        },
    }

    # Optional legacy allowlist files — off by default
    if write_allowlists:
        allow_payload = {"episode_index": entry["valid"]}
        qa_payload = {
            "valid_count": entry["valid_count"],
            "valid": entry["valid"],
            "invalid": entry["invalid"],
            "total_episodes": total,
            "source": source,
            "updated_at": now_iso(),
            "dataset_path": dataset_path,
        }
        meta_dir = f"{dataset_path.rstrip('/')}/meta"
        for path, payload in (
            (f"{meta_dir}/vision_episode_allowlist.json", allow_payload),
            (f"{meta_dir}/all_episode_allowlist.json", allow_payload),
            (f"{meta_dir}/sonic_qa_valid_invalid.json", qa_payload),
        ):
            write_json_at(path, payload, write_host)
        result["allowlist_path"] = f"{meta_dir}/vision_episode_allowlist.json"
        if sync_cluster:
            try:
                sync_info = sync_allowlist_meta_to_830(
                    dataset_path=dataset_path,
                    host_id=host_id,
                    remote_path=remote_path,
                    payloads={
                        "vision_episode_allowlist.json": allow_payload,
                        "all_episode_allowlist.json": allow_payload,
                        "sonic_qa_valid_invalid.json": qa_payload,
                    },
                )
                result["cluster_sync"] = sync_info
            except Exception as exc:  # noqa: BLE001
                result["cluster_sync"] = {"ok": False, "message": str(exc)}
    else:
        result["cluster_sync"] = {
            "ok": True,
            "synced_count": 2,
            "remote_path": str(Path(written["parent_path"]).parent),
            "message": "已更新 session labels.json 与父级 skill_N.json（未写 allowlist meta）",
        }

    return result


@app.post("/api/replay/export-valid-allowlist")
def export_valid_allowlist():
    """Export valid-only allowlist for training; optional labels JSON (valid/invalid)."""
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        sync_cluster = data.get("sync_cluster")
        if sync_cluster is None:
            sync_cluster = True
        result = do_export_valid_allowlist(
            dataset_path=str(data.get("dataset_path") or ""),
            host_id=str(data.get("host_id") or cfg.get("host_id") or "cluster_0"),
            source=str(data.get("source") or "issues"),
            labels_path=data.get("labels_path") or data.get("labels_json"),
            labels=data.get("labels"),
            remote_path=data.get("remote_path") or data.get("remote_830_path"),
            sync_cluster=bool(sync_cluster),
            write_allowlists=bool(data.get("write_allowlists") or False),
        )
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/sync-valid-to-cluster")
def sync_valid_to_cluster():
    """Sync already-exported allowlist meta to cluster_0 datasets/830/."""
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        result = sync_allowlist_meta_to_830(
            dataset_path=str(data.get("dataset_path") or ""),
            host_id=str(data.get("host_id") or cfg.get("host_id") or "cluster_0"),
            remote_path=data.get("remote_path") or data.get("remote_830_path"),
        )
        entry = issues_for_dataset(str(data.get("dataset_path") or ""))
        entry["allowlist_synced_at"] = now_iso()
        entry["allowlist_remote_path"] = result.get("remote_path")
        store.save()
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/parse-labels")
def parse_replay_labels():
    """Parse a valid/invalid JSON and return only valid for screening range."""
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        host = None
        try:
            host = host_by_id(str(data.get("host_id") or cfg.get("host_id") or "cluster_0"))
        except ValueError:
            pass
        raw = data.get("labels")
        labels_path = data.get("labels_path") or data.get("labels_json")
        if labels_path and raw is None:
            labels_path = require_abs(labels_path, "标签 JSON")
            raw = read_json_at(labels_path, cfg, None if Path(labels_path).is_file() else host)
            if raw is None:
                raise ValueError(f"无法读取: {labels_path}")
        if raw is None:
            raise ValueError("请提供 labels 或 labels_path")
        session_name = str(
            data.get("session_name")
            or data.get("session")
            or ""
        ).strip()
        if not session_name:
            # Infer from dataset path basename when labels are parent skill_N.json
            for key in ("dataset_path", "dataset", "path"):
                cand = str(data.get(key) or "").strip().rstrip("/")
                if cand:
                    session_name = Path(cand).name
                    break
        try:
            parsed = parse_labels_valid_invalid(raw, session_name=session_name)
        except ValueError as exc:
            # Flat allowlist / single-session still OK without session_name
            if session_name:
                raise
            # Retry only if structure is flat
            if isinstance(raw, dict) and ("valid" in raw or "episode_index" in raw or "episodes" in raw):
                parsed = parse_labels_valid_invalid(raw)
            else:
                raise ValueError(str(exc) + "；或在请求中传 session_name / dataset_path") from exc
        total = data.get("total_episodes")
        valid = [int(x) for x in parsed["valid"]]
        invalid = [int(x) for x in (parsed.get("invalid") or [])]
        # Skill card often keeps valid.json + invalid.json as siblings.
        if labels_path and not invalid:
            p = Path(str(labels_path))
            if p.name.lower() == "valid.json":
                inv_path = p.with_name("invalid.json")
                inv_raw = read_json_at(str(inv_path), cfg, None if inv_path.is_file() else host)
                if isinstance(inv_raw, list):
                    invalid = [int(x) for x in inv_raw]
                elif isinstance(inv_raw, dict) and ("invalid" in inv_raw or "valid" in inv_raw):
                    invalid = [int(x) for x in (inv_raw.get("invalid") or [])]
        if total is not None:
            total = int(total)
            valid = [ep for ep in valid if 0 <= ep < total]
            invalid = [ep for ep in invalid if 0 <= ep < total and ep not in set(valid)]
        else:
            invalid = [ep for ep in invalid if ep not in set(valid)]
        return jsonify({
            "ok": True,
            **parsed,
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "total_episodes": total,
            "labels_path": labels_path,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/sessions")
def start_replay_session():
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        host = None
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        try:
            host = host_by_id(host_id)
        except ValueError:
            pass
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        if not use_local_execution(cfg, dataset_path) and not host:
            raise ValueError("远端模式需要有效 Host")
        episodes_text = str(data.get("episodes") or "0-19").strip()
        meta = probe_dataset(dataset_path, cfg, host)
        total = int(meta.get("total_episodes") or 0)
        if total <= 0:
            raise ValueError("数据集中没有 Episode")
        parse_episode_selection(episodes_text, total)
        qa_root = require_abs(str(data.get("qa_root") or cfg["qa_root"]), "QA 根目录")
        dataset_name = data.get("dataset_name") or Path(dataset_path).name
        tag = f"{dataset_name}_ep{episodes_text.replace(' ', '_')}"
        out_dir = f"{qa_root}/experiments/{tag}"
        local = use_local_execution(cfg, dataset_path)
        item = {
            "id": "replay_" + uuid.uuid4().hex[:8],
            "kind": "replay",
            "host_id": host_id,
            "execution": "local" if local else "remote",
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "episodes": episodes_text,
            "qa_root": qa_root,
            "out_dir": out_dir,
            "viser_port": int(data.get("viser_port") or cfg.get("viser_port") or 8081),
            "live_ui": 1 if data.get("live_ui", True) else 0,
            "loop": int(data.get("loop") or 1),
            "loop_pause": float(data.get("loop_pause") or 1),
            "cuda_device": data.get("cuda_device"),
            "status": "running",
            "message": "准备启动回放",
            "created_at": now_iso(),
            "logs": [],
            "summary": None,
        }
        with store.lock:
            store.data["replay_jobs"].insert(0, item)
            store.data["replay_jobs"] = store.data["replay_jobs"][:40]
            store.save()
        tunnel_info: dict[str, Any] = {"auto": False, "alive": False}
        if not local and host:
            try:
                # Heal any stale orphan forward, then open tunnel together with playback.
                # Skip ssh -L when Host is this machine (cluster_0) — it steals Viser's port.
                if is_loopback_ssh_target(str(host.get("target") or host_id)):
                    reclaim_stale_viser_port(int(item["viser_port"]))
                    tunnel_info = {
                        "ok": True,
                        "alive": False,
                        "auto": False,
                        "skipped": True,
                        "message": f"本机回放：直接打开 http://127.0.0.1:{item['viser_port']}/",
                    }
                    log_line(item, tunnel_info["message"])
                else:
                    reclaim_stale_viser_port(int(item["viser_port"]))
                    tunnel_info = ensure_viser_tunnel(host["target"], int(item["viser_port"]))
                    log_line(item, tunnel_info.get("message") or "Viser 隧道已随回放启动")
            except Exception as tun_exc:  # noqa: BLE001
                tunnel_info = {"ok": False, "alive": False, "auto": False, "message": str(tun_exc)}
                log_line(item, f"Viser 自动隧道失败: {tun_exc}")
        threading.Thread(target=replay_worker, args=(item["id"],), daemon=True).start()
        if local or tunnel_info.get("skipped"):
            tunnel_hint = f"本机模式：浏览器直接访问 http://127.0.0.1:{item['viser_port']}/"
        elif tunnel_info.get("alive"):
            tunnel_hint = tunnel_info.get("message") or (
                f"已自动转发 http://127.0.0.1:{item['viser_port']}/ → {host['target'] if host else 'cluster_0'}"
            )
        else:
            tunnel_hint = (
                f"自动隧道失败，请手动执行: "
                f"ssh -L {item['viser_port']}:127.0.0.1:{item['viser_port']} "
                f"{host['target'] if host else 'cluster_0'}"
            )
        return jsonify({
            "ok": True,
            "job": item,
            "tunnel_hint": tunnel_hint,
            "tunnel": tunnel_info,
            "execution": item["execution"],
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/replay/sessions/<item_id>")
def get_replay_session(item_id: str):
    item = store.find("replay_jobs", item_id)
    if not item:
        return api_error("回放任务不存在", 404)
    return jsonify({"ok": True, "job": item})


@app.post("/api/replay/sessions/<item_id>/cancel")
def cancel_replay_session(item_id: str):
    item = store.find("replay_jobs", item_id)
    if not item:
        return api_error("回放任务不存在", 404)
    with worker_lock:
        worker = workers.get(item_id)
        proc = worker.get("proc") if worker else None
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    item["status"], item["message"] = "cancelled", "已取消回放"
    store.save()
    # Keep tunnel alive while other remote replays may still run; stop if none left.
    still = any(
        j.get("kind") == "replay" and j.get("status") == "running" and j.get("id") != item_id
        for j in (store.data.get("replay_jobs") or [])
    )
    tunnel = None
    if not still:
        tunnel = stop_viser_tunnel()
    return jsonify({"ok": True, "tunnel": tunnel})


@app.post("/api/replay/sessions/<item_id>/refresh-summary")
def refresh_replay_summary(item_id: str):
    item = store.find("replay_jobs", item_id)
    if not item:
        return api_error("回放任务不存在", 404)
    try:
        host = None
        try:
            host = host_by_id(item["host_id"])
        except ValueError:
            pass
        cfg = replay_config()
        summary = read_json_at(f"{item['out_dir']}/batch_replay_summary.json", cfg, host)
        if not summary:
            return api_error("尚未生成 batch_replay_summary.json")
        item["summary"] = summary
        merge_replay_summary(item["dataset_path"], summary)
        store.save()
        return jsonify({"ok": True, "summary": summary, "issues": issues_for_dataset(item["dataset_path"])})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/replay/prune")
def start_replay_prune():
    data = request.get_json(silent=True) or {}
    try:
        cfg = replay_config()
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0").strip()
        host = None
        try:
            host = host_by_id(host_id)
        except ValueError:
            host = None
        dataset_path = require_abs(data.get("dataset_path", ""), "数据集路径")
        delete_eps = sorted({int(x) for x in (data.get("delete_episodes") or [])})
        if not delete_eps:
            raise ValueError("请选择要删除的 Episode")
        # Prefer local when the dataset is mounted here — avoids "远端模式需要有效 Host"
        # when UI host_id is wiped but NFS path is readable.
        local = use_local_execution(cfg, dataset_path) or Path(dataset_path).expanduser().is_dir()
        if not local and not host:
            raise ValueError("远端模式需要有效 Host")
        meta = probe_dataset(dataset_path, cfg, None if local else host)
        total = int(meta.get("total_episodes") or 0)
        if total <= 0:
            raise ValueError("数据集中没有 Episode，无法清理")
        present = set(range(total))
        # Prefer concrete indices from meta when available (handles gaps).
        try:
            rows = meta.get("episodes") or meta.get("episode_indices") or []
            if isinstance(rows, list) and rows:
                if isinstance(rows[0], dict):
                    present = {int(r.get("episode_index", r.get("index"))) for r in rows}
                else:
                    present = {int(x) for x in rows}
        except Exception:  # noqa: BLE001
            present = set(range(total))
        missing = sorted(ep for ep in delete_eps if ep not in present)
        delete_eps = sorted(ep for ep in delete_eps if ep in present)
        if not delete_eps:
            entry = issues_for_dataset(dataset_path)
            entry["marked_delete"] = []
            entry["prune_status"] = "error"
            entry["prune_message"] = (
                f"勾选的 Episode 均不在当前数据集中（共 {total} 条，索引 0..{total-1}）；"
                f"已跳过: {missing}"
            )
            entry["updated_at"] = now_iso()
            store.save()
            raise ValueError(entry["prune_message"])
        # Stop Viser forward before dataset rewrite (user request).
        tunnel = stop_viser_tunnel()
        entry = issues_for_dataset(dataset_path)
        entry["marked_delete"] = delete_eps
        entry["prune_status"] = "running"
        skip_note = f"；已跳过不存在 {missing}" if missing else ""
        entry["prune_message"] = f"正在删除 {len(delete_eps)} 条并重排…{skip_note}"
        item = {
            "id": "prune_" + uuid.uuid4().hex[:8],
            "kind": "prune",
            "host_id": (host or {}).get("id") or host_id or "cluster_0",
            "dataset_path": dataset_path,
            "delete_episodes": delete_eps,
            "skipped_missing": missing,
            "execution": "local" if local else "remote",
            "status": "running",
            "message": entry["prune_message"],
            "created_at": now_iso(),
            "logs": [],
        }
        with store.lock:
            store.data["replay_jobs"].insert(0, item)
            store.data["replay_jobs"] = store.data["replay_jobs"][:40]
            store.save()
        threading.Thread(target=prune_worker, args=(item["id"],), daemon=True).start()
        return jsonify({"ok": True, "job": item, "tunnel": tunnel})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/replay/viser-check")
def replay_viser_check():
    port = int(request.args.get("port") or replay_config().get("viser_port") or 8081)
    auto = str(request.args.get("auto") or "").lower() in {"1", "true", "yes"}
    heal = str(request.args.get("heal") or "").lower() in {"1", "true", "yes"} or auto
    tunnel = viser_tunnel_status()
    reclaim_info = None
    http_ok, msg = (False, "本机端口未监听")

    if _local_port_open(port):
        http_ok, msg = probe_viser_http(port)
        # Stale *orphan* ssh -L (Studio 未托管) with remote Viser gone → browser blank page.
        # Do NOT kill Studio-owned tunnels every poll — remote may just be starting.
        studio_owned = bool(tunnel.get("alive") and tunnel.get("port") == port)
        if (
            heal
            and not http_ok
            and not studio_owned
            and msg in {"empty_response", "HTTP 502", "HTTP 503", "HTTP 504"}
        ):
            reclaim_info = reclaim_stale_viser_port(port)
            http_ok, msg = (False, "本机端口未监听")
            tunnel = viser_tunnel_status()

    if auto or heal:
        if not _local_port_open(port) or not http_ok:
            try:
                cfg = replay_config()
                host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
                target = str(host.get("target") or host.get("id") or "cluster_0")
                if is_loopback_ssh_target(target):
                    # Do not recreate self-tunnel; if a stale one holds the port, reclaim it.
                    if _local_port_open(port) and not http_ok and msg in {
                        "empty_response", "http_slow", "HTTP 502", "HTTP 503", "HTTP 504"
                    }:
                        reclaim_info = reclaim_stale_viser_port(port)
                        http_ok, msg = probe_viser_http(port) if _local_port_open(port) else (False, "本机端口未监听")
                    tunnel = {
                        "ok": True,
                        "alive": False,
                        "auto": False,
                        "skipped": True,
                        "port": port,
                        "target": target,
                        "message": f"本机无需隧道 → http://127.0.0.1:{port}/",
                    }
                else:
                    tunnel = ensure_viser_tunnel(host["target"], port)
                    if _local_port_open(port):
                        http_ok, msg = probe_viser_http(port)
            except Exception as exc:  # noqa: BLE001
                tunnel = {**(tunnel if isinstance(tunnel, dict) else {}), "ok": False, "message": str(exc)}

    if not isinstance(tunnel, dict):
        tunnel = viser_tunnel_status()
    port_open = bool(_local_port_open(port))
    tunnel = {**tunnel, "local_open": port_open}
    # reachable for iframe: need real HTTP (or slow=busy but alive). Empty/reset ≠ reachable.
    reachable = bool(http_ok or (port_open and msg == "http_slow"))
    payload: dict[str, Any] = {
        "ok": True,
        "reachable": reachable,
        "http_ok": bool(http_ok),
        "port": port,
        "url": f"http://127.0.0.1:{port}/",
        "tunnel": tunnel,
    }
    if reclaim_info:
        payload["reclaim"] = reclaim_info
    if not reachable:
        if msg == "empty_response":
            payload["message"] = "8081 隧道空响应（远端 Viser 未运行）。请先点「开始回放」"
        elif msg.startswith("HTTP 50"):
            payload["message"] = "隧道已通但远端 Viser 未就绪，请先开始回放或稍候"
        else:
            payload["message"] = msg or "Viser 不可用，请先开始回放"
    elif not http_ok and msg == "http_slow":
        payload["message"] = "端口已通，Viser/Isaac 繁忙或启动中（画面保留）"
    return jsonify(payload)


@app.get("/api/replay/viser-tunnel")
def replay_viser_tunnel_status():
    return jsonify({"ok": True, "tunnel": viser_tunnel_status()})


@app.post("/api/replay/viser-tunnel/stop")
def replay_viser_tunnel_stop():
    return jsonify({"ok": True, "tunnel": stop_viser_tunnel()})


# ---------------------------------------------------------------------------
# Train (05) — nosim distill via phi0_pipeline wrappers
# ---------------------------------------------------------------------------


def train_config() -> dict[str, Any]:
    with store.lock:
        cfg = store.data.setdefault("train_config", default_state()["train_config"])
        cfg.setdefault("custom_tasks", [])
        cfg.setdefault("hidden_task_ids", [])
        # Train runs on cluster_0 by default.
        if cfg.get("execution_mode") not in {"local", "remote", "auto"}:
            cfg["execution_mode"] = "remote"
        if not cfg.get("host_id"):
            cfg["host_id"] = "cluster_0"
        if not cfg.get("phi0_root"):
            cfg["phi0_root"] = "/mnt/data2/wpy/workspace/Phi_0_wpy"
        return dict(cfg)


def merged_train_catalog() -> dict[str, Any]:
    cfg = train_config()
    base = train_backend.load_train_catalog()
    catalog = train_backend.merge_catalog_tasks(base, cfg.get("custom_tasks") or [])
    skills = skill_backend.list_skills(skills_root_path())
    recipes = list(catalog.get("tasks") or [])
    recipe_by_id = {str(t.get("id")): t for t in recipes if t.get("id")}
    hidden = {str(x) for x in (cfg.get("hidden_task_ids") or []) if str(x).strip()}
    launcher = train_backend.launcher_script(catalog)

    def _finalize_task(t: dict[str, Any], recipe: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(t)
        recipe = recipe or {}
        profile = str(
            out.get("train_profile")
            or recipe.get("train_profile")
            or ""
        ).strip()
        if not profile and out.get("mix"):
            profile = "mix_vision_isaac"
        if not profile:
            profile = "vision_teleop"
        out["train_profile"] = profile
        out["script"] = launcher
        flags = {
            **dict(catalog.get("fixed_flags") or {}),
            **train_backend.profile_fixed_flags(profile),
            **dict(recipe.get("fixed_flags") or {}),
            **dict(out.get("fixed_flags") or {}),
        }
        out["fixed_flags"] = flags
        if out.get("train_blocked") and not out.get("train_block_reason"):
            out["train_block_reason"] = "训练被阻止：请检查 train_profile / REF"
        return out

    tasks: list[dict[str, Any]] = []
    for recipe in recipes:
        rid = str(recipe.get("id") or "")
        if rid and rid in hidden:
            continue
        skill = next((s for s in skills if s.get("id") == rid), None)
        if skill:
            merged = skill_backend.skill_as_train_task(skill, recipe)
            merged["id"] = rid
            merged["title"] = recipe.get("title") or merged["title"]
            merged["builtin"] = True
            tasks.append(_finalize_task(merged, recipe))
        else:
            t = dict(recipe)
            t["builtin"] = not bool(t.get("custom"))
            tasks.append(_finalize_task(t, recipe))

    covered = {str(t.get("id")) for t in tasks}
    for skill in skills:
        sid = str(skill.get("id") or "")
        if not sid or sid in covered or sid in recipe_by_id or sid in hidden:
            continue
        recipe = skill_backend.match_train_recipe(skill, catalog)
        t = skill_backend.skill_as_train_task(skill, recipe)
        tasks.append(_finalize_task(t, recipe))

    out = dict(catalog)
    out["script"] = launcher
    out["launcher"] = launcher
    out["tasks"] = tasks
    out["skills"] = skills
    out["hidden_task_ids"] = sorted(hidden)
    out["train_profiles"] = {
        k: {"title": v.get("title"), "description": v.get("description")}
        for k, v in train_backend.TRAIN_PROFILES.items()
    }
    out["source_mode"] = "recipes+skills"
    return out


def _run_train_host_bash(item: dict[str, Any], bash: str, *, timeout: int | None = 60) -> subprocess.CompletedProcess[str]:
    """Run a short bash snippet on the train execution host (local or SSH)."""
    if item.get("execution") == "local":
        return run_command(["bash", "-lc", bash], timeout=timeout)
    host = host_by_id(str(item.get("host_id") or "cluster_0"))
    return run_command(ssh_args(host["target"], ["bash", "-lc", bash]), timeout=timeout)


def train_tmux_alive(item: dict[str, Any]) -> bool:
    session = str(item.get("tmux_session") or "").strip()
    if not session:
        return False
    proc = _run_train_host_bash(item, train_backend.build_tmux_has_bash(session), timeout=20)
    return proc.returncode == 0


def _finalize_train_job(item: dict[str, Any], *, code: int | None, cancelled: bool = False) -> None:
    if cancelled or item.get("status") == "cancelled":
        item["status"] = "cancelled"
        item["message"] = item.get("message") or "已取消训练"
    elif code is None:
        item["status"] = "error"
        item["message"] = "训练会话已结束（无 exit 标记）"
    elif code == 0:
        item["status"] = "completed"
        item["message"] = "训练完成"
    else:
        item["status"] = "error"
        item["message"] = f"训练失败 (rc={code})"
    item["completed_at"] = now_iso()

    # Pull zoo / train_data back to cluster_0 (success, fail, or cancel — keep partial ckpts).
    if (
        train_backend.is_offbox_train_host(str(item.get("host_id") or ""))
        and item.get("staged_to_host")
        and item.get("status") in {"completed", "error", "cancelled"}
        and not item.get("pulled_back")
    ):
        try:
            pullback_train_job_from_host(item)
            if item.get("status") == "completed":
                item["message"] = "训练完成 · 已回传 zoo/train_data"
            elif item.get("status") == "error":
                item["message"] = f"{item.get('message') or '训练失败'} · 已尽量回传产物"
            elif item.get("status") == "cancelled":
                item["message"] = "已取消 · 已尽量回传 ckpt/pack"
        except Exception as pull_exc:  # noqa: BLE001
            log_line(item, f"[pullback] error: {pull_exc}")
            if item.get("status") == "completed":
                item["status"] = "error"
                item["message"] = f"训练完成但回传失败: {pull_exc}"

    skill_id = str(item.get("skill_id") or item.get("task_id") or "").strip()
    pack = item.get("pack") if isinstance(item.get("pack"), dict) else None
    if skill_id and item.get("status") == "completed" and item.get("out_dir"):
        try:
            # After pullback, list ckpts from local FS (same absolute path).
            ckpts = train_backend.list_student_ckpts(str(item["out_dir"]))
            if not ckpts:
                ckpts = _list_ckpts_at(
                    str(item["out_dir"]),
                    str(item.get("execution") or "remote"),
                    str(item.get("host_id") or "cluster_0"),
                )
            defaults = dict(item.get("params") or {})
            bind_ref = str(item.get("ref_root") or "")
            skill_backend.bind_train_artifacts(
                skill_id,
                out_dir=str(item["out_dir"]),
                ckpts=ckpts,
                ref_root=bind_ref,
                defaults={
                    k: defaults[k]
                    for k in (
                        "epochs", "num_envs", "ngpu", "lr", "warmup_ratio",
                        "ckpt_every", "ckpt_every_epoch", "ckpt_step_keep", "hand_mode", "horizon", "lr_scheduler",
                        "deploy_policy", "cuda_devices",
                    )
                    if k in defaults
                },
                prompt=str((item.get("params") or {}).get("prompt") or ""),
                root=skills_root_path(),
            )
            item["student_ckpt"] = next(
                (c.get("path") for c in ckpts if "last" in str(c.get("name") or "").lower()),
                (ckpts[0].get("path") if ckpts else None),
            )
            log_line(item, f"[skill-bind] 已回写技能卡 {skill_id} · ckpts={len(ckpts)}")
        except Exception as bind_exc:  # noqa: BLE001
            log_line(item, f"[skill-bind] 回写技能卡失败: {bind_exc}")
    if pack and item.get("status") == "completed":
        try:
            archived = train_backend.archive_train_pack_to_efs(pack, delete_local=True)
            item["pack_archive"] = {
                "dest": archived.get("dest"),
                "deleted_local": archived.get("deleted_local"),
                "verify": archived.get("verify"),
            }
            if archived.get("ok") and archived.get("archived"):
                log_line(
                    item,
                    f"[pack-archive] 已归档到 EFS 并校验通过 · dest={archived.get('dest')}"
                    + (" · 已删本地 pack" if archived.get("deleted_local") else ""),
                )
                if archived.get("deleted_local"):
                    item["message"] = (
                        (item.get("message") or "训练完成") + " · pack 已归档 EFS 并释放本地盘"
                    )
            else:
                errs = "; ".join(archived.get("errors") or []) or "unknown"
                log_line(item, f"[pack-archive] 跳过删除（归档/校验失败）· {errs}")
                log_line(item, f"[pack-keep] 训练数据包仍保留本地 · root={pack.get('pack_root')}")
        except Exception as arch_exc:  # noqa: BLE001
            log_line(item, f"[pack-archive] error: {arch_exc}")
            log_line(item, f"[pack-keep] 训练数据包仍保留本地 · root={pack.get('pack_root')}")
    elif pack:
        log_line(item, f"[pack-keep] 训练数据包已保留 · root={pack.get('pack_root')}")


def _poll_train_until_done(item: dict[str, Any], job_id: str) -> None:
    """Follow log + tmux liveness until the detached train session finishes."""
    idle_rounds = 0
    last_msg = ""
    has_tmux = bool(str(item.get("tmux_session") or "").strip())
    while True:
        if item.get("status") == "cancelled":
            return
        alive = train_tmux_alive(item) if has_tmux else True
        try:
            log_text = _read_text_at(
                item.get("log_file") or "",
                item.get("execution") or "remote",
                item.get("host_id") or "cluster_0",
                max_bytes=240_000,
            )
        except Exception:  # noqa: BLE001
            log_text = ""
        # UI tails the log file itself; here only refresh the status line.
        progress = ""
        for line in (log_text or "").splitlines():
            if "loss=" in line or "[distill]" in line:
                progress = line.strip()
        if progress and progress != last_msg:
            last_msg = progress
            item["message"] = progress[-180:]
            store.save()
            idle_rounds = 0
        else:
            idle_rounds += 1
        exit_code = train_backend.parse_train_exit_code(log_text)
        if exit_code is not None and (not has_tmux or not alive):
            if item.get("status") != "cancelled":
                _finalize_train_job(item, code=exit_code)
            return
        if has_tmux and not alive:
            # Session gone: wait a bit for tee to flush exit marker.
            if idle_rounds >= 3:
                if item.get("status") != "cancelled":
                    _finalize_train_job(item, code=exit_code)
                return
        time.sleep(2.0)


def train_worker(job_id: str, *, resume_only: bool = False) -> None:
    item = store.find("train_jobs", job_id)
    if not item:
        return
    catalog = merged_train_catalog()
    try:
        with worker_lock:
            if job_id in workers and workers[job_id].get("kind") == "train":
                # Another monitor already attached.
                if resume_only:
                    return
            workers[job_id] = {"kind": "train", "tmux_session": item.get("tmux_session") or ""}

        session = str(item.get("tmux_session") or "").strip()

        if resume_only:
            if session:
                alive = train_tmux_alive(item)
                if alive:
                    item["message"] = f"已重连 tmux:{session}，继续监视进度"
                    store.save()
                    _poll_train_until_done(item, job_id)
                    return
                log_text = _read_text_at(
                    item.get("log_file") or "",
                    item.get("execution") or "remote",
                    item.get("host_id") or "cluster_0",
                    max_bytes=120_000,
                )
                code = train_backend.parse_train_exit_code(log_text)
                if code is not None and item.get("status") == "running":
                    _finalize_train_job(item, code=code)
                    return
                if item.get("status") == "running":
                    _finalize_train_job(item, code=None)
                return
            # Pre-tmux / 外部启动：无 session，只跟日志直到出现 exit 标记。
            item["message"] = (
                (item.get("message") or "训练中") + " · legacy(非tmux)，网页监视日志"
            )[:180]
            item["status"] = "running"
            store.save()
            _poll_train_until_done(item, job_id)
            return

        session = session or train_backend.train_tmux_session_name(job_id)
        item["tmux_session"] = session

        # Off-box: rsync selected data / manifests, recreate pack stage, then tmux.
        if train_backend.is_offbox_train_host(str(item.get("host_id") or "")):
            stage_train_job_to_host(item)

        env_map = train_backend.build_train_env(item, catalog)
        phi0_root = item["phi0_root"].rstrip("/")
        rel_script = str(item.get("script") or "").strip() or train_backend.launcher_script(catalog)
        pack = item.get("pack") if isinstance(item.get("pack"), dict) else None
        if pack and pack.get("mix_slots_mode"):
            rel_script = train_backend.MIX_SLOTS_PACK_LAUNCHER
        elif pack and not str(rel_script).endswith("run_studio_selection_pack_and_distill.sh"):
            rel_script = train_backend.SELECTION_PACK_LAUNCHER
        item["script"] = rel_script
        script = f"{phi0_root}/{str(rel_script).lstrip('/')}"
        if item.get("execution") == "local" and not Path(script).is_file():
            raise RuntimeError(f"本机缺少训练入口: {script}")
        if train_backend.is_offbox_train_host(str(item.get("host_id") or "")):
            host = host_by_id(str(item.get("host_id")))
            chk = run_command(
                ssh_args(host["target"], ["bash", "-lc", f"test -f {shlex.quote(script)}"]),
                timeout=20,
            )
            if chk.returncode:
                raise RuntimeError(f"远端缺少训练入口: {script}")
        inner = train_backend.build_train_inner_bash(
            phi0_root=phi0_root,
            script=script,
            env_map=env_map,
            out_dir=str(item["out_dir"]),
            log_file=str(item["log_file"]),
        )
        launch = train_backend.build_tmux_launch_bash(session, inner)
        item["command"] = launch
        item["message"] = (
            f"正在 tmux 启动 pack+蒸馏（{session}）…"
            if item.get("pack")
            else f"正在 tmux 启动蒸馏训练（{session}）…"
        )
        store.save()
        proc = _run_train_host_bash(item, launch, timeout=60)
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in out.splitlines():
            if line.strip():
                log_line(item, line)
        if proc.returncode != 0:
            raise RuntimeError(out.strip() or f"tmux 启动失败 rc={proc.returncode}")
        parsed = train_backend.parse_tmux_session_from_log(out)
        if parsed:
            item["tmux_session"] = parsed
            session = parsed
        with worker_lock:
            workers[job_id] = {"kind": "train", "tmux_session": session}
        item["message"] = f"tmux:{session} 训练中（网页崩溃不会中断）"
        store.save()
        _poll_train_until_done(item, job_id)
    except Exception as exc:  # noqa: BLE001
        item["status"] = "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
        if item.get("staged_to_host") and train_backend.is_offbox_train_host(str(item.get("host_id") or "")):
            try:
                pullback_train_job_from_host(item)
                item["message"] = f"{exc} · 已尽量回传产物"
            except Exception as pull_exc:  # noqa: BLE001
                log_line(item, f"[pullback] after-error failed: {pull_exc}")
    finally:
        store.save()
        with worker_lock:
            workers.pop(job_id, None)


def resume_running_train_monitors() -> None:
    """After Studio restart, re-attach monitors to any still-running tmux train jobs."""
    with store.lock:
        jobs = list(store.data.get("train_jobs") or [])
    for item in jobs:
        if item.get("status") != "running":
            continue
        job_id = str(item.get("id") or "")
        if not job_id:
            continue
        with worker_lock:
            if job_id in workers:
                continue
        threading.Thread(
            target=train_worker,
            args=(job_id,),
            kwargs={"resume_only": True},
            daemon=True,
        ).start()


def ssh_args_ephemeral(target: str, remote_args: list[str]) -> list[str]:
    """SSH without ControlMaster — for short train/infer file tails on cluster_0.

    Avoids one stuck multiplexed session blocking metrics/log polls.
    Still fully remote (never reads host FS from Studio process).
    """
    args = ["ssh"]
    if SSH_CONFIG.is_file():
        args += ["-F", str(SSH_CONFIG)]
    return args + [
        "-o", "BatchMode=yes",
        "-o", "ClearAllForwardings=yes",
        "-o", "ControlMaster=no",
        "-o", "ConnectTimeout=8",
        "--", target, shlex.join(remote_args),
    ]


def _read_text_at(
    path: str,
    execution: str,
    host_id: str,
    *,
    max_bytes: int | None = 2_000_000,
    span: str = "tail",
) -> str:
    """Read file bytes according to execution mode.

    - local: Studio machine FS
    - remote (03 默认): always SSH to host_id (cluster_0), never short-circuit to local FS

    ``span``:
      - ``tail`` (default): keep the last ``max_bytes`` when truncating
      - ``full``: read the whole file (``max_bytes`` ignored / only soft remote cap)
    """
    if not path:
        return ""
    if execution == "local":
        p = Path(path)
        if not p.is_file():
            return ""
        data = p.read_bytes()
        if max_bytes is not None and len(data) > max_bytes and span != "full":
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    host = host_by_id(host_id)
    if span == "full":
        # Cap remote transfer; prefer head+tail merge so charts keep early + late steps.
        soft = int(max_bytes or 32_000_000)
        half = max(soft // 2, 1)
        py = (
            "import pathlib;p=pathlib.Path(%r);"
            "d=p.read_bytes() if p.is_file() else b'';"
            "soft=%d;half=%d;"
            "out=d if len(d)<=soft else (d[:half]+b'\\n#...truncated...\\n'+d[-half:]);"
            "print(out.decode('utf-8','replace'), end='')"
        ) % (path, soft, half)
    else:
        soft = int(max_bytes or 2_000_000)
        # Tail via remote python to avoid huge transfers
        py = (
            "import pathlib;p=pathlib.Path(%r);d=p.read_bytes() if p.is_file() else b'';"
            "print(d[-%d:].decode('utf-8','replace'), end='')"
        ) % (path, soft)
    try:
        proc = run_command(
            ssh_args_ephemeral(host["target"], ["python3", "-c", py]),
            timeout=25,
        )
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout if proc.returncode == 0 else ""


@app.get("/api/train/catalog")
def get_train_catalog():
    try:
        catalog = train_backend.enrich_catalog_for_ui(merged_train_catalog())
        cfg = train_config()
        return jsonify({
            "ok": True,
            "catalog": catalog,
            "config": cfg,
            "jobs": store.data.get("train_jobs") or [],
            "skills": catalog.get("skills") or skill_backend.list_skills(skills_root_path()),
            "hosts": store.data.get("hosts") or [],
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/train/mix")
def build_train_mix():
    """Combine selected skill/recipe cards into a temporary mix train task."""
    data = request.get_json(silent=True) or {}
    try:
        ids = data.get("skill_ids") or data.get("ids") or []
        if not isinstance(ids, list) or len(ids) < 2:
            raise ValueError("请至少选择 2 个技能进行混合训练")
        catalog = merged_train_catalog()
        root = skills_root_path()
        skills = []
        for sid in ids:
            sid = str(sid).strip()
            sk = skill_backend.load_skill(root, sid)
            if sk:
                skills.append(sk)
                continue
            # Allow mixing catalog recipe cards even without a local skill folder.
            recipe = next((t for t in (catalog.get("tasks") or []) if t.get("id") == sid), None)
            if not recipe:
                raise ValueError(f"技能不存在: {sid}")
            skills.append({
                "id": recipe.get("id"),
                "title": recipe.get("title"),
                "badge": recipe.get("badge"),
                "prompt": recipe.get("default_prompt") or "",
                "description": recipe.get("description") or "",
                "datasets": list(recipe.get("datasets") or []),
            })
        task = skill_backend.mix_train_task(skills, catalog)
        return jsonify({"ok": True, "task": task})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/train/config")
def save_train_config():
    data = request.get_json(silent=True) or {}
    try:
        cfg = train_config()
        if data.get("execution_mode") in {"local", "remote", "auto"}:
            cfg["execution_mode"] = data["execution_mode"]
        if data.get("host_id"):
            host = host_by_id(str(data["host_id"]))
            cfg["host_id"] = host["id"]
            if cfg.get("execution_mode") == "local" and host["id"] != train_backend.LOCAL_TRAIN_HOST_ID:
                cfg["execution_mode"] = "remote"
        if data.get("phi0_root"):
            cfg["phi0_root"] = require_abs(data["phi0_root"], "Phi0 根目录")
        with store.lock:
            store.data["train_config"] = cfg
            store.save()
        return jsonify({"ok": True, "config": cfg})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/train/tasks")
def upsert_train_task():
    """Create or update a custom skill training card (persisted in studio.json)."""
    data = request.get_json(silent=True) or {}
    try:
        catalog = train_backend.load_train_catalog()
        task = train_backend.normalize_custom_task(data, catalog)
        reserved = {t.get("id") for t in (catalog.get("tasks") or [])}
        if task["id"] in reserved:
            raise ValueError("不能覆盖内置任务卡片")
        with store.lock:
            cfg = store.data.setdefault("train_config", default_state()["train_config"])
            cfg.setdefault("custom_tasks", [])
            tasks = list(cfg.get("custom_tasks") or [])
            idx = next((i for i, t in enumerate(tasks) if t.get("id") == task["id"]), None)
            if idx is None:
                task["created_at"] = task.get("updated_at")
                tasks.append(task)
            else:
                task["created_at"] = tasks[idx].get("created_at") or tasks[idx].get("updated_at")
                tasks[idx] = task
            cfg["custom_tasks"] = tasks
            store.data["train_config"] = cfg
            store.save()
        return jsonify({
            "ok": True,
            "task": task,
            "catalog": train_backend.enrich_catalog_for_ui(
                train_backend.merge_catalog_tasks(catalog, tasks)
            ),
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.delete("/api/train/tasks/<task_id>")
def delete_train_task(task_id: str):
    """Remove a train card from the 03 grid.

    - custom_* → drop from studio custom_tasks
    - builtin recipe → hide via hidden_task_ids (does not edit train_catalog.json)
    """
    try:
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("缺少任务 id")
        catalog = train_backend.load_train_catalog()
        reserved = {str(t.get("id")) for t in (catalog.get("tasks") or []) if t.get("id")}
        with store.lock:
            cfg = store.data.setdefault("train_config", default_state()["train_config"])
            cfg.setdefault("custom_tasks", [])
            cfg.setdefault("hidden_task_ids", [])
            before = list(cfg.get("custom_tasks") or [])
            tasks = [t for t in before if str(t.get("id")) != task_id]
            removed_custom = len(tasks) < len(before)
            cfg["custom_tasks"] = tasks
            hidden = [str(x) for x in (cfg.get("hidden_task_ids") or []) if str(x).strip()]
            hid = False
            if task_id in reserved and task_id not in hidden:
                hidden.append(task_id)
                hid = True
            elif removed_custom:
                pass
            elif task_id in hidden:
                raise ValueError("该卡片已从列表移除")
            else:
                # Not custom and not a known builtin — still allow hide so mix/ephemeral
                # cards and stale ids can be cleaned from the UI.
                hidden.append(task_id)
                hid = True
            cfg["hidden_task_ids"] = hidden
            store.data["train_config"] = cfg
            store.save()
        return jsonify({
            "ok": True,
            "deleted": task_id,
            "removed_custom": removed_custom,
            "hidden": hid or task_id in reserved,
            "catalog": train_backend.enrich_catalog_for_ui(merged_train_catalog()),
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/train/tasks/restore-hidden")
def restore_hidden_train_tasks():
    """Restore hidden builtin recipes into the 03 card grid."""
    data = request.get_json(silent=True) or {}
    try:
        ids = data.get("ids")
        with store.lock:
            cfg = store.data.setdefault("train_config", default_state()["train_config"])
            hidden = [str(x) for x in (cfg.get("hidden_task_ids") or []) if str(x).strip()]
            if ids is None:
                restored = list(hidden)
                hidden = []
            else:
                want = {str(x).strip() for x in (ids or []) if str(x).strip()}
                restored = [x for x in hidden if x in want]
                hidden = [x for x in hidden if x not in want]
            cfg["hidden_task_ids"] = hidden
            store.data["train_config"] = cfg
            store.save()
        return jsonify({
            "ok": True,
            "restored": restored,
            "catalog": train_backend.enrich_catalog_for_ui(merged_train_catalog()),
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/train/datasets/probe")
def probe_train_dataset_api():
    data = request.get_json(silent=True) or {}
    try:
        path = require_abs(data.get("dataset_path", ""), "数据集路径")
        return jsonify({"ok": True, **train_backend.probe_train_dataset(path)})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


REMOTE_BROWSE_DATASETS = r'''
import json, os, sys
root = os.path.realpath(sys.argv[1])
out = {"root": root, "folders": []}
if not os.path.isdir(root):
    print(json.dumps({"ok": False, "error": "root missing", "root": root, "folders": []}))
    raise SystemExit(0)

def is_dataset(path):
    return os.path.isfile(os.path.join(path, "meta", "info.json"))

def read_prompt(path):
    tasks = os.path.join(path, "meta", "tasks.jsonl")
    if os.path.isfile(tasks):
        try:
            with open(tasks, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    t = str(row.get("task") or "").strip()
                    if t:
                        return t
        except Exception:
            pass
    return ""

def episode_count(path):
    info = os.path.join(path, "meta", "info.json")
    try:
        with open(info, encoding="utf-8") as f:
            return int(json.load(f).get("total_episodes") or 0)
    except Exception:
        return 0

# Direct children of root: skill folders (or datasets)
for name in sorted(os.listdir(root)):
    if name.startswith("."):
        continue
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        continue
    if is_dataset(path):
        out["folders"].append({
            "name": name, "path": path, "kind": "dataset",
            "prompt": read_prompt(path), "total_episodes": episode_count(path),
            "children": [],
        })
        continue
    children = []
    try:
        kids = sorted(os.listdir(path))
    except OSError:
        kids = []
    for kid in kids:
        if kid.startswith(".") or kid.endswith(".json") or kid.endswith(".lock"):
            continue
        cpath = os.path.join(path, kid)
        if not os.path.isdir(cpath):
            continue
        if is_dataset(cpath):
            children.append({
                "name": kid, "path": cpath, "kind": "dataset",
                "prompt": read_prompt(cpath), "total_episodes": episode_count(cpath),
            })
    out["folders"].append({
        "name": name, "path": path, "kind": "skill",
        "children": children, "child_count": len(children),
    })
print(json.dumps({"ok": True, **out}, ensure_ascii=False))
'''


@app.get("/api/train/browse-datasets")
def browse_train_datasets():
    """List LeRobot sessions under 830demo (or given root) as a folder tree."""
    root = str(request.args.get("root") or "").strip()
    if not root:
        root = str(
            (store.data.get("shared") or {}).get("root")
            or skills_config().get("remote_base")
            or DEFAULT_SHARED_ROOT
        ).strip()
    try:
        root = require_abs(root, "数据集根目录")
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))
    # Prefer local if mounted; else SSH via train/shared host.
    if Path(root).is_dir():
        proc = run_command(["python3", "-c", REMOTE_BROWSE_DATASETS, root], timeout=60)
    else:
        try:
            cfg = train_config()
            host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
        except Exception:
            try:
                _, host, _ = shared_config()
            except Exception as exc:  # noqa: BLE001
                return api_error(f"无法浏览远端目录: {exc}", 502)
        proc = run_command(
            ssh_args(host["target"], ["python3", "-c", REMOTE_BROWSE_DATASETS, root]),
            timeout=90,
        )
    if proc.returncode:
        return api_error((proc.stderr or proc.stdout or "浏览失败").strip()[:500], 502)
    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return api_error(f"浏览结果无法解析: {exc}", 502)
    if not payload.get("ok", True):
        return api_error(payload.get("error") or "浏览失败", 502)
    return jsonify(payload)


@app.post("/api/train/jobs")
def start_train_job():
    data = request.get_json(silent=True) or {}
    try:
        catalog = merged_train_catalog()
        cfg = train_config()
        task_id = str(data.get("task_id") or "")
        task = next((t for t in catalog.get("tasks") or [] if t.get("id") == task_id), None)
        if not task:
            raise ValueError("未知训练任务卡片")
        if task.get("train_blocked"):
            raise ValueError(str(task.get("train_block_reason") or "该技能卡当前不可训练"))
        train_profile = str(
            data.get("train_profile")
            or (data.get("params") or {}).get("train_profile")
            or task.get("train_profile")
            or ""
        ).strip()
        train_profile = train_backend.normalize_train_profile(train_profile)
        params = dict(task.get("defaults") or {})
        params.update(data.get("params") or {})
        params["train_profile"] = train_profile
        mix_slots_in = data.get("mix_slots") or params.get("mix_slots") or task.get("mix_slots")
        if isinstance(mix_slots_in, list) and mix_slots_in and (
            task.get("mix") or train_profile == "mix_vision_isaac" or data.get("mix_compose")
        ):
            # New compose mix: per-skill prompt + datasets; shared hyperparams.
            stamp_early = time.strftime("%Y%m%d_%H%M%S")
            job_id_early = "train_" + uuid.uuid4().hex[:8]
            pack_plan = train_backend.plan_mix_slots_pack(
                mix_slots_in,
                stamp=stamp_early,
                job_id=job_id_early,
            )
            ref_root = require_abs(pack_plan["ref_root"], "数据集路径")
            params["mix_slots"] = pack_plan.get("slots") or mix_slots_in
            params["dataset_paths"] = list(pack_plan.get("session_paths") or [])
            params["prompt"] = str(pack_plan.get("task_prompt") or params.get("prompt") or "")
            params["ref_resolve"] = (
                f"mix_slots:{pack_plan['session_count']}sess/"
                f"valid{pack_plan['valid_count']}/skills{len(pack_plan.get('slots') or [])}"
            )
            params["pack_fingerprint"] = pack_plan.get("fingerprint")
            params["valid_count"] = pack_plan["valid_count"]
            params["invalid_excluded"] = pack_plan["invalid_count"]
            params["ephemeral_pack"] = False
            params["pack_root"] = pack_plan.get("pack_root")
            params["mix_compose"] = True
            episode_allowlist = ""
            labels_from_sel = ""
            selected = ""
            ref_note = params["ref_resolve"]
        else:
            pack_plan = None
            episode_allowlist = ""
            ref_note = ""
            labels_from_sel = ""
            job_id_early = ""
            multi = data.get("dataset_paths") or params.get("dataset_paths") or []
            if isinstance(multi, str):
                multi = [ln.strip() for ln in multi.replace(",", "\n").splitlines() if ln.strip()]
            multi = [require_abs(p, "数据集路径") for p in multi if str(p).strip()]
            if not multi:
                fallback = str(
                    data.get("dataset_path")
                    or data.get("ref_root")
                    or data.get("remote_dataset_path")
                    or task.get("ref_root")
                    or ""
                ).strip()
                if fallback:
                    multi = [require_abs(fallback, "数据集路径")]
            classified = train_backend.classify_train_selection(multi, train_profile=train_profile)
            params["dataset_paths"] = list(classified.get("paths") or multi)
            selected = params["dataset_paths"][0] if params["dataset_paths"] else ""

            if classified.get("needs_pack"):
                # Multi / raw session selection → pack ALL selected sessions' valid eps, then distill.
                stamp_early = time.strftime("%Y%m%d_%H%M%S")
                job_id_early = "train_" + uuid.uuid4().hex[:8]
                pack_plan = train_backend.plan_selection_pack(
                    classified["raw_sessions"],
                    skill_id=str(data.get("skill_id") or task.get("skill") or task_id),
                    prompt=str(params.get("prompt") or task.get("default_prompt") or ""),
                    stamp=stamp_early,
                    job_id=job_id_early,
                )
                ref_root = require_abs(pack_plan["ref_root"], "数据集路径")
                ref_note = (
                    f"pack_valid_only:{pack_plan['session_count']}sess/"
                    f"valid{pack_plan['valid_count']}/invalid_excluded{pack_plan['invalid_count']}"
                )
                params["selection_path"] = ",".join(pack_plan.get("sessions") or [])
                params["ref_resolve"] = ref_note
                params["pack_fingerprint"] = pack_plan["fingerprint"]
                params["valid_count"] = pack_plan["valid_count"]
                params["invalid_excluded"] = pack_plan["invalid_count"]
                params["ephemeral_pack"] = False
                params["pack_root"] = pack_plan.get("pack_root")
                if pack_plan.get("skipped_empty_valid"):
                    params["skipped_empty_valid"] = pack_plan["skipped_empty_valid"]
            else:
                selected = str(classified.get("ref_root") or multi[0])
                # Existing *_unified (or mix pack): keep as REF; train only screened valid.
                ref_root, labels_from_sel, ref_note = skill_backend.resolve_train_ref_from_selection(
                    selected=selected,
                    task=task,
                    remote_dataset_path=str(data.get("remote_dataset_path") or ""),
                    train_profile=train_profile,
                )
                ref_root = require_abs(ref_root, "数据集路径")
                if not Path(ref_root).is_dir():
                    raise ValueError(
                        f"REF 路径不存在: {ref_root}。"
                        "请勾选 830demo 下的 raw session（将 pack 到 Phi_0_train_data），"
                        "或选择一个真实存在的 *_unified"
                    )
                if ref_note:
                    params["selection_path"] = selected
                    params["ref_resolve"] = ref_note
                if labels_from_sel and not data.get("labels_path") and not data.get("labels_json"):
                    data = {**data, "labels_path": labels_from_sel}

        # NOTE: legacy block below was inlined above; keep host resolution next.

        mode = str(data.get("execution_mode") or cfg.get("execution_mode") or "remote")
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        # UI may send host_id via execution_mode dropdown values like "h20-0".
        if mode in train_backend.OFFBOX_TRAIN_HOST_IDS or mode == train_backend.LOCAL_TRAIN_HOST_ID:
            host_id = mode
            mode = "remote"
        if mode == "local":
            host_id = train_backend.LOCAL_TRAIN_HOST_ID
        host = host_by_id(host_id)
        if mode != "local":
            train_backend.require_train_ready_host(host)
        # One running job per train host (different SSH nodes may run in parallel).
        with store.lock:
            busy = [
                j
                for j in (store.data.get("train_jobs") or [])
                if j.get("status") == "running"
                and str(j.get("host_id") or "") == str(host_id)
            ]
        if busy:
            other = busy[0]
            raise ValueError(
                f"主机 {host.get('name') or host_id} 上已有训练在跑"
                f"（{other.get('id')} · {other.get('task_title') or other.get('task_id') or ''}）。"
                f"请换其他 SSH 节点，或先停止该任务后再启动。"
            )
        local_root = train_backend.resolve_local_phi0_root(catalog)
        if mode == "auto":
            mode = "local" if local_root and (pack_plan or Path(ref_root).is_dir()) else "remote"

        if pack_plan:
            # Pack creates REF; skip preflight file checks on a path that does not exist yet.
            if mode == "local":
                phi0_root = str(data.get("phi0_root") or local_root or cfg.get("phi0_root") or "")
                if not phi0_root:
                    raise ValueError("本机未找到 Phi_0_wpy（缺少 run_online_vlm_mix_distill.sh）")
            else:
                phi0_root = str(
                    data.get("phi0_root")
                    or host.get("phi0_root")
                    or cfg.get("phi0_root")
                    or catalog.get("phi0_root_remote")
                )
                host_by_id(host_id)
            train_backend.validate_train_ref(
                ref_root,
                train_profile=train_profile,
                require_local_files=False,
            )
        else:
            # Always validate REF semantics (mix vs single) even for remote paths we can see locally.
            train_backend.validate_train_ref(
                ref_root,
                train_profile=train_profile,
                require_local_files=Path(ref_root).is_dir(),
            )
            if mode == "local":
                phi0_root = str(data.get("phi0_root") or local_root or cfg.get("phi0_root") or "")
                if not phi0_root:
                    raise ValueError("本机未找到 Phi_0_wpy（缺少 run_online_vlm_mix_distill.sh）")
                probe = train_backend.probe_train_dataset(ref_root)
                if not probe["ready"]:
                    missing = [
                        k for k, ok in (probe.get("checks") or {}).items()
                        if not ok and k in train_backend.READY_FILES
                    ]
                    raise ValueError("数据集未就绪，缺少: " + ", ".join(missing[:6]))
                if train_backend.is_mix_train_profile(train_profile) and not probe.get("mix_ready"):
                    raise ValueError("混训 REF 缺少 meta/all_episode_allowlist.json")
            else:
                phi0_root = str(
                    data.get("phi0_root")
                    or host.get("phi0_root")
                    or cfg.get("phi0_root")
                    or catalog.get("phi0_root_remote")
                )
                host_by_id(host_id)  # validate
                if not (
                    skill_backend.is_unified_dataset({"path": ref_root})
                    or skill_backend.is_mix_unified_dataset({"path": ref_root})
                ):
                    for ds in task.get("datasets") or []:
                        if ds.get("path") == ref_root and ds.get("remote_path"):
                            ref_root = ds["remote_path"]
                            break
                    if data.get("remote_dataset_path"):
                        remote_p = str(data.get("remote_dataset_path") or "")
                        if skill_backend.is_unified_dataset({"path": remote_p}) or skill_backend.is_mix_unified_dataset(
                            {"path": remote_p}
                        ):
                            ref_root = require_abs(remote_p, "远端数据集路径")
                train_backend.validate_train_ref(
                    ref_root,
                    train_profile=train_profile,
                    require_local_files=Path(ref_root).is_dir(),
                )

            # Existing unified: force distill allowlist to 02-screened valid only (never invalid).
            labels_path = str(
                data.get("labels_path") or data.get("labels_json") or task.get("valid_json") or ""
            ).strip()
            screened = train_backend.read_screened_valid(ref_root) if Path(ref_root).is_dir() else {
                "valid": [], "screened": False
            }
            if not (screened.get("screened") and screened.get("valid")):
                # Try labels on originally selected session / labels_path.
                if selected and Path(selected).is_dir() and Path(selected).resolve() != Path(ref_root).resolve():
                    sel = train_backend.read_screened_valid(selected)
                    if not sel.get("valid") and labels_path and Path(labels_path).is_file():
                        do_export_valid_allowlist(
                            dataset_path=selected,
                            host_id=host_id,
                            source="labels",
                            labels_path=labels_path,
                        )
                        sel = train_backend.read_screened_valid(selected)
                    if sel.get("valid"):
                        screened = {**sel, "screened": True}
                if not (screened.get("valid") and screened.get("screened")):
                    if labels_path and Path(labels_path).is_file():
                        export_target = selected if (
                            selected and Path(selected).is_dir() and "unified" not in selected.lower()
                        ) else ref_root
                        do_export_valid_allowlist(
                            dataset_path=export_target,
                            host_id=host_id,
                            source="labels",
                            labels_path=labels_path,
                        )
                        screened = train_backend.read_screened_valid(export_target)
            if screened.get("screened") and screened.get("valid"):
                allow_path = APP_DIR / "state" / "train_allowlists" / (
                    f"{Path(ref_root).name}_valid_only.json"
                )
                episode_allowlist = train_backend.write_episode_allowlist(
                    screened["valid"], allow_path
                )
                params["valid_count"] = len(screened["valid"])
                params["invalid_excluded"] = len(screened.get("invalid") or [])
                params["allowlist_source"] = screened.get("source") or "screened_valid"
            elif screened.get("valid") and not screened.get("screened"):
                # Pack vision allowlist without 02 labels — refuse; invalid may be mixed in.
                if not train_backend.is_mix_train_profile(train_profile):
                    raise ValueError(
                        "当前 *_unified 只有打包全量 allowlist，尚未经 02 筛选。"
                        "请先在 02 标注并「导出 valid」，或勾选已标注的 raw session 由系统按 valid 重新 pack"
                    )
                # Mix packs are pre-filtered at pack time.
                pass
            else:
                train_backend.require_screened_valid(ref_root, what="训练")

        stamp = str((pack_plan or {}).get("stamp") or time.strftime("%Y%m%d_%H%M%S"))
        num_envs = int(params.get("num_envs") or 32)
        horizon = int(params.get("horizon") or 32)
        epochs = int(params.get("epochs") or 0)
        if params.get("ckpt_every_epoch") is None:
            params["ckpt_every_epoch"] = int(
                task.get("defaults", {}).get("ckpt_every_epoch")
                if task.get("defaults", {}).get("ckpt_every_epoch") is not None
                else 1
            )
        if int(params.get("ckpt_every_epoch") or 0) > 0:
            params["ckpt_every"] = 0
        elif not params.get("ckpt_every"):
            params["ckpt_every"] = int(task.get("defaults", {}).get("ckpt_every") or 10000)
        # Align with official distill script: never set STUDENT_CKPT / EXTRA_STEPS.
        # Train action expert from scratch for Epochs; freeze VLM via catalog fixed_flags.
        if epochs <= 0:
            raise ValueError("请设置 Epochs")
        params["epochs"] = epochs
        params["train_steps"] = 0
        params["extra_steps"] = 0
        params.pop("student_ckpt", None)
        params["init_from_base"] = False
        student_ckpt = ""
        extra_steps = 0
        # NGPU always follows CUDA device list (never a separate mismatched knob).
        if not params.get("cuda_devices") and task.get("defaults", {}).get("cuda_devices"):
            params["cuda_devices"] = task["defaults"]["cuda_devices"]
        cuda_raw = str(params.get("cuda_devices") or "").strip()
        cuda_ids = [x.strip() for x in cuda_raw.split(",") if x.strip()]
        if not cuda_ids:
            fallback_n = int(params.get("ngpu") or task.get("defaults", {}).get("ngpu") or 8)
            cuda_ids = [str(i) for i in range(max(1, fallback_n))]
        params["cuda_devices"] = ",".join(cuda_ids)
        ngpu = len(cuda_ids)
        params["ngpu"] = ngpu
        tag = f"{task_id}_h{horizon}_b{num_envs}_ddp{ngpu}_e{epochs}"
        skill_key = re.sub(
            r"[^A-Za-z0-9_-]+",
            "",
            str(data.get("skill_id") or task.get("skill") or task_id),
        )[:40] or "skill"
        if mode == "local":
            out_base = str(Path(ref_root).parent / "train_runs")
            log_base = str(APP_DIR / "state" / "train_logs")
        else:
            out_base = str(host.get("model_zoo") or train_backend.DEFAULT_TRAIN_OUT_BASE)
            log_base = f"{phi0_root.rstrip('/')}/logs"
        out_dir = train_backend.allocate_train_out_dir(
            out_base=out_base,
            skill_key=skill_key,
            stamp=stamp,
            requested=str(data.get("out_dir") or ""),
        )
        log_file = str(data.get("log_file") or f"{log_base}/{tag}_{stamp}.log")
        if not params.get("prompt") or str(params.get("prompt") or "").strip().lower() in {
            "demo", "task", "none", "null", "n/a", "-",
        }:
            params["prompt"] = (
                train_backend.resolve_train_prompt(
                    params.get("prompt"),
                    task.get("default_prompt"),
                    train_backend.read_dataset_prompt(data.get("dataset_path") or ref_root),
                )
                or task.get("default_prompt")
                or ""
            )
        if pack_plan and (
            not pack_plan.get("task_prompt")
            or str(pack_plan.get("task_prompt") or "").strip().lower() in {"demo", "task", "none", "null", "n/a", "-"}
        ):
            pack_plan["task_prompt"] = params["prompt"] or "task"

        if pack_plan and pack_plan.get("mix_slots_mode"):
            launcher = train_backend.MIX_SLOTS_PACK_LAUNCHER
            msg = (
                f"排队：[{host_id}] mix {len(pack_plan.get('slots') or [])} 技能 / "
                f"valid {pack_plan['valid_count']} → 分技能 pack+merge → distill"
            )
        elif pack_plan:
            launcher = train_backend.SELECTION_PACK_LAUNCHER
            msg = (
                f"排队：[{host_id}] pack {pack_plan['session_count']} sessions / "
                f"valid {pack_plan['valid_count']} → distill"
            )
        else:
            launcher = train_backend.launcher_script(catalog)
            msg = f"排队启动训练（{train_profile}）@ {host_id}"

        item = {
            "id": job_id_early or ("train_" + uuid.uuid4().hex[:8]),
            "kind": "train",
            "task_id": task_id,
            "skill_id": str(data.get("skill_id") or task.get("skill") or task_id),
            "task_title": task.get("title"),
            "script": launcher,
            "train_profile": train_profile,
            "fixed_flags": dict(task.get("fixed_flags") or {}),
            "execution": mode,
            "host_id": host_id,
            "phi0_root": phi0_root,
            "ref_root": ref_root,
            "out_dir": out_dir,
            "log_file": log_file,
            "metrics_jsonl": f"{out_dir.rstrip('/')}/distill_metrics.jsonl",
            "metrics_json": f"{out_dir.rstrip('/')}/distill_metrics.json",
            # Hide prior runs that appended into the same out_dir metrics file.
            # Off-box: size is queried on the train host (usually 0 for a fresh out_dir).
            "metrics_baseline_bytes": _file_size_at(
                f"{out_dir.rstrip('/')}/distill_metrics.jsonl",
                mode,
                host_id,
            ),
            "student_ckpt": None,
            "params": params,
            "stamp": stamp,
            "use_wrapper": False,
            "status": "running",
            "message": msg,
            "created_at": now_iso(),
            "logs": [],
            "effective_batch": ngpu * num_envs,
            "tmux_session": "",
            "pack": pack_plan,
            "episode_allowlist": episode_allowlist or None,
        }
        item["tmux_session"] = train_backend.train_tmux_session_name(item["id"])
        # Persist launch params onto skill before job runs
        try:
            sid = str(item["skill_id"] or "").strip()
            if sid and skill_backend.load_skill(skills_root_path(), sid):
                skill_backend.bind_train_artifacts(
                    sid,
                    out_dir=out_dir,
                    ref_root=ref_root,
                    defaults={
                        k: params[k]
                        for k in (
                            "epochs", "num_envs", "ngpu", "lr", "warmup_ratio",
                            "ckpt_every", "ckpt_every_epoch", "ckpt_step_keep", "hand_mode", "horizon", "lr_scheduler",
                            "deploy_policy", "cuda_devices",
                        )
                        if k in params
                    },
                    prompt=str(params.get("prompt") or ""),
                    root=skills_root_path(),
                )
                # Remember which dataset was used for train
                preferred = Path(str(data.get("dataset_path") or ref_root)).name
                if preferred:
                    skill_backend.update_skill_fields(
                        sid, {"preferred_dataset_id": preferred}, root=skills_root_path()
                    )
        except Exception:  # noqa: BLE001
            pass
        with store.lock:
            store.data.setdefault("train_jobs", [])
            store.data["train_jobs"].insert(0, item)
            store.data["train_jobs"] = store.data["train_jobs"][:40]
            store.save()
        threading.Thread(target=train_worker, args=(item["id"],), daemon=True).start()
        return jsonify({"ok": True, "job": item})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/train/jobs")
def list_train_jobs():
    with store.lock:
        jobs = list(store.data.get("train_jobs") or [])
    for j in jobs:
        with worker_lock:
            monitor = j["id"] in workers
        tmux_live = False
        if j.get("status") == "running" and j.get("tmux_session"):
            try:
                tmux_live = train_tmux_alive(j)
            except Exception:  # noqa: BLE001
                tmux_live = False
        j["worker_alive"] = monitor or tmux_live
        j["tmux_alive"] = tmux_live
    return jsonify({"ok": True, "jobs": jobs})


@app.get("/api/train/jobs/<item_id>")
def get_train_job(item_id: str):
    item = store.find("train_jobs", item_id)
    if not item:
        return api_error("训练任务不存在", 404)
    with worker_lock:
        monitor = item_id in workers
    tmux_live = False
    if item.get("status") == "running" and item.get("tmux_session"):
        try:
            tmux_live = train_tmux_alive(item)
        except Exception:  # noqa: BLE001
            tmux_live = False
    return jsonify({
        "ok": True,
        "job": item,
        "worker_alive": monitor or tmux_live,
        "tmux_alive": tmux_live,
    })


@app.post("/api/train/jobs/<item_id>/cancel")
def cancel_train_job(item_id: str):
    item = store.find("train_jobs", item_id)
    if not item:
        return api_error("训练任务不存在", 404)
    data = request.get_json(silent=True) or {}
    keep_vlm = bool(
        data.get("keep_vlm_cache")
        or data.get("keep_pack")
        or request.args.get("keep_vlm_cache")
        or request.args.get("keep_pack")
    )
    item["status"], item["message"] = "cancelled", "已强制停止训练"
    pack = item.get("pack") if isinstance(item.get("pack"), dict) else None
    if keep_vlm and isinstance(pack, dict):
        pack["keep_vlm_cache"] = True
        pack["retained_for_reuse"] = True
        item["pack"] = pack
        item["message"] = "已强制停止训练（已保留 VLM cache / 训练数据包）"
    store.save()
    # Force-kill detached tmux + torchrun/DDP orphans (C-c alone is not enough).
    session = str(item.get("tmux_session") or "").strip() or train_backend.train_tmux_session_name(item_id)
    out_dir = str(item.get("out_dir") or "").strip()
    try:
        kill_bash = train_backend.build_tmux_kill_bash(
            session, out_dir=out_dir, job_id=item_id,
        )
        proc = _run_train_host_bash(item, kill_bash, timeout=45)
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        log_line(item, f"[cancel] force kill {session} rc={proc.returncode} {out[-200:]}")
    except Exception as exc:  # noqa: BLE001
        log_line(item, f"[cancel] force kill failed: {exc}")
    with worker_lock:
        slot = workers.get(item_id)
        proc = slot.get("proc") if slot else None
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    pack = item.get("pack") if isinstance(item.get("pack"), dict) else None
    if pack:
        log_line(item, f"[pack-keep] cancel 保留训练数据包 · root={pack.get('pack_root')}")
    # Off-box: async pullback of any ckpts / pack already written on the train host.
    if item.get("staged_to_host") and train_backend.is_offbox_train_host(str(item.get("host_id") or "")):
        schedule_pullback_train_job(item, reason="cancel")
        item["message"] = "已强制停止训练 · 正在回传远端产物…"
    store.save()
    return jsonify({"ok": True, "tmux_session": session, "force_killed": True, "keep_vlm_cache": keep_vlm})


def train_job_uses_local_fs(item: dict[str, Any] | None) -> bool:
    """True when train artifacts are on the Studio host FS (cluster_0 / local).

    Off-box hosts (h20-*/cluster_2) write metrics/logs only on the remote machine
    until pullback — always read via SSH for live monitoring.
    """
    item = item or {}
    if str(item.get("execution") or "") == "local":
        return True
    return not train_backend.is_offbox_train_host(str(item.get("host_id") or ""))


def _file_size_at(path: str, execution: str, host_id: str) -> int:
    """Size of a file on the train execution host (0 if missing)."""
    if not path:
        return 0
    if execution == "local" or (
        execution == "remote" and not train_backend.is_offbox_train_host(host_id)
    ):
        return train_backend.metrics_file_size(path)
    host = host_by_id(host_id)
    py = (
        "import os,sys;"
        "p=sys.argv[1];"
        "print(os.path.getsize(p) if os.path.isfile(p) else 0)"
    )
    proc = run_command(ssh_args(host["target"], ["python3", "-c", py, path]), timeout=20)
    if proc.returncode:
        return 0
    try:
        return max(0, int((proc.stdout or "0").strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return 0


@app.get("/api/train/jobs/<item_id>/metrics")
def get_train_metrics(item_id: str):
    item = store.find("train_jobs", item_id)
    if not item:
        return api_error("训练任务不存在", 404)
    try:
        exec_mode = item.get("execution") or "remote"
        host_id = item.get("host_id") or "cluster_0"
        metrics_path = item.get("metrics_jsonl") or ""
        local_fs = train_job_uses_local_fs(item)
        baseline = item.get("metrics_baseline_bytes")
        if baseline is None:
            baseline = 0
        else:
            baseline = max(0, int(baseline))

        log_text = _read_text_at(
            item.get("log_file") or "",
            exec_mode,
            host_id,
            max_bytes=8_000_000,
            span="full",
        )
        has_distill = train_backend.log_has_distill_progress(log_text)

        # This job's log has no distill yet: never paint a reused out_dir's old curve.
        if not has_distill:
            if item.get("metrics_baseline_bytes") is None and metrics_path:
                # Lazy lock for jobs started before baseline existed.
                item["metrics_baseline_bytes"] = _file_size_at(metrics_path, exec_mode, host_id)
                store.save()
                baseline = int(item["metrics_baseline_bytes"] or 0)
            return jsonify({
                "ok": True,
                "points": [],
                "live": {},
                "summary": {},
                "n_points": 0,
                "n_chart": 0,
                "step_first": None,
                "step_last": None,
                "source": "waiting",
                "waiting_current": True,
                "host_id": host_id,
                "message": (
                    f"当前训练尚未进入 distill（host={host_id}，可能仍在 pack / VLM cache），"
                    "暂不显示旧 out_dir 曲线"
                ),
            })

        summary_text = _read_text_at(
            item.get("metrics_json") or "", exec_mode, host_id, max_bytes=200_000,
        )
        summary = {}
        if summary_text.strip():
            try:
                summary = json.loads(summary_text)
            except ValueError:
                summary = {}
        # Full-history parse for charts — but only bytes appended after this job started.
        rows: list[dict[str, Any]] = []
        source = "jsonl"
        if metrics_path and local_fs and Path(metrics_path).is_file():
            rows = train_backend.parse_metrics_jsonl_file(
                metrics_path, byte_offset=baseline,
            )
        else:
            jsonl_text = _read_text_at(
                metrics_path, exec_mode, host_id, max_bytes=64_000_000, span="full",
            )
            rows = train_backend.parse_metrics_jsonl_bytes(
                jsonl_text, byte_offset=baseline,
            )
        # Phi0 only flushes jsonl every ~flush_every steps; until then parse log bars.
        if not rows:
            log_rows, log_meta = train_backend.parse_distill_log_metrics(log_text)
            if log_rows:
                rows = log_rows
                source = "log"
                summary = {}  # ignore stale summary from previous run
                for k, v in log_meta.items():
                    if v is not None:
                        summary.setdefault(k, v)
                params = item.get("params") or {}
                epochs_n = int(params.get("epochs") or 0)
                bar_total = int(log_meta.get("max_steps") or summary.get("max_steps") or 0)
                if epochs_n > 0 and bar_total > 0:
                    summary["steps_per_epoch"] = max(1, bar_total // epochs_n)
                    summary.setdefault("epochs", epochs_n)
                    summary["max_steps"] = bar_total
            else:
                summary = {}
        elif baseline > 0:
            # New segment from this job — don't mix old summary counters.
            summary = {k: v for k, v in summary.items() if k in ("steps_per_epoch", "epochs", "max_steps")}
        n_raw = len(rows)
        chart_rows = train_backend.downsample_metric_rows(rows, 1200)
        chart = train_backend.smooth_metric_series(
            chart_rows,
            key=["loss", "loss_z", "loss_z_phys", "loss_hand", "loss_q_head", "loss_smpl"],
        )
        live = train_backend.summarize_metrics(
            rows, summary, params=item.get("params") or {},
            created_at=item.get("created_at"),
        )
        return jsonify({
            "ok": True,
            "points": chart,
            "live": live,
            "summary": summary,
            "n_points": n_raw,
            "n_chart": len(chart),
            "step_first": chart[0]["step"] if chart else None,
            "step_last": chart[-1]["step"] if chart else None,
            "source": source,
            "metrics_baseline_bytes": baseline,
            "waiting_current": False,
            "host_id": host_id,
            "remote_monitor": not local_fs,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


def _list_ckpts_at(out_dir: str, execution: str, host_id: str) -> list[dict[str, Any]]:
    # Off-box: always SSH. cluster_0 remote may use local FS (same machine as Studio).
    if execution == "local" or (
        execution == "remote" and not train_backend.is_offbox_train_host(host_id)
    ):
        return train_backend.list_student_ckpts(out_dir)
    host = host_by_id(host_id)
    py = (
        "import json,glob,os,sys,re;"
        "root=sys.argv[1];"
        "named=re.compile(r'.+_\\d{8}_\\d{6}(?:_s\\d+)?\\.pt$');"
        "paths=[];"
        "last=os.path.join(root,'phi0_student_last.pt');"
        "paths+=[last] if os.path.isfile(last) else [];"
        "snaps=[p for p in glob.glob(os.path.join(root,'*.pt')) "
        "if not p.endswith('_optim.pt') and os.path.basename(p)!='phi0_student_last.pt' "
        "and (os.path.basename(p).startswith('phi0_student_step') or named.match(os.path.basename(p)))];"
        "snaps.sort(key=lambda p: os.path.getmtime(p));"
        "paths+=snaps;"
        "info={};"
        "ip=os.path.join(root,'info.json');"
        "info=json.load(open(ip)) if os.path.isfile(ip) else {};"
        "by={c['name']:c for c in (info.get('checkpoints') or []) if isinstance(c,dict) and c.get('name')};"
        "out=[];"
        "["
        "out.append({**{'path':p,'name':os.path.basename(p),'size':os.path.getsize(p),"
        "'mtime':int(os.path.getmtime(p))}, **{k:by.get(os.path.basename(p),{}).get(k) "
        "for k in ('loss','loss_z','steps','epoch','saved_at') "
        "if by.get(os.path.basename(p),{}).get(k) is not None}})"
        " for p in paths];"
        "print(json.dumps(out))"
    )
    proc = run_command(ssh_args(host["target"], ["python3", "-c", py, out_dir]), timeout=30)
    if proc.returncode:
        return []
    try:
        data = json.loads(proc.stdout.strip() or "[]")
        return data if isinstance(data, list) else []
    except ValueError:
        return []


@app.get("/api/train/jobs/<item_id>/ckpts")
def get_train_ckpts(item_id: str):
    item = store.find("train_jobs", item_id)
    if not item:
        return api_error("训练任务不存在", 404)
    try:
        ckpts = _list_ckpts_at(
            item.get("out_dir") or "",
            item.get("execution") or "remote",
            item.get("host_id") or "cluster_0",
        )
        return jsonify({"ok": True, "ckpts": ckpts, "out_dir": item.get("out_dir")})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/ckpts/scan")
def scan_infer_ckpts():
    """Scan a directory for *.pt. source=local|remote (cluster_0)."""
    dir_path = str(request.args.get("dir") or "").strip()
    recursive = str(request.args.get("recursive") or "0").strip() in {"1", "true", "yes"}
    source = str(request.args.get("source") or "auto").strip().lower()
    if not dir_path:
        return api_error("请提供目录 dir")
    try:
        cfg = infer_config()
        host_id = str(cfg.get("host_id") or "cluster_0")
        is_local = Path(dir_path).expanduser().is_dir()
        if source == "remote" or (source == "auto" and not is_local):
            host = host_by_id(host_id)
            files = _list_remote_pt_files(host, dir_path, recursive=recursive)
            source_used = "remote"
        else:
            files = train_backend.list_pt_files(dir_path, recursive=recursive)
            source_used = "local"
        cfg["last_ckpt_dir"] = dir_path
        with store.lock:
            store.data["infer_config"] = cfg
            store.save()
        return jsonify({
            "ok": True,
            "dir": dir_path,
            "files": files,
            "count": len(files),
            "source": source_used,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


def _list_remote_pt_files(host: dict[str, Any], dir_path: str, *, recursive: bool = False) -> list[dict[str, Any]]:
    py = (
        "import json,os\n"
        "from pathlib import Path\n"
        f"root=Path({dir_path!r}).expanduser()\n"
        "if not root.is_dir():\n"
        "  raise SystemExit('目录不存在: '+str(root))\n"
        f"recursive={1 if recursive else 0}\n"
        "paths=[]\n"
        "if recursive:\n"
        "  paths=sorted(root.rglob('*.pt'))\n"
        "else:\n"
        "  paths=sorted(root.glob('*.pt'))\n"
        "  for sub in sorted(root.iterdir()):\n"
        "    if sub.is_dir(): paths.extend(sorted(sub.glob('*.pt')))\n"
        "seen=set(); out=[]\n"
        "for p in paths:\n"
        "  key=str(p)\n"
        "  if key in seen: continue\n"
        "  seen.add(key)\n"
        "  try: st=p.stat()\n"
        "  except OSError: continue\n"
        "  try: rel=str(p.relative_to(root))\n"
        "  except Exception: rel=p.name\n"
        "  out.append({'path':str(p),'name':p.name,'rel':rel,'size':st.st_size,'mtime':int(st.st_mtime),'remote':True})\n"
        "out.sort(key=lambda x:(-x['mtime'],x['name']))\n"
        "print(json.dumps(out))\n"
    )
    proc = run_command(ssh_args(host["target"], ["python3", "-c", py]), timeout=60)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout or "远端扫描失败").strip()[-400:])
    data = json.loads(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "[]")
    return data if isinstance(data, list) else []


# In-memory ckpt pull jobs for progress UI (04 下拉并选用).
_ckpt_pull_jobs: dict[str, dict[str, Any]] = {}
_ckpt_pull_lock = threading.Lock()


@app.post("/api/infer/ckpts/pull")
def pull_infer_ckpt():
    """Start rsync of a remote .pt into local cache; poll GET …/pull/<job_id> for progress."""
    data = request.get_json(silent=True) or {}
    try:
        remote_path = require_abs(data.get("path") or "", "远端 .pt 路径")
        if not remote_path.endswith(".pt"):
            raise ValueError("仅支持拉取 .pt 文件")
        cfg = infer_config()
        host = host_by_id(str(cfg.get("host_id") or "cluster_0"))
        cache = Path(str(cfg.get("local_ckpt_cache") or (APP_DIR / "cache" / "ckpts")))
        remote = Path(remote_path)
        dest_dir = cache / remote.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / remote.name
        expected = int(data.get("size") or 0)
        if dest.is_file() and expected > 0 and dest.stat().st_size == expected:
            cfg["last_ckpt_dir"] = str(dest_dir)
            with store.lock:
                store.data["infer_config"] = cfg
                store.save()
            return jsonify({
                "ok": True,
                "cached": True,
                "remote_path": remote_path,
                "local_path": str(dest),
                "size": dest.stat().st_size,
                "pct": 100,
                "status": "done",
            })

        job_id = f"ckptpull_{int(time.time() * 1000)}_{remote.name}"
        with _ckpt_pull_lock:
            job = {
                "id": job_id,
                "status": "running",
                "pct": 0,
                "bytes_done": 0,
                "bytes_total": expected,
                "speed": "",
                "eta": "",
                "remote_path": remote_path,
                "local_path": str(dest),
                "message": "开始下拉…",
                "error": "",
            }
            _ckpt_pull_jobs[job_id] = job

        def _worker() -> None:
            try:
                def _on_progress(parsed: dict[str, Any]) -> None:
                    with _ckpt_pull_lock:
                        j = _ckpt_pull_jobs.get(job_id)
                        if not j:
                            return
                        j["pct"] = int(parsed.get("upload_progress") or 0)
                        j["bytes_done"] = int(parsed.get("upload_bytes_done") or 0)
                        j["speed"] = str(parsed.get("upload_speed") or "")
                        j["eta"] = str(parsed.get("upload_eta") or "")
                        j["message"] = f"下拉中 {j['pct']}%"

                cmd = [
                    "rsync", "-az", "--partial", "--info=progress2",
                    *rsync_ssh(),
                    f"{host['target']}:{remote_path}",
                    str(dest),
                ]
                result = rsync_with_progress(cmd, on_progress=_on_progress)
                if result.returncode or not dest.is_file():
                    detail = (result.stderr or result.stdout or "").strip()[-500:] or f"rsync exit {result.returncode}"
                    raise RuntimeError(detail)
                cfg2 = infer_config()
                cfg2["last_ckpt_dir"] = str(dest_dir)
                with store.lock:
                    store.data["infer_config"] = cfg2
                    store.save()
                with _ckpt_pull_lock:
                    j = _ckpt_pull_jobs.get(job_id) or {}
                    j.update({
                        "status": "done",
                        "pct": 100,
                        "bytes_done": dest.stat().st_size,
                        "bytes_total": dest.stat().st_size,
                        "local_path": str(dest),
                        "size": dest.stat().st_size,
                        "message": "下拉完成",
                    })
                    _ckpt_pull_jobs[job_id] = j
            except Exception as exc:  # noqa: BLE001
                with _ckpt_pull_lock:
                    j = _ckpt_pull_jobs.get(job_id) or {"id": job_id}
                    j.update({"status": "error", "error": str(exc), "message": str(exc)})
                    _ckpt_pull_jobs[job_id] = j

        threading.Thread(target=_worker, daemon=True, name=f"ckpt-pull-{job_id}").start()
        return jsonify({
            "ok": True,
            "async": True,
            "job_id": job_id,
            "remote_path": remote_path,
            "local_path": str(dest),
            "status": "running",
            "pct": 0,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/ckpts/pull/<job_id>")
def pull_infer_ckpt_status(job_id: str):
    with _ckpt_pull_lock:
        job = dict(_ckpt_pull_jobs.get(job_id) or {})
    if not job:
        return api_error("下拉任务不存在或已过期", 404)
    return jsonify({"ok": True, **job})


@app.get("/api/train/ckpts/probe")
def probe_train_ckpts():
    """List student ckpts under an out_dir (local or remote)."""
    out_dir = str(request.args.get("out_dir") or "").strip()
    if not out_dir:
        return api_error("缺少 out_dir")
    execution = str(request.args.get("execution") or "local")
    host_id = str(request.args.get("host_id") or "cluster_0")
    try:
        require_abs(out_dir, "输出目录")
        ckpts = _list_ckpts_at(out_dir, execution, host_id)
        return jsonify({"ok": True, "ckpts": ckpts, "out_dir": out_dir})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/train/jobs/<item_id>/logs")
def get_train_logs(item_id: str):
    item = store.find("train_jobs", item_id)
    if not item:
        return api_error("训练任务不存在", 404)
    try:
        file_tail = _read_text_at(item.get("log_file") or "", item.get("execution") or "remote",
                                  item.get("host_id") or "cluster_0", max_bytes=120_000)
        mem = "\n".join((item.get("logs") or [])[-200:])
        text = file_tail.strip() or mem
        return jsonify({"ok": True, "text": text, "log_file": item.get("log_file")})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


# ---------------------------------------------------------------------------
# Infer (06) — MuJoCo closed-loop via run_830_walk_student_cl_mujoco_viz.sh
# ---------------------------------------------------------------------------

import infer_backend  # noqa: E402


def infer_config() -> dict[str, Any]:
    with store.lock:
        cfg = store.data.setdefault("infer_config", default_state()["infer_config"])
        cfg.setdefault("skills", [])
        # Product default: MuJoCo / policy on cluster_0; Studio only plays preview/video.
        cfg["execution_mode"] = "remote"
        cfg.setdefault("last_ckpt_dir", "")
        cfg.setdefault("remote_ckpt_base", train_backend.DEFAULT_TRAIN_OUT_BASE)
        if str(cfg.get("remote_ckpt_base") or "").rstrip("/") == train_backend.LEGACY_TRAIN_OUT_BASE:
            cfg["remote_ckpt_base"] = train_backend.DEFAULT_TRAIN_OUT_BASE
        cfg.setdefault("local_ckpt_cache", str(APP_DIR / "cache" / "ckpts"))
        cfg.setdefault(
            "remote_phi0_py",
            "/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python",
        )
        if not cfg.get("host_id"):
            cfg["host_id"] = "cluster_0"
        if not cfg.get("phi0_root"):
            cfg["phi0_root"] = "/mnt/data2/wpy/workspace/Phi_0_wpy"
        return dict(cfg)


def merged_infer_catalog() -> dict[str, Any]:
    catalog = infer_backend.load_infer_catalog()
    builtins = list(catalog.get("skills") or [])
    builtin_ids = {s.get("id") for s in builtins if s.get("id")}
    uni_raw = skill_backend.list_skills(skills_root_path())
    uni = []
    for s in uni_raw:
        recipe = skill_backend.match_infer_recipe(s, catalog)
        card = skill_backend.skill_as_infer_skill(s, recipe)
        # Prefer catalog recipe ckpt/ref when skill card has empty values.
        if recipe:
            if not card.get("student_ckpt"):
                card["student_ckpt"] = recipe.get("student_ckpt") or ""
            if not card.get("ref_root"):
                card["ref_root"] = recipe.get("ref_root") or ""
            if not card.get("hand_obs"):
                card["hand_obs"] = recipe.get("hand_obs") or ""
        uni.append(card)
    uni_ids = {s.get("id") for s in uni}
    custom = []
    for s in (infer_config().get("skills") or []):
        if isinstance(s, dict) and s.get("id") and s["id"] not in uni_ids and s["id"] not in builtin_ids:
            custom.append({**s, "custom": True})
    out = dict(catalog)
    # Builtin recipes first, then skill cards not already present, then legacy custom.
    merged = list(builtins)
    for s in uni:
        if s.get("id") in builtin_ids:
            # Overlay skill datasets onto matching builtin.
            idx = next(i for i, b in enumerate(merged) if b.get("id") == s.get("id"))
            merged[idx] = {**merged[idx], **{k: v for k, v in s.items() if v not in ("", None, [], {})}, "id": s["id"]}
            if s.get("datasets"):
                merged[idx]["datasets"] = s["datasets"]
        else:
            merged.append(s)
    out["skills"] = merged + custom
    out["source_mode"] = "catalog+skills"
    return out


def infer_worker(job_id: str) -> None:
    item = store.find("infer_jobs", job_id)
    if not item:
        return
    catalog = merged_infer_catalog()
    try:
        env_map = infer_backend.build_infer_env(item, catalog)
        phi0_root = item["phi0_root"].rstrip("/")
        interactive = bool(item.get("interactive", True))
        if interactive:
            env_map["STUDIO_INTERACTIVE"] = "1"
            script = item.get("script") or catalog.get("script") or infer_backend.INTERACTIVE_SCRIPT
        else:
            env_map["STUDIO_INTERACTIVE"] = "0"
            script = (
                item.get("script_auto")
                or catalog.get("script_auto")
                or infer_backend.AUTO_SCRIPT
            )
        script_path = f"{phi0_root}/{script.lstrip('/')}"
        work_dir = item.get("work_dir") or infer_backend.infer_work_dir(item["out_dir"])
        item["work_dir"] = work_dir
        item["preview_jpeg"] = infer_backend.infer_preview_jpeg(item["out_dir"])
        env_map["STUDIO_PREVIEW_JPEG"] = item["preview_jpeg"]
        # Remote preview is pulled over SSH; slightly faster remote writes help smoothness.
        env_map["STUDIO_PREVIEW_INTERVAL_S"] = "0.12" if item.get("execution") == "remote" else "0.25"
        viewer = bool(item.get("viewer_interactive", True))
        if viewer:
            env_map["MUJOCO_GL"] = "glfw"
            item["mujoco_gl"] = "glfw"
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_map.items())
        # Local ZMQ monitor only makes sense when sim/deploy runs on this machine.
        if item.get("execution") == "local":
            try:
                mon = _start_infer_monitor(zmq_host=str((item.get("params") or {}).get("monitor_host") or "") or None)
                item["monitor_auto"] = bool(mon.get("ok"))
                if mon.get("ok"):
                    log_line(item, f"[studio] monitor auto-started host={mon.get('zmq_host')}")
                else:
                    log_line(item, f"[studio] monitor auto-start failed: {mon.get('detail')}")
            except Exception as mon_exc:  # noqa: BLE001
                item["monitor_auto"] = False
                log_line(item, f"[studio] monitor auto-start error: {mon_exc}")
        else:
            item["monitor_auto"] = False
            if viewer:
                log_line(item, "[studio] remote infer + ssh -Y：本机将弹出可拖动 MuJoCo 窗口（MUJOCO_GL=glfw）")
            else:
                log_line(item, "[studio] remote infer — 离屏 egl；本机仅视频预览")
            _start_infer_preview_puller(item["id"])
        bash = (
            f"mkdir -p -- {shlex.quote(str(Path(item['out_dir']).parent))} "
            f"{shlex.quote(str(Path(item['log_file']).parent))} "
            f"{shlex.quote(work_dir)} && "
            f": > {shlex.quote(work_dir + '/studio_cmd')} && "
            f"echo starting > {shlex.quote(work_dir + '/studio_phase')} && "
            f"cd {shlex.quote(phi0_root)} && {exports} "
            f"bash {shlex.quote(script_path)}; "
            f"rc=$?; "
            f"echo \"[studio] wrapper_exit=$rc\" | tee -a {shlex.quote(item['log_file'])}; "
            f"if [[ $rc -ne 0 ]]; then exit $rc; fi; "
            f"log={shlex.quote(item['engine_log'])}; "
            f"for i in $(seq 1 7200); do "
            f"  if [[ ! -f \"$log\" ]]; then sleep 2; continue; fi; "
            f"  if grep -qE 'done\\.|record stop|finished|Traceback|STUDIO phase=stopped' \"$log\" 2>/dev/null; then break; fi; "
            f"  sleep 2; "
            f"done; "
            f"exit $rc"
        )
        item["command"] = bash
        if viewer and interactive:
            item["message"] = "启动中：等待本机弹出 MuJoCo 窗口；就绪后 Deploy Terminal 按 y…"
        elif interactive:
            item["message"] = "一键启动中：Sim 就绪后请在 Deploy Terminal 按 y…"
        else:
            item["message"] = "正在启动全自动闭环推理…"
        item["phase"] = "starting"
        store.save()
        if item.get("execution") == "local":
            local_env = os.environ.copy()
            if viewer:
                disp = resolve_local_display()
                if disp:
                    local_env["DISPLAY"] = disp
                local_env["MUJOCO_GL"] = "glfw"
            proc = subprocess.Popen(
                ["bash", "-lc", bash],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, env=local_env,
            )
        else:
            host = host_by_id(item["host_id"])
            ssh_env = os.environ.copy()
            if viewer:
                disp = resolve_local_display()
                if not disp:
                    raise RuntimeError(
                        "本机没有可用的 DISPLAY，无法 ssh -Y 弹出 MuJoCo。"
                        "请在桌面会话里启动 Studio，或先 export DISPLAY=:0"
                    )
                ssh_env["DISPLAY"] = disp
                log_line(item, f"[studio] X11 forward DISPLAY={disp} → {host.get('target')}")
                proc = subprocess.Popen(
                    ssh_args_x11(host["target"], ["bash", "-lc", bash]),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=True, env=ssh_env,
                )
            else:
                proc = subprocess.Popen(
                    ssh_args(host["target"], ["bash", "-lc", bash]),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        with worker_lock:
            workers[job_id] = {"proc": proc, "kind": "infer"}
        assert proc.stdout is not None
        phase_msgs = {
            "sim_ready": "MuJoCo 就绪 — 可拖动窗口调视角，再点 Deploy",
            "deploy_starting": "Deploy / publisher 启动中",
            "init_done": "Init Done — 点击 ] 站立",
            "standing": "已站立 — 点击 Policy",
            "policy_starting": "Policy 武装中",
            "policy_ready": "Policy 就绪 — 按 Enter / P 开始推理",
            "streaming": "推理流式运行中",
            "done": "推理完成",
            "stopped": "已停止",
        }
        for line in proc.stdout:
            log_line(item, line)
            phase = infer_backend.parse_infer_phase("\n".join((item.get("logs") or [])[-120:]))
            item["phase"] = phase["phase"]
            if phase["phase"] in phase_msgs:
                item["message"] = phase_msgs[phase["phase"]]
            elif "sim" in phase.get("reached", []):
                item["message"] = "Sim 已启动"
            store.save()
        code = proc.wait()
        if item.get("status") != "cancelled":
            item["status"] = "completed" if code == 0 else "error"
            item["message"] = "推理完成" if code == 0 else f"推理失败 (rc={code})"
        item["completed_at"] = now_iso()
    except Exception as exc:  # noqa: BLE001
        item["status"] = "error"
        item["message"] = str(exc)
        log_line(item, f"错误: {exc}")
    finally:
        try:
            if item.get("monitor_auto"):
                _stop_infer_monitor()
                log_line(item, "[studio] monitor auto-stopped")
        except Exception:  # noqa: BLE001
            pass
        try:
            _stop_infer_preview_puller(job_id)
        except Exception:  # noqa: BLE001
            pass
        store.save()
        with worker_lock:
            workers.pop(job_id, None)


@app.get("/api/infer/catalog")
def get_infer_catalog():
    try:
        catalog = merged_infer_catalog()
        cfg = infer_config()
        return jsonify({
            "ok": True,
            "catalog": catalog,
            "config": cfg,
            "jobs": store.data.get("infer_jobs") or [],
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/config")
def save_infer_config():
    data = request.get_json(silent=True) or {}
    try:
        cfg = infer_config()
        if data.get("execution_mode") in {"local", "remote", "auto"}:
            cfg["execution_mode"] = data["execution_mode"]
        if data.get("host_id"):
            cfg["host_id"] = str(data["host_id"])
        if data.get("phi0_root"):
            cfg["phi0_root"] = require_abs(data["phi0_root"], "Phi0 根目录")
        if data.get("monitor_host") is not None:
            cfg["monitor_host"] = str(data.get("monitor_host") or "127.0.0.1").strip()
        if data.get("monitor_out"):
            cfg["monitor_out"] = require_abs(data["monitor_out"], "monitor_out")
        with store.lock:
            store.data["infer_config"] = cfg
            store.save()
        return jsonify({"ok": True, "config": cfg})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/skills")
def upsert_infer_skill():
    data = request.get_json(silent=True) or {}
    try:
        title = str(data.get("title") or "").strip()
        if not title:
            raise ValueError("请填写技能名称")
        ckpt = require_abs(data.get("student_ckpt") or "", "Student ckpt")
        ref = require_abs(data.get("ref_root") or "", "数据集 REF_ROOT")
        sid = str(data.get("id") or f"custom_{uuid.uuid4().hex[:8]}")
        skill = {
            "id": sid,
            "title": title,
            "badge": str(data.get("badge") or "SKILL").upper()[:12],
            "prompt": str(data.get("prompt") or "").strip(),
            "student_ckpt": ckpt,
            "ref_root": ref,
            "local_ref_root": str(data.get("local_ref_root") or "").strip() or None,
            "ep": int(data.get("ep") or 0),
            "script": "tools/eval/run_830_walk_student_cl_mujoco_viz.sh",
            "custom": True,
        }
        with store.lock:
            cfg = store.data.setdefault("infer_config", default_state()["infer_config"])
            skills = list(cfg.get("skills") or [])
            idx = next((i for i, s in enumerate(skills) if s.get("id") == sid), None)
            if idx is None:
                skills.append(skill)
            else:
                skills[idx] = skill
            cfg["skills"] = skills
            store.data["infer_config"] = cfg
            store.save()
        return jsonify({"ok": True, "skill": skill, "catalog": merged_infer_catalog()})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.delete("/api/infer/skills/<skill_id>")
def delete_infer_skill(skill_id: str):
    try:
        builtin = {s.get("id") for s in (infer_backend.load_infer_catalog().get("skills") or [])}
        if skill_id in builtin:
            raise ValueError("不能删除内置技能")
        with store.lock:
            cfg = store.data.setdefault("infer_config", default_state()["infer_config"])
            before = list(cfg.get("skills") or [])
            skills = [s for s in before if s.get("id") != skill_id]
            if len(skills) == len(before):
                raise ValueError("未找到自定义技能")
            cfg["skills"] = skills
            store.data["infer_config"] = cfg
            store.save()
        return jsonify({"ok": True, "deleted": skill_id})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/datasets/probe")
def probe_infer_dataset():
    data = request.get_json(silent=True) or {}
    try:
        ref = require_abs(data.get("ref_root") or data.get("dataset_path") or "", "REF_ROOT")
        ep = int(data.get("ep") or 0)
        # Prefer local mirror for video preview when available
        local = str(data.get("local_ref_root") or "").strip()
        probe_root = local if local and Path(local).is_dir() else ref
        if Path(probe_root).is_dir():
            probe = infer_backend.probe_local_videos(probe_root, ep)
            probe["remote_paths"] = infer_backend.episode_mp4_paths(ref, ep)
            return jsonify({"ok": True, **probe})

        # REF only on cluster — list videos / allowlist from training set remotely
        cfg = infer_config()
        host = host_by_id(str(data.get("host_id") or cfg.get("host_id") or "cluster_0"))
        proc = run_command(
            ssh_args(
                host["target"],
                ["python3", "-c", infer_backend.REMOTE_PROBE_INFER_VIDEOS, ref, str(ep)],
            ),
            timeout=90,
        )
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout or "远端 probe 失败").strip()[-400:])
        line = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else "{}"
        remote = json.loads(line)
        if not remote.get("ok", True):
            raise RuntimeError(remote.get("error") or "远端 REF 不可用")
        remote["remote_paths"] = dict(remote.get("paths") or infer_backend.episode_mp4_paths(ref, ep))
        remote["paths"] = dict(remote.get("paths") or {})
        # Keep absolute cluster paths so /api/infer/video can pull them
        return jsonify({"ok": True, **remote})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


def _stream_remote_mp4(host: dict[str, Any], path: str):
    """Byte-range capable stream of a remote mp4 (same pattern as shared media)."""
    size_proc = run_command(
        ssh_args(host["target"], ["python3", "-c", "import os,sys;print(os.path.getsize(sys.argv[1]))", path]),
        timeout=20,
    )
    if size_proc.returncode:
        raise RuntimeError((size_proc.stderr or size_proc.stdout or "无法读取远端视频大小").strip()[-200:])
    size = int((size_proc.stdout or "0").strip() or 0)
    if size <= 0:
        raise RuntimeError("远端视频为空")
    base_headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=30"}
    if request.method == "HEAD":
        return Response(
            status=200,
            headers=base_headers | {"Content-Length": str(size)},
            mimetype="video/mp4",
        )
    range_header = request.headers.get("Range")
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or (not match.group(1) and not match.group(2)):
            return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
        if match.group(1):
            start = int(match.group(1))
            requested_end = int(match.group(2)) if match.group(2) else size - 1
        else:
            suffix = int(match.group(2))
            start = max(0, size - suffix)
            requested_end = size - 1
        if start >= size or requested_end < start:
            return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
        end = min(size - 1, requested_end, start + MEDIA_CHUNK_SIZE - 1)
        length = end - start + 1
        proc = subprocess.run(
            ssh_args(host["target"], ["python3", "-c", REMOTE_MEDIA_READ, path, str(start), str(length)]),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode:
            raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[-200:] or "远端读视频失败")
        headers = base_headers | {
            "Content-Range": f"bytes {start}-{start + len(proc.stdout) - 1}/{size}",
            "Content-Length": str(len(proc.stdout)),
        }
        return Response(proc.stdout, status=206, headers=headers, mimetype="video/mp4")

    def generate():
        proc = subprocess.Popen(
            ssh_args(host["target"], ["cat", "--", path]),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(MEDIA_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.kill()

    return Response(
        stream_with_context(generate()),
        mimetype="video/mp4",
        headers=base_headers | {"Content-Length": str(size)},
    )


@app.get("/api/infer/display-status")
def infer_display_status():
    """Whether Studio host can pop X11 MuJoCo / desktop terminals."""
    disp = resolve_local_display()
    terms = [n for n in ("gnome-terminal", "xfce4-terminal", "konsole", "xterm", "x-terminal-emulator") if shutil.which(n)]
    return jsonify({
        "ok": True,
        "display": disp or None,
        "ready": bool(disp),
        "terminals": terms,
        "hint": (
            "一键启动会经 ssh -Y 在本机桌面弹出可拖动 MuJoCo（不在浏览器内嵌）。"
            if disp else
            "请在跑 Studio 的图形桌面里启动（export DISPLAY=:0），MuJoCo 才能弹窗。"
        ),
    })


@app.post("/api/infer/open-mujoco-terminal")
def open_mujoco_terminal():
    """From the webpage: open a desktop terminal with ssh -Y to cluster (interactive MuJoCo ready)."""
    data = request.get_json(silent=True) or {}
    try:
        cfg = infer_config()
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        host = host_by_id(host_id)
        target = str(host.get("target") or host_id)
        job_id = str(data.get("job_id") or "").strip()
        item = store.find("infer_jobs", job_id) if job_id else None
        if item and item.get("command"):
            # Re-run / attach to the same remote bash the job uses (X11 + glfw).
            remote_bash = str(item["command"])
            # Prefix glfw so a manual relaunch also gets a real window.
            remote_bash = f"export MUJOCO_GL=glfw; {remote_bash}"
            ssh_line = shlex.join(ssh_args_x11(target, ["bash", "-lc", remote_bash]))
            title = f"Studio MuJoCo · {item.get('id')}"
        else:
            phi0 = str(
                data.get("phi0_root")
                or cfg.get("phi0_root")
                or "/mnt/data2/wpy/workspace/Phi_0_wpy"
            ).rstrip("/")
            remote_bash = (
                f"export MUJOCO_GL=glfw; "
                f"cd {shlex.quote(phi0)}; "
                f"echo '[studio] DISPLAY='\"$DISPLAY\"' MUJOCO_GL='\"$MUJOCO_GL\"; "
                f"echo '[studio] 已 ssh -Y 到 {target}。请在网页点「一键启动」，或在此手动跑 CL 脚本。'; "
                f"echo '[studio] 交互 MuJoCo 窗口会弹到运行 Studio 的本机桌面。'; "
                f"exec bash"
            )
            ssh_line = shlex.join(ssh_args_x11(target, ["bash", "-lc", remote_bash]))
            title = f"Studio ssh -Y · {target}"
        inner = (
            f"echo {shlex.quote(title)}; "
            f"echo '[studio] DISPLAY='\"${{DISPLAY:-}}\"; "
            f"{ssh_line}; "
            f"echo; read -r -p '会话结束，按回车关闭终端…' _"
        )
        info = open_desktop_terminal(inner)
        return jsonify({
            "ok": True,
            "message": f"已打开桌面终端（{info['launcher']}）· ssh -Y {target}",
            **info,
            "target": target,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/jobs")
def start_infer_job():
    data = request.get_json(silent=True) or {}
    try:
        catalog = merged_infer_catalog()
        cfg = infer_config()
        skill_id = str(data.get("skill_id") or "")
        skill = next((s for s in (catalog.get("skills") or []) if s.get("id") == skill_id), None)
        params = dict(catalog.get("defaults") or {})
        if skill:
            params["prompt"] = skill.get("prompt") or params.get("prompt")
            if skill.get("hand_obs"):
                params["hand_obs"] = skill.get("hand_obs")
        params.update(data.get("params") or {})
        student_ckpt = require_abs(
            data.get("student_ckpt") or (skill or {}).get("student_ckpt") or "",
            "Student ckpt",
        )
        ref_root = str(
            data.get("ref_root") or (skill or {}).get("ref_root") or ""
        ).strip()
        # 04 产品约定：默认/强制走远端 cluster，不考虑本机 REF 播放。
        mode = "remote"
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        ref_root = require_abs(ref_root, "REF_ROOT")
        ep = int(data.get("ep") if data.get("ep") is not None else (skill or {}).get("ep") or 0)
        # 04 闭环：优先 02 valid；已 pack 的 *_unified 可用 vision allowlist（本机或远端）
        labels_path = str(
            data.get("labels_path")
            or data.get("labels_json")
            or (skill or {}).get("valid_json")
            or ""
        ).strip()
        screened = train_backend.read_screened_valid(ref_root)
        if not (screened.get("screened") and screened.get("valid")):
            if labels_path and Path(labels_path).is_file():
                do_export_valid_allowlist(
                    dataset_path=ref_root,
                    host_id=host_id,
                    source="labels",
                    labels_path=labels_path,
                )
                screened = train_backend.read_screened_valid(ref_root)
        if not screened.get("valid") and (mode == "remote" or not Path(ref_root).is_dir()):
            # REF only on cluster — read training allowlist remotely
            host0 = host_by_id(host_id)
            proc = run_command(
                ssh_args(
                    host0["target"],
                    ["python3", "-c", infer_backend.REMOTE_PROBE_INFER_VIDEOS, ref_root, str(ep)],
                ),
                timeout=90,
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                remote = json.loads((proc.stdout or "").strip().splitlines()[-1])
                screened = {
                    "valid": list(remote.get("valid") or remote.get("episodes") or []),
                    "invalid": list(remote.get("invalid") or []),
                    "screened": bool(remote.get("screened")),
                    "source": remote.get("allowlist_source") or "remote",
                    "valid_count": int(remote.get("valid_count") or len(remote.get("episodes") or [])),
                }
        if not screened.get("valid"):
            screened = train_backend.require_infer_episode_pool(ref_root)
        valid_set = set(int(x) for x in screened["valid"])
        if ep not in valid_set:
            src = screened.get("source") or "allowlist"
            tip = (
                "打包 allowlist"
                if src == "vision_allowlist" and not screened.get("screened")
                else "可用 episode 池"
            )
            raise ValueError(
                f"Episode {ep} 不在可用列表内（{tip}，共 {len(valid_set)} 条）。"
                f"请更换回放视频"
            )
        local_phi0 = (
            infer_backend.resolve_local_phi0_root(catalog)
            or train_backend.resolve_local_phi0_root()
        )
        remote_phi0 = str(
            catalog.get("phi0_root_remote")
            or cfg.get("phi0_root")
            or "/mnt/data2/wpy/workspace/Phi_0_wpy"
        ).rstrip("/")
        if mode == "remote":
            phi0_root = str(data.get("phi0_root") or remote_phi0)
            # Force cluster python; catalog defaults often point at a laptop path.
            params["phi0_py"] = str(
                params.get("phi0_py_remote")
                or cfg.get("remote_phi0_py")
                or "/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python"
            )
            if str(params.get("phi0_py") or "").startswith("/home/"):
                params["phi0_py"] = str(cfg.get("remote_phi0_py") or params["phi0_py"])
        else:
            phi0_root = str(
                data.get("phi0_root")
                or local_phi0
                or cfg.get("phi0_root")
                or remote_phi0
                or ""
            )
            if not phi0_root:
                raise ValueError("本机未找到 Phi_0_wpy（缺少 tools/eval CL 脚本）")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = str(data.get("tag") or f"studio_{(skill_id or 'cl')}_ep{ep}_{stamp}")
        out_dir = str(data.get("out_dir") or f"{phi0_root.rstrip('/')}/experiments/{tag}")
        log_file = f"{out_dir.rstrip('/')}/logs/studio_infer_{stamp}.log"
        engine_log = f"{out_dir.rstrip('/')}/logs/mujoco_cl.log"
        script = (skill or {}).get("script") or catalog.get("script")
        videos = infer_backend.episode_mp4_paths(ref_root, ep)
        action = str(data.get("action") or "init")
        # init / switch / step buttons → interactive; start → full auto
        interactive = action != "start"
        if data.get("interactive") is not None:
            interactive = bool(data.get("interactive"))
        if interactive:
            script = catalog.get("script") or infer_backend.INTERACTIVE_SCRIPT
            if skill and skill.get("script"):
                script = skill["script"]
        else:
            script = catalog.get("script_auto") or infer_backend.AUTO_SCRIPT
        work_dir = infer_backend.infer_work_dir(out_dir)
        item = {
            "id": "infer_" + uuid.uuid4().hex[:8],
            "kind": "infer",
            "skill_id": skill_id or None,
            "skill_title": (skill or {}).get("title") or skill_id or "custom",
            "execution": mode,
            "host_id": host_id,
            "phi0_root": phi0_root,
            "script": script,
            "interactive": interactive,
            "viewer_interactive": bool(data.get("viewer_interactive", True)),
            "mujoco_gl": "glfw" if bool(data.get("viewer_interactive", True)) else "egl",
            "student_ckpt": student_ckpt,
            "ref_root": ref_root,
            "local_ref_root": None,
            "ep": ep,
            "out_dir": out_dir,
            "work_dir": work_dir,
            "out_mp4": infer_backend.infer_out_mp4(out_dir),
            "preview_jpeg": infer_backend.infer_preview_jpeg(out_dir),
            "log_file": log_file,
            "engine_log": engine_log,
            "videos": videos,
            "params": params,
            "valid_hand_root": data.get("valid_hand_root") or None,
            "tag": tag,
            "stamp": stamp,
            "phase": "queued",
            "status": "running",
            "message": "排队启动远端推理" if mode == "remote" else "排队启动本机推理",
            "created_at": now_iso(),
            "logs": [],
            "action": action,
        }
        with store.lock:
            running = [j for j in (store.data.get("infer_jobs") or []) if j.get("status") == "running"]
            if running and not data.get("force"):
                raise ValueError("已有推理任务在运行，请先终止或使用切换技能")
            store.data.setdefault("infer_jobs", [])
            store.data["infer_jobs"].insert(0, item)
            store.data["infer_jobs"] = store.data["infer_jobs"][:30]
            store.save()
        threading.Thread(target=infer_worker, args=(item["id"],), daemon=True).start()
        terminal = None
        # Webpage one-click: also pop a desktop terminal so operators see ssh session / logs.
        if (
            item.get("viewer_interactive")
            and item.get("execution") == "remote"
            and data.get("open_terminal", True)
        ):
            try:
                host = host_by_id(item["host_id"])
                target = str(host.get("target") or item["host_id"])
                eng = item.get("engine_log") or ""
                follow = (
                    f"echo '[studio] MuJoCo 窗口会经 ssh -Y 弹到本机桌面（可拖动调视角）'; "
                    f"echo '[studio] 下面跟随远端日志：{eng}'; "
                    f"echo '[studio] Deploy 请在网页 Deploy Terminal 按 y → ] → Enter'; "
                    f"{shlex.join(ssh_args(target, ['bash', '-lc', f'tail -n 40 -F {shlex.quote(eng)}']))}; "
                    f"echo; read -r -p '按回车关闭…' _"
                )
                terminal = open_desktop_terminal(follow)
            except Exception as term_exc:  # noqa: BLE001
                terminal = {"ok": False, "message": str(term_exc)}
        return jsonify({"ok": True, "job": item, "terminal": terminal})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/jobs")
def list_infer_jobs():
    with store.lock:
        return jsonify({"ok": True, "jobs": list(store.data.get("infer_jobs") or [])})


@app.get("/api/infer/jobs/<item_id>")
def get_infer_job(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    return jsonify({"ok": True, "job": item})


@app.post("/api/infer/kill-deploy")
def kill_g1_deploy():
    """Force-stop g1_deploy on the active infer host (remote cluster or local)."""
    item = None
    with store.lock:
        for j in store.data.get("infer_jobs") or []:
            if j.get("status") == "running":
                item = j
                break
    if item and item.get("execution") == "remote":
        host = host_by_id(item.get("host_id") or "cluster_0")
        cp = run_command(
            ssh_args(host["target"], ["bash", "-lc", 'pkill -9 -f "g1_deploy" || true']),
            timeout=20,
        )
        msg = f'已在 {host.get("target") or host.get("id")} 执行 pkill -9 -f "g1_deploy"'
    else:
        cp = run_command(
            ["bash", "-lc", 'pkill -9 -f "g1_deploy" || true'],
            timeout=15,
        )
        msg = '已执行 pkill -9 -f "g1_deploy"'
    out = (cp.stdout or "").strip()
    err = (cp.stderr or "").strip()
    detail = out or err
    if detail:
        msg = f"{msg}: {detail[:200]}"
    return jsonify({"ok": True, "message": msg, "code": cp.returncode, "stdout": out, "stderr": err})


@app.post("/api/infer/jobs/<item_id>/cancel")
def cancel_infer_job(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    # Ask interactive orchestrator to stop cleanly first.
    work_dir = item.get("work_dir") or infer_backend.infer_work_dir(item.get("out_dir") or "")
    if work_dir:
        try:
            cmd_path = Path(work_dir) / "studio_cmd"
            if item.get("execution") == "local":
                cmd_path.parent.mkdir(parents=True, exist_ok=True)
                cmd_path.write_text("stop\n", encoding="utf-8")
            elif item.get("host_id"):
                host = host_by_id(item["host_id"])
                run_command(
                    ssh_args(host["target"], ["bash", "-lc", f"printf 'stop\\n' > {shlex.quote(str(cmd_path))}"]),
                    timeout=15,
                )
        except Exception:  # noqa: BLE001
            pass
    with worker_lock:
        w = workers.get(item_id)
    if w and w.get("proc") and w["proc"].poll() is None:
        try:
            os.killpg(w["proc"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    item["status"], item["message"] = "cancelled", "已终止推理"
    item["phase"] = "cancelled"
    try:
        _stop_infer_monitor()
        item["monitor_auto"] = False
    except Exception:  # noqa: BLE001
        pass
    try:
        _stop_infer_preview_puller(item_id)
    except Exception:  # noqa: BLE001
        pass
    store.save()
    return jsonify({"ok": True})


@app.post("/api/infer/jobs/<item_id>/cmd")
def post_infer_cmd(item_id: str):
    """Send staged command to interactive orchestrator (deploy/stand/policy/stream/...)."""
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    if item.get("status") != "running":
        return api_error("任务未在运行")
    data = request.get_json(silent=True) or {}
    cmd = str(data.get("cmd") or data.get("action") or "").strip()
    if cmd in {"]", "stand_up"}:
        cmd = "stand"
    elif cmd in {"y", "Y"}:
        cmd = "deploy"
    elif cmd in {"enter"}:
        cmd = "stream"
    allowed = {
        "deploy", "stand", "policy", "stream", "stream_p", "enter", "p",
        "stop", "go", "continue", "y", "Y", "]",
    }
    if cmd.startswith("key:") and len(cmd) > 4:
        pass
    elif cmd not in allowed:
        return api_error(f"未知命令: {cmd}")
    work_dir = item.get("work_dir") or infer_backend.infer_work_dir(item.get("out_dir") or "")
    if not work_dir:
        return api_error("缺少 work_dir")
    cmd_path = f"{work_dir.rstrip('/')}/studio_cmd"
    if item.get("execution") == "local":
        try:
            p = Path(cmd_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cmd + "\n", encoding="utf-8")
        except OSError as exc:
            return api_error(f"写入命令失败: {exc}")
    else:
        host = host_by_id(item.get("host_id") or "cluster_0")
        remote = (
            f"mkdir -p -- {shlex.quote(work_dir)} && "
            f"printf '%s\\n' {shlex.quote(cmd)} > {shlex.quote(cmd_path)}"
        )
        proc = run_command(ssh_args(host["target"], ["bash", "-lc", remote]), timeout=20)
        if proc.returncode != 0:
            return api_error(proc.stderr.strip() or proc.stdout.strip() or "写入命令失败")
    log_line(item, f"[studio] cmd={cmd}")
    item["last_cmd"] = cmd
    item["message"] = f"已发送命令: {cmd}"
    store.save()
    return jsonify({"ok": True, "cmd": cmd, "job": item})


@app.get("/api/infer/jobs/<item_id>/logs")
def get_infer_logs(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    try:
        exec_mode = item.get("execution") or "local"
        host_id = item.get("host_id") or "cluster_0"
        engine = _read_text_at(item.get("engine_log") or "", exec_mode, host_id, max_bytes=200_000)
        studio = _read_text_at(item.get("log_file") or "", exec_mode, host_id, max_bytes=80_000)
        work_dir = item.get("work_dir") or infer_backend.infer_work_dir(item.get("out_dir") or "")
        status_json = ""
        deploy_log = ""
        sim_log = ""
        replay_log = ""
        if work_dir:
            logs_root = f"{work_dir.rstrip('/')}/logs"
            status_json = _read_text_at(
                f"{work_dir.rstrip('/')}/studio_status.json",
                exec_mode,
                host_id,
                max_bytes=4096,
            )
            deploy_log = _read_text_at(f"{logs_root}/deploy.log", exec_mode, host_id, max_bytes=160_000)
            sim_log = _read_text_at(f"{logs_root}/sim.log", exec_mode, host_id, max_bytes=120_000)
            replay_log = _read_text_at(f"{logs_root}/replay.log", exec_mode, host_id, max_bytes=120_000)
        mem = "\n".join((item.get("logs") or [])[-200:])

        def _tail(s: str, n: int = 60_000) -> str:
            t = (s or "").strip()
            return t[-n:] if len(t) > n else t

        # Deploy Terminal prefers deploy.log (01-style); fall back to engine.
        deploy_view = _tail(deploy_log) or _tail(engine) or _tail(studio) or _tail(mem) or "（日志为空）"
        phase_src = "\n".join(
            x for x in (status_json, engine, deploy_log, sim_log, replay_log, studio, mem) if x
        )
        phase = infer_backend.studio_phase_from_status(status_json, phase_src)
        stack = infer_backend.infer_stack_status(
            phase=str(phase.get("phase") or ""),
            sim_text=sim_log,
            replay_text=replay_log,
            deploy_text=deploy_log,
        )
        item["phase"] = phase["phase"]
        if phase.get("message"):
            item["message"] = phase["message"]
        item["stack"] = stack
        store.save()
        return jsonify({
            "ok": True,
            "text": deploy_view,
            "engine_text": engine,
            "sim_text": sim_log,
            "replay_text": replay_log,
            "policy_text": replay_log,
            "deploy_text": deploy_log,
            "stack": stack,
            "phase": phase,
            "job": item,
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/jobs/<item_id>/mp4-status")
def get_infer_mp4_status(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    path = item.get("out_mp4") or ""
    exists, size = 0, 0
    if item.get("execution") == "local" or Path(path).is_file():
        p = Path(path)
        if p.is_file():
            exists, size = 1, p.stat().st_size
    else:
        host = host_by_id(item.get("host_id") or "cluster_0")
        py = (
            "import pathlib,os;p=pathlib.Path(%r);"
            "print(int(p.is_file()), p.stat().st_size if p.is_file() else 0)"
        ) % path
        proc = run_command(ssh_args(host["target"], ["python3", "-c", py]), timeout=20)
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split()
            if len(parts) >= 2:
                exists, size = int(parts[0]), int(parts[1])
    return jsonify({
        "ok": True,
        "path": path,
        "exists": bool(exists),
        "size": size,
        "url": f"/api/infer/jobs/{item_id}/mp4" if exists else None,
    })


@app.get("/api/infer/jobs/<item_id>/mp4")
def stream_infer_mp4(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    path = item.get("out_mp4") or ""
    local = Path(path)
    if item.get("execution") == "local" or local.is_file():
        if not local.is_file():
            return api_error("mp4 不存在", 404)
        return send_file(local, mimetype="video/mp4", conditional=True)
    host = host_by_id(item.get("host_id") or "cluster_0")

    def generate():
        proc = subprocess.Popen(
            ssh_args(host["target"], ["cat", "--", path]),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(MEDIA_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.kill()

    return Response(stream_with_context(generate()), mimetype="video/mp4")


def _infer_monitor_meta() -> dict[str, Any]:
    cfg = infer_config()
    out = str(cfg.get("monitor_out") or infer_backend.MONITOR_DEFAULT_OUT)
    paths = infer_backend.monitor_paths(out)
    catalog = merged_infer_catalog()
    local_phi0 = (
        infer_backend.resolve_local_phi0_root(catalog)
        or train_backend.resolve_local_phi0_root()
        or cfg.get("phi0_root")
        or "/mnt/data2/wpy/workspace/Phi_0_wpy"
    )
    return {
        "host_id": cfg.get("host_id") or "cluster_0",
        "zmq_host": str(cfg.get("monitor_host") or "127.0.0.1"),
        "phi0_root": str(local_phi0),
        "paths": paths,
        "execution": "local",
    }


def _start_infer_monitor(*, zmq_host: str | None = None) -> dict[str, Any]:
    cfg = infer_config()
    if zmq_host:
        cfg["monitor_host"] = str(zmq_host).strip() or "127.0.0.1"
        with store.lock:
            store.data["infer_config"] = cfg
            store.save()
    meta = _infer_monitor_meta()
    paths = meta["paths"]
    script = f"{meta['phi0_root'].rstrip('/')}/{infer_backend.MONITOR_SCRIPT}"
    bash = (
        f"mkdir -p -- {shlex.quote(paths['out'])}; "
        f"if [[ -f {shlex.quote(paths['pid'])} ]]; then "
        f"  kill \"$(cat {shlex.quote(paths['pid'])})\" 2>/dev/null || true; "
        f"fi; "
        f"nohup python3 {shlex.quote(script)} "
        f"--host {shlex.quote(meta['zmq_host'])} "
        f"--out {shlex.quote(paths['out'])} "
        f"> {shlex.quote(paths['out'] + '/monitor.log')} 2>&1 & "
        f"sleep 0.3; "
        f"test -f {shlex.quote(paths['pid'])} && echo OK || "
        f"(tail -20 {shlex.quote(paths['out'] + '/monitor.log')} ; exit 1)"
    )
    proc = run_command(["bash", "-lc", bash], timeout=30)
    return {
        "ok": proc.returncode == 0,
        "zmq_host": meta["zmq_host"],
        "paths": paths,
        "detail": (proc.stderr or proc.stdout or "").strip()[-400:],
    }


def _stop_infer_monitor() -> dict[str, Any]:
    meta = _infer_monitor_meta()
    paths = meta["paths"]
    bash = (
        f"if [[ -f {shlex.quote(paths['pid'])} ]]; then "
        f"  kill \"$(cat {shlex.quote(paths['pid'])})\" 2>/dev/null || true; "
        f"  rm -f {shlex.quote(paths['pid'])}; "
        f"fi; echo stopped"
    )
    proc = run_command(["bash", "-lc", bash], timeout=20)
    return {"ok": proc.returncode == 0}


# Back-compat aliases
_start_infer_monitor_remote = _start_infer_monitor
_stop_infer_monitor_remote = _stop_infer_monitor


@app.get("/api/infer/monitor/status")
def get_infer_monitor_status():
    try:
        meta = _infer_monitor_meta()
        paths = meta["paths"]
        pid_p = Path(paths["pid"])
        st_p = Path(paths["status"])
        alive, pid = False, None
        if pid_p.is_file():
            try:
                pid = int(pid_p.read_text().strip())
                os.kill(pid, 0)
                alive = True
            except Exception:  # noqa: BLE001
                alive = False
        status = json.loads(st_p.read_text()) if st_p.is_file() else {}
        return jsonify({
            "ok": True,
            "alive": alive,
            "pid": pid,
            "status": status,
            "paths": paths,
            "zmq_host": meta["zmq_host"],
        })
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/monitor/start")
def start_infer_monitor():
    data = request.get_json(silent=True) or {}
    try:
        cfg = infer_config()
        if data.get("monitor_out"):
            cfg["monitor_out"] = require_abs(data["monitor_out"], "monitor_out")
            with store.lock:
                store.data["infer_config"] = cfg
                store.save()
        result = _start_infer_monitor(zmq_host=data.get("monitor_host"))
        if not result.get("ok"):
            return api_error(result.get("detail") or "启动 monitor 失败")
        return jsonify({"ok": True, "message": "monitor 已启动", **result})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.post("/api/infer/monitor/stop")
def stop_infer_monitor():
    try:
        result = _stop_infer_monitor()
        if not result.get("ok"):
            return api_error("停止失败")
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/monitor/camera.jpg")
def get_infer_monitor_camera():
    try:
        meta = _infer_monitor_meta()
        path = Path(meta["paths"]["camera"])
        if not path.is_file():
            return api_error("相机帧不存在", 404)
        return send_file(path, mimetype="image/jpeg", conditional=True)
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/jobs/<item_id>/preview-status")
def get_infer_preview_status(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    path = item.get("preview_jpeg") or infer_backend.infer_preview_jpeg(item.get("out_dir") or "")
    cached = _preview_frame_cache.get(item_id) or {}
    exists = bool(cached.get("jpeg"))
    size = len(cached.get("jpeg") or b"")
    mtime = int(cached.get("mtime") or 0)
    if not exists:
        if item.get("execution") == "local" or Path(path).is_file():
            p = Path(path)
            exists = p.is_file()
            size = p.stat().st_size if exists else 0
            mtime = int(p.stat().st_mtime) if exists else 0
        elif item.get("execution") == "remote":
            # Kick puller if missing; status may catch up next poll.
            _start_infer_preview_puller(item_id)
    return jsonify({
        "ok": True,
        "path": path,
        "exists": exists,
        "size": size,
        "mtime": mtime,
        "execution": item.get("execution") or "local",
        "url": f"/api/infer/jobs/{item_id}/preview.jpg" if exists else None,
        # Do not advertise MJPEG to the UI — multipart streams crash Chromium/Edge tabs.
        "mjpeg": None,
    })


@app.get("/api/infer/jobs/<item_id>/preview.jpg")
def stream_infer_preview(item_id: str):
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    path = item.get("preview_jpeg") or infer_backend.infer_preview_jpeg(item.get("out_dir") or "")
    cached = _preview_frame_cache.get(item_id) or {}
    data = cached.get("jpeg") or b""
    if data:
        etag = f'W/"{cached.get("mtime")}-{len(data)}"'
        if request.headers.get("If-None-Match") == etag:
            return Response(status=304)
        return Response(
            data,
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "ETag": etag,
                "X-Preview-Source": "cache",
            },
        )
    local = Path(path)
    if item.get("execution") != "remote" and local.is_file():
        return send_file(local, mimetype="image/jpeg", conditional=True)
    if item.get("execution") == "remote":
        host = host_by_id(item.get("host_id") or "cluster_0")
        jpeg, mtime = _read_remote_preview_jpeg(host, path)
        if jpeg:
            _preview_frame_cache[item_id] = {"jpeg": jpeg, "mtime": mtime, "updated": time.time()}
            return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
    return api_error("预览不存在", 404)


@app.get("/api/infer/jobs/<item_id>/preview.mjpeg")
def stream_infer_preview_mjpeg(item_id: str):
    """Low-latency multipart JPEG stream for the Sim panel (local playback only)."""
    item = store.find("infer_jobs", item_id)
    if not item:
        return api_error("推理任务不存在", 404)
    if item.get("execution") == "remote":
        _start_infer_preview_puller(item_id)
    boundary = b"frame"

    def generate():
        last = b""
        idle = 0
        while idle < 80:  # ~10s after stall
            cur = store.find("infer_jobs", item_id)
            cached = _preview_frame_cache.get(item_id) or {}
            data = cached.get("jpeg") or b""
            if not data and cur and (cur.get("execution") or "local") == "local":
                path = cur.get("preview_jpeg") or infer_backend.infer_preview_jpeg(cur.get("out_dir") or "")
                p = Path(path)
                if p.is_file():
                    try:
                        data = p.read_bytes()
                        _preview_frame_cache[item_id] = {
                            "jpeg": data,
                            "mtime": int(p.stat().st_mtime_ns),
                            "updated": time.time(),
                        }
                    except OSError:
                        data = b""
            if data and data != last:
                last = data
                idle = 0
                header = (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                )
                yield header + data + b"\r\n"
            else:
                idle += 1
            if not cur or cur.get("status") not in {"running", "queued"}:
                if idle > 12:
                    break
            time.sleep(0.08)

    return Response(
        stream_with_context(generate()),
        mimetype=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
        headers={"Cache-Control": "no-store, no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Infer media cache (remote mp4 → local disk for smooth playback)
# ---------------------------------------------------------------------------
INFER_VIDEO_CACHE = APP_DIR / "cache" / "infer_videos"
_infer_video_cache_lock = threading.Lock()
_infer_video_cache_jobs: dict[str, dict[str, Any]] = {}
_preview_frame_cache: dict[str, dict[str, Any]] = {}
_preview_pullers: dict[str, dict[str, Any]] = {}


def _infer_video_cache_path(remote_path: str) -> Path:
    digest = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:20]
    return INFER_VIDEO_CACHE / digest / Path(remote_path).name


def _read_remote_preview_jpeg(host: dict[str, Any], path: str) -> tuple[bytes, int]:
    py = (
        "import pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "if not p.is_file():\n"
        "  sys.exit(2)\n"
        "st=p.stat()\n"
        "sys.stdout.buffer.write(f'{st.st_mtime_ns} {st.st_size}\\n'.encode())\n"
        "sys.stdout.buffer.write(p.read_bytes())\n"
    )
    proc = subprocess.run(
        ssh_args(host["target"], ["python3", "-c", py, path]),
        capture_output=True,
        timeout=20,
    )
    if proc.returncode != 0 or not proc.stdout:
        return b"", 0
    raw = proc.stdout
    nl = raw.find(b"\n")
    if nl < 0:
        return b"", 0
    try:
        meta = raw[:nl].decode("ascii", "replace").split()
        mtime = int(meta[0]) if meta else 0
    except ValueError:
        mtime = 0
    return raw[nl + 1 :], mtime


def _start_infer_preview_puller(job_id: str) -> None:
    with _infer_video_cache_lock:
        cur = _preview_pullers.get(job_id)
        if cur and cur.get("thread") and cur["thread"].is_alive():
            return
        stop = threading.Event()

        def loop() -> None:
            while not stop.is_set():
                item = store.find("infer_jobs", job_id)
                if not item or item.get("status") not in {"running", "queued"}:
                    break
                path = item.get("preview_jpeg") or infer_backend.infer_preview_jpeg(item.get("out_dir") or "")
                try:
                    if item.get("execution") == "local" or Path(path).is_file():
                        p = Path(path)
                        if p.is_file():
                            data = p.read_bytes()
                            _preview_frame_cache[job_id] = {
                                "jpeg": data,
                                "mtime": int(p.stat().st_mtime_ns),
                                "updated": time.time(),
                            }
                    else:
                        host = host_by_id(item.get("host_id") or "cluster_0")
                        jpeg, mtime = _read_remote_preview_jpeg(host, path)
                        if jpeg:
                            prev = _preview_frame_cache.get(job_id) or {}
                            if mtime != prev.get("mtime") or jpeg != prev.get("jpeg"):
                                _preview_frame_cache[job_id] = {
                                    "jpeg": jpeg,
                                    "mtime": mtime,
                                    "updated": time.time(),
                                }
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(0.1)
            _preview_pullers.pop(job_id, None)

        th = threading.Thread(target=loop, daemon=True, name=f"preview-{job_id}")
        _preview_pullers[job_id] = {"thread": th, "stop": stop}
        th.start()


def _stop_infer_preview_puller(job_id: str) -> None:
    with _infer_video_cache_lock:
        cur = _preview_pullers.pop(job_id, None)
    if cur and cur.get("stop"):
        cur["stop"].set()


def _cache_infer_video_worker(job_key: str, host_id: str, remote_path: str, local: Path) -> None:
    host = host_by_id(host_id)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".partial")
    try:
        with _infer_video_cache_lock:
            _infer_video_cache_jobs[job_key] = {
                "status": "running",
                "remote": remote_path,
                "local": str(local),
                "ready": False,
                "message": "拉取中…",
            }
        src = f"{host['target']}:{remote_path}"
        proc = subprocess.run(
            ["rsync", "-az", "--partial", "--info=progress2", *rsync_ssh(), src, str(tmp)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
            raise RuntimeError((proc.stderr or proc.stdout or "rsync 失败").strip()[-300:])
        tmp.replace(local)
        with _infer_video_cache_lock:
            _infer_video_cache_jobs[job_key] = {
                "status": "done",
                "remote": remote_path,
                "local": str(local),
                "ready": True,
                "size": local.stat().st_size,
                "message": "已缓存",
                "url": f"/api/infer/video?path={remote_path}",  # resolved via local prefer
            }
    except Exception as exc:  # noqa: BLE001
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass
        with _infer_video_cache_lock:
            _infer_video_cache_jobs[job_key] = {
                "status": "error",
                "remote": remote_path,
                "local": str(local),
                "ready": False,
                "message": str(exc)[:300],
            }


@app.post("/api/infer/videos/cache")
def cache_infer_videos():
    """Pull remote ego/wrist mp4 into local cache for stutter-free playback."""
    data = request.get_json(silent=True) or {}
    try:
        cfg = infer_config()
        host_id = str(data.get("host_id") or cfg.get("host_id") or "cluster_0")
        paths = [str(p) for p in (data.get("paths") or []) if str(p).startswith("/")]
        if not paths:
            raise ValueError("请提供 paths")
        out = []
        for remote in paths:
            local = _infer_video_cache_path(remote)
            key = hashlib.sha1(remote.encode("utf-8")).hexdigest()[:20]
            if local.is_file() and local.stat().st_size > 0:
                info = {
                    "remote": remote,
                    "local": str(local),
                    "ready": True,
                    "status": "done",
                    "size": local.stat().st_size,
                    "url": f"/api/infer/video?path={remote}",
                    "message": "已缓存",
                }
                with _infer_video_cache_lock:
                    _infer_video_cache_jobs[key] = info
                out.append(info)
                continue
            with _infer_video_cache_lock:
                cur = _infer_video_cache_jobs.get(key)
                if cur and cur.get("status") == "running":
                    out.append(cur)
                    continue
            threading.Thread(
                target=_cache_infer_video_worker,
                args=(key, host_id, remote, local),
                daemon=True,
            ).start()
            info = {
                "remote": remote,
                "local": str(local),
                "ready": False,
                "status": "running",
                "message": "开始拉取…",
            }
            with _infer_video_cache_lock:
                _infer_video_cache_jobs[key] = info
            out.append(info)
        return jsonify({"ok": True, "items": out})
    except Exception as exc:  # noqa: BLE001
        return api_error(str(exc))


@app.get("/api/infer/videos/cache")
def get_infer_video_cache_status():
    paths = request.args.getlist("path") or []
    with _infer_video_cache_lock:
        if not paths:
            return jsonify({"ok": True, "items": list(_infer_video_cache_jobs.values())})
        items = []
        for remote in paths:
            key = hashlib.sha1(remote.encode("utf-8")).hexdigest()[:20]
            local = _infer_video_cache_path(remote)
            cur = dict(_infer_video_cache_jobs.get(key) or {})
            if local.is_file() and local.stat().st_size > 0:
                cur = {
                    "remote": remote,
                    "local": str(local),
                    "ready": True,
                    "status": "done",
                    "size": local.stat().st_size,
                    "url": f"/api/infer/video?path={remote}",
                    "message": "已缓存",
                }
            elif not cur:
                cur = {"remote": remote, "ready": False, "status": "idle", "message": "未缓存"}
            items.append(cur)
    return jsonify({"ok": True, "items": items})


@app.get("/api/infer/video")
def stream_infer_dataset_video():
    """Stream ego/wrist dataset mp4 — prefer local cache, else cluster Range stream."""
    path = request.args.get("path") or ""
    if not path.startswith("/"):
        return api_error("path 必须是绝对路径")
    local = Path(path)
    if local.is_file():
        return send_file(local, mimetype="video/mp4", conditional=True, max_age=0)
    # Prefer Studio cache of a remote absolute path
    cached = _infer_video_cache_path(path)
    if cached.is_file() and cached.stat().st_size > 0:
        return send_file(cached, mimetype="video/mp4", conditional=True, max_age=3600)
    try:
        cfg = infer_config()
        host = host_by_id(str(request.args.get("host_id") or cfg.get("host_id") or "cluster_0"))
        # Kick cache in background so subsequent seeks are local/fast.
        key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:20]
        with _infer_video_cache_lock:
            cur = _infer_video_cache_jobs.get(key)
            running = bool(cur and cur.get("status") == "running")
        if not running:
            threading.Thread(
                target=_cache_infer_video_worker,
                args=(key, host.get("id") or cfg.get("host_id") or "cluster_0", path, cached),
                daemon=True,
            ).start()
        return _stream_remote_mp4(host, path)
    except Exception as exc:  # noqa: BLE001
        return api_error(f"视频不存在（本地/远端）: {exc}", 404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7890)
    args = parser.parse_args()
    try:
        resume_running_train_monitors()
    except Exception as exc:  # noqa: BLE001
        print(f"[studio] resume train monitors skipped: {exc}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
