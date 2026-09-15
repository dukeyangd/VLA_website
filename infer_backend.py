"""Sim closed-loop infer helpers for Humanoid Data Studio (MuJoCo CL)."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
CATALOG_PATH = APP_DIR / "phi0_pipeline" / "infer_catalog.json"

EGO_REL = "videos/chunk-000/observation.images.ego_view"
WRIST_REL = "videos/chunk-000/observation.images.left_wrist"


def load_infer_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text("utf-8"))


def resolve_local_phi0_root(catalog: dict[str, Any] | None = None) -> str | None:
    """Pick a local Phi_0 root that has the CL orchestrator (or any tools/eval)."""
    catalog = catalog or load_infer_catalog()
    candidates = list(catalog.get("phi0_root_local_candidates") or [])
    remote = catalog.get("phi0_root_remote")
    if remote:
        candidates.append(remote)
    for cand in candidates:
        root = Path(cand)
        orch = root / "tools" / "eval" / "studio_cl_orchestrator.sh"
        auto = root / "tools" / "eval" / "run_830_walk_student_cl_mujoco_viz.sh"
        if orch.is_file() or auto.is_file():
            return str(root.resolve())
    return None


def episode_mp4_paths(ref_root: str, ep: int) -> dict[str, str]:
    ep6 = f"{int(ep):06d}"
    root = ref_root.rstrip("/")
    return {
        "ego": f"{root}/{EGO_REL}/episode_{ep6}.mp4",
        "wrist": f"{root}/{WRIST_REL}/episode_{ep6}.mp4",
        # unified pack may also use file-NNN.parquet naming; videos stay episode_XXXXXX
        "parquet": f"{root}/data/chunk-000/file-{int(ep):03d}.parquet",
    }


def list_episode_indices(ref_root: str, *, valid_only: bool = True) -> list[int]:
    """Episode indices that have ego_view mp4 under a LeRobot-style REF_ROOT.

    When valid_only=True (default), intersect with 02-exported valid allowlist if present.
    """
    ego_dir = Path(ref_root) / EGO_REL
    if not ego_dir.is_dir():
        return []
    eps: list[int] = []
    for p in ego_dir.glob("episode_*.mp4"):
        m = re.match(r"episode_(\d+)\.mp4$", p.name)
        if m:
            eps.append(int(m.group(1)))
    eps = sorted(set(eps))
    if not valid_only:
        return eps
    try:
        import train_backend

        screened = train_backend.read_screened_valid(ref_root)
    except Exception:  # noqa: BLE001
        return eps
    if screened.get("screened") and screened.get("valid"):
        allow = set(int(x) for x in screened["valid"])
        return [e for e in eps if e in allow]
    if screened.get("valid"):
        # Unscreened pack allowlist: still intersect (usually = all)
        allow = set(int(x) for x in screened["valid"])
        return [e for e in eps if e in allow]
    return eps


def probe_local_videos(ref_root: str, ep: int) -> dict[str, Any]:
    paths = episode_mp4_paths(ref_root, ep)
    out = {"ref_root": ref_root, "ep": int(ep), "paths": paths, "exists": {}}
    for key in ("ego", "wrist", "parquet"):
        p = Path(paths[key])
        out["exists"][key] = p.is_file()
        if p.is_file():
            out[f"{key}_size"] = p.stat().st_size
    out["ready"] = bool(out["exists"].get("ego") and out["exists"].get("wrist"))
    all_eps = list_episode_indices(ref_root, valid_only=False)
    episodes = list_episode_indices(ref_root, valid_only=True)
    out["episodes_all"] = all_eps
    out["episodes"] = episodes
    out["episode_count"] = len(episodes)
    try:
        import train_backend

        screened = train_backend.read_screened_valid(ref_root)
        out["screened"] = bool(screened.get("screened"))
        out["valid"] = list(screened.get("valid") or [])
        out["invalid"] = list(screened.get("invalid") or [])
        out["valid_count"] = int(screened.get("valid_count") or 0)
        out["invalid_count"] = int(screened.get("invalid_count") or 0)
        out["allowlist_source"] = screened.get("source")
        if screened.get("invalid"):
            out["invalid_tip"] = f"[{','.join(str(x) for x in screened['invalid'])}]为invalid"
    except Exception:  # noqa: BLE001
        out["screened"] = False
    out["source"] = "local"
    return out


# Remote probe snippet used by Studio when REF is only on cluster_0.
REMOTE_PROBE_INFER_VIDEOS = r'''
import json, os, re, sys
from pathlib import Path
root = Path(sys.argv[1])
ep = int(sys.argv[2])
ego_rel = "videos/chunk-000/observation.images.ego_view"
wrist_rel = "videos/chunk-000/observation.images.left_wrist"
out = {
  "ok": True, "ref_root": str(root), "ep": ep, "exists": {}, "paths": {},
  "episodes": [], "episodes_all": [], "episode_count": 0,
  "screened": False, "valid": [], "invalid": [], "valid_count": 0, "invalid_count": 0,
  "allowlist_source": None, "source": "remote",
}
if not root.is_dir():
  out["ok"] = False
  out["error"] = "ref missing"
  print(json.dumps(out, ensure_ascii=False))
  raise SystemExit(0)
# allowlist FIRST — avoid NFS glob of hundreds of mp4s (was timing out Studio UI).
valid, invalid, screened, source = [], [], False, None
for rel, src in (
  ("meta/sonic_qa_valid_invalid.json", "sonic_qa"),
  ("labels.json", "labels"),
  ("meta/vision_episode_allowlist.json", "vision_allowlist"),
  ("meta/all_episode_allowlist.json", "vision_allowlist"),
):
  p = root / rel
  if not p.is_file():
    continue
  try:
    raw = json.loads(p.read_text(encoding="utf-8"))
  except Exception:
    continue
  if src == "labels" and isinstance(raw, dict) and root.name in raw and isinstance(raw[root.name], dict):
    raw = raw[root.name]
  if isinstance(raw, dict) and ("valid" in raw or "invalid" in raw):
    valid = sorted({int(x) for x in (raw.get("valid") or [])})
    invalid = sorted({int(x) for x in (raw.get("invalid") or [])})
    screened = bool(valid) and src in ("sonic_qa", "labels")
    source = src
    break
  if isinstance(raw, dict):
    arr = raw.get("episode_index") or raw.get("episodes") or raw.get("valid") or []
    if isinstance(arr, list) and arr:
      valid = sorted({int(x) for x in arr})
      screened = False
      source = src
      break
out["valid"] = valid
out["invalid"] = invalid
out["screened"] = screened
out["allowlist_source"] = source
out["valid_count"] = len(valid)
out["invalid_count"] = len(invalid)
eps = list(valid)
if not eps:
  # Fallback: cheap listdir (not Path.glob — slower on some NFS mounts).
  ego_dir = root / ego_rel
  if ego_dir.is_dir():
    for name in os.listdir(ego_dir):
      m = re.match(r"episode_(\d+)\.mp4$", name)
      if m:
        eps.append(int(m.group(1)))
    eps = sorted(set(eps))
out["episodes_all"] = eps
out["episodes"] = eps
out["episode_count"] = len(out["episodes"])
ep6 = f"{ep:06d}"
paths = {
  "ego": str(root / ego_rel / f"episode_{ep6}.mp4"),
  "wrist": str(root / wrist_rel / f"episode_{ep6}.mp4"),
  "parquet": str(root / "data" / "chunk-000" / f"file-{ep:03d}.parquet"),
}
out["paths"] = paths
for k, path in paths.items():
  out["exists"][k] = os.path.isfile(path)
  if out["exists"][k]:
    try:
      out[f"{k}_size"] = os.path.getsize(path)
    except OSError:
      pass
out["ready"] = bool(out["exists"].get("ego") and out["exists"].get("wrist"))
print(json.dumps(out, ensure_ascii=False))
'''


def attach_screened_fields(out: dict[str, Any], ref_root: str) -> dict[str, Any]:
    """Fill screened/allowlist fields onto a probe payload (local REF)."""
    try:
        import train_backend

        screened = train_backend.read_screened_valid(ref_root)
        out["screened"] = bool(screened.get("screened"))
        out["valid"] = list(screened.get("valid") or [])
        out["invalid"] = list(screened.get("invalid") or [])
        out["valid_count"] = int(screened.get("valid_count") or 0)
        out["invalid_count"] = int(screened.get("invalid_count") or 0)
        out["allowlist_source"] = screened.get("source")
        if screened.get("invalid"):
            out["invalid_tip"] = f"[{','.join(str(x) for x in screened['invalid'])}]为invalid"
    except Exception:  # noqa: BLE001
        out.setdefault("screened", False)
    return out


def script_defaults(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    catalog = catalog or load_infer_catalog()
    return dict(catalog.get("defaults") or {})


def find_skill(skill_id: str, catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    catalog = catalog or load_infer_catalog()
    return next((s for s in (catalog.get("skills") or []) if s.get("id") == skill_id), None)


def resolve_phi0_py(explicit: str | None = None, catalog: dict[str, Any] | None = None) -> str:
    """Prefer an executable local Python; fall back through known Phi-0 envs."""
    catalog = catalog or load_infer_catalog()
    defaults = script_defaults(catalog)
    candidates = [
        explicit,
        defaults.get("phi0_py"),
        "/home/neotix/miniforge3/envs/Phi-0-wpy/bin/python",
        str(Path.home() / "miniforge3/envs/Phi-0-wpy/bin/python"),
        str(Path.home() / "anaconda3/envs/Phi-0-wpy/bin/python"),
        "/home/neotix/noetix/conda-envs/Phi-0-wbc-newton-wpy/bin/python",
        "python3",
    ]
    for cand in candidates:
        if not cand:
            continue
        p = Path(str(cand))
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        # bare command name
        if "/" not in str(cand):
            return str(cand)
    return "python3"


def build_infer_env(job: dict[str, Any], catalog: dict[str, Any]) -> dict[str, str]:
    params = job.get("params") or {}
    defaults = script_defaults(catalog)
    horizon = int(params.get("horizon") or defaults.get("horizon") or 32)
    rtc_delay = int(params.get("rtc_inference_delay") or defaults.get("rtc_inference_delay") or 6)
    rtc_exec = int(params.get("rtc_execution_horizon") or defaults.get("rtc_execution_horizon") or 26)
    use_rtc = params.get("use_rtc")
    if use_rtc is None:
        use_rtc = defaults.get("use_rtc", True)
    env: dict[str, str] = {
        "STUDENT_CKPT": str(job["student_ckpt"]),
        "REF_ROOT": str(job["ref_root"]),
        "EP": str(int(job.get("ep") or 0)),
        "HORIZON": str(horizon),
        "TAG": str(job.get("tag") or f"studio_cl_{job.get('stamp')}"),
        "OUT": str(job["out_dir"]),
        "CUDA_VISIBLE_DEVICES": str(params.get("cuda_devices") or defaults.get("cuda_devices") or "0"),
        "PHI0_PY": resolve_phi0_py(str(params.get("phi0_py") or "") or None, catalog),
        "DEPLOY_POLICY_DIR": str(params.get("deploy_policy") or defaults.get("deploy_policy") or "sonic_v1_1"),
        "PHI0_HAND_MODE": str(params.get("hand_mode") or defaults.get("hand_mode") or "dex3"),
        "PHI0_NEWTON_REVO2": "0",
        "PHI0_USE_VLM_FRAME_LATENT_CACHE": "1",
        "PHI0_VLM_FRAME_CACHE_SKIP_VIDEO": "1",
        "PHI0_CL_VLM_SOURCE": str(params.get("vlm_source") or defaults.get("vlm_source") or "frame_cache"),
        "PHI0_CL_HAND_OBS": str(
            params.get("hand_obs")
            or job.get("hand_obs")
            or defaults.get("hand_obs")
            or "commanded"
        ),
        "PHI0_ADALN_ZERO_VISION": "1",
        "PHI0_VLM_ATTN": "flash_attention_2",
        "PHI0_ALLOW_SDPA_FALLBACK": "0",
        "USE_RTC": "1" if use_rtc else "0",
        "PHI0_RTC_INFERENCE_DELAY": str(rtc_delay),
        "PHI0_RTC_EXECUTION_HORIZON": str(rtc_exec),
        "CONTROL_FPS": str(int(params.get("control_fps") or defaults.get("control_fps") or 50)),
        "RECORD_FPS": "50",
    }
    venv_sim = str(params.get("venv_sim") or defaults.get("venv_sim") or "").strip()
    if not venv_sim:
        cand = Path(str(job.get("phi0_root") or APP_DIR / "phi0_pipeline")) / ".venv_sim"
        if (cand / "bin" / "python").exists():
            venv_sim = str(cand.resolve())
    if venv_sim:
        env["VENV_SIM"] = venv_sim
    prompt = str(params.get("prompt") or "").strip()
    if prompt:
        env["PHI0_CL_PROMPT_OVERRIDE"] = prompt
        env["PROMPT"] = prompt
        env["TASK_PROMPT"] = prompt
    if job.get("valid_hand_root"):
        env["VALID_HAND_ROOT"] = str(job["valid_hand_root"])
    # Interactive MuJoCo window (drag orbit / zoom). egl = offscreen video only.
    gl = str(params.get("mujoco_gl") or job.get("mujoco_gl") or "").strip().lower()
    if not gl:
        gl = "glfw" if job.get("viewer_interactive", True) else "egl"
    env["MUJOCO_GL"] = gl if gl in {"glfw", "egl", "osmesa"} else "glfw"
    if env["MUJOCO_GL"] == "glfw":
        # Helpful on NVIDIA hosts when opening a real viewer window.
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    return env


def parse_infer_phase(log_text: str) -> dict[str, Any]:
    """Best-effort stage detection from mujoco_cl / studio_phase / sonic_latent logs."""
    text = log_text or ""
    # Prefer explicit STUDIO phase markers from interactive orchestrator.
    studio_order = [
        "starting",
        "sim_ready",
        "deploy_starting",
        "init_done",
        "standing",
        "policy_starting",
        "policy_ready",
        "streaming",
        "done",
        "stopped",
        "error",
    ]
    m = re.findall(r"STUDIO phase=([a-z_]+)", text, re.I)
    if m:
        phase = m[-1].lower()
        reached = []
        if phase in studio_order:
            idx = studio_order.index(phase)
            reached = studio_order[: idx + 1]
        else:
            reached = [phase]
        return {"phase": phase, "reached": reached, "mode": "interactive"}

    phases = [
        ("sim", r"sim up|starting sim|SIM_PID|run_sim_loop_vla_record|sim_ready"),
        ("publisher", r"ChunkStudent closed-loop|bound tcp://|model publisher ready"),
        ("deploy", r"Init Done|deploy Init Done|starting deploy|init_done"),
        ("armed", r"transitioning to CONTROL|ZMQ STREAMING MODE: ENABLED|deploy in CONTROL|standing|policy_ready"),
        ("policy", r"sent ZMQ command start|REPLAY_READY|touch.*replay_go|recording|RECORD|streaming"),
        ("done", r"done\.|finished|mp4 written|record stop"),
    ]
    reached = []
    for name, pat in phases:
        if re.search(pat, text, re.I):
            reached.append(name)
    current = reached[-1] if reached else "queued"
    return {"phase": current, "reached": reached, "mode": "auto"}


def studio_phase_from_status(status_json: str, log_text: str = "") -> dict[str, Any]:
    """Merge studio_status.json with log parsing."""
    phase_info = parse_infer_phase(log_text)
    try:
        data = json.loads(status_json or "{}")
        if isinstance(data, dict) and data.get("phase"):
            phase_info["phase"] = str(data["phase"])
            phase_info["message"] = str(data.get("message") or "")
            phase_info["mode"] = "interactive"
    except Exception:  # noqa: BLE001
        pass
    return phase_info


def _log_has_error(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"Traceback \(most recent call last\)|ERROR:|FATAL|Segmentation fault",
            text,
            re.I,
        )
    )


def infer_stack_status(
    *,
    phase: str = "",
    sim_text: str = "",
    replay_text: str = "",
    deploy_text: str = "",
) -> dict[str, Any]:
    """01-style lamps for MuJoCo sim / policy publisher / deploy(sim)."""
    phase = str(phase or "").lower()
    order = [
        "starting",
        "sim_ready",
        "deploy_starting",
        "init_done",
        "standing",
        "policy_starting",
        "policy_ready",
        "streaming",
        "done",
    ]
    try:
        idx = order.index(phase) if phase in order else -1
    except ValueError:
        idx = -1

    def _lamp(started: bool, text: str) -> str:
        if _log_has_error(text):
            return "error"
        if started:
            return "ok"
        return "idle"

    sim_on = idx >= order.index("sim_ready") or bool(
        re.search(r"sim loop starting|Sensor server running|sim pid=", sim_text, re.I)
    )
    policy_on = idx >= order.index("deploy_starting") or bool(
        re.search(r"bound tcp://|ChunkStudent|model publisher|replay pid=", replay_text, re.I)
    )
    deploy_on = idx >= order.index("deploy_starting") or bool(
        re.search(r"Init Done|deploy pid=|g1_deploy_onnx", deploy_text, re.I)
    )
    lights = {
        "sim": _lamp(sim_on, sim_text),
        "policy": _lamp(policy_on, replay_text),
        "deploy": _lamp(deploy_on, deploy_text),
    }
    return {
        "lights": lights,
        "sim": lights["sim"] == "ok",
        "policy": lights["policy"] == "ok",
        "deploy": lights["deploy"] == "ok",
    }


def infer_out_mp4(out_dir: str) -> str:
    return f"{out_dir.rstrip('/')}/student_sonic_dex3_cl.mp4"


def infer_work_dir(out_dir: str) -> str:
    return f"{out_dir.rstrip('/')}/mujoco"


def infer_preview_jpeg(out_dir: str) -> str:
    return f"{out_dir.rstrip('/')}/mujoco/preview.jpg"


INTERACTIVE_SCRIPT = "tools/eval/studio_cl_orchestrator.sh"
AUTO_SCRIPT = "tools/eval/run_830_walk_student_cl_mujoco_viz.sh"
MONITOR_SCRIPT = "tools/deploy/studio_zmq_monitor.py"
MONITOR_DEFAULT_OUT = "/tmp/studio_zmq_monitor"


def monitor_paths(out_dir: str | None = None) -> dict[str, str]:
    root = (out_dir or MONITOR_DEFAULT_OUT).rstrip("/")
    return {
        "out": root,
        "status": f"{root}/status.json",
        "camera": f"{root}/camera.jpg",
        "pid": f"{root}/monitor.pid",
    }
