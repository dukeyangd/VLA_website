"""01 offline collect stack via tmux + ZMQ record keys + optional camera preview."""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR / "scripts"

DEFAULT_COLLECT_CONFIG: dict[str, Any] = {
    "gmr_root": "/home/user/YZY/DataCollection/GMR",
    "conda_sh": "/home/user/anaconda3/etc/profile.d/conda.sh",
    "conda_env": "gmr",
    "groot_root": "/home/user/YZY/DataCollection/GR00T-WholeBodyControl",
    "data_venv": "/home/user/YZY/DataCollection/GR00T-WholeBodyControl/.venv_data_collection",
    "deploy_root": "/home/user/YZY/DataCollection/GR00T-WholeBodyControl/gear_sonic_deploy",
    "xrt_pybind": (
        "/home/user/YZY/DataCollection/GR00T-WholeBodyControl/"
        "external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    ),
    "camera_repo": "/home/user/GR00T-WholeBodyControl-hand",
    "camera_venv": "/home/user/GR00T-WholeBodyControl-hand/.venv_sim",
    "camera_host": "192.168.123.165",
    "camera_port": 5555,
    "task_prompt": "demo",
    "sonic_zmq_host": "localhost",
    "state_zmq_host": "localhost",
    "hand_zmq_host": "192.168.123.165",
    "hand_zmq_port": 5558,
    "g1_network_interface": "enp129s0",
    "record_left_wrist_camera": True,
    "record_g1_pelvis_odom": True,
    "deploy_policy_variant": "v1_1",
    "deploy_input_type": "zmq",
    "deploy_zmq_host": "localhost",
    "deploy_zmq_port": 5556,
    "deploy_zmq_topic": "pose",
    "deploy_interface": "real",
    "keyboard_zmq_port": 5580,
    "auto_start_stack": True,
    # Studio web「开始录制」only — ignore Pico left-grip+A manager_state toggles.
    "ignore_manager_record_toggle": True,
    # Camera server runs on robot; Studio only optional MJPEG preview subscriber.
    "start_camera_preview": True,
    # Pico-dex PC Service (XRoboToolkit) — background only, shared across collects.
    "pico_dex_root": "/opt/apps/roboticsservice",
    "auto_start_pico_dex": True,
}

# Managed tmux windows (camera is NOT launched — user starts it on the robot).
TMUX_ROLES = ("gmr", "deploy", "exporter")
# Persistent Pico PC Service session (not killed when collect stack stops).
PICO_DEX_TMUX_SESSION = "studio_pico_dex"

LogFn = Callable[[str], None]

ERROR_HINT = re.compile(
    r"(Traceback \(most recent call last\)|FATAL|Error:|ERROR:|No such file|"
    r"ModuleNotFoundError|Address already in use|Deployment cancelled|"
    r"command not found|Permission denied)",
    re.I,
)


class RecordKeyboardPublisher:
    """Long-lived ZMQ PUB for exporter keyboard commands (t/y/x)."""

    def __init__(self, port: int = 5580, bind: str = "*") -> None:
        self.port = int(port)
        self.bind = bind
        self._sock = None
        self._ctx = None
        self._lock = threading.Lock()
        self._ready = False
        self._error = ""

    def start(self) -> None:
        try:
            import zmq  # type: ignore
        except ImportError as exc:
            self._error = f"pyzmq missing: {exc}"
            return
        with self._lock:
            if self._ready:
                return
            self._ctx = zmq.Context.instance()
            sock = self._ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, 8)
            sock.bind(f"tcp://{self.bind}:{self.port}")
            self._sock = sock
            self._ready = True
            time.sleep(0.25)

    def send(self, key: str) -> None:
        key = str(key or "").strip().lower()[:1]
        if key not in {"t", "y", "x"}:
            raise ValueError("录制命令仅支持 t / y / x")
        with self._lock:
            if not self._ready or self._sock is None:
                raise RuntimeError(self._error or "键盘 ZMQ PUB 未启动")
            self._sock.send_string(key)

    def stop(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close(0)
                except Exception:  # noqa: BLE001
                    pass
            self._sock = None
            self._ready = False


_record_pub: RecordKeyboardPublisher | None = None
_record_pub_lock = threading.Lock()


def get_record_publisher(port: int = 5580) -> RecordKeyboardPublisher:
    global _record_pub
    with _record_pub_lock:
        if _record_pub is None or _record_pub.port != int(port):
            if _record_pub is not None:
                _record_pub.stop()
            _record_pub = RecordKeyboardPublisher(port=port)
            _record_pub.start()
        elif not _record_pub._ready:
            _record_pub.start()
        return _record_pub


def merge_collect_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_COLLECT_CONFIG)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in cfg and value is not None and value != "":
                cfg[key] = value
    return cfg


def work_dir_for(item: dict[str, Any]) -> Path:
    """Studio control dir for logs / studio_cmd / camera preview.

    Must NOT live inside the LeRobot session folder: Gr00tDataExporter.create()
    rmtree's incomplete dataset roots, which would delete .studio_collect and
    break Deploy mailbox writes (ENOENT on studio_cmd).
    Layout: ``{collect_root}/.studio_collect/{session_name}/``
    """
    local = Path(str(item.get("local_dir") or ".")).resolve()
    collect_root = str(item.get("collect_root") or "").strip()
    session_name = str(item.get("session_name") or local.name).strip() or local.name
    if collect_root:
        path = Path(collect_root).resolve() / ".studio_collect" / session_name
    else:
        path = local.parent / ".studio_collect" / local.name
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "camera").mkdir(parents=True, exist_ok=True)
    return path


def tmux_session_name(item: dict[str, Any]) -> str:
    cid = str(item.get("id") or "collect").replace("-", "_")
    return f"studio_{cid}"[:48]


def _append_log(path: Path, line: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line if line.endswith("\n") else line + "\n")
    except OSError:
        pass


def _run(args: list[str], timeout: int | None = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)


def _tmux(args: list[str], timeout: int | None = 30) -> subprocess.CompletedProcess[str]:
    return _run(["tmux", *args], timeout=timeout)


def tmux_available() -> bool:
    return _run(["bash", "-lc", "command -v tmux"], timeout=5).returncode == 0


def build_gmr_cmd(cfg: dict[str, Any]) -> str:
    """Matches user Terminal 1 (xsense + pico)."""
    gmr = shlex.quote(str(cfg["gmr_root"]))
    conda_sh = shlex.quote(str(cfg["conda_sh"]))
    env_name = shlex.quote(str(cfg["conda_env"]))
    xrt = str(cfg["xrt_pybind"])
    pico_root = str(cfg.get("pico_dex_root") or "/opt/apps/roboticsservice")
    return (
        f"source {conda_sh} && conda activate {env_name} && "
        f"cd {gmr} && "
        f"export __GLX_VENDOR_LIBRARY_NAME=nvidia && "
        f"export __NV_PRIME_RENDER_OFFLOAD=0 && "
        f"export MUJOCO_GL=glfw && "
        f"export XRT_PYBIND={shlex.quote(xrt)} && "
        f"export PYTHONPATH=\"$XRT_PYBIND:${{PYTHONPATH:-}}\" && "
        f"export LD_LIBRARY_PATH=\"$XRT_PYBIND/lib:{shlex.quote(pico_root)}/SDK/x64:${{LD_LIBRARY_PATH:-}}\" && "
        f"python scripts/xsens_live_streaming.py --zmq_version 3 --pico_hands"
    )


def pico_dex_process_running() -> bool:
    """True if RoboticsServiceProcess is alive (ignore pgrep/tmux wrappers)."""
    try:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                raw = (proc / "cmdline").read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            cmd = raw.replace(b"\x00", b" ").decode("utf-8", "ignore")
            if "RoboticsServiceProcess" not in cmd:
                continue
            if "pgrep" in cmd or "ensure_pico" in cmd:
                continue
            return True
    except OSError:
        pass
    return False


def build_pico_dex_monitor_cmd(cfg: dict[str, Any], log_path: Path | None = None) -> str:
    """Keep a silent tmux pane that only watches the already-detached PC Service."""
    root = shlex.quote(str(cfg.get("pico_dex_root") or "/opt/apps/roboticsservice"))
    log_q = shlex.quote(str(log_path)) if log_path else "/dev/null"
    return (
        f"echo '[pico-dex] monitor attached (service is detached/nohup)'; "
        f"echo \"[pico-dex] root={root}\"; "
        f"while true; do "
        f"if pgrep -a -f '/opt/apps/roboticsservice/RoboticsServiceProcess|./RoboticsServiceProcess' "
        f"2>/dev/null | grep -v pgrep | grep -v monitor >/dev/null; then "
        f"echo \"[pico-dex] $(date '+%H:%M:%S') alive\"; "
        f"else "
        f"echo \"[pico-dex] $(date '+%H:%M:%S') NOT running\"; "
        f"fi; "
        f"sleep 30; "
        f"done >>{log_q} 2>&1"
    )


def ensure_pico_dex_service(
    cfg: dict[str, Any],
    *,
    work_dir: Path | None = None,
    on_line: LogFn | None = None,
) -> dict[str, Any]:
    """Start Pico-dex PC Service detached (survives collect stop); optional silent tmux monitor."""
    cfg = merge_collect_config(cfg)
    root = Path(str(cfg.get("pico_dex_root") or "/opt/apps/roboticsservice"))
    script = root / "runService.sh"
    binary = root / "RoboticsServiceProcess"
    log_path = (work_dir / "logs" / "pico_dex.log") if work_dir else (APP_DIR / "state" / "pico_dex.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined = (work_dir / "logs" / "collect_stack.log") if work_dir else None

    def _log(msg: str) -> None:
        if combined:
            _append_log(combined, msg)
        _append_log(log_path, msg)
        if on_line:
            on_line(msg)

    if not cfg.get("auto_start_pico_dex", True):
        _log("[pico-dex] skipped (auto_start_pico_dex=false)")
        return {"ok": True, "skipped": True, "running": pico_dex_process_running()}

    if not script.is_file() and not binary.is_file():
        msg = f"[pico-dex] missing {script}"
        _log(msg)
        return {"ok": False, "message": msg, "running": False}

    if pico_dex_process_running():
        _log("[pico-dex] already running — skip relaunch")
        if tmux_available():
            listed = _run(["tmux", "has-session", "-t", PICO_DEX_TMUX_SESSION], timeout=5)
            if listed.returncode != 0:
                try:
                    _create_window(PICO_DEX_TMUX_SESSION, "pico_dex", first=True)
                    _send_cmd(PICO_DEX_TMUX_SESSION, "pico_dex", build_pico_dex_monitor_cmd(cfg, log_path))
                    _log(f"[pico-dex] monitor tmux={PICO_DEX_TMUX_SESSION}")
                except Exception as exc:  # noqa: BLE001
                    _log(f"[pico-dex] monitor tmux skipped: {exc}")
        return {
            "ok": True,
            "already": True,
            "running": True,
            "tmux_session": PICO_DEX_TMUX_SESSION,
        }

    # Detached start: same env as runService.sh, but process is NOT a tmux child
    # (killing collect / studio_pico_dex session must not SIGHUP the service).
    env = os.environ.copy()
    lib = f"{root}:{root / 'lib'}:{root / 'SDK' / 'x64'}"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["QT_PLUGIN_PATH"] = f"{root / 'plugins'}:{env.get('QT_PLUGIN_PATH', '')}"
    env["QT_QML_PATH"] = f"{root / 'qml'}:{env.get('QT_QML_PATH', '')}"
    if not env.get("DISPLAY"):
        env["DISPLAY"] = ":1"
    _log(f"[pico-dex] starting detached DISPLAY={env.get('DISPLAY')} cwd={root}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"$ RoboticsServiceProcess (detached)\n")
        logf.flush()
        subprocess.Popen(
            [str(binary)],
            cwd=str(root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    ok = False
    for _ in range(50):
        if pico_dex_process_running():
            ok = True
            break
        time.sleep(0.2)

    if ok:
        _log("[pico-dex] started (detached)")
    else:
        _log("[pico-dex] FAILED to start within timeout")
        return {"ok": False, "message": "pico-dex start timeout", "running": False}

    # Silent background terminal for attach/debug only — does not own the process.
    if tmux_available():
        try:
            _kill_session(PICO_DEX_TMUX_SESSION)
            _create_window(PICO_DEX_TMUX_SESSION, "pico_dex", first=True)
            _send_cmd(PICO_DEX_TMUX_SESSION, "pico_dex", build_pico_dex_monitor_cmd(cfg, log_path))
            _log(f"[pico-dex] monitor tmux={PICO_DEX_TMUX_SESSION} (attach: tmux a -t {PICO_DEX_TMUX_SESSION})")
        except Exception as exc:  # noqa: BLE001
            _log(f"[pico-dex] monitor tmux skipped: {exc}")

    return {
        "ok": True,
        "started": True,
        "running": True,
        "tmux_session": PICO_DEX_TMUX_SESSION,
        "mode": "detached",
    }


def build_exporter_cmd(cfg: dict[str, Any], *, collect_root: str, session_name: str) -> str:
    """Matches user Terminal 3 (exporter); binds Studio session dir."""
    groot = shlex.quote(str(cfg["groot_root"]))
    venv = shlex.quote(str(Path(cfg["data_venv"]) / "bin" / "activate"))
    # Default: only Studio「开始录制」controls episodes (ignore Pico grip+A manager toggle).
    ignore_mgr = "1" if cfg.get("ignore_manager_record_toggle", True) else "0"
    parts = [
        f"cd {groot}",
        f"source {venv}",
        f"export STUDIO_IGNORE_MANAGER_RECORD_TOGGLE={shlex.quote(ignore_mgr)}",
        (
            f"python gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt {shlex.quote(str(cfg['task_prompt']))} "
            f"--root-output-dir {shlex.quote(collect_root)} "
            f"--dataset-name {shlex.quote(session_name)} "
            f"--camera-host {shlex.quote(str(cfg['camera_host']))} "
            f"--sonic-zmq-host {shlex.quote(str(cfg['sonic_zmq_host']))} "
            f"--state-zmq-host {shlex.quote(str(cfg['state_zmq_host']))} "
            f"--hand-zmq-host {shlex.quote(str(cfg['hand_zmq_host']))} "
            f"--hand-zmq-port {int(cfg['hand_zmq_port'])} "
            f"--g1-network-interface {shlex.quote(str(cfg['g1_network_interface']))}"
        ),
    ]
    if cfg.get("record_left_wrist_camera", True):
        parts[-1] += " --record-left-wrist-camera"
    if cfg.get("record_g1_pelvis_odom", True):
        parts[-1] += " --record-g1-pelvis-odom"
    return " && ".join(parts)


def build_deploy_cmd(cfg: dict[str, Any], work_dir: Path | None = None) -> str:
    """Launch Studio deploy wrapper (mailbox y/]/Enter); falls back to raw deploy.sh."""
    wrapper = SCRIPTS_DIR / "studio_collect_deploy.sh"
    root = str(cfg["deploy_root"])
    if work_dir is not None and wrapper.is_file():
        # Stable control plane: skip /dev/tty Proceed; wait on studio_cmd.
        env = {
            "DEPLOY_ROOT": root,
            "STUDIO_CMD_DIR": str(work_dir),
            "POLICY_VARIANT": str(cfg.get("deploy_policy_variant") or "v1_1"),
            "INPUT_TYPE": str(cfg.get("deploy_input_type") or "zmq"),
            "ZMQ_HOST": str(cfg.get("deploy_zmq_host") or "localhost"),
            "ZMQ_PORT": str(cfg.get("deploy_zmq_port") or "5556"),
            "ZMQ_TOPIC": str(cfg.get("deploy_zmq_topic") or "pose"),
            "INTERFACE_MODE": str(cfg.get("deploy_interface") or "real"),
        }
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        return f"export {exports} && bash {shlex.quote(str(wrapper))}"
    # Legacy interactive deploy.sh (Proceed? via /dev/tty + tmux send-keys)
    return (
        f"cd {shlex.quote(root)} && bash deploy.sh "
        f"--policy-variant {shlex.quote(str(cfg['deploy_policy_variant']))} "
        f"--input-type {shlex.quote(str(cfg['deploy_input_type']))} "
        f"--zmq-host {shlex.quote(str(cfg['deploy_zmq_host']))} "
        f"--zmq-port {shlex.quote(str(cfg['deploy_zmq_port']))} "
        f"--zmq-topic {shlex.quote(str(cfg['deploy_zmq_topic']))} "
        f"{shlex.quote(str(cfg['deploy_interface']))}"
    )


def build_camera_preview_cmd(cfg: dict[str, Any], work_dir: Path) -> str:
    """Optional local MJPEG subscriber only (does not start robot camera)."""
    py = shlex.quote(str(Path(cfg["camera_venv"]) / "bin" / "python"))
    script = shlex.quote(str(SCRIPTS_DIR / "studio_camera_mjpeg.py"))
    out = shlex.quote(str(work_dir / "camera"))
    return (
        f"{py} {script} "
        f"--camera-host {shlex.quote(str(cfg['camera_host']))} "
        f"--camera-port {int(cfg['camera_port'])} "
        f"--out-dir {out}"
    )


def _pane_dead(session: str, window: str) -> bool | None:
    """True if pane dead, False if alive, None if unknown."""
    proc = _tmux(["list-panes", "-t", f"{session}:{window}", "-F", "#{pane_dead}"])
    if proc.returncode != 0:
        return None
    val = (proc.stdout or "").strip().splitlines()
    if not val:
        return None
    return val[0].strip() == "1"


def _log_has_error(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    try:
        text = log_path.read_text("utf-8", errors="replace")[-8000:]
    except OSError:
        return False
    return bool(ERROR_HINT.search(text))


def _role_state(session: str, role: str, work: Path) -> str:
    """Return ok | error | idle."""
    dead = _pane_dead(session, role)
    log_path = work / "logs" / f"{role}.log"
    if dead is None:
        return "idle"
    if dead:
        return "error" if _log_has_error(log_path) or log_path.is_file() else "error"
    if _log_has_error(log_path):
        return "error"
    return "ok"


def stack_status(handles: dict[str, Any] | None) -> dict[str, Any]:
    handles = handles or {}
    session = str(handles.get("tmux_session") or "")
    work = handles.get("work_dir")
    work_path = Path(work) if work else Path(".")
    lights: dict[str, str] = {}
    for role in TMUX_ROLES:
        lights[role] = _role_state(session, role, work_path) if session else "idle"
    # Camera: robot-side; optional local preview process
    preview = handles.get("preview_proc")
    if preview is not None and getattr(preview, "poll", lambda: 0)() is None:
        lights["camera"] = "ok"
    elif preview is not None:
        lights["camera"] = "error"
    else:
        lights["camera"] = "skip"
    return {
        **{k: lights.get(k) == "ok" for k in (*TMUX_ROLES, "camera")},
        "lights": lights,
        "tmux_session": session,
        "record_pub": bool((_record_pub and _record_pub._ready)),
        "work_dir": str(work or ""),
    }


def _kill_session(session: str) -> None:
    _tmux(["kill-session", "-t", session])


def _create_window(session: str, name: str, first: bool) -> None:
    if first:
        _tmux(["new-session", "-d", "-s", session, "-n", name])
    else:
        _tmux(["new-window", "-t", session, "-n", name])


def _send_cmd(session: str, window: str, cmd: str) -> None:
    target = f"{session}:{window}"
    # Clear any leftover prompt noise then run.
    _tmux(["send-keys", "-t", target, "C-c"], timeout=5)
    time.sleep(0.05)
    _tmux(["send-keys", "-t", target, cmd, "C-m"])


def _pipe_log(session: str, window: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    # Append pane output to log file.
    _tmux([
        "pipe-pane", "-t", f"{session}:{window}", "-o",
        f"cat >> {shlex.quote(str(log_path))}",
    ])


def start_collect_stack(
    item: dict[str, Any],
    cfg: dict[str, Any],
    *,
    on_line: LogFn | None = None,
) -> dict[str, Any]:
    """Launch GMR / Deploy / Exporter in separate tmux windows."""
    cfg = merge_collect_config(cfg)
    if not tmux_available():
        raise RuntimeError("未找到 tmux，请先安装：sudo apt install tmux")

    work = work_dir_for(item)
    combined = work / "logs" / "collect_stack.log"
    session = tmux_session_name(item)
    _kill_session(session)
    # Stale g1_deploy / busy ports are a common cause of “deploy won't start”.
    kill_g1_deploy()
    try:
        (work / "studio_phase").write_text("waiting_deploy\n", encoding="utf-8")
        (work / "studio_cmd").write_text("", encoding="utf-8")
    except OSError:
        pass

    # Pico-dex PC Service first (PICO App → 本机 IP); silent background tmux, survives stop.
    pico_info = ensure_pico_dex_service(cfg, work_dir=work, on_line=on_line)
    if not pico_info.get("ok"):
        _append_log(
            combined,
            f"[pico-dex] warn: {pico_info.get('message') or 'start failed'} — continue stack",
        )

    get_record_publisher(int(cfg.get("keyboard_zmq_port") or 5580))

    collect_root = str(item.get("collect_root") or "")
    session_name = str(item.get("session_name") or Path(item["local_dir"]).name)

    cmds = {
        "gmr": build_gmr_cmd(cfg),
        "deploy": build_deploy_cmd(cfg, work),
        "exporter": build_exporter_cmd(cfg, collect_root=collect_root, session_name=session_name),
    }

    first = True
    for role in TMUX_ROLES:
        _create_window(session, role, first=first)
        first = False
        log_path = work / "logs" / f"{role}.log"
        _append_log(combined, f"[studio] tmux {session}:{role}")
        _append_log(log_path, f"$ {cmds[role]}")
        if on_line:
            on_line(f"[studio] tmux start {role}")
        _pipe_log(session, role, log_path)
        _send_cmd(session, role, cmds[role])
        time.sleep(0.35)

    # Optional local preview subscriber (robot camera must already be publishing).
    preview_proc = None
    if cfg.get("start_camera_preview", True):
        preview_cmd = build_camera_preview_cmd(cfg, work)
        preview_log = work / "logs" / "camera_preview.log"
        _append_log(preview_log, f"$ {preview_cmd}")
        preview_proc = subprocess.Popen(
            ["bash", "-lc", preview_cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def _drain() -> None:
            assert preview_proc and preview_proc.stdout
            for line in preview_proc.stdout:
                _append_log(combined, f"[preview] {line.rstrip()}")

        threading.Thread(target=_drain, daemon=True).start()

    _append_log(combined, f"[studio] tmux session={session} roles={','.join(TMUX_ROLES)}")
    if on_line:
        on_line(f"[studio] tmux session {session} ready — Deploy 网页按 y → ] → Enter（mailbox）")

    return {
        "tmux_session": session,
        "work_dir": work,
        "combined_log": combined,
        "cfg": cfg,
        "preview_proc": preview_proc,
        "cmds": cmds,
        "pico_dex": pico_info,
        "deploy_mode": "wrapper" if (SCRIPTS_DIR / "studio_collect_deploy.sh").is_file() else "legacy",
    }


def stop_collect_stack(handles: dict[str, Any] | None, *, kill_deploy: bool = True) -> None:
    if not handles:
        return
    work = handles.get("work_dir")
    if isinstance(work, Path):
        try:
            (work / "camera" / "DONE").write_text("1\n", encoding="utf-8")
        except OSError:
            pass
    session = str(handles.get("tmux_session") or "")
    if session:
        _kill_session(session)
    preview = handles.get("preview_proc")
    if preview is not None and preview.poll() is None:
        try:
            os.killpg(preview.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                preview.terminate()
            except Exception:  # noqa: BLE001
                pass
    if kill_deploy:
        kill_g1_deploy()


def write_deploy_cmd(work_dir: Path, cmd: str, *, tmux_session: str = "") -> Path:
    """Send interactive keys via studio_cmd mailbox (wrapper) or tmux (legacy deploy.sh)."""
    raw = str(cmd or "").strip()
    mapped = raw
    keys: list[str] = []
    if raw in {"]", "stand", "stand_up"}:
        mapped = "stand"
        keys = ["]"]
    elif raw in {"y", "Y", "deploy"}:
        mapped = "deploy"
        keys = ["y"]
    elif raw in {"enter", "stream", ""}:
        mapped = "stream"
        keys = ["Enter"]
    elif raw.startswith("key:"):
        mapped = raw
        keys = [raw[4:]]
    else:
        mapped = raw
        keys = [raw] if raw else []

    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    cmd_path = path / "studio_cmd"
    cmd_path.write_text(mapped + "\n", encoding="utf-8")

    # Wrapper mode reads studio_cmd; avoid also injecting into the pane (can race /dev/tty).
    phase_path = path / "studio_phase"
    wrapper_mode = False
    if phase_path.is_file():
        try:
            phase = phase_path.read_text("utf-8").strip().splitlines()[0].strip()
            wrapper_mode = phase in {
                "waiting_deploy", "deploy_starting", "deploy_running",
                "init_done", "standing", "streaming",
            }
        except OSError:
            wrapper_mode = False

    session = tmux_session.strip()
    if session and keys and not wrapper_mode:
        target = f"{session}:deploy"
        for k in keys:
            if k == "Enter":
                _tmux(["send-keys", "-t", target, "Enter"])
            else:
                if mapped == "deploy":
                    _tmux(["send-keys", "-t", target, k, "Enter"])
                else:
                    _tmux(["send-keys", "-t", target, "-l", k])
    return cmd_path


def kill_g1_deploy() -> str:
    proc = subprocess.run(
        ["bash", "-lc", 'pkill -9 -f "g1_deploy" || true'],
        text=True,
        capture_output=True,
        check=False,
    )
    return (proc.stdout or proc.stderr or '已执行 pkill -9 -f "g1_deploy"').strip()


_DEPLOY_SPAM = re.compile(
    r"\[ZMQEndpointInterface\]|\[StreamedMotionMerger\]|^Loop timing -|"
    r"^Motion streamed completed|Decode interval:|Merged streamed data|"
    r"Decoded body quaternions|Decoded smpl_|Raw message field|"
    r"Protocol v3: Received|catch_up field not present|"
    r"Left hand joints set:|Right hand joints set:|"
    r"^  Frame\[|joint_pos:|body_quat:|smpl_joints:|smpl_pose:|"
    r"Decoded data \(Version|incoming_frame_start|Copying old data|"
    r"Merged motion:|Processing \d+ frames",
    re.I,
)
_DEPLOY_PHASE_ORDER = (
    "starting",
    "waiting_deploy",
    "deploy_starting",
    "deploy_running",
    "init_done",
    "standing",
    "streaming",
    "stopped",
)
_DEPLOY_PHASE_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("streaming", re.compile(r"ZMQ STREAMING MODE:\s*ENABLED", re.I)),
    ("init_done", re.compile(r"\bInit Done\b")),
    ("deploy_starting", re.compile(r"g1_deploy|Starting deploy|Proceed\?|deploy_starting|studio_deploy.*got cmd", re.I)),
    ("waiting_deploy", re.compile(r"waiting for web command|waiting_deploy", re.I)),
]


def _phase_rank(phase: str) -> int:
    try:
        return _DEPLOY_PHASE_ORDER.index(phase)
    except ValueError:
        return -1


def _read_text_tail(path: Path, max_bytes: int = 256_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                data = fh.read()
            else:
                data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_text_head(path: Path, max_bytes: int = 128_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def infer_deploy_phase(work_dir: str | Path) -> str:
    """Detect deploy phase from studio_phase + deploy.log milestones."""
    work = Path(work_dir)
    phase_path = work / "studio_phase"
    phase = ""
    if phase_path.is_file():
        try:
            phase = phase_path.read_text("utf-8").strip().splitlines()[0].strip()
        except OSError:
            phase = ""
    # Already past init — trust persisted phase unless log shows later milestone.
    deploy_log = work / "logs" / "deploy.log"
    if not deploy_log.is_file():
        return phase or "starting"

    # Init Done appears early; ZMQ spam grows the file huge — scan head + recent tail.
    blob = _read_text_head(deploy_log, 160_000) + "\n" + _read_text_tail(deploy_log, 120_000)
    found = phase
    for name, pat in _DEPLOY_PHASE_MARKERS:
        if pat.search(blob) and _phase_rank(name) >= _phase_rank(found):
            found = name
    if found and found != phase:
        try:
            phase_path.write_text(found + "\n", encoding="utf-8")
        except OSError:
            pass
    return found or phase or "starting"


def format_deploy_terminal(work_dir: str | Path, *, max_chars: int = 14_000) -> str:
    """Curated Deploy Terminal view: keep Init Done visible, drop ZMQ spam."""
    work = Path(work_dir)
    deploy_log = work / "logs" / "deploy.log"
    if not deploy_log.is_file():
        stack = work / "logs" / "collect_stack.log"
        if stack.is_file():
            return stack.read_text("utf-8", errors="replace")[-max_chars:]
        return ""

    head = _read_text_head(deploy_log, 200_000)
    tail = _read_text_tail(deploy_log, 400_000)
    phase = infer_deploy_phase(work)

    milestones: list[str] = []
    for line in head.splitlines():
        if re.search(r"Init Done|Proceed\?|g1_deploy|ZMQ STREAMING MODE|Waiting for|ERROR|Traceback|killed|已杀死", line, re.I):
            milestones.append(line.rstrip())
    # Dedup while preserving order
    seen: set[str] = set()
    uniq_m: list[str] = []
    for line in milestones:
        if line in seen:
            continue
        seen.add(line)
        uniq_m.append(line)

    kept_tail: list[str] = []
    for line in tail.splitlines():
        s = line.rstrip()
        if not s:
            continue
        if _DEPLOY_SPAM.search(s):
            continue
        kept_tail.append(s)
    kept_tail = kept_tail[-180:]

    banner = f"[studio] deploy_phase={phase}"
    if phase == "init_done":
        banner += "  → 请点 ] 站立"
    elif phase == "standing":
        banner += "  → 可按 Enter 开始流式"
    elif phase in {"starting", "waiting_deploy", "deploy_starting"}:
        banner += "  → 先点 y Deploy，等待 Init Done"
    elif phase == "streaming":
        banner += "  → 流式运行中"

    parts = [banner, "", "=== milestones ==="]
    parts.extend(uniq_m[-40:] or ["(尚未捕获到 Init Done / Proceed)"])
    parts.append("")
    parts.append("=== recent (filtered) ===")
    head_text = "\n".join(parts)
    budget = max(2000, max_chars - len(head_text) - 1)
    # Keep milestones; only trim the noisy recent section.
    recent = "\n".join(kept_tail or ["(暂无非刷屏日志)"])
    if len(recent) > budget:
        recent = recent[-budget:]
    return head_text + "\n" + recent


def read_collect_logs(work_dir: str | Path, max_bytes: int = 120_000) -> str:
    work = Path(work_dir)
    # Deploy Terminal is the primary surface — prefer curated deploy view.
    deploy_view = format_deploy_terminal(work, max_chars=min(max_bytes, 20_000))
    chunks: list[str] = []
    if deploy_view.strip():
        chunks.append(deploy_view)
    for role in ("gmr", "exporter", "camera_preview"):
        path = work / "logs" / f"{role}.log"
        if path.is_file():
            try:
                text = path.read_text("utf-8", errors="replace")
                if text.strip():
                    chunks.append(f"===== {role} =====\n{text[-12_000:]}")
            except OSError:
                pass
    text = "\n\n".join(chunks)
    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        text = text[-max_bytes:]
    return text


def read_camera_jpeg(work_dir: str | Path, slot: str) -> tuple[bytes | None, str]:
    cam_dir = Path(work_dir) / "camera"
    meta = cam_dir / "cameras.json"
    keys: list[str] = []
    if meta.is_file():
        try:
            import json

            keys = list(json.loads(meta.read_text("utf-8")).get("keys") or [])
        except Exception:  # noqa: BLE001
            keys = []
    prefer = {
        "cam0": keys[0] if keys else "ego_view",
        "cam1": keys[1] if len(keys) > 1 else (keys[0] if keys else "left_wrist"),
    }.get(slot, slot)
    candidates = [prefer, slot]
    if prefer == "ego_view":
        candidates += ["ego_view", "head", "head_color_image"]
    if prefer == "left_wrist" or slot == "cam1":
        candidates += ["left_wrist", "left_wrist_color_image"]
    for name in candidates:
        path = cam_dir / f"{name}.jpg"
        if path.is_file() and path.stat().st_size > 0:
            return path.read_bytes(), name
    for path in sorted(cam_dir.glob("*.jpg")):
        if path.name.startswith("."):
            continue
        return path.read_bytes(), path.stem
    return None, prefer
