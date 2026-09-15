#!/usr/bin/env bash
# skill_2 SONIC v1.1 Newton 回放 QA
# =============================================================================
# >>> 设置区：直接改下面几行（绝对路径）<<<
# =============================================================================
# 可填「大目录」或「某个子集 session 目录」
DATA="/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy"
SESSION="2026-09-04-00-41-34"
# episode 选择（与下方命令行相同语法）；留空 = 该 session 全部 valid
#   闭区间: 0-19
#   单独选: "0 10 15" 或 0,10,15
EPISODES=""
# 1=只回放 skill_2.json / skill_2_labels.json 里的 valid（默认）
USE_VALID=1
# 0=默认 nodex 只驱躯干；1=换 with_hand/Dex3 USD 并写 teleop 手关节
REPLAY_HANDS=0
# =============================================================================
# 命令行：
#   bash run_qa.sh /大目录/.../skill_2_pico_pick_the_toy 2026-09-04-00-41-34
#   bash run_qa.sh /大目录/... 2026-09-04-00-41-34 0-19
#   bash run_qa.sh /大目录/... 2026-09-04-00-41-34 "0 10 15"
#   bash run_qa.sh /子集目录 0-19
#   bash run_qa.sh /子集目录 0 10 15
#   EPISODES="0 10" bash run_qa.sh /路径/到/session
# =============================================================================

set -euo pipefail

QA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 解析命令行 ---
# 形式 A: parent session [EPISODES...]
# 形式 B: session [EPISODES...]
_args=("$@")
if [[ ${#_args[@]} -ge 1 ]]; then
  DATA="${_args[0]}"
fi
DATA="${DATA%/}"

_is_parent() {
  [[ -f "$1/skill_2.json" && ! -d "$1/data/chunk-000" ]]
}
_is_session() {
  [[ -d "$1/data/chunk-000" ]]
}

SESSION="${SESSION:-}"
# 设置区已有 EPISODES；命令行传入时覆盖
_cli_eps=""

if [[ ${#_args[@]} -ge 1 ]]; then
  if _is_parent "${DATA}"; then
    if [[ ${#_args[@]} -lt 2 ]]; then
      echo "[QA] FATAL: 大目录需要指定子集名，例如:" >&2
      echo "  bash run_qa.sh ${DATA} 2026-09-04-00-41-34 0-19" >&2
      echo "  bash run_qa.sh ${DATA} 2026-09-04-00-41-34 \"0 10 15\"" >&2
      echo "  可选子集:" >&2
      ls -1 "${DATA}" | grep -E '^[0-9]' >&2 || true
      exit 1
    fi
    SESSION="${_args[1]}"
    if [[ ${#_args[@]} -ge 3 ]]; then
      _cli_eps="${_args[*]:2}"
    fi
  elif _is_session "${DATA}"; then
    SESSION="$(basename "${DATA}")"
    if [[ ${#_args[@]} -ge 2 ]]; then
      _cli_eps="${_args[*]:1}"
    fi
  else
    echo "[QA] FATAL: 路径既不是大目录(skill_2.json)也不是子集(data/chunk-000): ${DATA}" >&2
    exit 1
  fi
fi

if [[ -n "${_cli_eps}" ]]; then
  EPISODES="${_cli_eps}"
fi
EPISODES="${EPISODES-}"

if _is_parent "${DATA}"; then
  if [[ -z "${SESSION}" ]]; then
    echo "[QA] FATAL: 请设置 SESSION=子集目录名" >&2
    exit 1
  fi
  DATA_ROOT="${DATA}/${SESSION}"
else
  DATA_ROOT="${DATA}"
  SESSION="$(basename "${DATA_ROOT}")"
fi
DATA_ROOT="${DATA_ROOT%/}"

# --- cluster_0 / 本机 自动路径 ---
if [[ -d /mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830 ]]; then
  PHI0_ROOT="${PHI0_ROOT:-/mnt/data2/wpy/workspace/phi-0-wbc-newton}"
  GR00T_ROOT="${GR00T_ROOT:-/mnt/data2/wpy/workspace/GR00T-WholeBodyControl}"
  ISAACLAB_PATH_NEWTON="${ISAACLAB_PATH_NEWTON:-/mnt/data3/wpy/IsaacLab-3.0}"
  CONDA_ENV="${CONDA_ENV:-Phi-0-wbc-newton-wpy}"
  CONDA_PREFIX_ENV="${CONDA_PREFIX_ENV:-/mnt/data3/wpy/conda-envs/${CONDA_ENV}}"
else
  if [[ "${DATA_ROOT}" == /mnt/* ]] && [[ ! -d "${DATA_ROOT}" ]]; then
    DATA_ROOT="/home/neotix/noetix/dataprocess/skill_2_pico_new826_unified"
  fi
  PHI0_ROOT="${PHI0_ROOT:-/home/neotix/noetix/phi-0-wbc-newton}"
  GR00T_ROOT="${GR00T_ROOT:-/home/neotix/noetix/phi-0-810/GR00T-WholeBodyControl}"
  ISAACLAB_PATH_NEWTON="${ISAACLAB_PATH_NEWTON:-/home/neotix/noetix/IsaacLab-3.0}"
  CONDA_ENV="${CONDA_ENV:-Phi-0-wbc-newton-wpy}"
  CONDA_PREFIX_ENV="${CONDA_PREFIX_ENV:-/home/neotix/noetix/conda-envs/${CONDA_ENV}}"
fi
export PHI0_ROOT GR00T_ROOT ISAACLAB_PATH_NEWTON CONDA_ENV CONDA_PREFIX_ENV

OUT_BASE="${OUT_BASE:-${QA_ROOT}/experiments}"
DATASET_TAG="$(basename "$(dirname "${DATA_ROOT}")")_${SESSION}"
# 若 DATA_ROOT 本身就是带名字的 unified，用 basename 即可
if [[ ! -f "$(dirname "${DATA_ROOT}")/skill_2.json" ]]; then
  DATASET_TAG="$(basename "${DATA_ROOT}")"
fi

if [[ ! -f "${PHI0_ROOT}/scripts/run_newton_skill2_sonic_batch_replay.sh" ]]; then
  echo "[QA] FATAL: 找不到回放脚本: ${PHI0_ROOT}/scripts/run_newton_skill2_sonic_batch_replay.sh" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[QA] FATAL: 数据目录不存在: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/data/chunk-000" ]]; then
  echo "[QA] FATAL: 不是合法子集（缺 data/chunk-000）: ${DATA_ROOT}" >&2
  exit 1
fi

export DATA_ROOT
export LIVE_UI="${LIVE_UI:-1}"
export VISUALIZER="${VISUALIZER:-viser}"
export LOOP="${LOOP:-1}"
export LOOP_PAUSE="${LOOP_PAUSE:-1}"
export REPLAY_SPEED="${REPLAY_SPEED:-1.5}"
export REPLAY_HANDS="${REPLAY_HANDS:-0}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
export VISER_PORT="${VISER_PORT:-8080}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"
export USE_VALID="${USE_VALID:-1}"

# 展开 EPISODES（0-19 / "0 10" / 0,10）并与 valid 求交（原编号）
_EP_SEL="${EPISODES:-}"
EPISODES="$(
  python3 - "${DATA_ROOT}" "${USE_VALID}" "${_EP_SEL}" <<'PY'
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
use_valid = sys.argv[2].strip().lower() not in ("0", "false", "no", "off")
sel = sys.argv[3].strip()
session = root.name

def load_valid(root: Path):
    for path in (root / "skill_2_labels.json", root / "labels.json", root.parent / "skill_2.json"):
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("valid"), list):
            return sorted({int(x) for x in obj["valid"]})
        if isinstance(obj, dict) and session in obj and isinstance(obj[session], dict):
            v = obj[session].get("valid")
            if isinstance(v, list):
                return sorted({int(x) for x in v})
        if isinstance(obj, dict) and len(obj) == 1:
            only = next(iter(obj.values()))
            if isinstance(only, dict) and isinstance(only.get("valid"), list):
                return sorted({int(x) for x in only["valid"]})
    return None

def expand(sel: str) -> list[int]:
    if not sel:
        return []
    out: list[int] = []
    for tok in re.split(r"[\s,]+", sel.strip()):
        if not tok:
            continue
        if re.fullmatch(r"\d+-\d+", tok):
            a, b = (int(x) for x in tok.split("-", 1))
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    seen: set[int] = set()
    uniq: list[int] = []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq

valid = load_valid(root) if use_valid else None
cand = expand(sel)
if not cand:
    cand = list(valid) if valid is not None else list(range(0, 20))

if valid is not None:
    allow = set(valid)
    ids = [e for e in cand if e in allow]
else:
    ids = cand

if not ids:
    print("EMPTY", file=sys.stderr)
    raise SystemExit(2)
print(" ".join(str(e) for e in ids))
PY
)" || {
  echo "[QA] FATAL: EPISODES∩valid 为空（检查 EPISODES='${_EP_SEL}' 与 skill_2.json）" >&2
  exit 1
}
export EPISODES
unset EP_START EP_END EP_FROM EP_TO || true

N_EP="$(echo "${EPISODES}" | wc -w | tr -d ' ')"

if [[ -z "${OUT:-}" ]]; then
  if [[ -n "${_EP_SEL}" ]]; then
    _tag_eps="$(echo "${_EP_SEL}" | tr ' ,' '__' | tr -cd '0-9a-zA-Z._-')"
    TAG="${DATASET_TAG}_ep${_tag_eps}_n${N_EP}"
  else
    TAG="${DATASET_TAG}_valid_n${N_EP}"
  fi
  export OUT="${OUT_BASE}/${TAG}"
fi
mkdir -p "${OUT}"

# 清掉上一次 Viser / 回放残留
_free_port() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 0.4
    return 0
  fi
  if [[ -n "${pids}" ]]; then
    echo "[QA] 清理端口 ${port}"
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 0.5
    # shellcheck disable=SC2086
    kill -KILL ${pids} 2>/dev/null || true
    sleep 0.3
  fi
}
_old_replay="$(pgrep -f 'newton_skill2_sonic_batch_replay.py' 2>/dev/null || true)"
if [[ -n "${_old_replay}" ]]; then
  echo "[QA] 结束上次回放"
  # shellcheck disable=SC2086
  kill -TERM ${_old_replay} 2>/dev/null || true
  sleep 0.5
  # shellcheck disable=SC2086
  kill -KILL ${_old_replay} 2>/dev/null || true
fi
_free_port "${VISER_PORT}"

echo "[QA] ${SESSION}  valid筛选=${USE_VALID}  n=${N_EP}  http://127.0.0.1:${VISER_PORT}"
echo "[QA] data=${DATA_ROOT}"
echo "[QA] out=${OUT}"

exec bash "${PHI0_ROOT}/scripts/run_newton_skill2_sonic_batch_replay.sh"
