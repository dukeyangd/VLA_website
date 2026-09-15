#!/usr/bin/env bash
# cluster_0 一键：检查 SONIC v1.1 → 缺则下载 → 冒烟测试
set -euo pipefail
QA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHI0_ROOT="${PHI0_ROOT:-/mnt/data2/wpy/workspace/phi-0-wbc-newton}"
GR00T_ROOT="${GR00T_ROOT:-/mnt/data2/wpy/workspace/GR00T-WholeBodyControl}"
PY="${CONDA_PREFIX_ENV:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy}/bin/python"
V11_CKPT="${GR00T_ROOT}/sonic_v1_1/last.pt"
V11_CFG="${GR00T_ROOT}/sonic_v1_1/config.yaml"

echo "[setup] groot=${GR00T_ROOT}"
if [[ ! -f "${V11_CKPT}" || ! -f "${V11_CFG}" ]]; then
  echo "[setup] SONIC v1.1 missing → downloading from HuggingFace..."
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy='*' NO_PROXY='*'
  HF_HUB_ENABLE_HF_TRANSFER=0 "${PY}" "${PHI0_ROOT}/scripts/download_sonic_v1_1.py" --groot "${GR00T_ROOT}"
else
  echo "[setup] SONIC v1.1 OK: ${V11_CKPT}"
fi

echo "[setup] ensure trl>=0.28 for v1.1 ckpt (cluster may have 0.24)..."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*' NO_PROXY='*'
"${PY}" -m pip install -q 'trl>=0.28.0' 2>/dev/null || true

echo "[setup] smoke test ep0 headless..."
cd "${QA_ROOT}"
EPISODES=0 LIVE_UI=0 LOOP=0 FOREGROUND=1 MAX_STEPS=100 \
  bash run_qa.sh
echo "[setup] done — check experiments/ for summary"
