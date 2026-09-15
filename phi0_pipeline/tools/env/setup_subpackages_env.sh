#!/usr/bin/env bash
# Bootstrap in-tree subpackages: venv_sim, deploy ONNX weights, optional compile.
# Usage:
#   bash tools/env/setup_subpackages_env.sh
#   bash tools/env/setup_subpackages_env.sh --sync-weights-from /path/to/GR00T-WholeBodyControl
#   bash tools/env/setup_subpackages_env.sh --download-weights   # needs WEIGHTS_URL_BASE
#   bash tools/env/setup_subpackages_env.sh --build-deploy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

SYNC_FROM=""
DOWNLOAD=0
BUILD_DEPLOY=0
CREATE_VENV=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync-weights-from)
      SYNC_FROM="${2:?}"
      shift 2
      ;;
    --download-weights)
      DOWNLOAD=1
      shift
      ;;
    --build-deploy)
      BUILD_DEPLOY=1
      shift
      ;;
    --skip-venv)
      CREATE_VENV=0
      shift
      ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

SP="${PHI0_SUBPACKAGES:-${PHI0_ROOT}/subpackages}"
DEPLOY="${GEAR_SONIC_DEPLOY:-${SP}/gear_sonic_deploy}"

echo "[subpackages] PHI0_ROOT=${PHI0_ROOT}"
echo "[subpackages] SP=${SP}"
echo "[subpackages] DEPLOY=${DEPLOY}"

for req in fastwam gear_sonic gear_sonic_deploy; do
  if [[ ! -d "${SP}/${req}" ]]; then
    echo "ERROR: missing ${SP}/${req}" >&2
    exit 1
  fi
done
test -f "${SP}/fastwam/models/wan22/wan_video_dit.py"
test -f "${SP}/gear_sonic/__init__.py"
test -d "${DEPLOY}/g1/meshes"
test -f "${DEPLOY}/g1/g1_29dof_with_hand.xml"
echo "[subpackages] tree OK (fastwam + gear_sonic + g1 meshes/hand)"

_sync_weights() {
  local src="$1"
  local src_deploy="${src}/gear_sonic_deploy"
  if [[ ! -d "${src_deploy}" ]]; then
    echo "ERROR: no gear_sonic_deploy under ${src}" >&2
    exit 1
  fi
  echo "[subpackages] rsync policy/planner from ${src_deploy}"
  mkdir -p "${DEPLOY}/policy" "${DEPLOY}/planner"
  if [[ -d "${src_deploy}/policy" ]]; then
    rsync -a --exclude='*.md' "${src_deploy}/policy/" "${DEPLOY}/policy/"
  fi
  if [[ -d "${src_deploy}/planner" ]]; then
    rsync -a --exclude='*.md' "${src_deploy}/planner/" "${DEPLOY}/planner/"
  fi
  # Optional: prebuilt binary
  if [[ -x "${src_deploy}/target/release/g1_deploy_onnx_ref" ]]; then
    mkdir -p "${DEPLOY}/target/release"
    rsync -a "${src_deploy}/target/release/g1_deploy_onnx_ref" "${DEPLOY}/target/release/"
    echo "[subpackages] synced g1_deploy_onnx_ref binary"
  fi
  # Optional: cluster .venv_sim as seed
  if [[ "${CREATE_VENV}" == "1" && -d "${src}/.venv_sim" && ! -d "${PHI0_ROOT}/.venv_sim" ]]; then
    echo "[subpackages] seeding .venv_sim from ${src}/.venv_sim"
    rsync -a "${src}/.venv_sim/" "${PHI0_ROOT}/.venv_sim/"
  fi
}

_download_weights() {
  local base="${WEIGHTS_URL_BASE:-}"
  if [[ -z "${base}" ]]; then
    echo "ERROR: set WEIGHTS_URL_BASE to a directory/URL prefix hosting policy+planner tarballs" >&2
    echo "  or use --sync-weights-from /path/to/existing/GR00T-WholeBodyControl" >&2
    exit 1
  fi
  mkdir -p "${DEPLOY}/policy" "${DEPLOY}/planner" "${PHI0_ROOT}/.download_cache"
  local cache="${PHI0_ROOT}/.download_cache"
  echo "[subpackages] downloading weights from ${base}"
  # Convention: policy_release.tar.gz, policy_low_latency.tar.gz, planner_sonic.tar.gz
  for name in policy_release.tar.gz policy_low_latency.tar.gz planner_sonic.tar.gz; do
    local url="${base%/}/${name}"
    local out="${cache}/${name}"
    if [[ ! -f "${out}" ]]; then
      if command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
      else
        curl -L -o "${out}" "${url}"
      fi
    fi
  done
  tar -xzf "${cache}/policy_release.tar.gz" -C "${DEPLOY}/policy"
  tar -xzf "${cache}/policy_low_latency.tar.gz" -C "${DEPLOY}/policy"
  tar -xzf "${cache}/planner_sonic.tar.gz" -C "${DEPLOY}/planner"
}

if [[ -n "${SYNC_FROM}" ]]; then
  _sync_weights "${SYNC_FROM}"
elif [[ "${DOWNLOAD}" == "1" ]]; then
  _download_weights
else
  # Default on this cluster: sync from sibling GR00T if present and weights missing
  _sibling="${PHI0_WORKSPACE}/GR00T-WholeBodyControl"
  if [[ ! -f "${DEPLOY}/policy/release/model_decoder.onnx" && ! -f "${DEPLOY}/policy/release/model.onnx" ]]; then
    if [[ -d "${_sibling}/gear_sonic_deploy/policy" ]]; then
      echo "[subpackages] no in-tree policy weights; auto-sync from ${_sibling}"
      _sync_weights "${_sibling}"
    else
      echo "[subpackages] WARN: deploy ONNX not found; run with --sync-weights-from or --download-weights" >&2
    fi
  else
    echo "[subpackages] policy weights already present"
  fi
fi

if [[ "${CREATE_VENV}" == "1" ]]; then
  export VENV_SIM="${PHI0_ROOT}/.venv_sim"
  if [[ ! -d "${VENV_SIM}" ]]; then
    echo "[subpackages] creating ${VENV_SIM}"
    if command -v uv >/dev/null 2>&1; then
      uv venv "${VENV_SIM}" --python 3.10
      # Minimal sim deps; full stack may still need rsync from a known-good .venv_sim
      uv pip install --python "${VENV_SIM}/bin/python" mujoco tyro numpy msgpack pyzmq \
        -i https://pypi.tuna.tsinghua.edu.cn/simple || true
    else
      "${PHI0_PY}" -m venv "${VENV_SIM}"
      "${VENV_SIM}/bin/pip" install -q mujoco tyro numpy msgpack pyzmq \
        -i https://pypi.tuna.tsinghua.edu.cn/simple || true
    fi
  fi
  if [[ -d "${VENV_SIM}" ]]; then
    bash "${SCRIPT_DIR}/fix_venv_sim.sh" || echo "WARN: fix_venv_sim had issues" >&2
  fi
fi

if [[ "${BUILD_DEPLOY}" == "1" ]]; then
  if [[ -z "${TensorRT_ROOT:-}" || ! -d "${TensorRT_ROOT}" ]]; then
    echo "ERROR: TensorRT_ROOT required to --build-deploy" >&2
    exit 1
  fi
  echo "[subpackages] building g1_deploy_onnx_ref (see gear_sonic_deploy docs)"
  (
    cd "${DEPLOY}"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
  )
fi

echo "[subpackages] smoke import"
PYTHONPATH="${PHI0_ROOT}/src:${SP}:${PYTHONPATH:-}" "${PHI0_PY}" - <<'PY'
from fastwam.models.wan22.wan_video_dit import DiTBlock
import gear_sonic
from pathlib import Path
import os
root = Path(os.environ["PHI0_ROOT"])
mesh = root / "subpackages/gear_sonic_deploy/g1/meshes"
hand = root / "subpackages/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
assert mesh.is_dir() and any(mesh.glob("*.STL")), mesh
assert hand.is_file(), hand
print("ok DiTBlock", DiTBlock)
print("ok gear_sonic", gear_sonic.__file__)
print("ok g1 meshes", mesh)
print("ok hand mjcf", hand)
PY

echo "[subpackages] done"
