#!/usr/bin/env bash
# Source before training / smoke tests: sets local workspace paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-${HF_ENDPOINT}}"
export PHI0_ROOT="${PHI0_ROOT:-$ROOT}"
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-$(cd "${PHI0_ROOT}/.." && pwd)}"
# Always prefer in-tree subpackages; ignore stale env overrides that point elsewhere.
export PHI0_SUBPACKAGES="${PHI0_ROOT}/subpackages"
if [[ -n "${PHI0_SUBPACKAGES_OVERRIDE:-}" && -d "${PHI0_SUBPACKAGES_OVERRIDE}/gear_sonic" ]]; then
  export PHI0_SUBPACKAGES="${PHI0_SUBPACKAGES_OVERRIDE}"
fi
if [[ -z "${PHI0_PY:-}" || ! -x "${PHI0_PY}" ]]; then
  for _py in \
    "${HOME}/anaconda3/envs/Phi-0-wpy/bin/python" \
    "/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "${_py}" && -x "${_py}" ]]; then
      export PHI0_PY="${_py}"
      break
    fi
  done
fi
if [[ -z "${PHI0_PY:-}" || ! -x "${PHI0_PY}" ]]; then
  echo "setup_env: set PHI0_PY to a working Python (Phi-0-wpy)" >&2
  exit 1
fi
# GR00T_ROOT = in-tree subpackages only (no sibling GR00T-WholeBodyControl fallback).
export GR00T_ROOT="${PHI0_SUBPACKAGES}"
_SIM_REC="${GR00T_ROOT}/experiments/sonic_vla_overfit/scripts/run_sim_loop_vla_record.py"
if [[ ! -f "${_SIM_REC}" ]]; then
  echo "setup_env: warn missing ${_SIM_REC} (ok for offline/encode-only)" >&2
fi
unset _SIM_REC
export GEAR_SONIC_DEPLOY="${GEAR_SONIC_DEPLOY:-${GR00T_ROOT}/gear_sonic_deploy}"
export FASTWAM_SRC="${FASTWAM_SRC:-${PHI0_SUBPACKAGES}}"
export VENV_SIM="${VENV_SIM:-${PHI0_ROOT}/.venv_sim}"
export PYTHONPATH="${PHI0_ROOT}/src:${PHI0_SUBPACKAGES}:${PYTHONPATH:-}"
export TensorRT_ROOT="${TensorRT_ROOT:-/mnt/data2/TensorRT-10.13.3.9}"
export onnxruntime_ROOT="${onnxruntime_ROOT:-${PHI0_WORKSPACE}/deps/onnxruntime-linux-x64-1.16.3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# g1_deploy_onnx_ref links unitree_sdk2 DDS + TensorRT at runtime
_UNITREE_SDK2_LIB="${GEAR_SONIC_DEPLOY}/thirdparty/unitree_sdk2/thirdparty/lib/x86_64"
_VENV_SIM_CYCLONEDDS="${VENV_SIM}/lib/python3.10/site-packages/cyclonedds/.libs"
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${onnxruntime_ROOT}/lib:${_UNITREE_SDK2_LIB}:${_VENV_SIM_CYCLONEDDS}:${LD_LIBRARY_PATH:-}"
# unified ep447 -> valid ep544; avoids syncing 1.1G pick_tissue_valid/
export VALID_EP="${VALID_EP:-544}"
# MuJoCo sim DDS bridge (editable install into .venv_sim); optional
export UNITREE_SDK2_PYTHON="${UNITREE_SDK2_PYTHON:-${PHI0_SUBPACKAGES}/external_dependencies/unitree_sdk2_python}"
