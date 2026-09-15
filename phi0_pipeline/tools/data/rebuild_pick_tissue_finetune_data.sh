#!/usr/bin/env bash
# Rebuild pick-tissue valid LeRobot data (512-d unified source).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_GR00T="$(cd "${SCRIPT_DIR}/../../Isaac-GR00T" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-${PHI0_ROOT}/.venv-openpi/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

DATA_ROOT="${DATA_ROOT:-${ISAAC_GR00T}/data}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/pick_tissues.json}"
VALID_ROOT="${VALID_ROOT:-${DATA_ROOT}/pick_tissue_valid}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${ISAAC_GR00T}/examples/G1_SONIC/g1_sonic_ego_left_wrist_config.py}"

export PYTHONPATH="${PHI0_ROOT}/src:${PYTHONPATH:-}"

echo "[rebuild] step 1/2: merge valid episodes -> ${VALID_ROOT}"
"${PYTHON}" "${ISAAC_GR00T}/scripts/prepare_pick_tissue_dataset.py" \
  --manifest-path "${MANIFEST}" \
  --raw-root "${DATA_ROOT}" \
  --dst-root "${VALID_ROOT}"

echo "[rebuild] step 2/2: GR00T stats for ${VALID_ROOT}"
GR00T_PYTHON="${GR00T_PYTHON:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
if [[ -x "${GR00T_PYTHON}" && -f "${ISAAC_GR00T}/gr00t/data/stats.py" ]]; then
  (cd "${ISAAC_GR00T}" && "${GR00T_PYTHON}" gr00t/data/stats.py \
    --dataset-path "${VALID_ROOT}" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path "${MODALITY_CONFIG}") || echo "[rebuild] WARN: GR00T stats failed (non-fatal)"
else
  echo "[rebuild] skip GR00T stats (python or stats.py not found)"
fi

echo "[rebuild] done (sonic_unified 100-d convert removed; use pick_tissue_unified / 512-d path)."
