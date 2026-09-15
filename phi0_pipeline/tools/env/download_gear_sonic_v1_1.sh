#!/usr/bin/env bash
# Download nvidia/GEAR-SONIC sonic_v1_1 → gear_sonic_deploy/policy/sonic_v1_1/
#
#   bash tools/env/download_gear_sonic_v1_1.sh
#   HF_MIRROR=https://hf-mirror.com bash tools/env/download_gear_sonic_v1_1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"

HF_MIRROR="${HF_MIRROR:-${HF_ENDPOINT:-https://hf-mirror.com}}"
HF_MIRROR="${HF_MIRROR%/}"
REPO="nvidia/GEAR-SONIC"
REV="${HF_REV:-main}"
DEST="${GEAR_SONIC_DEPLOY}/policy/sonic_v1_1"
mkdir -p "${DEST}"

echo "[sonic_v1_1] mirror=${HF_MIRROR} repo=${REPO} rev=${REV}"
echo "[sonic_v1_1] dest=${DEST}"

FILES=(
  model_encoder.onnx
  model_decoder.onnx
  observation_config.yaml
  config.yaml
  model_config.yaml
  last.pt
)

for name in "${FILES[@]}"; do
  out="${DEST}/${name}"
  if [[ -f "${out}" && "$(stat -c%s "${out}" 2>/dev/null || echo 0)" -gt 0 ]]; then
    echo "[sonic_v1_1] skip (exists) ${name} ($(stat -c%s "${out}") bytes)"
    continue
  fi
  url="${HF_MIRROR}/${REPO}/resolve/${REV}/sonic_v1_1/${name}"
  tmp="${out}.part"
  echo "[sonic_v1_1] GET ${url}"
  curl -fL --retry 5 --retry-delay 3 -o "${tmp}" "${url}"
  mv -f "${tmp}" "${out}"
  echo "[sonic_v1_1] -> ${out} ($(stat -c%s "${out}") bytes)"
done

echo "[sonic_v1_1] done. infer/viz: DEPLOY_POLICY_DIR=sonic_v1_1"
