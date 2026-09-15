#!/usr/bin/env bash
# Pack gear_sonic_deploy policy/planner tarballs for WEIGHTS_URL_BASE mirrors.
# Usage:
#   source tools/env/setup_env.sh
#   bash tools/env/pack_deploy_weights.sh --out /tmp/phi0_deploy_weights
#   bash tools/env/pack_deploy_weights.sh --from /path/to/seed/gear_sonic_deploy --out /tmp/out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT}/tools/env/setup_env.sh"

FROM="${GEAR_SONIC_DEPLOY}"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    -h|--help)
      sed -n '1,8p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${OUT}" ]] || { echo "ERROR: --out DIR required" >&2; exit 1; }
FROM="$(cd "${FROM}" && pwd)"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"

need() {
  local p="$1"
  [[ -e "${FROM}/${p}" ]] || { echo "ERROR: missing ${FROM}/${p}" >&2; exit 1; }
}

need policy
need planner

echo "[pack] from=${FROM}"
echo "[pack] out=${OUT}"

# policy/release
if [[ -d "${FROM}/policy/release" ]]; then
  tar -C "${FROM}/policy" -czf "${OUT}/policy_release.tar.gz" release
  echo "[pack] wrote ${OUT}/policy_release.tar.gz"
else
  echo "WARN: no policy/release" >&2
fi

# policy/low_latency
if [[ -d "${FROM}/policy/low_latency" ]]; then
  tar -C "${FROM}/policy" -czf "${OUT}/policy_low_latency.tar.gz" low_latency
  echo "[pack] wrote ${OUT}/policy_low_latency.tar.gz"
else
  echo "WARN: no policy/low_latency" >&2
fi

# planner (prefer target_vel/V2 tree; else whole planner/)
if [[ -d "${FROM}/planner/target_vel" ]]; then
  tar -C "${FROM}/planner" -czf "${OUT}/planner_sonic.tar.gz" target_vel
elif [[ -d "${FROM}/planner" ]]; then
  tar -C "${FROM}" -czf "${OUT}/planner_sonic.tar.gz" planner
else
  echo "ERROR: no planner" >&2
  exit 1
fi
echo "[pack] wrote ${OUT}/planner_sonic.tar.gz"

ls -lh "${OUT}"/*.tar.gz
echo "[pack] serve ${OUT} as WEIGHTS_URL_BASE, then:"
echo "  WEIGHTS_URL_BASE=... bash tools/env/setup_subpackages_env.sh --download-weights"
