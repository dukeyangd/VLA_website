#!/usr/bin/env bash
# 820 mix interleave distill: P_PROPRIO vision/proprio coin + stand RSI frame0.
# Defaults: P_PROPRIO=0.1, STAND_EP=0 (idle statue). Override allowlist as needed.
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PHI0_ROOT
export PHI0_SUBPACKAGES="${PHI0_ROOT}/subpackages"
export GR00T_ROOT="${PHI0_SUBPACKAGES}"
export SIMULATOR="${SIMULATOR:-newton}"
export PHI0_INTERLEAVE_P_PROPRIO="${PHI0_INTERLEAVE_P_PROPRIO:-0.1}"
export PHI0_STAND_EPISODE_INDEX="${PHI0_STAND_EPISODE_INDEX:-0}"
export PHI0_STAND_RSI_PROB="${PHI0_STAND_RSI_PROB:-0.5}"
export PHI0_RSI_EP0_PROB="${PHI0_RSI_EP0_PROB:-1.0}"
export RSI_START="${RSI_START:-random}"
# Prefer caller export; default online (disk only when explicitly set upstream).
export TEACHER_Z_SOURCE="${TEACHER_Z_SOURCE:-online}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-release}"
exec bash "${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
