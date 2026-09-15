#!/usr/bin/env bash
# Studio 03：勾选 raw sessions（manifest 仅含 valid）→ pack → VLM cache → distill。
# 与官方流水线一致：复用 tools/data/run_830_skill2_pico_pack_and_cache.sh +
# tools/train/run_online_vlm_mix_distill.sh。
#
# 打包目录由 Studio 注入：Phi_0_model_zoo/Phi_0_train_data/<skill>_<stamp>/
# 训练结束后：归档到 /mnt/efs_1/gzy/workspace/Phi_0_train_data/ 并校验，通过后删除本地 pack。
# 原始 830demo session 只读，不会被改写。
#
# 必填环境变量：
#   RAW_ROOT MANIFEST OUT_DIR NVME_DIR WS_LINK TASK_PROMPT
#   （以及 distill 侧的 REF_ROOT / EPOCHS / NGPU / …，由 Studio 注入）
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${RAW_ROOT:?RAW_ROOT required}"
: "${MANIFEST:?MANIFEST required}"
: "${OUT_DIR:?OUT_DIR required}"
: "${NVME_DIR:?NVME_DIR required}"
: "${WS_LINK:?WS_LINK required}"
: "${TASK_PROMPT:?TASK_PROMPT required}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[studio_pack] missing MANIFEST=${MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "[studio_pack] missing RAW_ROOT=${RAW_ROOT}" >&2
  exit 1
fi

PACK_SCRIPT="${PHI0_ROOT}/tools/data/run_830_skill2_pico_pack_and_cache.sh"
DISTILL_SCRIPT="${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
if [[ ! -f "${PACK_SCRIPT}" ]]; then
  echo "[studio_pack] missing ${PACK_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${DISTILL_SCRIPT}" ]]; then
  echo "[studio_pack] missing ${DISTILL_SCRIPT}" >&2
  exit 1
fi

mkdir -p -- "${OUT_DIR}" "${NVME_DIR}" "$(dirname "${WS_LINK}")"

# Pack jobs always pack fresh (SKIP_PACK=0).
export RAW_ROOT MANIFEST OUT_DIR NVME_DIR WS_LINK TASK_PROMPT
export SKIP_PACK="${SKIP_PACK:-0}"
echo "[studio_pack] persistent pack (kept after distill)"
echo "[studio_pack] RAW_ROOT=${RAW_ROOT}"
echo "[studio_pack] MANIFEST=${MANIFEST}"
echo "[studio_pack] OUT_DIR=${OUT_DIR} → NVME=${NVME_DIR} link=${WS_LINK}"
echo "[studio_pack] SKIP_PACK=${SKIP_PACK}"

bash "${PACK_SCRIPT}"

# Distill always reads the workspace link created by pack.
export REF_ROOT="${WS_LINK}"
export ACTION_STATS_PATH="${REF_ROOT}/meta/stats.json"
# Pack 只打入 valid；allowlist = 包内全部 episode（即 valid-only）。
if [[ -z "${EPISODE_ALLOWLIST:-}" ]]; then
  if [[ "${VISION_ONLY:-0}" == "1" || "${VISION_ONLY:-}" == "true" ]]; then
    export EPISODE_ALLOWLIST="${REF_ROOT}/meta/vision_episode_allowlist.json"
  else
    export EPISODE_ALLOWLIST="${REF_ROOT}/meta/all_episode_allowlist.json"
  fi
fi

echo "[studio_pack] distill REF_ROOT=${REF_ROOT} allowlist=${EPISODE_ALLOWLIST}"
bash "${DISTILL_SCRIPT}"
echo "[studio_pack] distill done — pack kept (Studio does not delete)"
