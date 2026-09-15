#!/usr/bin/env bash
# Pack skill_2_pico_new826 → unified (dex3 + sonic v1.1 token), allowlist, lang + dual VLM cache.
#
#   bash tools/data/run_830_skill2_pico_pack_and_cache.sh
#   SKIP_PACK=1 bash ...   # resume from existing unified
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

RAW_ROOT="${RAW_ROOT:-/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill_2_pico_new826}"
MANIFEST="${MANIFEST:-${RAW_ROOT}/skill_2.json}"
OUT_DIR="${OUT_DIR:-/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill_2_pico_new826_unified}"
TASK_PROMPT="${TASK_PROMPT:-走到桌面，抓取篮中玩具并搬运篮子到另一区域。}"
NVME_DIR="${NVME_DIR:-/mnt/data3/wpy/datasets/830/skill_2_pico_new826_unified}"
WS_LINK="${WS_LINK:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_2_pico_new826_unified}"
# ponytail: VLM cache decodes mp4 on CPU; EFS NFS → GPU idle. Stage on NVMe first.
CACHE_ROOT="${CACHE_ROOT:-${NVME_DIR}}"
LOG_DIR="${LOG_DIR:-/mnt/data2/wpy/workspace/logs/830_skill2_pico_pack_cache_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DECODE_BATCH="${DECODE_BATCH:-64}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DECODE_WORKERS="${DECODE_WORKERS:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-cpu}"
NGPU_CACHE="${NGPU_CACHE:-8}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"

mkdir -p "${LOG_DIR}"

export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_VIDEO_DECODE_DEVICE="${DECODE_DEVICE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"

if [[ "${SKIP_PACK:-0}" != "1" ]]; then
  echo "[830_skill2] pack raw=${RAW_ROOT} → ${OUT_DIR}"
  "${PHI0_PY}" "${PHI0_ROOT}/tools/data/pack_teleop_qpos_unified_lerobot.py" \
    --raw-root "${RAW_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --manifest "${MANIFEST}" \
    --task-prompt "${TASK_PROMPT}" \
    --hand-mode dex3 \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/pack.log"
fi

echo "[830_skill2] write allowlists"
"${PHI0_PY}" - <<PY
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("${OUT_DIR}")
eps = sorted(set(pq.read_table(
    root / "meta/episodes/chunk-000/file-000.parquet",
    columns=["episode_index"],
).column("episode_index").to_pylist()))
allow = {"episode_index": eps}
for name in ("vision_episode_allowlist.json", "all_episode_allowlist.json"):
    (root / "meta" / name).write_text(json.dumps(allow, indent=2) + "\n")
print(f"wrote {len(eps)} episodes → {root}/meta/")
PY

if [[ ! -f "${OUT_DIR}/meta/lang_latents_qwen3vl/meta.json" ]]; then
  echo "[830_skill2] lang latent cache"
  CUDA_VISIBLE_DEVICES=0 \
    "${PHI0_PY}" "${PHI0_ROOT}/tools/data/cache_boneseed_task_lang_latents.py" \
      --dataset-root "${OUT_DIR}" --batch-size 1 --device cuda \
      2>&1 | tee "${LOG_DIR}/lang_cache.log"
else
  echo "[830_skill2] lang cache exists, skip"
fi

echo "[830_skill2] stage unified → NVMe (videos+data+meta; resume partial cache)"
mkdir -p "${NVME_DIR}"
rsync -a --info=stats2 \
  --exclude='meta/lang_latents_qwen3vl/' \
  "${OUT_DIR}/" "${NVME_DIR}/"
mkdir -p "$(dirname "${WS_LINK}")"
ln -sfn "${NVME_DIR}" "${WS_LINK}"
echo "[830_skill2] CACHE_ROOT=${CACHE_ROOT} (local videos)"

echo "[830_skill2] dual VLM frame cache (8 GPU, batch=${BATCH_SIZE}, decode=${DECODE_DEVICE}, workers=${DECODE_WORKERS})"
DATASET_ROOT="${CACHE_ROOT}" PROMPT="${TASK_PROMPT}" \
  BATCH_SIZE="${BATCH_SIZE}" DECODE_BATCH="${DECODE_BATCH}" \
  VIDEO_BACKEND="${VIDEO_BACKEND}" DECODE_WORKERS="${DECODE_WORKERS}" \
  DECODE_DEVICE="${DECODE_DEVICE}" \
  NGPU="${NGPU_CACHE}" \
  PHI0_PY="${PHI0_PY}" \
  bash "${PHI0_ROOT}/tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh" \
  2>&1 | tee "${LOG_DIR}/vlm_cache.log"

echo "[830_skill2] cache done on ${NVME_DIR}"
echo "[830_skill2] nvme=${NVME_DIR} ws_link=${WS_LINK}"
du -sh "${OUT_DIR}/meta/vlm_frame_latents_qwen3vl_dual" "${NVME_DIR}/meta/vlm_frame_latents_qwen3vl_dual" 2>/dev/null || true
