#!/usr/bin/env bash
# Pack 830demo/skill_2_pico_pick_the_toy → unified (dex3 + sonic v1.1), allowlist, dual VLM cache.
#
#   bash tools/data/run_830_skill2_pico_pick_toy_pack_and_cache.sh
#   SKIP_PACK=1 bash ...   # resume from existing unified
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

RAW_ROOT="${RAW_ROOT:-/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy}"
MANIFEST="${MANIFEST:-${RAW_ROOT}/skill_2.json}"
OUT_DIR="${OUT_DIR:-/mnt/data3/wpy/datasets/830/830demo_skill2_pico_pick_toy_unified}"
TASK_PROMPT="${TASK_PROMPT:-抓起黄色玩具放到篮子里。}"
NVME_DIR="${NVME_DIR:-${OUT_DIR}}"
WS_LINK="${WS_LINK:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill2_pico_pick_toy_unified}"
CACHE_ROOT="${CACHE_ROOT:-${NVME_DIR}}"
LOG_DIR="${LOG_DIR:-${PHI0_ROOT}/logs/830_skill2_pico_pick_toy_pack_cache_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DECODE_BATCH="${DECODE_BATCH:-64}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DECODE_WORKERS="${DECODE_WORKERS:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-cpu}"
NGPU_CACHE="${NGPU_CACHE:-4}"
CUDA_CACHE="${CUDA_CACHE:-4,5,6,7}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"

mkdir -p "${LOG_DIR}"

export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_VIDEO_DECODE_DEVICE="${DECODE_DEVICE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"

if [[ "${SKIP_PACK:-0}" != "1" ]]; then
  echo "[830_pico_pick] pack raw=${RAW_ROOT} → ${OUT_DIR}"
  "${PHI0_PY}" "${PHI0_ROOT}/tools/data/pack_teleop_qpos_unified_lerobot.py" \
    --raw-root "${RAW_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --manifest "${MANIFEST}" \
    --task-prompt "${TASK_PROMPT}" \
    --hand-mode dex3 \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/pack.log"
fi

echo "[830_pico_pick] write allowlists"
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
info = json.loads((root / "meta/info.json").read_text())
print(f"allowlist n={len(eps)} info_eps={info.get('total_episodes')} frames={info.get('total_frames')}")
PY

mkdir -p "$(dirname "${WS_LINK}")"
ln -sfn "${NVME_DIR}" "${WS_LINK}"
echo "[830_pico_pick] ws_link=${WS_LINK}"

echo "[830_pico_pick] dual VLM frame cache phys GPUs=${CUDA_CACHE} n=${NGPU_CACHE} batch=${BATCH_SIZE}"
IFS=',' read -r -a _PHYS_GPUS <<< "${CUDA_CACHE}"
if [[ "${#_PHYS_GPUS[@]}" -ne "${NGPU_CACHE}" ]]; then
  echo "[830_pico_pick] CUDA_CACHE must list ${NGPU_CACHE} GPUs (got ${CUDA_CACHE})" >&2
  exit 1
fi
VLM_LOG_DIR="${CACHE_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/logs"
mkdir -p "${VLM_LOG_DIR}"
"${PHI0_PY}" "${PHI0_ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
  --dataset-root "${CACHE_ROOT}" --init-only --max-seq-len 256 --prompt "${TASK_PROMPT}"
PIDS=()
for i in $(seq 0 $((NGPU_CACHE - 1))); do
  phys="${_PHYS_GPUS[$i]}"
  sl="${VLM_LOG_DIR}/shard_${i}_of_${NGPU_CACHE}_phys${phys}.log"
  echo "[830_pico_pick] shard ${i} -> GPU ${phys} log=${sl}"
  CUDA_VISIBLE_DEVICES="${phys}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    TORCH_NUM_THREADS="${TORCH_NUM_THREADS}" \
    "${PHI0_PY}" "${PHI0_ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
      --dataset-root "${CACHE_ROOT}" \
      --shard "${i}/${NGPU_CACHE}" \
      --batch-size "${BATCH_SIZE}" \
      --decode-batch "${DECODE_BATCH}" \
      --video-backend "${VIDEO_BACKEND}" \
      --decode-workers "${DECODE_WORKERS}" \
      --decode-device "${DECODE_DEVICE}" \
      --max-seq-len 256 \
      --resume \
      --prompt "${TASK_PROMPT}" \
      >"${sl}" 2>&1 &
  PIDS+=("$!")
done
fail=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || fail=1
done
[[ "${fail}" -eq 0 ]] || exit 1

echo "[830_pico_pick] cache done on ${NVME_DIR}"
du -sh "${OUT_DIR}/meta/vlm_frame_latents_qwen3vl_dual" 2>/dev/null || true
