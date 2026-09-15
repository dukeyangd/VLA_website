#!/usr/bin/env bash
# Build 830mix skill1+2+3 + demo5skill, encode missing VLM caches, then 3-ep mix train.
#
#   bash tools/data/run_830mix_skill123_demo5_pipeline.sh
#   SKIP_CACHE=1 SKIP_MERGE=1 bash ...   # train only
#   SKIP_TRAIN=1 bash ...                # cache+merge only
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"

SKILL1="${SKILL1:-/mnt/data3/wpy/datasets/830/830demo_skill1_walk_unified}"
# skill2 = pico + pure merge (not pico-only unified)
SKILL2="${SKILL2:-/mnt/data3/wpy/datasets/830/830mix_pico_pick_toy_pure_unified}"
SKILL3="${SKILL3:-/mnt/data3/wpy/datasets/830/830demo_skill3_pico_place_basket_unified}"
# BoneSEED demo5 GT: sonic_v1_1 tokens from **qpos** reencode (g1 mode), NOT SMPL
# encode and NOT llstyle explicit-obs pack.
DEMO5="${DEMO5:-/mnt/data2/wpy/workspace/820demo/820demo_demo5skill_sonic_v1_1_unified}"
OUT_MIX="${OUT_MIX:-/mnt/data3/wpy/datasets/830/830mix_skill123_demo5_unified}"
WS_LINK="${WS_LINK:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830mix_skill123_demo5_unified}"

CACHE_DIRNAME="${PHI0_VLM_FRAME_LATENTS_DIRNAME:-vlm_frame_latents_qwen3vl_dual}"
PROMPT_S1="${PROMPT_S1:-机器人朝黑箱子走过去。}"
PROMPT_S2="${PROMPT_S2:-抓起黄色玩具放到篮子里。}"
PROMPT_S3="${PROMPT_S3:-把篮子放到指定位置。}"

NGPU_CACHE="${NGPU_CACHE:-8}"
CUDA_CACHE="${CUDA_CACHE:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DECODE_BATCH="${DECODE_BATCH:-64}"

EPOCHS="${EPOCHS:-3}"
PHI0_P_VISION="${PHI0_P_VISION:-0.9}"
NGPU="${NGPU:-8}"
NUM_ENVS="${NUM_ENVS:-32}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PIPE_LOG="${PIPE_LOG:-${PHI0_ROOT}/logs/830mix_skill123_demo5_pipeline_${STAMP}.log}"
mkdir -p "$(dirname "${PIPE_LOG}")"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[pipe] stamp=${STAMP} log=${PIPE_LOG}"
echo "[pipe] s1=${SKILL1}"
echo "[pipe] s2=${SKILL2}"
echo "[pipe] s3=${SKILL3}"
echo "[pipe] demo5=${DEMO5}"
echo "[pipe] out=${OUT_MIX}"

_cache_complete() {
  local root="$1"
  "${PHI0_PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${PHI0_ROOT}/src")
from phi0.online.vlm_frame_latents import cache_dir_for_dataset, episode_done
import pyarrow.parquet as pq
root = Path("${root}")
ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
# vision-only check: eps with dual video
from pathlib import Path as P
vk0 = "observation.images.ego_view"
vision_eps = []
for r in ep.itertuples():
    ei = int(r.episode_index)
    ego = root / "videos/chunk-000" / vk0 / f"episode_{ei:06d}.mp4"
    if ego.is_file():
        vision_eps.append((ei, int(r.length)))
if not vision_eps:
    raise SystemExit("no vision eps")
cache = cache_dir_for_dataset(root)
if not (cache / "meta.json").is_file():
    raise SystemExit(1)
for ei, n in vision_eps:
    if not episode_done(cache, ei, n_frames=n):
        raise SystemExit(1)
print("ok", len(vision_eps), "vision eps cached")
PY
}

_cache_shards_running() {
  local root="$1"
  pgrep -af "cache_dual_vlm_frame_latents.py --dataset-root ${root}" >/dev/null 2>&1
}

_wait_cache_or_encode() {
  local root="$1"
  local prompt="$2"
  local tag="$3"
  local waited=0
  if _cache_complete "${root}"; then
    echo "[pipe] ${tag}: VLM cache complete — reuse"
    return 0
  fi
  if _cache_shards_running "${root}"; then
    echo "[pipe] ${tag}: cache job already running — wait"
    while ! _cache_complete "${root}"; do
      if ! _cache_shards_running "${root}"; then
        echo "[pipe] ${tag}: shard processes exited before complete" >&2
        break
      fi
      sleep 30
      waited=$((waited + 30))
      n_ep=$(ls "${root}/meta/${CACHE_DIRNAME}/ep" 2>/dev/null | wc -l)
      echo "[pipe] ${tag}: waiting ${waited}s cached_eps≈${n_ep}"
    done
    if _cache_complete "${root}"; then
      echo "[pipe] ${tag}: VLM cache complete — reuse"
      return 0
    fi
  fi
  echo "[pipe] ${tag}: encoding VLM cache → ${root}/meta/${CACHE_DIRNAME}"
  env -u LOG_DIR \
    DATASET_ROOT="${root}" \
    PROMPT="${prompt}" \
    NGPU="${NGPU_CACHE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_CACHE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    DECODE_BATCH="${DECODE_BATCH}" \
    bash "${PHI0_ROOT}/tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh"
  _cache_complete "${root}"
}

if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
  _wait_cache_or_encode "${SKILL1}" "${PROMPT_S1}" "skill1"
  _wait_cache_or_encode "${SKILL2}" "${PROMPT_S2}" "skill2"
  _wait_cache_or_encode "${SKILL3}" "${PROMPT_S3}" "skill3"
else
  echo "[pipe] SKIP_CACHE=1"
fi

if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
  echo "[pipe] merging → ${OUT_MIX}"
  # skill1 first so modality.json comes from teleop pack
  "${PHI0_PY}" "${PHI0_ROOT}/tools/data/merge_teleop_unified_vlm_cache.py" \
    --root "${SKILL1}" --tag skill1_walk \
    --root "${SKILL2}" --tag skill2_pico \
    --root "${SKILL3}" --tag skill3_place \
    --root "${DEMO5}" --tag demo5skill \
    --out-dir "${OUT_MIX}" \
    --allow-no-video \
    --overwrite \
    --task-prompt "${PROMPT_S2}"
  mkdir -p "$(dirname "${WS_LINK}")"
  ln -sfn "${OUT_MIX}" "${WS_LINK}"
  echo "[pipe] WS_LINK → ${WS_LINK}"
else
  echo "[pipe] SKIP_MERGE=1"
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[pipe] train epochs=${EPOCHS} p_vision=${PHI0_P_VISION}"
  # Unset inherited OUT/LOG from prior skill runs (pollutes PHI0_DISTILL_OUT).
  env -u LOG_DIR -u PHI0_DISTILL_OUT -u LOG_FILE \
    REF_ROOT="${OUT_MIX}" \
    EPOCHS="${EPOCHS}" \
    PHI0_P_VISION="${PHI0_P_VISION}" \
    NGPU="${NGPU}" \
    NUM_ENVS="${NUM_ENVS}" \
    HORIZON="${HORIZON}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    STAMP="${STAMP}" \
    bash "${PHI0_ROOT}/tools/train/run_830mix_skill123_demo5_vlm_cache_distill.sh"
else
  echo "[pipe] SKIP_TRAIN=1 — mix ready at ${OUT_MIX}"
fi

echo "[pipe] done $(date -Iseconds)"
