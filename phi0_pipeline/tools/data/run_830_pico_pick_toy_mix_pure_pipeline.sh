#!/usr/bin/env bash
# Mix skill_2_pico_pick_the_toy + skill_2_pico_pick_the_toy_pure:
#   - reuse existing pico VLM cache
#   - pack+cache pure if missing
#   - merge (hardlink videos/cache) → train EPOCHS (default 2)
#
#   bash tools/data/run_830_pico_pick_toy_mix_pure_pipeline.sh
#   SKIP_PURE_PACK=1 bash ...   # pure already packed+cached
#   SKIP_TRAIN=1 bash ...       # stop after merge
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"

PICO_RAW="${PICO_RAW:-/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy}"
PURE_RAW="${PURE_RAW:-/mnt/data2/wpy/workspace/830demo/skill_2_pico_pick_the_toy_pure}"
PICO_UNIFIED="${PICO_UNIFIED:-/mnt/data3/wpy/datasets/830/830demo_skill2_pico_pick_toy_unified}"
PURE_UNIFIED="${PURE_UNIFIED:-/mnt/data3/wpy/datasets/830/830demo_skill2_pico_pick_toy_pure_unified}"
MIX_UNIFIED="${MIX_UNIFIED:-/mnt/data3/wpy/datasets/830/830mix_pico_pick_toy_pure_unified}"
WS_LINK="${WS_LINK:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830mix_pico_pick_toy_pure_unified}"
TASK_PROMPT="${TASK_PROMPT:-抓起黄色玩具放到篮子里。}"
EPOCHS="${EPOCHS:-2}"
NGPU="${NGPU:-6}"
NUM_ENVS="${NUM_ENVS:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
CUDA_CACHE="${CUDA_CACHE:-4,5,6,7}"
NGPU_CACHE="${NGPU_CACHE:-4}"

echo "[mix_pico_pure] stamp=${STAMP}"
echo "[mix_pico_pure] pico=${PICO_UNIFIED}"
echo "[mix_pico_pure] pure=${PURE_UNIFIED}"
echo "[mix_pico_pure] mix=${MIX_UNIFIED}"

if [[ ! -f "${PICO_UNIFIED}/meta/vlm_frame_latents_qwen3vl_dual/meta.json" ]]; then
  echo "[mix_pico_pure] missing pico VLM cache under ${PICO_UNIFIED}" >&2
  exit 1
fi

_pure_cache_complete() {
  "${PHI0_PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${PHI0_ROOT}/src")
from phi0.online.vlm_frame_latents import cache_dir_for_dataset, episode_done
import pyarrow.parquet as pq
root = Path("${PURE_UNIFIED}")
if not (root / "meta/info.json").is_file():
    raise SystemExit(1)
ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
cache = cache_dir_for_dataset(root)
for r in ep.itertuples():
    if not episode_done(cache, int(r.episode_index), n_frames=int(r.length)):
        raise SystemExit(1)
print("ok", len(ep))
PY
}

if [[ "${SKIP_PURE_PACK:-0}" != "1" ]]; then
  if _pure_cache_complete; then
    echo "[mix_pico_pure] pure VLM cache complete — skip pack/cache"
  else
    echo "[mix_pico_pure] pack+cache pure → ${PURE_UNIFIED}"
    RAW_ROOT="${PURE_RAW}" \
      MANIFEST="${PURE_RAW}/skill_2.json" \
      OUT_DIR="${PURE_UNIFIED}" \
      NVME_DIR="${PURE_UNIFIED}" \
      WS_LINK="/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill2_pico_pick_toy_pure_unified" \
      CACHE_ROOT="${PURE_UNIFIED}" \
      CUDA_CACHE="${CUDA_CACHE}" \
      NGPU_CACHE="${NGPU_CACHE}" \
      TASK_PROMPT="${TASK_PROMPT}" \
      bash "${PHI0_ROOT}/tools/data/run_830_skill2_pico_pick_toy_pack_and_cache.sh"
  fi
fi

if ! _pure_cache_complete; then
  echo "[mix_pico_pure] pure VLM cache incomplete: ${PURE_UNIFIED}" >&2
  exit 1
fi

echo "[mix_pico_pure] merge → ${MIX_UNIFIED}"
"${PHI0_PY}" "${PHI0_ROOT}/tools/data/merge_teleop_unified_vlm_cache.py" \
  --root "${PICO_UNIFIED}" --tag pico \
  --root "${PURE_UNIFIED}" --tag pure \
  --out-dir "${MIX_UNIFIED}" \
  --task-prompt "${TASK_PROMPT}" \
  --overwrite

mkdir -p "$(dirname "${WS_LINK}")"
ln -sfn "${MIX_UNIFIED}" "${WS_LINK}"
echo "[mix_pico_pure] ws_link=${WS_LINK}"
du -sh "${MIX_UNIFIED}" "${MIX_UNIFIED}/meta/vlm_frame_latents_qwen3vl_dual" 2>/dev/null || true

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[mix_pico_pure] SKIP_TRAIN=1 — done after merge"
  exit 0
fi

export STAMP EPOCHS NGPU NUM_ENVS CUDA_VISIBLE_DEVICES
export REF_ROOT="${WS_LINK}"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/830mix_pico_pick_toy_pure_handcmd_lag1_vlm_cache_h32_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/830mix_pico_pick_toy_pure_handcmd_lag1_vlm_cache_h32_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"
# train uses lag1 handcmd (same as pico pick_toy default recipe)
export PHI0_TRAIN_HAND_OBS="${PHI0_TRAIN_HAND_OBS:-commanded}"
export PHI0_HAND_PROPRIO_LAG="${PHI0_HAND_PROPRIO_LAG:-1}"

echo "[mix_pico_pure] train epochs=${EPOCHS} out=${PHI0_DISTILL_OUT}"
exec bash "${PHI0_ROOT}/tools/train/run_830_skill2_pico_pick_toy_vlm_cache_distill.sh"
