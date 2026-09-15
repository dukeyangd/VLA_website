#!/usr/bin/env bash
# Re-cache 830mix pico+pure with Psi0 pretrained Qwen3-VL, then 2-ep train
# (same knobs as official-Instruct mix run, only VLM/cache dirname changes).
#
#   bash tools/data/run_830mix_pico_pure_psi0_vlm_cache_train.sh
#   SKIP_CACHE=1 bash ...   # cache already done
#   SKIP_TRAIN=1 bash ...
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/env/setup_env.sh"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
# paths.py remaps /mnt/data2/wpy/workspace → workspace_root(); keep them aligned
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}"
export PHI0_ROOT="${PHI0_ROOT:-${PHI0_WORKSPACE}/Phi_0_wpy}"

MIX_UNIFIED="${MIX_UNIFIED:-/mnt/data3/wpy/datasets/830/830mix_pico_pick_toy_pure_unified}"
WS_LINK="${WS_LINK:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/830mix_pico_pick_toy_pure_unified}"
PSI0_VLM="${PSI0_VLM:-/mnt/data2/wpy/workspace/Phi_0/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}"
CACHE_DIRNAME="${PHI0_VLM_FRAME_LATENTS_DIRNAME:-vlm_frame_latents_qwen3vl_dual_psi0}"
TASK_PROMPT="${TASK_PROMPT:-抓起黄色玩具放到篮子里。}"

NGPU_CACHE="${NGPU_CACHE:-8}"
CUDA_CACHE="${CUDA_CACHE:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DECODE_BATCH="${DECODE_BATCH:-64}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DECODE_WORKERS="${DECODE_WORKERS:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-cpu}"

EPOCHS="${EPOCHS:-2}"
NGPU="${NGPU:-6}"
NUM_ENVS="${NUM_ENVS:-32}"
HORIZON="${HORIZON:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

# Must set before Qwen3VLTower.from_pretrained (resolve_vlm_model_path reads this first).
export PHI0_VLM_MODEL_PATH="${PSI0_VLM}"
export PHI0_VLM_CKPT="${PSI0_VLM}"
export PHI0_VLM_FRAME_LATENTS_DIRNAME="${CACHE_DIRNAME}"
export PHI0_VLM_ATTN="${PHI0_VLM_ATTN:-flash_attention_2}"
export PHI0_ALLOW_SDPA_FALLBACK="${PHI0_ALLOW_SDPA_FALLBACK:-0}"
export PHI0_VIDEO_DECODE_DEVICE="${DECODE_DEVICE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"

# Optional local mirror so phi0_root()/checkpoints/... fallback works
mkdir -p "${PHI0_ROOT}/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/psi0"
ln -sfn "${PSI0_VLM}" \
  "${PHI0_ROOT}/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"

if [[ ! -d "${PSI0_VLM}" ]]; then
  echo "[psi0_mix] missing Psi0 VLM: ${PSI0_VLM}" >&2
  exit 1
fi
if [[ ! -f "${MIX_UNIFIED}/meta/info.json" ]]; then
  echo "[psi0_mix] missing mix unified: ${MIX_UNIFIED}" >&2
  exit 1
fi

CACHE_ROOT="${MIX_UNIFIED}/meta/${CACHE_DIRNAME}"
echo "[psi0_mix] stamp=${STAMP}"
echo "[psi0_mix] mix=${MIX_UNIFIED}"
echo "[psi0_mix] vlm=${PSI0_VLM}"
echo "[psi0_mix] cache_dirname=${CACHE_DIRNAME}"

_cache_complete() {
  "${PHI0_PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${PHI0_ROOT}/src")
from phi0.online.vlm_frame_latents import cache_dir_for_dataset, episode_done
import pyarrow.parquet as pq
root = Path("${MIX_UNIFIED}")
ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
cache = cache_dir_for_dataset(root)
meta = cache / "meta.json"
if not meta.is_file():
    raise SystemExit(1)
m = json.loads(meta.read_text())
assert "psi0" in str(m.get("vlm_path", "")).lower() or "pre.fast.1by1" in str(m.get("vlm_path", "")), m.get("vlm_path")
for r in ep.itertuples():
    if not episode_done(cache, int(r.episode_index), n_frames=int(r.length)):
        raise SystemExit(1)
print("ok", len(ep), "vlm", m.get("vlm_path"))
PY
}

if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
  if _cache_complete; then
    echo "[psi0_mix] psi0 VLM cache already complete — skip"
  else
    echo "[psi0_mix] caching with Psi0 VLM → ${CACHE_ROOT}"
    IFS=',' read -r -a _PHYS_GPUS <<< "${CUDA_CACHE}"
    if [[ "${#_PHYS_GPUS[@]}" -ne "${NGPU_CACHE}" ]]; then
      echo "[psi0_mix] CUDA_CACHE must list ${NGPU_CACHE} GPUs" >&2
      exit 1
    fi
    mkdir -p "${CACHE_ROOT}/logs"
    "${PHI0_PY}" "${PHI0_ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
      --dataset-root "${MIX_UNIFIED}" \
      --init-only \
      --max-seq-len 256 \
      --prompt "${TASK_PROMPT}" \
      --vlm-path "${PSI0_VLM}"
    PIDS=()
    for i in $(seq 0 $((NGPU_CACHE - 1))); do
      phys="${_PHYS_GPUS[$i]}"
      sl="${CACHE_ROOT}/logs/shard_${i}_of_${NGPU_CACHE}_phys${phys}.log"
      echo "[psi0_mix] shard ${i} -> GPU ${phys} log=${sl}"
      CUDA_VISIBLE_DEVICES="${phys}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        TORCH_NUM_THREADS="${TORCH_NUM_THREADS}" \
        PHI0_VLM_FRAME_LATENTS_DIRNAME="${CACHE_DIRNAME}" \
        "${PHI0_PY}" "${PHI0_ROOT}/tools/data/cache_dual_vlm_frame_latents.py" \
          --dataset-root "${MIX_UNIFIED}" \
          --shard "${i}/${NGPU_CACHE}" \
          --batch-size "${BATCH_SIZE}" \
          --decode-batch "${DECODE_BATCH}" \
          --video-backend "${VIDEO_BACKEND}" \
          --decode-workers "${DECODE_WORKERS}" \
          --decode-device "${DECODE_DEVICE}" \
          --max-seq-len 256 \
          --resume \
          --prompt "${TASK_PROMPT}" \
          --vlm-path "${PSI0_VLM}" \
          >"${sl}" 2>&1 &
      PIDS+=("$!")
    done
    fail=0
    for pid in "${PIDS[@]}"; do
      wait "${pid}" || fail=1
    done
    [[ "${fail}" -eq 0 ]] || exit 1
    _cache_complete
    du -sh "${CACHE_ROOT}" 2>/dev/null || true
  fi
fi

if ! _cache_complete; then
  echo "[psi0_mix] psi0 cache incomplete" >&2
  exit 1
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[psi0_mix] SKIP_TRAIN=1"
  exit 0
fi

mkdir -p "$(dirname "${WS_LINK}")"
ln -sfn "${MIX_UNIFIED}" "${WS_LINK}"

export STAMP EPOCHS NGPU NUM_ENVS HORIZON CUDA_VISIBLE_DEVICES
export REF_ROOT="${WS_LINK}"
export PHI0_VLM_FRAME_LATENTS_DIRNAME="${CACHE_DIRNAME}"
export PHI0_VLM_MODEL_PATH="${PSI0_VLM}"
# Psi0 HE hidden absmean ~6× Instruct; keep DiT first-step grads in band.
export PHI0_VLM_FRAME_CTX_SCALE="${PHI0_VLM_FRAME_CTX_SCALE:-0.2}"
export PHI0_TRAIN_HAND_OBS="${PHI0_TRAIN_HAND_OBS:-commanded}"
export PHI0_HAND_PROPRIO_LAG="${PHI0_HAND_PROPRIO_LAG:-1}"
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-/mnt/data3/wpy/830mix_pico_pick_toy_pure_psi0vlm_handcmd_lag1_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_e${EPOCHS}_${STAMP}}"
export LOG_FILE="${LOG_FILE:-${PHI0_ROOT}/logs/830mix_pico_pick_toy_pure_psi0vlm_handcmd_lag1_vlm_cache_h${HORIZON}_b${NUM_ENVS}_ddp${NGPU}_${STAMP}.log}"

echo "[psi0_mix] train epochs=${EPOCHS} ctx_scale=${PHI0_VLM_FRAME_CTX_SCALE} out=${PHI0_DISTILL_OUT}"
exec bash "${PHI0_ROOT}/tools/train/run_830_skill2_pico_pick_toy_vlm_cache_distill.sh"
