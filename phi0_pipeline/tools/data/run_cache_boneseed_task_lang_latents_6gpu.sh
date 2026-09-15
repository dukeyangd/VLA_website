#!/usr/bin/env bash
# Parallel Qwen3-VL task-lang latent cache on 6 GPUs.
# Safe to run while smpl_gmr_relroot data shards are still writing (reads tasks.parquet only).
set -euo pipefail
ROOT=/mnt/data2/wpy/workspace
PHI0=/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python
DS=${DS:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0}
LOG_DIR=${LOG_DIR:-$DS/meta/lang_latents_qwen3vl/logs}
# Comma-separated physical GPU ids (default skips busy gpu0).
GPUS=${GPUS:-1,2,3,4,5,6}
BATCH=${BATCH:-64}
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
N_GPUS=${#GPU_ARR[@]}
export PYTHONPATH="$ROOT/phi-0-wbc-newton/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"
echo "[launch] ds=$DS gpus=$GPUS (n=$N_GPUS) batch=$BATCH $(date -Is)" | tee "$LOG_DIR/launch.log"

echo "[launch] init-only sidecar…" | tee -a "$LOG_DIR/launch.log"
"$PHI0" "$ROOT/phi-0-wbc-newton/tools/data/cache_boneseed_task_lang_latents.py" \
  --dataset-root "$DS" --init-only --batch-size "$BATCH" \
  2>&1 | tee -a "$LOG_DIR/init.log"

pids=()
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_ARR[$i]}"
  log="$LOG_DIR/shard_${i}_of_${N_GPUS}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PHI0" \
    "$ROOT/phi-0-wbc-newton/tools/data/cache_boneseed_task_lang_latents.py" \
    --dataset-root "$DS" --shard "${i}/${N_GPUS}" --resume --batch-size "$BATCH" \
    >"$log" 2>&1 &
  pids+=($!)
  echo "shard $i/$N_GPUS gpu=$gpu pid=${pids[-1]} log=$log" | tee -a "$LOG_DIR/launch.log"
done
echo "${pids[*]}" >"$LOG_DIR/pids.txt"
echo "[launch] pids=${pids[*]}" | tee -a "$LOG_DIR/launch.log"
