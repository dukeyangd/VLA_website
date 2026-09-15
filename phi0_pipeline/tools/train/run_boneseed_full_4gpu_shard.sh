#!/usr/bin/env bash
# DEPRECATED / BUG — independent multi-process distill (NO gradient AllReduce).
# Canonical multi-GPU: bash tools/train/run_boneseed_full_distill_4gpu.sh (Fabric DDP).
# Opt-in only: ALLOW_SHARD_INDEPENDENT=1
# Each GPU gets 1/4 of RGs via SHARD_RANK/WORLD; each rank writes its own student
# ckpt under rank*/ — weights are NOT synced. See meta/NEWTON_train_deploy_pipeline.md §4.0
set -euo pipefail

if [[ "${ALLOW_SHARD_INDEPENDENT:-0}" != "1" ]]; then
  echo "[DEPRECATED] run_boneseed_full_4gpu_shard.sh is a bug path (no AllReduce)." >&2
  echo "  Use:  bash tools/train/run_boneseed_full_distill_4gpu.sh" >&2
  echo "  Or set ALLOW_SHARD_INDEPENDENT=1 to force this legacy launcher." >&2
  exit 2
fi

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PHI0_ROOT}/experiments/newton_boneseed_full_4gpu_${STAMP}}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_gmr_relroot_phi0}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_ENVS="${NUM_ENVS:-1024}"
HORIZON="${HORIZON:-8}"
NUM_CHUNKS="${NUM_CHUNKS:-50000}"
LR="${LR:-1e-3}"
RSI_START="${RSI_START:-random_episode}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
STUDENT_CKPT="${STUDENT_CKPT:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
# set NO_EPISODE_ALLOWLIST=1 for full corpus (default CLI otherwise demo5skill allowlist)
NO_EPISODE_ALLOWLIST="${NO_EPISODE_ALLOWLIST:-0}"
EPISODE_BLACKLIST="${EPISODE_BLACKLIST:-}"
ALLOW_ARGS=()
if [[ "${NO_EPISODE_ALLOWLIST}" == "1" ]]; then
  ALLOW_ARGS+=(--no_episode_allowlist)
fi
if [[ -n "${EPISODE_BLACKLIST:-}" && -f "${EPISODE_BLACKLIST}" ]]; then
  ALLOW_ARGS+=(--episode_blacklist "${EPISODE_BLACKLIST}")
fi

mkdir -p "${OUT_ROOT}" /mnt/data2/wpy/workspace/logs
echo "${OUT_ROOT}" >"${PHI0_ROOT}/experiments/_last_boneseed_4gpu_out.txt"
printf '%s\n' "OUT=${OUT_ROOT}" "REF=${REF_ROOT}" "NGPU=${NUM_GPUS}" "B=${NUM_ENVS}" \
  "CHUNKS=${NUM_CHUNKS}" "RSI=${RSI_START}" "MODE=shard_independent" \
  > /mnt/data2/wpy/workspace/logs/newton_boneseed_full_4gpu_latest.txt

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_GPUS}" ]]; then
  NUM_GPUS="${#GPUS[@]}"
fi

echo "[boneseed4] out=${OUT_ROOT}"
echo "[boneseed4] py=${PYTHON_BIN} gpus=${GPU_LIST} B/gpu=${NUM_ENVS} chunks=${NUM_CHUNKS} lr=${LR} rsi=${RSI_START}"

wait_rank_ready() {
  local rank="$1"
  local log="${OUT_ROOT}/rank${rank}/run.log"
  local t0
  t0=$(date +%s)
  echo "[boneseed4] waiting rank${rank} env-ready..."
  while true; do
    if [[ -f "${log}" ]] && rg -q 'Completed setting up the environment|DISTILL_RECIPE|distill stride1|loading rg=' "${log}"; then
      echo "[boneseed4] rank${rank} ready"
      return 0
    fi
    if [[ -f "${log}" ]] && rg -q 'Error executing job|AF_UNIX path too long|NameError:|ChildFailedError' "${log}"; then
      echo "[boneseed4] rank${rank} FAILED — see ${log}" >&2
      return 1
    fi
    if ! kill -0 "${PIDS[$rank]}" 2>/dev/null; then
      echo "[boneseed4] rank${rank} process died early" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    if (( $(date +%s) - t0 > READY_TIMEOUT_S )); then
      echo "[boneseed4] rank${rank} ready timeout" >&2
      return 1
    fi
    sleep 5
  done
}

cd "${GR00T_ROOT}"
PIDS=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
  gpu="${GPUS[$rank]}"
  RANK_OUT="${OUT_ROOT}/rank${rank}"
  mkdir -p "${RANK_OUT}"
  LOG="${RANK_OUT}/run.log"

  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PHI0_ISAAC_BACKEND=newton \
    PHI0_ONNX_ENCODE_GPU=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    EXP_PATH="${EXP_PATH:-${ISAACLAB_PATH}/apps}" \
    ISAACLAB_PATH="${ISAACLAB_PATH}" \
    PYTHONPATH="${PHI0_ROOT}/src:${PHI0_ROOT}/subpackages:${GR00T_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -u "${PHI0_ROOT}/tools/train/newton_boneseed_distill.py" \
      --num_envs "${NUM_ENVS}" \
      --num_chunks "${NUM_CHUNKS}" \
      --horizon "${HORIZON}" \
      --lr "${LR}" \
      --rsi_start "${RSI_START}" \
      --ref_root "${REF_ROOT}" \
      --rg_scan \
      --use_lang_latent_cache \
      --shard_rank "${rank}" \
      --shard_world "${NUM_GPUS}" \
      --ckpt_every "${CKPT_EVERY}" \
      ${STUDENT_CKPT:+--student_ckpt "${STUDENT_CKPT}"} \
      "${ALLOW_ARGS[@]}" \
      --out_dir "${RANK_OUT}" \
      --headless \
      --visualizer none \
    >"${LOG}" 2>&1 &
  PIDS+=($!)
  echo "[boneseed4] rank=${rank} gpu=${gpu} pid=${PIDS[-1]} log=${LOG}"

  if ! wait_rank_ready "${rank}"; then
    echo "[boneseed4] abort" >&2
    printf '%s\n' "${PIDS[@]}" >"${OUT_ROOT}/pids.txt"
    exit 1
  fi
done

printf '%s\n' "${PIDS[@]}" >"${OUT_ROOT}/pids.txt"
echo "[boneseed4] all launched pids=${PIDS[*]}"
echo "[boneseed4] monitor: tail -f ${OUT_ROOT}/rank0/run.log"
