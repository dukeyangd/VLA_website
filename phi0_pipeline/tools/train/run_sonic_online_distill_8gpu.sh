#!/usr/bin/env bash
# DEPRECATED / BUG — 8× independent Isaac processes (NO gradient AllReduce).
# Canonical multi-GPU: bash tools/train/run_boneseed_full_distill_4gpu.sh (Fabric DDP)
# or run_sonic_online_distill_fabric.sh with NGPU=….
# Opt-in only: ALLOW_SHARD_INDEPENDENT=1
# Serialize Kit startup: wait until previous rank finished env setup before
# launching the next (Isaac GPU foundation cannot init concurrently).

set -euo pipefail

if [[ "${ALLOW_SHARD_INDEPENDENT:-0}" != "1" ]]; then
  echo "[DEPRECATED] run_sonic_online_distill_8gpu.sh is a bug path (no AllReduce)." >&2
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

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PHI0_ROOT}/experiments/sonic_online_distill_protomotions_full8gpu_${STAMP}}"
REF_ROOT="${REF_ROOT:-/mnt/efs_1/Phi0_Dataset/Phi0-MixCorpus/datasets/bone_seed/train/smpl_qpos_phi0_protomotions_full}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENVS="${NUM_ENVS:-8}"
HORIZON="${HORIZON:-8}"
EPOCHS="${EPOCHS:-1}"
NUM_CHUNKS="${NUM_CHUNKS:-}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
# Hold AppLauncher filelock after Kit create (seconds).
LOCK_HOLD_S="${PHI0_ISAAC_LOCK_HOLD_S:-120}"
# Max wait for previous rank's env-ready marker.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"

export PHI0_ISAAC_LOCK_HOLD_S="${LOCK_HOLD_S}"

mkdir -p "${OUT_ROOT}"
echo "${OUT_ROOT}" >"${PHI0_ROOT}/experiments/_last_full8gpu_out.txt"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_GPUS}" ]]; then
  NUM_GPUS="${#GPUS[@]}"
fi

if [[ -z "${NUM_CHUNKS}" ]]; then
  T="$(sonic_count_ref_frames "${REF_ROOT}" 0)"
  STARTS=$(( T > HORIZON ? T - HORIZON + 1 : 1 ))
  STARTS_LOCAL=$(( (STARTS + NUM_GPUS - 1) / NUM_GPUS ))
  NUM_CHUNKS=$(( EPOCHS * (STARTS_LOCAL + NUM_ENVS - 1) / NUM_ENVS ))
fi

echo "[8gpu] out=${OUT_ROOT}"
echo "[8gpu] conda=${CONDA_ENV} python=${PYTHON_BIN}"
echo "[8gpu] mode=serialized-per-gpu gpus=${GPU_LIST} n=${NUM_GPUS} envs=${NUM_ENVS}"
echo "[8gpu] chunks/rank=${NUM_CHUNKS} lock_hold=${LOCK_HOLD_S}s ready_timeout=${READY_TIMEOUT_S}s"

wait_rank_ready() {
  local rank="$1"
  local log="${OUT_ROOT}/rank${rank}/run.log"
  local t0
  t0=$(date +%s)
  echo "[8gpu] waiting rank${rank} env-ready (Completed setting up / policy loaded)..."
  while true; do
    if [[ -f "${log}" ]] && rg -q 'Completed setting up the environment|Successfully loaded policy state dict|distill stride1' "${log}"; then
      echo "[8gpu] rank${rank} ready"
      return 0
    fi
    # GPU foundation warnings can appear transiently during Kit/Vulkan bring-up.
    # Only fail fast on hard launcher/config errors here; otherwise keep waiting
    # until the rank either becomes ready, dies, or times out.
    if [[ -f "${log}" ]] && rg -q 'Error executing job|AF_UNIX path too long|Traceback \(most recent call last\)' "${log}"; then
      echo "[8gpu] rank${rank} FAILED — see ${log}" >&2
      return 1
    fi
    if ! kill -0 "${PIDS[$rank]}" 2>/dev/null; then
      echo "[8gpu] rank${rank} process died early" >&2
      return 1
    fi
    if (( $(date +%s) - t0 > READY_TIMEOUT_S )); then
      echo "[8gpu] rank${rank} ready timeout ${READY_TIMEOUT_S}s" >&2
      return 1
    fi
    sleep 5
  done
}

PIDS=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
  gpu="${GPUS[$rank]}"
  RANK_OUT="${OUT_ROOT}/rank${rank}"
  mkdir -p "${RANK_OUT}"
  LOG="${RANK_OUT}/run.log"
  RANK_TMP="/tmp/d8r${rank}"
  rm -rf "${RANK_TMP}"
  mkdir -p "${RANK_TMP}/home" "${RANK_TMP}/xdg-config" "${RANK_TMP}/xdg-data" "${RANK_TMP}/xdg-state" "${RANK_TMP}/xdg-cache"

  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PHI0_RENDER_GPU_PHYSICAL_INDEX="${gpu}" \
    PHI0_DISTILL_OUT="${RANK_OUT}" \
    REF_ROOT="${REF_ROOT}" \
    HORIZON="${HORIZON}" \
    NUM_ENVS="${NUM_ENVS}" \
    MAX_REF_FRAMES=0 \
    LAZY_REF=1 \
    SHARD_RANK="${rank}" \
    SHARD_WORLD="${NUM_GPUS}" \
    CONDA_ENV="${CONDA_ENV}" \
    NUM_CHUNKS="${NUM_CHUNKS}" \
    PHI0_ISAAC_LOCK_HOLD_S="${LOCK_HOLD_S}" \
    TMPDIR="${RANK_TMP}" \
    TEMP="${RANK_TMP}" \
    TMP="${RANK_TMP}" \
    HOME="${RANK_TMP}/home" \
    XDG_CONFIG_HOME="${RANK_TMP}/xdg-config" \
    XDG_DATA_HOME="${RANK_TMP}/xdg-data" \
    XDG_STATE_HOME="${RANK_TMP}/xdg-state" \
    XDG_CACHE_HOME="${RANK_TMP}/xdg-cache" \
    CUDA_CACHE_PATH="${RANK_TMP}/cuda-cache" \
    __GL_SHADER_DISK_CACHE_PATH="${RANK_TMP}/gl-shader-cache" \
    bash -c 'unset EPOCHS; exec bash "'"${PHI0_ROOT}"'/tools/train/run_sonic_online_distill_isaac.sh"' \
    >"${LOG}" 2>&1 &
  PIDS+=($!)
  echo "[8gpu] rank=${rank} gpu=${gpu} pid=${PIDS[-1]} log=${LOG}"

  # Do not start next GPU until this one finished Kit/env create.
  if ! wait_rank_ready "${rank}"; then
    echo "[8gpu] abort — fix rank${rank} before continuing" >&2
    printf '%s\n' "${PIDS[@]}" >"${OUT_ROOT}/pids.txt"
    exit 1
  fi
done

printf '%s\n' "${PIDS[@]}" >"${OUT_ROOT}/pids.txt"
echo "[8gpu] all launched pids=${PIDS[*]}"
echo "[8gpu] monitor: tail -f ${OUT_ROOT}/rank0/run.log"
