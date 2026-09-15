#!/usr/bin/env bash
# qpos_student / student on Newton (Isaac Lab 3) → Newton-GL mp4 of live sim state.
# Student proprio: live body IL→MuJoCo remap (train tape order) + live Revo2.
# NEVER uses Phi-0-wpy / MuJoCo FK for model infer viz.
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/newton_hand_env.sh"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths
export CONDA_ENV="Phi-0-wbc-newton-wpy"
unset PYTHON_BIN CONDA_PREFIX_ENV
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath

case "${PYTHON_BIN}" in
  *Phi-0-wpy*) echo "[newton_viz] refusing Phi-0-wpy: ${PYTHON_BIN}" >&2; exit 1 ;;
esac

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data2/wpy/workspace/phi-0-wbc/experiments/egypt_overfit_onnx_gpu_b1024_20260725_110734/phi0_student_step002000.pt}"
CONTROL="${CONTROL:-qpos_student}"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/egypt_smplsem_clip}"
REF_START="${REF_START:-0}"
MAX_REF_FRAMES="${MAX_REF_FRAMES:-278}"
TAG="${TAG:-newton_isaac_${CONTROL}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PHI0_ROOT}/experiments/${TAG}}"
NUM_STEPS="${NUM_STEPS:-278}"
# Infer schedule (not distill train): H=32 open-loop chunk = prior distill + VLA.
# PHI0_CHUNK_EXEC=first → only token0 then replan (optional A/B).
# Force open_loop unless caller sets INFER_CHUNK_EXEC (parent shells often leave
# PHI0_CHUNK_EXEC=first exported from older jobs).
HORIZON="${HORIZON:-32}"
export PHI0_CHUNK_EXEC="${INFER_CHUNK_EXEC:-open_loop}"
FRAME_SKIP="${FRAME_SKIP:-2}"
USE_LANG_LATENT_CACHE="${USE_LANG_LATENT_CACHE:-0}"
# Default: dataset ego+wrist strip above Newton-GL at current tape frame (MuJoCo GT_PANEL=top).
GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-top}"
GT_PANEL_H="${GT_PANEL_H:-160}"
GT_PANEL_LABELS="${GT_PANEL_LABELS:-0}"
export GT_PANEL_LAYOUT GT_PANEL_H GT_PANEL_LABELS
# NEWTON_HAND=rubber (default) | dex3 | revo2
export NEWTON_HAND="${NEWTON_HAND:-rubber}"
apply_newton_hand_env
mkdir -p "${OUT}" "${TMPDIR:-/mnt/data3/tmp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TAG
export PYTHONPATH="${PHI0_ROOT}/src:${PHI0_ROOT}/subpackages:${GR00T_ROOT}:${PYTHONPATH:-}"
# RSI write: Newton foot mesh sits lower than MuJoCo FK — default +3cm (isaac_sim.py).
export PHI0_RSI_ROOT_Z_LIFT="${PHI0_RSI_ROOT_Z_LIFT:-0.03}"

LOG="${OUT}/newton_isaac_viz.log"
echo "[newton_viz] py=${PYTHON_BIN}"
echo "[newton_viz] control=${CONTROL} ckpt=${STUDENT_CKPT}"
echo "[newton_viz] ref=${REF_ROOT} ref_start=${REF_START} max_ref=${MAX_REF_FRAMES}"
echo "[newton_viz] out=${OUT} steps=${NUM_STEPS} H=${HORIZON} lang_cache=${USE_LANG_LATENT_CACHE} use_vlm=${USE_VLM:-0}"
echo "[newton_viz] gt_panel=${GT_PANEL_LAYOUT} h=${GT_PANEL_H} labels=${GT_PANEL_LABELS}"
echo "[newton_viz] rsi_root_z_lift=${PHI0_RSI_ROOT_Z_LIFT} mujoco_contacts=${PHI0_NEWTON_USE_MUJOCO_CONTACTS:-0}"
echo "[newton_viz] newton_hand=${NEWTON_HAND} mode=${PHI0_HAND_MODE} revo2_usd=${PHI0_NEWTON_REVO2}"
echo "[newton_viz] log=${LOG}"

LANG_ARGS=()
if [[ "${USE_LANG_LATENT_CACHE}" == "1" || "${USE_LANG_LATENT_CACHE}" == "true" ]]; then
  LANG_ARGS+=(--use_lang_latent_cache)
fi
if [[ "${USE_VLM:-0}" == "1" || "${USE_VLM:-}" == "true" ]]; then
  LANG_ARGS+=(--use_vlm)
  export PHI0_TRAIN_MODE="${PHI0_TRAIN_MODE:-online_vlm}"
  export USE_VLM=1
fi

cd "${GR00T_ROOT}"
# Same launch style as newton_egypt_qpos_gold: pin Newton conda python (not isaaclab.sh / Phi-0-wpy).
nohup env \
  NEWTON_HAND="${NEWTON_HAND}" \
  PHI0_HAND_MODE="${PHI0_HAND_MODE}" \
  PHI0_NEWTON_REVO2="${PHI0_NEWTON_REVO2}" \
  "${PYTHON_BIN}" "${PHI0_ROOT}/tools/eval/newton_qpos_student_isaac_viz.py" \
  --newton_hand "${NEWTON_HAND}" \
  --student_ckpt "${STUDENT_CKPT}" \
  --control "${CONTROL}" \
  --ref_root "${REF_ROOT}" \
  --ref_start "${REF_START}" \
  --max_ref_frames "${MAX_REF_FRAMES}" \
  --out_dir "${OUT}" \
  --num_steps "${NUM_STEPS}" \
  --horizon "${HORIZON}" \
  --frame_skip "${FRAME_SKIP}" \
  --gt_panel_layout "${GT_PANEL_LAYOUT}" \
  --gt_panel_h "${GT_PANEL_H}" \
  "${LANG_ARGS[@]}" \
  --headless \
  --visualizer none \
  >"${LOG}" 2>&1 &
PID=$!
echo "[newton_viz] pid=${PID}"
sleep 12
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "[newton_viz] died early; tail log:" >&2
  tail -n 80 "${LOG}" >&2
  exit 1
fi
tail -n 40 "${LOG}" || true
echo "[newton_viz] running; mp4 will be: ${OUT}/infer_${CONTROL}_newton_gl.mp4"
