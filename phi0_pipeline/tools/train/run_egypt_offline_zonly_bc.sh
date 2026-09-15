#!/usr/bin/env bash
# Egypt offline z-only BC: inject onnx_g1 z*, W_Z=1, no Isaac.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/tools/env/setup_env.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/egypt_smplsem_clip}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/egypt_offline_zonly_${STAMP}}"
Z_STAR_NPY="${Z_STAR_NPY:-${OUT_DIR}/z_star_onnx_g1.npy}"
Z_STAR_STATS="${Z_STAR_STATS:-${OUT_DIR}/z_star_stats.json}"
ACTION_STATS="${ACTION_STATS:-${REF_ROOT}/meta/stats.json}"
ALLOWLIST="${ALLOWLIST:-${OUT_DIR}/allowlist_ep0.json}"
HORIZON="${HORIZON:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-1e-4}"
CKPT_EVERY="${CKPT_EVERY:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES PHI0_ONNX_ENCODE_GPU="${PHI0_ONNX_ENCODE_GPU:-1}"

mkdir -p "${OUT_DIR}"
if [[ ! -f "${ALLOWLIST}" ]]; then
  echo '{"episode_index":[0]}' >"${ALLOWLIST}"
fi

if [[ ! -f "${Z_STAR_NPY}" ]]; then
  echo "[egypt-zonly] precompute z* → ${OUT_DIR}"
  "${PHI0_PY}" "${ROOT}/tools/data/precompute_egypt_zstar_onnx_g1.py" \
    --ref-root "${REF_ROOT}" --max-frames 278 --out-dir "${OUT_DIR}"
fi

cat >"${OUT_DIR}/RESOLVED_TRAIN_SETTINGS.yaml" <<EOF
no_sim: 1
loss: z_only
w_z: 1
w_hand: 0
w_q_head: 0
w_smpl: 0
z_teacher: onnx_g1_tape
anchor: fk_root_tape
ref_root: ${REF_ROOT}
horizon: ${HORIZON}
batch_size: ${BATCH_SIZE}
epochs: ${EPOCHS}
max_steps: ${MAX_STEPS}
lr: ${LR}
z_star_npy: ${Z_STAR_NPY}
z_star_stats: ${Z_STAR_STATS}
action_stats: ${ACTION_STATS}
note: ablation vs LOCKED online onnx_g1+qpos_student; CONTROL=student for infer
EOF

echo "[egypt-zonly] out=${OUT_DIR}"
exec "${PHI0_PY}" - <<PY
from pathlib import Path
from phi0.online.offline_chunk_bc import run_offline_chunk_bc

out = Path("${OUT_DIR}")
run_offline_chunk_bc(
    ref_root="${REF_ROOT}",
    episode_allowlist="${ALLOWLIST}",
    action_stats_path="${ACTION_STATS}",
    z_star_stats_path="${Z_STAR_STATS}",
    out_dir=out,
    horizon=int("${HORIZON}"),
    batch_size=int("${BATCH_SIZE}"),
    epochs=int("${EPOCHS}"),
    lr=float("${LR}"),
    ckpt_every=int("${CKPT_EVERY}"),
    state_dropout=0.0,
    w_z=1.0,
    w_hand=0.0,
    w_q_head=0.0,
    w_smpl=0.0,
    max_steps=int("${MAX_STEPS}"),
    z_ref_override="${Z_STAR_NPY}",
)
print("[egypt-zonly] done", out, flush=True)
PY
