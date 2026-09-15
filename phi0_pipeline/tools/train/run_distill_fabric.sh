#!/usr/bin/env bash
# Unified Fabric distill entry — ProtoMotions-style --simulator switch.
#   SIMULATOR=newton (default) → Lab3 kit-less + joint_pd expert
#   SIMULATOR=physx            → gear sonic native (phi-0-wbc) + direct_latent
#
# Conda / IsaacLab / PHI0_ISAAC_BACKEND are bound BEFORE Python starts.
# Extra args after -- are forwarded to the backend entry
# (newton_boneseed_distill_fabric.py or train_online_distill_fabric.py hydra).
#
# Examples:
#   SIMULATOR=newton NGPU=1 NUM_ENVS=4 bash tools/train/run_distill_fabric.sh -- --num_envs 4 --epochs 1 ...
#   SIMULATOR=physx  NGPU=1 bash tools/train/run_distill_fabric.sh -- ++num_envs=8 ...
set -euo pipefail

PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PHI0_ROOT}/tools/lib/sonic_isaac_common.sh"
sonic_init_paths

# Parse leading --simulator / SIMULATOR= before apply.
SIMULATOR="${SIMULATOR:-newton}"
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --simulator)
      SIMULATOR="$2"
      shift 2
      ;;
    --simulator=*)
      SIMULATOR="${1#--simulator=}"
      shift
      ;;
    --)
      shift
      FORWARD+=("$@")
      break
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

sonic_apply_simulator "${SIMULATOR}"
sonic_resolve_python
sonic_export_isaac_env
sonic_pythonpath
export PHI0_FABRIC=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29570}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

NGPU="${NGPU:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PHI0_DISTILL_OUT="${PHI0_DISTILL_OUT:-${PHI0_ROOT}/experiments/distill_${PHI0_SIMULATOR}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PHI0_DISTILL_OUT}"
cd "${GR00T_ROOT}"

echo "[distill_fabric] simulator=${PHI0_SIMULATOR} ngpu=${NGPU} py=${PYTHON_BIN} out=${PHI0_DISTILL_OUT}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PHI0_ROOT}/tools/train/distill_fabric_entry.py" \
  --simulator "${PHI0_SIMULATOR}" \
  "${FORWARD[@]}"
