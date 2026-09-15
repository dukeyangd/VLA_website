#!/usr/bin/env bash
# 830 skill_2_pico_new826_unified — ChunkStudent closed-loop MuJoCo (sonic v1.1 + Dex3).
#
#   EP=0 bash tools/eval/run_830_skill2_pico_student_cl_mujoco_viz.sh
#   STUDENT_CKPT=.../phi0_student_last.pt EP=5 bash tools/eval/run_830_skill2_pico_student_cl_mujoco_viz.sh
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export STUDENT_CKPT="${STUDENT_CKPT:-/mnt/data3/wpy/830_skill2_pico_nosim_h32_b32_ddp8_e10_20260827_081246/phi0_student_last.pt}"
export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/skill_2_pico_new826_unified}"
export EP="${EP:-0}"
export HORIZON="${HORIZON:-32}"
# Raw teleop motion_token in unified is sonic v1.1 — do not use low_latency decoder.
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
# Legacy unified on disk may still store teleop actuator-order gripper14; set 1 after repack.
export PHI0_DEX3_HAND_POLICY_ORDER="${PHI0_DEX3_HAND_POLICY_ORDER:-0}"
export TAG="${TAG:-830_skill2_pico_student_cl_ep${EP}_$(date +%Y%m%d_%H%M%S)}"

# Map unified ep → raw session dir for measured-hand GT panel (optional).
if [[ -z "${VALID_HAND_ROOT:-}" ]]; then
  VALID_HAND_ROOT="$("${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}" - <<PY
import json
from pathlib import Path
meta = json.loads(Path("${REF_ROOT}/meta.json").read_text())
raw = Path(meta.get("raw_root", "/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill_2_pico_new826"))
for row in meta.get("sources", []):
    if int(row["out_episode_index"]) == int("${EP}"):
        print(raw / row["session"])
        break
PY
)"
  export VALID_HAND_ROOT
fi

exec bash "${PHI0_ROOT}/tools/eval/run_830_walk_student_cl_mujoco_viz.sh"
