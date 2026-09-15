#!/usr/bin/env bash
# Studio 03 mix compose：每技能独立 prompt + raw sessions → pack/cache → merge → distill。
# Env from Studio: MIX_SLOTS_JSON MERGED_OUT WS_LINK (+ distill EPOCHS/NGPU/…)
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MIX_SLOTS_JSON:?MIX_SLOTS_JSON required}"
: "${MERGED_OUT:?MERGED_OUT required}"
: "${WS_LINK:?WS_LINK required}"

PACK_SCRIPT="${PHI0_ROOT}/tools/data/run_830_skill2_pico_pack_and_cache.sh"
MERGE_PY="${PHI0_ROOT}/tools/data/merge_teleop_unified_vlm_cache.py"
DISTILL_SCRIPT="${PHI0_ROOT}/tools/train/run_online_vlm_mix_distill.sh"
PHI0_PY="${PHI0_PY:-/mnt/data3/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python}"
if [[ ! -x "${PHI0_PY}" ]]; then
  if [[ -x /mnt/data2/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python ]]; then
    PHI0_PY=/mnt/data2/wpy/conda-envs/Phi-0-wbc-newton-wpy/bin/python
  fi
fi

for f in "${PACK_SCRIPT}" "${MERGE_PY}" "${DISTILL_SCRIPT}"; do
  [[ -f "${f}" ]] || { echo "[studio_mix] missing ${f}" >&2; exit 1; }
done

WORKDIR="$(mktemp -d /tmp/studio_mix_slots.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT
export MIX_SLOTS_JSON MERGED_OUT WS_LINK WORKDIR PACK_SCRIPT

"${PHI0_PY}" - <<'PY'
import json, os
from pathlib import Path
slots = json.loads(os.environ["MIX_SLOTS_JSON"])
if len(slots) < 2:
    raise SystemExit(f"need >=2 slots, got {len(slots)}")
wd = Path(os.environ["WORKDIR"])
lines = ["#!/usr/bin/env bash", "set -euo pipefail", "MERGE_ROOTS=()", "MERGE_TAGS=()", "FALLBACK_PROMPT='task'"]
for i, s in enumerate(slots):
    skill = str(s.get("skill_id") or f"skill{i}")
    raw = str(s.get("raw_root") or "")
    man = str(s.get("manifest_path") or "")
    out = str(s.get("out_dir") or "")
    nvme = str(s.get("nvme_dir") or "")
    link = str(s.get("ws_link") or "")
    for req, val in (("raw_root", raw), ("manifest", man), ("out_dir", out), ("nvme_dir", nvme)):
        if not val:
            raise SystemExit(f"slot {skill} missing {req}")
    lines += [
        f'echo "[studio_mix] slot {i}: skill={skill}"',
        f'if [[ "${{SKIP_PACK:-0}}" != "1" ]]; then',
        f'  RAW_ROOT={json.dumps(raw)} MANIFEST={json.dumps(man)} OUT_DIR={json.dumps(out)} \\',
        f'  NVME_DIR={json.dumps(nvme)} WS_LINK={json.dumps(link or nvme)} TASK_PROMPT={json.dumps(str(s.get("prompt") or "task"))} \\',
        f'  SKIP_PACK=0 bash "${{PACK_SCRIPT}}"',
        f'else echo "[studio_mix] SKIP_PACK=1 for {skill}"; fi',
        f'ROOT_FOR_MERGE={json.dumps(nvme)}',
        f'if [[ ! -d "${{ROOT_FOR_MERGE}}/meta" ]]; then ROOT_FOR_MERGE={json.dumps(out)}; fi',
        f'if [[ ! -d "${{ROOT_FOR_MERGE}}/meta" ]]; then echo "[studio_mix] missing unified meta" >&2; exit 1; fi',
        f'MERGE_ROOTS+=("${{ROOT_FOR_MERGE}}")',
        f'MERGE_TAGS+=({json.dumps(skill)})',
    ]
    if i == 0:
        lines.append(f"FALLBACK_PROMPT={json.dumps(str(s.get('prompt') or 'task'))}")
(wd / "pack_slots.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(slots))
PY

echo "[studio_mix] compose ${WORKDIR} → ${MERGED_OUT}"
# shellcheck disable=SC1091
source "${WORKDIR}/pack_slots.sh"

echo "[studio_mix] merge ${#MERGE_ROOTS[@]} roots"
MERGE_ARGS=()
for i in "${!MERGE_ROOTS[@]}"; do
  MERGE_ARGS+=(--root "${MERGE_ROOTS[$i]}" --tag "${MERGE_TAGS[$i]}")
done
"${PHI0_PY}" "${MERGE_PY}" \
  "${MERGE_ARGS[@]}" \
  --out-dir "${MERGED_OUT}" \
  --task-prompt "${FALLBACK_PROMPT}" \
  --allow-no-video \
  --overwrite

"${PHI0_PY}" - <<PY
import json
from pathlib import Path
import pyarrow.parquet as pq
root = Path("${MERGED_OUT}")
ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet", columns=["episode_index"])
eps = sorted(set(ep.column("episode_index").to_pylist()))
allow = {"episode_index": eps}
meta = root / "meta"
for name in ("vision_episode_allowlist.json", "all_episode_allowlist.json"):
    p = meta / name
    if not p.is_file():
        p.write_text(json.dumps(allow, indent=2) + "\n")
print(f"[studio_mix] allowlists n={len(eps)}")
PY

mkdir -p -- "$(dirname "${WS_LINK}")"
ln -sfn "${MERGED_OUT}" "${WS_LINK}"
export REF_ROOT="${WS_LINK}"
export ACTION_STATS_PATH="${REF_ROOT}/meta/stats.json"
if [[ -z "${EPISODE_ALLOWLIST:-}" ]]; then
  export EPISODE_ALLOWLIST="${REF_ROOT}/meta/all_episode_allowlist.json"
fi
echo "[studio_mix] distill REF_ROOT=${REF_ROOT}"
bash "${DISTILL_SCRIPT}"
echo "[studio_mix] done — pack kept"
