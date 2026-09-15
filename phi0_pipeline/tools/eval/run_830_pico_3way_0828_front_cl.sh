#!/usr/bin/env bash
# Today 3 pico-pick ablations × 0828 OOD vision × front cam → hstack compare mp4.
#   measured / handcmd / nohandobs  (hand proprio match each train recipe)
set -euo pipefail
PHI0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
# Ablation default: no RTC (override with USE_RTC=1)
export USE_RTC="${USE_RTC:-0}"
if [[ -z "${BASE_OUT:-}" ]]; then
  if [[ "${USE_RTC}" == "0" || "${USE_RTC}" == "false" ]]; then
    BASE_OUT="${PHI0_ROOT}/experiments/830_pico_3way_0828_front_nortc_${STAMP}"
  else
    BASE_OUT="${PHI0_ROOT}/experiments/830_pico_3way_0828_front_${STAMP}"
  fi
fi
mkdir -p "${BASE_OUT}/logs"

export REF_ROOT="${REF_ROOT:-/mnt/data2/wpy/workspace/local_nvme/datasets/830/0828_skill2_unified}"
export EP="${EP:-0}"
export HORIZON="${HORIZON:-32}"
export DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-sonic_v1_1}"
export PHI0_HAND_MODE=dex3
export PHI0_NEWTON_REVO2=0
export PHI0_DEX3_HAND_POLICY_ORDER="${PHI0_DEX3_HAND_POLICY_ORDER:-1}"
export PHI0_DEX3_SCALE_TO_UNITREE="${PHI0_DEX3_SCALE_TO_UNITREE:-0}"
export PHI0_USE_VLM_FRAME_LATENT_CACHE=0
export PHI0_CL_VLM_SOURCE=dataset_video
export PHI0_VLM_ENCODE_MIN_BATCH="${PHI0_VLM_ENCODE_MIN_BATCH:-2}"
export USE_VLM=1
export PHI0_CL_PROPRIO_BODY=live
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
export MAX_FRAMES="${MAX_FRAMES:-1500}"
export GT_ZMQ_RECORD_STALL_S="${GT_ZMQ_RECORD_STALL_S:-30}"
export PROMPT="${PROMPT:-抓起黄色玩具放到篮子里。}"
export TASK_PROMPT="${TASK_PROMPT:-${PROMPT}}"
# Front view: +X-facing robot, offset 90 (default 135 is 3/4)
export SIM_RECORD_CAM_AZIMUTH_OFFSET="${SIM_RECORD_CAM_AZIMUTH_OFFSET:-90}"

CKPT_MEASURED="${CKPT_MEASURED:-/mnt/data3/wpy/830demo_skill2_pico_pick_toy_vlm_cache_h32_b32_ddp6_e10_20260903_205811/phi0_student_last.pt}"
CKPT_HANDCMD="${CKPT_HANDCMD:-/mnt/data3/wpy/830demo_skill2_pico_pick_toy_handcmd_vlm_cache_h32_b32_ddp6_e10_20260904_010502/phi0_student_last.pt}"
CKPT_NOHANDOBS="${CKPT_NOHANDOBS:-/mnt/data3/wpy/830demo_skill2_pico_pick_toy_nohandobs_vlm_cache_h32_b32_ddp6_e10_20260904_031525/phi0_student_last.pt}"

echo "[3way] USE_RTC=${USE_RTC} BASE_OUT=${BASE_OUT}"

wait_mp4() {
  local mp4="$1" log="$2" timeout_s="${3:-900}"
  local t0=$SECONDS
  while (( SECONDS - t0 < timeout_s )); do
    if [[ -f "${log}" ]] && rg -q 'remuxed to H.264|video: |done work_dir=' "${log}" 2>/dev/null; then
      sleep 2
      if ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${mp4}" >/dev/null 2>&1; then
        return 0
      fi
    fi
    if [[ -f "${log}" ]] && rg -q 'ERROR:|Traceback' "${log}" 2>/dev/null; then
      echo "[3way] error in ${log}" >&2
      tail -n 40 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[3way] timeout waiting ${mp4}" >&2
  return 1
}

run_one() {
  local name="$1" ckpt="$2" zero_hand="$3" hand_obs="$4"
  local tag="830_pico_${name}_0828_front_ep${EP}_${STAMP}"
  local out="${BASE_OUT}/${name}"
  local mp4="${out}/student_sonic_dex3_cl.mp4"
  local clog="${out}/logs/mujoco_cl.log"
  echo "[3way] === ${name} zero_hand=${zero_hand} hand_obs=${hand_obs} az_off=${SIM_RECORD_CAM_AZIMUTH_OFFSET} ==="
  test -f "${ckpt}"
  # walk_cl backgrounds eval; capture launch log then wait for remux
  STUDENT_CKPT="${ckpt}" \
  PHI0_ZERO_PROPRIO_HAND="${zero_hand}" \
  PHI0_CL_HAND_OBS="${hand_obs}" \
  TAG="${tag}" \
  OUT="${out}" \
  bash "${PHI0_ROOT}/tools/eval/run_830_walk_student_cl_mujoco_viz.sh" \
    | tee "${BASE_OUT}/logs/${name}_launch.log"
  wait_mp4 "${mp4}" "${clog}" 1200
  echo "[3way] ok ${name} -> ${mp4} dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${mp4}")"
  printf '%s\n' "${mp4}" >"${BASE_OUT}/logs/${name}.mp4path"
}

echo "[3way] BASE_OUT=${BASE_OUT}"
run_one measured "${CKPT_MEASURED}" 0 g1_debug
run_one handcmd "${CKPT_HANDCMD}" 0 commanded
run_one nohandobs "${CKPT_NOHANDOBS}" 1 commanded

MP4_M=$(cat "${BASE_OUT}/logs/measured.mp4path")
MP4_H=$(cat "${BASE_OUT}/logs/handcmd.mp4path")
MP4_N=$(cat "${BASE_OUT}/logs/nohandobs.mp4path")
COMPARE="${BASE_OUT}/pico_3way_0828_front_compare.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -i "${MP4_M}" -i "${MP4_H}" -i "${MP4_N}" \
  -filter_complex "
    [0:v]scale=-2:480,drawtext=text='measured (obs=measured)':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55[v0];
    [1:v]scale=-2:480,drawtext=text='handcmd (obs=commanded)':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55[v1];
    [2:v]scale=-2:480,drawtext=text='nohandobs (obs=zero)':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55[v2];
    [v0][v1][v2]hstack=inputs=3[out]
  " -map '[out]' -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart \
  "${COMPARE}"

echo "[3way] compare=${COMPARE}"
echo "${COMPARE}" >"${BASE_OUT}/COMPARE.txt"
ls -la "${COMPARE}" "${MP4_M}" "${MP4_H}" "${MP4_N}"
