#!/usr/bin/env bash
# Studio 01 collect deploy wrapper (real robot).
# - Skips deploy.sh /dev/tty "Proceed? [Y/n]"
# - Waits for studio_cmd mailbox (y/deploy) before launching g1_deploy
# - Feeds g1_deploy stdin via FIFO; maps stand(]) / stream(Enter) from studio_cmd
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/user/YZY/DataCollection/GR00T-WholeBodyControl/gear_sonic_deploy}"
STUDIO_CMD_DIR="${STUDIO_CMD_DIR:?STUDIO_CMD_DIR required}"
POLICY_VARIANT="${POLICY_VARIANT:-v1_1}"
INPUT_TYPE="${INPUT_TYPE:-zmq}"
ZMQ_HOST="${ZMQ_HOST:-localhost}"
ZMQ_PORT="${ZMQ_PORT:-5556}"
ZMQ_TOPIC="${ZMQ_TOPIC:-pose}"
INTERFACE_MODE="${INTERFACE_MODE:-real}"
OUTPUT_TYPE="${OUTPUT_TYPE:-all}"
PLANNER="${PLANNER:-planner/target_vel/V2/planner_sonic.onnx}"
MOTION_DATA="${MOTION_DATA:-reference/example/}"

CMD_FILE="${STUDIO_CMD_DIR}/studio_cmd"
PHASE_FILE="${STUDIO_CMD_DIR}/studio_phase"
FIFO="${STUDIO_CMD_DIR}/deploy_stdin.fifo"
LOG_DIR="${STUDIO_CMD_DIR}/logs"
mkdir -p "${LOG_DIR}"
: > "${CMD_FILE}"
echo "waiting_deploy" > "${PHASE_FILE}"

cd "${DEPLOY_ROOT}"

# Resolve policy variant → checkpoint / obs-config (mirrors deploy.sh)
case "${POLICY_VARIANT}" in
  release) CHECKPOINT="policy/release/model"; OBS_CONFIG="policy/release/observation_config.yaml" ;;
  low_latency) CHECKPOINT="policy/low_latency/model"; OBS_CONFIG="policy/low_latency/observation_config.yaml" ;;
  v1_1) CHECKPOINT="policy/v1_1/model"; OBS_CONFIG="policy/v1_1/observation_config.yaml" ;;
  *) echo "[studio_deploy] unknown POLICY_VARIANT=${POLICY_VARIANT}" >&2; exit 1 ;;
esac
CHECKPOINT_DECODER="${CHECKPOINT}_decoder.onnx"
CHECKPOINT_ENCODER="${CHECKPOINT}_encoder.onnx"

# shellcheck source=/dev/null
# setup_env.sh expands optional vars like CMAKE_PREFIX_PATH / LD_LIBRARY_PATH;
# with `set -u` that aborts the whole wrapper before waiting for web "y".
set +u
source scripts/setup_env.sh 2>/dev/null || true
set -u

# Minimal interface resolve (real → 192.168.123.x)
resolve_target() {
  local mode="$1"
  if [[ "${mode}" == "sim" ]]; then
    echo "lo"
    return
  fi
  if [[ "${mode}" == "real" ]]; then
    local found
    found="$(ip -4 addr show 2>/dev/null | awk '
      /^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 }
      /inet / {
        split($2, a, "/")
        if (a[1] ~ /^192\.168\.123\./) { print iface; exit }
      }
    ')"
    if [[ -n "${found}" ]]; then
      echo "${found}"
      return
    fi
  fi
  # explicit iface or IP fallthrough
  if [[ "${mode}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    ip -4 addr show 2>/dev/null | awk -v tip="$mode" '
      /^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 }
      /inet / {
        split($2, a, "/")
        if (a[1] == tip) { print iface; exit }
      }
    '
    return
  fi
  echo "${mode}"
}

TARGET="$(resolve_target "${INTERFACE_MODE}")"
if [[ -z "${TARGET}" ]]; then
  echo "[studio_deploy] FATAL: cannot resolve interface for INTERFACE_MODE=${INTERFACE_MODE}" >&2
  exit 1
fi

echo "[studio_deploy] policy=${POLICY_VARIANT} target=${TARGET} input=${INPUT_TYPE} zmq=${ZMQ_HOST}:${ZMQ_PORT}/${ZMQ_TOPIC}"
echo "[studio_deploy] waiting for web command: y / deploy  (studio_cmd=${CMD_FILE})"
echo "waiting_deploy" > "${PHASE_FILE}"

studio_wait_cmd() {
  local want_re="$1"
  local label="$2"
  echo "[studio_deploy] WAIT ${label} (match: ${want_re})"
  while true; do
    if [[ -f "${CMD_FILE}" ]]; then
      local raw
      raw="$(tr -d '\r' < "${CMD_FILE}" | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      if [[ -n "${raw}" ]]; then
        : > "${CMD_FILE}"
        if [[ "${raw}" == "stop" || "${raw}" == "quit" ]]; then
          echo "[studio_deploy] stop requested"
          exit 0
        fi
        if [[ "${raw}" =~ ${want_re} ]]; then
          echo "[studio_deploy] got cmd='${raw}' for ${label}"
          echo "${raw}"
          return 0
        fi
        if [[ "${raw}" == key:* ]]; then
          printf '%s' "${raw#key:}" >&3 || true
        fi
        echo "[studio_deploy] ignored cmd='${raw}' (want ${label})"
      fi
    fi
    sleep 0.25
  done
}

# Gate 1: confirm launch (web y)
studio_wait_cmd '^(y|Y|deploy)$' "deploy/y" >/dev/null
echo "deploy_starting" > "${PHASE_FILE}"

rm -f "${FIFO}"
mkfifo "${FIFO}"

cleanup() {
  echo "stopped" > "${PHASE_FILE}" || true
  exec 3>&- 2>/dev/null || true
  rm -f "${FIFO}" || true
}
trap cleanup EXIT

# Start g1_deploy with FIFO as stdin
set +e
just run g1_deploy_onnx_ref "${TARGET}" "${CHECKPOINT_DECODER}" "${MOTION_DATA}" \
  --obs-config "${OBS_CONFIG}" \
  --encoder-file "${CHECKPOINT_ENCODER}" \
  --planner-file "${PLANNER}" \
  --input-type "${INPUT_TYPE}" \
  --output-type "${OUTPUT_TYPE}" \
  --zmq-host "${ZMQ_HOST}" \
  --zmq-port "${ZMQ_PORT}" \
  --zmq-topic "${ZMQ_TOPIC}" \
  < "${FIFO}" &
DEPLOY_PID=$!
set -e

# Keep FIFO open for writes
exec 3>"${FIFO}"

echo "[studio_deploy] g1_deploy pid=${DEPLOY_PID}"
echo "deploy_running" > "${PHASE_FILE}"

# Poll logs for Init Done then wait for ] / Enter
INIT_SEEN=0
while kill -0 "${DEPLOY_PID}" 2>/dev/null; do
  if [[ "${INIT_SEEN}" -eq 0 ]]; then
    if grep -q "Init Done" "${LOG_DIR}/collect_stack.log" 2>/dev/null \
      || grep -q "Init Done" "${LOG_DIR}/deploy.log" 2>/dev/null; then
      INIT_SEEN=1
      echo "init_done" > "${PHASE_FILE}"
      echo "[studio_deploy] Init Done — waiting for ] (stand)"
    fi
  fi

  if [[ -f "${CMD_FILE}" ]]; then
    raw="$(tr -d '\r' < "${CMD_FILE}" | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [[ -n "${raw}" ]]; then
      : > "${CMD_FILE}"
      case "${raw}" in
        stop|quit)
          echo "[studio_deploy] stop"
          kill "${DEPLOY_PID}" 2>/dev/null || true
          break
          ;;
        stand|']')
          printf ']' >&3
          echo "standing" > "${PHASE_FILE}"
          echo "[studio_deploy] sent ]"
          ;;
        stream|enter)
          printf '\n' >&3
          echo "streaming" > "${PHASE_FILE}"
          echo "[studio_deploy] sent Enter"
          ;;
        deploy|y|Y)
          echo "[studio_deploy] deploy already running; ignore"
          ;;
        key:*)
          printf '%s' "${raw#key:}" >&3
          ;;
        *)
          # single char pass-through
          if [[ ${#raw} -eq 1 ]]; then
            printf '%s' "${raw}" >&3
          else
            echo "[studio_deploy] unknown cmd=${raw}"
          fi
          ;;
      esac
    fi
  fi
  sleep 0.2
done

wait "${DEPLOY_PID}" || true
echo "[studio_deploy] g1_deploy exited"
echo "done" > "${PHASE_FILE}"
