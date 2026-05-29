#!/usr/bin/env bash
set -euo pipefail

CSI_DIR="${CSI_DIR:-data/csi}"
MAX_ACTIVE_CSI_FILE_GB="${MAX_ACTIVE_CSI_FILE_GB:-5}"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
PICOSCENES_COMMAND="${PICOSCENES_COMMAND:-PicoScenes \"-d debug -i 2 --mode logger --plot\"}"

mkdir -p "${CSI_DIR}"

picoscenes_pid=""

max_bytes() {
  python3 - <<PY
gb = float("${MAX_ACTIVE_CSI_FILE_GB}")
print(int(gb * 1024 * 1024 * 1024))
PY
}

newest_csi_file() {
  find "${CSI_DIR}" -maxdepth 1 -type f -name '*.csi' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}

start_picoscenes() {
  echo "[picoscenes-rotate] starting: ${PICOSCENES_COMMAND}"
  (
    cd "${CSI_DIR}"
    exec bash -lc "${PICOSCENES_COMMAND}"
  ) &
  picoscenes_pid="$!"
  echo "[picoscenes-rotate] pid=${picoscenes_pid}"
}

stop_picoscenes() {
  if [[ -z "${picoscenes_pid}" ]]; then
    return
  fi
  if kill -0 "${picoscenes_pid}" 2>/dev/null; then
    echo "[picoscenes-rotate] stopping pid=${picoscenes_pid}"
    kill "${picoscenes_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${picoscenes_pid}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "${picoscenes_pid}" 2>/dev/null; then
      echo "[picoscenes-rotate] killing pid=${picoscenes_pid}"
      kill -9 "${picoscenes_pid}" 2>/dev/null || true
    fi
  fi
  wait "${picoscenes_pid}" 2>/dev/null || true
  picoscenes_pid=""
}

cleanup() {
  stop_picoscenes
}
trap cleanup EXIT INT TERM

limit_bytes="$(max_bytes)"
echo "[picoscenes-rotate] CSI_DIR=${CSI_DIR}"
echo "[picoscenes-rotate] MAX_ACTIVE_CSI_FILE_GB=${MAX_ACTIVE_CSI_FILE_GB}"

start_picoscenes

while true; do
  if [[ -n "${picoscenes_pid}" ]] && ! kill -0 "${picoscenes_pid}" 2>/dev/null; then
    echo "[picoscenes-rotate] PicoScenes exited; restarting"
    wait "${picoscenes_pid}" 2>/dev/null || true
    picoscenes_pid=""
    start_picoscenes
  fi

  active_file="$(newest_csi_file || true)"
  if [[ -n "${active_file}" && -f "${active_file}" ]]; then
    size_bytes="$(stat -c '%s' "${active_file}")"
    if (( size_bytes >= limit_bytes )); then
      size_gb="$(python3 - <<PY
print(${size_bytes} / 1024 / 1024 / 1024)
PY
)"
      echo "[picoscenes-rotate] rotating ${active_file} size=${size_gb}GiB"
      stop_picoscenes
      rm -f "${active_file}"
      echo "[picoscenes-rotate] deleted ${active_file}"
      start_picoscenes
    fi
  fi

  sleep "${CHECK_INTERVAL}"
done
