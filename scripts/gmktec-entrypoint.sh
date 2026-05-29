#!/usr/bin/env bash
set -euo pipefail

if ! command -v PicoScenes >/dev/null 2>&1; then
  shopt -s nullglob
  debs=()
  for deb in /picoscenes-installer/*.deb; do
    case "$(basename "${deb}")" in
      picoscenes-source-updater*.deb)
        echo "[entrypoint] Skipping ${deb}; it configures host apt sources and is not PicoScenes itself."
        ;;
      *)
        debs+=("${deb}")
        ;;
    esac
  done
  shopt -u nullglob

  if [ "${#debs[@]}" -gt 0 ]; then
    echo "[entrypoint] Installing PicoScenes package(s): ${debs[*]}"
    apt-get update
    apt-get install -y --no-install-recommends "${debs[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
fi

if [ "${RUN_PICOSCENES}" = "true" ] && ! command -v PicoScenes >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[entrypoint] PicoScenes command was requested but PicoScenes is not installed.

Put the PicoScenes Linux .deb package in ./picoscenes-installer on the GMKtec
host, then rebuild/restart:

  docker compose -f docker-compose.gmktec.yml up --build

For preprocessing existing .csi files without launching PicoScenes, set:

  RUN_PICOSCENES=false
MSG
  exit 127
fi

args=(
  --watch-dir "${WATCH_DIR}"
  --pattern "${PATTERN}"
  --api-url "${API_URL}"
  --api-format "${API_FORMAT}"
  --legacy-series "${LEGACY_SERIES}"
  --sampling-rate "${SAMPLING_RATE}"
  --baseline-seconds "${BASELINE_SECONDS}"
  --smooth-seconds "${SMOOTH_SECONDS}"
  --max-tones "${MAX_TONES}"
  --chunk-mb "${CHUNK_MB}"
  --batch-size "${BATCH_SIZE}"
  --stable-seconds "${STABLE_SECONDS}"
  --poll-interval "${POLL_INTERVAL}"
)

if [ "${RUN_PICOSCENES}" = "true" ]; then
  args+=(--picoscenes-command "${PICOSCENES_COMMAND}")
fi

if [ "${DRY_RUN}" = "true" ]; then
  args+=(--dry-run)
fi

if [ "${PROCESS_ONCE}" = "true" ]; then
  args+=(--once)
fi

if [ "${FOLLOW_GROWING_FILES}" = "true" ]; then
  args+=(
    --follow-growing-files
    --follow-lag-bytes "${FOLLOW_LAG_BYTES}"
    --stream-start-at-end="${STREAM_START_AT_END}"
    --stream-read-mb "${STREAM_READ_MB}"
    --stream-post-interval "${STREAM_POST_INTERVAL}"
  )
fi

if [ "${CLEANUP_ENABLED}" = "true" ]; then
  args+=(--cleanup-enabled)
else
  args+=(--no-cleanup-enabled)
fi

args+=(
  --max-csi-dir-gb "${MAX_CSI_DIR_GB}"
  --max-active-csi-file-gb "${MAX_ACTIVE_CSI_FILE_GB}"
  --keep-latest-files "${KEEP_LATEST_FILES}"
  --cleanup-min-age-seconds "${CLEANUP_MIN_AGE_SECONDS}"
  --cleanup-interval "${CLEANUP_INTERVAL}"
)

if [ "${DELETE_PROCESSED_CSI}" = "true" ]; then
  args+=(--delete-processed-csi)
fi

echo "[entrypoint] Starting GMKtec CSI agent"
exec python /app/gmktec_csi_agent.py "${args[@]}"
