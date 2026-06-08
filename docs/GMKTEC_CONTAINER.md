# GMKtec CSI Agent Container

This container is the GMKtec-side component for the API-based exhibition setup.
It performs:

1. `.csi` file detection
2. CSI preprocessing into compact motion features
3. `POST /csi` to the current ML API container

It sends compact time-series samples, not raw `.csi` files.  This avoids moving hundreds of MB per capture into the ML API.

PicoScenes itself should run on the GMKtec host, not inside this container.
PicoScenes detects containers as a virtualization environment and may refuse to
run there.

## Directory Layout

```text
PicoScenes-Python-Toolbox/
  docker-compose.gmktec.yml
  Dockerfile.gmktec
  gmktec_csi_agent.py
  scripts/run-picoscenes-rotating.sh
  data/csi/                 # PicoScenes output and watched .csi files
```

## First Setup On GMKtec

Clone the repository:

```bash
git clone https://github.com/exyrias/PicoScenes-Python-Toolbox.git
cd PicoScenes-Python-Toolbox
```

Then build:

```bash
docker compose -f docker-compose.gmktec.yml build
```

## Run The Agent Container

Set the ML API URL and start the agent:

```bash
sudo env \
  RUN_PICOSCENES=false \
  FOLLOW_GROWING_FILES=true \
  STREAM_READ_MB=32 \
  STREAM_PARSER_LIMIT_GB=1.75 \
  MAX_CSI_DIR_GB=5 \
  API_URL=http://<ML_API_HOST>:8001/csi \
  docker compose -f docker-compose.gmktec.yml up -d --build
```

The container watches `data/csi` and posts feature batches to the ML API.

## Run PicoScenes On The Host With Rotation

Use the host-side wrapper instead of running `PicoScenes` directly.  It starts
PicoScenes from `data/csi`, monitors the active `.csi` file, and restarts
PicoScenes after deleting the active file when it exceeds the configured size.
Keep the limit below 2GB; long-running streams can otherwise hit invalid parser
offsets and stop producing usable CSI frames.

```bash
MAX_ACTIVE_CSI_FILE_GB=1.75 \
scripts/run-picoscenes-rotating.sh
```

The default PicoScenes command used by the wrapper is:

```bash
PicoScenes "-d debug -i 2 --mode logger --plot"
```

Override it when needed:

```bash
PICOSCENES_COMMAND='PicoScenes "-d debug -i 2 --mode logger --plot"' \
MAX_ACTIVE_CSI_FILE_GB=1.75 \
scripts/run-picoscenes-rotating.sh
```

## Process Existing CSI Files Only

Use this mode to test preprocessing without launching PicoScenes:

```bash
RUN_PICOSCENES=false \
DRY_RUN=true \
PROCESS_ONCE=true \
docker compose -f docker-compose.gmktec.yml up
```

Place `.csi` files under:

```text
data/csi/
```

## API Payload

The agent posts batches to:

```text
POST /csi
```

Default payload shape for the current ML API:

```json
{
  "samplingRateHz": 100.0,
  "pc1PhaseVariation": [-12.0, 8.1],
  "timestamp": "2026-05-28T12:00:00+00:00"
}
```

The agent computes several features internally.  In default `legacy` mode it
sends `pc1PhaseDiff` as `pc1PhaseVariation` because the current ML API schema
expects that field.  After the ML API is upgraded, set `API_FORMAT=features`
and `API_URL=http://<ML_API_HOST>:8001/csi/features` to send the extended
feature payload.

## Important Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_URL` | `http://127.0.0.1:8001/csi` | ML API endpoint |
| `API_FORMAT` | `legacy` | `legacy` sends the current ML API payload; `features` sends extended feature batches |
| `LEGACY_SERIES` | `motionScore` | Feature sent as `pc1PhaseVariation` in legacy mode |
| `RUN_PICOSCENES` | `false` | Keep this false; PicoScenes should run on the host |
| `PICOSCENES_COMMAND` | `PicoScenes "-d debug -i 2 --mode logger --plot"` | Acquisition command |
| `WATCH_DIR` | `/data/csi` | Directory watched for `.csi` files |
| `MAX_TONES` | `256` | Maximum subcarriers kept after downsampling |
| `CHUNK_MB` | `32` | Chunk size for parsing large `.csi` files |
| `BATCH_SIZE` | `500` | Samples per API POST |
| `FOLLOW_GROWING_FILES` | `true` | Process a growing `.csi` file instead of waiting for completion |
| `FOLLOW_LAG_BYTES` | `1048576` | Read this many bytes behind the file end to avoid partial frames |
| `STREAM_READ_MB` | `32` | Maximum bytes read from a growing `.csi` file per parser call |
| `STREAM_POST_INTERVAL` | `1.0` | Flush streaming feature batches at least this often |
| `CLEANUP_ENABLED` | `true` | Automatically delete old `.csi` files when the directory is too large |
| `MAX_CSI_DIR_GB` | `20` | Maximum size for the watched CSI directory |
| `MAX_ACTIVE_CSI_FILE_GB` | `10` | Host wrapper active-file rotation limit |
| `KEEP_LATEST_FILES` | `1` | Always keep this many newest `.csi` files |
| `CLEANUP_MIN_AGE_SECONDS` | `300` | Never delete files newer than this age |
| `CLEANUP_INTERVAL` | `60` | How often to check disk usage |
| `DELETE_PROCESSED_CSI` | `false` | Delete completed files immediately after successful processing |
| `DRY_RUN` | `false` | Print batches without POSTing |

## Real-Time Mode

The default container mode is now real-time oriented:

```text
FOLLOW_GROWING_FILES=true
```

In this mode the agent does not wait for PicoScenes to finish writing the file.
It tails the growing `.csi` file, stays about `FOLLOW_LAG_BYTES` behind the file
end to avoid incomplete frames, reads at most `STREAM_READ_MB` per parser call,
extracts features, and posts batches while the recording is still running.

Recommended GMKtec agent command:

```bash
sudo env \
  RUN_PICOSCENES=false \
  FOLLOW_GROWING_FILES=true \
  STREAM_READ_MB=32 \
  MAX_CSI_DIR_GB=5 \
  API_URL=http://<ML_API_HOST>:8001/csi \
  docker compose -f docker-compose.gmktec.yml up -d --build
```

Then run the host-side PicoScenes wrapper:

```bash
MAX_ACTIVE_CSI_FILE_GB=5 scripts/run-picoscenes-rotating.sh
```

The first `BASELINE_SECONDS` seconds are used to learn the empty-room baseline.
During that warm-up period, no feature batches are posted.  After the baseline
is ready, logs should include:

```text
[agent] streaming baseline ready rows=500 threshold=...
[agent] posted batch start=...
```

## Storage Cleanup

The container agent deletes older inactive `.csi` files when the watched
directory exceeds `MAX_CSI_DIR_GB`.  The host-side PicoScenes wrapper handles
the active file by stopping PicoScenes, deleting the oversized active file, and
starting PicoScenes again.

Default policy:

```text
CLEANUP_ENABLED=true
MAX_CSI_DIR_GB=20
KEEP_LATEST_FILES=1
CLEANUP_MIN_AGE_SECONDS=300
```

This means:

- inactive files are capped by the agent
- the active file is capped by `scripts/run-picoscenes-rotating.sh`
- do not delete files from the last 5 minutes
- cap the directory at about 20 GB by deleting oldest inactive files

For a tighter cap:

```bash
sudo env \
  MAX_CSI_DIR_GB=5 \
  RUN_PICOSCENES=false \
  FOLLOW_GROWING_FILES=true \
  API_URL=http://<ML_API_HOST>:8001/csi \
  docker compose -f docker-compose.gmktec.yml up -d --build

MAX_ACTIVE_CSI_FILE_GB=5 scripts/run-picoscenes-rotating.sh
```

If you want completed files deleted immediately after batch processing, enable:

```bash
DELETE_PROCESSED_CSI=true
```

## Notes

- The container uses `network_mode: host` so `127.0.0.1:8001` can reach an ML
  API running on the same GMKtec host.
- PicoScenes should run on the host.  The container only tails `.csi` files and
  sends compact batches to the ML API.
