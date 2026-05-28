# GMKtec CSI Agent Container

This container is the GMKtec-side component for the API-based exhibition setup.
It performs:

1. CSI acquisition with `PicoScenes`
2. `.csi` file detection
3. CSI preprocessing into compact motion features
4. `POST /csi/features` to the ML API container

It sends features, not raw `.csi` files.  This avoids moving hundreds of MB per
capture into the ML API.

## Directory Layout

```text
PicoScenes-Python-Toolbox/
  docker-compose.gmktec.yml
  Dockerfile.gmktec
  gmktec_csi_agent.py
  data/csi/                 # PicoScenes output and watched .csi files
  picoscenes-installer/     # optional PicoScenes Linux .deb package(s)
```

## First Setup On GMKtec

Clone the repository:

```bash
git clone https://github.com/exyrias/PicoScenes-Python-Toolbox.git
cd PicoScenes-Python-Toolbox
```

If PicoScenes is not already available inside the image, put the Linux
PicoScenes `.deb` package in:

```text
picoscenes-installer/
```

Then build:

```bash
docker compose -f docker-compose.gmktec.yml build
```

## Run With Acquisition Enabled

Set the ML API URL and start the agent:

```bash
API_URL=http://<ML_API_HOST>:8001/csi/features \
docker compose -f docker-compose.gmktec.yml up
```

The default PicoScenes command is:

```bash
PicoScenes "-d debug -i 2 --mode logger --plot"
```

Override it when needed:

```bash
PICOSCENES_COMMAND='PicoScenes "-d debug -i 2 --mode logger --plot"' \
API_URL=http://<ML_API_HOST>:8001/csi/features \
docker compose -f docker-compose.gmktec.yml up
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
POST /csi/features
```

Example payload shape:

```json
{
  "sessionId": "gmktec-xxxx-capture",
  "source": "gmktec",
  "sourceFile": "/data/csi/capture.csi",
  "samplingRateHz": 100.0,
  "timestamp": "2026-05-28T12:00:00+00:00",
  "batchStart": 0,
  "csiMetadata": {
    "originalNumTones": 2025,
    "numTones": 254,
    "subcarrierStride": 8,
    "numTx": 2,
    "numRx": 2
  },
  "featureMetadata": {
    "motionThreshold": 18.667,
    "featureColumns": [
      "motionScore",
      "motionThreshold",
      "activeFlag",
      "pc1PhaseDiff",
      "phaseDiffEnergy",
      "amplitudeDeltaEnergy"
    ]
  },
  "features": {
    "timeSeconds": [0.0, 0.01],
    "motionScore": [1.2, 1.4],
    "motionThreshold": [4.0, 4.0],
    "activeFlag": [0.0, 0.0],
    "pc1PhaseDiff": [-12.0, 8.1],
    "phaseDiffEnergy": [0.9, 1.0],
    "amplitudeDeltaEnergy": [0.7, 0.8]
  }
}
```

## Important Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_URL` | `http://127.0.0.1:8001/csi/features` | ML API endpoint |
| `RUN_PICOSCENES` | `true` | Launch PicoScenes from inside the container |
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

Recommended GMKtec command when PicoScenes runs on the host:

```bash
RUN_PICOSCENES=false \
FOLLOW_GROWING_FILES=true \
API_URL=http://<ML_API_HOST>:8001/csi/features \
docker compose -f docker-compose.gmktec.yml up -d --build
```

Then run PicoScenes from the watched directory:

```bash
cd data/csi
PicoScenes "-d debug -i 2 --mode logger --plot"
```

The first `BASELINE_SECONDS` seconds are used to learn the empty-room baseline.
During that warm-up period, no feature batches are posted.  After the baseline
is ready, logs should include:

```text
[agent] streaming baseline ready rows=500 threshold=...
[agent] posted batch start=...
```

## Storage Cleanup

The agent now keeps GMKtec storage bounded by default.  It does not delete the
currently written `.csi` file.  Instead, every `CLEANUP_INTERVAL` seconds it
checks `data/csi` and, if the directory exceeds `MAX_CSI_DIR_GB`, deletes older
`.csi` files first.

Default policy:

```text
CLEANUP_ENABLED=true
MAX_CSI_DIR_GB=20
KEEP_LATEST_FILES=1
CLEANUP_MIN_AGE_SECONDS=300
```

This means:

- keep the active/latest file
- do not delete files from the last 5 minutes
- cap the directory at about 20 GB by deleting oldest files

For a tighter cap:

```bash
MAX_CSI_DIR_GB=5 \
RUN_PICOSCENES=false \
FOLLOW_GROWING_FILES=true \
API_URL=http://<ML_API_HOST>:8001/csi/features \
docker compose -f docker-compose.gmktec.yml up -d --build
```

If you want completed files deleted immediately after batch processing, enable:

```bash
DELETE_PROCESSED_CSI=true
```

## Notes

- The container uses `network_mode: host` so `127.0.0.1:8001` can reach an ML
  API running on the same GMKtec host.
- The compose file runs with `privileged: true` and mounts `/dev` because
  PicoScenes hardware access may require device access.
- If `--plot` needs X11/GUI access, extra host display mounts may be required.
  For unattended operation, prefer running PicoScenes without `--plot`.
