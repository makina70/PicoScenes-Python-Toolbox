"""
GMKtec-side CSI agent.

This watches PicoScenes .csi output files, extracts motion-oriented CSI features,
and posts compact feature batches to the ML API.

Target API:
    POST /csi/features

The script intentionally sends features, not raw .csi files.  Raw CSI files can
be hundreds of MB and are expensive to move into the ML container.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from picoscenes import Picoscenes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_link_counts(csi: dict) -> tuple[int, int]:
    num_tx = int(csi.get("numTx") or csi.get("numSTS") or 1)
    num_rx = int(csi.get("numRx") or 1)
    return num_tx, num_rx


def reshape_csi(data: np.ndarray, num_tones: int, num_tx: int, num_rx: int) -> np.ndarray | None:
    expected = num_tones * num_tx * num_rx
    if data.size != expected:
        return None
    return data.reshape((num_tones, num_tx, num_rx), order="F")


def choose_subcarrier_stride(num_tones: int, max_tones: int) -> int:
    return max(1, int(np.ceil(num_tones / max_tones)))


def load_csi_tensor_chunked(
    file_path: str | Path,
    max_tones: int,
    chunk_bytes: int,
) -> tuple[np.ndarray, dict]:
    tensors: list[np.ndarray] = []
    metadata: dict | None = None
    selected_indices: np.ndarray | None = None
    pos = 0
    total_frames = 0
    kept_frames = 0

    while True:
        frames = Picoscenes(str(file_path), pos, pos + chunk_bytes)
        if not frames.raw:
            break

        for frame in frames.raw:
            total_frames += 1
            if "CSI" not in frame:
                continue

            csi = frame["CSI"]
            if "CSI" in csi:
                data = np.asarray(csi["CSI"], dtype=np.complex64).ravel()
            elif "Real" in csi and "Imag" in csi:
                data = np.asarray(csi["Real"], dtype=np.float32) + 1j * np.asarray(
                    csi["Imag"], dtype=np.float32
                )
                data = data.astype(np.complex64).ravel()
            else:
                continue

            num_tones = int(csi["numTones"])
            num_tx, num_rx = get_link_counts(csi)
            tensor = reshape_csi(data, num_tones, num_tx, num_rx)
            if tensor is None:
                continue

            if metadata is None:
                stride = choose_subcarrier_stride(num_tones, max_tones)
                selected_indices = np.arange(0, num_tones, stride)
                full_subcarrier_index = list(csi.get("SubcarrierIndex", []))
                if full_subcarrier_index:
                    subcarrier_index = [full_subcarrier_index[i] for i in selected_indices]
                else:
                    subcarrier_index = selected_indices.tolist()

                metadata = {
                    "numTones": int(len(selected_indices)),
                    "originalNumTones": num_tones,
                    "subcarrierStride": stride,
                    "numTx": num_tx,
                    "numRx": num_rx,
                    "subcarrierIndex": subcarrier_index,
                    "chunkBytes": chunk_bytes,
                }

            if selected_indices is None:
                continue
            if (num_tx, num_rx) != (metadata["numTx"], metadata["numRx"]):
                continue
            if num_tones < metadata["originalNumTones"]:
                continue

            tensors.append(tensor[selected_indices, :, :])
            kept_frames += 1

        next_pos = int(frames.next_pos)
        del frames
        if next_pos <= pos:
            break
        pos = next_pos

    if metadata is None or not tensors:
        raise ValueError(f"No usable CSI frames found: {file_path}")

    metadata["totalFramesSeen"] = total_frames
    metadata["keptCsiFrames"] = kept_frames
    return np.stack(tensors, axis=0).astype(np.complex64, copy=False), metadata


def clean_phase_per_link(csi_tensor: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(csi_tensor), axis=1).astype(np.float32)
    num_tones = phase.shape[1]
    x = np.arange(num_tones, dtype=np.float32)
    design = np.column_stack([x, np.ones(num_tones, dtype=np.float32)])

    flat = phase.reshape((-1, num_tones))
    coefs, _, _, _ = np.linalg.lstsq(design, flat.T, rcond=None)
    fitted = coefs[0, :, None] * x[None, :] + coefs[1, :, None]
    cleaned = (flat - fitted).astype(np.float32)
    return cleaned.reshape(phase.shape)


def robust_zscore_by_baseline(matrix: np.ndarray, baseline_rows: int, eps: float = 1e-9) -> np.ndarray:
    baseline = matrix[:baseline_rows]
    center = np.median(baseline, axis=0, keepdims=True)
    mad = np.median(np.abs(baseline - center), axis=0, keepdims=True)
    return ((matrix - center) / (1.4826 * mad + eps)).astype(np.float32)


def first_principal_component(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vh[0]
    variances = singular_values**2
    ratio = float(variances[0] / variances.sum()) if variances.sum() > 0 else 0.0
    return pc1.astype(np.float32), ratio


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def extract_features(
    csi_tensor: np.ndarray,
    sampling_rate_hz: float,
    baseline_seconds: float,
    smooth_seconds: float,
) -> tuple[np.ndarray, dict]:
    cleaned_phase = clean_phase_per_link(csi_tensor)
    phase_diff = np.diff(cleaned_phase, axis=0)
    feature_rows = phase_diff.shape[0]

    baseline_rows = int(round(baseline_seconds * sampling_rate_hz))
    baseline_rows = max(20, min(baseline_rows, max(20, feature_rows // 4)))

    phase_features = phase_diff.reshape((feature_rows, -1))
    phase_z = robust_zscore_by_baseline(phase_features, baseline_rows)
    pc1_phase, pc1_ratio = first_principal_component(phase_z)
    pc1_z = robust_zscore_by_baseline(pc1_phase.reshape(-1, 1), baseline_rows).ravel()

    amplitude = np.log(np.abs(csi_tensor) + 1e-9).astype(np.float32)
    amplitude_baseline = np.median(amplitude[: baseline_rows + 1], axis=0, keepdims=True)
    amplitude_delta = amplitude[1:] - amplitude_baseline
    amplitude_z_flat = robust_zscore_by_baseline(
        amplitude_delta.reshape((feature_rows, -1)), baseline_rows
    )

    phase_energy = np.sqrt(np.mean(phase_z**2, axis=1)).astype(np.float32)
    amplitude_energy = np.sqrt(np.mean(amplitude_z_flat**2, axis=1)).astype(np.float32)

    raw_motion_score = np.sqrt(phase_energy**2 + amplitude_energy**2 + np.abs(pc1_z))
    smooth_window = max(1, int(round(smooth_seconds * sampling_rate_hz)))
    motion_score = moving_average(raw_motion_score, smooth_window)

    baseline_score = motion_score[:baseline_rows]
    baseline_center = float(np.median(baseline_score))
    baseline_mad = float(np.median(np.abs(baseline_score - baseline_center)))
    threshold = baseline_center + 4.0 * 1.4826 * baseline_mad
    active_flag = (motion_score > threshold).astype(np.float32)

    features = np.column_stack(
        [
            motion_score,
            np.full(feature_rows, threshold, dtype=np.float32),
            active_flag,
            pc1_phase,
            phase_energy,
            amplitude_energy,
        ]
    ).astype(np.float32)

    metadata = {
        "baselineSeconds": baseline_seconds,
        "baselineRows": baseline_rows,
        "smoothSeconds": smooth_seconds,
        "motionThreshold": threshold,
        "pc1PhaseExplainedVarianceRatio": pc1_ratio,
        "featureColumns": [
            "motionScore",
            "motionThreshold",
            "activeFlag",
            "pc1PhaseDiff",
            "phaseDiffEnergy",
            "amplitudeDeltaEnergy",
        ],
    }
    return features, metadata


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def feature_batches(
    features: np.ndarray,
    sampling_rate_hz: float,
    batch_size: int,
) -> Iterable[tuple[int, dict]]:
    columns = [
        "motionScore",
        "motionThreshold",
        "activeFlag",
        "pc1PhaseDiff",
        "phaseDiffEnergy",
        "amplitudeDeltaEnergy",
    ]
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        payload_features = {
            name: batch[:, index].astype(float).tolist() for index, name in enumerate(columns)
        }
        payload_features["timeSeconds"] = (
            (np.arange(start, start + batch.shape[0], dtype=np.float64) / sampling_rate_hz)
            .astype(float)
            .tolist()
        )
        yield start, payload_features


def wait_until_file_stable(path: Path, stable_seconds: float, poll_interval: float) -> bool:
    last_size = -1
    stable_since: float | None = None
    while True:
        if not path.exists():
            return False
        size = path.stat().st_size
        now = time.monotonic()
        if size == last_size:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_seconds:
                return True
        else:
            last_size = size
            stable_since = None
        time.sleep(poll_interval)


def process_file(path: Path, args: argparse.Namespace, session_id: str) -> None:
    print(f"[agent] processing {path}")
    csi_tensor, csi_metadata = load_csi_tensor_chunked(
        path,
        max_tones=args.max_tones,
        chunk_bytes=args.chunk_mb * 1024 * 1024,
    )
    features, feature_metadata = extract_features(
        csi_tensor,
        sampling_rate_hz=args.sampling_rate,
        baseline_seconds=args.baseline_seconds,
        smooth_seconds=args.smooth_seconds,
    )

    base_payload = {
        "sessionId": session_id,
        "source": "gmktec",
        "sourceFile": str(path),
        "samplingRateHz": args.sampling_rate,
        "timestamp": utc_now(),
        "csiMetadata": csi_metadata,
        "featureMetadata": feature_metadata,
    }

    print(
        "[agent] extracted "
        f"frames={csi_metadata['keptCsiFrames']} tones={csi_metadata['numTones']} "
        f"batches={int(np.ceil(features.shape[0] / args.batch_size))}"
    )

    for start, batch_features in feature_batches(features, args.sampling_rate, args.batch_size):
        payload = {
            **base_payload,
            "batchStart": start,
            "features": batch_features,
        }
        if args.dry_run:
            print(
                "[agent] dry-run batch "
                f"start={start} samples={len(batch_features['motionScore'])} "
                f"latestMotionScore={batch_features['motionScore'][-1]:.3f}"
            )
            continue

        try:
            result = post_json(args.api_url, payload, timeout=args.timeout)
            print(f"[agent] posted batch start={start} result={result}")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[agent] POST failed start={start}: {exc}")
            if args.stop_on_error:
                raise


def run_picoscenes(command: str) -> subprocess.Popen:
    print(f"[agent] starting PicoScenes command: {command}")
    return subprocess.Popen(shlex.split(command))


def watch_loop(args: argparse.Namespace) -> None:
    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    processed: set[Path] = set()
    session_prefix = args.session_id or f"gmktec-{uuid.uuid4().hex[:8]}"

    pico_process: subprocess.Popen | None = None
    if args.picoscenes_command:
        pico_process = run_picoscenes(args.picoscenes_command)

    try:
        while True:
            for path in sorted(watch_dir.glob(args.pattern)):
                resolved = path.resolve()
                if resolved in processed:
                    continue
                if not wait_until_file_stable(path, args.stable_seconds, args.poll_interval):
                    continue
                session_id = f"{session_prefix}-{path.stem}"
                process_file(path, args, session_id=session_id)
                processed.add(resolved)

            if args.once:
                break
            if pico_process is not None and pico_process.poll() is not None:
                print(f"[agent] PicoScenes exited with code {pico_process.returncode}")
                pico_process = None
            time.sleep(args.poll_interval)
    finally:
        if pico_process is not None and pico_process.poll() is None:
            pico_process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8001/csi/features")
    parser.add_argument("--watch-dir", default=".")
    parser.add_argument("--pattern", default="*.csi")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--file", type=Path, help="Process one .csi file immediately.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--smooth-seconds", type=float, default=0.5)
    parser.add_argument("--max-tones", type=int, default=256)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--stable-seconds", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--picoscenes-command",
        help='Optional command to launch, e.g. PicoScenes "-d debug -i 2 --mode logger --plot"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file:
        session_id = args.session_id or f"gmktec-{uuid.uuid4().hex[:8]}-{args.file.stem}"
        process_file(args.file, args, session_id=session_id)
        return
    watch_loop(args)


if __name__ == "__main__":
    main()
