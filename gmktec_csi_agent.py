"""
GMKtec-side CSI agent.

This watches PicoScenes .csi output files, extracts motion-oriented CSI features,
and posts compact feature batches to the ML API.

Default target API:
    POST /csi

The script intentionally sends features, not raw .csi files.  Raw CSI files can
be hundreds of MB and are expensive to move into the ML container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import uuid
from collections import deque
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


def frame_to_csi_tensor(frame: dict) -> tuple[np.ndarray, dict] | None:
    if "CSI" not in frame:
        return None

    csi = frame["CSI"]
    if "CSI" in csi:
        data = np.asarray(csi["CSI"], dtype=np.complex64).ravel()
    elif "Real" in csi and "Imag" in csi:
        data = np.asarray(csi["Real"], dtype=np.float32) + 1j * np.asarray(
            csi["Imag"], dtype=np.float32
        )
        data = data.astype(np.complex64).ravel()
    else:
        return None

    num_tones = int(csi["numTones"])
    num_tx, num_rx = get_link_counts(csi)
    tensor = reshape_csi(data, num_tones, num_tx, num_rx)
    if tensor is None:
        return None

    metadata = {
        "numTones": num_tones,
        "numTx": num_tx,
        "numRx": num_rx,
        "subcarrierIndex": list(csi.get("SubcarrierIndex", [])),
    }
    return tensor, metadata


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


def clean_phase_frame(csi_frame: np.ndarray) -> np.ndarray:
    return clean_phase_per_link(csi_frame[np.newaxis, ...])[0]


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


class StreamingFeatureExtractor:
    def __init__(
        self,
        sampling_rate_hz: float,
        baseline_seconds: float,
        smooth_seconds: float,
        max_tones: int,
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.baseline_rows = max(20, int(round(baseline_seconds * sampling_rate_hz)))
        self.smooth_window = max(1, int(round(smooth_seconds * sampling_rate_hz)))
        self.max_tones = max_tones

        self.metadata: dict | None = None
        self.selected_indices: np.ndarray | None = None
        self.previous_phase: np.ndarray | None = None
        self.previous_motion_scores: list[float] = []
        self.sample_index = 0

        self.baseline_phase_diffs: list[np.ndarray] = []
        self.baseline_amplitudes: list[np.ndarray] = []
        self.phase_center: np.ndarray | None = None
        self.phase_scale: np.ndarray | None = None
        self.amplitude_baseline: np.ndarray | None = None
        self.amplitude_center: np.ndarray | None = None
        self.amplitude_scale: np.ndarray | None = None
        self.motion_threshold: float | None = None

    def _initialize_metadata(self, frame_metadata: dict) -> None:
        original_num_tones = int(frame_metadata["numTones"])
        stride = choose_subcarrier_stride(original_num_tones, self.max_tones)
        self.selected_indices = np.arange(0, original_num_tones, stride)

        full_subcarrier_index = frame_metadata.get("subcarrierIndex", [])
        if full_subcarrier_index:
            subcarrier_index = [full_subcarrier_index[i] for i in self.selected_indices]
        else:
            subcarrier_index = self.selected_indices.tolist()

        self.metadata = {
            "numTones": int(len(self.selected_indices)),
            "originalNumTones": original_num_tones,
            "subcarrierStride": stride,
            "numTx": int(frame_metadata["numTx"]),
            "numRx": int(frame_metadata["numRx"]),
            "subcarrierIndex": subcarrier_index,
            "streaming": True,
        }

    def _finalize_baseline(self) -> None:
        phase_baseline = np.asarray(self.baseline_phase_diffs, dtype=np.float32)
        amplitude_baseline_frames = np.asarray(self.baseline_amplitudes, dtype=np.float32)
        self.amplitude_baseline = np.median(amplitude_baseline_frames, axis=0, keepdims=True)

        amplitude_delta_baseline = amplitude_baseline_frames[1:] - self.amplitude_baseline
        amplitude_delta_flat = amplitude_delta_baseline.reshape((amplitude_delta_baseline.shape[0], -1))

        self.phase_center = np.median(phase_baseline, axis=0, keepdims=True)
        self.phase_scale = (
            1.4826 * np.median(np.abs(phase_baseline - self.phase_center), axis=0, keepdims=True)
            + 1e-9
        )
        self.amplitude_center = np.median(amplitude_delta_flat, axis=0, keepdims=True)
        self.amplitude_scale = (
            1.4826
            * np.median(np.abs(amplitude_delta_flat - self.amplitude_center), axis=0, keepdims=True)
            + 1e-9
        )

        phase_z = (phase_baseline - self.phase_center) / self.phase_scale
        amplitude_z = (amplitude_delta_flat - self.amplitude_center) / self.amplitude_scale
        phase_energy = np.sqrt(np.mean(phase_z**2, axis=1))
        amplitude_energy = np.sqrt(np.mean(amplitude_z**2, axis=1))
        baseline_scores = np.sqrt(phase_energy**2 + amplitude_energy**2)
        center = float(np.median(baseline_scores))
        mad = float(np.median(np.abs(baseline_scores - center)))
        self.motion_threshold = center + 4.0 * 1.4826 * mad

    def add_frame(self, csi_frame: np.ndarray, frame_metadata: dict) -> tuple[int, np.ndarray] | None:
        if self.metadata is None:
            self._initialize_metadata(frame_metadata)

        if self.selected_indices is None or self.metadata is None:
            return None

        if (int(frame_metadata["numTx"]), int(frame_metadata["numRx"])) != (
            self.metadata["numTx"],
            self.metadata["numRx"],
        ):
            return None
        if int(frame_metadata["numTones"]) < self.metadata["originalNumTones"]:
            return None

        frame = csi_frame[self.selected_indices, :, :]
        phase = clean_phase_frame(frame)
        amplitude = np.log(np.abs(frame) + 1e-9).astype(np.float32)

        self.baseline_amplitudes.append(amplitude)
        if len(self.baseline_amplitudes) > self.baseline_rows + 1:
            self.baseline_amplitudes = self.baseline_amplitudes[-(self.baseline_rows + 1) :]

        if self.previous_phase is None:
            self.previous_phase = phase
            return None

        phase_diff = (phase - self.previous_phase).reshape(1, -1).astype(np.float32)
        self.previous_phase = phase

        if self.motion_threshold is None:
            self.baseline_phase_diffs.append(phase_diff.ravel())
            if len(self.baseline_phase_diffs) >= self.baseline_rows:
                self._finalize_baseline()
                print(
                    "[agent] streaming baseline ready "
                    f"rows={self.baseline_rows} threshold={self.motion_threshold:.3f}"
                )
            return None

        assert self.phase_center is not None
        assert self.phase_scale is not None
        assert self.amplitude_baseline is not None
        assert self.amplitude_center is not None
        assert self.amplitude_scale is not None
        assert self.motion_threshold is not None

        amplitude_delta = (amplitude - self.amplitude_baseline[0]).reshape(1, -1)
        phase_z = (phase_diff - self.phase_center) / self.phase_scale
        amplitude_z = (amplitude_delta - self.amplitude_center) / self.amplitude_scale

        phase_energy = float(np.sqrt(np.mean(phase_z**2)))
        amplitude_energy = float(np.sqrt(np.mean(amplitude_z**2)))
        pc1_phase_diff = float(np.mean(phase_z))
        raw_motion_score = float(np.sqrt(phase_energy**2 + amplitude_energy**2))

        self.previous_motion_scores.append(raw_motion_score)
        if len(self.previous_motion_scores) > self.smooth_window:
            self.previous_motion_scores = self.previous_motion_scores[-self.smooth_window :]
        motion_score = float(np.mean(self.previous_motion_scores))
        active_flag = 1.0 if motion_score > self.motion_threshold else 0.0

        feature = np.asarray(
            [
                motion_score,
                self.motion_threshold,
                active_flag,
                pc1_phase_diff,
                phase_energy,
                amplitude_energy,
            ],
            dtype=np.float32,
        )
        index = self.sample_index
        self.sample_index += 1
        return index, feature


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


def features_to_payload(
    indexed_features: list[tuple[int, np.ndarray]],
    sampling_rate_hz: float,
) -> tuple[int, dict]:
    columns = [
        "motionScore",
        "motionThreshold",
        "activeFlag",
        "pc1PhaseDiff",
        "phaseDiffEnergy",
        "amplitudeDeltaEnergy",
    ]
    start = indexed_features[0][0]
    indices = np.asarray([item[0] for item in indexed_features], dtype=np.float64)
    batch = np.asarray([item[1] for item in indexed_features], dtype=np.float32)
    payload_features = {
        name: batch[:, index].astype(float).tolist() for index, name in enumerate(columns)
    }
    payload_features["timeSeconds"] = (indices / sampling_rate_hz).astype(float).tolist()
    return start, payload_features


def feature_series_hash(batch_features: dict, series_name: str) -> str:
    series = np.asarray(batch_features[series_name], dtype=np.float32)
    return hashlib.sha1(series.tobytes()).hexdigest()


def csi_tensor_hash(tensor: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(tensor)
    return hashlib.sha1(contiguous.view(np.uint8)).hexdigest()


def send_feature_payload(
    args: argparse.Namespace,
    base_payload: dict,
    batch_start: int,
    batch_features: dict,
) -> None:
    series = batch_features[args.legacy_series]
    if args.api_format == "legacy":
        payload = {
            "samplingRateHz": base_payload["samplingRateHz"],
            "pc1PhaseVariation": series,
            "timestamp": utc_now(),
        }
    else:
        payload = {
            **base_payload,
            "batchStart": batch_start,
            "features": batch_features,
        }

    if args.dry_run:
        print(
            "[agent] dry-run batch "
            f"start={batch_start} samples={len(batch_features['motionScore'])} "
            f"latestMotionScore={batch_features['motionScore'][-1]:.3f} "
            f"apiFormat={args.api_format}"
        )
        return

    series_array = np.asarray(series, dtype=np.float32)
    series_hash = hashlib.sha1(series_array.tobytes()).hexdigest()[:12]
    series_summary = (
        f"legacySeries={args.legacy_series} "
        f"n={series_array.size} "
        f"min={float(np.min(series_array)):.6g} "
        f"max={float(np.max(series_array)):.6g} "
        f"std={float(np.std(series_array)):.6g} "
        f"first={float(series_array[0]):.6g} "
        f"last={float(series_array[-1]):.6g} "
        f"sha1={series_hash}"
    )
    try:
        result = post_json(args.api_url, payload, timeout=args.timeout)
        print(f"[agent] posted batch start={batch_start} {series_summary} result={result}")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[agent] POST failed start={batch_start}: {exc}")
        if args.stop_on_error:
            raise


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


def cleanup_csi_files(
    watch_dir: Path,
    pattern: str,
    active_paths: set[Path],
    max_dir_gb: float,
    keep_latest_files: int,
    min_age_seconds: float,
) -> None:
    if max_dir_gb <= 0:
        return

    files = [path for path in watch_dir.glob(pattern) if path.is_file()]
    if not files:
        return

    stats: list[tuple[Path, int, float]] = []
    total_size = 0
    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        stats.append((path, stat.st_size, stat.st_mtime))
        total_size += stat.st_size

    max_bytes = int(max_dir_gb * 1024 * 1024 * 1024)
    if total_size <= max_bytes:
        return

    now = time.time()
    protected = {path.resolve() for path in active_paths}
    latest = sorted(stats, key=lambda item: item[2], reverse=True)[:keep_latest_files]
    protected.update(path.resolve() for path, _, _ in latest)

    candidates = sorted(stats, key=lambda item: item[2])
    for path, size, mtime in candidates:
        if total_size <= max_bytes:
            break
        if path.resolve() in protected:
            continue
        if now - mtime < min_age_seconds:
            continue
        try:
            path.unlink()
            total_size -= size
            print(f"[agent] cleanup deleted {path} size={size / (1024 * 1024):.1f}MiB")
        except OSError as exc:
            print(f"[agent] cleanup could not delete {path}: {exc}")


def maybe_cleanup(watch_dir: Path, active_paths: set[Path], args: argparse.Namespace) -> None:
    if not args.cleanup_enabled:
        return
    cleanup_csi_files(
        watch_dir=watch_dir,
        pattern=args.pattern,
        active_paths=active_paths,
        max_dir_gb=args.max_csi_dir_gb,
        keep_latest_files=args.keep_latest_files,
        min_age_seconds=args.cleanup_min_age_seconds,
    )


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
        send_feature_payload(args, base_payload, start, batch_features)

    if args.delete_processed_csi and not args.dry_run:
        try:
            path.unlink()
            print(f"[agent] deleted processed CSI file {path}")
        except OSError as exc:
            print(f"[agent] could not delete processed CSI file {path}: {exc}")


def follow_growing_file(path: Path, args: argparse.Namespace, session_id: str) -> str:
    print(f"[agent] following growing file {path}")
    extractor = StreamingFeatureExtractor(
        sampling_rate_hz=args.sampling_rate,
        baseline_seconds=args.baseline_seconds,
        smooth_seconds=args.smooth_seconds,
        max_tones=args.max_tones,
    )
    pos = 0
    pending: list[tuple[int, np.ndarray]] = []
    last_post = time.monotonic()
    last_cleanup = time.monotonic()
    base_payload: dict | None = None
    last_sent_hash: str | None = None
    recent_frame_hashes: deque[str] = deque(maxlen=4096)
    recent_frame_hash_set: set[str] = set()

    while True:
        if not path.exists():
            print(f"[agent] followed file disappeared: {path}")
            return "done"

        size = path.stat().st_size
        max_active_bytes = int(args.max_active_csi_file_gb * 1024 * 1024 * 1024)
        if max_active_bytes > 0 and args.picoscenes_command and size >= max_active_bytes:
            print(
                "[agent] active CSI file exceeded limit "
                f"path={path} size={size / (1024 * 1024 * 1024):.2f}GiB "
                f"limit={args.max_active_csi_file_gb:.2f}GiB"
            )
            return "rotate"

        readable_end = size - args.follow_lag_bytes
        if readable_end > pos + 4:
            chunk_end = min(readable_end, pos + args.stream_read_mb * 1024 * 1024)
            try:
                frames = Picoscenes(str(path), pos, chunk_end)
            except Exception as exc:
                print(f"[agent] waiting for complete CSI frame at pos={pos}: {exc}")
                time.sleep(args.poll_interval)
                continue

            for frame in frames.raw:
                parsed = frame_to_csi_tensor(frame)
                if parsed is None:
                    continue
                tensor, frame_metadata = parsed
                frame_hash = csi_tensor_hash(tensor)
                if frame_hash in recent_frame_hash_set:
                    continue
                if len(recent_frame_hashes) == recent_frame_hashes.maxlen:
                    oldest_hash = recent_frame_hashes.popleft()
                    recent_frame_hash_set.discard(oldest_hash)
                recent_frame_hashes.append(frame_hash)
                recent_frame_hash_set.add(frame_hash)
                result = extractor.add_frame(tensor, frame_metadata)
                if result is not None:
                    pending.append(result)

            next_pos = int(frames.next_pos)
            del frames
            if next_pos > pos:
                pos = next_pos
            else:
                pos = chunk_end
                print(f"[agent] advanced stream position to chunk_end={chunk_end} because next_pos did not move")

        if extractor.metadata is not None and base_payload is None:
            base_payload = {
                "sessionId": session_id,
                "source": "gmktec",
                "sourceFile": str(path),
                "samplingRateHz": args.sampling_rate,
                "timestamp": utc_now(),
                "csiMetadata": extractor.metadata,
                "featureMetadata": {
                    "baselineSeconds": args.baseline_seconds,
                    "baselineRows": extractor.baseline_rows,
                    "smoothSeconds": args.smooth_seconds,
                    "streaming": True,
                    "featureColumns": [
                        "motionScore",
                        "motionThreshold",
                        "activeFlag",
                        "pc1PhaseDiff",
                        "phaseDiffEnergy",
                        "amplitudeDeltaEnergy",
                    ],
                },
            }

        should_flush = pending and (
            len(pending) >= args.batch_size
            or time.monotonic() - last_post >= args.stream_post_interval
        )
        if should_flush and base_payload is not None:
            batch_start, batch_features = features_to_payload(pending, args.sampling_rate)
            current_hash = feature_series_hash(batch_features, args.legacy_series)
            if current_hash == last_sent_hash:
                print(
                    "[agent] skipped duplicate batch "
                    f"start={batch_start} legacySeries={args.legacy_series} "
                    f"sha1={current_hash[:12]}"
                )
            else:
                send_feature_payload(args, base_payload, batch_start, batch_features)
                last_sent_hash = current_hash
            pending.clear()
            last_post = time.monotonic()

        if time.monotonic() - last_cleanup >= args.cleanup_interval:
            maybe_cleanup(path.parent, {path.resolve()}, args)
            last_cleanup = time.monotonic()

        if args.once and size <= pos + args.follow_lag_bytes:
            if pending and base_payload is not None:
                batch_start, batch_features = features_to_payload(pending, args.sampling_rate)
                current_hash = feature_series_hash(batch_features, args.legacy_series)
                if current_hash == last_sent_hash:
                    print(
                        "[agent] skipped duplicate batch "
                        f"start={batch_start} legacySeries={args.legacy_series} "
                        f"sha1={current_hash[:12]}"
                    )
                else:
                    send_feature_payload(args, base_payload, batch_start, batch_features)
            if args.delete_processed_csi and not args.dry_run:
                try:
                    path.unlink()
                    print(f"[agent] deleted processed CSI file {path}")
                except OSError as exc:
                    print(f"[agent] could not delete processed CSI file {path}: {exc}")
            return "done"

        time.sleep(args.poll_interval)


def run_picoscenes(command: str) -> subprocess.Popen:
    print(f"[agent] starting PicoScenes command: {command}")
    return subprocess.Popen(shlex.split(command))


def stop_picoscenes(process: subprocess.Popen | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[agent] stopping PicoScenes pid={process.pid}")
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[agent] killing PicoScenes pid={process.pid}")
        process.kill()
        process.wait(timeout=timeout)


def watch_loop(args: argparse.Namespace) -> None:
    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    processed: set[Path] = set()
    session_prefix = args.session_id or f"gmktec-{uuid.uuid4().hex[:8]}"

    pico_process: subprocess.Popen | None = None
    last_cleanup = time.monotonic()
    if args.picoscenes_command:
        pico_process = run_picoscenes(args.picoscenes_command)

    try:
        while True:
            if args.picoscenes_command and pico_process is None:
                pico_process = run_picoscenes(args.picoscenes_command)

            for path in sorted(watch_dir.glob(args.pattern)):
                resolved = path.resolve()
                if resolved in processed:
                    continue
                if args.follow_growing_files:
                    session_id = f"{session_prefix}-{path.stem}"
                    result = follow_growing_file(path, args, session_id=session_id)
                    if result == "rotate":
                        stop_picoscenes(pico_process)
                        pico_process = None
                        try:
                            path.unlink()
                            print(f"[agent] deleted oversized active CSI file {path}")
                        except FileNotFoundError:
                            pass
                        except OSError as exc:
                            print(f"[agent] could not delete oversized active CSI file {path}: {exc}")
                    else:
                        processed.add(resolved)
                    continue
                if not wait_until_file_stable(path, args.stable_seconds, args.poll_interval):
                    continue
                session_id = f"{session_prefix}-{path.stem}"
                process_file(path, args, session_id=session_id)
                processed.add(resolved)

            if time.monotonic() - last_cleanup >= args.cleanup_interval:
                maybe_cleanup(watch_dir, set(), args)
                last_cleanup = time.monotonic()

            if args.once:
                break
            if pico_process is not None and pico_process.poll() is not None:
                print(f"[agent] PicoScenes exited with code {pico_process.returncode}")
                pico_process = None
            time.sleep(args.poll_interval)
    finally:
        stop_picoscenes(pico_process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8001/csi")
    parser.add_argument("--api-format", choices=["legacy", "features"], default="legacy")
    parser.add_argument(
        "--legacy-series",
        choices=[
            "pc1PhaseDiff",
            "motionScore",
            "phaseDiffEnergy",
            "amplitudeDeltaEnergy",
        ],
        default="pc1PhaseDiff",
        help="Feature series sent as pc1PhaseVariation when --api-format=legacy.",
    )
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
    parser.add_argument("--follow-growing-files", action="store_true")
    parser.add_argument("--follow-lag-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--stream-read-mb", type=int, default=32)
    parser.add_argument("--stream-post-interval", type=float, default=1.0)
    parser.add_argument("--cleanup-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-csi-dir-gb", type=float, default=20.0)
    parser.add_argument("--max-active-csi-file-gb", type=float, default=10.0)
    parser.add_argument("--keep-latest-files", type=int, default=1)
    parser.add_argument("--cleanup-min-age-seconds", type=float, default=300.0)
    parser.add_argument("--cleanup-interval", type=float, default=60.0)
    parser.add_argument("--delete-processed-csi", action="store_true")
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
