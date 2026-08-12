"""Leakage-safe Float Host detectors for Fast Reflex corrected-v2 traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from terrain_fast_reflex_v1 import (
    HAZARD_CONFIRMATION_SAMPLES,
    PRIMARY_WINDOWS_MS,
    TRACE_PRE_MS,
    FastReflexTrace, HazardThresholds,
    calibrate_hazard_thresholds,
    label_trace,
    validate_split_integrity, validate_trace,
)


DETECTORS = ("slip", "sink_tilt")
FUSION_CHANNELS = 10
TRAIN_ENDPOINT_STRIDE_MS = 2
REPLAY_PERSISTENCE_SAMPLES = 3
SEED = 20260812


def load_corrected_traces(
    source: Path, split: str | None = None
) -> tuple[list[FastReflexTrace], list[dict[str, str]]]:
    """Load preserved arrays without importing the MuJoCo dataset generator."""
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(source / "inputs_fusion10.npz", allow_pickle=False) as data:
        sensors = data["sensors"]
        timestamps = data["sample_time_s"]
        sensor_ids = data["run_id"].astype(str)
    with np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as data:
        oracle = data["oracle"]
        slip, sink, tilt = data["slip"], data["sink"], data["tilt"]
        oracle_ids = data["run_id"].astype(str)
    manifest_ids = np.asarray([row["run_id"] for row in rows])
    if not (len(rows) == len(sensors) == len(oracle)
            and np.array_equal(manifest_ids, sensor_ids)
            and np.array_equal(manifest_ids, oracle_ids)):
        raise ValueError("manifest/input/oracle run ordering does not match")
    traces, selected_rows = [], []
    for index, row in enumerate(rows):
        if split is not None and row["split"] != split:
            continue
        trace = FastReflexTrace(
            metadata=dict(row), timestamps_s=np.asarray(timestamps[index], dtype=np.float64),
            sensors=np.asarray(sensors[index], dtype=np.float64),
            oracle=np.asarray(oracle[index], dtype=np.float64),
            slip=np.asarray(slip[index], dtype=bool), sink=np.asarray(sink[index], dtype=bool),
            tilt=np.asarray(tilt[index], dtype=bool), valid=bool(int(row["valid"])),
            invalid_reason=row["invalid_reason"],
        )
        validate_trace(trace); traces.append(trace); selected_rows.append(row)
    return traces, selected_rows


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ChannelNormalizer":
        if values.ndim != 3 or values.shape[2] != FUSION_CHANNELS or not len(values):
            raise ValueError(f"expected non-empty (N,L,{FUSION_CHANNELS}), got {values.shape}")
        mean = values.astype(np.float64).mean(axis=(0, 1))
        std = values.astype(np.float64).std(axis=(0, 1))
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (values.astype(np.float32) - self.mean) / self.std
        if not np.all(np.isfinite(result)):
            raise ValueError("normalization produced NaN/Inf")
        return result.astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    clean_negative: np.ndarray
    run_ids: np.ndarray
    endpoints_ms: np.ndarray


def audit_corrected_v2(source: Path) -> tuple[list[FastReflexTrace], dict[str, object]]:
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("derived_artifact_revision") != 2:
        raise ValueError("source must be corrected-v2 (derived_artifact_revision=2)")
    traces, rows = load_corrected_traces(source)
    if not traces or not all(trace.valid for trace in traces):
        raise ValueError("source contains no traces or invalid traces")
    validate_split_integrity(rows)
    expected = {
        "train": {"multisine", "filtered_random", "sparse_aggregate"},
        "validation": {"crosshatch", "rounded_ridges"},
        "test": {"warped_multisine", "smooth_random_patches"},
    }
    for split, families in expected.items():
        actual = {t.metadata["surface_family"] for t in traces if t.metadata["split"] == split}
        if actual != families:
            raise ValueError(f"{split} family ownership mismatch: {sorted(actual)}")
    scenarios = {name: sum(t.metadata["scenario"] == name for t in traces) for name in (
        "marble_to_ice", "marble_to_sand", "marble_to_marble", "concrete_to_concrete"
    )}
    physical = [np.any(t.slip | t.sink | t.tilt) for t in traces]
    report = {
        "source": str(source.resolve()),
        "derived_artifact_revision": 2,
        "runs": len(traces),
        "valid_runs": sum(t.valid for t in traces),
        "fusion_shape": [len(traces), *traces[0].sensors.shape],
        "oracle_shape": [len(traces), *traces[0].oracle.shape],
        "scenarios": scenarios,
        "physical_hazard_runs": int(sum(physical)),
        "physical_normal_runs": int(len(traces) - sum(physical)),
        "physical_negative_by_split": {
            split: sum(not p for t, p in zip(traces, physical) if t.metadata["split"] == split)
            for split in expected
        },
        "fusion10_only": True,
        "sample_rate_hz": 1000,
        "split_leakage": False,
    }
    return traces, report


def _target(trace: FastReflexTrace, detector: str) -> np.ndarray:
    if detector == "slip":
        return trace.slip
    if detector == "sink_tilt":
        return trace.sink_or_tilt
    raise ValueError(f"unknown detector {detector!r}")


def make_windows(
    traces: Iterable[FastReflexTrace], detector: str, window_ms: int, stride_ms: int = 1
) -> WindowSet:
    if window_ms not in PRIMARY_WINDOWS_MS:
        raise ValueError(f"unsupported primary window {window_ms} ms")
    xs, ys, clean, ids, endpoints = [], [], [], [], []
    for trace in traces:
        target = _target(trace, detector)
        any_hazard = trace.slip | trace.sink | trace.tilt
        for endpoint_ms in range(0, 100, stride_ms):
            endpoint = TRACE_PRE_MS + endpoint_ms
            start = endpoint - window_ms + 1
            xs.append(trace.sensors[start : endpoint + 1])
            ys.append(target[endpoint])
            clean.append(not any_hazard[endpoint])
            ids.append(trace.metadata["run_id"])
            endpoints.append(endpoint_ms)
    return WindowSet(
        np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8),
        np.asarray(clean, dtype=bool), np.asarray(ids), np.asarray(endpoints, dtype=np.int16)
    )


def balanced_run_weights(data: WindowSet) -> np.ndarray:
    weights = np.empty(len(data.y), dtype=np.float32)
    unique, counts = np.unique(data.run_ids, return_counts=True)
    per_run = dict(zip(unique, counts))
    for i, run_id in enumerate(data.run_ids):
        weights[i] = 1.0 / per_run[run_id]
    for label in (0, 1):
        mask = data.y == label
        total = float(weights[mask].sum())
        if total <= 0:
            raise ValueError(f"training subset has no class {label}")
        weights[mask] *= 0.5 / total
    weights *= len(weights) / weights.sum()
    return weights


def build_model(window_ms: int, seed: int = SEED, pooling: str = "average"):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input((window_ms, FUSION_CHANNELS), name="fusion10_window")
    values = tf.keras.layers.Conv1D(12, 5, padding="same", activation="relu", name="conv1")(inputs)
    values = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu", name="conv2")(values)
    if pooling == "average":
        values = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(values)
    elif pooling == "max":
        values = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(values)
    elif pooling == "average_max":
        average = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(values)
        maximum = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(values)
        values = tf.keras.layers.Concatenate(name="pool_concat")([average, maximum])
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="hazard_score")(values)
    model = tf.keras.Model(inputs, outputs, name=f"fast_reflex_{window_ms}ms")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def resource_estimate(window_ms: int, pooling: str = "average") -> dict[str, int]:
    if pooling not in {"average", "max", "average_max"}:
        raise ValueError(f"unknown pooling {pooling!r}")
    pooled_channels = 32 if pooling == "average_max" else 16
    parameters = 5 * 10 * 12 + 12 + 3 * 12 * 16 + 16 + pooled_channels + 1
    return {
        "parameters": parameters,
        "float_parameter_bytes": 4 * parameters,
        "conv_dense_macs_per_inference": (
            window_ms * (5 * 10 * 12 + 3 * 12 * 16) + pooled_channels
        ),
    }


def select_threshold(y_true: np.ndarray, scores: np.ndarray, maximum_fpr: float = 0.05) -> float:
    negatives, positives = scores[y_true == 0], scores[y_true == 1]
    if not len(negatives) or not len(positives):
        raise ValueError("validation needs both positive and negative endpoints")
    candidates = np.r_[np.nextafter(float(scores.max()), np.inf), np.unique(scores)[::-1]]
    best = None
    for threshold in candidates:
        fpr = float(np.mean(negatives >= threshold))
        recall = float(np.mean(positives >= threshold))
        if fpr <= maximum_fpr + 1e-12:
            key = (recall, -fpr, threshold)
            if best is None or key > best[0]:
                best = (key, float(threshold))
    if best is None:
        raise RuntimeError("no threshold satisfies FPR constraint")
    return best[1]


def binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, clean_negative: np.ndarray
) -> dict[str, object]:
    prediction = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    negatives = y_true == 0
    clean = negatives & clean_negative
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(np.mean(prediction[negatives])), "tpr": float(recall),
        "normal_false_alarm_rate": float(np.mean(prediction[negatives])),
        "clean_negative_fpr_diagnostic": float(np.mean(prediction[clean])) if np.any(clean) else None,
        "support": int(len(y_true)), "positive_support": int(np.sum(y_true)),
        "negative_support": int(np.sum(negatives)), "threshold": float(threshold),
    }


def _first_stable(scores: np.ndarray, threshold: float, persistence: int) -> int | None:
    run = 0
    for endpoint, active in enumerate(scores >= threshold):
        run = run + 1 if active else 0
        if run >= persistence:
            return endpoint
    return None


def replay_rows(
    model, normalizer: ChannelNormalizer, traces: list[FastReflexTrace], detector: str,
    window_ms: int, threshold: float, split: str
) -> list[dict[str, object]]:
    rows = []
    if not traces:
        return rows
    replay = make_windows(traces, detector, window_ms)
    all_scores = model.predict(
        normalizer.transform(replay.x), batch_size=1024, verbose=0
    ).reshape(-1)
    if len(all_scores) != 100 * len(traces):
        raise ValueError("offline replay must produce 100 endpoint scores per trace")
    for trace_index, trace in enumerate(traces):
        scores = all_scores[trace_index * 100 : (trace_index + 1) * 100]
        detected = _first_stable(scores, threshold, REPLAY_PERSISTENCE_SAMPLES)
        target_indices = np.flatnonzero(_target(trace, detector)[TRACE_PRE_MS:])
        onset = int(target_indices[0]) if len(target_indices) else None
        anticipation = detected is not None and onset is not None and detected < onset
        rows.append({
            "detector": detector, "window_ms": window_ms, "split": split,
            "run_id": trace.metadata["run_id"], "scenario": trace.metadata["scenario"],
            "target_occurred": int(onset is not None), "hazard_onset_ms": onset if onset is not None else "",
            "stable_detection_ms": detected if detected is not None else "",
            "transition_to_stable_detection_ms": detected if detected is not None else "",
            "hazard_to_stable_detection_ms": (
                detected - onset if detected is not None and onset is not None and not anticipation else ""
            ),
            "early_anticipation": int(anticipation),
            "anticipation_lead_ms": onset - detected if anticipation else "",
            "missed_target": int(onset is not None and detected is None),
            "false_alarm_run": int(onset is None and detected is not None),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def threshold_sensitivity(
    traces: list[FastReflexTrace], canonical_thresholds: dict[str, float]
) -> list[dict[str, object]]:
    calibration = [t for t in traces if t.metadata["split"] == "train" and t.metadata["scenario"] in {
        "marble_to_marble", "concrete_to_concrete"
    }]
    variants = [
        ("canonical_p99_mad6_confirm3", 99.0, 6.0, 3),
        ("percentile_99.5", 99.5, 6.0, 3), ("percentile_99.9", 99.9, 6.0, 3),
        ("mad_multiplier_4", 99.0, 4.0, 3), ("mad_multiplier_8", 99.0, 8.0, 3),
        ("confirmation_2", 99.0, 6.0, 2), ("confirmation_5", 99.0, 6.0, 5),
    ]
    rows = []
    for name, percentile, mad, confirmation in variants:
        if name.startswith("canonical_"):
            thresholds = HazardThresholds(**canonical_thresholds)
            # The stored revision-2 masks are authoritative. Recalibration from the
            # float32 export can move a boundary by one sample through roundoff.
            labeled = [t for t in traces if t.metadata["split"] != "test"]
        else:
            thresholds = calibrate_hazard_thresholds(calibration, percentile, mad)
            labeled = [label_trace(t, thresholds, confirmation) for t in traces if t.metadata["split"] != "test"]
        def normal_rate(split: str) -> float:
            selected = [t for t in labeled if t.metadata["split"] == split and t.metadata["scenario"] in {
                "marble_to_marble", "concrete_to_concrete"
            }]
            return float(np.mean([np.any(t.slip | t.sink | t.tilt) for t in selected]))
        def target_stats(scenario: str, detector: str) -> tuple[float, object, object]:
            selected = [t for t in labeled if t.metadata["scenario"] == scenario]
            onsets = []
            for trace in selected:
                indices = np.flatnonzero(_target(trace, detector)[TRACE_PRE_MS:])
                if len(indices): onsets.append(int(indices[0]))
            return (
                len(onsets) / len(selected),
                float(np.median(onsets)) if onsets else "",
                float(np.percentile(onsets, 95)) if onsets else "",
            )
        ice = target_stats("marble_to_ice", "slip")
        sand = target_stats("marble_to_sand", "sink_tilt")
        rows.append({
            "threshold_variant": name, "normal_percentile": percentile,
            "mad_multiplier": mad, "consecutive_samples": confirmation,
            "train_normal_hazard_rate": normal_rate("train"),
            "validation_normal_hazard_rate": normal_rate("validation"),
            "ice_target_onset_coverage": ice[0], "ice_median_transition_to_onset_ms": ice[1],
            "ice_p95_transition_to_onset_ms": ice[2],
            "sand_target_onset_coverage": sand[0], "sand_median_transition_to_onset_ms": sand[1],
            "sand_p95_transition_to_onset_ms": sand[2],
            **{f"threshold_{key}": value for key, value in asdict(thresholds).items()},
        })
    return rows


def subset_traces(traces: list[FastReflexTrace], maximum: int | None) -> list[FastReflexTrace]:
    if maximum is None:
        return traces
    counts: dict[tuple[str, str], int] = {}
    result = []
    for trace in traces:
        key = (trace.metadata["split"], trace.metadata["scenario"])
        if counts.get(key, 0) < maximum:
            result.append(trace); counts[key] = counts.get(key, 0) + 1
    return result
