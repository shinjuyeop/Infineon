"""Validation-only diagnostics for Fast Reflex detector operating policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from terrain_fast_reflex_detector_v1 import TRACE_PRE_MS, FastReflexTrace, _target


PERSISTENCE_CANDIDATES = (1, 2, 3, 5, 8)
CHANNEL_NAMES = (
    "fsr1", "fsr2", "fsr3", "fsr4", "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
)
ONSET_BINS = ((0, 2, "0_2ms"), (3, 5, "3_5ms"), (6, 10, "6_10ms"),
              (11, 20, "11_20ms"), (21, 10_000, "over_20ms"))


@dataclass(frozen=True)
class ScoredRun:
    trace: FastReflexTrace
    scores: np.ndarray


def stable_endpoints(scores: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    active = np.asarray(scores) >= threshold
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    if persistence == 1:
        return np.flatnonzero(active)
    confirmed = np.convolve(active.astype(np.int16), np.ones(persistence, dtype=np.int16), "valid")
    return np.flatnonzero(confirmed == persistence) + persistence - 1


def target_onset(trace: FastReflexTrace, detector: str) -> int | None:
    indices = np.flatnonzero(_target(trace, detector)[TRACE_PRE_MS:])
    return None if not len(indices) else int(indices[0])


def run_policy_metrics(
    runs: Iterable[ScoredRun], detector: str, threshold: float, persistence: int
) -> dict[str, object]:
    runs = list(runs)
    target_count = detected_count = early_count = 0
    pre_onset_false = hazard_free_false = normal_terrain_false = target_negative_false = 0
    hazard_free_count = normal_terrain_count = target_negative_count = 0
    transition_latency, hazard_latency, anticipation_lead = [], [], []
    for item in runs:
        trace, scores = item.trace, item.scores
        stable = stable_endpoints(scores, threshold, persistence)
        onset = target_onset(trace, detector)
        cutoff = 100 if onset is None else onset
        early = stable[stable < cutoff]
        false_before = bool(len(early))
        pre_onset_false += false_before
        if false_before and onset is not None:
            early_count += 1
            anticipation_lead.append(float(onset - early[0]))

        any_physical = bool(np.any(trace.slip | trace.sink | trace.tilt))
        if not any_physical:
            hazard_free_count += 1; hazard_free_false += false_before
        if trace.metadata["scenario"] in {"marble_to_marble", "concrete_to_concrete"}:
            normal_terrain_count += 1; normal_terrain_false += false_before
        if onset is None:
            target_negative_count += 1; target_negative_false += bool(len(stable))
        else:
            target_count += 1
            post = stable[stable >= onset]
            if len(post):
                detected_count += 1
                transition_latency.append(float(post[0]))
                hazard_latency.append(float(post[0] - onset))
    def rate(count: int, total: int) -> float:
        return count / total if total else float("nan")
    def percentile(values: list[float], q: float) -> float | None:
        return None if not values else float(np.percentile(values, q))
    return {
        "threshold": float(threshold), "persistence": persistence,
        "run_count": len(runs), "target_runs": target_count,
        "run_target_detected": detected_count,
        "run_target_recall": rate(detected_count, target_count),
        "pre_onset_false_alarm_runs": pre_onset_false,
        "run_pre_onset_fpr": rate(pre_onset_false, len(runs)),
        "target_negative_runs": target_negative_count,
        "target_negative_false_alarm_runs": target_negative_false,
        "target_negative_run_fpr": rate(target_negative_false, target_negative_count),
        "hazard_free_runs": hazard_free_count,
        "hazard_free_false_alarm_runs": hazard_free_false,
        "hazard_free_run_fpr": rate(hazard_free_false, hazard_free_count),
        "normal_terrain_runs": normal_terrain_count,
        "normal_terrain_false_alarm_runs": normal_terrain_false,
        "normal_terrain_run_fpr": rate(normal_terrain_false, normal_terrain_count),
        "median_transition_to_detection_ms": percentile(transition_latency, 50),
        "p95_transition_to_detection_ms": percentile(transition_latency, 95),
        "median_hazard_to_detection_ms": percentile(hazard_latency, 50),
        "p95_hazard_to_detection_ms": percentile(hazard_latency, 95),
        "early_warning_runs": early_count,
        "median_anticipation_lead_ms": percentile(anticipation_lead, 50),
    }


def threshold_candidates(scores: np.ndarray, endpoint_threshold: float) -> np.ndarray:
    """A bounded 41-quantile grid plus the prior endpoint-selected threshold."""
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, 41))
    candidates = np.r_[np.nextafter(float(scores.max()), np.inf), candidates, endpoint_threshold]
    return np.unique(candidates)[::-1]


def best_run_policy(rows: list[dict[str, object]], maximum_fpr: float = 0.05) -> dict[str, object] | None:
    valid = [row for row in rows if float(row["run_pre_onset_fpr"]) <= maximum_fpr]
    if not valid:
        return None
    def key(row: dict[str, object]) -> tuple[float, ...]:
        latency = row["median_hazard_to_detection_ms"]
        return (
            float(row["run_target_recall"]), -float(row["run_pre_onset_fpr"]),
            -float("inf") if latency is None else -float(latency), float(row["threshold"]),
        )
    return max(valid, key=key)


def endpoint_confusion_rows(
    traces: list[FastReflexTrace], detector: str, window_ms: int,
    scores: np.ndarray, threshold: float,
) -> list[dict[str, object]]:
    labels = np.concatenate([_target(t, detector)[TRACE_PRE_MS:] for t in traces]).astype(bool)
    prediction = scores >= threshold
    windows = []
    for trace in traces:
        for endpoint in range(100):
            index = TRACE_PRE_MS + endpoint
            windows.append(trace.sensors[index - window_ms + 1:index + 1])
    values = np.asarray(windows)
    classes = {"TP": labels & prediction, "FN": labels & ~prediction,
               "FP": ~labels & prediction, "TN": ~labels & ~prediction}
    rows = []
    for name, mask in classes.items():
        row: dict[str, object] = {
            "detector": detector, "window_ms": window_ms, "confusion": name,
            "samples": int(mask.sum()), "score_median": float(np.median(scores[mask])) if mask.any() else None,
        }
        selected = values[mask]
        if len(selected):
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                row[f"{channel_name}_range_median"] = float(
                    np.median(np.ptp(selected[:, :, channel], axis=1))
                )
                row[f"{channel_name}_abs_delta_median"] = float(
                    np.median(np.abs(selected[:, -1, channel] - selected[:, 0, channel]))
                )
            fsr_sum = selected[:, :, :4].sum(axis=2)
            accel = np.linalg.norm(selected[:, :, 4:7], axis=2)
            gyro = np.linalg.norm(selected[:, :, 7:10], axis=2)
            imbalance = selected[:, :, :4].max(axis=2) - selected[:, :, :4].min(axis=2)
            row.update({
                "fsr_sum_range_median": float(np.median(np.ptp(fsr_sum, axis=1))),
                "fsr_spatial_imbalance_peak_median": float(np.median(imbalance.max(axis=1))),
                "accel_magnitude_range_median": float(np.median(np.ptp(accel, axis=1))),
                "gyro_magnitude_range_median": float(np.median(np.ptp(gyro, axis=1))),
            })
        rows.append(row)
    return rows


def sink_subgroup_rows(
    runs: list[ScoredRun], threshold: float, persistence: int
) -> list[dict[str, object]]:
    rows = []
    all_negative_scores = np.concatenate([
        item.scores[~item.trace.sink_or_tilt[TRACE_PRE_MS:]] for item in runs
    ])
    for subgroup in ("sink_only", "tilt_only", "sink_and_tilt"):
        selected = []
        for item in runs:
            has_sink, has_tilt = np.any(item.trace.sink), np.any(item.trace.tilt)
            actual = (has_sink and not has_tilt, has_tilt and not has_sink, has_sink and has_tilt)
            if actual[("sink_only", "tilt_only", "sink_and_tilt").index(subgroup)]:
                selected.append(item)
        positive_scores = np.concatenate([
            item.scores[item.trace.sink_or_tilt[TRACE_PRE_MS:]] for item in selected
        ]) if selected else np.asarray([])
        tp = int(np.sum(positive_scores >= threshold)); fn = len(positive_scores) - tp
        fp = int(np.sum(all_negative_scores >= threshold))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(positive_scores) if len(positive_scores) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        run_metrics = run_policy_metrics(selected, "sink_tilt", threshold, persistence) if selected else {}
        rows.append({
            "subgroup": subgroup, "runs": len(selected), "positive_endpoints": len(positive_scores),
            "endpoint_recall": recall, "endpoint_f1_vs_all_validation_negatives": f1,
            "score_median": float(np.median(positive_scores)) if len(positive_scores) else None,
            "score_p10": float(np.percentile(positive_scores, 10)) if len(positive_scores) else None,
            "score_p90": float(np.percentile(positive_scores, 90)) if len(positive_scores) else None,
            "run_recall": run_metrics.get("run_target_recall"),
            "median_hazard_to_detection_ms": run_metrics.get("median_hazard_to_detection_ms"),
            "p95_hazard_to_detection_ms": run_metrics.get("p95_hazard_to_detection_ms"),
            "tp_endpoints": tp, "fn_endpoints": fn,
        })
    return rows


def onset_relative_rows(
    runs: list[ScoredRun], detector: str, window_ms: int, threshold: float
) -> list[dict[str, object]]:
    bucket: dict[str, list[bool]] = {name: [] for _, _, name in ONSET_BINS}
    for item in runs:
        onset = target_onset(item.trace, detector)
        if onset is None:
            continue
        label = _target(item.trace, detector)[TRACE_PRE_MS:]
        for endpoint in np.flatnonzero(label):
            since = int(endpoint - onset)
            for low, high, name in ONSET_BINS:
                if low <= since <= high:
                    bucket[name].append(bool(item.scores[endpoint] >= threshold)); break
    return [{
        "detector": detector, "window_ms": window_ms, "time_since_onset_bin": name,
        "positive_endpoints": len(values), "true_positive_endpoints": int(sum(values)),
        "recall": float(np.mean(values)) if values else None,
    } for _, _, name in ONSET_BINS for values in [bucket[name]]]


def plot_slip_examples(
    runs: list[ScoredRun], threshold: float, output_dir: Path
) -> list[dict[str, object]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, tuple[ScoredRun, int]] = {}
    for item in runs:
        onset = target_onset(item.trace, "slip")
        if onset is None:
            continue
        labels = item.trace.slip[TRACE_PRE_MS:]
        predictions = item.scores >= threshold
        masks = {"TP": labels & predictions, "FN": labels & ~predictions,
                 "FP": ~labels & predictions, "TN": ~labels & ~predictions}
        for name, mask in masks.items():
            indices = np.flatnonzero(mask)
            if name not in candidates and len(indices): candidates[name] = (item, int(indices[0]))
    records = []
    for name in ("TP", "FN", "FP", "TN"):
        if name not in candidates:
            continue
        item, classified_endpoint = candidates[name]
        onset = target_onset(item.trace, "slip")
        assert onset is not None
        relative = np.arange(-5, 31)
        trace_indices = np.clip(TRACE_PRE_MS + onset + relative, 0, len(item.trace.sensors) - 1)
        sensors = item.trace.sensors[trace_indices]
        score_indices = onset + relative
        valid = (score_indices >= 0) & (score_indices < 100)
        score_plot = np.full(len(relative), np.nan); score_plot[valid] = item.scores[score_indices[valid]]
        oracle = item.trace.slip[trace_indices].astype(float)
        classified_relative = classified_endpoint - onset
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(relative, sensors[:, :4].sum(axis=1)); axes[0].set_ylabel("FSR sum")
        axes[1].plot(relative, sensors[:, 4:7]); axes[1].set_ylabel("Accel XYZ")
        axes[2].plot(relative, sensors[:, 7:10]); axes[2].set_ylabel("Gyro XYZ")
        axes[3].plot(relative, score_plot, label="score"); axes[3].axhline(threshold, color="r", ls="--")
        axes[3].step(relative, oracle, where="post", label="oracle slip"); axes[3].set_ylabel("score/state")
        for axis in axes:
            axis.axvline(classified_relative, color="black", ls=":", alpha=0.7)
        axes[3].set_xlabel("ms relative to physical slip onset"); axes[3].legend(loc="best")
        fig.suptitle(
            f"Slip 5 ms {name} at {classified_relative:+d} ms: {item.trace.metadata['run_id']}"
        )
        fig.tight_layout(); path = output_dir / f"slip_5ms_{name.lower()}.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        records.append({"confusion": name, "run_id": item.trace.metadata["run_id"],
                        "classified_endpoint_ms": classified_endpoint,
                        "hazard_onset_ms": onset, "plot": str(path)})
    return records
