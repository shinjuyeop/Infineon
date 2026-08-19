"""Pure causal-window, weighting, replay, and deterministic selection helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from walking_hazard_oracle_calibration_v1 import persistent_oracle


TERRAIN_LABELS = {"concrete": 0, "marble": 1, "ice": 2, "sand": 3}
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
MIXTURE_RATIOS = (0.25, 0.50, 1.00)
TRAINING_SEEDS = (20261001, 20261002, 20261003)
HAZARD_PERSISTENCE = (1, 3, 5, 8)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, np.float32) - self.mean) / self.std
        if not np.all(np.isfinite(result)):
            raise ValueError("normalization produced non-finite values")
        return result.astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "provenance": "combined train-only weighted source moments",
        }


def weighted_normalizer(
    groups: tuple[np.ndarray, ...], group_effective_weights: tuple[float, ...]
) -> Normalizer:
    if len(groups) != len(group_effective_weights) or not groups:
        raise ValueError("normalizer groups/weights mismatch")
    if any(array.ndim != 3 or array.shape[-1] != 10 for array in groups):
        raise ValueError("normalizer expects (N,T,10) groups")
    if any(weight <= 0.0 for weight in group_effective_weights):
        raise ValueError("normalizer group weights must be positive")
    total_weight = 0.0
    total = np.zeros(10, dtype=np.float64)
    total_square = np.zeros(10, dtype=np.float64)
    for array, effective in zip(groups, group_effective_weights):
        values = np.asarray(array, np.float64)
        per_value_weight = effective / (values.shape[0] * values.shape[1])
        total += per_value_weight * values.sum(axis=(0, 1))
        total_square += per_value_weight * np.square(values).sum(axis=(0, 1))
        total_weight += effective
    mean = total / total_weight
    variance = np.maximum(total_square / total_weight - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))


def causal_windows(
    fusion10: np.ndarray, endpoints: np.ndarray, window_ms: int
) -> np.ndarray:
    values = np.asarray(fusion10)
    endpoint_values = np.asarray(endpoints, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 10 or window_ms <= 0:
        raise ValueError("expected native Fusion10 trace and positive window")
    if np.any(endpoint_values < window_ms - 1) or np.any(endpoint_values >= len(values)):
        raise ValueError("endpoint cannot supply a complete causal window")
    return np.asarray([
        values[endpoint - window_ms + 1: endpoint + 1]
        for endpoint in endpoint_values
    ], dtype=np.float32)


def physical_oracle(
    trace: dict[str, np.ndarray], detector: str
) -> np.ndarray:
    if detector == "slip":
        observable = trace["tangential_anchor_drift_m"]
        valid = trace["slip_calibration_valid"]
        threshold, persistence = 0.050, 3
    elif detector == "sink":
        observable = trace["loaded_penetration_change_m"]
        valid = trace["sink_calibration_valid"]
        threshold, persistence = 0.0055, 20
    else:
        raise ValueError(detector)
    return persistent_oracle(
        observable, valid, trace["contact_episode_id"], threshold, persistence
    )


def stable_fire(
    scores: np.ndarray,
    threshold: float,
    persistence: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    probability = np.asarray(scores, dtype=float)
    allowed = np.ones(len(probability), dtype=bool) if eligible is None else np.asarray(eligible, bool)
    if probability.ndim != 1 or allowed.shape != probability.shape or persistence <= 0:
        raise ValueError("invalid stable-fire inputs")
    output = np.zeros(len(probability), dtype=bool)
    count = 0
    for index, active in enumerate(
        allowed & np.isfinite(probability) & (probability >= threshold)
    ):
        count = count + 1 if active else 0
        output[index] = count >= persistence
    return output


def percentile(values: list[float], quantile: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), quantile))


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, names: tuple[str, ...] = TERRAIN_NAMES
) -> dict[str, object]:
    y = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if y.shape != pred.shape or not len(y):
        raise ValueError("classification vectors must be non-empty and aligned")
    recalls = []
    by_class = {}
    for label, name in enumerate(names):
        mask = y == label
        recall = float(np.mean(pred[mask] == label)) if np.any(mask) else None
        if recall is not None:
            recalls.append(recall)
        by_class[name] = {"support": int(np.count_nonzero(mask)), "recall": recall}
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_accuracy": float(np.mean(recalls)),
        "class_metrics": by_class,
    }


def controlled_replay(
    run_id: np.ndarray,
    endpoint_ms: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    persistence: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for name in np.unique(run_id.astype(str)):
        mask = run_id.astype(str) == name
        order = np.argsort(endpoint_ms[mask])
        y = np.asarray(labels[mask][order], bool)
        q = np.asarray(scores[mask][order], float)
        ms = np.asarray(endpoint_ms[mask][order], int)
        fire = stable_fire(q, threshold, persistence)
        onset = int(np.flatnonzero(y)[0]) if np.any(y) else None
        first = int(np.flatnonzero(fire)[0]) if np.any(fire) else None
        post = (
            None if onset is None else next(
                (int(value) for value in np.flatnonzero(fire) if value >= onset), None
            )
        )
        pre = bool(onset is not None and first is not None and first < onset)
        rows.append({
            "run_id": name,
            "negative_only": onset is None,
            "target_onset_ms": None if onset is None else int(ms[onset]),
            "stable_firing_ms": None if first is None else int(ms[first]),
            "post_onset_stable_ms": None if post is None else int(ms[post]),
            "pre_onset_firing": pre,
            "post_onset_detection": post is not None,
        })
    negative = [row for row in rows if row["negative_only"]]
    positive = [row for row in rows if not row["negative_only"]]
    false_count = sum(row["stable_firing_ms"] is not None for row in negative) + sum(
        bool(row["pre_onset_firing"]) for row in positive
    )
    latency = [
        float(row["post_onset_stable_ms"] - row["target_onset_ms"])
        for row in positive if row["post_onset_detection"]
    ]
    return rows, {
        "runs": len(rows),
        "negative_runs": len(negative),
        "positive_runs": len(positive),
        "overall_causal_run_fpr": false_count / max(len(rows), 1),
        "negative_false_positive_runs": sum(
            row["stable_firing_ms"] is not None for row in negative
        ),
        "anticipation_runs": sum(bool(row["pre_onset_firing"]) for row in positive),
        "run_recall": sum(bool(row["post_onset_detection"]) for row in positive)
        / max(len(positive), 1),
        "latency_median_ms": percentile(latency, 50),
        "latency_p95_ms": percentile(latency, 95),
        "latency_max_ms": None if not latency else max(latency),
    }


def walking_hazard_replay(
    detector: str,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    scores: list[np.ndarray],
    threshold: float,
    persistence: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    latencies = []
    for metadata, trace, probability in zip(manifests, traces, scores):
        physical = physical_oracle(trace, detector)
        eligible = (
            np.asarray(trace["loaded_contact"], bool)
            & np.asarray(trace["pre_fall_valid"], bool)
            & ~np.asarray(trace["touchdown_transient"], bool)
        )
        fire = stable_fire(probability, threshold, persistence, eligible)
        physical_indices = np.flatnonzero(physical)
        model_indices = np.flatnonzero(fire)
        onset = None if not physical_indices.size else int(physical_indices[0])
        first = None if not model_indices.size else int(model_indices[0])
        post = None if onset is None else next(
            (int(value) for value in model_indices if value >= onset), None
        )
        anticipation = bool(onset is not None and first is not None and first < onset)
        if post is not None:
            latencies.append(post - int(onset))
        role = str(metadata["acquisition_role"])
        expected_positive = role == f"{detector}_candidate" and onset is not None
        rows.append({
            "detector": detector,
            "run_id": metadata["run_id"],
            "profile_name": metadata["profile_name"],
            "walking_speed_mps": float(metadata["walking_speed_mps"]),
            "variation_index": int(metadata["variation_index"]),
            "acquisition_role": role,
            "physical_source_valid": expected_positive,
            "physical_oracle_fire_sample": onset,
            "model_first_stable_sample": first,
            "post_oracle_detection_sample": post,
            "model_detected_post_onset": post is not None,
            "anticipation_false_alarm": anticipation,
            "physical_to_model_stable_latency_ms": None if post is None else post - int(onset),
            "normal_false_positive": bool(role == "hard_negative" and first is not None),
            "air_firing_samples": int(np.count_nonzero(fire & ~trace["left_contact"])),
            "post_fall_firing_samples": int(np.count_nonzero(fire & ~trace["pre_fall_valid"])),
            "touchdown_firing_samples": int(np.count_nonzero(fire & trace["touchdown_transient"])),
        })
    normal = [row for row in rows if row["acquisition_role"] == "hard_negative"]
    positive = [row for row in rows if row["physical_source_valid"]]
    detected = [row for row in positive if row["model_detected_post_onset"]]
    violations = sum(
        int(row["air_firing_samples"]) + int(row["post_fall_firing_samples"])
        + int(row["touchdown_firing_samples"]) for row in rows
    )
    profiles: dict[str, dict[str, object]] = {}
    for profile in sorted({str(row["profile_name"]) for row in positive}):
        group = [row for row in positive if row["profile_name"] == profile]
        profiles[profile] = {
            "physical_positive_runs": len(group),
            "detected_runs": sum(bool(row["model_detected_post_onset"]) for row in group),
            "detected_speeds_mps": sorted({
                float(row["walking_speed_mps"]) for row in group
                if row["model_detected_post_onset"]
            }),
        }
    return rows, {
        "runs": len(rows),
        "normal_runs": len(normal),
        "normal_false_positive_runs": sum(bool(row["normal_false_positive"]) for row in normal),
        "physical_positive_runs": len(positive),
        "detected_positive_runs": len(detected),
        "positive_run_recall": len(detected) / max(len(positive), 1),
        "anticipation_runs": sum(bool(row["anticipation_false_alarm"]) for row in positive),
        "label_mask_violation_samples": violations,
        "latency_median_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": None if not latencies else max(latencies),
        "profile_results": profiles,
    }


def terrain_candidate_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        float(row["walking_validation_macro_accuracy"]),
        float(row["walking_validation_accuracy"]),
        float(row["static_validation_accuracy"]),
        -int(row["parameters"]),
        -float(row["mixture_ratio"]),
        -int(row["training_seed"]),
    )


def select_terrain(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if bool(row["candidate_gate_pass"])]
    return None if not passing else max(passing, key=terrain_candidate_key)


def hazard_candidate_key(row: dict[str, object]) -> tuple[object, ...]:
    latency = row["walking_p95_stable_latency_ms"]
    return (
        -int(row["walking_normal_false_positive_runs"]),
        float(row["walking_positive_run_recall"]),
        float(row["controlled_validation_run_recall"]),
        -(float(latency) if latency is not None else 1e12),
        -int(row["parameters"]),
        -float(row["mixture_ratio"]),
        -int(row["training_seed"]),
        float(row["probability_threshold"]),
        -int(row["runtime_persistence"]),
    )


def select_hazard(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if bool(row["candidate_gate_pass"])]
    return None if not passing else max(passing, key=hazard_candidate_key)


def split_audit(manifests: list[dict[str, object]]) -> dict[str, object]:
    ownership: dict[str, set[str]] = defaultdict(set)
    episode_ownership: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in manifests:
        run_id = str(row["run_id"])
        split = str(row["split"])
        ownership[run_id].add(split)
        for episode in row.get("contact_episode_ids", []):
            episode_ownership[(run_id, int(episode))].add(split)
    leaking_runs = sorted(run for run, values in ownership.items() if len(values) != 1)
    leaking_episodes = sorted(
        f"{run}:{episode}" for (run, episode), values in episode_ownership.items()
        if len(values) != 1
    )
    return {
        "run_leakage_count": len(leaking_runs),
        "episode_leakage_count": len(leaking_episodes),
        "leaking_runs": leaking_runs,
        "leaking_episodes": leaking_episodes,
        "split_leakage_count": len(leaking_runs) + len(leaking_episodes),
    }
