"""Frozen Slip candidate metrics for the one-shot final test replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from terrain_fast_reflex_detector_v1 import TRACE_PRE_MS, FastReflexTrace
from terrain_fast_reflex_validation_v1 import stable_endpoints


FROZEN_ARCHITECTURE = "average_max"
FROZEN_WINDOW_MS = 5
FROZEN_THRESHOLD = 0.7317748725
FROZEN_PERSISTENCE = 3
EXPECTED_TEST_FAMILIES = {"warped_multisine", "smooth_random_patches"}


@dataclass(frozen=True)
class FrozenTestResult:
    metrics: dict[str, object]
    run_rows: list[dict[str, object]]
    latency_rows: list[dict[str, object]]


def _percentiles(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"median_{prefix}_ms": None if not values else float(np.percentile(values, 50)),
        f"p95_{prefix}_ms": None if not values else float(np.percentile(values, 95)),
        f"max_{prefix}_ms": None if not values else float(np.max(values)),
    }


def validate_test_ownership(traces: Iterable[FastReflexTrace]) -> list[FastReflexTrace]:
    traces = list(traces)
    if not traces:
        raise ValueError("test split is empty")
    splits = {trace.metadata["split"] for trace in traces}
    families = {trace.metadata["surface_family"] for trace in traces}
    if splits != {"test"}:
        raise ValueError(f"non-test trace supplied: {sorted(splits)}")
    if families != EXPECTED_TEST_FAMILIES:
        raise ValueError(f"test family ownership mismatch: {sorted(families)}")
    return traces


def evaluate_frozen_scores(
    traces: Iterable[FastReflexTrace], scores: np.ndarray
) -> FrozenTestResult:
    """Evaluate already-produced scores without selecting or tuning any policy value."""
    traces = validate_test_ownership(traces)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != 100 * len(traces) or not np.all(np.isfinite(scores)):
        raise ValueError("expected 100 finite endpoint scores per test trace")

    labels = np.concatenate([trace.slip[TRACE_PRE_MS:] for trace in traces]).astype(bool)
    prediction = scores >= FROZEN_THRESHOLD
    tp = int(np.sum(labels & prediction)); fp = int(np.sum(~labels & prediction))
    fn = int(np.sum(labels & ~prediction)); tn = int(np.sum(~labels & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    run_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    transition_latency: list[float] = []
    hazard_latency: list[float] = []
    anticipation_leads: list[float] = []
    target_runs = detected_runs = pre_onset_false_runs = 0
    target_negative_runs = target_negative_false_runs = 0
    hazard_free_runs = hazard_free_false_runs = 0
    normal_runs = normal_false_runs = 0
    anticipation_runs = anticipation_firings = 0

    for index, trace in enumerate(traces):
        run_scores = scores[index * 100:(index + 1) * 100]
        stable = stable_endpoints(run_scores, FROZEN_THRESHOLD, FROZEN_PERSISTENCE)
        onset_indices = np.flatnonzero(trace.slip[TRACE_PRE_MS:])
        onset = None if not len(onset_indices) else int(onset_indices[0])
        pre_onset = stable[stable < (100 if onset is None else onset)]
        post_onset = stable if onset is None else stable[stable >= onset]
        first_post = None if onset is None or not len(post_onset) else int(post_onset[0])
        pre_false = bool(len(pre_onset))
        any_stable = bool(len(stable))
        any_physical = bool(np.any(trace.slip | trace.sink | trace.tilt))
        normal_terrain = trace.metadata["scenario"] in {"marble_to_marble", "concrete_to_concrete"}

        pre_onset_false_runs += int(pre_false)
        if onset is None:
            target_negative_runs += 1
            target_negative_false_runs += int(any_stable)
        else:
            target_runs += 1
            detected_runs += int(first_post is not None)
            if first_post is not None:
                transition_latency.append(float(first_post))
                hazard_latency.append(float(first_post - onset))
            if pre_false:
                anticipation_runs += 1
                anticipation_firings += len(pre_onset)
                anticipation_leads.extend(float(onset - endpoint) for endpoint in pre_onset)
        if not any_physical:
            hazard_free_runs += 1
            hazard_free_false_runs += int(any_stable)
        if normal_terrain:
            normal_runs += 1
            normal_false_runs += int(pre_false)

        row = {
            "run_id": trace.metadata["run_id"],
            "scenario": trace.metadata["scenario"],
            "family": trace.metadata["surface_family"],
            "surface": trace.metadata.get("surface_id", trace.metadata.get("surface", "")),
            "target_occurred": int(onset is not None),
            "hazard_onset_ms": "" if onset is None else onset,
            "stable_firing_count": int(len(stable)),
            "pre_onset_firing_count": int(len(pre_onset)),
            "first_stable_detection_ms": "" if not len(stable) else int(stable[0]),
            "first_post_onset_stable_detection_ms": "" if first_post is None else first_post,
            "transition_to_stable_detection_ms": "" if first_post is None else first_post,
            "hazard_onset_to_stable_detection_ms": "" if first_post is None else first_post - onset,
            "anticipation_firing": int(onset is not None and pre_false),
            "first_anticipation_lead_ms": "" if onset is None or not pre_false else onset - int(pre_onset[0]),
            "pre_onset_false_alarm": int(pre_false),
            "target_missed": int(onset is not None and first_post is None),
        }
        run_rows.append(row)
        if onset is not None:
            latency_rows.append(row.copy())

    def rate(numerator: int, denominator: int) -> float | None:
        return None if not denominator else numerator / denominator

    run_metrics = {
        "run_count": len(traces),
        "target_runs": target_runs,
        "run_target_detected": detected_runs,
        "run_level_target_recall": rate(detected_runs, target_runs),
        "pre_onset_false_alarm_runs": pre_onset_false_runs,
        "pre_onset_run_fpr": rate(pre_onset_false_runs, len(traces)),
        "target_negative_runs": target_negative_runs,
        "target_negative_false_alarm_runs": target_negative_false_runs,
        "target_negative_run_fpr": rate(target_negative_false_runs, target_negative_runs),
        "completely_hazard_free_runs": hazard_free_runs,
        "completely_hazard_free_false_alarm_runs": hazard_free_false_runs,
        "completely_hazard_free_run_fpr": rate(hazard_free_false_runs, hazard_free_runs),
        "normal_terrain_runs": normal_runs,
        "normal_terrain_false_alarm_runs": normal_false_runs,
        "normal_terrain_run_fpr": rate(normal_false_runs, normal_runs),
        "anticipation_firing_runs": anticipation_runs,
        "anticipation_firing_count": anticipation_firings,
        **_percentiles(transition_latency, "transition_to_stable_detection"),
        **_percentiles(hazard_latency, "hazard_onset_to_stable_detection"),
        **_percentiles(anticipation_leads, "anticipation_lead"),
    }
    gate_pass = (
        run_metrics["run_level_target_recall"] is not None
        and run_metrics["run_level_target_recall"] >= 0.95
        and run_metrics["pre_onset_run_fpr"] is not None
        and run_metrics["pre_onset_run_fpr"] <= 0.05
    )
    metrics = {
        "endpoint": {
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else None,
        },
        "run_level": run_metrics,
        "gate": {
            "recall_requirement": 0.95,
            "pre_onset_run_fpr_limit": 0.05,
            "pass": bool(gate_pass),
            "classification": f"Slip Host Final Gate {'PASS' if gate_pass else 'FAIL'}",
        },
    }
    return FrozenTestResult(metrics, run_rows, latency_rows)
