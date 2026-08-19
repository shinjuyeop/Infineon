"""Pure helpers for nested walking-Slip oracle calibration and blind validation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from walking_hazard_oracle_calibration_v1 import persistent_oracle


FOLDS = (
    (0, (1, 2), 0),
    (1, (0, 2), 1),
    (2, (0, 1), 2),
)
TARGET_SPEEDS = frozenset((0.10, 0.15, 0.20))
MARGIN_FRACTION_OF_BEST = 0.80
MARGIN_ABSOLUTE_BAND_M = 0.005


def slip_fire(
    trace: dict[str, np.ndarray], threshold_m: float, persistence_ms: int
) -> np.ndarray:
    """Apply the physical Slip label-oracle using its existing validity mask."""
    return persistent_oracle(
        trace["tangential_anchor_drift_m"],
        trace["slip_calibration_valid"],
        trace["contact_episode_id"],
        threshold_m,
        persistence_ms,
    )


def run_max_m(trace: dict[str, np.ndarray]) -> float | None:
    values = np.asarray(trace["tangential_anchor_drift_m"], dtype=float)
    valid = np.asarray(trace["slip_calibration_valid"], dtype=bool)
    return float(np.nanmax(values[valid])) if np.any(valid) else None


def episode_latency_rows(
    run_id: str,
    speed_mps: float,
    variation_index: int,
    trace: dict[str, np.ndarray],
    fire: np.ndarray,
    persistence_ms: int,
) -> list[dict[str, object]]:
    """Report physical-onset and persistence-completion latency per episode.

    The persistence latency is a sample-count duration, so a three-sample
    streak reports 3 ms at 1 kHz.  The physical onset latency is the elapsed
    sample-index difference from the first post-touchdown valid sample.
    """
    valid = np.asarray(trace["slip_calibration_valid"], dtype=bool)
    episodes = np.asarray(trace["contact_episode_id"], dtype=np.int64)
    output = np.asarray(fire, dtype=bool)
    rows: list[dict[str, object]] = []
    for episode in sorted(int(value) for value in np.unique(episodes[valid]) if value >= 0):
        valid_indices = np.flatnonzero(valid & (episodes == episode))
        fire_indices = np.flatnonzero(output & (episodes == episode))
        if not valid_indices.size:
            continue
        first_fire = int(fire_indices[0]) if fire_indices.size else None
        crossing = None if first_fire is None else first_fire - persistence_ms + 1
        rows.append({
            "run_id": run_id,
            "walking_speed_mps": float(speed_mps),
            "variation_index": int(variation_index),
            "contact_episode_id": episode,
            "physical_label_onset_kind": "first_post_touchdown_slip_valid_sample",
            "physical_label_onset_sample": int(valid_indices[0]),
            "model_inference_onset": "not_applicable_physical_label_oracle_only",
            "detected": first_fire is not None,
            "threshold_crossing_sample": crossing,
            "persistence_completion_sample": first_fire,
            "physical_onset_to_fire_latency_ms": (
                None if first_fire is None else first_fire - int(valid_indices[0])
            ),
            "threshold_crossing_to_persistence_completion_ms": (
                None if first_fire is None else first_fire - int(crossing) + 1
            ),
        })
    return rows


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _partition_metrics(
    indices: list[int],
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    threshold_m: float,
    persistence_ms: int,
) -> dict[str, object]:
    normal = [i for i in indices if manifests[i]["acquisition_role"] == "hard_negative"]
    positive = [i for i in indices if manifests[i]["acquisition_role"] == "slip_candidate"]
    fires = {
        i: slip_fire(traces[i], threshold_m, persistence_ms)
        for i in indices
    }
    normal_fp = [i for i in normal if np.any(fires[i])]
    normal_maxima = [run_max_m(traces[i]) for i in normal]
    positive_maxima = [run_max_m(traces[i]) for i in positive]
    normal_values = [float(value) for value in normal_maxima if value is not None]
    positive_values = [float(value) for value in positive_maxima if value is not None]
    physical_positive = [
        i for i in positive
        if bool(manifests[i]["stable_loaded_contact_pre_fall"])
        and bool(manifests[i].get("slip_physical_source_valid", True))
        and run_max_m(traces[i]) is not None
    ]
    detected = [i for i in physical_positive if np.any(fires[i])]
    detected_speeds = {float(manifests[i]["walking_speed_mps"]) for i in detected}
    latencies: list[float] = []
    for i in detected:
        rows = episode_latency_rows(
            str(manifests[i]["run_id"]),
            float(manifests[i]["walking_speed_mps"]),
            int(manifests[i]["variation_index"]),
            traces[i], fires[i], persistence_ms,
        )
        latencies.extend(
            float(row["physical_onset_to_fire_latency_ms"])
            for row in rows if row["detected"]
        )
    air = sum(int(np.count_nonzero(fires[i] & ~traces[i]["left_contact"])) for i in indices)
    post_fall = sum(
        int(np.count_nonzero(fires[i] & ~traces[i]["pre_fall_valid"])) for i in indices
    )
    transient = sum(
        int(np.count_nonzero(fires[i] & traces[i]["touchdown_transient"])) for i in indices
    )
    normal_max = max(normal_values) if normal_values else None
    positive_min = min(positive_values) if positive_values else None
    lower = None if normal_max is None else threshold_m - normal_max
    upper = None if positive_min is None else positive_min - threshold_m
    margin = None if lower is None or upper is None else min(lower, upper)
    stats = _latency_stats(latencies)
    passing = bool(
        normal and positive
        and not normal_fp
        and len(physical_positive) == len(positive)
        and len(detected) == len(positive)
        and detected_speeds == TARGET_SPEEDS
        and air == 0 and post_fall == 0 and transient == 0
    )
    return {
        "normal_runs": len(normal),
        "normal_false_positive_runs": len(normal_fp),
        "normal_false_positive_samples": sum(
            int(np.count_nonzero(fires[i])) for i in normal
        ),
        "positive_runs": len(positive),
        "physical_positive_runs": len(physical_positive),
        "detected_positive_runs": len(detected),
        "detected_speed_count": len(detected_speeds),
        "detected_speeds_mps": "|".join(f"{value:.2f}" for value in sorted(detected_speeds)),
        "air_positive_count": air,
        "post_fall_positive_count": post_fall,
        "touchdown_transient_positive_count": transient,
        "normal_envelope_max_m": normal_max,
        "positive_run_min_max_m": positive_min,
        "normal_threshold_margin_m": lower,
        "positive_threshold_margin_m": upper,
        "minimum_threshold_margin_m": margin,
        "mean_physical_onset_latency_ms": stats["mean"],
        "median_physical_onset_latency_ms": stats["median"],
        "p95_physical_onset_latency_ms": stats["p95"],
        "max_physical_onset_latency_ms": stats["max"],
        "partition_pass": passing,
    }


def nested_candidate_metrics(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    thresholds_m: tuple[float, ...],
    persistence_grid_ms: tuple[int, ...],
) -> list[dict[str, object]]:
    """Evaluate a fixed grid without accepting an outer-validation argument."""
    if len(manifests) != len(traces):
        raise ValueError("manifest/trace count mismatch")
    allowed = {"hard_negative", "slip_candidate"}
    dev_indices = [
        i for i, row in enumerate(manifests)
        if str(row["acquisition_role"]) in allowed
        and int(row["variation_index"]) in {0, 1, 2}
    ]
    rows: list[dict[str, object]] = []
    for threshold in thresholds_m:
        for persistence in persistence_grid_ms:
            for fold_id, selection_variations, validation_variation in FOLDS:
                selection = [
                    i for i in dev_indices
                    if int(manifests[i]["variation_index"]) in selection_variations
                ]
                validation = [
                    i for i in dev_indices
                    if int(manifests[i]["variation_index"]) == validation_variation
                ]
                select_metrics = _partition_metrics(
                    selection, manifests, traces, float(threshold), int(persistence)
                )
                validation_metrics = _partition_metrics(
                    validation, manifests, traces, float(threshold), int(persistence)
                )
                fold_pass = bool(
                    select_metrics["partition_pass"]
                    and validation_metrics["partition_pass"]
                )
                rows.append({
                    "threshold_m": float(threshold),
                    "persistence_ms": int(persistence),
                    "fold_id": fold_id,
                    "selection_variations": "|".join(map(str, selection_variations)),
                    "internal_validation_variation": validation_variation,
                    **{f"selection_{key}": value for key, value in select_metrics.items()},
                    **{f"internal_validation_{key}": value for key, value in validation_metrics.items()},
                    "fold_pass": fold_pass,
                    "all_folds_pass": False,
                    "sufficient_margin": False,
                    "selection_score": "",
                    "tie_break_rank": "",
                    "selected": False,
                })
    _annotate_all_fold_status(rows)
    return rows


def _annotate_all_fold_status(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["threshold_m"]), int(row["persistence_ms"]))].append(row)
    for group in grouped.values():
        passed = len(group) == len(FOLDS) and all(bool(row["fold_pass"]) for row in group)
        for row in group:
            row["all_folds_pass"] = passed


def select_nested_candidate(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Select using hard gates, sufficient robustness margin, then latency.

    Margin is not an unconditional lexicographic winner: every all-fold-pass
    candidate within 80% of the best worst-fold margin and within 5 mm of it
    enters the sufficient-margin set.  Latency then decides, followed by lower
    persistence and threshold only as deterministic ties.
    """
    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["threshold_m"]), int(row["persistence_ms"]))].append(row)
    eligible = []
    for key, group in grouped.items():
        if len(group) != len(FOLDS) or not all(bool(row["fold_pass"]) for row in group):
            continue
        margins = [
            float(row[name])
            for row in group
            for name in (
                "selection_minimum_threshold_margin_m",
                "internal_validation_minimum_threshold_margin_m",
            )
        ]
        p95_values = [
            float(row[name])
            for row in group
            for name in (
                "selection_p95_physical_onset_latency_ms",
                "internal_validation_p95_physical_onset_latency_ms",
            )
            if row[name] is not None
        ]
        mean_values = [
            float(row[name])
            for row in group
            for name in (
                "selection_mean_physical_onset_latency_ms",
                "internal_validation_mean_physical_onset_latency_ms",
            )
            if row[name] is not None
        ]
        eligible.append({
            "threshold_m": key[0],
            "persistence_ms": key[1],
            "worst_fold_minimum_margin_m": min(margins),
            "worst_fold_p95_physical_onset_latency_ms": max(p95_values),
            "mean_fold_physical_onset_latency_ms": float(np.mean(mean_values)),
            "folds_passed": len(group),
        })
    if not eligible:
        return None
    best_margin = max(float(item["worst_fold_minimum_margin_m"]) for item in eligible)
    sufficient_floor = max(
        MARGIN_FRACTION_OF_BEST * best_margin,
        best_margin - MARGIN_ABSOLUTE_BAND_M,
    )
    sufficient = [
        item for item in eligible
        if float(item["worst_fold_minimum_margin_m"]) + 1e-15 >= sufficient_floor
    ]
    ordered = sorted(
        sufficient,
        key=lambda item: (
            float(item["worst_fold_p95_physical_onset_latency_ms"]),
            float(item["mean_fold_physical_onset_latency_ms"]),
            int(item["persistence_ms"]),
            float(item["threshold_m"]),
        ),
    )
    selected = dict(ordered[0])
    selected.update({
        "all_folds_pass": True,
        "best_worst_fold_margin_m": best_margin,
        "sufficient_margin_floor_m": sufficient_floor,
        "selection_rule": (
            "mandatory zero normal FP + all physical positives/all speeds + zero "
            "AIR/post-fall/touchdown violations in every selection and internal-validation "
            "partition; retain candidates within both 80% and 5 mm of best worst-fold "
            "margin; minimize worst-fold p95 then mean physical-onset latency; lower "
            "persistence then threshold are ties only"
        ),
        "selected_without_outer_data": True,
    })
    for rank, item in enumerate(ordered, start=1):
        key = (float(item["threshold_m"]), int(item["persistence_ms"]))
        for row in grouped[key]:
            row["sufficient_margin"] = True
            row["selection_score"] = (
                f"margin={item['worst_fold_minimum_margin_m']:.12g};"
                f"worst_p95_ms={item['worst_fold_p95_physical_onset_latency_ms']:.12g};"
                f"mean_ms={item['mean_fold_physical_onset_latency_ms']:.12g}"
            )
            row["tie_break_rank"] = rank
            row["selected"] = rank == 1
    return selected


def development_fold_manifest(
    manifests: list[dict[str, object]],
) -> dict[str, object]:
    runs = []
    for fold_id, selection_variations, validation_variation in FOLDS:
        for row in manifests:
            variation = int(row["variation_index"])
            if str(row["acquisition_role"]) not in {"hard_negative", "slip_candidate"}:
                continue
            if variation not in {*selection_variations, validation_variation}:
                continue
            runs.append({
                "fold_id": fold_id,
                "run_id": row["run_id"],
                "variation_index": variation,
                "fold_role": (
                    "selection" if variation in selection_variations else "internal_validation"
                ),
                "ownership": "whole run and all contact episodes",
            })
    keys: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in runs:
        keys[(str(row["run_id"]), int(row["fold_id"]))].add(str(row["fold_role"]))
    leakage = sum(len(owners) != 1 for owners in keys.values())
    return {
        "policy": "leave one complete variation out; whole runs and episodes remain together",
        "folds": [
            {
                "fold_id": fold_id,
                "selection_variations": list(selection),
                "internal_validation_variation": validation,
            }
            for fold_id, selection, validation in FOLDS
        ],
        "test_or_final_used": False,
        "run_fold_role_leakage_count": leakage,
        "runs": runs,
    }
