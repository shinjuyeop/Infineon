"""Pure helpers for walking-v2 candidate failure localization.

The helpers operate on stored development tensors and scores.  They do not
train production candidates, access holdouts, or alter model thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from walking_v2_bilateral_bounded_training import (
    PhysicalSlipEpisode,
    SlipStateConfig,
    evaluation_invalid_firing_count,
)


AUDIT_VARIANTS = ("R0", "R1", "R2", "R3", "R4")
TOO_EARLY_CLASSES = (
    "genuinely_over_100ms_before_next_independent_physical_episode",
    "previous_physical_episode_latch_carry_over",
    "same_episode_repeated_oracle_crossing",
    "physical_episode_merging_mismatch",
    "contact_episode_assignment_mismatch",
    "opposite_foot_attribution_mismatch",
    "threshold_chatter",
    "reporting_denominator_or_counting_error",
    "other",
)


def sha256_array(values: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes for deterministic tensor identity."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_statistics(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values)
    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    selected = numeric[finite]
    return {
        "shape": json.dumps(list(array.shape)),
        "dtype": str(array.dtype),
        "minimum": float(np.min(selected)) if selected.size else "",
        "maximum": float(np.max(selected)) if selected.size else "",
        "mean": float(np.mean(selected)) if selected.size else "",
        "std": float(np.std(selected)) if selected.size else "",
        "finite_count": int(np.sum(finite)),
        "element_count": int(array.size),
        "sha256": sha256_array(array),
    }


def sample_identity(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["run_id"]), int(row["foot_index"]), int(row["endpoint_sample"])


def join_sample_ledgers(
    diagnostic: Iterable[dict[str, object]],
    candidate: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Outer join sample ledgers on run/foot/current endpoint identity."""
    left_rows = list(diagnostic)
    right_rows = list(candidate)
    left = {sample_identity(row): row for row in left_rows}
    right = {sample_identity(row): row for row in right_rows}
    left_duplicates = len(left_rows) - len(left)
    right_duplicates = len(right_rows) - len(right)
    joined: list[dict[str, object]] = []
    summary = {
        "both": 0, "diagnostic_only": 0, "t2_only": 0,
        "label_mismatch": 0, "phase_mismatch": 0, "foot_mismatch": 0,
        "timestamp_mismatch": 0, "window_mismatch": 0, "mask_mismatch": 0,
        "diagnostic_duplicates": left_duplicates, "t2_duplicates": right_duplicates,
        "missing": 0,
    }
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        left_included = bool(a is not None and a["included"])
        right_included = bool(b is not None and b["included"])
        membership = (
            "both" if left_included and right_included
            else "diagnostic_only" if left_included
            else "t2_only" if right_included
            else "missing"
        )
        summary[membership] += 1
        label_mismatch = bool(a and b and a["label"] != b["label"])
        phase_mismatch = bool(a and b and a["contact_phase"] != b["contact_phase"])
        foot_mismatch = bool(a and b and a["foot"] != b["foot"])
        timestamp_mismatch = bool(
            a and b and float(a["window_end_s"]) != float(b["window_end_s"])
        )
        window_mismatch = bool(
            a and b and float(a["window_start_s"]) != float(b["window_start_s"])
        )
        mask_mismatch = bool(a and b and bool(a["included"]) != bool(b["included"]))
        for name, value in (
            ("label_mismatch", label_mismatch), ("phase_mismatch", phase_mismatch),
            ("foot_mismatch", foot_mismatch), ("timestamp_mismatch", timestamp_mismatch),
            ("window_mismatch", window_mismatch), ("mask_mismatch", mask_mismatch),
        ):
            summary[name] += int(value)
        joined.append({
            "run_id": key[0], "foot_index": key[1], "endpoint_sample": key[2],
            "membership": membership,
            "split": (a or b)["split"], "variation": (a or b)["variation"],
            "terrain": (a or b)["terrain"], "speed_mps": (a or b)["speed_mps"],
            "foot": (a or b)["foot"],
            "diagnostic_contact_phase": "" if a is None else a["contact_phase"],
            "t2_contact_phase": "" if b is None else b["contact_phase"],
            "diagnostic_contact_episode_id": "" if a is None else a["contact_episode_id"],
            "t2_contact_episode_id": "" if b is None else b["contact_episode_id"],
            "diagnostic_touchdown": "" if a is None else a["touchdown"],
            "t2_touchdown": "" if b is None else b["touchdown"],
            "diagnostic_window_start_s": "" if a is None else a["window_start_s"],
            "t2_window_start_s": "" if b is None else b["window_start_s"],
            "diagnostic_window_end_s": "" if a is None else a["window_end_s"],
            "t2_window_end_s": "" if b is None else b["window_end_s"],
            "diagnostic_source_index": "" if a is None else a["source_index"],
            "t2_source_index": "" if b is None else b["source_index"],
            "diagnostic_label": "" if a is None else a["label"],
            "t2_label": "" if b is None else b["label"],
            "diagnostic_included": False if a is None else a["included"],
            "t2_included": False if b is None else b["included"],
            "diagnostic_mask_reason": "missing" if a is None else a["mask_reason"],
            "t2_mask_reason": "missing" if b is None else b["mask_reason"],
            "label_mismatch": label_mismatch, "phase_mismatch": phase_mismatch,
            "foot_mismatch": foot_mismatch, "timestamp_mismatch": timestamp_mismatch,
            "window_mismatch": window_mismatch, "mask_mismatch": mask_mismatch,
        })
    return joined, summary


@dataclass(frozen=True)
class SlipStateTrace:
    firing: np.ndarray
    persistence_counter: np.ndarray
    active_state: np.ndarray
    threshold_crossing: np.ndarray
    reset_reason: np.ndarray
    activation_id: np.ndarray


def replay_slip_state(
    scores: np.ndarray,
    endpoints: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    config: SlipStateConfig,
    *,
    hard_contact_reset: bool,
) -> SlipStateTrace:
    """Replay stored score-to-state logic while exposing every internal state."""
    probability = np.asarray(scores, float)
    endpoint_values = np.asarray(endpoints, int)
    loaded = np.asarray(force_loaded, bool)
    age = np.asarray(contact_age, int)
    if probability.shape != endpoint_values.shape or loaded.shape != age.shape:
        raise ValueError("state arrays must align")
    firing = np.zeros(len(probability), bool)
    counter = np.zeros(len(probability), np.int16)
    state = np.zeros(len(probability), bool)
    crossing = probability >= config.threshold
    reset = np.full(len(probability), "none", dtype="<U20")
    activation = np.full(len(probability), -1, np.int32)
    active = False
    count = 0
    previous_age = 0
    activation_value = -1
    for index, endpoint in enumerate(endpoint_values):
        eligible = bool(loaded[endpoint] and age[endpoint] > 10)
        new_touchdown = bool(age[endpoint] <= previous_age and age[endpoint] > 0)
        reset_contact = bool(not eligible or (hard_contact_reset and new_touchdown))
        if reset_contact:
            active = False
            count = 0
            activation_value = -1
            reset[index] = "contact_loss" if not loaded[endpoint] else "new_touchdown"
        elif active:
            active = bool(probability[index] >= config.exit_threshold)
            if not active:
                count = 0
                activation_value = -1
                reset[index] = "score_recovery"
        else:
            count = count + 1 if crossing[index] else 0
            if count >= config.persistence_endpoints:
                active = True
                activation_value = index
        firing[index] = active and eligible
        state[index] = active
        counter[index] = count
        if firing[index]:
            activation[index] = activation_value
        previous_age = int(age[endpoint])
    return SlipStateTrace(firing, counter, state, crossing, reset, activation)


def next_episode(
    sample: int,
    contact_episode_id: int,
    episodes: list[PhysicalSlipEpisode],
) -> PhysicalSlipEpisode | None:
    future = [
        value for value in episodes
        if value.contact_episode_id == contact_episode_id and value.start > sample
    ]
    return min(future, key=lambda value: value.start) if future else None


def classify_too_early(
    sample: int,
    activation_sample: int,
    contact_episode_id: int,
    episodes: list[PhysicalSlipEpisode],
    *,
    horizon_ms: int = 100,
) -> tuple[str, PhysicalSlipEpisode | None]:
    """Assign one exhaustive primary class to an observed too-early firing."""
    upcoming = next_episode(sample, contact_episode_id, episodes)
    if upcoming is None or sample >= upcoming.start - horizon_ms:
        return "reporting_denominator_or_counting_error", upcoming
    previous = [
        value for value in episodes
        if value.contact_episode_id == contact_episode_id
        and value.end_exclusive <= sample
    ]
    if previous and activation_sample < previous[-1].end_exclusive:
        return "previous_physical_episode_latch_carry_over", upcoming
    return "genuinely_over_100ms_before_next_independent_physical_episode", upcoming


def invalid_accounting(
    air: int,
    touchdown: int,
    post_fall: int,
) -> dict[str, int]:
    return {
        "legacy_invalid_firings": evaluation_invalid_firing_count(
            air, touchdown, post_fall, strict_first_fall_censor=False
        ),
        "strict_censor_invalid_firings": evaluation_invalid_firing_count(
            air, touchdown, post_fall, strict_first_fall_censor=True
        ),
        "post_fall_state_outputs_reported_separately": int(post_fall),
    }


def model_signature(
    coefficients: np.ndarray,
    intercept: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    for value in (coefficients, intercept, mean, scale):
        digest.update(sha256_array(value).encode("ascii"))
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()
