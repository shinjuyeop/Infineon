"""Pure diagnostic helpers for the Walking-v2 Slip retraining failure audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from walking_v2_slip_targeted_retraining_v3 import (
    ACTIONABLE, EARLY, NORMAL, OperatingConfig, persistent_candidates,
    runtime_contact_age,
)


VARIANTS = ("V0", "V1", "V2", "V3", "V4")


@dataclass(frozen=True)
class DiagnosticCrossing:
    sample: int
    foot: int
    runtime_episode_id: int
    actionable_score: float
    normal_score: float
    early_score: float
    active_score: float
    persistence_count: int
    verifier_pass: bool
    owner_foot: int
    reset_before_update: bool


def state_crossings_variant(
    state_probability: np.ndarray,
    force_loaded: np.ndarray,
    valid_mask: np.ndarray,
    pre_fall_valid: np.ndarray,
    config: OperatingConfig,
    variant: str,
) -> tuple[list[DiagnosticCrossing], list[dict[str, object]]]:
    """Replay V0--V4 without retraining or changing an operating point."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    probability = np.asarray(state_probability, np.float32)
    loaded = np.asarray(force_loaded, bool)
    valid = np.asarray(valid_mask, bool)
    prefall = np.asarray(pre_fall_valid, bool)
    ages, episodes = runtime_contact_age(loaded)
    apply_prefall_before = variant in ("V1", "V4")
    hard_fall_reset = variant in ("V2", "V4")
    prohibit_invalid = variant in ("V3", "V4")
    candidates: list[DiagnosticCrossing] = []
    resets: list[dict[str, object]] = []
    for foot in range(2):
        action = probability[:, foot, ACTIONABLE]
        verifier = (
            (action >= config.threshold)
            & (action > probability[:, foot, NORMAL])
            & (action >= probability[:, foot, EARLY] + config.actionable_margin)
        )
        condition = loaded[:, foot] & (ages[:, foot] > 10) & verifier
        if apply_prefall_before:
            condition &= prefall
        if prohibit_invalid:
            condition &= valid[:, foot]
        # A hard fall reset breaks persistence at and after the first invalid
        # fall sample. This is distinct from merely masking reported output.
        if hard_fall_reset:
            condition &= prefall
        persisted = persistent_candidates(condition, config.persistence_ms)
        starts = np.flatnonzero(loaded[:, foot] & ~np.r_[False, loaded[:-1, foot]])
        ends = np.flatnonzero(loaded[:, foot] & ~np.r_[loaded[1:, foot], False])
        first_by_episode: dict[int, int] = {}
        for sample in np.flatnonzero(persisted):
            episode = int(episodes[sample, foot])
            first_by_episode.setdefault(episode, int(sample))
        fall_boundary = int(np.flatnonzero(~prefall)[0]) if np.any(~prefall) else None
        for episode, (start, end) in enumerate(zip(starts, ends)):
            if episode in first_by_episode:
                sample = first_by_episode[episode]
                candidates.append(DiagnosticCrossing(
                    sample=sample, foot=foot, runtime_episode_id=episode,
                    actionable_score=float(action[sample]),
                    normal_score=float(probability[sample, foot, NORMAL]),
                    early_score=float(probability[sample, foot, EARLY]),
                    active_score=float(probability[sample, foot, 3]),
                    persistence_count=config.persistence_ms, verifier_pass=bool(verifier[sample]),
                    owner_foot=foot,
                    reset_before_update=bool(fall_boundary is not None and sample >= fall_boundary
                                             and variant in ("V1", "V2", "V4")),
                ))
            resets.append({
                "foot": foot, "runtime_episode_id": episode,
                "touchdown_sample": int(start), "reset_sample": int(end) + 1,
                "fall_boundary_sample": "" if fall_boundary is None else fall_boundary,
                "fall_reset_enabled": hard_fall_reset,
                "invalid_persistence_prohibited": prohibit_invalid,
                "prefall_mask_before_update": apply_prefall_before,
                "latch_carryover": False,
            })
        if fall_boundary is not None and hard_fall_reset:
            resets.append({
                "foot": foot, "runtime_episode_id": int(episodes[fall_boundary, foot]),
                "touchdown_sample": "", "reset_sample": fall_boundary,
                "fall_boundary_sample": fall_boundary, "fall_reset_enabled": True,
                "invalid_persistence_prohibited": prohibit_invalid,
                "prefall_mask_before_update": apply_prefall_before,
                "latch_carryover": False, "reset_reason": "first_fall_boundary",
            })
    reconciled: list[DiagnosticCrossing] = []
    for sample in sorted({row.sample for row in candidates}):
        rows = [row for row in candidates if row.sample == sample]
        rows.sort(key=lambda row: (-row.actionable_score, row.foot))
        reconciled.append(rows[0])
    return reconciled, resets


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware rank AUROC without an optional ML dependency."""
    truth = np.asarray(target, bool)
    values = np.asarray(score, np.float64)
    positive = int(np.sum(truth))
    negative = len(truth) - positive
    if not positive or not negative:
        return float("nan")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((np.sum(ranks[truth]) - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(target, bool)
    if not np.any(truth):
        return float("nan")
    order = np.argsort(-np.asarray(score, np.float64), kind="stable")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered]) / np.sum(ordered))


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray, class_count: int = 4) -> float:
    recalls = []
    for state in range(class_count):
        mask = np.asarray(target) == state
        if np.any(mask):
            recalls.append(float(np.mean(np.asarray(prediction)[mask] == state)))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion(target: np.ndarray, prediction: np.ndarray, class_count: int = 4) -> np.ndarray:
    result = np.zeros((class_count, class_count), np.int64)
    for actual, predicted in zip(np.asarray(target).ravel(), np.asarray(prediction).ravel()):
        if 0 <= int(actual) < class_count:
            result[int(actual), int(predicted)] += 1
    return result


def contiguous_ranges(keys: Iterable[tuple[object, ...]]) -> list[tuple[int, int, tuple[object, ...]]]:
    values = list(keys)
    if not values:
        return []
    result = []
    start = 0
    current = values[0]
    for index in range(1, len(values)):
        if values[index] != current:
            result.append((start, index, current))
            start, current = index, values[index]
    result.append((start, len(values), current))
    return result
