"""Pure causal-model and runtime contracts for targeted Slip retraining v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import chain
from typing import Iterable

import numpy as np


SAMPLE_RATE_HZ = 1000
HORIZON_MS = 100
MAX_HISTORY_MS = 200
TOUCHDOWN_GUARD_MS = 10
SEEDS = (202608231, 202608232, 202608233)
STATE_NAMES = (
    "NORMAL_NO_EVENT", "EARLY_PRECURSOR", "ACTIONABLE_RISK",
    "PHYSICAL_ACTIVE_EVIDENCE",
)
NORMAL, EARLY, ACTIONABLE, ACTIVE = range(4)

# Ceilings are bounded by the largest verified E84-compatible v2 probe and
# its verified Fusion20 history allocation.  No Vela execution occurs here.
RESOURCE_CEILINGS = {
    "parameter_count": 2657,
    "macs_per_1khz_tick": 18657,
    "history_memory_bytes": 49600,
    "persistent_state_bytes": 1024,
}

THRESHOLD_GRID = (0.70, 0.85, 0.95, 0.99)
PERSISTENCE_GRID_MS = (3, 5)
MARGIN_GRID = (0.00, 0.10)
HYSTERESIS_GRID = (0.05,)
HARD_NEGATIVE_RATIO = 1.0
HARD_NEGATIVE_MULTIPLIER = 4.0
HARD_NEGATIVE_PER_RUN_FOOT_STATE = 32


@dataclass(frozen=True)
class CandidateFamily:
    candidate_id: str
    name: str
    history_ms: tuple[int, ...]
    hidden_width: int
    output_contract: str
    feature_contract: str
    activation: str = "relu"
    optimizer: str = "episode_weighted_diagonal_gaussian_closed_form"
    learning_rate: str = "not_applicable_closed_form"
    epochs: int = 1
    early_stopping: bool = False
    batch_construction: str = "complete eligible population; streaming sufficient statistics"
    loss_terms: tuple[str, ...] = (
        "balanced four-state negative log likelihood",
        "actionable_over_normal ordering",
        "actionable_over_early margin",
        "correct_foot_over_contralateral ordering",
    )
    loss_weights: tuple[float, ...] = (1.0, 0.25, 0.25, 0.25)

    @property
    def contract(self) -> dict[str, object]:
        return asdict(self)


FAMILIES = (
    CandidateFamily(
        "R1", "targeted_S4_C_reproduction", (50,), 12,
        "two-stage actionable proposal plus causal normal/early verifier and affected-foot comparator",
        "current bilateral Fusion20, 50 ms deltas/statistics, causal contact age"),
    CandidateFamily(
        "R2", "dual_timescale_horizon_verifier", (50, 200), 16,
        "shared per-foot four-state horizon head plus affected-foot localization head",
        "separate ipsilateral/contralateral 50 ms and 200 ms summaries"),
    CandidateFamily(
        "R3", "discrete_hazard_foot_localization", (25, 50, 100, 200), 20,
        "four preregistered 25 ms hazard bins, farther/no-event, active evidence, foot head",
        "causal maximum-200 ms hazard summaries and contact-scoped state"),
)


@dataclass(frozen=True)
class OperatingConfig:
    threshold: float
    persistence_ms: int
    actionable_margin: float
    hysteresis: float

    @property
    def config_id(self) -> str:
        return (
            f"t{self.threshold:.2f}_p{self.persistence_ms}_"
            f"m{self.actionable_margin:.2f}_h{self.hysteresis:.2f}")


def operating_grid() -> tuple[OperatingConfig, ...]:
    return tuple(
        OperatingConfig(threshold, persistence, margin, hysteresis)
        for threshold in THRESHOLD_GRID
        for persistence in PERSISTENCE_GRID_MS
        for margin in MARGIN_GRID
        for hysteresis in HYSTERESIS_GRID)


def candidate_matrix_payload() -> dict[str, object]:
    candidates = [{
        "candidate_id": f"{family.candidate_id}_seed_{seed}",
        "family": family.contract, "seed": seed,
    } for family in FAMILIES for seed in SEEDS]
    return {
        "version": "walking_v2_slip_retrain_candidate_matrix_v3",
        "frozen_before_first_training_job": True,
        "family_count": 3, "seed_count_per_family": 3,
        "candidate_count": 9, "fixed_seeds": list(SEEDS),
        "runtime_inputs": [
            "bilateral_canonical_Fusion20", "causal_FSR_contact_state",
            "causal_contact_age", "commanded_speed"],
        "privileged_runtime_inputs": [],
        "offline_label_only": [
            "physical_contact", "contact_episode_id", "future_physical_onset",
            "touchdown_transient", "first_fall_censor", "physical_oracle_active"],
        "normalization": "inner-training-only mean/std; epsilon 1e-6; clip +/-8",
        "projection_initialization": "fixed seeded Gaussian/sqrt(input width)",
        "hard_negative_ratio": HARD_NEGATIVE_RATIO,
        "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
        "hard_negative_per_run_foot_state": HARD_NEGATIVE_PER_RUN_FOOT_STATE,
        "threshold_grid": list(THRESHOLD_GRID),
        "persistence_grid_ms": list(PERSISTENCE_GRID_MS),
        "actionable_margin_grid": list(MARGIN_GRID),
        "hysteresis_grid": list(HYSTERESIS_GRID),
        "resource_ceilings": RESOURCE_CEILINGS,
        "candidates": candidates,
        "fourth_family_allowed": False,
        "post_result_architecture_change_allowed": False,
    }


def selection_policy_payload() -> dict[str, object]:
    return {
        "version": "walking_v2_slip_retrain_selection_policy_v3",
        "frozen_before_first_training_job": True,
        "mandatory_pooled_gates": {
            "actionable_episode_recall_min": 0.80,
            "every_speed_actionable_episode_recall_min": 0.70,
            "affected_foot_accuracy_min": 0.90,
            "normal_run_fp_max": 0,
            "normal_contact_episode_fp_max": 0,
            "too_early_activation_max": 0,
            "air_touchdown_invalid_postfall_latch_crossfoot_max_each": 0,
            "future_leakage_max": 0,
            "resource_gates": True,
        },
        "mandatory_fold_zero_safety_gates": [
            "normal_run_fp", "normal_contact_episode_fp", "too_early_activation",
            "air_firing", "touchdown_transient_firing", "invalid_firing",
            "postfall_firing", "latch_carryover", "cross_foot_ownership_violation"],
        "candidate_ranking": [
            "highest worst-speed actionable episode recall",
            "highest pooled actionable episode recall",
            "highest affected-foot accuracy", "lowest p95 late latency",
            "highest minimum fold recall", "lowest MAC count",
            "lowest parameter count", "lexical candidate ID", "lowest seed"],
        "operating_config_ranking": [
            "all inner-training zero-safety gates", "highest inner episode recall",
            "highest inner worst-speed recall", "highest affected-foot accuracy",
            "lowest p95 late latency", "highest threshold", "highest persistence",
            "highest actionable margin", "lexical config ID"],
        "diagnostic_config_ranking_if_none_safe": [
            "lowest summed safety violations", "highest threshold",
            "highest persistence", "highest actionable margin", "lexical config ID"],
        "final_config_reconciliation": (
            "most frequent outer-fold-selected config; tie by conservative "
            "threshold, persistence, margin, then lexical config ID"),
        "no_passing_candidate_action": {
            "production_model": False, "normalization": False,
            "selection_lock": False, "diagnostic_fallback_count": 1,
            "fresh_blind_holdout_authorized": False},
        "run_coverage_primary_metric": False,
        "post_result_threshold_search_allowed": False,
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def runtime_contact_age(loaded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive causal per-foot episode IDs and ages from force-loaded state."""
    values = np.asarray(loaded, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("loaded must have shape [time,2]")
    ages = np.zeros(values.shape, np.int32)
    episodes = np.full(values.shape, -1, np.int32)
    index = np.arange(1, len(values) + 1, dtype=np.int32)
    for foot in range(2):
        active = values[:, foot]
        before = np.r_[False, active[:-1]]
        touchdown = active & ~before
        last_unloaded = np.maximum.accumulate(np.where(~active, index, 0))
        ages[:, foot] = np.where(active, index - last_unloaded, 0)
        episodes[:, foot] = np.where(active, np.cumsum(touchdown) - 1, -1)
    return ages, episodes


def corrected_r4_labels(
    physical_episode_id: np.ndarray, physical_active: np.ndarray,
    physical_valid: np.ndarray, pre_fall_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Create offline four-state labels within foot/contact/fall ownership."""
    episodes = np.asarray(physical_episode_id, np.int32)
    active = np.asarray(physical_active, bool)
    valid = np.asarray(physical_valid, bool) & np.asarray(pre_fall_valid, bool)[:, None]
    if episodes.shape != active.shape or episodes.shape != valid.shape:
        raise ValueError("R4 arrays must share [time,2] shape")
    labels = np.full(episodes.shape, -1, np.int8)
    delta = np.full(episodes.shape, -1, np.int32)
    event_rows: list[dict[str, int]] = []
    for foot in range(2):
        for episode in sorted(set(episodes[:, foot].tolist()) - {-1}):
            mask = (episodes[:, foot] == episode) & valid[:, foot]
            samples = np.flatnonzero(mask)
            if not samples.size:
                continue
            onset_candidates = np.flatnonzero(mask & active[:, foot])
            labels[samples, foot] = NORMAL
            if not onset_candidates.size:
                continue
            onset = int(onset_candidates[0])
            event_rows.append({"foot": foot, "episode_id": episode, "onset_sample": onset})
            before = samples[samples < onset]
            distances = onset - before
            delta[before, foot] = distances
            labels[before[distances > HORIZON_MS], foot] = EARLY
            labels[before[(distances >= 1) & (distances <= HORIZON_MS)], foot] = ACTIONABLE
            active_samples = samples[active[samples, foot]]
            labels[active_samples, foot] = ACTIVE
            delta[active_samples, foot] = 0
    return labels, delta, event_rows


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, np.float32)
    prefix = np.vstack((np.zeros((1, array.shape[1]), np.float64),
                        np.cumsum(array, axis=0, dtype=np.float64)))
    indices = np.arange(len(array))
    starts = np.maximum(0, indices + 1 - width)
    counts = (indices + 1 - starts)[:, None]
    return ((prefix[indices + 1] - prefix[starts]) / counts).astype(np.float32)


def _rolling_std(values: np.ndarray, width: int) -> np.ndarray:
    mean = _rolling_mean(values, width)
    second = _rolling_mean(np.asarray(values, np.float32) ** 2, width)
    return np.sqrt(np.maximum(0.0, second - mean * mean)).astype(np.float32)


def _delta(values: np.ndarray, lag: int) -> np.ndarray:
    array = np.asarray(values, np.float32)
    prior = np.maximum(0, np.arange(len(array)) - lag)
    return array - array[prior]


def causal_feature_matrix(
    bilateral_canonical: np.ndarray, force_loaded: np.ndarray,
    speed_mps: float, family_id: str,
) -> np.ndarray:
    """Return [time,foot,feature] using current and past runtime observables only."""
    canonical = np.asarray(bilateral_canonical, np.float32)
    loaded = np.asarray(force_loaded, bool)
    if canonical.ndim != 2 or canonical.shape[1] != 20:
        raise ValueError("Fusion20 input must have shape [time,20]")
    if loaded.shape != (len(canonical), 2):
        raise ValueError("force_loaded input must have shape [time,2]")
    ages, _ = runtime_contact_age(loaded)
    feet = (canonical[:, :10], canonical[:, 10:20])
    result: list[np.ndarray] = []
    for foot in range(2):
        ipsi, contra = feet[foot], feet[1 - foot]
        pair = np.concatenate((ipsi, contra), axis=1)
        aggregates = np.column_stack((
            np.sum(ipsi[:, :4], axis=1), np.linalg.norm(ipsi[:, 4:7], axis=1),
            np.linalg.norm(ipsi[:, 7:10], axis=1),
            np.sum(contra[:, :4], axis=1), np.linalg.norm(contra[:, 4:7], axis=1),
            np.linalg.norm(contra[:, 7:10], axis=1),
        )).astype(np.float32)
        age = np.column_stack((
            np.minimum(ages[:, foot], MAX_HISTORY_MS) / MAX_HISTORY_MS,
            np.minimum(ages[:, 1 - foot], MAX_HISTORY_MS) / MAX_HISTORY_MS,
            np.full(len(canonical), speed_mps / 0.20, np.float32),
        )).astype(np.float32)
        if family_id == "R1":
            parts = (pair, _delta(pair, 50), _rolling_mean(aggregates, 50),
                     _rolling_std(aggregates, 50), age)
        elif family_id == "R2":
            parts = (pair, _delta(pair, 50), _delta(pair, 200),
                     _rolling_mean(aggregates, 50), _rolling_mean(aggregates, 200),
                     _rolling_std(aggregates, 50), _rolling_std(aggregates, 200), age)
        elif family_id == "R3":
            parts = (pair, *(_delta(aggregates, lag) for lag in (25, 50, 100, 200)),
                     _rolling_mean(aggregates, 50), _rolling_mean(aggregates, 200), age)
        else:
            raise ValueError(family_id)
        result.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.stack(result, axis=1)


def r3_training_targets(labels: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Expand actionable samples into four frozen discrete hazard bins."""
    target = np.asarray(labels, np.int8).copy()
    # 0 normal, 1 farther/early, 2..5 hazard bins, 6 active.
    expanded = np.full(target.shape, -1, np.int8)
    expanded[target == NORMAL] = 0
    expanded[target == EARLY] = 1
    expanded[target == ACTIVE] = 6
    actionable = target == ACTIONABLE
    bins = np.clip((np.maximum(1, delta) - 1) // 25, 0, 3)
    expanded[actionable] = (5 - bins[actionable]).astype(np.int8)
    return expanded


@dataclass
class Normalization:
    mean: np.ndarray
    scale: np.ndarray
    fit_run_ids: tuple[str, ...]
    fit_folds: tuple[int, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -8.0, 8.0).astype(np.float32)


@dataclass
class CausalRiskModel:
    family_id: str
    seed: int
    projection: np.ndarray
    projection_bias: np.ndarray
    class_mean: np.ndarray
    class_variance: np.ndarray
    class_log_prior: np.ndarray

    @property
    def class_count(self) -> int:
        return int(self.class_mean.shape[0])

    def hidden(self, normalized: np.ndarray) -> np.ndarray:
        return np.maximum(
            0.0, normalized @ self.projection + self.projection_bias).astype(np.float32)

    def raw_probabilities(self, normalized: np.ndarray) -> np.ndarray:
        hidden = self.hidden(normalized)
        difference = hidden[:, :, None, :] - self.class_mean[None, None, :, :]
        logit = -0.5 * np.sum(
            np.log(self.class_variance)[None, None, :, :]
            + difference * difference / self.class_variance[None, None, :, :], axis=3)
        logit += self.class_log_prior
        logit -= np.max(logit, axis=2, keepdims=True)
        probability = np.exp(logit)
        return (probability / np.sum(probability, axis=2, keepdims=True)).astype(np.float32)

    def state_probabilities(self, normalized: np.ndarray) -> np.ndarray:
        raw = self.raw_probabilities(normalized)
        if self.family_id != "R3":
            return raw
        state = np.zeros(raw.shape[:2] + (4,), np.float32)
        state[:, :, NORMAL] = raw[:, :, 0]
        state[:, :, EARLY] = raw[:, :, 1]
        state[:, :, ACTIONABLE] = np.sum(raw[:, :, 2:6], axis=2)
        state[:, :, ACTIVE] = raw[:, :, 6]
        return state


def projection_for(family_id: str, seed: int, input_width: int) -> tuple[np.ndarray, np.ndarray]:
    family = next(row for row in FAMILIES if row.candidate_id == family_id)
    generator = np.random.default_rng(seed)
    projection = generator.normal(
        0.0, 1.0 / np.sqrt(input_width), (input_width, family.hidden_width)).astype(np.float32)
    bias = generator.normal(0.0, 0.02, family.hidden_width).astype(np.float32)
    return projection, bias


def fit_gaussian_head(
    family_id: str, seed: int, normalized_batches: Iterable[
        tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> CausalRiskModel:
    """Fit all classes from complete weighted batches via sufficient statistics."""
    iterator = iter(normalized_batches)
    first = next(iterator, None)
    if first is None:
        raise ValueError("no training batches")
    width = int(first[0].shape[2])
    projection, bias = projection_for(family_id, seed, width)
    class_count = 7 if family_id == "R3" else 4
    hidden_width = projection.shape[1]
    mass = np.zeros(class_count, np.float64)
    sums = np.zeros((class_count, hidden_width), np.float64)
    squares = np.zeros((class_count, hidden_width), np.float64)
    for features, targets, weights in chain((first,), iterator):
        hidden = np.maximum(0.0, features @ projection + bias).astype(np.float32)
        for state in range(class_count):
            mask = targets == state
            if not np.any(mask):
                continue
            values = hidden[mask]
            selected_weight = weights[mask].astype(np.float64)
            mass[state] += np.sum(selected_weight)
            sums[state] += np.sum(values * selected_weight[:, None], axis=0)
            squares[state] += np.sum(values * values * selected_weight[:, None], axis=0)
    if np.any(mass <= 0):
        raise ValueError(f"missing training class mass: {mass}")
    mean = sums / mass[:, None]
    variance = np.maximum(1e-3, squares / mass[:, None] - mean * mean)
    prior = np.full(class_count, -np.log(class_count), np.float64)
    return CausalRiskModel(
        family_id, seed, projection, bias,
        mean.astype(np.float32), variance.astype(np.float32), prior.astype(np.float32))


def model_resource_cost(family_id: str, input_width: int) -> dict[str, object]:
    family = next(row for row in FAMILIES if row.candidate_id == family_id)
    classes = 7 if family_id == "R3" else 4
    width = family.hidden_width
    parameters = input_width * width + width + classes * width * 2 + classes + 2 * width + 2
    macs = 2 * (input_width * width + classes * width + 2 * width)
    history_bytes = MAX_HISTORY_MS * 20 * 4
    persistent = 2 * (4 * 4 + 7 * 4 + 4 * 4)
    result = {
        "family_id": family_id, "input_feature_count": input_width,
        "hidden_width": width, "output_class_count": classes,
        "parameter_count": parameters, "model_bytes_float32": parameters * 4,
        "history_memory_bytes": history_bytes,
        "persistent_state_bytes": persistent,
        "macs_per_1khz_tick": macs,
        "estimated_operations_per_second": macs * SAMPLE_RATE_HZ,
        "expected_operator_set": [
            "FULLY_CONNECTED", "RELU", "SUB", "MUL", "ADD", "SOFTMAX",
            "GREATER", "ARGMAX", "stateful reset/comparison"],
        "unsupported_operator_risk": "LOW_UNCOMPILED_REQUIRES_FUTURE_VELA_CONFIRMATION",
    }
    result["resource_gates_pass"] = bool(
        parameters <= RESOURCE_CEILINGS["parameter_count"]
        and macs <= RESOURCE_CEILINGS["macs_per_1khz_tick"]
        and history_bytes <= RESOURCE_CEILINGS["history_memory_bytes"]
        and persistent <= RESOURCE_CEILINGS["persistent_state_bytes"])
    return result


def persistent_candidates(condition: np.ndarray, persistence_ms: int) -> np.ndarray:
    """Vectorized causal persistence: true only after N consecutive samples."""
    values = np.asarray(condition, bool)
    prefix = np.r_[0, np.cumsum(values.astype(np.int32))]
    result = np.zeros(len(values), bool)
    if persistence_ms <= len(values):
        indices = np.arange(persistence_ms - 1, len(values))
        count = prefix[indices + 1] - prefix[indices + 1 - persistence_ms]
        result[indices] = count == persistence_ms
    return result


def runtime_crossings(
    state_probability: np.ndarray, force_loaded: np.ndarray,
    config: OperatingConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply independent contact-scoped state and deterministic two-foot ownership."""
    probability = np.asarray(state_probability, np.float32)
    loaded = np.asarray(force_loaded, bool)
    ages, episodes = runtime_contact_age(loaded)
    candidates: list[dict[str, object]] = []
    resets: list[dict[str, object]] = []
    for foot in range(2):
        action = probability[:, foot, ACTIONABLE]
        condition = (
            loaded[:, foot] & (ages[:, foot] > TOUCHDOWN_GUARD_MS)
            & (action >= config.threshold)
            & (action > probability[:, foot, NORMAL])
            & (action >= probability[:, foot, EARLY] + config.actionable_margin))
        persisted = persistent_candidates(condition, config.persistence_ms)
        first_by_episode: dict[int, int] = {}
        for sample in np.flatnonzero(persisted):
            first_by_episode.setdefault(int(episodes[sample, foot]), int(sample))
        active = loaded[:, foot]
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        for episode, (start, end) in enumerate(zip(starts, ends)):
            if episode in first_by_episode:
                sample = first_by_episode[episode]
                candidates.append({
                    "sample": sample, "foot": foot,
                    "runtime_episode_id": episode,
                    "actionable_score": float(action[sample]),
                    "normal_score": float(probability[sample, foot, NORMAL]),
                    "early_score": float(probability[sample, foot, EARLY]),
                    "active_score": float(probability[sample, foot, ACTIVE]),
                    "persistence": config.persistence_ms,
                    "owner_foot": foot,
                })
            resets.append({
                "foot": foot, "runtime_episode_id": episode,
                "touchdown_sample": int(start), "reset_sample": int(end) + 1,
                "reset_reason": "contact_loss_or_trace_end",
                "latch_carryover": False,
            })
    # At the same sample exactly one foot owns the crossing: higher actionable
    # score, then left (index 0).  Both feet are never counted correct.
    reconciled: list[dict[str, object]] = []
    for sample in sorted({int(row["sample"]) for row in candidates}):
        rows = [row for row in candidates if int(row["sample"]) == sample]
        rows.sort(key=lambda row: (-float(row["actionable_score"]), int(row["foot"])))
        selected = dict(rows[0])
        selected["simultaneous_candidate_count"] = len(rows)
        selected["reconciliation_rule"] = "highest_actionable_score_then_left"
        reconciled.append(selected)
    return reconciled, resets
