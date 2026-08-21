"""Pure helpers for corrected Walking-v2 targeted Slip retraining v4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from walking_hazard_ground_truth_v1 import gait_phase
from walking_v2_joint_terrain_slip_redesign_v1 import (
    causal_endpoints, episode_balanced_weights, raking_weights, runtime_feature,
)
from walking_v2_slip_redesign_iteration_v2 import (
    ACTIONABLE_RISK, EARLY_PRECURSOR, NORMAL_NO_EVENT,
    PHYSICAL_ACTIVE_EVIDENCE, RuntimeStateConfig, SlipV2Model,
    fit_slip_v2_model,
)
from run_walking_v2_slip_targeted_retraining_v3 import RunRecord
from walking_v2_slip_targeted_retraining_v3 import runtime_contact_age


VARIANTS = ("F0", "F1")
SEEDS = (202608241, 202608242, 202608243)
STATE_NAMES = (
    "NORMAL_NO_EVENT", "EARLY_PRECURSOR", "ACTIONABLE_RISK",
    "PHYSICAL_ACTIVE_EVIDENCE",
)
ENDPOINTS = causal_endpoints(3000, 200)
PHASE_CODE = {"AIR": 0, "TOUCHDOWN": 1, "LOADING": 2, "MID_STANCE": 3, "PUSH_OFF": 4}
HEAD_NAMES = ("normal", "early", "actionable", "active", "foot", "proposal")
RESOURCE_CEILINGS = {
    "parameter_count": 30_000, "macs_per_tick": 60_000,
    "history_bytes": 32 * 1024, "persistent_state_bytes": 4 * 1024,
}


@dataclass(frozen=True)
class OperatingConfig:
    state_threshold: float
    proposal_threshold: float
    early_margin: float
    normal_margin: float
    foot_threshold: float
    persistence_endpoints: int
    hysteresis: float

    @property
    def config_id(self) -> str:
        return (
            f"s{self.state_threshold:.2f}_p{self.proposal_threshold:.2f}"
            f"_e{self.early_margin:.2f}_n{self.normal_margin:.2f}"
            f"_f{self.foot_threshold:.2f}_k{self.persistence_endpoints}"
            f"_h{self.hysteresis:.2f}"
        )

    def as_runtime_config(self) -> RuntimeStateConfig:
        return RuntimeStateConfig(
            self.state_threshold, self.proposal_threshold, self.early_margin,
            self.normal_margin, self.foot_threshold, self.persistence_endpoints,
            self.hysteresis, True,
        )


def operating_grid() -> tuple[OperatingConfig, ...]:
    """Return the preregistered grid; it includes the exact prior S4-C point."""
    threshold_pairs = (
        (0.45, 0.45), (0.55, 0.55), (0.65, 0.65), (0.75, 0.75),
        (0.85, 0.85), (0.65, 0.55), (0.55, 0.65),
    )
    margin_pairs = ((0.15, 0.10), (0.25, 0.20))
    rows = [
        OperatingConfig(state, proposal, early, normal, 0.60, persistence, hysteresis)
        for state, proposal in threshold_pairs
        for early, normal in margin_pairs
        for persistence in (1, 2, 3)
        for hysteresis in (0.0, 0.05)
    ]
    if OperatingConfig(0.65, 0.65, 0.25, 0.20, 0.60, 2, 0.05) not in rows:
        raise AssertionError("prior S4-C operating point absent")
    return tuple(rows)


@dataclass
class TrainingRows:
    features: np.ndarray
    state: np.ndarray
    foot_target: np.ndarray
    balance_unit: np.ndarray
    identity: np.ndarray
    run_index: np.ndarray
    run_id: np.ndarray
    fold: np.ndarray
    endpoint: np.ndarray
    foot: np.ndarray
    phase: np.ndarray
    speed: np.ndarray
    context: np.ndarray
    variation: np.ndarray
    contact_episode: np.ndarray
    wrong_foot_context: np.ndarray
    eligible: np.ndarray


def derived_runtime_telemetry(loaded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct old S4-C contact age/phase from causal force-loaded state."""
    values = np.asarray(loaded, bool)
    ages, episodes = runtime_contact_age(values)
    phase = np.zeros(values.shape, np.int8)
    for foot in range(2):
        phase[:, foot] = np.asarray(
            [PHASE_CODE[value] for value in gait_phase(values[:, foot])], np.int8)
    return ages, episodes, phase


def build_training_rows(
    records: list[RunRecord], metadata_by_id: dict[str, dict[str, Any]],
) -> TrainingRows:
    """Build the exact S3 200-ms causal feature rows on frozen R4 labels."""
    feature_rows: list[np.ndarray] = []
    fields: dict[str, list[Any]] = {key: [] for key in (
        "state", "foot_target", "balance_unit", "identity", "run_index", "run_id",
        "fold", "endpoint", "foot", "phase", "speed", "context", "variation",
        "contact_episode", "wrong_foot_context", "eligible",
    )}
    for run_index, record in enumerate(records):
        metadata = metadata_by_id[record.run_id]
        ages, _, phases = derived_runtime_telemetry(record.loaded)
        event_by_owner = {
            (int(row["foot"]), int(row["episode_id"])): int(row["onset_sample"])
            for row in record.events
        }
        for endpoint in ENDPOINTS:
            endpoint_labels = record.labels[endpoint]
            for foot in range(2):
                state = int(endpoint_labels[foot])
                contact = int(record.physical_episode[endpoint, foot])
                onset = event_by_owner.get((foot, contact))
                event_owned = state in (EARLY_PRECURSOR, ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE)
                balance = (
                    f"event:{record.run_id}:{foot}:{contact}:{onset}"
                    if event_owned and onset is not None
                    else f"contact:{record.run_id}:{foot}:{contact}"
                )
                feature_rows.append(runtime_feature(
                    "S3", foot, int(endpoint), record.canonical, record.loaded, ages, phases,
                ))
                values = {
                    "state": state,
                    "foot_target": int(state in (ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE)),
                    "balance_unit": balance,
                    "identity": f"{record.run_id}:{endpoint}:{foot}",
                    "run_index": run_index, "run_id": record.run_id, "fold": record.fold,
                    "endpoint": int(endpoint), "foot": foot, "phase": int(phases[endpoint, foot]),
                    "speed": record.speed, "context": metadata["terrain_context"],
                    "variation": metadata["variation_group"], "contact_episode": contact,
                    "wrong_foot_context": bool(
                        state in (NORMAL_NO_EVENT, EARLY_PRECURSOR)
                        and int(endpoint_labels[1 - foot]) in (
                            ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE)),
                    "eligible": bool(
                        state >= 0 and record.loaded[endpoint, foot]
                        and ages[endpoint, foot] > 10
                        and not record.touchdown[endpoint, foot]
                        and record.valid[endpoint, foot] and record.prefall[endpoint]),
                }
                for key, value in values.items():
                    fields[key].append(value)
    return TrainingRows(
        np.asarray(feature_rows, np.float32), np.asarray(fields["state"], np.int8),
        np.asarray(fields["foot_target"], np.int8), np.asarray(fields["balance_unit"]),
        np.asarray(fields["identity"]), np.asarray(fields["run_index"], np.int32),
        np.asarray(fields["run_id"]), np.asarray(fields["fold"], np.int8),
        np.asarray(fields["endpoint"], np.int32), np.asarray(fields["foot"], np.int8),
        np.asarray(fields["phase"], np.int8), np.asarray(fields["speed"], np.float32),
        np.asarray(fields["context"]), np.asarray(fields["variation"]),
        np.asarray(fields["contact_episode"], np.int32),
        np.asarray(fields["wrong_foot_context"], bool), np.asarray(fields["eligible"], bool),
    )


def _targeted_cell_weights(
    target: np.ndarray, run_id: np.ndarray, foot: np.ndarray, contact: np.ndarray,
) -> np.ndarray:
    weight = np.zeros(len(target), np.float64)
    cells = np.asarray([
        f"{run_id[index]}:{int(foot[index])}:{int(contact[index])}:{int(target[index])}"
        for index in range(len(target))
    ])
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    weight *= len(weight) / weight.sum()
    return weight


def training_weights(
    rows: TrainingRows, indices: np.ndarray, variant: str, hard_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    state = rows.state[indices]
    if np.any(state < 0):
        raise ValueError("invalid row entered training")
    if variant == "F0":
        base = episode_balanced_weights(state, rows.balance_unit[indices])
        rake = raking_weights(np.column_stack((
            state, rows.context[indices], np.rint(rows.speed[indices] * 100).astype(int),
            rows.phase[indices], rows.foot[indices], rows.run_index[indices],
            rows.variation[indices],
        )))
        state_weight = np.sqrt(base * rake)
        state_weight *= np.asarray((1.5, 2.0, 2.5, 1.5))[state]
        foot_weight = episode_balanced_weights(
            rows.foot_target[indices], rows.balance_unit[indices])
        foot_weight *= np.where(rows.wrong_foot_context[indices], 5.0, 1.0)
        contract = "EXACT_S4C_EPISODE_RAKING_STATE_LOSS_AND_WRONG_FOOT_WEIGHTING"
    else:
        state_weight = _targeted_cell_weights(
            state, rows.run_id[indices], rows.foot[indices], rows.contact_episode[indices])
        foot_weight = _targeted_cell_weights(
            rows.foot_target[indices], rows.run_id[indices], rows.foot[indices],
            rows.contact_episode[indices])
        contract = "TARGETED_V3_RUN_FOOT_CONTACT_STATE_CELL_BALANCE"
    hard = np.isin(rows.identity[indices], list(hard_ids))
    state_weight *= np.where(hard, 5.0, 1.0)
    state_weight *= len(state_weight) / state_weight.sum()
    foot_weight *= len(foot_weight) / foot_weight.sum()
    audit = {
        "variant": variant, "contract": contract, "row_count": len(indices),
        "hard_negative_row_count": int(np.sum(hard)),
        "state_weight_min": float(np.min(state_weight)),
        "state_weight_max": float(np.max(state_weight)),
        "state_weight_sum": float(np.sum(state_weight)),
        "foot_weight_min": float(np.min(foot_weight)),
        "foot_weight_max": float(np.max(foot_weight)),
        "foot_weight_sum": float(np.sum(foot_weight)),
        "effective_sample_size": float(
            np.sum(state_weight) ** 2 / np.sum(np.square(state_weight))),
    }
    return state_weight, foot_weight, audit


def fit_candidate(
    rows: TrainingRows, variant: str, seed: int, fit_folds: tuple[int, ...],
    hard_ids: set[str], *, model_family: str = "S4-C",
) -> tuple[SlipV2Model, dict[str, Any]]:
    indices = np.flatnonzero(rows.eligible & np.isin(rows.fold, fit_folds))
    state_weight, foot_weight, audit = training_weights(rows, indices, variant, hard_ids)
    model, health = fit_slip_v2_model(
        model_family, seed, rows.features[indices], rows.state[indices],
        rows.foot_target[indices], state_weight, foot_weight,
    )
    score = model.scores(rows.features[indices])
    probability = np.column_stack(tuple(score[key] for key in HEAD_NAMES[:4]))
    health.update(audit)
    health.update({
        "fit_folds": ";".join(map(str, fit_folds)),
        "fit_run_count": len(set(rows.run_id[indices].tolist())),
        "feature_count": rows.features.shape[1], "projection_width": 40,
        "state_head_classes": 4, "foot_head_classes": 2, "proposal_head_classes": 2,
        "train_accuracy": float(accuracy_score(rows.state[indices], np.argmax(probability, axis=1))),
        "train_log_loss": float(log_loss(rows.state[indices], probability, labels=np.arange(4))),
        "normalization_finite": bool(
            np.all(np.isfinite(model.mean)) and np.all(np.isfinite(model.scale))),
        "normalization_min_scale": float(np.min(model.scale)),
        "normalization_max_scale": float(np.max(model.scale)),
    })
    return model, health


def mine_hard_negatives(
    rows: TrainingRows,
) -> tuple[dict[int, set[str]], list[dict[str, Any]]]:
    """Reconstruct the original S4-B fixed-seed inner-only miner."""
    result: dict[int, set[str]] = {fold: set() for fold in (0, 1, 2)}
    audit: list[dict[str, Any]] = []
    for outer_fold in (0, 1, 2):
        inner_folds = tuple(fold for fold in (0, 1, 2) if fold != outer_fold)
        for inner_index, mining_fold in enumerate(inner_folds):
            fit_fold = next(fold for fold in inner_folds if fold != mining_fold)
            model, _ = fit_candidate(
                rows, "F0", 202608221, (fit_fold,), set(), model_family="S4-B")
            indices = np.flatnonzero(rows.eligible & (rows.fold == mining_fold))
            scores = model.scores(rows.features[indices])
            reasons = (
                ("high_normal", rows.state[indices] == NORMAL_NO_EVENT, scores["actionable"]),
                ("high_early_precursor", rows.state[indices] == EARLY_PRECURSOR,
                 scores["actionable"]),
                ("wrong_foot", rows.wrong_foot_context[indices],
                 scores["actionable"] + scores["foot"]),
            )
            for reason, mask, ranking in reasons:
                candidates = np.flatnonzero(mask)
                if not len(candidates):
                    continue
                count = min(256, max(1, int(np.ceil(0.10 * len(candidates)))))
                chosen = candidates[np.argsort(ranking[candidates], kind="stable")[-count:]]
                for local in chosen:
                    row_index = int(indices[local])
                    identity = str(rows.identity[row_index])
                    result[outer_fold].add(identity)
                    audit.append({
                        "outer_fold": outer_fold, "inner_fold": inner_index,
                        "fit_fold": fit_fold, "mining_fold": mining_fold,
                        "reason": reason, "identity": identity,
                        "run_id": str(rows.run_id[row_index]),
                        "outer_validation_member": int(rows.fold[row_index]) == outer_fold,
                        "actionable_score": float(scores["actionable"][local]),
                        "foot_score": float(scores["foot"][local]),
                    })
    if any(row["outer_validation_member"] for row in audit):
        raise RuntimeError("hard-negative leakage")
    return result, audit


def score_model(model: SlipV2Model, rows: TrainingRows) -> dict[str, np.ndarray]:
    flat = model.scores(rows.features)
    run_count = len(set(rows.run_id.tolist()))
    return {
        key: np.asarray(value).reshape(run_count, len(ENDPOINTS), 2)
        for key, value in flat.items()
    }


@dataclass
class CorrectedStateOutput:
    stable_internal: np.ndarray
    firing: np.ndarray
    verifier: np.ndarray
    proposal: np.ndarray
    persistence_count: np.ndarray
    reset_reason: np.ndarray
    runtime_episode: np.ndarray
    crossings: list[dict[str, Any]]
    fall_boundary_sample: int | None


def corrected_v4_state(
    scores: dict[str, np.ndarray], loaded: np.ndarray, valid: np.ndarray,
    prefall: np.ndarray, touchdown: np.ndarray, physical_episode: np.ndarray,
    config: OperatingConfig,
) -> CorrectedStateOutput:
    """Apply masks/reset/ownership before persistence, then choose one foot."""
    sample_loaded = np.asarray(loaded, bool)
    sample_valid = np.asarray(valid, bool)
    sample_prefall = np.asarray(prefall, bool)
    ages, runtime_episodes, _ = derived_runtime_telemetry(sample_loaded)
    expected = (len(ENDPOINTS), 2)
    if any(np.asarray(scores[name]).shape != expected for name in HEAD_NAMES):
        raise ValueError("score shape mismatch")
    internal = np.zeros(expected, bool)
    firing = np.zeros(expected, bool)
    verifier = np.zeros(expected, bool)
    proposal_pass = np.zeros(expected, bool)
    persistence = np.zeros(expected, np.int16)
    reset = np.full(expected, "none", dtype="<U28")
    counters = np.zeros(2, np.int16)
    active = np.zeros(2, bool)
    owner = np.full(2, -1, np.int32)
    emitted: set[tuple[int, int]] = set()
    fall_reset_done = np.zeros(2, bool)
    crossings: list[dict[str, Any]] = []
    fall_candidates = np.flatnonzero(~sample_prefall)
    fall_boundary = int(fall_candidates[0]) if fall_candidates.size else None
    for row, endpoint in enumerate(ENDPOINTS):
        candidates: list[int] = []
        for foot in range(2):
            runtime_episode = int(runtime_episodes[endpoint, foot])
            masked = bool(
                sample_loaded[endpoint, foot] and not touchdown[endpoint, foot]
                and sample_valid[endpoint, foot] and sample_prefall[endpoint]
                and ages[endpoint, foot] > 10 and runtime_episode >= 0)
            new_owner = runtime_episode != owner[foot]
            if not sample_prefall[endpoint]:
                counters[foot] = 0
                active[foot] = False
                reset[row, foot] = (
                    "post_fall_mask" if fall_reset_done[foot]
                    else "first_fall_hard_reset")
                fall_reset_done[foot] = True
                owner[foot] = runtime_episode
                continue
            if not sample_loaded[endpoint, foot]:
                counters[foot] = 0
                active[foot] = False
                reset[row, foot] = "contact_loss"
                owner[foot] = -1
                continue
            if new_owner or touchdown[endpoint, foot]:
                counters[foot] = 0
                active[foot] = False
                reset[row, foot] = "new_touchdown"
                owner[foot] = runtime_episode
            if not masked:
                counters[foot] = 0
                active[foot] = False
                if reset[row, foot] == "none":
                    reset[row, foot] = "invalid_mask"
                continue
            shift = config.hysteresis if active[foot] else 0.0
            proposal_pass[row, foot] = bool(
                scores["proposal"][row, foot] >= config.proposal_threshold - shift)
            condition = bool(
                scores["actionable"][row, foot] >= config.state_threshold - shift
                and scores["actionable"][row, foot] - scores["early"][row, foot]
                >= config.early_margin - shift
                and scores["actionable"][row, foot] - scores["normal"][row, foot]
                >= config.normal_margin - shift
                and scores["foot"][row, foot] >= config.foot_threshold - shift
                and proposal_pass[row, foot])
            verifier[row, foot] = condition
            if condition:
                counters[foot] += 1
                active[foot] = counters[foot] >= config.persistence_endpoints
            else:
                counters[foot] = 0
                if active[foot]:
                    reset[row, foot] = "score_recovery"
                active[foot] = False
            persistence[row, foot] = counters[foot]
            internal[row, foot] = active[foot]
            if active[foot]:
                candidates.append(foot)
        if candidates:
            chosen = max(candidates, key=lambda foot: (
                float(scores["actionable"][row, foot] - scores["early"][row, foot]
                      + 0.25 * (scores["foot"][row, foot] - 0.5)), -foot))
            firing[row, chosen] = True
            episode = int(runtime_episodes[endpoint, chosen])
            key = (chosen, episode)
            if key not in emitted:
                emitted.add(key)
                crossings.append({
                    "endpoint_row": row, "sample": int(endpoint), "foot": chosen,
                    "runtime_episode_id": episode,
                    "physical_episode_id": int(physical_episode[endpoint, chosen]),
                    "normal_score": float(scores["normal"][row, chosen]),
                    "early_score": float(scores["early"][row, chosen]),
                    "actionable_score": float(scores["actionable"][row, chosen]),
                    "active_score": float(scores["active"][row, chosen]),
                    "foot_score": float(scores["foot"][row, chosen]),
                    "proposal_score": float(scores["proposal"][row, chosen]),
                    "persistence_count": int(persistence[row, chosen]),
                    "valid_mask": bool(sample_valid[endpoint, chosen]),
                    "prefall_mask": bool(sample_prefall[endpoint]),
                    "loaded_mask": bool(sample_loaded[endpoint, chosen]),
                    "touchdown_mask": bool(touchdown[endpoint, chosen]),
                })
    return CorrectedStateOutput(
        internal, firing, verifier, proposal_pass, persistence, reset,
        runtime_episodes[ENDPOINTS], crossings, fall_boundary,
    )


def candidate_id(variant: str, seed: int) -> str:
    return f"{variant}_seed_{seed}"


def serializable_config(config: OperatingConfig) -> dict[str, Any]:
    return {**config.__dict__, "config_id": config.config_id}


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def contiguous_ranges(values: Iterable[Any]) -> list[tuple[int, int, Any]]:
    items = list(values)
    if not items:
        return []
    result: list[tuple[int, int, Any]] = []
    start = 0
    for index in range(1, len(items) + 1):
        if index == len(items) or items[index] != items[start]:
            result.append((start, index, items[start]))
            start = index
    return result
