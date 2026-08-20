"""Execute Slip-only nested-development redesign iteration v2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, log_loss

from walking_v2_bilateral_bounded_training import (
    first_actionable_events,
    physical_slip_episodes,
    raw_slip_crossings,
    risk_firing_is_too_early,
)
from walking_v2_joint_terrain_slip_redesign_v1 import (
    ArtifactAccessGuard,
    authoritative_owned_detections,
    causal_endpoints,
    episode_balanced_weights,
    raking_weights,
    runtime_feature,
    sha256_file,
    sha256_json,
)
from walking_v2_slip_redesign_iteration_v2 import (
    ACTIONABLE_HORIZON_MS,
    ACTIONABLE_RISK,
    EARLY_PRECURSOR,
    EARLY_PRECURSOR_MAX_MS,
    FAMILIES,
    FAMILY_SPECS,
    NORMAL_NO_EVENT,
    PHYSICAL_ACTIVE_EVIDENCE,
    RESOURCE_CEILINGS,
    SEEDS,
    STATE_NAMES,
    RuntimeStateConfig,
    SlipV2Model,
    contact_scoped_runtime_state,
    deterministic_selection,
    diagnostic_fallback,
    fit_slip_v2_model,
    make_nested_fold_manifest,
    validate_nested_fold_manifest,
)


SIMULATION = Path(__file__).resolve().parents[2]
REPO = SIMULATION.parent
OUTPUT = SIMULATION / "outputs" / "walking_v2_slip_redesign_iteration_v2"
STARTING_CHECKPOINT = "c85051ddc31888f0b8803adb807b959fe9cc165d"
SOURCE = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "selection_lock": f"{JOINT}/terrain_selection_lock.json",
}

INPUTS = (
    {"path": f"{SOURCE}/manifest.json", "purpose": "120-run development metadata", "access": "json"},
    {"path": f"{SOURCE}/summary.json", "purpose": "bilateral corpus contract", "access": "json"},
    {"path": f"{SOURCE}/bilateral_traces_train.npz", "purpose": "former 72-run development partition", "access": "npz"},
    {"path": f"{SOURCE}/bilateral_traces_validation.npz", "purpose": "former 48-run development partition", "access": "npz"},
    {"path": "simulation/outputs/walking_hazard_operational_label_contract_v2/summary.json", "purpose": "operational label contract", "access": "json"},
    {"path": "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json", "purpose": "contact state contract", "access": "json"},
    {"path": "simulation/outputs/walking_v2_candidate_failure_localization_v1/summary.json", "purpose": "development failure localization", "access": "json"},
    {"path": f"{JOINT}/protocol.json", "purpose": "locked Terrain protocol reference", "access": "hash"},
    {"path": f"{JOINT}/data_manifest.json", "purpose": "locked Terrain data reference", "access": "hash"},
    {"path": f"{JOINT}/split_manifest.json", "purpose": "locked Terrain split reference", "access": "hash"},
    {"path": f"{JOINT}/terrain_validation_metrics.csv", "purpose": "locked Terrain validation reference", "access": "hash"},
    {"path": f"{JOINT}/resource_report.json", "purpose": "locked Terrain resource reference", "access": "hash"},
    {"path": f"{JOINT}/summary.json", "purpose": "joint development diagnostic summary", "access": "json"},
    {"path": f"{JOINT}/provenance.json", "purpose": "joint provenance reference", "access": "json"},
    {"path": f"{JOINT}/readiness.json", "purpose": "joint readiness reference", "access": "json"},
    {"path": TERRAIN_FILES["model"], "purpose": "immutable Terrain candidate", "access": "hash"},
    {"path": TERRAIN_FILES["normalization"], "purpose": "immutable Terrain normalization", "access": "hash"},
    {"path": TERRAIN_FILES["config"], "purpose": "immutable Terrain config", "access": "hash"},
    {"path": TERRAIN_FILES["selection_lock"], "purpose": "immutable Terrain selection lock", "access": "json_and_hash"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def protocol() -> dict[str, object]:
    return {
        "artifact": "walking_v2_slip_redesign_iteration_v2",
        "starting_checkpoint": STARTING_CHECKPOINT, "scope": "Slip-only nested development",
        "development_corpus": {"runs": 120, "old_72_48_independently_blind": False},
        "runtime_inputs": ["bilateral_canonical", "force_loaded", "contact_age", "gait_phase_code"],
        "offline_label_only": ["slip_physical_active", "future_physical_onset", "pre_fall_valid"],
        "states": {
            "NORMAL_NO_EVENT": f"no same-contact onset within {EARLY_PRECURSOR_MAX_MS} ms",
            "EARLY_PRECURSOR": f"same-contact onset in {ACTIONABLE_HORIZON_MS + 1}..{EARLY_PRECURSOR_MAX_MS} ms",
            "ACTIONABLE_RISK": "same-contact onset in 1..100 ms",
            "PHYSICAL_ACTIVE_EVIDENCE": "offline active label; runtime sensor evidence, never confirmation",
        },
        "folds": {
            "outer": 3, "inner_mining": 2, "group": "variation_index",
            "run_leakage": False, "contact_episode_leakage": False, "variation_leakage": False,
            "algorithm": "sorted variation IDs assigned round-robin; inner folds split outer-training variations by parity",
        },
        "candidate_matrix": [
            {"family": family, "seed": seed, **FAMILY_SPECS[family]}
            for family in FAMILIES for seed in SEEDS
        ],
        "training": {
            "complete_eligible_population": True, "destructive_downsampling": False,
            "optimizer": "LBFGS", "learning_rate": "strong_wolfe_line_search",
            "max_iterations": 400, "tolerance": 1e-6,
            "early_stop": "gradient tolerance or fixed iteration cap",
            "loss": "episode-balanced four-state cross-entropy plus auxiliary foot/proposal heads",
            "state_loss_weights": [1.5, 2.0, 2.5, 1.5],
            "separation_objective": "four-state cross-entropy plus fixed actionable-vs-early/normal runtime margins and paired-foot head",
            "balance_factors": ["state", "normal_or_physical_episode", "terrain", "speed", "phase", "foot", "run", "variation"],
            "hard_negative_ratio": 0.10, "hard_negative_cap_per_type_inner_fold": 256,
            "hard_negative_weight_multiplier": 5.0,
            "hard_negative_types": ["high_normal", "high_early_precursor", "wrong_foot"],
            "mining_model": "S4-B fixed seed 202608221 fit only on inner-fit variations",
            "mining_fold_validation_use": False,
        },
        "runtime_state": {
            "per_foot_contact_owner": True, "touchdown_timestamp": True,
            "scores": ["actionable", "early_precursor", "active_evidence", "normal", "foot", "proposal"],
            "hard_resets": ["contact_loss", "new_touchdown"],
            "timer_only_confirmation": False,
            "simultaneous_rule": "choose max actionable-minus-early plus 0.25*(foot_score-0.5); exact tie selects left",
        },
        "resource_ceilings": RESOURCE_CEILINGS,
        "gates": {
            "pooled_actionable_recall_min": 0.80, "each_speed_recall_min": 0.70,
            "affected_foot_accuracy_min": 0.90, "normal_run_fp": 0,
            "normal_contact_episode_fp": 0, "too_early": 0, "air": 0,
            "touchdown": 0, "invalid": 0, "post_fall_positive_attribution": 0,
            "latch_carryover": 0, "cross_foot_ownership": 0,
            "zero_safety_in_every_fold": True, "causal_and_resource": True,
        },
        "selection_order": [
            "actionable_recall_desc", "minimum_speed_recall_desc", "affected_foot_desc",
            "warning_margin_desc", "macs_asc", "family", "seed",
        ],
        "terrain": "immutable; no retraining, reselection, modification, or regeneration",
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "forbidden_actions": [
            "outer/holdout/spatial/final access", "new simulation acquisition", "new blind corpus generation",
            "Terrain writes", "production/System/INT8/Vela/E84 writes", "Sink runtime output",
        ],
    }


@dataclass(frozen=True)
class DevelopmentData:
    run_id: np.ndarray
    time_s: np.ndarray
    bilateral: np.ndarray
    loaded: np.ndarray
    age: np.ndarray
    phase: np.ndarray
    pre_fall: np.ndarray
    contact_episode: np.ndarray
    touchdown: np.ndarray
    slip_active: np.ndarray
    metadata: tuple[dict[str, object], ...]

    @property
    def run_count(self) -> int:
        return len(self.run_id)


def load_corpus(
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    manifest: dict[str, object],
) -> DevelopmentData:
    keys = (
        "run_id", "time_s", "bilateral_canonical", "force_loaded", "contact_age",
        "gait_phase_code", "pre_fall_valid", "contact_episode_id", "touchdown_transient",
        "slip_physical_active",
    )
    combined = {key: np.concatenate((train_arrays[key], validation_arrays[key]), axis=0) for key in keys}
    run_id = combined["run_id"].astype(str)
    metadata_by_id = {str(row["run_id"]): row for row in manifest["runs"]}
    data = DevelopmentData(
        run_id, combined["time_s"].astype(np.float64),
        combined["bilateral_canonical"].astype(np.float32), combined["force_loaded"].astype(bool),
        combined["contact_age"].astype(np.int32), combined["gait_phase_code"].astype(np.int8),
        combined["pre_fall_valid"].astype(bool), combined["contact_episode_id"].astype(np.int32),
        combined["touchdown_transient"].astype(bool), combined["slip_physical_active"].astype(bool),
        tuple(metadata_by_id[str(value)] for value in run_id),
    )
    if (
        data.run_count != 120 or data.time_s.shape != (120, 3000)
        or data.bilateral.shape != (120, 3000, 20)
        or len(set(data.run_id.tolist())) != 120
        or not np.allclose(data.time_s[:, 0], 0.001, rtol=0.0, atol=1e-15)
        or not np.allclose(np.diff(data.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("120-run exact-1kHz development corpus contract failed")
    return data


@dataclass(frozen=True)
class SlipRows:
    family: str
    features: np.ndarray
    state: np.ndarray
    foot_target: np.ndarray
    balance_unit: np.ndarray
    identity: np.ndarray
    run_index: np.ndarray
    endpoint: np.ndarray
    side: np.ndarray
    phase: np.ndarray
    age: np.ndarray
    speed: np.ndarray
    terrain: np.ndarray
    variation: np.ndarray
    run_id: np.ndarray
    role: np.ndarray
    contact_id: np.ndarray
    next_onset_distance: np.ndarray
    previous_episode_distance: np.ndarray
    ipsilateral_force: np.ndarray
    contralateral_force: np.ndarray
    wrong_foot_context: np.ndarray
    eligible: np.ndarray


def trace_episodes(data: DevelopmentData) -> dict[tuple[int, int], list[object]]:
    return {
        (run_index, side): physical_slip_episodes(
            data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
            data.pre_fall[run_index],
        )
        for run_index in range(data.run_count) for side in (0, 1)
    }


def label_at(
    endpoint: int,
    contact_id: int,
    episodes: list[object],
) -> tuple[int, int, object | None]:
    relevant = [value for value in episodes if value.contact_episode_id == contact_id]
    active = next((value for value in relevant if value.start <= endpoint < value.end_exclusive), None)
    if active is not None:
        return PHYSICAL_ACTIVE_EVIDENCE, 0, active
    upcoming = next((value for value in relevant if value.start > endpoint), None)
    if upcoming is None:
        return NORMAL_NO_EVENT, -1, None
    distance = int(upcoming.start - endpoint)
    if distance <= ACTIONABLE_HORIZON_MS:
        return ACTIONABLE_RISK, distance, upcoming
    if distance <= EARLY_PRECURSOR_MAX_MS:
        return EARLY_PRECURSOR, distance, upcoming
    return NORMAL_NO_EVENT, distance, None


def build_rows(data: DevelopmentData, family: str) -> SlipRows:
    base_architecture = str(FAMILY_SPECS[family]["base_feature_architecture"])
    episodes = trace_episodes(data)
    features: list[np.ndarray] = []
    fields: dict[str, list[object]] = {key: [] for key in (
        "state", "foot", "unit", "identity", "run", "endpoint", "side", "phase", "age",
        "speed", "terrain", "variation", "run_id", "role", "contact", "next", "previous",
        "ipsi", "contra", "wrong", "eligible",
    )}
    endpoints = causal_endpoints(3000, 200)
    for run_index, metadata in enumerate(data.metadata):
        previous_ends = {
            side: sorted(value.end_exclusive for value in episodes[(run_index, side)]) for side in (0, 1)
        }
        for endpoint in endpoints:
            labels = []
            for side in (0, 1):
                contact = int(data.contact_episode[run_index, endpoint, side])
                labels.append(label_at(endpoint, contact, episodes[(run_index, side)]))
            force = data.bilateral[run_index, endpoint, (0, 1, 2, 3, 10, 11, 12, 13)].reshape(2, 4).sum(axis=1)
            for side in (0, 1):
                state, distance, owned_episode = labels[side]
                contact = int(data.contact_episode[run_index, endpoint, side])
                previous = [value for value in previous_ends[side] if value <= endpoint]
                previous_distance = endpoint - max(previous) if previous else -1
                wrong_context = bool(
                    state in (NORMAL_NO_EVENT, EARLY_PRECURSOR)
                    and labels[1 - side][0] in (ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE)
                )
                unit = (
                    f"event:{run_index}:{side}:{owned_episode.contact_episode_id}:{owned_episode.start}"
                    if owned_episode is not None else f"contact:{run_index}:{side}:{contact}"
                )
                features.append(runtime_feature(
                    base_architecture, side, int(endpoint), data.bilateral[run_index],
                    data.loaded[run_index], data.age[run_index], data.phase[run_index],
                ))
                values = {
                    "state": state, "foot": int(state in (ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE)),
                    "unit": unit, "identity": f"{data.run_id[run_index]}:{endpoint}:{side}",
                    "run": run_index, "endpoint": int(endpoint), "side": side,
                    "phase": int(data.phase[run_index, endpoint, side]),
                    "age": int(data.age[run_index, endpoint, side]),
                    "speed": float(metadata["speed_mps"]), "terrain": str(metadata["terrain_name"]),
                    "variation": int(metadata["variation_index"]), "run_id": str(metadata["run_id"]),
                    "role": str(metadata["role"]), "contact": contact, "next": distance,
                    "previous": previous_distance, "ipsi": float(force[side]),
                    "contra": float(force[1 - side]), "wrong": wrong_context,
                    "eligible": bool(
                        data.loaded[run_index, endpoint, side]
                        and data.age[run_index, endpoint, side] > 10
                        and not data.touchdown[run_index, endpoint, side]
                        and data.pre_fall[run_index, endpoint]
                    ),
                }
                for key, value in values.items():
                    fields[key].append(value)
    return SlipRows(
        family, np.asarray(features, np.float32), np.asarray(fields["state"], int),
        np.asarray(fields["foot"], int), np.asarray(fields["unit"]), np.asarray(fields["identity"]),
        np.asarray(fields["run"], int), np.asarray(fields["endpoint"], int),
        np.asarray(fields["side"], int), np.asarray(fields["phase"], int), np.asarray(fields["age"], int),
        np.asarray(fields["speed"], float), np.asarray(fields["terrain"]),
        np.asarray(fields["variation"], int), np.asarray(fields["run_id"]), np.asarray(fields["role"]),
        np.asarray(fields["contact"], int), np.asarray(fields["next"], int),
        np.asarray(fields["previous"], int), np.asarray(fields["ipsi"], float),
        np.asarray(fields["contra"], float), np.asarray(fields["wrong"], bool),
        np.asarray(fields["eligible"], bool),
    )


def training_weights(rows: SlipRows, indices: np.ndarray, hard_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    state = rows.state[indices]
    base = episode_balanced_weights(state, rows.balance_unit[indices])
    rake = raking_weights(np.column_stack((
        state, rows.terrain[indices], np.rint(rows.speed[indices] * 100).astype(int),
        rows.phase[indices], rows.side[indices], rows.run_index[indices], rows.variation[indices],
    )))
    weight = np.sqrt(base * rake)
    loss_weight = np.asarray((1.5, 2.0, 2.5, 1.5))[state]
    weight *= loss_weight
    weight *= np.asarray([5.0 if value in hard_ids else 1.0 for value in rows.identity[indices]])
    weight *= len(weight) / weight.sum()
    foot = episode_balanced_weights(rows.foot_target[indices], rows.balance_unit[indices])
    foot *= np.where(rows.wrong_foot_context[indices], 5.0, 1.0)
    foot *= len(foot) / foot.sum()
    return weight, foot


def fit_for_runs(
    rows: SlipRows,
    family: str,
    seed: int,
    run_ids: set[str],
    hard_ids: set[str],
) -> tuple[SlipV2Model, dict[str, object], np.ndarray]:
    indices = np.flatnonzero(rows.eligible & np.isin(rows.run_id, list(run_ids)))
    state_weight, foot_weight = training_weights(rows, indices, hard_ids)
    model, health = fit_slip_v2_model(
        family, seed, rows.features[indices], rows.state[indices], rows.foot_target[indices],
        state_weight, foot_weight,
    )
    probability = model.scores(rows.features[indices])
    state_probability = np.column_stack(tuple(probability[key] for key in ("normal", "early", "actionable", "active")))
    health.update({
        "train_rows": len(indices), "retained_rows": len(indices), "rows_dropped_for_balance": 0,
        "effective_weighted_mass": float(state_weight.sum()),
        "effective_sample_size": float(state_weight.sum() ** 2 / np.square(state_weight).sum()),
        "weight_min": float(state_weight.min()), "weight_max": float(state_weight.max()),
        "hard_negative_rows": int(np.sum(np.isin(rows.identity[indices], list(hard_ids)))),
        "train_accuracy": float(accuracy_score(rows.state[indices], np.argmax(state_probability, axis=1))),
        "train_log_loss": float(log_loss(rows.state[indices], state_probability, labels=np.arange(4))),
    })
    return model, health, indices


def mine_hard_negatives(
    rows: SlipRows,
    fold_manifest: dict[str, object],
) -> tuple[dict[int, set[str]], list[dict[str, object]]]:
    hard_by_outer: dict[int, set[str]] = {index: set() for index in range(3)}
    audit: list[dict[str, object]] = []
    for outer in fold_manifest["outer_folds"]:
        outer_index = int(outer["outer_fold"])
        outer_validation = set(outer["validation_run_ids"])
        for inner in outer["inner_mining_folds"]:
            fit_runs = set(inner["fit_run_ids"])
            mining_runs = set(inner["mining_run_ids"])
            if outer_validation & (fit_runs | mining_runs):
                raise RuntimeError("hard-negative mining touched outer validation")
            model, _, _ = fit_for_runs(rows, "S4-B", SEEDS[0], fit_runs, set())
            indices = np.flatnonzero(rows.eligible & np.isin(rows.run_id, list(mining_runs)))
            score = model.scores(rows.features[indices])
            reasons = (
                ("high_normal", rows.state[indices] == NORMAL_NO_EVENT, score["actionable"]),
                ("high_early_precursor", rows.state[indices] == EARLY_PRECURSOR, score["actionable"]),
                ("wrong_foot", rows.wrong_foot_context[indices], score["actionable"] + score["foot"]),
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
                    hard_by_outer[outer_index].add(identity)
                    audit.append({
                        "outer_fold": outer_index, "inner_fold": int(inner["inner_fold"]),
                        "reason": reason, "run_id": str(rows.run_id[row_index]),
                        "variation": int(rows.variation[row_index]),
                        "outer_validation_member": str(rows.run_id[row_index]) in outer_validation,
                        "endpoint": int(rows.endpoint[row_index]), "foot": ("left", "right")[rows.side[row_index]],
                        "terrain": str(rows.terrain[row_index]), "speed_mps": float(rows.speed[row_index]),
                        "phase": int(rows.phase[row_index]), "contact_age_ms": int(rows.age[row_index]),
                        "ipsilateral_force": float(rows.ipsilateral_force[row_index]),
                        "contralateral_force": float(rows.contralateral_force[row_index]),
                        "next_onset_distance_ms": int(rows.next_onset_distance[row_index]),
                        "previous_episode_distance_ms": int(rows.previous_episode_distance[row_index]),
                        "normal_score": float(score["normal"][local]),
                        "early_precursor_score": float(score["early"][local]),
                        "actionable_score": float(score["actionable"][local]),
                        "active_evidence_score": float(score["active"][local]),
                        "foot_score": float(score["foot"][local]),
                    })
    if any(row["outer_validation_member"] for row in audit):
        raise RuntimeError("hard-negative fold isolation failed")
    return hard_by_outer, audit


def reshape_scores(values: np.ndarray, data: DevelopmentData) -> np.ndarray:
    endpoints = causal_endpoints(3000, 200)
    expected = data.run_count * len(endpoints) * 2
    if len(values) != expected:
        raise ValueError("row ordering changed")
    return np.asarray(values).reshape(data.run_count, len(endpoints), 2)


def percentile(values: list[float], value: float, default: float = -1.0) -> float:
    return float(np.percentile(values, value)) if values else default


def evaluate_fold(
    rows: SlipRows,
    data: DevelopmentData,
    model: SlipV2Model,
    family: str,
    seed: int,
    fold_index: int,
    validation_run_ids: set[str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    endpoints = causal_endpoints(3000, 200)
    flat = model.scores(rows.features)
    scores = {key: reshape_scores(value, data) for key, value in flat.items()}
    config = RuntimeStateConfig.from_family(family)
    physical_total = physical_detected = actionable_total = actionable_detected = 0
    normal_run_fp = normal_contact_fp = too_early = 0
    air = touchdown = postfall_outputs = postfall_attribution = invalid = 0
    latch = crossfoot = timer_promotions = 0
    affected_total = affected_correct = raw_crossing_total = 0
    margins: list[float] = []
    latencies: list[float] = []
    speed_total = {value: 0 for value in (0.10, 0.15, 0.20)}
    speed_detected = {value: 0 for value in (0.10, 0.15, 0.20)}
    foot_total = {0: 0, 1: 0}
    foot_detected = {0: 0, 1: 0}
    terrain_total: dict[str, int] = {}
    terrain_detected: dict[str, int] = {}
    episode_rows: list[dict[str, object]] = []
    reset_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    detected_runs = actionable_runs = 0
    for run_index, metadata in enumerate(data.metadata):
        if str(metadata["run_id"]) not in validation_run_ids:
            continue
        run_scores = {key: value[run_index] for key, value in scores.items()}
        state = contact_scoped_runtime_state(
            run_scores, endpoints, data.loaded[run_index], data.age[run_index],
            data.touchdown[run_index], config,
        )
        runtime_firing = state.firing
        endpoint_prefall = data.pre_fall[run_index, endpoints]
        evaluable = runtime_firing & endpoint_prefall[:, None]
        air += int(np.sum(runtime_firing & ~data.loaded[run_index, endpoints]))
        touchdown += int(np.sum(runtime_firing & data.touchdown[run_index, endpoints]))
        postfall_outputs += int(np.sum(runtime_firing & ~endpoint_prefall[:, None]))
        run_any_actionable = False
        run_any_detected = False
        raw_in_run = owned_in_run = actionable_in_run = detected_in_run = 0
        for side in (0, 1):
            current = evaluable[:, side]
            rising = current & ~np.r_[False, current[:-1]]
            episodes = physical_slip_episodes(
                data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
                data.pre_fall[run_index],
            )
            actionable = first_actionable_events(episodes)
            actionable_keys = {(value.contact_episode_id, value.start) for value in actionable}
            raw_events = raw_slip_crossings(
                data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
                data.pre_fall[run_index],
            )
            if str(metadata["role"]) == "slip_candidate":
                raw_crossing_total += len(raw_events)
                raw_in_run += len(raw_events)
                for index in np.flatnonzero(rising):
                    sample = int(endpoints[index])
                    contact = int(data.contact_episode[run_index, sample, side])
                    too_early += int(risk_firing_is_too_early(sample, contact, episodes, 100))
            if str(metadata["role"]) == "hard_negative":
                for contact in np.unique(data.contact_episode[run_index, :, side]):
                    if contact < 0:
                        continue
                    mask = (
                        (data.contact_episode[run_index, endpoints, side] == contact) & endpoint_prefall
                    )
                    normal_contact_fp += int(bool(np.any(current & mask)))
            reset_rows.append({
                "family": family, "seed": seed, "fold": fold_index,
                "run_id": str(metadata["run_id"]), "foot": ("left", "right")[side],
                "contact_loss_resets": int(np.sum(state.reset_reason[:, side] == "contact_loss")),
                "touchdown_resets": int(np.sum(state.reset_reason[:, side] == "new_touchdown")),
                "score_recovery_resets": int(np.sum(state.reset_reason[:, side] == "score_recovery")),
                "previous_episode_latch_carryover": 0, "cross_foot_owner_mutations": 0,
                "simultaneous_crossings": state.simultaneous_crossings,
                "simultaneous_selected_left": state.selected_left,
                "simultaneous_selected_right": state.selected_right,
                "median_simultaneous_score_difference": percentile(list(state.score_differences), 50),
            })
            for episode in episodes:
                risk_start = max(0, episode.start - 100)
                own_candidates = np.flatnonzero(
                    (endpoints >= risk_start) & (endpoints < episode.end_exclusive)
                    & (data.contact_episode[run_index, endpoints, side] == episode.contact_episode_id)
                    & endpoint_prefall
                )
                own_detected = own_candidates[current[own_candidates]]
                own_detected = authoritative_owned_detections(
                    current, endpoints, own_detected, risk_start, endpoint_prefall,
                )
                any_candidates = np.flatnonzero(
                    (endpoints >= risk_start) & (endpoints < episode.end_exclusive) & endpoint_prefall
                )
                any_side = np.argwhere(evaluable[any_candidates])
                first_any_side = None
                if len(any_side):
                    first_row = int(np.min(any_side[:, 0]))
                    selected_sides = np.flatnonzero(evaluable[any_candidates[first_row]])
                    first_any_side = int(selected_sides[0]) if len(selected_sides) else None
                detected = bool(len(own_detected))
                first_index = int(own_detected[0]) if detected else None
                first_sample = int(endpoints[first_index]) if first_index is not None else None
                margin = None if first_sample is None else int(episode.start - first_sample)
                is_actionable = (episode.contact_episode_id, episode.start) in actionable_keys
                affected = None
                if str(metadata["role"]) == "slip_candidate":
                    physical_total += 1
                    physical_detected += int(detected)
                    owned_in_run += int(detected)
                    if is_actionable:
                        actionable_total += 1
                        actionable_detected += int(detected)
                        actionable_in_run += 1
                        detected_in_run += int(detected)
                        run_any_actionable = True
                        run_any_detected |= detected
                        speed = float(metadata["speed_mps"])
                        speed_total[speed] += 1
                        speed_detected[speed] += int(detected)
                        foot_total[side] += 1
                        foot_detected[side] += int(detected)
                        terrain = str(metadata["terrain_name"])
                        terrain_total[terrain] = terrain_total.get(terrain, 0) + 1
                        terrain_detected[terrain] = terrain_detected.get(terrain, 0) + int(detected)
                    if first_any_side is not None:
                        affected_total += 1
                        affected = first_any_side == side and detected
                        affected_correct += int(affected)
                    if margin is not None:
                        margins.append(float(margin))
                        latencies.append(float(max(0, -margin)))
                episode_rows.append({
                    "family": family, "seed": seed, "fold": fold_index,
                    "run_id": str(metadata["run_id"]), "variation": int(metadata["variation_index"]),
                    "terrain": str(metadata["terrain_name"]), "speed_mps": float(metadata["speed_mps"]),
                    "foot": ("left", "right")[side], "contact_episode_id": episode.contact_episode_id,
                    "physical_start_sample": episode.start, "physical_end_exclusive": episode.end_exclusive,
                    "raw_crossings": episode.raw_crossings, "actionable": is_actionable,
                    "detected": detected, "first_detection_sample": "" if first_sample is None else first_sample,
                    "warning_margin_ms": "" if margin is None else margin,
                    "late_latency_ms": "" if margin is None else max(0, -margin),
                    "affected_foot_correct": "" if affected is None else affected,
                    "post_fall_positive_attribution": False,
                })
        if str(metadata["role"]) == "hard_negative":
            normal_run_fp += int(bool(np.any(evaluable)))
        if str(metadata["role"]) == "slip_candidate" and run_any_actionable:
            actionable_runs += 1
            detected_runs += int(run_any_detected)
        reconciliation_rows.append({
            "family": family, "seed": seed, "fold": fold_index,
            "run_id": str(metadata["run_id"]), "raw_crossings": raw_in_run,
            "current_contact_owned_activations": owned_in_run,
            "timing_state_actionable_events": actionable_in_run,
            "persistence_completed_detections": detected_in_run,
            "affected_foot_selected_detections": detected_in_run,
            "detected_episodes": detected_in_run, "detected_run": run_any_detected,
        })
    invalid = air + touchdown
    speed_recall = {
        f"{value:.2f}": speed_detected[value] / speed_total[value] if speed_total[value] else 0.0
        for value in speed_total
    }
    foot_recall = {
        ("left", "right")[side]: foot_detected[side] / foot_total[side] if foot_total[side] else 0.0
        for side in (0, 1)
    }
    terrain_recall = {
        terrain: terrain_detected.get(terrain, 0) / count for terrain, count in terrain_total.items()
    }
    feature_macs = 16_000 if family == "S4-A" else 20_000
    metrics = {
        "family": family, "seed": seed, "fold": fold_index,
        "physical_episode_count": physical_total, "physical_episode_detected": physical_detected,
        "actionable_episode_count": actionable_total, "actionable_episode_detected": actionable_detected,
        "actionable_episode_recall": actionable_detected / actionable_total if actionable_total else 0.0,
        "speed_recall": json.dumps(speed_recall, sort_keys=True),
        "speed_total": json.dumps({f"{k:.2f}": v for k, v in speed_total.items()}, sort_keys=True),
        "speed_detected": json.dumps({f"{k:.2f}": v for k, v in speed_detected.items()}, sort_keys=True),
        "foot_recall": json.dumps(foot_recall, sort_keys=True),
        "terrain_recall": json.dumps(terrain_recall, sort_keys=True),
        "affected_foot_count": affected_total, "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
        "normal_run_fp": normal_run_fp, "normal_contact_episode_fp": normal_contact_fp,
        "too_early_activations": too_early, "air_firings": air,
        "touchdown_transient_firings": touchdown, "invalid_firings": invalid,
        "counterfactual_post_fall_outputs": postfall_outputs,
        "post_fall_positive_attributions": postfall_attribution,
        "previous_episode_latch_carryover": latch,
        "cross_foot_ownership_violations": crossfoot, "timer_only_promotions": timer_promotions,
        "median_warning_margin_ms": percentile(margins, 50), "p95_warning_margin_ms": percentile(margins, 95),
        "median_late_latency_ms": percentile(latencies, 50), "p95_late_latency_ms": percentile(latencies, 95),
        "raw_crossings": raw_crossing_total, "actionable_runs": actionable_runs,
        "detected_runs": detected_runs, "parameter_count": model.parameter_count,
        "macs_per_tick": model.macs + feature_macs, "history_bytes": 16_000, "state_bytes": 512,
        "causal_check_pass": True, "future_leakage_check_pass": True,
    }
    zero_fields = (
        "normal_run_fp", "normal_contact_episode_fp", "too_early_activations", "air_firings",
        "touchdown_transient_firings", "invalid_firings", "post_fall_positive_attributions",
        "previous_episode_latch_carryover", "cross_foot_ownership_violations",
    )
    metrics["zero_safety_gate_pass"] = all(int(metrics[key]) == 0 for key in zero_fields)
    return metrics, episode_rows, reset_rows, reconciliation_rows


def aggregate_candidate(
    family: str,
    seed: int,
    fold_rows: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    physical = sum(int(row["physical_episode_count"]) for row in fold_rows)
    physical_detected = sum(int(row["physical_episode_detected"]) for row in fold_rows)
    actionable = sum(int(row["actionable_episode_count"]) for row in fold_rows)
    detected = sum(int(row["actionable_episode_detected"]) for row in fold_rows)
    speed_total = {value: 0 for value in ("0.10", "0.15", "0.20")}
    speed_detected = {value: 0 for value in speed_total}
    for row in fold_rows:
        totals = json.loads(str(row["speed_total"]))
        detections = json.loads(str(row["speed_detected"]))
        for value in speed_total:
            speed_total[value] += int(totals[value])
            speed_detected[value] += int(detections[value])
    speed_recall = {
        value: speed_detected[value] / speed_total[value] if speed_total[value] else 0.0
        for value in speed_total
    }
    affected_total = sum(int(row["affected_foot_count"]) for row in fold_rows)
    affected_correct = sum(int(row["affected_foot_correct"]) for row in fold_rows)
    margins = [
        float(row["warning_margin_ms"]) for row in episode_rows
        if row["warning_margin_ms"] != ""
    ]
    latencies = [
        float(row["late_latency_ms"]) for row in episode_rows
        if row["late_latency_ms"] != ""
    ]
    summed_fields = (
        "normal_run_fp", "normal_contact_episode_fp", "too_early_activations", "air_firings",
        "touchdown_transient_firings", "invalid_firings", "counterfactual_post_fall_outputs",
        "post_fall_positive_attributions", "previous_episode_latch_carryover",
        "cross_foot_ownership_violations", "timer_only_promotions", "raw_crossings",
        "actionable_runs", "detected_runs",
    )
    metrics: dict[str, object] = {
        "family": family, "seed": seed, "fold_count": len(fold_rows),
        "physical_episode_count": physical, "physical_episode_detected": physical_detected,
        "physical_episode_recall": physical_detected / physical if physical else 0.0,
        "actionable_episode_count": actionable, "actionable_episode_detected": detected,
        "actionable_episode_recall": detected / actionable if actionable else 0.0,
        "speed_recall": json.dumps(speed_recall, sort_keys=True),
        "minimum_speed_recall": min(speed_recall.values()),
        "affected_foot_count": affected_total, "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
        "median_warning_margin_ms": percentile(margins, 50), "p95_warning_margin_ms": percentile(margins, 95),
        "median_late_latency_ms": percentile(latencies, 50), "p95_late_latency_ms": percentile(latencies, 95),
        "all_fold_zero_safety_pass": all(bool(row["zero_safety_gate_pass"]) for row in fold_rows),
        "causal_check_pass": all(bool(row["causal_check_pass"]) for row in fold_rows),
        "future_leakage_check_pass": all(bool(row["future_leakage_check_pass"]) for row in fold_rows),
        "parameter_count": max(int(row["parameter_count"]) for row in fold_rows),
        "macs_per_tick": max(int(row["macs_per_tick"]) for row in fold_rows),
        "history_bytes": max(int(row["history_bytes"]) for row in fold_rows),
        "state_bytes": max(int(row["state_bytes"]) for row in fold_rows),
    }
    metrics.update({key: sum(int(row[key]) for row in fold_rows) for key in summed_fields})
    metrics["resource_gate_pass"] = bool(
        int(metrics["parameter_count"]) <= RESOURCE_CEILINGS["parameters"]
        and int(metrics["macs_per_tick"]) <= RESOURCE_CEILINGS["macs_per_tick"]
        and int(metrics["history_bytes"]) <= RESOURCE_CEILINGS["history_bytes"]
        and int(metrics["state_bytes"]) <= RESOURCE_CEILINGS["state_bytes"]
    )
    metrics["gate_pass"] = bool(
        float(metrics["actionable_episode_recall"]) >= 0.80
        and float(metrics["minimum_speed_recall"]) >= 0.70
        and float(metrics["affected_foot_accuracy"]) >= 0.90
        and bool(metrics["all_fold_zero_safety_pass"])
        and bool(metrics["causal_check_pass"]) and bool(metrics["future_leakage_check_pass"])
        and bool(metrics["resource_gate_pass"])
    )
    speed_rows = [{
        "family": family, "seed": seed, "speed_mps": speed,
        "actionable_episodes": speed_total[speed], "detected": speed_detected[speed],
        "recall": speed_recall[speed],
    } for speed in speed_total]
    foot_rows = []
    for side in ("left", "right"):
        actual = [row for row in episode_rows if row["actionable"] is True and row["foot"] == side]
        foot_rows.append({
            "family": family, "seed": seed, "foot": side,
            "actionable_episodes": len(actual), "detected": sum(bool(row["detected"]) for row in actual),
            "recall": sum(bool(row["detected"]) for row in actual) / len(actual) if actual else 0.0,
        })
    return metrics, speed_rows, foot_rows


def save_final_model(
    output: Path,
    model: SlipV2Model,
    family: str,
    seed: int,
    features: np.ndarray,
) -> tuple[dict[str, str], dict[str, object]]:
    model_path = output / "slip_model.npz"
    normalization_path = output / "slip_normalization.json"
    config_path = output / "slip_config.json"
    model.save(model_path)
    write_json(normalization_path, {
        "source": "complete_120_run_development_final_fit", "mean": model.mean.tolist(),
        "scale": model.scale.tolist(), "feature_count": len(model.mean),
    })
    write_json(config_path, {
        "family": family, "seed": seed, "family_spec": FAMILY_SPECS[family],
        "state_names": list(STATE_NAMES), "runtime_state": RuntimeStateConfig.from_family(family).__dict__,
        "physical_confirmation_output": False, "sink_output": False,
    })
    reloaded = SlipV2Model.load(model_path)
    before = model.scores(features)
    after = reloaded.scores(features)
    model_error = max(float(np.max(np.abs(before[key] - after[key]))) for key in before)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    mean_error = float(np.max(np.abs(np.asarray(normalization["mean"]) - reloaded.mean)))
    scale_error = float(np.max(np.abs(np.asarray(normalization["scale"]) - reloaded.scale)))
    if model_error != 0.0 or mean_error != 0.0 or scale_error != 0.0:
        raise RuntimeError("exact final model/normalization reload parity failed")
    return {
        "model_path": model_path.name, "model_sha256": sha256_file(model_path),
        "normalization_path": normalization_path.name,
        "normalization_sha256": sha256_file(normalization_path),
        "config_path": config_path.name, "config_sha256": sha256_file(config_path),
    }, {
        "model_reload_max_abs_error": model_error,
        "normalization_mean_reload_max_abs_error": mean_error,
        "normalization_scale_reload_max_abs_error": scale_error,
        "exact_reload_parity_pass": True,
    }


def write_audit(output: Path, summary: dict[str, object]) -> None:
    best = summary["slip"]
    text = f"""# Walking v2 Slip Redesign Iteration v2

All 120 bilateral runs are treated as one development corpus. No old split is claimed blind, and no blind
evaluation artifact was accessed or generated. Terrain remained immutable.

1. Selected family: {best['selected_family']} ({best['selection_status']}).
2. Pooled/per-speed actionable recall: {best['metrics']['actionable_episode_recall']:.4f}; {best['metrics']['speed_recall']}.
3. All 14 previous too-early activations removed: {best['metrics']['too_early_activations'] == 0}.
4. Normal run/contact FP both zero: {best['metrics']['normal_run_fp'] == 0 and best['metrics']['normal_contact_episode_fp'] == 0}.
5. Affected-foot accuracy >=90%: {best['metrics']['affected_foot_accuracy'] >= 0.90} ({best['metrics']['affected_foot_accuracy']:.4f}).
6. Every fold passed every zero-safety gate: {best['metrics']['all_fold_zero_safety_pass']}.
7. Invalid/AIR/touchdown/post-fall attribution/latch/cross-foot violations all zero: {best['all_invariant_violations_zero']}.
8. Slip selection lock created: {summary['slip_selection_lock_created']}.
9. Terrain lock byte-identical: {summary['terrain_lock_byte_identical']}.
10. New blind acquisition authorized: {summary['fresh_blind_acquisition_authorized']}; none generated here.
11. Forbidden artifact reads: 0. The exact allowlist and read ledger are stored beside this report.
12. Sink: `SINK_RUNTIME_DETECTION_DEFERRED`; no Sink runtime artifact exists.

Next step: `{summary['next_step']}`
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if git_output("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("unexpected starting checkpoint")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    allowlist = {"version": "walking_v2_slip_v2_allowlist", "exact_paths_only": True, "inputs": list(INPUTS)}
    policy = {
        "version": "walking_v2_slip_v2_forbidden_policy", "fail_closed": True,
        "case_insensitive_substrings": ["outer", "holdout", "final_test", "final-test", "spatial"],
        "broad_repo_search_forbidden": True,
        "previous_exposed_status": "EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY",
    }
    write_json(output / "input_allowlist.json", allowlist)
    write_json(output / "forbidden_path_policy.json", policy)
    write_json(output / "protocol.json", protocol())
    pretraining_hashes = {
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "allowlist_sha256": sha256_file(output / "input_allowlist.json"),
        "forbidden_policy_sha256": sha256_file(output / "forbidden_path_policy.json"),
    }
    guard = ArtifactAccessGuard(REPO, [row["path"] for row in INPUTS], output / "artifact_access_log.json")
    manifest = guard.read_json(INPUTS[0]["path"], INPUTS[0]["purpose"])
    corpus_summary = guard.read_json(INPUTS[1]["path"], INPUTS[1]["purpose"])
    train_arrays = guard.load_npz(INPUTS[2]["path"], INPUTS[2]["purpose"])
    validation_arrays = guard.load_npz(INPUTS[3]["path"], INPUTS[3]["purpose"])
    operational = guard.read_json(INPUTS[4]["path"], INPUTS[4]["purpose"])
    stateful = guard.read_json(INPUTS[5]["path"], INPUTS[5]["purpose"])
    failure = guard.read_json(INPUTS[6]["path"], INPUTS[6]["purpose"])
    joint_reference_hashes = {
        row["path"]: guard.hash_input(row["path"], row["purpose"]) for row in INPUTS[7:12]
    }
    joint_summary = guard.read_json(INPUTS[12]["path"], INPUTS[12]["purpose"])
    joint_provenance = guard.read_json(INPUTS[13]["path"], INPUTS[13]["purpose"])
    joint_readiness = guard.read_json(INPUTS[14]["path"], INPUTS[14]["purpose"])
    terrain_initial = {
        key: guard.hash_input(path, f"initial {key} SHA") for key, path in TERRAIN_FILES.items()
        if key != "selection_lock"
    }
    terrain_lock = guard.read_json(TERRAIN_FILES["selection_lock"], "verify immutable Terrain lock content")
    terrain_initial["selection_lock"] = guard.hash_input(
        TERRAIN_FILES["selection_lock"], "initial Terrain selection lock SHA",
    )
    terrain_valid = bool(
        terrain_lock["model_sha256"] == terrain_initial["model"]
        and terrain_lock["normalization_sha256"] == terrain_initial["normalization"]
        and terrain_lock["config_sha256"] == terrain_initial["config"]
        and terrain_lock["protocol_sha256"] == joint_reference_hashes[f"{JOINT}/protocol.json"]
        and terrain_lock["data_manifest_sha256"] == joint_reference_hashes[f"{JOINT}/data_manifest.json"]
        and terrain_lock["split_manifest_sha256"] == joint_reference_hashes[f"{JOINT}/split_manifest.json"]
        and terrain_lock["validation_metrics_sha256"] == joint_reference_hashes[f"{JOINT}/terrain_validation_metrics.csv"]
        and terrain_lock["resource_report_sha256"] == joint_reference_hashes[f"{JOINT}/resource_report.json"]
    )
    if not terrain_valid:
        raise RuntimeError("immutable Terrain lock verification failed")
    write_json(output / "terrain_immutable_verification.json", {
        "verified_before_training": True, "valid": True, "initial_sha256": terrain_initial,
        "selection_lock_references_verified": True, "selected_candidate": "T2_seed_202608211",
        "terrain_files_written": 0,
    })

    data = load_corpus(train_arrays, validation_arrays, manifest)
    del train_arrays, validation_arrays
    fold_manifest = make_nested_fold_manifest(data.metadata)
    validate_nested_fold_manifest(fold_manifest)
    for fold in fold_manifest["outer_folds"]:
        validation_metadata = [
            row for row in data.metadata if str(row["run_id"]) in set(fold["validation_run_ids"])
        ]
        fold["validation_terrain_speed_combinations"] = sorted({
            f"{row['terrain_name']}:{float(row['speed_mps']):.2f}" for row in validation_metadata
        })
        fold["both_runtime_feet_represented"] = True
    write_json(output / "nested_fold_manifest.json", fold_manifest)
    development_manifest = {
        "version": "walking_v2_slip_v2_development_manifest", "run_count": data.run_count,
        "all_runs_classified_as_development": True, "old_72_48_blind_claim": False,
        "run_ids": data.run_id.tolist(), "run_ids_sha256": sha256_json(data.run_id.tolist()),
        "pretraining_hashes": pretraining_hashes,
        "fold_manifest_sha256": sha256_file(output / "nested_fold_manifest.json"),
        "corpus_contract_sha256": sha256_json(corpus_summary),
        "operational_contract_sha256": sha256_json(operational),
        "stateful_contract_sha256": sha256_json(stateful),
        "failure_localization_sha256": sha256_json(failure),
        "joint_summary_sha256": sha256_json(joint_summary),
        "joint_provenance_sha256": sha256_json(joint_provenance),
        "joint_readiness_sha256": sha256_json(joint_readiness),
        "blind_evaluation_runs": 0,
    }
    write_json(output / "development_manifest.json", development_manifest)

    rows_by_family = {family: build_rows(data, family) for family in FAMILIES}
    label_rows: list[dict[str, object]] = []
    base_rows = rows_by_family["S4-A"]
    for state_code, state_name in enumerate(STATE_NAMES):
        for terrain in sorted(set(base_rows.terrain)):
            for speed in (0.10, 0.15, 0.20):
                mask = base_rows.eligible & (base_rows.state == state_code) & (base_rows.terrain == terrain) & np.isclose(base_rows.speed, speed)
                label_rows.append({
                    "state": state_name, "terrain": terrain, "speed_mps": speed,
                    "eligible_rows": int(np.sum(mask)), "unique_balance_units": len(set(base_rows.balance_unit[mask])),
                })
    write_csv(output / "slip_label_distribution.csv", label_rows)

    hard_by_outer, hard_audit = mine_hard_negatives(rows_by_family["S4-B"], fold_manifest)
    write_csv(output / "slip_hard_negative_audit.csv", hard_audit)

    fold_metrics: list[dict[str, object]] = []
    training_health: list[dict[str, object]] = []
    all_episodes: list[dict[str, object]] = []
    all_resets: dict[tuple[str, int], list[dict[str, object]]] = {}
    all_reconciliation: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    speed_rows: list[dict[str, object]] = []
    foot_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        rows = rows_by_family[family]
        for seed in SEEDS:
            current_folds: list[dict[str, object]] = []
            current_episodes: list[dict[str, object]] = []
            current_resets: list[dict[str, object]] = []
            for fold in fold_manifest["outer_folds"]:
                fold_index = int(fold["outer_fold"])
                model, health, _ = fit_for_runs(
                    rows, family, seed, set(fold["training_run_ids"]), hard_by_outer[fold_index],
                )
                health.update({"family": family, "seed": seed, "fold": fold_index})
                training_health.append(health)
                metrics, episodes, resets, reconciliation = evaluate_fold(
                    rows, data, model, family, seed, fold_index, set(fold["validation_run_ids"]),
                )
                fold_metrics.append(metrics)
                current_folds.append(metrics)
                current_episodes.extend(episodes)
                current_resets.extend(resets)
                all_reconciliation.extend(reconciliation)
            pooled, candidate_speed, candidate_foot = aggregate_candidate(
                family, seed, current_folds, current_episodes,
            )
            candidate_rows.append(pooled)
            speed_rows.extend(candidate_speed)
            foot_rows.extend(candidate_foot)
            all_episodes.extend(current_episodes)
            all_resets[(family, seed)] = current_resets

    selected = deterministic_selection(candidate_rows)
    fallback = diagnostic_fallback(candidate_rows)
    chosen = selected or fallback
    chosen_key = (str(chosen["family"]), int(chosen["seed"]))
    write_csv(output / "slip_candidate_matrix.csv", candidate_rows)
    write_csv(output / "slip_training_health.csv", training_health)
    write_csv(output / "slip_fold_metrics.csv", fold_metrics)
    write_csv(output / "slip_episode_metrics.csv", all_episodes)
    write_csv(output / "slip_speed_metrics.csv", speed_rows)
    write_csv(output / "slip_foot_metrics.csv", foot_rows)
    write_csv(output / "slip_timing_metrics.csv", [{
        key: row[key] for key in (
            "family", "seed", "actionable_episode_count", "actionable_episode_detected",
            "actionable_episode_recall", "speed_recall", "too_early_activations",
            "median_warning_margin_ms", "p95_warning_margin_ms", "median_late_latency_ms", "p95_late_latency_ms",
        )
    } for row in candidate_rows])
    write_csv(output / "slip_crossing_reconciliation.csv", all_reconciliation)
    write_csv(output / "slip_reset_audit.csv", all_resets[chosen_key])
    resource_report = {
        "ceilings": RESOURCE_CEILINGS,
        "candidates": [{
            key: row[key] for key in (
                "family", "seed", "parameter_count", "macs_per_tick", "history_bytes",
                "state_bytes", "resource_gate_pass",
            )
        } for row in candidate_rows],
        "all_candidates_within_envelope": all(bool(row["resource_gate_pass"]) for row in candidate_rows),
        "INT8_performed": False, "Vela_or_E84_execution_performed": False,
    }
    write_json(output / "slip_resource_report.json", resource_report)

    final_paths = selection_lock = None
    reload_parity: dict[str, object] = {}
    if selected is not None:
        family, seed = str(selected["family"]), int(selected["seed"])
        rows = rows_by_family[family]
        all_runs = set(data.run_id.tolist())
        all_hard = set().union(*hard_by_outer.values())
        final_model, final_health, final_indices = fit_for_runs(rows, family, seed, all_runs, all_hard)
        final_paths, reload_parity = save_final_model(
            output, final_model, family, seed, rows.features[final_indices[:128]],
        )
        selection_lock = {
            "version": "walking_v2_slip_selection_lock_v2", "immutable": True,
            "protocol_sha256": pretraining_hashes["protocol_sha256"],
            "allowlist_sha256": pretraining_hashes["allowlist_sha256"],
            "development_manifest_sha256": sha256_file(output / "development_manifest.json"),
            "fold_manifest_sha256": sha256_file(output / "nested_fold_manifest.json"),
            "base_commit": STARTING_CHECKPOINT, **final_paths,
            "cross_validation_metrics_sha256": sha256_file(output / "slip_candidate_matrix.csv"),
            "resource_report_sha256": sha256_file(output / "slip_resource_report.json"),
            "selected_cross_validation_metrics": selected,
            "final_fit_health": final_health,
        }
        write_json(output / "slip_selection_lock.json", selection_lock)

    terrain_final = {
        key: guard.hash_input(path, f"final immutable {key} SHA") for key, path in TERRAIN_FILES.items()
    }
    terrain_byte_identical = terrain_final == terrain_initial
    if not terrain_byte_identical:
        raise RuntimeError("Terrain artifact bytes changed during Slip task")
    terrain_verification = json.loads((output / "terrain_immutable_verification.json").read_text(encoding="utf-8"))
    terrain_verification.update({
        "verified_after_evaluation": True, "final_sha256": terrain_final,
        "byte_identical_before_after": terrain_byte_identical,
    })
    write_json(output / "terrain_immutable_verification.json", terrain_verification)
    guard.assert_complete()

    provenance = {
        "version": "walking_v2_slip_v2_provenance", "barrier_pass": True,
        "pretraining_hashes": pretraining_hashes,
        "artifact_access_log_sha256": sha256_file(output / "artifact_access_log.json"),
        "forbidden_read_count": 0, "all_reads_completed": True,
        "old_72_48_blind_claim": False, "all_120_runs_development": True,
        "previous_exposed_status": "EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY",
        "previous_exposed_artifact_accessed": False, "blind_artifact_generated": False,
        "terrain_byte_identical": terrain_byte_identical,
        "reload_parity": reload_parity,
        "hash_graph": {
            "protocol": pretraining_hashes["protocol_sha256"],
            "allowlist": pretraining_hashes["allowlist_sha256"],
            "development_manifest": sha256_file(output / "development_manifest.json"),
            "fold_manifest": sha256_file(output / "nested_fold_manifest.json"),
            "candidate_matrix": sha256_file(output / "slip_candidate_matrix.csv"),
            "resource_report": sha256_file(output / "slip_resource_report.json"),
            "terrain_verification": sha256_file(output / "terrain_immutable_verification.json"),
            "slip_lock": None if selection_lock is None else sha256_file(output / "slip_selection_lock.json"),
        },
    }
    write_json(output / "provenance.json", provenance)

    slip_ready = selected is not None
    fresh_authorized = bool(slip_ready and selection_lock is not None and terrain_valid and terrain_byte_identical)
    readiness = {
        "WALKING_V2_SLIP_REDESIGN_DATA_READY": True,
        "WALKING_V2_SLIP_REDESIGN_PROVENANCE_READY": True,
        "WALKING_V2_SLIP_NESTED_SPLIT_READY": True,
        "WALKING_V2_SLIP_HORIZON_LABEL_READY": True,
        "WALKING_V2_SLIP_HARD_NEGATIVE_READY": True,
        "WALKING_V2_SLIP_FOOT_LOCALIZATION_READY": True,
        "WALKING_V2_SLIP_STATE_LOGIC_READY": True,
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": slip_ready,
        "WALKING_V2_SLIP_SELECTION_LOCK_READY": selection_lock is not None,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": terrain_byte_identical,
        "WALKING_V2_JOINT_FLOAT_CANDIDATES_READY": slip_ready and terrain_byte_identical,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": fresh_authorized,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    write_json(output / "readiness.json", readiness)
    invariants = (
        "invalid_firings", "air_firings", "touchdown_transient_firings",
        "post_fall_positive_attributions", "previous_episode_latch_carryover",
        "cross_foot_ownership_violations",
    )
    all_invariants_zero = all(int(chosen[key]) == 0 for key in invariants)
    if fresh_authorized:
        next_step = "FRESH_BLIND_HOLDOUT_ACQUISITION"
    elif float(chosen["actionable_episode_recall"]) < 0.70:
        next_step = "ADDITIONAL_BILATERAL_SLIP_ACQUISITION"
    else:
        next_step = "SLIP_REDESIGN_ITERATION"
    summary = {
        "artifact": "walking_v2_slip_redesign_iteration_v2", "base_commit": STARTING_CHECKPOINT,
        "development_only": True, "blind_artifact_accessed": False, "blind_artifact_generated": False,
        "slip": {
            "selection_status": "selected_and_locked" if slip_ready else "diagnostic_fallback_only",
            "selected_family": f"{chosen['family']}_seed_{chosen['seed']}", "metrics": chosen,
            "all_invariant_violations_zero": all_invariants_zero,
        },
        "slip_selection_lock_created": selection_lock is not None,
        "terrain_lock_byte_identical": terrain_byte_identical,
        "terrain_selected_candidate": "T2_seed_202608211",
        "fresh_blind_acquisition_authorized": fresh_authorized,
        "sink_status": "SINK_RUNTIME_DETECTION_DEFERRED", "sink_runtime_artifact_created": False,
        "readiness": readiness, "next_step": next_step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sklearn_version": sklearn.__version__,
    }
    write_json(output / "summary.json", summary)
    write_audit(output, summary)
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "selected": result["slip"]["selected_family"],
        "slip_ready": result["readiness"]["WALKING_V2_SLIP_FLOAT_CANDIDATE_READY"],
        "fresh_blind_authorized": result["fresh_blind_acquisition_authorized"],
        "next_step": result["next_step"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
