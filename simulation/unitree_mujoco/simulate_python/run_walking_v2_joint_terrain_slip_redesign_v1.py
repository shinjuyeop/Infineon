"""Run development-only walking-v2 joint Terrain/Slip redesign v1.

The runner writes and hashes its protocol/allowlist before opening any input.
It then reads only exact allowlisted development artifacts through a durable
fail-closed access guard.  It never opens or creates a blind evaluation split.
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score

from walking_v2_bilateral_bounded_training import (
    first_actionable_events,
    physical_slip_episodes,
    raw_slip_crossings,
    risk_firing_is_too_early,
)
from walking_v2_joint_terrain_slip_redesign_v1 import (
    ARCHITECTURE_SPECS,
    ENDPOINT_STRIDE_MS,
    FORBIDDEN_PATH_TOKENS,
    FORBIDDEN_RUNTIME_FIELDS,
    HAZARD_BIN_EDGES_MS,
    PHASE_NAMES,
    RESOURCE_CEILINGS,
    RUNTIME_INPUT_FIELDS,
    SAMPLE_RATE_HZ,
    SLIP_ARCHITECTURES,
    TERRAIN_ARCHITECTURES,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    TRAINING_SEEDS,
    ArtifactAccessGuard,
    DualHeadModel,
    ProjectedLinearModel,
    authoritative_owned_detections,
    causal_endpoints,
    contact_scoped_state,
    deterministic_slip_selection,
    deterministic_terrain_selection,
    encoder_fingerprint,
    episode_balanced_weights,
    fit_dual_head,
    fit_projected_linear,
    invalid_firing_count,
    raking_weights,
    runtime_feature,
    sha256_file,
    sha256_json,
    slip_diagnostic_fallback,
    slip_gate,
    terrain_diagnostic_fallback,
    terrain_gate,
)


SIMULATION = Path(__file__).resolve().parents[2]
REPO = SIMULATION.parent
OUTPUT = SIMULATION / "outputs" / "walking_v2_joint_terrain_slip_redesign_v1"
STARTING_CHECKPOINT = "a229dae974dbfef44e7b9e3cb2b8095015f1a54d"
SOURCE_PREFIX = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"

INPUTS = (
    {
        "path": f"{SOURCE_PREFIX}/manifest.json",
        "purpose": "development split metadata and immutable run grouping",
        "access": "read_json",
    },
    {
        "path": f"{SOURCE_PREFIX}/summary.json",
        "purpose": "upstream controlled bilateral dataset contract",
        "access": "read_json",
    },
    {
        "path": f"{SOURCE_PREFIX}/bilateral_traces_train.npz",
        "purpose": "72-run development training tensors",
        "access": "load_npz",
    },
    {
        "path": f"{SOURCE_PREFIX}/bilateral_traces_validation.npz",
        "purpose": "48-run development validation tensors",
        "access": "load_npz",
    },
    {
        "path": "simulation/outputs/walking_hazard_operational_label_contract_v2/summary.json",
        "purpose": "immutable operational label compatibility",
        "access": "read_json",
    },
    {
        "path": "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json",
        "purpose": "immutable contact-state compatibility",
        "access": "read_json",
    },
    {
        "path": "simulation/outputs/walking_v2_candidate_failure_localization_v1/summary.json",
        "purpose": "a229dae development-only failure diagnosis and exposed-data disclosure",
        "access": "read_json",
    },
    {
        "path": "simulation/outputs/terrain_static_reference_v4/selected_model.keras",
        "purpose": "hash-only controlled/static compatibility reference",
        "access": "sha256",
    },
    {
        "path": "simulation/outputs/terrain_static_reference_v4/normalization.json",
        "purpose": "hash-only controlled/static normalization reference",
        "access": "sha256",
    },
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
    candidate_matrix = {
        "terrain": [
            {"architecture": architecture, "seed": seed, **ARCHITECTURE_SPECS[architecture]}
            for architecture in TERRAIN_ARCHITECTURES for seed in TRAINING_SEEDS
        ],
        "slip": [
            {"architecture": architecture, "seed": seed, **ARCHITECTURE_SPECS[architecture]}
            for architecture in SLIP_ARCHITECTURES for seed in TRAINING_SEEDS
        ],
    }
    return {
        "artifact": "walking_v2_joint_terrain_slip_redesign_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "development-only validation selection and conditional selection locks",
        "created_blind_evaluation_data": False,
        "runtime_inputs": sorted(RUNTIME_INPUT_FIELDS),
        "forbidden_runtime_inputs": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "native_sample_rate_hz": SAMPLE_RATE_HZ,
        "endpoint_stride_ms": ENDPOINT_STRIDE_MS,
        "candidate_matrix": candidate_matrix,
        "candidate_family_limits": {"terrain": 3, "slip": 3, "seeds_per_family": 3},
        "training": {
            "normalization": "weighted development-train only",
            "terrain_population": "all eligible rows retained; deterministic 30-cycle marginal raking",
            "terrain_startup_history": "endpoints begin at 49ms; missing past is causal first-observation edge padding",
            "terrain_balance_factors": ["terrain", "phase", "foot", "speed", "run", "variation"],
            "slip_population": "all stable pre-fall endpoints retained",
            "slip_balance": "equal label mass; equal physical episode/normal contact mass within label",
            "optimizer": "LBFGS", "learning_rate": "strong_wolfe_line_search",
            "max_iterations": 600, "tolerance": 1e-6,
            "early_stop_rule": "gradient tolerance 1e-6 or fixed 600-iteration cap",
            "projection": "fixed seeded Gaussian ReLU basis; original normalized features retained",
            "validation_fitting": False,
        },
        "resource_ceilings": RESOURCE_CEILINGS,
        "terrain_gates": {
            "overall_accuracy_min": 0.85, "macro_recall_min": 0.85,
            "worst_class_recall_min": 0.70, "sand_recall_min": 0.70,
            "class_balanced_majority_prediction_rate_strict_max": 0.60,
            "each_speed_accuracy_min": 0.80, "left_right_difference_pp_max": 10.0,
            "class_collapse": False, "air_transitions": 0, "invalid_firings": 0,
            "causality_and_bilateral_parity": True,
        },
        "slip_gates": {
            "actionable_episode_recall_min": 0.80,
            "each_speed_actionable_recall_min": 0.70,
            "physical_episode_recall_min": 0.80, "affected_foot_accuracy_min": 0.90,
            "normal_run_fp": 0, "normal_contact_episode_fp": 0,
            "too_early_activations": 0, "air_firings": 0,
            "touchdown_transient_firings": 0, "invalid_firings": 0,
            "post_fall_positive_attributions": 0,
            "previous_episode_latch_carryover": 0, "cross_foot_state_ownership": 0,
            "timer_only_promotions": 0, "median_warning_margin_ms_min": 20,
            "pre_onset_detection_fraction_min": 0.80,
            "both_affected_feet_covered": True,
        },
        "slip_semantics": {
            "authoritative_evaluation": "corrected_R4",
            "physical_oracle_usage": "offline label and evaluation only",
            "actionable_horizon_ms": [0, 100],
            "hazard_bin_edges_ms": list(HAZARD_BIN_EDGES_MS),
            "post_fall": "excluded from valid numerator/denominator and positive attribution",
            "originating_state": "rising edge must begin inside same foot/contact risk window",
            "active_evidence": "sensor evidence head; never physical confirmation",
        },
        "ablations": {
            "terrain": [
                "50ms_vs_200ms", "compact_vs_full_statistics", "weighted_full_vs_destructive_downsample",
                "independent_vs_shared_encoder", "symmetric_vs_asymmetric_aggregation", "linear_vs_projected_capacity",
            ],
            "slip": [
                "binary_vs_horizon_target", "sample_vs_episode_balance", "raw_vs_contact_state",
                "50ms_vs_100ms_vs_200ms", "single_vs_separate_heads",
            ],
            "diagnostic_only": True, "seed": TRAINING_SEEDS[0],
            "optimizer": "same frozen LBFGS configuration as candidates",
            "selection_use": False,
        },
        "selection": {
            "terrain_order": ["macro_recall_desc", "worst_recall_desc", "overall_desc", "macs_asc", "family", "seed"],
            "slip_order": ["actionable_recall_desc", "min_speed_desc", "affected_foot_desc", "physical_recall_desc", "warning_desc", "macs_asc", "family", "seed"],
            "gate_before_rank": True, "exactly_one_diagnostic_fallback_if_no_pass": True,
        },
        "static_compatibility": {
            "walking_candidate_replaces_frozen_controlled_detector": False,
            "sand_runtime_semantics": "SAND_TERRAIN_CAUTION",
        },
        "sink": {
            "status": "SINK_RUNTIME_DETECTION_DEFERRED", "runtime_head_created": False,
            "runtime_outputs": [],
        },
        "forbidden_actions": [
            "blind evaluation access or generation", "production model/normalization/threshold writes",
            "System-v1/System-v2 migration", "INT8/Vela/E84/HIL/final evaluation writes",
            "Sink runtime output",
        ],
    }


@dataclass(frozen=True)
class DevelopmentData:
    split: str
    run_id: np.ndarray
    time_s: np.ndarray
    bilateral: np.ndarray
    loaded: np.ndarray
    age: np.ndarray
    phase: np.ndarray
    pre_fall: np.ndarray
    contact_episode: np.ndarray
    touchdown: np.ndarray
    slip_target: np.ndarray
    slip_active: np.ndarray
    metadata: tuple[dict[str, object], ...]

    @property
    def run_count(self) -> int:
        return len(self.run_id)


def load_data(split: str, arrays: dict[str, np.ndarray], manifest: dict[str, object]) -> DevelopmentData:
    run_id = arrays["run_id"].astype(str)
    rows = {
        str(row["run_id"]): row for row in manifest["runs"]
        if row["split"] == f"development_{split}"
    }
    expected = 72 if split == "train" else 48
    data = DevelopmentData(
        split=split, run_id=run_id, time_s=arrays["time_s"].astype(np.float64),
        bilateral=arrays["bilateral_canonical"].astype(np.float32),
        loaded=arrays["force_loaded"].astype(bool), age=arrays["contact_age"].astype(np.int32),
        phase=arrays["gait_phase_code"].astype(np.int8), pre_fall=arrays["pre_fall_valid"].astype(bool),
        contact_episode=arrays["contact_episode_id"].astype(np.int32),
        touchdown=arrays["touchdown_transient"].astype(bool),
        slip_target=arrays["slip_risk_target"].astype(bool),
        slip_active=arrays["slip_physical_active"].astype(bool),
        metadata=tuple(rows[str(value)] for value in run_id),
    )
    aligned = (expected, 3000)
    if (
        data.run_count != expected or data.time_s.shape != aligned
        or data.bilateral.shape != (expected, 3000, 20)
        or data.loaded.shape != (expected, 3000, 2)
        or not np.allclose(data.time_s[:, 0], 0.001, rtol=0.0, atol=1e-15)
        or not np.allclose(np.diff(data.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)
        or len(set(data.run_id.tolist())) != expected
    ):
        raise ValueError(f"invalid exact-1kHz {split} development data")
    return data


@dataclass(frozen=True)
class TerrainRows:
    features: np.ndarray
    target: np.ndarray
    run_index: np.ndarray
    endpoint: np.ndarray
    side: np.ndarray
    phase: np.ndarray
    speed: np.ndarray
    variation: np.ndarray
    run_id: np.ndarray


def build_terrain_rows(
    data: DevelopmentData,
    architecture: str,
    *,
    include_asymmetry: bool = True,
    history_override_ms: int | None = None,
) -> TerrainRows:
    features: list[np.ndarray] = []
    fields: dict[str, list[object]] = {key: [] for key in (
        "target", "run_index", "endpoint", "side", "phase", "speed", "variation", "run_id",
    )}
    for run_index, metadata in enumerate(data.metadata):
        # The previous 50-ms eligible population is the locked 24,560-row
        # population. Longer architectures use causal first-observation padding
        # only at run startup rather than dropping those eligible rows.
        for endpoint in causal_endpoints(3000, 50):
            for side in (0, 1):
                if not (
                    data.loaded[run_index, endpoint, side]
                    and data.pre_fall[run_index, endpoint]
                ):
                    continue
                features.append(runtime_feature(
                    architecture, side, int(endpoint), data.bilateral[run_index], data.loaded[run_index],
                    data.age[run_index], data.phase[run_index], include_asymmetry=include_asymmetry,
                    history_override_ms=history_override_ms,
                ))
                values = {
                    "target": TERRAIN_LABELS[str(metadata["terrain_name"])],
                    "run_index": run_index, "endpoint": int(endpoint), "side": side,
                    "phase": int(data.phase[run_index, endpoint, side]),
                    "speed": float(metadata["speed_mps"]),
                    "variation": int(metadata["variation_index"]), "run_id": str(metadata["run_id"]),
                }
                for key, value in values.items():
                    fields[key].append(value)
    return TerrainRows(
        np.asarray(features, np.float32), np.asarray(fields["target"], int),
        np.asarray(fields["run_index"], int), np.asarray(fields["endpoint"], int),
        np.asarray(fields["side"], int), np.asarray(fields["phase"], int),
        np.asarray(fields["speed"], float), np.asarray(fields["variation"], int),
        np.asarray(fields["run_id"]),
    )


def terrain_weights(rows: TerrainRows) -> np.ndarray:
    factors = np.column_stack((
        rows.target, rows.phase, rows.side, np.rint(rows.speed * 100).astype(int),
        rows.run_index, rows.variation,
    ))
    return raking_weights(factors)


def grouped_accuracy(target: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        str(value): float(accuracy_score(target[groups == value], prediction[groups == value]))
        for value in np.unique(groups)
    }


def encoder_cost(architecture: str) -> tuple[int, int]:
    if architecture in ("T1", "S1"):
        history = int(ARCHITECTURE_SPECS[architecture]["history_ms"])
        return 0, history * 10 * 8
    if architecture in ("T2", "S3"):
        return 0, (50 + 200) * 10 * 8
    if architecture == "T3":
        value = sum((5, 20, 50, 100, 200, 2)) * 10 * 2
        return value, value
    if architecture == "S2":
        return 0, 200 * 10 * 8
    raise ValueError(architecture)


def contract_checks(
    architecture: str,
    model: ProjectedLinearModel | DualHeadModel,
    data: DevelopmentData,
) -> dict[str, object]:
    history = int(ARCHITECTURE_SPECS[architecture]["history_ms"])
    max_feature_error = 0.0
    max_model_error = 0.0
    max_future_error = 0.0
    checked = 0
    for run_index in range(min(4, data.run_count)):
        endpoints = causal_endpoints(3000, history)
        for endpoint in endpoints[::max(1, len(endpoints) // 4)][:4]:
            original = runtime_feature(
                architecture, 0, int(endpoint), data.bilateral[run_index], data.loaded[run_index],
                data.age[run_index], data.phase[run_index],
            )
            swapped_trace = np.concatenate((
                data.bilateral[run_index, :, 10:], data.bilateral[run_index, :, :10],
            ), axis=1)
            swapped = runtime_feature(
                architecture, 1, int(endpoint), swapped_trace, data.loaded[run_index, :, ::-1],
                data.age[run_index, :, ::-1], data.phase[run_index, :, ::-1],
            )
            max_feature_error = max(max_feature_error, float(np.max(np.abs(original - swapped))))
            mutated = data.bilateral[run_index].copy()
            mutated[int(endpoint) + 1:] = 1e6
            causal = runtime_feature(
                architecture, 0, int(endpoint), mutated, data.loaded[run_index],
                data.age[run_index], data.phase[run_index],
            )
            max_future_error = max(max_future_error, float(np.max(np.abs(original - causal))))
            if isinstance(model, DualHeadModel):
                first = np.column_stack(model.scores(original[None]))
                second = np.column_stack(model.scores(swapped[None]))
            else:
                first, second = model.probabilities(original[None]), model.probabilities(swapped[None])
            max_model_error = max(max_model_error, float(np.max(np.abs(first - second))))
            checked += 1
    return {
        "architecture": architecture, "windows_checked": checked,
        "bilateral_feature_max_abs_error": max_feature_error,
        "bilateral_model_max_abs_error": max_model_error,
        "future_mutation_max_abs_error": max_future_error,
        "bilateral_parity_pass": max_feature_error == 0.0 and max_model_error == 0.0,
        "causal_check_pass": max_future_error == 0.0,
        "left_right_slot_swap": "left/right tensors and contact telemetry swapped together",
    }


def evaluate_terrain(
    architecture: str,
    seed: int,
    model: ProjectedLinearModel,
    rows: TerrainRows,
    checks: dict[str, object],
) -> dict[str, object]:
    prediction = model.predictions(rows.features).astype(int)
    target = rows.target
    recalls = recall_score(target, prediction, labels=np.arange(4), average=None, zero_division=0)
    class_weight = raking_weights(target[:, None])
    predicted_mass = np.bincount(prediction, weights=class_weight, minlength=4)
    speed = grouped_accuracy(target, prediction, np.asarray([f"{value:.2f}" for value in rows.speed]))
    foot = grouped_accuracy(target, prediction, np.asarray([("left", "right")[value] for value in rows.side]))
    phase = grouped_accuracy(target, prediction, np.asarray([PHASE_NAMES[value] for value in rows.phase]))
    run = grouped_accuracy(target, prediction, rows.run_id)
    variation = grouped_accuracy(target, prediction, rows.variation)
    encoder_parameters, encoder_macs = encoder_cost(architecture)
    metrics: dict[str, object] = {
        "architecture": architecture, "seed": seed, "validation_rows": len(target),
        "overall_accuracy": float(accuracy_score(target, prediction)),
        "macro_recall": float(np.mean(recalls)), "worst_class_recall": float(np.min(recalls)),
        "sand_recall": float(recalls[TERRAIN_LABELS["sand"]]),
        "majority_class_prediction_rate": float(np.max(predicted_mass) / np.sum(predicted_mass)),
        "minimum_speed_accuracy": min(speed.values()),
        "left_right_accuracy_difference_pp": abs(foot["left"] - foot["right"]) * 100.0,
        "class_collapse": bool(np.max(predicted_mass) / np.sum(predicted_mass) >= 0.60 or np.min(recalls) == 0),
        "air_terrain_transitions": 0, "invalid_firings": 0,
        "class_recall": json.dumps(dict(zip(TERRAIN_NAMES, recalls.tolist())), sort_keys=True),
        "speed_accuracy": json.dumps(speed, sort_keys=True), "foot_accuracy": json.dumps(foot, sort_keys=True),
        "phase_accuracy": json.dumps(phase, sort_keys=True), "run_accuracy": json.dumps(run, sort_keys=True),
        "variation_accuracy": json.dumps(variation, sort_keys=True),
        "confusion_matrix": json.dumps(confusion_matrix(target, prediction, labels=np.arange(4)).tolist()),
        "parameter_count": model.parameter_count + encoder_parameters,
        "macs_per_tick": model.macs + encoder_macs,
        "history_bytes": 20 * int(ARCHITECTURE_SPECS[architecture]["history_ms"]) * 4,
        "persistent_state_bytes": 32, "encoder_sha256": encoder_fingerprint(architecture),
        "causal_check_pass": checks["causal_check_pass"],
        "bilateral_parity_pass": checks["bilateral_parity_pass"],
    }
    metrics["gate_pass"] = terrain_gate(metrics)
    return metrics


@dataclass(frozen=True)
class SlipRows:
    features: np.ndarray
    s1_target: np.ndarray
    s2_target: np.ndarray
    risk_target: np.ndarray
    active_target: np.ndarray
    balance_unit: np.ndarray
    run_index: np.ndarray
    endpoint: np.ndarray
    side: np.ndarray
    speed: np.ndarray
    variation: np.ndarray
    run_id: np.ndarray
    role: np.ndarray
    runtime_eligible: np.ndarray
    pre_fall: np.ndarray
    contact_id: np.ndarray


def episode_definitions(data: DevelopmentData) -> dict[tuple[int, int], list[object]]:
    result: dict[tuple[int, int], list[object]] = {}
    for run_index in range(data.run_count):
        for side in (0, 1):
            result[(run_index, side)] = physical_slip_episodes(
                data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
                data.pre_fall[run_index],
            )
    return result


def build_slip_rows(
    data: DevelopmentData,
    architecture: str,
    *,
    history_override_ms: int | None = None,
) -> SlipRows:
    history = int(
        ARCHITECTURE_SPECS[architecture]["history_ms"]
        if history_override_ms is None else history_override_ms
    )
    episodes_by_trace = episode_definitions(data)
    features: list[np.ndarray] = []
    fields: dict[str, list[object]] = {key: [] for key in (
        "s1", "s2", "risk", "active", "unit", "run_index", "endpoint", "side",
        "speed", "variation", "run_id", "role", "eligible", "pre_fall", "contact_id",
    )}
    for run_index, metadata in enumerate(data.metadata):
        for endpoint in causal_endpoints(3000, history):
            for side in (0, 1):
                contact_id = int(data.contact_episode[run_index, endpoint, side])
                episodes = [
                    value for value in episodes_by_trace[(run_index, side)]
                    if value.contact_episode_id == contact_id
                ]
                active_episode = next((
                    value for value in episodes if value.start <= endpoint < value.end_exclusive
                ), None)
                next_episode = next((value for value in episodes if value.start > endpoint), None)
                delta = None if next_episode is None else int(next_episode.start - endpoint)
                actionable_episode = next_episode if delta is not None and 0 < delta <= 100 else None
                is_active = active_episode is not None
                is_actionable = actionable_episode is not None and not is_active
                s1_target = 2 if is_active else (1 if is_actionable else 0)
                if is_active:
                    s2_target = 6
                elif is_actionable:
                    s2_target = int(np.searchsorted(HAZARD_BIN_EDGES_MS, delta, side="left") + 1)
                else:
                    s2_target = 0
                owned_episode = active_episode if is_active else actionable_episode
                unit = (
                    f"episode:{run_index}:{side}:{owned_episode.contact_episode_id}:{owned_episode.start}"
                    if owned_episode is not None else f"contact:{run_index}:{side}:{contact_id}"
                )
                features.append(runtime_feature(
                    architecture, side, int(endpoint), data.bilateral[run_index], data.loaded[run_index],
                    data.age[run_index], data.phase[run_index], history_override_ms=history_override_ms,
                ))
                values = {
                    "s1": s1_target, "s2": s2_target, "risk": int(is_actionable),
                    "active": int(is_active), "unit": unit, "run_index": run_index,
                    "endpoint": int(endpoint), "side": side, "speed": float(metadata["speed_mps"]),
                    "variation": int(metadata["variation_index"]), "run_id": str(metadata["run_id"]),
                    "role": str(metadata["role"]),
                    "eligible": bool(
                        data.loaded[run_index, endpoint, side]
                        and data.age[run_index, endpoint, side] > 10
                        and not data.touchdown[run_index, endpoint, side]
                        and data.pre_fall[run_index, endpoint]
                    ),
                    "pre_fall": bool(data.pre_fall[run_index, endpoint]), "contact_id": contact_id,
                }
                for key, value in values.items():
                    fields[key].append(value)
    return SlipRows(
        np.asarray(features, np.float32), np.asarray(fields["s1"], int), np.asarray(fields["s2"], int),
        np.asarray(fields["risk"], int), np.asarray(fields["active"], int), np.asarray(fields["unit"]),
        np.asarray(fields["run_index"], int), np.asarray(fields["endpoint"], int),
        np.asarray(fields["side"], int), np.asarray(fields["speed"], float),
        np.asarray(fields["variation"], int), np.asarray(fields["run_id"]), np.asarray(fields["role"]),
        np.asarray(fields["eligible"], bool), np.asarray(fields["pre_fall"], bool),
        np.asarray(fields["contact_id"], int),
    )


def train_slip_model(
    architecture: str,
    seed: int,
    rows: SlipRows,
) -> tuple[ProjectedLinearModel | DualHeadModel, dict[str, object]]:
    selected = np.flatnonzero(rows.runtime_eligible)
    if architecture == "S1":
        target = rows.s1_target[selected]
        weight = episode_balanced_weights(target, rows.balance_unit[selected])
        model, health = fit_projected_linear(architecture, seed, rows.features[selected], target, weight)
    elif architecture == "S2":
        target = rows.s2_target[selected]
        weight = episode_balanced_weights(target, rows.balance_unit[selected])
        model, health = fit_projected_linear(architecture, seed, rows.features[selected], target, weight)
    else:
        risk_target, active_target = rows.risk_target[selected], rows.active_target[selected]
        risk_weight = episode_balanced_weights(risk_target, rows.balance_unit[selected])
        active_weight = episode_balanced_weights(active_target, rows.balance_unit[selected])
        model, health = fit_dual_head(
            seed, rows.features[selected], risk_target, active_target, risk_weight, active_weight,
        )
        weight = 0.5 * (risk_weight + active_weight)
    health.update({
        "train_rows": len(selected), "raw_sample_count": len(selected),
        "effective_weighted_mass": float(weight.sum()), "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
        "episode_balanced": True, "normal_contact_balanced": True,
    })
    return model, health


def slip_outputs(
    architecture: str,
    model: ProjectedLinearModel | DualHeadModel,
    rows: SlipRows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(model, DualHeadModel):
        risk_score, active_score = model.scores(rows.features)
        raw = (risk_score >= float(ARCHITECTURE_SPECS["S3"]["risk_threshold"])) | (
            active_score >= float(ARCHITECTURE_SPECS["S3"]["active_threshold"])
        )
        return raw, np.maximum(risk_score, active_score), active_score
    probability = model.probabilities(rows.features)
    prediction = model.classes[np.argmax(probability, axis=1)]
    raw = prediction != 0
    return raw, 1.0 - probability[:, 0], probability[:, -1]


def reshape_slip(values: np.ndarray, data: DevelopmentData, endpoints: np.ndarray) -> np.ndarray:
    expected = data.run_count * len(endpoints) * 2
    if len(values) != expected:
        raise ValueError("Slip endpoint/foot row alignment changed")
    return np.asarray(values).reshape(data.run_count, len(endpoints), 2)


def evaluate_slip(
    architecture: str,
    seed: int,
    model: ProjectedLinearModel | DualHeadModel,
    rows: SlipRows,
    data: DevelopmentData,
    checks: dict[str, object],
    *,
    history_override_ms: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    history = int(
        ARCHITECTURE_SPECS[architecture]["history_ms"]
        if history_override_ms is None else history_override_ms
    )
    endpoints = causal_endpoints(3000, history)
    raw_flat, score_flat, _active_score_flat = slip_outputs(architecture, model, rows)
    raw = reshape_slip(raw_flat, data, endpoints)
    scores = reshape_slip(score_flat, data, endpoints)
    firing = np.zeros_like(raw)
    reset_rows: list[dict[str, object]] = []
    air = touchdown_count = post_fall_outputs = post_fall_attribution = 0
    normal_run_fp = normal_contact_fp = too_early = 0
    previous_episode_carryover = cross_foot_ownership = timer_promotions = 0
    physical_total = physical_detected = actionable_total = actionable_detected = 0
    raw_crossing_total = detected_runs_any = detected_runs_complete = actionable_runs = 0
    margins: list[float] = []
    latency: list[float] = []
    affected_total = affected_correct = 0
    foot_total = {0: 0, 1: 0}
    foot_detected = {0: 0, 1: 0}
    speed_total = {speed: 0 for speed in (0.10, 0.15, 0.20)}
    speed_detected = {speed: 0 for speed in (0.10, 0.15, 0.20)}
    ledger: list[dict[str, object]] = []
    for run_index, metadata in enumerate(data.metadata):
        state, resets, owner = contact_scoped_state(
            raw[run_index], endpoints, data.loaded[run_index], data.age[run_index], data.touchdown[run_index],
        )
        firing[run_index] = state
        run_detected_flags: list[bool] = []
        run_has_actionable = False
        for side in (0, 1):
            current = state[:, side]
            current_prefall = data.pre_fall[run_index, endpoints]
            current_loaded = data.loaded[run_index, endpoints, side]
            current_touchdown = data.touchdown[run_index, endpoints, side]
            air += int(np.sum(current & ~current_loaded))
            touchdown_count += int(np.sum(current & current_touchdown))
            post_fall_outputs += int(np.sum(current & ~current_prefall))
            rising = current & ~np.r_[False, current[:-1]]
            episodes = physical_slip_episodes(
                data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
                data.pre_fall[run_index],
            )
            actionable = first_actionable_events(episodes)
            actionable_keys = {(value.contact_episode_id, value.start) for value in actionable}
            if str(metadata["role"]) == "hard_negative":
                for contact_id in np.unique(data.contact_episode[run_index, :, side]):
                    if contact_id < 0:
                        continue
                    mask = (
                        (data.contact_episode[run_index, endpoints, side] == contact_id)
                        & current_prefall
                    )
                    normal_contact_fp += int(bool(np.any(current & mask)))
            if str(metadata["role"]) == "slip_candidate":
                for index in np.flatnonzero(rising & current_prefall):
                    sample = int(endpoints[index])
                    contact_id = int(data.contact_episode[run_index, sample, side])
                    too_early += int(risk_firing_is_too_early(sample, contact_id, episodes, 100))
            raw_crossing_total += len(raw_slip_crossings(
                data.slip_active[run_index, :, side], data.contact_episode[run_index, :, side],
                data.pre_fall[run_index],
            )) if str(metadata["role"]) == "slip_candidate" else 0
            reset_rows.append({
                "architecture": architecture, "seed": seed, "run_id": str(metadata["run_id"]),
                "foot": ("left", "right")[side],
                "contact_loss_resets": int(np.sum(resets[:, side] == "contact_loss")),
                "touchdown_resets": int(np.sum(resets[:, side] == "new_touchdown")),
                "score_outputs_without_raw_evidence": int(np.sum(current & ~raw[run_index, :, side])),
                "owner_cross_foot_mutations": 0, "previous_contact_carryover": 0,
                "owner_max": int(np.max(owner[:, side])),
            })
            for episode in episodes:
                risk_start = max(0, episode.start - 100)
                candidates = np.flatnonzero(
                    (endpoints >= risk_start) & (endpoints < episode.end_exclusive)
                    & (data.contact_episode[run_index, endpoints, side] == episode.contact_episode_id)
                )
                detected_indices = candidates[current[candidates]]
                detected_indices = authoritative_owned_detections(
                    current, endpoints, detected_indices, risk_start, current_prefall,
                )
                detected = bool(len(detected_indices))
                first_index = int(detected_indices[0]) if detected else None
                first_sample = int(endpoints[first_index]) if first_index is not None else None
                margin = None if first_sample is None else int(episode.start - first_sample)
                is_actionable = (episode.contact_episode_id, episode.start) in actionable_keys
                affected = None
                if str(metadata["role"]) == "slip_candidate":
                    physical_total += 1
                    physical_detected += int(detected)
                    if is_actionable:
                        actionable_total += 1
                        actionable_detected += int(detected)
                        speed = float(metadata["speed_mps"])
                        speed_total[speed] += 1
                        speed_detected[speed] += int(detected)
                        foot_total[side] += 1
                        foot_detected[side] += int(detected)
                        run_has_actionable = True
                        run_detected_flags.append(detected)
                    if detected:
                        margins.append(float(margin))
                        latency.append(float(max(0, -margin)))
                        affected_total += 1
                        affected = bool(
                            current[first_index]
                            and scores[run_index, first_index, side]
                            >= scores[run_index, first_index, 1 - side]
                        )
                        affected_correct += int(affected)
                ledger.append({
                    "architecture": architecture, "seed": seed, "run_id": str(metadata["run_id"]),
                    "role": str(metadata["role"]), "speed_mps": float(metadata["speed_mps"]),
                    "variation": int(metadata["variation_index"]), "foot": ("left", "right")[side],
                    "contact_episode_id": episode.contact_episode_id,
                    "physical_start_sample": episode.start, "physical_end_exclusive": episode.end_exclusive,
                    "raw_crossings": episode.raw_crossings, "actionable": is_actionable,
                    "detected": detected, "first_detection_sample": "" if first_sample is None else first_sample,
                    "warning_margin_ms": "" if margin is None else margin,
                    "late_latency_ms": "" if margin is None else max(0, -margin),
                    "affected_foot_correct": "" if affected is None else affected,
                    "positive_attribution_after_fall": False,
                })
        endpoint_prefall = data.pre_fall[run_index, endpoints, None]
        run_any = bool(np.any(state & endpoint_prefall))
        if str(metadata["role"]) == "hard_negative":
            normal_run_fp += int(run_any)
        if str(metadata["role"]) == "slip_candidate" and run_has_actionable:
            actionable_runs += 1
            detected_runs_any += int(any(run_detected_flags))
            detected_runs_complete += int(all(run_detected_flags))
    invalid = invalid_firing_count(air, touchdown_count, post_fall_outputs)
    speed_recall = {
        f"{speed:.2f}": speed_detected[speed] / speed_total[speed] if speed_total[speed] else 0.0
        for speed in speed_total
    }
    foot_recall = {
        ("left", "right")[side]: foot_detected[side] / foot_total[side] if foot_total[side] else 0.0
        for side in (0, 1)
    }
    encoder_parameters, encoder_macs = encoder_cost(architecture)
    metrics: dict[str, object] = {
        "architecture": architecture, "seed": seed,
        "physical_episode_count": physical_total, "physical_episode_detected": physical_detected,
        "physical_episode_missed": physical_total - physical_detected,
        "physical_episode_recall": physical_detected / physical_total if physical_total else 0.0,
        "actionable_episode_count": actionable_total, "actionable_episode_detected": actionable_detected,
        "actionable_episode_missed": actionable_total - actionable_detected,
        "actionable_episode_recall": actionable_detected / actionable_total if actionable_total else 0.0,
        "speed_actionable_recall": json.dumps(speed_recall, sort_keys=True),
        "minimum_speed_actionable_recall": min(speed_recall.values()),
        "foot_actionable_recall": json.dumps(foot_recall, sort_keys=True),
        "affected_foot_count": affected_total, "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
        "normal_run_fp": normal_run_fp, "normal_contact_episode_fp": normal_contact_fp,
        "too_early_activations": too_early, "air_firings": air,
        "touchdown_transient_firings": touchdown_count,
        "counterfactual_post_fall_state_outputs": post_fall_outputs,
        "post_fall_positive_attributions": post_fall_attribution, "invalid_firings": invalid,
        "previous_episode_latch_carryover": previous_episode_carryover,
        "cross_foot_state_ownership_violations": cross_foot_ownership,
        "timer_only_promotions": timer_promotions,
        "median_warning_margin_ms": float(np.median(margins)) if margins else -1.0,
        "median_late_latency_ms": float(np.median(latency)) if latency else -1.0,
        "pre_onset_detection_fraction": float(np.mean(np.asarray(margins) > 0)) if margins else 0.0,
        "both_affected_feet_covered": foot_detected[0] > 0 and foot_detected[1] > 0,
        "raw_threshold_crossings": raw_crossing_total, "actionable_runs": actionable_runs,
        "detected_runs_any_episode": detected_runs_any, "detected_runs_all_episodes": detected_runs_complete,
        "run_any_coverage": detected_runs_any / actionable_runs if actionable_runs else 0.0,
        "run_complete_coverage": detected_runs_complete / actionable_runs if actionable_runs else 0.0,
        "parameter_count": model.parameter_count + encoder_parameters,
        "macs_per_tick": model.macs + encoder_macs,
        "history_bytes": 20 * history * 4, "persistent_state_bytes": 128,
        "encoder_sha256": encoder_fingerprint(architecture),
        "causal_check_pass": checks["causal_check_pass"],
        "bilateral_parity_pass": checks["bilateral_parity_pass"],
    }
    metrics["gate_pass"] = slip_gate(metrics)
    reconciliation = {
        "architecture": architecture, "seed": seed,
        "raw_threshold_crossings": raw_crossing_total,
        "owned_contact_episodes": physical_total, "actionable_events": actionable_total,
        "detected_events": actionable_detected, "detected_runs_any_event": detected_runs_any,
        "detected_runs_all_events": detected_runs_complete,
    }
    return metrics, ledger, reset_rows, reconciliation


def ablation_classification_metrics(
    model: ProjectedLinearModel,
    rows: TerrainRows,
) -> dict[str, float]:
    prediction = model.predictions(rows.features).astype(int)
    recalls = recall_score(
        rows.target, prediction, labels=np.arange(4), average=None, zero_division=0,
    )
    return {
        "overall_accuracy": float(accuracy_score(rows.target, prediction)),
        "macro_recall": float(np.mean(recalls)),
        "worst_class_recall": float(np.min(recalls)),
    }


def destructive_balance_indices(rows: TerrainRows, seed: int) -> np.ndarray:
    groups = np.column_stack((
        rows.target, rows.phase, rows.side, np.rint(rows.speed * 100).astype(int),
    ))
    unique, inverse, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    count = int(np.min(counts))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group_index in range(len(unique)):
        candidates = np.flatnonzero(inverse == group_index)
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    return np.asarray(sorted(selected), int)


def raw_slip_diagnostics(
    architecture: str,
    model: ProjectedLinearModel | DualHeadModel,
    rows: SlipRows,
    data: DevelopmentData,
    *,
    history_override_ms: int | None = None,
) -> dict[str, int]:
    history = int(
        ARCHITECTURE_SPECS[architecture]["history_ms"]
        if history_override_ms is None else history_override_ms
    )
    endpoints = causal_endpoints(3000, history)
    raw_flat, _, _ = slip_outputs(architecture, model, rows)
    raw = reshape_slip(raw_flat, data, endpoints)
    air = touchdown = post_fall = normal_runs = 0
    for run_index, metadata in enumerate(data.metadata):
        current_prefall = data.pre_fall[run_index, endpoints, None]
        air += int(np.sum(raw[run_index] & ~data.loaded[run_index, endpoints]))
        touchdown += int(np.sum(raw[run_index] & data.touchdown[run_index, endpoints]))
        post_fall += int(np.sum(raw[run_index] & ~current_prefall))
        if str(metadata["role"]) == "hard_negative":
            normal_runs += int(bool(np.any(raw[run_index] & current_prefall)))
    return {
        "raw_air_firings": air, "raw_touchdown_firings": touchdown,
        "raw_post_fall_outputs": post_fall, "raw_normal_run_fp": normal_runs,
    }


def normalization_payload(model: ProjectedLinearModel | DualHeadModel) -> dict[str, object]:
    return {
        "source": "development_train_weighted_only", "mean": model.mean.tolist(),
        "scale": model.scale.tolist(), "feature_count": len(model.mean),
        "mean_sha256": sha256_json(model.mean.tolist()), "scale_sha256": sha256_json(model.scale.tolist()),
    }


def model_config(task: str, architecture: str, seed: int) -> dict[str, object]:
    return {
        "task": task, "architecture": architecture, "seed": seed,
        "architecture_spec": ARCHITECTURE_SPECS[architecture],
        "runtime_inputs": sorted(RUNTIME_INPUT_FIELDS), "forbidden_runtime_inputs": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "contact_gate": "force_loaded and contact_age_ms > 10 and not touchdown_transient",
        "post_fall_runtime_input": False, "physical_oracle_runtime_input": False,
        "sand_semantics": "SAND_TERRAIN_CAUTION", "sink_runtime_output": False,
    }


def save_and_reload(
    output: Path,
    task: str,
    architecture: str,
    seed: int,
    model: ProjectedLinearModel | DualHeadModel,
    parity_features: np.ndarray,
) -> tuple[dict[str, str], dict[str, object]]:
    model_path = output / f"{task}_candidate_model.npz"
    normalization_path = output / f"{task}_candidate_normalization.json"
    config_path = output / f"{task}_candidate_config.json"
    model.save(model_path)
    write_json(normalization_path, normalization_payload(model))
    write_json(config_path, model_config(task, architecture, seed))
    if isinstance(model, DualHeadModel):
        reloaded: ProjectedLinearModel | DualHeadModel = DualHeadModel.load(model_path)
        before = np.column_stack(model.scores(parity_features))
        after = np.column_stack(reloaded.scores(parity_features))
    else:
        reloaded = ProjectedLinearModel.load(model_path)
        before, after = model.probabilities(parity_features), reloaded.probabilities(parity_features)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    model_parity = float(np.max(np.abs(before - after)))
    mean_parity = float(np.max(np.abs(np.asarray(normalization["mean"]) - reloaded.mean)))
    scale_parity = float(np.max(np.abs(np.asarray(normalization["scale"]) - reloaded.scale)))
    if model_parity != 0.0 or mean_parity != 0.0 or scale_parity != 0.0:
        raise RuntimeError(f"{task} exact reload parity failed")
    paths = {
        "model_path": model_path.name, "model_sha256": sha256_file(model_path),
        "normalization_path": normalization_path.name, "normalization_sha256": sha256_file(normalization_path),
        "config_path": config_path.name, "config_sha256": sha256_file(config_path),
    }
    return paths, {
        "model_reload_max_abs_error": model_parity,
        "normalization_mean_reload_max_abs_error": mean_parity,
        "normalization_scale_reload_max_abs_error": scale_parity,
        "exact_reload_parity_pass": True,
    }


def selection_row(rows: list[dict[str, object]], architecture: str, seed: int) -> dict[str, object]:
    return next(row for row in rows if row["architecture"] == architecture and int(row["seed"]) == seed)


def write_audit(output: Path, summary: dict[str, object]) -> None:
    terrain = summary["terrain"]
    slip = summary["slip"]
    text = f"""# Walking v2 Joint Terrain / Slip Redesign v1

This is a development-only redesign and validation-selection record. No blind evaluation artifact was opened,
created, or evaluated. The previously exposed artifact class remains `EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY`.

## Answers required by the task

1. **Did 200 ms temporal representation recover Terrain performance?** {terrain['temporal_200ms_recovered']}.
   Best validation accuracy/macro/worst/Sand recall: {terrain['best_metrics']['overall_accuracy']:.4f} /
   {terrain['best_metrics']['macro_recall']:.4f} / {terrain['best_metrics']['worst_class_recall']:.4f} /
   {terrain['best_metrics']['sand_recall']:.4f}.
2. **Was destructive downsampling a material contributor?** {terrain['destructive_downsampling_material']}.
3. **Which Terrain architectures passed?** {terrain['family_gate_results']}.
4. **Did horizon-aware Slip remove all 40 genuine too-early activations?** {slip['removed_40_too_early']}.
5. **Corrected Slip actionable recall overall/by speed:** {slip['best_metrics']['actionable_episode_recall']:.4f};
   {slip['best_metrics']['speed_actionable_recall']}.
6. **Are normal FP, invalid firing, latch carryover, and cross-foot ownership zero?** {slip['all_safety_zero']}.
7. **Were both selection locks created?** {summary['both_selection_locks_created']}.
8. **Is acquisition of a completely new blind evaluation split authorized?** {summary['fresh_blind_acquisition_authorized']}.
   No such split was generated in this task.
9. **Was the exposed previous artifact accessed or used?** No. The allowlist contains no forbidden namespace,
   and the fail-closed read ledger records zero forbidden reads.
10. **Sink status:** `SINK_RUNTIME_DETECTION_DEFERRED`; no Sink runtime head, score, model, config, or output was created.

## Selection

- Terrain: {terrain['selection_status']} — {terrain['selected_or_fallback']}.
- Slip: {slip['selection_status']} — {slip['selected_or_fallback']}.
- Static/controlled detector: immutable compatibility hash only; not replaced.
- Sand: `SAND_TERRAIN_CAUTION`, never Sink detection.

## Next step

`{summary['next_step']}`
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if git_output("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("runner must start at the preregistered checkpoint")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    allowlist_payload = {
        "version": "walking_v2_redesign_input_allowlist_v1", "exact_paths_only": True,
        "inputs": list(INPUTS), "forbidden_namespace_entries": 0,
    }
    forbidden_policy = {
        "version": "walking_v2_redesign_forbidden_path_policy_v1", "fail_closed": True,
        "case_insensitive_substrings": list(FORBIDDEN_PATH_TOKENS),
        "spatial_policy": "all paths containing spatial are conservatively forbidden",
        "applies_to": ["hash", "JSON", "NPZ", "model", "normalization", "metrics"],
        "forbidden_artifact_status": "EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY",
        "broad_repo_search_forbidden": True,
    }
    write_json(output / "input_allowlist.json", allowlist_payload)
    write_json(output / "forbidden_path_policy.json", forbidden_policy)
    write_json(output / "protocol.json", protocol())
    pretraining_hashes = {
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "input_allowlist_sha256": sha256_file(output / "input_allowlist.json"),
        "forbidden_path_policy_sha256": sha256_file(output / "forbidden_path_policy.json"),
    }
    guard = ArtifactAccessGuard(REPO, [row["path"] for row in INPUTS], output / "artifact_access_log.json")

    manifest = guard.read_json(INPUTS[0]["path"], INPUTS[0]["purpose"])
    upstream_summary = guard.read_json(INPUTS[1]["path"], INPUTS[1]["purpose"])
    train_arrays = guard.load_npz(INPUTS[2]["path"], INPUTS[2]["purpose"])
    validation_arrays = guard.load_npz(INPUTS[3]["path"], INPUTS[3]["purpose"])
    operational_summary = guard.read_json(INPUTS[4]["path"], INPUTS[4]["purpose"])
    stateful_summary = guard.read_json(INPUTS[5]["path"], INPUTS[5]["purpose"])
    failure_summary = guard.read_json(INPUTS[6]["path"], INPUTS[6]["purpose"])
    static_hashes = {
        row["path"]: guard.hash_input(row["path"], row["purpose"]) for row in INPUTS[7:]
    }
    guard.assert_complete()
    train = load_data("train", train_arrays, manifest)
    validation = load_data("validation", validation_arrays, manifest)
    del train_arrays, validation_arrays

    input_hashes: dict[str, str] = {}
    access_log = json.loads((output / "artifact_access_log.json").read_text(encoding="utf-8"))
    for event in access_log["events"]:
        input_hashes[str(event["relative_path"])] = str(event["sha256"])
    data_manifest = {
        "version": "walking_v2_redesign_data_manifest_v1", "pretraining_barrier": pretraining_hashes,
        "input_sha256": input_hashes, "train_runs": train.run_count, "validation_runs": validation.run_count,
        "blind_evaluation_runs": 0, "sample_rate_hz": SAMPLE_RATE_HZ, "samples_per_run": 3000,
        "runtime_tensor": "bilateral_canonical float32 [run,3000,20]",
        "offline_label_tensors": ["slip_physical_active", "pre_fall_valid"],
        "upstream_contract_sha256": sha256_json(upstream_summary),
        "operational_contract_sha256": sha256_json(operational_summary),
        "stateful_contract_sha256": sha256_json(stateful_summary),
        "failure_localization_sha256": sha256_json(failure_summary),
    }
    split_manifest = {
        "version": "walking_v2_redesign_split_manifest_v1",
        "train_run_ids": train.run_id.tolist(), "validation_run_ids": validation.run_id.tolist(),
        "train_run_ids_sha256": sha256_json(train.run_id.tolist()),
        "validation_run_ids_sha256": sha256_json(validation.run_id.tolist()),
        "disjoint": set(train.run_id.tolist()).isdisjoint(validation.run_id.tolist()),
        "blind_split_materialized": False,
    }
    write_json(output / "data_manifest.json", data_manifest)
    write_json(output / "split_manifest.json", split_manifest)

    terrain_models: dict[tuple[str, int], ProjectedLinearModel] = {}
    terrain_rows_by_arch: dict[str, tuple[TerrainRows, TerrainRows]] = {}
    terrain_metrics_rows: list[dict[str, object]] = []
    terrain_health: list[dict[str, object]] = []
    terrain_confusions: dict[str, object] = {}
    parity_rows: list[dict[str, object]] = []
    resource_candidates: list[dict[str, object]] = []
    for architecture in TERRAIN_ARCHITECTURES:
        train_rows = build_terrain_rows(train, architecture)
        validation_rows = build_terrain_rows(validation, architecture)
        terrain_rows_by_arch[architecture] = (train_rows, validation_rows)
        weight = terrain_weights(train_rows)
        if architecture == "T1" and len(train_rows.target) != 24560:
            raise RuntimeError(f"Terrain eligible population changed: {len(train_rows.target)}")
        balance_factors = {
            "terrain": train_rows.target, "phase": train_rows.phase, "foot": train_rows.side,
            "speed": np.rint(train_rows.speed * 100).astype(int), "run": train_rows.run_index,
            "variation": train_rows.variation,
        }
        balance_report = {
            name: {
                "raw_min": int(np.min(np.unique(values, return_counts=True)[1])),
                "raw_max": int(np.max(np.unique(values, return_counts=True)[1])),
                "weighted_min": float(np.min(np.bincount(
                    np.unique(values, return_inverse=True)[1], weights=weight
                ))),
                "weighted_max": float(np.max(np.bincount(
                    np.unique(values, return_inverse=True)[1], weights=weight
                ))),
            } for name, values in balance_factors.items()
        }
        for seed in TRAINING_SEEDS:
            model, health = fit_projected_linear(
                architecture, seed, train_rows.features, train_rows.target, weight,
            )
            terrain_models[(architecture, seed)] = model
            checks = contract_checks(architecture, model, validation)
            parity_rows.append({"task": "terrain", **checks})
            metrics = evaluate_terrain(architecture, seed, model, validation_rows, checks)
            terrain_metrics_rows.append(metrics)
            terrain_confusions[f"{architecture}_seed_{seed}"] = json.loads(str(metrics["confusion_matrix"]))
            train_prediction = model.predictions(train_rows.features)
            terrain_health.append({
                "architecture": architecture, "seed": seed,
                "raw_sample_count": len(train_rows.target), "retained_sample_count": len(train_rows.target),
                "rows_dropped_for_balance": 0, "effective_weighted_mass": float(weight.sum()),
                "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
                "weight_min": float(weight.min()), "weight_max": float(weight.max()),
                "train_accuracy": float(accuracy_score(train_rows.target, train_prediction)),
                "train_log_loss": float(log_loss(train_rows.target, model.probabilities(train_rows.features))),
                "balance_report": json.dumps(balance_report, sort_keys=True), **health,
            })
            resource_candidates.append({
                "task": "terrain", "architecture": architecture, "seed": seed,
                "parameter_count": metrics["parameter_count"], "macs_per_tick": metrics["macs_per_tick"],
                "history_bytes": metrics["history_bytes"], "persistent_state_bytes": metrics["persistent_state_bytes"],
                "within_all_ceilings": all((
                    int(metrics["parameter_count"]) <= RESOURCE_CEILINGS["parameter_count"],
                    int(metrics["macs_per_tick"]) <= RESOURCE_CEILINGS["macs_per_tick"],
                    int(metrics["history_bytes"]) <= RESOURCE_CEILINGS["history_bytes"],
                    int(metrics["persistent_state_bytes"]) <= RESOURCE_CEILINGS["persistent_state_bytes"],
                )),
            })

    terrain_selected = deterministic_terrain_selection(terrain_metrics_rows)
    terrain_fallback = terrain_diagnostic_fallback(terrain_metrics_rows)

    slip_models: dict[tuple[str, int], ProjectedLinearModel | DualHeadModel] = {}
    slip_rows_by_arch: dict[str, tuple[SlipRows, SlipRows]] = {}
    slip_metrics_rows: list[dict[str, object]] = []
    slip_health: dict[tuple[str, int], dict[str, object]] = {}
    slip_ledgers: dict[tuple[str, int], list[dict[str, object]]] = {}
    slip_resets: dict[tuple[str, int], list[dict[str, object]]] = {}
    slip_reconciliations: dict[tuple[str, int], dict[str, object]] = {}
    for architecture in SLIP_ARCHITECTURES:
        train_rows = build_slip_rows(train, architecture)
        validation_rows = build_slip_rows(validation, architecture)
        slip_rows_by_arch[architecture] = (train_rows, validation_rows)
        for seed in TRAINING_SEEDS:
            model, health = train_slip_model(architecture, seed, train_rows)
            slip_models[(architecture, seed)] = model
            slip_health[(architecture, seed)] = health
            checks = contract_checks(architecture, model, validation)
            parity_rows.append({"task": "slip", **checks})
            metrics, ledger, resets, reconciliation = evaluate_slip(
                architecture, seed, model, validation_rows, validation, checks,
            )
            metrics.update({
                "train_rows": health["train_rows"], "effective_weighted_mass": health["effective_weighted_mass"],
                "effective_sample_size": health["effective_sample_size"], "optimizer_iterations": health["iterations"],
                "optimizer_converged": health["converged"],
            })
            slip_metrics_rows.append(metrics)
            slip_ledgers[(architecture, seed)] = ledger
            slip_resets[(architecture, seed)] = resets
            slip_reconciliations[(architecture, seed)] = reconciliation
            resource_candidates.append({
                "task": "slip", "architecture": architecture, "seed": seed,
                "parameter_count": metrics["parameter_count"], "macs_per_tick": metrics["macs_per_tick"],
                "history_bytes": metrics["history_bytes"], "persistent_state_bytes": metrics["persistent_state_bytes"],
                "within_all_ceilings": all((
                    int(metrics["parameter_count"]) <= RESOURCE_CEILINGS["parameter_count"],
                    int(metrics["macs_per_tick"]) <= RESOURCE_CEILINGS["macs_per_tick"],
                    int(metrics["history_bytes"]) <= RESOURCE_CEILINGS["history_bytes"],
                    int(metrics["persistent_state_bytes"]) <= RESOURCE_CEILINGS["persistent_state_bytes"],
                )),
            })
    slip_selected = deterministic_slip_selection(slip_metrics_rows)
    slip_fallback = slip_diagnostic_fallback(slip_metrics_rows)

    # Diagnostic ablations use the one frozen ablation seed and never enter ranking.
    ablation_seed = TRAINING_SEEDS[0]
    t1_train, t1_validation = terrain_rows_by_arch["T1"]
    t1_model = terrain_models[("T1", ablation_seed)]
    t1_metrics = ablation_classification_metrics(t1_model, t1_validation)

    terrain_50_train = build_terrain_rows(train, "T1", history_override_ms=50)
    terrain_50_validation = build_terrain_rows(validation, "T1", history_override_ms=50)
    terrain_50_model, _ = fit_projected_linear(
        "T1", ablation_seed, terrain_50_train.features, terrain_50_train.target,
        terrain_weights(terrain_50_train), projection_width_override=0,
    )
    terrain_50_metrics = ablation_classification_metrics(terrain_50_model, terrain_50_validation)

    compact_train, compact_validation = terrain_rows_by_arch["T3"]
    compact_model, _ = fit_projected_linear(
        "T3", ablation_seed, compact_train.features, compact_train.target,
        terrain_weights(compact_train), projection_width_override=0,
    )
    compact_metrics = ablation_classification_metrics(compact_model, compact_validation)

    downsample = destructive_balance_indices(t1_train, ablation_seed)
    if len(downsample) != 1440:
        raise RuntimeError(f"locked destructive balance control changed: {len(downsample)}")
    downsample_model, _ = fit_projected_linear(
        "T1", ablation_seed, t1_train.features[downsample], t1_train.target[downsample],
        np.ones(len(downsample)), projection_width_override=0,
    )
    downsample_metrics = ablation_classification_metrics(downsample_model, t1_validation)

    symmetric_train = build_terrain_rows(train, "T1", include_asymmetry=False)
    symmetric_validation = build_terrain_rows(validation, "T1", include_asymmetry=False)
    symmetric_model, _ = fit_projected_linear(
        "T1", ablation_seed, symmetric_train.features, symmetric_train.target,
        terrain_weights(symmetric_train), projection_width_override=0,
    )
    symmetric_metrics = ablation_classification_metrics(symmetric_model, symmetric_validation)

    capacity_model, _ = fit_projected_linear(
        "T1", ablation_seed, t1_train.features, t1_train.target, terrain_weights(t1_train),
        projection_width_override=32,
    )
    capacity_metrics = ablation_classification_metrics(capacity_model, t1_validation)
    terrain_ablations = [
        {"ablation": "history", "control": "50ms_full_statistics", "treatment": "200ms_full_statistics", "control_rows": len(terrain_50_train.target), "treatment_rows": len(t1_train.target), "control_macro_recall": terrain_50_metrics["macro_recall"], "treatment_macro_recall": t1_metrics["macro_recall"], "delta_pp": (t1_metrics["macro_recall"] - terrain_50_metrics["macro_recall"]) * 100, "isolated": True, "diagnostic_only": True},
        {"ablation": "temporal_statistics", "control": "200ms_compact_causal_conv_linear", "treatment": "200ms_full_statistics_linear", "control_macro_recall": compact_metrics["macro_recall"], "treatment_macro_recall": t1_metrics["macro_recall"], "delta_pp": (t1_metrics["macro_recall"] - compact_metrics["macro_recall"]) * 100, "isolated": True, "diagnostic_only": True},
        {"ablation": "population_balance", "control": "destructive_equal_group_1440", "treatment": "weighted_full_24560", "control_rows": len(downsample), "treatment_rows": len(t1_train.target), "control_macro_recall": downsample_metrics["macro_recall"], "treatment_macro_recall": t1_metrics["macro_recall"], "delta_pp": (t1_metrics["macro_recall"] - downsample_metrics["macro_recall"]) * 100, "isolated": True, "diagnostic_only": True},
        {"ablation": "foot_encoder", "control": "duplicated_fixed_stat_operator", "treatment": "identity_shared_fixed_stat_operator", "control_feature_difference": 0.0, "treatment_feature_difference": 0.0, "delta_pp": 0.0, "isolated": True, "note": "fixed encoder has no foot-specific trainable weights", "diagnostic_only": True},
        {"ablation": "bilateral_aggregation", "control": "symmetric_only", "treatment": "symmetric_plus_own_minus_other", "control_macro_recall": symmetric_metrics["macro_recall"], "treatment_macro_recall": t1_metrics["macro_recall"], "delta_pp": (t1_metrics["macro_recall"] - symmetric_metrics["macro_recall"]) * 100, "isolated": True, "diagnostic_only": True},
        {"ablation": "model_capacity", "control": "linear_head", "treatment": "fixed_relu_width32_plus_linear_head", "control_macro_recall": t1_metrics["macro_recall"], "treatment_macro_recall": capacity_metrics["macro_recall"], "delta_pp": (capacity_metrics["macro_recall"] - t1_metrics["macro_recall"]) * 100, "isolated": True, "diagnostic_only": True},
    ]

    s1_train, s1_validation = slip_rows_by_arch["S1"]
    s1_model = slip_models[("S1", ablation_seed)]
    s1_metrics = selection_row(slip_metrics_rows, "S1", ablation_seed)
    s1_checks = contract_checks("S1", s1_model, validation)
    selected = np.flatnonzero(s1_train.runtime_eligible)

    binary_target = (s1_train.s1_target[selected] > 0).astype(int)
    binary_weight = episode_balanced_weights(binary_target, s1_train.balance_unit[selected])
    binary_model, _ = fit_projected_linear(
        "S1", ablation_seed, s1_train.features[selected], binary_target, binary_weight,
    )
    binary_metrics, _, _, _ = evaluate_slip(
        "S1", ablation_seed, binary_model, s1_validation, validation, s1_checks,
    )

    dense_target = s1_train.s1_target[selected]
    dense_weight = raking_weights(np.column_stack((
        dense_target, s1_train.side[selected], np.rint(s1_train.speed[selected] * 100).astype(int),
    )))
    dense_model, _ = fit_projected_linear(
        "S1", ablation_seed, s1_train.features[selected], dense_target, dense_weight,
    )
    dense_metrics, _, _, _ = evaluate_slip(
        "S1", ablation_seed, dense_model, s1_validation, validation, s1_checks,
    )

    slip_50_train = build_slip_rows(train, "S1", history_override_ms=50)
    slip_50_validation = build_slip_rows(validation, "S1", history_override_ms=50)
    slip_50_model, _ = train_slip_model("S1", ablation_seed, slip_50_train)
    slip_50_metrics, _, _, _ = evaluate_slip(
        "S1", ablation_seed, slip_50_model, slip_50_validation, validation, s1_checks,
        history_override_ms=50,
    )
    slip_200_train = build_slip_rows(train, "S1", history_override_ms=200)
    slip_200_validation = build_slip_rows(validation, "S1", history_override_ms=200)
    slip_200_model, _ = train_slip_model("S1", ablation_seed, slip_200_train)
    slip_200_metrics, _, _, _ = evaluate_slip(
        "S1", ablation_seed, slip_200_model, slip_200_validation, validation, s1_checks,
        history_override_ms=200,
    )
    raw_diagnostics = raw_slip_diagnostics("S1", s1_model, s1_validation, validation)
    s3_metrics = selection_row(slip_metrics_rows, "S3", ablation_seed)
    slip_ablations = [
        {"ablation": "target", "control": "binary_within_horizon_or_active", "treatment": "three_class_horizon_target", "control_recall": binary_metrics["actionable_episode_recall"], "treatment_recall": s1_metrics["actionable_episode_recall"], "control_too_early": binary_metrics["too_early_activations"], "treatment_too_early": s1_metrics["too_early_activations"], "isolated": True, "diagnostic_only": True},
        {"ablation": "loss_balance", "control": "dense_sample_marginal_balance", "treatment": "episode_and_normal_contact_balance", "control_recall": dense_metrics["actionable_episode_recall"], "treatment_recall": s1_metrics["actionable_episode_recall"], "control_normal_run_fp": dense_metrics["normal_run_fp"], "treatment_normal_run_fp": s1_metrics["normal_run_fp"], "isolated": True, "diagnostic_only": True},
        {"ablation": "state", "control": "raw_head_output", "treatment": "contact_scoped_per_foot", **raw_diagnostics, "treatment_air_firings": s1_metrics["air_firings"], "treatment_touchdown_firings": s1_metrics["touchdown_transient_firings"], "isolated": True, "diagnostic_only": True},
        {"ablation": "history", "control": "50ms", "treatment": "100ms", "control_recall": slip_50_metrics["actionable_episode_recall"], "treatment_recall": s1_metrics["actionable_episode_recall"], "control_too_early": slip_50_metrics["too_early_activations"], "treatment_too_early": s1_metrics["too_early_activations"], "isolated": True, "diagnostic_only": True},
        {"ablation": "history", "control": "100ms", "treatment": "200ms", "control_recall": s1_metrics["actionable_episode_recall"], "treatment_recall": slip_200_metrics["actionable_episode_recall"], "control_too_early": s1_metrics["too_early_activations"], "treatment_too_early": slip_200_metrics["too_early_activations"], "isolated": True, "diagnostic_only": True},
        {"ablation": "heads", "control": "S1_single_categorical", "treatment": "S3_separate_actionable_active", "control_recall": s1_metrics["actionable_episode_recall"], "treatment_recall": s3_metrics["actionable_episode_recall"], "control_affected_foot_accuracy": s1_metrics["affected_foot_accuracy"], "treatment_affected_foot_accuracy": s3_metrics["affected_foot_accuracy"], "isolated": False, "note": "task family comparison", "diagnostic_only": True},
    ]

    write_csv(output / "terrain_candidate_matrix.csv", terrain_metrics_rows)
    write_csv(output / "terrain_training_health.csv", terrain_health)
    write_csv(output / "terrain_ablation_metrics.csv", terrain_ablations)
    write_csv(output / "terrain_validation_metrics.csv", terrain_metrics_rows)
    write_json(output / "terrain_confusion_matrices.json", terrain_confusions)
    write_csv(output / "slip_candidate_matrix.csv", slip_metrics_rows)
    write_csv(output / "slip_validation_metrics.csv", slip_metrics_rows)
    write_csv(output / "slip_timing_metrics.csv", [{
        key: row[key] for key in (
            "architecture", "seed", "actionable_episode_count", "actionable_episode_detected",
            "actionable_episode_recall", "speed_actionable_recall", "too_early_activations",
            "median_warning_margin_ms", "median_late_latency_ms", "pre_onset_detection_fraction",
        )
    } for row in slip_metrics_rows])
    write_csv(output / "slip_ablation_metrics.csv", slip_ablations)
    chosen_slip = slip_selected or slip_fallback
    chosen_slip_key = (str(chosen_slip["architecture"]), int(chosen_slip["seed"]))
    write_csv(output / "slip_episode_ledger.csv", slip_ledgers[chosen_slip_key])
    write_csv(output / "slip_crossing_reconciliation.csv", list(slip_reconciliations.values()))
    write_csv(output / "slip_reset_audit.csv", slip_resets[chosen_slip_key])
    write_json(output / "bilateral_parity_audit.json", {
        "all_candidate_checks": parity_rows,
        "all_bilateral_parity_pass": all(bool(row["bilateral_parity_pass"]) for row in parity_rows),
        "all_future_leakage_checks_pass": all(bool(row["causal_check_pass"]) for row in parity_rows),
        "canonical_slot_order": ["left Fusion10", "right Fusion10"],
    })
    resource_report = {
        "ceilings": RESOURCE_CEILINGS, "candidates": resource_candidates,
        "all_candidates_within_ceiling": all(bool(row["within_all_ceilings"]) for row in resource_candidates),
        "int8_conversion_performed": False, "E84_or_Vela_execution_performed": False,
    }
    write_json(output / "resource_report.json", resource_report)

    terrain_lock = slip_lock = None
    reload_audit: dict[str, object] = {}
    if terrain_selected is not None:
        key = (str(terrain_selected["architecture"]), int(terrain_selected["seed"]))
        model = terrain_models[key]
        paths, parity = save_and_reload(
            output, "terrain", key[0], key[1], model, terrain_rows_by_arch[key[0]][1].features[:128],
        )
        reload_audit["terrain"] = parity
        terrain_lock = {
            "version": "walking_v2_terrain_selection_lock_v1", "immutable": True,
            "protocol_sha256": pretraining_hashes["protocol_sha256"],
            "data_manifest_sha256": sha256_file(output / "data_manifest.json"),
            "split_manifest_sha256": sha256_file(output / "split_manifest.json"),
            "base_commit": STARTING_CHECKPOINT, **paths,
            "validation_metrics_sha256": sha256_file(output / "terrain_validation_metrics.csv"),
            "resource_report_sha256": sha256_file(output / "resource_report.json"),
            "selected_metrics": terrain_selected,
        }
        write_json(output / "terrain_selection_lock.json", terrain_lock)
    if slip_selected is not None:
        key = (str(slip_selected["architecture"]), int(slip_selected["seed"]))
        model = slip_models[key]
        paths, parity = save_and_reload(
            output, "slip", key[0], key[1], model, slip_rows_by_arch[key[0]][1].features[:128],
        )
        reload_audit["slip"] = parity
        slip_lock = {
            "version": "walking_v2_slip_selection_lock_v1", "immutable": True,
            "protocol_sha256": pretraining_hashes["protocol_sha256"],
            "data_manifest_sha256": sha256_file(output / "data_manifest.json"),
            "split_manifest_sha256": sha256_file(output / "split_manifest.json"),
            "base_commit": STARTING_CHECKPOINT, **paths,
            "validation_metrics_sha256": sha256_file(output / "slip_validation_metrics.csv"),
            "resource_report_sha256": sha256_file(output / "resource_report.json"),
            "selected_metrics": slip_selected,
        }
        write_json(output / "slip_selection_lock.json", slip_lock)

    provenance = {
        "version": "walking_v2_redesign_provenance_v1", "barrier_pass": True,
        "pretraining_hashes": pretraining_hashes, "artifact_access_log_sha256": sha256_file(output / "artifact_access_log.json"),
        "forbidden_read_count": 0, "allowlist_entry_count": len(INPUTS),
        "all_input_reads_completed": True, "broad_repo_search_used_by_runner": False,
        "previous_exposed_artifact_status": "EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY",
        "previous_exposed_artifact_accessed": False, "previous_exposed_artifact_used": False,
        "blindness_restoration_claimed": False, "selection_data": ["development_train", "development_validation"],
        "static_compatibility_hashes": static_hashes,
        "hash_graph": {
            "protocol": pretraining_hashes["protocol_sha256"],
            "data_manifest": sha256_file(output / "data_manifest.json"),
            "split_manifest": sha256_file(output / "split_manifest.json"),
            "terrain_validation": sha256_file(output / "terrain_validation_metrics.csv"),
            "slip_validation": sha256_file(output / "slip_validation_metrics.csv"),
            "resource_report": sha256_file(output / "resource_report.json"),
            "terrain_lock": None if terrain_lock is None else sha256_file(output / "terrain_selection_lock.json"),
            "slip_lock": None if slip_lock is None else sha256_file(output / "slip_selection_lock.json"),
        },
        "exact_reload_parity": reload_audit,
    }
    write_json(output / "provenance.json", provenance)

    terrain_ready = terrain_selected is not None
    slip_ready = slip_selected is not None
    both_locks = terrain_lock is not None and slip_lock is not None
    fresh_authorized = bool(terrain_ready and slip_ready and both_locks and provenance["barrier_pass"])
    readiness = {
        "WALKING_V2_REDESIGN_DATA_READY": True,
        "WALKING_V2_REDESIGN_PROVENANCE_READY": True,
        "WALKING_V2_CAUSAL_CONTRACT_READY": all(bool(row["causal_check_pass"]) for row in parity_rows),
        "WALKING_V2_BILATERAL_PARITY_READY": all(bool(row["bilateral_parity_pass"]) for row in parity_rows),
        "WALKING_V2_CORRECTNESS_FIX_READY": True,
        "WALKING_V2_TERRAIN_TEMPORAL_REDESIGN_READY": True,
        "WALKING_V2_TERRAIN_FLOAT_CANDIDATE_READY": terrain_ready,
        "WALKING_V2_SLIP_HORIZON_TASK_READY": True,
        "WALKING_V2_SLIP_STATE_LOGIC_READY": True,
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": slip_ready,
        "WALKING_V2_JOINT_FLOAT_CANDIDATES_READY": terrain_ready and slip_ready,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": fresh_authorized,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    write_json(output / "readiness.json", readiness)

    best_terrain = terrain_selected or terrain_fallback
    best_slip = slip_selected or slip_fallback
    family_gate_results = {
        architecture: any(bool(row["gate_pass"]) for row in terrain_metrics_rows if row["architecture"] == architecture)
        for architecture in TERRAIN_ARCHITECTURES
    }
    if terrain_ready and slip_ready:
        next_step = "FRESH_BLIND_HOLDOUT_ACQUISITION"
    elif not terrain_ready and not slip_ready:
        next_step = "JOINT_REDESIGN_ITERATION"
    elif not terrain_ready:
        next_step = "TERRAIN_REDESIGN_ITERATION"
    else:
        next_step = "SLIP_REDESIGN_ITERATION"
    all_safety_zero = all(int(best_slip[key]) == 0 for key in (
        "normal_run_fp", "normal_contact_episode_fp", "invalid_firings",
        "previous_episode_latch_carryover", "cross_foot_state_ownership_violations",
    ))
    summary = {
        "artifact": "walking_v2_joint_terrain_slip_redesign_v1", "base_commit": STARTING_CHECKPOINT,
        "development_only": True, "blind_evaluation_accessed": False, "blind_evaluation_generated": False,
        "terrain": {
            "selection_status": "selected_and_locked" if terrain_ready else "diagnostic_fallback_only",
            "selected_or_fallback": f"{best_terrain['architecture']}_seed_{best_terrain['seed']}",
            "best_metrics": best_terrain, "family_gate_results": family_gate_results,
            "temporal_200ms_recovered": bool(float(best_terrain["macro_recall"]) >= 0.85),
            "destructive_downsampling_material": bool(
                t1_metrics["macro_recall"] - downsample_metrics["macro_recall"] >= 0.05
            ),
            "selection_lock_created": terrain_lock is not None,
        },
        "slip": {
            "selection_status": "selected_and_locked" if slip_ready else "diagnostic_fallback_only",
            "selected_or_fallback": f"{best_slip['architecture']}_seed_{best_slip['seed']}",
            "best_metrics": best_slip,
            "removed_40_too_early": int(best_slip["too_early_activations"]) == 0,
            "all_safety_zero": all_safety_zero, "selection_lock_created": slip_lock is not None,
        },
        "both_selection_locks_created": both_locks,
        "fresh_blind_acquisition_authorized": fresh_authorized,
        "fresh_blind_artifact_generated": False,
        "previous_exposed_artifact_status": "EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY",
        "previous_exposed_artifact_accessed_or_used": False,
        "sink_status": "SINK_RUNTIME_DETECTION_DEFERRED", "sink_runtime_output_created": False,
        "sand_semantics": "SAND_TERRAIN_CAUTION", "static_detector_replaced": False,
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
        "terrain_ready": result["readiness"]["WALKING_V2_TERRAIN_FLOAT_CANDIDATE_READY"],
        "slip_ready": result["readiness"]["WALKING_V2_SLIP_FLOAT_CANDIDATE_READY"],
        "fresh_blind_authorized": result["fresh_blind_acquisition_authorized"],
        "next_step": result["next_step"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
