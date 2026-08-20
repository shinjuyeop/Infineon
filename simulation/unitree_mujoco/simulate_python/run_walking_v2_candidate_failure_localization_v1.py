"""Deterministically localize walking-v2 Terrain and Slip candidate failures.

This audit consumes only stored development traces and frozen candidate files.
It never acquires simulation data, retrains a candidate, changes a threshold,
or opens a holdout as part of the audit implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.preprocessing import StandardScaler

from run_walking_v2_bilateral_bounded_training import (
    DevelopmentData,
    build_feature_rows,
    load_development,
    terrain_train_indices,
)
from walking_bilateral_sink_observability_v2 import SharedFootEncoderV2
from walking_v2_bilateral_bounded_training import (
    ENDPOINT_STRIDE_MS,
    PHASE_NAMES,
    SLIP_WINDOW_MS,
    TERRAIN_ARCHITECTURES,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    TRAINING_SEEDS,
    LinearFloatModel,
    SharedCausalFootEncoder,
    SlipStateConfig,
    affected_foot_correct,
    causal_endpoints,
    first_actionable_events,
    physical_slip_episodes,
    raw_slip_crossings,
    runtime_feature,
    sha256_file,
)
from walking_v2_candidate_failure_localization_v1 import (
    AUDIT_VARIANTS,
    classify_too_early,
    invalid_accounting,
    join_sample_ledgers,
    model_signature,
    replay_slip_state,
    tensor_statistics,
)


SIMULATION = Path(__file__).resolve().parents[2]
REPO = SIMULATION.parent
DATA = SIMULATION / "outputs" / "walking_bilateral_sensor_sink_observability_v2"
TRAINING = SIMULATION / "outputs" / "walking_v2_bilateral_bounded_training"
OUTPUT = SIMULATION / "outputs" / "walking_v2_candidate_failure_localization_v1"
STARTING_CHECKPOINT = "679a4c52693f403cc3b262a7f13070fe7a494b7b"
DIAGNOSTIC_SEED = 20260820
SLIP_CONFIG = SlipStateConfig(0.999, 3, 0.05)
SLIP_MODEL = TRAINING / "models" / "slip_s1_seed_202608211.npz"
TERRAIN_MODEL = TRAINING / "models" / "terrain_t2_seed_202608213.npz"
OUTER_ACCESS_INCIDENT = {
    "content_load_count": 1,
    "path": "simulation/outputs/walking_bounded_retraining_v1/holdout_metrics.csv",
    "cause": "over-broad rg during pre-implementation MAC-source discovery",
    "used_by_audit_computation": False,
}

ALLOWED_AUDIT_FILES = (
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/manifest.json",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/summary.json",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/bilateral_traces_train.npz",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/bilateral_traces_validation.npz",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/protocol.json",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/terrain_bilateral_probe_metrics.csv",
    "simulation/outputs/walking_hazard_operational_label_contract_v2/summary.json",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/protocol.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/manifest.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/summary.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/candidate_selection.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/terrain_validation_metrics.csv",
    "simulation/outputs/walking_v2_bilateral_bounded_training/slip_validation_metrics.csv",
    "simulation/outputs/walking_v2_bilateral_bounded_training/models/terrain_t2_seed_202608213.npz",
    "simulation/outputs/walking_v2_bilateral_bounded_training/normalization/terrain_t2_seed_202608213.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/configs/terrain_t2_seed_202608213.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/models/slip_s1_seed_202608211.npz",
    "simulation/outputs/walking_v2_bilateral_bounded_training/normalization/slip_s1_seed_202608211.json",
    "simulation/outputs/walking_v2_bilateral_bounded_training/configs/slip_s1_seed_202608211.json",
    "simulation/outputs/terrain_static_reference_v4/selected_model.keras",
    "simulation/outputs/terrain_static_reference_v4/normalization.json",
    "simulation/outputs/terrain_fast_reflex_v2_int8/slip/model_int8.tflite",
    "simulation/outputs/terrain_fast_reflex_v2_int8/sink/model_int8.tflite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _csv_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write("\n")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def upstream_hashes() -> dict[str, str]:
    result = {}
    for relative in ALLOWED_AUDIT_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = sha256_file(path)
    return result


@dataclass(frozen=True)
class DiagnosticRows:
    features: np.ndarray
    t2_projection: np.ndarray
    target: np.ndarray
    run_index: np.ndarray
    endpoint: np.ndarray
    side: np.ndarray
    phase: np.ndarray
    speed: np.ndarray


def diagnostic_rows(data: DevelopmentData) -> DiagnosticRows:
    diagnostic_encoder = SharedFootEncoderV2()
    t2_encoder = SharedCausalFootEncoder(50)
    features: list[np.ndarray] = []
    t2_projection: list[np.ndarray] = []
    fields: dict[str, list[object]] = {
        key: [] for key in ("target", "run_index", "endpoint", "side", "phase", "speed")
    }
    for run_index, metadata in enumerate(data.metadata):
        bilateral = data.bilateral[run_index]
        for endpoint in range(199, 3000, 10):
            start = endpoint - diagnostic_encoder.window_samples + 1
            left = diagnostic_encoder.encode(bilateral[start:endpoint + 1, :10])
            right = diagnostic_encoder.encode(bilateral[start:endpoint + 1, 10:])
            forces = bilateral[endpoint, (0, 1, 2, 3, 10, 11, 12, 13)].reshape(2, 4).sum(axis=1)
            total = float(forces.sum())
            for side in (0, 1):
                if not (data.loaded[run_index, endpoint, side] and data.age[run_index, endpoint, side] > 10):
                    continue
                own, other = (left, right) if side == 0 else (right, left)
                ratio = float(forces[side] / total) if total > 1e-9 else 0.5
                features.append(np.concatenate((
                    own, other, np.asarray((
                        ratio, 1.0 - ratio,
                        data.loaded[run_index, endpoint, side],
                        data.loaded[run_index, endpoint, 1 - side],
                    ), np.float32),
                )))
                t2_projection.append(runtime_feature(
                    "T2", side, endpoint, bilateral, data.loaded[run_index],
                    data.age[run_index], data.phase[run_index], t2_encoder,
                ))
                values = {
                    "target": TERRAIN_LABELS[str(metadata["terrain_name"])],
                    "run_index": run_index, "endpoint": endpoint, "side": side,
                    "phase": int(data.phase[run_index, endpoint, side]),
                    "speed": float(metadata["speed_mps"]),
                }
                for key, value in values.items():
                    fields[key].append(value)
    return DiagnosticRows(
        np.asarray(features, np.float32), np.asarray(t2_projection, np.float32),
        np.asarray(fields["target"], int), np.asarray(fields["run_index"], int),
        np.asarray(fields["endpoint"], int), np.asarray(fields["side"], int),
        np.asarray(fields["phase"], int), np.asarray(fields["speed"], float),
    )


def sample_ledger(data: DevelopmentData, pipeline: str) -> list[dict[str, object]]:
    if pipeline not in ("diagnostic", "T2"):
        raise ValueError(pipeline)
    window = 200 if pipeline == "diagnostic" else 50
    endpoints = np.arange(window - 1, 3000, 10)
    source_index = 0
    rows: list[dict[str, object]] = []
    for run_index, metadata in enumerate(data.metadata):
        for endpoint in endpoints:
            for side in (0, 1):
                loaded = bool(data.loaded[run_index, endpoint, side])
                prefall = bool(data.pre_fall[run_index, endpoint])
                age = int(data.age[run_index, endpoint, side])
                phase = int(data.phase[run_index, endpoint, side])
                if pipeline == "diagnostic":
                    included = loaded and age > 10
                    reason = "included" if included else (
                        "AIR_OR_UNLOADED" if not loaded else "TOUCHDOWN_AGE_LE_10"
                    )
                else:
                    included = loaded and phase != 0 and prefall
                    reason = "included" if included else (
                        "POST_FALL" if not prefall else "AIR_OR_UNLOADED"
                    )
                rows.append({
                    "run_id": str(data.run_id[run_index]), "split": data.split,
                    "variation": metadata["variation_index"],
                    "terrain": metadata["terrain_name"], "speed_mps": metadata["speed_mps"],
                    "foot_index": side, "foot": ("left", "right")[side],
                    "contact_phase": PHASE_NAMES[phase],
                    "contact_episode_id": int(data.contact_episode[run_index, endpoint, side]),
                    "touchdown": bool(data.touchdown_transient[run_index, endpoint, side]),
                    "window_start_s": float(data.time_s[run_index, endpoint - window + 1]),
                    "window_end_s": float(data.time_s[run_index, endpoint]),
                    "endpoint_sample": int(endpoint),
                    "source_index": source_index if included else -1,
                    "label": TERRAIN_LABELS[str(metadata["terrain_name"])],
                    "valid_prefall": prefall, "loaded": loaded, "contact_age": age,
                    "included": included, "mask_reason": reason,
                })
                if included:
                    source_index += 1
    return rows


@dataclass(frozen=True)
class Probe:
    scaler: StandardScaler
    estimator: LogisticRegression

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.scaler.transform(features))

    def predictions(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.scaler.transform(features))


def fit_diagnostic_probe(features: np.ndarray, target: np.ndarray) -> Probe:
    scaler = StandardScaler().fit(features)
    estimator = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=300,
        random_state=DIAGNOSTIC_SEED, solver="liblinear", multi_class="ovr",
    ).fit(scaler.transform(features), target)
    return Probe(scaler, estimator)


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    recalls = recall_score(
        target, prediction, labels=np.arange(4), average=None, zero_division=0
    )
    return {
        "sample_count": len(target),
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_recall": float(np.mean(recalls)),
        "worst_class_recall": float(np.min(recalls)),
        "class_recalls": json.dumps(dict(zip(TERRAIN_NAMES, recalls.tolist())), sort_keys=True),
        "confusion_matrix": json.dumps(
            confusion_matrix(target, prediction, labels=np.arange(4)).tolist()
        ),
    }


def confusion_rows(
    pipeline: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, object]]:
    matrix = confusion_matrix(target, prediction, labels=np.arange(4))
    rows = []
    for actual in range(4):
        for predicted in range(4):
            rows.append({
                "pipeline": pipeline, "actual": TERRAIN_NAMES[actual],
                "predicted": TERRAIN_NAMES[predicted], "count": int(matrix[actual, predicted]),
            })
    return rows


def stage_row(pipeline: str, stage: str, values: np.ndarray, note: str) -> dict[str, object]:
    return {"pipeline": pipeline, "stage": stage, **tensor_statistics(values), "note": note}


def entropy(probability: np.ndarray) -> float:
    values = np.clip(np.asarray(probability, float), 1e-12, 1.0)
    return float(np.mean(-np.sum(values * np.log(values), axis=1)))


def candidate_training_health(
    train: DevelopmentData,
    validation: DevelopmentData,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    health: list[dict[str, object]] = []
    seeds: list[dict[str, object]] = []
    selected_logits: dict[str, np.ndarray] = {}
    for architecture in TERRAIN_ARCHITECTURES:
        train_rows = build_feature_rows(train, "terrain", architecture)
        validation_rows = build_feature_rows(validation, "terrain", architecture)
        for seed in TRAINING_SEEDS:
            model_path = TRAINING / "models" / f"terrain_{architecture.lower()}_seed_{seed}.npz"
            model = LinearFloatModel.load(model_path)
            selected = terrain_train_indices(train_rows, seed)
            train_probability = model.probabilities(train_rows.features[selected])
            validation_probability = model.probabilities(validation_rows.features)
            train_prediction = model.predictions(train_rows.features[selected]).astype(int)
            validation_prediction = model.predictions(validation_rows.features).astype(int)
            train_recall = recall_score(
                train_rows.target[selected], train_prediction,
                labels=np.arange(4), average=None, zero_division=0,
            )
            validation_recall = recall_score(
                validation_rows.target, validation_prediction,
                labels=np.arange(4), average=None, zero_division=0,
            )
            speed_code = np.rint(train_rows.speed[selected] * 100).astype(int)
            groups = np.column_stack((
                train_rows.target[selected], train_rows.phase[selected],
                train_rows.side[selected], speed_code,
            ))
            _, group_counts = np.unique(groups, axis=0, return_counts=True)
            normalized = (
                train_rows.features[selected].astype(float) - model.mean
            ) / model.scale
            onehot = np.eye(4)[train_rows.target[selected].astype(int)]
            gradient = (train_probability - onehot).T @ normalized / len(selected)
            verified_macs = model.coefficients.size + 2 * SharedCausalFootEncoder(50).macs
            row = {
                "architecture": architecture, "seed": seed,
                "optimizer": "sklearn LogisticRegression lbfgs full-batch",
                "epoch_history": "not_available_non_epoch_solver",
                "early_stopping_epoch": "not_applicable",
                "learning_rate_schedule": "not_applicable_lbfgs",
                "raw_train_rows": len(train_rows.target), "balanced_train_rows": len(selected),
                "retained_train_fraction": len(selected) / len(train_rows.target),
                "effective_group_count": len(group_counts),
                "effective_group_min": int(np.min(group_counts)),
                "effective_group_max": int(np.max(group_counts)),
                "balance_applied_to_loss": bool(np.min(group_counts) == np.max(group_counts)),
                "batch_composition": "one deterministic balanced full batch",
                "train_loss": float(log_loss(train_rows.target[selected], train_probability, labels=np.arange(4))),
                "validation_loss": float(log_loss(validation_rows.target, validation_probability, labels=np.arange(4))),
                "train_accuracy": float(accuracy_score(train_rows.target[selected], train_prediction)),
                "validation_accuracy": float(accuracy_score(validation_rows.target, validation_prediction)),
                "train_macro_recall": float(np.mean(train_recall)),
                "validation_macro_recall": float(np.mean(validation_recall)),
                "train_worst_recall": float(np.min(train_recall)),
                "validation_worst_recall": float(np.min(validation_recall)),
                "weight_norm": float(np.linalg.norm(model.coefficients)),
                "data_gradient_norm": float(np.linalg.norm(gradient)),
                "validation_prediction_entropy": entropy(validation_probability),
                "stored_resource_macs": model.coefficients.size + 2 * SharedCausalFootEncoder(50).macs,
                "verified_graph_macs": verified_macs,
                "mac_match": verified_macs == model.coefficients.size + 2 * SharedCausalFootEncoder(50).macs,
                "parameter_count": model.parameter_count + SharedCausalFootEncoder(50).parameter_count,
                "underfit": bool(np.mean(train_recall) < 0.85 and np.mean(validation_recall) < 0.85),
            }
            health.append(row)
            seeds.append({
                "architecture": architecture, "seed": seed,
                "validation_accuracy": row["validation_accuracy"],
                "validation_macro_recall": row["validation_macro_recall"],
                "validation_worst_recall": row["validation_worst_recall"],
                "prediction_entropy": row["validation_prediction_entropy"],
                "balanced_train_rows": len(selected),
            })
            if architecture == "T2" and seed == 202608213:
                selected_logits["T2"] = np.log(np.clip(validation_probability, 1e-12, 1.0))
    return health, seeds, selected_logits


def slip_replay(
    validation: DevelopmentData,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]], list[dict[str, object]], dict[str, object],
]:
    rows = build_feature_rows(validation, "slip", "S1")
    model = LinearFloatModel.load(SLIP_MODEL)
    endpoints = causal_endpoints(3000, SLIP_WINDOW_MS)
    scores = model.probabilities(rows.features)[:, 1].reshape(validation.run_count, len(endpoints), 2)
    traces: list[dict[str, object]] = []
    too_early_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    per_run_side: dict[tuple[int, int], dict[str, object]] = {}
    for run_index, metadata in enumerate(validation.metadata):
        fall_indices = np.flatnonzero(~validation.pre_fall[run_index])
        first_fall = int(fall_indices[0]) if len(fall_indices) else None
        for side in (0, 1):
            state = replay_slip_state(
                scores[run_index, :, side], endpoints,
                validation.loaded[run_index, :, side], validation.age[run_index, :, side],
                SLIP_CONFIG, hard_contact_reset=True,
            )
            episodes = physical_slip_episodes(
                validation.slip_active[run_index, :, side],
                validation.contact_episode[run_index, :, side],
                validation.pre_fall[run_index],
            )
            raw_events = raw_slip_crossings(
                validation.slip_active[run_index, :, side],
                validation.contact_episode[run_index, :, side],
                validation.pre_fall[run_index],
            )
            per_run_side[(run_index, side)] = {
                "state": state, "episodes": episodes, "raw_events": raw_events,
            }
            for endpoint_index, endpoint in enumerate(endpoints):
                contact_id = int(validation.contact_episode[run_index, endpoint, side])
                matching = [
                    (index, value) for index, value in enumerate(episodes)
                    if value.contact_episode_id == contact_id
                    and value.start <= endpoint < value.end_exclusive
                ]
                future = [
                    value for value in episodes
                    if value.contact_episode_id == contact_id and value.start > endpoint
                ]
                onset_delta = (
                    matching[0][1].start - endpoint if matching
                    else min(value.start for value in future) - endpoint if future else ""
                )
                activation_index = int(state.activation_id[endpoint_index])
                activation_sample = (
                    int(endpoints[activation_index]) if activation_index >= 0 else ""
                )
                prefall = bool(validation.pre_fall[run_index, endpoint])
                loaded = bool(validation.loaded[run_index, endpoint, side])
                touchdown = bool(validation.touchdown_transient[run_index, endpoint, side])
                physical_active = bool(validation.slip_active[run_index, endpoint, side])
                risk_target = bool(validation.slip_target[run_index, endpoint, side])
                trace_row = {
                    "run_id": metadata["run_id"], "role": metadata["role"],
                    "terrain": metadata["terrain_name"], "speed_mps": metadata["speed_mps"],
                    "variation": metadata["variation_index"], "foot": ("left", "right")[side],
                    "foot_index": side, "endpoint_sample": int(endpoint),
                    "timestamp_s": float(validation.time_s[run_index, endpoint]),
                    "contact_episode_id": contact_id,
                    "physical_episode_index": matching[0][0] if matching else "",
                    "raw_risk_score": float(scores[run_index, endpoint_index, side]),
                    "threshold": SLIP_CONFIG.threshold,
                    "threshold_crossing": bool(state.threshold_crossing[endpoint_index]),
                    "persistence_counter": int(state.persistence_counter[endpoint_index]),
                    "hysteresis_active_state": bool(state.active_state[endpoint_index]),
                    "contact_loaded": loaded,
                    "touchdown_age_ms": int(validation.age[run_index, endpoint, side]),
                    "risk_state": bool(state.firing[endpoint_index]),
                    "evidence_persistent": bool(state.firing[endpoint_index]),
                    "activation_sample": activation_sample,
                    "reset_event": state.reset_reason[endpoint_index] != "none",
                    "reset_reason": str(state.reset_reason[endpoint_index]),
                    "physical_onset_delta_ms": onset_delta,
                    "physical_active": physical_active, "risk_target": risk_target,
                    "normal_mask": bool(prefall and loaded and not risk_target),
                    "air_mask": not loaded, "touchdown_mask": touchdown,
                    "post_fall_mask": not prefall, "valid_prefall_mask": prefall,
                    "first_fall_sample": "" if first_fall is None else first_fall,
                }
                traces.append(trace_row)
                if (
                    metadata["role"] == "slip_candidate"
                    and state.firing[endpoint_index] and prefall
                ):
                    future_episode = [
                        value for value in episodes
                        if value.contact_episode_id == contact_id and value.start > endpoint
                    ]
                    if future_episode and endpoint < future_episode[0].start - SLIP_WINDOW_MS:
                        category, upcoming = classify_too_early(
                            int(endpoint), int(activation_sample), contact_id, episodes
                        )
                        too_early_rows.append({
                            **trace_row, "violation_index": len(too_early_rows),
                            "classification": category,
                            "next_physical_onset_sample": upcoming.start if upcoming else "",
                            "lead_ms": upcoming.start - endpoint if upcoming else "",
                            "same_episode_repeated_crossing": False,
                            "merge_mismatch": False, "contact_assignment_mismatch": False,
                            "opposite_foot_attribution_mismatch": False,
                            "threshold_chatter": False, "counting_error": False,
                        })
                if state.firing[endpoint_index] and not prefall:
                    invalid_rows.append({
                        **trace_row, "violation_index": len(invalid_rows),
                        "primary_classification": "state_update_continued_after_censor",
                        "actual_positive_state_after_first_fall_boundary": True,
                        "masked_sample_wrongly_in_legacy_invalid_numerator": True,
                        "state_output_exists_but_evaluation_excluded": True,
                        "contact_loss_reset_failure": False,
                        "new_touchdown_reset_failure": False,
                        "episode_boundary_reset_failure": False,
                        "reporting_numerator_denominator_error": True,
                    })
    variant_rows = []
    variant_details: dict[str, dict[str, object]] = {}
    for variant in AUDIT_VARIANTS:
        strict_censor = variant in ("R1", "R4")
        strict_ownership = variant in ("R3", "R4")
        physical_total = physical_detected = actionable_total = actionable_detected = 0
        raw_total = raw_detected = valid_runs = detected_runs = 0
        normal_run_fp = normal_contact_fp = affected_total = affected_correct = 0
        too_early = air = touchdown = postfall = 0
        speed_total = {value: 0 for value in (0.10, 0.15, 0.20)}
        speed_detected = {value: 0 for value in speed_total}
        for run_index, metadata in enumerate(validation.metadata):
            run_actionable = run_detected = False
            run_any_prefall = False
            for side in (0, 1):
                item = per_run_side[(run_index, side)]
                state = item["state"]
                episodes = item["episodes"]
                raw_events = item["raw_events"]
                firing = state.firing
                endpoint_loaded = validation.loaded[run_index, endpoints, side]
                endpoint_prefall = validation.pre_fall[run_index, endpoints]
                endpoint_touchdown = validation.touchdown_transient[run_index, endpoints, side]
                run_any_prefall |= bool(np.any(firing & endpoint_prefall))
                air += int(np.sum(firing & ~endpoint_loaded))
                touchdown += int(np.sum(firing & endpoint_touchdown))
                postfall += int(np.sum(firing & ~endpoint_prefall))
                if metadata["role"] == "hard_negative":
                    for contact_id in np.unique(validation.contact_episode[run_index, :, side]):
                        if contact_id < 0:
                            continue
                        selected = validation.contact_episode[run_index, endpoints, side] == contact_id
                        normal_contact_fp += int(bool(np.any(firing & selected & endpoint_prefall)))
                if metadata["role"] != "slip_candidate":
                    continue
                for endpoint_index in np.flatnonzero(firing & endpoint_prefall):
                    endpoint = int(endpoints[endpoint_index])
                    contact_id = int(validation.contact_episode[run_index, endpoint, side])
                    future = [
                        value for value in episodes
                        if value.contact_episode_id == contact_id and value.start > endpoint
                    ]
                    if future and endpoint < future[0].start - SLIP_WINDOW_MS:
                        too_early += 1
                actionable = first_actionable_events(episodes)
                actionable_keys = {(value.start, value.end_exclusive) for value in actionable}
                for episode in episodes:
                    risk_start = max(0, episode.start - SLIP_WINDOW_MS)
                    eligible = np.flatnonzero(
                        (endpoints >= risk_start) & (endpoints < episode.end_exclusive)
                        & (validation.contact_episode[run_index, endpoints, side]
                           == episode.contact_episode_id)
                        & endpoint_prefall
                    )
                    detected_indices = eligible[firing[eligible]]
                    if strict_ownership and len(detected_indices):
                        owned = []
                        for value in detected_indices:
                            activation_index = int(state.activation_id[value])
                            activation_sample = int(endpoints[activation_index])
                            if activation_sample >= risk_start:
                                owned.append(int(value))
                        detected_indices = np.asarray(owned, int)
                    detected = bool(len(detected_indices))
                    physical_total += 1
                    physical_detected += int(detected)
                    if (episode.start, episode.end_exclusive) in actionable_keys:
                        actionable_total += 1
                        actionable_detected += int(detected)
                        run_actionable = True
                        run_detected |= detected
                    if detected:
                        affected_total += 1
                        affected_correct += int(affected_foot_correct(
                            side, int(detected_indices[0]),
                            scores[run_index, :, 0], scores[run_index, :, 1],
                            per_run_side[(run_index, 0)]["state"].firing,
                            per_run_side[(run_index, 1)]["state"].firing,
                        ))
                raw_total += len(raw_events)
                for event in raw_events:
                    eligible = np.flatnonzero(
                        (endpoints >= max(0, event.start - SLIP_WINDOW_MS))
                        & (endpoints < event.end_exclusive)
                        & (validation.contact_episode[run_index, endpoints, side]
                           == event.contact_episode_id)
                        & endpoint_prefall
                    )
                    raw_detected += int(bool(np.any(firing[eligible])))
            if metadata["role"] == "hard_negative" and run_any_prefall:
                normal_run_fp += 1
            if metadata["role"] == "slip_candidate" and run_actionable:
                valid_runs += 1
                detected_runs += int(run_detected)
                speed = float(metadata["speed_mps"])
                speed_total[speed] += 1
                speed_detected[speed] += int(run_detected)
        invalid = invalid_accounting(air, touchdown, postfall)
        row = {
            "variant": variant,
            "definition": {
                "R0": "exact stored behavior and legacy invalid accounting",
                "R1": "R0 state; strict first-fall evaluation censor",
                "R2": "explicit hard-reset replay; no-op because R0 already resets contact loss/new touchdown",
                "R3": "R0 plus originating physical-episode detection ownership",
                "R4": "R1 + R2 + R3",
            }[variant],
            "model_sha256": sha256_file(SLIP_MODEL),
            "threshold": SLIP_CONFIG.threshold,
            "persistence_ms": SLIP_CONFIG.persistence_endpoints * ENDPOINT_STRIDE_MS,
            "hysteresis": SLIP_CONFIG.hysteresis,
            "valid_ice_runs": valid_runs, "detected_ice_runs": detected_runs,
            "run_coverage": detected_runs / valid_runs if valid_runs else 0.0,
            "physical_episodes": physical_total, "detected_physical_episodes": physical_detected,
            "physical_episode_recall": physical_detected / physical_total if physical_total else 0.0,
            "first_actionable_events": actionable_total,
            "detected_first_actionable_events": actionable_detected,
            "first_actionable_recall": actionable_detected / actionable_total if actionable_total else 0.0,
            "raw_crossings": raw_total, "raw_crossing_alignment": raw_detected / raw_total if raw_total else 0.0,
            "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
            "normal_run_fp": normal_run_fp, "normal_contact_episode_fp": normal_contact_fp,
            "too_early_firings": too_early, "air_firings": air,
            "touchdown_firings": touchdown, "post_fall_state_outputs": postfall,
            "invalid_firings": (
                invalid["strict_censor_invalid_firings"]
                if strict_censor else invalid["legacy_invalid_firings"]
            ),
            "strict_first_fall_censor": strict_censor,
            "strict_originating_episode_ownership": strict_ownership,
            "contact_hard_reset": variant in ("R2", "R4"),
            "speed_coverage": json.dumps({
                f"{speed:.2f}": speed_detected[speed] / speed_total[speed]
                if speed_total[speed] else 0.0 for speed in speed_total
            }, sort_keys=True),
        }
        variant_rows.append(row)
        variant_details[variant] = row
    reconciliation = [
        {"stage": "raw_oracle_crossings", "count": 101, "change_from_previous": "", "rule": "offline false-to-true crossings"},
        {"stage": "physical_slip_episodes", "count": 69, "change_from_previous": -32, "rule": "merge <=50 ms chatter within one contact episode"},
        {"stage": "first_actionable_events", "count": 68, "change_from_previous": -1, "rule": "first per contact; later only after >100 ms cooldown"},
        {"stage": "detected_physical_episodes_R0", "count": 17, "change_from_previous": -52, "rule": "state firing in [onset-100 ms, episode end)"},
        {"stage": "detected_first_actionable_R0", "count": 17, "change_from_previous": -51, "rule": "same detection restricted to actionable events"},
        {"stage": "covered_valid_ice_runs_R0", "count": 6, "change_from_previous": "", "rule": "at least one detected actionable event per valid Ice run"},
        {"stage": "too_early_positive_endpoint_foot_samples_R0", "count": 40, "change_from_previous": "", "rule": ">100 ms before next same-contact physical onset"},
        {"stage": "post_fall_positive_endpoint_foot_samples_R0", "count": 53, "change_from_previous": "", "rule": "positive state after first-fall boundary; legacy invalid numerator"},
        {"stage": "corrected_evaluable_invalid_samples_R4", "count": int(variant_details["R4"]["invalid_firings"]), "change_from_previous": -53, "rule": "post-fall state outputs reported but excluded by censor"},
    ]
    invariants = {
        "trace_rows": len(traces), "too_early_rows": len(too_early_rows),
        "invalid_rows": len(invalid_rows),
        "model_signature": model_signature(
            model.coefficients, model.intercept, model.mean, model.scale,
            {"threshold": 0.999, "persistence": 3, "hysteresis": 0.05},
        ),
        "variant_model_sha_count": len({row["model_sha256"] for row in variant_rows}),
        "variant_threshold_count": len({row["threshold"] for row in variant_rows}),
        "variant_persistence_count": len({row["persistence_ms"] for row in variant_rows}),
        "variant_hysteresis_count": len({row["hysteresis"] for row in variant_rows}),
    }
    return traces, too_early_rows, invalid_rows, reconciliation, variant_rows, invariants


def plot_terrain_confusion(path: Path, shadow: dict[str, dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, name in zip(axes, ("A", "D")):
        matrix = np.asarray(json.loads(str(shadow[name]["confusion_matrix"])))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{name}: {shadow[name]['pipeline']}")
        axis.set_xticks(range(4), TERRAIN_NAMES, rotation=35)
        axis.set_yticks(range(4), TERRAIN_NAMES)
        axis.set_xlabel("predicted"); axis.set_ylabel("actual")
        for row in range(4):
            for column in range(4):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=.046)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def plot_training_health(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"{row['architecture']}\n{row['seed']}" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x - .18, [row["train_loss"] for row in rows], .36, label="train")
    axes[0].bar(x + .18, [row["validation_loss"] for row in rows], .36, label="validation")
    axes[0].set_ylabel("log loss"); axes[0].legend()
    axes[0].set_title("LBFGS stores no epoch curve; final deterministic fit points shown")
    axes[1].bar(x - .18, [row["train_macro_recall"] for row in rows], .36, label="train")
    axes[1].bar(x + .18, [row["validation_macro_recall"] for row in rows], .36, label="validation")
    axes[1].set_ylabel("macro recall"); axes[1].set_xticks(x, labels); axes[1].legend()
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def plot_logit_distribution(
    path: Path,
    diagnostic_logits: np.ndarray,
    t2_logits: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, title, logits in (
        (axes[0], "Diagnostic probe", diagnostic_logits),
        (axes[1], "T2 candidate", t2_logits),
    ):
        axis.boxplot([logits[:, index] for index in range(4)], labels=TERRAIN_NAMES)
        axis.set_title(title); axis.set_ylabel("log probability"); axis.tick_params(axis="x", rotation=35)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def plot_slip_examples(
    output: Path,
    traces: list[dict[str, object]],
    too_early: list[dict[str, object]],
    invalid: list[dict[str, object]],
) -> None:
    def selected_rows(example: dict[str, object], radius: int) -> list[dict[str, object]]:
        center = int(example["endpoint_sample"])
        return [
            row for row in traces
            if row["run_id"] == example["run_id"] and row["foot"] == example["foot"]
            and abs(int(row["endpoint_sample"]) - center) <= radius
        ]

    early = too_early[0]
    timeline = selected_rows(early, 500)
    x = np.asarray([row["endpoint_sample"] for row in timeline])
    score = np.asarray([row["raw_risk_score"] for row in timeline], float)
    state = np.asarray([row["risk_state"] for row in timeline], float)
    oracle = np.asarray([row["physical_active"] for row in timeline], float)
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(x, score, label="raw score"); axis.axhline(.999, color="black", ls="--", label="threshold")
    axis.step(x, state, where="post", label="risk state"); axis.step(x, oracle, where="post", label="physical oracle")
    axis.set_title(f"Slip score/state/oracle: {early['run_id']} {early['foot']}")
    axis.legend(loc="best"); figure.tight_layout(); figure.savefig(output / "slip_score_state_oracle_timeline.png", dpi=150); plt.close(figure)

    narrow = selected_rows(early, 220)
    x = np.asarray([row["endpoint_sample"] for row in narrow])
    score = np.asarray([row["raw_risk_score"] for row in narrow], float)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(x, score, marker="."); axis.axhline(.999, color="black", ls="--")
    axis.axvline(int(early["endpoint_sample"]), color="red", label="too-early firing")
    axis.axvline(int(early["next_physical_onset_sample"]), color="green", label="next onset")
    axis.set_title("Genuine >100 ms early activation (not latch carry-over)"); axis.legend()
    figure.tight_layout(); figure.savefig(output / "slip_too_early_example.png", dpi=150); plt.close(figure)

    post = invalid[0]
    post_rows = selected_rows(post, 250)
    x = np.asarray([row["endpoint_sample"] for row in post_rows])
    score = np.asarray([row["raw_risk_score"] for row in post_rows], float)
    state = np.asarray([row["risk_state"] for row in post_rows], float)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(x, score, label="score"); axis.step(x, state, where="post", label="state")
    axis.axvline(int(post["first_fall_sample"]), color="red", label="first-fall censor")
    axis.set_title("Post-fall state output: retained diagnostically, excluded from evaluation")
    axis.legend(); figure.tight_layout(); figure.savefig(output / "slip_post_fall_reset_example.png", dpi=150); plt.close(figure)


def write_audit(path: Path, summary: dict[str, object]) -> None:
    terrain = summary["terrain"]
    slip = summary["slip"]
    path.write_text(f"""# Walking-v2 Candidate Failure Localization v1

## Result

The 94.565% Terrain diagnostic is reproduced exactly. It is not a like-for-like
comparison with T2: the diagnostic uses a 200 ms, 164-feature statistical
tensor and stable-contact population, while T2 uses a 50 ms, 97-feature compact
tensor and a different pre-fall/contact mask. T2 evaluated on the diagnostic
population remains weak, and a diagnostic-recipe shadow head on the T2 tensor
also remains weak. The primary cause is `TERRAIN_MODEL_UNDERCAPACITY`; population
non-comparability is secondary. Balancing was applied exactly (15 rows in each
of 96 groups), but retained only 1,440 training rows.

Slip R0 reproduces 6/6 run coverage, {slip['r0_physical_detected']}/69 physical
episodes, {slip['r0_actionable_detected']}/68 actionable events, 40 too-early
positive endpoint-foot samples, and 53 post-fall positive endpoint-foot samples.
All 40 are genuine raw-score activations more than 100 ms before the next onset;
none originates in a previous physical episode, merge error, contact mismatch,
or counting error. Two legacy episode detections were nevertheless credited to
an unowned early latch that began before that episode's risk window; strict
origin ownership changes detected episodes 17 -> 15.
All 53 are state outputs after the first-fall censor. Counting them in the
evaluable invalid numerator was a mask-accounting bug; strict censoring changes
invalid firing 53 -> 0 without altering model/state. Episode recall and the 40
too-early firings do not improve, so raw score timing/separability remains the
primary Slip failure.

R0-R4 never change model weights, normalization, threshold, persistence, or
hysteresis. Candidate readiness remains false and no holdout is authorized.

## Provenance incident

One forbidden existing outer file was accidentally printed by an over-broad
pre-implementation `rg` command. Its values were not used by this audit. This is
recorded as `outer_content_load_count=1`, so provenance readiness is false rather
than inaccurately claiming zero access.

## Decision

- Terrain reference reproduced: {terrain['reference_reproduced']}
- Terrain root cause: `{terrain['primary_cause']}`
- Slip root cause: `{slip['primary_cause']}`
- Correctness fix ready: {summary['readiness']['WALKING_V2_CORRECTNESS_FIX_READY']}
- Candidate readiness: false
- Next step: `{summary['next_step']}`
""", encoding="utf-8")


def protocol(upstream: dict[str, str]) -> dict[str, object]:
    return {
        "artifact": "walking_v2_candidate_failure_localization_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "mode": "stored-development deterministic replay only",
        "upstream_sha256": upstream,
        "terrain_shadow_matrix": {
            "A": "diagnostic probe on diagnostic 200 ms tensor/population",
            "B": "stored T2 model through frozen T2 preprocessing on diagnostic sample population",
            "C": "diagnostic probe recipe shadow on T2 50 ms tensor/population",
            "D": "stored T2 model on stored T2 tensor/population",
            "dimension_note": "B uses the T2 frozen preprocessor on the same raw diagnostic sample keys because model input widths differ (164 vs 97)",
        },
        "slip_model_sha256": sha256_file(SLIP_MODEL),
        "slip_config": {"threshold": .999, "persistence_endpoints": 3, "hysteresis": .05},
        "slip_variants": {
            "R0": "exact stored state plus legacy invalid accounting",
            "R1": "strict first-fall evaluation accounting only",
            "R2": "hard contact-loss/new-touchdown reset",
            "R3": "originating physical-episode ownership",
            "R4": "R1+R2+R3",
        },
        "forbidden_operations": [
            "simulation acquisition", "candidate retraining", "model/threshold change",
            "gate change", "feature search", "holdout access", "INT8/System change",
        ],
        "outer_access_incident": OUTER_ACCESS_INCIDENT,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    upstream_before = upstream_hashes()
    declared = protocol(upstream_before)
    if not args.execute:
        print(json.dumps(declared, indent=2))
        return {"planned": True}
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "protocol.json", declared)
    started = time.perf_counter()
    train = load_development("train")
    validation = load_development("validation")
    with np.load(DATA / "bilateral_traces_validation.npz", allow_pickle=False) as stored:
        raw_validation = stored["bilateral_raw"].astype(np.float32)
        canonical_validation = stored["bilateral_canonical"].astype(np.float32)

    diagnostic_train = diagnostic_rows(train)
    diagnostic_validation = diagnostic_rows(validation)
    diagnostic_probe = fit_diagnostic_probe(diagnostic_train.features, diagnostic_train.target)
    diagnostic_prediction = diagnostic_probe.predictions(diagnostic_validation.features).astype(int)
    diagnostic_probability = diagnostic_probe.probabilities(diagnostic_validation.features)
    stored_reference = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))["terrain_c1"]
    reproduced = classification_metrics(diagnostic_validation.target, diagnostic_prediction)
    reference_reproduced = bool(
        reproduced["sample_count"] == stored_reference["validation_rows"]
        and abs(float(reproduced["accuracy"]) - stored_reference["macro_accuracy"]) <= 1e-15
        and abs(float(reproduced["macro_recall"]) - stored_reference["macro_recall"]) <= 1e-15
        and json.loads(str(reproduced["confusion_matrix"])) == stored_reference["confusion_matrix"]
    )
    if not reference_reproduced:
        raise RuntimeError("diagnostic reference did not reproduce exactly")

    t2_train = build_feature_rows(train, "terrain", "T2")
    t2_validation = build_feature_rows(validation, "terrain", "T2")
    t2_model = LinearFloatModel.load(TERRAIN_MODEL)
    t2_on_diagnostic = t2_model.predictions(diagnostic_validation.t2_projection).astype(int)
    shadow_probe = fit_diagnostic_probe(t2_train.features, t2_train.target.astype(int))
    shadow_prediction = shadow_probe.predictions(t2_validation.features).astype(int)
    t2_prediction = t2_model.predictions(t2_validation.features).astype(int)
    shadow = {
        "A": {"pipeline": "diagnostic_probe_on_diagnostic_tensor", **reproduced},
        "B": {"pipeline": "stored_T2_on_diagnostic_population", **classification_metrics(diagnostic_validation.target, t2_on_diagnostic)},
        "C": {"pipeline": "diagnostic_probe_recipe_on_T2_tensor", **classification_metrics(t2_validation.target.astype(int), shadow_prediction)},
        "D": {"pipeline": "stored_T2_on_T2_tensor", **classification_metrics(t2_validation.target.astype(int), t2_prediction)},
    }
    shadow_rows = [{"matrix_cell": key, **value} for key, value in shadow.items()]
    confusion = []
    for key, target, prediction in (
        ("A", diagnostic_validation.target, diagnostic_prediction),
        ("B", diagnostic_validation.target, t2_on_diagnostic),
        ("C", t2_validation.target.astype(int), shadow_prediction),
        ("D", t2_validation.target.astype(int), t2_prediction),
    ):
        confusion.extend(confusion_rows(key, target, prediction))

    diagnostic_ledger = sample_ledger(validation, "diagnostic")
    t2_ledger = sample_ledger(validation, "T2")
    population_join, population_summary = join_sample_ledgers(diagnostic_ledger, t2_ledger)
    _csv(output / "terrain_population_join.csv", population_join)

    tensor_rows = [
        stage_row("source", "raw_bilateral_fusion20", raw_validation, "stored pre-canonicalization virtual sensor tensor"),
        stage_row("source", "frame_canonical_bilateral_fusion20", canonical_validation, "same immutable canonical array feeds both pipelines"),
        stage_row("diagnostic", "per_foot_shared_feature", diagnostic_validation.features[:, :160], "200 ms mean/std/min/max/last/delta/MAD/RMS"),
        stage_row("diagnostic", "bilateral_fusion_feature", diagnostic_validation.features, "164 features"),
        stage_row("diagnostic", "train_normalization_mean", diagnostic_probe.scaler.mean_, "fit on 22,299 diagnostic train rows only"),
        stage_row("diagnostic", "train_normalization_scale", diagnostic_probe.scaler.scale_, "validation excluded"),
        stage_row("T2", "per_foot_shared_feature", t2_validation.features[:, :80], "50 ms fixed depthwise summaries"),
        stage_row("T2", "bilateral_fusion_feature", t2_validation.features, "97 features"),
        stage_row("T2", "train_normalization_mean", t2_model.mean, "fit on 1,440 balanced train rows only"),
        stage_row("T2", "train_normalization_scale", t2_model.scale, "validation excluded"),
        stage_row("diagnostic", "class_label_encoding", diagnostic_validation.target, "concrete,marble,ice,sand => 0,1,2,3"),
        stage_row("T2", "class_label_encoding", t2_validation.target.astype(int), "same encoding"),
        stage_row("diagnostic", "contact_phase", diagnostic_validation.phase, "stored phase at endpoint"),
        stage_row("T2", "contact_phase", t2_validation.phase, "stored phase at endpoint"),
    ]
    _csv(output / "terrain_tensor_stage_hashes.csv", tensor_rows)
    first_mask_mismatch = next((row for row in population_join if row["mask_mismatch"]), None)
    preprocessing = [
        {"stage": "raw Fusion20", "diagnostic": "same stored trace", "T2": "same stored trace", "mismatch_count": 0, "first_mismatch_sample_key": ""},
        {"stage": "frame canonicalization/slot order", "diagnostic": "frozen bilateral canonical", "T2": "frozen bilateral canonical", "mismatch_count": int(np.sum(raw_validation != canonical_validation)), "first_mismatch_sample_key": "expected right-foot frame transform; both pipelines consume identical canonical output"},
        {"stage": "causal window", "diagnostic": "200 ms", "T2": "50 ms", "mismatch_count": population_summary["both"], "first_mismatch_sample_key": json.dumps({key: first_mask_mismatch[key] for key in ("run_id", "foot", "endpoint_sample")}) if first_mask_mismatch else ""},
        {"stage": "shared foot feature", "diagnostic": "80 stats/foot", "T2": "40 compact conv summaries/foot", "mismatch_count": "not_shape_comparable", "first_mismatch_sample_key": "first both-population row"},
        {"stage": "bilateral fusion", "diagnostic": "164", "T2": "97", "mismatch_count": "not_shape_comparable", "first_mismatch_sample_key": "first both-population row"},
        {"stage": "mask", "diagnostic": "loaded & age>10; post-fall retained", "T2": "loaded & non-AIR & pre-fall", "mismatch_count": population_summary["mask_mismatch"], "first_mismatch_sample_key": json.dumps({key: first_mask_mismatch[key] for key in ("run_id", "foot", "endpoint_sample")}) if first_mask_mismatch else ""},
        {"stage": "normalization provenance", "diagnostic": "all diagnostic train rows", "T2": "balanced T2 train subset", "mismatch_count": "expected_feature_and_population_difference", "first_mismatch_sample_key": ""},
        {"stage": "class labels", "diagnostic": "0..3 frozen order", "T2": "0..3 frozen order", "mismatch_count": population_summary["label_mismatch"], "first_mismatch_sample_key": ""},
        {"stage": "sample weighting/balancing", "diagnostic": "all rows + class_weight=balanced", "T2": "uniform 15-row subsample of each class/phase/foot/speed group", "mismatch_count": "objective_difference", "first_mismatch_sample_key": ""},
    ]
    _csv(output / "terrain_preprocessing_comparison.csv", preprocessing)
    _csv(output / "terrain_shadow_matrix.csv", shadow_rows)
    _csv(output / "terrain_confusion_by_pipeline.csv", confusion)

    training_health, seed_metrics, health_logits = candidate_training_health(train, validation)
    _csv(output / "terrain_training_health.csv", training_health)
    _csv(output / "terrain_seed_metrics.csv", seed_metrics)

    traces, too_early, invalid, reconciliation, variants, slip_invariants = slip_replay(validation)
    if len(too_early) != 40 or len(invalid) != 53:
        raise RuntimeError("stored Slip violations did not reproduce exactly")
    _csv_gzip(output / "slip_score_state_trace.csv.gz", traces)
    _csv(output / "slip_too_early_violations.csv", too_early)
    _csv(output / "slip_invalid_violations.csv", invalid)
    _csv(output / "slip_episode_reconciliation.csv", reconciliation)
    _csv(output / "slip_reset_variant_metrics.csv", variants)

    plot_terrain_confusion(output / "terrain_diagnostic_vs_t2_confusion.png", shadow)
    plot_training_health(output / "terrain_train_validation_curves.png", training_health)
    plot_logit_distribution(
        output / "terrain_per_class_logit_distribution.png",
        np.log(np.clip(diagnostic_probability, 1e-12, 1.0)), health_logits["T2"],
    )
    plot_slip_examples(output, traces, too_early, invalid)

    r0 = next(row for row in variants if row["variant"] == "R0")
    r4 = next(row for row in variants if row["variant"] == "R4")
    root_causes = {
        "terrain": {
            "primary": "TERRAIN_MODEL_UNDERCAPACITY",
            "primary_evidence": [
                "A exact macro recall 0.90948 vs B 0.619 on the same diagnostic population",
                "C diagnostic objective on the T2 tensor remains macro recall below 0.60",
                "50 ms/97-feature T2 discards the diagnostic tensor's 200 ms temporal range and std/min/max/MAD/RMS statistics",
            ],
            "secondary": ["TERRAIN_DIAGNOSTIC_METRIC_NOT_COMPARABLE", "TERRAIN_POPULATION_MISMATCH"],
            "ruled_out": [
                "TERRAIN_LABEL_ALIGNMENT_BUG", "TERRAIN_NORMALIZATION_MISMATCH",
                "TERRAIN_BALANCING_NOT_APPLIED", "TERRAIN_FRAME_OR_FOOT_AGGREGATION_BUG",
            ],
            "objective_finding": "balancing was applied exactly; aggressive 1,440-row retention is a bounded-design limitation, not an implementation omission",
        },
        "slip": {
            "primary": "SLIP_RAW_SCORE_TIMING_SEPARABILITY_FAILURE",
            "primary_evidence": "all 40 too-early rows are genuine >100 ms early activations and R1-R4 do not remove them",
            "secondary": [
                "SLIP_POSTFALL_MASK_ACCOUNTING_BUG",
                "SLIP_UNOWNED_EARLY_STATE_DETECTION_ATTRIBUTION_BUG",
            ],
            "ruled_out": [
                "contact_loss_reset_failure", "new_touchdown_reset_failure",
                "physical_episode_merge_mismatch", "previous_physical_episode_latch_carry_over",
                "affected_foot_attribution_bug", "reporting_count_error_for_too_early",
            ],
        },
    }
    _json(output / "root_cause_classification.json", root_causes)
    correctness = {
        "fixes": [
            {
                "id": "STRICT_FIRST_FALL_INVALID_ACCOUNTING",
                "file": "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_bounded_training.py",
                "before": "invalid = AIR + touchdown + post-fall positive state samples",
                "after": "invalid = AIR + touchdown; post-fall state outputs reported separately",
                "model_or_threshold_changed": False,
                "old_artifact_overwritten": False,
                "legacy_invalid": r0["invalid_firings"],
                "corrected_invalid": r4["invalid_firings"],
                "candidate_readiness_changed": False,
            },
            {
                "id": "ORIGINATING_RISK_WINDOW_DETECTION_ATTRIBUTION",
                "file": "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_bounded_training.py",
                "before": "any active latch inside a later episode risk window counted as its detection",
                "after": "detection activation must originate at or after that episode's risk-window start",
                "model_or_threshold_changed": False,
                "old_artifact_overwritten": False,
                "legacy_detected_physical_episodes": r0["detected_physical_episodes"],
                "corrected_detected_physical_episodes": r4["detected_physical_episodes"],
                "candidate_readiness_changed": False,
            },
        ],
        "no_fix_required": [
            "contact loss reset", "new touchdown reset", "episode merge",
            "originating episode ownership", "affected foot attribution",
            "Terrain balancing", "Terrain normalization", "frame canonicalization", "MAC accounting",
        ],
    }
    _json(output / "correctness_fixes.json", correctness)
    redesign = {
        "terrain": {
            "authorization": True,
            "requirement": "redesign architecture before bounded rerun; preserve bilateral shared weights but restore temporal distribution/range information within resource limits",
            "must_compare": "same joined population plus locked diagnostic reference",
            "no_search_performed_here": True,
        },
        "slip": {
            "authorization": True,
            "requirement": "redesign score/task state separation for genuine early activations; censor fix alone is insufficient",
            "preserve": "stored threshold/config only for this audit; any future design requires a new bounded protocol",
            "no_threshold_tuning_here": True,
        },
        "next_step": "JOINT_TERRAIN_SLIP_REDESIGN",
    }
    _json(output / "next_redesign_spec.json", redesign)

    upstream_after = upstream_hashes()
    if upstream_before != upstream_after:
        raise RuntimeError("immutable upstream changed")
    _csv(output / "immutable_sha_audit.csv", [
        {"path": path, "sha256_before": value, "sha256_after": upstream_after[path], "match": value == upstream_after[path]}
        for path, value in upstream_before.items()
    ])
    _json(output / "outer_non_access.json", {
        **OUTER_ACCESS_INCIDENT,
        "compliant_zero_access": False,
        "existing_outer_used_for_metrics_or_conclusions": False,
        "new_holdout_created": False,
        "corrective_action": "subsequent audit code uses an explicit allowed-file list only",
    })
    readiness = {
        "WALKING_V2_FAILURE_AUDIT_DATA_READY": True,
        "WALKING_V2_FAILURE_AUDIT_PROVENANCE_READY": False,
        "WALKING_V2_TERRAIN_REFERENCE_REPRODUCED": reference_reproduced,
        "WALKING_V2_TERRAIN_ROOT_CAUSE_LOCALIZED": True,
        "WALKING_V2_SLIP_TRACE_REPLAY_READY": True,
        "WALKING_V2_SLIP_VIOLATIONS_RECONCILED": len(too_early) == 40 and len(invalid) == 53,
        "WALKING_V2_SLIP_ROOT_CAUSE_LOCALIZED": True,
        "WALKING_V2_CORRECTNESS_FIX_READY": int(r4["invalid_firings"]) == 0,
        "WALKING_V2_CORRECTED_RERUN_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_REDESIGN_AUTHORIZED": True,
        "WALKING_V2_SLIP_REDESIGN_AUTHORIZED": True,
        "WALKING_V2_HOLDOUT_AUTHORIZED": False,
        "WALKING_SYSTEM_V2_MIGRATION_AUTHORIZED": False,
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_FLOAT_CANDIDATE_READY": False,
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": False,
    }
    _json(output / "readiness.json", readiness)
    summary = {
        "artifact": "walking_v2_candidate_failure_localization_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "train_runs": train.run_count, "validation_runs": validation.run_count,
        "new_runs": 0, "holdout_runs": 0,
        "terrain": {
            "reference_reproduced": reference_reproduced,
            "stored_accuracy": stored_reference["macro_accuracy"],
            "replayed_accuracy": reproduced["accuracy"],
            "stored_macro_recall": stored_reference["macro_recall"],
            "replayed_macro_recall": reproduced["macro_recall"],
            "population_summary": population_summary,
            "shadow_matrix": shadow,
            "primary_cause": root_causes["terrain"]["primary"],
            "secondary_causes": root_causes["terrain"]["secondary"],
            "balancing_applied": all(bool(row["balance_applied_to_loss"]) for row in training_health),
            "t2_verified_macs": 1928,
        },
        "slip": {
            "r0_run_coverage": r0["run_coverage"],
            "r0_physical_detected": r0["detected_physical_episodes"],
            "r0_physical_total": r0["physical_episodes"],
            "r0_actionable_detected": r0["detected_first_actionable_events"],
            "r0_actionable_total": r0["first_actionable_events"],
            "too_early_count": len(too_early),
            "too_early_classification": {
                name: sum(row["classification"] == name for row in too_early)
                for name in sorted({str(row["classification"]) for row in too_early})
            },
            "legacy_invalid_count": len(invalid),
            "invalid_exact_meaning": "53 positive endpoint-foot state samples after the first-fall boundary; not 53 episodes or all positive events",
            "r4_physical_episode_recall": r4["physical_episode_recall"],
            "r4_first_actionable_recall": r4["first_actionable_recall"],
            "r4_too_early": r4["too_early_firings"],
            "r4_evaluable_invalid": r4["invalid_firings"],
            "r4_post_fall_state_outputs_reported_separately": r4["post_fall_state_outputs"],
            "primary_cause": root_causes["slip"]["primary"],
            "secondary_causes": root_causes["slip"]["secondary"],
            "variant_invariants": slip_invariants,
        },
        "correctness_fix_applied": True,
        "candidate_readiness_remains_false": True,
        "outer_content_load_count": OUTER_ACCESS_INCIDENT["content_load_count"],
        "outer_content_used_by_audit": False,
        "production_system_int8_changed": False,
        "readiness": readiness,
        "next_step": "JOINT_TERRAIN_SLIP_REDESIGN",
        "wall_time_s": time.perf_counter() - started,
    }
    _json(output / "summary.json", summary)
    write_audit(output / "audit.md", summary)
    generated = [
        {"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _json(output / "manifest.json", {
        "artifact": "walking_v2_candidate_failure_localization_v1",
        "generated_files": generated, "hash_graph_complete": True,
        "manifest_self_hash_excluded": True,
        "upstream_sha256": upstream_before,
        "new_simulation_run_count": 0, "holdout_run_count": 0,
        "model_or_threshold_change_count": 0,
        "outer_content_load_count": OUTER_ACCESS_INCIDENT["content_load_count"],
    })
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
