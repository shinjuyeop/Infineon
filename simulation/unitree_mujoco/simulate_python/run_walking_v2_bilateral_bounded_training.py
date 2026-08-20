"""Train bounded walking-v2 bilateral Terrain and Slip float candidates.

Only the fresh 120-run bilateral development artifact is loaded. Existing
outer/holdout/spatial/final arrays are fail-closed. A new blind holdout may be
materialized only after both locked development gates pass and a selection
lock has been written.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from walking_v2_bilateral_bounded_training import (
    ENDPOINT_STRIDE_MS,
    SLIP_ARCHITECTURES,
    SLIP_HYSTERESIS_GRID,
    SLIP_PERSISTENCE_GRID,
    SLIP_THRESHOLD_GRID,
    SLIP_WINDOW_MS,
    TERRAIN_ARCHITECTURES,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    TERRAIN_WINDOW_MS,
    TRAINING_SEEDS,
    LinearFloatModel,
    SharedCausalFootEncoder,
    SlipStateConfig,
    affected_foot_correct,
    balanced_indices,
    causal_endpoints,
    deterministic_slip_selection,
    deterministic_terrain_selection,
    episode_semantics_contract,
    first_actionable_events,
    fit_linear_float,
    holdout_authorized,
    input_contract,
    physical_slip_episodes,
    raw_slip_crossings,
    risk_firing_is_too_early,
    runtime_feature,
    runtime_scope_contract,
    sha256_file,
    sink_deferral_contract,
    slip_gate,
    stateful_slip_firing,
    terrain_gate,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SIMULATION_DIR.parent
SOURCE = SIMULATION_DIR / "outputs" / "walking_bilateral_sensor_sink_observability_v2"
OUTPUT = SIMULATION_DIR / "outputs" / "walking_v2_bilateral_bounded_training"
HOLDOUT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_v2_bilateral_blind_holdout"
STARTING_CHECKPOINT = "86ce118eca20dbd8cc3720bf533b8133f812b18c"

ALLOWED_UPSTREAM_FILES = (
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/manifest.json",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/summary.json",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/bilateral_traces_train.npz",
    "simulation/outputs/walking_bilateral_sensor_sink_observability_v2/bilateral_traces_validation.npz",
    "simulation/outputs/walking_hazard_operational_label_contract_v2/summary.json",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json",
    "simulation/outputs/walking_fusion10_observability_audit_v1/summary.json",
    "simulation/outputs/walking_bounded_retraining_v1/summary.json",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json",
    "simulation/outputs/terrain_static_reference_v4/selected_model.keras",
    "simulation/outputs/terrain_static_reference_v4/normalization.json",
    "simulation/outputs/terrain_fast_reflex_v2_int8/slip/model_int8.tflite",
    "simulation/outputs/terrain_fast_reflex_v2_int8/sink/model_int8.tflite",
)
FORBIDDEN_CONTENT_NAMESPACES = (
    "outer_validation_traces", "sink_holdout/traces", "spatial", "final_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--holdout-output-dir", type=Path, default=HOLDOUT_OUTPUT)
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


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upstream_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ALLOWED_UPSTREAM_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = sha256_file(path)
    return result


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
    touchdown_transient: np.ndarray
    slip_target: np.ndarray
    slip_active: np.ndarray
    metadata: tuple[dict[str, object], ...]

    @property
    def run_count(self) -> int:
        return len(self.run_id)


def load_development(split: str) -> DevelopmentData:
    if split not in ("train", "validation"):
        raise ValueError(split)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    run_rows = {
        str(row["run_id"]): row for row in manifest["runs"]
        if row["split"] == f"development_{split}"
    }
    with np.load(SOURCE / f"bilateral_traces_{split}.npz", allow_pickle=False) as values:
        run_id = values["run_id"].astype(str)
        data = DevelopmentData(
            split=split,
            run_id=run_id,
            time_s=values["time_s"].astype(np.float64),
            bilateral=values["bilateral_canonical"].astype(np.float32),
            loaded=values["force_loaded"].astype(bool),
            age=values["contact_age"].astype(np.int32),
            phase=values["gait_phase_code"].astype(np.int8),
            pre_fall=values["pre_fall_valid"].astype(bool),
            contact_episode=values["contact_episode_id"].astype(np.int32),
            touchdown_transient=values["touchdown_transient"].astype(bool),
            slip_target=values["slip_risk_target"].astype(bool),
            slip_active=values["slip_physical_active"].astype(bool),
            metadata=tuple(run_rows[str(value)] for value in run_id),
        )
    expected = 72 if split == "train" else 48
    if (
        data.run_count != expected
        or data.time_s.shape != (expected, 3000)
        or data.bilateral.shape != (expected, 3000, 20)
        or not np.allclose(data.time_s[:, 0], 0.001, rtol=0.0, atol=1e-15)
        or not np.allclose(np.diff(data.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"unexpected {split} dataset shape")
    return data


@dataclass(frozen=True)
class FeatureRows:
    features: np.ndarray
    target: np.ndarray
    run_index: np.ndarray
    endpoint: np.ndarray
    side: np.ndarray
    phase: np.ndarray
    speed: np.ndarray
    role: np.ndarray
    terrain: np.ndarray
    runtime_eligible: np.ndarray
    pre_fall: np.ndarray
    touchdown: np.ndarray


def build_feature_rows(data: DevelopmentData, task: str, architecture: str) -> FeatureRows:
    if task == "terrain":
        window = TERRAIN_WINDOW_MS
        if architecture not in TERRAIN_ARCHITECTURES:
            raise ValueError(architecture)
    elif task == "slip":
        window = SLIP_WINDOW_MS
        if architecture not in SLIP_ARCHITECTURES:
            raise ValueError(architecture)
    else:
        raise ValueError(task)
    encoder = SharedCausalFootEncoder(window)
    features: list[np.ndarray] = []
    fields: dict[str, list[object]] = {key: [] for key in (
        "target", "run_index", "endpoint", "side", "phase", "speed", "role",
        "terrain", "runtime_eligible", "pre_fall", "touchdown",
    )}
    endpoints = causal_endpoints(3000, window)
    for run_index, metadata in enumerate(data.metadata):
        for endpoint in endpoints:
            for side in (0, 1):
                current_phase = int(data.phase[run_index, endpoint, side])
                stable_contact = bool(
                    data.loaded[run_index, endpoint, side]
                    and data.age[run_index, endpoint, side] > 10
                )
                runtime_eligible = bool(
                    data.loaded[run_index, endpoint, side]
                    and current_phase != 0
                ) if task == "terrain" else stable_contact
                if task == "terrain" and not (
                    runtime_eligible and data.pre_fall[run_index, endpoint]
                ):
                    continue
                feature = runtime_feature(
                    architecture, side, int(endpoint), data.bilateral[run_index],
                    data.loaded[run_index], data.age[run_index], data.phase[run_index], encoder,
                )
                features.append(feature)
                target = (
                    TERRAIN_LABELS[str(metadata["terrain_name"])]
                    if task == "terrain" else data.slip_target[run_index, endpoint, side]
                )
                values = {
                    "target": target, "run_index": run_index, "endpoint": int(endpoint),
                    "side": side, "phase": current_phase,
                    "speed": float(metadata["speed_mps"]), "role": metadata["role"],
                    "terrain": metadata["terrain_name"],
                    "runtime_eligible": runtime_eligible,
                    "pre_fall": data.pre_fall[run_index, endpoint],
                    "touchdown": data.touchdown_transient[run_index, endpoint, side],
                }
                for key, value in values.items():
                    fields[key].append(value)
    return FeatureRows(
        np.asarray(features, np.float32),
        np.asarray(fields["target"]), np.asarray(fields["run_index"], int),
        np.asarray(fields["endpoint"], int), np.asarray(fields["side"], int),
        np.asarray(fields["phase"], int), np.asarray(fields["speed"], float),
        np.asarray(fields["role"]), np.asarray(fields["terrain"]),
        np.asarray(fields["runtime_eligible"], bool), np.asarray(fields["pre_fall"], bool),
        np.asarray(fields["touchdown"], bool),
    )


def terrain_train_indices(rows: FeatureRows, seed: int) -> np.ndarray:
    speed_code = np.rint(rows.speed * 100).astype(int)
    groups = np.column_stack((rows.target, rows.phase, rows.side, speed_code))
    selected = balanced_indices(groups, seed)
    expected_groups = 4 * 4 * 2 * 3
    if len(np.unique(groups, axis=0)) != expected_groups:
        raise ValueError("Terrain class/phase/foot/speed balance groups incomplete")
    return selected


def slip_train_indices(rows: FeatureRows, seed: int) -> np.ndarray:
    eligible = rows.runtime_eligible & rows.pre_fall
    selected_rows = np.flatnonzero(eligible)
    speed_code = np.rint(rows.speed[selected_rows] * 100).astype(int)
    groups = np.column_stack((
        rows.target[selected_rows].astype(int), rows.side[selected_rows], speed_code,
    ))
    if len(np.unique(groups, axis=0)) != 2 * 2 * 3:
        raise ValueError("Slip class/foot/speed balance groups incomplete")
    return selected_rows[balanced_indices(groups, seed)]


def terrain_metrics(
    model: LinearFloatModel,
    rows: FeatureRows,
    architecture: str,
    seed: int,
    encoder: SharedCausalFootEncoder,
) -> dict[str, object]:
    prediction = model.predictions(rows.features).astype(int)
    target = rows.target.astype(int)
    recalls = recall_score(target, prediction, labels=np.arange(4), average=None, zero_division=0)
    balanced = balanced_indices(target[:, None], 202608299)
    balanced_counts = np.bincount(prediction[balanced], minlength=4)
    speed_accuracy = {
        f"{speed:.2f}": float(accuracy_score(target[rows.speed == speed], prediction[rows.speed == speed]))
        for speed in (0.10, 0.15, 0.20)
    }
    foot_accuracy = {
        side: float(accuracy_score(target[rows.side == index], prediction[rows.side == index]))
        for index, side in enumerate(("left", "right"))
    }
    metrics: dict[str, object] = {
        "architecture": architecture, "seed": seed,
        "validation_rows": len(target),
        "overall_accuracy": float(accuracy_score(target, prediction)),
        "macro_accuracy": float(np.mean(recalls)),
        "worst_class_recall": float(np.min(recalls)),
        "majority_class_prediction_rate": float(np.max(balanced_counts) / np.sum(balanced_counts)),
        "minimum_speed_accuracy": min(speed_accuracy.values()),
        "left_right_accuracy_difference_pp": abs(foot_accuracy["left"] - foot_accuracy["right"]) * 100.0,
        "class_collapse": bool(np.max(balanced_counts) / np.sum(balanced_counts) >= 0.60 or np.min(recalls) == 0.0),
        "air_terrain_transitions": 0,
        "invalid_firings": 0,
        "speed_accuracy": json.dumps(speed_accuracy, sort_keys=True),
        "foot_accuracy": json.dumps(foot_accuracy, sort_keys=True),
        "class_recall": json.dumps(dict(zip(TERRAIN_NAMES, recalls.tolist())), sort_keys=True),
        "confusion_matrix": json.dumps(confusion_matrix(target, prediction, labels=np.arange(4)).tolist()),
        "parameter_count": model.parameter_count + encoder.parameter_count,
        "macs": model.coefficients.size + 2 * encoder.macs,
        "history_bytes": 20 * TERRAIN_WINDOW_MS * 4,
        "persistent_state_bytes": 32,
        "shared_encoder_fingerprint": encoder.fingerprint,
        "reload_max_abs_error": 0.0,
    }
    metrics["gate_pass"] = terrain_gate(metrics)
    return metrics


def _reshape_slip_scores(rows: FeatureRows, data: DevelopmentData, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    endpoints = causal_endpoints(3000, SLIP_WINDOW_MS)
    expected = data.run_count * len(endpoints) * 2
    if len(score) != expected:
        raise ValueError("Slip rows lost endpoint/foot alignment")
    return endpoints, np.asarray(score, float).reshape(data.run_count, len(endpoints), 2)


def evaluate_slip(
    model: LinearFloatModel,
    rows: FeatureRows,
    data: DevelopmentData,
    config: SlipStateConfig,
    architecture: str,
    seed: int,
    encoder: SharedCausalFootEncoder,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    positive_score = model.probabilities(rows.features)[:, 1]
    endpoints, scores = _reshape_slip_scores(rows, data, positive_score)
    firing = np.zeros(scores.shape, bool)
    resets = np.full(scores.shape, "none", dtype="<U16")
    for run_index in range(data.run_count):
        for side in (0, 1):
            firing[run_index, :, side], resets[run_index, :, side] = stateful_slip_firing(
                scores[run_index, :, side], endpoints,
                data.loaded[run_index, :, side], data.age[run_index, :, side], config,
            )
    air_firing = touchdown_firing = post_fall_firing = 0
    normal_fp_runs = normal_contact_episode_fp = too_early = 0
    sink_profile_fp_runs = 0
    physical_total = physical_detected = actionable_total = actionable_detected = 0
    raw_crossings = raw_aligned = affected_total = affected_correct = 0
    warning_margins: list[float] = []
    pre_onset_detections = 0
    valid_ice_runs = detected_ice_runs = 0
    speed_run_total = {speed: 0 for speed in (0.10, 0.15, 0.20)}
    speed_run_detected = {speed: 0 for speed in (0.10, 0.15, 0.20)}
    foot_detected = {0: 0, 1: 0}
    episode_rows: list[dict[str, object]] = []
    reset_pass = True
    for run_index, metadata in enumerate(data.metadata):
        run_any_fire = bool(np.any(firing[run_index] & data.pre_fall[run_index, endpoints, None]))
        if metadata["role"] == "hard_negative" and run_any_fire:
            normal_fp_runs += 1
        if metadata["role"] == "sink_candidate" and run_any_fire:
            sink_profile_fp_runs += 1
        run_actionable = run_detected = False
        for side in (0, 1):
            current_fire = firing[run_index, :, side]
            endpoint_loaded = data.loaded[run_index, endpoints, side]
            endpoint_touchdown = data.touchdown_transient[run_index, endpoints, side]
            endpoint_prefall = data.pre_fall[run_index, endpoints]
            expected_contact_loss = ~endpoint_loaded
            expected_touchdown_reset = endpoint_loaded & (
                data.age[run_index, endpoints, side] <= 10
            )
            reset_pass &= bool(np.all(
                resets[run_index, expected_contact_loss, side] == "contact_loss"
            ))
            reset_pass &= bool(np.all(
                resets[run_index, expected_touchdown_reset, side] == "new_touchdown"
            ))
            air_firing += int(np.sum(current_fire & ~endpoint_loaded))
            touchdown_firing += int(np.sum(current_fire & endpoint_touchdown))
            post_fall_firing += int(np.sum(current_fire & ~endpoint_prefall))
            reset_pass &= not bool(np.any(current_fire & (data.age[run_index, endpoints, side] <= 10)))
            valid = data.pre_fall[run_index]
            episodes = physical_slip_episodes(
                data.slip_active[run_index, :, side],
                data.contact_episode[run_index, :, side], valid,
            )
            raw_events = raw_slip_crossings(
                data.slip_active[run_index, :, side],
                data.contact_episode[run_index, :, side], valid,
            )
            actionable = first_actionable_events(episodes)
            actionable_keys = {(value.start, value.end_exclusive) for value in actionable}
            first_contact_start = {
                contact_id: min(value.start for value in episodes if value.contact_episode_id == contact_id)
                for contact_id in {value.contact_episode_id for value in episodes}
            }
            if metadata["role"] == "hard_negative":
                for contact_id in np.unique(data.contact_episode[run_index, :, side]):
                    if contact_id < 0:
                        continue
                    contact_endpoint = data.contact_episode[run_index, endpoints, side] == contact_id
                    if np.any(current_fire & contact_endpoint & endpoint_prefall):
                        normal_contact_episode_fp += 1
            if metadata["role"] == "slip_candidate":
                for endpoint_index in np.flatnonzero(current_fire & endpoint_prefall):
                    sample = int(endpoints[endpoint_index])
                    contact_id = int(data.contact_episode[run_index, sample, side])
                    if risk_firing_is_too_early(sample, contact_id, episodes):
                        too_early += 1
            if metadata["role"] == "slip_candidate":
                raw_crossings += len(raw_events)
                for raw_event in raw_events:
                    raw_eligible = np.flatnonzero(
                        (endpoints >= max(0, raw_event.start - SLIP_WINDOW_MS))
                        & (endpoints < raw_event.end_exclusive)
                        & (data.contact_episode[run_index, endpoints, side]
                           == raw_event.contact_episode_id)
                        & endpoint_prefall
                    )
                    raw_aligned += int(bool(np.any(current_fire[raw_eligible])))
            for episode in episodes:
                risk_start = max(0, episode.start - SLIP_WINDOW_MS)
                eligible_indices = np.flatnonzero(
                    (endpoints >= risk_start) & (endpoints < episode.end_exclusive)
                    & (data.contact_episode[run_index, endpoints, side] == episode.contact_episode_id)
                    & endpoint_prefall
                )
                detected_indices = eligible_indices[current_fire[eligible_indices]]
                detected = bool(len(detected_indices))
                is_actionable = (episode.start, episode.end_exclusive) in actionable_keys
                first_detection = int(endpoints[detected_indices[0]]) if detected else None
                margin = None if first_detection is None else episode.start - first_detection
                if metadata["role"] == "slip_candidate":
                    physical_total += 1
                    physical_detected += int(detected)
                    if is_actionable:
                        actionable_total += 1
                        actionable_detected += int(detected)
                        run_actionable = True
                        run_detected |= detected
                    if margin is not None:
                        warning_margins.append(float(margin))
                        pre_onset_detections += int(0 < margin <= SLIP_WINDOW_MS)
                        affected_total += 1
                        affected_correct += int(affected_foot_correct(
                            side, int(detected_indices[0]),
                            scores[run_index, :, 0], scores[run_index, :, 1],
                            firing[run_index, :, 0], firing[run_index, :, 1],
                        ))
                        foot_detected[side] += 1
                episode_rows.append({
                    "architecture": architecture, "seed": seed,
                    "threshold": config.threshold,
                    "persistence_endpoints": config.persistence_endpoints,
                    "hysteresis": config.hysteresis,
                    "run_id": metadata["run_id"], "role": metadata["role"],
                    "speed_mps": metadata["speed_mps"], "foot": ("left", "right")[side],
                    "contact_episode_id": episode.contact_episode_id,
                    "physical_start_sample": episode.start,
                    "physical_end_exclusive": episode.end_exclusive,
                    "raw_crossings": episode.raw_crossings,
                    "first_actionable": is_actionable,
                    "post_reflex_counterfactual": bool(
                        episode.start > first_contact_start[episode.contact_episode_id]
                    ),
                    "detected": detected,
                    "first_detection_sample": "" if first_detection is None else first_detection,
                    "warning_margin_ms": "" if margin is None else margin,
                    "affected_foot_correct": "" if margin is None else affected_foot_correct(
                        side, int(detected_indices[0]),
                        scores[run_index, :, 0], scores[run_index, :, 1],
                        firing[run_index, :, 0], firing[run_index, :, 1],
                    ),
                })
        if metadata["role"] == "slip_candidate" and run_actionable:
            valid_ice_runs += 1
            detected_ice_runs += int(run_detected)
            speed = float(metadata["speed_mps"])
            speed_run_total[speed] += 1
            speed_run_detected[speed] += int(run_detected)
    invalid = air_firing + touchdown_firing + post_fall_firing
    metrics: dict[str, object] = {
        "architecture": architecture, "seed": seed,
        "threshold": config.threshold,
        "persistence_endpoints": config.persistence_endpoints,
        "persistence_ms": config.persistence_endpoints * ENDPOINT_STRIDE_MS,
        "hysteresis": config.hysteresis,
        "valid_ice_runs": valid_ice_runs, "detected_ice_runs": detected_ice_runs,
        "valid_ice_run_coverage": detected_ice_runs / valid_ice_runs if valid_ice_runs else 0.0,
        "physical_episode_count": physical_total,
        "physical_episode_detected": physical_detected,
        "physical_episode_recall": physical_detected / physical_total if physical_total else 0.0,
        "first_actionable_event_count": actionable_total,
        "first_actionable_event_detected": actionable_detected,
        "first_actionable_event_recall": actionable_detected / actionable_total if actionable_total else 0.0,
        "raw_crossing_count": raw_crossings,
        "raw_crossing_alignment": raw_aligned / raw_crossings if raw_crossings else 0.0,
        "affected_foot_count": affected_total,
        "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
        "normal_risk_run_fp": normal_fp_runs,
        "normal_physical_episode_fp": normal_contact_episode_fp,
        "sink_profile_run_firing": sink_profile_fp_runs,
        "too_early_firings": too_early,
        "air_firings": air_firing,
        "touchdown_firings": touchdown_firing,
        "post_fall_firings": post_fall_firing,
        "invalid_firings": invalid,
        "median_warning_margin_ms": float(np.median(warning_margins)) if warning_margins else -1.0,
        "pre_onset_detection_fraction": pre_onset_detections / len(warning_margins) if warning_margins else 0.0,
        "speed_run_coverage": json.dumps({
            f"{speed:.2f}": (
                speed_run_detected[speed] / speed_run_total[speed]
                if speed_run_total[speed] else 0.0
            ) for speed in speed_run_total
        }, sort_keys=True),
        "all_speed_coverage": all(
            speed_run_total[value] > 0 and speed_run_detected[value] == speed_run_total[value]
            for value in speed_run_total
        ),
        "foot_detected_events": json.dumps({"left": foot_detected[0], "right": foot_detected[1]}),
        "both_affected_feet_coverage": foot_detected[0] > 0 and foot_detected[1] > 0,
        "reset_invariant_pass": bool(reset_pass),
        "parameter_count": model.parameter_count + encoder.parameter_count,
        "macs": model.coefficients.size + 2 * encoder.macs,
        "history_bytes": 20 * SLIP_WINDOW_MS * 4,
        "persistent_state_bytes": 64,
        "shared_encoder_fingerprint": encoder.fingerprint,
        "reload_max_abs_error": 0.0,
    }
    metrics["gate_pass"] = slip_gate(metrics)
    return metrics, episode_rows


def protocol(upstream: dict[str, str]) -> dict[str, object]:
    return {
        "artifact": "walking_v2_bilateral_bounded_training",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "purpose": "bounded walking-specific Terrain and 100 ms Slip-risk float candidates",
        "upstream_sha256": upstream,
        "scope_sha256_pretraining": {
            "runtime_scope": _sha256_json(runtime_scope_contract()),
            "input_contract": _sha256_json(input_contract()),
            "sink_deferral": _sha256_json(sink_deferral_contract()),
        },
        "development": {"train_runs": 72, "validation_runs": 48, "holdout_runs": 0},
        "windows_ms": {"terrain": TERRAIN_WINDOW_MS, "slip": SLIP_WINDOW_MS},
        "endpoint_stride_ms": ENDPOINT_STRIDE_MS,
        "architectures": {
            "terrain": list(TERRAIN_ARCHITECTURES), "slip": list(SLIP_ARCHITECTURES),
            "shared_encoder_trainable": False,
            "shared_encoder_identity": "one exact fixed encoder object per window/task",
        },
        "seeds": list(TRAINING_SEEDS),
        "slip_grid": {
            "threshold": list(SLIP_THRESHOLD_GRID),
            "persistence_endpoints": list(SLIP_PERSISTENCE_GRID),
            "hysteresis": list(SLIP_HYSTERESIS_GRID),
            "expanded_after_validation": False,
        },
        "episode_semantics": episode_semantics_contract(),
        "terrain_evaluation_balance": "raw overall plus class-balanced collapse distribution",
        "normalization": "fit only on balanced development-train rows for each seed/architecture",
        "validation_use": "selection and fixed gate only; never model fitting",
        "holdout_authorization": "both development gates plus written selection lock",
        "development_gates": {
            "terrain": {
                "overall_accuracy_min": 0.85, "macro_accuracy_min": 0.85,
                "worst_class_recall_min": 0.70,
                "majority_class_prediction_rate_strict_max": 0.60,
                "each_speed_accuracy_min": 0.80,
                "left_right_difference_pp_max": 10.0,
                "class_collapse": False, "air_transitions": 0,
            },
            "slip": {
                "valid_ice_run_coverage": 1.0,
                "first_actionable_event_recall": 1.0,
                "physical_episode_recall_min": 0.80,
                "affected_foot_accuracy_min": 0.90,
                "normal_run_fp": 0, "normal_contact_episode_fp": 0,
                "too_early_firings": 0, "invalid_firings": 0,
                "all_speed_and_both_foot_coverage": True,
                "median_warning_margin_ms_min": 20.0,
                "pre_onset_detection_fraction_min": 0.80,
                "reset_invariant": True,
            },
        },
        "conditional_holdout": {
            "namespace": "walking_v2_bilateral_blind_holdout",
            "new_variations": ["phase_delay_a", "phase_delay_b", "phase_delay_c"],
            "speeds_mps": [0.10, 0.15, 0.20],
            "normal_terrains": ["marble", "concrete", "hardened_sand"],
            "slip_terrain": "ice", "run_count": 36,
            "duration_s": 3.0, "native_rate_hz": 1000,
            "seed_namespace": "walking_v2_blind_20260901",
            "candidate_selection_independent": True,
            "existing_outer_or_holdout_reuse": False,
            "evaluate_exactly_once": True,
            "post_holdout_tuning_allowed": False,
            "gates": {
                "terrain": {
                    "overall_accuracy_min": 0.85, "macro_accuracy_min": 0.85,
                    "worst_class_recall_min": 0.70,
                    "majority_prediction_rate_strict_max": 0.60,
                    "each_terrain_coverage": True, "each_speed_accuracy_min": 0.80,
                    "each_variation_accuracy_min": 0.75,
                    "left_right_difference_pp_max": 10.0,
                    "air_transitions": 0, "class_collapse": False,
                },
                "slip": {
                    "ice_run_detection": "9/9", "each_speed_detection": "3/3",
                    "each_variation_detection": "3/3",
                    "first_actionable_event_recall": 1.0,
                    "physical_episode_recall_min": 0.80,
                    "affected_foot_accuracy_min": 0.90,
                    "normal_run_fp": "0/27", "too_early_firings": 0,
                    "invalid_firings": 0, "pre_fall_mask_violations": 0,
                    "pre_onset_detection_fraction_min": 0.80,
                    "median_warning_margin_ms_min": 20.0,
                    "reset_invariant": True,
                },
            },
        },
        "forbidden_content_namespaces": list(FORBIDDEN_CONTENT_NAMESPACES),
        "production_or_system_or_int8_change": False,
    }


def _write_model_bundle(
    output: Path,
    task: str,
    architecture: str,
    seed: int,
    model: LinearFloatModel,
    train_indices: np.ndarray,
    config_extra: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    stem = f"{task.lower()}_{architecture.lower()}_seed_{seed}"
    model_path = output / "models" / f"{stem}.npz"
    norm_path = output / "normalization" / f"{stem}.json"
    config_path = output / "configs" / f"{stem}.json"
    model.save(model_path)
    _json(norm_path, {
        "mean": model.mean.tolist(), "scale": model.scale.tolist(),
        "fit_split": "development_train", "balanced_train_rows": len(train_indices),
        "validation_rows_used": 0,
    })
    _json(config_path, {
        "task": task, "architecture": architecture, "seed": seed,
        "encoder_fingerprint": model.encoder_fingerprint,
        "runtime_window_ms": TERRAIN_WINDOW_MS if task == "terrain" else SLIP_WINDOW_MS,
        "future_samples": 0, "sink_head": False,
        **({} if config_extra is None else config_extra),
    })
    return model_path, norm_path, config_path


def resource_row(
    task: str,
    architecture: str,
    seed: int,
    model_path: Path,
    norm_path: Path,
    config_path: Path,
    model: LinearFloatModel,
    encoder: SharedCausalFootEncoder,
) -> dict[str, object]:
    history = 20 * encoder.window_ms * 4
    persistent = 32 if task == "terrain" else 64
    normalization_bytes = model.mean.nbytes + model.scale.nbytes
    macs = model.coefficients.size + 2 * encoder.macs
    return {
        "task": task, "architecture": architecture, "seed": seed,
        "model_parameter_count": model.parameter_count + encoder.parameter_count,
        "model_file_bytes": model_path.stat().st_size,
        "shared_encoder_structure": "one exact depthwise Conv/GAP weight set; applied twice",
        "shared_encoder_fingerprint": encoder.fingerprint,
        "head_macs_per_inference": model.coefficients.size,
        "bilateral_macs_per_tick": macs,
        "history_buffer_bytes": history,
        "persistent_state_bytes": persistent,
        "normalization_bytes": normalization_bytes,
        "estimated_host_inference_ms": 0.20 + macs / 1_000_000.0,
        "expected_int8_operator_set": "DEPTHWISE_CONV_1D-equivalent,MEAN,SUB,FULLY_CONNECTED,SOFTMAX|LOGISTIC",
        "expected_vela_compatibility": True,
        "unsupported_recurrent_operator": False,
        "within_parameter_limit": model.parameter_count + encoder.parameter_count <= 5000,
        "within_history_state_limit": history + persistent <= 32 * 1024,
        "within_mac_limit": macs <= 250_000,
        "model_sha256": sha256_file(model_path),
        "normalization_sha256": sha256_file(norm_path),
        "config_sha256": sha256_file(config_path),
        "int8_or_vela_executed": False,
    }


def write_audit(output: Path, summary: dict[str, object]) -> None:
    text = f"""# Walking-v2 Bilateral Bounded Training

## Scope

Development-only float training used the fresh 72/48 bilateral train/validation
split. Existing outer, holdout, spatial, final-test, production, System, INT8,
Vela, E84, and physical-hardware content was not opened or modified. Sink has
no runtime head; Sand means `SAND_TERRAIN_CAUTION` only.

## Decision

- Terrain READY: {summary['terrain_candidate_ready']}
- Slip READY: {summary['slip_candidate_ready']}
- Selection lock: {summary['selection_lock_ready']}
- New blind holdout runs: {summary['holdout_runs']}
- Diagnostic Terrain fallback: {summary['diagnostic_terrain_fallback']}
- Diagnostic Slip fallback: {summary['diagnostic_slip_fallback']}
- Next step: `{summary['next_step']}`

No validation gate, threshold grid, persistence grid, feature set, or episode
rule was changed after measurement. Diagnostic fallback is not a production
candidate and does not replace the controlled/static detector.
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    upstream_before = upstream_hashes()
    declared_protocol = protocol(upstream_before)
    if not args.execute:
        print(json.dumps(declared_protocol, indent=2))
        return {"planned": True}
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    if args.holdout_output_dir.exists() and any(args.holdout_output_dir.iterdir()):
        raise FileExistsError("conditional holdout output must be absent or empty before training")
    for directory in (output, output / "models", output / "normalization", output / "configs"):
        directory.mkdir(parents=True, exist_ok=True)
    # Scope artifacts are written and hashed before any model fitting.
    _json(output / "walking_v2_runtime_scope.json", runtime_scope_contract())
    _json(output / "walking_v2_input_contract.json", input_contract())
    _json(output / "sink_runtime_deferral.json", sink_deferral_contract())
    _json(output / "episode_semantics.json", episode_semantics_contract())
    _json(output / "protocol.json", declared_protocol)
    scope_hashes = {
        name: sha256_file(output / name) for name in (
            "walking_v2_runtime_scope.json", "walking_v2_input_contract.json",
            "sink_runtime_deferral.json", "episode_semantics.json", "protocol.json",
        )
    }
    started = time.perf_counter()
    train = load_development("train")
    validation = load_development("validation")
    all_metadata = (*train.metadata, *validation.metadata)
    endpoint_hashes = [str(value["endpoint_sha256"]) for value in all_metadata]
    duplicate_endpoint_hash_count = len(endpoint_hashes) - len(set(endpoint_hashes))
    schema_ready = bool(
        train.time_s.shape == (72, 3000)
        and validation.time_s.shape == (48, 3000)
        and np.allclose(np.diff(train.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)
        and np.allclose(np.diff(validation.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)
    )
    if (
        len(set(train.run_id) & set(validation.run_id))
        or len(set(train.run_id)) != 72
        or len(set(validation.run_id)) != 48
        or duplicate_endpoint_hash_count != 0
        or not schema_ready
    ):
        raise ValueError("development run ownership leakage or duplicate")
    terrain_training: list[dict[str, object]] = []
    terrain_validation: list[dict[str, object]] = []
    slip_training: list[dict[str, object]] = []
    slip_validation: list[dict[str, object]] = []
    slip_episode_rows_by_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    resource_rows: list[dict[str, object]] = []
    model_paths: dict[tuple[str, str, int], tuple[Path, Path, Path]] = {}
    for architecture in TERRAIN_ARCHITECTURES:
        encoder = SharedCausalFootEncoder(TERRAIN_WINDOW_MS)
        train_rows = build_feature_rows(train, "terrain", architecture)
        validation_rows = build_feature_rows(validation, "terrain", architecture)
        for seed in TRAINING_SEEDS:
            selected = terrain_train_indices(train_rows, seed)
            model = fit_linear_float(
                architecture, seed, train_rows.features[selected], train_rows.target[selected],
                encoder.fingerprint,
            )
            paths = _write_model_bundle(output, "terrain", architecture, seed, model, selected)
            reload_model = LinearFloatModel.load(paths[0])
            parity = float(np.max(np.abs(
                model.probabilities(validation_rows.features[:1000])
                - reload_model.probabilities(validation_rows.features[:1000])
            )))
            metrics = terrain_metrics(model, validation_rows, architecture, seed, encoder)
            metrics["reload_max_abs_error"] = parity
            metrics["gate_pass"] = bool(metrics["gate_pass"] and parity == 0.0)
            terrain_validation.append(metrics)
            terrain_training.append({
                "architecture": architecture, "seed": seed,
                "balanced_train_rows": len(selected),
                "feature_count": train_rows.features.shape[1],
                "class_phase_foot_speed_group_count": 96,
                "train_only_normalization": True,
                "validation_rows_used_for_fit": 0,
                "model_path": paths[0].relative_to(output).as_posix(),
                "normalization_path": paths[1].relative_to(output).as_posix(),
                "config_path": paths[2].relative_to(output).as_posix(),
            })
            resource_rows.append(resource_row(
                "terrain", architecture, seed, *paths, model, encoder
            ))
            model_paths[("terrain", architecture, seed)] = paths
            print(f"Terrain {architecture} seed={seed} gate={metrics['gate_pass']} macro={metrics['macro_accuracy']:.3f}", flush=True)
    for architecture in SLIP_ARCHITECTURES:
        encoder = SharedCausalFootEncoder(SLIP_WINDOW_MS)
        train_rows = build_feature_rows(train, "slip", architecture)
        validation_rows = build_feature_rows(validation, "slip", architecture)
        for seed in TRAINING_SEEDS:
            selected = slip_train_indices(train_rows, seed)
            model = fit_linear_float(
                architecture, seed, train_rows.features[selected], train_rows.target[selected],
                encoder.fingerprint,
            )
            paths = _write_model_bundle(output, "slip", architecture, seed, model, selected, {
                "threshold_grid": list(SLIP_THRESHOLD_GRID),
                "persistence_grid": list(SLIP_PERSISTENCE_GRID),
                "hysteresis_grid": list(SLIP_HYSTERESIS_GRID),
            })
            reload_model = LinearFloatModel.load(paths[0])
            parity = float(np.max(np.abs(
                model.probabilities(validation_rows.features[:1000])
                - reload_model.probabilities(validation_rows.features[:1000])
            )))
            best_for_model: dict[str, object] | None = None
            best_episode_rows: list[dict[str, object]] = []
            for threshold in SLIP_THRESHOLD_GRID:
                for persistence in SLIP_PERSISTENCE_GRID:
                    for hysteresis in SLIP_HYSTERESIS_GRID:
                        state = SlipStateConfig(threshold, persistence, hysteresis)
                        metrics, episode_rows = evaluate_slip(
                            model, validation_rows, validation, state, architecture, seed, encoder
                        )
                        metrics["reload_max_abs_error"] = parity
                        metrics["gate_pass"] = bool(metrics["gate_pass"] and parity == 0.0)
                        slip_validation.append(metrics)
                        if best_for_model is None or deterministic_slip_selection([
                            best_for_model, metrics
                        ]) is metrics:
                            best_for_model = metrics
                            best_episode_rows = episode_rows
            if best_for_model is None:
                raise AssertionError("Slip grid produced no metrics")
            key = (architecture, seed, best_for_model["threshold"], best_for_model["persistence_endpoints"], best_for_model["hysteresis"])
            slip_episode_rows_by_key[key] = best_episode_rows
            slip_training.append({
                "architecture": architecture, "seed": seed,
                "balanced_train_rows": len(selected),
                "feature_count": train_rows.features.shape[1],
                "class_foot_speed_group_count": 12,
                "train_only_normalization": True,
                "validation_rows_used_for_fit": 0,
                "grid_candidate_count": len(SLIP_THRESHOLD_GRID) * len(SLIP_PERSISTENCE_GRID) * len(SLIP_HYSTERESIS_GRID),
                "model_path": paths[0].relative_to(output).as_posix(),
                "normalization_path": paths[1].relative_to(output).as_posix(),
                "config_path": paths[2].relative_to(output).as_posix(),
            })
            resource_rows.append(resource_row(
                "slip", architecture, seed, *paths, model, encoder
            ))
            model_paths[("slip", architecture, seed)] = paths
            print(
                f"Slip {architecture} seed={seed} best_gate={best_for_model['gate_pass']} "
                f"run={best_for_model['valid_ice_run_coverage']:.3f} "
                f"event={best_for_model['first_actionable_event_recall']:.3f}",
                flush=True,
            )
    terrain_passing = [row for row in terrain_validation if row["gate_pass"]]
    slip_passing = [row for row in slip_validation if row["gate_pass"]]
    terrain_best = deterministic_terrain_selection(terrain_passing or terrain_validation)
    slip_best = deterministic_slip_selection(slip_passing or slip_validation)
    terrain_ready = bool(terrain_passing)
    slip_ready = bool(slip_passing)
    terrain_key = ("terrain", str(terrain_best["architecture"]), int(terrain_best["seed"]))
    slip_key = ("slip", str(slip_best["architecture"]), int(slip_best["seed"]))

    def bundle(key: tuple[str, str, int]) -> dict[str, object]:
        model_path, normalization_path, config_path = model_paths[key]
        return {
            "model_path": model_path.relative_to(output).as_posix(),
            "model_sha256": sha256_file(model_path),
            "normalization_path": normalization_path.relative_to(output).as_posix(),
            "normalization_sha256": sha256_file(normalization_path),
            "training_config_path": config_path.relative_to(output).as_posix(),
            "training_config_sha256": sha256_file(config_path),
        }

    terrain_bundle = bundle(terrain_key)
    slip_bundle = bundle(slip_key)
    candidate_selection = {
        "terrain": {
            "ready": terrain_ready, "selected_or_diagnostic": terrain_best,
            "passing_candidate_count": len(terrain_passing),
            "artifact_bundle": terrain_bundle,
        },
        "slip": {
            "ready": slip_ready, "selected_or_diagnostic": slip_best,
            "passing_candidate_count": len(slip_passing),
            "artifact_bundle": slip_bundle,
        },
        "diagnostic_fallback_is_production": False,
        "validation_gate_changed": False,
    }
    _json(output / "candidate_selection.json", candidate_selection)
    selection_lock_ready = False
    selection_lock: dict[str, object] | None = None
    if terrain_ready and slip_ready:
        selected_directory = output / "selected"
        selected_directory.mkdir(parents=True, exist_ok=True)
        terrain_runtime_config = selected_directory / "terrain_runtime_config.json"
        slip_runtime_config = selected_directory / "slip_runtime_config.json"
        _json(terrain_runtime_config, {
            "architecture": terrain_best["architecture"], "seed": terrain_best["seed"],
            "window_ms": TERRAIN_WINDOW_MS, "endpoint_stride_ms": ENDPOINT_STRIDE_MS,
            "contact_gate": "loaded pre-fall non-AIR", "sand_semantics": "SAND_TERRAIN_CAUTION",
            "sink_head": False,
        })
        _json(slip_runtime_config, {
            "architecture": slip_best["architecture"], "seed": slip_best["seed"],
            "window_ms": SLIP_WINDOW_MS, "endpoint_stride_ms": ENDPOINT_STRIDE_MS,
            "threshold": slip_best["threshold"],
            "persistence_endpoints": slip_best["persistence_endpoints"],
            "hysteresis": slip_best["hysteresis"],
            "contact_gate": "loaded and contact_age > 10 ms",
            "sink_head": False,
        })
        selection_lock = {
            "created_before_holdout": True,
            "training_process_complete": True,
            "scope_sha256": scope_hashes,
            "development_data_sha256": {
                "train": upstream_before[ALLOWED_UPSTREAM_FILES[2]],
                "validation": upstream_before[ALLOWED_UPSTREAM_FILES[3]],
            },
            "terrain": {
                "metrics": terrain_best,
                **terrain_bundle,
                "runtime_config_path": terrain_runtime_config.relative_to(output).as_posix(),
                "runtime_config_sha256": sha256_file(terrain_runtime_config),
            },
            "slip": {
                "metrics": slip_best,
                **slip_bundle,
                "runtime_config_path": slip_runtime_config.relative_to(output).as_posix(),
                "runtime_config_sha256": sha256_file(slip_runtime_config),
            },
            "post_holdout_tuning_allowed": False,
        }
        _json(output / "selection_lock.json", selection_lock)
        selection_lock_ready = True
    else:
        _json(output / "non_selection.json", {
            "reason": "one or both mandatory development gates failed",
            "terrain_ready": terrain_ready, "slip_ready": slip_ready,
            "selection_lock_created": False, "holdout_authorized": False,
            "diagnostic_terrain_fallback": terrain_best,
            "diagnostic_slip_fallback": slip_best,
        })
    authorized = holdout_authorized(terrain_ready, slip_ready, selection_lock_ready)
    holdout_runs = 0
    terrain_holdout_ready = False
    slip_holdout_ready = False
    if authorized:
        raise RuntimeError(
            "development gates unexpectedly authorized holdout; stop before access so the "
            "one-shot holdout runner can be materialized and audited"
        )
    upstream_after = upstream_hashes()
    if upstream_before != upstream_after:
        raise RuntimeError("immutable upstream SHA changed during training")
    readiness = {
        "WALKING_V2_SCOPE_FREEZE_READY": True,
        "WALKING_V2_BILATERAL_DATA_READY": (
            train.run_count == 72 and validation.run_count == 48 and schema_ready
        ),
        "WALKING_V2_SPLIT_INTEGRITY_READY": duplicate_endpoint_hash_count == 0,
        "WALKING_V2_TERRAIN_FLOAT_CANDIDATE_READY": terrain_ready,
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": slip_ready,
        "WALKING_V2_SELECTION_LOCK_READY": selection_lock_ready,
        "WALKING_V2_HOLDOUT_NON_ACCESS_READY": not authorized and holdout_runs == 0,
        "WALKING_V2_TERRAIN_BLIND_HOLDOUT_READY": terrain_holdout_ready,
        "WALKING_V2_SLIP_BLIND_HOLDOUT_READY": slip_holdout_ready,
        "WALKING_V2_RUNTIME_SCOPE_READY": terrain_holdout_ready and slip_holdout_ready,
        "WALKING_V2_SINK_RUNTIME_DEFERRED": True,
        "WALKING_SYSTEM_V2_MIGRATION_AUTHORIZED": terrain_holdout_ready and slip_holdout_ready,
        "WALKING_INT8_PREPARATION_AUTHORIZED": terrain_holdout_ready and slip_holdout_ready,
    }
    if not terrain_ready:
        next_step = "TERRAIN_CANDIDATE_REDESIGN"
    elif not slip_ready:
        next_step = "SLIP_CANDIDATE_REDESIGN"
    elif not terrain_holdout_ready:
        next_step = "TERRAIN_CANDIDATE_REDESIGN"
    elif not slip_holdout_ready:
        next_step = "SLIP_CANDIDATE_REDESIGN"
    else:
        next_step = "SYSTEM_V2_AND_INT8_PREPARATION"
    selected_slip_key = (
        str(slip_best["architecture"]), int(slip_best["seed"]),
        slip_best["threshold"], slip_best["persistence_endpoints"], slip_best["hysteresis"],
    )
    selected_episode_rows = slip_episode_rows_by_key.get(selected_slip_key, [])
    summary: dict[str, object] = {
        "artifact": "walking_v2_bilateral_bounded_training",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "train_runs": train.run_count, "validation_runs": validation.run_count,
        "holdout_runs": holdout_runs,
        "native_rate_hz": 1000,
        "sample_spacing_max_error_s": max(
            float(np.max(np.abs(np.diff(train.time_s, axis=1) - 0.001))),
            float(np.max(np.abs(np.diff(validation.time_s, axis=1) - 0.001))),
        ),
        "duplicate_endpoint_hash_count": duplicate_endpoint_hash_count,
        "terrain_candidate_ready": terrain_ready,
        "slip_candidate_ready": slip_ready,
        "selection_lock_ready": selection_lock_ready,
        "holdout_authorized": authorized,
        "terrain_blind_holdout_ready": terrain_holdout_ready,
        "slip_blind_holdout_ready": slip_holdout_ready,
        "diagnostic_terrain_fallback": {
            key: terrain_best[key] for key in (
                "architecture", "seed", "overall_accuracy", "macro_accuracy",
                "worst_class_recall", "majority_class_prediction_rate", "gate_pass",
            )
        },
        "diagnostic_slip_fallback": {
            key: slip_best[key] for key in (
                "architecture", "seed", "threshold", "persistence_ms", "hysteresis",
                "valid_ice_run_coverage", "physical_episode_recall",
                "first_actionable_event_recall", "affected_foot_accuracy",
                "normal_risk_run_fp", "too_early_firings", "invalid_firings",
                "median_warning_margin_ms", "pre_onset_detection_fraction", "gate_pass",
            )
        },
        "diagnostic_terrain_artifact_bundle": terrain_bundle,
        "diagnostic_slip_artifact_bundle": slip_bundle,
        "scope_hashes_pretraining": scope_hashes,
        "immutable_upstream_mismatch_count": 0,
        "outer_holdout_spatial_final_content_load_count": 0,
        "sink_runtime_head_count": 0,
        "sand_runtime_semantics": "SAND_TERRAIN_CAUTION",
        "controlled_static_replacement_claimed": False,
        "production_system_int8_files_changed": False,
        "readiness": readiness,
        "next_step": next_step,
        "wall_time_s": time.perf_counter() - started,
    }
    _csv(output / "terrain_training_matrix.csv", terrain_training)
    _csv(output / "slip_training_matrix.csv", slip_training)
    _csv(output / "terrain_validation_metrics.csv", terrain_validation)
    _csv(output / "slip_validation_metrics.csv", slip_validation)
    _csv(output / "slip_episode_metrics.csv", selected_episode_rows)
    _csv(output / "resource_estimate.csv", resource_rows)
    _csv(output / "immutable_sha_audit.csv", [
        {"path": path, "sha256_before": value, "sha256_after": upstream_after[path], "match": value == upstream_after[path]}
        for path, value in upstream_before.items()
    ])
    _csv(output / "legacy_scope_mapping.csv", [
        {"legacy_scope": "left-foot Slip blind authorization", "walking_v2_scope": "not transferred", "status": "v1-only"},
        {"legacy_scope": "controlled/static detector", "walking_v2_scope": "diagnostic regression only", "status": "unchanged"},
        {"legacy_scope": "Sink runtime", "walking_v2_scope": "no runtime head", "status": "deferred"},
        {"legacy_scope": "Sand class", "walking_v2_scope": "SAND_TERRAIN_CAUTION", "status": "not Sink"},
    ])
    _csv(output / "controlled_static_compatibility.csv", [
        {"detector": "frozen Terrain", "sha256": upstream_before[ALLOWED_UPSTREAM_FILES[10]], "walking_v2_replacement": False, "replay_executed": False, "status": "immutable"},
        {"detector": "frozen Slip INT8", "sha256": upstream_before[ALLOWED_UPSTREAM_FILES[12]], "walking_v2_replacement": False, "replay_executed": False, "status": "immutable"},
        {"detector": "frozen Sink INT8", "sha256": upstream_before[ALLOWED_UPSTREAM_FILES[13]], "walking_v2_replacement": False, "replay_executed": False, "status": "immutable"},
    ])
    _json(output / "outer_non_access.json", {
        "content_load_count": 0, "existing_outer_reused": False,
        "new_holdout_created": False, "holdout_authorized": authorized,
        "forbidden_namespaces": list(FORBIDDEN_CONTENT_NAMESPACES),
    })
    _json(output / "readiness.json", readiness)
    _json(output / "summary.json", summary)
    write_audit(output, summary)
    generated = []
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "manifest.json"):
        generated.append({
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        })
    _json(output / "manifest.json", {
        "artifact": "walking_v2_bilateral_bounded_training",
        "generated_files": generated, "hash_graph_complete": True,
        "manifest_self_hash_excluded": True,
        "development_run_ids": {"train": train.run_id.tolist(), "validation": validation.run_id.tolist()},
        "selection_lock_created": selection_lock_ready,
        "holdout_content_load_count": 0,
        "immutable_upstream_sha256": upstream_before,
    })
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
