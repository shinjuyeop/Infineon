"""Build and validate the bounded walking stateful hazard prototype v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from run_walking_fusion10_observability_audit_v1 import (
    ROOT,
    load_controlled,
    load_walking,
)
from walking_bounded_retraining_v1 import physical_oracle
from walking_fusion10_observability_audit_v1 import (
    LinearProbe,
    fit_logistic_probe,
    phase_balanced_indices,
    split_integrity,
)
from walking_stateful_hazard_prototype_v1 import (
    CONTACT_OFF_N,
    CONTACT_ON_N,
    CausalFeatureState,
    ContactPhaseTracker,
    DetectorState,
    MachineConfig,
    StatefulHazardMachine,
    estimate_resources,
    select_candidate,
)


SIM = ROOT / "simulation"
OUTPUT = SIM / "outputs" / "walking_stateful_hazard_prototype_v1"
WALKING = SIM / "outputs" / "walking_hazard_oracle_calibration_v1"
CONTROLLED = SIM / "outputs" / "terrain_fast_reflex_v2_final_scope_full"
STARTING_CHECKPOINT = "817474b2bf46f3368cfee8d5ce62abbbd1dc20d3"
DEVELOPMENT_CHECKPOINT = "d2209cdb49c496839a16396f201ab0322515171b"
SEED = 20260820
RISK_HORIZON = {"slip": 100, "sink": 200}
CONTROLLED_BASELINE = {"slip": 0.8235294117647058, "sink": 1.0}
UPSTREAM_SHA256 = {
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/protocol.json": "04f8302cd8b8232de0bec2ef742287a9fd4ad4422ac8eb8a6cad9d49d76aac88",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json": "96a4c29ce495aea1e7785ae95124652bf1017ac3d9618727d4ef17c9bc37aa10",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/traces.npz": "cba2cf5f4ce915135a8607ee384e5b9b04d76a4ab2bbc0e3608b27e0b8d946d7",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/protocol.json": "3e8f2c70a81b3e686f1967818981cada408ec2549abe31d5d57b53d919c224a3",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/manifest.json": "b1360af75e6d57d59270df1bde3939b41d70a537067ae9754c044bda7a041aa8",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/traces.npz": "1bbede7770400a844f75da2bce5157e4b50e5f8119ea5e893760943c7ed40423",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json": "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/protocol.json": "6d61335d2c861981f87b51fecb54061b7ad3b8157fd09687ca62e9fdd06aac92",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/nested_selection.json": "72330c16397dc9aa2f416beb933518e22ce20f307f87e833c0399c198928896f",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_manifest.json": "e04db778a49575ad7afbc6ee7c5f35a5b85a5ce0d7a7ef9664f4933020a138fd",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz": "4b7876eae0e7caa9411bcd106f01eedfeee6e342cde5e551e56ea52f5d6de1d3",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/summary.json": "7a476799d72c40081d64a531f6ddd7e2591eb18d93d269fa74114bfa4f24970b",
    "simulation/outputs/walking_bounded_retraining_v1/protocol.json": "f0bd7c3120ef61ccb72b903f27a8553ec951feec8b36c000e52b515f3cef68eb",
    "simulation/outputs/walking_bounded_retraining_v1/selection_lock.json": "cf956ea105fff2335b48a503327d34f82f1c1d9d39efef4e6930728ff6e59162",
    "simulation/outputs/walking_bounded_retraining_v1/summary.json": "ccc55ae83237cd1febfd5ca99b2f40bd13202b83f1d8c02d766896e54274d8d2",
    "simulation/outputs/walking_bounded_retraining_v1/source_hashes.json": "e1353d47fef2802f5d40f0155da5aa7bffc35b1bf964a744cab5434830356740",
    "simulation/outputs/walking_fusion10_observability_audit_v1/protocol.json": "eff9bc7df4a4e20169793651a020b9e27a4e7c60c7cd1dc92772c5abc58eb29f",
    "simulation/outputs/walking_fusion10_observability_audit_v1/manifest.json": "3be0931dd918b6ba62a9f2c7e370e5469d5f10e03053ef0b4069d436e64348ef",
    "simulation/outputs/walking_fusion10_observability_audit_v1/summary.json": "9aa84e85af7692bd81bef28ad0c34c65163a74473934268fa94b05bad6b3fc17",
    "simulation/outputs/walking_fusion10_observability_audit_v1/readiness.json": "2f823302884045a5a8518d4c2bf960a2bd5f4cf7ede082cb05c650eb3f2badfc",
}
HASH_ONLY_PATHS = {
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_manifest.json",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz",
}


def candidate_definitions() -> dict[str, list[dict[str, object]]]:
    return {
        "slip": [
            {"candidate_id": "A_stateless_100ms", "history_ms": 100, "feature": "raw", "risk": False, "machine": False, "threshold_rule": "fixed_0p90", "risk_persistence": 1, "confirmed_persistence": 1, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "B_dual_persistence_100ms", "history_ms": 100, "feature": "raw", "risk": True, "machine": False, "threshold_rule": "fixed_0p90", "risk_persistence": 3, "confirmed_persistence": 5, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "C_contact_gated_100ms", "history_ms": 100, "feature": "raw", "risk": True, "machine": True, "threshold_rule": "train_safe", "risk_persistence": 5, "confirmed_persistence": 5, "require_risk": True, "confirmation_delta_min": 0.01},
            {"candidate_id": "D_touchdown_reference_100ms", "history_ms": 100, "feature": "stateful", "risk": True, "machine": True, "threshold_rule": "train_safe", "risk_persistence": 5, "confirmed_persistence": 5, "require_risk": True, "confirmation_delta_min": 0.01},
        ],
        "sink": [
            {"candidate_id": "A_stateless_20ms", "history_ms": 20, "feature": "raw", "risk": False, "machine": False, "threshold_rule": "train_safe", "risk_persistence": 1, "confirmed_persistence": 1, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "B_causal_50ms", "history_ms": 50, "feature": "raw", "risk": False, "machine": False, "threshold_rule": "train_safe", "risk_persistence": 1, "confirmed_persistence": 1, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "C_dual_history_200ms", "history_ms": 200, "feature": "raw", "risk": True, "machine": False, "threshold_rule": "fixed_0p90", "risk_persistence": 5, "confirmed_persistence": 10, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "D_touchdown_stateful_200ms", "history_ms": 200, "feature": "stateful", "risk": True, "machine": False, "threshold_rule": "fixed_0p90", "risk_persistence": 5, "confirmed_persistence": 10, "require_risk": False, "confirmation_delta_min": 0.0},
            {"candidate_id": "E_contact_gated_stateful_200ms", "history_ms": 200, "feature": "stateful", "risk": True, "machine": True, "threshold_rule": "train_safe", "risk_persistence": 5, "confirmed_persistence": 10, "require_risk": True, "confirmation_delta_min": 0.01},
        ],
    }


def predeclared_protocol() -> dict[str, object]:
    return {
        "prototype": "walking_stateful_hazard_prototype_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "development-only diagnostic prototype; not production retraining or INT8 preparation",
        "written_before_data_load_and_candidate_execution": True,
        "split": {"train": "d2209cd v00/v01", "validation_selection": "d2209cd v02", "run_episode_overlap_allowed": False},
        "outer_boundary": {"outer_content_loads": 0, "outer_experiments_rerun": False, "nested_outer_files": "SHA256 provenance only", "new_sink_holdout_access": 0, "spatial_access": 0, "final_test_access": 0},
        "runtime_inputs": ["current/past Fusion10", "FSR hysteresis contact", "time since touchdown", "touchdown/loading reference", "causal EMA/delta/slope/variance", "cumulative evidence", "previous state/score/persistence"],
        "runtime_forbidden": ["future sample", "penetration ground truth", "anchor drift ground truth", "physical label flag", "fall flag as detection feature", "terrain ID", "simulator gait phase", "run/variation/speed/profile ID", "validation truth routing"],
        "risk_horizon_ms": RISK_HORIZON,
        "candidates": candidate_definitions(),
        "candidate_count": 9,
        "model_family": "two independent deterministic logistic probes (risk and confirmed) plus engineered causal state",
        "feature_definition": {"raw": "80 summary values over causal history", "stateful": "raw + touchdown elapsed/channel delta/contact mean delta/EMA delta/exact channel variance/FSR deltas/valid bit", "contact": f"FSR sum on/off {CONTACT_ON_N}/{CONTACT_OFF_N} N, 30 ms loading"},
        "threshold_candidates": [0.90, "train stable/hard-negative maximum plus one float step"],
        "hysteresis_candidates": ["off = 0.80 * on"],
        "persistence_candidates_samples": [1, 3, 5, 10],
        "random_seed": SEED,
        "selection_priority": ["invalid firing", "normal confirmed run FP", "pre-onset confirmed firing", "confirmed recall", "risk early-warning coverage", "confirmed latency", "parameter/state memory"],
        "mandatory_common": {"invalid_firing_samples": 0, "normal_confirmed_fp_runs": 0, "pre_onset_confirmed_runs": 0, "future_leakage": 0, "run_episode_leakage": 0, "reload_parity": "exact", "reset_invariant": True},
        "slip_gate": {"physical_coverage": "3/3", "risk_coverage": "3/3", "confirmed_detection": "3/3", "normal_risk_fp": "0/9", "risk_horizon_ms": 100, "confirmed_latency_median_ms_max": 100, "controlled_recall_drop_pp_max": 10},
        "sink_gate": {"physical_coverage": "6/6", "confirmed_detection_min": "5/6", "per_profile_detection_min": "2/3", "normal_risk_fp": "0/9", "normal_confirmed_fp": "0/9", "confirmed_latency_median_ms_max": 250, "controlled_recall_drop_pp_max": 10},
        "tie_break": "candidate order after all declared metrics/resource fields",
        "failure_fallback": "select exactly one diagnostic best by fixed priority; keep readiness false and do not promote",
        "forbidden_changes": ["production Terrain/Slip/Sink", "production normalization/threshold", "frozen detector/oracle", "bounded retraining v1 lock", "INT8/Vela", "E84/HIL", "System-v1", "final test", "outer/holdout"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=portable) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("\n")
            return
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_upstream() -> dict[str, object]:
    files = []
    for relative, expected in UPSTREAM_SHA256.items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"immutable upstream mismatch: {relative}")
        files.append({
            "path": relative,
            "sha256": actual,
            "access_mode": "sha256_provenance_only" if relative in HASH_ONLY_PATHS else "development_artifact_or_metadata",
            "content_loaded": False if relative in HASH_ONLY_PATHS else relative.endswith(("traces.npz", "inputs_fusion10.npz", "oracle_diagnostics.npz")),
        })
    return {"files": files, "mismatch_count": 0, "outer_content_load_count": 0}


def extract_runtime_trace(fusion10: np.ndarray, history_ms: int, stateful: bool) -> dict[str, np.ndarray]:
    tracker = ContactPhaseTracker()
    feature_state = CausalFeatureState(history_ms)
    feature_count = 124 if stateful else 80
    features = np.full((len(fusion10), feature_count), np.nan, np.float32)
    phases = np.full(len(fusion10), DetectorState.AIR.value, dtype="<U20")
    elapsed = np.full(len(fusion10), -1, np.int32)
    reset_reason = np.full(len(fusion10), "none", dtype="<U20")
    for index, sample in enumerate(fusion10):
        phase, since_touchdown, reason = tracker.update(sample)
        phases[index] = phase.value
        elapsed[index] = since_touchdown
        reset_reason[index] = reason
        if phase in (DetectorState.AIR, DetectorState.RESET):
            feature_state.reset()
            continue
        feature_state.update(sample, reason == "touchdown")
        features[index] = feature_state.vector(stateful)
    return {"features": features, "phase": phases, "elapsed": elapsed, "reset_reason": reset_reason}


def offline_labels(trace: dict[str, np.ndarray], detector: str, runtime_phase: np.ndarray) -> dict[str, np.ndarray]:
    physical = physical_oracle(trace, detector)
    risk = np.zeros(len(physical), bool)
    onset_values = np.flatnonzero(physical)
    if onset_values.size:
        onset = int(onset_values[0])
        risk[max(0, onset - RISK_HORIZON[detector]):onset] = True
    confirmed = np.zeros(len(physical), bool)
    if onset_values.size:
        confirmed[int(onset_values[0]):] = True
    valid = (
        np.asarray(trace["loaded_contact"], bool)
        & np.asarray(trace["pre_fall_valid"], bool)
        & ~np.asarray(trace["touchdown_transient"], bool)
        & np.isin(runtime_phase.astype(str), [DetectorState.STABLE_CONTACT.value, DetectorState.RECOVERY.value])
    )
    confirmed &= valid
    risk &= valid
    stable_normal = valid & ~risk & ~confirmed
    return {"physical": physical, "risk": risk, "confirmed": confirmed, "valid": valid, "stable_normal": stable_normal}


def build_walking_feature_sets(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> dict[tuple[int, bool], list[dict[str, np.ndarray]]]:
    specs = {(100, False), (100, True), (20, False), (50, False), (200, False), (200, True)}
    output: dict[tuple[int, bool], list[dict[str, np.ndarray]]] = {}
    for history, stateful in sorted(specs):
        runs = []
        for metadata, trace in zip(manifests, traces):
            runtime = extract_runtime_trace(trace["fusion10"], history, stateful)
            runs.append({**runtime, "run_id": np.asarray(str(metadata["run_id"])), "split": np.asarray(str(metadata["split"]))})
        output[(history, stateful)] = runs
    return output


def balanced_indices(labels: np.ndarray, cap: int = 5000) -> np.ndarray:
    phase = np.full(len(labels), "all")
    return phase_balanced_indices(
        labels.astype(np.int8), phase,
        balance_class=True, balance_phase=False, cap_per_cell=cap,
    )


def fit_probe_pairs(
    output: Path,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    feature_sets: dict[tuple[int, bool], list[dict[str, np.ndarray]]],
) -> tuple[dict[tuple[str, int, bool], dict[str, object]], dict[str, object]]:
    model_dir = output / "models"
    model_dir.mkdir()
    result: dict[tuple[str, int, bool], dict[str, object]] = {}
    parity: dict[str, object] = {}
    needed = {
        ("slip", 100, False), ("slip", 100, True),
        ("sink", 20, False), ("sink", 50, False),
        ("sink", 200, False), ("sink", 200, True),
    }
    for detector, history, stateful in sorted(needed):
        x: list[np.ndarray] = []
        risk_y: list[np.ndarray] = []
        confirmed_y: list[np.ndarray] = []
        stable_normal: list[np.ndarray] = []
        confirmed_negative: list[np.ndarray] = []
        for metadata, trace, runtime in zip(manifests, traces, feature_sets[(history, stateful)]):
            if metadata["split"] != "diagnostic_train":
                continue
            labels = offline_labels(trace, detector, runtime["phase"])
            finite = labels["valid"] & np.isfinite(runtime["features"]).all(1)
            x.append(runtime["features"][finite])
            risk_y.append(labels["risk"][finite].astype(np.int8))
            confirmed_y.append(labels["confirmed"][finite].astype(np.int8))
            stable_normal.append(labels["stable_normal"][finite])
            confirmed_negative.append(~labels["confirmed"][finite])
        values = np.concatenate(x)
        risk_label = np.concatenate(risk_y)
        confirmed_label = np.concatenate(confirmed_y)
        normal_mask = np.concatenate(stable_normal)
        confirm_negative_mask = np.concatenate(confirmed_negative)
        risk_index = balanced_indices(risk_label)
        confirm_index = balanced_indices(confirmed_label)
        risk_probe = fit_logistic_probe(values[risk_index], risk_label[risk_index])
        confirmed_probe = fit_logistic_probe(values[confirm_index], confirmed_label[confirm_index])
        risk_scores = risk_probe.positive_score(values)
        confirmed_scores = confirmed_probe.positive_score(values)
        risk_safe = float(np.nextafter(np.max(risk_scores[normal_mask]), 1.0))
        confirmed_safe = float(np.nextafter(np.max(confirmed_scores[confirm_negative_mask]), 1.0))
        key = f"{detector}_{history}ms_{'stateful' if stateful else 'raw'}"
        risk_path = model_dir / f"{key}_risk.npz"
        confirmed_path = model_dir / f"{key}_confirmed.npz"
        risk_probe.save(risk_path)
        confirmed_probe.save(confirmed_path)
        risk_loaded = LinearProbe.load(risk_path)
        confirmed_loaded = LinearProbe.load(confirmed_path)
        sample = values[:1024]
        exact = bool(
            np.array_equal(risk_probe.positive_score(sample), risk_loaded.positive_score(sample))
            and np.array_equal(confirmed_probe.positive_score(sample), confirmed_loaded.positive_score(sample))
        )
        parity[key] = {"samples": len(sample), "exact": exact}
        result[(detector, history, stateful)] = {
            "risk_probe": risk_probe,
            "confirmed_probe": confirmed_probe,
            "risk_safe_threshold": risk_safe,
            "confirmed_safe_threshold": confirmed_safe,
            "train_samples": len(values),
            "risk_train_samples": len(risk_index),
            "confirmed_train_samples": len(confirm_index),
            "risk_positive_samples": int(np.sum(risk_label)),
            "confirmed_positive_samples": int(np.sum(confirmed_label)),
            "reload_parity": exact,
        }
    return result, parity


def machine_config(
    detector: str,
    definition: dict[str, object],
    model: dict[str, object],
) -> MachineConfig:
    if definition["threshold_rule"] == "fixed_0p90":
        risk_on = 0.90
        confirmed_on = 0.90
    else:
        risk_on = float(model["risk_safe_threshold"])
        confirmed_on = float(model["confirmed_safe_threshold"])
    return MachineConfig(
        detector=detector,
        candidate_id=str(definition["candidate_id"]),
        history_ms=int(definition["history_ms"]),
        use_touchdown_reference=definition["feature"] == "stateful",
        risk_enabled=bool(definition["risk"]),
        confirmed_enabled=True,
        require_risk_before_confirm=bool(definition["require_risk"]),
        risk_on=risk_on,
        risk_off=0.80 * risk_on,
        confirmed_on=confirmed_on,
        confirmed_off=0.80 * confirmed_on,
        risk_persistence=int(definition["risk_persistence"]),
        confirmed_persistence=int(definition["confirmed_persistence"]),
        recovery_persistence=10,
        confirmation_delta_min=float(definition["confirmation_delta_min"]),
    )


def simulate_machine(
    machine: StatefulHazardMachine,
    fusion10: np.ndarray,
    first_censor_sample: int | None = None,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[Any]] = {
        "risk": [], "confirmed": [], "state": [], "elapsed": [],
        "reset_reason": [], "risk_score": [], "confirmed_score": [],
    }
    for index, sample in enumerate(fusion10):
        if first_censor_sample is not None and index == first_censor_sample:
            machine.request_reset("fall_censor")
        if first_censor_sample is not None and index > first_censor_sample:
            output = machine._output(DetectorState.RESET, -1, "fall_censor")
        else:
            output = machine.step(sample)
        risk = output.slip_risk or output.sink_risk
        confirmed = output.slip_confirmed or output.sink_confirmed
        fields["risk"].append(risk)
        fields["confirmed"].append(confirmed)
        fields["state"].append(output.contact_state)
        fields["elapsed"].append(output.time_since_touchdown)
        fields["reset_reason"].append(output.reset_reason)
        fields["risk_score"].append(output.risk_score)
        fields["confirmed_score"].append(output.confirmed_score)
    return {key: np.asarray(value) for key, value in fields.items()}


def first_or_none(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if not indices.size else int(indices[0])


def walking_candidate_evaluation(
    detector: str,
    config: MachineConfig,
    model: dict[str, object],
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray] | None, float]:
    rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    reset_rows: list[dict[str, object]] = []
    representative: dict[str, np.ndarray] | None = None
    started = time.perf_counter()
    evaluated_samples = 0
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != "diagnostic_validation":
            continue
        first_invalid = first_or_none(~np.asarray(trace["pre_fall_valid"], bool))
        machine = StatefulHazardMachine(
            config, model["risk_probe"], model["confirmed_probe"]
        )
        output = simulate_machine(machine, trace["fusion10"], first_invalid)
        evaluated_samples += len(trace["fusion10"])
        physical = physical_oracle(trace, detector)
        onset = first_or_none(physical)
        risk_indices = np.flatnonzero(output["risk"])
        confirmed_indices = np.flatnonzero(output["confirmed"])
        first_risk = None if not risk_indices.size else int(risk_indices[0])
        first_confirmed = None if not confirmed_indices.size else int(confirmed_indices[0])
        post_confirmed = (
            None if onset is None else next(
                (int(value) for value in confirmed_indices if physical[value]), None
            )
        )
        risk_horizon_start = None if onset is None else max(0, onset - RISK_HORIZON[detector])
        valid_risk = bool(
            onset is not None and np.any(
                output["risk"][risk_horizon_start:onset + 1]
            )
        )
        post_risk = bool(onset is not None and np.any(output["risk"] & physical))
        too_early_risk = bool(
            onset is not None and first_risk is not None and first_risk < risk_horizon_start
        )
        pre_confirmed = bool(
            onset is not None and first_confirmed is not None and first_confirmed < onset
        )
        role = str(metadata["acquisition_role"])
        physical_positive = role == f"{detector}_candidate" and onset is not None
        normal_risk_fp = bool(role == "hard_negative" and risk_indices.size)
        normal_confirmed_fp = bool(role == "hard_negative" and confirmed_indices.size)
        air = ~np.asarray(trace["loaded_contact"], bool)
        touchdown = np.asarray(trace["touchdown_transient"], bool)
        postfall = ~np.asarray(trace["pre_fall_valid"], bool)
        air_firing = int(np.count_nonzero((output["risk"] | output["confirmed"]) & air))
        touchdown_confirmed = int(np.count_nonzero(output["confirmed"] & touchdown))
        postfall_firing = int(np.count_nonzero((output["risk"] | output["confirmed"]) & postfall))
        rows.append({
            "detector": detector, "candidate_id": config.candidate_id,
            "run_id": metadata["run_id"], "profile": metadata["profile_name"],
            "speed_mps": float(metadata["walking_speed_mps"]), "role": role,
            "physical_positive": physical_positive, "physical_onset_sample": onset,
            "first_risk_sample": first_risk, "risk_in_horizon": valid_risk,
            "risk_after_onset": post_risk, "risk_too_early": too_early_risk,
            "first_confirmed_sample": first_confirmed,
            "pre_onset_confirmed": pre_confirmed,
            "post_onset_confirmed_sample": post_confirmed,
            "confirmed_detected": post_confirmed is not None,
            "confirmed_latency_ms": None if post_confirmed is None else post_confirmed - int(onset),
            "normal_risk_fp": normal_risk_fp,
            "normal_confirmed_fp": normal_confirmed_fp,
        })
        invalid_rows.append({
            "detector": detector, "candidate_id": config.candidate_id,
            "run_id": metadata["run_id"], "AIR_firing_samples": air_firing,
            "touchdown_confirmed_samples": touchdown_confirmed,
            "post_fall_firing_samples": postfall_firing,
            "invalid_firing_samples": air_firing + touchdown_confirmed + postfall_firing,
        })
        inferred_resets = np.flatnonzero(output["state"].astype(str) == DetectorState.RESET.value)
        causal_contact = False
        expected_contact_losses: list[int] = []
        for sample_index, sample in enumerate(trace["fusion10"]):
            if first_invalid is not None and sample_index >= first_invalid:
                break
            fsr_sum = float(np.sum(sample[:4]))
            if not causal_contact and fsr_sum >= CONTACT_ON_N:
                causal_contact = True
            elif causal_contact and fsr_sum <= CONTACT_OFF_N:
                expected_contact_losses.append(sample_index)
                causal_contact = False
        actual_contact_losses = np.flatnonzero(
            output["reset_reason"].astype(str) == "contact_loss"
        )
        missed = sum(
            not np.any(actual_contact_losses == loss)
            for loss in expected_contact_losses
        )
        unexpected = sum(
            not np.any(np.asarray(expected_contact_losses) == loss)
            for loss in actual_contact_losses
        )
        hazard_on_reset = int(np.count_nonzero(
            (output["risk"] | output["confirmed"])
            & (output["state"].astype(str) == DetectorState.RESET.value)
        ))
        reset_rows.append({
            "detector": detector, "candidate_id": config.candidate_id,
            "run_id": metadata["run_id"],
            "causal_fsr_contact_losses": len(expected_contact_losses),
            "contact_loss_reset_events": len(actual_contact_losses),
            "reset_events": len(inferred_resets), "missed_contact_loss_resets": missed,
            "unexpected_contact_loss_resets": unexpected,
            "hazard_true_on_reset": hazard_on_reset,
            "fall_censor_reset_present": bool(
                first_invalid is None
                or output["reset_reason"][first_invalid] == "fall_censor"
            ),
            "episode_boundary_initialized_clean": True,
            "reset_invariant_pass": (
                missed == 0 and unexpected == 0 and hazard_on_reset == 0
            ),
        })
        if representative is None and physical_positive:
            representative = {
                "time_s": trace["time_s"].astype(np.float64),
                "risk_score": output["risk_score"].astype(np.float32),
                "confirmed_score": output["confirmed_score"].astype(np.float32),
                "risk": output["risk"].astype(bool),
                "confirmed": output["confirmed"].astype(bool),
                "state": output["state"].astype("<U20"),
                "physical": physical.astype(bool),
                "run_id": np.asarray(str(metadata["run_id"])),
            }
    host_us = (time.perf_counter() - started) * 1e6 / max(evaluated_samples, 1)
    return rows, invalid_rows, reset_rows, representative, host_us


def controlled_candidate_recall(
    detector: str,
    config: MachineConfig,
    model: dict[str, object],
    controlled: dict[str, object],
) -> dict[str, object]:
    target_name = "confirmed_slip" if detector == "slip" else "sustained_sink"
    target = np.asarray(controlled[target_name], bool)
    detected = 0
    positives = 0
    false_runs = 0
    for index, metadata in enumerate(controlled["rows"]):
        if metadata["split"] != "validation":
            continue
        machine = StatefulHazardMachine(
            config, model["risk_probe"], model["confirmed_probe"]
        )
        output = simulate_machine(machine, np.asarray(controlled["sensors"])[index])
        onset = first_or_none(target[index])
        confirmed = np.flatnonzero(output["confirmed"])
        if onset is not None:
            positives += 1
            if any(value >= onset for value in confirmed):
                detected += 1
        elif confirmed.size:
            false_runs += 1
    recall = detected / max(positives, 1)
    return {
        "positive_runs": positives, "detected_runs": detected,
        "run_recall": recall, "negative_false_positive_runs": false_runs,
        "frozen_baseline_recall": CONTROLLED_BASELINE[detector],
        "retention_pass": recall >= CONTROLLED_BASELINE[detector] - 0.10,
    }


def aggregate_candidate(
    detector: str,
    order: int,
    definition: dict[str, object],
    config: MachineConfig,
    model: dict[str, object],
    run_rows: list[dict[str, object]],
    invalid_rows: list[dict[str, object]],
    reset_rows: list[dict[str, object]],
    controlled: dict[str, object],
    host_us: float,
) -> dict[str, object]:
    normal = [row for row in run_rows if row["role"] == "hard_negative"]
    positive = [row for row in run_rows if row["physical_positive"]]
    detected = [row for row in positive if row["confirmed_detected"]]
    risk_covered = [row for row in positive if row["risk_in_horizon"] or row["risk_after_onset"]]
    latencies = [float(row["confirmed_latency_ms"]) for row in detected]
    resources = estimate_resources(config, model["risk_probe"], model["confirmed_probe"])
    profile_counts: dict[str, dict[str, int]] = {}
    for profile in sorted({str(row["profile"]) for row in positive}):
        group = [row for row in positive if row["profile"] == profile]
        profile_counts[profile] = {
            "physical": len(group),
            "detected": sum(bool(row["confirmed_detected"]) for row in group),
            "risk_covered": sum(bool(row["risk_in_horizon"] or row["risk_after_onset"]) for row in group),
        }
    invalid = sum(int(row["invalid_firing_samples"]) for row in invalid_rows)
    reset_pass = all(bool(row["reset_invariant_pass"]) and bool(row["fall_censor_reset_present"]) for row in reset_rows)
    runtime_common = bool(
        invalid == 0 and reset_pass and bool(model["reload_parity"])
    )
    confirmed_common = bool(
        runtime_common
        and sum(bool(row["normal_confirmed_fp"]) for row in normal) == 0
        and sum(bool(row["pre_onset_confirmed"]) for row in positive) == 0
    )
    risk_normal_fp = sum(bool(row["normal_risk_fp"]) for row in normal)
    controlled_ok = bool(controlled["retention_pass"])
    if detector == "slip":
        risk_gate = bool(
            len(positive) == 3
            and len(risk_covered) == 3
            and risk_normal_fp == 0
            and not any(bool(row["risk_too_early"]) for row in positive)
        )
        confirmed_gate = bool(
            len(positive) == 3
            and len(detected) == 3
            and latencies and float(np.median(latencies)) <= 100.0
            and controlled_ok
        )
    else:
        risk_gate = bool(
            len(positive) == 6
            and len(risk_covered) == 6
            and risk_normal_fp == 0
            and not any(bool(row["risk_too_early"]) for row in positive)
        )
        confirmed_gate = bool(
            len(positive) == 6
            and len(detected) >= 5
            and profile_counts
            and all(value["detected"] >= 2 for value in profile_counts.values())
            and latencies and float(np.median(latencies)) <= 250.0
            and controlled_ok
        )
    return {
        "detector": detector, "candidate_id": config.candidate_id,
        "candidate_order": order, "comparison": definition["candidate_id"],
        "history_ms": config.history_ms,
        "feature_set": definition["feature"], "state_machine": bool(definition["machine"]),
        "risk_threshold": config.risk_on, "risk_off_threshold": config.risk_off,
        "confirmed_threshold": config.confirmed_on,
        "confirmed_off_threshold": config.confirmed_off,
        "risk_persistence": config.risk_persistence,
        "confirmed_persistence": config.confirmed_persistence,
        "threshold_rule": definition["threshold_rule"],
        "train_samples": model["train_samples"],
        "validation_runs": len(run_rows), "normal_runs": len(normal),
        "physical_positive_runs": len(positive),
        "risk_run_coverage": len(risk_covered) / max(len(positive), 1),
        "risk_covered_runs": len(risk_covered),
        "normal_risk_fp_runs": risk_normal_fp,
        "risk_too_early_runs": sum(bool(row["risk_too_early"]) for row in positive),
        "confirmed_run_recall": len(detected) / max(len(positive), 1),
        "confirmed_detected_runs": len(detected),
        "normal_confirmed_fp_runs": sum(bool(row["normal_confirmed_fp"]) for row in normal),
        "pre_onset_confirmed_runs": sum(bool(row["pre_onset_confirmed"]) for row in positive),
        "confirmed_latency_median_ms": None if not latencies else float(np.median(latencies)),
        "confirmed_latency_p95_ms": None if not latencies else float(np.percentile(latencies, 95)),
        "invalid_firing_samples": invalid,
        "reset_invariant_pass": reset_pass,
        "reload_parity_exact": model["reload_parity"],
        "profile_results_json": json.dumps(profile_counts, sort_keys=True),
        "controlled_run_recall": controlled["run_recall"],
        "controlled_baseline_recall": controlled["frozen_baseline_recall"],
        "controlled_retention_pass": controlled_ok,
        "host_inference_us_per_sample": host_us,
        **resources,
        "risk_gate_pass": runtime_common and risk_gate,
        "confirmed_gate_pass": confirmed_common and confirmed_gate,
        "mandatory_gate_pass": (
            runtime_common and confirmed_common and risk_gate and confirmed_gate
        ),
        "production_candidate": False,
    }


def contact_phase_audit(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare FSR-only phase inference with simulator diagnostics offline."""
    confusion: dict[tuple[str, str], int] = {}
    phase_segments = {state: 0 for state in ("AIR", "LOADING", "STABLE", "RECOVERY")}
    timing_rows: list[dict[str, object]] = []
    total = 0
    matching = 0
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != "diagnostic_validation":
            continue
        tracker = ContactPhaseTracker()
        inferred: list[str] = []
        for sample in trace["fusion10"]:
            state, _, _ = tracker.update(sample)
            inferred.append(state.value)
        predicted = np.asarray(inferred).astype(str)
        predicted_mapped = np.asarray([
            "AIR" if value in ("AIR", "RESET") else
            "STABLE" if value in ("STABLE_CONTACT", "HAZARD_RISK", "HAZARD_CONFIRMED") else
            value
            for value in predicted
        ])
        diagnostic = np.asarray(trace["gait_phase"]).astype(str)
        truth = np.asarray([
            "LOADING" if value in ("TOUCHDOWN", "LOADING") else
            "STABLE" if value == "MID_STANCE" else
            "RECOVERY" if value == "PUSH_OFF" else "AIR"
            for value in diagnostic
        ])
        for true_value, predicted_value in zip(truth, predicted_mapped):
            confusion[(true_value, predicted_value)] = (
                confusion.get((true_value, predicted_value), 0) + 1
            )
        total += len(truth)
        matching += int(np.count_nonzero(truth == predicted_mapped))
        for state in phase_segments:
            mask = predicted_mapped == state
            phase_segments[state] += int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))
        physical_contact = np.asarray(trace["loaded_contact"], bool)
        physical_onsets = np.flatnonzero(physical_contact & ~np.r_[False, physical_contact[:-1]])
        inferred_onsets = np.flatnonzero(
            np.isin(predicted, ["LOADING", "STABLE_CONTACT", "RECOVERY"])
            & ~np.r_[False, np.isin(predicted[:-1], ["LOADING", "STABLE_CONTACT", "RECOVERY"])]
        )
        for episode_index, onset in enumerate(physical_onsets):
            inferred_onset = next(
                (int(value) for value in inferred_onsets if value >= onset - 10), None
            )
            timing_rows.append({
                "row_type": "touchdown_timing",
                "run_id": metadata["run_id"],
                "episode_index": episode_index,
                "physical_touchdown_sample": int(onset),
                "inferred_touchdown_sample": inferred_onset,
                "phase_inference_latency_ms": (
                    None if inferred_onset is None else inferred_onset - int(onset)
                ),
            })
    confusion_rows = [
        {
            "row_type": "phase_confusion",
            "diagnostic_phase": true_value,
            "inferred_phase": predicted_value,
            "sample_count": count,
        }
        for (true_value, predicted_value), count in sorted(confusion.items())
    ]
    latencies = [
        float(row["phase_inference_latency_ms"])
        for row in timing_rows if row["phase_inference_latency_ms"] is not None
    ]
    summary = {
        "evaluation_role": "read-only; simulator gait phase was never a runtime input",
        "phase_segment_counts": phase_segments,
        "sample_agreement": matching / max(total, 1),
        "touchdown_events": len(timing_rows),
        "touchdown_detected": len(latencies),
        "touchdown_latency_median_ms": None if not latencies else float(np.median(latencies)),
        "touchdown_latency_p95_abs_ms": None if not latencies else float(np.percentile(np.abs(latencies), 95)),
        "contact_state_ready": bool(total and len(latencies) == len(timing_rows)),
    }
    summary_rows = [{"row_type": "summary", **summary, "phase_segment_counts": json.dumps(phase_segments, sort_keys=True)}]
    return summary_rows + confusion_rows + timing_rows, summary


def runtime_feature_contract() -> dict[str, object]:
    rows = [
        ("Fusion10 current sample", "current FSR4 + IMU6", "yes"),
        ("causal history", "ring buffer ending at current sample", "yes"),
        ("FSR contact hysteresis", f"sum(FSR4), on={CONTACT_ON_N} N/off={CONTACT_OFF_N} N", "yes"),
        ("time since touchdown", "integer samples since causal contact-on", "yes"),
        ("touchdown reference", "Fusion10 captured on causal contact-on", "yes"),
        ("history statistics", "mean/std/min/max/first/last/delta/diff RMS plus exact channel variance", "yes"),
        ("EMA", "0.95 previous + 0.05 current; delta from touchdown reference is a probe feature", "yes"),
        ("cumulative evidence", "contact-episode sum and count", "yes"),
        ("previous state/score", "bounded machine persistence and hysteresis", "yes"),
    ]
    return {
        "sensor_rate_hz": 1000,
        "runtime_api": "StatefulHazardMachine.step(fusion10_sample)",
        "feature_rows": [
            {"feature": name, "definition": definition, "e84_reproducible": value}
            for name, definition, value in rows
        ],
        "forbidden_runtime_inputs": predeclared_protocol()["runtime_forbidden"],
        "future_sample_count": 0,
        "oracle_input_count": 0,
        "simulator_phase_input_count": 0,
    }


def state_transition_definition() -> dict[str, object]:
    return {
        "states": [state.value for state in DetectorState],
        "transitions": [
            {"from": "AIR", "to": "LOADING", "guard": "FSR sum crosses contact_on"},
            {"from": "LOADING", "to": "STABLE_CONTACT", "guard": "30 causal contact samples"},
            {"from": "STABLE_CONTACT", "to": "HAZARD_RISK", "guard": "independent risk evidence + persistence"},
            {"from": "HAZARD_RISK", "to": "HAZARD_CONFIRMED", "guard": "independent confirmed evidence + persistence; never elapsed-time-only"},
            {"from": "HAZARD_RISK/HAZARD_CONFIRMED", "to": "RECOVERY", "guard": "both scores below hysteresis-off for recovery persistence"},
            {"from": "any contact state", "to": "RESET", "guard": "contact loss, fall censor, episode boundary, or operator reset"},
            {"from": "RESET", "to": "AIR", "guard": "next finite sample below contact_on"},
        ],
        "independent_machines": ["slip", "sink"],
        "runtime_outputs": ["slip_risk", "slip_confirmed", "sink_risk", "sink_confirmed", "contact_state", "time_since_touchdown", "reset_reason"],
        "confirmation_not_timer_promoted": True,
    }


def profile_speed_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = sorted({str(row["candidate_id"]) for row in run_rows})
    for candidate in candidates:
        source = [row for row in run_rows if row["candidate_id"] == candidate]
        groups = sorted({(str(row["profile"]), float(row["speed_mps"])) for row in source})
        for profile, speed in groups:
            group = [row for row in source if row["profile"] == profile and row["speed_mps"] == speed]
            positive = [row for row in group if row["physical_positive"]]
            rows.append({
                "row_type": "profile_speed",
                "detector": group[0]["detector"], "candidate_id": candidate,
                "profile": profile, "speed_mps": speed, "run_count": len(group),
                "physical_positive_runs": len(positive),
                "risk_covered_runs": sum(bool(row["risk_in_horizon"] or row["risk_after_onset"]) for row in positive),
                "confirmed_detected_runs": sum(bool(row["confirmed_detected"]) for row in positive),
                "normal_risk_fp_runs": sum(bool(row["normal_risk_fp"]) for row in group),
                "normal_confirmed_fp_runs": sum(bool(row["normal_confirmed_fp"]) for row in group),
            })
    return rows


def conclusion(risk_ready: bool, confirmed_ready: bool) -> dict[str, str]:
    if risk_ready and confirmed_ready:
        return {"primary": "STATEFUL_PROTOTYPE_READY", "secondary": "INCONCLUSIVE"}
    if risk_ready:
        return {"primary": "RISK_ONLY_READY", "secondary": "LABEL_SEMANTICS_REDESIGN_REQUIRED"}
    if confirmed_ready:
        return {"primary": "CONFIRMED_ONLY_READY", "secondary": "LABEL_SEMANTICS_REDESIGN_REQUIRED"}
    return {"primary": "STATEFUL_HISTORY_INSUFFICIENT", "secondary": "LABEL_SEMANTICS_REDESIGN_REQUIRED"}


def make_timeline_plot(output: Path, detector: str, values: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = values["time_s"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(time_s, values["risk_score"], label="risk score")
    axes[0].plot(time_s, values["confirmed_score"], label="confirmed score")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("score")
    axes[1].step(time_s, values["risk"].astype(int), where="post", label=f"{detector}_risk")
    axes[1].step(time_s, values["confirmed"].astype(int), where="post", label=f"{detector}_confirmed")
    axes[1].step(time_s, values["physical"].astype(int), where="post", label="offline physical oracle", alpha=0.7)
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("state")
    figure.suptitle(f"{detector.title()} selected diagnostic timeline: {str(values['run_id'])}")
    figure.tight_layout()
    figure.savefig(output / f"{detector}_representative_state_score_timeline.png", dpi=150)
    plt.close(figure)


def artifact_manifest(output: Path, upstream: dict[str, object]) -> dict[str, object]:
    generated = []
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "manifest.json"):
        generated.append({
            "path": str(path.relative_to(output)), "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    return {
        "artifact": "walking_stateful_hazard_prototype_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "development_checkpoint": DEVELOPMENT_CHECKPOINT,
        "generated_files": generated,
        "hash_graph_complete": True,
        "manifest_self_hash_excluded": True,
        "immutable_upstream_sha256": {
            row["path"]: row["sha256"] for row in upstream["files"]
        },
        "outer_content_load_count": 0,
        "production_artifacts_changed": False,
        "int8_or_vela_executed": False,
    }


def write_audit(output: Path, summary: dict[str, object]) -> None:
    slip = summary["detectors"]["slip"]
    sink = summary["detectors"]["sink"]
    text = f"""# Walking Stateful Hazard Detector Prototype v1

This is a development-only v00/v01 training and v02 selection result. No nested outer,
new Sink holdout, spatial, final-test, production, INT8, or Vela content was used.

## Result

- Slip: {slip['conclusion']['primary']} / {slip['conclusion']['secondary']}
- Sink: {sink['conclusion']['primary']} / {sink['conclusion']['secondary']}
- Slip selection: {slip['selected_candidate']} (fallback={str(slip['fallback']).lower()})
- Sink selection: {sink['selected_candidate']} (fallback={str(sink['fallback']).lower()})
- Full prototype ready: {str(summary['readiness']['WALKING_STATEFUL_PROTOTYPE_READY']).lower()}

RISK is a bounded pre-onset early-warning state. CONFIRMED uses a separately fitted
score and persistence; it is never promoted solely by waiting after RISK. Physical
oracles and simulator gait phase are used only by offline labels/evaluation.

## Boundary audit

- Outer content loads: 0
- New Sink holdout/spatial/final-test accesses: 0
- Production/INT8/Vela/E84/System-v1 writes: 0
- Candidate count: 9 (four Slip, five Sink)
- Search/gate changes after v02 observation: 0
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run_prototype(output: Path) -> dict[str, object]:
    started = time.perf_counter()
    upstream = verify_upstream()
    manifests, traces = load_walking()
    controlled = load_controlled()
    run_ids: list[str] = []
    splits: list[str] = []
    episode_ids: list[int] = []
    for metadata, trace in zip(manifests, traces):
        for episode in np.unique(trace["contact_episode_id"]):
            if int(episode) >= 0:
                run_ids.append(str(metadata["run_id"]))
                splits.append(str(metadata["split"]))
                episode_ids.append(int(episode))
    leakage = split_integrity(np.asarray(run_ids), np.asarray(splits), np.asarray(episode_ids))
    if leakage["split_leakage_count"]:
        raise ValueError(f"walking split leakage: {leakage}")
    feature_sets = build_walking_feature_sets(manifests, traces)
    models, parity = fit_probe_pairs(output, manifests, traces, feature_sets)
    definitions = candidate_definitions()
    aggregate: dict[str, list[dict[str, object]]] = {"slip": [], "sink": []}
    all_runs: dict[str, list[dict[str, object]]] = {"slip": [], "sink": []}
    all_invalid: list[dict[str, object]] = []
    all_reset: list[dict[str, object]] = []
    representatives: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    controlled_results: dict[str, dict[str, object]] = {}
    for detector in ("slip", "sink"):
        for order, definition in enumerate(definitions[detector]):
            key = (detector, int(definition["history_ms"]), definition["feature"] == "stateful")
            model = models[key]
            config = machine_config(detector, definition, model)
            run_rows, invalid_rows, reset_rows, representative, host_us = walking_candidate_evaluation(
                detector, config, model, manifests, traces
            )
            controlled_result = controlled_candidate_recall(detector, config, model, controlled)
            row = aggregate_candidate(
                detector, order, definition, config, model, run_rows,
                invalid_rows, reset_rows, controlled_result, host_us,
            )
            aggregate[detector].append(row)
            all_runs[detector].extend(run_rows)
            all_invalid.extend(invalid_rows)
            all_reset.extend(reset_rows)
            controlled_results[f"{detector}:{config.candidate_id}"] = controlled_result
            if representative is not None:
                representatives[(detector, config.candidate_id)] = representative
    selected: dict[str, dict[str, object]] = {}
    any_passing: dict[str, bool] = {}
    for detector in ("slip", "sink"):
        selected[detector], any_passing[detector] = select_candidate(aggregate[detector])
    selection = {
        "selection_rule": predeclared_protocol()["selection_priority"],
        "detectors": {
            detector: {
                "selected_candidate": selected[detector]["candidate_id"],
                "mandatory_candidate_available": any_passing[detector],
                "fallback_used": not any_passing[detector],
                "production_candidate": False,
                "candidate_metrics": aggregate[detector],
            }
            for detector in ("slip", "sink")
        },
    }
    write_json(output / "candidate_selection.json", selection)
    for detector in ("slip", "sink"):
        metric_rows = [
            {"row_type": "candidate", **row} for row in aggregate[detector]
        ] + profile_speed_rows(all_runs[detector]) + [
            {"row_type": "run", **row} for row in all_runs[detector]
        ]
        write_csv(output / f"{detector}_stateful_metrics.csv", metric_rows)
    latency_rows = [
        row for detector in ("slip", "sink") for row in all_runs[detector]
        if row["physical_positive"]
    ]
    write_csv(output / "latency_metrics.csv", latency_rows)
    write_csv(output / "invalid_firing_audit.csv", all_invalid)
    write_csv(output / "reset_invariant_audit.csv", all_reset)
    write_csv(output / "resource_estimate.csv", [
        row for detector in ("slip", "sink") for row in aggregate[detector]
    ])
    phase_rows, phase_summary = contact_phase_audit(manifests, traces)
    write_csv(output / "contact_phase_metrics.csv", phase_rows)
    write_json(output / "state_transition_definition.json", state_transition_definition())
    write_json(output / "runtime_feature_contract.json", runtime_feature_contract())
    write_json(output / "reload_parity.json", parity)
    access = {
        "outer_content_load_count": 0,
        "new_sink_holdout_content_load_count": 0,
        "spatial_content_load_count": 0,
        "final_test_content_load_count": 0,
        "loaded_development_files": [
            str((WALKING / "manifest.json").relative_to(ROOT)),
            str((WALKING / "traces.npz").relative_to(ROOT)),
            str((CONTROLLED / "protocol.json").relative_to(ROOT)),
            str((CONTROLLED / "manifest.csv").relative_to(ROOT)),
            str((CONTROLLED / "inputs_fusion10.npz").relative_to(ROOT)),
            str((CONTROLLED / "oracle_diagnostics.npz").relative_to(ROOT)),
        ],
        "hash_only_files": sorted(HASH_ONLY_PATHS),
    }
    write_json(output / "accessed_file_manifest.json", access)
    for detector in ("slip", "sink"):
        representative = representatives[(detector, str(selected[detector]["candidate_id"]))]
        make_timeline_plot(output, detector, representative)
        np.savez_compressed(
            output / f"{detector}_representative_diagnostics.npz", **representative
        )
    risk_ready = {
        detector: any(bool(row["risk_gate_pass"]) for row in aggregate[detector])
        for detector in ("slip", "sink")
    }
    confirmed_ready = {
        detector: any(bool(row["confirmed_gate_pass"]) for row in aggregate[detector])
        for detector in ("slip", "sink")
    }
    readiness = {
        "WALKING_STATEFUL_DATA_READY": True,
        "WALKING_STATEFUL_SPLIT_INTEGRITY_READY": leakage["split_leakage_count"] == 0,
        "WALKING_CAUSAL_FEATURE_CONTRACT_READY": True,
        "WALKING_CONTACT_STATE_READY": bool(phase_summary["contact_state_ready"]),
        "WALKING_SLIP_RISK_PROTOTYPE_READY": risk_ready["slip"],
        "WALKING_SLIP_CONFIRMED_PROTOTYPE_READY": confirmed_ready["slip"],
        "WALKING_SINK_RISK_PROTOTYPE_READY": risk_ready["sink"],
        "WALKING_SINK_CONFIRMED_PROTOTYPE_READY": confirmed_ready["sink"],
        "WALKING_STATEFUL_PROTOTYPE_READY": bool(any_passing["slip"] and any_passing["sink"]),
        "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED": bool(any_passing["slip"] and any_passing["sink"]),
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
    }
    summary = {
        "artifact": "walking_stateful_hazard_prototype_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "development_checkpoint": DEVELOPMENT_CHECKPOINT,
        "scope": "development-only diagnostic prototype",
        "data": {
            "training_variations": ["v00", "v01"], "selection_variations": ["v02"],
            "run_episode_split_integrity": leakage,
            "sensor_rate_hz": 1000, "walking_trace_shape": [3000, 10],
            "outer_content_load_count": 0,
        },
        "state_machine": state_transition_definition(),
        "contact_phase": phase_summary,
        "detectors": {
            detector: {
                "selected_candidate": selected[detector]["candidate_id"],
                "fallback": not any_passing[detector],
                "risk_ready": risk_ready[detector],
                "confirmed_ready": confirmed_ready[detector],
                "risk_gate_candidates": [
                    row["candidate_id"] for row in aggregate[detector]
                    if bool(row["risk_gate_pass"])
                ],
                "confirmed_gate_candidates": [
                    row["candidate_id"] for row in aggregate[detector]
                    if bool(row["confirmed_gate_pass"])
                ],
                "mandatory_gate_pass": bool(selected[detector]["mandatory_gate_pass"]),
                "conclusion": conclusion(
                    risk_ready[detector], confirmed_ready[detector],
                ),
                "selected_metrics": selected[detector],
                "controlled_retention": controlled_results[f"{detector}:{selected[detector]['candidate_id']}"],
            }
            for detector in ("slip", "sink")
        },
        "candidate_count": 9,
        "validation_adaptive_changes": 0,
        "readiness": readiness,
        "production_candidate_approved": False,
        "int8_or_vela_executed": False,
        "immutable_upstream_mismatch_count": upstream["mismatch_count"],
        "wall_time_s": time.perf_counter() - started,
    }
    next_step = (
        "bounded retraining v2"
        if readiness["WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED"]
        else "label/task redesign"
    )
    summary["next_step"] = next_step
    write_json(output / "summary.json", summary)
    write_json(output / "readiness.json", {
        "gates": readiness,
        "overall_ready": readiness["WALKING_STATEFUL_PROTOTYPE_READY"],
        "diagnostic_only": True,
        "bounded_retraining_v2_authorized": readiness["WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED"],
        "int8_preparation_authorized": False,
        "next_step": next_step,
    })
    write_audit(output, summary)
    write_json(output / "manifest.json", artifact_manifest(output, upstream))
    return summary


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("Dry run only. Use --execute; locked outer/spatial arrays are SHA-only.")
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty prototype output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # The protocol is durably fixed before any source is loaded or candidate runs.
    write_json(output / "protocol.json", predeclared_protocol())
    summary = run_prototype(output)
    print(json.dumps({
        "output": str(output), "readiness": summary["readiness"],
        "next_step": summary["next_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
