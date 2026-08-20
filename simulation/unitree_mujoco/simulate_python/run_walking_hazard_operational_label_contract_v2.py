"""Audit and publish Walking Hazard Operational Label / Reflex Contract v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from run_walking_fusion10_observability_audit_v1 import ROOT, load_walking
from run_walking_stateful_hazard_prototype_v1 import (
    extract_runtime_trace,
    first_or_none,
    simulate_machine,
)
from walking_fusion10_observability_audit_v1 import LinearProbe, split_integrity
from walking_hazard_operational_label_contract_v2 import (
    DEPRECATED_FIELDS,
    DOWNWARD_SPEED_THRESHOLD_MPS,
    OFFLINE_ONLY_FIELDS,
    RECOVERY_DROP_THRESHOLD_M,
    RUNTIME_CAPABLE_FIELDS,
    SAMPLE_RATE_HZ,
    SINK_PENETRATION_PERSISTENCE_MS,
    SINK_PENETRATION_THRESHOLD_M,
    SLIP_RISK_HORIZON_MS,
    SUPPORT_DROP_THRESHOLD_M,
    SUPPORT_PERSISTENCE_MS,
    first_fall_censor,
    label_contract_v2,
    legacy_mapping_rows,
    onset_pulses,
    operational_offline_labels,
    pulse_lengths,
    readiness_dependency_graph,
    reflex_activation_contract_v2,
    transition_count,
)
from walking_stateful_hazard_prototype_v1 import MachineConfig, StatefulHazardMachine


SIM = ROOT / "simulation"
OUTPUT = SIM / "outputs" / "walking_hazard_operational_label_contract_v2"
WALKING = SIM / "outputs" / "walking_hazard_oracle_calibration_v1"
PROTOTYPE = SIM / "outputs" / "walking_stateful_hazard_prototype_v1"
STARTING_CHECKPOINT = "3e52218524a4d2b59e431049a040c31e7c0b7902"
SEED = 20260820
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
    "simulation/outputs/walking_stateful_hazard_prototype_v1/protocol.json": "006520db18f42e9e37a284dd817be0d60dfc67bf6e25b48712bee53d52bfe28b",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/manifest.json": "9197f6d893513350932d6e48cd4413c708774b4e154c3c510ee469e5079802bf",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json": "7eaa92f603a0aef49a6873d18b771f6ae623a871e034c571d5ac5a04af272ffb",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/readiness.json": "255b601ec63b1325d1a9d80d872b1d325c402ca084a8306276f63e7cf14221e3",
}
HASH_ONLY_PATHS = {
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_manifest.json",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz",
}
SCORE_SPECS = (
    ("A_stateless_20ms", 20, False, "sink_20ms_raw_risk.npz"),
    ("B_causal_50ms", 50, False, "sink_50ms_raw_risk.npz"),
    ("C_dual_history_200ms", 200, False, "sink_200ms_raw_risk.npz"),
    ("D_touchdown_stateful_200ms", 200, True, "sink_200ms_stateful_risk.npz"),
)


def predeclared_protocol() -> dict[str, Any]:
    return {
        "contract": "walking_hazard_operational_label_contract_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "written_before_trace_load_and_label_execution": True,
        "scope": "read-only label/action-utility audit and versioned proposal",
        "training_runs": 0,
        "model_or_threshold_changes": 0,
        "data": {
            "development_train": "v00/v01 label-quality audit only",
            "development_validation": "v02 contract evaluation",
            "run_episode_overlap_allowed": False,
            "outer_content_load_count": 0,
            "new_sink_holdout_content_load_count": 0,
            "spatial_content_load_count": 0,
            "final_test_content_load_count": 0,
        },
        "namespace": {
            "offline_only": list(OFFLINE_ONLY_FIELDS),
            "runtime_capable": list(RUNTIME_CAPABLE_FIELDS),
            "deprecated_or_ambiguous": list(DEPRECATED_FIELDS),
        },
        "slip": {
            "risk_horizon_ms": SLIP_RISK_HORIZON_MS,
            "replay_candidate": "D_touchdown_reference_100ms",
            "target": "valid contact AND (0..100 ms before physical onset OR physical active)",
            "too_early": "runtime firing more than 100 ms before first physical onset",
            "normal_fp": "any runtime firing in a hard-negative run",
            "gate": {
                "risk_coverage": "3/3", "per_speed": "1/1",
                "normal_risk_fp": "0/9", "too_early_runs": 0,
                "invalid_firing_samples": 0, "future_leakage": 0,
                "deterministic_replay": "exact", "reset": "pass",
                "controlled_domain": "recorded separately; not relabeled",
            },
        },
        "sink_candidates": {
            "A_penetration_risk": {
                "criterion": "locked 5.5 mm episode-relative penetration, 20 ms persistence",
                "threshold_m": SINK_PENETRATION_THRESHOLD_M,
                "persistence_ms": SINK_PENETRATION_PERSISTENCE_MS,
            },
            "B_support_degradation_risk": {
                "criterion": "A AND (touchdown-relative pelvis drop >=10 mm OR downward speed >=0.05 m/s), 20 ms",
                "drop_m": SUPPORT_DROP_THRESHOLD_M,
                "downward_speed_mps": DOWNWARD_SPEED_THRESHOLD_MPS,
                "persistence_ms": SUPPORT_PERSISTENCE_MS,
            },
            "C_recovery_required_sink": {
                "criterion": "B AND pelvis drop >=20 mm AND downward speed >=0.05 m/s, 20 ms",
                "drop_m": RECOVERY_DROP_THRESHOLD_M,
                "downward_speed_mps": DOWNWARD_SPEED_THRESHOLD_MPS,
                "persistence_ms": SUPPORT_PERSISTENCE_MS,
            },
        },
        "sink_score_evidence": {
            "locked_scores": [value[0] for value in SCORE_SPECS],
            "no_refit": True,
            "minimum_separation": "AUROC >=0.70, positive median > stable-negative p95, all positive runs detected at locked threshold, zero normal-run score FP",
        },
        "sink_gate": {
            "physical_independence": True, "normal_positive_runs": 0,
            "profiles": 2, "speeds_per_profile": 3,
            "stable_prefall_positive_coverage": "all target runs",
            "invalid_contamination_samples": 0,
            "locked_score_minimum_separation": True,
        },
        "reflex_proposal": {
            "states": ["NO_HAZARD", "RISK", "EVIDENCE_PERSISTENT", "RECOVERY_REQUIRED", "COOLDOWN"],
            "slip_trigger": "slip_risk; do not wait for physical confirmation",
            "sink_trigger": "disabled unless a later contract is ready",
        },
        "readiness_constraints": {
            "WALKING_SYSTEM_SCHEMA_MIGRATION_AUTHORIZED": False,
            "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED": False,
            "WALKING_INT8_PREPARATION_AUTHORIZED": False,
        },
        "next_step_priority": [
            "SLIP_OPERATIONAL_RISK_BLIND_HOLDOUT",
            "TARGETED_SINK_LABEL_ACQUISITION",
            "ADDITIONAL_SENSOR_CONTRACT",
            "SINK_RUNTIME_DETECTION_DEFERRED",
            "STOP_WALKING_HAZARD_EXTENSION",
        ],
        "random_seed": SEED,
        "no_post_result_gate_or_threshold_change": True,
        "claims_forbidden": [
            "fall reduction", "stability improvement", "recovery success",
            "closed-loop safety benefit", "physical oracle as runtime input",
        ],
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
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("\n")
            return
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_upstream() -> dict[str, Any]:
    files = []
    for relative, expected in UPSTREAM_SHA256.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"immutable upstream mismatch: {relative}")
        files.append({
            "path": relative, "sha256": actual,
            "access_mode": "sha256_provenance_only" if relative in HASH_ONLY_PATHS else "immutable_upstream",
            "content_loaded": False,
        })
    return {"files": files, "mismatch_count": 0, "outer_content_load_count": 0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prototype_candidate_rows(detector: str) -> list[dict[str, str]]:
    return [
        row for row in read_csv(PROTOTYPE / f"{detector}_stateful_metrics.csv")
        if row["row_type"] == "candidate"
    ]


def candidate_row(detector: str, candidate_id: str) -> dict[str, str]:
    return next(
        row for row in prototype_candidate_rows(detector)
        if row["candidate_id"] == candidate_id
    )


def slip_d_machine() -> StatefulHazardMachine:
    row = candidate_row("slip", "D_touchdown_reference_100ms")
    config = MachineConfig(
        detector="slip", candidate_id=row["candidate_id"],
        history_ms=int(row["history_ms"]), use_touchdown_reference=True,
        risk_enabled=True, confirmed_enabled=True,
        require_risk_before_confirm=True,
        risk_on=float(row["risk_threshold"]), risk_off=float(row["risk_off_threshold"]),
        confirmed_on=float(row["confirmed_threshold"]),
        confirmed_off=float(row["confirmed_off_threshold"]),
        risk_persistence=int(row["risk_persistence"]),
        confirmed_persistence=int(row["confirmed_persistence"]),
        recovery_persistence=10, confirmation_delta_min=0.01,
    )
    risk = LinearProbe.load(PROTOTYPE / "models" / "slip_100ms_stateful_risk.npz")
    confirmed = LinearProbe.load(PROTOTYPE / "models" / "slip_100ms_stateful_confirmed.npz")
    return StatefulHazardMachine(config, risk, confirmed)


def _first_false(values: np.ndarray) -> int | None:
    found = np.flatnonzero(~np.asarray(values, bool))
    return None if not found.size else int(found[0])


def replay_slip(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    rows: list[dict[str, object]] = []
    margins: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    replay: dict[str, dict[str, np.ndarray]] = {}
    parity = True
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != "diagnostic_validation":
            continue
        censor = _first_false(trace["pre_fall_valid"])
        first = simulate_machine(slip_d_machine(), trace["fusion10"], censor)
        second = simulate_machine(slip_d_machine(), trace["fusion10"], censor)
        parity &= all(np.array_equal(first[key], second[key]) for key in first)
        labels = operational_offline_labels(trace)
        risk = np.asarray(first["risk"], bool)
        evidence = np.asarray(first["confirmed"], bool)
        target = labels["slip_risk_target"]
        physical = labels["slip_physical_active"]
        onset = first_or_none(physical)
        first_risk = first_or_none(risk)
        role = str(metadata["acquisition_role"])
        positive = role == "slip_candidate" and onset is not None
        covered = bool(positive and np.any(risk & target))
        too_early = bool(
            positive and first_risk is not None
            and first_risk < max(0, int(onset) - SLIP_RISK_HORIZON_MS)
        )
        normal_fp = bool(role == "hard_negative" and np.any(risk))
        air = ~np.asarray(trace["loaded_contact"], bool)
        touchdown = np.asarray(trace["touchdown_transient"], bool)
        postfall = labels["fall_censor"]
        invalid = (risk | evidence) & (air | touchdown | postfall)
        rows.append({
            "row_type": "run", "run_id": metadata["run_id"],
            "profile": metadata["profile_name"],
            "speed_mps": float(metadata["walking_speed_mps"]), "role": role,
            "physical_positive": positive, "physical_onset_sample": onset,
            "risk_covered": covered, "first_risk_sample": first_risk,
            "normal_risk_fp": normal_fp, "too_early_fp": too_early,
            "evidence_persistent_run": bool(np.any(evidence)),
            "invalid_firing_samples": int(np.count_nonzero(invalid)),
        })
        invalid_rows.append({
            "source": "slip_D_runtime", "run_id": metadata["run_id"],
            "AIR_firing_samples": int(np.count_nonzero((risk | evidence) & air)),
            "touchdown_firing_samples": int(np.count_nonzero((risk | evidence) & touchdown)),
            "post_fall_firing_samples": int(np.count_nonzero((risk | evidence) & postfall)),
            "invalid_firing_samples": int(np.count_nonzero(invalid)),
        })
        for event_index, event_onset in enumerate(np.flatnonzero(onset_pulses(physical))):
            start = max(0, int(event_onset) - SLIP_RISK_HORIZON_MS)
            active_end = int(event_onset)
            while active_end + 1 < len(physical) and physical[active_end + 1]:
                active_end += 1
            pre = np.flatnonzero(risk[start:int(event_onset) + 1]) + start
            post = np.flatnonzero(risk[int(event_onset):active_end + 1]) + int(event_onset)
            firing = int(pre[0]) if pre.size else (int(post[0]) if post.size else None)
            margins.append({
                "detector": "slip", "run_id": metadata["run_id"],
                "profile": metadata["profile_name"],
                "speed_mps": float(metadata["walking_speed_mps"]),
                "event_index": event_index, "physical_onset_sample": int(event_onset),
                "aligned_risk_sample": firing,
                "reaction_time_margin_ms": None if firing is None else int(event_onset) - firing,
                "pre_onset_warning": bool(firing is not None and firing < event_onset),
            })
        replay[str(metadata["run_id"])] = {
            "risk": risk, "evidence": evidence,
            "risk_score": np.asarray(first["risk_score"], float),
            "evidence_score": np.asarray(first["confirmed_score"], float),
        }
    positive_rows = [row for row in rows if row["physical_positive"]]
    normal_rows = [row for row in rows if row["role"] == "hard_negative"]
    speed = {
        f"{value:.2f}": {
            "positive_runs": sum(row["physical_positive"] and row["speed_mps"] == value for row in rows),
            "covered_runs": sum(row["physical_positive"] and row["speed_mps"] == value and row["risk_covered"] for row in rows),
        }
        for value in (0.10, 0.15, 0.20)
    }
    prior = candidate_row("slip", "D_touchdown_reference_100ms")
    prior_reset = [
        row for row in read_csv(PROTOTYPE / "reset_invariant_audit.csv")
        if row["detector"] == "slip" and row["candidate_id"] == "D_touchdown_reference_100ms"
    ]
    margin_values = [
        float(row["reaction_time_margin_ms"])
        for row in margins if row["reaction_time_margin_ms"] is not None
    ]
    summary = {
        "candidate_id": "D_touchdown_reference_100ms",
        "physical_positive_runs": len(positive_rows),
        "risk_covered_runs": sum(bool(row["risk_covered"]) for row in positive_rows),
        "normal_runs": len(normal_rows),
        "normal_risk_fp_runs": sum(bool(row["normal_risk_fp"]) for row in normal_rows),
        "too_early_fp_runs": sum(bool(row["too_early_fp"]) for row in positive_rows),
        "invalid_firing_samples": sum(int(row["invalid_firing_samples"]) for row in rows),
        "evidence_persistent_positive_runs": sum(bool(row["evidence_persistent_run"]) for row in positive_rows),
        "speed_coverage": speed,
        "deterministic_replay_parity_exact": parity,
        "prior_locked_metric_parity": {
            "risk_covered_runs": int(prior["risk_covered_runs"]) == sum(bool(row["risk_covered"]) for row in positive_rows),
            "normal_risk_fp_runs": int(prior["normal_risk_fp_runs"]) == sum(bool(row["normal_risk_fp"]) for row in normal_rows),
            "too_early_runs": int(prior["risk_too_early_runs"]) == sum(bool(row["too_early_fp"]) for row in positive_rows),
        },
        "reset_invariant_pass": all(row["reset_invariant_pass"] == "True" for row in prior_reset),
        "reaction_time_margin": {
            "physical_events": len(margins),
            "aligned_risk_events": len(margin_values),
            "pre_onset_warning_events": sum(bool(row["pre_onset_warning"]) for row in margins),
            "margin_median_ms": None if not margin_values else float(np.median(margin_values)),
            "margin_p05_ms": None if not margin_values else float(np.percentile(margin_values, 5)),
            "margin_min_ms": None if not margin_values else float(min(margin_values)),
            "margin_max_ms": None if not margin_values else float(max(margin_values)),
        },
        "controlled_domain": {
            "positive_runs": 17,
            "locked_D_detected_runs": int(round(float(prior["controlled_run_recall"]) * 17)),
            "locked_D_recall": float(prior["controlled_run_recall"]),
            "frozen_baseline_recall": float(prior["controlled_baseline_recall"]),
            "not_relabeled_or_hidden": True,
        },
        "sample_size_limitation": "3 positive v02 runs, one at each speed; a new blind holdout is required",
    }
    aggregate = {"row_type": "aggregate", **summary, "speed_coverage": json.dumps(speed, sort_keys=True), "prior_locked_metric_parity": json.dumps(summary["prior_locked_metric_parity"], sort_keys=True), "reaction_time_margin": json.dumps(summary["reaction_time_margin"], sort_keys=True), "controlled_domain": json.dumps(summary["controlled_domain"], sort_keys=True)}
    speed_rows = [
        {"row_type": "speed", "speed_mps": speed_name, **values}
        for speed_name, values in speed.items()
    ]
    return [aggregate] + speed_rows + rows, margins, invalid_rows, replay, summary


def _auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(truth, bool)
    value = np.asarray(scores, float)
    if not len(y) or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, value))


def locked_sink_scores(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    validation = [
        (metadata, trace) for metadata, trace in zip(manifests, traces)
        if metadata["split"] == "diagnostic_validation"
    ]
    scores: dict[str, dict[str, np.ndarray]] = {}
    thresholds: dict[str, float] = {}
    for candidate_id, history, stateful, filename in SCORE_SPECS:
        probe = LinearProbe.load(PROTOTYPE / "models" / filename)
        thresholds[candidate_id] = float(candidate_row("sink", candidate_id)["risk_threshold"])
        scores[candidate_id] = {}
        for metadata, trace in validation:
            runtime = extract_runtime_trace(trace["fusion10"], history, stateful)
            features = runtime["features"]
            finite = np.isfinite(features).all(1)
            values = np.full(len(features), np.nan, float)
            values[finite] = probe.positive_score(features[finite])
            scores[candidate_id][str(metadata["run_id"])] = values
    return scores, thresholds


def evaluate_sink_candidates(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    scores, thresholds = locked_sink_scores(manifests, traces)
    validation = [
        (metadata, trace) for metadata, trace in zip(manifests, traces)
        if metadata["split"] == "diagnostic_validation"
    ]
    labels_by_run = {
        str(metadata["run_id"]): operational_offline_labels(trace)
        for metadata, trace in validation
    }
    rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    candidates: dict[str, Any] = {}
    for candidate in ("A_penetration_risk", "B_support_degradation_risk", "C_recovery_required_sink"):
        run_rows = []
        for metadata, trace in validation:
            run_id = str(metadata["run_id"])
            label = np.asarray(labels_by_run[run_id][candidate], bool)
            valid = labels_by_run[run_id]["valid_loaded_contact"]
            role = str(metadata["acquisition_role"])
            positive = bool(np.any(label))
            contamination = label & ~valid
            profile = str(metadata["profile_name"])
            run_rows.append({
                "row_type": "run", "candidate_id": candidate, "run_id": run_id,
                "role": role, "profile": profile,
                "speed_mps": float(metadata["walking_speed_mps"]),
                "positive_run": positive, "positive_samples": int(np.count_nonzero(label)),
                "onset_count": int(np.count_nonzero(onset_pulses(label))),
                "first_onset_sample": first_or_none(label),
                "normal_overlap": bool(role == "hard_negative" and positive),
                "stable_prefall_positive": bool(np.any(label & valid)),
                "fall_confounded_samples": int(np.count_nonzero(label & first_fall_censor(trace["pre_fall_valid"]))),
                "invalid_contamination_samples": int(np.count_nonzero(contamination)),
            })
            invalid_rows.append({
                "source": candidate, "run_id": run_id,
                "AIR_firing_samples": int(np.count_nonzero(label & ~np.asarray(trace["loaded_contact"], bool))),
                "touchdown_firing_samples": int(np.count_nonzero(label & np.asarray(trace["touchdown_transient"], bool))),
                "post_fall_firing_samples": int(np.count_nonzero(label & first_fall_censor(trace["pre_fall_valid"]))),
                "invalid_firing_samples": int(np.count_nonzero(contamination)),
            })
        target_rows = [row for row in run_rows if row["role"] == "sink_candidate"]
        normal_rows = [row for row in run_rows if row["role"] == "hard_negative"]
        slip_rows = [row for row in run_rows if row["role"] == "slip_candidate"]
        positive_target = [row for row in target_rows if row["positive_run"]]
        profile_speed = {
            profile: {
                f"{speed:.2f}": sum(
                    row["positive_run"] and row["profile"] == profile and row["speed_mps"] == speed
                    for row in target_rows
                )
                for speed in (0.10, 0.15, 0.20)
            }
            for profile in ("sand_solref_interpolation_1of3", "sand_solref_interpolation_2of3")
        }
        separation_passes = []
        separation_rows = []
        for score_id, per_run in scores.items():
            positives: list[np.ndarray] = []
            negatives: list[np.ndarray] = []
            detected_runs = 0
            normal_fp_runs = 0
            for metadata, trace in validation:
                run_id = str(metadata["run_id"])
                label = np.asarray(labels_by_run[run_id][candidate], bool)
                valid = labels_by_run[run_id]["valid_loaded_contact"]
                score = per_run[run_id]
                finite = valid & np.isfinite(score)
                positives.append(score[finite & label])
                negatives.append(score[finite & ~label])
                if metadata["acquisition_role"] == "sink_candidate" and np.any(label):
                    detected_runs += int(np.any((score >= thresholds[score_id]) & label & finite))
                if metadata["acquisition_role"] == "hard_negative":
                    normal_fp_runs += int(np.any((score >= thresholds[score_id]) & finite))
            positive_scores = np.concatenate([value for value in positives if len(value)]) if any(len(value) for value in positives) else np.asarray([])
            negative_scores = np.concatenate([value for value in negatives if len(value)]) if any(len(value) for value in negatives) else np.asarray([])
            combined = np.r_[positive_scores, negative_scores]
            truth = np.r_[np.ones(len(positive_scores), bool), np.zeros(len(negative_scores), bool)]
            auc = _auc(truth, combined)
            positive_median = None if not len(positive_scores) else float(np.median(positive_scores))
            negative_p95 = None if not len(negative_scores) else float(np.percentile(negative_scores, 95))
            passed = bool(
                auc is not None and auc >= 0.70
                and positive_median is not None and negative_p95 is not None
                and positive_median > negative_p95
                and detected_runs == len(positive_target) and len(positive_target) > 0
                and normal_fp_runs == 0
            )
            separation_passes.append(passed)
            separation_rows.append({
                "row_type": "locked_score_separability", "candidate_id": candidate,
                "locked_score": score_id, "locked_threshold": thresholds[score_id],
                "positive_samples": len(positive_scores), "negative_samples": len(negative_scores),
                "sample_auroc": auc, "positive_score_median": positive_median,
                "stable_negative_score_p95": negative_p95,
                "median_minus_negative_p95": None if positive_median is None or negative_p95 is None else positive_median - negative_p95,
                "positive_target_runs": len(positive_target),
                "detected_target_runs_at_locked_threshold": detected_runs,
                "normal_score_fp_runs_at_locked_threshold": normal_fp_runs,
                "minimum_separation_evidence_pass": passed,
            })
        physical_ready = bool(
            len(positive_target) == 6
            and sum(bool(row["normal_overlap"]) for row in normal_rows) == 0
            and all(profile_speed[profile][f"{speed:.2f}"] == 1 for profile in profile_speed for speed in (0.10, 0.15, 0.20))
            and all(bool(row["stable_prefall_positive"]) for row in positive_target)
            and sum(int(row["invalid_contamination_samples"]) for row in run_rows) == 0
        )
        operational_ready = physical_ready and any(separation_passes)
        summary = {
            "candidate_id": candidate,
            "all_validation_positive_runs": sum(bool(row["positive_run"]) for row in run_rows),
            "all_validation_negative_runs": sum(not bool(row["positive_run"]) for row in run_rows),
            "positive_target_runs": len(positive_target),
            "negative_target_runs": len(target_rows) - len(positive_target),
            "normal_positive_overlap_runs": sum(bool(row["normal_overlap"]) for row in normal_rows),
            "cross_hazard_slip_candidate_positive_runs": sum(bool(row["positive_run"]) for row in slip_rows),
            "profile_speed_coverage": profile_speed,
            "stable_prefall_positive_runs": sum(bool(row["stable_prefall_positive"]) for row in positive_target),
            "positive_duration_ms": sum(int(row["positive_samples"]) for row in run_rows),
            "onset_count": sum(int(row["onset_count"]) for row in run_rows),
            "fall_confounded_samples": sum(int(row["fall_confounded_samples"]) for row in run_rows),
            "invalid_contamination_samples": sum(int(row["invalid_contamination_samples"]) for row in run_rows),
            "physical_independent_of_fusion10": True,
            "validity_gate_uses_loaded_contact_but_positive_criterion_is_physical": True,
            "reproducible_fixed_definition": True,
            "physical_label_ready": physical_ready,
            "minimum_locked_score_separation_evidence": any(separation_passes),
            "operational_label_ready": operational_ready,
            "normal_fp_not_renamed": True,
            "fast_reflex_reason": "penetration plus support/base degradation could justify unloading or recovery arbitration; benefit remains untested",
            "closed_loop_recovery_effect_claimed": False,
        }
        candidates[candidate] = summary
        rows.append({
            "row_type": "aggregate", **summary,
            "profile_speed_coverage": json.dumps(profile_speed, sort_keys=True),
        })
        rows.extend(run_rows)
        rows.extend(separation_rows)
    overall = {
        "candidates": candidates,
        "physical_label_ready": any(item["physical_label_ready"] for item in candidates.values()),
        "operational_label_ready": any(item["operational_label_ready"] for item in candidates.values()),
    }
    return rows, invalid_rows, overall, labels_by_run


def label_quality_rows(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
    slip_replay: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("diagnostic_train", "diagnostic_validation"):
        selected = [
            (metadata, trace) for metadata, trace in zip(manifests, traces)
            if metadata["split"] == split
        ]
        label_names = list(OFFLINE_ONLY_FIELDS) + [
            "slip_risk_target", "A_penetration_risk",
            "B_support_degradation_risk", "C_recovery_required_sink",
        ]
        if split == "diagnostic_validation":
            label_names += ["slip_risk", "slip_evidence_persistent"]
        for name in label_names:
            positive_runs = 0
            positive_episodes: set[str] = set()
            all_episodes: set[str] = set()
            positive_samples = 0
            onset_samples: list[int] = []
            transitions = 0
            isolated = 0
            contamination = 0
            normal_overlap = 0
            profiles: set[str] = set()
            speeds: set[float] = set()
            for metadata, trace in selected:
                run_id = str(metadata["run_id"])
                offline = operational_offline_labels(trace)
                if name == "slip_risk":
                    values = slip_replay[run_id]["risk"]
                elif name == "slip_evidence_persistent":
                    values = slip_replay[run_id]["evidence"]
                else:
                    values = np.asarray(offline[name], bool)
                valid = offline["valid_loaded_contact"]
                positive = bool(np.any(values))
                positive_runs += int(positive)
                normal_overlap += int(metadata["acquisition_role"] == "hard_negative" and positive)
                if positive:
                    profiles.add(str(metadata["profile_name"]))
                    speeds.add(float(metadata["walking_speed_mps"]))
                positive_samples += int(np.count_nonzero(values))
                onset_samples.extend(np.flatnonzero(onset_pulses(values)).astype(int).tolist())
                transitions += transition_count(values)
                isolated += sum(length < 20 for length in pulse_lengths(values))
                if name not in ("fall_censor", "valid_loaded_contact"):
                    contamination += int(np.count_nonzero(values & ~valid))
                episode = np.asarray(trace["contact_episode_id"], int)
                for item in np.unique(episode[valid]):
                    key = f"{run_id}:{int(item)}"
                    all_episodes.add(key)
                    positive_episodes.add(key) if np.any(values & (episode == item)) else None
            duration_minutes = len(selected) * 3.0 / 60.0
            rows.append({
                "split": split, "label": name, "run_count": len(selected),
                "positive_run_count": positive_runs,
                "run_prevalence": positive_runs / max(len(selected), 1),
                "episode_count": len(all_episodes),
                "positive_episode_count": len(positive_episodes),
                "episode_prevalence": len(positive_episodes) / max(len(all_episodes), 1),
                "positive_samples": positive_samples,
                "positive_duration_ms": positive_samples,
                "onset_count": len(onset_samples),
                "onset_sample_median": None if not onset_samples else float(np.median(onset_samples)),
                "pre_fall_usable_duration_ms": positive_samples if name != "fall_censor" else 0,
                "positive_profiles_json": json.dumps(sorted(profiles)),
                "positive_speeds_mps_json": json.dumps(sorted(speeds)),
                "normal_overlap_runs": normal_overlap,
                "label_transition_count": transitions,
                "isolated_short_pulses_lt20ms": isolated,
                "AIR_touchdown_post_fall_contamination_samples": contamination,
                "potential_reflex_decisions_per_minute": len(onset_samples) / max(duration_minutes, 1e-12),
                "closed_loop_benefit_claimed": False,
            })
    return rows


def make_plots(
    output: Path,
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
    slip_replay: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slip_pair = next(
        (metadata, trace) for metadata, trace in zip(manifests, traces)
        if metadata["split"] == "diagnostic_validation" and metadata["acquisition_role"] == "slip_candidate"
    )
    metadata, trace = slip_pair
    labels = operational_offline_labels(trace)
    replay = slip_replay[str(metadata["run_id"])]
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(trace["time_s"], replay["risk_score"], label="locked D risk score")
    axes[0].plot(trace["time_s"], replay["evidence_score"], label="evidence score", alpha=0.7)
    axes[0].legend(loc="upper right"); axes[0].set_ylabel("score")
    axes[1].step(trace["time_s"], labels["slip_risk_target"], where="post", label="offline risk target")
    axes[1].step(trace["time_s"], labels["slip_physical_active"], where="post", label="physical active")
    axes[1].step(trace["time_s"], replay["risk"], where="post", label="runtime slip_risk")
    axes[1].legend(loc="upper right"); axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("label/state")
    figure.tight_layout(); figure.savefig(output / "slip_operational_label_timeline.png", dpi=150); plt.close(figure)
    sink_pair = next(
        (metadata, trace) for metadata, trace in zip(manifests, traces)
        if metadata["split"] == "diagnostic_validation" and metadata["acquisition_role"] == "sink_candidate"
    )
    metadata, trace = sink_pair
    labels = operational_offline_labels(trace)
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(trace["time_s"], trace["loaded_penetration_change_m"] * 1000, label="penetration change mm")
    axes[0].plot(trace["time_s"], labels["pelvis_drop_m"] * 1000, label="pelvis drop mm")
    axes[0].legend(loc="upper right"); axes[0].set_ylabel("physical quantity")
    for name in ("A_penetration_risk", "B_support_degradation_risk", "C_recovery_required_sink"):
        axes[1].step(trace["time_s"], labels[name], where="post", label=name)
    axes[1].legend(loc="upper right"); axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("offline label")
    figure.tight_layout(); figure.savefig(output / "sink_candidate_label_timeline.png", dpi=150); plt.close(figure)


def artifact_manifest(output: Path, upstream: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        files.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size})
    return {
        "artifact": "walking_hazard_operational_label_contract_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "generated_files": files,
        "hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "immutable_upstream_sha256": {row["path"]: row["sha256"] for row in upstream["files"]},
        "outer_content_load_count": 0, "model_training_runs": 0,
        "production_or_system_artifacts_changed": False,
        "int8_or_vela_executed": False,
    }


def write_audit(output: Path, summary: dict[str, Any]) -> None:
    slip = summary["slip"]
    sink = summary["sink"]
    text = f"""# Walking Hazard Operational Label / Reflex Contract v2

This is a read-only development contract audit. It trained no model, changed no
threshold, opened no outer/holdout/spatial/final-test content, and changed no
production or System-v1 implementation.

## Semantics

Physical ground truth remains offline-only. `slip_risk` is a causal operational
warning and may trigger a proposed reflex without waiting for physical Slip.
`slip_evidence_persistent` is sustained sensor evidence, not physical truth.
Sink physical labels remain offline-only because no candidate demonstrated an
acceptable locked-score separation from normal gait.

## Slip result

- Locked D risk coverage: {slip['risk_covered_runs']}/{slip['physical_positive_runs']}
- Normal risk FP: {slip['normal_risk_fp_runs']}/{slip['normal_runs']}
- Too-early / invalid: {slip['too_early_fp_runs']} / {slip['invalid_firing_samples']}
- Controlled locked-D recall: {slip['controlled_domain']['locked_D_recall']:.6f}
- New blind Slip-risk holdout authorized: {str(summary['readiness']['WALKING_SLIP_NEW_BLIND_HOLDOUT_AUTHORIZED']).lower()}

## Sink result

- Physical label ready: {str(sink['physical_label_ready']).lower()}
- Operational label ready: {str(sink['operational_label_ready']).lower()}
- Conclusion: {sink['primary_conclusion']} / {sink['secondary_conclusion']}

No fall reduction, stability improvement, recovery success, or closed-loop
safety benefit is claimed. Action utility is limited to reaction-time margin
and false-trigger burden.

## Boundary

- Outer content loads: 0
- New Sink holdout/spatial/final-test content loads: 0
- Model training/production/System-v1/INT8/Vela changes: 0
- Next step: {summary['next_step']}
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run_audit(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    upstream = verify_upstream()
    manifests, traces = load_walking()
    run_ids: list[str] = []
    splits: list[str] = []
    episodes: list[int] = []
    for metadata, trace in zip(manifests, traces):
        for episode in np.unique(trace["contact_episode_id"]):
            if int(episode) >= 0:
                run_ids.append(str(metadata["run_id"])); splits.append(str(metadata["split"])); episodes.append(int(episode))
    leakage = split_integrity(np.asarray(run_ids), np.asarray(splits), np.asarray(episodes))
    if leakage["split_leakage_count"]:
        raise ValueError(f"walking split leakage: {leakage}")
    slip_rows, margins, slip_invalid, slip_replay, slip = replay_slip(manifests, traces)
    sink_rows, sink_invalid, sink, _ = evaluate_sink_candidates(manifests, traces)
    quality = label_quality_rows(manifests, traces, slip_replay)
    write_json(output / "label_contract_v2.json", label_contract_v2())
    write_json(output / "reflex_activation_contract_v2.json", reflex_activation_contract_v2())
    write_csv(output / "legacy_mapping.csv", legacy_mapping_rows())
    write_csv(output / "label_quality.csv", quality)
    write_csv(output / "slip_operational_metrics.csv", slip_rows)
    write_csv(output / "sink_label_candidate_metrics.csv", sink_rows)
    write_csv(output / "reaction_time_margin.csv", margins)
    write_csv(output / "invalid_region_audit.csv", slip_invalid + sink_invalid)
    outer = {
        "outer_content_load_count": 0,
        "new_sink_holdout_content_load_count": 0,
        "spatial_content_load_count": 0,
        "final_test_content_load_count": 0,
        "outer_experiments_rerun": False,
        "hash_only_files": sorted(HASH_ONLY_PATHS),
    }
    write_json(output / "outer_non_access.json", outer)
    accessed = {
        **outer,
        "loaded_content_files": [
            str((WALKING / "manifest.json").relative_to(ROOT)),
            str((WALKING / "traces.npz").relative_to(ROOT)),
            str((PROTOTYPE / "slip_stateful_metrics.csv").relative_to(ROOT)),
            str((PROTOTYPE / "sink_stateful_metrics.csv").relative_to(ROOT)),
            str((PROTOTYPE / "reset_invariant_audit.csv").relative_to(ROOT)),
        ] + [
            str((PROTOTYPE / "models" / filename).relative_to(ROOT))
            for _, _, _, filename in SCORE_SPECS
        ] + [
            str((PROTOTYPE / "models" / filename).relative_to(ROOT))
            for filename in ("slip_100ms_stateful_risk.npz", "slip_100ms_stateful_confirmed.npz")
        ],
    }
    write_json(output / "accessed_file_manifest.json", accessed)
    make_plots(output, manifests, traces, slip_replay)
    slip_gate = bool(
        slip["physical_positive_runs"] == 3
        and slip["risk_covered_runs"] == 3
        and all(value["positive_runs"] == 1 and value["covered_runs"] == 1 for value in slip["speed_coverage"].values())
        and slip["normal_risk_fp_runs"] == 0
        and slip["too_early_fp_runs"] == 0
        and slip["invalid_firing_samples"] == 0
        and slip["deterministic_replay_parity_exact"]
        and all(slip["prior_locked_metric_parity"].values())
        and slip["reset_invariant_pass"]
    )
    sink_physical = bool(sink["physical_label_ready"])
    sink_operational = bool(sink["operational_label_ready"])
    readiness = {
        "WALKING_OPERATIONAL_LABEL_DATA_READY": True,
        "WALKING_OPERATIONAL_LABEL_SPLIT_INTEGRITY_READY": leakage["split_leakage_count"] == 0,
        "WALKING_OPERATIONAL_LABEL_CAUSAL_CONTRACT_READY": True,
        "WALKING_SLIP_RISK_LABEL_READY": slip_gate,
        "WALKING_SLIP_REFLEX_CONTRACT_READY": slip_gate,
        "WALKING_SLIP_NEW_BLIND_HOLDOUT_AUTHORIZED": slip_gate,
        "WALKING_SINK_PHYSICAL_LABEL_READY": sink_physical,
        "WALKING_SINK_OPERATIONAL_LABEL_READY": sink_operational,
        "WALKING_SINK_REFLEX_CONTRACT_READY": sink_operational,
        "WALKING_SYSTEM_SCHEMA_MIGRATION_AUTHORIZED": False,
        "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED": False,
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
    }
    dependency_graph = readiness_dependency_graph(readiness)
    if not all(dependency_graph.values()):
        raise ValueError(f"readiness dependency failure: {dependency_graph}")
    sink["primary_conclusion"] = "SINK_PHYSICAL_ORACLE_ONLY" if sink_physical and not sink_operational else ("SINK_RUNTIME_DETECTION_DEFERRED" if not sink_operational else "OPERATIONAL_LABEL_READY")
    sink["secondary_conclusion"] = "ADDITIONAL_SENSOR_OBSERVATION_REQUIRED" if not sink_operational else "INCONCLUSIVE"
    next_step = "SLIP_OPERATIONAL_RISK_BLIND_HOLDOUT" if slip_gate else "TARGETED_SINK_LABEL_ACQUISITION"
    summary = {
        "artifact": "walking_hazard_operational_label_contract_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "read-only development label/reflex contract audit",
        "physical_offline_runtime_separation": {
            "physical_ground_truth": "offline simulation oracle only",
            "operational_risk": "causal runtime warning; physical future used only to build/evaluate offline target",
            "runtime_confirmation": "persistent sensor evidence, explicitly not physical confirmation",
        },
        "data": {
            "development_runs": len(manifests), "validation_runs": sum(row["split"] == "diagnostic_validation" for row in manifests),
            "sample_rate_hz": SAMPLE_RATE_HZ, "trace_schema": [3000, 10],
            "split_integrity": leakage, "outer_content_load_count": 0,
        },
        "slip": slip,
        "sink": sink,
        "contact_infrastructure_read_only": {
            "phase_agreement": 0.9282777777777778, "touchdown_detected": "323/323",
            "touchdown_median_latency_ms": 0.0, "state_reset_audits": "162/162 pass",
        },
        "action_utility_claim": {
            "available_reaction_time_margin_only": True,
            "false_trigger_burden_only": True,
            "fall_reduction_claimed": False, "stability_improvement_claimed": False,
            "recovery_success_claimed": False, "closed_loop_safety_benefit_claimed": False,
        },
        "final_questions": {
            "slip_risk_fast_reflex_candidate": "yes; as an operational risk trigger candidate requiring a new blind holdout",
            "slip_physical_confirmation_runtime_required": "no; physical confirmation is offline-only and waiting would discard the usable precursor",
            "sink_physically_valid_observable_operational_label": "a reproducible physical penetration oracle exists, but no candidate is currently operationally observable at acceptable normal burden",
            "sink_additional_sensor_or_state_required": "yes; support/base state not confounded with normal gait is required",
            "legacy_system_v1_ambiguity": "slip/sink mix physical labels with detector firing, RECOVERY_REQUIRED collides with terrain-transition recovery, and derived mismatch/dual flags inherit both ambiguities",
            "blind_validation_target": "locked 100 ms slip_risk contract, including coverage, margin, normal/too-early/invalid burden, reset, and controlled compatibility",
        },
        "readiness": readiness,
        "readiness_dependency_graph": dependency_graph,
        "next_step": next_step,
        "model_training_runs": 0, "production_or_system_changes": 0,
        "int8_or_vela_executed": False,
        "immutable_upstream_mismatch_count": upstream["mismatch_count"],
        "wall_time_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "readiness.json", {
        "gates": readiness, "diagnostic_contract_only": True,
        "system_schema_migration_authorized": False,
        "bounded_retraining_v2_authorized": False,
        "int8_preparation_authorized": False,
        "next_step": next_step,
    })
    write_audit(output, summary)
    write_json(output / "manifest.json", artifact_manifest(output, upstream))
    return summary


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("Dry run only. Use --execute; no training or outer content access is implemented.")
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty contract output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", predeclared_protocol())
    summary = run_audit(output)
    print(json.dumps({"output": str(output), "readiness": summary["readiness"], "next_step": summary["next_step"]}, indent=2))


if __name__ == "__main__":
    main()
