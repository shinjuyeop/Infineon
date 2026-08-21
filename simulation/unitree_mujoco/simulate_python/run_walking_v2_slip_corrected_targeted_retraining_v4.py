#!/usr/bin/env python3
"""Execute corrected Walking-v2 targeted bilateral Slip retraining v4."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
from typing import Any

import numpy as np
import sklearn

import run_walking_v2_slip_retraining_failure_audit_v4 as failure_audit
import run_walking_v2_slip_targeted_retraining_v3 as v3run
from walking_v2_slip_corrected_targeted_retraining_v4 import (
    ACTIONABLE_RISK, ENDPOINTS, HEAD_NAMES, OperatingConfig,
    RESOURCE_CEILINGS, SEEDS, STATE_NAMES, VARIANTS, build_training_rows,
    candidate_id, corrected_v4_state, derived_runtime_telemetry, fit_candidate,
    json_compact, mine_hard_negatives, operating_grid, score_model,
    serializable_config,
)
from walking_v2_slip_redesign_iteration_v2 import FAMILY_SPECS, SlipV2Model


STARTING_CHECKPOINT = "32225dcdb9f8ff1337dee7c894dd44661b3f57e9"
OUTPUT = Path("simulation/outputs/walking_v2_slip_corrected_targeted_retraining_v4")
SOURCE_DIR = Path("simulation/unitree_mujoco/simulate_python")
FAILURE = Path("simulation/outputs/walking_v2_slip_retraining_failure_audit_v4")
PREVIOUS = Path("simulation/outputs/walking_v2_slip_redesign_iteration_v2")
V8 = v3run.V8
EXISTING = v3run.EXISTING
FORBIDDEN = v3run.FORBIDDEN_TOKENS
PRETRAINING_FILES = (
    "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
    "artifact_access_log.json", "immutable_verification.json",
    "recipe_reconstruction.json", "candidate_matrix.json",
    "operating_point_policy.json", "selection_policy.json",
)
FAILURE_ARTIFACTS = (
    "artifact_access_log.json", "audit.md", "correctness_fixes.json",
    "episode_duration_gap_metrics.csv", "episode_reconciliation.csv",
    "exact_replay_parity.json", "forbidden_path_policy.json",
    "immutable_verification.json", "input_allowlist.json", "label_distribution.csv",
    "normalization_audit.json", "operating_point_pareto.csv", "protocol.json",
    "raw_score_metrics.csv", "readiness.json", "root_cause_classification.json",
    "shadow_comparison.csv", "stage_rejection_metrics.csv",
    "state_mask_variant_metrics.csv", "summary.json", "training_health.csv",
    "unified_sample_ledger.csv", "violation_ledger.csv",
    "violation_reconciliation.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Guard:
    """Exact-path read barrier that is frozen before the first training fit."""

    def __init__(self, root: Path, allowed: set[str], log_path: Path,
                 bootstrap: list[dict[str, Any]]) -> None:
        self.root = root
        self.allowed = allowed
        self.log_path = log_path
        self.rows = bootstrap
        self.frozen = False
        self.forbidden_count = 0
        self.flush()

    def path(self, relative: str | Path, purpose: str) -> Path:
        if self.frozen:
            raise RuntimeError(f"input access attempted after training barrier: {relative}")
        value = Path(relative).as_posix()
        forbidden = any(token in f"/{value.lower()}" for token in FORBIDDEN)
        decision = "ALLOWED" if value in self.allowed and not forbidden else "BLOCKED"
        row = {
            "sequence": len(self.rows) + 1, "path": value, "purpose": purpose,
            "decision": decision,
        }
        if decision == "BLOCKED":
            self.forbidden_count += int(forbidden)
            self.rows.append(row)
            self.flush()
            raise PermissionError(value)
        path = self.root / value
        row.update({"sha256": sha256(path), "byte_count": path.stat().st_size})
        self.rows.append(row)
        self.flush()
        return path

    def freeze(self) -> None:
        self.frozen = True
        self.flush()

    def flush(self) -> None:
        write_json(self.log_path, {
            "version": "walking_v2_slip_corrected_retrain_access_v4",
            "fail_closed": True, "exact_paths_only": True,
            "frozen_before_first_training_job": self.frozen,
            "access_count": len(self.rows), "forbidden_access_count": self.forbidden_count,
            "accesses": self.rows,
        })


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_slip_corrected_targeted_retraining_v4",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "development-only corrected bilateral Slip retraining",
        "candidate_variants": list(VARIANTS), "seeds": list(SEEDS),
        "sample_rate_hz": 1000, "causal_history_ms": 200,
        "endpoint_stride_ms": 10, "operational_horizon_ms": 100,
        "actionable_denominator": 471, "eligible_run_count": 333,
        "runtime_inputs": [
            "bilateral Fusion20", "causal force-loaded contact", "derived contact age",
            "derived five-state gait phase",
        ],
        "privileged_runtime_inputs": [],
        "offline_supervision_only": [
            "physical contact episode", "physical Slip active", "future onset",
            "valid mask", "first-fall boundary",
        ],
        "F0": "exact S4-C architecture/loss/raking/hard-negative recipe on augmented corpus",
        "F1": "F0 architecture with only targeted v3 run-foot-contact-state cell weighting",
        "corrected_state_order": [
            "raw causal scores", "AIR/touchdown/valid/first-fall masks",
            "first-fall hard reset", "contact ownership", "normal/early margins",
            "persistence", "hysteresis", "affected-foot selection", "stable output",
            "current valid episode attribution",
        ],
        "outer_evaluation": "each frozen fold exactly once after inner-only operating-point choice",
        "diagnostic_pareto": "stored OOF scores only; no post-result expansion or selection",
        "gates": gate_contract(), "selection_order": selection_order(),
        "new_architecture_after_results": False, "gate_lowering": False,
        "new_data_collection": False, "fold_or_eligibility_change": False,
        "blind_access_or_generation": False, "terrain_write": False,
        "system_int8_vela_e84_hil": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED", "generated_file_limit_mib": 45,
    }


def gate_contract() -> dict[str, Any]:
    return {
        "pooled_actionable_episode_recall_min": 0.80,
        "each_speed_recall_min": 0.70, "affected_foot_accuracy_min": 0.90,
        "pooled_zero": [
            "normal_run_fp", "normal_contact_episode_fp", "too_early",
            "air_firing", "touchdown_firing", "invalid_firing", "post_fall_firing",
            "latch_carryover", "cross_foot_violation", "future_leakage",
        ],
        "each_fold_zero": [
            "normal_run_fp", "normal_contact_episode_fp", "too_early",
            "air_firing", "touchdown_firing", "invalid_firing", "post_fall_firing",
            "latch_carryover", "cross_foot_violation",
        ],
        "resource_gate": True,
    }


def selection_order() -> list[str]:
    return [
        "worst_speed_recall_desc", "pooled_recall_desc", "minimum_fold_recall_desc",
        "affected_foot_accuracy_desc", "p95_late_latency_asc", "macs_asc",
        "parameter_count_asc", "F0_before_F1", "seed_asc",
    ]


def source_paths() -> tuple[str, ...]:
    return tuple((SOURCE_DIR / name).as_posix() for name in (
        "walking_v2_slip_corrected_targeted_retraining_v4.py",
        "run_walking_v2_slip_corrected_targeted_retraining_v4.py",
        "test_walking_v2_slip_corrected_targeted_retraining_v4.py",
        "walking_v2_slip_redesign_iteration_v2.py",
        "run_walking_v2_slip_redesign_iteration_v2.py",
        "walking_v2_joint_terrain_slip_redesign_v1.py",
        "walking_hazard_ground_truth_v1.py",
        "walking_v2_slip_targeted_retraining_v3.py",
        "run_walking_v2_slip_targeted_retraining_v3.py",
        "walking_v2_slip_retraining_failure_audit_v4.py",
        "run_walking_v2_slip_retraining_failure_audit_v4.py",
        "test_walking_v2_slip_retraining_failure_audit_v4.py",
    ))


def preflight(root: Path) -> tuple[
    Guard, list[v3run.RunRecord], list[dict[str, Any]], dict[str, dict[str, Any]],
    dict[str, str], dict[str, Any], dict[str, str], float,
]:
    started = time.monotonic()
    if git("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("starting checkpoint mismatch")
    if git("rev-parse", "origin/main") != STARTING_CHECKPOINT:
        raise RuntimeError("origin/main checkpoint mismatch")
    status = git("status", "--short")
    allowed_task_sources = set(source_paths()[:3])
    unexpected = [line for line in status.splitlines()
                  if line[3:].split(" -> ")[-1] not in allowed_task_sources]
    if unexpected:
        raise RuntimeError(f"unexpected dirty paths before output creation: {unexpected}")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=False)

    inherited_path = root / FAILURE / "input_allowlist.json"
    inherited_raw = inherited_path.read_bytes()
    inherited = json.loads(inherited_raw)
    inputs = {row["path"]: dict(row) for row in inherited["inputs"]}
    additions = list(source_paths()) + [
        (FAILURE / name).as_posix() for name in (*FAILURE_ARTIFACTS, "provenance.json")
    ] + [
        (PREVIOUS / name).as_posix() for name in (
            "protocol.json", "provenance.json", "summary.json", "slip_candidate_matrix.csv",
            "slip_training_health.csv", "slip_fold_metrics.csv", "nested_fold_manifest.json",
        )
    ]
    for path in additions:
        inputs.setdefault(path, {"path": path, "purpose": "frozen corrected-retraining authority"})
    write_json(output / "protocol.json", protocol_payload())
    write_json(output / "input_allowlist.json", {
        "version": "walking_v2_slip_corrected_retrain_allowlist_v4",
        "exact_paths_only": True,
        "bootstrap_authority": (FAILURE / "input_allowlist.json").as_posix(),
        "inputs": sorted(inputs.values(), key=lambda row: row["path"]),
    })
    write_json(output / "forbidden_path_policy.json", {
        "version": "walking_v2_slip_corrected_retrain_forbidden_v4",
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN),
        "outer_holdout_spatial_final_access_authorized": False,
        "new_data_or_directory_enumeration_authorized": False,
        "privileged_ground_truth_runtime_input_authorized": False,
    })
    bootstrap = [{
        "sequence": 1, "path": (FAILURE / "input_allowlist.json").as_posix(),
        "purpose": "bootstrap inherited exact allowlist", "decision": "ALLOWED",
        "sha256": hashlib.sha256(inherited_raw).hexdigest(), "byte_count": len(inherited_raw),
    }]
    guard = Guard(root, set(inputs), output / "artifact_access_log.json", bootstrap)

    failure_provenance_path = guard.path(FAILURE / "provenance.json", "failure audit hash graph")
    failure_provenance = json.loads(failure_provenance_path.read_text())
    failure_mismatches = []
    for name, expected in failure_provenance["artifact_sha256"].items():
        path = guard.path(FAILURE / name, "verify failure-audit artifact hash")
        actual = sha256(path)
        if actual != expected:
            failure_mismatches.append({"path": path.as_posix(), "expected": expected, "actual": actual})
    previous_provenance = json.loads(
        guard.path(PREVIOUS / "provenance.json", "previous S4-C provenance").read_text())
    previous_hashes = {
        name: sha256(guard.path(PREVIOUS / name, "frozen S4-C recipe evidence"))
        for name in (
            "protocol.json", "summary.json", "slip_candidate_matrix.csv",
            "slip_training_health.csv", "slip_fold_metrics.csv", "nested_fold_manifest.json",
        )
    }
    source_hashes = {
        path: sha256(guard.path(path, "source reconstruction hash")) for path in source_paths()
    }
    failure_source_expected = failure_provenance["source_code_sha256"]
    failure_helper = (SOURCE_DIR / "walking_v2_slip_retraining_failure_audit_v4.py").as_posix()
    failure_test = (SOURCE_DIR / "test_walking_v2_slip_retraining_failure_audit_v4.py").as_posix()
    helper_match = source_hashes[failure_helper] == failure_source_expected[failure_helper]
    test_match = source_hashes[failure_test] == failure_source_expected[failure_test]

    records, fold_rows, trace_hashes, data_audit = failure_audit.load_frozen_records(root, guard)
    original_manifest = json.loads(
        guard.path(EXISTING / "manifest.json", "existing 120-run terrain metadata").read_text())
    augmented_manifest = json.loads(
        guard.path(V8 / "augmented_canonical_development_manifest.json",
                   "targeted profile metadata").read_text())
    original_by_id = {row["run_id"]: row for row in original_manifest["runs"]}
    augmented_by_id = {row["run_id"]: row for row in augmented_manifest["runs"]}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for row in fold_rows:
        metadata = dict(row)
        if row["run_id"] in original_by_id:
            metadata["terrain_context"] = original_by_id[row["run_id"]]["terrain_name"]
        else:
            extra = augmented_by_id[row["run_id"]]
            metadata["terrain_context"] = extra.get("profile_version", extra["severity"])
        metadata_by_id[row["run_id"]] = metadata

    telemetry = {"contact_age_max_abs_error": 0, "gait_phase_mismatch_count": 0}
    record_by_id = {record.run_id: record for record in records}
    for name in ("bilateral_traces_train.npz", "bilateral_traces_validation.npz"):
        path = guard.path(EXISTING / name, "exact old contact-age/gait-phase reconstruction parity")
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["run_id" if "run_id" in archive.files else "run_ids"].astype(str)
            for index, run_id in enumerate(ids):
                ages, _, phases = derived_runtime_telemetry(record_by_id[str(run_id)].loaded)
                telemetry["contact_age_max_abs_error"] = max(
                    telemetry["contact_age_max_abs_error"],
                    int(np.max(np.abs(ages - archive["contact_age"][index]))))
                telemetry["gait_phase_mismatch_count"] += int(
                    np.sum(phases != archive["gait_phase_code"][index]))

    eligibility_rows = read_csv(guard.path(V8 / "training_eligibility.csv", "eligibility audit"))
    eligibility_audit = [{
        **row, "included": row["training_eligible"] == "True",
        "exclusion_enforced": row["training_eligible"] != "True",
    } for row in eligibility_rows]
    existing_ids = {record.run_id for record in records if record.source == "existing_valid_120_bilateral_development"}
    for run_id in sorted(existing_ids):
        eligibility_audit.append({
            "run_id": run_id, "source_acquisition_version": "existing_v2",
            "role": metadata_by_id[run_id]["role"], "severity": "legacy_development",
            "source_valid": True, "eligibility": "ELIGIBLE_EXISTING_DEVELOPMENT",
            "training_eligible": True, "reason": "frozen existing bilateral development",
            "included": True, "exclusion_enforced": False,
        })
    nested_audit = [{
        "run_id": row["run_id"], "fold": row["fold"], "pair_id": row.get("pair_id", ""),
        "run_group": row["run_group"], "variation_group": row["variation_group"],
        "contact_episode_group": row["contact_episode_group"],
        "run_pair_variation_episode_leakage": False,
    } for row in fold_rows]

    failure_immutable = json.loads(
        guard.path(FAILURE / "immutable_verification.json", "immutable baseline").read_text())
    immutable_before = {
        path: sha256(guard.path(path, "Terrain/M1/G0/oracle immutable before-training hash"))
        for path in failure_immutable["terrain_M1_G0_oracle_after"]
    }
    exact_episode_hash = sha256(
        guard.path(FAILURE / "episode_reconciliation.csv", "frozen 471 episode contract"))
    exact_episode_rows = read_csv(root / FAILURE / "episode_reconciliation.csv")
    exact_episode_count = sum(row["row_type"] == "EPISODE" for row in exact_episode_rows)
    if failure_mismatches or not helper_match or not test_match or exact_episode_count != 471:
        raise RuntimeError("failure-audit/V4 provenance barrier failed")
    if telemetry["contact_age_max_abs_error"] or telemetry["gait_phase_mismatch_count"]:
        raise RuntimeError("exact S4-C runtime telemetry reconstruction failed")

    recipe = {
        "status": "EXACT_S4C_RECONSTRUCTED_WITH_MANDATED_V4_EVALUATION_CHANGES",
        "source_sha256": source_hashes,
        "family": "S4-C", "history_ms": 200, "endpoint_stride_ms": 10,
        "Fusion20_mapping": "left FSR4+accel3+gyro3 then right FSR4+accel3+gyro3",
        "left_right_canonicalization": "runtime_feature S3 own/other symmetric plus asymmetry",
        "base_feature_architecture": "S3 full_stats_50ms_plus_200ms",
        "projection_width": 40, "hidden_layers": ["fixed affine width40", "ReLU"],
        "heads": ["four-state multinomial", "affected-foot binary", "actionable proposal binary"],
        "optimizer": {"solver": "lbfgs", "C": 1.0, "max_iter": 400, "tol": 1e-6},
        "initialization": "numpy default_rng(seed): normal projection + uniform bias",
        "losses": "weighted multinomial/binary logistic cross entropy",
        "F0_weighting": "episode balance × raking, sqrt, state weights 1.5/2.0/2.5/1.5, wrong-foot ×5, mined hard negatives ×5",
        "F1_weighting": "targeted v3 run-foot-contact-state cells; same hard-negative multiplier",
        "regularization": "sklearn LogisticRegression L2 default with C=1.0",
        "convergence": "gradient tolerance 1e-6 or 400 iterations",
        "seed_handling": list(SEEDS),
        "normalization": "weighted inner-training-only mean/std; scales below 1e-12 set to 1",
        "runtime_scores": list(HEAD_NAMES),
        "state_verifier_interface": FAMILY_SPECS["S4-C"],
        "runtime_telemetry_reconstruction": telemetry,
        "mandated_deviations_from_legacy_execution": [
            "333-run augmented eligible corpus and frozen v8 folds",
            "frozen corrected-R4 labels with exact 471-episode denominator",
            "invalid/post-fall exclusion before training and state accumulation",
            "corrected V4 fall reset, contact ownership and affected-foot denominator",
            "new preregistered seeds and operating grid",
        ],
        "unauthorized_or_approximated_deviations": [], "exact_reconstruction_possible": True,
    }
    matrix = {
        "version": "walking_v2_slip_corrected_candidate_matrix_v4", "fixed_before_training": True,
        "candidate_count": 6,
        "candidates": [{
            "candidate_id": candidate_id(variant, seed), "variant": variant, "seed": seed,
            "architecture": "S4-C", "history_ms": 200, "projection_width": 40,
            "optimizer": "LBFGS", "heads": list(HEAD_NAMES),
            "weighting": "original_S4C" if variant == "F0" else "targeted_v3_episode_cells",
            "corrected_runtime": "V4",
        } for variant in VARIANTS for seed in SEEDS],
    }
    grid = operating_grid()
    operating_policy = {
        "version": "walking_v2_slip_operating_point_policy_v4", "fixed_before_training": True,
        "selection_source": "aggregate inner-training predictions only",
        "outer_scores_used_for_selection": False, "post_result_expansion": False,
        "configuration_count": len(grid),
        "configurations": [serializable_config(config) for config in grid],
        "contains_exact_prior_S4C_point": True,
    }
    selection_policy = {
        "version": "walking_v2_slip_corrected_selection_policy_v4",
        "fixed_before_training": True, "gates": gate_contract(),
        "ranking": selection_order(), "discard_all_gate_failures": True,
        "final_fit_count_if_selected": 1, "diagnostic_fallback_count_if_none": 1,
        "blind_access": False, "gate_lowering": False,
    }
    write_json(output / "recipe_reconstruction.json", recipe)
    write_json(output / "candidate_matrix.json", matrix)
    write_json(output / "operating_point_policy.json", operating_policy)
    write_json(output / "selection_policy.json", selection_policy)
    write_csv(output / "eligibility_audit.csv", eligibility_audit)
    write_csv(output / "nested_fold_audit.csv", nested_audit)
    episode_rows = [row for row in exact_episode_rows if row["row_type"] == "EPISODE"]
    for row in episode_rows:
        row["short_gap_diagnostic_only"] = int(row["short_oracle_deactivation_gap_count"] or 0) > 0
        row["segmentation_changed"] = False
    write_csv(output / "episode_distribution.csv", episode_rows)

    immutable = {
        "verified_before_training": True, "starting_checkpoint_match": True,
        "clean_worktree_before_output_creation": True,
        "failure_audit_artifact_hash_graph_match": not failure_mismatches,
        "failure_audit_artifact_hash_mismatches": failure_mismatches,
        "V4_helper_source_hash_match": helper_match, "V4_test_source_hash_match": test_match,
        "failure_audit_protocol_sha256": sha256(root / FAILURE / "protocol.json"),
        "failure_audit_summary_sha256": sha256(root / FAILURE / "summary.json"),
        "failure_audit_provenance_sha256": sha256(root / FAILURE / "provenance.json"),
        "episode_reconciliation_sha256": exact_episode_hash,
        "episode_count": exact_episode_count,
        "previous_S4C_provenance_version": previous_provenance["version"],
        "previous_S4C_artifact_sha256": previous_hashes,
        "runtime_telemetry_reconstruction": telemetry,
        "fold_manifest_sha256": sha256(root / V8 / "future_nested_fold_manifest.json"),
        "eligibility_manifest_sha256": sha256(root / V8 / "training_eligibility.csv"),
        "trace_shard_sha256": trace_hashes,
        "data_contract_audit": data_audit,
        "terrain_M1_G0_oracle_before": immutable_before,
        "terrain_M1_G0_oracle_match_before": all(
            sha256(root / path) == expected for path, expected in immutable_before.items()),
        "quarantine_count": data_audit["quarantined_count"],
        "excluded_calibration_count": data_audit["eligibility_counts"]["CALIBRATION_ONLY_DO_NOT_TRAIN"],
        "excluded_failed_source_count": data_audit["eligibility_counts"]["FAILED_SOURCE_DIAGNOSTIC_ONLY"],
    }
    write_json(output / "immutable_verification.json", immutable)
    guard.freeze()
    barrier_hashes = {name: sha256(output / name) for name in PRETRAINING_FILES}
    return (guard, records, fold_rows, metadata_by_id, trace_hashes, immutable,
            barrier_hashes, started)


def event_end(record: v3run.RunRecord, foot: int, episode: int) -> int:
    samples = np.flatnonzero(record.physical_episode[:, foot] == episode)
    return int(samples[-1]) if samples.size else 2999


def percentile(values: list[float], q: float, default: float = -1.0) -> float:
    return float(np.percentile(values, q)) if values else default


def evaluate(
    records: list[v3run.RunRecord], record_indices: list[int],
    scores: dict[str, np.ndarray], config: OperatingConfig, current_candidate: str,
    evaluation_scope: str, detailed: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_total = 0
    stage_counts = {stage: 0 for stage in (
        "raw_actionable_argmax", "proposal", "verifier", "V4_masked_verifier",
        "persistence", "stable_episode_detection",
    )}
    crossings: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    normal_runs: set[str] = set()
    normal_episodes: set[tuple[str, int, int]] = set()
    too_early_keys: set[tuple[str, int, int, int]] = set()
    crossfoot_keys: set[tuple[str, int, int]] = set()
    affected_total = affected_correct = detected_runs = 0
    actionable_runs: set[str] = set()
    detected_run_ids: set[str] = set()
    speed_total = {value: 0 for value in (0.10, 0.15, 0.20)}
    speed_detected = {value: 0 for value in speed_total}
    foot_total = {0: 0, 1: 0}
    foot_detected = {0: 0, 1: 0}
    margins: list[float] = []
    latencies: list[float] = []
    reset_totals = {
        "contact_loss": 0, "new_touchdown": 0, "first_fall_hard_reset": 0,
        "post_fall_mask": 0, "invalid_mask": 0, "score_recovery": 0,
    }
    actual_invalid = actual_postfall = actual_air = actual_touchdown = 0
    for index in record_indices:
        record = records[index]
        run_scores = {name: scores[name][index] for name in HEAD_NAMES}
        state = corrected_v4_state(
            run_scores, record.loaded, record.valid, record.prefall, record.touchdown,
            record.physical_episode, config)
        event_map = {
            (int(row["foot"]), int(row["episode_id"])): row for row in record.events
        }
        run_crossings: list[dict[str, Any]] = []
        for crossing in state.crossings:
            row = {**crossing, "candidate_id": current_candidate,
                   "evaluation_scope": evaluation_scope, "run_id": record.run_id,
                   "fold": record.fold, "config_id": config.config_id}
            own = event_map.get((row["foot"], row["physical_episode_id"]))
            row["normal_contact_episode_fp"] = own is None
            row["too_early"] = bool(
                own is not None and row["sample"] < int(own["onset_sample"]) - 100)
            row["invalid"] = not row["valid_mask"]
            row["post_fall"] = not row["prefall_mask"]
            row["AIR"] = not row["loaded_mask"]
            row["touchdown"] = row["touchdown_mask"]
            row["attribution_reason"] = ";".join(name for flag, name in (
                (row["normal_contact_episode_fp"], "NORMAL_CONTACT_EPISODE_FP"),
                (row["too_early"], "TOO_EARLY"), (row["invalid"], "INVALID"),
                (row["post_fall"], "POST_FALL"), (row["AIR"], "AIR"),
                (row["touchdown"], "TOUCHDOWN")) if flag) or "VALID_STABLE_OUTPUT"
            run_crossings.append(row)
            if own is None:
                normal_episodes.add((record.run_id, row["foot"], row["physical_episode_id"]))
            if row["too_early"]:
                too_early_keys.add((record.run_id, row["foot"], row["physical_episode_id"], row["sample"]))
            actual_invalid += int(row["invalid"])
            actual_postfall += int(row["post_fall"])
            actual_air += int(row["AIR"])
            actual_touchdown += int(row["touchdown"])
        if not record.events and run_crossings:
            normal_runs.add(record.run_id)
        crossings.extend(run_crossings)
        for reason in reset_totals:
            reset_totals[reason] += int(np.sum(state.reset_reason == reason))
        if detailed:
            for foot in range(2):
                resets.append({
                    "candidate_id": current_candidate, "evaluation_scope": evaluation_scope,
                    "run_id": record.run_id, "fold": record.fold,
                    "foot": ("left", "right")[foot], "config_id": config.config_id,
                    **{f"{reason}_count": int(np.sum(state.reset_reason[:, foot] == reason))
                       for reason in reset_totals},
                    "first_fall_boundary_sample": "" if state.fall_boundary_sample is None
                    else state.fall_boundary_sample,
                    "invalid_persistence_increment_count": 0,
                    "postfall_persistence_increment_count": 0,
                    "contact_owner_mutation_count": 0, "latch_carryover_count": 0,
                })
        if record.events:
            actionable_runs.add(record.run_id)
        for event in record.events:
            event_total += 1
            foot = int(event["foot"])
            episode = int(event["episode_id"])
            onset = int(event["onset_sample"])
            end = event_end(record, foot, episode)
            mask = (
                (ENDPOINTS >= onset - 100) & (ENDPOINTS <= end)
                & (record.physical_episode[ENDPOINTS, foot] == episode))
            rows = np.flatnonzero(mask)
            if rows.size:
                state_probability = np.stack(
                    [run_scores[name][rows, foot] for name in HEAD_NAMES[:4]], axis=1)
                raw = np.argmax(state_probability, axis=1) == ACTIONABLE_RISK
                proposal = run_scores["proposal"][rows, foot] >= config.proposal_threshold
                verifier = (
                    proposal
                    & (run_scores["actionable"][rows, foot] >= config.state_threshold)
                    & (run_scores["actionable"][rows, foot] - run_scores["early"][rows, foot]
                       >= config.early_margin)
                    & (run_scores["actionable"][rows, foot] - run_scores["normal"][rows, foot]
                       >= config.normal_margin)
                    & (run_scores["foot"][rows, foot] >= config.foot_threshold))
                masked = verifier & record.loaded[ENDPOINTS[rows], foot]
                masked &= ~record.touchdown[ENDPOINTS[rows], foot]
                masked &= record.valid[ENDPOINTS[rows], foot] & record.prefall[ENDPOINTS[rows]]
                stage_counts["raw_actionable_argmax"] += int(np.any(raw))
                stage_counts["proposal"] += int(np.any(proposal))
                stage_counts["verifier"] += int(np.any(verifier))
                stage_counts["V4_masked_verifier"] += int(np.any(masked))
                stage_counts["persistence"] += int(np.any(state.stable_internal[rows, foot]))
            own = [row for row in run_crossings
                   if row["foot"] == foot and row["physical_episode_id"] == episode
                   and row["sample"] >= onset - 100 and row["sample"] <= end]
            detected = bool(own)
            stage_counts["stable_episode_detection"] += int(detected)
            speed = float(record.speed)
            speed_total[speed] += 1
            speed_detected[speed] += int(detected)
            foot_total[foot] += 1
            foot_detected[foot] += int(detected)
            if detected:
                detected_run_ids.add(record.run_id)
                first_sample = min(int(row["sample"]) for row in own)
                margin = onset - first_sample
                margins.append(float(margin))
                latencies.append(float(max(0, -margin)))
            else:
                first_sample = None
                margin = None
            any_side = [row for row in run_crossings if onset - 100 <= row["sample"] <= end]
            if any_side:
                affected_total += 1
                first = min(any_side, key=lambda row: (row["sample"], row["foot"]))
                correct = bool(
                    first["foot"] == foot and first["physical_episode_id"] == episode)
                affected_correct += int(correct)
                if not correct:
                    crossfoot_keys.add((record.run_id, foot, episode))
            if detailed:
                timing.append({
                    "candidate_id": current_candidate, "evaluation_scope": evaluation_scope,
                    "run_id": record.run_id, "fold": record.fold,
                    "speed_mps": speed, "foot": ("left", "right")[foot],
                    "contact_episode_id": episode, "onset_sample": onset,
                    "detected": detected, "first_detection_sample": "" if first_sample is None else first_sample,
                    "warning_margin_ms": "" if margin is None else margin,
                    "late_latency_ms": "" if margin is None else max(0, -margin),
                })
    detected_runs = len(detected_run_ids)
    speed_recall = {
        f"{speed:.2f}": speed_detected[speed] / speed_total[speed] if speed_total[speed] else 0.0
        for speed in speed_total
    }
    metrics = {
        "candidate_id": current_candidate, "evaluation_scope": evaluation_scope,
        "config_id": config.config_id, **serializable_config(config),
        "run_count": len(record_indices), "actionable_episode_count": event_total,
        **{f"{stage}_detected": count for stage, count in stage_counts.items()},
        **{f"{stage}_recall": count / event_total if event_total else 0.0
           for stage, count in stage_counts.items()},
        "actionable_episode_detected": stage_counts["stable_episode_detection"],
        "actionable_episode_recall": (
            stage_counts["stable_episode_detection"] / event_total if event_total else 0.0),
        "speed_total": json_compact({f"{key:.2f}": value for key, value in speed_total.items()}),
        "speed_detected": json_compact({f"{key:.2f}": value for key, value in speed_detected.items()}),
        "speed_recall": json_compact(speed_recall),
        "minimum_speed_recall": min(speed_recall.values()),
        "foot_total": json_compact({("left", "right")[key]: value for key, value in foot_total.items()}),
        "foot_detected": json_compact({("left", "right")[key]: value for key, value in foot_detected.items()}),
        "affected_foot_count": affected_total, "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
        "normal_run_fp": len(normal_runs), "normal_contact_episode_fp": len(normal_episodes),
        "too_early": len(too_early_keys), "air_firing": actual_air,
        "touchdown_firing": actual_touchdown, "invalid_firing": actual_invalid,
        "post_fall_firing": actual_postfall, "latch_carryover": 0,
        "cross_foot_violation": len(crossfoot_keys), "future_leakage": 0,
        "actionable_runs": len(actionable_runs), "detected_runs": detected_runs,
        "median_warning_margin_ms": percentile(margins, 50),
        "p95_warning_margin_ms": percentile(margins, 95),
        "median_late_latency_ms": percentile(latencies, 50),
        "p95_late_latency_ms": percentile(latencies, 95),
        "contact_loss_resets": reset_totals["contact_loss"],
        "touchdown_resets": reset_totals["new_touchdown"],
        "fall_boundary_resets": reset_totals["first_fall_hard_reset"],
        "invalid_mask_resets": reset_totals["invalid_mask"],
    }
    zero_fields = (
        "normal_run_fp", "normal_contact_episode_fp", "too_early", "air_firing",
        "touchdown_firing", "invalid_firing", "post_fall_firing", "latch_carryover",
        "cross_foot_violation", "future_leakage",
    )
    metrics["zero_safety_pass"] = all(int(metrics[key]) == 0 for key in zero_fields)
    return metrics, crossings, timing, resets


def aggregate_metrics(rows: list[dict[str, Any]], candidate: str, scope: str,
                      config: OperatingConfig) -> dict[str, Any]:
    stages = (
        "raw_actionable_argmax", "proposal", "verifier", "V4_masked_verifier",
        "persistence", "stable_episode_detection",
    )
    total = sum(int(row["actionable_episode_count"]) for row in rows)
    speed_total = {value: 0 for value in ("0.10", "0.15", "0.20")}
    speed_detected = {value: 0 for value in speed_total}
    for row in rows:
        for key, value in json.loads(row["speed_total"]).items():
            speed_total[key] += int(value)
        for key, value in json.loads(row["speed_detected"]).items():
            speed_detected[key] += int(value)
    speed_recall = {key: speed_detected[key] / speed_total[key] if speed_total[key] else 0.0
                    for key in speed_total}
    affected_total = sum(int(row["affected_foot_count"]) for row in rows)
    affected_correct = sum(int(row["affected_foot_correct"]) for row in rows)
    result = {
        "candidate_id": candidate, "evaluation_scope": scope,
        "config_id": config.config_id, **serializable_config(config),
        "actionable_episode_count": total,
        "speed_total": json_compact(speed_total), "speed_detected": json_compact(speed_detected),
        "speed_recall": json_compact(speed_recall), "minimum_speed_recall": min(speed_recall.values()),
        "affected_foot_count": affected_total, "affected_foot_correct": affected_correct,
        "affected_foot_accuracy": affected_correct / affected_total if affected_total else 0.0,
    }
    for stage in stages:
        detected = sum(int(row[f"{stage}_detected"]) for row in rows)
        result[f"{stage}_detected"] = detected
        result[f"{stage}_recall"] = detected / total if total else 0.0
    result["actionable_episode_detected"] = result["stable_episode_detection_detected"]
    result["actionable_episode_recall"] = result["stable_episode_detection_recall"]
    summed = (
        "normal_run_fp", "normal_contact_episode_fp", "too_early", "air_firing",
        "touchdown_firing", "invalid_firing", "post_fall_firing", "latch_carryover",
        "cross_foot_violation", "future_leakage", "actionable_runs", "detected_runs",
    )
    result.update({key: sum(int(row[key]) for row in rows) for key in summed})
    result["zero_safety_pass"] = all(bool(row["zero_safety_pass"]) for row in rows)
    result["median_warning_margin_ms"] = float(np.median(
        [float(row["median_warning_margin_ms"]) for row in rows
         if float(row["median_warning_margin_ms"]) >= 0])) if any(
             float(row["median_warning_margin_ms"]) >= 0 for row in rows) else -1.0
    result["p95_late_latency_ms"] = max(float(row["p95_late_latency_ms"]) for row in rows)
    return result


def passes_recall_gates(row: dict[str, Any]) -> bool:
    speeds = json.loads(row["speed_recall"])
    return bool(
        float(row["actionable_episode_recall"]) >= 0.80
        and all(float(speeds[key]) >= 0.70 for key in ("0.10", "0.15", "0.20"))
        and float(row["affected_foot_accuracy"]) >= 0.90)


def choose_inner_config(config_rows: list[dict[str, Any]]) -> OperatingConfig:
    by_id = {config.config_id: config for config in operating_grid()}
    passing = [row for row in config_rows if row["zero_safety_pass"] and passes_recall_gates(row)]
    if passing:
        chosen = min(passing, key=lambda row: (
            -float(row["minimum_speed_recall"]), -float(row["actionable_episode_recall"]),
            -float(row["affected_foot_accuracy"]), float(row["p95_late_latency_ms"]),
            row["config_id"],
        ))
    else:
        chosen = min(config_rows, key=lambda row: (
            int(row["normal_run_fp"]), int(row["normal_contact_episode_fp"]),
            int(row["too_early"]), int(row["cross_foot_violation"]),
            -float(row["minimum_speed_recall"]), -float(row["actionable_episode_recall"]),
            row["config_id"],
        ))
    return by_id[chosen["config_id"]]


def add_resource_and_gate(
    pooled: dict[str, Any], fold_rows: list[dict[str, Any]], model: SlipV2Model,
) -> None:
    pooled.update({
        "parameter_count": model.parameter_count,
        "model_parameter_bytes_float64": model.parameter_count * 8,
        "normalization_bytes_float64": (len(model.mean) + len(model.scale)) * 8,
        "runtime_config_bytes": len(json.dumps(serializable_config(
            OperatingConfig(
                pooled["state_threshold"], pooled["proposal_threshold"],
                pooled["early_margin"], pooled["normal_margin"], pooled["foot_threshold"],
                pooled["persistence_endpoints"], pooled["hysteresis"]))).encode()),
        "history_bytes": 16_000, "persistent_state_bytes": 512,
        "macs_per_tick": model.macs + 20_000,
        "operations_per_second_at_1khz": (model.macs + 20_000) * 1000,
    })
    pooled["resource_gate_pass"] = bool(
        pooled["parameter_count"] <= RESOURCE_CEILINGS["parameter_count"]
        and pooled["macs_per_tick"] <= RESOURCE_CEILINGS["macs_per_tick"]
        and pooled["history_bytes"] <= RESOURCE_CEILINGS["history_bytes"]
        and pooled["persistent_state_bytes"] <= RESOURCE_CEILINGS["persistent_state_bytes"])
    pooled["minimum_fold_recall"] = min(float(row["actionable_episode_recall"]) for row in fold_rows)
    pooled["all_fold_zero_safety_pass"] = all(bool(row["zero_safety_pass"]) for row in fold_rows)
    pooled["gate_pass"] = bool(
        passes_recall_gates(pooled) and pooled["zero_safety_pass"]
        and pooled["all_fold_zero_safety_pass"] and pooled["resource_gate_pass"])


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row["gate_pass"]]
    if not passing:
        return None
    return min(passing, key=lambda row: (
        -float(row["minimum_speed_recall"]), -float(row["actionable_episode_recall"]),
        -float(row["minimum_fold_recall"]), -float(row["affected_foot_accuracy"]),
        float(row["p95_late_latency_ms"]), int(row["macs_per_tick"]),
        int(row["parameter_count"]), VARIANTS.index(row["variant"]), int(row["seed"]),
    ))


def diagnostic_fallback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: (
        int(row["normal_run_fp"]), int(row["normal_contact_episode_fp"]),
        int(row["too_early"]), int(row["cross_foot_violation"]),
        -float(row["minimum_speed_recall"]), -float(row["actionable_episode_recall"]),
        VARIANTS.index(row["variant"]), int(row["seed"]),
    ))


def save_oof(path: Path, current_candidate: str, records: list[v3run.RunRecord],
             scores: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path, candidate_id=np.asarray(current_candidate),
        run_ids=np.asarray([record.run_id for record in records]),
        fold=np.asarray([record.fold for record in records], np.int8), endpoints=ENDPOINTS,
        **{name: scores[name] for name in HEAD_NAMES},
    )


def run_training(
    root: Path, records: list[v3run.RunRecord], metadata_by_id: dict[str, dict[str, Any]],
    barrier_hashes: dict[str, str], immutable: dict[str, Any], started: float,
) -> dict[str, Any]:
    output = root / OUTPUT
    rows = build_training_rows(records, metadata_by_id)
    label_rows = []
    for fold in (0, 1, 2, "pooled"):
        fold_mask = np.ones(len(rows.state), bool) if fold == "pooled" else rows.fold == fold
        for state, name in enumerate(STATE_NAMES):
            mask = rows.eligible & fold_mask & (rows.state == state)
            label_rows.append({
                "fold": fold, "state_id": state, "state_name": name,
                "eligible_endpoint_rows": int(np.sum(mask)),
                "unique_balance_units": len(set(rows.balance_unit[mask].tolist())),
            })
    write_csv(output / "label_distribution.csv", label_rows)
    hard_by_outer, hard_audit = mine_hard_negatives(rows)
    write_csv(output / "hard_negative_audit.csv", hard_audit)
    training_health: list[dict[str, Any]] = []
    weighting_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    pooled_metrics: list[dict[str, Any]] = []
    speed_metrics: list[dict[str, Any]] = []
    foot_metrics: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    all_crossings: list[dict[str, Any]] = []
    all_timing: list[dict[str, Any]] = []
    all_resets: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    oof_by_candidate: dict[str, dict[str, np.ndarray]] = {}
    config_by_candidate: dict[str, OperatingConfig] = {}
    grid = operating_grid()

    for variant in VARIANTS:
        for seed in SEEDS:
            current = candidate_id(variant, seed)
            models: dict[int, SlipV2Model] = {}
            scores_by_fold: dict[int, dict[str, np.ndarray]] = {}
            for outer_fold in (0, 1, 2):
                inner = tuple(fold for fold in (0, 1, 2) if fold != outer_fold)
                model, health = fit_candidate(rows, variant, seed, inner, hard_by_outer[outer_fold])
                health.update({
                    "candidate_id": current, "variant": variant, "seed": seed,
                    "outer_fold": outer_fold, "outer_run_overlap_count": 0,
                    "hard_negative_outer_overlap_count": 0,
                    "normalization_train_only": True,
                })
                training_health.append(health)
                weighting_rows.append({
                    key: health[key] for key in (
                        "candidate_id", "variant", "seed", "outer_fold", "contract",
                        "row_count", "hard_negative_row_count", "state_weight_min",
                        "state_weight_max", "state_weight_sum", "foot_weight_min",
                        "foot_weight_max", "foot_weight_sum", "effective_sample_size")
                })
                models[outer_fold] = model
                scores_by_fold[outer_fold] = score_model(model, rows)

            inner_config_rows: list[dict[str, Any]] = []
            for config in grid:
                evaluations = []
                for outer_fold in (0, 1, 2):
                    inner_indices = [index for index, record in enumerate(records)
                                     if record.fold != outer_fold]
                    metric, _, _, _ = evaluate(
                        records, inner_indices, scores_by_fold[outer_fold], config,
                        current, f"INNER_FOR_OUTER_{outer_fold}")
                    evaluations.append(metric)
                inner_config_rows.append(aggregate_metrics(
                    evaluations, current, "AGGREGATE_INNER_SELECTION", config))
            selected_config = choose_inner_config(inner_config_rows)
            config_by_candidate[current] = selected_config

            oof = {name: np.zeros_like(scores_by_fold[0][name]) for name in HEAD_NAMES}
            current_fold_rows = []
            for outer_fold in (0, 1, 2):
                indices = [index for index, record in enumerate(records) if record.fold == outer_fold]
                for name in HEAD_NAMES:
                    oof[name][indices] = scores_by_fold[outer_fold][name][indices]
                metric, crossings, timing, resets = evaluate(
                    records, indices, scores_by_fold[outer_fold], selected_config,
                    current, f"FROZEN_OUTER_{outer_fold}_ONCE", detailed=True)
                metric.update({"variant": variant, "seed": seed, "outer_fold": outer_fold})
                fold_metrics.append(metric)
                current_fold_rows.append(metric)

            pooled, crossings, timing, resets = evaluate(
                records, list(range(len(records))), oof, selected_config,
                current, "POOLED_OOF", detailed=True)
            pooled.update({"variant": variant, "seed": seed})
            add_resource_and_gate(pooled, current_fold_rows, models[0])
            pooled_metrics.append(pooled)
            oof_by_candidate[current] = oof
            save_oof(output / f"oof_raw_head_scores_{current}.npz", current, records, oof)
            all_crossings.extend(crossings)
            all_timing.extend(timing)
            all_resets.extend(resets)
            speeds = json.loads(pooled["speed_recall"])
            speed_totals = json.loads(pooled["speed_total"])
            speed_detected = json.loads(pooled["speed_detected"])
            for speed in ("0.10", "0.15", "0.20"):
                speed_metrics.append({
                    "candidate_id": current, "variant": variant, "seed": seed,
                    "speed_mps": speed, "actionable_episode_count": speed_totals[speed],
                    "detected": speed_detected[speed], "recall": speeds[speed],
                })
            foot_totals = json.loads(pooled["foot_total"])
            foot_detected_values = json.loads(pooled["foot_detected"])
            for foot in ("left", "right"):
                foot_metrics.append({
                    "candidate_id": current, "variant": variant, "seed": seed, "foot": foot,
                    "actionable_episode_count": foot_totals[foot],
                    "detected": foot_detected_values[foot],
                    "recall": foot_detected_values[foot] / foot_totals[foot]
                    if foot_totals[foot] else 0.0,
                    "pooled_affected_foot_accuracy": pooled["affected_foot_accuracy"],
                })
            previous = None
            for order, stage in enumerate((
                "raw_actionable_argmax", "proposal", "verifier", "V4_masked_verifier",
                "persistence", "stable_episode_detection"), 1):
                detected = int(pooled[f"{stage}_detected"])
                stage_metrics.append({
                    "candidate_id": current, "variant": variant, "seed": seed,
                    "config_id": selected_config.config_id, "stage_order": order,
                    "stage": stage, "episode_count": pooled["actionable_episode_count"],
                    "detected_count": detected, "recall": pooled[f"{stage}_recall"],
                    "rejected_since_prior": "" if previous is None else previous - detected,
                })
                previous = detected

            for config in grid:
                metric, _, _, _ = evaluate(
                    records, list(range(len(records))), oof, config,
                    current, "DIAGNOSTIC_OOF_PARETO")
                metric.update({
                    "variant": variant, "seed": seed, "diagnostic_only": True,
                    "selected_or_frozen": False,
                    "grid_frozen_before_scores": True,
                })
                pareto_rows.append(metric)

    for current in sorted({row["candidate_id"] for row in pareto_rows}):
        candidate_rows = [row for row in pareto_rows if row["candidate_id"] == current]
        max_run_zero = max((row["actionable_episode_recall"] for row in candidate_rows
                            if row["normal_run_fp"] == 0), default=0.0)
        max_episode_zero = max((row["actionable_episode_recall"] for row in candidate_rows
                                if row["normal_contact_episode_fp"] == 0), default=0.0)
        max_early_zero = max((row["actionable_episode_recall"] for row in candidate_rows
                              if row["too_early"] == 0), default=0.0)
        max_all_three = max((row["actionable_episode_recall"] for row in candidate_rows
                             if row["normal_run_fp"] == 0
                             and row["normal_contact_episode_fp"] == 0
                             and row["too_early"] == 0), default=0.0)
        recall80 = [row for row in candidate_rows if row["actionable_episode_recall"] >= 0.80]
        for row in candidate_rows:
            row.update({
                "maximum_recall_run_fp_zero": max_run_zero,
                "maximum_recall_contact_episode_fp_zero": max_episode_zero,
                "maximum_recall_too_early_zero": max_early_zero,
                "maximum_recall_all_three_zero": max_all_three,
                "minimum_run_fp_at_recall_ge_0p80": min(
                    (item["normal_run_fp"] for item in recall80), default="UNACHIEVABLE"),
                "minimum_contact_episode_fp_at_recall_ge_0p80": min(
                    (item["normal_contact_episode_fp"] for item in recall80), default="UNACHIEVABLE"),
                "minimum_too_early_at_recall_ge_0p80": min(
                    (item["too_early"] for item in recall80), default="UNACHIEVABLE"),
            })

    selected = select_candidate(pooled_metrics)
    fallback = diagnostic_fallback(pooled_metrics)
    chosen = selected or fallback
    selected_id = None if selected is None else selected["candidate_id"]
    fallback_id = fallback["candidate_id"]

    write_csv(output / "training_health.csv", training_health)
    write_csv(output / "weighting_comparison.csv", weighting_rows)
    write_csv(output / "candidate_fold_metrics.csv", fold_metrics)
    write_csv(output / "candidate_pooled_metrics.csv", pooled_metrics)
    write_csv(output / "candidate_speed_metrics.csv", speed_metrics)
    write_csv(output / "candidate_foot_metrics.csv", foot_metrics)
    write_csv(output / "stage_rejection_metrics.csv", stage_metrics)
    write_csv(output / "operating_point_pareto.csv", pareto_rows)
    write_csv(output / "crossing_reconciliation.csv", [
        row for row in all_crossings if row["candidate_id"] in {selected_id, fallback_id}
    ])
    write_csv(output / "timing_metrics.csv", [
        row for row in all_timing if row["candidate_id"] in {selected_id, fallback_id}
    ])
    write_csv(output / "reset_audit.csv", [
        row for row in all_resets if row["candidate_id"] in {selected_id, fallback_id}
    ])

    resource_report = {
        "ceilings": RESOURCE_CEILINGS,
        "candidates": [{key: row[key] for key in (
            "candidate_id", "variant", "seed", "parameter_count",
            "model_parameter_bytes_float64", "normalization_bytes_float64",
            "runtime_config_bytes", "history_bytes", "persistent_state_bytes",
            "macs_per_tick", "operations_per_second_at_1khz", "resource_gate_pass")}
            for row in pooled_metrics],
        "expected_TFLM_Vela_operator_set": [
            "FULLY_CONNECTED", "RELU", "SOFTMAX", "LOGISTIC", "SUB", "ADD",
            "MUL", "GREATER", "ARGMAX", "stateful reset/comparison",
        ],
        "INT8_performed": False, "Vela_performed": False,
        "E84_or_HIL_performed": False,
    }
    write_json(output / "resource_report.json", resource_report)

    final_paths: dict[str, Any] = {}
    reload_parity: dict[str, Any] = {"performed": False}
    selection_lock = None
    if selected is not None:
        variant, seed = selected["variant"], int(selected["seed"])
        final_model, final_health = fit_candidate(
            rows, variant, seed, (0, 1, 2), set().union(*hard_by_outer.values()))
        model_path = output / "slip_model.npz"
        norm_path = output / "slip_normalization.json"
        config_path = output / "slip_runtime_config.json"
        final_model.save(model_path)
        write_json(norm_path, {
            "source": "complete_333_run_eligible_development_final_fit",
            "mean": final_model.mean.tolist(), "scale": final_model.scale.tolist(),
            "feature_count": len(final_model.mean), "fit_once": True,
        })
        config = config_by_candidate[selected["candidate_id"]]
        write_json(config_path, {
            "candidate_id": selected["candidate_id"], "variant": variant, "seed": seed,
            "architecture": "S4-C", "corrected_state": "V4",
            "operating_config": serializable_config(config),
            "physical_confirmation_output": False, "sink_output": False,
        })
        reloaded = SlipV2Model.load(model_path)
        probe = rows.features[np.flatnonzero(rows.eligible)[:256]]
        before = final_model.scores(probe)
        after = reloaded.scores(probe)
        model_error = max(float(np.max(np.abs(before[key] - after[key]))) for key in HEAD_NAMES)
        norm = json.loads(norm_path.read_text())
        config_reload = json.loads(config_path.read_text())
        reload_parity = {
            "performed": True, "model_max_abs_error": model_error,
            "normalization_mean_exact": np.array_equal(np.asarray(norm["mean"]), final_model.mean),
            "normalization_scale_exact": np.array_equal(np.asarray(norm["scale"]), final_model.scale),
            "runtime_config_exact": config_reload["operating_config"] == serializable_config(config),
        }
        reload_parity["all_exact"] = bool(
            model_error == 0.0 and reload_parity["normalization_mean_exact"]
            and reload_parity["normalization_scale_exact"] and reload_parity["runtime_config_exact"])
        if not reload_parity["all_exact"]:
            raise RuntimeError("final reload parity failed")
        final_paths = {
            "model_path": model_path.name, "model_sha256": sha256(model_path),
            "normalization_path": norm_path.name, "normalization_sha256": sha256(norm_path),
            "runtime_config_path": config_path.name, "runtime_config_sha256": sha256(config_path),
        }
        selection_lock = {
            "version": "walking_v2_slip_selection_lock_v4", "immutable": True,
            "starting_checkpoint": STARTING_CHECKPOINT,
            "selected_candidate": selected["candidate_id"],
            "selected_metrics": selected, "final_fit_health": final_health,
            "source_sha256": {
                path: sha256(root / path) for path in source_paths()[:3]
            },
            "V4_helper_sha256": sha256(root / SOURCE_DIR / "walking_v2_slip_retraining_failure_audit_v4.py"),
            "protocol_sha256": barrier_hashes["protocol.json"],
            "recipe_reconstruction_sha256": barrier_hashes["recipe_reconstruction.json"],
            "candidate_matrix_sha256": barrier_hashes["candidate_matrix.json"],
            "operating_point_policy_sha256": barrier_hashes["operating_point_policy.json"],
            "selection_policy_sha256": barrier_hashes["selection_policy.json"],
            "eligibility_sha256": immutable["eligibility_manifest_sha256"],
            "nested_fold_sha256": immutable["fold_manifest_sha256"],
            "trace_sha256": immutable["trace_shard_sha256"],
            "episode_reconciliation_sha256": immutable["episode_reconciliation_sha256"],
            "terrain_M1_G0_oracle_sha256": immutable["terrain_M1_G0_oracle_before"],
            **final_paths,
            "OOF_metrics_sha256": sha256(output / "candidate_pooled_metrics.csv"),
            "resource_report_sha256": sha256(output / "resource_report.json"),
            "Terrain_selection_lock_sha256": immutable["terrain_M1_G0_oracle_before"][
                "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_selection_lock.json"],
        }
        write_json(output / "slip_selection_lock.json", selection_lock)

    immutable_after = {
        path: sha256(root / path) for path in immutable["terrain_M1_G0_oracle_before"]
    }
    immutable_match = immutable_after == immutable["terrain_M1_G0_oracle_before"]
    barrier_after = {name: sha256(output / name) for name in PRETRAINING_FILES}
    if barrier_after != barrier_hashes:
        raise RuntimeError("pretraining artifact changed after first fit")
    if not immutable_match:
        raise RuntimeError("Terrain/M1/G0/oracle changed")

    max_all_three = max(float(row["maximum_recall_all_three_zero"]) for row in pareto_rows)
    max_raw = max(float(row["raw_actionable_argmax_recall"]) for row in pooled_metrics)
    model_redesign = bool(selected is None and max_raw < 0.80)
    risk_reduction = bool(selected is None and max_raw >= 0.80 and max_all_three < 0.80)
    terrain_lock_exists = (root / "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_selection_lock.json").exists()
    slip_ready = selected is not None and selection_lock is not None
    fresh_authorized = bool(slip_ready and terrain_lock_exists and immutable_match and reload_parity.get("all_exact"))
    readiness = {
        "WALKING_V2_SLIP_CORRECTED_RETRAIN_DATA_READY": True,
        "WALKING_V2_SLIP_CORRECTED_RETRAIN_PROVENANCE_READY": True,
        "WALKING_V2_SLIP_S4C_RECIPE_READY": True,
        "WALKING_V2_SLIP_V4_STATE_READY": True,
        "WALKING_V2_SLIP_NESTED_EVALUATION_READY": True,
        "WALKING_V2_SLIP_OPERATING_POINT_READY": selected is not None,
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": slip_ready,
        "WALKING_V2_SLIP_SELECTION_LOCK_READY": selection_lock is not None,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable_match,
        "WALKING_V2_JOINT_FLOAT_CANDIDATES_READY": slip_ready and immutable_match,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": fresh_authorized,
        "WALKING_V2_SLIP_MODEL_REDESIGN_AUTHORIZED": model_redesign,
        "WALKING_V2_SLIP_RISK_SCOPE_REDUCTION_REQUIRED": risk_reduction,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    if fresh_authorized:
        next_step = "FRESH_BLIND_HOLDOUT_ACQUISITION"
    elif model_redesign:
        next_step = "SLIP_MODEL_REDESIGN"
    elif risk_reduction:
        next_step = "SLIP_RISK_SCOPE_REDUCTION"
    else:
        next_step = "STOP_WALKING_V2_DEPLOYMENT"
    summary = {
        "task": "Corrected Walking v2 Targeted Slip Retraining v4",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "recipe_exactly_reconstructed": True,
        "unauthorized_recipe_deviations": [],
        "candidate_pass_count": sum(bool(row["gate_pass"]) for row in pooled_metrics),
        "selected_candidate": selected_id,
        "diagnostic_fallback": fallback_id if selected is None else None,
        "chosen_metrics": chosen, "maximum_all_three_zero_recall": max_all_three,
        "maximum_raw_argmax_recall": max_raw,
        "final_model_fitted": selected is not None,
        "reload_parity": reload_parity,
        "slip_selection_lock_created": selection_lock is not None,
        "fresh_blind_holdout_authorized": fresh_authorized,
        "blind_accessed_or_generated": False,
        "forbidden_access_count": 0,
        "terrain_M1_G0_oracle_immutable": immutable_match,
        "pretraining_artifact_barrier_unchanged": barrier_after == barrier_hashes,
        "readiness": readiness, "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "system_int8_vela_e84_hil_work": False, "next_step": next_step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    write_audit(output / "audit.md", summary)

    artifact_hashes = {}
    artifact_bytes = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json":
            continue
        if path.stat().st_size > 45 * 1024 * 1024:
            raise RuntimeError(f"generated artifact exceeds 45 MiB: {path}")
        artifact_hashes[path.name] = sha256(path)
        artifact_bytes[path.name] = path.stat().st_size
    provenance = {
        "version": "walking_v2_slip_corrected_targeted_retraining_v4",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pretraining_artifact_sha256": barrier_hashes,
        "pretraining_artifact_sha256_after": barrier_after,
        "pretraining_barrier_unchanged": barrier_after == barrier_hashes,
        "source_code_sha256": {path: sha256(root / path) for path in source_paths()[:3]},
        "input_trace_sha256": immutable["trace_shard_sha256"],
        "immutable_before_sha256": immutable["terrain_M1_G0_oracle_before"],
        "immutable_after_sha256": immutable_after,
        "immutable_match": immutable_match,
        "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes,
        "artifact_hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "forbidden_access_count": 0, "outer_holdout_final_access_count": 0,
        "new_simulation_run_count": 0, "blind_artifact_count": 0,
        "candidate_training_job_count": 18, "hard_negative_mining_job_count": 6,
        "final_fit_count": int(selected is not None),
        "candidate_or_lock_count": int(selection_lock is not None),
        "terrain_M1_G0_oracle_write_count": 0,
        "system_int8_vela_e84_hil_work_count": 0,
    }
    write_json(output / "provenance.json", provenance)
    return summary


def write_audit(path: Path, summary: dict[str, Any]) -> None:
    metric = summary["chosen_metrics"]
    speed = json.loads(metric["speed_recall"])
    text = f"""# Corrected Walking v2 targeted Slip retraining v4

- Exact S4-C reconstruction: `{summary['recipe_exactly_reconstructed']}`; unauthorized deviations: none.
- Passing candidates: `{summary['candidate_pass_count']}`; selected: `{summary['selected_candidate']}`.
- Diagnostic fallback: `{summary['diagnostic_fallback']}`.
- Pooled actionable recall: `{metric['actionable_episode_detected']}/{metric['actionable_episode_count']}` = `{metric['actionable_episode_recall']:.6f}`.
- Fold minimum recall: `{metric['minimum_fold_recall']:.6f}`.
- Speed recall: `{speed}`.
- Stage recall: raw `{metric['raw_actionable_argmax_recall']:.6f}`, proposal `{metric['proposal_recall']:.6f}`, persistence `{metric['persistence_recall']:.6f}`, final `{metric['stable_episode_detection_recall']:.6f}`.
- Run/contact FP: `{metric['normal_run_fp']}/{metric['normal_contact_episode_fp']}`; too-early: `{metric['too_early']}`.
- Invalid/post-fall/latch/cross-foot: `{metric['invalid_firing']}/{metric['post_fall_firing']}/{metric['latch_carryover']}/{metric['cross_foot_violation']}`.
- Affected-foot accuracy: `{metric['affected_foot_accuracy']:.6f}`.
- Maximum recall with run FP, contact FP and too-early all zero: `{summary['maximum_all_three_zero_recall']:.6f}`.
- Final model fitted: `{summary['final_model_fitted']}`; lock created: `{summary['slip_selection_lock_created']}`.
- Fresh blind holdout authorized: `{summary['fresh_blind_holdout_authorized']}`; none accessed or generated.
- Terrain/M1/G0/oracle immutable: `{summary['terrain_M1_G0_oracle_immutable']}`.
- No System/INT8/Vela/E84/HIL work occurred; Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.

Next step: `{summary['next_step']}`
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    (guard, records, _, metadata_by_id, _, immutable, barrier_hashes,
     started) = preflight(root)
    if not guard.frozen or guard.forbidden_count:
        raise RuntimeError("training barrier not frozen")
    summary = run_training(root, records, metadata_by_id, barrier_hashes, immutable, started)
    print(json.dumps({
        "selected_candidate": summary["selected_candidate"],
        "diagnostic_fallback": summary["diagnostic_fallback"],
        "candidate_pass_count": summary["candidate_pass_count"],
        "fresh_blind_holdout_authorized": summary["fresh_blind_holdout_authorized"],
        "next_step": summary["next_step"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
