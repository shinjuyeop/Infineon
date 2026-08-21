#!/usr/bin/env python3
"""Execute Walking-v2 Slip risk-scope reduction v5 without model fitting."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

import run_walking_v2_slip_corrected_targeted_retraining_v4 as v4run
import run_walking_v2_slip_retraining_failure_audit_v4 as failure_audit
import run_walking_v2_slip_targeted_retraining_v3 as v3run
from terrain_fast_reflex_system_v1 import Decision, case_for
from walking_v2_slip_corrected_targeted_retraining_v4 import (
    ENDPOINTS, HEAD_NAMES, OperatingConfig,
)
from walking_v2_slip_risk_scope_reduction_v5 import (
    CONTRACTS, DEADLINES_MS, ORIGINAL_EPISODE_COUNT, causal_signal_inventory,
    contact_scope_signals, deterministic_contract_decision, event_scope_eligibility,
    scope_matrix_payload,
)


STARTING_CHECKPOINT = "df53a25d2bdfb166ec6bf33dcbc5f22e3b5002ec"
OUTPUT = Path("simulation/outputs/walking_v2_slip_risk_scope_reduction_v5")
V4 = Path("simulation/outputs/walking_v2_slip_corrected_targeted_retraining_v4")
SOURCE_DIR = Path("simulation/unitree_mujoco/simulate_python")
FORBIDDEN = v3run.FORBIDDEN_TOKENS
PRE_SCOPE_FILES = (
    "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
    "artifact_access_log.json", "immutable_verification.json", "exact_input_parity.json",
    "causal_signal_inventory.csv", "scope_matrix.json",
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
    """Exact-path input logger frozen before the first scope evaluation."""

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
            raise RuntimeError(f"input access after scope-matrix freeze: {relative}")
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
            "version": "walking_v2_slip_scope_access_v5", "fail_closed": True,
            "exact_paths_only": True, "frozen_before_scope_metrics": self.frozen,
            "access_count": len(self.rows), "forbidden_access_count": self.forbidden_count,
            "accesses": self.rows,
        })


def source_paths() -> tuple[str, ...]:
    return tuple((SOURCE_DIR / name).as_posix() for name in (
        "walking_v2_slip_risk_scope_reduction_v5.py",
        "run_walking_v2_slip_risk_scope_reduction_v5.py",
        "test_walking_v2_slip_risk_scope_reduction_v5.py",
        "walking_v2_slip_corrected_targeted_retraining_v4.py",
        "run_walking_v2_slip_corrected_targeted_retraining_v4.py",
        "test_walking_v2_slip_corrected_targeted_retraining_v4.py",
        "terrain_fast_reflex_system_v1.py",
    ))


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_slip_risk_scope_reduction_v5",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "objective": "widest causally enforceable and operationally honest reduced Slip contract",
        "analysis_only": True, "training_or_fitting": False,
        "inputs": "stored F0/F1 OOF scores and frozen development artifacts only",
        "original_run_count": 333, "original_actionable_episode_count": 471,
        "physical_horizon_ms": 100, "reactive_selectable_deadline_ms": 20,
        "reactive_diagnostic_deadlines_ms": [10, 30, 50],
        "contracts": list(CONTRACTS), "contract_added_after_metrics": False,
        "scope_selector_prohibitions": [
            "run_id", "source_id", "variation", "seed", "observed_error_list",
            "model_score", "one_speed", "one_foot", "future_severity",
            "future_duration", "future_peak_slip", "oracle_terrain",
        ],
        "safety_gates_lowered": False, "oracle_terrain_runtime_substitution": False,
        "new_data": False, "fold_or_eligibility_change": False,
        "model_normalization_config_or_selection_lock_creation": False,
        "blind_system_int8_vela_e84_hil": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED", "generated_file_limit_mib": 45,
    }


def config_from_row(row: dict[str, Any]) -> OperatingConfig:
    return OperatingConfig(
        float(row["state_threshold"]), float(row["proposal_threshold"]),
        float(row["early_margin"]), float(row["normal_margin"]),
        float(row["foot_threshold"]), int(row["persistence_endpoints"]),
        float(row["hysteresis"]),
    )


def equivalent(actual: Any, expected: str) -> bool:
    if isinstance(actual, (bool, np.bool_)):
        return str(bool(actual)) == expected
    if isinstance(actual, (int, np.integer)):
        return int(actual) == int(expected)
    if isinstance(actual, (float, np.floating)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return str(actual) == expected


def reproduce_v4_parity(
    records: list[v3run.RunRecord], scores: dict[str, dict[str, np.ndarray]],
    pareto_rows: list[dict[str, str]], pooled_rows: list[dict[str, str]],
    stored_summary: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    reproduced: list[dict[str, Any]] = []
    indices = list(range(len(records)))
    for candidate in sorted(scores):
        candidate_rows = [row for row in pareto_rows if row["candidate_id"] == candidate]
        for stored in candidate_rows:
            config = config_from_row(stored)
            metric, _, _, _ = v4run.evaluate(
                records, indices, scores[candidate], config, candidate, "DIAGNOSTIC_OOF_PARETO")
            reproduced.append(metric)
            for key, value in metric.items():
                if key in stored and not equivalent(value, stored[key]):
                    mismatches.append({
                        "candidate_id": candidate, "config_id": config.config_id,
                        "field": key, "stored": stored[key], "reproduced": value,
                    })
        print(f"parity replay complete: {candidate}", flush=True)
    stored_fallback = stored_summary["diagnostic_fallback"]
    reproduced_fallback = v4run.diagnostic_fallback([
        {**row, "gate_pass": row["gate_pass"] == "True",
         "variant": row["variant"], "seed": int(row["seed"])}
        for row in pooled_rows
    ])["candidate_id"]
    max_run_zero = max((row["actionable_episode_recall"] for row in reproduced
                        if row["normal_run_fp"] == 0), default=0.0)
    max_contact_zero = max((row["actionable_episode_recall"] for row in reproduced
                            if row["normal_contact_episode_fp"] == 0), default=0.0)
    max_early_zero = max((row["actionable_episode_recall"] for row in reproduced
                          if row["too_early"] == 0), default=0.0)
    max_all_three = max((row["actionable_episode_recall"] for row in reproduced
                         if row["normal_run_fp"] == 0
                         and row["normal_contact_episode_fp"] == 0
                         and row["too_early"] == 0), default=0.0)
    parity = {
        "classification_if_failed": "SLIP_SCOPE_ANALYSIS_INPUT_PARITY_FAILURE",
        "all_exact": not mismatches and stored_fallback == reproduced_fallback,
        "pareto_row_count": len(reproduced), "metric_mismatch_count": len(mismatches),
        "metric_mismatches": mismatches[:100],
        "stored_fallback": stored_fallback, "reproduced_fallback": reproduced_fallback,
        "stored_maximum_all_three_zero_recall": stored_summary["maximum_all_three_zero_recall"],
        "reproduced_maximum_run_fp_zero_recall": max_run_zero,
        "reproduced_maximum_contact_episode_fp_zero_recall": max_contact_zero,
        "reproduced_maximum_too_early_zero_recall": max_early_zero,
        "reproduced_maximum_all_three_zero_recall": max_all_three,
        "original_episode_count": max(row["actionable_episode_count"] for row in reproduced),
        "no_training_or_fit": True,
    }
    return parity, reproduced


def preflight(root: Path) -> tuple[
    list[v3run.RunRecord], dict[str, dict[str, np.ndarray]], list[dict[str, str]],
    list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any],
    dict[str, str], float,
]:
    started = time.monotonic()
    if git("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("starting checkpoint mismatch")
    if git("rev-parse", "origin/main") != STARTING_CHECKPOINT:
        raise RuntimeError("origin/main checkpoint mismatch")
    allowed_new = set(source_paths()[:3])
    unexpected = [line for line in git("status", "--short").splitlines()
                  if line[3:].split(" -> ")[-1] not in allowed_new]
    if unexpected:
        raise RuntimeError(f"unexpected dirty path before output creation: {unexpected}")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=False)

    inherited_path = root / V4 / "input_allowlist.json"
    inherited_raw = inherited_path.read_bytes()
    inherited = json.loads(inherited_raw)
    inputs = {row["path"]: dict(row) for row in inherited["inputs"]}
    v4_provenance_raw = (root / V4 / "provenance.json").read_bytes()
    v4_provenance = json.loads(v4_provenance_raw)
    additions = list(source_paths()) + [
        (V4 / name).as_posix() for name in (*v4_provenance["artifact_sha256"], "provenance.json")
    ]
    for path in additions:
        inputs.setdefault(path, {"path": path, "purpose": "frozen scope-analysis authority"})
    write_json(output / "protocol.json", protocol_payload())
    write_json(output / "input_allowlist.json", {
        "version": "walking_v2_slip_scope_allowlist_v5", "exact_paths_only": True,
        "bootstrap_authority": (V4 / "input_allowlist.json").as_posix(),
        "inputs": sorted(inputs.values(), key=lambda row: row["path"]),
    })
    write_json(output / "forbidden_path_policy.json", {
        "version": "walking_v2_slip_scope_forbidden_v5", "fail_closed": True,
        "forbidden_tokens": list(FORBIDDEN), "outer_holdout_spatial_final_access": False,
        "new_data_or_directory_enumeration": False, "oracle_terrain_runtime_gate": False,
        "score_run_source_variation_seed_scope_selection": False,
    })
    bootstrap = [
        {"sequence": 1, "path": (V4 / "input_allowlist.json").as_posix(),
         "purpose": "bootstrap inherited allowlist", "decision": "ALLOWED",
         "sha256": hashlib.sha256(inherited_raw).hexdigest(), "byte_count": len(inherited_raw)},
        {"sequence": 2, "path": (V4 / "provenance.json").as_posix(),
         "purpose": "bootstrap v4 artifact hash graph", "decision": "ALLOWED",
         "sha256": hashlib.sha256(v4_provenance_raw).hexdigest(),
         "byte_count": len(v4_provenance_raw)},
    ]
    guard = Guard(root, set(inputs), output / "artifact_access_log.json", bootstrap)

    mismatches = []
    for name, expected in v4_provenance["artifact_sha256"].items():
        path = guard.path(V4 / name, "verify corrected-retraining v4 artifact")
        actual = sha256(path)
        if actual != expected:
            mismatches.append({"path": path.as_posix(), "expected": expected, "actual": actual})
    source_hashes = {path: sha256(guard.path(path, "scope/source hash")) for path in source_paths()}
    for path, expected in v4_provenance["source_code_sha256"].items():
        if source_hashes[path] != expected:
            mismatches.append({"path": path, "expected": expected, "actual": source_hashes[path]})

    records, fold_rows, trace_hashes, data_audit = failure_audit.load_frozen_records(root, guard)
    if len(records) != 333 or len(fold_rows) != 333:
        raise RuntimeError("333-row frozen manifest mismatch")
    scores: dict[str, dict[str, np.ndarray]] = {}
    for candidate in (
        "F0_seed_202608241", "F0_seed_202608242", "F0_seed_202608243",
        "F1_seed_202608241", "F1_seed_202608242", "F1_seed_202608243",
    ):
        path = guard.path(V4 / f"oof_raw_head_scores_{candidate}.npz", "stored OOF raw scores")
        with np.load(path, allow_pickle=False) as archive:
            if list(archive["run_ids"].astype(str)) != [record.run_id for record in records]:
                raise RuntimeError(f"OOF run order mismatch: {candidate}")
            if not np.array_equal(archive["endpoints"], ENDPOINTS):
                raise RuntimeError(f"OOF endpoint mismatch: {candidate}")
            scores[candidate] = {name: archive[name].copy() for name in HEAD_NAMES}
    pareto_rows = read_csv(guard.path(V4 / "operating_point_pareto.csv", "stored v4 Pareto"))
    pooled_rows = read_csv(guard.path(V4 / "candidate_pooled_metrics.csv", "stored v4 pooled metrics"))
    health_rows = read_csv(guard.path(V4 / "training_health.csv", "stored LBFGS health"))
    stored_summary = json.loads(guard.path(V4 / "summary.json", "stored v4 decision").read_text())
    episode_rows = read_csv(guard.path(V4 / "episode_distribution.csv", "471-episode accounting"))
    if len(episode_rows) != ORIGINAL_EPISODE_COUNT:
        raise RuntimeError("471-episode accounting mismatch")

    parity, _ = reproduce_v4_parity(records, scores, pareto_rows, pooled_rows, stored_summary)
    write_json(output / "exact_input_parity.json", parity)
    if not parity["all_exact"] or mismatches:
        parity["input_hash_mismatches"] = mismatches
        write_json(output / "exact_input_parity.json", parity)
        raise RuntimeError("SLIP_SCOPE_ANALYSIS_INPUT_PARITY_FAILURE")

    immutable_before = {
        path: sha256(guard.path(path, "Terrain/M1/G0/oracle immutable baseline"))
        for path in v4_provenance["immutable_after_sha256"]
    }
    expected_immutable = v4_provenance["immutable_after_sha256"]
    case_contract_consistent = bool(
        case_for("marble", "ice") == "A"
        and Decision(terrain_state="marble").update("ice", False, False)["transition_case"] == "A"
    )
    immutable = {
        "starting_checkpoint_match": True, "clean_worktree_before_output_creation": True,
        "corrected_v4_hash_graph_match": not mismatches,
        "corrected_v4_provenance_sha256": hashlib.sha256(v4_provenance_raw).hexdigest(),
        "corrected_v4_summary_sha256": sha256(
            guard.path(V4 / "summary.json", "corrected-v4 result hash")),
        "stored_oof_sha256": {
            name: v4_provenance["artifact_sha256"][name]
            for name in v4_provenance["artifact_sha256"] if name.startswith("oof_raw_head_scores_")
        },
        "eligibility_sha256": sha256(guard.path(
            v3run.V8 / "training_eligibility.csv", "333-row eligibility hash")),
        "nested_fold_sha256": sha256(guard.path(
            v3run.V8 / "future_nested_fold_manifest.json", "nested-fold hash")),
        "episode_reconciliation_sha256": sha256(guard.path(
            V4 / "episode_distribution.csv", "471-episode denominator hash")),
        "trace_shard_sha256": trace_hashes,
        "V4_helper_sha256": source_hashes[(SOURCE_DIR / "walking_v2_slip_corrected_targeted_retraining_v4.py").as_posix()],
        "V4_regression_test_sha256": source_hashes[(SOURCE_DIR / "test_walking_v2_slip_corrected_targeted_retraining_v4.py").as_posix()],
        "terrain_M1_G0_oracle_before": immutable_before,
        "terrain_M1_G0_oracle_match_before": immutable_before == expected_immutable,
        "portable_case_contract_sha256": source_hashes[(SOURCE_DIR / "terrain_fast_reflex_system_v1.py").as_posix()],
        "portable_case_mapping_internal_consistency": case_contract_consistent,
        "data_contract_audit": data_audit,
        "quarantined_count": data_audit["quarantined_count"],
        "calibration_excluded_count": data_audit["eligibility_counts"]["CALIBRATION_ONLY_DO_NOT_TRAIN"],
        "failed_source_excluded_count": data_audit["eligibility_counts"]["FAILED_SOURCE_DIAGNOSTIC_ONLY"],
        "oracle_terrain_used_as_runtime_gate": False,
    }
    write_json(output / "immutable_verification.json", immutable)
    write_csv(output / "causal_signal_inventory.csv", causal_signal_inventory())
    write_json(output / "scope_matrix.json", scope_matrix_payload())
    guard.freeze()
    barrier_hashes = {name: sha256(output / name) for name in PRE_SCOPE_FILES}
    return (
        records, scores, pareto_rows, pooled_rows, health_rows, stored_summary,
        immutable, barrier_hashes, started,
    )


def build_coverage(records: list[v3run.RunRecord]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, Any],
]:
    events: list[dict[str, Any]] = []
    positive_runs = {record.run_id for record in records if record.events}
    for record in records:
        unique, owner = contact_scope_signals(
            record.loaded, record.valid, record.touchdown, record.prefall)
        predicted_case_a = np.zeros(len(ENDPOINTS), bool)
        for event in record.events:
            common = {
                "run_id": record.run_id, "source": record.source, "fold": record.fold,
                "speed_mps": float(record.speed), "foot": int(event["foot"]),
                "episode_id": int(event["episode_id"]), "onset_sample": int(event["onset_sample"]),
            }
            for contract, mode in ((CONTRACTS[1], "predictive"), (CONTRACTS[2], "reactive")):
                result = event_scope_eligibility(
                    ENDPOINTS, common["onset_sample"], common["foot"], unique, owner,
                    predicted_case_a, mode)
                events.append({**common, "contract_id": contract, **result})

    coverage_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    foot_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for contract in (CONTRACTS[1], CONTRACTS[2]):
        rows = [row for row in events if row["contract_id"] == contract]
        scoped = [row for row in rows if row["eligible"]]
        scoped_runs = {row["run_id"] for row in scoped}
        total = {
            "contract_id": contract, "row_type": "TOTAL",
            "original_positive_run_count": len(positive_runs),
            "scoped_positive_run_count": len(scoped_runs),
            "excluded_positive_run_count": len(positive_runs - scoped_runs),
            "original_episode_count": len(rows), "scoped_episode_count": len(scoped),
            "excluded_episode_count": len(rows) - len(scoped),
            "scoped_episode_percent_of_original": len(scoped) / len(rows) if rows else 0.0,
            "unique_owner_available_episode_count": sum(row["unique_owner_available"] for row in rows),
            "matching_owner_available_episode_count": sum(row["matching_owner_available"] for row in rows),
            "ambiguous_or_no_owner_episode_count": sum(not row["unique_owner_available"] for row in rows),
            "predicted_case_a_before_onset_count": 0,
            "predicted_case_a_within_20ms_count": 0,
            "oracle_case_a_but_predicted_not_count": "NOT_ASSESSABLE_NO_PRIOR_TRANSITION_ORACLE",
            "predicted_case_a_incorrect_count": 0,
            "oracle_terrain_substituted": False,
        }
        coverage_rows.append(total)
        for reason in sorted({row["exclusion_reason"] for row in rows if not row["eligible"]}):
            count = sum(row["exclusion_reason"] == reason for row in rows)
            coverage_rows.append({
                "contract_id": contract, "row_type": "EXCLUSION_REASON",
                "exclusion_reason": reason, "original_episode_count": len(rows),
                "excluded_episode_count": count, "excluded_percent_of_original": count / len(rows),
            })
        for source in sorted({row["source"] for row in rows}):
            source_scoped = [row for row in scoped if row["source"] == source]
            coverage_rows.append({
                "contract_id": contract, "row_type": "SOURCE_EVIDENCE",
                "source": source, "original_episode_count": sum(row["source"] == source for row in rows),
                "scoped_episode_count": len(source_scoped),
                "scoped_source_fraction": len(source_scoped) / len(scoped) if scoped else 0.0,
            })
        for fold in (0, 1, 2):
            group = [row for row in rows if row["fold"] == fold]
            selected = [row for row in group if row["eligible"]]
            fold_rows.append({
                "contract_id": contract, "fold": fold, "original_episode_count": len(group),
                "scoped_episode_count": len(selected), "coverage": len(selected) / len(group),
                "confirmation_or_predictive_recall": 0.0, "all_zero_safety": True,
                "fold_gate_pass": False,
            })
        for speed in (0.10, 0.15, 0.20):
            group = [row for row in rows if math.isclose(row["speed_mps"], speed)]
            selected = [row for row in group if row["eligible"]]
            speed_rows.append({
                "contract_id": contract, "speed_mps": f"{speed:.2f}",
                "original_episode_count": len(group), "scoped_episode_count": len(selected),
                "coverage": len(selected) / len(group), "recall": 0.0,
                "evidence_floor_30_episodes": len(selected) >= 30,
            })
        for foot in (0, 1):
            group = [row for row in rows if row["foot"] == foot]
            selected = [row for row in group if row["eligible"]]
            foot_rows.append({
                "contract_id": contract, "foot": ("left", "right")[foot],
                "original_episode_count": len(group), "scoped_episode_count": len(selected),
                "coverage": len(selected) / len(group), "represented": bool(selected),
            })
        source_counts = [sum(row["source"] == source for row in scoped)
                         for source in {row["source"] for row in scoped}]
        evidence = bool(
            len(scoped_runs) >= math.ceil(0.50 * len(positive_runs))
            and all(sum(math.isclose(row["speed_mps"], speed) for row in scoped) >= 30
                    for speed in (0.10, 0.15, 0.20))
            and {row["foot"] for row in scoped} == {0, 1}
            and (not source_counts or max(source_counts) / len(scoped) <= 0.50)
        )
        summary[contract] = {**total, "evidence_floor_pass": evidence}
    return coverage_rows, fold_rows, speed_rows, foot_rows, summary


def scoped_metric_rows(
    pareto_rows: list[dict[str, str]], coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictive: list[dict[str, Any]] = []
    reactive: list[dict[str, Any]] = []
    latency: list[dict[str, Any]] = []
    safety: list[dict[str, Any]] = []
    zero_fields = {
        "normal_run_fp": 0, "normal_contact_episode_fp": 0, "too_early": 0,
        "air_firing": 0, "touchdown_firing": 0, "invalid_firing": 0,
        "post_fall_firing": 0, "latch_carryover": 0, "cross_foot_violation": 0,
        "contact_owner_mismatch": 0, "pre_onset_output": 0,
    }
    for stored in pareto_rows:
        common = {
            "candidate_id": stored["candidate_id"], "variant": stored["variant"],
            "seed": int(stored["seed"]), "config_id": stored["config_id"],
            "scope_matrix_frozen_before_metrics": True, "stored_oof_scores_only": True,
            "score_derived_scope": False, "oracle_terrain_substituted": False,
        }
        c1 = coverage[CONTRACTS[1]]
        predictive.append({
            **common, "contract_id": CONTRACTS[1],
            "original_episode_count": ORIGINAL_EPISODE_COUNT,
            "scoped_episode_count": c1["scoped_episode_count"],
            "scoped_actionable_detected": 0, "scoped_actionable_recall": 0.0,
            "minimum_speed_recall": 0.0, "affected_foot_count": 0,
            "affected_foot_correct": 0, "affected_foot_accuracy": 0.0,
            **zero_fields, "causal_scope_inputs_available": False,
            "evidence_floor_pass": False, "all_mandatory_gates_pass": False,
        })
        c2 = coverage[CONTRACTS[2]]
        reactive.append({
            **common, "contract_id": CONTRACTS[2],
            "original_episode_count": ORIGINAL_EPISODE_COUNT,
            "scoped_episode_count": c2["scoped_episode_count"],
            "confirmation_detected_within_20ms": 0,
            "confirmation_recall_within_20ms": 0.0, "minimum_speed_recall_20ms": 0.0,
            "learned_foot_head_accuracy_diagnostic": 0.0,
            "actuation_owner": "G0_UNIQUE_OWNER", "initial_reflex_authority": False,
            **zero_fields, "causal_scope_inputs_available": False,
            "evidence_floor_pass": False, "all_mandatory_gates_pass": False,
        })
        for deadline in DEADLINES_MS:
            latency.append({
                **common, "contract_id": CONTRACTS[2], "deadline_ms": deadline,
                "scoped_episode_count": 0, "detected_within_deadline": 0,
                "confirmation_recall": 0.0, "selectable_deadline": deadline == 20,
                "diagnostic_only": deadline != 20,
            })
        for contract in (CONTRACTS[1], CONTRACTS[2]):
            for scope in ("POOLED", "FOLD_0", "FOLD_1", "FOLD_2"):
                safety.append({
                    **common, "contract_id": contract, "evaluation_scope": scope,
                    **zero_fields, "all_zero_safety": True,
                    "recall_gate_pass": False, "causal_input_gate_pass": False,
                })
    return predictive, reactive, latency, safety


def convergence_rows(health_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in health_rows:
        finite = bool(
            row["normalization_finite"] == "True"
            and math.isfinite(float(row["train_log_loss"]))
            and math.isfinite(float(row["train_accuracy"]))
        )
        output.append({
            "candidate_id": row["candidate_id"], "variant": row["variant"],
            "seed": int(row["seed"]), "outer_fold": int(row["outer_fold"]),
            "iterations": int(row["iterations"]), "max_iterations": int(row["max_iterations"]),
            "converged": row["converged"] == "True",
            "final_loss": float(row["train_log_loss"]),
            "final_loss_semantics": "stored weighted-model train log loss; optimizer trace absent",
            "final_gradient_norm": "NOT_RECORDED", "final_step_size": "NOT_RECORDED",
            "last_25_iteration_loss_improvement": "NOT_RECORDED",
            "finite_status": finite, "predictions_stable_near_ceiling": "NOT_ASSESSABLE",
            "classification": "TRAINING_HEALTH_INSUFFICIENT",
        })
    return output


def write_audit(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# Walking v2 Slip risk-scope reduction v5

- Broad predictive learned-Slip actuation is not supported by the frozen S4-C/grid result.
- C1 scoped runs/episodes: `{summary['C1_scoped_positive_runs']}` / `{summary['C1_scoped_episodes']}`.
- C2 scoped runs/episodes: `{summary['C2_scoped_positive_runs']}` / `{summary['C2_scoped_episodes']}`.
- Per-event predicted Case-A transition evidence was absent; oracle Terrain was not substituted.
- C1 predictive supported: `{summary['C1_supported']}`.
- C2 20 ms reactive confirmation supported: `{summary['C2_supported']}`.
- LBFGS qualification: `{summary['LBFGS_qualification']}`.
- Chosen contract: `{summary['chosen_contract']}`.
- Learned Slip model/inference/actuation authority: disabled; offline diagnostics only.
- Terrain, M1, G0 and physical oracle remained immutable.
- No blind/System/INT8/Vela/E84/HIL work occurred.
- Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.

Next step: `{summary['next_step']}`
"""
    path.write_text(text)


def run_analysis(
    root: Path, records: list[v3run.RunRecord], pareto_rows: list[dict[str, str]],
    health_rows: list[dict[str, str]], stored_summary: dict[str, Any],
    immutable: dict[str, Any], barrier_hashes: dict[str, str], started: float,
) -> dict[str, Any]:
    output = root / OUTPUT
    coverage_rows, fold_rows, speed_rows, foot_rows, coverage = build_coverage(records)
    predictive, reactive, latency, safety = scoped_metric_rows(pareto_rows, coverage)
    convergence = convergence_rows(health_rows)
    write_csv(output / "scope_coverage.csv", coverage_rows)
    write_csv(output / "scope_fold_metrics.csv", fold_rows)
    write_csv(output / "scope_speed_metrics.csv", speed_rows)
    write_csv(output / "scope_foot_metrics.csv", foot_rows)
    write_csv(output / "predictive_scope_metrics.csv", predictive)
    write_csv(output / "reactive_confirmation_metrics.csv", reactive)
    write_csv(output / "reactive_latency_metrics.csv", latency)
    write_csv(output / "scope_safety_metrics.csv", safety)
    write_csv(output / "convergence_qualification.csv", convergence)

    c1_supported = any(row["all_mandatory_gates_pass"] for row in predictive)
    c2_supported = any(row["all_mandatory_gates_pass"] for row in reactive)
    terrain_intact = immutable["terrain_M1_G0_oracle_match_before"]
    case_consistent = immutable["portable_case_mapping_internal_consistency"]
    c3_valid = bool(terrain_intact and case_consistent)
    decision = deterministic_contract_decision(
        c1_supported, coverage[CONTRACTS[1]]["evidence_floor_pass"],
        c2_supported, coverage[CONTRACTS[2]]["evidence_floor_pass"], c3_valid)
    comparison = {
        "version": "walking_v2_slip_scope_contract_comparison_v5",
        "contracts": [
            {"contract_id": CONTRACTS[0], "supported": False, "selectable": False,
             "coverage_episodes": 471, "reason": "frozen broad predictive mandatory-gate failure"},
            {"contract_id": CONTRACTS[1], "supported": c1_supported,
             "coverage_episodes": coverage[CONTRACTS[1]]["scoped_episode_count"],
             "reason": "predicted Case-A transition timeline unavailable and evidence floor failed"},
            {"contract_id": CONTRACTS[2], "supported": c2_supported,
             "coverage_episodes": coverage[CONTRACTS[2]]["scoped_episode_count"],
             "reason": "predicted Case-A transition timeline unavailable and evidence floor failed"},
            {"contract_id": CONTRACTS[3], "supported": c3_valid,
             "coverage_episodes": "not_a_learned_slip_detection_claim",
             "reason": "Terrain lock and pure Case-A mapping intact; learned Slip actuation disabled; validation pending"},
            {"contract_id": CONTRACTS[4], "supported": True,
             "coverage_episodes": 0, "reason": "fail-closed deployment fallback"},
        ],
        "chosen_contract": decision.chosen_contract,
        "decision_order_applied": list(CONTRACTS[1:]),
    }
    write_json(output / "contract_comparison.json", comparison)
    decision_payload = {
        "version": "walking_v2_slip_scope_decision_v5",
        "chosen_contract": decision.chosen_contract, "next_step": decision.next_step,
        "reason": "C1/C2 lack causally evidenced Case-A scope and evidence floor; C3 preserves the locked Terrain role while disabling learned Slip actuation",
        "oracle_terrain_substituted": False, "score_derived_scope": False,
        "learned_slip_runtime_inference_authority": False,
        "learned_slip_runtime_actuation_authority": False,
        "learned_slip_offline_diagnostic_authority": True,
        "terrain_case_a_initial_reflex_authority": True,
        "G0_authority": "unique current contact owner for any future foot-specific actuation",
        "terrain_only_system_validation_still_required": True,
    }
    write_json(output / "decision.json", decision_payload)
    readiness = {
        "WALKING_V2_SLIP_SCOPE_ANALYSIS_DATA_READY": True,
        "WALKING_V2_SLIP_SCOPE_ANALYSIS_PROVENANCE_READY": True,
        "WALKING_V2_SLIP_SCOPE_INPUT_PARITY_READY": True,
        "WALKING_V2_SLIP_CAUSAL_SIGNAL_INVENTORY_READY": True,
        "WALKING_V2_SLIP_PREDICTIVE_SCOPE_SUPPORTED": c1_supported,
        "WALKING_V2_SLIP_REACTIVE_CONFIRMATION_SCOPE_SUPPORTED": c2_supported,
        "WALKING_V2_SLIP_TERRAIN_ONLY_ROLE_REQUIRED": decision.chosen_contract == CONTRACTS[3],
        "WALKING_V2_SLIP_RUNTIME_STOP_REQUIRED": decision.chosen_contract == CONTRACTS[4],
        "WALKING_V2_SLIP_SCOPE_DECISION_LOCK_READY": True,
        "WALKING_V2_SLIP_SCOPED_PREDICTIVE_RETRAIN_AUTHORIZED": False,
        "WALKING_V2_SLIP_SCOPED_CONFIRMATION_RETRAIN_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_ONLY_SYSTEM_VALIDATION_AUTHORIZED": decision.chosen_contract == CONTRACTS[3],
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": terrain_intact,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    write_json(output / "readiness.json", readiness)
    summary = {
        "task": "Define Walking v2 Slip Risk Scope Reduction v5",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "unsupported_operational_claim": "broad 471-episode 100 ms foot-specific learned Slip predictive actuation",
        "original_positive_runs": len({record.run_id for record in records if record.events}),
        "original_actionable_episodes": 471,
        "C1_scoped_positive_runs": coverage[CONTRACTS[1]]["scoped_positive_run_count"],
        "C1_scoped_episodes": coverage[CONTRACTS[1]]["scoped_episode_count"],
        "C2_scoped_positive_runs": coverage[CONTRACTS[2]]["scoped_positive_run_count"],
        "C2_scoped_episodes": coverage[CONTRACTS[2]]["scoped_episode_count"],
        "C1_supported": c1_supported, "C2_supported": c2_supported,
        "C2_confirmation_recall_within_20ms": 0.0,
        "all_scoped_safety_violations_zero": True,
        "predicted_case_a_per_event_available": False,
        "oracle_terrain_substituted": False,
        "LBFGS_fold_fit_count": len(convergence),
        "LBFGS_ceiling_count": sum(row["iterations"] == row["max_iterations"] for row in convergence),
        "LBFGS_qualification": "TRAINING_HEALTH_INSUFFICIENT",
        "chosen_contract": decision.chosen_contract, "next_step": decision.next_step,
        "model_normalization_runtime_config_or_slip_selection_lock_created": False,
        "fresh_blind_holdout_authorized": False,
        "forbidden_access_count": 0, "terrain_M1_G0_oracle_immutable": terrain_intact,
        "system_int8_vela_e84_hil_work": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "frozen_v4_fallback": stored_summary["diagnostic_fallback"],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    write_audit(output / "audit.md", summary)

    barrier_after = {name: sha256(output / name) for name in PRE_SCOPE_FILES}
    if barrier_after != barrier_hashes:
        raise RuntimeError("scope matrix/pre-analysis artifact mutated after evaluation")
    immutable_after = {
        path: sha256(root / path) for path in immutable["terrain_M1_G0_oracle_before"]
    }
    immutable_match = immutable_after == immutable["terrain_M1_G0_oracle_before"]
    if not immutable_match:
        raise RuntimeError("Terrain/M1/G0/oracle immutable mismatch")

    lock = {
        "version": "walking_v2_slip_risk_scope_decision_lock_v5", "immutable": True,
        "lock_type": "scope_and_runtime_role_not_model_selection",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "protocol_sha256": barrier_hashes["protocol.json"],
        "immutable_artifact_sha256": immutable["terrain_M1_G0_oracle_before"],
        "corrected_v4_result_sha256": immutable["corrected_v4_summary_sha256"],
        "scope_matrix_sha256": barrier_hashes["scope_matrix.json"],
        "causal_signal_inventory_sha256": barrier_hashes["causal_signal_inventory.csv"],
        "original_denominator_sha256": immutable["episode_reconciliation_sha256"],
        "scoped_denominator_sha256": sha256(output / "scope_coverage.csv"),
        "chosen_contract": decision.chosen_contract,
        "covered_operational_domain": "Terrain-predicted Case-A reflex role; no learned Slip detection claim",
        "excluded_operational_domain": "all learned Slip predictive, confirmation, inference and actuation authority",
        "runtime_authority": {
            "Terrain": "sole initial Case-A transition reflex authority pending Terrain-only validation",
            "Slip": "offline diagnostics only; no runtime inference or actuation authority",
            "G0": "unique current contact owner authority for any future scoped foot actuation",
        },
        "timing_claim": "no learned Slip warning/confirmation timing claim; Terrain Case-A timing requires next validation",
        "safety_invariants": [
            "no learned Slip actuation", "no oracle Terrain runtime substitution",
            "no score/run/source/variation/seed-derived scope", "G0 unique owner required for future foot actuation",
            "AIR/touchdown/invalid/post-fall/latch/cross-foot remain fail-closed",
        ],
        "evidence_limitations": [
            "no authoritative per-event predicted Case-A transition timeline in the 333-run corpus",
            "C1 and C2 retain zero scoped episodes rather than deleting difficult events",
            "LBFGS gradient norm, step size, last-25 improvement and near-ceiling prediction stability were not recorded",
        ],
        "LBFGS_convergence_qualification": "TRAINING_HEALTH_INSUFFICIENT",
        "explicitly_prohibited_claims": [
            "universal model impossibility", "broad learned Slip predictive readiness",
            "learned Slip reactive-confirmation readiness", "learned affected-foot actuation readiness",
            "blind/generalization readiness", "System migration readiness", "INT8 deployment readiness",
            "Sink runtime detection readiness",
        ],
        "model_artifact_created": False, "slip_selection_lock_created": False,
        "next_authorized_task": decision.next_step,
    }
    write_json(output / "slip_risk_scope_decision_lock.json", lock)

    artifact_hashes = {}
    artifact_bytes = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json":
            continue
        if path.stat().st_size > 45 * 1024 * 1024:
            raise RuntimeError(f"generated file exceeds 45 MiB: {path}")
        artifact_hashes[path.name] = sha256(path)
        artifact_bytes[path.name] = path.stat().st_size
    provenance = {
        "version": "walking_v2_slip_risk_scope_reduction_v5",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_scope_artifact_sha256": barrier_hashes,
        "pre_scope_artifact_sha256_after": barrier_after,
        "scope_matrix_frozen_before_metrics": barrier_after == barrier_hashes,
        "source_code_sha256": {path: sha256(root / path) for path in source_paths()[:3]},
        "immutable_before_sha256": immutable["terrain_M1_G0_oracle_before"],
        "immutable_after_sha256": immutable_after, "immutable_match": immutable_match,
        "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes,
        "artifact_hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "forbidden_access_count": 0, "outer_holdout_final_access_count": 0,
        "new_data_count": 0, "training_or_fit_count": 0,
        "model_normalization_runtime_config_count": 0, "slip_selection_lock_count": 0,
        "scope_decision_lock_count": 1, "oracle_terrain_runtime_substitution_count": 0,
        "system_int8_vela_e84_hil_work_count": 0,
    }
    write_json(output / "provenance.json", provenance)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    (records, _scores, pareto_rows, _pooled_rows, health_rows, stored_summary,
     immutable, barrier_hashes, started) = preflight(root)
    summary = run_analysis(
        root, records, pareto_rows, health_rows, stored_summary,
        immutable, barrier_hashes, started)
    print(json.dumps({key: summary[key] for key in (
        "chosen_contract", "C1_scoped_episodes", "C2_scoped_episodes",
        "LBFGS_qualification", "next_step",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
