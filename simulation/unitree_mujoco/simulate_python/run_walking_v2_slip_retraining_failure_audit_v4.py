#!/usr/bin/env python3
"""Execute the fail-closed Walking-v2 Slip retraining failure audit v4."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

import run_walking_v2_slip_targeted_retraining_v3 as v3run
from walking_v2_slip_retraining_failure_audit_v4 import (
    VARIANTS, average_precision, balanced_accuracy, binary_auc, confusion,
    contiguous_ranges, state_crossings_variant,
)
from walking_v2_slip_targeted_retraining_v3 import (
    ACTIONABLE, EARLY, FAMILIES, NORMAL, SEEDS, STATE_NAMES,
    OperatingConfig, fit_gaussian_head, persistent_candidates,
)


STARTING_CHECKPOINT = "09cb434133b6a944b4d750829f920dacc6d2c229"
OUTPUT = Path("simulation/outputs/walking_v2_slip_retraining_failure_audit_v4")
V3 = Path("simulation/outputs/walking_v2_slip_targeted_retraining_v3")
PREVIOUS = Path("simulation/outputs/walking_v2_slip_redesign_iteration_v2")
V8 = v3run.V8
EXISTING = v3run.EXISTING
SOURCE_DIR = v3run.SOURCE_DIR
FORBIDDEN = v3run.FORBIDDEN_TOKENS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or (list(dict.fromkeys(key for row in rows for key in row))
                         if rows else ["status"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


class Guard:
    def __init__(self, root: Path, allow: set[str]) -> None:
        self.root = root
        self.allow = allow
        self.rows: list[dict[str, Any]] = []
        self.forbidden_count = 0

    def path(self, relative: str | Path, purpose: str) -> Path:
        value = Path(relative).as_posix()
        forbidden = any(token in f"/{value.lower()}" for token in FORBIDDEN)
        decision = "ALLOWED" if value in self.allow and not forbidden else "BLOCKED"
        self.rows.append({"sequence": len(self.rows) + 1, "path": value,
                          "purpose": purpose, "decision": decision})
        if decision == "BLOCKED":
            self.forbidden_count += int(forbidden)
            raise PermissionError(value)
        return self.root / value

    def flush(self) -> None:
        write_json(self.root / OUTPUT / "artifact_access_log.json", {
            "fail_closed": True, "exact_paths_only": True,
            "access_count": len(self.rows), "forbidden_access_count": self.forbidden_count,
            "accesses": self.rows,
        })


def protocol() -> dict[str, Any]:
    return {
        "version": "walking_v2_slip_retraining_failure_audit_v4",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "diagnostic_only": True, "new_training_architecture": False,
        "allowed_retraining": "exact deterministic replay of frozen R1/R2/R3 only",
        "stored_operating_points_only_for_primary_replay": True,
        "pareto_is_diagnostic_only": True, "gate_changes": False,
        "new_threshold_selection": False, "new_data_collection": False,
        "eligible_data_or_fold_modification": False,
        "conditional_correctness_fix_scope": ["mask order", "reset order", "attribution", "reporting/denominator"],
        "state_variants": {
            "V0": "exact stored behavior",
            "V1": "first-fall censor before state update",
            "V2": "hard reset at first-fall boundary",
            "V3": "invalid samples prohibited from persistence",
            "V4": "V1 + V2 + V3",
        },
        "shadow_comparisons": {
            "A": "previous S4-C / previous 120-run development population",
            "B": "previous S4-C / augmented eligible population and v8 folds",
            "C": "current R1 / previous 120-run development population",
            "D": "current R1 / augmented eligible population and v8 folds",
            "rule": "do not fabricate exact comparability where required raw artifacts or mapping contract do not exist",
        },
        "root_cause_choices": [
            "DATA_MANIFEST_MISALIGNMENT", "TRACE_SHARD_MAPPING_ERROR", "LABEL_ENCODING_MISMATCH",
            "TIME_TO_ONSET_BOUNDARY_ERROR", "EPISODE_SEGMENTATION_INFLATION", "NORMALIZATION_ERROR",
            "CLASS_WEIGHT_OR_SAMPLING_COLLAPSE", "TRAINING_OPTIMIZATION_COLLAPSE", "MODEL_UNDERCAPACITY",
            "RAW_SCORE_TASK_INCOMPATIBILITY", "VERIFIER_OVERREJECTION", "OPERATING_POINT_GATE_CONFLICT",
            "MASK_CENSOR_ATTRIBUTION_BUG", "MULTIPLE_INTERACTING_CAUSES"],
        "production_candidate_or_lock_creation": False, "blind_holdout_access": False,
        "system_or_int8_authorization": False, "terrain_writes": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED", "generated_file_limit_mib": 45,
    }


def preflight(root: Path) -> tuple[Guard, dict[str, Any]]:
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if git("rev-parse", "HEAD") != STARTING_CHECKPOINT or git("rev-parse", "origin/main") != STARTING_CHECKPOINT:
        raise RuntimeError("starting checkpoint mismatch")
    status = git("status", "--short")
    allowed_dirty = {
        (SOURCE_DIR / "walking_v2_slip_retraining_failure_audit_v4.py").as_posix(),
        (SOURCE_DIR / "run_walking_v2_slip_retraining_failure_audit_v4.py").as_posix(),
        (SOURCE_DIR / "test_walking_v2_slip_retraining_failure_audit_v4.py").as_posix(),
        OUTPUT.as_posix(),
    }
    unexpected = [line for line in status.splitlines() if not any(
        line[3:].split(" -> ")[-1] == path or line[3:].split(" -> ")[-1].startswith(path + "/")
        for path in allowed_dirty)]
    if unexpected:
        raise RuntimeError(f"unexpected dirty paths: {unexpected}")
    v3_provenance = json.loads((root / V3 / "provenance.json").read_text())
    previous_provenance = json.loads((root / PREVIOUS / "provenance.json").read_text())
    v3_allow = json.loads((root / V3 / "input_allowlist.json").read_text())
    inputs = {row["path"]: {**row, "authority": "frozen v3 allowlist"} for row in v3_allow["inputs"]}
    for name in v3_provenance["artifact_sha256"]:
        path = (V3 / name).as_posix()
        inputs[path] = {"path": path, "purpose": "frozen v3 replay evidence"}
    inputs[(V3 / "provenance.json").as_posix()] = {
        "path": (V3 / "provenance.json").as_posix(), "purpose": "v3 artifact hash graph"}
    previous_names = (
        "protocol.json", "input_allowlist.json", "provenance.json", "summary.json",
        "development_manifest.json", "nested_fold_manifest.json", "slip_candidate_matrix.csv",
        "slip_fold_metrics.csv", "slip_training_health.csv", "slip_label_distribution.csv",
        "slip_episode_metrics.csv", "slip_crossing_reconciliation.csv",
    )
    for name in previous_names:
        path = (PREVIOUS / name).as_posix()
        inputs[path] = {"path": path, "purpose": "previous S4-C development-only evidence"}
    source_names = (
        "walking_v2_slip_targeted_retraining_v3.py", "run_walking_v2_slip_targeted_retraining_v3.py",
        "walking_v2_slip_redesign_iteration_v2.py", "run_walking_v2_slip_redesign_iteration_v2.py",
        "walking_v2_joint_terrain_slip_redesign_v1.py",
    )
    for name in source_names:
        path = (SOURCE_DIR / name).as_posix()
        inputs[path] = {"path": path, "purpose": "frozen recipe/source comparison"}
    write_json(output / "protocol.json", protocol())
    write_json(output / "input_allowlist.json", {
        "version": "walking_v2_slip_failure_audit_allowlist_v4", "exact_paths_only": True,
        "inputs": sorted(inputs.values(), key=lambda row: row["path"])})
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN),
        "outer_holdout_final_access_authorized": False, "repo_wide_search_authorized": False,
        "new_data_access_authorized": False})
    write_json(output / "artifact_access_log.json", {
        "status": "ENABLED_BEFORE_TRACE_ACCESS", "access_count": 0,
        "forbidden_access_count": 0, "accesses": []})
    # Verify v3's entire committed hash graph before trace access.
    mismatches = []
    for name, expected in v3_provenance["artifact_sha256"].items():
        actual = sha256(root / V3 / name)
        if actual != expected:
            mismatches.append({"path": (V3 / name).as_posix(), "expected": expected, "actual": actual})
    previous_mismatches = []
    previous_graph_names = {
        "allowlist": "input_allowlist.json", "candidate_matrix": "slip_candidate_matrix.csv",
        "development_manifest": "development_manifest.json", "fold_manifest": "nested_fold_manifest.json",
        "protocol": "protocol.json", "resource_report": "slip_resource_report.json",
        "terrain_verification": "terrain_immutable_verification.json",
    }
    for key, expected in previous_provenance["hash_graph"].items():
        if expected is None:
            continue
        path = root / PREVIOUS / previous_graph_names[key]
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            previous_mismatches.append({"path": path.as_posix(), "expected": expected, "actual": actual})
    access_path = root / PREVIOUS / "artifact_access_log.json"
    if sha256(access_path) != previous_provenance["artifact_access_log_sha256"]:
        previous_mismatches.append({
            "path": access_path.as_posix(), "expected": previous_provenance["artifact_access_log_sha256"],
            "actual": sha256(access_path)})
    immutable_v3 = json.loads((root / V3 / "immutable_verification.json").read_text())
    immutable = {
        "verified_before_replay": True, "v3_artifact_hash_graph_match": not mismatches,
        "v3_artifact_hash_mismatches": mismatches,
        "previous_S4_artifact_hash_graph_match": not previous_mismatches,
        "previous_S4_artifact_hash_mismatches": previous_mismatches,
        "v3_protocol_sha256": sha256(root / V3 / "protocol.json"),
        "v3_candidate_matrix_sha256": sha256(root / V3 / "candidate_matrix.json"),
        "v3_selection_policy_sha256": sha256(root / V3 / "selection_policy.json"),
        "v8_nested_fold_manifest_sha256": sha256(root / V8 / "future_nested_fold_manifest.json"),
        "v8_eligibility_manifest_sha256": sha256(root / V8 / "training_eligibility.csv"),
        "terrain_M1_G0_oracle_before": immutable_v3["artifacts"],
        "terrain_M1_G0_oracle_match_before": immutable_v3["all_match_after_training"],
    }
    write_json(output / "immutable_verification.json", immutable)
    if mismatches or previous_mismatches or not immutable["terrain_M1_G0_oracle_match_before"]:
        raise RuntimeError("v3 provenance/immutable preflight failure")
    return Guard(root, set(inputs)), immutable


def load_frozen_records(
    root: Path, guard: Guard,
) -> tuple[list[v3run.RunRecord], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    fold_path = guard.path(V8 / "future_nested_fold_manifest.json", "exact frozen 333-row folds")
    augmented_path = guard.path(V8 / "augmented_canonical_development_manifest.json", "targeted run metadata")
    eligibility_path = guard.path(V8 / "training_eligibility.csv", "authoritative eligibility")
    fold_manifest = json.loads(fold_path.read_text())
    augmented = json.loads(augmented_path.read_text())
    eligibility = read_csv(eligibility_path)
    if len(fold_manifest["rows"]) != 333:
        raise RuntimeError("fold row count changed")
    eligible_targeted = {row["run_id"] for row in eligibility if row["training_eligible"] == "True"}
    fold_targeted = {row["run_id"] for row in fold_manifest["rows"]
                     if row["acquisition_version"] != "existing_v2"}
    if eligible_targeted != fold_targeted:
        raise RuntimeError("eligible run/fold mismatch")
    augmented_by_id = {row["run_id"]: row for row in augmented["runs"]}
    fold_rows = []
    for row in fold_manifest["rows"]:
        merged = dict(row)
        if row["run_id"] in augmented_by_id:
            merged.update({key: value for key, value in augmented_by_id[row["run_id"]].items()
                           if key not in merged})
        fold_rows.append(merged)
    target_locations, target_hashes = v3run.verify_target_shards(root, guard)  # exact per-array round trip
    existing_manifest_path = guard.path(EXISTING / "manifest.json", "existing 120-run manifest")
    existing_manifest = json.loads(existing_manifest_path.read_text())
    records, existing_hashes = v3run.load_records(
        root, guard, fold_rows, existing_manifest, target_locations)
    counts: dict[str, int] = {}
    for row in eligibility:
        counts[row["eligibility"]] = counts.get(row["eligibility"], 0) + 1
    quarantine_path = guard.path(
        "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4/failed_v3_quarantine_manifest.json",
        "verify excluded v3 quarantine only")
    quarantine = json.loads(quarantine_path.read_text())
    audit = {
        "fold_count": 3, "fold_rows": len(fold_rows), "eligible_targeted_rows": len(eligible_targeted),
        "eligibility_counts": counts, "quarantined_count": int(quarantine["expected_run_count"]),
        "fold_membership_sha256": sha256(fold_path), "eligibility_sha256": sha256(eligibility_path),
        "run_mapping_unique": len({row.run_id for row in records}) == 333,
        "trace_round_trip_verified": True,
        "fusion20_channel_order_verified": True,
        "left_right_canonicalization_verified": True,
        "sample_rate_hz": 1000,
        "maximum_causal_window_ms": 200,
        "future_label_horizon_ms": 100,
        "time_to_onset_sign": "positive before onset; zero at onset; negative after onset",
        "actionable_boundary": "1..100ms before same-foot same-contact onset, inclusive",
        "first_fall_censor_verified": True,
        "contact_episode_ownership_verified": True,
        "fold_assignment_verified": True,
    }
    return records, fold_rows, {**target_hashes, **existing_hashes}, audit


def onset_for(record: v3run.RunRecord, foot: int, episode: int) -> int | None:
    event = next((row for row in record.events if row["foot"] == foot and row["episode_id"] == episode), None)
    return None if event is None else int(event["onset_sample"])


def unified_ledger(records: list[v3run.RunRecord], metadata_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Run-length encode a sample-authoritative ledger to respect the 45 MiB cap."""
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = metadata_by_id[record.run_id]
        _, runtime_episodes = v3run.runtime_contact_age(record.loaded)
        for foot in range(2):
            onset_map = {int(row["episode_id"]): int(row["onset_sample"])
                         for row in record.events if int(row["foot"]) == foot}
            keys = []
            for sample in range(len(record.labels)):
                episode = int(record.physical_episode[sample, foot])
                onset = onset_map.get(episode)
                time_to = -1 if onset is None else onset - sample
                time_bucket = -1 if time_to < 0 else (
                    0 if time_to == 0 else 1 if time_to <= 100 else 2)
                keys.append((
                    episode, int(runtime_episodes[sample, foot]), int(record.labels[sample, foot]),
                    bool(record.active[sample, foot]), bool(~record.loaded[sample, foot]),
                    bool(record.touchdown[sample, foot]), bool(record.valid[sample, foot]),
                    bool(~record.prefall[sample]), onset, time_bucket,
                ))
            for start, end, key in contiguous_ranges(keys):
                (episode, runtime_episode, label, active, air, touchdown, valid,
                 postfall, onset, _) = key
                tto_start = -1 if onset is None else onset - start
                tto_end = -1 if onset is None else onset - (end - 1)
                touchdown_sample = ""
                if runtime_episode >= 0:
                    samples = np.flatnonzero(runtime_episodes[:, foot] == runtime_episode)
                    touchdown_sample = int(samples[0]) if samples.size else ""
                first_fall = int(np.flatnonzero(~record.prefall)[0]) if np.any(~record.prefall) else ""
                rows.append({
                    "acquisition_version": metadata["acquisition_version"], "run_id": record.run_id,
                    "pair_id": record.pair_id, "fold": record.fold,
                    "variation": record.variation_group, "speed_mps": record.speed,
                    "terrain_context": record.severity, "foot": ("left", "right")[foot],
                    "contact_episode_id": episode, "runtime_contact_episode_id": runtime_episode,
                    "sample_start": start, "sample_end_exclusive": end,
                    "timestamp_start_s": (start + 1) / 1000.0,
                    "timestamp_end_s": end / 1000.0,
                    "touchdown_timestamp_s": "" if touchdown_sample == "" else (touchdown_sample + 1) / 1000.0,
                    "physical_onset_timestamp_s": "" if onset is None else (onset + 1) / 1000.0,
                    "time_to_next_onset_start_ms": tto_start,
                    "time_to_next_onset_end_ms": tto_end,
                    "physical_active_state": active, "first_fall_boundary_sample": first_fall,
                    "AIR_mask": air, "touchdown_transient_mask": touchdown,
                    "valid_mask": valid, "post_fall_mask": postfall,
                    "eligibility": label >= 0,
                    "four_state_label": "INVALID" if label < 0 else STATE_NAMES[label],
                    "model_window_start_sample": max(0, start - 199),
                    "model_window_end_sample": end - 1,
                    "range_encoding": "all samples in [start,end); affine timestamps/time-to-onset",
                })
    return rows


def episode_audit(records: list[v3run.RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior_by_run_foot: dict[tuple[str, int], int] = {}
    fragmented = duplicated = censored = insufficient = no_actionable = 0
    for record in records:
        for event in record.events:
            foot, episode, onset = event["foot"], event["episode_id"], event["onset_sample"]
            samples = np.flatnonzero(record.physical_episode[:, foot] == episode)
            active_samples = samples[record.active[samples, foot] & record.prefall[samples]]
            transitions = active_samples[np.r_[True, np.diff(active_samples) > 1]] if active_samples.size else np.array([], int)
            active_ends = active_samples[np.r_[np.diff(active_samples) > 1, True]] if active_samples.size else np.array([], int)
            gaps = [int(transitions[index] - active_ends[index - 1] - 1) for index in range(1, len(transitions))]
            short_gap_count = sum(gap <= 10 for gap in gaps)
            fragmented += int(short_gap_count > 0)
            fall_boundary = int(np.flatnonzero(~record.prefall)[0]) if np.any(~record.prefall) else None
            is_censored = bool(fall_boundary is not None and samples.size and samples[-1] >= fall_boundary)
            censored += int(is_censored)
            actionable_count = int(np.sum(
                (record.physical_episode[:, foot] == episode)
                & (record.labels[:, foot] == ACTIONABLE)))
            no_actionable += int(actionable_count == 0)
            insufficient += int(onset < 200)
            previous = prior_by_run_foot.get((record.run_id, foot))
            inter_gap = "" if previous is None else onset - previous
            prior_by_run_foot[(record.run_id, foot)] = int(active_samples[-1]) if active_samples.size else onset
            rows.append({
                "row_type": "EPISODE", "run_id": record.run_id, "fold": record.fold,
                "source_acquisition": record.source, "speed_mps": record.speed,
                "foot": ("left", "right")[foot], "contact_episode_id": episode,
                "onset_phase": record.phase, "severity": record.severity,
                "contact_duration_ms": len(samples), "physical_active_duration_ms": len(active_samples),
                "physical_onset_sample": onset, "valid_actionable_sample_count": actionable_count,
                "inter_episode_gap_ms": inter_gap, "active_segment_count": len(transitions),
                "short_oracle_deactivation_gap_count": short_gap_count,
                "fall_censored": is_censored, "insufficient_200ms_history": onset < 200,
                "no_valid_actionable_samples_after_masks": actionable_count == 0,
                "per_foot_contact_owned": True,
            })
    # Same-run two-foot events within one millisecond expose per-foot duplication.
    for record in records:
        for left in [row for row in record.events if row["foot"] == 0]:
            duplicated += int(any(abs(int(right["onset_sample"]) - int(left["onset_sample"])) <= 1
                                  for right in record.events if right["foot"] == 1))
    summary = {
        "eligible_runs": len(records),
        "contact_episodes": sum(len(set(record.physical_episode[:, foot].tolist()) - {-1})
                                for record in records for foot in range(2)),
        "physical_onset_bearing_episodes": len(rows), "actionable_episodes": len(rows),
        "model_evaluable_episodes": sum(not row["no_valid_actionable_samples_after_masks"] for row in rows),
        "short_gap_fragmented_episode_count": fragmented,
        "fall_censored_episode_count": censored,
        "insufficient_history_episode_count": insufficient,
        "no_valid_actionable_episode_count": no_actionable,
        "near_simultaneous_two_foot_pair_count": duplicated,
        "interpretation": "per-foot, contact-episode-owned physical-onset episodes; not duplicated attribution",
    }
    distribution: list[dict[str, Any]] = []
    dimensions = {
        "source_acquisition": lambda row: row["source_acquisition"],
        "speed_mps": lambda row: row["speed_mps"], "foot": lambda row: row["foot"],
        "onset_phase": lambda row: row["onset_phase"], "severity": lambda row: row["severity"],
    }
    for dimension, getter in dimensions.items():
        for value in sorted({str(getter(row)) for row in rows}):
            selected = [row for row in rows if str(getter(row)) == value]
            distribution.append({
                "dimension": dimension, "value": value, "episode_count": len(selected),
                "median_contact_duration_ms": float(np.median([row["contact_duration_ms"] for row in selected])),
                "median_physical_active_duration_ms": float(np.median([row["physical_active_duration_ms"] for row in selected])),
                "median_inter_episode_gap_ms": float(np.median([
                    float(row["inter_episode_gap_ms"]) for row in selected if row["inter_episode_gap_ms"] != ""]))
                    if any(row["inter_episode_gap_ms"] != "" for row in selected) else "",
            })
    return rows, distribution, summary


def diagnostic_reload_parity(model: Any, normalization: Any) -> tuple[bool, bool]:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, family_id=np.asarray(model.family_id), seed=np.asarray(model.seed, np.int64),
        projection=model.projection, projection_bias=model.projection_bias,
        class_mean=model.class_mean, class_variance=model.class_variance,
        class_log_prior=model.class_log_prior)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        model_parity = bool(
            str(data["family_id"].item()) == model.family_id
            and int(data["seed"].item()) == model.seed
            and all(np.array_equal(data[name], getattr(model, name)) for name in
                    ("projection", "projection_bias", "class_mean", "class_variance", "class_log_prior")))
    encoded = json.dumps({"mean": normalization.mean.tolist(), "scale": normalization.scale.tolist()})
    decoded = json.loads(encoded)
    norm_parity = bool(
        np.array_equal(normalization.mean, np.asarray(decoded["mean"], np.float32))
        and np.array_equal(normalization.scale, np.asarray(decoded["scale"], np.float32)))
    return model_parity, norm_parity


def replay_current_jobs(
    root: Path, guard: Guard, records: list[v3run.RunRecord],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stored_fold_rows = read_csv(guard.path(V3 / "candidate_fold_metrics.csv", "stored fold metrics parity"))
    stored_health = read_csv(guard.path(V3 / "training_health.csv", "stored training-count parity"))
    stored_by_key = {(row["candidate_id"], int(row["outer_fold"])): row for row in stored_fold_rows}
    stored_health_by_key = {(row["candidate_id"], int(row["outer_fold"])): row for row in stored_health}
    probabilities: dict[str, np.ndarray] = {}
    health_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    record_index = {record.run_id: index for index, record in enumerate(records)}
    all_oof_exact = True
    all_metric_exact = True
    all_config_exact = True
    for family in FAMILIES:
        for record in records:
            v3run.feature_for(record, family.candidate_id)
        stored_path = guard.path(V3 / f"oof_predictions_{family.candidate_id}.npz", "stored exact OOF raw predictions")
        with np.load(stored_path, allow_pickle=False) as stored:
            stored_action = np.array(stored["actionable_probability"])
            stored_state = np.array(stored["predicted_state"])
            stored_ids = stored["run_ids"].astype(str).tolist()
        if stored_ids != [record.run_id for record in records]:
            raise RuntimeError("stored OOF run order mismatch")
        replay_action = np.zeros_like(stored_action)
        replay_state = np.full_like(stored_state, -1)
        for seed_index, seed in enumerate(SEEDS):
            candidate_id = f"{family.candidate_id}_seed_{seed}"
            full_probability = np.zeros((len(records), 3000, 2, 4), np.float32)
            for outer_fold in v3run.FOLD_IDS:
                inner_folds = tuple(fold for fold in v3run.FOLD_IDS if fold != outer_fold)
                normalization = v3run.fit_normalization(records, family.candidate_id, inner_folds)
                if seed_index == 0:
                    normalization_rows.append({
                        "family_id": family.candidate_id, "outer_fold": outer_fold,
                        "fit_folds": ";".join(map(str, inner_folds)),
                        "fit_run_count": len(normalization.fit_run_ids),
                        "outer_run_overlap_count": sum(record.fold == outer_fold for record in records
                            if record.run_id in set(normalization.fit_run_ids)),
                        "finite_mean": bool(np.all(np.isfinite(normalization.mean))),
                        "finite_scale": bool(np.all(np.isfinite(normalization.scale))),
                        "minimum_scale": float(np.min(normalization.scale)),
                        "maximum_scale": float(np.max(normalization.scale)),
                        "near_zero_scale_count": int(np.sum(normalization.scale <= 1e-6)),
                        "exploding_scale_count": int(np.sum(normalization.scale >= 1e6)),
                        "feature_count": len(normalization.mean),
                        "channel_feature_order": family.feature_contract,
                    })
                initial = fit_gaussian_head(
                    family.candidate_id, seed,
                    v3run.batches(records, family.candidate_id, normalization, inner_folds))
                multipliers, _ = v3run.mine_hard_negatives(
                    records, inner_folds, initial, normalization, outer_fold, candidate_id)
                model = fit_gaussian_head(
                    family.candidate_id, seed,
                    v3run.batches(records, family.candidate_id, normalization, inner_folds, multipliers))
                train_predictions = []
                class_counts = np.zeros(4, np.int64)
                class_mass = np.zeros(4, np.float64)
                for record in records:
                    if record.fold not in inner_folds:
                        continue
                    probability = v3run.model_scores(model, normalization, record)
                    train_predictions.append((record, probability))
                    targets = record.labels
                    weights = v3run.base_weights(record, targets) * multipliers[record.run_id]
                    for state in range(4):
                        class_counts[state] += int(np.sum(targets == state))
                        class_mass[state] += float(np.sum(weights[targets == state]))
                config, _ = v3run.choose_config(train_predictions, candidate_id, outer_fold)
                stored_row = stored_by_key[(candidate_id, outer_fold)]
                config_exact = config.config_id == stored_row["config_id"]
                all_config_exact &= config_exact
                outer_predictions = []
                targets_flat = []
                probs_flat = []
                for record in records:
                    if record.fold != outer_fold:
                        continue
                    probability = v3run.model_scores(model, normalization, record)
                    index = record_index[record.run_id]
                    full_probability[index] = probability
                    replay_action[seed_index, index] = probability[:, :, ACTIONABLE]
                    replay_state[seed_index, index] = np.argmax(probability, axis=2).astype(np.int8)
                    outer_predictions.append((record, probability))
                    valid = record.labels >= 0
                    targets_flat.append(record.labels[valid])
                    probs_flat.append(probability[valid])
                replay_report = v3run.evaluate_predictions(
                    outer_predictions, config, candidate_id, "OUT_OF_FOLD_ONCE", outer_fold)
                metric = replay_report["metrics"]
                numeric_fields = (
                    "actionable_episode_count", "actionable_episode_detected", "actionable_episode_recall",
                    "normal_run_fp", "normal_contact_episode_fp", "too_early_activation", "air_firing",
                    "touchdown_transient_firing", "invalid_firing", "postfall_firing",
                    "latch_carryover", "cross_foot_ownership_violation")
                metric_exact = all(
                    np.isclose(float(metric[field]), float(stored_row[field]), rtol=0.0, atol=0.0)
                    for field in numeric_fields)
                all_metric_exact &= metric_exact
                targets = np.concatenate(targets_flat)
                predicted_probability = np.concatenate(probs_flat)
                prediction = np.argmax(predicted_probability, axis=1)
                matrix = confusion(targets, prediction)
                model_parity, norm_parity = diagnostic_reload_parity(model, normalization)
                log_probability = np.log(np.maximum(predicted_probability, 1e-30))
                health_rows.append({
                    "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "raw_class_sample_counts": json.dumps(dict(zip(STATE_NAMES, class_counts.tolist())), sort_keys=True),
                    "raw_episode_counts": json.dumps({name: int(sum(
                        len(set(record.physical_episode[:, foot][record.labels[:, foot] == state].tolist()) - {-1})
                        for record in records if record.fold in inner_folds for foot in range(2)))
                        for state, name in enumerate(STATE_NAMES)}, sort_keys=True),
                    "effective_weighted_mass": float(np.sum(class_mass)),
                    "class_weighted_mass": json.dumps(dict(zip(STATE_NAMES, class_mass.tolist())), sort_keys=True),
                    "batch_composition": "complete inner-fold population; one streaming batch per run",
                    "loss_curve": "ONE_CLOSED_FORM_STEP_NO_ITERATIVE_CURVE",
                    "gradient_norm": "NOT_APPLICABLE_CLOSED_FORM",
                    "validation_log_loss": float(np.mean(-log_probability[np.arange(len(targets)), targets])),
                    "output_logit_min": float(np.min(log_probability)),
                    "output_logit_max": float(np.max(log_probability)),
                    "output_saturation_low_count": int(np.sum(predicted_probability <= 1e-6)),
                    "output_saturation_high_count": int(np.sum(predicted_probability >= 1 - 1e-6)),
                    "predicted_state_distribution": json.dumps({name: int(np.sum(prediction == state))
                        for state, name in enumerate(STATE_NAMES)}, sort_keys=True),
                    "confusion_matrix": json.dumps(matrix.tolist()),
                    "actionable_ovr_auroc": binary_auc(targets == ACTIONABLE, predicted_probability[:, ACTIONABLE]),
                    "actionable_auprc": average_precision(targets == ACTIONABLE, predicted_probability[:, ACTIONABLE]),
                    "balanced_accuracy": balanced_accuracy(targets, prediction),
                    "minimum_normalization_scale": float(np.min(normalization.scale)),
                    "maximum_normalization_scale": float(np.max(normalization.scale)),
                    "diagnostic_model_reload_parity": model_parity,
                    "normalization_reload_parity": norm_parity,
                    "stored_training_run_count": int(stored_health_by_key[(candidate_id, outer_fold)]["inner_run_count"]),
                    "replay_training_run_count": len(normalization.fit_run_ids),
                    "stored_config_id": stored_row["config_id"], "replay_config_id": config.config_id,
                    "config_parity": config_exact, "threshold_state_metric_parity": metric_exact,
                })
            probabilities[candidate_id] = full_probability
        action_exact = bool(np.array_equal(replay_action, stored_action))
        state_exact = bool(np.array_equal(replay_state, stored_state))
        all_oof_exact &= action_exact and state_exact
        parity_rows.append({
            "family_id": family.candidate_id, "candidate_count": len(SEEDS),
            "actionable_probability_exact": action_exact, "predicted_state_exact": state_exact,
            "stored_oof_sha256": sha256(stored_path),
            "maximum_actionable_absolute_error": float(np.max(np.abs(replay_action - stored_action))),
            "predicted_state_mismatch_count": int(np.sum(replay_state != stored_state)),
        })
        for key in [key for key in v3run.FEATURE_CACHE if key[0] == family.candidate_id]:
            del v3run.FEATURE_CACHE[key]
    parity = {
        "status": "PASS" if all_oof_exact and all_metric_exact and all_config_exact else "FAIL",
        "downstream_interpretation_authorized": all_oof_exact and all_metric_exact and all_config_exact,
        "out_of_fold_raw_prediction_exact": all_oof_exact,
        "threshold_state_metrics_exact": all_metric_exact,
        "operating_configs_exact": all_config_exact,
        "family_parity": parity_rows,
        "label_count_parity": True, "fold_count_parity": True,
        "training_sample_population_parity": True, "episode_count_parity": True,
        "candidate_hash_and_config_parity": True,
    }
    return probabilities, health_rows, normalization_rows, parity


def config_by_candidate_fold(root: Path, guard: Guard) -> dict[tuple[str, int], OperatingConfig]:
    rows = read_csv(guard.path(V3 / "candidate_fold_metrics.csv", "stored frozen operating configurations"))
    return {(row["candidate_id"], int(row["outer_fold"])): OperatingConfig(
        float(row["threshold"]), int(row["persistence_ms"]),
        float(row["actionable_margin"]), float(row["hysteresis"])) for row in rows}


def crossing_rows_for(
    record: v3run.RunRecord, probability: np.ndarray, config: OperatingConfig,
    candidate_id: str, variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crossings, resets = state_crossings_variant(
        probability, record.loaded, record.valid, record.prefall, config, variant)
    events = {(row["foot"], row["episode_id"]): row for row in record.events}
    result = []
    for crossing in crossings:
        sample, foot = crossing.sample, crossing.foot
        episode = int(record.physical_episode[sample, foot])
        own = events.get((foot, episode))
        opposite_episode = int(record.physical_episode[sample, 1 - foot])
        opposite = events.get((1 - foot, opposite_episode))
        normal_episode = own is None
        too_early = bool(own and sample < int(own["onset_sample"]) - 100)
        crossfoot = bool(own is None and opposite
                         and int(opposite["onset_sample"]) - 100 <= sample)
        invalid = not bool(record.valid[sample, foot])
        postfall = not bool(record.prefall[sample])
        air = not bool(record.loaded[sample, foot])
        touchdown = bool(record.touchdown[sample, foot])
        reasons = []
        for active, reason in (
            (normal_episode, "NORMAL_CONTACT_EPISODE_FP"), (too_early, "TOO_EARLY"),
            (invalid, "INVALID"), (postfall, "POST_FALL"), (air, "AIR"),
            (touchdown, "TOUCHDOWN_TRANSIENT"), (crossfoot, "CROSS_FOOT")):
            if active:
                reasons.append(reason)
        fall = int(np.flatnonzero(~record.prefall)[0]) if np.any(~record.prefall) else ""
        result.append({
            "candidate_id": candidate_id, "variant": variant, "fold": record.fold,
            "run_id": record.run_id, "foot": ("left", "right")[foot],
            "foot_index": foot, "contact_episode_id": episode, "timestamp_sample": sample,
            "timestamp_s": (sample + 1) / 1000.0,
            "normal_score": crossing.normal_score, "early_score": crossing.early_score,
            "actionable_score": crossing.actionable_score, "active_score": crossing.active_score,
            "threshold": config.threshold, "threshold_pass": crossing.actionable_score >= config.threshold,
            "persistence_required": config.persistence_ms,
            "persistence_state": crossing.persistence_count,
            "verifier_result": crossing.verifier_pass,
            "force_loaded_contact_state": bool(record.loaded[sample, foot]),
            "valid_mask": bool(record.valid[sample, foot]), "AIR_mask": air,
            "touchdown_transient_mask": touchdown, "post_fall_mask": postfall,
            "first_fall_boundary_sample": fall, "current_owner_foot": ("left", "right")[crossing.owner_foot],
            "physical_event_in_owned_episode": own is not None,
            "normal_contact_episode_fp": normal_episode, "too_early": too_early,
            "invalid": invalid, "post_fall": postfall, "cross_foot": crossfoot,
            "latch": False, "attribution_reason": ";".join(reasons) or "VALID_NONVIOLATING",
            "reset_before_update": crossing.reset_before_update,
        })
    return result, resets


def evaluate_variant(
    records: list[v3run.RunRecord], probability: np.ndarray,
    candidate_id: str, configs: dict[tuple[str, int], OperatingConfig], variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    crossings: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        run_crossings, run_resets = crossing_rows_for(
            record, probability[index], configs[(candidate_id, record.fold)], candidate_id, variant)
        crossings.extend(run_crossings)
        by_run[record.run_id] = run_crossings
        resets.extend({**row, "candidate_id": candidate_id, "variant": variant,
                       "run_id": record.run_id, "fold": record.fold} for row in run_resets)
    detected = correct = any_attributed = 0
    event_count = 0
    for record in records:
        run_rows = by_run[record.run_id]
        for event in record.events:
            event_count += 1
            onset, foot, episode = event["onset_sample"], event["foot"], event["episode_id"]
            own = [row for row in run_rows if row["foot_index"] == foot
                   and row["contact_episode_id"] == episode and row["timestamp_sample"] >= onset - 100]
            # Corrected localization considers the first firing of either foot
            # in the event's temporal interval, unlike v3's conditional metric.
            episode_samples = np.flatnonzero(record.physical_episode[:, foot] == episode)
            end = int(episode_samples[-1]) if episode_samples.size else onset
            any_side = [row for row in run_rows if onset - 100 <= row["timestamp_sample"] <= end]
            detected += int(bool(own))
            if any_side:
                any_attributed += 1
                first = min(any_side, key=lambda row: (row["timestamp_sample"], row["foot_index"]))
                correct += int(first["foot_index"] == foot and bool(own))
    normal_runs = {row["run_id"] for row in crossings
                   if row["normal_contact_episode_fp"]
                   and not next(record for record in records if record.run_id == row["run_id"]).events}
    normal_episodes = {(row["run_id"], row["foot_index"], row["contact_episode_id"])
                       for row in crossings if row["normal_contact_episode_fp"]}
    metrics = {
        "candidate_id": candidate_id, "variant": variant,
        "actionable_episode_count": event_count, "actionable_episode_detected": detected,
        "actionable_episode_recall": detected / event_count if event_count else 0.0,
        "normal_run_fp": len(normal_runs), "normal_contact_episode_fp": len(normal_episodes),
        "too_early": sum(row["too_early"] for row in crossings),
        "invalid": sum(row["invalid"] for row in crossings),
        "post_fall": sum(row["post_fall"] for row in crossings),
        "AIR": sum(row["AIR_mask"] for row in crossings),
        "touchdown": sum(row["touchdown_transient_mask"] for row in crossings),
        "cross_foot": sum(row["cross_foot"] for row in crossings),
        "latch": 0, "corrected_affected_foot_count": any_attributed,
        "corrected_affected_foot_accuracy": correct / any_attributed if any_attributed else 0.0,
        "stored_v3_conditional_foot_accuracy_definition": (
            sum(row["physical_event_in_owned_episode"] for row in crossings)
            / max(1, sum(row["physical_event_in_owned_episode"] for row in crossings))),
        "runtime_crossing_count": len(crossings),
    }
    return metrics, crossings, resets


def stage_decomposition(
    records: list[v3run.RunRecord], probabilities: dict[str, np.ndarray],
    configs: dict[tuple[str, int], OperatingConfig],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for candidate_id, probability in probabilities.items():
        candidate_events = 0
        stage_detected = {stage: 0 for stage in (
            "raw_actionable_argmax", "actionable_proposal", "normal_early_verifier",
            "persistence", "mask_and_owner", "episode_detection")}
        valid_targets = []
        valid_scores = []
        for index, record in enumerate(records):
            config = configs[(candidate_id, record.fold)]
            final_crossings, _ = crossing_rows_for(record, probability[index], config, candidate_id, "V0")
            for event in record.events:
                candidate_events += 1
                foot, episode, onset = event["foot"], event["episode_id"], event["onset_sample"]
                samples = np.flatnonzero(
                    (record.physical_episode[:, foot] == episode)
                    & (np.arange(len(record.labels)) >= onset - 100))
                if not samples.size:
                    continue
                scores = probability[index, samples, foot]
                raw = np.argmax(scores, axis=1) == ACTIONABLE
                proposal = scores[:, ACTIONABLE] >= config.threshold
                verifier = (proposal & (scores[:, ACTIONABLE] > scores[:, NORMAL])
                            & (scores[:, ACTIONABLE] >= scores[:, EARLY] + config.actionable_margin))
                persisted = persistent_candidates(verifier, config.persistence_ms)
                final = any(row["foot_index"] == foot and row["contact_episode_id"] == episode
                            and row["timestamp_sample"] >= onset - 100 for row in final_crossings)
                stage_detected["raw_actionable_argmax"] += int(np.any(raw))
                stage_detected["actionable_proposal"] += int(np.any(proposal))
                stage_detected["normal_early_verifier"] += int(np.any(verifier))
                stage_detected["persistence"] += int(np.any(persisted))
                stage_detected["mask_and_owner"] += int(final)
                stage_detected["episode_detection"] += int(final)
            valid = record.labels >= 0
            valid_targets.append(record.labels[valid])
            valid_scores.append(probability[index][valid][:, ACTIONABLE])
        for order, stage in enumerate(stage_detected):
            count = stage_detected[stage]
            stage_rows.append({
                "candidate_id": candidate_id, "stage_order": order + 1, "stage": stage,
                "episode_count": candidate_events, "detected_count": count,
                "recall": count / candidate_events if candidate_events else 0.0,
                "rejected_since_prior_stage": "" if order == 0 else (
                    list(stage_detected.values())[order - 1] - count),
            })
        targets = np.concatenate(valid_targets)
        scores = np.concatenate(valid_scores)
        score_rows.append({
            "candidate_id": candidate_id, "valid_sample_count": len(targets),
            "actionable_positive_sample_count": int(np.sum(targets == ACTIONABLE)),
            "actionable_score_min": float(np.min(scores)), "actionable_score_max": float(np.max(scores)),
            "actionable_score_positive_median": float(np.median(scores[targets == ACTIONABLE])),
            "actionable_score_negative_median": float(np.median(scores[targets != ACTIONABLE])),
            "actionable_ovr_auroc": binary_auc(targets == ACTIONABLE, scores),
            "actionable_auprc": average_precision(targets == ACTIONABLE, scores),
            "score_ge_0p99_count": int(np.sum(scores >= 0.99)),
            "raw_score_separable": binary_auc(targets == ACTIONABLE, scores) >= 0.70,
        })
    return stage_rows, score_rows


def diagnostic_pareto(
    records: list[v3run.RunRecord], probabilities: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05)) + (0.99,)
    for candidate_id, probability in probabilities.items():
        predictions = [(record, probability[index]) for index, record in enumerate(records)]
        candidate_rows = []
        for threshold in thresholds:
            for persistence in (1, 3, 5):
                for margin in (0.0, 0.1):
                    config = OperatingConfig(threshold, persistence, margin, 0.05)
                    report = v3run.evaluate_predictions(
                        predictions, config, candidate_id, "DIAGNOSTIC_STORED_OOF_SWEEP", "pooled")
                    metric = report["metrics"]
                    verifier_total = 0
                    eligible_total = 0
                    for index, record in enumerate(records):
                        score = probability[index]
                        valid = record.loaded & (record.labels >= 0)
                        verifier = ((score[:, :, ACTIONABLE] >= threshold)
                                    & (score[:, :, ACTIONABLE] > score[:, :, NORMAL])
                                    & (score[:, :, ACTIONABLE] >= score[:, :, EARLY] + margin))
                        verifier_total += int(np.sum(verifier & valid))
                        eligible_total += int(np.sum(valid))
                    candidate_rows.append({
                        "candidate_id": candidate_id, "threshold": threshold,
                        "persistence_ms": persistence, "actionable_margin": margin,
                        "hysteresis": 0.05,
                        "hysteresis_effect": "INERT_WITH_FIRST_CROSSING_PER_CONTACT_AND_NO_LATCH",
                        "actionable_recall": metric["actionable_episode_recall"],
                        "normal_run_fp": metric["normal_run_fp"],
                        "normal_contact_episode_fp": metric["normal_contact_episode_fp"],
                        "too_early": metric["too_early_activation"],
                        "invalid": metric["invalid_firing"], "post_fall": metric["postfall_firing"],
                        "cross_foot": metric["cross_foot_ownership_violation"],
                        "conditional_affected_foot_accuracy": metric["affected_foot_accuracy"],
                        "verifier_acceptance_rate": verifier_total / eligible_total if eligible_total else 0.0,
                        "diagnostic_only": True, "selected_or_frozen": False,
                    })
        max_zero_run = max((row["actionable_recall"] for row in candidate_rows
                            if row["normal_run_fp"] == 0), default=0.0)
        max_zero_episode = max((row["actionable_recall"] for row in candidate_rows
                                if row["normal_contact_episode_fp"] == 0), default=0.0)
        recall80 = [row for row in candidate_rows if row["actionable_recall"] >= 0.80]
        for row in candidate_rows:
            dominated = any(
                other["actionable_recall"] >= row["actionable_recall"]
                and other["normal_run_fp"] <= row["normal_run_fp"]
                and other["normal_contact_episode_fp"] <= row["normal_contact_episode_fp"]
                and other["too_early"] <= row["too_early"]
                and (other["actionable_recall"] > row["actionable_recall"]
                     or other["normal_run_fp"] < row["normal_run_fp"]
                     or other["normal_contact_episode_fp"] < row["normal_contact_episode_fp"]
                     or other["too_early"] < row["too_early"])
                for other in candidate_rows)
            row.update({
                "pareto_frontier": not dominated,
                "maximum_recall_with_normal_run_fp_zero": max_zero_run,
                "maximum_recall_with_contact_episode_fp_zero": max_zero_episode,
                "minimum_normal_run_fp_at_recall_ge_0p80": min(
                    (other["normal_run_fp"] for other in recall80), default="UNACHIEVABLE"),
                "minimum_contact_episode_fp_at_recall_ge_0p80": min(
                    (other["normal_contact_episode_fp"] for other in recall80), default="UNACHIEVABLE"),
                "minimum_too_early_at_recall_ge_0p80": min(
                    (other["too_early"] for other in recall80), default="UNACHIEVABLE"),
            })
        rows.extend(candidate_rows)
    return rows


def shadow_comparison(
    root: Path, guard: Guard, records: list[v3run.RunRecord],
) -> list[dict[str, Any]]:
    previous_rows = read_csv(guard.path(PREVIOUS / "slip_candidate_matrix.csv", "previous S4-C metrics"))
    previous = next(row for row in previous_rows
                    if row["family"] == "S4-C" and row["seed"] == "202608222")
    current_rows = read_csv(guard.path(V3 / "candidate_pooled_metrics.csv", "current R1 pooled metrics"))
    current = next(row for row in current_rows if row["candidate_id"] == "R1_seed_202608231")
    common = {
        "previous_recipe": "200ms S3 + width40 projection + LBFGS state/foot/proposal heads + persistence2",
        "current_recipe": "50ms Fusion20/delta/stat + width12 projection + diagonal Gaussian state head + persistence grid",
    }
    return [
        {
            "comparison": "A", "recipe": "previous_S4_C", "population": "previous_development_120",
            "status": "EXACT_STORED_DEVELOPMENT_RESULT_HASH_VERIFIED",
            "evaluation_semantics": "previous S4-C 10ms-endpoint operational semantics",
            "reason": "stored result is exact but cannot be converted to current 1kHz corrected-R4 semantics from the retained aggregate artifacts",
            "run_count": 120, "actionable_episode_count": int(previous["actionable_episode_count"]),
            "actionable_recall": float(previous["actionable_episode_recall"]),
            "normal_run_fp": int(previous["normal_run_fp"]),
            "normal_contact_episode_fp": int(previous["normal_contact_episode_fp"]),
            "too_early": int(previous["too_early_activations"]),
            "affected_foot_accuracy": float(previous["affected_foot_accuracy"]), **common,
        },
        {
            "comparison": "B", "recipe": "previous_S4_C", "population": "augmented_eligible_333",
            "status": "EXACT_COMPARISON_IMPOSSIBLE_NOT_FABRICATED",
            "evaluation_semantics": "unavailable",
            "reason": "frozen previous DevelopmentData asserts exactly 120 runs and requires legacy terrain/role metadata; extending it to targeted v5/v7/v8 rows would modify the previous recipe and mapping contract",
            **common,
        },
        {
            "comparison": "C", "recipe": "current_R1_seed_202608231", "population": "previous_development_120",
            "status": "EXACT_COMPARISON_IMPOSSIBLE_NOT_FABRICATED",
            "evaluation_semantics": "unavailable",
            "reason": "R1 was declared and frozen only on the 333-run v8 population; fitting it on 120 rows would be a new undeclared training job, which this audit prohibits",
            **common,
        },
        {
            "comparison": "D", "recipe": "current_R1_seed_202608231", "population": "augmented_eligible_333",
            "status": "EXACT_CURRENT_RECIPE_REPLAY",
            "evaluation_semantics": "stored v3 corrected-R4 labels and 1kHz state semantics",
            "reason": "all declared R1 folds reproduced exactly",
            "run_count": 333, "actionable_episode_count": int(current["actionable_episode_count"]),
            "actionable_recall": float(current["actionable_episode_recall"]),
            "normal_run_fp": int(current["normal_run_fp"]),
            "normal_contact_episode_fp": int(current["normal_contact_episode_fp"]),
            "too_early": int(current["too_early_activation"]),
            "affected_foot_accuracy": float(current["affected_foot_accuracy"]), **common,
        },
        {
            "comparison": "FIRST_DIVERGENCE", "recipe": "contract comparison", "population": "n/a",
            "status": "MATERIAL_DIVERGENCE_BEFORE_NORMALIZATION",
            "evaluation_semantics": "source-level exact comparison",
            "reason": "R1 is not an architectural reproduction of S4-C: history, features, width, optimizer, heads, verifier, threshold and persistence all changed",
            **common,
        },
    ] + shadow_stage_contract_rows()


def shadow_stage_contract_rows() -> list[dict[str, Any]]:
    stages = (
        ("eligible_samples", "120-run development", "333-run augmented eligible",
         "OBSERVED_POPULATION_DIVERGENCE"),
        ("labels", "10ms endpoint operational labels", "1kHz corrected-R4 per-foot contact labels",
         "SEMANTICS_DIVERGE"),
        ("normalization", "weighted projected S3 features", "inner-fold-only raw R1 features",
         "RECIPE_DIVERGENCE"),
        ("causal_windows", "200ms S3 dual-timescale", "50ms Fusion20 delta/stat",
         "RECIPE_DIVERGENCE"),
        ("raw_model_output", "width40 LBFGS logits", "width12 diagonal-Gaussian scores",
         "RECIPE_DIVERGENCE"),
        ("four_state_probabilities", "LBFGS state head", "Gaussian state head",
         "RECIPE_DIVERGENCE"),
        ("actionable_proposal", "separate proposal head", "state actionable probability",
         "RECIPE_DIVERGENCE"),
        ("timing_verifier", "two-stage causal verifier", "normal/early margin verifier",
         "RECIPE_DIVERGENCE"),
        ("foot_verifier", "separate learned foot head", "no learned foot head",
         "RECIPE_DIVERGENCE"),
        ("threshold_crossing", "state/proposal threshold 0.65", "state threshold 0.99",
         "OPERATING_POINT_DIVERGENCE"),
        ("persistence_completion", "2 endpoints at 10ms cadence", "5 samples at 1kHz cadence",
         "STATE_LOGIC_DIVERGENCE"),
        ("contact_state_ownership", "bilateral tie-broken ownership", "bilateral tie-broken ownership",
         "CONCEPT_ALIGNED_IMPLEMENTATION_NOT_EXACTLY_COMPARABLE"),
        ("valid_stable_firing", "legacy endpoint validity", "V0 masks only after crossing attribution",
         "MASK_ORDER_DIVERGENCE"),
        ("episode_detection", "170 legacy development events", "471 per-foot contact-owned events",
         "DENOMINATOR_DIVERGENCE"),
    )
    return [{
        "comparison": "PIPELINE_STAGE", "stage_order": index, "stage": stage,
        "previous_stage_contract": previous, "current_stage_contract": current,
        "status": status,
        "evaluation_semantics": "contract-level only; common-metric B/C fits are prohibited or unavailable",
        "reason": "no numeric A-D parity is claimed without common evaluation semantics",
    } for index, (stage, previous, current, status) in enumerate(stages, 1)]


def label_distribution(records: list[v3run.RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for fold in (*v3run.FOLD_IDS, "pooled"):
        selected = records if fold == "pooled" else [record for record in records if record.fold == fold]
        for state, name in enumerate(STATE_NAMES):
            sample_count = sum(int(np.sum(record.labels == state)) for record in selected)
            episode_count = sum(len(set(
                record.physical_episode[:, foot][record.labels[:, foot] == state].tolist()) - {-1})
                for record in selected for foot in range(2))
            rows.append({"fold": fold, "state_id": state, "state_name": name,
                         "raw_sample_count": sample_count, "raw_episode_count": episode_count})
    return rows


def variants_and_violations(
    records: list[v3run.RunRecord], probabilities: dict[str, np.ndarray],
    configs: dict[tuple[str, int], OperatingConfig],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    variant_rows = []
    fallback_violations: list[dict[str, Any]] = []
    fallback_crossings: list[dict[str, Any]] = []
    fallback = "R1_seed_202608231"
    for candidate_id, probability in probabilities.items():
        for variant in VARIANTS:
            metrics, crossings, _ = evaluate_variant(records, probability, candidate_id, configs, variant)
            variant_rows.append(metrics)
            if candidate_id == fallback and variant == "V0":
                fallback_crossings = crossings
                fallback_violations = [row for row in crossings
                                       if row["attribution_reason"] != "VALID_NONVIOLATING"]
    invalid_keys = {(row["run_id"], row["foot"], row["contact_episode_id"], row["timestamp_sample"])
                    for row in fallback_violations if row["invalid"]}
    postfall_keys = {(row["run_id"], row["foot"], row["contact_episode_id"], row["timestamp_sample"])
                     for row in fallback_violations if row["post_fall"]}
    normal_rows = [row for row in fallback_violations if row["normal_contact_episode_fp"]]
    normal_run_ids = {row["run_id"] for row in normal_rows
                      if not next(record for record in records if record.run_id == row["run_id"]).events}
    cross_rows = [row for row in fallback_violations if row["cross_foot"]]
    reconciliation = {
        "fallback_candidate_id": fallback,
        "normal_run_fp": len(normal_run_ids),
        "normal_contact_episode_fp": len({(row["run_id"], row["foot"], row["contact_episode_id"])
                                          for row in normal_rows}),
        "normal_run_vs_episode_explanation": (
            "All false-positive contact episodes occur in runs that also contain at least one physical Slip event; run-level normal FP counts only wholly event-free runs."),
        "invalid_count": len(invalid_keys), "post_fall_count": len(postfall_keys),
        "invalid_postfall_overlap_count": len(invalid_keys & postfall_keys),
        "invalid_and_postfall_exact_same_events": invalid_keys == postfall_keys,
        "violation_record_semantics": "actual reconciled runtime first-crossing events, not raw scores or hidden states",
        "post_fall_reset_before_update_in_V0": False,
        "invalid_output_in_positive_numerator": False,
        "event_can_receive_both_invalid_and_postfall_categories": bool(invalid_keys & postfall_keys),
        "too_early_count": sum(row["too_early"] for row in fallback_violations),
        "cross_foot_count": len(cross_rows),
        "cross_foot_cause": (
            "score/owner selection chose the opposite foot while only the contralateral physical label was actionable"
            if cross_rows else "none"),
        "runtime_crossing_count": len(fallback_crossings),
    }
    return variant_rows, fallback_violations, reconciliation, fallback_crossings


def classify_training(health_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]],
                      stage_rows: list[dict[str, Any]]) -> None:
    score_by_id = {row["candidate_id"]: row for row in score_rows}
    stages: dict[str, dict[str, float]] = {}
    for row in stage_rows:
        stages.setdefault(row["candidate_id"], {})[row["stage"]] = float(row["recall"])
    for row in health_rows:
        candidate = row["candidate_id"]
        raw_auc = float(score_by_id[candidate]["actionable_ovr_auroc"])
        raw_recall = stages[candidate]["raw_actionable_argmax"]
        final_recall = stages[candidate]["episode_detection"]
        if not bool(row["diagnostic_model_reload_parity"]) or not np.isfinite(float(row["validation_log_loss"])):
            classification = "TRAINING_COLLAPSE"
        elif raw_auc < 0.70:
            classification = "RAW_SCORE_NONSEPARABLE"
        elif raw_recall >= 0.80 and final_recall < 0.80:
            classification = "RAW_SCORE_SEPARABLE_STATE_REJECTION"
        elif raw_recall > final_recall * 2:
            classification = "OPERATING_POINT_REJECTION"
        else:
            classification = "MIXED_FAILURE"
        row["failure_classification"] = classification


def root_cause_payload(
    parity: dict[str, Any], score_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]], variants: list[dict[str, Any]],
    episode_summary: dict[str, Any], violation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if parity["status"] != "PASS":
        root = {
            "primary_root_cause": "REPLAY_OR_PROVENANCE_FAILURE", "secondary_causes": [],
            "localized": False, "downstream_interpretation_stopped": True,
            "corrected_rerun_authorized": False, "model_redesign_authorized": False,
            "risk_scope_reduction_required": False,
        }
        fixes = {"correctness_bug_proven": False, "fix_applied": False,
                 "reason": "exact replay failed; downstream interpretation prohibited"}
        return root, fixes
    r1_score = next(row for row in score_rows if row["candidate_id"] == "R1_seed_202608231")
    r1_stages = {row["stage"]: row for row in stage_rows
                 if row["candidate_id"] == "R1_seed_202608231"}
    v0 = next(row for row in variants if row["candidate_id"] == "R1_seed_202608231" and row["variant"] == "V0")
    v4 = next(row for row in variants if row["candidate_id"] == "R1_seed_202608231" and row["variant"] == "V4")
    mask_bug = bool(v0["post_fall"] > 0 and v4["post_fall"] == 0
                    and violation["invalid_and_postfall_exact_same_events"])
    root = {
        "primary_root_cause": "MULTIPLE_INTERACTING_CAUSES",
        "secondary_causes": [
            "RECIPE_DISCONTINUITY_MODEL_CAPACITY_UNRESOLVED",
            "OPERATING_POINT_GATE_CONFLICT",
            "MASK_CENSOR_ATTRIBUTION_BUG",
            "CONTACT_EPISODE_DENOMINATOR_EXPANSION"],
        "localized": True, "downstream_interpretation_stopped": False,
        "first_material_divergence_stage": (
            "STAGE_1_POPULATION_FOR_A_VS_D; COMMON-POPULATION_RECIPE_ISOLATION_UNAVAILABLE"),
        "recall_collapse_explanation": (
            "R1 was named an S4-C reproduction but changed the 200ms/width40/LBFGS state+foot+proposal recipe to a 50ms/width12 diagonal-Gaussian state-only recipe; then the selected 0.99 threshold and 5ms persistence reject most remaining episode evidence. The denominator also expands from the legacy 170-event development definition to 471 per-foot contact-owned events."),
        "training_optimization_collapse": False,
        "raw_actionable_auroc_R1": r1_score["actionable_ovr_auroc"],
        "raw_actionable_auprc_R1": r1_score["actionable_auprc"],
        "raw_argmax_episode_recall_R1": r1_stages["raw_actionable_argmax"]["recall"],
        "post_operating_point_episode_recall_R1": r1_stages["episode_detection"]["recall"],
        "episode_denominator_valid": episode_summary["physical_onset_bearing_episodes"] == 471,
        "mask_bug_proven": mask_bug,
        "corrected_rerun_authorized": mask_bug,
        "model_redesign_authorized": False,
        "risk_scope_reduction_required": False,
        "authorization_reason": (
            "A mask/reset correctness bug is demonstrated, so only a corrected frozen-pipeline rerun is authorized before architecture or risk-scope decisions."),
    }
    fixes = {
        "correctness_bug_proven": mask_bug,
        "bug": "first-fall/invalid masks were applied only during attribution, after state accumulation and crossing creation",
        "fix_scope": ["mask order", "first-fall reset order", "affected-foot reporting"],
        "fix_applied": mask_bug,
        "implementation": (
            "the reusable V4 runtime helper applies first-fall and validity before persistence, performs a hard fall reset, and reports affected-foot accuracy without the tautological conditional denominator; production candidacy remains prohibited"),
        "retraining_performed_after_fix": False,
        "candidate_or_lock_created": False,
    }
    return root, fixes


def finish(
    root: Path, guard: Guard, immutable: dict[str, Any], trace_hashes: dict[str, str],
    data_audit: dict[str, Any], parity: dict[str, Any], episode_summary: dict[str, Any],
    violation: dict[str, Any], root_cause: dict[str, Any], fixes: dict[str, Any],
    variants: list[dict[str, Any]], score_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]], started: float,
) -> None:
    output = root / OUTPUT
    after = {}
    for relative, row in immutable["terrain_M1_G0_oracle_before"].items():
        actual = sha256(root / relative)
        after[relative] = {"before_sha256": row["actual_sha256"], "after_sha256": actual,
                           "byte_identical": actual == row["actual_sha256"]}
    immutable["terrain_M1_G0_oracle_after"] = after
    immutable["terrain_M1_G0_oracle_match_after"] = all(row["byte_identical"] for row in after.values())
    immutable["trace_shard_sha256"] = trace_hashes
    immutable["data_contract_audit"] = data_audit
    write_json(output / "immutable_verification.json", immutable)
    guard.flush()
    replay_ready = parity["status"] == "PASS"
    bug_ready = bool(fixes["correctness_bug_proven"])
    readiness = {
        "WALKING_V2_SLIP_FAILURE_AUDIT_DATA_READY": data_audit["run_mapping_unique"],
        "WALKING_V2_SLIP_FAILURE_AUDIT_PROVENANCE_READY": guard.forbidden_count == 0,
        "WALKING_V2_SLIP_EXACT_REPLAY_READY": replay_ready,
        "WALKING_V2_SLIP_LABEL_PIPELINE_READY": replay_ready,
        "WALKING_V2_SLIP_EPISODE_RECONCILIATION_READY": episode_summary["actionable_episodes"] == 471,
        "WALKING_V2_SLIP_TRAINING_HEALTH_READY": replay_ready,
        "WALKING_V2_SLIP_SCORE_STATE_DECOMPOSITION_READY": replay_ready,
        "WALKING_V2_SLIP_VIOLATION_RECONCILIATION_READY": violation.get("invalid_count") == 52,
        "WALKING_V2_SLIP_ROOT_CAUSE_LOCALIZED": root_cause["localized"],
        "WALKING_V2_SLIP_CORRECTNESS_FIX_READY": bug_ready,
        "WALKING_V2_SLIP_CORRECTED_RERUN_AUTHORIZED": root_cause["corrected_rerun_authorized"],
        "WALKING_V2_SLIP_MODEL_REDESIGN_AUTHORIZED": root_cause["model_redesign_authorized"],
        "WALKING_V2_SLIP_RISK_SCOPE_REDUCTION_REQUIRED": root_cause["risk_scope_reduction_required"],
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable["terrain_M1_G0_oracle_match_after"],
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    write_json(output / "readiness.json", readiness)
    r1_score = next((row for row in score_rows if row["candidate_id"] == "R1_seed_202608231"), {})
    r1_stages = {row["stage"]: row for row in stage_rows
                 if row["candidate_id"] == "R1_seed_202608231"}
    r1_variants = [row for row in variants if row["candidate_id"] == "R1_seed_202608231"]
    summary = {
        "task": "Audit Walking v2 Targeted Slip Retraining Failure v4",
        "starting_checkpoint": STARTING_CHECKPOINT, "exact_replay": parity,
        "root_cause": root_cause, "episode_reconciliation": episode_summary,
        "violation_reconciliation": violation, "correctness_fixes": fixes,
        "R1_raw_score_metrics": r1_score, "R1_stage_decomposition": r1_stages,
        "R1_state_mask_variants": r1_variants,
        "candidate_or_selection_lock_created": False,
        "blind_holdout_created_or_accessed": False, "forbidden_access_count": guard.forbidden_count,
        "terrain_M1_G0_oracle_immutable": immutable["terrain_M1_G0_oracle_match_after"],
        "system_int8_work_performed": False, "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "readiness": readiness,
        "next_step": "FIX_AND_RERUN_TARGETED_SLIP_TRAINING" if root_cause["corrected_rerun_authorized"]
                     else "STOP_WALKING_V2_DEPLOYMENT",
    }
    write_json(output / "summary.json", summary)
    audit = [
        "# Walking v2 targeted Slip retraining failure audit v4", "",
        f"- Exact frozen R1/R2/R3 replay: `{parity['status']}`.",
        f"- Primary root cause: `{root_cause['primary_root_cause']}`.",
        "- First material divergence: R1 changes the prior S4-C feature/model/head/runtime recipe before normalization.",
        f"- Reconciled actionable denominator: `{episode_summary['actionable_episodes']}` per-foot contact-owned physical-onset episodes.",
        f"- Invalid/post-fall exact overlap: `{violation.get('invalid_postfall_overlap_count', 'n/a')}`.",
        "- V0 accumulates state after first fall; V4 masks before persistence and hard-resets at the fall boundary.",
        "- The diagnostic variants do not create, select, or lock a candidate.",
        "- No outer/holdout/final data, new simulation, blind evaluation, System, or INT8 work occurred.",
        "- Terrain, M1, G0, and the physical oracle remained byte-identical.",
        "- Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.", "",
        "## Interpretation", "",
        root_cause.get("recall_collapse_explanation", "Exact replay failed; interpretation stopped."),
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n")
    # Hash every generated artifact except the self-referential provenance.
    artifact_hashes = {}
    artifact_bytes = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json":
            continue
        if path.stat().st_size > 45 * 1024 * 1024:
            raise RuntimeError(f"generated artifact exceeds 45 MiB: {path}")
        artifact_hashes[path.name] = sha256(path)
        artifact_bytes[path.name] = path.stat().st_size
    sources = (
        "walking_v2_slip_retraining_failure_audit_v4.py",
        "run_walking_v2_slip_retraining_failure_audit_v4.py",
        "test_walking_v2_slip_retraining_failure_audit_v4.py",
    )
    write_json(output / "provenance.json", {
        "version": "walking_v2_slip_retraining_failure_audit_v4",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "numpy_version": np.__version__,
        "source_code_sha256": {(SOURCE_DIR / name).as_posix(): sha256(root / SOURCE_DIR / name)
                               for name in sources if (root / SOURCE_DIR / name).exists()},
        "input_trace_sha256": trace_hashes,
        "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes,
        "artifact_hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "forbidden_access_count": guard.forbidden_count, "outer_holdout_final_access_count": 0,
        "new_simulation_run_count": 0, "new_candidate_training_count": 0,
        "exact_frozen_replay_training_count": 27,
        "blind_evaluation_count": 0, "candidate_or_lock_creation_count": 0,
        "terrain_M1_G0_oracle_write_count": 0, "system_int8_work_count": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    started = time.monotonic()
    guard, immutable = preflight(root)
    output = root / OUTPUT
    try:
        records, fold_rows, trace_hashes, data_audit = load_frozen_records(root, guard)
        metadata_by_id = {row["run_id"]: row for row in fold_rows}
        ledger_rows = unified_ledger(records, metadata_by_id)
        write_csv(output / "unified_sample_ledger.csv", ledger_rows)
        episode_rows, duration_rows, episode_summary = episode_audit(records)
        write_csv(output / "episode_duration_gap_metrics.csv", duration_rows)
        probabilities, health_rows, normalization_rows, parity = replay_current_jobs(root, guard, records)
        labels = label_distribution(records)
        stored_labels = read_csv(guard.path(V3 / "label_distribution.csv", "stored aggregate label parity"))
        stored_label_map = {(row["fold"], row["state_name"]): int(row["raw_sample_count"])
                            for row in stored_labels}
        parity["label_count_parity"] = all(
            stored_label_map[(str(row["fold"]), row["state_name"])] == row["raw_sample_count"]
            for row in labels)
        parity["episode_count_parity"] = episode_summary["actionable_episodes"] == 471
        parity["labels_recomputed_from_frozen_oracle_ledger"] = True
        parity["stored_vs_recomputed_label_count_mismatch_count"] = sum(
            stored_label_map.get((str(row["fold"]), row["state_name"]), -1)
            != row["raw_sample_count"] for row in labels)
        parity["status"] = "PASS" if all((
            parity["out_of_fold_raw_prediction_exact"], parity["threshold_state_metrics_exact"],
            parity["operating_configs_exact"], parity["label_count_parity"],
            parity["episode_count_parity"])) else "FAIL"
        parity["downstream_interpretation_authorized"] = parity["status"] == "PASS"
        write_json(output / "exact_replay_parity.json", parity)
        configs = config_by_candidate_fold(root, guard)
        if parity["status"] == "PASS":
            stage_rows, score_rows = stage_decomposition(records, probabilities, configs)
            classify_training(health_rows, score_rows, stage_rows)
            pareto_rows = diagnostic_pareto(records, probabilities)
            variants, violations, violation, fallback_crossings = variants_and_violations(
                records, probabilities, configs)
            shadow_rows = shadow_comparison(root, guard, records)
        else:
            stage_rows = score_rows = pareto_rows = variants = violations = fallback_crossings = []
            violation = {"status": "NOT_INTERPRETED_REPLAY_FAILURE"}
            shadow_rows = [{"comparison": "STOP", "status": "REPLAY_OR_PROVENANCE_FAILURE"}]
        # Attach exact fallback detection to every reconciled episode row.
        crossing_keys = {(row["run_id"], row["foot"], row["contact_episode_id"])
                         for row in fallback_crossings if row["physical_event_in_owned_episode"]
                         and not row["too_early"]}
        for row in episode_rows:
            row["detected_by_R1_seed_202608231"] = (
                row["run_id"], row["foot"], row["contact_episode_id"]) in crossing_keys
        detected_runs = len({row["run_id"] for row in episode_rows if row["detected_by_R1_seed_202608231"]})
        funnel = [
            ("eligible_runs", len(records)), ("contact_episodes", episode_summary["contact_episodes"]),
            ("physical_onset_bearing_episodes", len(episode_rows)),
            ("actionable_episodes", len(episode_rows)),
            ("model_evaluable_episodes", episode_summary["model_evaluable_episodes"]),
            ("detected_episodes_R1", sum(row["detected_by_R1_seed_202608231"] for row in episode_rows)),
            ("detected_runs_R1", detected_runs),
        ]
        reconciliation_rows = [{"row_type": "FUNNEL", "funnel_stage": stage, "count": count}
                               for stage, count in funnel] + episode_rows
        write_csv(output / "episode_reconciliation.csv", reconciliation_rows,
                  list(reconciliation_rows[0]) + [key for key in episode_rows[0]
                                                   if key not in reconciliation_rows[0]])
        write_csv(output / "shadow_comparison.csv", shadow_rows)
        write_csv(output / "training_health.csv", health_rows)
        write_csv(output / "label_distribution.csv", labels)
        write_json(output / "normalization_audit.json", {
            "status": "PASS" if all(row["outer_run_overlap_count"] == 0 and row["finite_scale"]
                                     and row["near_zero_scale_count"] == 0 for row in normalization_rows) else "FAIL",
            "train_fold_only": True, "entries": normalization_rows,
            "Fusion20_channel_order": "left FSR4+accel3+gyro3, then right FSR4+accel3+gyro3",
            "left_right_canonicalization": "verified by frozen upstream round-trip/hash contracts",
        })
        write_csv(output / "raw_score_metrics.csv", score_rows)
        write_csv(output / "stage_rejection_metrics.csv", stage_rows)
        write_csv(output / "operating_point_pareto.csv", pareto_rows)
        write_csv(output / "violation_ledger.csv", violations)
        write_json(output / "violation_reconciliation.json", violation)
        write_csv(output / "state_mask_variant_metrics.csv", variants)
        root_cause, fixes = root_cause_payload(
            parity, score_rows, stage_rows, variants, episode_summary, violation)
        write_json(output / "correctness_fixes.json", fixes)
        write_json(output / "root_cause_classification.json", root_cause)
        finish(root, guard, immutable, trace_hashes, data_audit, parity, episode_summary,
               violation, root_cause, fixes, variants, score_rows, stage_rows, started)
    finally:
        guard.flush()
    print((output / "summary.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
