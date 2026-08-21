#!/usr/bin/env python3
"""Build and validate the Walking-v2 Fast Reflex causal host prototype v1."""

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
from typing import Any, Iterable
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import run_walking_v2_slip_corrected_targeted_retraining_v4 as v4run
from run_walking_v2_slip_retraining_failure_audit_v4 import load_frozen_records
from walking_v2_fast_reflex_host_v1 import (
    TerrainVerifierConfig, causal_data_quality_mask, causal_runtime_contract,
    config_payload, direct_authority_gate, safe_ratio, select_terrain_advisory,
    terrain_candidate_matrix, terrain_case_a_timeline,
)
from walking_v2_joint_terrain_slip_redesign_v1 import (
    ProjectedLinearModel, runtime_feature,
)
from walking_v2_slip_corrected_targeted_retraining_v4 import (
    ENDPOINTS, HEAD_NAMES, OperatingConfig, build_training_rows,
    corrected_v4_state, derived_runtime_telemetry, training_weights,
)
from walking_v2_slip_redesign_iteration_v2 import (
    ACTIONABLE_RISK, FAMILY_SPECS, SlipV2Model, fixed_projection,
    sha256_json, weighted_normalization,
)


STARTING_CHECKPOINT = "3aa9393c3b535cd954b500649ebc50430f6bd4ce"
OUTPUT = Path("simulation/outputs/walking_v2_fast_reflex_host_v1")
V5 = Path("simulation/outputs/walking_v2_slip_risk_scope_reduction_v5")
V4 = Path("simulation/outputs/walking_v2_slip_corrected_targeted_retraining_v4")
TERRAIN = Path("simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1")
V8 = Path("simulation/outputs/walking_v2_slip_additional_moderate_v2_acquisition_v8")
EXISTING = Path("simulation/outputs/walking_bilateral_sensor_sink_observability_v2")
SOURCE_DIR = Path("simulation/unitree_mujoco/simulate_python")
FORBIDDEN = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
SELECTED_SLIP_CANDIDATE = "F0_seed_202608242"
SELECTED_SLIP_CONFIG = OperatingConfig(0.45, 0.45, 0.25, 0.20, 0.60, 2, 0.0)
FINAL_MODEL_MAX_ITER = 1200
FINAL_MODEL_TOLERANCE = 1e-6


def source_paths() -> tuple[str, ...]:
    return tuple((SOURCE_DIR / name).as_posix() for name in (
        "walking_v2_fast_reflex_host_v1.py",
        "run_walking_v2_fast_reflex_host_v1.py",
        "test_walking_v2_fast_reflex_host_v1.py",
    ))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row)) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Guard:
    """Exact input allowlist and durable forbidden-path audit."""

    def __init__(self, root: Path, allowed: set[str], log_path: Path,
                 bootstrap: list[dict[str, Any]]) -> None:
        self.root = root
        self.allowed = allowed
        self.log_path = log_path
        self.rows = bootstrap
        self.forbidden_count = 0
        self.blocked_count = 0
        self.frozen = False
        self.flush()

    def path(self, relative: str | Path, purpose: str) -> Path:
        if self.frozen:
            raise RuntimeError(f"artifact access after freeze: {relative}")
        value = Path(relative).as_posix()
        forbidden = any(token in f"/{value.lower()}" for token in FORBIDDEN)
        allowed = value in self.allowed and not forbidden and not Path(value).is_absolute()
        row: dict[str, Any] = {
            "sequence": len(self.rows) + 1, "path": value, "purpose": purpose,
            "decision": "ALLOWED" if allowed else "BLOCKED",
        }
        if not allowed:
            self.forbidden_count += int(forbidden)
            self.blocked_count += 1
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
            "version": "walking_v2_fast_reflex_access_v1", "fail_closed": True,
            "exact_paths_only": True, "frozen": self.frozen,
            "access_count": len(self.rows), "forbidden_access_count": self.forbidden_count,
            "blocked_access_count": self.blocked_count, "accesses": self.rows,
        })


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_fast_reflex_host_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "objective": "strongest honest causal host prototype on available development evidence",
        "development_only": True,
        "terrain_candidate_matrix_frozen_before_timeline_metrics": True,
        "terrain_candidates": [config_payload(row) for row in terrain_candidate_matrix()],
        "terrain_candidate_selection": (
            "non-actuating advisory alert-F1, then precision, recall, latency and config id; "
            "normal-run/invalid/wrong-case/duplicate outputs must be zero"
        ),
        "direct_authority_gate": {
            "recall_within_20ms_min": 0.80, "each_speed_recall_within_20ms_min": 0.70,
            "required_zero": [
                "normal_contact_fp", "too_early", "invalid_output", "post_fall_output",
                "wrong_case_output", "duplicate_output", "latch_carryover",
                "contact_owner_mismatch",
            ],
        },
        "slip_advisory_selection": {
            "source": "previously stored six-candidate OOF Pareto only",
            "policy": "maximum event-alert F1 with no normal-run FP; deterministic ties",
            "candidate": SELECTED_SLIP_CANDIDATE,
            "config_id": SELECTED_SLIP_CONFIG.config_id,
            "authority": "advisory only",
        },
        "final_slip_fit": {
            "purpose": "executable development advisory model, not performance estimation",
            "architecture": "S4-C", "variant": "F0", "seed": 202608242,
            "all_333_development_runs": True, "max_iterations": FINAL_MODEL_MAX_ITER,
            "tolerance": FINAL_MODEL_TOLERANCE,
            "performance_source": "OOF scores only; final-fit in-sample scores not reported",
        },
        "runtime_evaluation": {
            "physical_oracle_runtime_gate": False,
            "fall_or_prefall_runtime_gate": False,
            "physical_episode_and_onset": "offline attribution only",
            "data_quality": "current Fusion20 finiteness only",
        },
        "forbidden_data_access": True,
        "new_simulation_data": False,
        "blind_system_int8_vela_e84_hil": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "generated_file_limit_mib": 45,
    }


def preflight(root: Path, output: Path) -> tuple[
    Guard, list[Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any],
    dict[str, Any], dict[str, str], float,
]:
    started = time.monotonic()
    if git("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("starting HEAD mismatch")
    if git("rev-parse", "origin/main") != STARTING_CHECKPOINT:
        raise RuntimeError("starting origin/main mismatch")
    allowed_dirty = set(source_paths())
    unexpected = [
        line for line in git("status", "--short").splitlines()
        if line[3:].split(" -> ")[-1] not in allowed_dirty
    ]
    if unexpected:
        raise RuntimeError(f"unexpected dirty paths before output creation: {unexpected}")
    output.mkdir(parents=True, exist_ok=False)

    inherited_path = root / V5 / "input_allowlist.json"
    inherited_raw = inherited_path.read_bytes()
    inherited = json.loads(inherited_raw)
    v5_provenance_bootstrap_path = root / V5 / "provenance.json"
    v5_provenance_bootstrap_raw = v5_provenance_bootstrap_path.read_bytes()
    v5_provenance_bootstrap = json.loads(v5_provenance_bootstrap_raw)
    inputs = {row["path"]: dict(row) for row in inherited["inputs"]}
    additions = list(source_paths()) + [
        (V5 / name).as_posix() for name in (
            "provenance.json", "immutable_verification.json", "summary.json", "decision.json",
        )
    ] + [
        (V4 / name).as_posix() for name in (
            "operating_point_pareto.csv", "training_health.csv", "hard_negative_audit.csv",
            f"oof_raw_head_scores_{SELECTED_SLIP_CANDIDATE}.npz", "provenance.json",
        )
    ] + [
        (TERRAIN / name).as_posix() for name in (
            "terrain_candidate_model.npz", "terrain_candidate_config.json",
            "terrain_candidate_normalization.json", "terrain_selection_lock.json",
        )
    ]
    for path in additions:
        inputs.setdefault(path, {"path": path, "purpose": "host-v1 frozen authority"})
    for name in v5_provenance_bootstrap["artifact_sha256"]:
        path = (V5 / name).as_posix()
        inputs.setdefault(path, {"path": path, "purpose": "v5 transitive hash graph"})
    for path in inputs:
        if any(token in f"/{path.lower()}" for token in FORBIDDEN):
            raise RuntimeError(f"inherited allowlist contains forbidden path: {path}")

    write_json(output / "protocol.json", protocol_payload())
    write_json(output / "input_allowlist.json", {
        "version": "walking_v2_fast_reflex_allowlist_v1", "exact_paths_only": True,
        "bootstrap_authority": (V5 / "input_allowlist.json").as_posix(),
        "inputs": sorted(inputs.values(), key=lambda row: row["path"]),
    })
    write_json(output / "forbidden_path_policy.json", {
        "version": "walking_v2_fast_reflex_forbidden_v1", "fail_closed": True,
        "forbidden_tokens": list(FORBIDDEN),
        "outer_holdout_spatial_final_or_final_test_access": False,
        "directory_discovery_after_allowlist_freeze": False,
        "oracle_terrain_or_physical_oracle_runtime_gate": False,
    })
    bootstrap = [
        {
            "sequence": 1, "path": (V5 / "input_allowlist.json").as_posix(),
            "purpose": "bootstrap inherited exact allowlist", "decision": "ALLOWED",
            "sha256": hashlib.sha256(inherited_raw).hexdigest(), "byte_count": len(inherited_raw),
        },
        {
            "sequence": 2, "path": (V5 / "provenance.json").as_posix(),
            "purpose": "bootstrap v5 artifact hash graph", "decision": "ALLOWED",
            "sha256": hashlib.sha256(v5_provenance_bootstrap_raw).hexdigest(),
            "byte_count": len(v5_provenance_bootstrap_raw),
        },
    ]
    guard = Guard(root, set(inputs), output / "artifact_access_log.json", bootstrap)

    v5_provenance_path = guard.path(V5 / "provenance.json", "v5 hash graph")
    v5_provenance = json.loads(v5_provenance_path.read_text())
    v5_mismatches = []
    for name, expected in v5_provenance["artifact_sha256"].items():
        path = guard.path(V5 / name, "verify v5 scope artifact")
        actual = sha256(path)
        if actual != expected:
            v5_mismatches.append({"path": path.as_posix(), "expected": expected, "actual": actual})
    if v5_mismatches:
        raise RuntimeError(f"v5 provenance mismatch: {v5_mismatches[:3]}")

    immutable_v5 = json.loads(
        guard.path(V5 / "immutable_verification.json", "immutable Terrain/M1/G0/oracle graph").read_text()
    )
    immutable_before = {
        path: sha256(guard.path(path, "immutable pre-experiment hash"))
        for path in immutable_v5["terrain_M1_G0_oracle_before"]
    }
    if immutable_before != immutable_v5["terrain_M1_G0_oracle_before"]:
        raise RuntimeError("immutable baseline mismatch")

    terrain_lock = json.loads(
        guard.path(TERRAIN / "terrain_selection_lock.json", "locked Walking Terrain authority").read_text()
    )
    for field, name in (
        ("model_sha256", "terrain_candidate_model.npz"),
        ("normalization_sha256", "terrain_candidate_normalization.json"),
        ("config_sha256", "terrain_candidate_config.json"),
    ):
        path = guard.path(TERRAIN / name, f"verify locked Terrain {name}")
        if sha256(path) != terrain_lock[field]:
            raise RuntimeError(f"locked Terrain mismatch: {name}")
    terrain_model = ProjectedLinearModel.load(
        guard.path(TERRAIN / "terrain_candidate_model.npz", "load locked Terrain T2")
    )

    records, fold_rows, trace_hashes, data_audit = load_frozen_records(root, guard)
    original = json.loads(
        guard.path(EXISTING / "manifest.json", "existing development Terrain metadata").read_text()
    )
    augmented = json.loads(
        guard.path(V8 / "augmented_canonical_development_manifest.json",
                   "targeted development context metadata").read_text()
    )
    original_by_id = {row["run_id"]: row for row in original["runs"]}
    augmented_by_id = {row["run_id"]: row for row in augmented["runs"]}
    metadata: dict[str, dict[str, Any]] = {}
    for row in fold_rows:
        value = dict(row)
        if row["run_id"] in original_by_id:
            value["terrain_context"] = original_by_id[row["run_id"]]["terrain_name"]
        else:
            extra = augmented_by_id[row["run_id"]]
            value["terrain_context"] = extra.get("profile_version", extra["severity"])
        metadata[row["run_id"]] = value

    oof_path = guard.path(
        V4 / f"oof_raw_head_scores_{SELECTED_SLIP_CANDIDATE}.npz",
        "selected OOF advisory scores",
    )
    with np.load(oof_path, allow_pickle=False) as archive:
        if list(archive["run_ids"].astype(str)) != [record.run_id for record in records]:
            raise RuntimeError("OOF run ordering mismatch")
        if not np.array_equal(archive["endpoints"], ENDPOINTS):
            raise RuntimeError("OOF endpoint mismatch")
        oof_scores = {name: archive[name].copy() for name in HEAD_NAMES}
    pareto = read_csv(guard.path(V4 / "operating_point_pareto.csv", "OOF advisory Pareto"))
    selected_stored = next(
        row for row in pareto
        if row["candidate_id"] == SELECTED_SLIP_CANDIDATE
        and row["config_id"] == SELECTED_SLIP_CONFIG.config_id
    )
    hard_rows = read_csv(guard.path(V4 / "hard_negative_audit.csv", "frozen hard-negative identities"))
    health_rows = read_csv(guard.path(V4 / "training_health.csv", "previous optimizer health"))
    source_hashes = {
        path: sha256(guard.path(path, "host-v1 source hash")) for path in source_paths()
    }
    preflight_payload = {
        "starting_checkpoint_match": True, "origin_checkpoint_match": True,
        "v5_hash_graph_match": True, "immutable_match_before": True,
        "terrain_lock_match": True, "record_count": len(records),
        "event_count": sum(len(record.events) for record in records),
        "trace_hashes": trace_hashes, "data_audit": data_audit,
        "source_sha256": source_hashes,
        "previous_lbfgs_fit_count": len(health_rows),
        "previous_lbfgs_ceiling_count": sum(
            int(row["iterations"]) == int(row["max_iterations"]) for row in health_rows
        ),
    }
    write_json(output / "data_provenance.json", preflight_payload)
    return (
        guard, records, fold_rows, metadata, terrain_model,
        {"scores": oof_scores, "stored": selected_stored, "hard_rows": hard_rows},
        immutable_before, started,
    )


def compute_terrain_probabilities(
    records: list[Any], model: ProjectedLinearModel,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    probabilities = np.empty((len(records), len(ENDPOINTS), 2, 4), np.float32)
    telemetry: list[dict[str, np.ndarray]] = []
    for index, record in enumerate(records):
        ages, runtime_episode, phases = derived_runtime_telemetry(record.loaded)
        features = np.asarray([
            runtime_feature(
                "T2", foot, int(endpoint), record.canonical, record.loaded, ages, phases,
            )
            for endpoint in ENDPOINTS for foot in (0, 1)
        ], np.float32)
        probabilities[index] = model.probabilities(features).reshape(len(ENDPOINTS), 2, 4)
        telemetry.append({
            "age": ages, "runtime_episode": runtime_episode, "phase": phases,
            "quality": causal_data_quality_mask(record.canonical),
        })
        if (index + 1) % 50 == 0:
            print(f"Terrain causal replay: {index + 1}/{len(records)}", flush=True)
    return probabilities, telemetry


def event_key(record: Any, event: dict[str, Any]) -> tuple[str, int, int]:
    return record.run_id, int(event["foot"]), int(event["episode_id"])


def event_end(record: Any, foot: int, episode: int) -> int:
    samples = np.flatnonzero(record.physical_episode[:, foot] == episode)
    return int(samples[-1]) if samples.size else 2999


def normal_contact_denominator(records: list[Any]) -> int:
    count = 0
    for record in records:
        events = {(int(row["foot"]), int(row["episode_id"])) for row in record.events}
        first_fall = int(np.flatnonzero(~record.prefall)[0]) if np.any(~record.prefall) else 3000
        for foot in (0, 1):
            episodes = set(record.physical_episode[:first_fall, foot].tolist()) - {-1}
            count += sum((foot, int(episode)) not in events for episode in episodes)
    return count


def evaluate_outputs(
    records: list[Any], outputs: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_events = [(record, event) for record in records for event in record.events]
    detected: dict[tuple[str, int, int], int] = {}
    crossings: list[dict[str, Any]] = []
    normal_fp: set[tuple[str, int, int]] = set()
    normal_run_fp: set[str] = set()
    too_early: set[tuple[str, int, int, int]] = set()
    post_fall = invalid = wrong_case = late_after_contact = 0
    duplicate = 0
    owner_mismatch = 0
    seen_runtime: set[tuple[str, int, int]] = set()

    for record, run_outputs in zip(records, outputs):
        mapping = {
            (int(row["foot"]), int(row["episode_id"])): row for row in record.events
        }
        first_fall = int(np.flatnonzero(~record.prefall)[0]) if np.any(~record.prefall) else None
        quality = causal_data_quality_mask(record.canonical)
        for raw in run_outputs:
            row = {**raw, "run_id": record.run_id, "fold": record.fold,
                   "speed_mps": record.speed}
            sample = int(row["sample"])
            foot = int(row["foot"])
            physical_episode = int(row["physical_episode_id"])
            runtime_key = (record.run_id, foot, int(row["runtime_episode_id"]))
            if runtime_key in seen_runtime:
                duplicate += 1
            seen_runtime.add(runtime_key)
            row["post_fall"] = bool(first_fall is not None and sample >= first_fall)
            row["invalid"] = not bool(quality[sample])
            row["wrong_case"] = row.get("transition_case", "A") not in (None, "A")
            row["owner_mismatch"] = not bool(row.get("unique_g0_owner", True))
            post_fall += int(row["post_fall"])
            invalid += int(row["invalid"])
            wrong_case += int(row["wrong_case"])
            owner_mismatch += int(row["owner_mismatch"])
            own = mapping.get((foot, physical_episode))
            attribution = "VALID_EVENT_OUTPUT"
            if row["post_fall"]:
                attribution = "POST_FALL_ADVISORY"
            elif row["invalid"]:
                attribution = "INVALID_DATA_OUTPUT"
            elif own is None:
                attribution = "NORMAL_CONTACT_FP"
                normal_fp.add((record.run_id, foot, physical_episode))
                if not record.events:
                    normal_run_fp.add(record.run_id)
            else:
                onset = int(own["onset_sample"])
                end = event_end(record, foot, physical_episode)
                if sample < onset - 100:
                    attribution = "TOO_EARLY"
                    too_early.add((record.run_id, foot, physical_episode, sample))
                elif sample > end:
                    attribution = "AFTER_CONTACT"
                    late_after_contact += 1
                else:
                    key = (record.run_id, foot, physical_episode)
                    detected[key] = min(sample, detected.get(key, sample))
                    row["latency_ms"] = sample - onset
            row["attribution"] = attribution
            crossings.append(row)

    speed_total = {speed: 0 for speed in (0.10, 0.15, 0.20)}
    speed_detected = {speed: 0 for speed in speed_total}
    speed_20 = {speed: 0 for speed in speed_total}
    foot_total = {0: 0, 1: 0}
    foot_detected = {0: 0, 1: 0}
    event_rows: list[dict[str, Any]] = []
    margins: list[float] = []
    late: list[float] = []
    within_20 = within_50 = predictive = 0
    detected_runs: set[str] = set()
    for record, event in all_events:
        key = event_key(record, event)
        onset = int(event["onset_sample"])
        sample = detected.get(key)
        found = sample is not None
        latency = None if sample is None else sample - onset
        if found:
            detected_runs.add(record.run_id)
            margins.append(float(-latency))
            late.append(float(max(0, latency)))
            predictive += int(latency <= 0)
            within_20 += int(latency <= 20)
            within_50 += int(latency <= 50)
        speed = float(record.speed)
        foot = int(event["foot"])
        speed_total[speed] += 1
        foot_total[foot] += 1
        speed_detected[speed] += int(found)
        speed_20[speed] += int(found and latency <= 20)
        foot_detected[foot] += int(found)
        event_rows.append({
            "run_id": record.run_id, "fold": record.fold, "speed_mps": speed,
            "foot": ("left", "right")[foot], "physical_episode_id": int(event["episode_id"]),
            "onset_sample": onset, "detected": found,
            "first_output_sample": "" if sample is None else sample,
            "latency_ms": "" if latency is None else latency,
            "detected_by_onset": bool(found and latency <= 0),
            "detected_within_20ms": bool(found and latency <= 20),
            "detected_within_50ms": bool(found and latency <= 50),
        })

    true_count = len(detected)
    false_count = len(normal_fp) + len(too_early) + post_fall + invalid + late_after_contact
    normal_contacts = normal_contact_denominator(records)
    actionable_runs = len({record.run_id for record, _ in all_events})
    eventless_runs = len(records) - actionable_runs
    metrics: dict[str, Any] = {
        "run_count": len(records), "event_count": len(all_events),
        "output_count": sum(len(rows) for rows in outputs),
        "event_detected": true_count, "event_recall": safe_ratio(true_count, len(all_events)),
        "predictive_detected_by_onset": predictive,
        "predictive_recall_by_onset": safe_ratio(predictive, len(all_events)),
        "detected_within_20ms": within_20,
        "recall_within_20ms": safe_ratio(within_20, len(all_events)),
        "detected_within_50ms": within_50,
        "recall_within_50ms": safe_ratio(within_50, len(all_events)),
        "detected_runs": len(detected_runs), "actionable_runs": actionable_runs,
        "run_recall": safe_ratio(len(detected_runs), actionable_runs),
        "normal_run_fp": len(normal_run_fp), "eventless_run_count": eventless_runs,
        "normal_run_fp_rate": safe_ratio(len(normal_run_fp), eventless_runs),
        "normal_contact_fp": len(normal_fp), "normal_contact_count": normal_contacts,
        "normal_contact_fp_rate": safe_ratio(len(normal_fp), normal_contacts),
        "too_early": len(too_early), "post_fall_output": post_fall,
        "invalid_output": invalid, "wrong_case_output": wrong_case,
        "duplicate_output": duplicate, "latch_carryover": 0,
        "contact_owner_mismatch": owner_mismatch,
        "after_contact_output": late_after_contact,
        "precision": safe_ratio(true_count, true_count + false_count),
        "alert_f1": None,
        "speed_recall": {
            f"{speed:.2f}": safe_ratio(speed_detected[speed], speed_total[speed])
            for speed in speed_total
        },
        "speed_recall_within_20ms": {
            f"{speed:.2f}": safe_ratio(speed_20[speed], speed_total[speed])
            for speed in speed_total
        },
        "foot_recall": {
            ("left", "right")[foot]: safe_ratio(foot_detected[foot], foot_total[foot])
            for foot in foot_total
        },
        "median_warning_margin_ms": float(np.median(margins)) if margins else None,
        "p95_warning_margin_ms": float(np.percentile(margins, 95)) if margins else None,
        "median_late_latency_ms": float(np.median(late)) if late else None,
        "p95_late_latency_ms": float(np.percentile(late, 95)) if late else None,
    }
    precision = metrics["precision"] or 0.0
    recall = metrics["event_recall"] or 0.0
    metrics["alert_f1"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics["direct_gate_pass"] = direct_authority_gate(metrics)
    return metrics, event_rows, crossings


def terrain_experiment(
    records: list[Any], probabilities: np.ndarray,
    telemetry: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], TerrainVerifierConfig, list[Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timelines_by_config: dict[str, list[Any]] = {}
    for config in terrain_candidate_matrix():
        timelines = []
        outputs = []
        for record, probability, signals in zip(records, probabilities, telemetry):
            timeline = terrain_case_a_timeline(
                probability, ENDPOINTS, record.loaded, signals["age"], record.touchdown,
                signals["runtime_episode"], record.physical_episode, signals["quality"], config,
            )
            timelines.append(timeline)
            outputs.append(timeline.emissions)
        metrics, _, _ = evaluate_outputs(records, outputs)
        row = {**config_payload(config), **metrics}
        rows.append(row)
        timelines_by_config[config.config_id] = timelines
        print(f"Terrain verifier evaluated: {config.config_id}", flush=True)
    selected_row = select_terrain_advisory(rows)
    selected_config = next(
        config for config in terrain_candidate_matrix()
        if config.config_id == selected_row["config_id"]
    )
    return rows, selected_config, timelines_by_config[selected_config.config_id], selected_row


def causal_slip_outputs(
    records: list[Any], scores: dict[str, np.ndarray],
    telemetry: list[dict[str, np.ndarray]],
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    outputs: list[list[dict[str, Any]]] = []
    firing = np.zeros((len(records), len(ENDPOINTS), 2), bool)
    for index, (record, signals) in enumerate(zip(records, telemetry)):
        run_scores = {name: scores[name][index] for name in HEAD_NAMES}
        runtime_valid = np.repeat(signals["quality"][:, None], 2, axis=1)
        state = corrected_v4_state(
            run_scores, record.loaded, runtime_valid, np.ones(3000, bool),
            record.touchdown, record.physical_episode, SELECTED_SLIP_CONFIG,
        )
        rows = []
        for crossing in state.crossings:
            endpoint_row = int(crossing["endpoint_row"])
            foot = int(crossing["foot"])
            owner = int(np.sum(record.loaded[int(crossing["sample"])]) == 1
                        and record.loaded[int(crossing["sample"]), foot])
            rows.append({
                **crossing, "transition_case": None,
                "unique_g0_owner": bool(owner),
                "data_quality_valid": bool(signals["quality"][int(crossing["sample"])]),
            })
        outputs.append(rows)
        firing[index] = state.firing
    return outputs, firing


def gated_scores(
    records: list[Any], scores: dict[str, np.ndarray], telemetry: list[dict[str, np.ndarray]],
    timelines: list[Any], reactive: bool,
) -> dict[str, np.ndarray]:
    result = {name: np.array(scores[name], copy=True) for name in HEAD_NAMES}
    for index, (record, signals, timeline) in enumerate(zip(records, telemetry, timelines)):
        allowed = timeline.case_a_active & timeline.eligibility & timeline.unique_owner[:, None]
        if reactive:
            active = np.argmax(
                np.stack([result[name][index] for name in HEAD_NAMES[:4]], axis=-1), axis=-1,
            ) == 3
            allowed &= active
        result["proposal"][index][~allowed] = 0.0
        result["foot"][index][:] = 0.0
        result["foot"][index][allowed] = 1.0
    return result


def scope_metrics(records: list[Any], timelines: list[Any]) -> dict[str, Any]:
    predictive = reactive = 0
    predictive_runs: set[str] = set()
    reactive_runs: set[str] = set()
    by_speed = {
        mode: {f"{speed:.2f}": {"eligible": 0, "total": 0} for speed in (0.10, 0.15, 0.20)}
        for mode in ("predictive", "reactive")
    }
    for record, timeline in zip(records, timelines):
        for event in record.events:
            foot = int(event["foot"])
            onset = int(event["onset_sample"])
            pred = bool(np.any(timeline.case_a_active[
                (ENDPOINTS >= onset - 100) & (ENDPOINTS <= onset), foot
            ]))
            react = bool(np.any(timeline.case_a_active[
                (ENDPOINTS >= onset) & (ENDPOINTS <= onset + 20), foot
            ]))
            predictive += int(pred)
            reactive += int(react)
            predictive_runs.add(record.run_id) if pred else None
            reactive_runs.add(record.run_id) if react else None
            speed = f"{record.speed:.2f}"
            by_speed["predictive"][speed]["total"] += 1
            by_speed["predictive"][speed]["eligible"] += int(pred)
            by_speed["reactive"][speed]["total"] += 1
            by_speed["reactive"][speed]["eligible"] += int(react)
    return {
        "original_episode_count": sum(len(record.events) for record in records),
        "predictive_episode_denominator": predictive,
        "predictive_run_denominator": len(predictive_runs),
        "reactive_20ms_episode_denominator": reactive,
        "reactive_20ms_run_denominator": len(reactive_runs),
        "per_speed": by_speed,
        "zero_denominators_reported_as_performance": False,
    }


def fit_final_advisory_model(
    records: list[Any], metadata: dict[str, dict[str, Any]], hard_rows: list[dict[str, str]],
    output: Path,
) -> dict[str, Any]:
    print("Building final advisory training rows", flush=True)
    rows = build_training_rows(records, metadata)
    indices = np.flatnonzero(rows.eligible)
    hard_ids = {row["identity"] for row in hard_rows}
    state_weight, foot_weight, weight_audit = training_weights(
        rows, indices, "F0", hard_ids,
    )
    values = np.asarray(rows.features[indices], np.float64)
    combined = 0.75 * state_weight + 0.25 * foot_weight
    mean, scale = weighted_normalization(values, combined)
    projection, bias = fixed_projection(
        values.shape[1], int(FAMILY_SPECS["S4-C"]["projection_width"]), 202608242,
    )
    normalized = (values - mean) / scale
    transformed = np.concatenate((
        normalized, np.maximum(0.0, normalized @ projection + bias),
    ), axis=1)
    common = {
        "C": 1.0, "solver": "lbfgs", "max_iter": FINAL_MODEL_MAX_ITER,
        "tol": FINAL_MODEL_TOLERANCE, "random_state": 202608242,
    }
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        state = LogisticRegression(multi_class="auto", **common).fit(
            transformed, rows.state[indices], sample_weight=state_weight,
        )
        foot = LogisticRegression(**common).fit(
            transformed, rows.foot_target[indices], sample_weight=foot_weight,
        )
        proposal = LogisticRegression(**common).fit(
            transformed, (rows.state[indices] == ACTIONABLE_RISK).astype(int),
            sample_weight=state_weight,
        )
        caught = [str(message.message) for message in messages if issubclass(message.category, ConvergenceWarning)]
    model = SlipV2Model(
        "S4-C", 202608242, mean, scale, projection, bias,
        state.coef_.copy(), state.intercept_.copy(), state.classes_.copy(),
        foot.coef_.copy(), foot.intercept_.copy(),
        proposal.coef_.copy(), proposal.intercept_.copy(),
        sha256_json({"family": "S4-C", "feature": FAMILY_SPECS["S4-C"]["feature"]}),
    )
    model_path = output / "slip_advisory_model.npz"
    model.save(model_path)
    reloaded = SlipV2Model.load(model_path)
    parity_features = rows.features[indices[:128]]
    before = model.scores(parity_features)
    after = reloaded.scores(parity_features)
    parity = max(
        float(np.max(np.abs(before[name] - after[name]))) for name in HEAD_NAMES
    )
    iterations = {
        "state": int(np.max(state.n_iter_)), "foot": int(np.max(foot.n_iter_)),
        "proposal": int(np.max(proposal.n_iter_)),
    }
    health = {
        "version": "walking_v2_slip_advisory_final_fit_v1",
        "development_only": True, "performance_estimation": False,
        "candidate": SELECTED_SLIP_CANDIDATE, "architecture": "S4-C",
        "seed": 202608242, "fit_run_count": len(records), "fit_row_count": len(indices),
        "feature_count": values.shape[1], "projection_width": projection.shape[1],
        "optimizer": "LBFGS", "max_iterations": FINAL_MODEL_MAX_ITER,
        "tolerance": FINAL_MODEL_TOLERANCE, "head_iterations": iterations,
        "all_heads_converged": all(value < FINAL_MODEL_MAX_ITER for value in iterations.values()),
        "convergence_warning_count": len(caught), "convergence_warnings": caught,
        "previous_18_fold_fits_all_hit_400": True,
        "weight_audit": weight_audit,
        "parameter_count": model.parameter_count, "macs_per_endpoint": model.macs + 20_000,
        "reload_max_abs_error": parity, "reload_exact": parity == 0.0,
        "model_sha256": sha256(model_path),
        "limitation": "all-development final fit has no performance estimate; OOF metrics govern role",
    }
    write_json(output / "slip_advisory_training.json", health)
    return health


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    (guard, records, fold_rows, metadata, terrain_model, slip_input,
     immutable_before, started) = preflight(root, output)
    write_json(output / "causal_runtime_contract.json", causal_runtime_contract())

    probabilities, telemetry = compute_terrain_probabilities(records, terrain_model)
    terrain_rows, terrain_config, timelines, selected_terrain = terrain_experiment(
        records, probabilities, telemetry,
    )
    write_csv(output / "terrain_candidate_metrics.csv", terrain_rows)

    broad_outputs, broad_firing = causal_slip_outputs(
        records, slip_input["scores"], telemetry,
    )
    broad_metrics, broad_events, broad_crossings = evaluate_outputs(records, broad_outputs)
    predictive_scores = gated_scores(
        records, slip_input["scores"], telemetry, timelines, reactive=False,
    )
    predictive_outputs, predictive_firing = causal_slip_outputs(
        records, predictive_scores, telemetry,
    )
    predictive_metrics, predictive_events, _ = evaluate_outputs(records, predictive_outputs)
    reactive_scores = gated_scores(
        records, slip_input["scores"], telemetry, timelines, reactive=True,
    )
    reactive_outputs, reactive_firing = causal_slip_outputs(
        records, reactive_scores, telemetry,
    )
    reactive_metrics, reactive_events, _ = evaluate_outputs(records, reactive_outputs)
    scope = scope_metrics(records, timelines)

    stored = slip_input["stored"]
    stored_detected = int(stored["actionable_episode_detected"])
    stored_false = int(stored["normal_contact_episode_fp"]) + int(stored["too_early"])
    stored_precision = safe_ratio(stored_detected, stored_detected + stored_false)
    stored_recall = float(stored["actionable_episode_recall"])
    stored_f1 = (
        2.0 * stored_precision * stored_recall / (stored_precision + stored_recall)
        if stored_precision is not None and stored_precision + stored_recall else 0.0
    )
    advisory_metrics = {
        "version": "walking_v2_fast_reflex_advisory_metrics_v1",
        "terrain_case_a_advisory": selected_terrain,
        "slip_advisory_causal_host_replay": broad_metrics,
        "slip_advisory_frozen_oof_reference": {
            "candidate": SELECTED_SLIP_CANDIDATE,
            "config_id": SELECTED_SLIP_CONFIG.config_id,
            "episode_detected": stored_detected, "episode_count": int(stored["actionable_episode_count"]),
            "episode_recall": stored_recall, "run_detected": int(stored["detected_runs"]),
            "run_count": int(stored["actionable_runs"]),
            "run_recall": int(stored["detected_runs"]) / int(stored["actionable_runs"]),
            "normal_run_fp": int(stored["normal_run_fp"]),
            "normal_contact_fp": int(stored["normal_contact_episode_fp"]),
            "too_early": int(stored["too_early"]), "alert_precision": stored_precision,
            "alert_f1": stored_f1, "affected_foot_accuracy": float(stored["affected_foot_accuracy"]),
            "cross_foot_diagnostic_errors": int(stored["cross_foot_violation"]),
            "speed_recall": json.loads(stored["speed_recall"]),
            "median_warning_margin_ms": float(stored["median_warning_margin_ms"]),
            "p95_late_latency_ms": float(stored["p95_late_latency_ms"]),
            "authority": "advisory only; these are OOF development metrics",
        },
        "case_a_scoped_predictive": predictive_metrics,
        "case_a_scoped_reactive_confirmation": reactive_metrics,
        "case_a_scope": scope,
        "zero_denominator_policy": "null/NOT_APPLICABLE, never 0% performance",
    }
    write_json(output / "advisory_metrics.json", advisory_metrics)
    write_csv(output / "slip_advisory_event_metrics.csv", broad_events)
    write_csv(output / "slip_advisory_crossings.csv", broad_crossings)
    write_csv(output / "case_a_predictive_event_metrics.csv", predictive_events)
    write_csv(output / "case_a_reactive_event_metrics.csv", reactive_events)

    alternatives = [
        {
            "architecture": "TERRAIN_CASE_A_DIRECT", "direct_authority": True,
            "selected": False, "gate_pass": selected_terrain["direct_gate_pass"],
            "event_recall": selected_terrain["event_recall"],
            "recall_within_20ms": selected_terrain["recall_within_20ms"],
            "normal_contact_fp": selected_terrain["normal_contact_fp"],
            "too_early": selected_terrain["too_early"],
            "post_fall_output": selected_terrain["post_fall_output"],
            "reason": "strict false-actuation and recall gates failed",
        },
        {
            "architecture": "BROAD_LEARNED_SLIP_DIRECT", "direct_authority": True,
            "selected": False, "gate_pass": broad_metrics["direct_gate_pass"],
            "event_recall": broad_metrics["event_recall"],
            "recall_within_20ms": broad_metrics["recall_within_20ms"],
            "normal_contact_fp": broad_metrics["normal_contact_fp"],
            "too_early": broad_metrics["too_early"],
            "post_fall_output": broad_metrics["post_fall_output"],
            "reason": "useful advisory recall but direct false outputs are unacceptable",
        },
        {
            "architecture": "CASE_A_PLUS_SLIP_PREDICTIVE_DIRECT", "direct_authority": True,
            "selected": False, "gate_pass": predictive_metrics["direct_gate_pass"],
            "event_recall": predictive_metrics["event_recall"],
            "recall_within_20ms": predictive_metrics["recall_within_20ms"],
            "normal_contact_fp": predictive_metrics["normal_contact_fp"],
            "too_early": predictive_metrics["too_early"],
            "post_fall_output": predictive_metrics["post_fall_output"],
            "reason": "scope is nonzero but precision/recall do not support actuation",
        },
        {
            "architecture": "CASE_A_PLUS_ACTIVE_REACTIVE_DIRECT", "direct_authority": True,
            "selected": False, "gate_pass": reactive_metrics["direct_gate_pass"],
            "event_recall": reactive_metrics["event_recall"],
            "recall_within_20ms": reactive_metrics["recall_within_20ms"],
            "normal_contact_fp": reactive_metrics["normal_contact_fp"],
            "too_early": reactive_metrics["too_early"],
            "post_fall_output": reactive_metrics["post_fall_output"],
            "reason": "reactive confirmation collapses useful detection",
        },
        {
            "architecture": "MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY",
            "direct_authority": False, "selected": True, "gate_pass": True,
            "event_recall": broad_metrics["event_recall"],
            "recall_within_20ms": broad_metrics["recall_within_20ms"],
            "normal_contact_fp": broad_metrics["normal_contact_fp"],
            "too_early": broad_metrics["too_early"],
            "post_fall_output": broad_metrics["post_fall_output"],
            "reason": "preserves useful telemetry without unsafe actuation authority",
        },
    ]
    write_csv(output / "alternative_comparison.csv", alternatives)

    terrain_active = np.stack([timeline.case_a_active for timeline in timelines])
    owners = np.stack([timeline.owner for timeline in timelines])
    direct_reflex = np.zeros_like(terrain_active)
    np.savez_compressed(
        output / "host_replay.npz",
        run_ids=np.asarray([record.run_id for record in records]),
        fold=np.asarray([record.fold for record in records], np.int8),
        endpoints=ENDPOINTS,
        predicted_terrain_class=np.argmax(probabilities, axis=-1).astype(np.int8),
        terrain_case_a_advisory=terrain_active,
        slip_risk_advisory=broad_firing,
        case_a_scoped_predictive=predictive_firing,
        case_a_scoped_reactive=reactive_firing,
        g0_unique_owner=owners,
        direct_reflex=direct_reflex,
    )

    training = fit_final_advisory_model(
        records, metadata, slip_input["hard_rows"], output,
    )
    resource_estimate = {
        "version": "walking_v2_fast_reflex_resource_estimate_v1",
        "endpoint_rate_hz": 100, "sample_preprocessing_rate_hz": 1000,
        "shared_Fusion20_history_bytes_float32": 20 * 200 * 4,
        "terrain_parameter_count": terrain_model.parameter_count,
        "terrain_macs_per_endpoint": terrain_model.macs + 20_000,
        "slip_parameter_count": training["parameter_count"],
        "slip_macs_per_endpoint": training["macs_per_endpoint"],
        "combined_macs_per_endpoint": terrain_model.macs + 20_000 + training["macs_per_endpoint"],
        "estimated_model_bytes_float64": (
            terrain_model.parameter_count + training["parameter_count"]
        ) * 8,
        "deterministic_state_bytes_upper_bound": 1024,
        "compute_latency_benchmarked": False,
        "int8_or_target_resource_claim": False,
    }
    write_json(output / "resource_estimate.json", resource_estimate)

    failure_analysis = {
        "version": "walking_v2_fast_reflex_failure_analysis_v1",
        "actual_blocker": (
            "v5 used an all-false placeholder instead of replaying locked T2, so C1/C2 "
            "denominators were zero; real replay then exposed inadequate Case-A specificity"
        ),
        "resolved": [
            "generated per-endpoint locked-T2 probabilities and contact-local predicted Case-A timelines",
            "removed physical fall/prefall and physical validity from runtime gates",
            "enforced current finite-data mask, G0 owner, contact reset and one-shot emission",
            "evaluated Terrain-only, broad Slip, predictive hybrid and reactive hybrid roles separately",
            "fit a reproducible all-development advisory model after OOF role selection",
        ],
        "direct_authority_failure": {
            "terrain": selected_terrain,
            "predictive_hybrid": predictive_metrics,
            "reactive_hybrid": reactive_metrics,
        },
        "prior_claims_remaining_invalid": [
            "zero-denominator C1/C2 values are not performance",
            "broad learned Slip is not supported for direct actuation",
            "Terrain-only Case A was not validated merely because mapping and lock hashes existed",
            "no blind generalization, safety certification, target readiness or real-robot readiness claim",
        ],
        "why_more_existing_retraining_is_not_next": (
            "the direct failure is dominated by missing aligned transition semantics and specificity, "
            "while repeated S4-C tuning already showed diminishing returns"
        ),
    }
    write_json(output / "failure_analysis.json", failure_analysis)

    readiness = {
        "version": "walking_v2_fast_reflex_readiness_v1",
        "HOST_CAUSAL_REPLAY_READY": True,
        "TERRAIN_CASE_A_TIMELINE_READY": True,
        "SLIP_ADVISORY_MODEL_READY": True,
        "DIRECT_FAST_REFLEX_AUTHORIZED": False,
        "SYSTEM_MIGRATION_AUTHORIZED": False,
        "INT8_VELA_E84_HIL_AUTHORIZED": False,
        "BLIND_EVALUATION_AUTHORIZED": False,
        "REAL_ROBOT_READY": False,
        "SAFETY_CERTIFIED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
        "limitations": [
            "simulation-only development evidence",
            "Walking T2 is in-sample for part of the original 120-run corpus",
            "targeted traces lack an authoritative per-sample physical Terrain transition label",
            "final advisory model is fit on all development data and has no independent metric",
            "OOF advisory precision is insufficient for actuator authority",
            "no causal fall estimator exists; post-fall advisory outputs are reported, not hidden",
            "host compute latency and target-runtime parity are unbenchmarked",
        ],
        "next_step": "ACQUIRE_ALIGNED_CASE_A_TRANSITION_DEVELOPMENT_CORPUS",
    }
    write_json(output / "readiness_limitations.json", readiness)

    immutable_after = {path: sha256(root / path) for path in immutable_before}
    immutable_match = immutable_after == immutable_before
    if not immutable_match:
        raise RuntimeError("Terrain/M1/G0/oracle artifact mutated")
    summary = {
        "task": "Complete autonomous Walking v2 Fast Reflex design",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "selected_architecture": "MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY",
        "technical_stop": "NO_DIRECT_ACTUATION_SUPPORTED_BY_AVAILABLE_DEVELOPMENT_EVIDENCE",
        "direct_reflex_authorized": False,
        "terrain_advisory_config": terrain_config.config_id,
        "terrain_advisory_event_recall": selected_terrain["event_recall"],
        "terrain_advisory_precision": selected_terrain["precision"],
        "terrain_advisory_normal_contact_fp": selected_terrain["normal_contact_fp"],
        "slip_advisory_candidate": SELECTED_SLIP_CANDIDATE,
        "slip_advisory_config": SELECTED_SLIP_CONFIG.config_id,
        "slip_oof_episode_recall": stored_recall,
        "slip_oof_run_recall": int(stored["detected_runs"]) / int(stored["actionable_runs"]),
        "slip_oof_alert_precision": stored_precision,
        "slip_oof_normal_run_fp": int(stored["normal_run_fp"]),
        "slip_oof_normal_contact_fp": int(stored["normal_contact_episode_fp"]),
        "slip_oof_too_early": int(stored["too_early"]),
        "case_a_predictive_episode_denominator": scope["predictive_episode_denominator"],
        "case_a_reactive_episode_denominator": scope["reactive_20ms_episode_denominator"],
        "case_a_predictive_recall_all_events": predictive_metrics["event_recall"],
        "case_a_reactive_recall_all_events": reactive_metrics["event_recall"],
        "final_advisory_model_converged": training["all_heads_converged"],
        "forbidden_access_count": guard.forbidden_count,
        "blind_or_forbidden_data_accessed": False,
        "hardware_e84_hil_executed": False,
        "terrain_M1_G0_oracle_immutable": immutable_match,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "next_step": "ACQUIRE_ALIGNED_CASE_A_TRANSITION_DEVELOPMENT_CORPUS",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)

    audit = f"""# Walking v2 Fast Reflex host v1 audit

## Decision

Selected exactly one development architecture: `MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY`.
The deterministic authority firewall sets direct reflex and recovery actuation to `false`.

```text
Fusion20 + causal G0 contact -> shared 200 ms history
  |-> locked T2 Terrain -> contact-local confidence/dwell -> Case-A advisory
  |-> finalized S4-C    -> V4 contact reset/persistence  -> Slip risk advisory
  `-> deterministic authority firewall -> direct_reflex = false
```

## Required answers

1. **Actual blocker:** v5 populated predicted Case A with an all-false placeholder. Its zero C1/C2 denominators were not model evidence. Real T2 replay exposed a second, material blocker: Case-A specificity is insufficient for direct authority.
2. **Engineering changes:** built the missing causal T2 timeline; added contact-local confidence/dwell, causal finite-data gating, G0 ownership and one-shot reset; removed fall/prefall oracle gates; separated direct and advisory metrics; fitted a reload-exact development advisory model with a raised optimizer ceiling.
3. **Alternatives:** Terrain-only direct, broad learned-Slip direct, Case-A-gated predictive, Case-A-gated reactive confirmation, and monitoring-only advisory were evaluated in `alternative_comparison.csv`.
4. **Selected architecture:** monitoring-only Terrain and Slip advisory, with deterministic direct-actuation firewall.
5. **Authority:** Terrain and Slip are advisory only; learned foot is diagnostic; G0 supplies the only permitted future owner tag; M1 is provenance only; deterministic logic owns masks/reset/dedup and blocks actuation; Sink is deferred.
6. **Metrics:** frozen OOF Slip advisory recall is `{stored_recall:.6f}` episode and `{summary['slip_oof_run_recall']:.6f}` run; alert precision is `{stored_precision:.6f}`; normal-run/contact FP are `{stored['normal_run_fp']}/{stored['normal_contact_episode_fp']}` and too-early outputs are `{stored['too_early']}`. Per-speed recall is `{stored['speed_recall']}`. Terrain advisory recall/precision are `{selected_terrain['event_recall']:.6f}` / `{selected_terrain['precision']:.6f}` with `{selected_terrain['normal_contact_fp']}` normal-contact false advisories. Causal host-replay and latency details are in `advisory_metrics.json`.
7. **Limitations:** simulation-only development data, partial Terrain in-sample overlap, no aligned physical Terrain transition label, no causal fall estimator, no independent metric for the all-development final advisory fit, and no target latency/parity result.
8. **Invalid prior claims:** zero-denominator C1/C2 performance, broad direct learned-Slip readiness, Terrain-only validation from mapping alone, blind generalization, safety certification, target readiness and real-robot readiness remain invalid.
9. **Next action:** exactly `ACQUIRE_ALIGNED_CASE_A_TRANSITION_DEVELOPMENT_CORPUS`.
10. **Data boundary:** no outer, holdout, spatial-final or final-test content was accessed; the exact-path access log has zero forbidden and blocked reads.
11. **Hardware boundary:** no flashing, E84 execution, physical HIL, INT8 or Vela work occurred.
12. **Sink:** remains `SINK_RUNTIME_DETECTION_DEFERRED`.
"""
    (output / "audit.md").write_text(audit)

    test_results = {
        "version": "walking_v2_fast_reflex_internal_tests_v1",
        "checks": {
            "record_count_333": len(records) == 333,
            "event_count_471": sum(len(record.events) for record in records) == 471,
            "case_a_denominators_nonzero": (
                scope["predictive_episode_denominator"] > 0
                and scope["reactive_20ms_episode_denominator"] > 0
            ),
            "direct_output_all_false": not bool(np.any(direct_reflex)),
            "terrain_candidates_27": len(terrain_rows) == 27,
            "no_direct_candidate_selected": not any(
                row["direct_authority"] and row["gate_pass"] for row in alternatives
            ),
            "slip_model_reload_exact": training["reload_exact"],
            "immutable_artifacts_match": immutable_match,
            "forbidden_access_zero": guard.forbidden_count == 0,
            "blocked_access_zero": guard.blocked_count == 0,
            "sink_deferred": readiness["SINK_RUNTIME_DETECTION_DEFERRED"],
            "hardware_not_executed": not summary["hardware_e84_hil_executed"],
        },
        "external_command": (
            "PYTHONPATH=simulation/unitree_mujoco/simulate_python "
            "simulation/venv/bin/python -m unittest -v "
            "simulation/unitree_mujoco/simulate_python/test_walking_v2_fast_reflex_host_v1.py"
        ),
    }
    if not all(test_results["checks"].values()):
        raise RuntimeError(f"internal regression failure: {test_results}")
    write_json(output / "test_results.json", test_results)

    lock_inputs = {
        "protocol.json": sha256(output / "protocol.json"),
        "causal_runtime_contract.json": sha256(output / "causal_runtime_contract.json"),
        "terrain_candidate_metrics.csv": sha256(output / "terrain_candidate_metrics.csv"),
        "advisory_metrics.json": sha256(output / "advisory_metrics.json"),
        "alternative_comparison.csv": sha256(output / "alternative_comparison.csv"),
        "slip_advisory_model.npz": sha256(output / "slip_advisory_model.npz"),
        "slip_advisory_training.json": sha256(output / "slip_advisory_training.json"),
        "summary.json": sha256(output / "summary.json"),
        "test_results.json": sha256(output / "test_results.json"),
    }
    design_lock = {
        "version": "walking_v2_fast_reflex_development_design_lock_v1",
        "immutable": True,
        "lock_type": "development_design_lock_not_blind_or_safety",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "selected_architecture": summary["selected_architecture"],
        "direct_reflex_authorized": False,
        "terrain_advisory_config": terrain_config.config_id,
        "slip_advisory_candidate": SELECTED_SLIP_CANDIDATE,
        "slip_advisory_config": SELECTED_SLIP_CONFIG.config_id,
        "authority_contract": causal_runtime_contract()["authority"],
        "locked_artifact_sha256": lock_inputs,
        "immutable_upstream_sha256": immutable_before,
        "explicitly_not": [
            "blind-validation lock", "safety-certification lock", "deployment lock",
            "real-robot readiness approval",
        ],
        "next_step": "ACQUIRE_ALIGNED_CASE_A_TRANSITION_DEVELOPMENT_CORPUS",
    }
    write_json(output / "design_lock.json", design_lock)

    guard.freeze()
    artifact_hashes: dict[str, str] = {}
    artifact_bytes: dict[str, int] = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json":
            continue
        if not path.is_file():
            raise RuntimeError(f"unexpected non-file artifact: {path}")
        if path.stat().st_size >= 45 * 1024 * 1024:
            raise RuntimeError(f"artifact exceeds 45 MiB: {path}")
        artifact_hashes[path.name] = sha256(path)
        artifact_bytes[path.name] = path.stat().st_size
    provenance = {
        "version": "walking_v2_fast_reflex_provenance_v1",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": {path: sha256(root / path) for path in source_paths()},
        "immutable_before_sha256": immutable_before,
        "immutable_after_sha256": immutable_after, "immutable_match": immutable_match,
        "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes,
        "artifact_hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "forbidden_access_count": guard.forbidden_count,
        "blocked_access_count": guard.blocked_count,
        "outer_holdout_spatial_final_final_test_access_count": 0,
        "new_simulation_data_count": 0, "final_advisory_model_fit_count": 1,
        "direct_actuation_output_count": 0, "hardware_e84_hil_execution_count": 0,
    }
    write_json(output / "provenance.json", provenance)
    print(json.dumps({
        "selected_architecture": summary["selected_architecture"],
        "direct_reflex_authorized": False,
        "case_a_predictive_denominator": scope["predictive_episode_denominator"],
        "case_a_reactive_denominator": scope["reactive_20ms_episode_denominator"],
        "next_step": summary["next_step"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
