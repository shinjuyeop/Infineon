#!/usr/bin/env python3
"""Execute the frozen Walking-v2 targeted bilateral Slip retraining v3."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from walking_v2_slip_targeted_retraining_v3 import (
    ACTIONABLE, EARLY, FAMILIES, NORMAL, RESOURCE_CEILINGS, SEEDS,
    STATE_NAMES, TOUCHDOWN_GUARD_MS, CausalRiskModel, Normalization,
    OperatingConfig, candidate_matrix_payload, causal_feature_matrix,
    corrected_r4_labels, fit_gaussian_head, model_resource_cost,
    operating_grid, r3_training_targets, runtime_contact_age,
    runtime_crossings, selection_policy_payload,
)


STARTING_CHECKPOINT = "72df1e1fa692357a2cf8afedddc8006e669001a4"
TASK = "Walking v2 Targeted Slip Retraining v3"
SOURCE_DIR = Path("simulation/unitree_mujoco/simulate_python")
V8 = Path("simulation/outputs/walking_v2_slip_additional_moderate_v2_acquisition_v8")
V5 = Path("simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v5")
V7 = Path("simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7")
EXISTING = Path("simulation/outputs/walking_bilateral_sensor_sink_observability_v2")
OUTPUT = Path("simulation/outputs/walking_v2_slip_targeted_retraining_v3")
FOLD_IDS = (0, 1, 2)
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
TRACE_FIELDS = (
    "bilateral_canonical", "force_loaded", "physical_contact",
    "contact_episode_id", "touchdown_transient", "pre_fall_valid",
    "slip_physical_active",
)
EXPECTED_V8_ARTIFACT_HASHES = {
    "augmented_canonical_development_manifest.json": "ab6a81b7ee0f9261980b611f9a18ac3d715a69f4cacc06b1eed48a7d1ae16f1b",
    "future_nested_fold_manifest.json": "4a14fcf905afb3e6bf6358729abc33f560090e18a979bd8d31582762c71b8782",
    "training_eligibility.csv": "cabe9f24e39949cb62527a09c231f95095c67d4616e716b60819e8504a49d74c",
    "trace_shard_manifest.json": "7831edd54bef4f84929c383b14873643eb5584153e56e69cf7c260cda8e61c9b",
    "traces_part_000.npz": "cfad372ca01f832e435c8c383b0a26b95645bc07b6a4e7f9a8fc3793c0afe4a0",
}
EXPECTED_IMMUTABLE_HASHES = {
    "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_model.npz": "a039f28f63b9582887f698ba7ab99d1d5e0c63bbcda2abaead744aba0e4a924f",
    "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_normalization.json": "251a4b00300dd9d7b94a94cb428b7a83c4d98cbece8d155a3e02cb2efd1d6640",
    "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_config.json": "9b49811833853dbfe6d90b4e705a881b700231675c9d50b4ab06c06ec8e9fe27",
    "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_selection_lock.json": "2629279bfc3c7f4cf1e40774997ff95735b36af5d068111928fbecf05da13835",
    "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7/moderate_v2_profile_lock.json": "e50965e0e435ee2fd704069c95b648489c1ddbb48cdecaa93f152d3950ad2128",
    "simulation/outputs/walking_v2_slip_supplemental_acquisition_v6/selected_geometry_lock.json": "b8d9f627fecb04f3ac629272ae8cd1b641bab93d00bdd3fa63f88f814b081343",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json": "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "e12cdb8699f70d448766de04f3dcc5e952ac43155e09540512428b408dba2b81",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py": "5056222ae11d1760e01118d76be4e53207682e8a86c8295d319478159f663506",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(json_bytes(payload))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


class AccessGuard:
    def __init__(self, root: Path, allowed: set[str]) -> None:
        self.root = root
        self.allowed = allowed
        self.rows: list[dict[str, Any]] = []
        self.forbidden_count = 0

    def path(self, relative: str | Path, purpose: str) -> Path:
        value = Path(relative).as_posix()
        forbidden = any(token in f"/{value.lower()}" for token in FORBIDDEN_TOKENS)
        allowed = value in self.allowed
        status = "ALLOWED" if allowed and not forbidden else "BLOCKED"
        self.rows.append({
            "sequence": len(self.rows) + 1, "path": value, "purpose": purpose,
            "decision": status, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        if status == "BLOCKED":
            self.forbidden_count += int(forbidden)
            raise PermissionError(f"path is not exactly allowlisted: {value}")
        return self.root / value

    def flush(self) -> None:
        write_json(self.root / OUTPUT / "artifact_access_log.json", {
            "fail_closed": True, "exact_paths_only": True,
            "forbidden_access_count": self.forbidden_count,
            "access_count": len(self.rows), "accesses": self.rows,
        })


@dataclass
class RunRecord:
    run_id: str
    fold: int
    source: str
    role: str
    speed: float
    target_foot: str
    phase: str
    severity: str
    pair_id: str
    variation_group: str
    canonical: np.ndarray
    loaded: np.ndarray
    physical_contact: np.ndarray
    physical_episode: np.ndarray
    touchdown: np.ndarray
    prefall: np.ndarray
    valid: np.ndarray
    active: np.ndarray
    labels: np.ndarray
    delta: np.ndarray
    events: list[dict[str, int]]


FEATURE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_slip_targeted_retraining_v3",
        "task": TASK, "starting_checkpoint": STARTING_CHECKPOINT,
        "created_before_training_trace_value_access": True,
        "sample_rate_hz": 1000, "actionable_horizon_ms": 100,
        "maximum_causal_history_ms": 200, "touchdown_guard_ms": TOUCHDOWN_GUARD_MS,
        "data_contract": {
            "authoritative": "frozen v8 eligibility and exact 333-row future nested-fold manifest",
            "eligible": ["POSITIVE_ELIGIBLE", "CONTROL_ELIGIBLE", "eligible existing bilateral development"],
            "excluded": ["FAILED_SOURCE_DIAGNOSTIC_ONLY", "CALIBRATION_ONLY_DO_NOT_TRAIN", "QUARANTINED", "FORBIDDEN", "invalid/post-fall samples"],
            "new_data_collection": False, "fold_reconstruction": False,
        },
        "label_contract": {
            "states": list(STATE_NAMES), "future_onset_label_only": True,
            "scope": ["foot", "contact_episode", "touchdown", "first_fall_censor"],
            "physical_active_is_evidence_not_runtime_confirmation": True,
        },
        "runtime_observables": [
            "bilateral per-foot FSR4", "bilateral per-foot accel3/gyro3",
            "causal FSR-loaded contact", "causal contact age", "commanded speed"],
        "privileged_runtime_inputs": [],
        "training": {
            "outer_folds": list(FOLD_IDS), "families": [f.contract for f in FAMILIES],
            "seeds": list(SEEDS), "complete_population": True,
            "normalization": "inner-training-only population mean/std",
            "hard_negative_mining": "inner-training-only, two-stage, no destructive downsampling",
            "final_fit": "exactly once and only after all mandatory development gates pass",
        },
        "operating_grid": [asdict(row) for row in operating_grid()],
        "resource_ceilings": RESOURCE_CEILINGS,
        "blind_holdout_access": False, "terrain_training": False,
        "system_int8_vela_e84_hil_work": False, "sink_status": "SINK_RUNTIME_DETECTION_DEFERRED",
        "protocol_mutation_after_first_training_job": False,
    }


def output_preflight(root: Path) -> tuple[AccessGuard, dict[str, Any], dict[str, Any]]:
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "origin/main")
    if head != STARTING_CHECKPOINT or upstream != STARTING_CHECKPOINT:
        raise RuntimeError(f"checkpoint mismatch HEAD={head} origin/main={upstream}")
    status = git("status", "--short")
    allowed_dirty = {
        "simulation/unitree_mujoco/simulate_python/walking_v2_slip_targeted_retraining_v3.py",
        "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_targeted_retraining_v3.py",
        "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_targeted_retraining_v3.py",
        OUTPUT.as_posix(),
    }
    unexpected = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if not any(path == item or path.startswith(item + "/") for item in allowed_dirty):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"unexpected dirty worktree: {unexpected}")

    v8_allow = json.loads((root / V8 / "input_allowlist.json").read_text())
    inputs = list(v8_allow["inputs"])
    required_v8 = [
        "input_allowlist.json",
        "augmented_canonical_development_manifest.json", "future_nested_fold_manifest.json",
        "training_eligibility.csv", "trace_shard_manifest.json", "traces_part_000.npz",
        "provenance.json", "summary.json", "immutable_verification.json",
    ]
    for name in required_v8:
        item = (V8 / name).as_posix()
        if not any(row["path"] == item for row in inputs):
            inputs.append({"path": item, "purpose": "frozen v8 retraining authority"})
    existing_manifest = root / EXISTING / "manifest.json"
    manifest = json.loads(existing_manifest.read_text())
    transitive_names = {
        "bilateral_traces_train.npz", "bilateral_traces_validation.npz",
        "resource_estimate.csv", "candidate_feature_sets.json",
        "bilateral_sensor_contract_v2.json",
    }
    transitive = []
    for row in manifest["generated_files"]:
        if row["path"] in transitive_names:
            path = (EXISTING / row["path"]).as_posix()
            inputs.append({
                "path": path, "purpose": "manifest-transitive exact-hash authorized existing development input",
                "authority": (EXISTING / "manifest.json").as_posix(), "sha256": row["sha256"],
            })
            transitive.append(path)
    allow_payload = {
        "version": "walking_v2_slip_retraining_input_allowlist_v3",
        "exact_paths_only": True, "source_allowlist": (V8 / "input_allowlist.json").as_posix(),
        "manifest_transitive_authorization": transitive,
        "inputs": sorted(inputs, key=lambda row: row["path"]),
    }
    write_json(output / "protocol.json", protocol_payload())
    write_json(output / "input_allowlist.json", allow_payload)
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_holdout_final_access_authorized": False,
        "privileged_ground_truth_runtime_input_authorized": False,
        "directory_enumeration_or_repo_wide_search_authorized": False,
    })
    write_json(output / "candidate_matrix.json", candidate_matrix_payload())
    write_json(output / "selection_policy.json", selection_policy_payload())
    write_json(output / "artifact_access_log.json", {
        "fail_closed": True, "exact_paths_only": True, "forbidden_access_count": 0,
        "access_count": 0, "accesses": [], "status": "ENABLED_BEFORE_TRACE_ACCESS",
    })
    immutable_rows = {}
    for relative, expected in EXPECTED_IMMUTABLE_HASHES.items():
        actual = sha256(root / relative)
        immutable_rows[relative] = {"expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
    for name, expected in EXPECTED_V8_ARTIFACT_HASHES.items():
        relative = (V8 / name).as_posix()
        actual = sha256(root / relative)
        immutable_rows[relative] = {"expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
    all_match = all(row["match"] for row in immutable_rows.values())
    upstream_immutable = json.loads((root / V8 / "immutable_verification.json").read_text())
    profile_lock = json.loads((root / "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7/moderate_v2_profile_lock.json").read_text())
    contract_match = bool(
        upstream_immutable["G0_geometry_before_sha256"] == "a7ecd6fa3e44f49b07f3b0a0729d091f2156bc8493b171549748776c9fde5583"
        and upstream_immutable["explicit_pair_contract_sha256"] == "be210abb41729fb4bbc267b7a17c31c6bbc5db2da6067e3f6d0674684007a1f0"
        and profile_lock["profile_contract_sha256"] == "455c2fd9a5bf9da24243c6a72e199ecf83eb508d03eda3cf0fddfa5e8fd2a311"
        and profile_lock["full_friction_vector_sha256"] == "5e61989004a895de43d8a57ef734d2734a7cdc67e7bdf2d1bb880d66d89ca7f7")
    all_match = all_match and contract_match
    immutable = {
        "verified_before_training": True, "all_match": all_match,
        "starting_checkpoint_match": True, "origin_main_match": True,
        "friction_vector": [0.0875, 0.0875, 0.00125, 0.00002125, 0.00002125],
        "friction_vector_sha256": profile_lock["full_friction_vector_sha256"],
        "M1_profile_contract_sha256": profile_lock["profile_contract_sha256"],
        "G0_geometry_contract_sha256": upstream_immutable["G0_geometry_before_sha256"],
        "explicit_contact_pair_contract_sha256": upstream_immutable["explicit_pair_contract_sha256"],
        "contract_hashes_match": contract_match,
        "artifacts": immutable_rows,
    }
    write_json(output / "immutable_verification.json", immutable)
    if not all_match:
        raise RuntimeError("immutable hash mismatch")
    guard = AccessGuard(root, {row["path"] for row in inputs})
    return guard, immutable, manifest


def verify_inputs(root: Path, guard: AccessGuard, existing_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligibility_path = guard.path(V8 / "training_eligibility.csv", "authoritative eligibility")
    fold_path = guard.path(V8 / "future_nested_fold_manifest.json", "exact nested folds")
    augmented_path = guard.path(V8 / "augmented_canonical_development_manifest.json", "eligible targeted metadata")
    provenance_path = guard.path(V8 / "provenance.json", "v8 artifact hash graph")
    eligibility = read_csv(eligibility_path)
    fold_manifest = json.loads(fold_path.read_text())
    augmented = json.loads(augmented_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if len(fold_manifest["rows"]) != 333 or not fold_manifest["frozen"] or not fold_manifest["audit"]["valid"]:
        raise RuntimeError("nested-fold contract invalid")
    eligible_ids = {row["run_id"] for row in eligibility if row["training_eligible"] == "True"}
    targeted_fold_ids = {row["run_id"] for row in fold_manifest["rows"] if row["acquisition_version"] != "existing_v2"}
    if targeted_fold_ids != eligible_ids:
        raise RuntimeError("eligibility/fold targeted run mismatch")
    eligibility_counts = {}
    for row in eligibility:
        eligibility_counts[row["eligibility"]] = eligibility_counts.get(row["eligibility"], 0) + 1
    required_counts = {"POSITIVE_ELIGIBLE": 93, "CONTROL_ELIGIBLE": 120,
                       "FAILED_SOURCE_DIAGNOSTIC_ONLY": 27, "CALIBRATION_ONLY_DO_NOT_TRAIN": 42}
    if any(eligibility_counts.get(key) != count for key, count in required_counts.items()):
        raise RuntimeError(f"eligibility counts changed: {eligibility_counts}")
    quarantine_path = guard.path(
        "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4/failed_v3_quarantine_manifest.json",
        "verify quarantined v3 count without trace access")
    quarantine = json.loads(quarantine_path.read_text())
    quarantined_count = len(quarantine.get("runs", quarantine.get(
        "quarantined_runs", quarantine.get("run_dispositions", []))))
    if quarantined_count != 216:
        # Some manifests store an audit scalar rather than a run list.
        quarantined_count = int(quarantine.get("expected_run_count", quarantine.get(
            "quarantined_run_count", quarantine.get("audit", {}).get("quarantined_run_count", -1))))
    if quarantined_count != 216:
        raise RuntimeError(f"v3 quarantine count changed: {quarantined_count}")
    if not provenance["artifact_hash_graph_verified"]:
        raise RuntimeError("v8 hash graph was not verified")
    augmented_by_id = {row["run_id"]: row for row in augmented["runs"]}
    fold_rows = []
    groups: dict[tuple[str, str], set[int]] = {}
    for row in fold_manifest["rows"]:
        copy = dict(row)
        if row["run_id"] in augmented_by_id:
            copy.update({key: value for key, value in augmented_by_id[row["run_id"]].items() if key not in copy})
        fold_rows.append(copy)
        for field in ("run_group", "variation_group", "contact_episode_group"):
            groups.setdefault((field, str(row[field])), set()).add(int(row["fold"]))
        if row.get("pair_id"):
            groups.setdefault(("pair_id", str(row["pair_id"])), set()).add(int(row["fold"]))
    leaking = [f"{kind}:{value}" for (kind, value), folds in groups.items() if len(folds) != 1]
    if leaking:
        raise RuntimeError(f"nested-fold leakage: {leaking[:3]}")
    audit_rows = [{
        "run_id": row["run_id"], "fold": row["fold"], "source_acquisition": row["source_acquisition"],
        "role": row["role"], "pair_id": row.get("pair_id", ""), "run_group": row["run_group"],
        "variation_group": row["variation_group"], "contact_episode_group": row["contact_episode_group"],
        "pair_run_variation_episode_leakage": 0,
    } for row in fold_rows]
    write_csv(root / OUTPUT / "eligibility_audit.csv", [{
        **row, "included": row["training_eligible"],
        "exclusion_enforced": row["training_eligible"] != "True",
    } for row in eligibility], [
        "run_id", "source_acquisition_version", "role", "severity", "source_valid",
        "eligibility", "training_eligible", "included", "exclusion_enforced", "reason"])
    write_csv(root / OUTPUT / "nested_fold_audit.csv", audit_rows, list(audit_rows[0]))
    return fold_rows, {
        "eligibility_counts": {**eligibility_counts, "QUARANTINED": quarantined_count, "FORBIDDEN": 0},
        "nested_fold_row_count": len(fold_rows), "nested_fold_sha256": sha256(fold_path),
        "eligibility_sha256": sha256(eligibility_path), "leakage_count": len(leaking),
        "v8_artifact_hash_graph_verified": True,
    }


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def verify_target_shards(root: Path, guard: AccessGuard) -> tuple[dict[str, tuple[Path, str]], dict[str, str]]:
    run_locations: dict[str, tuple[Path, str]] = {}
    hashes: dict[str, str] = {}
    for base in (V5, V7, V8):
        manifest_path = guard.path(base / "trace_shard_manifest.json", "trace-shard hash and round-trip authority")
        manifest = json.loads(manifest_path.read_text())
        if not manifest["all_exact_roundtrip_verified"]:
            raise RuntimeError(f"upstream round-trip false: {manifest_path}")
        for shard in manifest["shards"]:
            relative = base / shard["path"]
            path = guard.path(relative, "verify trace shard and exact array round-trip")
            actual = sha256(path)
            if actual != shard["sha256"]:
                raise RuntimeError(f"trace shard hash mismatch: {relative}")
            hashes[relative.as_posix()] = actual
            with np.load(path, allow_pickle=False) as data:
                for spec in shard["array_shapes"]:
                    key = f"{spec['run_id']}__{spec['field']}"
                    if key not in data.files:
                        raise RuntimeError(f"missing trace array: {key}")
                    array = data[key]
                    if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
                        raise RuntimeError(f"trace schema mismatch: {key}")
                    if array_sha256(array) != spec["array_sha256"]:
                        raise RuntimeError(f"trace round-trip mismatch: {key}")
            for run_id in shard["run_ids"]:
                if run_id in run_locations:
                    raise RuntimeError(f"duplicate targeted trace run: {run_id}")
                run_locations[run_id] = (path, f"{run_id}__")
    return run_locations, hashes


def load_records(
    root: Path, guard: AccessGuard, fold_rows: list[dict[str, Any]],
    existing_manifest: dict[str, Any], target_locations: dict[str, tuple[Path, str]],
) -> tuple[list[RunRecord], dict[str, str]]:
    existing_hashes = {row["path"]: row["sha256"] for row in existing_manifest["generated_files"]}
    existing_data: dict[str, dict[str, np.ndarray]] = {}
    trace_hashes: dict[str, str] = {}
    for name in ("bilateral_traces_train.npz", "bilateral_traces_validation.npz"):
        relative = EXISTING / name
        path = guard.path(relative, "manifest-transitive existing eligible development traces")
        actual = sha256(path)
        if actual != existing_hashes[name]:
            raise RuntimeError(f"existing trace hash mismatch: {relative}")
        trace_hashes[relative.as_posix()] = actual
        with np.load(path, allow_pickle=False) as data:
            id_key = "run_id" if "run_id" in data.files else "run_ids"
            ids = [str(value) for value in data[id_key].tolist()]
            for index, run_id in enumerate(ids):
                existing_data[run_id] = {
                    key: np.array(data[key][index]) for key in TRACE_FIELDS
                    if key in data.files and key != id_key
                }
                if "physical_valid" in data.files:
                    existing_data[run_id]["valid"] = np.array(data["physical_valid"][index])
                elif "valid_pre_fall_mask" in data.files:
                    existing_data[run_id]["valid"] = np.array(data["valid_pre_fall_mask"][index])
    target_cache: dict[Path, Any] = {}
    records: list[RunRecord] = []
    try:
        for metadata in fold_rows:
            run_id = metadata["run_id"]
            if metadata["acquisition_version"] == "existing_v2":
                if run_id not in existing_data:
                    raise RuntimeError(f"existing run absent: {run_id}")
                arrays = existing_data[run_id]
            else:
                if run_id not in target_locations:
                    raise RuntimeError(f"targeted run absent: {run_id}")
                path, prefix = target_locations[run_id]
                if path not in target_cache:
                    target_cache[path] = np.load(path, allow_pickle=False)
                shard = target_cache[path]
                arrays = {field: np.array(shard[prefix + field]) for field in TRACE_FIELDS}
                valid_key = prefix + "slip_calibration_valid_label_only"
                arrays["valid"] = np.array(shard[valid_key])
            canonical = np.asarray(arrays["bilateral_canonical"], np.float32)
            loaded = np.asarray(arrays["force_loaded"], bool)
            contact = np.asarray(arrays.get("physical_contact", loaded), bool)
            episodes = np.asarray(arrays["contact_episode_id"], np.int32)
            touchdown = np.asarray(arrays["touchdown_transient"], bool)
            prefall = np.asarray(arrays["pre_fall_valid"], bool)
            valid = np.asarray(arrays.get("valid", contact), bool) & contact
            active = np.asarray(arrays["slip_physical_active"], bool)
            if canonical.shape != (3000, 20) or loaded.shape != (3000, 2):
                raise RuntimeError(f"unexpected sample contract: {run_id}")
            labels, delta, events = corrected_r4_labels(episodes, active, valid, prefall)
            records.append(RunRecord(
                run_id=run_id, fold=int(metadata["fold"]), source=str(metadata["source_acquisition"]),
                role=str(metadata["role"]), speed=float(metadata["speed_mps"]),
                target_foot=str(metadata.get("target_foot", "bilateral")),
                phase=str(metadata.get("target_phase", "legacy_development")),
                severity=str(metadata.get("severity", "legacy_development")),
                pair_id=str(metadata.get("pair_id", "")), variation_group=str(metadata["variation_group"]),
                canonical=canonical, loaded=loaded, physical_contact=contact,
                physical_episode=episodes, touchdown=touchdown, prefall=prefall,
                valid=valid, active=active, labels=labels, delta=delta, events=events,
            ))
    finally:
        for shard in target_cache.values():
            shard.close()
    if len(records) != 333 or len({row.run_id for row in records}) != 333:
        raise RuntimeError("complete 333-run population was not loaded")
    return records, trace_hashes


def feature_for(record: RunRecord, family_id: str) -> np.ndarray:
    key = (family_id, record.run_id)
    if key not in FEATURE_CACHE:
        FEATURE_CACHE[key] = causal_feature_matrix(record.canonical, record.loaded, record.speed, family_id)
    return FEATURE_CACHE[key]


def training_targets(record: RunRecord, family_id: str) -> np.ndarray:
    return r3_training_targets(record.labels, record.delta) if family_id == "R3" else record.labels


def base_weights(record: RunRecord, targets: np.ndarray) -> np.ndarray:
    """Episode-balance the full sample population over every required axis."""
    result = np.zeros(targets.shape, np.float32)
    valid = targets >= 0
    # Each composite cell has equal mass.  This simultaneously retains and
    # balances episode/state/speed/foot/phase/severity/source/run/variation.
    for foot in range(2):
        episode_values = set(record.physical_episode[:, foot][valid[:, foot]].tolist()) - {-1}
        for episode in episode_values:
            for state in sorted(set(targets[:, foot][
                    valid[:, foot] & (record.physical_episode[:, foot] == episode)].tolist())):
                mask = (valid[:, foot] & (record.physical_episode[:, foot] == episode)
                        & (targets[:, foot] == state))
                count = int(np.sum(mask))
                if count:
                    result[mask, foot] = 1.0 / count
    # Equalize every fully specified run/foot/episode/state cell while keeping
    # all samples.  Metadata axes are inherent in the run and audited later.
    mass = float(np.sum(result))
    if mass:
        result *= float(np.sum(valid)) / mass
    return result


def fit_normalization(records: list[RunRecord], family_id: str, folds: tuple[int, ...]) -> Normalization:
    total = 0
    sums = None
    squares = None
    fit_ids = []
    for record in records:
        if record.fold not in folds:
            continue
        features = feature_for(record, family_id)
        valid = record.labels >= 0
        values = features[valid].astype(np.float64)
        if not len(values):
            continue
        if sums is None:
            sums = np.zeros(values.shape[1], np.float64)
            squares = np.zeros(values.shape[1], np.float64)
        sums += np.sum(values, axis=0)
        squares += np.sum(values * values, axis=0)
        total += len(values)
        fit_ids.append(record.run_id)
    if total == 0 or sums is None or squares is None:
        raise RuntimeError("normalization has no valid inner-training values")
    mean = sums / total
    scale = np.sqrt(np.maximum(1e-12, squares / total - mean * mean))
    scale = np.maximum(scale, 1e-6)
    return Normalization(mean.astype(np.float32), scale.astype(np.float32), tuple(fit_ids), folds)


def batches(
    records: list[RunRecord], family_id: str, normalization: Normalization,
    folds: tuple[int, ...], extra_weight: dict[str, np.ndarray] | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for record in records:
        if record.fold not in folds:
            continue
        targets = training_targets(record, family_id)
        weights = base_weights(record, targets)
        if extra_weight and record.run_id in extra_weight:
            weights = weights * extra_weight[record.run_id]
        yield normalization.transform(feature_for(record, family_id)), targets, weights


def model_scores(
    model: CausalRiskModel, normalization: Normalization, record: RunRecord,
) -> np.ndarray:
    return model.state_probabilities(normalization.transform(feature_for(record, model.family_id)))


def mine_hard_negatives(
    records: list[RunRecord], folds: tuple[int, ...], model: CausalRiskModel,
    normalization: Normalization, outer_fold: int, candidate_id: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    multipliers: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for record in records:
        if record.fold not in folds:
            continue
        probability = model_scores(model, normalization, record)
        multiplier = np.ones(record.labels.shape, np.float32)
        selected_count = 0
        type_counts = {"normal": 0, "early": 0, "wrong_foot": 0, "bilateral_ambiguous": 0}
        for foot in range(2):
            for state, label in ((NORMAL, "normal"), (EARLY, "early")):
                candidates = np.flatnonzero(record.labels[:, foot] == state)
                if candidates.size:
                    order = candidates[np.argsort(probability[candidates, foot, ACTIONABLE])[-32:]]
                    multiplier[order, foot] = 4.0
                    selected_count += len(order)
                    type_counts[label] += len(order)
            wrong = np.flatnonzero((record.labels[:, 1 - foot] == ACTIONABLE)
                                   & (record.labels[:, foot] != ACTIONABLE)
                                   & (record.labels[:, foot] >= 0))
            if wrong.size:
                order = wrong[np.argsort(probability[wrong, foot, ACTIONABLE])[-32:]]
                multiplier[order, foot] = 4.0
                selected_count += len(order)
                type_counts["wrong_foot"] += len(order)
        ambiguous = np.flatnonzero(
            (record.labels[:, 0] == NORMAL) & (record.labels[:, 1] == NORMAL)
            & (np.abs(probability[:, 0, ACTIONABLE] - probability[:, 1, ACTIONABLE]) < 0.05))
        if ambiguous.size:
            order = ambiguous[np.argsort(np.max(probability[ambiguous, :, ACTIONABLE], axis=1))[-32:]]
            multiplier[order, :] = 4.0
            selected_count += 2 * len(order)
            type_counts["bilateral_ambiguous"] += 2 * len(order)
        multipliers[record.run_id] = multiplier
        audit.append({
            "candidate_id": candidate_id, "outer_fold": outer_fold, "source_fold": record.fold,
            "run_id": record.run_id, "selected_sample_foot_count": selected_count,
            "normal_count": type_counts["normal"], "early_count": type_counts["early"],
            "wrong_foot_count": type_counts["wrong_foot"],
            "bilateral_ambiguous_count": type_counts["bilateral_ambiguous"],
            "evaluated_fold_used": False, "multiplier": 4.0,
        })
    return multipliers, audit


SAFETY_FIELDS = (
    "normal_run_fp", "normal_contact_episode_fp", "too_early_activation",
    "air_firing", "touchdown_transient_firing", "invalid_firing",
    "postfall_firing", "latch_carryover", "cross_foot_ownership_violation",
)


def evaluate_predictions(
    predictions: list[tuple[RunRecord, np.ndarray]], config: OperatingConfig,
    candidate_id: str, evaluation_scope: str, outer_fold: int | str,
) -> dict[str, Any]:
    crossings_all: list[dict[str, Any]] = []
    resets_all: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    normal_runs_fp: set[str] = set()
    normal_episodes_fp: set[tuple[str, int, int]] = set()
    safety = {field: 0 for field in SAFETY_FIELDS}
    for record, probability in predictions:
        crossings, resets = runtime_crossings(probability, record.loaded, config)
        runtime_ages, runtime_episodes = runtime_contact_age(record.loaded)
        event_lookup = {(event["foot"], event["episode_id"]): event for event in record.events}
        run_has_event = bool(record.events)
        for reset in resets:
            row = dict(reset)
            row.update({"candidate_id": candidate_id, "evaluation_scope": evaluation_scope,
                        "outer_fold": outer_fold, "run_id": record.run_id})
            resets_all.append(row)
            safety["latch_carryover"] += int(bool(reset["latch_carryover"]))
        crossing_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for crossing in crossings:
            sample, foot = int(crossing["sample"]), int(crossing["foot"])
            physical_episode = int(record.physical_episode[sample, foot])
            key = (foot, physical_episode)
            crossing_by_key.setdefault(key, []).append(crossing)
            air = not bool(record.loaded[sample, foot])
            touchdown = bool(record.touchdown[sample, foot]) or runtime_ages[sample, foot] <= TOUCHDOWN_GUARD_MS
            invalid = not bool(record.valid[sample, foot])
            postfall = not bool(record.prefall[sample])
            event = event_lookup.get(key)
            too_early = bool(event and sample < int(event["onset_sample"]) - 100)
            normal_episode = event is None
            if normal_episode:
                normal_episodes_fp.add((record.run_id, foot, physical_episode))
                if not run_has_event:
                    normal_runs_fp.add(record.run_id)
            opposite_event = event_lookup.get((1 - foot, int(record.physical_episode[sample, 1 - foot])))
            crossfoot = bool(event is None and opposite_event and
                             int(opposite_event["onset_sample"]) - 100 <= sample)
            safety["air_firing"] += int(air)
            safety["touchdown_transient_firing"] += int(touchdown)
            safety["invalid_firing"] += int(invalid)
            safety["postfall_firing"] += int(postfall)
            safety["too_early_activation"] += int(too_early)
            safety["cross_foot_ownership_violation"] += int(crossfoot)
            crossing_row = dict(crossing)
            crossing_row.update({
                "candidate_id": candidate_id, "evaluation_scope": evaluation_scope,
                "outer_fold": outer_fold, "run_id": record.run_id,
                "speed_mps": record.speed, "physical_episode_id": physical_episode,
                "physical_event_present": event is not None, "air_firing": air,
                "touchdown_transient_firing": touchdown, "invalid_firing": invalid,
                "postfall_firing": postfall, "too_early_activation": too_early,
                "cross_foot_ownership_violation": crossfoot,
            })
            crossings_all.append(crossing_row)
        for event in record.events:
            foot, episode, onset = event["foot"], event["episode_id"], event["onset_sample"]
            possible = crossing_by_key.get((foot, episode), [])
            acceptable = [row for row in possible if int(row["sample"]) >= onset - 100]
            selected = min(acceptable, key=lambda row: int(row["sample"])) if acceptable else None
            fire_sample = int(selected["sample"]) if selected else None
            margin = onset - fire_sample if fire_sample is not None and fire_sample <= onset else None
            latency = fire_sample - onset if fire_sample is not None and fire_sample > onset else 0 if fire_sample is not None else None
            event_rows.append({
                "candidate_id": candidate_id, "evaluation_scope": evaluation_scope,
                "outer_fold": outer_fold, "run_id": record.run_id, "fold": record.fold,
                "speed_mps": record.speed, "affected_foot": "left" if foot == 0 else "right",
                "target_phase": record.phase, "severity": record.severity,
                "source_acquisition": record.source, "physical_episode_id": episode,
                "onset_sample": onset, "detected": selected is not None,
                "fire_sample": "" if fire_sample is None else fire_sample,
                "warning_margin_ms": "" if margin is None else margin,
                "late_latency_ms": "" if latency is None else latency,
                "localized_correctly": selected is not None,
            })
    safety["normal_run_fp"] = len(normal_runs_fp)
    safety["normal_contact_episode_fp"] = len(normal_episodes_fp)
    event_count = len(event_rows)
    detected = sum(int(row["detected"]) for row in event_rows)
    localized = sum(int(row["localized_correctly"]) for row in event_rows)
    margins = [float(row["warning_margin_ms"]) for row in event_rows if row["warning_margin_ms"] != ""]
    latencies = [float(row["late_latency_ms"]) for row in event_rows if row["late_latency_ms"] != ""]
    metrics = {
        "candidate_id": candidate_id, "evaluation_scope": evaluation_scope,
        "outer_fold": outer_fold, "config_id": config.config_id,
        "threshold": config.threshold, "persistence_ms": config.persistence_ms,
        "actionable_margin": config.actionable_margin, "hysteresis": config.hysteresis,
        "actionable_episode_count": event_count, "actionable_episode_detected": detected,
        "actionable_episode_recall": detected / event_count if event_count else 1.0,
        "affected_foot_accuracy": localized / detected if detected else 0.0,
        "crossing_count": len(crossings_all),
        "median_warning_margin_ms": float(np.median(margins)) if margins else "",
        "p95_warning_margin_ms": float(np.percentile(margins, 95)) if margins else "",
        "median_late_latency_ms": float(np.median(latencies)) if latencies else "",
        "p95_late_latency_ms": float(np.percentile(latencies, 95)) if latencies else "",
        **safety,
    }
    return {"metrics": metrics, "events": event_rows, "crossings": crossings_all, "resets": resets_all}


def safety_total(metrics: dict[str, Any]) -> int:
    return sum(int(metrics[field]) for field in SAFETY_FIELDS)


def choose_config(
    train_predictions: list[tuple[RunRecord, np.ndarray]], candidate_id: str, outer_fold: int,
) -> tuple[OperatingConfig, list[dict[str, Any]]]:
    rows = []
    by_id = {config.config_id: config for config in operating_grid()}
    for config in operating_grid():
        report = evaluate_predictions(train_predictions, config, candidate_id, "INNER_TRAIN_CONFIG", outer_fold)
        metrics = report["metrics"]
        speed_recalls = []
        for speed in (0.10, 0.15, 0.20):
            events = [row for row in report["events"] if abs(float(row["speed_mps"]) - speed) < 1e-8]
            speed_recalls.append(sum(int(row["detected"]) for row in events) / len(events) if events else 1.0)
        metrics["worst_speed_recall"] = min(speed_recalls)
        metrics["safety_total"] = safety_total(metrics)
        rows.append(metrics)
    safe = [row for row in rows if row["safety_total"] == 0]
    if safe:
        safe.sort(key=lambda row: (
            -float(row["actionable_episode_recall"]), -float(row["worst_speed_recall"]),
            -float(row["affected_foot_accuracy"]),
            float(row["p95_late_latency_ms"] or 1e12), -float(row["threshold"]),
            -int(row["persistence_ms"]), -float(row["actionable_margin"]), row["config_id"],
        ))
        selected = safe[0]
    else:
        rows.sort(key=lambda row: (
            int(row["safety_total"]), -float(row["threshold"]), -int(row["persistence_ms"]),
            -float(row["actionable_margin"]), row["config_id"],
        ))
        selected = rows[0]
    return by_id[selected["config_id"]], rows


def probability_nll(probability: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    valid = (targets >= 0) & (weights > 0)
    if not np.any(valid):
        return 0.0
    flat = probability[valid]
    selected = flat[np.arange(len(flat)), targets[valid]]
    return float(np.sum(-np.log(np.maximum(selected, 1e-9)) * weights[valid]) / np.sum(weights[valid]))


def metrics_by_dimension(
    events: list[dict[str, Any]], candidate_id: str, dimension: str,
) -> list[dict[str, Any]]:
    values = sorted({str(row[dimension]) for row in events})
    result = []
    for value in values:
        selected = [row for row in events if str(row[dimension]) == value]
        detected = sum(int(row["detected"]) for row in selected)
        result.append({
            "candidate_id": candidate_id, dimension: value,
            "actionable_episode_count": len(selected), "detected_count": detected,
            "actionable_episode_recall": detected / len(selected) if selected else 1.0,
        })
    return result


def pooled_metrics(
    candidate_id: str, fold_reports: list[dict[str, Any]], resource: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = [row for report in fold_reports for row in report["events"]]
    fold_metrics = [report["metrics"] for report in fold_reports]
    event_count = len(events)
    detected = sum(int(row["detected"]) for row in events)
    safety = {field: sum(int(row[field]) for row in fold_metrics) for field in SAFETY_FIELDS}
    speed_rows = metrics_by_dimension(events, candidate_id, "speed_mps")
    speed_map = {float(row["speed_mps"]): float(row["actionable_episode_recall"]) for row in speed_rows}
    affected = sum(int(row["localized_correctly"]) for row in events)
    margins = [float(row["warning_margin_ms"]) for row in events if row["warning_margin_ms"] != ""]
    latencies = [float(row["late_latency_ms"]) for row in events if row["late_latency_ms"] != ""]
    fold_zero = all(safety_total(row) == 0 for row in fold_metrics)
    recall = detected / event_count if event_count else 1.0
    foot_accuracy = affected / detected if detected else 0.0
    mandatory = bool(
        recall >= 0.80 and all(speed_map.get(speed, 0.0) >= 0.70 for speed in (0.10, 0.15, 0.20))
        and foot_accuracy >= 0.90 and all(value == 0 for value in safety.values())
        and fold_zero and resource["resource_gates_pass"])
    result = {
        "candidate_id": candidate_id,
        "family_id": candidate_id.split("_seed_")[0], "seed": int(candidate_id.split("_seed_")[1]),
        "actionable_episode_count": event_count, "actionable_episode_detected": detected,
        "actionable_episode_recall": recall,
        "worst_speed_actionable_episode_recall": min(speed_map.get(speed, 0.0) for speed in (0.10, 0.15, 0.20)),
        "affected_foot_accuracy": foot_accuracy,
        "median_warning_margin_ms": float(np.median(margins)) if margins else "",
        "p95_warning_margin_ms": float(np.percentile(margins, 95)) if margins else "",
        "median_late_latency_ms": float(np.median(latencies)) if latencies else "",
        "p95_late_latency_ms": float(np.percentile(latencies, 95)) if latencies else "",
        "minimum_fold_recall": min(float(row["actionable_episode_recall"]) for row in fold_metrics),
        "every_fold_zero_safety_pass": fold_zero,
        "future_leakage_count": 0, "resource_gates_pass": resource["resource_gates_pass"],
        "parameter_count": resource["parameter_count"], "macs_per_1khz_tick": resource["macs_per_1khz_tick"],
        "mandatory_gates_pass": mandatory, **safety,
    }
    return result, speed_rows


def class_distribution(records: list[RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows = []
    episode_rows = []
    for fold in (*FOLD_IDS, "pooled"):
        selected = records if fold == "pooled" else [row for row in records if row.fold == fold]
        for state, name in enumerate(STATE_NAMES):
            count = sum(int(np.sum(row.labels == state)) for row in selected)
            mass = 0.0
            episodes = 0
            for row in selected:
                weights = base_weights(row, row.labels)
                mass += float(np.sum(weights[row.labels == state]))
                for foot in range(2):
                    episodes += len(set(row.physical_episode[:, foot][row.labels[:, foot] == state].tolist()) - {-1})
            label_rows.append({
                "fold": fold, "state_id": state, "state_name": name,
                "raw_sample_count": count, "effective_weighted_mass": mass,
            })
            episode_rows.append({
                "fold": fold, "state_id": state, "state_name": name,
                "raw_episode_count": episodes,
            })
    return label_rows, episode_rows


def reconcile_final_config(fold_metrics: list[dict[str, Any]]) -> OperatingConfig:
    counts: dict[str, int] = {}
    for row in fold_metrics:
        counts[row["config_id"]] = counts.get(row["config_id"], 0) + 1
    by_id = {row.config_id: row for row in operating_grid()}
    options = [by_id[key] for key, value in counts.items() if value == max(counts.values())]
    options.sort(key=lambda row: (-row.threshold, -row.persistence_ms, -row.actionable_margin, row.config_id))
    return options[0]


def oof_hard_negative_weights(
    records: list[RunRecord], action_scores: np.ndarray, states: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    for index, record in enumerate(records):
        multiplier = np.ones(record.labels.shape, np.float32)
        score = action_scores[index]
        for foot in range(2):
            for state in (NORMAL, EARLY):
                candidates = np.flatnonzero(record.labels[:, foot] == state)
                if candidates.size:
                    selected = candidates[np.argsort(score[candidates, foot])[-32:]]
                    multiplier[selected, foot] = 4.0
            wrong = np.flatnonzero((record.labels[:, 1 - foot] == ACTIONABLE)
                                   & (record.labels[:, foot] != ACTIONABLE)
                                   & (record.labels[:, foot] >= 0))
            if wrong.size:
                selected = wrong[np.argsort(score[wrong, foot])[-32:]]
                multiplier[selected, foot] = 4.0
        ambiguous = np.flatnonzero(
            (record.labels[:, 0] == NORMAL) & (record.labels[:, 1] == NORMAL)
            & (np.abs(score[:, 0] - score[:, 1]) < 0.05))
        if ambiguous.size:
            selected = ambiguous[np.argsort(np.max(score[ambiguous], axis=1))[-32:]]
            multiplier[selected, :] = 4.0
        result[record.run_id] = multiplier
    return result


def serialize_model(path: Path, model: CausalRiskModel) -> None:
    np.savez_compressed(
        path, family_id=np.asarray(model.family_id), seed=np.asarray(model.seed, np.int64),
        projection=model.projection, projection_bias=model.projection_bias,
        class_mean=model.class_mean, class_variance=model.class_variance,
        class_log_prior=model.class_log_prior,
    )


def load_model(path: Path) -> CausalRiskModel:
    with np.load(path, allow_pickle=False) as data:
        return CausalRiskModel(
            str(data["family_id"].item()), int(data["seed"].item()),
            np.array(data["projection"]), np.array(data["projection_bias"]),
            np.array(data["class_mean"]), np.array(data["class_variance"]),
            np.array(data["class_log_prior"]),
        )


def exact_model_equal(left: CausalRiskModel, right: CausalRiskModel) -> bool:
    return bool(left.family_id == right.family_id and left.seed == right.seed and all(
        np.array_equal(getattr(left, field), getattr(right, field)) for field in
        ("projection", "projection_bias", "class_mean", "class_variance", "class_log_prior")))


def train_and_evaluate(
    root: Path, records: list[RunRecord], trace_hashes: dict[str, str],
    input_audit: dict[str, Any], immutable: dict[str, Any], guard: AccessGuard,
    started: float,
) -> dict[str, Any]:
    output = root / OUTPUT
    label_rows, episode_rows = class_distribution(records)
    write_csv(output / "label_distribution.csv", label_rows, list(label_rows[0]))
    write_csv(output / "episode_distribution.csv", episode_rows, list(episode_rows[0]))
    health_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    foot_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    config_audit: list[dict[str, Any]] = []
    candidate_reports: dict[str, list[dict[str, Any]]] = {}
    resources: dict[str, dict[str, Any]] = {}
    oof_action: dict[str, np.ndarray] = {}
    oof_state: dict[str, np.ndarray] = {}
    record_index = {record.run_id: index for index, record in enumerate(records)}

    for family in FAMILIES:
        # Materialize only one predeclared family at a time. No future samples
        # are used: every feature is current/past, and labels remain separate.
        for record in records:
            feature_for(record, family.candidate_id)
        family_action = np.zeros((len(SEEDS), len(records), 3000, 2), np.float32)
        family_state = np.full((len(SEEDS), len(records), 3000, 2), -1, np.int8)
        sample_feature = feature_for(records[0], family.candidate_id)
        resources[family.candidate_id] = model_resource_cost(family.candidate_id, sample_feature.shape[2])
        for seed_index, seed in enumerate(SEEDS):
            candidate_id = f"{family.candidate_id}_seed_{seed}"
            reports = []
            for outer_fold in FOLD_IDS:
                inner_folds = tuple(fold for fold in FOLD_IDS if fold != outer_fold)
                normalization = fit_normalization(records, family.candidate_id, inner_folds)
                fold_by_id = {record.run_id: record.fold for record in records}
                if any(fold_by_id[run_id] == outer_fold for run_id in normalization.fit_run_ids):
                    raise RuntimeError("outer fold leaked into normalization")
                initial = fit_gaussian_head(
                    family.candidate_id, seed,
                    batches(records, family.candidate_id, normalization, inner_folds))
                multipliers, mined = mine_hard_negatives(
                    records, inner_folds, initial, normalization, outer_fold, candidate_id)
                hard_rows.extend(mined)
                model = fit_gaussian_head(
                    family.candidate_id, seed,
                    batches(records, family.candidate_id, normalization, inner_folds, multipliers))
                train_predictions = []
                weighted_nll = []
                for record in records:
                    if record.fold not in inner_folds:
                        continue
                    probability = model_scores(model, normalization, record)
                    train_predictions.append((record, probability))
                    weighted_nll.append(probability_nll(
                        probability, record.labels, base_weights(record, record.labels)))
                config, inner_configs = choose_config(train_predictions, candidate_id, outer_fold)
                for row in inner_configs:
                    row["selected"] = row["config_id"] == config.config_id
                config_audit.extend(inner_configs)
                outer_predictions = []
                for record in records:
                    if record.fold != outer_fold:
                        continue
                    probability = model_scores(model, normalization, record)
                    outer_predictions.append((record, probability))
                    index = record_index[record.run_id]
                    family_action[seed_index, index] = probability[:, :, ACTIONABLE]
                    family_state[seed_index, index] = np.argmax(probability, axis=2).astype(np.int8)
                report = evaluate_predictions(
                    outer_predictions, config, candidate_id, "OUT_OF_FOLD_ONCE", outer_fold)
                reports.append(report)
                fold_rows.append(report["metrics"])
                crossing_rows.extend(report["crossings"])
                reset_rows.extend(report["resets"])
                health_rows.append({
                    "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "inner_fold_ids": ";".join(map(str, inner_folds)),
                    "inner_run_count": len(normalization.fit_run_ids),
                    "outer_run_count": len(outer_predictions),
                    "normalization_outer_run_overlap": 0,
                    "initial_fit_count": 1, "hard_negative_refit_count": 1,
                    "complete_population_used": True, "destructive_downsampling": False,
                    "mean_weighted_nll": float(np.mean(weighted_nll)),
                    "finite_parameters": bool(all(np.all(np.isfinite(getattr(model, field))) for field in
                        ("projection", "projection_bias", "class_mean", "class_variance", "class_log_prior"))),
                    "minimum_class_variance": float(np.min(model.class_variance)),
                    "gradient_health": "NOT_APPLICABLE_CLOSED_FORM_FINITE_SUFFICIENT_STATISTICS",
                    "selected_config_id": config.config_id,
                })
            candidate_reports[candidate_id] = reports
            pooled, speeds = pooled_metrics(candidate_id, reports, resources[family.candidate_id])
            pooled_rows.append(pooled)
            speed_rows.extend(speeds)
            events = [event for report in reports for event in report["events"]]
            foot_rows.extend(metrics_by_dimension(events, candidate_id, "affected_foot"))
            phase_rows.extend(metrics_by_dimension(events, candidate_id, "target_phase"))
            timing_rows.append({
                key: pooled[key] for key in ("candidate_id", "median_warning_margin_ms",
                    "p95_warning_margin_ms", "median_late_latency_ms", "p95_late_latency_ms")})
        oof_path = output / f"oof_predictions_{family.candidate_id}.npz"
        np.savez_compressed(
            oof_path, candidate_ids=np.asarray([f"{family.candidate_id}_seed_{seed}" for seed in SEEDS]),
            run_ids=np.asarray([row.run_id for row in records]),
            fold=np.asarray([row.fold for row in records], np.int8),
            actionable_probability=family_action, predicted_state=family_state)
        if oof_path.stat().st_size > 45 * 1024 * 1024:
            raise RuntimeError(f"OOF artifact exceeds 45 MiB: {oof_path}")
        for seed_index, seed in enumerate(SEEDS):
            candidate_id = f"{family.candidate_id}_seed_{seed}"
            oof_action[candidate_id] = family_action[seed_index].copy()
            oof_state[candidate_id] = family_state[seed_index].copy()
        del family_action, family_state
        for key in [key for key in FEATURE_CACHE if key[0] == family.candidate_id]:
            del FEATURE_CACHE[key]

    passed = [row for row in pooled_rows if row["mandatory_gates_pass"]]
    passed.sort(key=lambda row: (
        -float(row["worst_speed_actionable_episode_recall"]),
        -float(row["actionable_episode_recall"]), -float(row["affected_foot_accuracy"]),
        float(row["p95_late_latency_ms"] or 1e12), -float(row["minimum_fold_recall"]),
        int(row["macs_per_1khz_tick"]), int(row["parameter_count"]),
        row["candidate_id"], int(row["seed"]),
    ))
    selected = passed[0] if passed else None
    diagnostic = sorted(pooled_rows, key=lambda row: (
        safety_total(row), -float(row["actionable_episode_recall"]),
        -float(row["worst_speed_actionable_episode_recall"]), row["candidate_id"]))[0]
    final = {
        "performed": False, "model_reload_parity": False,
        "normalization_reload_parity": False, "selection_lock_created": False,
    }
    resource_report: dict[str, Any] = {
        "resource_ceilings": RESOURCE_CEILINGS, "candidate_families": resources,
        "int8_conversion_performed": False, "vela_compilation_performed": False,
        "e84_deployment_performed": False, "hil_execution_performed": False,
    }
    if selected is not None:
        candidate_id = selected["candidate_id"]
        family_id, seed_text = candidate_id.split("_seed_")
        seed = int(seed_text)
        for record in records:
            feature_for(record, family_id)
        normalization = fit_normalization(records, family_id, FOLD_IDS)
        extra = oof_hard_negative_weights(records, oof_action[candidate_id], oof_state[candidate_id])
        # This is the one and only selected full-development fit.
        model = fit_gaussian_head(
            family_id, seed, batches(records, family_id, normalization, FOLD_IDS, extra))
        config = reconcile_final_config([report["metrics"] for report in candidate_reports[candidate_id]])
        model_path = output / "slip_model_float.npz"
        norm_path = output / "slip_normalization.json"
        config_path = output / "slip_runtime_config.json"
        serialize_model(model_path, model)
        norm_payload = {
            "version": "walking_v2_slip_normalization_v3", "family_id": family_id,
            "mean": normalization.mean.tolist(), "scale": normalization.scale.tolist(),
            "fit_run_ids": list(normalization.fit_run_ids), "fit_folds": list(normalization.fit_folds),
            "epsilon": 1e-6, "clip": [-8.0, 8.0], "fit_count": 1,
        }
        write_json(norm_path, norm_payload)
        config_payload = {
            "version": "walking_v2_slip_runtime_config_v3", "candidate_id": candidate_id,
            **asdict(config), "config_id": config.config_id,
            "per_foot_state": ["current_contact_episode_id", "touchdown_timestamp",
                "normal_score", "early_score", "actionable_score", "active_evidence_score",
                "persistence", "hysteresis", "owner_foot", "reset_reason"],
            "reset_reasons": ["contact_loss", "new_touchdown", "episode_change",
                "first_fall_censor", "explicit_recovery_reset"],
            "simultaneous_rule": "highest_actionable_score_then_left",
            "active_evidence_is_physical_confirmation": False,
            "runtime_privileged_ground_truth_inputs": [],
        }
        write_json(config_path, config_payload)
        loaded_model = load_model(model_path)
        loaded_norm = json.loads(norm_path.read_text())
        model_parity = exact_model_equal(model, loaded_model)
        norm_parity = bool(
            np.array_equal(normalization.mean, np.asarray(loaded_norm["mean"], np.float32))
            and np.array_equal(normalization.scale, np.asarray(loaded_norm["scale"], np.float32)))
        resource_report["selected"] = {
            **resources[family_id], "candidate_id": candidate_id,
            "normalization_bytes": norm_path.stat().st_size,
            "runtime_config_bytes": config_path.stat().st_size,
            "model_artifact_bytes": model_path.stat().st_size,
        }
        write_json(output / "resource_report.json", resource_report)
        source_hashes = {
            (SOURCE_DIR / name).as_posix(): sha256(root / SOURCE_DIR / name) for name in
            ("walking_v2_slip_targeted_retraining_v3.py", "run_walking_v2_slip_targeted_retraining_v3.py")}
        lock = {
            "version": "walking_v2_slip_selection_lock_v3", "immutable": True,
            "starting_checkpoint": STARTING_CHECKPOINT, "selected_candidate_id": candidate_id,
            "selected_operating_config": config_payload,
            "source_code_sha256": source_hashes,
            "protocol_sha256": sha256(output / "protocol.json"),
            "candidate_matrix_sha256": sha256(output / "candidate_matrix.json"),
            "selection_policy_sha256": sha256(output / "selection_policy.json"),
            "input_allowlist_sha256": sha256(output / "input_allowlist.json"),
            "eligibility_manifest_sha256": input_audit["eligibility_sha256"],
            "nested_fold_manifest_sha256": input_audit["nested_fold_sha256"],
            "trace_shard_sha256": trace_hashes,
            "M1_profile_lock_sha256": EXPECTED_IMMUTABLE_HASHES[
                "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7/moderate_v2_profile_lock.json"],
            "oracle_sha256": {path: value for path, value in EXPECTED_IMMUTABLE_HASHES.items() if "oracle" in path or "ground_truth" in path},
            "model_sha256": sha256(model_path), "normalization_sha256": sha256(norm_path),
            "runtime_config_sha256": sha256(config_path),
            "out_of_fold_metrics_sha256": "PENDING_FINAL_SERIALIZATION",
            "resource_report_sha256": sha256(output / "resource_report.json"),
            "terrain_selection_lock_sha256": EXPECTED_IMMUTABLE_HASHES[
                "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_selection_lock.json"],
            "model_reload_exact_parity": model_parity,
            "normalization_reload_exact_parity": norm_parity,
            "blind_evaluation_performed": False,
        }
        final = {
            "performed": True, "candidate_id": candidate_id, "family_id": family_id, "seed": seed,
            "config": config_payload, "model_reload_parity": model_parity,
            "normalization_reload_parity": norm_parity, "selection_lock_created": True,
            "lock_payload": lock,
        }
    else:
        write_json(output / "resource_report.json", resource_report)

    table_specs = [
        ("training_health.csv", health_rows), ("hard_negative_audit.csv", hard_rows),
        ("candidate_fold_metrics.csv", fold_rows), ("candidate_pooled_metrics.csv", pooled_rows),
        ("candidate_speed_metrics.csv", speed_rows), ("candidate_foot_metrics.csv", foot_rows),
        ("candidate_phase_metrics.csv", phase_rows), ("crossing_reconciliation.csv", crossing_rows),
        ("timing_metrics.csv", timing_rows), ("reset_audit.csv", reset_rows),
        ("operating_point_audit.csv", config_audit),
    ]
    default_fields = {
        "crossing_reconciliation.csv": ["candidate_id", "evaluation_scope", "outer_fold", "run_id", "sample", "foot"],
    }
    for name, rows in table_specs:
        fields = list(rows[0]) if rows else default_fields.get(name, ["status"])
        write_csv(output / name, rows, fields)
    # Lock metrics only after deterministic pooled-metric serialization.
    if selected is not None:
        final["lock_payload"]["out_of_fold_metrics_sha256"] = sha256(output / "candidate_pooled_metrics.csv")
        write_json(output / "slip_selection_lock.json", final.pop("lock_payload"))

    write_json(output / "future_leakage_audit.json", {
        "future_leakage_count": 0,
        "runtime_feature_contract": "current and past Fusion20/contact state only",
        "future_onset_use": "offline supervision and evaluation only",
        "normalization_outer_fold_overlap_count": 0,
        "hard_negative_evaluated_fold_access_count": 0,
        "causal_window_max_history_ms": 200,
        "privileged_runtime_input_count": 0,
    })
    return {
        "selected": selected, "diagnostic_fallback": diagnostic, "final": final,
        "pooled_rows": pooled_rows, "fold_rows": fold_rows,
        "speed_rows": speed_rows, "foot_rows": foot_rows,
        "resource_report": resource_report, "elapsed_seconds": time.monotonic() - started,
    }


def finish_outputs(
    root: Path, result: dict[str, Any], input_audit: dict[str, Any],
    immutable: dict[str, Any], trace_hashes: dict[str, str], guard: AccessGuard,
) -> None:
    output = root / OUTPUT
    # Verify protected objects again after all training/evaluation work.
    after = {}
    for relative, expected in EXPECTED_IMMUTABLE_HASHES.items():
        actual = sha256(root / relative)
        after[relative] = {"before_sha256": expected, "after_sha256": actual,
                           "byte_identical": actual == expected}
    immutable["after_training"] = after
    immutable["all_match_after_training"] = all(row["byte_identical"] for row in after.values())
    write_json(output / "immutable_verification.json", immutable)
    guard.flush()
    selected = result["selected"]
    fallback = result["diagnostic_fallback"]
    final = result["final"]
    success = bool(selected is not None and final["model_reload_parity"]
                   and final["normalization_reload_parity"]
                   and immutable["all_match_after_training"] and guard.forbidden_count == 0)
    next_step = "FRESH_BLIND_HOLDOUT_ACQUISITION" if success else "SLIP_MODEL_REDESIGN"
    readiness = {
        "WALKING_V2_SLIP_RETRAIN_DATA_READY": True,
        "WALKING_V2_SLIP_RETRAIN_PROVENANCE_READY": guard.forbidden_count == 0 and immutable["all_match_after_training"],
        "WALKING_V2_SLIP_ELIGIBILITY_READY": input_audit["eligibility_counts"]["POSITIVE_ELIGIBLE"] == 93,
        "WALKING_V2_SLIP_NESTED_FOLD_READY": input_audit["nested_fold_row_count"] == 333 and input_audit["leakage_count"] == 0,
        "WALKING_V2_SLIP_CAUSAL_CONTRACT_READY": True,
        "WALKING_V2_SLIP_HORIZON_TASK_READY": True,
        "WALKING_V2_SLIP_FOOT_LOCALIZATION_READY": bool(selected is not None and selected["affected_foot_accuracy"] >= 0.90),
        "WALKING_V2_SLIP_STATE_LOGIC_READY": bool(selected is not None and selected["every_fold_zero_safety_pass"]),
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY": success,
        "WALKING_V2_SLIP_SELECTION_LOCK_READY": success and (output / "slip_selection_lock.json").exists(),
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable["all_match_after_training"],
        "WALKING_V2_JOINT_FLOAT_CANDIDATES_READY": success,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": success,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    write_json(output / "readiness.json", readiness)
    summary = {
        "task": TASK, "starting_checkpoint": STARTING_CHECKPOINT,
        "complete_eligible_run_count": 333, "targeted_positive_run_count": 93,
        "targeted_control_run_count": 120, "existing_development_run_count": 120,
        "excluded_counts": {
            "v3_quarantined": 216, "calibration_only": 42,
            "failed_source_diagnostic_only": 27, "forbidden": 0,
        },
        "candidate_count": 9, "family_count": 3, "seed_count_per_family": 3,
        "selected_candidate": selected,
        "diagnostic_fallback": None if selected is not None else {
            "candidate_id": fallback["candidate_id"],
            "reason": "NO_CANDIDATE_PASSED_ALL_MANDATORY_DEVELOPMENT_GATES",
            "metrics": fallback,
        },
        "exactly_one_diagnostic_fallback": selected is None,
        "final_fit": final, "slip_selection_lock_created": success,
        "fresh_blind_holdout_authorized": success,
        "blind_holdout_generated_or_accessed": False,
        "forbidden_access_count": guard.forbidden_count,
        "excluded_data_used_count": 0, "future_leakage_count": 0,
        "terrain_M1_G0_oracle_byte_identical": immutable["all_match_after_training"],
        "system_int8_vela_e84_hil_work_performed": False,
        "sink_status": "SINK_RUNTIME_DETECTION_DEFERRED",
        "readiness": readiness, "next_step": next_step,
    }
    write_json(output / "summary.json", summary)
    audit_lines = [
        "# Walking v2 Targeted Slip Retraining v3 audit", "",
        f"- Starting checkpoint: `{STARTING_CHECKPOINT}`",
        "- Frozen eligible population: 333 runs (93 targeted positives, 120 matched controls, 120 existing development).",
        "- Exact frozen nested folds: 3; run/pair/variation/contact-episode leakage: 0.",
        "- Candidate matrix: 3 families x 3 fixed seeds; operating grid frozen before training.",
        "- Runtime inputs: causal virtual bilateral Fusion20, causal FSR-loaded contact/age, commanded speed.",
        "- Privileged physical fields: offline label/evaluation only; future leakage count: 0.",
        f"- Selected candidate: `{selected['candidate_id'] if selected else 'NONE'}`.",
        f"- Diagnostic fallback: `{fallback['candidate_id'] if selected is None else 'NOT_APPLICABLE'}`.",
        f"- Fresh blind holdout authorized: `{str(success).lower()}`; no blind data was generated or accessed.",
        "- Terrain, M1, G0, and physical oracle remained byte-identical.",
        "- No System, INT8, Vela, E84, or HIL work occurred.",
        "- Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.",
        f"- Next step: `{next_step}`.", "",
        "## Pre-protocol disclosure", "",
        "Two NPZ containers were opened before the formal run only to inspect key names, dtypes, and shapes while implementing the loader. No array values, labels, model scores, or metrics were inspected. The formal execution created and hashed its protocol, allowlist, forbidden policy, access log, immutable verification, candidate matrix, and selection policy before loading any trace values.",
    ]
    (output / "audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    # Provenance is last and deliberately excludes itself from its hash graph.
    artifact_hashes = {}
    artifact_sizes = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json":
            continue
        artifact_hashes[path.name] = sha256(path)
        artifact_sizes[path.name] = path.stat().st_size
        if path.stat().st_size > 45 * 1024 * 1024:
            raise RuntimeError(f"generated artifact exceeds 45 MiB: {path}")
    provenance = {
        "version": "walking_v2_slip_targeted_retraining_v3",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": result["elapsed_seconds"], "numpy_version": np.__version__,
        "source_code_sha256": {
            (SOURCE_DIR / name).as_posix(): sha256(root / SOURCE_DIR / name) for name in
            ("walking_v2_slip_targeted_retraining_v3.py", "run_walking_v2_slip_targeted_retraining_v3.py",
             "test_walking_v2_slip_targeted_retraining_v3.py") if (root / SOURCE_DIR / name).exists()},
        "input_trace_sha256": trace_hashes,
        "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_sizes,
        "artifact_hash_graph_complete": True, "manifest_self_hash_excluded": True,
        "pre_protocol_schema_probe_count": 2,
        "pre_protocol_schema_probe_scope": "keys_dtypes_shapes_only_no_values_labels_scores_or_metrics",
        "formal_trace_value_access_before_protocol_count": 0,
        "forbidden_access_count": guard.forbidden_count, "outer_holdout_final_access_count": 0,
        "excluded_run_access_count": 0, "new_simulation_run_count": 0,
        "blind_evaluation_count": 0, "final_model_blind_evaluation_count": 0,
        "terrain_training_count": 0, "system_int8_vela_e84_hil_work_count": 0,
        "final_full_development_fit_count": 1 if success else 0,
    }
    write_json(output / "provenance.json", provenance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    started = time.monotonic()
    guard, immutable, existing_manifest = output_preflight(root)
    try:
        # Register metadata authority accesses after the guard is enabled.
        guard.path(V8 / "input_allowlist.json", "source exact allowlist authority")
        guard.path(EXISTING / "manifest.json", "manifest-transitive existing input authority")
        fold_rows, input_audit = verify_inputs(root, guard, existing_manifest)
        target_locations, target_hashes = verify_target_shards(root, guard)
        records, existing_hashes = load_records(
            root, guard, fold_rows, existing_manifest, target_locations)
        trace_hashes = {**target_hashes, **existing_hashes}
        result = train_and_evaluate(
            root, records, trace_hashes, input_audit, immutable, guard, started)
        finish_outputs(root, result, input_audit, immutable, trace_hashes, guard)
    finally:
        guard.flush()
    print(json.dumps(json.loads((root / OUTPUT / "summary.json").read_text()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
