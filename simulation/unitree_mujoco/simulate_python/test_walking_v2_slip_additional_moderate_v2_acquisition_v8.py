"""Focused contracts and executed-artifact tests for Slip acquisition v8."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from run_walking_v2_slip_additional_moderate_v2_acquisition_v8 import (
    AccessGuard, DEFAULT_OUTPUT, POLICY, REPO, STARTING_CHECKPOINT,
    TERRAIN_FILES, future_folds, sha256_file,
)
import run_walking_v2_slip_moderate_profile_recalibration_v7 as v7_runner
from walking_hazard_ground_truth_v1 import derive_contact_signals
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S, SAMPLE_RATE_HZ,
)
from walking_v2_slip_additional_moderate_v2_acquisition_v8 import (
    COMMAND_DELAYS_S, PHASE_FRACTIONS, PHASE_OFFSETS, PREREGISTERED_SEEDS,
    supplemental_matrix, supplemental_variations, variation_contract_payload,
    variation_contract_sha256,
)
from walking_v2_slip_moderate_profile_recalibration_v7 import candidate_profile
from walking_v2_slip_supplemental_acquisition_v6 import GEOMETRY_CANDIDATES


def _checkpoint_hash(relative: str) -> str:
    value = subprocess.check_output(
        ("git", "show", f"{STARTING_CHECKPOINT}:{relative}"), cwd=REPO)
    return hashlib.sha256(value).hexdigest()


def _csv(name: str) -> list[dict[str, str]]:
    with (DEFAULT_OUTPUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_variation_contract_is_frozen_exact_three_by_four_lattice() -> None:
    variations = supplemental_variations()
    assert len(variations) == 12
    assert len({row.variation_id for row in variations}) == 12
    assert len({row.seed for row in variations}) == 12
    assert tuple(row.seed for row in variations) == PREREGISTERED_SEEDS
    assert {(row.phase_fraction, row.command_delay_s) for row in variations} == {
        (phase, delay) for phase in PHASE_FRACTIONS for delay in COMMAND_DELAYS_S}
    assert PHASE_OFFSETS == pytest.approx((-0.14, 0.0, 0.14))
    assert sum(row.control_type == "hard_normal" for row in variations) == 6
    assert sum(row.control_type == "near_slip_non_event" for row in variations) == 6
    assert len(variation_contract_sha256()) == 64
    payload = variation_contract_payload()
    assert payload["frozen_before_first_supplemental_result"]
    assert payload["primary_diagnosis"] == "PATCH_ENTRY_POSE_SENSITIVITY"
    assert not payload["adaptive_additions"]
    assert not payload["seed_replacement"]


def test_supplemental_matrix_is_exact_unique_matched_and_balanced() -> None:
    rows = supplemental_matrix()
    positives = [row for row in rows if row.role == "positive"]
    controls = [row for row in rows if row.role == "control"]
    assert len(rows) == 24
    assert len(positives) == len(controls) == 12
    assert len({row.run_id for row in rows}) == 24
    assert len({row.pair_id for row in rows}) == 12
    assert len({row.pair_fingerprint for row in positives}) == 12
    assert len({row.seed for row in positives}) == 12
    assert all(row.speed_mps == 0.15 for row in rows)
    assert all(row.target_foot == "left" for row in rows)
    assert all(row.target_phase == "mid_late_stance" for row in rows)
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.pair_fingerprint for row in pair}) == 1
        assert len({row.material_profile for row in pair}) == 2


def test_M1_G0_timing_and_fusion20_contracts_are_exact() -> None:
    profile = candidate_profile("M1")
    assert profile.friction5 == pytest.approx(
        (0.0875, 0.0875, 0.00125, 0.00002125, 0.00002125), abs=1e-15)
    assert GEOMETRY_CANDIDATES[0].candidate_id == "G0"
    assert vars(GEOMETRY_CANDIDATES[0].bounds) == {
        "x_min_m": -1.0, "x_max_m": 2.0,
        "y_min_m": -0.8, "y_max_m": 0.8, "top_z_m": 0.0}
    assert SAMPLE_RATE_HZ == 1000
    assert PHYSICS_TIMESTEP_S == 0.0005
    assert PHYSICS_STEPS_PER_SAMPLE == 2


def test_M1_lock_and_terrain_files_equal_starting_checkpoint() -> None:
    lock = "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7/moderate_v2_profile_lock.json"
    assert sha256_file(REPO / lock) == _checkpoint_hash(lock)
    for relative in TERRAIN_FILES.values():
        assert sha256_file(REPO / relative) == _checkpoint_hash(relative)


def test_short_pair_has_exact_timing_friction_parity_and_no_double_contact() -> None:
    pair = supplemental_matrix(1.0)[:2]
    results = [v7_runner.collect_run(row, REPO / POLICY) for row in pair]
    traces = [row[0] for row in results]
    manifests = [row[1] for row in results]
    contacts = [item for result in results for item in result[2]]
    _, parity = v7_runner.build_pair_audits(pair, traces, manifests)
    assert all(trace["bilateral_fusion20_raw"].shape == (1000, 20) for trace in traces)
    assert all(np.all(np.isfinite(trace["bilateral_fusion20_raw"])) for trace in traces)
    assert all(float(meta["sample_spacing_max_error_s"]) < 1e-12 for meta in manifests)
    assert parity[0]["parity_pass"]
    assert parity[0]["postcontact_trajectory_diverged"]
    assert not parity[0]["full_trace_identical"]
    assert parity[0]["positive_effective_friction_contract"]
    assert parity[0]["control_effective_friction_contract"]
    assert parity[0]["patch_base_double_contact_count"] == 0
    assert contacts
    assert all(int(row["explicit_pair_id"]) >= 0 for row in contacts)
    assert all(int(row["efc_address_min"]) >= 0 for row in contacts)
    assert all(row["patch_base_double_contact"] is False for row in contacts)
    assert all(row["friction_configured_before_collision"] is True for row in contacts)


def test_physical_oracle_excludes_air_touchdown_transient_and_first_fall() -> None:
    contact = np.zeros(40, bool); contact[1:39] = True
    xyz = np.zeros((40, 3)); xyz[1:39, 0] = np.linspace(0.0, 0.20, 38)
    signals = derive_contact_signals(
        contact, contact, xyz, np.zeros((40, 3)), np.zeros(40),
        first_fall_sample=25)
    active = persistent_oracle(
        signals.tangential_anchor_drift_m, signals.slip_calibration_valid,
        signals.contact_episode_id, 0.050, 3)
    assert not signals.pre_fall_valid[25:].any()
    assert not np.any(signals.slip_calibration_valid & signals.touchdown_transient)
    assert not np.any(active & ~signals.pre_fall_valid)
    assert not np.any(active & ~contact)


def test_forbidden_guard_fails_closed(tmp_path: Path) -> None:
    guard = AccessGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.path("simulation/outputs/outer/secret.json")
    log = json.loads((tmp_path / "artifact_access_log.json").read_text())
    assert log["blocked_access_count"] == 1
    assert log["detector_or_model_score_access_count"] == 0


def test_future_fold_helper_has_no_pair_variation_run_or_episode_leakage() -> None:
    augmented_rows = []
    for variation in range(3):
        for severity, source in (
            ("native_strong_ice", "v5_strong"),
            ("moderate_v2", "v7_moderate_v2"),
        ):
            for speed in (0.10, 0.15, 0.20):
                for foot in ("left", "right"):
                    for phase in (
                        "early_loading", "mid_loading_early_stance", "mid_late_stance"):
                        for role, control in (
                            ("positive", "hard_normal"),
                            ("control", "near_slip_non_event"),
                        ):
                            tag = f"{source}_{variation}_{speed}_{foot}_{phase}_{role}"
                            augmented_rows.append({
                                "source_acquisition": source,
                                "acquisition_version": source.split("_")[0],
                                "run_id": tag, "pair_id": f"pair:{tag[:-len(role)]}",
                                "role": role, "speed_mps": speed,
                                "target_foot": foot, "target_phase": phase,
                                "severity": severity, "control_type": control,
                                "variation_index": variation, "source_valid": True,
                                "future_training_eligible": True,
                                "development_only": True,
                            })
    existing = [{
        "run_id": f"existing_{index}", "variation_index": index,
        "speed_mps": (0.10, 0.15, 0.20)[index], "role": "hard_negative"}
        for index in range(3)]
    folds = future_folds({"audit": {"pass": True}, "runs": augmented_rows}, existing)
    assert folds["audit"]["leakage_count"] == 0
    assert folds["audit"]["positive_control_pairs_same_fold"]
    assert folds["audit"]["run_leakage_count"] == 0
    assert folds["audit"]["contact_episode_leakage_count"] == 0
    assert folds["audit"]["variation_leakage_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_required_artifacts_and_readiness() -> None:
    required = {
        "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
        "artifact_access_log.json", "immutable_verification.json",
        "variation_failure_diagnosis.csv", "variation_root_cause.json",
        "supplemental_variation_contract.json", "supplemental_run_manifest.csv",
        "supplemental_pair_manifest.csv", "supplemental_contact_audit.csv",
        "supplemental_physical_episode_ledger.csv",
        "supplemental_positive_source_audit.csv",
        "supplemental_control_source_audit.csv", "supplemental_pair_parity.csv",
        "supplemental_fall_censor_audit.csv", "supplemental_variation_metrics.csv",
        "duplicate_audit.json", "trace_shard_manifest.json", "provenance.json",
        "readiness.json", "summary.json", "audit.md", "resource_size_audit.json",
    }
    names = {path.name for path in DEFAULT_OUTPUT.iterdir()}
    assert required <= names
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["supplemental_positive_runs"] == 12
    assert summary["supplemental_control_runs"] == 12
    assert summary["base_valid_positive_count"] == 87
    assert summary["base_positive_attempts"] == 108
    assert summary["base_factorial_coverage"] == 35
    if summary["supplemental_gates"]["pass"]:
        conditional = {
            "canonical_base_manifest.json", "supplemental_cell_manifest.json",
            "augmented_canonical_development_manifest.json",
            "augmented_factorial_coverage.csv", "training_eligibility.csv",
            "future_nested_fold_manifest.json"}
        assert conditional <= names


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_diagnosis_is_exactly_one_physical_primary_cause() -> None:
    root = json.loads((DEFAULT_OUTPUT / "variation_root_cause.json").read_text())
    rows = _csv("variation_failure_diagnosis.csv")
    assert root["primary_cause"] == "PATCH_ENTRY_POSE_SENSITIVITY"
    assert root["exactly_one_primary_cause"]
    assert root["diagnosis_uses_physical_signals_only"]
    assert not root["detector_or_model_scores_used"]
    assert len(rows) == 4
    assert sum(row["comparison"] == "SUCCESSFUL_CALIBRATION" for row in rows) == 1
    assert sum(row["comparison"] == "FAILED_V7_VARIATION" for row in rows) == 3


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_supplemental_physics_onset_and_parity_gates() -> None:
    manifests = _csv("supplemental_run_manifest.csv")
    positives = _csv("supplemental_positive_source_audit.csv")
    controls = _csv("supplemental_control_source_audit.csv")
    parity = _csv("supplemental_pair_parity.csv")
    assert len(manifests) == 24
    assert len(positives) == len(controls) == len(parity) == 12
    assert all(row["sample_count"] == "3000" for row in manifests)
    assert all(row["fusion20_shape"] == "[3000, 20]" for row in manifests)
    assert all(row["parity_pass"] == "True" for row in parity)
    assert all(row["postcontact_trajectory_diverged"] == "True" for row in parity)
    assert all(row["full_trace_identical"] == "False" for row in parity)
    assert all(row["patch_base_double_contact_count"] == "0" for row in parity)
    valid = [row for row in positives if row["source_valid"] == "True"]
    assert len(valid) >= 6
    assert len({row["variation_id"] for row in valid}) >= 3
    assert all(row["actual_onset_phase"] == "mid_late_stance" for row in valid)
    assert all(row["target_first"] == "True" for row in valid)
    assert all(row["physical_slip_onset_free"] == "True" for row in controls)
    assert all(row["control_source_valid"] == "True" for row in controls)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_eligibility_augmented_manifest_and_base_denominator() -> None:
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    if not summary["supplemental_gates"]["pass"]:
        pytest.skip("conditional augmented artifacts correctly absent")
    base = json.loads((DEFAULT_OUTPUT / "canonical_base_manifest.json").read_text())
    augmented = json.loads(
        (DEFAULT_OUTPUT / "augmented_canonical_development_manifest.json").read_text())
    eligibility = _csv("training_eligibility.csv")
    assert base["base_valid_positive_count"] == 87
    assert base["base_positive_attempts"] == 108
    assert base["base_valid_positive_rate"] == pytest.approx(87 / 108)
    assert base["base_factorial_coverage"] == "35/36"
    assert not base["base_v7_claimed_ready"]
    assert augmented["supplemental_attempts_not_added_to_base_efficiency_denominator"]
    assert augmented["audit"]["augmented_factorial_coverage"] == 36
    assert augmented["audit"]["missing_cell_distinct_valid_count"] >= 3
    assert augmented["audit"]["pass"]
    excluded = {
        "FAILED_SOURCE_DIAGNOSTIC_ONLY", "CALIBRATION_ONLY_DO_NOT_TRAIN",
        "QUARANTINED", "FORBIDDEN"}
    assert not any(
        row["training_eligible"] == "True" for row in eligibility
        if row["eligibility"] in excluded)
    assert sum(row["eligibility"] == "CALIBRATION_ONLY_DO_NOT_TRAIN"
               for row in eligibility) == 42
    assert sum(row["eligibility"] == "QUARANTINED" for row in eligibility) == 216


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_future_folds_have_full_coverage_and_no_leakage() -> None:
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    if not summary["future_fold_manifest_created"]:
        pytest.skip("conditional fold manifest correctly absent")
    folds = json.loads((DEFAULT_OUTPUT / "future_nested_fold_manifest.json").read_text())
    audit = folds["audit"]
    assert audit["valid"]
    assert audit["existing_valid_development_run_count"] == 120
    assert audit["leakage_count"] == 0
    assert audit["positive_control_pairs_same_fold"]
    assert audit["supplemental_variations_distributed_across_folds"]
    assert audit["coverage_pass"]
    assert all(all(value for key, value in row.items() if key != "fold")
               for row in audit["coverage"])
    assert audit["calibration_only_run_count"] == 0
    assert audit["failed_source_run_count"] == 0
    assert audit["v3_quarantined_run_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v8 runner first")
def test_executed_shards_hash_graph_immutability_access_and_size() -> None:
    shards = json.loads((DEFAULT_OUTPUT / "trace_shard_manifest.json").read_text())
    assert shards["all_under_45_mib"]
    assert shards["all_exact_roundtrip_verified"]
    assert sum(row["run_count"] for row in shards["shards"]) == 24
    for row in shards["shards"]:
        assert sha256_file(DEFAULT_OUTPUT / row["path"]) == row["sha256"]
        with np.load(DEFAULT_OUTPUT / row["path"], allow_pickle=False) as loaded:
            assert loaded.files
    provenance = json.loads((DEFAULT_OUTPUT / "provenance.json").read_text())
    assert provenance["artifact_hash_graph_verified"]
    for name, digest in provenance["artifact_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / name) == digest
    immutable = json.loads((DEFAULT_OUTPUT / "immutable_verification.json").read_text())
    assert immutable["all_immutable_after_execution"]
    assert immutable["M1_profile_lock_byte_identical"]
    assert immutable["terrain_byte_identical"]
    assert immutable["physical_oracle_byte_identical"]
    assert immutable["strong_native_ice_unchanged"]
    assert immutable["G0_geometry_unchanged"]
    access = json.loads((DEFAULT_OUTPUT / "artifact_access_log.json").read_text())
    assert access["blocked_access_count"] == 0
    assert access["detector_or_model_score_access_count"] == 0
    sizes = json.loads((DEFAULT_OUTPUT / "resource_size_audit.json").read_text())
    assert sizes["all_files_within_limit"]
