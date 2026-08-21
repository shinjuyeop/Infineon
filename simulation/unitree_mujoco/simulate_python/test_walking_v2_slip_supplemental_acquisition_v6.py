"""Focused contracts and executed-artifact tests for Slip supplement v6."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from run_walking_v2_slip_supplemental_acquisition_v6 import (
    AccessGuard, DEFAULT_OUTPUT, POLICY, REPO, build_combined_manifest,
    build_future_folds, collect_run, sha256_file,
)
from walking_hazard_ground_truth_v1 import derive_contact_signals
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, SLIP_PERSISTENCE_MS, SLIP_THRESHOLD_M, material_profiles,
)
from walking_v2_slip_supplemental_acquisition_v6 import (
    CALIBRATION_DISPOSITION, GEOMETRY_CANDIDATES, SUPPLEMENTAL_VARIATIONS,
    calibration_matrix, frozen_profile_sha256, geometry_contract_sha256,
    select_geometry, supplemental_matrix,
)


def test_geometry_candidates_are_bounded_and_change_only_longitudinal_contract() -> None:
    assert [row.candidate_id for row in GEOMETRY_CANDIDATES] == ["G0", "G1", "G2", "G3"]
    frozen = GEOMETRY_CANDIDATES[0]
    assert frozen.bounds.x_min_m == -1.0
    assert frozen.bounds.x_max_m == 2.0
    assert len({row.sha256 for row in GEOMETRY_CANDIDATES}) == 4
    assert len(geometry_contract_sha256()) == 64
    for row in GEOMETRY_CANDIDATES:
        assert row.width_m == frozen.width_m
        assert row.top_height_m == frozen.top_height_m
        assert row.contract["explicit_sole_pair_count"] == 8
        assert row.contract["friction_profile_unchanged"]
        assert row.contract["solref_solimp_unchanged"]
        assert not row.contract["patch_base_overlap"]


def test_calibration_matrix_is_exact_matched_and_calibration_only() -> None:
    rows = calibration_matrix()
    assert len(rows) == 24
    assert len({row.run_id for row in rows}) == 24
    assert len({row.pair_id for row in rows}) == 12
    assert sum(row.role == "positive" for row in rows) == 12
    assert sum(row.role == "control" for row in rows) == 12
    assert {row.geometry_candidate for row in rows} == {"G0", "G1", "G2", "G3"}
    assert {row.speed_mps for row in rows} == {0.10, 0.15, 0.20}
    assert all(row.disposition == CALIBRATION_DISPOSITION for row in rows)
    assert all(row.variation_index == 1 for row in rows)
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.pair_fingerprint for row in pair}) == 1
        assert len({row.material_profile for row in pair}) == 2


def test_supplemental_matrix_is_fixed_new_and_balanced_within_speed() -> None:
    rows = supplemental_matrix("G3")
    assert len(rows) == 36
    assert len({row.run_id for row in rows}) == 36
    assert len({row.pair_id for row in rows}) == 18
    positives = [row for row in rows if row.role == "positive"]
    controls = [row for row in rows if row.role == "control"]
    assert len(positives) == len(controls) == 18
    assert {row.variation_index for row in positives} == {
        row.index for row in SUPPLEMENTAL_VARIATIONS}
    assert all(row.geometry_candidate == "G3" for row in rows)
    for speed in (0.10, 0.15, 0.20):
        selected = [row for row in controls if row.speed_mps == speed]
        assert len(selected) == 6
        assert sum(row.control_type == CONTROL_TYPES[0] for row in selected) == 3
        assert sum(row.control_type == CONTROL_TYPES[1] for row in selected) == 3


def test_geometry_selection_is_fail_closed_and_uses_preregistered_tiebreak() -> None:
    assert select_geometry([]) is None
    rows = [
        {"geometry_candidate": "G3", "candidate_pass": True},
        {"geometry_candidate": "G2", "candidate_pass": True},
    ]
    assert select_geometry(rows) == "G2"
    rows.append({"geometry_candidate": "G1", "candidate_pass": True})
    assert select_geometry(rows) == "G1"
    rows.append({"geometry_candidate": "G0", "candidate_pass": True})
    assert select_geometry(rows) == "G0"


def test_moderate_and_strong_profiles_are_frozen() -> None:
    before = frozen_profile_sha256()
    profiles = material_profiles()
    assert profiles["moderate_ice_preregistered"].friction3 == (0.1, 0.00125, 2.125e-05)
    assert profiles["native_strong_ice"].friction3 == (0.05, 0.001, 1e-05)
    assert frozen_profile_sha256() == before


def test_short_calibration_pair_has_exact_schema_solver_friction_and_parity() -> None:
    conditions = calibration_matrix(0.5)[:2]
    results = [collect_run(row, REPO / POLICY) for row in conditions]
    traces = [row[0] for row in results]
    manifests = [row[1] for row in results]
    contacts = [contact for row in results for contact in row[2]]
    assert all(trace["time_s"].shape == (500,) for trace in traces)
    assert all(trace["bilateral_fusion20_raw"].shape == (500, 20) for trace in traces)
    assert all(np.all(np.isfinite(trace["bilateral_fusion20_raw"])) for trace in traces)
    assert manifests[0]["pair_fingerprint"] == manifests[1]["pair_fingerprint"]
    assert manifests[0]["initial_qpos_sha256"] == manifests[1]["initial_qpos_sha256"]
    assert manifests[0]["initial_qvel_sha256"] == manifests[1]["initial_qvel_sha256"]
    assert manifests[0]["initial_policy_observation_sha256"] == manifests[1]["initial_policy_observation_sha256"]
    assert contacts
    assert all(int(row["explicit_pair_id"]) >= 0 for row in contacts)
    assert all(int(row["efc_address_min"]) >= 0 for row in contacts)
    assert all(row["patch_base_double_contact"] is False for row in contacts)
    assert all(row["friction_configured_before_collision"] is True for row in contacts)
    assert all(row["constraints_constructed_at_configuration"] is False for row in contacts)


def test_frozen_oracle_preserves_air_touchdown_and_first_fall_censoring() -> None:
    contact = np.zeros(30, bool)
    contact[1:29] = True
    xyz = np.zeros((30, 3))
    xyz[1:29, 0] = np.linspace(0.0, 0.15, 28)
    signals = derive_contact_signals(
        contact, contact, xyz, np.zeros((30, 3)), np.zeros(30),
        first_fall_sample=20)
    active = persistent_oracle(
        signals.tangential_anchor_drift_m, signals.slip_calibration_valid,
        signals.contact_episode_id, SLIP_THRESHOLD_M, SLIP_PERSISTENCE_MS)
    assert not signals.pre_fall_valid[20:].any()
    assert not np.any(signals.slip_calibration_valid & signals.touchdown_transient)
    assert not np.any(active & ~signals.pre_fall_valid)
    assert not np.any(active & ~contact)


def test_forbidden_guard_fails_closed(tmp_path: Path) -> None:
    guard = AccessGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.path("simulation/outputs/outer/secret.json")
    log = json.loads((tmp_path / "artifact_access_log.json").read_text())
    assert log["blocked_access_count"] == 1
    assert not log["model_score_used_for_diagnosis"]


def test_combined_manifest_excludes_calibration_failed_and_v3() -> None:
    existing = [{
        "run_id": f"existing_{index}", "variation_index": index % 3,
        "speed_mps": (0.1, 0.15, 0.2)[index % 3], "role": "hard_negative",
    } for index in range(3)]
    v5_meta = [
        {"run_id": "v5_good", "pair_id": "p0", "variation_index": "0",
         "speed_mps": "0.1", "target_foot": "left", "target_phase": "early_loading",
         "severity": "native_strong_ice", "role": "positive", "control_type": "hard_normal"},
        {"run_id": "v5_bad", "pair_id": "p1", "variation_index": "1",
         "speed_mps": "0.15", "target_foot": "left", "target_phase": "mid_late_stance",
         "severity": "moderate_ice_preregistered", "role": "positive", "control_type": "hard_normal"},
    ]
    v5_positive = [
        {"run_id": "v5_good", "source_valid": "True", "failure_reason": "",
         "target_foot": "left", "actual_onset_phase": "early_loading",
         "severity": "native_strong_ice", "speed_mps": "0.1"},
        {"run_id": "v5_bad", "source_valid": "False", "failure_reason": "NO_PHYSICAL_ONSET",
         "target_foot": "left", "actual_onset_phase": "none",
         "severity": "moderate_ice_preregistered", "speed_mps": "0.15"},
    ]
    coverage = [{"has_valid_support": index < 35} for index in range(36)]
    result = build_combined_manifest(
        existing, {"left": 1, "right": 1}, v5_meta, v5_positive, [], [], [], [],
        ["calibration_only"], coverage)
    ids = {row["run_id"] for row in result["eligible_runs"]}
    assert "v5_good" in ids
    assert "v5_bad" not in ids
    assert "calibration_only" not in ids
    assert result["audit"]["calibration_run_count_included"] == 0
    assert result["audit"]["failed_source_run_count_included"] == 0
    assert result["audit"]["v3_run_count_included"] == 0
    assert not result["no_onset_hard_negative_eligibility_audit"][0]["training_use_assigned"]


def test_future_fold_manifest_has_no_pair_run_episode_or_variation_leakage() -> None:
    eligible = []
    for variation in range(3):
        for role in ("positive", "control"):
            eligible.append({
                "source_acquisition_version": "targeted_acquisition_v5",
                "run_id": f"run_{variation}_{role}", "pair_id": f"pair_{variation}",
                "variation_index": variation, "speed_mps": (0.1, 0.15, 0.2)[variation],
                "target_foot": ("left", "right", "left")[variation],
                "target_phase": ("early_loading", "mid_loading_early_stance", "mid_late_stance")[variation],
                "severity": ("native_strong_ice", "moderate_ice_preregistered", "native_strong_ice")[variation],
                "role": role, "control_type": CONTROL_TYPES[variation % 2],
                "development_only": True,
            })
    combined = {"eligible_runs": eligible, "audit": {"pass": True}}
    folds = build_future_folds(combined)
    assert folds["audit"]["valid"]
    assert folds["audit"]["group_leakage_count"] == 0
    assert folds["audit"]["calibration_only_run_count"] == 0
    assert folds["audit"]["v3_run_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v6 runner first")
def test_executed_required_artifacts_and_fail_closed_branch() -> None:
    required = {
        "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
        "artifact_access_log.json", "immutable_verification.json",
        "failed_cell_diagnosis.csv", "failure_root_cause.json",
        "geometry_candidate_contract.json", "geometry_calibration_manifest.csv",
        "geometry_calibration_metrics.csv", "selected_geometry_lock.json",
        "supplemental_run_manifest.csv", "supplemental_pair_manifest.csv",
        "supplemental_contact_audit.csv", "supplemental_physical_episode_ledger.csv",
        "supplemental_positive_source_audit.csv", "supplemental_control_source_audit.csv",
        "supplemental_pair_parity.csv", "supplemental_fall_censor_audit.csv",
        "combined_development_manifest.json", "combined_factorial_coverage.csv",
        "future_nested_fold_manifest.json", "duplicate_audit.json",
        "trace_shard_manifest.json", "provenance.json", "readiness.json",
        "summary.json", "audit.md",
    }
    assert required <= {path.name for path in DEFAULT_OUTPUT.iterdir()}
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["original_v5"]["V5_ACQUISITION_DATA_READY"] is False
    assert summary["calibration"]["executed_unique_runs"] == 24
    assert summary["primary_failure_cause"] == "LEFT_RIGHT_GAIT_ASYMMETRY"
    if summary["selected_geometry_candidate"] is None:
        assert summary["supplemental_gates"]["executed_positive_attempts"] == 0
        assert summary["next_step"] == "SLIP_MODERATE_PROFILE_RECALIBRATION"


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v6 runner first")
def test_executed_calibration_is_retained_nontraining_and_pair_valid() -> None:
    with (DEFAULT_OUTPUT / "geometry_calibration_manifest.csv").open(newline="") as stream:
        manifests = list(csv.DictReader(stream))
    with (DEFAULT_OUTPUT / "geometry_calibration_pair_parity.csv").open(newline="") as stream:
        parity = list(csv.DictReader(stream))
    assert len(manifests) == 24
    assert len({row["run_id"] for row in manifests}) == 24
    assert all(row["disposition"] == CALIBRATION_DISPOSITION for row in manifests)
    assert all(row["training_use_assigned"] == "False" for row in manifests)
    assert len(parity) == 12
    assert all(row["parity_pass"] == "True" for row in parity)
    assert all(row["postcontact_trajectory_diverged"] == "True" for row in parity)
    assert all(row["patch_base_double_contact_count"] == "0" for row in parity)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v6 runner first")
def test_executed_hash_graph_shards_immutability_and_size() -> None:
    shards = json.loads((DEFAULT_OUTPUT / "trace_shard_manifest.json").read_text())
    assert shards["all_under_45_mib"]
    assert shards["all_exact_roundtrip_verified"]
    assert sum(row["run_count"] for row in shards["shards"]) >= 24
    for row in shards["shards"]:
        assert sha256_file(DEFAULT_OUTPUT / row["path"]) == row["sha256"]
    provenance = json.loads((DEFAULT_OUTPUT / "provenance.json").read_text())
    assert provenance["artifact_hash_graph_verified"]
    for name, digest in provenance["artifact_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / name) == digest
    immutable = json.loads((DEFAULT_OUTPUT / "immutable_verification.json").read_text())
    assert immutable["all_immutable_after_execution"]
    assert immutable["v5_artifacts_unchanged"]
    assert immutable["strong_moderate_profiles_unchanged"]
    assert immutable["physical_slip_oracle_byte_identical"]
    assert immutable["terrain_byte_identical"]
    sizes = json.loads((DEFAULT_OUTPUT / "resource_size_audit.json").read_text())
    assert sizes["all_files_within_limit"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v6 runner first")
def test_executed_diagnosis_uses_no_model_scores_and_access_is_clean() -> None:
    root = json.loads((DEFAULT_OUTPUT / "failure_root_cause.json").read_text())
    assert root["exactly_one_primary_failure_cause"]
    assert not root["model_scores_used"]
    with (DEFAULT_OUTPUT / "failed_cell_diagnosis.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 45
    assert all(row["model_score_used"] == "False" for row in rows)
    access = json.loads((DEFAULT_OUTPUT / "artifact_access_log.json").read_text())
    assert access["blocked_access_count"] == 0
    assert not access["model_score_used_for_diagnosis"]
