"""Focused contracts and executed-artifact tests for moderate Slip v7."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from run_walking_v2_slip_moderate_profile_recalibration_v7 import (
    AccessGuard, DEFAULT_OUTPUT, POLICY, REPO, collect_run, future_folds,
    moderate_v1_disposition, sha256_file,
)
from walking_hazard_ground_truth_v1 import derive_contact_signals
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_slip_moderate_profile_recalibration_v7 import (
    CALIBRATION_ARMS, CALIBRATION_DISPOSITION, CANDIDATE_IDS,
    SEVERITY_ORDER, SEVERITY_ORDER_TOLERANCE_M, candidate_profile, friction_grid_sha256,
    friction_profiles, moderate_v2_matrix, profile_calibration_matrix,
    select_candidate,
)


def test_friction_grid_is_exact_sliding_only_and_distinct_from_strong() -> None:
    profiles = friction_profiles()
    strong = profiles["native_strong_ice"]
    candidates = [candidate_profile(value) for value in CANDIDATE_IDS]
    assert [row.friction3[0] for row in candidates] == pytest.approx(
        [0.1, 0.0875, 0.075, 0.0625])
    assert all(row.friction3[0] > strong.friction3[0] for row in candidates)
    assert all(row.friction3[1:] == candidates[0].friction3[1:] for row in candidates)
    assert candidates[0].friction3 == (0.1, 0.00125, 2.125e-05)
    assert strong.friction3 == (0.05, 0.001, 1e-05)
    assert len({row.friction5 for row in candidates}) == 4
    assert len(friction_grid_sha256()) == 64


def test_profile_calibration_matrix_is_exact_three_matched_six_arm_groups() -> None:
    rows = profile_calibration_matrix()
    assert len(rows) == 18
    assert len({row.run_id for row in rows}) == 18
    assert len({row.pair_id for row in rows}) == 3
    assert all(row.disposition == CALIBRATION_DISPOSITION for row in rows)
    for group_id in {row.pair_id for row in rows}:
        group = [row for row in rows if row.pair_id == group_id]
        assert {row.calibration_arm for row in group} == set(CALIBRATION_ARMS)
        assert len({row.pair_fingerprint for row in group}) == 1
        assert len({row.material_profile for row in group}) == 6
        assert len({row.seed for row in group}) == 1


def test_moderate_v2_matrix_is_complete_matched_and_control_balanced() -> None:
    rows = moderate_v2_matrix("M2")
    assert len(rows) == 108
    assert len({row.run_id for row in rows}) == 108
    assert len({row.pair_id for row in rows}) == 54
    positives = [row for row in rows if row.role == "positive"]
    controls = [row for row in rows if row.role == "control"]
    assert len(positives) == len(controls) == 54
    assert len({(row.speed_mps, row.target_foot, row.target_phase) for row in positives}) == 18
    assert all(row.material_profile == "moderate_v2_M2" for row in positives)
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.pair_fingerprint for row in pair}) == 1
    for speed in (0.10, 0.15, 0.20):
        selected = [row for row in controls if row.speed_mps == speed]
        counts = {name: sum(row.control_type == name for row in selected)
                  for name in ("hard_normal", "near_slip_non_event")}
        assert abs(counts["hard_normal"] - counts["near_slip_non_event"]) <= 1


def test_profile_selection_uses_highest_passing_friction_and_rejects_m0() -> None:
    assert select_candidate([]) is None
    rows = [
        {"candidate_id": "M3", "candidate_pass": True},
        {"candidate_id": "M2", "candidate_pass": True},
        {"candidate_id": "M0", "candidate_pass": True},
    ]
    assert select_candidate(rows) == "M2"
    rows.append({"candidate_id": "M1", "candidate_pass": True})
    assert select_candidate(rows) == "M1"


def test_severity_order_tolerance_is_preregistered() -> None:
    assert SEVERITY_ORDER_TOLERANCE_M == 0.005
    assert SEVERITY_ORDER == (
        "hard_control", "M0", "M1", "M2", "M3", "strong_reference")


def test_short_multiarm_runs_share_precontact_state_and_use_solver_friction() -> None:
    rows = profile_calibration_matrix(0.5)
    conditions = [rows[0], rows[4]]  # hard and M2 at 0.10 m/s
    results = [collect_run(row, REPO / POLICY) for row in conditions]
    hard_trace, candidate_trace = results[0][0], results[1][0]
    hard_meta, candidate_meta = results[0][1], results[1][1]
    assert hard_meta["pair_fingerprint"] == candidate_meta["pair_fingerprint"]
    assert hard_meta["initial_qpos_sha256"] == candidate_meta["initial_qpos_sha256"]
    assert hard_meta["initial_qvel_sha256"] == candidate_meta["initial_qvel_sha256"]
    assert hard_trace["bilateral_fusion20_raw"].shape == (500, 20)
    assert candidate_trace["bilateral_fusion20_raw"].shape == (500, 20)
    mu = candidate_trace["effective_patch_friction"][:, 0]
    assert np.any(np.isclose(mu[np.isfinite(mu)], 0.075, atol=1e-12, rtol=0))
    contacts = results[1][2]
    assert contacts
    assert all(int(row["explicit_pair_id"]) >= 0 for row in contacts)
    assert all(int(row["efc_address_min"]) >= 0 for row in contacts)
    assert all(row["patch_base_double_contact"] is False for row in contacts)
    assert all(row["friction_configured_before_collision"] is True for row in contacts)


def test_frozen_oracle_excludes_air_touchdown_transient_and_postfall() -> None:
    contact = np.zeros(30, bool); contact[1:29] = True
    xyz = np.zeros((30, 3)); xyz[1:29, 0] = np.linspace(0.0, 0.15, 28)
    signals = derive_contact_signals(
        contact, contact, xyz, np.zeros((30, 3)), np.zeros(30),
        first_fall_sample=20)
    active = persistent_oracle(
        signals.tangential_anchor_drift_m, signals.slip_calibration_valid,
        signals.contact_episode_id, 0.050, 3)
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
    assert not log["model_scores_used_for_profile_selection"]


def test_moderate_v1_disposition_preserves_version_and_assigns_no_training_use() -> None:
    rows = [{
        "run_id": "valid", "severity": "moderate_ice_preregistered",
        "source_valid": "True", "failure_reason": ""}, {
        "run_id": "no_onset", "severity": "moderate_ice_preregistered",
        "source_valid": "False", "failure_reason": "NO_PHYSICAL_ONSET"}]
    result = moderate_v1_disposition(rows)
    assert result["kept_separately_versioned"]
    assert result["disposition"] == "SUPERSEDED_PROFILE_DEVELOPMENT_DIAGNOSTIC"
    assert result["physically_valid_run_count"] == 1
    assert result["no_onset_run_count"] == 1
    assert not result["possible_future_hard_negative_eligibility"][0]["training_use_assigned"]


def test_future_fold_groups_have_no_pair_run_episode_or_variation_leakage() -> None:
    runs = []
    for variation in range(3):
        for severity, source in (("native_strong_ice", "v5_strong"),
                                 ("moderate_v2", "v7_moderate_v2")):
            for role in ("positive", "control"):
                runs.append({
                    "source_acquisition": source,
                    "run_id": f"{source}_{variation}_{role}",
                    "pair_id": f"{source}:pair_{variation}", "role": role,
                    "speed_mps": (0.1, 0.15, 0.2)[variation],
                    "target_foot": ("left", "right", "left")[variation],
                    "target_phase": ("early_loading", "mid_loading_early_stance", "mid_late_stance")[variation],
                    "severity": severity, "control_type": ("hard_normal", "near_slip_non_event")[variation % 2],
                    "variation_index": variation, "future_training_eligible": True,
                    "development_only": True})
    canonical = {"runs": runs, "audit": {"pass": True}}
    existing = [{
        "run_id": f"existing_{index}", "variation_index": index,
        "speed_mps": (0.1, 0.15, 0.2)[index], "role": "hard_negative"}
        for index in range(3)]
    folds = future_folds(canonical, existing)
    assert folds["audit"]["leakage_count"] == 0
    assert folds["audit"]["positive_control_pairs_same_fold"]
    assert folds["audit"]["v3_run_count"] == 0
    assert folds["audit"]["v7_profile_calibration_run_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v7 runner first")
def test_executed_artifacts_selection_lock_and_readiness() -> None:
    required = {
        "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
        "artifact_access_log.json", "immutable_verification.json",
        "friction_candidate_contract.json", "profile_calibration_manifest.csv",
        "profile_calibration_metrics.csv", "profile_severity_ordering.csv",
        "moderate_v1_disposition.json", "duplicate_audit.json",
        "trace_shard_manifest.json", "provenance.json", "readiness.json",
        "summary.json", "audit.md"}
    names = {path.name for path in DEFAULT_OUTPUT.iterdir()}
    assert required <= names
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["profile_calibration"]["executed_unique_runs"] == 18
    assert summary["original_v5_readiness"] is False
    assert summary["moderate_v1_separate_version"]
    if summary["selected_candidate"] is not None:
        conditional = {
            "moderate_v2_profile_lock.json", "moderate_v2_run_manifest.csv",
            "moderate_v2_pair_manifest.csv", "moderate_v2_contact_audit.csv",
            "moderate_v2_physical_episode_ledger.csv",
            "moderate_v2_positive_source_audit.csv",
            "moderate_v2_control_source_audit.csv", "moderate_v2_pair_parity.csv",
            "moderate_v2_fall_censor_audit.csv"}
        assert conditional <= names
        lock = json.loads((DEFAULT_OUTPUT / "moderate_v2_profile_lock.json").read_text())
        assert lock["candidate_id"] == summary["selected_candidate"]
        assert lock["locked_before_reacquisition"]
        assert not lock["changed_after_lock"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v7 runner first")
def test_executed_reacquisition_pair_physics_and_canonical_manifest() -> None:
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    if summary["selected_candidate"] is None:
        pytest.skip("no candidate passed; conditional reacquisition correctly absent")
    with (DEFAULT_OUTPUT / "moderate_v2_run_manifest.csv").open(newline="") as stream:
        manifests = list(csv.DictReader(stream))
    with (DEFAULT_OUTPUT / "moderate_v2_pair_parity.csv").open(newline="") as stream:
        parity = list(csv.DictReader(stream))
    assert len(manifests) == 108
    assert all(row["sample_count"] == "3000" for row in manifests)
    assert all(row["fusion20_shape"] == "[3000, 20]" for row in manifests)
    assert len(parity) == 54
    assert all(row["parity_pass"] == "True" for row in parity)
    assert all(row["postcontact_trajectory_diverged"] == "True" for row in parity)
    assert all(row["full_trace_identical"] == "False" for row in parity)
    assert all(row["patch_base_double_contact_count"] == "0" for row in parity)
    if not summary["moderate_v2_acquisition_gates"]["pass"]:
        assert not (DEFAULT_OUTPUT / "canonical_development_manifest.json").exists()
        assert not (DEFAULT_OUTPUT / "future_nested_fold_manifest.json").exists()
        return
    canonical = json.loads((DEFAULT_OUTPUT / "canonical_development_manifest.json").read_text())
    assert canonical["audit"]["positive_attempts"] == 108
    assert canonical["audit"]["control_attempts"] == 108
    assert canonical["audit"]["total_runs"] == 216
    assert canonical["audit"]["strong_source_valid_count"] == 48
    assert canonical["audit"]["v5_moderate_v1_run_count"] == 0
    assert canonical["audit"]["v7_profile_calibration_run_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v7 runner first")
def test_executed_shards_hash_graph_immutability_and_size() -> None:
    shards = json.loads((DEFAULT_OUTPUT / "trace_shard_manifest.json").read_text())
    assert shards["all_under_45_mib"]
    assert shards["all_exact_roundtrip_verified"]
    assert sum(row["run_count"] for row in shards["shards"]) >= 18
    for row in shards["shards"]:
        assert sha256_file(DEFAULT_OUTPUT / row["path"]) == row["sha256"]
    provenance = json.loads((DEFAULT_OUTPUT / "provenance.json").read_text())
    assert provenance["artifact_hash_graph_verified"]
    for name, digest in provenance["artifact_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / name) == digest
    immutable = json.loads((DEFAULT_OUTPUT / "immutable_verification.json").read_text())
    assert immutable["all_immutable_after_execution"]
    assert immutable["terrain_byte_identical"]
    assert immutable["physical_oracle_byte_identical"]
    assert immutable["strong_native_ice_unchanged"]
    assert immutable["g0_geometry_unchanged"]
    sizes = json.loads((DEFAULT_OUTPUT / "resource_size_audit.json").read_text())
    assert sizes["all_files_within_limit"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v7 runner first")
def test_executed_access_profile_version_and_no_silent_mutation() -> None:
    access = json.loads((DEFAULT_OUTPUT / "artifact_access_log.json").read_text())
    assert access["blocked_access_count"] == 0
    assert not access["model_scores_used_for_profile_selection"]
    disposition = json.loads((DEFAULT_OUTPUT / "moderate_v1_disposition.json").read_text())
    assert disposition["kept_separately_versioned"]
    assert disposition["attempt_count"] == 54
    duplicate = json.loads((DEFAULT_OUTPUT / "duplicate_audit.json").read_text())
    assert duplicate["discarded_count"] == 0
    assert duplicate["replaced_count"] == 0
    assert duplicate["silently_relabelled_count"] == 0
