"""Focused contracts and executed-artifact tests for Slip acquisition v5."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from run_walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    AccessGuard, DEFAULT_OUTPUT, POLICY, REPO, _patch_contract_snapshot,
    build_future_fold_manifest, build_pair_audits, collect_run,
    save_trace_shards, sha256_file,
)
from walking_hazard_ground_truth_v1 import derive_contact_signals
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHASE_BINS, SEVERITIES, SIDES, SPEEDS_MPS,
    acquisition_matrix, material_profiles,
)


def test_matrix_is_exact_balanced_and_pair_matched() -> None:
    rows = acquisition_matrix()
    assert len(rows) == 216
    assert len({row.run_id for row in rows}) == 216
    assert len({row.pair_id for row in rows}) == 108
    positives = [row for row in rows if row.role == "positive"]
    controls = [row for row in rows if row.role == "control"]
    assert len(positives) == len(controls) == 108
    cells = {
        (row.speed_mps, row.target_foot, row.target_phase, row.severity)
        for row in positives}
    assert len(cells) == 36
    assert all(sum(
        (row.speed_mps, row.target_foot, row.target_phase, row.severity) == cell
        for row in positives) == 3 for cell in cells)
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.pair_fingerprint for row in pair}) == 1
    assert {row.control_type for row in controls} == set(CONTROL_TYPES)


def test_profiles_are_frozen_and_strictly_ordered() -> None:
    profiles = material_profiles()
    assert set(profiles) == set(SEVERITIES) | set(CONTROL_TYPES)
    assert profiles["hard_normal"].friction3 == (1.0, 0.005, 0.0001)
    assert profiles["near_slip_non_event"].friction3 == (
        0.35000000000000003, 0.0025, 7.75e-05)
    assert profiles["moderate_ice_preregistered"].friction3[0] == 0.1
    assert profiles["native_strong_ice"].friction3[0] == 0.05
    assert len({profile.sha256 for profile in profiles.values()}) == 4


def test_v4_patch_contract_has_all_eight_explicit_pairs() -> None:
    snapshot = _patch_contract_snapshot()
    assert snapshot["explicit_pair_count"] == 8
    assert len(snapshot["pair_payload"]["pair_geom1"]) == 8
    assert len(snapshot["pair_payload"]["pair_geom2"]) == 8
    assert snapshot["geometry_payload"]["top_height_delta_m"] == 0.0


def test_short_pair_has_precontact_parity_and_postcontact_divergence() -> None:
    conditions = acquisition_matrix(0.5)[:2]
    traces, manifests = [], []
    for condition in conditions:
        trace, meta, contacts = collect_run(condition, REPO / POLICY)
        assert meta["sample_count"] == 500
        assert meta["patch_base_double_contact_count"] == 0
        assert contacts and all(int(row["efc_address_min"]) >= 0 for row in contacts)
        traces.append(trace); manifests.append(meta)
    _, parity = build_pair_audits(conditions, traces, manifests)
    assert parity[0]["parity_pass"]
    assert parity[0]["precontact_fusion20_equal"]
    assert parity[0]["postcontact_trajectory_diverged"]
    assert not parity[0]["full_trace_identical"]


def test_near_slip_pair_friction_is_solver_effective_before_constraints() -> None:
    condition = next(
        row for row in acquisition_matrix(0.6)
        if row.role == "control"
        and row.material_profile == "near_slip_non_event"
        and row.target_phase == "mid_loading_early_stance")
    trace, meta, contacts = collect_run(condition, REPO / POLICY)
    side = SIDES.index(condition.target_foot)
    friction = trace["effective_patch_friction"][:, side]
    assert np.nanmin(friction) == pytest.approx(0.35)
    assert np.nanmax(friction) == pytest.approx(1.0)
    assert meta["patch_base_double_contact_count"] == 0
    assert all(row["friction_configured_before_collision"] for row in contacts)
    assert all(not row["constraints_constructed_at_configuration"] for row in contacts)


def test_frozen_oracle_excludes_air_touchdown_transient_and_postfall() -> None:
    sample_count = 20
    contact = np.zeros(sample_count, bool); contact[1:19] = True
    loaded = contact.copy()
    xyz = np.zeros((sample_count, 3))
    xyz[1:19, 0] = np.linspace(0.0, 0.12, 18)
    velocity = np.zeros((sample_count, 3))
    penetration = np.zeros(sample_count)
    signals = derive_contact_signals(
        contact, loaded, xyz, velocity, penetration, first_fall_sample=15)
    active = persistent_oracle(
        signals.tangential_anchor_drift_m,
        signals.slip_calibration_valid, signals.contact_episode_id,
        threshold_m=0.050, persistence_ms=3)
    assert not signals.pre_fall_valid[15:].any()
    assert not signals.slip_calibration_valid[0]
    assert not np.any(signals.slip_calibration_valid & signals.touchdown_transient)
    assert not np.any(active & ~signals.pre_fall_valid)
    assert not np.any(active & ~contact)


def test_forbidden_guard_fails_closed(tmp_path: Path) -> None:
    guard = AccessGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.path("simulation/outputs/outer/secret.json")
    record = json.loads((tmp_path / "artifact_access_log.json").read_text())
    assert record["blocked_access_count"] == 1


def test_short_trace_shard_roundtrip(tmp_path: Path) -> None:
    conditions = acquisition_matrix(0.02)[:2]
    traces = [collect_run(row, REPO / POLICY)[0] for row in conditions]
    manifest = save_trace_shards(tmp_path, conditions, traces, shard_size=2)
    assert manifest["shard_count"] == 1
    assert manifest["all_exact_roundtrip_verified"]
    assert manifest["all_under_45_mib"]


def test_future_fold_grouping_has_no_pair_run_episode_or_variation_leakage() -> None:
    conditions = acquisition_matrix(0.01)
    existing = {"runs": [{
        "run_id": f"existing_{index:03d}",
        "variation_index": index % 6,
        "speed_mps": SPEEDS_MPS[index % 3],
        "role": "hard_negative",
    } for index in range(120)]}
    manifests = [{**vars(row)} for row in conditions]
    traces = [{
        "contact_episode_id": np.asarray(((0, 0), (0, 0)), dtype=np.int32)
    } for _ in conditions]
    positives = [{
        "run_id": row.run_id, "source_valid": True
    } for row in conditions if row.role == "positive"]
    controls = [{
        "run_id": row.run_id, "control_source_valid": True
    } for row in conditions if row.role == "control"]
    result = build_future_fold_manifest(
        existing, manifests, traces, positives, controls)
    assert result["audit"]["valid"]
    assert result["audit"]["group_leakage_count"] == 0
    assert result["audit"]["run_leakage_count"] == 0
    assert result["audit"]["episode_leakage_count"] == 0
    assert result["audit"]["variation_leakage_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v5 runner first")
def test_executed_artifacts_and_readiness() -> None:
    required = {
        "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
        "artifact_access_log.json", "generator_immutable_verification.json",
        "patch_contract_verification.json", "oracle_immutable_verification.json",
        "terrain_immutable_verification.json", "v3_quarantine_verification.json",
        "provenance_precheck.json", "acquisition_manifest.json",
        "material_profiles.json", "patch_geometry.json",
        "counterfactual_pair_manifest.csv", "run_manifest.csv",
        "active_patch_contact_audit.csv", "physical_episode_ledger.csv",
        "positive_source_audit.csv", "control_source_audit.csv",
        "onset_phase_distribution.csv", "affected_foot_distribution.csv",
        "speed_distribution.csv", "severity_distribution.csv",
        "fall_censor_audit.csv", "pair_parity_audit.csv",
        "duplicate_audit.json", "causal_feature_pair_statistics.csv",
        "future_nested_fold_manifest.json", "trace_shard_manifest.json",
        "provenance.json", "readiness.json", "summary.json", "audit.md",
    }
    assert required <= {path.name for path in DEFAULT_OUTPUT.iterdir()}
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["executed_unique_runs"] == 216
    assert summary["gates"]["schema_and_provenance"]["exact_1khz_timestamps"]
    assert summary["gates"]["schema_and_provenance"]["exact_fusion20_schema"]
    assert summary["gates"]["counterfactual_physics"]["patch_base_double_contact_count"] == 0
    assert summary["gates"]["counterfactual_physics"]["full_trace_identity_count"] == 0
    # Authorization follows the mandatory gates; this executed acquisition is
    # intentionally retained even when physical-source coverage fails.
    assert not summary["gates"]["positive_source_pass"]
    assert not summary["readiness"]["WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v5 runner first")
def test_executed_contact_pair_and_duplicate_audits() -> None:
    with (DEFAULT_OUTPUT / "pair_parity_audit.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 108
    assert all(row["parity_pass"] == "True" for row in pairs)
    assert all(row["postcontact_trajectory_diverged"] == "True" for row in pairs)
    with (DEFAULT_OUTPUT / "active_patch_contact_audit.csv").open(newline="") as stream:
        contacts = list(csv.DictReader(stream))
    assert contacts
    assert all(int(row["explicit_pair_id"]) >= 0 for row in contacts)
    assert all(int(row["efc_address_min"]) >= 0 for row in contacts)
    assert all(row["patch_base_double_contact"] == "False" for row in contacts)
    duplicate = json.loads((DEFAULT_OUTPUT / "duplicate_audit.json").read_text())
    assert duplicate["duplicate_count"] == 0


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v5 runner first")
def test_shards_hash_graph_folds_and_immutability() -> None:
    shards = json.loads((DEFAULT_OUTPUT / "trace_shard_manifest.json").read_text())
    assert shards["all_under_45_mib"]
    assert shards["all_exact_roundtrip_verified"]
    assert sum(row["run_count"] for row in shards["shards"]) == 216
    for row in shards["shards"]:
        assert sha256_file(DEFAULT_OUTPUT / row["path"]) == row["sha256"]
    folds = json.loads((DEFAULT_OUTPUT / "future_nested_fold_manifest.json").read_text())
    assert folds["not_created"]
    assert not folds["audit"]["valid"]
    assert folds["rows"] == []
    provenance = json.loads((DEFAULT_OUTPUT / "provenance.json").read_text())
    assert provenance["artifact_hash_graph_verified"]
    for name, expected in provenance["artifact_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / name) == expected
    terrain = json.loads((DEFAULT_OUTPUT / "terrain_immutable_verification.json").read_text())
    oracle = json.loads((DEFAULT_OUTPUT / "oracle_immutable_verification.json").read_text())
    generator = json.loads((DEFAULT_OUTPUT / "generator_immutable_verification.json").read_text())
    assert terrain["byte_identical"]
    assert oracle["unchanged_after_acquisition"]
    assert generator["unchanged_after_acquisition"]


def test_declared_factorial_axes_are_complete() -> None:
    assert SPEEDS_MPS == (0.10, 0.15, 0.20)
    assert SIDES == ("left", "right")
    assert len(PHASE_BINS) == 3
    assert len(SEVERITIES) == 2
