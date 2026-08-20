"""Focused physics and artifact regressions for Slip generator v4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from run_walking_v2_slip_scenario_generator_redesign_v4 import (
    AccessGuard, DEFAULT_OUTPUT, REPO, _microbench_model, _run_microbench,
    build_local_model, sha256_file,
)
from walking_hazard_ground_truth_v1 import derive_contact_signals
from walking_v2_slip_scenario_generator_redesign_v4 import (
    PHYSICS_TIMESTEP_S, friction_profiles, patch_bounds, pilot_matrix,
)


def test_bounded_matrix_has_24_matched_pairs_and_48_unique_runs() -> None:
    rows = pilot_matrix()
    assert len(rows) == 48
    assert len({row.run_id for row in rows}) == 48
    assert len({row.pair_id for row in rows}) == 24
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.parity_sha256 for row in pair}) == 1


def test_friction_profiles_have_expected_strict_order() -> None:
    profiles = friction_profiles()
    assert (
        profiles["hard_control"].friction3[0]
        > profiles["moderate_ice_preregistered"].friction3[0]
        > profiles["native_strong_ice"].friction3[0]
    )


def test_equal_priority_without_pair_uses_maximum_friction() -> None:
    model = _microbench_model("native_strong_ice", explicit_pair=False)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
        if data.ncon:
            break
    assert data.ncon > 0
    assert data.contact[0].friction[0] == pytest.approx(1.0)


def test_precompiled_pair_is_solver_effective_and_monotonic() -> None:
    results = {
        name: _run_microbench(name)
        for name in ("hard_control", "moderate_ice_preregistered", "native_strong_ice")
    }
    mu = [results[name][1]["effective_friction_mean"] for name in results]
    displacement = [results[name][1]["displacement_m"] for name in results]
    assert mu[0] > mu[1] > mu[2]
    assert displacement[0] < displacement[1] < displacement[2]


def test_post_step1_mutation_does_not_change_integrated_trajectory() -> None:
    hard, _ = _run_microbench("hard_control")
    late, _ = _run_microbench("native_strong_ice", post_step1_mutation=True)
    assert not np.array_equal(
        hard["effective_friction"], late["effective_friction"], equal_nan=True)
    assert np.array_equal(hard["x_m"], late["x_m"])
    assert np.array_equal(hard["vx_mps"], late["vx_mps"])


def test_local_patch_contract_covers_all_sole_geoms_without_overlap() -> None:
    condition = pilot_matrix(0.01)[0]
    model, info = build_local_model(condition)
    assert model.npair == 8
    assert len(info["pair_ids"]["left"]) == 4
    assert len(info["pair_ids"]["right"]) == 4
    assert info["top_height_delta_m"] == 0.0
    bounds = patch_bounds("left")
    assert bounds.top_z_m == 0.0
    assert model.geom("slip_patch").contype[0] == 0
    assert model.geom("slip_patch").conaffinity[0] == 0
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.opt.timestep == pytest.approx(PHYSICS_TIMESTEP_S)


def test_first_fall_censor_and_air_semantics_are_frozen() -> None:
    contact = np.asarray([False, True, True, True, True, False])
    loaded = np.asarray([False, True, True, True, True, False])
    xyz = np.zeros((6, 3)); xyz[1:5, 0] = np.arange(4) * 0.02
    velocity = np.zeros((6, 3)); penetration = np.zeros(6)
    signals = derive_contact_signals(
        contact, loaded, xyz, velocity, penetration, first_fall_sample=3,
        touchdown_transient_samples=0)
    assert not np.any(signals.slip_calibration_valid[[0, 3, 4, 5]])
    assert np.all(signals.slip_calibration_valid[[1, 2]])


def test_forbidden_path_guard_fails_closed(tmp_path: Path) -> None:
    guard = AccessGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.path("simulation/outputs/outer/secret.json")
    record = json.loads((tmp_path / "artifact_access_log.json").read_text())
    assert record["blocked_access_count"] == 1


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v4 runner first")
def test_executed_artifact_graph_and_physics_gates() -> None:
    required = {
        "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
        "artifact_access_log.json", "failed_v3_quarantine_manifest.json",
        "terrain_immutable_verification.json", "oracle_immutable_verification.json",
        "friction_intervention_call_graph.json", "friction_root_cause.json",
        "geom_contact_contract.csv", "step_order_audit.json",
        "microbench_manifest.csv", "microbench_metrics.csv", "microbench_traces.npz",
        "whole_surface_reference_metrics.csv", "local_patch_definition.json",
        "local_patch_contact_audit.csv", "pilot_run_manifest.csv",
        "pilot_pair_parity.csv", "pilot_physical_episode_ledger.csv",
        "pilot_contact_force_metrics.csv", "pilot_fall_censor_audit.csv",
        "duplicate_audit.json", "resource_size_audit.json", "provenance.json",
        "readiness.json", "summary.json", "audit.md",
    }
    assert required <= {path.name for path in DEFAULT_OUTPUT.iterdir()}
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["pilot_gates"]["patch_base_double_contact_count"] == 0
    assert summary["pilot_gates"]["precontact_parity"]
    assert summary["pilot_gates"]["postcontact_identity_count"] == 0
    assert summary["pilot_gates"]["control_valid_physical_slip_onset_count"] == 0
    assert summary["readiness"]["WALKING_V2_SLIP_SCENARIO_GENERATOR_READY"]
    provenance = json.loads((DEFAULT_OUTPUT / "provenance.json").read_text())
    assert provenance["artifact_hash_graph_verified"]
    for name, expected in provenance["artifact_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / name) == expected
    resources = json.loads((DEFAULT_OUTPUT / "resource_size_audit.json").read_text())
    assert resources["all_files_within_limit"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="execute v4 runner first")
def test_executed_pair_and_contact_audits() -> None:
    with (DEFAULT_OUTPUT / "pilot_pair_parity.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 24
    assert all(row["configuration_parity"] == "True" for row in pairs)
    assert all(row["collision_geometry_identical"] == "True" for row in pairs)
    assert all(row["precontact_trace_parity"] == "True" for row in pairs)
    assert all(row["positive_control_full_trace_identical"] == "False" for row in pairs)
    with (DEFAULT_OUTPUT / "local_patch_contact_audit.csv").open(newline="") as stream:
        contacts = [row for row in csv.DictReader(stream) if row["patch_inclusion"] == "True"]
    assert contacts
    assert all(int(row["explicit_pair_id"]) >= 0 for row in contacts)
    assert all(int(row["efc_address_min"]) >= 0 for row in contacts)


def test_locked_terrain_files_still_match_executed_hashes() -> None:
    verification_path = DEFAULT_OUTPUT / "terrain_immutable_verification.json"
    if not verification_path.is_file():
        pytest.skip("execute v4 runner first")
    verification = json.loads(verification_path.read_text())
    assert verification["byte_identical"]
    for key, relative in {
        "model": "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_model.npz",
        "normalization": "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_normalization.json",
        "config": "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_candidate_config.json",
        "lock": "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1/terrain_selection_lock.json",
    }.items():
        assert sha256_file(REPO / relative) == verification["after_sha256"][key]
