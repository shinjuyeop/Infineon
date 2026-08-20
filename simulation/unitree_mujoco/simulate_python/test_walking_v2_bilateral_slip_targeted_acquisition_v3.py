"""Focused tests for targeted bilateral Slip acquisition v3."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_walking_v2_bilateral_slip_targeted_acquisition_v3 import (  # noqa: E402
    ArtifactAccessGuard,
)
from terrain_profiles import TERRAIN_PROFILES  # noqa: E402
from walking_hazard_ground_truth_v1 import derive_contact_signals  # noqa: E402
from walking_hazard_oracle_calibration_v1 import persistent_oracle  # noqa: E402
from walking_v2_bilateral_slip_targeted_acquisition_v3 import (  # noqa: E402
    CONTROL_TYPES,
    PHASE_BINS,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ,
    SEVERITIES,
    SLIP_PERSISTENCE_MS,
    SLIP_THRESHOLD_M,
    SPEEDS_MPS,
    acquisition_matrix,
    deterministic_initial_perturbation,
    friction_vector,
    full_trace_sha256,
    material_profiles,
    onset_phase,
    patch_for_foot,
    validate_future_fold_rows,
)


def test_fixed_matrix_exact_counts_and_unique_ids():
    rows = acquisition_matrix()
    assert len(rows) == 216
    assert len({row.run_id for row in rows}) == 216
    assert len({row.pair_id for row in rows}) == 108
    assert sum(row.role == "positive" for row in rows) == 108
    assert sum(row.role == "control" for row in rows) == 108


def test_each_pair_differs_only_by_material_role():
    rows = acquisition_matrix()
    for pair_id in {row.pair_id for row in rows}:
        pair = [row for row in rows if row.pair_id == pair_id]
        assert {row.role for row in pair} == {"positive", "control"}
        assert len({row.pair_fingerprint for row in pair}) == 1
        assert len({row.material_profile for row in pair}) == 2
        assert len({row.seed for row in pair}) == 1


def test_positive_cells_and_control_balance_are_frozen():
    positives = [row for row in acquisition_matrix() if row.role == "positive"]
    cells = {
        (row.speed_mps, row.target_foot, row.target_phase, row.severity)
        for row in positives
    }
    assert len(cells) == 3 * 2 * 3 * 2
    assert all(sum(
        (row.speed_mps, row.target_foot, row.target_phase, row.severity) == cell
        for row in positives
    ) == 3 for cell in cells)
    for speed in SPEEDS_MPS:
        for foot in ("left", "right"):
            for phase in PHASE_BINS:
                for variation in range(3):
                    assert {
                        row.control_type for row in positives
                        if row.speed_mps == speed and row.target_foot == foot
                        and row.target_phase == phase.name and row.variation_index == variation
                    } == set(CONTROL_TYPES)


def test_two_positive_materials_and_preregistered_moderate_derivation():
    profiles = material_profiles()
    assert set(SEVERITIES) <= set(profiles)
    ice = np.asarray(TERRAIN_PROFILES["ice"].friction)
    marble = np.asarray(TERRAIN_PROFILES["marble"].friction)
    expected = ice + 0.125 * (marble - ice)
    np.testing.assert_allclose(profiles["moderate_ice_preregistered"].friction, expected)
    assert profiles["moderate_ice_preregistered"].friction[0] > profiles["native_strong_ice"].friction[0]
    assert profiles["near_slip_non_event"].friction[0] > profiles["moderate_ice_preregistered"].friction[0]
    assert friction_vector(profiles["native_strong_ice"]).shape == (5,)


def test_height_preserving_target_foot_patch_geometry():
    left = patch_for_foot("left"); right = patch_for_foot("right")
    assert left.height_delta_m == right.height_delta_m == 0.0
    assert left.contains(np.asarray((0.5, 0.1, 0.0)))
    assert not left.contains(np.asarray((0.5, -0.1, 0.0)))
    assert right.contains(np.asarray((0.5, -0.1, 0.0)))
    assert not right.contains(np.asarray((0.5, 0.1, 0.0)))
    with pytest.raises(ValueError):
        patch_for_foot("pelvis")


def test_touchdown_relative_phase_boundaries_are_frozen_and_disjoint():
    assert [(row.minimum_onset_ms, row.maximum_onset_ms) for row in PHASE_BINS] == [
        (11, 120), (121, 260), (261, 600),
    ]
    for phase in PHASE_BINS:
        assert onset_phase(phase.minimum_onset_ms) == phase.name
        assert onset_phase(phase.maximum_onset_ms) == phase.name
    assert onset_phase(10) == "out_of_contract"
    assert onset_phase(601) == "out_of_contract"


def test_exact_one_khz_sampling_contract():
    assert SAMPLE_RATE_HZ == 1000
    assert PHYSICS_TIMESTEP_S * PHYSICS_STEPS_PER_SAMPLE == pytest.approx(0.001, abs=1e-15)


def test_pair_shared_seed_perturbation_is_deterministic_and_distinct():
    assert deterministic_initial_perturbation(202608300) == deterministic_initial_perturbation(202608300)
    assert deterministic_initial_perturbation(202608300) != deterministic_initial_perturbation(202608301)


def test_full_trace_hash_frames_key_dtype_shape_and_values():
    first = {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}
    second = {"a": first["a"].copy()}
    assert full_trace_sha256(first, ("a",)) == full_trace_sha256(second, ("a",))
    second["a"][1, 2] += 1
    assert full_trace_sha256(first, ("a",)) != full_trace_sha256(second, ("a",))


def test_frozen_oracle_respects_contact_and_first_fall_censor():
    count = 20
    contact = np.ones(count, bool)
    loaded = np.ones(count, bool)
    xyz = np.zeros((count, 3), float)
    xyz[:, 0] = np.linspace(0.0, 0.10, count)
    velocity = np.zeros((count, 3), float)
    penetration = np.zeros(count, float)
    signals = derive_contact_signals(contact, loaded, xyz, velocity, penetration, 15)
    active = persistent_oracle(
        signals.tangential_anchor_drift_m,
        signals.slip_calibration_valid,
        signals.contact_episode_id,
        SLIP_THRESHOLD_M,
        SLIP_PERSISTENCE_MS,
    )
    assert np.any(active[:15])
    assert not np.any(active[15:])
    assert not np.any(active & signals.touchdown_transient)


def test_future_fold_validator_detects_pair_run_episode_and_variation_leakage():
    clean = [
        {"fold": fold, "pair_id": f"p{fold}", "variation_group": f"v{fold}",
         "run_id": f"r{fold}", "episode_group": f"e{fold}"}
        for fold in range(3)
    ]
    assert validate_future_fold_rows(clean)["valid"]
    leaking = clean + [{
        "fold": 1, "pair_id": "p0", "variation_group": "v0",
        "run_id": "r0", "episode_group": "e0",
    }]
    audit = validate_future_fold_rows(leaking)
    assert not audit["valid"] and audit["group_leakage_count"] == 4


def test_artifact_guard_is_exact_allowlist_and_fail_closed(tmp_path: Path):
    allowed = tmp_path / "safe.json"
    allowed.write_text(json.dumps({"development_only": True}), encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    guard = ArtifactAccessGuard(tmp_path, ["safe.json"], ledger)
    assert guard.read_json("safe.json", "test") == {"development_only": True}
    with pytest.raises(PermissionError):
        guard.read_json("holdout.json", "forbidden")
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["blocked_access_count"] == 1
    assert [row["status"] for row in saved["events"]] == ["completed", "blocked"]
