from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from walking_v2_joint_terrain_slip_redesign_v1 import (
    ARCHITECTURE_SPECS,
    SLIP_ARCHITECTURES,
    TERRAIN_ARCHITECTURES,
    TRAINING_SEEDS,
    ArtifactAccessGuard,
    DualHeadModel,
    ProjectedLinearModel,
    authoritative_owned_detections,
    contact_scoped_state,
    episode_balanced_weights,
    fit_dual_head,
    fit_projected_linear,
    invalid_firing_count,
    raking_weights,
    runtime_feature,
)


def test_candidate_matrix_is_preregistered_and_bounded() -> None:
    assert TERRAIN_ARCHITECTURES == ("T1", "T2", "T3")
    assert SLIP_ARCHITECTURES == ("S1", "S2", "S3")
    assert len(TRAINING_SEEDS) == 3
    assert all(int(ARCHITECTURE_SPECS[name]["history_ms"]) <= 200 for name in ARCHITECTURE_SPECS)


def test_artifact_guard_records_exact_allowlisted_read(tmp_path: Path) -> None:
    artifact = tmp_path / "safe" / "input.json"
    artifact.parent.mkdir()
    artifact.write_text('{"value": 7}\n', encoding="utf-8")
    log = tmp_path / "access.json"
    guard = ArtifactAccessGuard(tmp_path, ("safe/input.json",), log)
    assert guard.read_json("safe/input.json", "unit test") == {"value": 7}
    guard.assert_complete()
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert payload["read_event_count"] == 1
    assert payload["events"][0]["status"] == "completed"
    assert len(payload["events"][0]["sha256"]) == 64


@pytest.mark.parametrize("token", ("outer", "holdout", "final_test", "final-test", "spatial"))
def test_artifact_guard_fails_closed_for_forbidden_names(tmp_path: Path, token: str) -> None:
    with pytest.raises(PermissionError):
        ArtifactAccessGuard(tmp_path, (f"safe/{token}/input.json",), tmp_path / "access.json")


def test_corrected_r4_rejects_previous_episode_latch_and_postfall_attribution() -> None:
    endpoints = np.asarray((0, 10, 20, 30))
    latched_from_before_window = np.asarray((True, True, True, True))
    candidates = np.asarray((1, 2, 3))
    prefall = np.ones(4, bool)
    assert not len(authoritative_owned_detections(
        latched_from_before_window, endpoints, candidates, 10, prefall,
    ))

    rising_only_after_fall = np.asarray((False, False, True, True))
    prefall = np.asarray((True, True, False, False))
    assert not len(authoritative_owned_detections(
        rising_only_after_fall, endpoints, np.asarray((2, 3)), 10, prefall,
    ))


def test_postfall_outputs_are_excluded_from_invalid_numerator() -> None:
    assert invalid_firing_count(0, 0, 53) == 0
    assert invalid_firing_count(2, 3, 53) == 5


def test_contact_state_hard_resets_and_keeps_feet_separate() -> None:
    endpoints = np.asarray((0, 10, 20, 30, 40))
    raw = np.asarray((
        (True, False), (True, False), (True, True), (True, False), (True, False),
    ))
    loaded = np.zeros((41, 2), bool)
    age = np.zeros((41, 2), int)
    touchdown = np.zeros((41, 2), bool)
    loaded[10:30, 0] = True
    age[10, 0], age[20, 0] = 1, 11
    touchdown[10, 0] = True
    loaded[:, 1] = True
    age[:, 1] = np.arange(1, 42)
    loaded[40, 0] = True
    age[40, 0] = 1
    touchdown[40, 0] = True

    state, reset, owner = contact_scoped_state(raw, endpoints, loaded, age, touchdown)
    assert not state[1, 0]
    assert state[2, 0]
    assert not state[3, 0]
    assert not state[4, 0]
    assert state[2, 1]
    assert not state[2, 0] == state[2, 1] == False
    assert reset[3, 0] == "contact_loss"
    assert reset[4, 0] == "new_touchdown"
    assert owner[2, 0] >= 0 and owner[2, 1] >= 0


def test_runtime_features_are_causal_and_slot_symmetric() -> None:
    rng = np.random.default_rng(9)
    trace = rng.normal(size=(300, 20)).astype(np.float32)
    loaded = np.ones((300, 2), bool)
    age = np.tile(np.arange(1, 301)[:, None], (1, 2))
    phase = np.full((300, 2), 3, np.int8)
    endpoint = 240
    for architecture in (*TERRAIN_ARCHITECTURES, *SLIP_ARCHITECTURES):
        original = runtime_feature(architecture, 0, endpoint, trace, loaded, age, phase)
        future_mutated = trace.copy()
        future_mutated[endpoint + 1:] = 1e9
        assert np.array_equal(
            original,
            runtime_feature(architecture, 0, endpoint, future_mutated, loaded, age, phase),
        )
        swapped = np.concatenate((trace[:, 10:], trace[:, :10]), axis=1)
        assert np.array_equal(
            original,
            runtime_feature(architecture, 1, endpoint, swapped, loaded[:, ::-1], age[:, ::-1], phase[:, ::-1]),
        )


def test_200ms_terrain_startup_padding_is_causal() -> None:
    rng = np.random.default_rng(11)
    trace = rng.normal(size=(80, 20)).astype(np.float32)
    loaded = np.ones((80, 2), bool)
    age = np.tile(np.arange(1, 81)[:, None], (1, 2))
    phase = np.full((80, 2), 2, np.int8)
    endpoint = 49
    original = runtime_feature("T1", 0, endpoint, trace, loaded, age, phase)
    mutated = trace.copy()
    mutated[endpoint + 1:] = 1e8
    assert len(original) == 164
    assert np.array_equal(original, runtime_feature("T1", 0, endpoint, mutated, loaded, age, phase))


def test_weighting_retains_rows_and_equalizes_declared_mass() -> None:
    factors = np.asarray(((0, 0), (0, 0), (0, 1), (1, 0), (1, 1), (1, 1), (1, 1)))
    weight = raking_weights(factors)
    assert len(weight) == len(factors)
    assert np.all(weight > 0)
    for column in range(2):
        _, inverse = np.unique(factors[:, column], return_inverse=True)
        mass = np.bincount(inverse, weights=weight)
        assert np.max(mass) - np.min(mass) < 1e-8

    target = np.asarray((0, 0, 0, 1, 1, 1))
    unit = np.asarray(("c0", "c0", "c1", "e0", "e1", "e1"))
    episode_weight = episode_balanced_weights(target, unit)
    assert np.isclose(episode_weight[target == 0].sum(), episode_weight[target == 1].sum())
    assert np.isclose(episode_weight[unit == "c0"].sum(), episode_weight[unit == "c1"].sum())


def test_exact_model_and_normalization_reload(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(90, 8))
    target = np.tile(np.arange(3), 30)
    weight = np.ones(90)
    original, _ = fit_projected_linear("T1", TRAINING_SEEDS[0], features, target, weight)
    path = tmp_path / "model.npz"
    original.save(path)
    reloaded = ProjectedLinearModel.load(path)
    assert np.array_equal(original.mean, reloaded.mean)
    assert np.array_equal(original.scale, reloaded.scale)
    assert np.array_equal(original.probabilities(features), reloaded.probabilities(features))

    binary_a = (features[:, 0] > 0).astype(int)
    binary_b = (features[:, 1] > 0).astype(int)
    dual, _ = fit_dual_head(
        TRAINING_SEEDS[0], features, binary_a, binary_b, weight, weight,
    )
    dual_path = tmp_path / "dual.npz"
    dual.save(dual_path)
    dual_reload = DualHeadModel.load(dual_path)
    before = np.column_stack(dual.scores(features))
    after = np.column_stack(dual_reload.scores(features))
    assert np.array_equal(before, after)
