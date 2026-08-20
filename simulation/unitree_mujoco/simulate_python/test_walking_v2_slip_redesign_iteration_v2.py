from __future__ import annotations

from pathlib import Path

import numpy as np

from run_walking_v2_slip_redesign_iteration_v2 import label_at
from walking_v2_bilateral_bounded_training import PhysicalSlipEpisode
from walking_v2_slip_redesign_iteration_v2 import (
    ACTIONABLE_RISK,
    EARLY_PRECURSOR,
    FAMILIES,
    NORMAL_NO_EVENT,
    PHYSICAL_ACTIVE_EVIDENCE,
    SEEDS,
    RuntimeStateConfig,
    SlipV2Model,
    contact_scoped_runtime_state,
    deterministic_selection,
    fit_slip_v2_model,
    make_nested_fold_manifest,
    validate_nested_fold_manifest,
)


def synthetic_metadata() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": f"run_{variation}_{context}", "variation_index": variation,
            "terrain_name": ("concrete", "ice")[context % 2],
            "speed_mps": (0.10, 0.15, 0.20)[context % 3],
        }
        for variation in range(6) for context in range(6)
    )


def test_nested_folds_have_no_run_or_variation_leakage() -> None:
    manifest = make_nested_fold_manifest(synthetic_metadata())
    validate_nested_fold_manifest(manifest)
    validation_runs: set[str] = set()
    for outer in manifest["outer_folds"]:
        train = set(outer["training_run_ids"])
        validation = set(outer["validation_run_ids"])
        assert not train & validation
        assert not set(outer["training_variations"]) & set(outer["validation_variations"])
        validation_runs.update(validation)
        for inner in outer["inner_mining_folds"]:
            assert not set(inner["fit_run_ids"]) & set(inner["mining_run_ids"])
            assert not validation & (set(inner["fit_run_ids"]) | set(inner["mining_run_ids"]))
    assert len(validation_runs) == len(synthetic_metadata())


def test_four_authoritative_label_states() -> None:
    episodes = [PhysicalSlipEpisode(3, 600, 650, 1)]
    assert label_at(50, 3, episodes)[0] == NORMAL_NO_EVENT
    assert label_at(200, 3, episodes)[0] == EARLY_PRECURSOR
    assert label_at(520, 3, episodes)[0] == ACTIONABLE_RISK
    assert label_at(620, 3, episodes)[0] == PHYSICAL_ACTIVE_EVIDENCE
    assert label_at(520, 9, episodes)[0] == NORMAL_NO_EVENT


def state_inputs(length: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    endpoints = np.asarray((0, 10, 20, 30, 40))
    shape = (length, 2)
    scores = {
        "normal": np.full(shape, 0.05), "early": np.full(shape, 0.05),
        "actionable": np.full(shape, 0.90), "active": np.full(shape, 0.0),
        "foot": np.full(shape, 0.90), "proposal": np.full(shape, 0.90),
    }
    loaded = np.ones((41, 2), bool)
    age = np.tile(np.arange(1, 42)[:, None], (1, 2))
    touchdown = np.zeros((41, 2), bool)
    return scores, endpoints, loaded, age, touchdown


def test_persistence_and_simultaneous_rule_select_exactly_one_foot() -> None:
    scores, endpoints, loaded, age, touchdown = state_inputs()
    scores["actionable"][:, 0] = 0.95
    scores["actionable"][:, 1] = 0.90
    output = contact_scoped_runtime_state(
        scores, endpoints, loaded, age, touchdown, RuntimeStateConfig.from_family("S4-A"),
    )
    assert not np.any(output.firing[:2])
    assert np.all(np.sum(output.firing[2:], axis=1) == 1)
    assert np.all(output.firing[2:, 0])
    assert output.simultaneous_crossings == 3


def test_contact_loss_and_new_touchdown_hard_reset_each_foot() -> None:
    scores, endpoints, loaded, age, touchdown = state_inputs()
    loaded[30, 0] = False
    age[40, 0] = 1
    touchdown[40, 0] = True
    output = contact_scoped_runtime_state(
        scores, endpoints, loaded, age, touchdown, RuntimeStateConfig.from_family("S4-A"),
    )
    assert output.reset_reason[3, 0] == "contact_loss"
    assert output.reset_reason[4, 0] == "new_touchdown"
    assert not output.firing[3, 0] and not output.firing[4, 0]
    assert output.owner_id[2, 0] != output.owner_id[4, 0]


def test_active_evidence_or_timer_cannot_promote_action() -> None:
    scores, endpoints, loaded, age, touchdown = state_inputs()
    scores["actionable"][:] = 0.10
    scores["active"][:] = 0.99
    output = contact_scoped_runtime_state(
        scores, endpoints, loaded, age, touchdown, RuntimeStateConfig.from_family("S4-C"),
    )
    assert not np.any(output.firing)


def test_future_samples_do_not_change_runtime_scores() -> None:
    from walking_v2_joint_terrain_slip_redesign_v1 import runtime_feature

    rng = np.random.default_rng(4)
    trace = rng.normal(size=(300, 20)).astype(np.float32)
    loaded = np.ones((300, 2), bool)
    age = np.tile(np.arange(1, 301)[:, None], (1, 2))
    phase = np.full((300, 2), 3, np.int8)
    endpoint = 240
    original = runtime_feature("S3", 0, endpoint, trace, loaded, age, phase)
    trace[endpoint + 1:] = 1e8
    assert np.array_equal(original, runtime_feature("S3", 0, endpoint, trace, loaded, age, phase))


def test_model_and_normalization_reload_are_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(160, 12))
    state = np.tile(np.arange(4), 40)
    foot = np.tile((0, 1), 80)
    weight = np.ones(160)
    model, _ = fit_slip_v2_model("S4-A", SEEDS[0], features, state, foot, weight, weight)
    path = tmp_path / "slip.npz"
    model.save(path)
    reloaded = SlipV2Model.load(path)
    before, after = model.scores(features), reloaded.scores(features)
    assert np.array_equal(model.mean, reloaded.mean)
    assert np.array_equal(model.scale, reloaded.scale)
    assert all(np.array_equal(before[key], after[key]) for key in before)


def test_selection_requires_gate_pass_and_is_deterministic() -> None:
    rows = []
    for index, family in enumerate(FAMILIES):
        rows.append({
            "family": family, "seed": SEEDS[index], "gate_pass": family != "S4-A",
            "actionable_episode_recall": 0.85, "minimum_speed_recall": 0.75,
            "affected_foot_accuracy": 0.95, "median_warning_margin_ms": 30,
            "macs_per_tick": 20_000 + index,
        })
    assert deterministic_selection(rows)["family"] == "S4-B"
    assert deterministic_selection([{**row, "gate_pass": False} for row in rows]) is None
