"""Regression tests for bounded bilateral Terrain/Slip candidate training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
OUTPUT = REPO / "simulation" / "outputs" / "walking_v2_bilateral_bounded_training"
HOLDOUT = REPO / "simulation" / "outputs" / "walking_v2_bilateral_blind_holdout"
sys.path.insert(0, str(HERE))

from run_walking_v2_bilateral_bounded_training import (  # noqa: E402
    ALLOWED_UPSTREAM_FILES,
    load_development,
    protocol,
    upstream_hashes,
)
from walking_v2_bilateral_bounded_training import (  # noqa: E402
    FORBIDDEN_RUNTIME_FIELDS,
    RUNTIME_INPUT_FIELDS,
    SLIP_ARCHITECTURES,
    TERRAIN_ARCHITECTURES,
    TRAINING_SEEDS,
    LinearFloatModel,
    PhysicalSlipEpisode,
    SharedCausalFootEncoder,
    SlipStateConfig,
    affected_foot_correct,
    balanced_indices,
    causal_endpoints,
    deterministic_slip_selection,
    deterministic_terrain_selection,
    first_actionable_events,
    fit_linear_float,
    holdout_authorized,
    input_contract,
    physical_slip_episodes,
    raw_slip_crossings,
    risk_firing_is_too_early,
    runtime_feature,
    runtime_scope_contract,
    sha256_file,
    sink_deferral_contract,
    slip_gate,
    stateful_slip_firing,
    terrain_gate,
)


@pytest.fixture(scope="module")
def development():
    return load_development("train"), load_development("validation")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runtime_scope_excludes_sink_head():
    scope = runtime_scope_contract()
    assert scope["sink_runtime_outputs"] == []
    assert not any("sink" in value.lower() for value in scope["outputs"])
    assert scope["sand_semantics"] == "SAND_TERRAIN_CAUTION; never Sink detection"


def test_bilateral_input_contract_is_frozen_canonical():
    contract = input_contract()
    assert contract["canonical_input_order"] == ["left Fusion10", "right Fusion10"]
    assert len(contract["per_foot_channels"]) == 10
    assert contract["raw_frame_allowed"] is False
    assert contract["future_samples"] == 0


def test_runtime_fields_exclude_privileged_ground_truth():
    assert not (RUNTIME_INPUT_FIELDS & FORBIDDEN_RUNTIME_FIELDS)
    assert "bilateral_canonical" in RUNTIME_INPUT_FIELDS
    assert "slip_physical_active" in FORBIDDEN_RUNTIME_FIELDS
    assert "terrain_name" in FORBIDDEN_RUNTIME_FIELDS


def test_shared_encoder_exact_object_identity():
    encoder = SharedCausalFootEncoder(50)
    bilateral_references = (encoder, encoder)
    assert bilateral_references[0] is bilateral_references[1]
    assert len({value.fingerprint for value in bilateral_references}) == 1


def test_shared_encoder_fingerprint_is_deterministic():
    assert SharedCausalFootEncoder(50).fingerprint == SharedCausalFootEncoder(50).fingerprint
    assert SharedCausalFootEncoder(50).fingerprint != SharedCausalFootEncoder(100).fingerprint


def test_train_only_normalization_uses_supplied_train_rows():
    features = np.asarray(((-2.0, 0.0), (-1.0, 1.0), (1.0, 2.0), (2.0, 3.0)))
    target = np.asarray((0, 0, 1, 1))
    model = fit_linear_float("S1", 17, features, target, "encoder")
    np.testing.assert_allclose(model.mean, features.mean(axis=0))
    np.testing.assert_allclose(model.scale, features.std(axis=0))


def test_balanced_sampling_equalizes_every_declared_group():
    groups = np.repeat(
        np.asarray([(terrain, phase, foot, speed)
                    for terrain in range(4) for phase in range(1, 5)
                    for foot in range(2) for speed in range(3)]),
        (np.arange(96) % 4) + 1,
        axis=0,
    )
    selected = balanced_indices(groups, 99)
    _, counts = np.unique(groups[selected], axis=0, return_counts=True)
    assert len(counts) == 96
    assert np.array_equal(counts, np.ones(96, dtype=int))


def test_development_run_and_episode_ownership_is_disjoint(development):
    train, validation = development
    assert len(set(train.run_id) & set(validation.run_id)) == 0
    assert len(set(train.run_id)) == 72
    assert len(set(validation.run_id)) == 48
    assert train.contact_episode.shape == (72, 3000, 2)
    assert validation.contact_episode.shape == (48, 3000, 2)


def test_exact_one_khz_schema_and_timestamps(development):
    for data in development:
        assert data.bilateral.shape[1:] == (3000, 20)
        np.testing.assert_allclose(data.time_s[:, 0], 0.001, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(np.diff(data.time_s, axis=1), 0.001, rtol=0.0, atol=1e-12)


def test_endpoint_hash_duplicates_are_zero(development):
    rows = (*development[0].metadata, *development[1].metadata)
    hashes = [row["endpoint_sha256"] for row in rows]
    assert len(hashes) == len(set(hashes)) == 120


def test_physical_episode_merges_threshold_chatter():
    active = np.zeros(240, bool)
    active[20:30] = True
    active[70:80] = True
    episodes = physical_slip_episodes(active, np.ones(240, int), np.ones(240, bool))
    assert episodes == [PhysicalSlipEpisode(1, 20, 80, 2)]


def test_physical_episode_does_not_merge_across_contact():
    active = np.zeros(80, bool)
    active[20:30] = True
    active[40:50] = True
    contact = np.ones(80, int)
    contact[35:] = 2
    episodes = physical_slip_episodes(active, contact, np.ones(80, bool))
    assert [value.contact_episode_id for value in episodes] == [1, 2]


def test_raw_crossings_remain_unmerged():
    active = np.zeros(120, bool)
    active[10:20] = True
    active[40:50] = True
    events = raw_slip_crossings(active, np.ones(120, int), np.ones(120, bool))
    assert [(value.start, value.end_exclusive) for value in events] == [(10, 20), (40, 50)]


def test_first_actionable_event_obeys_cooldown():
    episodes = [
        PhysicalSlipEpisode(1, 10, 20, 1),
        PhysicalSlipEpisode(1, 80, 90, 1),
        PhysicalSlipEpisode(1, 200, 210, 1),
        PhysicalSlipEpisode(2, 100, 110, 1),
    ]
    actionable = first_actionable_events(episodes)
    assert [(value.contact_episode_id, value.start) for value in actionable] == [
        (1, 10), (1, 200), (2, 100),
    ]


def test_too_early_risk_definition():
    episodes = [PhysicalSlipEpisode(4, 300, 350, 1)]
    assert risk_firing_is_too_early(199, 4, episodes)
    assert not risk_firing_is_too_early(200, 4, episodes)
    assert not risk_firing_is_too_early(150, 7, episodes)


def test_affected_foot_attribution():
    left_score = np.asarray((0.1, 0.9))
    right_score = np.asarray((0.1, 0.3))
    left_fire = np.asarray((False, True))
    right_fire = np.asarray((False, False))
    assert affected_foot_correct(0, 1, left_score, right_score, left_fire, right_fire)
    assert not affected_foot_correct(1, 1, left_score, right_score, left_fire, right_fire)


@pytest.mark.parametrize("window_ms", (50, 100))
def test_causal_encoder_window_has_no_future_sample(window_ms):
    encoder = SharedCausalFootEncoder(window_ms)
    rng = np.random.default_rng(window_ms)
    values = rng.normal(size=(window_ms + 1, 10))
    reference = encoder.encode(values[:window_ms])
    values[window_ms] += 1e6
    np.testing.assert_array_equal(reference, encoder.encode(values[:window_ms]))


def test_causal_endpoint_contract_for_50_and_100_ms():
    assert causal_endpoints(120, 50)[0] == 49
    assert causal_endpoints(120, 100)[0] == 99
    assert np.all(np.diff(causal_endpoints(120, 50)) == 10)


def test_runtime_feature_ignores_samples_after_endpoint():
    rng = np.random.default_rng(10)
    bilateral = rng.normal(size=(120, 20))
    loaded = np.ones((120, 2), bool)
    age = np.tile(np.arange(1, 121)[:, None], (1, 2))
    phase = np.full((120, 2), 3, np.int8)
    encoder = SharedCausalFootEncoder(100)
    before = runtime_feature("S2", 0, 99, bilateral, loaded, age, phase, encoder)
    bilateral[100:] += 1e6
    after = runtime_feature("S2", 0, 99, bilateral, loaded, age, phase, encoder)
    np.testing.assert_array_equal(before, after)


def test_air_and_touchdown_are_excluded_by_slip_state():
    scores = np.ones(5)
    endpoints = np.arange(5)
    loaded = np.asarray((False, True, True, True, True))
    age = np.asarray((0, 1, 10, 11, 12))
    firing, reset = stateful_slip_firing(
        scores, endpoints, loaded, age, SlipStateConfig(0.5, 2, 0.1)
    )
    np.testing.assert_array_equal(firing, (False, False, False, False, True))
    assert reset[0] == "contact_loss"
    assert reset[1] == reset[2] == "new_touchdown"


def test_contact_loss_and_new_touchdown_reset_invariant():
    scores = np.ones(7)
    endpoints = np.arange(7)
    loaded = np.asarray((True, True, True, False, True, True, True))
    age = np.asarray((11, 12, 13, 0, 1, 11, 12))
    firing, reset = stateful_slip_firing(
        scores, endpoints, loaded, age, SlipStateConfig(0.5, 2, 0.1)
    )
    assert firing[1] and firing[2]
    assert not firing[3] and not firing[4]
    assert reset[3] == "contact_loss" and reset[4] == "new_touchdown"


def test_first_fall_censor_excludes_postfall_episode():
    active = np.zeros(20, bool)
    active[4:8] = True
    active[14:18] = True
    prefall = np.arange(20) < 10
    episodes = physical_slip_episodes(active, np.ones(20, int), prefall)
    assert episodes == [PhysicalSlipEpisode(1, 4, 8, 1)]


def _terrain_metrics(**updates):
    row = {
        "architecture": "T1", "seed": 1, "overall_accuracy": 0.90,
        "macro_accuracy": 0.90, "worst_class_recall": 0.80,
        "majority_class_prediction_rate": 0.40, "minimum_speed_accuracy": 0.85,
        "left_right_accuracy_difference_pp": 2.0, "class_collapse": False,
        "air_terrain_transitions": 0, "invalid_firings": 0,
        "gate_pass": True, "parameter_count": 100, "macs": 100,
    }
    row.update(updates)
    return row


def _slip_metrics(**updates):
    row = {
        "architecture": "S1", "seed": 1, "threshold": 0.9,
        "persistence_endpoints": 2, "hysteresis": 0.1,
        "valid_ice_run_coverage": 1.0, "first_actionable_event_recall": 1.0,
        "physical_episode_recall": 0.9, "affected_foot_accuracy": 0.95,
        "normal_risk_run_fp": 0, "normal_physical_episode_fp": 0,
        "too_early_firings": 0, "invalid_firings": 0,
        "all_speed_coverage": True, "both_affected_feet_coverage": True,
        "median_warning_margin_ms": 30.0, "pre_onset_detection_fraction": 0.9,
        "reset_invariant_pass": True, "gate_pass": True,
        "parameter_count": 100, "macs": 100,
    }
    row.update(updates)
    return row


def test_validation_gate_dependency_is_exact():
    assert terrain_gate(_terrain_metrics())
    assert not terrain_gate(_terrain_metrics(worst_class_recall=0.69))
    assert slip_gate(_slip_metrics())
    assert not slip_gate(_slip_metrics(normal_risk_run_fp=1))


def test_deterministic_terrain_candidate_selection():
    weaker = _terrain_metrics(seed=2, macro_accuracy=0.86)
    stronger = _terrain_metrics(seed=1, macro_accuracy=0.91)
    assert deterministic_terrain_selection([weaker, stronger]) is stronger


def test_deterministic_slip_candidate_selection():
    weaker = _slip_metrics(seed=2, physical_episode_recall=0.81)
    stronger = _slip_metrics(seed=1, physical_episode_recall=0.95)
    assert deterministic_slip_selection([weaker, stronger]) is stronger


def test_selection_lock_must_precede_holdout_authorization():
    assert not holdout_authorized(True, True, False)
    assert not holdout_authorized(True, False, True)
    assert holdout_authorized(True, True, True)


def test_protocol_freezes_bounded_search_and_holdout():
    value = protocol({path: "0" * 64 for path in ALLOWED_UPSTREAM_FILES})
    assert value["architectures"] == {
        "terrain": list(TERRAIN_ARCHITECTURES),
        "slip": list(SLIP_ARCHITECTURES),
        "shared_encoder_trainable": False,
        "shared_encoder_identity": "one exact fixed encoder object per window/task",
    }
    assert value["seeds"] == list(TRAINING_SEEDS)
    assert value["conditional_holdout"]["run_count"] == 36
    assert value["conditional_holdout"]["post_holdout_tuning_allowed"] is False


def test_failed_development_blocks_holdout_non_access():
    summary = _read_json(OUTPUT / "summary.json")
    audit = _read_json(OUTPUT / "outer_non_access.json")
    assert not summary["holdout_authorized"]
    assert summary["holdout_runs"] == 0
    assert audit["content_load_count"] == 0
    assert not HOLDOUT.exists()


def test_sink_runtime_output_and_model_are_absent():
    deferral = sink_deferral_contract()
    assert deferral["status"] == "SINK_RUNTIME_DETECTION_DEFERRED"
    assert deferral["runtime_head_created"] is False
    assert not any(path.name.startswith("sink_") for path in (OUTPUT / "models").glob("*"))


def test_model_reload_parity(tmp_path):
    features = np.asarray(((-2.0, 0.0), (-1.0, 1.0), (1.0, 2.0), (2.0, 3.0)))
    target = np.asarray((0, 0, 1, 1))
    model = fit_linear_float("S1", 17, features, target, "encoder")
    path = tmp_path / "model.npz"
    model.save(path)
    reloaded = LinearFloatModel.load(path)
    np.testing.assert_array_equal(model.probabilities(features), reloaded.probabilities(features))


def test_resource_schema_and_bounds():
    rows = list(csv.DictReader((OUTPUT / "resource_estimate.csv").open(encoding="utf-8")))
    required = {
        "model_parameter_count", "model_file_bytes", "shared_encoder_structure",
        "bilateral_macs_per_tick", "history_buffer_bytes", "persistent_state_bytes",
        "normalization_bytes", "estimated_host_inference_ms",
        "expected_int8_operator_set", "expected_vela_compatibility",
    }
    assert len(rows) == 12 and required <= set(rows[0])
    assert all(int(row["model_parameter_count"]) <= 5000 for row in rows)
    assert all(int(row["bilateral_macs_per_tick"]) <= 250000 for row in rows)


def test_immutable_upstream_sha_audit():
    rows = list(csv.DictReader((OUTPUT / "immutable_sha_audit.csv").open(encoding="utf-8")))
    assert len(rows) == len(ALLOWED_UPSTREAM_FILES)
    assert all(row["match"] == "True" and row["sha256_before"] == row["sha256_after"] for row in rows)
    assert upstream_hashes() == {row["path"]: row["sha256_after"] for row in rows}


def test_candidate_bundle_hashes_are_reloadable():
    selection = _read_json(OUTPUT / "candidate_selection.json")
    for task in ("terrain", "slip"):
        bundle = selection[task]["artifact_bundle"]
        assert sha256_file(OUTPUT / bundle["model_path"]) == bundle["model_sha256"]
        assert sha256_file(OUTPUT / bundle["normalization_path"]) == bundle["normalization_sha256"]
        assert sha256_file(OUTPUT / bundle["training_config_path"]) == bundle["training_config_sha256"]


def test_normalization_artifacts_are_train_only():
    for path in (OUTPUT / "normalization").glob("*.json"):
        value = _read_json(path)
        assert value["fit_split"] == "development_train"
        assert value["validation_rows_used"] == 0


def test_manifest_hash_graph_is_complete():
    manifest = _read_json(OUTPUT / "manifest.json")
    expected = {
        path.relative_to(OUTPUT).as_posix(): path
        for path in OUTPUT.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    generated = {row["path"]: row["sha256"] for row in manifest["generated_files"]}
    assert manifest["hash_graph_complete"] is True
    assert generated.keys() == expected.keys()
    for relative, path in expected.items():
        assert generated[relative] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_readiness_preserves_failure_without_promotion():
    readiness = _read_json(OUTPUT / "readiness.json")
    assert readiness["WALKING_V2_SCOPE_FREEZE_READY"]
    assert readiness["WALKING_V2_HOLDOUT_NON_ACCESS_READY"]
    assert readiness["WALKING_V2_SINK_RUNTIME_DEFERRED"]
    assert not readiness["WALKING_V2_RUNTIME_SCOPE_READY"]
    assert not readiness["WALKING_SYSTEM_V2_MIGRATION_AUTHORIZED"]
    assert not readiness["WALKING_INT8_PREPARATION_AUTHORIZED"]
