"""Regression tests for walking-v2 candidate failure localization v1."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
OUTPUT = REPO / "simulation" / "outputs" / "walking_v2_candidate_failure_localization_v1"
TRAINING = REPO / "simulation" / "outputs" / "walking_v2_bilateral_bounded_training"
sys.path.insert(0, str(HERE))

from run_walking_v2_candidate_failure_localization_v1 import (  # noqa: E402
    ALLOWED_AUDIT_FILES,
    SLIP_MODEL,
    TERRAIN_MODEL,
    upstream_hashes,
)
from walking_v2_bilateral_bounded_training import (  # noqa: E402
    PhysicalSlipEpisode,
    SlipStateConfig,
    affected_foot_correct,
    evaluation_invalid_firing_count,
    first_actionable_events,
    originating_risk_window_detections,
    physical_slip_episodes,
    sha256_file,
)
from walking_v2_candidate_failure_localization_v1 import (  # noqa: E402
    AUDIT_VARIANTS,
    classify_too_early,
    invalid_accounting,
    join_sample_ledgers,
    model_signature,
    replay_slip_state,
    sha256_array,
    tensor_statistics,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_diagnostic_t2_sample_key_join():
    base = {
        "run_id": "r", "foot_index": 0, "endpoint_sample": 199,
        "split": "validation", "variation": 3, "terrain": "ice",
        "speed_mps": .1, "foot": "left", "contact_phase": "MID_STANCE",
        "contact_episode_id": 1, "touchdown": False,
        "window_start_s": .001, "window_end_s": .2, "source_index": 0,
        "label": 2, "included": True, "mask_reason": "included",
    }
    other = {**base, "window_start_s": .151}
    rows, summary = join_sample_ledgers([base], [other])
    assert len(rows) == 1 and summary["both"] == 1
    assert rows[0]["window_mismatch"]


def test_population_mismatch_detection():
    summary = _json(OUTPUT / "summary.json")["terrain"]["population_summary"]
    assert summary["both"] == 15023
    assert summary["diagnostic_only"] == 322
    assert summary["t2_only"] == 2176
    assert summary["mask_mismatch"] == 1404
    assert summary["label_mismatch"] == 0


def test_tensor_stage_hashing_is_shape_dtype_and_value_sensitive():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert sha256_array(values) == sha256_array(values.copy())
    assert sha256_array(values) != sha256_array(values.astype(np.float64))
    assert sha256_array(values) != sha256_array(values.reshape(4, 3))
    stats = tensor_statistics(values)
    assert stats["finite_count"] == 12 and stats["element_count"] == 12


def test_raw_and_canonical_tensor_stages_are_separate():
    rows = _csv(OUTPUT / "terrain_tensor_stage_hashes.csv")
    by_stage = {row["stage"]: row for row in rows if row["pipeline"] == "source"}
    assert by_stage["raw_bilateral_fusion20"]["sha256"] != by_stage[
        "frame_canonical_bilateral_fusion20"
    ]["sha256"]


def test_train_only_normalization_provenance():
    for task, architecture, seed in (("terrain", "t2", 202608213), ("slip", "s1", 202608211)):
        value = _json(TRAINING / "normalization" / f"{task}_{architecture}_seed_{seed}.json")
        assert value["fit_split"] == "development_train"
        assert value["validation_rows_used"] == 0


def test_effective_sample_weight_is_applied():
    rows = _csv(OUTPUT / "terrain_training_health.csv")
    assert len(rows) == 6
    assert all(row["balance_applied_to_loss"] == "True" for row in rows)
    assert all(row["balanced_train_rows"] == "1440" for row in rows)


def test_class_phase_foot_speed_full_batch_balance():
    rows = _csv(OUTPUT / "terrain_training_health.csv")
    assert all(row["effective_group_count"] == "96" for row in rows)
    assert all(row["effective_group_min"] == row["effective_group_max"] == "15" for row in rows)
    assert all(row["batch_composition"] == "one deterministic balanced full batch" for row in rows)


def test_exact_diagnostic_probe_reproduction():
    terrain = _json(OUTPUT / "summary.json")["terrain"]
    assert terrain["reference_reproduced"]
    assert terrain["stored_accuracy"] == terrain["replayed_accuracy"] == 0.9456500488758554
    assert terrain["stored_macro_recall"] == terrain["replayed_macro_recall"] == 0.9094796501531782


def test_four_way_shadow_matrix_is_complete():
    rows = _csv(OUTPUT / "terrain_shadow_matrix.csv")
    assert {row["matrix_cell"] for row in rows} == {"A", "B", "C", "D"}
    metrics = {row["matrix_cell"]: row for row in rows}
    assert float(metrics["A"]["accuracy"]) > float(metrics["B"]["accuracy"])
    assert float(metrics["C"]["macro_recall"]) < .60
    assert float(metrics["D"]["macro_recall"]) == 0.6090079361501065


def test_model_graph_mac_verification():
    rows = _csv(OUTPUT / "terrain_training_health.csv")
    t2 = [row for row in rows if row["architecture"] == "T2"]
    assert all(row["mac_match"] == "True" for row in t2)
    assert all(row["verified_graph_macs"] == "1928" for row in t2)


def test_physical_episode_merge_determinism():
    active = np.zeros(200, bool)
    active[20:30] = True
    active[70:80] = True
    first = physical_slip_episodes(active, np.ones(200, int), np.ones(200, bool))
    second = physical_slip_episodes(active, np.ones(200, int), np.ones(200, bool))
    assert first == second == [PhysicalSlipEpisode(1, 20, 80, 2)]


def test_first_actionable_event_determinism():
    episodes = [
        PhysicalSlipEpisode(1, 10, 20, 1),
        PhysicalSlipEpisode(1, 80, 90, 1),
        PhysicalSlipEpisode(1, 200, 210, 1),
    ]
    assert first_actionable_events(episodes) == first_actionable_events(episodes)
    assert [value.start for value in first_actionable_events(episodes)] == [10, 200]


def test_too_early_classification_is_exhaustive_for_observed_rows():
    rows = _csv(OUTPUT / "slip_too_early_violations.csv")
    assert len(rows) == 40
    assert {row["classification"] for row in rows} == {
        "genuinely_over_100ms_before_next_independent_physical_episode"
    }


def test_too_early_classifier_distinguishes_previous_episode_latch():
    episodes = [
        PhysicalSlipEpisode(1, 100, 150, 1),
        PhysicalSlipEpisode(1, 400, 450, 1),
    ]
    category, upcoming = classify_too_early(250, 120, 1, episodes)
    assert category == "previous_physical_episode_latch_carry_over"
    assert upcoming == episodes[1]


def test_latch_originating_risk_window_ownership():
    firing = np.asarray((False, True, True, True, False, True, True))
    endpoints = np.arange(0, 70, 10)
    selected = np.asarray((2, 3, 5, 6))
    owned = originating_risk_window_detections(firing, endpoints, selected, 40)
    np.testing.assert_array_equal(owned, (5, 6))


def test_contact_loss_hard_reset():
    scores = np.ones(5)
    endpoints = np.arange(5)
    loaded = np.asarray((True, True, False, True, True))
    age = np.asarray((11, 12, 0, 1, 11))
    state = replay_slip_state(
        scores, endpoints, loaded, age, SlipStateConfig(.5, 2, .1),
        hard_contact_reset=True,
    )
    assert state.firing[1]
    assert not state.firing[2] and state.reset_reason[2] == "contact_loss"


def test_new_touchdown_hard_reset():
    scores = np.ones(5)
    endpoints = np.arange(5)
    loaded = np.ones(5, bool)
    age = np.asarray((11, 12, 1, 11, 12))
    state = replay_slip_state(
        scores, endpoints, loaded, age, SlipStateConfig(.5, 2, .1),
        hard_contact_reset=True,
    )
    assert state.reset_reason[2] == "new_touchdown"
    assert not state.firing[2]


def test_first_fall_censor_accounting_fix():
    assert evaluation_invalid_firing_count(0, 0, 53, strict_first_fall_censor=False) == 53
    assert evaluation_invalid_firing_count(0, 0, 53, strict_first_fall_censor=True) == 0
    assert invalid_accounting(0, 0, 53) == {
        "legacy_invalid_firings": 53,
        "strict_censor_invalid_firings": 0,
        "post_fall_state_outputs_reported_separately": 53,
    }


def test_invalid_denominator_integrity_and_exact_meaning():
    rows = _csv(OUTPUT / "slip_invalid_violations.csv")
    assert len(rows) == 53
    assert all(row["post_fall_mask"] == "True" and row["risk_state"] == "True" for row in rows)
    assert all(row["state_output_exists_but_evaluation_excluded"] == "True" for row in rows)


def test_affected_foot_attribution():
    left_score = np.asarray((.1, .9)); right_score = np.asarray((.1, .2))
    left_fire = np.asarray((False, True)); right_fire = np.asarray((False, False))
    assert affected_foot_correct(0, 1, left_score, right_score, left_fire, right_fire)


def test_r0_r4_deterministic_replay_and_correction():
    rows = _csv(OUTPUT / "slip_reset_variant_metrics.csv")
    assert [row["variant"] for row in rows] == list(AUDIT_VARIANTS)
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["R0"]["detected_physical_episodes"] == "17"
    assert by_variant["R4"]["detected_physical_episodes"] == "15"
    assert by_variant["R0"]["invalid_firings"] == "53"
    assert by_variant["R4"]["invalid_firings"] == "0"
    assert all(row["too_early_firings"] == "40" for row in rows)


def test_no_model_or_threshold_modification_across_variants():
    rows = _csv(OUTPUT / "slip_reset_variant_metrics.csv")
    for key in ("model_sha256", "threshold", "persistence_ms", "hysteresis"):
        assert len({row[key] for row in rows}) == 1
    assert rows[0]["model_sha256"] == sha256_file(SLIP_MODEL)


def test_model_signature_detects_any_model_or_config_change():
    base = model_signature(
        np.ones((1, 2)), np.zeros(1), np.zeros(2), np.ones(2), {"threshold": .9}
    )
    changed = model_signature(
        np.ones((1, 2)), np.zeros(1), np.zeros(2), np.ones(2), {"threshold": .8}
    )
    assert base != changed


def test_outer_access_incident_is_fail_closed_and_unused():
    value = _json(OUTPUT / "outer_non_access.json")
    readiness = _json(OUTPUT / "readiness.json")
    assert value["content_load_count"] == 1
    assert value["existing_outer_used_for_metrics_or_conclusions"] is False
    assert value["compliant_zero_access"] is False
    assert readiness["WALKING_V2_FAILURE_AUDIT_PROVENANCE_READY"] is False


def test_allowed_runtime_audit_list_excludes_outer_holdout_final_paths():
    forbidden = ("outer_validation", "holdout_metrics", "sink_holdout", "spatial", "final_test")
    assert not any(any(token in path for token in forbidden) for path in ALLOWED_AUDIT_FILES)


def test_immutable_upstream_sha():
    rows = _csv(OUTPUT / "immutable_sha_audit.csv")
    assert all(row["match"] == "True" and row["sha256_before"] == row["sha256_after"] for row in rows)
    assert upstream_hashes() == {row["path"]: row["sha256_after"] for row in rows}


def test_trace_row_count_and_replay_parity():
    with gzip.open(OUTPUT / "slip_score_state_trace.csv.gz", "rt", encoding="utf-8") as stream:
        count = sum(1 for _ in stream) - 1
    assert count == 27936
    summary = _json(OUTPUT / "summary.json")
    assert summary["slip"]["variant_invariants"]["trace_rows"] == count


def test_required_representative_plots_exist():
    names = (
        "terrain_diagnostic_vs_t2_confusion.png", "terrain_train_validation_curves.png",
        "terrain_per_class_logit_distribution.png", "slip_score_state_oracle_timeline.png",
        "slip_too_early_example.png", "slip_post_fall_reset_example.png",
    )
    assert all((OUTPUT / name).stat().st_size > 10_000 for name in names)


def test_candidate_readiness_and_forbidden_authorizations_remain_false():
    readiness = _json(OUTPUT / "readiness.json")
    for key in (
        "WALKING_V2_TERRAIN_FLOAT_CANDIDATE_READY",
        "WALKING_V2_SLIP_FLOAT_CANDIDATE_READY",
        "WALKING_V2_HOLDOUT_AUTHORIZED",
        "WALKING_SYSTEM_V2_MIGRATION_AUTHORIZED",
        "WALKING_INT8_PREPARATION_AUTHORIZED",
    ):
        assert readiness[key] is False


def test_frozen_candidate_files_match_recorded_sha():
    protocol = _json(OUTPUT / "protocol.json")
    assert sha256_file(TERRAIN_MODEL) == protocol["upstream_sha256"][
        "simulation/outputs/walking_v2_bilateral_bounded_training/models/terrain_t2_seed_202608213.npz"
    ]
    assert sha256_file(SLIP_MODEL) == protocol["slip_model_sha256"]


def test_artifact_hash_graph():
    manifest = _json(OUTPUT / "manifest.json")
    expected = {
        path.relative_to(OUTPUT).as_posix(): path
        for path in OUTPUT.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    actual = {row["path"]: row["sha256"] for row in manifest["generated_files"]}
    assert expected.keys() == actual.keys()
    for relative, path in expected.items():
        assert actual[relative] == hashlib.sha256(path.read_bytes()).hexdigest()
