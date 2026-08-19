import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from run_walking_hazard_slip_nested_calibration_v2 import (
    EXPECTED_FIRST_COMMAND_OBSERVATIONS_S,
    IMMUTABLE_SHA256,
    STARTING_CHECKPOINT,
    VARIATIONS,
    immutable_hash_audit,
    outer_conditions,
)
from walking_hazard_slip_nested_calibration_v2 import (
    FOLDS,
    development_fold_manifest,
    episode_latency_rows,
    select_nested_candidate,
)


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "simulation" / "outputs" / "walking_hazard_slip_nested_calibration_v2"


def test_outer_variations_are_new_bounded_and_controller_discrete():
    assert [item.index for item in VARIATIONS] == [3, 4, 5]
    assert np.allclose(
        [item.initial_locomotion_phase_fraction for item in VARIATIONS],
        [1.0 / 6.0, 1.0 / 2.0, 5.0 / 6.0],
    )
    assert [item.command_onset_delay_s for item in VARIATIONS] == [0.06, 0.08, 0.10]
    assert EXPECTED_FIRST_COMMAND_OBSERVATIONS_S == (0.08, 0.10, 0.12)
    assert not set(EXPECTED_FIRST_COMMAND_OBSERVATIONS_S) & {0.02, 0.04, 0.06}


def test_outer_matrix_is_exactly_27_normal_plus_9_ice():
    conditions = outer_conditions(3.0)
    assert len(conditions) == 36
    assert sum(item.acquisition_role == "hard_negative" for item in conditions) == 27
    assert sum(item.acquisition_role == "slip_candidate" for item in conditions) == 9
    assert {item.walking_speed_mps for item in conditions} == {0.10, 0.15, 0.20}
    assert {item.variation.index for item in conditions} == {3, 4, 5}


def test_fold_manifest_keeps_each_run_and_episodes_in_one_role():
    manifests = [
        {
            "run_id": f"run_v{variation}",
            "variation_index": variation,
            "acquisition_role": "hard_negative",
        }
        for variation in range(3)
    ]
    result = development_fold_manifest(manifests)
    assert result["run_fold_role_leakage_count"] == 0
    assert len(result["runs"]) == 9
    for fold_id, selection, validation in FOLDS:
        rows = [row for row in result["runs"] if row["fold_id"] == fold_id]
        assert {row["variation_index"] for row in rows if row["fold_role"] == "selection"} == set(selection)
        assert {row["variation_index"] for row in rows if row["fold_role"] == "internal_validation"} == {validation}
        assert all(row["ownership"] == "whole run and all contact episodes" for row in rows)


def _candidate_rows(threshold, persistence, margin, p95, mean):
    return [{
        "threshold_m": threshold,
        "persistence_ms": persistence,
        "fold_id": fold,
        "fold_pass": True,
        "all_folds_pass": True,
        "selection_minimum_threshold_margin_m": margin,
        "internal_validation_minimum_threshold_margin_m": margin,
        "selection_p95_physical_onset_latency_ms": p95,
        "internal_validation_p95_physical_onset_latency_ms": p95,
        "selection_mean_physical_onset_latency_ms": mean,
        "internal_validation_mean_physical_onset_latency_ms": mean,
        "sufficient_margin": False,
        "selection_score": "",
        "tie_break_rank": "",
        "selected": False,
    } for fold in range(3)]


def test_selector_uses_sufficient_margin_then_latency_not_longer_persistence():
    rows = (
        _candidate_rows(0.050, 20, 0.025, 130.0, 80.0)
        + _candidate_rows(0.050, 3, 0.021, 115.0, 70.0)
        + _candidate_rows(0.030, 3, 0.010, 100.0, 60.0)
    )
    selected = select_nested_candidate(rows)
    assert selected["threshold_m"] == 0.050
    assert selected["persistence_ms"] == 3
    assert "80%" in selected["selection_rule"]
    assert "lower persistence" in selected["selection_rule"]


def test_selector_has_no_outer_argument_or_outer_data_dependency():
    parameters = inspect.signature(select_nested_candidate).parameters
    assert list(parameters) == ["rows"]
    rows = _candidate_rows(0.050, 3, 0.020, 100.0, 70.0)
    selected_before = select_nested_candidate(rows)
    poisoned_outer = np.full((9, 3000), np.nan)
    selected_after = select_nested_candidate(rows)
    assert poisoned_outer.shape == (9, 3000)
    assert selected_before == selected_after
    assert selected_after["selected_without_outer_data"] is True


def test_episode_latency_separates_physical_onset_and_persistence_completion():
    trace = {
        "slip_calibration_valid": np.asarray([0, 1, 1, 1, 1, 1, 0], bool),
        "contact_episode_id": np.asarray([-1, 2, 2, 2, 2, 2, -1]),
    }
    fire = np.asarray([0, 0, 0, 0, 1, 1, 0], bool)
    rows = episode_latency_rows("r", 0.1, 3, trace, fire, 3)
    assert len(rows) == 1
    assert rows[0]["physical_label_onset_sample"] == 1
    assert rows[0]["threshold_crossing_sample"] == 2
    assert rows[0]["persistence_completion_sample"] == 4
    assert rows[0]["physical_onset_to_fire_latency_ms"] == 3
    assert rows[0]["threshold_crossing_to_persistence_completion_ms"] == 3
    assert rows[0]["model_inference_onset"].startswith("not_applicable")


def test_d2209cd_and_earlier_frozen_inputs_are_byte_unchanged():
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in IMMUTABLE_SHA256
    }
    assert actual == IMMUTABLE_SHA256
    audit = immutable_hash_audit()
    assert audit["starting_checkpoint"] == STARTING_CHECKPOINT
    assert audit["mismatch_count"] == 0
    assert audit["checkpoint_is_ancestor"] is True


def test_generated_outer_artifacts_schema_and_gates_when_present():
    if not OUTPUT.is_dir():
        return
    required = {
        "protocol.json", "development_fold_manifest.json",
        "nested_candidate_metrics.csv", "nested_selection.json",
        "outer_validation_manifest.json", "outer_validation_metrics.csv",
        "duplicate_trace_audit.csv", "fall_censor_audit.csv", "summary.json",
        "audit.md", "outer_validation_traces.npz",
    }
    assert required <= {path.name for path in OUTPUT.iterdir()}
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    selection = json.loads((OUTPUT / "nested_selection.json").read_text())
    outer = json.loads((OUTPUT / "outer_validation_manifest.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert protocol["sample_rate_hz"] == 1000
    assert protocol["data_boundaries"]["outer_data_access_during_selection"] is False
    assert selection["outer_trace_count_at_selection"] == 0
    assert selection["threshold_m"] == 0.050
    assert selection["persistence_ms"] == 3
    assert outer["development_outer_run_overlap_count"] == 0
    assert len(outer["runs"]) == 36
    assert len({row["full_endpoint_sha256"] for row in outer["runs"]}) == 36
    assert all(value is True for value in summary["gates"].values())
    with np.load(OUTPUT / "outer_validation_traces.npz", allow_pickle=False) as traces:
        assert traces["time_s"].shape == (36, 3000)
        assert traces["fusion10"].shape == (36, 3000, 10)
        assert traces["slip_calibration_valid"].dtype == np.bool_
        assert traces["slip_label_oracle_frozen_candidate"].shape == (36, 3000)
