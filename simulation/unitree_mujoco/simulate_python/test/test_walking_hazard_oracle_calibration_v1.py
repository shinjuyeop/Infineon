import argparse
import hashlib
from pathlib import Path

import numpy as np

from run_walking_hazard_oracle_calibration_v1 import (
    VARIATIONS,
    acquisition_conditions,
    select_candidate,
)
from walking_hazard_oracle_calibration_v1 import (
    DIVERSITY_TRACE_KEYS,
    assign_calibration_splits,
    duplicate_trace_audit,
    persistent_oracle,
    split_integrity,
)


def test_persistence_resets_at_invalid_samples_and_episode_boundaries():
    observable = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    valid = np.asarray([0, 1, 1, 0, 1, 1, 1], bool)
    episodes = np.asarray([-1, 0, 0, 0, 1, 1, 1])
    fire = persistent_oracle(observable, valid, episodes, 0.1, 3)
    assert fire.tolist() == [False, False, False, False, False, False, True]


def test_persistence_never_fires_in_air_post_fall_or_touchdown_invalid_mask():
    observable = np.full(12, 1.0)
    loaded = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)
    transient = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    pre_fall = np.asarray([1] * 8 + [0] * 4, bool)
    valid = loaded & ~transient & pre_fall
    episodes = np.asarray([-1, -1] + [0] * 10)
    fire = persistent_oracle(observable, valid, episodes, 0.5, 3)
    assert not np.any(fire[~loaded])
    assert not np.any(fire[transient])
    assert not np.any(fire[~pre_fall])
    assert fire.tolist() == [False, False, False, False, False, False, True, True, False, False, False, False]


def _trace(offset=0.0):
    result = {}
    for key in DIVERSITY_TRACE_KEYS:
        if key in ("left_contact", "loaded_contact", "pre_fall_valid"):
            result[key] = np.asarray([False, True, True])
        elif key == "fusion10":
            result[key] = np.full((3, 10), offset)
        elif key.endswith("xyz"):
            result[key] = np.full((3, 3), offset)
        else:
            result[key] = np.asarray([np.nan, offset, offset + 0.1])
    return result


def test_duplicate_trace_audit_detects_names_only_replication():
    manifests = [
        {
            "run_id": f"r{i}", "condition_name": "c", "profile_name": "p",
            "walking_speed_mps": 0.1, "variation_index": i,
        }
        for i in range(3)
    ]
    rows = duplicate_trace_audit(manifests, [_trace(), _trace(), _trace(0.01)])
    assert len(rows) == 3
    assert sum(row["duplicate"] for row in rows) == 1
    assert next(row for row in rows if row["duplicate"])["run_id_a"] == "r0"
    assert next(row for row in rows if row["duplicate"])["run_id_b"] == "r1"


def test_three_variations_are_bounded_and_record_actual_values():
    assert [item.index for item in VARIATIONS] == [0, 1, 2]
    assert np.allclose(
        [item.initial_locomotion_phase_fraction for item in VARIATIONS],
        [0.0, 1.0 / 3.0, 2.0 / 3.0],
    )
    assert [item.command_onset_delay_s for item in VARIATIONS] == [0.0, 0.02, 0.04]
    args = argparse.Namespace(
        replicates=3, duration_s=3.0, speeds=[0.10, 0.15, 0.20], smoke=False
    )
    conditions = acquisition_conditions(args)
    assert len(conditions) == 54
    assert {item.variation.index for item in conditions} == {0, 1, 2}


def test_whole_run_split_has_no_episode_leakage():
    manifests = [
        {
            "run_id": f"r{i}", "condition_name": "c", "profile_name": "p",
            "acquisition_role": "hard_negative", "walking_speed_mps": 0.1,
            "variation_index": i, "variation_seed": 10 + i,
        }
        for i in range(3)
    ]
    splits = assign_calibration_splits(manifests)
    assert [row["split"] for row in splits] == [
        "calibration_train", "calibration_train", "calibration_validation"
    ]
    episodes = [
        {"run_id": row["run_id"], "contact_episode_id": episode}
        for row in manifests for episode in range(2)
    ]
    audit = split_integrity(splits, episodes)
    assert audit["train_run_count"] == 2
    assert audit["validation_run_count"] == 1
    assert audit["split_leakage_count"] == 0


def test_candidate_selection_prefers_margin_then_persistence():
    rows = [
        {
            "candidate_pass": True, "complete_positive_profile_count": 1,
            "minimum_threshold_margin_m": 0.002, "persistence_ms": 5,
            "mean_detection_latency_ms": 4.0, "selected": False,
        },
        {
            "candidate_pass": True, "complete_positive_profile_count": 1,
            "minimum_threshold_margin_m": 0.003, "persistence_ms": 10,
            "mean_detection_latency_ms": 9.0, "selected": False,
        },
    ]
    assert select_candidate(rows) is rows[1]
    assert rows[1]["selected"] is True


def test_40aa6af_recorder_runner_and_pilot_are_byte_unchanged():
    root = Path(__file__).resolve().parents[4]
    expected = {
        "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "e12cdb8699f70d448766de04f3dcc5e952ac43155e09540512428b408dba2b81",
        "simulation/unitree_mujoco/simulate_python/run_walking_hazard_ground_truth_v1.py": "c7e53baa44f9cbd6241949f2c797c370a6185be872247da820168b4cb8e0a503",
        "simulation/outputs/walking_hazard_ground_truth_v1_pilot/protocol.json": "04f8302cd8b8232de0bec2ef742287a9fd4ad4422ac8eb8a6cad9d49d76aac88",
        "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json": "96a4c29ce495aea1e7785ae95124652bf1017ac3d9618727d4ef17c9bc37aa10",
        "simulation/outputs/walking_hazard_ground_truth_v1_pilot/traces.npz": "cba2cf5f4ce915135a8607ee384e5b9b04d76a4ab2bbc0e3608b27e0b8d946d7",
    }
    actual = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in expected
    }
    assert actual == expected
