import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from run_walking_bounded_retraining_v1 import (
    DEVELOPMENT_CHECKPOINT,
    IMMUTABLE_SHA256,
    MIXTURE_RATIOS,
    STARTING_CHECKPOINT,
    WINDOWS,
    sink_holdout_conditions,
    verify_sources,
)
from walking_bounded_retraining_v1 import (
    Normalizer,
    causal_windows,
    physical_oracle,
    select_hazard,
    select_terrain,
    split_audit,
    stable_fire,
    weighted_normalizer,
)


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "simulation" / "outputs" / "walking_bounded_retraining_v1"


def test_causal_window_ends_exactly_at_label_endpoint():
    values = np.arange(100 * 10, dtype=np.float32).reshape(100, 10)
    endpoints = np.asarray([4, 17, 99])
    windows = causal_windows(values, endpoints, 5)
    assert windows.shape == (3, 5, 10)
    for index, endpoint in enumerate(endpoints):
        assert np.array_equal(windows[index], values[endpoint - 4:endpoint + 1])
        assert np.array_equal(windows[index, -1], values[endpoint])


def _hazard_trace():
    return {
        "tangential_anchor_drift_m": np.asarray([0, 0.06, 0.06, 0.06, 0, 0]),
        "slip_calibration_valid": np.asarray([0, 1, 1, 1, 0, 0], bool),
        "loaded_penetration_change_m": np.asarray([0] + [0.006] * 20 + [0]),
        "sink_calibration_valid": np.asarray([0] + [1] * 20 + [0], bool),
        "contact_episode_id": np.asarray([-1, 0, 0, 0, -1, -1]),
    }


def test_frozen_physical_oracle_reproduction():
    trace = _hazard_trace()
    slip = physical_oracle(trace, "slip")
    assert slip.tolist() == [False, False, False, True, False, False]
    trace["contact_episode_id"] = np.asarray([-1] + [0] * 20 + [-1])
    sink = physical_oracle(trace, "sink")
    assert sink[-2] and np.count_nonzero(sink) == 1


def test_stable_fire_resets_across_air_touchdown_and_postfall_masks():
    score = np.ones(10)
    eligible = np.asarray([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], bool)
    fire = stable_fire(score, 0.5, 3, eligible)
    assert fire.tolist() == [False, False, False, False, True, False, False, False, False, False]
    assert not np.any(fire[~eligible])


def test_weighted_normalizer_uses_only_declared_groups_and_is_finite():
    first = np.zeros((4, 5, 10), np.float32)
    second = np.full((2, 5, 10), 2.0, np.float32)
    norm = weighted_normalizer((first, second), (1.0, 1.0))
    assert isinstance(norm, Normalizer)
    assert np.allclose(norm.mean, 1.0)
    assert np.allclose(norm.std, 1.0)
    assert np.isfinite(norm.transform(np.concatenate([first, second]))).all()


def test_deterministic_selection_and_holdout_non_access_api():
    terrain = [
        {"candidate_gate_pass": True, "walking_validation_macro_accuracy": 0.8,
         "walking_validation_accuracy": 0.9, "static_validation_accuracy": 0.96,
         "parameters": 100, "mixture_ratio": 0.5, "training_seed": 2},
        {"candidate_gate_pass": True, "walking_validation_macro_accuracy": 0.9,
         "walking_validation_accuracy": 0.9, "static_validation_accuracy": 0.95,
         "parameters": 100, "mixture_ratio": 1.0, "training_seed": 3},
    ]
    assert select_terrain(terrain) is terrain[1]
    hazard = [
        {"candidate_gate_pass": True, "walking_normal_false_positive_runs": 0,
         "walking_positive_run_recall": 1.0, "controlled_validation_run_recall": 0.9,
         "walking_p95_stable_latency_ms": 20, "parameters": 100,
         "mixture_ratio": 0.5, "training_seed": 2, "probability_threshold": 0.9,
         "runtime_persistence": 3},
        {"candidate_gate_pass": True, "walking_normal_false_positive_runs": 0,
         "walking_positive_run_recall": 1.0, "controlled_validation_run_recall": 0.9,
         "walking_p95_stable_latency_ms": 10, "parameters": 100,
         "mixture_ratio": 1.0, "training_seed": 3, "probability_threshold": 0.8,
         "runtime_persistence": 1},
    ]
    assert select_hazard(hazard) is hazard[1]
    assert list(inspect.signature(select_hazard).parameters) == ["rows"]
    assert list(inspect.signature(select_terrain).parameters) == ["rows"]


def test_run_and_episode_split_audit():
    manifests = [
        {"run_id": "a", "split": "walking_train", "contact_episode_ids": [0, 1]},
        {"run_id": "b", "split": "walking_validation", "contact_episode_ids": [0, 1]},
    ]
    assert split_audit(manifests)["split_leakage_count"] == 0
    manifests.append({"run_id": "a", "split": "walking_validation", "contact_episode_ids": [0]})
    result = split_audit(manifests)
    assert result["run_leakage_count"] == 1
    assert result["episode_leakage_count"] == 1


def test_bounded_grid_architecture_and_sink_holdout_matrix():
    assert MIXTURE_RATIOS == (0.25, 0.50, 1.00)
    assert WINDOWS == {"terrain": 50, "slip": 5, "sink": 20}
    conditions = sink_holdout_conditions()
    assert len(conditions) == 18
    assert {item.variation.index for item in conditions} == {3, 4, 5}
    assert {item.walking_speed_mps for item in conditions} == {0.10, 0.15, 0.20}
    assert len({item.profile.name for item in conditions}) == 2


def test_all_immutable_sources_match_checkpoint_regression():
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in IMMUTABLE_SHA256
    }
    assert actual == IMMUTABLE_SHA256
    audit = verify_sources()
    assert audit["starting_checkpoint"] == STARTING_CHECKPOINT
    assert audit["mismatch_count"] == 0


def test_generated_artifacts_and_model_reload_parity_when_present():
    if not (OUTPUT / "summary.json").is_file():
        return
    required = {
        "protocol.json", "source_hashes.json", "dataset_manifest.csv",
        "dataset_manifest.json", "split_audit.json", "dataset_statistics.json",
        "mixture_candidates.csv", "seed_metrics.csv", "terrain_selection.json",
        "slip_selection.json", "sink_selection.json", "baseline_walking_replay.csv",
        "validation_metrics.csv", "holdout_metrics.csv", "latency_metrics.csv",
        "spatial_transition_diagnostic.csv", "summary.json", "audit.md",
    }
    assert required <= {path.name for path in OUTPUT.iterdir()}
    summary = json.loads((OUTPUT / "summary.json").read_text())
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    lock = json.loads((OUTPUT / "selection_lock.json").read_text())
    assert protocol["fusion_contract"]["sample_rate_hz"] == 1000
    assert protocol["fusion_contract"]["channels"][-1] == "gyro_z"
    with np.load(OUTPUT / "walking_window_provenance.npz", allow_pickle=False) as provenance:
        checkpoint_keys = [
            key for key in provenance.files if key.endswith("__source_checkpoint")
        ]
        assert checkpoint_keys
        for key in checkpoint_keys:
            assert np.array_equal(np.unique(provenance[key]), [DEVELOPMENT_CHECKPOINT])
    assert lock["fd5b9f0_outer_trace_access_count_before_lock"] == 0
    assert lock["new_sink_holdout_runs_before_lock"] == 0
    assert all(item["parity"] for item in summary["model_reload_parity"].values())
    assert summary["production_artifacts_changed"] is False
