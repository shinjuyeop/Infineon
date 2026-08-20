import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from run_walking_fusion10_observability_audit_v1 import (
    DEVELOPMENT_CHECKPOINT,
    OUTPUT,
    STARTING_CHECKPOINT,
    TERRAIN_VIEWS,
    UPSTREAM_SHA256,
    protocol,
    train_zero_fp_threshold,
    verify_upstreams,
)
from walking_fusion10_observability_audit_v1 import (
    LinearProbe,
    anticipation_semantics,
    assert_diagnostic_source,
    causal_windows,
    contact_reference_features,
    fit_logistic_probe,
    phase_balanced_indices,
    split_integrity,
    terrain_collapse_gate,
    window_features,
)


ROOT = Path(__file__).resolve().parents[4]


def test_causal_window_construction_and_no_future_data():
    trace = np.arange(120 * 10, dtype=np.float32).reshape(120, 10)
    endpoints = np.asarray([19, 50, 119])
    before = causal_windows(trace, endpoints, 20)
    changed = trace.copy()
    changed[51:] = -999999
    after = causal_windows(changed, endpoints[:2], 20)
    assert before.shape == (3, 20, 10)
    assert np.array_equal(before[0], trace[:20])
    assert np.array_equal(before[1], trace[31:51])
    assert np.array_equal(before[:2], after)


def test_run_level_split_integrity_detects_leakage():
    clean = split_integrity(np.asarray(["a", "b"]), np.asarray(["train", "validation"]))
    assert clean["split_leakage_count"] == 0
    leaking = split_integrity(
        np.asarray(["a", "a", "b"]), np.asarray(["train", "validation", "validation"])
    )
    assert leaking["run_leakage_count"] == 1
    assert leaking["episode_leakage_count"] == 1
    assert leaking["split_leakage_count"] == 2


def test_phase_and_class_balanced_sampling_is_exact_and_deterministic():
    labels = np.repeat(np.arange(4), 16)
    phases = np.tile(np.repeat(["LOADING", "MID_STANCE"], 8), 4)
    first = phase_balanced_indices(
        labels, phases, balance_class=True, balance_phase=True, cap_per_cell=5
    )
    second = phase_balanced_indices(
        labels, phases, balance_class=True, balance_phase=True, cap_per_cell=5
    )
    assert np.array_equal(first, second)
    cells = [(int(labels[index]), str(phases[index])) for index in first]
    assert set(cells) == {(label, phase) for label in range(4) for phase in ("LOADING", "MID_STANCE")}
    assert all(cells.count(cell) == 5 for cell in set(cells))
    assert "class_phase_balanced" in TERRAIN_VIEWS


def test_outer_and_spatial_arrays_are_fail_closed():
    forbidden = (
        "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz",
        "simulation/outputs/walking_bounded_retraining_v1_sink_holdout/traces.npz",
        "simulation/outputs/walking_terrain_transition_v1_pilot/transition_traces.npz",
    )
    for path in forbidden:
        with pytest.raises(PermissionError):
            assert_diagnostic_source(path)
    assert_diagnostic_source("simulation/outputs/walking_bounded_retraining_v1/summary.json")
    assert protocol()["outer_boundary"]["fd5b9f0_v03_v04_v05_trace_access_count"] == 0


def test_deterministic_probe_evaluation_and_reload(tmp_path):
    rng = np.random.default_rng(20260820)
    values = rng.normal(size=(160, 12))
    labels = (values[:, 0] + 0.2 * values[:, 1] > 0).astype(np.int8)
    first = fit_logistic_probe(values, labels)
    second = fit_logistic_probe(values, labels)
    assert np.array_equal(first.predict(values), second.predict(values))
    assert np.allclose(first.coefficient, second.coefficient, atol=0.0, rtol=0.0)
    path = tmp_path / "probe.npz"
    first.save(path)
    loaded = LinearProbe.load(path)
    assert np.array_equal(first.predict(values), loaded.predict(values))


def test_terrain_collapse_gate_rejects_majority_and_zero_recall():
    failed = terrain_collapse_gate({
        "macro_accuracy": 0.28,
        "worst_class_recall": 0.0,
        "majority_class_prediction_rate": 0.82,
    })
    assert failed["pass"] is False
    assert failed["class_collapse_absent"] is False
    passed = terrain_collapse_gate({
        "macro_accuracy": 0.80,
        "worst_class_recall": 0.65,
        "majority_class_prediction_rate": 0.40,
    })
    assert passed["pass"] is True


def test_slip_anticipation_semantics_keeps_pre_and_post_distinct():
    oracle = np.zeros(20, bool)
    oracle[10:] = True
    firing = np.zeros(20, bool)
    firing[[8, 12]] = True
    result = anticipation_semantics(oracle, firing)
    assert result["anticipation"] is True
    assert result["post_onset_detection"] is True
    assert result["latency_ms"] == 2


def test_train_threshold_is_stricter_than_every_hard_negative():
    dataset = {"role": np.asarray(["hard_negative", "hard_negative", "slip_candidate"])}
    score = np.asarray([0.2, 0.8, 0.9])
    threshold = train_zero_fp_threshold(dataset, score)
    assert threshold > 0.8
    assert threshold < 0.9


def test_sink_stateful_contact_feature_is_causal():
    fusion = np.arange(30 * 10, dtype=np.float32).reshape(30, 10)
    episode = np.r_[-np.ones(5, int), np.zeros(20, int), -np.ones(5, int)]
    loaded = episode == 0
    before = contact_reference_features(fusion, 17, episode, loaded)
    modified = fusion.copy()
    modified[18:] = -1e6
    after = contact_reference_features(modified, 17, episode, loaded)
    assert np.array_equal(before, after)
    assert before[0] == 12


def test_window_features_are_finite_and_fixed_definition():
    values = np.ones((3, 50, 10), np.float32)
    features = window_features(values)
    assert features.shape == (3, 80)
    assert np.isfinite(features).all()


def test_immutable_artifact_provenance_matches_checkpoint():
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in UPSTREAM_SHA256
    }
    assert actual == UPSTREAM_SHA256
    verified = verify_upstreams()
    assert verified["starting_checkpoint"] == STARTING_CHECKPOINT
    assert verified["development_checkpoint"] == DEVELOPMENT_CHECKPOINT
    assert verified["mismatch_count"] == 0


def test_generated_artifact_hash_graph_and_invariants_when_present():
    if not (OUTPUT / "manifest.json").is_file():
        return
    required = {
        "protocol.json", "manifest.json", "summary.json", "audit.md",
        "terrain_confusion.csv", "terrain_phase_class_balance.csv",
        "terrain_probe_metrics.csv", "slip_window_ablation.csv",
        "slip_semantics_compatibility.csv", "sink_window_ablation.csv",
        "sink_feature_correlation.csv", "domain_compatibility.csv",
        "readiness.json", "diagnostic_arrays.npz",
        "hazard_diagnostic_probes.npz",
    }
    assert required <= {path.name for path in OUTPUT.iterdir()}
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["generated_files"]:
        path = OUTPUT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    audit_protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    assert manifest["outer_trace_access_count"] == 0
    assert audit_protocol["sample_rate_hz"] == 1000
    assert audit_protocol["causality"]["future_input_samples"] == 0
    assert summary["data_integrity"]["run_episode_split"]["split_leakage_count"] == 0
    assert summary["readiness"]["DETERMINISTIC_RELOAD_PARITY_READY"] is True
    assert summary["production_artifacts_changed"] is False
    assert summary["next_step"] in {
        "bounded retraining v2", "stateful detector prototype",
        "walking-specific/gait-routed detector", "label/task redesign",
        "additional targeted acquisition", "stop due to sensor observability limitation",
    }
