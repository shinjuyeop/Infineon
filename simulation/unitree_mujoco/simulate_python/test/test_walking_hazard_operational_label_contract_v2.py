import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from run_walking_hazard_operational_label_contract_v2 import (
    HASH_ONLY_PATHS,
    OUTPUT,
    UPSTREAM_SHA256,
    predeclared_protocol,
)
from walking_fusion10_observability_audit_v1 import split_integrity
from walking_hazard_operational_label_contract_v2 import (
    DEPRECATED_FIELDS,
    OFFLINE_ONLY_FIELDS,
    RUNTIME_CAPABLE_FIELDS,
    first_fall_censor,
    label_contract_v2,
    legacy_mapping_rows,
    onset_pulses,
    operational_offline_labels,
    pulse_lengths,
    readiness_dependency_graph,
    reflex_activation_contract_v2,
    risk_target_from_physical,
    sink_candidate_labels,
    sustained_within_episode,
    transition_count,
    valid_loaded_contact,
)
from walking_stateful_hazard_prototype_v1 import StatefulHazardMachine


ROOT = Path(__file__).resolve().parents[4]


def synthetic_trace(length: int = 180) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {
        "loaded_contact": np.ones(length, bool),
        "touchdown_transient": np.zeros(length, bool),
        "pre_fall_valid": np.ones(length, bool),
        "slip_calibration_valid": np.ones(length, bool),
        "sink_calibration_valid": np.ones(length, bool),
        "tangential_anchor_drift_m": np.zeros(length),
        "loaded_penetration_change_m": np.zeros(length),
        "contact_episode_id": np.zeros(length, np.int32),
        "pelvis_xyz": np.zeros((length, 3)),
        "pelvis_velocity_xyz": np.zeros((length, 3)),
    }
    trace["pelvis_xyz"][:, 2] = 0.75
    return trace


def complete_flags() -> dict[str, bool]:
    return {
        "WALKING_OPERATIONAL_LABEL_DATA_READY": True,
        "WALKING_OPERATIONAL_LABEL_SPLIT_INTEGRITY_READY": True,
        "WALKING_OPERATIONAL_LABEL_CAUSAL_CONTRACT_READY": True,
        "WALKING_SLIP_RISK_LABEL_READY": True,
        "WALKING_SLIP_REFLEX_CONTRACT_READY": True,
        "WALKING_SLIP_NEW_BLIND_HOLDOUT_AUTHORIZED": True,
        "WALKING_SINK_PHYSICAL_LABEL_READY": True,
        "WALKING_SINK_OPERATIONAL_LABEL_READY": False,
        "WALKING_SINK_REFLEX_CONTRACT_READY": False,
        "WALKING_SYSTEM_SCHEMA_MIGRATION_AUTHORIZED": False,
        "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED": False,
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
    }


def test_label_namespace_uniqueness():
    contract = label_contract_v2()
    names = OFFLINE_ONLY_FIELDS + RUNTIME_CAPABLE_FIELDS + DEPRECATED_FIELDS
    assert len(names) == len(set(names)) == 15
    assert contract["namespace_unique"] is True


def test_offline_runtime_field_separation():
    assert set(OFFLINE_ONLY_FIELDS).isdisjoint(RUNTIME_CAPABLE_FIELDS)
    contract = label_contract_v2()
    assert all(not value["runtime_available"] for value in contract["offline_only"].values())
    assert contract["physical_oracle_is_runtime_contract"] is False


def test_slip_risk_horizon_boundary_is_inclusive():
    active = np.zeros(140, bool)
    active[110:120] = True
    valid = np.ones(140, bool)
    episode = np.zeros(140, int)
    target = risk_target_from_physical(active, valid, episode, 100)
    assert target[9] is np.False_
    assert target[10] is np.True_
    assert target[109] is np.True_
    assert target[119] is np.True_


def test_runtime_contract_has_no_future_or_oracle_input():
    slip = label_contract_v2()["runtime_capable"]["slip_risk"]
    assert "runtime: none" in slip["future_information"]
    assert tuple(inspect.signature(StatefulHazardMachine.step).parameters) == (
        "self", "fusion10_sample",
    )


def test_normal_fp_definition_is_not_relabeling():
    protocol = predeclared_protocol()
    assert protocol["slip"]["normal_fp"] == "any runtime firing in a hard-negative run"
    assert protocol["sink_gate"]["normal_positive_runs"] == 0


def test_too_early_fp_definition_uses_fixed_100ms():
    protocol = predeclared_protocol()
    assert protocol["slip"]["risk_horizon_ms"] == 100
    assert "more than 100 ms" in protocol["slip"]["too_early"]


def test_air_is_excluded_from_valid_loaded_contact():
    trace = synthetic_trace(8)
    trace["loaded_contact"][:3] = False
    valid = valid_loaded_contact(trace)
    assert not np.any(valid[:3]) and np.all(valid[3:])


def test_touchdown_transient_is_excluded():
    trace = synthetic_trace(8)
    trace["touchdown_transient"][2:5] = True
    valid = valid_loaded_contact(trace)
    assert not np.any(valid[2:5])


def test_first_fall_censor_latches_through_run_end():
    valid = np.asarray([True, True, False, True, True])
    censor = first_fall_censor(valid)
    assert np.array_equal(censor, [False, False, True, True, True])


def test_episode_reset_prevents_persistence_carryover():
    condition = np.ones(8, bool)
    valid = np.ones(8, bool)
    episode = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    output = sustained_within_episode(condition, valid, episode, 3)
    assert np.array_equal(output, [False, False, True, True, False, False, True, True])


def test_deterministic_label_generation():
    trace = synthetic_trace()
    trace["tangential_anchor_drift_m"][60:] = 0.06
    trace["loaded_penetration_change_m"][90:] = 0.006
    trace["pelvis_xyz"][90:, 2] -= 0.03
    trace["pelvis_velocity_xyz"][90:, 2] = -0.1
    first = operational_offline_labels(trace)
    second = operational_offline_labels(trace)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_sink_candidates_are_nested_and_physically_gated():
    trace = synthetic_trace()
    trace["loaded_penetration_change_m"][50:] = 0.006
    trace["pelvis_xyz"][50:, 2] -= 0.03
    trace["pelvis_velocity_xyz"][50:, 2] = -0.1
    labels = sink_candidate_labels(trace)
    assert np.all(~labels["B_support_degradation_risk"] | labels["A_penetration_risk"])
    assert np.all(~labels["C_recovery_required_sink"] | labels["B_support_degradation_risk"])


def test_run_and_episode_split_integrity():
    result = split_integrity(
        np.asarray(["v00", "v01", "v02"]),
        np.asarray(["train", "train", "validation"]),
        np.asarray([0, 0, 0]),
    )
    assert result["split_leakage_count"] == 0


def test_outer_content_non_access_contract():
    protocol = predeclared_protocol()
    assert protocol["data"]["outer_content_load_count"] == 0
    assert protocol["data"]["new_sink_holdout_content_load_count"] == 0
    assert len(HASH_ONLY_PATHS) == 2
    assert all("outer_validation" in path for path in HASH_ONLY_PATHS)


def test_legacy_mapping_is_complete():
    expected = {
        "slip", "sink", "HAZARD_REFLEX_REQUIRED", "CASE_REFLEX_REQUIRED",
        "RECOVERY_REQUIRED", "mismatch/dual-hazard flags",
    }
    assert {row["legacy_field"] for row in legacy_mapping_rows()} == expected


def test_reflex_contract_has_all_states_and_no_sink_trigger():
    contract = reflex_activation_contract_v2()
    assert set(contract["states"]) == {
        "NO_HAZARD", "RISK", "EVIDENCE_PERSISTENT", "RECOVERY_REQUIRED", "COOLDOWN",
    }
    assert "prohibited" in contract["trigger_semantics"]["sink"]
    assert contract["trigger_semantics"]["physical_oracle_input"] is False


def test_readiness_dependency_graph():
    graph = readiness_dependency_graph(complete_flags())
    assert all(graph.values())
    broken = complete_flags()
    broken["WALKING_SLIP_RISK_LABEL_READY"] = False
    assert readiness_dependency_graph(broken)["slip_holdout_dependency_pass"] is False


def test_pulse_and_transition_quality_helpers():
    values = np.asarray([False, True, True, False, True, False])
    assert np.array_equal(onset_pulses(values), [False, True, False, False, True, False])
    assert pulse_lengths(values) == [2, 1]
    assert transition_count(values) == 4


def test_protocol_forbids_training_and_authorizations():
    protocol = predeclared_protocol()
    assert protocol["training_runs"] == 0
    assert protocol["model_or_threshold_changes"] == 0
    assert not any(protocol["readiness_constraints"].values())


def test_immutable_upstream_sha():
    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in UPSTREAM_SHA256
    }
    assert actual == UPSTREAM_SHA256


def test_generated_artifact_hash_graph_when_present():
    if not (OUTPUT / "manifest.json").is_file():
        return
    required = {
        "protocol.json", "manifest.json", "summary.json", "readiness.json",
        "audit.md", "label_contract_v2.json", "reflex_activation_contract_v2.json",
        "legacy_mapping.csv", "label_quality.csv", "slip_operational_metrics.csv",
        "sink_label_candidate_metrics.csv", "reaction_time_margin.csv",
        "invalid_region_audit.csv", "outer_non_access.json",
        "slip_operational_label_timeline.png", "sink_candidate_label_timeline.png",
    }
    assert required <= {path.name for path in OUTPUT.iterdir()}
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["generated_files"]:
        path = OUTPUT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert manifest["outer_content_load_count"] == 0
    assert manifest["model_training_runs"] == 0
    assert manifest["int8_or_vela_executed"] is False
