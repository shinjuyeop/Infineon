import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from run_walking_stateful_hazard_prototype_v1 import (
    HASH_ONLY_PATHS,
    OUTPUT,
    UPSTREAM_SHA256,
    candidate_definitions,
    predeclared_protocol,
)
from walking_fusion10_observability_audit_v1 import LinearProbe, split_integrity
from walking_stateful_hazard_prototype_v1 import (
    CausalFeatureState,
    ContactPhaseTracker,
    DetectorState,
    MachineConfig,
    StatefulHazardMachine,
    estimate_resources,
    select_candidate,
)


ROOT = Path(__file__).resolve().parents[4]


def constant_probe(feature_count: int, probability: float) -> LinearProbe:
    intercept = np.log(probability / (1.0 - probability))
    return LinearProbe(
        mean=np.zeros(feature_count),
        scale=np.ones(feature_count),
        coefficient=np.zeros((1, feature_count)),
        intercept=np.asarray([intercept]),
        classes=np.asarray([0, 1]),
    )


def config(**changes: object) -> MachineConfig:
    values: dict[str, object] = {
        "detector": "slip", "candidate_id": "synthetic", "history_ms": 20,
        "use_touchdown_reference": False, "risk_enabled": True,
        "confirmed_enabled": True, "require_risk_before_confirm": True,
        "risk_on": 0.8, "risk_off": 0.5, "confirmed_on": 0.8,
        "confirmed_off": 0.5, "risk_persistence": 2,
        "confirmed_persistence": 2, "recovery_persistence": 2,
        "confirmation_delta_min": 0.0,
    }
    values.update(changes)
    return MachineConfig(**values)


def sample(fsr_sum: float) -> np.ndarray:
    result = np.zeros(10, np.float32)
    result[:4] = fsr_sum / 4.0
    return result


def enter_stable(machine: StatefulHazardMachine) -> None:
    for _ in range(31):
        machine.step(sample(10.0))


def test_causal_history_construction_is_bounded():
    state = CausalFeatureState(3)
    for value in range(5):
        state.update(np.full(10, value, np.float32), touchdown=value == 0)
    assert len(state.history) == 3
    assert np.array_equal(state.history[0], np.full(10, 2))
    assert state.vector(False).shape == (80,)


def test_no_future_sample_invariant():
    first = CausalFeatureState(5)
    second = CausalFeatureState(5)
    trace = np.arange(10 * 10, dtype=np.float32).reshape(10, 10)
    changed = trace.copy()
    changed[6:] = -1e6
    for index in range(6):
        first.update(trace[index], touchdown=index == 0)
        second.update(changed[index], touchdown=index == 0)
    assert np.array_equal(first.vector(True), second.vector(True))


def test_touchdown_reference_initialization_and_delta():
    state = CausalFeatureState(4)
    state.update(np.arange(10, dtype=np.float32), touchdown=True)
    at_touchdown = state.vector(True)
    state.update(np.arange(10, dtype=np.float32) + 2, touchdown=False)
    after = state.vector(True)
    assert at_touchdown.shape == (124,)
    assert np.all(at_touchdown[81:91] == 0)
    assert np.all(after[81:91] == 2)


def test_air_loading_stable_transition():
    tracker = ContactPhaseTracker()
    assert tracker.update(sample(0))[0] == DetectorState.AIR
    assert tracker.update(sample(10))[0] == DetectorState.LOADING
    for _ in range(29):
        assert tracker.update(sample(10))[0] == DetectorState.LOADING
    assert tracker.update(sample(10))[0] == DetectorState.STABLE_CONTACT


def test_risk_to_confirmed_uses_independent_evidence():
    machine = StatefulHazardMachine(
        config(), constant_probe(80, 0.95), constant_probe(80, 0.96)
    )
    enter_stable(machine)
    output = machine.step(sample(10))
    assert output.contact_state == DetectorState.HAZARD_RISK.value
    output = machine.step(sample(10))
    assert output.slip_confirmed is True
    assert machine.risk_probe is not machine.confirmed_probe


def test_risk_recovery_hysteresis():
    machine = StatefulHazardMachine(
        config(risk_persistence=1, confirmed_persistence=10),
        constant_probe(80, 0.95), constant_probe(80, 0.95),
    )
    enter_stable(machine)
    assert machine.state == DetectorState.HAZARD_RISK
    machine.risk_probe = constant_probe(80, 0.1)
    machine.confirmed_probe = constant_probe(80, 0.1)
    machine.step(sample(10))
    output = machine.step(sample(10))
    assert output.contact_state == DetectorState.RECOVERY.value
    assert output.slip_risk is False


def test_contact_loss_reset_clears_hazard():
    machine = StatefulHazardMachine(
        config(risk_persistence=1), constant_probe(80, 0.95), constant_probe(80, 0.95)
    )
    enter_stable(machine)
    output = machine.step(sample(0))
    assert output.contact_state == DetectorState.RESET.value
    assert output.reset_reason == "contact_loss"
    assert not output.slip_risk and not output.slip_confirmed


def test_episode_boundary_reset():
    machine = StatefulHazardMachine(
        config(), constant_probe(80, 0.95), constant_probe(80, 0.95)
    )
    enter_stable(machine)
    machine.request_reset("episode_boundary")
    output = machine.step(sample(10))
    assert output.contact_state == DetectorState.RESET.value
    assert output.reset_reason == "episode_boundary"


def test_fall_censor_reset():
    machine = StatefulHazardMachine(
        config(), constant_probe(80, 0.95), constant_probe(80, 0.95)
    )
    enter_stable(machine)
    machine.request_reset("fall_censor")
    output = machine.step(sample(10))
    assert output.reset_reason == "fall_censor"
    assert not output.slip_risk and not output.slip_confirmed


def test_runtime_has_no_oracle_or_metadata_input():
    parameters = tuple(inspect.signature(StatefulHazardMachine.step).parameters)
    assert parameters == ("self", "fusion10_sample")
    fields = set(inspect.signature(StatefulHazardMachine.request_reset).parameters)
    assert fields == {"self", "reason"}


def test_run_and_episode_split_integrity():
    clean = split_integrity(
        np.asarray(["v00", "v01", "v02"]),
        np.asarray(["train", "train", "validation"]),
        np.asarray([0, 0, 0]),
    )
    assert clean["split_leakage_count"] == 0


def test_outer_non_access_is_predeclared_and_bounded():
    protocol = predeclared_protocol()
    assert protocol["outer_boundary"]["outer_content_loads"] == 0
    assert protocol["outer_boundary"]["new_sink_holdout_access"] == 0
    assert len(HASH_ONLY_PATHS) == 2
    assert all("outer_validation" in value for value in HASH_ONLY_PATHS)


def test_threshold_selection_is_deterministic():
    base = {
        "invalid_firing_samples": 0, "normal_confirmed_fp_runs": 0,
        "pre_onset_confirmed_runs": 0, "confirmed_run_recall": 0.5,
        "risk_run_coverage": 1.0, "confirmed_latency_median_ms": 20,
        "parameter_count": 10, "persistent_state_bytes": 100,
        "mandatory_gate_pass": False,
    }
    rows = [
        {**base, "candidate_id": "a", "candidate_order": 0},
        {**base, "candidate_id": "b", "candidate_order": 1},
    ]
    first = select_candidate(rows)
    second = select_candidate(list(reversed(rows)))
    assert first[0]["candidate_id"] == second[0]["candidate_id"] == "a"


def test_probe_reload_parity_is_exact(tmp_path: Path):
    probe = constant_probe(80, 0.91)
    path = tmp_path / "probe.npz"
    probe.save(path)
    loaded = LinearProbe.load(path)
    values = np.arange(240, dtype=np.float64).reshape(3, 80)
    assert np.array_equal(probe.positive_score(values), loaded.positive_score(values))


def test_resource_estimate_schema():
    cfg = config()
    resources = estimate_resources(cfg, constant_probe(80, 0.9), constant_probe(80, 0.9))
    required = {
        "parameter_count", "persistent_state_bytes", "history_buffer_bytes",
        "estimated_macs_per_1khz_sample", "feature_count", "e84_u55_compatible",
        "vela_unsupported_ops_expected", "int8_or_vela_executed",
    }
    assert required == set(resources)
    assert resources["int8_or_vela_executed"] is False


def test_candidate_count_and_fixed_primary_histories():
    candidates = candidate_definitions()
    assert len(candidates["slip"]) == 4
    assert len(candidates["sink"]) == 5
    assert candidates["slip"][-1]["history_ms"] == 100
    assert candidates["sink"][-1]["history_ms"] == 200


def test_runtime_output_semantics_are_separate():
    machine = StatefulHazardMachine(
        config(detector="sink", risk_persistence=1, confirmed_persistence=1),
        constant_probe(80, 0.95), constant_probe(80, 0.95),
    )
    enter_stable(machine)
    output = machine.step(sample(10))
    assert output.sink_risk and output.sink_confirmed
    assert not output.slip_risk and not output.slip_confirmed


def test_immutable_upstream_sha_provenance():
    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in UPSTREAM_SHA256
    }
    assert actual == UPSTREAM_SHA256


def test_generated_artifact_provenance_hash_graph_when_present():
    if not (OUTPUT / "manifest.json").is_file():
        return
    required = {
        "protocol.json", "manifest.json", "summary.json", "readiness.json",
        "audit.md", "state_transition_definition.json", "runtime_feature_contract.json",
        "candidate_selection.json", "slip_stateful_metrics.csv",
        "sink_stateful_metrics.csv", "latency_metrics.csv",
        "invalid_firing_audit.csv", "reset_invariant_audit.csv",
        "resource_estimate.csv", "contact_phase_metrics.csv",
    }
    assert required <= {path.name for path in OUTPUT.iterdir()}
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["generated_files"]:
        path = OUTPUT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert manifest["outer_content_load_count"] == 0
    assert manifest["int8_or_vela_executed"] is False
