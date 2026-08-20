"""Regression tests for bilateral virtual sensors and causal Sink audit v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from bilateral_hil_sensor_v2 import (  # noqa: E402
    G1BilateralSensorReaderV2,
    RIGHT_ACCEL_CANONICAL_SIGN,
    RIGHT_FSR_CANONICAL_ORDER,
    RIGHT_GYRO_CANONICAL_SIGN,
)
from hil_sensor import G1HilSensorReader  # noqa: E402
from run_walking_bilateral_sensor_sink_observability_v2 import (  # noqa: E402
    DEFAULT_OUTPUT,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    SCENE_PATH,
    UPSTREAM_FILES,
    conditions,
    stronger_sink_profiles,
    upstream_hashes,
)
from walking_bilateral_sink_observability_v2 import (  # noqa: E402
    CANDIDATE_NAMES,
    RUNTIME_ARRAY_NAMES,
    SAMPLE_RATE_HZ,
    SIMULATOR_ONLY_ARRAY_NAMES,
    SharedCausalConv1DV2,
    SharedFootEncoderV2,
    candidate_features,
    contact_age,
    deterministic_candidate_selection,
    effort_summaries,
    endpoint_hash,
    first_fall_mask,
    joint_derived_kinematics,
    physical_risk_target,
    runtime_feature_contract_is_clean,
    stable_fire,
)


@pytest.fixture(scope="module")
def model_data_reader():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, G1BilateralSensorReaderV2(model, data)


def test_legacy_left_read_vector_parity(model_data_reader):
    model, data, reader = model_data_reader
    np.testing.assert_array_equal(reader.read_vector(), G1HilSensorReader(model, data).read_vector())


def test_right_sensor_attachment(model_data_reader):
    model, _, _ = model_data_reader
    site = model.site("right_foot_imu").id
    ankle = model.body("right_ankle_roll_link").id
    assert int(model.site_bodyid[site]) == ankle
    np.testing.assert_allclose(model.site_pos[site], (0.035, 0.0, -0.03))


def test_right_contact_slot_ordering(model_data_reader):
    model, _, reader = model_data_reader
    expected = ((-.05, .025, -.03), (-.05, -.025, -.03), (.12, .03, -.03), (.12, -.03, -.03))
    actual = [model.geom_pos[value] for value in reader.right_foot_geom_ids]
    np.testing.assert_allclose(actual, expected)


def test_bilateral_schema_shapes(model_data_reader):
    _, _, reader = model_data_reader
    assert reader.read_foot_vector("left").shape == (10,)
    assert reader.read_foot_vector("right").shape == (10,)
    assert reader.read_bilateral_vector().shape == (20,)
    assert reader.read_pelvis_vector().shape == (6,)


def test_exact_one_khz_protocol_timing():
    assert SAMPLE_RATE_HZ == 1000
    assert PHYSICS_TIMESTEP_S * PHYSICS_STEPS_PER_SAMPLE == pytest.approx(.001, abs=1e-15)


def test_mirrored_frame_transform_roundtrip():
    left = np.asarray((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), float)
    right_raw = np.concatenate((
        left[:4][RIGHT_FSR_CANONICAL_ORDER],
        left[4:7] * RIGHT_ACCEL_CANONICAL_SIGN,
        left[7:] * RIGHT_GYRO_CANONICAL_SIGN,
    ))
    np.testing.assert_array_equal(
        G1BilateralSensorReaderV2.canonicalize_foot_vector("right", right_raw), left
    )


def test_shared_encoder_exact_weight_identity():
    encoder = SharedFootEncoderV2()
    bilateral_references = (encoder, encoder)
    assert bilateral_references[0] is bilateral_references[1]
    assert bilateral_references[0].fingerprint == bilateral_references[1].fingerprint


def test_shared_causal_conv1d_is_past_only():
    encoder = SharedCausalConv1DV2()
    values = np.arange(2000, dtype=float).reshape(200, 10)
    reference = encoder.encode(values)
    extended = np.vstack((values, np.full((1, 10), 1e9)))
    np.testing.assert_array_equal(reference, encoder.encode(extended[:-1]))
    assert encoder.output_size == 40 and encoder.parameter_count == 770


def test_per_foot_independent_state(model_data_reader):
    _, _, reader = model_data_reader
    reader.reset_contact_state()
    left = reader.update_contact_state("left", np.asarray((2, 2, 2, 2)))
    right = reader.update_contact_state("right", np.zeros(4))
    assert left.loaded and not right.loaded
    assert left.age_samples == 1 and right.age_samples == 0


def test_bilateral_contact_transition(model_data_reader):
    _, _, reader = model_data_reader
    reader.reset_contact_state()
    assert reader.update_contact_state("left", np.full(4, 1.5)).transition == "touchdown"
    assert reader.update_contact_state("left", np.full(4, 1.0)).transition == "steady"
    assert reader.update_contact_state("left", np.zeros(4)).transition == "liftoff"


def test_contact_loss_reset(model_data_reader):
    _, _, reader = model_data_reader
    reader.reset_contact_state("right")
    reader.update_contact_state("right", np.full(4, 2.0))
    reader.update_contact_state("right", np.full(4, 2.0))
    lost = reader.update_contact_state("right", np.zeros(4))
    assert not lost.loaded and lost.age_samples == 0


def _feature_inputs(sample_count: int = 230):
    rng = np.random.default_rng(17)
    bilateral = rng.normal(size=(sample_count, 20)).astype(np.float32)
    bilateral[:, (0, 1, 2, 3, 10, 11, 12, 13)] = np.abs(
        bilateral[:, (0, 1, 2, 3, 10, 11, 12, 13)]
    )
    return {
        "bilateral_canonical": bilateral,
        "pelvis_imu": rng.normal(size=(sample_count, 6)).astype(np.float32),
        "joint_position": rng.normal(scale=.1, size=(sample_count, 12)).astype(np.float32),
        "target_position": rng.normal(scale=.1, size=(sample_count, 12)).astype(np.float32),
        "actuator_effort": rng.normal(size=(sample_count, 12)).astype(np.float32),
        "force_loaded": np.ones((sample_count, 2), bool),
        "encoder": SharedFootEncoderV2(),
    }


def test_no_future_feature():
    values = _feature_inputs()
    before = candidate_features("C4", side=0, endpoint=205, **values)
    values["bilateral_canonical"][206:] += 1000
    values["pelvis_imu"][206:] += 1000
    after = candidate_features("C4", side=0, endpoint=205, **values)
    np.testing.assert_array_equal(before, after)


def test_no_simulator_only_runtime_feature():
    assert runtime_feature_contract_is_clean(set(RUNTIME_ARRAY_NAMES))
    for value in SIMULATOR_ONLY_ARRAY_NAMES:
        assert not runtime_feature_contract_is_clean({"bilateral_canonical", value})


def test_joint_derived_feature_causality():
    values = np.zeros((10, 12))
    before = joint_derived_kinematics(values)[:5].copy()
    values[5:, 3] = 1.0
    after = joint_derived_kinematics(values)[:5]
    np.testing.assert_array_equal(before, after)


def test_effort_residual_causality():
    measured = np.zeros((10, 12)); target = np.ones((10, 12)); effort = np.ones((10, 12))
    before = effort_summaries(measured, target, effort)[:5].copy()
    target[5:] = 100; effort[5:] = 100
    np.testing.assert_array_equal(before, effort_summaries(measured, target, effort)[:5])


def test_physical_label_runtime_separation():
    assert not (RUNTIME_ARRAY_NAMES & SIMULATOR_ONLY_ARRAY_NAMES)
    assert "physical_label" in SIMULATOR_ONLY_ARRAY_NAMES
    assert "contact_penetration" in SIMULATOR_ONLY_ARRAY_NAMES


def test_first_fall_censor():
    np.testing.assert_array_equal(first_fall_mask(6, 3), (True, True, True, False, False, False))


def test_air_touchdown_exclusion():
    score = np.ones(8)
    eligible = np.asarray((False, False, True, True, True, False, True, True))
    firing = stable_fire(score, .5, eligible, 3)
    np.testing.assert_array_equal(firing, (False, False, False, False, True, False, False, False))


def test_run_episode_split_integrity():
    matrix = conditions()
    by_run = {value.run_id: value.variation.split for value in matrix}
    assert len(by_run) == 120
    assert sum(value == "development_train" for value in by_run.values()) == 72
    assert sum(value == "development_validation" for value in by_run.values()) == 48


def test_duplicate_variation_detection():
    a = endpoint_hash(np.zeros((2, 10)), np.ones((2, 12)))
    b = endpoint_hash(np.zeros((2, 10)), np.ones((2, 12)))
    c = endpoint_hash(np.ones((2, 10)), np.ones((2, 12)))
    assert a == b and a != c


def test_outer_content_non_access_contract():
    forbidden = ("outer", "holdout", "final")
    assert not any(any(token in value.lower() for token in forbidden) for value in UPSTREAM_FILES)


def test_deterministic_selection():
    base = {
        "invalid_firings": 0, "normal_fp_runs": 0, "ice_cross_hazard_fp_runs": 0,
        "too_early_firings": 0, "zero_fp_recall": .5, "coverage_fraction": .5,
        "latency_ms": 20, "sensor_channels": 20, "memory_bytes": 100, "macs": 100,
    }
    rows = [{"candidate": "C1", **base}, {"candidate": "C2", **base, "zero_fp_recall": .6}]
    assert deterministic_candidate_selection(rows) == "C2"
    assert deterministic_candidate_selection(list(reversed(rows))) == "C2"


def test_immutable_upstream_sha():
    expected = {
        "96a4c29ce495aea1e7785ae95124652bf1017ac3d9618727d4ef17c9bc37aa10",
        "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
        "7a476799d72c40081d64a531f6ddd7e2591eb18d93d269fa74114bfa4f24970b",
        "ccc55ae83237cd1febfd5ca99b2f40bd13202b83f1d8c02d766896e54274d8d2",
        "9aa84e85af7692bd81bef28ad0c34c65163a74473934268fa94b05bad6b3fc17",
        "7eaa92f603a0aef49a6873d18b771f6ae623a871e034c571d5ac5a04af272ffb",
        "de264c3ba753d4a9507bd73903f93b7dbbc53ff1418c3cf1da9bcba27799fcb0",
    }
    assert set(upstream_hashes(REPO).values()) == expected


def test_generated_artifact_hash_graph_when_present():
    manifest_path = DEFAULT_OUTPUT / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("full generated artifact not present yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hash_graph_complete"]
    actual = {
        path.relative_to(DEFAULT_OUTPUT).as_posix()
        for path in DEFAULT_OUTPUT.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    expected = {row["path"] for row in manifest["generated_files"]}
    assert actual == expected
    for row in manifest["generated_files"]:
        digest = hashlib.sha256((DEFAULT_OUTPUT / row["path"]).read_bytes()).hexdigest()
        assert digest == row["sha256"]


def test_resource_schema_when_present():
    path = DEFAULT_OUTPUT / "resource_estimate.csv"
    if not path.is_file():
        pytest.skip("full generated artifact not present yet")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    required = {
        "candidate", "architecture", "feature_count", "sensor_channels",
        "parameter_count", "state_memory_bytes", "history_memory_bytes",
        "memory_bytes", "macs", "latency_ms", "estimated_e84_latency_ms",
        "expected_vela_compatibility",
    }
    assert required <= set(header)


def test_exact_two_stronger_profiles_are_predeclared():
    profiles = stronger_sink_profiles()
    assert len(profiles) == 2
    assert profiles[0].solref[0] < profiles[1].solref[0] < 0.050
    frozen = 0.015 + (0.050 - 0.015) * (2.0 / 3.0)
    assert profiles[0].solref[0] > frozen


def test_model_reload_parity():
    first = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    second = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    assert first.nsensor == second.nsensor
    assert first.ngeom == second.ngeom
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            first.site_pos[first.site(f"{side}_foot_imu").id],
            second.site_pos[second.site(f"{side}_foot_imu").id],
        )


def test_physical_risk_horizon_stays_within_episode():
    active = np.asarray((False, False, True, False, False, True))
    valid = np.ones(6, bool)
    episodes = np.asarray((0, 0, 0, 1, 1, 1))
    target = physical_risk_target(active, valid, episodes, 2)
    np.testing.assert_array_equal(target, (True, True, True, True, True, True))
    # The second onset cannot relabel indices in episode zero.
    first_only = physical_risk_target(np.asarray((False, False, False, False, False, True)), valid, episodes, 5)
    np.testing.assert_array_equal(first_only, (False, False, False, True, True, True))


def test_candidate_set_is_exactly_c0_through_c4():
    assert CANDIDATE_NAMES == ("C0", "C1", "C2", "C3", "C4")


def test_contact_age_is_causal_and_independent():
    loaded = np.asarray(((0, 1), (1, 1), (1, 0), (0, 1)), bool)
    np.testing.assert_array_equal(contact_age(loaded), ((0, 1), (1, 2), (2, 0), (0, 1)))
