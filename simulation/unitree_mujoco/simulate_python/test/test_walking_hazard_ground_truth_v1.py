import hashlib
from pathlib import Path

import mujoco
import numpy as np

from walking_hazard_ground_truth_v1 import (
    box_surface_top_z,
    calibration_mask,
    contact_penetration_m,
    derive_contact_signals,
    sole_sphere_lowest_point_z,
)


def _signals(contact, loaded, xyz, *, fall=None, transient=1):
    velocity = np.zeros_like(xyz, dtype=float)
    penetration = np.linspace(0.001, 0.002, len(contact))
    return derive_contact_signals(
        np.asarray(contact, bool),
        np.asarray(loaded, bool),
        np.asarray(xyz, float),
        velocity,
        penetration,
        fall,
        touchdown_transient_samples=transient,
    )


def test_contact_episode_start_resets_anchor():
    contact = [0, 1, 1, 0, 1, 1]
    xyz = np.asarray(
        [[50, 0, 0], [10, 2, 0], [11, 2, 0], [70, 0, 0], [100, 4, 0], [102, 4, 0]],
        float,
    )
    signals = _signals(contact, contact, xyz)
    assert signals.contact_episode_id.tolist() == [-1, 0, 0, -1, 1, 1]
    assert np.allclose(signals.tangential_anchor_drift_m[[1, 2, 4, 5]], [0, 1, 0, 2])
    assert np.allclose(signals.anchor_xy_m[1], [10, 2])
    assert np.allclose(signals.anchor_xy_m[4], [100, 4])


def test_air_is_invalid_and_never_supplies_physical_hazard_evidence():
    contact = np.asarray([0, 1, 1, 0], bool)
    xyz = np.zeros((4, 3))
    signals = _signals(contact, contact, xyz, transient=0)
    air = ~contact
    assert np.all(signals.contact_episode_id[air] == -1)
    assert np.all(np.isnan(signals.tangential_anchor_drift_m[air]))
    assert not np.any(signals.slip_calibration_valid[air])
    assert not np.any(signals.sink_calibration_valid[air])


def test_pre_touchdown_motion_is_not_included_in_anchor_drift():
    contact = [0, 0, 1, 1]
    xyz = np.asarray([[0, 0, 0], [9, 0, 0], [20, 0, 0], [20.1, 0, 0]])
    signals = _signals(contact, contact, xyz, transient=0)
    assert np.isnan(signals.tangential_anchor_drift_m[1])
    assert signals.tangential_anchor_drift_m[2] == 0.0
    assert np.isclose(signals.tangential_anchor_drift_m[3], 0.1)


def test_first_fall_sample_and_later_samples_are_censored():
    assert calibration_mask(6, 3).tolist() == [True, True, True, False, False, False]
    contact = np.ones(6, bool)
    xyz = np.zeros((6, 3))
    signals = _signals(contact, contact, xyz, fall=3, transient=0)
    assert signals.pre_fall_valid.tolist() == [True, True, True, False, False, False]
    assert not np.any(signals.slip_calibration_valid[3:])
    assert not np.any(signals.sink_calibration_valid[3:])


def test_world_frame_offset_does_not_change_anchor_relative_drift():
    contact = np.asarray([0, 1, 1, 1, 0], bool)
    xyz = np.zeros((5, 3))
    xyz[1:4, :2] = [[0, 0], [0.03, 0.04], [0.06, 0.08]]
    shifted = xyz + np.asarray([123.0, -77.0, 8.0])
    base = _signals(contact, contact, xyz, transient=0)
    moved = _signals(contact, contact, shifted, transient=0)
    assert np.allclose(
        base.tangential_anchor_drift_m,
        moved.tangential_anchor_drift_m,
        equal_nan=True,
    )


def test_mujoco_contact_distance_sign_convention():
    assert contact_penetration_m(-0.012) == 0.012
    assert contact_penetration_m(0.0) == 0.0
    assert contact_penetration_m(0.004) == 0.0


def test_box_top_and_sole_sphere_lowest_point_use_world_geometry():
    xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="box" pos="0 0 0.3" size="1 1 0.2"/>
        <geom name="sole_a" type="sphere" pos="0 0 1.0" size="0.1"/>
        <geom name="sole_b" type="sphere" pos="0 0 0.8" size="0.05"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.isclose(box_surface_top_z(model, data, model.geom("ground").id), 0.5)
    assert np.isclose(
        sole_sphere_lowest_point_z(
            model, data, (model.geom("sole_a").id, model.geom("sole_b").id)
        ),
        0.75,
    )


def test_checkpoint_frozen_runners_and_outputs_are_byte_unchanged():
    root = Path(__file__).resolve().parents[4]
    expected = {
        "simulation/unitree_mujoco/simulate_python/run_walking_domain_failure_audit_v1.py": "abdb3b5babf8f90a73773940b25f69392c087bbfd2ccc170d94b479b2b6d8266",
        "simulation/unitree_mujoco/simulate_python/run_walking_terrain_transition_v1.py": "646805bdb9bed4adf302319275e5209a3d07a0c2fb7e0a215f9943524fff108f",
        "simulation/unitree_mujoco/simulate_python/run_terrain_transition.py": "b89079b2927245f2288cd88873879cd90fb8ff91bf1c4cc145c1e8bdea140aa2",
        "simulation/outputs/walking_domain_failure_audit_v1/summary.json": "46b988d14034e55272528d1472904b40619bc479d59949ee79ad623508772649",
        "simulation/outputs/walking_terrain_transition_v1_pilot/summary.json": "b05300449415923862ccaf69ffb4ebcd09e3f6848d5d12d256aa3b1e08a3ebd9",
        "simulation/outputs/walking_oracle_label_compatibility_audit_v1/summary.json": "82f5caa872887175d31daca779ad9487be16e21dc088ef9c562111c719b89e14",
    }
    actual = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in expected
    }
    assert actual == expected
