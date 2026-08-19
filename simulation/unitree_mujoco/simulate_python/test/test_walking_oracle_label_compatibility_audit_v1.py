import numpy as np

from run_walking_oracle_label_compatibility_audit_v1 import (
    contact_episodes,
    episode_rows,
    gait_phase,
)


def test_contact_episodes_are_half_open_and_filter_chatter():
    mask = np.asarray([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert contact_episodes(mask, minimum_samples=3) == [(4, 7)]


def test_gait_phase_keeps_air_negative_and_assigns_loaded_edges():
    force = np.r_[np.zeros(2), np.full(50, 10.0), np.zeros(2)]
    phase = gait_phase(force)
    assert np.all(phase[:2] == "AIR")
    assert np.all(phase[2:12] == "TOUCHDOWN")
    assert np.all(phase[12:32] == "LOADING")
    assert np.all(phase[32:42] == "MID_STANCE")
    assert np.all(phase[42:52] == "PUSH_OFF")
    assert np.all(phase[52:] == "AIR")


def test_episode_metrics_use_contact_anchor_not_trace_origin():
    n = 20
    xyz = np.zeros((n, 3), float)
    xyz[:, 0] = 4.0  # arbitrary world-frame offset must cancel
    xyz[5:15, 0] += np.linspace(0.0, 0.09, 10)
    velocity = np.zeros_like(xyz)
    force = np.zeros(n)
    force[5:15] = 10.0
    contact = force > 0
    rows = episode_rows("r", "d", "role", "ice", force, contact, xyz, velocity,
                        np.zeros(n, bool), np.zeros(n, bool))
    assert len(rows) == 1
    assert np.isclose(rows[0]["max_anchor_drift_m"], 0.09)


def test_first_allowed_sample_censors_pre_transition_episode_portion():
    n = 20
    xyz = np.zeros((n, 3), float)
    xyz[:, 0] = np.arange(n) * 0.01
    force = np.full(n, 10.0)
    rows = episode_rows("r", "d", "role", "A", force, np.ones(n, bool), xyz,
                        np.zeros_like(xyz), np.zeros(n, bool), np.zeros(n, bool),
                        first_allowed_sample=10)
    assert rows[0]["start_sample"] == 10
    assert rows[0]["duration_ms"] == 10
    assert np.isclose(rows[0]["max_anchor_drift_m"], 0.09)
