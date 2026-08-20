"""Pure causal features and audit helpers for bilateral walking v2.

Runtime feature builders in this module accept only virtual sensor or
controller-telemetry arrays.  Physical-oracle arrays are accepted exclusively
by the explicitly named offline label helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


SAMPLE_RATE_HZ = 1000
WINDOW_SAMPLES = 200
ENDPOINT_STRIDE = 10
SINK_RISK_HORIZON_SAMPLES = 200
SLIP_RISK_HORIZON_SAMPLES = 100
CANDIDATE_NAMES = ("C0", "C1", "C2", "C3", "C4")
RUNTIME_ARRAY_NAMES = frozenset({
    "bilateral_canonical", "pelvis_imu", "joint_position", "joint_velocity",
    "target_position", "actuator_effort", "force_loaded", "contact_age",
})
SIMULATOR_ONLY_ARRAY_NAMES = frozenset({
    "foot_world_xyz", "foot_world_velocity", "pelvis_world_xyz",
    "pelvis_world_velocity", "terrain_id", "profile_id", "run_id",
    "contact_penetration", "anchor_drift", "physical_label", "fall_flag",
})
LEG_LINK_LENGTHS_M = (0.30001, 0.30001)


@dataclass(frozen=True)
class SharedFootEncoderV2:
    """Fixed 80-stat causal foot encoder applied by exact object sharing."""

    window_samples: int = WINDOW_SAMPLES

    @property
    def output_size(self) -> int:
        return 80

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def macs_per_endpoint(self) -> int:
        # Conservative sums/squares/differences estimate for 10 channels.
        return self.window_samples * 10 * 4

    @property
    def fingerprint(self) -> str:
        definition = (
            f"shared-foot-stat-v2:{self.window_samples}:"
            "mean,std,min,max,last,delta,mean_abs_diff,rms"
        )
        return hashlib.sha256(definition.encode("utf-8")).hexdigest()

    def encode(self, foot_window: np.ndarray) -> np.ndarray:
        values = np.asarray(foot_window, dtype=np.float64)
        if values.shape != (self.window_samples, 10):
            raise ValueError(
                f"shared encoder expects ({self.window_samples},10), got {values.shape}"
            )
        difference = np.diff(values, axis=0)
        features = np.stack((
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
            values[-1],
            values[-1] - values[0],
            np.mean(np.abs(difference), axis=0),
            np.sqrt(np.mean(np.square(values), axis=0)),
        ), axis=1)
        output = features.reshape(-1).astype(np.float32)
        if output.shape != (self.output_size,) or not np.all(np.isfinite(output)):
            raise ValueError("invalid shared foot encoding")
        return output


@dataclass(frozen=True)
class SharedCausalConv1DV2:
    """Small fixed depthwise causal Conv1D probe with exact foot sharing.

    Four predeclared filters (5/20/50-sample averages and a one-step
    difference) are applied independently to every Fusion10 channel.  A
    trained small head may follow these 40 current-endpoint convolution values.
    """

    window_samples: int = WINDOW_SAMPLES

    @property
    def output_size(self) -> int:
        return 40

    @property
    def parameter_count(self) -> int:
        return 10 * (5 + 20 + 50 + 2)

    @property
    def macs_per_endpoint(self) -> int:
        return self.parameter_count

    @property
    def fingerprint(self) -> str:
        definition = "shared-depthwise-causal-conv1d-v2:mean5,mean20,mean50,diff1"
        return hashlib.sha256(definition.encode("utf-8")).hexdigest()

    def encode(self, foot_window: np.ndarray) -> np.ndarray:
        values = np.asarray(foot_window, dtype=np.float64)
        if values.shape != (self.window_samples, 10):
            raise ValueError(
                f"causal Conv1D expects ({self.window_samples},10), got {values.shape}"
            )
        output = np.stack((
            values[-5:].mean(axis=0),
            values[-20:].mean(axis=0),
            values[-50:].mean(axis=0),
            values[-1] - values[-2],
        ), axis=1).reshape(-1).astype(np.float32)
        if output.shape != (self.output_size,) or not np.all(np.isfinite(output)):
            raise ValueError("invalid shared causal Conv1D encoding")
        return output


def causal_endpoints(sample_count: int) -> np.ndarray:
    """Return fixed 10-ms endpoints that all have a 200-ms past window."""
    if sample_count < WINDOW_SAMPLES:
        return np.empty(0, dtype=np.int64)
    return np.arange(WINDOW_SAMPLES - 1, sample_count, ENDPOINT_STRIDE, dtype=np.int64)


def contact_age(loaded: np.ndarray) -> np.ndarray:
    """Causal number of consecutive force-loaded samples, reset in AIR."""
    values = np.asarray(loaded, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("loaded state must have shape (samples,2)")
    age = np.zeros(values.shape, dtype=np.int32)
    for index in range(len(values)):
        if index == 0:
            age[index] = values[index].astype(np.int32)
        else:
            age[index] = np.where(values[index], age[index - 1] + 1, 0)
    return age


def joint_derived_kinematics(joint_position: np.ndarray) -> np.ndarray:
    """Return causal encoder-only leg length and equivalent vertical motion.

    Input order is the controller's six left leg joints followed by six right
    leg joints.  No MuJoCo body/world position is read.  The finite difference
    is backward-only and therefore deployable from encoder telemetry.
    """
    position = np.asarray(joint_position, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 12:
        raise ValueError("joint position must have shape (samples,12)")
    upper, lower = LEG_LINK_LENGTHS_M
    knees = position[:, (3, 9)]
    length = np.sqrt(np.maximum(
        0.0, upper * upper + lower * lower + 2.0 * upper * lower * np.cos(knees)
    ))
    velocity = np.zeros_like(length)
    velocity[1:] = (length[1:] - length[:-1]) * SAMPLE_RATE_HZ
    relative_vertical = -velocity
    return np.column_stack((length, velocity, relative_vertical)).reshape(len(position), 6)


def effort_summaries(
    joint_position: np.ndarray,
    target_position: np.ndarray,
    actuator_effort: np.ndarray,
) -> np.ndarray:
    """Summarize commanded/measured residual and telemetry effort per leg."""
    measured = np.asarray(joint_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    effort = np.asarray(actuator_effort, dtype=np.float64)
    if measured.shape != target.shape or measured.shape != effort.shape:
        raise ValueError("joint, target, and effort arrays must align")
    if measured.ndim != 2 or measured.shape[1] != 12:
        raise ValueError("effort inputs must have shape (samples,12)")
    residual = target - measured
    columns: list[np.ndarray] = []
    for side_slice in (slice(0, 6), slice(6, 12)):
        side_residual = np.abs(residual[:, side_slice])
        side_effort = np.abs(effort[:, side_slice])
        columns.extend((
            side_residual.mean(axis=1), side_residual.max(axis=1),
            np.sqrt(np.mean(np.square(side_residual), axis=1)),
            side_effort.mean(axis=1), side_effort.max(axis=1),
            np.sqrt(np.mean(np.square(side_effort), axis=1)),
        ))
    return np.column_stack(columns)


def _summary(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array):
        raise ValueError("summary requires a non-empty two-dimensional window")
    return np.concatenate((array[-1], array.mean(axis=0), array.std(axis=0))).astype(np.float32)


def candidate_features(
    candidate: str,
    *,
    side: int,
    endpoint: int,
    bilateral_canonical: np.ndarray,
    pelvis_imu: np.ndarray,
    joint_position: np.ndarray,
    target_position: np.ndarray,
    actuator_effort: np.ndarray,
    force_loaded: np.ndarray,
    encoder: SharedFootEncoderV2,
) -> np.ndarray:
    """Build one predeclared C0--C4 feature vector using past/current data."""
    if candidate not in CANDIDATE_NAMES or side not in (0, 1):
        raise ValueError("invalid candidate or side")
    start = endpoint - encoder.window_samples + 1
    if start < 0:
        raise ValueError("endpoint has no complete causal history")
    feet = np.asarray(bilateral_canonical, dtype=np.float64)
    if feet.ndim != 2 or feet.shape[1] != 20:
        raise ValueError("bilateral canonical input must have shape (samples,20)")
    left = encoder.encode(feet[start:endpoint + 1, :10])
    right = encoder.encode(feet[start:endpoint + 1, 10:])
    if candidate == "C0":
        return left
    own, other = (left, right) if side == 0 else (right, left)
    load = feet[:, (0, 1, 2, 3, 10, 11, 12, 13)].reshape(len(feet), 2, 4).sum(axis=2)
    total = float(load[endpoint].sum())
    ratio = float(load[endpoint, side] / total) if total > 1e-9 else 0.5
    loaded = np.asarray(force_loaded, dtype=bool)
    if loaded.shape != (len(feet), 2):
        raise ValueError("force-loaded input must align with bilateral samples")
    base = np.concatenate((
        own, other,
        np.asarray((ratio, 1.0 - ratio, loaded[endpoint, side], loaded[endpoint, 1 - side]), np.float32),
    ))
    if candidate == "C1":
        return base
    pelvis = np.asarray(pelvis_imu, dtype=np.float64)
    if pelvis.shape != (len(feet), 6):
        raise ValueError("pelvis IMU must align and contain six channels")
    if candidate == "C2":
        return np.concatenate((base, _summary(pelvis[start:endpoint + 1])))
    kinematics = joint_derived_kinematics(joint_position)
    age = contact_age(loaded)
    # Orient side-paired values as own then other for a common fusion head.
    own_indices = (side, side + 2, side + 4)
    other_indices = (1 - side, 3 - side, 5 - side)
    oriented_kinematics = np.column_stack((
        kinematics[:, own_indices], kinematics[:, other_indices],
        age[:, side] / SAMPLE_RATE_HZ, age[:, 1 - side] / SAMPLE_RATE_HZ,
    ))
    c3 = np.concatenate((base, _summary(oriented_kinematics[start:endpoint + 1])))
    if candidate == "C3":
        return c3
    effort = effort_summaries(joint_position, target_position, actuator_effort)
    own_effort = effort[:, :6] if side == 0 else effort[:, 6:]
    other_effort = effort[:, 6:] if side == 0 else effort[:, :6]
    pelvis_vertical = pelvis[:, (2, 3, 4, 5)]
    return np.concatenate((
        c3,
        _summary(np.column_stack((own_effort, other_effort))[start:endpoint + 1]),
        _summary(pelvis_vertical[start:endpoint + 1]),
    ))


def physical_risk_target(
    physical_active: np.ndarray,
    valid: np.ndarray,
    episode_id: np.ndarray,
    horizon_samples: int,
) -> np.ndarray:
    """Offline-only target with a within-episode pre-onset horizon."""
    active = np.asarray(physical_active, dtype=bool)
    allowed = np.asarray(valid, dtype=bool)
    episode = np.asarray(episode_id, dtype=np.int64)
    if not (active.shape == allowed.shape == episode.shape) or horizon_samples < 0:
        raise ValueError("offline label inputs must align")
    result = active & allowed
    onset = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    for index in onset:
        start = max(0, int(index) - horizon_samples)
        selected = np.arange(start, int(index) + 1)
        same_episode = episode[selected] == int(episode[index])
        selected = selected[allowed[selected] & same_episode]
        result[selected] = True
    return result


def support_degradation_label(
    sink_active: np.ndarray,
    valid: np.ndarray,
    pelvis_z: np.ndarray,
    pelvis_vertical_velocity: np.ndarray,
    episode_id: np.ndarray,
) -> np.ndarray:
    """Label-only Sink consequence independent of the runtime feature path."""
    sink = np.asarray(sink_active, bool)
    allowed = np.asarray(valid, bool)
    z = np.asarray(pelvis_z, float)
    velocity = np.asarray(pelvis_vertical_velocity, float)
    episode = np.asarray(episode_id, int)
    if not (sink.shape == allowed.shape == z.shape == velocity.shape == episode.shape):
        raise ValueError("support label arrays must align")
    result = np.zeros(len(sink), dtype=bool)
    reference: dict[int, float] = {}
    persistence = 0
    previous = -1
    for index in range(len(sink)):
        current = int(episode[index])
        if current != previous:
            persistence = 0
        if allowed[index] and current >= 0 and current not in reference:
            reference[current] = float(z[index])
        drop = 0.0 if current not in reference else reference[current] - float(z[index])
        condition = bool(sink[index] and (drop >= 0.010 or velocity[index] <= -0.050))
        persistence = persistence + 1 if condition and allowed[index] else 0
        result[index] = persistence >= 20
        previous = current
    return result


def first_fall_mask(sample_count: int, first_fall_sample: int | None) -> np.ndarray:
    """Return true before, and false at/after, the first sampled fall."""
    result = np.ones(sample_count, dtype=bool)
    if first_fall_sample is not None:
        if not 0 <= first_fall_sample < sample_count:
            raise ValueError("first fall sample outside trace")
        result[first_fall_sample:] = False
    return result


def stable_fire(
    score: np.ndarray,
    threshold: float,
    eligible: np.ndarray,
    persistence_samples: int = 3,
) -> np.ndarray:
    """Causal persistent score with immediate reset outside eligible contact."""
    values = np.asarray(score, float)
    allowed = np.asarray(eligible, bool)
    if values.shape != allowed.shape or persistence_samples <= 0:
        raise ValueError("stateful score inputs must align")
    firing = np.zeros(len(values), dtype=bool)
    count = 0
    for index in range(len(values)):
        count = count + 1 if allowed[index] and values[index] >= threshold else 0
        firing[index] = count >= persistence_samples
    return firing


def zero_false_positive_threshold(negative_train_scores: np.ndarray) -> float:
    """Predeclared train-only threshold immediately above every negative."""
    values = np.asarray(negative_train_scores, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("zero-FP threshold needs train negatives")
    return float(np.nextafter(np.max(values), np.inf))


def endpoint_hash(*arrays: np.ndarray) -> str:
    """Stable duplicate audit hash over declared endpoint content."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def runtime_feature_contract_is_clean(feature_sources: set[str]) -> bool:
    """Reject any simulator-only or undeclared runtime source."""
    return bool(feature_sources) and feature_sources <= RUNTIME_ARRAY_NAMES and not (
        feature_sources & SIMULATOR_ONLY_ARRAY_NAMES
    )


def deterministic_candidate_selection(rows: list[dict[str, object]]) -> str:
    """Apply the locked candidate order without validation-driven additions."""
    if not rows:
        raise ValueError("selection requires candidates")
    def key(row: dict[str, object]) -> tuple[object, ...]:
        return (
            int(row["invalid_firings"]),
            int(row["normal_fp_runs"]),
            int(row["ice_cross_hazard_fp_runs"]),
            int(row["too_early_firings"]),
            -float(row["zero_fp_recall"]),
            -float(row["coverage_fraction"]),
            float(row["latency_ms"]),
            int(row["sensor_channels"]),
            int(row["memory_bytes"]),
            int(row["macs"]),
            CANDIDATE_NAMES.index(str(row["candidate"])),
        )
    return str(min(rows, key=key)["candidate"])
