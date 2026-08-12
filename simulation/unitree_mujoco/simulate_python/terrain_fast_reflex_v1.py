"""Schema, oracle diagnostics, hazard labels, and windows for Fast Reflex v1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hil_sensor import HIL_SENSOR_CHANNELS


SCHEMA_NAME = "terrain_fast_reflex_v1"
SCHEMA_VERSION = 1
PHYSICS_TIMESTEP_S = 0.0005
SENSOR_RATE_HZ = 1000
PHYSICS_STEPS_PER_SAMPLE = 2
TRACE_PRE_MS = 50
TRACE_POST_MS = 100
TRACE_SAMPLES = TRACE_PRE_MS + TRACE_POST_MS
RELATIVE_TRANSITION_TIME_MS = np.arange(-TRACE_PRE_MS, TRACE_POST_MS, dtype=np.int16)
OBSERVATION_WINDOWS_MS = (1, 2, 5, 10, 15, 20, 30, 50)
PRIMARY_WINDOWS_MS = (5, 10, 15, 20, 30, 50)
HAZARD_CONFIRMATION_SAMPLES = 3

ORACLE_CHANNELS = (
    "left_contact",
    "contact_normal_force_N",
    "contact_tangential_force_N",
    "Ft_over_Fn",
    "foot_velocity_x_mps",
    "foot_velocity_y_mps",
    "foot_velocity_z_mps",
    "foot_angular_velocity_x_rad_s",
    "foot_angular_velocity_y_rad_s",
    "foot_angular_velocity_z_rad_s",
    "foot_roll_rad",
    "foot_pitch_rad",
    "foot_z_m",
    "foot_horizontal_speed_mps",
    "foot_sink_depth_m",
    "foot_tilt_change_rad",
)
ORACLE_INDEX = {name: index for index, name in enumerate(ORACLE_CHANNELS)}


@dataclass(frozen=True)
class HazardThresholds:
    """Thresholds calibrated only from train-family normal transition traces."""

    minimum_load_N: float
    slip_speed_mps: float
    sink_depth_m: float
    downward_speed_mps: float
    tilt_change_rad: float


@dataclass(frozen=True)
class FastReflexTrace:
    metadata: dict[str, Any]
    timestamps_s: np.ndarray
    sensors: np.ndarray
    oracle: np.ndarray
    slip: np.ndarray
    sink: np.ndarray
    tilt: np.ndarray
    valid: bool
    invalid_reason: str

    @property
    def sink_or_tilt(self) -> np.ndarray:
        return self.sink | self.tilt


def _robust_upper(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("normal calibration has no finite samples")
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    return float(max(np.percentile(finite, percentile), center + 6.0 * mad))


def calibrate_hazard_thresholds(normal_traces: list[FastReflexTrace]) -> HazardThresholds:
    """Derive oracle thresholds from train-side normal runs only.

    Calibration uses the post-transition 0..99 ms interval under the same pulse,
    so normal contact transients are part of the negative distribution.
    """
    if not normal_traces:
        raise ValueError("at least one normal calibration trace is required")
    oracle = np.concatenate([trace.oracle[TRACE_PRE_MS:] for trace in normal_traces])
    contact = oracle[:, ORACLE_INDEX["left_contact"]] > 0.5
    loaded_normal = oracle[contact, ORACLE_INDEX["contact_normal_force_N"]]
    if loaded_normal.size == 0:
        raise ValueError("normal calibration contains no loaded left-foot samples")
    minimum_load = float(np.percentile(loaded_normal, 10.0))
    loaded = contact & (
        oracle[:, ORACLE_INDEX["contact_normal_force_N"]] >= minimum_load
    )
    if np.count_nonzero(loaded) < HAZARD_CONFIRMATION_SAMPLES:
        raise ValueError("normal calibration has insufficient loaded samples")
    return HazardThresholds(
        minimum_load_N=minimum_load,
        slip_speed_mps=_robust_upper(
            oracle[loaded, ORACLE_INDEX["foot_horizontal_speed_mps"]]
        ),
        sink_depth_m=_robust_upper(
            oracle[loaded, ORACLE_INDEX["foot_sink_depth_m"]]
        ),
        downward_speed_mps=_robust_upper(
            np.maximum(0.0, -oracle[loaded, ORACLE_INDEX["foot_velocity_z_mps"]])
        ),
        tilt_change_rad=_robust_upper(
            oracle[loaded, ORACLE_INDEX["foot_tilt_change_rad"]]
        ),
    )


def _confirmed(mask: np.ndarray, confirmation_samples: int) -> np.ndarray:
    result = np.zeros(len(mask), dtype=bool)
    run = 0
    for index, active in enumerate(mask):
        run = run + 1 if active else 0
        if run >= confirmation_samples:
            result[index - confirmation_samples + 1 : index + 1] = True
    return result


def label_trace(
    trace: FastReflexTrace,
    thresholds: HazardThresholds,
    confirmation_samples: int = HAZARD_CONFIRMATION_SAMPLES,
) -> FastReflexTrace:
    oracle = trace.oracle
    contact = oracle[:, ORACLE_INDEX["left_contact"]] > 0.5
    loaded = contact & (
        oracle[:, ORACLE_INDEX["contact_normal_force_N"]]
        >= thresholds.minimum_load_N
    )
    post = RELATIVE_TRANSITION_TIME_MS >= 0
    slip = _confirmed(
        post
        & loaded
        & (
            oracle[:, ORACLE_INDEX["foot_horizontal_speed_mps"]]
            > thresholds.slip_speed_mps
        ),
        confirmation_samples,
    )
    sink = _confirmed(
        post
        & loaded
        & (oracle[:, ORACLE_INDEX["foot_sink_depth_m"]] > thresholds.sink_depth_m)
        & (
            -oracle[:, ORACLE_INDEX["foot_velocity_z_mps"]]
            > thresholds.downward_speed_mps
        ),
        confirmation_samples,
    )
    tilt = _confirmed(
        post
        & loaded
        & (oracle[:, ORACLE_INDEX["foot_tilt_change_rad"]] > thresholds.tilt_change_rad),
        confirmation_samples,
    )
    return replace(trace, slip=slip, sink=sink, tilt=tilt)


def onset_time_s(trace: FastReflexTrace, label: np.ndarray) -> float | None:
    indices = np.flatnonzero(label)
    return None if indices.size == 0 else float(trace.timestamps_s[indices[0]])


def extract_prefix(
    trace: FastReflexTrace,
    window_ms: int,
    alignment: str = "transition",
    hazard_label: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if window_ms not in OBSERVATION_WINDOWS_MS:
        raise ValueError(f"unsupported observation window {window_ms} ms")
    if alignment == "transition":
        anchor = float(trace.metadata["transition_time_s"])
    elif alignment == "hazard":
        if hazard_label is None:
            raise ValueError("hazard alignment requires a hazard label")
        onset = onset_time_s(trace, hazard_label)
        if onset is None:
            raise ValueError("hazard-aligned prefix requested without an onset")
        anchor = onset
    else:
        raise ValueError(f"unknown alignment {alignment!r}")
    indices = np.flatnonzero(
        (trace.timestamps_s >= anchor - 1e-9)
        & (trace.timestamps_s < anchor + window_ms / 1000.0 - 1e-9)
    )
    if len(indices) != window_ms or (
        len(indices) > 1 and not np.all(np.diff(indices) == 1)
    ):
        raise ValueError(
            f"{alignment}-aligned {window_ms} ms prefix selected {len(indices)} samples"
        )
    return trace.timestamps_s[indices].copy(), trace.sensors[indices].copy()


def validate_trace(trace: FastReflexTrace) -> None:
    if trace.timestamps_s.shape != (TRACE_SAMPLES,):
        raise ValueError(f"unexpected timestamp shape {trace.timestamps_s.shape}")
    if trace.sensors.shape != (TRACE_SAMPLES, len(HIL_SENSOR_CHANNELS)):
        raise ValueError(f"unexpected Fusion10 shape {trace.sensors.shape}")
    if trace.oracle.shape != (TRACE_SAMPLES, len(ORACLE_CHANNELS)):
        raise ValueError(f"unexpected oracle shape {trace.oracle.shape}")
    if not np.all(np.isfinite(trace.sensors)) or not np.all(np.isfinite(trace.oracle)):
        raise ValueError("NaN/Inf in transition trace")
    if not np.allclose(np.diff(trace.timestamps_s), 0.001, rtol=0.0, atol=1e-9):
        raise ValueError("transition trace is not native 1 kHz")
    relative = (trace.timestamps_s - float(trace.metadata["transition_time_s"])) * 1000.0
    if not np.allclose(relative, RELATIVE_TRANSITION_TIME_MS, rtol=0.0, atol=1e-6):
        raise ValueError("transition trace alignment is not [-50,+100) ms")


def validate_split_integrity(rows: list[dict[str, Any]]) -> None:
    for key in ("surface_family", "surface_seed", "session_id", "run_id"):
        owners: dict[str, str] = {}
        for row in rows:
            value, split = str(row[key]), str(row["split"])
            previous = owners.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{key} leaks across {previous} and {split}")
