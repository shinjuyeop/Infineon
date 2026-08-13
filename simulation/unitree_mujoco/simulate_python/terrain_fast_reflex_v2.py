"""Fast Reflex v2 schema, train-normal calibration, and causal oracle labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCHEMA_NAME = "terrain_fast_reflex_v2"
SCHEMA_VERSION = 2
PHYSICS_TIMESTEP_S = .0005
SENSOR_RATE_HZ = 1000
PHYSICS_STEPS_PER_SAMPLE = 2
TRACE_PRE_MS = 50
TRACE_POST_MS = 100
TRACE_SAMPLES = TRACE_PRE_MS + TRACE_POST_MS
RELATIVE_TRANSITION_TIME_MS = np.arange(-TRACE_PRE_MS, TRACE_POST_MS, dtype=np.int16)
MODES = ("normal_sand", "sink_dominant", "tilt_dominant", "boundary_front_rear",
         "boundary_left_right", "sink_and_tilt")
FINAL_TEST_SPLIT = "final_test"

# Kept explicitly local so schema-only tests do not need MuJoCo; names/units
# remain byte-for-byte aligned with the preserved v1 oracle contract.
HAZARD_CONFIRMATION_SAMPLES = 3
ORACLE_CHANNELS = (
    "left_contact", "contact_normal_force_N", "contact_tangential_force_N", "Ft_over_Fn",
    "foot_velocity_x_mps", "foot_velocity_y_mps", "foot_velocity_z_mps",
    "foot_angular_velocity_x_rad_s", "foot_angular_velocity_y_rad_s", "foot_angular_velocity_z_rad_s",
    "foot_roll_rad", "foot_pitch_rad", "foot_z_m", "foot_horizontal_speed_mps",
    "foot_sink_depth_m", "foot_tilt_change_rad",
)

V2_ORACLE_CHANNELS = (*ORACLE_CHANNELS, "ground_a_contact", "ground_b_contact")
V2_ORACLE_INDEX = {name: index for index, name in enumerate(V2_ORACLE_CHANNELS)}


def validate_final_test_request(families: list[str], include_final_test: bool) -> None:
    """Fail closed: old v1 test families can never materialize as v2 final data."""
    forbidden = {"warped_multisine", "smooth_random_patches"}.intersection(families)
    if forbidden and not include_final_test:
        raise ValueError("final-test family requested without --include-final-test")
    if forbidden:
        raise ValueError("v1 test families are permanently ineligible for v2 final test")


@dataclass(frozen=True)
class V2Calibration:
    minimum_load_N: float
    risk_speed_mps: float
    risk_speed_trend_mps_per_ms: float
    confirmed_speed_mps: float
    sink_depth_m: float
    downward_speed_mps: float
    tilt_change_rad: float
    angular_rate_rad_s: float
    persistence_samples: int = HAZARD_CONFIRMATION_SAMPLES
    provenance: str = "train-normal only; robust percentile/MAD statistics"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def robust_upper(values: np.ndarray, percentile: float, mad_multiplier: float) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite calibration values")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return float(max(np.percentile(values, percentile), center + mad_multiplier * mad))


def _sustained(mask: np.ndarray, samples: int) -> np.ndarray:
    if samples < 1:
        raise ValueError("persistence must be positive")
    active = np.asarray(mask, bool)
    hit = np.convolve(active.astype(np.int8), np.ones(samples, dtype=np.int8), "valid") == samples
    result = np.zeros_like(active)
    if hit.any():
        # State begins at the first sample of a causal confirmation sequence.
        for endpoint in np.flatnonzero(hit) + samples - 1:
            result[endpoint - samples + 1:endpoint + 1] = True
    return result


def calibrate_v2(train_normal_oracle: list[np.ndarray]) -> V2Calibration:
    """Derive every threshold from fresh v2 train normal traces only."""
    if not train_normal_oracle:
        raise ValueError("at least one train normal trace is required")
    oracle = np.concatenate([item[TRACE_PRE_MS:] for item in train_normal_oracle])
    contact = oracle[:, V2_ORACLE_INDEX["left_contact"]] > .5
    fn = oracle[:, V2_ORACLE_INDEX["contact_normal_force_N"]]
    loaded = contact & (fn >= np.percentile(fn[contact], 10))
    if loaded.sum() < HAZARD_CONFIRMATION_SAMPLES:
        raise ValueError("insufficient loaded normal contact for calibration")
    speed = oracle[:, V2_ORACLE_INDEX["foot_horizontal_speed_mps"]]
    trend = np.maximum(0.0, np.diff(speed, prepend=speed[0]))
    angular = np.linalg.norm(oracle[:, [V2_ORACLE_INDEX["foot_angular_velocity_x_rad_s"],
                                       V2_ORACLE_INDEX["foot_angular_velocity_y_rad_s"]]], axis=1)
    risk = robust_upper(speed[loaded], 95.0, 4.0)
    confirmed = robust_upper(speed[loaded], 99.0, 6.0)
    # Preserve ordered states even when the normal distribution is degenerate.
    risk = min(risk, confirmed * .95)
    return V2Calibration(
        minimum_load_N=float(np.percentile(fn[contact], 10)),
        risk_speed_mps=float(risk),
        risk_speed_trend_mps_per_ms=robust_upper(trend[loaded], 95.0, 4.0),
        confirmed_speed_mps=float(confirmed),
        sink_depth_m=robust_upper(oracle[loaded, V2_ORACLE_INDEX["foot_sink_depth_m"]], 99.0, 6.0),
        downward_speed_mps=robust_upper(np.maximum(0.0, -oracle[loaded, V2_ORACLE_INDEX["foot_velocity_z_mps"]]), 99.0, 6.0),
        tilt_change_rad=robust_upper(oracle[loaded, V2_ORACLE_INDEX["foot_tilt_change_rad"]], 99.0, 6.0),
        angular_rate_rad_s=robust_upper(angular[loaded], 99.0, 6.0),
    )


def label_v2(oracle: np.ndarray, calibration: V2Calibration) -> dict[str, np.ndarray]:
    """Create SAFE/INCIPIENT_RISK/CONFIRMED plus sustained Sand physical labels."""
    if oracle.shape != (TRACE_SAMPLES, len(V2_ORACLE_CHANNELS)):
        raise ValueError(f"unexpected v2 oracle shape {oracle.shape}")
    contact = oracle[:, V2_ORACLE_INDEX["left_contact"]] > .5
    fn = oracle[:, V2_ORACLE_INDEX["contact_normal_force_N"]]
    loaded = contact & (fn >= calibration.minimum_load_N)
    speed = oracle[:, V2_ORACLE_INDEX["foot_horizontal_speed_mps"]]
    trend = np.maximum(0.0, np.diff(speed, prepend=speed[0]))
    confirmed = _sustained(loaded & (speed > calibration.confirmed_speed_mps), calibration.persistence_samples)
    risk_raw = loaded & (speed > calibration.risk_speed_mps) & (trend > calibration.risk_speed_trend_mps_per_ms)
    risk = _sustained(risk_raw, calibration.persistence_samples) | confirmed
    incipient = risk & ~confirmed
    angular = np.linalg.norm(oracle[:, [V2_ORACLE_INDEX["foot_angular_velocity_x_rad_s"],
                                        V2_ORACLE_INDEX["foot_angular_velocity_y_rad_s"]]], axis=1)
    sink = _sustained(
        loaded & (oracle[:, V2_ORACLE_INDEX["foot_sink_depth_m"]] > calibration.sink_depth_m)
        & (np.maximum(0.0, -oracle[:, V2_ORACLE_INDEX["foot_velocity_z_mps"]]) > calibration.downward_speed_mps),
        calibration.persistence_samples,
    )
    tilt = _sustained(
        loaded & (oracle[:, V2_ORACLE_INDEX["foot_tilt_change_rad"]] > calibration.tilt_change_rad)
        & (angular > calibration.angular_rate_rad_s), calibration.persistence_samples,
    )
    safe = ~(risk | sink | tilt)
    return {"safe": safe, "incipient_risk": incipient, "confirmed_slip": confirmed,
            "slip_risk": risk, "sustained_sink": sink, "sustained_tilt": tilt}


def onset_ms(mask: np.ndarray) -> int | None:
    found = np.flatnonzero(np.asarray(mask)[TRACE_PRE_MS:])
    return None if not len(found) else int(found[0])


def validate_state_order(labels: dict[str, np.ndarray]) -> None:
    confirmed, risk = labels["confirmed_slip"], labels["slip_risk"]
    if np.any(confirmed & ~risk):
        raise ValueError("confirmed slip must be included in slip risk")
    risk_onset, confirmed_onset = onset_ms(risk), onset_ms(confirmed)
    if risk_onset is not None and confirmed_onset is not None and risk_onset > confirmed_onset:
        raise ValueError("risk cannot begin after confirmed slip")
