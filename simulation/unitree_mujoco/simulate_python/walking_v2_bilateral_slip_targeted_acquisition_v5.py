"""Frozen contracts for targeted bilateral Slip development acquisition v5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from walking_v2_bilateral_slip_targeted_acquisition_v3 import (
    material_profiles as v3_material_profiles,
)
from walking_v2_slip_scenario_generator_redesign_v4 import (
    friction_profiles as v4_friction_profiles,
    patch_bounds,
)


SAMPLE_RATE_HZ = 1000
PHYSICS_TIMESTEP_S = 0.0005
PHYSICS_STEPS_PER_SAMPLE = 2
SPEEDS_MPS = (0.10, 0.15, 0.20)
SIDES = ("left", "right")
SEVERITIES = ("native_strong_ice", "moderate_ice_preregistered")
CONTROL_TYPES = ("hard_normal", "near_slip_non_event")
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_MS = 3


@dataclass(frozen=True)
class PhaseBin:
    name: str
    minimum_onset_ms: int
    maximum_onset_ms: int
    friction_activation_ms: int


PHASE_BINS = (
    PhaseBin("early_loading", 11, 120, 0),
    PhaseBin("mid_loading_early_stance", 121, 260, 70),
    PhaseBin("mid_late_stance", 261, 600, 190),
)


@dataclass(frozen=True)
class Variation:
    index: int
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float


VARIATIONS = (
    Variation(0, 0.07, 0.000, -0.004),
    Variation(1, 0.41, 0.040, 0.000),
    Variation(2, 0.83, 0.080, 0.004),
)


@dataclass(frozen=True)
class MaterialProfile:
    name: str
    friction3: tuple[float, float, float]
    role: str
    frozen_source: str

    @property
    def friction5(self) -> tuple[float, float, float, float, float]:
        sliding, torsional, rolling = self.friction3
        return sliding, sliding, torsional, rolling, rolling

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def material_profiles() -> dict[str, MaterialProfile]:
    """Join the immutable v4 positives/hard profile with frozen v3 controls."""
    v4 = v4_friction_profiles()
    v3 = v3_material_profiles()
    if tuple(v4["native_strong_ice"].friction3) != tuple(v3["native_strong_ice"].friction):
        raise ValueError("strong profile drift between frozen v3 and v4")
    if tuple(v4["moderate_ice_preregistered"].friction3) != tuple(
        v3["moderate_ice_preregistered"].friction
    ):
        raise ValueError("moderate profile drift between frozen v3 and v4")
    return {
        "native_strong_ice": MaterialProfile(
            "native_strong_ice", tuple(v4["native_strong_ice"].friction3),
            "positive", "v4 native strong Ice"),
        "moderate_ice_preregistered": MaterialProfile(
            "moderate_ice_preregistered",
            tuple(v4["moderate_ice_preregistered"].friction3),
            "positive", "v4 frozen moderate Ice"),
        "hard_normal": MaterialProfile(
            "hard_normal", tuple(v4["hard_control"].friction3),
            "control", "v4 native Concrete hard control"),
        "near_slip_non_event": MaterialProfile(
            "near_slip_non_event", tuple(v3["near_slip_non_event"].friction),
            "control", "v3 preregistered near-slip non-event"),
    }


@dataclass(frozen=True)
class RunCondition:
    run_id: str
    pair_id: str
    pair_index: int
    role: str
    speed_mps: float
    target_foot: str
    target_phase: str
    severity: str
    material_profile: str
    control_type: str
    variation_index: int
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float
    seed: int
    duration_s: float

    @property
    def pair_configuration(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "speed_mps": self.speed_mps,
            "target_foot": self.target_foot,
            "target_phase": self.target_phase,
            "severity": self.severity,
            "variation_index": self.variation_index,
            "phase_fraction": self.phase_fraction,
            "command_delay_s": self.command_delay_s,
            "lateral_offset_m": self.lateral_offset_m,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "patch": asdict(patch_bounds(self.target_foot)),
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "censoring": "frozen first-fall physical-oracle contract",
        }

    @property
    def pair_fingerprint(self) -> str:
        payload = json.dumps(
            self.pair_configuration, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _control_for(
    severity_index: int, foot_index: int, phase_index: int,
    variation_index: int,
) -> str:
    parity = severity_index + foot_index + phase_index + variation_index
    return CONTROL_TYPES[parity % 2]


def acquisition_matrix(duration_s: float = 3.0) -> list[RunCondition]:
    """Return the frozen 108-positive plus 108-control development matrix."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    rows: list[RunCondition] = []
    pair_index = 0
    for speed in SPEEDS_MPS:
        for foot_index, foot in enumerate(SIDES):
            for phase_index, phase in enumerate(PHASE_BINS):
                for severity_index, severity in enumerate(SEVERITIES):
                    for variation in VARIATIONS:
                        pair_id = f"v5_pair_{pair_index:03d}"
                        tag = (
                            f"{speed:.2f}_{foot}_{phase.name}_{severity}_v{variation.index}"
                        ).replace(".", "p")
                        control_type = _control_for(
                            severity_index, foot_index, phase_index,
                            variation.index)
                        common = dict(
                            pair_id=pair_id, pair_index=pair_index,
                            speed_mps=speed, target_foot=foot,
                            target_phase=phase.name, severity=severity,
                            control_type=control_type,
                            variation_index=variation.index,
                            phase_fraction=variation.phase_fraction,
                            command_delay_s=variation.command_delay_s,
                            lateral_offset_m=variation.lateral_offset_m,
                            seed=202608500 + pair_index, duration_s=duration_s,
                        )
                        rows.append(RunCondition(
                            run_id=f"positive_{tag}", role="positive",
                            material_profile=severity, **common))
                        rows.append(RunCondition(
                            run_id=f"control_{tag}", role="control",
                            material_profile=control_type, **common))
                        pair_index += 1
    validate_matrix(rows)
    return rows


def validate_matrix(rows: Iterable[RunCondition]) -> None:
    values = list(rows)
    if len(values) != 216 or len({row.run_id for row in values}) != 216:
        raise ValueError("matrix requires exactly 216 unique runs")
    grouped: dict[str, list[RunCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != 108:
        raise ValueError("matrix requires exactly 108 pair IDs")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"invalid pair roles: {pair_id}")
        if len({row.pair_fingerprint for row in pair}) != 1:
            raise ValueError(f"pair configuration mismatch: {pair_id}")
        if len({row.material_profile for row in pair}) != 2:
            raise ValueError(f"pair friction intervention missing: {pair_id}")
    positives = [row for row in values if row.role == "positive"]
    cells = {
        (row.speed_mps, row.target_foot, row.target_phase, row.severity)
        for row in positives
    }
    if len(positives) != 108 or len(cells) != 36:
        raise ValueError("positive factorial coverage is incomplete")
    if any(sum(
        (row.speed_mps, row.target_foot, row.target_phase, row.severity) == cell
        for row in positives) != 3 for cell in cells):
        raise ValueError("every positive cell requires three variations")
    for speed in SPEEDS_MPS:
        for foot in SIDES:
            for phase in PHASE_BINS:
                selected = [
                    row for row in values
                    if row.role == "control" and row.speed_mps == speed
                    and row.target_foot == foot and row.target_phase == phase.name
                ]
                if {row.control_type for row in selected} != set(CONTROL_TYPES):
                    raise ValueError("control types are not balanced")


def phase_spec(name: str) -> PhaseBin:
    try:
        return next(value for value in PHASE_BINS if value.name == name)
    except StopIteration as exc:
        raise ValueError(name) from exc


def onset_phase(age_ms: int) -> str:
    for phase in PHASE_BINS:
        if phase.minimum_onset_ms <= age_ms <= phase.maximum_onset_ms:
            return phase.name
    return "out_of_contract"


def deterministic_initial_perturbation(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform((-0.002, -0.001, -0.002), (0.002, 0.001, 0.002)))


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()

def trace_sha256(trace: dict[str, np.ndarray], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(trace[key])
        digest.update(key.encode() + b"\0")
        digest.update(value.dtype.str.encode() + b"\0")
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()
