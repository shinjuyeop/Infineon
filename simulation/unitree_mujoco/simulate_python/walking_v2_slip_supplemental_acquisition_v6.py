"""Pure contracts for the bounded Walking-v2 Slip supplement v6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ, SLIP_PERSISTENCE_MS, SLIP_THRESHOLD_M, SPEEDS_MPS,
    VARIATIONS as V5_VARIATIONS, material_profiles,
)
from walking_v2_slip_scenario_generator_redesign_v4 import PatchBounds


TARGET_FOOT = "left"
TARGET_PHASE = "mid_late_stance"
TARGET_SEVERITY = "moderate_ice_preregistered"
CALIBRATION_DISPOSITION = "SOURCE_CALIBRATION_ONLY_DO_NOT_TRAIN"
DEVELOPMENT_DISPOSITION = "DEVELOPMENT_ONLY"


@dataclass(frozen=True)
class GeometryCandidate:
    candidate_id: str
    upstream_shift_m: float
    length_multiplier: float
    width_m: float = 1.6
    top_height_m: float = 0.0

    @property
    def bounds(self) -> PatchBounds:
        frozen_center = 0.5
        frozen_length = 3.0
        center = frozen_center - self.upstream_shift_m
        half_length = frozen_length * self.length_multiplier / 2.0
        return PatchBounds(
            center - half_length, center + half_length,
            -self.width_m / 2.0, self.width_m / 2.0,
            self.top_height_m,
        )

    @property
    def deviation_m(self) -> float:
        return self.upstream_shift_m + 3.0 * abs(self.length_multiplier - 1.0)

    @property
    def contract(self) -> dict[str, object]:
        return {
            **asdict(self), "bounds": asdict(self.bounds),
            "length_m": self.bounds.x_max_m - self.bounds.x_min_m,
            "collision_masks": {"patch_contype": 0, "patch_conaffinity": 0},
            "explicit_sole_pair_count": 8,
            "patch_base_overlap": False,
            "width_unchanged": True,
            "top_height_unchanged": True,
            "solref_solimp_unchanged": True,
            "friction_profile_unchanged": True,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.contract, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


GEOMETRY_CANDIDATES = (
    GeometryCandidate("G0", upstream_shift_m=0.0, length_multiplier=1.0),
    GeometryCandidate("G1", upstream_shift_m=0.15, length_multiplier=1.0),
    GeometryCandidate("G2", upstream_shift_m=0.0, length_multiplier=1.15),
    GeometryCandidate("G3", upstream_shift_m=0.15, length_multiplier=1.15),
)


@dataclass(frozen=True)
class SupplementalVariation:
    index: int
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float


# New, fixed perturbations; indices do not overlap the three v5 variations.
SUPPLEMENTAL_VARIATIONS = (
    SupplementalVariation(3, 0.13, 0.010, -0.006),
    SupplementalVariation(4, 0.27, 0.020, -0.003),
    SupplementalVariation(5, 0.55, 0.030, 0.000),
    SupplementalVariation(6, 0.69, 0.050, 0.003),
    SupplementalVariation(7, 0.91, 0.060, 0.006),
    SupplementalVariation(8, 0.97, 0.090, 0.001),
)


@dataclass(frozen=True)
class RunCondition:
    run_id: str
    pair_id: str
    pair_index: int
    role: str
    block: str
    geometry_candidate: str
    geometry_sha256: str
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
    disposition: str

    @property
    def geometry(self) -> GeometryCandidate:
        return geometry_candidate(self.geometry_candidate)

    @property
    def pair_configuration(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "block": self.block,
            "geometry": self.geometry.contract,
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
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
            "censoring": "frozen first-fall physical-oracle contract",
        }

    @property
    def pair_fingerprint(self) -> str:
        payload = json.dumps(
            self.pair_configuration, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def geometry_candidate(candidate_id: str) -> GeometryCandidate:
    try:
        return next(row for row in GEOMETRY_CANDIDATES if row.candidate_id == candidate_id)
    except StopIteration as exc:
        raise ValueError(candidate_id) from exc


def geometry_contract_sha256() -> str:
    payload = json.dumps(
        [row.contract for row in GEOMETRY_CANDIDATES],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def calibration_matrix(duration_s: float = 3.0) -> list[RunCondition]:
    """Return exactly 12 positive and 12 matched calibration-only controls."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    variation = V5_VARIATIONS[1]
    rows: list[RunCondition] = []
    pair_index = 0
    for speed_index, speed in enumerate(SPEEDS_MPS):
        for candidate_index, geometry in enumerate(GEOMETRY_CANDIDATES):
            pair_id = f"v6_cal_pair_{pair_index:02d}"
            control_type = CONTROL_TYPES[(speed_index + candidate_index) % 2]
            tag = f"{speed:.2f}_{geometry.candidate_id}".replace(".", "p")
            common = dict(
                pair_id=pair_id, pair_index=pair_index, block="calibration",
                geometry_candidate=geometry.candidate_id,
                geometry_sha256=geometry.sha256, speed_mps=speed,
                target_foot=TARGET_FOOT, target_phase=TARGET_PHASE,
                severity=TARGET_SEVERITY, control_type=control_type,
                variation_index=variation.index,
                phase_fraction=variation.phase_fraction,
                command_delay_s=variation.command_delay_s,
                lateral_offset_m=variation.lateral_offset_m,
                seed=202608600 + pair_index, duration_s=duration_s,
                disposition=CALIBRATION_DISPOSITION,
            )
            rows.append(RunCondition(
                run_id=f"cal_positive_{tag}", role="positive",
                material_profile=TARGET_SEVERITY, **common))
            rows.append(RunCondition(
                run_id=f"cal_control_{tag}", role="control",
                material_profile=control_type, **common))
            pair_index += 1
    validate_pair_matrix(rows, positive_count=12, pair_count=12)
    return rows


def supplemental_matrix(
    selected_candidate: str, duration_s: float = 3.0,
) -> list[RunCondition]:
    """Return the fixed 18-positive plus 18-control supplemental block."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    geometry = geometry_candidate(selected_candidate)
    rows: list[RunCondition] = []
    pair_index = 0
    for speed_index, speed in enumerate(SPEEDS_MPS):
        for variation_offset, variation in enumerate(SUPPLEMENTAL_VARIATIONS):
            pair_id = f"v6_sup_pair_{pair_index:02d}"
            control_type = CONTROL_TYPES[variation_offset % 2]
            tag = f"{speed:.2f}_v{variation.index}".replace(".", "p")
            common = dict(
                pair_id=pair_id, pair_index=pair_index, block="supplemental",
                geometry_candidate=geometry.candidate_id,
                geometry_sha256=geometry.sha256, speed_mps=speed,
                target_foot=TARGET_FOOT, target_phase=TARGET_PHASE,
                severity=TARGET_SEVERITY, control_type=control_type,
                variation_index=variation.index,
                phase_fraction=variation.phase_fraction,
                command_delay_s=variation.command_delay_s,
                lateral_offset_m=variation.lateral_offset_m,
                seed=202608700 + pair_index, duration_s=duration_s,
                disposition=DEVELOPMENT_DISPOSITION,
            )
            rows.append(RunCondition(
                run_id=f"supplemental_positive_{tag}", role="positive",
                material_profile=TARGET_SEVERITY, **common))
            rows.append(RunCondition(
                run_id=f"supplemental_control_{tag}", role="control",
                material_profile=control_type, **common))
            pair_index += 1
    validate_pair_matrix(rows, positive_count=18, pair_count=18)
    return rows


def validate_pair_matrix(
    rows: Iterable[RunCondition], *, positive_count: int, pair_count: int,
) -> None:
    values = list(rows)
    if len(values) != pair_count * 2 or len({row.run_id for row in values}) != len(values):
        raise ValueError("matrix run count or uniqueness failed")
    if sum(row.role == "positive" for row in values) != positive_count:
        raise ValueError("matrix positive count failed")
    grouped: dict[str, list[RunCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != pair_count:
        raise ValueError("matrix pair count failed")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"invalid pair roles: {pair_id}")
        if len({row.pair_fingerprint for row in pair}) != 1:
            raise ValueError(f"pair mismatch: {pair_id}")
        if len({row.material_profile for row in pair}) != 2:
            raise ValueError(f"friction intervention missing: {pair_id}")
        if any(row.geometry_sha256 != row.geometry.sha256 for row in pair):
            raise ValueError(f"geometry hash mismatch: {pair_id}")


def select_geometry(candidate_metrics: Iterable[dict[str, object]]) -> str | None:
    """Select only passing geometry, using the preregistered tie-break order."""
    passed = {
        str(row["geometry_candidate"])
        for row in candidate_metrics if bool(row["candidate_pass"])
    }
    eligible = [row for row in GEOMETRY_CANDIDATES if row.candidate_id in passed]
    if not eligible:
        return None
    eligible.sort(key=lambda row: (
        row.deviation_m,
        row.bounds.x_max_m - row.bounds.x_min_m,
        row.upstream_shift_m,
        row.candidate_id,
    ))
    return eligible[0].candidate_id


def frozen_profile_sha256() -> dict[str, str]:
    return {name: profile.sha256 for name, profile in material_profiles().items()}


def oracle_contract() -> dict[str, object]:
    return {
        "threshold_m": SLIP_THRESHOLD_M,
        "persistence_ms": SLIP_PERSISTENCE_MS,
        "changed": False,
    }
