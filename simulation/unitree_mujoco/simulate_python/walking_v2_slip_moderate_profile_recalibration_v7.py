"""Pure contracts for moderate bilateral Slip profile recalibration v7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    PHASE_BINS, SAMPLE_RATE_HZ, SIDES, SPEEDS_MPS, VARIATIONS,
    material_profiles as v5_material_profiles,
)
from walking_v2_slip_supplemental_acquisition_v6 import GEOMETRY_CANDIDATES


TARGET_FOOT = "left"
TARGET_PHASE = "mid_late_stance"
CALIBRATION_DISPOSITION = "PROFILE_CALIBRATION_ONLY_DO_NOT_TRAIN"
DEVELOPMENT_DISPOSITION = "DEVELOPMENT_ONLY"
SEVERITY_ORDER_TOLERANCE_M = 0.005
CANDIDATE_IDS = ("M0", "M1", "M2", "M3")
CALIBRATION_ARMS = ("hard_control", "strong_reference") + CANDIDATE_IDS
SEVERITY_ORDER = ("hard_control",) + CANDIDATE_IDS + ("strong_reference",)


@dataclass(frozen=True)
class FrictionProfile:
    name: str
    candidate_id: str
    friction3: tuple[float, float, float]
    role: str
    frozen_source: str
    interpolation_fraction_from_strong: float | None

    @property
    def friction5(self) -> tuple[float, float, float, float, float]:
        sliding, torsional, rolling = self.friction3
        return sliding, sliding, torsional, rolling, rolling

    @property
    def contract(self) -> dict[str, object]:
        return {
            **asdict(self), "friction5": self.friction5,
            "interpolated_components": [0, 1],
            "torsional_component_frozen_from": "moderate-v1",
            "rolling_components_frozen_from": "moderate-v1",
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.contract, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def friction_profiles() -> dict[str, FrictionProfile]:
    """Return the frozen references, controls, and exact four-arm v2 grid."""
    frozen = v5_material_profiles()
    strong = frozen["native_strong_ice"].friction3
    moderate = frozen["moderate_ice_preregistered"].friction3
    hard = frozen["hard_normal"].friction3
    near = frozen["near_slip_non_event"].friction3
    fractions = {"M0": 1.0, "M1": 0.75, "M2": 0.50, "M3": 0.25}
    result = {
        "hard_normal": FrictionProfile(
            "hard_normal", "hard_control", hard, "control",
            "v5 frozen hard-normal", None),
        "near_slip_non_event": FrictionProfile(
            "near_slip_non_event", "near_slip_non_event", near, "control",
            "v5 frozen near-slip non-event", None),
        "native_strong_ice": FrictionProfile(
            "native_strong_ice", "strong_reference", strong, "reference",
            "v5 frozen native strong Ice", 0.0),
    }
    for candidate_id, fraction in fractions.items():
        sliding = strong[0] + fraction * (moderate[0] - strong[0])
        name = "moderate_v1_M0" if candidate_id == "M0" else f"moderate_v2_{candidate_id}"
        result[name] = FrictionProfile(
            name, candidate_id,
            (sliding, moderate[1], moderate[2]),
            "candidate", "preregistered v7 sliding-only interpolation", fraction)
    validate_friction_grid(result)
    return result


def candidate_profile(candidate_id: str) -> FrictionProfile:
    for profile in friction_profiles().values():
        if profile.candidate_id == candidate_id and profile.role == "candidate":
            return profile
    raise ValueError(candidate_id)


def validate_friction_grid(profiles: dict[str, FrictionProfile]) -> None:
    strong = profiles["native_strong_ice"].friction3
    candidates = [candidate_profile_from(profiles, value) for value in CANDIDATE_IDS]
    expected = (0.1000, 0.0875, 0.0750, 0.0625)
    if tuple(round(row.friction3[0], 10) for row in candidates) != expected:
        raise ValueError("candidate sliding grid drift")
    if not all(row.friction3[0] > strong[0] for row in candidates):
        raise ValueError("candidate must remain numerically above strong Ice")
    if len({row.friction5 for row in candidates}) != 4:
        raise ValueError("candidate vectors must be distinct")
    moderate_torsion_roll = candidates[0].friction3[1:]
    if any(row.friction3[1:] != moderate_torsion_roll for row in candidates):
        raise ValueError("torsional or rolling friction changed across candidates")


def candidate_profile_from(
    profiles: dict[str, FrictionProfile], candidate_id: str,
) -> FrictionProfile:
    return next(
        row for row in profiles.values()
        if row.candidate_id == candidate_id and row.role == "candidate")


def friction_grid_sha256() -> str:
    profiles = friction_profiles()
    payload = [candidate_profile_from(profiles, value).contract for value in CANDIDATE_IDS]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RunCondition:
    run_id: str
    pair_id: str
    pair_index: int
    role: str
    block: str
    calibration_arm: str
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
    def pair_configuration(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id, "block": self.block,
            "speed_mps": self.speed_mps, "target_foot": self.target_foot,
            "target_phase": self.target_phase, "severity": self.severity,
            "variation_index": self.variation_index,
            "phase_fraction": self.phase_fraction,
            "command_delay_s": self.command_delay_s,
            "lateral_offset_m": self.lateral_offset_m,
            "seed": self.seed, "duration_s": self.duration_s,
            "patch_geometry": GEOMETRY_CANDIDATES[0].contract,
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


def profile_calibration_matrix(duration_s: float = 3.0) -> list[RunCondition]:
    """Return exactly three matched six-arm groups (18 unique runs)."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    variation = VARIATIONS[1]
    profiles = friction_profiles()
    arm_profiles = {
        "hard_control": "hard_normal",
        "strong_reference": "native_strong_ice",
        **{candidate_id: candidate_profile_from(profiles, candidate_id).name
           for candidate_id in CANDIDATE_IDS},
    }
    rows: list[RunCondition] = []
    for speed_index, speed in enumerate(SPEEDS_MPS):
        group_id = f"v7_cal_group_{speed_index}"
        for arm_index, arm in enumerate(CALIBRATION_ARMS):
            profile_name = arm_profiles[arm]
            rows.append(RunCondition(
                run_id=f"cal_{speed:.2f}_{arm}".replace(".", "p"),
                pair_id=group_id, pair_index=speed_index,
                role="control" if arm == "hard_control" else "positive",
                block="profile_calibration", calibration_arm=arm,
                speed_mps=speed, target_foot=TARGET_FOOT,
                target_phase=TARGET_PHASE, severity="profile_calibration",
                material_profile=profile_name,
                control_type="hard_normal" if arm == "hard_control" else "multi_arm",
                variation_index=variation.index,
                phase_fraction=variation.phase_fraction,
                command_delay_s=variation.command_delay_s,
                lateral_offset_m=variation.lateral_offset_m,
                seed=202608800 + speed_index, duration_s=duration_s,
                disposition=CALIBRATION_DISPOSITION,
            ))
    validate_calibration_matrix(rows)
    return rows


def validate_calibration_matrix(rows: Iterable[RunCondition]) -> None:
    values = list(rows)
    if len(values) != 18 or len({row.run_id for row in values}) != 18:
        raise ValueError("calibration requires 18 unique runs")
    grouped: dict[str, list[RunCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != 3:
        raise ValueError("calibration requires three speed groups")
    for group_id, group in grouped.items():
        if {row.calibration_arm for row in group} != set(CALIBRATION_ARMS):
            raise ValueError(f"six-arm group incomplete: {group_id}")
        if len({row.pair_fingerprint for row in group}) != 1:
            raise ValueError(f"non-friction calibration mismatch: {group_id}")
        if len({row.material_profile for row in group}) != 6:
            raise ValueError(f"friction arms not distinct: {group_id}")


def moderate_v2_matrix(
    selected_candidate: str, duration_s: float = 3.0,
) -> list[RunCondition]:
    """Return the complete 54-positive plus 54-control moderate-v2 matrix."""
    if selected_candidate not in {"M1", "M2", "M3"}:
        raise ValueError("locked moderate-v2 must be strictly between references")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    profile = candidate_profile(selected_candidate)
    phase_names = tuple(row.name for row in PHASE_BINS)
    rows: list[RunCondition] = []
    pair_index = 0
    for speed_index, speed in enumerate(SPEEDS_MPS):
        for foot_index, foot in enumerate(SIDES):
            for phase_index, phase in enumerate(phase_names):
                for variation in VARIATIONS:
                    pair_id = f"v7_pair_{pair_index:03d}"
                    control_type = CONTROL_TYPES[
                        (speed_index + foot_index + phase_index + variation.index) % 2]
                    tag = f"{speed:.2f}_{foot}_{phase}_v{variation.index}".replace(".", "p")
                    common = dict(
                        pair_id=pair_id, pair_index=pair_index,
                        block="moderate_v2_reacquisition", calibration_arm="",
                        speed_mps=speed, target_foot=foot, target_phase=phase,
                        severity="moderate_v2", control_type=control_type,
                        variation_index=variation.index,
                        phase_fraction=variation.phase_fraction,
                        command_delay_s=variation.command_delay_s,
                        lateral_offset_m=variation.lateral_offset_m,
                        seed=202608900 + pair_index, duration_s=duration_s,
                        disposition=DEVELOPMENT_DISPOSITION,
                    )
                    rows.append(RunCondition(
                        run_id=f"positive_{tag}_{selected_candidate}", role="positive",
                        material_profile=profile.name, **common))
                    rows.append(RunCondition(
                        run_id=f"control_{tag}_{selected_candidate}", role="control",
                        material_profile=control_type, **common))
                    pair_index += 1
    validate_reacquisition_matrix(rows)
    return rows


def validate_reacquisition_matrix(rows: Iterable[RunCondition]) -> None:
    values = list(rows)
    if len(values) != 108 or len({row.run_id for row in values}) != 108:
        raise ValueError("reacquisition requires 108 unique runs")
    grouped: dict[str, list[RunCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != 54:
        raise ValueError("reacquisition requires 54 matched pairs")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"invalid pair: {pair_id}")
        if len({row.pair_fingerprint for row in pair}) != 1:
            raise ValueError(f"pair configuration mismatch: {pair_id}")
    positives = [row for row in values if row.role == "positive"]
    cells = {(row.speed_mps, row.target_foot, row.target_phase) for row in positives}
    if len(positives) != 54 or len(cells) != 18:
        raise ValueError("moderate-v2 factorial coverage incomplete")
    if any(sum(
        (row.speed_mps, row.target_foot, row.target_phase) == cell
        for row in positives) != 3 for cell in cells):
        raise ValueError("every moderate-v2 cell needs three variations")
    for speed in SPEEDS_MPS:
        selected = [row for row in values if row.role == "control" and row.speed_mps == speed]
        counts = {name: sum(row.control_type == name for row in selected)
                  for name in CONTROL_TYPES}
        if abs(counts[CONTROL_TYPES[0]] - counts[CONTROL_TYPES[1]]) > 1:
            raise ValueError("controls not balanced within speed")


def select_candidate(candidate_rows: Iterable[dict[str, object]]) -> str | None:
    passed = {
        str(row["candidate_id"]) for row in candidate_rows
        if bool(row["candidate_pass"]) and str(row["candidate_id"]) in {"M1", "M2", "M3"}
    }
    if not passed:
        return None
    profiles = friction_profiles()
    eligible = [candidate_profile_from(profiles, value) for value in passed]
    eligible.sort(key=lambda row: (-row.friction3[0], row.candidate_id))
    return eligible[0].candidate_id
