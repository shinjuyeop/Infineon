"""Pure contracts for additional moderate-v2 Slip acquisition v8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ, deterministic_initial_perturbation,
)
from walking_v2_slip_moderate_profile_recalibration_v7 import (
    DEVELOPMENT_DISPOSITION, RunCondition, candidate_profile,
)
from walking_v2_slip_supplemental_acquisition_v6 import GEOMETRY_CANDIDATES


TARGET_SPEED_MPS = 0.15
TARGET_FOOT = "left"
TARGET_PHASE = "mid_late_stance"
TARGET_SEVERITY = "moderate_v2"
LOCKED_CANDIDATE = "M1"
CALIBRATION_PHASE_FRACTION = 0.41

# These exact values already exist in the frozen v5/v6 variation contracts.
PHASE_FRACTIONS = (0.27, 0.41, 0.55)
PHASE_OFFSETS = tuple(value - CALIBRATION_PHASE_FRACTION for value in PHASE_FRACTIONS)
COMMAND_DELAYS_S = (0.02, 0.03, 0.04, 0.05)

# The twelve closest unused deterministic perturbations to the successful
# 0.15 m/s calibration perturbation in the frozen [202608803, 202608899]
# seed neighborhood.  Selection uses prior calibration evidence only.
PREREGISTERED_SEEDS = (
    202608814, 202608839, 202608834, 202608895,
    202608855, 202608897, 202608856, 202608871,
    202608812, 202608898, 202608893, 202608872,
)
VARIATION_IDS = tuple(f"V8_L{index:02d}" for index in range(12))


@dataclass(frozen=True)
class VariationContract:
    variation_id: str
    variation_index: int
    phase_offset: float
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float
    seed: int
    control_type: str
    execution_pair_index: int

    @property
    def initial_perturbation(self) -> tuple[float, float, float]:
        return deterministic_initial_perturbation(self.seed)

    @property
    def contract(self) -> dict[str, object]:
        return {
            **asdict(self),
            "initial_perturbation": self.initial_perturbation,
            "speed_mps": TARGET_SPEED_MPS,
            "target_foot": TARGET_FOOT,
            "target_phase": TARGET_PHASE,
            "severity": TARGET_SEVERITY,
            "candidate_id": LOCKED_CANDIDATE,
            "profile_sha256": candidate_profile(LOCKED_CANDIDATE).sha256,
            "geometry_candidate": "G0",
            "geometry_sha256": GEOMETRY_CANDIDATES[0].sha256,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
            "censoring": "frozen first-fall physical-oracle contract",
        }


def supplemental_variations() -> tuple[VariationContract, ...]:
    """Return the frozen 3-by-4 preregistered lattice."""
    rows: list[VariationContract] = []
    index = 0
    for phase_fraction, phase_offset in zip(PHASE_FRACTIONS, PHASE_OFFSETS):
        for command_delay in COMMAND_DELAYS_S:
            rows.append(VariationContract(
                variation_id=VARIATION_IDS[index],
                variation_index=9 + index,
                phase_offset=phase_offset,
                phase_fraction=phase_fraction,
                command_delay_s=command_delay,
                lateral_offset_m=0.0,
                seed=PREREGISTERED_SEEDS[index],
                control_type=CONTROL_TYPES[index % 2],
                execution_pair_index=index,
            ))
            index += 1
    validate_variations(rows)
    return tuple(rows)


def validate_variations(rows: Iterable[VariationContract]) -> None:
    values = list(rows)
    if len(values) != 12 or len({row.variation_id for row in values}) != 12:
        raise ValueError("supplement requires exactly 12 unique variations")
    if {(row.phase_fraction, row.command_delay_s) for row in values} != {
        (phase, delay) for phase in PHASE_FRACTIONS for delay in COMMAND_DELAYS_S
    }:
        raise ValueError("supplemental lattice is not the frozen Cartesian product")
    if len({row.seed for row in values}) != 12:
        raise ValueError("supplemental seeds must be unique")
    perturbations = [np.asarray(row.initial_perturbation) for row in values]
    if len({tuple(value.tolist()) for value in perturbations}) != 12:
        raise ValueError("initial physical policy states must be distinct")
    if {row.control_type for row in values} != set(CONTROL_TYPES):
        raise ValueError("both control types are required")
    if sum(row.control_type == CONTROL_TYPES[0] for row in values) != 6:
        raise ValueError("controls must be balanced 6/6")


def variation_contract_payload() -> dict[str, object]:
    rows = supplemental_variations()
    return {
        "version": "walking_v2_moderate_v2_supplemental_variations_v8",
        "frozen_before_first_supplemental_result": True,
        "primary_diagnosis": "PATCH_ENTRY_POSE_SENSITIVITY",
        "lattice_shape": [3, 4],
        "phase_fraction_value_source": {
            "0.27": "frozen v6 supplemental variation 4",
            "0.41": "successful frozen v7 calibration variation 1",
            "0.55": "frozen v6 supplemental variation 5",
        },
        "command_delay_value_source": {
            "0.02": "frozen v6 supplemental variation 4",
            "0.03": "frozen v6 supplemental variation 5",
            "0.04": "successful frozen v7 calibration variation 1",
            "0.05": "frozen v6 supplemental variation 6",
        },
        "seed_selection": {
            "anchor_seed": 202608801,
            "candidate_seed_interval_inclusive": [202608803, 202608899],
            "rule": "twelve smallest Euclidean deterministic-perturbation distances; seed tie-break",
            "uses_supplemental_results": False,
        },
        "patch_placement": GEOMETRY_CANDIDATES[0].contract,
        "locked_profile": candidate_profile(LOCKED_CANDIDATE).contract,
        "run_order": [row.variation_id for row in rows],
        "variations": [row.contract for row in rows],
        "adaptive_additions": False,
        "seed_replacement": False,
        "result_conditioned_changes": False,
    }


def variation_contract_sha256() -> str:
    payload = json.dumps(
        variation_contract_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def supplemental_matrix(duration_s: float = 3.0) -> list[RunCondition]:
    """Return exactly 12 positive/control pairs in frozen execution order."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    profile = candidate_profile(LOCKED_CANDIDATE)
    rows: list[RunCondition] = []
    for variation in supplemental_variations():
        pair_id = f"v8_pair_{variation.execution_pair_index:02d}"
        tag = variation.variation_id.lower()
        common = dict(
            pair_id=pair_id,
            pair_index=variation.execution_pair_index,
            block="moderate_v2_supplemental_cell",
            calibration_arm="",
            speed_mps=TARGET_SPEED_MPS,
            target_foot=TARGET_FOOT,
            target_phase=TARGET_PHASE,
            severity=TARGET_SEVERITY,
            control_type=variation.control_type,
            variation_index=variation.variation_index,
            phase_fraction=variation.phase_fraction,
            command_delay_s=variation.command_delay_s,
            lateral_offset_m=variation.lateral_offset_m,
            seed=variation.seed,
            duration_s=duration_s,
            disposition=DEVELOPMENT_DISPOSITION,
        )
        rows.append(RunCondition(
            run_id=f"supplemental_positive_{tag}", role="positive",
            material_profile=profile.name, **common))
        rows.append(RunCondition(
            run_id=f"supplemental_control_{tag}", role="control",
            material_profile=variation.control_type, **common))
    validate_matrix(rows)
    return rows


def validate_matrix(rows: Iterable[RunCondition]) -> None:
    values = list(rows)
    if len(values) != 24 or len({row.run_id for row in values}) != 24:
        raise ValueError("supplement requires exactly 24 unique runs")
    groups: dict[str, list[RunCondition]] = {}
    for row in values:
        groups.setdefault(row.pair_id, []).append(row)
    if len(groups) != 12:
        raise ValueError("supplement requires exactly 12 pairs")
    for pair_id, pair in groups.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"pair roles invalid: {pair_id}")
        if len({row.pair_fingerprint for row in pair}) != 1:
            raise ValueError(f"counterfactual mismatch: {pair_id}")
        if len({row.material_profile for row in pair}) != 2:
            raise ValueError(f"friction intervention missing: {pair_id}")
    positives = [row for row in values if row.role == "positive"]
    controls = [row for row in values if row.role == "control"]
    if len(positives) != 12 or len(controls) != 12:
        raise ValueError("positive/control counts must be 12/12")
    if len({row.pair_fingerprint for row in positives}) != 12:
        raise ValueError("complete positive configurations must be distinct")
    if sum(row.control_type == "hard_normal" for row in controls) != 6:
        raise ValueError("hard-normal controls must total six")
    if sum(row.control_type == "near_slip_non_event" for row in controls) != 6:
        raise ValueError("near-slip controls must total six")
