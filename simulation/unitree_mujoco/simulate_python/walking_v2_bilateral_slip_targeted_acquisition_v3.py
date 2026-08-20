"""Pure contracts and audits for targeted bilateral Slip acquisition v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from terrain_profiles import TERRAIN_PROFILES


SAMPLE_RATE_HZ = 1000
PHYSICS_TIMESTEP_S = 0.0005
PHYSICS_STEPS_PER_SAMPLE = 2
SPEEDS_MPS = (0.10, 0.15, 0.20)
SIDES = ("left", "right")
SEVERITIES = ("native_strong_ice", "moderate_ice_preregistered")
CONTROL_TYPES = ("hard_normal", "near_slip_non_event")
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_MS = 3
TOUCHDOWN_TRANSIENT_MS = 10


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
    friction: tuple[float, float, float]
    derivation: str
    source_role: str


def material_profiles() -> dict[str, MaterialProfile]:
    """Return exactly two positives and two predeclared control profiles."""
    ice = np.asarray(TERRAIN_PROFILES["ice"].friction, dtype=float)
    marble = np.asarray(TERRAIN_PROFILES["marble"].friction, dtype=float)
    concrete = np.asarray(TERRAIN_PROFILES["concrete"].friction, dtype=float)
    moderate = ice + 0.125 * (marble - ice)
    near = ice + 0.75 * (marble - ice)
    return {
        "native_strong_ice": MaterialProfile(
            "native_strong_ice", tuple(ice.tolist()),
            "existing native Ice friction, unchanged", "positive",
        ),
        "moderate_ice_preregistered": MaterialProfile(
            "moderate_ice_preregistered", tuple(moderate.tolist()),
            "Ice + 1/8 * (Marble - Ice), fixed before acquisition", "positive",
        ),
        "hard_normal": MaterialProfile(
            "hard_normal", tuple(concrete.tolist()),
            "existing native Concrete friction, unchanged", "control",
        ),
        "near_slip_non_event": MaterialProfile(
            "near_slip_non_event", tuple(near.tolist()),
            "Ice + 3/4 * (Marble - Ice), fixed before acquisition", "control",
        ),
    }


@dataclass(frozen=True)
class PatchGeometry:
    target_foot: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    surface_z_m: float = 0.0
    height_delta_m: float = 0.0

    def contains(self, position_xyz: np.ndarray) -> bool:
        position = np.asarray(position_xyz, dtype=float)
        return bool(
            self.x_min_m <= position[0] <= self.x_max_m
            and self.y_min_m <= position[1] <= self.y_max_m
        )


def patch_for_foot(side: str) -> PatchGeometry:
    if side == "left":
        return PatchGeometry(side, -0.30, 1.50, 0.015, 0.260)
    if side == "right":
        return PatchGeometry(side, -0.30, 1.50, -0.260, -0.015)
    raise ValueError(f"unknown side {side!r}")


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
    patch: PatchGeometry

    @property
    def pair_configuration(self) -> dict[str, object]:
        """Configuration that must be exactly equal inside a pair."""
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
            "patch": asdict(self.patch),
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
        }

    @property
    def pair_fingerprint(self) -> str:
        payload = json.dumps(
            self.pair_configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _control_for(
    severity_index: int, foot_index: int, phase_index: int, variation_index: int
) -> str:
    parity = severity_index + foot_index + phase_index + variation_index
    return CONTROL_TYPES[parity % len(CONTROL_TYPES)]


def acquisition_matrix(duration_s: float = 3.0) -> list[RunCondition]:
    """Create the frozen 108 positive + 108 matched-control matrix."""
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    rows: list[RunCondition] = []
    pair_index = 0
    for speed in SPEEDS_MPS:
        for foot_index, foot in enumerate(SIDES):
            for phase_index, phase in enumerate(PHASE_BINS):
                for severity_index, severity in enumerate(SEVERITIES):
                    for variation in VARIATIONS:
                        pair_id = f"pair_{pair_index:03d}"
                        speed_id = f"{speed:.2f}".replace(".", "p")
                        base = (
                            f"{speed_id}_{foot}_{phase.name}_{severity}_v{variation.index}"
                        )
                        seed = 202608300 + pair_index
                        control = _control_for(
                            severity_index, foot_index, phase_index, variation.index
                        )
                        common = {
                            "pair_id": pair_id,
                            "pair_index": pair_index,
                            "speed_mps": speed,
                            "target_foot": foot,
                            "target_phase": phase.name,
                            "severity": severity,
                            "control_type": control,
                            "variation_index": variation.index,
                            "phase_fraction": variation.phase_fraction,
                            "command_delay_s": variation.command_delay_s,
                            "lateral_offset_m": variation.lateral_offset_m,
                            "seed": seed,
                            "duration_s": duration_s,
                            "patch": patch_for_foot(foot),
                        }
                        rows.append(RunCondition(
                            run_id=f"positive_{base}", role="positive",
                            material_profile=severity, **common,
                        ))
                        rows.append(RunCondition(
                            run_id=f"control_{base}", role="control",
                            material_profile=control, **common,
                        ))
                        pair_index += 1
    validate_matrix(rows)
    return rows


def validate_matrix(rows: Iterable[RunCondition]) -> None:
    values = list(rows)
    if len(values) != 216:
        raise ValueError("acquisition matrix must contain exactly 216 runs")
    if len({row.run_id for row in values}) != 216:
        raise ValueError("run IDs must be unique")
    grouped: dict[str, list[RunCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != 108:
        raise ValueError("matrix must contain exactly 108 unique pair IDs")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"invalid matched pair {pair_id}")
        if len({row.pair_fingerprint for row in pair}) != 1:
            raise ValueError(f"configuration parity failed for {pair_id}")
        if pair[0].material_profile == pair[1].material_profile:
            raise ValueError(f"material intervention missing for {pair_id}")
    positives = [row for row in values if row.role == "positive"]
    cells = {
        (row.speed_mps, row.target_foot, row.target_phase, row.severity)
        for row in positives
    }
    if len(positives) != 108 or len(cells) != 36:
        raise ValueError("positive matrix cell coverage is incomplete")
    for cell in cells:
        if sum(
            (row.speed_mps, row.target_foot, row.target_phase, row.severity) == cell
            for row in positives
        ) != 3:
            raise ValueError(f"cell does not have three variations: {cell}")
    for speed in SPEEDS_MPS:
        for foot in SIDES:
            for phase in PHASE_BINS:
                for variation in VARIATIONS:
                    controls = {
                        row.control_type for row in positives
                        if row.speed_mps == speed
                        and row.target_foot == foot
                        and row.target_phase == phase.name
                        and row.variation_index == variation.index
                    }
                    if controls != set(CONTROL_TYPES):
                        raise ValueError("control types are not balanced in each base cell")


def phase_spec(name: str) -> PhaseBin:
    try:
        return next(value for value in PHASE_BINS if value.name == name)
    except StopIteration as exc:
        raise ValueError(f"unknown target phase {name!r}") from exc


def onset_phase(age_ms: int) -> str:
    for phase in PHASE_BINS:
        if phase.minimum_onset_ms <= age_ms <= phase.maximum_onset_ms:
            return phase.name
    return "out_of_contract"


def friction_vector(profile: MaterialProfile) -> np.ndarray:
    """Map MuJoCo geom (slide, torsion, roll) to contact's five coefficients."""
    sliding, torsional, rolling = profile.friction
    return np.asarray((sliding, sliding, torsional, rolling, rolling), dtype=float)


def deterministic_initial_perturbation(seed: int) -> tuple[float, float, float]:
    """Small pair-shared pose perturbation makes all attempts physically distinct."""
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform((-0.002, -0.001, -0.002), (0.002, 0.001, 0.002)))


def full_trace_sha256(trace: dict[str, np.ndarray], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        array = np.ascontiguousarray(trace[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def assigned_future_fold(source: str, variation_index: int) -> int:
    """Variation-grouped deterministic three-fold contract."""
    if source not in {"existing_development", "targeted_acquisition_v3"}:
        raise ValueError(source)
    return int(variation_index) % 3


def validate_future_fold_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    owners: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        fold = int(row["fold"])
        for kind in ("pair_id", "variation_group", "run_id", "episode_group"):
            value = str(row.get(kind, ""))
            if value:
                owners.setdefault((kind, value), set()).add(fold)
    leaks = [f"{kind}:{value}" for (kind, value), folds in owners.items() if len(folds) != 1]
    fold_ids = sorted({int(row["fold"]) for row in rows})
    return {
        "row_count": len(rows),
        "fold_ids": fold_ids,
        "group_leakage_count": len(leaks),
        "leaking_groups": leaks,
        "valid": fold_ids == [0, 1, 2] and not leaks,
    }
