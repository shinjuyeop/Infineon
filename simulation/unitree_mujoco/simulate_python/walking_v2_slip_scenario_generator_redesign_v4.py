"""Pure contracts for the bounded Walking-v2 Slip generator v4 audit."""

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
SPEEDS_MPS = (0.10, 0.20)
SIDES = ("left", "right")
PHASES = ("early", "middle", "late")
SEVERITIES = ("native_strong_ice", "moderate_ice_preregistered")
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_MS = 3
ROOT_CAUSES = (
    "LABEL_ONLY_PATCH", "VISUAL_ONLY_PATCH", "WRONG_GEOM_ID",
    "WRONG_CONTACT_PAIR", "POST_SOLVER_MUTATION",
    "POST_CONSTRAINT_MUTATION", "GEOM_FRICTION_MAX_OVERRIDE",
    "PRIORITY_OVERRIDE_FAILURE", "DOUBLE_CONTACT_WITH_BASE_FLOOR",
    "PATCH_NEVER_ACTIVE", "MODEL_COPY_NOT_USED", "STEP_ORDER_ERROR",
    "OTHER_WITH_EVIDENCE",
)


@dataclass(frozen=True)
class FrictionProfile:
    name: str
    friction3: tuple[float, float, float]
    source: str

    @property
    def friction5(self) -> tuple[float, float, float, float, float]:
        slide, torsion, rolling = self.friction3
        return slide, slide, torsion, rolling, rolling


def friction_profiles() -> dict[str, FrictionProfile]:
    ice = np.asarray(TERRAIN_PROFILES["ice"].friction, dtype=float)
    marble = np.asarray(TERRAIN_PROFILES["marble"].friction, dtype=float)
    moderate = ice + 0.125 * (marble - ice)
    concrete = np.asarray(TERRAIN_PROFILES["concrete"].friction, dtype=float)
    return {
        "hard_control": FrictionProfile(
            "hard_control", tuple(concrete.tolist()), "native Concrete"),
        "moderate_ice_preregistered": FrictionProfile(
            "moderate_ice_preregistered", tuple(moderate.tolist()),
            "frozen v3 Ice + 1/8*(Marble-Ice)"),
        "native_strong_ice": FrictionProfile(
            "native_strong_ice", tuple(ice.tolist()), "native Ice"),
    }


@dataclass(frozen=True)
class PatchBounds:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    top_z_m: float = 0.0

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.x_min_m + self.x_max_m) / 2,
            (self.y_min_m + self.y_max_m) / 2,
            self.top_z_m - 0.1,
        )

    @property
    def half_size(self) -> tuple[float, float, float]:
        return (
            (self.x_max_m - self.x_min_m) / 2,
            (self.y_max_m - self.y_min_m) / 2,
            0.1,
        )


def patch_bounds(side: str) -> PatchBounds:
    if side not in SIDES:
        raise ValueError(side)
    # A finite cross-walk patch keeps every coplanar seam outside the bounded
    # three-second reachable envelope.  Unilaterality is provided by the
    # explicit target-sole pair contract, which the task explicitly permits.
    return PatchBounds(-1.0, 2.0, -0.8, 0.8)


@dataclass(frozen=True)
class PilotCondition:
    run_id: str
    pair_id: str
    role: str
    speed_mps: float
    target_foot: str
    target_phase: str
    severity: str
    profile: str
    seed: int
    duration_s: float = 3.0
    phase_fraction: float = 0.07
    command_delay_s: float = 0.0

    @property
    def parity_payload(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("run_id")
        value.pop("role")
        value.pop("profile")
        return value

    @property
    def parity_sha256(self) -> str:
        encoded = json.dumps(
            self.parity_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def pilot_matrix(duration_s: float = 3.0) -> list[PilotCondition]:
    """Return exactly 24 positive/control pairs and 48 unique runs."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    rows: list[PilotCondition] = []
    index = 0
    for speed in SPEEDS_MPS:
        for foot in SIDES:
            for phase in PHASES:
                for severity in SEVERITIES:
                    pair_id = f"pilot_pair_{index:02d}"
                    tag = f"{speed:.2f}_{foot}_{phase}_{severity}".replace(".", "p")
                    common = dict(
                        pair_id=pair_id, speed_mps=speed, target_foot=foot,
                        target_phase=phase, severity=severity,
                        seed=202608400 + index, duration_s=duration_s,
                    )
                    rows.append(PilotCondition(
                        run_id=f"positive_{tag}", role="positive",
                        profile=severity, **common))
                    rows.append(PilotCondition(
                        run_id=f"control_{tag}", role="control",
                        profile="hard_control", **common))
                    index += 1
    validate_pilot_matrix(rows)
    return rows


def validate_pilot_matrix(rows: Iterable[PilotCondition]) -> None:
    values = list(rows)
    if len(values) != 48 or len({row.run_id for row in values}) != 48:
        raise ValueError("pilot must contain exactly 48 unique runs")
    grouped: dict[str, list[PilotCondition]] = {}
    for row in values:
        grouped.setdefault(row.pair_id, []).append(row)
    if len(grouped) != 24:
        raise ValueError("pilot must contain exactly 24 pairs")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row.role for row in pair} != {"positive", "control"}:
            raise ValueError(f"invalid pair {pair_id}")
        if len({row.parity_sha256 for row in pair}) != 1:
            raise ValueError(f"parity failure {pair_id}")
        if len({row.profile for row in pair}) != 2:
            raise ValueError(f"missing intervention {pair_id}")
    positives = [row for row in values if row.role == "positive"]
    cells = {
        (row.speed_mps, row.target_foot, row.target_phase, row.severity)
        for row in positives
    }
    if len(positives) != 24 or len(cells) != 24:
        raise ValueError("pilot factorial coverage incomplete")


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
        array = np.ascontiguousarray(trace[key])
        digest.update(key.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def first_persistent(mask: np.ndarray, persistence: int = 3) -> int | None:
    values = np.asarray(mask, dtype=bool)
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    run = 0
    for index, active in enumerate(values):
        run = run + 1 if active else 0
        if run >= persistence:
            return index - persistence + 1
    return None
