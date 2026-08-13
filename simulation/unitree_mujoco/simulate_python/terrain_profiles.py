"""MuJoCo terrain profiles for relative HIL signal-separation experiments.

These profiles are engineering approximations.  They intentionally create
relative contact differences and are not measurements of real materials.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class TerrainProfile:
    name: str
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    description: str
    priority: int = 1
    condim: int = 3


TERRAIN_PROFILES = {
    "concrete": TerrainProfile(
        name="concrete",
        friction=(1.00, 0.005, 0.0001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard, relatively high-friction PoC reference",
    ),
    "marble": TerrainProfile(
        name="marble",
        friction=(0.45, 0.003, 0.0001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard contact with lower sliding friction than concrete",
    ),
    "ice": TerrainProfile(
        name="ice",
        friction=(0.05, 0.001, 0.00001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard contact with very low friction",
    ),
    "sand": TerrainProfile(
        name="sand",
        friction=(0.70, 0.010, 0.0010),
        solref=(0.050, 1.5),
        solimp=(0.70, 0.90, 0.010, 0.5, 2.0),
        description="softer, more damped and lower-impedance PoC contact",
    ),
    "sand_slightly_compliant": TerrainProfile(
        name="sand_slightly_compliant",
        friction=(0.70, 0.010, 0.0010),
        solref=(0.060, 1.5),
        solimp=(0.65, 0.88, 0.012, 0.5, 2.0),
        description="bounded softer variant derived from native Sand",
    ),
    "sand_moderately_compliant": TerrainProfile(
        name="sand_moderately_compliant",
        friction=(0.70, 0.010, 0.0010),
        solref=(0.070, 1.5),
        solimp=(0.60, 0.86, 0.015, 0.5, 2.0),
        description="bounded more-compliant variant derived from native Sand",
    ),
}


def apply_terrain_profile(
    model: mujoco.MjModel, profile: TerrainProfile, floor_name: str = "floor"
) -> int:
    """Apply one profile to the named floor geom and return its runtime id."""
    floor_id = model.geom(floor_name).id

    friction = np.asarray(profile.friction, dtype=np.float64)
    solref = np.asarray(profile.solref, dtype=np.float64)
    solimp = np.asarray(profile.solimp, dtype=np.float64)
    if friction.shape != (3,) or np.any(friction < 0):
        raise ValueError(f"invalid friction profile for {profile.name}")
    if solref.shape != (2,) or np.any(solref <= 0):
        raise ValueError(f"invalid positive-format solref for {profile.name}")
    if solimp.shape != (5,) or not np.all(np.isfinite(solimp)):
        raise ValueError(f"invalid solimp profile for {profile.name}")
    d_zero, d_width, width, midpoint, power = solimp
    if not (
        0.0 < d_zero < 1.0
        and 0.0 < d_width < 1.0
        and width > 0.0
        and 0.0 < midpoint < 1.0
        and power >= 1.0
    ):
        raise ValueError(f"solimp values are out of range for {profile.name}")
    if solref[0] < 2.0 * model.opt.timestep:
        raise ValueError(
            f"{profile.name} solref time constant must be at least 2*timestep"
        )

    model.geom_friction[floor_id] = friction
    model.geom_solref[floor_id] = solref
    model.geom_solimp[floor_id] = solimp
    model.geom_priority[floor_id] = profile.priority
    model.geom_condim[floor_id] = profile.condim
    return floor_id


def describe_profile(profile: TerrainProfile) -> str:
    return (
        f"{profile.name}: friction={profile.friction}, solref={profile.solref}, "
        f"solimp={profile.solimp}, priority={profile.priority}, "
        f"condim={profile.condim} ({profile.description})"
    )
