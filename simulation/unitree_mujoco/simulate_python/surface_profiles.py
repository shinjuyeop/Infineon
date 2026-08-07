"""Deterministic, modest MuJoCo hfield profiles for the surface-rate study."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from terrain_profiles import TerrainProfile, apply_terrain_profile


HFIELD_NAME = "study_surface"
SURFACE_FLOOR_NAME = "surface_floor"
GRID_SPACING_M = 0.010
WAVELENGTHS_M = (0.040, 0.065, 0.110)
CONCRETE_PEAK_TO_VALLEY_M = 0.0004
MARBLE_PEAK_TO_VALLEY_M = 0.00002


@dataclass(frozen=True)
class SurfaceStatistics:
    peak_to_valley_m: float
    rms_height_m: float
    grid_spacing_m: float
    wavelengths_m: tuple[float, ...]


def _normalized_irregular_surface(nrow: int, ncol: int) -> np.ndarray:
    """Return a repeatable, zero-centered multi-scale surface in [-1, 1]."""
    x = np.linspace(-0.6, 0.6, ncol)
    y = np.linspace(-0.4, 0.4, nrow)
    xx, yy = np.meshgrid(x, y)
    surface = (
        0.52 * np.sin(2.0 * np.pi * xx / WAVELENGTHS_M[0] + 0.37)
        * np.cos(2.0 * np.pi * yy / WAVELENGTHS_M[1] - 0.19)
        + 0.31 * np.sin(2.0 * np.pi * (xx + 0.43 * yy) / WAVELENGTHS_M[1] + 1.11)
        + 0.17 * np.cos(2.0 * np.pi * (0.29 * xx - yy) / WAVELENGTHS_M[2] - 0.73)
    )
    surface -= surface.mean()
    surface /= np.max(np.abs(surface))
    return surface


def configure_surface_floor(
    model: mujoco.MjModel, profile: TerrainProfile
) -> tuple[int, SurfaceStatistics]:
    """Enable a low-complexity hfield without changing terrain contact parameters."""
    if profile.name not in ("concrete", "marble"):
        raise ValueError(f"surface study does not define {profile.name!r}")
    surface_id = model.geom(SURFACE_FLOOR_NAME).id
    hfield_id = model.hfield(HFIELD_NAME).id

    peak_to_valley = (
        CONCRETE_PEAK_TO_VALLEY_M
        if profile.name == "concrete"
        else MARBLE_PEAK_TO_VALLEY_M
    )
    normalized = _populate_surface(model, surface_id, hfield_id, peak_to_valley)
    floor_id = apply_terrain_profile(model, profile, SURFACE_FLOOR_NAME)
    heights = 0.5 * peak_to_valley * normalized
    statistics = SurfaceStatistics(
        peak_to_valley_m=peak_to_valley,
        rms_height_m=float(np.sqrt(np.mean(heights**2))),
        grid_spacing_m=GRID_SPACING_M,
        wavelengths_m=WAVELENGTHS_M,
    )
    return floor_id, statistics


def _populate_surface(
    model: mujoco.MjModel,
    surface_id: int,
    hfield_id: int,
    peak_to_valley: float,
) -> np.ndarray:
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    normalized = _normalized_irregular_surface(nrow, ncol)
    model.hfield_size[hfield_id, 2] = peak_to_valley
    model.geom_pos[surface_id, 2] = -0.5 * peak_to_valley
    address = int(model.hfield_adr[hfield_id])
    model.hfield_data[address : address + nrow * ncol] = (
        0.5 * (normalized + 1.0)
    ).ravel()
    return normalized


def configure_factorized_surface(
    model: mujoco.MjModel,
    friction_profile: TerrainProfile,
    roughness_source: str,
) -> tuple[int, SurfaceStatistics]:
    """Combine an existing friction profile with an existing hfield amplitude."""
    if roughness_source not in ("concrete", "marble"):
        raise ValueError("roughness_source must be concrete or marble")
    surface_id = model.geom(SURFACE_FLOOR_NAME).id
    hfield_id = model.hfield(HFIELD_NAME).id
    peak_to_valley = (
        CONCRETE_PEAK_TO_VALLEY_M
        if roughness_source == "concrete"
        else MARBLE_PEAK_TO_VALLEY_M
    )
    normalized = _populate_surface(model, surface_id, hfield_id, peak_to_valley)
    floor_id = apply_terrain_profile(model, friction_profile, SURFACE_FLOOR_NAME)
    heights = 0.5 * peak_to_valley * normalized
    return floor_id, SurfaceStatistics(
        peak_to_valley_m=peak_to_valley,
        rms_height_m=float(np.sqrt(np.mean(heights**2))),
        grid_spacing_m=GRID_SPACING_M,
        wavelengths_m=WAVELENGTHS_M,
    )


def surface_floor_configurator(
    model: mujoco.MjModel, profile: TerrainProfile
) -> int:
    floor_id, _ = configure_surface_floor(model, profile)
    return floor_id
