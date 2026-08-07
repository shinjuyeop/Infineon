"""Surface-family design and cost model for Expanded Dataset v1."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from surface_profiles import HFIELD_NAME, SURFACE_FLOOR_NAME
from terrain_dataset_v1 import (
    DATASET_SEED,
    DOMAIN_RANGES,
    RunSpecification,
    TERRAIN_LABELS,
    make_surface_parameters,
)
from terrain_profiles import TERRAIN_PROFILES, TerrainProfile, apply_terrain_profile


EXPANDED_SCHEMA_NAME = "terrain_dataset_v1_expanded"
EXPANDED_SCHEMA_VERSION = 1
EXPANDED_DATASET_SEED = DATASET_SEED + 100_000_000
SURFACES_PER_FAMILY = 8
RUNS_PER_SURFACE = 20
PILOT_CANDIDATES = 1_200
PILOT_VALID = 1_189
PILOT_RUNTIME_S = 756.4941010475159
PILOT_STORAGE_BYTES = 110_186_372


@dataclass(frozen=True)
class SurfaceFamily:
    name: str
    split: str
    description: str
    spatial_scale_m: tuple[float, float]


SURFACE_FAMILIES = (
    SurfaceFamily("multisine", "train", "Three bounded oblique sinusoidal components.", (0.036, 0.121)),
    SurfaceFamily("filtered_random", "train", "Deterministic low-pass random height field.", (0.080, 0.250)),
    SurfaceFamily("sparse_aggregate", "train", "Sparse shallow rounded bumps and depressions.", (0.050, 0.120)),
    SurfaceFamily("crosshatch", "validation", "Two smooth crossing ridge directions.", (0.058, 0.121)),
    SurfaceFamily("rounded_ridges", "validation", "Modulated broad rounded ridges.", (0.058, 0.363)),
    SurfaceFamily("warped_multisine", "test", "Multisine morphology with smooth coordinate warping.", (0.036, 0.363)),
    SurfaceFamily("smooth_random_patches", "test", "Held-out broad random Fourier patches.", (0.075, 0.180)),
)
FAMILY_BY_NAME = {family.name: family for family in SURFACE_FAMILIES}


@dataclass(frozen=True)
class ExpandedSurfaceParameters:
    terrain: str
    family: str
    split: str
    surface_index: int
    surface_seed: int
    peak_to_valley_m: float
    wavelengths_m: tuple[float, float, float]
    phases_rad: tuple[float, float, float]
    weights: tuple[float, float, float]


@dataclass(frozen=True)
class ExecutionCostEstimate:
    candidates: int
    expected_valid: int
    estimated_runtime_s: float
    estimated_storage_bytes: int
    pilot_valid_rate: float


def family_for_name(name: str) -> SurfaceFamily:
    try:
        return FAMILY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown surface family {name!r}") from exc


def candidate_count(
    surfaces_per_family: int = SURFACES_PER_FAMILY,
    runs_per_surface: int = RUNS_PER_SURFACE,
) -> int:
    if surfaces_per_family <= 0 or runs_per_surface <= 0:
        raise ValueError("surface and run counts must be positive")
    return (
        len(TERRAIN_LABELS)
        * len(SURFACE_FAMILIES)
        * surfaces_per_family
        * runs_per_surface
    )


def estimate_execution_cost(
    surfaces_per_family: int = SURFACES_PER_FAMILY,
    runs_per_surface: int = RUNS_PER_SURFACE,
) -> ExecutionCostEstimate:
    candidates = candidate_count(surfaces_per_family, runs_per_surface)
    scale = candidates / PILOT_CANDIDATES
    valid_rate = PILOT_VALID / PILOT_CANDIDATES
    return ExecutionCostEstimate(
        candidates=candidates,
        expected_valid=round(candidates * valid_rate),
        estimated_runtime_s=PILOT_RUNTIME_S * scale,
        estimated_storage_bytes=round(PILOT_STORAGE_BYTES * scale),
        pilot_valid_rate=valid_rate,
    )


def make_expanded_surface_parameters(
    terrain: str,
    family_name: str,
    surface_index: int,
) -> ExpandedSurfaceParameters:
    if terrain not in TERRAIN_LABELS:
        raise ValueError(f"unknown terrain {terrain!r}")
    family = family_for_name(family_name)
    if surface_index < 0:
        raise ValueError("surface_index must be non-negative")
    family_index = tuple(item.name for item in SURFACE_FAMILIES).index(family_name)
    surface_seed = (
        EXPANDED_DATASET_SEED
        + 1_000_000 * TERRAIN_LABELS[terrain]
        + 10_000 * family_index
        + 100 * surface_index
    )
    base = make_surface_parameters(terrain, surface_seed)
    return ExpandedSurfaceParameters(
        terrain=terrain,
        family=family.name,
        split=family.split,
        surface_index=surface_index,
        surface_seed=surface_seed,
        peak_to_valley_m=base.peak_to_valley_m,
        wavelengths_m=base.wavelengths_m,
        phases_rad=base.phases_rad,
        weights=base.weights,
    )


def make_expanded_run_specification(
    terrain: str,
    family_name: str,
    surface_index: int,
    run_index: int,
) -> RunSpecification:
    if run_index < 0:
        raise ValueError("run_index must be non-negative")
    surface = make_expanded_surface_parameters(terrain, family_name, surface_index)
    physics_seed = surface.surface_seed + run_index + 1
    sensor_noise_seed = physics_seed + 1_000_000_000
    rng = np.random.default_rng(physics_seed)
    domain = DOMAIN_RANGES[terrain]
    center = TERRAIN_PROFILES[terrain].friction
    friction = (
        float(rng.uniform(*domain.sliding_friction)),
        float(center[1] * rng.uniform(*domain.torsional_scale)),
        float(center[2] * rng.uniform(*domain.rolling_scale)),
    )
    family_index = tuple(item.name for item in SURFACE_FAMILIES).index(family_name)
    direction_x = 1.0 if (family_index + surface_index + run_index) % 2 == 0 else -1.0
    run_id = f"{terrain}_{family_name}_s{surface_index:02d}_r{run_index:03d}"
    session_id = f"{terrain}_{surface.split}_{family_name}_surface_{surface_index:02d}"
    return RunSpecification(
        terrain=terrain,
        terrain_id=TERRAIN_LABELS[terrain],
        split=surface.split,
        surface_index=surface_index,
        surface_seed=surface.surface_seed,
        session_id=session_id,
        run_id=run_id,
        run_group=run_id,
        physics_seed=physics_seed,
        sensor_noise_seed=sensor_noise_seed,
        friction=friction,
        initial_velocity_x=float(rng.uniform(-0.02, 0.02)),
        initial_velocity_y=float(rng.uniform(-0.01, 0.01)),
        base_height_offset=float(rng.uniform(0.0, 0.008)),
        base_roll_deg=float(rng.uniform(-0.4, 0.4)),
        base_pitch_deg=float(rng.uniform(-0.4, 0.4)),
        pulse_magnitude=float(rng.uniform(76.0, 84.0)),
        pulse_start=float(rng.uniform(0.245, 0.255)),
        pulse_duration=0.20,
        pulse_direction_x=direction_x,
        pulse_direction_y=0.0,
    )


def build_candidate_manifest(
    surfaces_per_family: int = SURFACES_PER_FAMILY,
    runs_per_surface: int = RUNS_PER_SURFACE,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for terrain in TERRAIN_LABELS:
        for family in SURFACE_FAMILIES:
            for surface_index in range(surfaces_per_family):
                for run_index in range(runs_per_surface):
                    spec = make_expanded_run_specification(
                        terrain, family.name, surface_index, run_index
                    )
                    rows.append(
                        {
                            "candidate_index": len(rows),
                            "terrain_class": terrain,
                            "terrain_id": spec.terrain_id,
                            "surface_family": family.name,
                            "split": family.split,
                            "surface_index": surface_index,
                            "run_index": run_index,
                            "surface_seed": spec.surface_seed,
                            "session_id": spec.session_id,
                            "run_id": spec.run_id,
                            "run_group": spec.run_group,
                            "physics_seed": spec.physics_seed,
                            "sensor_noise_seed": spec.sensor_noise_seed,
                        }
                    )
    validate_family_manifest(rows)
    return rows


def validate_family_manifest(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("expanded manifest is empty")
    family_owners: dict[object, str] = {}
    for row in rows:
        family = row["surface_family"]
        split = str(row["split"])
        expected = family_for_name(str(family)).split
        if split != expected:
            raise ValueError(f"family {family} assigned to {split}, expected {expected}")
        previous = family_owners.setdefault(family, split)
        if previous != split:
            raise ValueError(f"surface family {family} leaks across splits")
    for key in ("surface_seed", "session_id", "run_group"):
        owners: dict[object, str] = {}
        for row in rows:
            value, split = row[key], str(row["split"])
            previous = owners.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{key} leaks across {previous} and {split}")
    expected_pairs = {
        (terrain, family.name) for terrain in TERRAIN_LABELS for family in SURFACE_FAMILIES
    }
    actual_pairs = {(str(row["terrain_class"]), str(row["surface_family"])) for row in rows}
    if actual_pairs != expected_pairs:
        raise ValueError("manifest does not cover every terrain/family pair")
    pair_counts = {
        pair: sum(
            row["terrain_class"] == pair[0] and row["surface_family"] == pair[1]
            for row in rows
        )
        for pair in expected_pairs
    }
    if len(set(pair_counts.values())) != 1:
        raise ValueError("terrain/family candidate counts are imbalanced")


def _coordinates(nrow: int, ncol: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-0.6, 0.6, ncol)
    y = np.linspace(-0.4, 0.4, nrow)
    return np.meshgrid(x, y)


def _resize_bilinear(values: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    source_y = np.linspace(0.0, 1.0, values.shape[0])
    source_x = np.linspace(0.0, 1.0, values.shape[1])
    target_y = np.linspace(0.0, 1.0, nrow)
    target_x = np.linspace(0.0, 1.0, ncol)
    along_x = np.vstack([np.interp(target_x, source_x, row) for row in values])
    return np.vstack(
        [np.interp(target_y, source_y, along_x[:, column]) for column in range(ncol)]
    ).T


def normalized_family_surface(
    nrow: int,
    ncol: int,
    surface: ExpandedSurfaceParameters,
) -> np.ndarray:
    if nrow < 2 or ncol < 2:
        raise ValueError("hfield must have at least two rows and columns")
    xx, yy = _coordinates(nrow, ncol)
    rng = np.random.default_rng(surface.surface_seed + 17)
    w1, w2, w3 = surface.wavelengths_m
    p1, p2, p3 = surface.phases_rad
    a1, a2, a3 = surface.weights

    if surface.family == "multisine":
        values = (
            a1 * np.sin(2.0 * np.pi * xx / w1 + p1) * np.cos(2.0 * np.pi * yy / w2 - p2)
            + a2 * np.sin(2.0 * np.pi * (xx + 0.43 * yy) / w2 + p2)
            + a3 * np.cos(2.0 * np.pi * (0.29 * xx - yy) / w3 + p3)
        )
    elif surface.family == "filtered_random":
        coarse = rng.normal(size=(11, 15))
        for _ in range(3):
            coarse = (
                4.0 * coarse
                + np.roll(coarse, 1, axis=0)
                + np.roll(coarse, -1, axis=0)
                + np.roll(coarse, 1, axis=1)
                + np.roll(coarse, -1, axis=1)
            ) / 8.0
        values = _resize_bilinear(coarse, nrow, ncol)
    elif surface.family == "sparse_aggregate":
        values = np.zeros_like(xx)
        for _ in range(18):
            center_x = rng.uniform(-0.55, 0.55)
            center_y = rng.uniform(-0.35, 0.35)
            sigma = rng.uniform(0.025, 0.060)
            sign = rng.choice((-1.0, 1.0))
            values += sign * rng.uniform(0.6, 1.0) * np.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
            )
    elif surface.family == "crosshatch":
        values = 0.55 * np.sin(2.0 * np.pi * (xx + 0.55 * yy) / w2 + p1)
        values += 0.45 * np.sin(2.0 * np.pi * (xx - 0.70 * yy) / w3 + p2)
    elif surface.family == "rounded_ridges":
        carrier = np.sin(2.0 * np.pi * (0.25 * xx + yy) / w3 + p1)
        modulation = 0.65 + 0.35 * np.cos(2.0 * np.pi * xx / (3.0 * w3) + p2)
        values = modulation * carrier + 0.2 * np.sin(2.0 * np.pi * xx / w2 + p3)
    elif surface.family == "warped_multisine":
        warp_x = xx + 0.18 * w3 * np.sin(2.0 * np.pi * yy / (2.5 * w3) + p3)
        warp_y = yy + 0.15 * w3 * np.cos(2.0 * np.pi * xx / (3.0 * w3) - p2)
        values = a1 * np.sin(2.0 * np.pi * warp_x / w1 + p1)
        values += a2 * np.cos(2.0 * np.pi * (warp_x + 0.4 * warp_y) / w2 + p2)
        values += a3 * np.sin(2.0 * np.pi * warp_y / w3 + p3)
    elif surface.family == "smooth_random_patches":
        values = np.zeros_like(xx)
        for _ in range(10):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            wavelength = rng.uniform(0.075, 0.180)
            phase = rng.uniform(-np.pi, np.pi)
            coordinate = np.cos(angle) * xx + np.sin(angle) * yy
            values += rng.uniform(0.3, 1.0) * np.sin(2.0 * np.pi * coordinate / wavelength + phase)
    else:  # guarded by make_expanded_surface_parameters, retained for direct construction
        raise ValueError(f"unsupported family {surface.family!r}")

    values = np.asarray(values, dtype=np.float64)
    values -= values.mean()
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError(f"degenerate surface for {surface.family}")
    return values / scale


def configure_expanded_run_surface(
    model: mujoco.MjModel,
    profile: TerrainProfile,
    surface: ExpandedSurfaceParameters,
    friction: tuple[float, float, float],
) -> int:
    surface_id = model.geom(SURFACE_FLOOR_NAME).id
    hfield_id = model.hfield(HFIELD_NAME).id
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    values = normalized_family_surface(nrow, ncol, surface)
    model.hfield_size[hfield_id, 2] = surface.peak_to_valley_m
    model.geom_pos[surface_id, 2] = -0.5 * surface.peak_to_valley_m
    address = int(model.hfield_adr[hfield_id])
    model.hfield_data[address : address + values.size] = (0.5 * (values + 1.0)).ravel()
    return apply_terrain_profile(
        model, replace(profile, friction=friction), SURFACE_FLOOR_NAME
    )
