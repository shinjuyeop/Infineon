"""Core schema, domain variation, sensor model, and evaluation for Dataset v1."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from hil_sensor import HIL_SENSOR_CHANNELS
from surface_profiles import HFIELD_NAME, SURFACE_FLOOR_NAME
from terrain_profiles import TerrainProfile, apply_terrain_profile


SCHEMA_NAME = "terrain_dataset_v1_pilot"
SCHEMA_VERSION = 1
TERRAIN_LABELS = {"concrete": 0, "marble": 1, "ice": 2, "sand": 3}
PHYSICS_TIMESTEP_S = 0.0005
SENSOR_RATE_HZ = 100.0
DURATION_S = 1.0
SURFACES_PER_TERRAIN = 15
RUNS_PER_SURFACE = 20
DATASET_SEED = 20260807


@dataclass(frozen=True)
class DomainRange:
    sliding_friction: tuple[float, float]
    torsional_scale: tuple[float, float]
    rolling_scale: tuple[float, float]
    roughness_ptv_m: tuple[float, float]
    wavelength_scale: tuple[float, float] = (0.90, 1.10)


DOMAIN_RANGES = {
    "concrete": DomainRange((0.80, 1.10), (0.8, 1.2), (0.8, 1.2), (0.00030, 0.00050)),
    "marble": DomainRange((0.35, 0.60), (0.8, 1.2), (0.8, 1.2), (0.00001, 0.000035)),
    "ice": DomainRange((0.03, 0.12), (0.8, 1.2), (0.8, 1.2), (0.000005, 0.000020)),
    "sand": DomainRange((0.55, 0.85), (0.8, 1.2), (0.8, 1.2), (0.00020, 0.00055)),
}
BASE_WAVELENGTHS_M = (0.040, 0.065, 0.110)


@dataclass(frozen=True)
class SurfaceParameters:
    terrain: str
    surface_seed: int
    peak_to_valley_m: float
    wavelengths_m: tuple[float, float, float]
    phases_rad: tuple[float, float, float]
    weights: tuple[float, float, float]


@dataclass(frozen=True)
class RunSpecification:
    terrain: str
    terrain_id: int
    split: str
    surface_index: int
    surface_seed: int
    session_id: str
    run_id: str
    run_group: str
    physics_seed: int
    sensor_noise_seed: int
    friction: tuple[float, float, float]
    initial_velocity_x: float
    initial_velocity_y: float
    base_height_offset: float
    base_roll_deg: float
    base_pitch_deg: float
    pulse_magnitude: float
    pulse_start: float
    pulse_duration: float
    pulse_direction_x: float
    pulse_direction_y: float


def split_for_surface(surface_index: int) -> str:
    if not 0 <= surface_index < SURFACES_PER_TERRAIN:
        raise ValueError("surface index is outside the pilot design")
    if surface_index < 9:
        return "train"
    if surface_index < 12:
        return "validation"
    return "test"


def make_surface_parameters(terrain: str, surface_seed: int) -> SurfaceParameters:
    domain = DOMAIN_RANGES[terrain]
    rng = np.random.default_rng(surface_seed)
    scale = rng.uniform(*domain.wavelength_scale, size=3)
    raw_weights = np.asarray((0.52, 0.31, 0.17)) * rng.uniform(0.85, 1.15, size=3)
    weights = raw_weights / raw_weights.sum()
    return SurfaceParameters(
        terrain=terrain,
        surface_seed=surface_seed,
        peak_to_valley_m=float(rng.uniform(*domain.roughness_ptv_m)),
        wavelengths_m=tuple(float(x) for x in np.asarray(BASE_WAVELENGTHS_M) * scale),
        phases_rad=tuple(float(x) for x in rng.uniform(-np.pi, np.pi, size=3)),
        weights=tuple(float(x) for x in weights),
    )


def make_run_specification(
    terrain: str, surface_index: int, run_index: int
) -> RunSpecification:
    terrain_id = TERRAIN_LABELS[terrain]
    surface_seed = DATASET_SEED + 10_000 * terrain_id + 100 * surface_index
    physics_seed = surface_seed + run_index + 1
    sensor_noise_seed = physics_seed + 1_000_000
    rng = np.random.default_rng(physics_seed)
    base = DOMAIN_RANGES[terrain]
    from terrain_profiles import TERRAIN_PROFILES

    center = TERRAIN_PROFILES[terrain].friction
    friction = (
        float(rng.uniform(*base.sliding_friction)),
        float(center[1] * rng.uniform(*base.torsional_scale)),
        float(center[2] * rng.uniform(*base.rolling_scale)),
    )
    split = split_for_surface(surface_index)
    direction_x = 1.0 if (surface_index + run_index) % 2 == 0 else -1.0
    run_id = f"{terrain}_s{surface_index:02d}_r{run_index:03d}"
    return RunSpecification(
        terrain=terrain,
        terrain_id=terrain_id,
        split=split,
        surface_index=surface_index,
        surface_seed=surface_seed,
        session_id=f"{terrain}_{split}_surface_{surface_index:02d}",
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


def normalized_surface(model: mujoco.MjModel, surface: SurfaceParameters) -> np.ndarray:
    hfield_id = model.hfield(HFIELD_NAME).id
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    x = np.linspace(-0.6, 0.6, ncol)
    y = np.linspace(-0.4, 0.4, nrow)
    xx, yy = np.meshgrid(x, y)
    w1, w2, w3 = surface.wavelengths_m
    p1, p2, p3 = surface.phases_rad
    a1, a2, a3 = surface.weights
    values = (
        a1 * np.sin(2.0 * np.pi * xx / w1 + p1) * np.cos(2.0 * np.pi * yy / w2 - p2)
        + a2 * np.sin(2.0 * np.pi * (xx + 0.43 * yy) / w2 + p2)
        + a3 * np.cos(2.0 * np.pi * (0.29 * xx - yy) / w3 + p3)
    )
    values -= values.mean()
    values /= max(float(np.max(np.abs(values))), 1e-12)
    return values


def configure_run_surface(
    model: mujoco.MjModel,
    profile: TerrainProfile,
    surface: SurfaceParameters,
    friction: tuple[float, float, float],
) -> int:
    surface_id = model.geom(SURFACE_FLOOR_NAME).id
    hfield_id = model.hfield(HFIELD_NAME).id
    values = normalized_surface(model, surface)
    model.hfield_size[hfield_id, 2] = surface.peak_to_valley_m
    model.geom_pos[surface_id, 2] = -0.5 * surface.peak_to_valley_m
    address = int(model.hfield_adr[hfield_id])
    model.hfield_data[address : address + values.size] = (0.5 * (values + 1.0)).ravel()
    varied_profile = replace(profile, friction=friction)
    return apply_terrain_profile(model, varied_profile, SURFACE_FLOOR_NAME)


def load_clean_run(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sensors = np.asarray(
        [[float(row[name]) for name in HIL_SENSOR_CHANNELS] for row in rows],
        dtype=np.float64,
    )
    return timestamps, sensors


def apply_sensor_imperfections(clean: np.ndarray, seed: int) -> np.ndarray:
    """Apply conservative per-run pilot assumptions; never mutate clean input."""
    if clean.ndim != 2 or clean.shape[1] != 10 or not np.all(np.isfinite(clean)):
        raise ValueError("clean signal must be finite with shape (T, 10)")
    rng = np.random.default_rng(seed)
    noisy = clean.astype(np.float64, copy=True)
    noisy[:, :4] = (
        noisy[:, :4] * rng.uniform(0.985, 1.015, size=4)
        + rng.uniform(-0.5, 0.5, size=4)
        + rng.normal(0.0, 0.15, size=(len(clean), 4))
    )
    noisy[:, :4] = np.clip(noisy[:, :4], 0.0, 250.0)
    noisy[:, 4:7] = (
        noisy[:, 4:7] * rng.uniform(0.995, 1.005, size=3)
        + rng.normal(0.0, 0.03, size=3)
        + rng.normal(0.0, 0.02, size=(len(clean), 3))
    )
    noisy[:, 7:10] = (
        noisy[:, 7:10] * rng.uniform(0.995, 1.005, size=3)
        + rng.normal(0.0, 0.002, size=3)
        + rng.normal(0.0, 0.001, size=(len(clean), 3))
    )
    return noisy.astype(np.float32)


def is_valid_run(metrics: dict[str, float | int | str]) -> bool:
    """Reuse the established run validity and extreme-outlier decisions."""
    return bool(
        int(metrics["valid_run"]) == 1
        and int(metrics["body_collision"]) == 0
        and int(metrics["extreme_force_outlier"]) == 0
        and int(metrics["extreme_accel_outlier"]) == 0
    )


def validate_split_integrity(rows: list[dict[str, object]]) -> None:
    valid = [row for row in rows if int(row["valid_flag"]) == 1]
    for key in ("surface_seed", "session_id", "run_group"):
        owners: dict[object, str] = {}
        for row in valid:
            value, split = row[key], str(row["split"])
            if value in owners and owners[value] != split:
                raise ValueError(f"{key} leaks across {owners[value]} and {split}")
            owners[value] = split
    for terrain in TERRAIN_LABELS:
        split_surfaces = {
            split: {row["surface_seed"] for row in valid if row["terrain_class"] == terrain and row["split"] == split}
            for split in ("train", "validation", "test")
        }
        if any(split_surfaces[a] & split_surfaces[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"surface realization leakage for {terrain}")


def statistical_features(x: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    selected = x[:, :, channels].astype(np.float64)
    features = (
        selected.mean(axis=1),
        selected.std(axis=1),
        np.sqrt(np.mean(selected**2, axis=1)),
        selected.min(axis=1),
        selected.max(axis=1),
        np.ptp(selected, axis=1),
    )
    return np.concatenate(features, axis=1)


def separation_ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    combined = np.vstack((a, b))
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    aa, bb = a / scale, b / scale
    distance = float(np.linalg.norm(aa.mean(axis=0) - bb.mean(axis=0)))
    spread = float(np.sqrt(np.mean(np.var(aa, axis=0) + np.var(bb, axis=0))))
    return distance, spread, distance / max(spread, 1e-12)


def fit_and_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    channels: tuple[int, ...],
    seed: int = DATASET_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features = statistical_features(x, channels)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    train = split == "train"
    model.fit(features[train], y[train])
    metric_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    names = tuple(TERRAIN_LABELS)
    for split_name in ("train", "validation", "test"):
        mask = split == split_name
        prediction = model.predict(features[mask])
        precision, recall, f1, support = precision_recall_fscore_support(
            y[mask], prediction, labels=np.arange(4), zero_division=0
        )
        metric_rows.append(
            {"split": split_name, "class": "overall", "accuracy": float(np.mean(prediction == y[mask])), "precision": "", "recall": "", "f1": "", "support": int(mask.sum())}
        )
        for index, name in enumerate(names):
            metric_rows.append(
                {"split": split_name, "class": name, "accuracy": "", "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            )
        if split_name == "test":
            matrix = confusion_matrix(y[mask], prediction, labels=np.arange(4))
            for actual, actual_name in enumerate(names):
                for predicted, predicted_name in enumerate(names):
                    matrix_rows.append({"actual": actual_name, "predicted": predicted_name, "count": int(matrix[actual, predicted])})
    return metric_rows, matrix_rows
