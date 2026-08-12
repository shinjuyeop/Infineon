"""Plan or generate Expanded Dataset v1 with surface-family-disjoint splits."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import time

import numpy as np

from controlled_excitation import ExcitationCondition, HorizontalPulse
from expanded_terrain_dataset_v1 import (
    EXPANDED_DATASET_SEED,
    EXPANDED_SCHEMA_NAME,
    EXPANDED_SCHEMA_VERSION,
    RUNS_PER_SURFACE,
    SURFACES_PER_FAMILY,
    SURFACE_FAMILIES,
    build_candidate_manifest,
    configure_expanded_run_surface,
    estimate_execution_cost,
    make_expanded_run_specification,
    make_expanded_surface_parameters,
)
from hil_sensor import HIL_SENSOR_CHANNELS
from pulse_windows import WINDOW_PROFILES, extract_window
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window
from run_surface_sampling_rate_study import SURFACE_SCENE_PATH, write_dict_rows
from terrain_dataset_v1 import (
    DOMAIN_RANGES,
    DURATION_S,
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    TERRAIN_LABELS,
    apply_sensor_imperfections,
    fit_and_evaluate,
    is_valid_run,
    load_clean_run,
    validate_split_integrity,
)


OUTPUT_DIR = SIMULATION_DIR / "outputs" / EXPANDED_SCHEMA_NAME
SAMPLE_COUNT = 50
RATE_ABLATION_WINDOW_START_S = 0.25
CHANNEL_GROUPS = {
    "fsr_only": (0, 1, 2, 3),
    "imu_only": (4, 5, 6, 7, 8, 9),
    "fusion": tuple(range(10)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--surfaces-per-family", type=int, default=SURFACES_PER_FAMILY)
    parser.add_argument("--runs-per-surface", type=int, default=RUNS_PER_SURFACE)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        choices=(100, 500, 1000),
        default=int(SENSOR_RATE_HZ),
        help="native MuJoCo sensor read rate; 100 preserves the canonical Dataset v1 path",
    )
    parser.add_argument(
        "--window-start-s",
        type=float,
        default=None,
        help="rate-ablation window start (default: pulse onset 0.25 s for rates above 100 Hz)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run MuJoCo generation; without this flag only print the design/cost report",
    )
    return parser.parse_args()


def protocol(
    surfaces_per_family: int,
    runs_per_surface: int,
    sample_rate_hz: int = int(SENSOR_RATE_HZ),
    window_start_s: float | None = None,
) -> dict[str, object]:
    if sample_rate_hz not in (100, 500, 1000):
        raise ValueError("sample_rate_hz must be one of 100, 500, or 1000")
    steps_per_sample_float = 1.0 / (PHYSICS_TIMESTEP_S * sample_rate_hz)
    steps_per_sample = int(round(steps_per_sample_float))
    if not np.isclose(steps_per_sample_float, steps_per_sample, rtol=0.0, atol=1e-12):
        raise ValueError("sample rate must divide the 2 kHz physics rate exactly")
    canonical = sample_rate_hz == int(SENSOR_RATE_HZ) and window_start_s is None
    start_s = (
        WINDOW_PROFILES["medium_response"].start_time
        if canonical
        else RATE_ABLATION_WINDOW_START_S if window_start_s is None else window_start_s
    )
    end_s = (
        WINDOW_PROFILES["medium_response"].end_time
        if canonical
        else start_s + SAMPLE_COUNT / sample_rate_hz
    )
    cost = estimate_execution_cost(surfaces_per_family, runs_per_surface)
    return {
        "dataset_name": EXPANDED_SCHEMA_NAME,
        "schema_version": EXPANDED_SCHEMA_VERSION,
        "purpose": "Unseen procedural surface-family generalization with low-frequency MuJoCo signals.",
        "channels": HIL_SENSOR_CHANNELS,
        "label_mapping": TERRAIN_LABELS,
        "tensor_shape": ["N", SAMPLE_COUNT, 10],
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "physics_rate_hz": int(round(1.0 / PHYSICS_TIMESTEP_S)),
        "sensor_rate_hz": sample_rate_hz,
        "physics_steps_per_sample": steps_per_sample,
        "sample_count": SAMPLE_COUNT,
        "observation_duration_s": SAMPLE_COUNT / sample_rate_hz,
        "window": {
            "name": "medium_response" if canonical else "pulse_onset_rate_ablation",
            "start_s": start_s,
            "end_s": end_s,
            "interval": "left-closed/right-open",
        },
        "surface_families": [vars(family) for family in SURFACE_FAMILIES],
        "family_split": {
            split: [family.name for family in SURFACE_FAMILIES if family.split == split]
            for split in ("train", "validation", "test")
        },
        "surfaces_per_family": surfaces_per_family,
        "runs_per_surface": runs_per_surface,
        "candidate_samples": cost.candidates,
        "estimated_valid_samples": cost.expected_valid,
        "cost_basis": {
            "pilot_candidates": 1_200,
            "pilot_valid": 1_189,
            "pilot_runtime_s": 756.4941010475159,
            "pilot_storage_bytes": 110_186_372,
            "estimated_runtime_s": cost.estimated_runtime_s,
            "estimated_storage_bytes": cost.estimated_storage_bytes,
            "note": "Linear estimate from the local completed pilot; wall time depends on host load.",
        },
        "domain_ranges": {name: vars(value) for name, value in DOMAIN_RANGES.items()},
        "surface_amplitude_policy": "Terrain-specific Dataset v1 peak-to-valley ranges are unchanged; only normalized morphology family changes.",
        "pulse": {
            "magnitude_N": [76.0, 84.0],
            "start_s": [0.245, 0.255],
            "duration_s": 0.20,
            "directions": ["+X", "-X"],
        },
        "support_ratio": 0.70,
        "dataset_seed": EXPANDED_DATASET_SEED,
        "split_seed_policy": "Identical surface/physics/sensor seeds across sampling rates.",
        "excluded_features": [
            "raw high-frequency vibration PSD",
            "dominant vibration frequency",
            "timestep-sensitive micro-contact spectrum",
        ],
        "overwrite_policy": "Refuse any non-empty output directory.",
    }


def print_plan(payload: dict[str, object]) -> None:
    cost = payload["cost_basis"]
    print(json.dumps(payload, indent=2))
    print(
        "\nEstimated full run: "
        f"{payload['candidate_samples']} candidates, "
        f"~{payload['estimated_valid_samples']} valid, "
        f"~{float(cost['estimated_runtime_s']) / 60.0:.1f} min, "
        f"~{float(cost['estimated_storage_bytes']) / (1024.0**2):.1f} MiB."
    )
    print("Dry run only. Pass --execute to create the dataset.")


def extract_native_rate_window(
    timestamps: np.ndarray,
    sensors: np.ndarray,
    sample_rate_hz: int,
    window_start_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 50 consecutive native samples from an exact physics-step grid."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if timestamps.ndim != 1 or sensors.shape != (len(timestamps), len(HIL_SENSOR_CHANNELS)):
        raise ValueError("expected timestamps=(N,) and sensors=(N,10)")
    expected_times = np.arange(1, len(timestamps) + 1, dtype=np.float64) / sample_rate_hz
    if not np.allclose(timestamps, expected_times, rtol=0.0, atol=1e-9):
        raise ValueError("native samples do not match the configured physics-step grid")
    window_end_s = window_start_s + SAMPLE_COUNT / sample_rate_hz
    tolerance = np.finfo(np.float64).eps * 16.0
    indices = np.flatnonzero(
        (expected_times >= window_start_s - tolerance)
        & (expected_times < window_end_s - tolerance)
    )
    if len(indices) != SAMPLE_COUNT or not np.array_equal(
        np.diff(indices), np.ones(SAMPLE_COUNT - 1, dtype=np.int64)
    ):
        raise ValueError(
            f"rate-ablation window selected {len(indices)} non-consecutive samples, "
            f"expected {SAMPLE_COUNT}"
        )
    return timestamps[indices].copy(), sensors[indices].copy()


def main() -> None:
    args = parse_args()
    payload = protocol(
        args.surfaces_per_family,
        args.runs_per_surface,
        args.sample_rate_hz,
        args.window_start_s,
    )
    candidate_manifest = build_candidate_manifest(
        args.surfaces_per_family, args.runs_per_surface
    )
    if not args.execute:
        print_plan(payload)
        return

    default_name = (
        EXPANDED_SCHEMA_NAME
        if args.sample_rate_hz == int(SENSOR_RATE_HZ)
        else f"{EXPANDED_SCHEMA_NAME}_{args.sample_rate_hz}hz"
    )
    output = (
        args.output_dir if args.output_dir is not None else SIMULATION_DIR / "outputs" / default_name
    ).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_dict_rows(output / "candidate_manifest.csv", candidate_manifest)

    clean_windows: list[np.ndarray] = []
    noisy_windows: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    families: list[str] = []
    metadata: list[dict[str, object]] = []
    total = len(candidate_manifest)
    generation_start = time.perf_counter()
    window_start_s = float(payload["window"]["start_s"])

    for candidate in candidate_manifest:
        terrain = str(candidate["terrain_class"])
        family = str(candidate["surface_family"])
        surface_index = int(candidate["surface_index"])
        run_index = int(candidate["run_index"])
        surface = make_expanded_surface_parameters(terrain, family, surface_index)
        spec = make_expanded_run_specification(terrain, family, surface_index, run_index)
        condition = ExcitationCondition(
            run_id=spec.run_id,
            initial_velocity_x=spec.initial_velocity_x,
            initial_velocity_y=spec.initial_velocity_y,
            base_height_offset=spec.base_height_offset,
            base_roll_deg=spec.base_roll_deg,
            base_pitch_deg=spec.base_pitch_deg,
        )
        pulse = HorizontalPulse(
            spec.pulse_start,
            spec.pulse_duration,
            spec.pulse_magnitude,
            spec.pulse_direction_x,
            spec.pulse_direction_y,
        )
        raw_root = output / "raw_clean" / spec.split / spec.session_id
        pelvis_root = output / "pelvis_diagnostic" / spec.split / spec.session_id
        metrics = run_window(
            terrain,
            condition,
            DURATION_S,
            args.sample_rate_hz,
            spec.physics_seed,
            0.70,
            pulse,
            raw_root,
            session_id=spec.session_id,
            scene_path=SURFACE_SCENE_PATH,
            matched_pelvis_output_dir=pelvis_root,
            model_configurator=partial(
                configure_expanded_run_surface,
                surface=surface,
                friction=spec.friction,
            ),
            physics_timestep=PHYSICS_TIMESTEP_S,
        )
        valid = int(is_valid_run(metrics))
        raw_path = raw_root / terrain / f"{spec.run_id}.csv"
        timestamps, sensors = load_clean_run(raw_path)
        row: dict[str, object] = {
            **candidate,
            "friction_slide": spec.friction[0],
            "friction_torsional": spec.friction[1],
            "friction_rolling": spec.friction[2],
            "roughness_peak_to_valley_m": surface.peak_to_valley_m,
            "roughness_wavelengths_m": "|".join(
                f"{value:.9g}" for value in surface.wavelengths_m
            ),
            "family_spatial_scale_min_m": next(
                item.spatial_scale_m[0] for item in SURFACE_FAMILIES if item.name == family
            ),
            "family_spatial_scale_max_m": next(
                item.spatial_scale_m[1] for item in SURFACE_FAMILIES if item.name == family
            ),
            "pulse_magnitude_N": spec.pulse_magnitude,
            "pulse_direction_x": spec.pulse_direction_x,
            "pulse_start_s": spec.pulse_start,
            "pulse_duration_s": spec.pulse_duration,
            "support_ratio": 0.70,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "sensor_rate_hz": args.sample_rate_hz,
            "physics_steps_per_sample": payload["physics_steps_per_sample"],
            "sample_count": SAMPLE_COUNT,
            "observation_duration_s": payload["observation_duration_s"],
            "support_target_force_N": metrics["support_target_force"],
            "initial_velocity_x": spec.initial_velocity_x,
            "initial_velocity_y": spec.initial_velocity_y,
            "base_height_offset": spec.base_height_offset,
            "base_roll_deg": spec.base_roll_deg,
            "base_pitch_deg": spec.base_pitch_deg,
            "valid_flag": valid,
            "body_collision": metrics["body_collision"],
            "extreme_force_outlier": metrics["extreme_force_outlier"],
            "extreme_accel_outlier": metrics["extreme_accel_outlier"],
            "contact_duration_s": metrics["contact_duration"],
            "raw_clean_path": str(raw_path.relative_to(output)),
        }
        metadata.append(row)
        if valid:
            if args.sample_rate_hz == int(SENSOR_RATE_HZ) and args.window_start_s is None:
                _, clean = extract_window(
                    timestamps, sensors, WINDOW_PROFILES["medium_response"], args.sample_rate_hz
                )
            else:
                _, clean = extract_native_rate_window(
                    timestamps, sensors, args.sample_rate_hz, window_start_s
                )
            clean_windows.append(clean.astype(np.float32))
            noisy_windows.append(apply_sensor_imperfections(clean, spec.sensor_noise_seed))
            labels.append(spec.terrain_id)
            splits.append(spec.split)
            families.append(family)
        completed = int(candidate["candidate_index"]) + 1
        if completed % 50 == 0 or completed == total:
            valid_count = sum(int(item["valid_flag"]) for item in metadata)
            print(f"completed {completed}/{total}; valid={valid_count}")

    validate_split_integrity(metadata)
    x_clean = np.asarray(clean_windows, dtype=np.float32)
    x_noisy = np.asarray(noisy_windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)
    family_array = np.asarray(families)
    expected_shape = (len(y), 50, 10)
    if x_clean.shape != expected_shape or x_noisy.shape != expected_shape:
        raise ValueError(f"unexpected tensors {x_clean.shape} and {x_noisy.shape}")
    if not np.all(np.isfinite(x_clean)) or not np.all(np.isfinite(x_noisy)):
        raise ValueError("NaN/Inf in expanded dataset tensors")

    shared = {"y": y, "split": split_array, "surface_family": family_array}
    np.savez_compressed(output / "dataset_clean.npz", X=x_clean, **shared)
    np.savez_compressed(output / "dataset_noisy.npz", X=x_noisy, **shared)
    write_dict_rows(output / "run_metadata.csv", metadata)
    valid_metadata = [row for row in metadata if int(row["valid_flag"]) == 1]
    split_manifest = [
        {
            "sample_index": index,
            "terrain_class": row["terrain_class"],
            "terrain_id": row["terrain_id"],
            "split": row["split"],
            "surface_family": row["surface_family"],
            "surface_seed": row["surface_seed"],
            "session_id": row["session_id"],
            "run_id": row["run_id"],
            "run_group": row["run_group"],
            "sensor_noise_seed": row["sensor_noise_seed"],
        }
        for index, row in enumerate(valid_metadata)
    ]
    write_dict_rows(output / "split_manifest.csv", split_manifest)

    baseline_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for variant, tensor in (("clean", x_clean), ("noisy", x_noisy)):
        for group, channels in CHANNEL_GROUPS.items():
            metrics, matrix = fit_and_evaluate(tensor, y, split_array, channels)
            baseline_rows.extend(
                {"variant": variant, "sensor_group": group, **row} for row in metrics
            )
            confusion_rows.extend(
                {"variant": variant, "sensor_group": group, **row} for row in matrix
            )
    write_dict_rows(output / "random_forest_metrics.csv", baseline_rows)
    write_dict_rows(output / "random_forest_confusion.csv", confusion_rows)

    family_counts = {
        family.name: int(np.sum(family_array == family.name)) for family in SURFACE_FAMILIES
    }
    generation_time_s = time.perf_counter() - generation_start
    payload["measured_generation"] = {
        "wall_time_s": generation_time_s,
        "candidates": len(metadata),
        "valid_samples": len(y),
        "invalid_samples": len(metadata) - len(y),
    }
    (output / "protocol.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        f"{EXPANDED_SCHEMA_NAME} schema_version={EXPANDED_SCHEMA_VERSION}",
        f"X_clean={x_clean.shape} X_noisy={x_noisy.shape} y={y.shape}",
        f"valid={len(y)}/{len(metadata)} invalid={len(metadata) - len(y)}",
        f"native_sampling={args.sample_rate_hz}Hz steps_per_sample={payload['physics_steps_per_sample']} observation={float(payload['observation_duration_s']) * 1000.0:.1f}ms",
        f"generation_wall_time_s={generation_time_s:.6f}",
        "valid family counts: "
        + ", ".join(f"{name}={count}" for name, count in family_counts.items()),
        "Train/validation/test surface families are disjoint by construction and manifest validation.",
        "No high-frequency vibration or PSD feature is used.",
    ]
    (output / "dataset_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))


if __name__ == "__main__":
    main()
