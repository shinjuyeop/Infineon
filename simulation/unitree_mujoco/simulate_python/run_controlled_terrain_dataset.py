"""Generate repeated, supported G1 contact windows for each terrain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

import config
from controlled_excitation import (
    ExcitationCondition,
    VerticalElasticBandSupport,
    apply_excitation_condition,
    find_allowed_foot_geom_ids,
    generate_excitation_conditions,
    has_nonfoot_floor_contact,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = SIMULATE_PYTHON_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "controlled_terrain_data"
DEFAULT_SEED = 20260805
RUN_COLUMNS = (
    "run_id",
    "terrain",
    "dataset_seed",
    "initial_velocity_x",
    "initial_velocity_y",
    "base_height_offset",
    "base_roll_deg",
    "base_pitch_deg",
    "support_ratio",
    "timestamp",
    *HIL_SENSOR_CHANNELS,
    "body_floor_collision",
    "body_collision_latched",
    "valid_foot_contact",
    "candidate_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled repeated G1 HIL terrain windows."
    )
    parser.add_argument(
        "--terrain", choices=(*TERRAIN_PROFILES.keys(), "all"), required=True
    )
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--support-ratio", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def settling_time(force_sum: np.ndarray, timestamps: np.ndarray) -> float | None:
    reference = float(np.median(force_sum[-10:]))
    tolerance = max(5.0, 0.1 * abs(reference))
    peak_index = int(np.argmax(force_sum))
    for index in range(peak_index, len(force_sum) - 4):
        if np.all(np.abs(force_sum[index : index + 5] - reference) <= tolerance):
            return float(timestamps[index])
    return None


def calculate_run_metrics(
    timestamps: np.ndarray,
    sensor_values: np.ndarray,
    body_collision_latched: np.ndarray,
    valid_foot_contact: np.ndarray,
    sample_rate: float,
) -> dict[str, float | int | str]:
    pre_collision = ~body_collision_latched
    forces = sensor_values[:, :4]
    force_sum = forces.sum(axis=1)
    supported_forces = forces.copy()
    supported_forces[~pre_collision] = 0.0
    supported_force_sum = supported_forces.sum(axis=1)
    sample_period = 1.0 / sample_rate
    impulse_by_point = supported_forces.sum(axis=0) * sample_period
    total_impulse = float(impulse_by_point.sum())
    if total_impulse > 0.0:
        force_distribution = impulse_by_point / total_impulse
    else:
        force_distribution = np.zeros(4)

    motion_mask = pre_collision
    accelerometer = sensor_values[motion_mask, 4:7]
    gyroscope = sensor_values[motion_mask, 7:10]
    accel_rms = float(np.sqrt(np.mean(np.sum(accelerometer**2, axis=1))))
    gyro_rms = float(np.sqrt(np.mean(np.sum(gyroscope**2, axis=1))))
    settle = settling_time(supported_force_sum, timestamps)

    candidate_count = int(np.count_nonzero(pre_collision & valid_foot_contact))
    body_collision = bool(np.any(body_collision_latched))
    return {
        "sample_count": int(sensor_values.shape[0]),
        "candidate_sample_count": candidate_count,
        "contact_duration": float(np.count_nonzero(valid_foot_contact & pre_collision) * sample_period),
        "body_collision": int(body_collision),
        "valid_run": int(not body_collision and candidate_count >= int(0.8 * len(timestamps))),
        "force_impulse": total_impulse,
        "peak_point_force": float(supported_forces.max()),
        "peak_force_sum": float(supported_force_sum.max()),
        "force_distribution_1": float(force_distribution[0]),
        "force_distribution_2": float(force_distribution[1]),
        "force_distribution_3": float(force_distribution[2]),
        "force_distribution_4": float(force_distribution[3]),
        "accel_rms": accel_rms,
        "gyro_rms": gyro_rms,
        "settling_time": "" if settle is None else settle,
        "extreme_force_outlier": int(float(supported_forces.max()) >= 1000.0),
        "extreme_accel_outlier": int(float(np.abs(accelerometer).max()) >= 100.0),
    }


def validate_run(
    timestamps: np.ndarray,
    sensor_values: np.ndarray,
    expected_samples: int,
) -> None:
    if sensor_values.shape != (expected_samples, len(HIL_SENSOR_CHANNELS)):
        raise ValueError(
            f"expected sensor shape ({expected_samples}, 10), got {sensor_values.shape}"
        )
    if timestamps.shape != (expected_samples,):
        raise ValueError(f"unexpected timestamp shape {timestamps.shape}")
    if not np.all(np.isfinite(sensor_values)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("NaN or Inf detected in controlled run")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps are not strictly monotonic")


def write_run_csv(
    output_path: Path,
    terrain: str,
    dataset_seed: int,
    condition: ExcitationCondition,
    support_ratio: float,
    timestamps: np.ndarray,
    sensor_values: np.ndarray,
    body_floor_collision: np.ndarray,
    body_collision_latched: np.ndarray,
    valid_foot_contact: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(RUN_COLUMNS)
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                (
                    condition.run_id,
                    terrain,
                    dataset_seed,
                    f"{condition.initial_velocity_x:.9f}",
                    f"{condition.initial_velocity_y:.9f}",
                    f"{condition.base_height_offset:.9f}",
                    f"{condition.base_roll_deg:.9f}",
                    f"{condition.base_pitch_deg:.9f}",
                    f"{support_ratio:.9f}",
                    f"{timestamp:.9f}",
                    *(f"{value:.9f}" for value in sensor_values[index]),
                    int(body_floor_collision[index]),
                    int(body_collision_latched[index]),
                    int(valid_foot_contact[index]),
                    int(valid_foot_contact[index] and not body_collision_latched[index]),
                )
            )


def run_window(
    terrain: str,
    condition: ExcitationCondition,
    duration: float,
    sample_rate: float,
    dataset_seed: int,
    support_ratio: float,
    output_dir: Path,
) -> dict[str, float | int | str]:
    scene_path = (SIMULATE_PYTHON_DIR / config.ROBOT_SCENE).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = config.SIMULATE_DT
    physics_rate = 1.0 / model.opt.timestep
    if sample_rate > physics_rate:
        raise ValueError("sample rate exceeds MuJoCo physics rate")
    floor_id = apply_terrain_profile(model, TERRAIN_PROFILES[terrain])
    data = mujoco.MjData(model)
    qpos_address, dof_address = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(
        model,
        data,
        qpos_address,
        dof_address,
        support_ratio=support_ratio,
    )
    reader = G1HilSensorReader(model, data)
    allowed_foot_geom_ids = find_allowed_foot_geom_ids(model)

    sample_period = 1.0 / sample_rate
    expected_samples = int(round(duration * sample_rate))
    next_sample_time = sample_period
    collision_since_sample = False
    collision_latched = False
    timestamps = []
    sensor_values = []
    body_floor_collision = []
    body_collision_latched = []
    valid_foot_contact = []

    while data.time + 1e-12 < duration:
        support.apply()
        mujoco.mj_step(model, data)
        collision_since_sample |= has_nonfoot_floor_contact(
            data, floor_id, allowed_foot_geom_ids
        )
        collision_latched |= collision_since_sample
        if data.time + 1e-12 >= next_sample_time:
            vector = reader.read_vector()
            timestamps.append(float(data.time))
            sensor_values.append(vector)
            body_floor_collision.append(collision_since_sample)
            body_collision_latched.append(collision_latched)
            valid_foot_contact.append(bool(vector[:4].sum() > 1.0))
            collision_since_sample = False
            next_sample_time += sample_period

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    sensor_array = np.asarray(sensor_values, dtype=np.float64)
    body_collision_array = np.asarray(body_floor_collision, dtype=bool)
    body_collision_latched_array = np.asarray(body_collision_latched, dtype=bool)
    valid_foot_contact_array = np.asarray(valid_foot_contact, dtype=bool)
    validate_run(timestamp_array, sensor_array, expected_samples)

    relative_path = Path(terrain) / f"{condition.run_id}.csv"
    write_run_csv(
        output_dir / relative_path,
        terrain,
        dataset_seed,
        condition,
        support_ratio,
        timestamp_array,
        sensor_array,
        body_collision_array,
        body_collision_latched_array,
        valid_foot_contact_array,
    )
    metrics = calculate_run_metrics(
        timestamp_array,
        sensor_array,
        body_collision_latched_array,
        valid_foot_contact_array,
        sample_rate,
    )
    metrics.update(
        {
            "run_id": condition.run_id,
            "terrain": terrain,
            "dataset_seed": dataset_seed,
            "initial_velocity_x": condition.initial_velocity_x,
            "initial_velocity_y": condition.initial_velocity_y,
            "base_height_offset": condition.base_height_offset,
            "base_roll_deg": condition.base_roll_deg,
            "base_pitch_deg": condition.base_pitch_deg,
            "support_ratio": support_ratio,
            "support_target_force": support.target_support_force,
            "csv_path": str(relative_path),
        }
    )
    return metrics


def write_manifest(
    manifest_path: Path, rows: list[dict[str, float | int | str]]
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    if config.ROBOT != "g1":
        raise ValueError(f'config.ROBOT must be "g1", got {config.ROBOT!r}')
    if args.duration <= 0.0 or args.sample_rate <= 0.0:
        raise ValueError("duration and sample rate must be positive")
    expected_samples = args.duration * args.sample_rate
    if not np.isclose(expected_samples, round(expected_samples)):
        raise ValueError("duration * sample_rate must be an integer")

    output_dir = args.output_dir.resolve()
    conditions = generate_excitation_conditions(args.runs, args.seed)
    terrains = TERRAIN_PROFILES.keys() if args.terrain == "all" else (args.terrain,)
    manifest_rows = []
    for terrain in terrains:
        print(f"collecting terrain={terrain} runs={args.runs}")
        for condition in conditions:
            metrics = run_window(
                terrain,
                condition,
                duration=args.duration,
                sample_rate=args.sample_rate,
                dataset_seed=args.seed,
                support_ratio=args.support_ratio,
                output_dir=output_dir,
            )
            manifest_rows.append(metrics)
            print(
                f"{terrain}/{condition.run_id} valid={metrics['valid_run']} "
                f"candidate_samples={metrics['candidate_sample_count']} "
                f"body_collision={metrics['body_collision']} "
                f"peak={metrics['peak_point_force']:.3f}N"
            )

    for terrain in terrains:
        terrain_rows = [row for row in manifest_rows if row["terrain"] == terrain]
        write_manifest(output_dir / terrain / "manifest.csv", terrain_rows)

    root_manifest_path = output_dir / "run_manifest.csv"
    combined_rows: list[dict[str, float | int | str]] = manifest_rows
    if args.terrain != "all" and root_manifest_path.exists():
        with root_manifest_path.open(newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))
        selected_terrains = set(terrains)
        preserved_rows = [
            row for row in existing_rows if row["terrain"] not in selected_terrains
        ]
        combined_rows = [*preserved_rows, *manifest_rows]
    manifest_path = write_manifest(root_manifest_path, combined_rows)
    print(f"manifest={manifest_path}")
    for terrain in terrains:
        terrain_rows = [row for row in manifest_rows if row["terrain"] == terrain]
        valid_count = sum(int(row["valid_run"]) for row in terrain_rows)
        print(f"terrain={terrain} valid_runs={valid_count}/{len(terrain_rows)}")


if __name__ == "__main__":
    main()
