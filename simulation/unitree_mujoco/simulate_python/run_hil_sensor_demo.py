"""Generate deterministic G1 HIL virtual-sensor CSV data by terrain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

import config
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from terrain_profiles import (
    TERRAIN_PROFILES,
    TerrainProfile,
    apply_terrain_profile,
    describe_profile,
)


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = SIMULATE_PYTHON_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "terrain_data"
CSV_COLUMNS = ("timestamp", *HIL_SENSOR_CHANNELS, "terrain")


def parse_args() -> argparse.Namespace:
    terrain_choices = (*TERRAIN_PROFILES.keys(), "all")
    parser = argparse.ArgumentParser(
        description="Generate G1 10-channel virtual-sensor terrain data."
    )
    parser.add_argument("--terrain", choices=terrain_choices, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument(
        "--initial-velocity-x",
        type=float,
        default=0.35,
        help="identical initial pelvis X velocity used to excite friction",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_samples(
    timestamps: np.ndarray,
    samples: np.ndarray,
    duration: float,
    sample_rate: float,
) -> None:
    if samples.ndim != 2 or samples.shape[1] != len(HIL_SENSOR_CHANNELS):
        raise ValueError(f"expected N x 10 samples, got {samples.shape}")
    if timestamps.shape != (samples.shape[0],):
        raise ValueError("timestamp and sample counts differ")
    if samples.shape[0] == 0:
        raise ValueError("no samples were collected")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("NaN or Inf detected in dataset")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps are not strictly monotonic")

    expected_count = int(round(duration * sample_rate))
    if abs(samples.shape[0] - expected_count) > 1:
        raise ValueError(
            f"sample count {samples.shape[0]} differs from expected {expected_count}"
        )


def print_summary(
    profile: TerrainProfile,
    timestamps: np.ndarray,
    samples: np.ndarray,
    duration: float,
    sample_rate: float,
) -> None:
    print(describe_profile(profile))
    print(
        f"duration={duration:.3f}s samples={samples.shape[0]} "
        f"sample_rate={sample_rate:.1f}Hz "
        f"first_timestamp={timestamps[0]:.3f}s "
        f"last_timestamp={timestamps[-1]:.3f}s"
    )
    print("channel,min,max,mean,std")
    for index, channel in enumerate(HIL_SENSOR_CHANNELS):
        values = samples[:, index]
        print(
            f"{channel},{values.min():.6f},{values.max():.6f},"
            f"{values.mean():.6f},{values.std():.6f}"
        )


def write_csv(
    output_path: Path,
    profile: TerrainProfile,
    timestamps: np.ndarray,
    samples: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        for timestamp, vector in zip(timestamps, samples):
            writer.writerow(
                [f"{timestamp:.9f}", *(f"{value:.9f}" for value in vector), profile.name]
            )


def collect_terrain(
    profile: TerrainProfile,
    duration: float,
    sample_rate: float,
    initial_velocity_x: float,
    output_dir: Path,
) -> Path:
    if config.ROBOT != "g1":
        raise ValueError(f'config.ROBOT must be "g1", got {config.ROBOT!r}')
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample rate must be greater than zero")

    scene_path = (SIMULATE_PYTHON_DIR / config.ROBOT_SCENE).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = config.SIMULATE_DT
    physics_rate = 1.0 / model.opt.timestep
    if sample_rate > physics_rate:
        raise ValueError(
            f"sample rate {sample_rate}Hz exceeds physics rate {physics_rate}Hz"
        )

    floor_id = apply_terrain_profile(model, profile)
    data = mujoco.MjData(model)
    base_joint_id = model.joint("floating_base_joint").id
    if model.jnt_type[base_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("G1 floating_base_joint is not a free joint")
    base_dof_address = model.jnt_dofadr[base_joint_id]
    data.qvel[base_dof_address] = initial_velocity_x
    mujoco.mj_forward(model, data)
    reader = G1HilSensorReader(model, data)

    print(
        f"collecting terrain={profile.name} scene={scene_path} floor_id={floor_id} "
        f"physics_rate={physics_rate:.1f}Hz initial_velocity_x={initial_velocity_x:.3f}m/s"
    )

    sample_period = 1.0 / sample_rate
    next_sample_time = sample_period
    timestamps: list[float] = []
    samples: list[np.ndarray] = []

    while data.time + 1e-12 < duration:
        mujoco.mj_step(model, data)
        if data.time + 1e-12 >= next_sample_time:
            timestamps.append(float(data.time))
            samples.append(reader.read_vector())
            next_sample_time += sample_period

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    sample_array = np.asarray(samples, dtype=np.float64)
    validate_samples(timestamp_array, sample_array, duration, sample_rate)
    print(
        "validation=PASS channels=10 finite=True "
        "timestamps_strictly_monotonic=True"
    )

    output_path = output_dir.resolve() / f"{profile.name}.csv"
    write_csv(output_path, profile, timestamp_array, sample_array)
    print_summary(profile, timestamp_array, sample_array, duration, sample_rate)
    print(f"csv={output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    terrain_names = (
        TERRAIN_PROFILES.keys() if args.terrain == "all" else (args.terrain,)
    )
    for terrain_name in terrain_names:
        collect_terrain(
            TERRAIN_PROFILES[terrain_name],
            duration=args.duration,
            sample_rate=args.sample_rate,
            initial_velocity_x=args.initial_velocity_x,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
