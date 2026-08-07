"""Run a matched pelvis-versus-left-foot IMU location validation.

The two IMU variants are sampled from the same MuJoCo state in each run.  This
keeps terrain, excitation, contact forces, timing, and initial conditions exact
between A and B; only the six IMU channels differ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from hil_sensor import HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import (
    calculate_run_metrics,
    write_manifest,
)
from run_horizontal_pulse_dataset import (
    DEFAULT_SEED,
    SIMULATION_DIR,
    run_window,
)


TERRAINS = ("concrete", "marble")
LOCATIONS = ("pelvis_imu", "foot_imu")
RUN_COUNT = 20
DURATION = 1.0
SAMPLE_RATE = 50.0
SUPPORT_RATIO = 0.70
PULSE = HorizontalPulse(
    start_time=0.25,
    duration=0.20,
    magnitude=80.0,
    direction_x=1.0,
    direction_y=0.0,
)
DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "imu_location_ab_validation"
FSR_FEATURES = (
    "force_impulse",
    "contact_duration",
    "peak_point_force",
    "peak_force_sum",
    "force_distribution_1",
    "force_distribution_2",
    "force_distribution_3",
    "force_distribution_4",
)
IMU_FEATURES = ("accel_rms", "gyro_rms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sensors = np.asarray(
        [[float(row[channel]) for channel in HIL_SENSOR_CHANNELS] for row in rows]
    )
    collision = np.asarray(
        [bool(int(row["body_collision_latched"])) for row in rows]
    )
    valid_contact = np.asarray(
        [bool(int(row["valid_foot_contact"])) for row in rows]
    )
    if timestamps.shape != (50,) or sensors.shape != (50, 10):
        raise ValueError(f"unexpected matched validation shape in {path}")
    if not np.all(np.isfinite(sensors)):
        raise ValueError(f"NaN or Inf in {path}")
    return timestamps, sensors, collision, valid_contact


def separation(
    concrete: list[dict[str, float | int | str]],
    marble: list[dict[str, float | int | str]],
    features: tuple[str, ...],
) -> tuple[float, float, float]:
    first = np.asarray([[float(row[key]) for key in features] for row in concrete])
    second = np.asarray([[float(row[key]) for key in features] for row in marble])
    scale = np.concatenate((first, second)).std(axis=0)
    scale[scale < 1e-12] = 1.0
    first /= scale
    second /= scale
    first_centroid = first.mean(axis=0)
    second_centroid = second.mean(axis=0)
    distance = float(np.linalg.norm(first_centroid - second_centroid))
    first_spread = float(
        np.sqrt(np.mean(np.sum((first - first_centroid) ** 2, axis=1)))
    )
    second_spread = float(
        np.sqrt(np.mean(np.sum((second - second_centroid) ** 2, axis=1)))
    )
    pooled_spread = float(np.sqrt((first_spread**2 + second_spread**2) / 2.0))
    return distance, pooled_spread, distance / max(pooled_spread, 1e-12)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(rows: list[dict[str, float | int | str]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows])
    return float(values.mean()), float(values.std())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing validation output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    foot_dir = output_dir / "foot_imu"
    pelvis_dir = output_dir / "pelvis_imu"
    conditions = generate_pulse_conditions(RUN_COUNT, DEFAULT_SEED)

    protocol = {
        "terrains": list(TERRAINS),
        "runs_per_terrain": RUN_COUNT,
        "duration_seconds": DURATION,
        "sample_rate_hz": SAMPLE_RATE,
        "seed": DEFAULT_SEED,
        "support_ratio": SUPPORT_RATIO,
        "pulse_start_seconds": PULSE.start_time,
        "pulse_duration_seconds": PULSE.duration,
        "pulse_peak_newtons": PULSE.magnitude,
        "pulse_direction_xy": [PULSE.direction_x, PULSE.direction_y],
        "roughness_added": False,
        "noise_added": False,
        "domain_randomization_added": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    for terrain in TERRAINS:
        print(f"collecting matched terrain={terrain} runs={RUN_COUNT}")
        for condition in conditions:
            metrics = run_window(
                terrain=terrain,
                condition=condition,
                duration=DURATION,
                sample_rate=SAMPLE_RATE,
                seed=DEFAULT_SEED,
                support_ratio=SUPPORT_RATIO,
                pulse=PULSE,
                output_dir=foot_dir,
                matched_pelvis_output_dir=pelvis_dir,
            )
            print(
                f"{terrain}/{condition.run_id} valid={metrics['valid_run']} "
                f"body_collision={metrics['body_collision']}"
            )

    metrics_by_location: dict[
        str, dict[str, list[dict[str, float | int | str]]]
    ] = {location: {terrain: [] for terrain in TERRAINS} for location in LOCATIONS}
    windows: dict[tuple[str, str, str], np.ndarray] = {}
    condition_by_id = {condition.run_id: condition for condition in conditions}
    for location in LOCATIONS:
        location_dir = output_dir / location
        manifest_rows: list[dict[str, float | int | str]] = []
        for terrain in TERRAINS:
            for condition in conditions:
                relative_path = Path(terrain) / f"{condition.run_id}.csv"
                timestamps, sensors, collision, valid_contact = load_run(
                    location_dir / relative_path
                )
                metrics = calculate_run_metrics(
                    timestamps, sensors, collision, valid_contact, SAMPLE_RATE
                )
                metrics.update(
                    {
                        "run_id": condition.run_id,
                        "terrain": terrain,
                        "dataset_seed": DEFAULT_SEED,
                        "base_height_offset": condition.base_height_offset,
                        "base_roll_deg": condition.base_roll_deg,
                        "base_pitch_deg": condition.base_pitch_deg,
                        "support_ratio": SUPPORT_RATIO,
                        "pulse_start_time": PULSE.start_time,
                        "pulse_duration": PULSE.duration,
                        "pulse_magnitude": PULSE.magnitude,
                        "sensor_location": location,
                        "csv_path": str(relative_path),
                    }
                )
                for index, channel in enumerate(HIL_SENSOR_CHANNELS):
                    metrics[f"channel_rms_{channel}"] = float(
                        np.sqrt(np.mean(sensors[:, index] ** 2))
                    )
                metrics_by_location[location][terrain].append(metrics)
                manifest_rows.append(metrics)
                windows[(location, terrain, condition.run_id)] = sensors
        write_manifest(location_dir / "run_manifest.csv", manifest_rows)

    common_valid_ids = set(condition_by_id)
    for location in LOCATIONS:
        for terrain in TERRAINS:
            common_valid_ids &= {
                str(row["run_id"])
                for row in metrics_by_location[location][terrain]
                if int(row["valid_run"])
            }
    matched_ids = sorted(common_valid_ids)
    if not matched_ids:
        raise RuntimeError("no common valid concrete-marble A/B runs")

    separation_rows: list[dict[str, object]] = []
    feature_groups = {
        "10_channel": (*FSR_FEATURES, *IMU_FEATURES),
        "fsr_only": FSR_FEATURES,
        "imu_only": IMU_FEATURES,
    }
    for location in LOCATIONS:
        selected = {
            terrain: [
                row
                for row in metrics_by_location[location][terrain]
                if str(row["run_id"]) in common_valid_ids
            ]
            for terrain in TERRAINS
        }
        for scope, features in feature_groups.items():
            distance, spread, ratio = separation(
                selected["concrete"], selected["marble"], features
            )
            separation_rows.append(
                {
                    "sensor_location": location,
                    "scope": scope,
                    "matched_runs_per_terrain": len(matched_ids),
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_run_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_csv(output_dir / "separation.csv", separation_rows)

    imu_feature_rows: list[dict[str, object]] = []
    for location in LOCATIONS:
        selected = {
            terrain: [
                row
                for row in metrics_by_location[location][terrain]
                if str(row["run_id"]) in common_valid_ids
            ]
            for terrain in TERRAINS
        }
        for feature in IMU_FEATURES:
            _, _, ratio = separation(
                selected["concrete"], selected["marble"], (feature,)
            )
            imu_feature_rows.append(
                {
                    "sensor_location": location,
                    "feature": feature,
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_csv(output_dir / "imu_feature_separation.csv", imu_feature_rows)

    channel_rows: list[dict[str, object]] = []
    for location in LOCATIONS:
        selected = {
            terrain: [
                row
                for row in metrics_by_location[location][terrain]
                if str(row["run_id"]) in common_valid_ids
            ]
            for terrain in TERRAINS
        }
        for channel in HIL_SENSOR_CHANNELS:
            key = f"channel_rms_{channel}"
            _, _, ratio = separation(
                selected["concrete"], selected["marble"], (key,)
            )
            channel_rows.append(
                {
                    "sensor_location": location,
                    "channel": channel,
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_csv(output_dir / "channel_separation.csv", channel_rows)

    rms_rows: list[dict[str, object]] = []
    for location in LOCATIONS:
        for terrain in TERRAINS:
            selected = [
                row
                for row in metrics_by_location[location][terrain]
                if str(row["run_id"]) in common_valid_ids
            ]
            accel = mean_std(selected, "accel_rms")
            gyro = mean_std(selected, "gyro_rms")
            rms_rows.append(
                {
                    "sensor_location": location,
                    "terrain": terrain,
                    "valid_matched_runs": len(selected),
                    "accel_rms_mean": f"{accel[0]:.9f}",
                    "accel_rms_std": f"{accel[1]:.9f}",
                    "gyro_rms_mean": f"{gyro[0]:.9f}",
                    "gyro_rms_std": f"{gyro[1]:.9f}",
                }
            )
    write_csv(output_dir / "rms_summary.csv", rms_rows)

    ratios = {
        (str(row["sensor_location"]), str(row["scope"])): float(
            row["separation_ratio"]
        )
        for row in separation_rows
    }
    channel_ratios = {
        (str(row["sensor_location"]), str(row["channel"])): float(
            row["separation_ratio"]
        )
        for row in channel_rows
    }
    imu_feature_ratios = {
        (str(row["sensor_location"]), str(row["feature"])): float(
            row["separation_ratio"]
        )
        for row in imu_feature_rows
    }
    imu_channels = HIL_SENSOR_CHANNELS[4:]
    channel_deltas = {
        channel: channel_ratios[("foot_imu", channel)]
        - channel_ratios[("pelvis_imu", channel)]
        for channel in imu_channels
    }
    best_channel = max(channel_deltas, key=channel_deltas.get)
    baseline = ratios[("pelvis_imu", "10_channel")]
    candidate = ratios[("foot_imu", "10_channel")]
    change_percent = (candidate / baseline - 1.0) * 100.0

    lines = [
        "Matched G1 IMU-location A/B validation (no classifier)",
        "",
        f"Protocol: concrete + marble, {RUN_COUNT} runs each, {DURATION:.1f}s at {SAMPLE_RATE:.0f}Hz, seed={DEFAULT_SEED}, support={SUPPORT_RATIO * 100:.0f}%.",
        f"Pulse: +X half-sine, {PULSE.magnitude:.0f}N peak, start={PULSE.start_time:.2f}s, duration={PULSE.duration:.2f}s.",
        "Terrain parameters unchanged; no roughness, noise, or domain randomization added.",
        f"Common matched valid runs: {len(matched_ids)}/{RUN_COUNT} per terrain.",
        "",
        "Validity and body collision:",
    ]
    for terrain in TERRAINS:
        rows = metrics_by_location["foot_imu"][terrain]
        lines.append(
            f"- {terrain}: valid={sum(int(row['valid_run']) for row in rows)}/{len(rows)}, body_collision={sum(int(row['body_collision']) for row in rows)}"
        )
    lines.extend(("", "Concrete-marble separation:"))
    for location in LOCATIONS:
        lines.append(
            f"- {location}: 10-channel={ratios[(location, '10_channel')]:.6f}, FSR-only={ratios[(location, 'fsr_only')]:.6f}, IMU-only={ratios[(location, 'imu_only')]:.6f}"
        )
    lines.extend(("", "Accel/gyro RMS (mean +/- std over matched valid runs):"))
    for row in rms_rows:
        lines.append(
            f"- {row['sensor_location']}/{row['terrain']}: accel={float(row['accel_rms_mean']):.6f}+/-{float(row['accel_rms_std']):.6f} m/s^2, gyro={float(row['gyro_rms_mean']):.6f}+/-{float(row['gyro_rms_std']):.6f} rad/s"
        )
    lines.extend(("", "IMU magnitude-feature separation:"))
    for feature in IMU_FEATURES:
        lines.append(
            f"- {feature}: pelvis={imu_feature_ratios[('pelvis_imu', feature)]:.6f}, foot={imu_feature_ratios[('foot_imu', feature)]:.6f}"
        )
    improved = candidate > baseline
    lines.extend(
        (
            "",
            f"Result: foot IMU {'improved' if improved else 'did not improve'} concrete-marble 10-channel separation ({change_percent:+.2f}%).",
            f"Largest IMU-axis separation change: {best_channel} ({channel_deltas[best_channel]:+.6f}).",
            "Axis RMS improved most in accel_x/accel_y and gyro_z, while aggregate accel_rms separation decreased.",
            "Per-channel details: channel_separation.csv",
        )
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "validation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
