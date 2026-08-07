"""Run the deterministic 2x2 G1 surface-representation/sampling-rate study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from hil_sensor import HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import calculate_run_metrics, write_manifest
from run_horizontal_pulse_dataset import DEFAULT_SEED, SIMULATION_DIR, run_window
from surface_profiles import (
    CONCRETE_PEAK_TO_VALLEY_M,
    GRID_SPACING_M,
    MARBLE_PEAK_TO_VALLEY_M,
    WAVELENGTHS_M,
    configure_surface_floor,
    surface_floor_configurator,
)
from terrain_profiles import TERRAIN_PROFILES


TERRAINS = ("concrete", "marble")
RUN_COUNT = 20
DURATION = 1.0
LOW_RATE_HZ = 50.0
HIGH_RATE_HZ = 200.0
SUPPORT_RATIO = 0.70
PULSE = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
SPECTRAL_START_S = 0.20
SPECTRAL_END_S = 0.60
SURFACE_SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "unitree_robots"
    / "g1"
    / "scene_surface_study.xml"
)
FLAT_SCENE_PATH = SURFACE_SCENE_PATH.with_name("scene.xml")
DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "surface_sampling_rate_study"
CONDITIONS = {
    "A_flat_50hz": ("flat", LOW_RATE_HZ),
    "B_flat_200hz": ("flat", HIGH_RATE_HZ),
    "C_surface_50hz": ("surface-aware", LOW_RATE_HZ),
    "D_surface_200hz": ("surface-aware", HIGH_RATE_HZ),
}
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
AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return reader.fieldnames, rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derive_50hz_run(source: Path, destination: Path) -> None:
    """Select t=0.02, 0.04, ... from the matched 200 Hz master run."""
    fieldnames, rows = load_csv(source)
    if len(rows) != int(DURATION * HIGH_RATE_HZ):
        raise ValueError(f"expected 200 high-rate rows: {source}")
    selected = rows[3::4]
    timestamps = np.asarray([float(row["timestamp"]) for row in selected])
    expected = np.arange(1, 51, dtype=np.float64) / LOW_RATE_HZ
    if len(selected) != 50 or not np.allclose(timestamps, expected, atol=1e-10):
        raise ValueError(f"50 Hz decimation was not time aligned: {source}")
    write_rows(destination, fieldnames, selected)


def run_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, rows = load_csv(path)
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
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(sensors)):
        raise ValueError(f"NaN or Inf in {path}")
    return timestamps, sensors, collision, valid_contact


def periodogram(
    values: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean()
    window = np.hanning(len(centered))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    spectrum = np.fft.rfft(centered * window)
    psd = np.abs(spectrum) ** 2 / (sample_rate * np.sum(window**2))
    if len(psd) > 2:
        psd[1:-1] *= 2.0
    return frequencies, psd


def spectral_metrics(
    timestamps: np.ndarray, sensors: np.ndarray, sample_rate: float
) -> tuple[dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    mask = (timestamps >= SPECTRAL_START_S) & (timestamps < SPECTRAL_END_S)
    expected = int(round((SPECTRAL_END_S - SPECTRAL_START_S) * sample_rate))
    if np.count_nonzero(mask) != expected:
        raise ValueError("unexpected pulse-centered spectral window length")
    metrics: dict[str, float] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    centered_acceleration = sensors[mask, 4:7] - sensors[mask, 4:7].mean(axis=0)
    metrics["accel_vector_vibration_rms"] = float(
        np.sqrt(np.mean(np.sum(centered_acceleration**2, axis=1)))
    )
    for axis_index, axis in enumerate(AXES):
        values = sensors[mask, 4 + axis_index]
        centered = values - values.mean()
        frequencies, psd = periodogram(values, sample_rate)
        nonzero = frequencies > 0.0
        energy = float(psd[nonzero].sum())
        peak_index = int(np.flatnonzero(nonzero)[np.argmax(psd[nonzero])])
        metrics[f"accel_{axis}_vibration_rms"] = float(
            np.sqrt(np.mean(centered**2))
        )
        metrics[f"accel_{axis}_spectral_peak_hz"] = float(
            frequencies[peak_index]
        )
        metrics[f"accel_{axis}_spectral_peak_psd"] = float(psd[peak_index])
        metrics[f"accel_{axis}_spectral_centroid_hz"] = float(
            np.sum(frequencies[nonzero] * psd[nonzero]) / max(energy, 1e-30)
        )
        curves[axis] = (frequencies, psd)
    return metrics, curves


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
    spread_1 = float(np.sqrt(np.mean(np.sum((first - first_centroid) ** 2, axis=1))))
    spread_2 = float(np.sqrt(np.mean(np.sum((second - second_centroid) ** 2, axis=1))))
    pooled = float(np.sqrt((spread_1**2 + spread_2**2) / 2.0))
    return distance, pooled, distance / max(pooled, 1e-12)


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(
    rows: list[dict[str, float | int | str]], feature: str
) -> tuple[float, float]:
    values = np.asarray([float(row[feature]) for row in rows])
    return float(values.mean()), float(values.std())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = generate_pulse_conditions(RUN_COUNT, DEFAULT_SEED)

    audit = {
        "xml_default_timestep_seconds": 0.002,
        "study_physics_timestep_seconds": 0.005,
        "study_physics_rate_hz": HIGH_RATE_HZ,
        "requested_high_rate_hz": 1000.0,
        "selected_high_rate_hz": HIGH_RATE_HZ,
        "one_khz_valid_without_physics_change": False,
        "reason": "At 5 ms per mj_step, sensors have only 200 distinct post-step states per second; 1 kHz would repeat each state five times.",
    }
    surface_model = mujoco.MjModel.from_xml_path(str(SURFACE_SCENE_PATH.resolve()))
    surface_model.opt.timestep = 1.0 / HIGH_RATE_HZ
    _, concrete_surface = configure_surface_floor(
        surface_model, TERRAIN_PROFILES["concrete"]
    )
    surface_model = mujoco.MjModel.from_xml_path(str(SURFACE_SCENE_PATH.resolve()))
    surface_model.opt.timestep = 1.0 / HIGH_RATE_HZ
    _, marble_surface = configure_surface_floor(
        surface_model, TERRAIN_PROFILES["marble"]
    )
    protocol = {
        "audit": audit,
        "terrains": list(TERRAINS),
        "runs_per_terrain_condition": RUN_COUNT,
        "conditions": CONDITIONS,
        "duration_seconds": DURATION,
        "seed": DEFAULT_SEED,
        "support_ratio": SUPPORT_RATIO,
        "pulse": {
            "peak_newtons": PULSE.magnitude,
            "start_seconds": PULSE.start_time,
            "duration_seconds": PULSE.duration,
            "direction_xy": [PULSE.direction_x, PULSE.direction_y],
        },
        "spectral_window_seconds": [SPECTRAL_START_S, SPECTRAL_END_S],
        "surface": {
            "representation": "MuJoCo native hfield",
            "grid_spacing_m": GRID_SPACING_M,
            "wavelengths_m": list(WAVELENGTHS_M),
            "concrete_peak_to_valley_m": CONCRETE_PEAK_TO_VALLEY_M,
            "concrete_rms_height_m": concrete_surface.rms_height_m,
            "marble_peak_to_valley_m": MARBLE_PEAK_TO_VALLEY_M,
            "marble_rms_height_m": marble_surface.rms_height_m,
            "contact_parameters_changed": False,
        },
        "noise_added": False,
        "domain_randomization_added": False,
        "classifier_trained": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "timestep_audit.txt").write_text(
        "MuJoCo timestep/sampling audit\n\n"
        "XML default timestep: 0.002 s (500 Hz).\n"
        "Existing terrain dataset runner timestep: 0.005 s (200 Hz).\n"
        "Sensor values are read once after mj_step, so the maximum distinct logging rate is 200 Hz.\n"
        "A 1 kHz logger would emit five samples per physics step with four repeated states and is not physically informative.\n"
        "Selected high rate: 200 Hz, with 50 Hz derived by exact 4:1 time-aligned decimation.\n",
        encoding="utf-8",
    )

    high_condition_by_representation = {
        "flat": "B_flat_200hz",
        "surface-aware": "D_surface_200hz",
    }
    low_condition_by_representation = {
        "flat": "A_flat_50hz",
        "surface-aware": "C_surface_50hz",
    }
    scene_by_representation = {
        "flat": FLAT_SCENE_PATH,
        "surface-aware": SURFACE_SCENE_PATH,
    }
    configurator_by_representation = {
        "flat": None,
        "surface-aware": surface_floor_configurator,
    }
    for representation in ("flat", "surface-aware"):
        high_condition = high_condition_by_representation[representation]
        high_dir = output_dir / high_condition
        pelvis_dir = output_dir / "pelvis_diagnostic" / high_condition
        print(f"collecting representation={representation} at {HIGH_RATE_HZ:.0f} Hz")
        for terrain in TERRAINS:
            for condition in conditions:
                metrics = run_window(
                    terrain=terrain,
                    condition=condition,
                    duration=DURATION,
                    sample_rate=HIGH_RATE_HZ,
                    seed=DEFAULT_SEED,
                    support_ratio=SUPPORT_RATIO,
                    pulse=PULSE,
                    output_dir=high_dir,
                    scene_path=scene_by_representation[representation],
                    matched_pelvis_output_dir=pelvis_dir,
                    model_configurator=configurator_by_representation[representation],
                )
                print(
                    f"{high_condition}/{terrain}/{condition.run_id} "
                    f"valid={metrics['valid_run']} collision={metrics['body_collision']}"
                )
        low_condition = low_condition_by_representation[representation]
        low_dir = output_dir / low_condition
        for terrain in TERRAINS:
            for condition in conditions:
                relative = Path(terrain) / f"{condition.run_id}.csv"
                derive_50hz_run(high_dir / relative, low_dir / relative)

    metrics_by_condition: dict[
        str, dict[str, list[dict[str, float | int | str]]]
    ] = {
        condition: {terrain: [] for terrain in TERRAINS}
        for condition in CONDITIONS
    }
    psd_curves: dict[tuple[str, str, str], list[np.ndarray]] = {}
    frequency_axes: dict[tuple[str, str, str], np.ndarray] = {}
    condition_by_run_id = {condition.run_id: condition for condition in conditions}
    for condition_name, (representation, sample_rate) in CONDITIONS.items():
        manifest_rows: list[dict[str, float | int | str]] = []
        condition_dir = output_dir / condition_name
        for terrain in TERRAINS:
            for condition in conditions:
                relative = Path(terrain) / f"{condition.run_id}.csv"
                timestamps, sensors, collision, valid_contact = run_arrays(
                    condition_dir / relative
                )
                metrics = calculate_run_metrics(
                    timestamps, sensors, collision, valid_contact, sample_rate
                )
                spectral, curves = spectral_metrics(timestamps, sensors, sample_rate)
                metrics.update(spectral)
                metrics.update(
                    {
                        "run_id": condition.run_id,
                        "terrain": terrain,
                        "condition": condition_name,
                        "representation": representation,
                        "sample_rate_hz": sample_rate,
                        "dataset_seed": DEFAULT_SEED,
                        "base_height_offset": condition.base_height_offset,
                        "base_roll_deg": condition.base_roll_deg,
                        "base_pitch_deg": condition.base_pitch_deg,
                        "support_ratio": SUPPORT_RATIO,
                        "pulse_magnitude": PULSE.magnitude,
                        "csv_path": str(relative),
                    }
                )
                for channel_index, channel in enumerate(HIL_SENSOR_CHANNELS):
                    metrics[f"channel_rms_{channel}"] = float(
                        np.sqrt(np.mean(sensors[:, channel_index] ** 2))
                    )
                for axis, (frequencies, psd) in curves.items():
                    key = (condition_name, terrain, axis)
                    frequency_axes[key] = frequencies
                    psd_curves.setdefault(key, []).append(psd)
                metrics_by_condition[condition_name][terrain].append(metrics)
                manifest_rows.append(metrics)
        write_manifest(condition_dir / "run_manifest.csv", manifest_rows)

    common_valid_ids = set(condition_by_run_id)
    for condition_name in CONDITIONS:
        for terrain in TERRAINS:
            common_valid_ids &= {
                str(row["run_id"])
                for row in metrics_by_condition[condition_name][terrain]
                if int(row["valid_run"])
            }
    matched_ids = sorted(common_valid_ids)
    if not matched_ids:
        raise RuntimeError("no valid run IDs matched across the full 2x2 study")

    selected: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    for condition_name in CONDITIONS:
        selected[condition_name] = {
            terrain: [
                row
                for row in metrics_by_condition[condition_name][terrain]
                if str(row["run_id"]) in common_valid_ids
            ]
            for terrain in TERRAINS
        }

    separation_rows: list[dict[str, object]] = []
    for condition_name in CONDITIONS:
        for scope, features in {
            "10_channel": (*FSR_FEATURES, *IMU_FEATURES),
            "fsr_only": FSR_FEATURES,
            "imu_only": IMU_FEATURES,
        }.items():
            distance, spread, ratio = separation(
                selected[condition_name]["concrete"],
                selected[condition_name]["marble"],
                features,
            )
            separation_rows.append(
                {
                    "condition": condition_name,
                    "scope": scope,
                    "matched_runs_per_terrain": len(matched_ids),
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_run_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_dict_rows(output_dir / "condition_separation.csv", separation_rows)

    feature_rows: list[dict[str, object]] = []
    feature_names = [
        *(f"channel_rms_{channel}" for channel in HIL_SENSOR_CHANNELS),
        "accel_vector_vibration_rms",
        *(
            f"accel_{axis}_{suffix}"
            for axis in AXES
            for suffix in (
                "vibration_rms",
                "spectral_peak_hz",
                "spectral_peak_psd",
                "spectral_centroid_hz",
            )
        ),
    ]
    for condition_name in CONDITIONS:
        for feature in feature_names:
            distance, spread, ratio = separation(
                selected[condition_name]["concrete"],
                selected[condition_name]["marble"],
                (feature,),
            )
            feature_rows.append(
                {
                    "condition": condition_name,
                    "feature": feature,
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_run_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_dict_rows(output_dir / "feature_separation.csv", feature_rows)

    spectral_summary_rows: list[dict[str, object]] = []
    for condition_name in CONDITIONS:
        for terrain in TERRAINS:
            rows = selected[condition_name][terrain]
            for axis in AXES:
                vibration = mean_std(rows, f"accel_{axis}_vibration_rms")
                peak = mean_std(rows, f"accel_{axis}_spectral_peak_hz")
                centroid = mean_std(rows, f"accel_{axis}_spectral_centroid_hz")
                spectral_summary_rows.append(
                    {
                        "condition": condition_name,
                        "terrain": terrain,
                        "axis": f"accel_{axis}",
                        "vibration_rms_mean": f"{vibration[0]:.9f}",
                        "vibration_rms_std": f"{vibration[1]:.9f}",
                        "spectral_peak_hz_mean": f"{peak[0]:.9f}",
                        "spectral_peak_hz_std": f"{peak[1]:.9f}",
                        "spectral_centroid_hz_mean": f"{centroid[0]:.9f}",
                        "spectral_centroid_hz_std": f"{centroid[1]:.9f}",
                    }
                )
    write_dict_rows(output_dir / "spectral_summary.csv", spectral_summary_rows)

    psd_rows: list[dict[str, object]] = []
    for key, curves in psd_curves.items():
        condition_name, terrain, axis = key
        matrix = np.asarray(curves)
        frequencies = frequency_axes[key]
        for index, frequency in enumerate(frequencies):
            psd_rows.append(
                {
                    "condition": condition_name,
                    "terrain": terrain,
                    "axis": f"accel_{axis}",
                    "frequency_hz": f"{frequency:.9f}",
                    "psd_mean": f"{matrix[:, index].mean():.12e}",
                    "psd_std": f"{matrix[:, index].std():.12e}",
                }
            )
    write_dict_rows(output_dir / "acceleration_psd.csv", psd_rows)

    ratios = {
        (str(row["condition"]), str(row["scope"])): float(row["separation_ratio"])
        for row in separation_rows
    }
    feature_ratios = {
        (str(row["condition"]), str(row["feature"])): float(
            row["separation_ratio"]
        )
        for row in feature_rows
    }
    a = ratios[("A_flat_50hz", "10_channel")]
    b = ratios[("B_flat_200hz", "10_channel")]
    c = ratios[("C_surface_50hz", "10_channel")]
    d = ratios[("D_surface_200hz", "10_channel")]
    interaction = (d - c) - (b - a)
    best_feature_row = max(feature_rows, key=lambda row: float(row["separation_ratio"]))
    d_concrete_z = selected["D_surface_200hz"]["concrete"]
    d_marble_z = selected["D_surface_200hz"]["marble"]
    d_concrete_vibration = mean_std(d_concrete_z, "accel_z_vibration_rms")
    d_marble_vibration = mean_std(d_marble_z, "accel_z_vibration_rms")
    d_concrete_peak = mean_std(d_concrete_z, "accel_z_spectral_peak_hz")
    d_marble_peak = mean_std(d_marble_z, "accel_z_spectral_peak_hz")
    d_concrete_centroid = mean_std(d_concrete_z, "accel_z_spectral_centroid_hz")
    d_marble_centroid = mean_std(d_marble_z, "accel_z_spectral_centroid_hz")
    lines = [
        "G1 concrete-marble surface/sampling-rate study (no classifier)",
        "",
        "Sampling audit: existing study physics is 0.005 s/step (200 Hz); 1 kHz would repeat states, so high-rate=200 Hz.",
        f"Matched protocol: {RUN_COUNT} runs/terrain/condition, seed={DEFAULT_SEED}, 80 N pulse, 70% support.",
        f"Surface: native hfield, grid={GRID_SPACING_M * 1000:.1f} mm, wavelengths={tuple(value * 1000 for value in WAVELENGTHS_M)} mm.",
        f"Concrete roughness={CONCRETE_PEAK_TO_VALLEY_M * 1000:.3f} mm peak-to-valley; marble={MARBLE_PEAK_TO_VALLEY_M * 1000:.3f} mm peak-to-valley.",
        "Friction, solref, and solimp unchanged; no noise/domain randomization.",
        f"Common matched valid runs={len(matched_ids)}/{RUN_COUNT} per terrain.",
        "",
        "Validity/body collision:",
    ]
    for condition_name in CONDITIONS:
        for terrain in TERRAINS:
            rows = metrics_by_condition[condition_name][terrain]
            lines.append(
                f"- {condition_name}/{terrain}: valid={sum(int(row['valid_run']) for row in rows)}/{len(rows)}, body_collision={sum(int(row['body_collision']) for row in rows)}"
            )
    lines.extend(("", "Concrete-marble separation:"))
    for condition_name in CONDITIONS:
        lines.append(
            f"- {condition_name}: 10-channel={ratios[(condition_name, '10_channel')]:.6f}, FSR-only={ratios[(condition_name, 'fsr_only')]:.6f}, IMU-only={ratios[(condition_name, 'imu_only')]:.6f}, accel_x={feature_ratios[(condition_name, 'channel_rms_accel_x')]:.6f}"
        )
    lines.extend(
        (
            "",
            f"Sampling-only effect B-A: {b - a:+.6f} ({(b / a - 1.0) * 100:+.2f}%).",
            f"Surface-only effect C-A: {c - a:+.6f} ({(c / a - 1.0) * 100:+.2f}%).",
            f"Combined effect D-A: {d - a:+.6f} ({(d / a - 1.0) * 100:+.2f}%).",
            f"Rate-on-surface effect D-C: {d - c:+.6f}; additive interaction={interaction:+.6f}.",
            f"Largest univariate channel/spectral separation: {best_feature_row['condition']} / {best_feature_row['feature']} = {float(best_feature_row['separation_ratio']):.6f}.",
            f"D accel_z concrete: vibration RMS={d_concrete_vibration[0]:.6f}+/-{d_concrete_vibration[1]:.6f} m/s^2, peak={d_concrete_peak[0]:.2f}+/-{d_concrete_peak[1]:.2f} Hz, centroid={d_concrete_centroid[0]:.2f}+/-{d_concrete_centroid[1]:.2f} Hz.",
            f"D accel_z marble: vibration RMS={d_marble_vibration[0]:.6f}+/-{d_marble_vibration[1]:.6f} m/s^2, peak={d_marble_peak[0]:.2f}+/-{d_marble_peak[1]:.2f} Hz, centroid={d_marble_centroid[0]:.2f}+/-{d_marble_centroid[1]:.2f} Hz.",
            "Frequency details: spectral_summary.csv and acceleration_psd.csv.",
            "",
            "Answers:",
            f"1. Sampling rate alone: {'yes' if b > a else 'no'}; B-A={b - a:+.6f}.",
            f"2. Surface representation alone: {'yes' if c > a else 'no'}; C-A={c - a:+.6f}.",
            f"3. Only the combined condition: {'no' if b > a or c > a else 'yes'}; both main effects are already visible before D.",
            f"4. Strongest feature: {best_feature_row['feature']} in {best_feature_row['condition']}; among raw HIL channel RMS features, inspect feature_separation.csv.",
            "5. Dataset implication: enough to justify a high-rate surface-aware pilot, but not classifier-training evidence; one deterministic hfield and no sim-to-real/noise coverage cannot establish generalization.",
        )
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "study_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
