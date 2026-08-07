"""Generate and evaluate the leakage-safe MuJoCo terrain Dataset v1 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np

from controlled_excitation import ExcitationCondition, HorizontalPulse
from hil_sensor import HIL_SENSOR_CHANNELS
from pulse_windows import WINDOW_PROFILES, extract_window
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window
from run_surface_sampling_rate_study import SURFACE_SCENE_PATH, write_dict_rows
from terrain_dataset_v1 import (
    DATASET_SEED,
    DOMAIN_RANGES,
    DURATION_S,
    PHYSICS_TIMESTEP_S,
    RUNS_PER_SURFACE,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SENSOR_RATE_HZ,
    SURFACES_PER_TERRAIN,
    TERRAIN_LABELS,
    apply_sensor_imperfections,
    configure_run_surface,
    fit_and_evaluate,
    is_valid_run,
    load_clean_run,
    make_run_specification,
    make_surface_parameters,
    separation_ratio,
    statistical_features,
    validate_split_integrity,
)


OUTPUT_DIR = SIMULATION_DIR / "outputs" / "terrain_dataset_v1_pilot"
CHANNEL_GROUPS = {
    "fsr_only": (0, 1, 2, 3),
    "imu_only": (4, 5, 6, 7, 8, 9),
    "fusion": tuple(range(10)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--surfaces-per-terrain", type=int, default=SURFACES_PER_TERRAIN)
    parser.add_argument("--runs-per-surface", type=int, default=RUNS_PER_SURFACE)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    write_dict_rows(path, rows)


def main() -> None:
    args = parse_args()
    if args.surfaces_per_terrain != SURFACES_PER_TERRAIN:
        raise ValueError("pilot split design requires exactly 15 surfaces per terrain")
    if args.runs_per_surface <= 0:
        raise ValueError("runs-per-surface must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "purpose": "Pilot classification feasibility with low-frequency MuJoCo dynamics; engineering variation, not measured material distributions.",
        "channels": HIL_SENSOR_CHANNELS,
        "label_mapping": TERRAIN_LABELS,
        "tensor_shape": ["N", 50, 10],
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "physics_rate_hz": 1.0 / PHYSICS_TIMESTEP_S,
        "sensor_rate_hz": SENSOR_RATE_HZ,
        "sampling_rationale": "100 Hz preserves pulse-aligned low-frequency transients better than 50 Hz while halving 200 Hz storage; no new rate sweep was performed.",
        "window": {"name": "medium_response", "start_s": 0.15, "end_s": 0.65, "definition_source": "pulse_windows.py"},
        "duration_s": DURATION_S,
        "surfaces_per_terrain": SURFACES_PER_TERRAIN,
        "surface_split": {"train": list(range(9)), "validation": list(range(9, 12)), "test": list(range(12, 15))},
        "runs_per_surface": args.runs_per_surface,
        "expected_samples": len(TERRAIN_LABELS) * SURFACES_PER_TERRAIN * args.runs_per_surface,
        "domain_ranges": {name: vars(value) for name, value in DOMAIN_RANGES.items()},
        "pulse": {"magnitude_N": [76.0, 84.0], "start_s": [0.245, 0.255], "duration_s": 0.20, "directions": ["+X", "-X"]},
        "support_ratio": 0.70,
        "initial_variation": {"height_m": [0.0, 0.008], "roll_deg": [-0.4, 0.4], "pitch_deg": [-0.4, 0.4], "velocity_x_m_s": [-0.02, 0.02], "velocity_y_m_s": [-0.01, 0.01]},
        "sensor_model_status": "pilot engineering assumptions; repository confirms BMI270 presence but contains no validated mounted-sensor noise density",
        "sensor_noise": {"fsr_gain": [0.985, 1.015], "fsr_offset_N": [-0.5, 0.5], "fsr_noise_std_N": 0.15, "fsr_clip_N": [0.0, 250.0], "accel_gain": [0.995, 1.005], "accel_bias_std_m_s2": 0.03, "accel_noise_std_m_s2": 0.02, "gyro_gain": [0.995, 1.005], "gyro_bias_std_rad_s": 0.002, "gyro_noise_std_rad_s": 0.001},
        "excluded_features": ["raw high-frequency vibration PSD", "dominant vibration frequency", "timestep-sensitive micro-contact spectrum"],
        "dataset_seed": DATASET_SEED,
        "baseline": "RandomForest on per-channel mean/std/RMS/min/max/peak-to-peak; no spectral features",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    clean_windows: list[np.ndarray] = []
    noisy_windows: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    metadata: list[dict[str, object]] = []
    total = protocol["expected_samples"]
    completed = 0
    for terrain in TERRAIN_LABELS:
        for surface_index in range(SURFACES_PER_TERRAIN):
            surface = make_surface_parameters(terrain, DATASET_SEED + 10_000 * TERRAIN_LABELS[terrain] + 100 * surface_index)
            for run_index in range(args.runs_per_surface):
                spec = make_run_specification(terrain, surface_index, run_index)
                condition = ExcitationCondition(
                    run_id=spec.run_id,
                    initial_velocity_x=spec.initial_velocity_x,
                    initial_velocity_y=spec.initial_velocity_y,
                    base_height_offset=spec.base_height_offset,
                    base_roll_deg=spec.base_roll_deg,
                    base_pitch_deg=spec.base_pitch_deg,
                )
                pulse = HorizontalPulse(spec.pulse_start, spec.pulse_duration, spec.pulse_magnitude, spec.pulse_direction_x, spec.pulse_direction_y)
                raw_root = output / "raw_clean" / spec.split / spec.session_id
                pelvis_root = output / "pelvis_diagnostic" / spec.split / spec.session_id
                metrics = run_window(
                    terrain, condition, DURATION_S, SENSOR_RATE_HZ, spec.physics_seed,
                    0.70, pulse, raw_root,
                    session_id=spec.session_id,
                    scene_path=SURFACE_SCENE_PATH,
                    matched_pelvis_output_dir=pelvis_root,
                    model_configurator=partial(configure_run_surface, surface=surface, friction=spec.friction),
                    physics_timestep=PHYSICS_TIMESTEP_S,
                )
                valid = int(is_valid_run(metrics))
                raw_path = raw_root / terrain / f"{spec.run_id}.csv"
                timestamps, sensors = load_clean_run(raw_path)
                row: dict[str, object] = {
                    "terrain_class": terrain, "terrain_id": spec.terrain_id,
                    "split": spec.split, "run_id": spec.run_id, "session_id": spec.session_id,
                    "run_group": spec.run_group, "surface_seed": spec.surface_seed,
                    "physics_seed": spec.physics_seed, "sensor_noise_seed": spec.sensor_noise_seed,
                    "friction_slide": spec.friction[0], "friction_torsional": spec.friction[1], "friction_rolling": spec.friction[2],
                    "roughness_peak_to_valley_m": surface.peak_to_valley_m,
                    "roughness_wavelengths_m": "|".join(f"{x:.9g}" for x in surface.wavelengths_m),
                    "pulse_magnitude_N": spec.pulse_magnitude, "pulse_direction_x": spec.pulse_direction_x,
                    "pulse_start_s": spec.pulse_start, "pulse_duration_s": spec.pulse_duration,
                    "support_ratio": 0.70, "support_target_force_N": metrics["support_target_force"],
                    "initial_total_force_N": float(sensors[0, :4].sum()),
                    "initial_velocity_x": spec.initial_velocity_x,
                    "initial_velocity_y": spec.initial_velocity_y, "base_height_offset": spec.base_height_offset,
                    "base_roll_deg": spec.base_roll_deg, "base_pitch_deg": spec.base_pitch_deg,
                    "valid_flag": valid, "body_collision": metrics["body_collision"],
                    "extreme_force_outlier": metrics["extreme_force_outlier"], "extreme_accel_outlier": metrics["extreme_accel_outlier"],
                    "contact_duration_s": metrics["contact_duration"], "raw_clean_path": str(raw_path.relative_to(output)),
                }
                metadata.append(row)
                if valid:
                    _, clean = extract_window(timestamps, sensors, WINDOW_PROFILES["medium_response"], SENSOR_RATE_HZ)
                    noisy = apply_sensor_imperfections(clean, spec.sensor_noise_seed)
                    clean_windows.append(clean.astype(np.float32))
                    noisy_windows.append(noisy)
                    labels.append(spec.terrain_id)
                    splits.append(spec.split)
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"completed {completed}/{total}; valid={sum(int(row['valid_flag']) for row in metadata)}")

    validate_split_integrity(metadata)
    x_clean = np.asarray(clean_windows, dtype=np.float32)
    x_noisy = np.asarray(noisy_windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)
    if x_clean.shape != (len(y), 50, 10) or x_noisy.shape != x_clean.shape:
        raise ValueError(f"unexpected tensors {x_clean.shape} and {x_noisy.shape}")
    if not np.all(np.isfinite(x_clean)) or not np.all(np.isfinite(x_noisy)):
        raise ValueError("NaN/Inf in dataset tensors")
    np.savez_compressed(output / "dataset_clean.npz", X=x_clean, y=y)
    np.savez_compressed(output / "dataset_noisy.npz", X=x_noisy, y=y)
    write_csv(output / "run_metadata.csv", metadata)
    manifest = [{"sample_index": i, "terrain_class": tuple(TERRAIN_LABELS)[int(y[i])], "terrain_id": int(y[i]), "split": split_array[i], "surface_seed": row["surface_seed"], "session_id": row["session_id"], "run_id": row["run_id"], "run_group": row["run_group"], "sensor_noise_seed": row["sensor_noise_seed"]} for i, row in enumerate(row for row in metadata if int(row["valid_flag"]) == 1)]
    write_csv(output / "split_manifest.csv", manifest)

    statistic_rows: list[dict[str, object]] = []
    for variant, tensor in (("clean", x_clean), ("noisy", x_noisy)):
        for index, channel in enumerate(HIL_SENSOR_CHANNELS):
            values = tensor[:, :, index]
            statistic_rows.append({"variant": variant, "channel": channel, "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())})
    write_csv(output / "channel_statistics.csv", statistic_rows)

    pair_rows: list[dict[str, object]] = []
    test = split_array == "test"
    terrains = tuple(TERRAIN_LABELS)
    for variant, tensor in (("clean", x_clean), ("noisy", x_noisy)):
        for left_index, left in enumerate(terrains):
            for right in terrains[left_index + 1 :]:
                for group, channels in CHANNEL_GROUPS.items():
                    features = statistical_features(tensor, channels)
                    distance, spread, ratio = separation_ratio(features[test & (y == TERRAIN_LABELS[left])], features[test & (y == TERRAIN_LABELS[right])])
                    pair_rows.append({"variant": variant, "split": "test", "terrain_a": left, "terrain_b": right, "sensor_group": group, "centroid_distance": distance, "pooled_spread": spread, "separation_ratio": ratio})
    write_csv(output / "pair_separation.csv", pair_rows)

    baseline_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for variant, tensor in (("clean", x_clean), ("noisy", x_noisy)):
        for group, channels in CHANNEL_GROUPS.items():
            metrics, matrix = fit_and_evaluate(tensor, y, split_array, channels)
            baseline_rows.extend({"variant": variant, "sensor_group": group, **row} for row in metrics)
            confusion_rows.extend({"variant": variant, "sensor_group": group, **row} for row in matrix)
    write_csv(output / "baseline_metrics.csv", baseline_rows)
    write_csv(output / "confusion_matrix.csv", confusion_rows)

    counts = {(terrain, split): int(np.sum((y == terrain_id) & (split_array == split))) for terrain, terrain_id in TERRAIN_LABELS.items() for split in ("train", "validation", "test")}
    noisy_fusion = next(row for row in baseline_rows if row["variant"] == "noisy" and row["sensor_group"] == "fusion" and row["split"] == "test" and row["class"] == "overall")
    summary = [
        f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION}",
        f"X_clean={x_clean.shape} X_noisy={x_noisy.shape} y={y.shape} dtype={x_clean.dtype}",
        f"valid={len(y)}/{len(metadata)} invalid={len(metadata)-len(y)}",
        "class/split counts: " + ", ".join(f"{terrain}/{split}={counts[(terrain, split)]}" for terrain in TERRAIN_LABELS for split in ("train", "validation", "test")),
        f"noisy fusion unseen-surface test accuracy={float(noisy_fusion['accuracy']):.6f}",
        "Raw MuJoCo diagnostics, pelvis diagnostics, clean AI tensors, and noisy AI tensors are stored separately.",
        "No high-frequency PSD, dominant-frequency, or synthetic vibration feature is used.",
    ]
    (output / "dataset_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
