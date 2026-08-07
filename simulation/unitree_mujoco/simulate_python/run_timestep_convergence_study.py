"""Run and analyze 5/2/1 ms MuJoCo timestep convergence."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from hil_sensor import HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import calculate_run_metrics, write_manifest
from run_horizontal_pulse_dataset import DEFAULT_SEED, SIMULATION_DIR, run_window
from run_surface_factorization_study import SCENE_PATH, factor_configurator
from run_surface_sampling_rate_study import (
    FSR_FEATURES,
    IMU_FEATURES,
    periodogram,
    run_arrays,
    spectral_metrics,
    write_dict_rows,
)
from terrain_profiles import TERRAIN_PROFILES


OUTPUT_DIR = SIMULATION_DIR / "outputs" / "timestep_convergence_study"
RUN_COUNT = 10
PULSE = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
TIMESTEPS = {"dt_5ms": 0.005, "dt_2ms": 0.002, "dt_1ms": 0.001}
REPRESENTATIVES = {
    "concrete_equivalent": ("concrete", "concrete"),
    "marble_equivalent": ("marble", "marble"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def timestep_metrics(
    timestamps: np.ndarray, sensors: np.ndarray, sample_rate: float
) -> dict[str, float]:
    spectral, _ = spectral_metrics(timestamps, sensors, sample_rate)
    total = sensors[:, :4].sum(axis=1)
    result = {
        "total_force_rms": float(np.sqrt(np.mean(total**2))),
        "total_force_peak": float(total.max()),
    }
    result.update(spectral)
    mask = (timestamps >= 0.20) & (timestamps < 0.60)
    frequencies, psd = periodogram(sensors[mask, 6], sample_rate)
    df = frequencies[1] - frequencies[0]
    for low, high in ((0, 20), (20, 50), (50, 90)):
        band = (frequencies > low) & (frequencies <= high)
        result[f"accel_z_band_{low}_{high}_power"] = float(psd[band].sum() * df)
    return result


def interpolate_channels(
    timestamps: np.ndarray, values: np.ndarray, target: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [np.interp(target, timestamps, values[:, index]) for index in range(values.shape[1])]
    )


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    conditions = generate_pulse_conditions(RUN_COUNT, DEFAULT_SEED)
    protocol = {
        "representatives": REPRESENTATIVES,
        "timesteps_s": TIMESTEPS,
        "runs_per_representative_timestep": RUN_COUNT,
        "duration_s": 1.0,
        "logging": "one distinct sample after every mj_step",
        "seed": DEFAULT_SEED,
        "pulse": {"peak_N": 80.0, "start_s": 0.25, "duration_s": 0.20},
        "support_ratio": 0.70,
        "reference": "dt_1ms",
        "comparison_grid": "200 Hz (5 ms); 1 ms exact decimation and 2 ms interpolation",
        "noise": False,
        "domain_randomization": False,
        "classifier": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    execution: dict[tuple[str, str, str], dict[str, float | int | str]] = {}
    for representative, (friction_source, roughness_source) in REPRESENTATIVES.items():
        configurator = partial(
            factor_configurator, roughness_source=roughness_source
        )
        for dt_name, timestep in TIMESTEPS.items():
            rate = 1.0 / timestep
            print(f"collecting {representative}/{dt_name} at {rate:.0f} Hz")
            for condition in conditions:
                metrics = run_window(
                    friction_source,
                    condition,
                    1.0,
                    rate,
                    DEFAULT_SEED,
                    0.70,
                    PULSE,
                    output / representative / dt_name,
                    scene_path=SCENE_PATH,
                    matched_pelvis_output_dir=output / "pelvis_diagnostic" / representative / dt_name,
                    model_configurator=configurator,
                    physics_timestep=timestep,
                )
                execution[(representative, dt_name, condition.run_id)] = metrics
                print(
                    f"{representative}/{dt_name}/{condition.run_id} "
                    f"valid={metrics['valid_run']} collision={metrics['body_collision']}"
                )

    rows_by_key: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    arrays: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for representative, (friction_source, _) in REPRESENTATIVES.items():
        for dt_name, timestep in TIMESTEPS.items():
            rate = 1.0 / timestep
            rows_by_key[(representative, dt_name)] = []
            for condition in conditions:
                relative = Path(friction_source) / f"{condition.run_id}.csv"
                timestamps, sensors, collision, contact = run_arrays(
                    output / representative / dt_name / relative
                )
                expected = int(round(1.0 / timestep))
                if len(timestamps) != expected or np.any(np.diff(timestamps) <= 0):
                    raise ValueError("duplicate/missing timestep samples")
                if not np.allclose(np.diff(timestamps), timestep, atol=1e-10):
                    raise ValueError("logging is not one sample per distinct physics step")
                metrics = calculate_run_metrics(
                    timestamps, sensors, collision, contact, rate
                )
                metrics.update(timestep_metrics(timestamps, sensors, rate))
                metrics.update(
                    {
                        "representative": representative,
                        "dt_name": dt_name,
                        "timestep_s": timestep,
                        "sample_rate_hz": rate,
                        "run_id": condition.run_id,
                        "support_target_force": execution[(representative, dt_name, condition.run_id)]["support_target_force"],
                        "pulse_contact_fraction": execution[(representative, dt_name, condition.run_id)]["pulse_contact_fraction"],
                        "csv_path": str(relative),
                    }
                )
                rows_by_key[(representative, dt_name)].append(metrics)
                arrays[(representative, dt_name, condition.run_id)] = (
                    timestamps,
                    sensors,
                )
            write_manifest(
                output / representative / dt_name / "run_manifest.csv",
                rows_by_key[(representative, dt_name)],
            )

    error_rows: list[dict[str, object]] = []
    waveform_rows: list[dict[str, object]] = []
    summary_features = (*FSR_FEATURES, *IMU_FEATURES)
    target = np.arange(1, 201, dtype=np.float64) * 0.005
    for representative in REPRESENTATIVES:
        reference_rows = {
            str(row["run_id"]): row
            for row in rows_by_key[(representative, "dt_1ms")]
        }
        for dt_name in ("dt_5ms", "dt_2ms"):
            per_metric: dict[str, list[float]] = {
                key: []
                for key in (
                    "peak_frequency",
                    "vibration_rms",
                    "accel_rms",
                    "total_force_rms",
                    "summary_10",
                    "psd_shape_correlation",
                )
            }
            waveform_error = {channel: [] for channel in (*HIL_SENSOR_CHANNELS, "total_force")}
            for row in rows_by_key[(representative, dt_name)]:
                run_id = str(row["run_id"])
                ref = reference_rows[run_id]
                per_metric["peak_frequency"].append(relative_error(float(row["accel_z_spectral_peak_hz"]), float(ref["accel_z_spectral_peak_hz"])))
                per_metric["vibration_rms"].append(relative_error(float(row["accel_z_vibration_rms"]), float(ref["accel_z_vibration_rms"])))
                per_metric["accel_rms"].append(relative_error(float(row["accel_rms"]), float(ref["accel_rms"])))
                per_metric["total_force_rms"].append(relative_error(float(row["total_force_rms"]), float(ref["total_force_rms"])))
                vector = np.asarray([float(row[key]) for key in summary_features])
                ref_vector = np.asarray([float(ref[key]) for key in summary_features])
                per_metric["summary_10"].append(float(np.linalg.norm(vector - ref_vector) / max(np.linalg.norm(ref_vector), 1e-12)))
                time, sensors = arrays[(representative, dt_name, run_id)]
                ref_time, ref_sensors = arrays[(representative, "dt_1ms", run_id)]
                common = interpolate_channels(time, sensors, target)
                ref_common = interpolate_channels(ref_time, ref_sensors, target)
                for index, channel in enumerate(HIL_SENSOR_CHANNELS):
                    waveform_error[channel].append(float(np.sqrt(np.mean((common[:, index] - ref_common[:, index]) ** 2)) / max(np.sqrt(np.mean(ref_common[:, index] ** 2)), 1e-12)))
                waveform_error["total_force"].append(float(np.sqrt(np.mean((common[:, :4].sum(axis=1) - ref_common[:, :4].sum(axis=1)) ** 2)) / max(np.sqrt(np.mean(ref_common[:, :4].sum(axis=1) ** 2)), 1e-12)))
                mask = (time >= 0.20) & (time < 0.60)
                ref_mask = (ref_time >= 0.20) & (ref_time < 0.60)
                freq, psd = periodogram(sensors[mask, 6], 1.0 / TIMESTEPS[dt_name])
                ref_freq, ref_psd = periodogram(ref_sensors[ref_mask, 6], 1000.0)
                grid = ref_freq[(ref_freq > 0) & (ref_freq <= 90)]
                a = np.interp(grid, freq, psd); b = np.interp(grid, ref_freq, ref_psd)
                per_metric["psd_shape_correlation"].append(float(np.corrcoef(a, b)[0, 1]))
            for metric, values in per_metric.items():
                error_rows.append({"representative": representative, "dt_name": dt_name, "metric": metric, "mean": f"{np.mean(values):.9f}", "std": f"{np.std(values):.9f}"})
            for channel, values in waveform_error.items():
                waveform_rows.append({"representative": representative, "dt_name": dt_name, "channel": channel, "normalized_rmse_mean": f"{np.mean(values):.9f}", "normalized_rmse_std": f"{np.std(values):.9f}"})
    write_dict_rows(output / "reference_errors.csv", error_rows)
    write_dict_rows(output / "waveform_errors_200hz_grid.csv", waveform_rows)

    spectral_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for key, rows in rows_by_key.items():
        representative, dt_name = key
        for feature in ("accel_z_vibration_rms", "accel_z_spectral_peak_hz", "accel_z_spectral_centroid_hz", "accel_z_band_0_20_power", "accel_z_band_20_50_power", "accel_z_band_50_90_power"):
            values=np.asarray([float(row[feature]) for row in rows])
            spectral_rows.append({"representative":representative,"dt_name":dt_name,"feature":feature,"mean":f"{values.mean():.9f}","std":f"{values.std():.9f}"})
        cell_rows.append({"representative":representative,"dt_name":dt_name,"valid_runs":sum(int(r['valid_run']) for r in rows),"body_collision":sum(int(r['body_collision']) for r in rows),"extreme_force":sum(int(r['extreme_force_outlier']) for r in rows),"extreme_accel":sum(int(r['extreme_accel_outlier']) for r in rows),"support_force_mean":f"{np.mean([float(r['support_target_force']) for r in rows]):.9f}","pulse_contact_mean":f"{np.mean([float(r['pulse_contact_fraction']) for r in rows]):.9f}"})
    write_dict_rows(output / "spectral_summary.csv", spectral_rows)
    write_dict_rows(output / "cell_summary.csv", cell_rows)
    lookup={(r['representative'],r['dt_name'],r['feature']):float(r['mean']) for r in spectral_rows}
    err={(r['representative'],r['dt_name'],r['metric']):float(r['mean']) for r in error_rows}
    lines=["MuJoCo timestep convergence (1 ms reference)","",f"Runs: {RUN_COUNT} matched per representative/timestep."]
    for representative in REPRESENTATIVES:
        lines.append("")
        for dt_name in TIMESTEPS:
            lines.append(f"- {representative}/{dt_name}: vibration_rms={lookup[(representative,dt_name,'accel_z_vibration_rms')]:.6f}, peak={lookup[(representative,dt_name,'accel_z_spectral_peak_hz')]:.2f}Hz, centroid={lookup[(representative,dt_name,'accel_z_spectral_centroid_hz')]:.2f}Hz")
        for dt_name in ('dt_5ms','dt_2ms'):
            lines.append(f"  errors {dt_name}: peak={err[(representative,dt_name,'peak_frequency')]*100:.1f}%, vibration={err[(representative,dt_name,'vibration_rms')]*100:.1f}%, accel={err[(representative,dt_name,'accel_rms')]*100:.2f}%, force={err[(representative,dt_name,'total_force_rms')]*100:.2f}%, summary10={err[(representative,dt_name,'summary_10')]*100:.2f}%, PSDcorr={err[(representative,dt_name,'psd_shape_correlation')]:.3f}")
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == "__main__":
    main()
