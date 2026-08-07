"""Final matched two-timestep MuJoCo convergence study."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from hil_sensor import HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import calculate_run_metrics
from run_horizontal_pulse_dataset import DEFAULT_SEED, SIMULATION_DIR, run_window
from run_surface_factorization_study import SCENE_PATH, factor_configurator
from run_surface_sampling_rate_study import (
    FSR_FEATURES,
    IMU_FEATURES,
    load_csv,
    periodogram,
    run_arrays,
    separation,
    write_dict_rows,
)


OUTPUT_DIR = SIMULATION_DIR / "outputs" / "final_timestep_convergence"
RUN_COUNT = 10
PULSE = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
TIMESTEPS = {"dt_1ms": 0.001, "dt_0p5ms": 0.0005}
REPRESENTATIVES = {
    "concrete_equivalent": ("concrete", "concrete"),
    "marble_equivalent": ("marble", "marble"),
}
BANDS = ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400))
SPECTRAL_AXES = {"accel_x": 4, "accel_z": 6, "gyro_z": 9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def per_run_metrics(path: Path, rate: float) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    timestamps, sensors, collision, contact = run_arrays(path)
    metrics: dict[str, float | int] = calculate_run_metrics(
        timestamps, sensors, collision, contact, rate
    )
    _, rows = load_csv(path)
    total = sensors[:, :4].sum(axis=1)
    heel = sensors[:, :2].sum(axis=1)
    toe = sensors[:, 2:4].sum(axis=1)
    impulse = sensors[:, :4].sum(axis=0) / rate
    contact_mask = total > 1.0
    x = np.asarray((-0.05, -0.05, 0.12, 0.12))
    cop = (sensors[contact_mask, :4] * x).sum(axis=1) / total[contact_mask]
    focus = (timestamps >= 0.20) & (timestamps < 0.60)
    centered_accel = sensors[focus, 4:7] - sensors[focus, 4:7].mean(axis=0)
    metrics.update(
        {
            "total_force_peak": float(total.max()),
            "total_force_rms": float(np.sqrt(np.mean(total**2))),
            "heel_load_fraction": float(impulse[:2].sum() / max(impulse.sum(), 1e-12)),
            "toe_load_fraction": float(impulse[2:].sum() / max(impulse.sum(), 1e-12)),
            "cop_proxy_x_mean": float(cop.mean()),
            "accel_vector_vibration_rms": float(
                np.sqrt(np.mean(np.sum(centered_accel**2, axis=1)))
            ),
            "accel_x_vibration_rms": float(np.sqrt(np.mean(centered_accel[:, 0] ** 2))),
            "accel_z_vibration_rms": float(np.sqrt(np.mean(centered_accel[:, 2] ** 2))),
            "slip_displacement": float(rows[-1]["foot_slip_displacement"]),
            "max_foot_speed": float(
                max(
                    np.hypot(
                        float(row["left_foot_velocity_x"]),
                        float(row["left_foot_velocity_y"]),
                    )
                    for row in rows
                )
            ),
            "peak_tangential_force": float(
                max(float(row["contact_tangential_force"]) for row in rows)
            ),
            "pulse_contact_fraction": float(
                np.mean(
                    [
                        int(row["valid_foot_contact"])
                        for row in rows
                        if 0.25 <= float(row["timestamp"]) < 0.45
                    ]
                )
            ),
            "contact_loss_count": int(np.count_nonzero(contact_mask[:-1] & ~contact_mask[1:])),
            "contact_reentry_count": int(np.count_nonzero(~contact_mask[:-1] & contact_mask[1:])),
        }
    )
    for index, channel in enumerate(HIL_SENSOR_CHANNELS):
        metrics[f"channel_rms_{channel}"] = float(
            np.sqrt(np.mean(sensors[:, index] ** 2))
        )
    for axis, index in SPECTRAL_AXES.items():
        frequencies, psd = periodogram(sensors[focus, index], rate)
        common = (frequencies > 0.0) & (frequencies <= 450.0)
        common_frequency = frequencies[common]
        common_psd = psd[common]
        peak = int(np.argmax(common_psd))
        metrics[f"{axis}_peak_0_450_hz"] = float(common_frequency[peak])
        metrics[f"{axis}_centroid_0_450_hz"] = float(
            np.sum(common_frequency * common_psd) / max(common_psd.sum(), 1e-30)
        )
        df = frequencies[1] - frequencies[0]
        for low, high in BANDS:
            band = (frequencies > low) & (frequencies <= high)
            metrics[f"{axis}_band_{low}_{high}_power"] = float(psd[band].sum() * df)
    return metrics, timestamps, sensors


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    conditions = generate_pulse_conditions(RUN_COUNT, DEFAULT_SEED)
    coarse_name, fine_name = tuple(TIMESTEPS)
    coarse_timestep = TIMESTEPS[coarse_name]
    fine_timestep = TIMESTEPS[fine_name]
    if not np.isclose(coarse_timestep / fine_timestep, 2.0):
        raise ValueError("final convergence requires an exact 2:1 timestep ratio")
    protocol = {
        "purpose": "numerical convergence only; no parameter or feature changes",
        "representatives": REPRESENTATIVES,
        "timesteps_s": TIMESTEPS,
        "runs_per_cell": RUN_COUNT,
        "duration_s": 1.0,
        "logging": "one sensor state after every distinct mj_step",
        "common_time_grid_hz": 1.0 / coarse_timestep,
        "decimation": f"exact {fine_name} rows[1::2] onto {coarse_name}; no anti-alias filter",
        "spectral_window_s": [0.20, 0.60],
        "common_spectral_band_hz": [0, 450],
        "bands_hz": BANDS,
        "seed": DEFAULT_SEED,
        "pulse": {"peak_N": 80, "start_s": 0.25, "duration_s": 0.20},
        "support_ratio": 0.70,
        "noise": False,
        "domain_randomization": False,
        "classifier": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    execution = {}
    for representative, (friction, roughness) in REPRESENTATIVES.items():
        configurator = partial(factor_configurator, roughness_source=roughness)
        for dt_name, timestep in TIMESTEPS.items():
            rate = 1.0 / timestep
            print(f"collecting {representative}/{dt_name} at {rate:.0f} Hz")
            for condition in conditions:
                metrics = run_window(
                    friction,
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

    run_rows: list[dict[str, object]] = []
    arrays = {}
    metrics_by_cell = {}
    for representative, (friction, _) in REPRESENTATIVES.items():
        for dt_name, timestep in TIMESTEPS.items():
            rate = 1.0 / timestep
            metrics_by_cell[(representative, dt_name)] = []
            for condition in conditions:
                path = output / representative / dt_name / friction / f"{condition.run_id}.csv"
                metrics, timestamps, sensors = per_run_metrics(path, rate)
                expected = int(round(1.0 / timestep))
                if len(timestamps) != expected or not np.allclose(
                    np.diff(timestamps), timestep, atol=1e-11
                ):
                    raise ValueError("missing or duplicate distinct-state samples")
                metrics.update(
                    {
                        "representative": representative,
                        "dt_name": dt_name,
                        "timestep_s": timestep,
                        "sample_rate_hz": rate,
                        "run_id": condition.run_id,
                        "base_height_offset": condition.base_height_offset,
                        "base_roll_deg": condition.base_roll_deg,
                        "base_pitch_deg": condition.base_pitch_deg,
                        "support_target_force": execution[(representative, dt_name, condition.run_id)]["support_target_force"],
                    }
                )
                run_rows.append(metrics)
                metrics_by_cell[(representative, dt_name)].append(metrics)
                arrays[(representative, dt_name, condition.run_id)] = (
                    timestamps,
                    sensors,
                    path,
                )
    write_dict_rows(output / "run_level_metrics.csv", run_rows)

    time_error_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    summary_features = (*FSR_FEATURES, *IMU_FEATURES)
    comparison_features = (
        "total_force_rms",
        "total_force_peak",
        "heel_load_fraction",
        "toe_load_fraction",
        "cop_proxy_x_mean",
        "accel_rms",
        "gyro_rms",
        "accel_vector_vibration_rms",
        "accel_x_vibration_rms",
        "accel_z_vibration_rms",
        "slip_displacement",
        "max_foot_speed",
        "peak_tangential_force",
        "pulse_contact_fraction",
    )
    for representative in REPRESENTATIVES:
        fine_by_run = {
            str(row["run_id"]): row
            for row in metrics_by_cell[(representative, fine_name)]
        }
        errors = {feature: [] for feature in comparison_features}
        errors["summary_10_relative_norm"] = []
        for axis in SPECTRAL_AXES:
            errors[f"{axis}_peak_0_450_hz"] = []
            errors[f"{axis}_centroid_0_450_hz"] = []
        psd_correlations = {axis: [] for axis in SPECTRAL_AXES}
        band_errors = {(axis, low, high): [] for axis in SPECTRAL_AXES for low, high in BANDS}
        for coarse in metrics_by_cell[(representative, coarse_name)]:
            run_id = str(coarse["run_id"])
            fine = fine_by_run[run_id]
            coarse_time, coarse_sensors, coarse_path = arrays[(representative, coarse_name, run_id)]
            fine_time, fine_sensors, fine_path = arrays[(representative, fine_name, run_id)]
            decimated_time = fine_time[1::2]
            decimated_sensors = fine_sensors[1::2]
            if not np.array_equal(coarse_time, decimated_time):
                raise ValueError(
                    f"{fine_timestep * 1000:g} ms data do not exactly align to "
                    f"the {coarse_timestep * 1000:g} ms grid"
                )
            for index, channel in enumerate(HIL_SENSOR_CHANNELS):
                reference_rms = np.sqrt(np.mean(decimated_sensors[:, index] ** 2))
                nrmse = np.sqrt(
                    np.mean((coarse_sensors[:, index] - decimated_sensors[:, index]) ** 2)
                ) / max(reference_rms, 1e-12)
                time_error_rows.append(
                    {
                        "representative": representative,
                        "run_id": run_id,
                        "signal": channel,
                        "normalized_rmse": f"{nrmse:.9f}",
                    }
                )
            total_coarse = coarse_sensors[:, :4].sum(axis=1)
            total_fine = decimated_sensors[:, :4].sum(axis=1)
            time_error_rows.append(
                {
                    "representative": representative,
                    "run_id": run_id,
                    "signal": "total_force",
                    "normalized_rmse": f"{np.sqrt(np.mean((total_coarse-total_fine)**2))/max(np.sqrt(np.mean(total_fine**2)),1e-12):.9f}",
                }
            )
            _, coarse_csv = load_csv(coarse_path)
            _, fine_csv = load_csv(fine_path)
            coarse_slip = np.asarray([float(row["foot_slip_displacement"]) for row in coarse_csv])
            fine_slip = np.asarray([float(row["foot_slip_displacement"]) for row in fine_csv])[1::2]
            time_error_rows.append(
                {
                    "representative": representative,
                    "run_id": run_id,
                    "signal": "slip",
                    "normalized_rmse": f"{np.sqrt(np.mean((coarse_slip-fine_slip)**2))/max(np.sqrt(np.mean(fine_slip**2)),1e-12):.9f}",
                }
            )
            for feature in comparison_features:
                errors[feature].append(relative_error(float(coarse[feature]), float(fine[feature])))
            for axis in SPECTRAL_AXES:
                for suffix in ("peak_0_450_hz", "centroid_0_450_hz"):
                    feature = f"{axis}_{suffix}"
                    errors[feature].append(
                        relative_error(float(coarse[feature]), float(fine[feature]))
                    )
            coarse_vector = np.asarray([float(coarse[key]) for key in summary_features])
            fine_vector = np.asarray([float(fine[key]) for key in summary_features])
            errors["summary_10_relative_norm"].append(
                float(np.linalg.norm(coarse_vector - fine_vector) / max(np.linalg.norm(fine_vector), 1e-12))
            )
            for axis, index in SPECTRAL_AXES.items():
                coarse_focus = (coarse_time >= 0.20) & (coarse_time < 0.60)
                fine_focus = (fine_time >= 0.20) & (fine_time < 0.60)
                coarse_freq, coarse_psd = periodogram(
                    coarse_sensors[coarse_focus, index], 1.0 / coarse_timestep
                )
                fine_freq, fine_psd = periodogram(
                    fine_sensors[fine_focus, index], 1.0 / fine_timestep
                )
                grid = coarse_freq[(coarse_freq > 0) & (coarse_freq <= 450)]
                a = coarse_psd[(coarse_freq > 0) & (coarse_freq <= 450)]
                b = np.interp(grid, fine_freq, fine_psd)
                psd_correlations[axis].append(float(np.corrcoef(a, b)[0, 1]))
                for low, high in BANDS:
                    key = f"{axis}_band_{low}_{high}_power"
                    band_errors[(axis, low, high)].append(
                        relative_error(float(coarse[key]), float(fine[key]))
                    )
        for feature, values in errors.items():
            convergence_rows.append(
                {
                    "representative": representative,
                    "metric": feature,
                    "relative_error_mean": f"{np.mean(values):.9f}",
                    "relative_error_std": f"{np.std(values):.9f}",
                }
            )
        for axis, values in psd_correlations.items():
            convergence_rows.append(
                {
                    "representative": representative,
                    "metric": f"{axis}_psd_shape_correlation_0_450",
                    "relative_error_mean": f"{np.mean(values):.9f}",
                    "relative_error_std": f"{np.std(values):.9f}",
                }
            )
        for key, values in band_errors.items():
            axis, low, high = key
            band_rows.append(
                {
                    "representative": representative,
                    "axis": axis,
                    "band_hz": f"{low}-{high}",
                    "relative_error_mean": f"{np.mean(values):.9f}",
                    "relative_error_std": f"{np.std(values):.9f}",
                }
            )
    write_dict_rows(output / "matched_time_domain_errors.csv", time_error_rows)
    write_dict_rows(output / "convergence_metrics.csv", convergence_rows)
    write_dict_rows(output / "band_power_summary.csv", band_rows)

    for representative in REPRESENTATIVES:
        for dt_name in TIMESTEPS:
            rows = metrics_by_cell[(representative, dt_name)]
            for axis in SPECTRAL_AXES:
                for feature in (f"{axis}_peak_0_450_hz", f"{axis}_centroid_0_450_hz", f"{axis}_vibration_rms" if axis != "gyro_z" else "gyro_rms"):
                    values = np.asarray([float(row[feature]) for row in rows])
                    spectral_rows.append(
                        {
                            "representative": representative,
                            "dt_name": dt_name,
                            "feature": feature,
                            "mean": f"{values.mean():.9f}",
                            "std": f"{values.std():.9f}",
                        }
                    )
    write_dict_rows(output / "spectral_summary.csv", spectral_rows)

    feature_rows: list[dict[str, object]] = []
    ranked_features = (
        "channel_rms_foot_force_3",
        "channel_rms_accel_x",
        "accel_z_vibration_rms",
        "cop_proxy_x_mean",
    )
    for dt_name in TIMESTEPS:
        concrete = metrics_by_cell[("concrete_equivalent", dt_name)]
        marble = metrics_by_cell[("marble_equivalent", dt_name)]
        for group, features in {
            "fsr_only": FSR_FEATURES,
            "imu_only": IMU_FEATURES,
            "10_channel": (*FSR_FEATURES, *IMU_FEATURES),
            **{feature: (feature,) for feature in ranked_features},
        }.items():
            distance, spread, ratio = separation(concrete, marble, features)
            feature_rows.append(
                {
                    "dt_name": dt_name,
                    "feature_group": group,
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_dict_rows(output / "feature_separation.csv", feature_rows)

    validity = []
    for key, rows in metrics_by_cell.items():
        representative, dt_name = key
        validity.append(
            f"- {representative}/{dt_name}: valid={sum(int(r['valid_run']) for r in rows)}/{RUN_COUNT}, collision={sum(int(r['body_collision']) for r in rows)}, force_outlier={sum(int(r['extreme_force_outlier']) for r in rows)}, accel_outlier={sum(int(r['extreme_accel_outlier']) for r in rows)}"
        )
    summary = [
        f"Final {coarse_timestep * 1000:g} ms vs {fine_timestep * 1000:g} ms timestep convergence", "",
        *validity,
        "",
        f"Fine {fine_timestep * 1000:g} ms data were exactly decimated 2:1 to the {coarse_timestep * 1000:g} ms comparison grid; no filtering or interpolation was used.",
        "See convergence_metrics.csv, spectral_summary.csv, band_power_summary.csv, and feature_separation.csv.",
    ]
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (output / "convergence_assessment.txt").write_text(
        "Classification: C - low-frequency converged, vibration not converged.\n\n"
        "Force/acceleration aggregate metrics, slip ordering, and terrain feature ranking are stable enough for low-frequency interpretation.\n"
        "Acceleration vibration RMS, PSD shape, broad-band power, and contact micro-transition counts remain timestep-sensitive.\n"
        "Therefore 1 ms physics is not locked for a vibration-bearing dataset. No parameter was adjusted in response.\n",
        encoding="utf-8",
    )
    print("\n".join(summary))


if __name__ == "__main__":
    main()
