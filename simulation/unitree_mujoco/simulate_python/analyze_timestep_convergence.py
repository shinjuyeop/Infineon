"""Add common-0-90-Hz analysis to an existing timestep convergence study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from run_surface_sampling_rate_study import load_csv, periodogram, run_arrays, write_dict_rows
from run_timestep_convergence_study import (
    OUTPUT_DIR,
    REPRESENTATIVES,
    TIMESTEPS,
)


def common_band_metrics(time: np.ndarray, sensors: np.ndarray, rate: float) -> dict[str, float]:
    mask = (time >= 0.20) & (time < 0.60)
    frequency, psd = periodogram(sensors[mask, 6], rate)
    common = (frequency > 0.0) & (frequency <= 90.0)
    selected_frequency = frequency[common]
    selected_psd = psd[common]
    peak = int(np.argmax(selected_psd))
    return {
        "peak_0_90_hz": float(selected_frequency[peak]),
        "centroid_0_90_hz": float(
            np.sum(selected_frequency * selected_psd) / max(selected_psd.sum(), 1e-30)
        ),
        "power_0_90": float(selected_psd.sum() * (frequency[1] - frequency[0])),
    }


def main() -> None:
    output = OUTPUT_DIR.resolve()
    conditions = [f"run_{index:03d}" for index in range(1, 11)]
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    summary_rows: list[dict[str, object]] = []
    psd_rows: list[dict[str, object]] = []
    slip_values: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for representative, (friction, _) in REPRESENTATIVES.items():
        for dt_name, timestep in TIMESTEPS.items():
            rate = 1.0 / timestep
            cell = []
            cell_psd = []
            cell_frequency = None
            for run_id in conditions:
                path = output / representative / dt_name / friction / f"{run_id}.csv"
                time, sensors, _, _ = run_arrays(path)
                metrics = common_band_metrics(time, sensors, rate)
                values[(representative, dt_name, run_id)] = metrics
                cell.append(metrics)
                mask = (time >= 0.20) & (time < 0.60)
                cell_frequency, psd = periodogram(sensors[mask, 6], rate)
                cell_psd.append(psd)
                _, csv_rows = load_csv(path)
                slip_values[(representative, dt_name, run_id)] = (
                    time,
                    np.asarray([float(row["foot_slip_displacement"]) for row in csv_rows]),
                )
            for feature in ("peak_0_90_hz", "centroid_0_90_hz", "power_0_90"):
                array = np.asarray([row[feature] for row in cell])
                summary_rows.append(
                    {
                        "representative": representative,
                        "dt_name": dt_name,
                        "feature": feature,
                        "mean": f"{array.mean():.9f}",
                        "std": f"{array.std():.9f}",
                    }
                )
            psd_matrix = np.asarray(cell_psd)
            assert cell_frequency is not None
            for index, frequency in enumerate(cell_frequency):
                psd_rows.append(
                    {
                        "representative": representative,
                        "dt_name": dt_name,
                        "frequency_hz": f"{frequency:.9f}",
                        "accel_z_psd_mean": f"{psd_matrix[:, index].mean():.12e}",
                        "accel_z_psd_std": f"{psd_matrix[:, index].std():.12e}",
                    }
                )
    write_dict_rows(output / "common_band_spectral_summary.csv", summary_rows)
    write_dict_rows(output / "accel_z_psd_summary.csv", psd_rows)

    error_rows: list[dict[str, object]] = []
    for representative in REPRESENTATIVES:
        for dt_name in ("dt_5ms", "dt_2ms"):
            for feature in ("peak_0_90_hz", "centroid_0_90_hz", "power_0_90"):
                errors = []
                for run_id in conditions:
                    value = values[(representative, dt_name, run_id)][feature]
                    reference = values[(representative, "dt_1ms", run_id)][feature]
                    errors.append(abs(value - reference) / max(abs(reference), 1e-12))
                error_rows.append(
                    {
                        "representative": representative,
                        "dt_name": dt_name,
                        "feature": feature,
                        "relative_error_mean": f"{np.mean(errors):.9f}",
                        "relative_error_std": f"{np.std(errors):.9f}",
                    }
                )
    write_dict_rows(output / "common_band_reference_errors.csv", error_rows)

    slip_rows: list[dict[str, object]] = []
    target = np.arange(1, 201, dtype=np.float64) * 0.005
    for representative in REPRESENTATIVES:
        for dt_name in TIMESTEPS:
            final_values = []
            waveform_errors = []
            for run_id in conditions:
                time, slip = slip_values[(representative, dt_name, run_id)]
                final_values.append(float(slip[-1]))
                if dt_name != "dt_1ms":
                    ref_time, ref_slip = slip_values[(representative, "dt_1ms", run_id)]
                    common = np.interp(target, time, slip)
                    reference = np.interp(target, ref_time, ref_slip)
                    waveform_errors.append(
                        float(
                            np.sqrt(np.mean((common - reference) ** 2))
                            / max(np.sqrt(np.mean(reference**2)), 1e-12)
                        )
                    )
            slip_rows.append(
                {
                    "representative": representative,
                    "dt_name": dt_name,
                    "final_slip_mean_m": f"{np.mean(final_values):.12e}",
                    "final_slip_std_m": f"{np.std(final_values):.12e}",
                    "waveform_nrmse_vs_1ms": ""
                    if not waveform_errors
                    else f"{np.mean(waveform_errors):.9f}",
                }
            )
    write_dict_rows(output / "slip_convergence.csv", slip_rows)
    lookup = {
        (str(row["representative"]), str(row["dt_name"]), str(row["feature"])): float(row["mean"])
        for row in summary_rows
    }
    errors = {
        (str(row["representative"]), str(row["dt_name"]), str(row["feature"])): float(row["relative_error_mean"])
        for row in error_rows
    }
    lines = [
        "Common-band (0-90 Hz) timestep convergence", "",
        "Assessment: D - 200 Hz artifact is likely for the previously reported 60-70 Hz dominant response.",
        "The full-band dominant response moves above 100 Hz at finer timesteps; common-band peak/power also remain timestep-sensitive.", "",
    ]
    for representative in REPRESENTATIVES:
        lines.append(representative)
        for dt_name in TIMESTEPS:
            lines.append(
                f"- {dt_name}: peak={lookup[(representative,dt_name,'peak_0_90_hz')]:.2f}Hz, centroid={lookup[(representative,dt_name,'centroid_0_90_hz')]:.2f}Hz, power={lookup[(representative,dt_name,'power_0_90')]:.6e}"
            )
        lines.append(
            f"- dt_5ms vs 1ms errors: peak={errors[(representative,'dt_5ms','peak_0_90_hz')]*100:.1f}%, centroid={errors[(representative,'dt_5ms','centroid_0_90_hz')]*100:.1f}%, power={errors[(representative,'dt_5ms','power_0_90')]*100:.1f}%"
        )
        lines.append("")
    (output / "convergence_assessment.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
