"""Run the terminal matched 0.5 ms versus 0.25 ms convergence check."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import analyze_final_timestep_convergence as analyzer
import run_final_timestep_convergence as study


OUTPUT_DIR = study.SIMULATION_DIR / "outputs" / "final_025ms_convergence"
TIMESTEPS = {"dt_0p5ms": 0.0005, "dt_0p25ms": 0.00025}


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_terminal_assessment() -> None:
    convergence = read_rows("convergence_metrics.csv")
    bands = read_rows("band_power_summary.csv")
    runs = read_rows("run_level_metrics.csv")

    def metric(representative: str, name: str) -> float:
        row = next(
            row
            for row in convergence
            if row["representative"] == representative and row["metric"] == name
        )
        return float(row["relative_error_mean"])

    def band(representative: str, axis: str, frequency_band: str) -> float:
        row = next(
            row
            for row in bands
            if row["representative"] == representative
            and row["axis"] == axis
            and row["band_hz"] == frequency_band
            and row["row_type"] == "dt_0p5ms_vs_dt_0p25ms"
        )
        return float(row["relative_error_mean"])

    def contacts(representative: str, dt_name: str, feature: str) -> float:
        values = [
            float(row[feature])
            for row in runs
            if row["representative"] == representative and row["dt_name"] == dt_name
        ]
        return float(np.mean(values))

    concrete = "concrete_equivalent"
    marble = "marble_equivalent"
    assessment = f"""CASE B — high-frequency vibration is not converged.

The 0.5 ms -> 0.25 ms mean relative errors for accel-Z vibration RMS are {metric(concrete, 'accel_z_vibration_rms'):.1%} (concrete) and {metric(marble, 'accel_z_vibration_rms'):.1%} (marble); accel-X errors are {metric(concrete, 'accel_x_vibration_rms'):.1%} and {metric(marble, 'accel_x_vibration_rms'):.1%}.
Accel-Z PSD shape correlation over 0-450 Hz is {metric(concrete, 'accel_z_psd_shape_correlation_0_450'):.3f} (concrete) and {metric(marble, 'accel_z_psd_shape_correlation_0_450'):.3f} (marble).
Accel-Z 200-400 Hz band-power relative error is {band(concrete, 'accel_z', '200-400'):.1%} (concrete) and {band(marble, 'accel_z', '200-400'):.1%} (marble).
Mean contact loss/re-entry counts change from {contacts(concrete, 'dt_0p5ms', 'contact_loss_count'):.1f}/{contacts(concrete, 'dt_0p5ms', 'contact_reentry_count'):.1f} to {contacts(concrete, 'dt_0p25ms', 'contact_loss_count'):.1f}/{contacts(concrete, 'dt_0p25ms', 'contact_reentry_count'):.1f} for concrete and from {contacts(marble, 'dt_0p5ms', 'contact_loss_count'):.1f}/{contacts(marble, 'dt_0p5ms', 'contact_reentry_count'):.1f} to {contacts(marble, 'dt_0p25ms', 'contact_loss_count'):.1f}/{contacts(marble, 'dt_0p25ms', 'contact_reentry_count'):.1f} for marble.

Aggregate force/acceleration and terrain ordering remain suitable for low-frequency interpretation, but absolute foot-vibration waveform/PSD remains timestep-dependent. Stop timestep exploration here; do not test 0.125 ms or smaller and do not tune contact parameters to force convergence.

Recommended split: use MuJoCo for friction, slip, normal load, load distribution, CoP, and low-frequency foot dynamics; use a separate vibration sensor model for surface-induced high-frequency vibration, IMU mounting/bandwidth/filtering, and sensor noise.
"""
    (OUTPUT_DIR / "convergence_assessment.txt").write_text(
        assessment, encoding="utf-8"
    )


def main() -> None:
    study.OUTPUT_DIR = OUTPUT_DIR
    study.RUN_COUNT = 5
    study.TIMESTEPS = TIMESTEPS
    study.main()

    analyzer.OUTPUT_DIR = OUTPUT_DIR
    analyzer.TIMESTEPS = TIMESTEPS
    analyzer.main()
    write_terminal_assessment()


if __name__ == "__main__":
    main()
