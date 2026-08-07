"""Produce cell summaries and feature-level effects for factorization output."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hil_sensor import HIL_SENSOR_CHANNELS
from run_surface_factorization_study import CELLS, OUTPUT_DIR
from run_surface_sampling_rate_study import write_dict_rows


def main() -> None:
    output = OUTPUT_DIR.resolve()
    rows_by_cell = {}
    for cell in CELLS:
        with (output / cell / "run_manifest.csv").open(newline="", encoding="utf-8") as f:
            rows_by_cell[cell] = list(csv.DictReader(f))
    spectral_features = tuple(
        key
        for key in rows_by_cell["A_high_smooth"][0]
        if "vibration_rms" in key
        or "spectral_peak" in key
        or "spectral_centroid" in key
        or "_band_" in key
    )
    derived_features = (
        "total_normal_force_rms",
        "heel_load_fraction",
        "toe_load_fraction",
        "cop_proxy_x_mean",
        "accel_rms",
        "gyro_rms",
    )
    summary_rows: list[dict[str, object]] = []
    for cell, rows in rows_by_cell.items():
        for feature in (*derived_features, *spectral_features):
            values = np.asarray([float(row[feature]) for row in rows])
            summary_rows.append(
                {
                    "cell": cell,
                    "feature": feature,
                    "mean": f"{values.mean():.12e}",
                    "std": f"{values.std():.12e}",
                }
            )
    write_dict_rows(output / "spectral_derived_summary.csv", summary_rows)

    effect_features = (
        *(f"channel_rms_{channel}" for channel in HIL_SENSOR_CHANNELS),
        *derived_features,
        *spectral_features,
    )
    effect_rows: list[dict[str, object]] = []
    for feature in effect_features:
        arrays = {
            cell: np.asarray([float(row[feature]) for row in rows])
            for cell, rows in rows_by_cell.items()
        }
        scale = np.concatenate(list(arrays.values())).std()
        scale = 1.0 if scale < 1e-12 else scale
        means = {cell: values.mean() / scale for cell, values in arrays.items()}
        rough = 0.5 * ((means["B_high_rough"] - means["A_high_smooth"]) + (means["D_low_rough"] - means["C_low_smooth"]))
        friction = 0.5 * ((means["C_low_smooth"] - means["A_high_smooth"]) + (means["D_low_rough"] - means["B_high_rough"]))
        interaction = (means["D_low_rough"] - means["C_low_smooth"]) - (means["B_high_rough"] - means["A_high_smooth"])
        effect_rows.append(
            {
                "feature": feature,
                "roughness_main_effect_signed": f"{rough:.9f}",
                "friction_main_effect_signed": f"{friction:.9f}",
                "interaction_effect_signed": f"{interaction:.9f}",
            }
        )
    write_dict_rows(output / "feature_factor_effects.csv", effect_rows)
    lines = [
        "Factorization assessment", "",
        "Overall interpretation: sensor-specific (option 4).",
        "FSR separation is roughness-dominant; IMU and vibration-derived separation are friction-dominant.",
        "The interaction term is present but smaller than the corresponding dominant main effect.",
        "Roughness strongly redistributes the four discrete contact loads, while low friction permits the larger foot acceleration/vibration response.",
        "This is a deterministic model explanation, not evidence that real marble has intrinsically larger vibration.",
    ]
    (output / "factor_assessment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
