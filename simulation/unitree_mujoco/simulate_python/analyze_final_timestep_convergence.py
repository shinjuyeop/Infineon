"""Finalize band-power tables from the completed final convergence runs."""

from __future__ import annotations

import csv

import numpy as np

from run_final_timestep_convergence import (
    BANDS,
    OUTPUT_DIR,
    REPRESENTATIVES,
    SPECTRAL_AXES,
)
from run_surface_sampling_rate_study import write_dict_rows


def main() -> None:
    output = OUTPUT_DIR.resolve()
    with (output / "run_level_metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result: list[dict[str, object]] = []
    for representative in REPRESENTATIVES:
        for axis in SPECTRAL_AXES:
            for low, high in BANDS:
                feature = f"{axis}_band_{low}_{high}_power"
                values_by_dt = {}
                for dt_name in ("dt_1ms", "dt_0p5ms"):
                    values = np.asarray(
                        [
                            float(row[feature])
                            for row in rows
                            if row["representative"] == representative
                            and row["dt_name"] == dt_name
                        ]
                    )
                    values_by_dt[dt_name] = values
                    result.append(
                        {
                            "representative": representative,
                            "axis": axis,
                            "band_hz": f"{low}-{high}",
                            "row_type": dt_name,
                            "power_mean": f"{values.mean():.12e}",
                            "power_std": f"{values.std():.12e}",
                            "relative_error_mean": "",
                        }
                    )
                coarse = values_by_dt["dt_1ms"]
                fine = values_by_dt["dt_0p5ms"]
                errors = np.abs(coarse - fine) / np.maximum(np.abs(fine), 1e-12)
                result.append(
                    {
                        "representative": representative,
                        "axis": axis,
                        "band_hz": f"{low}-{high}",
                        "row_type": "dt_1ms_vs_dt_0p5ms",
                        "power_mean": "",
                        "power_std": "",
                        "relative_error_mean": f"{errors.mean():.9f}",
                    }
                )
    write_dict_rows(output / "band_power_summary.csv", result)


if __name__ == "__main__":
    main()
