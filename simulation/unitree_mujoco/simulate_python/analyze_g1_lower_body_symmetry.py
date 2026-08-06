"""Compare reduced-model nominal ±X smoke results with the preserved full-body baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_horizontal_pulse_dataset import SIMULATION_DIR


LOWER_DATA = SIMULATION_DIR / "outputs" / "g1_lower_body_symmetry_smoke"
FULL_DATA = SIMULATION_DIR / "outputs" / "bidirectional_pulse_validation"
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "g1_lower_body_symmetry_analysis"
TERRAINS = ("concrete", "ice")
DIRECTIONS = ("positive_x", "negative_x")
METRICS = (
    "pulse_slip_displacement", "pulse_max_foot_speed",
    "pulse_peak_tangential_force", "pulse_contact_fraction",
    "pulse_accel_rms", "pulse_gyro_rms",
    "pulse_force_distribution_1", "pulse_force_distribution_2",
    "pulse_force_distribution_3", "pulse_force_distribution_4",
)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def summarize(model_name: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for terrain in TERRAINS:
        for direction in DIRECTIONS:
            group = [row for row in rows if row["terrain"] == terrain and row["pulse_direction"] == direction]
            valid = [row for row in group if int(row["valid_run"])]
            item: dict[str, object] = {
                "model": model_name, "terrain": terrain, "pulse_direction": direction,
                "total_runs": len(group), "valid_runs": len(valid),
                "body_collision_count": sum(int(row["body_collision"]) for row in group),
                "extreme_force_count": sum(int(row["extreme_force_outlier"]) for row in group),
                "extreme_accel_count": sum(int(row["extreme_accel_outlier"]) for row in group),
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in valid])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std())
            output.append(item)
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(rows: list[dict[str, object]], model: str, terrain: str, direction: str) -> dict[str, object]:
    return next(
        row for row in rows
        if row["model"] == model and row["terrain"] == terrain and row["pulse_direction"] == direction
    )


def asymmetry(positive: float, negative: float) -> float:
    return max(abs(positive), abs(negative)) / max(min(abs(positive), abs(negative)), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-data", type=Path, default=LOWER_DATA)
    parser.add_argument("--full-data", type=Path, default=FULL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = [
        *summarize("full_g1_29dof", load(args.full_data.resolve() / "manifest.csv")),
        *summarize("g1_lower_body_12dof", load(args.lower_data.resolve() / "manifest.csv")),
    ]
    write_rows(output_dir / "direction_metric_summary.csv", summary)
    comparison = []
    for terrain in TERRAINS:
        for model in ("full_g1_29dof", "g1_lower_body_12dof"):
            positive = group(summary, model, terrain, "positive_x")
            negative = group(summary, model, terrain, "negative_x")
            slip_pos = float(positive["pulse_slip_displacement_mean"])
            slip_neg = float(negative["pulse_slip_displacement_mean"])
            force_pos = np.asarray([float(positive[f"pulse_force_distribution_{i}_mean"]) for i in range(1, 5)])
            force_neg = np.asarray([float(negative[f"pulse_force_distribution_{i}_mean"]) for i in range(1, 5)])
            comparison.append(
                {
                    "model": model,
                    "terrain": terrain,
                    "slip_positive_x_m": slip_pos,
                    "slip_negative_x_m": slip_neg,
                    "slip_asymmetry_ratio": asymmetry(slip_pos, slip_neg),
                    "force_distribution_l1_direction_difference": float(np.abs(force_pos - force_neg).sum()),
                    "accel_rms_asymmetry_ratio": asymmetry(
                        float(positive["pulse_accel_rms_mean"]), float(negative["pulse_accel_rms_mean"])
                    ),
                    "gyro_rms_asymmetry_ratio": asymmetry(
                        float(positive["pulse_gyro_rms_mean"]), float(negative["pulse_gyro_rms_mean"])
                    ),
                }
            )
    write_rows(output_dir / "direction_asymmetry_comparison.csv", comparison)

    lines = ["G1 lower-body nominal symmetry smoke test", ""]
    for terrain in TERRAINS:
        lines.append(f"{terrain}:")
        for model in ("full_g1_29dof", "g1_lower_body_12dof"):
            item = next(row for row in comparison if row["model"] == model and row["terrain"] == terrain)
            positive = group(summary, model, terrain, "positive_x")
            negative = group(summary, model, terrain, "negative_x")
            lines.append(
                f"- {model}: valid +X={positive['valid_runs']}/{positive['total_runs']}, "
                f"-X={negative['valid_runs']}/{negative['total_runs']}, "
                f"slip +X={float(item['slip_positive_x_m'])*1000:.6f}mm, "
                f"-X={float(item['slip_negative_x_m'])*1000:.6f}mm, "
                f"ratio={float(item['slip_asymmetry_ratio']):.6f}, "
                f"force_distribution_L1={float(item['force_distribution_l1_direction_difference']):.6f}"
            )
    lines.extend((
        "",
        "Full-body rows are the preserved 3-session/20-run baseline with seeded pose variation.",
        "Lower-body rows are five repeated nominal-pose deterministic runs per terrain/direction.",
        "The comparison is descriptive and not a matched statistical experiment.",
    ))
    (output_dir / "symmetry_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
