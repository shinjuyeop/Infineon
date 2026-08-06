"""Small reproducible sweep used to select a safe horizontal pulse magnitude."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=(80, 90, 100, 120, 160, 200, 240))
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--pulse-start", type=float, default=0.25)
    parser.add_argument("--pulse-duration", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=SIMULATION_DIR / "outputs" / "horizontal_pulse_sweep.csv")
    args = parser.parse_args()

    conditions = generate_pulse_conditions(args.runs, args.seed)
    rows = []
    for magnitude in args.magnitudes:
        pulse = HorizontalPulse(args.pulse_start, args.pulse_duration, magnitude)
        for terrain in ("concrete", "marble", "ice"):
            metrics = [
                run_window(terrain, condition, 1.0, 50.0, args.seed, 0.70, pulse, None)
                for condition in conditions
            ]
            slips = np.asarray([float(row["pulse_slip_displacement"]) for row in metrics])
            rows.append(
                {
                    "pulse_magnitude": magnitude,
                    "terrain": terrain,
                    "runs": args.runs,
                    "valid_runs": sum(int(row["valid_run"]) for row in metrics),
                    "body_collisions": sum(int(row["body_collision"]) for row in metrics),
                    "pulse_contact_fraction_mean": np.mean([float(row["pulse_contact_fraction"]) for row in metrics]),
                    "pulse_slip_mean_m": slips.mean(),
                    "pulse_slip_std_m": slips.std(),
                    "pulse_max_foot_speed_mean": np.mean([float(row["pulse_max_foot_speed"]) for row in metrics]),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"sweep={args.output.resolve()}")


if __name__ == "__main__":
    main()
