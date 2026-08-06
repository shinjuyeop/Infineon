"""Collect paired +X/-X pulse runs across deterministic independent sessions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import config
from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from run_controlled_terrain_dataset import write_manifest
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window
from terrain_profiles import TERRAIN_PROFILES


DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "bidirectional_pulse_validation"
DEFAULT_SESSION_SEEDS = (20260805, 20260806, 20260807)
DIRECTIONS = {
    "positive_x": (1.0, 0.0),
    "negative_x": (-1.0, 0.0),
}


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("session seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("session seeds must be non-empty and unique")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--session-seeds", type=parse_seeds, default=DEFAULT_SESSION_SEEDS)
    parser.add_argument("--support-ratio", type=float, default=0.70)
    parser.add_argument("--pulse-start", type=float, default=0.25)
    parser.add_argument("--pulse-duration", type=float, default=0.20)
    parser.add_argument("--pulse-magnitude", type=float, default=80.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_new_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite existing data: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_protocol(path: Path, args: argparse.Namespace) -> None:
    rows = [
        ("robot", config.ROBOT),
        ("scene", config.ROBOT_SCENE),
        ("physics_timestep", f"{config.SIMULATE_DT:.9f}"),
        ("duration", f"{args.duration:.9f}"),
        ("sample_rate", f"{args.sample_rate:.9f}"),
        ("support_ratio", f"{args.support_ratio:.9f}"),
        ("pulse_waveform", "half_sine"),
        ("pulse_start", f"{args.pulse_start:.9f}"),
        ("pulse_duration", f"{args.pulse_duration:.9f}"),
        ("pulse_magnitude", f"{args.pulse_magnitude:.9f}"),
        ("session_seeds", ";".join(str(seed) for seed in args.session_seeds)),
        ("runs_per_terrain_direction", str(args.runs)),
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("parameter", "value"))
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if config.ROBOT != "g1":
        raise ValueError('config.ROBOT must remain "g1"')
    if args.runs <= 0 or args.duration <= 0.0 or args.sample_rate <= 0.0:
        raise ValueError("runs, duration, and sample rate must be positive")
    if not np.isclose(args.duration * args.sample_rate, round(args.duration * args.sample_rate)):
        raise ValueError("duration * sample-rate must be an integer")
    if args.pulse_start + args.pulse_duration >= args.duration:
        raise ValueError("pulse must end before the run window")

    output_dir = args.output_dir.resolve()
    ensure_new_output_directory(output_dir)
    write_protocol(output_dir / "protocol.csv", args)
    manifest_rows: list[dict[str, float | int | str]] = []

    for session_index, seed in enumerate(args.session_seeds, start=1):
        session_id = f"session_{session_index:02d}"
        conditions = generate_pulse_conditions(args.runs, seed)
        for direction_name, (direction_x, direction_y) in DIRECTIONS.items():
            pulse = HorizontalPulse(
                args.pulse_start,
                args.pulse_duration,
                args.pulse_magnitude,
                direction_x,
                direction_y,
            )
            subgroup_dir = output_dir / session_id / direction_name
            subgroup_rows: list[dict[str, float | int | str]] = []
            for terrain in TERRAIN_PROFILES:
                print(
                    f"collecting session={session_id} seed={seed} "
                    f"direction={direction_name} terrain={terrain} runs={args.runs}"
                )
                terrain_rows = []
                for condition in conditions:
                    metrics = run_window(
                        terrain,
                        condition,
                        args.duration,
                        args.sample_rate,
                        seed,
                        args.support_ratio,
                        pulse,
                        subgroup_dir,
                        session_id=session_id,
                        pulse_direction_label=direction_name,
                    )
                    metrics["csv_path"] = str(
                        Path(session_id) / direction_name / metrics["csv_path"]
                    )
                    terrain_rows.append(metrics)
                    subgroup_rows.append(metrics)
                    manifest_rows.append(metrics)
                write_manifest(subgroup_dir / terrain / "manifest.csv", terrain_rows)
                valid_count = sum(int(row["valid_run"]) for row in terrain_rows)
                print(f"completed {session_id}/{direction_name}/{terrain}: valid={valid_count}/{args.runs}")
            write_manifest(subgroup_dir / "manifest.csv", subgroup_rows)

    write_manifest(output_dir / "manifest.csv", manifest_rows)
    print(f"manifest={output_dir / 'manifest.csv'}")
    print(f"total_runs={len(manifest_rows)}")


if __name__ == "__main__":
    main()
