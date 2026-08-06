"""Run a nominal concrete/ice ±X smoke test on the reduced G1 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controlled_excitation import ExcitationCondition, HorizontalPulse
from run_controlled_terrain_dataset import write_manifest
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window


LOWER_SCENE = SIMULATION_DIR / "models" / "g1_lower_body" / "scene.xml"
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "g1_lower_body_symmetry_smoke"
DEFAULT_GUI_OUTPUT = SIMULATION_DIR / "outputs" / "g1_lower_body_gui_preview"
TERRAINS = ("concrete", "ice")
DIRECTIONS = {"positive_x": (1.0, 0.0), "negative_x": (-1.0, 0.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=(*TERRAINS, "all"), default="all")
    parser.add_argument("--direction", choices=(*DIRECTIONS, "both"), default="both")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--support-ratio", type=float, default=0.70)
    parser.add_argument("--pulse-start", type=float, default=0.25)
    parser.add_argument("--pulse-duration", type=float, default=0.20)
    parser.add_argument("--pulse-magnitude", type=float, default=80.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=0.5)
    parser.add_argument("--gui-hold-seconds", type=float, default=5.0)
    return parser.parse_args()


def nominal_conditions(count: int) -> tuple[ExcitationCondition, ...]:
    return tuple(
        ExcitationCondition(f"run_{index:03d}", 0.0, 0.0, 0.0, 0.0, 0.0)
        for index in range(1, count + 1)
    )


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.duration <= 0.0 or args.sample_rate <= 0.0:
        raise ValueError("runs, duration, and sample rate must be positive")
    if not np.isclose(args.duration * args.sample_rate, round(args.duration * args.sample_rate)):
        raise ValueError("duration * sample-rate must be an integer")
    if args.gui and (args.runs != 1 or args.terrain == "all" or args.direction == "both"):
        raise ValueError("--gui requires --runs 1, one terrain, and one direction")
    output_dir = (
        DEFAULT_GUI_OUTPUT if args.gui and args.output_dir == DEFAULT_OUTPUT else args.output_dir
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    terrains = TERRAINS if args.terrain == "all" else (args.terrain,)
    directions = tuple(DIRECTIONS) if args.direction == "both" else (args.direction,)
    conditions = nominal_conditions(args.runs)
    rows = []
    for direction_name in directions:
        direction_x, direction_y = DIRECTIONS[direction_name]
        pulse = HorizontalPulse(
            args.pulse_start, args.pulse_duration, args.pulse_magnitude,
            direction_x, direction_y,
        )
        direction_rows = []
        for terrain in terrains:
            terrain_rows = []
            print(f"lower-body terrain={terrain} direction={direction_name} runs={args.runs}")
            for condition in conditions:
                metrics = run_window(
                    terrain,
                    condition,
                    args.duration,
                    args.sample_rate,
                    0,
                    args.support_ratio,
                    pulse,
                    output_dir / direction_name,
                    gui=args.gui,
                    realtime_factor=args.realtime_factor,
                    gui_hold_seconds=args.gui_hold_seconds,
                    session_id="nominal",
                    pulse_direction_label=direction_name,
                    scene_path=LOWER_SCENE,
                    support_body_name="equivalent_upper_body",
                    support_site_name="support_point",
                    pulse_body_name="equivalent_upper_body",
                    pulse_site_name="pulse_point",
                )
                metrics["model"] = "g1_lower_body_12dof"
                metrics["csv_path"] = str(Path(direction_name) / metrics["csv_path"])
                terrain_rows.append(metrics)
                direction_rows.append(metrics)
                rows.append(metrics)
                print(
                    f"  {condition.run_id}: valid={metrics['valid_run']} "
                    f"contact={float(metrics['pulse_contact_fraction']):.2f} "
                    f"slip={float(metrics['pulse_slip_displacement'])*1000:.4f}mm "
                    f"collision={metrics['body_collision']}"
                )
            write_manifest(output_dir / direction_name / terrain / "manifest.csv", terrain_rows)
        write_manifest(output_dir / direction_name / "manifest.csv", direction_rows)
    write_manifest(output_dir / "manifest.csv", rows)
    print(f"manifest={output_dir / 'manifest.csv'}")
    print(f"valid={sum(int(row['valid_run']) for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
