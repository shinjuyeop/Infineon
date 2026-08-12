"""Run a short terrain/speed qualification before walking dataset generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from g1_upstream_locomotion import TESTED_POLICY_SHA256, UPSTREAM_REVISION
from run_surface_sampling_rate_study import SIMULATION_DIR, write_dict_rows
from run_walking_touchdown_dataset_v1 import (
    CONTACT_MODELS,
    CONTACT_PARAMETER_MODES,
    _run_one,
)


OUTPUT_DIR = SIMULATION_DIR / "outputs" / "terrain_walking_stability_sweep_v1"
DEFAULT_TERRAINS = ("ice", "sand")
DEFAULT_SPEEDS_MPS = (0.10, 0.15, 0.20)
DEFAULT_RUNS_PER_CONDITION = 2
DEFAULT_DURATION_S = 3.0
DEFAULT_SETTLING_S = 0.6
MIN_TOUCHDOWNS = 3
MIN_FORWARD_PROGRESS_RATIO = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--terrains", nargs="+", choices=("concrete", "marble", "ice", "sand"),
        default=list(DEFAULT_TERRAINS),
    )
    parser.add_argument(
        "--speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS_MPS)
    )
    parser.add_argument("--runs-per-condition", type=int, default=DEFAULT_RUNS_PER_CONDITION)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--settling-s", type=float, default=DEFAULT_SETTLING_S)
    parser.add_argument("--minimum-touchdowns", type=int, default=MIN_TOUCHDOWNS)
    parser.add_argument(
        "--minimum-forward-progress-ratio",
        type=float,
        default=MIN_FORWARD_PROGRESS_RATIO,
        help="minimum actual/commanded x displacement ratio for a passing run",
    )
    parser.add_argument("--contact-model", choices=CONTACT_MODELS, default="full-body")
    parser.add_argument(
        "--contact-parameters",
        choices=CONTACT_PARAMETER_MODES,
        default="terrain-native",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def run_passes(
    row: dict[str, object],
    minimum_touchdowns: int,
    minimum_forward_progress_ratio: float = 0.0,
) -> bool:
    commanded_displacement = (
        float(row.get("walking_speed_mps", 0.0))
        * float(row.get("duration_s", 0.0))
    )
    progress_passes = (
        commanded_displacement <= 0.0
        or float(row.get("forward_displacement_m", 0.0))
        >= minimum_forward_progress_ratio * commanded_displacement
    )
    return bool(
        int(row["run_valid"]) == 1
        and int(row["valid_events"]) >= minimum_touchdowns
        and int(row["stopped_early"]) == 0
        and progress_passes
    )


def summarize_stability(
    rows: list[dict[str, object]],
    minimum_touchdowns: int,
    minimum_forward_progress_ratio: float = 0.0,
) -> tuple[list[dict[str, object]], list[float]]:
    matrix: list[dict[str, object]] = []
    terrains = tuple(dict.fromkeys(str(row["terrain_name"]) for row in rows))
    speeds = sorted({float(row["walking_speed_mps"]) for row in rows})
    for terrain in terrains:
        for speed in speeds:
            group = [
                row for row in rows
                if str(row["terrain_name"]) == terrain
                and np.isclose(float(row["walking_speed_mps"]), speed)
            ]
            if not group:
                continue
            failures = sorted(
                {str(row["failure_reason"]) for row in group if row["failure_reason"]}
            )
            failure_times = [
                float(row["failure_time_s"])
                for row in group if row["failure_time_s"] != ""
            ]
            pass_count = sum(
                run_passes(row, minimum_touchdowns, minimum_forward_progress_ratio)
                for row in group
            )
            matrix.append(
                {
                    "terrain_name": terrain,
                    "walking_speed_mps": speed,
                    "runs": len(group),
                    "pass_runs": pass_count,
                    "all_pass": int(pass_count == len(group)),
                    "touchdown_events": sum(int(row["touchdown_events"]) for row in group),
                    "valid_events": sum(int(row["valid_events"]) for row in group),
                    "minimum_touchdowns_per_run": minimum_touchdowns,
                    "minimum_forward_progress_ratio": minimum_forward_progress_ratio,
                    "earliest_failure_time_s": "" if not failure_times else min(failure_times),
                    "failure_reasons": "|".join(failures),
                    "min_base_height_m": min(float(row["min_base_height_m"]) for row in group),
                    "min_upright_z": min(float(row["min_upright_z"]) for row in group),
                    "mean_forward_displacement_m": float(
                        np.mean([float(row["forward_displacement_m"]) for row in group])
                    ),
                }
            )
    common_pass_speeds = [
        speed for speed in speeds
        if all(
            any(
                row["terrain_name"] == terrain
                and np.isclose(float(row["walking_speed_mps"]), speed)
                and int(row["all_pass"]) == 1
                for row in matrix
            )
            for terrain in terrains
        )
    ]
    return matrix, common_pass_speeds


def main() -> None:
    args = parse_args()
    if args.runs_per_condition <= 0 or args.minimum_touchdowns <= 0:
        raise ValueError("run count and minimum touchdowns must be positive")
    if not 0.0 <= args.minimum_forward_progress_ratio <= 1.0:
        raise ValueError("minimum forward progress ratio must be within [0,1]")
    if any(speed < 0.1 or speed > 0.5 for speed in args.speeds):
        raise ValueError("speeds must be within the policy gait range [0.1,0.5]")
    plan = {
        "name": "terrain_walking_stability_sweep_v1",
        "purpose": "qualify locomotion before generating walking terrain events",
        "terrains": args.terrains,
        "speeds_mps": args.speeds,
        "runs_per_condition": args.runs_per_condition,
        "duration_s": args.duration_s,
        "settling_s": args.settling_s,
        "minimum_touchdowns_per_run": args.minimum_touchdowns,
        "minimum_forward_progress_ratio": args.minimum_forward_progress_ratio,
        "contact_model": args.contact_model,
        "contact_parameters": args.contact_parameters,
        "candidate_runs": len(args.terrains) * len(args.speeds) * args.runs_per_condition,
        "stop_on_fall": True,
        "pass_gate": "run_valid=1, no early stop, minimum touchdown count, and forward progress",
        "upstream_revision": UPSTREAM_REVISION,
        "tested_policy_sha256": TESTED_POLICY_SHA256,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("Dry run only. Pass --execute and --policy-path to run the sweep.")
        return
    if args.policy_path is None or not args.policy_path.is_file():
        raise FileNotFoundError("--policy-path must identify the pinned ONNX policy")
    policy_path = args.policy_path.resolve()
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if policy_hash != TESTED_POLICY_SHA256:
        raise ValueError(f"policy hash mismatch: {policy_hash}")
    output = (args.output_dir or OUTPUT_DIR).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for terrain in args.terrains:
        for speed in args.speeds:
            for run_index in range(args.runs_per_condition):
                run_args = argparse.Namespace(
                    duration_s=args.duration_s,
                    settling_s=args.settling_s,
                    walking_speed=float(speed),
                    stop_on_fall=True,
                    gui=False,
                    contact_model=args.contact_model,
                    contact_parameters=args.contact_parameters,
                )
                _, row = _run_one(
                    run_args,
                    policy_path,
                    terrain,
                    "multisine",
                    0,
                    run_index,
                )
                row["stability_pass"] = int(
                    run_passes(
                        row,
                        args.minimum_touchdowns,
                        args.minimum_forward_progress_ratio,
                    )
                )
                rows.append(row)
                print(
                    f"terrain={terrain} speed={speed:.2f} run={run_index} "
                    f"pass={row['stability_pass']} valid_td={row['valid_events']} "
                    f"failure={row['failure_reason'] or 'none'} "
                    f"failure_t={row['failure_time_s'] or 'none'}"
                )

    matrix, common_pass_speeds = summarize_stability(
        rows, args.minimum_touchdowns, args.minimum_forward_progress_ratio
    )
    write_dict_rows(output / "stability_runs.csv", rows)
    write_dict_rows(output / "stability_matrix.csv", matrix)
    plan["policy_path"] = str(policy_path)
    plan["policy_sha256"] = policy_hash
    plan["measured"] = {
        "runs": len(rows),
        "pass_runs": sum(int(row["stability_pass"]) for row in rows),
        "common_pass_speeds_mps": common_pass_speeds,
    }
    (output / "protocol.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "terrain_walking_stability_sweep_v1",
        f"runs={len(rows)} pass={sum(int(row['stability_pass']) for row in rows)}",
        "common_pass_speeds_mps="
        + (",".join(f"{speed:.2f}" for speed in common_pass_speeds) or "none"),
    ]
    for row in matrix:
        summary.append(
            f"{row['terrain_name']}@{float(row['walking_speed_mps']):.2f}: "
            f"pass={row['pass_runs']}/{row['runs']} "
            f"valid_td={row['valid_events']} failures={row['failure_reasons'] or 'none'}"
        )
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
