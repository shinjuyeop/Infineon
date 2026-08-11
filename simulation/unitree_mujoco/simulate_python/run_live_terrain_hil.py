#!/usr/bin/env python3
"""Stream live Dataset-v1 MuJoCo sensors to a physical E84 over TRN2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

from controlled_excitation import ExcitationCondition, HorizontalPulse
from expanded_terrain_dataset_v1 import (
    SURFACE_FAMILIES,
    configure_expanded_run_surface,
    make_expanded_run_specification,
    make_expanded_surface_parameters,
)
from hil_sensor import HIL_SENSOR_CHANNELS
from run_horizontal_pulse_dataset import SIMULATION_DIR, run_window
from run_surface_sampling_rate_study import SURFACE_SCENE_PATH
from terrain_dataset_v1 import (
    DURATION_S,
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    TERRAIN_LABELS,
    RunSpecification,
)


REPO = SIMULATION_DIR.parent
DEPLOY_TOOLS = REPO / "Infineon_test/projects/terrain_e84_deploy/tools"
sys.path.insert(0, str(DEPLOY_TOOLS))

from terrain_preprocessing import quantize_physical  # noqa: E402
from terrain_shadow import TerrainShadowModel  # noqa: E402
from terrain_stream_client import (  # noqa: E402
    DEFAULT_PORT,
    TerrainStreamLink,
    describe,
)


CLASS_NAMES = tuple(TERRAIN_LABELS)
PHYSICAL_COLUMNS = tuple(HIL_SENSOR_CHANNELS)
QUANTIZED_COLUMNS = tuple(f"q_{name}" for name in HIL_SENSOR_CHANNELS)
RESULT_COLUMNS = tuple(f"e84_raw{index}" for index in range(4))
HOST_COLUMNS = tuple(f"host_raw{index}" for index in range(4))
CSV_COLUMNS = (
    "run_id", "session_id", "terrain_name", "terrain_class",
    "surface_family", "surface_index", "run_index", "sensor_variant",
    "simulation_time_s", "host_wall_time_s", "sequence",
    *PHYSICAL_COLUMNS, *QUANTIZED_COLUMNS,
    "fill", "warmup", "inferred", "medium_response_aligned",
    "e84_class", *RESULT_COLUMNS, "host_class", *HOST_COLUMNS,
    "host_e84_parity", "cpu_cycles", "npu_cycles", "rtt_ms",
    "send_lateness_ms", "deadline_miss", "device_error_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=(*TERRAIN_LABELS, "all"), required=True)
    parser.add_argument(
        "--family",
        choices=tuple(family.name for family in SURFACE_FAMILIES),
        default="multisine",
        help="existing Expanded Dataset-v1 surface family (default: multisine)",
    )
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-hold-seconds", type=float, default=2.0)
    return parser.parse_args()


class LiveTerrainSink:
    """Pace, send, shadow-infer, compare, and log one simulation run."""

    def __init__(
        self,
        link: TerrainStreamLink,
        shadow: TerrainShadowModel,
        output_path: Path,
        spec: RunSpecification,
        family: str,
        run_index: int,
        stride: int,
    ) -> None:
        self.link = link
        self.shadow = shadow
        self.output_path = output_path
        self.spec = spec
        self.family = family
        self.run_index = run_index
        self.stride = stride
        self.period = 1.0 / SENSOR_RATE_HZ
        self.quantized_ring: deque[np.ndarray] = deque(maxlen=50)
        self.rows: list[dict[str, object]] = []
        self.send_times: list[float] = []
        self.session_start = 0.0
        self.device_errors = 0
        self.deadline_misses = 0
        self.parity_failures = 0

    def start_session(self) -> None:
        self.link.start_session()
        self.quantized_ring.clear()
        self.rows.clear()
        self.send_times.clear()
        self.device_errors = 0
        self.deadline_misses = 0
        self.parity_failures = 0
        # Start pacing at the first produced sensor sample, excluding model/XML
        # construction from the 100 Hz streaming deadline.
        self.session_start = 0.0

    def __call__(
        self, sample_index: int, simulation_time_s: float, physical: np.ndarray
    ) -> None:
        if sample_index != self.link.next_sequence:
            raise RuntimeError(
                f"live callback sequence mismatch: sample={sample_index} "
                f"link={self.link.next_sequence}"
            )
        if sample_index == 0:
            self.session_start = time.monotonic()
        target = self.session_start + sample_index * self.period
        remaining = target - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

        quantized = quantize_physical(physical)
        self.quantized_ring.append(quantized.copy())
        exchange = self.link.send_quantized(quantized, self.stride)
        result = exchange.result
        self.send_times.append(exchange.send_time)
        self.device_errors += len(exchange.device_errors)
        send_lateness_ms = (exchange.send_time - target) * 1000.0
        deadline_miss = int(
            exchange.receive_time > target + self.period
            or exchange.send_time - target > self.period
        )
        self.deadline_misses += deadline_miss

        expected_fill = min(sample_index + 1, 50)
        expected_warmup = int(sample_index < 49)
        expected_inferred = int(
            sample_index >= 49
            and (sample_index - 49) % self.stride == 0
        )
        if (
            result["fill"] != expected_fill
            or result["warmup"] != expected_warmup
            or result["inferred"] != expected_inferred
        ):
            raise RuntimeError(
                f"E84 ring cadence mismatch at sequence {sample_index}: {result}"
            )

        e84_raw = np.asarray(
            [result[f"raw{index}"] for index in range(4)], dtype=np.int8
        )
        host_raw: np.ndarray | None = None
        host_class: int | None = None
        parity: int | None = None
        medium_aligned = 0
        if result["inferred"]:
            if len(self.quantized_ring) != 50:
                raise RuntimeError("Host ring was not full when E84 inferred")
            window = np.asarray(self.quantized_ring, dtype=np.int8)
            host_raw, host_class = self.shadow.infer(window)
            parity = int(
                np.array_equal(host_raw, e84_raw)
                and host_class == result["class"]
            )
            self.parity_failures += int(not parity)
            window_start = simulation_time_s - 49.0 / SENSOR_RATE_HZ
            medium_aligned = int(
                abs(window_start - 0.15) <= 1e-6
                and abs(simulation_time_s - 0.64) <= 1e-6
            )

        row: dict[str, object] = {
            "run_id": self.spec.run_id,
            "session_id": self.spec.session_id,
            "terrain_name": self.spec.terrain,
            "terrain_class": self.spec.terrain_id,
            "surface_family": self.family,
            "surface_index": self.spec.surface_index,
            "run_index": self.run_index,
            "sensor_variant": "clean_live",
            "simulation_time_s": f"{simulation_time_s:.9f}",
            "host_wall_time_s": f"{exchange.send_time - self.session_start:.9f}",
            "sequence": result["sequence"],
            **{
                name: f"{float(value):.9f}"
                for name, value in zip(PHYSICAL_COLUMNS, physical)
            },
            **{
                name: int(value)
                for name, value in zip(QUANTIZED_COLUMNS, quantized)
            },
            "fill": result["fill"],
            "warmup": result["warmup"],
            "inferred": result["inferred"],
            "medium_response_aligned": medium_aligned,
            "e84_class": result["class"] if result["inferred"] else "",
            **{
                name: (int(e84_raw[index]) if result["inferred"] else "")
                for index, name in enumerate(RESULT_COLUMNS)
            },
            "host_class": "" if host_class is None else host_class,
            **{
                name: ("" if host_raw is None else int(host_raw[index]))
                for index, name in enumerate(HOST_COLUMNS)
            },
            "host_e84_parity": "" if parity is None else parity,
            "cpu_cycles": result["cpu_cycles"],
            "npu_cycles": result["npu_cycles"],
            "rtt_ms": f"{exchange.rtt_ms:.6f}",
            "send_lateness_ms": f"{send_lateness_ms:.6f}",
            "deadline_miss": deadline_miss,
            "device_error_count": len(exchange.device_errors),
        }
        self.rows.append(row)

    def finish(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)


def stats(values: list[float]) -> dict[str, float]:
    return describe(values)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    inferred = [row for row in rows if int(row["inferred"]) == 1]
    aligned = [row for row in inferred if int(row["medium_response_aligned"]) == 1]
    send_times = [float(row["_send_time"]) for row in rows]
    send_periods = [
        (send_times[index] - send_times[index - 1]) * 1000.0
        for index in range(1, len(send_times))
        if rows[index]["session_id"] == rows[index - 1]["session_id"]
        and rows[index]["run_id"] == rows[index - 1]["run_id"]
    ]

    def accuracy(selected: list[dict[str, object]]) -> float | None:
        if not selected:
            return None
        return float(np.mean([
            int(row["e84_class"]) == int(row["terrain_class"])
            for row in selected
        ]))

    terrain_results: dict[str, object] = {}
    for terrain in CLASS_NAMES:
        terrain_inferred = [
            row for row in inferred if row["terrain_name"] == terrain
        ]
        terrain_aligned = [
            row for row in terrain_inferred
            if int(row["medium_response_aligned"]) == 1
        ]
        terrain_results[terrain] = {
            "samples": sum(row["terrain_name"] == terrain for row in rows),
            "inferences": len(terrain_inferred),
            "continuous_ground_truth_accuracy": accuracy(terrain_inferred),
            "medium_response_aligned_inferences": len(terrain_aligned),
            "medium_response_aligned_ground_truth_accuracy": accuracy(terrain_aligned),
            "predicted_class_counts": {
                name: sum(int(row["e84_class"]) == class_index for row in terrain_inferred)
                for class_index, name in enumerate(CLASS_NAMES)
            },
        }

    return {
        "samples": len(rows),
        "inferences": len(inferred),
        "medium_response_aligned_inferences": len(aligned),
        "send_period_ms": stats(send_periods),
        "rtt_ms": stats([float(row["rtt_ms"]) for row in rows]),
        "inferred_rtt_ms": stats([float(row["rtt_ms"]) for row in inferred]),
        "deadline_misses": sum(int(row["deadline_miss"]) for row in rows),
        "drops": 0,
        "timeouts": 0,
        "device_errors": sum(int(row["device_error_count"]) for row in rows),
        "host_e84_parity_failures": sum(
            int(row["host_e84_parity"]) != 1 for row in inferred
        ),
        "host_e84_exact_parity": (
            float(np.mean([int(row["host_e84_parity"]) for row in inferred]))
            if inferred else None
        ),
        "cpu_cycles": stats([float(row["cpu_cycles"]) for row in inferred]),
        "npu_cycles": stats([float(row["npu_cycles"]) for row in inferred]),
        "continuous_ground_truth_accuracy": accuracy(inferred),
        "medium_response_aligned_ground_truth_accuracy": accuracy(aligned),
        "terrain_results": terrain_results,
    }


def main() -> None:
    args = parse_args()
    if args.surface_index < 0 or args.run_index < 0 or args.runs <= 0:
        raise ValueError("surface/run indices must be non-negative and --runs positive")
    if args.duration * SENSOR_RATE_HZ != round(args.duration * SENSOR_RATE_HZ):
        raise ValueError("duration must contain an integer number of 100 Hz samples")
    if args.duration < 0.50:
        raise ValueError("duration must be at least 0.50 s for ring warm-up")
    if not 1 <= args.stride <= 65535:
        raise ValueError("stride must be in [1,65535]")
    if args.gui and (args.terrain == "all" or args.runs != 1):
        raise ValueError("--gui requires one named terrain and --runs 1")
    terrains = tuple(TERRAIN_LABELS) if args.terrain == "all" else (args.terrain,)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = (
        SIMULATION_DIR / "outputs" / f"terrain_live_hil_{timestamp}"
        if args.output_dir is None else args.output_dir.resolve()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    shadow = TerrainShadowModel()
    all_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    with TerrainStreamLink(args.port, args.timeout) as link:
        for terrain in terrains:
            for run_index in range(args.run_index, args.run_index + args.runs):
                surface = make_expanded_surface_parameters(
                    terrain, args.family, args.surface_index
                )
                spec = make_expanded_run_specification(
                    terrain, args.family, args.surface_index, run_index
                )
                condition = ExcitationCondition(
                    run_id=spec.run_id,
                    initial_velocity_x=spec.initial_velocity_x,
                    initial_velocity_y=spec.initial_velocity_y,
                    base_height_offset=spec.base_height_offset,
                    base_roll_deg=spec.base_roll_deg,
                    base_pitch_deg=spec.base_pitch_deg,
                )
                pulse = HorizontalPulse(
                    spec.pulse_start, spec.pulse_duration, spec.pulse_magnitude,
                    spec.pulse_direction_x, spec.pulse_direction_y,
                )
                csv_path = output / terrain / f"{spec.run_id}.csv"
                sink = LiveTerrainSink(
                    link, shadow, csv_path, spec, args.family, run_index, args.stride
                )
                sink.start_session()
                print(
                    f"live_start terrain={terrain} run={spec.run_id} "
                    f"session={spec.session_id} sequence=0"
                )
                metrics = run_window(
                    terrain, condition, args.duration, SENSOR_RATE_HZ,
                    spec.physics_seed, 0.70, pulse, None,
                    gui=args.gui, realtime_factor=1.0,
                    gui_hold_seconds=args.gui_hold_seconds,
                    session_id=spec.session_id,
                    scene_path=SURFACE_SCENE_PATH,
                    model_configurator=partial(
                        configure_expanded_run_surface,
                        surface=surface,
                        friction=spec.friction,
                    ),
                    physics_timestep=PHYSICS_TIMESTEP_S,
                    sample_callback=sink,
                )
                sink.finish()
                for row, send_time in zip(sink.rows, sink.send_times):
                    row["_send_time"] = send_time
                all_rows.extend(sink.rows)
                inferred = [row for row in sink.rows if int(row["inferred"]) == 1]
                run_record = {
                    "run_id": spec.run_id,
                    "session_id": spec.session_id,
                    "terrain": terrain,
                    "samples": len(sink.rows),
                    "inferences": len(inferred),
                    "parity_failures": sink.parity_failures,
                    "deadline_misses": sink.deadline_misses,
                    "device_errors": sink.device_errors,
                    "valid_run": int(metrics["valid_run"]),
                    "csv": str(csv_path.relative_to(output)),
                }
                run_records.append(run_record)
                print(
                    f"live_done terrain={terrain} samples={len(sink.rows)} "
                    f"inferences={len(inferred)} parity_failures={sink.parity_failures} "
                    f"deadline_misses={sink.deadline_misses} csv={csv_path}"
                )

    summary = summarize(all_rows)
    summary.update(
        {
            "probe_port": str(args.port),
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "sensor_rate_hz": SENSOR_RATE_HZ,
            "window_samples": 50,
            "stride": args.stride,
            "sensor_variant": "clean_live",
            "model": shadow.identity,
            "runs": run_records,
            "invalid_runs": sum(not record["valid_run"] for record in run_records),
            "interpretation": (
                "medium_response_aligned selects the sliding window at 0.15-0.64 s; "
                "all other inferred windows are exploratory continuous trajectories. "
                "Live input is the clean MuJoCo virtual sensor stream; Dataset-v1 "
                "sensor imperfections remain offline augmentation."
            ),
        }
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={summary_path}")
    if any(
        summary[name]
        for name in (
            "host_e84_parity_failures",
            "deadline_misses",
            "drops",
            "timeouts",
            "device_errors",
            "invalid_runs",
        )
    ):
        raise RuntimeError("live HIL correctness/timing gate failed; see summary")


if __name__ == "__main__":
    main()
