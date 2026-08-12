"""Rebuild Fast Reflex v1 derived artifacts from preserved raw traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_terrain_fast_reflex_v1 import (
    _onset_fields,
    _plot_trace,
    _save_artifacts,
    separation_summary,
)
from run_surface_sampling_rate_study import write_dict_rows
from terrain_fast_reflex_v1 import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    FastReflexTrace,
    validate_split_integrity,
    validate_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_traces(source: Path) -> tuple[list[FastReflexTrace], list[dict[str, str]]]:
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(source / "inputs_fusion10.npz", allow_pickle=False) as inputs:
        sensors = inputs["sensors"]
        timestamps = inputs["sample_time_s"]
        input_ids = inputs["run_id"].astype(str)
    with np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as oracle_file:
        oracle = oracle_file["oracle"]
        slip = oracle_file["slip"]
        sink = oracle_file["sink"]
        tilt = oracle_file["tilt"]
        oracle_ids = oracle_file["run_id"].astype(str)
    manifest_ids = np.asarray([row["run_id"] for row in rows])
    if not (
        len(rows) == len(sensors) == len(oracle)
        and np.array_equal(manifest_ids, input_ids)
        and np.array_equal(manifest_ids, oracle_ids)
    ):
        raise ValueError("manifest/input/oracle run ordering does not match")
    traces = []
    for index, row in enumerate(rows):
        trace = FastReflexTrace(
            metadata=dict(row),
            timestamps_s=np.asarray(timestamps[index], dtype=np.float64),
            sensors=np.asarray(sensors[index], dtype=np.float64),
            oracle=np.asarray(oracle[index], dtype=np.float64),
            slip=np.asarray(slip[index], dtype=bool),
            sink=np.asarray(sink[index], dtype=bool),
            tilt=np.asarray(tilt[index], dtype=bool),
            valid=bool(int(row["valid"])),
            invalid_reason=row["invalid_reason"],
        )
        validate_trace(trace)
        traces.append(trace)
    return traces, rows


def main() -> None:
    args = parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    traces, source_rows = load_traces(source)
    rows = []
    for trace, source_row in zip(traces, source_rows):
        rows.append({**source_row, **_onset_fields(trace)})
    validate_split_integrity(rows)
    _save_artifacts(output, traces, rows)
    separation_rows = separation_summary(traces)
    if separation_rows:
        write_dict_rows(output / "window_separation.csv", separation_rows)

    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    protocol["derived_artifact_revision"] = 2
    protocol["derived_from"] = str(source)
    protocol["hazard_metadata_semantics"] = (
        "hazard_type/onset follow physical oracle masks for every scenario; "
        "expected_hazard remains scenario intent only"
    )
    protocol["measured"]["physical_hazard_runs"] = sum(
        int(row["physical_hazard"]) for row in rows
    )
    protocol["measured"]["physical_normal_runs"] = sum(
        not int(row["physical_hazard"]) for row in rows
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot:
        for scenario in ("marble_to_ice", "marble_to_sand"):
            trace = next(
                (item for item in traces if item.metadata["scenario"] == scenario),
                None,
            )
            if trace is not None:
                _plot_trace(output, trace)

    summary = [
        f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION} derived_revision=2",
        f"runs={len(traces)} valid={sum(trace.valid for trace in traces)}",
        "native_sampling=1000Hz physics=2000Hz spacing=1ms",
        f"derived_from={source}",
    ]
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        selected = [row for row in rows if row["scenario"] == scenario]
        physical = [row for row in selected if int(row["physical_hazard"])]
        target = [
            row for row in selected if row["transition_to_target_hazard_ms"] != ""
        ]
        summary.append(
            f"{scenario}: runs={len(selected)} physical_hazard={len(physical)} "
            f"target_onsets={len(target)}"
        )
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
