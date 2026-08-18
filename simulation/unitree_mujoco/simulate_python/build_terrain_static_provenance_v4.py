"""Repackage the authoritative 1-kHz static corpus with run-level provenance.

The source simulations are not regenerated: this is a lossless provenance
corpus built from their raw manifest and noisy Fusion10 windows.  It reserves
whole surface realizations, never individual windows, for architecture choice.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from run_terrain_transition_aware_v2 import SIM, STATIC


SOURCE = STATIC.parent
OUT = SIM / "outputs/terrain_static_provenance_v4"
SURFACE_PATTERN = re.compile(r"_s(?P<surface>\d+)_r(?P<run>\d+)$")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def split_for(row: dict[str, str]) -> str:
    # The existing test families remain untouched.  The unused selection
    # reservation consists of complete s06/s07 train-family realizations.
    if row["split"] == "test": return "test"
    if row["split"] != "train": return "excluded_legacy_validation"
    match = SURFACE_PATTERN.search(row["run_id"])
    if not match: raise ValueError(f"cannot parse run id {row['run_id']}")
    return "architecture_selection" if int(match.group("surface")) >= 6 else "train"


def audit(rows: list[dict[str, object]]) -> dict[str, object]:
    owners: dict[str, str] = {}; surface_owners: dict[tuple[str, str, str], str] = {}
    for row in rows:
        split = str(row["split_v4"]); run = str(row["source_run_id"]); previous = owners.setdefault(run, split)
        if previous != split: raise ValueError(f"run leakage: {run}")
        key = (str(row["terrain_class"]), str(row["surface_family"]), str(row["surface_realization"])); previous = surface_owners.setdefault(key, split)
        if previous != split: raise ValueError(f"surface realization leakage: {key}")
    result = {"run_id_leakage": False, "surface_realization_leakage": False, "splits": {}}
    for split in ("train", "architecture_selection", "test"):
        selected = [row for row in rows if row["split_v4"] == split]
        classes = {row["terrain_class"] for row in selected}
        if len(classes) != 4: raise ValueError(f"missing class in {split}: {classes}")
        result["splits"][split] = {"runs": len(selected), "classes": {name: sum(row["terrain_class"] == name for row in selected) for name in sorted(classes)}, "families": sorted({row["surface_family"] for row in selected}), "surface_realizations": len({(row["terrain_class"], row["surface_family"], row["surface_realization"]) for row in selected})}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    with np.load(SOURCE / "dataset_noisy.npz") as values: x, y = values["X"], values["y"]
    with (SOURCE / "split_manifest.csv").open() as stream: source_rows = list(csv.DictReader(stream))
    if len(source_rows) != len(x): raise ValueError("manifest/tensor cardinality mismatch")
    rows = []
    for index, row in enumerate(source_rows):
        match = SURFACE_PATTERN.search(row["run_id"])
        if not match: raise ValueError(row["run_id"])
        rows.append({"sample_index": index, "terrain_class": row["terrain_class"], "terrain_id": int(row["terrain_id"]), "source_run_id": row["run_id"], "surface_family": row["surface_family"], "surface_realization": f"{row['surface_family']}:s{match.group('surface')}", "surface_seed": row["surface_seed"], "physics_seed": row.get("physics_seed", ""), "sensor_noise_seed": row["sensor_noise_seed"], "sensor_rate_hz": 1000, "simulation_config": "expanded_dataset_v1_1khz_fusion10", "split_v4": split_for(row)})
    report = audit(rows); out.mkdir(parents=True)
    split = np.asarray([row["split_v4"] for row in rows]); family = np.asarray([row["surface_family"] for row in rows]); run_id = np.asarray([row["source_run_id"] for row in rows]); realization = np.asarray([row["surface_realization"] for row in rows])
    np.savez_compressed(out / "dataset_noisy_provenance.npz", X=x, y=y, split=split, surface_family=family, source_run_id=run_id, surface_realization=realization)
    write_rows(out / "static_split_manifest.csv", rows)
    protocol = {"dataset": "terrain_static_provenance_v4", "source": str(SOURCE), "source_schema": "terrain_dataset_v1_expanded_1000hz_full", "fusion10": ["FSR1", "FSR2", "FSR3", "FSR4", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"], "sensor_rate_hz": 1000, "window_ms": 50, "split_policy": "whole surface realization s06/s07 of original train families reserved for architecture selection; original test families remain test; previously used source validation families are excluded", "historical_97_098_percent_baseline_preserved": True}
    (out / "static_dataset_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n"); (out / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol": protocol, "audit": report, "samples": int(len(x))}, indent=2))


if __name__ == "__main__": main()
