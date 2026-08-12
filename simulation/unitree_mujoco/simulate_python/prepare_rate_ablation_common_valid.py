"""Create overwrite-safe, run-aligned quick datasets for sampling-rate comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_terrain_1d_cnn import validate_dataset_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", action="append", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.dataset) < 2 or len(args.dataset) != len(args.output):
        raise ValueError("provide matching --dataset/--output arguments for at least two rates")
    datasets = [path.resolve() for path in args.dataset]
    outputs = [path.resolve() for path in args.output]
    candidate_text = [(path / "candidate_manifest.csv").read_bytes() for path in datasets]
    if any(content != candidate_text[0] for content in candidate_text[1:]):
        raise ValueError("rate datasets do not have identical candidate manifests")

    manifests = [read_rows(path / "split_manifest.csv") for path in datasets]
    common_run_ids = set.intersection(
        *({row["run_id"] for row in manifest} for manifest in manifests)
    )
    if not common_run_ids:
        raise ValueError("rate datasets have no common valid runs")

    reference_order = [
        row["run_id"] for row in read_rows(datasets[0] / "candidate_manifest.csv")
        if row["run_id"] in common_run_ids
    ]
    summary: dict[str, object] = {
        "policy": "Intersection of valid run_id values; candidate manifests must match byte-for-byte.",
        "common_valid_samples": len(reference_order),
        "source_datasets": [str(path) for path in datasets],
    }
    for source, output, manifest in zip(datasets, outputs, manifests):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite {output}")
        output.mkdir(parents=True, exist_ok=True)
        index_by_run = {row["run_id"]: int(row["sample_index"]) for row in manifest}
        indices = np.asarray([index_by_run[run_id] for run_id in reference_order])
        aligned_rows: list[dict[str, object]] = []
        for new_index, run_id in enumerate(reference_order):
            original = manifest[index_by_run[run_id]]
            aligned_rows.append({**original, "sample_index": new_index})
        for variant in ("clean", "noisy"):
            with np.load(source / f"dataset_{variant}.npz") as payload:
                arrays = {
                    name: np.asarray(payload[name])[indices]
                    for name in ("X", "y", "split", "surface_family")
                }
            validate_dataset_arrays(
                arrays["X"], arrays["y"], arrays["split"], arrays["surface_family"]
            )
            np.savez_compressed(output / f"dataset_{variant}.npz", **arrays)
        write_rows(output / "split_manifest.csv", aligned_rows)
        source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
        (output / "protocol.json").write_text(
            json.dumps(
                {
                    **source_protocol,
                    "comparison_filter": summary["policy"],
                    "source_valid_samples": len(manifest),
                    "common_valid_samples": len(reference_order),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{source.name}: {len(manifest)} -> {len(reference_order)} common valid samples")


if __name__ == "__main__":
    main()
