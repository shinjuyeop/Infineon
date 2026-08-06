#!/usr/bin/env python3
"""Validate synthetic distributions with a four-feature nearest-centroid baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.terrain.params import load_config
from infineon_hil.terrain.validation import run_sanity_baseline, validate_sensor_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/terrain_v0_1.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    with np.load(args.dataset) as data:
        x, y = data["X"], data["y"]
    metadata = pd.read_csv(args.metadata)
    if len(metadata) != len(y) or not np.array_equal(metadata["class_id"].to_numpy(), y):
        raise ValueError("Metadata rows/class IDs do not align with dataset labels")
    validate_sensor_tensor(x, y)
    baseline = run_sanity_baseline(
        x, y, sampling_rate_hz=config.sampling_rate_hz, seed=args.seed
    )
    labels = baseline.labels.tolist()
    names = {terrain.class_id: terrain.name for terrain in config.terrains.values()}
    result = {
        "accuracy": baseline.accuracy,
        "per_class_recall": {names[label]: float(value) for label, value in zip(labels, baseline.recalls)},
        "ice_recall": float(baseline.recalls[labels.index(config.terrains["ice"].class_id)]),
        "confusion_matrix": baseline.confusion_matrix.tolist(),
        "test_samples": baseline.test_samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(baseline.confusion_matrix, display_labels=[names[label] for label in labels]).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Nearest-centroid sanity baseline\n(4 canonical synthetic features)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print("NOTE: This baseline is not expected to reproduce the report's undefined 12-feature result.")


if __name__ == "__main__":
    main()
