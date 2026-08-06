#!/usr/bin/env python3
"""Evaluate the float deployment model on the deterministic held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.deployment.evaluate import calculate_metrics, save_confusion_matrix_plot
from infineon_hil.deployment.metadata import DeploymentMetadata
from infineon_hil.deployment.model import require_tensorflow
from infineon_hil.model.dataset import load_npz_dataset
from infineon_hil.terrain.dataset import stratified_split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--plot",
        type=Path,
        default=PROJECT_ROOT / "outputs/plots/deployment/float_confusion_matrix.png",
    )
    args = parser.parse_args()

    metadata = DeploymentMetadata.load(args.metadata)
    x, y = load_npz_dataset(args.dataset)
    test_indices = stratified_split_indices(y, seed=metadata.split_seed)["test"]
    tf = require_tensorflow()
    model = tf.keras.models.load_model(args.model)
    probabilities = np.asarray(
        model.predict(metadata.normalize(x[test_indices]), verbose=0), dtype=np.float32
    )
    metrics = calculate_metrics(probabilities, y[test_indices])
    save_confusion_matrix_plot(
        metrics.confusion_matrix,
        metadata.class_names,
        args.plot,
        title="Float deployment CNN (synthetic test set)",
    )
    per_class = {
        name: {
            "precision": float(metrics.precision[index]),
            "recall": float(metrics.recall[index]),
            "f1": float(metrics.f1[index]),
        }
        for index, name in enumerate(metadata.class_names)
    }
    output = {
        "evaluation": "held-out synthetic test set; not real-terrain generalization",
        "test_samples": len(test_indices),
        "accuracy": metrics.accuracy,
        "per_class": per_class,
        "ice_recall": float(metrics.recall[2]),
        "confusion_matrix": metrics.confusion_matrix.tolist(),
        "kpi_accuracy_at_least_0_85": "PASS" if metrics.accuracy >= 0.85 else "FAIL",
        "kpi_ice_recall_at_least_0_90": "PASS" if metrics.recall[2] >= 0.90 else "FAIL",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
