#!/usr/bin/env python3
"""Compare float Keras and INT8 TFLite predictions on the same held-out samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.deployment.evaluate import save_confusion_matrix_plot
from infineon_hil.deployment.inference import TFLiteInferenceEngine
from infineon_hil.deployment.metadata import DeploymentMetadata
from infineon_hil.deployment.model import require_tensorflow
from infineon_hil.deployment.parity import compare_float_int8
from infineon_hil.model.dataset import load_npz_dataset
from infineon_hil.terrain.dataset import stratified_split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--float-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/plots/deployment"
    )
    args = parser.parse_args()

    metadata = DeploymentMetadata.load(args.metadata)
    x, y = load_npz_dataset(args.dataset)
    test_indices = stratified_split_indices(y, seed=metadata.split_seed)["test"]
    tf = require_tensorflow()
    float_model = tf.keras.models.load_model(args.float_model)
    int8_engine = TFLiteInferenceEngine(args.int8_model, metadata)
    report = compare_float_int8(float_model, int8_engine, x[test_indices], y[test_indices])
    save_confusion_matrix_plot(
        report.float_metrics.confusion_matrix,
        metadata.class_names,
        args.output_dir / "float_confusion_matrix.png",
        title="Float deployment CNN (synthetic test set)",
    )
    save_confusion_matrix_plot(
        report.int8_metrics.confusion_matrix,
        metadata.class_names,
        args.output_dir / "int8_confusion_matrix.png",
        title="INT8 TFLite CNN (synthetic test set)",
    )
    output = {
        "evaluation": "float/INT8 parity on held-out synthetic data",
        "test_samples": len(test_indices),
        "float_accuracy": report.float_metrics.accuracy,
        "int8_accuracy": report.int8_metrics.accuracy,
        "accuracy_delta": report.accuracy_delta,
        "float_ice_recall": float(report.float_metrics.recall[2]),
        "int8_ice_recall": float(report.int8_metrics.recall[2]),
        "prediction_agreement": report.prediction_agreement,
        "float_confusion_matrix": report.float_metrics.confusion_matrix.tolist(),
        "int8_confusion_matrix": report.int8_metrics.confusion_matrix.tolist(),
        "warning": (
            "INT8 accuracy drop exceeds 0.03 or INT8 ice recall is below 0.90"
            if report.has_warning
            else None
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
