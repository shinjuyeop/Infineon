#!/usr/bin/env python3
"""Run one physical-unit dataset window through the host INT8 TFLite path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepet_hil.deployment.inference import TFLiteInferenceEngine
from deepet_hil.deployment.metadata import DeploymentMetadata
from deepet_hil.model.dataset import load_npz_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    args = parser.parse_args()

    x, y = load_npz_dataset(args.dataset)
    if args.sample_index < 0 or args.sample_index >= len(y):
        raise IndexError(f"sample-index must be between 0 and {len(y) - 1}")
    metadata = DeploymentMetadata.load(args.metadata)
    engine = TFLiteInferenceEngine(args.model, metadata)
    prediction = engine.predict(x[args.sample_index])
    ground_truth = metadata.class_names[int(y[args.sample_index])]
    output = {
        "sample_index": args.sample_index,
        "ground_truth": ground_truth,
        "prediction": prediction.class_name,
        "confidence": prediction.confidence,
        "probabilities": {
            name: float(value)
            for name, value in zip(metadata.class_names, prediction.probabilities)
        },
        "result": "PASS" if prediction.class_name == ground_truth else "FAIL",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
