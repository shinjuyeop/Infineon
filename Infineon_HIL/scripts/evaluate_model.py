#!/usr/bin/env python3
"""Evaluate the trained CNN on the deterministic held-out test partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.model.dataset import load_npz_dataset
from infineon_hil.model.evaluation import evaluate_model
from infineon_hil.model.inference import TerrainInferenceEngine
from infineon_hil.terrain.dataset import stratified_split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    x, y = load_npz_dataset(args.dataset)
    test = stratified_split_indices(y, seed=args.seed)["test"]
    engine = TerrainInferenceEngine.load(args.model)
    result = evaluate_model(engine, x[test], y[test])
    output = {
        "evaluation": "held-out CNN test set (not nearest-centroid baseline)",
        "test_samples": len(test),
        "accuracy": result.accuracy,
        "per_class_recall": {
            name: float(value) for name, value in zip(engine.class_names, result.per_class_recall)
        },
        "confusion_matrix": result.confusion_matrix.tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

