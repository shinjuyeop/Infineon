#!/usr/bin/env python3
"""Run direct host CNN inference for one raw dataset sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepet_hil.model.dataset import load_npz_dataset
from deepet_hil.model.inference import TerrainInferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    args = parser.parse_args()
    x, y = load_npz_dataset(args.dataset)
    if not 0 <= args.sample_index < len(x):
        raise IndexError(f"sample-index must be in [0, {len(x) - 1}]")
    engine = TerrainInferenceEngine.load(args.model)
    prediction = engine.predict(x[args.sample_index])
    expected = engine.class_names[int(y[args.sample_index])]
    print(f"Sample ID: {args.sample_index}")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction.class_name}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print(f"Result: {'PASS' if prediction.class_id == int(y[args.sample_index]) else 'FAIL'}")


if __name__ == "__main__":
    main()

