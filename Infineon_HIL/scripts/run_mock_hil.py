#!/usr/bin/env python3
"""Run one or more samples through Host SignalPlayer -> Mock HIL -> Mock E84."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.hil.simulation import run_mock_sample
from infineon_hil.model.dataset import load_npz_dataset
from infineon_hil.model.inference import TerrainInferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="Consecutive samples to replay")
    parser.add_argument("--playback-rate-hz", type=float, default=1000.0)
    args = parser.parse_args()
    x, y = load_npz_dataset(args.dataset)
    if args.count <= 0 or args.sample_index < 0 or args.sample_index + args.count > len(x):
        raise IndexError("Requested sample range is outside the dataset")
    engine = TerrainInferenceEngine.load(args.model)
    results = [
        run_mock_sample(
            x[index], int(y[index]), index, engine, playback_rate_hz=args.playback_rate_hz
        )
        for index in range(args.sample_index, args.sample_index + args.count)
    ]
    if len(results) == 1:
        result = results[0]
        print(f"Sample ID: {result.sample_id}")
        print(f"Frames played: {result.frames_played}")
        print(f"Expected: {engine.class_names[result.expected_class_id]}")
        print(f"Predicted: {result.prediction.class_name}")
        print(f"Confidence: {result.prediction.confidence:.4f}")
        print(f"Result: {'PASS' if result.passed else 'FAIL'}")
        return
    expected = np.asarray([result.expected_class_id for result in results])
    predicted = np.asarray([result.prediction.class_id for result in results])
    labels = np.arange(len(engine.class_names))
    summary = {
        "evaluation": "mock HIL replay (not physical HIL)",
        "start_sample_id": args.sample_index,
        "sample_count": len(results),
        "frames_per_sample": results[0].frames_played,
        "accuracy": float(np.mean(expected == predicted)),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

