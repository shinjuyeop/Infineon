#!/usr/bin/env python3
"""Export the deployment Keras model with strict full-integer TFLite quantization."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepet_hil.deployment.metadata import DeploymentMetadata
from deepet_hil.deployment.model import require_tensorflow
from deepet_hil.deployment.quantization import export_full_int8_tflite
from deepet_hil.model.dataset import load_npz_dataset
from deepet_hil.terrain.dataset import stratified_split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/deployment/terrain_cnn_v0_1_int8.tflite",
    )
    parser.add_argument("--representative-samples", type=int, default=200)
    args = parser.parse_args()

    metadata = DeploymentMetadata.load(args.metadata)
    x, y = load_npz_dataset(args.dataset)
    train_indices = stratified_split_indices(y, seed=metadata.split_seed)["train"]
    count = min(args.representative_samples, len(train_indices))
    if count <= 0:
        raise ValueError("Representative sample count must be positive")
    representative = metadata.normalize(x[train_indices[:count]])
    tf = require_tensorflow()
    model = tf.keras.models.load_model(args.model)
    input_spec, output_spec = export_full_int8_tflite(model, representative, args.output)
    metadata = metadata.with_quantization(input_spec, output_spec)
    metadata.save(args.metadata)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "representative_partition": "train",
                "representative_samples": count,
                "input": asdict(input_spec),
                "output": asdict(output_spec),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
