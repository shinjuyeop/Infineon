#!/usr/bin/env python3
"""Train and save the lightweight Scenario 1 NumPy 1D CNN."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepet_hil.model.dataset import prepare_dataset
from deepet_hil.model.network import TerrainCNN
from deepet_hil.model.train import TrainingConfig, accuracy, train_model
from deepet_hil.terrain.params import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--terrain-config", type=Path, default=PROJECT_ROOT / "configs/terrain_v0_1.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/terrain_cnn_v0_1.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = prepare_dataset(args.dataset, seed=args.seed)
    terrain_config = load_config(args.terrain_config)
    terrains = sorted(terrain_config.terrains.values(), key=lambda value: value.class_id)
    class_names = tuple(terrain.name for terrain in terrains)
    model = TerrainCNN(
        input_steps=dataset.x.shape[1],
        input_channels=dataset.x.shape[2],
        num_classes=len(class_names),
        seed=args.seed,
    )
    training_config = TrainingConfig(args.epochs, args.batch_size, args.learning_rate, args.seed)
    history = train_model(
        model,
        dataset.x,
        dataset.y,
        dataset.train_indices,
        dataset.validation_indices,
        training_config,
    )
    model.save(
        args.output,
        channel_mean=dataset.standardizer.mean,
        channel_scale=dataset.standardizer.scale,
        class_names=class_names,
    )
    summary = {
        "model": str(args.output),
        "architecture": "Conv1D(12,k=5)-ReLU-Conv1D(16,k=3)-ReLU-GAP-Dense(4)",
        "training_config": asdict(training_config),
        "split_sizes": {
            "train": len(dataset.train_indices),
            "validation": len(dataset.validation_indices),
            "test": len(dataset.test_indices),
        },
        "final_train_accuracy": accuracy(model, dataset.x[dataset.train_indices], dataset.y[dataset.train_indices]),
        "final_validation_accuracy": accuracy(model, dataset.x[dataset.validation_indices], dataset.y[dataset.validation_indices]),
        "history": [asdict(row) for row in history],
    }
    history_path = args.output.with_name(f"{args.output.stem}_training.json")
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    main()

