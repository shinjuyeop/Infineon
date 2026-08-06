#!/usr/bin/env python3
"""Train and save the deployment-target TensorFlow terrain CNN."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infineon_hil.deployment.dataset import prepare_deployment_dataset
from infineon_hil.deployment.evaluate import calculate_metrics
from infineon_hil.deployment.metadata import DeploymentMetadata
from infineon_hil.deployment.model import build_deployment_model
from infineon_hil.deployment.train import (
    DeploymentTrainingConfig,
    save_training_history_plot,
    train_deployment_model,
)
from infineon_hil.schema import CHANNEL_NAMES
from infineon_hil.terrain.params import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--terrain-config", type=Path, default=PROJECT_ROOT / "configs/terrain_v0_1.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/deployment/terrain_cnn_v0_1.keras",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "models/deployment/terrain_cnn_v0_1_metadata.json",
    )
    parser.add_argument(
        "--history-plot",
        type=Path,
        default=PROJECT_ROOT / "outputs/plots/deployment/training_history.png",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int)
    args = parser.parse_args()

    dataset = prepare_deployment_dataset(args.dataset, seed=args.seed)
    terrain_config = load_config(args.terrain_config)
    terrains = sorted(terrain_config.terrains.values(), key=lambda value: value.class_id)
    class_names = tuple(terrain.name for terrain in terrains)
    model = build_deployment_model(
        input_shape=dataset.prepared.x.shape[1:], num_classes=len(class_names), seed=args.seed
    )
    training_config = DeploymentTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )
    history = train_deployment_model(model, dataset, training_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    save_training_history_plot(history, args.history_plot)

    metadata = DeploymentMetadata(
        model_version="terrain_cnn_v0_1",
        channel_order=CHANNEL_NAMES,
        channel_mean=tuple(float(value) for value in dataset.prepared.standardizer.mean),
        channel_std=tuple(float(value) for value in dataset.prepared.standardizer.scale),
        class_names=class_names,
        input_shape=tuple(int(value) for value in dataset.prepared.x.shape[1:]),
        sampling_rate_hz=terrain_config.sampling_rate_hz,
        window_length_ms=int(terrain_config.raw["sampling"]["window_length_ms"]),
        split_seed=args.seed,
    )
    metadata.save(args.metadata)

    train_metrics = calculate_metrics(
        model.predict(dataset.x_train, verbose=0), dataset.y_train
    )
    validation_metrics = calculate_metrics(
        model.predict(dataset.x_validation, verbose=0), dataset.y_validation
    )
    summary = {
        "model": str(args.output),
        "metadata": str(args.metadata),
        "architecture": "Conv1D(16,k=5)-ReLU-MaxPool(2)-Conv1D(32,k=3)-ReLU-GAP-Dense(16)-ReLU-Dense(4)-Softmax",
        "parameters": int(model.count_params()),
        "training_config": asdict(training_config),
        "split_sizes": {
            "train": len(dataset.y_train),
            "validation": len(dataset.y_validation),
            "test": len(dataset.y_test),
        },
        "train_accuracy": train_metrics.accuracy,
        "validation_accuracy": validation_metrics.accuracy,
        "epochs_completed": len(history.history["loss"]),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    history_path = args.output.with_name(f"{args.output.stem}_training.json")
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    print(f"Saved training history: {history_path}")
    print("WARNING: Results use synthetic engineering-estimate data, not measured terrain data.")


if __name__ == "__main__":
    main()
