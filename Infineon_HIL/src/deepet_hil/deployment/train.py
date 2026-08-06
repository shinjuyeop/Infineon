"""Training helpers for the deployment-target TensorFlow CNN."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import DeploymentDataset
from .model import require_tensorflow


@dataclass(frozen=True)
class DeploymentTrainingConfig:
    """Reproducible Keras training configuration."""

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    early_stopping_patience: int | None = None


def configure_determinism(seed: int) -> None:
    """Set reproducible random state within TensorFlow's supported limits."""

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def train_deployment_model(
    model: Any,
    dataset: DeploymentDataset,
    config: DeploymentTrainingConfig,
    *,
    verbose: int = 1,
) -> Any:
    """Compile and train using only the designated train/validation partitions."""

    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0.0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    tf = require_tensorflow()
    configure_determinism(config.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks: list[Any] = []
    if config.early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    return model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_validation, dataset.y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )


def save_training_history_plot(history: Any, path: str | Path) -> None:
    """Save compact loss and accuracy curves."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Deployment CNN loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set(title="Deployment CNN accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
