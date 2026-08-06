"""Mini-batch Adam training for the small NumPy terrain CNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .network import TerrainCNN


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.003
    seed: int = 42


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    train_accuracy: float
    validation_accuracy: float


def accuracy(model: TerrainCNN, x: NDArray[np.floating], y: NDArray[np.integer]) -> float:
    prediction = np.argmax(model.predict_proba(x), axis=1)
    return float(np.mean(prediction == y))


def train_model(
    model: TerrainCNN,
    x: NDArray[np.floating],
    y: NDArray[np.integer],
    train_indices: NDArray[np.integer],
    validation_indices: NDArray[np.integer],
    config: TrainingConfig = TrainingConfig(),
    *,
    verbose: bool = True,
) -> list[EpochMetrics]:
    """Train in place with Adam and return deterministic epoch metrics."""

    rng = np.random.default_rng(config.seed)
    first_moment = {name: np.zeros_like(value) for name, value in model.params.items()}
    second_moment = {name: np.zeros_like(value) for name, value in model.params.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    update = 0
    history: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        shuffled = rng.permutation(train_indices)
        losses: list[float] = []
        for start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[start : start + config.batch_size]
            loss, gradients = model.loss_and_gradients(x[batch], y[batch])
            losses.append(loss)
            update += 1
            for name, gradient in gradients.items():
                first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
                second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * np.square(gradient)
                corrected_first = first_moment[name] / (1.0 - beta1**update)
                corrected_second = second_moment[name] / (1.0 - beta2**update)
                model.params[name] -= config.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + epsilon
                )
        metrics = EpochMetrics(
            epoch=epoch,
            loss=float(np.mean(losses)),
            train_accuracy=accuracy(model, x[train_indices], y[train_indices]),
            validation_accuracy=accuracy(model, x[validation_indices], y[validation_indices]),
        )
        history.append(metrics)
        if verbose:
            print(
                f"epoch={epoch:02d} loss={metrics.loss:.4f} "
                f"train_acc={metrics.train_accuracy:.4f} val_acc={metrics.validation_accuracy:.4f}"
            )
    return history

