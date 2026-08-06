"""Metrics and plots shared by float and INT8 deployment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassificationMetrics:
    """Complete fixed-label metrics for the four terrain classes."""

    accuracy: float
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    f1: NDArray[np.float64]
    confusion_matrix: NDArray[np.int64]
    predictions: NDArray[np.int64]


def calculate_metrics(
    probabilities: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    num_classes: int = 4,
) -> ClassificationMetrics:
    """Calculate metrics without dropping classes absent from a prediction."""

    scores = np.asarray(probabilities)
    truth = np.asarray(labels, dtype=np.int64)
    if scores.shape != (len(truth), num_classes):
        raise ValueError(f"Expected probabilities {(len(truth), num_classes)}, got {scores.shape}")
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    class_ids = np.arange(num_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        labels=class_ids,
        average=None,
        zero_division=0,
    )
    return ClassificationMetrics(
        accuracy=float(np.mean(predictions == truth)),
        precision=np.asarray(precision, dtype=np.float64),
        recall=np.asarray(recall, dtype=np.float64),
        f1=np.asarray(f1, dtype=np.float64),
        confusion_matrix=np.asarray(
            confusion_matrix(truth, predictions, labels=class_ids), dtype=np.int64
        ),
        predictions=predictions,
    )


def save_confusion_matrix_plot(
    matrix: NDArray[np.integer],
    class_names: tuple[str, ...],
    path: str | Path,
    *,
    title: str,
) -> None:
    """Render one small confusion-matrix artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=title,
        xlabel="Predicted",
        ylabel="Ground truth",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
