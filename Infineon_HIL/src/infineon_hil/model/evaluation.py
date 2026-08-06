"""CNN evaluation metrics kept distinct from the feature sanity baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, recall_score

from .inference import TerrainInferenceEngine


@dataclass(frozen=True)
class ModelEvaluation:
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    per_class_recall: NDArray[np.float64]
    predictions: NDArray[np.int64]


def evaluate_model(
    engine: TerrainInferenceEngine,
    x: NDArray[np.floating],
    y: NDArray[np.integer],
) -> ModelEvaluation:
    probabilities = engine.predict_batch(x)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    labels = np.arange(len(engine.class_names))
    return ModelEvaluation(
        accuracy=float(np.mean(predictions == y)),
        confusion_matrix=np.asarray(confusion_matrix(y, predictions, labels=labels), dtype=np.int64),
        per_class_recall=np.asarray(
            recall_score(y, predictions, labels=labels, average=None, zero_division=0),
            dtype=np.float64,
        ),
        predictions=predictions,
    )

