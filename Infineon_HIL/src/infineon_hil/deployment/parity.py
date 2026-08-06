"""Float Keras versus INT8 TFLite held-out comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluate import ClassificationMetrics, calculate_metrics
from .inference import TFLiteInferenceEngine


@dataclass(frozen=True)
class ParityReport:
    """Accuracy and prediction agreement across the two deployment formats."""

    float_metrics: ClassificationMetrics
    int8_metrics: ClassificationMetrics
    accuracy_delta: float
    prediction_agreement: float

    @property
    def has_warning(self) -> bool:
        return self.accuracy_delta < -0.03 or self.int8_metrics.recall[2] < 0.90


def compare_float_int8(
    float_model: Any,
    int8_engine: TFLiteInferenceEngine,
    raw_samples: NDArray[np.floating],
    labels: NDArray[np.integer],
) -> ParityReport:
    """Evaluate both formats on identical physical-unit test windows."""

    standardized = int8_engine.metadata.normalize(raw_samples)
    float_probabilities = np.asarray(float_model.predict(standardized, verbose=0), dtype=np.float32)
    int8_probabilities = int8_engine.predict_standardized_batch(standardized)
    float_metrics = calculate_metrics(float_probabilities, labels)
    int8_metrics = calculate_metrics(int8_probabilities, labels)
    return ParityReport(
        float_metrics=float_metrics,
        int8_metrics=int8_metrics,
        accuracy_delta=int8_metrics.accuracy - float_metrics.accuracy,
        prediction_agreement=float(
            np.mean(float_metrics.predictions == int8_metrics.predictions)
        ),
    )
