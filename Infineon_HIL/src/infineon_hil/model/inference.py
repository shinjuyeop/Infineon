"""Shared host/MockE84 preprocessing and CNN inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .network import TerrainCNN


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: NDArray[np.float32]


class TerrainInferenceEngine:
    """Own the model and its exact saved channel standardization."""

    def __init__(
        self,
        model: TerrainCNN,
        channel_mean: NDArray[np.floating],
        channel_scale: NDArray[np.floating],
        class_names: tuple[str, ...],
    ) -> None:
        self.model = model
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
        self.channel_scale = np.asarray(channel_scale, dtype=np.float32)
        self.class_names = class_names
        if self.channel_mean.shape != (model.input_channels,) or self.channel_scale.shape != (model.input_channels,):
            raise ValueError("Preprocessing shape does not match model channels")
        if len(class_names) != model.num_classes:
            raise ValueError("Class names do not match model output")

    @classmethod
    def load(cls, path: str | Path) -> "TerrainInferenceEngine":
        model, mean, scale, names = TerrainCNN.load(path)
        return cls(model, mean, scale, names)

    def predict_batch(self, samples: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(samples, dtype=np.float32)
        standardized = (values - self.channel_mean) / self.channel_scale
        return self.model.predict_proba(standardized)

    def predict(self, sample: NDArray[np.floating]) -> Prediction:
        values = np.asarray(sample, dtype=np.float32)
        if values.shape != (self.model.input_steps, self.model.input_channels):
            raise ValueError(
                f"Expected sample {(self.model.input_steps, self.model.input_channels)}, got {values.shape}"
            )
        probabilities = self.predict_batch(values[None, ...])[0]
        class_id = int(np.argmax(probabilities))
        return Prediction(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=float(probabilities[class_id]),
            probabilities=probabilities,
        )

