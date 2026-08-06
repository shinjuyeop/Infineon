"""Reusable synthetic-data checks and the deliberately simple sanity baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deepet_hil.schema import Channel, NUM_CHANNELS

from .features import extract_features


@dataclass(frozen=True)
class BaselineResult:
    """Metrics from the four-feature nearest-centroid sanity baseline."""

    accuracy: float
    recalls: NDArray[np.float64]
    confusion_matrix: NDArray[np.int64]
    labels: NDArray[np.int64]
    test_samples: int


def validate_sensor_tensor(x: NDArray[np.floating], y: NDArray[np.integer]) -> None:
    """Raise ValueError for structural, finite-value, or FSR-range failures."""

    if x.ndim != 3 or x.shape[-1] != NUM_CHANNELS or y.shape != (x.shape[0],):
        raise ValueError(f"Expected X=(n, time, {NUM_CHANNELS}) and y=(n,)")
    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN or Inf")
    if np.any(x[:, :, : Channel.VIBRATION] < 0.0) or np.any(x[:, :, : Channel.VIBRATION] > 200.0):
        raise ValueError("FSR data is outside 0..200 N")


def run_sanity_baseline(
    x: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    sampling_rate_hz: float = 1000.0,
    seed: int = 42,
    test_size: float = 0.25,
) -> BaselineResult:
    """Run a stratified nearest-centroid classifier on four canonical features."""

    validate_sensor_tensor(x, y)
    indices = np.arange(len(y))
    train, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )
    features = extract_features(x, sampling_rate_hz)
    model = make_pipeline(StandardScaler(), NearestCentroid())
    model.fit(features[train], y[train])
    prediction = model.predict(features[test])
    labels = np.unique(y).astype(np.int64)
    return BaselineResult(
        accuracy=float(accuracy_score(y[test], prediction)),
        recalls=np.asarray(
            recall_score(y[test], prediction, labels=labels, average=None, zero_division=0),
            dtype=np.float64,
        ),
        confusion_matrix=np.asarray(
            confusion_matrix(y[test], prediction, labels=labels), dtype=np.int64
        ),
        labels=labels,
        test_samples=len(test),
    )
