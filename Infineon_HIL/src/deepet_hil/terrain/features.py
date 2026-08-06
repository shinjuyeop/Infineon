"""Canonical four-feature sanity checks explicitly defined for Scenario 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from deepet_hil.schema import Channel, NUM_CHANNELS

FEATURE_NAMES = (
    "vibration_rms_g",
    "spectral_centroid_hz",
    "peak_fsr_n",
    "max_load_drop_rate_n_per_ms",
)


def vibration_rms(vibration: ArrayLike) -> float:
    """Return sqrt(mean(vibration**2)) in g."""

    values = np.asarray(vibration, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def spectral_centroid(vibration: ArrayLike, sampling_rate_hz: float = 1000.0) -> float:
    """Return the one-sided power-spectrum centroid in Hz."""

    values = np.asarray(vibration, dtype=float)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(values)) ** 2
    denominator = float(power.sum())
    return 0.0 if denominator <= np.finfo(float).eps else float(np.dot(frequencies, power) / denominator)


def peak_fsr(fsr: ArrayLike) -> float:
    """Return the maximum across all four FSR channels and timesteps in N."""

    values = np.asarray(fsr, dtype=float)
    return float(np.max(values))


def max_load_drop_rate(fsr: ArrayLike, sampling_rate_hz: float = 1000.0) -> float:
    """Return the largest total-FSR decrease magnitude in N/ms."""

    values = np.asarray(fsr, dtype=float)
    total = values.sum(axis=-1)
    delta_ms = 1000.0 / sampling_rate_hz
    derivative = np.diff(total) / delta_ms
    return float(max(0.0, -np.min(derivative)))


def extract_features(
    x: ArrayLike, sampling_rate_hz: float = 1000.0
) -> NDArray[np.float64]:
    """Extract the four canonical features from one window or a batch."""

    values = np.asarray(x)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != NUM_CHANNELS:
        raise ValueError(f"Expected shape (time, {NUM_CHANNELS}) or (n, time, {NUM_CHANNELS})")
    rows = [
        (
            vibration_rms(window[:, Channel.VIBRATION]),
            spectral_centroid(window[:, Channel.VIBRATION], sampling_rate_hz),
            peak_fsr(window[:, : Channel.VIBRATION]),
            max_load_drop_rate(window[:, : Channel.VIBRATION], sampling_rate_hz),
        )
        for window in values
    ]
    return np.asarray(rows, dtype=np.float64)


def feature_frame(x: ArrayLike, sampling_rate_hz: float = 1000.0) -> pd.DataFrame:
    """Return canonical features as a named DataFrame."""

    return pd.DataFrame(extract_features(x, sampling_rate_hz), columns=FEATURE_NAMES)
