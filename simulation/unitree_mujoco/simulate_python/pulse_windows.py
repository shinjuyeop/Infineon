"""Canonical pulse-aligned windows for terrain-response analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowProfile:
    name: str
    start_time: float
    end_time: float | None
    description: str


WINDOW_PROFILES = {
    "pulse_only": WindowProfile(
        "pulse_only", 0.25, 0.45, "half-sine pulse interval [0.25, 0.45)",
    ),
    "short_response": WindowProfile(
        "short_response", 0.20, 0.55, "pre/pulse/early response [0.20, 0.55)",
    ),
    "medium_response": WindowProfile(
        "medium_response", 0.15, 0.65, "pre/pulse/medium response [0.15, 0.65)",
    ),
    "full": WindowProfile(
        "full", 0.0, None, "complete recorded run",
    ),
}


def sample_indices(
    sample_count: int,
    sample_rate: float,
    profile: WindowProfile,
) -> np.ndarray:
    """Return deterministic indices for samples recorded at n / sample_rate.

    Non-full profiles use a left-closed, right-open interval. The full profile
    selects every recorded sample, including the sample at the run end.
    """
    if sample_count <= 0 or sample_rate <= 0.0:
        raise ValueError("sample_count and sample_rate must be positive")
    ideal_times = np.arange(1, sample_count + 1, dtype=np.float64) / sample_rate
    tolerance = np.finfo(np.float64).eps * max(1.0, sample_count / sample_rate) * 8.0
    mask = ideal_times >= profile.start_time - tolerance
    if profile.end_time is not None:
        mask &= ideal_times < profile.end_time - tolerance
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"window {profile.name!r} selects no samples")
    return indices


def extract_window(
    timestamps: np.ndarray,
    sensors: np.ndarray,
    profile: WindowProfile,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one profile without modifying the original run arrays."""
    if timestamps.ndim != 1 or sensors.ndim != 2:
        raise ValueError("timestamps must be 1-D and sensors must be 2-D")
    if len(timestamps) != len(sensors):
        raise ValueError("timestamp and sensor lengths differ")
    indices = sample_indices(len(timestamps), sample_rate, profile)
    expected = np.arange(1, len(timestamps) + 1, dtype=np.float64) / sample_rate
    tolerance = max(1e-9, 0.01 / sample_rate)
    if not np.allclose(timestamps, expected, rtol=0.0, atol=tolerance):
        raise ValueError("timestamps do not match the configured deterministic sample grid")
    return timestamps[indices].copy(), sensors[indices].copy()


def window_shapes(
    sample_count: int,
    sample_rate: float,
    channel_count: int = 10,
) -> dict[str, tuple[int, int]]:
    return {
        name: (len(sample_indices(sample_count, sample_rate, profile)), channel_count)
        for name, profile in WINDOW_PROFILES.items()
    }
