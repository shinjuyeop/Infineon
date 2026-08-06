"""Future HIL voltage mapping interfaces; datasets remain in N and g."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def physical_to_normalized(
    value: ArrayLike, *, physical_min: float, physical_max: float
) -> NDArray[np.float64]:
    """Map a declared physical range to [0, 1] without hardware assumptions."""

    if physical_max <= physical_min:
        raise ValueError("physical_max must be greater than physical_min")
    values = np.asarray(value, dtype=float)
    return np.clip((values - physical_min) / (physical_max - physical_min), 0.0, 1.0)


def fsr_n_to_normalized(
    value: ArrayLike, *, full_scale_n: float = 200.0
) -> NDArray[np.float64]:
    """Map 0..full-scale FSR newtons to normalized values."""

    return physical_to_normalized(value, physical_min=0.0, physical_max=full_scale_n)


def vibration_g_to_normalized(
    value: ArrayLike, *, full_scale_g: float
) -> NDArray[np.float64]:
    """Map bipolar vibration g to normalized values around 0.5."""

    if full_scale_g <= 0:
        raise ValueError("full_scale_g must be positive")
    return physical_to_normalized(
        value, physical_min=-full_scale_g, physical_max=full_scale_g
    )


def normalized_to_output_code(value: ArrayLike, *, max_code: int) -> NDArray[np.int64]:
    """Quantize [0, 1] only when a future backend supplies its max code."""

    if max_code <= 0:
        raise ValueError("max_code must be positive")
    normalized = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return np.rint(normalized * max_code).astype(np.int64)


def output_code_to_physical(
    code: ArrayLike, *, max_code: int, physical_min: float, physical_max: float
) -> NDArray[np.float64]:
    """Invert a declared linear code mapping for validation/reconstruction."""

    if max_code <= 0 or physical_max <= physical_min:
        raise ValueError("Invalid code or physical range")
    normalized = np.clip(np.asarray(code, dtype=float), 0.0, max_code) / max_code
    return physical_min + normalized * (physical_max - physical_min)


def fsr_n_to_voltage(
    value: ArrayLike, *, full_scale_n: float = 200.0, full_scale_v: float = 3.3
) -> NDArray[np.float64]:
    """Apply a conceptual voltage mapping, not a validated KIT_PSE84_AI interface."""

    return np.clip(np.asarray(value, dtype=float), 0.0, full_scale_n) * full_scale_v / full_scale_n


def vibration_g_to_voltage(
    value: ArrayLike,
    *,
    full_scale_g: float,
    midrail_v: float = 1.65,
    amplitude_v: float = 1.5,
) -> NDArray[np.float64]:
    """Apply a conceptual mid-rail mapping using an explicitly supplied scale."""

    if full_scale_g <= 0:
        raise ValueError("full_scale_g must be positive")
    normalized = np.clip(np.asarray(value, dtype=float) / full_scale_g, -1.0, 1.0)
    return midrail_v + normalized * amplitude_v
