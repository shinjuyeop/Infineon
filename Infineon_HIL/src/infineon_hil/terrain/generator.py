"""Pure synthetic waveform generation for Scenario 1.

The parameters consumed here are engineering estimates, not measured signals.
No device communication or class label is embedded in the returned waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .params import TerrainConfig, TerrainParameters

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class GeneratedWindow:
    """A sensor-only window and separate generation metadata."""

    waveform: FloatArray
    metadata: dict[str, Any]


def damping_envelope(
    time_steps: int, damping: float, decay_scale: float
) -> NDArray[np.float64]:
    """Map relative synthesis damping to a replaceable exponential envelope."""

    normalized_time = np.linspace(0.0, 1.0, time_steps)
    return np.exp(-float(damping) * float(decay_scale) * normalized_time)


def _total_load_waveform(
    terrain: TerrainParameters,
    config: TerrainConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], float, float]:
    assumptions = config.raw["engineering_assumptions"]["fsr"]
    steps = config.time_steps
    peak = float(rng.uniform(*terrain.fsr_peak_n))
    rise_ms = float(rng.uniform(*terrain.rise_time_ms))
    start = int(rng.integers(
        int(assumptions["contact_start_ms"]["min"]),
        int(assumptions["contact_start_ms"]["max"]) + 1,
    ))
    rise_end = min(start + max(1, round(rise_ms)), steps - 2)
    plateau_ms = int(rng.integers(
        int(assumptions["plateau_duration_ms"]["min"]),
        int(assumptions["plateau_duration_ms"]["max"]) + 1,
    ))
    plateau_end = min(rise_end + plateau_ms, steps - 1)
    plateau_level = peak * rng.uniform(0.91, 1.0)
    control_x = np.array([0, start, rise_end, plateau_end, steps - 1])
    control_y = np.array([0.0, 0.0, peak, plateau_level, 0.0])
    # Remove repeated control positions possible at the short-window boundary.
    unique_x, unique_indices = np.unique(control_x, return_index=True)
    total = np.interp(np.arange(steps), unique_x, control_y[unique_indices])
    return total, peak, rise_ms


def _spatial_fsr(
    total: NDArray[np.float64],
    terrain: TerrainParameters,
    config: TerrainConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    assumptions = config.raw["engineering_assumptions"]["fsr"]
    base = np.asarray(assumptions["base_spatial_weights"], dtype=float)
    weight_std = float(assumptions["spatial_weight_std"][terrain.name])
    weights = np.clip(base + rng.normal(0.0, weight_std, 4), 0.03, None)
    weights /= weights.sum()
    dynamic = np.ones((config.time_steps, 4), dtype=float)
    if terrain.name == "sand":
        amplitude = float(assumptions["sand_low_frequency_variation_fraction"])
        phase = rng.uniform(0.0, 2.0 * np.pi, 4)
        frequency = rng.uniform(0.5, 2.0, 4)
        t = np.linspace(0.0, 1.0, config.time_steps)
        dynamic += amplitude * np.sin(2.0 * np.pi * t[:, None] * frequency + phase)
    dynamic_weights = weights[None, :] * np.clip(dynamic, 0.05, None)
    dynamic_weights /= dynamic_weights.sum(axis=1, keepdims=True)
    return total[:, None] * dynamic_weights


def _apply_micro_slip(
    fsr: NDArray[np.float64],
    config: TerrainConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, float | None]]:
    slip = config.raw["engineering_assumptions"]["micro_slip"]
    start = int(rng.integers(int(slip["slip_start_ms"]["min"]), int(slip["slip_start_ms"]["max"]) + 1))
    duration = int(rng.integers(int(slip["slip_duration_ms"]["min"]), int(slip["slip_duration_ms"]["max"]) + 1))
    drop = float(rng.uniform(float(slip["slip_drop_fraction"]["min"]), float(slip["slip_drop_fraction"]["max"])))
    channel_variation = float(slip["channel_drop_variation"])
    channel_drops = np.clip(drop * (1.0 + rng.normal(0.0, channel_variation, 4)), 0.0, 0.95)
    end = min(start + duration, config.time_steps)
    output = fsr.copy()
    output[start:end] *= 1.0 - channel_drops[None, :]
    if end < config.time_steps:
        recovery = float(rng.uniform(float(slip["recovery_fraction"]["min"]), float(slip["recovery_fraction"]["max"])))
        residual = channel_drops * (1.0 - recovery)
        ramp = np.linspace(1.0, 0.0, config.time_steps - end)[:, None]
        output[end:] *= 1.0 - residual[None, :] * ramp
    return output, {
        "micro_slip_start_ms": float(start),
        "micro_slip_duration_ms": float(duration),
        "micro_slip_drop_fraction": drop,
    }


def _vibration(
    terrain: TerrainParameters,
    config: TerrainConfig,
    rng: np.random.Generator,
    micro_slip: bool,
    slip_start_ms: float | None,
) -> tuple[NDArray[np.float64], float]:
    assumptions = config.raw["engineering_assumptions"]["vibration"]
    steps = config.time_steps
    frequencies = np.fft.rfftfreq(steps, d=1.0 / config.sampling_rate_hz)
    low, high = terrain.vibration_band_hz
    band = (frequencies >= low) & (frequencies <= min(high, config.sampling_rate_hz / 2))
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    spectrum[band] = rng.normal(size=band.sum()) + 1j * rng.normal(size=band.sum())
    spectrum[0] = 0.0
    if steps % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    signal = np.fft.irfft(spectrum, n=steps)
    signal *= damping_envelope(steps, terrain.damping, float(assumptions["damping_decay_scale"]))
    noise_fraction = rng.uniform(
        float(assumptions["additive_noise_fraction"]["min"]),
        float(assumptions["additive_noise_fraction"]["max"]),
    )
    signal += rng.normal(0.0, max(np.std(signal), 1e-9) * noise_fraction, steps)
    rms_variation = float(assumptions["rms_randomization_fraction"])
    target_rms = terrain.vibration_rms_g * rng.uniform(1.0 - rms_variation, 1.0 + rms_variation)
    current_rms = float(np.sqrt(np.mean(np.square(signal))))
    signal *= target_rms / max(current_rms, 1e-12)

    if terrain.impulsive and micro_slip and slip_start_ms is not None:
        duration_cfg = assumptions["impulse_duration_ms"]
        duration = int(rng.integers(int(duration_cfg["min"]), int(duration_cfg["max"]) + 1))
        multiplier_cfg = assumptions["impulse_amplitude_multiplier"]
        multiplier = rng.uniform(float(multiplier_cfg["min"]), float(multiplier_cfg["max"]))
        start = int(slip_start_ms)
        end = min(start + duration, steps)
        burst_length = end - start
        if burst_length:
            burst = rng.normal(0.0, target_rms * multiplier, burst_length)
            burst *= np.hanning(burst_length + 2)[1:-1]
            signal[start:end] += burst
    gain_error = float(assumptions["sensor_gain_error_fraction"])
    signal *= rng.uniform(1.0 - gain_error, 1.0 + gain_error)
    return signal, target_rms


def generate_window(
    terrain_name: str,
    config: TerrainConfig,
    rng: np.random.Generator,
    *,
    random_seed: int | None = None,
) -> GeneratedWindow:
    """Generate one `(time_steps, 5)` sensor window in N and g."""

    try:
        terrain = config.terrains[terrain_name]
    except KeyError as error:
        raise ValueError(f"Unknown terrain: {terrain_name}") from error
    total, sampled_peak, sampled_rise = _total_load_waveform(terrain, config, rng)
    fsr = _spatial_fsr(total, terrain, config, rng)
    slip_occurred = bool(rng.random() < terrain.micro_slip_probability)
    slip_metadata: dict[str, float | None] = {
        "micro_slip_start_ms": None,
        "micro_slip_duration_ms": None,
        "micro_slip_drop_fraction": None,
    }
    if slip_occurred:
        fsr, slip_metadata = _apply_micro_slip(fsr, config, rng)

    fsr_assumptions = config.raw["engineering_assumptions"]["fsr"]
    gain = float(fsr_assumptions["sensor_gain_error_fraction"])
    fsr *= rng.uniform(1.0 - gain, 1.0 + gain, 4)[None, :]
    offset = float(fsr_assumptions["sensor_offset_n"])
    fsr += rng.uniform(-offset, offset, 4)[None, :]
    noise_fraction = rng.uniform(
        float(fsr_assumptions["temporal_noise_fraction"]["min"]),
        float(fsr_assumptions["temporal_noise_fraction"]["max"]),
    )
    fsr += rng.normal(0.0, sampled_peak * noise_fraction / 4.0, fsr.shape)
    output_range = fsr_assumptions["output_range_n"]
    low, high = float(output_range["min"]), float(output_range["max"])
    clipping_count = int(np.count_nonzero((fsr < low) | (fsr > high)))
    fsr = np.clip(fsr, low, high)

    vibration, target_rms = _vibration(
        terrain, config, rng, slip_occurred, slip_metadata["micro_slip_start_ms"]
    )
    waveform = np.column_stack((fsr, vibration)).astype(np.float32)
    actual_rms = float(np.sqrt(np.mean(np.square(vibration))))
    metadata: dict[str, Any] = {
        "class_id": terrain.class_id,
        "terrain_name": terrain.name,
        "random_seed": random_seed,
        "sampled_fsr_peak_n": sampled_peak,
        "sampled_rise_time_ms": sampled_rise,
        "target_vibration_rms_g": target_rms,
        "actual_vibration_rms_g": actual_rms,
        "micro_slip_occurred": slip_occurred,
        **slip_metadata,
        "friction_mu": terrain.friction_mu,
        "relative_stiffness": terrain.relative_stiffness,
        "damping": terrain.damping,
        "fsr_clipping_count": clipping_count,
    }
    return GeneratedWindow(waveform=waveform, metadata=metadata)

