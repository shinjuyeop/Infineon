"""Portable preprocessing and quantization metadata for future firmware integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TensorQuantization:
    """One TFLite tensor's fixed-point interface."""

    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorQuantization":
        return cls(
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            scale=float(value["scale"]),
            zero_point=int(value["zero_point"]),
        )


@dataclass(frozen=True)
class DeploymentMetadata:
    """Information required to reproduce training-time preprocessing."""

    model_version: str
    channel_order: tuple[str, ...]
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]
    class_names: tuple[str, ...]
    input_shape: tuple[int, int]
    sampling_rate_hz: int
    window_length_ms: int
    split_seed: int
    tflite_input: TensorQuantization | None = None
    tflite_output: TensorQuantization | None = None

    def __post_init__(self) -> None:
        channels = self.input_shape[1]
        if len(self.channel_order) != channels:
            raise ValueError("Channel order does not match input shape")
        if len(self.channel_mean) != channels or len(self.channel_std) != channels:
            raise ValueError("Preprocessing vectors do not match input channels")
        if not np.isfinite(self.channel_mean).all() or not np.isfinite(self.channel_std).all():
            raise ValueError("Preprocessing statistics must be finite")
        if any(value <= 0.0 for value in self.channel_std):
            raise ValueError("Channel standard deviations must be positive")

    def normalize(self, samples: NDArray[np.floating]) -> NDArray[np.float32]:
        """Apply the exact train-only channel standardization."""

        values = np.asarray(samples, dtype=np.float32)
        if values.shape[-2:] != self.input_shape:
            raise ValueError(f"Expected trailing input shape {self.input_shape}, got {values.shape}")
        mean = np.asarray(self.channel_mean, dtype=np.float32)
        std = np.asarray(self.channel_std, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)

    def with_quantization(
        self,
        input_tensor: TensorQuantization,
        output_tensor: TensorQuantization,
    ) -> "DeploymentMetadata":
        """Return a copy enriched with the exported TFLite tensor interface."""

        values = asdict(self)
        values["tflite_input"] = input_tensor
        values["tflite_output"] = output_tensor
        return DeploymentMetadata(**values)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentMetadata":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("channel_order", "channel_mean", "channel_std", "class_names", "input_shape"):
            values[key] = tuple(values[key])
        for key in ("tflite_input", "tflite_output"):
            if values.get(key) is not None:
                values[key] = TensorQuantization.from_dict(values[key])
        return cls(**values)
