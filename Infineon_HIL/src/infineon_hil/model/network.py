"""A minimal trainable NumPy Conv1D network with portable NPZ weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from infineon_hil.schema import NUM_CHANNELS

FloatArray = NDArray[np.float32]


def _conv1d_same(x: FloatArray, weight: FloatArray, bias: FloatArray) -> tuple[FloatArray, FloatArray]:
    kernel = weight.shape[0]
    left = kernel // 2
    right = kernel - 1 - left
    padded = np.pad(x, ((0, 0), (left, right), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=1)
    windows = windows.transpose(0, 1, 3, 2)
    output = np.einsum("btkc,kco->bto", windows, weight, optimize=True) + bias
    return output.astype(np.float32), windows


def _conv1d_backward(
    gradient: FloatArray, windows: FloatArray, weight: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    kernel = weight.shape[0]
    time_steps = gradient.shape[1]
    grad_weight = np.einsum("btkc,bto->kco", windows, gradient, optimize=True)
    grad_bias = gradient.sum(axis=(0, 1))
    grad_padded = np.zeros(
        (gradient.shape[0], time_steps + kernel - 1, weight.shape[1]), dtype=np.float32
    )
    for offset in range(kernel):
        grad_padded[:, offset : offset + time_steps] += gradient @ weight[offset].T
    left = kernel // 2
    grad_input = grad_padded[:, left : left + time_steps]
    return grad_input, grad_weight.astype(np.float32), grad_bias.astype(np.float32)


class TerrainCNN:
    """Conv1D-ReLU-Conv1D-ReLU-GAP-Dense network for `(time, 5)` input."""

    def __init__(
        self,
        *,
        input_steps: int = 50,
        input_channels: int = NUM_CHANNELS,
        conv1_channels: int = 12,
        conv2_channels: int = 16,
        num_classes: int = 4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_steps = input_steps
        self.input_channels = input_channels
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.num_classes = num_classes
        self.params: dict[str, FloatArray] = {
            "conv1_w": self._xavier(rng, (5, input_channels, conv1_channels)),
            "conv1_b": np.zeros(conv1_channels, dtype=np.float32),
            "conv2_w": self._xavier(rng, (3, conv1_channels, conv2_channels)),
            "conv2_b": np.zeros(conv2_channels, dtype=np.float32),
            "dense_w": self._xavier(rng, (conv2_channels, num_classes)),
            "dense_b": np.zeros(num_classes, dtype=np.float32),
        }

    @staticmethod
    def _xavier(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        fan_in = int(np.prod(shape[:-1]))
        fan_out = shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, shape).astype(np.float32)

    def _validate_input(self, x: NDArray[np.floating]) -> FloatArray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (self.input_steps, self.input_channels):
            raise ValueError(
                f"Expected input (batch, {self.input_steps}, {self.input_channels}), got {values.shape}"
            )
        return values

    def forward(self, x: NDArray[np.floating], *, cache: bool = False) -> Any:
        """Return logits and, when requested, intermediates for backpropagation."""

        values = self._validate_input(x)
        z1, windows1 = _conv1d_same(values, self.params["conv1_w"], self.params["conv1_b"])
        a1 = np.maximum(z1, 0.0).astype(np.float32)
        z2, windows2 = _conv1d_same(a1, self.params["conv2_w"], self.params["conv2_b"])
        a2 = np.maximum(z2, 0.0).astype(np.float32)
        pooled = a2.mean(axis=1)
        logits = pooled @ self.params["dense_w"] + self.params["dense_b"]
        if not cache:
            return logits.astype(np.float32)
        return logits.astype(np.float32), (z1, windows1, z2, windows2, pooled)

    def loss_and_gradients(
        self, x: NDArray[np.floating], y: NDArray[np.integer]
    ) -> tuple[float, dict[str, FloatArray]]:
        """Compute mean softmax cross-entropy and exact network gradients."""

        labels = np.asarray(y, dtype=np.int64)
        logits, cache = self.forward(x, cache=True)
        if labels.shape != (logits.shape[0],):
            raise ValueError("Labels must have one entry per batch sample")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()
        grad_logits = probabilities
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        grad_logits = (grad_logits / len(labels)).astype(np.float32)
        z1, windows1, z2, windows2, pooled = cache
        gradients: dict[str, FloatArray] = {}
        gradients["dense_w"] = (pooled.T @ grad_logits).astype(np.float32)
        gradients["dense_b"] = grad_logits.sum(axis=0).astype(np.float32)
        grad_pooled = grad_logits @ self.params["dense_w"].T
        grad_a2 = np.repeat(
            (grad_pooled / self.input_steps)[:, None, :], self.input_steps, axis=1
        ).astype(np.float32)
        grad_z2 = grad_a2 * (z2 > 0.0)
        grad_a1, gradients["conv2_w"], gradients["conv2_b"] = _conv1d_backward(
            grad_z2, windows2, self.params["conv2_w"]
        )
        grad_z1 = grad_a1 * (z1 > 0.0)
        _, gradients["conv1_w"], gradients["conv1_b"] = _conv1d_backward(
            grad_z1, windows1, self.params["conv1_w"]
        )
        return float(loss), gradients

    def predict_proba(self, x: NDArray[np.floating]) -> FloatArray:
        """Return normalized four-class probabilities."""

        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def save(
        self,
        path: str | Path,
        *,
        channel_mean: NDArray[np.floating],
        channel_scale: NDArray[np.floating],
        class_names: tuple[str, ...],
    ) -> None:
        """Save weights, preprocessing, class mapping, and architecture to NPZ."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        architecture = {
            "input_steps": self.input_steps,
            "input_channels": self.input_channels,
            "conv1_channels": self.conv1_channels,
            "conv2_channels": self.conv2_channels,
            "num_classes": self.num_classes,
            "format": "infineon_hil_numpy_cnn_v1",
        }
        np.savez_compressed(
            output,
            **self.params,
            channel_mean=np.asarray(channel_mean, dtype=np.float32),
            channel_scale=np.asarray(channel_scale, dtype=np.float32),
            class_names=np.asarray(class_names),
            architecture=np.asarray(json.dumps(architecture)),
        )

    @classmethod
    def load(
        cls, path: str | Path
    ) -> tuple["TerrainCNN", FloatArray, FloatArray, tuple[str, ...]]:
        """Load a complete host inference artifact without pickle."""

        with np.load(Path(path), allow_pickle=False) as data:
            architecture = json.loads(str(data["architecture"]))
            if architecture.get("format") != "infineon_hil_numpy_cnn_v1":
                raise ValueError("Unsupported model artifact format")
            model = cls(
                input_steps=int(architecture["input_steps"]),
                input_channels=int(architecture["input_channels"]),
                conv1_channels=int(architecture["conv1_channels"]),
                conv2_channels=int(architecture["conv2_channels"]),
                num_classes=int(architecture["num_classes"]),
            )
            for name in model.params:
                model.params[name] = np.asarray(data[name], dtype=np.float32)
            mean = np.asarray(data["channel_mean"], dtype=np.float32)
            scale = np.asarray(data["channel_scale"], dtype=np.float32)
            class_names = tuple(str(value) for value in data["class_names"].tolist())
        return model, mean, scale, class_names

