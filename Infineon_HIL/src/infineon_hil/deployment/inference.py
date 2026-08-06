"""Host-side inference through the exported INT8 TensorFlow Lite model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metadata import DeploymentMetadata, TensorQuantization
from .model import require_tensorflow
from .quantization import inspect_tflite_model


@dataclass(frozen=True)
class TFLitePrediction:
    """Dequantized prediction returned by the host interpreter."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: NDArray[np.float32]


def _quantize(values: NDArray[np.float32], spec: TensorQuantization) -> NDArray[np.int8]:
    quantized = np.rint(values / spec.scale + spec.zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def _dequantize(values: NDArray[np.integer], spec: TensorQuantization) -> NDArray[np.float32]:
    return ((values.astype(np.float32) - spec.zero_point) * spec.scale).astype(np.float32)


class TFLiteInferenceEngine:
    """Own exact preprocessing, fixed-point conversion, and interpreter invocation."""

    def __init__(self, model_path: str | Path, metadata: DeploymentMetadata) -> None:
        tf = require_tensorflow()
        self.metadata = metadata
        self.interpreter: Any = tf.lite.Interpreter(model_path=str(Path(model_path)))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_spec, self.output_spec = inspect_tflite_model(model_path)
        if metadata.tflite_input is not None and metadata.tflite_input != self.input_spec:
            raise ValueError("TFLite input details do not match metadata")
        if metadata.tflite_output is not None and metadata.tflite_output != self.output_spec:
            raise ValueError("TFLite output details do not match metadata")

    def predict_standardized_batch(
        self, samples: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Run already-standardized windows through the fixed batch-one model."""

        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != self.metadata.input_shape:
            raise ValueError(f"Expected (N, {self.metadata.input_shape}), got {values.shape}")
        outputs = np.empty((len(values), len(self.metadata.class_names)), dtype=np.float32)
        for index, sample in enumerate(values):
            input_value = _quantize(sample[None, ...], self.input_spec)
            self.interpreter.set_tensor(self.input_detail["index"], input_value)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_detail["index"])
            outputs[index] = _dequantize(raw_output, self.output_spec)[0]
        return outputs

    def predict_batch(self, samples: NDArray[np.floating]) -> NDArray[np.float32]:
        """Normalize physical-unit windows and run INT8 inference."""

        return self.predict_standardized_batch(self.metadata.normalize(samples))

    def predict(self, sample: NDArray[np.floating]) -> TFLitePrediction:
        values = np.asarray(sample, dtype=np.float32)
        if values.shape != self.metadata.input_shape:
            raise ValueError(f"Expected sample {self.metadata.input_shape}, got {values.shape}")
        probabilities = self.predict_batch(values[None, ...])[0]
        class_id = int(np.argmax(probabilities))
        return TFLitePrediction(
            class_id=class_id,
            class_name=self.metadata.class_names[class_id],
            confidence=float(probabilities[class_id]),
            probabilities=probabilities,
        )
