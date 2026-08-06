"""Strict full-INT8 TensorFlow Lite export and tensor-interface inspection."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metadata import TensorQuantization
from .model import require_tensorflow


def _tensor_quantization(detail: dict[str, Any]) -> TensorQuantization:
    scale, zero_point = detail["quantization"]
    if float(scale) <= 0.0:
        raise ValueError(f"Tensor {detail['name']} has no valid per-tensor quantization scale")
    return TensorQuantization(
        shape=tuple(int(value) for value in detail["shape"]),
        dtype=np.dtype(detail["dtype"]).name,
        scale=float(scale),
        zero_point=int(zero_point),
    )


def inspect_tflite_model(path: str | Path) -> tuple[TensorQuantization, TensorQuantization]:
    """Return and validate the single INT8 input/output tensor descriptions."""

    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(Path(path)))
    interpreter.allocate_tensors()
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if get_ops is not None:
        flex_ops = [
            detail["op_name"]
            for detail in get_ops()
            if str(detail.get("op_name", "")).startswith("Flex")
        ]
        if flex_ops:
            raise ValueError(f"Flex operators are not allowed: {sorted(set(flex_ops))}")
    floating_tensors = [
        detail["name"]
        for detail in interpreter.get_tensor_details()
        if np.issubdtype(np.dtype(detail["dtype"]), np.floating)
    ]
    if floating_tensors:
        raise ValueError(
            "Full-integer model contains floating-point tensors: "
            + ", ".join(floating_tensors)
        )
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Deployment model must expose exactly one input and one output")
    input_tensor = _tensor_quantization(inputs[0])
    output_tensor = _tensor_quantization(outputs[0])
    if input_tensor.dtype != "int8" or output_tensor.dtype != "int8":
        raise ValueError(
            f"Expected full INT8 I/O, got input={input_tensor.dtype}, output={output_tensor.dtype}"
        )
    if input_tensor.shape != (1, 50, 5) or output_tensor.shape != (1, 4):
        raise ValueError(
            f"Unexpected tensor shapes: input={input_tensor.shape}, output={output_tensor.shape}"
        )
    return input_tensor, output_tensor


def export_full_int8_tflite(
    model: Any,
    representative_samples: NDArray[np.floating],
    output_path: str | Path,
) -> tuple[TensorQuantization, TensorQuantization]:
    """Export with INT8-only built-ins and reject non-INT8 external tensors."""

    samples = np.asarray(representative_samples, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[1:] != (50, 5) or len(samples) == 0:
        raise ValueError("Representative data must have shape (N, 50, 5) with N > 0")
    tf = require_tensorflow()

    def representative_dataset():
        for sample in samples:
            yield [sample[None, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converted = converter.convert()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".tflite", delete=False
        ) as stream:
            stream.write(converted)
            temporary_path = Path(stream.name)
        tensor_specs = inspect_tflite_model(temporary_path)
        temporary_path.replace(output)
        return tensor_specs
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
