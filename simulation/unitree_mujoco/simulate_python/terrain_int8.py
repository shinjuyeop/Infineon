"""Strict full-INT8 export and parity helpers for Dataset v1 Fusion10."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SHAPE = (50, 10)
OUTPUT_CLASSES = 4


@dataclass(frozen=True)
class TensorQuantization:
    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required; install simulate_python/requirements-cnn.txt"
        ) from exc
    return tf


def _tensor_quantization(detail: dict[str, Any]) -> TensorQuantization:
    scale, zero_point = detail["quantization"]
    if float(scale) <= 0.0:
        raise ValueError(f"tensor {detail['name']} has no valid quantization scale")
    return TensorQuantization(
        shape=tuple(int(value) for value in detail["shape"]),
        dtype=np.dtype(detail["dtype"]).name,
        scale=float(scale),
        zero_point=int(zero_point),
    )


def inspect_tflite_model(path: str | Path) -> tuple[TensorQuantization, TensorQuantization]:
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(Path(path)))
    interpreter.allocate_tensors()
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if get_ops is not None:
        flex_ops = [
            str(detail.get("op_name", ""))
            for detail in get_ops()
            if str(detail.get("op_name", "")).startswith("Flex")
        ]
        if flex_ops:
            raise ValueError(f"Flex operators are not allowed: {sorted(set(flex_ops))}")
    floating = [
        str(detail["name"])
        for detail in interpreter.get_tensor_details()
        if np.issubdtype(np.dtype(detail["dtype"]), np.floating)
    ]
    if floating:
        raise ValueError("full-INT8 model contains floating tensors: " + ", ".join(floating))
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("model must expose exactly one input and one output")
    input_spec = _tensor_quantization(inputs[0])
    output_spec = _tensor_quantization(outputs[0])
    if input_spec.dtype != "int8" or output_spec.dtype != "int8":
        raise ValueError(
            f"expected INT8 I/O, got input={input_spec.dtype}, output={output_spec.dtype}"
        )
    if input_spec.shape != (1, *INPUT_SHAPE) or output_spec.shape != (1, OUTPUT_CLASSES):
        raise ValueError(
            f"unexpected tensor shapes: input={input_spec.shape}, output={output_spec.shape}"
        )
    return input_spec, output_spec


def inspect_binary_tflite_model(path: str | Path, window_ms: int) -> tuple[TensorQuantization, TensorQuantization, list[str]]:
    """Strict no-Flex INT8 audit for a Fast Reflex binary `(L,10)->(1,)` model."""
    tf = require_tensorflow(); interpreter = tf.lite.Interpreter(model_path=str(Path(path))); interpreter.allocate_tensors()
    ops = [str(d.get("op_name", "")) for d in interpreter._get_ops_details() if str(d.get("op_name", "")) != "DELEGATE"]
    if any(not op or op.startswith("Flex") for op in ops): raise ValueError(f"Flex/invalid ops prohibited: {ops}")
    floating = [str(d["name"]) for d in interpreter.get_tensor_details() if np.issubdtype(np.dtype(d["dtype"]), np.floating)]
    if floating: raise ValueError("full-INT8 model contains floating tensors: " + ", ".join(floating))
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1: raise ValueError("binary model must have exactly one input/output")
    inp, out = _tensor_quantization(inputs[0]), _tensor_quantization(outputs[0])
    if inp.dtype != "int8" or out.dtype != "int8" or inp.shape != (1, window_ms, 10) or out.shape not in ((1, 1), (1,)):
        raise ValueError(f"unexpected binary INT8 interface input={inp} output={out}")
    return inp, out, ops


def export_full_int8_binary(model: Any, representative_samples: np.ndarray, output_path: str | Path, window_ms: int) -> tuple[TensorQuantization, TensorQuantization, list[str]]:
    samples = np.asarray(representative_samples, np.float32)
    if samples.ndim != 3 or samples.shape[1:] != (window_ms, 10) or not len(samples) or not np.isfinite(samples).all(): raise ValueError("invalid binary representative samples")
    tf = require_tensorflow()
    def representative_dataset():
        for sample in samples: yield [sample[None, ...]]
    converter = tf.lite.TFLiteConverter.from_keras_model(model); converter.optimizations = [tf.lite.Optimize.DEFAULT]; converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; converter.inference_input_type = tf.int8; converter.inference_output_type = tf.int8
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(converter.convert())
    return inspect_binary_tflite_model(output, window_ms)


def predict_tflite_binary(path: str | Path, standardized: np.ndarray, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(standardized, np.float32)
    if values.ndim != 3 or values.shape[1:] != (window_ms, 10): raise ValueError("unexpected binary inference shape")
    tf = require_tensorflow(); interpreter = tf.lite.Interpreter(model_path=str(Path(path))); interpreter.allocate_tensors()
    inp, out, _ = inspect_binary_tflite_model(path, window_ms); input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    raw = np.empty(len(values), np.int8)
    for i, sample in enumerate(values):
        interpreter.set_tensor(input_detail["index"], quantize(sample[None], inp)); interpreter.invoke(); raw[i] = interpreter.get_tensor(output_detail["index"]).reshape(-1)[0]
    return dequantize(raw, out).reshape(-1), raw


def list_tflite_operators(path: str | Path) -> list[str]:
    """Return the ordered builtin operator names used by a validated model."""
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(Path(path)))
    interpreter.allocate_tensors()
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if get_ops is None:
        raise RuntimeError("TensorFlow Lite operator inspection is unavailable")
    # DELEGATE entries describe this host interpreter's execution plan, not
    # operators serialized in the TFLite flatbuffer.
    operators = [
        str(detail.get("op_name", ""))
        for detail in get_ops()
        if str(detail.get("op_name", "")) != "DELEGATE"
    ]
    if any(not name or name.startswith("Flex") for name in operators):
        raise ValueError(f"invalid operator list: {operators}")
    return operators


def export_full_int8_tflite(
    model: Any,
    representative_samples: np.ndarray,
    output_path: str | Path,
) -> tuple[TensorQuantization, TensorQuantization]:
    samples = np.asarray(representative_samples, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[1:] != INPUT_SHAPE or len(samples) == 0:
        raise ValueError(f"representative data must have shape (N, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]})")
    if not np.all(np.isfinite(samples)):
        raise ValueError("representative data contains NaN/Inf")
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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".tflite", delete=False
        ) as stream:
            stream.write(converted)
            temporary = Path(stream.name)
        specs = inspect_tflite_model(temporary)
        temporary.replace(output)
        return specs
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_calibration_indices(
    split: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    split = np.asarray(split)
    labels = np.asarray(labels)
    families = np.asarray(families)
    if not (len(split) == len(labels) == len(families)):
        raise ValueError("calibration metadata arrays have inconsistent lengths")
    if count <= 0:
        raise ValueError("calibration sample count must be positive")
    family_owners: dict[str, set[str]] = {}
    for family, split_name in zip(families.tolist(), split.tolist()):
        family_owners.setdefault(str(family), set()).add(str(split_name))
    leaking = {name: values for name, values in family_owners.items() if len(values) != 1}
    if leaking:
        raise ValueError(f"surface family leakage in dataset: {leaking}")
    train_indices = np.flatnonzero(split == "train")
    if count > len(train_indices):
        raise ValueError("requested more calibration samples than the train partition contains")
    strata = sorted(
        {
            (int(labels[index]), str(families[index]))
            for index in train_indices
        }
    )
    if not strata:
        raise ValueError("train partition has no calibration strata")
    rng = np.random.default_rng(seed)
    base, remainder = divmod(count, len(strata))
    selected: list[int] = []
    for position, (label, family) in enumerate(strata):
        candidates = np.flatnonzero(
            (split == "train") & (labels == label) & (families == family)
        )
        take = base + int(position < remainder)
        if take > len(candidates):
            raise ValueError(f"calibration stratum {(label, family)} is too small")
        selected.extend(int(value) for value in rng.choice(candidates, size=take, replace=False))
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    if len(result) != count or np.any(split[result] != "train"):
        raise RuntimeError("calibration selection violated the train-only contract")
    return result


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if values.shape[-2:] != INPUT_SHAPE or mean.shape != (10,) or std.shape != (10,):
        raise ValueError("Fusion normalization expects (N,50,10) and ten channel statistics")
    if np.any(std <= 0.0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("normalization statistics are invalid")
    result = (values - mean) / std
    if not np.all(np.isfinite(result)):
        raise ValueError("normalization produced NaN/Inf")
    return result.astype(np.float32, copy=False)


def quantize(values: np.ndarray, spec: TensorQuantization) -> np.ndarray:
    if spec.dtype != "int8" or spec.scale <= 0.0:
        raise ValueError("quantization requires a valid INT8 tensor spec")
    transformed = np.rint(np.asarray(values, dtype=np.float32) / spec.scale + spec.zero_point)
    return np.clip(transformed, -128, 127).astype(np.int8)


def dequantize(values: np.ndarray, spec: TensorQuantization) -> np.ndarray:
    if spec.dtype != "int8" or spec.scale <= 0.0:
        raise ValueError("dequantization requires a valid INT8 tensor spec")
    return (np.asarray(values, dtype=np.float32) - spec.zero_point) * spec.scale


def predict_tflite(path: str | Path, standardized: np.ndarray) -> np.ndarray:
    values = np.asarray(standardized, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != INPUT_SHAPE:
        raise ValueError("TFLite prediction expects (N,50,10)")
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(Path(path)))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_spec = _tensor_quantization(input_detail)
    output_spec = _tensor_quantization(output_detail)
    predictions = np.empty((len(values), OUTPUT_CLASSES), dtype=np.float32)
    for index, sample in enumerate(values):
        interpreter.set_tensor(
            input_detail["index"], quantize(sample[None, ...], input_spec)
        )
        interpreter.invoke()
        predictions[index] = dequantize(
            interpreter.get_tensor(output_detail["index"]), output_spec
        )[0]
    if not np.all(np.isfinite(predictions)):
        raise ValueError("TFLite inference produced NaN/Inf")
    return predictions


def parity_gate(
    float_accuracy: float,
    int8_accuracy: float,
    float_macro_f1: float,
    int8_macro_f1: float,
    float_pair_confusion: float,
    int8_pair_confusion: float,
) -> dict[str, object]:
    accuracy_delta = int8_accuracy - float_accuracy
    macro_f1_delta = int8_macro_f1 - float_macro_f1
    pair_confusion_delta = int8_pair_confusion - float_pair_confusion
    checks = {
        "accuracy_delta_at_least_minus_1pp": accuracy_delta >= -0.01,
        "macro_f1_delta_at_least_minus_1pp": macro_f1_delta >= -0.01,
        "concrete_marble_confusion_increase_at_most_1pp": pair_confusion_delta <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "accuracy_delta": accuracy_delta,
        "macro_f1_delta": macro_f1_delta,
        "concrete_marble_confusion_delta": pair_confusion_delta,
    }
