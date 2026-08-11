#!/usr/bin/env python3
"""Host shadow inference for the immutable strict-INT8 terrain model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from terrain_preprocessing import MODEL, verify_canonical_model
except ModuleNotFoundError:
    from tools.terrain_preprocessing import MODEL, verify_canonical_model


def _interpreter_factory():
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        return Interpreter, {
            "experimental_op_resolver_type": OpResolverType.BUILTIN_REF
        }
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
        return Interpreter, {
            "experimental_op_resolver_type": OpResolverType.BUILTIN_REF
        }
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, {
            "experimental_op_resolver_type": (
                tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
        }
    except ImportError as exc:
        raise RuntimeError(
            "Host shadow inference requires ai-edge-litert, tflite-runtime, "
            "or TensorFlow"
        ) from exc


class TerrainShadowModel:
    def __init__(self, model_path: Path = MODEL) -> None:
        self.identity = verify_canonical_model(model_path)
        interpreter_class, options = _interpreter_factory()
        # Reference INT8 kernels are intentional: optimized Host delegates can
        # use different legal rounding paths than TFLM/Ethos-U55 and differ by
        # a few output LSBs even when the class is unchanged.
        interpreter = interpreter_class(model_path=str(model_path), **options)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        if (
            tuple(int(value) for value in input_detail["shape"]) != (1, 50, 10)
            or np.dtype(input_detail["dtype"]) != np.dtype(np.int8)
            or tuple(int(value) for value in output_detail["shape"]) != (1, 4)
            or np.dtype(output_detail["dtype"]) != np.dtype(np.int8)
        ):
            raise ValueError(
                "canonical Host shadow model does not have "
                "INT8 (1,50,10)->(1,4) I/O"
            )
        self.interpreter = interpreter
        self.input_index = int(input_detail["index"])
        self.output_index = int(output_detail["index"])

    def infer(self, quantized_window: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(quantized_window, dtype=np.int8)
        if values.shape != (50, 10):
            raise ValueError(f"Host shadow expects (50,10), got {values.shape}")
        self.interpreter.set_tensor(self.input_index, values[None, ...])
        self.interpreter.invoke()
        raw = np.asarray(
            self.interpreter.get_tensor(self.output_index)[0], dtype=np.int8
        ).copy()
        return raw, int(np.argmax(raw))
