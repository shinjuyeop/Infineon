"""Deployment-target TensorFlow model definition."""

from __future__ import annotations

import os
from typing import Any

from deepet_hil.schema import NUM_CHANNELS


def require_tensorflow() -> Any:
    """Import TensorFlow only when deployment functionality is requested."""

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for deployment commands. "
            "Install it with: pip install -r requirements-deploy.txt"
        ) from error
    return tf


def build_deployment_model(
    *,
    input_shape: tuple[int, int] = (50, NUM_CHANNELS),
    num_classes: int = 4,
    seed: int = 42,
) -> Any:
    """Build the fixed Scenario 1 Conv1D architecture for TFLite deployment."""

    if input_shape != (50, NUM_CHANNELS):
        raise ValueError(f"Deployment model requires input shape {(50, NUM_CHANNELS)}")
    if num_classes != 4:
        raise ValueError("Deployment model requires four terrain classes")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=input_shape, name="sensor_window")
    values = tf.keras.layers.Conv1D(16, 5, padding="same", name="conv1")(inputs)
    values = tf.keras.layers.ReLU(name="relu1")(values)
    values = tf.keras.layers.MaxPooling1D(pool_size=2, name="max_pool")(values)
    values = tf.keras.layers.Conv1D(32, 3, padding="same", name="conv2")(values)
    values = tf.keras.layers.ReLU(name="relu2")(values)
    values = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(values)
    values = tf.keras.layers.Dense(16, name="dense16")(values)
    values = tf.keras.layers.ReLU(name="relu3")(values)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="terrain_probabilities")(values)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="terrain_deployment_cnn_v0_1")
