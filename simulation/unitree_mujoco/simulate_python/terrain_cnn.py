"""Shared compact 1D-CNN and split-safe preprocessing for terrain ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


TIME_STEPS = 50
CLASS_COUNT = 4
CONV1_FILTERS = 12
CONV2_FILTERS = 16
CONV1_KERNEL = 5
CONV2_KERNEL = 3
CHANNEL_GROUPS = {
    "fsr_only": tuple(range(4)),
    "imu_only": tuple(range(4, 10)),
    "fusion": tuple(range(10)),
}
FUSION_CHANNEL_NAMES = (
    "foot_force_1",
    "foot_force_2",
    "foot_force_3",
    "foot_force_4",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "ChannelNormalizer":
        if train.ndim != 3 or train.shape[1] <= 0:
            raise ValueError(f"expected (N, T, C) with T>0, got {train.shape}")
        if len(train) == 0 or not np.all(np.isfinite(train)):
            raise ValueError("normalizer training tensor must be non-empty and finite")
        mean = train.astype(np.float64).mean(axis=(0, 1))
        std = train.astype(np.float64).std(axis=(0, 1))
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 3 or values.shape[2] != len(self.mean):
            raise ValueError("normalizer channel count does not match tensor")
        normalized = (values.astype(np.float32) - self.mean) / self.std
        if not np.all(np.isfinite(normalized)):
            raise ValueError("normalization produced NaN/Inf")
        return normalized.astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass(frozen=True)
class ModelResourceEstimate:
    channels: int
    parameters: int
    float_parameter_bytes: int
    int8_parameter_payload_bytes: int
    float_activation_working_set_bytes: int
    int8_activation_working_set_bytes: int


def estimate_model_resources(channels: int, aggregation: str = "gap", time_steps: int = TIME_STEPS) -> ModelResourceEstimate:
    if channels <= 0 or time_steps <= 0:
        raise ValueError("channels and time_steps must be positive")
    conv1_weights = CONV1_KERNEL * channels * CONV1_FILTERS
    conv1_bias = CONV1_FILTERS
    conv2_weights = CONV2_KERNEL * CONV1_FILTERS * CONV2_FILTERS
    conv2_bias = CONV2_FILTERS
    if aggregation not in {"gap", "last_step", "recent", "global_recent"}:
        raise ValueError(f"unsupported temporal aggregation: {aggregation}")
    dense_inputs = CONV2_FILTERS * (2 if aggregation == "global_recent" else 1)
    dense_weights = dense_inputs * CLASS_COUNT
    dense_bias = CLASS_COUNT
    parameters = (
        conv1_weights
        + conv1_bias
        + conv2_weights
        + conv2_bias
        + dense_weights
        + dense_bias
    )
    int8_parameter_payload = (
        conv1_weights
        + 4 * conv1_bias
        + conv2_weights
        + 4 * conv2_bias
        + dense_weights
        + 4 * dense_bias
    )
    activation_elements = max(
        time_steps * channels + time_steps * CONV1_FILTERS,
        time_steps * CONV1_FILTERS + time_steps * CONV2_FILTERS,
        time_steps * CONV2_FILTERS + dense_inputs,
        dense_inputs + CLASS_COUNT,
    )
    return ModelResourceEstimate(
        channels=channels,
        parameters=parameters,
        float_parameter_bytes=4 * parameters,
        int8_parameter_payload_bytes=int8_parameter_payload,
        float_activation_working_set_bytes=4 * activation_elements,
        int8_activation_working_set_bytes=activation_elements,
    )


def build_compact_1d_cnn(
    channels: int, seed: int = 20260807, aggregation: str = "gap", recent_k: int = 8, time_steps: int = TIME_STEPS, causal: bool = False,
):
    """Build the compact causal-window CNN with an explicit temporal aggregator."""
    if channels <= 0 or time_steps <= 0:
        raise ValueError("channels and time_steps must be positive")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised in the simulation-only venv
        raise RuntimeError(
            "TensorFlow is required for CNN training; install requirements-cnn.txt"
        ) from exc
    if aggregation not in {"gap", "last_step", "recent", "global_recent"}:
        raise ValueError(f"unsupported temporal aggregation: {aggregation}")
    if aggregation in {"recent", "global_recent"} and not 1 <= recent_k <= time_steps:
        raise ValueError(f"recent_k must be in [1, {time_steps}]")
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(time_steps, channels), name="terrain_window")
    padding = "causal" if causal else "same"
    values = tf.keras.layers.Conv1D(
        CONV1_FILTERS, CONV1_KERNEL, padding=padding, activation="relu", name="conv1"
    )(inputs)
    values = tf.keras.layers.Conv1D(
        CONV2_FILTERS, CONV2_KERNEL, padding=padding, activation="relu", name="conv2"
    )(values)
    if aggregation == "gap":
        values = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(values)
    elif aggregation == "last_step":
        values = tf.keras.layers.Cropping1D((time_steps - 1, 0), name="endpoint_only")(values)
        values = tf.keras.layers.Flatten(name="endpoint_feature")(values)
    elif aggregation == "recent":
        values = tf.keras.layers.Cropping1D((time_steps - recent_k, 0), name=f"recent_{recent_k}_ms")(values)
        values = tf.keras.layers.GlobalAveragePooling1D(name="recent_average_pool")(values)
    else:
        global_values = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(values)
        recent_values = tf.keras.layers.Cropping1D((time_steps - recent_k, 0), name=f"recent_{recent_k}_ms")(values)
        recent_values = tf.keras.layers.GlobalAveragePooling1D(name="recent_average_pool")(recent_values)
        values = tf.keras.layers.Concatenate(name="global_recent_concat")([global_values, recent_values])
    outputs = tf.keras.layers.Dense(CLASS_COUNT, activation="softmax", name="class_scores")(
        values
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"terrain_cnn_{'causal_' if causal else ''}{aggregation}_c{channels}")


def estimate_model_macs(channels: int, time_steps: int = TIME_STEPS) -> int:
    """Multiply-accumulates for one `(50, channels)` inference, excluding pooling."""
    if channels <= 0 or time_steps <= 0:
        raise ValueError("channels and time_steps must be positive")
    return (
        time_steps * CONV1_KERNEL * channels * CONV1_FILTERS
        + time_steps * CONV2_KERNEL * CONV1_FILTERS * CONV2_FILTERS
        + CONV2_FILTERS * CLASS_COUNT
    )


def evaluation_rows(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    terrain_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if probabilities.shape != (len(y_true), len(terrain_names)):
        raise ValueError("probability shape does not match labels/classes")
    prediction = np.argmax(probabilities, axis=1)
    labels = np.arange(len(terrain_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=labels, zero_division=0
    )
    rows: list[dict[str, object]] = [
        {
            "class": "overall",
            "accuracy": float(np.mean(prediction == y_true)),
            "precision": "",
            "recall": "",
            "f1": float(np.mean(f1)),
            "support": int(len(y_true)),
        }
    ]
    for index, name in enumerate(terrain_names):
        rows.append(
            {
                "class": name,
                "accuracy": "",
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    matrix = confusion_matrix(y_true, prediction, labels=labels)
    matrix_rows = [
        {
            "actual": terrain_names[actual],
            "predicted": terrain_names[predicted],
            "count": int(matrix[actual, predicted]),
        }
        for actual in labels
        for predicted in labels
    ]
    return rows, matrix_rows


def mutual_pair_confusion(
    y_true: np.ndarray,
    prediction: np.ndarray,
    left_label: int,
    right_label: int,
) -> tuple[int, int, float]:
    if y_true.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    pair_mask = (y_true == left_label) | (y_true == right_label)
    support = int(np.sum(pair_mask))
    count = int(
        np.sum((y_true == left_label) & (prediction == right_label))
        + np.sum((y_true == right_label) & (prediction == left_label))
    )
    return count, support, count / max(support, 1)
