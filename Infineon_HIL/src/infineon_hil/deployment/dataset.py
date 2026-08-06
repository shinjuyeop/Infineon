"""Deployment dataset adapter reusing the reference split and preprocessing policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from infineon_hil.model.dataset import PreparedDataset, prepare_dataset
from infineon_hil.schema import NUM_CHANNELS


@dataclass(frozen=True)
class DeploymentDataset:
    """Standardized tensors and non-overlapping partitions for deployment training."""

    prepared: PreparedDataset

    @property
    def x_train(self) -> NDArray[np.float32]:
        return self.prepared.x[self.prepared.train_indices]

    @property
    def y_train(self) -> NDArray[np.int64]:
        return self.prepared.y[self.prepared.train_indices]

    @property
    def x_validation(self) -> NDArray[np.float32]:
        return self.prepared.x[self.prepared.validation_indices]

    @property
    def y_validation(self) -> NDArray[np.int64]:
        return self.prepared.y[self.prepared.validation_indices]

    @property
    def x_test(self) -> NDArray[np.float32]:
        return self.prepared.x[self.prepared.test_indices]

    @property
    def y_test(self) -> NDArray[np.int64]:
        return self.prepared.y[self.prepared.test_indices]


def _assert_disjoint_partitions(dataset: PreparedDataset) -> None:
    partitions = (
        set(dataset.train_indices.tolist()),
        set(dataset.validation_indices.tolist()),
        set(dataset.test_indices.tolist()),
    )
    if any(partitions[left] & partitions[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, validation, and test partitions overlap")
    combined = np.concatenate(
        (dataset.train_indices, dataset.validation_indices, dataset.test_indices)
    )
    if len(combined) != len(dataset.y) or not np.array_equal(
        np.sort(combined), np.arange(len(dataset.y))
    ):
        raise ValueError("Dataset partitions do not cover each sample exactly once")


def prepare_deployment_dataset(
    path: str | Path,
    *,
    seed: int = 42,
    input_steps: int = 50,
) -> DeploymentDataset:
    """Load the canonical dataset using the shared 70/15/15 preprocessing path."""

    prepared = prepare_dataset(path, seed=seed)
    expected_shape = (input_steps, NUM_CHANNELS)
    if prepared.x.shape[1:] != expected_shape:
        raise ValueError(f"Expected sample shape {expected_shape}, got {prepared.x.shape[1:]}")
    _assert_disjoint_partitions(prepared)
    return DeploymentDataset(prepared)
