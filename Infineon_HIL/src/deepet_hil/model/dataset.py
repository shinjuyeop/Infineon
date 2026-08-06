"""Dataset loading, deterministic splitting, and train-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from deepet_hil.schema import NUM_CHANNELS
from deepet_hil.terrain.dataset import stratified_split_indices


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel standardization fitted only on training samples."""

    mean: NDArray[np.float32]
    scale: NDArray[np.float32]

    @classmethod
    def fit(cls, x: NDArray[np.floating]) -> "ChannelStandardizer":
        mean = np.mean(x, axis=(0, 1), dtype=np.float64)
        scale = np.std(x, axis=(0, 1), dtype=np.float64)
        scale = np.maximum(scale, 1e-6)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class PreparedDataset:
    """Standardized full tensor plus non-overlapping index partitions."""

    x: NDArray[np.float32]
    y: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    standardizer: ChannelStandardizer


def load_npz_dataset(path: str | Path) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load and structurally validate an `(N, time, 5)` terrain dataset."""

    with np.load(Path(path)) as data:
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
    if x.ndim != 3 or x.shape[-1] != NUM_CHANNELS or y.shape != (x.shape[0],):
        raise ValueError(f"Expected X=(N, time, {NUM_CHANNELS}) and y=(N,)")
    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN or Inf")
    return x, y


def prepare_dataset(path: str | Path, seed: int = 42) -> PreparedDataset:
    """Load, split 70/15/15, and standardize using training statistics only."""

    x, y = load_npz_dataset(path)
    splits = stratified_split_indices(y, seed=seed)
    standardizer = ChannelStandardizer.fit(x[splits["train"]])
    return PreparedDataset(
        x=standardizer.transform(x),
        y=y,
        train_indices=splits["train"],
        validation_indices=splits["validation"],
        test_indices=splits["test"],
        standardizer=standardizer,
    )

