"""Dataset assembly, persistence, and deterministic stratified splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from infineon_hil.schema import NUM_CHANNELS

from .generator import generate_window
from .params import TerrainConfig


def build_dataset(
    config: TerrainConfig,
    samples_per_class: int = 500,
    seed: int = 42,
) -> tuple[NDArray[np.float32], NDArray[np.int64], pd.DataFrame]:
    """Build all classes using independent child RNG streams."""

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    terrains = sorted(config.terrains.values(), key=lambda item: item.class_id)
    total_samples = samples_per_class * len(terrains)
    seed_sequences = np.random.SeedSequence(seed).spawn(total_samples)
    x = np.empty((total_samples, config.time_steps, NUM_CHANNELS), dtype=np.float32)
    y = np.empty(total_samples, dtype=np.int64)
    records: list[dict[str, object]] = []
    index = 0
    for terrain in terrains:
        for _ in range(samples_per_class):
            child_seed = int(seed_sequences[index].generate_state(1, dtype=np.uint64)[0])
            generated = generate_window(
                terrain.name,
                config,
                np.random.default_rng(child_seed),
                random_seed=child_seed,
            )
            x[index] = generated.waveform
            y[index] = terrain.class_id
            records.append({"sample_id": index, **generated.metadata})
            index += 1
    return x, y, pd.DataFrame.from_records(records)


def save_dataset(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    metadata: pd.DataFrame,
    dataset_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Save sensor tensors/labels separately from descriptive ground truth."""

    npz_path = Path(dataset_path)
    csv_path = Path(metadata_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, X=x.astype(np.float32, copy=False), y=y)
    metadata.to_csv(csv_path, index=False)


def stratified_split_indices(
    y: NDArray[np.integer],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, NDArray[np.int64]]:
    """Split existing sample indices while preserving per-class proportions."""

    if not np.isclose(sum(ratios), 1.0) or any(value <= 0 for value in ratios):
        raise ValueError("Split ratios must be positive and sum to one")
    rng = np.random.default_rng(seed)
    split_lists: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for class_id in np.unique(y):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        train_end = round(len(indices) * ratios[0])
        validation_end = train_end + round(len(indices) * ratios[1])
        split_lists["train"].extend(indices[:train_end])
        split_lists["validation"].extend(indices[train_end:validation_end])
        split_lists["test"].extend(indices[validation_end:])
    return {
        name: np.asarray(rng.permutation(values), dtype=np.int64)
        for name, values in split_lists.items()
    }
