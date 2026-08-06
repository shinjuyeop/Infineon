"""Dataset-level shape, balance, probability, and split checks."""

from __future__ import annotations

import numpy as np

from infineon_hil.terrain.dataset import build_dataset, stratified_split_indices
from infineon_hil.terrain.validation import run_sanity_baseline


def test_full_dataset_shape_balance_and_range(full_dataset) -> None:
    x, y, metadata = full_dataset
    assert x.shape == (2000, 50, 5)
    assert y.shape == (2000,)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert np.all((x[:, :, :4] >= 0.0) & (x[:, :, :4] <= 200.0))
    np.testing.assert_array_equal(np.bincount(y), [500, 500, 500, 500])
    assert metadata.groupby("terrain_name").size().to_dict() == {
        "concrete": 500,
        "ice": 500,
        "marble": 500,
        "sand": 500,
    }


def test_dataset_is_deterministic(config, full_dataset) -> None:
    x, y, metadata = full_dataset
    x_again, y_again, metadata_again = build_dataset(config, 500, 42)
    np.testing.assert_array_equal(x, x_again)
    np.testing.assert_array_equal(y, y_again)
    assert metadata.equals(metadata_again)


def test_micro_slip_rates_and_vibration_rms(config, full_dataset) -> None:
    _, _, metadata = full_dataset
    for name, terrain in config.terrains.items():
        rows = metadata[metadata["terrain_name"] == name]
        rate = float(rows["micro_slip_occurred"].mean())
        # Binomial sampling tolerance; intentionally not tuned to exact counts.
        tolerance = max(0.025, 4.0 * np.sqrt(terrain.micro_slip_probability * (1.0 - terrain.micro_slip_probability) / len(rows)))
        assert abs(rate - terrain.micro_slip_probability) <= tolerance
        mean_rms = float(rows["actual_vibration_rms_g"].mean())
        assert 0.75 * terrain.vibration_rms_g <= mean_rms <= 1.75 * terrain.vibration_rms_g


def test_stratified_split_uses_existing_samples(full_dataset) -> None:
    _, y, _ = full_dataset
    splits = stratified_split_indices(y, seed=42)
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 1400,
        "validation": 300,
        "test": 300,
    }
    combined = np.concatenate(list(splits.values()))
    np.testing.assert_array_equal(np.sort(combined), np.arange(2000))
    for indices in splits.values():
        assert np.all(np.bincount(y[indices], minlength=4) == len(indices) // 4)


def test_baseline_returns_complete_metrics(full_dataset) -> None:
    x, y, _ = full_dataset
    result = run_sanity_baseline(x, y, seed=42)
    assert result.test_samples == 500
    assert result.confusion_matrix.shape == (4, 4)
    assert result.recalls.shape == (4,)
    assert 0.0 <= result.accuracy <= 1.0
