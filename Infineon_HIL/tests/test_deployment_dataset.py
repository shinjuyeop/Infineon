"""TensorFlow-independent deployment preprocessing and metadata tests."""

from __future__ import annotations

import numpy as np

from infineon_hil.deployment.dataset import prepare_deployment_dataset
from infineon_hil.deployment.metadata import DeploymentMetadata
from infineon_hil.schema import CHANNEL_NAMES
from infineon_hil.terrain.dataset import save_dataset


def test_deployment_split_and_normalizer_are_leak_free(tmp_path, full_dataset) -> None:
    x, y, frame = full_dataset
    dataset_path = tmp_path / "dataset.npz"
    save_dataset(x, y, frame, dataset_path, tmp_path / "metadata.csv")
    dataset = prepare_deployment_dataset(dataset_path, seed=42)
    assert (len(dataset.y_train), len(dataset.y_validation), len(dataset.y_test)) == (1400, 300, 300)
    assert np.isfinite(dataset.prepared.x).all()
    expected_mean = x[dataset.prepared.train_indices].mean(axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(dataset.prepared.standardizer.mean, expected_mean, rtol=1e-5)
    train = set(dataset.prepared.train_indices.tolist())
    validation = set(dataset.prepared.validation_indices.tolist())
    test = set(dataset.prepared.test_indices.tolist())
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_metadata_round_trip_and_normalization(tmp_path) -> None:
    metadata = DeploymentMetadata(
        model_version="terrain_cnn_v0_1",
        channel_order=CHANNEL_NAMES,
        channel_mean=(1.0, 2.0, 3.0, 4.0, 5.0),
        channel_std=(1.0, 2.0, 3.0, 4.0, 5.0),
        class_names=("concrete", "marble", "ice", "sand"),
        input_shape=(50, 5),
        sampling_rate_hz=1000,
        window_length_ms=50,
        split_seed=42,
    )
    path = tmp_path / "metadata.json"
    metadata.save(path)
    loaded = DeploymentMetadata.load(path)
    assert loaded == metadata
    normalized = loaded.normalize(np.ones((2, 50, 5), dtype=np.float32))
    assert normalized.shape == (2, 50, 5)
    assert np.isfinite(normalized).all()
