"""Fast CNN shape, training, and artifact round-trip tests."""

from __future__ import annotations

import numpy as np

from infineon_hil.model.dataset import ChannelStandardizer
from infineon_hil.model.inference import TerrainInferenceEngine
from infineon_hil.model.network import TerrainCNN
from infineon_hil.model.train import TrainingConfig, train_model


def test_cnn_input_output_shape() -> None:
    model = TerrainCNN(seed=7)
    x = np.zeros((3, 50, 5), dtype=np.float32)
    logits = model.forward(x)
    probabilities = model.predict_proba(x)
    assert logits.shape == (3, 4)
    assert probabilities.shape == (3, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_training_smoke_and_model_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(24, 50, 5)).astype(np.float32)
    y = np.tile(np.arange(4), 6).astype(np.int64)
    standardizer = ChannelStandardizer.fit(x[:16])
    standardized = standardizer.transform(x)
    model = TerrainCNN(seed=3)
    history = train_model(
        model,
        standardized,
        y,
        np.arange(16),
        np.arange(16, 24),
        TrainingConfig(epochs=1, batch_size=8, learning_rate=0.001, seed=3),
        verbose=False,
    )
    assert len(history) == 1
    assert np.isfinite(history[0].loss)
    artifact = tmp_path / "model.npz"
    names = ("concrete", "marble", "ice", "sand")
    model.save(
        artifact,
        channel_mean=standardizer.mean,
        channel_scale=standardizer.scale,
        class_names=names,
    )
    engine = TerrainInferenceEngine.load(artifact)
    prediction = engine.predict(x[0])
    assert prediction.probabilities.shape == (4,)
    assert prediction.class_name in names

