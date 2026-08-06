"""Unit checks for pure single-window synthesis."""

from __future__ import annotations

import numpy as np

from infineon_hil.terrain.features import extract_features
from infineon_hil.terrain.generator import generate_window


def test_class_mapping(config) -> None:
    assert {name: value.class_id for name, value in config.terrains.items()} == {
        "concrete": 0,
        "marble": 1,
        "ice": 2,
        "sand": 3,
    }


def test_single_window_shape_range_and_finiteness(config) -> None:
    for index, name in enumerate(config.terrains):
        result = generate_window(name, config, np.random.default_rng(index))
        assert result.waveform.shape == (50, 5)
        assert result.waveform.dtype == np.float32
        assert np.isfinite(result.waveform).all()
        assert np.all((result.waveform[:, :4] >= 0.0) & (result.waveform[:, :4] <= 200.0))
        assert result.metadata["class_id"] == config.terrains[name].class_id


def test_seed_reproducibility_and_variation(config) -> None:
    first = generate_window("ice", config, np.random.default_rng(123)).waveform
    repeated = generate_window("ice", config, np.random.default_rng(123)).waveform
    different = generate_window("ice", config, np.random.default_rng(124)).waveform
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_feature_extraction_is_finite(config) -> None:
    window = generate_window("sand", config, np.random.default_rng(55)).waveform
    features = extract_features(window, config.sampling_rate_hz)
    assert features.shape == (1, 4)
    assert np.isfinite(features).all()
    assert np.all(features >= 0)

