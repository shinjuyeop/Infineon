"""Fast architecture checks for the optional TensorFlow deployment model."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from deepet_hil.deployment.model import build_deployment_model

pytestmark = pytest.mark.deployment


def test_deployment_model_architecture_and_probabilities() -> None:
    model = build_deployment_model(seed=7)
    output = model(np.zeros((3, 50, 5), dtype=np.float32), training=False).numpy()
    assert output.shape == (3, 4)
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    assert model.count_params() == 2580

    conv1 = model.get_layer("conv1")
    pool = model.get_layer("max_pool")
    conv2 = model.get_layer("conv2")
    gap = model.get_layer("global_average_pool")
    dense16 = model.get_layer("dense16")
    output_layer = model.get_layer("terrain_probabilities")
    assert isinstance(conv1, tf.keras.layers.Conv1D)
    assert conv1.filters == 16 and conv1.kernel_size == (5,) and conv1.padding == "same"
    assert isinstance(pool, tf.keras.layers.MaxPooling1D) and pool.pool_size == (2,)
    assert isinstance(conv2, tf.keras.layers.Conv1D)
    assert conv2.filters == 32 and conv2.kernel_size == (3,) and conv2.padding == "same"
    assert isinstance(gap, tf.keras.layers.GlobalAveragePooling1D)
    assert isinstance(dense16, tf.keras.layers.Dense) and dense16.units == 16
    assert isinstance(output_layer, tf.keras.layers.Dense) and output_layer.units == 4
    assert output_layer.activation.__name__ == "softmax"
