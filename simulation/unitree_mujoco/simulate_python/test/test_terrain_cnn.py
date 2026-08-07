"""Tests for the shared compact FSR/IMU/Fusion 1D-CNN design."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

from terrain_cnn import (  # noqa: E402
    CHANNEL_GROUPS,
    ChannelNormalizer,
    build_compact_1d_cnn,
    estimate_model_resources,
    evaluation_rows,
    mutual_pair_confusion,
)
from train_terrain_1d_cnn import validate_dataset_arrays  # noqa: E402


class TerrainCnnTest(unittest.TestCase):
    def test_channel_groups_are_exact_ablation(self) -> None:
        self.assertEqual(CHANNEL_GROUPS["fsr_only"], (0, 1, 2, 3))
        self.assertEqual(CHANNEL_GROUPS["imu_only"], (4, 5, 6, 7, 8, 9))
        self.assertEqual(CHANNEL_GROUPS["fusion"], tuple(range(10)))

    def test_normalizer_is_fitted_per_channel(self) -> None:
        train = np.zeros((3, 50, 2), dtype=np.float32)
        train[:, :, 0] = 2.0
        train[:, :, 1] = np.linspace(-1.0, 1.0, 50)
        normalizer = ChannelNormalizer.fit(train)
        normalized = normalizer.transform(train)
        np.testing.assert_allclose(normalized[:, :, 0], 0.0)
        self.assertAlmostEqual(float(normalized[:, :, 1].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:, :, 1].std()), 1.0, places=6)

    def test_resource_estimates_match_fixed_architecture(self) -> None:
        expected = {"fsr_only": 912, "imu_only": 1_032, "fusion": 1_272}
        for group, channels in CHANNEL_GROUPS.items():
            estimate = estimate_model_resources(len(channels))
            self.assertEqual(estimate.parameters, expected[group])
            self.assertLessEqual(estimate.float_activation_working_set_bytes, 5_600)
            self.assertLess(estimate.int8_parameter_payload_bytes, 1_400)

    def test_dataset_validation_rejects_family_leakage(self) -> None:
        x = np.zeros((12, 50, 10), dtype=np.float32)
        y = np.tile(np.arange(4), 3)
        split = np.repeat(np.asarray(["train", "validation", "test"]), 4)
        families = np.repeat(np.asarray(["a", "b", "c"]), 4)
        validate_dataset_arrays(x, y, split, families)
        families[-1] = "a"
        with self.assertRaises(ValueError):
            validate_dataset_arrays(x, y, split, families)

    def test_metrics_include_macro_f1_and_confusion(self) -> None:
        y = np.arange(4)
        probabilities = np.eye(4, dtype=np.float32)
        rows, matrix = evaluation_rows(y, probabilities, ("concrete", "marble", "ice", "sand"))
        self.assertEqual(rows[0]["accuracy"], 1.0)
        self.assertEqual(rows[0]["f1"], 1.0)
        self.assertEqual(len(matrix), 16)
        count, support, ratio = mutual_pair_confusion(
            np.asarray([0, 0, 1, 1, 2]), np.asarray([1, 0, 0, 1, 1]), 0, 1
        )
        self.assertEqual((count, support), (2, 4))
        self.assertEqual(ratio, 0.5)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is an optional CNN dependency")
    def test_keras_parameter_count_matches_estimate(self) -> None:
        for channels in (4, 6, 10):
            model = build_compact_1d_cnn(channels, seed=1)
            self.assertEqual(model.count_params(), estimate_model_resources(channels).parameters)


if __name__ == "__main__":
    unittest.main()
