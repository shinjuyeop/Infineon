"""Unit and smoke tests for the leakage-safe MuJoCo Dataset v1 pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

from terrain_dataset_v1 import (  # noqa: E402
    TERRAIN_LABELS,
    apply_sensor_imperfections,
    fit_and_evaluate,
    is_valid_run,
    make_run_specification,
    make_surface_parameters,
    statistical_features,
    validate_split_integrity,
)


class TerrainDatasetV1Test(unittest.TestCase):
    def test_surface_and_run_seeds_are_reproducible(self) -> None:
        self.assertEqual(make_surface_parameters("concrete", 123), make_surface_parameters("concrete", 123))
        self.assertEqual(make_run_specification("sand", 14, 2), make_run_specification("sand", 14, 2))

    def test_noise_is_paired_reproducible_and_non_mutating(self) -> None:
        clean = np.arange(500, dtype=np.float32).reshape(50, 10) / 10.0
        original = clean.copy()
        first = apply_sensor_imperfections(clean, 456)
        second = apply_sensor_imperfections(clean, 456)
        np.testing.assert_array_equal(clean, original)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (50, 10))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertFalse(np.array_equal(first, clean))

    def test_split_integrity_rejects_surface_leakage(self) -> None:
        rows = [
            {"valid_flag": 1, "terrain_class": "concrete", "split": "train", "surface_seed": 1, "session_id": "a", "run_group": "r1"},
            {"valid_flag": 1, "terrain_class": "concrete", "split": "test", "surface_seed": 1, "session_id": "b", "run_group": "r2"},
        ]
        with self.assertRaises(ValueError):
            validate_split_integrity(rows)

    def test_unseen_surface_design_and_class_balance(self) -> None:
        rows = []
        for terrain in TERRAIN_LABELS:
            for surface in range(15):
                spec = make_run_specification(terrain, surface, 0)
                rows.append({"valid_flag": 1, "terrain_class": terrain, "split": spec.split, "surface_seed": spec.surface_seed, "session_id": spec.session_id, "run_group": spec.run_group})
        validate_split_integrity(rows)
        for terrain in TERRAIN_LABELS:
            counts = {split: sum(row["terrain_class"] == terrain and row["split"] == split for row in rows) for split in ("train", "validation", "test")}
            self.assertEqual(counts, {"train": 9, "validation": 3, "test": 3})

    def test_feature_shape_and_baseline_training_smoke(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(48, 50, 10)).astype(np.float32)
        y = np.repeat(np.arange(4), 12)
        x += y[:, None, None]
        split = np.tile(np.asarray(["train"] * 8 + ["validation"] * 2 + ["test"] * 2), 4)
        features = statistical_features(x, tuple(range(10)))
        self.assertEqual(features.shape, (48, 60))
        metrics, matrix = fit_and_evaluate(x, y, split, tuple(range(10)), seed=1)
        self.assertEqual(len(metrics), 15)
        self.assertEqual(len(matrix), 16)

    def test_valid_run_filtering_reuses_collision_and_outlier_rules(self) -> None:
        valid = {"valid_run": 1, "body_collision": 0, "extreme_force_outlier": 0, "extreme_accel_outlier": 0}
        self.assertTrue(is_valid_run(valid))
        for field in valid:
            rejected = dict(valid)
            rejected[field] = 0 if field == "valid_run" else 1
            self.assertFalse(is_valid_run(rejected))


if __name__ == "__main__":
    unittest.main()
