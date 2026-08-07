"""Tests for Dataset v1 train-only calibration and strict INT8 export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

from terrain_cnn import build_compact_1d_cnn  # noqa: E402
from terrain_int8 import (  # noqa: E402
    TensorQuantization,
    dequantize,
    export_full_int8_tflite,
    normalize,
    parity_gate,
    quantize,
    select_calibration_indices,
)


class TerrainInt8Test(unittest.TestCase):
    def _metadata(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows: list[tuple[str, int, str]] = []
        for split, family_names in (
            ("train", ("train_a", "train_b", "train_c")),
            ("validation", ("val_a", "val_b")),
            ("test", ("test_a", "test_b")),
        ):
            for family in family_names:
                for label in range(4):
                    for _ in range(5):
                        rows.append((split, label, family))
        split, labels, families = zip(*rows)
        return np.asarray(split), np.asarray(labels), np.asarray(families)

    def test_calibration_is_balanced_deterministic_and_train_only(self) -> None:
        split, labels, families = self._metadata()
        first = select_calibration_indices(split, labels, families, 24, seed=9)
        second = select_calibration_indices(split, labels, families, 24, seed=9)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(split[first].tolist()), {"train"})
        self.assertEqual({int(label): int(np.sum(labels[first] == label)) for label in range(4)}, {0: 6, 1: 6, 2: 6, 3: 6})
        self.assertEqual(
            {family: int(np.sum(families[first] == family)) for family in ("train_a", "train_b", "train_c")},
            {"train_a": 8, "train_b": 8, "train_c": 8},
        )

    def test_calibration_rejects_family_leakage(self) -> None:
        split, labels, families = self._metadata()
        families[-1] = "train_a"
        with self.assertRaises(ValueError):
            select_calibration_indices(split, labels, families, 12, seed=1)

    def test_normalization_and_quantization_are_bounded(self) -> None:
        values = np.ones((2, 50, 10), dtype=np.float32)
        normalized = normalize(values, np.ones(10), np.full(10, 2.0))
        np.testing.assert_array_equal(normalized, np.zeros_like(values))
        spec = TensorQuantization((1, 50, 10), "int8", 0.05, -3)
        source = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
        restored = dequantize(quantize(source, spec), spec)
        self.assertLessEqual(float(np.max(np.abs(source - restored))), spec.scale / 2 + 1e-6)

    def test_parity_gate_thresholds(self) -> None:
        passed = parity_gate(0.99, 0.985, 0.99, 0.984, 0.015, 0.020)
        self.assertTrue(passed["passed"])
        failed = parity_gate(0.99, 0.97, 0.99, 0.97, 0.015, 0.030)
        self.assertFalse(failed["passed"])

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is an optional CNN dependency")
    def test_export_has_strict_fusion10_interface(self) -> None:
        model = build_compact_1d_cnn(10, seed=3)
        representative = np.random.default_rng(3).normal(size=(16, 50, 10)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fusion_int8.tflite"
            input_spec, output_spec = export_full_int8_tflite(model, representative, output)
            self.assertTrue(output.is_file())
            self.assertEqual(input_spec.shape, (1, 50, 10))
            self.assertEqual(input_spec.dtype, "int8")
            self.assertEqual(output_spec.shape, (1, 4))
            self.assertEqual(output_spec.dtype, "int8")


if __name__ == "__main__":
    unittest.main()
