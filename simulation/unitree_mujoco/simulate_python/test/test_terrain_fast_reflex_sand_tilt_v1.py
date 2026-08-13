import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrain_fast_reflex_sand_tilt_v1 import (
    FEATURE_NAMES, assert_validation_only, physical_features, rank_auc, stable_endpoints,
)


class SandTiltValidationTest(unittest.TestCase):
    def test_split_safeguard_fails_closed(self):
        rows = [
            {"split": "train", "surface_family": family}
            for family in ("multisine", "filtered_random", "sparse_aggregate")
        ] + [
            {"split": "validation", "surface_family": family}
            for family in ("crosshatch", "rounded_ridges")
        ] + [{"split": "test", "surface_family": "warped_multisine"}]
        self.assertEqual(assert_validation_only(rows, {"validation"}), [3, 4])
        with self.assertRaises(ValueError): assert_validation_only(rows, {"test"})

    def test_physical_feature_mapping_and_derivative(self):
        sensors = np.zeros((3, 10))
        sensors[0, :4] = [1, 2, 3, 4]
        sensors[2, :4] = [2, 3, 7, 8]
        sensors[2, 7:9] = [3, 4]
        features = physical_features(sensors, 3)
        last = dict(zip(FEATURE_NAMES, features[-1]))
        self.assertAlmostEqual(last["front_minus_rear"], 10)
        self.assertAlmostEqual(last["left_minus_right"], -2)
        self.assertAlmostEqual(last["gyro_xy_magnitude"], 5)
        self.assertAlmostEqual(last["d_fsr_sum"], 5)

    def test_metrics_and_persistence(self):
        np.testing.assert_array_equal(stable_endpoints(np.array([.9, .9, .1, .9]), .5, 2), [1])
        self.assertEqual(rank_auc(np.array([0, 0, 1, 1]), np.array([0, 1, 2, 3])), 1.0)


if __name__ == "__main__":
    unittest.main()
