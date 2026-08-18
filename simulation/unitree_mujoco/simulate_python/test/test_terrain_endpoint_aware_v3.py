"""Non-simulator invariants for the Terrain v3 architecture ablation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_terrain_endpoint_aware_v3 import architecture_gates, frozen_fit_indices  # noqa: E402


class EndpointAwareV3Test(unittest.TestCase):
    def test_frozen_mixture_uses_train_only_static_and_transition_windows(self):
        # First 12 rows are static; only 0..7 and 12..23 belong to train.
        y = np.tile(np.arange(4), 6)
        split = np.asarray(["train"] * 8 + ["validation"] * 4 + ["train"] * 12)
        indices, mixture = frozen_fit_indices(y, split, static_count=12)
        self.assertTrue(np.all(split[indices] == "train"))
        self.assertTrue(np.all((indices < 12) | (indices >= 12)))
        self.assertEqual(mixture["effective_static_windows"], 8)
        self.assertFalse(mixture["global_inverse_frequency_weighting"])

    def test_transition_gate_requires_case_c_occupancy_not_detection_alone(self):
        metrics = {case: {"detection_rate": 1.0, "occupancy_mean": .9} for case in "ABCD"}
        self.assertTrue(all(architecture_gates(metrics).values()))
        metrics["C"] = {"detection_rate": 1.0, "occupancy_mean": .799}
        self.assertFalse(architecture_gates(metrics)["C"])


if __name__ == "__main__":
    unittest.main()
