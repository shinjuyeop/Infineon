"""Schema-only regression tests for Terrain Transition v1."""

import unittest

import numpy as np

import run_terrain_transition as transition


class TerrainTransitionTest(unittest.TestCase):
    def test_four_case_mapping_is_bidirectional_and_marble_is_hard(self):
        self.assertEqual(transition.CASES["A"]["before"], "marble")
        self.assertEqual(transition.CASES["A"]["after"], "ice")
        self.assertEqual(transition.CASES["B"]["after"], "sand")
        self.assertEqual(transition.CASES["C"]["after"], "marble")
        self.assertEqual(transition.CASES["D"]["after"], "marble")

    def test_frozen_oracle_requires_physical_conditions_not_terrain_name(self):
        oracle = np.zeros((8, len(transition.V2_ORACLE_CHANNELS)), dtype=float)
        labels = transition.label(oracle, 4)
        self.assertFalse(any(value.any() for value in labels.values()))

    def test_sustained_is_causal(self):
        result = transition._sustained(np.array([False, True, True, True, False]), 3)
        np.testing.assert_array_equal(result, [False, True, True, True, False])


if __name__ == "__main__":
    unittest.main()
