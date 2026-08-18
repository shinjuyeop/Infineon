import unittest

import numpy as np

from run_terrain_transition_ai_replay import stable_endpoint, sustained_endpoint


class TerrainTransitionAiReplayTest(unittest.TestCase):
    def test_stable_terrain_endpoint_is_third_causal_target(self):
        self.assertEqual(stable_endpoint(np.array([-1, 1, 2, 2, 2]), 2, 1), 4)

    def test_stable_terrain_returns_null_for_chatter(self):
        self.assertIsNone(stable_endpoint(np.array([1, 2, 1, 2, 1]), 2, 0))

    def test_reflex_persistence_uses_raw_int8_threshold(self):
        self.assertEqual(sustained_endpoint(np.array([120, 121, 121]), 121, 2, 0), 2)


if __name__ == "__main__":
    unittest.main()
