import unittest
import numpy as np

from run_terrain_transition_aware_v2 import LABEL, stable


class TransitionAwareV2Test(unittest.TestCase):
    def test_endpoint_label_semantics(self):
        before, after = LABEL['marble'], LABEL['ice']
        self.assertEqual(before, 1)
        self.assertEqual(after, 2)  # endpoint, not the majority of its window

    def test_t1_is_third_target_endpoint(self):
        self.assertEqual(stable(np.array([-1] * 650 + [2, 2, 2]), 2), 652)


if __name__ == '__main__':
    unittest.main()
