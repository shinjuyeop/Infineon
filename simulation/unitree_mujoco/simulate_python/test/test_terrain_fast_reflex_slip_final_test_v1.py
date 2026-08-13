from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from terrain_fast_reflex_slip_final_test_v1 import (  # noqa: E402
    FROZEN_THRESHOLD, evaluate_frozen_scores, validate_test_ownership,
)
from test_terrain_fast_reflex_detector_v1 import make_trace  # noqa: E402
from terrain_fast_reflex_v1 import TRACE_PRE_MS  # noqa: E402


def test_trace(run_id: str, family: str):
    trace = make_trace(run_id, "marble_to_ice")
    trace.metadata.update({"split": "test", "surface_family": family})
    return trace


class FrozenSlipFinalTestV1Test(unittest.TestCase):
    def test_rejects_non_test_input(self) -> None:
        trace = test_trace("x", "warped_multisine")
        trace.metadata["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "non-test"):
            validate_test_ownership([trace])

    def test_rejects_incomplete_family_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "ownership mismatch"):
            validate_test_ownership([test_trace("x", "warped_multisine")])

    def test_metrics_use_frozen_policy_and_post_onset_detection(self) -> None:
        target = test_trace("target", "warped_multisine")
        target.slip[TRACE_PRE_MS + 10:] = True
        clean = test_trace("clean", "smooth_random_patches")
        scores = np.zeros(200)
        scores[2:5] = FROZEN_THRESHOLD
        scores[12:15] = FROZEN_THRESHOLD
        scores[100 + 20:100 + 23] = FROZEN_THRESHOLD
        result = evaluate_frozen_scores([target, clean], scores)
        run = result.metrics["run_level"]
        self.assertEqual(run["run_target_detected"], 1)
        self.assertEqual(run["pre_onset_false_alarm_runs"], 2)
        self.assertEqual(run["median_hazard_onset_to_stable_detection_ms"], 4.0)
        self.assertEqual(run["anticipation_firing_runs"], 1)
        self.assertEqual(run["anticipation_firing_count"], 1)


if __name__ == "__main__":
    unittest.main()
