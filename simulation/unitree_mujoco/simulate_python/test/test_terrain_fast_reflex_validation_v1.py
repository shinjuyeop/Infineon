from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from terrain_fast_reflex_validation_v1 import (  # noqa: E402
    ScoredRun, best_run_policy, run_policy_metrics, stable_endpoints,
    threshold_candidates,
)
from test_terrain_fast_reflex_detector_v1 import make_trace  # noqa: E402
from terrain_fast_reflex_v1 import TRACE_PRE_MS  # noqa: E402


class FastReflexValidationV1Test(unittest.TestCase):
    def test_stable_endpoint_is_confirmation_not_run_start(self) -> None:
        scores = np.asarray([0.9, 0.9, 0.9, 0.1, 0.9])
        np.testing.assert_array_equal(stable_endpoints(scores, 0.5, 3), [2])

    def test_pre_onset_firing_is_false_alarm_not_negative_latency(self) -> None:
        trace = make_trace("target", "marble_to_ice")
        trace.slip[TRACE_PRE_MS + 10:] = True
        scores = np.zeros(100); scores[5:8] = 1.0; scores[12:15] = 1.0
        row = run_policy_metrics([ScoredRun(trace, scores)], "slip", 0.5, 3)
        self.assertEqual(row["pre_onset_false_alarm_runs"], 1)
        self.assertEqual(row["early_warning_runs"], 1)
        self.assertEqual(row["run_target_detected"], 1)
        self.assertEqual(row["median_hazard_to_detection_ms"], 4.0)
        self.assertEqual(row["median_anticipation_lead_ms"], 3.0)

    def test_run_fpr_denominators_are_distinct(self) -> None:
        clean = make_trace("clean")
        target = make_trace("target", "marble_to_ice")
        target.slip[TRACE_PRE_MS + 10:] = True
        score = np.zeros(100); score[1:4] = 1.0
        row = run_policy_metrics(
            [ScoredRun(clean, score), ScoredRun(target, np.zeros(100))], "slip", 0.5, 3
        )
        self.assertEqual(row["run_pre_onset_fpr"], 0.5)
        self.assertEqual(row["hazard_free_run_fpr"], 1.0)
        self.assertEqual(row["target_negative_run_fpr"], 1.0)

    def test_threshold_grid_is_bounded_and_preserves_endpoint_value(self) -> None:
        candidates = threshold_candidates(np.linspace(0, 1, 10_000), 0.56789)
        self.assertLessEqual(len(candidates), 43)
        self.assertIn(0.56789, candidates)

    def test_best_policy_obeys_run_fpr_and_ties(self) -> None:
        rows = [
            {"run_target_recall": .95, "run_pre_onset_fpr": .06,
             "median_hazard_to_detection_ms": 1, "threshold": .5},
            {"run_target_recall": .90, "run_pre_onset_fpr": .05,
             "median_hazard_to_detection_ms": 4, "threshold": .5},
            {"run_target_recall": .90, "run_pre_onset_fpr": .04,
             "median_hazard_to_detection_ms": 5, "threshold": .6},
        ]
        self.assertIs(best_run_policy(rows), rows[2])


if __name__ == "__main__":
    unittest.main()
