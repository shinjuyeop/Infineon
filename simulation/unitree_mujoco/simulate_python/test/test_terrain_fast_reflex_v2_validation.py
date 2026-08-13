import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_terrain_fast_reflex_v2_validation import replay, select_candidate, stable_starts


def data(y, run_id=None, family=None, mode=None):
    n = len(y)
    return {"y": np.asarray(y), "run_id": np.asarray(run_id or ["r"] * n),
            "endpoint_ms": np.arange(n), "family": np.asarray(family or ["crosshatch"] * n),
            "mode": np.asarray(mode or ["normal_sand"] * n)}


class V2ValidationTest(unittest.TestCase):
    def test_persistence_and_causal_post_onset_policy(self):
        self.assertEqual(stable_starts(np.array([.9, .9, .1, .9, .9, .9]), .5, 3), [3])
        rows, metrics = replay(data([0, 0, 1, 1, 1]), np.array([.9, .9, .2, .9, .9]), .5, 2)
        self.assertTrue(rows[0]["pre_onset_firing"]); self.assertTrue(rows[0]["post_onset_detection"])
        self.assertEqual(metrics["pre_onset_false_alarm_rate"], 1.0); self.assertEqual(metrics["latency_median_ms"], 1.0)

    def test_negative_only_fpr_and_anticipation_are_separate(self):
        d = data([0, 0, 0, 0, 0, 1], ["negative", "negative", "negative", "positive", "positive", "positive"])
        rows, metrics = replay(d, np.array([.9, .9, .1, .9, .9, .1]), .5, 2)
        self.assertEqual(metrics["negative_only_run_fpr"], 1.0); self.assertEqual(metrics["pre_onset_false_alarm_rate"], 1.0)
        self.assertEqual(metrics["overall_causal_run_fpr"], 1.0); self.assertEqual(metrics["anticipation_runs"], 1)
        self.assertEqual(metrics["anticipation_lead_max_ms"], 2); self.assertEqual({r["negative_only"] for r in rows}, {True, False})

    def test_selection_tie_breaks_latency_then_window(self):
        base = {"overall_causal_run_fpr": .04, "run_recall": .8, "threshold": .7}
        slower = {**base, "window_ms": 5, "latency_p95_ms": 9}; faster_long = {**base, "window_ms": 10, "latency_p95_ms": 5}; faster_short = {**base, "window_ms": 5, "latency_p95_ms": 5}
        self.assertEqual(select_candidate([slower, faster_long, faster_short]), faster_short)
        self.assertIsNone(select_candidate([{**base, "overall_causal_run_fpr": .051, "window_ms": 5, "latency_p95_ms": 1}]))
