from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from terrain_fast_reflex_detector_v1 import (  # noqa: E402
    ChannelNormalizer, balanced_run_weights, build_model, make_windows,
    resource_estimate, select_threshold,
)
from terrain_fast_reflex_v1 import (  # noqa: E402
    ORACLE_CHANNELS, ORACLE_INDEX, RELATIVE_TRANSITION_TIME_MS, TRACE_PRE_MS,
    TRACE_SAMPLES, FastReflexTrace,
)


def make_trace(run_id: str = "run", scenario: str = "marble_to_marble") -> FastReflexTrace:
    timestamps = 0.25 + RELATIVE_TRANSITION_TIME_MS.astype(np.float64) / 1000.0
    oracle = np.zeros((TRACE_SAMPLES, len(ORACLE_CHANNELS)), dtype=np.float64)
    oracle[:, ORACLE_INDEX["left_contact"]] = 1.0
    oracle[:, ORACLE_INDEX["contact_normal_force_N"]] = 100.0
    return FastReflexTrace(
        metadata={"run_id": run_id, "scenario": scenario, "expected_hazard": "normal",
                  "transition_time_s": 0.25},
        timestamps_s=timestamps, sensors=np.zeros((TRACE_SAMPLES, 10), dtype=np.float64),
        oracle=oracle, slip=np.zeros(TRACE_SAMPLES, dtype=bool),
        sink=np.zeros(TRACE_SAMPLES, dtype=bool), tilt=np.zeros(TRACE_SAMPLES, dtype=bool),
        valid=True, invalid_reason="",
    )


class FastReflexDetectorV1Test(unittest.TestCase):
    def test_endpoint_label_does_not_leak_future_hazard(self) -> None:
        trace = make_trace("ice", "marble_to_ice")
        trace.sensors[:] = np.arange(150)[:, None]
        trace.slip[TRACE_PRE_MS + 11 :] = True
        data = make_windows([trace], "slip", 10)
        self.assertEqual(data.x.shape, (100, 10, 10))
        self.assertEqual(data.y[:11].tolist(), [0] * 11)
        self.assertEqual(data.y[11], 1)
        np.testing.assert_array_equal(data.x[10, :, 0], np.arange(51, 61))

    def test_other_hazard_is_canonical_non_target_negative(self) -> None:
        trace = make_trace("sand", "marble_to_sand")
        trace.tilt[TRACE_PRE_MS:] = True
        slip = make_windows([trace], "slip", 5)
        self.assertFalse(np.any(slip.y))
        self.assertFalse(np.any(slip.clean_negative))
        sink_tilt = make_windows([trace], "sink_tilt", 5)
        self.assertTrue(np.all(sink_tilt.y))

    def test_train_only_normalizer_and_run_balancing(self) -> None:
        left, right = make_trace("left"), make_trace("right")
        left.sensors[:] = 1.0; right.sensors[:] = 3.0
        left.slip[TRACE_PRE_MS + 50 :] = True
        right.slip[TRACE_PRE_MS + 50 :] = True
        data = make_windows([left, right], "slip", 5, stride_ms=2)
        normalizer = ChannelNormalizer.fit(data.x)
        self.assertAlmostEqual(float(normalizer.mean[0]), 2.0)
        weights = balanced_run_weights(data)
        for label in (0, 1):
            self.assertAlmostEqual(float(weights[data.y == label].sum()), len(weights) / 2, places=4)
        self.assertAlmostEqual(float(weights[data.run_ids == "left"].sum()),
                               float(weights[data.run_ids == "right"].sum()), places=4)

    def test_validation_threshold_obeys_fpr_constraint(self) -> None:
        y = np.asarray([0] * 20 + [1] * 4)
        scores = np.asarray(list(np.linspace(0.0, 0.9, 20)) + [0.4, 0.6, 0.8, 1.0])
        threshold = select_threshold(y, scores, maximum_fpr=0.05)
        self.assertLessEqual(float(np.mean(scores[y == 0] >= threshold)), 0.05)

    def test_binary_model_resource_count(self) -> None:
        self.assertEqual(resource_estimate(5)["parameters"], 1221)
        self.assertEqual(resource_estimate(50)["conv_dense_macs_per_inference"], 58816)
        self.assertEqual(resource_estimate(5, "average_max")["parameters"], 1237)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow optional")
    def test_binary_model_shape_and_parameters(self) -> None:
        for pooling, parameters in (("average", 1221), ("max", 1221), ("average_max", 1237)):
            model = build_model(10, seed=1, pooling=pooling)
            self.assertEqual(model.input_shape, (None, 10, 10))
            self.assertEqual(model.output_shape, (None, 1))
            self.assertEqual(model.count_params(), parameters)


if __name__ == "__main__":
    unittest.main()
