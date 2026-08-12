from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from terrain_fast_reflex_v1 import (  # noqa: E402
    ORACLE_CHANNELS,
    ORACLE_INDEX,
    RELATIVE_TRANSITION_TIME_MS,
    TRACE_PRE_MS,
    TRACE_SAMPLES,
    FastReflexTrace,
    calibrate_hazard_thresholds,
    extract_prefix,
    label_trace,
    onset_time_s,
    validate_split_integrity,
    validate_trace,
)
from run_terrain_fast_reflex_v1 import _onset_fields  # noqa: E402


def make_trace(run_id: str = "run", scenario: str = "marble_to_marble") -> FastReflexTrace:
    timestamps = 0.25 + RELATIVE_TRANSITION_TIME_MS.astype(np.float64) / 1000.0
    oracle = np.zeros((TRACE_SAMPLES, len(ORACLE_CHANNELS)), dtype=np.float64)
    oracle[:, ORACLE_INDEX["left_contact"]] = 1.0
    oracle[:, ORACLE_INDEX["contact_normal_force_N"]] = 100.0
    return FastReflexTrace(
        metadata={
            "run_id": run_id,
            "scenario": scenario,
            "expected_hazard": "normal",
            "transition_time_s": 0.25,
        },
        timestamps_s=timestamps,
        sensors=np.zeros((TRACE_SAMPLES, 10), dtype=np.float64),
        oracle=oracle,
        slip=np.zeros(TRACE_SAMPLES, dtype=bool),
        sink=np.zeros(TRACE_SAMPLES, dtype=bool),
        tilt=np.zeros(TRACE_SAMPLES, dtype=bool),
        valid=True,
        invalid_reason="",
    )


class TerrainFastReflexV1Test(unittest.TestCase):
    def test_native_trace_and_prefixes_are_exact(self) -> None:
        trace = make_trace()
        trace.sensors[:] = np.arange(TRACE_SAMPLES)[:, None]
        validate_trace(trace)
        timestamps, prefix = extract_prefix(trace, 20)
        self.assertEqual(prefix.shape, (20, 10))
        self.assertAlmostEqual(float(timestamps[0]), 0.25)
        self.assertAlmostEqual(float(timestamps[-1]), 0.269)
        np.testing.assert_array_equal(prefix[:, 0], np.arange(TRACE_PRE_MS, TRACE_PRE_MS + 20))

    def test_thresholds_are_calibrated_from_normal_and_label_physics(self) -> None:
        normal = make_trace("normal")
        normal.oracle[:, ORACLE_INDEX["foot_horizontal_speed_mps"]] = 0.01
        normal.oracle[:, ORACLE_INDEX["foot_sink_depth_m"]] = 0.001
        normal.oracle[:, ORACLE_INDEX["foot_velocity_z_mps"]] = -0.01
        normal.oracle[:, ORACLE_INDEX["foot_tilt_change_rad"]] = 0.01
        thresholds = calibrate_hazard_thresholds([normal])

        hazard = make_trace("hazard", "marble_to_ice")
        start = TRACE_PRE_MS + 5
        hazard.oracle[start:, ORACLE_INDEX["foot_horizontal_speed_mps"]] = 0.1
        labeled = label_trace(hazard, thresholds)
        self.assertAlmostEqual(onset_time_s(labeled, labeled.slip), 0.255)
        self.assertFalse(np.any(labeled.sink))
        self.assertFalse(np.any(labeled.tilt))

    def test_hazard_alignment_uses_onset_not_transition(self) -> None:
        trace = make_trace("hazard", "marble_to_ice")
        trace.slip[TRACE_PRE_MS + 10 :] = True
        timestamps, prefix = extract_prefix(
            trace, 5, alignment="hazard", hazard_label=trace.slip
        )
        self.assertEqual(prefix.shape, (5, 10))
        self.assertAlmostEqual(float(timestamps[0]), 0.260)

    def test_normal_scenario_does_not_hide_physical_hazard(self) -> None:
        trace = make_trace("normal_with_slip", "marble_to_marble")
        trace.slip[TRACE_PRE_MS + 7 :] = True
        fields = _onset_fields(trace)
        self.assertEqual(fields["hazard_type"], "slip")
        self.assertEqual(fields["physical_hazard"], 1)
        self.assertAlmostEqual(float(fields["transition_to_hazard_ms"]), 7.0)
        self.assertEqual(fields["target_hazard_onset_time_s"], "")

    def test_split_leakage_is_rejected(self) -> None:
        base = {
            "surface_family": "multisine", "surface_seed": 1,
            "session_id": "session", "run_id": "run",
        }
        validate_split_integrity([{**base, "split": "train"}])
        with self.assertRaisesRegex(ValueError, "surface_seed"):
            validate_split_integrity([
                {**base, "split": "train"},
                {
                    **base, "split": "test", "surface_family": "warped_multisine",
                    "session_id": "other", "run_id": "other",
                },
            ])


if __name__ == "__main__":
    unittest.main()
