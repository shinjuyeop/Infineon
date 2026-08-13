from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrain_fast_reflex_v2 import (  # noqa: E402
    TRACE_SAMPLES, V2_ORACLE_CHANNELS, V2_ORACLE_INDEX, V2Calibration,
    calibration_scenario_configs, default_scenario_configs, label_v2,
    front_rear_torque_calibration_configs, validate_final_test_request, validate_state_order,
)


class FastReflexV2Test(unittest.TestCase):
    def test_final_test_is_fail_closed(self) -> None:
        families = ["warped_multisine"]
        with self.assertRaisesRegex(ValueError, "final-test"):
            validate_final_test_request(families, False)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            validate_final_test_request(families, True)

    def test_risk_contains_confirmed_and_states_are_ordered(self) -> None:
        oracle = np.zeros((TRACE_SAMPLES, len(V2_ORACLE_CHANNELS)))
        oracle[:, V2_ORACLE_INDEX["left_contact"]] = 1.0
        oracle[:, V2_ORACLE_INDEX["contact_normal_force_N"]] = 100.0
        oracle[55:60, V2_ORACLE_INDEX["foot_horizontal_speed_mps"]] = .6
        calibration = V2Calibration(10, .1, .01, .5, .2, .2, .2, .2)
        labels = label_v2(oracle, calibration)
        self.assertTrue(labels["confirmed_slip"].any())
        self.assertTrue(np.all(labels["confirmed_slip"] <= labels["slip_risk"]))
        validate_state_order(labels)

    def test_sustained_sink_requires_persistence(self) -> None:
        oracle = np.zeros((TRACE_SAMPLES, len(V2_ORACLE_CHANNELS)))
        oracle[:, V2_ORACLE_INDEX["left_contact"]] = 1.0
        oracle[:, V2_ORACLE_INDEX["contact_normal_force_N"]] = 100.0
        oracle[60:62, V2_ORACLE_INDEX["foot_sink_depth_m"]] = .5
        oracle[60:62, V2_ORACLE_INDEX["foot_velocity_z_mps"]] = -.5
        calibration = V2Calibration(10, .9, .9, 1., .1, .1, 1., 1., persistence_samples=3)
        self.assertFalse(label_v2(oracle, calibration)["sustained_sink"].any())

    def test_scenario_configs_serialize_and_include_slip(self) -> None:
        configs = calibration_scenario_configs()
        self.assertEqual(len(configs), 19)
        self.assertIn("slip_risk_dominant", {config.mode for config in configs})
        self.assertTrue(all(config.as_dict()["config_id"] == config.config_id for config in configs))

    def test_default_mode_layout_and_seam_are_explicit(self) -> None:
        configs = {config.mode: config for config in default_scenario_configs()}
        self.assertEqual(configs["boundary_left_right"].layout, "left_right")
        self.assertNotEqual(configs["boundary_left_right"].seam_offset_m, 0.0)
        self.assertTrue(configs["slip_risk_dominant"].switch_to_ice)
        self.assertEqual(configs["normal_sand"].horizontal_force_N, 0.0)
        self.assertEqual(configs["normal_sand"].pitch_torque_Nm, 0.0)
        self.assertEqual(configs["tilt_dominant"].pitch_torque_Nm, 0.0)
        torque_configs = front_rear_torque_calibration_configs()
        self.assertEqual(len(torque_configs), 7)
        self.assertTrue(any(config.pitch_torque_Nm > 0.0 for config in torque_configs))


if __name__ == "__main__":
    unittest.main()
