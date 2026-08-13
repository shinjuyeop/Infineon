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
    front_rear_torque_calibration_configs, local_compliance_calibration_configs,
    final_tilt_physics_calibration_configs, validate_final_test_request, validate_state_order,
    DEPLOYMENT_SCOPE, final_scope_calibration_configs, sand_sink_hazard,
    final_scope_pilot_configs, final_scope_full_configs, slip_hazard,
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

    def test_local_compliance_seams_use_measured_sole_extent_ratios(self) -> None:
        configs = local_compliance_calibration_configs()
        self.assertEqual(len(configs), 7)
        seams = {config.seam_offset_m for config in configs if config.mode == "tilt_dominant"}
        self.assertEqual(seams, {.035, .052, .069})
        self.assertTrue(all(config.pitch_torque_Nm == 0.0 for config in configs))

    def test_final_tilt_physics_is_bounded_and_torque_free(self) -> None:
        configs = final_tilt_physics_calibration_configs()
        self.assertEqual(len(configs), 9)
        self.assertTrue(np.allclose(sorted({config.height_offset_m for config in configs if config.height_offset_m}), [25e-6, 50e-6, 75e-6, 100e-6]))
        self.assertTrue(all(config.horizontal_force_N == config.vertical_force_N == config.pitch_torque_Nm == 0.0 for config in configs))

    def test_final_scope_excludes_isolated_tilt_and_maps_sink_target(self) -> None:
        self.assertEqual(DEPLOYMENT_SCOPE["slip_hazard"]["target"], "confirmed_slip")
        self.assertTrue(DEPLOYMENT_SCOPE["incipient_slip"]["diagnostic_only"])
        self.assertTrue(DEPLOYMENT_SCOPE["sand_isolated_tilt"]["diagnostic_only"])
        self.assertEqual(len(final_scope_calibration_configs()), 17)
        labels={"sustained_sink":np.array([False, True]), "sustained_tilt":np.array([True, True])}
        self.assertTrue(np.array_equal(sand_sink_hazard(labels), [False, True]))
        self.assertFalse(sand_sink_hazard({"sustained_sink": np.array([False]), "sustained_tilt": np.array([False])})[0])
        self.assertTrue(np.array_equal(slip_hazard({"confirmed_slip":np.array([False,True]),"incipient_risk":np.array([True,False])}),[False,True]))
        self.assertEqual({config.mode for config in final_scope_pilot_configs()}, {"normal_sand","slip_risk_dominant","sink_dominant","sink_and_tilt"})
        self.assertEqual({config.mode for config in final_scope_full_configs()}, {"normal_sand","slip_risk_dominant","sink_dominant"})


if __name__ == "__main__":
    unittest.main()
