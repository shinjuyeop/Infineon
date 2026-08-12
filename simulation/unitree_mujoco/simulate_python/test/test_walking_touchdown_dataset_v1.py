from __future__ import annotations

import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from g1_upstream_locomotion import ACTUATOR_NAMES, gravity_orientation  # noqa: E402
from controlled_excitation import find_allowed_foot_geom_ids  # noqa: E402
from run_walking_stability_sweep import run_passes, summarize_stability  # noqa: E402
from run_walking_touchdown_dataset_v1 import (  # noqa: E402
    SCENE_PATH,
    configure_contact_model,
)
from walking_touchdown_dataset_v1 import (  # noqa: E402
    EVENT_SAMPLES,
    POST_SAMPLES,
    PRE_SAMPLES,
    RELATIVE_TIME_MS,
    TouchdownEventCollector,
    stack_events,
    validate_event_split_integrity,
)


class WalkingTouchdownDatasetV1Test(unittest.TestCase):
    def test_exact_native_window_and_fsr_diagnostic(self) -> None:
        collector = TouchdownEventCollector(min_air_samples=10, contact_confirmation_samples=3)
        for index in range(100):
            contact = 30 <= index < 70
            sensor = np.zeros(10, dtype=np.float64)
            if contact:
                sensor[:4] = 2.0
                sensor[4:] = np.arange(6) + index
            collector.append((index + 1) / 1000.0, sensor, contact)
        collector.finish()
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertTrue(event.valid, event.invalid_reason)
        self.assertEqual(event.sensors.shape, (EVENT_SAMPLES, 10))
        np.testing.assert_allclose(
            (event.timestamps_s - event.touchdown_time_s) * 1000.0,
            RELATIVE_TIME_MS,
            rtol=0.0,
            atol=1e-9,
        )
        self.assertFalse(np.any(event.contacts[:PRE_SAMPLES]))
        self.assertTrue(event.contacts[PRE_SAMPLES])
        self.assertAlmostEqual(event.fsr_threshold_crossing_time_s, event.touchdown_time_s)
        arrays = stack_events(collector.events)
        self.assertEqual(arrays["sensors"].shape, (1, PRE_SAMPLES + POST_SAMPLES, 10))

    def test_contact_chatter_is_not_touchdown(self) -> None:
        collector = TouchdownEventCollector(min_air_samples=10, contact_confirmation_samples=4)
        contacts = [False] * 20 + [True, False] + [False] * 15 + [True] * 20
        for index, contact in enumerate(contacts):
            collector.append((index + 1) / 1000.0, np.zeros(10), contact)
        collector.finish()
        self.assertEqual(collector.rejected_contact_chatter, 1)
        self.assertEqual(len(collector.events), 0)  # final valid candidate lacks +50 ms tail
        self.assertEqual(collector.incomplete_at_end, 1)

    def test_non_native_timestamp_is_rejected(self) -> None:
        collector = TouchdownEventCollector()
        collector.append(0.001, np.zeros(10), False)
        with self.assertRaisesRegex(ValueError, "1 kHz"):
            collector.append(0.0021, np.zeros(10), False)

    def test_split_is_owned_above_event_level(self) -> None:
        base = {
            "surface_family": "multisine", "surface_seed": 1,
            "session_id": "session", "run_id": "run",
        }
        validate_event_split_integrity([{**base, "split": "train"}, {**base, "split": "train"}])
        with self.assertRaisesRegex(ValueError, "family"):
            validate_event_split_integrity([
                {**base, "split": "train"},
                {**base, "split": "test", "surface_seed": 2, "session_id": "s2", "run_id": "r2"},
            ])

    def test_upstream_controller_conventions(self) -> None:
        self.assertEqual(len(ACTUATOR_NAMES), 29)
        np.testing.assert_allclose(gravity_orientation(np.asarray((1.0, 0.0, 0.0, 0.0))), (0.0, 0.0, -1.0))

    def test_stability_gate_requires_valid_complete_run_and_touchdowns(self) -> None:
        passing = {
            "run_valid": 1, "valid_events": 3, "stopped_early": 0,
            "walking_speed_mps": 0.2, "duration_s": 3.0,
            "forward_displacement_m": 0.4,
        }
        self.assertTrue(run_passes(passing, minimum_touchdowns=3))
        for field, value in (
            ("run_valid", 0),
            ("valid_events", 2),
            ("stopped_early", 1),
        ):
            row = {**passing, field: value}
            self.assertFalse(run_passes(row, minimum_touchdowns=3))
        backward = {**passing, "forward_displacement_m": -0.2}
        self.assertFalse(
            run_passes(backward, minimum_touchdowns=3, minimum_forward_progress_ratio=0.5)
        )

    def test_foot_sphere_contact_model_is_runtime_only_and_explicit(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        floor_id = model.geom("surface_floor").id
        allowed = find_allowed_foot_geom_ids(model)
        disabled = configure_contact_model(model, floor_id, "foot-spheres-only")
        self.assertGreater(len(disabled), 0)
        self.assertNotIn(floor_id, allowed)
        self.assertNotEqual(model.geom_contype[floor_id], 0)
        for geom_id in allowed:
            self.assertNotEqual(model.geom_contype[geom_id], 0)
        for geom_id in range(model.ngeom):
            if geom_id != floor_id and geom_id not in allowed:
                self.assertEqual(model.geom_contype[geom_id], 0)
                self.assertEqual(model.geom_conaffinity[geom_id], 0)

    def test_stability_summary_finds_only_common_all_run_speed(self) -> None:
        def row(terrain: str, speed: float, run_index: int, passed: bool) -> dict[str, object]:
            return {
                "terrain_name": terrain,
                "walking_speed_mps": speed,
                "run_index": run_index,
                "run_valid": int(passed),
                "valid_events": 3 if passed else 0,
                "stopped_early": int(not passed),
                "failure_reason": "" if passed else "fallen_orientation",
                "failure_time_s": "" if passed else 1.25,
                "touchdown_events": 3 if passed else 1,
                "min_base_height_m": 0.75 if passed else 0.5,
                "min_upright_z": 0.99 if passed else 0.4,
                "forward_displacement_m": 0.2,
            }

        rows = [
            row(terrain, speed, run_index, not (speed == 0.2 and terrain == "sand"))
            for terrain in ("ice", "sand")
            for speed in (0.1, 0.2)
            for run_index in range(2)
        ]
        matrix, common = summarize_stability(rows, minimum_touchdowns=3)
        self.assertEqual(common, [0.1])
        sand_fast = next(
            item for item in matrix
            if item["terrain_name"] == "sand" and item["walking_speed_mps"] == 0.2
        )
        self.assertEqual(sand_fast["pass_runs"], 0)
        self.assertEqual(sand_fast["failure_reasons"], "fallen_orientation")


if __name__ == "__main__":
    unittest.main()
