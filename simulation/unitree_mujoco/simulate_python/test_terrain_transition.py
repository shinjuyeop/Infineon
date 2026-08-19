"""Schema-only regression tests for Terrain Transition v1."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import run_terrain_transition as transition
import visualize_terrain_transition as visualization


class TerrainTransitionTest(unittest.TestCase):
    def test_four_case_mapping_is_bidirectional_and_marble_is_hard(self):
        self.assertEqual(transition.CASES["A"]["before"], "marble")
        self.assertEqual(transition.CASES["A"]["after"], "ice")
        self.assertEqual(transition.CASES["B"]["after"], "sand")
        self.assertEqual(transition.CASES["C"]["after"], "marble")
        self.assertEqual(transition.CASES["D"]["after"], "marble")

    def test_frozen_oracle_requires_physical_conditions_not_terrain_name(self):
        oracle = np.zeros((8, len(transition.V2_ORACLE_CHANNELS)), dtype=float)
        labels = transition.label(oracle, 4)
        self.assertFalse(any(value.any() for value in labels.values()))

    def test_sustained_is_causal(self):
        result = transition._sustained(np.array([False, True, True, True, False]), 3)
        np.testing.assert_array_equal(result, [False, True, True, True, False])

    def test_visualization_cli_accepts_each_case_and_deterministic_controls(self):
        for case_id in transition.CASES:
            args = visualization.parse_args([
                "--case", case_id, "--run-index", "2", "--surface-family",
                "filtered_random", "--surface-index", "4", "--no-viewer",
                "--hold-seconds", "0",
            ])
            self.assertEqual(args.case, case_id)
            self.assertEqual(args.run_index, 2)
            self.assertEqual(args.surface_family, "filtered_random")
            self.assertEqual(args.surface_index, 4)
            self.assertFalse(args.viewer)
            self.assertEqual(args.hold_seconds, 0.0)

    def test_t0_observer_event_is_aligned_with_frozen_profile_switch(self):
        events = []

        def observer(model, data, frame):
            del model
            if frame.phase == "transition":
                events.append((data.time, frame.terrain, frame.physics_steps))

        run = transition.run_one("A", 0, observer=observer)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], transition.TRANSITION_TIME_S, places=12)
        self.assertEqual(events[0][1:], ("ice", 1300))
        self.assertEqual(run.metadata["transition_sample"], 650)
        self.assertTrue(np.all(run.terrain_gt[:650] == "marble"))
        self.assertTrue(np.all(run.terrain_gt[650:] == "ice"))

    def test_viewer_disabled_visualization_is_headless_physics_identical(self):
        headless = transition.run_one("C", 0)
        visual_disabled = visualization.run_visualization("C", viewer=False, speed=0.25)
        np.testing.assert_array_equal(visual_disabled.fusion10, headless.fusion10)
        np.testing.assert_array_equal(visual_disabled.oracle, headless.oracle)
        np.testing.assert_array_equal(visual_disabled.terrain_gt, headless.terrain_gt)
        np.testing.assert_array_equal(visual_disabled.final_qpos, headless.final_qpos)
        np.testing.assert_array_equal(visual_disabled.final_qvel, headless.final_qvel)
        self.assertEqual(visual_disabled.metadata["transition_sample"], headless.metadata["transition_sample"])
        labels = transition.label(visual_disabled.oracle, 650)
        headless_labels = transition.label(headless.oracle, 650)
        for key in labels:
            np.testing.assert_array_equal(labels[key], headless_labels[key])

    def test_visual_only_terrain_colours_do_not_change_physics(self):
        headless = transition.run_one("B", 0)

        def colour_observer(model, data, frame):
            del data
            if frame.phase in {"initialize", "transition"}:
                visualization.apply_visual_terrain_appearance(model, frame.terrain)

        coloured = transition.run_one("B", 0, observer=colour_observer)
        np.testing.assert_array_equal(coloured.fusion10, headless.fusion10)
        np.testing.assert_array_equal(coloured.oracle, headless.oracle)
        np.testing.assert_array_equal(coloured.final_qpos, headless.final_qpos)
        np.testing.assert_array_equal(coloured.final_qvel, headless.final_qvel)

    def test_recording_refuses_to_overwrite_an_existing_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "case_a.mp4"
            output.touch()
            with self.assertRaises(FileExistsError):
                visualization.run_visualization("A", viewer=False, record_path=output)


if __name__ == "__main__":
    unittest.main()
