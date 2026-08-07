"""Functional checks for distinct-state timestep convergence runs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
import unittest

import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

from controlled_excitation import HorizontalPulse, generate_pulse_conditions  # noqa: E402
from run_horizontal_pulse_dataset import DEFAULT_SEED, run_window  # noqa: E402
from run_surface_factorization_study import SCENE_PATH, factor_configurator  # noqa: E402


class TimestepConvergenceTest(unittest.TestCase):
    def test_each_timestep_advances_distinct_finite_states(self) -> None:
        condition = generate_pulse_conditions(1, DEFAULT_SEED)[0]
        pulse = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
        support_forces = []
        for timestep in (0.005, 0.002, 0.001, 0.0005, 0.00025):
            metrics = run_window(
                "concrete",
                condition,
                1.0,
                1.0 / timestep,
                DEFAULT_SEED,
                0.70,
                pulse,
                None,
                scene_path=SCENE_PATH,
                model_configurator=partial(
                    factor_configurator, roughness_source="concrete"
                ),
                physics_timestep=timestep,
            )
            self.assertEqual(metrics["sample_count"], int(round(1.0 / timestep)))
            self.assertEqual(metrics["valid_run"], 1)
            self.assertEqual(metrics["body_collision"], 0)
            self.assertEqual(metrics["extreme_force_outlier"], 0)
            self.assertEqual(metrics["extreme_accel_outlier"], 0)
            self.assertTrue(np.isfinite(float(metrics["accel_rms"])))
            support_forces.append(float(metrics["support_target_force"]))
        np.testing.assert_allclose(support_forces, support_forces[0])

    def test_pulse_timing_is_seconds_based(self) -> None:
        pulse = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
        for timestep in (0.005, 0.002, 0.001, 0.0005, 0.00025):
            active_steps = sum(
                pulse.force_at(index * timestep)[1]
                for index in range(int(round(1.0 / timestep)))
            )
            self.assertAlmostEqual(active_steps * timestep, pulse.duration)

    def test_pulse_timing_survives_accumulated_roundoff(self) -> None:
        pulse = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
        for timestep in (0.001, 0.0005, 0.00025):
            time = 0.0
            active_steps = 0
            for _ in range(int(round(1.0 / timestep))):
                active_steps += int(pulse.force_at(time)[1])
                time += timestep
            self.assertEqual(active_steps, int(round(pulse.duration / timestep)))


if __name__ == "__main__":
    unittest.main()
