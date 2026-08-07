"""Regression tests for the native-hfield surface/sampling-rate study."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

import config  # noqa: E402
from run_surface_sampling_rate_study import (  # noqa: E402
    HIGH_RATE_HZ,
    SURFACE_SCENE_PATH,
    periodogram,
)
from surface_profiles import (  # noqa: E402
    CONCRETE_PEAK_TO_VALLEY_M,
    MARBLE_PEAK_TO_VALLEY_M,
    configure_surface_floor,
    configure_factorized_surface,
)
from terrain_profiles import TERRAIN_PROFILES  # noqa: E402


class SurfaceSamplingStudyTest(unittest.TestCase):
    def load_model(self) -> mujoco.MjModel:
        model = mujoco.MjModel.from_xml_path(str(SURFACE_SCENE_PATH.resolve()))
        model.opt.timestep = config.SIMULATE_DT
        return model

    def test_high_rate_matches_distinct_physics_states(self) -> None:
        self.assertEqual(config.SIMULATE_DT, 0.005)
        self.assertEqual(HIGH_RATE_HZ, 1.0 / config.SIMULATE_DT)
        self.assertLess(HIGH_RATE_HZ, 1000.0)

    def test_surface_amplitudes_and_contact_parameters(self) -> None:
        for terrain, expected_amplitude in (
            ("concrete", CONCRETE_PEAK_TO_VALLEY_M),
            ("marble", MARBLE_PEAK_TO_VALLEY_M),
        ):
            model = self.load_model()
            floor_id, statistics = configure_surface_floor(
                model, TERRAIN_PROFILES[terrain]
            )
            self.assertEqual(model.geom(floor_id).name, "surface_floor")
            self.assertAlmostEqual(statistics.peak_to_valley_m, expected_amplitude)
            np.testing.assert_allclose(
                model.geom_friction[floor_id], TERRAIN_PROFILES[terrain].friction
            )
            np.testing.assert_allclose(
                model.geom_solref[floor_id], TERRAIN_PROFILES[terrain].solref
            )
            np.testing.assert_allclose(
                model.geom_solimp[floor_id], TERRAIN_PROFILES[terrain].solimp
            )

    def test_periodogram_finds_known_peak(self) -> None:
        timestamps = np.arange(80) / HIGH_RATE_HZ
        signal = np.sin(2.0 * np.pi * 20.0 * timestamps)
        frequencies, psd = periodogram(signal, HIGH_RATE_HZ)
        self.assertAlmostEqual(float(frequencies[np.argmax(psd)]), 20.0)

    def test_friction_and_roughness_are_independent_factors(self) -> None:
        for friction_source in ("concrete", "marble"):
            for roughness_source, amplitude in (
                ("concrete", CONCRETE_PEAK_TO_VALLEY_M),
                ("marble", MARBLE_PEAK_TO_VALLEY_M),
            ):
                model = self.load_model()
                floor_id, statistics = configure_factorized_surface(
                    model,
                    TERRAIN_PROFILES[friction_source],
                    roughness_source,
                )
                np.testing.assert_allclose(
                    model.geom_friction[floor_id],
                    TERRAIN_PROFILES[friction_source].friction,
                )
                self.assertAlmostEqual(statistics.peak_to_valley_m, amplitude)


if __name__ == "__main__":
    unittest.main()
