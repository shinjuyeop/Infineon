"""Tests for Expanded Dataset v1 surface families and split design."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

from expanded_terrain_dataset_v1 import (  # noqa: E402
    SURFACE_FAMILIES,
    build_candidate_manifest,
    candidate_count,
    estimate_execution_cost,
    make_expanded_run_specification,
    make_expanded_surface_parameters,
    normalized_family_surface,
    validate_family_manifest,
)
from terrain_dataset_v1 import DOMAIN_RANGES, TERRAIN_LABELS  # noqa: E402
from run_expanded_terrain_dataset_v1 import (  # noqa: E402
    extract_native_rate_window,
    protocol,
)
from hil_sensor import HIL_SENSOR_CHANNELS  # noqa: E402


class ExpandedTerrainDatasetV1Test(unittest.TestCase):
    def test_default_design_is_4480_balanced_candidates(self) -> None:
        rows = build_candidate_manifest()
        self.assertEqual(candidate_count(), 4_480)
        self.assertEqual(len(rows), 4_480)
        self.assertEqual(
            {split: sum(family.split == split for family in SURFACE_FAMILIES) for split in ("train", "validation", "test")},
            {"train": 3, "validation": 2, "test": 2},
        )
        counts = {
            (terrain, family.name): sum(
                row["terrain_class"] == terrain and row["surface_family"] == family.name
                for row in rows
            )
            for terrain in TERRAIN_LABELS
            for family in SURFACE_FAMILIES
        }
        self.assertEqual(set(counts.values()), {160})

    def test_family_leakage_is_rejected(self) -> None:
        rows = build_candidate_manifest(surfaces_per_family=1, runs_per_surface=1)
        broken = deepcopy(rows)
        broken[0]["split"] = "test"
        with self.assertRaises(ValueError):
            validate_family_manifest(broken)

    def test_surface_families_are_deterministic_bounded_and_distinct(self) -> None:
        surfaces = []
        for family in SURFACE_FAMILIES:
            self.assertGreaterEqual(family.spatial_scale_m[0], 0.036)
            self.assertGreater(family.spatial_scale_m[1], family.spatial_scale_m[0])
            parameters = make_expanded_surface_parameters("concrete", family.name, 2)
            first = normalized_family_surface(81, 121, parameters)
            second = normalized_family_surface(81, 121, parameters)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (81, 121))
            self.assertTrue(np.all(np.isfinite(first)))
            self.assertAlmostEqual(float(first.mean()), 0.0, places=12)
            self.assertAlmostEqual(float(np.max(np.abs(first))), 1.0, places=12)
            surfaces.append(first)
        for left in range(len(surfaces)):
            for right in range(left + 1, len(surfaces)):
                correlation = abs(float(np.corrcoef(surfaces[left].ravel(), surfaces[right].ravel())[0, 1]))
                self.assertLess(correlation, 0.98)

    def test_terrain_amplitude_ranges_are_not_expanded(self) -> None:
        for terrain, domain in DOMAIN_RANGES.items():
            for family in SURFACE_FAMILIES:
                parameters = make_expanded_surface_parameters(terrain, family.name, 0)
                self.assertGreaterEqual(parameters.peak_to_valley_m, domain.roughness_ptv_m[0])
                self.assertLessEqual(parameters.peak_to_valley_m, domain.roughness_ptv_m[1])

    def test_run_specification_is_reproducible_and_family_disjoint(self) -> None:
        first = make_expanded_run_specification("sand", "warped_multisine", 3, 4)
        second = make_expanded_run_specification("sand", "warped_multisine", 3, 4)
        self.assertEqual(first, second)
        self.assertEqual(first.split, "test")
        self.assertIn("warped_multisine", first.session_id)

    def test_cost_estimate_scales_from_completed_pilot(self) -> None:
        estimate = estimate_execution_cost()
        self.assertEqual(estimate.candidates, 4_480)
        self.assertEqual(estimate.expected_valid, 4_439)
        self.assertGreater(estimate.estimated_runtime_s, 45.0 * 60.0)
        self.assertLess(estimate.estimated_runtime_s, 50.0 * 60.0)
        self.assertGreater(estimate.estimated_storage_bytes, 390 * 1024**2)
        self.assertLess(estimate.estimated_storage_bytes, 410 * 1024**2)

    def test_rate_ablation_protocol_uses_native_integer_step_cadence(self) -> None:
        expected = {100: (20, 0.5), 500: (4, 0.1), 1000: (2, 0.05)}
        for rate, (steps, duration) in expected.items():
            payload = protocol(4, 10, rate)
            self.assertEqual(payload["physics_steps_per_sample"], steps)
            self.assertEqual(payload["sample_count"], 50)
            self.assertAlmostEqual(payload["observation_duration_s"], duration)
            self.assertEqual(payload["channels"], tuple(HIL_SENSOR_CHANNELS))
        self.assertEqual(protocol(4, 10, 100)["window"]["name"], "medium_response")
        self.assertEqual(
            protocol(4, 10, 1000)["window"],
            {
                "name": "pulse_onset_rate_ablation",
                "start_s": 0.25,
                "end_s": 0.30,
                "interval": "left-closed/right-open",
            },
        )

    def test_1khz_native_timestamps_extract_exact_50ms_window(self) -> None:
        timestamps = np.arange(1, 1001, dtype=np.float64) / 1000.0
        sensors = np.arange(1000 * 10, dtype=np.float32).reshape(1000, 10)
        selected_times, selected = extract_native_rate_window(
            timestamps, sensors, 1000, 0.25
        )
        self.assertEqual(selected.shape, (50, 10))
        self.assertAlmostEqual(float(selected_times[0]), 0.25)
        self.assertAlmostEqual(float(selected_times[-1]), 0.299)
        np.testing.assert_allclose(np.diff(selected_times), 0.001, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(selected, sensors[249:299])

    def test_1khz_window_rejects_non_native_timestamp_grid(self) -> None:
        timestamps = np.arange(1, 1001, dtype=np.float64) / 1000.0
        timestamps[250] += 0.0001
        sensors = np.zeros((1000, 10), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "physics-step grid"):
            extract_native_rate_window(timestamps, sensors, 1000, 0.25)


if __name__ == "__main__":
    unittest.main()
