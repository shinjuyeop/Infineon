import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_terrain_fast_reflex_v2_final import FROZEN, SURFACE_INDICES, FINAL_SESSION_SEED, FINAL_EXCITATION_OFFSET, planned_runs, prepare_evaluation_output, verify_preflight, verify_selected_values
from terrain_fast_reflex_v2 import validate_final_test_request


class V2FinalProtocolTest(unittest.TestCase):
    def test_frozen_configs_and_allocation_are_exact(self):
        self.assertEqual((FROZEN["slip"]["window_ms"], FROZEN["slip"]["persistence"]), (5, 3))
        self.assertEqual((FROZEN["sink"]["window_ms"], FROZEN["sink"]["persistence"]), (20, 1))
        self.assertEqual(planned_runs(), 90)
        self.assertEqual((min(SURFACE_INDICES), max(SURFACE_INDICES), FINAL_SESSION_SEED, FINAL_EXCITATION_OFFSET), (9110, 9114, 20260903, 921000))

    def test_alternative_selection_values_are_rejected(self):
        for key, value in (("threshold", .5), ("window_ms", 10), ("persistence", 2)):
            actual = {name: FROZEN["slip"][name] for name in ("threshold", "window_ms", "persistence")}
            actual[key] = value
            with self.assertRaisesRegex(ValueError, key): verify_selected_values("slip", actual)

    def test_hash_and_one_shot_output_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final"
            report = verify_preflight(output)
            self.assertTrue(report["FINAL_TEST_READY"]); self.assertEqual(report["final_test_materialized"], 0)
            output.mkdir()
            with self.assertRaises(FileExistsError): verify_preflight(output)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("run_terrain_fast_reflex_v2_final.sha256", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"): verify_preflight(Path(temporary) / "final")

    def test_legacy_v1_test_families_stay_rejected(self):
        for family in ("warped_multisine", "smooth_random_patches"):
            with self.assertRaisesRegex(ValueError, "ineligible"):
                validate_final_test_request([family], True)

    def test_only_empty_pre_model_load_evaluation_staging_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            self.assertEqual(prepare_evaluation_output(output), output / "plots")
            self.assertEqual(prepare_evaluation_output(output), output / "plots")
            (output / "slip_final_metrics.json").write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "non-pristine"):
                prepare_evaluation_output(output)


if __name__ == "__main__":
    unittest.main()
