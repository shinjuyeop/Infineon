"""Short static tests for separate 100 Hz and fast-1000 deployment profiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generate_terrain_artifacts import (
    PROFILES,
    ensure_writable_targets,
    target_paths,
    validate_source,
)
from terrain_preprocessing import paths_for_profile, preprocessor_for_profile
from terrain_fixed_test_client import verify_log
from terrain_hil_client import verify_fixed_golden


class TerrainDeploymentProfilesTest(unittest.TestCase):
    def test_model_names_and_generated_targets_are_disjoint(self) -> None:
        baseline = PROFILES["100hz"]
        fast = PROFILES["fast1000"]
        self.assertNotEqual(baseline.cpu_name, fast.cpu_name)
        self.assertNotEqual(baseline.u55_name, fast.u55_name)
        self.assertTrue(
            set(target_paths(baseline, "all")).isdisjoint(target_paths(fast, "all"))
        )

    def test_fast_source_identity_shape_and_operators(self) -> None:
        metadata = validate_source(PROFILES["fast1000"])
        self.assertEqual(metadata["sample_rate_hz"], 1000)
        self.assertEqual(metadata["tflite"]["size_bytes"], 7048)
        self.assertEqual(metadata["tflite"]["input"]["shape"], [1, 50, 10])
        self.assertEqual(metadata["tflite"]["output"]["shape"], [1, 4])

    def test_overwrite_guard_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.c"
            target.write_text("user artifact", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_writable_targets([target], force=False)
            ensure_writable_targets([target], force=True)

    def test_fast_preprocessor_is_separate_and_quantized(self) -> None:
        metadata, model, dataset = paths_for_profile("fast1000")
        self.assertIn("1000hz", str(metadata))
        self.assertIn("1000hz", str(model))
        self.assertIn("1000hz", str(dataset))
        preprocessor = preprocessor_for_profile("fast1000")
        self.assertEqual(preprocessor.input_zero_point, 3)

    def test_existing_stream_client_import_contract_is_preserved(self) -> None:
        import terrain_hil_client
        import terrain_stream_client

        self.assertEqual(terrain_stream_client.DATASET, terrain_hil_client.DATASET)
        self.assertIn("terrain_dataset_v1_expanded", str(terrain_stream_client.DATASET))

    def test_fixed_log_requires_exact_raw_and_class_parity(self) -> None:
        metadata = {
            "host_output_raw": [114, -114, -128, -128],
            "host_class": 0,
            "expected_class": "concrete",
        }
        log = (
            b"TERRAIN_RESULT, output[0]_raw=[114,-114,-128,-128], "
            b"host_raw=[114,-114,-128,-128], device_class=0, host_class=0\r\n"
            b"Output[0] : PASS with accuracy percentage =100.00, total_cnt=1\r\n"
        )
        self.assertTrue(verify_log(log, metadata)["passed"])
        with self.assertRaises(RuntimeError):
            verify_log(log.replace(b"[114,-114", b"[113,-114", 1), metadata)

    def test_existing_trn1_result_matches_fast_golden(self) -> None:
        # Use a temporary metadata binding so this test does not require generated files.
        import json
        import terrain_hil_client

        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "fixed.json"
            metadata.write_text(
                json.dumps({"host_output_raw": [114, -114, -128, -128], "host_class": 0}),
                encoding="utf-8",
            )
            original = terrain_hil_client.FIXED_METADATA["fast1000"]
            terrain_hil_client.FIXED_METADATA["fast1000"] = metadata
            try:
                result = verify_fixed_golden(
                    b"HIL_RESULT raw=[114,-114,-128,-128],class=0,cpu_cyc=1,npu_cyc=2",
                    "fast1000",
                )
                self.assertTrue(result["passed"])
            finally:
                terrain_hil_client.FIXED_METADATA["fast1000"] = original


if __name__ == "__main__":
    unittest.main()
