#!/usr/bin/env python3
"""Focused contracts for Walking-v2 targeted Slip retraining v3."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from run_walking_v2_slip_targeted_retraining_v3 import (
    EXPECTED_IMMUTABLE_HASHES, OUTPUT, SAFETY_FIELDS, STARTING_CHECKPOINT,
    V8, load_model,
)
from walking_v2_slip_targeted_retraining_v3 import (
    ACTIONABLE, ACTIVE, EARLY, NORMAL, OperatingConfig, candidate_matrix_payload,
    causal_feature_matrix, corrected_r4_labels, operating_grid,
    runtime_contact_age, runtime_crossings, selection_policy_payload,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / OUTPUT


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CausalContractTests(unittest.TestCase):
    def test_corrected_r4_exact_boundaries_and_first_fall(self) -> None:
        episode = np.full((300, 2), -1, np.int32)
        episode[10:250, 0] = 0
        active = np.zeros((300, 2), bool)
        active[200:210, 0] = True
        valid = episode >= 0
        prefall = np.ones(300, bool)
        prefall[205:] = False
        labels, delta, events = corrected_r4_labels(episode, active, valid, prefall)
        self.assertEqual(labels[99, 0], EARLY)       # onset is 101 ms away
        self.assertEqual(labels[100, 0], ACTIONABLE) # onset is 100 ms away
        self.assertEqual(labels[199, 0], ACTIONABLE)
        self.assertEqual(labels[200, 0], ACTIVE)
        self.assertEqual(labels[204, 0], ACTIVE)
        self.assertEqual(labels[205, 0], -1)
        self.assertEqual(delta[100, 0], 100)
        self.assertEqual(events, [{"foot": 0, "episode_id": 0, "onset_sample": 200}])

    def test_features_are_causal_through_200_ms(self) -> None:
        generator = np.random.default_rng(7)
        sensor = generator.normal(size=(400, 20)).astype(np.float32)
        loaded = np.ones((400, 2), bool)
        for family in ("R1", "R2", "R3"):
            before = causal_feature_matrix(sensor, loaded, 0.15, family)
            modified = sensor.copy()
            modified[251:] += 1000.0
            after = causal_feature_matrix(modified, loaded, 0.15, family)
            self.assertTrue(np.array_equal(before[:251], after[:251]), family)

    def test_contact_age_episode_and_reset(self) -> None:
        loaded = np.zeros((30, 2), bool)
        loaded[2:9, 0] = True
        loaded[12:25, 0] = True
        ages, episodes = runtime_contact_age(loaded)
        self.assertEqual(ages[2, 0], 1)
        self.assertEqual(ages[8, 0], 7)
        self.assertEqual(ages[12, 0], 1)
        self.assertEqual(episodes[8, 0], 0)
        self.assertEqual(episodes[12, 0], 1)
        probability = np.zeros((30, 2, 4), np.float32)
        probability[:, :, NORMAL] = 1.0
        probability[15:25, 0, NORMAL] = 0.01
        probability[15:25, 0, ACTIONABLE] = 0.99
        crossings, resets = runtime_crossings(
            probability, loaded, OperatingConfig(0.95, 3, 0.0, 0.05))
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]["runtime_episode_id"], 1)
        self.assertTrue(all(not row["latch_carryover"] for row in resets))

    def test_simultaneous_crossing_has_one_deterministic_owner(self) -> None:
        loaded = np.ones((30, 2), bool)
        probability = np.zeros((30, 2, 4), np.float32)
        probability[:, :, NORMAL] = 1.0
        probability[12:18, :, NORMAL] = 0.01
        probability[12:18, :, ACTIONABLE] = 0.99
        crossings, _ = runtime_crossings(
            probability, loaded, OperatingConfig(0.95, 3, 0.0, 0.05))
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]["foot"], 0)
        self.assertEqual(crossings[0]["simultaneous_candidate_count"], 2)

    def test_frozen_candidate_and_operating_matrix(self) -> None:
        matrix = candidate_matrix_payload()
        self.assertEqual(matrix["candidate_count"], 9)
        self.assertEqual(matrix["family_count"], 3)
        self.assertEqual(len(operating_grid()), 16)
        self.assertFalse(matrix["fourth_family_allowed"])
        self.assertFalse(selection_policy_payload()["post_result_threshold_search_allowed"])


class ExecutedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUT / "summary.json").exists():
            raise AssertionError("run the frozen retraining runner before focused artifact tests")
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.readiness = json.loads((OUT / "readiness.json").read_text())

    def test_all_required_artifacts(self) -> None:
        required = {
            "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
            "artifact_access_log.json", "immutable_verification.json", "candidate_matrix.json",
            "selection_policy.json", "eligibility_audit.csv", "nested_fold_audit.csv",
            "label_distribution.csv", "episode_distribution.csv", "training_health.csv",
            "hard_negative_audit.csv", "candidate_fold_metrics.csv",
            "candidate_pooled_metrics.csv", "candidate_speed_metrics.csv",
            "candidate_foot_metrics.csv", "candidate_phase_metrics.csv",
            "crossing_reconciliation.csv", "timing_metrics.csv", "reset_audit.csv",
            "future_leakage_audit.json", "resource_report.json", "provenance.json",
            "readiness.json", "summary.json", "audit.md",
        }
        self.assertFalse(required - {path.name for path in OUT.iterdir()})
        self.assertTrue(all(path.stat().st_size <= 45 * 1024 * 1024 for path in OUT.iterdir()))

    def test_authoritative_eligibility_and_exclusions(self) -> None:
        with (OUT / "eligibility_audit.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        eligible = [row for row in rows if row["training_eligible"] == "True"]
        excluded = [row for row in rows if row["training_eligible"] != "True"]
        self.assertEqual(len(eligible), 213)
        self.assertEqual(sum(row["eligibility"] == "POSITIVE_ELIGIBLE" for row in eligible), 93)
        self.assertEqual(sum(row["eligibility"] == "CONTROL_ELIGIBLE" for row in eligible), 120)
        self.assertEqual(sum(row["eligibility"] == "FAILED_SOURCE_DIAGNOSTIC_ONLY" for row in excluded), 27)
        self.assertEqual(sum(row["eligibility"] == "CALIBRATION_ONLY_DO_NOT_TRAIN" for row in excluded), 42)
        self.assertTrue(all(row["included"] == "False" for row in excluded))
        self.assertEqual(self.summary["excluded_data_used_count"], 0)

    def test_exact_nested_folds_and_all_leakage_zero(self) -> None:
        frozen = json.loads((ROOT / V8 / "future_nested_fold_manifest.json").read_text())
        with (OUT / "nested_fold_audit.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(row["run_id"], str(row["fold"])) for row in frozen["rows"]}
        actual = {(row["run_id"], row["fold"]) for row in rows}
        self.assertEqual(actual, expected)
        self.assertEqual(len(rows), 333)
        self.assertTrue(all(row["pair_run_variation_episode_leakage"] == "0" for row in rows))
        leakage = json.loads((OUT / "future_leakage_audit.json").read_text())
        self.assertEqual(leakage["future_leakage_count"], 0)
        self.assertEqual(leakage["normalization_outer_fold_overlap_count"], 0)
        self.assertEqual(leakage["hard_negative_evaluated_fold_access_count"], 0)

    def test_train_only_normalization_and_hard_negative_isolation(self) -> None:
        with (OUT / "training_health.csv").open(newline="") as stream:
            health = list(csv.DictReader(stream))
        self.assertEqual(len(health), 27)
        self.assertTrue(all(row["normalization_outer_run_overlap"] == "0" for row in health))
        self.assertTrue(all(row["complete_population_used"] == "True" for row in health))
        with (OUT / "hard_negative_audit.csv").open(newline="") as stream:
            hard = list(csv.DictReader(stream))
        self.assertTrue(hard)
        self.assertTrue(all(row["evaluated_fold_used"] == "False" for row in hard))
        self.assertTrue(all(row["source_fold"] != row["outer_fold"] for row in hard))

    def test_oof_population_and_candidate_counts(self) -> None:
        for family in ("R1", "R2", "R3"):
            with np.load(OUT / f"oof_predictions_{family}.npz", allow_pickle=False) as data:
                self.assertEqual(data["actionable_probability"].shape, (3, 333, 3000, 2))
                self.assertEqual(data["predicted_state"].shape, (3, 333, 3000, 2))
                self.assertEqual(len(set(data["run_ids"].tolist())), 333)
        with (OUT / "candidate_pooled_metrics.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 9)
        with (OUT / "candidate_fold_metrics.csv").open(newline="") as stream:
            folds = list(csv.DictReader(stream))
        self.assertEqual(len(folds), 27)

    def test_safety_gates_and_conditional_artifacts(self) -> None:
        selected = self.summary["selected_candidate"]
        conditional = {
            "slip_model_float.npz", "slip_normalization.json",
            "slip_runtime_config.json", "slip_selection_lock.json"}
        present = {path.name for path in OUT.iterdir()}
        if selected is None:
            self.assertFalse(conditional & present)
            self.assertFalse(self.readiness["WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED"])
            self.assertTrue(self.summary["exactly_one_diagnostic_fallback"])
        else:
            self.assertFalse(conditional - present)
            self.assertTrue(selected["mandatory_gates_pass"])
            self.assertTrue(all(int(selected[field]) == 0 for field in SAFETY_FIELDS))
            self.assertTrue(self.summary["final_fit"]["model_reload_parity"])
            self.assertTrue(self.summary["final_fit"]["normalization_reload_parity"])
            self.assertIsNotNone(load_model(OUT / "slip_model_float.npz"))

    def test_immutables_forbidden_policy_and_hash_graph(self) -> None:
        immutable = json.loads((OUT / "immutable_verification.json").read_text())
        self.assertTrue(immutable["all_match_after_training"])
        for relative, expected in EXPECTED_IMMUTABLE_HASHES.items():
            self.assertEqual(file_sha(ROOT / relative), expected)
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertEqual(access["forbidden_access_count"], 0)
        provenance = json.loads((OUT / "provenance.json").read_text())
        for name, expected in provenance["artifact_sha256"].items():
            self.assertEqual(file_sha(OUT / name), expected)
        self.assertEqual(provenance["outer_holdout_final_access_count"], 0)
        self.assertEqual(provenance["starting_checkpoint"], STARTING_CHECKPOINT)

    def test_deferred_work_remains_deferred(self) -> None:
        self.assertFalse(self.readiness["WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED"])
        self.assertFalse(self.readiness["WALKING_V2_INT8_PREPARATION_AUTHORIZED"])
        self.assertTrue(self.readiness["SINK_RUNTIME_DETECTION_DEFERRED"])
        self.assertFalse(self.summary["blind_holdout_generated_or_accessed"])
        self.assertFalse(self.summary["system_int8_vela_e84_hil_work_performed"])


if __name__ == "__main__":
    unittest.main()
