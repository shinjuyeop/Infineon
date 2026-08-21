#!/usr/bin/env python3
"""Focused tests for corrected Walking-v2 Slip retraining v4."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from run_walking_v2_slip_corrected_targeted_retraining_v4 import (
    OUTPUT, PRETRAINING_FILES, STARTING_CHECKPOINT, evaluate,
)
from run_walking_v2_slip_targeted_retraining_v3 import RunRecord
from walking_v2_slip_corrected_targeted_retraining_v4 import (
    ENDPOINTS, HEAD_NAMES, OperatingConfig, corrected_v4_state,
    derived_runtime_telemetry, operating_grid,
)
from walking_v2_slip_redesign_iteration_v2 import FAMILY_SPECS


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / OUTPUT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(default_normal: float = 1.0) -> dict[str, np.ndarray]:
    result = {name: np.zeros((len(ENDPOINTS), 2), np.float64) for name in HEAD_NAMES}
    result["normal"][:] = default_normal
    return result


class PureCorrectedRetrainingTests(unittest.TestCase):
    def test_exact_recipe_and_preregistered_grid(self) -> None:
        spec = FAMILY_SPECS["S4-C"]
        self.assertEqual(spec["history_ms"], 200)
        self.assertEqual(spec["projection_width"], 40)
        self.assertTrue(spec["two_stage"])
        grid = operating_grid()
        self.assertEqual(len(grid), 84)
        self.assertIn(OperatingConfig(0.65, 0.65, 0.25, 0.20, 0.60, 2, 0.05), grid)
        self.assertEqual(len({row.config_id for row in grid}), len(grid))

    def test_runtime_telemetry_is_causal_and_deterministic(self) -> None:
        loaded = np.zeros((100, 2), bool)
        loaded[10:70, 0] = True
        loaded[30:55, 1] = True
        ages, episodes, phase = derived_runtime_telemetry(loaded)
        self.assertEqual(ages[10, 0], 1)
        self.assertEqual(ages[69, 0], 60)
        self.assertEqual(episodes[10, 0], 0)
        self.assertEqual(phase[10, 0], 1)  # touchdown
        self.assertEqual(phase[20, 0], 2)  # loading
        self.assertEqual(phase[45, 0], 3)  # mid stance
        self.assertEqual(phase[65, 0], 4)  # push off

    def test_invalid_and_postfall_cannot_increment_persistence(self) -> None:
        values = scores()
        values["normal"][:] = 0.0
        for name in ("actionable", "proposal", "foot"):
            values[name][:3, 0] = 1.0
        loaded = np.ones((3000, 2), bool)
        valid = np.ones((3000, 2), bool)
        valid[ENDPOINTS[1], 0] = False
        prefall = np.ones(3000, bool)
        prefall[ENDPOINTS[2]:] = False
        touchdown = np.zeros((3000, 2), bool)
        episode = np.zeros((3000, 2), np.int32)
        output = corrected_v4_state(
            values, loaded, valid, prefall, touchdown, episode,
            OperatingConfig(0.65, 0.65, 0.25, 0.20, 0.60, 2, 0.05))
        self.assertFalse(output.crossings)
        self.assertEqual(int(np.sum(output.reset_reason == "first_fall_hard_reset")), 2)
        self.assertTrue(np.all(output.persistence_count[2:] == 0))

    def test_contact_loss_and_new_touchdown_reset_state(self) -> None:
        values = scores(0.0)
        for name in ("actionable", "proposal", "foot"):
            values[name][:, 0] = 1.0
        loaded = np.ones((3000, 2), bool)
        loaded[251:301, 0] = False
        valid = loaded.copy()
        prefall = np.ones(3000, bool)
        touchdown = np.zeros((3000, 2), bool)
        episode = np.zeros((3000, 2), np.int32)
        output = corrected_v4_state(
            values, loaded, valid, prefall, touchdown, episode,
            OperatingConfig(0.65, 0.65, 0.25, 0.20, 0.60, 2, 0.05))
        self.assertTrue(np.any(output.reset_reason[:, 0] == "contact_loss"))
        self.assertGreaterEqual(np.sum(output.reset_reason[:, 0] == "new_touchdown"), 2)

    def test_affected_foot_denominator_includes_incorrect_detection(self) -> None:
        length = 3000
        loaded = np.ones((length, 2), bool)
        labels = np.zeros((length, 2), np.int8)
        physical_episode = np.zeros((length, 2), np.int32)
        record = RunRecord(
            "synthetic", 0, "test", "positive", 0.10, "left", "loading", "test", "",
            "v0", np.zeros((length, 20), np.float32), loaded, loaded,
            physical_episode, np.zeros((length, 2), bool), np.ones(length, bool),
            np.ones((length, 2), bool), np.zeros((length, 2), bool), labels,
            np.zeros((length, 2), np.int32),
            [{"foot": 0, "episode_id": 0, "onset_sample": 500}],
        )
        values = {name: np.zeros((1, len(ENDPOINTS), 2), np.float64) for name in HEAD_NAMES}
        values["normal"][:] = 1.0
        mask = (ENDPOINTS >= 400) & (ENDPOINTS <= 600)
        values["normal"][0, mask, 1] = 0.0
        for name in ("actionable", "proposal", "foot"):
            values[name][0, mask, 1] = 1.0
        metric, _, _, _ = evaluate(
            [record], [0], values,
            OperatingConfig(0.65, 0.65, 0.25, 0.20, 0.60, 2, 0.05),
            "synthetic", "test")
        self.assertEqual(metric["affected_foot_count"], 1)
        self.assertEqual(metric["affected_foot_correct"], 0)
        self.assertEqual(metric["affected_foot_accuracy"], 0.0)
        self.assertEqual(metric["cross_foot_violation"], 1)


class ExecutedCorrectedRetrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUT / "summary.json").exists():
            raise AssertionError("execute corrected retraining runner before integration tests")
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.provenance = json.loads((OUT / "provenance.json").read_text())

    def test_required_artifacts_and_size_limit(self) -> None:
        required = {
            "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
            "artifact_access_log.json", "immutable_verification.json",
            "recipe_reconstruction.json", "candidate_matrix.json",
            "operating_point_policy.json", "selection_policy.json", "eligibility_audit.csv",
            "nested_fold_audit.csv", "label_distribution.csv", "episode_distribution.csv",
            "training_health.csv", "weighting_comparison.csv", "candidate_fold_metrics.csv",
            "candidate_pooled_metrics.csv", "candidate_speed_metrics.csv",
            "candidate_foot_metrics.csv", "stage_rejection_metrics.csv",
            "operating_point_pareto.csv", "crossing_reconciliation.csv",
            "timing_metrics.csv", "reset_audit.csv", "resource_report.json",
            "provenance.json", "readiness.json", "summary.json", "audit.md",
        }
        names = {path.name for path in OUT.iterdir()}
        self.assertFalse(required - names)
        self.assertTrue(all(path.stat().st_size <= 45 * 1024 * 1024 for path in OUT.iterdir()))
        self.assertEqual(len([name for name in names if name.startswith("oof_raw_head_scores_")]), 6)

    def test_recipe_grid_and_pretraining_barrier(self) -> None:
        recipe = json.loads((OUT / "recipe_reconstruction.json").read_text())
        policy = json.loads((OUT / "operating_point_policy.json").read_text())
        self.assertTrue(recipe["exact_reconstruction_possible"])
        self.assertFalse(recipe["unauthorized_or_approximated_deviations"])
        self.assertEqual(policy["configuration_count"], 84)
        self.assertTrue(policy["fixed_before_training"])
        self.assertTrue(policy["contains_exact_prior_S4C_point"])
        self.assertTrue(self.provenance["pretraining_barrier_unchanged"])
        self.assertEqual(self.provenance["pretraining_artifact_sha256"],
                         self.provenance["pretraining_artifact_sha256_after"])
        self.assertEqual(set(self.provenance["pretraining_artifact_sha256"]),
                         set(PRETRAINING_FILES))

    def test_exact_nested_evaluation_and_denominator(self) -> None:
        with (OUT / "candidate_pooled_metrics.csv").open(newline="") as stream:
            pooled = list(csv.DictReader(stream))
        with (OUT / "candidate_fold_metrics.csv").open(newline="") as stream:
            folds = list(csv.DictReader(stream))
        self.assertEqual(len(pooled), 6)
        self.assertEqual(len(folds), 18)
        self.assertTrue(all(int(row["actionable_episode_count"]) == 471 for row in pooled))
        self.assertEqual({row["variant"] for row in pooled}, {"F0", "F1"})
        self.assertEqual({int(row["seed"]) for row in pooled}, {202608241, 202608242, 202608243})
        self.assertTrue(all(row["evaluation_scope"].startswith("FROZEN_OUTER_") for row in folds))

    def test_leakage_normalization_and_hard_negative_isolation(self) -> None:
        with (OUT / "nested_fold_audit.csv").open(newline="") as stream:
            nested = list(csv.DictReader(stream))
        with (OUT / "training_health.csv").open(newline="") as stream:
            health = list(csv.DictReader(stream))
        with (OUT / "hard_negative_audit.csv").open(newline="") as stream:
            hard = list(csv.DictReader(stream))
        self.assertEqual(len(nested), 333)
        self.assertTrue(all(row["run_pair_variation_episode_leakage"] == "False" for row in nested))
        self.assertEqual(len(health), 18)
        self.assertTrue(all(row["outer_run_overlap_count"] == "0"
                            and row["hard_negative_outer_overlap_count"] == "0"
                            and row["normalization_train_only"] == "True"
                            and row["normalization_finite"] == "True" for row in health))
        self.assertTrue(all(row["outer_validation_member"] == "False" for row in hard))

    def test_v4_safety_invariants_and_affected_denominator(self) -> None:
        with (OUT / "candidate_pooled_metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            self.assertEqual(int(row["invalid_firing"]), 0)
            self.assertEqual(int(row["post_fall_firing"]), 0)
            self.assertEqual(int(row["air_firing"]), 0)
            self.assertEqual(int(row["touchdown_firing"]), 0)
            self.assertEqual(int(row["latch_carryover"]), 0)
            self.assertEqual(int(row["affected_foot_count"]),
                             int(row["affected_foot_correct"]) + int(row["cross_foot_violation"]))
        with (OUT / "reset_audit.csv").open(newline="") as stream:
            reset = list(csv.DictReader(stream))
        self.assertTrue(all(row["invalid_persistence_increment_count"] == "0"
                            and row["postfall_persistence_increment_count"] == "0"
                            and row["contact_owner_mutation_count"] == "0"
                            and row["latch_carryover_count"] == "0" for row in reset))

    def test_pareto_is_diagnostic_and_grid_frozen(self) -> None:
        with (OUT / "operating_point_pareto.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6 * 84)
        self.assertTrue(all(row["diagnostic_only"] == "True"
                            and row["selected_or_frozen"] == "False"
                            and row["grid_frozen_before_scores"] == "True" for row in rows))

    def test_conditional_final_artifacts_and_authorization(self) -> None:
        selected = self.summary["candidate_pass_count"] > 0
        conditional = {
            "slip_model.npz", "slip_normalization.json", "slip_runtime_config.json",
            "slip_selection_lock.json",
        }
        names = {path.name for path in OUT.iterdir()}
        self.assertEqual(conditional <= names, selected)
        self.assertEqual(self.summary["final_model_fitted"], selected)
        self.assertEqual(self.summary["slip_selection_lock_created"], selected)
        self.assertEqual(self.summary["fresh_blind_holdout_authorized"], selected)
        if selected:
            self.assertTrue(self.summary["reload_parity"]["all_exact"])

    def test_exclusions_forbidden_and_immutable_hashes(self) -> None:
        with (OUT / "eligibility_audit.csv").open(newline="") as stream:
            eligibility = list(csv.DictReader(stream))
        excluded = [row for row in eligibility if row["eligibility"] in {
            "FAILED_SOURCE_DIAGNOSTIC_ONLY", "CALIBRATION_ONLY_DO_NOT_TRAIN", "QUARANTINED"}]
        self.assertTrue(excluded)
        self.assertTrue(all(row["included"] == "False" and row["exclusion_enforced"] == "True"
                            for row in excluded))
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertEqual(access["forbidden_access_count"], 0)
        self.assertTrue(access["frozen_before_first_training_job"])
        self.assertTrue(all(row["decision"] == "ALLOWED" for row in access["accesses"]))
        immutable = json.loads((OUT / "immutable_verification.json").read_text())
        self.assertTrue(immutable["terrain_M1_G0_oracle_match_before"])
        self.assertTrue(self.provenance["immutable_match"])
        self.assertEqual(self.provenance["immutable_before_sha256"],
                         self.provenance["immutable_after_sha256"])

    def test_artifact_hash_graph_and_deferred_work(self) -> None:
        self.assertEqual(self.provenance["starting_checkpoint"], STARTING_CHECKPOINT)
        for name, expected in self.provenance["artifact_sha256"].items():
            self.assertEqual(digest(OUT / name), expected)
        self.assertEqual(self.provenance["outer_holdout_final_access_count"], 0)
        self.assertEqual(self.provenance["blind_artifact_count"], 0)
        readiness = self.summary["readiness"]
        self.assertFalse(readiness["WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED"])
        self.assertFalse(readiness["WALKING_V2_INT8_PREPARATION_AUTHORIZED"])
        self.assertTrue(readiness["SINK_RUNTIME_DETECTION_DEFERRED"])


if __name__ == "__main__":
    unittest.main()
