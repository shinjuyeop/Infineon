#!/usr/bin/env python3
"""Focused regression tests for the Walking-v2 Slip failure audit v4."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from run_walking_v2_slip_retraining_failure_audit_v4 import OUTPUT, STARTING_CHECKPOINT
from walking_v2_slip_retraining_failure_audit_v4 import (
    average_precision, balanced_accuracy, binary_auc, contiguous_ranges,
    state_crossings_variant,
)
from walking_v2_slip_targeted_retraining_v3 import ACTIONABLE, NORMAL, OperatingConfig


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / OUTPUT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PureAuditTests(unittest.TestCase):
    def test_first_fall_mask_and_reset_precede_persistence_in_v4(self) -> None:
        probability = np.zeros((40, 2, 4), np.float32)
        probability[:, :, NORMAL] = 1.0
        probability[25:, 0, NORMAL] = 0.0
        probability[25:, 0, ACTIONABLE] = 1.0
        loaded = np.ones((40, 2), bool)
        valid = np.ones((40, 2), bool)
        prefall = np.ones(40, bool)
        prefall[20:] = False
        valid[20:] = False
        config = OperatingConfig(0.99, 5, 0.1, 0.05)
        v0, _ = state_crossings_variant(probability, loaded, valid, prefall, config, "V0")
        v4, resets = state_crossings_variant(probability, loaded, valid, prefall, config, "V4")
        self.assertEqual(len(v0), 1)
        self.assertEqual(len(v4), 0)
        self.assertTrue(any(row.get("reset_reason") == "first_fall_boundary" for row in resets))

    def test_invalid_samples_cannot_accumulate_in_v3(self) -> None:
        probability = np.zeros((30, 2, 4), np.float32)
        probability[:, :, NORMAL] = 1.0
        probability[15:, 0, NORMAL] = 0.0
        probability[15:, 0, ACTIONABLE] = 1.0
        loaded = np.ones((30, 2), bool)
        valid = np.ones((30, 2), bool)
        valid[14:] = False
        prefall = np.ones(30, bool)
        config = OperatingConfig(0.99, 3, 0.1, 0.05)
        v0, _ = state_crossings_variant(probability, loaded, valid, prefall, config, "V0")
        v3, _ = state_crossings_variant(probability, loaded, valid, prefall, config, "V3")
        self.assertEqual(len(v0), 1)
        self.assertEqual(len(v3), 0)

    def test_metric_helpers_and_range_encoding(self) -> None:
        target = np.array([0, 0, 1, 1], bool)
        score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(binary_auc(target, score), 1.0)
        self.assertEqual(average_precision(target, score), 1.0)
        self.assertEqual(balanced_accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])), 1.0)
        self.assertEqual(contiguous_ranges([(1,), (1,), (2,)]), [(0, 2, (1,)), (2, 3, (2,))])


class ExecutedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUT / "summary.json").exists():
            raise AssertionError("execute failure-audit runner before artifact tests")
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.parity = json.loads((OUT / "exact_replay_parity.json").read_text())

    def test_required_artifacts_and_size_limit(self) -> None:
        required = {
            "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
            "artifact_access_log.json", "immutable_verification.json", "exact_replay_parity.json",
            "unified_sample_ledger.csv", "episode_reconciliation.csv",
            "episode_duration_gap_metrics.csv", "shadow_comparison.csv", "training_health.csv",
            "label_distribution.csv", "normalization_audit.json", "raw_score_metrics.csv",
            "stage_rejection_metrics.csv", "operating_point_pareto.csv", "violation_ledger.csv",
            "violation_reconciliation.json", "state_mask_variant_metrics.csv",
            "correctness_fixes.json", "root_cause_classification.json", "provenance.json",
            "readiness.json", "summary.json", "audit.md",
        }
        names = {path.name for path in OUT.iterdir()}
        self.assertFalse(required - names)
        self.assertTrue(all(path.stat().st_size <= 45 * 1024 * 1024 for path in OUT.iterdir()))

    def test_exact_replay_and_label_fold_parity(self) -> None:
        self.assertEqual(self.parity["status"], "PASS")
        self.assertTrue(self.parity["out_of_fold_raw_prediction_exact"])
        self.assertTrue(self.parity["threshold_state_metrics_exact"])
        self.assertTrue(self.parity["operating_configs_exact"])
        self.assertTrue(self.parity["label_count_parity"])
        self.assertTrue(self.parity["labels_recomputed_from_frozen_oracle_ledger"])
        self.assertEqual(self.parity["stored_vs_recomputed_label_count_mismatch_count"], 0)
        self.assertTrue(all(row["maximum_actionable_absolute_error"] == 0.0
                            for row in self.parity["family_parity"]))

    def test_trace_run_mapping_and_frozen_data_contract(self) -> None:
        immutable = json.loads((OUT / "immutable_verification.json").read_text())
        audit = immutable["data_contract_audit"]
        self.assertEqual(audit["fold_rows"], 333)
        self.assertEqual(audit["fold_count"], 3)
        self.assertTrue(audit["run_mapping_unique"])
        self.assertTrue(audit["trace_round_trip_verified"])
        self.assertTrue(audit["fusion20_channel_order_verified"])
        self.assertTrue(audit["left_right_canonicalization_verified"])
        self.assertEqual(audit["sample_rate_hz"], 1000)
        self.assertEqual(audit["maximum_causal_window_ms"], 200)
        self.assertEqual(audit["future_label_horizon_ms"], 100)
        self.assertEqual(audit["eligibility_counts"]["CALIBRATION_ONLY_DO_NOT_TRAIN"], 42)
        self.assertEqual(audit["eligibility_counts"]["FAILED_SOURCE_DIAGNOSTIC_ONLY"], 27)
        self.assertEqual(audit["quarantined_count"], 216)

    def test_episode_denominator_reconciles_to_471(self) -> None:
        episode = self.summary["episode_reconciliation"]
        self.assertEqual(episode["eligible_runs"], 333)
        self.assertEqual(episode["physical_onset_bearing_episodes"], 471)
        self.assertEqual(episode["actionable_episodes"], 471)
        with (OUT / "episode_reconciliation.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(row["row_type"] == "EPISODE" for row in rows), 471)

    def test_violation_reconciliation_and_variants(self) -> None:
        violation = self.summary["violation_reconciliation"]
        self.assertEqual(violation["normal_run_fp"], 0)
        self.assertEqual(violation["normal_contact_episode_fp"], 57)
        self.assertEqual(violation["invalid_count"], 52)
        self.assertEqual(violation["post_fall_count"], 52)
        self.assertEqual(violation["invalid_postfall_overlap_count"], 52)
        self.assertTrue(violation["invalid_and_postfall_exact_same_events"])
        variants = {row["variant"]: row for row in self.summary["R1_state_mask_variants"]}
        self.assertEqual(variants["V0"]["post_fall"], 52)
        self.assertEqual(variants["V4"]["post_fall"], 0)
        self.assertEqual(variants["V4"]["invalid"], 0)

    def test_root_cause_authorization_is_fail_closed(self) -> None:
        root = self.summary["root_cause"]
        self.assertEqual(root["primary_root_cause"], "MULTIPLE_INTERACTING_CAUSES")
        self.assertTrue(root["corrected_rerun_authorized"])
        self.assertFalse(root["model_redesign_authorized"])
        self.assertFalse(root["risk_scope_reduction_required"])
        self.assertFalse(self.summary["candidate_or_selection_lock_created"])
        self.assertFalse(self.summary["blind_holdout_created_or_accessed"])

    def test_shadow_comparison_exposes_recipe_discontinuity(self) -> None:
        with (OUT / "shadow_comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        first = next(row for row in rows if row["comparison"] == "FIRST_DIVERGENCE")
        self.assertEqual(first["status"], "MATERIAL_DIVERGENCE_BEFORE_NORMALIZATION")
        unavailable = next(row for row in rows if row["comparison"] == "B")
        self.assertEqual(unavailable["status"], "EXACT_COMPARISON_IMPOSSIBLE_NOT_FABRICATED")
        current_previous = next(row for row in rows if row["comparison"] == "C")
        self.assertEqual(current_previous["status"], "EXACT_COMPARISON_IMPOSSIBLE_NOT_FABRICATED")
        self.assertIn("undeclared training job", current_previous["reason"])
        stages = [row for row in rows if row["comparison"] == "PIPELINE_STAGE"]
        self.assertEqual([int(row["stage_order"]) for row in stages], list(range(1, 15)))

    def test_normalization_training_health_and_stage_decomposition(self) -> None:
        normalization = json.loads((OUT / "normalization_audit.json").read_text())
        self.assertEqual(normalization["status"], "PASS")
        self.assertTrue(normalization["train_fold_only"])
        self.assertEqual(len(normalization["entries"]), 9)
        self.assertTrue(all(row["outer_run_overlap_count"] == 0
                            and row["finite_mean"] and row["finite_scale"]
                            and row["near_zero_scale_count"] == 0
                            and row["exploding_scale_count"] == 0
                            for row in normalization["entries"]))
        with (OUT / "training_health.csv").open(newline="") as stream:
            health = list(csv.DictReader(stream))
        self.assertEqual(len(health), 27)
        self.assertTrue(all(row["diagnostic_model_reload_parity"] == "True"
                            and row["normalization_reload_parity"] == "True"
                            and row["threshold_state_metric_parity"] == "True" for row in health))
        stages = self.summary["R1_stage_decomposition"]
        self.assertEqual(stages["raw_actionable_argmax"]["detected_count"], 460)
        self.assertEqual(stages["actionable_proposal"]["detected_count"], 75)
        self.assertEqual(stages["persistence"]["detected_count"], 11)
        self.assertEqual(stages["episode_detection"]["detected_count"], 6)

    def test_diagnostic_pareto_never_selects_and_cannot_meet_safety_recall(self) -> None:
        with (OUT / "operating_point_pareto.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertTrue(all(row["diagnostic_only"] == "True"
                            and row["selected_or_frozen"] == "False" for row in rows))
        self.assertTrue(all(row["hysteresis_effect"]
                            == "INERT_WITH_FIRST_CROSSING_PER_CONTACT_AND_NO_LATCH" for row in rows))
        by_candidate = {row["candidate_id"] for row in rows}
        for candidate in by_candidate:
            candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
            self.assertLess(float(candidate_rows[0]["maximum_recall_with_normal_run_fp_zero"]), 0.8)
            self.assertLess(float(candidate_rows[0]["maximum_recall_with_contact_episode_fp_zero"]), 0.8)

    def test_forbidden_immutable_and_artifact_hash_graph(self) -> None:
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertEqual(access["forbidden_access_count"], 0)
        immutable = json.loads((OUT / "immutable_verification.json").read_text())
        self.assertTrue(immutable["terrain_M1_G0_oracle_match_after"])
        provenance = json.loads((OUT / "provenance.json").read_text())
        self.assertEqual(set(provenance["artifact_sha256"]),
                         {path.name for path in OUT.iterdir() if path.name != "provenance.json"})
        for name, expected in provenance["artifact_sha256"].items():
            self.assertEqual(digest(OUT / name), expected)
        self.assertEqual(provenance["outer_holdout_final_access_count"], 0)
        self.assertEqual(provenance["starting_checkpoint"], STARTING_CHECKPOINT)

    def test_forbidden_policy_and_access_log_are_fail_closed(self) -> None:
        policy = json.loads((OUT / "forbidden_path_policy.json").read_text())
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertTrue(policy["fail_closed"])
        self.assertFalse(policy["outer_holdout_final_access_authorized"])
        self.assertEqual(access["forbidden_access_count"], 0)
        self.assertTrue(all(row["decision"] == "ALLOWED" for row in access["accesses"]))

    def test_deferred_authorizations(self) -> None:
        readiness = self.summary["readiness"]
        self.assertFalse(readiness["WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED"])
        self.assertFalse(readiness["WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED"])
        self.assertFalse(readiness["WALKING_V2_INT8_PREPARATION_AUTHORIZED"])
        self.assertTrue(readiness["SINK_RUNTIME_DETECTION_DEFERRED"])


if __name__ == "__main__":
    unittest.main()
