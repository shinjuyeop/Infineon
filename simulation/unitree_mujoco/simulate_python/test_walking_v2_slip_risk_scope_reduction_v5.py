#!/usr/bin/env python3
"""Focused tests for Walking-v2 Slip risk-scope reduction v5."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from walking_v2_slip_corrected_targeted_retraining_v4 import ENDPOINTS, OperatingConfig
from walking_v2_slip_risk_scope_reduction_v5 import (
    CONTRACTS, causal_signal_inventory, contact_scope_signals,
    deterministic_contract_decision, event_scope_eligibility, replay_scoped_state,
    scope_matrix_payload,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "simulation/outputs/walking_v2_slip_risk_scope_reduction_v5"


def raw_scores() -> dict[str, np.ndarray]:
    values = {
        name: np.zeros((len(ENDPOINTS), 2), np.float64)
        for name in ("normal", "early", "actionable", "active", "foot", "proposal")
    }
    values["normal"][:] = 1.0
    return values


class PureScopeReductionTests(unittest.TestCase):
    def test_matrix_is_exact_and_frozen(self) -> None:
        matrix = scope_matrix_payload()
        self.assertEqual([row["contract_id"] for row in matrix["contracts"]], list(CONTRACTS))
        self.assertTrue(matrix["frozen_before_scope_metrics"])
        self.assertFalse(matrix["gates_lowered"])

    def test_inventory_separates_prediction_from_oracle(self) -> None:
        rows = {row["signal"]: row for row in causal_signal_inventory()}
        self.assertTrue(rows["predicted_terrain_state"]["model_predicted"])
        self.assertFalse(rows["predicted_terrain_state"]["oracle_derived"])
        self.assertTrue(rows["M1_state"]["oracle_derived"])
        self.assertFalse(rows["M1_state"]["allowed_runtime_gate"])
        self.assertFalse(rows["first_fall_boundary"]["allowed_runtime_gate"])

    def test_scope_cannot_use_oracle_or_scores(self) -> None:
        samples = np.asarray([90, 100, 110])
        unique = np.ones(200, bool)
        owner = np.zeros(200, np.int8)
        predicted = np.zeros(3, bool)
        result = event_scope_eligibility(samples, 100, 0, unique, owner, predicted, "predictive")
        self.assertFalse(result["eligible"])
        predicted[1] = True
        result = event_scope_eligibility(samples, 100, 0, unique, owner, predicted, "predictive")
        self.assertTrue(result["eligible"])

    def test_unique_owner_is_causal_and_unambiguous(self) -> None:
        loaded = np.zeros((100, 2), bool)
        loaded[10:30, 0] = True
        loaded[20:40, 1] = True
        valid = np.ones_like(loaded)
        touchdown = np.zeros_like(loaded)
        prefall = np.ones(100, bool)
        unique, owner = contact_scope_signals(loaded, valid, touchdown, prefall)
        self.assertEqual(owner[15], 0)
        self.assertFalse(unique[25])
        self.assertEqual(owner[35], 1)

    def test_reactive_gate_precedes_v4_persistence(self) -> None:
        length = 3000
        scores = raw_scores()
        rows = (ENDPOINTS >= 490) & (ENDPOINTS <= 540)
        scores["normal"][rows, 0] = 0.0
        scores["actionable"][rows, 0] = 0.8
        scores["active"][rows, 0] = 0.9
        scores["proposal"][rows, 0] = 0.9
        scores["foot"][rows, 0] = 0.9
        loaded = np.zeros((length, 2), bool)
        loaded[:, 0] = True
        valid = loaded.copy()
        prefall = np.ones(length, bool)
        touchdown = np.zeros_like(loaded)
        episode = np.full((length, 2), -1, np.int32)
        episode[:, 0] = 0
        case_a = ENDPOINTS >= 500
        state, _, _, _ = replay_scoped_state(
            scores, loaded, valid, prefall, touchdown, episode, case_a,
            OperatingConfig(0.45, 0.45, 0.15, 0.10, 0.60, 1, 0.0), True)
        self.assertTrue(state.crossings)
        self.assertGreaterEqual(state.crossings[0]["sample"], 500)

    def test_deterministic_decision_order(self) -> None:
        self.assertEqual(
            deterministic_contract_decision(False, False, False, False, True).chosen_contract,
            CONTRACTS[3])
        self.assertEqual(
            deterministic_contract_decision(False, False, False, False, False).chosen_contract,
            CONTRACTS[4])


class ExecutedScopeReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUT / "summary.json").exists():
            raise AssertionError("run the v5 scope-reduction analysis first")
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.provenance = json.loads((OUT / "provenance.json").read_text())

    def test_required_artifacts_and_no_model_artifacts(self) -> None:
        required = {
            "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
            "artifact_access_log.json", "immutable_verification.json", "exact_input_parity.json",
            "causal_signal_inventory.csv", "scope_matrix.json", "scope_coverage.csv",
            "scope_fold_metrics.csv", "scope_speed_metrics.csv", "scope_foot_metrics.csv",
            "predictive_scope_metrics.csv", "reactive_confirmation_metrics.csv",
            "reactive_latency_metrics.csv", "scope_safety_metrics.csv",
            "convergence_qualification.csv", "contract_comparison.json", "decision.json",
            "provenance.json", "readiness.json", "summary.json", "audit.md",
            "slip_risk_scope_decision_lock.json",
        }
        names = {path.name for path in OUT.iterdir()}
        self.assertFalse(required - names)
        prohibited = {
            "slip_model.npz", "slip_normalization.json", "slip_runtime_config.json",
            "slip_selection_lock.json",
        }
        self.assertFalse(names & prohibited)
        self.assertTrue(all(path.stat().st_size <= 45 * 1024 * 1024 for path in OUT.iterdir()))

    def test_exact_parity_and_scope_denominators(self) -> None:
        parity = json.loads((OUT / "exact_input_parity.json").read_text())
        self.assertTrue(parity["all_exact"])
        self.assertEqual(parity["pareto_row_count"], 504)
        with (OUT / "scope_coverage.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for contract in (CONTRACTS[1], CONTRACTS[2]):
            row = next(item for item in rows if item["contract_id"] == contract and item["row_type"] == "TOTAL")
            self.assertEqual(int(row["original_episode_count"]), 471)
            self.assertEqual(int(row["scoped_episode_count"]), 0)
            self.assertEqual(int(row["excluded_episode_count"]), 471)

    def test_no_score_derived_scope_and_oracle_separation(self) -> None:
        matrix = json.loads((OUT / "scope_matrix.json").read_text())
        encoded = json.dumps(matrix)
        self.assertIn("model score", encoded)
        self.assertIn("oracle Terrain", encoded)
        decision = json.loads((OUT / "decision.json").read_text())
        self.assertFalse(decision["oracle_terrain_substituted"])
        self.assertFalse(decision["score_derived_scope"])

    def test_reactive_safety_and_latency_accounting(self) -> None:
        with (OUT / "reactive_confirmation_metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 504)
        self.assertTrue(all(row["pre_onset_output"] == "0" for row in rows))
        with (OUT / "reactive_latency_metrics.csv").open(newline="") as stream:
            latency = list(csv.DictReader(stream))
        self.assertEqual(len(latency), 504 * 4)
        self.assertEqual({int(row["deadline_ms"]) for row in latency}, {10, 20, 30, 50})

    def test_v4_safety_unique_owner_and_folds(self) -> None:
        with (OUT / "scope_safety_metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        zero = (
            "normal_run_fp", "normal_contact_episode_fp", "too_early", "air_firing",
            "touchdown_firing", "invalid_firing", "post_fall_firing", "latch_carryover",
            "cross_foot_violation", "contact_owner_mismatch", "pre_onset_output",
        )
        self.assertTrue(all(all(int(row[key]) == 0 for key in zero) for row in rows))

    def test_convergence_is_qualified_not_invented(self) -> None:
        with (OUT / "convergence_qualification.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(row["classification"] == "TRAINING_HEALTH_INSUFFICIENT" for row in rows))
        self.assertTrue(all(row["final_gradient_norm"] == "NOT_RECORDED" for row in rows))

    def test_decision_lock_readiness_and_hash_graph(self) -> None:
        self.assertEqual(self.summary["chosen_contract"], CONTRACTS[3])
        self.assertEqual(self.summary["next_step"], "VALIDATE_TERRAIN_ONLY_CASE_REFLEX")
        self.assertFalse(self.summary["fresh_blind_holdout_authorized"])
        readiness = json.loads((OUT / "readiness.json").read_text())
        self.assertTrue(readiness["WALKING_V2_SLIP_TERRAIN_ONLY_ROLE_REQUIRED"])
        self.assertFalse(readiness["WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED"])
        self.assertFalse(readiness["WALKING_V2_INT8_PREPARATION_AUTHORIZED"])
        self.assertTrue(readiness["SINK_RUNTIME_DETECTION_DEFERRED"])
        for name, expected in self.provenance["artifact_sha256"].items():
            actual = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

    def test_forbidden_and_immutable(self) -> None:
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertEqual(access["forbidden_access_count"], 0)
        immutable = json.loads((OUT / "immutable_verification.json").read_text())
        self.assertTrue(immutable["terrain_M1_G0_oracle_match_before"])
        self.assertTrue(self.provenance["immutable_match"])
        self.assertEqual(self.provenance["immutable_before_sha256"], self.provenance["immutable_after_sha256"])


if __name__ == "__main__":
    unittest.main()
