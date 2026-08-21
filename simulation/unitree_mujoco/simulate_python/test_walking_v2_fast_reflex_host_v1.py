#!/usr/bin/env python3
"""Regression and executed-artifact tests for Walking-v2 host v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from walking_v2_fast_reflex_host_v1 import (
    TerrainVerifierConfig, causal_runtime_contract, direct_authority_gate,
    safe_ratio, terrain_candidate_matrix, terrain_case_a_timeline,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "simulation/outputs/walking_v2_fast_reflex_host_v1"


def synthetic_timeline(*, dual_at_transition: bool = False):
    endpoints = np.arange(19, 120, 10)
    sample_count = 140
    probability = np.zeros((len(endpoints), 2, 4), np.float64)
    probability[..., 0] = 0.99
    probability[3:, 0, 0] = 0.0
    probability[3:, 0, 2] = 0.99
    loaded = np.zeros((sample_count, 2), bool)
    loaded[:, 0] = True
    if dual_at_transition:
        loaded[49:80, 1] = True
    age = np.column_stack((np.arange(sample_count), np.arange(sample_count)))
    touchdown = loaded & (age <= 10)
    runtime_episode = np.zeros((sample_count, 2), np.int32)
    physical_episode = np.zeros((sample_count, 2), np.int32)
    quality = np.ones(sample_count, bool)
    return terrain_case_a_timeline(
        probability, endpoints, loaded, age, touchdown, runtime_episode,
        physical_episode, quality, TerrainVerifierConfig(2, 3, 0.9),
    )


class PureHostTests(unittest.TestCase):
    def test_candidate_matrix_is_bounded_and_deterministic(self) -> None:
        first = terrain_candidate_matrix()
        self.assertEqual(first, terrain_candidate_matrix())
        self.assertEqual(len(first), 27)
        self.assertEqual(len({row.config_id for row in first}), 27)

    def test_zero_denominator_is_not_performance(self) -> None:
        self.assertIsNone(safe_ratio(0, 0))
        self.assertEqual(safe_ratio(1, 2), 0.5)

    def test_case_a_is_one_shot_and_contact_local(self) -> None:
        result = synthetic_timeline()
        self.assertEqual(len(result.emissions), 1)
        self.assertEqual(result.emissions[0]["transition_case"], "A")
        self.assertTrue(result.emissions[0]["unique_g0_owner"])
        self.assertTrue(np.any(result.case_a_active))

    def test_pending_case_waits_for_unique_g0_owner(self) -> None:
        result = synthetic_timeline(dual_at_transition=True)
        self.assertEqual(len(result.emissions), 1)
        self.assertGreaterEqual(result.emissions[0]["sample"], 79)

    def test_direct_gate_is_strict(self) -> None:
        base = {
            "normal_contact_fp": 0, "too_early": 0, "invalid_output": 0,
            "post_fall_output": 0, "wrong_case_output": 0, "duplicate_output": 0,
            "latch_carryover": 0, "recall_within_20ms": 0.85,
            "speed_recall_within_20ms": {"0.10": 0.8, "0.15": 0.8, "0.20": 0.8},
            "output_count": 10,
        }
        self.assertTrue(direct_authority_gate(base))
        self.assertFalse(direct_authority_gate({**base, "normal_contact_fp": 1}))
        self.assertFalse(direct_authority_gate({**base, "recall_within_20ms": 0.79}))

    def test_contract_has_authority_firewall_and_no_oracle(self) -> None:
        contract = causal_runtime_contract()
        self.assertFalse(contract["outputs"]["direct_reflex"])
        self.assertEqual(contract["authority"]["Sink"], "SINK_RUNTIME_DETECTION_DEFERRED")
        prohibited = " ".join(contract["prohibited_runtime_inputs"])
        self.assertIn("fall oracle", prohibited)
        self.assertIn("physical Slip oracle", prohibited)


@unittest.skipUnless(OUT.exists(), "executed output not generated")
class ExecutedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.provenance = json.loads((OUT / "provenance.json").read_text())

    def test_required_artifacts_and_size_limit(self) -> None:
        required = {
            "protocol.json", "input_allowlist.json", "forbidden_path_policy.json",
            "artifact_access_log.json", "causal_runtime_contract.json",
            "terrain_candidate_metrics.csv", "alternative_comparison.csv",
            "advisory_metrics.json", "failure_analysis.json", "resource_estimate.json",
            "readiness_limitations.json", "host_replay.npz", "slip_advisory_model.npz",
            "slip_advisory_training.json", "design_lock.json", "summary.json",
            "audit.md", "test_results.json", "provenance.json",
        }
        self.assertFalse(required - {path.name for path in OUT.iterdir()})
        self.assertTrue(all(path.stat().st_size < 45 * 1024 * 1024 for path in OUT.iterdir()))

    def test_direct_authority_is_fail_closed(self) -> None:
        self.assertEqual(self.summary["selected_architecture"],
                         "MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY")
        self.assertFalse(self.summary["direct_reflex_authorized"])
        with np.load(OUT / "host_replay.npz", allow_pickle=False) as archive:
            self.assertFalse(np.any(archive["direct_reflex"]))

    def test_forbidden_and_hardware_boundaries(self) -> None:
        access = json.loads((OUT / "artifact_access_log.json").read_text())
        self.assertEqual(access["forbidden_access_count"], 0)
        self.assertEqual(access["blocked_access_count"], 0)
        self.assertFalse(self.summary["hardware_e84_hil_executed"])
        self.assertEqual(self.summary["sink"], "SINK_RUNTIME_DETECTION_DEFERRED")

    def test_lock_and_hash_graph(self) -> None:
        lock = json.loads((OUT / "design_lock.json").read_text())
        self.assertTrue(lock["immutable"])
        self.assertEqual(lock["lock_type"], "development_design_lock_not_blind_or_safety")
        for name, expected in self.provenance["artifact_sha256"].items():
            actual = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

    def test_no_zero_denominator_claim(self) -> None:
        metrics = json.loads((OUT / "advisory_metrics.json").read_text())
        self.assertGreater(metrics["case_a_scope"]["predictive_episode_denominator"], 0)
        self.assertGreater(metrics["case_a_scope"]["reactive_20ms_episode_denominator"], 0)


if __name__ == "__main__":
    unittest.main()
