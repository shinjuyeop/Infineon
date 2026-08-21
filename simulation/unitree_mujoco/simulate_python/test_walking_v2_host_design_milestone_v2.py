"""Regression tests for the Walking-v2 host design milestone v2."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from walking_v2_host_design_milestone_v2 import (
    LinearTerrainModel, TransitionPolicy, acquisition_matrix,
    case_a_transition_timeline, causal_contact_telemetry, causal_g0_owner,
    runtime_contract, transition_policy_matrix,
)


REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "simulation/outputs/walking_v2_host_design_milestone_v2"


class HostDesignMilestoneV2Tests(unittest.TestCase):
    def test_frozen_acquisition_matrix(self) -> None:
        rows = acquisition_matrix()
        self.assertEqual(len(rows), 420)
        self.assertEqual({row.case_id for row in rows}, {"A", "B", "C", "D", "N", "S"})
        self.assertEqual({row.speed_mps for row in rows}, {0.10, 0.15, 0.20})
        self.assertEqual({row.variation_index for row in rows}, set(range(5)))
        self.assertEqual(len({row.run_id for row in rows}), 420)
        self.assertEqual(len({row.pair_id for row in rows}), 420)
        self.assertEqual(len({row.source_id for row in rows}), 420)
        self.assertEqual(len({row.seed for row in rows}), 420)

    def test_contact_telemetry_and_owner_are_causal(self) -> None:
        loaded = np.asarray([
            [0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0],
        ], bool)
        touchdown, age, episode = causal_contact_telemetry(loaded)
        self.assertEqual(age[:, 0].tolist(), [0, 1, 2, 3, 0, 0, 1])
        self.assertEqual(episode[:, 0].tolist(), [-1, 0, 0, 0, -1, -1, 1])
        self.assertTrue(touchdown[1, 0]); self.assertTrue(touchdown[3, 1])
        owner = causal_g0_owner(loaded, age, np.ones(len(loaded), bool), minimum_age_ms=1)
        self.assertEqual(owner.tolist(), [-1, 0, 0, -1, 1, -1, 0])

    def test_transition_state_has_hard_direct_firewall(self) -> None:
        sample_count = 230
        endpoints = np.asarray((199, 209, 219), np.int32)
        loaded = np.zeros((sample_count, 2), bool); loaded[150:, 0] = True
        _, age, episode = causal_contact_telemetry(loaded)
        probability = np.zeros((3, 2, 4), np.float32)
        probability[:, :, 0] = 0.95
        probability[1:, 0, 0] = 0.01; probability[1:, 0, 2] = 0.98
        timeline = case_a_transition_timeline(
            probability, endpoints, loaded, age, episode,
            np.ones(sample_count, bool), TransitionPolicy(0.8, 1, 5),
            direct_authorized=False,
        )
        self.assertEqual(len(timeline.emissions), 1)
        self.assertEqual(timeline.emissions[0]["sample"], 209)
        self.assertFalse(np.any(timeline.direct_reflex))

    def test_policy_and_runtime_contract_are_frozen(self) -> None:
        self.assertEqual(len(transition_policy_matrix()), 24)
        contract = runtime_contract()
        self.assertFalse(contract["authority"]["direct_reflex"])
        self.assertFalse(contract["authority"]["recovery_actuation"])
        self.assertEqual(contract["sample_rate_hz"], 1000)
        prohibited = " ".join(contract["prohibited_runtime_inputs"])
        self.assertIn("physical Slip oracle", prohibited)
        self.assertIn("future", prohibited)

    def test_linear_model_roundtrip_exact(self) -> None:
        model = LinearTerrainModel(
            "T1", 200, np.zeros(3), np.ones(3),
            np.arange(12, dtype=float).reshape(4, 3) / 10,
            np.arange(4, dtype=float), np.arange(4, dtype=np.int8),
        )
        values = np.arange(15, dtype=float).reshape(5, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(str(path)); loaded = LinearTerrainModel.load(str(path))
            self.assertTrue(np.array_equal(model.probabilities(values), loaded.probabilities(values)))

    def test_generated_lock_when_present(self) -> None:
        if not OUTPUT.exists():
            self.skipTest("generated milestone output not present")
        lock = json.loads((OUTPUT / "design_lock.json").read_text())
        summary = json.loads((OUTPUT / "summary.json").read_text())
        self.assertTrue(lock["immutable"])
        self.assertFalse(lock["direct_reflex_enabled"])
        self.assertEqual(summary["next_step"], "ACQUIRE_FRESH_BLIND_HOST_HOLDOUT")
        with np.load(OUTPUT / "host_replay.npz", allow_pickle=False) as archive:
            self.assertFalse(np.any(archive["direct_reflex"]))
            self.assertFalse(np.any(archive["recovery_actuation"]))


if __name__ == "__main__":
    unittest.main()
