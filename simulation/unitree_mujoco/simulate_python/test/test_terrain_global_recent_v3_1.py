"""Reservation invariants for the Terrain v3.1 dual-pooling ablation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_terrain_global_recent_v3_1 import reserved_transition_split  # noqa: E402


class GlobalRecentV31Test(unittest.TestCase):
    def test_run_index_reservation_is_disjoint_and_covers_each_direction(self):
        rows = []
        for case in "ABCD":
            for family in ("f0", "f1"):
                for realization in range(2):
                    for run in range(4):
                        rows.append({"split": "train", "case_id": case, "run_index": str(run), "run_id": f"{case}-{family}-{realization}-{run}"})
        train, selection = reserved_transition_split(rows)
        self.assertEqual(len(selection), 16)
        self.assertEqual(len(train), 48)
        self.assertTrue(all(rows[index]["run_index"] == "3" for index in selection))
        self.assertTrue(all(rows[index]["run_index"] != "3" for index in train))
        self.assertEqual({rows[index]["case_id"] for index in selection}, set("ABCD"))


if __name__ == "__main__":
    unittest.main()
