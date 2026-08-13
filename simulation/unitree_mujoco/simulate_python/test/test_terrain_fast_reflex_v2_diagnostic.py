from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_terrain_fast_reflex_v2_diagnostic import classify_pre_onset_firing  # noqa: E402


class FastReflexV2DiagnosticTest(unittest.TestCase):
    def test_classification_preserves_canonical_outcome(self) -> None:
        self.assertEqual(classify_pre_onset_firing(4, 10, True), "incipient_slip_candidate")
        self.assertEqual(classify_pre_onset_firing(4, None, True), "true_false_alarm")
        self.assertEqual(classify_pre_onset_firing(10, 4, True), "ambiguous")
        self.assertEqual(classify_pre_onset_firing(4, 10, False), "ambiguous")


if __name__ == "__main__":
    unittest.main()
