# Walking-v2 Bilateral Bounded Training

## Scope

Development-only float training used the fresh 72/48 bilateral train/validation
split. Existing outer, holdout, spatial, final-test, production, System, INT8,
Vela, E84, and physical-hardware content was not opened or modified. Sink has
no runtime head; Sand means `SAND_TERRAIN_CAUTION` only.

## Decision

- Terrain READY: False
- Slip READY: False
- Selection lock: False
- New blind holdout runs: 0
- Diagnostic Terrain fallback: {'architecture': 'T2', 'seed': 202608213, 'overall_accuracy': 0.6549217977789407, 'macro_accuracy': 0.6090079361501065, 'worst_class_recall': 0.3611525086934923, 'majority_class_prediction_rate': 0.28614008941877794, 'gate_pass': False}
- Diagnostic Slip fallback: {'architecture': 'S1', 'seed': 202608211, 'threshold': 0.999, 'persistence_ms': 30, 'hysteresis': 0.05, 'valid_ice_run_coverage': 1.0, 'physical_episode_recall': 0.2463768115942029, 'first_actionable_event_recall': 0.25, 'affected_foot_accuracy': 0.9411764705882353, 'normal_risk_run_fp': 0, 'too_early_firings': 40, 'invalid_firings': 53, 'median_warning_margin_ms': 36.0, 'pre_onset_detection_fraction': 0.8235294117647058, 'gate_pass': False}
- Next step: `TERRAIN_CANDIDATE_REDESIGN`

No validation gate, threshold grid, persistence grid, feature set, or episode
rule was changed after measurement. Diagnostic fallback is not a production
candidate and does not replace the controlled/static detector.
